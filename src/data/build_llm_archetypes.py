import argparse
import logging
import subprocess
import json
import polars as pl
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)

# Paths
SALES_PATH = Path("data/parquet/sales.parquet")
PRICES_PATH = Path("data/parquet/prices.parquet")
OUTPUT_PATH = Path("data/processed/llm_item_archetypes.parquet")

def create_item_dna() -> pl.DataFrame:
    """Calculates the statistical behavior of every item using Polars."""
    logging.info("🧬 Calculating Item DNA from raw Parquet files...")
    
    sales = pl.scan_parquet(SALES_PATH)
    prices = pl.scan_parquet(PRICES_PATH)
    
    # 1. Unpivot the wide sales data into a single 'sales' column
    sales_long = sales.unpivot(
        index=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        variable_name="d",
        value_name="sales"
    )
    
    # 2. Sales Stats
    item_stats = sales_long.group_by("item_id").agg([
        pl.col("sales").mean().cast(pl.Float32).alias("avg_daily_sales"),
        pl.col("sales").std().cast(pl.Float32).alias("sales_volatility"),
        (pl.col("sales") == 0).mean().cast(pl.Float32).alias("zero_sales_pct")
    ])
    
    # 3. Price Stats
    price_stats = prices.group_by("item_id").agg([
        pl.col("sell_price").mean().cast(pl.Float32).alias("avg_price")
    ])
    
    # Join and collect into memory (30k rows fits easily in RAM)
    profiles = item_stats.join(price_stats, on="item_id", how="inner").collect()
    
    # Fill nulls in case any items have no variance
    return profiles.fill_null(0.0)

def query_local_llm_batch(batch_records) -> dict:
    """Sends a batch to the LLM and strictly enforces a flat Key-Value JSON dictionary."""
    
    items_text = ""
    for row in batch_records:
        items_text += f"""
        Item ID: {row['item_id']}
        - Avg Daily Sales: {row['avg_daily_sales']:.2f}
        - Sales Volatility: {row['sales_volatility']:.2f}
        - Zero Sales %: {row['zero_sales_pct'] * 100:.1f}%
        - Avg Price: ${row['avg_price']:.2f}
        ---\n"""

    prompt = f"""
    You are an expert retail analyst. Categorize the following products into ONE of these Demand Archetypes:
    "High-Volume Staple", "Sporadic High-Ticket", "Highly Volatile", "Slow-Moving Inventory".
    
    Here are the items:
    {items_text}
    
    Output ONLY a single, valid JSON object where the keys are the exact Item IDs and the values are the Archetypes.
    Do not include any other text.
    Example:
    {{
      "FOODS_3_090": "Highly Volatile",
      "HOBBIES_1_004": "Slow-Moving Inventory"
    }}
    """
    
    try:
        # Force strict JSON formatting from Llama 3.2
        result = subprocess.run(
            ["ollama", "run", "llama3.2", "--format", "json", prompt], 
            capture_output=True, text=True, check=True
        )
        
        # Parse the flat dictionary
        data = json.loads(result.stdout.strip())
        return data if isinstance(data, dict) else {}
        
    except Exception as e:
        logging.error(f"Batch failed to parse: {e}")
        return {}

def main(args):
    # 1. Generate the statistical profiles
    dna_df = create_item_dna()
    
    # ==========================================
    # 🛑 Checkpointing & Resume Logic
    # ==========================================
    existing_results = []
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if OUTPUT_PATH.exists():
        # Load the existing progress
        existing_df = pl.read_parquet(OUTPUT_PATH)
        existing_results = existing_df.to_dicts()
        processed_ids = {row["item_id"] for row in existing_results}
        
        logging.info(f"🔄 Found existing checkpoint! {len(processed_ids)} items already processed.")
        
        # Filter the dna_df to only include items we HAVEN'T done yet
        dna_df = dna_df.filter(~pl.col("item_id").is_in(list(processed_ids)))
        logging.info(f"⏭️ Skipping finished items. {len(dna_df)} items remaining.")
        
    if len(dna_df) == 0:
        logging.info("🎉 All items have already been processed! You are good to go.")
        
        # Print final dist if we just instantly finished
        if len(existing_results) > 0:
            final_df = pl.DataFrame(existing_results)
            print("\n📊 Final Archetype Distribution:")
            print(final_df["llm_archetype"].value_counts())
        return
    # ==========================================

    if args.test_run:
        logging.info("🧪 TEST RUN: Only processing 60 items...")
        dna_df = dna_df.head(60)
    
    records = dna_df.to_dicts()
    
    # Start our results list with the data we already processed
    results = existing_results 
    
    # BATCH SIZE: 10 items per prompt is the sweet spot for 3B models
    batch_size = 10
    batches = [records[i:i + batch_size] for i in range(0, len(records), batch_size)]
    
    logging.info(f"🤖 Booting up Llama 3.2 to process {len(batches)} batches...")
    
    # Strict category safeguard
    valid_categories = [
        "High-Volume Staple", 
        "Sporadic High-Ticket", 
        "Highly Volatile", 
        "Slow-Moving Inventory"
    ]
    
    # 2. Process batches
    for i, batch in enumerate(tqdm(batches, desc="Processing Batches")):
        batch_results_dict = query_local_llm_batch(batch)
        
        for item in batch:
            item_id = item["item_id"]
            # Look up the ID in the LLM's response
            archetype = batch_results_dict.get(item_id, "Unknown")
            
            # Catch LLM hallucinating category names
            if archetype not in valid_categories:
                archetype = "Unknown"
                
            results.append({
                "item_id": item_id,
                "llm_archetype": archetype
            })
            
        # 💾 SAVE CHECKPOINT EVERY 5 BATCHES
        if (i + 1) % 5 == 0 or (i + 1) == len(batches):
            pl.DataFrame(results).write_parquet(OUTPUT_PATH)
    
    logging.info(f"✅ LLM Archetypes fully processed and saved to {OUTPUT_PATH}")
    
    final_df = pl.DataFrame(results)
    print("\n📊 Final Archetype Distribution:")
    print(final_df["llm_archetype"].value_counts())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", action="store_true", help="Run on only 60 items to test the LLM connection.")
    args = parser.parse_args()
    main(args)