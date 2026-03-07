import logging
from pathlib import Path
import polars as pl

# Assuming your files are structured this way
from src.data.load import load_data_lazy
from src.data.transform import transform_data
from src.data.features import build_features

# Setup logging for M1 monitoring
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

OUTPUT_PATH = Path("data/processed/final.parquet")

def main():
    logging.info("🚀 Starting M5 High-Performance Pipeline (M1 Optimized)")

    # 1. Global Performance Settings
    # String cache is REQUIRED when joining categorical columns
    pl.enable_string_cache() 
    
    # 2. Load (Lazy)
    # This step is instant because it only defines the plan
    logging.info("📥 Scanning Parquet files...")
    sales, calendar, prices = load_data_lazy()

    # 3. Transform & Feature Engineering
    # We chain these to keep the computation graph clean
    logging.info("🔄 Building computation graph (Transform + Features)...")
    df_lazy = transform_data(sales, calendar, prices)
    df_lazy = build_features(df_lazy)

    # 4. Execute with Streaming (The 'Turbo' Step)
    # On M1, streaming=True is the difference between 2 minutes and a crash.
    logging.info("⚡ Executing pipeline with Streaming Engine...")
    try:
        # collect(streaming=True) allows Polars to process data in chunks 
        # that fit into your M1's L2 cache/RAM.
        df = df_lazy.collect(engine="streaming")        
        logging.info(f"📊 Dataset prepared: {df.estimated_size('mb'):.2f} MB")
        logging.info(f"📈 Total rows: {len(df):,}")

    except Exception as e:
        logging.error(f"❌ Execution failed: {e}")
        return

    # 5. Save
    logging.info(f"💾 Saving to {OUTPUT_PATH}...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Writing to Parquet is much faster than CSV on Mac SSDs
    df.write_parquet(OUTPUT_PATH, compression="zstd")

    logging.info("✅ Pipeline Complete!")

if __name__ == "__main__":
    main()