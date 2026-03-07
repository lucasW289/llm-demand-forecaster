import polars as pl
import joblib
import re
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from ollama import Client

# --- 🚨 THE PICKLE RESCUE 🚨 ---
from src.model.train_lgb import M5FeatureEngineer
import __main__
__main__.M5FeatureEngineer = M5FeatureEngineer
# -------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

MODEL_PATH = Path("models/lgbm_global_20260304.joblib") # Ensure date matches your file
DATA_PATH = Path("data/processed/final.parquet")

def build_hybrid_prompt(history_df: pl.DataFrame, target_row: dict, lgb_prediction: float) -> str:
    """Standardized prompt building for the LLM."""
    history_text = "Recent Sales History:\n"
    for row in history_df.iter_rows(named=True):
        date = row.get("date")
        sales = row.get("sales")
        event = row.get("event_name_1")
        event_str = f" | Event: {event}" if event and event != "None" else ""
        history_text += f"- {date}: {sales} units{event_str}\n"

    target_date = target_row.get("date")
    target_event = target_row.get("event_name_1")
    target_event_str = (
        f" Tomorrow is a special event: {target_event}." 
        if target_event and target_event != "None" 
        else " There are no special events tomorrow."
    )
    
    prompt = (
        f"{history_text}\n"
        f"Task: Predict the final sales volume for {target_date}.\n"
        f"Context:{target_event_str}\n"
        f"Detailed Day Features:\n"
        f"- Current Price: ${target_row.get('sell_price'):.2f}\n"
        f"- Target Encoded Item Mean: {target_row.get('item_target_enc'):.2f} units\n"
        f"- 7-Day Lag: {target_row.get('lag_7')} units\n"
        f"- 28-Day Lag: {target_row.get('lag_28')} units\n\n"
        f"Statistical Baseline: LightGBM predicts {lgb_prediction:.2f} units.\n\n"
        "Instruction: Provide ONLY a single integer as your final predicted sales volume. No text."
    )
    return prompt

def run_evaluation(item_id="FOODS_3_021", store_id="CA_1"):
    logging.info(f"Loading Global Artifacts...")
    artifacts = joblib.load(MODEL_PATH)
    lgb_model = artifacts["model"]
    engineer = artifacts["engineer"]
    
    # 1. Load Data
    lf = pl.scan_parquet(DATA_PATH)
    item_lf = lf.filter((pl.col("item_id") == item_id) & (pl.col("store_id") == store_id)).sort("date")
    full_df = item_lf.collect()
    
    # 2. Define the Test Window (The final 28 days)
    test_days = 28
    context_days = 7
    start_idx = len(full_df) - test_days
    
    actuals, lgb_preds, hybrid_preds = [], [], []
    client = Client(host='http://127.0.0.1:11434')

    logging.info(f"Evaluating {test_days} days for {item_id}...")

    for i in tqdm(range(start_idx, len(full_df)), desc="Backtesting"):
        # Slice the history and target day
        history_df = full_df.slice(i - context_days, context_days)
        target_day_raw = full_df.slice(i, 1)
        
        # Transform for LightGBM
        processed_target = engineer.transform(target_day_raw.lazy())
        X_test = processed_target.select(engineer.feature_cols).to_numpy()
        
        # --- Normal Model Prediction ---
        lgb_val = lgb_model.predict(X_test)[0]
        
        # --- Hybrid Model Prediction ---
        target_dict = {**target_day_raw.row(0, named=True), **processed_target.row(0, named=True)}
        prompt = build_hybrid_prompt(history_df, target_dict, lgb_val)
        
        try:
            response = client.chat(model='llama3.2:1b', messages=[{'role': 'user', 'content': prompt}])
            match = re.search(r'\d+', response['message']['content'])
            hybrid_val = int(match.group()) if match else round(lgb_val)
        except:
            hybrid_val = round(lgb_val)

        actuals.append(target_day_raw["sales"].item())
        lgb_preds.append(lgb_val)
        hybrid_preds.append(hybrid_val)

    # 3. Final Comparison
    lgb_mae = mean_absolute_error(actuals, lgb_preds)
    hybrid_mae = mean_absolute_error(actuals, hybrid_preds)
    
    print("\n" + "="*40)
    print("📊 MODEL COMPARISON RESULTS (MAE)")
    print("="*40)
    print(f"Normal Model (LightGBM) : {lgb_mae:.4f}")
    print(f"Hybrid Model (LGBM+LLM) : {hybrid_mae:.4f}")
    print("="*40)
    
    improvement = (lgb_mae - hybrid_mae) / lgb_mae * 100
    if hybrid_mae < lgb_mae:
        print(f"🏆 SUCCESS: Hybrid Model is {improvement:.2f}% better!")
    else:
        print(f"📉 Hybrid Model added {abs(improvement):.2f}% error.")

if __name__ == "__main__":
    run_evaluation()