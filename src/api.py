import polars as pl
import joblib
import re
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from ollama import Client

# --- 🚨 THE PICKLE RESCUE 🚨 ---
# 1. Import the class definition from your training script
from src.model.train_lgb import M5FeatureEngineer

# 2. Trick joblib into finding the class exactly where it expects it
import __main__
__main__.M5FeatureEngineer = M5FeatureEngineer
# -------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(title="Hybrid Agentic Forecaster API")

# Path to the Global Model Artifact and Data
MODEL_PATH = Path("models/lgbm_global_20260304.joblib")
DATA_PATH = Path("data/processed/final.parquet")

# 1. Load the bundled artifacts on startup
logging.info(f"Loading Global Artifacts from {MODEL_PATH}...")
try:
    artifacts = joblib.load(MODEL_PATH)
    lgb_model = artifacts["model"]
    engineer = artifacts["engineer"]
    logging.info(f"Artifacts loaded. Model trained at: {artifacts['metadata']['trained_at']}")
except Exception as e:
    logging.error(f"Failed to load model artifacts: {e}")
    raise e

class PredictRequest(BaseModel):
    item_id: str
    store_id: str
    target_date: str
    context_days: int = 7

@app.post("/predict")
def predict(req: PredictRequest):
    # 2. Convert string input to Polars Date for type-safe filtering
    try:
        # We parse the string to a date object using Polars logic
        target_dt = pl.lit(req.target_date).str.to_date()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    # 3. Lazily scan the data
    lf = pl.scan_parquet(DATA_PATH)
    
    # Filter for the specific item/store history
    item_lf = lf.filter(
        (pl.col("item_id") == req.item_id) & 
        (pl.col("store_id") == req.store_id) &
        (pl.col("date") <= target_dt)
    ).sort("date")
    
    # We need the target day AND the context days before it for the prompt
    try:
        df = item_lf.tail(req.context_days + 1).collect()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data read error: {e}")
    
    # Check if we actually found the target date
    if len(df) == 0 or df.row(-1, named=True)["date"] != df.select(target_dt).item():
        raise HTTPException(status_code=404, detail="Target date not found for this item/store.")
        
    history_df = df.head(req.context_days)
    target_day_df = df.tail(1)
    
    # 4. Apply exactly identical Feature Engineering to the target day
    # We pass it as a LazyFrame to match the M5FeatureEngineer's signature
    processed_target = engineer.transform(target_day_df.lazy())
    
    # 5. LightGBM Inference (Mathematical Baseline)
    X_infer = processed_target.select(engineer.feature_cols).to_numpy()
    lgb_pred = lgb_model.predict(X_infer)[0]
    
    # 6. Build Feature-Augmented Prompt
    target_row = processed_target.row(0, named=True)
    price = target_row.get("sell_price")
    lag_7 = target_row.get("lag_7")
    lag_28 = target_row.get("lag_28")
    rm_7 = target_row.get("rolling_mean_7")
    rm_28 = target_row.get("rolling_mean_28")
    item_enc = target_row.get("item_target_enc")
    
    raw_target_row = target_day_df.row(0, named=True)
    target_event = raw_target_row.get("event_name_1")
    target_event_str = f" Tomorrow is a special event: {target_event}." if target_event and target_event != "None" else " There are no special events tomorrow."
    
    history_text = "Recent Sales History:\n"
    for row in history_df.iter_rows(named=True):
        date = row.get("date")
        sales = row.get("sales")
        event = row.get("event_name_1")
        event_str = f" | Event: {event}" if event and event != "None" else ""
        history_text += f"- {date}: {sales} units{event_str}\n"

    prompt = (
        f"{history_text}\n"
        f"Task: Predict the final sales volume for {req.target_date}.\n"
        f"Context:{target_event_str}\n"
        f"Detailed Day Features:\n"
        f"- Current Price: ${price:.2f}\n"
        f"- Target Encoded Item Mean: {item_enc:.2f} units\n"
        f"- 7-Day Lag: {lag_7} units\n"
        f"- 28-Day Lag: {lag_28} units\n"
        f"- 7-Day Rolling Mean: {rm_7:.2f} units\n"
        f"- 28-Day Rolling Mean: {rm_28:.2f} units\n\n"
        f"Statistical Baseline: Our LightGBM mathematical model evaluated these exact features and predicts {lgb_pred:.2f} units.\n\n"
        "Instruction: You are an expert retail demand planner. Consider the recent sales history, the specific features above, and the statistical baseline. "
        "Provide ONLY a single integer as your final predicted sales volume. Do not output any text, explanation, or punctuation."
    )
    
    # 7. LLM Agentic Adjustment (Using Llama 3.2 1B)
    try:
        client = Client(host='http://127.0.0.1:11434')
        response = client.chat(model='llama3.2:1b', messages=[{'role': 'user', 'content': prompt}])
        raw_output = response['message']['content'].strip()
        # Extract first continuous number found in output
        match = re.search(r'\d+', raw_output)
        llm_prediction = int(match.group()) if match else round(lgb_pred)
    except Exception as e:
        logging.error(f"LLM Reasoning failed: {e}")
        llm_prediction = round(lgb_pred)
        
    actual_sales = raw_target_row.get("sales")
    
    return {
        "item_id": req.item_id,
        "store_id": req.store_id,
        "target_date": req.target_date,
        "actual_sales": actual_sales,
        "lightgbm_baseline": round(lgb_pred, 2),
        "llm_final_prediction": llm_prediction,
        "adjustment_made": round(abs(llm_prediction - lgb_pred), 2),
        "error": abs(llm_prediction - actual_sales) if actual_sales is not None else None
    }