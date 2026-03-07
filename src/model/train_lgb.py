import argparse
import logging
import joblib
import math
import polars as pl
import lightgbm as lgb
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_absolute_error

# Set up structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)

class M5FeatureEngineer:
    def __init__(self):
        self.item_means = None
        self.global_mean = None
        
        # 1. Load the LLM Archetypes into memory once
        archetype_path = Path("data/processed/llm_item_archetypes.parquet")
        if archetype_path.exists():
            self.archetypes = pl.read_parquet(archetype_path)
            logging.info(f"🧠 Loaded {len(self.archetypes)} LLM Archetypes!")
        else:
            raise FileNotFoundError(f"Could not find {archetype_path}")
        
        # Added llm_archetype to the feature list
        self.feature_cols = [
            "item_id", "store_id", "item_target_enc", "sell_price",
            "wday_sin", "wday_cos", "month_sin", "month_cos",
            "year", "lag_7", "lag_28", "rmean_7", "rmean_28", 
            "price_rel_mean", "llm_archetype" 
        ]

    def fit(self, lf: pl.LazyFrame):
        """Calculates lightweight Smoothed Target Encoding."""
        logging.info("🧠 Fitting Feature Engineer (Target Encoding)...")
        
        # Global mean
        self.global_mean = lf.select(pl.mean("sales")).collect().item()

        # Smoothed Target Encoding (Lighter, RAM-safe)
        m = 10
        self.item_means = (
            lf.group_by("item_id")
            .agg([
                pl.len().alias("n"),
                pl.mean("sales").alias("i_mean")
            ])
            .with_columns(
                item_target_enc=(
                    (pl.col("n") * pl.col("i_mean") + m * self.global_mean)
                    / (pl.col("n") + m)
                )
            )
            .select(["item_id", "item_target_enc"])
            .collect()
        )

    def transform(self, lf: pl.LazyFrame) -> pl.DataFrame:
        """Applies transformations natively in Rust/Polars."""
        
        # Lightweight Broadcast Join for Target Encoding
        lf = lf.join(self.item_means.lazy(), on="item_id", how="left")
        lf = lf.with_columns(pl.col("item_target_enc").fill_null(self.global_mean))

        # Inject the LLM Archetype (WITH DTYPE FIX)
        # Temporarily cast item_id to String to match the archetypes file
        lf = lf.with_columns(pl.col("item_id").cast(pl.String))
        
        lf = lf.join(self.archetypes.lazy(), on="item_id", how="left")
        
        # Cast item_id back to Categorical (vital for LightGBM) and cast our new feature
        lf = lf.with_columns([
            pl.col("item_id").cast(pl.Categorical),
            pl.col("llm_archetype").fill_null("Unknown").cast(pl.Categorical)
        ])

        # Native Polars Math for Cyclical Features
        lf = lf.with_columns([
            (pl.col("wday") * 2 * math.pi / 7).sin().alias("wday_sin"),
            (pl.col("wday") * 2 * math.pi / 7).cos().alias("wday_cos"),
            (pl.col("month") * 2 * math.pi / 12).sin().alias("month_sin"),
            (pl.col("month") * 2 * math.pi / 12).cos().alias("month_cos"),
        ])

        # Memory Optimization: Drop unused columns BEFORE collecting
        available_cols = lf.collect_schema().names()
        keep_cols = [c for c in self.feature_cols + ["sales", "date"] if c in available_cols]
        lf = lf.select(keep_cols)

        logging.info("⚙️ Executing Polars LazyGraph (Optimized RAM footprint)...")
        df = lf.collect(engine="streaming")

        return df

def train_pipeline(args):
    data_path = Path(args.data_path)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Lazy Scan
    logging.info(f"📥 Scanning Parquet: {data_path}")
    lf = pl.scan_parquet(data_path)

    # 2. Time-Series Split Strategy 
    dates = lf.select("date").unique().sort("date").collect()["date"]
    train_end_date = dates[-(args.test_days + args.val_days + 1)]
    val_end_date   = dates[-(args.test_days + 1)]

    logging.info(f"✂️ Split: Train <= {train_end_date} | Val <= {val_end_date}")

    # 3. Feature Engineering
    engineer = M5FeatureEngineer()
    train_lf = lf.filter(pl.col("date") <= train_end_date)
    engineer.fit(train_lf)

    logging.info("🔄 Transforming Train Split...")
    train_df = engineer.transform(train_lf)
    
    logging.info("🔄 Transforming Validation Split...")
    val_df = engineer.transform(
        lf.filter((pl.col("date") > train_end_date) & (pl.col("date") <= val_end_date))
    )

    # 4. Memory-Safe Conversion to Pandas
    actual_features = [c for c in engineer.feature_cols if c in train_df.columns]
    
    logging.info("🔄 Converting to Pandas (Preserving lightweight dtypes)...")
    X_train = train_df.select(actual_features).to_pandas()
    y_train = train_df["sales"].to_pandas()  

    X_val = val_df.select(actual_features).to_pandas()
    y_val = val_df["sales"].to_pandas()      

    # 5. Explicitly cast to pandas categorical for LightGBM compatibility
    cat_features = ["item_id", "store_id", "llm_archetype"]
    for col in cat_features:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")
            X_val[col] = X_val[col].astype("category")

    # 6. LightGBM Configuration (The M5 Kaggle Setup with Tweedie)
    model = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.05,
        num_leaves=63,           
        min_child_samples=50,    
        subsample=0.8,
        colsample_bytree=0.8,
        objective="tweedie",          # <--- THE MAGIC BULLET FOR ZERO-INFLATED DATA
        tweedie_variance_power=1.1,   # <--- TUNING PARAMETER FOR M5
        n_jobs=-1,               
        random_state=42
    )

    logging.info("⚡ Starting LightGBM Training on M1 CPU Cores (Tweedie Objective)...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        categorical_feature=cat_features,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )

    # 7. Evaluation & Persistence
    logging.info("📊 Generating Validation Predictions...")
    val_preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_preds)
    logging.info(f"🏆 Global Validation MAE: {mae:.4f}")

    # Check Feature Importance to prove LLM Lift
    importance = model.feature_importances_
    feat_imp = sorted(zip(actual_features, importance), key=lambda x: x[1], reverse=True)
    logging.info("🌟 Feature Importance (Splits):")
    for name, val in feat_imp:
        logging.info(f" - {name:20}: {val}")

    save_obj = {
        "model": model,
        "engineer": engineer,
        "metadata": {
            "trained_at": datetime.now().isoformat(),
            "mae": mae,
            "features": actual_features
        }
    }

    model_path = model_dir / f"lgbm_m5_{datetime.now().strftime('%Y%m%d_%H%M')}.joblib"
    joblib.dump(save_obj, model_path)
    logging.info(f"💾 Artifacts saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/processed/final.parquet", help="Path to processed dataset")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--val-days", type=int, default=28)
    parser.add_argument("--test-days", type=int, default=28)
    args = parser.parse_args()

    train_pipeline(args)