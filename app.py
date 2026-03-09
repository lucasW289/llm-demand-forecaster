import streamlit as st
import polars as pl
import pandas as pd
import joblib
import glob
import math
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ==========================================
# 1. THE "GHOST" CLASS (For Joblib)
# ==========================================
# We must define the class here so Joblib knows how to reconstruct 
# the engineer object it saved during training.
class M5FeatureEngineer:
    def __init__(self):
        self.item_means = None
        self.global_mean = None
        self.archetypes = None
        self.feature_cols = [
            "item_id", "store_id", "item_target_enc", "sell_price",
            "wday_sin", "wday_cos", "month_sin", "month_cos",
            "year", "lag_7", "lag_28", "rmean_7", "rmean_28", 
            "price_rel_mean", "llm_archetype" 
        ]

    def fit(self, lf: pl.LazyFrame): pass

    def transform(self, lf: pl.LazyFrame) -> pl.DataFrame:
        # 1. Target Encoding
        lf = lf.join(self.item_means.lazy(), on="item_id", how="left")
        lf = lf.with_columns(pl.col("item_target_enc").fill_null(self.global_mean))

        # 2. LLM Archetypes
        lf = lf.with_columns(pl.col("item_id").cast(pl.String))
        lf = lf.join(self.archetypes.lazy(), on="item_id", how="left")
        lf = lf.with_columns([
            pl.col("item_id").cast(pl.Categorical),
            pl.col("llm_archetype").fill_null("Unknown").cast(pl.Categorical)
        ])

        # 3. Cyclical Math
        lf = lf.with_columns([
            (pl.col("wday") * 2 * math.pi / 7).sin().alias("wday_sin"),
            (pl.col("wday") * 2 * math.pi / 7).cos().alias("wday_cos"),
            (pl.col("month") * 2 * math.pi / 12).sin().alias("month_sin"),
            (pl.col("month") * 2 * math.pi / 12).cos().alias("month_cos"),
        ])

        available_cols = lf.collect_schema().names()
        keep_cols = [c for c in self.feature_cols + ["sales", "date"] if c in available_cols]
        return lf.select(keep_cols).collect(engine="streaming")

# ==========================================
# 2. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="LLM Demand Forecast",
    page_icon="📈",
    layout="wide"
)

# ==========================================
# 3. CACHED DATA & MODEL LOADING
# ==========================================
@st.cache_resource
def load_model():
    model_files = glob.glob("models/lgbm_m5_*.joblib")
    if not model_files:
        st.error("No model found!")
        st.stop()
    latest_model_path = max(model_files, key=lambda x: Path(x).stat().st_mtime)
    return joblib.load(latest_model_path)

@st.cache_data
def load_archetypes():
    return pl.read_parquet("data/processed/llm_item_archetypes.parquet")

@st.cache_data
def get_unique_entities():
    lf = pl.scan_parquet("data/processed/final.parquet")
    items = lf.select("item_id").unique().collect().to_series().to_list()
    stores = lf.select("store_id").unique().collect().to_series().to_list()
    return sorted(items), sorted(stores)

# ==========================================
# 4. UI LAYOUT & SIDEBAR
# ==========================================
st.title("🔮 Retail Demand Forecaster")
st.markdown("**Machine Learning Engineer Portfolio Project** | *LightGBM + LLM Semantic Features*")

st.sidebar.header("Filter Forecast")
items, stores = get_unique_entities()

default_item = "FOODS_3_090" if "FOODS_3_090" in items else items[0]
selected_item = st.sidebar.selectbox("Select Item", items, index=items.index(default_item))
selected_store = st.sidebar.selectbox("Select Store", stores)

try:
    model_dict = load_model()
    model = model_dict["model"]
    engineer = model_dict["engineer"]
    features = model_dict["metadata"]["features"]
    archetypes_df = load_archetypes()
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()

# ==========================================
# 5. ON-THE-FLY INFERENCE ENGINE
# ==========================================
with st.spinner("Scanning Parquet and generating forecast..."):
    lf = pl.scan_parquet("data/processed/final.parquet")
    lf = lf.filter((pl.col("item_id") == selected_item) & (pl.col("store_id") == selected_store))
    
    # Run the raw data through your custom Feature Engineer!
    df = engineer.transform(lf)

if df.height == 0:
    st.warning("No data found for this Item/Store combination.")
    st.stop()

# Prepare for LightGBM
X = df.select(features).to_pandas()
cat_features = ["item_id", "store_id", "llm_archetype"]
for col in cat_features:
    if col in X.columns:
        X[col] = X[col].astype("category")

# Predict!
df = df.with_columns(pl.Series("predicted_sales", model.predict(X)))

# Split into History and Validation
val_days = 28
dates = df.sort("date")["date"].to_list()
split_date = dates[-val_days]

history_df = df.filter(pl.col("date") < split_date)
val_df = df.filter(pl.col("date") >= split_date)

# ==========================================
# 6. DASHBOARD METRICS
# ==========================================
item_archetype = archetypes_df.filter(pl.col("item_id") == selected_item)["llm_archetype"]
archetype_label = item_archetype[0] if len(item_archetype) > 0 else "Unknown"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Selected Item", selected_item)
col2.metric("LLM Behavioral Archetype", archetype_label)
col3.metric("Historical Avg (Units/Day)", round(history_df["sales"].mean(), 2))
col4.metric("Holdout MAE", round(model_dict["metadata"]["mae"], 4))

# ==========================================
# 7. VISUALIZATION
# ==========================================
st.subheader(f"Demand Forecast: Last 60 Days vs 28-Day Holdout")

plot_df = df.tail(90).to_pandas()
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=plot_df["date"], y=plot_df["sales"], 
    mode='lines', name='Actual Sales',
    line=dict(color='gray', width=2)
))

val_plot_df = val_df.to_pandas()
fig.add_trace(go.Scatter(
    x=val_plot_df["date"], y=val_plot_df["predicted_sales"], 
    mode='lines', name='LightGBM Forecast',
    line=dict(color='blue', width=2, dash='dash')
))

fig.add_vline(x=split_date, line_width=2, line_dash="dash", line_color="red")
fig.add_annotation(x=split_date, y=plot_df["sales"].max(), text="Forecast Horizon Start", showarrow=False, xshift=70)

fig.update_layout(
    xaxis_title="Date", yaxis_title="Units Sold",
    hovermode="x unified", template="plotly_white", height=500
)

st.plotly_chart(fig, use_container_width=True)
# ==========================================
# 8. ARCHITECTURE & STRATEGY
# ==========================================
with st.expander("Show Model Architecture & LLM Strategy"):
    st.markdown("""
    ### 🧠 The LLM "Cold Start" Strategy
    Traditional time-series models rely heavily on historical lags and exact `item_id` target encodings. While highly accurate for mature inventory, this creates a major **Cold Start Problem**: *how do we forecast a brand new product with zero sales history?*
    
    By utilizing an LLM to analyze item metadata and assign behavioral archetypes (e.g., *Highly Volatile*, *High-Volume Staple*), we create a semantic baseline. This allows the model to infer variance patterns for new inventory from Day 1, before statistical lags can be calculated.
    
    ---
    
    ### ⚙️ Backend Pipeline Specs
    * **Data Engine:** `Polars` (LazyFrames utilized for out-of-core memory management to process 44M+ rows on local hardware).
    * **Algorithm:** `LightGBM Regressor` (M1 multi-core optimized).
    * **Objective Function:** `Tweedie` (Specifically chosen to handle zero-inflated, highly intermittent retail demand spikes).
    * **Feature Engineering:** Lightweight Smoothed Target Encoding & Cyclic Temporal Math.
    """)