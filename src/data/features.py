import polars as pl

def build_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    # Sort is required for window functions
    lf = lf.sort(["item_id", "store_id", "date"])

    # Rolling & Lag Features
    lf = lf.with_columns([
        # Target Lags
        pl.col("sales").shift(7).over(["item_id", "store_id"]).alias("lag_7"),
        pl.col("sales").shift(28).over(["item_id", "store_id"]).alias("lag_28"),

        # Rolling Means (Shifted 1 to prevent leakage)
        pl.col("sales").shift(1)
            .rolling_mean(window_size=7)
            .over(["item_id", "store_id"])
            .alias("rmean_7"),
        
        pl.col("sales").shift(1)
            .rolling_mean(window_size=28)
            .over(["item_id", "store_id"])
            .alias("rmean_28"),
            
        # Price Momentum (No shift needed for prices in M5)
        (pl.col("sell_price") / pl.col("sell_price").mean().over(["item_id", "store_id"]))
            .alias("price_rel_mean")
    ])

    # Calendar Features
    lf = lf.with_columns([
        pl.col("date").dt.weekday().cast(pl.Int8).alias("wday"),
        pl.col("date").dt.month().cast(pl.Int8).alias("month"),
        pl.col("date").dt.year().cast(pl.Int16).alias("year"),
    ])

    # Drop nulls created by lags and streaming-collect
    return lf.drop_nulls()