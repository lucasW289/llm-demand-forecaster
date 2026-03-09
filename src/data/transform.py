import polars as pl

def transform_data(sales: pl.LazyFrame, calendar: pl.LazyFrame, prices: pl.LazyFrame):
    # 1. Optimize Calendar
    calendar = calendar.select([
        pl.col("d").str.strip_prefix("d_").cast(pl.Int16),
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"),
        pl.col("wm_yr_wk").cast(pl.Int32),
    ])

    # 2. Optimize Prices - CRITICAL FIX HERE
    prices = prices.with_columns([
        pl.col("sell_price").cast(pl.Float32),
        pl.col("wm_yr_wk").cast(pl.Int32),
        # Must match the sales table types for the join to work
        pl.col(["store_id", "item_id"]).cast(pl.Categorical) 
    ])

    # 3. Melt/Unpivot with Downcasting
    sales_long = (
        sales.unpivot(
            index=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
            variable_name="d",
            value_name="sales"
        )
        .with_columns([
            pl.col("d").str.strip_prefix("d_").cast(pl.Int16),
            pl.col("sales").cast(pl.Int16),
            # Cast these so they match the 'prices' table
            pl.col(["item_id", "store_id", "cat_id", "dept_id"]).cast(pl.Categorical)
        ])
    )

    # 4. Join
    df = sales_long.join(calendar, on="d", how="left")
    
    # Now this join will work because both sides are Categorical
    df = df.join(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    return df
    # 1. Optimize Calendar: Convert 'd' to Int16 and cast categories
    calendar = calendar.select([
        pl.col("d").str.strip_prefix("d_").cast(pl.Int16),
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"),
        pl.col("wm_yr_wk").cast(pl.Int32),
        pl.col("event_name_1").cast(pl.Categorical),
    ])

    # 2. Optimize Prices: Float32 is enough for Walmart prices
    prices = prices.with_columns([
        pl.col("sell_price").cast(pl.Float32),
        pl.col("wm_yr_wk").cast(pl.Int32),
    ])

    # 3. Melt/Unpivot with Downcasting
    # This is the "Heavy" step; we keep types small to stay in cache
    df = (
        sales.unpivot(
            index=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
            variable_name="d",
            value_name="sales"
        )
        .with_columns([
            pl.col("d").str.strip_prefix("d_").cast(pl.Int16),
            pl.col("sales").cast(pl.Int16),
            pl.col(["item_id", "store_id", "cat_id", "dept_id"]).cast(pl.Categorical)
        ])
    )

    # 4. Join on Integers (Fast)
    df = df.join(calendar, on="d", how="left")
    df = df.join(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    return df