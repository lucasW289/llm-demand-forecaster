import polars as pl
from pathlib import Path

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/parquet")

def load_data_lazy():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    files = {
        "sales": ("sales_train_validation.csv", "sales.parquet"),
        "calendar": ("calendar.csv", "calendar.parquet"),
        "prices": ("sell_prices.csv", "prices.parquet")
    }
    
    paths = {}
    for key, (csv_name, pq_name) in files.items():
        csv_path = DATA_DIR / csv_name
        pq_path = PROCESSED_DIR / pq_name
        
        # Auto-convert to Parquet if not exists
        if not pq_path.exists():
            print(f"Converting {csv_name} to Parquet for M1 speed...")
            pl.read_csv(csv_path).write_parquet(pq_path)
        
        paths[key] = pq_path

    return (
        pl.scan_parquet(paths["sales"]),
        pl.scan_parquet(paths["calendar"]),
        pl.scan_parquet(paths["prices"])
    )