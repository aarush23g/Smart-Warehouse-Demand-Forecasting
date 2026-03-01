from pathlib import Path
import pandas as pd
from loguru import logger
import sys
from data.schemas.sales_schema import sales_schema

logger.remove()
logger.add(sys.stdout, format="{message}")

RAW_PATH = Path("data/raw")
PROCESSED_PATH = Path("data/processed")

SALES_FILE = RAW_PATH / "sales_train_validation.csv"
CALENDAR_FILE = RAW_PATH / "calendar.csv"
PRICES_FILE = RAW_PATH / "sell_prices.csv"


def load_sales():
    logger.info("Loading sales data...")
    df = pd.read_csv(SALES_FILE)

    logger.info(f"Sales shape: {df.shape}")

    # Validate schema (ID columns)
    df = sales_schema.validate(df)

    return df


def melt_sales(df: pd.DataFrame):
    logger.info("Transforming wide format → long format...")

    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    value_cols = [col for col in df.columns if col.startswith("d_")]

    df_long = df.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="d",
        value_name="sales",
    )

    logger.info(f"Long format shape: {df_long.shape}")

    return df_long


def load_calendar():
    logger.info("Loading calendar data...")
    cal = pd.read_csv(CALENDAR_FILE)

    cal["date"] = pd.to_datetime(cal["date"])

    return cal


def merge_calendar(df_long, cal):
    logger.info("Merging sales with calendar...")
    df_merged = df_long.merge(cal, on="d", how="left")

    return df_merged


def save_processed(df):
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    output_file = PROCESSED_PATH / "clean_sales.parquet"
    df.to_parquet(output_file, index=False)

    logger.success(f"Saved processed data to {output_file}")


def run_ingestion():
    sales = load_sales()
    sales_long = melt_sales(sales)

    calendar = load_calendar()
    merged = merge_calendar(sales_long, calendar)

    save_processed(merged)


if __name__ == "__main__":
    run_ingestion()