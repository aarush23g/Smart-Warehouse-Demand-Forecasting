from pathlib import Path
import pandas as pd
from loguru import logger
import sys
from src.features.feature_defs import TARGET, TIME_COLUMN

logger.remove()
logger.add(sys.stdout, format="{message}")

RAW_PATH = Path("data/raw")
PROCESSED_PATH = Path("data/processed")

INPUT_FILE = PROCESSED_PATH / "clean_sales.parquet"
PRICE_FILE = RAW_PATH / "sell_prices.csv"

OUTPUT_FILE = PROCESSED_PATH / "features.parquet"


def load_data():
    logger.info("Loading processed sales data...")
    df = pd.read_parquet(INPUT_FILE)

    logger.info(f"Original shape: {df.shape}")

    # 🔽 Reduce memory
    df["id"] = df["id"].astype("category")
    df[TARGET] = df[TARGET].astype("float32")

    return df


def keep_required_columns(df: pd.DataFrame):
    logger.info("Keeping only required columns for feature generation...")

    required_cols = [
        "id",
        "item_id",
        "store_id",
        "date",
        "wm_yr_wk",
        TARGET,
        "event_name_1",
        "event_type_1",
    ]

    df = df[required_cols]

    return df


def create_time_features(df: pd.DataFrame):
    logger.info("Creating time features...")

    df["year"] = df[TIME_COLUMN].dt.year.astype("int16")
    df["month"] = df[TIME_COLUMN].dt.month.astype("int8")
    df["week"] = df[TIME_COLUMN].dt.isocalendar().week.astype("int8")

    return df


def create_lag_features(df: pd.DataFrame):
    logger.info("Creating lag features (memory-safe)...")

    df = df.sort_values(["id", TIME_COLUMN])

    df["lag_7"] = df.groupby("id")[TARGET].shift(7)
    df["lag_28"] = df.groupby("id")[TARGET].shift(28)

    return df


def create_rolling_features(df: pd.DataFrame):
    logger.info("Creating rolling features (memory-safe)...")

    df["rolling_mean_7"] = (
        df.groupby("id")["lag_7"]
        .transform(lambda x: x.rolling(7).mean())
        .astype("float32")
    )

    df["rolling_std_28"] = (
        df.groupby("id")["lag_28"]
        .transform(lambda x: x.rolling(28).std())
        .astype("float32")
    )

    return df


def load_prices():
    logger.info("Loading price data...")

    prices = pd.read_csv(PRICE_FILE)

    prices["sell_price"] = prices["sell_price"].astype("float32")

    return prices


def merge_prices(df: pd.DataFrame, prices: pd.DataFrame):
    logger.info("Merging price data...")

    df = df.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )

    return df


def drop_na_rows(df: pd.DataFrame):
    logger.info("Dropping rows with NA lag features...")

    df = df.dropna(subset=["lag_7", "lag_28"])

    return df


def save_features(df: pd.DataFrame):
    logger.info("Saving feature dataset...")

    df.to_parquet(OUTPUT_FILE, index=False)

    logger.success(f"Saved features to {OUTPUT_FILE}")


def run_feature_pipeline():
    df = load_data()

    df = keep_required_columns(df)

    df = create_time_features(df)

    df = create_lag_features(df)

    df = create_rolling_features(df)

    prices = load_prices()
    df = merge_prices(df, prices)

    df = drop_na_rows(df)

    save_features(df)


if __name__ == "__main__":
    run_feature_pipeline()