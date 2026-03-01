from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
import sys

# Clean logger output
logger.remove()
logger.add(sys.stdout, format="{message}")

PROCESSED_PATH = Path("data/processed")
FEATURE_FILE = PROCESSED_PATH / "features.parquet"


def load_data():
    logger.info("Loading feature dataset...")
    df = pd.read_parquet(FEATURE_FILE)

    logger.info(f"Data shape: {df.shape}")

    return df


def time_based_split(df: pd.DataFrame):
    logger.info("Performing time-based split...")

    df = df.sort_values("date")

    split_date = df["date"].quantile(0.8)

    train = df[df["date"] <= split_date]
    val = df[df["date"] > split_date]

    logger.info(f"Train shape: {train.shape}")
    logger.info(f"Validation shape: {val.shape}")

    return train, val


def naive_forecast(val: pd.DataFrame):
    logger.info("Generating naive forecast using lag_7...")

    val = val.copy()
    val["prediction"] = val["lag_7"]

    return val


def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


def bias(y_true, y_pred):
    return np.sum(y_pred - y_true) / np.sum(y_true)


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def evaluate(val: pd.DataFrame):
    logger.info("Evaluating baseline model...")

    y_true = val["sales"]
    y_pred = val["prediction"]

    metrics = {
        "WAPE": wape(y_true, y_pred),
        "Bias": bias(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
    }

    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")

    return metrics


def run_baseline():
    df = load_data()

    train, val = time_based_split(df)

    val = naive_forecast(val)

    evaluate(val)


if __name__ == "__main__":
    run_baseline()