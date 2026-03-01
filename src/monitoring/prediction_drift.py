from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from loguru import logger
import sys

from src.features.feature_defs import FEATURE_COLUMNS, TARGET

logger.remove()
logger.add(sys.stdout, format="{message}")

FEATURE_FILE = Path("data/processed/features.parquet")
MODEL_FILE = Path("artifacts/lgbm_p50.joblib")
ENCODER_FILE = Path("artifacts/label_encoders_prob.joblib")


# -----------------------------
# WAPE
# -----------------------------
def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


# -----------------------------
# LOAD DATA
# -----------------------------
def load_recent_data():
    df = pd.read_parquet(FEATURE_FILE)

    latest_date = df["date"].max()
    start_date = latest_date - pd.Timedelta(days=28)

    df = df[(df["date"] >= start_date) & (df["date"] <= latest_date)]

    logger.info(f"After date filter: {df.shape}")

    # Keep top SKUs by demand (same as evaluation logic)
    top_items = (
        df.groupby("item_id")[TARGET]
        .sum()
        .nlargest(1000)
        .index
    )

    df = df[df["item_id"].isin(top_items)]

    logger.info(f"After top-SKU filter: {df.shape}")

    return df


# -----------------------------
# ENCODE CATEGORICALS
# -----------------------------
def encode_categoricals(df, encoders):
    categorical_cols = [
        "item_id",
        "store_id",
        "event_name_1",
        "event_type_1",
    ]

    for col in categorical_cols:
        le = encoders[col]

        df[col] = df[col].astype(str).fillna("NA")

        # Create mapping dict once
        mapping = {cls: idx for idx, cls in enumerate(le.classes_)}

        df[col] = df[col].map(mapping).fillna(0).astype(int)

    return df


# -----------------------------
# RUN PREDICTION DRIFT
# -----------------------------
def run_prediction_drift():
    df = load_recent_data()

    logger.info("Loading model and encoders...")
    model = joblib.load(MODEL_FILE)
    encoders = joblib.load(ENCODER_FILE)

    df = encode_categoricals(df, encoders)

    df = df.dropna(subset=[TARGET])

    X = df[FEATURE_COLUMNS]
    y_true = df[TARGET]

    logger.info("Generating predictions...")
    df["p50"] = model.predict(X)

    # Weekly aggregation
    df["week"] = df["date"].dt.to_period("W")

    weekly_wape = (
        df.groupby("week")
        .apply(lambda x: wape(x[TARGET], x["p50"]))
        .reset_index(name="wape")
    )

    logger.info("Weekly WAPE:")
    logger.info(weekly_wape)

    latest_wape = weekly_wape["wape"].iloc[-1]

    if latest_wape > 0.75:
        logger.warning("Prediction drift detected! Retraining recommended.")
    else:
        logger.success("Prediction performance stable.")

    return weekly_wape


if __name__ == "__main__":
    run_prediction_drift()