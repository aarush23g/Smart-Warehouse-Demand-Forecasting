from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from loguru import logger
import sys
import scipy.stats as stats

from src.features.feature_defs import FEATURE_COLUMNS, TARGET

logger.remove()
logger.add(sys.stdout, format="{message}")

PROCESSED_PATH = Path("data/processed")
ARTIFACT_PATH = Path("artifacts")

FEATURE_FILE = PROCESSED_PATH / "features.parquet"

P50_MODEL_FILE = ARTIFACT_PATH / "lgbm_p50.joblib"
P90_MODEL_FILE = ARTIFACT_PATH / "lgbm_p90.joblib"
ENCODER_FILE = ARTIFACT_PATH / "label_encoders_prob.joblib"


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data():
    logger.info("Loading evaluation dataset...")

    df = pd.read_parquet(FEATURE_FILE)

    # 🔽 Use last 90 days instead of full year
    cutoff_date = df["date"].max() - pd.Timedelta(days=90)
    df = df[df["date"] >= cutoff_date]

    logger.info(f"After date filter: {df.shape}")

    # 🔽 Keep only top SKUs by total demand (reduces sparsity)
    top_items = (
        df.groupby("item_id")[TARGET]
        .sum()
        .sort_values(ascending=False)
        .head(500)   # adjust if needed
        .index
    )

    df = df[df["item_id"].isin(top_items)]

    logger.info(f"After top-SKU filter: {df.shape}")

    return df


# -----------------------------
# ENCODE CATEGORICALS
# -----------------------------
def encode_categoricals(df: pd.DataFrame, encoders: dict):
    categorical_cols = [
        "item_id",
        "store_id",
        "event_name_1",
        "event_type_1",
    ]

    for col in categorical_cols:
        le = encoders[col]
        df[col] = df[col].astype(str).fillna("NA")
        df[col] = df[col].map(lambda x: x if x in le.classes_ else "NA")
        df[col] = le.transform(df[col])

    return df


# -----------------------------
# ML METRICS
# -----------------------------
def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5)))


def p90_error(y_true, y_pred):
    return np.percentile(np.abs(y_true - y_pred), 90)


# -----------------------------
# BUSINESS SIMULATION
# -----------------------------
def simulate_inventory(df, p50_preds, p90_preds):
    logger.info("Simulating inventory decisions...")

    df = df.copy()

    df["p50"] = p50_preds
    df["p90"] = p90_preds

    lead_time = 7
    service_level = 0.95
    Z = stats.norm.ppf(service_level)

    df["uncertainty"] = df["p90"] - df["p50"]
    df["safety_stock"] = Z * df["uncertainty"] * np.sqrt(lead_time)
    df["reorder_point"] = df["p50"] * lead_time + df["safety_stock"]

    # Actual demand during lead time
    df["actual_lead_demand"] = df[TARGET] * lead_time

    # Stockout condition
    df["stockout"] = df["reorder_point"] < df["actual_lead_demand"]

    # Excess inventory
    df["excess_inventory"] = df["reorder_point"] - df["actual_lead_demand"]
    df["excess_inventory"] = df["excess_inventory"].clip(lower=0)

    return df


# -----------------------------
# COST METRICS
# -----------------------------
def compute_business_metrics(df):
    stockout_rate = df["stockout"].mean()

    holding_cost_per_unit = 1.0
    stockout_cost_per_unit = 5.0

    total_holding_cost = (df["excess_inventory"] * holding_cost_per_unit).sum()
    total_stockout_cost = (
        df["stockout"].astype(int) * stockout_cost_per_unit
    ).sum()

    service_level = 1 - stockout_rate

    metrics = {
        "Stockout Rate": stockout_rate,
        "Service Level": service_level,
        "Holding Cost": total_holding_cost,
        "Stockout Cost": total_stockout_cost,
    }

    return metrics


# -----------------------------
# RUN EVALUATION
# -----------------------------
def run_evaluation():
    df = load_data()

    logger.info("Loading models...")
    p50_model = joblib.load(P50_MODEL_FILE)
    p90_model = joblib.load(P90_MODEL_FILE)
    encoders = joblib.load(ENCODER_FILE)

    df = encode_categoricals(df, encoders)

    # Feature alignment check
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    X = df[FEATURE_COLUMNS]

    logger.info("Generating predictions...")
    p50_preds = p50_model.predict(X)
    p90_preds = p90_model.predict(X)

    y_true = df[TARGET]

    logger.info("Computing ML metrics...")
    logger.info(f"WAPE: {wape(y_true, p50_preds):.4f}")
    logger.info(
        f"MAPE (not reliable for intermittent demand): {mape(y_true, p50_preds):.4f}"
    )
    logger.info(f"P90 Error: {p90_error(y_true, p50_preds):.4f}")

    df_sim = simulate_inventory(df, p50_preds, p90_preds)

    business_metrics = compute_business_metrics(df_sim)

    logger.info("Business metrics:")
    for k, v in business_metrics.items():
        logger.info(f"{k}: {v:.4f}")


if __name__ == "__main__":
    run_evaluation()