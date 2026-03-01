from pathlib import Path
import pandas as pd
import joblib
from loguru import logger
import sys

from src.features.feature_defs import FEATURE_COLUMNS
from src.inference.prediction_logic import compute_inventory_decision

logger.remove()
logger.add(sys.stdout, format="{message}")

PROCESSED_PATH = Path("data/processed")
OUTPUT_PATH = Path("data/outputs")
ARTIFACT_PATH = Path("artifacts")

FEATURE_FILE = PROCESSED_PATH / "features.parquet"

P50_MODEL_FILE = ARTIFACT_PATH / "lgbm_p50.joblib"
P90_MODEL_FILE = ARTIFACT_PATH / "lgbm_p90.joblib"
ENCODER_FILE = ARTIFACT_PATH / "label_encoders_prob.joblib"


# -----------------------------
# LOAD DATA FOR LATEST DATE
# -----------------------------
def load_latest_features():
    logger.info("Loading feature dataset for inference...")

    df = pd.read_parquet(FEATURE_FILE)

    latest_date = df["date"].max()
    df = df[df["date"] == latest_date]

    logger.info(f"Planning date: {latest_date}")
    logger.info(f"Rows for inference: {df.shape[0]}")

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

        # Safe mapping for unseen values
        df[col] = df[col].apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else 0
        )

    return df


# -----------------------------
# RUN BATCH INFERENCE
# -----------------------------
def run_batch_inference():
    df = load_latest_features()

    logger.info("Loading probabilistic models and encoders...")
    p50_model = joblib.load(P50_MODEL_FILE)
    p90_model = joblib.load(P90_MODEL_FILE)
    encoders = joblib.load(ENCODER_FILE)

    df = encode_categoricals(df, encoders)

    # Keep identifiers
    id_cols = ["item_id", "store_id"]

    current_inventory = df["lag_7"].copy()

    X = df[FEATURE_COLUMNS]

    logger.info("Generating probabilistic forecasts...")
    df["p50"] = p50_model.predict(X)
    df["p90"] = p90_model.predict(X)

    logger.info("Computing inventory plan...")

    decisions = df.apply(
        lambda row: compute_inventory_decision(
            p50=row["p50"],
            p90=row["p90"],
            current_inventory=row["lag_7"],
            lead_time=7,
            service_level=0.95,
        ),
        axis=1,
    )

    decision_df = pd.DataFrame(list(decisions))

    df = pd.concat([df[id_cols], df[["p50", "p90"]], decision_df], axis=1)

    OUTPUT_PATH.mkdir(exist_ok=True)

    output_file = OUTPUT_PATH / "inventory_plan.parquet"

    df.to_parquet(output_file, index=False)

    logger.success(f"Inventory plan saved to {output_file}")


if __name__ == "__main__":
    run_batch_inference()