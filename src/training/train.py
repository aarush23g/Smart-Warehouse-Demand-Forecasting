from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from loguru import logger
import sys
import gc

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from src.features.feature_defs import FEATURE_COLUMNS, TARGET

# -----------------------------
# LOGGER
# -----------------------------
logger.remove()
logger.add(sys.stdout, format="{message}")

# -----------------------------
# PATHS
# -----------------------------
PROCESSED_PATH = Path("data/processed")
ARTIFACT_PATH = Path("artifacts")

FEATURE_FILE = PROCESSED_PATH / "features.parquet"
MODEL_FILE = ARTIFACT_PATH / "xgb_model.joblib"
ENCODER_FILE = ARTIFACT_PATH / "label_encoders.joblib"

# -----------------------------
# LOAD DATA (memory-safe)
# -----------------------------
def load_data():
    logger.info("Loading feature dataset (column-pruned)...")

    columns_needed = FEATURE_COLUMNS + [TARGET, "date"]

    df = pd.read_parquet(FEATURE_FILE, columns=columns_needed)

    logger.info(f"Loaded shape: {df.shape}")

    # Rolling 90-day window for Docker-safe training
    max_date = df["date"].max()
    cutoff_date = max_date - pd.Timedelta(days=90)
    df = df[df["date"] >= cutoff_date]

    logger.info(f"Filtered shape (last 90 days): {df.shape}")

    # Downcast numeric types early
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")

    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("int32")

    gc.collect()

    return df


# -----------------------------
# ENCODE CATEGORICAL FEATURES
# -----------------------------
def encode_categoricals(df: pd.DataFrame):
    logger.info("Encoding categorical features...")

    encoders = {}

    categorical_cols = [
        "item_id",
        "store_id",
        "event_name_1",
        "event_type_1",
    ]

    for col in categorical_cols:
        le = LabelEncoder()

        # Ensure NA consistency
        df[col] = df[col].astype(str).fillna("NA")

        df[col] = le.fit_transform(df[col])

        encoders[col] = le

    gc.collect()

    return df, encoders


# -----------------------------
# TIME-BASED SPLIT
# -----------------------------
def time_based_split(df: pd.DataFrame):
    logger.info("Performing time-based split...")

    df = df.sort_values("date")

    split_date = df["date"].quantile(0.8)

    train = df[df["date"] <= split_date].copy()
    val = df[df["date"] > split_date].copy()

    logger.info(f"Train shape: {train.shape}")
    logger.info(f"Validation shape: {val.shape}")

    # Drop date from features
    train = train.drop(columns=["date"])
    val = val.drop(columns=["date"])

    gc.collect()

    return train, val


# -----------------------------
# METRIC
# -----------------------------
def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


# -----------------------------
# TRAIN MODEL
# -----------------------------
def train_model(train: pd.DataFrame, val: pd.DataFrame):
    logger.info("Training XGBoost global model...")

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET]

    X_val = val[FEATURE_COLUMNS]
    y_val = val[TARGET]

    model = XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",  # memory efficient
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_val)

    score = wape(y_val, preds)

    logger.info(f"Validation WAPE: {score:.4f}")

    return model


# -----------------------------
# SAVE ARTIFACTS
# -----------------------------
def save_artifacts(model, encoders):
    logger.info("Saving model artifacts...")

    ARTIFACT_PATH.mkdir(exist_ok=True)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(encoders, ENCODER_FILE)

    logger.success("Artifacts saved.")


# -----------------------------
# RUN PIPELINE
# -----------------------------
def run_training():
    df = load_data()

    df, encoders = encode_categoricals(df)

    train, val = time_based_split(df)

    model = train_model(train, val)

    save_artifacts(model, encoders)


if __name__ == "__main__":
    run_training()