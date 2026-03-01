from pathlib import Path
import pandas as pd
import joblib
from loguru import logger
import sys
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor

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
    logger.info("Loading training data for probabilistic model...")

    df = pd.read_parquet(FEATURE_FILE)

    cutoff_date = df["date"].max() - pd.Timedelta(days=730)
    df = df[df["date"] >= cutoff_date]

    logger.info(f"Training data shape: {df.shape}")

    return df


# -----------------------------
# ENCODE CATEGORICALS
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
        df[col] = df[col].astype(str).fillna("NA")
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders


# -----------------------------
# TIME-BASED SPLIT
# -----------------------------
def time_based_split(df: pd.DataFrame):
    df = df.sort_values("date")

    split_date = df["date"].quantile(0.8)

    train = df[df["date"] <= split_date]
    val = df[df["date"] > split_date]

    logger.info(f"Train shape: {train.shape}")
    logger.info(f"Validation shape: {val.shape}")

    return train, val


# -----------------------------
# TRAIN QUANTILE MODEL
# -----------------------------
def train_quantile_model(train, val, quantile, name):
    logger.info(f"Training LightGBM quantile model: {name}")

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET]

    X_val = val[FEATURE_COLUMNS]
    y_val = val[TARGET]

    model = LGBMRegressor(
        objective="quantile",
        alpha=quantile,
        n_estimators=200,
        learning_rate=0.1,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_val)

    wape = (abs(y_val - preds).sum()) / (abs(y_val).sum())

    logger.info(f"{name} WAPE: {wape:.4f}")

    return model


# -----------------------------
# SAVE ARTIFACTS
# -----------------------------
def save_artifacts(p50_model, p90_model, encoders):
    logger.info("Saving probabilistic model artifacts...")

    ARTIFACT_PATH.mkdir(exist_ok=True)

    joblib.dump(p50_model, P50_MODEL_FILE)
    joblib.dump(p90_model, P90_MODEL_FILE)
    joblib.dump(encoders, ENCODER_FILE)

    logger.success("Probabilistic models saved.")


# -----------------------------
# RUN PIPELINE
# -----------------------------
def run_training():
    df = load_data()

    df, encoders = encode_categoricals(df)

    train, val = time_based_split(df)

    p50_model = train_quantile_model(train, val, 0.5, "P50")
    p90_model = train_quantile_model(train, val, 0.9, "P90")

    save_artifacts(p50_model, p90_model, encoders)


if __name__ == "__main__":
    run_training()