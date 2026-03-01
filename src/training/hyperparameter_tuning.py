from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import optuna
from lightgbm import LGBMRegressor
from loguru import logger
import sys

from src.features.feature_defs import FEATURE_COLUMNS, TARGET

logger.remove()
logger.add(sys.stdout, format="{message}")

PROCESSED_PATH = Path("data/processed")
ARTIFACT_PATH = Path("artifacts")

FEATURE_FILE = PROCESSED_PATH / "features.parquet"
ENCODER_FILE = ARTIFACT_PATH / "label_encoders_prob.joblib"
BEST_MODEL_FILE = ARTIFACT_PATH / "lgbm_p50_tuned.joblib"


# -----------------------------
# LOAD DATA (FAST EVAL SET)
# -----------------------------
def load_data():
    logger.info("Loading tuning dataset...")

    df = pd.read_parquet(FEATURE_FILE)

    cutoff_date = df["date"].max() - pd.Timedelta(days=180)
    df = df[df["date"] >= cutoff_date]

    # Top SKUs by demand
    top_items = (
        df.groupby("item_id")[TARGET]
        .sum()
        .sort_values(ascending=False)
        .head(300)
        .index
    )
    df = df[df["item_id"].isin(top_items)]

    logger.info(f"Tuning data shape: {df.shape}")

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
        df[col] = df[col].map(lambda x: x if x in le.classes_ else "NA")
        df[col] = le.transform(df[col])

    return df


# -----------------------------
# TIME SPLIT
# -----------------------------
def time_based_split(df):
    df = df.sort_values("date")
    split_date = df["date"].quantile(0.8)

    train = df[df["date"] <= split_date]
    val = df[df["date"] > split_date]

    return train, val


# -----------------------------
# METRIC
# -----------------------------
def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


# -----------------------------
# OPTUNA OBJECTIVE
# -----------------------------
def objective(trial):
    params = {
        "objective": "quantile",
        "alpha": 0.5,
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15),
        "num_leaves": trial.suggest_int("num_leaves", 31, 128),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
        "n_jobs": -1,
    }

    model = LGBMRegressor(**params)

    model.fit(X_train, y_train)

    preds = model.predict(X_val)

    score = wape(y_val, preds)

    return score


# -----------------------------
# RUN TUNING
# -----------------------------
def run_tuning():
    df = load_data()

    encoders = joblib.load(ENCODER_FILE)
    df = encode_categoricals(df, encoders)

    train, val = time_based_split(df)

    global X_train, y_train, X_val, y_val

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET]

    X_val = val[FEATURE_COLUMNS]
    y_val = val[TARGET]

    logger.info("Starting Optuna tuning...")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    logger.info(f"Best WAPE: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    # Train final model with best params
    best_model = LGBMRegressor(
        objective="quantile",
        alpha=0.5,
        **study.best_params,
        n_jobs=-1,
    )

    best_model.fit(X_train, y_train)

    joblib.dump(best_model, BEST_MODEL_FILE)

    logger.success("Tuned model saved.")


if __name__ == "__main__":
    run_tuning()