from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
import sys

from src.features.feature_defs import FEATURE_COLUMNS

logger.remove()
logger.add(sys.stdout, format="{message}")

FEATURE_FILE = Path("data/processed/features.parquet")


# -----------------------------
# PSI CALCULATION
# -----------------------------
def calculate_psi(expected, actual, bins=10):
    expected = expected.dropna()
    actual = actual.dropna()

    breakpoints = np.linspace(0, 100, bins + 1)
    breakpoints = np.percentile(expected, breakpoints)

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)

    psi = np.sum(
        (actual_perc - expected_perc)
        * np.log((actual_perc + 1e-6) / (expected_perc + 1e-6))
    )

    return psi


# -----------------------------
# LOAD DATA
# -----------------------------
def load_reference_and_current():
    df = pd.read_parquet(FEATURE_FILE)

    latest_date = df["date"].max()

    # Current inference data
    current = df[df["date"] == latest_date]

    # Reference = last 180 days before latest
    ref_start = latest_date - pd.Timedelta(days=180)
    reference = df[(df["date"] >= ref_start) & (df["date"] < latest_date)]

    logger.info(f"Reference shape: {reference.shape}")
    logger.info(f"Current shape: {current.shape}")

    return reference, current


# -----------------------------
# RUN DRIFT ANALYSIS
# -----------------------------
def run_drift_detection():
    reference, current = load_reference_and_current()

    drift_report = []

    logger.info("Calculating PSI for features...")

    for col in FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(reference[col]):
            logger.info(f"Skipping categorical feature: {col}")
            continue

        psi = calculate_psi(reference[col], current[col])

        drift_flag = "YES" if psi > 0.25 else "NO"

        drift_report.append(
            {
                "feature": col,
                "psi": psi,
                "drift": drift_flag,
            }
        )

    drift_df = pd.DataFrame(drift_report).sort_values("psi", ascending=False)

    logger.info("Drift report:")
    logger.info(drift_df)

    significant_drift = drift_df[drift_df["drift"] == "YES"]

    if not significant_drift.empty:
        logger.warning("Significant drift detected!")
    else:
        logger.success("No significant drift.")

    return drift_df


if __name__ == "__main__":
    run_drift_detection()