import pandas as pd
from pathlib import Path

from src.features.feature_defs import FEATURE_COLUMNS

FEATURE_FILE = Path("data/processed/features.parquet")


def test_feature_file_exists():
    assert FEATURE_FILE.exists(), "Feature file not found"


def test_required_feature_columns():
    df = pd.read_parquet(FEATURE_FILE)

    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]

    assert len(missing_cols) == 0, f"Missing columns: {missing_cols}"


def test_feature_order_and_selection():
    """
    Ensure that the feature matrix used for inference
    has the correct columns in the correct order.
    This prevents silent model input bugs.
    """
    df = pd.read_parquet(FEATURE_FILE)

    latest_date = df["date"].max()
    df = df[df["date"] == latest_date]

    X = df[FEATURE_COLUMNS]

    assert list(X.columns) == FEATURE_COLUMNS