import pytest
from pathlib import Path

try:
    import pandas as pd
except Exception:
    pd = None


FEATURE_FILE = Path("data/processed/features.parquet")


# Skip entire module safely if requirements are missing
if pd is None or not FEATURE_FILE.exists():
    pytest.skip(
        "Skipping feature tests because dataset is not available in CI environment",
        allow_module_level=True
    )


def test_feature_file_exists():
    """Verify that the feature file exists."""
    assert FEATURE_FILE.exists()


def test_required_feature_columns():
    """Verify required columns exist in dataset."""

    df = pd.read_parquet(FEATURE_FILE)

    required_cols = [
        "item_id",
        "store_id",
        "date",
        "sales"
    ]

    for col in required_cols:
        assert col in df.columns


def test_feature_order_and_selection():
    """Ensure model feature columns exist."""

    df = pd.read_parquet(FEATURE_FILE)

    feature_cols = [
        "sales"
    ]

    for col in feature_cols:
        assert col in df.columns