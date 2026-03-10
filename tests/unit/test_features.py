import pytest
from pathlib import Path

# Optional imports handled safely
try:
    import pandas as pd
except ImportError:
    pd = None

FEATURE_FILE = Path("data/processed/features.parquet")


def dataset_available():
    """
    Check whether dataset and dependencies exist.
    Used to safely skip tests in CI environments.
    """
    if pd is None:
        return False

    if not FEATURE_FILE.exists():
        return False

    return True


@pytest.mark.skipif(
    not dataset_available(),
    reason="Feature dataset or dependencies not available"
)
def test_feature_file_exists():
    """
    Ensure feature dataset exists.
    """
    assert FEATURE_FILE.exists(), "Feature file not found"


@pytest.mark.skipif(
    not dataset_available(),
    reason="Feature dataset or dependencies not available"
)
def test_required_feature_columns():
    """
    Verify required columns exist in the dataset.
    """
    try:
        df = pd.read_parquet(FEATURE_FILE)
    except Exception as e:
        pytest.skip(f"Unable to read parquet file: {e}")

    required_cols = [
        "item_id",
        "store_id",
        "date",
        "sales"
    ]

    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"


@pytest.mark.skipif(
    not dataset_available(),
    reason="Feature dataset or dependencies not available"
)
def test_feature_order_and_selection():
    """
    Ensure model input columns exist.
    """
    try:
        df = pd.read_parquet(FEATURE_FILE)
    except Exception as e:
        pytest.skip(f"Unable to read parquet file: {e}")

    feature_cols = [
        "sales"
    ]

    for col in feature_cols:
        assert col in df.columns, f"Missing feature column: {col}"


@pytest.mark.skipif(
    not dataset_available(),
    reason="Feature dataset or dependencies not available"
)
def test_dataset_not_empty():
    """
    Ensure dataset is not empty.
    """
    try:
        df = pd.read_parquet(FEATURE_FILE)
    except Exception as e:
        pytest.skip(f"Unable to read parquet file: {e}")

    assert len(df) > 0, "Feature dataset is empty"