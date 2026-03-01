from pathlib import Path
import pandas as pd
import numpy as np
import json
import joblib

FEATURE_FILE = Path("data/processed/features.parquet")
INVENTORY_FILE = Path("data/outputs/inventory_plan.parquet")
ENCODER_FILE = Path("artifacts/label_encoders.joblib")
ARTIFACT_FILE = Path("artifacts/metrics.json")

LEAD_TIME_DAYS = 7

# -----------------------------
# LOAD INVENTORY
# -----------------------------
print("Loading inventory forecasts...")
inv = pd.read_parquet(INVENTORY_FILE)
print("Inventory raw shape:", inv.shape)

print("Loading encoders...")
encoders = joblib.load(ENCODER_FILE)

# -----------------------------
# CLEAN + DECODE INVENTORY IDS
# -----------------------------
print("Cleaning inventory IDs...")

inv = inv.dropna(subset=["item_id", "store_id"]).copy()

inv["item_id"] = inv["item_id"].astype(float).astype(int)
inv["store_id"] = inv["store_id"].astype(float).astype(int)

print("Decoding inventory IDs...")

inv["item_id"] = encoders["item_id"].inverse_transform(inv["item_id"])
inv["store_id"] = encoders["store_id"].inverse_transform(inv["store_id"])

print("Inventory cleaned shape:", inv.shape)
print(inv[["item_id", "store_id", "p50"]].head())

# -----------------------------
# LOAD RECENT FEATURE DATA ONLY
# -----------------------------
print("Loading recent feature data (last 180 days only)...")

df = pd.read_parquet(
    FEATURE_FILE,
    columns=["item_id", "store_id", "date", "sales"],
)

cutoff = df["date"].max() - pd.Timedelta(days=180)
df = df[df["date"] >= cutoff].copy()

print("Feature shape after date filter:", df.shape)

# -----------------------------
# BUILD BASELINE FORECAST (28D MA)
# -----------------------------
print("Computing 28-day rolling baseline...")

df = df.sort_values(["item_id", "store_id", "date"])

df["baseline_daily"] = (
    df.groupby(["item_id", "store_id"])["sales"]
    .transform(lambda x: x.shift(1).rolling(28).mean())
)

df = df.dropna(subset=["baseline_daily"])
print("Rows after baseline creation:", df.shape)

# -----------------------------
# MERGE MODEL FORECAST (P50)
# -----------------------------
print("Merging model forecasts...")

df = df.merge(
    inv[["item_id", "store_id", "p50"]],
    on=["item_id", "store_id"],
    how="inner"
)

print("Merged shape:", df.shape)

if len(df) == 0:
    raise ValueError("❌ Merge produced 0 rows. Check ID decoding or coverage.")

# -----------------------------
# LEAD-TIME AGGREGATION (7 DAYS)
# -----------------------------
print(f"Aggregating to {LEAD_TIME_DAYS}-day demand windows...")

df["actual_lt"] = (
    df.groupby(["item_id", "store_id"])["sales"]
    .transform(lambda x: x.rolling(LEAD_TIME_DAYS).sum())
)

df["baseline_lt"] = df["baseline_daily"] * LEAD_TIME_DAYS
df["model_lt"] = df["p50"] * LEAD_TIME_DAYS

df = df.dropna(subset=["actual_lt"])

print("Rows after lead-time aggregation:", df.shape)

# -----------------------------
# WAPE FUNCTION
# -----------------------------
def wape(y_true, y_pred):
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return np.nan
    return np.sum(np.abs(y_true - y_pred)) / denom

# -----------------------------
# COMPUTE METRICS (LEAD-TIME)
# -----------------------------
print("Computing LEAD-TIME WAPE metrics...")

baseline_wape = wape(df["actual_lt"].values, df["baseline_lt"].values)
model_wape = wape(df["actual_lt"].values, df["model_lt"].values)

improvement = (baseline_wape - model_wape) / baseline_wape * 100

metrics = {
    "baseline_wape_7d": float(baseline_wape),
    "model_wape_7d": float(model_wape),
    "improvement_percent": float(improvement),
    "rows_evaluated": int(len(df)),
    "unique_skus_evaluated": int(
        df[["item_id", "store_id"]].drop_duplicates().shape[0]
    ),
    "lead_time_days": LEAD_TIME_DAYS,
}

print("\n📊 LEAD-TIME RESULTS")
print(json.dumps(metrics, indent=2))

# -----------------------------
# SAVE METRICS
# -----------------------------
ARTIFACT_FILE.parent.mkdir(exist_ok=True)

with open(ARTIFACT_FILE, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n✅ Metrics saved to {ARTIFACT_FILE}")