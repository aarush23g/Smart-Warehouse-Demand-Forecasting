from fastapi import FastAPI, HTTPException
from pathlib import Path
import pandas as pd
import joblib
import json

app = FastAPI(title="Inventory Policy Optimization API")

# -----------------------------
# PATHS
# -----------------------------
INVENTORY_FILE = Path("data/outputs/inventory_plan.parquet")
ENCODER_FILE = Path("artifacts/label_encoders.joblib")
SIM_METRICS_FILE = Path("artifacts/inventory_simulation_metrics.json")

# -----------------------------
# LOAD ARTIFACTS (ON STARTUP)
# -----------------------------
inventory_df = pd.read_parquet(INVENTORY_FILE)
encoders = joblib.load(ENCODER_FILE)

with open(SIM_METRICS_FILE, "r") as f:
    sim_metrics = json.load(f)

BEST_POLICY = sim_metrics["best_policy"]
BEST_QUANTILE = BEST_POLICY["quantile"]
BEST_SERVICE = BEST_POLICY["service_level"]
BEST_COST = BEST_POLICY["total_cost"]

# -----------------------------
# CLEAN + DECODE IDS
# -----------------------------
inventory_df = inventory_df.dropna(subset=["item_id", "store_id"]).copy()

inventory_df["item_id"] = encoders["item_id"].inverse_transform(
    inventory_df["item_id"].astype(int)
)

inventory_df["store_id"] = encoders["store_id"].inverse_transform(
    inventory_df["store_id"].astype(int)
)

# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# POLICY ENDPOINT
# -----------------------------
@app.post("/optimize-policy")
def optimize_policy(payload: dict):

    item_id = payload.get("item_id")
    store_id = payload.get("store_id")

    if item_id is None or store_id is None:
        raise HTTPException(status_code=400, detail="item_id and store_id required")

    sku = inventory_df[
        (inventory_df["item_id"] == item_id) &
        (inventory_df["store_id"] == store_id)
    ]

    if sku.empty:
        raise HTTPException(status_code=404, detail="SKU not found")

    row = sku.iloc[0]

    return {
        "item_id": item_id,
        "store_id": store_id,
        "recommended_quantile": BEST_QUANTILE,
        "reorder_point": float(row["reorder_point"]),
        "safety_stock": float(row["safety_stock"]),
        "expected_service_level": BEST_SERVICE,
        "expected_total_cost": BEST_COST
    }