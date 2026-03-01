from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import joblib

app = FastAPI(title="Demand Forecast API")

ARTIFACT_PATH = Path("artifacts")

MODEL_FILE = ARTIFACT_PATH / "xgb_model.joblib"
ENCODER_FILE = ARTIFACT_PATH / "label_encoders.joblib"
INVENTORY_FILE = Path("data/outputs/inventory_plan.parquet")

print("🔄 Loading artifacts...")

model = joblib.load(MODEL_FILE)
encoders = joblib.load(ENCODER_FILE)
inventory_df = pd.read_parquet(INVENTORY_FILE)

print("🔄 Decoding inventory IDs...")

# -----------------------------
# CLEAN + DECODE INVENTORY IDS
# -----------------------------
inventory_df = inventory_df.dropna(subset=["item_id", "store_id"]).copy()

# Convert to int safely
inventory_df["item_id"] = inventory_df["item_id"].astype(float).astype(int)
inventory_df["store_id"] = inventory_df["store_id"].astype(float).astype(int)

# Decode to original string IDs
inventory_df["item_id"] = encoders["item_id"].inverse_transform(inventory_df["item_id"])
inventory_df["store_id"] = encoders["store_id"].inverse_transform(inventory_df["store_id"])

# -----------------------------
# BUILD FAST LOOKUP INDEX
# -----------------------------
inventory_df["sku_key"] = (
    inventory_df["item_id"].astype(str) + "|" +
    inventory_df["store_id"].astype(str)
)

inventory_lookup = inventory_df.set_index("sku_key")

print(f"✅ Inventory loaded: {len(inventory_lookup)} SKUs")

# -----------------------------
# REQUEST SCHEMA
# -----------------------------
class SKURequest(BaseModel):
    item_id: str
    store_id: str

# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# HELPER: FAST LOOKUP
# -----------------------------
def get_inventory_row(item_id: str, store_id: str):
    key = f"{item_id}|{store_id}"

    if key not in inventory_lookup.index:
        raise HTTPException(status_code=404, detail="SKU not found")

    return inventory_lookup.loc[key]

# -----------------------------
# FORECAST ENDPOINT
# -----------------------------
@app.post("/forecast")
def forecast(req: SKURequest):
    row = get_inventory_row(req.item_id, req.store_id)

    return {
        "item_id": req.item_id,
        "store_id": req.store_id,
        "p50": float(row["p50"]),
        "p90": float(row["p90"])
    }

# -----------------------------
# INVENTORY PLAN ENDPOINT
# -----------------------------
@app.post("/inventory-plan")
def inventory_plan(req: SKURequest):
    row = get_inventory_row(req.item_id, req.store_id)

    return {
        "item_id": req.item_id,
        "store_id": req.store_id,
        "p50": float(row["p50"]),
        "p90": float(row["p90"]),
        "reorder_point": float(row["reorder_point"]),
        "safety_stock": float(row["safety_stock"])
    }