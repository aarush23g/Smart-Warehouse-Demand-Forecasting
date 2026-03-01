from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json

FEATURE_FILE = Path("data/processed/features.parquet")
INVENTORY_FILE = Path("data/outputs/inventory_plan.parquet")
ENCODER_FILE = Path("artifacts/label_encoders.joblib")

ARTIFACT_JSON = Path("artifacts/inventory_simulation_metrics.json")
ARTIFACT_CSV = Path("artifacts/simulation_results.csv")

HOLDING_COST = 0.1
STOCKOUT_COST = 5.0
LEAD_TIME = 7
SIM_DAYS = 60

print("Loading inventory plan...")
inv = pd.read_parquet(INVENTORY_FILE)

print("Loading encoders...")
encoders = joblib.load(ENCODER_FILE)

# -----------------------------
# DECODE IDS
# -----------------------------
inv = inv.dropna(subset=["item_id", "store_id"]).copy()

inv["item_id"] = encoders["item_id"].inverse_transform(inv["item_id"].astype(int))
inv["store_id"] = encoders["store_id"].inverse_transform(inv["store_id"].astype(int))

# -----------------------------
# LOAD RECENT SALES
# -----------------------------
print("Loading recent sales (last 120 days)...")

df = pd.read_parquet(
    FEATURE_FILE,
    columns=["item_id", "store_id", "date", "sales"]
)

cutoff = df["date"].max() - pd.Timedelta(days=120)
df = df[df["date"] >= cutoff]

df = df.sort_values(["item_id", "store_id", "date"])

# Keep only SKUs in inventory plan
df = df.merge(
    inv[["item_id", "store_id", "p50", "p90", "reorder_point"]],
    on=["item_id", "store_id"],
    how="inner"
)

print("Simulation dataset shape:", df.shape)

# -----------------------------
# BASELINE FORECAST (28d MA)
# -----------------------------
df["baseline_daily"] = (
    df.groupby(["item_id", "store_id"])["sales"]
    .transform(lambda x: x.shift(1).rolling(28).mean())
)

df = df.dropna(subset=["baseline_daily"])

# -----------------------------
# SIMULATION FUNCTION
# -----------------------------
def simulate(policy="baseline"):
    sku_results = []

    global_stockout = 0
    global_holding = 0
    global_stockout_cost = 0
    global_demand = 0

    for (item, store), g in df.groupby(["item_id", "store_id"]):
        g = g.tail(SIM_DAYS)

        inventory = 20
        pipeline = []

        sku_stockout = 0
        sku_holding = 0
        sku_stockout_cost = 0
        sku_demand = 0
        orders_placed = 0
        inventory_sum = 0

        for _, row in g.iterrows():
            demand = row["sales"]
            sku_demand += demand

            # Receive pipeline
            pipeline = [p - 1 for p in pipeline]
            arrivals = pipeline.count(0)

            if arrivals > 0:
                if policy == "model":
                    order_qty = row["p50"] * LEAD_TIME
                else:
                    order_qty = row["baseline_daily"] * LEAD_TIME

                inventory += arrivals * order_qty

            pipeline = [p for p in pipeline if p > 0]

            # Demand fulfillment
            if inventory >= demand:
                inventory -= demand
            else:
                lost = demand - inventory
                sku_stockout += lost
                sku_stockout_cost += lost * STOCKOUT_COST
                inventory = 0

            # Reorder logic
            if policy == "model":
                reorder_point = row["reorder_point"]
            else:
                reorder_point = row["baseline_daily"] * LEAD_TIME

            if inventory <= reorder_point:
                pipeline.append(LEAD_TIME)
                orders_placed += 1

            sku_holding += inventory * HOLDING_COST
            inventory_sum += inventory

        avg_inventory = inventory_sum / len(g) if len(g) > 0 else 0
        service_level = 1 - (sku_stockout / sku_demand) if sku_demand > 0 else 1

        sku_results.append({
            "item_id": item,
            "store_id": store,
            "policy": policy,
            "stockout_units": sku_stockout,
            "holding_cost": sku_holding,
            "stockout_cost": sku_stockout_cost,
            "total_cost": sku_holding + sku_stockout_cost,
            "service_level": service_level,
            "total_demand": sku_demand,
            "orders_placed": orders_placed,
            "avg_inventory": avg_inventory,
        })

        global_stockout += sku_stockout
        global_holding += sku_holding
        global_stockout_cost += sku_stockout_cost
        global_demand += sku_demand

    global_service_level = 1 - (global_stockout / global_demand)

    global_metrics = {
        "stockout_units": float(global_stockout),
        "holding_cost": float(global_holding),
        "stockout_cost": float(global_stockout_cost),
        "total_cost": float(global_holding + global_stockout_cost),
        "service_level": float(global_service_level),
    }

    return pd.DataFrame(sku_results), global_metrics

# -----------------------------
# RUN SIMULATIONS
# -----------------------------
print("Running baseline simulation...")
baseline_df, baseline_metrics = simulate(policy="baseline")

print("Running model simulation...")
model_df, model_metrics = simulate(policy="model")

# -----------------------------
# SAVE SKU-LEVEL CSV
# -----------------------------
sku_df = pd.concat([baseline_df, model_df], ignore_index=True)

ARTIFACT_CSV.parent.mkdir(exist_ok=True)
sku_df.to_csv(ARTIFACT_CSV, index=False)

print(f"✅ SKU-level results saved → {ARTIFACT_CSV}")

# -----------------------------
# GLOBAL COMPARISON METRICS
# -----------------------------
results = {
    "baseline": baseline_metrics,
    "model": model_metrics,
    "stockout_reduction_percent": (
        (baseline_metrics["stockout_units"] - model_metrics["stockout_units"])
        / baseline_metrics["stockout_units"] * 100
    ),
    "service_level_improvement": (
        model_metrics["service_level"] - baseline_metrics["service_level"]
    ),
    "holding_cost_change_percent": (
        (model_metrics["holding_cost"] - baseline_metrics["holding_cost"])
        / baseline_metrics["holding_cost"] * 100
    ),
    "total_cost_change_percent": (
        (model_metrics["total_cost"] - baseline_metrics["total_cost"])
        / baseline_metrics["total_cost"] * 100
    ),
}

print("\n📊 GLOBAL RESULTS")
print(json.dumps(results, indent=2))

with open(ARTIFACT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print(f"✅ Global metrics saved → {ARTIFACT_JSON}")