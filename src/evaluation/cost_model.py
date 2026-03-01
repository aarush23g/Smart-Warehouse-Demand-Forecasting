from pathlib import Path
import pandas as pd
import json

SIMULATION_FILE = Path("artifacts/simulation_results.csv")
OUTPUT_FILE = Path("artifacts/cost_comparison.csv")
SUMMARY_FILE = Path("artifacts/cost_summary.json")

HOLDING_COST_PER_UNIT_PER_DAY = 0.02
STOCKOUT_PENALTY_PER_UNIT = 5
ORDER_COST = 50

print("Loading SKU simulation results...")
df = pd.read_csv(SIMULATION_FILE)

print("Input shape:", df.shape)

# -----------------------------
# RECOMPUTE COSTS WITH NEW PARAMS
# -----------------------------
df["holding_cost_adj"] = df["avg_inventory"] * HOLDING_COST_PER_UNIT_PER_DAY * 60
df["stockout_cost_adj"] = df["stockout_units"] * STOCKOUT_PENALTY_PER_UNIT
df["ordering_cost"] = df["orders_placed"] * ORDER_COST

df["total_cost_adj"] = (
    df["holding_cost_adj"] +
    df["stockout_cost_adj"] +
    df["ordering_cost"]
)

# -----------------------------
# AGGREGATE BY POLICY
# -----------------------------
agg = (
    df.groupby("policy")
    .agg(
        holding_cost=("holding_cost_adj", "sum"),
        stockout_cost=("stockout_cost_adj", "sum"),
        ordering_cost=("ordering_cost", "sum"),
        total_cost=("total_cost_adj", "sum"),
    )
    .reset_index()
)

# -----------------------------
# COMPUTE IMPROVEMENT %
# -----------------------------
baseline_cost = agg.loc[agg["policy"] == "baseline", "total_cost"].values[0]
model_cost = agg.loc[agg["policy"] == "model", "total_cost"].values[0]

cost_reduction = (baseline_cost - model_cost) / baseline_cost * 100

summary = {
    "baseline_total_cost": float(baseline_cost),
    "model_total_cost": float(model_cost),
    "cost_reduction_percent": float(cost_reduction),
}

print("\n📊 COST COMPARISON")
print(agg)

print("\n📊 SUMMARY")
print(summary)

# -----------------------------
# SAVE FILES
# -----------------------------
OUTPUT_FILE.parent.mkdir(exist_ok=True)
agg.to_csv(OUTPUT_FILE, index=False)

with open(SUMMARY_FILE, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n✅ Cost comparison saved → {OUTPUT_FILE}")
print(f"✅ Cost summary saved → {SUMMARY_FILE}")