import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import json

st.set_page_config(layout="wide")

# =============================
# FILE PATHS
# =============================
DATA_OUTPUT = Path("data/outputs/inventory_plan.parquet")
FEATURE_FILE = Path("data/processed/features.parquet")
ENCODER_FILE = Path("artifacts/label_encoders.joblib")

SIM_RESULTS_FILE = Path("artifacts/simulation_results.csv")
COST_FILE = Path("artifacts/cost_comparison.csv")
COST_SUMMARY_FILE = Path("artifacts/cost_summary.json")
GLOBAL_FILE = Path("artifacts/inventory_simulation_metrics.json")

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_inventory():
    return pd.read_parquet(DATA_OUTPUT)

@st.cache_data
def load_encoders():
    return joblib.load(ENCODER_FILE)

@st.cache_data
def load_simulation():
    return pd.read_csv(SIM_RESULTS_FILE)

@st.cache_data
def load_cost():
    return pd.read_csv(COST_FILE)

@st.cache_data
def load_cost_summary():
    with open(COST_SUMMARY_FILE) as f:
        return json.load(f)

@st.cache_data
def load_global():
    with open(GLOBAL_FILE) as f:
        return json.load(f)

inventory_df = load_inventory()
encoders = load_encoders()
sim_df = load_simulation()
cost_df = load_cost()
cost_summary = load_cost_summary()
global_metrics = load_global()

# =============================
# BUILD SKU KEY
# =============================
inventory_df["sku_key"] = (
    inventory_df["item_id"].astype(str) + "_" +
    inventory_df["store_id"].astype(str)
)

# =============================
# SIDEBAR
# =============================
st.sidebar.title("SKU Selector")

sku_list = sorted(inventory_df["sku_key"].unique())
selected_sku = st.sidebar.selectbox("Select SKU", sku_list)

sku_inventory = inventory_df[inventory_df["sku_key"] == selected_sku]

encoded_item, encoded_store = selected_sku.split("_", 1)

decoded_item = encoders["item_id"].inverse_transform(
    [int(float(encoded_item))]
)[0]

decoded_store = encoders["store_id"].inverse_transform(
    [int(float(encoded_store))]
)[0]

st.sidebar.write("Decoded item_id:", decoded_item)
st.sidebar.write("Decoded store_id:", decoded_store)

# =============================
# LOAD SKU HISTORY
# =============================
@st.cache_data
def load_sku_history(item_id, store_id):
    df = pd.read_parquet(
        FEATURE_FILE,
        columns=["item_id", "store_id", "date", "sales"],
        filters=[
            ("item_id", "==", item_id),
            ("store_id", "==", store_id),
        ],
    )

    if df.empty:
        return df

    df = df.sort_values("date").tail(90).copy()

    fake_end_date = pd.Timestamp("2026-02-01")
    real_end_date = df["date"].max()
    df["date"] = df["date"] + (fake_end_date - real_end_date)

    return df

sku_history = load_sku_history(decoded_item, decoded_store)

# =============================
# TABS
# =============================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📈 Forecast",
        "📦 Inventory Simulation",
        "📊 Monitoring",
        "💰 Business Impact",
        "📋 SKU Analytics",
    ]
)

# =========================================================
# TAB 1 — FORECAST
# =========================================================
with tab1:
    st.title("Recent Demand + Future Forecast")

    if sku_inventory.empty or sku_history.empty:
        st.warning("No data for selected SKU")
    else:
        recent = sku_history.sort_values("date")

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(
            recent["date"],
            recent["sales"],
            label="Actual Demand (Nov 2025 – Feb 2026)"
        )

        p50 = sku_inventory["p50"].values[0]
        p90 = sku_inventory["p90"].values[0]
        reorder_point = sku_inventory["reorder_point"].values[0]

        last_date = recent["date"].max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=90,
            freq="D"
        )

        ax.plot(future_dates, [p50] * 90, linestyle="--", label="P50 Forecast")
        ax.plot(future_dates, [p90] * 90, linestyle=":", label="P90 Forecast")
        ax.axhline(reorder_point, linestyle="-.", label="Reorder Point")

        ax.legend()
        ax.set_title("Demand Trend and Future Forecast")
        st.pyplot(fig)

# =========================================================
# TAB 2 — INVENTORY SIMULATION (VISUAL)
# =========================================================
with tab2:
    st.title("Inventory Level Simulation")

    if sku_inventory.empty or sku_history.empty:
        st.warning("No data for selected SKU")
    else:
        df = sku_history.copy()

        inventory = 20
        lead_time = 7

        inventory_levels = []
        reorder_points = []
        stockouts = []

        p50 = sku_inventory["p50"].values[0]
        reorder_point = sku_inventory["reorder_point"].values[0]

        for demand in df["sales"]:
            inventory -= demand

            if inventory <= reorder_point:
                inventory += p50 * lead_time

            if inventory < 0:
                stockouts.append(0)
                inventory = 0
            else:
                stockouts.append(np.nan)

            inventory_levels.append(inventory)
            reorder_points.append(reorder_point)

        df["inventory"] = inventory_levels
        df["reorder_point"] = reorder_points
        df["stockout"] = stockouts

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df["date"], df["inventory"], label="Inventory Level")
        ax.plot(df["date"], df["reorder_point"], linestyle="--", label="Reorder Point")
        ax.scatter(df["date"], df["stockout"], label="Stockout", marker="x")
        ax.legend()
        st.pyplot(fig)

# =========================================================
# TAB 3 — MONITORING
# =========================================================
with tab3:
    st.title("Model Monitoring")

    if sku_history.empty:
        st.warning("No history for monitoring")
    else:
        df = sku_history.copy()
        df["error"] = np.abs(df["sales"] - df["sales"].shift(7).fillna(0))

        weekly = df.groupby(pd.Grouper(key="date", freq="W")).agg(
            demand_sum=("sales", "sum"),
            error_sum=("error", "sum"),
        )

        weekly["wape"] = weekly["error_sum"] / weekly["demand_sum"]
        st.line_chart(weekly["wape"])

# =========================================================
# TAB 4 — BUSINESS IMPACT
# =========================================================
with tab4:
    st.title("Business Impact")

    st.metric(
        "Total Cost Reduction (%)",
        f"{cost_summary['cost_reduction_percent']:.2f}%"
    )

    st.subheader("Cost Comparison")
    st.dataframe(cost_df)

    st.subheader("Global Service Level")
    st.write(global_metrics)

# =========================================================
# TAB 5 — SKU ANALYTICS
# =========================================================
with tab5:
    st.title("Per-SKU Performance")

    st.subheader("Top 10 Worst SKUs (Lowest Service Level)")
    worst = sim_df.sort_values("service_level").head(10)
    st.dataframe(worst)

    st.subheader("Top 10 Best SKUs (Lowest Total Cost)")
    best = sim_df.sort_values("total_cost").head(10)
    st.dataframe(best)

    st.subheader("Full SKU Table")
    st.dataframe(sim_df)