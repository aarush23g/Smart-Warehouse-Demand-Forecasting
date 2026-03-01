from fastapi import FastAPI
import pandas as pd
import joblib
from pathlib import Path

from src.inference.schemas import ForecastRequest, ForecastResponse
from src.features.feature_defs import FEATURE_COLUMNS
from src.inference.prediction_logic import compute_inventory_decision

app = FastAPI(title="Warehouse Demand Forecast API")

ARTIFACT_PATH = Path("artifacts")

P50_MODEL_FILE = ARTIFACT_PATH / "lgbm_p50.joblib"
P90_MODEL_FILE = ARTIFACT_PATH / "lgbm_p90.joblib"
ENCODER_FILE = ARTIFACT_PATH / "label_encoders_prob.joblib"


# -----------------------------
# LOAD MODELS AT STARTUP
# -----------------------------
@app.on_event("startup")
def load_models():
    global p50_model, p90_model, encoders

    p50_model = joblib.load(P50_MODEL_FILE)
    p90_model = joblib.load(P90_MODEL_FILE)
    encoders = joblib.load(ENCODER_FILE)


# -----------------------------
# ENCODE CATEGORICALS
# -----------------------------
def encode_input(data: dict):
    categorical_cols = [
        "item_id",
        "store_id",
        "event_name_1",
        "event_type_1",
    ]

    for col in categorical_cols:
        le = encoders[col]
        value = str(data[col])

        if value not in le.classes_:
            value = "NA"

        data[col] = le.transform([value])[0]

    return data


# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------------
# FORECAST ENDPOINT
# -----------------------------
@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    data = request.dict()

    # Current inventory proxy
    current_inventory = data["lag_7"]

    # Encode categorical features
    data = encode_input(data)

    df = pd.DataFrame([data])

    # Feature alignment check
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    X = df[FEATURE_COLUMNS]

    # Generate probabilistic predictions
    p50 = float(p50_model.predict(X)[0])
    p90 = float(p90_model.predict(X)[0])

    # Compute inventory decision using shared logic
    decision = compute_inventory_decision(
        p50=p50,
        p90=p90,
        current_inventory=current_inventory,
        lead_time=7,
        service_level=0.95,
    )

    return ForecastResponse(
        p50=p50,
        p90=p90,
        safety_stock=decision["safety_stock"],
        reorder_point=decision["reorder_point"],
        order_qty=decision["order_qty"],
    )