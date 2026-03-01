from loguru import logger
import sys
from datetime import datetime

from src.monitoring.data_drift import run_drift_detection
from src.monitoring.prediction_drift import run_prediction_drift
from src.training.train import run_training

logger.remove()
logger.add(sys.stdout, format="{message}")

WAPE_THRESHOLD = 0.75


def check_data_drift():
    drift_df = run_drift_detection()
    significant = drift_df[drift_df["drift"] == "YES"]
    return not significant.empty


def check_prediction_drift():
    weekly_wape = run_prediction_drift()
    latest_wape = weekly_wape["wape"].iloc[-1]
    return latest_wape > WAPE_THRESHOLD, latest_wape


def retrain_model():
    logger.warning("Retraining model...")

    run_training()

    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.success(f"New model version trained at {version}")


def run_retraining_pipeline():
    logger.info("Running retraining pipeline...")

    data_drift_flag = check_data_drift()
    pred_drift_flag, latest_wape = check_prediction_drift()

    logger.info(f"Latest WAPE: {latest_wape:.4f}")

    if data_drift_flag:
        logger.warning("Data drift detected.")
    if pred_drift_flag:
        logger.warning("Prediction drift detected.")

    if data_drift_flag or pred_drift_flag:
        retrain_model()
    else:
        logger.success("No retraining required.")


if __name__ == "__main__":
    run_retraining_pipeline()