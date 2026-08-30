"""
main.py
-------
Serves the churn model behind a REST API.
"""
import json
import logging
import os
import time
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from schemas import ChurnFeatures, PredictionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn-api")

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

MODEL = None
SCALER = None
FEATURE_NAMES = None
MODEL_VERSION = "unknown"


def load_artifacts():
    global MODEL, SCALER, FEATURE_NAMES, MODEL_VERSION

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    model_uri = os.environ.get("MODEL_URI", "models:/churn-classifier/Production")

    if tracking_uri:
        try:
            import mlflow
            import mlflow.sklearn

            mlflow.set_tracking_uri(tracking_uri)
            MODEL = mlflow.sklearn.load_model(model_uri)
            MODEL_VERSION = model_uri
            logger.info(f"Loaded model from MLflow registry: {model_uri}")
        except Exception as e:
            logger.warning(f"Falling back to local artifacts, MLflow load failed: {e}")

    if MODEL is None:
        base = os.environ.get("ARTIFACTS_DIR", "artifacts")
        MODEL = joblib.load(os.path.join(base, "model.joblib"))
        SCALER = joblib.load(os.path.join(base, "scaler.joblib"))
        with open(os.path.join(base, "feature_names.json")) as f:
            FEATURE_NAMES = json.load(f)
        MODEL_VERSION = "local-artifact"
        logger.info("Loaded model from local artifacts/")


@app.on_event("startup")
def startup():
    load_artifacts()


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION, "time": datetime.now(timezone.utc).isoformat()}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: ChurnFeatures):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    row = pd.DataFrame([features.model_dump()])

    if SCALER is not None:
        row = row[FEATURE_NAMES]
        row_values = SCALER.transform(row)
    else:
        row_values = row.values

    proba = float(MODEL.predict_proba(row_values)[0, 1])
    pred = int(proba >= 0.5)
    latency_ms = (time.time() - start) * 1000

    logger.info(json.dumps({
        "event": "prediction",
        "features": features.model_dump(),
        "churn_probability": proba,
        "churn_prediction": pred,
        "latency_ms": round(latency_ms, 2),
        "model_version": MODEL_VERSION,
    }))

    return PredictionResponse(
        churn_probability=proba, churn_prediction=pred, model_version=MODEL_VERSION
    )
