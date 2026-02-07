from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.xgboost
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.preprocess import load_feature_columns, to_model_vector

# --- Project paths / MLflow local tracking (mlruns folder) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
mlflow.set_tracking_uri(MLRUNS_DIR.resolve().as_uri())

EXPERIMENT_NAME = "Telco Churn - XGBoost"  # must match your notebook experiment name

app = FastAPI(title="Telco Churn Prediction API", version="1.0.0")

_model = None
_run_id: Optional[str] = None
_feature_cols: Optional[list[str]] = None


class PredictRequest(BaseModel):
    features: Dict[str, Any]
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    run_id: str
    churn_probability: float
    churn_prediction: int
    threshold: float


def _get_latest_run_id(experiment_name: str) -> str:
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"MLflow experiment not found: {experiment_name}")

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError(f"No MLflow runs found for experiment: {experiment_name}")

    return str(runs.loc[0, "run_id"])


def _load_latest_model() -> None:
    global _model, _run_id
    _run_id = _get_latest_run_id(EXPERIMENT_NAME)
    model_uri = f"runs:/{_run_id}/model"
    _model = mlflow.xgboost.load_model(model_uri)


def _ensure_loaded() -> None:
    """Ensure model and feature schema are loaded (handles reload edge cases)."""
    global _model, _run_id, _feature_cols

    if _feature_cols is None:
        _feature_cols = load_feature_columns()

    if _model is None or _run_id is None:
        _load_latest_model()


@app.on_event("startup")
def startup_event():
    try:
        _ensure_loaded()
    except Exception as e:
        # Don’t crash the server silently—surface it in /health
        print(f"[startup] Failed to load model/schema: {e}")


@app.get("/health")
def health():
    ok = (_model is not None) and (_feature_cols is not None)
    return {
        "status": "ok" if ok else "not_ready",
        "experiment": EXPERIMENT_NAME,
        "run_id": _run_id,
        "num_features": len(_feature_cols) if _feature_cols is not None else None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    global _feature_cols

    # Make sure model + schema are loaded
    try:
        _ensure_loaded()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Startup assets not loaded: {e}")

    # Convert raw input -> model vector
    try:
        x = to_model_vector(payload.features, _feature_cols)  # shape (1, num_features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {e}")

    # Enforce expected feature size (you observed 25)
    if x.shape[1] != len(_feature_cols):
        raise HTTPException(
            status_code=400,
            detail=f"Feature vector size mismatch: got {x.shape[1]}, expected {len(_feature_cols)}.",
        )

    # Predict
    try:
        proba = float(_model.predict_proba(x)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    threshold = float(payload.threshold)
    pred = int(proba >= threshold)

    return PredictResponse(
        run_id=_run_id or "",
        churn_probability=proba,
        churn_prediction=pred,
        threshold=threshold,
    )
