"""
src/predict.py
──────────────
Loads a saved model + scaler and generates speed forecasts.

Returns:
  - predicted_speed_mph  : float
  - congestion_level     : str key ('free_flow', 'moderate', 'heavy', 'severe')
  - congestion_label     : str display label with emoji
  - model_name           : str
  - horizon_min          : int
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    MODELS_DIR, CLEANED_DATA_PATH, FORECAST_HORIZONS_MIN, LOG_FORMAT, LOG_LEVEL
)
from src.feature_engineering import make_features_for_sensor, FEATURE_COLS
from src.congestion import classify_speed, get_label

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
def _latest_model_paths(horizon_min: int):
    """Find the most recently saved model and scaler for a given horizon."""
    models_dir = Path(MODELS_DIR)
    model_files = sorted(models_dir.glob(f"best_model_{horizon_min}min_v*.joblib"))
    scaler_files = sorted(models_dir.glob(f"scaler_{horizon_min}min_v*.joblib"))
    meta_files   = sorted(models_dir.glob(f"model_meta_{horizon_min}min.json"))

    if not model_files:
        return None, None, None
    return model_files[-1], scaler_files[-1] if scaler_files else None, meta_files[-1] if meta_files else None


def load_model(horizon_min: int):
    """Load the best saved model + scaler + metadata for the given horizon."""
    model_path, scaler_path, meta_path = _latest_model_paths(horizon_min)
    if model_path is None:
        raise FileNotFoundError(
            f"No trained model found for horizon {horizon_min} min. "
            f"Run scripts/03_train_models.py first."
        )
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path else None
    meta   = json.loads(meta_path.read_text()) if meta_path else {}
    log.debug("Loaded model: %s", model_path.name)
    return model, scaler, meta


# ─────────────────────────────────────────────────────────────────
def predict_for_sensor(
    sensor_id: str,
    horizon_min: int,
    long_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Generate a single-point forecast for `sensor_id` at `horizon_min` minutes ahead.

    Uses the most recent observation as the basis for feature construction.
    """
    try:
        model, scaler, meta = load_model(horizon_min)
    except FileNotFoundError as e:
        log.error(str(e))
        return {"error": str(e), "horizon_min": horizon_min, "sensor_id": sensor_id}

    feat_df = make_features_for_sensor(long_df, sensor_id, horizon_min)
    if feat_df.empty:
        return {"error": f"Insufficient data for sensor {sensor_id}", "horizon_min": horizon_min}

    # Use the most recent row (latest available features)
    latest = feat_df.tail(1)
    X = latest[FEATURE_COLS].values
    if scaler is not None:
        X = scaler.transform(X)

    pred_speed = float(model.predict(X)[0])
    pred_speed = max(0.0, pred_speed)  # speed cannot be negative

    level = classify_speed(pred_speed)
    label = get_label(level)
    forecast_time = latest["timestamp"].iloc[0] + pd.Timedelta(minutes=horizon_min)

    return {
        "sensor_id":           sensor_id,
        "horizon_min":         horizon_min,
        "generated_at":        datetime.utcnow().isoformat(),
        "forecast_for":        str(forecast_time),
        "predicted_speed_mph": round(pred_speed, 2),
        "congestion_level":    level,
        "congestion_label":    label,
        "model_name":          meta.get("model_name", "unknown"),
        "model_version":       meta.get("version", "unknown"),
    }


# ─────────────────────────────────────────────────────────────────
def predict_all_horizons(
    sensor_id: str,
    long_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Predict for all configured horizons (15, 30, 60 min) at once."""
    results = []
    for h in FORECAST_HORIZONS_MIN:
        r = predict_for_sensor(sensor_id, h, long_df)
        results.append(r)
    return results


# ─────────────────────────────────────────────────────────────────
def evaluate_on_test(
    sensor_id: str,
    horizon_min: int,
    long_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Run the model on held-out test data and return actual vs predicted arrays
    for plotting and metric display.
    """
    from src.feature_engineering import chronological_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    try:
        model, scaler, meta = load_model(horizon_min)
    except FileNotFoundError as e:
        return {"error": str(e)}

    feat_df = make_features_for_sensor(long_df, sensor_id, horizon_min)
    if feat_df.empty:
        return {"error": f"No features for sensor {sensor_id}"}

    _, _, test_df = chronological_split(feat_df)
    if len(test_df) < 10:
        return {"error": "Insufficient test samples"}

    X_te = test_df[FEATURE_COLS].values
    y_te = test_df["target_speed"].values
    if scaler is not None:
        X_te = scaler.transform(X_te)

    preds = model.predict(X_te)
    preds = np.maximum(preds, 0)

    return {
        "sensor_id":     sensor_id,
        "horizon_min":   horizon_min,
        "timestamps":    test_df["timestamp"].tolist(),
        "actual":        y_te.tolist(),
        "predicted":     preds.tolist(),
        "mae":           float(mean_absolute_error(y_te, preds)),
        "rmse":          float(np.sqrt(mean_squared_error(y_te, preds))),
        "model_name":    meta.get("model_name", "unknown"),
        "model_version": meta.get("version", "unknown"),
    }
