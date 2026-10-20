"""
src/train_model.py
──────────────────
Trains and evaluates four models per forecast horizon:
  1. Persistence baseline
  2. Historical seasonal baseline
  3. Random Forest Regressor
  4. XGBoost Regressor

Saves best model + metadata to models/ directory.
Saves evaluation metrics to reports/model_metrics.json.
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    MODELS_DIR, REPORTS_DIR, CLEANED_DATA_PATH, FORECAST_HORIZONS_MIN,
    TOP_N_SENSORS_FOR_TRAINING, RF_PARAMS, XGB_PARAMS, LOG_FORMAT, LOG_LEVEL,
)
from src.feature_engineering import (
    make_features_for_sensor, chronological_split, get_X_y, FEATURE_COLS
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


# ─────────────────────────────────────────────────────────────────
def persistence_predict(test_df: pd.DataFrame) -> np.ndarray:
    """Baseline: predict last observed speed as next value."""
    return test_df["lag_1"].values


def seasonal_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Baseline: use historical mean for same hour + day_of_week."""
    hist = (
        train_df.groupby(["hour", "day_of_week"])["target_speed"]
        .mean()
        .reset_index()
        .rename(columns={"target_speed": "hist_mean"})
    )
    merged = test_df[["hour", "day_of_week"]].merge(hist, on=["hour", "day_of_week"], how="left")
    # fill any gaps with global mean
    global_mean = train_df["target_speed"].mean()
    return merged["hist_mean"].fillna(global_mean).values


# ─────────────────────────────────────────────────────────────────
def train_for_horizon(
    long_df: pd.DataFrame,
    horizon_min: int,
    sensors: List[str],
) -> Dict[str, Any]:
    """
    Train all models for a given horizon across the selected sensors.
    Returns evaluation metrics dict.
    """
    log.info("── Horizon: %d min ──────────────────────────────", horizon_min)
    all_train, all_val, all_test = [], [], []

    for sid in sensors:
        feat_df = make_features_for_sensor(long_df, sid, horizon_min)
        if feat_df.empty:
            continue
        tr, va, te = chronological_split(feat_df)
        all_train.append(tr)
        all_val.append(va)
        all_test.append(te)

    if not all_train:
        log.warning("No data for horizon %d min.", horizon_min)
        return {}

    train = pd.concat(all_train, ignore_index=True)
    val   = pd.concat(all_val,   ignore_index=True)
    test  = pd.concat(all_test,  ignore_index=True)

    X_tr, y_tr = get_X_y(train)
    X_va, y_va = get_X_y(val)
    X_te, y_te = get_X_y(test)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    results: Dict[str, Dict] = {}

    # ── 1. Persistence ─────────────────────────────────────
    p_preds = persistence_predict(test)
    results["persistence"] = {
        "test_mae":  mae(y_te, p_preds),
        "test_rmse": rmse(y_te, p_preds),
    }
    log.info("  Persistence   MAE=%.3f  RMSE=%.3f", results["persistence"]["test_mae"], results["persistence"]["test_rmse"])

    # ── 2. Seasonal Baseline ────────────────────────────────
    s_preds = seasonal_predict(train, test)
    results["seasonal"] = {
        "test_mae":  mae(y_te, s_preds),
        "test_rmse": rmse(y_te, s_preds),
    }
    log.info("  Seasonal      MAE=%.3f  RMSE=%.3f", results["seasonal"]["test_mae"], results["seasonal"]["test_rmse"])

    # ── 3. Random Forest ────────────────────────────────────
    log.info("  Training Random Forest …")
    t0 = time.time()
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_tr_s, y_tr)
    rf_preds = rf.predict(X_te_s)
    rf_val_preds = rf.predict(X_va_s)
    results["random_forest"] = {
        "val_mae":   mae(y_va, rf_val_preds),
        "val_rmse":  rmse(y_va, rf_val_preds),
        "test_mae":  mae(y_te, rf_preds),
        "test_rmse": rmse(y_te, rf_preds),
        "train_sec": round(time.time() - t0, 1),
    }
    log.info("  RF            MAE=%.3f  RMSE=%.3f  (%.1fs)",
             results["random_forest"]["test_mae"], results["random_forest"]["test_rmse"],
             results["random_forest"]["train_sec"])

    # ── 4. XGBoost ─────────────────────────────────────────
    best_model = rf
    best_scaler = scaler
    best_model_name = "random_forest"

    if HAS_XGB:
        log.info("  Training XGBoost …")
        t0 = time.time()
        xgb_params = {k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"}
        xgb_model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=XGB_PARAMS["early_stopping_rounds"])
        xgb_model.fit(
            X_tr_s, y_tr,
            eval_set=[(X_va_s, y_va)],
            verbose=False,
        )
        xgb_preds = xgb_model.predict(X_te_s)
        xgb_val_preds = xgb_model.predict(X_va_s)
        results["xgboost"] = {
            "val_mae":   mae(y_va, xgb_val_preds),
            "val_rmse":  rmse(y_va, xgb_val_preds),
            "test_mae":  mae(y_te, xgb_preds),
            "test_rmse": rmse(y_te, xgb_preds),
            "train_sec": round(time.time() - t0, 1),
        }
        log.info("  XGBoost       MAE=%.3f  RMSE=%.3f  (%.1fs)",
                 results["xgboost"]["test_mae"], results["xgboost"]["test_rmse"],
                 results["xgboost"]["train_sec"])

        # pick best model by validation MAE
        if results["xgboost"]["val_mae"] < results["random_forest"]["val_mae"]:
            best_model = xgb_model
            best_model_name = "xgboost"

    # ── Save best model ─────────────────────────────────────
    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_path  = Path(MODELS_DIR) / f"best_model_{horizon_min}min_v{version}.joblib"
    scaler_path = Path(MODELS_DIR) / f"scaler_{horizon_min}min_v{version}.joblib"
    meta_path   = Path(MODELS_DIR) / f"model_meta_{horizon_min}min.json"

    joblib.dump(best_model, model_path)
    joblib.dump(best_scaler, scaler_path)

    meta = {
        "model_name": best_model_name,
        "version": version,
        "horizon_min": horizon_min,
        "sensors_trained_on": sensors,
        "n_train": len(train),
        "n_val":   len(val),
        "n_test":  len(test),
        "feature_cols": FEATURE_COLS,
        "model_path":  str(model_path),
        "scaler_path": str(scaler_path),
        "trained_at":  datetime.utcnow().isoformat(),
        "test_preds_sample": rf_preds[:10].tolist(),
        "test_actual_sample": y_te[:10].tolist(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info("  Best model: %s  |  saved to %s", best_model_name, model_path)
    return {
        "horizon_min": horizon_min,
        "best_model": best_model_name,
        "models": results,
        "meta": meta,
    }


# ─────────────────────────────────────────────────────────────────
def train_all(
    cleaned_csv: str | None = None,
    sensor_subset: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Full training pipeline for all horizons.
    Returns complete metrics dict saved to reports/model_metrics.json.
    """
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

    csv_path = cleaned_csv or str(CLEANED_DATA_PATH)
    log.info("Loading cleaned data from %s …", csv_path)
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df["sensor_id"] = df["sensor_id"].astype(str)

    # Select sensors to train on
    if sensor_subset:
        sensors = sensor_subset
    else:
        completeness = df.groupby("sensor_id")["observed_speed"].count().sort_values(ascending=False)
        sensors = completeness.head(TOP_N_SENSORS_FOR_TRAINING).index.tolist()
    log.info("Training on %d sensors: %s", len(sensors), sensors)

    all_results: Dict[str, Any] = {"sensors": sensors, "horizons": {}}

    for h in FORECAST_HORIZONS_MIN:
        horizon_result = train_for_horizon(df, h, sensors)
        if horizon_result:
            all_results["horizons"][str(h)] = horizon_result

    # Save metrics
    metrics_path = Path(REPORTS_DIR) / "model_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("Metrics saved to %s", metrics_path)

    return all_results


if __name__ == "__main__":
    results = train_all()
    print("\n── Model Evaluation Summary ─────────────────────────")
    for h, r in results["horizons"].items():
        print(f"\n  Horizon: {h} min  |  Best: {r['best_model']}")
        for model_name, metrics in r["models"].items():
            print(f"    {model_name:<20} MAE={metrics['test_mae']:.3f}  RMSE={metrics['test_rmse']:.3f}")
