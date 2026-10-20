"""
src/feature_engineering.py
───────────────────────────
Creates ML features from long-format speed data.

All features are derived exclusively from PAST observations to prevent
target leakage. The target for each horizon is the speed N steps ahead.

Features produced per (sensor_id, timestamp):
  - lag_{n}        : speed n × 5 min in the past
  - roll_mean_{w}  : rolling mean over w × 5 min windows
  - roll_std_{w}   : rolling std  over w × 5 min windows
  - hour            : 0–23
  - day_of_week     : 0 Mon … 6 Sun
  - is_weekend      : 0 / 1
  - month           : 1–12
  - hour_sin/cos    : cyclic encoding of hour
  - dow_sin/cos     : cyclic encoding of day_of_week
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    LAG_INTERVALS, ROLLING_WINDOWS, FORECAST_HORIZONS_MIN,
    TRAIN_RATIO, VAL_RATIO, LOG_FORMAT, LOG_LEVEL,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
log = logging.getLogger(__name__)

INTERVAL_MIN = 5  # dataset sampling interval


# ─────────────────────────────────────────────────────────────────
def make_features_for_sensor(
    df: pd.DataFrame,
    sensor_id: str,
    horizon_min: int,
) -> pd.DataFrame:
    """
    Build a feature matrix for one sensor and one forecast horizon.
    Returns a DataFrame with feature columns + target column 'target_speed'.

    Parameters
    ----------
    df          : long-format DataFrame (timestamp, sensor_id, observed_speed)
    sensor_id   : the sensor to process
    horizon_min : forecast horizon in minutes (15, 30, or 60)
    """
    horizon_steps = horizon_min // INTERVAL_MIN  # number of 5-min steps ahead

    s = (
        df[df["sensor_id"] == sensor_id]
        .set_index("timestamp")["observed_speed"]
        .sort_index()
    )
    if len(s) < 100:
        log.debug("Sensor %s: too few rows (%d), skipping.", sensor_id, len(s))
        return pd.DataFrame()

    feat = pd.DataFrame(index=s.index)
    feat["observed_speed"] = s

    # ── Lag features ────────────────────────────────────────
    for n in LAG_INTERVALS:
        feat[f"lag_{n}"] = s.shift(n)

    # ── Rolling statistics ───────────────────────────────────
    for w in ROLLING_WINDOWS:
        feat[f"roll_mean_{w}"] = s.shift(1).rolling(w).mean()
        feat[f"roll_std_{w}"]  = s.shift(1).rolling(w).std()

    # ── Time features ────────────────────────────────────────
    idx = feat.index
    feat["hour"]        = idx.hour
    feat["day_of_week"] = idx.dayofweek
    feat["is_weekend"]  = (idx.dayofweek >= 5).astype(int)
    feat["month"]       = idx.month

    # Cyclic encoding
    feat["hour_sin"] = np.sin(2 * np.pi * feat["hour"] / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * feat["hour"] / 24)
    feat["dow_sin"]  = np.sin(2 * np.pi * feat["day_of_week"] / 7)
    feat["dow_cos"]  = np.cos(2 * np.pi * feat["day_of_week"] / 7)

    # ── Target: speed horizon_steps ahead ───────────────────
    feat["target_speed"] = s.shift(-horizon_steps)

    # Drop rows that have NaN in features or target
    feat = feat.dropna()
    feat["sensor_id"] = sensor_id
    feat = feat.reset_index()  # restore timestamp as column
    return feat


# ─────────────────────────────────────────────────────────────────
def chronological_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a feature DataFrame chronologically into train / val / test.
    No shuffling — respects time order to prevent leakage.
    """
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    train = df.iloc[:train_end].copy()
    val   = df.iloc[train_end:val_end].copy()
    test  = df.iloc[val_end:].copy()

    log.debug(
        "Split → train:%d  val:%d  test:%d", len(train), len(val), len(test)
    )
    return train, val, test


# ─────────────────────────────────────────────────────────────────
FEATURE_COLS = (
    [f"lag_{n}" for n in LAG_INTERVALS]
    + [f"roll_mean_{w}" for w in ROLLING_WINDOWS]
    + [f"roll_std_{w}"  for w in ROLLING_WINDOWS]
    + ["hour", "day_of_week", "is_weekend", "month",
       "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
)
TARGET_COL = "target_speed"


def get_X_y(df: pd.DataFrame):
    """Return (X, y) arrays from a feature DataFrame."""
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    return X, y
