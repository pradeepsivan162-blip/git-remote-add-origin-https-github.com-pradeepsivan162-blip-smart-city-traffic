"""
tests/test_prediction.py
─────────────────────────
Unit tests for feature engineering and prediction pipeline.
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.feature_engineering import (
    make_features_for_sensor, chronological_split, get_X_y, FEATURE_COLS
)


# ── Fixtures ───────────────────────────────────────────────────────
def make_long_df(n=200, sensor_id="400001"):
    timestamps = pd.date_range("2017-01-01", periods=n, freq="5T")
    return pd.DataFrame({
        "timestamp":      timestamps,
        "sensor_id":      sensor_id,
        "observed_speed": np.random.uniform(20, 80, n),
        "outlier_flag":   0,
    })


# ── Feature generation tests ────────────────────────────────────────
def test_feature_columns_present():
    df = make_long_df(n=200)
    feat = make_features_for_sensor(df, "400001", horizon_min=15)
    assert not feat.empty
    for col in FEATURE_COLS:
        assert col in feat.columns, f"Missing feature column: {col}"


def test_target_column_present():
    df = make_long_df(n=200)
    feat = make_features_for_sensor(df, "400001", horizon_min=15)
    assert "target_speed" in feat.columns


def test_no_nan_in_features():
    df = make_long_df(n=300)
    feat = make_features_for_sensor(df, "400001", horizon_min=15)
    assert feat[FEATURE_COLS].isna().sum().sum() == 0


def test_no_nan_in_target():
    df = make_long_df(n=300)
    feat = make_features_for_sensor(df, "400001", horizon_min=15)
    assert feat["target_speed"].isna().sum() == 0


def test_different_horizons_different_targets():
    df = make_long_df(n=300)
    feat_15 = make_features_for_sensor(df, "400001", horizon_min=15)
    feat_60 = make_features_for_sensor(df, "400001", horizon_min=60)
    # 60-min horizon shifts target 12 steps vs 3 steps — targets should differ
    common_idx = feat_15.index.intersection(feat_60.index)
    if len(common_idx) > 10:
        assert not (feat_15.loc[common_idx, "target_speed"].values
                    == feat_60.loc[common_idx, "target_speed"].values).all()


def test_no_future_leakage():
    """
    Ensure lag features use only past values (shift > 0).
    Feature at time t should not contain the speed at time t.
    """
    df = make_long_df(n=200)
    feat = make_features_for_sensor(df, "400001", horizon_min=15)
    # lag_1 at row i should equal observed_speed at row i-1
    # We verify this by checking the lag_1 is not the same as observed_speed
    # (they would be equal if no shift was applied)
    corr = feat["lag_1"].corr(feat["observed_speed"])
    # High but not 1.0 (because it's a shift)
    assert corr < 1.0


def test_feature_output_shape():
    df = make_long_df(n=200)
    feat = make_features_for_sensor(df, "400001", horizon_min=15)
    X, y = get_X_y(feat)
    assert X.shape[1] == len(FEATURE_COLS)
    assert len(y) == len(feat)


# ── Train/val/test split tests ─────────────────────────────────────
def test_split_sizes():
    df = make_long_df(n=300)
    feat = make_features_for_sensor(df, "400001", horizon_min=15)
    train, val, test = chronological_split(feat)
    total = len(train) + len(val) + len(test)
    assert total == len(feat)


def test_split_chronological_order():
    df = make_long_df(n=300)
    feat = make_features_for_sensor(df, "400001", horizon_min=15)
    train, val, test = chronological_split(feat)
    assert train["timestamp"].max() <= val["timestamp"].min()
    assert val["timestamp"].max()   <= test["timestamp"].min()


def test_empty_sensor_returns_empty_df():
    df = make_long_df(n=5, sensor_id="400001")  # too few rows
    feat = make_features_for_sensor(df, "400001", horizon_min=15)
    assert feat.empty


def test_missing_sensor_returns_empty_df():
    df = make_long_df(n=200, sensor_id="400001")
    feat = make_features_for_sensor(df, "999999", horizon_min=15)
    assert feat.empty


# ── Missing model file test ─────────────────────────────────────────
def test_missing_model_raises():
    from src.predict import load_model
    with pytest.raises(FileNotFoundError):
        load_model(horizon_min=99)  # 99 min horizon never exists
