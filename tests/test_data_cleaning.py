"""
tests/test_data_cleaning.py
────────────────────────────
Unit tests for data cleaning functions.
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from io import StringIO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_cleaning import validate_timestamps, validate_sensors, wide_to_long, flag_outliers


# ── Fixtures ───────────────────────────────────────────────────────
def make_sample_wide(n=20, n_sensors=5):
    timestamps = pd.date_range("2017-01-01", periods=n, freq="5T")
    data = {"timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S")}
    for i in range(n_sensors):
        sid = f"4000{i:02d}"
        data[sid] = np.random.uniform(20, 80, n)
    return pd.DataFrame(data)


# ── Timestamp tests ────────────────────────────────────────────────
def test_timestamp_parsing_success():
    df = make_sample_wide()
    result, report = validate_timestamps(df)
    assert "timestamp" in result.columns
    assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])
    assert report["ts_invalid_dropped"] == 0


def test_timestamp_invalid_rows_dropped():
    df = make_sample_wide(n=10)
    df.loc[3, df.columns[0]] = "NOT_A_DATE"
    result, report = validate_timestamps(df)
    assert report["ts_invalid_dropped"] == 1
    assert len(result) == 9


def test_timestamp_duplicates_removed():
    df = make_sample_wide(n=10)
    df.loc[5, df.columns[0]] = df.iloc[4][df.columns[0]]  # duplicate
    result, report = validate_timestamps(df)
    assert report["ts_duplicates_removed"] >= 1


def test_timestamp_sorted_ascending():
    df = make_sample_wide(n=10)
    df = df.iloc[::-1].reset_index(drop=True)  # reverse order
    result, _ = validate_timestamps(df)
    assert result["timestamp"].is_monotonic_increasing


# ── Sensor tests ───────────────────────────────────────────────────
def test_sensor_column_count():
    df = make_sample_wide(n=10, n_sensors=5)
    df, _ = validate_timestamps(df)
    result, report = validate_sensors(df)
    assert report["sensor_count"] == 5


def test_sensor_id_preserved():
    df = make_sample_wide(n=10, n_sensors=3)
    df, _ = validate_timestamps(df)
    result, report = validate_sensors(df)
    sensor_cols = [c for c in result.columns if c != "timestamp"]
    assert "400000" in sensor_cols


def test_non_numeric_coerced_to_nan():
    df = make_sample_wide(n=10, n_sensors=2)
    df.iloc[3, 1] = "bad_value"
    df, _ = validate_timestamps(df)
    result, report = validate_sensors(df)
    assert result.iloc[:, 1].isna().sum() >= 1
    assert report["missing_values"] >= 1


# ── Wide-to-long conversion tests ──────────────────────────────────
def test_wide_to_long_columns():
    df = make_sample_wide(n=10, n_sensors=3)
    df, _ = validate_timestamps(df)
    df, _ = validate_sensors(df)
    df, _, lower, upper = flag_outliers(df)
    long = wide_to_long(df, lower, upper)
    assert set(["timestamp", "sensor_id", "observed_speed", "outlier_flag"]).issubset(long.columns)


def test_wide_to_long_row_count():
    n_ts, n_sensors = 10, 3
    df = make_sample_wide(n=n_ts, n_sensors=n_sensors)
    df, _ = validate_timestamps(df)
    df, _ = validate_sensors(df)
    df, _, lower, upper = flag_outliers(df)
    long = wide_to_long(df, lower, upper)
    # rows = n_ts * n_sensors (no NaNs introduced)
    assert len(long) == n_ts * n_sensors


def test_no_duplicate_timestamp_sensor_pairs():
    df = make_sample_wide(n=15, n_sensors=4)
    df, _ = validate_timestamps(df)
    df, _ = validate_sensors(df)
    df, _, lower, upper = flag_outliers(df)
    long = wide_to_long(df, lower, upper)
    dups = long.duplicated(subset=["timestamp", "sensor_id"]).sum()
    assert dups == 0


def test_nan_speed_rows_dropped():
    df = make_sample_wide(n=10, n_sensors=2)
    df.iloc[0, 1] = np.nan
    df, _ = validate_timestamps(df)
    df, _ = validate_sensors(df)
    df, _, lower, upper = flag_outliers(df)
    long = wide_to_long(df, lower, upper)
    assert long["observed_speed"].isna().sum() == 0


# ── Outlier tests ───────────────────────────────────────────────────
def test_outlier_flag_column_binary():
    df = make_sample_wide(n=20, n_sensors=3)
    df, _ = validate_timestamps(df)
    df, _ = validate_sensors(df)
    df, _, lower, upper = flag_outliers(df)
    long = wide_to_long(df, lower, upper)
    assert long["outlier_flag"].isin([0, 1]).all()


def test_negative_speed_flagged():
    df = make_sample_wide(n=10, n_sensors=2)
    df.iloc[0, 1] = -5.0  # impossible speed
    df, _ = validate_timestamps(df)
    df, _ = validate_sensors(df)
    df, _, lower, upper = flag_outliers(df)
    long = wide_to_long(df, lower, upper)
    flagged = long[long["observed_speed"] < 0]["outlier_flag"]
    assert (flagged == 1).all()
