"""
src/analytics.py
─────────────────
Pure-Python analytics functions that operate on the long-format DataFrame.
Used by dashboard pages to avoid direct SQL dependencies in demo mode.
All functions return pandas DataFrames or scalar values.
"""

import logging
import sys
from pathlib import Path
from functools import lru_cache
from typing import Optional, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import LOG_FORMAT, LOG_LEVEL, CLEANED_DATA_PATH

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_cleaned_data(path: str | None = None) -> pd.DataFrame:
    """Load and cache the cleaned long-format CSV, falling back to demo data if absent."""
    from config.settings import SAMPLE_DATA_PATH
    src = path or str(CLEANED_DATA_PATH)
    if not Path(src).exists():
        if Path(SAMPLE_DATA_PATH).exists():
            log.info("Full dataset not found at %s. Falling back to sample dataset: %s", src, SAMPLE_DATA_PATH)
            src = str(SAMPLE_DATA_PATH)
        else:
            raise FileNotFoundError(f"Neither full dataset ({CLEANED_DATA_PATH}) nor sample ({SAMPLE_DATA_PATH}) found.")
    
    log.info("Loading cleaned data from %s …", src)
    df = pd.read_csv(src, parse_dates=["timestamp"])
    df["sensor_id"] = df["sensor_id"].astype(str)
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["date"]        = df["timestamp"].dt.date
    df["month"]       = df["timestamp"].dt.month
    log.info("Loaded %d rows, %d sensors.", len(df), df["sensor_id"].nunique())
    return df


# ─────────────────────────────────────────────────────────────────
# Overview KPIs
# ─────────────────────────────────────────────────────────────────
def overview_kpis(df: pd.DataFrame) -> dict:
    """Return top-level KPI values for the Executive Overview page."""
    return {
        "total_sensors":      int(df["sensor_id"].nunique()),
        "total_observations": int(len(df)),
        "avg_speed_mph":      round(float(df["observed_speed"].mean()), 2),
        "min_speed_mph":      round(float(df["observed_speed"].min()),  2),
        "max_speed_mph":      round(float(df["observed_speed"].max()),  2),
        "date_start":         str(df["timestamp"].min().date()),
        "date_end":           str(df["timestamp"].max().date()),
        "outlier_pct":        round(float(df["outlier_flag"].mean() * 100), 2),
    }


# ─────────────────────────────────────────────────────────────────
# Sensor stats
# ─────────────────────────────────────────────────────────────────
def sensor_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-sensor descriptive statistics."""
    stats = (
        df.groupby("sensor_id")["observed_speed"]
        .agg(
            mean_speed="mean",
            min_speed="min",
            max_speed="max",
            std_speed="std",
            count="count",
        )
        .reset_index()
        .sort_values("mean_speed")
    )
    stats["mean_speed"] = stats["mean_speed"].round(2)
    stats["std_speed"]  = stats["std_speed"].round(2)
    return stats


def lowest_speed_sensors(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the N sensors with the lowest average observed speed."""
    return sensor_stats(df).head(n)


# ─────────────────────────────────────────────────────────────────
# Temporal patterns
# ─────────────────────────────────────────────────────────────────
def hourly_avg_speed(df: pd.DataFrame, sensor_id: str | None = None) -> pd.DataFrame:
    """Average speed by hour of day (optionally filtered to one sensor)."""
    subset = df if sensor_id is None else df[df["sensor_id"] == sensor_id]
    return (
        subset.groupby("hour")["observed_speed"]
        .mean()
        .reset_index()
        .rename(columns={"observed_speed": "avg_speed_mph"})
    )


def daily_avg_speed(df: pd.DataFrame, sensor_id: str | None = None) -> pd.DataFrame:
    """Average speed by day of week."""
    dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    subset = df if sensor_id is None else df[df["sensor_id"] == sensor_id]
    agg = (
        subset.groupby("day_of_week")["observed_speed"]
        .mean()
        .reset_index()
        .rename(columns={"observed_speed": "avg_speed_mph"})
    )
    agg["day_name"] = agg["day_of_week"].map(dow_map)
    return agg


def weekday_vs_weekend(df: pd.DataFrame) -> pd.DataFrame:
    """Average speed split by weekday vs weekend."""
    return (
        df.groupby("is_weekend")["observed_speed"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .assign(category=lambda x: x["is_weekend"].map({0: "Weekday", 1: "Weekend"}))
        .rename(columns={"mean": "avg_speed_mph", "std": "std_speed"})
    )


def hourly_heatmap(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Pivot table of average speed by (sensor_id, hour) for a heatmap.
    Limited to top_n sensors by data count for readability.
    """
    top_sensors = (
        df.groupby("sensor_id")["observed_speed"].count()
        .nlargest(top_n).index.tolist()
    )
    subset = df[df["sensor_id"].isin(top_sensors)]
    pivot = subset.pivot_table(
        index="sensor_id", columns="hour",
        values="observed_speed", aggfunc="mean"
    ).round(2)
    return pivot


# ─────────────────────────────────────────────────────────────────
# Sensor time-series
# ─────────────────────────────────────────────────────────────────
def sensor_history(
    df: pd.DataFrame,
    sensor_id: str,
    start_date=None,
    end_date=None,
    resample: str | None = "15T",
) -> pd.DataFrame:
    """
    Return speed history for a sensor between start/end dates.
    Optionally resampled (default: 15-min averages for chart performance).
    """
    subset = df[df["sensor_id"] == sensor_id].copy()
    if start_date:
        subset = subset[subset["timestamp"] >= pd.Timestamp(start_date)]
    if end_date:
        subset = subset[subset["timestamp"] <= pd.Timestamp(end_date)]
    subset = subset.set_index("timestamp").sort_index()
    if resample and len(subset) > 5000:
        subset = subset["observed_speed"].resample(resample).mean().reset_index()
        subset.columns = ["timestamp", "observed_speed"]
    else:
        subset = subset[["observed_speed"]].reset_index()
    return subset


# ─────────────────────────────────────────────────────────────────
# Peak periods
# ─────────────────────────────────────────────────────────────────
def peak_periods(df: pd.DataFrame) -> pd.DataFrame:
    """Identify top 5 hours with lowest average speed (= most congested)."""
    hourly = hourly_avg_speed(df)
    return hourly.sort_values("avg_speed_mph").head(5)


def low_speed_events(
    df: pd.DataFrame,
    threshold_mph: float = 30.0,
    min_duration_min: int = 15,
) -> pd.DataFrame:
    """
    Find (sensor, start_time, duration) events where speed drops below threshold
    for at least min_duration_min consecutive minutes.
    """
    records = []
    interval = 5  # minutes

    for sid, grp in df.groupby("sensor_id"):
        grp = grp.sort_values("timestamp")
        low_mask = grp["observed_speed"] < threshold_mph
        # find consecutive runs
        low_grp = (low_mask != low_mask.shift()).cumsum()
        for run_id, run in grp.groupby(low_grp):
            if not run["observed_speed"].iloc[0] < threshold_mph:
                continue
            duration = len(run) * interval
            if duration >= min_duration_min:
                records.append({
                    "sensor_id":  sid,
                    "start_time": run["timestamp"].iloc[0],
                    "end_time":   run["timestamp"].iloc[-1],
                    "duration_min": duration,
                    "min_speed_mph": round(run["observed_speed"].min(), 2),
                    "avg_speed_mph": round(run["observed_speed"].mean(), 2),
                })
    return pd.DataFrame(records).sort_values("duration_min", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────
# Congestion summary
# ─────────────────────────────────────────────────────────────────
def congestion_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Count observations by congestion category across all sensors."""
    from src.congestion import classify_series
    df = df.copy()
    df["congestion_level"] = classify_series(df["observed_speed"])
    return (
        df["congestion_level"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "level", "congestion_level": "count"})
    )


def sensor_congestion_profile(df: pd.DataFrame) -> pd.DataFrame:
    """For each sensor, compute % of time in each congestion category."""
    from src.congestion import classify_series
    df = df.copy()
    df["congestion_level"] = classify_series(df["observed_speed"])
    pivot = (
        df.groupby(["sensor_id", "congestion_level"])
        .size()
        .unstack(fill_value=0)
    )
    pivot = pivot.div(pivot.sum(axis=1), axis=0).round(4) * 100
    return pivot.reset_index()
