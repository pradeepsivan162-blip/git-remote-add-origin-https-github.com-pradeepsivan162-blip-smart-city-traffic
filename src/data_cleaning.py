"""
src/data_cleaning.py
────────────────────
Loads PEMS-BAY raw CSV, validates, cleans, and converts to long format.

Dataset: PEMS-BAY (Caltrans Performance Measurement System, Bay Area)
  - ~52,116 timestamps × 325 sensors
  - 5-minute intervals, Jan–Jun 2017
  - Values represent observed traffic speed in mph
  - Source: https://zenodo.org/record/5146275  (Creative Commons)
  - Limitation: speed only, no GPS coords, no volume in this file

Transformations applied (all documented in returned metadata dict):
  1. Timestamp parsing and deduplication
  2. Sensor column numeric coercion
  3. IQR-based outlier flagging  (values not dropped, only flagged)
  4. Wide → long format melt
  5. Drop rows where observed_speed is NaN after melt
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

# ── allow imports from project root ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import RAW_DATA_PATH, CLEANED_DATA_PATH, LOG_FORMAT, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
def load_raw(path: str | None = None) -> pd.DataFrame:
    """Load the wide-format PEMS-BAY CSV."""
    src = path or RAW_DATA_PATH
    log.info("Loading raw dataset from: %s", src)
    df = pd.read_csv(src, low_memory=False)
    log.info("Raw shape: %s rows × %s cols", *df.shape)
    return df


# ─────────────────────────────────────────────────────────────────
def validate_timestamps(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Parse and validate the timestamp column. Returns df + report."""
    report: Dict[str, Any] = {}

    ts_col = df.columns[0]  # first column is always the timestamp
    log.info("Parsing timestamps from column: '%s'", ts_col)

    original_count = len(df)
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

    n_invalid = df[ts_col].isna().sum()
    if n_invalid:
        log.warning("Dropping %d rows with un-parseable timestamps.", n_invalid)
        df = df.dropna(subset=[ts_col])

    # remove duplicates on timestamp
    n_dups = df.duplicated(subset=[ts_col]).sum()
    if n_dups:
        log.warning("Removing %d duplicate timestamps (keeping first).", n_dups)
        df = df.drop_duplicates(subset=[ts_col], keep="first")

    df = df.rename(columns={ts_col: "timestamp"})
    df = df.sort_values("timestamp").reset_index(drop=True)

    report["timestamp_range_start"] = str(df["timestamp"].min())
    report["timestamp_range_end"]   = str(df["timestamp"].max())
    report["rows_after_ts_clean"]   = len(df)
    report["ts_invalid_dropped"]    = n_invalid
    report["ts_duplicates_removed"] = n_dups
    report["sampling_interval_min"] = (
        df["timestamp"].diff().dropna().mode()[0].seconds // 60
    )
    log.info(
        "Timestamp range: %s → %s  |  interval: %s min",
        report["timestamp_range_start"],
        report["timestamp_range_end"],
        report["sampling_interval_min"],
    )
    return df, report


# ─────────────────────────────────────────────────────────────────
def validate_sensors(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Coerce sensor columns to numeric; report missing-value stats."""
    report: Dict[str, Any] = {}
    sensor_cols = [c for c in df.columns if c != "timestamp"]
    log.info("Validating %d sensor columns …", len(sensor_cols))

    df[sensor_cols] = df[sensor_cols].apply(pd.to_numeric, errors="coerce")

    missing = df[sensor_cols].isna()
    report["total_sensor_cells"]    = missing.size
    report["missing_values"]        = int(missing.sum().sum())
    report["missing_pct"]           = round(missing.mean().mean() * 100, 2)
    report["sensors_with_any_missing"] = int((missing.sum() > 0).sum())
    report["sensor_count"]          = len(sensor_cols)

    log.info(
        "Missing values: %s / %s  (%.2f%%)",
        report["missing_values"],
        report["total_sensor_cells"],
        report["missing_pct"],
    )
    return df, report


# ─────────────────────────────────────────────────────────────────
def flag_outliers(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    IQR-based outlier flagging across all speed observations.
    Values are NOT dropped — they are preserved with an 'outlier_flag' column
    added in the final long-format output.
    Speed ≤ 0 or > 200 mph are also considered anomalous.
    """
    report: Dict[str, Any] = {}
    sensor_cols = [c for c in df.columns if c != "timestamp"]

    speeds = df[sensor_cols].values.flatten()
    speeds = speeds[~np.isnan(speeds)]

    q1, q3 = np.percentile(speeds, [25, 75])
    iqr = q3 - q1
    lower = q1 - 3.0 * iqr
    upper = q3 + 3.0 * iqr
    # also cap at physical limits
    lower = max(lower, 0.0)
    upper = min(upper, 200.0)

    report["iqr_lower_bound"] = round(lower, 2)
    report["iqr_upper_bound"] = round(upper, 2)

    outlier_mask = (df[sensor_cols] < lower) | (df[sensor_cols] > upper)
    report["outlier_cells_flagged"] = int(outlier_mask.sum().sum())
    log.info(
        "Outlier flag bounds: [%.2f, %.2f]  |  %d cells flagged",
        lower, upper, report["outlier_cells_flagged"],
    )
    return df, report, lower, upper


# ─────────────────────────────────────────────────────────────────
def wide_to_long(
    df: pd.DataFrame,
    outlier_lower: float,
    outlier_upper: float,
) -> pd.DataFrame:
    """Melt wide → long; add outlier_flag column."""
    log.info("Melting wide → long format …")
    sensor_cols = [c for c in df.columns if c != "timestamp"]

    long_df = df.melt(
        id_vars=["timestamp"],
        value_vars=sensor_cols,
        var_name="sensor_id",
        value_name="observed_speed",
    )
    # drop rows with no speed reading
    before = len(long_df)
    long_df = long_df.dropna(subset=["observed_speed"])
    log.info("Dropped %d NaN speed rows; remaining: %d", before - len(long_df), len(long_df))

    long_df["outlier_flag"] = (
        (long_df["observed_speed"] < outlier_lower) |
        (long_df["observed_speed"] > outlier_upper)
    ).astype(int)

    long_df = long_df.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)
    log.info("Long-format shape: %s", long_df.shape)
    return long_df


# ─────────────────────────────────────────────────────────────────
def clean(
    raw_path: str | None = None,
    output_path: str | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Full cleaning pipeline. Returns (cleaned_long_df, metadata_report).
    """
    out_path = Path(output_path or CLEANED_DATA_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_raw(raw_path)

    df, ts_report  = validate_timestamps(df)
    df, sen_report = validate_sensors(df)
    df, out_report, lower, upper = flag_outliers(df)
    long_df = wide_to_long(df, lower, upper)

    log.info("Saving cleaned data → %s", out_path)
    long_df.to_csv(out_path, index=False)

    metadata = {
        "source_file": str(raw_path or RAW_DATA_PATH),
        "output_file": str(out_path),
        "dataset": "PEMS-BAY (California Bay Area, Jan–Jun 2017)",
        "units": "mph (miles per hour)",
        "license": "Creative Commons Attribution 4.0 International",
        **ts_report,
        **sen_report,
        **out_report,
        "final_long_rows": len(long_df),
        "final_sensors": long_df["sensor_id"].nunique(),
    }
    log.info("Cleaning complete. Final dataset: %d rows, %d sensors.",
             metadata["final_long_rows"], metadata["final_sensors"])
    return long_df, metadata


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df, meta = clean()
    print("\n── Cleaning Report ──────────────────────────────────")
    for k, v in meta.items():
        print(f"  {k:<35} {v}")
