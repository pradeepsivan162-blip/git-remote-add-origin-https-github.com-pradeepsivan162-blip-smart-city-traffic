"""
src/congestion.py
─────────────────
Congestion classification based on predicted or observed speed (mph).

Thresholds are demonstration rules derived from general US freeway norms.
They are NOT calibrated to specific road speed limits in the PEMS-BAY dataset.
Adjust CONGESTION_THRESHOLDS in config/settings.py if verified local data
is available.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    CONGESTION_THRESHOLDS, CONGESTION_LABELS, CONGESTION_COLORS
)


def classify_speed(speed_mph: float) -> str:
    """
    Return a congestion level key for a given speed (mph).

    Returns one of: 'free_flow', 'moderate', 'heavy', 'severe'
    """
    if speed_mph >= CONGESTION_THRESHOLDS["free_flow"]:
        return "free_flow"
    elif speed_mph >= CONGESTION_THRESHOLDS["moderate"]:
        return "moderate"
    elif speed_mph >= CONGESTION_THRESHOLDS["heavy"]:
        return "heavy"
    else:
        return "severe"


def classify_series(series: pd.Series) -> pd.Series:
    """Vectorised congestion classification for a pandas Series of speeds."""
    return series.apply(classify_speed)


def get_label(level_key: str) -> str:
    """Return the emoji-decorated display label for a congestion level key."""
    return CONGESTION_LABELS.get(level_key, "Unknown")


def get_color(level_key: str) -> str:
    """Return hex color string for a congestion level key."""
    return CONGESTION_COLORS.get(level_key, "#AAAAAA")


def label_dataframe(df: pd.DataFrame, speed_col: str = "predicted_speed_kmph") -> pd.DataFrame:
    """
    Add 'congestion_level', 'congestion_label', and 'congestion_color' columns
    to a DataFrame that has a speed column.
    """
    df = df.copy()
    df["congestion_level"] = classify_series(df[speed_col])
    df["congestion_label"] = df["congestion_level"].map(CONGESTION_LABELS)
    df["congestion_color"] = df["congestion_level"].map(CONGESTION_COLORS)
    return df


def threshold_summary() -> str:
    """Return a human-readable string describing current thresholds."""
    t = CONGESTION_THRESHOLDS
    return (
        f"🟢 Free Flow  : ≥ {t['free_flow']} mph\n"
        f"🟡 Moderate   : {t['moderate']}–{t['free_flow']-1} mph\n"
        f"🟠 Heavy      : {t['heavy']}–{t['moderate']-1} mph\n"
        f"🔴 Severe     : < {t['heavy']} mph\n"
        f"\n⚠️  Demonstration thresholds — not calibrated to verified speed limits."
    )
