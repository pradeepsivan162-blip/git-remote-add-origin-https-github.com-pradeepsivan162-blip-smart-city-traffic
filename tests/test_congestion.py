"""
tests/test_congestion.py
─────────────────────────
Unit tests for the congestion classification module.
"""

import sys
import pytest
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.congestion import classify_speed, classify_series, get_label, get_color


# ── Boundary tests ─────────────────────────────────────────────────
def test_free_flow_boundary():
    assert classify_speed(60.0) == "free_flow"
    assert classify_speed(100.0) == "free_flow"
    assert classify_speed(59.9) == "moderate"


def test_moderate_boundary():
    assert classify_speed(45.0) == "moderate"
    assert classify_speed(44.9) == "heavy"


def test_heavy_boundary():
    assert classify_speed(30.0) == "heavy"
    assert classify_speed(29.9) == "severe"


def test_severe():
    assert classify_speed(0.0) == "severe"
    assert classify_speed(10.0) == "severe"
    assert classify_speed(29.0) == "severe"


def test_zero_speed_severe():
    assert classify_speed(0.0) == "severe"


def test_very_high_speed_free_flow():
    assert classify_speed(120.0) == "free_flow"


# ── Series tests ───────────────────────────────────────────────────
def test_classify_series_returns_series():
    s = pd.Series([10, 35, 50, 70])
    result = classify_series(s)
    assert isinstance(result, pd.Series)
    assert len(result) == 4


def test_classify_series_correct_values():
    s = pd.Series([10.0, 35.0, 50.0, 70.0])
    result = classify_series(s).tolist()
    assert result == ["severe", "heavy", "moderate", "free_flow"]


# ── Label / Color tests ─────────────────────────────────────────────
def test_labels_not_empty():
    for level in ["free_flow", "moderate", "heavy", "severe"]:
        assert len(get_label(level)) > 0


def test_colors_hex_format():
    for level in ["free_flow", "moderate", "heavy", "severe"]:
        color = get_color(level)
        assert color.startswith("#")
        assert len(color) == 7


def test_unknown_level_graceful():
    assert get_label("unknown_level") == "Unknown"
