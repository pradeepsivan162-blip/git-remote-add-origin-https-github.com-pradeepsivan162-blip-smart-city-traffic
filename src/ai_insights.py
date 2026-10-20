"""
src/ai_insights.py
───────────────────
Generates natural-language traffic insights from verified analytical results.

Priority order:
  1. Rule-based engine  (always available, no API key needed)
  2. Google Gemini API  (if GEMINI_API_KEY is set)
  3. OpenAI GPT API     (if OPENAI_API_KEY is set)

IMPORTANT: The LLM is NEVER given raw sensor data. It only receives
pre-computed, verified summaries (numbers, sensor IDs, dates) from
the analytics module. This prevents hallucinated observations.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    GEMINI_API_KEY, OPENAI_API_KEY, LLM_PROVIDER, LOG_FORMAT, LOG_LEVEL
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Rule-based engine (always runs)
# ─────────────────────────────────────────────────────────────────
def rule_based_insights(
    kpis: Dict[str, Any],
    peak_hours: pd.DataFrame,
    low_sensors: pd.DataFrame,
    weekday_weekend: pd.DataFrame,
    low_speed_events: pd.DataFrame | None = None,
    model_metrics: Dict[str, Any] | None = None,
) -> List[str]:
    """
    Generate a list of plain-English insight strings from verified analytics.
    Every statement is derived directly from the passed DataFrames and dicts.
    """
    insights = []

    # ── Dataset coverage ────────────────────────────────────────
    insights.append(
        f"📊 The dataset covers **{kpis['date_start']}** to **{kpis['date_end']}** "
        f"across **{kpis['total_sensors']} sensors** with **{kpis['total_observations']:,} observations**."
    )

    # ── Average speed ───────────────────────────────────────────
    avg = kpis["avg_speed_mph"]
    if avg >= 60:
        speed_desc = "generally free-flowing"
    elif avg >= 45:
        speed_desc = "moderately congested"
    elif avg >= 30:
        speed_desc = "frequently congested"
    else:
        speed_desc = "heavily congested"
    insights.append(
        f"🚗 The overall network average speed is **{avg} mph**, indicating traffic is **{speed_desc}**."
    )

    # ── Peak congestion hours ────────────────────────────────────
    if not peak_hours.empty:
        worst_hour = int(peak_hours.iloc[0]["hour"])
        worst_speed = round(peak_hours.iloc[0]["avg_speed_mph"], 1)
        h_str = f"{worst_hour:02d}:00–{worst_hour+1:02d}:00"
        insights.append(
            f"⏰ The most congested hour is **{h_str}** with an average speed of "
            f"**{worst_speed} mph** across all sensors."
        )

        top5_hours = ", ".join(f"{int(h):02d}:00" for h in peak_hours["hour"].head(5))
        insights.append(f"🔴 Top 5 congested hours: **{top5_hours}**")

    # ── Low-speed sensors ────────────────────────────────────────
    if not low_sensors.empty:
        worst_sensor = low_sensors.iloc[0]["sensor_id"]
        worst_avg    = round(low_sensors.iloc[0]["mean_speed"], 1)
        insights.append(
            f"⚠️  Sensor **{worst_sensor}** has the lowest historical average speed "
            f"of **{worst_avg} mph** — consistently congested."
        )
        bottom5 = ", ".join(low_sensors.head(5)["sensor_id"].astype(str).tolist())
        insights.append(f"🔴 Five lowest-speed sensors: **{bottom5}**")

    # ── Weekday vs weekend ───────────────────────────────────────
    if not weekday_weekend.empty:
        wkd_row = weekday_weekend[weekday_weekend["category"] == "Weekday"]
        wke_row = weekday_weekend[weekday_weekend["category"] == "Weekend"]
        if not wkd_row.empty and not wke_row.empty:
            wkd_avg = round(float(wkd_row["avg_speed_mph"].values[0]), 1)
            wke_avg = round(float(wke_row["avg_speed_mph"].values[0]), 1)
            diff    = round(wke_avg - wkd_avg, 1)
            direction = "faster" if diff > 0 else "slower"
            insights.append(
                f"📅 Weekend traffic averages **{wke_avg} mph**, which is **{abs(diff)} mph "
                f"{direction}** than weekday traffic ({wkd_avg} mph)."
            )

    # ── Low-speed events ─────────────────────────────────────────
    if low_speed_events is not None and not low_speed_events.empty:
        n_events = len(low_speed_events)
        longest  = low_speed_events.iloc[0]
        insights.append(
            f"🚨 **{n_events}** sustained low-speed events (< 30 mph for ≥ 15 min) detected. "
            f"Longest: **{longest['duration_min']} min** on sensor **{longest['sensor_id']}** "
            f"starting at **{longest['start_time']}**."
        )

    # ── Model performance ─────────────────────────────────────────
    if model_metrics and "horizons" in model_metrics:
        for h, hr in model_metrics["horizons"].items():
            best = hr.get("best_model", "unknown")
            models_data = hr.get("models", {})
            if best in models_data:
                m = models_data[best]
                insights.append(
                    f"🤖 For the **{h}-minute** forecast, the best model is **{best.replace('_', ' ').title()}** "
                    f"with MAE = **{m['test_mae']:.2f} mph** and RMSE = **{m['test_rmse']:.2f} mph**."
                )

    return insights


# ─────────────────────────────────────────────────────────────────
# Gemini wrapper
# ─────────────────────────────────────────────────────────────────
def _gemini_narrative(context_summary: str, bullet_insights: List[str]) -> str:
    """
    Convert pre-computed bullet insights into a polished paragraph using Gemini.
    The LLM receives ONLY the verified bullet points — never raw sensor data.
    """
    if not GEMINI_API_KEY:
        return ""
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")

        bullets = "\n".join(f"- {b}" for b in bullet_insights)
        prompt = (
            "You are a traffic analyst. Convert the following verified data points into "
            "a concise, professional 2-3 paragraph executive summary. "
            "Do NOT add any observations, predictions, or data not present below.\n\n"
            f"Verified findings:\n{bullets}\n\n"
            "Write the summary now:"
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        log.warning("Gemini API call failed: %s. Falling back to rule-based.", e)
        return ""


# ─────────────────────────────────────────────────────────────────
# OpenAI wrapper
# ─────────────────────────────────────────────────────────────────
def _openai_narrative(bullet_insights: List[str]) -> str:
    if not OPENAI_API_KEY:
        return ""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        bullets = "\n".join(f"- {b}" for b in bullet_insights)
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content":
                 "You are a traffic analyst. Summarise ONLY the provided facts. Do not invent data."},
                {"role": "user", "content":
                 f"Write a 2-3 paragraph executive summary based ONLY on these verified findings:\n\n{bullets}"},
            ],
            max_tokens=400,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log.warning("OpenAI API call failed: %s. Falling back to rule-based.", e)
        return ""


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────
def generate_insights(
    kpis: Dict[str, Any],
    peak_hours: pd.DataFrame,
    low_sensors: pd.DataFrame,
    weekday_weekend: pd.DataFrame,
    low_speed_events: pd.DataFrame | None = None,
    model_metrics: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Main entry point. Returns dict with:
      - 'bullets'   : list of rule-based insight strings (always present)
      - 'narrative' : LLM-generated paragraph (empty if no API key)
      - 'provider'  : which engine was used
    """
    bullets = rule_based_insights(
        kpis, peak_hours, low_sensors, weekday_weekend,
        low_speed_events, model_metrics
    )

    narrative = ""
    provider = "rule_based"

    if LLM_PROVIDER == "gemini":
        narrative = _gemini_narrative("", bullets)
        if narrative:
            provider = "gemini"
    elif LLM_PROVIDER == "openai":
        narrative = _openai_narrative(bullets)
        if narrative:
            provider = "openai"

    return {
        "bullets":   bullets,
        "narrative": narrative,
        "provider":  provider,
    }
