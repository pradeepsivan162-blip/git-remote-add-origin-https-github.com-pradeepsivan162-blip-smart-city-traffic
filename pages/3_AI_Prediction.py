"""
pages/3_AI_Prediction.py
─────────────────────────
Page 3: AI-based speed forecasting — select sensor + horizon,
display predicted speed, recent history, and actual vs predicted comparison.
"""

import json
import sys
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.analytics import load_cleaned_data, sensor_history
from src.congestion import get_color, get_label
from config.settings import CLEANED_DATA_PATH, MODELS_DIR, FORECAST_HORIZONS_MIN, CHART_HEIGHT

st.set_page_config(page_title="AI Prediction", page_icon="🤖", layout="wide")
css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

st.markdown("## 🤖 AI Speed Prediction")
st.markdown("*Forecast traffic speed 15, 30, or 60 minutes ahead using XGBoost/Random Forest.*")

@st.cache_data(ttl=3600)
def get_data():
    try:
        return load_cleaned_data()
    except Exception as e:
        st.error(f"Data not found: {e}.")
        st.stop()

df = get_data()
all_sensors = sorted(df["sensor_id"].unique().tolist())

def models_trained() -> bool:
    return any(Path(MODELS_DIR).glob("best_model_*.joblib"))

# ── Sidebar controls ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Prediction Settings")
    sensor_id = st.selectbox("Select Sensor", all_sensors, index=0)
    horizon_min = st.radio(
        "Forecast Horizon",
        options=FORECAST_HORIZONS_MIN,
        format_func=lambda x: f"{x} minutes ahead",
        index=0,
    )

# ── Check model availability ────────────────────────────────────────
if not models_trained():
    st.warning(
        "⚠️ No trained models found. Run `python scripts/03_train_models.py` to train models first.\n\n"
        "The prediction page will show actual vs persistence-baseline comparison in the meantime."
    )
    # Show persistence baseline as demo
    hist = sensor_history(df, sensor_id, resample=None)
    if not hist.empty:
        hist = hist.tail(200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist["timestamp"], y=hist["observed_speed"],
            name="Actual Speed", line=dict(color="#00D4FF")))
        fig.add_trace(go.Scatter(
            x=hist["timestamp"], y=hist["observed_speed"].shift(horizon_min // 5),
            name="Persistence Baseline", line=dict(color="#FFD700", dash="dash")))
        fig.update_layout(
            height=400, template="plotly_dark",
            paper_bgcolor="#0D1B2A", plot_bgcolor="#0F2337",
            title=f"Sensor {sensor_id} — Persistence Baseline ({horizon_min} min)",
        )
        st.plotly_chart(fig, use_container_width=True)
    st.stop()

# ── Generate prediction ─────────────────────────────────────────────
from src.predict import predict_for_sensor, evaluate_on_test

with st.spinner("Generating forecast …"):
    pred = predict_for_sensor(sensor_id, horizon_min, df)

if "error" in pred:
    st.error(f"Prediction failed: {pred['error']}")
    st.stop()

# ── Forecast display ────────────────────────────────────────────────
level = pred["congestion_level"]
color = get_color(level)
label = get_label(level)
speed = pred["predicted_speed_mph"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("🎯 Predicted Speed", f"{speed:.1f} mph")
col2.metric("🚦 Congestion Level", label)
col3.metric("⏰ Horizon", f"{horizon_min} min")
col4.metric("📡 Sensor", sensor_id)

# Colored congestion banner
st.markdown(
    f"""<div style='background:{color}22; border:1px solid {color}; border-radius:8px;
        padding:1rem; text-align:center; margin:1rem 0;'>
        <span style='font-size:1.4rem; color:{color}; font-weight:700;'>
            {label} — Predicted Speed: {speed:.1f} mph in {horizon_min} minutes
        </span></div>""",
    unsafe_allow_html=True,
)

st.divider()

# ── Actual vs Predicted chart ────────────────────────────────────────
st.markdown("### 📊 Actual vs Predicted (Test Set)")
with st.spinner("Evaluating on test data …"):
    eval_result = evaluate_on_test(sensor_id, horizon_min, df)

if "error" in eval_result:
    st.warning(f"Evaluation not available: {eval_result['error']}")
else:
    mae_val  = eval_result["mae"]
    rmse_val = eval_result["rmse"]
    model    = eval_result["model_name"]
    version  = eval_result["model_version"]

    m1, m2, m3 = st.columns(3)
    m1.metric("📉 MAE",  f"{mae_val:.3f} mph")
    m2.metric("📉 RMSE", f"{rmse_val:.3f} mph")
    m3.metric("🤖 Model", model.replace("_", " ").title())

    actual    = eval_result["actual"][:500]
    predicted = eval_result["predicted"][:500]
    timestamps = eval_result["timestamps"][:500]

    fig_avp = go.Figure()
    fig_avp.add_trace(go.Scatter(
        x=timestamps, y=actual,
        name="Actual", line=dict(color="#00D4FF", width=1.5)))
    fig_avp.add_trace(go.Scatter(
        x=timestamps, y=predicted,
        name="Predicted", line=dict(color="#FF8C00", width=1.5, dash="dot")))
    fig_avp.update_layout(
        height=CHART_HEIGHT,
        template="plotly_dark",
        paper_bgcolor="#0D1B2A", plot_bgcolor="#0F2337",
        xaxis=dict(gridcolor="#1E3A5F", title="Timestamp"),
        yaxis=dict(gridcolor="#1E3A5F", title="Speed (mph)"),
        legend=dict(bgcolor="#0F2337"),
        margin=dict(l=0, r=0, t=30, b=0),
        hovermode="x unified",
        title=f"Sensor {sensor_id} — {horizon_min}-min Forecast | {model} | v{version}"
    )
    st.plotly_chart(fig_avp, use_container_width=True)

    # ── Residual plot ──────────────────────────────────────────────
    st.markdown("### 📉 Residual Analysis")
    residuals = np.array(actual) - np.array(predicted)
    fig_res = go.Figure()
    fig_res.add_trace(go.Histogram(
        x=residuals, nbinsx=50,
        marker_color="#00D4FF", opacity=0.75,
        name="Residuals"))
    fig_res.add_vline(x=0, line_color="#FF1744", line_dash="dash")
    fig_res.update_layout(
        height=300,
        template="plotly_dark",
        paper_bgcolor="#0D1B2A", plot_bgcolor="#0F2337",
        xaxis_title="Residual (mph)", yaxis_title="Count",
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_res, use_container_width=True)
    st.caption(f"Model: {model} | Version: {version} | Horizon: {horizon_min} min")
