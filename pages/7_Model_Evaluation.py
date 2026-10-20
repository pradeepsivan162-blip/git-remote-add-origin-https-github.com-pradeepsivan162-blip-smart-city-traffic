"""
pages/7_Model_Evaluation.py
────────────────────────────
Page 7: Model evaluation — MAE/RMSE by model and horizon,
actual vs predicted, residuals, model version info.
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
from src.analytics import load_cleaned_data
from config.settings import CLEANED_DATA_PATH, REPORTS_DIR, MODELS_DIR, FORECAST_HORIZONS_MIN, CHART_HEIGHT

st.set_page_config(page_title="Model Evaluation", page_icon="📊", layout="wide")
css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

st.markdown("## 📊 Model Evaluation")
st.markdown("*Rigorous comparison of all forecasting models across all horizons.*")

# ── Load metrics ────────────────────────────────────────────────────
metrics_path = Path(REPORTS_DIR) / "model_metrics.json"

if not metrics_path.exists():
    st.warning(
        "⚠️ No model metrics found. Run `python scripts/03_train_models.py` first."
    )
    st.stop()

with open(metrics_path) as f:
    metrics = json.load(f)

# ── Build comparison table ──────────────────────────────────────────
rows = []
for h_str, hr in metrics.get("horizons", {}).items():
    for model_name, m in hr.get("models", {}).items():
        rows.append({
            "Horizon (min)": int(h_str),
            "Model":         model_name.replace("_", " ").title(),
            "Val MAE":       round(m.get("val_mae",  float("nan")), 3),
            "Val RMSE":      round(m.get("val_rmse", float("nan")), 3),
            "Test MAE":      round(m.get("test_mae",  float("nan")), 3),
            "Test RMSE":     round(m.get("test_rmse", float("nan")), 3),
            "Best":          "✅" if hr.get("best_model") == model_name else "",
        })

results_df = pd.DataFrame(rows).sort_values(["Horizon (min)", "Test MAE"])

# ── Overview KPIs ───────────────────────────────────────────────────
best_rows = results_df[results_df["Best"] == "✅"]
if not best_rows.empty:
    col1, col2, col3 = st.columns(3)
    avg_mae  = best_rows["Test MAE"].mean()
    avg_rmse = best_rows["Test RMSE"].mean()
    best_mod = best_rows["Model"].mode()[0] if not best_rows.empty else "N/A"
    col1.metric("🏆 Best Model",  best_mod)
    col2.metric("📉 Avg Test MAE",  f"{avg_mae:.3f} mph")
    col3.metric("📉 Avg Test RMSE", f"{avg_rmse:.3f} mph")

st.divider()

# ── Full metrics table ──────────────────────────────────────────────
st.markdown("### 📋 All Model Results")
st.dataframe(results_df, use_container_width=True, hide_index=True)

st.divider()

# ── MAE by horizon grouped bar chart ───────────────────────────────
st.markdown("### 📉 Test MAE by Model and Horizon")
fig_mae = px.bar(
    results_df[results_df["Test MAE"].notna()],
    x="Horizon (min)", y="Test MAE",
    color="Model",
    barmode="group",
    color_discrete_sequence=["#00D4FF", "#00C851", "#FFD700", "#FF8C00"],
    template="plotly_dark",
    labels={"Test MAE": "Test MAE (mph)"},
)
fig_mae.update_layout(
    height=380,
    paper_bgcolor="#0D1B2A", plot_bgcolor="#0F2337",
    xaxis=dict(gridcolor="#1E3A5F", type="category"),
    yaxis=dict(gridcolor="#1E3A5F"),
    legend=dict(bgcolor="#0F2337"),
    margin=dict(l=0, r=0, t=30, b=0),
)
st.plotly_chart(fig_mae, use_container_width=True)

# ── RMSE by horizon ────────────────────────────────────────────────
st.markdown("### 📉 Test RMSE by Model and Horizon")
fig_rmse = px.bar(
    results_df[results_df["Test RMSE"].notna()],
    x="Horizon (min)", y="Test RMSE",
    color="Model",
    barmode="group",
    color_discrete_sequence=["#00D4FF", "#00C851", "#FFD700", "#FF8C00"],
    template="plotly_dark",
)
fig_rmse.update_layout(
    height=350,
    paper_bgcolor="#0D1B2A", plot_bgcolor="#0F2337",
    xaxis=dict(gridcolor="#1E3A5F", type="category"),
    yaxis=dict(gridcolor="#1E3A5F"),
    legend=dict(bgcolor="#0F2337"),
    margin=dict(l=0, r=0, t=30, b=0),
)
st.plotly_chart(fig_rmse, use_container_width=True)

st.divider()

# ── Actual vs Predicted per horizon (from saved meta) ──────────────
st.markdown("### 🔬 Actual vs Predicted — Sample (from saved meta)")
horizons_available = list(metrics.get("horizons", {}).keys())
sel_h = st.selectbox(
    "Select Horizon",
    options=horizons_available,
    format_func=lambda x: f"{x} minutes",
)

if sel_h:
    hr_data = metrics["horizons"][sel_h]
    meta    = hr_data.get("meta", {})
    actuals = meta.get("test_actual_sample", [])
    preds   = meta.get("test_preds_sample",  [])

    if actuals and preds:
        n = min(len(actuals), len(preds))
        fig_avp = go.Figure()
        fig_avp.add_trace(go.Scatter(
            x=list(range(n)), y=actuals[:n],
            name="Actual", line=dict(color="#00D4FF")))
        fig_avp.add_trace(go.Scatter(
            x=list(range(n)), y=preds[:n],
            name="Predicted", line=dict(color="#FF8C00", dash="dot")))
        fig_avp.update_layout(
            height=350, template="plotly_dark",
            paper_bgcolor="#0D1B2A", plot_bgcolor="#0F2337",
            xaxis=dict(title="Sample Index", gridcolor="#1E3A5F"),
            yaxis=dict(title="Speed (mph)", gridcolor="#1E3A5F"),
            legend=dict(bgcolor="#0F2337"),
            margin=dict(l=0, r=0, t=30, b=0),
            title=f"{hr_data.get('best_model','').replace('_',' ').title()} — {sel_h}-min horizon (first 10 test samples)",
        )
        st.plotly_chart(fig_avp, use_container_width=True)
        st.caption(f"Note: Only 10 sample predictions are stored in meta. Use Page 3 for full test-set evaluation.")
    else:
        st.info("Run training to populate actual vs predicted samples.")

# ── Model version info ─────────────────────────────────────────────
st.divider()
st.markdown("### 🏷️ Model Version Information")
model_files = sorted(Path(MODELS_DIR).glob("model_meta_*.json"))
if model_files:
    for mf in model_files:
        with open(mf) as f:
            m = json.load(f)
        st.markdown(
            f"**{m.get('model_name','?').replace('_',' ').title()}** "
            f"| Horizon: {m.get('horizon_min','?')} min "
            f"| Version: `{m.get('version','?')}` "
            f"| Trained: {m.get('trained_at','?')[:19]}"
        )
else:
    st.info("No model metadata found. Train models first.")
