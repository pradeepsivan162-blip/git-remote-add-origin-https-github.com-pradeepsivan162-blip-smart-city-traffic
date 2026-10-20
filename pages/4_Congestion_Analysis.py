"""
pages/4_Congestion_Analysis.py
───────────────────────────────
Page 4: Congestion analysis — rankings, category distributions,
low-speed events, hourly/daily patterns.
"""

import sys
from pathlib import Path
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.analytics import (
    load_cleaned_data, congestion_distribution,
    sensor_congestion_profile, low_speed_events, hourly_avg_speed
)
from src.congestion import threshold_summary, CONGESTION_COLORS, CONGESTION_LABELS
from config.settings import CLEANED_DATA_PATH, CHART_HEIGHT

st.set_page_config(page_title="Congestion Analysis", page_icon="🚨", layout="wide")
css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

st.markdown("## 🚨 Congestion Analysis")
st.markdown("*Identify congestion hotspots, patterns, and sustained low-speed events.*")

@st.cache_data(ttl=3600)
def get_data():
    try:
        return load_cleaned_data()
    except Exception as e:
        st.error(f"Data not found: {e}.")
        st.stop()

df = get_data()

# ── Threshold info ─────────────────────────────────────────────────
with st.expander("ℹ️ Congestion Thresholds (click to expand)"):
    st.code(threshold_summary())

# ── Row 1: Donut chart + Hourly congestion pattern ─────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 🍩 Congestion Level Distribution")
    dist = congestion_distribution(df)
    # ensure consistent ordering
    level_order = ["free_flow", "moderate", "heavy", "severe"]
    dist["level"] = pd.Categorical(dist["congestion_level"], categories=level_order, ordered=True)
    dist = dist.sort_values("level")

    fig_donut = px.pie(
        dist,
        names=dist["congestion_level"].map(CONGESTION_LABELS),
        values="count",
        hole=0.55,
        color=dist["congestion_level"],
        color_discrete_map={v: CONGESTION_COLORS[k] for k, v in CONGESTION_LABELS.items()},
        template="plotly_dark",
    )
    fig_donut.update_traces(textposition="inside", textinfo="percent+label")
    fig_donut.update_layout(
        height=360,
        paper_bgcolor="#0D1B2A",
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with col2:
    st.markdown("### ⏰ Hourly Congestion Heatmap")
    hourly = hourly_avg_speed(df)
    hourly["congestion"] = hourly["avg_speed_mph"].apply(
        lambda s: "Severe" if s < 30 else ("Heavy" if s < 45 else ("Moderate" if s < 60 else "Free Flow"))
    )
    color_map = {"Free Flow": "#00C851", "Moderate": "#FFD700", "Heavy": "#FF8C00", "Severe": "#FF1744"}
    fig_hourly = px.bar(
        hourly, x="hour", y="avg_speed_mph",
        color="congestion",
        color_discrete_map=color_map,
        labels={"hour": "Hour", "avg_speed_mph": "Avg Speed (mph)", "congestion": "Level"},
        template="plotly_dark",
    )
    fig_hourly.update_layout(
        height=360,
        paper_bgcolor="#0D1B2A", plot_bgcolor="#0F2337",
        xaxis=dict(tickmode="linear", dtick=2, gridcolor="#1E3A5F"),
        yaxis=dict(gridcolor="#1E3A5F"),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(bgcolor="#0F2337"),
    )
    st.plotly_chart(fig_hourly, use_container_width=True)

st.divider()

# ── Row 2: Per-sensor congestion profile ──────────────────────────
st.markdown("### 🌡️ Per-Sensor Congestion Profile (Top 20 Sensors)")
with st.spinner("Computing congestion profiles …"):
    profile = sensor_congestion_profile(df)

# Select top 20 by severe + heavy %
profile = profile.copy()
for col in ["severe", "heavy", "moderate", "free_flow"]:
    if col not in profile.columns:
        profile[col] = 0.0
profile["congestion_score"] = profile.get("severe", 0) + profile.get("heavy", 0)
top20 = profile.nlargest(20, "congestion_score")

fig_stacked = go.Figure()
for level, color in CONGESTION_COLORS.items():
    if level in top20.columns:
        fig_stacked.add_trace(go.Bar(
            name=CONGESTION_LABELS[level],
            x=top20["sensor_id"].astype(str),
            y=top20[level],
            marker_color=color,
        ))

fig_stacked.update_layout(
    barmode="stack",
    height=380,
    template="plotly_dark",
    paper_bgcolor="#0D1B2A", plot_bgcolor="#0F2337",
    xaxis=dict(tickangle=-45, title="Sensor ID", gridcolor="#1E3A5F"),
    yaxis=dict(title="% of Observations", gridcolor="#1E3A5F"),
    legend=dict(bgcolor="#0F2337"),
    margin=dict(l=0, r=0, t=40, b=0),
)
st.plotly_chart(fig_stacked, use_container_width=True)

st.divider()

# ── Row 3: Sustained low-speed events ────────────────────────────
st.markdown("### 🔴 Sustained Low-Speed Events (< 30 mph for ≥ 15 min)")
threshold = st.slider("Speed threshold (mph)", min_value=10, max_value=60, value=30, step=5)
min_dur   = st.slider("Minimum duration (minutes)", min_value=5, max_value=60, value=15, step=5)

with st.spinner("Detecting low-speed events …"):
    events = low_speed_events(df, threshold_mph=threshold, min_duration_min=min_dur)

if events.empty:
    st.success(f"No sustained low-speed events found below {threshold} mph for ≥ {min_dur} min.")
else:
    st.warning(f"Found **{len(events):,}** sustained low-speed events.")
    st.dataframe(
        events.head(50).rename(columns={
            "sensor_id": "Sensor", "start_time": "Start",
            "end_time": "End", "duration_min": "Duration (min)",
            "min_speed_mph": "Min Speed", "avg_speed_mph": "Avg Speed",
        }),
        use_container_width=True, hide_index=True,
    )
    csv = events.to_csv(index=False)
    st.download_button("📥 Download Events CSV", csv, "low_speed_events.csv", "text/csv")
