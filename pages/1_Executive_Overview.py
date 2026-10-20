"""
pages/1_Executive_Overview.py
──────────────────────────────
Page 1: Executive Overview — KPIs, speed trends, peak periods.
"""

import sys
from pathlib import Path
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.analytics import load_cleaned_data, overview_kpis, hourly_avg_speed, daily_avg_speed, lowest_speed_sensors, peak_periods
from config.settings import CLEANED_DATA_PATH, DEFAULT_SENSOR_COUNT, CHART_HEIGHT

st.set_page_config(page_title="Executive Overview", page_icon="🏠", layout="wide")

# Load CSS
css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

st.markdown("## 🏠 Executive Overview")
st.markdown("*Real-time KPIs and traffic trends across the PEMS-BAY sensor network.*")

# ── Load data ──────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading traffic data …", ttl=3600)
def get_data():
    try:
        return load_cleaned_data()
    except Exception as e:
        st.error(f"Data not found: {e}. Please ensure dataset or sample exists.")
        st.stop()

df = get_data()
kpis = overview_kpis(df)

# ── KPI Row ────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("📡 Total Sensors",    f"{kpis['total_sensors']:,}")
col2.metric("📊 Observations",     f"{kpis['total_observations']:,}")
col3.metric("🚗 Avg Speed",        f"{kpis['avg_speed_mph']} mph")
col4.metric("🔴 Min Speed",        f"{kpis['min_speed_mph']} mph")
col5.metric("⚠️  Outlier Readings", f"{kpis['outlier_pct']}%")

st.divider()

# ── Date range info ────────────────────────────────────────────────
st.info(
    f"📅 Dataset covers **{kpis['date_start']}** to **{kpis['date_end']}** "
    f"at 5-minute intervals (Jan–Jun 2017, Bay Area, CA)"
)

# ── Row 1: Hourly speed trend + Daily pattern ──────────────────────
col_a, col_b = st.columns([3, 2])

with col_a:
    st.markdown("### 📈 Average Speed by Hour of Day")
    hourly = hourly_avg_speed(df)
    fig_hourly = px.area(
        hourly, x="hour", y="avg_speed_mph",
        labels={"hour": "Hour of Day", "avg_speed_mph": "Avg Speed (mph)"},
        color_discrete_sequence=["#00D4FF"],
        template="plotly_dark",
    )
    fig_hourly.update_layout(
        height=CHART_HEIGHT,
        paper_bgcolor="#0D1B2A",
        plot_bgcolor="#0F2337",
        xaxis=dict(tickmode="linear", dtick=2, gridcolor="#1E3A5F"),
        yaxis=dict(gridcolor="#1E3A5F"),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    fig_hourly.add_vrect(x0=7, x1=9, fillcolor="#FF1744", opacity=0.1,
                         annotation_text="AM Peak", annotation_position="top left")
    fig_hourly.add_vrect(x0=17, x1=19, fillcolor="#FF8C00", opacity=0.1,
                         annotation_text="PM Peak", annotation_position="top left")
    st.plotly_chart(fig_hourly, use_container_width=True)

with col_b:
    st.markdown("### 📅 Average Speed by Day")
    daily = daily_avg_speed(df)
    fig_daily = px.bar(
        daily, x="day_name", y="avg_speed_mph",
        color="avg_speed_mph",
        color_continuous_scale=["#FF1744", "#FFD700", "#00C851"],
        labels={"day_name": "Day", "avg_speed_mph": "Avg Speed (mph)"},
        template="plotly_dark",
    )
    fig_daily.update_layout(
        height=CHART_HEIGHT,
        paper_bgcolor="#0D1B2A",
        plot_bgcolor="#0F2337",
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_daily, use_container_width=True)

st.divider()

# ── Row 2: Peak periods + Lowest sensors ──────────────────────────
col_c, col_d = st.columns([1, 2])

with col_c:
    st.markdown("### 🔴 Peak Congestion Hours")
    peaks = peak_periods(df)
    peaks["hour_str"] = peaks["hour"].apply(lambda h: f"{int(h):02d}:00–{int(h)+1:02d}:00")
    peaks["avg_speed_mph"] = peaks["avg_speed_mph"].round(1)
    st.dataframe(
        peaks[["hour_str", "avg_speed_mph"]].rename(columns={
            "hour_str": "Hour", "avg_speed_mph": "Avg Speed (mph)"
        }),
        use_container_width=True, hide_index=True,
    )

with col_d:
    st.markdown(f"### 🐢 Bottom {DEFAULT_SENSOR_COUNT} Sensors by Average Speed")
    low = lowest_speed_sensors(df, n=DEFAULT_SENSOR_COUNT)
    fig_low = px.bar(
        low, x="sensor_id", y="mean_speed",
        color="mean_speed",
        color_continuous_scale=["#FF1744", "#FF8C00", "#FFD700"],
        labels={"sensor_id": "Sensor ID", "mean_speed": "Avg Speed (mph)"},
        template="plotly_dark",
        error_y="std_speed",
    )
    fig_low.update_layout(
        height=320,
        paper_bgcolor="#0D1B2A",
        plot_bgcolor="#0F2337",
        coloraxis_showscale=False,
        xaxis=dict(tickangle=-45),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_low, use_container_width=True)

st.divider()
st.caption(
    "Source: PEMS-BAY dataset — California Bay Area, Jan–Jun 2017. "
    "Speed values in mph. Thresholds are demonstration rules."
)
