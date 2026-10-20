"""
pages/2_Traffic_Explorer.py
────────────────────────────
Page 2: Interactive sensor explorer with date/time filters.
"""

import sys
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from io import StringIO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.analytics import load_cleaned_data, sensor_stats, sensor_history
from config.settings import CLEANED_DATA_PATH, CHART_HEIGHT

st.set_page_config(page_title="Traffic Explorer", page_icon="🔍", layout="wide")
css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

st.markdown("## 🔍 Traffic Explorer")
st.markdown("*Drill down into individual sensor data with date and time filters.*")

@st.cache_data(ttl=3600)
def get_data():
    try:
        return load_cleaned_data()
    except Exception as e:
        st.error(f"Data not found: {e}.")
        st.stop()

df = get_data()
all_sensors = sorted(df["sensor_id"].unique().tolist())

# ── Sidebar filters ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Filters")
    selected_sensors = st.multiselect(
        "Select Sensors (up to 5)",
        options=all_sensors,
        default=all_sensors[:3],
        max_selections=5,
    )
    date_range = st.date_input(
        "Date Range",
        value=(pd.Timestamp("2017-01-01").date(), pd.Timestamp("2017-01-31").date()),
        min_value=pd.Timestamp("2017-01-01").date(),
        max_value=pd.Timestamp("2017-06-30").date(),
    )
    resample_opt = st.selectbox(
        "Resample (for large ranges)",
        options=["5T (raw)", "15T", "1H", "4H"],
        index=1,
    )
    resample_freq = None if resample_opt == "5T (raw)" else resample_opt.split(" ")[0]

if not selected_sensors:
    st.warning("Select at least one sensor from the sidebar.")
    st.stop()

start_date = pd.Timestamp(date_range[0]) if len(date_range) >= 1 else None
end_date   = pd.Timestamp(date_range[1]) if len(date_range) >= 2 else None

# ── Speed timeline chart ────────────────────────────────────────────
st.markdown("### 📈 Speed Timeline")
colors = ["#00D4FF", "#00C851", "#FFD700", "#FF8C00", "#FF1744"]

fig = go.Figure()
for i, sid in enumerate(selected_sensors):
    hist = sensor_history(df, sid, start_date, end_date, resample=resample_freq)
    if not hist.empty:
        fig.add_trace(go.Scatter(
            x=hist["timestamp"], y=hist["observed_speed"],
            name=f"Sensor {sid}",
            mode="lines",
            line=dict(color=colors[i % len(colors)], width=1.5),
        ))

fig.update_layout(
    height=CHART_HEIGHT,
    paper_bgcolor="#0D1B2A",
    plot_bgcolor="#0F2337",
    xaxis=dict(gridcolor="#1E3A5F", title="Timestamp"),
    yaxis=dict(gridcolor="#1E3A5F", title="Speed (mph)"),
    legend=dict(bgcolor="#0F2337", bordercolor="#1E3A5F"),
    margin=dict(l=0, r=0, t=30, b=0),
    hovermode="x unified",
    template="plotly_dark",
)
st.plotly_chart(fig, use_container_width=True)

# ── Sensor stats comparison ────────────────────────────────────────
st.markdown("### 📊 Sensor Statistics Comparison")
stats = sensor_stats(df)
filtered_stats = stats[stats["sensor_id"].isin(selected_sensors)]

col1, col2 = st.columns([2, 1])
with col1:
    fig_box = px.box(
        df[df["sensor_id"].isin(selected_sensors)],
        x="sensor_id", y="observed_speed",
        color="sensor_id",
        color_discrete_sequence=colors,
        template="plotly_dark",
        labels={"sensor_id": "Sensor ID", "observed_speed": "Speed (mph)"},
    )
    fig_box.update_layout(
        paper_bgcolor="#0D1B2A", plot_bgcolor="#0F2337",
        height=360, margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig_box, use_container_width=True)

with col2:
    st.dataframe(
        filtered_stats[["sensor_id", "mean_speed", "min_speed", "max_speed", "std_speed", "count"]]
        .rename(columns={
            "sensor_id": "Sensor", "mean_speed": "Mean", "min_speed": "Min",
            "max_speed": "Max", "std_speed": "Std", "count": "N",
        })
        .set_index("Sensor"),
        use_container_width=True,
    )

# ── Download section ───────────────────────────────────────────────
st.divider()
st.markdown("### ⬇️ Download Filtered Data")
download_frames = []
for sid in selected_sensors:
    h = sensor_history(df, sid, start_date, end_date, resample=None)
    if not h.empty:
        h["sensor_id"] = sid
        download_frames.append(h)

if download_frames:
    download_df = pd.concat(download_frames).sort_values(["sensor_id", "timestamp"])
    csv_str = download_df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv_str,
        file_name=f"traffic_data_{'_'.join(selected_sensors[:3])}.csv",
        mime="text/csv",
    )
    st.caption(f"Ready to download: {len(download_df):,} rows")
