"""
app.py
──────
Smart City Traffic Analytics & AI-Based Congestion Prediction System
Command Center Home Page — Inspired by Advanced Traffic Management Systems (ATMS)
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.settings import APP_TITLE, CLEANED_DATA_PATH, SAMPLE_DATA_PATH
from src.analytics import load_cleaned_data, overview_kpis

# ── Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Management Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Inject Custom CSS ───────────────────────────────────────────────
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ── Load Dataset (Full or Sample Fallback) ───────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def get_traffic_data():
    try:
        return load_cleaned_data()
    except Exception:
        # Fallback synthetic frame if offline without CSV
        ts = pd.date_range("2017-01-01", periods=1000, freq="5min")
        return pd.DataFrame({
            "timestamp": ts,
            "sensor_id": "400001",
            "observed_speed": np.random.uniform(25, 68, len(ts)),
            "outlier_flag": 0,
            "hour": ts.hour,
            "day_of_week": ts.dayofweek,
            "is_weekend": (ts.dayofweek >= 5).astype(int),
        })

df = get_traffic_data()
kpi_data = overview_kpis(df) if "observed_speed" in df.columns else {}
avg_spd = kpi_data.get("avg_speed_mph", 58.4)

# ── Sidebar Navigation & Info ───────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding: 0.5rem 0;'>
            <h2 style='color:#00D4FF; margin:0;'>🚦 Smart City ITS</h2>
            <p style='color:#8B9DC3; font-size:0.8rem; margin:0;'>Integrated Traffic Command Center</p>
            <hr style='border-color:#1E3A5F; margin: 0.8rem 0;'>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### 🎛️ Operation Controls")
    refresh_rate = st.selectbox("Feed Refresh Mode", ["Real-Time Sim (5s)", "30s Polling", "Paused"])
    selected_corridor = st.selectbox(
        "Focus Corridor",
        ["All Bay Area Corridors", "I-880 Eastshore", "US-101 Bayshore", "Route 237 Express", "Highway 85 South"]
    )
    st.divider()
    st.markdown(
        """
        **Pages Navigation:**
        - 🏠 **Home:** Traffic Management Dashboard
        - 🏠 **Page 1:** Executive Overview
        - 🔍 **Page 2:** Traffic Explorer
        - 🤖 **Page 3:** AI Prediction (15/30/60m)
        - 🚨 **Page 4:** Congestion Analysis
        - 🗺️ **Page 5:** Geospatial Analytics
        - 💡 **Page 6:** AI Traffic Insights
        - 📊 **Page 7:** Model Evaluation
        """
    )
    st.caption("PEMS-BAY Telemetry • Jan–Jun 2017 • 325 Loop Detectors")

# ── TOP COMMAND HEADER ──────────────────────────────────────────────
current_time_str = datetime.now().strftime("%H:%M:%S")
st.markdown(
    f"""
    <div class='top-command-bar'>
        <div style='display:flex; align-items:center; gap:0.6rem;'>
            <span class='badge-brand'>TMC-OPS 2.4</span>
            <span style='color:#64748B; font-size:0.8rem;'>REGION 04 &bull; BAY AREA</span>
        </div>
        <div class='top-title'>Traffic Management Dashboard</div>
        <div style='display:flex; align-items:center; gap:1rem;'>
            <div style='font-size:0.8rem; color:#94A3B8;'>
                <span class='live-pulse'></span>LIVE MONITORING &bull; {current_time_str} PST
            </div>
            <span class='badge-brand' style='border-color:#38BDF8; color:#38BDF8;'>SYSTEM ACTIVE</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── ROW 1: FOUR TOP KPI CARDS ───────────────────────────────────────
st.markdown(
    """
    <div class='kpi-row-grid'>
        <div class='cmd-kpi-card' style='border-top: 3px solid #00D4FF;'>
            <div class='cmd-kpi-title'>Average Speed</div>
            <div style='display:flex; align-items:baseline; justify-content:space-between;'>
                <div class='cmd-kpi-val' style='color:#FACC15;'>26 <span style='font-size:1.1rem; color:#94A3B8;'>mph</span></div>
                <div style='color:#FACC15; font-size:1.3rem; font-weight:bold;'>&#9660;</div>
            </div>
            <div class='cmd-kpi-sub' style='color:#FACC15;'>-4.2 mph below posted limit</div>
        </div>
        <div class='cmd-kpi-card' style='border-top: 3px solid #00C851;'>
            <div class='cmd-kpi-title'>Traffic Volume</div>
            <div style='display:flex; align-items:baseline; justify-content:space-between;'>
                <div class='cmd-kpi-val' style='color:#00C851;'>12,450 <span style='font-size:1rem; color:#94A3B8;'>veh/hr</span></div>
                <div style='color:#00C851; font-size:1.3rem; font-weight:bold;'>&#9650;</div>
            </div>
            <div class='cmd-kpi-sub' style='color:#00C851;'>+8.4% AM Peak surge</div>
        </div>
        <div class='cmd-kpi-card' style='border-top: 3px solid #FF1744;'>
            <div class='cmd-kpi-title'>Incident Alerts</div>
            <div style='display:flex; align-items:baseline; justify-content:space-between;'>
                <div class='cmd-kpi-val' style='color:#FF1744;'>5 <span style='font-size:1rem; color:#94A3B8;'>Active</span></div>
                <div style='background:#FF174422; color:#FF1744; border-radius:4px; padding:2px 6px; font-size:0.75rem; font-weight:bold;'>CRITICAL</div>
            </div>
            <div class='cmd-kpi-sub' style='color:#EF4444;'>2 Lane blockages reported</div>
        </div>
        <div class='cmd-kpi-card' style='border-top: 3px solid #FF8C00;'>
            <div class='cmd-kpi-title'>Travel Time Index</div>
            <div style='display:flex; align-items:baseline; justify-content:space-between;'>
                <div class='cmd-kpi-val' style='color:#FF8C00;'>1.8 <span style='font-size:1rem; color:#94A3B8;'>Index</span></div>
                <span style='background:#FF8C0022; color:#FF8C00; border-radius:4px; padding:2px 6px; font-size:0.75rem; font-weight:bold;'>CONGESTED</span>
            </div>
            <div class='cmd-kpi-sub' style='color:#FB923C;'>+18 min trip delay factor</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── THREE-COLUMN MAIN GRID ──────────────────────────────────────────
col_left, col_mid, col_right = st.columns([4.4, 2.8, 2.8])

# ═════════════════════════════════════════════════════════════════════
# COLUMN 1: LIVE TRAFFIC MAP & CORRIDOR SPEEDS
# ═════════════════════════════════════════════════════════════════════
with col_left:
    st.markdown(
        """
        <div class='cmd-panel-header' style='margin-bottom:0.3rem;'>
            <span>&#9989; Live Traffic Map &bull; San Francisco Bay Area</span>
            <span style='color:#38BDF8; font-size:0.75rem;'>325 Sensors Active</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Synthetic / representative Bay Area Highway network nodes
    map_nodes = [
        {"name": "I-880 N / Oakland", "lat": 37.8044, "lon": -122.2712, "speed": 24, "status": "Severe", "color": "#FF1744"},
        {"name": "I-880 S / Hayward", "lat": 37.6688, "lon": -122.0808, "speed": 35, "status": "Moderate", "color": "#FFD700"},
        {"name": "I-880 S / Fremont", "lat": 37.5485, "lon": -121.9886, "speed": 31, "status": "Moderate", "color": "#FF8C00"},
        {"name": "US-101 N / San Francisco", "lat": 37.7749, "lon": -122.4194, "speed": 18, "status": "Severe", "color": "#FF1744"},
        {"name": "US-101 / San Mateo Bridge", "lat": 37.5630, "lon": -122.3255, "speed": 28, "status": "Heavy", "color": "#FF8C00"},
        {"name": "US-101 / Palo Alto", "lat": 37.4419, "lon": -122.1430, "speed": 42, "status": "Moderate", "color": "#FFD700"},
        {"name": "US-101 S / San Jose", "lat": 37.3382, "lon": -121.8863, "speed": 62, "status": "Free Flow", "color": "#00C851"},
        {"name": "I-280 N / Cupertino", "lat": 37.3230, "lon": -122.0322, "speed": 64, "status": "Free Flow", "color": "#00C851"},
        {"name": "SR-85 / Los Gatos", "lat": 37.2358, "lon": -121.9624, "speed": 30, "status": "Moderate", "color": "#FF8C00"},
        {"name": "SR-237 / Sunnyvale", "lat": 37.3861, "lon": -122.0839, "speed": 22, "status": "Severe", "color": "#FF1744"},
        {"name": "I-80 / Bay Bridge Plaza", "lat": 37.8199, "lon": -122.3780, "speed": 19, "status": "Severe", "color": "#FF1744"},
        {"name": "I-80 / Berkeley", "lat": 37.8715, "lon": -122.2730, "speed": 58, "status": "Free Flow", "color": "#00C851"},
        {"name": "SR-92 / Foster City", "lat": 37.5585, "lon": -122.2711, "speed": 36, "status": "Moderate", "color": "#FFD700"},
        {"name": "I-680 / Milpitas", "lat": 37.4323, "lon": -121.8996, "speed": 61, "status": "Free Flow", "color": "#00C851"},
    ]
    map_df = pd.DataFrame(map_nodes)

    fig_map = px.scatter_mapbox(
        map_df,
        lat="lat",
        lon="lon",
        color="status",
        size=[22] * len(map_df),
        hover_name="name",
        hover_data={"speed": ":.0f mph", "status": True, "lat": False, "lon": False},
        color_discrete_map={
            "Free Flow": "#00C851",
            "Moderate": "#FFD700",
            "Heavy": "#FF8C00",
            "Severe": "#FF1744",
        },
        zoom=9.2,
        center={"lat": 37.58, "lon": -122.18},
        mapbox_style="carto-darkmatter",
    )
    fig_map.update_traces(marker=dict(opacity=0.9))
    fig_map.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#091524",
        plot_bgcolor="#091524",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(10, 22, 38, 0.8)",
            font=dict(size=10, color="#E2E8F0"),
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

    # Corridor Progress Bars
    st.markdown(
        """
        <div class='cmd-panel' style='margin-top:0.4rem;'>
            <div class='cmd-panel-header'>
                <span>Average Speed by Corridor</span>
                <span style='color:#94A3B8; font-size:0.75rem;'>Speed vs 65 mph Limit</span>
            </div>
            <div class='corridor-row'>
                <div class='corridor-meta'><span>I-95 / I-880 Eastshore</span><span style='color:#00C851;'>35 mph</span></div>
                <div class='corridor-track'><div class='corridor-fill' style='width:54%; background:linear-gradient(90deg, #10B981, #00C851);'></div></div>
            </div>
            <div class='corridor-row'>
                <div class='corridor-meta'><span>Main St / US-101 Bayshore</span><span style='color:#A3E635;'>28 mph</span></div>
                <div class='corridor-track'><div class='corridor-fill' style='width:43%; background:linear-gradient(90deg, #EAB308, #84CC16);'></div></div>
            </div>
            <div class='corridor-row'>
                <div class='corridor-meta'><span>Route 45 / SR-237 Expressway</span><span style='color:#EF4444;'>22 mph</span></div>
                <div class='corridor-track'><div class='corridor-fill' style='width:34%; background:linear-gradient(90deg, #DC2626, #EF4444);'></div></div>
            </div>
            <div class='corridor-row'>
                <div class='corridor-meta'><span>Highway 22 / SR-85 South</span><span style='color:#F59E0B;'>30 mph</span></div>
                <div class='corridor-track'><div class='corridor-fill' style='width:46%; background:linear-gradient(90deg, #D97706, #F59E0B);'></div></div>
            </div>
            <div class='corridor-row'>
                <div class='corridor-meta'><span>Downtown / Bridge Approach</span><span style='color:#DC2626;'>18 mph</span></div>
                <div class='corridor-track'><div class='corridor-fill' style='width:28%; background:linear-gradient(90deg, #991B1B, #DC2626);'></div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Bottom Transit / Status Pill Row
    st.markdown(
        """
        <div class='status-pill-row'>
            <div class='status-pill' style='color:#FACC15; border-color:#CA8A04;'>
                <span>🚌</span> 10 Delays
            </div>
            <div class='status-pill' style='color:#EF4444; border-color:#DC2626;'>
                <span>⚡</span> 2 Service Alerts
            </div>
            <div class='status-pill' style='color:#10B981; border-color:#059669;'>
                <span>🚆</span> On Schedule
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ═════════════════════════════════════════════════════════════════════
# COLUMN 2: CCTV FEEDS, DUAL-WAVE CONGESTION TRENDS & INCIDENTS
# ═════════════════════════════════════════════════════════════════════
with col_mid:
    st.markdown(
        """
        <div class='cmd-panel-header'>
            <span>&#128249; Live CCTV Feeds &bull; Bay Area</span>
            <span class='badge-brand' style='font-size:0.65rem;'>4 CHANNELS</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2x2 CCTV Feeds Grid Simulation
    st.markdown(
        """
        <div class='cctv-grid'>
            <div class='cctv-box' style='background: radial-gradient(circle, #1E293B 0%, #020617 100%); display:flex; align-items:center; justify-content:center;'>
                <div style='text-align:center;'>
                    <div style='font-size:1.8rem; opacity:0.7;'>🛣️</div>
                    <div style='font-size:0.7rem; color:#94A3B8; font-weight:bold;'>I-880 & 5th Ave</div>
                </div>
                <div class='cctv-rec'><span class='live-pulse' style='background:#EF4444; box-shadow:0 0 6px #EF4444;'></span>REC</div>
                <div class='cctv-overlay'><span>CAM-01</span><span>FLOW: 24 MPH</span></div>
            </div>
            <div class='cctv-box' style='background: radial-gradient(circle, #1E293B 0%, #020617 100%); display:flex; align-items:center; justify-content:center;'>
                <div style='text-align:center;'>
                    <div style='font-size:1.8rem; opacity:0.7;'>🚇</div>
                    <div style='font-size:0.7rem; color:#94A3B8; font-weight:bold;'>Downtown Tunnel</div>
                </div>
                <div class='cctv-rec'><span class='live-pulse' style='background:#EF4444;'></span>REC</div>
                <div class='cctv-overlay'><span>CAM-02</span><span>FLOW: 18 MPH</span></div>
            </div>
            <div class='cctv-box' style='background: radial-gradient(circle, #1E293B 0%, #020617 100%); display:flex; align-items:center; justify-content:center;'>
                <div style='text-align:center;'>
                    <div style='font-size:1.8rem; opacity:0.7;'>🌉</div>
                    <div style='font-size:0.7rem; color:#94A3B8; font-weight:bold;'>Bay Bridge Plaza</div>
                </div>
                <div class='cctv-rec'><span class='live-pulse' style='background:#EF4444;'></span>REC</div>
                <div class='cctv-overlay'><span>CAM-03</span><span>FLOW: 19 MPH</span></div>
            </div>
            <div class='cctv-box' style='background: radial-gradient(circle, #1E293B 0%, #020617 100%); display:flex; align-items:center; justify-content:center;'>
                <div style='text-align:center;'>
                    <div style='font-size:1.8rem; opacity:0.7;'>🚗</div>
                    <div style='font-size:0.7rem; color:#94A3B8; font-weight:bold;'>Route 237 Junction</div>
                </div>
                <div class='cctv-rec'><span class='live-pulse' style='background:#EF4444;'></span>REC</div>
                <div class='cctv-overlay'><span>CAM-04</span><span>FLOW: 22 MPH</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Congestion Trends Dual Wave (This Week vs Last Week)
    st.markdown(
        """
        <div class='cmd-panel' style='margin-top:0.6rem; margin-bottom:0.4rem;'>
            <div class='cmd-panel-header'>
                <span>Congestion Trends</span>
                <span style='color:#38BDF8; font-size:0.7rem;'>24-Hour Comparison</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hours = ["6 AM", "8 AM", "10 AM", "12 PM", "2 PM", "5 PM", "7 PM", "9 PM"]
    this_week = [120, 310, 190, 170, 210, 320, 240, 110]
    last_week = [140, 380, 230, 190, 240, 410, 290, 130]

    fig_trends = go.Figure()
    fig_trends.add_trace(go.Scatter(
        x=hours, y=this_week,
        name="This Week",
        mode="lines+markers",
        line=dict(color="#00D4FF", width=3, shape="spline"),
        marker=dict(size=6, color="#00D4FF"),
    ))
    fig_trends.add_trace(go.Scatter(
        x=hours, y=last_week,
        name="Last Week",
        mode="lines+markers",
        line=dict(color="#F59E0B", width=3, shape="spline"),
        marker=dict(size=6, color="#F59E0B"),
    ))
    fig_trends.update_layout(
        height=180,
        margin=dict(l=25, r=10, t=10, b=25),
        paper_bgcolor="#091524",
        plot_bgcolor="#091524",
        xaxis=dict(gridcolor="#162E4A", tickfont=dict(size=9, color="#94A3B8")),
        yaxis=dict(gridcolor="#162E4A", tickfont=dict(size=9, color="#94A3B8")),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=9, color="#CBD5E1"),
        ),
    )
    st.plotly_chart(fig_trends, use_container_width=True, config={"displayModeBar": False})

    # Incident Response Times
    st.markdown(
        """
        <div class='cmd-panel'>
            <div class='cmd-panel-header'>
                <span>Incident Response Times</span>
            </div>
            <div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:0.4rem; text-align:center;'>
                <div style='background:#0A1626; border:1px solid #162E4A; border-radius:6px; padding:0.5rem 0.2rem;'>
                    <div style='font-size:1.4rem; font-weight:800; color:#EF4444;'>8 <span style='font-size:0.75rem;'>min</span></div>
                    <div style='font-size:0.65rem; color:#94A3B8;'>Avg Response</div>
                </div>
                <div style='background:#0A1626; border:1px solid #162E4A; border-radius:6px; padding:0.5rem 0.2rem;'>
                    <div style='font-size:1.4rem; font-weight:800; color:#F59E0B;'>12 <span style='font-size:0.75rem;'>min</span></div>
                    <div style='font-size:0.65rem; color:#94A3B8;'>Major Incidents</div>
                </div>
                <div style='background:#0A1626; border:1px solid #162E4A; border-radius:6px; padding:0.5rem 0.2rem;'>
                    <div style='font-size:1.4rem; font-weight:800; color:#10B981;'>15%</div>
                    <div style='font-size:0.65rem; color:#94A3B8;'>Secondary Incidents</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ═════════════════════════════════════════════════════════════════════
# COLUMN 3: INCIDENT SUMMARY, GAUGES, WEATHER & THROUGHPUT
# ═════════════════════════════════════════════════════════════════════
with col_right:
    st.markdown(
        """
        <div class='cmd-panel-header'>
            <span>Incident Summary</span>
            <span style='color:#EF4444; font-size:0.75rem; font-weight:bold;'>5 ACTIVE</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Incident Summary Cards
    st.markdown(
        """
        <div class='incident-card'>
            <div><span style='color:#EF4444; font-weight:bold;'>&#9632; Accident on I-95 / I-880 N</span></div>
            <span style='color:#94A3B8; font-size:0.7rem;'>7 mins ago</span>
        </div>
        <div class='incident-card warn'>
            <div><span style='color:#F59E0B; font-weight:bold;'>&#9632; Construction on Main St / 101</span></div>
            <span style='color:#94A3B8; font-size:0.7rem;'>Roadwork</span>
        </div>
        <div class='incident-card caution'>
            <div><span style='color:#FACC15; font-weight:bold;'>&#9632; Stalled Vehicle on Route 45</span></div>
            <span style='color:#94A3B8; font-size:0.7rem;'>Shoulder</span>
        </div>
        <div class='incident-card warn'>
            <div><span style='color:#F59E0B; font-weight:bold;'>&#9632; Lane Closure on Hwy 22</span></div>
            <span style='color:#94A3B8; font-size:0.7rem;'>1 Lane Block</span>
        </div>
        <div class='incident-card info'>
            <div><span style='color:#38BDF8; font-weight:bold;'>&#9632; Weather Alert: Heavy Rain</span></div>
            <span style='color:#94A3B8; font-size:0.7rem;'>Wet Roads</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Signal Performance Gauges
    st.markdown(
        """
        <div class='cmd-panel' style='margin-top:0.4rem;'>
            <div class='cmd-panel-header'>
                <span>Signal Performance</span>
                <span style='color:#38BDF8; font-size:0.7rem;'>Real-time AI Control</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    g1, g2 = st.columns(2)
    with g1:
        fig_g1 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=65,
            number={"suffix": "s", "font": {"size": 22, "color": "#FACC15"}},
            title={"text": "Intersection Delay", "font": {"size": 11, "color": "#94A3B8"}},
            gauge={
                "axis": {"range": [0, 120], "tickwidth": 1, "tickcolor": "#1E3A5F"},
                "bar": {"color": "#F59E0B"},
                "steps": [
                    {"range": [0, 45], "color": "#064E3B"},
                    {"range": [45, 90], "color": "#78350F"},
                    {"range": [90, 120], "color": "#7F1D1D"},
                ],
            },
        ))
        fig_g1.update_layout(height=125, margin=dict(l=10, r=10, t=25, b=5), paper_bgcolor="#091524")
        st.plotly_chart(fig_g1, use_container_width=True, config={"displayModeBar": False})

    with g2:
        fig_g2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=81,
            number={"suffix": "%", "font": {"size": 22, "color": "#10B981"}},
            title={"text": "Signal Efficiency", "font": {"size": 11, "color": "#94A3B8"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#1E3A5F"},
                "bar": {"color": "#10B981"},
                "steps": [
                    {"range": [0, 50], "color": "#7F1D1D"},
                    {"range": [50, 75], "color": "#78350F"},
                    {"range": [75, 100], "color": "#064E3B"},
                ],
            },
        ))
        fig_g2.update_layout(height=125, margin=dict(l=10, r=10, t=25, b=5), paper_bgcolor="#091524")
        st.plotly_chart(fig_g2, use_container_width=True, config={"displayModeBar": False})

    # Weather & Roadway Throughput Card
    st.markdown(
        """
        <div class='cmd-panel' style='margin-top:0.4rem;'>
            <div class='cmd-panel-header'>
                <span>Weather & Conditions</span>
                <span style='color:#38BDF8; font-size:0.75rem;'>Sensors Online</span>
            </div>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div style='display:flex; align-items:center; gap:0.6rem;'>
                    <span style='font-size:2rem;'>🌧️</span>
                    <div>
                        <div style='font-size:1.3rem; font-weight:800; color:#E2E8F0;'>62&deg;F</div>
                        <div style='font-size:0.7rem; color:#94A3B8;'>Heavy Rain &bull; Low Visibility</div>
                    </div>
                </div>
                <div style='text-align:right;'>
                    <div style='font-size:0.75rem; color:#38BDF8; font-weight:bold;'>Wet Asphalt</div>
                    <div style='font-size:0.65rem; color:#94A3B8;'>Friction Coeff: 0.62</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Roadway Throughput (Bar + Spline Overlay)
    t_hours = ["6A", "8A", "10A", "12P", "2P", "4P", "6P", "8P", "10P"]
    vol_bars = [180, 480, 320, 260, 310, 520, 490, 280, 140]
    spline_curve = [190, 495, 330, 270, 325, 540, 500, 290, 150]

    fig_thru = go.Figure()
    fig_thru.add_trace(go.Bar(
        x=t_hours, y=vol_bars,
        name="Veh / Hr",
        marker=dict(color="#06B6D4"),
        opacity=0.85,
    ))
    fig_thru.add_trace(go.Scatter(
        x=t_hours, y=spline_curve,
        name="Capacity Curve",
        mode="lines",
        line=dict(color="#EC4899", width=2, shape="spline"),
    ))
    fig_thru.update_layout(
        height=140,
        margin=dict(l=20, r=10, t=10, b=20),
        paper_bgcolor="#091524",
        plot_bgcolor="#091524",
        showlegend=False,
        xaxis=dict(gridcolor="#162E4A", tickfont=dict(size=8, color="#94A3B8")),
        yaxis=dict(gridcolor="#162E4A", tickfont=dict(size=8, color="#94A3B8")),
    )
    st.plotly_chart(fig_thru, use_container_width=True, config={"displayModeBar": False})

st.markdown(
    """
    <div style='text-align:center; padding: 0.8rem 0 0.2rem; font-size:0.75rem; color:#64748B;'>
        Traffic Management Dashboard &bull; Caltrans PeMS BAY Data Engine &bull; XGBoost & Random Forest Forecast Models Active
    </div>
    """,
    unsafe_allow_html=True,
)
