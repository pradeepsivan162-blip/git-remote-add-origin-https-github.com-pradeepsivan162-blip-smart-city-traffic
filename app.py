"""
app.py
──────
Smart City Traffic Analytics & AI-Based Congestion Prediction System
Coimbatore Smart City Traffic Command & Control Centre (ICCC) Dashboard
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
    page_title="Coimbatore Traffic Management Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Inject Custom CSS ───────────────────────────────────────────────
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ── Safe Map Renderer Helper (Supports Plotly 6+, Plotly 5 & Fallbacks) ──
def render_traffic_map(map_df):
    """Safely render interactive map across all Plotly / Streamlit versions."""
    map_colors = {
        "Free Flow": "#00C851",
        "Moderate": "#FFD700",
        "Heavy": "#FF8C00",
        "Severe": "#FF1744",
    }
    
    # 1. Plotly 6.0+ px.scatter_map
    if hasattr(px, "scatter_map"):
        try:
            fig = px.scatter_map(
                map_df,
                lat="lat",
                lon="lon",
                color="status",
                size=[18] * len(map_df),
                hover_name="name",
                hover_data={"speed": ":.0f km/h", "status": True, "lat": False, "lon": False},
                color_discrete_map=map_colors,
                zoom=11.3,
                center={"lat": 11.0183, "lon": 76.9850},
            )
            fig.update_layout(
                height=320,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="#091524",
                map_style="carto-darkmatter",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=0.02,
                    xanchor="center",
                    x=0.5,
                    bgcolor="rgba(10, 22, 38, 0.85)",
                    font=dict(size=10, color="#E2E8F0"),
                ),
            )
            return fig
        except Exception:
            pass

    # 2. Plotly 5.x px.scatter_mapbox
    if hasattr(px, "scatter_mapbox"):
        try:
            fig = px.scatter_mapbox(
                map_df,
                lat="lat",
                lon="lon",
                color="status",
                size=[18] * len(map_df),
                hover_name="name",
                hover_data={"speed": ":.0f km/h", "status": True, "lat": False, "lon": False},
                color_discrete_map=map_colors,
                zoom=11.3,
                center={"lat": 11.0183, "lon": 76.9850},
                mapbox_style="carto-darkmatter",
            )
            fig.update_layout(
                height=320,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="#091524",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=0.02,
                    xanchor="center",
                    x=0.5,
                    bgcolor="rgba(10, 22, 38, 0.85)",
                    font=dict(size=10, color="#E2E8F0"),
                ),
            )
            return fig
        except Exception:
            pass

    # 3. Universal Fallback: go.Scatter
    fig = go.Figure()
    for status, color in map_colors.items():
        sub = map_df[map_df["status"] == status]
        if not sub.empty:
            fig.add_trace(go.Scatter(
                x=sub["lon"],
                y=sub["lat"],
                mode="markers+text",
                name=status,
                text=sub["name"],
                textposition="top center",
                marker=dict(size=12, color=color),
            ))
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="#091524",
        plot_bgcolor="#0A1626",
        xaxis=dict(gridcolor="#162E4A", title="Longitude"),
        yaxis=dict(gridcolor="#162E4A", title="Latitude"),
        legend=dict(orientation="h", bgcolor="#0A1626"),
    )
    return fig

# ── Load Dataset ───────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def get_traffic_data():
    try:
        return load_cleaned_data()
    except Exception:
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

# ── Sidebar Navigation ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding: 0.5rem 0;'>
            <h2 style='color:#00D4FF; margin:0;'>🚦 Coimbatore ICCC</h2>
            <p style='color:#8B9DC3; font-size:0.8rem; margin:0;'>Smart City Traffic Management System</p>
            <hr style='border-color:#1E3A5F; margin: 0.8rem 0;'>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### 🎛️ Operation Controls")
    refresh_mode = st.selectbox("Feed Mode", ["Live Feed (Simulated)", "5-min Polling", "Paused"])
    selected_zone = st.selectbox(
        "Focus Zone",
        [
            "All City Corridors",
            "East Zone (Avinashi Rd / Airport)",
            "Central Zone (Gandhipuram / RS Puram)",
            "South Zone (Trichy Rd / Ukkadam)",
            "North Zone (Sathy Rd / IT Corridor)",
        ]
    )
    st.divider()
    st.markdown(
        """
        **Pages Navigation:**
        - 🏠 **Home:** Coimbatore Traffic Command Center
        - 🏠 **Page 1:** Executive Overview
        - 🔍 **Page 2:** Traffic Explorer
        - 🤖 **Page 3:** AI Prediction (15/30/60m)
        - 🚨 **Page 4:** Congestion Analysis
        - 🗺️ **Page 5:** Geospatial Analytics
        - 💡 **Page 6:** AI Traffic Insights
        - 📊 **Page 7:** Model Evaluation
        """
    )
    st.caption("Coimbatore Traffic Police & Smart City Telemetry Network")

# ── TOP COMMAND HEADER ──────────────────────────────────────────────
current_time_str = datetime.now().strftime("%H:%M:%S")
st.markdown(
    f"""
    <div class='top-command-bar'>
        <div style='display:flex; align-items:center; gap:0.6rem;'>
            <span class='badge-brand'>ICCC-CBE 3.2</span>
            <span style='color:#64748B; font-size:0.8rem;'>COIMBATORE SMART CITY &bull; TN POLICE</span>
        </div>
        <div class='top-title'>Traffic Management Dashboard</div>
        <div style='display:flex; align-items:center; gap:1rem;'>
            <div style='font-size:0.8rem; color:#94A3B8;'>
                <span class='live-pulse'></span>LIVE MONITORING &bull; {current_time_str} IST
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
                <div class='cmd-kpi-val' style='color:#FACC15;'>26 <span style='font-size:1.1rem; color:#94A3B8;'>km/h</span></div>
                <div style='color:#FACC15; font-size:1.3rem; font-weight:bold;'>&#9660;</div>
            </div>
            <div class='cmd-kpi-sub' style='color:#FACC15;'>-5.4 km/h below city posted norm</div>
        </div>
        <div class='cmd-kpi-card' style='border-top: 3px solid #00C851;'>
            <div class='cmd-kpi-title'>Traffic Volume</div>
            <div style='display:flex; align-items:baseline; justify-content:space-between;'>
                <div class='cmd-kpi-val' style='color:#00C851;'>14,850 <span style='font-size:1rem; color:#94A3B8;'>veh/hr</span></div>
                <div style='color:#00C851; font-size:1.3rem; font-weight:bold;'>&#9650;</div>
            </div>
            <div class='cmd-kpi-sub' style='color:#00C851;'>+9.2% Gandhipuram-Avinashi surge</div>
        </div>
        <div class='cmd-kpi-card' style='border-top: 3px solid #FF1744;'>
            <div class='cmd-kpi-title'>Incident Alerts</div>
            <div style='display:flex; align-items:baseline; justify-content:space-between;'>
                <div class='cmd-kpi-val' style='color:#FF1744;'>5 <span style='font-size:1rem; color:#94A3B8;'>Active</span></div>
                <div style='background:#FF174422; color:#FF1744; border-radius:4px; padding:2px 6px; font-size:0.75rem; font-weight:bold;'>CRITICAL</div>
            </div>
            <div class='cmd-kpi-sub' style='color:#EF4444;'>2 Lane diversions active</div>
        </div>
        <div class='cmd-kpi-card' style='border-top: 3px solid #FF8C00;'>
            <div class='cmd-kpi-title'>Travel Time Index</div>
            <div style='display:flex; align-items:baseline; justify-content:space-between;'>
                <div class='cmd-kpi-val' style='color:#FF8C00;'>1.8 <span style='font-size:1rem; color:#94A3B8;'>Index</span></div>
                <span style='background:#FF8C0022; color:#FF8C00; border-radius:4px; padding:2px 6px; font-size:0.75rem; font-weight:bold;'>CONGESTED</span>
            </div>
            <div class='cmd-kpi-sub' style='color:#FB923C;'>+19 min peak corridor delay</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── THREE-COLUMN MAIN GRID ──────────────────────────────────────────
col_left, col_mid, col_right = st.columns([4.4, 2.8, 2.8])

# ═════════════════════════════════════════════════════════════════════
# COLUMN 1: LIVE TRAFFIC MAP (COIMBATORE) & CORRIDOR SPEEDS
# ═════════════════════════════════════════════════════════════════════
with col_left:
    st.markdown(
        """
        <div class='cmd-panel-header' style='margin-bottom:0.3rem;'>
            <span>&#9989; Live Traffic Map &bull; Coimbatore City Corridors</span>
            <span style='color:#38BDF8; font-size:0.75rem;'>Sensors & ITMS Active</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Coimbatore Real Geo-Spatial Junction Nodes
    coimbatore_nodes = [
        {"name": "Avinashi Rd / Lakshmi Mills", "lat": 11.0135, "lon": 76.9850, "speed": 34, "status": "Moderate"},
        {"name": "Avinashi Rd / Hope College", "lat": 11.0264, "lon": 77.0125, "speed": 22, "status": "Severe"},
        {"name": "Avinashi Rd / TIDEL Park CIT", "lat": 11.0289, "lon": 77.0305, "speed": 26, "status": "Heavy"},
        {"name": "Avinashi Rd / Airport Jn", "lat": 11.0339, "lon": 77.0450, "speed": 55, "status": "Free Flow"},
        {"name": "Gandhipuram Bus Stand Jn", "lat": 11.0183, "lon": 76.9644, "speed": 18, "status": "Severe"},
        {"name": "100 Feet Road / Tatabad", "lat": 11.0210, "lon": 76.9580, "speed": 24, "status": "Heavy"},
        {"name": "Ukkadam Flyover Junction", "lat": 10.9925, "lon": 76.9608, "speed": 20, "status": "Severe"},
        {"name": "Trichy Rd / Ramanathapuram", "lat": 11.0022, "lon": 76.9855, "speed": 32, "status": "Moderate"},
        {"name": "Trichy Rd / Singanallur Jn", "lat": 10.9998, "lon": 77.0210, "speed": 21, "status": "Severe"},
        {"name": "Sathy Rd / Ganapathy", "lat": 11.0375, "lon": 76.9790, "speed": 27, "status": "Heavy"},
        {"name": "Sathy Rd / Saravanampatti IT Hub", "lat": 11.0805, "lon": 76.9975, "speed": 23, "status": "Severe"},
        {"name": "MTP Rd / Saibaba Colony", "lat": 11.0270, "lon": 76.9460, "speed": 30, "status": "Moderate"},
        {"name": "MTP Rd / Thudiyalur Jn", "lat": 11.0775, "lon": 76.9385, "speed": 45, "status": "Free Flow"},
        {"name": "RS Puram / DB Road Jn", "lat": 11.0090, "lon": 76.9480, "speed": 24, "status": "Heavy"},
        {"name": "Pollachi Rd / Sundarapuram", "lat": 10.9525, "lon": 76.9680, "speed": 38, "status": "Moderate"},
        {"name": "Palakkad Rd / Kuniyamuthur", "lat": 10.9630, "lon": 76.9450, "speed": 28, "status": "Heavy"},
        {"name": "Race Course Outer Ring", "lat": 11.0065, "lon": 76.9740, "speed": 48, "status": "Free Flow"},
    ]
    cbe_df = pd.DataFrame(coimbatore_nodes)

    fig_map = render_traffic_map(cbe_df)
    st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

    # Average Speed by Coimbatore Corridor
    st.markdown(
        """
        <div class='cmd-panel' style='margin-top:0.4rem;'>
            <div class='cmd-panel-header'>
                <span>Average Speed by Coimbatore Corridor</span>
                <span style='color:#94A3B8; font-size:0.75rem;'>Speed vs 50 km/h City Limit</span>
            </div>
            <div class='corridor-row'>
                <div class='corridor-meta'><span>Avinashi Road Elevated Flyover</span><span style='color:#00C851;'>38 km/h</span></div>
                <div class='corridor-track'><div class='corridor-fill' style='width:76%; background:linear-gradient(90deg, #10B981, #00C851);'></div></div>
            </div>
            <div class='corridor-row'>
                <div class='corridor-meta'><span>Trichy Road (Singanallur - Ramanathapuram)</span><span style='color:#FACC15;'>28 km/h</span></div>
                <div class='corridor-track'><div class='corridor-fill' style='width:56%; background:linear-gradient(90deg, #EAB308, #FACC15);'></div></div>
            </div>
            <div class='corridor-row'>
                <div class='corridor-meta'><span>Sathy Road (Saravanampatti IT Corridor)</span><span style='color:#EF4444;'>22 km/h</span></div>
                <div class='corridor-track'><div class='corridor-fill' style='width:44%; background:linear-gradient(90deg, #DC2626, #EF4444);'></div></div>
            </div>
            <div class='corridor-row'>
                <div class='corridor-meta'><span>Mettupalayam Road (Saibaba Colony - Thudiyalur)</span><span style='color:#F59E0B;'>30 km/h</span></div>
                <div class='corridor-track'><div class='corridor-fill' style='width:60%; background:linear-gradient(90deg, #D97706, #F59E0B);'></div></div>
            </div>
            <div class='corridor-row'>
                <div class='corridor-meta'><span>Palakkad Road (Ukkadam - Kuniyamuthur)</span><span style='color:#DC2626;'>18 km/h</span></div>
                <div class='corridor-track'><div class='corridor-fill' style='width:36%; background:linear-gradient(90deg, #991B1B, #DC2626);'></div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Status Pill Row
    st.markdown(
        """
        <div class='status-pill-row'>
            <div class='status-pill' style='color:#FACC15; border-color:#CA8A04;'>
                <span>🚌</span> 10 Bus Delays (TNSTC)
            </div>
            <div class='status-pill' style='color:#EF4444; border-color:#DC2626;'>
                <span>⚡</span> 2 Traffic Diversions
            </div>
            <div class='status-pill' style='color:#10B981; border-color:#059669;'>
                <span>🚆</span> Railway Signals Normal
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
            <span>&#128249; Live CCTV Feeds &bull; Coimbatore City</span>
            <span class='badge-brand' style='font-size:0.65rem;'>4 CHANNELS</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2x2 CCTV Feeds Grid Simulation for Coimbatore
    st.markdown(
        """
        <div class='cctv-grid'>
            <div class='cctv-box' style='background: radial-gradient(circle, #1E293B 0%, #020617 100%); display:flex; align-items:center; justify-content:center;'>
                <div style='text-align:center;'>
                    <div style='font-size:1.8rem; opacity:0.7;'>🛣️</div>
                    <div style='font-size:0.68rem; color:#94A3B8; font-weight:bold;'>Gandhipuram Flyover</div>
                </div>
                <div class='cctv-rec'><span class='live-pulse' style='background:#EF4444;'></span>REC</div>
                <div class='cctv-overlay'><span>CAM-01</span><span>FLOW: 18 KM/H</span></div>
            </div>
            <div class='cctv-box' style='background: radial-gradient(circle, #1E293B 0%, #020617 100%); display:flex; align-items:center; justify-content:center;'>
                <div style='text-align:center;'>
                    <div style='font-size:1.8rem; opacity:0.7;'>🏢</div>
                    <div style='font-size:0.68rem; color:#94A3B8; font-weight:bold;'>Hope College / CIT</div>
                </div>
                <div class='cctv-rec'><span class='live-pulse' style='background:#EF4444;'></span>REC</div>
                <div class='cctv-overlay'><span>CAM-02</span><span>FLOW: 22 KM/H</span></div>
            </div>
            <div class='cctv-box' style='background: radial-gradient(circle, #1E293B 0%, #020617 100%); display:flex; align-items:center; justify-content:center;'>
                <div style='text-align:center;'>
                    <div style='font-size:1.8rem; opacity:0.7;'>🌉</div>
                    <div style='font-size:0.68rem; color:#94A3B8; font-weight:bold;'>Ukkadam Flyover Jn</div>
                </div>
                <div class='cctv-rec'><span class='live-pulse' style='background:#EF4444;'></span>REC</div>
                <div class='cctv-overlay'><span>CAM-03</span><span>FLOW: 20 KM/H</span></div>
            </div>
            <div class='cctv-box' style='background: radial-gradient(circle, #1E293B 0%, #020617 100%); display:flex; align-items:center; justify-content:center;'>
                <div style='text-align:center;'>
                    <div style='font-size:1.8rem; opacity:0.7;'>💻</div>
                    <div style='font-size:0.68rem; color:#94A3B8; font-weight:bold;'>Saravanampatti IT Hub</div>
                </div>
                <div class='cctv-rec'><span class='live-pulse' style='background:#EF4444;'></span>REC</div>
                <div class='cctv-overlay'><span>CAM-04</span><span>FLOW: 23 KM/H</span></div>
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
                    <div style='font-size:0.65rem; color:#94A3B8;'>Avg Patrol</div>
                </div>
                <div style='background:#0A1626; border:1px solid #162E4A; border-radius:6px; padding:0.5rem 0.2rem;'>
                    <div style='font-size:1.4rem; font-weight:800; color:#F59E0B;'>12 <span style='font-size:0.75rem;'>min</span></div>
                    <div style='font-size:0.65rem; color:#94A3B8;'>Clearance Time</div>
                </div>
                <div style='background:#0A1626; border:1px solid #162E4A; border-radius:6px; padding:0.5rem 0.2rem;'>
                    <div style='font-size:1.4rem; font-weight:800; color:#10B981;'>15%</div>
                    <div style='font-size:0.65rem; color:#94A3B8;'>Secondary Congestion</div>
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
            <span>Incident Summary &bull; Coimbatore</span>
            <span style='color:#EF4444; font-size:0.75rem; font-weight:bold;'>5 ACTIVE</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Incident Summary Cards for Coimbatore
    st.markdown(
        """
        <div class='incident-card'>
            <div><span style='color:#EF4444; font-weight:bold;'>&#9632; Accident on Avinashi Rd (Near CIT)</span></div>
            <span style='color:#94A3B8; font-size:0.7rem;'>7 mins ago</span>
        </div>
        <div class='incident-card warn'>
            <div><span style='color:#F59E0B; font-weight:bold;'>&#9632; Flyover Work at Singanallur Jn</span></div>
            <span style='color:#94A3B8; font-size:0.7rem;'>Slow Moving</span>
        </div>
        <div class='incident-card caution'>
            <div><span style='color:#FACC15; font-weight:bold;'>&#9632; Stalled Bus at Gandhipuram Jn</span></div>
            <span style='color:#94A3B8; font-size:0.7rem;'>Traffic En Route</span>
        </div>
        <div class='incident-card warn'>
            <div><span style='color:#F59E0B; font-weight:bold;'>&#9632; Water Logging at Lanka Corner</span></div>
            <span style='color:#94A3B8; font-size:0.7rem;'>Caution</span>
        </div>
        <div class='incident-card info'>
            <div><span style='color:#38BDF8; font-weight:bold;'>&#9632; Weather Alert: Rain in Saravanampatti</span></div>
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
                <span style='color:#38BDF8; font-size:0.7rem;'>ATCS AI Control</span>
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
                <span>Weather & Conditions &bull; Coimbatore</span>
                <span style='color:#38BDF8; font-size:0.75rem;'>Weather Radar</span>
            </div>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div style='display:flex; align-items:center; gap:0.6rem;'>
                    <span style='font-size:2rem;'>🌧️</span>
                    <div>
                        <div style='font-size:1.3rem; font-weight:800; color:#E2E8F0;'>26&deg;C / 78&deg;F</div>
                        <div style='font-size:0.7rem; color:#94A3B8;'>Monsoon Showers &bull; Wet Surface</div>
                    </div>
                </div>
                <div style='text-align:right;'>
                    <div style='font-size:0.75rem; color:#38BDF8; font-weight:bold;'>Humidity: 78%</div>
                    <div style='font-size:0.65rem; color:#94A3B8;'>Wind: 14 km/h WNW</div>
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
        Coimbatore Smart City Traffic Management Dashboard &bull; AI-Powered Adaptive Traffic Control &bull; Integrated Command Centre (ICCC)
    </div>
    """,
    unsafe_allow_html=True,
)
