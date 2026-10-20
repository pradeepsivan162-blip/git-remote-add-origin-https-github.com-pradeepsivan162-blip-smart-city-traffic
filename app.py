"""
app.py
──────
Main Streamlit entry point for the
Smart City Traffic Analytics and AI-Based Congestion Prediction System.

Run:
  streamlit run app.py
"""

import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Page configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart City Traffic Analytics",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark-mode CSS injection ────────────────────────────────────────
def load_css():
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ── Sidebar branding ───────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding: 1rem 0;'>
            <h2 style='color:#00D4FF;'>🚦 Smart City</h2>
            <p style='color:#8B9DC3; font-size:0.85rem;'>Traffic Analytics & AI Prediction</p>
            <hr style='border-color:#1E3A5F;'>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### 📋 Navigation")
    st.markdown(
        """
        Use the **Pages** menu above to navigate between:
        - 🏠 Executive Overview
        - 🔍 Traffic Explorer
        - 🤖 AI Prediction
        - 🚨 Congestion Analysis
        - 🗺️ Geospatial Analytics
        - 💡 AI Insights
        - 📊 Model Evaluation
        """,
        unsafe_allow_html=True,
    )

# ── Landing page ───────────────────────────────────────────────────
st.markdown(
    """
    <div style='text-align:center; padding: 3rem 0 2rem;'>
        <h1 style='color:#00D4FF; font-size:3rem;'>🚦 Smart City Traffic Analytics</h1>
        <h3 style='color:#8B9DC3;'>AI-Based Congestion Prediction System</h3>
        <p style='color:#607080; font-size:1rem; max-width:700px; margin:1rem auto;'>
            Analyzing the <strong style='color:#00C851;'>PEMS-BAY</strong> dataset —
            325 sensors, 52,116 timestamps, January–June 2017 (Bay Area, California)
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📡 Sensors",        "325")
with col2:
    st.metric("📅 Timestamps",     "52,116")
with col3:
    st.metric("📊 Total Readings", "~16.9M")
with col4:
    st.metric("📆 Coverage",       "Jan–Jun 2017")

st.divider()

st.markdown(
    """
    ### Getting Started

    | Step | Command | Description |
    |------|---------|-------------|
    | 1 | `python scripts/01_clean_data.py` | Clean & convert dataset |
    | 2 | `python scripts/03_train_models.py` | Train forecasting models |
    | 3 | Navigate pages via the sidebar | Explore the dashboard |

    > **Dataset:** PEMS-BAY — California Bay Area traffic speed readings (mph)
    > at 5-minute intervals. Source: Caltrans PeMS.
    > License: Creative Commons Attribution 4.0.
    """,
    unsafe_allow_html=True,
)
