"""
pages/5_Geospatial_Analytics.py
─────────────────────────────────
Page 5: Geospatial Analytics.

PEMS-BAY CSV does NOT include verified GPS coordinates.
This page checks for an optional `data/sensor_locations.csv` file.
If found: renders an interactive Folium/PyDeck map.
If not found: shows a sensor comparison chart with a clear explanation.
"""

import sys
from pathlib import Path
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.analytics import load_cleaned_data, sensor_stats
from config.settings import CLEANED_DATA_PATH, BASE_DIR

st.set_page_config(page_title="Geospatial Analytics", page_icon="🗺️", layout="wide")
css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

st.markdown("## 🗺️ Geospatial Analytics")

@st.cache_data(ttl=3600)
def get_data():
    try:
        return load_cleaned_data()
    except Exception as e:
        st.error(f"Data not found: {e}.")
        st.stop()

df = get_data()
stats = sensor_stats(df)

# ── Check for sensor locations file ───────────────────────────────
locations_path = Path(BASE_DIR) / "data" / "sensor_locations.csv"
has_locations = locations_path.exists()

if has_locations:
    try:
        locs = pd.read_csv(locations_path)
        required_cols = {"sensor_id", "latitude", "longitude"}
        if not required_cols.issubset(set(locs.columns)):
            has_locations = False
            st.warning(
                f"`sensor_locations.csv` must have columns: {required_cols}. "
                "Falling back to comparison chart."
            )
        else:
            locs["sensor_id"] = locs["sensor_id"].astype(str)
    except Exception as e:
        has_locations = False
        st.warning(f"Could not read sensor_locations.csv: {e}")

if has_locations:
    # ── MAP MODE ──────────────────────────────────────────────────
    st.success("✅ Sensor location file detected — displaying interactive map.")
    map_df = stats.merge(locs, on="sensor_id", how="inner")
    map_df["congestion_color"] = map_df["mean_speed"].apply(
        lambda s: "#00C851" if s >= 60 else ("#FFD700" if s >= 45 else ("#FF8C00" if s >= 30 else "#FF1744"))
    )

    # Try pydeck first, fallback to plotly scatter_mapbox
    try:
        import pydeck as pdk
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[longitude, latitude]",
            get_fill_color="congestion_color",
            get_radius=200,
            pickable=True,
            auto_highlight=True,
        )
        view_state = pdk.ViewState(
            latitude=map_df["latitude"].mean(),
            longitude=map_df["longitude"].mean(),
            zoom=10, pitch=30,
        )
        st.pydeck_chart(pdk.Deck(
            layers=[layer], initial_view_state=view_state,
            tooltip={"text": "Sensor: {sensor_id}\nAvg Speed: {mean_speed} mph"},
            map_style="mapbox://styles/mapbox/dark-v10",
        ))
    except Exception:
        if hasattr(px, "scatter_map"):
            fig_map = px.scatter_map(
                map_df,
                lat="latitude", lon="longitude",
                color="mean_speed",
                color_continuous_scale=["#FF1744", "#FF8C00", "#FFD700", "#00C851"],
                size_max=15, zoom=9,
                hover_data=["sensor_id", "mean_speed", "min_speed", "max_speed"],
                template="plotly_dark",
            )
            fig_map.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0), map_style="carto-darkmatter")
            st.plotly_chart(fig_map, use_container_width=True)
        elif hasattr(px, "scatter_mapbox"):
            fig_map = px.scatter_mapbox(
                map_df,
                lat="latitude", lon="longitude",
                color="mean_speed",
                color_continuous_scale=["#FF1744", "#FF8C00", "#FFD700", "#00C851"],
                size_max=15, zoom=9,
                hover_data=["sensor_id", "mean_speed", "min_speed", "max_speed"],
                mapbox_style="carto-darkmatter",
                template="plotly_dark",
            )
            fig_map.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.map(map_df.rename(columns={"latitude": "lat", "longitude": "lon"}))
else:
    # ── NO-MAP MODE ──────────────────────────────────────────────
    st.info(
        """
        ⚠️ **No verified GPS coordinates available for PEMS-BAY sensors.**

        The raw `PEMS-BAY.csv` (compressed_data.csv) does not include latitude/longitude data.
        To enable the interactive map:
        1. Obtain verified sensor coordinates (e.g., from the Caltrans PeMS portal).
        2. Create `data/sensor_locations.csv` with columns: `sensor_id`, `latitude`, `longitude`.
        3. Refresh this page.

        Showing a **sensor comparison chart** instead.
        """
    )

    # ── Sensor speed comparison radar / bar ───────────────────────
    tab1, tab2 = st.tabs(["📊 Speed Distribution by Sensor", "🔢 Sensor Statistics Table"])

    with tab1:
        st.markdown("### 📊 Speed Range by Sensor (Top 30)")
        top30 = stats.nlargest(30, "count").sort_values("mean_speed")
        fig_range = go.Figure()
        fig_range.add_trace(go.Bar(
            name="Min Speed",
            x=top30["sensor_id"].astype(str),
            y=top30["min_speed"],
            marker_color="#FF1744", opacity=0.7,
        ))
        fig_range.add_trace(go.Bar(
            name="Avg Speed",
            x=top30["sensor_id"].astype(str),
            y=top30["mean_speed"],
            marker_color="#00D4FF",
        ))
        fig_range.add_trace(go.Bar(
            name="Max Speed",
            x=top30["sensor_id"].astype(str),
            y=top30["max_speed"],
            marker_color="#00C851", opacity=0.7,
        ))
        fig_range.update_layout(
            barmode="overlay",
            height=500,
            template="plotly_dark",
            paper_bgcolor="#0D1B2A", plot_bgcolor="#0F2337",
            xaxis=dict(tickangle=-45, title="Sensor ID", gridcolor="#1E3A5F"),
            yaxis=dict(title="Speed (mph)", gridcolor="#1E3A5F"),
            legend=dict(bgcolor="#0F2337"),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_range, use_container_width=True)

    with tab2:
        st.markdown("### 🔢 All Sensor Statistics")
        n_top = st.slider("Show top N sensors by observation count", 10, 325, 50)
        display = stats.nlargest(n_top, "count").rename(columns={
            "sensor_id": "Sensor", "mean_speed": "Avg Speed",
            "min_speed": "Min", "max_speed": "Max",
            "std_speed": "Std Dev", "count": "Observations",
        })
        st.dataframe(display.set_index("Sensor"), use_container_width=True)
        st.download_button(
            "📥 Download Sensor Stats CSV",
            stats.to_csv(index=False),
            "sensor_stats.csv",
            "text/csv",
        )

st.divider()
st.caption(
    "Geospatial data limitation: PEMS-BAY CSV does not include verified GPS coordinates. "
    "Supply `data/sensor_locations.csv` to enable the interactive map."
)
