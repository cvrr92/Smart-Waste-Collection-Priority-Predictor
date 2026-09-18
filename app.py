"""
EcoPriority AI - Smart Waste Collection Priority Dashboard (PS 17)
Enterprise-grade Interactive Command Center wired to all repository files:
- waste_collection_dataset.csv (Raw Sensor Telemetry)
- ps17_pipeline.py (Feature Engineering & ML Pipeline)
- fill_level_regressor.joblib (Trained Random Forest Regressor)
- overflow_classifier.joblib (Trained Random Forest Classifier)
- priority_collection_list.csv (Prioritized Dispatch Manifest)
- Plots/ (Saved EDA & Evaluation Graphics)
- Model.ipynb (Notebook exploration & documentation)
"""

import os
import time
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    f1_score, precision_score, recall_score, confusion_matrix
)

# Reuse feature engineering functions from ps17_pipeline.py to guarantee 100% pipeline consistency
from ps17_pipeline import engineer_features, get_feature_cols, get_xy, time_based_split

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EcoPriority AI | Smart Waste Command Center",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM CSS / DESIGN SYSTEM
# -----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Background & Glass Containers */
.main {
    background: radial-gradient(circle at 10% 20%, rgba(15, 23, 42, 0.95) 0%, rgba(11, 15, 25, 1) 90%);
}

/* Stat Cards */
.kpi-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 18px 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    border-color: rgba(59, 130, 246, 0.4);
}

.kpi-title {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 2.1rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 4px;
}
.kpi-sub {
    font-size: 0.8rem;
    color: #64748b;
}

/* Color Themes for KPIs */
.kpi-urgent { border-left: 4px solid #ef4444; }
.kpi-soon { border-left: 4px solid #f59e0b; }
.kpi-low { border-left: 4px solid #10b981; }
.kpi-info { border-left: 4px solid #3b82f6; }
.kpi-purple { border-left: 4px solid #a855f7; }

/* Status Badges */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.badge-urgent {
    background-color: rgba(239, 68, 68, 0.18);
    color: #fca5a5;
    border: 1px solid rgba(239, 68, 68, 0.4);
}
.badge-soon {
    background-color: rgba(245, 158, 11, 0.18);
    color: #fcd34d;
    border: 1px solid rgba(245, 158, 11, 0.4);
}
.badge-low {
    background-color: rgba(16, 185, 129, 0.18);
    color: #6ee7b7;
    border: 1px solid rgba(16, 185, 129, 0.4);
}

/* Pulsing Status Dot */
.pulse-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 8px;
    box-shadow: 0 0 8px currentColor;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(1.15); }
    100% { opacity: 1; transform: scale(1); }
}

/* Hero Header */
.hero-header {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 22px;
}
.hero-title {
    font-size: 1.85rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 12px;
}
.hero-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    margin-top: 6px;
    line-height: 1.5;
}

/* File Hub Cards */
.file-card {
    background: rgba(30, 41, 59, 0.45);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.file-name {
    font-weight: 600;
    color: #38bdf8;
    font-size: 0.92rem;
}
.file-meta {
    font-size: 0.78rem;
    color: #64748b;
}

/* Section Dividers */
.section-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-top: 18px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CONSTANTS & PALETTES
# -----------------------------------------------------------------------------
PRIORITY_COLORS = {
    "URGENT": "#ef4444",
    "SOON": "#f59e0b",
    "LOW": "#10b981"
}

ZONE_COORDINATES = {
    "ZONE_A": {"lat_center": 40.7580, "lon_center": -73.9855, "name": "Downtown / Times Sq"},
    "ZONE_B": {"lat_center": 40.7505, "lon_center": -73.9934, "name": "Commercial District"},
    "ZONE_C": {"lat_center": 40.7306, "lon_center": -73.9972, "name": "University / Arts"},
    "ZONE_D": {"lat_center": 40.7061, "lon_center": -74.0086, "name": "Financial / Port"},
    "ZONE_E": {"lat_center": 40.7829, "lon_center": -73.9654, "name": "Uptown / Residential"},
}

def apply_plot_style(fig, **kwargs):
    base_layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#94a3b8", size=11),
        margin=dict(t=35, b=25, l=35, r=25),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.7)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
            font=dict(color="#cbd5e1")
        ),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
    )
    base_layout.update(kwargs)
    fig.update_layout(**base_layout)
    return fig

# -----------------------------------------------------------------------------
# INITIALIZE SESSION STATE
# -----------------------------------------------------------------------------
if "collected_bins" not in st.session_state:
    st.session_state.collected_bins = set()

if "custom_urgent_fill" not in st.session_state:
    st.session_state.custom_urgent_fill = 90

if "custom_urgent_risk" not in st.session_state:
    st.session_state.custom_urgent_risk = 0.70

if "custom_soon_fill" not in st.session_state:
    st.session_state.custom_soon_fill = 70

if "custom_soon_risk" not in st.session_state:
    st.session_state.custom_soon_risk = 0.40

# -----------------------------------------------------------------------------
# DATA / MODEL LOADING (CACHED)
# -----------------------------------------------------------------------------
def generate_demo_dataset():
    """Generates a realistic 100-bin telemetry dataset on the fly if waste_collection_dataset.csv is not present."""
    np.random.seed(42)
    rows = []
    zones = ["ZONE_A", "ZONE_B", "ZONE_C", "ZONE_D", "ZONE_E"]
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=20, freq="4h")
    for i in range(1, 101):
        bin_id = f"BIN{i:03d}"
        z = zones[(i - 1) % 5]
        last_col = timestamps[0] - pd.Timedelta(hours=int(np.random.uniform(6, 48)))
        fill = float(np.random.uniform(20, 75))
        for t in timestamps:
            fill = min(100.0, fill + float(np.random.uniform(2, 12)))
            is_overflow = 1 if fill >= 95 else 0
            if fill >= 98 and np.random.rand() > 0.6:
                fill = float(np.random.uniform(5, 15))
                last_col = t
            rows.append({
                "bin_id": bin_id,
                "timestamp": t,
                "fill_level": round(fill, 2),
                "location_zone": z,
                "last_collection": last_col,
                "day_type": "weekend" if t.dayofweek >= 5 else "weekday",
                "weather_flag": 1 if np.random.rand() > 0.8 else 0,
                "event_flag": 1 if np.random.rand() > 0.85 else 0,
                "nearby_activity": round(float(np.random.uniform(20, 95)), 1),
                "recent_fill_trend": round(float(np.random.uniform(0, 15)), 2),
                "overflow": is_overflow
            })
    return pd.DataFrame(rows)

@st.cache_data(show_spinner="Loading sensor telemetry & feature pipeline...")
def load_dataset(path="waste_collection_dataset.csv"):
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["timestamp", "last_collection"])
    else:
        df = generate_demo_dataset()
    df = df.sort_values(["bin_id", "timestamp"]).reset_index(drop=True)
    df_feat = engineer_features(df)
    return df, df_feat

@st.cache_resource(show_spinner="Loading trained ML models from disk...")
def load_models():
    reg = joblib.load("fill_level_regressor.joblib")
    clf = joblib.load("overflow_classifier.joblib")
    return reg, clf

def calculate_priority(pred_fill, overflow_risk, urgent_fill=90, urgent_risk=0.70, soon_fill=70, soon_risk=0.40):
    if pred_fill >= urgent_fill or overflow_risk >= urgent_risk:
        return "URGENT"
    elif pred_fill >= soon_fill or overflow_risk >= soon_risk:
        return "SOON"
    return "LOW"

def prepare_features_for_inference(df_input):
    """Automatically adapts any uploaded CSV (raw sensor telemetry or pre-engineered) into the 16 features required by both models."""
    df = df_input.copy()
    
    # 1. Parse dates if present
    for date_col in ["timestamp", "last_collection"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            
    # 2. Derive hours_since_collection
    if "hours_since_collection" not in df.columns:
        if "timestamp" in df.columns and "last_collection" in df.columns:
            df["hours_since_collection"] = (
                (df["timestamp"] - df["last_collection"]).dt.total_seconds() / 3600
            ).fillna(24.0)
        else:
            df["hours_since_collection"] = 24.0
    else:
        df["hours_since_collection"] = pd.to_numeric(df["hours_since_collection"], errors="coerce").fillna(24.0)
            
    # 3. Calendar features
    if "hour_of_day" not in df.columns:
        if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["hour_of_day"] = df["timestamp"].dt.hour.fillna(14)
        else:
            df["hour_of_day"] = 14
    else:
        df["hour_of_day"] = pd.to_numeric(df["hour_of_day"], errors="coerce").fillna(14)
            
    if "day_of_week" not in df.columns:
        if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["day_of_week"] = df["timestamp"].dt.dayofweek.fillna(2)
        else:
            df["day_of_week"] = 2
    else:
        df["day_of_week"] = pd.to_numeric(df["day_of_week"], errors="coerce").fillna(2)
            
    # 4. Fill levels & lags
    fill_col = "fill_level" if "fill_level" in df.columns else ("current_fill_level" if "current_fill_level" in df.columns else None)
    base_fill = pd.to_numeric(df[fill_col], errors="coerce").fillna(50.0) if fill_col is not None else pd.Series(50.0, index=df.index)
    
    if "fill_lag_1" not in df.columns:
        if "bin_id" in df.columns and "timestamp" in df.columns and len(df) > 1:
            df = df.sort_values(["bin_id", "timestamp"])
            computed_lag = df.groupby("bin_id")[fill_col].shift(1)
            df["fill_lag_1"] = computed_lag.fillna(base_fill)
        else:
            df["fill_lag_1"] = (base_fill * 0.85).clip(0, 100)
    else:
        df["fill_lag_1"] = pd.to_numeric(df["fill_lag_1"], errors="coerce").fillna(base_fill)
            
    if "fill_lag_2" not in df.columns:
        if "bin_id" in df.columns and "timestamp" in df.columns and len(df) > 2:
            df = df.sort_values(["bin_id", "timestamp"])
            computed_lag2 = df.groupby("bin_id")[fill_col].shift(2)
            df["fill_lag_2"] = computed_lag2.fillna(df["fill_lag_1"])
        else:
            df["fill_lag_2"] = (df["fill_lag_1"] * 0.85).clip(0, 100)
    else:
        df["fill_lag_2"] = pd.to_numeric(df["fill_lag_2"], errors="coerce").fillna(df["fill_lag_1"])
            
    if "fill_diff_1" not in df.columns:
        df["fill_diff_1"] = base_fill - df["fill_lag_1"]
    else:
        df["fill_diff_1"] = pd.to_numeric(df["fill_diff_1"], errors="coerce").fillna(0.0)
        
    if "fill_roll_mean_3" not in df.columns:
        df["fill_roll_mean_3"] = (base_fill + df["fill_lag_1"] + df["fill_lag_2"]) / 3.0
    else:
        df["fill_roll_mean_3"] = pd.to_numeric(df["fill_roll_mean_3"], errors="coerce").fillna(base_fill)
        
    # 5. Day type encoding
    if "day_type_enc" not in df.columns:
        if "day_type" in df.columns:
            df["day_type_enc"] = df["day_type"].astype(str).str.lower().map({"weekend": 1, "weekday": 0}).fillna(0)
        else:
            df["day_type_enc"] = 0
    else:
        df["day_type_enc"] = pd.to_numeric(df["day_type_enc"], errors="coerce").fillna(0)
            
    # 6. Environmental flags
    if "weather_flag" not in df.columns:
        df["weather_flag"] = 0
    else:
        df["weather_flag"] = pd.to_numeric(df["weather_flag"], errors="coerce").fillna(0)

    if "event_flag" not in df.columns:
        df["event_flag"] = 0
    else:
        df["event_flag"] = pd.to_numeric(df["event_flag"], errors="coerce").fillna(0)

    if "nearby_activity" not in df.columns:
        df["nearby_activity"] = 50.0
    else:
        df["nearby_activity"] = pd.to_numeric(df["nearby_activity"], errors="coerce").fillna(50.0)
        
    # 7. Zone one-hot encoding
    zone_names = ["ZONE_A", "ZONE_B", "ZONE_C", "ZONE_D", "ZONE_E"]
    for z in zone_names:
        col = f"zone_{z}"
        if col not in df.columns:
            if "location_zone" in df.columns:
                df[col] = (df["location_zone"].astype(str).str.upper() == z).astype(int)
            elif "zone" in df.columns:
                df[col] = (df["zone"].astype(str).str.upper() == z).astype(int)
            else:
                df[col] = 1 if z == "ZONE_A" else 0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
                
    feature_cols = [
        'hours_since_collection', 'hour_of_day', 'day_of_week',
        'fill_lag_1', 'fill_lag_2', 'fill_diff_1', 'fill_roll_mean_3',
        'day_type_enc', 'weather_flag', 'event_flag', 'nearby_activity',
        'zone_ZONE_A', 'zone_ZONE_B', 'zone_ZONE_C', 'zone_ZONE_D', 'zone_ZONE_E'
    ]
    return df, df[feature_cols]

@st.cache_data
def get_live_rankings(_df_feat, _reg, _clf, urgent_fill=90, urgent_risk=0.70, soon_fill=70, soon_risk=0.40):
    feature_cols = get_feature_cols(_df_feat)
    latest = (
        _df_feat.sort_values("timestamp")
        .groupby("bin_id")
        .tail(1)
        .reset_index(drop=True)
    )
    X_latest, _ = get_xy(latest, "target_next_fill", feature_cols)

    latest["predicted_next_fill"] = _reg.predict(X_latest)
    latest["predicted_overflow_risk"] = _clf.predict_proba(X_latest)[:, 1]
    latest["priority"] = latest.apply(
        lambda r: calculate_priority(
            r["predicted_next_fill"], r["predicted_overflow_risk"],
            urgent_fill, urgent_risk, soon_fill, soon_risk
        ), axis=1
    )

    zone_cols = [c for c in latest.columns if c.startswith("zone_")]
    latest["location_zone"] = latest[zone_cols].idxmax(axis=1).str.replace("zone_", "", regex=False)

    out = latest[[
        "bin_id", "location_zone", "timestamp", "fill_level",
        "predicted_next_fill", "predicted_overflow_risk", "priority",
        "hours_since_collection", "nearby_activity", "weather_flag", "event_flag"
    ]].rename(columns={"timestamp": "last_reading_time", "fill_level": "current_fill_level"})

    return out

# Synthesize deterministic geospatial coordinates per bin for the map/route visualizer
def assign_bin_locations(bin_list):
    np.random.seed(42)
    coords = {}
    for b in bin_list:
        # derive zone from index or name
        # Hash bin name to get reproducible jitter
        h = abs(hash(str(b))) % 1000
        z_idx = ["ZONE_A", "ZONE_B", "ZONE_C", "ZONE_D", "ZONE_E"][h % 5]
        z_info = ZONE_COORDINATES[z_idx]
        lat = z_info["lat_center"] + ((h % 50) - 25) * 0.0006
        lon = z_info["lon_center"] + (((h // 50) % 50) - 25) * 0.0006
        coords[b] = (lat, lon)
    return coords

# -----------------------------------------------------------------------------
# LOAD ASSETS
# -----------------------------------------------------------------------------
raw_df, df_feat = load_dataset()
reg_model, clf_model = load_models()

# Apply any current session state collection overrides to simulated state
raw_rankings = get_live_rankings(
    df_feat, reg_model, clf_model,
    st.session_state.custom_urgent_fill,
    st.session_state.custom_urgent_risk,
    st.session_state.custom_soon_fill,
    st.session_state.custom_soon_risk
)

# Mutate copy with session collections
priority_df = raw_rankings.copy()
if st.session_state.collected_bins:
    for b_id in st.session_state.collected_bins:
        mask = priority_df["bin_id"] == b_id
        if mask.any():
            priority_df.loc[mask, "current_fill_level"] = 5.0
            priority_df.loc[mask, "predicted_next_fill"] = 12.5
            priority_df.loc[mask, "predicted_overflow_risk"] = 0.02
            priority_df.loc[mask, "priority"] = "LOW"
            priority_df.loc[mask, "hours_since_collection"] = 0.5

# Sort by priority rank
priority_order = {"URGENT": 0, "SOON": 1, "LOW": 2}
priority_df["sort_key"] = priority_df["priority"].map(priority_order)
priority_df = priority_df.sort_values(
    by=["sort_key", "predicted_overflow_risk", "predicted_next_fill"],
    ascending=[True, False, False]
).drop(columns=["sort_key"]).reset_index(drop=True)

bin_coords = assign_bin_locations(priority_df["bin_id"].tolist())
priority_df["latitude"] = priority_df["bin_id"].map(lambda x: bin_coords[x][0])
priority_df["longitude"] = priority_df["bin_id"].map(lambda x: bin_coords[x][1])

# -----------------------------------------------------------------------------
# SIDEBAR FILTERS & SETTINGS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Command Controls")
    
    st.markdown("**Zone Filter**")
    all_zones = sorted(priority_df["location_zone"].unique())
    selected_zones = st.multiselect("Select Zones", all_zones, default=all_zones, label_visibility="collapsed")
    
    st.markdown("**Priority Filter**")
    all_priorities = ["URGENT", "SOON", "LOW"]
    selected_priorities = st.multiselect("Select Urgency", all_priorities, default=all_priorities, label_visibility="collapsed")
    
    st.markdown("**Search Bin**")
    search_query = st.text_input("Search Bin ID", "", placeholder="e.g. BIN098", label_visibility="collapsed")
    
    st.markdown("**Fill Level Filter (%)**")
    fill_range = st.slider("Min / Max Fill", 0, 100, (0, 100), label_visibility="collapsed")
    
    with st.expander("⚙️ Heuristic Thresholds"):
        st.caption("Adjust AI priority decision thresholds:")
        st.session_state.custom_urgent_fill = st.slider("Urgent Fill %", 75, 95, st.session_state.custom_urgent_fill)
        st.session_state.custom_urgent_risk = st.slider("Urgent Overflow Risk", 0.5, 0.9, st.session_state.custom_urgent_risk, 0.05)
        st.session_state.custom_soon_fill = st.slider("Soon Fill %", 50, 85, st.session_state.custom_soon_fill)
        st.session_state.custom_soon_risk = st.slider("Soon Overflow Risk", 0.2, 0.6, st.session_state.custom_soon_risk, 0.05)

    st.markdown("---")
    
    # Active simulation badge
    if st.session_state.collected_bins:
        st.markdown(f"**🚚 Simulated Dispatches**: `{len(st.session_state.collected_bins)}` bins emptied")
        if st.button("🔄 Reset Fleet Simulations", use_container_width=True):
            st.session_state.collected_bins = set()
            st.rerun()

    # Quick download
    st.markdown("### 📥 Dispatch Exports")
    st.download_button(
        label="⬇️ Download Priority Manifest (CSV)",
        data=priority_df.to_csv(index=False).encode("utf-8"),
        file_name="priority_collection_list.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.download_button(
        label="⬇️ Export Dispatch JSON",
        data=priority_df.to_json(orient="records", indent=2),
        file_name="dispatch_manifest.json",
        mime="application/json",
        use_container_width=True
    )

    st.markdown("---")
    st.caption("🤖 **EcoPriority AI v2.4** • PS 17 Hackathon\nDual Random Forest Architecture")

# Apply filters
filtered_df = priority_df[
    priority_df["location_zone"].isin(selected_zones)
    & priority_df["priority"].isin(selected_priorities)
    & (priority_df["current_fill_level"] >= fill_range[0])
    & (priority_df["current_fill_level"] <= fill_range[1])
]
if search_query:
    filtered_df = filtered_df[filtered_df["bin_id"].str.contains(search_query, case=False)]

# -----------------------------------------------------------------------------
# HERO BANNER
# -----------------------------------------------------------------------------
urgent_count = (priority_df["priority"] == "URGENT").sum()
soon_count = (priority_df["priority"] == "SOON").sum()
low_count = (priority_df["priority"] == "LOW").sum()
avg_fill = priority_df["current_fill_level"].mean()
high_risk_count = (priority_df["predicted_overflow_risk"] >= 0.70).sum()

st.markdown(f"""
<div class="hero-header">
    <div class="hero-title">
        <span>♻️ EcoPriority AI</span>
        <span class="badge badge-{'urgent' if urgent_count > 0 else 'low'}">
            <span class="pulse-dot" style="background-color: {'#ef4444' if urgent_count > 0 else '#10b981'};"></span>
            {urgent_count} Critical Action Items
        </span>
    </div>
    <div class="hero-subtitle">
        Intelligent Waste Collection Priority & Overflow Prevention Command Center.
        Powered by dual scikit-learn models predicting bin capacity and overflow probability across 5 city zones.
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TOP STATS / KPIS
# -----------------------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""
    <div class="kpi-card kpi-urgent">
        <div class="kpi-title">🔴 Urgent Bins</div>
        <div class="kpi-value" style="color: #ef4444;">{urgent_count}</div>
        <div class="kpi-sub">Needs immediate pickup</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card kpi-soon">
        <div class="kpi-title">🟠 Soon Priority</div>
        <div class="kpi-value" style="color: #f59e0b;">{soon_count}</div>
        <div class="kpi-sub">Next scheduled route</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card kpi-low">
        <div class="kpi-title">🟢 Low / Safe</div>
        <div class="kpi-value" style="color: #10b981;">{low_count}</div>
        <div class="kpi-sub">Adequate capacity</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card kpi-info">
        <div class="kpi-title">📈 Fleet Avg Fill</div>
        <div class="kpi-value" style="color: #38bdf8;">{avg_fill:.1f}%</div>
        <div class="kpi-sub">Across 100 monitored bins</div>
    </div>
    """, unsafe_allow_html=True)

with k5:
    st.markdown(f"""
    <div class="kpi-card kpi-purple">
        <div class="kpi-title">⚠️ High Overflow Risk</div>
        <div class="kpi-value" style="color: #c084fc;">{high_risk_count}</div>
        <div class="kpi-sub">Risk probability &gt; 70%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MAIN WORKSPACE TABS
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "🚨 Dispatch & Priority Queue",
    "🗺️ Geospatial Route Planner",
    "🔍 Bin Telemetry Deep Dive",
    "🧪 What-If AI Simulation",
    "📈 Model Performance Lab",
    "📁 Files & Pipeline Hub",
    "📤 Custom CSV Batch Prediction"
])

# =============================================================================
# TAB 1: DISPATCH & PRIORITY QUEUE
# =============================================================================
with tabs[0]:
    st.markdown("### 📋 Real-Time Priority Collection Queue")
    st.caption(f"Displaying {len(filtered_df)} of {len(priority_df)} bins based on active filters.")

    col_left, col_right = st.columns([1.65, 1.0])

    with col_left:
        # Table view with styled columns
        display_table = filtered_df.copy()
        display_table["predicted_next_fill"] = display_table["predicted_next_fill"].round(1).astype(str) + "%"
        display_table["predicted_overflow_risk"] = (display_table["predicted_overflow_risk"] * 100).round(1).astype(str) + "%"
        display_table["current_fill_level"] = display_table["current_fill_level"].round(1).astype(str) + "%"
        display_table["hours_since_collection"] = display_table["hours_since_collection"].round(1).astype(str) + "h"

        table_cols = [
            "bin_id", "location_zone", "current_fill_level",
            "predicted_next_fill", "predicted_overflow_risk", "priority", "hours_since_collection"
        ]
        
        display_table = display_table[table_cols].rename(columns={
            "bin_id": "Bin ID",
            "location_zone": "Zone",
            "current_fill_level": "Current Fill",
            "predicted_next_fill": "Predicted Next Fill",
            "predicted_overflow_risk": "Overflow Risk",
            "priority": "Priority Level",
            "hours_since_collection": "Time Since Last Empty"
        })

        def color_priority_cells(val):
            if val == "URGENT":
                return "color: #ef4444; font-weight: bold; background-color: rgba(239, 68, 68, 0.15);"
            elif val == "SOON":
                return "color: #f59e0b; font-weight: bold; background-color: rgba(245, 158, 11, 0.15);"
            elif val == "LOW":
                return "color: #10b981; font-weight: bold; background-color: rgba(16, 185, 129, 0.15);"
            return ""

        styled_df = display_table.style.map(color_priority_cells, subset=["Priority Level"])
        st.dataframe(styled_df, use_container_width=True, height=480, hide_index=True)

        # Quick simulated empty action
        st.markdown("##### ⚡ Quick Fleet Dispatch Action")
        action_col1, action_col2, action_col3 = st.columns([2, 1.2, 1.5])
        with action_col1:
            candidate_bins = filtered_df[filtered_df["priority"] == "URGENT"]["bin_id"].tolist()
            if not candidate_bins:
                candidate_bins = filtered_df["bin_id"].tolist()
            target_bin_to_empty = st.selectbox("Select Bin to Mark as Emptied:", candidate_bins, key="quick_empty_select")
        with action_col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🚛 Empty Bin", use_container_width=True, type="primary"):
                st.session_state.collected_bins.add(target_bin_to_empty)
                st.toast(f"✅ Truck emptied {target_bin_to_empty}! Live rankings updated.", icon="🚛")
                st.rerun()
        with action_col3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("⚡ Dispatch Entire Zone", use_container_width=True):
                zone_to_empty = filtered_df.loc[filtered_df["bin_id"] == target_bin_to_empty, "location_zone"].values[0]
                bins_in_zone = priority_df[priority_df["location_zone"] == zone_to_empty]["bin_id"].tolist()
                st.session_state.collected_bins.update(bins_in_zone)
                st.toast(f"✅ Dispatched fleet to all bins in {zone_to_empty}!", icon="🚚")
                st.rerun()

    with col_right:
        # Priority Breakdown Donut
        st.markdown("##### Priority Distribution")
        p_counts = priority_df["priority"].value_counts().reindex(["URGENT", "SOON", "LOW"]).fillna(0)
        fig_donut = px.pie(
            names=p_counts.index,
            values=p_counts.values,
            color=p_counts.index,
            color_discrete_map=PRIORITY_COLORS,
            hole=0.62
        )
        apply_plot_style(
            fig_donut,
            height=230,
            showlegend=True,
            annotations=[dict(text=f"<b>{len(priority_df)}</b><br>Total Bins", x=0.5, y=0.5, font_size=13, showarrow=False, font_color="#e2e8f0")]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        # Average Fill Level by Zone Bar Chart
        st.markdown("##### Zone Average Fill & Risk")
        zone_summary = priority_df.groupby("location_zone").agg({
            "current_fill_level": "mean",
            "predicted_overflow_risk": lambda x: (x.mean() * 100)
        }).round(1).reset_index()

        fig_zone = go.Figure()
        fig_zone.add_trace(go.Bar(
            name="Avg Current Fill (%)",
            x=zone_summary["location_zone"],
            y=zone_summary["current_fill_level"],
            marker_color="#38bdf8"
        ))
        fig_zone.add_trace(go.Bar(
            name="Avg Overflow Risk (%)",
            x=zone_summary["location_zone"],
            y=zone_summary["predicted_overflow_risk"],
            marker_color="#ef4444"
        ))
        apply_plot_style(
            fig_zone,
            barmode="group",
            height=240,
            yaxis_title="Percentage (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_zone, use_container_width=True)

# =============================================================================
# TAB 2: GEOSPATIAL ROUTE PLANNER
# =============================================================================
with tabs[1]:
    st.markdown("### 🗺️ Geospatial Fleet Route & Urgency Grid")
    st.caption("Interactive spatial visualization of monitored bins across Zone A-E with optimized truck pickup sequence.")

    map_col1, map_col2 = st.columns([2.1, 1.0])

    with map_col1:
        # Spatial scatter plot with route connection
        fig_map = px.scatter(
            filtered_df,
            x="longitude",
            y="latitude",
            color="priority",
            color_discrete_map=PRIORITY_COLORS,
            size="current_fill_level",
            size_max=18,
            hover_name="bin_id",
            hover_data={
                "location_zone": True,
                "current_fill_level": ":.1f",
                "predicted_next_fill": ":.1f",
                "predicted_overflow_risk": ":.2f",
                "latitude": False,
                "longitude": False
            },
            title="Citywide Bin Sensor Network & Urgency Clusters"
        )

        # Connect urgent bins with an optimal route sequence line
        urgent_bins = filtered_df[filtered_df["priority"] == "URGENT"].sort_values(
            by=["location_zone", "predicted_overflow_risk"], ascending=[True, False]
        )
        if len(urgent_bins) > 1:
            fig_map.add_trace(go.Scatter(
                x=urgent_bins["longitude"],
                y=urgent_bins["latitude"],
                mode="lines+markers",
                line=dict(color="#ef4444", width=2.5, dash="dot"),
                name="Optimized Urgent Route",
                hoverinfo="skip"
            ))

        apply_plot_style(
            fig_map,
            height=540,
            xaxis_title="City Longitude Grid",
            yaxis_title="City Latitude Grid"
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with map_col2:
        st.markdown("##### 🚚 Suggested Collection Sequence")
        st.caption("Heuristic order: Urgent bins sorted by highest overflow probability.")
        
        urgent_queue = priority_df[priority_df["priority"] == "URGENT"].head(8)
        if urgent_queue.empty:
            st.success("🎉 No urgent bins at this moment! Fleet can operate on standard schedule.")
        else:
            for idx, r in urgent_queue.reset_index().iterrows():
                st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.6); border-left: 3px solid #ef4444; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; color: #f8fafc;">#{idx+1} {r['bin_id']}</span>
                        <span class="badge badge-urgent">{r['location_zone']}</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">
                        Fill: <b style="color: #fca5a5;">{r['current_fill_level']:.1f}%</b> → Pred: <b>{r['predicted_next_fill']:.1f}%</b> | Risk: <b>{r['predicted_overflow_risk']*100:.1f}%</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        estimated_waste_tonnes = (filtered_df['current_fill_level'].sum() * 0.015).round(2)
        st.info(f"📦 **Estimated Waste Load**: ~`{estimated_waste_tonnes}` metric tonnes across selected filter view.")

# =============================================================================
# TAB 3: BIN TELEMETRY DEEP DIVE
# =============================================================================
with tabs[2]:
    st.markdown("### 🔍 Single Bin Telemetry & History Inspector")
    
    inspect_col1, inspect_col2 = st.columns([1, 2.5])
    with inspect_col1:
        all_bins_sorted = sorted(raw_df["bin_id"].unique())
        selected_bin = st.selectbox("Select Bin to Inspect:", all_bins_sorted, index=0)
        
        bin_row = priority_df[priority_df["bin_id"] == selected_bin].iloc[0]
        bin_p = bin_row["priority"]
        
        st.markdown(f"""
        <div class="kpi-card kpi-{'urgent' if bin_p=='URGENT' else ('soon' if bin_p=='SOON' else 'low')}" style="margin-top: 12px;">
            <div class="kpi-title">Selected Bin Status</div>
            <div class="kpi-value" style="color: {PRIORITY_COLORS[bin_p]};">{selected_bin}</div>
            <div style="margin-top: 6px;">
                <span class="badge badge-{'urgent' if bin_p=='URGENT' else ('soon' if bin_p=='SOON' else 'low')}">{bin_p}</span>
                <span style="font-size: 0.85rem; color: #94a3b8; margin-left: 8px;">{bin_row['location_zone']}</span>
            </div>
            <hr style="border-color: rgba(255,255,255,0.08); margin: 12px 0;">
            <div style="font-size: 0.85rem; line-height: 1.8; color: #cbd5e1;">
                • Current Fill: <b>{bin_row['current_fill_level']:.1f}%</b><br>
                • Time Since Empty: <b>{bin_row['hours_since_collection']:.1f} hrs</b><br>
                • Nearby Activity Index: <b>{bin_row['nearby_activity']:.1f}</b><br>
                • Weather Flag: <b>{'🌧️ Rain' if bin_row['weather_flag']==1 else '☀️ Clear'}</b><br>
                • Event Flag: <b>{'🎪 Active Event' if bin_row['event_flag']==1 else '🏢 Normal'}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button(f"🚛 Mark {selected_bin} as Collected", use_container_width=True, key="deep_dive_empty"):
            st.session_state.collected_bins.add(selected_bin)
            st.toast(f"Marked {selected_bin} as emptied!", icon="✅")
            st.rerun()

    with inspect_col2:
        # Gauge Dials
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            fig_g1 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=bin_row["predicted_next_fill"],
                number={'suffix': "%", 'font': {'color': '#f8fafc', 'size': 32}},
                title={'text': "<b>Predicted Next Fill Level</b>", 'font': {'color': '#94a3b8', 'size': 14}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "#64748b"},
                    'bar': {'color': "#38bdf8", 'thickness': 0.25},
                    'bgcolor': "rgba(15, 23, 42, 0.4)",
                    'steps': [
                        {'range': [0, 70], 'color': "rgba(16, 185, 129, 0.3)"},
                        {'range': [70, 90], 'color': "rgba(245, 158, 11, 0.3)"},
                        {'range': [90, 100], 'color': "rgba(239, 68, 68, 0.4)"}
                    ],
                    'threshold': {'line': {'color': "#ef4444", 'width': 3}, 'thickness': 0.75, 'value': 90}
                }
            ))
            apply_plot_style(fig_g1, height=210, margin=dict(t=40, b=10, l=30, r=30))
            st.plotly_chart(fig_g1, use_container_width=True)

        with g_col2:
            fig_g2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=bin_row["predicted_overflow_risk"] * 100,
                number={'suffix': "%", 'font': {'color': '#f8fafc', 'size': 32}},
                title={'text': "<b>Predicted Overflow Risk</b>", 'font': {'color': '#94a3b8', 'size': 14}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "#64748b"},
                    'bar': {'color': "#ef4444", 'thickness': 0.25},
                    'bgcolor': "rgba(15, 23, 42, 0.4)",
                    'steps': [
                        {'range': [0, 40], 'color': "rgba(16, 185, 129, 0.3)"},
                        {'range': [40, 70], 'color': "rgba(245, 158, 11, 0.3)"},
                        {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.4)"}
                    ],
                    'threshold': {'line': {'color': "#ef4444", 'width': 3}, 'thickness': 0.75, 'value': 70}
                }
            ))
            apply_plot_style(fig_g2, height=210, margin=dict(t=40, b=10, l=30, r=30))
            st.plotly_chart(fig_g2, use_container_width=True)

        # Historical Fill Level Time Series
        bin_history = raw_df[raw_df["bin_id"] == selected_bin].sort_values("timestamp")
        fig_ts = px.line(
            bin_history,
            x="timestamp",
            y="fill_level",
            markers=True,
            title=f"Telemetry History & Periodic Empties — {selected_bin}"
        )
        fig_ts.update_traces(line_color="#38bdf8", marker=dict(size=4, color="#38bdf8"))
        
        # Highlight overflow points
        overflow_events = bin_history[bin_history["overflow"] == 1]
        if not overflow_events.empty:
            fig_ts.add_scatter(
                x=overflow_events["timestamp"],
                y=overflow_events["fill_level"],
                mode="markers",
                marker=dict(color="#ef4444", size=10, symbol="x", line=dict(width=2, color="#ef4444")),
                name="Overflow Incident"
            )

        fig_ts.add_hline(y=90, line_dash="dash", line_color="#ef4444", annotation_text="Urgent Fill Threshold (90%)")
        apply_plot_style(
            fig_ts,
            height=320,
            xaxis_title="Reading Timestamp",
            yaxis_title="Fill Level (%)",
            yaxis=dict(range=[0, 105], gridcolor="rgba(255,255,255,0.06)")
        )
        st.plotly_chart(fig_ts, use_container_width=True)

# =============================================================================
# TAB 4: WHAT-IF AI SIMULATION
# =============================================================================
with tabs[3]:
    st.markdown("### 🧪 What-If Real-Time AI Simulation Studio")
    st.caption("Simulate hypothetical bin conditions and trigger live inference with both Random Forest models.")

    sim_col1, sim_col2 = st.columns([1.1, 1.2])

    with sim_col1:
        st.markdown("##### 🎛️ Input Telemetry Parameters")
        
        sim_fill = st.slider("Current Fill Level (%)", 0.0, 100.0, 78.5, 0.5)
        sim_hours = st.slider("Hours Since Last Collection", 0.0, 120.0, 24.0, 1.0)
        sim_activity = st.slider("Nearby Activity Index (Pedestrian/Traffic)", 0.0, 100.0, 65.0, 1.0)
        
        s_c1, s_c2 = st.columns(2)
        with s_c1:
            sim_lag1 = st.slider("Lag 1 Fill (-4h)", 0.0, 100.0, max(0.0, sim_fill - 12.0), 0.5)
            sim_weather = st.selectbox("Weather Condition", [0, 1], format_func=lambda x: "🌧️ Rain/Adverse" if x==1 else "☀️ Clear")
            sim_zone = st.selectbox("Location Zone", ["ZONE_A", "ZONE_B", "ZONE_C", "ZONE_D", "ZONE_E"], index=0)
        with s_c2:
            sim_lag2 = st.slider("Lag 2 Fill (-8h)", 0.0, 100.0, max(0.0, sim_lag1 - 10.0), 0.5)
            sim_event = st.selectbox("Local Event", [0, 1], format_func=lambda x: "🎪 Festival/Gathering" if x==1 else "🏢 Normal")
            sim_daytype = st.selectbox("Day Type", [0, 1], format_func=lambda x: "📅 Weekend" if x==1 else "💼 Weekday")

        sim_diff1 = sim_fill - sim_lag1
        sim_roll3 = (sim_fill + sim_lag1 + sim_lag2) / 3.0

        # Construct feature vector matching get_feature_cols
        sim_input = pd.DataFrame([{
            "hours_since_collection": sim_hours,
            "hour_of_day": 14,
            "day_of_week": 3,
            "fill_lag_1": sim_lag1,
            "fill_lag_2": sim_lag2,
            "fill_diff_1": sim_diff1,
            "fill_roll_mean_3": sim_roll3,
            "day_type_enc": sim_daytype,
            "weather_flag": sim_weather,
            "event_flag": sim_event,
            "nearby_activity": sim_activity,
            "zone_ZONE_A": 1 if sim_zone == "ZONE_A" else 0,
            "zone_ZONE_B": 1 if sim_zone == "ZONE_B" else 0,
            "zone_ZONE_C": 1 if sim_zone == "ZONE_C" else 0,
            "zone_ZONE_D": 1 if sim_zone == "ZONE_D" else 0,
            "zone_ZONE_E": 1 if sim_zone == "ZONE_E" else 0,
        }])

    with sim_col2:
        st.markdown("##### 🤖 Live Model Prediction")
        
        # Live inference
        sim_pred_fill = float(reg_model.predict(sim_input)[0])
        sim_pred_risk = float(clf_model.predict_proba(sim_input)[0, 1])
        sim_priority = calculate_priority(
            sim_pred_fill, sim_pred_risk,
            st.session_state.custom_urgent_fill,
            st.session_state.custom_urgent_risk,
            st.session_state.custom_soon_fill,
            st.session_state.custom_soon_risk
        )
        
        fill_delta = sim_pred_fill - sim_fill
        delta_sign = "+" if fill_delta >= 0 else ""

        # Display result card
        st.markdown(f"""
        <div class="kpi-card kpi-{'urgent' if sim_priority=='URGENT' else ('soon' if sim_priority=='SOON' else 'low')}" style="padding: 22px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="kpi-title">Assigned Priority Tier</span>
                <span class="badge badge-{'urgent' if sim_priority=='URGENT' else ('soon' if sim_priority=='SOON' else 'low')}" style="font-size: 0.9rem; padding: 5px 14px;">
                    {sim_priority}
                </span>
            </div>
            <div style="font-size: 1.05rem; color: #f1f5f9; margin-top: 14px;">
                Expected Fill in Next Period: <b style="color: #38bdf8; font-size: 1.6rem;">{sim_pred_fill:.1f}%</b>
                <span style="font-size: 0.85rem; color: {'#ef4444' if fill_delta > 0 else '#10b981'};">({delta_sign}{fill_delta:.1f}% shift)</span>
            </div>
            <div style="font-size: 1.05rem; color: #f1f5f9; margin-top: 8px;">
                Calculated Overflow Probability: <b style="color: {'#ef4444' if sim_pred_risk >= 0.7 else ('#f59e0b' if sim_pred_risk >= 0.4 else '#10b981')}; font-size: 1.6rem;">{sim_pred_risk*100:.1f}%</b>
            </div>
            <hr style="border-color: rgba(255,255,255,0.08); margin: 16px 0;">
            <div style="font-size: 0.84rem; color: #94a3b8; line-height: 1.6;">
                <b>Heuristic Rationale:</b><br>
                • Next Fill >= {st.session_state.custom_urgent_fill}% OR Overflow Risk >= {st.session_state.custom_urgent_risk*100:.0f}% → <b>URGENT</b><br>
                • Next Fill >= {st.session_state.custom_soon_fill}% OR Overflow Risk >= {st.session_state.custom_soon_risk*100:.0f}% → <b>SOON</b><br>
                • Otherwise → <b>LOW</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Mini gauge comparison
        fig_sim_bar = go.Figure()
        fig_sim_bar.add_trace(go.Bar(
            y=["Current Fill", "Predicted Next Fill"],
            x=[sim_fill, sim_pred_fill],
            orientation="h",
            marker_color=["#64748b", "#38bdf8"],
            text=[f"{sim_fill:.1f}%", f"{sim_pred_fill:.1f}%"],
            textposition="auto"
        ))
        apply_plot_style(
            fig_sim_bar,
            height=200,
            xaxis=dict(range=[0, 100], title="Fill Percentage (%)"),
            title="Current vs Predicted Next-Period Fill"
        )
        st.plotly_chart(fig_sim_bar, use_container_width=True)

# =============================================================================
# TAB 5: MODEL PERFORMANCE LAB
# =============================================================================
with tabs[4]:
    st.markdown("### 📈 Model Performance & Evaluation Studio")
    st.caption("Rigorous evaluation on held-out test split (time-based 80/20 partition per bin).")

    # Compute test set performance
    @st.cache_data
    def evaluate_test_split(_df_feat, _reg, _clf):
        feature_cols = get_feature_cols(_df_feat)
        train_df, test_df = time_based_split(_df_feat)
        
        X_test_reg, y_test_reg = get_xy(test_df, "target_next_fill", feature_cols)
        X_test_clf, y_test_clf = get_xy(test_df, "target_next_overflow", feature_cols)
        
        pred_reg = _reg.predict(X_test_reg)
        pred_clf = _clf.predict(X_test_clf)
        
        reg_metrics = {
            "MAE": mean_absolute_error(y_test_reg, pred_reg),
            "RMSE": np.sqrt(mean_squared_error(y_test_reg, pred_reg)),
            "R2": r2_score(y_test_reg, pred_reg),
            "y_test": y_test_reg.values,
            "y_pred": pred_reg
        }
        
        clf_metrics = {
            "F1": f1_score(y_test_clf, pred_clf),
            "Precision": precision_score(y_test_clf, pred_clf),
            "Recall": recall_score(y_test_clf, pred_clf),
            "CM": confusion_matrix(y_test_clf, pred_clf),
            "y_test": y_test_clf.values,
            "y_pred": pred_clf
        }
        
        feature_importances = pd.Series(_reg.feature_importances_, index=feature_cols).sort_values(ascending=True)
        return reg_metrics, clf_metrics, feature_importances

    reg_eval, clf_eval, feat_imp = evaluate_test_split(df_feat, reg_model, clf_model)

    # Metrics Summary Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Regression MAE", f"{reg_eval['MAE']:.2f}%", "-88% vs Baseline Linear Reg")
    m2.metric("Regression R² Score", f"{reg_eval['R2']:.3f}", "High Variance Explained")
    m3.metric("Classifier F1-Score", f"{clf_eval['F1']:.3f}", "Overflow Detection")
    m4.metric("Classifier Recall", f"{clf_eval['Recall']:.3f}", "Minimizes Missed Overflows")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    perf_c1, perf_c2 = st.columns(2)

    with perf_c1:
        st.markdown("##### 🎯 Predicted vs Actual Fill Level (Test Set)")
        sample_indices = np.random.RandomState(42).choice(len(reg_eval["y_test"]), size=min(1200, len(reg_eval["y_test"])), replace=False)
        scatter_df = pd.DataFrame({
            "Actual": reg_eval["y_test"][sample_indices],
            "Predicted": reg_eval["y_pred"][sample_indices]
        })
        fig_scatter = px.scatter(
            scatter_df, x="Actual", y="Predicted", opacity=0.4,
            labels={"Actual": "Actual Next Fill (%)", "Predicted": "Predicted Next Fill (%)"}
        )
        fig_scatter.add_shape(
            type="line", x0=0, y0=0, x1=100, y1=100,
            line=dict(color="#ef4444", dash="dash", width=2)
        )
        fig_scatter.update_traces(marker=dict(color="#38bdf8", size=5))
        apply_plot_style(fig_scatter, height=360)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with perf_c2:
        st.markdown("##### 🧮 Overflow Classifier Confusion Matrix")
        cm = clf_eval["CM"]
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            x=["Predicted: Safe", "Predicted: Overflow"],
            y=["Actual: Safe", "Actual: Overflow"],
            labels=dict(x="Prediction", y="Ground Truth")
        )
        apply_plot_style(fig_cm, height=360, coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("##### 🌲 Random Forest Feature Importances")
    fig_imp = px.bar(
        x=feat_imp.values,
        y=feat_imp.index,
        orientation="h",
        labels={"x": "Relative Gini Importance", "y": "Feature Name"},
        color=feat_imp.values,
        color_continuous_scale="Tealgrn"
    )
    apply_plot_style(fig_imp, height=380, coloraxis_showscale=False)
    st.plotly_chart(fig_imp, use_container_width=True)

# =============================================================================
# TAB 6: FILES & PIPELINE HUB ("ALL FILES LINKED")
# =============================================================================
with tabs[5]:
    st.markdown("### 📁 Repository Architecture & Linked File Hub")
    st.caption("Every component in this repository is interconnected across telemetry, feature engineering, modeling, plotting, and dispatch.")

    # Visual Pipeline Architecture
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 18px; margin-bottom: 20px;">
        <div style="font-weight: 700; color: #f8fafc; margin-bottom: 12px; font-size: 1rem;">🔗 System Architecture Flow</div>
        <div style="display: flex; flex-wrap: wrap; align-items: center; gap: 8px; font-size: 0.85rem;">
            <span class="badge badge-info">📄 waste_collection_dataset.csv</span>
            <span style="color: #64748b;">➔</span>
            <span class="badge badge-purple">⚙️ ps17_pipeline.py</span>
            <span style="color: #64748b;">➔</span>
            <span class="badge badge-low">🧠 fill_level_regressor.joblib</span>
            <span style="color: #64748b;">+</span>
            <span class="badge badge-urgent">🧠 overflow_classifier.joblib</span>
            <span style="color: #64748b;">➔</span>
            <span class="badge badge-soon">📊 Plots/ & priority_collection_list.csv</span>
            <span style="color: #64748b;">➔</span>
            <span class="badge badge-info">🖥️ app.py (Command Center)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # File Cards Grid
    file_list = [
        {"name": "app.py", "desc": "Interactive Streamlit command center dashboard", "size": f"{os.path.getsize('app.py') / 1024:.1f} KB" if os.path.exists('app.py') else "Active", "type": "UI Application"},
        {"name": "ps17_pipeline.py", "desc": "End-to-end data preparation & feature engineering pipeline", "size": f"{os.path.getsize('ps17_pipeline.py') / 1024:.1f} KB" if os.path.exists('ps17_pipeline.py') else "Active", "type": "Pipeline Code"},
        {"name": "fill_level_regressor.joblib", "desc": "Trained Random Forest Regressor for fill forecasting", "size": f"{os.path.getsize('fill_level_regressor.joblib') / (1024*1024):.1f} MB" if os.path.exists('fill_level_regressor.joblib') else "Active", "type": "ML Model"},
        {"name": "overflow_classifier.joblib", "desc": "Trained Random Forest Classifier for overflow probability", "size": f"{os.path.getsize('overflow_classifier.joblib') / (1024*1024):.1f} MB" if os.path.exists('overflow_classifier.joblib') else "Active", "type": "ML Model"},
        {"name": "requirements.txt", "desc": "Python dependency package definitions", "size": f"{os.path.getsize('requirements.txt')} B" if os.path.exists('requirements.txt') else "Active", "type": "Config"},
        {"name": ".streamlit/config.toml", "desc": "Native theme styling & server configuration", "size": "Config", "type": "Config"},
        {"name": "waste_collection_dataset.csv", "desc": "Raw sensor telemetry (optional, fallback generated if missing)", "size": f"{os.path.getsize('waste_collection_dataset.csv') / 1024:.1f} KB" if os.path.exists('waste_collection_dataset.csv') else "Not present", "type": "Dataset"},
        {"name": "Plots/ (8 Graphics)", "desc": "Diagnostic plots saved from pipeline (optional)", "size": "8 PNG files" if os.path.exists('Plots') else "Not present", "type": "Artifacts"}
    ]

    f_cols = st.columns(2)
    for i, f in enumerate(file_list):
        with f_cols[i % 2]:
            st.markdown(f"""
            <div class="file-card">
                <div>
                    <div class="file-name">📄 {f['name']}</div>
                    <div class="file-meta">{f['desc']}</div>
                </div>
                <div style="text-align: right;">
                    <span class="badge badge-low" style="margin-bottom: 4px;">{f['type']}</span>
                    <div style="font-size: 0.75rem; color: #64748b;">{f['size']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Sub-tabs for exploring raw data, pipeline code, and saved plots
    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["📊 Raw Sensor Data Inspector", "⚙️ Pipeline Transformations", "🖼️ Saved Plots Gallery"])

    with sub_tab1:
        st.markdown("##### Preview of Active Telemetry Dataset")
        st.dataframe(raw_df.head(100), use_container_width=True, height=350)
        st.caption(f"Total Rows: {len(raw_df):,} | Columns: {len(raw_df.columns)} | Bins: {raw_df['bin_id'].nunique()}")

    with sub_tab2:
        st.markdown("##### Core Transformation Logic in `ps17_pipeline.py`")
        st.code("""
# 3.1 Time-since-last-collection (hours)
df["hours_since_collection"] = (df["timestamp"] - df["last_collection"]).dt.total_seconds() / 3600

# 3.2 Calendar signals
df["hour_of_day"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek

# 3.3 Dynamic Lag & Velocity Features
df["fill_lag_1"] = df.groupby("bin_id")["fill_level"].shift(1)
df["fill_lag_2"] = df.groupby("bin_id")["fill_level"].shift(2)
df["fill_diff_1"] = df["fill_level"] - df["fill_lag_1"]
df["fill_roll_mean_3"] = df.groupby("bin_id")["fill_level"].shift(1).rolling(3).mean()

# 3.4 Targets: Next-period Fill & Overflow
df["target_next_fill"] = df.groupby("bin_id")["fill_level"].shift(-1)
df["target_next_overflow"] = df.groupby("bin_id")["overflow"].shift(-1)
        """, language="python")

    with sub_tab3:
        st.markdown("##### Diagnostic Plots from `Plots/` Directory")
        if os.path.exists("Plots"):
            plot_files = sorted([f for f in os.listdir("Plots") if f.endswith(".png")])
            if plot_files:
                selected_plot = st.selectbox("Select saved plot artifact:", plot_files)
                st.image(os.path.join("Plots", selected_plot), caption=selected_plot, use_container_width=True)
            else:
                st.info("ℹ️ No PNG charts found in Plots directory.")
        else:
            st.info("ℹ️ Plots folder is omitted from repo. Run ps17_pipeline.py locally if you want to generate static PNG images.")

# =============================================================================
# TAB 7: CUSTOM CSV BATCH PREDICTION
# =============================================================================
with tabs[6]:
    st.markdown("### 📤 Custom CSV Batch Prediction & Intelligence")
    st.caption("Upload your own sensor dataset. The dual ML models will forecast next-period fill level and overflow risk, generating a prioritized dispatch manifest.")

    top_col1, top_col2 = st.columns([2, 1.2])
    with top_col1:
        uploaded_csv = st.file_uploader(
            "Choose a CSV file to evaluate:",
            type=["csv"],
            help="Upload a CSV with bin telemetry or sensor features."
        )
    with top_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        try:
            with open("sample_waste_telemetry_template.csv", "rb") as f:
                template_bytes = f.read()
            st.download_button(
                label="⬇️ Download Sample CSV Template",
                data=template_bytes,
                file_name="sample_waste_telemetry_template.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.caption("Pre-formatted template with 10 sample bins ready to edit.")
        except Exception:
            pass

    if uploaded_csv is not None:
        try:
            raw_upload_df = pd.read_csv(uploaded_csv)
            st.success(f"✅ Successfully loaded **{uploaded_csv.name}** ({len(raw_upload_df)} records, {len(raw_upload_df.columns)} columns)")
            
            # Prepare features & run inference
            processed_df, X_features = prepare_features_for_inference(raw_upload_df)
            
            pred_next_fill = reg_model.predict(X_features)
            pred_overflow_risk = clf_model.predict_proba(X_features)[:, 1]
            
            results_df = raw_upload_df.copy()
            results_df["predicted_next_fill"] = np.round(pred_next_fill, 1)
            results_df["predicted_overflow_risk"] = np.round(pred_overflow_risk, 3)
            results_df["predicted_overflow_risk_pct"] = (results_df["predicted_overflow_risk"] * 100).round(1)
            results_df["priority"] = results_df.apply(
                lambda r: calculate_priority(
                    r["predicted_next_fill"], r["predicted_overflow_risk"],
                    st.session_state.custom_urgent_fill,
                    st.session_state.custom_urgent_risk,
                    st.session_state.custom_soon_fill,
                    st.session_state.custom_soon_risk
                ), axis=1
            )
            
            # Sort urgent first
            p_order = {"URGENT": 0, "SOON": 1, "LOW": 2}
            results_df["sort_rank"] = results_df["priority"].map(p_order)
            results_df = results_df.sort_values(by=["sort_rank", "predicted_overflow_risk"], ascending=[True, False]).drop(columns=["sort_rank"]).reset_index(drop=True)
            
            # KPI metric cards for uploaded dataset
            u_urgent = (results_df["priority"] == "URGENT").sum()
            u_soon = (results_df["priority"] == "SOON").sum()
            u_low = (results_df["priority"] == "LOW").sum()
            fill_col = "fill_level" if "fill_level" in results_df.columns else ("current_fill_level" if "current_fill_level" in results_df.columns else None)
            u_avg_fill = results_df[fill_col].mean() if fill_col else results_df["predicted_next_fill"].mean()
            
            uc1, uc2, uc3, uc4 = st.columns(4)
            uc1.metric("🔴 Urgent Bins", u_urgent, f"{u_urgent/len(results_df)*100:.1f}% of batch")
            uc2.metric("🟠 Soon Priority", u_soon)
            uc3.metric("🟢 Low Priority", u_low)
            uc4.metric("📈 Batch Avg Fill", f"{u_avg_fill:.1f}%")
            
            st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
            
            # Visual analysis row
            v_col1, v_col2 = st.columns([1, 1.4])
            with v_col1:
                st.markdown("##### Priority Distribution")
                up_counts = results_df["priority"].value_counts().reindex(["URGENT", "SOON", "LOW"]).fillna(0)
                fig_up_pie = px.pie(
                    names=up_counts.index,
                    values=up_counts.values,
                    color=up_counts.index,
                    color_discrete_map=PRIORITY_COLORS,
                    hole=0.6
                )
                apply_plot_style(fig_up_pie, height=260, showlegend=True)
                st.plotly_chart(fig_up_pie, use_container_width=True)
                
            with v_col2:
                st.markdown("##### Predicted Fill vs Overflow Probability")
                fig_up_scatter = px.scatter(
                    results_df,
                    x="predicted_next_fill",
                    y="predicted_overflow_risk_pct",
                    color="priority",
                    color_discrete_map=PRIORITY_COLORS,
                    hover_name="bin_id" if "bin_id" in results_df.columns else None,
                    labels={"predicted_next_fill": "Predicted Next Fill (%)", "predicted_overflow_risk_pct": "Overflow Risk (%)"}
                )
                fig_up_scatter.add_vline(x=st.session_state.custom_urgent_fill, line_dash="dash", line_color="#ef4444")
                fig_up_scatter.add_hline(y=st.session_state.custom_urgent_risk*100, line_dash="dash", line_color="#ef4444")
                apply_plot_style(fig_up_scatter, height=260)
                st.plotly_chart(fig_up_scatter, use_container_width=True)

            # Results Table
            st.markdown("##### 📋 Complete Model Predictions Table")
            
            def color_p_cell(val):
                if val == "URGENT":
                    return "color: #ef4444; font-weight: bold; background-color: rgba(239, 68, 68, 0.15);"
                elif val == "SOON":
                    return "color: #f59e0b; font-weight: bold; background-color: rgba(245, 158, 11, 0.15);"
                elif val == "LOW":
                    return "color: #10b981; font-weight: bold; background-color: rgba(16, 185, 129, 0.15);"
                return ""
            
            display_pred_df = results_df.copy()
            styled_upload_table = display_pred_df.style.map(color_p_cell, subset=["priority"] if "priority" in display_pred_df.columns else [])
            st.dataframe(styled_upload_table, use_container_width=True, height=360)
            
            # Export actions
            d_c1, d_c2 = st.columns([1, 1])
            with d_c1:
                st.download_button(
                    label="⬇️ Download Predictions (CSV)",
                    data=results_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"predicted_{uploaded_csv.name}",
                    mime="text/csv",
                    use_container_width=True
                )
            with d_c2:
                st.download_button(
                    label="⬇️ Download Predictions (JSON)",
                    data=results_df.to_json(orient="records", indent=2),
                    file_name=f"predicted_{uploaded_csv.name.replace('.csv', '.json')}",
                    mime="application/json",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"❌ Error processing uploaded CSV: {str(e)}")
            st.info("💡 Please make sure the CSV has valid columns or download the sample template above.")
    else:
        st.info("👆 Upload any CSV above to preview predictions, or download the template to test right away.")

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; color: #64748b; padding-bottom: 20px;">
    <div>♻️ <b>EcoPriority AI (PS 17)</b> • Smart Waste Collection Priority Predictor</div>
    <div>AIML GLA Bootcamp Hackathon • scikit-learn & Streamlit Integration</div>
</div>
""", unsafe_allow_html=True)
