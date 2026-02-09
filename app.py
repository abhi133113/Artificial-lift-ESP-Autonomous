"""
Leucipa-Style ESP Digital Twin

A "Single Pane of Glass" dashboard for autonomous field management.
Features:
- Field Connect (Map View)
- Production Management (KPIs)
- Prescriptive Analytics (Closed-loop control)
- Dark, High-Contrast UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import joblib
import os
from physics_engine import PhysicsEngine

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Leucipa™ Digital Twin",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS (Leucipa Style) ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Cards/Containers */
    .css-1r6slb0, .css-12w0qpk {
        background-color: #1e2130;
        border: 1px solid #2b2f44;
        border-radius: 8px;
        padding: 20px;
    }
    
    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #161924;
        border: 1px solid #262a3b;
        padding: 15px;
        border-radius: 8px;
    }
    div[data-testid="stMetricLabel"] {
        color: #a0a4b8;
        font-size: 0.9rem;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff;
        font-weight: 600;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #00d4ff;
        color: #000000;
        font-weight: bold;
        border: none;
        border-radius: 4px;
    }
    .stButton button:hover {
        background-color: #00b8dd;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #11141d;
        border-right: 1px solid #2b2f44;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZATION ---
physics = PhysicsEngine()

@st.cache_resource
def load_model():
    if os.path.exists('models/copod_model.joblib'):
        model = joblib.load('models/copod_model.joblib')
        scaler = joblib.load('models/copod_scaler.joblib')
        return model, scaler
    return None, None

model, scaler = load_model()

# --- DATA LOADING ---
@st.cache_data
def load_data():
    if os.path.exists('data/real_pump_cleaned.csv'):
        return pd.read_csv('data/real_pump_cleaned.csv')
    return pd.DataFrame()

df_full = load_data()

# --- SESSION STATE ---
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'hz' not in st.session_state:
    st.session_state.hz = 60.0
if 'mode' not in st.session_state:
    st.session_state.mode = "Autonomous"

# --- SIDEBAR (Navigation) ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Baker_Hughes_logo.svg/2560px-Baker_Hughes_logo.svg.png", width=150)
    st.markdown("### Leucipa™ Field Connect")
    st.markdown("---")
    
    st.markdown("**Asset Hierarchy**")
    st.selectbox("Region", ["Permian Basin", "Eagle Ford", "Bakken"])
    st.selectbox("Field", ["Wolfcamp A", "Wolfcamp B"])
    well_id = st.selectbox("Well ID", ["ESP-2024-001", "ESP-2024-002", "ESP-2024-003"])
    
    st.markdown("---")
    st.markdown("**Simulation Controls**")
    scenario = st.selectbox("Scenario", ["Normal Operation", "Gas Lock Event"])
    
    if st.button("Start/Stop Simulation"):
        st.session_state.simulation_running = not st.session_state.simulation_running

# --- MAIN LAYOUT ---

# 1. HEADER & KPI STRIP (Production Management)
col_logo, col_title, col_mode = st.columns([1, 4, 2])
with col_title:
    st.title(f"Well: {well_id}")
    st.caption("🟢 ONLINE | Last Update: Just now")
with col_mode:
    st.markdown(f"### Mode: **{st.session_state.mode}**")

# KPI Strip
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Net Oil", "1,245 bbl/d", "+12")
kpi2.metric("Water Cut", "62.4 %", "-0.5")
kpi3.metric("Gas/Oil Ratio", "450 scf/stb", "+20")
kpi4.metric("Pump Uptime", "98.5 %", "0.0")
kpi5.metric("Est. Revenue", "$93,375 /d", "+$900")

st.markdown("---")

# 2. MAIN CONTENT GRID
row1_col1, row1_col2 = st.columns([2, 1])

# Select data based on scenario
if not df_full.empty:
    if scenario == "Normal Operation":
        df = df_full[df_full['machine_status'] == 'NORMAL'].sample(n=500, random_state=42).sort_index()
    else:
        # Failure event window
        df = df_full.iloc[950:1150].reset_index(drop=True)
else:
    df = pd.DataFrame()

# Simulation Logic
if st.session_state.simulation_running and not df.empty:
    placeholder = st.empty()
    
    features = ['PIP', 'Discharge_Press', 'Amps', 'Temp', 'Vibration',
                'PIP_rolling_mean', 'PIP_rolling_std',
                'Amps_rolling_mean', 'Amps_rolling_std',
                'Vibration_rolling_mean', 'Vibration_rolling_std']
    
    for i in range(len(df)):
        latest = df.iloc[i]
        
        # AI & Physics
        input_data = latest[features].values.reshape(1, -1)
        score = 0.0
        is_anomaly = 0
        if model and scaler:
            input_scaled = scaler.transform(input_data)
            score = model.decision_function(input_scaled)[0]
            is_anomaly = model.predict(input_scaled)[0]
            
        deg = physics.check_health(latest['PIP'], latest['Discharge_Press'], latest['Flow_Rate'])
        
        # Autonomous Logic
        status_msg = "Optimal"
        status_color = "green"
        recommendation = "Maintain current settings"
        
        if is_anomaly or deg > 0.15:
            status_msg = "Deviation Detected"
            status_color = "red"
            recommendation = "Reduce Frequency to 55Hz to mitigate Gas Lock"
            if st.session_state.mode == "Autonomous":
                st.session_state.hz = 55.0
        else:
            st.session_state.hz = 60.0

        with placeholder.container():
            # --- ROW 1: Charts & Map ---
            c1, c2 = st.columns([2, 1])
            
            with c1:
                st.subheader("Real-Time Trends")
                # Dual axis chart
                fig, ax1 = plt.subplots(figsize=(10, 3.5))
                ax1.set_facecolor('#1e2130')
                fig.patch.set_facecolor('#1e2130')
                
                # Plot window
                window = 60
                start = max(0, i - window)
                data_window = df.iloc[start:i+1]
                
                color = '#00d4ff'
                ax1.set_xlabel('Time', color='white')
                ax1.set_ylabel('PIP (psi)', color=color)
                ax1.plot(data_window.index, data_window['PIP'], color=color, linewidth=2)
                ax1.tick_params(axis='y', labelcolor=color, colors='white')
                ax1.tick_params(axis='x', colors='white')
                
                ax2 = ax1.twinx()
                color = '#ff0055'
                ax2.set_ylabel('Amps (A)', color=color)
                ax2.plot(data_window.index, data_window['Amps'], color=color, linewidth=2)
                ax2.tick_params(axis='y', labelcolor=color, colors='white')
                
                ax1.grid(True, alpha=0.1)
                st.pyplot(fig)
                
            with c2:
                st.subheader("Field Map")
                # Simulated map data
                map_data = pd.DataFrame({
                    'lat': [31.5],
                    'lon': [-102.5],
                    'status': [status_color]
                })
                st.map(map_data, zoom=10, size=500, color='status')
                
                st.info(f"📍 **Location**: Permian Basin\n\n📡 **Comms**: 98% Signal")

            # --- ROW 2: Prescriptive Analytics (The "Leucipa" Value) ---
            st.markdown("---")
            st.subheader("💡 Prescriptive Analytics & Insights")
            
            pa1, pa2, pa3 = st.columns(3)
            
            with pa1:
                st.markdown(f"#### Asset Health: :{status_color}[{status_msg}]")
                st.metric("Anomaly Score", f"{score:.2f}", delta="Critical" if is_anomaly else "Normal", delta_color="inverse")
                st.progress(min(1.0, max(0.0, score/5.0))) # Normalize score roughly
                
            with pa2:
                st.markdown("#### Recommendation")
                st.warning(f"**{recommendation}**")
                st.markdown(f"**Current Frequency**: {st.session_state.hz} Hz")
                if st.button("Execute Intervention", disabled=(st.session_state.mode=="Autonomous")):
                    st.toast("Command Sent to VSD!")
            
            with pa3:
                st.markdown("#### Physics Model")
                st.metric("Pump Efficiency", f"{(1-deg)*100:.1f}%", f"-{deg*100:.1f}%")
                st.caption("Based on Manufacturer Curve comparison")

            time.sleep(0.1)

else:
    # Static View (Placeholder)
    st.info("Select a scenario and click 'Start Simulation' in the sidebar.")
    
    # Show static map
    map_data = pd.DataFrame({'lat': [31.5], 'lon': [-102.5]})
    st.map(map_data, zoom=10)
