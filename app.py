# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

st.set_page_config(page_title="Global EV Supply Chain AI System", layout="wide", initial_sidebar_state="expanded")

# App Header
st.title("🔋 Global EV Supply Chain & Infrastructure Planner")
st.markdown("An end-to-end AI application predicting charging infrastructure demands based on active battery supply metrics.")

# Cache loading files for fast rendering
@st.cache_data
def load_data():
    master_data = pd.read_csv("data/cleaned_master_data.csv")
    plants_data = pd.read_csv("data/battery_plants.csv")
    return master_data, plants_data

master_df, plants_df = load_data()

# Navigation panel
page = st.sidebar.radio("Project Sections", ["📈 Analytics Dashboard", "🔮 AI Demand Predictor"])

if page == "📈 Analytics Dashboard":
    st.header("📊 Global Market Analytics")
    
    # KPI Matrix cards
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Tracked Global Chargers", f"{int(master_df['total_stations'].sum()):,}")
    c2.metric("Active Factory Tech Profiles", f"{plants_df['technology'].nunique()}")
    c3.metric("Max Recorded Battery Demand (MWh)", f"{int(master_df['battery_demand_mwh_est'].max()):,}")
    
    # Visual 1: Fleet vs Stations Scatter plot
    st.subheader("🚙 Fleet Expansion vs Installed Charger Station Capacities")
    fig = px.scatter(master_df, x="ev_stock", y="total_stations", color="country_iso",
                     size="avg_power_kw", hover_name="country_iso", log_x=True, log_y=True,
                     labels={"ev_stock": "Total EV Fleet", "total_stations": "Charging Stations Summary"})
    st.plotly_chart(fig, use_container_width=True)

elif page == "🔮 AI Demand Predictor":
    st.header("🔮 Smart Infrastructure Optimization Simulator")
    st.write("Modify future vehicle profiles using the inputs below to predict required localized operational infrastructure.")
    
    try:
        model = joblib.load("src/infrastructure_model.joblib")
    except FileNotFoundError:
        st.error("❌ Pre-trained model binary file not found! Please execute 'python src/train_model.py' first.")
        st.stop()
        
    # Input panel layout
    with st.form("simulation_parameters"):
        col1, col2 = st.columns(2)
        
        with col1:
            year = st.slider("Target Implementation Year", 2020, 2035, 2026)
            ev_stock = st.number_input("Projected EV Fleet (Vehicles)", min_value=100, value=25000)
            ev_sales = st.number_input("Projected Annual EV Purchases", min_value=10, value=5000)
            
        with col2:
            prev_stations = st.number_input("Current Operational Station Base Count", min_value=5, value=450)
            prev_ev_stock = st.number_input("Current Active EV Stock Fleet", min_value=10, value=20000)
            battery_demand = st.number_input("Target Est. Battery Allocation (MWh)", min_value=0, value=12000)
            
        submit_btn = st.form_submit_button("Run Live Model Inference")
        
    if submit_btn:
        try:
            # Load both model and scaler
            model = joblib.load("src/infrastructure_model.joblib")
            scaler = joblib.load("src/scaler.pkl")
            
            # 1. Arrange inputs into the exact sequence matching your features
            input_vector = [[year, ev_stock, ev_sales, prev_stations, prev_ev_stock, battery_demand]]
            
            # 2. Scale the user input vector before feeding it to KNN
            input_vector_scaled = scaler.transform(input_vector)
            
            # 3. Run prediction
            prediction = model.predict(input_vector_scaled)[0]
            
            st.success("🎉 Machine Learning Inference Matrix Processed via KNN!")
            st.metric(label="🎯 AI Predicted Required Charging Stations", value=f"{int(max(0, prediction)):,} Stations")
            
        except FileNotFoundError:
            st.error("❌ Pre-trained model or scaler file missing! Please execute 'python src/train_model.py' first.")
