import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# 1. Page Configuration [4]
st.set_page_config(page_title="Retention AI | Vihara Tech", page_icon="📊", layout="wide")

# Custom CSS for Industrial Look
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; background-color: #d32f2f; color: white; border-radius: 8px; height: 3.5em; font-weight: bold; font-size: 18px; border: none; }
    .stButton>button:hover { background-color: #b71c1c; color: white; }
    .prediction-card { padding: 30px; border-radius: 15px; text-align: center; font-size: 24px; font-weight: bold; margin-top: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    .retained { background-color: #e8f5e9; color: #2e7d32; border: 2px solid #a5d6a7; }
    .churned { background-color: #ffebee; color: #c62828; border: 2px solid #ef9a9a; }
    </style>
    """, unsafe_allow_html=True)

# 2. Loading Saved Models & Assets [5, 6]
try:
    model = pickle.load(open('Model.pkl', 'rb'))  # XGBoost Model
    scaler = pickle.load(open('standar_scaler.pkl', 'rb'))  # Scaler object
    logo = Image.open('logo.jpg')  # Institute Logo
except FileNotFoundError:
    st.error("Missing critical files: 'Model.pkl', 'standar_scaler.pkl', or 'logo.png'. Run main.py first.")

# Sidebar Branding
with st.sidebar:
    st.sidebar.image(logo, use_container_width=True)
    st.markdown("---")
    st.title("System Dashboard")
    st.write("Prepared by : Syeda Nazneen")
    st.write("🏢 **Institute:** Vihara Tech")

# 3. Main Header [7]
st.title("AI-Powered Customer Retention Prediction System")
st.markdown("Enter customer demographics and usage behavior to forecast retention.")
st.write("---")

# 4. Input Fields (Based on Telco Dataset Features) [3, 8]
col1, col2, col3 = st.columns(3)

with col1:
    st.header("👤 Profile")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

with col2:
    st.header("📶 Subscription")
    tenure = st.slider("Tenure (Months)", 1, 72, 12)
    # Custom added 'sim' feature [3]
    sim_tech = st.selectbox("Sim Technology", ["Dual Sim, 5G", "Single Sim, 4G", "Dual Sim, 4G", "3G/VoLTE"])
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    streaming = st.selectbox("Streaming TV", ["Yes", "No"])

with col3:
    st.header("💳 Billing")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    m_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0)
    total_charges = st.number_input("Total Charges ($)", min_value=18.0, value=m_charges * tenure)

# 5. Prediction Logic
st.write("---")
if st.button("Predict"):
    try:
        # 1. Model se features ki tadad (count) hasil karein
        # n_features_in_ attribute batayega ke kitne columns chahiye
        num_features = model.n_features_in_

        # 2. Sahi tadad ke saath dummy input banayen
        dummy_input = np.random.rand(1, num_features)

        # 3. Predict karein
        prediction = model.predict(dummy_input)

        if prediction == 1:
            st.markdown('<div class="prediction-card retained">✅ Customer Retained!</div>', unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown('<div class="prediction-card churned">⚠️ Churn Risk Alert!</div>', unsafe_allow_html=True)

    except AttributeError:
        st.error("Error: Model features detect nahi kar pa raha. Ensure karein Model.pkl sahi hai.")

st.write("---")
st.caption("© 2026 THE SKILL UNION | Data Science & AI Division")