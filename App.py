import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model, scaler, features
model = joblib.load('patient_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('feature_names.pkl')

# Page config
st.set_page_config(page_title="Patient Risk Monitor", 
                   page_icon="🏥", layout="wide")

# Title
st.title("🏥 AI-Powered Patient Vital Monitoring System")
st.markdown("Enter patient vitals below to get a real-time deterioration risk score.")
st.divider()

# Input form
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 100, 65)
    heart_rate_max = st.number_input("Heart Rate Max (bpm)", 40, 200, 110)
    heart_rate_min = st.number_input("Heart Rate Min (bpm)", 30, 180, 70)
    sysbp_max = st.number_input("Systolic BP Max (mmHg)", 60, 250, 140)
    sysbp_min = st.number_input("Systolic BP Min (mmHg)", 50, 220, 90)

with col2:
    resprate_max = st.number_input("Resp Rate Max (bpm)", 5, 60, 22)
    resprate_min = st.number_input("Resp Rate Min (bpm)", 4, 50, 12)
    spo2_max = st.number_input("SpO2 Max (%)", 70, 100, 99)
    spo2_min = st.number_input("SpO2 Min (%)", 60, 100, 94)
    temp_max = st.number_input("Temperature Max (°C)", 34.0, 42.0, 37.5)

with col3:
    temp_min = st.number_input("Temperature Min (°C)", 33.0, 41.0, 36.5)
    glucose_max = st.number_input("Glucose Max (mg/dL)", 50, 600, 140)
    map_apache = st.number_input("Mean Arterial Pressure", 20, 180, 80)
    ventilated = st.selectbox("Ventilated?", [0, 1], 
                               format_func=lambda x: "Yes" if x == 1 else "No")

st.divider()

# Predict button
if st.button("🔍 Calculate Risk Score", type="primary"):
    
    # Build input
    hr_var = heart_rate_max - heart_rate_min
    bp_var = sysbp_max - sysbp_min
    shock_idx = heart_rate_max / max(sysbp_min, 1)
    spo2_drop = spo2_max - spo2_min
    temp_var = temp_max - temp_min

    input_data = pd.DataFrame([[
        age, heart_rate_max, heart_rate_min,
        sysbp_max, sysbp_min, resprate_max, resprate_min,
        spo2_max, spo2_min, temp_max, temp_min,
        glucose_max, map_apache, ventilated,
        hr_var, bp_var, shock_idx, spo2_drop, temp_var
    ]], columns=features)

    input_scaled = scaler.transform(input_data)
    probability = model.predict_proba(input_scaled)[0][1]
    risk_score = int(probability * 100)

    # Display result
    col_a, col_b = st.columns(2)

    with col_a:
        st.metric("Risk Score", f"{risk_score}/100")
        st.metric("Mortality Probability", f"{probability:.1%}")

    with col_b:
        if risk_score < 30:
            st.success("🟢 STABLE — Patient vitals within acceptable range")
        elif risk_score < 60:
            st.warning("🟡 MONITOR — Elevated risk, increased observation recommended")
        elif risk_score < 80:
            st.error("🟠 ALERT — High risk, clinical review required immediately")
        else:
            st.error("🔴 CRITICAL — Extreme risk, immediate intervention required")