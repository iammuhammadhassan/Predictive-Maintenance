import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide"
)

# ==========================================
# LOAD MODELS
# ==========================================
rf = joblib.load("random_forest_model.pkl")
xgb = joblib.load("xgboost_model.pkl")

# ==========================================
# TITLE
# ==========================================
st.title("⚙️ Predictive Maintenance AI Dashboard")
st.markdown("Machine Failure Prediction using ML + Feature Engineering")

# ==========================================
# SIDEBAR INPUTS
# ==========================================
st.sidebar.header("Machine Parameters")

air_temp = st.sidebar.slider("Air Temperature (K)", 295.0, 305.0, 300.0)
process_temp = st.sidebar.slider("Process Temperature (K)", 305.0, 315.0, 310.0)
rpm = st.sidebar.slider("Rotational Speed (RPM)", 1100, 2900, 1500)
torque = st.sidebar.slider("Torque (Nm)", 0.0, 80.0, 40.0)
tool_wear = st.sidebar.slider("Tool Wear (min)", 0, 260, 100)

machine_type = st.sidebar.selectbox("Machine Type", ["H", "M", "L"])
model_choice = st.sidebar.radio("Model", ["Random Forest", "XGBoost"])

# ==========================================
# FEATURE ENGINEERING
# ==========================================
temp_diff = process_temp - air_temp
torque_wear = torque * tool_wear
torque_rpm = torque * rpm

type_L = 1 if machine_type == "L" else 0
type_M = 1 if machine_type == "M" else 0

# ==========================================
# STANDARD INPUT (FOR XGBOOST)
# ==========================================
input_df = pd.DataFrame({
    'Air_temperature_K': [air_temp],
    'Process_temperature_K': [process_temp],
    'Rotational_speed_rpm': [rpm],
    'Torque_Nm': [torque],
    'Tool_wear_min': [tool_wear],
    'Temp_Diff': [temp_diff],
    'Torque_Wear': [torque_wear],
    'Torque_RPM': [torque_rpm],
    'Type_L': [type_L],
    'Type_M': [type_M]
})

# ==========================================
# RF FORMAT CONVERTER
# ==========================================
def convert_for_rf(df):
    return pd.DataFrame({
        'Air temperature [K]': df['Air_temperature_K'],
        'Process temperature [K]': df['Process_temperature_K'],
        'Rotational speed [rpm]': df['Rotational_speed_rpm'],
        'Torque [Nm]': df['Torque_Nm'],
        'Tool wear [min]': df['Tool_wear_min'],
        'Temp_Diff': df['Temp_Diff'],
        'Torque_Wear': df['Torque_Wear'],
        'Torque_RPM': df['Torque_RPM'],
        'Type_L': df['Type_L'],
        'Type_M': df['Type_M']
    })

# ==========================================
# MODEL SELECTION
# ==========================================
if model_choice == "Random Forest":
    model_input = convert_for_rf(input_df)
    model = rf
else:
    model_input = input_df
    model = xgb

# ==========================================
# PREDICTION
# ==========================================
prediction = model.predict(model_input)[0]
probability = model.predict_proba(model_input)[0][1]
health = round((1 - probability) * 100, 2)

# ==========================================
# METRICS
# ==========================================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Failure Probability", f"{probability:.2%}")
col2.metric("Health Score", f"{health}%")
col3.metric("Torque", f"{torque} Nm")
col4.metric("Tool Wear", f"{tool_wear} min")

st.divider()

# ==========================================
# GAUGE
# ==========================================
left, right = st.columns([2, 1])

with left:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Failure Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 80], 'color': "orange"},
                {'range': [80, 100], 'color': "red"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# STATUS PANEL
# ==========================================
with right:
    st.subheader("Machine Status")

    if probability < 0.2:
        st.success("🟢 LOW RISK")
    elif probability < 0.5:
        st.warning("🟡 MEDIUM RISK")
    elif probability < 0.8:
        st.warning("🟠 HIGH RISK")
    else:
        st.error("🔴 CRITICAL RISK")

    st.info(f"Model: {model_choice}")

# ==========================================
# RESULT
# ==========================================
st.divider()

if prediction == 1:
    st.error(f"⚠️ FAILURE PREDICTED ({probability:.2%})")
else:
    st.success(f"✅ MACHINE HEALTHY ({(1 - probability):.2%})")

# ==========================================
# SNAPSHOT
# ==========================================
st.subheader("Machine Snapshot")
st.dataframe(model_input, use_container_width=True)

# ==========================================
# ENGINEERING FEATURES
# ==========================================
st.subheader("Engineering Indicators")

eng_df = pd.DataFrame({
    "Feature": ["Temp Diff", "Torque Wear", "Torque RPM"],
    "Value": [temp_diff, torque_wear, torque_rpm]
})

st.plotly_chart(px.bar(eng_df, x="Feature", y="Value"), use_container_width=True)

# ==========================================
# AI INSIGHTS
# ==========================================
st.subheader("AI Insights")

insights = []

if torque > 50:
    insights.append("High torque detected")
if tool_wear > 150:
    insights.append("High tool wear detected")
if temp_diff > 12:
    insights.append("Possible overheating risk")
if rpm < 1300:
    insights.append("Low RPM under load")

if len(insights) == 0:
    st.success("No major risk detected")
else:
    for i in insights:
        st.warning(i)

# ==========================================
# FOOTER
# ==========================================
st.divider()
st.caption("Predictive Maintenance AI | RF + XGBoost | Feature Engineering")