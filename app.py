import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="ABC Ltd. Client Analytics Portal", layout="wide")

st.title("📊 ABC Ltd.: Client Revenue & Churn Risk Prediction Tool")
st.markdown("Decision support tool for account executives and operational leaders.")

# Load Serialized Assets
@st.cache_resource
def load_assets():
    lin_model = pickle.load(open("model_linear.sav", "rb"))
    log_model = pickle.load(open("model_logistic.sav", "rb"))
    scaler_std = pickle.load(open("scaler_std.sav", "rb"))
    scaler_minmax = pickle.load(open("scaler_minmax.sav", "rb"))
    feature_cols = pickle.load(open("model_features.sav", "rb"))
    return lin_model, log_model, scaler_std, scaler_minmax, feature_cols

lin_model, log_model, scaler_std, scaler_minmax, feature_cols = load_assets()

# Sidebar Input Controls
st.sidebar.header("Client Operational Parameters")
tenure = st.sidebar.slider("Account Tenure (Months)", 0, 72, 24)
monthly_charges = st.sidebar.slider("Monthly Billing Amount (USD)", 18.0, 120.0, 65.0)
contract_type = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_type = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
tech_support = st.sidebar.selectbox("Tech Support Included", ["No", "Yes"])

# Construct Input Vector
input_dict = {f: 0 for f in feature_cols}

tenure_scaled_val = scaler_minmax.transform(pd.DataFrame([[tenure]], columns=['tenure']))[0][0]
monthly_std_val = scaler_std.transform(pd.DataFrame([[monthly_charges]], columns=['MonthlyCharges']))[0][0]

input_dict['tenure'] = tenure
input_dict['MonthlyCharges'] = monthly_charges
input_dict['Tenure_Scaled'] = tenure_scaled_val
input_dict['MonthlyCharges_Std'] = monthly_std_val

if f'Contract_{contract_type}' in input_dict:
    input_dict[f'Contract_{contract_type}'] = 1
if f'InternetService_{internet_type}' in input_dict:
    input_dict[f'InternetService_{internet_type}'] = 1
if f'PaymentMethod_{payment_method}' in input_dict:
    input_dict[f'PaymentMethod_{payment_method}'] = 1
if f'TechSupport_{tech_support}' in input_dict:
    input_dict[f'TechSupport_{tech_support}'] = 1

df_single = pd.DataFrame([input_dict])[feature_cols]

# Generate Predictions
pred_revenue = lin_model.predict(df_single)[0]
churn_prob = log_model.predict_proba(df_single)[0][1]

# Display Metrics
col1, col2 = st.columns(2)

with col1:
    st.subheader("Predicted Lifetime Revenue")
    st.metric(label="Expected Total Value (USD)", value=f"${max(0, pred_revenue):,.2f}")
    if pred_revenue < 1000:
        st.warning("Low Value Account: High-touch interventions recommended.")
    else:
        st.success("High Value Key Account.")

with col2:
    st.subheader("Churn Risk Assessment")
    st.metric(label="Churn Probability", value=f"{churn_prob * 100:.1f}%")
    if churn_prob > 0.30:
        st.error("ACTION REQUIRED: High Risk of Churn (Above 30% Threshold)!")
    else:
        st.info("Account Healthy: Standard Engagement Schedule.")
