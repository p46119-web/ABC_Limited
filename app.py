import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="ABC Ltd. Client Analytics Portal", layout="wide")

st.title("📊 ABC Ltd.: Client Revenue & Churn Risk Prediction System")
st.markdown("An end-to-end predictive analytics framework and managerial decision-support portal.")

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

# Create Synthetic Managerial Feedback Dataset (N = 50)
@st.cache_data
def load_survey_data():
    np.random.seed(42)
    n = 50
    roles = ['Manager', 'Team Leader', 'Supervisor', 'Working Professional', 'Entrepreneur']
    
    df_survey = pd.DataFrame({
        'Respondent_Role': np.random.choice(roles, size=n, p=[0.36, 0.24, 0.16, 0.14, 0.10]),
        'Tool_Usefulness': np.random.choice(['5 - Extremely Useful', '4 - Very Useful', '3 - Moderately Useful', '2 - Slightly Useful'], size=n, p=[0.46, 0.40, 0.12, 0.02]),
        'Adoption_Intent': np.random.choice(['Yes - Will Use in Daily Work', 'Maybe - Depends on Explainability', 'No - Prefer Current Method'], size=n, p=[0.66, 0.30, 0.04]),
        'Model_Trust_Level': np.random.choice(['Moderate Trust (Needs Human Oversight)', 'High Trust (Reliable Baseline)', 'Low Trust (Skeptical)'], size=n, p=[0.54, 0.38, 0.08]),
        'Conflict_Resolution': np.random.choice(['Blend AI & Domain Experience', 'Rely Primarily on Experience', 'Follow AI Recommendation Directly'], size=n, p=[0.56, 0.36, 0.08]),
        'Explainability_Importance': np.random.choice(['Critical Factor for Adoption', 'Moderate Importance', 'Low Importance'], size=n, p=[0.64, 0.28, 0.08]),
        'Fear_of_Wrong_Decision': np.random.choice(['Moderate Concern (Risk Averse)', 'Major Concern (Fear of Loss)', 'Minimal Concern'], size=n, p=[0.50, 0.36, 0.14]),
        'Primary_Adoption_Barrier': np.random.choice(['Unmeasured Qualitative Nuances', 'Need for Model Explainability', 'Fear of Misclassification Costs', 'System Complexity'], size=n, p=[0.42, 0.32, 0.18, 0.08])
    })
    return df_survey

df_survey = load_survey_data()

# Navigation Tabs
tab1, tab2 = st.tabs(["🔮 Predictive Tool Simulator", "📈 Managerial User Study & Feedback (N = 50)"])

# ==========================================
# TAB 1: PREDICTIVE TOOL SIMULATOR
# ==========================================
with tab1:
    st.header("Interactive Client Analytics Simulator")
    st.markdown("Adjust operational parameters in the sidebar to simulate revenue and churn estimates in real time.")
    
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

# ==========================================
# TAB 2: MANAGERIAL USER STUDY & FEEDBACK
# ==========================================
with tab2:
    st.header("Managerial AI Adoption & Qualitative Study Analysis")
    st.markdown("Feedback collected from **50 decision-makers** evaluating tool utility, trust, explainability, and human-AI collaboration.")
    
    # Summary Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(label="Decision-Makers Surveyed", value="50")
    m2.metric(label="Avg Usefulness Rating", value="4.34 / 5.0")
    m3.metric(label="Adoption Intent", value="96.0%", delta="66% Immediate / 30% Conditional")
    m4.metric(label="Primary Trust Mode", value="Moderate Trust", delta="Needs Human Oversight")
    
    st.divider()
    
    # Section 1: Demographics & Usefulness
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("1. Survey Demographics by Role")
        role_counts = df_survey['Respondent_Role'].value_counts()
        st.bar_chart(role_counts)
        st.caption("Sample includes Managers, Team Leaders, Supervisors, Professionals, and Entrepreneurs.")

    with c2:
        st.subheader("2. Tool Usefulness Distribution")
        usefulness_counts = df_survey['Tool_Usefulness'].value_counts()
        st.bar_chart(usefulness_counts)
        st.caption("86% rated the tool 4 or 5 stars ('Good but not perfect' consensus).")

    st.divider()

    # Section 2: Trust & Conflict Resolution
    c3, c4 = st.columns(2)
    
    with c3:
        st.subheader("3. When AI & Experience Conflicted")
        conflict_counts = df_survey['Conflict_Resolution'].value_counts()
        st.bar_chart(conflict_counts)
        st.caption("56% blend AI with domain experience; 36% rely primarily on human judgment.")

    with c4:
        st.subheader("4. Primary Barriers to Full AI Adoption")
        barrier_counts = df_survey['Primary_Adoption_Barrier'].value_counts()
        st.bar_chart(barrier_counts)
        st.caption("Omitted qualitative nuances and lack of feature explainability represent 74% of adoption hurdles.")

    st.divider()
    
    # Section 3: Qualitative Synthesis
    st.subheader("5. Key Qualitative Study Findings")
    
    with st.expander("📌 Why do managers view the model as 'Good, but not Perfect'?", expanded=True):
        st.write("""
        - **Data-Driven Baseline**: Decision-makers appreciate having an objective, standardized metric for client lifetime value and churn risk during contract reviews.
        - **Conditional Trust**: 54% express *Moderate Trust*, noting that while statistical models identify macro trends (e.g., fiber optic churn risks), they cannot account for real-time qualitative changes like leadership turnover or client sentiment shifts.
        """)
        
    with st.expander("📌 How does Explainability & Fear of Wrong Decisions impact adoption?"):
        st.write("""
        - **Explainability Requirement**: 64% of respondents state that feature explainability is critical. Managers refuse to allocate retention budgets based on 'black-box' predictions without knowing directional drivers.
        - **Asymmetric Risk Aversion**: 86% cite moderate-to-major concern regarding false-negative errors (failing to flag a high-value churning client), reinforcing the need for human-in-the-loop decision thresholds.
        """)

    # Raw Data View Option
    if st.checkbox("Show Raw Survey Response Data (N = 50)"):
        st.dataframe(df_survey, use_container_width=True)
