import streamlit as st
import pandas as pd
import joblib

# Load trained model and feature names
model, feature_names = joblib.load("loan_model.pkl")

st.title("💰 Loan Approval Prediction App")
st.write("Enter applicant details to predict loan approval:")

# User inputs (same as dataset columns)
no_of_dependents = st.number_input("Number of Dependents", min_value=0)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.number_input("Annual Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.selectbox("Loan Term (months)", [12, 36, 60, 120, 180, 240, 360])
cibil_score = st.slider("CIBIL Score", 300, 900, 700)
residential_assets_value = st.number_input("Residential Assets Value", min_value=0)
commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0)
luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0)

# Convert inputs into dataframe
data = pd.DataFrame({
    "no_of_dependents": [no_of_dependents],
    "education": [education],
    "self_employed": [self_employed],
    "income_annum": [income_annum],
    "loan_amount": [loan_amount],
    "loan_term": [loan_term],
    "cibil_score": [cibil_score],
    "residential_assets_value": [residential_assets_value],
    "commercial_assets_value": [commercial_assets_value],
    "luxury_assets_value": [luxury_assets_value],
    "bank_asset_value": [bank_asset_value],
})

# One-hot encoding (must match training)
data = pd.get_dummies(data, drop_first=True)

# Align columns with training features
X = pd.DataFrame(columns=feature_names)
for col in feature_names:
    X[col] = data[col] if col in data else 0

# Prediction
if st.button("Predict Loan Approval"):
    pred = model.predict(X)[0]
    if pred == 1:
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Not Approved")


