import streamlit as st

import pandas as pd

import numpy as np

import shap

import joblib


# Load the trained model and scaler

model = joblib.load('loan_default_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Loan Default Predictor", layout="centered")
st.title("🏦 Loan Default Risk Prediction with Explainability")

# 📌 Collecting user input
st.header("Enter Applicant Details")

RevolvingUtilizationOfUnsecuredLines = st.slider('Revolving Utilization of Unsecured Lines', 0.0, 1.0, 0.1)
age = st.slider('Age', 18, 100, 35)
DebtRatio = st.slider('Debt Ratio', 0.0, 10.0, 1.0)
MonthlyIncome = st.number_input('Monthly Income', min_value=0, value=5000)
NumberOfOpenCreditLinesAndLoans = st.slider('Number of Open Credit Lines and Loans', 0, 30, 5)
NumberOfTimes90DaysLate = st.slider('Number of Times 90 Days Late', 0, 10, 0)
NumberRealEstateLoansOrLines = st.slider('Number of Real Estate Loans or Lines', 0, 10, 1)
NumberOfTime60_89DaysPastDueNotWorse = st.slider('Number of Times 60-89 Days Past Due Not Worse', 0, 10, 0)
NumberOfDependents = st.slider('Number of Dependents', 0, 10, 0)

# ✨ Format input data
input_data = np.array([[
    RevolvingUtilizationOfUnsecuredLines,
    age,
    DebtRatio,
    MonthlyIncome,
    NumberOfOpenCreditLinesAndLoans,
    NumberOfTimes90DaysLate,
    NumberRealEstateLoansOrLines,
    NumberOfTime60_89DaysPastDueNotWorse,
    NumberOfDependents
]])

# 🧠 Make prediction
if st.button("Predict Loan Default Risk"):
    try:
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)[0][1]

        if prediction[0] == 1:
            st.error(f"❌ High Risk: Applicant is likely to default. (Probability: {prediction_proba:.2f})")
        else:
            st.success(f"✅ Low Risk: Applicant is unlikely to default. (Probability: {prediction_proba:.2f})")

        # 💡 Explainability using SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(scaled_input)
        st.subheader("Feature Contributions to Prediction")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.initjs()
        shap.force_plot(explainer.expected_value[1], shap_values[1], scaled_input, matplotlib=True, show=False)
        import matplotlib.pyplot as plt
        plt.tight_layout()
        st.pyplot(bbox_inches='tight')

    except Exception as e:
        st.error(f"⚠️ Error: {e}")