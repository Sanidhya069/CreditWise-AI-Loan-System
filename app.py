import streamlit as st
import pickle
import numpy as np
import pandas as pd

from llm_module.llm_advisor import generate_advice
from nlp_module.nlp_model import predict_text_risk

# Load model
model = pickle.load(open("model/loan_model.pkl", "rb"))

st.title("💰 CreditWise AI Loan System")
st.write("AI-powered system for Loan Prediction, Risk Analysis, and Financial Advice")
st.header("Enter Financial Details")

income = st.number_input("Income (Annual)", value=500000)
loan_amount = st.number_input("Loan Amount", value=200000)
cibil_score = st.slider("CIBIL Score", 300, 900, 700)

text_input = st.text_area("Describe your financial situation")

if st.button("Predict", key="predict_button"):

    # Dummy input structure (simplified)
    assets = income * 0.5

    input_data = np.array([[
        0,          # loan_id
        2,          # dependents
        1,          # education
        0,          # self_employed
        income,
        loan_amount,
        12,         # loan_term
        cibil_score,
        assets,
        0,
        0,
        assets
    ]])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    risk_score = int((1 - probability) * 100)
    
    debt_ratio = loan_amount / income if income != 0 else 999
    
    if income < loan_amount * 0.3 or cibil_score < 500:
        prediction = 0   # Force reject
        risk_score = max(risk_score, 80)

    if debt_ratio > 5:
       prediction = 0
       risk_score = max(risk_score, 85)

    # FORCE APPROVAL
    if cibil_score > 750 and debt_ratio < 0.5:
      prediction = 1
      risk_score = min(risk_score, 30)   

    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

    with col2:
        st.metric("Risk Score", f"{risk_score}%")

    # NLP
    if text_input:
        text_risk = predict_text_risk(text_input)
        st.info(f"🧠 Text Risk: {text_risk}")

    # LLM Advice
    st.subheader("🤖 AI Financial Advice")
    advice = generate_advice(risk_score, income, loan_amount, cibil_score)
    st.text(advice)

    st.divider()
