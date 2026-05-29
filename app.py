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

income = st.number_input(
    "Income (Annual)",
    min_value=0,
    value=500000,
    help="Enter your total pre-tax yearly income.",
)
loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    value=200000,
    help="The total amount of money you are requesting to borrow.",
)
cibil_score = st.slider(
    "CIBIL Score",
    300,
    900,
    700,
    help="Your current credit score. Higher scores lower your risk profile.",
)

loan_term = st.selectbox(
    "Loan Term (Months)",
    options=[12, 36, 60, 120, 180, 360],
    index=5,
    help="Duration of the loan. Standard mortgages are 360 months.",
)

text_input = st.text_area(
    "Describe your financial situation",
    help="Our NLP engine analyzes this text for financial sentiment and risk factors.",
)

if st.button("Predict", key="predict_button"):

    # Dummy input structure (simplified)
    assets = income * 0.5

    input_data = np.array(
        [
            [
                0,  # loan_id
                0,  # dependents (lowered to 0 to remove baseline penalty)
                1,  # education
                0,  # self_employed
                income,
                loan_amount,
                loan_term,  # <--- Dynamic user input instead of hardcoded 12
                cibil_score,
                assets,
                0,
                0,
                assets,
            ]
        ]
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    risk_score = int((1 - probability) * 100)

    debt_ratio = loan_amount / income if income != 0 else 999

    # 1. Negative & Zero Input Guardrail
    if income <= 0 or loan_amount <= 0:
        st.error("⚠️ Please enter valid numbers greater than 0.")
        st.stop()

    # 2. Strict Rejection Rule
    if income < loan_amount * 0.3 or cibil_score < 500:
        prediction = 0  # Force reject
        risk_score = max(risk_score, 80)

    # 3. High Debt Rejection
    if debt_ratio > 5:
        prediction = 0
        risk_score = max(risk_score, 85)

    # 4. THE FIX: Guaranteed Approval for Excellent Profiles
    if cibil_score >= 750 and debt_ratio <= 0.5:
        prediction = 1
        risk_score = min(risk_score, 15)

    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.success("✅ Loan Approved (Low Risk)")
        else:
            st.error("❌ Loan Rejected")

    with col2:
        st.metric(
            "Approval Odds",
            f"{100 - risk_score}%",
            help="Higher is better. Based on your financial profile.",
        )
        # Flip the progress bar so green/full = good (Approval Odds)
        safe_odds = min(100, max(0, 100 - risk_score))
        st.progress(safe_odds / 100)

    # Technical Details Expander (Hides the raw ML metrics)
    with st.expander("🔍 View Technical Assessment Details"):
        st.caption("Raw machine learning metrics used for this decision.")
        st.write(f"**Calculated Risk Score:** {risk_score}%")
        st.write(f"**Debt-to-Income Ratio:** {debt_ratio:.2f}")

        if text_input:
            text_risk = predict_text_risk(text_input)
            st.info(f"🧠 NLP Sentiment Analysis: {text_risk}")

    # LLM Advice
    st.subheader("🤖 AI Financial Advice")
    advice = generate_advice(risk_score, income, loan_amount, cibil_score)

    # Make the advice visually pop based on the risk score
    if risk_score > 70:
        st.error(advice)
    elif risk_score > 40:
        st.warning(advice)
    else:
        st.success(advice)

    st.divider()

    # Application Receipt Export
    st.download_button(
        label="📥 Download AI Assessment Receipt",
        data=f"CREDITWISE AI LOAN ASSESSMENT\n\nIncome: ${income}\nLoan Amount: ${loan_amount}\nCIBIL: {cibil_score}\nLoan Term: {loan_term} Months\n\nAI ADVICE:\n{advice}",
        file_name="creditwise_assessment.txt",
        mime="text/plain",
    )

# Compliance and Legal Disclaimer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.75rem;'>
        <b>Disclaimer:</b> CreditWise AI is an educational demonstration. 
        The financial advice and risk scores provided by this AI model do not constitute official financial, legal, or professional advice. 
        Always consult with a certified financial planner before making loan decisions.
    </div>
    """,
    unsafe_allow_html=True,
)
