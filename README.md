# 💰 CreditWise AI Loan System

An intelligent, AI-driven loan prediction system that evaluates financial risk, processes applicant text via NLP, and generates automated financial advice. Built with Python and Streamlit.

## 🚀 Features
- **Machine Learning Prediction:** Evaluates applicant data against a trained classification model.
- **Dynamic Risk Scoring:** Calculates an adaptive risk percentage based on model probability outputs.
- **NLP Text Analysis:** Scans the applicant's self-reported financial situation for risk indicators.
- **Generative AI Advice:** Provides actionable, personalized financial feedback based on the final risk profile.
- **Interactive Web UI:** Fully accessible Streamlit frontend.

## 🧠 Under the Hood (Decision Logic)
The system doesn't just blindly trust the ML model; it includes hardcoded financial guardrails for safety:
- **Auto-Approval:** CIBIL scores > 750 with a Debt-to-Income ratio < 0.5 bypass standard checks for immediate approval (Low Risk).
- **Auto-Rejection:** CIBIL scores < 500 or extreme Debt-to-Income ratios (> 5.0) trigger immediate rejection and maximum risk scoring.

## 💻 Run Locally
1. Clone the repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt