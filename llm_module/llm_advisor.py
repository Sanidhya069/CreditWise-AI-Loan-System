def generate_advice(risk_score, income, loan_amount, cibil_score):
    
    if risk_score > 70:
        return f"""
High Risk Detected ⚠️

Your loan application is likely to be rejected due to:
- High risk score ({risk_score}%)
- Possible low income or high loan burden

Suggestions:
- Reduce loan amount
- Improve CIBIL score (current: {cibil_score})
- Ensure stable income source
"""

    elif risk_score > 40:
        return f"""
Moderate Risk ⚠️

Your loan may be approved but with caution.

Suggestions:
- Maintain stable income
- Try to reduce liabilities
- Improve credit behavior
"""

    else:
        return f"""
Low Risk ✅

Your financial profile looks strong.

Suggestions:
- You are likely eligible for loan approval
- Maintain your current financial discipline
"""