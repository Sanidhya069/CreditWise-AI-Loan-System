## 🖥️ Application Preview

<img src="https://github.com/Sanidhya069/CreditWise-AI-Loan-System/blob/main/assests/app.png?raw=true"/>

## 📊 Prediction Result (Approved)
<img src="https://github.com/Sanidhya069/CreditWise-AI-Loan-System/blob/main/assests/approved.png?raw=true"/>

## ⚠️ Prediction Result (Rejected)
<img src="https://github.com/Sanidhya069/CreditWise-AI-Loan-System/blob/main/assests/rejected.png?raw=true"/>

## 🏗️ Project Architecture

- ML Model → Loan prediction  
- NLP Module → Text risk analysis  
- LLM Module → Financial advice generation  
- Streamlit → User Interface

## 📌 Example

**Input:**
- Income: 1,200,000  
- Loan: 200,000  
- CIBIL: 820  

**Output:**
- Loan Approved ✅  
- Risk Score: 30%  

## 🚀 Features

* 📊 Loan Approval Prediction using Machine Learning

---

## 📚 Key Learnings

- Built end-to-end ML system  
- Combined ML with rule-based logic  
- Designed interactive UI using Streamlit  
- Improved model reliability using domain rules

## 🧠 Tech Stack

* Python
* Scikit-learn
* Pandas, NumPy
* Streamlit
* NLP (TF-IDF + Logistic Regression)

---

## 📊 How It Works

1. User inputs financial details (income, loan amount, CIBIL score)
2. ML model predicts loan approval
3. Risk score is calculated
4. NLP analyzes user text input
5. AI generates personalized financial advice

---

## ⚠️ Key Highlight

This system combines:

* Machine Learning
* Rule-based logic
* NLP understanding

to handle real-world financial edge cases effectively.

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📌 Future Improvements

* Improve model with larger dataset
* Add real-time credit APIs
* Deploy on cloud

---

## 🧠 Under the Hood (Decision Logic)
The system doesn't just blindly trust the ML model; it includes hardcoded financial guardrails for safety:
- **Auto-Approval:** CIBIL scores > 750 with a Debt-to-Income ratio < 0.5 bypass standard checks for immediate approval (Low Risk).
- **Auto-Rejection:** CIBIL scores < 500 or extreme Debt-to-Income ratios (> 5.0) trigger immediate rejection and maximum risk scoring.

---

## 👨‍💻 Author

Sanidhya