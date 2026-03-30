import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv("data/loan_data.csv")  #Load dataset

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_") # Clean column names (remove spaces and weird prefixes)

print("Columns:", df.columns )# Check columns


target_col = "loan_status" # Target column  # we will fix if needed

for col in df.columns:
    if "loan_status" in col:
        target_col = col

df = df.ffill() # Handle missing values

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# Features & Target
X = df.drop(target_col, axis=1)
y = df[target_col]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Get probabilities
y_prob = model.predict_proba(X_test)

# Convert to risk score (probability of rejection)
risk_scores = (1 - y_prob[:, 1]) * 100

print("\nSample Risk Scores:")
print(risk_scores[:5])

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model
with open("model/loan_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")

#feature importance

print("\n--- Feature Importance Section ---")

importances = model.feature_importances_
features = X.columns

feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nTop 5 Important Features:")
print(feature_importance_df.head(5).to_string())