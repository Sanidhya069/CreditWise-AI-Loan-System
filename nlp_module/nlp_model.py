<<<<<<< HEAD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample training data (we'll simulate for now)
texts = [
    "stable income low debt",
    "high income good credit",
    "low income high debt",
    "unstable job no savings",
    "good salary low loan",
    "high debt poor credit score"
]

labels = [
    "LOW",
    "LOW",
    "HIGH",
    "HIGH",
    "LOW",
    "HIGH"
]

# Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train model
model = LogisticRegression()
model.fit(X, labels)

# Function to predict risk from text
def predict_text_risk(user_input):
    user_vec = vectorizer.transform([user_input])
    prediction = model.predict(user_vec)[0]
    return prediction


# Test
if __name__ == "__main__":
    user_text = input("Enter financial description: ")
    result = predict_text_risk(user_text)
=======
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample training data (we'll simulate for now)
texts = [
    "stable income low debt",
    "high income good credit",
    "low income high debt",
    "unstable job no savings",
    "good salary low loan",
    "high debt poor credit score"
]

labels = [
    "LOW",
    "LOW",
    "HIGH",
    "HIGH",
    "LOW",
    "HIGH"
]

# Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train model
model = LogisticRegression()
model.fit(X, labels)

# Function to predict risk from text
def predict_text_risk(user_input):
    user_vec = vectorizer.transform([user_input])
    prediction = model.predict(user_vec)[0]
    return prediction


# Test
if __name__ == "__main__":
    user_text = input("Enter financial description: ")
    result = predict_text_risk(user_text)
>>>>>>> c021548f57cc4f429b0109f72f06e6fb81bd9062
    print(f"Predicted Risk: {result}")