import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
NUM_ITERATIONS = 2000

# Load dataset
df = pd.read_csv("prompt_injection_dataset.csv")

# Extract inputs and labels
X = df["text"]
y = df["detected"]

# Convert text to numeric vectors
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=NUM_ITERATIONS)
model.fit(X_vec, y)

# Save model and vectorizer together
with open("our_scanner.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)

print("Model trained and saved to our_scanner.pkl")
