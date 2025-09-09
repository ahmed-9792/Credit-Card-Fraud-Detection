import pandas as pd
import numpy as np
import joblib
import json
import os

# ------------------------------
# Paths
# ------------------------------
DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

SAMPLE_CSV = os.path.join(DATA_DIR, "sample_transactions.csv")
PREDICTED_CSV = os.path.join(DATA_DIR, "sample_transactions_predicted.csv")
MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/feature_list.json"

# ------------------------------
# Generate sample transactions
# ------------------------------
with open(FEATURES_PATH) as f:
    features = json.load(f)

num_samples = 10  # Number of transactions to generate
data = pd.DataFrame(np.random.randn(num_samples, len(features)), columns=features)

# If Amount is a feature, generate realistic amounts
if "Amount" in data.columns:
    data["Amount"] = np.random.randint(1, 5000, size=num_samples)

# Save sample transactions
data.to_csv(SAMPLE_CSV, index=False)
print(f"Sample transactions saved at: {SAMPLE_CSV}")

# ------------------------------
# Load model and scaler
# ------------------------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ------------------------------
# Prepare data for prediction
# ------------------------------
X = data[features]
X_scaled = scaler.transform(X)

# ------------------------------
# Make predictions
# ------------------------------
predictions = model.predict(X_scaled)
probabilities = model.predict_proba(X_scaled)[:, 1]  # Probability of fraud

data["Fraud_Prediction"] = predictions
data["Fraud_Probability"] = probabilities

# Save predictions
data.to_csv(PREDICTED_CSV, index=False)
print(f"Predictions saved at: {PREDICTED_CSV}")
print(data)
