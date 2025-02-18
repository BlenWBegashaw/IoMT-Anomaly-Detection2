import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Load dataset
df = pd.read_csv("data/IoMT.csv")

# Data Cleaning
df = df.drop_duplicates().fillna(0)

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(df)

# Train Isolation Forest Model
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
model.fit(X)

# Save the trained model and scaler
joblib.dump(model, "model/isolation_forest.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("✅ Model and scaler saved in 'model/' folder.")
