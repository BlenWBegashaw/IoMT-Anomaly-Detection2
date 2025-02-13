import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Ensure the model folder exists
os.makedirs("model", exist_ok=True)

print("Loading dataset...")
# Use IoMT.csv if it is structured, or use Training_parsed.csv if you parsed it.
df = pd.read_csv("data/IoMT.csv")
# If you parsed a text file, you might use:
# df = pd.read_csv("data/Training_parsed.csv")
print("Dataset loaded. Shape:", df.shape)

# ---- Data Cleaning ----
# 1. Remove duplicate rows (if any)
df = df.drop_duplicates()
# 2. Fill missing values with a default (here, 0; adjust based on your data)
df_clean = df.fillna(0)
# 3. Optionally, drop columns that are irrelevant (adjust as needed)
# df_clean = df_clean.drop(columns=["irrelevant_column_name"])

print("Data cleaning completed. Data shape after cleaning:", df_clean.shape)

# ---- Feature Scaling ----
scaler = StandardScaler()
X = scaler.fit_transform(df_clean)

# ---- Model Training ----
# Train an Isolation Forest for unsupervised anomaly detection.
# Adjust 'contamination' to the expected proportion of anomalies.
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
model.fit(X)

# Save the model and scaler for later use
joblib.dump(model, "model/isolation_forest.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print("Model and scaler saved in the model/ folder.")

# Optional: Display prediction distribution on training data
predictions = model.predict(X)  # 1 = normal, -1 = anomaly
unique, counts = np.unique(predictions, return_counts=True)
print("Prediction distribution:", dict(zip(unique, counts)))

