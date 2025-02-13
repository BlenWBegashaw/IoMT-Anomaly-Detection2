import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Ensure the model folder exists
os.makedirs("model", exist_ok=True)

print("Loading the IoMT dataset...")
df = pd.read_csv("data/IoMT.csv")
print("Dataset loaded. Shape:", df.shape)

# Data Cleaning: Fill missing values (adjust strategy as needed)
df_filled = df.fillna(0)

# Feature Scaling: Scale numerical features
scaler = StandardScaler()
X = scaler.fit_transform(df_filled)

# Train an Isolation Forest model for anomaly detection
# Adjust the 'contamination' parameter based on your estimation of anomalies in the data.
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
model.fit(X)

# Save the trained model and scaler for later use
joblib.dump(model, "model/isolation_forest.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print("Anomaly detection model and scaler saved in the model/ folder.")

# (Optional) Print distribution of predictions on the training data
predictions = model.predict(X)  # Returns 1 for normal, -1 for anomaly
unique, counts = np.unique(predictions, return_counts=True)
print("Prediction distribution on training data:", dict(zip(unique, counts)))
