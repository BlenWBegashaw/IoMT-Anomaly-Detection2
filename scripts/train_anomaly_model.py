import pandas as pd
import joblib
import time
import numpy as np

# Load the trained model and scaler
model = joblib.load("model/isolation_forest.pkl")
scaler = joblib.load("model/scaler.pkl")

# Load the dataset (simulating real-time)
file_path = "data/IoMT.csv"
df = pd.read_csv(file_path)

print("🔴 Real-Time IoMT Threat Detection System Started...")

# Simulate real-time streaming
for index, row in df.iterrows():
    data_point = row.values.reshape(1, -1)  # Convert row to 2D array
    scaled_point = scaler.transform(data_point)  # Scale the data
    
    prediction = model.predict(scaled_point)  # Predict if it's an anomaly
    is_anomaly = prediction[0] == -1  # Isolation Forest returns -1 for anomalies
    
    # Print result with alert
    status = "⚠️ Anomaly Detected!" if is_anomaly else "✅ Normal Operation"
    print(f"New Data Received: {row.to_dict()} - {status}")

    time.sleep(1)  # Simulate real-time delay


