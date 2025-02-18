import pandas as pd
import joblib
import time

# Load trained model and scaler
model = joblib.load("model/isolation_forest.pkl")
scaler = joblib.load("model/scaler.pkl")

# Load dataset (simulating real-time)
file_path = "data/IoMT.csv"
df = pd.read_csv(file_path)

print("🔴 Real-Time IoMT Threat Detection Started...")

# Simulate live streaming row-by-row
for index, row in df.iterrows():
    data_point = row.values.reshape(1, -1)  # Convert row to 2D array
    scaled_point = scaler.transform(data_point)  # Scale the data
    
    # Predict anomaly
    prediction = model.predict(scaled_point)
    is_anomaly = prediction[0] == -1  # Isolation Forest returns -1 for anomalies

    # Print result with alert
    status = "⚠️ Anomaly Detected!" if is_anomaly else "✅ Normal Operation"
    print(f"New Data: {row.to_dict()} - {status}")

    time.sleep(1)  # Simulate real-time processing
