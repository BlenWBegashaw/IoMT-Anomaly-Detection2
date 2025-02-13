import pandas as pd
import time
import joblib

# Load the saved model and scaler
model = joblib.load("model/isolation_forest.pkl")
scaler = joblib.load("model/scaler.pkl")

# Load the dataset for simulation; choose the same file used for training or a new one
df = pd.read_csv("data/IoMT.csv")
df_clean = df.fillna(0)
X = scaler.transform(df_clean)

# Simulate streaming: iterate through the data with a delay between events
for idx, row in enumerate(X):
    sample = row.reshape(1, -1)
    prediction = model.predict(sample)[0]
    status = "Anomaly" if prediction == -1 else "Normal"
    print(f"Event {idx}: {status}")
    time.sleep(0.1)  # Adjust the delay to simulate real-time traffic
