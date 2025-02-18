import streamlit as st
import pandas as pd
import joblib
import time

# Load trained model and scaler
model = joblib.load("model/isolation_forest.pkl")
scaler = joblib.load("model/scaler.pkl")

st.title("🔴 Real-Time IoMT Security Monitoring")

# Load dataset
file_path = "data/IoMT.csv"
df = pd.read_csv(file_path)

# Initialize empty DataFrame for live anomaly tracking
live_df = pd.DataFrame(columns=df.columns)

for i in range(len(df)):
    # Read new row
    row = df.iloc[i:i+1]  # Extract single-row DataFrame
    scaled_row = scaler.transform(row)  # Scale it
    
    # Predict anomaly
    prediction = model.predict(scaled_row)
    is_anomaly = prediction[0] == -1

    # Append to live DataFrame
    live_df = pd.concat([live_df, row])
    st.dataframe(live_df)  # Display live table

    # Show alert
    if is_anomaly:
        st.error(f"⚠️ Anomaly Detected at index {i}")
    else:
        st.success(f"✅ Normal at index {i}")

    time.sleep(1)  # Simulate real-time updates
