import streamlit as st
import pandas as pd
import joblib
import time
import os
import numpy as np
import socket
import threading
import uvicorn
from sklearn.metrics import classification_report

# ✅ Load trained models and scaler
try:
    model_rf = joblib.load("model/rf.pkl")
    model_svm = joblib.load("model/svm.pkl")
    print("✅ RandomForest and SVM models loaded.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    raise ValueError("❌ Error: Models not found! Train them first.")

try:
    scaler = joblib.load("model/sr.pkl")
except Exception as e:
    print(f"❌ Error loading scaler: {e}")
    raise ValueError("❌ Error: Scaler not found! Train the model first.")

# ✅ Load and prepare dataset
file_path = "data/IP_Based_Flows_Dataset.csv"
df = pd.read_csv(file_path, nrows=50000)

if 'is_attack' not in df.columns:
    raise ValueError("❌ Dataset must contain 'is_attack' column (0=normal, 1=anomaly).")

X = df.drop(columns=['is_attack'])
y = df['is_attack']

# ✅ Initialize tracking DataFrames
live_df = pd.DataFrame(columns=X.columns)
anomalies_rf = pd.DataFrame(columns=X.columns)
anomalies_svm = pd.DataFrame(columns=X.columns)

# ✅ Detection log for evaluation
detection_log = []

# ✅ Simulate real-time detection with debugging
def detect_anomalies():
    global live_df, anomalies_rf, anomalies_svm, detection_log
    live_df = pd.DataFrame(columns=X.columns)
    anomalies_rf = pd.DataFrame(columns=X.columns)
    anomalies_svm = pd.DataFrame(columns=X.columns)
    detection_log.clear()

    for i in range(len(df)):
        row = df.iloc[i:i+1]
        row_numeric = row.select_dtypes(include=[np.number])  # ✅ match training input

        # Ensure 'is_attack' column is not included during scaling or prediction
        row_numeric = row_numeric.drop(columns=['is_attack'], errors='ignore')

        # Scale the row data for predictions
        try:
            scaled_row = scaler.transform(row_numeric)
        except Exception as e:
            print(f"❌ Error during scaling: {e}")
            continue  # Skip this iteration on error

        # Predictions
        try:
            pred_rf = model_rf.predict(scaled_row)[0]
            pred_svm = model_svm.predict(scaled_row)[0]
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            continue  # Skip this iteration on error

        # Log
        detection_log.append({'index': i, 'true': y.iloc[i], 'RF': pred_rf, 'SVM': pred_svm})

        # Update live data with 'is_attack' column for display only
        row.loc[:, 'is_attack'] = 1 if pred_rf == 1 or pred_svm == 1 else 0
        # Set 'is_attack' as 1 for anomaly, 0 for no anomaly
        live_df = pd.concat([live_df, row[['is_attack']]]) if not live_df.empty else row[['is_attack']].copy()

        # Track anomalies
        if pred_rf == 1:
            anomalies_rf = pd.concat([anomalies_rf, row[['is_attack']]]) if not anomalies_rf.empty else row[['is_attack']].copy()

        if pred_svm == 1:
            anomalies_svm = pd.concat([anomalies_svm, row[['is_attack']]]) if not anomalies_svm.empty else row[['is_attack']].copy()

        # Status message for display
        true_label = y.iloc[i]
        if true_label == 1:
            if pred_rf == 1 or pred_svm == 1:
                status_msg = f"⚠️ True Positive (RF: {pred_rf}, SVM: {pred_svm})"
            else:
                status_msg = f"❌ Missed Anomaly (False Negative) (RF: {pred_rf}, SVM: {pred_svm})"
        elif pred_rf == 1 or pred_svm == 1:
            status_msg = f"❗ False Positive (RF: {pred_rf}, SVM: {pred_svm})"
        else:
            status_msg = "✅ Normal"

        # Yield for Streamlit UI
        time.sleep(0.3)
        print(f"Debug: i={i}, Status={status_msg}")  # Debug message to trace flow
        yield live_df, anomalies_rf, anomalies_svm, status_msg

    # ✅ Save log after simulation
    log_df = pd.DataFrame(detection_log)
    log_df.to_csv("dashboard/detection_log.csv", index=False)
    print("✅ Detection log saved to 'dashboard/detection_log.csv'")

    # ✅ Evaluate models
    y_true = [entry['true'] for entry in detection_log]
    y_rf_pred = [entry['RF'] for entry in detection_log]
    y_svm_pred = [entry['SVM'] for entry in detection_log]

    print("\n📈 RandomForest Evaluation")
    print(classification_report(y_true, y_rf_pred))

    print("\n📈 SVM Evaluation")
    print(classification_report(y_true, y_svm_pred))


# ✅ Streamlit Dashboard UI
st.set_page_config(page_title="IoMT Anomaly Detection", layout="wide")
st.title("🩺 IoMT Anomaly Detection Dashboard")

placeholder = st.empty()

# ✅ Run detection and update UI live
for live_data, rf_anomalies, svm_anomalies, status in detect_anomalies():
    with placeholder.container():
        st.subheader("📊 Live IoMT Data Stream")
        
        # Show only the 'is_attack' column
        st.dataframe(live_data.tail(10))

        st.subheader("🚨 RandomForest Detected Anomalies")
        st.dataframe(rf_anomalies.tail(5))

        st.subheader("⚡ SVM Detected Anomalies")
        st.dataframe(svm_anomalies.tail(5))

        st.subheader("Status")
        # Display status with feedback
        if "✅" in status:
            st.success(status)
        elif "⚠️" in status:
            st.warning(status)
        else:
            st.error(status)

# ✅ Optional: Run FastAPI alongside Streamlit
def run_fastapi():
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.start()


