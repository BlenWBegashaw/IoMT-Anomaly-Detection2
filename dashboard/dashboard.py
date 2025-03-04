import gradio as gr
import pandas as pd
import joblib
import time
import os
import numpy as np

# ✅ Load trained models and scaler
try:
    model_rf = joblib.load("model/random_forest.pkl")
    model_svm = joblib.load("model/svm_model.pkl")
    print("✅ RandomForest and SVM models loaded.")
except:
    raise ValueError("❌ Error: Models not found! Train them first.")

scaler = joblib.load("model/scaler.pkl")

# ✅ Load dataset
file_path = "data/IoMT.csv"
df = pd.read_csv(file_path)

# ✅ Initialize empty DataFrames for tracking anomalies
live_df = pd.DataFrame(columns=df.columns)
anomalies_rf = pd.DataFrame(columns=df.columns)
anomalies_svm = pd.DataFrame(columns=df.columns)

# ✅ Function to simulate real-time detection
def detect_anomalies():
    global live_df, anomalies_rf, anomalies_svm
    live_df = pd.DataFrame(columns=df.columns)  # Reset live data
    anomalies_rf = pd.DataFrame(columns=df.columns)  # Reset anomaly tracking
    anomalies_svm = pd.DataFrame(columns=df.columns)

    for i in range(len(df)):
        row = df.iloc[i:i+1]  # Extract a single-row DataFrame
        scaled_row = scaler.transform(row)  # Scale it
        
        # Predict anomaly using both models
        pred_rf = model_rf.predict(scaled_row)[0]
        pred_svm = model_svm.predict(scaled_row)[0]
        
        # Determine anomaly status
        is_anomaly_rf = pred_rf == 1  # Assuming 1 = Anomaly
        is_anomaly_svm = pred_svm == 1

        # Append new row to live DataFrame
        if not live_df.empty:
            live_df = pd.concat([live_df, row])
        else:
            live_df = row.copy()

        # Track anomalies separately
        if is_anomaly_rf:
            anomalies_rf = pd.concat([anomalies_rf, row]) if not anomalies_rf.empty else row.copy()

        if is_anomaly_svm:
            anomalies_svm = pd.concat([anomalies_svm, row]) if not anomalies_svm.empty else row.copy()

        # Status message
        status_msg = "✅ Normal"
        if is_anomaly_rf or is_anomaly_svm:
            status_msg = f"⚠️ Anomaly Detected (RF: {is_anomaly_rf}, SVM: {is_anomaly_svm})"

        # Update Gradio interface
        yield live_df, anomalies_rf, anomalies_svm, status_msg

        time.sleep(3)  # Simulate real-time processing

# ✅ Gradio Interface
iface = gr.Interface(
    fn=detect_anomalies,
    inputs=[],
    outputs=[
        gr.Dataframe(label="📊 Live IoMT Data"),
        gr.Dataframe(label="🚨 RandomForest Detected Anomalies"),
        gr.Dataframe(label="⚡ SVM Detected Anomalies"),
        gr.Textbox(label="Status"),
    ],
    live=True
)

# ✅ Run Gradio with Public Access (for GitHub Codespaces)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # Use Codespace assigned port
    iface.launch(server_name="0.0.0.0", server_port=port, share=True)

