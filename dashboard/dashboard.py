import streamlit as st
import pandas as pd
import time

st.title("🔴 IoMT Security: Real-Time Threat Monitoring")

# Load dataset
file_path = "data/IoMT.csv"
df = pd.read_csv(file_path)

# Initialize empty DataFrame for live updates
live_df = pd.DataFrame(columns=df.columns)

# Stream data in real-time
for i in range(len(df)):
    live_df = pd.concat([live_df, df.iloc[i:i+1]])  # Append new row
    st.dataframe(live_df)  # Show updated data
    time.sleep(1)  # Simulate real-time updates
