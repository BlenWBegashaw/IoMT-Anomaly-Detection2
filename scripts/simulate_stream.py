import pandas as pd
import time

# Load dataset
file_path = "data/IoMT.csv"  # Ensure dataset is in "data" folder
df = pd.read_csv(file_path)

# Simulate live updates
for index, row in df.iterrows():
    print(f"New Data Received: {row.to_dict()}")  # Convert row to dictionary
    time.sleep(1)  # Simulate real-time (adjust speed as needed)
