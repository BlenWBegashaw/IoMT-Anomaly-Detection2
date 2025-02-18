import pandas as pd
import paho.mqtt.client as mqtt
import time

# Load dataset
file_path = "data/IoMT.csv"
df = pd.read_csv(file_path)

# MQTT setup
broker = "test.mosquitto.org"
topic = "iomt/security"
client = mqtt.Client()

client.connect(broker)

for index, row in df.iterrows():
    message = str(row.to_dict())  # Convert row to JSON
    client.publish(topic, message)  # Send data
    print(f"Sent: {message}")
    time.sleep(1)  # Simulate real-time delay
