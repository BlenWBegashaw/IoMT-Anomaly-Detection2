import paho.mqtt.client as mqtt

broker = "test.mosquitto.org"
topic = "iomt/security"

def on_message(client, userdata, msg):
    print(f"Received Data: {msg.payload.decode()}")

client = mqtt.Client()
client.on_message = on_message

client.connect(broker)
client.subscribe(topic)

client.loop_forever()
