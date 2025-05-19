import paho.mqtt.client as mqtt
import time
import json

# MQTT Configuration
MQTT_BROKER = "wss://broker.emqx.io:8084/mqtt"
MQTT_PORT = 1883
MQTT_TOPIC = "aruco@123"

def on_connect(client, userdata, flags, rc):
    """Callback for when the client connects to the broker."""
    if rc == 0:
        print("\nSuccessfully connected to MQTT broker!")
        print(f"Subscribing to topic: {MQTT_TOPIC}")
        # Subscribe to the topic
        client.subscribe(MQTT_TOPIC)
        print("Waiting for messages...\n")
    else:
        print(f"Connection failed with code {rc}")

def on_message(client, userdata, message):
    """Callback for when a message is received."""
    try:
        # Get the received message
        payload = message.payload.decode()
        
        print("\nNew message received:")
        print(f"Topic: {message.topic}")
        print(f"{payload}")
        
        # If the message contains detected markers
        # if payload != "No markers detected":
        #     # Split the values and process them
        #     ascii_values = payload.split(", ")
        #     print("\nDetected markers:")
        #     for i, value in enumerate(ascii_values, 1):
        #         print(f"Marker {i}: {value}")
        # else:
        #     print("No ArUco markers were detected in the image")
            
        print("\nWaiting for next message...")
        
    except Exception as e:
        print(f"Error processing message: {e}")

def main():
    # Create MQTT client
    client = mqtt.Client()
    
    # Set callbacks
    client.on_connect = on_connect
    client.on_message = on_message
    
    print(f"Attempting to connect to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
    
    try:
        # Connect to MQTT broker
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        
        # Start the loop
        client.loop_forever()
        
    except KeyboardInterrupt:
        print("\nDisconnecting from broker...")
        client.disconnect()
        print("Done!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()