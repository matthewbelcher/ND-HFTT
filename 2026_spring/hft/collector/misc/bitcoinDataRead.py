#Python Script to read the bitcoin data. 
import jwt
import time
import json
import csv
import secrets
import os
from cryptography.hazmat.primitives import serialization
import websocket


with open("../misc/cdp_api_key.json", "r") as file:
    keys = json.load(file)


# --- CONFIGURATION ---
API_KEY = keys["name"]
SIGNING_KEY = keys["privateKey"]
PRODUCT_ID = "BTC-USD"
CSV_FILE = "rawdata/bitcoin_data2.csv"


def init_csv():
    
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "raw_json"])


def build_jwt():
    ##Generates a 2-minute JWT for Coinbase WebSocket Authentication.
    private_key_bytes = SIGNING_KEY.encode('utf-8')
    private_key = serialization.load_pem_private_key(private_key_bytes, password=None)

    payload = {
        'iss': "cdp",
        'nbf': int(time.time()),
        'exp': int(time.time()) + 120,
        'sub': API_KEY,
    }

    return jwt.encode(
        payload,
        private_key,
        algorithm='ES256',
        headers={'kid': API_KEY, 'nonce': secrets.token_hex(16)}
    )


def on_message(ws, message):

    #print(f"Raw message: {message}") FILLS UP THE TERMINAL. ONLY DO SO WHEN DEBUGGING. 

    timestamp = time.time()

    # Write raw message to CSV
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, message])

    data = json.loads(message)

    # Handle subscription confirmation
    if data.get("type") == "subscriptions":
        print(f"Subscription confirmed: {data}")
        return

    # Handle level2 data
    if data.get("channel") == "level2":
        events = data.get("events", [])
        for event in events:
            pass


def on_open(ws):
    print("Socket Opened. Sending subscription...")

    subscribe_msg = {
        "type": "subscribe",
        "product_ids": [PRODUCT_ID],
        "channel": "level2",
        "jwt": build_jwt()
    }

    ws.send(json.dumps(subscribe_msg))


def on_error(ws, error):
    print(f"Error: {error}")


def on_close(ws, status, msg):
    print("Connection Closed")


if __name__ == "__main__":

    init_csv()

    ws = websocket.WebSocketApp(
        "wss://advanced-trade-ws.coinbase.com",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    ws.run_forever()