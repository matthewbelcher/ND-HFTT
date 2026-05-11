#python script to read the kalshi market data 

import asyncio
import base64
import json
import time
import websockets
import csv
from datetime import datetime
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

# Configuration
KEY_ID = "<KALSHI_KEY_ID>"
PRIVATE_KEY_PATH = "TestExample1.pem" #Downloaded Key file

MARKET_TICKER = "KXBTC15M-26MAR161815-15"  #Go to timeline and payout for a market and you will see the market ticker. 
WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2" #I think this the same for everythign
CSV_FILENAME = "rawdata/" + MARKET_TICKER + ".csv"

def sign_pss_text(private_key, text: str) -> str:
    """Sign message using RSA-PSS"""
    message = text.encode('utf-8')
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH
        ),
        hashes.SHA256()
    )
    return base64.b64encode(signature).decode('utf-8')

def create_headers(private_key, method: str, path: str) -> dict:
    """Create authentication headers"""
    timestamp = str(int(time.time() * 1000))
    msg_string = timestamp + method + path.split('?')[0]
    signature = sign_pss_text(private_key, msg_string)

    return {
        "Content-Type": "application/json",
        "KALSHI-ACCESS-KEY": KEY_ID,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
    }

def log_to_csv(timestamp, msg_type, raw_data):
    """Appends the WebSocket message to a CSV file"""
    
    with open(CSV_FILENAME, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Dump the raw JSON into a single column so you don't lose any nested data. Will change later
        writer.writerow([timestamp, msg_type, json.dumps(raw_data)])

async def orderbook_websocket():
    """Connect to WebSocket and subscribe to orderbook"""
    # Load private key
    with open(PRIVATE_KEY_PATH, 'rb') as f:
        private_key = serialization.load_pem_private_key(
            f.read(),
            password=None
        )

    # Initialize CSV with headers if it's completely empty
    with open(CSV_FILENAME, mode='a', newline='') as file:
        if file.tell() == 0:
            csv.writer(file).writerow(["Timestamp", "Message Type", "Raw JSON Data"])

    # Create WebSocket headers
    ws_headers = create_headers(private_key, "GET", "/trade-api/ws/v2")

    async with websockets.connect(WS_URL, additional_headers=ws_headers) as websocket:
        print(f"Connected! Subscribing to orderbook for {MARKET_TICKER}")
        print(f"Logging data to {CSV_FILENAME}...")

        # Subscribe to orderbook
        subscribe_msg = {
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_ticker": MARKET_TICKER
            }
        }
        await websocket.send(json.dumps(subscribe_msg))

        # Process messages
        async for message in websocket:
            data = json.loads(message)
            msg_type = data.get("type")
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-6] #microsecond to match kalshi

            # Log everything to CSV
            log_to_csv(current_time, msg_type, data)

            # Keep console output clean and brief
            if msg_type == "subscribed":
                print(f"[{current_time}] ✅ Successfully subscribed.")
            elif msg_type == "orderbook_snapshot":
                print(f"[{current_time}] 📸 Downloaded initial orderbook snapshot.")
            #elif msg_type == "orderbook_delta":
                #print(f"[{current_time}] ⚡ Orderbook changed (Delta received).")
            elif msg_type == "error":
                print(f"[{current_time}] ❌ Error: {data}")

# Run the example
if __name__ == "__main__":
    asyncio.run(orderbook_websocket())