import requests
import datetime
import base64
import uuid
from urllib.parse import urlparse
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import padding
import time

with open('KalshiBitcoin/CryptoBot.txt', 'r', encoding='utf-8') as f:
    first_line = f.readline().strip()
    second_line = f.readline().strip()

# --- Configuration ---
API_KEY_ID = first_line
PRIVATE_KEY_PATH = second_line
BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

def load_private_key(key_path):
    with open(key_path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None, backend=default_backend())

def create_signature(private_key, timestamp, method, path):
    # Strip query params before signing
    path_without_query = path.split('?')[0]
    message = f"{timestamp}{method}{path_without_query}".encode('utf-8')
    signature = private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256()
    )
    return base64.b64encode(signature).decode('utf-8')

def get(private_key, api_key_id, path):
    timestamp = str(int(datetime.datetime.now().timestamp() * 1000))
    sign_path = urlparse(BASE_URL + path).path
    signature = create_signature(private_key, timestamp, "GET", sign_path)
    headers = {
        'KALSHI-ACCESS-KEY': api_key_id,
        'KALSHI-ACCESS-SIGNATURE': signature,
        'KALSHI-ACCESS-TIMESTAMP': timestamp
    }
    #print(f" ID: {api_key_id}, Signature: {signature}, Timestamp: {timestamp}")
    return requests.get(BASE_URL + path, headers=headers)

def post(private_key, api_key_id, path, data):
    timestamp = str(int(datetime.datetime.now().timestamp() * 1000))
    sign_path = urlparse(BASE_URL + path).path
    signature = create_signature(private_key, timestamp, "POST", sign_path)
    headers = {
        'KALSHI-ACCESS-KEY': api_key_id,
        'KALSHI-ACCESS-SIGNATURE': signature,
        'KALSHI-ACCESS-TIMESTAMP': timestamp,
        'Content-Type': 'application/json'
    }
    return requests.post(BASE_URL + path, headers=headers, json=data)

def create_yes_order(market_ticker, action, size, price):
    path = "/portfolio/orders"
    data = {
        "ticker": market_ticker,
        "action": action,  # "BUY" or "SELL"
        "side": "yes",      # "YES" or "NO"
        "count": size,      # e.g. 100 for $1.00 worth of contracts
        "yes_price_dollars": price,     # Required for LIMIT orders
        "client_order_id": str(uuid.uuid4()),  # Unique ID for idempotency
        "expiration_ts": int(datetime.datetime.now().timestamp()) + 60
    }
    
    response = post(private_key, API_KEY_ID, path, data)
    if response.status_code == 201:
        order = response.json()['order']
        print(f"✅ Order placed! ID: {order['order_id']}, Status: {order['status']}")
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
    

def create_no_order(market_ticker, action, size, price):
    path = "/portfolio/orders"
    data = {
        "ticker": market_ticker,
        "action": action,  # "BUY" or "SELL"
        "side": "no",      # "YES" or "NO"
        "count": size,      # e.g. 100 for $1.00 worth of contracts
        "no_price_dollars": price,     # Required for LIMIT orders
        "client_order_id": str(uuid.uuid4()),  # Unique ID for idempotency
        "expiration_ts": int(datetime.datetime.now().timestamp()) + 60
    }
    
    response = post(private_key, API_KEY_ID, path, data)
    if response.status_code == 201:
        order = response.json()['order']
        print(f"✅ Order placed! ID: {order['order_id']}, Status: {order['status']}")
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
    
def get_position(ticker):
    response = get(private_key, API_KEY_ID, f"/portfolio/positions?ticker={ticker}")
    
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return None
    
    positions = response.json().get('market_positions', [])
    
    if not positions:
        print("No position in this market.")
        return None
    
    position = positions[0]
    yes_contracts = position.get('position', 0)  # positive = YES, negative = NO
    
    if yes_contracts > 0:
        print(f"You hold {yes_contracts} YES contracts")
    elif yes_contracts < 0:
        print(f"You hold {abs(yes_contracts)} NO contracts")
    else:
        print("No open position.")
    
    return position

# Load your key
private_key = load_private_key(PRIVATE_KEY_PATH)

response = get(private_key, API_KEY_ID, "/portfolio/balance")
print(f"Balance: ${response.json()['balance'] / 100:.2f}")

#position = get_position("KXBTC15M-26APR040015-15")
#print(position)

#response = requests.get('https://elections-api.kalshi.co/trade-api/v2/markets?limit=5&status=open')
#markets = response.json()['markets']

# Make sure this base URL matches whichever BASE_URL you set for orders
#response = requests.get(
#    'https://api.elections.kalshi.com/trade-api/v2/markets',
#    params={
#        'series_ticker': 'KXBTC15M',
#        'status': 'open',
#        'limit': 5
#    }
#)
#markets = response.json()['markets']

#for m in markets:
#    print(f"Ticker: {m['ticker']} — {m['title']}")

#create_yes_order('KXBTC15M-26APR031715-15', "buy", 1, 1)