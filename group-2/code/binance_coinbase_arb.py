#Websocket Attempt
import websocket
import json
import threading
import time
import ssl
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Globals
latest_prices = {"binance": None, "coinbase": None}
last_opportunity_time = time.time()

# Functions
def on_message_binance(ws, message):
    global latest_prices
    # global binance_quantity
    # logging.debug(f"Binance raw message: {message}")
    try:
        data = json.loads(message)
        price = float(data['p'])  # Extract trade price
        latest_prices["binance"] = price
        # binance_quantity = float(data['q'])  # Extract trade quantity
        # logging.info("------------------------------------------------")
        # logging.info(f"Binance price: ${price:.2f}| Coinbase price: ${latest_prices['coinbase']:.2f}")
        check_arbitrage() # only check on binance so we can buy
    except Exception as e:
        # logging.error(f"Error parsing Binance message: {e}")
        pass
        
def on_message_coinbase(ws, message):
    global latest_prices
    data = json.loads(message)
    if "price" in data:
        price = float(data["price"])
        latest_prices["coinbase"] = price
        # logging.info("------------------------------------------------")
        # logging.info(f"Coinbase price: ${price:.2f}| Binance price: ${latest_prices['binance']:.2f}")
        # logging.debug(f"Coinbase price: ${price:.2f}")
        # check_arbitrage()
        
def on_error(ws, error):
    logging.error(f"WebSocket Error: {error}")
    time.sleep(5)  # Retry after a short delay
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})  # Retry the connection
    
def on_close(ws, close_status_code, close_msg):
    logging.info(f"WebSocket Closed: {close_status_code} - {close_msg}")
    time.sleep(5)  # Retry after a short delay
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})  # Retry the connection
    
def on_open_binance(ws):
    logging.info(":white_check_mark: Connected to Binance WebSocket")
    subscribe_message = {
        "method": "SUBSCRIBE",
        "params": ["btcusdt@trade"],
        "id": 1
    }
    ws.send(json.dumps(subscribe_message))
    
def on_open_coinbase(ws):
    logging.info(":white_check_mark: Connected to Coinbase WebSocket")
    subscribe_message = {
        "type": "subscribe",
        "channels": [{"name": "ticker", "product_ids": ["BTC-USD"]}]
    }
    ws.send(json.dumps(subscribe_message))
    
def check_arbitrage():
    if latest_prices["binance"] and latest_prices["coinbase"]:
        binance_price = latest_prices["binance"]
        coinbase_price = latest_prices["coinbase"]
        spread = coinbase_price - binance_price
        fees = binance_price * 0.0005 + coinbase_price * 0.0005
        profit = spread - fees
        if profit > 0:
            global last_opportunity_time
            last_opportunity_time = time.time()
            logging.info("------------------------------------------------")
            logging.info(":rocket: Arbitrage Opportunity Detected! :rocket:")
            logging.info(f"Binance Price: ${binance_price:.2f} | Coinbase Price: ${coinbase_price:.2f}")
            logging.info(f"Spread: ${spread:.2f} | Fees: ${fees:.2f} | Profit: ${profit:.2f}")
            logging.info("------------------------------------------------")
        # else:
        #     logging.info("No arbitrage opportunity at the moment.")
            
def binance_ws():
    url = "wss://stream.binance.us:9443/ws/btcusdt@trade"
    ws = websocket.WebSocketApp(url,
                                on_message=on_message_binance,
                                on_error=on_error,
                                on_close=on_close,
                                on_open=on_open_binance)
    logging.info("Attempting to connect to Binance WebSocket...")
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    
def coinbase_ws():
    url = "wss://ws-feed.exchange.coinbase.com"
    ws = websocket.WebSocketApp(url,
                                on_message=on_message_coinbase,
                                on_error=on_error,
                                on_close=on_close,
                                on_open=on_open_coinbase)
    logging.info("Attempting to connect to Coinbase WebSocket...")
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})  # Disable SSL verification

def main():
    global last_opportunity_time
    # Run both WebSocket connections in separate threads
    binance_thread = threading.Thread(target=binance_ws, daemon=True)
    coinbase_thread = threading.Thread(target=coinbase_ws, daemon=True)
    binance_thread.start()
    coinbase_thread.start()
    
    # Keep script running
    while True:
        time.sleep(1)
        if time.time() - last_opportunity_time > 3600:
            logging.info("No arbitrage opportunity detected in the last hour.")
            last_opportunity_time = time.time()

if __name__ == "__main__":
    main()