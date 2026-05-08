### What the program needs to do
# 1. Subscribe to a crypto data API
# 2. Measure any short spikes in bitcoin price proportional to the next 15m mark
# 3. If there is a spike, either buy yes or no immediately
# 4a. Immediately place contracts to sell for profit proportional to price jump
# 4b. Or sell after a fixed time has passed

"""
btc_feed.py  -  Subscribe to live BTC price ticks from Binance + Coinbase
Install: pip install aiohttp orjson
Run:     python btc_feed.py
"""

import asyncio
from collections import deque
from datetime import date, datetime
import json
import time
import aiohttp
import orjson
import requests
import test_order

async def binance_feed():
    url = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url) as ws:
            print("[binance]  Connected ✓")
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data  = orjson.loads(msg.data)
                    price = float(data["p"])
                    print(f"[binance]  ${price:,.2f}")

history = deque(maxlen=1000)  # store recent price history for spike detection
import math
from collections import deque

history = deque(maxlen=500)
SIGNIFICANCE_THRESHOLD = 0.10  # tune this
MIN_ESTIMATED_KALSHI_MOVE = 10  # only act if we expect >10¢ swing
WINDOW_DURATION = 15 * 60      # 15 minutes in seconds
CURRENT_VARIANCE = 0.0004
BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


def window_start():
    # Kalshi 15-min windows start at :00, :15, :30, :45
    now = time.time()
    return now - (now % WINDOW_DURATION)

def get_contract_price_cents():
    event_ticker = get_todays_BTC_15min_event_ticker()
    markets = get_event_markets(event_ticker)
    for market in markets:
        #print(market)
        yes_bid = float(market.get("yes_bid_dollars"))
        #yes_bid_size = market.get("yes_bid_size_fp")
        yes_ask = float(market.get("yes_ask_dollars"))
        #yes_ask_size = market.get("yes_ask_size_fp")
        no_bid = float(market.get("no_bid_dollars"))
        no_ask = float(market.get("no_ask_dollars"))
        #no_bid_size = market.get("no_bid_size_fp")
        #no_ask_size = market.get("no_ask_size_fp")
        #print(f"Debug: {event_ticker} - target_price: {target_price}, yes_bid: {yes_bid} ({yes_bid_size}), yes_ask: {yes_ask} ({yes_ask_size}), no_bid: {no_bid} ({no_bid_size}), no_ask: {no_ask} ({no_ask_size})")
        return [yes_bid, yes_ask, no_bid, no_ask]

    
def get_todays_BTC_15min_event_ticker():
    """Build today's 5pm EST BTC event ticker dynamically."""
    today = date.today()
    # Format: KXBTC15M-YYMMDD1430  (1430 = 2:30pm)
    current_time = datetime.now()
    current_hour = current_time.hour
    current_minute = current_time.minute
    next_15min_mark = ((current_minute // 15) + 1) * 15
    if next_15min_mark == 60:
        next_15min_mark = 0
        current_hour += 1
    return f"KXBTC15M-{today.strftime('%y%b%d').upper()}{current_hour:02d}{next_15min_mark:02d}"

def get_market_ticker(event_ticker):
    event_ticker = get_todays_BTC_15min_event_ticker()
    current_time = datetime.now()
    current_hour = current_time.hour
    current_minute = current_time.minute
    next_15min_mark = ((current_minute // 15) + 1) * 15
    if next_15min_mark == 60:
        next_15min_mark = 0
        current_hour += 1
    market_ticker = event_ticker + f"-{next_15min_mark:02d}"
    return market_ticker

def get_event_markets(event_ticker):
    """Fetch all markets (strike options) within a specific event."""
    url = f"{BASE_URL}/events/{event_ticker}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return data.get("markets", [])

def minutes_remaining():
    elapsed = time.time() - window_start()
    return (WINDOW_DURATION - elapsed) / 60

def sensitivity(contract_price_cents):
    p = contract_price_cents / 100
    return 4 * p * (1 - p)

def on_price(price, timestamp):
    history.append((timestamp, price))
    
    mins_left = minutes_remaining()
    if mins_left < 1 or mins_left > 13:
        return  # don't trade right at open or close

    # Rolling 30s window
    cutoff = timestamp - 10
    window = [p for t, p in history if t >= cutoff]
    if len(window) < 5:
        return

    pct_move = abs(price - window[0]) / window[0] * 100
    direction = "up" if price > window[0] else "down"

    # Time-adjusted significance
    significance = pct_move / math.sqrt(mins_left) / math.sqrt(CURRENT_VARIANCE)  # Adjust for volatility   

    # Estimated Kalshi impact
    K = 40
    # sens = sensitivity(contract_price_cents)
    kalshi_move = K * significance

    if significance > SIGNIFICANCE_THRESHOLD and kalshi_move > MIN_ESTIMATED_KALSHI_MOVE:
        ticker = get_market_ticker()
        if direction == "up":
            price = get_contract_price_cents()[1]  # yes_ask
            print(f"📈 Detected {pct_move:.2f}% spike up with significance {significance:.2f} and estimated Kalshi move of {kalshi_move:.2f}¢. Placing YES order.")
            test_order.create_yes_order(ticker, "buy", 100, str(price))
            time.sleep(5)
            price = get_contract_price_cents()[0]  # yes_bid
            test_order.create_yes_order(ticker, "sell", 100, str(price))
        else:
            price = get_contract_price_cents()[3]  # no_ask
            print(f"📉 Detected {pct_move:.2f}% spike down with significance {significance:.2f} and estimated Kalshi move of {kalshi_move:.2f}¢. Placing NO order.")
            test_order.create_no_order(ticker, "buy", 100, str(price))
            time.sleep(5)  # Wait a few seconds before placing exit order
            price = get_contract_price_cents()[2]  # no_bid
            test_order.create_no_order(ticker, "sell", 100, str(price))
        # Wait a few seconds
        # Exit order


async def coinbase_feed():
    url = "wss://advanced-trade-ws.coinbase.com"
    sub = {
        "type": "subscribe",
        "product_ids": ["BTC-USD"],
        "channel": "market_trades",
    }
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url) as ws:
            await ws.send_str(json.dumps(sub))
            print("[coinbase] Connected ✓")
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = orjson.loads(msg.data)
                    for event in data.get("events", []):
                        for trade in event.get("trades", []):
                            price = float(trade["price"])
                            print(f"[coinbase] ${price:,.2f}")
                            on_price(price, time.time())
                            
async def main():
    print("Connecting to Binance and Coinbase BTC feeds...\n")
    await asyncio.gather(coinbase_feed())


if __name__ == "__main__":
    asyncio.run(main())