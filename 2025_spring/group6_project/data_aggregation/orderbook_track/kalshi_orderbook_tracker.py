import asyncio
import json
import sqlite3
import websockets
import os
from datetime import datetime, timezone, timedelta
import logging
import requests
import time
import base64
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("kalshi_tracker.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("kalshi_tracker")

# Database setup
DB_PATH = "kalshi_orderbooks.db"

def setup_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create markets table to track market info
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS markets (
        market_ticker TEXT PRIMARY KEY,
        asset TEXT,
        series_ticker TEXT,
        event_ticker TEXT,
        description TEXT,
        last_updated TIMESTAMP
    )
    ''')
    
    # Create orderbook table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS orderbook_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        market_ticker TEXT,
        timestamp TIMESTAMP,
        snapshot_data TEXT,
        FOREIGN KEY (market_ticker) REFERENCES markets (market_ticker)
    )
    ''')
    
    # Create table for deltas
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS orderbook_deltas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        market_ticker TEXT,
        timestamp TIMESTAMP,
        price INTEGER,
        delta INTEGER, 
        side TEXT,
        seq INTEGER,
        FOREIGN KEY (market_ticker) REFERENCES markets (market_ticker)
    )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database setup complete")

def store_market_info(market_ticker, asset, series_ticker, event_ticker, description):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT OR REPLACE INTO markets (market_ticker, asset, series_ticker, event_ticker, description, last_updated)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (market_ticker, asset, series_ticker, event_ticker, description, datetime.now(timezone.utc)))
    
    conn.commit()
    conn.close()

def store_orderbook_snapshot(market_ticker, snapshot_data, seq):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO orderbook_snapshots (market_ticker, timestamp, snapshot_data)
    VALUES (?, ?, ?)
    ''', (market_ticker, datetime.now(timezone.utc), json.dumps(snapshot_data)))
    
    conn.commit()
    conn.close()
    logger.info(f"Stored orderbook snapshot for {market_ticker} with seq {seq}")

def store_orderbook_delta(market_ticker, price, delta, side, seq):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO orderbook_deltas (market_ticker, timestamp, price, delta, side, seq)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (market_ticker, datetime.now(timezone.utc), price, delta, side, seq))
    
    conn.commit()
    conn.close()
    logger.debug(f"Stored orderbook delta for {market_ticker}: {price} {side} {delta} (seq {seq})")

def load_private_key(key_path):
    """Load the private key from file."""
    with open(key_path, "rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            password=None,  # Add password if the key is encrypted
            backend=default_backend()
        )
    return private_key

def sign_pss_text(private_key, text):
    """Sign a message using RSA PSS."""
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

class KalshiOrderbookTracker:
    def __init__(self, api_key, private_key_path):
        # Store credentials
        self.api_key = api_key
        self.private_key_path = private_key_path
        self.private_key = load_private_key(private_key_path)
        logger.info("Private key loaded successfully")
        
        # API endpoints with new domain
        self.ws_url = "wss://api.elections.kalshi.com/trade-api/ws/v2"
        self.rest_base_url = "https://api.elections.kalshi.com/trade-api/v2"
        
        self.orderbook_snapshots = {}
        self.orderbook_sequences = {}
        self.subscription_ids = {}
        self.command_id = 1
        
        # Market tracking info
        self.btc_series = "KXBTC"
        self.eth_series = "KXETH"
        self.btc_event = None
        self.eth_event = None
        self.btc_market = None
        self.eth_market = None
        
        # List to track top markets by liquidity
        self.btc_markets_to_track = []
        self.eth_markets_to_track = []
        
    def get_headers(self, method, path, timestamp=None):
        """Generate the headers for Kalshi API requests."""
        if timestamp is None:
            current_time = datetime.now()
            timestamp = str(int(current_time.timestamp() * 1000))
            
        msg_string = f"{timestamp}{method}{path}"
        sig = sign_pss_text(self.private_key, msg_string)
        
        headers = {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
        }
        return headers
        
    async def fetch_current_markets(self):
        """
        Determine the current active markets for BTC and ETH using REST API,
        specifically targeting the next day's 5 PM EDT market.
        """
        try:
            # Initialize arrays to track BTC and ETH markets with their liquidity
            btc_market_liquidity = []
            eth_market_liquidity = []
            
            # --- Fetch and select the target event tickers ---
            self.btc_event = None
            self.eth_event = None
            
            # Determine the target date and time (5 PM EDT of today or tomorrow)
            # Assuming EDT is UTC-4. Adjust edt_offset if needed for daylight saving.
            edt_offset = timedelta(hours=-4) 
            now_edt = datetime.now(timezone(edt_offset))
            cutoff_hour = 17 # 5 PM
            
            if now_edt.hour < cutoff_hour:
                target_date = now_edt.date()
                logger.info(f"Current EDT time ({now_edt.strftime('%H:%M')}) is before {cutoff_hour}:00. Targeting TODAY's 5 PM market ({target_date}).")
            else:
                target_date = now_edt.date() + timedelta(days=1)
                logger.info(f"Current EDT time ({now_edt.strftime('%H:%M')}) is {cutoff_hour}:00 or later. Targeting TOMORROW's 5 PM market ({target_date}).")
                
            target_time_str_suffix = "1700" # 5 PM marker in ticker
            target_date_str_format = "%d%b" # Format like 26APR
            target_date_str = target_date.strftime(target_date_str_format).upper()
            
            # --- Fetch BTC Events and Find Target ---
            logger.info(f"Fetching events for {self.btc_series}")
            timestamp = str(int(datetime.now().timestamp() * 1000))
            method = "GET"
            path = f'/trade-api/v2/events?series_ticker={self.btc_series}'
            headers = self.get_headers(method, path, timestamp)
            events_url = f"{self.rest_base_url.split('/trade-api/v2')[0]}{path}"
            events_response = requests.get(events_url, headers=headers)
            events_response.raise_for_status()
            response_data = events_response.json()
            
            btc_events = response_data.get("events", [])
            if btc_events:
                logger.info(f"--- Found {len(btc_events)} BTC Events --- ")
                for event in btc_events:
                    # Log the entire event structure for debugging
                    logger.info(f"  Raw BTC Event Data: {event}") 
                    
                    event_ticker_log = event.get('event_ticker', 'N/A')
                    event_title_log = event.get('title', 'N/A')
                    event_subtitle_log = event.get('sub_title', '') # Get subtitle
                    event_close_date_log = event.get('close_date', 'N/A')
                    logger.info(f"  Checking BTC Event: {event_ticker_log} - {event_title_log} (Closes: {event_close_date_log})")
                    
                    # --- Match based on Title/Subtitle --- 
                    try:
                        # Combine title and subtitle for searching
                        search_text = f"{event_title_log} {event_subtitle_log}".lower()
                        
                        # Check for target time (5pm / 17:00)
                        time_match = "5pm edt" in search_text or "17:00 edt" in search_text # Basic check
                        
                        # Check for target date (e.g., "Apr 16")
                        # Format target date like "Apr 16"
                        target_date_title_format = target_date.strftime("%b %d").lower()
                        date_match = target_date_title_format in search_text

                        if time_match and date_match:
                            self.btc_event = event_ticker_log
                            logger.info(f"*** Found matching BTC event via title/subtitle for target date {target_date_str}: {self.btc_event} ***")
                            break # Found the target event
                                
                    except Exception as e:
                        logger.warning(f"Error processing title/subtitle for event {event_ticker_log}: {e}")
                
                if not self.btc_event:
                    logger.warning(f"Could not find a matching BTC event for target date {target_date_str} at 5 PM via title/subtitle.")

            # --- Fetch ETH Events and Find Target ---
            logger.info(f"Fetching events for {self.eth_series}")
            timestamp = str(int(datetime.now().timestamp() * 1000))
            path = f'/trade-api/v2/events?series_ticker={self.eth_series}'
            headers = self.get_headers(method, path, timestamp)
            events_url = f"{self.rest_base_url.split('/trade-api/v2')[0]}{path}"
            events_response = requests.get(events_url, headers=headers)
            events_response.raise_for_status()
            response_data = events_response.json()
            
            eth_events = response_data.get("events", [])
            if eth_events:
                logger.info(f"--- Found {len(eth_events)} ETH Events --- ")
                for event in eth_events:
                     # Log the entire event structure for debugging
                     logger.info(f"  Raw ETH Event Data: {event}") 
                     
                     event_ticker_log = event.get('event_ticker', 'N/A')
                     event_title_log = event.get('title', 'N/A')
                     event_subtitle_log = event.get('sub_title', '') # Get subtitle
                     event_close_date_log = event.get('close_date', 'N/A')
                     logger.info(f"  Checking ETH Event: {event_ticker_log} - {event_title_log} (Closes: {event_close_date_log})")
                     
                     # --- Match based on Title/Subtitle --- 
                     try:
                         # Combine title and subtitle for searching
                         search_text = f"{event_title_log} {event_subtitle_log}".lower()
                         
                         # Check for target time (5pm / 17:00)
                         time_match = "5pm edt" in search_text or "17:00 edt" in search_text # Basic check
                         
                         # Check for target date (e.g., "Apr 16")
                         # Format target date like "Apr 16"
                         target_date_title_format = target_date.strftime("%b %d").lower()
                         date_match = target_date_title_format in search_text

                         if time_match and date_match:
                             self.eth_event = event_ticker_log
                             logger.info(f"*** Found matching ETH event via title/subtitle for target date {target_date_str}: {self.eth_event} ***")
                             break # Found the target event

                     except Exception as e:
                         logger.warning(f"Error processing title/subtitle for event {event_ticker_log}: {e}")

                if not self.eth_event:
                     logger.warning(f"Could not find a matching ETH event for target date {target_date_str} at 5 PM via title/subtitle.")

            # --- Fetch markets for the selected events ---
            
            # Process BTC markets if event was found
            if self.btc_event:
                timestamp = str(int(datetime.now().timestamp() * 1000))
                path = f'/trade-api/v2/markets?event_ticker={self.btc_event}'
                headers = self.get_headers(method, path, timestamp)
                markets_url = f"{self.rest_base_url.split('/trade-api/v2')[0]}{path}"
                markets_response = requests.get(markets_url, headers=headers)
                markets_response.raise_for_status()
                markets_data = markets_response.json()
                logger.info(f"BTC markets response keys: {markets_data.keys()}")

                btc_markets = []
                if "markets" in markets_data:
                    logger.info(f"Found {len(markets_data['markets'])} total BTC markets for event {self.btc_event}")
                    for market in markets_data["markets"]:
                         market_ticker = market.get("ticker", "")
                         market_title = market.get("title", "")
                         #logger.info(f"BTC market found: {market_ticker} - {market_title}") # Less verbose logging
                         btc_markets.append(market)
                
                # Get orderbook data for each BTC market
                for market in btc_markets:
                    market_ticker = market.get("ticker")
                    market_title = market.get("title")
                    if market_ticker:
                         # Get orderbook to check liquidity
                         timestamp_ob = str(int(datetime.now().timestamp() * 1000))
                         orderbook_path = f'/trade-api/v2/markets/{market_ticker}/orderbook'
                         orderbook_headers = self.get_headers("GET", orderbook_path, timestamp_ob)
                         orderbook_url = f"{self.rest_base_url.split('/trade-api/v2')[0]}{orderbook_path}"
                         try:
                             orderbook_response = requests.get(orderbook_url, headers=orderbook_headers)
                             if orderbook_response.status_code == 200:
                                 orderbook_data = orderbook_response.json()
                                 total_liquidity = 0
                                 yes_liquidity = 0
                                 has_yes_orders = False
                                 
                                 if "orderbook" in orderbook_data and orderbook_data["orderbook"] is not None:
                                     yes_orders = orderbook_data["orderbook"].get("yes", []) or []
                                     no_orders = orderbook_data["orderbook"].get("no", []) or []
                                     
                                     # Check if there are any YES orders
                                     if yes_orders and len(yes_orders) > 0:
                                         has_yes_orders = True
                                         for order in yes_orders:
                                             if isinstance(order, list) and len(order) >= 2:
                                                 yes_liquidity += order[1]
                                                 total_liquidity += order[1]
                                                 
                                     for order in no_orders:
                                         if isinstance(order, list) and len(order) >= 2:
                                             total_liquidity += order[1]
                                             
                                 btc_market_liquidity.append({
                                     "ticker": market_ticker, 
                                     "liquidity": total_liquidity, 
                                     "yes_liquidity": yes_liquidity,
                                     "has_yes_orders": has_yes_orders,
                                     "title": market_title
                                 })
                                 #logger.info(f"BTC Market {market_ticker} ({market_title}) has {total_liquidity} total lots") # Less verbose
                             else:
                                 logger.warning(f"Orderbook request failed for {market_ticker}: {orderbook_response.status_code}")
                                 btc_market_liquidity.append({
                                     "ticker": market_ticker, 
                                     "liquidity": 0, 
                                     "yes_liquidity": 0,
                                     "has_yes_orders": False,
                                     "title": market_title
                                 })
                         except Exception as e:
                             logger.error(f"Error getting orderbook for {market_ticker}: {str(e)}")
                             btc_market_liquidity.append({
                                 "ticker": market_ticker, 
                                 "liquidity": 0, 
                                 "yes_liquidity": 0,
                                 "has_yes_orders": False,
                                 "title": market_title
                             })
                         
                         # Store market info in database
                         store_market_info(market_ticker, "BTC", self.btc_series, self.btc_event, market_title)

            # Process ETH markets if event was found
            if self.eth_event:
                timestamp = str(int(datetime.now().timestamp() * 1000))
                path = f'/trade-api/v2/markets?event_ticker={self.eth_event}'
                headers = self.get_headers(method, path, timestamp)
                markets_url = f"{self.rest_base_url.split('/trade-api/v2')[0]}{path}"
                markets_response = requests.get(markets_url, headers=headers)
                markets_response.raise_for_status()
                markets_data = markets_response.json()
                logger.info(f"ETH markets response keys: {markets_data.keys()}")

                eth_markets = []
                if "markets" in markets_data:
                    logger.info(f"Found {len(markets_data['markets'])} total ETH markets for event {self.eth_event}")
                    for market in markets_data["markets"]:
                        market_ticker = market.get("ticker", "")
                        market_title = market.get("title", "")
                        #logger.info(f"ETH market found: {market_ticker} - {market_title}") # Less verbose logging
                        eth_markets.append(market)
                
                # Get orderbook data for each ETH market
                for market in eth_markets:
                    market_ticker = market.get("ticker")
                    market_title = market.get("title")
                    if market_ticker:
                        # Get orderbook to check liquidity
                        timestamp_ob = str(int(datetime.now().timestamp() * 1000))
                        orderbook_path = f'/trade-api/v2/markets/{market_ticker}/orderbook'
                        orderbook_headers = self.get_headers("GET", orderbook_path, timestamp_ob)
                        orderbook_url = f"{self.rest_base_url.split('/trade-api/v2')[0]}{orderbook_path}"
                        try:
                            orderbook_response = requests.get(orderbook_url, headers=orderbook_headers)
                            if orderbook_response.status_code == 200:
                                orderbook_data = orderbook_response.json()
                                total_liquidity = 0
                                yes_liquidity = 0
                                has_yes_orders = False
                                
                                if "orderbook" in orderbook_data and orderbook_data["orderbook"] is not None:
                                    yes_orders = orderbook_data["orderbook"].get("yes", []) or []
                                    no_orders = orderbook_data["orderbook"].get("no", []) or []
                                    
                                    # Check if there are any YES orders
                                    if yes_orders and len(yes_orders) > 0:
                                        has_yes_orders = True
                                        for order in yes_orders:
                                            if isinstance(order, list) and len(order) >= 2:
                                                yes_liquidity += order[1]
                                                total_liquidity += order[1]
                                                
                                    for order in no_orders:
                                        if isinstance(order, list) and len(order) >= 2:
                                            total_liquidity += order[1]
                                            
                                eth_market_liquidity.append({
                                    "ticker": market_ticker, 
                                    "liquidity": total_liquidity, 
                                    "yes_liquidity": yes_liquidity,
                                    "has_yes_orders": has_yes_orders,
                                    "title": market_title
                                })
                                #logger.info(f"ETH Market {market_ticker} ({market_title}) has {total_liquidity} total lots") # Less verbose
                            else:
                                logger.warning(f"Orderbook request failed for {market_ticker}: {orderbook_response.status_code}")
                                eth_market_liquidity.append({
                                    "ticker": market_ticker, 
                                    "liquidity": 0, 
                                    "yes_liquidity": 0,
                                    "has_yes_orders": False,
                                    "title": market_title
                                })
                        except Exception as e:
                            logger.error(f"Error getting orderbook for {market_ticker}: {str(e)}")
                            eth_market_liquidity.append({
                                "ticker": market_ticker, 
                                "liquidity": 0, 
                                "yes_liquidity": 0,
                                "has_yes_orders": False,
                                "title": market_title
                            })
                        
                        # Store market info in database
                        store_market_info(market_ticker, "ETH", self.eth_series, self.eth_event, market_title)

            # --- Select top markets and set tracking ---
            
            # First, filter markets that have YES orders
            btc_markets_with_yes = [m for m in btc_market_liquidity if m["has_yes_orders"]]
            eth_markets_with_yes = [m for m in eth_market_liquidity if m["has_yes_orders"]]
            
            # Then sort by YES liquidity
            btc_markets_with_yes.sort(key=lambda x: x["yes_liquidity"], reverse=True)
            eth_markets_with_yes.sort(key=lambda x: x["yes_liquidity"], reverse=True)
            
            # Select the top 5 markets by YES liquidity
            top_btc_markets = btc_markets_with_yes[:5] if btc_markets_with_yes else []
            top_eth_markets = eth_markets_with_yes[:5] if eth_markets_with_yes else []
            
            # If we don't have 5 markets with YES orders, add some from the original list sorted by total liquidity
            if len(top_btc_markets) < 5:
                remaining_slots = 5 - len(top_btc_markets)
                btc_market_liquidity.sort(key=lambda x: x["liquidity"], reverse=True)
                existing_tickers = [m["ticker"] for m in top_btc_markets]
                additional_markets = [m for m in btc_market_liquidity if m["ticker"] not in existing_tickers][:remaining_slots]
                top_btc_markets.extend(additional_markets)
            
            if len(top_eth_markets) < 5:
                remaining_slots = 5 - len(top_eth_markets)
                eth_market_liquidity.sort(key=lambda x: x["liquidity"], reverse=True)
                existing_tickers = [m["ticker"] for m in top_eth_markets]
                additional_markets = [m for m in eth_market_liquidity if m["ticker"] not in existing_tickers][:remaining_slots]
                top_eth_markets.extend(additional_markets)
            
            # Set primary market ticker (used for logging/fallback in check_for_market_updates)
            self.btc_market = top_btc_markets[0]["ticker"] if top_btc_markets else None
            self.eth_market = top_eth_markets[0]["ticker"] if top_eth_markets else None

            # Log the top markets selected
            if top_btc_markets:
                logger.info(f"Top 5 BTC markets for {self.btc_event}:")
                for i, market in enumerate(top_btc_markets):
                    logger.info(f"  {i+1}. {market['ticker']} - {market['title']} - YES: {market['yes_liquidity']} lots, Total: {market['liquidity']} lots")
            else:
                logger.warning(f"No BTC markets found or orderbooks fetched for event {self.btc_event}")

            if top_eth_markets:
                logger.info(f"Top 5 ETH markets for {self.eth_event}:")
                for i, market in enumerate(top_eth_markets):
                    logger.info(f"  {i+1}. {market['ticker']} - {market['title']} - YES: {market['yes_liquidity']} lots, Total: {market['liquidity']} lots")
            else:
                logger.warning(f"No ETH markets found or orderbooks fetched for event {self.eth_event}")
            
            # Set the actual lists of markets to track for subscriptions
            self.btc_markets_to_track = [market["ticker"] for market in top_btc_markets]
            self.eth_markets_to_track = [market["ticker"] for market in top_eth_markets]
            
            logger.info(f"Tracking top BTC markets: {self.btc_markets_to_track}")
            logger.info(f"Tracking top ETH markets: {self.eth_markets_to_track}")
            
        except Exception as e:
            logger.error(f"Error fetching markets: {str(e)}")
            logger.exception("Detailed exception info:")
            # Clear market tracking on error to avoid subscribing to outdated markets
            self.btc_event = None
            self.eth_event = None
            self.btc_market = None
            self.eth_market = None
            self.btc_markets_to_track = []
            self.eth_markets_to_track = []
            logger.warning("Cleared market tracking due to error.")
    
    async def connect(self):
        logger.info(f"Connecting to {self.ws_url}")
        
        # Generate WebSocket auth headers
        timestamp = str(int(datetime.now().timestamp() * 1000))
        method = "GET"
        path = '/trade-api/ws/v2'  # WebSocket path
        headers = self.get_headers(method, path, timestamp)
        
        try:
            # Use additional_headers instead of extra_headers for websockets v15+
            self.websocket = await websockets.connect(self.ws_url, extra_headers=headers)
            logger.info("WebSocket connection established")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    async def subscribe_to_orderbooks(self):
        # Make sure we have the latest markets
        await self.fetch_current_markets()
        
        # Subscribe to orderbook for top 5 BTC and ETH markets
        markets_to_track = self.btc_markets_to_track + self.eth_markets_to_track
        
        # Ensure we have unique markets and remove any empty values
        markets_to_track = [m for m in markets_to_track if m]
        
        if not markets_to_track:
            logger.warning("No markets to track, using fallback")
            markets_to_track = [self.btc_market, self.eth_market]
        
        logger.info(f"Subscribing to {len(markets_to_track)} markets: {markets_to_track}")
        
        subscribe_cmd = {
            "id": self.command_id,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_tickers": markets_to_track
            }
        }
        self.command_id += 1
        
        await self.websocket.send(json.dumps(subscribe_cmd))
        logger.info(f"Sent subscription request for markets: {markets_to_track}")
    
    async def handle_message(self, message):
        data = json.loads(message)
        message_type = data.get("type")
        
        if message_type == "subscribed":
            logger.info(f"Successfully subscribed: {data}")
            self.subscription_ids[data["msg"]["channel"]] = data["msg"]["sid"]
            
        elif message_type == "orderbook_snapshot":
            market_ticker = data["msg"]["market_ticker"]
            seq = data.get("seq", 0)
            logger.info(f"Received orderbook snapshot for {market_ticker} with seq {seq}")
            
            # Store the snapshot in memory
            self.orderbook_snapshots[market_ticker] = data["msg"]
            self.orderbook_sequences[market_ticker] = seq
            
            # Store in database
            store_orderbook_snapshot(market_ticker, data["msg"], seq)
            
        elif message_type == "orderbook_delta":
            market_ticker = data["msg"]["market_ticker"]
            seq = data.get("seq", 0)
            current_seq = self.orderbook_sequences.get(market_ticker, 0)
            
            # Check for sequence gaps
            if seq != current_seq + 1 and current_seq != 0:
                logger.warning(f"Sequence gap detected for {market_ticker}: expected {current_seq + 1}, got {seq}")
                # We should resubscribe to get a fresh snapshot
                await self.resubscribe_market(market_ticker)
                return
            
            price = data["msg"]["price"]
            delta = data["msg"]["delta"]
            side = data["msg"]["side"]
            
            logger.debug(f"Received orderbook delta for {market_ticker}: {price} {side} {delta} (seq {seq})")
            
            # Update our in-memory snapshot
            self.orderbook_sequences[market_ticker] = seq
            
            # Store delta in database
            store_orderbook_delta(market_ticker, price, delta, side, seq)
            
        elif message_type == "error":
            logger.error(f"Received error: {data}")
    
    async def resubscribe_market(self, market_ticker):
        """Resubscribe to a market to get a fresh orderbook snapshot"""
        logger.info(f"Resubscribing to {market_ticker}")
        
        # Unsubscribe first
        if "orderbook_delta" in self.subscription_ids:
            unsubscribe_cmd = {
                "id": self.command_id,
                "cmd": "unsubscribe",
                "params": {
                    "sids": [self.subscription_ids["orderbook_delta"]]
                }
            }
            self.command_id += 1
            
            await self.websocket.send(json.dumps(unsubscribe_cmd))
            logger.info(f"Sent unsubscribe request")
            
            # Wait for unsubscribe confirmation
            while True:
                message = await self.websocket.recv()
                data = json.loads(message)
                if data.get("type") == "unsubscribed":
                    logger.info("Successfully unsubscribed")
                    break
        
        # Subscribe again
        subscribe_cmd = {
            "id": self.command_id,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_tickers": [market_ticker]
            }
        }
        self.command_id += 1
        
        await self.websocket.send(json.dumps(subscribe_cmd))
        logger.info(f"Sent resubscription request for market: {market_ticker}")
    
    async def heartbeat(self):
        """Send periodic ping to keep connection alive"""
        while True:
            try:
                await self.websocket.ping()
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Heartbeat failed: {e}")
                break
    
    async def check_for_market_updates(self):
        """Periodically check for new markets"""
        while True:
            try:
                # Check every 6 hours
                await asyncio.sleep(6 * 60 * 60)
                
                # Check for updated markets
                old_btc_market = self.btc_market
                old_eth_market = self.eth_market
                
                await self.fetch_current_markets()
                
                # If markets have changed, resubscribe
                if (old_btc_market != self.btc_market) or (old_eth_market != self.eth_market):
                    logger.info("Markets have changed, resubscribing")
                    await self.subscribe_to_orderbooks()
            except Exception as e:
                logger.error(f"Error checking for market updates: {e}")
                await asyncio.sleep(60)  # Retry after a minute if there was an error
    
    async def listen(self):
        """Listen for messages from the websocket"""
        heartbeat_task = asyncio.create_task(self.heartbeat())
        market_check_task = asyncio.create_task(self.check_for_market_updates())
        
        try:
            while True:
                message = await self.websocket.recv()
                await self.handle_message(message)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error in listen loop: {e}")
        finally:
            heartbeat_task.cancel()
            market_check_task.cancel()
    
    async def reconnect_loop(self):
        """Attempt to reconnect if connection is lost"""
        retry_count = 0
        
        while True:  # Changed from "while retry_count < max_retries:" to try indefinitely
            try:
                logger.info(f"Attempting connection (attempt #{retry_count + 1})...")
                connected = await self.connect()
                if connected:
                    logger.info("Connection successful, subscribing to orderbooks...")
                    await self.subscribe_to_orderbooks()
                    logger.info("Starting to listen for messages...")
                    await self.listen()
                    # If we get here, the connection was closed somewhat gracefully
                    logger.warning("WebSocket connection closed, will attempt to reconnect...")
                else:
                    logger.error("Failed to establish connection")
            except Exception as e:
                logger.error(f"Connection error: {e}")
                logger.exception("Detailed exception info:")
            
            retry_count += 1
            wait_time = min(300, 5 * (2 ** min(retry_count, 10)))  # Exponential backoff, capped at 300 seconds
            logger.info(f"Reconnection attempt {retry_count} failed. Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)
            logger.info("Retrying connection now...")
    
    async def run(self):
        """Main execution function"""
        try:
            # Setup database first
            setup_database()
            
            # Initial connection
            await self.reconnect_loop()  # This will now run indefinitely
        except Exception as e:
            logger.critical(f"Fatal error in run method: {e}")
            logger.exception("Detailed exception info:")
            # We shouldn't get here anymore unless there's a truly fatal error
    
async def main():
    # Replace with your Kalshi API key and private key path
    api_key = "717c10a4-88f2-44d7-bc0f-e3bcba996fa0"  # Replace with your actual API key
    private_key_path = "/root/home/ACE/orderbook_track/api_key.txt"  # Replace with your private key path
    
    while True:  # Added outer loop to restart the entire tracker if needed
        try:
            logger.info("Starting Kalshi Orderbook Tracker...")
            tracker = KalshiOrderbookTracker(api_key, private_key_path)
            await tracker.run()
        except Exception as e:
            logger.critical(f"Unexpected error in main function: {e}")
            logger.exception("Detailed exception info:")
            
        # If we somehow exit the tracker's run method, wait and restart
        logger.warning("Main tracker process exited. Restarting in 60 seconds...")
        await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}") 