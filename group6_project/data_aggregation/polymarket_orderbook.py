#!/usr/bin/env python3
import requests
import sqlite3
import json
import time
from datetime import datetime
import os
import sys
import random

# Graph API endpoints - using the specific endpoint provided
POLYMARKET_SUBGRAPH_URL = "https://gateway.thegraph.com/api/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC"

# API key for The Graph
API_KEY = "f7d84699619f1e9627d58c169604a54e"

# Request headers with API key
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# Market conditions for Trump and Harris (we'll determine these dynamically)
TRUMP_CONDITION = None
HARRIS_CONDITION = None

# Token IDs for Trump and Harris
TRUMP_YES_TOKEN = "21742633143463906290569050155826241533067272736897614950488156847949938836455"
TRUMP_NO_TOKEN = "48331043336612883890938759509493159234755048973500640148014422747788308965732"
HARRIS_YES_TOKEN = "69236923620077691027083946871148646972011131466059644796654161903044970987404"
HARRIS_NO_TOKEN = "87584955359245246404952128082451897287778571240979823316620093987046202296181"

# Database setup
DB_FILE = "polymarket_orderbook.db"

def create_database():
    """Create SQLite database with necessary tables for orderbook reconstruction"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS market_data (
        id TEXT PRIMARY KEY,
        condition TEXT,
        outcome_index INTEGER,
        candidate TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS order_filled_events (
        id TEXT PRIMARY KEY,
        transaction_hash TEXT,
        timestamp INTEGER,
        order_hash TEXT,
        maker TEXT,
        taker TEXT,
        maker_asset_id TEXT,
        taker_asset_id TEXT,
        maker_amount_filled INTEGER,
        taker_amount_filled INTEGER,
        fee INTEGER,
        processed INTEGER DEFAULT 0,
        candidate TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS reconstructed_orderbook (
        timestamp INTEGER,
        price REAL,
        side TEXT,  -- 'BUY' or 'SELL'
        size REAL,
        remaining_size REAL,
        maker TEXT,
        asset_id TEXT,
        candidate TEXT
    )
    ''')
    
    # Create indices for faster queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_order_filled_timestamp ON order_filled_events (timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_orderbook_timestamp ON reconstructed_orderbook (timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_orderbook_candidate ON reconstructed_orderbook (candidate)')
    
    conn.commit()
    conn.close()

def test_api_connection():
    """Test connection to the Graph API endpoint"""
    print("Testing API connection...")
    
    # Simple test query
    test_query = """
    {
      orderFilledEvents(first: 5) {
        id
        timestamp
      }
    }
    """
    
    try:
        response = requests.post(
            POLYMARKET_SUBGRAPH_URL,
            json={"query": test_query},
            headers=HEADERS
        )
        
        print(f"API Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if "data" in data and "orderFilledEvents" in data["data"]:
                events = data["data"]["orderFilledEvents"]
                print(f"✅ Connected to API - found {len(events)} sample events")
                if events:
                    print(f"Sample event: {events[0]}")
            else:
                print("❌ Connected but received unexpected response format")
                print(f"Response: {data}")
        else:
            print(f"❌ Failed to connect: {response.status_code}")
            print(f"Response: {response.text}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ Error connecting to API: {e}")
        return False

def find_presidential_markets():
    """Find current presidential election markets by searching for asset IDs"""
    print("\nUsing presidential election token IDs...")
    
    # Token IDs for Trump and Harris from the election market
    trump_yes_asset = TRUMP_YES_TOKEN
    harris_yes_asset = HARRIS_YES_TOKEN
    
    print(f"Trump Yes token ID: {trump_yes_asset}")
    print(f"Trump No token ID: {TRUMP_NO_TOKEN}")
    print(f"Harris Yes token ID: {harris_yes_asset}")
    print(f"Harris No token ID: {HARRIS_NO_TOKEN}")
    
    return trump_yes_asset, harris_yes_asset

def fetch_order_filled_events(asset_id, candidate, first=1000, start_timestamp=None):
    """
    Fetch OrderFilledEvent data for a specific asset ID in batches
    
    If start_timestamp is provided, only fetch events with timestamp greater than that value
    This allows us to paginate by moving forward in time
    """
    if not asset_id:
        print(f"No asset ID provided for {candidate}, skipping")
        return []
    
    timestamp_filter = ""
    if start_timestamp:
        timestamp_filter = f', timestamp_gt: "{start_timestamp}"'
    
    print(f"Fetching order events for {candidate} (asset ID: {asset_id})...")
    
    query = """
    {
      orderFilledEvents(
        first: %d
        where: {makerAssetId: "%s"%s}
        orderBy: timestamp
        orderDirection: asc
      ) {
        id
        transactionHash
        timestamp
        orderHash
        maker
        taker
        makerAssetId
        takerAssetId
        makerAmountFilled
        takerAmountFilled
        fee
      }
    }
    """ % (first, asset_id, timestamp_filter)
    
    try:
        response = requests.post(
            POLYMARKET_SUBGRAPH_URL,
            json={"query": query},
            headers=HEADERS
        )
        
        if response.status_code == 200:
            data = response.json()
            if "data" in data and "orderFilledEvents" in data["data"]:
                events = data["data"]["orderFilledEvents"]
                print(f"  Found {len(events)} events for asset {asset_id}")
                
                # Add candidate information to each event
                for event in events:
                    event['candidate'] = candidate
                
                return events
            else:
                print(f"❌ Unexpected response format")
                if "errors" in data:
                    print(f"Errors: {data['errors']}")
                return []
        else:
            print(f"❌ Failed to fetch events: {response.status_code}")
            print(f"Response: {response.text}")
            return []
    except Exception as e:
        print(f"❌ Error fetching events: {e}")
        return []

def save_order_filled_events(events):
    """Save order filled events to the database"""
    if not events:
        return 0
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    saved_count = 0
    
    for event in events:
        try:
            cursor.execute(
                '''
                INSERT OR IGNORE INTO order_filled_events 
                (id, transaction_hash, timestamp, order_hash, maker, taker, 
                 maker_asset_id, taker_asset_id, maker_amount_filled, 
                 taker_amount_filled, fee, candidate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    event.get('id', ''),
                    event.get('transactionHash', ''),
                    event.get('timestamp', 0),
                    event.get('orderHash', ''),
                    event.get('maker', ''),
                    event.get('taker', ''),
                    event.get('makerAssetId', ''),
                    event.get('takerAssetId', ''),
                    event.get('makerAmountFilled', 0),
                    event.get('takerAmountFilled', 0),
                    event.get('fee', 0),
                    event.get('candidate', '')
                )
            )
            saved_count += cursor.rowcount
        except Exception as e:
            print(f"❌ Error saving event: {e}")
    
    conn.commit()
    conn.close()
    
    return saved_count

def reconstruct_orderbook():
    """
    Reconstruct the orderbook based on filled orders
    
    This is an approximation since we don't have complete orderbook data.
    We use order fills to infer what was likely on the book.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Get unprocessed order filled events
    cursor.execute('''
        SELECT 
            timestamp, maker, maker_asset_id, taker_asset_id, 
            maker_amount_filled, taker_amount_filled, id, candidate
        FROM order_filled_events
        WHERE processed = 0
        ORDER BY timestamp ASC
    ''')
    
    events = cursor.fetchall()
    print(f"Processing {len(events)} unprocessed order events...")
    
    for event in events:
        timestamp, maker, maker_asset_id, taker_asset_id, maker_amount, taker_amount, event_id, candidate = event
        
        # Calculate price (this is a simplification)
        try:
            price = float(taker_amount) / float(maker_amount) if float(maker_amount) > 0 else 0
        except (ValueError, TypeError):
            price = 0
            print(f"Warning: Could not calculate price for event {event_id}")
        
        # Determine the side (BUY or SELL) based on which asset is being traded
        if candidate == "Trump":
            # For Trump trades:
            if maker_asset_id == TRUMP_YES_TOKEN:
                # Maker is providing YES tokens, so they are selling YES (or buying NO)
                side = "SELL"
            elif maker_asset_id == TRUMP_NO_TOKEN:
                # Maker is providing NO tokens, which is equivalent to buying YES
                side = "BUY"
            elif taker_asset_id == TRUMP_YES_TOKEN:
                # Taker is receiving YES tokens, so maker is selling NO (buying YES)
                side = "BUY"
            elif taker_asset_id == TRUMP_NO_TOKEN:
                # Taker is receiving NO tokens, so maker is selling YES
                side = "SELL"
            else:
                # Default if we can't determine
                side = "BUY"
                print(f"Warning: Could not determine side for Trump trade. maker_asset: {maker_asset_id}, taker_asset: {taker_asset_id}")
                
        elif candidate == "Harris":
            # For Harris trades:
            if maker_asset_id == HARRIS_YES_TOKEN:
                # Maker is providing YES tokens, so they are selling YES
                side = "SELL"
            elif maker_asset_id == HARRIS_NO_TOKEN:
                # Maker is providing NO tokens, which is equivalent to buying YES
                side = "BUY"
            elif taker_asset_id == HARRIS_YES_TOKEN:
                # Taker is receiving YES tokens, so maker is selling NO (buying YES)
                side = "BUY"
            elif taker_asset_id == HARRIS_NO_TOKEN:
                # Taker is receiving NO tokens, so maker is selling YES
                side = "SELL"
            else:
                # Default if we can't determine
                side = "BUY"
                print(f"Warning: Could not determine side for Harris trade. maker_asset: {maker_asset_id}, taker_asset: {taker_asset_id}")
        else:
            # Unknown candidate
            side = "BUY"
            print(f"Warning: Unknown candidate {candidate}")
            
        # Log some details
        print(f"Event: {candidate}, Side: {side}, Price: {price}, Size: {float(maker_amount) / 1e18 if maker_amount else 0}")
        
        # Add to reconstructed orderbook
        cursor.execute('''
            INSERT INTO reconstructed_orderbook
            (timestamp, price, side, size, remaining_size, maker, asset_id, candidate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp,
            price,
            side,
            float(maker_amount) / 1e18 if maker_amount else 0,  # Assuming 18 decimals, adjust if needed
            0,  # Since these are filled orders, remaining size is 0
            maker,
            maker_asset_id,
            candidate
        ))
        
        # Mark as processed
        cursor.execute('UPDATE order_filled_events SET processed = 1 WHERE id = ?', (event_id,))
    
    conn.commit()
    conn.close()

def fetch_all_data():
    """Fetch all available data in batches for Trump and Harris"""
    print("\nUsing all presidential election token IDs with The Graph API...")
    
    # Token IDs for Trump and Harris from the election market
    trump_yes_asset = TRUMP_YES_TOKEN
    trump_no_asset = TRUMP_NO_TOKEN
    harris_yes_asset = HARRIS_YES_TOKEN
    harris_no_asset = HARRIS_NO_TOKEN
    
    print(f"Trump Yes token ID: {trump_yes_asset}")
    print(f"Trump No token ID: {trump_no_asset}")
    print(f"Harris Yes token ID: {harris_yes_asset}")
    print(f"Harris No token ID: {harris_no_asset}")
    
    # Verify API connection
    if not test_api_connection():
        print("❌ API connection failed. Please check your API key and network connection.")
        sys.exit(1)
    
    batch_size = 1000
    
    # Fetch events for all token types
    trump_events_count = 0
    harris_events_count = 0
    
    # Process all Trump orders (both YES and NO tokens)
    print(f"\nFetching Trump market events (YES tokens)...")
    trump_events_count += fetch_token_events(trump_yes_asset, "Trump", batch_size)
    
    print(f"\nFetching Trump market events (NO tokens)...")
    trump_events_count += fetch_token_events(trump_no_asset, "Trump", batch_size)
    
    print(f"Total Trump events: {trump_events_count}")
    
    # Process all Harris orders (both YES and NO tokens)
    print(f"\nFetching Harris market events (YES tokens)...")
    harris_events_count += fetch_token_events(harris_yes_asset, "Harris", batch_size)
    
    print(f"\nFetching Harris market events (NO tokens)...")
    harris_events_count += fetch_token_events(harris_no_asset, "Harris", batch_size)
    
    print(f"Total Harris events: {harris_events_count}")
    
    total_events = trump_events_count + harris_events_count
    print(f"\nTotal events fetched: {total_events}")
    
    return trump_yes_asset, harris_yes_asset

def fetch_token_events(asset_id, candidate, batch_size):
    """Fetch all events for a specific token ID with pagination"""
    total_events = 0
    max_retries = 3
    
    # Keep track of the newest timestamp we've seen
    max_timestamp = None
    keep_fetching = True
    retries = 0
    
    while keep_fetching:
        events = fetch_order_filled_events(asset_id, candidate, batch_size, max_timestamp)
        
        if not events:
            retries += 1
            if retries >= max_retries:
                print(f"No more events found after {max_retries} retries")
                keep_fetching = False
            else:
                print(f"No events found, retry {retries}/{max_retries}")
                time.sleep(2)  # Wait a bit longer before retrying
            continue
        
        # Reset retries on successful fetch
        retries = 0
            
        count = save_order_filled_events(events)
        total_events += count
        print(f"Saved {total_events} {candidate} events so far...")
        
        # Find the maximum timestamp in this batch to use as the starting point for the next batch
        if events:
            batch_max_timestamp = max(int(event.get('timestamp', 0)) for event in events)
            max_timestamp = batch_max_timestamp
            
            # If we got fewer events than batch_size, we've reached the end
            if len(events) < batch_size:
                keep_fetching = False
                print("Fetched less than batch size, likely reached the end of events")
            
            # Give the API a break
            time.sleep(1)
        else:
            keep_fetching = False
    
    return total_events

def analyze_orderbook():
    """Print summary statistics about the reconstructed orderbook"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    candidates = ["Trump", "Harris"]
    
    for candidate in candidates:
        print(f"\n{candidate} Orderbook Analysis:")
        
        # Count total orders
        cursor.execute('SELECT COUNT(*) FROM reconstructed_orderbook WHERE candidate = ?', (candidate,))
        total_orders = cursor.fetchone()[0]
        
        # Count buy orders
        cursor.execute('SELECT COUNT(*) FROM reconstructed_orderbook WHERE side = "BUY" AND candidate = ?', (candidate,))
        buy_orders = cursor.fetchone()[0]
        
        # Count sell orders
        cursor.execute('SELECT COUNT(*) FROM reconstructed_orderbook WHERE side = "SELL" AND candidate = ?', (candidate,))
        sell_orders = cursor.fetchone()[0]
        
        # Get price range
        cursor.execute('SELECT MIN(price), MAX(price), AVG(price) FROM reconstructed_orderbook WHERE candidate = ?', (candidate,))
        min_price, max_price, avg_price = cursor.fetchone()
        
        # Get timestamp range
        cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM reconstructed_orderbook WHERE candidate = ?', (candidate,))
        min_ts, max_ts = cursor.fetchone()
        
        if min_ts and max_ts:
            min_date = datetime.fromtimestamp(int(min_ts)).strftime('%Y-%m-%d %H:%M:%S')
            max_date = datetime.fromtimestamp(int(max_ts)).strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"Total orders: {total_orders}")
            print(f"Buy orders: {buy_orders}")
            print(f"Sell orders: {sell_orders}")
            
            # Fix the formatting issue
            if min_price is not None:
                min_price_str = f"{min_price:.6f}"
            else:
                min_price_str = "N/A"
                
            if max_price is not None:
                max_price_str = f"{max_price:.6f}"
            else:
                max_price_str = "N/A"
                
            if avg_price is not None:
                avg_price_str = f"{avg_price:.6f}"
            else:
                avg_price_str = "N/A"
                
            print(f"Price range: {min_price_str} to {max_price_str}")
            print(f"Average price: {avg_price_str}")
            print(f"Date range: {min_date} to {max_date}")
        else:
            print(f"No data available for {candidate}")
    
    conn.close()

def main():
    print("\n===== Polymarket Presidential Election Orderbook Reconstruction =====")
    print("Focusing on Trump and Harris markets with exact token IDs\n")
    
    # Check if database exists
    db_exists = os.path.exists(DB_FILE)
    
    if not db_exists:
        print("Creating database...")
        create_database()
    
    print("Fetching data from The Graph API using exact token IDs...")
    fetch_all_data()
    
    print("\nReconstructing orderbook...")
    reconstruct_orderbook()
    
    print("\nAnalysis complete!")
    analyze_orderbook()
    
    print(f"\nData stored in {DB_FILE}")
    print("You can now run the HFT orderbook reconstruction with: python hft_example.py")

if __name__ == "__main__":
    main() 