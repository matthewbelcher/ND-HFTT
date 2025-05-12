import os
import time
import json
import sys
import csv
from datetime import datetime, timedelta
from dotenv import load_dotenv
from coinbase.rest import RESTClient
import threading

load_dotenv()

class CoinbaseTradeTracker:
    def __init__(self, product_id='BTC-USD', verbose=False):
        self.client = RESTClient(
            api_key=os.getenv("COINBASE_KEY"),
            api_secret=os.getenv("COINBASE_SECRETKEY"),
            verbose=verbose
        )
        self.product_id = product_id
        self.csv_dir = "trade_data"
        self.raw_dir = "raw_responses"
        self.running = False
        
        # Create directories if they don't exist
        for directory in [self.csv_dir, self.raw_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                
        # Initialize the CSV file with headers
        self.current_date = datetime.now().strftime("%Y%m%d")
        self.csv_filename = os.path.join(self.csv_dir, f"trades_{self.product_id}_{self.current_date}.csv")
        self.initialize_csv_file()
                
        print(f"Initialized CoinbaseTradeTracker for {product_id}")
        
    def initialize_csv_file(self):
        """Create the CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.csv_filename):
            with open(self.csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['timestamp', 'trade_id', 'price', 'size', 'side', 'spread', 'bid', 'ask'])
            print(f"Created new CSV file: {self.csv_filename}")
        
    def get_trades(self):
        """Get recent trades for the product"""
        try:
            endpoint = f"/api/v3/brokerage/products/{self.product_id}/ticker"
            response = self.client.get(endpoint)
            return response
        except Exception as e:
            print(f"Error fetching trades: {e}")
            return None
            
    def get_orderbook(self, limit=3):
        """Get the current orderbook for the product to calculate spread"""
        try:
            endpoint = f"/api/v3/brokerage/products/{self.product_id}/book"
            params = {"limit": limit}
            
            response = self.client.get(endpoint, params=params)
            return response
        except Exception as e:
            print(f"Error fetching orderbook: {e}")
            return None
            
    def save_raw_response(self, response, prefix):
        """Save raw response to a file in the raw_responses directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.raw_dir, f"{prefix}_{timestamp}.json")
        
        try:
            with open(filename, 'w') as f:
                json.dump(response, f, indent=2)
            return filename
        except Exception as e:
            print(f"Error saving raw response: {e}")
            return None
            
    def record_trade(self, trade_data, orderbook_data=None):
        """Record a trade to CSV"""
        if not trade_data:
            print("No trade data to record")
            return False
        
        # Calculate spread if orderbook data is available
        spread = None
        bid = None
        ask = None
        
        if orderbook_data and 'pricebook' in orderbook_data:
            bids = orderbook_data.get('pricebook', {}).get('bids', [])
            asks = orderbook_data.get('pricebook', {}).get('asks', [])
            
            if bids:
                bid = float(bids[0].get('price', 0))
            
            if asks:
                ask = float(asks[0].get('price', 0))
            
            if bid and ask:
                spread = ask - bid
        
        # Parse trade data
        timestamp = trade_data.get('time')
        trade_id = trade_data.get('trade_id')
        price = trade_data.get('price')
        size = trade_data.get('size')
        side = trade_data.get('side')  # SELL or BUY
        
        # Write data to CSV
        try:
            with open(self.csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    timestamp,
                    trade_id,
                    price,
                    size,
                    side,
                    spread,
                    bid,
                    ask
                ])
            return True
        except Exception as e:
            print(f"Error writing to CSV: {e}")
            return False
    
    def start_tracking(self, interval=1.0):
        """Start tracking trades at the specified interval (in seconds)"""
        self.running = True
        last_trade_id = None
        
        try:
            while self.running:
                start_time = time.time()
                
                # Check if we need a new file for a new day
                now = datetime.now()
                new_date = now.strftime("%Y%m%d")
                if new_date != self.current_date:
                    self.current_date = new_date
                    self.csv_filename = os.path.join(self.csv_dir, f"trades_{self.product_id}_{self.current_date}.csv")
                    self.initialize_csv_file()
                
                # Get the latest trade
                trade_data = self.get_trades()
                
                # Get orderbook for spread calculation
                orderbook_data = self.get_orderbook()
                
                # Save raw responses if available
                if trade_data:
                    self.save_raw_response(trade_data, f"trade_{self.product_id}")
                if orderbook_data:
                    self.save_raw_response(orderbook_data, f"orderbook_{self.product_id}")
                
                # Record trade if it's new
                if trade_data and 'trade_id' in trade_data:
                    current_trade_id = trade_data['trade_id']
                    if current_trade_id != last_trade_id:
                        success = self.record_trade(trade_data, orderbook_data)
                        if success:
                            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Recorded trade: {trade_data.get('side')} {trade_data.get('size')} @ {trade_data.get('price')}")
                            last_trade_id = current_trade_id
                
                # Calculate sleep time to maintain the interval
                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            print("\nStopping trade tracking...")
        except Exception as e:
            print(f"Error in tracking loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
    
    def stop_tracking(self):
        """Stop the trade tracking"""
        self.running = False


def analyze_trade_data(csv_filename):
    """Analyze the trade data collected"""
    if not os.path.exists(csv_filename):
        print(f"File {csv_filename} does not exist")
        return
    
    try:
        buys = 0
        sells = 0
        buy_volume = 0.0
        sell_volume = 0.0
        trades = 0
        
        with open(csv_filename, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                trades += 1
                side = row.get('side', '').upper()
                size = float(row.get('size', 0))
                
                if side == 'BUY':
                    buys += 1
                    buy_volume += size
                elif side == 'SELL':
                    sells += 1
                    sell_volume += size
        
        if trades == 0:
            print(f"CSV file exists but contains no trade data (only header)")
            return
            
        print(f"\nTrade Analysis:")
        print(f"- Total trades: {trades}")
        print(f"- Buy trades: {buys} ({buys/trades*100:.1f}%)")
        print(f"- Sell trades: {sells} ({sells/trades*100:.1f}%)")
        print(f"- Buy volume: {buy_volume:.8f}")
        print(f"- Sell volume: {sell_volume:.8f}")
        print(f"- Net volume (buy-sell): {buy_volume-sell_volume:.8f}")
            
    except Exception as e:
        print(f"Error analyzing trade data: {e}")
        import traceback
        traceback.print_exc()


def main():
    product_id = 'BIT-25APR25-CDE'  # Set to a default that should work
    
    # Allow command line override of product_id
    if len(sys.argv) > 1:
        product_id = sys.argv[1]
    
    print(f"Starting Coinbase Trade Tracker for {product_id}")
    print("This script will continuously capture trade data second by second.")
    print("Press Ctrl+C to stop tracking.")
    print()
    
    tracker = CoinbaseTradeTracker(product_id=product_id, verbose=True)
    
    # Start tracking in the main thread
    try:
        print(f"Beginning trade tracking at 1-second intervals...")
        print(f"Raw responses will be saved to: {tracker.raw_dir}")
        print(f"CSV files will be saved to: {tracker.csv_filename}")
        print()
        
        tracker.start_tracking(interval=1.0)
    except KeyboardInterrupt:
        print("\nStopping trade tracking...")
    finally:
        # After tracking is stopped, analyze the collected data
        print("\nAnalyzing collected trade data...")
        analyze_trade_data(tracker.csv_filename)
        
        print("\nTrade tracking complete.")
        print(f"- Raw JSON responses saved to: {tracker.raw_dir}")
        print(f"- CSV trade data saved to: {tracker.csv_filename}")


if __name__ == "__main__":
    main()