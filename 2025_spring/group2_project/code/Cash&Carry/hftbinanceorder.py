import websocket
import json
import threading
import time
import csv
from datetime import datetime
import ssl
import os
import signal
import sys

class BinanceUSOrderBookDataCapture:
    def __init__(self, symbol='BTCUSD', csv_file=None, update_frequency=1):
        """
        Initialize Binance US Order Book Data Capture
        
        Parameters:
        -----------
        symbol : str, optional
            Trading pair symbol (default is 'BTCUSD')
        csv_file : str, optional
            File path to save order book data (default is generated based on symbol and timestamp)
        update_frequency : int, optional
            How often to print updates to console (in messages, default is 1 for every message)
        """
        self.symbol = symbol.lower()
        self.update_frequency = update_frequency
        
        # Create data directory if it doesn't exist
        self.data_dir = "binance_orderbook_data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # Generate CSV filename with timestamp if not provided
        if csv_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.csv_file = os.path.join(self.data_dir, f'binance_us_{self.symbol}_{timestamp}.csv')
        else:
            self.csv_file = csv_file
        
        # For Binance US, use this socket format - note that some endpoints might require authentication
        self.socket = f'wss://stream.binance.us:9443/ws/{self.symbol}@ticker'
        
        # For tracking connection status
        self.is_connected = False
        self.ws = None
        self.reconnect_delay = 5  # seconds
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        
        # Statistics tracking
        self.messages_received = 0
        self.last_update_time = None
        self.start_time = None
        self.csv_rows_written = 0
        
        # Initialize CSV file with headers
        self.initialize_csv()
        
        print(f"Data will be saved to: {self.csv_file}")
        # Debug settings
        self.debug_mode = True
        self.debug_limit = 10

    def initialize_csv(self):
        """
        Initialize CSV file with headers
        """
        with open(self.csv_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([
                'timestamp_utc',
                'timestamp_local', 
                'best_bid', 
                'best_ask', 
                'weighted_mid', 
                'spread',
                'spread_pct',
                'imbalance'
            ])
        print(f"CSV file initialized with headers: {self.csv_file}")

    def calculate_imbalance(self, bid, ask):
        """
        Calculate order book imbalance
        
        Parameters:
        -----------
        bid : float
            Best bid price
        ask : float
            Best ask price
        
        Returns:
        --------
        float
            Order book imbalance (0-1 range)
        """
        if bid is None or ask is None or bid == 0 or ask == 0:
            return 0
        
        try:
            # Simple imbalance calculation
            imbalance = bid / (bid + ask)
            return max(0, min(1, imbalance))
        except Exception:
            return 0

    def on_message(self, ws, message):
        """
        Process incoming websocket message
        
        Parameters:
        -----------
        ws : websocket connection
        message : str
            JSON message from Binance US websocket
        """
        try:
            # Debug message at the beginning
            if self.debug_mode and self.messages_received < self.debug_limit:
                print(f"Debug - Raw message #{self.messages_received+1}: {message[:200]}...")
            
            # Update statistics
            self.messages_received += 1
            self.last_update_time = datetime.now()
            
            # Parse the message
            data = json.loads(message)
            
            # Try different field names for the ticker data
            # The @ticker stream uses 'b' and 'a' for best bid/ask
            best_bid = None
            best_ask = None
            
            # Try parsing different field combinations
            if 'b' in data and 'a' in data:
                best_bid = float(data['b'])
                best_ask = float(data['a'])
            elif 'bidPrice' in data and 'askPrice' in data:
                best_bid = float(data['bidPrice'])
                best_ask = float(data['askPrice'])
            elif 'bestBid' in data and 'bestAsk' in data:
                best_bid = float(data['bestBid'])
                best_ask = float(data['bestAsk'])
            
            # Debug the extracted values
            if self.debug_mode and self.messages_received < self.debug_limit:
                print(f"Debug - Available fields: {list(data.keys())}")
                print(f"Debug - Extracted values: bid={best_bid}, ask={best_ask}")
            
            # If we couldn't find valid bid/ask prices, try using the last price as a fallback
            if (best_bid is None or best_ask is None) and 'c' in data:
                last_price = float(data['c'])
                if self.debug_mode and self.messages_received < self.debug_limit:
                    print(f"Debug - Using last price as fallback: {last_price}")
                
                # Set both bid and ask to last price with a small spread
                best_bid = last_price * 0.9999
                best_ask = last_price * 1.0001
            
            # Skip if we still don't have valid data
            if best_bid is None or best_ask is None or best_bid == 0 or best_ask == 0:
                if self.debug_mode or self.messages_received % 100 == 0:
                    print(f"Warning: Could not extract valid bid/ask prices from message {self.messages_received}")
                return
            
            # Calculate metrics
            weighted_mid = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            spread_pct = (spread / weighted_mid) * 100 if weighted_mid > 0 else 0
            imbalance = self.calculate_imbalance(best_bid, best_ask)
            
            # Current timestamp (both UTC and local)
            utc_time = datetime.utcnow().isoformat()
            local_time = self.last_update_time.isoformat()
            
            # Prepare row for CSV
            row = [
                utc_time,
                local_time, 
                best_bid, 
                best_ask, 
                weighted_mid,
                spread,
                spread_pct, 
                imbalance
            ]
            
            # Debug the row we're about to write
            if self.debug_mode and self.messages_received < self.debug_limit:
                print(f"Debug - Row to write: {row}")
            
            # Write to CSV file
            try:
                with open(self.csv_file, 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(row)
                    self.csv_rows_written += 1
                    
                    # Debug successful write
                    if self.debug_mode and self.messages_received < self.debug_limit:
                        print(f"Debug - Successfully wrote row #{self.csv_rows_written} to CSV")
                        
                        # Verify file content immediately after writing
                        with open(self.csv_file, 'r') as f:
                            line_count = sum(1 for _ in f)
                        print(f"Debug - CSV now contains {line_count} lines (including header)")
            except Exception as csv_error:
                print(f"Error writing to CSV: {csv_error}")
            
            # Print status based on update frequency
            if self.messages_received % self.update_frequency == 0:
                elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
                rate = self.messages_received / elapsed if elapsed > 0 else 0
                print(f"[{local_time.split('.')[0]}] Bid: ${best_bid:.2f} | Ask: ${best_ask:.2f} | Mid: ${weighted_mid:.2f} | Spread: {spread_pct:.4f}% | Rows written: {self.csv_rows_written}")
            
            # Turn off debug mode after a certain number of messages
            if self.debug_mode and self.messages_received >= self.debug_limit:
                print("Debug mode disabled after first 10 messages")
                self.debug_mode = False
        
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON received: {e}")
            if self.debug_mode:
                print(f"Raw message: {message[:100]}...")
        except Exception as e:
            print(f"Error processing message: {e}")
            if self.debug_mode:
                print(f"Raw message: {message[:100]}...")

    def on_error(self, ws, error):
        """
        Handle websocket errors
        
        Parameters:
        -----------
        ws : websocket connection
        error : Exception
            Error encountered
        """
        print(f"Websocket Error: {error}")
        self.is_connected = False
        
        # Attempt to reconnect
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            print(f"Attempting to reconnect (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}) in {self.reconnect_delay} seconds...")
            time.sleep(self.reconnect_delay)
            self.start_data_capture()
        else:
            print("Maximum reconnection attempts reached. Please restart the script.")

    def on_close(self, ws, close_status_code, close_msg):
        """
        Handle websocket closure
        
        Parameters:
        -----------
        ws : websocket connection
        close_status_code : int
            Status code for websocket closure
        close_msg : str
            Closure message
        """
        self.is_connected = False
        print("### Websocket Connection Closed ###")
        print(f"Close status code: {close_status_code}")
        print(f"Close message: {close_msg}")
        
        # Check if CSV has data
        try:
            file_size = os.path.getsize(self.csv_file)
            print(f"CSV file size: {file_size} bytes")
            
            with open(self.csv_file, 'r') as csvfile:
                line_count = sum(1 for _ in csvfile)
            print(f"CSV contains {line_count} lines (including header)")
            print(f"CSV rows written: {self.csv_rows_written}")
        except Exception as e:
            print(f"Error checking CSV file: {e}")
        
        # Print summary statistics
        if self.start_time:
            total_runtime = (datetime.now() - self.start_time).total_seconds()
            print(f"Total runtime: {total_runtime:.1f} seconds")
            print(f"Messages received: {self.messages_received}")
            print(f"Rows written to CSV: {self.csv_rows_written}")
            print(f"Average message rate: {self.messages_received / total_runtime:.2f} messages/second")
        
        print(f"Data saved to: {self.csv_file}")

    def on_open(self, ws):
        """
        Handle websocket opening
        
        Parameters:
        -----------
        ws : websocket connection
        """
        self.is_connected = True
        self.reconnect_attempts = 0
        self.start_time = datetime.now()
        print(f"Connected to Binance US {self.symbol.upper()} ticker stream")
        print(f"Started at: {self.start_time.isoformat()}")
        print(f"Updates will be printed every {self.update_frequency} message(s)")
        print(f"Press Ctrl+C to stop data collection")
        print(f"Debug mode is ON for first {self.debug_limit} messages")
        
        # Test different streams if needed
        if self.debug_mode:
            # Try subscribing to a different stream type if the current one fails
            stream_types = ["ticker", "bookTicker", "depth"]
            current_stream = self.socket.split('@')[-1]
            print(f"Current stream type: {current_stream}")

    def start_data_capture(self):
        """
        Start the websocket connection and data capture
        """
        # Test CSV file permissions
        try:
            with open(self.csv_file, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['test_write'])
            
            # Remove the test row
            with open(self.csv_file, 'r') as f:
                lines = f.readlines()
            with open(self.csv_file, 'w') as f:
                for i, line in enumerate(lines):
                    if i > 0 and 'test_write' in line:
                        continue
                    f.write(line)
                
            print("CSV file is writable")
        except Exception as e:
            print(f"Warning: CSV file write test failed: {e}")
            print("Will attempt to proceed anyway")
        
        # Create websocket connection with SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Try an alternative method for connecting
        self.ws = websocket.WebSocketApp(
            self.socket,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        
        # Run websocket in a separate thread
        ws_thread = threading.Thread(target=lambda: self.ws.run_forever(
            sslopt={"context": ssl_context},
            ping_interval=30,  # Send ping every 30 seconds
            ping_timeout=10    # Wait 10 seconds for ping response
        ))
        ws_thread.daemon = True
        ws_thread.start()
        
        return ws_thread
    
    def stop_data_capture(self):
        """
        Stop the websocket connection
        """
        if self.ws and self.is_connected:
            print("Closing websocket connection...")
            self.ws.close()
        
        print("Stopping data capture...")
        
        # Final check of CSV file
        try:
            with open(self.csv_file, 'r') as csvfile:
                reader = csv.reader(csvfile)
                row_count = sum(1 for _ in reader)
            print(f"CSV file contains {row_count} rows (including header)")
            
            if row_count <= 1:
                print("WARNING: CSV file appears to be empty (only contains header)")
            else:
                print(f"CSV file contains {row_count - 1} data rows")
        except Exception as e:
            print(f"Error checking final CSV file: {e}")
        
        print(f"Data saved to: {self.csv_file}")
        
        # If no data was captured, suggest alternatives
        if self.csv_rows_written == 0:
            print("\nNo data was captured. Possible solutions:")
            print("1. Try a different trading pair (e.g., ETHUSD instead of BTCUSD)")
            print("2. Check your Internet connection")
            print("3. Verify that Binance US API is accessible from your location")
            print("4. Try a different stream type (modify the code to use @depth, @trade, or @aggTrade)")


def signal_handler(sig, frame):
    """
    Handle Ctrl+C gracefully
    """
    print("\nDetected Ctrl+C, stopping data capture...")
    if 'data_capture' in globals():
        data_capture.stop_data_capture()
    print("Exiting program")
    sys.exit(0)


def main():
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Print banner
    print("=" * 80)
    print(" Binance US Real-Time Order Book Data Capture".center(80))
    print("=" * 80)
    
    # Prompt for symbol, but also print available pairs
    print("\nPopular trading pairs on Binance US:")
    print("BTC/USD, ETH/USD, SOL/USD, XRP/USD, DOGE/USD, ADA/USD")
    symbol = input("\nEnter trading pair (e.g., BTCUSD, default is BTCUSD): ").strip() or 'BTCUSD'
    
    # Ensure symbol is in the correct format
    symbol = symbol.replace('/', '').upper()
    
    # Ask for update frequency
    update_freq_str = input("\nHow often to print updates? (1=every message, 5=every 5th message, default=1): ").strip() or '1'
    update_freq = int(update_freq_str)
    
    # Provide stream options
    print("\nAvailable stream types:")
    print("1. ticker - General ticker information")
    print("2. bookTicker - Best bid/ask information")
    print("3. aggTrade - Aggregated trade information")
    print("4. depth - Order book depth")
    stream_choice = input("Select stream type (1-4, default=1): ").strip() or '1'
    
    # Map stream choice to stream type
    stream_type = {
        '1': 'ticker',
        '2': 'bookTicker',
        '3': 'aggTrade',
        '4': 'depth@100ms'
    }.get(stream_choice, 'ticker')
    
    # Create data capture instance
    global data_capture
    data_capture = BinanceUSOrderBookDataCapture(symbol=symbol, update_frequency=update_freq)
    
    # Override the socket URL based on stream choice
    data_capture.socket = f'wss://stream.binance.us:9443/ws/{symbol.lower()}@{stream_type}'
    print(f"Using stream URL: {data_capture.socket}")
    
    print(f"\nStarting {symbol} Real-Time Order Book Data Capture...")
    print("Press Ctrl+C in terminal to stop the data collection")
    print("\nInitializing websocket connection...")
    
    # Start data capture
    ws_thread = data_capture.start_data_capture()
    
    try:
        # Keep main thread running and show periodic stats
        counter = 0
        while True:
            time.sleep(1)  # Check stats every second
            counter += 1
            
            # Check if file is growing
            if counter % 10 == 0:
                try:
                    if data_capture.csv_rows_written == 0 and data_capture.messages_received > 10:
                        print("WARNING: Messages are being received but no rows are being written to CSV")
                        print("This could indicate a parsing issue. Check the debug output.")
                except Exception:
                    pass
            
            # Only show heartbeat message every 30 seconds
            if counter % 30 == 0 and data_capture.is_connected:
                elapsed = (datetime.now() - data_capture.start_time).total_seconds() if data_capture.start_time else 0
                msg_rate = data_capture.messages_received / elapsed if elapsed > 0 else 0
                print(f"[HEARTBEAT] Running for {elapsed:.1f} seconds | {data_capture.messages_received} messages | {data_capture.csv_rows_written} rows written | {msg_rate:.1f} msg/sec")
    
    except Exception as e:
        print(f"Error in main loop: {e}")
        data_capture.stop_data_capture()


if __name__ == "__main__":
    main()