import os
import time
import hmac
import hashlib
import requests
import base64
import json
import numpy as np
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load API credentials from environment variables
API_KEY = os.getenv("KRAKEN_API_KEY")
API_SECRET = os.getenv("KRAKEN_API_SECRET")

# Verify that environment variables are loaded
if not API_KEY or not API_SECRET:
    raise ValueError("""
    API credentials not found in environment variables!
    Please make sure you have a .env file in the test-trading directory with:
    KRAKEN_API_KEY=your_api_key
    KRAKEN_API_SECRET=your_api_secret
    """)

# Kraken Futures API base URL (demo/testnet)
BASE_URL = "https://demo-futures.kraken.com/derivatives/api/v3"

# Trading parameters
TRADING_PAIRS = ["PI_XBTUSD", "PI_ETHUSD"]  # Bitcoin and Ethereum perpetual futures
POSITION_SIZE = 0.01  # Size of each trade in BTC/ETH
MAX_POSITIONS = 3  # Maximum number of concurrent positions
STOP_LOSS_PERCENT = 0.02  # 2% stop loss
TAKE_PROFIT_PERCENT = 0.03  # 3% take profit

class CryptoTrader:
    def __init__(self):
        self.positions = {}
        self.last_prices = {}
        self.price_history = {}
        self.volatility = {}
        self.trades = []
        self.start_time = datetime.now(timezone.utc)
        self.total_pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.position_direction = {}
        self.last_trade_time = {}
        self.min_trade_interval = 300
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3
        self.cooldown_until = None
        self.peak_pnl = 0
        self.max_drawdown = 0.05  # 5% maximum drawdown
        
        # Create log directory if it doesn't exist
        self.log_dir = "trading_logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Create log file with timestamp
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"trading_summary_{timestamp}.log")
        
        # Write initial header to log file
        with open(self.log_file, 'w') as f:
            f.write("="*20 + " Trading Log " + "="*20 + "\n")
            f.write(f"Start Time: {self.start_time}\n")
            f.write(f"Trading Pairs: {', '.join(TRADING_PAIRS)}\n")
            f.write(f"Position Size: {POSITION_SIZE} BTC/ETH\n")
            f.write("="*60 + "\n\n")

    def print_status(self, symbol, current_price, regime, volatility):
        """Print current trading status"""
        print(f"\n{'='*20} {symbol} Status {'='*20}")
        print(f"Time: {datetime.now(timezone.utc)}")
        print(f"Current Price: ${current_price:,.2f}")
        print(f"Market Regime: {regime}")
        print(f"Volatility: {volatility*100:.2f}%")
        
        if symbol in self.price_history and len(self.price_history[symbol]) >= 20:
            sma20 = np.mean(self.price_history[symbol][-20:])
            print(f"20-period SMA: ${sma20:,.2f}")
            print(f"Distance from SMA: {((current_price - sma20) / sma20 * 100):.2f}%")
        
        print(f"Active Positions: {len(self.positions.get(symbol, []))}")
        print(f"Total Trades Today: {len([t for t in self.trades if t['symbol'] == symbol])}")
        print('='*50)

    def print_trade_summary(self, write_to_file=False):
        """Print daily trading summary"""
        summary = []
        summary.append("\n" + "="*20 + " Daily Trading Summary " + "="*20)
        summary.append(f"Trading Duration: {datetime.now(timezone.utc) - self.start_time}")
        summary.append(f"Total Trades: {len(self.trades)}")
        
        if self.trades:
            profitable_trades = len([t for t in self.trades if t.get('pnl', 0) > 0])
            summary.append(f"Profitable Trades: {profitable_trades}")
            summary.append(f"Win Rate: {(profitable_trades/len(self.trades)*100):.2f}%")
            
            summary.append(f"Total Realized PnL: ${self.realized_pnl:,.2f}")
            summary.append(f"Current Unrealized PnL: ${self.unrealized_pnl:,.2f}")
            summary.append(f"Total PnL: ${self.total_pnl:,.2f}")
            
            # Print recent trades
            summary.append("\nRecent Trades:")
            for trade in self.trades[-5:]:  # Show last 5 trades
                summary.append(f"Time: {trade['timestamp']}")
                summary.append(f"Symbol: {trade['symbol']}")
                summary.append(f"Side: {trade['side']}")
                summary.append(f"Price: ${trade['price']:,.2f}")
                summary.append(f"PnL: ${trade['pnl']:,.2f} ({trade['pnl_percent']:.2f}%)")
                summary.append(f"Status: {trade['status']}")
                summary.append("-" * 30)
        
        summary.append('='*60 + "\n")
        
        # Print to console
        print("\n".join(summary))
        
        # Write to file if requested
        if write_to_file:
            with open(self.log_file, 'a') as f:
                f.write("\n".join(summary) + "\n")

    def sign_request(self, path, nonce, data=""):
        """Sign the request using HMAC SHA256"""
        message = nonce + path + data
        signature = hmac.new(
            API_SECRET.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def get_headers(self, path, data=""):
        """Generate headers for authenticated requests"""
        nonce = str(int(time.time() * 1000))
        signature = self.sign_request(path, nonce, data)
        return {
            "APIKey": API_KEY,
            "Nonce": nonce,
            "Authent": signature
        }

    def get_ticker(self, symbol):
        """Get current price for a symbol"""
        try:
            # Use the correct endpoint for Kraken Futures
            path = f"/tickers"
            headers = self.get_headers(path)
            response = requests.get(BASE_URL + path, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                if 'result' in data and data['result'] == 'success':
                    tickers = data.get('tickers', [])
                    # Find the ticker for our symbol
                    for ticker in tickers:
                        if ticker.get('symbol') == symbol:
                            base_price = float(ticker.get('last', 0))
                            # Add some random price movement
                            price_change = np.random.normal(0, 0.001)  # 0.1% standard deviation
                            current_price = base_price * (1 + price_change)
                            return current_price
                    print(f"No ticker data found for {symbol}")
                else:
                    print(f"API error for {symbol}: {data.get('error', 'Unknown error')}")
            else:
                print(f"Error getting ticker for {symbol}: {response.text}")
            return None
        except Exception as e:
            print(f"Error getting ticker for {symbol}: {e}")
            return None

    def get_historical_prices(self, symbol, interval='1h', limit=100):
        """Get historical price data"""
        try:
            # Get current time in seconds
            end_time = int(time.time())
            # Get time 100 hours ago in seconds
            start_time = end_time - (limit * 3600)
            
            # Use the correct endpoint for Kraken Futures
            path = f"/tickers"
            headers = self.get_headers(path)
            
            print(f"Fetching historical data for {symbol} from {datetime.fromtimestamp(start_time)}")
            response = requests.get(BASE_URL + path, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                if 'result' in data and data['result'] == 'success':
                    tickers = data.get('tickers', [])
                    # Find our symbol's ticker
                    ticker = next((t for t in tickers if t.get('symbol') == symbol), None)
                    if ticker:
                        current_price = float(ticker.get('last', 0))
                        if current_price > 0:
                            # Create a simulated price history using the current price and 24h data
                            history = []
                            
                            # Calculate realistic price bounds
                            # Use 2% range if 24h data is not available or unrealistic
                            min_range = current_price * 0.02
                            max_range = current_price * 0.05  # Maximum 5% range
                            
                            # Get 24h data with validation
                            high_24h = float(ticker.get('high24h', 0))
                            low_24h = float(ticker.get('low24h', 0))
                            
                            # Validate 24h data
                            if (high_24h <= 0 or low_24h <= 0 or 
                                high_24h < current_price or 
                                low_24h > current_price or 
                                high_24h - low_24h > max_range):
                                # Use default ranges if 24h data is invalid
                                high_24h = current_price * 1.01  # 1% above
                                low_24h = current_price * 0.99   # 1% below
                            
                            # Ensure minimum range
                            if high_24h - low_24h < min_range:
                                high_24h = current_price * (1 + min_range/2)
                                low_24h = current_price * (1 - min_range/2)
                            
                            # Final validation of price bounds
                            high_24h = min(high_24h, current_price * 1.05)  # Max 5% above
                            low_24h = max(low_24h, current_price * 0.95)    # Max 5% below
                            
                            volume_24h = float(ticker.get('volume24h', 0))
                            
                            # Generate price history with a trend
                            trend = np.random.choice([-1, 1])  # Random trend direction
                            trend_strength = np.random.uniform(0.0005, 0.001)  # Much stronger trend
                            
                            # Generate 100 hourly candles
                            for i in range(limit):
                                # Calculate base price with trend
                                if i == 0:
                                    price = current_price
                                else:
                                    # Add trend and increased randomness
                                    # Increased volatility to 0.5% per hour
                                    price = history[-1]['close'] * (1 + trend * trend_strength + np.random.normal(0, 0.005))
                                    # Ensure price stays within bounds
                                    price = max(min(price, high_24h), low_24h)
                                
                                # Create a candle with more volatile price movement
                                # Increased candle range to 0.3-0.8% per candle
                                candle_range = np.random.uniform(0.003, 0.008)
                                open_price = price * (1 + np.random.normal(0, candle_range/2))
                                high_price = max(open_price, price) * (1 + abs(np.random.normal(0, candle_range/2)))
                                low_price = min(open_price, price) * (1 - abs(np.random.normal(0, candle_range/2)))
                                close_price = price
                                
                                # Ensure high is highest and low is lowest
                                high_price = max(open_price, close_price, high_price)
                                low_price = min(open_price, close_price, low_price)
                                
                                # Ensure all prices are positive and within bounds
                                high_price = min(high_price, high_24h)
                                low_price = max(low_price, low_24h)
                                
                                history.append({
                                    'time': int((start_time + i * 3600) * 1000),
                                    'open': open_price,
                                    'high': high_price,
                                    'low': low_price,
                                    'close': close_price,
                                    'volume': volume_24h / limit * (1 + np.random.normal(0, 0.1))  # Add some volume variation
                                })
                            
                            print(f"Generated {len(history)} historical prices for {symbol}")
                            print(f"Current price: ${current_price:,.2f}")
                            print(f"24h range: ${low_24h:,.2f} - ${high_24h:,.2f}")
                            return history
                    print(f"No ticker data found for {symbol}")
                else:
                    print(f"API error for {symbol}: {data.get('error', 'Unknown error')}")
            else:
                print(f"Error getting ticker data for {symbol}: {response.text}")
            return []
        except Exception as e:
            print(f"Error getting historical prices for {symbol}: {e}")
            return []

    def calculate_volatility(self, symbol):
        """Calculate price volatility"""
        prices = self.get_historical_prices(symbol)
        if len(prices) < 2:
            print(f"Not enough price data for {symbol} to calculate volatility")
            return 0
        
        try:
            # Extract closing prices
            price_array = np.array([float(p.get('close', 0)) for p in prices if 'close' in p])
            if len(price_array) < 2:
                print(f"Invalid price data for {symbol}. Available fields: {list(prices[0].keys()) if prices else 'No data'}")
                return 0
                
            returns = np.diff(price_array) / price_array[:-1]
            volatility = np.std(returns) * np.sqrt(24)  # Annualized volatility
            print(f"Calculated volatility for {symbol}: {volatility*100:.2f}%")
            return volatility
        except Exception as e:
            print(f"Error calculating volatility for {symbol}: {e}")
            return 0

    def detect_market_regime(self, symbol):
        """Detect if market is trending or ranging"""
        prices = self.get_historical_prices(symbol)
        if len(prices) < 20:
            print(f"Not enough price data for {symbol} to detect market regime")
            return "unknown"
        
        try:
            # Extract closing prices
            price_array = np.array([float(p.get('close', 0)) for p in prices if 'close' in p])
            if len(price_array) < 20:
                print(f"Invalid price data for {symbol}. Available fields: {list(prices[0].keys()) if prices else 'No data'}")
                return "unknown"
                
            sma20 = np.mean(price_array[-20:])
            current_price = price_array[-1]
            
            # Calculate trend strength
            trend_strength = abs(current_price - sma20) / sma20
            
            # Reduced threshold from 0.02 to 0.01 (1% deviation)
            if trend_strength > 0.01:
                print(f"Trend detected for {symbol}: {trend_strength*100:.2f}% deviation from SMA20")
                return "trending"
            print(f"Ranging market for {symbol}: {trend_strength*100:.2f}% deviation from SMA20")
            return "ranging"
        except Exception as e:
            print(f"Error detecting market regime for {symbol}: {e}")
            return "unknown"

    def calculate_position_pnl(self, position, current_price):
        """Calculate PnL for a single position"""
        entry_price = position.get('price', 0)
        side = position.get('side', '')
        size = position.get('size', POSITION_SIZE)
        
        if entry_price > 0:
            if side == 'buy':
                pnl = (current_price - entry_price) * size
                pnl_percent = (current_price - entry_price) / entry_price * 100
            else:
                pnl = (entry_price - current_price) * size
                pnl_percent = (entry_price - current_price) / entry_price * 100
            return pnl, pnl_percent
        return 0.0, 0.0

    def close_position(self, symbol, position, current_price, reason=""):
        """Close a position and update PnL"""
        side = position.get('side', '')
        pnl, pnl_percent = self.calculate_position_pnl(position, current_price)
        
        print(f"\nClosing {symbol} {side} position:")
        print(f"Entry: ${position['price']:,.2f}")
        print(f"Exit: ${current_price:,.2f}")
        print(f"PnL: ${pnl:,.2f} ({pnl_percent:.2f}%)")
        if reason:
            print(f"Reason: {reason}")
        
        # Place closing order
        self.place_order(symbol, "sell" if side == "buy" else "buy", POSITION_SIZE)
        
        # Update PnL tracking
        self.realized_pnl += pnl
        self.total_pnl += pnl
        
        # Remove position
        self.positions[symbol].remove(position)
        if not self.positions[symbol]:
            self.position_direction[symbol] = None

    def update_pnl(self, symbol, current_price):
        """Update PnL for all positions"""
        self.unrealized_pnl = 0.0
        for position in self.positions.get(symbol, []):
            pnl, _ = self.calculate_position_pnl(position, current_price)
            self.unrealized_pnl += pnl

    def can_trade(self, symbol):
        """Check if we can trade based on various conditions"""
        # Check if we're in cooldown
        if self.cooldown_until and datetime.now(timezone.utc) < self.cooldown_until:
            print(f"In cooldown until {self.cooldown_until}")
            return False
            
        # Check if we've hit maximum consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            print(f"Maximum consecutive losses reached ({self.consecutive_losses})")
            self.cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=5)
            self.consecutive_losses = 0
            return False
            
        # Check if we've hit maximum drawdown
        if self.total_pnl > self.peak_pnl:
            self.peak_pnl = self.total_pnl
        current_drawdown = (self.peak_pnl - self.total_pnl) / abs(self.peak_pnl) if self.peak_pnl != 0 else 0
        if current_drawdown > self.max_drawdown:
            print(f"Maximum drawdown reached ({current_drawdown*100:.2f}%)")
            self.cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=10)
            return False
            
        # Check minimum trade interval
        if symbol not in self.last_trade_time:
            return True
        time_since_last_trade = (datetime.now(timezone.utc) - self.last_trade_time[symbol]).total_seconds()
        return time_since_last_trade >= self.min_trade_interval

    def place_order(self, symbol, side, size, price=None, order_type="market"):
        """Place an order"""
        try:
            path = "/sendorder"
            order = {
                "order": {
                    "symbol": symbol,
                    "side": side,
                    "orderType": order_type,
                    "size": str(size),
                }
            }
            
            if price and order_type == "limit":
                order["order"]["limitPrice"] = str(price)
            
            body = json.dumps(order)
            headers = self.get_headers(path, body)
            
            print(f"\nPlacing {side} order for {size} {symbol}...")
            response = requests.post(BASE_URL + path, headers=headers, data=body)
            
            if response.status_code == 200:
                order_data = response.json()
                print(f"Order placed successfully!")
                print(f"Order ID: {order_data.get('order_id', 'N/A')}")
                print(f"Status: {order_data.get('status', 'N/A')}")
                
                # Get current price for PnL calculation
                current_price = self.get_ticker(symbol)
                
                # Record the trade
                trade = {
                    'timestamp': datetime.now(timezone.utc),
                    'symbol': symbol,
                    'side': side,
                    'size': size,
                    'price': price or current_price,
                    'order_id': order_data.get('order_id'),
                    'pnl': 0.0,
                    'pnl_percent': 0.0,
                    'status': 'open'
                }
                
                # If this is a closing order, calculate PnL
                if symbol in self.positions:
                    for position in self.positions[symbol]:
                        if position['side'] != side:  # Opposite side means closing
                            pnl, pnl_percent = self.calculate_position_pnl(position, current_price)
                            trade['pnl'] = pnl
                            trade['pnl_percent'] = pnl_percent
                            self.realized_pnl += pnl
                            self.total_pnl += pnl
                            position['status'] = 'closed'
                            
                            # Update consecutive losses
                            if pnl < 0:
                                self.consecutive_losses += 1
                            else:
                                self.consecutive_losses = 0
                                
                            print(f"Position closed with PnL: ${pnl:,.2f} ({pnl_percent:.2f}%)")
                
                self.trades.append(trade)
                
                # Update positions
                if symbol not in self.positions:
                    self.positions[symbol] = []
                if trade['status'] == 'open':
                    self.positions[symbol].append(trade)
                    self.last_trade_time[symbol] = datetime.now(timezone.utc)
                
                return order_data
            else:
                print(f"Error placing order: {response.text}")
                return None
        except Exception as e:
            print(f"Error placing order: {e}")
            return None

    def implement_strategy(self):
        print("------------------Trading Strategy------------------")
        for symbol in TRADING_PAIRS:
            current_price = self.get_ticker(symbol)
            if not current_price:
                print(f"Could not get current price for {symbol}")
                continue
                
            # Update price history
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            self.price_history[symbol].append(current_price)
            if len(self.price_history[symbol]) > 100:
                self.price_history[symbol].pop(0)
            
            # Calculate volatility
            self.volatility[symbol] = self.calculate_volatility(symbol)
            
            # Detect market regime
            regime = self.detect_market_regime(symbol)
            
            # Update PnL for current positions
            self.update_pnl(symbol, current_price)
            
            # Print current status
            self.print_status(symbol, current_price, regime, self.volatility[symbol])
            
            # Get current positions for this symbol
            current_positions = self.positions.get(symbol, [])
            
            print(f"\nDetailed Analysis for {symbol}:")
            print(f"Market Regime: {regime}")
            print(f"Current Price: ${current_price:,.2f}")
            print(f"Volatility: {self.volatility[symbol]*100:.2f}%")
            print(f"Active Positions: {len(current_positions)}")
            print(f"Current Unrealized PnL: ${self.unrealized_pnl:,.2f}")
            print(f"Consecutive Losses: {self.consecutive_losses}")
            
            # Initialize indicators
            sma20 = None
            sma50 = None
            std20 = None
            price_momentum = None
            
            # Calculate indicators if we have enough data
            if len(self.price_history[symbol]) >= 20:
                sma20 = np.mean(self.price_history[symbol][-20:])
                sma50 = np.mean(self.price_history[symbol][-50:]) if len(self.price_history[symbol]) >= 50 else sma20
                std20 = np.std(self.price_history[symbol][-20:])
                
                # Print trend analysis
                print(f"\nTrend Following Analysis:")
                print(f"SMA20: ${sma20:,.2f}")
                print(f"SMA50: ${sma50:,.2f}")
                print(f"Price vs SMA20: {((current_price - sma20) / sma20 * 100):.2f}%")
                print(f"SMA20 vs SMA50: {((sma20 - sma50) / sma50 * 100):.2f}%")
                
                # Print mean reversion analysis
                print(f"\nMean Reversion Analysis:")
                print(f"SMA20: ${sma20:,.2f}")
                print(f"Standard Deviation: ${std20:,.2f}")
                print(f"Upper Band (SMA20 + 1.0*std): ${(sma20 + 1.0 * std20):,.2f}")
                print(f"Lower Band (SMA20 - 1.0*std): ${(sma20 - 1.0 * std20):,.2f}")
                print(f"Distance from SMA20: {((current_price - sma20) / sma20 * 100):.2f}%")
                print(f"Distance in std devs: {((current_price - sma20) / std20):.2f}")
            
            # Calculate momentum if we have enough data
            if len(self.price_history[symbol]) >= 10:
                recent_prices = self.price_history[symbol][-10:]
                price_momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                
                print(f"\nMomentum Analysis:")
                print(f"10-period momentum: {price_momentum*100:.2f}%")
                print(f"Price change: ${recent_prices[-1] - recent_prices[0]:,.2f}")
                print(f"Start price: ${recent_prices[0]:,.2f}")
                print(f"End price: ${recent_prices[-1]:,.2f}")
            
            # Calculate recent performance
            recent_trades = [t for t in self.trades if t['symbol'] == symbol][-5:]
            if recent_trades:
                recent_win_rate = len([t for t in recent_trades if t.get('pnl', 0) > 0]) / len(recent_trades)
                print(f"\nRecent Performance:")
                print(f"Last 5 trades win rate: {recent_win_rate*100:.2f}%")
            
            # Check existing positions first
            for position in current_positions[:]:
                pnl, pnl_percent = self.calculate_position_pnl(position, current_price)
                
                print(f"\nPosition Analysis for {symbol} {position['side']} position:")
                print(f"Entry: ${position['price']:,.2f}")
                print(f"Current: ${current_price:,.2f}")
                print(f"PnL: ${pnl:,.2f} ({pnl_percent:.2f}%)")
                
                # Close positions based on conditions
                if pnl_percent < -STOP_LOSS_PERCENT * 100:
                    self.close_position(symbol, position, current_price, "Stop loss triggered")
                elif pnl_percent > TAKE_PROFIT_PERCENT * 100:
                    self.close_position(symbol, position, current_price, "Take profit triggered")
                elif sma20 is not None:  # Only check trend-based conditions if we have SMA20
                    if regime == "trending":
                        # Only close on strong trend reversals
                        if position['side'] == "sell" and current_price > sma20 * 1.005:  # 0.5% above SMA20
                            self.close_position(symbol, position, current_price, "Strong trend reversal")
                        elif position['side'] == "buy" and current_price < sma20 * 0.995:  # 0.5% below SMA20
                            self.close_position(symbol, position, current_price, "Strong trend reversal")
                    elif regime == "ranging":
                        # Only close on strong mean reversion
                        if abs((current_price - sma20) / sma20) < 0.0005:  # Within 0.05% of mean
                            self.close_position(symbol, position, current_price, "Strong mean reversion")
            
            # Only open new positions if we don't have any, have enough data, and enough time has passed
            if not current_positions and sma20 is not None and std20 is not None and self.can_trade(symbol):
                # Adjust thresholds based on recent performance and consecutive losses
                recent_trades = [t for t in self.trades if t['symbol'] == symbol][-5:]
                if recent_trades:
                    recent_win_rate = len([t for t in recent_trades if t.get('pnl', 0) > 0]) / len(recent_trades)
                    # If recent performance is poor or we have consecutive losses, require stronger signals
                    if recent_win_rate < 0.4 or self.consecutive_losses > 0:
                        trend_threshold = 0.008  # 0.8% for trending
                        mean_rev_threshold = 2.0  # 2.0 std devs for ranging
                    else:
                        trend_threshold = 0.005  # 0.5% for trending
                        mean_rev_threshold = 1.5  # 1.5 std devs for ranging
                else:
                    trend_threshold = 0.005
                    mean_rev_threshold = 1.5
                
                if regime == "trending":
                    # Require stronger trend signals
                    if current_price > sma20 * (1 + trend_threshold):  # Above SMA20
                        print(f"\nStrong uptrend detected!")
                        print(f"Price: ${current_price:,.2f} > SMA20: ${sma20:,.2f}")
                        self.place_order(symbol, "buy", POSITION_SIZE)
                        self.position_direction[symbol] = "buy"
                    elif current_price < sma20 * (1 - trend_threshold):  # Below SMA20
                        print(f"\nStrong downtrend detected!")
                        print(f"Price: ${current_price:,.2f} < SMA20: ${sma20:,.2f}")
                        self.place_order(symbol, "sell", POSITION_SIZE)
                        self.position_direction[symbol] = "sell"
                
                elif regime == "ranging":
                    # Require stronger mean reversion signals
                    if current_price < sma20 - mean_rev_threshold * std20:  # Below mean
                        print(f"\nStrong oversold condition detected!")
                        print(f"Price: ${current_price:,.2f} < Lower Band: ${(sma20 - mean_rev_threshold * std20):,.2f}")
                        self.place_order(symbol, "buy", POSITION_SIZE)
                        self.position_direction[symbol] = "buy"
                    elif current_price > sma20 + mean_rev_threshold * std20:  # Above mean
                        print(f"\nStrong overbought condition detected!")
                        print(f"Price: ${current_price:,.2f} > Upper Band: ${(sma20 + mean_rev_threshold * std20):,.2f}")
                        self.place_order(symbol, "sell", POSITION_SIZE)
                        self.position_direction[symbol] = "sell"
            
            print(f"\n{'-'*50}")  # Add separator between symbols

    def run(self):
        """Main trading loop"""
        print("\n" + "="*20 + " Starting Crypto Trading System " + "="*20)
        print(f"Start Time: {self.start_time}")
        print(f"Trading Pairs: {', '.join(TRADING_PAIRS)}")
        print(f"Position Size: {POSITION_SIZE} BTC/ETH")
        print("="*70 + "\n")
        
        iteration = 0
        while True:
            try:
                iteration += 1
                print(f"\nIteration {iteration} - {datetime.now(timezone.utc)}")
                print("-" * 50)
                
                # Implement trading strategies
                self.implement_strategy()
                
                # Print daily summary every 10 iterations
                if iteration % 10 == 0:
                    self.print_trade_summary(write_to_file=True)
                
                # Sleep for 5 seconds
                time.sleep(5)
                
            except KeyboardInterrupt:
                print("\n" + "="*20 + " Stopping Trading System " + "="*20)
                self.print_trade_summary(write_to_file=True)
                print(f"\nTrading log saved to: {self.log_file}")
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(5)

if __name__ == "__main__":
    trader = CryptoTrader()
    trader.run()
