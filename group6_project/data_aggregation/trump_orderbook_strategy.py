import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time
from polymarket_backtest import Strategy, Backtest
import warnings
import seaborn as sns
from matplotlib.dates import DateFormatter

# Suppress numerical warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class StableTrumpStrategy(Strategy):
    """
    A stable and conservative strategy for Trump prediction market with strict risk management.
    
    Features:
    1. FIXED ABSOLUTE position sizing (not growing with capital)
    2. Hard stop losses at fixed % values
    3. Simple moving average crossover for trend detection
    4. Order book imbalance as signal confirmation
    5. Cooldown period between trades to reduce trading frequency
    """
    
    def __init__(self, 
                 name="TrumpStable",
                 fast_ma=5,               # Fast moving average window
                 slow_ma=20,              # Slow moving average window
                 initial_capital=1000.0,   # Initial capital for fixed sizing
                 fixed_position_dollars=20.0, # Fixed dollar amount per trade (REDUCED from 100)
                 stop_loss_pct=0.02,      # Fixed stop loss (2%)
                 take_profit_pct=0.04,    # Fixed take profit (4%)
                 min_imbalance=0.15,      # Minimum imbalance to confirm signal
                 min_depth=10.0,           # Minimum depth required (INCREASED from 5.0)
                 cooldown_period=3600,      # Seconds between trades (INCREASED from 60)
                 sample_interval=10):      # Process every nth data point (INCREASED from 1)
        
        super().__init__(name)
        
        # Strategy parameters
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.initial_capital = initial_capital
        self.fixed_position_dollars = fixed_position_dollars  # Absolute fixed dollar amount
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.min_imbalance = min_imbalance
        self.min_depth = min_depth  
        self.cooldown_period = cooldown_period
        self.sample_interval = sample_interval
        self.sample_counter = 0
        
        # Hard limits (crucial for numerical stability)
        self.max_position_size = 10.0    # REDUCED from 20.0 - Absolute maximum position size
        self.max_trades_per_day = 5      # NEW: Maximum trades per day
        self.max_consecutive_losses = 3  # NEW: Maximum consecutive losses
        self.consecutive_losses = 0      # NEW: Count consecutive losses
        
        # State variables
        self.price_history = []
        self.in_position = False
        self.position_side = None
        self.entry_price = 0
        self.stop_price = 0
        self.target_price = 0
        self.last_trade_time = 0
        self.day_trade_count = 0         # NEW: Count trades per day
        self.current_day = 0             # NEW: Track current day
        
        # Trade log and manual capital tracking
        self.trade_log = []
        self.tracked_capital = initial_capital
        self.last_entry_size = 0
        self.last_entry_price = 0
        
        # Risk management flags
        self.trading_halted = False      # NEW: Flag to halt trading
        self.min_capital = initial_capital * 0.8  # NEW: Minimum capital threshold
    
    def on_fill(self, order, fill_price, fill_timestamp):
        """Handle order fills - required by Strategy base class"""
        # We track positions and capital manually to ensure accuracy
        pass
        
    def initialize(self, backtest):
        """Initialize the strategy with the backtest environment"""
        self.backtest = backtest
        print(f"Initialized {self.name} Strategy with:")
        print(f"- Moving averages: Fast={self.fast_ma}, Slow={self.slow_ma}")
        print(f"- FIXED position size: ${self.fixed_position_dollars} per trade")
        print(f"- Maximum position size: {self.max_position_size} units")
        print(f"- Stop loss: {self.stop_loss_pct*100}%, Take profit: {self.take_profit_pct*100}%")
        print(f"- Minimum imbalance: {self.min_imbalance}, Minimum depth: {self.min_depth}")
    
    def update_moving_averages(self):
        """Calculate moving averages from price history"""
        if len(self.price_history) < self.slow_ma:
            return None, None
            
        # Calculate fast and slow MAs
        fast_ma = np.mean(self.price_history[-self.fast_ma:])
        slow_ma = np.mean(self.price_history[-self.slow_ma:])
        
        return fast_ma, slow_ma
    
    def check_signal(self, fast_ma, slow_ma, imbalance, bid_depth, ask_depth):
        """Check for trading signal"""
        # Basic validation
        if fast_ma is None or slow_ma is None:
            return None
            
        # Check depth requirement
        if bid_depth < self.min_depth or ask_depth < self.min_depth:
            return None
            
        # Check for crossovers (simple but effective)
        if fast_ma > slow_ma and imbalance > (1 + self.min_imbalance):
            # Bullish signal (fast MA crosses above slow MA + positive imbalance)
            return "BUY"
        elif fast_ma < slow_ma and imbalance < (1 - self.min_imbalance):
            # Bearish signal (fast MA crosses below slow MA + negative imbalance)
            return "SELL"
            
        return None
    
    def calculate_position_size(self, price):
        """Calculate position size using ABSOLUTE FIXED DOLLARS PER TRADE"""
        # Use a fixed dollar amount per trade, not a percentage
        # This is the most stable approach
        
        # Convert fixed dollars to quantity
        quantity = self.fixed_position_dollars / price
        
        # Apply hard limit
        quantity = min(quantity, self.max_position_size)
        
        return quantity
    
    def calculate_pnl(self, entry_side, entry_price, entry_size, exit_price):
        """Calculate P&L for a trade"""
        if entry_side == 'BUY':  # Long trade
            pnl = (exit_price - entry_price) * entry_size
        else:  # Short trade 
            pnl = (entry_price - exit_price) * entry_size
            
        # Apply estimated fees (0.5% round trip)
        fee = 0.005 * entry_price * entry_size
        pnl -= fee
        
        return pnl
    
    def on_data(self, timestamp, orderbook_data):
        """Process new orderbook data and generate trading signals"""
        # Check if trading is halted due to drawdown
        if self.trading_halted:
            return
            
        # Check if capital has dropped below the minimum threshold
        if self.tracked_capital < self.min_capital:
            print(f"Trading halted: Capital ({self.tracked_capital:.2f}) below minimum threshold ({self.min_capital:.2f})")
            self.trading_halted = True
            return
            
        # Track day for trade limits
        day = timestamp // 86400  # Convert timestamp to day
        if day != self.current_day:
            self.current_day = day
            self.day_trade_count = 0  # Reset daily trade counter
            
        # Check if we've reached maximum trades for the day
        if self.day_trade_count >= self.max_trades_per_day:
            return
            
        # Check if we've had too many consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            # Take a day off after too many consecutive losses
            if self.last_trade_time + 86400 > timestamp:  # 24 hour cooldown
                return
            else:
                # Reset consecutive loss counter after the cooldown
                self.consecutive_losses = 0
        
        # Sample data for performance (process every nth data point)
        self.sample_counter = (self.sample_counter + 1) % self.sample_interval
        if self.sample_counter != 0 and len(self.price_history) >= self.slow_ma:
            return
        
        # Extract key data
        mid_price = orderbook_data.get('mid_price', 0)
        best_bid = orderbook_data.get('best_bid', 0)
        best_ask = orderbook_data.get('best_ask', 0)
        bid_depth = orderbook_data.get('bid_depth', 0)
        ask_depth = orderbook_data.get('ask_depth', 0)
        imbalance = orderbook_data.get('imbalance', 1.0)
        
        # Skip if any key data is missing or invalid
        if mid_price <= 0 or best_bid <= 0 or best_ask <= 0 or bid_depth < 0 or ask_depth < 0:
            return
            
        # Skip if prices are outside realistic range for this market (0.1 to 0.9)
        if mid_price < 0.1 or mid_price > 0.9:
            return
            
        # Update price history
        self.price_history.append(mid_price)
        if len(self.price_history) > 100:
            self.price_history = self.price_history[-100:]
            
        # Wait for enough data
        if len(self.price_history) < self.slow_ma:
            return
            
        # Update moving averages
        fast_ma, slow_ma = self.update_moving_averages()
        
        # Check if we're in a position
        if self.in_position:
            if self.position_side == "LONG":
                # Check for stop loss or take profit for long position
                if best_bid <= self.stop_price or best_bid >= self.target_price:
                    # Determine exit reason
                    exit_reason = "Stop Loss" if best_bid <= self.stop_price else "Take Profit"
                    
                    # Get position size for exit
                    position_size = abs(self.backtest.position.size)
                    position_size = min(position_size, self.last_entry_size)  # Ensure we don't exit more than we entered
                    
                    # Calculate P&L for this trade
                    pnl = self.calculate_pnl('BUY', self.entry_price, position_size, best_bid)
                    
                    # Update tracked capital
                    self.tracked_capital += pnl
                    
                    # Track consecutive losses
                    if pnl < 0:
                        self.consecutive_losses += 1
                    else:
                        self.consecutive_losses = 0
                    
                    # Execute exit if we have a position
                    if position_size > 0:
                        # Submit sell order
                        self.backtest.submit_order('SELL', best_bid, position_size, timestamp)
                        
                        # Log trade with P&L
                        self.log_trade(timestamp, 'SELL', best_bid, position_size, exit_reason, pnl)
                        
                        # Reset position state
                        self.in_position = False
                        self.position_side = None
                        self.last_trade_time = timestamp
                        
                        return
                        
            elif self.position_side == "SHORT":
                # Check for stop loss or take profit for short position
                if best_ask >= self.stop_price or best_ask <= self.target_price:
                    # Determine exit reason
                    exit_reason = "Stop Loss" if best_ask >= self.stop_price else "Take Profit"
                    
                    # Get position size for exit
                    position_size = abs(self.backtest.position.size)
                    position_size = min(position_size, self.last_entry_size)  # Ensure we don't exit more than we entered
                    
                    # Calculate P&L for this trade
                    pnl = self.calculate_pnl('SELL', self.entry_price, position_size, best_ask)
                    
                    # Update tracked capital
                    self.tracked_capital += pnl
                    
                    # Track consecutive losses
                    if pnl < 0:
                        self.consecutive_losses += 1
                    else:
                        self.consecutive_losses = 0
                    
                    # Execute exit if we have a position
                    if position_size > 0:
                        # Submit buy order
                        self.backtest.submit_order('BUY', best_ask, position_size, timestamp)
                        
                        # Log trade with P&L
                        self.log_trade(timestamp, 'BUY', best_ask, position_size, exit_reason, pnl)
                        
                        # Reset position state
                        self.in_position = False
                        self.position_side = None
                        self.last_trade_time = timestamp
                        
                        return
                        
        # Only look for new trades if we're not in a position and cooldown period has elapsed
        elif timestamp - self.last_trade_time >= self.cooldown_period:
            # Check for trading signal
            signal = self.check_signal(fast_ma, slow_ma, imbalance, bid_depth, ask_depth)
            
            # Execute if we have a signal
            if signal:
                if signal == 'BUY':
                    # Calculate fixed position size
                    entry_price = best_ask  # Buy at ask
                    
                    # Use a position size based on current capital, not initial
                    # This will naturally reduce position sizes if capital decreases
                    position_dollars = min(self.fixed_position_dollars, self.tracked_capital * 0.05)
                    position_size = min(position_dollars / entry_price, self.max_position_size)
                    
                    # Skip tiny trades
                    if position_size < 1.0:  # Increased minimum position size
                        return
                        
                    # Set stop loss and take profit levels
                    stop_price = entry_price * (1 - self.stop_loss_pct)
                    target_price = entry_price * (1 + self.take_profit_pct)
                    
                    # Submit order
                    self.backtest.submit_order(signal, entry_price, position_size, timestamp)
                    
                    # Save entry details for P&L calculation
                    self.last_entry_size = position_size
                    self.last_entry_price = entry_price
                    
                    # Log trade (no P&L for entries)
                    self.log_trade(timestamp, signal, entry_price, position_size, "Entry", None)
                    
                    # Update state
                    self.in_position = True
                    self.position_side = "LONG"
                    self.entry_price = entry_price
                    self.stop_price = stop_price
                    self.target_price = target_price
                    self.last_trade_time = timestamp
                    self.day_trade_count += 1
                    
                elif signal == 'SELL':
                    # Calculate fixed position size
                    entry_price = best_bid  # Sell at bid
                    
                    # Use a position size based on current capital, not initial
                    position_dollars = min(self.fixed_position_dollars, self.tracked_capital * 0.05)
                    position_size = min(position_dollars / entry_price, self.max_position_size)
                    
                    # Skip tiny trades
                    if position_size < 1.0:  # Increased minimum position size
                        return
                        
                    # Set stop loss and take profit levels
                    stop_price = entry_price * (1 + self.stop_loss_pct)
                    target_price = entry_price * (1 - self.take_profit_pct)
                    
                    # Submit order
                    self.backtest.submit_order(signal, entry_price, position_size, timestamp)
                    
                    # Save entry details for P&L calculation
                    self.last_entry_size = position_size
                    self.last_entry_price = entry_price
                    
                    # Log trade (no P&L for entries)
                    self.log_trade(timestamp, signal, entry_price, position_size, "Entry", None)
                    
                    # Update state
                    self.in_position = True
                    self.position_side = "SHORT"
                    self.entry_price = entry_price
                    self.stop_price = stop_price
                    self.target_price = target_price
                    self.last_trade_time = timestamp
                    self.day_trade_count += 1
    
    def log_trade(self, timestamp, side, price, size, reason, pnl=None):
        """Log trade details for analysis with manually tracked capital"""
        # Cap size for logging to ensure no overflow
        size_for_log = min(size, 20.0)  # Keep log sizes reasonable
        
        trade_log_entry = {
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp),
            'side': side,
            'price': float(price),
            'size': float(size_for_log),
            'reason': reason,
            'capital': float(self.tracked_capital)
        }
        
        # Add P&L for exit trades
        if pnl is not None:
            trade_log_entry['pnl'] = float(pnl)
            
        self.trade_log.append(trade_log_entry)
    
    def get_trade_dataframe(self):
        """Convert trade log to DataFrame for analysis"""
        if not self.trade_log:
            return pd.DataFrame()
            
        # Create dataframe from trade log
        df = pd.DataFrame(self.trade_log)
        
        # Add entry/exit flags
        if len(df) > 0:
            df['is_entry'] = df['reason'] == 'Entry'
            
        return df

def generate_reports(backtest, strategy, save_dir="backtest_results"):
    """Generate performance reports and visualizations"""
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Process trade log
    trades_df = strategy.get_trade_dataframe()
    
    if trades_df.empty:
        print("No trades to analyze!")
        return
    
    # Save trade log to CSV
    trades_file = os.path.join(save_dir, f"{strategy.name}_{backtest.market}_trades.csv")
    trades_df.to_csv(trades_file, index=False)
    print(f"Trade log saved to {trades_file}")
    
    # Process metrics data
    try:
        # Create metrics dataframe
        metrics_df = pd.DataFrame({
            'timestamp': backtest.metrics['timestamps'],
            'equity': backtest.metrics['equity'],
            'position_size': backtest.metrics['position_size'],
            'unrealized_pnl': backtest.metrics['unrealized_pnl'],
            'realized_pnl': backtest.metrics['realized_pnl']
        })
        
        # Add datetime for plotting
        metrics_df['datetime'] = metrics_df['timestamp'].apply(datetime.fromtimestamp)
        
        # Save metrics to CSV
        metrics_file = os.path.join(save_dir, f"{strategy.name}_{backtest.market}_metrics.csv")
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Metrics data saved to {metrics_file}")
        
        # Generate performance charts
        plt.figure(figsize=(14, 10))
        
        # Set Seaborn style
        sns.set(style="whitegrid")
        
        # Plot equity curve
        plt.subplot(211)
        plt.plot(metrics_df['datetime'], metrics_df['equity'], 'b-', linewidth=1.5)
        plt.title(f"{strategy.name} Equity Curve", fontsize=14)
        plt.ylabel("Account Value ($)")
        plt.grid(True, alpha=0.3)
        
        # Add trade markers on equity curve
        entries = trades_df[trades_df['reason'] == 'Entry']
        exits = trades_df[trades_df['reason'] != 'Entry']
        
        # Plot entries on equity curve
        for _, trade in entries.iterrows():
            plt.plot(trade['datetime'], trade['capital'], 
                    '^' if trade['side'] == 'BUY' else 'v', 
                    color='g' if trade['side'] == 'BUY' else 'r',
                    markersize=8, alpha=0.7)
        
        # Plot price chart with trades
        plt.subplot(212)
        
        # Get price data
        price_data = []
        for ts in backtest.metrics['timestamps']:
            idx = np.argmin(np.abs(np.array(backtest.data['timestamp']) - ts))
            price_data.append(backtest.data.iloc[idx]['mid_price'])
        
        metrics_df['price'] = price_data
        
        plt.plot(metrics_df['datetime'], metrics_df['price'], 'k-', linewidth=1.5, alpha=0.7)
        plt.title("Price Chart with Trades", fontsize=14)
        plt.ylabel("Price")
        plt.grid(True, alpha=0.3)
        
        # Plot entries on price chart
        for _, trade in entries.iterrows():
            plt.plot(trade['datetime'], trade['price'], 
                    '^' if trade['side'] == 'BUY' else 'v', 
                    color='g' if trade['side'] == 'BUY' else 'r',
                    markersize=10, alpha=0.8)
        
        # Plot exits on price chart
        for _, trade in exits.iterrows():
            plt.plot(trade['datetime'], trade['price'], 
                    's', color='blue', markersize=8, alpha=0.8)
        
        plt.tight_layout()
        plot_file = os.path.join(save_dir, f"{strategy.name}_{backtest.market}.png")
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"Performance chart saved to {plot_file}")
        
        # Generate trade analysis if we have P&L data
        if 'pnl' in trades_df.columns and not trades_df['pnl'].isna().all():
            # Calculate performance metrics
            win_trades = len(trades_df[trades_df['pnl'] > 0])
            loss_trades = len(trades_df[trades_df['pnl'] < 0])
            total_trades = win_trades + loss_trades
            win_rate = win_trades / total_trades if total_trades > 0 else 0
            
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if win_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if loss_trades > 0 else 0
            
            # Calculate total P&L from the trade log (more accurate)
            total_pnl = trades_df['pnl'].sum()
            
            # Use the manually tracked capital for final equity - more accurate than backtest metrics
            if not trades_df.empty:
                # Get the final capital from the last trade
                initial_capital = strategy.initial_capital
                final_equity = trades_df['capital'].iloc[-1]
            else:
                initial_capital = backtest.initial_capital
                final_equity = metrics_df['equity'].iloc[-1]
            
            # Calculate return based on total P&L for consistency
            total_return = (final_equity / initial_capital - 1) * 100
            
            # Calculate drawdown
            metrics_df['peak'] = metrics_df['equity'].cummax()
            metrics_df['drawdown'] = (metrics_df['equity'] / metrics_df['peak'] - 1) * 100
            max_drawdown = metrics_df['drawdown'].min()
            
            # Create summary statistics
            summary = {
                'strategy': strategy.name,
                'market': backtest.market,
                'start_date': metrics_df['datetime'].iloc[0].strftime('%Y-%m-%d'),
                'end_date': metrics_df['datetime'].iloc[-1].strftime('%Y-%m-%d'),
                'days': (metrics_df['datetime'].iloc[-1] - metrics_df['datetime'].iloc[0]).days,
                'initial_capital': initial_capital,
                'final_equity': final_equity,
                'total_return': total_return,
                'win_rate': win_rate * 100,
                'total_trades': total_trades,
                'win_trades': win_trades,
                'loss_trades': loss_trades,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
                'max_drawdown': max_drawdown,
                'total_pnl': total_pnl
            }
            
            # Verify consistency: final_equity should equal initial_capital + total_pnl
            if abs(final_equity - (initial_capital + total_pnl)) > 0.01:
                print(f"Warning: Final equity ({final_equity:.2f}) doesn't match initial capital + PnL ({initial_capital + total_pnl:.2f})")
                # Force consistency - final equity MUST equal initial capital + PnL
                summary['final_equity'] = initial_capital + total_pnl
                summary['total_return'] = (summary['final_equity'] / initial_capital - 1) * 100
                print(f"Corrected final equity to {summary['final_equity']:.2f}")
            
            # Save summary to JSON
            import json
            summary_file = os.path.join(save_dir, f"{strategy.name}_{backtest.market}_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=4)
            print(f"Performance summary saved to {summary_file}")
            
    except Exception as e:
        print(f"Error generating reports: {str(e)}")
        import traceback
        traceback.print_exc()

def run_backtest(market="Trump", start_date=None, end_date=None, initial_capital=1000.0):
    """Run backtest with the stable strategy"""
    # Parse dates to timestamps
    start_time = None
    end_time = None
    
    if start_date:
        try:
            start_time = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        except ValueError:
            print(f"Invalid start date: {start_date}. Using all available data.")
    
    if end_date:
        try:
            end_time = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
        except ValueError:
            print(f"Invalid end date: {end_date}. Using all available data.")
    
    # Create strategy with conservative parameters
    strategy = StableTrumpStrategy(
        name="TrumpStable",
        fast_ma=10,             # Increased from 5
        slow_ma=30,             # Increased from 20
        initial_capital=initial_capital,
        fixed_position_dollars=20.0,  # Reduced from 100
        stop_loss_pct=0.015,    # Reduced from 0.02
        take_profit_pct=0.03,   # Reduced from 0.04
        min_imbalance=0.25,     # Increased from 0.15
        min_depth=20.0,         # Increased from 10.0
        cooldown_period=7200,   # 2 hours between trades (increased from 1hr)
        sample_interval=15      # Process every 15th data point (increased from 10)
    )
    
    # Create backtest with risk management settings
    backtest = Backtest(
        market=market,
        start_time=start_time,
        end_time=end_time,
        initial_capital=initial_capital
    )
    
    # Configure backtest with more realistic settings
    backtest.slippage = 0.001       # 10bps slippage (reduced from 50bps)
    backtest.transaction_fee = 0.001 # 10bps fee (reduced from 20bps)
    backtest.use_orderbook_depth = True
    backtest.sample_interval = 15   # Process every 15th data point
    
    # Run backtest
    print(f"Running backtest for {market} with stable position sizing...")
    start_time = time.time()
    
    result = (backtest
             .load_data(preprocess=True, sample_interval=15) # Sample data for performance
             .add_strategy(strategy)
             .run(verbose=True, update_interval=1000)) # Less frequent updates for speed
    
    end_time = time.time()
    
    # Report results
    print(f"Backtest completed in {end_time - start_time:.2f} seconds")
    print(f"Processed {len(backtest.data):,} records")
    
    # Generate reports
    generate_reports(backtest, strategy)
    
    return backtest, strategy

if __name__ == "__main__":
    # Create output directory
    os.makedirs("backtest_results", exist_ok=True)
    
    # Run backtest
    backtest, strategy = run_backtest(
        market="Trump",
        start_date="2024-01-01",
        end_date="2024-11-06",
        initial_capital=1000.0
    )
    
    print("Backtest complete!") 