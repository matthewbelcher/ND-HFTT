import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import json
import time
from abc import ABC, abstractmethod
from tqdm import tqdm
import numba

class Order:
    """Class representing an order in the market"""
    def __init__(self, side, price, size, timestamp):
        self.side = side  # 'BUY' or 'SELL'
        self.price = price
        self.size = size
        self.timestamp = timestamp
        self.status = 'OPEN'  # 'OPEN', 'FILLED', 'CANCELLED'
        self.fill_price = None
        self.fill_timestamp = None
        
    def __str__(self):
        return f"Order({self.side}, ${self.price:.4f}, {self.size:.4f}, {self.status})"
    
    def mark_filled(self, fill_price, fill_timestamp):
        self.status = 'FILLED'
        self.fill_price = fill_price
        self.fill_timestamp = fill_timestamp
    
    def cancel(self):
        self.status = 'CANCELLED'

class Position:
    """Class to track a trading position"""
    def __init__(self):
        self.size = 0
        self.cost_basis = 0
        self.realized_pnl = 0
        
    def update(self, side, size, price):
        """Update position based on a new fill"""
        if side == 'BUY':
            # Calculate new cost basis when buying
            if self.size < 0:  # Covering a short
                self.realized_pnl += (-size) * (self.cost_basis - price)
            
            old_value = self.size * self.cost_basis
            new_value = size * price
            self.size += size
            
            if self.size != 0:
                self.cost_basis = (old_value + new_value) / self.size
            else:
                self.cost_basis = 0
                
        elif side == 'SELL':
            # Calculate realized PnL when selling
            if self.size > 0:  # Selling from a long
                self.realized_pnl += size * (price - self.cost_basis)
            
            old_value = self.size * self.cost_basis
            new_value = size * price
            self.size -= size
            
            if self.size != 0:
                self.cost_basis = (old_value - new_value) / abs(self.size)
            else:
                self.cost_basis = 0
    
    def unrealized_pnl(self, current_price):
        """Calculate unrealized PnL at current market price"""
        if self.size > 0:
            return self.size * (current_price - self.cost_basis)
        elif self.size < 0:
            return self.size * (self.cost_basis - current_price)
        return 0
    
    def total_pnl(self, current_price):
        """Calculate total PnL (realized + unrealized)"""
        return self.realized_pnl + self.unrealized_pnl(current_price)
    
    def __str__(self):
        return f"Position(size={self.size:.4f}, cost_basis=${self.cost_basis:.4f}, realized_pnl=${self.realized_pnl:.4f})"

class Strategy(ABC):
    """Abstract base class for trading strategies"""
    def __init__(self, name):
        self.name = name
        
    @abstractmethod
    def initialize(self, backtest):
        """Initialize the strategy with the backtest environment"""
        pass
    
    @abstractmethod
    def on_data(self, timestamp, orderbook_data):
        """Process new orderbook data and optionally generate orders"""
        pass
    
    @abstractmethod
    def on_fill(self, order, fill_price, fill_timestamp):
        """Handle order fills"""
        pass
    
    def on_vectorized_data(self, data_df):
        """Process data in a vectorized manner (optional)"""
        raise NotImplementedError("Vectorized processing not implemented in this strategy")

class Backtest:
    """Backtesting engine for Polymarket orderbook data"""
    def __init__(self, market, start_time=None, end_time=None, initial_capital=1000.0):
        self.market = market
        self.start_time = start_time
        self.end_time = end_time
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.data = None
        self.position = Position()
        self.orders = []
        self.fills = []
        self.strategy = None
        self.metrics = {
            'timestamps': [],
            'equity': [],
            'position_size': [],
            'unrealized_pnl': [],
            'realized_pnl': []
        }
        self.slippage = 0.0005  # 5bps default slippage
        self.transaction_fee = 0.002  # 20bps default fee
        self.sample_interval = 1  # Process every nth data point
        self.max_records = None  # Maximum number of records to process
        self.verbose = True  # Whether to print detailed logs
        self.use_optimized_mode = True  # Use optimized mode by default
        self.run_in_parallel = False  # Parallel processing flag
        self.use_vectorized_mode = False  # Use vectorized processing mode
        self.use_orderbook_depth = True  # Use orderbook depth for more realistic execution
        
    def load_data(self, data_dir='orderbook_data/metrics', sample_interval=1, max_records=None, preprocess=True):
        """
        Load market data from parquet file
        
        Parameters:
        -----------
        data_dir : str
            Directory containing market data files
        sample_interval : int
            Process every nth data point (1 = all data, 2 = every other point, etc.)
        max_records : int or None
            Maximum number of records to process, None for all
        preprocess : bool
            Whether to preprocess data for faster backtest execution
        """
        print(f"Loading market data for {self.market}...")
        self.sample_interval = max(1, sample_interval)
        self.max_records = max_records
        
        file_path = os.path.join(data_dir, f"{self.market}.parquet")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Market data file not found: {file_path}")
        
        start_load_time = time.time()
        
        # Try optimized reading - only load columns we need
        required_cols = ['timestamp', 'mid_price', 'best_bid', 'best_ask', 'bid_depth', 'ask_depth', 'imbalance', 'spread']
        try:
            df = pd.read_parquet(file_path, columns=required_cols)
        except:
            # Fallback to reading all columns
            df = pd.read_parquet(file_path)
            for col in required_cols:
                if col not in df.columns and col != 'imbalance' and col != 'spread':
                    raise ValueError(f"Required column '{col}' not found in data")

        # Ensure imbalance and spread exist or create them
        if 'imbalance' not in df.columns:
            # Calculate imbalance from bid_depth and ask_depth if available
            if 'bid_depth' in df.columns and 'ask_depth' in df.columns:
                # Prevent division by zero
                epsilon = 1e-10
                df['imbalance'] = df['bid_depth'] / (df['ask_depth'] + epsilon)
            else:
                df['imbalance'] = 1.0  # Default value
                
        if 'spread' not in df.columns:
            if 'best_ask' in df.columns and 'best_bid' in df.columns:
                df['spread'] = df['best_ask'] - df['best_bid']
            else:
                df['spread'] = 0.001  # Default small spread

        # Check for negative and invalid values and clean them
        if 'best_bid' in df.columns:
            # Fix negative bids
            mask = df['best_bid'] < 0
            if mask.any():
                if self.verbose:
                    print(f"Fixing {mask.sum()} negative bid values")
                df.loc[mask, 'best_bid'] = df.loc[mask, 'mid_price'] * 0.999  # Adjust to slightly below mid price
        
        if 'best_ask' in df.columns:
            # Fix zero/negative asks
            mask = df['best_ask'] <= 0
            if mask.any():
                if self.verbose:
                    print(f"Fixing {mask.sum()} zero/negative ask values")
                df.loc[mask, 'best_ask'] = df.loc[mask, 'mid_price'] * 1.001  # Adjust to slightly above mid price

        # Fix crossed markets (where bid > ask)
        if 'best_bid' in df.columns and 'best_ask' in df.columns:
            crossed_mask = df['best_bid'] > df['best_ask']
            if crossed_mask.any():
                if self.verbose:
                    print(f"Fixing {crossed_mask.sum()} crossed markets")
                # Set to mid_price +/- small spread
                mid_price = df.loc[crossed_mask, 'mid_price']
                df.loc[crossed_mask, 'best_bid'] = mid_price * 0.999
                df.loc[crossed_mask, 'best_ask'] = mid_price * 1.001
        
        # Filter by time range if specified
        if self.start_time:
            df = df[df['timestamp'] >= self.start_time]
        if self.end_time:
            df = df[df['timestamp'] <= self.end_time]
            
        # Sort by timestamp to ensure chronological order
        df = df.sort_values('timestamp')
        
        # Apply sampling to reduce data size
        if self.sample_interval > 1:
            df = df.iloc[::self.sample_interval]
        
        # Limit number of records if specified
        if self.max_records and len(df) > self.max_records:
            df = df.head(self.max_records)
        
        # Preprocess data for faster backtesting
        if preprocess:
            # Pre-calculate common values
            df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Store min/max dates for reporting
            self.min_date = df['timestamp_dt'].min()
            self.max_date = df['timestamp_dt'].max()
            
            # Create a dictionary of records for faster iteration
            self.data_dicts = df.to_dict('records')
        
        self.data = df
        
        load_time = time.time() - start_load_time
        
        if self.verbose:
            print(f"Loaded {len(df):,} records for {self.market} in {load_time:.2f} seconds")
            print(f"Data range: {pd.to_datetime(df['timestamp'].min(), unit='s')} to {pd.to_datetime(df['timestamp'].max(), unit='s')}")
            print(f"Sampling: 1:{self.sample_interval}, Records: {len(df):,}")
        
        return self
    
    def add_strategy(self, strategy):
        """Add a trading strategy to the backtest"""
        self.strategy = strategy
        strategy.initialize(self)
        return self
    
    def submit_order(self, side, price, size, timestamp):
        """Submit a new order"""
        order = Order(side, price, size, timestamp)
        self.orders.append(order)
        return order
    
    def execute_order(self, order, current_data):
        """Try to execute an order based on current market data, with improved order depth handling"""
        if order.status != 'OPEN':
            return False
        
        # Check for required orderbook fields
        if 'best_bid' not in current_data or 'best_ask' not in current_data:
            return False
            
        # Improved execution logic using orderbook depth
        if order.side == 'BUY':
            # Get ask price and depth
            ask_price = current_data['best_ask']
            if ask_price <= 0:  # Invalid ask price
                return False
                
            # Check if order price is sufficient considering slippage
            max_buy_price = order.price * (1 + self.slippage)
            
            # For market depth consideration
            if self.use_orderbook_depth and 'ask_depth' in current_data and current_data['ask_depth'] > 0:
                # Adjust execution price based on order size vs ask depth
                depth_ratio = min(order.size / current_data['ask_depth'], 1.0)
                # More impact if order size is large relative to available depth
                depth_impact = depth_ratio * self.slippage * 4  # Amplify impact
                effective_price = ask_price * (1 + depth_impact)
            else:
                # Standard execution without depth consideration
                effective_price = ask_price
            
            # Can we execute at this price?
            if max_buy_price >= effective_price:
                fill_price = effective_price
                fill_timestamp = current_data['timestamp']
                
                # Apply transaction fee
                execution_price = fill_price * (1 + self.transaction_fee)
                
                # Update position and capital
                self.position.update(order.side, order.size, execution_price)
                self.current_capital -= order.size * execution_price
                
                # Mark order as filled
                order.mark_filled(fill_price, fill_timestamp)
                self.fills.append({
                    'timestamp': fill_timestamp,
                    'side': order.side,
                    'size': order.size,
                    'price': fill_price,
                    'effective_price': execution_price,
                    'depth_ratio': depth_ratio if self.use_orderbook_depth and 'ask_depth' in current_data else 0
                })
                
                # Notify strategy
                self.strategy.on_fill(order, fill_price, fill_timestamp)
                return True
            
        elif order.side == 'SELL':
            # Get bid price and depth
            bid_price = current_data['best_bid']
            if bid_price <= 0:  # Invalid bid price
                return False
                
            # Check if order price is sufficient considering slippage
            min_sell_price = order.price * (1 - self.slippage)
            
            # For market depth consideration
            if self.use_orderbook_depth and 'bid_depth' in current_data and current_data['bid_depth'] > 0:
                # Adjust execution price based on order size vs bid depth
                depth_ratio = min(order.size / current_data['bid_depth'], 1.0)
                # More impact if order size is large relative to available depth
                depth_impact = depth_ratio * self.slippage * 4  # Amplify impact
                effective_price = bid_price * (1 - depth_impact)
            else:
                # Standard execution without depth consideration
                effective_price = bid_price
            
            # Can we execute at this price?
            if min_sell_price <= effective_price:
                fill_price = effective_price
                fill_timestamp = current_data['timestamp']
                
                # Apply transaction fee
                execution_price = fill_price * (1 - self.transaction_fee)
                
                # Update position and capital
                self.position.update(order.side, order.size, execution_price)
                self.current_capital += order.size * execution_price
                
                # Mark order as filled
                order.mark_filled(fill_price, fill_timestamp)
                self.fills.append({
                    'timestamp': fill_timestamp,
                    'side': order.side,
                    'size': order.size,
                    'price': fill_price,
                    'effective_price': execution_price,
                    'depth_ratio': depth_ratio if self.use_orderbook_depth and 'bid_depth' in current_data else 0
                })
                
                # Notify strategy
                self.strategy.on_fill(order, fill_price, fill_timestamp)
                return True
                
        return False
    
    def update_metrics(self, timestamp, current_price):
        """Update performance metrics"""
        unrealized_pnl = self.position.unrealized_pnl(current_price)
        total_equity = self.current_capital + unrealized_pnl
        
        self.metrics['timestamps'].append(timestamp)
        self.metrics['equity'].append(total_equity)
        self.metrics['position_size'].append(self.position.size)
        self.metrics['unrealized_pnl'].append(unrealized_pnl)
        self.metrics['realized_pnl'].append(self.position.realized_pnl)
    
    def run(self, verbose=True, use_optimized_mode=True, update_interval=500, use_vectorized_mode=False):
        """
        Run the backtest
        
        Parameters:
        -----------
        verbose : bool
            Whether to print detailed logs and progress bar
        use_optimized_mode : bool
            Use pre-processing and optimized calculations for speed
        update_interval : int
            Update metrics every N data points for faster processing
        use_vectorized_mode : bool
            Use vectorized processing if supported by the strategy
        """
        self.verbose = verbose
        self.use_optimized_mode = use_optimized_mode
        self.use_vectorized_mode = use_vectorized_mode
        
        if not hasattr(self, 'data') or self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if self.strategy is None:
            raise ValueError("No strategy added. Call add_strategy() first.")
        
        start_time = time.time()
        
        if self.verbose:
            print(f"Running backtest for {self.market} with strategy {self.strategy.name}")
            if self.use_optimized_mode:
                print("Using optimized processing mode")
            if self.use_vectorized_mode:
                print("Using vectorized processing (where supported)")
        
        # Initialize metrics
        self.metrics = {
            'timestamps': [],
            'equity': [],
            'position_size': [],
            'unrealized_pnl': [],
            'realized_pnl': []
        }
        
        # Try vectorized processing if supported by the strategy
        if self.use_vectorized_mode:
            try:
                if self.verbose:
                    print("Attempting vectorized processing...")
                # Reset position and metrics for a clean run
                self.position = Position()
                self.orders = []
                self.fills = []
                
                # Call the vectorized strategy method
                self.strategy.on_vectorized_data(self.data)
                
                # Add final metrics
                if len(self.metrics['timestamps']) == 0 and self.data is not None and len(self.data) > 0:
                    last_price = self.data['mid_price'].iloc[-1]
                    last_timestamp = self.data['timestamp'].iloc[-1]
                    self.update_metrics(last_timestamp, last_price)
                
                if self.verbose:
                    print("Vectorized processing completed successfully")
                    
                # Skip to results display
                end_time = time.time()
                if self.verbose:
                    print(f"Backtest completed in {end_time - start_time:.2f} seconds ({len(self.data):,} records)")
                    print(f"Processed {len(self.data) / (end_time - start_time):,.1f} records/second")
                
                self.display_results()
                return self
            except (NotImplementedError, Exception) as e:
                if self.verbose:
                    print(f"Vectorized processing not available or failed: {str(e)}")
                    print("Falling back to standard processing...")
                # Continue with standard processing
        
        # Process data using the optimized approach
        if self.use_optimized_mode and hasattr(self, 'data_dicts'):
            # Pre-process data for faster iteration
            data_iterator = self.data_dicts
            total_records = len(data_iterator)
            
            if self.verbose:
                data_iterator = tqdm(data_iterator, desc="Processing data")
            
            # Process all records - with optimized metrics update
            last_metrics_update = 0
            
            for i, current_data in enumerate(data_iterator):
                timestamp = current_data['timestamp']
                
                # Execute any open orders
                open_orders = [order for order in self.orders if order.status == 'OPEN']
                for order in open_orders:
                    self.execute_order(order, current_data)
                
                # Get strategy decisions
                self.strategy.on_data(timestamp, current_data)
                
                # Update metrics - but only periodically to save time
                # Update if: beginning, end, position changed, or update_interval records passed
                if (i == 0 or i == total_records - 1 or 
                    self.position.size != self.metrics['position_size'][-1] if self.metrics['position_size'] else True or
                    i - last_metrics_update >= update_interval):
                    
                    mid_price = current_data['mid_price']
                    self.update_metrics(timestamp, mid_price)
                    last_metrics_update = i
        else:
            # Traditional processing with dataframe iterrows (slower)
            if self.verbose:
                iterator = tqdm(self.data.iterrows(), total=len(self.data), desc="Processing data")
            else:
                iterator = self.data.iterrows()
            
            # Process each row
            for i, (_, row) in enumerate(iterator):
                current_data = row.to_dict()
                timestamp = current_data['timestamp']
                
                # Execute orders
                open_orders = [order for order in self.orders if order.status == 'OPEN']
                for order in open_orders:
                    self.execute_order(order, current_data)
                
                # Get strategy decisions
                self.strategy.on_data(timestamp, current_data)
                
                # Update metrics less frequently for speed
                if i % update_interval == 0 or i == len(self.data) - 1:
                    mid_price = current_data['mid_price']
                    self.update_metrics(timestamp, mid_price)
        
        # Cancel any remaining open orders
        for order in self.orders:
            if order.status == 'OPEN':
                order.cancel()
        
        # Make sure we have at least one metrics data point
        if not self.metrics['timestamps'] and self.data_dicts:
            last_data = self.data_dicts[-1]
            self.update_metrics(last_data['timestamp'], last_data['mid_price'])
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if self.verbose:
            print(f"Backtest completed in {elapsed_time:.2f} seconds ({len(self.data):,} records)")
            records_per_second = len(self.data) / elapsed_time if elapsed_time > 0 else 0
            print(f"Processed {records_per_second:,.1f} records/second")
        
        # Display results
        self.display_results()
        
        return self
    
    def display_results(self):
        """Display backtest results and performance metrics"""
        if not self.metrics['timestamps']:
            print("No metrics available. Run the backtest first.")
            return
        
        # Convert metrics to DataFrame for analysis
        df_metrics = pd.DataFrame({
            'timestamp': self.metrics['timestamps'],
            'equity': self.metrics['equity'],
            'position_size': self.metrics['position_size'],
            'unrealized_pnl': self.metrics['unrealized_pnl'],
            'realized_pnl': self.metrics['realized_pnl']
        })
        
        df_metrics['datetime'] = pd.to_datetime(df_metrics['timestamp'], unit='s')
        df_metrics.set_index('datetime', inplace=True)
        
        # Use strategy's tracked capital if available (for more accurate reporting)
        initial_equity = self.initial_capital
        
        # Check if the strategy has tracked_capital and trade_log
        if hasattr(self.strategy, 'tracked_capital') and hasattr(self.strategy, 'trade_log') and self.strategy.trade_log:
            # Use the more accurate manually tracked capital from the strategy
            final_equity = self.strategy.tracked_capital
            print("Using strategy's manually tracked capital for accuracy")
        else:
            # Fallback to backtest's equity tracking
            final_equity = df_metrics['equity'].iloc[-1]
        
        total_return = (final_equity / initial_equity - 1) * 100
        
        # Calculate daily returns
        df_metrics['daily_return'] = df_metrics['equity'].pct_change()
        
        # Annualized return (assuming 365 days)
        days = (df_metrics.index[-1] - df_metrics.index[0]).days
        if days > 0:
            annualized_return = ((1 + total_return/100) ** (365/days) - 1) * 100
        else:
            annualized_return = total_return  # If less than a day, don't annualize
        
        # Standard deviation of daily returns (annualized)
        daily_std = df_metrics['daily_return'].std()
        annualized_volatility = daily_std * (365 ** 0.5) * 100 if days > 0 else 0
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # Maximum drawdown
        df_metrics['cummax'] = df_metrics['equity'].cummax()
        df_metrics['drawdown'] = (df_metrics['equity'] / df_metrics['cummax'] - 1) * 100
        max_drawdown = df_metrics['drawdown'].min()
        
        # Count trades
        num_trades = len(self.fills)
        
        # Calculate fill quality metrics
        if self.fills:
            fills_df = pd.DataFrame(self.fills)
            avg_depth_ratio = fills_df.get('depth_ratio', pd.Series([0])).mean()
            slippage_pct = (fills_df['effective_price'] - fills_df['price']).abs().mean() / fills_df['price'].mean() * 100
        else:
            avg_depth_ratio = 0
            slippage_pct = 0
        
        # Print results
        print("\n===== BACKTEST RESULTS =====")
        print(f"Strategy: {self.strategy.name}")
        print(f"Market: {self.market}")
        print(f"Period: {df_metrics.index[0]} to {df_metrics.index[-1]} ({days} days)")
        print(f"Initial Capital: ${initial_equity:.2f}")
        print(f"Final Equity: ${final_equity:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Annualized Return: {annualized_return:.2f}%")
        print(f"Annualized Volatility: {annualized_volatility:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        print(f"Number of Trades: {num_trades}")
        print(f"Trades per Day: {num_trades/max(1, days):.2f}")
        print(f"Average Depth Impact Ratio: {avg_depth_ratio:.4f}")
        print(f"Average Slippage+Fee: {slippage_pct:.2f}%")
        
        # Plot equity curve 
        try:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(3, 1, 1)
            plt.plot(df_metrics.index, df_metrics['equity'], label='Equity Curve')
            plt.title(f"{self.strategy.name} on {self.market}")
            plt.ylabel('Equity ($)')
            plt.grid(True)
            plt.legend()
            
            plt.subplot(3, 1, 2)
            plt.plot(df_metrics.index, df_metrics['position_size'], label='Position Size')
            plt.ylabel('Position Size')
            plt.grid(True)
            plt.legend()
            
            plt.subplot(3, 1, 3)
            plt.plot(df_metrics.index, df_metrics['drawdown'], label='Drawdown', color='red')
            plt.ylabel('Drawdown (%)')
            plt.xlabel('Date')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            
            # Create results directory if it doesn't exist
            os.makedirs('backtest_results', exist_ok=True)
            
            # Save plot
            plt.savefig(f"backtest_results/{self.strategy.name}_{self.market}.png")
        except Exception as e:
            print(f"Error generating plots: {str(e)}")
        
        try:
            # Save metrics to CSV
            df_metrics.to_csv(f"backtest_results/{self.strategy.name}_{self.market}_metrics.csv")
            
            # Save trade log
            trades_df = pd.DataFrame(self.fills)
            if len(trades_df) > 0:
                trades_df['datetime'] = pd.to_datetime(trades_df['timestamp'], unit='s')
                trades_df.to_csv(f"backtest_results/{self.strategy.name}_{self.market}_trades.csv", index=False)
            
            # Save summary as JSON
            summary = {
                'strategy': self.strategy.name,
                'market': self.market,
                'start_date': df_metrics.index[0].strftime('%Y-%m-%d'),
                'end_date': df_metrics.index[-1].strftime('%Y-%m-%d'),
                'days': days,
                'initial_capital': initial_equity,
                'final_equity': final_equity,
                'total_return_pct': total_return,
                'annualized_return_pct': annualized_return,
                'annualized_volatility_pct': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown,
                'num_trades': num_trades,
                'avg_depth_impact': avg_depth_ratio,
                'avg_slippage_pct': slippage_pct
            }
            
            with open(f"backtest_results/{self.strategy.name}_{self.market}_summary.json", 'w') as f:
                json.dump(summary, f, indent=4)
                
            print(f"Results saved to backtest_results/ directory")
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            
        return df_metrics 