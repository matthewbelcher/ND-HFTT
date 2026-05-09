import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from tqdm import tqdm

# Load Trump data
print("Loading data...")
df_trump = pd.read_parquet('orderbook_data/metrics/Trump.parquet')

# Sort by timestamp
df_sorted = df_trump.sort_values('timestamp').reset_index(drop=True)

# Remove duplicate timestamps (keep first occurrence)
df_unique = df_sorted.drop_duplicates(subset=['timestamp']).reset_index(drop=True)

print(f"Unique timestamps: {len(df_unique)}")
print(f"Time range: {datetime.fromtimestamp(df_unique['timestamp'].min())} to {datetime.fromtimestamp(df_unique['timestamp'].max())}")

# Create output directory
output_dir = "backtest_results"
os.makedirs(output_dir, exist_ok=True)

# Simple trading strategy parameters
ENTRY_THRESHOLD = 0.15  # Imbalance threshold to enter position
EXIT_THRESHOLD = 0.05   # Profit target
STOP_LOSS = 0.05        # Stop loss 
TRADE_SIZE = 1.0        # Number of contracts per trade
MAX_POSITIONS = 5       # Maximum open positions
SLIPPAGE = 0.005        # Assumed slippage per trade
TRANSACTION_FEE = 0.01  # Fixed fee per trade

class BacktestSimulator:
    def __init__(self, data):
        self.data = data
        self.positions = []  # List of open positions
        self.trades = []     # List of completed trades
        self.balance = 100.0  # Starting balance
        self.equity_curve = []
        
    def run_backtest(self):
        """Simulate a backtest on the orderbook data"""
        print("Running backtest simulation...")
        
        # Initialize equity curve with starting balance
        self.equity_curve.append((self.data.iloc[0]['timestamp'], self.balance))
        
        # Iterate through each timestamp
        for i, row in tqdm(self.data.iterrows(), total=len(self.data)):
            # Check if we should close any positions
            self._check_exits(row)
            
            # Check if we should open a new position
            self._check_entries(row)
            
            # Record current equity
            self.equity_curve.append((row['timestamp'], self.balance + self._get_unrealized_pnl(row)))
        
        # Close any remaining positions at the last price
        last_row = self.data.iloc[-1]
        for position in self.positions.copy():
            self._close_position(position, last_row, "Forced Exit")
        
        # Convert equity curve to DataFrame
        self.equity_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        
        # Calculate trade statistics
        self._calculate_statistics()
        
    def _check_entries(self, row):
        """Check if we should enter a new position based on current data"""
        # Skip if we're at maximum positions
        if len(self.positions) >= MAX_POSITIONS:
            return
        
        # Simple strategy: Enter when imbalance is extreme
        # Long if bid/ask imbalance favors bids (> ENTRY_THRESHOLD)
        # Short if bid/ask imbalance favors asks (< (1-ENTRY_THRESHOLD))
        if row['imbalance'] > 1 + ENTRY_THRESHOLD:
            # Strong buying pressure, enter long
            entry_price = row['best_ask'] + SLIPPAGE  # Buy at ask plus slippage
            
            # Ensure entry price doesn't exceed 1.0 (100% probability)
            entry_price = min(entry_price, 1.0)
            
            position = {
                'type': 'long',
                'entry_price': entry_price,
                'entry_time': row['timestamp'],
                'size': TRADE_SIZE,
                'target_price': min(entry_price + EXIT_THRESHOLD, 1.0),
                'stop_price': max(entry_price - STOP_LOSS, 0.0),
            }
            
            # Deduct transaction fee
            self.balance -= TRANSACTION_FEE
            
            self.positions.append(position)
            
        elif row['imbalance'] < (1 - ENTRY_THRESHOLD):
            # Strong selling pressure, enter short
            entry_price = row['best_bid'] - SLIPPAGE  # Sell at bid minus slippage
            
            # Ensure entry price doesn't go below 0.0 (0% probability)
            entry_price = max(entry_price, 0.0)
            
            position = {
                'type': 'short',
                'entry_price': entry_price,
                'entry_time': row['timestamp'],
                'size': TRADE_SIZE,
                'target_price': max(entry_price - EXIT_THRESHOLD, 0.0),
                'stop_price': min(entry_price + STOP_LOSS, 1.0),
            }
            
            # Deduct transaction fee
            self.balance -= TRANSACTION_FEE
            
            self.positions.append(position)
    
    def _check_exits(self, row):
        """Check if any existing positions should be closed"""
        for position in self.positions.copy():
            exit_reason = None
            
            current_bid = row['best_bid']
            current_ask = row['best_ask']
            
            if position['type'] == 'long':
                # For long positions, we sell at bid price
                if current_bid >= position['target_price']:
                    exit_reason = "Target Reached"
                elif current_bid <= position['stop_price']:
                    exit_reason = "Stop Loss"
            else:  # short position
                # For short positions, we buy back at ask price
                if current_ask <= position['target_price']:
                    exit_reason = "Target Reached"
                elif current_ask >= position['stop_price']:
                    exit_reason = "Stop Loss"
            
            # Exit the position if we have a reason
            if exit_reason:
                self._close_position(position, row, exit_reason)
    
    def _close_position(self, position, row, reason):
        """Close a position and record the trade"""
        # Remove from active positions
        self.positions.remove(position)
        
        # Calculate exit price with slippage
        if position['type'] == 'long':
            # For long positions, we sell at bid price minus slippage
            exit_price = row['best_bid'] - SLIPPAGE
            exit_price = max(exit_price, 0.0)  # Ensure price doesn't go below 0
            pnl = (exit_price - position['entry_price']) * position['size']
        else:
            # For short positions, we buy back at ask price plus slippage
            exit_price = row['best_ask'] + SLIPPAGE
            exit_price = min(exit_price, 1.0)  # Ensure price doesn't exceed 1
            pnl = (position['entry_price'] - exit_price) * position['size']
        
        # Deduct transaction fee
        pnl -= TRANSACTION_FEE
        
        # Update balance
        self.balance += pnl
        
        # Record the trade
        trade = {
            'type': position['type'],
            'entry_time': position['entry_time'],
            'entry_price': position['entry_price'],
            'exit_time': row['timestamp'],
            'exit_price': exit_price,
            'size': position['size'],
            'pnl': pnl,
            'exit_reason': reason
        }
        
        self.trades.append(trade)
    
    def _get_unrealized_pnl(self, row):
        """Calculate unrealized PnL for all open positions"""
        unrealized_pnl = 0
        
        for position in self.positions:
            if position['type'] == 'long':
                # For long positions, use bid price to calculate current value
                current_price = row['best_bid']
                unrealized_pnl += (current_price - position['entry_price']) * position['size']
            else:
                # For short positions, use ask price to calculate current value
                current_price = row['best_ask']
                unrealized_pnl += (position['entry_price'] - current_price) * position['size']
        
        return unrealized_pnl
    
    def _calculate_statistics(self):
        """Calculate and print backtest statistics"""
        if not self.trades:
            print("No trades executed!")
            return
        
        # Convert trades to DataFrame
        self.trades_df = pd.DataFrame(self.trades)
        
        # Ensure datetime format for plotting
        self.trades_df['entry_time'] = self.trades_df['entry_time'].apply(datetime.fromtimestamp)
        self.trades_df['exit_time'] = self.trades_df['exit_time'].apply(datetime.fromtimestamp)
        
        # Calculate basic statistics
        total_trades = len(self.trades_df)
        profitable_trades = len(self.trades_df[self.trades_df['pnl'] > 0])
        losing_trades = len(self.trades_df[self.trades_df['pnl'] <= 0])
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Calculate returns
        total_pnl = self.trades_df['pnl'].sum()
        final_balance = self.balance
        roi = (final_balance - 100.0) / 100.0 * 100  # ROI in percentage
        
        # Calculate average trade metrics
        avg_profit = self.trades_df[self.trades_df['pnl'] > 0]['pnl'].mean() if profitable_trades > 0 else 0
        avg_loss = self.trades_df[self.trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Calculate profit factor
        gross_profit = self.trades_df[self.trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(self.trades_df[self.trades_df['pnl'] <= 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Print statistics
        print("\nBacktest Results:")
        print(f"Total Trades: {total_trades}")
        print(f"Profitable Trades: {profitable_trades} ({win_rate:.2%})")
        print(f"Losing Trades: {losing_trades} ({1-win_rate:.2%})")
        print(f"Total P&L: ${total_pnl:.2f}")
        print(f"Final Balance: ${final_balance:.2f}")
        print(f"ROI: {roi:.2f}%")
        print(f"Average Profit: ${avg_profit:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        
        # Save the trades to CSV
        trades_file = os.path.join(output_dir, "backtest_trades.csv")
        self.trades_df.to_csv(trades_file, index=False)
        print(f"Detailed trades saved to {trades_file}")
    
    def plot_results(self):
        """Plot backtest results"""
        if not hasattr(self, 'equity_df') or not hasattr(self, 'trades_df') or len(self.trades_df) == 0:
            print("No backtest results to plot!")
            return
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Convert timestamp to datetime for plotting
        equity_dates = [datetime.fromtimestamp(ts) for ts in self.equity_df['timestamp']]
        
        # Plot equity curve
        axes[0].plot(equity_dates, self.equity_df['equity'], 'b-', linewidth=1)
        axes[0].set_title('Equity Curve')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Account Balance ($)')
        axes[0].grid(True)
        
        # Plot the mid price
        price_df = self.data[['timestamp', 'mid_price']].copy()
        price_df['datetime'] = price_df['timestamp'].apply(datetime.fromtimestamp)
        axes[1].plot(price_df['datetime'], price_df['mid_price'], 'k-', linewidth=1)
        axes[1].set_title('Trump Contract Price')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Price')
        axes[1].grid(True)
        
        # Plot trade entry and exit points
        for _, trade in self.trades_df.iterrows():
            marker = '^' if trade['type'] == 'long' else 'v'
            color = 'g' if trade['pnl'] > 0 else 'r'
            
            # Plot entry point
            axes[1].plot(trade['entry_time'], trade['entry_price'], marker, color='blue', markersize=8)
            
            # Plot exit point
            axes[1].plot(trade['exit_time'], trade['exit_price'], 's', color=color, markersize=8)
            
            # Draw line connecting entry and exit
            axes[1].plot([trade['entry_time'], trade['exit_time']], 
                      [trade['entry_price'], trade['exit_price']], 
                      '-', color=color, alpha=0.5)
        
        # Plot trade PnL
        trade_dates = self.trades_df['exit_time']
        pnl_values = self.trades_df['pnl']
        axes[2].bar(trade_dates, pnl_values, color=['g' if p > 0 else 'r' for p in pnl_values])
        axes[2].set_title('Trade P&L')
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('Profit/Loss ($)')
        axes[2].grid(True)
        
        # Add legend
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], marker='^', color='blue', markersize=8, linestyle='None', label='Long Entry'),
            Line2D([0], [0], marker='v', color='blue', markersize=8, linestyle='None', label='Short Entry'),
            Line2D([0], [0], marker='s', color='g', markersize=8, linestyle='None', label='Profitable Exit'),
            Line2D([0], [0], marker='s', color='r', markersize=8, linestyle='None', label='Losing Exit')
        ]
        axes[1].legend(handles=custom_lines, loc='upper right')
        
        # Adjust layout and save
        plt.tight_layout()
        output_file = os.path.join(output_dir, "backtest_results.png")
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        print(f"Backtest visualization saved to {output_file}")

# Sample the data to speed up the backtest (use every n-th row)
# For a full backtest, set sampling_rate to 1
sampling_rate = 100
df_sample = df_unique.iloc[::sampling_rate].reset_index(drop=True)

print(f"Running backtest on {len(df_sample)} data points (sampled from {len(df_unique)})")

# Create and run the backtest
backtest = BacktestSimulator(df_sample)
backtest.run_backtest()
backtest.plot_results()

print("Backtest simulation complete!") 