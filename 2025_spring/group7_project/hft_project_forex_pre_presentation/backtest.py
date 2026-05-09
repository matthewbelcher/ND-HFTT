"""
Backtesting module for statistical arbitrage strategies based on cointegration analysis.
This module simulates trading on historical data and evaluates strategy performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

# Import our modules
from fetch_oanda_data import OandaDataFetcher
from cointegration_analysis import find_cointegrated_pairs
from statistical_arbitrage import StatArbitrageStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('Backtest')

class StatArbBacktester:
    """Class for backtesting statistical arbitrage strategies."""
    
    def __init__(
        self,
        pair_data: Dict[str, pd.DataFrame],
        z_entry: float = 2.0,
        z_exit: float = 0.0,
        p_value_threshold: float = 0.05,
        max_z_score: float = 4.0,
        stop_loss_z: float = 4.0,
        initial_capital: float = 10000.0,
        position_size_pct: float = 0.02,
        commission_pct: float = 0.001,
        slippage_pips: float = 1.0,
        output_dir: str = "backtest_results"
    ):
        """
        Initialize the backtester.
        
        Args:
            pair_data: Dictionary of currency pairs with their price DataFrames
            z_entry: Z-score threshold for trade entry
            z_exit: Z-score threshold for trade exit
            p_value_threshold: P-value threshold for cointegration
            max_z_score: Maximum allowed z-score
            stop_loss_z: Z-score based stop loss
            initial_capital: Initial capital for backtesting
            position_size_pct: Position size as percentage of capital
            commission_pct: Commission percentage per trade
            slippage_pips: Slippage in pips per trade
            output_dir: Directory for output files
        """
        self.pair_data = pair_data
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.p_value_threshold = p_value_threshold
        self.max_z_score = max_z_score
        self.stop_loss_z = stop_loss_z
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.commission_pct = commission_pct
        self.slippage_pips = slippage_pips
        
        # Create strategy instance
        self.strategy = StatArbitrageStrategy(
            z_score_entry=z_entry,
            z_score_exit=z_exit,
            max_z_score=max_z_score,
            stop_loss_z=stop_loss_z
        )
        
        # Set up output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Track performance
        self.equity_curve = []
        self.trades = []
        self.open_positions = []
        self.daily_returns = []
        self.trade_signals = []
        
        logger.info(f"Initialized StatArbBacktester with z_entry={z_entry}, z_exit={z_exit}")
        
    def find_cointegrated_pairs(self) -> List[Dict]:
        """
        Find cointegrated pairs from the provided price data.
        
        Returns:
            List of cointegration analysis results
        """
        logger.info("Finding cointegrated pairs for backtesting")
        return find_cointegrated_pairs(self.pair_data, self.p_value_threshold)
        
    def backtest(self, cointegration_results: List[Dict]) -> Dict:
        """
        Run the backtest on historical data.
        
        Args:
            cointegration_results: List of cointegration analysis results
            
        Returns:
            Dictionary with backtest results
        """
        logger.info("Starting backtest")
        
        # Determine the common date range across all pairs
        start_date, end_date = self._get_common_date_range()
        logger.info(f"Backtest period: {start_date} to {end_date}")
        
        # Initialize portfolio values
        capital = self.initial_capital
        portfolio_history = []
        
        # Create trading days sequence
        trading_days = self._generate_trading_days(start_date, end_date)
        
        # For each trading day
        for current_date in trading_days:
            logger.info(f"Processing date: {current_date}")
            
            # Update pair data up to current date
            current_pair_data = self._filter_data_until_date(current_date)
            
            # Skip if insufficient data
            if not current_pair_data:
                continue
            
            # Calculate updated cointegration relationships
            if len(trading_days) % 20 == 0:  # Recalculate every 20 days
                current_cointegration = find_cointegrated_pairs(current_pair_data, self.p_value_threshold)
            else:
                current_cointegration = self._update_cointegration(cointegration_results, current_pair_data)
            
            # Generate signals
            signals = self.strategy.generate_signals(current_cointegration)
            
            # Process signals (Open and close positions)
            capital = self._process_signals(signals, current_date, capital, current_pair_data)
            
            # Update open positions
            capital = self._update_positions(current_date, capital, current_pair_data)
            
            # Record portfolio value
            portfolio_history.append({
                'date': current_date,
                'equity': capital,
                'open_positions': len(self.open_positions)
            })
        
        # Close any remaining positions at the end
        for position in self.open_positions[:]:  # Copy the list for iteration
            exit_signal = {
                'pair1': position['pair1'],
                'pair2': position['pair2'],
                'action1': 'SELL' if position['action1'] == 'BUY' else 'BUY',
                'action2': 'BUY' if position['action2'] == 'SELL' else 'SELL',
                'signal_type': 'EXIT',
                'z_score': 0
            }
            self._close_position(position, exit_signal, end_date, capital, self.pair_data)
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics(portfolio_history)
        
        # Save and plot results
        self._save_backtest_results(results, portfolio_history)
        
        logger.info(f"Backtest completed: Final equity: ${results['final_equity']:.2f}")
        return results
    
    def _get_common_date_range(self) -> Tuple[datetime, datetime]:
        """Determine the common date range across all pairs."""
        start_dates = []
        end_dates = []
        
        for pair, df in self.pair_data.items():
            if 'timestamp' in df.columns:
                start_dates.append(df['timestamp'].min())
                end_dates.append(df['timestamp'].max())
        
        return max(start_dates), min(end_dates)
    
    def _generate_trading_days(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """Generate a sequence of trading days for the backtest."""
        # For simplicity, we'll use calendar days but skip weekends
        # In a real forex system, you'd need more sophisticated handling of trading hours
        
        days = []
        current = start_date
        
        while current <= end_date:
            # Skip weekends (0 = Monday, 6 = Sunday)
            if current.weekday() < 5:  # 0-4 are Monday to Friday
                days.append(current)
            current += timedelta(days=1)
        
        return days
    
    def _filter_data_until_date(self, current_date: datetime) -> Dict[str, pd.DataFrame]:
        """Filter data up to the current date for each pair."""
        filtered_data = {}
        
        for pair, df in self.pair_data.items():
            if 'timestamp' in df.columns:
                filtered_df = df[df['timestamp'] <= current_date].copy()
                if len(filtered_df) > 30:  # Ensure enough data for analysis
                    filtered_data[pair] = filtered_df
        
        return filtered_data
    
    def _update_cointegration(
        self, 
        previous_results: List[Dict], 
        current_pair_data: Dict[str, pd.DataFrame]
    ) -> List[Dict]:
        """Update cointegration results with latest data."""
        # In real-time, you'd want to update the z-scores and spreads
        # without recalculating the full cointegration tests
        
        updated_results = []
        
        for result in previous_results:
            pair1 = result['pair1']
            pair2 = result['pair2']
            
            if pair1 in current_pair_data and pair2 in current_pair_data:
                # Get latest data
                df1 = current_pair_data[pair1]
                df2 = current_pair_data[pair2]
                
                # Update spread and z-score with existing hedge ratio
                if 'timestamp' in df1.columns:
                    df1 = df1.set_index('timestamp')
                if 'timestamp' in df2.columns:
                    df2 = df2.set_index('timestamp')
                
                # Use common dates
                common_idx = df1.index.intersection(df2.index)
                price1 = df1.loc[common_idx, 'close']
                price2 = df2.loc[common_idx, 'close']
                
                # Calculate spread with existing hedge ratio
                hedge_ratio = result['hedge_ratio']
                spread = price1 - hedge_ratio * price2
                
                # Calculate z-score
                mean = spread.mean()
                std = spread.std()
                z_score = (spread - mean) / std
                
                # Update result
                updated_result = result.copy()
                updated_result['spread_mean'] = float(mean)
                updated_result['spread_std'] = float(std)
                updated_result['current_spread'] = float(spread.iloc[-1])
                updated_result['current_z_score'] = float(z_score.iloc[-1])
                updated_result['spread_series'] = spread.values.tolist()
                updated_result['z_score_series'] = z_score.values.tolist()
                updated_result['timestamps'] = common_idx.tolist()
                
                updated_results.append(updated_result)
        
        return updated_results
    
    def _process_signals(
        self, 
        signals: List[Dict], 
        current_date: datetime, 
        capital: float, 
        pair_data: Dict[str, pd.DataFrame]
    ) -> float:
        """Process trading signals and update positions."""
        # Track all signals for analysis
        for signal in signals:
            signal['signal_date'] = current_date
            self.trade_signals.append(signal)
        
        # Process exit signals first
        exit_signals = [s for s in signals if s['signal_type'] == 'EXIT']
        for signal in exit_signals:
            # Find matching position
            for position in self.open_positions[:]:  # Copy for safe iteration
                if position['pair1'] == signal['pair1'] and position['pair2'] == signal['pair2']:
                    # Close position
                    capital = self._close_position(position, signal, current_date, capital, pair_data)
        
        # Process entry signals
        entry_signals = [s for s in signals if s['signal_type'] == 'ENTRY']
        for signal in entry_signals:
            # Check if we already have this pair
            if any(p['pair1'] == signal['pair1'] and p['pair2'] == signal['pair2'] for p in self.open_positions):
                continue
                
            # Open new position
            capital = self._open_position(signal, current_date, capital, pair_data)
        
        return capital
    
    def _open_position(
        self, 
        signal: Dict, 
        date: datetime, 
        capital: float, 
        pair_data: Dict[str, pd.DataFrame]
    ) -> float:
        """Open a new position based on a signal."""
        pair1 = signal['pair1']
        pair2 = signal['pair2']
        
        # Get current prices
        price1 = self._get_latest_price(pair_data, pair1)
        price2 = self._get_latest_price(pair_data, pair2)
        
        if price1 is None or price2 is None:
            return capital
        
        # Calculate position size (simplified)
        risk_amount = capital * self.position_size_pct
        pair1_units = risk_amount / (2 * price1)  # Split risk between the two pairs
        pair2_units = risk_amount / (2 * price2)
        
        # Adjust units for hedge ratio
        hedge_ratio = signal.get('ratio', 1.0)
        pair2_units = pair2_units * hedge_ratio
        
        # Account for direction
        if signal['action1'] == 'SELL':
            pair1_units = -pair1_units
        if signal['action2'] == 'SELL':
            pair2_units = -pair2_units
        
        # Calculate costs (commission + slippage)
        commission1 = abs(pair1_units * price1 * self.commission_pct)
        commission2 = abs(pair2_units * price2 * self.commission_pct)
        
        # Slippage (simplified)
        pip_value1 = 0.0001 if 'JPY' not in pair1 else 0.01
        pip_value2 = 0.0001 if 'JPY' not in pair2 else 0.01
        slippage1 = abs(pair1_units) * self.slippage_pips * pip_value1
        slippage2 = abs(pair2_units) * self.slippage_pips * pip_value2
        
        # Total costs
        total_cost = commission1 + commission2 + slippage1 + slippage2
        
        # Create position object
        position = {
            'pair1': pair1,
            'pair2': pair2,
            'action1': signal['action1'],
            'action2': signal['action2'],
            'entry_date': date,
            'entry_price1': price1,
            'entry_price2': price2,
            'units1': pair1_units,
            'units2': pair2_units,
            'entry_cost': total_cost,
            'hedge_ratio': hedge_ratio,
            'entry_z_score': signal['z_score'],
            'stop_loss_z': signal.get('stop_loss_z', None)
        }
        
        # Add to open positions
        self.open_positions.append(position)
        
        # Deduct costs from capital
        capital -= total_cost
        
        # Log the trade
        logger.info(f"Opened position: {signal['action1']} {pair1} / {signal['action2']} {pair2} " +
                   f"at z-score: {signal['z_score']:.2f}, cost: ${total_cost:.2f}")
        
        return capital
    
    def _close_position(
        self, 
        position: Dict, 
        signal: Dict, 
        date: datetime, 
        capital: float, 
        pair_data: Dict[str, pd.DataFrame]
    ) -> float:
        """Close an existing position and calculate PnL."""
        pair1 = position['pair1']
        pair2 = position['pair2']
        
        # Get current prices
        exit_price1 = self._get_latest_price(pair_data, pair1)
        exit_price2 = self._get_latest_price(pair_data, pair2)
        
        if exit_price1 is None or exit_price2 is None:
            return capital
        
        # Calculate P&L
        pair1_pnl = position['units1'] * (exit_price1 - position['entry_price1'])
        pair2_pnl = position['units2'] * (exit_price2 - position['entry_price2'])
        
        # Total P&L
        total_pnl = pair1_pnl + pair2_pnl
        
        # Calculate costs
        exit_cost1 = abs(position['units1'] * exit_price1 * self.commission_pct)
        exit_cost2 = abs(position['units2'] * exit_price2 * self.commission_pct)
        
        # Slippage (simplified)
        pip_value1 = 0.0001 if 'JPY' not in pair1 else 0.01
        pip_value2 = 0.0001 if 'JPY' not in pair2 else 0.01
        slippage1 = abs(position['units1']) * self.slippage_pips * pip_value1
        slippage2 = abs(position['units2']) * self.slippage_pips * pip_value2
        
        # Total exit costs
        total_exit_cost = exit_cost1 + exit_cost2 + slippage1 + slippage2
        
        # Net P&L after costs
        net_pnl = total_pnl - total_exit_cost - position['entry_cost']
        
        # Update capital
        capital += net_pnl + position['entry_cost']  # Add back entry cost which was deducted earlier
        
        # Record the trade
        trade_record = {
            'pair1': pair1,
            'pair2': pair2,
            'entry_date': position['entry_date'],
            'exit_date': date,
            'duration_days': (date - position['entry_date']).days,
            'action1': position['action1'],
            'action2': position['action2'],
            'entry_price1': position['entry_price1'],
            'entry_price2': position['entry_price2'],
            'exit_price1': exit_price1,
            'exit_price2': exit_price2,
            'units1': position['units1'],
            'units2': position['units2'],
            'costs': position['entry_cost'] + total_exit_cost,
            'pnl': net_pnl,
            'entry_z_score': position['entry_z_score'],
            'exit_z_score': signal['z_score']
        }
        
        self.trades.append(trade_record)
        
        # Remove from open positions
        self.open_positions.remove(position)
        
        # Log the trade
        logger.info(f"Closed position: {pair1}/{pair2}, held for {trade_record['duration_days']} days, " +
                   f"P&L: ${net_pnl:.2f}")
        
        return capital
    
    def _update_positions(
        self, 
        current_date: datetime, 
        capital: float, 
        pair_data: Dict[str, pd.DataFrame]
    ) -> float:
        """Update open positions and check for stop losses."""
        for position in self.open_positions[:]:  # Copy for safe iteration
            pair1 = position['pair1']
            pair2 = position['pair2']
            
            # Calculate current z-score
            current_z_score = self._calculate_current_zscore(position, pair_data)
            
            if current_z_score is None:
                continue
            
            # Check for stop loss
            if position['stop_loss_z'] is not None:
                if position['action1'] == 'BUY' and current_z_score > position['stop_loss_z']:
                    # Stop loss hit for long position
                    exit_signal = {
                        'pair1': pair1,
                        'pair2': pair2,
                        'action1': 'SELL',
                        'action2': 'BUY',
                        'signal_type': 'STOP_LOSS',
                        'z_score': current_z_score
                    }
                    capital = self._close_position(position, exit_signal, current_date, capital, pair_data)
                    continue
                    
                elif position['action1'] == 'SELL' and current_z_score < -position['stop_loss_z']:
                    # Stop loss hit for short position
                    exit_signal = {
                        'pair1': pair1,
                        'pair2': pair2,
                        'action1': 'BUY',
                        'action2': 'SELL',
                        'signal_type': 'STOP_LOSS',
                        'z_score': current_z_score
                    }
                    capital = self._close_position(position, exit_signal, current_date, capital, pair_data)
                    continue
        
        # Calculate and track daily equity
        total_position_value = 0
        for position in self.open_positions:
            pair1_price = self._get_latest_price(pair_data, position['pair1'])
            pair2_price = self._get_latest_price(pair_data, position['pair2'])
            
            if pair1_price is not None and pair2_price is not None:
                pair1_value = position['units1'] * pair1_price
                pair2_value = position['units2'] * pair2_price
                total_position_value += pair1_value + pair2_value
        
        # Record daily equity (cash + positions)
        self.equity_curve.append({
            'date': current_date,
            'equity': capital + total_position_value
        })
        
        return capital
    
    def _calculate_current_zscore(self, position: Dict, pair_data: Dict[str, pd.DataFrame]) -> Optional[float]:
        """Calculate the current z-score for a position."""
        pair1 = position['pair1']
        pair2 = position['pair2']
        
        df1 = pair_data.get(pair1)
        df2 = pair_data.get(pair2)
        
        if df1 is None or df2 is None:
            return None
        
        # Convert to Series if not already
        if 'timestamp' in df1.columns:
            df1 = df1.set_index('timestamp')
        if 'timestamp' in df2.columns:
            df2 = df2.set_index('timestamp')
        
        # Use common dates
        common_idx = df1.index.intersection(df2.index)
        if len(common_idx) < 30:
            return None
            
        price1 = df1.loc[common_idx, 'close']
        price2 = df2.loc[common_idx, 'close']
        
        # Calculate spread with position's hedge ratio
        hedge_ratio = position['hedge_ratio']
        spread = price1 - hedge_ratio * price2
        
        # Calculate z-score
        mean = spread.mean()
        std = spread.std()
        
        if std == 0:
            return None
            
        current_spread = spread.iloc[-1]
        z_score = (current_spread - mean) / std
        
        return z_score
    
    def _get_latest_price(self, pair_data: Dict[str, pd.DataFrame], pair: str) -> Optional[float]:
        """Get the latest price for a currency pair."""
        df = pair_data.get(pair)
        
        if df is None or df.empty:
            return None
        
        if 'close' in df.columns:
            return df['close'].iloc[-1]
        
        return None
    
    def _calculate_performance_metrics(self, portfolio_history: List[Dict]) -> Dict:
        """Calculate performance metrics from backtest results."""
        if not portfolio_history or not self.trades:
            return {
                'final_equity': self.initial_capital,
                'total_return_pct': 0.0,
                'annualized_return_pct': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown_pct': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0
            }
        
        # Convert to DataFrame
        equity_df = pd.DataFrame(portfolio_history)
        
        # Calculate returns
        initial_equity = self.initial_capital
        final_equity = equity_df['equity'].iloc[-1]
        total_return = final_equity - initial_equity
        total_return_pct = (total_return / initial_equity) * 100
        
        # Calculate daily returns
        equity_df['return'] = equity_df['equity'].pct_change()
        
        # Days in backtest
        days = (equity_df['date'].iloc[-1] - equity_df['date'].iloc[0]).days
        years = days / 365.0
        
        # Annualized return
        annualized_return_pct = ((1 + total_return_pct/100) ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0%)
        daily_returns = equity_df['return'].dropna()
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        
        # Calculate maximum drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['cummax'] - equity_df['equity']) / equity_df['cummax']
        max_drawdown_pct = equity_df['drawdown'].max() * 100
        
        # Calculate trade statistics
        trades_df = pd.DataFrame(self.trades)
        
        if trades_df.empty:
            win_rate = 0.0
            profit_factor = 0.0
            total_trades = 0
        else:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] <= 0]
            
            win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
            
            total_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0
            total_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0
            
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            total_trades = len(trades_df)
        
        # Compile results
        results = {
            'initial_equity': initial_equity,
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'annualized_return_pct': annualized_return_pct,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown_pct,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'backtest_days': days,
            'backtest_years': years
        }
        
        return results
    
    def _save_backtest_results(self, results: Dict, portfolio_history: List[Dict]) -> None:
        """Save backtest results to files and create visualizations."""
        logger.info("Saving backtest results")
        
        # Save performance metrics
        metrics_df = pd.DataFrame([results])
        metrics_file = os.path.join(self.output_dir, "performance_metrics.csv")
        metrics_df.to_csv(metrics_file, index=False)
        
        # Save portfolio history
        portfolio_df = pd.DataFrame(portfolio_history)
        portfolio_file = os.path.join(self.output_dir, "equity_curve.csv")
        portfolio_df.to_csv(portfolio_file, index=False)
        
        # Save trades
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_file = os.path.join(self.output_dir, "trades.csv")
            trades_df.to_csv(trades_file, index=False)
        
        # Save signals
        if self.trade_signals:
            signals_df = pd.DataFrame(self.trade_signals)
            signals_file = os.path.join(self.output_dir, "signals.csv")
            signals_df.to_csv(signals_file, index=False)
        
        # Create visualizations
        self.plot_equity_curve(portfolio_df)
        self.plot_drawdown(portfolio_df)
        self.plot_trade_distribution(self.trades)
        self.plot_monthly_returns(portfolio_df)
        self.plot_performance_summary(results)
    
    def plot_equity_curve(self, portfolio_df: pd.DataFrame) -> None:
        """Plot equity curve."""
        if portfolio_df.empty:
            return
            
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_df['date'], portfolio_df['equity'], linewidth=2)
        plt.title('Equity Curve', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Equity ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()
        
        # Add horizontal line for initial capital
        plt.axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.7, 
                    label=f'Initial Capital (${self.initial_capital:,.2f})')
        
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, "equity_curve.png"))
        plt.close()
    
    def plot_drawdown(self, portfolio_df: pd.DataFrame) -> None:
        """Plot drawdown over time."""
        if portfolio_df.empty:
            return
            
        # Calculate drawdown
        portfolio_df['cummax'] = portfolio_df['equity'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['cummax'] - portfolio_df['equity']) / portfolio_df['cummax'] * 100
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(portfolio_df['date'], 0, portfolio_df['drawdown'], color='red', alpha=0.3)
        plt.plot(portfolio_df['date'], portfolio_df['drawdown'], color='red', linewidth=1)
        
        plt.title('Drawdown Over Time', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()
        
        # Invert y-axis (drawdowns are negative)
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, "drawdown.png"))
        plt.close()
    
    def plot_trade_distribution(self, trades: List[Dict]) -> None:
        """Plot trade distribution by pair and profit/loss."""
        if not trades:
            return
            
        trades_df = pd.DataFrame(trades)
        
        # Plot distribution of trades by currency pair
        plt.figure(figsize=(12, 6))
        
        # Create pair labels
        trades_df['pair_label'] = trades_df['pair1'] + '/' + trades_df['pair2']
        
        # Count trades by pair
        pair_counts = trades_df['pair_label'].value_counts()
        
        # Plot
        colors = plt.cm.viridis(np.linspace(0, 1, len(pair_counts)))
        pair_counts.plot(kind='bar', color=colors)
        
        plt.title('Number of Trades by Currency Pair', fontsize=14)
        plt.xlabel('Currency Pair', fontsize=12)
        plt.ylabel('Number of Trades', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, "trade_distribution.png"))
        plt.close()
        
        # Plot distribution of profit/loss
        plt.figure(figsize=(12, 6))
        
        # Add profit/loss category
        trades_df['result'] = trades_df['pnl'].apply(lambda x: 'Profit' if x > 0 else 'Loss')
        
        # Calculate average profit and loss
        avg_profit = trades_df[trades_df['pnl'] > 0]['pnl'].mean()
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean()
        
        # Plot histogram of P&L
        plt.hist(trades_df['pnl'], bins=20, alpha=0.7, color='skyblue')
        
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        plt.axvline(x=avg_profit, color='green', linestyle='--', alpha=0.7, 
                   label=f'Avg Profit: ${avg_profit:.2f}')
        plt.axvline(x=avg_loss, color='orange', linestyle='--', alpha=0.7, 
                   label=f'Avg Loss: ${avg_loss:.2f}')
        
        plt.title('Distribution of Trade Profit/Loss', fontsize=14)
        plt.xlabel('Profit/Loss ($)', fontsize=12)
        plt.ylabel('Number of Trades', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, "pnl_distribution.png"))
        plt.close()
    
    def plot_monthly_returns(self, portfolio_df: pd.DataFrame) -> None:
        """Plot monthly returns heatmap."""
        if portfolio_df.empty:
            return
            
        # Convert date column to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(portfolio_df['date']):
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        
        # Calculate daily returns
        portfolio_df['daily_return'] = portfolio_df['equity'].pct_change()
        
        # Create month and year columns
        portfolio_df['year'] = portfolio_df['date'].dt.year
        portfolio_df['month'] = portfolio_df['date'].dt.month
        
        # Calculate monthly returns
        monthly_returns = portfolio_df.groupby(['year', 'month'])['daily_return'].apply(
            lambda x: (1 + x).prod() - 1
        ).reset_index()
        
        # Create a pivot table for the heatmap
        pivot_table = monthly_returns.pivot(index='month', columns='year', values='daily_return')
        
        # Convert month numbers to names
        month_names = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        pivot_table.index = [month_names[m] for m in pivot_table.index]
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        
        # Define colormap (red for negative, green for positive)
        cmap = plt.cm.RdYlGn
        
        # Create heatmap
        sns_heatmap = plt.pcolormesh(pivot_table.columns, range(len(pivot_table.index)), 
                                    pivot_table.values, cmap=cmap, vmin=-0.1, vmax=0.1)
        
        # Add colorbar
        cbar = plt.colorbar(sns_heatmap)
        cbar.set_label('Monthly Return (%)', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                try:
                    value = pivot_table.iloc[i, j]
                    if not np.isnan(value):
                        text_color = 'white' if abs(value) > 0.05 else 'black'
                        plt.text(j + 0.5, i + 0.5, f'{value:.1%}', 
                                 ha='center', va='center', color=text_color)
                except IndexError:
                    pass
        
        # Set labels and title
        plt.title('Monthly Returns Heatmap', fontsize=14)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Month', fontsize=12)
        
        # Set ticks
        plt.yticks(np.arange(len(pivot_table.index)) + 0.5, pivot_table.index)
        plt.xticks(np.arange(len(pivot_table.columns)) + 0.5, pivot_table.columns)
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, "monthly_returns.png"))
        plt.close()
    
    def plot_performance_summary(self, results: Dict) -> None:
        """Plot performance summary."""
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Title for the entire figure
        fig.suptitle('Performance Summary', fontsize=16)
        
        # 1. Returns and Drawdown
        axs[0, 0].bar(['Total Return', 'Max Drawdown'], 
                     [results['total_return_pct'], -results['max_drawdown_pct']], 
                     color=['green', 'red'])
        
        axs[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axs[0, 0].set_title('Returns vs Drawdown (%)')
        axs[0, 0].grid(axis='y', alpha=0.3)
        
        # Add values on top of bars
        for i, v in enumerate([results['total_return_pct'], -results['max_drawdown_pct']]):
            axs[0, 0].text(i, v + (5 if v >= 0 else -5), f'{v:.2f}%', 
                         ha='center', color=('black' if v >= 0 else 'white'))
        
        # 2. Win Rate and Profit Factor
        axs[0, 1].bar(['Win Rate', 'Profit Factor'], 
                     [results['win_rate'], results['profit_factor']], 
                     color=['skyblue', 'orange'])
        
        axs[0, 1].set_title('Win Rate (%) and Profit Factor')
        axs[0, 1].grid(axis='y', alpha=0.3)
        
        # Add values on top of bars
        for i, v in enumerate([results['win_rate'], results['profit_factor']]):
            axs[0, 1].text(i, v + 2, f'{v:.2f}', ha='center')
        
        # 3. Trade Statistics
        trade_stats = [
            results['total_trades'], 
            results['total_trades'] * results['win_rate'] / 100,
            results['total_trades'] * (1 - results['win_rate'] / 100)
        ]
        
        axs[1, 0].bar(['Total Trades', 'Winning Trades', 'Losing Trades'], 
                     trade_stats, 
                     color=['purple', 'green', 'red'])
        
        axs[1, 0].set_title('Trade Statistics')
        axs[1, 0].grid(axis='y', alpha=0.3)
        
        # Add values on top of bars
        for i, v in enumerate(trade_stats):
            axs[1, 0].text(i, v + 1, f'{int(v)}', ha='center')
        
        # 4. Key Metrics
        metrics = [
            f'Initial Capital: ${self.initial_capital:,.2f}',
            f'Final Equity: ${results["final_equity"]:,.2f}',
            f'Total Return: {results["total_return_pct"]:.2f}%',
            f'Annualized Return: {results["annualized_return_pct"]:.2f}%',
            f'Sharpe Ratio: {results["sharpe_ratio"]:.2f}',
            f'Max Drawdown: {results["max_drawdown_pct"]:.2f}%',
            f'Win Rate: {results["win_rate"]:.2f}%',
            f'Profit Factor: {results["profit_factor"]:.2f}',
            f'Total Trades: {results["total_trades"]}',
            f'Backtest Period: {results["backtest_days"]} days'
        ]
        
        # Empty plot with text
        axs[1, 1].axis('off')
        y_pos = 0.95
        
        for metric in metrics:
            axs[1, 1].text(0.05, y_pos, metric, fontsize=11)
            y_pos -= 0.1
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, "performance_summary.png"))
        plt.close()

def run_backtest(
    pair_data: Dict[str, pd.DataFrame],
    z_entry: float = 2.0,
    z_exit: float = 0.0,
    p_value: float = 0.05,
    initial_capital: float = 10000.0,
    position_size_pct: float = 0.02,
    output_dir: str = "backtest_results"
) -> Dict:
    """
    Run a backtest of the statistical arbitrage strategy.
    
    Args:
        pair_data: Dictionary of currency pairs with price data
        z_entry: Z-score threshold for trade entry
        z_exit: Z-score threshold for trade exit
        p_value: P-value threshold for cointegration
        initial_capital: Initial capital for the backtest
        position_size_pct: Position size as percentage of capital
        output_dir: Directory for output files
        
    Returns:
        Dictionary with backtest results
    """
    logger.info(f"Starting backtest with z_entry={z_entry}, z_exit={z_exit}, p_value={p_value}")
    
    # Initialize backtester
    backtester = StatArbBacktester(
        pair_data=pair_data,
        z_entry=z_entry,
        z_exit=z_exit,
        p_value_threshold=p_value,
        initial_capital=initial_capital,
        position_size_pct=position_size_pct,
        output_dir=output_dir
    )
    
    # Find cointegrated pairs
    cointegration_results = backtester.find_cointegrated_pairs()
    logger.info(f"Found {len(cointegration_results)} cointegrated pairs")
    
    # Run backtest
    results = backtester.backtest(cointegration_results)
    
    # Print summary
    print("\n=== Backtest Results ===")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Equity: ${results['final_equity']:,.2f}")
    print(f"Total Return: {results['total_return_pct']:.2f}%")
    print(f"Annualized Return: {results['annualized_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Backtest Period: {results['backtest_days']} days")
    print("\nVisualizations saved to:", output_dir)
    
    return results

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Backtest statistical arbitrage strategy')
    
    parser.add_argument('--timeframe', type=str, default='H4',
                        help='Timeframe for analysis (e.g., H1, H4, D)')
    
    parser.add_argument('--lookback', type=int, default=60,
                        help='Number of days to look back for historical data')
    
    parser.add_argument('--z-entry', type=float, default=2.0,
                        help='Z-score threshold for trade entry')
    
    parser.add_argument('--z-exit', type=float, default=0.0,
                        help='Z-score threshold for trade exit')
    
    parser.add_argument('--p-value', type=float, default=0.05,
                        help='P-value threshold for cointegration test')
    
    parser.add_argument('--capital', type=float, default=10000.0,
                        help='Initial capital for backtest')
    
    parser.add_argument('--position-size', type=float, default=0.02,
                        help='Position size as percentage of capital')
    
    parser.add_argument('--pairs', type=str, 
                        default='EUR_USD,GBP_USD,USD_JPY,USD_CAD,AUD_USD,NZD_USD,EUR_GBP,EUR_JPY',
                        help='Comma-separated list of currency pairs to analyze')
    
    parser.add_argument('--output-dir', type=str, default='backtest_results',
                        help='Directory for output files')
    
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing cached data files')
    
    args = parser.parse_args()
    
    # Fetch data (or load from cache)
    fetcher = OandaDataFetcher()
    pair_data = fetcher.fetch_for_cointegration(
        lookback_days=args.lookback,
        timeframe=args.timeframe,
        instruments=args.pairs.split(',')
    )
    
    # Run backtest
    run_backtest(
        pair_data=pair_data,
        z_entry=args.z_entry,
        z_exit=args.z_exit,
        p_value=args.p_value,
        initial_capital=args.capital,
        position_size_pct=args.position_size,
        output_dir=args.output_dir
    )