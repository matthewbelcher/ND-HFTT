"""
Module for creating visualizations of cointegration data and trading signals.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple
import logging
from datetime import datetime
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('Visualization')

class CointegrationVisualizer:
    """Class for visualizing cointegration relationships and trading signals."""
    
    def __init__(self, output_dir: str = "plots"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory for saving plots (default: "plots")
        """
        # Store as string to avoid path issues
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure plot style
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        logger.info(f"Initialized CointegrationVisualizer with output directory: {output_dir}")
    
    def plot_price_series(
        self,
        price1: pd.Series,
        price2: pd.Series,
        pair1: str,
        pair2: str,
        title: Optional[str] = None,
        filename: Optional[str] = None
    ):
        """
        Plot two price series on the same chart.
        
        Args:
            price1: Price series for first currency pair
            price2: Price series for second currency pair
            pair1: Name of first currency pair
            pair2: Name of second currency pair
            title: Optional custom title
            filename: Optional filename for saving the plot
        """
        plt.figure(figsize=(12, 6))
        
        # Normalize prices to start at 100 for comparison
        norm_price1 = price1 / price1.iloc[0] * 100
        norm_price2 = price2 / price2.iloc[0] * 100
        
        plt.plot(norm_price1.index, norm_price1, label=pair1, color=self.colors[0], linewidth=2)
        plt.plot(norm_price2.index, norm_price2, label=pair2, color=self.colors[1], linewidth=2)
        
        if not title:
            title = f"Normalized Price Comparison: {pair1} vs {pair2}"
        
        plt.title(title, fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Normalized Price (Base=100)', fontsize=12)
        plt.legend()
        plt.tight_layout()
        
        if filename:
            try:
                plt.savefig(filename)
                logger.info(f"Saved price comparison plot to {filename}")
            except Exception as e:
                logger.error(f"Error saving price comparison plot: {str(e)}")
        
        plt.close()
    
    def plot_spread_and_zscore(
        self, 
        result: Dict,
        filename: Optional[str] = None,
        show_signals: bool = False,
        signals: Optional[List[Dict]] = None
    ):
        """
        Plot spread and z-score from cointegration analysis.
        
        Args:
            result: Cointegration analysis result dictionary
            filename: Optional filename for saving the plot
            show_signals: Whether to overlay trading signals
            signals: List of trading signals (required if show_signals is True)
        """
        if 'timestamps' not in result or 'spread_series' not in result or 'z_score_series' not in result:
            logger.error("Missing required data in result dictionary")
            return
        
        pair1 = result['pair1']
        pair2 = result['pair2']
        
        timestamps = pd.to_datetime(result['timestamps'])
        spread = result['spread_series']
        z_score = result['z_score_series']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot spread
        ax1.plot(timestamps, spread, label='Spread', color=self.colors[0], linewidth=2)
        ax1.set_title(f'Spread between {pair1} and {pair2}', fontsize=14)
        ax1.set_ylabel('Spread Value', fontsize=12)
        ax1.axhline(y=result['spread_mean'], color='gray', linestyle='--', alpha=0.7, label='Mean')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot z-score
        ax2.plot(timestamps, z_score, label='Z-Score', color=self.colors[1], linewidth=2)
        ax2.set_title(f'Z-Score of Spread', fontsize=14)
        ax2.set_ylabel('Z-Score', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        
        # Add horizontal lines for z-score thresholds
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.7)
        ax2.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='Entry (+2)')
        ax2.axhline(y=-2, color='green', linestyle='--', alpha=0.7, label='Entry (-2)')
        ax2.axhline(y=1, color='orange', linestyle=':', alpha=0.5, label='Exit (+1)')
        ax2.axhline(y=-1, color='orange', linestyle=':', alpha=0.5, label='Exit (-1)')
        
        # Highlight current z-score
        current_z = result['current_z_score']
        ax2.scatter(timestamps[-1], current_z, s=100, marker='o', color='purple', 
                   label=f'Current Z-Score: {current_z:.2f}')
        
        # Overlay trading signals if requested
        if show_signals and signals:
            self._add_signals_to_plot(ax2, signals, timestamps, pair1, pair2)
        
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if filename:
            try:
                plt.savefig(filename)
                logger.info(f"Saved spread and z-score plot to {filename}")
            except Exception as e:
                logger.error(f"Error saving spread and z-score plot: {str(e)}")
        
        plt.close()
    
    def _add_signals_to_plot(self, ax, signals, timestamps, pair1, pair2):
        """Add trading signals to a plot."""
        for signal in signals:
            if signal['pair1'] == pair1 and signal['pair2'] == pair2:
                signal_time = pd.to_datetime(signal['signal_time'])
                # Find the closest timestamp
                closest_idx = min(range(len(timestamps)), 
                                 key=lambda i: abs(timestamps[i] - signal_time))
                
                if closest_idx < len(timestamps):
                    z_score = signal['z_score']
                    
                    if signal['signal_type'] == 'ENTRY':
                        marker = '^' if signal['action1'] == 'BUY' else 'v'
                        color = 'green' if signal['action1'] == 'BUY' else 'red'
                        ax.scatter(timestamps[closest_idx], z_score, s=120, marker=marker, 
                                  color=color, edgecolors='black', linewidth=1.5,
                                  label=f"{signal['action1']} Signal")
                    else:  # EXIT
                        ax.scatter(timestamps[closest_idx], z_score, s=120, marker='X', 
                                  color='blue', edgecolors='black', linewidth=1.5,
                                  label='Exit Signal')
    
    def plot_correlation_heatmap(self, pair_data: Dict[str, pd.DataFrame], filename: Optional[str] = None):
        """
        Plot correlation heatmap between currency pairs.
        
        Args:
            pair_data: Dictionary of currency pairs with their price DataFrames
            filename: Optional filename for saving the plot
        """
        # Create a correlation matrix
        pairs = list(pair_data.keys())
        closing_prices = {}
        
        # Get closing prices
        for pair in pairs:
            if 'close' in pair_data[pair].columns:
                closing_prices[pair] = pair_data[pair]['close']
            elif 'timestamp' in pair_data[pair].columns and pair_data[pair].set_index('timestamp').index.is_unique:
                closing_prices[pair] = pair_data[pair].set_index('timestamp')['close']
        
        # If we have enough data to create a correlation matrix
        if len(closing_prices) > 1:
            # Create DataFrame with all pairs aligned
            corr_df = pd.DataFrame()
            for pair, prices in closing_prices.items():
                corr_df[pair] = prices
            
            # Drop rows with NaN values
            corr_df = corr_df.dropna()
            
            # Calculate correlation matrix
            corr_matrix = corr_df.corr()
            
            # Plot heatmap
            plt.figure(figsize=(10, 8))
            plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            
            # Add colorbar
            plt.colorbar(label='Correlation')
            
            # Add labels
            plt.xticks(range(len(pairs)), pairs, rotation=45, ha='right')
            plt.yticks(range(len(pairs)), pairs)
            
            # Add correlation values
            for i in range(len(pairs)):
                for j in range(len(pairs)):
                    plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                             ha='center', va='center', color='black')
            
            plt.title('Correlation Matrix of Currency Pairs', fontsize=14)
            plt.tight_layout()
            
            if filename:
                try:
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    plt.savefig(filename)
                    logger.info(f"Saved correlation heatmap to {filename}")
                except Exception as e:
                    logger.error(f"Error saving correlation heatmap: {str(e)}")
            
            plt.close()
        else:
            logger.warning("Not enough data to create correlation heatmap")
    
    def plot_cointegration_results(self, results: List[Dict], filename: Optional[str] = None):
        """
        Plot summary of cointegration results.
        
        Args:
            results: List of cointegration analysis results
            filename: Optional filename for saving the plot
        """
        if not results:
            logger.warning("No cointegration results to plot")
            return
        
        # Extract p-values and pair names
        pairs = [f"{r['pair1']}/{r['pair2']}" for r in results]
        p_values = [r['p_value'] for r in results]
        
        # Sort by p-value
        sorted_indices = np.argsort(p_values)
        pairs = [pairs[i] for i in sorted_indices]
        p_values = [p_values[i] for i in sorted_indices]
        
        # Limit to top 15 for readability
        if len(pairs) > 15:
            pairs = pairs[:15]
            p_values = p_values[:15]
        
        # Plot
        plt.figure(figsize=(12, 6))
        bars = plt.barh(pairs, p_values, color=self.colors[2])
        
        # Add 0.05 threshold line
        plt.axvline(x=0.05, color='red', linestyle='--', 
                    label='Significance Threshold (p=0.05)')
        
        # Annotate p-values
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                    f'{p_values[i]:.4f}', va='center')
        
        plt.xlabel('P-Value (lower is more significant)', fontsize=12)
        plt.ylabel('Currency Pairs', fontsize=12)
        plt.title('Cointegration Test Results (P-Values)', fontsize=14)
        plt.legend()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if filename:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                plt.savefig(filename)
                logger.info(f"Saved cointegration results plot to {filename}")
            except Exception as e:
                logger.error(f"Error saving cointegration results plot: {str(e)}")
        
        plt.close()
    
    def plot_portfolio_performance(self, 
                                 performance_data: pd.DataFrame, 
                                 filename: Optional[str] = None):
        """
        Plot performance of statistical arbitrage strategy.
        
        Args:
            performance_data: DataFrame with performance metrics
            filename: Optional filename for saving the plot
        """
        if performance_data.empty:
            logger.warning("No performance data to plot")
            return
        
        # Ensure we have a datetime index
        if not isinstance(performance_data.index, pd.DatetimeIndex):
            if 'date' in performance_data.columns:
                performance_data = performance_data.set_index('date')
                performance_data.index = pd.to_datetime(performance_data.index)
            else:
                logger.error("Performance data must have a datetime index or 'date' column")
                return
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True, 
                               gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot equity curve
        if 'equity' in performance_data.columns:
            axes[0].plot(performance_data.index, performance_data['equity'], 
                        label='Equity', color=self.colors[0], linewidth=2)
            
            # Add initial equity as horizontal line
            initial_equity = performance_data['equity'].iloc[0]
            axes[0].axhline(y=initial_equity, color='gray', linestyle='--', 
                          alpha=0.7, label=f'Initial Equity ({initial_equity:.2f})')
            
            axes[0].set_title('Equity Curve', fontsize=14)
            axes[0].set_ylabel('Equity', fontsize=12)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Plot drawdowns
        if 'drawdown' in performance_data.columns:
            axes[1].fill_between(performance_data.index, 0, 
                               performance_data['drawdown'] * 100, 
                               color=self.colors[3], alpha=0.5)
            axes[1].set_title('Drawdown (%)', fontsize=14)
            axes[1].set_ylabel('Drawdown %', fontsize=12)
            axes[1].set_ylim(bottom=max(performance_data['drawdown'] * 100) * 1.1, top=0)
            axes[1].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Date', fontsize=12)
        
        plt.tight_layout()
        
        if filename:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                plt.savefig(filename)
                logger.info(f"Saved portfolio performance plot to {filename}")
            except Exception as e:
                logger.error(f"Error saving portfolio performance plot: {str(e)}")
        
        plt.close()
    
    def create_dashboard(self, 
                       cointegration_results: List[Dict], 
                       pair_data: Dict[str, pd.DataFrame],
                       signals: Optional[List[Dict]] = None,
                       performance_data: Optional[pd.DataFrame] = None):
        """
        Create a comprehensive dashboard with multiple plots.
        
        Args:
            cointegration_results: List of cointegration analysis results
            pair_data: Dictionary of currency pairs with their price DataFrames
            signals: Optional list of trading signals
            performance_data: Optional performance data DataFrame
        """
        logger.info("Creating comprehensive dashboard")
        
        try:
            # Create timestamp for dashboard
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create dashboard directory (using absolute paths to avoid confusion)
            dashboard_name = f"dashboard_{timestamp}"
            dashboard_dir = os.path.abspath(os.path.join("output", dashboard_name))
            os.makedirs(dashboard_dir, exist_ok=True)
            
            logger.info(f"Dashboard directory: {dashboard_dir}")
            
            # 1. Correlation heatmap
            correlation_file = os.path.join(dashboard_dir, "correlation_heatmap.png")
            self.plot_correlation_heatmap(pair_data, filename=correlation_file)
            
            # 2. Cointegration results summary
            if cointegration_results:
                cointegration_file = os.path.join(dashboard_dir, "cointegration_results.png")
                self.plot_cointegration_results(cointegration_results, filename=cointegration_file)
            
            # 3. Top cointegrated pairs
            for i, result in enumerate(cointegration_results[:5]):  # Top 5
                pair1 = result['pair1']
                pair2 = result['pair2']
                
                # Price comparison
                if pair1 in pair_data and pair2 in pair_data:
                    df1 = pair_data[pair1]
                    df2 = pair_data[pair2]
                    
                    # Convert to Series if DataFrames
                    price1 = df1['close'] if 'close' in df1.columns else df1
                    price2 = df2['close'] if 'close' in df2.columns else df2
                    
                    # Ensure Series have DatetimeIndex
                    if not isinstance(price1.index, pd.DatetimeIndex) and 'timestamp' in df1.columns:
                        price1 = df1.set_index('timestamp')['close']
                    
                    if not isinstance(price2.index, pd.DatetimeIndex) and 'timestamp' in df2.columns:
                        price2 = df2.set_index('timestamp')['close']
                    
                    price_file = os.path.join(dashboard_dir, f"price_comparison_{pair1}_{pair2}.png")
                    self.plot_price_series(price1, price2, pair1, pair2, filename=price_file)
                
                # Spread and z-score
                spread_file = os.path.join(dashboard_dir, f"spread_zscore_{pair1}_{pair2}.png")
                self.plot_spread_and_zscore(
                    result, 
                    filename=spread_file,
                    show_signals=signals is not None,
                    signals=signals
                )
            
            # 4. Performance data if available
            if performance_data is not None and not performance_data.empty:
                perf_file = os.path.join(dashboard_dir, "performance.png")
                self.plot_portfolio_performance(performance_data, filename=perf_file)
            
            logger.info(f"Dashboard created in {dashboard_dir}")
            return dashboard_dir
            
        except Exception as e:
            import traceback
            logger.error(f"Error creating dashboard: {str(e)}")
            logger.error(traceback.format_exc())
            return None