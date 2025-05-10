"""
Example script to run a backtest of the statistical arbitrage strategy.
This script shows a complete workflow from data fetching to backtest analysis.
"""

import os
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Import our modules
from fetch_oanda_data import OandaDataFetcher
from cointegration_analysis import find_cointegrated_pairs
from backtest import run_backtest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('RunBacktest')

def main():
    """Run a complete backtest of the statistical arbitrage strategy."""
    print("=== Statistical Arbitrage Backtest ===")
    
    # Step 1: Set parameters
    timeframe = "H4"  # 4-hour candles
    lookback_days = 365 # 6 months of data
    z_entry = 1.0  # Entry when z-score exceeds 2.0
    z_exit = 0.05  # Exit when z-score crosses zero
    p_value = 0.5  # Cointegration threshold
    initial_capital = 10000.0  # Starting capital
    position_size = 0.2  # 2% of capital per trade
    
    # Currency pairs to analyze
    pairs = [
        "EUR_USD", "GBP_USD", "USD_JPY", "USD_CAD", 
        "AUD_USD", "NZD_USD", "EUR_GBP", "EUR_JPY"
    ]
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("backtest_results", f"backtest_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 2: Fetch or load data
    print("\nFetching historical data...")
    fetcher = OandaDataFetcher()
    
    try:
        pair_data = fetcher.fetch_for_cointegration(
            lookback_days=lookback_days,
            timeframe=timeframe,
            instruments=pairs
        )
        
        # Save fetched data
        data_dir = os.path.join(output_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        for pair, df in pair_data.items():
            filename = f"{pair}_{timeframe}.csv"
            filepath = os.path.join(data_dir, filename)
            df.to_csv(filepath, index=False)
            print(f"Saved {len(df)} rows of {pair} data to {filepath}")
    
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        return
    
    # Step 3: Run backtest
    print("\nRunning backtest...")
    try:
        results = run_backtest(
            pair_data=pair_data,
            z_entry=z_entry,
            z_exit=z_exit,
            p_value=p_value,
            initial_capital=initial_capital,
            position_size_pct=position_size,
            output_dir=output_dir
        )
        
        # Step 4: Display additional analysis
        print("\nGenerating additional analysis...")
        generate_parameter_sensitivity(
            pair_data=pair_data,
            base_params={
                'z_entry': z_entry,
                'z_exit': z_exit,
                'p_value': p_value,
                'initial_capital': initial_capital,
                'position_size_pct': position_size
            },
            output_dir=output_dir
        )
        
        print(f"\nBacktest completed successfully. Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error in backtest: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"Backtest failed. See log for details.")

def generate_parameter_sensitivity(
    pair_data, 
    base_params, 
    output_dir
):
    """
    Generate parameter sensitivity analysis.
    
    Args:
        pair_data: Dictionary of currency pairs with price data
        base_params: Base parameters for the backtest
        output_dir: Directory for output files
    """
    # Create directory for sensitivity analysis
    sensitivity_dir = os.path.join(output_dir, "sensitivity")
    os.makedirs(sensitivity_dir, exist_ok=True)
    
    # 1. Z-entry sensitivity
    print("  Testing z-entry sensitivity...")
    z_entry_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    z_entry_results = []
    
    for z_entry in z_entry_values:
        try:
            result = run_backtest(
                pair_data=pair_data,
                z_entry=z_entry,
                z_exit=base_params['z_exit'],
                p_value=base_params['p_value'],
                initial_capital=base_params['initial_capital'],
                position_size_pct=base_params['position_size_pct'],
                output_dir=os.path.join(sensitivity_dir, f"z_entry_{z_entry}")
            )
            result['z_entry'] = z_entry
            z_entry_results.append(result)
        except Exception as e:
            logger.error(f"Error in z-entry sensitivity test with value {z_entry}: {str(e)}")
    
    if z_entry_results:
        plot_sensitivity(z_entry_results, 'z_entry', 'Z-Entry Threshold', 
                       ['total_return_pct', 'sharpe_ratio', 'win_rate'],
                       ['Total Return (%)', 'Sharpe Ratio', 'Win Rate (%)'],
                       sensitivity_dir)
    
    # 2. P-value sensitivity
    print("  Testing p-value sensitivity...")
    p_value_values = [0.01, 0.025, 0.05, 0.075, 0.1]
    p_value_results = []
    
    for p_value in p_value_values:
        try:
            result = run_backtest(
                pair_data=pair_data,
                z_entry=base_params['z_entry'],
                z_exit=base_params['z_exit'],
                p_value=p_value,
                initial_capital=base_params['initial_capital'],
                position_size_pct=base_params['position_size_pct'],
                output_dir=os.path.join(sensitivity_dir, f"p_value_{p_value}")
            )
            result['p_value'] = p_value
            p_value_results.append(result)
        except Exception as e:
            logger.error(f"Error in p-value sensitivity test with value {p_value}: {str(e)}")
    
    if p_value_results:
        plot_sensitivity(p_value_results, 'p_value', 'P-Value Threshold',
                       ['total_return_pct', 'total_trades', 'win_rate'],
                       ['Total Return (%)', 'Total Trades', 'Win Rate (%)'],
                       sensitivity_dir)

def plot_sensitivity(
    results, 
    param_name, 
    param_label, 
    metrics, 
    metric_labels, 
    output_dir
):
    """
    Plot parameter sensitivity analysis.
    
    Args:
        results: List of backtest results
        param_name: Name of the parameter
        param_label: Label for the parameter
        metrics: List of metrics to plot
        metric_labels: Labels for the metrics
        output_dir: Directory for output files
    """
    if not results:
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Create figure with multiple y-axes
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 8), sharex=True)
    if len(metrics) == 1:
        axes = [axes]  # Make it a list for consistent indexing
    
    # Plot each metric
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        axes[i].plot(df[param_name], df[metric], marker='o', linewidth=2)
        axes[i].set_ylabel(label, fontsize=12)
        axes[i].grid(True, alpha=0.3)
        
        # Add values on the chart
        for x, y in zip(df[param_name], df[metric]):
            axes[i].annotate(f'{y:.2f}', (x, y), xytext=(0, 5), 
                          textcoords='offset points', ha='center')
    
    # Set x-axis label on bottom chart
    axes[-1].set_xlabel(param_label, fontsize=12)
    
    # Add title
    plt.suptitle(f'Parameter Sensitivity Analysis: {param_label}', fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    
    # Save plot
    plt.savefig(os.path.join(output_dir, f"{param_name}_sensitivity.png"))
    plt.close()
    
    # Save results to CSV
    filepath = os.path.join(output_dir, f"{param_name}_sensitivity.csv")
    df.to_csv(filepath, index=False)

if __name__ == "__main__":
    main()