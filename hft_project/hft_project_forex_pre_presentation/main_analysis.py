"""
Main entry point for the statistical arbitrage analysis system.
This script combines data fetching, cointegration analysis, trading signal generation,
and visualization to create a complete forex statistical arbitrage workflow.
"""

import os
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path

# Import our modules
from fetch_oanda_data import OandaDataFetcher
from cointegration_analysis import find_cointegrated_pairs
from statistical_arbitrage import StatArbitrageStrategy
from visualization import CointegrationVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stat_arb_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('StatArbMain')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Statistical Arbitrage Analysis')
    
    parser.add_argument('--timeframe', type=str, default='H1',
                        help='Timeframe for analysis (e.g., H1, H4, D)')
    
    parser.add_argument('--lookback', type=int, default=180,
                        help='Number of days to look back for historical data')
    
    parser.add_argument('--p-value', type=float, default=0.05,
                        help='P-value threshold for cointegration test')
    
    parser.add_argument('--z-entry', type=float, default=2.0,
                        help='Z-score threshold for trade entry')
    
    parser.add_argument('--z-exit', type=float, default=0.0,
                        help='Z-score threshold for trade exit')
    
    parser.add_argument('--pairs', type=str, 
                        default='EUR_USD,GBP_USD,USD_JPY,USD_CAD,AUD_USD,NZD_USD,EUR_GBP,EUR_JPY',
                        help='Comma-separated list of currency pairs to analyze')
    
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory for output files')
    
    parser.add_argument('--skip-fetch', action='store_true',
                        help='Skip data fetching and use cached data')
    
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing cached data files')
    
    return parser.parse_args()

def load_cached_data(data_dir: str, pairs: List[str], timeframe: str) -> Dict[str, pd.DataFrame]:
    """
    Load data from cached CSV files.
    
    Args:
        data_dir: Directory containing data files
        pairs: List of currency pairs to load
        timeframe: Timeframe of the data
        
    Returns:
        Dictionary of currency pairs with their price DataFrames
    """
    logger.info(f"Loading cached data from {data_dir}")
    data_dir = Path(data_dir)
    pair_data = {}
    
    for pair in pairs:
        # Find the latest file for this pair and timeframe
        files = list(data_dir.glob(f"{pair}_{timeframe}_*.csv"))
        
        if not files:
            logger.warning(f"No cached data found for {pair}_{timeframe}")
            continue
            
        # Get the most recent file
        latest_file = max(files, key=lambda p: p.stat().st_mtime)
        
        try:
            df = pd.read_csv(latest_file)
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            pair_data[pair] = df
            logger.info(f"Loaded {len(df)} rows for {pair} from {latest_file}")
        except Exception as e:
            logger.error(f"Error loading {latest_file}: {str(e)}")
    
    return pair_data

def generate_results_summary(
    cointegration_results: List[Dict], 
    signals: List[Dict]
) -> pd.DataFrame:
    """
    Generate a summary DataFrame of cointegration results and signals.
    
    Args:
        cointegration_results: List of cointegration analysis results
        signals: List of trading signals
        
    Returns:
        DataFrame with summary information
    """
    if not cointegration_results:
        return pd.DataFrame()
    
    # Create summary rows
    rows = []
    
    for result in cointegration_results:
        pair1 = result['pair1']  # Define pair1 here
        pair2 = result['pair2']  # Define pair2 here
        
        row = {
            'pair1': pair1,
            'pair2': pair2,
            'p_value': result['p_value'],
            'hedge_ratio': result['hedge_ratio'],
            'current_z_score': result['current_z_score'],
            'spread_mean': result['spread_mean'],
            'spread_std': result['spread_std']
        }
        
        # Add signal information if available
        signal_match = [s for s in signals if s['pair1'] == pair1 and s['pair2'] == pair2]
        if signal_match:
            signal = signal_match[0]
            row['signal'] = f"{signal['action1']} {pair1} / {signal['action2']} {pair2}"
            row['signal_strength'] = signal.get('signal_strength', 0)
            row['signal_type'] = signal.get('signal_type', 'NONE')
        else:
            row['signal'] = 'NONE'
            row['signal_strength'] = 0
            row['signal_type'] = 'NONE'
        
        rows.append(row)
    
    # Create DataFrame and sort by p-value
    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values('p_value')
    
    return summary_df

def save_results_to_csv(results: Any, filename: str, output_dir: str):
    """Save results to CSV file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    filepath = output_dir / filename
    
    if isinstance(results, pd.DataFrame):
        results.to_csv(filepath, index=False)
    elif isinstance(results, list) and results:
        pd.DataFrame(results).to_csv(filepath, index=False)
    else:
        logger.warning(f"Cannot save results to {filepath}: invalid type")
        return
    
    logger.info(f"Saved results to {filepath}")

def run_analysis(args):
    """Run the full statistical arbitrage analysis workflow."""
    logger.info("Starting statistical arbitrage analysis")
    
    # Parse currency pairs
    pairs = args.pairs.split(',')
    logger.info(f"Analyzing {len(pairs)} currency pairs: {pairs}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Fetch or load data
    if args.skip_fetch:
        # Load from cached files
        pair_data = load_cached_data(args.data_dir, pairs, args.timeframe)
    else:
        # Fetch from OANDA
        logger.info(f"Fetching data for {len(pairs)} pairs, looking back {args.lookback} days")
        fetcher = OandaDataFetcher()
        pair_data = fetcher.fetch_for_cointegration(
            lookback_days=args.lookback,
            timeframe=args.timeframe,
            instruments=pairs
        )
        
        # Save fetched data
        fetcher.save_data_to_csv(pair_data, timeframe=args.timeframe, output_dir=args.data_dir)
    
    if not pair_data:
        logger.error("No data available for analysis. Exiting.")
        return
    
    logger.info(f"Data loaded for {len(pair_data)} currency pairs")
    
    # Step 2: Perform cointegration analysis
    logger.info("Performing cointegration analysis")
    cointegration_results = find_cointegrated_pairs(pair_data, threshold=args.p_value)
    
    if not cointegration_results:
        logger.warning("No cointegrated pairs found. Exiting.")
        return
    
    logger.info(f"Found {len(cointegration_results)} cointegrated pairs")
    
    # Save cointegration results
    save_results_to_csv(
        cointegration_results, 
        f"cointegration_results_{args.timeframe}_{datetime.now().strftime('%Y%m%d')}.csv", 
        output_dir
    )
    
    # Step 3: Generate trading signals
    logger.info("Generating trading signals")
    strategy = StatArbitrageStrategy(
        z_score_entry=args.z_entry,
        z_score_exit=args.z_exit
    )
    
    signals = strategy.generate_signals(cointegration_results)
    
    logger.info(f"Generated {len(signals)} trading signals")
    
    # Save signals
    save_results_to_csv(
        signals,
        f"trading_signals_{args.timeframe}_{datetime.now().strftime('%Y%m%d')}.csv",
        output_dir
    )
    
    # Create results summary
    summary_df = generate_results_summary(cointegration_results, signals)
    save_results_to_csv(
        summary_df,
        f"analysis_summary_{args.timeframe}_{datetime.now().strftime('%Y%m%d')}.csv",
        output_dir
    )
    
    # Step 4: Create visualizations
    logger.info("Creating visualizations")
    visualizer = CointegrationVisualizer(output_dir=str(output_dir / "plots"))
    
    # Create dashboard
    dashboard_dir = visualizer.create_dashboard(
        cointegration_results=cointegration_results,
        pair_data=pair_data,
        signals=signals
    )
    
    logger.info(f"Analysis complete. Results saved to {output_dir}")
    logger.info(f"Visualizations saved to {dashboard_dir}")
    
    # Print summary of top pairs
    if not summary_df.empty:
        print("\n====== TOP COINTEGRATED PAIRS ======")
        pd.set_option('display.max_rows', 10)
        pd.set_option('display.width', 120)
        print(summary_df.head(10))
        
        # Print trading signals
        entry_signals = [s for s in signals if s['signal_type'] == 'ENTRY']
        if entry_signals:
            print("\n====== TRADING SIGNALS ======")
            for signal in entry_signals:
                print(f"{signal['pair1']}/{signal['pair2']}: {signal['action1']} {signal['pair1']} / "
                     f"{signal['action2']} {signal['pair2']} (z-score: {signal['z_score']:.2f})")
    
    return {
        'cointegration_results': cointegration_results,
        'signals': signals,
        'summary': summary_df,
        'dashboard_dir': dashboard_dir
    }

if __name__ == "__main__":
    args = parse_args()
    run_analysis(args)