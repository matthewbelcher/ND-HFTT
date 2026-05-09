import multiprocessing
import logging
from dask.distributed import Client, LocalCluster
from data_loader import load_and_preprocess
from pair_selector import PairSelector
from stat_arb_strat import StatArbStrategy
from optuna_optimizer import OptunaOptimizer
import dask.dataframe as dd
import pandas as pd
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("distributed.shuffle").setLevel(logging.ERROR)
logging.getLogger("distributed.worker").setLevel(logging.ERROR)
logging.getLogger("distributed.core").setLevel(logging.ERROR)

def plot_results(price_df, signals, spread, sym1, sym2):
    """Visualize the trading strategy results."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    spread_rolling = spread.rolling(window=20).mean()
    z_score = (spread - spread_rolling) / spread.rolling(window=20).std()
    
    ax1.plot(price_df.index, price_df[sym1], label=sym1)
    ax1.plot(price_df.index, price_df[sym2], label=sym2)
    ax1.set_title(f"Prices: {sym1} vs {sym2}")
    ax1.legend()
    
    ax2.plot(spread.index, spread, label='Spread')
    ax2.plot(spread.index, spread_rolling, label='Rolling Mean')
    ax2.set_title('Spread and Rolling Mean')
    ax2.legend()
    
    ax3.plot(z_score.index, z_score, label='Z-Score')
    ax3.plot(signals.index, signals, label='Signals', linestyle='--')
    ax3.set_title('Z-Score and Trading Signals')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('trading_results.png')
    logging.info("Saved visualization to trading_results.png")

def main():
    """Main function to run the statistical arbitrage strategy."""
    parquet_path = 'data.parquet'
    csv_filepath = 'path/to/your/data.csv'  # REPLACE WITH ACTUAL CSV FILE PATH
    try:
        ddf = load_and_preprocess(parquet_path=parquet_path, start_date='2023-01-01', end_date='2023-12-31', filepath=csv_filepath)
        logging.info("Data loaded and preprocessed.")
    except Exception as e:
        logging.error(f"Failed to load data: {str(e)}")
        return

    # Compute top symbols by tick count
    top_syms = (
        ddf.SYM_ROOT
           .value_counts()
           .nlargest(50)
           .compute()
           .index
           .tolist()
    )
    logging.info(f"Top symbols: {top_syms}")

    # Filter data to top symbols
    ddf_small = ddf[ddf.SYM_ROOT.isin(top_syms)].persist()

    # Select cointegrated pairs
    selector = PairSelector(ddf_small, top_syms)
    pairs = selector.select_pairs(corr_threshold=0.70, coint_pval_threshold=0.05)
    if not pairs:
        logging.error("No cointegrated pairs found. Exiting.")
        return

    # Select the first cointegrated pair
    sym1, sym2, corr, pval, lag, lag_corr = pairs[0]
    logging.info(f"Selected pair: {sym1}/{sym2}, corr={corr:.3f}, pval={pval:.4f}, lag={lag}, lag_corr={lag_corr:.3f}")

    # Prepare price data for the selected pair
    d1 = (
        ddf_small[ddf_small.SYM_ROOT == sym1][['datetime', 'mid']]
        .rename(columns={'mid': 'm1'})
    )
    d2 = (
        ddf_small[ddf_small.SYM_ROOT == sym2][['datetime', 'mid']]
        .rename(columns={'mid': 'm2'})
    )
    merged = dd.merge(d1, d2, on='datetime', how='inner').compute()
    price_df = (
        merged
        .set_index('datetime')
        .sort_index()[['m1', 'm2']]
        .rename(columns={'m1': sym1, 'm2': sym2})
    )

    # Split data into train, validation, and test sets
    train = price_df['2023-01-01':'2023-06-30']
    val = price_df['2023-07-01':'2023-11-30']
    test = price_df['2023-12-01':'2023-12-31']

    # Optimize parameters using Optuna
    logging.info("Optimizing parameters...")
    optimizer = OptunaOptimizer(train, trade_target=5000)
    best_params = optimizer.optimize(n_trials=500)
    logging.info(f"Optimized parameters: {best_params}")

    # Backtest on validation set
    logging.info("Backtesting on validation set...")
    strat_val = StatArbStrategy(val)
    strat_val.fit_hedge_ratio()
    spread_val = strat_val.compute_spread()
    signals_val = strat_val.generate_signals(**best_params)
    val_returns = strat_val.backtest_returns(signals_val)
    val_pnl = val_returns.sum()
    logging.info(f"Validation simple PnL for {sym1}/{sym2}: {val_pnl:.4f}")

    # Backtest on test set
    logging.info("Backtesting on test set...")
    strat = StatArbStrategy(test)
    strat.fit_hedge_ratio()
    spread = strat.compute_spread()
    signals = strat.generate_signals(**best_params)
    test_returns = strat.backtest_returns(signals)
    signal_changes = signals.diff().fillna(0)
    n_trades = (signal_changes != 0).sum()
    simple_pnl = test_returns.sum()
    logging.info(f"Number of trades: {n_trades}")
    logging.info(f"Test simple PnL for {sym1}/{sym2}: {simple_pnl:.4f}")

    # Visualize the results
    plot_results(test, signals, spread, sym1, sym2)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    cluster = LocalCluster(silence_logs=logging.ERROR, dashboard_address=None)
    client = Client(cluster)
    logging.info("Dask client started.")
    try:
        main()
    finally:
        client.close()
        cluster.close()