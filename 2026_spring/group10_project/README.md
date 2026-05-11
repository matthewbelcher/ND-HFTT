# JEVT — CPI and FOMC News Reaction in BTC-USDT Futures

**Team JEVT:** Jack Rellinger, Ty Friedman, Vinny Galassi, Ethan Koran
**Course:** CSE 40438 — High-Frequency Trading Technologies (Spring 2026)
**Instructor:** Prof. Matthew Belcher, University of Notre Dame

## Project Overview

This project studies whether scheduled macroeconomic releases create a tradable signal in BTC-USDT, and how that signal degrades as execution latency grows. The research question is approached in three phases:

1. **One-minute event study.** Align Binance BTC-USDT klines around CPI release times and characterize normalized returns before and after each event.
2. **Delay/hold backtesting.** Convert the CPI surprise (actual − forecast) into a directional trade, then sweep entry delay (0, 1, 2, 5, 10, 30 minutes) × holding period (1, 5, 10, 30, 60 minutes) to measure how profitability decays with latency.
3. **Trade-level feature research.** Move from 1-minute bars to per-trade data; engineer order-flow imbalance, early returns, pre-release drift, post-release continuation/reversal, and volatility features around CPI release windows. Validate with leave-one-out cross-validation, walk-forward testing, and permutation significance tests.

The pipeline is later extended to FOMC events. Results are presented across 22 CPI release windows from January 2024 through March 2026.

## Repository Layout

- `final_report.md` — Final report
- `PROJECT_NOTES.md` — Detailed project notes (phases, findings, lessons learned)
- `README.md` — This file
- `src/scripts/` — Phase 1 (event study) and Phase 2 (delay/hold backtest) scripts:
  - `klines_data_extraction.py`, `ticker_data_extraction.py` — Binance data collection
  - `clean_trades.py`, `build_cpi_signal_table.py` — Data prep
  - `signal_detection.py` — Per-release event alignment + plotting
  - `backtest_signals_1m_klines.py` — Delay/hold backtest
  - `plot_1m_klines_backtest_results.py` — Heatmap + win-summary plots
- `src/scripts/trades/` — Phase 3 trade-level feature research package:
  - `data_loader.py`, `features.py`, `signals.py` — Feature engineering and signal definitions
  - `validation.py` — Walk-forward, LOOCV, and permutation tests
  - `analysis.py`, `plot_signal_performance.py` — Statistics and visualization
  - `run_cpi_research.py` — Entry point
  - `config.py` — Hyperparameters and validation constants
- `results/plots/` — Per-event return paths, delay/hold heatmaps, feature correlations, signal pass/fail matrix, and permutation p-value plots

## Build / Run

```bash
pip install pandas numpy matplotlib scipy scikit-learn
cd 2026_spring/group10_project

# Phase 1 + 2: event study and delay/hold backtest
python src/scripts/signal_detection.py
python src/scripts/backtest_signals_1m_klines.py

# Phase 3: trade-level feature research
python src/scripts/trades/run_cpi_research.py
```

Data sources: Binance BTC-USDT klines and trade data, BLS CPI release timing, historical actual-vs-forecast CPI from investing.com, FOMC event timestamps.

## Authors

Jack Rellinger, Ty Friedman, Vinny Galassi, Ethan Koran
