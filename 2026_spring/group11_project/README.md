# Kalshi Order-Book Predictor — KXBTC15M

**Team HFT:** Michael Yang, Lang Li, Derick Shi
**Course:** CSE 40438 — High-Frequency Trading Technologies (Spring 2026)
**Instructor:** Prof. Matthew Belcher, University of Notre Dame

## Project Overview

This project studies short-horizon price movement on Kalshi's KXBTC15M market — the rolling "BTC up or down in 15 minutes" prediction contract — and progressively refines four strategies against captured tick-level order-book data.

A multi-threaded C++ collector subscribes to Kalshi over WebSocket and to Coinbase's ticker stream in parallel, scheduling a dedicated worker thread for each KXBTC15M market before it opens and reconstructing an in-memory price-by-quantity order book from the orderbook_delta stream. Several hundred 15-minute sessions are captured along with synchronized BTC reference data.

On top of that data, the team tests four progressively-refined strategies: an OBI-delta market-taking signal, a calibrated micro-price correction, a gradient-boosted ensemble predicting 10-second forward direction, and a simple two-rule market maker that posts inside the spread under an OBI + BTC-momentum filter. The final maker strategy is validated on a 114-session held-out test set and a six-hour live trading run against Kalshi.

## Repository Layout

- `Final_Report.pdf` — Final 12-page report
- `README.md` — This file
- `RUNBOOK.md` — Build and run instructions
- `environment.yml`, `environment_clean.yaml` — Conda environments
- `collector/` — C++ multi-threaded data collector and live MM client
  - `Makefile`, `include/`, `src/`, `misc/`, `results/` — Build, headers, sources, utilities, and per-session MM JSON summaries
  - Workers: `kalshi_worker.hpp`, `coinbase_worker.hpp`; REST: `kalshi_rest.hpp`; order book reconstruction: `orderbook.hpp`; market maker: `market_maker.hpp`
- `analysis/` — Python signal-research and simulation library
  - `exploratory.ipynb` — Exploratory analysis notebook
  - `obi_signal_test.py` — OBI-delta market-taking signal evaluation
  - `calibrate_microprice.py` — Micro-price calibration (10 imbalance x 3 spread buckets, ~2M transitions)
  - `signals/microprice.py` — Calibrated micro-price signal
  - `market_maker_sim.py`, `market_maker_sim_gbm.py` — Maker simulators (honest fill model and GBM-filtered variant)
  - `btc_cross_momentum.py`, `ensemble_with_btc_mom.py`, `ensemble_profitability.py` — BTC cross-momentum + gradient-boosted ensemble
  - `param_sweep.py`, `signal_runner.py`, `trade_vs_cancel.py` — Sweep and execution analysis
  - `merge_plot.py`, `dashboard.py` — Visualizations
- `kalshi_live/` — Python live trading client
  - `run_live.py` — Live MM driver against Kalshi REST + WebSocket
  - `paper_simulator.py` — Paper-trading simulator
  - `features_live.py`, `orderbook.py`, `market_discovery.py` — Live feature computation and book maintenance

## Build / Run

See `RUNBOOK.md` for the full build/run flow. Briefly:

```bash
cd 2026_spring/group11_project/collector && make
./build/collector <KALSHI_KEY_ID> ../secrets/<key>.pem ../secrets/cdp_api_key.json ../data
```

Live MM:

```bash
cd 2026_spring/group11_project/kalshi_live
python run_live.py
```

Per-session market-maker JSON summaries are written to `collector/results/`.

## Authors

Michael Yang, Lang Li, Derick Shi
