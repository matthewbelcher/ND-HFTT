# HFT-Bitcoin-Project: Predicting Polymarket BTC Outcomes

**Team 2:** Aynaz Namik, Leonardo Molina, Marco Tchernychev
**Course:** CSE 40438 — High-Frequency Trading Technologies (Spring 2026)
**Instructor:** Prof. Matthew Belcher, University of Notre Dame

## Project Overview

This project studies the relationship between Bitcoin spot price on Coinbase and Polymarket's BTC Up/Down 5-minute prediction contracts. The team built a multi-component pipeline: a Coinbase trade-data collector, a Polymarket REST + WebSocket collector (routed through a Mullvad VPN proxy to bypass U.S. geoblocking), an 18-feature engineering layer, four model families (logistic regression, XGBoost, random forest, SVM) with chronological train/val/test splits and per-model feature-subset search, and a bankroll-style backtester with binomial significance testing.

## Repository Layout

- `hft_project.pdf` — Final report
- `README.md` — This file
- `RUNBOOK.md` — Setup steps for the Mullvad/Docker proxy and data collection pipeline
- `requirements.txt` — Python dependencies
- `.env.example` — Sample environment file
- `scripts/`
  - `bitcoin_datacollector.py` — Coinbase Advanced Trade websocket BTC trade collector
  - `polymarket_datacollector.py` — Polymarket REST + websocket collector via SOCKS proxy
  - `README.md` — Data dictionary for collector outputs
- `src/`
  - `features.py` — Feature engineering + standalone CLI
  - `model.py` — `BaseModel` ABC with LR / XGB / RF / SVM subclasses
  - `backtest.py` — Bankroll simulator with binomial significance test
  - `train.py` — Offline training and walk-forward evaluation entrypoint
  - `pipeline.py` — Live data → features → predict orchestrator

## Build / Run

See `RUNBOOK.md` for the full data-collection setup (Mullvad VPN proxy, Docker, tmux sessions). Once data is collected:

```bash
pip install -r requirements.txt
python src/train.py --data Data/BitcoinData/btcusd_trades_*.csv --skip-svm
```

## Authors

Aynaz Namik, Leonardo Molina, Marco Tchernychev
