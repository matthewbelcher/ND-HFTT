# An Analysis of Arbitrage Opportunities on S&P 500 ETFs

**Team 4:** Andrew Cotaj, Luke Lagunowich, Jack Blake
**Course:** CSE 40438 — High-Frequency Trading Technologies (Spring 2026)
**Instructor:** Prof. Matthew Belcher, University of Notre Dame

## Project Overview

This project is a tick-level empirical study of arbitrage opportunities across S&P 500 ETFs — SPY, VOO, IVV, and the leveraged SPXL — using NYSE TAQ NBBO data from WRDS and ETF constituent / fund-flow data from ETF Global (Massive). The pipeline:

1. Downloads NBBO ticks for the ETF and ~500 underlying constituents
2. Aggregates each constituent's quotes into weighted mid-prices (the imbalance-aware micro-price approximation)
3. Computes a basket fair-value (intraday NAV proxy) and compares against the ETF mid
4. Identifies mispricing episodes, classifies them by magnitude and duration, and simulates execution under three named latency tiers (16 µs co-located optimistic, 100 µs co-located realistic, 1.6 ms remote DC)
5. Extends the analysis to pairs trading between ETFs and a millisecond-grid lead-lag study
6. Covers normal trading sessions plus high-volatility regimes (FOMC Sept 2024, Liberation Day April 2025)

## Repository Layout

- `Final_Report.pdf` — Final 18-page report
- `code/`
  - `download_massive.py`, `build_dataset.py`, `build_etf_pairs_dataset.py` — WRDS data collection
  - `inav_calculator.py` — Weighted mid-price + basket NAV computation
  - `rescale_data.py`, `rescale_returns.py` — Drift-correction and return normalization
  - `misvaluation_analysis.py` — Basis, gap-duration, and distribution statistics
  - `execution_simulator.py` — Three-tier latency model + execution viability simulator
  - `etf_pairs_stats.py`, `lead_lag.py` — Pairs and cross-correlation analyses
  - `arb_stats.py`, `plot_*.py` — Aggregate stats and plot generation
  - `requirements.txt` — Python dependencies
  - `data_etf_pairs/`, `part3_output_rescaled/` — Cached parquet outputs and plots
  - `plots/` — Per-session ETF / basket / spread / arb plots

## Build / Run

```bash
cd code
pip install -r requirements.txt
# Run the pipeline stages in order (download → build → rescale → inav → analyze → simulate):
python download_massive.py
python build_dataset.py
python rescale_data.py
python inav_calculator.py
python misvaluation_analysis.py
python execution_simulator.py
```

WRDS access is required for the download stage.

## Authors

Andrew Cotaj, Luke Lagunowich, Jack Blake
