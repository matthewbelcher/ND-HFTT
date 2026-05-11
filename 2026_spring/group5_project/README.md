# Predicting Implied Volatility Surfaces with Transformers

**Team 5:** Anthony Tsiantis, Natalie Sekerak
**Course:** CSE 40438 — High-Frequency Trading Technologies (Spring 2026)
**Instructor:** Prof. Matthew Belcher, University of Notre Dame

## Project Overview

This project models the S&P 500 implied volatility surface with a PatchTST-style transformer time-series model. Rather than predicting individual option prices, the model learns the structure of the full volatility surface across option deltas and maturities, then forecasts the next-day surface change.

The pipeline pulls 2005–2025 SPX options data from WRDS OptionMetrics together with zero-coupon risk-free rates, S&P 500 daily prices, and VIX features. Each surface node (10 maturities × 17 deltas = 170 channels) is treated as a time-series channel; the temporal transformer encoder operates over learned patches, fused with a context MLP and per-node embeddings before a regression head. Validation uses walk-forward folds with early stopping.

## Repository Layout

- `README.md` — This file
- `data/`
  - `data.py` — WRDS data collector (OptionMetrics, zero-coupon rates, S&P 500, VIX) → merged Excel workbook
- `transformer/`
  - `model.py` — `PatchTSTSurfaceModel` (transformer architecture, dataclass config)
  - `dataset.py` — Loader, surface-grid validation, tensor builder, walk-forward folds
  - `train.py` — Training loop, evaluation, plotting, walk-forward orchestration
- `artifacts/training_plots/` — Per-fold training-curve and predicted-vs-actual surface plots

## Build / Run

Requires WRDS access and a CUDA-capable GPU for full training.

```bash
python -m venv venv
source venv/bin/activate    # or .\venv\Scripts\Activate.ps1 on Windows
pip install numpy pandas matplotlib torch openpyxl wrds

# (Optional) regenerate the merged dataset from WRDS:
python data/data.py

# Train + walk-forward evaluate:
python transformer/train.py
```

The default training configuration uses 252-day lookback, 32-sample batches, 15% held-out test, 90-day validation windows on a 90-day walk-forward step, with 25 epochs/fold and 5-epoch early stopping.

## Authors

Anthony Tsiantis, Natalie Sekerak
