# Predicting Implied Volatility Surfaces with Transformers

By Anthony Tsiantis and Natalie Sekerak

## Project Overview

This project models the S&P 500 implied volatility surface with a transformer-based time-series model. In options markets, implied volatility is a normalized view of the market's expectation of future uncertainty. Instead of predicting individual option prices directly, this project learns the structure of the full volatility surface across option deltas and maturities, then forecasts the next surface movement.

The original project proposal describes an end-to-end options trading pipeline: collect Wharton Research Data Services (WRDS) options data, forecast short-horizon implied volatility surface changes, convert those forecasts into theoretical option prices, and evaluate signals in a transaction-cost-aware backtest. The current repository implements the data collection and daily transformer forecasting pieces. The model uses historical daily volatility surfaces, risk-free rate curves, S&P 500 features, and VIX features to predict the next-day change in the implied volatility surface.

## Repository Structure

```text
.
|-- README.md
|-- data/
|   |-- data.py
|   `-- raw/
|       `-- all_data.xlsx
|-- transformer/
|   |-- dataset.py
|   |-- model.py
|   |-- train.py
|   `-- results.json
`-- artifacts/
    `-- training_plots/
```

## File Guide

- `data/data.py` connects to WRDS, collects OptionMetrics volatility surface data, S&P 500 daily price data, zero-coupon risk-free rates, and VIX data, then writes the merged dataset to `data/raw/all_data.xlsx`.
- `data/raw/all_data.xlsx` is the cleaned model input file. This directory is ignored by git because the raw workbook is large.
- `transformer/dataset.py` loads `all_data.xlsx`, validates the surface grid, builds daily tensors, normalizes inputs, creates train/validation/test splits, and builds walk-forward validation folds.
- `transformer/model.py` defines the PatchTST-style transformer model. Each maturity/delta surface node is treated as a time-series channel, historical values are split into temporal patches, and the encoded surface representation is fused with rate and market context features.
- `transformer/train.py` contains the training loop, validation loop, walk-forward validation routine, plotting utilities, and default training configuration.
- `transformer/results.json` stores the printed aggregate metrics from a previous validation run. Despite the `.json` extension, the current file is Python-style printed output rather than strict JSON.
- `artifacts/training_plots/` contains generated training-curve plots and predicted-vs-actual surface comparison plots for validation folds.

## Data

The training code expects an Excel workbook at:

```text
data/raw/all_data.xlsx
```

The workbook must include these columns:

```text
date
sp500_close
sp500_daily_simple_return
sp500_open
sp500_high
sp500_low
days_to_exp
delta
impl_volatility
risk_free_rate
vix_close
vix_open
vix_high
vix_low
```

To regenerate the dataset from WRDS:

```powershell
python data/data.py
```

This requires WRDS access and a working `wrds` Python configuration. The script currently collects SPX data from 2005 through 2025 and saves the result as `data/raw/all_data.xlsx`.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the Python dependencies:

```powershell
pip install numpy pandas matplotlib torch openpyxl wrds
```

If you plan to train on a GPU, install a CUDA-enabled PyTorch build that matches your machine. The default training configuration requests CUDA.

## Running the Model

Run the full walk-forward validation:

```powershell
python transformer/train.py
```

The default configuration in `transformer/train.py` uses:

- 252 trading days of lookback history
- 32 samples per batch
- 15% of the data reserved for the final test period
- 90-day validation windows
- 90-day walk-forward steps
- 25 maximum epochs per fold
- early stopping after 5 epochs without validation improvement
- CUDA as the default device

Training prints fold-by-fold loss, MAE, and RMSE values. It also writes plots to:

```text
artifacts/training_plots/
```

Use `device='cpu'` or `device='auto'` if CUDA is not available.

## Current Results

A previous walk-forward validation run produced 47 validation folds with approximately:

- mean validation loss: `1.2618`
- mean validation MAE: `0.0061`
- mean validation RMSE: `0.0099`

The saved fold plots compare training/validation curves and visualize predicted versus actual implied volatility surfaces.