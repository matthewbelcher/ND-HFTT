# Statistical Arbitrage Backtester

A Python-based framework for exploring and backtesting simple statistical‑arbitrage strategies on tick‑level market data.  
It leverages **Dask** for scalable data loading/preprocessing, **statsmodels** for hedge‑ratio estimation, and **Optuna** for hyperparameter optimization.

This code is generalized and can take in any number of symbols, we are focused on HD and LOW for this project.

---

## 📁 Project Structure

```
hft_project/
├── data_loader.py        # load + preprocess raw CSV ↔ Parquet cache
├── pair_selector.py      # compute correlations & select cointegrated pairs
├── stat_arb_strat.py     # estimate hedge ratio, generate signals, backtest returns
├── optuna_optimizer.py   # tune entry/exit z‑scores & window via Optuna
├── main.py               # orchestrates end‑to‑end workflow
├── requirements.txt      # pip dependencies
└── README.md             # this file
```

---

## 🛠️ Installation

1. **Clone the repo**  
   ```bash
   git clone <your-repo-url>
   cd hft_project
   ```

2. **Create a Python environment** (optional but recommended)  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .venv\Scripts\activate    # Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

## 🔧 Configuration

- **Date range**  
  This is already done and preprocessed so the csv file isnt up on the repo, only the parquets.
  In `load_and_preprocess()`, adjust `start_date`/`end_date` or pass new defaults:
  ```python
  ddf = load_and_preprocess(
      filepath,
      start_date='2023-01-01',
      end_date='2023-12-31',
      parquet_path='data.parquet'
  )
  ```

- **Correlation threshold / window size**  
  - In `PairSelector(threshold=…)` you can change the minimum |ρ|.  
  - In `generate_signals(entry_z, exit_z, window)` and in `optuna_optimizer.py`, tune your own defaults or let Optuna search.

- **Optuna trials**  
  In `main.py`:
  ```python
  best_params = optimizer.optimize(n_trials=100)
  ```
  Lower or raise `n_trials` to control search time.

---

## ▶️ Usage

Just run the orchestrator:

```bash
python main.py
```

Typical console output will walk you through:

1. Dask cluster startup  
2. Data loading & Parquet caching  
3. Top‑50 symbol selection  
4. Pair correlation & cointegration testing  
5. Optuna hyperparameter search  
6. Final trade count & PnL printout  

---

## 📦 Dependencies

See `requirements.txt`:

- **dask** ≥ 2022.2.0  
- **distributed** ≥ 2022.2.0  
- **numpy** ≥ 1.21.0  
- **pandas** ≥ 1.3.0  
- **statsmodels** ≥ 0.13.0  
- **optuna** ≥ 3.2.0  

---

## 📝 Notes

- Equity data comes from WRDS for HD and LOW
- This data was quite large, and was split into parquets for processing
- This code uses the same logic as our previous project (pre-presentation), though uses equity data rather than forex data
- Future work needs to be done to include visualization, more rigourous backtesting, and more optimized parameters for trading