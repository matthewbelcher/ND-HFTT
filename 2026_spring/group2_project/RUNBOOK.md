# RUNBOOK: HFT-Bitcoin-Project Setup

Detailed setup steps for the Mullvad/Docker proxy and data collection pipeline.

## Respository Structure
```bash
HFT-BITCOIN-PROJECT/
├── README.md
├── requirements.txt
├── scripts/
│   ├── bitcoin_datacollector.py          # Coinbase WebSocket BTC trade collector
│   ├── polymarket_datacollector.py       # Polymarket REST + WebSocket collector
│   └── README.md                         # Data dictionary of all output files
├── mullvad-proxy/                        # Docker VPN proxy (cloned from GitHub)
├── src/
│   ├── features.py                       # Feature engineering + standalone CLI
│   ├── model.py                          # BaseModel and per-family subclasses (LR, XGB, RF, SVM)
│   ├── backtest.py                       # Bankroll simulation + binomial significance test
│   ├── train.py                          # Offline training and evaluation entry point
│   └── pipeline.py                       # Live data -> features -> predict orchestrator
└── Data/                                 # CSV output files (not tracked by git)
    ├── BitcoinData/
    │   └── btcusd_trades_*.csv
    └── PolymarketData/
        ├── polymarket_btc5m_history_*.csv
        ├── polymarket_btc15m_history_*.csv
        ├── polymarket_btc5m_trades_*.csv
        ├── polymarket_btc15m_trades_*.csv
        ├── polymarket_btc5_15m_live_*.csv
        └── polymarket_btc5_15m_orderbook_*.csv
```

> **Note:** The `Data/` directory is excluded from version control via `.gitignore`
> because the files are large. See the **Step 6: start data collection** section below to regenerate them.

## Setup and Reproduction

### Prerequisites

- Ubuntu 22.04 or 24.04 server (1GB+ RAM)
- Python 3.10+
- Docker
- A Mullvad VPN account number (16-digit number from mullvad.net)
- tmux

### Step 1: Install dependencies

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
```

### Step 2: Clone this repository

```bash
git clone https://github.com/lmolina32/HFT-Bitcoin-Project.git
cd HFT-Bitcoin-Project
mkdir -p Data/BitcoinData Data/PolymarketData

# Install Python packages
pip install -r requirements.txt
```

### Step 3: Set up the Mullvad Docker proxy

```bash
# Clone the proxy stack
git clone https://github.com/bernardko/mullvad-proxy.git
cd mullvad-proxy

# Configure your account number
cp .env.example .env
vi .env

# Build and start the containers
./setup.sh
```

### Step 4: Connect to a non-US Mullvad server

```bash
# Log in
docker exec -it mvpn mullvad account login YOUR_ACCOUNT_NUMBER

# Set exit to Canada (do not use US, Netherlands, France, or Germany)
docker exec -it mvpn mullvad relay set location ca tor

# Connect
docker exec -it mvpn mullvad connect

# Verify
docker exec -it mvpn mullvad status
```
### Step 5: Verify the proxy is unblocked

```bash
# Should show a Canadian IP
curl --socks5-hostname 127.0.0.1:1080 https://am.i.mullvad.net

# Should return {"blocked": false, "country": "CA"}
curl --socks5-hostname 127.0.0.1:1080 https://polymarket.com/api/geoblock
```

If `blocked` is still `true`, try a different relay:

```bash
docker exec -it mvpn mullvad relay set location jp tok
docker exec -it mvpn mullvad disconnect && docker exec -it mvpn mullvad connect
```

### Step 6: Start data collection

Run both collectors in separate tmux sessions so they survive SSH disconnection.

```bash
# Bitcoin spot data from Coinbase (no proxy needed — Coinbase is not geoblocked)
tmux new -s btc
python scripts/bitcoin_datacollector.py --output-dir Data/BitcoinData
# Detach: Ctrl+B then D
```

```bash
# Polymarket contract data (routes through Mullvad proxy)
tmux new -s polymarket
python scripts/polymarket_datacollector.py \
  --interval all \
  --proxy socks5://127.0.0.1:1080 \
  --lookback 30 \
  --live \
  --poll-interval 10 \
  --output-dir Data/PolymarketData
# Detach: Ctrl+B then D
```

### Step 7: Engineer features (optional standalone step)

Convert a raw Coinbase trade CSV into the engineered feature matrix
(one row per 5-minute bucket, 18 features + the `next_up_down` target).

```bash
python src/features.py \
  --data Data/BitcoinData/btcusd_trades_YYYYMMDD_HHMMSS.csv \
  --output Data/features.csv
```

Omit `--output` to compute the features without writing them to disk.

### Step 8: Train, evaluate, and backtest the models

`src/train.py` runs the full offline pipeline: load -> feature engineering
-> chronological 60/20/20 split -> per-model feature subset search on the
validation slice -> refit on train+val -> backtest each model on the
held-out test slice.

Quickest run (skips SVM, charts open interactively, models are not saved):

```bash
python src/train.py \
  --data Data/BitcoinData/btcusd_trades_YYYYMMDD_HHMMSS.csv \
  --skip-svm
```

Full run with persisted artefacts:

```bash
python src/train.py \
  --data Data/BitcoinData/btcusd_trades_YYYYMMDD_HHMMSS.csv \
  --output-dir runs/2026-05-05 \
  --strategies flat long_only confidence \
  --bankroll 1000 \
  --bet-fraction 0.10
```

The output directory is populated with:

- `model_comparison.png`, `feature_importance_*.png`, `bankroll_*.png`
- `models/<ClassName>/` for each trained model (estimator, scaler, metadata)
- `backtest_summary.json` with the per-strategy results table

Resume from previously-saved models without retraining:

```bash
python src/train.py \
  --data Data/BitcoinData/btcusd_trades_YYYYMMDD_HHMMSS.csv \
  --output-dir runs/2026-05-05 \
  --load-from runs/2026-05-05/models
```

Other useful flags:

| Flag             | Default | Purpose                                          |
| ---------------- | ------- | ------------------------------------------------ |
| `--train-frac`   | 0.60    | Fraction of windows used for training            |
| `--val-frac`     | 0.20    | Fraction used for validation (0 disables search) |
| `--skip-svm`     | off     | Skip the slowest model                           |
| `--strategies`   | flat long_only | Backtest strategies to evaluate           |

### Step 9: Run the live prediction pipeline

`src/pipeline.py` chains data discovery -> feature engineering ->
model load (or fresh train) -> backtest -> live prediction.

Train a fresh model on the most recent CSV in a directory and save it:

```bash
python src/pipeline.py \
  --data-dir Data/BitcoinData \
  --model-dir runs/live/XGBoostModel \
  --model-cls xgb \
  --train
```

Load a previously-saved model and emit a one-shot prediction for the
latest fully-formed 5-minute bucket:

```bash
python src/pipeline.py \
  --data-dir Data/BitcoinData \
  --model-dir runs/live/XGBoostModel
```

Add `--backtest` to also run a held-out backtest against the active CSV,
or `--watch` to poll the data directory and emit a fresh prediction
every 5 minutes:

```bash
python src/pipeline.py \
  --data-dir Data/BitcoinData \
  --model-dir runs/live/XGBoostModel \
  --watch \
  --poll-seconds 300
```

`--model-cls` accepts `lr`, `xgb`, `rf`, or `svm` (used only with `--train`).

### Programmatic use

The same building blocks can be imported and composed in a notebook
or another script:

```python
from features import build_features_from_path
from model    import XGBoostModel, make_split, refit_on_train_and_val
from backtest import backtest_model

data = build_features_from_path("Data/BitcoinData/btcusd_trades_*.csv")
split = make_split(data, train_frac=0.60, val_frac=0.20)

model = XGBoostModel()
model.train(split.X_train, split.y_train, X_val=split.X_val, y_val=split.y_val)
refit_on_train_and_val(model, data)
model.save("runs/live/XGBoostModel")

result = backtest_model(model, split.X_test, split.y_test, strategy="long_only")
print(result.summary())
```

## Requirements

See `requirements.txt`


## References

- [Polymarket API documentation](https://docs.polymarket.com)
- [Chainlink BTC/USD data stream](https://data.chain.link/streams/btc-usd-cexprice-streams)
- [Mullvad VPN](https://mullvad.net)
- [mullvad-proxy Docker setup](https://github.com/bernardko/mullvad-proxy)
- [Coinbase Advanced Trade API](https://docs.cdp.coinbase.com/advanced-trade/docs/ws-overview)

## License

For academic use only. This project does not constitute financial advice.
