# Kalshi Culture/Event Flow Analyzer — Design Spec (v0.1)

## Purpose
Measure and rank suspicious “aggressive YES buying” in culture/entertainment Kalshi events (e.g., reality TV winners) and build a durable local dataset for historical and live analysis.

## Key Assumptions About the Kalshi API
This implementation targets Kalshi’s `trade-api/v2` REST endpoints and assumes the following:

- **Event → markets expansion**
  - `GET /events/{event_ticker}?with_nested_markets=true` can return nested markets for an event.
  - `GET /markets?event_ticker=...` returns full market objects (including top-of-book fields and sizes).
- **Market metadata & resolution**
  - `GET /markets/{ticker}` includes resolution fields (e.g., `result`, settlement value) and status.
- **Order book**
  - `GET /markets/{ticker}/orderbook` returns **bids only** for YES and NO; asks are implied from the opposite side (YES ask = 1 − best NO bid; NO ask = 1 − best YES bid).
- **Trades / prints**
  - `GET /markets/trades` returns trade prints including `taker_side` and timestamps when available.
- **Historical access**
  - `GET /historical/cutoff` indicates the cutoff for historical data.
  - `GET /historical/markets` and `GET /historical/markets/{ticker}` can fetch older market metadata.
  - `GET /historical/fills` may provide older trade data (if supported for the market).
  - Live endpoints are expected to stop serving historical records around March 6, 2026, so the collector routes older queries to historical endpoints.
- **Auth**
  - API key signing uses a signature of `timestamp + method + path` (no query string) and is optional for public endpoints.
- **Price units**
  - Cent-denominated fields are deprecated; dollars fields (e.g., `yes_bid_dollars`) are used and stored as decimals.

If any endpoint is unavailable, the system falls back to whatever is present (e.g., market list without trades, or historical metadata without fills). Trades can be inferred from top‑of‑book changes when needed and labeled as inferred.

## Module Diagram & Data Flow
```
CLI
 ├── ticker_parser -> (event_tickers, tags)
 ├── kalshi_client -> REST fetch + retry/backoff + auth
 ├── collector      (poll loop)
 │    ├── event_expander -> markets
 │    ├── snapshots/trades -> storage
 │    └── features + rules -> scores
 ├── resolve        (update outcomes)
 ├── replay         (recompute features/scores)
 └── report/export  (summaries, CSV/parquet)

Storage (SQLite)
 ├── events, markets
 ├── snapshots, trades
 ├── features, scores
 └── outcomes, collector_state
```

## Defaults (Configurable)
Designed for long‑horizon culture events where “HFT” is unnecessary.

- **Polling interval**: 10 minutes (default) with optional adaptive range 2–15 minutes.
  - Reason: long‑horizon events have slow-moving orderbooks; 5–15 min captures changes without spamming the API.
- **Aggressive trade tolerance**: 0.5 cents (0.005 dollars) vs. BBO.
  - Reason: tolerate minor tick/rounding differences.
- **Large trade definition** (any condition triggers “large”):
  - Absolute contracts ≥ 500
  - OR ≥ 20% of trailing 7‑day volume
  - OR ≥ 30% of top‑of‑book liquidity (if sizes are available)
  - Reason: long events often trade sparsely; percent-of-trailing-volume better reflects “unusually large.”
- **Step‑change rule**: YES mid +10c within 24h **and** aggressive YES volume ≥ 200 in that window.
- **Sustained dominance**: Market is favorite with YES mid ≥ 0.65 for ≥ 7 days pre‑resolution.

All thresholds live in `config.yaml` and are overrideable by CLI.

## Schema Overview (SQLite)
**events**
- `event_ticker` (PK), `title`, `category`, `series_ticker`, `status`, `tags_json`, `raw_json`

**markets**
- `market_ticker` (PK), `event_ticker` (FK), `title`, `status`, `close_time`, `result`, `settlement_value`, `raw_json`

**snapshots** (BBO + market summary)
- PK: (`market_ticker`, `ts`, `snapshot_type`)
- Fields: `yes_bid`, `yes_ask`, `no_bid`, `no_ask`, `mid`, `spread`, `volume`, `open_interest`, `yes_bid_size`, `yes_ask_size`, `raw_json`

**trades**
- `trade_id` (PK), `market_ticker`, `ts`, `price`, `count`, `taker_side`, `is_inferred`, `raw_json`

**features**
- PK: (`market_ticker`, `ts`, `feature_set`)
- Fields: `features_json`

**scores**
- PK: (`market_ticker`, `ts`, `config_hash`)
- Fields: `score`, `rules_json`, `explanation`

**outcomes**
- `market_ticker` (PK), `result`, `settlement_value`, `resolved_ts`

**collector_state**
- `market_ticker` (PK), `last_trade_ts`, `last_snapshot_ts`, `last_cursor`

## Success Criteria
- Live mode can collect snapshots/trades and persist them idempotently.
- Replay mode can recompute scores from stored raw data.
- Report mode can rank suspicious markets/events and summarize post‑resolution accuracy.
- All thresholds are tunable via CLI + `config.yaml`.
