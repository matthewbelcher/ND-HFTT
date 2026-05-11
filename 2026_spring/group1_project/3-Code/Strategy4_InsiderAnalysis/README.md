# Kalshi Culture/Event Flow Analyzer

Lean, testable toolkit for collecting Kalshi event/market data, detecting aggressive YES flow, and evaluating whether it correlates with outcomes. Focused on culture/entertainment events (reality TV, awards, etc.), not sports.

## Quick Start

1) Create a tickers file:

```text
# comments allowed
KXSURVIVOR-26DEC31 # reality
```

2) Run collection (polls until interrupted):

```bash
kalshi-analyzer collect --tickers tickers.txt --db data.sqlite --poll-minutes 10
```

3) Generate a report:

```bash
kalshi-analyzer report --db data.sqlite --top 10
kalshi-analyzer report --db data.sqlite --event KXSURVIVOR-26DEC31
```

4) Update outcomes (after markets resolve):

```bash
kalshi-analyzer resolve --db data.sqlite
```

5) Recompute scores from stored data:

```bash
kalshi-analyzer replay --db data.sqlite --rules-config config.yaml
```

6) Export:

```bash
kalshi-analyzer export --db data.sqlite --format csv --out exports/
```

## Configuration

All thresholds and polling settings are configurable via YAML.

```yaml
polling:
  poll_minutes: 10
  adaptive: false
  adaptive_min_minutes: 2
  adaptive_max_minutes: 15
  adaptive_trade_burst_count: 5
  adaptive_window_minutes: 30

aggressive:
  price_tolerance: 0.005
  burst_window_minutes: 60
  burst_trade_count: 3
  burst_contracts: 200

large_trade:
  absolute_contracts: 500
  pct_trailing_volume: 0.20
  pct_top_book: 0.30
  trailing_volume_window: 7d

step_change:
  step_change_dollars: 0.10
  step_window_hours: 24
  step_aggressive_contracts: 200

favorite:
  favorite_high: 0.65
  favorite_min_days: 7

scoring:
  score_threshold_flag: 70
  weights:
    aggressive_burst: 30
    large_aggressive: 25
    step_change: 25
    sustained_favorite: 20

backoff:
  max_retries: 5
  base_seconds: 1
  max_seconds: 30

api:
  data_api_base: https://demo-api.kalshi.co
  trade_api_base: https://demo-api.kalshi.co

auth:
  key_id: "${KALSHI_KEY_ID}"
  private_key_pem: "${KALSHI_PRIVATE_KEY_PEM}"
```

## Environment Variables

- `KALSHI_KEY_ID`
- `KALSHI_PRIVATE_KEY_PEM` (path to PEM)
- Optional trade-specific:
  - `KALSHI_TRADE_KEY_ID`
  - `KALSHI_TRADE_PRIVATE_KEY_PEM`
- Base URLs:
  - `KALSHI_DATA_API_BASE`
  - `KALSHI_TRADE_API_BASE`
  - `KALSHI_BASE_URL` (fallback)

## Notes

- Orderbooks return bids only; asks are inferred from the opposite side.
- If trades are unavailable, the collector infers a trade from volume deltas and flags it as inferred.
- For older data, use `--historical` to query historical endpoints when available.

## Development

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .
pytest
```

