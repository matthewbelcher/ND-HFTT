# polymarket_btc_updown_collector.py
Fetches tick data (price history + individual trades + live stream) for
Polymarket BTC Up/Down contracts — both **5-minute** and **15-minute** windows.

## DATA SOURCES
  - Gamma API : https://gamma-api.polymarket.com   (market/event metadata)
  - CLOB  API : https://clob.polymarket.com         (price history, individual trades, orderbook)
  - Data  API : https://data-api.polymarket.com      (matched trade records)
  - WS        : wss://ws-subscriptions-clob.polymarket.com/ws/market  (live)


## CONTRACT INTERVALS
    --interval 5      5-minute BTC Up/Down contracts  (slug: btc-updown-5m-{ts})
    --interval 15     15-minute BTC Up/Down contracts  (slug: btc-updown-15m-{ts})
    --interval all    Fetch both 5m and 15m contracts

## SLUG FORMAT
  Polymarket uses deterministic slugs based on the Unix timestamp of the
  window start, rounded to the interval boundary:

    5-min  →  btc-updown-5m-{ts}   where ts = now - (now % 300)
    15-min →  btc-updown-15m-{ts}  where ts = now - (now % 900)

  The script can auto-generate slugs for recent windows or accept explicit
  slugs via --event-slug.

## USAGE
    # Auto-discover latest 15m contracts (default)
    python polymarket_btc_updown_collector.py

    # Fetch 5-minute contracts
    python polymarket_btc_updown_collector.py --interval 5

    # Fetch both 5m and 15m
    python polymarket_btc_updown_collector.py --interval all

    # With SOCKS5 proxy (DigitalOcean NYC → non-US exit)
    python polymarket_btc_updown_collector.py --proxy socks5://user:pass@nl-proxy:1080

    # With HTTP proxy
    python polymarket_btc_updown_collector.py --proxy http://proxy.example.com:8080

    # Explicit event slug
    python polymarket_btc_updown_collector.py --event-slug btc-updown-5m-1775263800

    # Stream live ticks + orderbook polling
    python polymarket_btc_updown_collector.py --interval 5 --live

    # Only fetch individual trades (skip aggregated price history)
    python polymarket_btc_updown_collector.py --trades-only --interval all

## OPTIONS
    --event-slug SLUG    Polymarket event slug. Defaults to auto-search.
    --interval {5,15,all}  Contract interval in minutes (default: 15)
    --output-dir DIR     Directory to write CSV files (default: current dir)
    --lookback N         Number of past contracts to fetch (default: 20)
    --live               After fetching history, stream live ticks via WebSocket
    --trades-only        Skip price-history; only fetch individual trade records
    --fidelity N         Price-history bucket size in minutes (default: 1)
    --trades-limit N     Max individual trades to fetch per token (default: 5000)
    --poll-interval N    Orderbook poll frequency in seconds when --live (default: 5)
    --proxy URL          SOCKS5 or HTTP(S) proxy URL for all API requests
    --check-geo          Test geoblock status before fetching data

## OUTPUT FILES
    polymarket_btc{interval}m_history_YYYYMMDD_HHMMSS.csv
    polymarket_btc{interval}m_trades_YYYYMMDD_HHMMSS.csv
    polymarket_btc{interval}m_live_YYYYMMDD_HHMMSS.csv       (--live only)
    polymarket_btc{interval}m_orderbook_YYYYMMDD_HHMMSS.csv  (--live only)

## Data Dictionary

### `polymarket_btc5m_trades_*.csv` and `polymarket_btc15m_trades_*.csv`

Individual matched trades from the Polymarket Data API. Each row is one CLOB fill.

| Column | Type | Description |
|---|---|---|
| `interval_minutes` | int | `5` or `15` |
| `event_slug` | string | e.g. `btc-updown-5m-1775334300` |
| `outcome` | string | `Up` or `Down` |
| `token_id` | string | 70-digit Polygon token identifier |
| `start_date` | datetime | Window open time |
| `end_date` | datetime | Window close time |
| `trade_id` | string | Unique trade identifier |
| `timestamp_utc` | datetime | Trade execution time |
| `price` | float | Trade price (0–1, equals probability) |
| `side` | string | `BUY` or `SELL` |
| `size` | float | Number of shares traded |
| `maker_address` | string | Maker wallet address (Polygon) |
| `taker_address` | string | Taker wallet address (Polygon) |

### `polymarket_btc5m_history_*.csv` and `polymarket_btc15m_history_*.csv`

Aggregated mid-point probability snapshots from the CLOB prices-history endpoint. 
One row per price change event (sparse — quiet markets have few rows).

| Column | Type | Description |
|---|---|---|
| `interval_minutes` | int | `5` or `15` |
| `event_slug` | string | Contract identifier slug |
| `outcome` | string | `Up` or `Down` |
| `token_id` | string | Token identifier |
| `timestamp_utc` | datetime | Snapshot time |
| `price` | float | Mid-point probability at this time |
| `status` | string | `active`, `closed`, or `resolved` |

### `polymarket_btc5_15m_orderbook_*.csv`

Orderbook snapshots polled every 10 seconds from the CLOB `/book` endpoint.

| Column | Type | Description |
|---|---|---|
| `timestamp_utc` | datetime | Snapshot time |
| `interval_minutes` | int | `5` or `15` |
| `event_slug` | string | Contract identifier slug |
| `outcome` | string | `Up` or `Down` |
| `token_id` | string | Token identifier |
| `best_bid` | float | Highest current bid price |
| `best_ask` | float | Lowest current ask price |
| `mid` | float | Midpoint `(best_bid + best_ask) / 2` |
| `spread` | float | `best_ask - best_bid` |

### `polymarket_btc5_15m_live_*.csv`

Real-time trade events from the Polymarket WebSocket stream.

| Column | Type | Description |
|---|---|---|
| `interval_minutes` | int | `5` or `15` |
| `event_slug` | string | Contract identifier slug |
| `outcome` | string | `Up` or `Down` |
| `token_id` | string | Token identifier |
| `timestamp_utc` | datetime | Event time |
| `event_type` | string | `trade`, `last_trade_price`, or `price_change` |
| `price` | float | Trade or price update value |
| `side` | string | `BUY` or `SELL` (trade events only) |
| `size` | float | Trade size (trade events only) |

---


# btcusdt_trade_collector.py
Collects tick-by-tick trade data from the Coinbase Advanced Trade
WebSocket API and writes it to a timestamped CSV file.

## USAGE
    python btcusdt_trade_collector.py [OPTIONS]

## OPTIONS
    --symbol SYMBOL        Trading pair symbol (default: BTC-USD)
    --output-dir DIR       Directory to write CSV files (default: current dir)
    --duration SECONDS     Stop after N seconds (default: run forever)
    --buffer-size N        Flush after N rows (default: 500)

## Data Dictionary

### `btcusd_trades_*.csv`

Individual BTC/USD trade records from the Coinbase Advanced Trade WebSocket API.

| Column | Type | Description |
|---|---|---|
| `trade_id` | string | Unique Coinbase trade identifier |
| `time` | datetime (UTC) | Trade execution timestamp |
| `price` | float | Trade price in USD |
| `size` | float | Trade size in BTC |
| `side` | string | `BUY` or `SELL` (taker side) |