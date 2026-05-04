# Matchmaker

Async process that scans open orders and executes YES/NO matches with atomic DB writes.

## Concurrency model

- Multiple matchmaker instances can run at once.
- A lightweight DB lease table (`matchmaker_market_leases`) gives one active matcher owner per market for a short lease window.
- Each trade execution is done in one transaction with conditional updates and OCC retry.
- Order selection uses price-time priority within each side (`price DESC`, then `global_seq ASC`).
- Candidate markets are scanned by oldest open order (`global_seq`) to keep backlog fairness.
- Self-trading is skipped (same account on both sides) and matching continues on remaining candidates.

This keeps single-instance behavior simple while allowing horizontal scaling without duplicate fills.

## Run

```bash
cd matchmaker
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export MATCHMAKER_DB_DSN='postgresql://<user>:<password>@<host>:5432/postgres?sslmode=require'
export MATCHMAKER_INSTANCE_ID='matchmaker-a'
python3 matchmaker.py
```

Useful settings:

- `MATCHMAKER_POLL_INTERVAL_MS` (default: `500`)
- `MATCHMAKER_MARKET_SCAN_LIMIT` (default: `50`)
- `MATCHMAKER_ORDER_SCAN_LIMIT` (default: `2000`)
- `MATCHMAKER_LEASE_SECONDS` (default: `5`)
