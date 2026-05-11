# Executor

Distributed settlement worker that processes resolved markets at/after `resolve_time`.

## Concurrency model

- Multiple executor instances can run at once.
- `execution_market_leases` provides short market ownership leases.
- `market_settlement_runs` tracks in-progress/completed settlement status and prevents double payout.
- Settlement writes are transactional and idempotent via `market_settlements (market_id, account_id)` uniqueness.

## Settlement behavior

- Candidate market: `markets.status='RESOLVED'` and `resolve_time <= now` (or null), with a row in `market_resolutions`.
- Outcome payout per account uses settled shares in `positions`:
  - `YES`: `(yes_shares + locked_yes_shares) * 100`
  - `NO`: `(no_shares + locked_no_shares) * 100`
  - `CANCELLED`: project approximation `(total_shares) * 50`
- On settlement, positions are zeroed and `ledger_entries` + `market_settlements` are inserted.

## Run

```bash
cd executor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export EXECUTOR_DB_DSN='postgresql://<user>:<password>@<host>:5432/postgres?sslmode=require'
export EXECUTOR_INSTANCE_ID='executor-a'
python3 executor.py
```

Useful settings:

- `EXECUTOR_POLL_INTERVAL_MS` (default: `1000`)
- `EXECUTOR_MARKET_SCAN_LIMIT` (default: `50`)
- `EXECUTOR_LEASE_SECONDS` (default: `10`)
