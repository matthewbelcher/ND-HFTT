# Signal Bazaar: Distributed Prediction Market Exchange

**Team 3:** Matt Daly, Swindar Zhou, Mariam Jafri
**Course:** CSE 40438 — High-Frequency Trading Technologies (Spring 2026)
**Instructor:** Prof. Matthew Belcher, University of Notre Dame

## Project Overview

Signal Bazaar is a distributed prediction-market exchange built on AWS. Authenticated clients submit price-time-priority limit orders over a TCP JSON-line RPC to listener nodes; matchmaker workers cross YES/NO orders and persist trades; executor workers settle resolved markets. All state lives in Amazon Aurora DSQL (serverless distributed SQL with optimistic concurrency control). A browser "Signal Bazaar Terminal" frontend provides market tape, depth, trade ticket, wallet, and admin lifecycle controls.

The project explores the throughput / fairness tradeoff for a cloud-native prediction-market venue and quantifies the DSQL write-contention ceiling under load.

## Repository Layout

- `Signal_Bazaar_HFT_Report.pdf` — Final 17-page report
- `tables-create-dsql.sql` — Authoritative Aurora DSQL schema (single source of truth)
- `tables-create.sql` — Reference schema for non-DSQL deployments
- `start_services.sh` — Supervisor-style launcher
- `trading_terminal.png` — Frontend demo screenshot
- `client/`
  - `client-listener/` — TCP JSON-line RPC server (auth, validation, collateral lock, DB write)
  - `order_client.py` — Basic CLI order submitter
- `matchmaker/` — Distributed matching worker with DB-backed leases
- `executor/` — Distributed settlement worker (idempotent payouts)
- `frontend/` — SPA terminal (server + static assets)
- `deploy/` — EC2 / NLB / multi-listener topology and runbooks (`DEPLOY.md`, systemd units, bootstrap scripts)
- `test/` — End-to-end concurrency tests, RPC surface tests, benchmarks, and results

## Build / Run

Per-component instructions live in each subdirectory's README. Quick start (after database is provisioned):

```bash
# Apply schema
deploy/apply_schema.sh

# Start all services
./start_services.sh
```

EC2 / NLB topology is documented in `deploy/DEPLOY.md`.

## Tests

```bash
./test/run_system_test_ec2.sh
```

Bootstraps the Python venv, generates an Aurora DSQL IAM auth token, and runs `system_test.py` + `rpc_features_test.py`.

## Authors

Matt Daly, Swindar Zhou, Mariam Jafri
