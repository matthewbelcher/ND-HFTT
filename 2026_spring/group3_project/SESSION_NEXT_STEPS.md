# Future Session Handoff

## Current State (as of this handoff)
- Authoritative schema: `tables-create-dsql.sql`.
- Core distributed services are active: listener, matchmaker, executor.
- Listener now supports full RPC surface for order lifecycle, balances, markets, and reads.
- End-to-end test scripts are in place and pass via:
  - `test/system_test.py`
  - `test/rpc_features_test.py`
  - `test/run_system_test_ec2.sh`

## Primary Next Goal: Advanced Client
Build a significantly more capable client layer on top of current RPCs.

### 1. Client SDK + typed models (highest priority)
Create a reusable client package (`client/sdk`) with:
- typed request/response models,
- retries/backoff for transient failures,
- idempotency helper utilities,
- pagination helpers for list endpoints,
- shared auth/session configuration.

### 2. Interactive trading CLI
Add a real operator CLI (`client/cli.py`) with commands for:
- market discovery, order book view, open orders view,
- submit/cancel/replace/cancel-all,
- balances/positions/trades,
- deposit/withdraw,
- market admin actions (for privileged roles).

### 3. Streaming client view
Add near-real-time polling/stream mode in client:
- watch order status transitions,
- watch top-of-book changes,
- watch fills and position changes.

### 4. Client-side safety rails
Implement preflight checks before submit/replace:
- price/tick-size validation,
- market status validation cache,
- balance sufficiency hints,
- clear error mapping to actionable messages.

## Additional Required Features (System)

### 1. IOC/FOK enforcement (matching semantics)
- Implement actual IOC/FOK behavior in order lifecycle.
- Extend tests with explicit IOC/FOK correctness cases.

### 2. Invariants checker
Add `test/invariants_check.py` and run after suites:
- non-negative account/position invariants,
- order-state consistency,
- ledger consistency checks,
- settlement uniqueness and conservation checks.

### 3. Observability baseline
- structured JSON logs for each service,
- counters/timers for ingest/match/settle throughput,
- error rate and OCC retry metrics.

### 4. Deployment hardening
- systemd units per service,
- restart policy and environment templates,
- health/readiness probes and startup ordering.

### 5. Security hardening
- default deployed auth mode to Secrets Manager,
- least-privilege IAM policies by service role,
- auth token rotation and operational runbook.

### 6. Performance testing
- stress/soak test suite with higher concurrency,
- benchmark report for throughput, p95/p99 latency, retry rates.

## Suggested Execution Order
1. Client SDK + CLI
2. IOC/FOK + invariants checker
3. Observability and deployment hardening
4. Security hardening
5. Stress/soak performance validation

## Success Criteria for Next Session
- Advanced client can execute full lifecycle (submit/cancel/replace/portfolio views) cleanly.
- IOC/FOK tests pass.
- Invariants checker passes after both existing test suites.
- Basic metrics/logging and deployment units are checked in.
