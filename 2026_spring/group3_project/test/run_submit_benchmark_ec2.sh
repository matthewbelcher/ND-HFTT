#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/component_lib.sh"

HOST="${DSQL_HOST:-4btv7qq43k4ztw3tjubmbyf3su.dsql.us-east-2.on.aws}"
REGION="${AWS_REGION:-us-east-2}"
AUTH_TOKEN="${AUTH_TOKEN:-dev-shared-token}"
CLIENTS="${CLIENTS:-1,2,4,8,16,32,48,64,96,128}"
ORDERS_PER_CLIENT="${ORDERS_PER_CLIENT:-1000}"
CLIENT_RATE="${CLIENT_RATE:-1000}"
LISTENER_HOST="${LISTENER_HOST:-127.0.0.1}"
LISTENER_PORT="${LISTENER_PORT:-9501}"
START_LISTENER="${START_LISTENER:-0}"
OUTPUT_CSV="${OUTPUT_CSV:-$ROOT_DIR/test/results/submit-scaling-$(date +%Y%m%d-%H%M%S).csv}"

ensure_venv test/requirements.txt client/client-listener/requirements.txt
refresh_dsql_dsn DB_DSN "$HOST" "$REGION"

PY_ARGS=(
  --db-dsn "$DB_DSN"
  --auth-token "$AUTH_TOKEN"
  --listener-host "$LISTENER_HOST"
  --listener-port "$LISTENER_PORT"
  --clients "$CLIENTS"
  --orders-per-client "$ORDERS_PER_CLIENT"
  --client-rate "$CLIENT_RATE"
  --output-csv "$OUTPUT_CSV"
)

if [[ "$START_LISTENER" == "1" ]]; then
  PY_ARGS+=(--start-listener)
fi

"$ROOT_DIR/.venv-pm/bin/python" "$ROOT_DIR/test/benchmark_submit_scaling.py" "${PY_ARGS[@]}"

printf 'CSV written to %s\n' "$OUTPUT_CSV"
