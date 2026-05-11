#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/component_lib.sh"

HOST="${DSQL_HOST:-4btv7qq43k4ztw3tjubmbyf3su.dsql.us-east-2.on.aws}"
REGION="${AWS_REGION:-us-east-2}"
AUTH_TOKEN="${AUTH_TOKEN:-dev-shared-token}"
LISTENER_BIND_HOST="${LISTENER_BIND_HOST:-0.0.0.0}"
LISTENER_PORT="${LISTENER_PORT:-9501}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/test/.logs}"
PID_FILE="${PID_FILE:-$LOG_DIR/submit-benchmark-listener.pid}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/submit-benchmark-listener.log}"

ensure_venv client/client-listener/requirements.txt
refresh_dsql_dsn DB_DSN "$HOST" "$REGION"
mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "listener already running with pid $(cat "$PID_FILE")"
  exit 0
fi

(
  cd "$ROOT_DIR/client/client-listener"
  env \
    CLIENT_LISTENER_HOST="$LISTENER_BIND_HOST" \
    CLIENT_LISTENER_PORT="$LISTENER_PORT" \
    CLIENT_LISTENER_DB_DSN="$DB_DSN" \
    CLIENT_LISTENER_AUTH_MODE="static-token" \
    CLIENT_LISTENER_AUTH_TOKEN="$AUTH_TOKEN" \
    "$ROOT_DIR/.venv-pm/bin/python" server.py
) >"$LOG_FILE" 2>&1 &

echo $! >"$PID_FILE"
echo "listener started pid=$(cat "$PID_FILE") host=$LISTENER_BIND_HOST port=$LISTENER_PORT log=$LOG_FILE"
