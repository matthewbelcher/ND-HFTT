#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LISTENER_PORT="${LISTENER_PORT:-9000}"
LISTENER_BIND_HOST="${LISTENER_BIND_HOST:-0.0.0.0}"
MAX_LIFETIME_SECONDS="${COMPONENT_MAX_LIFETIME_SECONDS:-600}"
RESTART_DELAY_SECONDS="${COMPONENT_RESTART_DELAY_SECONDS:-5}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/test/.logs}"
PID_FILE="${PID_FILE:-$LOG_DIR/submit-benchmark-listener-${LISTENER_PORT}.pid}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/submit-benchmark-listener-${LISTENER_PORT}.log}"
SUPERVISOR_LOG="${SUPERVISOR_LOG:-$LOG_DIR/submit-benchmark-listener-${LISTENER_PORT}-supervisor.log}"
mkdir -p "$LOG_DIR"

while true; do
  printf [%s] starting benchmark listener on port %sn "$(date -Is)" "$LISTENER_PORT" >>"$SUPERVISOR_LOG"
  LISTENER_BIND_HOST="$LISTENER_BIND_HOST" \
  LISTENER_PORT="$LISTENER_PORT" \
  LOG_DIR="$LOG_DIR" \
  PID_FILE="$PID_FILE" \
  LOG_FILE="$LOG_FILE" \
  bash "$ROOT_DIR/test/run_submit_listener_ec2.sh" >>"$SUPERVISOR_LOG" 2>&1 || true

  sleep "$MAX_LIFETIME_SECONDS"

  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    printf [%s] stopping benchmark listener pid %s on port %s for token refreshn "$(date -Is)" "$(cat "$PID_FILE")" "$LISTENER_PORT" >>"$SUPERVISOR_LOG"
    kill "$(cat "$PID_FILE")" || true
    sleep 2
  fi
  rm -f "$PID_FILE"
  sleep "$RESTART_DELAY_SECONDS"
done
