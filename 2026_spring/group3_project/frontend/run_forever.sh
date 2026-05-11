#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/tmp/prediction-market-logs"
LOG_FILE="$LOG_DIR/frontend.log"
MAX_LIFETIME_SECONDS="${COMPONENT_MAX_LIFETIME_SECONDS:-600}"
mkdir -p "$LOG_DIR"

while true; do
  printf '[%s] starting frontend\n' "$(date -Is)" >>"$LOG_FILE"
  set +e
  if command -v timeout >/dev/null 2>&1; then
    timeout --foreground "$MAX_LIFETIME_SECONDS" "$SCRIPT_DIR/run_component.sh" >>"$LOG_FILE" 2>&1
  else
    "$SCRIPT_DIR/run_component.sh" >>"$LOG_FILE" 2>&1
  fi
  status=$?
  set -e
  if [[ "$status" == "124" ]]; then
    printf '[%s] frontend reached max lifetime (%ss), restarting for token refresh\n' "$(date -Is)" "$MAX_LIFETIME_SECONDS" >>"$LOG_FILE"
  elif [[ "$status" != "0" ]]; then
    printf '[%s] frontend exited with status %s, restarting in 5 seconds\n' "$(date -Is)" "$status" >>"$LOG_FILE"
  else
    printf '[%s] frontend exited cleanly, restarting in 5 seconds\n' "$(date -Is)" >>"$LOG_FILE"
  fi
  sleep 5
done
