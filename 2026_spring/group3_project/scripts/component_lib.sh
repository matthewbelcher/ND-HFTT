#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PREDICTION_MARKET_VENV:-$ROOT_DIR/.venv-pm}"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"

require_aws_cli() {
  if [[ -x /usr/local/bin/aws ]]; then
    printf '%s\n' /usr/local/bin/aws
    return
  fi
  if command -v aws >/dev/null 2>&1; then
    command -v aws
    return
  fi
  echo 'aws CLI not found; install aws cli and retry.' >&2
  exit 1
}

ensure_venv() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    python3 -m venv "$VENV_DIR"
  fi

  "$PIP_BIN" install --quiet --upgrade pip >/dev/null
  for rel_req in "$@"; do
    if [[ -f "$ROOT_DIR/$rel_req" ]]; then
      "$PIP_BIN" install --quiet -r "$ROOT_DIR/$rel_req" >/dev/null
    fi
  done
}

refresh_dsql_dsn() {
  local target_var="$1"
  local host="$2"
  local region="$3"
  local aws_cli token enc_token dsn

  aws_cli="$(require_aws_cli)"
  token="$($aws_cli dsql generate-db-connect-admin-auth-token --hostname "$host" --region "$region" --output text)"
  enc_token="$(RAW_TOKEN="$token" "$PYTHON_BIN" -c 'import os, urllib.parse; print(urllib.parse.quote(os.environ["RAW_TOKEN"], safe=""))')"
  dsn="postgresql://admin:${enc_token}@${host}:5432/postgres?sslmode=require"
  printf -v "$target_var" '%s' "$dsn"
  export "$target_var"
}
