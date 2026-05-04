#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$ROOT_DIR/scripts/component_lib.sh"

ensure_venv
exec "$PYTHON_BIN" "$SCRIPT_DIR/order_client.py" "$@"
