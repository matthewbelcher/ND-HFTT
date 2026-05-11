#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/component_lib.sh"

AWS_REGION="${AWS_REGION:-us-east-2}"
NODES="${NODES:-1,2,3,4}"
CLIENTS="${CLIENTS:-1,2,4,8}"
PAIRS_PER_CLIENT="${PAIRS_PER_CLIENT:-50}"
CLIENT_RATE="${CLIENT_RATE:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/test/results}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-fill-node-scale}"
HEALTH_WAIT_TIMEOUT_SECONDS="${HEALTH_WAIT_TIMEOUT_SECONDS:-300}"
HEALTH_POLL_SECONDS="${HEALTH_POLL_SECONDS:-10}"
FILL_TIMEOUT_SECONDS="${FILL_TIMEOUT_SECONDS:-10}"
POLL_INTERVAL_MS="${POLL_INTERVAL_MS:-50}"
GENERATE_PLOT="${GENERATE_PLOT:-1}"

: "${DSQL_HOST:?DSQL_HOST must be set}"
: "${TG_ARN:?TG_ARN must be set}"
: "${NLB_DNS:?NLB_DNS must be set}"

mkdir -p "$OUTPUT_DIR"
ensure_venv test/requirements.txt client/client-listener/requirements.txt
refresh_dsql_dsn DB_DSN "$DSQL_HOST" "$AWS_REGION"

split_csv() {
  local raw="$1"
  local var_name="$2"
  local old_ifs="$IFS"
  local part
  IFS=','
  eval "$var_name=()"
  for part in $raw; do
    part="$(echo "$part" | tr -d '[:space:]')"
    [[ -n "$part" ]] && eval "$var_name+=(\"\$part\")"
  done
  IFS="$old_ifs"
}

target_args() {
  local id
  local args=()
  for id in "$@"; do
    [[ -n "$id" ]] && args+=("Id=$id")
  done
  printf '%s\n' "${args[@]}"
}

discover_listener_ids() {
  aws ec2 describe-instances \
    --region "$AWS_REGION" \
    --filters "Name=tag:role,Values=listener" "Name=instance-state-name,Values=running" \
    --query 'Reservations[].Instances[] | sort_by(@, &LaunchTime)[].InstanceId' \
    --output text
}

if [[ -n "${LISTENER_INSTANCE_IDS:-}" ]]; then
  split_csv "$LISTENER_INSTANCE_IDS" ALL_LISTENERS
else
  ALL_LISTENERS=()
  while IFS= read -r listener_id; do
    [[ -n "$listener_id" ]] && ALL_LISTENERS+=("$listener_id")
  done < <(discover_listener_ids | tr '\t' '\n' | sed '/^$/d')
fi

if [[ "${#ALL_LISTENERS[@]}" -eq 0 ]]; then
  echo "No listener instances found. Set LISTENER_INSTANCE_IDS=i-...,i-... or tag listeners with role=listener." >&2
  exit 1
fi

split_csv "$NODES" NODE_COUNTS

echo "Listeners discovered: ${ALL_LISTENERS[*]}"
echo "Node counts: ${NODE_COUNTS[*]}"
echo "Clients: $CLIENTS"
echo "Output dir: $OUTPUT_DIR"

wait_for_healthy() {
  local selected=("$@")
  local deadline=$((SECONDS + HEALTH_WAIT_TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    local healthy_count
    healthy_count="$(
      {
      aws elbv2 describe-target-health \
        --target-group-arn "$TG_ARN" \
        --region "$AWS_REGION" \
        --query 'TargetHealthDescriptions[?TargetHealth.State==`healthy`].Target.Id' \
        --output text | tr '\t' '\n' | grep -Fxf <(printf '%s\n' "${selected[@]}") || true
      } | wc -l | tr -d ' '
    )"
    if [[ "$healthy_count" == "${#selected[@]}" ]]; then
      return 0
    fi
    aws elbv2 describe-target-health \
      --target-group-arn "$TG_ARN" \
      --region "$AWS_REGION" \
      --query 'TargetHealthDescriptions[].{id:Target.Id,state:TargetHealth.State,reason:TargetHealth.Reason}' \
      --output table
    sleep "$HEALTH_POLL_SECONDS"
  done
  echo "Timed out waiting for selected targets to become healthy: ${selected[*]}" >&2
  return 1
}

run_for_node_count() {
  local n="$1"
  if (( n < 1 || n > ${#ALL_LISTENERS[@]} )); then
    echo "Skipping node count $n; available listener instances: ${#ALL_LISTENERS[@]}" >&2
    return 0
  fi

  local selected=("${ALL_LISTENERS[@]:0:n}")
  local dereg_args=()
  while IFS= read -r arg; do
    [[ -n "$arg" ]] && dereg_args+=("$arg")
  done < <(target_args "${ALL_LISTENERS[@]}")

  local reg_args=()
  while IFS= read -r arg; do
    [[ -n "$arg" ]] && reg_args+=("$arg")
  done < <(target_args "${selected[@]}")

  echo
  echo "=== Configuring NLB for $n listener node(s): ${selected[*]} ==="
  aws elbv2 deregister-targets \
    --target-group-arn "$TG_ARN" \
    --targets "${dereg_args[@]}" \
    --region "$AWS_REGION" || true
  sleep 5
  aws elbv2 register-targets \
    --target-group-arn "$TG_ARN" \
    --targets "${reg_args[@]}" \
    --region "$AWS_REGION"

  wait_for_healthy "${selected[@]}"

  local out_csv="$OUTPUT_DIR/${OUTPUT_PREFIX}-${n}-listeners.csv"
  echo "=== Running fill benchmark for $n listener node(s); CSV: $out_csv ==="
  "$ROOT_DIR/.venv-pm/bin/python" "$ROOT_DIR/test/benchmark_fill_latency.py" \
    --db-dsn "$DB_DSN" \
    --auth-token "${AUTH_TOKEN:-${TOKEN:-dev-shared-token}}" \
    --listener-host "$NLB_DNS" \
    --listener-port 80 \
    --clients "$CLIENTS" \
    --pairs-per-client "$PAIRS_PER_CLIENT" \
    --client-rate "$CLIENT_RATE" \
    --fill-timeout-seconds "$FILL_TIMEOUT_SECONDS" \
    --poll-interval-ms "$POLL_INTERVAL_MS" \
    --output-csv "$out_csv"
}

for n in "${NODE_COUNTS[@]}"; do
  n="$(echo "$n" | tr -d '[:space:]')"
  [[ -n "$n" ]] && run_for_node_count "$n"
done

if [[ "$GENERATE_PLOT" == "1" ]]; then
  OUTPUT_DIR="$OUTPUT_DIR" OUTPUT_PREFIX="$OUTPUT_PREFIX" "$ROOT_DIR/.venv-pm/bin/python" - <<'PY'
import csv
import os
import subprocess
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "matplotlib"])
    import matplotlib.pyplot as plt

output_dir = Path(os.environ.get("OUTPUT_DIR", "test/results"))
prefix = os.environ.get("OUTPUT_PREFIX", "fill-node-scale")
paths = sorted(output_dir.glob(f"{prefix}-*-listeners.csv"))

if not paths:
    print("no CSV files found to plot")
    raise SystemExit(0)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

for path in paths:
    node_count = path.stem.split("-")[-2]
    rows = list(csv.DictReader(path.open()))
    clients = [int(r["clients"]) for r in rows]
    throughput = [float(r["fill_throughput_pairs_sec"]) for r in rows]
    p95 = [float(r["p95_fill_latency_ms"]) for r in rows]
    label = f"{node_count} listener" + ("" if node_count == "1" else "s")
    ax1.plot(clients, throughput, marker="o", label=label)
    ax2.plot(clients, p95, marker="o", label=label)

ax1.set_title("Fill Throughput vs Concurrent Clients")
ax1.set_ylabel("Filled pairs/sec")
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.set_title("Submit-to-Fill P95 Latency vs Concurrent Clients")
ax2.set_xlabel("Concurrent clients")
ax2.set_ylabel("P95 fill latency (ms)")
ax2.set_yscale("log")
ax2.grid(True, alpha=0.3)
ax2.legend()

out = output_dir / f"{prefix}-summary.png"
fig.tight_layout()
fig.savefig(out, dpi=160)
print(f"wrote {out}")
PY
fi

echo
echo "Done. CSVs and plot are in $OUTPUT_DIR"
