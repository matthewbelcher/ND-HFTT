#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

AWS_REGION="${AWS_REGION:-us-east-2}"
AUTH_TOKEN="${AUTH_TOKEN:-${TOKEN:-dev-shared-token}}"
NODES="${NODES:-1,2,3}"
CLIENTS="${CLIENTS:-1,2,4,8,16,32,64}"
ORDERS_PER_CLIENT="${ORDERS_PER_CLIENT:-200}"
CLIENT_RATE="${CLIENT_RATE:-1000}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/test/results}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-node-scale}"
HEALTH_WAIT_TIMEOUT_SECONDS="${HEALTH_WAIT_TIMEOUT_SECONDS:-300}"
HEALTH_POLL_SECONDS="${HEALTH_POLL_SECONDS:-10}"
GENERATE_PLOT="${GENERATE_PLOT:-1}"

: "${DSQL_HOST:?DSQL_HOST must be set}"
: "${TG_ARN:?TG_ARN must be set}"
: "${NLB_DNS:?NLB_DNS must be set}"

mkdir -p "$OUTPUT_DIR"

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
  local dereg_args
  dereg_args=()
  while IFS= read -r arg; do
    [[ -n "$arg" ]] && dereg_args+=("$arg")
  done < <(target_args "${ALL_LISTENERS[@]}")
  local reg_args
  reg_args=()
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
  echo "=== Running benchmark for $n listener node(s); CSV: $out_csv ==="
  DSQL_HOST="$DSQL_HOST" \
  AWS_REGION="$AWS_REGION" \
  AUTH_TOKEN="$AUTH_TOKEN" \
  LISTENER_HOST="$NLB_DNS" \
  LISTENER_PORT=80 \
  CLIENTS="$CLIENTS" \
  ORDERS_PER_CLIENT="$ORDERS_PER_CLIENT" \
  CLIENT_RATE="$CLIENT_RATE" \
  OUTPUT_CSV="$out_csv" \
  "$ROOT_DIR/test/run_submit_benchmark_ec2.sh"
}

for n in "${NODE_COUNTS[@]}"; do
  n="$(echo "$n" | tr -d '[:space:]')"
  [[ -n "$n" ]] && run_for_node_count "$n"
done

if [[ "$GENERATE_PLOT" == "1" ]]; then
  "$ROOT_DIR/.venv-pm/bin/python" - <<'PY'
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
prefix = os.environ.get("OUTPUT_PREFIX", "node-scale")
paths = sorted(output_dir.glob(f"{prefix}-*-listeners.csv"))

if not paths:
    print("no CSV files found to plot")
    raise SystemExit(0)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

for path in paths:
    node_count = path.stem.split("-")[-2]
    rows = list(csv.DictReader(path.open()))
    clients = [int(r["clients"]) for r in rows]
    throughput = [float(r["avg_throughput_ops"]) for r in rows]
    p95 = [float(r["p95_latency_ms"]) for r in rows]
    label = f"{node_count} listener" + ("" if node_count == "1" else "s")
    ax1.plot(clients, throughput, marker="o", label=label)
    ax2.plot(clients, p95, marker="o", label=label)

ax1.set_title("Throughput vs Concurrent Clients")
ax1.set_ylabel("Successful orders/sec")
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.set_title("P95 Latency vs Concurrent Clients")
ax2.set_xlabel("Concurrent clients")
ax2.set_ylabel("P95 latency (ms)")
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
