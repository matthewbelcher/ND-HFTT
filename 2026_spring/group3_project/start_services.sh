#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/ubuntu/Distributed-Prediction-Market-Server"
LOG_DIR="/tmp/prediction-market-logs"
mkdir -p "$LOG_DIR"

pkill -f "$ROOT_DIR/client/client-listener/run_forever.sh" || true
pkill -f "$ROOT_DIR/client/client-listener/server.py" || true
pkill -f "$ROOT_DIR/matchmaker/run_forever.sh" || true
pkill -f "$ROOT_DIR/matchmaker/matchmaker.py" || true
pkill -f "$ROOT_DIR/executor/run_forever.sh" || true
pkill -f "$ROOT_DIR/executor/executor.py" || true
pkill -f "$ROOT_DIR/frontend/run_forever.sh" || true
pkill -f "$ROOT_DIR/frontend/server.py" || true

nohup "$ROOT_DIR/client/client-listener/run_forever.sh" >"$LOG_DIR/listener-supervisor.log" 2>&1 &
nohup "$ROOT_DIR/matchmaker/run_forever.sh" >"$LOG_DIR/matchmaker-supervisor.log" 2>&1 &
nohup "$ROOT_DIR/executor/run_forever.sh" >"$LOG_DIR/executor-supervisor.log" 2>&1 &
nohup "$ROOT_DIR/frontend/run_forever.sh" >"$LOG_DIR/frontend-supervisor.log" 2>&1 &

sleep 3
ps -ef | grep -E 'client-listener/run_forever.sh|client-listener/server.py|matchmaker/run_forever.sh|matchmaker/matchmaker.py|executor/run_forever.sh|executor/executor.py|frontend/run_forever.sh|frontend/server.py' | grep -v grep || true
