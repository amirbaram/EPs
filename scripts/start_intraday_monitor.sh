#!/usr/bin/env bash
# scripts/start_intraday_monitor.sh — Start Intraday Watchlist Monitor Daemon
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs data/cache

INTERVAL="${1:-300}"

echo "Starting Intraday Watchlist Monitor Daemon (Interval: ${INTERVAL}s / $((${INTERVAL}/60))min)..."
nohup .venv/bin/python intraday_watchlist_monitor.py --daemon --interval "${INTERVAL}" >> logs/intraday_monitor.log 2>&1 &

sleep 1
.venv/bin/python intraday_watchlist_monitor.py --status
