#!/usr/bin/env bash
# scripts/stop_intraday_monitor.sh — Stop Intraday Watchlist Monitor Daemon
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

.venv/bin/python intraday_watchlist_monitor.py --stop
