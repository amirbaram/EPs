#!/usr/bin/env bash
# Start the institutional EOD daily updater daemon in the background.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=".venv/bin/python"

if [ ! -x "$PY" ]; then
    echo "❌ Error: Python virtual environment not found at $PY"
    exit 1
fi

# If an older daemon instance is already running, restart it cleanly so latest code & flags apply
if "$PY" daily_update.py --status | grep -q "RUNNING"; then
    echo "🔄 Restarting existing Daily Updater Daemon to apply latest configuration..."
    "$PY" daily_update.py --stop
    sleep 1
fi

mkdir -p logs data
echo "🚀 Starting Daily Updater Daemon (16:45 US/Eastern schedule, full universe + 4 pivot models)..."
nohup "$PY" daily_update.py --daemon --full-bars "$@" >> logs/daily_update.log 2>&1 &

sleep 1
"$PY" daily_update.py --status
echo "📋 Tail output logs anytime with: tail -f logs/daily_update.log"
