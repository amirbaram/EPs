#!/usr/bin/env bash
# Start the institutional EOD daily updater daemon in the background.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=".venv/bin/python"

if [ ! -x "$PY" ]; then
    echo "❌ Error: Python virtual environment not found at $PY"
    exit 1
fi

# Check if already running
if "$PY" daily_update.py --status | grep -q "RUNNING"; then
    echo "⚠️ Daily updater is already running:"
    "$PY" daily_update.py --status
    exit 0
fi

mkdir -p logs data
echo "🚀 Starting Daily Updater Daemon (16:45 US/Eastern schedule)..."
nohup "$PY" daily_update.py --daemon >> logs/daily_update.log 2>&1 &

sleep 1
"$PY" daily_update.py --status
echo "📋 Tail output logs anytime with: tail -f logs/daily_update.log"
