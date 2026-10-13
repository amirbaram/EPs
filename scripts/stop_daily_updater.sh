#!/usr/bin/env bash
# Gracefully stop the institutional EOD daily updater daemon.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=".venv/bin/python"

if [ ! -x "$PY" ]; then
    echo "❌ Error: Python virtual environment not found at $PY"
    exit 1
fi

"$PY" daily_update.py --stop
