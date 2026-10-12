"""update_rs_database.py — Automated incremental updater for the Institutional RS Database.

Checks whether new daily bars exist in `data/bars/` beyond the latest date stored in
`data/rs_history/`, and incrementally updates the yearly Parquet files and latest live snapshot
(`data/simulations/rs_universe_scores.parquet`).

Usage:
  .venv/bin/python update_rs_database.py             # Check and update if new trading days exist
  .venv/bin/python update_rs_database.py --check     # Dry-run inspection only (no writes)
  .venv/bin/python update_rs_database.py --force     # Force rebuild from current year start
  .venv/bin/python update_rs_database.py --since 2026-10-01  # Refresh starting from specific date
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

import build_rs_history
import config
import datastore

LATEST_SNAPSHOT_PATH = Path("data/simulations/rs_universe_scores.parquet")


def get_db_status() -> Tuple[Optional[str], Optional[str], int]:
    """Return (db_latest_date, datastore_latest_date, total_sessions_in_db)."""
    dates = build_rs_history.list_available_dates()
    db_latest = dates[-1] if dates else None
    db_count = len(dates)

    spy = datastore.load_bars("SPY")
    spy_latest = spy.index[-1].strftime("%Y-%m-%d") if spy is not None and not spy.empty else None

    return db_latest, spy_latest, db_count


def update_database(since: Optional[str] = None, force: bool = False, check_only: bool = False) -> bool:
    """Run incremental update if new data exists or if forced."""
    db_latest, spy_latest, db_count = get_db_status()

    print("══════════════════════════════════════════════════════════════════════════════")
    print(" Institutional Relative Strength (RS) Database Status Check")
    print("══════════════════════════════════════════════════════════════════════════════")
    print(f"  • RS Database latest session : {db_latest or 'None (Not Built)'} ({db_count:,} sessions)")
    print(f"  • Datastore (SPY) latest bar : {spy_latest or 'None'}")

    if not spy_latest:
        print("❌ Error: No SPY bar data found in datastore. Aborting.")
        return False

    # Determine whether an update is needed
    needs_update = False
    start_since = since

    if force:
        needs_update = True
        if not start_since:
            # By default under force, refresh the current year
            current_year = pd.Timestamp(spy_latest).year
            start_since = f"{current_year}-01-01"
        print(f"  • Force flag active: Recomputing from {start_since}...")
    elif start_since:
        needs_update = True
        print(f"  • Custom start date specified: Recomputing from {start_since}...")
    elif not db_latest:
        needs_update = True
        print("  • Database does not exist yet: Running full build from 2016-01-01...")
    elif pd.Timestamp(spy_latest) > pd.Timestamp(db_latest):
        needs_update = True
        # Find the next business day after db_latest
        start_since = db_latest
        print(f"  • New trading sessions detected! Updating since {start_since}...")
    else:
        print(f"  ✔ RS Database is fully up to date through {db_latest}.")

    if check_only:
        print("\n[Dry-run check complete. No files modified.]")
        return not needs_update

    if not needs_update:
        return True

    # Execute incremental build
    t0 = time.time()
    print("\nExecuting build_rs_history.build()...")
    build_rs_history.build(since=start_since)

    # Refresh the live snapshot in data/simulations/rs_universe_scores.parquet
    print("Updating latest live snapshot in data/simulations/rs_universe_scores.parquet...")
    new_dates = build_rs_history.list_available_dates()
    new_latest_date = new_dates[-1] if new_dates else spy_latest
    latest_df = build_rs_history.load_rs_history(new_latest_date)

    if not latest_df.empty:
        # Standardize column naming for live app compatibility
        snap_df = latest_df.copy()
        snap_df["last_date"] = pd.to_datetime(snap_df["date"]).dt.strftime("%Y-%m-%d")
        snap_df["bars_count"] = 252  # representative historical window

        # Ensure directory exists and save
        LATEST_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        snap_df.to_parquet(LATEST_SNAPSHOT_PATH, index=False)
        print(f"  ✔ Saved {len(snap_df):,} records for {new_latest_date} to {LATEST_SNAPSHOT_PATH}")

    elapsed = time.time() - t0
    print(f"\n✔ Update completed successfully in {elapsed:.1f}s.")
    print("══════════════════════════════════════════════════════════════════════════════")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Only check status; do not write any files")
    parser.add_argument("--force", action="store_true", help="Force rebuild of current year sessions")
    parser.add_argument("--since", type=str, help="Start date to update from (YYYY-MM-DD)")
    args = parser.parse_args()

    success = update_database(since=args.since, force=args.force, check_only=args.check)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
