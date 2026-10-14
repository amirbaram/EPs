"""update_qm_database.py — Automated Updater & Ingestion Pipeline for QM Setups (Port 8785).

Checks whether new daily bars exist in `data/bars/` beyond the latest setup date stored in
`data/simulations/qm_historical_setups.parquet`. Coordinates the sequential updates of:
1. The Institutional RS Database (`data/rs_history/` via `update_rs_database.py`).
2. The QM Setup Miner & EP Cross-Referencer (`mine_qm_setups.py`).
3. Hot-reloads the active `serve_qm.py` instance on port 8785 via `/api/reload_dataset`.

Usage:
  .venv/bin/python update_qm_database.py            # Check and run full update if new daily bars exist
  .venv/bin/python update_qm_database.py --check    # Dry-run inspection only (no writes)
  .venv/bin/python update_qm_database.py --force    # Force re-mining and dataset refresh
  .venv/bin/python update_qm_database.py --reload   # Send hot-reload signal to live 8785 server
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

import build_qm_strategy_dataset
import config
import datastore
import update_rs_database
import mine_qm_setups
import qm_ml_engine

QM_DATASET_PATH = Path("data/simulations/qm_historical_setups.parquet")
SERVER_URL = "http://127.0.0.1:8785"


def get_qm_status() -> Tuple[Optional[str], Optional[str], int]:
    """Return (qm_latest_date, datastore_latest_date, total_qm_setups)."""
    qm_latest = None
    qm_count = 0
    if QM_DATASET_PATH.exists():
        try:
            df = pd.read_parquet(QM_DATASET_PATH, columns=["date"])
            qm_count = len(df)
            if not df.empty and "date" in df.columns:
                qm_latest = str(df["date"].max())
        except Exception as e:
            print(f"Warning: Failed reading {QM_DATASET_PATH}: {e}")

    spy = datastore.load_bars("SPY")
    spy_latest = spy.index[-1].strftime("%Y-%m-%d") if spy is not None and not spy.empty else None

    return qm_latest, spy_latest, qm_count


def reload_live_server(port: int = 8785) -> bool:
    """Sends a hot-reload request to the running serve_qm server."""
    url = f"http://127.0.0.1:{port}/api/reload_dataset"
    try:
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                print(f"✅ Successfully notified live server at {url} to hot-reload dataset.")
                return True
    except urllib.error.URLError:
        print(f"ℹ️  Live server at {url} is not running. Changes will take effect on next launch.")
    except Exception as e:
        print(f"⚠️  Could not trigger live reload at {url}: {e}")
    return False


def run_update(force: bool = False, check_only: bool = False, reload_only: bool = False) -> bool:
    """Executes the QM update pipeline."""
    if reload_only:
        return reload_live_server()

    qm_latest, spy_latest, qm_count = get_qm_status()

    print("══════════════════════════════════════════════════════════════════════════════")
    print(" 🚀 Institutional QM Setups Database Status Check (Port 8785)")
    print("══════════════════════════════════════════════════════════════════════════════")
    print(f"  • QM Setups latest date   : {qm_latest or 'None (Not Built)'} ({qm_count:,} setups)")
    print(f"  • Datastore (SPY) latest : {spy_latest or 'None'}")

    if not spy_latest:
        print("❌ Error: No SPY bar data found in datastore. Aborting.")
        return False

    needs_update = False
    if not qm_latest or force:
        needs_update = True
    elif spy_latest > qm_latest:
        needs_update = True

    if not needs_update:
        print(f"✨ QM database is already up to date with datastore (latest session: {qm_latest}).")
        return True

    print(f"\n⚡ New candle data detected: datastore is at {spy_latest}, QM dataset is at {qm_latest or 'start'}.")

    if check_only:
        print("🔍 Check-only mode active. Skipping mining updates.")
        return True

    # Step 1: Ensure RS database is synchronized
    print("\n------------------------------------------------------------------------------")
    print("Step 1/3: Ensuring Institutional RS Database is up to date...")
    print("------------------------------------------------------------------------------")
    update_rs_database.update_database(force=force)

    # Step 2: Run QM Miner to capture new setups, EP cross-references, and RS technicals
    print("\n------------------------------------------------------------------------------")
    print("Step 2/3: Mining QM Setups & Cross-Referencing Episodic Pivots...")
    print("------------------------------------------------------------------------------")
    t0 = time.time()
    
    # Run mining across symbol universe
    sys.argv = ["mine_qm_setups.py", "--workers", "4"]
    mine_qm_setups.main()
    print(f"Mining completed in {time.time() - t0:.1f}s.")

    # Step 3: Run QM Strategy Builder to compute multi-EMA and ML climax exits
    print("\n------------------------------------------------------------------------------")
    print("Step 3/4: Building Strategy Execution Dataset & Climax Exhaustion Exits...")
    print("------------------------------------------------------------------------------")
    t1 = time.time()
    strat_trades_df = build_qm_strategy_dataset.build_strategy_trades_dataset()
    print(f"Strategy dataset build completed in {time.time() - t1:.1f}s.")

    # Step 3b: Enrich setups dataset with ML setup quality tiers
    print("\n------------------------------------------------------------------------------")
    print("Step 3b/4: Enriching Historical & Fresh Setups with ML Setup Quality Tiers...")
    print("------------------------------------------------------------------------------")
    if QM_DATASET_PATH.exists():
        t2 = time.time()
        df_setups = pd.read_parquet(QM_DATASET_PATH)
        df_enriched = qm_ml_engine.get_qm_ml_engine().enrich_setups_with_ml(df_setups, strat_trades_df)
        df_enriched.to_parquet(QM_DATASET_PATH, index=False)
        print(f"Enriched {len(df_enriched)} setups with ML quality tiers in {time.time() - t2:.1f}s.")

    # Step 4: Trigger live reload on serve_qm
    print("\n------------------------------------------------------------------------------")
    print("Step 4/4: Notifying QM Setup Viewer server (port 8785)...")
    print("------------------------------------------------------------------------------")
    reload_live_server()

    print("\n✅ QM Setups Database update pipeline completed successfully.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Automated updater for the QM Setups Database.")
    parser.add_argument("--check", action="store_true", help="Dry-run check without modifying data files")
    parser.add_argument("--force", action="store_true", help="Force rebuild regardless of timestamps")
    parser.add_argument("--reload", action="store_true", help="Hot-reload dataset in running serve_qm server")
    args = parser.parse_args()

    success = run_update(force=args.force, check_only=args.check, reload_only=args.reload)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
