#!/usr/bin/env python3
"""Merge Historical Delayed Reaction EP (DRE) Setups into the Master Scored Dataset.

Extracts debounced institutional DRE setups from `data/simulations/historical_dre_setups.parquet`,
computes the full 81-column feature matrix, ML rolling predictions, and trade simulation lifecycle
using `build_ep_row`, and atomically merges them into `data/simulations/ep_combined_study_scored.parquet`.
"""

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import thematic_engine
import ep_catalyst_enricher
from scripts.update_ep_dataset import build_ep_row

DATA_PATH = ROOT / "data" / "simulations" / "ep_combined_study_scored.parquet"
DRE_PATH = ROOT / "data" / "simulations" / "historical_dre_setups.parquet"
PARTS_DIR = ROOT / "data" / "simulations" / "_dre_parts"

_WORKER_THM = None


def _init_worker():
    global _WORKER_THM
    # Silence network requests during offline backfill
    ep_catalyst_enricher.get_catalyst_dossier = lambda *a, **kw: {}
    _WORKER_THM = thematic_engine.get_thematic_engine()


def _process_item(item: tuple[str, str, str]) -> dict | None:
    sym, dt_str, subtype = item
    try:
        return build_ep_row(sym, dt_str, subtype, _WORKER_THM)
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser(description="Merge historical DRE setups into ep_combined_study_scored.parquet")
    parser.add_argument("--workers", type=int, default=10, help="Number of worker processes (default: 10)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of setups to process (0 = all)")
    parser.add_argument("--min-gap", type=float, default=5.0, help="Minimum gap percentage (default: 5.0%)")
    parser.add_argument("--chunk-size", type=int, default=100, help="Chunk size for batch processing (default: 100)")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing to main parquet file")
    args = parser.parse_args()

    print("=" * 80)
    print("🚀 INSTITUTIONAL DELAYED REACTION EP (DRE) HISTORICAL MERGE PIPELINE")
    print("=" * 80)

    if not DATA_PATH.exists():
        print(f"❌ Error: Master dataset {DATA_PATH} not found.")
        sys.exit(1)
    if not DRE_PATH.exists():
        print(f"❌ Error: DRE candidates dataset {DRE_PATH} not found.")
        sys.exit(1)

    df_orig = pd.read_parquet(DATA_PATH)
    print(f"📊 Existing Scored EP Dataset: {len(df_orig)} events across {df_orig['symbol'].nunique()} symbols.")
    seen_keys = set(zip(df_orig["symbol"], df_orig["date"]))

    df_dre = pd.read_parquet(DRE_PATH)
    print(f"📁 Raw Historical DRE Setups: {len(df_dre)} events.")

    # Apply institutional DRE criteria:
    # 1. Gap >= min-gap%
    # 2. RVOL >= 2.4x
    # 3. Dollar volume >= $15M
    # 4. Close position < 0.65 (messy close)
    # 5. Price >= $3.0
    mask = (
        (df_dre["gap_pct"] >= args.min_gap) &
        (df_dre["rvol"] >= 2.4) &
        (df_dre["dvol_m"] >= 15.0) &
        (df_dre["close_pos"] < 0.65) &
        (df_dre["d1_close"] >= 3.0)
    )
    cands = df_dre[mask].sort_values(by=["symbol", "date"]).reset_index(drop=True)
    print(f"🔍 Candidates meeting institutional filters (gap >= {args.min_gap}%, rvol >= 2.4x, $vol >= $15M): {len(cands)}")

    # Debounce 15 sessions between events for the same ticker
    debounced = []
    for sym, grp in cands.groupby("symbol"):
        last_idx = -999
        for _, r in grp.iterrows():
            b_idx = r["bar_idx"]
            if b_idx - last_idx >= 15:
                debounced.append(r)
                last_idx = b_idx

    deb_df = pd.DataFrame(debounced)
    print(f"🛡️ Debounced candidates (15 sessions apart): {len(deb_df)}")

    # Filter out events already present in master dataset
    items_to_process = []
    for _, r in deb_df.iterrows():
        key = (r["symbol"], str(r["date"]))
        if key not in seen_keys:
            items_to_process.append((r["symbol"], str(r["date"]), "dre"))

    if args.limit > 0:
        items_to_process = items_to_process[:args.limit]

    print(f"🎯 New unique DRE setups to process: {len(items_to_process)}")
    if not items_to_process:
        print("🎉 No new DRE setups to merge! Master dataset is already up to date.")
        return

    # Clean parts directory
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    for old_p in PARTS_DIR.glob("part_*.parquet"):
        old_p.unlink(missing_ok=True)

    t0 = time.time()
    total_items = len(items_to_process)
    processed = 0
    successful_rows = 0

    print(f"\n⚙️ Running batch feature extraction with {args.workers} worker processes...")

    with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=50, initializer=_init_worker) as executor:
        for i in range(0, total_items, args.chunk_size):
            chunk = items_to_process[i : i + args.chunk_size]
            results = list(executor.map(_process_item, chunk))
            valid_chunk = [r for r in results if r is not None]

            if valid_chunk:
                part_df = pd.DataFrame(valid_chunk).reindex(columns=df_orig.columns)
                part_path = PARTS_DIR / f"part_{i:06d}.parquet"
                part_df.to_parquet(part_path, index=False)
                successful_rows += len(valid_chunk)

            processed += len(chunk)
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            eta_sec = (total_items - processed) / rate if rate > 0 else 0
            sys.stdout.write(
                f"\rProgress: {processed}/{total_items} ({processed/total_items*100:.1f}%) | "
                f"Valid: {successful_rows} | Rate: {rate:.1f} rows/s | ETA: {eta_sec/60:.1f}m"
            )
            sys.stdout.flush()

    print(f"\n\n✅ Feature extraction complete in {time.time() - t0:.1f}s. Valid rows constructed: {successful_rows}")

    if args.dry_run:
        print("🔍 Dry run requested. Main parquet file was not modified.")
        return

    # Collect all parts
    all_parts = sorted(PARTS_DIR.glob("part_*.parquet"))
    if not all_parts:
        print("❌ No parts generated. Aborting merge.")
        return

    print("\n📦 Loading and concatenating generated parts...")
    part_dfs = [pd.read_parquet(p) for p in all_parts]
    df_new_all = pd.concat(part_dfs, ignore_index=True)

    # Backup original master dataset
    backup_file = DATA_PATH.parent / f"ep_combined_study_scored_backup_pre_dre_{time.strftime('%Y%m%d_%H%M%S')}.parquet"
    shutil.copy2(DATA_PATH, backup_file)
    print(f"💾 Created safe backup of master dataset at: {backup_file}")

    # Merge, deduplicate, sort
    combined = pd.concat([df_orig, df_new_all], ignore_index=True)
    combined = combined.drop_duplicates(subset=["symbol", "date"], keep="last")
    combined = combined.sort_values(by=["date", "symbol"], ascending=[False, True]).reset_index(drop=True)

    tmp_out = DATA_PATH.with_suffix(".tmp.parquet")
    combined.to_parquet(tmp_out, index=False)
    tmp_out.replace(DATA_PATH)
    print(f"🚀 Success! Master dataset updated from {len(df_orig)} to {len(combined)} events.")

    # Cleanup temp parts
    for p in all_parts:
        p.unlink(missing_ok=True)
    try:
        PARTS_DIR.rmdir()
    except Exception:
        pass

    # Send server reload signal
    print("\n🔄 Reloading active review server on port 8782...")
    try:
        resp = requests.post("http://127.0.0.1:8782/api/reload_dataset", timeout=5)
        if resp.status_code == 200:
            info = resp.json()
            print(f"✨ Server reloaded successfully! Active events: {info.get('rows')}")
        else:
            print(f"⚠️ Server returned status {resp.status_code}")
    except Exception as e:
        print(f"⚠️ Could not contact server at http://127.0.0.1:8782: {e}")


if __name__ == "__main__":
    main()
