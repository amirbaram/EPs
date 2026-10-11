#!/usr/bin/env python3
"""Institutional Historical EP Catalyst & SEC EDGAR Batch Enricher.

Pre-mines and caches fundamental catalyst classifications, SEC EDGAR 8-K/6-K/10-Q filings,
and earnings calendar concurrences for historical EP events without requiring paid data subscriptions.

Usage:
    .venv/bin/python scripts/enrich_historical_catalysts.py --since 2026-01-01
    .venv/bin/python scripts/enrich_historical_catalysts.py --symbols SMMT,CCL,KOD,NVDA,TSLA
    .venv/bin/python scripts/enrich_historical_catalysts.py --limit 50
    .venv/bin/python scripts/enrich_historical_catalysts.py --all
"""

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Root repository setup
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ep_catalyst_enricher

DATA_PATH = ROOT / "data" / "simulations" / "ep_combined_study_scored.parquet"


def main():
    parser = argparse.ArgumentParser(description="Enrich historical EP events with SEC EDGAR filings and earnings catalysts")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated list of symbols to enrich (e.g. SMMT,CCL,KOD)")
    parser.add_argument("--since", type=str, default="", help="Enrich EP events on or after this date (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of events to process (0 = all matching)")
    parser.add_argument("--all", action="store_true", help="Process all historical events in the dataset")
    parser.add_argument("--delay", type=float, default=0.12, help="Delay in seconds between SEC API calls (default 0.12s for fair access)")
    args = parser.parse_args()

    print("=" * 80)
    print("🏛️ INSTITUTIONAL EP CATALYST & SEC EDGAR BATCH ENRICHER")
    print("=" * 80)

    if not DATA_PATH.exists():
        print(f"❌ Error: {DATA_PATH} not found!")
        sys.exit(1)

    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df)} total historical EP events across {df['symbol'].nunique()} symbols.")

    # Filter by symbol
    if args.symbols:
        sym_list = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        df = df[df["symbol"].isin(sym_list)]
        print(f"Filtered by symbols {sym_list}: {len(df)} events.")

    # Filter by date
    if args.since:
        df = df[df["date"] >= args.since]
        print(f"Filtered by date >= {args.since}: {len(df)} events.")
    elif not args.all and not args.symbols and not args.limit:
        # Default to the most recent 1 year of events if neither --all nor --since nor --symbols specified
        one_year_ago = (pd.Timestamp.now() - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
        df = df[df["date"] >= one_year_ago]
        print(f"Defaulting to recent 1-year window (>= {one_year_ago}): {len(df)} events.")

    if args.limit > 0:
        df = df.head(args.limit)
        print(f"Applying limit: processing {len(df)} events.")

    if df.empty:
        print("No events to process. Exiting.")
        return

    # Sort descending by date so most recent events are processed first
    df = df.sort_values(by="date", ascending=False)
    events = df[["symbol", "date", "sector", "theme"]].drop_duplicates(subset=["symbol", "date"]).to_dict("records")
    total_events = len(events)
    print(f"Processing {total_events} unique (symbol, date) setups...")

    t0 = time.time()
    enriched_count = 0
    filings_count = 0
    earnings_count = 0

    for i, ev in enumerate(events, 1):
        sym = ev["symbol"]
        date_str = ev["date"]
        sec = ev.get("sector") or ""
        thm = ev.get("theme") or ""

        try:
            dossier = ep_catalyst_enricher.get_catalyst_dossier(sym, date_str, sector=sec, theme=thm)
            enriched_count += 1
            n_filings = len(dossier.get("sec_filings", []))
            if n_filings > 0:
                filings_count += 1
            if dossier.get("has_earnings_concurrence"):
                earnings_count += 1

            if i % 10 == 0 or i == total_events:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(f"[{i}/{total_events}] ({i/total_events*100:.1f}%) | {sym} {date_str} -> {dossier['category']} ({n_filings} filings) [{rate:.1f} ev/s]")

            time.sleep(args.delay)
        except Exception as e:
            print(f"⚠️ Error processing {sym} on {date_str}: {e}")

    print("=" * 80)
    print("✅ ENRICHMENT COMPLETE")
    print(f"Total Events Enriched: {enriched_count} / {total_events}")
    print(f"Events with Official SEC Filings: {filings_count} ({filings_count/max(1, enriched_count)*100:.1f}%)")
    print(f"Events with Verified Earnings Concurrence: {earnings_count} ({earnings_count/max(1, enriched_count)*100:.1f}%)")
    print(f"Total Runtime: {time.time() - t0:.1f}s")
    print(f"Cache Location: {ep_catalyst_enricher.HIST_CATALYSTS_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
