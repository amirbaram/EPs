#!/usr/bin/env python3
"""mine_qm_setups.py — Precompute Historical QM Setups & EP Cross-References.

Iterates through common stock daily bars in `data/bars/*.parquet`, detects Qullamaggie
Momentum Continuation (QM) setups (both breakout and mature building states), collapses
consecutive consolidation days into unique episodes, cross-references each against the
curated Episodic Pivot dataset (`data/simulations/ep_combined_study_scored.parquet`),
and writes the enriched historical setups to `data/simulations/qm_historical_setups.parquet`.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
import datastore
import ep_catalyst_enricher
import indicators
import labels
import setups
import universe

EP_STUDY_PATH = Path("data/simulations/ep_combined_study_scored.parquet")
OUTPUT_PATH = Path("data/simulations/qm_historical_setups.parquet")

_GLOBAL_EP_MAP: Dict[str, List[Dict[str, Any]]] = {}


def load_ep_reference_map() -> Dict[str, List[Dict[str, Any]]]:
    """Loads curated EP events grouped by symbol for fast O(1) proximity matching."""
    if not EP_STUDY_PATH.exists():
        print(f"Warning: {EP_STUDY_PATH} not found; EP cross-referencing will be empty.")
        return {}
    df = pd.read_parquet(EP_STUDY_PATH)
    ep_map: Dict[str, List[Dict[str, Any]]] = {}
    for _, row in df.iterrows():
        sym = str(row["symbol"]).strip().upper()
        d_str = str(row["date"]).split("T")[0]
        rec = {
            "symbol": sym,
            "date": d_str,
            "gap_pct": round(float(row.get("gap_pct", 0.0)), 1),
            "rvol": round(float(row.get("rvol", 0.0)), 1),
            "close_pos": round(float(row.get("close_pos", 0.5)), 2),
            "subtype": str(row.get("subtype", "classic")),
            "dt": pd.to_datetime(d_str),
        }
        ep_map.setdefault(sym, []).append(rec)
    for sym in ep_map:
        ep_map[sym].sort(key=lambda x: x["dt"])
    return ep_map


def _enrich_with_rs_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """Enriches QM setups with institutional Relative Strength (RS), Stage 2, and 52W High distance."""
    rs_dir = Path("data/rs_history")
    if not rs_dir.exists():
        return df
    rs_files = sorted(rs_dir.glob("rs_*.parquet"))
    if not rs_files:
        return df
    try:
        cols = ["date", "symbol", "rs_score", "stage2", "dist_52w_high"]
        rs_dfs = [pd.read_parquet(p, columns=cols) for p in rs_files]
        all_rs = pd.concat(rs_dfs, ignore_index=True)
        all_rs["date"] = all_rs["date"].dt.strftime("%Y-%m-%d")
        for c in ["rs_score", "stage2", "dist_52w_high"]:
            if c in df.columns:
                df = df.drop(columns=[c])
        enriched = df.merge(all_rs, on=["symbol", "date"], how="left")
        print(f"Enriched {len(enriched)} setups with RS metrics (matched: {enriched['rs_score'].notna().sum()})")
        return enriched
    except Exception as e:
        print(f"Warning: Failed to enrich setups with RS metrics: {e}")
        return df


def _init_worker(ep_map: Dict[str, List[Dict[str, Any]]]) -> None:
    global _GLOBAL_EP_MAP
    _GLOBAL_EP_MAP = ep_map
    labels.load()


def _match_eps_for_setup(
    sym: str, qm_date_str: str, ep_list: List[Dict[str, Any]]
) -> Dict[str, Any]:
    qm_dt = pd.to_datetime(qm_date_str)
    prior_matches = []
    post_matches = []

    for ep in ep_list:
        delta_days = (qm_dt - ep["dt"]).days
        if 0 <= delta_days <= 60:
            prior_matches.append((delta_days, ep))
        elif -60 <= delta_days <= 0:
            post_matches.append((-delta_days, ep))

    # Closest prior EP (smallest positive delta)
    prior_ep = None
    if prior_matches:
        prior_matches.sort(key=lambda x: x[0])
        prior_ep = prior_matches[0][1]
        prior_days_ago = prior_matches[0][0]
    else:
        prior_days_ago = None

    # Closest post EP (smallest positive delta later)
    post_ep = None
    if post_matches:
        post_matches.sort(key=lambda x: x[0])
        post_ep = post_matches[0][1]
        post_days_later = post_matches[0][0]
    else:
        post_days_later = None

    has_prior = prior_ep is not None
    has_post = post_ep is not None

    # Badge classification
    if has_prior and prior_days_ago is not None and prior_days_ago <= 45:
        if has_post and post_days_later is not None and post_days_later <= 20:
            badge = "EP-Ignited & Boosted"
        else:
            badge = "EP-Ignited Pole"
    elif has_post and post_days_later is not None and post_days_later <= 20:
        badge = "EP-Boosted Breakout"
    elif has_prior or has_post:
        badge = "EP-Linked"
    else:
        badge = "Pure Technical QM"

    # Pure offline catalyst summary
    prior_cat = ""
    if has_prior and prior_ep:
        prior_cat = f"{prior_ep['subtype'].upper()} Catalyst (+{prior_ep['gap_pct']:.1f}% gap, {prior_ep['rvol']:.1f}x RVOL)"

    post_cat = ""
    if has_post and post_ep:
        post_cat = f"{post_ep['subtype'].upper()} Catalyst (+{post_ep['gap_pct']:.1f}% gap, {post_ep['rvol']:.1f}x RVOL)"

    return {
        "has_prior_ep": has_prior,
        "prior_ep_date": prior_ep["date"] if prior_ep else None,
        "prior_ep_days_ago": prior_days_ago,
        "prior_ep_gap_pct": prior_ep["gap_pct"] if prior_ep else None,
        "prior_ep_rvol": prior_ep["rvol"] if prior_ep else None,
        "prior_ep_subtype": prior_ep["subtype"] if prior_ep else None,
        "prior_ep_close_pos": prior_ep["close_pos"] if prior_ep else None,
        "prior_ep_catalyst": prior_cat,
        "has_post_ep": has_post,
        "post_ep_date": post_ep["date"] if post_ep else None,
        "post_ep_days_later": post_days_later,
        "post_ep_gap_pct": post_ep["gap_pct"] if post_ep else None,
        "post_ep_rvol": post_ep["rvol"] if post_ep else None,
        "post_ep_subtype": post_ep["subtype"] if post_ep else None,
        "post_ep_close_pos": post_ep["close_pos"] if post_ep else None,
        "post_ep_catalyst": post_cat,
        "ep_badge": badge,
    }


def scan_single_symbol(sym: str) -> List[Dict[str, Any]]:
    """Scans one symbol across its history, debounces episodes, and matches EPs."""
    try:
        p = config.BARS_DIR / f"{sym}.parquet"
        if not p.exists() or p.stat().st_size < 12000:
            return []
        df = datastore.load_bars(sym)
        if df is None or len(df) < 120:
            return []

        df = indicators.add_indicators(df)
        highs = df["high"].to_numpy(float)
        lows = df["low"].to_numpy(float)
        n = len(highs)
        W = setups.config.QMB_MOVE_LOOKBACK + setups.config.QMB_CONS_MAX
        min_bars = W + 30
        if n < min_bars:
            return []

        raw_hits = []
        for i in range(min_bars, n):
            # Fast candidate filter in numpy before heavy setup logic
            w_start = i - (W + 1)
            sub_hi = highs[w_start:i]
            hi_rel = int(np.argmax(sub_hi))
            hi_pos = w_start + hi_rel
            cons_bars = i - hi_pos
            if not (setups.config.QMB_CONS_MIN <= cons_bars <= setups.config.QMB_CONS_MAX):
                continue
            mstart = max(0, hi_pos - setups.config.QMB_MOVE_MAX_BARS)
            move_low = float(np.min(lows[mstart : hi_pos + 1]))
            if move_low <= 0 or (highs[hi_pos] / move_low - 1.0) * 100.0 < setups.config.QMB_MIN_MOVE:
                continue
            cons_low = float(np.min(lows[hi_pos : i + 1]))
            if (1.0 - cons_low / highs[hi_pos]) * 100.0 > setups.config.QMB_MAX_PULLBACK:
                continue

            res = setups.detect_qm_breakout(df.iloc[: i + 1])
            if res:
                res["symbol"] = sym
                res["date"] = df.index[i].strftime("%Y-%m-%d")
                raw_hits.append(res)

        if not raw_hits:
            return []

        # Group into unique consolidation episodes by pole (pole_start, pole_end)
        poles: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for h in raw_hits:
            pole_key = tuple(h["qm_pole"])
            poles.setdefault(pole_key, []).append(h)

        ep_list = _GLOBAL_EP_MAP.get(sym, [])
        sec = labels.sector(sym) or "Unknown"
        th_list = labels.themes(sym) or []
        primary_theme = th_list[0] if th_list else "General"
        themes_str = ", ".join(th_list) if th_list else primary_theme

        episodes = []
        for pole_key, items in poles.items():
            breakouts = [x for x in items if x["state"] == "breakout"]
            if breakouts:
                # Initial breakout
                chosen = breakouts[0]
            else:
                # Terminal mature building setup
                chosen = items[-1]

            ep_meta = _match_eps_for_setup(sym, chosen["date"], ep_list)

            rec = {
                "symbol": sym,
                "date": chosen["date"],
                "state": chosen["state"],
                "move_pct": round(float(chosen["move_pct"]), 1),
                "cons_bars": int(chosen["cons_bars"]),
                "pullback_pct": round(float(chosen["pullback_pct"]), 1),
                "ride_frac": round(float(chosen["ride_frac"]), 2),
                "shakeouts": int(chosen["shakeouts"]),
                "vol_x": round(float(chosen["vol_x"]), 2) if chosen.get("vol_x") is not None else None,
                "breakouts": int(chosen["breakouts"]),
                "levels": int(chosen["levels"]),
                "pattern": str(chosen["pattern"]),
                "dist_to_trigger_pct": round(float(chosen["dist_to_trigger_pct"]), 2),
                "level": round(float(chosen["level"]), 2),
                "trigger": round(float(chosen["trigger"]), 2),
                "qm_pole_start": chosen["qm_pole"][0],
                "qm_pole_end": chosen["qm_pole"][1],
                "qm_cons_start": chosen["qm_cons"][0],
                "qm_cons_end": chosen["qm_cons"][1],
                "stack_levels": [round(float(x), 2) for x in chosen.get("stack_levels", [])],
                "sector": sec,
                "theme": primary_theme,
                "themes": themes_str,
            }
            rec.update(ep_meta)
            episodes.append(rec)

        return episodes
    except Exception as e:
        return []
    finally:
        try:
            import trendlab
            trendlab._TRENDLAB_CACHE.clear()
        except Exception:
            pass
        gc.collect()


def rematch_existing_setups(output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    """Fast O(N) re-cross-reference of existing precomputed QM setups against updated EP study."""
    if not output_path.exists():
        raise FileNotFoundError(f"{output_path} not found; run full mining first.")
    ep_map = load_ep_reference_map()
    qm_df = pd.read_parquet(output_path)
    print(f"Re-matching {len(qm_df)} existing QM setups against {len(ep_map)} EP symbols...")

    t0 = time.time()
    updated_records = []
    for _, row in qm_df.iterrows():
        sym = str(row["symbol"]).strip().upper()
        d_str = str(row["date"])
        ep_list = ep_map.get(sym, [])
        ep_meta = _match_eps_for_setup(sym, d_str, ep_list)
        rec = row.to_dict()
        rec.update(ep_meta)
        updated_records.append(rec)

    new_df = pd.DataFrame(updated_records)
    new_df.to_parquet(output_path, index=False)
    print(f"Updated {len(new_df)} setups in {time.time() - t0:.2f}s -> {output_path}")

    # Summary
    bks = (new_df["state"] == "breakout").sum()
    bld = (new_df["state"] == "building").sum()
    prior_pct = (new_df["has_prior_ep"]).mean() * 100
    post_pct = (new_df["has_post_ep"]).mean() * 100
    print("-----------------------------------------------------------------")
    print(f"Summary: Total {len(new_df)} | Breakouts: {bks} | Building: {bld}")
    print(f"Prior EP Link (<=60d): {prior_pct:.1f}% | Post EP Link (<=60d): {post_pct:.1f}%")
    print("Badge distribution:")
    print(new_df["ep_badge"].value_counts().to_string())
    print("-----------------------------------------------------------------")
    return new_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute Historical QM Setups")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated symbols to scan")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker count (default: 4 for safe memory)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of symbols (0 = all)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH), help="Output parquet path")
    parser.add_argument("--rematch-eps", action="store_true", help="Quickly re-match existing QM setups against latest EP study dataset")
    args = parser.parse_args()

    if args.rematch_eps:
        rematch_existing_setups(Path(args.output))
        return

    print("=================================================================")
    print("🚀 MINING HISTORICAL QM SETUPS & EP CROSS-REFERENCES (LEAN MEMORY)")
    print("=================================================================")

    labels.load()
    ep_map = load_ep_reference_map()
    print(f"Loaded EP reference map with {len(ep_map)} unique symbols.")

    if args.symbols:
        sym_list = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        u = universe.load_universe()
        common = set(u[~u["etf"]]["yf_symbol"])
        bar_syms = set(p.stem for p in config.BARS_DIR.glob("*.parquet"))
        ep_syms = set(ep_map.keys())

        # Target universe: Common stocks with bar files + all EP symbols with bars
        candidate_syms = sorted(common.intersection(bar_syms).union(ep_syms.intersection(bar_syms)))

        # Filter out negligible files (< 15KB)
        sym_list = []
        for s in candidate_syms:
            p = config.BARS_DIR / f"{s}.parquet"
            if not p.exists() or p.stat().st_size < 15000:
                continue
            sym_list.append(s)

    if args.limit > 0:
        sym_list = sym_list[: args.limit]

    print(f"Total symbols to scan: {len(sym_list)} using {args.workers} workers (streaming to disk)...")
    t0 = time.time()

    temp_dir = Path("data/simulations/_qm_parts")
    temp_dir.mkdir(parents=True, exist_ok=True)
    # Clean previous temporary parts if any
    for old_f in temp_dir.glob("part_*.parquet"):
        old_f.unlink(missing_ok=True)

    processed = 0
    total = len(sym_list)
    total_setups_count = 0

    chunk_size = 100
    with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=50, initializer=_init_worker, initargs=(ep_map,)) as ex:
        for i in range(0, total, chunk_size):
            chunk = sym_list[i : i + chunk_size]
            results = list(ex.map(scan_single_symbol, chunk))
            chunk_setups = []
            for res in results:
                chunk_setups.extend(res)

            if chunk_setups:
                part_path = temp_dir / f"part_{i:06d}.parquet"
                pd.DataFrame(chunk_setups).to_parquet(part_path, index=False)
                total_setups_count += len(chunk_setups)

            processed += len(chunk)
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            sys.stdout.write(
                f"\rProgress: {processed}/{total} symbols ({processed/total*100:.1f}%) | "
                f"Setups: {total_setups_count} | Rate: {rate:.1f} sym/s"
            )
            sys.stdout.flush()

            del chunk_setups, results
            gc.collect()

    print()
    t1 = time.time()
    print(f"Finished scan in {t1 - t0:.1f}s. Total historical QM setups found: {total_setups_count}")

    part_files = sorted(temp_dir.glob("part_*.parquet"))
    if not part_files:
        print("Warning: No setups found!")
        return

    print("Consolidating streamed parquet parts...")
    res_df = pd.concat([pd.read_parquet(pf) for pf in part_files], ignore_index=True)
    res_df = _enrich_with_rs_technicals(res_df)
    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_parquet(out_p, index=False)
    print(f"Saved {len(res_df)} historical setups to {out_p}")

    # Clean up part files
    for pf in part_files:
        pf.unlink(missing_ok=True)
    try:
        temp_dir.rmdir()
    except Exception:
        pass

    # Summary statistics
    bks = (res_df["state"] == "breakout").sum()
    bld = (res_df["state"] == "building").sum()
    prior_pct = (res_df["has_prior_ep"]).mean() * 100
    post_pct = (res_df["has_post_ep"]).mean() * 100
    print("-----------------------------------------------------------------")
    print(f"Summary: Total {len(res_df)} | Breakouts: {bks} | Building: {bld}")
    print(f"Prior EP Link (<=60d): {prior_pct:.1f}% | Post EP Link (<=60d): {post_pct:.1f}%")
    print("Badge distribution:")
    print(res_df["ep_badge"].value_counts().to_string())
    print("-----------------------------------------------------------------")


if __name__ == "__main__":
    main()
