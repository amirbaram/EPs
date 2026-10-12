#!/usr/bin/env python3
"""Institutional EP Dataset Updater.

Updates `data/simulations/ep_combined_study_scored.parquet` up to the present day:
1. Downloads recent missing daily bars from yfinance (datastore.update_daily).
2. Scans active universe for new qualifying Episodic Pivots since the last dataset cutoff.
3. Computes point-in-time sector/theme context, confirmation windows, and forward excursion metrics.
4. Infers V2 rolling conviction models (ml_50, ml_100, ml_150, ml_200, dynamic_stop_loss_pct).
5. Runs zero-lookahead Trade 1 Delayed Breakout and Multi-Leg Continuation simulations.
6. Deduplicates on ['symbol', 'date'] and atomically saves to `ep_combined_study_scored.parquet`.
7. Reloads the active Historical Review Server (port 8782) via `/api/reload_dataset`.

Usage:
    .venv/bin/python scripts/update_ep_dataset.py
    .venv/bin/python scripts/update_ep_dataset.py --skip-bars-download
    .venv/bin/python scripts/update_ep_dataset.py --since 2026-09-01
"""

import argparse
import datetime as dt
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Ensure repository root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import datastore
import indicators
import labels
import scanner_core
import setups
import thematic_engine
import universe
from ep_ml_engine import engine
import serve_ep
import trading_calendar

DATA_PATH = ROOT / "data" / "simulations" / "ep_combined_study_scored.parquet"


def download_latest_bars(full: bool = False):
    print("=" * 80)
    print("📡 STEP 1: DOWNLOADING LATEST EOD PRICE BARS")
    print("=" * 80)
    t0 = time.time()
    try:
        datastore.update_daily(full=full)
        print(f"✅ Daily bars successfully updated in {time.time() - t0:.1f}s.")
    except Exception as e:
        print(f"⚠️ Warning during bar download: {e}. Proceeding with local bars...")


import json

def scan_for_new_eps(since_date: str) -> list[tuple[str, str, str]]:
    print("\n" + "=" * 80)
    print(f"🔍 STEP 2: SCANNING FOR FRESH EPISODIC PIVOTS SINCE {since_date}")
    print("=" * 80)
    t0 = time.time()
    active_syms = list(universe.active_symbols())
    print(f"Scanning {len(active_syms)} active universe symbols...")

    fresh_eps = []
    seen_keys = set()
    processed = 0

    for sym in active_syms:
        processed += 1
        raw = datastore.load_bars(sym)
        if raw is None or len(raw) < 50:
            continue

        # Fast tail slicing (last 150 bars) gives ample lookback for 60d med vol and indicators
        tail = raw.iloc[-150:].copy()
        d = indicators.add_indicators(tail)
        if d is None or len(d) < 30:
            continue

        # 1. Classical / EP9M / BigMove EP setups
        eps = setups.iter_fresh_ep_events(d)
        for idx, subtype in eps:
            dt_str = str(d.index[idx].date())
            if dt_str >= since_date:
                key = (sym, dt_str)
                if key not in seen_keys:
                    fresh_eps.append((sym, dt_str, subtype))
                    seen_keys.add(key)

        # 2. Pradeep Bonde Delayed Reaction EPs (DRE):
        # Catalyst energy: Gap >= 5.0% or 9M volume or chg_pct >= 8.0%,
        # RVOL >= 2.4x (or round(rvol, 1) >= 2.5), $Vol >= $15M, holding Day 1 Low
        n = len(d)
        med60 = (d["close"] * d["volume"]).rolling(60, min_periods=10).median().shift(1)
        for j in range(30, n):
            dt_str = str(d.index[j].date())
            if dt_str < since_date:
                continue
            key = (sym, dt_str)
            if key in seen_keys:
                continue

            row = d.iloc[j]
            gap = float(row["gap_pct"]) if pd.notna(row.get("gap_pct")) else 0.0
            rvol = float(row["rvol"]) if pd.notna(row.get("rvol")) else 0.0
            vol_shares = float(row["volume"])
            dvol_m = float(row["close"]) * vol_shares / 1e6
            chg = float(row["chg_pct"]) if pd.notna(row.get("chg_pct")) else 0.0

            m60_val = med60.iloc[j] if pd.notna(med60.iloc[j]) else None
            sub = setups._is_ep_event(row, m60_val)
            if sub:
                continue

            fwd = d.iloc[j:]
            held_d1_low = float(fwd["low"].min()) >= (float(row["low"]) * 0.999)

            is_mech_vol = trading_calendar.is_mechanical_volume_date(dt_str)
            valid_catalyst = gap >= 5.0 if is_mech_vol else (gap >= 5.0 or vol_shares >= 9_000_000 or chg >= 8.0)
            if valid_catalyst and (rvol >= 2.4 or round(rvol, 1) >= 2.5) and dvol_m >= 15.0 and held_d1_low:
                fresh_eps.append((sym, dt_str, "dre"))
                seen_keys.add(key)

    # 3. Ingest any confirmed tracker EPs from data/cache/tracker_eps.json if available
    tracker_path = ROOT / "data" / "cache" / "tracker_eps.json"
    if tracker_path.exists():
        try:
            with open(tracker_path) as f:
                tracker_data = json.load(f)
            for item in tracker_data:
                tsym = item.get("symbol")
                tev_dt = item.get("event_date")
                tsubtype = item.get("subtype", "dre")
                if tsym and tev_dt and tev_dt >= since_date:
                    key = (tsym, tev_dt)
                    if key not in seen_keys:
                        fresh_eps.append((tsym, tev_dt, tsubtype))
                        seen_keys.add(key)
        except Exception as e:
            print(f"Warning reading tracker cache: {e}")

    # Sort deterministically by date descending, symbol ascending
    fresh_eps.sort(key=lambda x: (x[1], x[0]), reverse=True)

    print(f"✅ Discovered {len(fresh_eps)} fresh EP events since {since_date} in {time.time() - t0:.1f}s.")
    for sym, dt_str, subtype in fresh_eps:
        print(f"   • {sym:6s} | {dt_str} | {subtype}")

    return fresh_eps


def build_ep_row(sym: str, dt_str: str, subtype: str, thm_engine) -> dict | None:
    raw = datastore.load_bars(sym)
    if raw is None or len(raw) < 30:
        return None

    d = indicators.add_indicators(raw)
    d = scanner_core.calc_larssson_line(d)
    if d is None or len(d) < 30:
        return None

    ts = pd.Timestamp(dt_str)
    if ts not in d.index:
        return None
    pos = d.index.get_loc(ts)
    if isinstance(pos, (slice, np.ndarray)):
        pos = int(pos[0]) if isinstance(pos, np.ndarray) else int(pos.start)

    d1 = d.iloc[pos]
    d1_close = float(d1["close"])
    d1_open = float(d1["open"])
    d1_high = float(d1["high"])
    d1_low = float(d1["low"])
    d1_vol = float(d1["volume"])

    gap_pct = float(d1.get("gap_pct", 0.0)) if pd.notna(d1.get("gap_pct")) else 0.0
    rvol = float(d1.get("rvol", 1.0)) if pd.notna(d1.get("rvol")) else 1.0
    dvol_m = float(d1_close * d1_vol / 1e6)
    rng = d1_high - d1_low
    close_pos = float((d1_close - d1_low) / rng) if rng > 1e-4 else 0.5

    # 6-month neglect
    neglect_6m = 0.0
    if pos >= 126:
        neglect_6m = float((d["close"].iloc[pos - 1] / d["close"].iloc[pos - 126] - 1.0) * 100.0)

    # Sector / Theme context
    sec_name = labels.sector(sym) or "Unknown"
    thm_list = labels.themes(sym)
    thm_name = thm_list[0] if thm_list else "General"

    thm_eval = thm_engine.evaluate_symbol(sym)
    sec_info = thm_eval.get("sector_info", {})
    thm_info = thm_eval.get("theme_info", {})
    sec_m1_pctile = float(sec_info.get("m1_pct", 50.0))
    sec_m3_pctile = float(sec_info.get("m3_pct", 50.0))
    thm_m1_pctile = float(thm_info.get("m1_pct", 50.0))

    # Previous day gap (only flagged if gap > 0.33 * ADR)
    prev_day_gap_pct = 0.0
    prev_day_gap_up = False
    adr_pct_val = float(d["adr_pct"].iloc[pos - 1]) if ("adr_pct" in d.columns and pos > 0 and pd.notna(d["adr_pct"].iloc[pos - 1])) else 5.0
    if pos > 0 and "gap_pct" in d.columns:
        prev_gap = d["gap_pct"].iloc[pos - 1]
        if pd.notna(prev_gap):
            prev_day_gap_pct = float(prev_gap)
            prev_day_gap_up = bool(prev_day_gap_pct > (0.33 * adr_pct_val))

    # Institutional RS Score & Structural Posture (1-99 RS rating, Stage 2, 52W High distance)
    rs_score = 50.0
    stage2 = False
    dist_52w_high = -20.0
    try:
        rs_file = Path(f"data/rs_history/rs_{dt_str[:4]}.parquet")
        if rs_file.exists():
            rs_df = pd.read_parquet(
                rs_file,
                filters=[("symbol", "==", sym), ("date", "==", pd.Timestamp(dt_str))],
                columns=["rs_score", "stage2", "dist_52w_high"]
            )
            if not rs_df.empty:
                rs_score = float(rs_df["rs_score"].iloc[0])
                stage2 = bool(rs_df["stage2"].iloc[0])
                dist_52w_high = float(rs_df["dist_52w_high"].iloc[0])
    except Exception:
        pass
    if rs_score == 50.0 and pos >= 0:
        try:
            c = float(d["close"].iloc[pos])
            h52 = float(d["high"].iloc[max(0, pos - 252):pos + 1].max()) if pos > 0 else c
            dist_52w_high = float((c / h52 - 1.0) * 100.0) if h52 > 0 else 0.0
            if "ema10" in d.columns and "ema20" in d.columns and "sma50" in d.columns and "sma200" in d.columns:
                c_val = float(d["close"].iloc[pos])
                e10 = float(d["ema10"].iloc[pos])
                e20 = float(d["ema20"].iloc[pos])
                s50 = float(d["sma50"].iloc[pos])
                s200 = float(d["sma200"].iloc[pos])
                if pd.notna(e10) and pd.notna(e20) and pd.notna(s50) and pd.notna(s200):
                    stage2 = bool(c_val >= e10 >= e20 >= s50 >= s200)
        except Exception:
            pass

    # Forward tracking window
    fwd = d.iloc[pos:].copy()
    n_fwd = len(fwd)

    violated_48h = False
    if n_fwd >= 2:
        d2_low = float(fwd["low"].iloc[1])
        d1_midpoint = (d1_open + d1_close) / 2.0
        violated_48h = bool(d2_low < d1_midpoint)

    w5_fwd = fwd.iloc[1:min(n_fwd, 6)]
    breached_d1_low_5d = bool(w5_fwd["low"].min() < d1_low) if len(w5_fwd) > 0 else False
    mfe_5d_pct = float((w5_fwd["high"].max() / d1_close - 1.0) * 100.0) if len(w5_fwd) > 0 else 0.0
    mae_5d_pct = float((w5_fwd["low"].min() / d1_close - 1.0) * 100.0) if len(w5_fwd) > 0 else 0.0
    close_5d_pct = float((fwd["close"].iloc[min(5, n_fwd - 1)] / d1_close - 1.0) * 100.0)

    ret_10d = float((fwd["close"].iloc[10] / d1_close - 1.0) * 100.0) if n_fwd > 10 else np.nan
    ret_20d = float((fwd["close"].iloc[20] / d1_close - 1.0) * 100.0) if n_fwd > 20 else np.nan
    ret_60d = float((fwd["close"].iloc[60] / d1_close - 1.0) * 100.0) if n_fwd > 60 else np.nan
    mfe_60d_pct = float((fwd["high"].iloc[:min(n_fwd, 61)].max() / d1_close - 1.0) * 100.0)

    # Long-term horizons
    ret_120d = float((fwd["close"].iloc[120] / d1_close - 1.0) * 100.0) if n_fwd > 120 else np.nan
    mfe_120d = float((fwd["high"].iloc[:min(n_fwd, 121)].max() / d1_close - 1.0) * 100.0) if n_fwd > 120 else np.nan
    ret_180d = float((fwd["close"].iloc[180] / d1_close - 1.0) * 100.0) if n_fwd > 180 else np.nan
    mfe_180d = float((fwd["high"].iloc[:min(n_fwd, 181)].max() / d1_close - 1.0) * 100.0) if n_fwd > 180 else np.nan
    ret_250d = float((fwd["close"].iloc[250] / d1_close - 1.0) * 100.0) if n_fwd > 250 else np.nan
    mfe_250d = float((fwd["high"].iloc[:min(n_fwd, 251)].max() / d1_close - 1.0) * 100.0) if n_fwd > 250 else np.nan

    max_gain = float((fwd["high"].max() / d1_close - 1.0) * 100.0)
    hit_50pct = bool(max_gain >= 50.0)
    hit_100pct = bool(max_gain >= 100.0)
    hit_200pct = bool(max_gain >= 200.0)
    hit_300pct = bool(max_gain >= 300.0)

    # EMA / SMA fractions
    above_ema20_frac = np.nan
    if "ema20" in fwd.columns:
        above_ema20_frac = float((fwd["close"] > fwd["ema20"]).mean())
    above_sma50_frac = np.nan
    if "sma50" in fwd.columns:
        above_sma50_frac = float((fwd["close"] > fwd["sma50"]).mean())

    # Outcome classification
    if breached_d1_low_5d:
        outcome = "Gap & Crap Trap"
    elif hit_50pct or max_gain >= 30.0:
        outcome = "Runaway Monster"
    elif mfe_60d_pct >= 15.0 or (pd.notna(ret_20d) and ret_20d > 5.0):
        outcome = "Digestion & 2nd Break"
    else:
        outcome = "Fade to Black / Churn"

    # ML Rolling Conviction Features & Predictions
    t_idx = min(len(d) - 1, pos + 4)
    ml_feats = engine.compute_rolling_features(sym, pos, t_idx, df=d, allow_stopped=True)
    ml_preds = engine.predict_rolling(ml_feats) if ml_feats else {}

    ml_50 = float(ml_preds.get("prob_50", np.nan))
    ml_100 = float(ml_preds.get("prob_100", np.nan))
    ml_150 = float(ml_preds.get("prob_150", np.nan))
    ml_200 = float(ml_preds.get("prob_200", np.nan))
    dynamic_stop_loss_pct = float(ml_preds.get("dynamic_stop_loss_pct", np.nan))
    is_toxic = bool(ml_preds.get("is_toxic", False))

    # Dossier Trade Simulations across combinations
    ev_meta = {
        "sector": sec_name, "theme": thm_name, "sec_m1_pctile": sec_m1_pctile,
        "sec_m3_pctile": sec_m3_pctile, "thm_m1_pctile": thm_m1_pctile,
        "gap_pct": gap_pct, "rvol": rvol, "dvol_m": dvol_m, "close_pos": close_pos,
        "violated_48h": violated_48h, "dynamic_stop_loss_pct": dynamic_stop_loss_pct
    }

    # Delayed Breakout combinations
    dos_del_base = serve_ep.compute_dossier(sym, dt_str, d, pos, ev_meta, t1_mode="delayed", use_pyramid=False, use_dynamic_trailer=False, use_climax_exit=False)
    dos_del_pyr  = serve_ep.compute_dossier(sym, dt_str, d, pos, ev_meta, t1_mode="delayed", use_pyramid=True,  use_dynamic_trailer=False, use_climax_exit=False)
    dos_del_clx  = serve_ep.compute_dossier(sym, dt_str, d, pos, ev_meta, t1_mode="delayed", use_pyramid=False, use_dynamic_trailer=False, use_climax_exit=True)
    dos_del_dyn  = serve_ep.compute_dossier(sym, dt_str, d, pos, ev_meta, t1_mode="delayed", use_pyramid=False, use_dynamic_trailer=True,  use_climax_exit=False)
    dos_del_cd   = serve_ep.compute_dossier(sym, dt_str, d, pos, ev_meta, t1_mode="delayed", use_pyramid=False, use_dynamic_trailer=True,  use_climax_exit=True)
    dos_del_dp   = serve_ep.compute_dossier(sym, dt_str, d, pos, ev_meta, t1_mode="delayed", use_pyramid=True,  use_dynamic_trailer=True,  use_climax_exit=False)
    dos_del_cp   = serve_ep.compute_dossier(sym, dt_str, d, pos, ev_meta, t1_mode="delayed", use_pyramid=True,  use_dynamic_trailer=False, use_climax_exit=True)
    dos_del_opt  = serve_ep.compute_dossier(sym, dt_str, d, pos, ev_meta, t1_mode="delayed", use_pyramid=True,  use_dynamic_trailer=True,  use_climax_exit=True)

    tr1_del = dos_del_base["trade_1"]
    has_delayed_entry = not tr1_del["skipped"]
    t1_delayed_r = float(tr1_del.get("trade_r", tr1_del.get("r_mult", 0.0)))
    t1_delayed_shelf_r = t1_delayed_r
    t1_delayed_entry_px = float(tr1_del.get("entry_price", 0.0))
    t1_delayed_stop_px = float(tr1_del.get("stop_price", 0.0))
    t1_delayed_risk_pct = float(tr1_del.get("risk_pct", 0.0))
    t1_delayed_prob_50 = round(ml_50, 3) if pd.notna(ml_50) else 0.0
    t1_delayed_exit_reason = str(tr1_del.get("exit_reason", ""))

    t1_delayed_pyramid_r = float(dos_del_pyr["trade_1"].get("trade_r", dos_del_pyr["trade_1"].get("r_mult", 0.0)))
    t1_delayed_climax_r = float(dos_del_clx["trade_1"].get("trade_r", dos_del_clx["trade_1"].get("r_mult", 0.0)))
    t1_delayed_dyn_r = float(dos_del_dyn["trade_1"].get("trade_r", dos_del_dyn["trade_1"].get("r_mult", 0.0)))
    t1_delayed_climax_dyn_r = float(dos_del_cd["trade_1"].get("trade_r", dos_del_cd["trade_1"].get("r_mult", 0.0)))
    t1_delayed_dyn_pyramid_r = float(dos_del_dp["trade_1"].get("trade_r", dos_del_dp["trade_1"].get("r_mult", 0.0)))
    t1_delayed_climax_pyramid_r = float(dos_del_cp["trade_1"].get("trade_r", dos_del_cp["trade_1"].get("r_mult", 0.0)))
    t1_delayed_ai_opt_r = float(dos_del_opt["trade_1"].get("trade_r", dos_del_opt["trade_1"].get("r_mult", 0.0)))

    # Day 1 Close combinations (baseline)
    dos_d1_base = serve_ep.compute_dossier(sym, dt_str, d, pos, ev_meta, t1_mode="d1_close", use_pyramid=False, use_dynamic_trailer=False, use_climax_exit=False)
    dos_d1_pyr  = serve_ep.compute_dossier(sym, dt_str, d, pos, ev_meta, t1_mode="d1_close", use_pyramid=True,  use_dynamic_trailer=False, use_climax_exit=False)
    dos_d1_clx  = serve_ep.compute_dossier(sym, dt_str, d, pos, ev_meta, t1_mode="d1_close", use_pyramid=False, use_dynamic_trailer=False, use_climax_exit=True)
    dos_d1_dyn  = serve_ep.compute_dossier(sym, dt_str, d, pos, ev_meta, t1_mode="d1_close", use_pyramid=False, use_dynamic_trailer=True,  use_climax_exit=False)
    dos_d1_cd   = serve_ep.compute_dossier(sym, dt_str, d, pos, ev_meta, t1_mode="d1_close", use_pyramid=False, use_dynamic_trailer=True,  use_climax_exit=True)
    dos_d1_dp   = serve_ep.compute_dossier(sym, dt_str, d, pos, ev_meta, t1_mode="d1_close", use_pyramid=True,  use_dynamic_trailer=True,  use_climax_exit=False)
    dos_d1_cp   = serve_ep.compute_dossier(sym, dt_str, d, pos, ev_meta, t1_mode="d1_close", use_pyramid=True,  use_dynamic_trailer=False, use_climax_exit=True)
    dos_d1_cdp  = serve_ep.compute_dossier(sym, dt_str, d, pos, ev_meta, t1_mode="d1_close", use_pyramid=True,  use_dynamic_trailer=True,  use_climax_exit=True)

    tr1_d1 = dos_d1_base["trade_1"]
    t1_real_r = float(tr1_d1.get("trade_r", tr1_d1.get("r_mult", 0.0)))
    t1_real_ret = float(tr1_d1.get("trade_return_pct", tr1_d1.get("return_pct", 0.0)))
    t1_real_risk_pct = float(tr1_d1.get("risk_pct", 0.0))
    t1_real_exit_reason = str(tr1_d1.get("exit_reason", ""))
    t1_real_hold_days = int(tr1_d1.get("hold_days", 0))

    t1_pyramid_r = float(dos_d1_pyr["trade_1"].get("trade_r", dos_d1_pyr["trade_1"].get("r_mult", 0.0)))
    t1_climax_r = float(dos_d1_clx["trade_1"].get("trade_r", dos_d1_clx["trade_1"].get("r_mult", 0.0)))
    t1_climax_triggered = bool(dos_d1_clx["trade_1"].get("climax_exit_triggered", False))
    t1_dyn_r = float(dos_d1_dyn["trade_1"].get("trade_r", dos_d1_dyn["trade_1"].get("r_mult", 0.0)))
    t1_climax_dyn_r = float(dos_d1_cd["trade_1"].get("trade_r", dos_d1_cd["trade_1"].get("r_mult", 0.0)))
    t1_climax_pyramid_r = float(dos_d1_cp["trade_1"].get("trade_r", dos_d1_cp["trade_1"].get("r_mult", 0.0)))
    t1_dyn_pyramid_r = float(dos_d1_dp["trade_1"].get("trade_r", dos_d1_dp["trade_1"].get("r_mult", 0.0)))
    t1_climax_dyn_pyramid_r = float(dos_d1_cdp["trade_1"].get("trade_r", dos_d1_cdp["trade_1"].get("r_mult", 0.0)))

    # Multi-Leg Continuations (Trade 2 & 3)
    tr2 = dos_del_base["trade_2"]
    t2_has_reentry = bool(tr2.get("has_reentry", False))
    t2_real_r = float(tr2.get("r_mult", 0.0))
    t2_prob_reentry = float(tr2.get("ml_prob", 0.5)) if tr2.get("ml_prob") is not None else 0.5
    t2_track = str(tr2.get("track", "none"))

    tr3 = dos_del_base["trade_3"]
    t3_has_reentry = bool(tr3.get("has_reentry", False))
    t3_real_r = float(tr3.get("r_mult", 0.0))
    t3_prob_reentry = float(tr3.get("ml_prob", 0.0)) if tr3.get("ml_prob") is not None else 0.0

    return {
        "symbol": sym,
        "date": dt_str,
        "year": int(dt_str[:4]),
        "subtype": subtype,
        "sector": sec_name,
        "theme": thm_name,
        "gap_pct": round(gap_pct, 1),
        "rvol": round(rvol, 1),
        "dvol_m": round(dvol_m, 1),
        "close_pos": round(close_pos, 2),
        "neglect_6m": round(neglect_6m, 1),
        "sec_m1_pctile": round(sec_m1_pctile, 1),
        "sec_m3_pctile": round(sec_m3_pctile, 1),
        "thm_m1_pctile": round(thm_m1_pctile, 1),
        "violated_48h": violated_48h,
        "breached_d1_low_5d": breached_d1_low_5d,
        "mfe_5d_pct": round(mfe_5d_pct, 1),
        "mae_5d_pct": round(mae_5d_pct, 1),
        "close_5d_pct": round(close_5d_pct, 1),
        "ret_10d": round(ret_10d, 1) if pd.notna(ret_10d) else np.nan,
        "ret_20d": round(ret_20d, 1) if pd.notna(ret_20d) else np.nan,
        "ret_60d": round(ret_60d, 1) if pd.notna(ret_60d) else np.nan,
        "mfe_60d_pct": round(mfe_60d_pct, 1),
        "above_ema20_frac": round(above_ema20_frac, 2) if pd.notna(above_ema20_frac) else np.nan,
        "outcome": outcome,
        "ret_120d": round(ret_120d, 1) if pd.notna(ret_120d) else np.nan,
        "mfe_120d": round(mfe_120d, 1) if pd.notna(mfe_120d) else np.nan,
        "ret_180d": round(ret_180d, 1) if pd.notna(ret_180d) else np.nan,
        "mfe_180d": round(mfe_180d, 1) if pd.notna(mfe_180d) else np.nan,
        "ret_250d": round(ret_250d, 1) if pd.notna(ret_250d) else np.nan,
        "mfe_250d": round(mfe_250d, 1) if pd.notna(mfe_250d) else np.nan,
        "hit_50pct": hit_50pct,
        "hit_100pct": hit_100pct,
        "hit_200pct": hit_200pct,
        "hit_300pct": hit_300pct,
        "max_gain": round(max_gain, 1),
        "above_sma50_frac": round(above_sma50_frac, 2) if pd.notna(above_sma50_frac) else np.nan,
        "ml_50": ml_50,
        "ml_100": ml_100,
        "ml_150": ml_150,
        "ml_200": ml_200,
        "is_toxic": is_toxic,
        "dynamic_stop_loss_pct": dynamic_stop_loss_pct,
        "t1_real_r": round(t1_real_r, 3),
        "t1_real_ret": round(t1_real_ret, 2),
        "t1_real_risk_pct": round(t1_real_risk_pct, 2),
        "t1_real_exit_reason": t1_real_exit_reason,
        "t1_real_hold_days": t1_real_hold_days,
        "t1_pyramid_r": round(t1_pyramid_r, 3),
        "t2_has_reentry": t2_has_reentry,
        "t2_real_r": round(t2_real_r, 2),
        "t2_prob_reentry": t2_prob_reentry,
        "t3_has_reentry": t3_has_reentry,
        "t3_real_r": round(t3_real_r, 2),
        "t3_prob_reentry": t3_prob_reentry,
        "t2_track": t2_track,
        "t1_delayed_has_entry": has_delayed_entry,
        "t1_delayed_r": round(t1_delayed_r, 3),
        "t1_delayed_shelf_r": round(t1_delayed_shelf_r, 3),
        "t1_delayed_pyramid_r": round(t1_delayed_pyramid_r, 3),
        "t1_delayed_entry_px": round(t1_delayed_entry_px, 2),
        "t1_delayed_stop_px": round(t1_delayed_stop_px, 2),
        "t1_delayed_risk_pct": round(t1_delayed_risk_pct, 1),
        "t1_delayed_prob_50": round(t1_delayed_prob_50, 3),
        "t1_delayed_exit_reason": t1_delayed_exit_reason,
        "t1_climax_r": round(t1_climax_r, 3),
        "t1_climax_triggered": t1_climax_triggered,
        "t1_dyn_r": round(t1_dyn_r, 3),
        "t1_climax_dyn_r": round(t1_climax_dyn_r, 3),
        "t1_delayed_climax_r": round(t1_delayed_climax_r, 3),
        "t1_delayed_dyn_r": round(t1_delayed_dyn_r, 3),
        "t1_delayed_climax_dyn_r": round(t1_delayed_climax_dyn_r, 3),
        "t1_delayed_dyn_pyramid_r": round(t1_delayed_dyn_pyramid_r, 3),
        "t1_delayed_ai_opt_r": round(t1_delayed_ai_opt_r, 3),
        "t1_climax_pyramid_r": round(t1_climax_pyramid_r, 3),
        "t1_dyn_pyramid_r": round(t1_dyn_pyramid_r, 3),
        "t1_climax_dyn_pyramid_r": round(t1_climax_dyn_pyramid_r, 3),
        "t1_delayed_climax_pyramid_r": round(t1_delayed_climax_pyramid_r, 3),
        "adr_pct": round(adr_pct_val, 2),
        "prev_day_gap_pct": round(prev_day_gap_pct, 2),
        "prev_day_gap_up": prev_day_gap_up,
        "is_opex": trading_calendar.is_monthly_opex(dt_str),
        "is_quad_witching": trading_calendar.is_quad_witching(dt_str),
        "is_russell_recon": trading_calendar.is_russell_reconstitution(dt_str),
        "is_month_end": trading_calendar.is_month_end(dt_str),
        "is_quarter_end": trading_calendar.is_quarter_end(dt_str),
        "is_mechanical_volume_date": trading_calendar.is_mechanical_volume_date(dt_str),
        "calendar_anomaly": trading_calendar.get_calendar_anomaly_type(dt_str),
        "rs_score": round(rs_score, 1),
        "stage2": stage2,
        "dist_52w_high": round(dist_52w_high, 2)
    }


def update_pipeline(since: str | None = None, skip_download: bool = False, full_download: bool = False, dry_run: bool = False):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Baseline dataset not found at {DATA_PATH}")

    df_orig = pd.read_parquet(DATA_PATH)
    latest_date_in_df = str(df_orig["date"].max())
    print(f"📊 Baseline Dataset: {len(df_orig)} events across {df_orig['symbol'].nunique()} symbols. Cutoff: {latest_date_in_df}")

    default_lookback = (dt.date.today() - dt.timedelta(days=45)).strftime("%Y-%m-%d")
    target_since = since if since else min(latest_date_in_df, default_lookback)

    # 1. Download bars
    if not skip_download:
        download_latest_bars(full=full_download)
    else:
        print("⏭️ Skipping bar download as requested.")

    # 2. Scan active universe
    fresh_eps = scan_for_new_eps(since_date=target_since)
    if not fresh_eps:
        print(f"\n🎉 Dataset is already up to date! No new EP events found after {target_since}.")
        return

    # 3. Build records
    print("\n" + "=" * 80)
    print(f"⚙️ STEP 3: COMPUTING ML FEATURES & TRADE SIMULATIONS FOR {len(fresh_eps)} EVENTS")
    print("=" * 80)

    thm_engine = thematic_engine.get_thematic_engine()
    new_rows = []
    t0 = time.time()

    for idx, (sym, dt_str, subtype) in enumerate(fresh_eps):
        try:
            row = build_ep_row(sym, dt_str, subtype, thm_engine)
            if row:
                new_rows.append(row)
                print(f"  [{idx+1}/{len(fresh_eps)}] {sym:6s} | {dt_str} | Conviction: {row['ml_50']*100:.1f}% | Delayed T1: {row['t1_delayed_r']:+.2f}R ({row['t1_delayed_exit_reason']})")
        except Exception as e:
            print(f"  [{idx+1}/{len(fresh_eps)}] ⚠️ Error processing {sym} on {dt_str}: {e}")

    if not new_rows:
        print("❌ No valid rows constructed.")
        return

    df_new = pd.DataFrame(new_rows)
    print(f"\n✅ Constructed {len(df_new)} new EP event rows in {time.time() - t0:.1f}s.")

    if dry_run:
        print("🔍 Dry run complete. Not modifying parquet file.")
        return

    # 4. Backup & Merge
    print("\n" + "=" * 80)
    print("💾 STEP 4: MERGING & ATOMICALLY SAVING TO PARQUET")
    print("=" * 80)

    backup_path = DATA_PATH.parent / f"ep_combined_study_scored_backup_{dt.date.today().strftime('%Y%m%d')}.parquet"
    if not backup_path.exists():
        shutil.copy2(DATA_PATH, backup_path)
        print(f"📦 Created safe backup at: {backup_path}")

    # Reindex df_new to exactly match df_orig schema
    df_new = df_new.reindex(columns=df_orig.columns)
    combined = pd.concat([df_orig, df_new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["symbol", "date"], keep="last")
    combined = combined.sort_values(by=["date", "symbol"], ascending=[False, True]).reset_index(drop=True)

    tmp_path = DATA_PATH.with_suffix(".tmp.parquet")
    combined.to_parquet(tmp_path, index=False)
    tmp_path.replace(DATA_PATH)

    new_cutoff = str(combined["date"].max())
    print(f"🚀 Success! Dataset updated from {len(df_orig)} to {len(combined)} events. New cutoff: {new_cutoff}")

    # 5. Reload server
    print("\n" + "=" * 80)
    print("🔄 STEP 5: RELOADING ACTIVE SERVER ON PORT 8782")
    print("=" * 80)
    try:
        resp = requests.post("http://127.0.0.1:8782/api/reload_dataset", timeout=5)
        if resp.status_code == 200:
            info = resp.json()
            print(f"✨ Server reloaded successfully! Active events: {info.get('rows')}, Max Date: {info.get('max_date')}")
        else:
            print(f"Server returned status {resp.status_code}.")
    except Exception as e:
        print(f"Note: Could not reach review server on port 8782 ({e}). It will load updated dataset on next start.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update EP Historical Dataset to Present Day")
    parser.add_argument("--since", type=str, default=None, help="Scan since specific date (YYYY-MM-DD)")
    parser.add_argument("--skip-bars-download", action="store_true", help="Skip downloading price bars from yfinance")
    parser.add_argument("--full-bars", action="store_true", help="Download full bar refresh across entire universe")
    parser.add_argument("--dry-run", action="store_true", help="Run scan and feature calculation without modifying parquet")
    args = parser.parse_args()

    update_pipeline(
        since=args.since,
        skip_download=args.skip_bars_download,
        full_download=args.full_bars,
        dry_run=args.dry_run
    )
