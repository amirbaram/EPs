"""Dual-Track Continuation Simulation & Comparison Engine.

Evaluates Trade 2 / Trade 3 performance across:
1. All 1,689 historical Day 1 low breaches (Track 1 focus: Institutional Undercut & Reclaim)
2. The full EP universe (6,501 events) before and after Dual-Track Continuation Architecture.

Zero-lookahead execution:
- Signal on bar B, Entry on B+1 at Open.
- Realistic gap-down stop loss execution.
- Day 1 entry bar stop loss execution.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import datastore
import scanner_core


def run_dual_track_simulation(ribbon_spans=(8, 12, 16, 21), max_eps: int | None = None):
    study_path = Path("data/simulations/ep_combined_study_scored.parquet")
    backup_path = Path("data/simulations/ep_combined_study_scored_pre_dual_track.parquet")
    source_path = backup_path if backup_path.exists() else study_path
    print(f"Loading baseline EP dataset from {source_path}...")
    df_ep = pd.read_parquet(source_path)
    if max_eps:
        df_ep = df_ep.head(max_eps)

    total_eps = len(df_ep)
    print(f"Loaded {total_eps} EP events. Running Dual-Track Continuation Engine...")

    t0 = time.time()
    results = []

    for i, (_, row) in enumerate(df_ep.iterrows()):
        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i}/{total_eps} EPs ({time.time() - t0:.1f}s)...")

        sym = row["symbol"]
        date_str = str(row["date"])
        raw = datastore.load_bars(sym)
        if raw is None or len(raw) < 30:
            continue

        # Ensure intermediate ribbon is present
        d = scanner_core.calc_intermediate_ribbon(raw, spans=ribbon_spans)
        if "sma50" not in d.columns:
            d["sma50"] = d["close"].rolling(50).mean()

        ts = pd.Timestamp(date_str)
        if ts not in d.index:
            continue
        pos = d.index.get_loc(ts)
        if isinstance(pos, (slice, np.ndarray)):
            pos = pos[0] if isinstance(pos, np.ndarray) else pos.start
        if pos >= len(d) - 2:
            continue

        fwd = d.iloc[pos:min(len(d), pos + 250)].copy()
        if len(fwd) < 3:
            continue

        d1 = fwd.iloc[0]
        d1_close = float(d1["close"])
        d1_low = float(d1["low"])
        d1_high = float(d1["high"])
        risk1_pct = (d1_close - d1_low) / d1_close * 100.0 if d1_close > 0 else 11.1

        # --- SIMULATE TRADE 1 ---
        t1_stopped = False
        t1_exit_bar = None
        t1_exit_price = None
        t1_peak = d1_close
        seen_bull = False

        for b in range(1, len(fwd)):
            c = float(fwd["close"].iloc[b])
            l = float(fwd["low"].iloc[b])
            h = float(fwd["high"].iloc[b])
            b_open = float(fwd["open"].iloc[b])
            r_state = fwd["ribbon_state"].iloc[b]
            if h > t1_peak:
                t1_peak = h
            if pd.notna(r_state) and r_state in ["yellow", "gray"]:
                seen_bull = True

            # Hard stop check at Day 1 low (with gap down zero-lookahead fill)
            if l <= d1_low:
                t1_stopped = True
                t1_exit_bar = b
                t1_exit_price = b_open if b_open < d1_low else d1_low
                t1_reason = "Day 1 Low Stop Hit"
                break

            # Intermediate ribbon reverse flip exit (Blue)
            if seen_bull and pd.notna(r_state) and r_state == "blue":
                t1_exit_bar = b
                t1_exit_price = c
                t1_reason = "Ribbon Reverse Flip Exit (Blue)"
                break

            # 50 SMA Breakdown
            sma50 = fwd["sma50"].iloc[b]
            if seen_bull and pd.notna(sma50) and c < sma50:
                t1_exit_bar = b
                t1_exit_price = c
                t1_reason = "50 SMA Breakdown"
                break

        if t1_exit_bar is None:
            t1_exit_bar = len(fwd) - 1
            t1_exit_price = float(fwd["close"].iloc[-1])
            t1_reason = "Active / Window End"

        t1_ret = (t1_exit_price / d1_close - 1.0) * 100.0
        t1_r = t1_ret / risk1_pct if risk1_pct > 0 else 0.0

        # --- SIMULATE TRADE 2 (DUAL-TRACK) ---
        trade_2 = {"has_reentry": False}
        t2_exit_bar = None
        t2_r = 0.0
        t2_track = None

        # Track 1 Evaluation: Institutional Undercut & Reclaim (U&R)
        # Condition: Trade 1 stopped out at Day 1 Low within first 10 sessions
        ur_signal_bar = None
        ur_stop = None

        if t1_stopped and 1 <= t1_exit_bar <= 10:
            shakeout_low = float(fwd["low"].iloc[t1_exit_bar])
            controlled_ur = (shakeout_low >= d1_low * 0.85)

            # Scan subsequent sessions up to 15 sessions post stop-out
            if controlled_ur:
                ur_end_search = min(len(fwd), t1_exit_bar + 16)
                for b in range(t1_exit_bar + 1, ur_end_search):
                    l_b = float(fwd["low"].iloc[b])
                    c_b = float(fwd["close"].iloc[b])
                    if l_b < shakeout_low:
                        shakeout_low = l_b

                    # Controlled check: undercut low must stay within 15% of Day 1 Low
                    if shakeout_low < d1_low * 0.85:
                        controlled_ur = False
                        break

                    # Entry trigger: close back ABOVE Day 1 Low
                    if c_b >= d1_low:
                        ur_signal_bar = b
                        ur_stop = round(shakeout_low, 2)
                        break

            if controlled_ur and ur_signal_bar is not None:
                entry_bar = ur_signal_bar + 1
                if entry_bar < len(fwd):
                    t2_entry = float(fwd["open"].iloc[entry_bar])
                    t2_risk_pct = (t2_entry - ur_stop) / t2_entry * 100.0

                    if 1.0 <= t2_risk_pct <= 35.0:
                        t2_stopped = False
                        t2_exit_price = None

                        # Zero-lookahead Day 1 check on entry bar
                        day1_low = float(fwd["low"].iloc[entry_bar])
                        day1_open = float(fwd["open"].iloc[entry_bar])
                        if day1_low <= ur_stop:
                            t2_stopped = True
                            t2_exit_bar = entry_bar
                            t2_exit_price = day1_open if day1_open < ur_stop else ur_stop
                            t2_reason = "Undercut Low Stop Hit (Day 1)"
                        else:
                            for b in range(entry_bar + 1, len(fwd)):
                                c = float(fwd["close"].iloc[b])
                                l = float(fwd["low"].iloc[b])
                                b_open = float(fwd["open"].iloc[b])
                                s = fwd["ribbon_state"].iloc[b]
                                sma50 = fwd["sma50"].iloc[b]

                                if l <= ur_stop:
                                    t2_stopped = True
                                    t2_exit_bar = b
                                    t2_exit_price = b_open if b_open < ur_stop else ur_stop
                                    t2_reason = "Undercut Low Stop Hit"
                                    break
                                if pd.notna(s) and s == "blue":
                                    t2_exit_bar = b
                                    t2_exit_price = c
                                    t2_reason = "Ribbon Reverse Flip Exit (Blue)"
                                    break
                                if pd.notna(sma50) and c < sma50:
                                    t2_exit_bar = b
                                    t2_exit_price = c
                                    t2_reason = "50 SMA Breakdown"
                                    break

                        if t2_exit_bar is None:
                            t2_exit_bar = len(fwd) - 1
                            t2_exit_price = float(fwd["close"].iloc[-1])
                            t2_reason = "Active / Window End"

                        t2_ret = (t2_exit_price / t2_entry - 1.0) * 100.0
                        t2_r = t2_ret / t2_risk_pct if t2_risk_pct > 0 else 0.0
                        t2_track = "track_1_ur"
                        trade_2 = {
                            "has_reentry": True,
                            "track": t2_track,
                            "signal_date": fwd.index[ur_signal_bar].strftime("%Y-%m-%d"),
                            "entry_date": fwd.index[entry_bar].strftime("%Y-%m-%d"),
                            "entry_price": round(t2_entry, 2),
                            "stop_price": round(ur_stop, 2),
                            "risk_pct": round(t2_risk_pct, 1),
                            "cons_days": int(ur_signal_bar - t1_exit_bar),
                            "drop_from_peak": round((d1_low - ur_stop) / d1_low * 100.0, 1),
                            "exit_date": fwd.index[t2_exit_bar].strftime("%Y-%m-%d"),
                            "exit_price": round(t2_exit_price, 2),
                            "exit_reason": t2_reason,
                            "stopped_out": bool(t2_stopped),
                            "hold_days": int(t2_exit_bar - entry_bar),
                            "return_pct": round(t2_ret, 1),
                            "r_mult": round(t2_r, 2),
                            "combined_net_r": round(t1_r + t2_r, 2),
                            "ribbon_spans": list(ribbon_spans),
                        }

        # Track 2 Evaluation: Corrected Dynamic Base Breakout / Intermediate Ribbon
        # Evaluated if Track 1 did not take a trade
        if not trade_2["has_reentry"]:
            seen_cons = False
            reentry_signal_bar = None
            cons_low = float(fwd["low"].iloc[t1_exit_bar])
            curr_peak = t1_peak

            search_end = min(len(fwd), t1_exit_bar + 66)
            for b in range(t1_exit_bar, search_end):
                l = float(fwd["low"].iloc[b])
                h = float(fwd["high"].iloc[b])
                c = float(fwd["close"].iloc[b])
                s = fwd["ribbon_state"].iloc[b]
                sma50 = fwd["sma50"].iloc[b]

                # Dynamic peak-to-trough anchor fix:
                # Reset cons_low dynamically whenever a new peak is reached
                if h > curr_peak:
                    curr_peak = h
                    cons_low = l
                elif l < cons_low:
                    cons_low = l

                if pd.notna(s) and s in ["blue", "gray"]:
                    seen_cons = True

                # Retracement from current peak
                pullback_pct = (curr_peak - cons_low) / curr_peak * 100.0 if curr_peak > 0 else 0.0

                # Trigger 1: Classic Yellow Flip after Blue/Gray consolidation
                if seen_cons and b > t1_exit_bar and pd.notna(s) and s == "yellow":
                    reentry_signal_bar = b
                    break

                # Trigger 2: Persistent Yellow Shallow Pullback Re-entry (>=5% pullback, 5D high breakout or EMA8 reclaim)
                if not seen_cons and b > t1_exit_bar + 2 and pullback_pct >= 5.0 and pd.notna(s) and s == "yellow":
                    prev_5d_high = float(fwd["high"].iloc[max(0, b - 5):b].max())
                    ema8 = float(fwd[f"ribbon_ema{ribbon_spans[0]}"].iloc[b]) if f"ribbon_ema{ribbon_spans[0]}" in fwd.columns else None
                    prev_c = float(fwd["close"].iloc[b - 1])
                    prev_ema8 = float(fwd[f"ribbon_ema{ribbon_spans[0]}"].iloc[b - 1]) if f"ribbon_ema{ribbon_spans[0]}" in fwd.columns else None

                    reclaimed_ema8 = (ema8 is not None and prev_ema8 is not None and prev_c <= prev_ema8 and c > ema8)
                    broke_5d_high = c > prev_5d_high

                    if broke_5d_high or reclaimed_ema8:
                        reentry_signal_bar = b
                        break

            if reentry_signal_bar is not None:
                c_sig = float(fwd["close"].iloc[reentry_signal_bar])
                sma50_sig = fwd["sma50"].iloc[reentry_signal_bar] if "sma50" in fwd.columns else None
                held_50 = pd.isna(sma50_sig) or c_sig >= sma50_sig
                elapsed_days = int(reentry_signal_bar - t1_exit_bar)
                allowed_cons = (elapsed_days <= 65 if held_50 else elapsed_days <= 45)

                if allowed_cons:
                    drop_from_peak = (curr_peak - cons_low) / curr_peak * 100.0 if curr_peak > 0 else 0.0
                    if drop_from_peak <= 55.0:
                        swing5_low = float(fwd["low"].iloc[max(0, reentry_signal_bar - 4):reentry_signal_bar + 1].min())
                        t2_stop = round(swing5_low, 2)
                        entry_bar = reentry_signal_bar + 1

                        if entry_bar < len(fwd):
                            t2_entry = float(fwd["open"].iloc[entry_bar])
                            t2_risk_pct = (t2_entry - t2_stop) / t2_entry * 100.0

                            if 1.0 <= t2_risk_pct <= 35.0:
                                t2_stopped = False
                                t2_exit_price = None

                                day1_low = float(fwd["low"].iloc[entry_bar])
                                day1_open = float(fwd["open"].iloc[entry_bar])
                                if day1_low <= t2_stop:
                                    t2_stopped = True
                                    t2_exit_bar = entry_bar
                                    t2_exit_price = day1_open if day1_open < t2_stop else t2_stop
                                    t2_reason = "Swing Low Stop Hit (Day 1)"
                                else:
                                    for b in range(entry_bar + 1, len(fwd)):
                                        c = float(fwd["close"].iloc[b])
                                        l = float(fwd["low"].iloc[b])
                                        b_open = float(fwd["open"].iloc[b])
                                        s = fwd["ribbon_state"].iloc[b]
                                        sma50 = fwd["sma50"].iloc[b]
                                        if l <= t2_stop:
                                            t2_stopped = True
                                            t2_exit_bar = b
                                            t2_exit_price = b_open if b_open < t2_stop else t2_stop
                                            t2_reason = "Swing Low Stop Hit"
                                            break
                                        if pd.notna(s) and s == "blue":
                                            t2_exit_bar = b
                                            t2_exit_price = c
                                            t2_reason = "Ribbon Reverse Flip Exit (Blue)"
                                            break
                                        if pd.notna(sma50) and c < sma50:
                                            t2_exit_bar = b
                                            t2_exit_price = c
                                            t2_reason = "50 SMA Breakdown"
                                            break

                                if t2_exit_bar is None:
                                    t2_exit_bar = len(fwd) - 1
                                    t2_exit_price = float(fwd["close"].iloc[-1])
                                    t2_reason = "Active / Window End"

                                t2_ret = (t2_exit_price / t2_entry - 1.0) * 100.0
                                t2_r = t2_ret / t2_risk_pct if t2_risk_pct > 0 else 0.0
                                t2_track = "track_2_ribbon"

                                trade_2 = {
                                    "has_reentry": True,
                                    "track": t2_track,
                                    "signal_date": fwd.index[reentry_signal_bar].strftime("%Y-%m-%d"),
                                    "entry_date": fwd.index[entry_bar].strftime("%Y-%m-%d"),
                                    "entry_price": round(t2_entry, 2),
                                    "stop_price": round(t2_stop, 2),
                                    "risk_pct": round(t2_risk_pct, 1),
                                    "cons_days": elapsed_days,
                                    "drop_from_peak": round(drop_from_peak, 1),
                                    "exit_date": fwd.index[t2_exit_bar].strftime("%Y-%m-%d"),
                                    "exit_price": round(t2_exit_price, 2),
                                    "exit_reason": t2_reason,
                                    "stopped_out": bool(t2_stopped),
                                    "hold_days": int(t2_exit_bar - entry_bar),
                                    "return_pct": round(t2_ret, 1),
                                    "r_mult": round(t2_r, 2),
                                    "combined_net_r": round(t1_r + t2_r, 2),
                                    "ribbon_spans": list(ribbon_spans),
                                }

        # --- SIMULATE TRADE 3 (LEG 3 CONTINUATION) ---
        trade_3 = {"has_reentry": False}
        t3_r = 0.0
        if trade_2.get("has_reentry") and t2_exit_bar is not None and t2_exit_bar < len(fwd) - 10:
            t2_entry_idx = entry_bar
            t2_highs = fwd["high"].iloc[t2_entry_idx:t2_exit_bar + 1]
            t2_peak = float(t2_highs.max())
            peak_offset = int(np.argmax(t2_highs.values))
            peak_bar = t2_entry_idx + peak_offset
            cons_low_3 = float(fwd["low"].iloc[peak_bar:t2_exit_bar + 1].min())

            seen_cons_3 = False
            reentry_signal_3 = None

            search_end_3 = min(len(fwd), t2_exit_bar + 66)
            for b in range(t2_exit_bar, search_end_3):
                l = float(fwd["low"].iloc[b])
                h = float(fwd["high"].iloc[b])
                c = float(fwd["close"].iloc[b])
                s = fwd["ribbon_state"].iloc[b]
                sma50 = fwd["sma50"].iloc[b]

                if h > t2_peak:
                    t2_peak = h
                    cons_low_3 = l
                elif l < cons_low_3:
                    cons_low_3 = l

                if pd.notna(s) and s in ["blue", "gray"]:
                    seen_cons_3 = True

                pullback_3 = (t2_peak - cons_low_3) / t2_peak * 100.0 if t2_peak > 0 else 0.0

                if seen_cons_3 and b > t2_exit_bar and pd.notna(s) and s == "yellow":
                    reentry_signal_3 = b
                    break

                if not seen_cons_3 and b > t2_exit_bar + 2 and pullback_3 >= 5.0 and pd.notna(s) and s == "yellow":
                    prev_5d_high_3 = float(fwd["high"].iloc[max(0, b - 5):b].max())
                    ema8_3 = float(fwd[f"ribbon_ema{ribbon_spans[0]}"].iloc[b]) if f"ribbon_ema{ribbon_spans[0]}" in fwd.columns else None
                    prev_c_3 = float(fwd["close"].iloc[b - 1])
                    prev_ema8_3 = float(fwd[f"ribbon_ema{ribbon_spans[0]}"].iloc[b - 1]) if f"ribbon_ema{ribbon_spans[0]}" in fwd.columns else None
                    if (ema8_3 and prev_ema8_3 and prev_c_3 <= prev_ema8_3 and c > ema8_3) or (c > prev_5d_high_3):
                        reentry_signal_3 = b
                        break

            if reentry_signal_3 is not None:
                c_sig_3 = float(fwd["close"].iloc[reentry_signal_3])
                sma50_sig_3 = fwd["sma50"].iloc[reentry_signal_3] if "sma50" in fwd.columns else None
                held_50_3 = pd.isna(sma50_sig_3) or c_sig_3 >= sma50_sig_3
                elapsed_days_3 = int(reentry_signal_3 - t2_exit_bar)
                allowed_cons_3 = (elapsed_days_3 <= 65 if held_50_3 else elapsed_days_3 <= 45)

                if allowed_cons_3:
                    drop_3 = (t2_peak - cons_low_3) / t2_peak * 100.0 if t2_peak > 0 else 0.0
                    if drop_3 <= 55.0:
                        swing5_3 = float(fwd["low"].iloc[max(0, reentry_signal_3 - 4):reentry_signal_3 + 1].min())
                        entry_bar_3 = reentry_signal_3 + 1
                        if entry_bar_3 < len(fwd):
                            t3_entry = float(fwd["open"].iloc[entry_bar_3])
                            t3_risk_pct = (t3_entry - swing5_3) / t3_entry * 100.0
                            if 1.0 <= t3_risk_pct <= 35.0:
                                t3_stopped = False
                                t3_exit_bar = None
                                t3_exit_price = None

                                day1_low_3 = float(fwd["low"].iloc[entry_bar_3])
                                day1_open_3 = float(fwd["open"].iloc[entry_bar_3])
                                if day1_low_3 <= swing5_3:
                                    t3_stopped = True
                                    t3_exit_bar = entry_bar_3
                                    t3_exit_price = day1_open_3 if day1_open_3 < swing5_3 else swing5_3
                                    t3_reason = "Swing Low Stop Hit (Day 1)"
                                else:
                                    for b in range(entry_bar_3 + 1, len(fwd)):
                                        c = float(fwd["close"].iloc[b])
                                        l = float(fwd["low"].iloc[b])
                                        b_open = float(fwd["open"].iloc[b])
                                        s = fwd["ribbon_state"].iloc[b]
                                        sma50 = fwd["sma50"].iloc[b]
                                        if l <= swing5_3:
                                            t3_stopped = True
                                            t3_exit_bar = b
                                            t3_exit_price = b_open if b_open < swing5_3 else swing5_3
                                            t3_reason = "Swing Low Stop Hit"
                                            break
                                        if pd.notna(s) and s == "blue":
                                            t3_exit_bar = b
                                            t3_exit_price = c
                                            t3_reason = "Ribbon Reverse Flip Exit (Blue)"
                                            break
                                        if pd.notna(sma50) and c < sma50:
                                            t3_exit_bar = b
                                            t3_exit_price = c
                                            t3_reason = "50 SMA Breakdown"
                                            break

                                if t3_exit_bar is None:
                                    t3_exit_bar = len(fwd) - 1
                                    t3_exit_price = float(fwd["close"].iloc[-1])
                                    t3_reason = "Active / Window End"

                                t3_ret = (t3_exit_price / t3_entry - 1.0) * 100.0
                                t3_r = t3_ret / t3_risk_pct if t3_risk_pct > 0 else 0.0
                                trade_3 = {
                                    "has_reentry": True,
                                    "track": "track_2_ribbon",
                                    "signal_date": fwd.index[reentry_signal_3].strftime("%Y-%m-%d"),
                                    "entry_date": fwd.index[entry_bar_3].strftime("%Y-%m-%d"),
                                    "entry_price": round(t3_entry, 2),
                                    "stop_price": round(swing5_3, 2),
                                    "risk_pct": round(t3_risk_pct, 1),
                                    "cons_days": elapsed_days_3,
                                    "drop_from_peak": round(drop_3, 1),
                                    "exit_date": fwd.index[t3_exit_bar].strftime("%Y-%m-%d"),
                                    "exit_price": round(t3_exit_price, 2),
                                    "exit_reason": t3_reason,
                                    "stopped_out": bool(t3_stopped),
                                    "hold_days": int(t3_exit_bar - entry_bar_3),
                                    "return_pct": round(t3_ret, 1),
                                    "r_mult": round(t3_r, 2),
                                    "combined_net_r": round(t1_r + t2_r + t3_r, 2),
                                    "ribbon_spans": list(ribbon_spans),
                                }

        results.append({
            "symbol": sym,
            "date": date_str,
            "breached_d1_low_5d": bool(row.get("breached_d1_low_5d", False)),
            "old_t2_has_reentry": bool(row.get("t2_has_reentry", False)),
            "old_t2_real_r": float(row.get("t2_real_r", 0.0)) if pd.notna(row.get("t2_real_r")) else 0.0,
            "old_t3_has_reentry": bool(row.get("t3_has_reentry", False)),
            "old_t3_real_r": float(row.get("t3_real_r", 0.0)) if pd.notna(row.get("t3_real_r")) else 0.0,
            "t1_stopped": t1_stopped,
            "t1_r": round(t1_r, 3),
            "t1_exit_bar": t1_exit_bar,
            "new_t2_has_reentry": trade_2["has_reentry"],
            "new_t2_track": t2_track,
            "new_t2_r": round(trade_2.get("r_mult", 0.0), 3) if trade_2["has_reentry"] else 0.0,
            "new_t3_has_reentry": trade_3["has_reentry"],
            "new_t3_r": round(trade_3.get("r_mult", 0.0), 3) if trade_3["has_reentry"] else 0.0,
        })

    res_df = pd.DataFrame(results)
    print(f"\nSimulation complete in {time.time() - t0:.1f}s.")
    return res_df


def calc_metrics(r_series: pd.Series):
    if len(r_series) == 0:
        return {"trades": 0, "win_rate": 0.0, "ev_r": 0.0, "total_r": 0.0, "pf": 0.0, "max_dd_r": 0.0}
    n = len(r_series)
    wins = r_series[r_series > 0]
    losses = r_series[r_series <= 0]
    wr = len(wins) / n * 100.0
    ev = r_series.mean()
    tot = r_series.sum()
    loss_sum = abs(losses.sum())
    pf = wins.sum() / loss_sum if loss_sum > 0 else 999.0
    cum = r_series.cumsum()
    peak = cum.cummax()
    max_dd = (cum - peak).min()
    return {
        "trades": n,
        "win_rate": round(wr, 1),
        "ev_r": round(ev, 3),
        "total_r": round(tot, 1),
        "pf": round(pf, 2),
        "max_dd_r": round(max_dd, 2)
    }


def main():
    parser = argparse.ArgumentParser(description="Dual-Track Continuation Simulation")
    parser.add_argument("--max-eps", type=int, default=None, help="Max EPs to simulate")
    parser.add_argument("--update-parquet", action="store_true", help="Update ep_combined_study_scored.parquet with new trades")
    args = parser.parse_args()

    res_df = run_dual_track_simulation(max_eps=args.max_eps)

    if args.update_parquet:
        study_path = Path("data/simulations/ep_combined_study_scored.parquet")
        backup_path = Path("data/simulations/ep_combined_study_scored_pre_dual_track.parquet")
        if study_path.exists() and not backup_path.exists():
            import shutil
            shutil.copy2(study_path, backup_path)
            print(f"Backed up original scored dataset to {backup_path}")

        df_orig = pd.read_parquet(study_path)
        # Merge by symbol and date
        merge_cols = ["symbol", "date", "new_t2_has_reentry", "new_t2_track", "new_t2_r", "new_t3_has_reentry", "new_t3_r"]
        merged = df_orig.merge(res_df[merge_cols], on=["symbol", "date"], how="left")
        merged["t2_has_reentry"] = merged["new_t2_has_reentry"].fillna(False)
        merged["t2_real_r"] = merged["new_t2_r"].fillna(0.0)
        merged["t2_track"] = merged["new_t2_track"].fillna("none")
        merged["t3_has_reentry"] = merged["new_t3_has_reentry"].fillna(False)
        merged["t3_real_r"] = merged["new_t3_r"].fillna(0.0)
        merged = merged.drop(columns=["new_t2_has_reentry", "new_t2_track", "new_t2_r", "new_t3_has_reentry", "new_t3_r"])

        tmp_p = study_path.with_suffix(".tmp.parquet")
        merged.to_parquet(tmp_p, index=False)
        tmp_p.replace(study_path)
        print(f"✅ Successfully updated {study_path} with Dual-Track continuation columns!")

    # 1. Performance across 1,689 Day 1 Breaches
    breaches = res_df[res_df["breached_d1_low_5d"]]
    print("\n" + "=" * 80)
    print(f"📊 COHORT 1: HISTORICAL DAY 1 BREACHES (N = {len(breaches)})")
    print("=" * 80)

    old_b_t2 = breaches[breaches["old_t2_has_reentry"]]["old_t2_real_r"]
    new_b_t2 = breaches[breaches["new_t2_has_reentry"]]["new_t2_r"]

    m_old_b = calc_metrics(old_b_t2)
    m_new_b = calc_metrics(new_b_t2)

    print(f"{'Metric':<20} | {'Before (Old T2)':<18} | {'After (Dual-Track T2)':<20} | {'Delta':<15}")
    print("-" * 80)
    for k in ["trades", "win_rate", "ev_r", "total_r", "pf", "max_dd_r"]:
        unit = "%" if k == "win_rate" else (" R" if "r" in k else "")
        v_old = m_old_b[k]
        v_new = m_new_b[k]
        diff = v_new - v_old
        diff_str = f"{diff:+.1f}{unit}" if isinstance(diff, float) else f"{diff:+d}"
        print(f"{k:<20} | {v_old}{unit:<16} | {v_new}{unit:<18} | {diff_str:<15}")

    # Track breakdown for breaches
    track_counts = breaches[breaches["new_t2_has_reentry"]]["new_t2_track"].value_counts()
    print("\n--- Track Breakdown on Breaches ---")
    for trk, cnt in track_counts.items():
        sub_r = breaches[breaches["new_t2_track"] == trk]["new_t2_r"]
        m_trk = calc_metrics(sub_r)
        print(f"  {trk}: {cnt} trades, WR: {m_trk['win_rate']}%, EV: {m_trk['ev_r']} R, Total: {m_trk['total_r']} R, PF: {m_trk['pf']}")

    # 2. Performance across Full EP Universe
    print("\n" + "=" * 80)
    print(f"📊 COHORT 2: FULL EP UNIVERSE (N = {len(res_df)})")
    print("=" * 80)

    old_full_t2 = res_df[res_df["old_t2_has_reentry"]]["old_t2_real_r"]
    new_full_t2 = res_df[res_df["new_t2_has_reentry"]]["new_t2_r"]

    m_old_f = calc_metrics(old_full_t2)
    m_new_f = calc_metrics(new_full_t2)

    print(f"{'Metric':<20} | {'Before (Old T2)':<18} | {'After (Dual-Track T2)':<20} | {'Delta':<15}")
    print("-" * 80)
    for k in ["trades", "win_rate", "ev_r", "total_r", "pf", "max_dd_r"]:
        unit = "%" if k == "win_rate" else (" R" if "r" in k else "")
        v_old = m_old_f[k]
        v_new = m_new_f[k]
        diff = v_new - v_old
        diff_str = f"{diff:+.1f}{unit}" if isinstance(diff, float) else f"{diff:+d}"
        print(f"{k:<20} | {v_old}{unit:<16} | {v_new}{unit:<18} | {diff_str:<15}")

    # Combined Sequence (T1 + T2 + T3)
    old_seq_r = res_df["t1_r"] + res_df["old_t2_real_r"] + res_df["old_t3_real_r"]
    new_seq_r = res_df["t1_r"] + res_df["new_t2_r"] + res_df["new_t3_r"]
    m_old_seq = calc_metrics(old_seq_r)
    m_new_seq = calc_metrics(new_seq_r)

    print("\n" + "=" * 80)
    print("📊 FULL LIFECYCLE COMBINED SEQUENCE (T1 + T2 + T3)")
    print("=" * 80)
    print(f"{'Metric':<20} | {'Before (Old Seq)':<18} | {'After (Dual-Track Seq)':<20} | {'Delta':<15}")
    print("-" * 80)
    for k in ["trades", "win_rate", "ev_r", "total_r", "pf", "max_dd_r"]:
        unit = "%" if k == "win_rate" else (" R" if "r" in k else "")
        v_old = m_old_seq[k]
        v_new = m_new_seq[k]
        diff = v_new - v_old
        diff_str = f"{diff:+.1f}{unit}" if isinstance(diff, float) else f"{diff:+d}"
        print(f"{k:<20} | {v_old}{unit:<16} | {v_new}{unit:<18} | {diff_str:<15}")


if __name__ == "__main__":
    main()
