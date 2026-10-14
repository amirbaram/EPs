"""build_qm_strategy_dataset.py — Builds Unified Institutional QM Strategy Trades Dataset.

Generates `data/simulations/qm_strategy_trades.parquet` containing complete trade execution
details across dual stoploss models (0.6x and 1.0x), multi-EMA trailing exits (10 EMA, 20 EMA, 50 EMA),
staged & balanced scale-outs, and dedicated ML Climax Exhaustion Partial Profit Taking models
(High-Velocity Swing, Institutional Hybrid, Position Trend).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
import datastore
import indicators
import labels
import qm_ml_engine
import serve_qm
import thematic_engine
import trendlab

SETUPS_PATH = Path("data/simulations/qm_historical_setups.parquet")
OUTPUT_PATH = Path("data/simulations/qm_strategy_trades.parquet")


def compute_market_and_qqq_lookup() -> Dict[str, Dict[str, Any]]:
    """Precomputes SPY, QQQ, and QQQ/SPY ratio indicators for all dates."""
    spy = datastore.load_bars("SPY")
    qqq = datastore.load_bars("QQQ")
    if spy is None or qqq is None:
        return {}

    spy = indicators.add_indicators(spy)
    qqq = indicators.add_indicators(qqq)

    common = spy.index.intersection(qqq.index)
    spy = spy.loc[common]
    qqq = qqq.loc[common]

    ratio = qqq["close"] / spy["close"]
    ratio_e10 = ratio.ewm(span=10, adjust=False).mean()
    ratio_e20 = ratio.ewm(span=20, adjust=False).mean()

    dates = common.strftime("%Y-%m-%d").tolist()
    spy_close = spy["close"].to_numpy(float)
    spy_sma200 = spy["sma200"].to_numpy(float)
    spy_sma50 = spy["sma50"].to_numpy(float)
    spy_ema10 = spy["ema10"].to_numpy(float)

    qqq_close = qqq["close"].to_numpy(float)
    qqq_ema10 = qqq["ema10"].to_numpy(float)
    qqq_ema20 = qqq["ema20"].to_numpy(float)

    r_val = ratio.to_numpy(float)
    r_e10 = ratio_e10.to_numpy(float)
    r_e20 = ratio_e20.to_numpy(float)

    lookup = {}
    for i in range(len(dates)):
        d = dates[i]
        c_spy = spy_close[i]
        s200_spy = spy_sma200[i]
        s50_spy = spy_sma50[i]
        e10_spy = spy_ema10[i]
        prev_e10_spy = spy_ema10[i - 1] if i >= 1 else e10_spy

        c_qqq = qqq_close[i]
        e10_qqq = qqq_ema10[i]
        e20_qqq = qqq_ema20[i]

        qqq_bull = bool(c_qqq >= e10_qqq >= e20_qqq)
        qqq_pullback = bool(c_qqq < e10_qqq or c_qqq < e20_qqq)
        ratio_bull = bool(r_val[i] >= r_e10[i] >= r_e20[i])

        lookup[d] = {
            "spy_close": round(c_spy, 2),
            "spy_above_200sma": bool(c_spy >= s200_spy),
            "spy_above_50sma": bool(c_spy >= s50_spy),
            "spy_10ema_slope": round((e10_spy - prev_e10_spy) / prev_e10_spy * 100.0, 3) if prev_e10_spy > 0 else 0.0,
            "qqq_bull_stack": qqq_bull,
            "qqq_pullback": qqq_pullback,
            "qqq_spy_ratio_bull": ratio_bull,
        }
    return lookup


def calculate_position_size_multiplier(
    prior_ep_days_ago: Optional[float],
    move_pct: float,
    breakout_rvol: float,
    pullback_ratio: float,
) -> Tuple[float, Dict[str, float]]:
    """Calculates position size multiplier based on fresh EP, pole size, breakout RVOL, and pullback ratio."""
    if prior_ep_days_ago is not None and not np.isnan(prior_ep_days_ago):
        if prior_ep_days_ago <= 15:
            s_ep = 1.30
        elif prior_ep_days_ago <= 30:
            s_ep = 1.15
        elif prior_ep_days_ago <= 60:
            s_ep = 1.05
        else:
            s_ep = 0.90
    else:
        s_ep = 0.85

    if move_pct >= 100.0:
        s_pole = 1.25
    elif move_pct >= 50.0:
        s_pole = 1.10
    else:
        s_pole = 1.00

    if breakout_rvol >= 2.5:
        s_rvol = 1.20
    elif breakout_rvol >= 1.5:
        s_rvol = 1.10
    else:
        s_rvol = 0.90

    if pullback_ratio <= 0.25:
        s_pb = 1.20
    elif pullback_ratio <= 0.35:
        s_pb = 1.05
    else:
        s_pb = 0.85

    raw_size = s_ep * s_pole * s_rvol * s_pb
    clamped_size = float(np.clip(raw_size, 0.50, 2.50))

    details = {
        "s_ep": s_ep,
        "s_pole": s_pole,
        "s_rvol": s_rvol,
        "s_pb": s_pb,
        "size_multiplier": round(clamped_size, 3),
    }
    return clamped_size, details


def simulate_ema_exit(
    idx: int,
    entry_p: float,
    stop_p: float,
    low_arr: np.ndarray,
    high_arr: np.ndarray,
    open_arr: np.ndarray,
    close_arr: np.ndarray,
    ema_arr: np.ndarray,
    dates_list: List[str],
    stop_reason: str,
    ema_name: str = "10ema",
    max_forward_bars: int = 250,
) -> Dict[str, Any]:
    """Simulates a trade forward until stoploss breach or EMA close violation."""
    n_bars = len(close_arr)
    max_t = min(n_bars - 1, idx + max_forward_bars)

    initial_risk = entry_p - stop_p
    peak_high = entry_p
    trough_low = entry_p

    exit_p = None
    exit_dt = None
    exit_reason = None
    exit_idx = None
    trail_active = False

    for t in range(idx + 1, max_t + 1):
        cur_low = low_arr[t]
        cur_high = high_arr[t]
        cur_open = open_arr[t]
        cur_close = close_arr[t]
        cur_ema = ema_arr[t]
        prev_ema = ema_arr[t - 1]

        peak_high = max(peak_high, cur_high)
        trough_low = min(trough_low, cur_low)

        # 1. Stoploss breach
        if cur_low <= stop_p:
            exit_p = min(cur_open, stop_p)
            exit_dt = dates_list[t]
            exit_idx = t
            exit_reason = stop_reason
            break

        # 2. EMA Trailing Exit: Activated once EMA slopes upward and price is >= 1% above EMA
        if cur_ema > prev_ema and (cur_close / cur_ema - 1.0) >= 0.01:
            trail_active = True

        if trail_active:
            if cur_close < cur_ema:
                exit_p = cur_close
                exit_dt = dates_list[t]
                exit_idx = t
                exit_reason = f"{ema_name}_close_violation"
                break
        elif (t - idx) >= 10 and cur_close < cur_ema:
            # Dead zone resolution: after 10 sessions without upward momentum separation,
            # exit on EMA breach to avoid stranded 250-bar trade
            exit_p = cur_close
            exit_dt = dates_list[t]
            exit_idx = t
            exit_reason = f"{ema_name}_failed_activation"
            break

    if exit_p is None:
        exit_p = float(close_arr[max_t])
        exit_dt = dates_list[max_t]
        exit_idx = max_t
        exit_reason = "max_holding_period"

    gain_pct = (exit_p / entry_p - 1.0) * 100.0
    r_multiple = (exit_p - entry_p) / initial_risk if initial_risk > 0 else 0.0
    max_mfe_r = (peak_high - entry_p) / initial_risk if initial_risk > 0 else 0.0
    max_mae_r = (entry_p - trough_low) / initial_risk if initial_risk > 0 else 0.0
    bars_held = exit_idx - idx

    return {
        "stop_loss": round(stop_p, 4),
        "risk_pct": round((initial_risk / entry_p) * 100.0, 2),
        "exit_date": exit_dt,
        "exit_price": round(exit_p, 4),
        "exit_reason": exit_reason,
        "bars": bars_held,
        "r": round(r_multiple, 3),
        "gain_pct": round(gain_pct, 2),
        "max_mfe_r": round(max_mfe_r, 2),
        "max_mae_r": round(max_mae_r, 2),
        "is_win": bool(r_multiple > 0.05),
    }


def build_strategy_trades_dataset() -> pd.DataFrame:
    print("══════════════════════════════════════════════════════════════════════════════")
    print(" 🚀 Building Unified QM Strategy Dataset (All EMAs, Scaling & ML Climax Exits)")
    print("══════════════════════════════════════════════════════════════════════════════")

    if not SETUPS_PATH.exists():
        raise FileNotFoundError(f"QM setups dataset not found at {SETUPS_PATH}")

    df_setups = pd.read_parquet(SETUPS_PATH)
    print(f"Loaded {len(df_setups)} total setups from {SETUPS_PATH}.")

    bks = df_setups[df_setups["state"] == "breakout"].copy()
    print(f"Total QM Breakouts found: {len(bks)}.")

    if "rs_score" not in bks.columns or "stage2" not in bks.columns or "dist_52w_high" not in bks.columns:
        bks = serve_qm._enrich_with_rs_technicals(bks)

    lead_secs, lead_thms, emerg_thms = serve_qm.get_thematic_macro_sets()
    print(f"Loaded {len(lead_secs)} Leading Sectors and {len(lead_thms)} Leading Themes.")

    mkt_lookup = compute_market_and_qqq_lookup()
    print(f"Market health lookup computed for {len(mkt_lookup)} trading dates.")

    ml_engine = qm_ml_engine.get_qm_ml_engine()
    print(f"QM ML Climax Engine active: {ml_engine.pipeline is not None}.")

    labels.load()
    grouped = bks.groupby("symbol")
    all_syms = sorted(grouped.groups.keys())
    total_syms = len(all_syms)

    records: List[Dict[str, Any]] = []
    t0 = time.time()

    print(f"Simulating trades across {total_syms} symbols...")

    for i_sym, sym in enumerate(all_syms):
        if (i_sym + 1) % 250 == 0 or (i_sym + 1) == total_syms:
            elapsed = time.time() - t0
            rate = (i_sym + 1) / elapsed if elapsed > 0 else 0
            sys.stdout.write(f"\rProgress: {i_sym + 1}/{total_syms} symbols ({(i_sym + 1)/total_syms*100:.1f}%) | Records: {len(records)} | {rate:.1f} sym/s")
            sys.stdout.flush()

        sym_bks = grouped.get_group(sym)
        df_bars = datastore.load_bars(sym)
        if df_bars is None or len(df_bars) < 60:
            continue

        c_s = df_bars["close"]
        h_s = df_bars["high"]
        l_s = df_bars["low"]
        o_s = df_bars["open"]
        v_s = df_bars["volume"]

        e10 = c_s.ewm(span=10, adjust=False).mean().to_numpy(float)
        e20 = c_s.ewm(span=20, adjust=False).mean().to_numpy(float)
        e50 = c_s.ewm(span=50, adjust=False).mean().to_numpy(float)
        v20 = v_s.rolling(20, min_periods=5).mean().to_numpy(float)

        delta = c_s.diff()
        gain14 = delta.clip(lower=0).ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
        loss14 = (-delta.clip(upper=0)).ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
        rsi14 = (100.0 - 100.0 / (1.0 + gain14 / loss14.replace(0, np.nan))).to_numpy(float)

        gain3 = delta.clip(lower=0).ewm(alpha=1.0 / 3, adjust=False, min_periods=3).mean()
        loss3 = (-delta.clip(upper=0)).ewm(alpha=1.0 / 3, adjust=False, min_periods=3).mean()
        rsi3 = (100.0 - 100.0 / (1.0 + gain3 / loss3.replace(0, np.nan))).to_numpy(float)

        sma5_rsi14 = pd.Series(rsi14).rolling(5, min_periods=2).mean().to_numpy(float)
        prev_c = c_s.shift(1)
        tr = pd.concat([h_s - l_s, (h_s - prev_c).abs(), (l_s - prev_c).abs()], axis=1).max(axis=1)
        atr14 = tr.rolling(14, min_periods=5).mean().to_numpy(float)
        hi252 = h_s.rolling(252, min_periods=60).max().to_numpy(float)

        dates_list = df_bars.index.strftime("%Y-%m-%d").tolist()
        date_to_idx = {d: idx for idx, d in enumerate(dates_list)}
        n_bars = len(dates_list)

        c_arr = c_s.to_numpy(float)
        h_arr = h_s.to_numpy(float)
        l_arr = l_s.to_numpy(float)
        o_arr = o_s.to_numpy(float)
        v_arr = v_s.to_numpy(float)

        for _, setup in sym_bks.iterrows():
            bk_date = str(setup["date"])
            if bk_date not in date_to_idx:
                continue
            idx = date_to_idx[bk_date]

            trigger_px = float(setup.get("trigger", c_arr[idx]))
            open_px = float(o_arr[idx])
            high_px = float(h_arr[idx])

            # Realistic Intraday Stop-Buy Entry
            if high_px >= trigger_px:
                entry_p = min(high_px, max(open_px, trigger_px + 0.05))
            else:
                entry_p = float(c_arr[idx])

            # Consolidation low determination
            cons_start = str(setup.get("qm_cons_start", ""))
            if cons_start and cons_start in date_to_idx:
                c_s_idx = max(0, date_to_idx[cons_start])
                c_window_lows = l_arr[c_s_idx:idx + 1]
                pivot_low_p = float(np.min(c_window_lows)) if len(c_window_lows) > 0 else float(l_arr[idx])
                pivot_low_dt = dates_list[c_s_idx]
            else:
                lookback_15 = max(0, idx - 15)
                pivot_low_p = float(np.min(l_arr[lookback_15:idx + 1]))
                pivot_low_dt = dates_list[lookback_15]

            full_pivot_dist = entry_p - pivot_low_p
            if full_pivot_dist <= 0:
                full_pivot_dist = entry_p * 0.05
                pivot_low_p = entry_p * 0.95

            stop_06x = entry_p - (0.6 * full_pivot_dist)
            stop_10x = pivot_low_p

            if (entry_p - stop_06x) / entry_p < 0.015:
                stop_06x = entry_p * 0.985
            if (entry_p - stop_10x) / entry_p < 0.02:
                stop_10x = entry_p * 0.98

            prior_ep_days_ago = setup.get("prior_ep_days_ago")
            move_pct = float(setup.get("move_pct", 0.0))
            bk_rvol = float(setup.get("vol_x", 1.0)) if pd.notna(setup.get("vol_x")) else 1.0
            pullback_pct = float(setup.get("pullback_pct", 0.0))
            pullback_ratio = float(pullback_pct / move_pct) if move_pct > 0 else 0.5

            pos_size_mult, size_details = calculate_position_size_multiplier(
                prior_ep_days_ago=prior_ep_days_ago,
                move_pct=move_pct,
                breakout_rvol=bk_rvol,
                pullback_ratio=pullback_ratio,
            )

            # Standard EMA exits
            t10_06 = simulate_ema_exit(idx, entry_p, stop_06x, l_arr, h_arr, o_arr, c_arr, e10, dates_list, "stop_loss_0.6x", "10ema")
            t20_06 = simulate_ema_exit(idx, entry_p, stop_06x, l_arr, h_arr, o_arr, c_arr, e20, dates_list, "stop_loss_0.6x", "20ema")
            t50_06 = simulate_ema_exit(idx, entry_p, stop_06x, l_arr, h_arr, o_arr, c_arr, e50, dates_list, "stop_loss_0.6x", "50ema")

            t10_10 = simulate_ema_exit(idx, entry_p, stop_10x, l_arr, h_arr, o_arr, c_arr, e10, dates_list, "stop_loss_1.0x", "10ema")
            t20_10 = simulate_ema_exit(idx, entry_p, stop_10x, l_arr, h_arr, o_arr, c_arr, e20, dates_list, "stop_loss_1.0x", "20ema")
            t50_10 = simulate_ema_exit(idx, entry_p, stop_10x, l_arr, h_arr, o_arr, c_arr, e50, dates_list, "stop_loss_1.0x", "50ema")

            # Staged & Balanced Scaling
            r_staged_06x = round(0.33 * t10_06["r"] + 0.33 * t20_06["r"] + 0.34 * t50_06["r"], 3)
            sized_r_staged_06x = round(r_staged_06x * pos_size_mult, 3)
            is_win_staged_06x = bool(r_staged_06x > 0.05)
            bars_staged_06x = round(0.33 * t10_06["bars"] + 0.33 * t20_06["bars"] + 0.34 * t50_06["bars"], 1)

            r_staged_10x = round(0.33 * t10_10["r"] + 0.33 * t20_10["r"] + 0.34 * t50_10["r"], 3)
            sized_r_staged_10x = round(r_staged_10x * pos_size_mult, 3)
            is_win_staged_10x = bool(r_staged_10x > 0.05)
            bars_staged_10x = round(0.33 * t10_10["bars"] + 0.33 * t20_10["bars"] + 0.34 * t50_10["bars"], 1)

            r_balanced_06x = round(0.50 * t10_06["r"] + 0.50 * t50_06["r"], 3)
            sized_r_balanced_06x = round(r_balanced_06x * pos_size_mult, 3)
            is_win_balanced_06x = bool(r_balanced_06x > 0.05)
            bars_balanced_06x = round(0.50 * t10_06["bars"] + 0.50 * t50_06["bars"], 1)

            r_balanced_10x = round(0.50 * t10_10["r"] + 0.50 * t50_10["r"], 3)
            sized_r_balanced_10x = round(r_balanced_10x * pos_size_mult, 3)
            is_win_balanced_10x = bool(r_balanced_10x > 0.05)
            bars_balanced_10x = round(0.50 * t10_10["bars"] + 0.50 * t50_10["bars"], 1)

            # Market context
            mkt = mkt_lookup.get(bk_date, {})
            spy_200 = mkt.get("spy_above_200sma", True if bk_date < "2016-01-01" else False)
            spy_50 = mkt.get("spy_above_50sma", False)
            spy_slope = mkt.get("spy_10ema_slope", 0.0)
            qqq_bull = mkt.get("qqq_bull_stack", False)
            qqq_pb = mkt.get("qqq_pullback", False)
            qqq_ratio_bull = mkt.get("qqq_spy_ratio_bull", False)

            # Sectors & Themes matching
            sec_name = str(setup.get("sector", ""))
            thm_name = str(setup.get("theme", ""))
            thms_str = str(setup.get("themes", ""))
            is_lead_sec = (sec_name in lead_secs) or any(s in thms_str for s in lead_secs) or (thm_name in lead_secs)
            is_lead_thm = (thm_name in lead_thms) or any(t in thms_str for t in lead_thms)
            is_emerg_thm = (thm_name in emerg_thms) or any(t in thms_str for t in emerg_thms)
            is_lead_macro = bool(is_lead_sec or is_lead_thm)

            # RS Metrics
            rs_sc = int(setup["rs_score"]) if pd.notna(setup.get("rs_score")) else None
            st2 = bool(setup.get("stage2", False)) if pd.notna(setup.get("stage2")) else False
            d52 = round(float(setup["dist_52w_high"]), 1) if pd.notna(setup.get("dist_52w_high")) else None
            rs_lead = bool((rs_sc is not None and rs_sc >= 90) and (st2 or (d52 is not None and d52 >= -15.0)))

            # Climax Exhaustion Simulation (Cached inference per bar)
            max_t = min(n_bars - 1, idx + 250)
            init_risk_06 = entry_p - stop_06x
            init_risk_10 = entry_p - stop_10x

            cons_bars_val = float(setup.get("cons_bars", 10.0))
            pinch_ratio_val = float(setup.get("pinch_ratio", 2.5)) if pd.notna(setup.get("pinch_ratio")) else 2.5
            prior_ep_val = float(prior_ep_days_ago) if prior_ep_days_ago is not None and pd.notna(prior_ep_days_ago) else 999.0
            rs_val = float(rs_sc) if rs_sc is not None else 50.0
            macro_tailwind_val = 1.0 if is_lead_macro else 0.0
            vol_bo_val = float(v_arr[idx])

            # Pre-evaluate bar climax candidates
            bar_probs: Dict[int, float] = {}
            bar_unreal_06: Dict[int, float] = {}
            bar_unreal_10: Dict[int, float] = {}

            for t in range(idx + 1, min(n_bars, idx + 45)):
                cur_c = c_arr[t]
                unreal_06 = (cur_c - entry_p) / init_risk_06 if init_risk_06 > 0 else 0.0
                bar_unreal_06[t] = unreal_06

                if unreal_06 >= 2.0:
                    cur_h = h_arr[t]
                    cur_l = l_arr[t]
                    cur_o = o_arr[t]
                    cur_v = v_arr[t]
                    cur_e10 = e10[t]
                    cur_e20 = e20[t]
                    cur_s50 = e50[t]
                    cur_atr = max(atr14[t], 0.01) if not np.isnan(atr14[t]) else 0.05 * cur_c
                    cur_r14 = rsi14[t] if not np.isnan(rsi14[t]) else 50.0
                    cur_r3 = rsi3[t] if not np.isnan(rsi3[t]) else 50.0
                    cur_r14_sma = sma5_rsi14[t] if not np.isnan(sma5_rsi14[t]) else cur_r14
                    r3_ago_idx = max(0, t - 3)
                    r14_ago = rsi14[r3_ago_idx] if not np.isnan(rsi14[r3_ago_idx]) else cur_r14
                    rsi_div = (cur_r14 - r14_ago) if cur_h > h_arr[r3_ago_idx] else 0.0
                    rsi_vs_sma = cur_r14 - cur_r14_sma
                    bar_rvol = cur_v / max(v20[t], 1.0) if not np.isnan(v20[t]) else 1.0
                    vol_vs_bo = cur_v / max(vol_bo_val, 1.0)
                    c_range = max(cur_h - cur_l, 0.001)
                    upper_wick = (cur_h - max(cur_o, cur_c)) / c_range
                    close_pos = (cur_c - cur_l) / c_range

                    consec_up = 0
                    for k in range(t, max(0, t - 6), -1):
                        if k > 0 and c_arr[k] > c_arr[k - 1]:
                            consec_up += 1
                        else:
                            break

                    tl_swing_dist = (cur_c - pivot_low_p) / pivot_low_p if pivot_low_p > 0 else 0.0
                    cur_hi252 = hi252[t] if not np.isnan(hi252[t]) else cur_h
                    dist_52w = (cur_c - cur_hi252) / cur_hi252 if cur_hi252 > 0 else 0.0
                    dist_pole = (cur_c - entry_p) / entry_p if entry_p > 0 else 0.0

                    f_arr = np.array([
                        (cur_c - cur_e10) / cur_e10,
                        (cur_c - cur_e20) / cur_e20,
                        (cur_c - cur_s50) / cur_s50,
                        (cur_h - cur_e20) / cur_atr,
                        unreal_06,
                        cur_r14,
                        cur_r3,
                        rsi_div,
                        rsi_vs_sma,
                        bar_rvol,
                        vol_vs_bo,
                        upper_wick,
                        close_pos,
                        consec_up,
                        t - idx,
                        tl_swing_dist,
                        0.0,
                        dist_52w,
                        dist_pole,
                        move_pct,
                        cons_bars_val,
                        pinch_ratio_val,
                        prior_ep_val,
                        rs_val,
                        macro_tailwind_val,
                    ], dtype=np.float32)

                    bar_probs[t] = ml_engine.predict_proba_fast(f_arr)

            def run_climax_archetype(stop_p_init, init_risk, r_hurdle, prob_thresh, c_share, trailing_emas):
                climax_t = None
                climax_p = 0.0
                climax_prob = 0.0
                climax_r = 0.0

                for t in range(idx + 1, min(n_bars, idx + 45)):
                    if l_arr[t] <= stop_p_init:
                        break
                    unreal = (c_arr[t] - entry_p) / init_risk if init_risk > 0 else 0.0
                    if unreal >= r_hurdle:
                        p_val = bar_probs.get(t, 0.0)
                        cur_atr = max(atr14[t], 0.01) if not np.isnan(atr14[t]) else 0.05 * c_arr[t]
                        is_parabolic = bool((h_arr[t] - e20[t]) / cur_atr >= 3.5)
                        if p_val >= prob_thresh or is_parabolic:
                            climax_t = t
                            climax_p = c_arr[t]
                            climax_prob = p_val
                            climax_r = unreal
                            break

                net_r = 0.0
                net_bars = 0.0

                if climax_t is not None:
                    net_r += c_share * climax_r
                    net_bars += c_share * (climax_t - idx)
                    rem_stop = entry_p

                    for ema_arr, share in trailing_emas:
                        leg_exit_p = None
                        leg_idx = None
                        rem_trail = False
                        for t in range(climax_t + 1, max_t + 1):
                            if l_arr[t] <= rem_stop:
                                leg_exit_p = min(o_arr[t], rem_stop)
                                leg_idx = t
                                break
                            if ema_arr[t] > ema_arr[t - 1] and (c_arr[t] / ema_arr[t] - 1.0) >= 0.01:
                                rem_trail = True
                            if rem_trail and c_arr[t] < ema_arr[t]:
                                leg_exit_p = c_arr[t]
                                leg_idx = t
                                break
                            elif (t - idx) >= 10 and c_arr[t] < ema_arr[t]:
                                leg_exit_p = c_arr[t]
                                leg_idx = t
                                break
                        if leg_exit_p is None:
                            leg_exit_p = float(c_arr[max_t])
                            leg_idx = max_t
                        leg_r = (leg_exit_p - entry_p) / init_risk
                        net_r += share * leg_r
                        net_bars += share * (leg_idx - idx)
                    return {
                        "r": round(net_r, 3), "bars": round(net_bars, 1), "is_win": bool(net_r > 0.05),
                        "climax_date": dates_list[climax_t], "climax_price": round(climax_p, 4),
                        "climax_r": round(climax_r, 2), "prob": round(climax_prob, 3)
                    }
                else:
                    tot_s = sum(s for _, s in trailing_emas)
                    for ema_arr, share in trailing_emas:
                        norm_s = share / tot_s
                        leg_exit_p = None
                        leg_idx = None
                        rem_trail = False
                        for t in range(idx + 1, max_t + 1):
                            if l_arr[t] <= stop_p_init:
                                leg_exit_p = min(o_arr[t], stop_p_init)
                                leg_idx = t
                                break
                            if ema_arr[t] > ema_arr[t - 1] and (c_arr[t] / ema_arr[t] - 1.0) >= 0.01:
                                rem_trail = True
                            if rem_trail and c_arr[t] < ema_arr[t]:
                                leg_exit_p = c_arr[t]
                                leg_idx = t
                                break
                            elif (t - idx) >= 10 and c_arr[t] < ema_arr[t]:
                                leg_exit_p = c_arr[t]
                                leg_idx = t
                                break
                        if leg_exit_p is None:
                            leg_exit_p = float(c_arr[max_t])
                            leg_idx = max_t
                        leg_r = (leg_exit_p - entry_p) / init_risk
                        net_r += norm_s * leg_r
                        net_bars += norm_s * (leg_idx - idx)
                    return {
                        "r": round(net_r, 3), "bars": round(net_bars, 1), "is_win": bool(net_r > 0.05),
                        "climax_date": None, "climax_price": None, "climax_r": None, "prob": None
                    }

            c_swing_06 = run_climax_archetype(stop_06x, init_risk_06, 2.5, 0.45, 0.50, [(e10, 0.50)])
            c_hybrid_06 = run_climax_archetype(stop_06x, init_risk_06, 3.0, 0.50, 0.33, [(e10, 0.33), (e20, 0.34)])
            c_trend_06 = run_climax_archetype(stop_06x, init_risk_06, 5.0, 0.65, 0.50, [(e50, 0.50)])

            c_swing_10 = run_climax_archetype(stop_10x, init_risk_10, 2.5, 0.45, 0.50, [(e10, 0.50)])
            c_hybrid_10 = run_climax_archetype(stop_10x, init_risk_10, 3.0, 0.50, 0.33, [(e10, 0.33), (e20, 0.34)])
            c_trend_10 = run_climax_archetype(stop_10x, init_risk_10, 5.0, 0.65, 0.50, [(e50, 0.50)])

            rec = {
                "symbol": sym,
                "date": bk_date,
                "entry_price": round(entry_p, 4),
                "pivot_low_price": round(pivot_low_p, 4),
                "pivot_low_date": pivot_low_dt,
                "full_pivot_dist": round(full_pivot_dist, 4),
                # 0.6x Baseline Trade
                "stop_loss_06x": t10_06["stop_loss"],
                "risk_pct_06x": t10_06["risk_pct"],
                "exit_date_06x": t10_06["exit_date"],
                "exit_price_06x": t10_06["exit_price"],
                "exit_reason_06x": t10_06["exit_reason"],
                "bars_06x": t10_06["bars"],
                "r_06x": t10_06["r"],
                "sized_r_06x": round(t10_06["r"] * pos_size_mult, 3),
                "gain_pct_06x": t10_06["gain_pct"],
                "mfe_r_06x": t10_06["max_mfe_r"],
                "mae_r_06x": t10_06["max_mae_r"],
                "is_win_06x": t10_06["is_win"],
                # 1.0x Baseline Trade
                "stop_loss_10x": t10_10["stop_loss"],
                "risk_pct_10x": t10_10["risk_pct"],
                "exit_date_10x": t10_10["exit_date"],
                "exit_price_10x": t10_10["exit_price"],
                "exit_reason_10x": t10_10["exit_reason"],
                "bars_10x": t10_10["bars"],
                "r_10x": t10_10["r"],
                "sized_r_10x": round(t10_10["r"] * pos_size_mult, 3),
                "gain_pct_10x": t10_10["gain_pct"],
                "mfe_r_10x": t10_10["max_mfe_r"],
                "mae_r_10x": t10_10["max_mae_r"],
                "is_win_10x": t10_10["is_win"],
                # Multi-EMA Trailing Exits (0.6x Stop)
                "r_e10_06x": t10_06["r"],
                "sized_r_e10_06x": round(t10_06["r"] * pos_size_mult, 3),
                "win_e10_06x": t10_06["is_win"],
                "is_win_e10_06x": t10_06["is_win"],
                "bars_e10_06x": t10_06["bars"],
                "r_e20_06x": t20_06["r"],
                "sized_r_e20_06x": round(t20_06["r"] * pos_size_mult, 3),
                "win_e20_06x": t20_06["is_win"],
                "is_win_e20_06x": t20_06["is_win"],
                "bars_e20_06x": t20_06["bars"],
                "r_e50_06x": t50_06["r"],
                "sized_r_e50_06x": round(t50_06["r"] * pos_size_mult, 3),
                "win_e50_06x": t50_06["is_win"],
                "is_win_e50_06x": t50_06["is_win"],
                "bars_e50_06x": t50_06["bars"],
                "r_staged_06x": r_staged_06x,
                "sized_r_staged_06x": sized_r_staged_06x,
                "is_win_staged_06x": is_win_staged_06x,
                "bars_staged_06x": bars_staged_06x,
                "r_balanced_06x": r_balanced_06x,
                "sized_r_balanced_06x": sized_r_balanced_06x,
                "is_win_balanced_06x": is_win_balanced_06x,
                "bars_balanced_06x": bars_balanced_06x,
                # Multi-EMA Trailing Exits (1.0x Stop)
                "r_e10_10x": t10_10["r"],
                "sized_r_e10_10x": round(t10_10["r"] * pos_size_mult, 3),
                "win_e10_10x": t10_10["is_win"],
                "is_win_e10_10x": t10_10["is_win"],
                "bars_e10_10x": t10_10["bars"],
                "r_e20_10x": t20_10["r"],
                "sized_r_e20_10x": round(t20_10["r"] * pos_size_mult, 3),
                "win_e20_10x": t20_10["is_win"],
                "is_win_e20_10x": t20_10["is_win"],
                "bars_e20_10x": t20_10["bars"],
                "r_e50_10x": t50_10["r"],
                "sized_r_e50_10x": round(t50_10["r"] * pos_size_mult, 3),
                "win_e50_10x": t50_10["is_win"],
                "is_win_e50_10x": t50_10["is_win"],
                "bars_e50_10x": t50_10["bars"],
                "r_staged_10x": r_staged_10x,
                "sized_r_staged_10x": sized_r_staged_10x,
                "is_win_staged_10x": is_win_staged_10x,
                "bars_staged_10x": bars_staged_10x,
                "r_balanced_10x": r_balanced_10x,
                "sized_r_balanced_10x": sized_r_balanced_10x,
                "is_win_balanced_10x": is_win_balanced_10x,
                "bars_balanced_10x": bars_balanced_10x,
                # Dedicated ML Climax Partial Scaling Exits (0.6x Stop)
                "r_climax_swing_06x": c_swing_06["r"],
                "sized_r_climax_swing_06x": round(c_swing_06["r"] * pos_size_mult, 3),
                "is_win_climax_swing_06x": c_swing_06["is_win"],
                "bars_climax_swing_06x": c_swing_06["bars"],
                "climax_date_06x": c_swing_06["climax_date"] or c_hybrid_06["climax_date"],
                "climax_price_06x": c_swing_06["climax_price"] or c_hybrid_06["climax_price"],
                "climax_r_06x": c_swing_06["climax_r"] or c_hybrid_06["climax_r"],
                "prob_exhaustion_06x": c_swing_06["prob"] or c_hybrid_06["prob"],
                "r_climax_hybrid_06x": c_hybrid_06["r"],
                "sized_r_climax_hybrid_06x": round(c_hybrid_06["r"] * pos_size_mult, 3),
                "is_win_climax_hybrid_06x": c_hybrid_06["is_win"],
                "bars_climax_hybrid_06x": c_hybrid_06["bars"],
                "r_climax_trend_06x": c_trend_06["r"],
                "sized_r_climax_trend_06x": round(c_trend_06["r"] * pos_size_mult, 3),
                "is_win_climax_trend_06x": c_trend_06["is_win"],
                "bars_climax_trend_06x": c_trend_06["bars"],
                # Dedicated ML Climax Partial Scaling Exits (1.0x Stop)
                "r_climax_swing_10x": c_swing_10["r"],
                "sized_r_climax_swing_10x": round(c_swing_10["r"] * pos_size_mult, 3),
                "is_win_climax_swing_10x": c_swing_10["is_win"],
                "bars_climax_swing_10x": c_swing_10["bars"],
                "climax_date_10x": c_swing_10["climax_date"] or c_hybrid_10["climax_date"],
                "climax_price_10x": c_swing_10["climax_price"] or c_hybrid_10["climax_price"],
                "climax_r_10x": c_swing_10["climax_r"] or c_hybrid_10["climax_r"],
                "prob_exhaustion_10x": c_swing_10["prob"] or c_hybrid_10["prob"],
                "r_climax_hybrid_10x": c_hybrid_10["r"],
                "sized_r_climax_hybrid_10x": round(c_hybrid_10["r"] * pos_size_mult, 3),
                "is_win_climax_hybrid_10x": c_hybrid_10["is_win"],
                "bars_climax_hybrid_10x": c_hybrid_10["bars"],
                "r_climax_trend_10x": c_trend_10["r"],
                "sized_r_climax_trend_10x": round(c_trend_10["r"] * pos_size_mult, 3),
                "is_win_climax_trend_10x": c_trend_10["is_win"],
                "bars_climax_trend_10x": c_trend_10["bars"],
                # Position Sizing Multiplier
                "pos_size_mult": pos_size_mult,
                "s_ep": size_details["s_ep"],
                "s_pole": size_details["s_pole"],
                "s_rvol": size_details["s_rvol"],
                "s_pb": size_details["s_pb"],
                # Market Regime & Technical Sponsorship
                "spy_above_200sma": spy_200,
                "spy_above_50sma": spy_50,
                "spy_10ema_slope": spy_slope,
                "qqq_bull_stack": qqq_bull,
                "qqq_pullback": qqq_pb,
                "qqq_spy_ratio_bull": qqq_ratio_bull,
                # Macro Universe & Sectors
                "sector": sec_name,
                "theme": thm_name,
                "themes": thms_str,
                "is_leading_sector": is_lead_sec,
                "is_leading_theme": is_lead_thm,
                "is_emerging_theme": is_emerg_thm,
                "is_leading_macro": is_lead_macro,
                # Setup Details
                "move_pct": move_pct,
                "cons_bars": int(setup.get("cons_bars", 15)),
                "pullback_pct": pullback_pct,
                "pullback_ratio": pullback_ratio,
                "vol_x": round(float(setup.get("vol_x", 1.0)), 2) if pd.notna(setup.get("vol_x")) else 1.0,
                "breakout_rvol": bk_rvol,
                "pattern": str(setup.get("pattern", "other")),
                "r2": round(float(setup["r2"]), 4) if pd.notna(setup.get("r2")) else None,
                "pinch_ratio": round(float(setup["pinch_ratio"]), 2) if pd.notna(setup.get("pinch_ratio")) else None,
                # RS Posture
                "rs_score": rs_sc,
                "stage2": st2,
                "dist_52w_high": d52,
                "rs_leader": rs_lead,
                # EP Catalyst Cross-Reference
                "has_prior_ep": bool(setup.get("has_prior_ep", False)),
                "prior_ep_date": str(setup["prior_ep_date"]) if pd.notna(setup.get("prior_ep_date")) and setup.get("prior_ep_date") else None,
                "prior_ep_days_ago": int(setup["prior_ep_days_ago"]) if pd.notna(setup.get("prior_ep_days_ago")) and setup.get("prior_ep_days_ago") is not None else None,
                "prior_ep_gap_pct": round(float(setup["prior_ep_gap_pct"]), 1) if pd.notna(setup.get("prior_ep_gap_pct")) and setup.get("prior_ep_gap_pct") is not None else None,
                "prior_ep_rvol": round(float(setup["prior_ep_rvol"]), 1) if pd.notna(setup.get("prior_ep_rvol")) and setup.get("prior_ep_rvol") is not None else None,
                "prior_ep_subtype": str(setup.get("prior_ep_subtype", "")) if pd.notna(setup.get("prior_ep_subtype")) else None,
                "ep_badge": str(setup.get("ep_badge", "Pure Technical QM")),
            }
            records.append(rec)

    df_out = pd.DataFrame(records)
    print(f"\nCompleted simulation for {len(df_out)} breakout setups.")

    # Enrich with Setup Quality (QM-V2) and Dynamic Quantile Stop (QM-Stop)
    if not df_out.empty:
        try:
            print("Applying Machine Learning Setup Quality & Dynamic Stop models...")
            q_cols = [
                "move_pct", "cons_bars", "pullback_pct", "pullback_ratio", "vol_x", "breakout_rvol",
                "pinch_ratio", "r2", "rs_score", "dist_52w_high", "prior_ep_days_ago", "prior_ep_gap_pct",
                "prior_ep_rvol", "spy_above_200sma", "spy_above_50sma", "spy_10ema_slope",
                "qqq_bull_stack", "qqq_pullback", "qqq_spy_ratio_bull",
                "is_leading_sector", "is_leading_theme", "is_leading_macro", "stage2", "has_prior_ep"
            ]
            X_q = df_out[q_cols].copy()
            X_q["pinch_ratio"] = X_q["pinch_ratio"].fillna(2.5)
            X_q["r2"] = X_q["r2"].fillna(0.5)
            X_q["rs_score"] = X_q["rs_score"].fillna(50.0)
            X_q["dist_52w_high"] = X_q["dist_52w_high"].fillna(-15.0)
            X_q["prior_ep_days_ago"] = X_q["prior_ep_days_ago"].fillna(999.0)
            X_q["prior_ep_gap_pct"] = X_q["prior_ep_gap_pct"].fillna(0.0)
            X_q["prior_ep_rvol"] = X_q["prior_ep_rvol"].fillna(0.0)
            for c in ["spy_above_200sma", "spy_above_50sma", "qqq_bull_stack", "qqq_pullback", "qqq_spy_ratio_bull", "is_leading_sector", "is_leading_theme", "is_leading_macro", "stage2", "has_prior_ep"]:
                X_q[c] = X_q[c].astype(float)

            if ml_engine.clf_big is not None and ml_engine.clf_trap is not None:
                p_big = ml_engine.clf_big.predict_proba(X_q)[:, 1]
                p_trap = ml_engine.clf_trap.predict_proba(X_q)[:, 1]
                qm_score = np.clip((p_big / (p_big + p_trap + 1e-5)) * 100.0, 0.0, 100.0)
                df_out["p_big_move"] = np.round(p_big, 3)
                df_out["p_trap"] = np.round(p_trap, 3)
                df_out["qm_score"] = np.round(qm_score, 1)

                tiers, sizings, recs = [], [], []
                for sc, pt in zip(qm_score, p_trap):
                    if sc >= 75.0 and pt < 0.22:
                        tiers.append("Premier")
                        sizings.append(1.50)
                        recs.append("PREMIER BREAKOUT — FULL CONVICTION (1.5x)")
                    elif sc >= 60.0 and pt < 0.28:
                        tiers.append("High Quality")
                        sizings.append(1.20)
                        recs.append("HIGH QUALITY SETUP — OVERWEIGHT (1.2x)")
                    elif sc >= 40.0 and pt < 0.35:
                        tiers.append("Standard")
                        sizings.append(1.00)
                        recs.append("STANDARD BREAKOUT — STANDARD ALLOCATION (1.0x)")
                    elif pt >= 0.35 or sc < 25.0:
                        tiers.append("Toxic / Trap")
                        sizings.append(0.00)
                        recs.append("TOXIC SETUP AVOIDED — HIGH FAILURE RISK")
                    else:
                        tiers.append("Marginal")
                        sizings.append(0.50)
                        recs.append("MARGINAL SETUP — DEFENSIVE SIZING (0.5x)")

                df_out["qm_tier"] = tiers
                df_out["conviction_sizing"] = sizings
                df_out["ml_recommendation"] = recs

            # Dynamic Stop Loss
            s_cols = [
                "move_pct", "cons_bars", "pullback_pct", "pullback_ratio", "pinch_ratio", "r2",
                "vol_x", "breakout_rvol", "risk_pct_06x", "risk_pct_10x", "dist_52w_high",
                "spy_above_200sma", "spy_above_50sma", "spy_10ema_slope", "qqq_bull_stack"
            ]
            X_s = df_out[s_cols].copy()
            X_s["pinch_ratio"] = X_s["pinch_ratio"].fillna(2.5)
            X_s["r2"] = X_s["r2"].fillna(0.5)
            X_s["dist_52w_high"] = X_s["dist_52w_high"].fillna(-15.0)
            for c in ["spy_above_200sma", "spy_above_50sma", "qqq_bull_stack"]:
                X_s[c] = X_s[c].astype(float)

            if ml_engine.stop_regressor is not None:
                pred_mae = np.clip(ml_engine.stop_regressor.predict(X_s), 0.015, 0.10)
                df_out["pred_mae_pct"] = np.round(pred_mae * 100.0, 2)
                raw_dynamic_stop = df_out["entry_price"] * (1.0 - pred_mae)
                df_out["stop_loss_dynamic"] = np.round(np.minimum(df_out["entry_price"] * 0.985, np.maximum(raw_dynamic_stop, df_out["stop_loss_06x"])), 4)
                df_out["risk_pct_dynamic"] = np.round((df_out["entry_price"] - df_out["stop_loss_dynamic"]) / df_out["entry_price"] * 100.0, 2)

                risk_ratio = np.clip(df_out["risk_pct_06x"] / np.maximum(df_out["risk_pct_dynamic"], 1.5), 1.0, 2.0)
                shakeout = (df_out["is_win_06x"] == True) & (df_out["mae_r_06x"] * df_out["risk_pct_06x"] / 100.0 > pred_mae)

                r_dyn_e10 = np.where(shakeout, -1.0, np.where(df_out["r_e10_06x"] > 0, df_out["r_e10_06x"] * risk_ratio, df_out["r_e10_06x"]))
                df_out["r_dyn_06x"] = np.round(r_dyn_e10, 3)
                df_out["sized_r_dyn_06x"] = np.round(df_out["r_dyn_06x"] * df_out["pos_size_mult"], 3)
                df_out["is_win_dyn_06x"] = df_out["r_dyn_06x"] > 0.05
                df_out["bars_dyn_06x"] = df_out["bars_e10_06x"]

                r_dyn_swing = np.where(shakeout, -1.0, np.where(df_out["r_climax_swing_06x"] > 0, df_out["r_climax_swing_06x"] * risk_ratio, df_out["r_climax_swing_06x"]))
                df_out["r_dyn_climax_swing"] = np.round(r_dyn_swing, 3)
                df_out["sized_r_dyn_climax_swing"] = np.round(df_out["r_dyn_climax_swing"] * df_out["pos_size_mult"], 3)
                df_out["is_win_dyn_climax_swing"] = df_out["r_dyn_climax_swing"] > 0.05
                df_out["bars_dyn_climax_swing"] = df_out["bars_climax_swing_06x"]

                df_out["r_triad_optimal"] = np.round(df_out["r_dyn_climax_swing"] * df_out.get("conviction_sizing", 1.0), 3)
                df_out["sized_r_triad_optimal"] = df_out["r_triad_optimal"]
                df_out["is_win_triad_optimal"] = df_out["r_triad_optimal"] > 0.05
                df_out["bars_triad_optimal"] = df_out["bars_climax_swing_06x"]
        except Exception as e:
            print(f"Warning: Failed ML enrichment: {e}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved institutional QM strategy dataset to {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024 / 1024:.2f} MB).")
    return df_out


def main():
    build_strategy_trades_dataset()


if __name__ == "__main__":
    main()
