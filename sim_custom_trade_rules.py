#!/usr/bin/env python3
"""sim_custom_trade_rules.py — Institutional Simulation & Statistical Feature Pattern Mining.

Tests specific trade rules requested across 7 Key Institutional Archetypes:
1. Pinnacle Elite
2. Emerging Leaders
3. Institutional Sweet Spot
4. Multi Quarter Leaders
5. Idiosyncratic Alpha
6. Conservative Swing
7. High Beta Momentum

Trade Rules Tested:
- Entry: Break above Day 1 High within up to 8 weeks (40 trading days).
- V2 Add/Entry: Entry on Day 5 if V2 conviction holds very good expectancy (ml_50 >= 0.20).
- Stop Loss: ML predicted 20 MAE level below Day 1 Close (dynamic_stop_loss_pct).
- Trailing Exit: Once 10 EMA is sloping upward with price closing well above 10 EMA, exit positions on first close below 10 EMA.

Saves complete per-trade dataset to:
- `data/simulations/custom_trade_rules_study.parquet`
- `data/simulations/custom_trade_rules_study.csv`

Performs deep win-vs-loss pattern mining:
- Sector & Theme outperformance / underperformance
- Market health at EP & Entry (SPY posture, 50 SMA slope, 20d momentum)
- Chart to the LEFT of EP (52W high distance, 20d consolidation tightness, RS score, 50/200 SMA)
- Chart to the RIGHT of EP (absorption, days to breakout, RVOL on breakout)
- Feature importance and high-conviction winning/losing decision rules
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy import stats

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import datastore
import indicators

OUTPUT_PARQUET = ROOT / "data" / "simulations" / "custom_trade_rules_study.parquet"
OUTPUT_CSV = ROOT / "data" / "simulations" / "custom_trade_rules_study.csv"
EP_DATASET_PATH = ROOT / "data" / "simulations" / "ep_combined_study_scored.parquet"

# --- Archetype Definitions ---
TOP_SECTORS = [
    "Computer Hardware", "Biotechnology", "Healthcare", "Technology", "Semiconductors",
    "Industrials", "Capital Markets", "Financial Services", "Specialty Industrial Machinery"
]
TOP_THEMES = [
    "Semiconductors", "Biotechnology", "Software - Infrastructure", "Computer Hardware",
    "Health Information Services", "Medical Devices", "Software - Application"
]
TOP_CLUSTERS = [
    "Computer Hardware", "Semiconductors", "Biotechnology", "Semiconductor Equipment",
    "Bitcoin Miners", "Quantum Computing", "Uranium & Nuclear", "Clean Energy",
    "Software - Application", "Software - Infrastructure", "Aerospace & Defense"
]
HB_THEMES = [
    "Bitcoin Miners", "Quantum Computing", "Semiconductor Equipment", "Clean Energy",
    "Data Centers", "Solar", "Nuclear & Uranium"
]


def assign_archetypes(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Generates boolean masks for all 7 institutional archetypes."""
    df["violated_48h"] = df["violated_48h"].astype(bool)

    masks: Dict[str, pd.Series] = {}

    # 1. Pinnacle Elite
    masks["Pinnacle Elite"] = (
        (~df["violated_48h"]) &
        (df["close_pos"] >= 0.65) &
        (df["rvol"] >= 2.5) &
        (df["gap_pct"] >= 5.0) &
        (df["sector"].isin(TOP_SECTORS) | df["theme"].isin(TOP_THEMES)) &
        ((df["sec_m1_pctile"] >= 50) | (df["thm_m1_pctile"] >= 50))
    )

    # 2. Emerging Leaders
    masks["Emerging Leaders"] = (
        (~df["violated_48h"]) &
        (df["close_pos"] >= 0.65) &
        (df["rvol"] >= 2.5) &
        ((df["sec_m1_pctile"] >= 65) | (df["thm_m1_pctile"] >= 65) | (df["sec_m3_pctile"] >= 65))
    )

    # 3. Institutional Sweet Spot
    masks["Institutional Sweet Spot"] = (
        (~df["violated_48h"]) &
        (df["close_pos"] >= 0.65) &
        (df["rvol"] >= 3.0) &
        (df["gap_pct"] >= 6.0)
    )

    # 4. Multi Quarter Leaders
    masks["Multi Quarter Leaders"] = (
        (~df["violated_48h"]) &
        (df["sector"].isin(TOP_CLUSTERS) | df["theme"].isin(TOP_CLUSTERS))
    )

    # 5. Idiosyncratic Alpha
    is_headwind = (
        ((df["sec_m1_pctile"] < 35) & (df["sec_m3_pctile"] < 35)) |
        (df["thm_m1_pctile"] < 35)
    )
    masks["Idiosyncratic Alpha"] = (
        is_headwind &
        (~df["violated_48h"]) &
        (df["close_pos"] >= 0.80) &
        (df["rvol"] >= 6.0) &
        (df["dvol_m"] >= 75.0)
    )

    # 6. Conservative Swing
    masks["Conservative Swing"] = (
        (~df["violated_48h"]) &
        (df["close_pos"] >= 0.65)
    )

    # 7. High Beta Momentum
    masks["High Beta Momentum"] = (
        (df["theme"].isin(HB_THEMES)) &
        (df["rvol"] >= 3.5)
    )

    return masks


def simulate_trade_for_event(
    bars: pd.DataFrame,
    d1_idx: int,
    row: pd.Series,
    spy_bars: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    """Simulates custom trade rules for a single EP event.
    
    Rules:
    - Entry: Breakout above Day 1 High within 40 bars (8 weeks), OR Day 5 entry if ml_50 >= 0.20.
    - Stop Loss: ML 20 MAE level below Day 1 Close.
    - Exit: Once 10 EMA is sloping up with price well above 10 EMA, exit on first close below 10 EMA.
    """
    n_bars = len(bars)
    if d1_idx >= n_bars - 2:
        return None

    d1_high = float(bars["high"].iloc[d1_idx])
    d1_close = float(bars["close"].iloc[d1_idx])
    d1_low = float(bars["low"].iloc[d1_idx])
    d1_date = bars.index[d1_idx].strftime("%Y-%m-%d")

    # Stop Loss calculation: ML 20 MAE level below Day 1 Close
    dyn_stop = row.get("dynamic_stop_loss_pct", np.nan)
    if pd.isna(dyn_stop) or dyn_stop is None:
        buf = 0.073  # default median
    else:
        buf = abs(float(dyn_stop))
        if buf > 1.0:
            buf = buf / 100.0
        if buf < 0.01:
            buf = 0.073

    stop_price = d1_close * (1.0 - buf)

    # Entry evaluation (up to 40 trading days / 8 weeks forward)
    max_scan_window = min(40, n_bars - d1_idx - 1)
    entry_bar: Optional[int] = None
    entry_price: Optional[float] = None
    entry_type: Optional[str] = None
    days_to_breakout: Optional[int] = None

    ml_50 = float(row.get("ml_50", 0.0))
    v2_expectancy_high = (ml_50 >= 0.20)

    for b in range(1, max_scan_window + 1):
        curr_bar = d1_idx + b
        bar_high = float(bars["high"].iloc[curr_bar])
        bar_open = float(bars["open"].iloc[curr_bar])
        bar_close = float(bars["close"].iloc[curr_bar])

        # Rule A: Break above Day 1 High
        if bar_high > d1_high:
            entry_bar = curr_bar
            entry_price = max(bar_open, d1_high + 0.01)
            entry_type = "D1_High_Breakout"
            days_to_breakout = b
            break

        # Rule B: Day 5 entry if V2 gives very good expectancy (b == 4 is Day 5)
        if b == 4 and v2_expectancy_high:
            entry_bar = curr_bar
            entry_price = bar_close
            entry_type = "Day_5_V2_Expectancy"
            days_to_breakout = b
            break

    if entry_bar is None or entry_price is None:
        return None  # No trigger within 8 weeks

    # Check for Pyramiding on Day 5 if entered earlier (Days 2-4)
    pyramid_added = False
    pyramid_add_px = 0.0
    if days_to_breakout is not None and days_to_breakout < 4 and v2_expectancy_high and (d1_idx + 4) < n_bars:
        d5_bar = d1_idx + 4
        # Add only if price is holding above stop loss
        if float(bars["low"].iloc[d5_bar]) > stop_price:
            pyramid_added = True
            pyramid_add_px = float(bars["close"].iloc[d5_bar])

    # Initial risk calculation
    initial_risk_dollar = entry_price - stop_price
    if initial_risk_dollar <= 0:
        # If entered below stop price on Day 5 or stop is above entry, clamp risk to Day 1 low or 5%
        stop_price = min(float(bars["low"].iloc[entry_bar]), entry_price * 0.95)
        initial_risk_dollar = entry_price - stop_price

    initial_risk_pct = (initial_risk_dollar / entry_price) * 100.0

    # Ensure 10 EMA is computed
    ema10 = bars["ema10"]

    # Trade Forward Simulation
    max_hold_bars = min(250, n_bars - entry_bar - 1)
    if max_hold_bars < 1:
        return None

    exit_bar = entry_bar
    exit_price = entry_price
    exit_reason = "Open / End of Data"
    trend_armed = False
    peak_px = entry_price
    trough_px = entry_price

    # Check Day of Entry for Stop Hit
    entry_day_low = float(bars["low"].iloc[entry_bar])
    entry_day_open = float(bars["open"].iloc[entry_bar])
    if entry_day_low <= stop_price:
        exit_bar = entry_bar
        exit_price = min(entry_day_open, stop_price)
        exit_reason = "Initial Stop Loss Hit (Entry Day)"
    else:
        for f in range(1, max_hold_bars + 1):
            curr_fwd = entry_bar + f
            c = float(bars["close"].iloc[curr_fwd])
            l = float(bars["low"].iloc[curr_fwd])
            h = float(bars["high"].iloc[curr_fwd])
            o = float(bars["open"].iloc[curr_fwd])
            e10 = float(ema10.iloc[curr_fwd])
            e10_prev = float(ema10.iloc[curr_fwd - 1])

            if h > peak_px:
                peak_px = h
            if l < trough_px:
                trough_px = l

            # 1. Stop loss check
            if l <= stop_price:
                exit_bar = curr_fwd
                exit_price = min(o, stop_price)
                exit_reason = "Initial Stop Loss Hit"
                break

            # 2. Check 10 EMA Trend Arming condition:
            # "Once the 10ema is sloping upward with price closing well above 10 ema"
            is_sloping_up = (e10 > e10_prev)
            is_well_above = (c >= e10 * 1.015)  # closed at least 1.5% above 10 EMA

            if is_sloping_up and is_well_above:
                trend_armed = True

            # 3. Exit condition: first close below 10 EMA once armed
            if trend_armed and (c < e10):
                exit_bar = curr_fwd
                exit_price = c
                exit_reason = "Close Below 10 EMA"
                break
        else:
            # Held through max bars without exit
            last_fwd = entry_bar + max_hold_bars
            exit_bar = last_fwd
            exit_price = float(bars["close"].iloc[last_fwd])
            exit_reason = "Max Holding (250d)"

    trade_ret_pct = ((exit_price - entry_price) / entry_price) * 100.0
    trade_r = (exit_price - entry_price) / initial_risk_dollar
    days_in_trade = exit_bar - entry_bar

    # Blended Pyramided R-Multiple if add triggered
    if pyramid_added and pyramid_add_px > 0:
        # Base trade: 1.0 unit. Pyramid add: 0.5 unit. Total risk: 1.0 + 0.5 * (add_risk / base_risk)
        # Net R = base_r + 0.5 * ((exit_price - pyramid_add_px) / initial_risk_dollar)
        add_r = (exit_price - pyramid_add_px) / initial_risk_dollar
        pyramid_r = (trade_r + 0.5 * add_r) / 1.5
    else:
        pyramid_r = trade_r

    mfe_pct = ((peak_px - entry_price) / entry_price) * 100.0
    mae_pct = ((entry_price - trough_px) / entry_price) * 100.0
    peak_r = (peak_px - entry_price) / initial_risk_dollar

    entry_date = bars.index[entry_bar].strftime("%Y-%m-%d")
    exit_date = bars.index[exit_bar].strftime("%Y-%m-%d")

    # --- Chart Features to the LEFT of the EP ---
    left_start = max(0, d1_idx - 250)
    left_bars = bars.iloc[left_start:d1_idx]
    pre_20_bars = bars.iloc[max(0, d1_idx - 20):d1_idx]

    pre_dist_52w_high = 0.0
    pre_stage2 = False
    pre_consol_tightness_20d = 0.0
    pre_runup_20d = 0.0
    pre_runup_65d = 0.0
    pre_volume_dryup_5d = 1.0
    pre_above_sma50 = False
    pre_above_sma200 = False
    pre_sma50_slope = 0.0

    if not left_bars.empty:
        hi_52w = float(left_bars["high"].max())
        d0_close = float(left_bars["close"].iloc[-1])
        pre_dist_52w_high = ((hi_52w - d0_close) / hi_52w) * 100.0 if hi_52w > 0 else 0.0

        if "sma50" in bars.columns and "sma200" in bars.columns:
            s50 = bars["sma50"].iloc[d1_idx - 1]
            s200 = bars["sma200"].iloc[d1_idx - 1]
            if pd.notna(s50) and pd.notna(s200):
                pre_above_sma50 = (d0_close > s50)
                pre_above_sma200 = (d0_close > s200)
                pre_stage2 = (s50 > s200) and pre_above_sma50
                s50_prev20 = bars["sma50"].iloc[max(0, d1_idx - 21)]
                pre_sma50_slope = ((s50 - s50_prev20) / s50_prev20) * 100.0 if pd.notna(s50_prev20) and s50_prev20 > 0 else 0.0

    if not pre_20_bars.empty:
        c20_min = float(pre_20_bars["low"].min())
        c20_max = float(pre_20_bars["high"].max())
        pre_consol_tightness_20d = ((c20_max - c20_min) / d0_close) * 100.0 if d0_close > 0 else 0.0
        pre_runup_20d = ((d0_close - float(pre_20_bars["close"].iloc[0])) / float(pre_20_bars["close"].iloc[0])) * 100.0

    if len(left_bars) >= 65:
        c65 = float(left_bars["close"].iloc[-65])
        pre_runup_65d = ((d0_close - c65) / c65) * 100.0 if c65 > 0 else 0.0

    if len(left_bars) >= 50:
        vol_5d = float(left_bars["volume"].iloc[-5:].mean())
        vol_50d = float(left_bars["volume"].iloc[-50:].mean())
        pre_volume_dryup_5d = (vol_5d / vol_50d) if vol_50d > 0 else 1.0

    # --- Market Health (SPY) at EP and at Entry ---
    spy_above_50_ep = False
    spy_above_200_ep = False
    spy_ret_20d_ep = 0.0
    spy_above_50_entry = False
    spy_ret_20d_entry = 0.0

    if spy_bars is not None and not spy_bars.empty:
        if d1_date in spy_bars.index:
            spy_ep_idx = spy_bars.index.get_loc(d1_date)
            spy_c = float(spy_bars["close"].iloc[spy_ep_idx])
            s50 = spy_bars["sma50"].iloc[spy_ep_idx] if "sma50" in spy_bars.columns else np.nan
            s200 = spy_bars["sma200"].iloc[spy_ep_idx] if "sma200" in spy_bars.columns else np.nan
            spy_above_50_ep = bool(pd.notna(s50) and spy_c > s50)
            spy_above_200_ep = bool(pd.notna(s200) and spy_c > s200)
            if spy_ep_idx >= 20:
                spy_prev20 = float(spy_bars["close"].iloc[spy_ep_idx - 20])
                spy_ret_20d_ep = ((spy_c - spy_prev20) / spy_prev20) * 100.0

        if entry_date in spy_bars.index:
            spy_ent_idx = spy_bars.index.get_loc(entry_date)
            spy_c = float(spy_bars["close"].iloc[spy_ent_idx])
            s50 = spy_bars["sma50"].iloc[spy_ent_idx] if "sma50" in spy_bars.columns else np.nan
            spy_above_50_entry = bool(pd.notna(s50) and spy_c > s50)
            if spy_ent_idx >= 20:
                spy_prev20 = float(spy_bars["close"].iloc[spy_ent_idx - 20])
                spy_ret_20d_entry = ((spy_c - spy_prev20) / spy_prev20) * 100.0

    # --- Chart Features to the RIGHT of the EP (Breakout Day) ---
    breakout_vol = float(bars["volume"].iloc[entry_bar])
    vol_50d_before_entry = float(bars["volume"].iloc[max(0, entry_bar - 50):entry_bar].mean())
    breakout_rvol = (breakout_vol / vol_50d_before_entry) if vol_50d_before_entry > 0 else 1.0
    prev_close = float(bars["close"].iloc[entry_bar - 1])
    breakout_gap_pct = ((float(bars["open"].iloc[entry_bar]) - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0
    shelf_low_before_entry = float(bars["low"].iloc[d1_idx:entry_bar + 1].min())
    shelf_pullback_pct = ((d1_high - shelf_low_before_entry) / d1_high) * 100.0 if d1_high > 0 else 0.0

    # Day 1 candle features
    d1_open = float(bars["open"].iloc[d1_idx])
    d1_rng = d1_high - d1_low if d1_high > d1_low else 1.0
    d1_body_pct = (abs(d1_close - d1_open) / d1_rng) * 100.0
    d1_upper_wick_pct = ((d1_high - max(d1_open, d1_close)) / d1_rng) * 100.0

    return {
        "symbol": row["symbol"],
        "ep_date": d1_date,
        "entry_date": entry_date,
        "exit_date": exit_date,
        "entry_type": entry_type,
        "days_to_breakout": days_to_breakout,
        "entry_price": round(entry_price, 2),
        "stop_price": round(stop_price, 2),
        "exit_price": round(exit_price, 2),
        "initial_risk_pct": round(initial_risk_pct, 2),
        "initial_risk_dollar": round(initial_risk_dollar, 2),
        "return_pct": round(trade_ret_pct, 2),
        "r_multiple": round(trade_r, 3),
        "pyramid_added": pyramid_added,
        "pyramid_r": round(pyramid_r, 3),
        "exit_reason": exit_reason,
        "is_winner": bool(trade_r > 0),
        "is_big_winner": bool(trade_r >= 2.0),
        "is_home_run": bool(trade_r >= 5.0),
        "holding_days": days_in_trade,
        "peak_r": round(peak_r, 2),
        "mfe_pct": round(mfe_pct, 2),
        "mae_pct": round(mae_pct, 2),
        # Setup metadata
        "sector": row.get("sector", "Unknown"),
        "theme": row.get("theme", "Unknown"),
        "gap_pct": round(float(row.get("gap_pct", 0.0)), 2),
        "rvol": round(float(row.get("rvol", 0.0)), 2),
        "dvol_m": round(float(row.get("dvol_m", 0.0)), 1),
        "close_pos": round(float(row.get("close_pos", 0.0)), 2),
        "adr_pct": round(float(row.get("adr_pct", 0.0)), 2),
        "ml_50": round(ml_50, 3),
        "dynamic_stop_loss_pct": round(buf * 100.0, 2),
        "rs_score": float(row.get("rs_score", 50.0)),
        "stage2": bool(row.get("stage2", False)),
        # Chart to the LEFT
        "pre_dist_52w_high": round(pre_dist_52w_high, 2),
        "pre_stage2": pre_stage2,
        "pre_consol_tightness_20d": round(pre_consol_tightness_20d, 2),
        "pre_runup_20d": round(pre_runup_20d, 2),
        "pre_runup_65d": round(pre_runup_65d, 2),
        "pre_volume_dryup_5d": round(pre_volume_dryup_5d, 2),
        "pre_above_sma50": pre_above_sma50,
        "pre_above_sma200": pre_above_sma200,
        "pre_sma50_slope": round(pre_sma50_slope, 2),
        # Chart Day 1
        "day1_body_pct": round(d1_body_pct, 2),
        "day1_upper_wick_pct": round(d1_upper_wick_pct, 2),
        "prev_day_gap_up": bool(row.get("prev_day_gap_up", False)),
        "is_mechanical_volume_date": bool(row.get("is_mechanical_volume_date", False)),
        # Market Health
        "spy_above_50_ep": spy_above_50_ep,
        "spy_above_200_ep": spy_above_200_ep,
        "spy_ret_20d_ep": round(spy_ret_20d_ep, 2),
        "spy_above_50_entry": spy_above_50_entry,
        "spy_ret_20d_entry": round(spy_ret_20d_entry, 2),
        # Chart to the RIGHT
        "breakout_rvol": round(breakout_rvol, 2),
        "breakout_gap_pct": round(breakout_gap_pct, 2),
        "shelf_pullback_pct": round(shelf_pullback_pct, 2),
    }


def run_full_study():
    t_start = time.time()
    print("=" * 80)
    print("🚀 INSTITUTIONAL EP TRADING RULES SIMULATION & PATTERN MINING")
    print("=" * 80)

    print("📖 Loading baseline dataset from:", EP_DATASET_PATH)
    df_ep = pd.read_parquet(EP_DATASET_PATH)
    print(f"Loaded {len(df_ep):,} total EP events across {df_ep['symbol'].nunique():,} symbols.")

    # 1. Generate Archetype Masks
    archetype_masks = assign_archetypes(df_ep)
    for arch, mask in archetype_masks.items():
        print(f"  • {arch:28s}: {mask.sum():5,d} candidate setups")

    # Union of all 7 target archetypes
    target_mask = pd.Series(False, index=df_ep.index)
    for mask in archetype_masks.values():
        target_mask |= mask

    df_target = df_ep[target_mask].copy().reset_index(drop=True)
    print(f"\n🎯 Total unique candidate setups across 7 archetypes: {len(df_target):,}")

    # 2. Pre-load SPY bars
    print("📈 Pre-loading SPY market benchmark bars...")
    spy_bars = datastore.load_bars("SPY")
    if spy_bars is not None and not spy_bars.empty:
        spy_bars["sma50"] = spy_bars["close"].rolling(50).mean()
        spy_bars["sma200"] = spy_bars["close"].rolling(200).mean()
        spy_bars.index = spy_bars.index.strftime("%Y-%m-%d")

    # 3. Simulate Trades
    print("\n⚡ Executing trade simulations with strict zero-lookahead causality...")
    trades: List[Dict[str, Any]] = []
    symbol_cache: Dict[str, pd.DataFrame] = {}

    symbols = df_target["symbol"].unique()
    print(f"Loading bars for {len(symbols)} unique symbols...")
    t0 = time.time()
    for s in symbols:
        b = datastore.load_bars(s)
        if b is not None and len(b) >= 20:
            b = b.copy()
            b["ema10"] = b["close"].ewm(span=10, adjust=False).mean()
            b["sma50"] = b["close"].rolling(50).mean()
            b["sma200"] = b["close"].rolling(200).mean()
            symbol_cache[s] = b
    print(f"Successfully loaded and precomputed indicators for {len(symbol_cache)} symbols in {time.time() - t0:.2f}s.")

    t0 = time.time()
    for idx, row in df_target.iterrows():
        sym = row["symbol"]
        bars = symbol_cache.get(sym)
        if bars is None:
            continue

        d1_str = str(row["date"])[:10]
        # Find integer bar index in bars
        date_matches = np.where(bars.index.strftime("%Y-%m-%d") == d1_str)[0]
        if len(date_matches) == 0:
            continue
        d1_idx = int(date_matches[0])

        trade_record = simulate_trade_for_event(bars, d1_idx, row, spy_bars)
        if trade_record is not None:
            # Tag which archetypes this trade belonged to
            arch_tags = [arch for arch, mask in archetype_masks.items() if mask.iloc[row.name if row.name in df_ep.index else 0]]
            trade_record["archetypes"] = ", ".join(arch_tags)
            trade_record["primary_archetype"] = arch_tags[0] if arch_tags else "General"
            for arch in archetype_masks:
                trade_record[f"is_{arch.lower().replace(' ', '_')}"] = arch in arch_tags

            trades.append(trade_record)

    df_trades = pd.DataFrame(trades)
    print(f"\n✅ Simulation completed: {len(df_trades):,} trades triggered in {time.time() - t0:.2f}s.")

    # 4. Save per-trade dataset to Parquet and CSV
    print("\n💾 Storing full per-trade dataset for audit and pattern discovery...")
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_trades.to_parquet(OUTPUT_PARQUET, index=False)
    df_trades.to_csv(OUTPUT_CSV, index=False)
    print(f"  • Parquet: {OUTPUT_PARQUET} ({OUTPUT_PARQUET.stat().st_size / 1e6:.2f} MB)")
    print(f"  • CSV    : {OUTPUT_CSV} ({OUTPUT_CSV.stat().st_size / 1e6:.2f} MB)")

    # 5. Compute Full Trading Stats Across Archetypes
    print("\n" + "=" * 80)
    print("📊 OVERALL STRATEGY TRADING PERFORMANCE (ALL 7 ARCHETYPES)")
    print("=" * 80)

    def print_stats(sub: pd.DataFrame, label: str):
        n = len(sub)
        if n == 0:
            print(f"{label}: 0 trades.")
            return

        wins = sub[sub["r_multiple"] > 0]
        losses = sub[sub["r_multiple"] <= 0]
        win_rate = (len(wins) / n) * 100.0

        gross_profit = wins["r_multiple"].sum()
        gross_loss = abs(losses["r_multiple"].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.nan

        ev_r = sub["r_multiple"].mean()
        total_r = sub["r_multiple"].sum()
        pyramid_total_r = sub["pyramid_r"].sum()
        pyramid_ev_r = sub["pyramid_r"].mean()

        avg_win_r = wins["r_multiple"].mean() if not wins.empty else 0.0
        avg_loss_r = losses["r_multiple"].mean() if not losses.empty else 0.0
        payoff_ratio = abs(avg_win_r / avg_loss_r) if avg_loss_r != 0 else np.nan

        big_wins = sub[sub["r_multiple"] >= 2.0]
        home_runs = sub[sub["r_multiple"] >= 5.0]

        # Drawdown calculation in R
        cumsum = sub["r_multiple"].cumsum()
        peak = cumsum.cummax()
        dd = cumsum - peak
        max_dd = dd.min()

        print(f"┌─────────────────────────────────────────────────────────────────────────────┐")
        print(f"│ 🎯 {label.upper():67s} │")
        print(f"├─────────────────────────────────────────────┬───────────────────────────────┤")
        print(f"│ Total Trades Triggered                      │ {n:6,d}                        │")
        print(f"│ Trade Win Rate                              │ {win_rate:6.1f}%                       │")
        print(f"│ Expectancy per Trade (EV)                   │ {ev_r:+6.3f} R                     │")
        print(f"│ Total Realized P&L                          │ {total_r:+7.1f} R                    │")
        print(f"│ Total Realized P&L (Pyramid +50%)           │ {pyramid_total_r:+7.1f} R                    │")
        print(f"│ Profit Factor                               │ {profit_factor:6.2f}                        │")
        print(f"│ Payoff Ratio (Avg Win / Avg Loss)           │ {payoff_ratio:6.2f}                        │")
        print(f"│ Average Winner                              │ {avg_win_r:+6.2f} R                     │")
        print(f"│ Average Loser                               │ {avg_loss_r:+6.2f} R                     │")
        print(f"│ Big Winners (>= 2.0 R)                      │ {len(big_wins):5,d} ({len(big_wins)/n*100:4.1f}%)               │")
        print(f"│ Home Runs (>= 5.0 R)                        │ {len(home_runs):5,d} ({len(home_runs)/n*100:4.1f}%)               │")
        print(f"│ Max Drawdown (in R)                         │ {max_dd:6.1f} R                     │")
        print(f"│ Average Holding Time                        │ {sub['holding_days'].mean():5.1f} days                  │")
        print(f"│ 10 EMA Trailing Exits                       │ {(sub['exit_reason'] == 'Close Below 10 EMA').sum():5,d} ({(sub['exit_reason'] == 'Close Below 10 EMA').mean()*100:4.1f}%)               │")
        print(f"│ Initial Stop Hits                           │ {sub['exit_reason'].str.contains('Stop').sum():5,d} ({sub['exit_reason'].str.contains('Stop').mean()*100:4.1f}%)               │")
        print(f"└─────────────────────────────────────────────┴───────────────────────────────┘")

    print_stats(df_trades, "Combined Universe (All 7 Archetypes)")

    # Breakdown by Archetype
    print("\n" + "=" * 80)
    print("🏆 PERFORMANCE BREAKDOWN BY INSTITUTIONAL ARCHETYPE")
    print("=" * 80)
    arch_perf = []
    for arch in archetype_masks:
        col = f"is_{arch.lower().replace(' ', '_')}"
        sub = df_trades[df_trades[col] == True]
        if len(sub) > 0:
            print_stats(sub, arch)
            n = len(sub)
            wins = sub[sub["r_multiple"] > 0]
            losses = sub[sub["r_multiple"] <= 0]
            gross_profit = wins["r_multiple"].sum()
            gross_loss = abs(losses["r_multiple"].sum())
            arch_perf.append({
                "Archetype": arch,
                "Trades": n,
                "Win Rate (%)": round((len(wins) / n) * 100.0, 1),
                "EV (R)": round(sub["r_multiple"].mean(), 3),
                "Total P&L (R)": round(sub["r_multiple"].sum(), 1),
                "Total P&L Pyramid (R)": round(sub["pyramid_r"].sum(), 1),
                "Profit Factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else np.nan,
                "Payoff Ratio": round(abs(wins["r_multiple"].mean() / losses["r_multiple"].mean()), 2) if not losses.empty and losses["r_multiple"].mean() != 0 else np.nan,
                "Avg Win (R)": round(wins["r_multiple"].mean(), 2) if not wins.empty else 0.0,
                "Avg Loser (R)": round(losses["r_multiple"].mean(), 2) if not losses.empty else 0.0,
                "Big Wins (>=2R)": len(sub[sub["r_multiple"] >= 2.0]),
            })

    print("\n" + "=" * 80)
    print("📋 ARCHETYPE LEADERBOARD SUMMARY TABLE")
    print("=" * 80)
    df_arch_summary = pd.DataFrame(arch_perf).sort_values("EV (R)", ascending=False)
    print(df_arch_summary.to_string(index=False))

    # Breakdown by Entry Type (Breakout vs Day 5 V2)
    print("\n" + "=" * 80)
    print("🚪 ENTRY MECHANIC COMPARISON: D1 HIGH BREAKOUT VS DAY 5 V2 EXPECTANCY")
    print("=" * 80)
    for et in df_trades["entry_type"].unique():
        sub_et = df_trades[df_trades["entry_type"] == et]
        print_stats(sub_et, f"Entry Mode: {et}")

    # 6. Deep Pattern Mining: Winners vs Losers
    print("\n" + "=" * 80)
    print("🔬 DEEP STATISTICAL PATTERN DISCOVERY: WINNERS VS LOSERS")
    print("=" * 80)
    analyze_patterns(df_trades)


def analyze_patterns(df: pd.DataFrame):
    """Deep statistical and ML analysis comparing winning trades vs losing trades."""
    winners = df[df["r_multiple"] > 0]
    losers = df[df["r_multiple"] <= 0]
    big_winners = df[df["r_multiple"] >= 2.0]

    print(f"Total Trades Analyzed: {len(df):,} | Winners: {len(winners):,} ({len(winners)/len(df)*100:.1f}%) | Losers: {len(losers):,} ({len(losers)/len(df)*100:.1f}%)")

    # --- 1. Sector & Theme Patterns ---
    print("\n" + "-" * 80)
    print("🌐 1. SECTOR & THEME PATTERNS")
    print("-" * 80)

    # Top Sectors by Win Rate & Total R (min 30 trades)
    sec_stats = []
    for sec, grp in df.groupby("sector"):
        if len(grp) >= 30:
            w = (grp["r_multiple"] > 0).mean() * 100.0
            ev = grp["r_multiple"].mean()
            tot = grp["r_multiple"].sum()
            sec_stats.append({
                "Sector": sec, "Trades": len(grp), "Win Rate %": round(w, 1),
                "EV (R)": round(ev, 3), "Total R": round(tot, 1)
            })
    df_sec = pd.DataFrame(sec_stats).sort_values("EV (R)", ascending=False)
    print("\nTop 8 Best Performing Sectors (min 30 trades):")
    print(df_sec.head(8).to_string(index=False))
    print("\nBottom 5 Worst Performing Sectors (min 30 trades):")
    print(df_sec.tail(5).to_string(index=False))

    # Top Themes by Win Rate & Total R (min 20 trades)
    thm_stats = []
    for thm, grp in df.groupby("theme"):
        if len(grp) >= 20 and thm not in ["Unknown", ""]:
            w = (grp["r_multiple"] > 0).mean() * 100.0
            ev = grp["r_multiple"].mean()
            tot = grp["r_multiple"].sum()
            thm_stats.append({
                "Theme": thm, "Trades": len(grp), "Win Rate %": round(w, 1),
                "EV (R)": round(ev, 3), "Total R": round(tot, 1)
            })
    df_thm = pd.DataFrame(thm_stats).sort_values("EV (R)", ascending=False)
    print("\nTop 8 Best Performing Themes (min 20 trades):")
    print(df_thm.head(8).to_string(index=False))
    print("\nBottom 5 Worst Performing Themes (min 20 trades):")
    print(df_thm.tail(5).to_string(index=False))

    # --- 2. Market Health (SPY) Patterns ---
    print("\n" + "-" * 80)
    print("📈 2. MARKET HEALTH (SPY) IMPACT ON TRADE OUTCOMES")
    print("-" * 80)
    mkt_comparisons = [
        ("SPY Above 50 SMA at EP Event", "spy_above_50_ep"),
        ("SPY Above 200 SMA at EP Event", "spy_above_200_ep"),
        ("SPY Above 50 SMA at Breakout Entry", "spy_above_50_entry"),
    ]
    for label, col in mkt_comparisons:
        t_sub = df[df[col] == True]
        f_sub = df[df[col] == False]
        t_win = (t_sub["r_multiple"] > 0).mean() * 100.0 if not t_sub.empty else 0
        f_win = (f_sub["r_multiple"] > 0).mean() * 100.0 if not f_sub.empty else 0
        t_ev = t_sub["r_multiple"].mean() if not t_sub.empty else 0
        f_ev = f_sub["r_multiple"].mean() if not f_sub.empty else 0
        print(f"  • {label:38s}:")
        print(f"      TRUE  (N={len(t_sub):,}) -> Win Rate: {t_win:4.1f}% | EV: {t_ev:+5.3f} R | Total: {t_sub['r_multiple'].sum():+6.1f} R")
        print(f"      FALSE (N={len(f_sub):,}) -> Win Rate: {f_win:4.1f}% | EV: {f_ev:+5.3f} R | Total: {f_sub['r_multiple'].sum():+6.1f} R")

    # SPY 20d momentum buckets
    print("\nSPY 20-Day Momentum at Breakout Entry:")
    df["spy_mom_bucket"] = pd.qcut(df["spy_ret_20d_entry"], q=4, labels=["Strong Headwind (Q1)", "Mild Headwind (Q2)", "Mild Tailwind (Q3)", "Strong Tailwind (Q4)"])
    for bucket, grp in df.groupby("spy_mom_bucket", observed=False):
        w = (grp["r_multiple"] > 0).mean() * 100.0
        ev = grp["r_multiple"].mean()
        print(f"  • {bucket:24s} (N={len(grp):,}): Win Rate: {w:4.1f}% | EV: {ev:+5.3f} R | Total: {grp['r_multiple'].sum():+6.1f} R")

    # --- 3. Chart to the LEFT of the EP Event ---
    print("\n" + "-" * 80)
    print("👈 3. CHART TO THE LEFT OF THE EP EVENT (PRE-EP ATTRIBUTES)")
    print("-" * 80)

    left_features = [
        ("Distance to 52-Week High (%)", "pre_dist_52w_high"),
        ("Relative Strength Score (1-99)", "rs_score"),
        ("Pre-EP Stage 2 Bullish", "pre_stage2"),
        ("20D Consolidation Tightness (%)", "pre_consol_tightness_20d"),
        ("Pre-EP 20D Runup (%)", "pre_runup_20d"),
        ("Pre-EP 65D Runup (%)", "pre_runup_65d"),
        ("5D Volume Dryup Ratio", "pre_volume_dryup_5d"),
        ("Pre-EP Above 50 SMA", "pre_above_sma50"),
        ("Pre-EP Above 200 SMA", "pre_above_sma200"),
        ("Pre-EP 50 SMA 20D Slope (%)", "pre_sma50_slope"),
    ]

    print(f"{'Feature':32s} | {'Winners Mean':13s} | {'Losers Mean':13s} | {'Diff / Effect':15s} | {'p-value':8s}")
    print("-" * 88)
    for name, col in left_features:
        if df[col].dtype == bool:
            w_val = winners[col].mean() * 100.0
            l_val = losers[col].mean() * 100.0
            diff = w_val - l_val
            # Chi-square or t-test
            _, p_val = stats.ttest_ind(winners[col].astype(float), losers[col].astype(float), equal_var=False)
            print(f"{name:32s} | {w_val:11.1f}% | {l_val:11.1f}% | {diff:+11.1f}% pts  | {p_val:.4f}")
        else:
            w_val = winners[col].mean()
            l_val = losers[col].mean()
            diff = w_val - l_val
            _, p_val = stats.ttest_ind(winners[col].dropna(), losers[col].dropna(), equal_var=False)
            print(f"{name:32s} | {w_val:13.2f} | {l_val:13.2f} | {diff:+15.2f} | {p_val:.4f}")

    # --- 4. Chart at EP Event (Day 1) & To the RIGHT (Breakout) ---
    print("\n" + "-" * 80)
    print("👉 4. CHART AT EP EVENT & TO THE RIGHT (DAY 1 & BREAKOUT ATTRIBUTES)")
    print("-" * 80)

    right_features = [
        ("Day 1 Gap %", "gap_pct"),
        ("Day 1 RVOL", "rvol"),
        ("Day 1 Dollar Volume ($M)", "dvol_m"),
        ("Day 1 Close Position in Range", "close_pos"),
        ("Day 1 Candle Body %", "day1_body_pct"),
        ("Day 1 Upper Wick %", "day1_upper_wick_pct"),
        ("ML 20 MAE Stop Distance (%)", "dynamic_stop_loss_pct"),
        ("V2 Conviction P(+50%)", "ml_50"),
        ("Days from EP to Breakout", "days_to_breakout"),
        ("Breakout Day RVOL", "breakout_rvol"),
        ("Breakout Day Gap %", "breakout_gap_pct"),
        ("Digestion Pullback from D1 High (%)", "shelf_pullback_pct"),
    ]

    print(f"{'Feature':36s} | {'Winners Mean':13s} | {'Losers Mean':13s} | {'Diff / Effect':15s} | {'p-value':8s}")
    print("-" * 92)
    for name, col in right_features:
        w_val = winners[col].mean()
        l_val = losers[col].mean()
        diff = w_val - l_val
        _, p_val = stats.ttest_ind(winners[col].dropna(), losers[col].dropna(), equal_var=False)
        print(f"{name:36s} | {w_val:13.2f} | {l_val:13.2f} | {diff:+15.2f} | {p_val:.4f}")

    # Days to breakout analysis
    print("\nDays to Breakout Breakdown:")
    df["breakout_timing_bucket"] = pd.cut(
        df["days_to_breakout"],
        bins=[-1, 2, 5, 10, 20, 41],
        labels=["Immediate (Days 1-2)", "Fast (Days 3-5)", "Medium (Days 6-10)", "Base Consolidation (Days 11-20)", "Extended Base (Days 21-40)"]
    )
    for bucket, grp in df.groupby("breakout_timing_bucket", observed=False):
        w = (grp["r_multiple"] > 0).mean() * 100.0
        ev = grp["r_multiple"].mean()
        tot = grp["r_multiple"].sum()
        print(f"  • {bucket:32s} (N={len(grp):4,d}): Win Rate: {w:4.1f}% | EV: {ev:+5.3f} R | Total: {tot:+7.1f} R")

    # --- 5. High-Impact Filters: What Excludes Losers and What Boosts Winners ---
    print("\n" + "=" * 80)
    print("💡 5. ACTIONABLE FILTER DISCOVERY: RULES THAT BOOST WINNERS & EXCLUDE LOSERS")
    print("=" * 80)

    candidate_filters = [
        ("Base Strategy (No Additional Filters)", df.index),
        ("Rule: RS Score >= 80", df[df["rs_score"] >= 80].index),
        ("Rule: RS Score >= 90", df[df["rs_score"] >= 90].index),
        ("Rule: Pre-EP Within 15% of 52W High", df[df["pre_dist_52w_high"] <= 15.0].index),
        ("Rule: Pre-EP Stage 2 Bullish (SMA50 > SMA200)", df[df["pre_stage2"] == True].index),
        ("Rule: Pre-EP Tight Base (20d range <= 20%)", df[df["pre_consol_tightness_20d"] <= 20.0].index),
        ("Rule: SPY Above 50 SMA at Breakout Entry", df[df["spy_above_50_entry"] == True].index),
        ("Rule: Exclude Mechanical Volume Spikes (OpEx/Quad)", df[df["is_mechanical_volume_date"] == False].index),
        ("Rule: Fast Breakout (Within 10 Trading Days)", df[df["days_to_breakout"] <= 10].index),
        ("Rule: High V2 Conviction (ml_50 >= 0.25)", df[df["ml_50"] >= 0.25].index),
        ("Rule: Tight ML Dynamic Stop (Stop <= 10%)", df[df["dynamic_stop_loss_pct"] <= 10.0].index),
        ("COMBO: RS >= 85 + Pre-EP Within 20% of 52W High", df[(df["rs_score"] >= 85) & (df["pre_dist_52w_high"] <= 20.0)].index),
        ("COMBO: RS >= 85 + SPY Above 50 SMA at Entry", df[(df["rs_score"] >= 85) & (df["spy_above_50_entry"] == True)].index),
        ("COMBO: Stage 2 + RS >= 80 + Breakout <= 10 Days", df[(df["pre_stage2"] == True) & (df["rs_score"] >= 80) & (df["days_to_breakout"] <= 10)].index),
    ]

    print(f"{'Filter Description':48s} | {'Trades':7s} | {'Win Rate':9s} | {'EV (R)':8s} | {'Profit Factor':13s} | {'Total Realized R':16s}")
    print("-" * 115)
    for label, idxs in candidate_filters:
        sub = df.loc[idxs]
        n = len(sub)
        if n == 0:
            continue
        w = (sub["r_multiple"] > 0).mean() * 100.0
        ev = sub["r_multiple"].mean()
        tot = sub["r_multiple"].sum()
        wins = sub[sub["r_multiple"] > 0]
        losses = sub[sub["r_multiple"] <= 0]
        gp = wins["r_multiple"].sum()
        gl = abs(losses["r_multiple"].sum())
        pf = (gp / gl) if gl > 0 else np.nan
        print(f"{label:48s} | {n:7,d} | {w:7.1f}% | {ev:+7.3f} | {pf:13.2f} | {tot:+14.1f} R")


if __name__ == "__main__":
    run_full_study()
