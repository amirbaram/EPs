"""sim_leading_qm_breakouts.py — Quantitative Study: Leading Sectors/Themes QM Breakouts.

Trade Rules:
1. Universe: Leading Sectors and Leading Themes only.
2. Market Filter: SPY >= 200 SMA on entry date.
3. Stoploss: 0.6 x distance between breakout close and previous confirmed TrendLab pivot low.
   stop_dist = 0.60 * (entry_price - pivot_low)
   stop_loss = entry_price - stop_dist
4. Trailing Exit: Once 10 EMA is sloping upward with price closing well above 10 EMA (>= 1.0%),
   exit on first close below 10 EMA.
5. Position Sizing Multiplier (S):
   Based on recency of fresh EP, size of pole, breakout RVOL, and pullback ratio.
   Sized R = S * R_multiple.
6. Market Regime Analytics:
   - QQQ > EMA10 > EMA20
   - (QQQ/SPY) > EMA10 > EMA20
   - Stock Relative Strength (RS 1-99) & Technical Posture (Stage 2, Near 52W High)

Saves results to: `data/simulations/qm_leading_breakouts_trades.parquet`.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import config
import datastore
import indicators
import labels
import serve_qm
import trendlab

SETUPS_PATH = Path("data/simulations/qm_historical_setups.parquet")
OUTPUT_PATH = Path("data/simulations/qm_leading_breakouts_trades.parquet")


def compute_market_and_qqq_lookup() -> Dict[str, Dict[str, Any]]:
    """Precompute SPY, QQQ, and QQQ/SPY ratio indicators for all dates."""
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
        
        # QQQ > 10 EMA > 20 EMA
        qqq_bull = bool(c_qqq >= e10_qqq >= e20_qqq)
        
        # QQQ/SPY > 10 EMA > 20 EMA
        ratio_bull = bool(r_val[i] >= r_e10[i] >= r_e20[i])
        
        lookup[d] = {
            "spy_close": round(c_spy, 2),
            "spy_above_200sma": bool(c_spy >= s200_spy),
            "spy_above_50sma": bool(c_spy >= s50_spy),
            "spy_10ema_slope": round((e10_spy - prev_e10_spy) / prev_e10_spy * 100.0, 3) if prev_e10_spy > 0 else 0.0,
            "qqq_bull_stack": qqq_bull,
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
    # 1. Recency of Fresh EP
    if prior_ep_days_ago is not None and not np.isnan(prior_ep_days_ago):
        if prior_ep_days_ago <= 15:
            s_ep = 1.30  # Fresh EP within 15 days: +30%
        elif prior_ep_days_ago <= 30:
            s_ep = 1.15  # Fresh EP within 30 days: +15%
        elif prior_ep_days_ago <= 60:
            s_ep = 1.05  # Within 60 days: +5%
        else:
            s_ep = 0.90
    else:
        s_ep = 0.85  # No EP catalyst penalty
        
    # 2. Size of Pole
    if move_pct >= 100.0:
        s_pole = 1.25  # Monster Pole >= 100%: +25%
    elif move_pct >= 50.0:
        s_pole = 1.10  # Strong Pole 50-100%: +10%
    else:
        s_pole = 1.00  # Standard Pole 30-50%: 1.0x baseline
        
    # 3. Breakout RVOL
    if breakout_rvol >= 2.5:
        s_rvol = 1.20  # Massive volume explosion >= 2.5x: +20%
    elif breakout_rvol >= 1.5:
        s_rvol = 1.10  # Healthy volume expansion >= 1.5x: +10%
    else:
        s_rvol = 0.90  # Sub-1.5x volume: -10%
        
    # 4. Pullback Ratio (pullback_pct / move_pct)
    if pullback_ratio <= 0.25:
        s_pb = 1.20  # Very tight absorption (<= 25% of move): +20%
    elif pullback_ratio <= 0.35:
        s_pb = 1.05  # Orderly consolidation: +5%
    else:
        s_pb = 0.85  # Loose digestion (> 35% of move): -15%
        
    raw_size = s_ep * s_pole * s_rvol * s_pb
    # Clamp between 0.50x and 2.50x
    clamped_size = float(np.clip(raw_size, 0.50, 2.50))
    
    details = {
        "s_ep": s_ep,
        "s_pole": s_pole,
        "s_rvol": s_rvol,
        "s_pb": s_pb,
        "size_multiplier": round(clamped_size, 3),
    }
    return clamped_size, details


def run_leading_qm_study():
    print("==========================================================================================")
    print("🚀 QUANTITATIVE STUDY: LEADING SECTORS & THEMES QM BREAKOUT STRATEGY")
    print("   Filters: Leading Sectors/Themes | SPY >= 200 SMA")
    print("   Stoploss: 0.6x Distance to Previous TrendLab Pivot Low")
    print("   Position Sizing: Dynamic Multiplier (Fresh EP, Pole Size, RVOL, Pullback Ratio)")
    print("   Exit: First Close Below 10 EMA (Once 10 EMA is Rising & Price Closed Well Above)")
    print("==========================================================================================")
    
    df_setups = pd.read_parquet(SETUPS_PATH)
    bks = df_setups[df_setups["state"] == "breakout"].copy()
    print(f"Total QM Breakouts in database: {len(bks)} across {bks['symbol'].nunique()} symbols.")
    
    # Identify Leading Sectors & Leading Themes
    lead_secs, lead_thms, emerg_thms = serve_qm.get_thematic_macro_sets()
    print(f"Loaded {len(lead_secs)} Leading Sectors and {len(lead_thms)} Leading Themes.")
    
    # Filter 1: Universe of Leading Sectors OR Leading Themes
    def is_leading_macro(row):
        sec = str(row.get("sector", ""))
        thm = str(row.get("theme", ""))
        thms = str(row.get("themes", ""))
        if sec in lead_secs:
            return True
        if thm in lead_thms or any(t in thms for t in lead_thms):
            return True
        return False
        
    bks["is_leading_macro"] = bks.apply(is_leading_macro, axis=1)
    bks_macro = bks[bks["is_leading_macro"]].copy()
    print(f"Filtered to Leading Sectors & Themes: {len(bks_macro)} breakouts.")
    
    # Precompute Market Health (SPY, QQQ, QQQ/SPY)
    mkt_lookup = compute_market_and_qqq_lookup()
    print(f"Precomputed Market & QQQ indicators for {len(mkt_lookup)} trading dates.")
    
    # Filter 2: SPY >= 200 SMA on entry date
    valid_bks = []
    for _, row in bks_macro.iterrows():
        d = row["date"]
        mkt = mkt_lookup.get(d)
        if mkt is not None:
            if mkt.get("spy_above_200sma", False):
                valid_bks.append(row)
        else:
            # If historical date precedes SPY data, include if date < 2016
            valid_bks.append(row)
            
    df_study = pd.DataFrame(valid_bks)
    print(f"Filtered out SPY < 200 SMA (Bear Markets): {len(df_study)} qualifying breakouts remaining.")
    
    grouped = df_study.groupby("symbol")
    all_syms = sorted(grouped.groups.keys())
    
    labels.load()
    trades: List[Dict[str, Any]] = []
    trade_id = 0
    t0 = time.time()
    
    total_syms = len(all_syms)
    print(f"Simulating trades across {total_syms} unique symbols...")
    
    for i_sym, sym in enumerate(all_syms):
        if (i_sym + 1) % 150 == 0 or (i_sym + 1) == total_syms:
            elapsed = time.time() - t0
            rate = (i_sym + 1) / elapsed if elapsed > 0 else 0
            sys.stdout.write(f"\rProgress: {i_sym + 1}/{total_syms} symbols ({(i_sym + 1)/total_syms*100:.1f}%) | Trades: {len(trades)} | {rate:.1f} sym/s")
            sys.stdout.flush()
            
        sym_bks = grouped.get_group(sym)
        df_bars = datastore.load_bars(sym)
        if df_bars is None or len(df_bars) < 60:
            continue
            
        df_bars = indicators.add_indicators(df_bars)
        
        try:
            r_tl = trendlab._compute(df_bars)
            tl_pidx = r_tl["pidx"]
            tl_pprice = r_tl["pprice"]
            tl_ptype = r_tl["ptype"]
            tl_pconf = r_tl["pconf"]
        except Exception:
            tl_pidx, tl_pprice, tl_ptype, tl_pconf = [], [], [], []
            
        dates_list = df_bars.index.strftime("%Y-%m-%d").tolist()
        date_to_idx = {d: i for i, d in enumerate(dates_list)}
        
        close_arr = df_bars["close"].to_numpy(float)
        open_arr = df_bars["open"].to_numpy(float)
        high_arr = df_bars["high"].to_numpy(float)
        low_arr = df_bars["low"].to_numpy(float)
        vol_arr = df_bars["volume"].to_numpy(float)
        ema10_arr = df_bars["ema10"].to_numpy(float)
        vol_ma20_arr = df_bars["volume"].rolling(20, min_periods=5).mean().to_numpy(float)
        n_bars = len(df_bars)
        
        for _, setup in sym_bks.iterrows():
            bk_date = setup["date"]
            if bk_date not in date_to_idx:
                continue
            idx = date_to_idx[bk_date]
            if idx >= n_bars - 1:
                continue
                
            entry_p = float(close_arr[idx])
            
            # Find previous confirmed TrendLab pivot low (pconf <= idx, pidx <= idx)
            conf_lows = [
                (tl_pidx[k], float(tl_pprice[k]))
                for k in range(len(tl_pidx))
                if tl_ptype[k] == trendlab.PIVOT_LOW and tl_pconf[k] <= idx and tl_pidx[k] <= idx
            ]
            
            if conf_lows:
                pivot_low_idx, pivot_low_p = conf_lows[-1]
                pivot_low_dt = dates_list[pivot_low_idx]
                pivot_low_bars_ago = idx - pivot_low_idx
            else:
                c_start_idx = max(0, idx - int(setup.get("cons_bars", 15)))
                pivot_low_p = float(np.min(low_arr[c_start_idx : idx + 1]))
                pivot_low_idx = c_start_idx
                pivot_low_dt = dates_list[pivot_low_idx]
                pivot_low_bars_ago = idx - pivot_low_idx
                
            if pivot_low_p >= entry_p:
                pivot_low_p = float(low_arr[idx]) * 0.985
                
            # Full pivot distance D
            full_pivot_dist = entry_p - pivot_low_p
            if full_pivot_dist <= 0.001:
                continue
                
            # STOPLOSS RULE: 0.6 x distance between breakout and previous pivot low!
            stop_dist = 0.60 * full_pivot_dist
            stop_loss = round(entry_p - stop_dist, 4)
            initial_risk = stop_dist  # unit of 1R
            risk_pct = (initial_risk / entry_p) * 100.0
            
            # Breakout day candle features
            bk_vol = vol_arr[idx]
            bk_vol_ma = vol_ma20_arr[idx] if vol_ma20_arr[idx] > 0 else 1.0
            bk_rvol = bk_vol / bk_vol_ma
            
            move_pct = float(setup.get("move_pct", 30.0))
            pullback_pct = float(setup.get("pullback_pct", 15.0))
            pullback_ratio = pullback_pct / max(1.0, move_pct)
            prior_ep_days = setup.get("prior_ep_days_ago")
            
            # POSITION SIZING RULE
            pos_size_mult, size_details = calculate_position_size_multiplier(
                prior_ep_days_ago=prior_ep_days,
                move_pct=move_pct,
                breakout_rvol=bk_rvol,
                pullback_ratio=pullback_ratio,
            )
            
            # --- Trade Simulation ---
            exit_p = None
            exit_dt = None
            exit_idx = None
            exit_reason = None
            trail_active = False
            trail_act_bar = None
            
            peak_high = entry_p
            trough_low = entry_p
            max_forward_bars = min(n_bars - 1, idx + 250)
            
            for t in range(idx + 1, max_forward_bars + 1):
                cur_low = low_arr[t]
                cur_high = high_arr[t]
                cur_open = open_arr[t]
                cur_close = close_arr[t]
                cur_e10 = ema10_arr[t]
                prev_e10 = ema10_arr[t - 1]
                
                peak_high = max(peak_high, cur_high)
                trough_low = min(trough_low, cur_low)
                
                # Rule 1: Stoploss breach (0.6x pivot low stop)
                if cur_low <= stop_loss:
                    exit_p = min(cur_open, stop_loss)
                    exit_dt = dates_list[t]
                    exit_idx = t
                    exit_reason = "stop_loss_0.6x"
                    break
                    
                # Rule 2: 10 EMA Sloping Upward with Close Well Above (>= 1.0%)
                ema_rising = (cur_e10 > prev_e10)
                well_above = (cur_close >= cur_e10 * 1.01)
                
                if ema_rising and well_above and not trail_active:
                    trail_active = True
                    trail_act_bar = t - idx
                    
                # Rule 3: Trailing Exit on First Close Below 10 EMA
                if trail_active and (cur_close < cur_e10):
                    exit_p = cur_close
                    exit_dt = dates_list[t]
                    exit_idx = t
                    exit_reason = "close_below_10ema"
                    break
                    
            if exit_p is None:
                last_t = max_forward_bars
                exit_p = close_arr[last_t]
                exit_dt = dates_list[last_t]
                exit_idx = last_t
                exit_reason = "max_holding_or_end"
                
            realized_r = (exit_p - entry_p) / initial_risk
            sized_r = realized_r * pos_size_mult
            gain_pct = (exit_p / entry_p - 1.0) * 100.0
            max_mfe_r = (peak_high - entry_p) / initial_risk
            max_mae_r = (entry_p - trough_low) / initial_risk
            bars_held = exit_idx - idx
            
            # Market / QQQ context
            mkt = mkt_lookup.get(bk_date, {})
            qqq_bull = mkt.get("qqq_bull_stack")
            qqq_spy_bull = mkt.get("qqq_spy_ratio_bull")
            
            # RS technicals
            rs_sc = setup.get("rs_score")
            rs_sc = float(rs_sc) if pd.notna(rs_sc) else None
            st2 = bool(setup.get("stage2", False))
            d52 = setup.get("dist_52w_high")
            d52 = float(d52) if pd.notna(d52) else None
            is_leader = (rs_sc is not None and rs_sc >= 90.0) and (st2 or (d52 is not None and d52 >= -15.0))
            
            trade_id += 1
            rec = {
                "trade_id": trade_id,
                "symbol": sym,
                "entry_date": bk_date,
                "entry_price": round(entry_p, 4),
                "stop_loss": round(stop_loss, 4),
                "full_pivot_dist": round(full_pivot_dist, 4),
                "stop_dist_06x": round(initial_risk, 4),
                "risk_pct": round(risk_pct, 2),
                "pivot_low_price": round(pivot_low_p, 4),
                "pivot_low_date": pivot_low_dt,
                "exit_date": exit_dt,
                "exit_price": round(exit_p, 4),
                "exit_reason": exit_reason,
                "bars_in_trade": bars_held,
                "trail_activated": trail_active,
                "trail_act_bar": trail_act_bar,
                # Outcomes
                "r_multiple": round(realized_r, 3),
                "sized_r_multiple": round(sized_r, 3),
                "pos_size_mult": round(pos_size_mult, 3),
                "gain_pct": round(gain_pct, 2),
                "is_win": bool(realized_r > 0),
                "is_big_win": bool(realized_r >= 3.0),
                "is_monster_win": bool(realized_r >= 5.0),
                "is_scratch": bool(abs(realized_r) <= 0.3),
                "is_full_loss": bool(realized_r <= -0.9),
                "max_mfe_r": round(max_mfe_r, 3),
                "max_mae_r": round(max_mae_r, 3),
                # Setup Features
                "move_pct": round(move_pct, 1),
                "cons_bars": int(setup.get("cons_bars", 0)),
                "pullback_pct": round(pullback_pct, 1),
                "pullback_ratio": round(pullback_ratio, 3),
                "ride_frac": round(float(setup.get("ride_frac", 0.0)), 2),
                "vol_x": round(float(setup.get("vol_x", 1.0)), 2) if pd.notna(setup.get("vol_x")) else 1.0,
                "breakout_rvol": round(bk_rvol, 2),
                "pattern": str(setup.get("pattern", "other")),
                # EP Catalysts
                "has_prior_ep": bool(setup.get("has_prior_ep", False)),
                "prior_ep_days_ago": prior_ep_days,
                "prior_ep_gap_pct": setup.get("prior_ep_gap_pct"),
                "prior_ep_rvol": setup.get("prior_ep_rvol"),
                "ep_badge": setup.get("ep_badge", "Pure Technical"),
                # Institutional RS & Posture
                "rs_score": rs_sc,
                "stage2": st2,
                "dist_52w_high": d52,
                "rs_leader": is_leader,
                # Sectors & Themes
                "sector": str(setup.get("sector", "Unknown")),
                "theme": str(setup.get("theme", "General")),
                # Market / QQQ Indicators
                "qqq_bull_stack": qqq_bull,
                "qqq_spy_ratio_bull": qqq_spy_bull,
                "spy_above_50sma": mkt.get("spy_above_50sma"),
                "spy_10ema_slope": mkt.get("spy_10ema_slope"),
            }
            trades.append(rec)
            
    print()
    print(f"Simulation completed in {time.time() - t0:.1f}s. Simulated {len(trades)} trades.")
    df_trades = pd.DataFrame(trades)
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_trades.to_parquet(OUTPUT_PATH, index=False)
    print(f"✅ Saved leading trades dataset to {OUTPUT_PATH} ({len(df_trades):,} records).")
    return df_trades


if __name__ == "__main__":
    run_leading_qm_study()
