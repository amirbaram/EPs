"""build_qm_strategy_dataset.py — Builds Institutional QM Strategy Trades Dataset.

Generates `data/simulations/qm_strategy_trades.parquet` containing trade execution details,
dual stoploss models (0.6x and 1.0x), dynamic position sizing factors, QQQ/SPY regime filters,
and institutional RS technicals for all QM breakout setups.
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
        
        # QQQ > 10 EMA > 20 EMA
        qqq_bull = bool(c_qqq >= e10_qqq >= e20_qqq)
        qqq_pullback = bool(c_qqq < e10_qqq or c_qqq < e20_qqq)
        
        # QQQ/SPY > 10 EMA > 20 EMA
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
    # 1. Recency of Fresh EP
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
        
    # 2. Size of Pole
    if move_pct >= 100.0:
        s_pole = 1.25
    elif move_pct >= 50.0:
        s_pole = 1.10
    else:
        s_pole = 1.00
        
    # 3. Breakout RVOL
    if breakout_rvol >= 2.5:
        s_rvol = 1.20
    elif breakout_rvol >= 1.5:
        s_rvol = 1.10
    else:
        s_rvol = 0.90
        
    # 4. Pullback Ratio
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


def simulate_single_trade(
    idx: int,
    entry_p: float,
    stop_p: float,
    low_arr: np.ndarray,
    high_arr: np.ndarray,
    open_arr: np.ndarray,
    close_arr: np.ndarray,
    ema10_arr: np.ndarray,
    dates_list: List[str],
    stop_reason: str,
    max_forward_bars: int = 250,
) -> Dict[str, Any]:
    """Simulates a trade forward until stoploss breach or 10 EMA upward slope violation."""
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
        cur_e10 = ema10_arr[t]
        prev_e10 = ema10_arr[t - 1]
        
        peak_high = max(peak_high, cur_high)
        trough_low = min(trough_low, cur_low)
        
        # 1. Stoploss breach
        if cur_low <= stop_p:
            exit_p = min(cur_open, stop_p)
            exit_dt = dates_list[t]
            exit_idx = t
            exit_reason = stop_reason
            break
            
        # 2. 10 EMA Trailing Exit
        # Sloping upward & closed well above (>= 1.0% above 10 EMA)
        if cur_e10 > prev_e10 and (cur_close / cur_e10 - 1.0) >= 0.01:
            trail_active = True
            
        if trail_active:
            if cur_close < cur_e10:
                exit_p = cur_close
                exit_dt = dates_list[t]
                exit_idx = t
                exit_reason = "10ema_close_violation"
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
        "bars_in_trade": bars_held,
        "r_multiple": round(r_multiple, 3),
        "gain_pct": round(gain_pct, 2),
        "max_mfe_r": round(max_mfe_r, 2),
        "max_mae_r": round(max_mae_r, 2),
        "is_win": bool(r_multiple > 0.05),
    }


def build_dataset():
    print(f"Loading setups from {SETUPS_PATH}...")
    df_setups = pd.read_parquet(SETUPS_PATH)
    bks = df_setups[df_setups["state"] == "breakout"].copy()
    print(f"Total QM Breakouts found: {len(bks)}.")
    
    # Enrich with RS technicals if not present
    if "rs_score" not in bks.columns or "stage2" not in bks.columns or "dist_52w_high" not in bks.columns:
        bks = serve_qm._enrich_with_rs_technicals(bks)
        
    lead_secs, lead_thms, emerg_thms = serve_qm.get_thematic_macro_sets()
    print(f"Loaded {len(lead_secs)} Leading Sectors and {len(lead_thms)} Leading Themes.")
    
    mkt_lookup = compute_market_and_qqq_lookup()
    print(f"Market health lookup computed for {len(mkt_lookup)} trading dates.")
    
    labels.load()
    grouped = bks.groupby("symbol")
    all_syms = sorted(grouped.groups.keys())
    total_syms = len(all_syms)
    
    records: List[Dict[str, Any]] = []
    t0 = time.time()
    
    print(f"Simulating trades across {total_syms} symbols...")
    
    for i_sym, sym in enumerate(all_syms):
        if (i_sym + 1) % 150 == 0 or (i_sym + 1) == total_syms:
            elapsed = time.time() - t0
            rate = (i_sym + 1) / elapsed if elapsed > 0 else 0
            sys.stdout.write(f"\rProgress: {i_sym + 1}/{total_syms} symbols ({(i_sym + 1)/total_syms*100:.1f}%) | Records: {len(records)} | {rate:.1f} sym/s")
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
            
            # Find previous confirmed TrendLab pivot low
            conf_lows = [
                (tl_pidx[k], float(tl_pprice[k]))
                for k in range(len(tl_pidx))
                if tl_ptype[k] == trendlab.PIVOT_LOW and tl_pconf[k] <= idx and tl_pidx[k] <= idx
            ]
            
            if conf_lows:
                pivot_low_idx, pivot_low_p = conf_lows[-1]
                pivot_low_dt = dates_list[pivot_low_idx]
            else:
                c_start_idx = max(0, idx - int(setup.get("cons_bars", 15)))
                pivot_low_p = float(np.min(low_arr[c_start_idx : idx + 1]))
                pivot_low_idx = c_start_idx
                pivot_low_dt = dates_list[pivot_low_idx]
                
            if pivot_low_p >= entry_p:
                pivot_low_p = float(low_arr[idx]) * 0.985
                
            full_pivot_dist = entry_p - pivot_low_p
            if full_pivot_dist <= 0.001:
                continue
                
            # 0.6x Stoploss
            stop_06x = round(entry_p - 0.60 * full_pivot_dist, 4)
            # 1.0x Full Pivot Stoploss
            stop_10x = round(entry_p - 1.00 * full_pivot_dist, 4)
            
            # Breakout day candle features
            bk_vol = vol_arr[idx]
            bk_vol_ma = vol_ma20_arr[idx] if vol_ma20_arr[idx] > 0 else 1.0
            bk_rvol = round(float(bk_vol / bk_vol_ma), 2)
            
            move_pct = float(setup.get("move_pct", 30.0))
            pullback_pct = float(setup.get("pullback_pct", 15.0))
            pullback_ratio = round(pullback_pct / max(1.0, move_pct), 3)
            prior_ep_days = setup.get("prior_ep_days_ago")
            
            # Dynamic position sizing multiplier
            pos_size_mult, size_details = calculate_position_size_multiplier(
                prior_ep_days_ago=prior_ep_days,
                move_pct=move_pct,
                breakout_rvol=bk_rvol,
                pullback_ratio=pullback_ratio,
            )
            
            # Simulate 0.6x stoploss trade
            t_06x = simulate_single_trade(
                idx=idx,
                entry_p=entry_p,
                stop_p=stop_06x,
                low_arr=low_arr,
                high_arr=high_arr,
                open_arr=open_arr,
                close_arr=close_arr,
                ema10_arr=ema10_arr,
                dates_list=dates_list,
                stop_reason="stop_loss_0.6x",
            )
            
            # Simulate 1.0x stoploss trade
            t_10x = simulate_single_trade(
                idx=idx,
                entry_p=entry_p,
                stop_p=stop_10x,
                low_arr=low_arr,
                high_arr=high_arr,
                open_arr=open_arr,
                close_arr=close_arr,
                ema10_arr=ema10_arr,
                dates_list=dates_list,
                stop_reason="stop_loss_1.0x",
            )
            
            # Market context
            mkt = mkt_lookup.get(bk_date, {})
            spy_200 = mkt.get("spy_above_200sma", True if bk_date < "2016-01-01" else False)
            spy_50 = mkt.get("spy_above_50sma", False)
            spy_slope = mkt.get("spy_10ema_slope", 0.0)
            qqq_bull = mkt.get("qqq_bull_stack", False)
            qqq_pb = mkt.get("qqq_pullback", False)
            qqq_ratio_bull = mkt.get("qqq_spy_ratio_bull", False)
            
            # Sectors & Themes
            sec_name = str(setup.get("sector", ""))
            thm_name = str(setup.get("theme", ""))
            thms_str = str(setup.get("themes", ""))
            is_lead_sec = sec_name in lead_secs
            is_lead_thm = (thm_name in lead_thms) or any(t in thms_str for t in lead_thms)
            is_emerg_thm = (thm_name in emerg_thms) or any(t in thms_str for t in emerg_thms)
            
            # RS Metrics
            rs_sc = int(setup["rs_score"]) if pd.notna(setup.get("rs_score")) else None
            st2 = bool(setup.get("stage2", False)) if pd.notna(setup.get("stage2")) else False
            d52 = round(float(setup["dist_52w_high"]), 1) if pd.notna(setup.get("dist_52w_high")) else None
            rs_lead = bool((rs_sc is not None and rs_sc >= 90) and (st2 or (d52 is not None and d52 >= -15.0)))
            
            sized_r_06x = round(t_06x["r_multiple"] * pos_size_mult, 3)
            sized_r_10x = round(t_10x["r_multiple"] * pos_size_mult, 3)
            
            rec = {
                "symbol": sym,
                "date": bk_date,
                "entry_price": round(entry_p, 4),
                "pivot_low_price": round(pivot_low_p, 4),
                "pivot_low_date": pivot_low_dt,
                "full_pivot_dist": round(full_pivot_dist, 4),
                # 0.6x Stoploss Trade
                "stop_loss_06x": t_06x["stop_loss"],
                "risk_pct_06x": t_06x["risk_pct"],
                "exit_date_06x": t_06x["exit_date"],
                "exit_price_06x": t_06x["exit_price"],
                "exit_reason_06x": t_06x["exit_reason"],
                "bars_06x": t_06x["bars_in_trade"],
                "r_06x": t_06x["r_multiple"],
                "sized_r_06x": sized_r_06x,
                "gain_pct_06x": t_06x["gain_pct"],
                "mfe_r_06x": t_06x["max_mfe_r"],
                "mae_r_06x": t_06x["max_mae_r"],
                "is_win_06x": t_06x["is_win"],
                # 1.0x Full Pivot Stoploss Trade
                "stop_loss_10x": t_10x["stop_loss"],
                "risk_pct_10x": t_10x["risk_pct"],
                "exit_date_10x": t_10x["exit_date"],
                "exit_price_10x": t_10x["exit_price"],
                "exit_reason_10x": t_10x["exit_reason"],
                "bars_10x": t_10x["bars_in_trade"],
                "r_10x": t_10x["r_multiple"],
                "sized_r_10x": sized_r_10x,
                "gain_pct_10x": t_10x["gain_pct"],
                "mfe_r_10x": t_10x["max_mfe_r"],
                "mae_r_10x": t_10x["max_mae_r"],
                "is_win_10x": t_10x["is_win"],
                # Sizing Multiplier
                "pos_size_mult": pos_size_mult,
                "s_ep": size_details["s_ep"],
                "s_pole": size_details["s_pole"],
                "s_rvol": size_details["s_rvol"],
                "s_pb": size_details["s_pb"],
                # Market Context
                "spy_above_200sma": spy_200,
                "spy_above_50sma": spy_50,
                "spy_10ema_slope": spy_slope,
                "qqq_bull_stack": qqq_bull,
                "qqq_pullback": qqq_pb,
                "qqq_spy_ratio_bull": qqq_ratio_bull,
                # Macro
                "sector": sec_name,
                "theme": thm_name,
                "themes": thms_str,
                "is_leading_sector": is_lead_sec,
                "is_leading_theme": is_lead_thm,
                "is_emerging_theme": is_emerg_thm,
                "is_leading_macro": bool(is_lead_sec or is_lead_thm),
                # Setup Characteristics
                "move_pct": move_pct,
                "cons_bars": int(setup.get("cons_bars", 15)),
                "pullback_pct": pullback_pct,
                "pullback_ratio": pullback_ratio,
                "vol_x": round(float(setup.get("vol_x", 1.0)), 2) if pd.notna(setup.get("vol_x")) else 1.0,
                "breakout_rvol": bk_rvol,
                "pattern": str(setup.get("pattern", "other")),
                # RS Posture
                "rs_score": rs_sc,
                "stage2": st2,
                "dist_52w_high": d52,
                "rs_leader": rs_lead,
                # EP Catalyst
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
    df_out.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved institutional QM strategy dataset to {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024 / 1024:.2f} MB).")


if __name__ == "__main__":
    build_dataset()
