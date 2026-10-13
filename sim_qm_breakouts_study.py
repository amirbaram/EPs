"""sim_qm_breakouts_study.py — Full Historical Simulation & Feature Extraction for QM Breakouts.

Simulates long positions entered on the close of QM breakout candles:
1. Initial Stop Loss: Placed strictly below previous confirmed TrendLab pivot low.
2. Trailing Stop: Once 10 EMA is sloping upward with price closing well above 10 EMA (>= 1.0%),
   exits on first close below 10 EMA.
3. Extracted Features:
   - Setup geometry "To the Left" (pole gain, velocity, base length, pullback, ride frac, pattern, volume ratio).
   - Breakout day technicals (gain %, close position, range/ATR, gap, distance to moving averages).
   - Catalysts & EP Links (prior igniter EP, gap %, RVOL, subtype, post booster EP).
   - Institutional Relative Strength (1-99 RS score, Stage 2 bullish alignment, 52W high distance, leader status).
   - Market Health on Entry (SPY 10/20/50/200 MA posture, MA slopes, returns, 52W high distance).
   - Sectors & Themes (macro status, leading/emerging status).
   - Trade Lifecycle "To the Right" (duration, realized R, gain %, MFE R, MAE R, exit reason, win/loss flags).

Saves results to: `data/simulations/qm_breakout_trades_full.parquet`.
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import config
import datastore
import indicators
import labels
import trendlab

SETUPS_PATH = Path("data/simulations/qm_historical_setups.parquet")
OUTPUT_PATH = Path("data/simulations/qm_breakout_trades_full.parquet")


def compute_market_health_lookup() -> Dict[str, Dict[str, Any]]:
    """Precompute SPY market health context for fast O(1) date lookups."""
    spy = datastore.load_bars("SPY")
    if spy is None or spy.empty:
        return {}
    spy = indicators.add_indicators(spy)
    
    close = spy["close"].to_numpy(float)
    high = spy["high"].to_numpy(float)
    ema10 = spy["ema10"].to_numpy(float)
    ema20 = spy["ema20"].to_numpy(float)
    sma50 = spy["sma50"].to_numpy(float)
    sma200 = spy["sma200"].to_numpy(float)
    dates = spy.index.strftime("%Y-%m-%d").tolist()
    
    lookup: Dict[str, Dict[str, Any]] = {}
    n = len(dates)
    
    # 252-day rolling high
    roll_hi_252 = spy["high"].rolling(252, min_periods=30).max().to_numpy(float)
    
    for i in range(1, n):
        d_str = dates[i]
        c = close[i]
        e10 = ema10[i]
        e20 = ema20[i]
        s50 = sma50[i]
        s200 = sma200[i]
        
        # slopes
        slope10 = (e10 - ema10[i - 1]) / ema10[i - 1] * 100.0 if i >= 1 else 0.0
        slope20 = (e20 - ema20[i - 1]) / ema20[i - 1] * 100.0 if i >= 1 else 0.0
        slope50 = (s50 - sma50[i - 1]) / sma50[i - 1] * 100.0 if i >= 1 else 0.0
        
        # returns
        ret5 = (c / close[i - 5] - 1.0) * 100.0 if i >= 5 else 0.0
        ret21 = (c / close[i - 21] - 1.0) * 100.0 if i >= 21 else 0.0
        
        # dist to 52w high
        hi252 = roll_hi_252[i]
        dist_52w = (c / hi252 - 1.0) * 100.0 if hi252 > 0 else 0.0
        
        # Stage 2 alignment for market
        market_s2 = bool(c >= e10 >= e20 >= s50 >= s200)
        
        lookup[d_str] = {
            "spy_close": round(c, 2),
            "spy_above_10ema": bool(c >= e10),
            "spy_above_20ema": bool(c >= e20),
            "spy_above_50sma": bool(c >= s50),
            "spy_above_200sma": bool(c >= s200),
            "spy_10ema_slope": round(slope10, 3),
            "spy_20ema_slope": round(slope20, 3),
            "spy_50sma_slope": round(slope50, 3),
            "spy_trend_aligned": market_s2,
            "spy_return_5d": round(ret5, 2),
            "spy_return_21d": round(ret21, 2),
            "spy_dist_52w_high": round(dist_52w, 2),
        }
    return lookup


def simulate_all_trades(max_symbols: int = 0) -> pd.DataFrame:
    print(f"Loading QM historical setups from {SETUPS_PATH}...")
    df_setups = pd.read_parquet(SETUPS_PATH)
    bks = df_setups[df_setups["state"] == "breakout"].copy()
    print(f"Total QM Breakouts found: {len(bks)} across {bks['symbol'].nunique()} symbols.")
    
    print("Precomputing SPY market health lookups...")
    market_lookup = compute_market_health_lookup()
    print(f"SPY market context available for {len(market_lookup)} trading sessions.")
    
    # Group breakouts by symbol
    grouped = bks.groupby("symbol")
    all_syms = sorted(grouped.groups.keys())
    if max_symbols > 0:
        all_syms = all_syms[:max_symbols]
        
    trades: List[Dict[str, Any]] = []
    trade_id = 0
    t0 = time.time()
    
    labels.load()
    total_syms = len(all_syms)
    processed_syms = 0
    
    print(f"Starting simulation across {total_syms} symbols...")
    
    for sym in all_syms:
        processed_syms += 1
        if processed_syms % 200 == 0 or processed_syms == total_syms:
            elapsed = time.time() - t0
            rate = processed_syms / elapsed if elapsed > 0 else 0
            sys.stdout.write(f"\rProgress: {processed_syms}/{total_syms} symbols ({processed_syms/total_syms*100:.1f}%) | Trades: {len(trades)} | {rate:.1f} sym/s")
            sys.stdout.flush()
            
        sym_bks = grouped.get_group(sym)
        df_bars = datastore.load_bars(sym)
        if df_bars is None or len(df_bars) < 60:
            continue
            
        df_bars = indicators.add_indicators(df_bars)
        
        # Run TrendLab compute ONCE for the symbol
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
        ema20_arr = df_bars["ema20"].to_numpy(float)
        sma50_arr = df_bars["sma50"].to_numpy(float)
        sma200_arr = df_bars["sma200"].to_numpy(float)
        atr_arr = df_bars["atr14"].to_numpy(float)
        vol_ma20_arr = df_bars["volume"].rolling(20, min_periods=5).mean().to_numpy(float)
        n_bars = len(df_bars)
        
        for _, setup in sym_bks.iterrows():
            bk_date = setup["date"]
            if bk_date not in date_to_idx:
                continue
            idx = date_to_idx[bk_date]
            if idx >= n_bars - 1:
                continue  # no subsequent bars to trade
                
            entry_p = float(close_arr[idx])
            
            # Find previous confirmed TrendLab pivot low with pconf <= idx and pidx <= idx
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
                # Fallback to minimum low of the consolidation base or past 15 bars
                c_start_idx = max(0, idx - int(setup.get("cons_bars", 15)))
                pivot_low_p = float(np.min(low_arr[c_start_idx : idx + 1]))
                pivot_low_idx = c_start_idx
                pivot_low_dt = dates_list[pivot_low_idx]
                pivot_low_bars_ago = idx - pivot_low_idx
                
            # Sanity check stop loss below entry price
            if pivot_low_p >= entry_p:
                pivot_low_p = float(low_arr[idx]) * 0.985
                
            stop_loss = round(pivot_low_p * 0.999, 4)
            initial_risk = entry_p - stop_loss
            if initial_risk <= 0.001:
                continue
                
            risk_pct = (initial_risk / entry_p) * 100.0
            
            # Breakout day candle features
            bk_open = open_arr[idx]
            bk_high = high_arr[idx]
            bk_low = low_arr[idx]
            bk_atr = atr_arr[idx] if atr_arr[idx] > 0 else 0.01
            bk_vol = vol_arr[idx]
            bk_vol_ma = vol_ma20_arr[idx] if vol_ma20_arr[idx] > 0 else 1.0
            
            bk_candle_gain = (entry_p / bk_open - 1.0) * 100.0 if bk_open > 0 else 0.0
            bk_close_pos = (entry_p - bk_low) / (bk_high - bk_low) if bk_high > bk_low else 0.5
            bk_range_atr = (bk_high - bk_low) / bk_atr
            bk_rvol = bk_vol / bk_vol_ma
            prev_close = close_arr[idx - 1] if idx >= 1 else bk_open
            bk_gap_pct = (bk_open / prev_close - 1.0) * 100.0 if prev_close > 0 else 0.0
            
            # Distances to moving averages on entry
            dist_10ema = (entry_p / ema10_arr[idx] - 1.0) * 100.0 if ema10_arr[idx] > 0 else 0.0
            dist_20ema = (entry_p / ema20_arr[idx] - 1.0) * 100.0 if ema20_arr[idx] > 0 else 0.0
            dist_50sma = (entry_p / sma50_arr[idx] - 1.0) * 100.0 if sma50_arr[idx] > 0 else 0.0
            dist_200sma = (entry_p / sma200_arr[idx] - 1.0) * 100.0 if sma200_arr[idx] > 0 else 0.0
            ema10_slope_5d = (ema10_arr[idx] / ema10_arr[idx - 5] - 1.0) * 100.0 if idx >= 5 and ema10_arr[idx - 5] > 0 else 0.0
            
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
                
                # Rule 1: Intraday Stop Loss Breach
                if cur_low <= stop_loss:
                    exit_p = min(cur_open, stop_loss)  # accounts for gap-down slippage
                    exit_dt = dates_list[t]
                    exit_idx = t
                    exit_reason = "stop_loss"
                    break
                    
                # Rule 2: 10 EMA Sloping Upward with Price Closing Well Above (>= 1.0%)
                ema_rising = (cur_e10 > prev_e10)
                well_above = (cur_close >= cur_e10 * 1.01)
                
                if ema_rising and well_above and not trail_active:
                    trail_active = True
                    trail_act_bar = t - idx
                    
                # Rule 3: Trailing Exit on First Close Below 10 EMA (once activated)
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
            gain_pct = (exit_p / entry_p - 1.0) * 100.0
            max_mfe_r = (peak_high - entry_p) / initial_risk
            max_mfe_pct = (peak_high / entry_p - 1.0) * 100.0
            max_mae_r = (entry_p - trough_low) / initial_risk
            max_mae_pct = (1.0 - trough_low / entry_p) * 100.0
            bars_held = exit_idx - idx
            capture_ratio = (realized_r / max_mfe_r) if max_mfe_r > 0 else 0.0
            
            # Market health on entry date
            mkt = market_lookup.get(bk_date, {})
            
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
                "initial_risk": round(initial_risk, 4),
                "risk_pct": round(risk_pct, 2),
                "pivot_low_price": round(pivot_low_p, 4),
                "pivot_low_date": pivot_low_dt,
                "pivot_low_bars_ago": pivot_low_bars_ago,
                "exit_date": exit_dt,
                "exit_price": round(exit_p, 4),
                "exit_reason": exit_reason,
                "bars_in_trade": bars_held,
                "trail_activated": trail_active,
                "trail_act_bar": trail_act_bar,
                # Outcomes
                "r_multiple": round(realized_r, 3),
                "gain_pct": round(gain_pct, 2),
                "is_win": bool(realized_r > 0),
                "is_big_win": bool(realized_r >= 3.0),
                "is_scratch": bool(abs(realized_r) <= 0.3),
                "is_full_loss": bool(realized_r <= -0.9),
                "max_mfe_r": round(max_mfe_r, 3),
                "max_mfe_pct": round(max_mfe_pct, 2),
                "max_mae_r": round(max_mae_r, 3),
                "max_mae_pct": round(max_mae_pct, 2),
                "capture_ratio": round(capture_ratio, 3),
                # Setup "To the Left"
                "move_pct": round(float(setup.get("move_pct", 0.0)), 1),
                "cons_bars": int(setup.get("cons_bars", 0)),
                "pullback_pct": round(float(setup.get("pullback_pct", 0.0)), 1),
                "pullback_ratio": round(float(setup.get("pullback_pct", 0.0)) / max(1.0, float(setup.get("move_pct", 1.0))), 3),
                "ride_frac": round(float(setup.get("ride_frac", 0.0)), 2),
                "shakeouts": int(setup.get("shakeouts", 0)),
                "vol_x": round(float(setup.get("vol_x", 1.0)), 2) if pd.notna(setup.get("vol_x")) else 1.0,
                "pattern": str(setup.get("pattern", "other")),
                "breakout_candle_gain": round(bk_candle_gain, 2),
                "breakout_close_pos": round(bk_close_pos, 2),
                "breakout_range_atr": round(bk_range_atr, 2),
                "breakout_rvol": round(bk_rvol, 2),
                "breakout_gap_pct": round(bk_gap_pct, 2),
                "dist_10ema": round(dist_10ema, 2),
                "dist_20ema": round(dist_20ema, 2),
                "dist_50sma": round(dist_50sma, 2),
                "dist_200sma": round(dist_200sma, 2),
                "ema10_slope_5d": round(ema10_slope_5d, 2),
                # EP Catalysts
                "has_prior_ep": bool(setup.get("has_prior_ep", False)),
                "prior_ep_days_ago": setup.get("prior_ep_days_ago"),
                "prior_ep_gap_pct": setup.get("prior_ep_gap_pct"),
                "prior_ep_rvol": setup.get("prior_ep_rvol"),
                "prior_ep_subtype": setup.get("prior_ep_subtype"),
                "ep_badge": setup.get("ep_badge", "Pure Technical"),
                "has_post_ep": bool(setup.get("has_post_ep", False)),
                "post_ep_days_later": setup.get("post_ep_days_later"),
                "post_ep_gap_pct": setup.get("post_ep_gap_pct"),
                "post_ep_rvol": setup.get("post_ep_rvol"),
                # Institutional RS & Posture
                "rs_score": rs_sc,
                "stage2": st2,
                "dist_52w_high": d52,
                "rs_leader": is_leader,
                # Sectors & Themes
                "sector": str(setup.get("sector", "Unknown")),
                "theme": str(setup.get("theme", "General")),
                # Market Health Context (SPY)
                "spy_above_10ema": mkt.get("spy_above_10ema"),
                "spy_above_20ema": mkt.get("spy_above_20ema"),
                "spy_above_50sma": mkt.get("spy_above_50sma"),
                "spy_above_200sma": mkt.get("spy_above_200sma"),
                "spy_10ema_slope": mkt.get("spy_10ema_slope"),
                "spy_20ema_slope": mkt.get("spy_20ema_slope"),
                "spy_50sma_slope": mkt.get("spy_50sma_slope"),
                "spy_trend_aligned": mkt.get("spy_trend_aligned"),
                "spy_return_5d": mkt.get("spy_return_5d"),
                "spy_return_21d": mkt.get("spy_return_21d"),
                "spy_dist_52w_high": mkt.get("spy_dist_52w_high"),
            }
            trades.append(rec)
            
    print()
    print(f"Simulation completed in {time.time() - t0:.1f}s. Simulated {len(trades)} trades.")
    df_trades = pd.DataFrame(trades)
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_trades.to_parquet(OUTPUT_PATH, index=False)
    print(f"✅ Saved full trades dataset to {OUTPUT_PATH} ({len(df_trades):,} records, {OUTPUT_PATH.stat().st_size / 1024 / 1024:.2f} MB).")
    return df_trades


if __name__ == "__main__":
    simulate_all_trades()
