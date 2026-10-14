"""Simulates EMA 10, EMA 20, and EMA 50 trailing exits exactly aligned with qm_strategy_trades.parquet."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import datastore

TRADES_PATH = Path("data/simulations/qm_strategy_trades.parquet")

def simulate_ema_exit(
    idx: int,
    entry_p: float,
    stop_p: float,
    low_arr: np.ndarray,
    high_arr: np.ndarray,
    open_arr: np.ndarray,
    close_arr: np.ndarray,
    ema_arr: np.ndarray,
    dates_list: list,
    stop_reason: str,
    max_forward_bars: int = 250,
):
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
                exit_reason = "ema_close_violation"
                break
                
    if exit_p is None:
        exit_p = float(close_arr[max_t])
        exit_dt = dates_list[max_t]
        exit_idx = max_t
        exit_reason = "max_holding_period"
        
    gain_pct = (exit_p / entry_p - 1.0) * 100.0
    r_multiple = (exit_p - entry_p) / initial_risk if initial_risk > 0 else 0.0
    bars_held = exit_idx - idx
    
    return {
        "exit_price": round(exit_p, 4),
        "bars": bars_held,
        "r": round(r_multiple, 3),
        "gain_pct": round(gain_pct, 2),
        "is_win": bool(r_multiple > 0.05),
    }

def main():
    print(f"Loading trades from {TRADES_PATH}...")
    trades = pd.read_parquet(TRADES_PATH)
    print(f"Loaded {len(trades)} trades across {trades['symbol'].nunique()} symbols.")
    
    grouped = trades.groupby("symbol")
    symbols = list(grouped.groups.keys())
    n_syms = len(symbols)
    
    results = []
    t0 = time.time()
    
    for i, sym in enumerate(symbols):
        if (i + 1) % 500 == 0 or (i + 1) == n_syms:
            print(f"Progress: {i+1}/{n_syms} symbols ({(i+1)/n_syms*100:.1f}%) | {len(results)} records | {time.time()-t0:.1f}s")
            
        bars = datastore.load_bars(sym)
        if bars is None or len(bars) < 50:
            continue
            
        close_s = bars["close"]
        low_s = bars["low"]
        high_s = bars["high"]
        open_s = bars["open"]
        
        e10 = close_s.ewm(span=10, adjust=False).mean().to_numpy(float)
        e20 = close_s.ewm(span=20, adjust=False).mean().to_numpy(float)
        e50 = close_s.ewm(span=50, adjust=False).mean().to_numpy(float)
        
        low_arr = low_s.to_numpy(float)
        high_arr = high_s.to_numpy(float)
        open_arr = open_s.to_numpy(float)
        close_arr = close_s.to_numpy(float)
        
        dates_list = bars.index.strftime("%Y-%m-%d").tolist()
        date_to_idx = {d: idx for idx, d in enumerate(dates_list)}
        
        sym_trades = grouped.get_group(sym)
        for _, tr in sym_trades.iterrows():
            d = str(tr["date"])
            if d not in date_to_idx:
                continue
            idx = date_to_idx[d]
            
            entry_p = float(tr["entry_price"])
            stop_06x = float(tr["stop_loss_06x"])
            stop_10x = float(tr["stop_loss_10x"])
            mult = float(tr["pos_size_mult"])
            
            # 0.6x Stop simulations
            t10_06 = simulate_ema_exit(idx, entry_p, stop_06x, low_arr, high_arr, open_arr, close_arr, e10, dates_list, "stop_06x")
            t20_06 = simulate_ema_exit(idx, entry_p, stop_06x, low_arr, high_arr, open_arr, close_arr, e20, dates_list, "stop_06x")
            t50_06 = simulate_ema_exit(idx, entry_p, stop_06x, low_arr, high_arr, open_arr, close_arr, e50, dates_list, "stop_06x")
            
            # 1.0x Stop simulations
            t10_10 = simulate_ema_exit(idx, entry_p, stop_10x, low_arr, high_arr, open_arr, close_arr, e10, dates_list, "stop_10x")
            t20_10 = simulate_ema_exit(idx, entry_p, stop_10x, low_arr, high_arr, open_arr, close_arr, e20, dates_list, "stop_10x")
            t50_10 = simulate_ema_exit(idx, entry_p, stop_10x, low_arr, high_arr, open_arr, close_arr, e50, dates_list, "stop_10x")
            
            row = tr.to_dict()
            row["r_e10_06x"] = t10_06["r"]
            row["sized_r_e10_06x"] = round(t10_06["r"] * mult, 3)
            row["win_e10_06x"] = t10_06["is_win"]
            row["bars_e10_06x"] = t10_06["bars"]
            
            row["r_e20_06x"] = t20_06["r"]
            row["sized_r_e20_06x"] = round(t20_06["r"] * mult, 3)
            row["win_e20_06x"] = t20_06["is_win"]
            row["bars_e20_06x"] = t20_06["bars"]
            
            row["r_e50_06x"] = t50_06["r"]
            row["sized_r_e50_06x"] = round(t50_06["r"] * mult, 3)
            row["win_e50_06x"] = t50_06["is_win"]
            row["bars_e50_06x"] = t50_06["bars"]
            
            # 1.0x
            row["r_e10_10x"] = t10_10["r"]
            row["sized_r_e10_10x"] = round(t10_10["r"] * mult, 3)
            row["win_e10_10x"] = t10_10["is_win"]
            row["bars_e10_10x"] = t10_10["bars"]
            
            row["r_e20_10x"] = t20_10["r"]
            row["sized_r_e20_10x"] = round(t20_10["r"] * mult, 3)
            row["win_e20_10x"] = t20_10["is_win"]
            row["bars_e20_10x"] = t20_10["bars"]
            
            row["r_e50_10x"] = t50_10["r"]
            row["sized_r_e50_10x"] = round(t50_10["r"] * mult, 3)
            row["win_e50_10x"] = t50_10["is_win"]
            row["bars_e50_10x"] = t50_10["bars"]
            
            results.append(row)
            
    df_out = pd.DataFrame(results)
    out_path = Path("exact_ema_study_results.parquet")
    df_out.to_parquet(out_path, index=False)
    print(f"Saved {len(df_out)} aligned records to {out_path} in {time.time()-t0:.1f}s.")

if __name__ == "__main__":
    main()
