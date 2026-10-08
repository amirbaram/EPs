import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

import setups
import datastore
import labels
import universe
import indicators
from build_ml_dataset import ContextBuilder

def build_rolling_dataset():
    print("🚀 INITIALIZING ROLLING MULTI-DAY DATASET (DAY 1 TO DAY 20)")
    
    active_symbols = universe.active_symbols()
    print(f"[*] Loading symbols...")
    
    cached_bars = {}
    inventory = []
    
    spy = datastore.load_bars("SPY")
    if spy is not None and not spy.empty:
        cached_bars["SPY"] = spy
    
    for sym in active_symbols:
        df = datastore.load_bars(sym)
        if df is not None and len(df) > 50:
            cached_bars[sym] = df
            inventory.append(sym)
            
    print(f"[*] Loaded {len(inventory)} equity symbols.")
    
    closes = {sym: cached_bars[sym]["close"] for sym in inventory}
    close_panel = pd.DataFrame(closes)
    ctxb = ContextBuilder(close_panel, cached_bars.get("SPY", None))
    
    all_records = []
    
    for i, sym in enumerate(inventory):
        if i and i % 500 == 0:
            print(f"  Processed {i}/{len(inventory)}...")
            
        df = cached_bars[sym].copy()
        if 'rvol' not in df.columns:
            df = indicators.add_indicators(df)
            
        df = df.loc[:, ~df.columns.duplicated()]
        df["ema10"] = df["close"].ewm(span=10, adjust=False).mean()
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
        
        import trendlab
        r = trendlab._compute(df)
        df["is_pivot_high"] = False
        df["pivot_conf_idx"] = -1
        df["pivot_price"] = np.nan
        hp = [(r["pidx"][k], r["pconf"][k], r["pprice"][k]) for k in range(len(r["pidx"])) if r["ptype"][k] == trendlab.PIVOT_HIGH]
        for pidx, pconf, pprice in hp:
            if pidx < len(df):
                df.loc[df.index[pidx], "is_pivot_high"] = True
                df.loc[df.index[pidx], "pivot_conf_idx"] = pconf
                df.loc[df.index[pidx], "pivot_price"] = pprice
        
        c_data_full = ctxb.for_symbol(sym, df.index)
        
        for ep_tuple in setups.iter_fresh_ep_events(df):
            idx = ep_tuple[0]
            
            if idx < 250 or idx > len(df) - 6:
                continue
                
            d1_low = df["low"].iloc[idx]
            d1_close = df["close"].iloc[idx]
            d1_vol = df["volume"].iloc[idx]
            gap_pct = df["gap_pct"].iloc[idx]
            rvol = df["rvol"].iloc[idx]
            close_pos = df["close_pos"].iloc[idx]
            
            pre_189_vol = df["volume"].iloc[max(0, idx-189):idx]
            is_novel_vol = int(d1_vol > pre_189_vol.max()) if len(pre_189_vol) > 0 else 0
            is_inst_sweet_spot = int(rvol >= 5.0 and close_pos >= 0.65)
            
            pre_20 = df.iloc[max(0, idx-20):idx]
            tightness_1m = (pre_20["high"].max() - pre_20["low"].min()) / pre_20["low"].min() if len(pre_20) > 0 and pre_20["low"].min() > 0 else 0
            
            # -------------------------------------------------------------------------
            # [FIX]: Pre-calculate static target outcomes ONCE for the entire event
            # -------------------------------------------------------------------------
            entry = d1_close
            stop = d1_low
            risk = entry - stop
            
            if risk <= 0:
                continue
                
            f_end = min(len(df), idx + 250)
            full_f_window = df.iloc[idx+1:f_end]
            targets = {50: entry * 1.5, 100: entry * 2.0, 150: entry * 2.5, 200: entry * 3.0}
            
            event_mae_pct = float('nan')

            # [FIX]: Vectorize the target evaluation
            if len(full_f_window) > 0:
                stop_mask = full_f_window["low"] <= stop
                first_stop_idx = stop_mask.idxmax() if stop_mask.any() else full_f_window.index[-1] + 1
                # Check for same-day stop hit (idxmax returns the first True, or the first index if all False, so we verify)
                if not stop_mask.any():
                    first_stop_idx = full_f_window.index[-1] + 1
                
                target_50_mask = full_f_window["high"] >= targets[50]
                first_target_50_idx = target_50_mask.idxmax() if target_50_mask.any() else full_f_window.index[-1] + 1
                if not target_50_mask.any():
                    first_target_50_idx = full_f_window.index[-1] + 1

                if first_target_50_idx < first_stop_idx and first_target_50_idx <= full_f_window.index[-1]:
                    mae_price = min(entry, full_f_window.loc[:first_target_50_idx]["low"].min())
                    event_mae_pct = (mae_price - entry) / entry

            is_timeout = (f_end == idx + 250)
            event_outcomes = {}
            for t_pct, t_price in targets.items():
                if len(full_f_window) > 0:
                    target_mask = full_f_window["high"] >= t_price
                    first_target_idx = target_mask.idxmax() if target_mask.any() else full_f_window.index[-1] + 1
                    if not target_mask.any():
                        first_target_idx = full_f_window.index[-1] + 1

                    if first_target_idx < first_stop_idx and first_target_idx <= full_f_window.index[-1]:
                        event_outcomes[t_pct] = 1
                    elif first_stop_idx <= full_f_window.index[-1]:
                        event_outcomes[t_pct] = 0
                    elif is_timeout:
                        event_outcomes[t_pct] = 0
                    else:
                        event_outcomes[t_pct] = float('nan')
                else:
                    event_outcomes[t_pct] = float('nan')

            # -------------------------------------------------------------------------
            # Now track the setup for up to 20 days post-EP using the pre-calculated outcome
            # -------------------------------------------------------------------------
            for days_since_ep in range(1, 21):
                t_idx = idx + days_since_ep
                if t_idx >= len(df):
                    break
                    
                t_close = df["close"].iloc[t_idx]
                
                # If it breached the D1 low before or on this day, setup is dead. Stop tracking.
                if df["low"].iloc[idx:t_idx+1].min() < d1_low:
                    break
                    
                # Dynamic Features up to day t
                d1_to_t = df.iloc[idx:t_idx+1]
                if len(d1_to_t) > 1:
                    tightness_since_ep = (d1_to_t["high"].max() - d1_to_t["low"].min()) / d1_to_t["low"].min()
                else:
                    tightness_since_ep = 0.0
                    
                d2_t = df.iloc[idx+1:t_idx+1]
                if len(d2_t) > 0:
                    up_vol = d2_t[d2_t["close"] > d2_t["open"]]["volume"].sum()
                    dn_vol = d2_t[d2_t["close"] <= d2_t["open"]]["volume"].sum()
                    digestion_ratio = up_vol / (dn_vol + 1)
                else:
                    digestion_ratio = 1.0
                
                ret_since_ep = (t_close - d1_close) / d1_close if d1_close > 0 else 0
                dist_to_d1_low = (t_close - d1_low) / d1_low if d1_low > 0 else 0
                
                # [FIX 4]: Add +1 to match ep_ml_engine.py exactly
                lookback_pivots = df.iloc[max(0, t_idx-250):t_idx+1]
                valid_pivots = lookback_pivots[(lookback_pivots["is_pivot_high"]) & (lookback_pivots["pivot_conf_idx"] <= t_idx)]["pivot_price"]
                overhead = valid_pivots[valid_pivots > t_close]
                
                if len(overhead) == 0:
                    is_blue_sky = 1
                    dist_to_overhead = 1.0 
                else:
                    is_blue_sky = 0
                    dist_to_overhead = (overhead.min() - t_close) / t_close
                    
                c_data = c_data_full.iloc[t_idx]
                    
                record = {
                    "symbol": sym,
                    "date": str(df.index[t_idx].date()),
                    "entry_date": str(df.index[idx].date()),
                    "feature_days_since_ep": days_since_ep,
                    "feature_tightness_1m": tightness_1m,
                    "feature_gap_pct": gap_pct,
                    "feature_rvol": rvol,
                    "feature_close_pos": close_pos,
                    "feature_is_novel_vol_9m": is_novel_vol,
                    "feature_is_inst_sweet_spot": is_inst_sweet_spot,
                    
                    "feature_tightness_since_ep": tightness_since_ep,
                    "feature_digestion_ratio": digestion_ratio,
                    "feature_ret_since_ep": ret_since_ep,
                    "feature_dist_to_d1_low": dist_to_d1_low,
                    
                    "feature_is_blue_sky": is_blue_sky,
                    "feature_dist_to_overhead_pct": dist_to_overhead,
                    "feature_ind_rank_3m": c_data["ind_rank_3m"],
                    "feature_tk_rs_spy_3m": c_data["tk_rs_spy_3m"],
                    "feature_strongest_theme": c_data.get("strongest_theme", "General"),
                    "feature_is_qullamagi_linear": int(df["ema10"].iloc[t_idx] > df["ema20"].iloc[t_idx] and df["ema20"].iloc[t_idx] > df["ema50"].iloc[t_idx]),
                    
                    "audit_mae_pct": event_mae_pct,
                    
                    "target_50": event_outcomes[50],
                    "target_100": event_outcomes[100],
                    "target_150": event_outcomes[150],
                    "target_200": event_outcomes[200],
                }
                all_records.append(record)

    out_df = pd.DataFrame(all_records)
    print(f"[*] Generated {len(out_df)} Rolling Panel events.")
    
    output_dir = Path("data/ml_datasets/amir_rolling")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "dataset_rolling.parquet"
    out_df.to_parquet(out_file, index=False)
    print(f"[*] Saved rolling dataset to {out_file}")

if __name__ == "__main__":
    build_rolling_dataset()
