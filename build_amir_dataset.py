import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import tempfile
import shutil
from pathlib import Path

import setups
import datastore
import labels
import universe
import indicators
from build_ml_dataset import ContextBuilder, get_file_hash, DATA_DIR

def build_amir_dataset():
    print("🚀 INITIALIZING AMIR ORIGINAL SPEC MODEL DATASET (DAY-5 ENTRY)")
    
    active_symbols = universe.active_symbols()
    print(f"[*] Loading {len(active_symbols)} symbols...")
    
    cached_bars = {}
    inventory = []
    
    spy = datastore.load_bars("SPY")
    if spy is not None and not spy.empty:
        cached_bars["SPY"] = spy
    
    for i, sym in enumerate(active_symbols):
        df = datastore.load_bars(sym)
        if df is not None and len(df) > 50:
            cached_bars[sym] = df
            inventory.append(sym)
            
    print(f"[*] Successfully loaded {len(inventory)} equity symbols.")
    
    print("[*] Building Point-in-Time Context...")
    closes = {sym: cached_bars[sym]["close"] for sym in inventory}
    close_panel = pd.DataFrame(closes)
    ctxb = ContextBuilder(close_panel, cached_bars.get("SPY", None))
    
    all_records = []
    
    print("[*] Extracting Qullamägi, Bonde, and Pivot features...")
    for i, sym in enumerate(inventory):
        if i and i % 500 == 0:
            print(f"  Processed {i}/{len(inventory)}...")
            
        df = cached_bars[sym].copy()
        if 'rvol' not in df.columns:
            df = indicators.add_indicators(df)
            
        df = df.loc[:, ~df.columns.duplicated()]
        
        # [FIX 1]: Calculate missing EMAs to prevent KeyError
        df["ema10"] = df["close"].ewm(span=10, adjust=False).mean()
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
        
        # [FIX 2]: Use trendlab instead of global shifts to prevent look-ahead bias
        import trendlab
        import numpy as np
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
        
        s_sector = labels.sector(sym)
        s_themes = labels.themes(sym)
        
        c_data_full = ctxb.for_symbol(sym, df.index)
        for ep_tuple in setups.iter_fresh_ep_events(df):
            idx = ep_tuple[0]
            subtype = ep_tuple[1]
            
            if idx < 250 or idx > len(df) - 6:
                continue
                
            d1_low = df["low"].iloc[idx]
            d1_close = df["close"].iloc[idx]
            d1_vol = df["volume"].iloc[idx]
            d5_idx = idx + 4
            
            survival_window = df["low"].iloc[idx:d5_idx+1]
            if survival_window.min() < d1_low:
                continue 
                
            d5_close = df["close"].iloc[d5_idx]
            
            pre_20 = df.iloc[idx-20:idx]
            tightness = (pre_20["high"].max() - pre_20["low"].min()) / pre_20["low"].min() if pre_20["low"].min() > 0 else 0
            
            d5_ema10 = df["ema10"].iloc[d5_idx]
            d5_ema20 = df["ema20"].iloc[d5_idx]
            d5_ema50 = df["ema50"].iloc[d5_idx]
            is_linear = int((d5_ema10 > d5_ema20) and (d5_ema20 > d5_ema50))
            
            pre_189_vol = df["volume"].iloc[max(0, idx-189):idx]
            is_novel_vol = int(d1_vol > pre_189_vol.max()) if len(pre_189_vol) > 0 else 0
            
            is_inst_sweet_spot = int(df["rvol"].iloc[idx] >= 5.0 and df["close_pos"].iloc[idx] >= 0.65)
            
            d2_d5 = df.iloc[idx+1:d5_idx+1]
            up_vol = d2_d5[d2_d5["close"] > d2_d5["open"]]["volume"].sum()
            dn_vol = d2_d5[d2_d5["close"] <= d2_d5["open"]]["volume"].sum()
            digestion_ratio = up_vol / (dn_vol + 1)
            
            # [FIX 3]: Lookback resistance should go up to d5_idx+1 and strictly use pivot_conf_idx
            lookback_pivots = df.iloc[max(0, d5_idx-250):d5_idx+1]
            valid_pivots = lookback_pivots[(lookback_pivots["is_pivot_high"]) & (lookback_pivots["pivot_conf_idx"] <= d5_idx)]["pivot_price"]
            
            overhead = valid_pivots[valid_pivots > d5_close]
            if len(overhead) == 0:
                is_blue_sky = 1
                dist_to_overhead = 1.0 
            else:
                is_blue_sky = 0
                closest_pivot = overhead.min()
                dist_to_overhead = (closest_pivot - d5_close) / d5_close
                
            c_data = c_data_full.iloc[d5_idx]
            
            # [FIX]: Evaluate the event's static outcome globally from Day 1 to Horizon
            f_end = min(len(df), idx + 250)
            full_f_window = df.iloc[idx+1:f_end]
            
            entry = d1_close
            stop = d1_low
            risk = entry - stop
            
            if risk <= 0:
                continue
                
            targets = {
                50: entry * 1.5,
                100: entry * 2.0,
                150: entry * 2.5,
                200: entry * 3.0
            }
            
            audit_mae_pct = float('nan')
            mae_price = entry
            hit_target_50 = False
            for j in range(len(full_f_window)):
                bar = full_f_window.iloc[j]
                if bar["low"] < mae_price:
                    mae_price = bar["low"]
                if bar["low"] <= stop:
                    break
                if bar["high"] >= targets[50]:
                    hit_target_50 = True
                    break
            if hit_target_50:
                audit_mae_pct = (mae_price - entry) / entry

            is_timeout = (f_end == idx + 250)
            outcomes = {}
            for t_pct, t_price in targets.items():
                hit_target = False
                hit_stop = False
                for j in range(len(full_f_window)):
                    bar = full_f_window.iloc[j]
                    if bar["low"] <= stop:
                        hit_stop = True
                        break
                    if bar["high"] >= t_price:
                        hit_target = True
                        break
                if hit_target:
                    outcomes[t_pct] = 1
                elif hit_stop:
                    outcomes[t_pct] = 0
                elif is_timeout:
                    outcomes[t_pct] = 0  # Timeouts are failures, not censored data
                else:
                    outcomes[t_pct] = float('nan')
                    
            record = {
                "symbol": sym,
                "entry_date": str(df.index[d5_idx].date()),
                "feature_gap_pct": df["gap_pct"].iloc[idx],
                "feature_rvol": df["rvol"].iloc[idx],
                "feature_close_pos": df["close_pos"].iloc[idx],
                
                "feature_tightness_1m": tightness,
                "feature_is_qullamagi_linear": is_linear,
                "feature_is_novel_vol_9m": is_novel_vol,
                "feature_is_inst_sweet_spot": is_inst_sweet_spot,
                "feature_digestion_ratio": digestion_ratio,
                "feature_day5_ret_vs_day1": (d5_close - d1_close) / d1_close,
                
                "feature_is_blue_sky": is_blue_sky,
                "feature_dist_to_overhead_pct": dist_to_overhead,
                
                "feature_ind_rank_3m": c_data["ind_rank_3m"],
                "feature_tk_rs_spy_3m": c_data["tk_rs_spy_3m"],
                "feature_strongest_theme": c_data["strongest_theme"],
                
                "audit_mae_pct": audit_mae_pct,  # [FIX]: Appended to dictionary
                
                "target_50": outcomes[50],
                "target_100": outcomes[100],
                "target_150": outcomes[150],
                "target_200": outcomes[200],
            }
            all_records.append(record)

    out_df = pd.DataFrame(all_records)
    print(f"[*] Generated {len(out_df)} Day-5 Entry events.")
    
    output_dir = Path("data/ml_datasets/amir_spec")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "dataset.parquet"
    out_df.to_parquet(out_file, index=False)
    print(f"[*] Saved custom dataset to {out_file}")

if __name__ == "__main__":
    build_amir_dataset()
