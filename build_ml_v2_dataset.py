import pandas as pd
import numpy as np
import os
import sys
import json
from pathlib import Path

import setups
import datastore
import labels
import universe
import indicators
from build_ml_dataset import ContextBuilder

def build_amir_v2_dataset():
    print("🚀 INITIALIZING ML V2 (ADVANCED INSIGHTS) DATASET (DAY-5 ENTRY)")
    
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
    
    for i, sym in enumerate(inventory):
        if i and i % 500 == 0:
            print(f"  Processed {i}/{len(inventory)}...")
            
        df = cached_bars[sym].copy()
        if 'rvol' not in df.columns:
            df = indicators.add_indicators(df)
            
        df = df.loc[:, ~df.columns.duplicated()]
        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["sma50"] = df["close"].rolling(50).mean()
        df["sma200"] = df["close"].rolling(200).mean()
        
        df["is_pivot_high"] = (
            (df["high"] > df["high"].shift(1)) & 
            (df["high"] > df["high"].shift(2)) & 
            (df["high"] > df["high"].shift(-1)) & 
            (df["high"] > df["high"].shift(-2))
        )
        
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
            
            rvol = df["rvol"].iloc[idx]
            close_pos = df["close_pos"].iloc[idx]
            gap_pct = df["gap_pct"].iloc[idx]
            
            is_inst_sweet_spot = int(rvol >= 5.0 and close_pos >= 0.65)
            
            d2_d5 = df.iloc[idx+1:d5_idx+1]
            up_vol = d2_d5[d2_d5["close"] > d2_d5["open"]]["volume"].sum()
            dn_vol = d2_d5[d2_d5["close"] <= d2_d5["open"]]["volume"].sum()
            digestion_ratio = up_vol / (dn_vol + 1)
            
            lookback_pivots = df.iloc[max(0, d5_idx-250):idx]
            valid_pivots = lookback_pivots[lookback_pivots["is_pivot_high"]]["high"]
            
            overhead = valid_pivots[valid_pivots > d5_close]
            if len(overhead) == 0:
                is_blue_sky = 1
                dist_to_overhead = 1.0 
            else:
                is_blue_sky = 0
                dist_to_overhead = (overhead.min() - d5_close) / d5_close
                
            c_data = c_data_full.iloc[d5_idx]
            
            # --- V2 NEW FEATURES ---
            # 1. DRE (Delayed Reaction EP)
            is_dre = int(gap_pct >= 5.0 and rvol >= 2.5 and close_pos < 0.65)
            
            # 2. 48H Upper Body Absorption
            d1_high = df["high"].iloc[idx]
            d2_high = df["high"].iloc[idx+1]
            d3_high = df["high"].iloc[idx+2]
            held_48h = int(d2_high >= d1_high * 0.98 or d3_high >= d1_high * 0.98)
            
            # 3. Above 50 SMA / Distance to 200 SMA
            sma50_val = df["sma50"].iloc[idx]
            sma200_val = df["sma200"].iloc[idx]
            above_sma50 = int(d1_close > sma50_val) if pd.notna(sma50_val) else 0
            dist_200sma = (d1_close - sma200_val) / sma200_val if pd.notna(sma200_val) and sma200_val > 0 else 0
            
            # 4. Neglected Depth
            pre_6m = df.iloc[max(0, idx-126):idx]
            neglect_6m = (pre_6m["close"].iloc[-1] - pre_6m["close"].iloc[0]) / pre_6m["close"].iloc[0] * 100 if len(pre_6m) > 100 else 0
            
            # 5. Thematic/Sector Momentum and Change
            sec_m1 = c_data.get("ind_ret_1m", np.nan)
            sec_m3 = c_data.get("ind_ret_3m", np.nan)
            thm_m1 = c_data.get("theme_ret_1m", np.nan)
            thm_m3 = c_data.get("theme_ret_3m", np.nan)
            
            sec_change = sec_m1 - sec_m3 if pd.notna(sec_m1) and pd.notna(sec_m3) else 0
            thm_change = thm_m1 - thm_m3 if pd.notna(thm_m1) and pd.notna(thm_m3) else 0
            
            # 6. Archetype (One-hot mapped or ordinal)
            # Pinnacle = 4, Turnaround = 3, Beta = 2, General = 1
            if neglect_6m <= -15.0 and close_pos >= 0.70 and rvol >= 4.0:
                archetype_score = 3 # Turnaround
            elif close_pos >= 0.85 and rvol >= 4.0 and gap_pct >= 15.0:
                archetype_score = 4 # Pinnacle
            elif gap_pct >= 8.0 and is_inst_sweet_spot:
                archetype_score = 2 # Beta
            else:
                archetype_score = 1 # General
            # -----------------------
            
            f_end = min(len(df), d5_idx + 250)
            f_window = df.iloc[d5_idx+1:f_end]
            
            entry = d5_close
            stop = d1_low
            risk = entry - stop
            
            if risk <= 0:
                continue
                
            targets = {50: entry * 1.5, 100: entry * 2.0, 150: entry * 2.5, 200: entry * 3.0}
            outcomes = {}
            for t_pct, t_price in targets.items():
                hit_target = False
                for j in range(len(f_window)):
                    bar = f_window.iloc[j]
                    if bar["low"] <= stop:
                        break
                    if bar["high"] >= t_price:
                        hit_target = True
                        break
                outcomes[t_pct] = 1 if hit_target else 0
                
            record = {
                "symbol": sym,
                "entry_date": str(df.index[d5_idx].date()),
                
                # V1 FEATURES
                "feature_gap_pct": gap_pct,
                "feature_rvol": rvol,
                "feature_close_pos": close_pos,
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
                
                # V2 ADVANCED FEATURES
                "feature_is_dre": is_dre,
                "feature_held_48h": held_48h,
                "feature_above_50sma": above_sma50,
                "feature_dist_200sma": dist_200sma,
                "feature_neglect_6m": neglect_6m,
                "feature_sec_change": sec_change,
                "feature_thm_change": thm_change,
                "feature_archetype_score": archetype_score,
                
                "target_50": outcomes[50],
                "target_100": outcomes[100],
                "target_150": outcomes[150],
                "target_200": outcomes[200],
            }
            all_records.append(record)

    out_df = pd.DataFrame(all_records)
    print(f"[*] Generated {len(out_df)} Day-5 Entry events with V2 Features.")
    
    output_dir = Path("data/ml_datasets/amir_spec")
    out_file = output_dir / "dataset_v2.parquet"
    out_df.to_parquet(out_file, index=False)
    print(f"[*] Saved custom dataset to {out_file}")

if __name__ == "__main__":
    build_amir_v2_dataset()
