#!/usr/bin/env python3
"""
Batch Portfolio Simulation for Episodic Pivot (EP) Strategy
Calculates real ground truth performance across Trade 1 (Baseline, Pyramiding, Dynamic Stop),
Trade 2 (First Continuation Leg), and Trade 3 (Second Continuation Runner Leg)
using the newly audited intermediate ribbon (8, 12, 16, 21) under strict zero-lookahead,
realistic gap-down stop loss execution.
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import datastore
from ep_ml_engine import engine

REENTRY_FEATS = [
    "feature_days_since_ep", "feature_drawdown_from_peak", "feature_dist_10ema",
    "feature_dist_20ema", "feature_dist_50sma", "feature_vol_contraction",
    "feature_is_yellow_flip", "feature_ribbon_compression",
    "feature_ribbon_bandwidth", "feature_dist_ribbon_ema8",
    "feature_dist_ribbon_ema21"
]

ALL_EMA_SPANS = [8, 10, 12, 16, 20, 21]

def precalculate_symbol_data(symbols):
    print(f"📦 Precalculating EMAs and indicators for {len(symbols)} unique symbols...")
    t0 = time.time()
    cache = {}
    for sym in symbols:
        df = datastore.load_bars(sym)
        if df is None or len(df) < 50:
            continue
        c = df["close"]
        emas = {span: c.ewm(span=span, adjust=False).mean().values for span in ALL_EMA_SPANS}
        emas["sma50"] = c.rolling(50).mean().values
        emas["vol20"] = df["volume"].rolling(20).mean().values
        emas["open"] = df["open"].values
        emas["high"] = df["high"].values
        emas["low"] = df["low"].values
        emas["close"] = c.values
        emas["volume"] = df["volume"].values
        emas["dates"] = [str(d.date()) if hasattr(d, "date") else str(d)[:10] for d in df.index]
        emas["date_to_idx"] = {d: i for i, d in enumerate(emas["dates"])}
        emas["n_bars"] = len(df)
        
        # Precompute intermediate ribbon (8, 12, 16, 21) states
        e8 = emas[8]
        e12 = emas[12]
        e16 = emas[16]
        e21 = emas[21]
        bullish = (e8 >= e12) & (e12 >= e16) & (e16 >= e21)
        bearish = (e8 < e12) & (e12 < e16) & (e16 < e21)
        states = np.zeros(len(df), dtype=np.int8)
        states[bearish] = -1
        states[bullish] = 1
        emas["states"] = states
        
        cache[sym] = emas
    t1 = time.time()
    print(f"✅ Precalculated {len(cache)} symbols in {t1 - t0:.2f}s")
    return cache


def run_batch_simulation():
    print("=" * 80)
    print("🚀 BATCH PORTFOLIO SIMULATION: INTERMEDIATE RIBBON (8, 12, 16, 21)")
    print("=" * 80)
    
    parquet_path = Path("data/simulations/ep_combined_study_scored.parquet")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} historical EP events.")
    
    symbols = df["symbol"].unique()
    cache = precalculate_symbol_data(symbols)
    
    clf = engine.reentry_classifier
    
    results = []
    t2_feat_rows = []
    t2_feat_indices = []
    t3_feat_rows = []
    t3_feat_indices = []
    
    t_start = time.time()
    
    for i, (idx, row) in enumerate(df.iterrows()):
        sym = row["symbol"]
        dt = str(row["date"])[:10]
        
        if sym not in cache or dt not in cache[sym]["date_to_idx"]:
            results.append(None)
            continue
            
        sym_data = cache[sym]
        d1_idx = sym_data["date_to_idx"][dt]
        n_bars = sym_data["n_bars"]
        
        if d1_idx + 1 >= n_bars:
            results.append(None)
            continue
            
        c_arr = sym_data["close"]
        o_arr = sym_data["open"]
        h_arr = sym_data["high"]
        l_arr = sym_data["low"]
        v_arr = sym_data["volume"]
        vol20_arr = sym_data["vol20"]
        sma50_arr = sym_data["sma50"]
        ema10_arr = sym_data[10]
        ema20_arr = sym_data[20]
        states = sym_data["states"]
        
        d1_close = c_arr[d1_idx]
        d1_low = l_arr[d1_idx]
        d1_high = h_arr[d1_idx]
        risk = d1_close - d1_low
        
        if risk <= 0.01:
            results.append(None)
            continue
            
        risk_pct = (risk / d1_close) * 100.0
        max_fwd_idx = min(n_bars, d1_idx + 250)
        
        # --- 1. Trade 1: Mechanical Baseline & Dynamic Stop ---
        t1_stopped = False
        t1_exit_bar = None
        t1_exit_price = None
        t1_reason = None
        t1_peak = d1_close
        seen_bull = False
        
        curr_dyn_stop = d1_low
        dyn_hurdle_cleared = False
        dyn_exit_bar = None
        dyn_exit_price = None
        
        dyn_stop_pct = row.get("dynamic_stop_loss_pct", -0.15)
        ml_50 = row.get("ml_50", 0.0)
        
        for b in range(d1_idx + 1, max_fwd_idx):
            c = c_arr[b]
            l = l_arr[b]
            h = h_arr[b]
            o = o_arr[b]
            s = states[b]
            sma50 = sma50_arr[b]
            
            if h > t1_peak: t1_peak = h
            if s == 1: seen_bull = True
            
            # Baseline Stop check with gap-down fill
            if t1_exit_bar is None and l <= d1_low:
                t1_stopped = True
                t1_exit_bar = b
                t1_exit_price = o if o < d1_low else d1_low
                t1_reason = "Gap Down Stop" if o < d1_low else "D1 Low Stop"
                
            # Baseline 50 SMA exit
            if t1_exit_bar is None and seen_bull and b >= d1_idx + 5 and not np.isnan(sma50) and c < sma50:
                t1_exit_bar = b
                t1_exit_price = c
                t1_reason = "Close < 50 SMA"
                
            # Dynamic stop tracking
            if dyn_exit_bar is None:
                if l <= curr_dyn_stop:
                    dyn_exit_bar = b
                    dyn_exit_price = o if o < curr_dyn_stop else curr_dyn_stop
                    
                unrealized_r = (c - d1_close) / risk
                if unrealized_r >= 2.0:
                    dyn_hurdle_cleared = True
                    
                if dyn_hurdle_cleared:
                    cand_stop = max(d1_close, c * (1.0 + dyn_stop_pct)) if pd.notna(dyn_stop_pct) else d1_close
                    if cand_stop > curr_dyn_stop:
                        curr_dyn_stop = cand_stop
                        
                if seen_bull and b >= d1_idx + 5 and not np.isnan(sma50) and c < sma50:
                    dyn_exit_bar = b
                    dyn_exit_price = c
                    
            if t1_exit_bar is not None and dyn_exit_bar is not None:
                break
                
        if t1_exit_bar is None:
            t1_exit_bar = max_fwd_idx - 1
            t1_exit_price = c_arr[t1_exit_bar]
            t1_reason = "Window End (250D)"
            
        if dyn_exit_bar is None:
            dyn_exit_bar = max_fwd_idx - 1
            dyn_exit_price = c_arr[dyn_exit_bar]
            
        t1_ret = (t1_exit_price / d1_close - 1.0) * 100.0
        t1_r = (t1_exit_price - d1_close) / risk
        
        # Progressive Pyramiding
        t1_pyramid_r = t1_r
        if pd.notna(ml_50) and ml_50 >= 0.50 and d1_idx + 6 < n_bars:
            c5 = c_arr[d1_idx + 5]
            if c5 > d1_close and t1_exit_bar > d1_idx + 5:
                add_r = (t1_exit_price - c5) / risk * 0.5
                t1_pyramid_r = t1_r + add_r
                
        dyn_r = (dyn_exit_price - d1_close) / risk
        
        # --- 2. Trade 2: First Continuation Re-entry with Ribbon (8, 12, 16, 21) ---
        t2_search_start = t1_exit_bar
        t1_peak_so_far = max(d1_high, d1_close, t1_peak)
        seen_cons = False
        cons_days = 0
        reentry_bar = None
        cons_low = l_arr[t2_search_start]
        peak_high = t1_peak_so_far
        
        search_end = min(max_fwd_idx, t2_search_start + 90)
        for b in range(t2_search_start, search_end):
            low_b = l_arr[b]
            high_b = h_arr[b]
            s_b = states[b]
            
            if low_b < cons_low: cons_low = low_b
            if high_b > peak_high: peak_high = high_b
            
            if s_b <= 0:
                seen_cons = True
                cons_days += 1
                
            if seen_cons and b > t2_search_start and s_b == 1:
                reentry_bar = b
                break
                
        t2_has = False
        t2_r = 0.0
        t2_exit_bar = None
        
        if reentry_bar is not None and 1 <= cons_days <= 45:
            drop_from_peak = (peak_high - cons_low) / peak_high * 100.0 if peak_high > 0 else 0.0
            sma50_val = sma50_arr[reentry_bar]
            holds_sma50 = np.isnan(sma50_val) or (c_arr[reentry_bar] >= sma50_val * 0.98)
            
            if drop_from_peak <= 55.0 and holds_sma50:
                swing5_low = np.min(l_arr[max(0, reentry_bar - 5): reentry_bar + 1])
                t2_stop = swing5_low
                
                # Zero-lookahead next open order execution
                if reentry_bar + 1 < n_bars:
                    entry_bar = reentry_bar + 1
                    t2_entry = o_arr[entry_bar]
                else:
                    entry_bar = None
                    t2_entry = None
                    
                if entry_bar is not None and t2_entry > t2_stop:
                    risk_dollars = t2_entry - t2_stop
                    risk_pct_t2 = risk_dollars / t2_entry * 100.0
                    
                    if 0.1 <= risk_pct_t2 <= 35.0:
                        t2_has = True
                        stopped = False
                        exit_bar = None
                        exit_price = None
                        current_stop = t2_stop
                        
                        for b in range(entry_bar, min(max_fwd_idx, entry_bar + 90)):
                            b_open = o_arr[b]
                            b_low = l_arr[b]
                            b_close = c_arr[b]
                            b_state = states[b]
                            
                            if b == entry_bar:
                                if b_low <= current_stop:
                                    stopped = True
                                    exit_bar = b
                                    exit_price = current_stop
                                    break
                            else:
                                if b_open < current_stop:
                                    stopped = True
                                    exit_bar = b
                                    exit_price = b_open
                                    break
                                if b_low <= current_stop:
                                    stopped = True
                                    exit_bar = b
                                    exit_price = current_stop
                                    break
                                    
                            # Exit rule: reverse flip to bearish (-1)
                            if b_state == -1:
                                exit_bar = b
                                exit_price = b_close
                                break
                                
                        if exit_bar is None:
                            exit_bar = min(max_fwd_idx - 1, entry_bar + 89)
                            exit_price = c_arr[exit_bar]
                            
                        t2_r = (exit_price - t2_entry) / risk_dollars
                        t2_exit_bar = exit_bar
                        
                        # Feature extraction for calibrated ML scoring at reentry_bar
                        c_t = c_arr[reentry_bar]
                        e8 = sym_data[8][reentry_bar]
                        e12 = sym_data[12][reentry_bar]
                        e16 = sym_data[16][reentry_bar]
                        e21 = sym_data[21][reentry_bar]
                        e_arr = np.array([e8, e12, e16, e21])
                        vol20 = vol20_arr[reentry_bar]
                        sma50_v = sma50_arr[reentry_bar]
                        
                        t2_feat_rows.append({
                            "feature_days_since_ep": reentry_bar - d1_idx,
                            "feature_drawdown_from_peak": -drop_from_peak / 100.0,
                            "feature_dist_10ema": (c_t - ema10_arr[reentry_bar]) / ema10_arr[reentry_bar] if ema10_arr[reentry_bar] > 0 else 0.0,
                            "feature_dist_20ema": (c_t - ema20_arr[reentry_bar]) / ema20_arr[reentry_bar] if ema20_arr[reentry_bar] > 0 else 0.0,
                            "feature_dist_50sma": (c_t - sma50_v) / sma50_v if (not np.isnan(sma50_v) and sma50_v > 0) else 0.0,
                            "feature_vol_contraction": v_arr[reentry_bar] / vol20 if vol20 > 0 else 1.0,
                            "feature_is_yellow_flip": 1,
                            "feature_ribbon_compression": float(np.std(e_arr, ddof=1) / np.mean(e_arr)) if np.mean(e_arr) > 0 else 0.0,
                            "feature_ribbon_bandwidth": (e8 - e21) / e21 if e21 > 0 else 0.0,
                            "feature_dist_ribbon_ema8": (c_t - e8) / e8 if e8 > 0 else 0.0,
                            "feature_dist_ribbon_ema21": (c_t - e21) / e21 if e21 > 0 else 0.0,
                        })
                        t2_feat_indices.append(i)

        # --- 3. Trade 3: Second Continuation Re-entry (Runner Leg) ---
        t3_has = False
        t3_r = 0.0
        
        if t2_exit_bar is not None and t2_exit_bar + 10 < max_fwd_idx:
            t2_peak_so_far = max(c_arr[t2_exit_bar], h_arr[t2_exit_bar])
            seen_cons_3 = False
            cons_days_3 = 0
            cons_low_3 = l_arr[t2_exit_bar]
            reentry_bar_3 = None
            
            search_end_3 = min(max_fwd_idx, t2_exit_bar + 90)
            for b in range(t2_exit_bar, search_end_3):
                low_b = l_arr[b]
                high_b = h_arr[b]
                s_b = states[b]
                
                if high_b > t2_peak_so_far: t2_peak_so_far = high_b
                if low_b < cons_low_3: cons_low_3 = low_b
                
                if s_b <= 0:
                    seen_cons_3 = True
                    cons_days_3 += 1
                    
                if seen_cons_3 and b > t2_exit_bar and s_b == 1:
                    reentry_bar_3 = b
                    break
                    
            if reentry_bar_3 is not None and 1 <= cons_days_3 <= 45:
                drop_3 = (t2_peak_so_far - cons_low_3) / t2_peak_so_far * 100.0 if t2_peak_so_far > 0 else 0.0
                sma50_val3 = sma50_arr[reentry_bar_3]
                holds_sma50_3 = np.isnan(sma50_val3) or (c_arr[reentry_bar_3] >= sma50_val3 * 0.98)
                
                if drop_3 <= 55.0 and holds_sma50_3:
                    swing5_3 = np.min(l_arr[max(0, reentry_bar_3 - 5): reentry_bar_3 + 1])
                    t3_stop = swing5_3
                    
                    if reentry_bar_3 + 1 < n_bars:
                        entry_bar_3 = reentry_bar_3 + 1
                        t3_entry = o_arr[entry_bar_3]
                    else:
                        entry_bar_3 = None
                        t3_entry = None
                        
                    if entry_bar_3 is not None and t3_entry > t3_stop:
                        risk_dollars_3 = t3_entry - t3_stop
                        risk_pct_t3 = risk_dollars_3 / t3_entry * 100.0
                        
                        if 0.1 <= risk_pct_t3 <= 35.0:
                            t3_has = True
                            stopped_3 = False
                            exit_bar_3 = None
                            exit_price_3 = None
                            current_stop_3 = t3_stop
                            
                            for b in range(entry_bar_3, min(max_fwd_idx, entry_bar_3 + 90)):
                                b_open = o_arr[b]
                                b_low = l_arr[b]
                                b_close = c_arr[b]
                                b_state = states[b]
                                
                                if b == entry_bar_3:
                                    if b_low <= current_stop_3:
                                        stopped_3 = True
                                        exit_bar_3 = b
                                        exit_price_3 = current_stop_3
                                        break
                                else:
                                    if b_open < current_stop_3:
                                        stopped_3 = True
                                        exit_bar_3 = b
                                        exit_price_3 = b_open
                                        break
                                    if b_low <= current_stop_3:
                                        stopped_3 = True
                                        exit_bar_3 = b
                                        exit_price_3 = current_stop_3
                                        break
                                        
                                if b_state == -1:
                                    exit_bar_3 = b
                                    exit_price_3 = b_close
                                    break
                                    
                            if exit_bar_3 is None:
                                exit_bar_3 = min(max_fwd_idx - 1, entry_bar_3 + 89)
                                exit_price_3 = c_arr[exit_bar_3]
                                
                            t3_r = (exit_price_3 - t3_entry) / risk_dollars_3
                            
                            # Feature extraction for ML scoring
                            c_t3 = c_arr[reentry_bar_3]
                            e8 = sym_data[8][reentry_bar_3]
                            e12 = sym_data[12][reentry_bar_3]
                            e16 = sym_data[16][reentry_bar_3]
                            e21 = sym_data[21][reentry_bar_3]
                            e_arr = np.array([e8, e12, e16, e21])
                            vol20_3 = vol20_arr[reentry_bar_3]
                            sma50_v3 = sma50_arr[reentry_bar_3]
                            
                            t3_feat_rows.append({
                                "feature_days_since_ep": reentry_bar_3 - d1_idx,
                                "feature_drawdown_from_peak": -drop_3 / 100.0,
                                "feature_dist_10ema": (c_t3 - ema10_arr[reentry_bar_3]) / ema10_arr[reentry_bar_3] if ema10_arr[reentry_bar_3] > 0 else 0.0,
                                "feature_dist_20ema": (c_t3 - ema20_arr[reentry_bar_3]) / ema20_arr[reentry_bar_3] if ema20_arr[reentry_bar_3] > 0 else 0.0,
                                "feature_dist_50sma": (c_t3 - sma50_v3) / sma50_v3 if (not np.isnan(sma50_v3) and sma50_v3 > 0) else 0.0,
                                "feature_vol_contraction": v_arr[reentry_bar_3] / vol20_3 if vol20_3 > 0 else 1.0,
                                "feature_is_yellow_flip": 1,
                                "feature_ribbon_compression": float(np.std(e_arr, ddof=1) / np.mean(e_arr)) if np.mean(e_arr) > 0 else 0.0,
                                "feature_ribbon_bandwidth": (e8 - e21) / e21 if e21 > 0 else 0.0,
                                "feature_dist_ribbon_ema8": (c_t3 - e8) / e8 if e8 > 0 else 0.0,
                                "feature_dist_ribbon_ema21": (c_t3 - e21) / e21 if e21 > 0 else 0.0,
                            })
                            t3_feat_indices.append(i)

        results.append({
            "t1_real_r": round(float(t1_r), 3),
            "t1_real_ret": round(float(t1_ret), 2),
            "t1_real_risk_pct": round(float(risk_pct), 2),
            "t1_real_exit_reason": t1_reason,
            "t1_real_hold_days": int(t1_exit_bar - d1_idx),
            "t1_pyramid_r": round(float(t1_pyramid_r), 3),
            "t1_dyn_r": round(float(dyn_r), 3),
            "t2_has_reentry": bool(t2_has),
            "t2_real_r": round(float(t2_r), 3) if t2_has else 0.0,
            "t2_prob_reentry": 0.0,
            "t3_has_reentry": bool(t3_has),
            "t3_real_r": round(float(t3_r), 3) if t3_has else 0.0,
            "t3_prob_reentry": 0.0,
        })

    t_end = time.time()
    print(f"✅ Executed event simulations in {t_end - t_start:.2f}s")
    
    # Batch predict Re-entry probabilities
    if t2_feat_rows and clf is not None:
        print(f"🤖 Predicting ML Re-entry conviction on {len(t2_feat_rows)} Trade 2 setups...")
        p2 = clf.predict_proba(pd.DataFrame(t2_feat_rows)[REENTRY_FEATS])[:, 1]
        for idx_pos, prob in zip(t2_feat_indices, p2):
            if results[idx_pos] is not None:
                results[idx_pos]["t2_prob_reentry"] = round(float(prob), 3)

    if t3_feat_rows and clf is not None:
        print(f"🤖 Predicting ML Re-entry conviction on {len(t3_feat_rows)} Trade 3 setups...")
        p3 = clf.predict_proba(pd.DataFrame(t3_feat_rows)[REENTRY_FEATS])[:, 1]
        for idx_pos, prob in zip(t3_feat_indices, p3):
            if results[idx_pos] is not None:
                results[idx_pos]["t3_prob_reentry"] = round(float(prob), 3)

    # Convert to DataFrame
    res_df = pd.DataFrame([r if r is not None else {
        "t1_real_r": 0.0, "t1_real_ret": 0.0, "t1_real_risk_pct": 10.0,
        "t1_real_exit_reason": "Missing Bars", "t1_real_hold_days": 0,
        "t1_pyramid_r": 0.0, "t1_dyn_r": 0.0,
        "t2_has_reentry": False, "t2_real_r": 0.0, "t2_prob_reentry": 0.0,
        "t3_has_reentry": False, "t3_real_r": 0.0, "t3_prob_reentry": 0.0,
    } for r in results])

    for col in res_df.columns:
        df[col] = res_df[col]

    df.to_parquet(parquet_path, index=False)
    print(f"💾 Successfully saved updated simulation columns to {parquet_path}!")

    # Summary Statistics
    valid = df[df["t1_real_exit_reason"] != "Missing Bars"]
    print("\n" + "=" * 80)
    print(f"📊 UNIVERSE GROUND TRUTH PERFORMANCE (N={len(valid)} EP EVENTS):")
    print("=" * 80)
    print(f"Trade 1 Baseline:         N={len(valid)} | EV: +{valid['t1_real_r'].mean():.2f} R (Win Rate: {(valid['t1_real_r'] > 0).mean():.1%}, Total: +{valid['t1_real_r'].sum():.1f} R)")
    print(f"Trade 1 Pyramiding:       N={len(valid)} | EV: +{valid['t1_pyramid_r'].mean():.2f} R (Win Rate: {(valid['t1_pyramid_r'] > 0).mean():.1%}, Total: +{valid['t1_pyramid_r'].sum():.1f} R)")
    print(f"Trade 1 Dynamic Stop:     N={len(valid)} | EV: +{valid['t1_dyn_r'].mean():.2f} R (Win Rate: {(valid['t1_dyn_r'] > 0).mean():.1%}, Total: +{valid['t1_dyn_r'].sum():.1f} R)")
    
    t2_active = valid[valid["t2_has_reentry"]]
    t3_active = valid[valid["t3_has_reentry"]]
    print(f"Trade 2 Continuation:     N={len(t2_active)} | EV: +{t2_active['t2_real_r'].mean():.2f} R (Win Rate: {(t2_active['t2_real_r'] > 0).mean():.1%}, Total: +{t2_active['t2_real_r'].sum():.1f} R)")
    print(f"Trade 3 Continuation:     N={len(t3_active)} | EV: +{t3_active['t3_real_r'].mean():.2f} R (Win Rate: {(t3_active['t3_real_r'] > 0).mean():.1%}, Total: +{t3_active['t3_real_r'].sum():.1f} R)")
    
    all_cont = np.concatenate([t2_active["t2_real_r"].values, t3_active["t3_real_r"].values])
    print(f"Total Continuation (2+3): N={len(all_cont)} | EV: +{all_cont.mean():.2f} R (Win Rate: {(all_cont > 0).mean():.1%}, Total: +{all_cont.sum():.1f} R)")
    
    full_port = np.concatenate([valid["t1_real_r"].values, all_cont])
    print(f"Total Portfolio Trades:   N={len(full_port)} | EV: +{full_port.mean():.2f} R (Win Rate: {(full_port > 0).mean():.1%}, Total: +{full_port.sum():.1f} R)")

    # Pinnacle Elite Filters (Consistent with serve_ep.py)
    from serve_ep import TOP_SECTORS, TOP_THEMES
    pin = valid[
        (valid['close_pos'] >= 0.65) &
        (valid['rvol'] >= 2.5) &
        (valid['gap_pct'] >= 5.0) &
        (valid['sector'].isin(TOP_SECTORS) | valid['theme'].isin(TOP_THEMES)) &
        ((valid['sec_m1_pctile'] >= 50) | (valid['thm_m1_pctile'] >= 50))
    ]
    
    pin_t2 = pin[pin["t2_has_reentry"]]
    pin_t3 = pin[pin["t3_has_reentry"]]
    pin_cont = np.concatenate([pin_t2["t2_real_r"].values, pin_t3["t3_real_r"].values])
    pin_all = np.concatenate([pin["t1_real_r"].values, pin_cont])
    
    print("\n" + "=" * 80)
    print(f"🏆 REAL PINNACLE ELITE GROUND TRUTH (ZERO LOOKAHEAD N={len(pin)} SETUPS):")
    print("=" * 80)
    print(f"Trade 1 Baseline:         N={len(pin)} | EV: +{pin['t1_real_r'].mean():.2f} R (Win Rate: {(pin['t1_real_r'] > 0).mean():.1%}, Total: +{pin['t1_real_r'].sum():.1f} R)")
    print(f"Trade 1 Pyramiding:       N={len(pin)} | EV: +{pin['t1_pyramid_r'].mean():.2f} R (Win Rate: {(pin['t1_pyramid_r'] > 0).mean():.1%}, Total: +{pin['t1_pyramid_r'].sum():.1f} R)")
    print(f"Trade 1 Dynamic Stop:     N={len(pin)} | EV: +{pin['t1_dyn_r'].mean():.2f} R (Win Rate: {(pin['t1_dyn_r'] > 0).mean():.1%}, Total: +{pin['t1_dyn_r'].sum():.1f} R)")
    print(f"Trade 2 Continuation:     N={len(pin_t2)} | EV: +{pin_t2['t2_real_r'].mean():.2f} R (Win Rate: {(pin_t2['t2_real_r'] > 0).mean():.1%}, Total: +{pin_t2['t2_real_r'].sum():.1f} R)")
    print(f"Trade 3 Continuation:     N={len(pin_t3)} | EV: +{pin_t3['t3_real_r'].mean():.2f} R (Win Rate: {(pin_t3['t3_real_r'] > 0).mean():.1%}, Total: +{pin_t3['t3_real_r'].sum():.1f} R)")
    print(f"Pinnacle Continuation:    N={len(pin_cont)} | EV: +{pin_cont.mean():.2f} R (Win Rate: {(pin_cont > 0).mean():.1%}, Total: +{pin_cont.sum():.1f} R)")
    print(f"Pinnacle Full Portfolio:  N={len(pin_all)} | EV: +{pin_all.mean():.2f} R (Win Rate: {(pin_all > 0).mean():.1%}, Total: +{pin_all.sum():.1f} R)")
    
    # ML Gated Continuation
    pin_t2_ml = pin_t2[pin_t2["t2_prob_reentry"] >= 0.35]
    pin_t3_ml = pin_t3[pin_t3["t3_prob_reentry"] >= 0.35]
    pin_cont_ml = np.concatenate([pin_t2_ml["t2_real_r"].values, pin_t3_ml["t3_real_r"].values])
    print(f"\n🤖 Pinnacle ML Gated (≥35%): N={len(pin_cont_ml)} trades ({len(pin_t2_ml)} T2 + {len(pin_t3_ml)} T3) | EV: +{pin_cont_ml.mean():.2f} R (Win Rate: {(pin_cont_ml > 0).mean():.1%}, Total: +{pin_cont_ml.sum():.1f} R)")


if __name__ == "__main__":
    run_batch_simulation()
