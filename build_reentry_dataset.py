import pandas as pd
import numpy as np
import datastore
import os

def build_reentry_dataset():
    print("🚀 BUILDING RE-ENTRY ML DATASET (TRADE 2 / TRADE 3)")
    df_ep = pd.read_parquet("data/simulations/ep_combined_study_scored.parquet")
    
    events = df_ep[["symbol", "date"]].to_dict('records')
    dataset = []
    
    for i, ev in enumerate(events):
        if i % 100 == 0:
            print(f"Processing {i}/{len(events)}...")
            
        sym = ev["symbol"]
        ep_date = ev["date"]
        
        df = datastore.load_bars(sym)
        if df is None or ep_date not in df.index: continue
        
        ep_idx = df.index.get_loc(ep_date)
        if ep_idx + 20 >= len(df): continue
        
        df = df.copy()
        df["ema3"] = df["close"].ewm(span=3, adjust=False).mean()
        df["ema8_low"] = df["low"].ewm(span=8, adjust=False).mean()
        df["ema10"] = df["close"].ewm(span=10, adjust=False).mean()
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["sma50"] = df["close"].rolling(50).mean()
        df["vol20"] = df["volume"].rolling(20).mean()
        
        # Identify Yellow Flips (Larsson Line)
        df["is_yellow_flip"] = ((df["ema3"] > df["ema8_low"]) & (df["ema3"].shift(1) <= df["ema8_low"].shift(1))).astype(int)
        
        # Scan days 5 to 120 post-EP
        for t_idx in range(ep_idx + 5, min(ep_idx + 120, len(df) - 20)):
            c = df["close"].iloc[t_idx]
            sma50 = df["sma50"].iloc[t_idx]
            
            # Re-entry candidates must be structurally intact (above 50 SMA)
            if pd.isna(sma50) or c < sma50: continue
            
            # Find the peak since EP
            peak_high = df["high"].iloc[ep_idx:t_idx].max()
            drawdown_from_peak = (c - peak_high) / peak_high if peak_high > 0 else 0
            
            # We only want days where the stock is in a pullback (at least -5% from peak)
            if drawdown_from_peak > -0.05: continue
            
            ema10 = df["ema10"].iloc[t_idx]
            ema20 = df["ema20"].iloc[t_idx]
            vol20 = df["vol20"].iloc[t_idx]
            vol = df["volume"].iloc[t_idx]
            
            dist_10 = (c - ema10) / ema10 if ema10 > 0 else 0
            dist_20 = (c - ema20) / ema20 if ema20 > 0 else 0
            dist_50 = (c - sma50) / sma50 if sma50 > 0 else 0
            vol_contraction = vol / vol20 if vol20 > 0 else 1.0
            
            # Forward 20-day evaluation
            f_20d = df.iloc[t_idx+1:t_idx+21]
            max_fwd = (f_20d["high"].max() - c) / c
            min_fwd = (f_20d["low"].min() - c) / c
            
            # Target: Does it hit +15% before dropping -8%?
            hit_15 = False
            for f_idx in range(len(f_20d)):
                if f_20d["low"].iloc[f_idx] < c * 0.92:
                    break
                if f_20d["high"].iloc[f_idx] >= c * 1.15:
                    hit_15 = True
                    break
                    
            dataset.append({
                "symbol": sym,
                "date": str(df.index[t_idx].date()),
                "entry_date": ep_date, # the original EP anchor
                "feature_days_since_ep": t_idx - ep_idx,
                "feature_drawdown_from_peak": drawdown_from_peak,
                "feature_dist_10ema": dist_10,
                "feature_dist_20ema": dist_20,
                "feature_dist_50sma": dist_50,
                "feature_vol_contraction": vol_contraction,
                "feature_is_yellow_flip": int(df["is_yellow_flip"].iloc[t_idx]),
                "target_15pct_win": int(hit_15),
                "forward_max_20d": max_fwd,
                "forward_min_20d": min_fwd
            })
            
    out_df = pd.DataFrame(dataset)
    os.makedirs("data/ml_datasets/reentry", exist_ok=True)
    out_df.to_parquet("data/ml_datasets/reentry/dataset_reentry.parquet")
    print(f"✅ Re-entry dataset built with {len(out_df)} candidate rows.")
    print(f"Positive +15% Hit Rate: {out_df['target_15pct_win'].mean()*100:.2f}%")

if __name__ == "__main__":
    build_reentry_dataset()
