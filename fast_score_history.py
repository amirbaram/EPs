import pandas as pd
import numpy as np
import time
from pathlib import Path
from ep_ml_engine import engine
import datastore

def run_scoring():
    print("🚀 FAST BATCH SCORING WITH NEW ZERO-LOOKAHEAD RETRAINED V2 MODELS")
    in_path = Path("data/simulations/ep_combined_study.parquet")
    df = pd.read_parquet(in_path)
    print(f"Loaded {len(df)} EP events across {df['symbol'].nunique()} unique symbols.")
    
    # Pre-allocate result columns
    n = len(df)
    ml_50 = np.full(n, np.nan, dtype=np.float32)
    ml_100 = np.full(n, np.nan, dtype=np.float32)
    ml_150 = np.full(n, np.nan, dtype=np.float32)
    ml_200 = np.full(n, np.nan, dtype=np.float32)
    dyn_stop = np.full(n, np.nan, dtype=np.float32)
    is_toxic = np.zeros(n, dtype=bool)

    t0 = time.time()
    processed = 0

    # Group rows by symbol to load bars exactly ONCE per symbol
    grouped = df.groupby("symbol")
    
    for sym, group in grouped:
        bars = datastore.load_bars(sym)
        if bars is None:
            continue
            
        for orig_idx, row in group.iterrows():
            d_str = row["date"]
            try:
                d1_idx = bars.index.get_loc(d_str)
                t_idx = min(len(bars) - 1, d1_idx + 4)
                feats = engine.compute_rolling_features(sym, d1_idx, t_idx)
                if feats is not None:
                    preds = engine.predict_rolling(feats)
                    ml_50[orig_idx] = preds["prob_50"]
                    ml_100[orig_idx] = preds["prob_100"]
                    ml_150[orig_idx] = preds["prob_150"]
                    ml_200[orig_idx] = preds["prob_200"]
                    dyn_stop[orig_idx] = preds.get("dynamic_stop_loss_pct", np.nan)
            except Exception:
                pass
            processed += 1
            if processed % 1000 == 0:
                elapsed = time.time() - t0
                print(f"  Processed {processed}/{n} events ({elapsed:.1f}s, {processed/elapsed:.1f} ev/s)...")

    df["ml_50"] = ml_50
    df["ml_100"] = ml_100
    df["ml_150"] = ml_150
    df["ml_200"] = ml_200
    df["is_toxic"] = is_toxic
    df["dynamic_stop_loss_pct"] = dyn_stop

    out_path = Path("data/simulations/ep_combined_study_scored.parquet")
    backup_path = Path("data/simulations/ep_combined_study_scored_v1_backup.parquet")
    if out_path.exists() and not backup_path.exists():
        import shutil
        shutil.copy(out_path, backup_path)
        print(f"Backed up previous scored file to {backup_path}")
        
    df.to_parquet(out_path)
    total_time = time.time() - t0
    print(f"✅ Successfully re-scored all {n} events in {total_time:.1f}s and saved to {out_path}!")

if __name__ == "__main__":
    run_scoring()
