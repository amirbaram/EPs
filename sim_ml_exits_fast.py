import pandas as pd
import numpy as np
import datastore
from ep_ml_engine import engine
from sim_pinnacle_tests import load_pinnacle_events
import time

def run_ml_simulation():
    events = load_pinnacle_events()
    print(f"Loaded {len(events)} Pinnacle Elite events for ML Simulation.")
    
    # Just run on a random subset of 100 to get a statistically significant estimate
    import random
    random.seed(42)
    events = random.sample(events, min(100, len(events)))
    
    res_ml = []
    res_base = []
    
    for i, ev in enumerate(events):
        if i % 10 == 0:
            print(f"Processing {i}/{len(events)}...")
            
        sym = ev["symbol"]
        date = ev["date"]
        
        df = datastore.load_bars(sym)
        if df is None or date not in df.index: continue
        
        idx = df.index.get_loc(date)
        if idx < 20 or idx + 1 >= len(df): continue
        
        d1_close = df["close"].iloc[idx]
        d1_low = df["low"].iloc[idx]
        risk_d1 = d1_close - d1_low
        if risk_d1 <= 0: continue
            
        ml_r = 0.0
        base_r = 0.0
        
        shares = 1.0
        base_shares = 1.0
        
        current_stop = d1_low
        
        df["sma50"] = df["close"].rolling(50).mean()
        
        for t_idx in range(idx + 1, len(df)):
            c = df["close"].iloc[t_idx]
            h = df["high"].iloc[t_idx]
            l = df["low"].iloc[t_idx]
            sma50 = df["sma50"].iloc[t_idx]
            
            # Base Exits
            if base_shares > 0:
                if l < d1_low:
                    base_r -= base_shares * 1.0
                    base_shares = 0
                elif not pd.isna(sma50) and c < sma50:
                    base_r += (c - d1_close) / risk_d1 * base_shares
                    base_shares = 0
            
            # ML Exits
            if shares > 0:
                if l < current_stop:
                    ml_r += (current_stop - d1_close) / risk_d1 * shares
                    shares = 0
                else:
                    # ML Inference (limit up to 60 days to prevent infinite loops)
                    if t_idx - idx <= 60:
                        features = engine.compute_rolling_features(sym, idx, t_idx)
                        if features is not None:
                            preds = engine.predict_rolling(features)
                            prob_exh = preds.get("prob_exhaustion", 0.0)
                            if prob_exh > 0.80 and shares == 1.0:
                                ml_r += (c - d1_close) / risk_d1 * 0.5
                                shares -= 0.5
                                
                            dyn_stop_pct = preds.get("dynamic_stop_loss_pct", None)
                            if dyn_stop_pct is not None and not np.isnan(dyn_stop_pct):
                                new_stop = c * (1.0 + dyn_stop_pct)
                                if new_stop > current_stop:
                                    current_stop = new_stop
                    else:
                        # Fallback to 50MA if trade goes past 60 days
                        if not pd.isna(sma50) and c < sma50:
                            ml_r += (c - d1_close) / risk_d1 * shares
                            shares = 0

            if shares == 0 and base_shares == 0:
                break
                
        res_ml.append(ml_r)
        res_base.append(base_r)

    print("\n===============================")
    print("   PINNACLE ELITE + ML EXITS   ")
    print("===============================\n")
    
    r_ml = np.array(res_ml)
    r_base = np.array(res_base)
    
    print("BASE (50 MA) Exit")
    print(f"   Win Rate:     {(r_base > 0).mean()*100:.2f}%")
    print(f"   Average EV:   {r_base.mean():.2f} R")
    
    print("\nML Exits | Stop: ML Dynamic | Exit: ML Exhaustion (1/2)")
    print(f"   Win Rate:     {(r_ml > 0).mean()*100:.2f}%")
    print(f"   Average EV:   {r_ml.mean():.2f} R")

if __name__ == "__main__":
    run_ml_simulation()
