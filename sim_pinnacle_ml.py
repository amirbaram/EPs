import pandas as pd
import numpy as np
import datastore
from ep_ml_engine import engine
from sim_pinnacle_tests import load_pinnacle_events

def run_ml_simulation():
    events = load_pinnacle_events()
    print(f"Loaded {len(events)} Pinnacle Elite events for ML Simulation.")
    
    res_ml = []
    
    for i, ev in enumerate(events):
        if i % 50 == 0:
            print(f"Processing {i}/{len(events)}...")
            
        sym = ev["symbol"]
        date = ev["date"]
        
        df = datastore.load_bars(sym)
        if df is None or date not in df.index: continue
        
        idx = df.index.get_loc(date)
        if idx < 20 or idx + 1 >= len(df): continue
        
        # --- D1 Entry Variables ---
        d1_close = df["close"].iloc[idx]
        d1_low = df["low"].iloc[idx]
        risk_d1 = d1_close - d1_low
        
        if risk_d1 <= 0: continue
            
        ml_r = 0.0
        shares = 1.0
        
        # We start with the D1 Low as the absolute stop
        current_stop = d1_low
        
        for t_idx in range(idx + 1, len(df)):
            c = df["close"].iloc[t_idx]
            h = df["high"].iloc[t_idx]
            l = df["low"].iloc[t_idx]
            
            # 1. Did we hit our trailing stop intraday?
            if l < current_stop:
                if shares > 0:
                    ml_r += (current_stop - d1_close) / risk_d1 * shares
                    shares = 0
                break
                
            # If we survived today, calculate ML inferences based on today's close for TOMORROW
            # We don't want lookahead bias.
            features = engine.compute_rolling_features(sym, idx, t_idx)
            if features is None:
                # E.g. we dropped below D1 low at some point but maybe it was a gap?
                continue
                
            preds = engine.predict_rolling(features)
            
            # Check Exhaustion (SELL HALF)
            prob_exh = preds.get("prob_exhaustion", 0.0)
            if prob_exh > 0.80 and shares == 1.0:
                # Sell half at tomorrow's open (we use today's close as a proxy for simplicity)
                ml_r += (c - d1_close) / risk_d1 * 0.5
                shares -= 0.5
                
            # Check Dynamic Stop
            dyn_stop_pct = preds.get("dynamic_stop_loss_pct", None)
            if dyn_stop_pct is not None and not np.isnan(dyn_stop_pct):
                # The regressor predicted the 10th percentile MAE.
                # So the expected "maximum normal pullback" is dyn_stop_pct.
                # We place our stop just below it.
                new_stop = c * (1.0 + dyn_stop_pct)
                if new_stop > current_stop:
                    current_stop = new_stop
                    
        res_ml.append(ml_r)

    print("\n===============================")
    print("   PINNACLE ELITE + ML EXITS   ")
    print("===============================\n")
    
    r_ml = np.array(res_ml)
    print("ML Exits | Stop: ML Dynamic | Exit: ML Exhaustion (1/2)")
    print(f"   Total Trades: {len(r_ml)}")
    print(f"   Win Rate:     {(r_ml > 0).mean()*100:.2f}%")
    print(f"   Average EV:   {r_ml.mean():.2f} R")
    print(f"   Total P&L:    {r_ml.sum():.2f} R\n")

if __name__ == "__main__":
    run_ml_simulation()
