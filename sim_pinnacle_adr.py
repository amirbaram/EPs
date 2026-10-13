import pandas as pd
import numpy as np
import datastore

def load_pinnacle_events():
    df = pd.read_parquet("data/simulations/ep_combined_study_scored.parquet")
    TOP_SECTORS = [
        "Computer Hardware", "Biotechnology", "Healthcare", "Technology", "Semiconductors",
        "Industrials", "Capital Markets", "Financial Services", "Specialty Industrial Machinery"
    ]
    TOP_THEMES = [
        "Semiconductors", "Biotechnology", "Software - Infrastructure", "Computer Hardware",
        "Health Information Services", "Medical Devices", "Software - Application"
    ]
    q = df[
        (~df["violated_48h"]) &
        (df["close_pos"] >= 0.65) &
        (df["rvol"] >= 2.5) &
        (df["gap_pct"] >= 5.0) &
        (df["sector"].isin(TOP_SECTORS) | df["theme"].isin(TOP_THEMES)) &
        ((df["sec_m1_pctile"] >= 50) | (df["thm_m1_pctile"] >= 50))
    ]
    return q[["symbol", "date"]].to_dict('records')

def run_simulation():
    events = load_pinnacle_events()
    print(f"Loaded {len(events)} Pinnacle Elite events.")
    
    # Store results for multiple scenarios
    # Key: (entry_type, target1_mult, target2_mult)
    scenarios = [
        ("D1", 1, 2),
        ("D1", 3, 5),
        ("D1", 5, 8),
        ("D2", 1, 2),
        ("D2", 3, 5),
        ("D2", 5, 8)
    ]
    
    results = {s: [] for s in scenarios}
    
    for i, ev in enumerate(events):
        sym = ev["symbol"]
        date = ev["date"]
        
        df = datastore.load_bars(sym)
        if df is None or date not in df.index: continue
        
        idx = df.index.get_loc(date)
        if idx < 20 or idx + 1 >= len(df): continue
        
        df["sma50"] = df["close"].rolling(50).mean()
        adr_20 = (df["high"].iloc[idx-20:idx] - df["low"].iloc[idx-20:idx]).mean()
        if pd.isna(adr_20) or adr_20 <= 0: continue
        
        # --- D1 Entry ---
        d1_close = df["close"].iloc[idx]
        d1_low = df["low"].iloc[idx]
        risk_d1 = d1_close - d1_low
        
        # --- D2 Breakout ---
        d2_high = df["high"].iloc[idx+1]
        d2_low = df["low"].iloc[idx+1]
        risk_d2 = d2_high - d2_low
        
        # Helper to run a single test
        def run_test(entry_price, stop_price, risk, start_idx, adr_m1, adr_m2):
            t1 = entry_price + (adr_m1 * adr_20)
            t2 = entry_price + (adr_m2 * adr_20)
            
            r_total = 0.0
            shares = 1.0
            hit_t1 = False
            hit_t2 = False
            
            for t_idx in range(start_idx, len(df)):
                c = df["close"].iloc[t_idx]
                h = df["high"].iloc[t_idx]
                l = df["low"].iloc[t_idx]
                sma50 = df["sma50"].iloc[t_idx]
                
                if l < stop_price:
                    if shares > 0:
                        r_total -= shares * 1.0
                        shares = 0
                    break
                    
                if not hit_t1 and h >= t1 and shares > 0:
                    hit_t1 = True
                    r_total += (t1 - entry_price) / risk * (1/3)
                    shares -= 1/3
                    
                if not hit_t2 and h >= t2 and shares > 0:
                    hit_t2 = True
                    r_total += (t2 - entry_price) / risk * (1/3)
                    shares -= 1/3
                    
                if not pd.isna(sma50) and c < sma50:
                    if shares > 0:
                        r_total += (c - entry_price) / risk * shares
                        shares = 0
                    break
            return r_total
            
        # Run D1 Scenarios
        if risk_d1 > 0:
            for mult1, mult2 in [(1,2), (3,5), (5,8)]:
                res = run_test(d1_close, d1_low, risk_d1, idx + 1, mult1, mult2)
                results[("D1", mult1, mult2)].append(res)
                
        # Run D2 Scenarios
        if risk_d2 > 0 and idx + 2 < len(df):
            triggered = False
            trigger_idx = -1
            trigger_price = 0.0
            
            for t_idx in range(idx + 2, min(len(df), idx + 7)):
                if df["low"].iloc[t_idx] < d2_low: break
                if df["high"].iloc[t_idx] >= d2_high:
                    triggered = True
                    trigger_idx = t_idx
                    trigger_price = max(d2_high, df["open"].iloc[t_idx])
                    break
                    
            if triggered:
                for mult1, mult2 in [(1,2), (3,5), (5,8)]:
                    res = run_test(trigger_price, d2_low, risk_d2, trigger_idx, mult1, mult2)
                    results[("D2", mult1, mult2)].append(res)

    print("\n===============================")
    print("      PARTIAL PROFITS @ HIGHER ADR     ")
    print("===============================\n")
    
    for s in scenarios:
        r = np.array(results[s])
        entry_str = "D1 Close" if s[0] == "D1" else "D2 High Breakout"
        print(f"--- {entry_str} | Target 1: {s[1]} ADR, Target 2: {s[2]} ADR ---")
        print(f"Total Trades: {len(r)}")
        print(f"Win Rate:     {(r > 0).mean()*100:.2f}%")
        print(f"Average EV:   {r.mean():.2f} R")
        print(f"Total P&L:    {r.sum():.2f} R\n")

if __name__ == "__main__":
    run_simulation()
