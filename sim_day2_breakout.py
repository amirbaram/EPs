import pandas as pd
import numpy as np
import datastore

def simulate_day2_breakout():
    # Load all events from parquet
    try:
        study = pd.read_parquet("data/simulations/ep_combined_study_scored.parquet")
    except Exception:
        study = pd.read_parquet("data/simulations/ep_combined_study.parquet")
        
    events = study[["symbol", "date"]].to_dict('records')
    print(f"Total events to scan: {len(events)}")
    
    trades = []
    
    for i, ev in enumerate(events):
        if i % 1000 == 0: print(f"Processed {i}/{len(events)}")
        sym = ev["symbol"]
        date = ev["date"]
        
        df = datastore.load_bars(sym)
        if df is None or date not in df.index:
            continue
            
        idx = df.index.get_loc(date)
        
        # We need at least Day 1 and Day 2, and future bars
        if idx + 2 >= len(df):
            continue
            
        d2_idx = idx + 1
        d2_high = df["high"].iloc[d2_idx]
        d2_low = df["low"].iloc[d2_idx]
        
        # The stop loss is Day 2 Low
        stop_price = d2_low
        entry_trigger = d2_high
        
        entered = False
        entry_idx = -1
        entry_price = 0.0
        
        # Look for entry within 10 days of D2 (up to d2_idx + 10)
        for t_idx in range(d2_idx + 1, min(len(df), d2_idx + 11)):
            # If it breaks the stop BEFORE or ON the trigger day, the setup is void
            if df["low"].iloc[t_idx] < stop_price:
                # If the high of this bar ALSO broke the trigger, it's ambiguous. Let's assume stopped out (worst case)
                break
                
            if df["high"].iloc[t_idx] > entry_trigger:
                entry_price = max(entry_trigger, df["open"].iloc[t_idx])
                entered = True
                entry_idx = t_idx
                break
                
        if not entered:
            continue
            
        # We are in the trade!
        target_50 = entry_price * 1.50
        target_100 = entry_price * 2.00
        target_150 = entry_price * 2.50
        
        risk = entry_price - stop_price
        if risk <= 0: continue
        
        # Sim outcomes
        t1_hit = False
        t2_hit = False
        t3_hit = False
        stopped = False
        
        for t_idx in range(entry_idx, min(len(df), entry_idx + 250)):
            if df["low"].iloc[t_idx] < stop_price:
                stopped = True
                break
                
            high = df["high"].iloc[t_idx]
            if high >= target_50: t1_hit = True
            if high >= target_100: t2_hit = True
            if high >= target_150: t3_hit = True
            
            if t3_hit: break
                
        r_total = 0.0
        
        if t1_hit: r_total += (target_50 - entry_price) / risk * (1/3)
        else: r_total -= 1.0 * (1/3)
        
        if t2_hit: r_total += (target_100 - entry_price) / risk * (1/3)
        else: r_total -= 1.0 * (1/3)
        
        if t3_hit: r_total += (target_150 - entry_price) / risk * (1/3)
        else: r_total -= 1.0 * (1/3)
        
        trades.append(r_total)

    trades = np.array(trades)
    print("\n--- RESULTS: Day 2 Breakout Strategy (Full Universe) ---")
    print(f"Total Triggered Trades: {len(trades)}")
    print(f"Win Rate (>0 R): {(trades > 0).mean() * 100:.2f}%")
    print(f"Average Expected Value (EV): {trades.mean():.2f} R")
    print(f"Total Strategy P&L: {trades.sum():.2f} R")

if __name__ == "__main__":
    simulate_day2_breakout()
