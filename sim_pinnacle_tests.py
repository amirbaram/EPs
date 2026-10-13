import pandas as pd
import numpy as np
import datastore

TOP_SECTORS = [
    "Computer Hardware", "Biotechnology", "Healthcare", "Technology", "Semiconductors",
    "Industrials", "Capital Markets", "Financial Services", "Specialty Industrial Machinery"
]
TOP_THEMES = [
    "Semiconductors", "Biotechnology", "Software - Infrastructure", "Computer Hardware",
    "Health Information Services", "Medical Devices", "Software - Application"
]

def load_pinnacle_events():
    df = pd.read_parquet("data/simulations/ep_combined_study_scored.parquet")
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
    
    res_s1 = [] # D1 Close, Full 50MA
    res_s2 = [] # D1 Close, Partial ADR + 50MA
    res_s3 = [] # D2 Breakout, Full 50MA
    res_s4 = [] # D2 Breakout, Partial ADR + 50MA
    
    for i, ev in enumerate(events):
        if i % 100 == 0:
            print(f"Processing {i}/{len(events)}...")
            
        sym = ev["symbol"]
        date = ev["date"]
        
        df = datastore.load_bars(sym)
        if df is None or date not in df.index: continue
        
        idx = df.index.get_loc(date)
        if idx < 20 or idx + 1 >= len(df): continue
        
        # Calculate 50MA
        df["sma50"] = df["close"].rolling(50).mean()
        
        # Calculate ADR (Average Daily Range over 20 days prior)
        adr_20 = (df["high"].iloc[idx-20:idx] - df["low"].iloc[idx-20:idx]).mean()
        if pd.isna(adr_20) or adr_20 <= 0: continue
        
        # --- D1 Entry Variables ---
        d1_close = df["close"].iloc[idx]
        d1_low = df["low"].iloc[idx]
        risk_d1 = d1_close - d1_low
        
        # --- D2 Breakout Variables ---
        d2_high = df["high"].iloc[idx+1]
        d2_low = df["low"].iloc[idx+1]
        risk_d2 = d2_high - d2_low
        
        # --- Run D1 Close Strategies ---
        if risk_d1 > 0:
            s1_r = 0.0
            s2_r = 0.0
            shares_s1 = 1.0
            shares_s2 = 1.0
            
            s2_t1 = d1_close + (1.0 * adr_20)
            s2_t2 = d1_close + (2.0 * adr_20)
            
            t1_hit = False
            t2_hit = False
            
            for t_idx in range(idx + 1, len(df)):
                c = df["close"].iloc[t_idx]
                h = df["high"].iloc[t_idx]
                l = df["low"].iloc[t_idx]
                sma50 = df["sma50"].iloc[t_idx]
                
                # Check D1 stop loss
                if l < d1_low:
                    # Stopped out
                    if shares_s1 > 0:
                        s1_r -= shares_s1 * 1.0
                        shares_s1 = 0
                    if shares_s2 > 0:
                        s2_r -= shares_s2 * 1.0
                        shares_s2 = 0
                    break
                    
                # S2 Partial Targets
                if not t1_hit and h >= s2_t1 and shares_s2 > 0:
                    t1_hit = True
                    s2_r += (s2_t1 - d1_close) / risk_d1 * (1/3)
                    shares_s2 -= 1/3
                if not t2_hit and h >= s2_t2 and shares_s2 > 0:
                    t2_hit = True
                    s2_r += (s2_t2 - d1_close) / risk_d1 * (1/3)
                    shares_s2 -= 1/3
                    
                # 50 MA Exit Check
                if not pd.isna(sma50) and c < sma50:
                    if shares_s1 > 0:
                        s1_r += (c - d1_close) / risk_d1 * shares_s1
                        shares_s1 = 0
                    if shares_s2 > 0:
                        s2_r += (c - d1_close) / risk_d1 * shares_s2
                        shares_s2 = 0
                    break
                    
            res_s1.append(s1_r)
            res_s2.append(s2_r)
            
        # --- Run D2 Breakout Strategies ---
        if risk_d2 > 0 and idx + 2 < len(df):
            # Did it trigger?
            triggered = False
            trigger_idx = -1
            trigger_price = 0.0
            
            # Allow 5 days for D2 Breakout
            for t_idx in range(idx + 2, min(len(df), idx + 7)):
                if df["low"].iloc[t_idx] < d2_low:
                    break # Failed
                if df["high"].iloc[t_idx] >= d2_high:
                    triggered = True
                    trigger_idx = t_idx
                    trigger_price = max(d2_high, df["open"].iloc[t_idx])
                    break
                    
            if triggered:
                s3_r = 0.0
                s4_r = 0.0
                shares_s3 = 1.0
                shares_s4 = 1.0
                
                s4_t1 = trigger_price + (1.0 * adr_20)
                s4_t2 = trigger_price + (2.0 * adr_20)
                
                t1_hit = False
                t2_hit = False
                
                for t_idx in range(trigger_idx, len(df)):
                    c = df["close"].iloc[t_idx]
                    h = df["high"].iloc[t_idx]
                    l = df["low"].iloc[t_idx]
                    sma50 = df["sma50"].iloc[t_idx]
                    
                    if l < d2_low:
                        if shares_s3 > 0:
                            s3_r -= shares_s3 * 1.0
                            shares_s3 = 0
                        if shares_s4 > 0:
                            s4_r -= shares_s4 * 1.0
                            shares_s4 = 0
                        break
                        
                    if not t1_hit and h >= s4_t1 and shares_s4 > 0:
                        t1_hit = True
                        s4_r += (s4_t1 - trigger_price) / risk_d2 * (1/3)
                        shares_s4 -= 1/3
                    if not t2_hit and h >= s4_t2 and shares_s4 > 0:
                        t2_hit = True
                        s4_r += (s4_t2 - trigger_price) / risk_d2 * (1/3)
                        shares_s4 -= 1/3
                        
                    if not pd.isna(sma50) and c < sma50:
                        if shares_s3 > 0:
                            s3_r += (c - trigger_price) / risk_d2 * shares_s3
                            shares_s3 = 0
                        if shares_s4 > 0:
                            s4_r += (c - trigger_price) / risk_d2 * shares_s4
                            shares_s4 = 0
                        break
                        
                res_s3.append(s3_r)
                res_s4.append(s4_r)

    print("\n===============================")
    print("      PINNACLE ELITE TESTS     ")
    print("===============================\n")
    
    r1 = np.array(res_s1)
    print("1) Day 1 Close Entry | Stop: D1 Low | Exit: Full at Close < 50MA")
    print(f"   Total Trades: {len(r1)}")
    print(f"   Win Rate:     {(r1 > 0).mean()*100:.2f}%")
    print(f"   Average EV:   {r1.mean():.2f} R")
    print(f"   Total P&L:    {r1.sum():.2f} R\n")
    
    r2 = np.array(res_s2)
    print("2) Day 1 Close Entry | Stop: D1 Low | Exit: 1/3 at 1 ADR, 1/3 at 2 ADR, Runner at 50MA")
    print(f"   Total Trades: {len(r2)}")
    print(f"   Win Rate:     {(r2 > 0).mean()*100:.2f}%")
    print(f"   Average EV:   {r2.mean():.2f} R")
    print(f"   Total P&L:    {r2.sum():.2f} R\n")
    
    r3 = np.array(res_s3)
    print("3) Day 2 High Breakout | Stop: D2 Low | Exit: Full at Close < 50MA")
    print(f"   Total Trades: {len(r3)}")
    print(f"   Win Rate:     {(r3 > 0).mean()*100:.2f}%")
    print(f"   Average EV:   {r3.mean():.2f} R")
    print(f"   Total P&L:    {r3.sum():.2f} R\n")
    
    r4 = np.array(res_s4)
    print("4) Day 2 High Breakout | Stop: D2 Low | Exit: 1/3 at 1 ADR, 1/3 at 2 ADR, Runner at 50MA")
    print(f"   Total Trades: {len(r4)}")
    print(f"   Win Rate:     {(r4 > 0).mean()*100:.2f}%")
    print(f"   Average EV:   {r4.mean():.2f} R")
    print(f"   Total P&L:    {r4.sum():.2f} R\n")

if __name__ == "__main__":
    run_simulation()

