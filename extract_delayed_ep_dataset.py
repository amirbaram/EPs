import pandas as pd
import numpy as np
import datastore
from ep_ml_engine import engine

df_all = pd.read_parquet("data/simulations/ep_combined_study_scored.parquet")

TOP_SECTORS = [
    "Computer Hardware", "Biotechnology", "Healthcare", "Technology", "Semiconductors",
    "Industrials", "Capital Markets", "Financial Services", "Specialty Industrial Machinery"
]
TOP_THEMES = [
    "Semiconductors", "Biotechnology", "Software - Infrastructure", "Computer Hardware",
    "AI / Machine Learning", "Cybersecurity", "Clean Energy", "Uranium & Nuclear"
]

pinnacle = df_all[
    (df_all["close_pos"] >= 0.65) &
    (df_all["rvol"] >= 2.5) &
    (df_all["gap_pct"] >= 5.0) &
    (df_all["sector"].isin(TOP_SECTORS) | df_all["theme"].isin(TOP_THEMES)) &
    ((df_all["sec_m1_pctile"] >= 50) | (df_all["thm_m1_pctile"] >= 50))
].copy()

print(f"Loaded {len(pinnacle)} Pinnacle Elite Setups.")

# We will collect delayed EP trade events across Pinnacle Elite
delayed_trades = []

for idx, row in pinnacle.iterrows():
    sym = row["symbol"]
    ev_dt = str(row["date"])[:10]
    
    bars = datastore.load_bars(sym)
    if bars is None or ev_dt not in bars.index:
        continue
    pos = bars.index.get_loc(ev_dt)
    if pos >= len(bars) - 10:
        continue
        
    d1 = bars.iloc[pos]
    d1_close = float(d1['close'])
    d1_low = float(d1['low'])
    d1_high = float(d1['high'])
    
    # Delayed breakout window: Days 2 to 5
    fwd_window = bars.iloc[pos + 1 : min(len(bars), pos + 6)]
    
    trigger_day = None
    shelf_low = d1_low
    cancelled = False
    
    for b_i in range(len(fwd_window)):
        b_low = float(fwd_window['low'].iloc[b_i])
        b_high = float(fwd_window['high'].iloc[b_i])
        b_open = float(fwd_window['open'].iloc[b_i])
        b_close = float(fwd_window['close'].iloc[b_i])
        
        if b_low < d1_low:
            cancelled = True
            break
            
        if b_low < shelf_low:
            shelf_low = b_low
            
        if b_high > d1_high:
            trigger_day = pos + 1 + b_i
            entry_px = max(b_open, d1_high)
            
            # Extract V2 rolling features on trigger bar
            feats = engine.compute_rolling_features(sym, pos, trigger_day)
            preds = engine.predict_rolling(feats) if feats else {}
            
            # Prior day low (before trigger bar)
            prior_low = float(bars['low'].iloc[trigger_day - 1])
            # 3-day swing low (prior 3 sessions up to trigger day)
            swing3_low = float(bars['low'].iloc[max(0, trigger_day - 3):trigger_day].min())
            
            # Forward hold analysis for MAE / Profit target
            fwd_trade = bars.iloc[trigger_day:].copy()
            fwd_trade['sma50'] = fwd_trade['close'].rolling(50).mean()
            
            # Max Gain over 60 days
            max_gain_pct = (fwd_trade['high'].iloc[:min(len(fwd_trade), 60)].max() / entry_px - 1.0) * 100.0
            
            # Check MAE before hitting peak or 50 SMA exit
            # Exit on 50 SMA breakdown
            exit_px = float(fwd_trade['close'].iloc[-1])
            exit_bar = len(fwd_trade) - 1
            seen_bull = False
            min_low_during_trade = entry_px
            
            for t_b in range(1, len(fwd_trade)):
                c = float(fwd_trade['close'].iloc[t_b])
                l = float(fwd_trade['low'].iloc[t_b])
                sma50 = fwd_trade['sma50'].iloc[t_b]
                
                if l < min_low_during_trade:
                    min_low_during_trade = l
                    
                if c > entry_px:
                    seen_bull = True
                    
                if seen_bull and t_b >= 5 and pd.notna(sma50) and c < sma50:
                    exit_px = c
                    exit_bar = t_b
                    break
                    
            mae_pct = (min_low_during_trade / entry_px - 1.0) * 100.0 # negative number
            trade_ret_pct = (exit_px / entry_px - 1.0) * 100.0
            is_winner = bool(trade_ret_pct > 0 and max_gain_pct >= 20.0)
            
            delayed_trades.append({
                'symbol': sym,
                'date': ev_dt,
                'year': int(ev_dt[:4]),
                'trigger_bar_offset': b_i + 1,
                'entry_px': entry_px,
                'd1_low': d1_low,
                'shelf_low': shelf_low,
                'prior_low': prior_low,
                'swing3_low': swing3_low,
                'exit_px': exit_px,
                'exit_bar': exit_bar,
                'trade_ret_pct': trade_ret_pct,
                'max_gain_pct': max_gain_pct,
                'mae_pct': mae_pct,
                'is_winner': is_winner,
                'prob_50': preds.get('prob_50', 0.0),
                'prob_100': preds.get('prob_100', 0.0),
                'prob_200': preds.get('prob_200', 0.0),
                'dyn_stop_pct': preds.get('dynamic_stop_loss_pct', -0.12),
                'feats': feats
            })
            break

df_dt = pd.DataFrame(delayed_trades)
print(f"Extracted {len(df_dt)} Delayed Breakout trades.")
print(f"Winners (>=20% max gain and positive return): {df_dt['is_winner'].sum()} / {len(df_dt)} ({df_dt['is_winner'].mean()*100:.1f}%)")

# Inspect MAE distribution of WINNERS
winners = df_dt[df_dt['is_winner'] == True]
print("\nMAE Distribution of WINNING Delayed EP Trades (Drawdown before profit):")
print(winners['mae_pct'].describe(percentiles=[0.05, 0.10, 0.20, 0.25, 0.50]))

# Save to disk for ML model training
df_dt.to_parquet("data/simulations/delayed_ep_trades.parquet")
print("Saved data/simulations/delayed_ep_trades.parquet")
