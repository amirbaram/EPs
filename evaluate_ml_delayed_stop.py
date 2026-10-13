import pandas as pd
import numpy as np
import datastore

df_trades = pd.read_parquet("data/simulations/delayed_ep_trades.parquet")

def evaluate_stop_strategy(name, subset, stop_source='ml_stop_price', use_pyramid=False):
    results = []
    for idx, row in subset.iterrows():
        sym = row['symbol']
        ev_dt = row['date']
        bars = datastore.load_bars(sym)
        if bars is None or ev_dt not in bars.index:
            continue
        pos = bars.index.get_loc(ev_dt)
        trigger_day = pos + row['trigger_bar_offset']
        entry_px = row['entry_px']
        
        if stop_source == 'ml_stop_price':
            stop_px = row['ml_stop_price']
        elif stop_source == 'd1_low':
            stop_px = row['d1_low']
        elif stop_source == 'shelf_low':
            stop_px = row['shelf_low']
        elif stop_source == 'prior_low':
            stop_px = row['prior_low']
        else:
            stop_px = row['shelf_low']
            
        risk_pct = (entry_px - stop_px) / entry_px * 100.0
        if risk_pct <= 0.1:
            continue
            
        fwd = bars.iloc[trigger_day:].copy()
        fwd['sma50'] = fwd['close'].rolling(50).mean()
        
        exit_px = None
        exit_bar = len(fwd) - 1
        seen_bull = False
        stopped = False
        
        # Day 1 check (trigger bar)
        d0_low = float(fwd['low'].iloc[0])
        d0_open = float(fwd['open'].iloc[0])
        if d0_low <= stop_px:
            stopped = True
            exit_px = d0_open if d0_open < stop_px else stop_px
            exit_bar = 0
        else:
            for b in range(1, len(fwd)):
                c = float(fwd['close'].iloc[b])
                l = float(fwd['low'].iloc[b])
                o = float(fwd['open'].iloc[b])
                sma50 = fwd['sma50'].iloc[b]
                
                if c > entry_px:
                    seen_bull = True
                if l <= stop_px:
                    stopped = True
                    exit_px = o if o < stop_px else stop_px
                    exit_bar = b
                    break
                if seen_bull and b >= 5 and pd.notna(sma50) and c < sma50:
                    exit_px = c
                    exit_bar = b
                    break
                    
        if exit_px is None:
            exit_px = float(fwd['close'].iloc[-1])
            
        ret_pct = (exit_px / entry_px - 1.0) * 100.0
        r_mult = ret_pct / risk_pct
        
        # Pyramiding logic: If prob_50 >= 0.20 and held Day 5, add +50% size
        if use_pyramid and row['prob_50'] >= 0.20 and exit_bar > 4 and len(fwd) > 5:
            c5 = float(fwd['close'].iloc[min(4, len(fwd)-1)])
            risk_dollars = entry_px - stop_px
            if risk_dollars > 0 and c5 > entry_px:
                tranche2_r = 0.5 * (exit_px - c5) / risk_dollars
                r_mult = round(r_mult + tranche2_r, 2)
                
        results.append({
            'symbol': sym,
            'year': row['year'],
            'ret_pct': ret_pct,
            'r_mult': r_mult,
            'stopped': stopped,
            'risk_pct': risk_pct
        })
        
    tdf = pd.DataFrame(results)
    if len(tdf) == 0:
        return
    n = len(tdf)
    wr = (tdf['r_mult'] > 0).mean() * 100.0
    wins = tdf[tdf['r_mult'] > 0]['r_mult']
    losses = tdf[tdf['r_mult'] <= 0]['r_mult']
    avg_w = wins.mean() if len(wins) > 0 else 0.0
    avg_l = losses.mean() if len(losses) > 0 else 0.0
    ev = tdf['r_mult'].mean()
    net_r = tdf['r_mult'].sum()
    gw = wins.sum()
    gl = abs(losses.sum())
    pf = gw / gl if gl > 0 else 0.0
    avg_stop = tdf['risk_pct'].mean()
    
    print(f"{name:<50} | {n:<5} | {wr:5.1f}% | {avg_stop:5.1f}% | {avg_w:+6.2f} R | {avg_l:+6.2f} R | {ev:+6.2f} R | {net_r:+8.1f} R | {pf:4.2f}")

print("\n" + "="*112)
print(f"{'Strategy Configuration':<50} | {'N':<5} | {'WR %':<6} | {'Stop':<6} | {'Avg W':<8} | {'Avg L':<8} | {'EV (R)':<8} | {'Net R':<9} | {'PF':<5}")
print("="*112)

# 1. Day 1 Close Entry Baseline (from earlier)
print(f"{'0. Baseline Day 1 Close Entry (Stop: Day 1 Low)':<50} | {'985':<5} | {'35.3%':<6} | {'11.1%':<6} | {'+4.04 R':<8} | {'-1.01 R':<8} | {'+0.77 R':<8} | {'+763.0 R':<9} | {'2.19':<5}")

# 2. Delayed Breakout - D1 Low Stop
evaluate_stop_strategy("1. Delayed Breakout (Stop: Day 1 Low)", df_trades, 'd1_low', False)

# 3. Delayed Breakout - Shelf Low Stop
evaluate_stop_strategy("2. Delayed Breakout (Stop: Shelf Low)", df_trades, 'shelf_low', False)

# 4. Delayed Breakout - Prior Day Low Stop
evaluate_stop_strategy("3. Delayed Breakout (Stop: Prior Day Low)", df_trades, 'prior_low', False)

# 5. Delayed Breakout - ML Predicted Stoploss
evaluate_stop_strategy("4. Delayed Breakout (Stop: 🤖 ML Dynamic Stop)", df_trades, 'ml_stop_price', False)

# 6. Delayed Breakout + V2 Conviction (prob_50 >= 0.20) + Shelf Low Stop
evaluate_stop_strategy("5. Delayed + V2 Conviction (Stop: Shelf Low)", df_trades[df_trades['prob_50'] >= 0.20], 'shelf_low', False)

# 7. Delayed Breakout + V2 Conviction (prob_50 >= 0.20) + ML Dynamic Stop
evaluate_stop_strategy("6. Delayed + V2 Conviction (Stop: 🤖 ML Dynamic)", df_trades[df_trades['prob_50'] >= 0.20], 'ml_stop_price', False)

# 8. Delayed Breakout + V2 Conviction + ML Dynamic Stop + Pyramiding (+50%)
evaluate_stop_strategy("7. Delayed + V2 Conv + 🤖 ML Stop + Pyramiding", df_trades[df_trades['prob_50'] >= 0.20], 'ml_stop_price', True)

print("="*112)
