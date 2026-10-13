import pandas as pd
import numpy as np
import datastore

df_raw = pd.read_parquet("data/simulations/historical_dre_setups.parquet")
print(f"Loaded {len(df_raw)} raw DRE candidates")

# Debounce: sort by symbol, date, drop any events within 15 bars of a prior event for same symbol
df_raw = df_raw.sort_values(by=['symbol', 'date']).reset_index(drop=True)
debounced = []
for sym, grp in df_raw.groupby('symbol'):
    last_idx = -999
    for _, r in grp.iterrows():
        b_idx = r['bar_idx']
        if b_idx - last_idx >= 15:
            debounced.append(r)
            last_idx = b_idx

df_dre = pd.DataFrame(debounced).reset_index(drop=True)
print(f"Debounced DRE candidates (15-session gap): {len(df_dre)}")

def calculate_larsson_line(df):
    close = df['close']
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    diff = ema12 - ema26
    dea = diff.ewm(span=9, adjust=False).mean()
    macd_hist = diff - dea

    state = []
    for i in range(len(df)):
        c = close.iloc[i]
        e12 = ema12.iloc[i]
        e26 = ema26.iloc[i]
        h = macd_hist.iloc[i]

        if c > e12 and e12 > e26 and h > 0:
            state.append('yellow')
        elif c < e12 and e12 < e26 and h < 0:
            state.append('blue')
        else:
            state.append('gray')
    df['larsson_state'] = state
    return df

for max_wait in [1, 3, 5, 10]:
    results = []
    cancelled_early = 0
    no_trigger = 0
    
    for idx, row in df_dre.iterrows():
        sym = row['symbol']
        ev_dt = row['date']
        
        bars = datastore.load_bars(sym)
        if bars is None or ev_dt not in bars.index:
            continue
            
        d1_pos = bars.index.get_loc(ev_dt)
        if d1_pos >= len(bars) - 5:
            continue
            
        d1_high = row['d1_high']
        d1_low = row['d1_low']
        
        # Look ahead up to max_wait bars
        fwd_window = bars.iloc[d1_pos + 1 : min(len(bars), d1_pos + 1 + max_wait)]
        
        entry_bar = None
        entry_price = None
        cons_low = d1_low
        
        cancelled = False
        for b_idx in range(len(fwd_window)):
            b_low = float(fwd_window['low'].iloc[b_idx])
            b_high = float(fwd_window['high'].iloc[b_idx])
            b_open = float(fwd_window['open'].iloc[b_idx])
            
            # Breached Day 1 Low before breakout -> DRE setup invalidated!
            if b_low < d1_low:
                cancelled = True
                break
                
            if b_low < cons_low:
                cons_low = b_low
                
            # Breakout trigger above Day 1 High
            if b_high > d1_high:
                entry_bar = d1_pos + 1 + b_idx
                entry_price = max(b_open, d1_high)
                break
                
        if cancelled:
            cancelled_early += 1
            continue
            
        if entry_bar is None:
            no_trigger += 1
            continue
            
        # Entry triggered! Stop is Day 1 Low (or tight shelf cons_low)
        risk = entry_price - d1_low
        if risk <= 0.01:
            continue
        risk_pct = (risk / entry_price) * 100.0
        
        # Also measure tight stop risk (consolidation low)
        tight_risk = entry_price - cons_low
        tight_risk_pct = (tight_risk / entry_price) * 100.0 if tight_risk > 0.01 else risk_pct
        
        # Simulate trade lifecycle
        trade_bars = bars.iloc[entry_bar : min(len(bars), entry_bar + 250)].copy()
        trade_bars = calculate_larsson_line(trade_bars)
        trade_bars['sma50'] = trade_bars['close'].rolling(50).mean()
        
        exit_bar = None
        exit_price = None
        seen_bull = False
        
        for t_b in range(1, len(trade_bars)):
            c = float(trade_bars['close'].iloc[t_b])
            l = float(trade_bars['low'].iloc[t_b])
            o = float(trade_bars['open'].iloc[t_b])
            s = trade_bars['larsson_state'].iloc[t_b]
            sma50 = trade_bars['sma50'].iloc[t_b]
            
            if pd.notna(s) and s in ['yellow', 'gray']:
                seen_bull = True
                
            # Stop loss hit (Day 1 Low)
            if l <= d1_low:
                exit_bar = t_b
                exit_price = o if o < d1_low else d1_low
                break
                
            # 50 SMA exit after Day 5
            if seen_bull and t_b >= 5 and pd.notna(sma50) and c < sma50:
                exit_bar = t_b
                exit_price = c
                break
                
        if exit_bar is None:
            exit_price = float(trade_bars['close'].iloc[-1])
            
        pnl_pct = (exit_price / entry_price - 1.0) * 100.0
        r_mult = pnl_pct / risk_pct
        r_tight = pnl_pct / tight_risk_pct
        
        results.append({
            'symbol': sym,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'risk_pct': risk_pct,
            'tight_risk_pct': tight_risk_pct,
            'r_mult': r_mult,
            'r_tight': r_tight,
            'pnl_pct': pnl_pct
        })
        
    tdf = pd.DataFrame(results)
    print(f"\n=========================================================")
    print(f"--- WEAK CLOSES: BONDE DRE STRATEGY (MAX WAIT: {max_wait} DAYS) ---")
    print(f"Total Debounced Candidates: {len(df_dre)}")
    print(f"Filtered / Cancelled (Breached D1 Low before trigger): {cancelled_early} (saved from loss!)")
    print(f"Never Broke Day 1 High within {max_wait} days: {no_trigger}")
    print(f"Triggered Trades: {len(tdf)} ({(len(tdf)/len(df_dre))*100:.1f}% participation rate)")
    
    if len(tdf) > 0:
        wr = (tdf['r_mult'] > 0).mean() * 100.0
        ev = tdf['r_mult'].mean()
        tot_pnl = tdf['r_mult'].sum()
        gw = tdf[tdf['r_mult'] > 0]['r_mult'].sum()
        gl = abs(tdf[tdf['r_mult'] <= 0]['r_mult'].sum())
        pf = gw / gl if gl > 0 else 0
        avg_w = tdf[tdf['r_mult'] > 0]['r_mult'].mean() if (tdf['r_mult'] > 0).any() else 0
        avg_l = tdf[tdf['r_mult'] <= 0]['r_mult'].mean() if (tdf['r_mult'] <= 0).any() else 0
        
        print(f"DRE BASELINE STOP (D1 Low): Win Rate: {wr:.1f}% | EV: +{ev:.2f} R | Total P&L: +{tot_pnl:.1f} R | PF: {pf:.2f}")
        print(f"Avg Win: +{avg_w:.2f} R | Avg Loss: {avg_l:.2f} R | Avg Stop Distance: {tdf['risk_pct'].mean():.1f}%")
        
        # Tight shelf stop stats
        ev_tight = tdf['r_tight'].mean()
        print(f"DRE TIGHT SHELF STOP: EV: +{ev_tight:.2f} R | Avg Stop Distance: {tdf['tight_risk_pct'].mean():.1f}%")
