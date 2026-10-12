import datastore, universe
import pandas as pd
import numpy as np
import time
import trading_calendar

_mech_cache = {}
def is_mech_date(d_str):
    if d_str not in _mech_cache:
        _mech_cache[d_str] = trading_calendar.is_mechanical_volume_date(d_str)
    return _mech_cache[d_str]

syms = universe.active_symbols()
print(f"Scanning {len(syms)} universe symbols for historical DRE candidates (2015-2026)...")

start_t = time.time()
dre_events = []

for idx, sym in enumerate(syms):
    bars = datastore.load_bars(sym)
    if bars is None or len(bars) < 65:
        continue
        
    c = bars['close']
    o = bars['open']
    h = bars['high']
    l = bars['low']
    v = bars['volume']
    
    # 20-day average volume for RVOL
    vol_sma20 = v.rolling(20).mean().shift(1)
    rvol = v / vol_sma20
    
    # Dollar volume in Millions
    dvol_m = (c * v) / 1e6
    
    # Gap percentage: (open - prev_close) / prev_close * 100
    prev_close = c.shift(1)
    gap_pct = (o - prev_close) / prev_close * 100.0
    
    # Close Position: (close - low) / (high - low)
    rng = h - l
    close_pos = (c - l) / np.where(rng > 1e-4, rng, 1.0)
    
    # DRE conditions on Day 1:
    # 1. Catalyst energy: Gap >= 5.0% (on normal days, raw 9M volume also accepted;
    #    on mechanical volume anomaly dates like OpEx/Russell/Month-End, strictly requires gap >= 5.0%)
    date_strs = bars.index.astype(str).str[:10]
    is_mech_vol = np.array([is_mech_date(d) for d in date_strs])
    vol_or_gap = np.where(is_mech_vol, gap_pct >= 5.0, (gap_pct >= 5.0) | (v >= 9_000_000))

    # 2. RVOL >= 2.5x
    # 3. Dollar volume >= 15.0M
    # 4. Weak close: close_pos < 0.65
    mask = (
        vol_or_gap &
        (rvol >= 2.5) &
        (dvol_m >= 15.0) &
        (close_pos < 0.65) &
        (c >= 3.0) & # liquid price floor
        (bars.index >= '2015-01-01')
    )
    
    hit_indices = np.where(mask)[0]
    for hi in hit_indices:
        dt = bars.index[hi]
        dre_events.append({
            'symbol': sym,
            'date': str(dt)[:10],
            'bar_idx': hi,
            'gap_pct': round(float(gap_pct.iloc[hi]), 1),
            'rvol': round(float(rvol.iloc[hi]), 1),
            'dvol_m': round(float(dvol_m.iloc[hi]), 1),
            'close_pos': round(float(close_pos.iloc[hi]), 2),
            'd1_open': float(o.iloc[hi]),
            'd1_high': float(h.iloc[hi]),
            'd1_low': float(l.iloc[hi]),
            'd1_close': float(c.iloc[hi]),
            'd1_vol': float(v.iloc[hi])
        })

print(f"Scanned {len(syms)} symbols in {time.time() - start_t:.1f}s. Found {len(dre_events)} historical DRE setups.")
dre_df = pd.DataFrame(dre_events)
dre_df.to_parquet("data/simulations/historical_dre_setups.parquet")
print("Saved to data/simulations/historical_dre_setups.parquet")
