import numpy as np
import pandas as pd
from scipy.stats import linregress
from openbb import obb

def load_and_prep(symbol):
    """Loads historical price data from openbb, reindexes to fill missing dates, and formats columns."""
    data = obb.equity.price.historical(symbol, start_date="2015-07-01", provider="yfinance")
    df = data if isinstance(data, pd.DataFrame) else data.to_df()
    calendar_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    df = df.reindex(calendar_dates).interpolate(method="linear")
    df = df.reset_index().rename(columns={'index': 'time'})
    df.columns = [c.lower() for c in df.columns]
    df['time'] = pd.to_datetime(df['time']).astype('datetime64[ns]')
    return df

def _pine_ema(series, length):
    """Calculates an EMA with TradingView's SMA seed, Vectorized."""
    import pandas as pd
    import numpy as np
    
    source = pd.to_numeric(series, errors='coerce').astype(float)
    if len(source) < length:
        return pd.Series(np.nan, index=source.index, dtype=float)
        
    sma_val = source.iloc[:length].mean()
    
    seed_series = pd.Series([sma_val], index=[source.index[length - 1]])
    rest = source.iloc[length:]
    combined = pd.concat([seed_series, rest])
    
    ema_vals = combined.ewm(span=length, adjust=False).mean()
    
    result = pd.Series(np.nan, index=source.index, dtype=float)
    result.loc[ema_vals.index] = ema_vals
    return result

def calc_larssson_line(df):
    """Adds the four Larsson EMAs plus its yellow, blue, or gray state."""
    result = df.copy()
    result['larsson_ema32'] = _pine_ema(result['close'], 32)
    result['larsson_ema35'] = _pine_ema(result['close'], 35)
    result['larsson_ema50'] = _pine_ema(result['close'], 50)
    result['larsson_ema58'] = _pine_ema(result['close'], 58)

    bullish = (
        (result['larsson_ema32'] >= result['larsson_ema35'])
        & (result['larsson_ema35'] >= result['larsson_ema50'])
        & (result['larsson_ema50'] >= result['larsson_ema58'])
    )
    bearish = (
        (result['larsson_ema32'] < result['larsson_ema35'])
        & (result['larsson_ema35'] < result['larsson_ema50'])
        & (result['larsson_ema50'] < result['larsson_ema58'])
    )
    valid = result[['larsson_ema32', 'larsson_ema35', 'larsson_ema50', 'larsson_ema58']].notna().all(axis=1)

    result['larsson_state'] = pd.Series(None, index=result.index, dtype='object')
    result.loc[valid, 'larsson_state'] = 'gray'
    result.loc[valid & bullish, 'larsson_state'] = 'yellow'
    result.loc[valid & bearish, 'larsson_state'] = 'blue'

    result['larsson_color'] = pd.Series(None, index=result.index, dtype='object')
    result.loc[result['larsson_state'] == 'gray', 'larsson_color'] = 'rgba(158, 158, 158, 1.0)'
    result.loc[result['larsson_state'] == 'yellow', 'larsson_color'] = 'rgba(255, 193, 7, 1.0)'
    result.loc[result['larsson_state'] == 'blue', 'larsson_color'] = 'rgba(33, 150, 243, 1.0)'
    return result

def detectLarssonFlip(df):
    """Returns dates flipping into blue and dates flipping into yellow."""
    result = df.copy() if 'larsson_state' in df.columns else calc_larssson_line(df)
    state = result['larsson_state']
    previous = state.shift(1)
    flipped = state.notna() & previous.notna() & state.fillna('').ne(previous.fillna(''))
    dates = pd.to_datetime(result['time'])
    blue_dates = dates[flipped & result['larsson_state'].eq('blue')].tolist()
    yellow_dates = dates[flipped & result['larsson_state'].eq('yellow')].tolist()
    return blue_dates, yellow_dates

def calc_intermediate_ribbon(df, spans=(8, 12, 16, 21)):
    """Calculates a 4-EMA intermediate continuation ribbon (default: 8, 12, 16, 21).
    Used for Trade 2 & Trade 3 continuation detection, triggers, and reverse flip exits.
    Also supports smooth configuration (10, 15, 20, 30).
    """
    result = df.copy()
    s1, s2, s3, s4 = spans
    col1 = f'ribbon_ema{s1}'
    col2 = f'ribbon_ema{s2}'
    col3 = f'ribbon_ema{s3}'
    col4 = f'ribbon_ema{s4}'

    result[col1] = _pine_ema(result['close'], s1)
    result[col2] = _pine_ema(result['close'], s2)
    result[col3] = _pine_ema(result['close'], s3)
    result[col4] = _pine_ema(result['close'], s4)

    bullish = (
        (result[col1] >= result[col2])
        & (result[col2] >= result[col3])
        & (result[col3] >= result[col4])
    )
    bearish = (
        (result[col1] < result[col2])
        & (result[col2] < result[col3])
        & (result[col3] < result[col4])
    )
    valid = result[[col1, col2, col3, col4]].notna().all(axis=1)

    result['ribbon_state'] = pd.Series(None, index=result.index, dtype='object')
    result.loc[valid, 'ribbon_state'] = 'gray'
    result.loc[valid & bullish, 'ribbon_state'] = 'yellow'
    result.loc[valid & bearish, 'ribbon_state'] = 'blue'

    result['ribbon_color'] = pd.Series(None, index=result.index, dtype='object')
    result.loc[result['ribbon_state'] == 'gray', 'ribbon_color'] = 'rgba(158, 158, 158, 1.0)'
    result.loc[result['ribbon_state'] == 'yellow', 'ribbon_color'] = 'rgba(255, 193, 7, 1.0)'
    result.loc[result['ribbon_state'] == 'blue', 'ribbon_color'] = 'rgba(33, 150, 243, 1.0)'

    emas_sub = result[[col1, col2, col3, col4]]
    result['ribbon_compression'] = emas_sub.std(axis=1) / emas_sub.mean(axis=1)
    result['ribbon_bandwidth'] = (result[col1] - result[col4]) / result[col4]

    state = result['ribbon_state']
    prev = state.shift(1)
    result['ribbon_flip'] = 'none'
    result.loc[(state == 'yellow') & (prev != 'yellow'), 'ribbon_flip'] = 'yellow_flip'
    result.loc[(state == 'blue') & (prev != 'blue'), 'ribbon_flip'] = 'blue_flip'
    return result

def detect_intermediate_ribbon_flip(df, spans=(8, 12, 16, 21)):
    """Returns dates flipping into blue and dates flipping into yellow for the intermediate ribbon."""
    result = df.copy() if 'ribbon_state' in df.columns else calc_intermediate_ribbon(df, spans=spans)
    state = result['ribbon_state']
    previous = state.shift(1)
    flipped = state.notna() & previous.notna() & state.fillna('').ne(previous.fillna(''))
    time_col = 'time' if 'time' in result.columns else ('date' if 'date' in result.columns else None)
    if time_col is not None:
        dates = pd.to_datetime(result[time_col])
    else:
        dates = pd.to_datetime(result.index)
    blue_dates = dates[flipped & result['ribbon_state'].eq('blue')].tolist()
    yellow_dates = dates[flipped & result['ribbon_state'].eq('yellow')].tolist()
    return blue_dates, yellow_dates

import numpy as np

def get_normalized_angle(src_series, lookback):
    x = np.arange(lookback)
    raw_slope = src_series.rolling(lookback).apply(lambda y: np.polyfit(x, y, 1)[0], raw=True)
    start_val = src_series.shift(lookback - 1)
    norm_slope = np.where(start_val == 0, 0, raw_slope / start_val)
    adjusted_slope = norm_slope * 100
    angles = np.degrees(np.arctan(adjusted_slope))
    return pd.Series(angles, index=src_series.index).fillna(0.0)

def add_emas_and_colors(df):
    """Calculates EMA10/20 and dynamically colors the line Red/Green based on 1-bar ATAS logic and New Angle logic."""
    df = df.copy()
    if 'index' in df.columns: df = df.rename(columns={'index': 'time'})
    if 'date' in df.columns: df = df.rename(columns={'date': 'time'})
    df['time'] = pd.to_datetime(df['time']).astype('datetime64[ns]')
    
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    slope10 = df['ema10'] - df['ema10'].shift(1)
    slope20 = df['ema20'] - df['ema20'].shift(1)
    
    is_up = (slope10 > 0) & (slope20 > 0) & (df['ema10'] > df['ema20'])
    is_down = (slope10 < 0) & (slope20 < 0) & (df['ema10'] < df['ema20'])
    
    df['color'] = 'rgba(158, 158, 158, 1.0)'  # Neutral Gray
    df.loc[is_up, 'color'] = 'rgba(76, 175, 80, 1.0)'   # Bullish Green
    df.loc[is_down, 'color'] = 'rgba(244, 67, 54, 1.0)' # Bearish Red
    
    # New logic (Trend Angle)
    angle1 = get_normalized_angle(df['ema10'], 15)
    angle2 = get_normalized_angle(df['ema20'], 15)
    
    is_up_new = (angle1 > 0) & (angle2 > 0) & (df['ema10'] > df['ema20'])
    is_down_new = (angle1 < 0) & (angle2 < 0) & (df['ema10'] < df['ema20'])
    
    df['color_new'] = 'rgba(158, 158, 158, 1.0)'
    df.loc[is_up_new, 'color_new'] = 'rgba(76, 175, 80, 1.0)'
    df.loc[is_down_new, 'color_new'] = 'rgba(244, 67, 54, 1.0)'
    return df

def find_pole_highs(df, highs, lows, minRet=0.5, minR2=0.0):
    """Finds valid pole structures given the swing highs and lows, filtering by retracement and linear regression."""
    valid_phs = []
    for ph_idx, ph_price in highs:
        post_window = df.iloc[ph_idx + 1 : ph_idx + 8]
        if len(post_window) == 0: continue
        if post_window['high'].max() > ph_price: continue
            
        lowest_post_PH = post_window['low'].min()
        valid_lls = []
        lookback_limit = max(0, ph_idx - 150)
        
        for ll_idx, ll_price in lows:
            if ll_idx < lookback_limit or ll_idx >= ph_idx: continue
                
            intermediate_highs = [p for i, p in highs if ll_idx < i < ph_idx]
            if intermediate_highs and max(intermediate_highs) > ph_price: continue
                
            move_pct = (ph_price - ll_price) / ll_price
            if move_pct < 0.30: continue
                
            pole_height = ph_price - ll_price
            pullback_depth = ph_price - lowest_post_PH
            if pole_height <= 0: continue
                
            retracement = pullback_depth / pole_height
            if retracement >= minRet: continue
                
            y = df['close'].iloc[ll_idx : ph_idx + 1].values
            x = np.arange(len(y))
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            r_squared = r_value ** 2
            
            if r_squared < minR2: continue
                
            valid_lls.append({
                'll_idx': ll_idx, 'll_date': df['time'].iloc[ll_idx], 'll_price': ll_price,
                'move_pct': move_pct * 100, 'retracement': retracement * 100,
                'r_squared': r_squared, 'slope': slope
            })
            
        if valid_lls:
            valid_phs.append({'ph_idx': ph_idx, 'ph_date': df['time'].iloc[ph_idx], 'ph_price': ph_price, 'll_candidates': valid_lls})
            
    return valid_phs

def analyze_breakouts_for_poles(df, qqq_df, ratio_df, ph_data, highs, lows, req_qqq, req_ratio, bo_type='high_pivot', minContract=0.5, req_ext_under_5=False, vol_mult=0.95, sym=None, sl_type='prev_low'):
    """Analyzes breakouts from valid poles, simulates trades, and applies regime filters."""
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['vol_sma20'] = df['volume'].rolling(20).mean()
    df['range'] = df['high'] - df['low']
    df['adr_14_abs'] = df['range'].rolling(14).mean()
    df['adr_14_pct'] = (df['range'] / df['close']).rolling(14).mean() * 100
    df['ext_ema50'] = (df['close'] - df['ema50']) / df['adr_14_abs']
    ph_dict = dict(highs)
    
    def simulate(bo_idx, bo_price, adr_pct, adr_abs):
        
        if sl_type == '0.5ATR':
            sl_value = bo_price - (0.5 * adr_abs)
        elif sl_type == '1ATR':
            sl_value = bo_price - (1.0 * adr_abs)
        else:
            sl_value = df['low'].iloc[bo_idx-1] * 1
            
        risk = bo_price - sl_value
        if risk <= 0: risk = bo_price * 0.001 
        
        if adr_pct < 5: trailing_ema_col = 'ema50'
        elif adr_pct < 10: trailing_ema_col = 'ema20'
        else: trailing_ema_col = 'ema10'
        
        position = 1.0
        realized_r = 0.0
        tp1_price = bo_price + (3 * adr_abs)
        tp1_hit = False
        trailing_active = False
        current_sl = sl_value
        next_climax = 7
        exit_idx = len(df) - 1
        
        # --- Day 1 Stop Check using 5m data ---
        if df['low'].iloc[bo_idx] < current_sl and sym:
            import config
            from pathlib import Path
            import pandas as pd
            p5 = config.DATA_DIR / "tiingo" / "bars_5min" / f"{sym}.parquet"
            if p5.exists():
                try:
                    df5 = pd.read_parquet(p5)
                except Exception:
                    print(f"  WARNING: Missing pyarrow engine or corrupted file. Run with .venv/bin/python!")
                    df5 = pd.DataFrame()
                    
                if not df5.empty:
                    # Ensure UTC and localize
                    if 'time' in df5.columns:
                        df5['time'] = pd.to_datetime(df5['time'], utc=True)
                        df5 = df5.set_index('time')
                    else:
                        df5.index = pd.to_datetime(df5.index, utc=True)
                    
                    bo_date = df['time'].iloc[bo_idx]
                    day5 = df5[df5.index.date == bo_date.date()].copy()
                else:
                    day5 = pd.DataFrame()
                    
                if not day5.empty:
                    # Sanitize 5m
                    day5['low'] = day5[['open', 'high', 'low', 'close']].min(axis=1)
                    day5['high'] = day5[['open', 'high', 'low', 'close']].max(axis=1)
                    
                    entry_hit = False
                    for k in range(len(day5)):
                        if not entry_hit and day5['high'].iloc[k] >= bo_price:
                            entry_hit = True
                        if entry_hit and day5['low'].iloc[k] < current_sl:
                            realized_r += position * ((current_sl - bo_price) / risk)
                            position = 0
                            exit_idx = bo_idx
                            break
                            
            else:
                # If no 5m data, assume stopped out (pessimistic)
                realized_r += position * ((current_sl - bo_price) / risk)
                position = 0
                exit_idx = bo_idx
                
        if position == 0:
            return {
                'sl_value': sl_value, 
                'trailing_ema': trailing_ema_col, 
                'rr_final': realized_r,
                'tp1_hit': tp1_hit,
                'climax_level': 6,
                'exit_idx': exit_idx
            }

        # --- Remaining Days ---
        for j in range(bo_idx + 1, len(df)):
            low_j, high_j, close_j = df['low'].iloc[j], df['high'].iloc[j], df['close'].iloc[j]
            ema_j = df[trailing_ema_col].iloc[j]
            ema_prev = df[trailing_ema_col].iloc[j-1] if j > 0 else ema_j
            ema50_j = df['ema50'].iloc[j]
            
            while True:
                climax_price = ema50_j + (next_climax * adr_abs)
                if high_j >= climax_price:
                    sold = position * 0.20
                    position -= sold
                    realized_r += sold * ((climax_price - bo_price) / risk)
                    next_climax += 1
                    if position < 0.001: 
                        exit_idx = j
                        break
                else:
                    break
                    
            if position < 0.001: 
                exit_idx = j
                break
            
            if low_j < current_sl:
                realized_r += position * ((current_sl - bo_price) / risk)
                position = 0
                exit_idx = j
                break
                
            if not tp1_hit and high_j >= tp1_price:
                tp1_hit = True
                sold = position * 0.50
                position -= sold
                realized_r += sold * ((tp1_price - bo_price) / risk)
                current_sl = bo_price # SL to breakeven
                
            if tp1_hit:
                if not trailing_active:
                    if ema_j > ema_prev and low_j > ema_j:
                        trailing_active = True
                
                if trailing_active:
                    if close_j < ema_j:
                        realized_r += position * ((close_j - bo_price) / risk)
                        position = 0
                        exit_idx = j
                        break

        if position > 0:
            realized_r += position * ((df['close'].iloc[-1] - bo_price) / risk)
            
        return {
            'sl_value': sl_value, 
            'trailing_ema': trailing_ema_col, 
            'rr_final': realized_r,
            'tp1_hit': tp1_hit,
            'climax_level': next_climax - 1,
            'exit_idx': exit_idx
        }

        
    for n, ph in enumerate(ph_data):
        ph_idx = ph['ph_idx']; ph_price = ph['ph_price']
        best_ll = max(ph['ll_candidates'], key=lambda x: x['move_pct'])
        ll_price = best_ll['ll_price']; pole_height = ph_price - ll_price
        
        minRet = 0.5; lowest_cons_price = ph_price; stack = [ph_price]; breakouts = []
        next_ph_idx = ph_data[n+1]['ph_idx'] if n + 1 < len(ph_data) else len(df)
        end_idx = min(len(df), ph_idx + 57, next_ph_idx)
        
        active_trade_exit_idx = -1
        
        for i in range(ph_idx + 1, end_idx):
            lowest_cons_price = min(lowest_cons_price, df['low'].iloc[i])
            retracement = (ph_price - lowest_cons_price) / pole_height
            if retracement > minRet: break 
                
            if i in ph_dict:
                stack.append(ph_dict[i])
                
            if i <= active_trade_exit_idx:
                continue
                    
            vol_ok = False
            if pd.notna(df['vol_sma20'].iloc[i]) and df['vol_sma20'].iloc[i] > 0:
                vol_ok = df['volume'].iloc[i] > df['vol_sma20'].iloc[i] * vol_mult
             
             
            if (i - ph_idx) < 10:
                vol_ok = False
                   
            if vol_ok:
                popped_level = None
                
                if bo_type == 'high_pivot':
                    popped_list = []
                    while len(stack) > 0 and df['high'].iloc[i] >= stack[-1] + 0.001 :
                        popped_list.append(stack.pop())
                    if popped_list:
                        popped_level = max(popped_list)
                elif bo_type == 'contraction':
                    if i >= 3:
                        r2, a2 = df['range'].iloc[i-2], df['adr_14_abs'].iloc[i-2]
                        r1, a1 = df['range'].iloc[i-1], df['adr_14_abs'].iloc[i-1]
                        if pd.notna(a2) and pd.notna(a1) and a2 > 0 and a1 > 0:
                            if (r2 < a2 * minContract) and (r1 < a1 * minContract):
                                day1_high = df['high'].iloc[i-2]
                                day2_high = df['high'].iloc[i-1]
                                if day2_high <= day1_high + (0.2 * a2):
                                    if df['high'].iloc[i] > day2_high:
                                        popped_level = day2_high
                elif bo_type == 'eq':
                    past_highs = [p for p in highs if p[0] < i]
                    past_lows = [p for p in lows if p[0] < i]
                    if len(past_highs) >= 2 and len(past_lows) >= 2:
                        ph1 = past_highs[-2][1]
                        ph2_idx, ph2 = past_highs[-1]
                        pl1 = past_lows[-2][1]
                        pl2 = past_lows[-1][1]
                        if ph2 < ph1 and pl2 > pl1:
                            if i > ph2_idx:
                                highest_since = df['high'].iloc[ph2_idx + 1 : i].max() if i > ph2_idx + 1 else 0
                                if highest_since <= ph2:
                                    if df['high'].iloc[i] > ph2:
                                        popped_level = ph2
                    
                if popped_level is not None:
                    bo_date = df['time'].iloc[i]
                    
                    if req_ext_under_5 and df['ext_ema50'].iloc[i] >= 5: continue
                    
                    if req_qqq and req_qqq != 'QQQ: OFF' and qqq_df is not None:
                        q_row = qqq_df[qqq_df['time'] == bo_date]
                        c_col = 'color_new' if 'New' in req_qqq else 'color'
                        if q_row.empty or q_row[c_col].iloc[0] != 'rgba(76, 175, 80, 1.0)': continue
                            
                    if req_ratio and req_ratio != 'QQQ/SPY: OFF' and ratio_df is not None:
                        r_row = ratio_df[ratio_df['time'] == bo_date]
                        c_col = 'color_new' if 'New' in req_ratio else 'color'
                        if r_row.empty or r_row[c_col].iloc[0] != 'rgba(76, 175, 80, 1.0)': continue
                            
                    bo_price = max(popped_level, df['open'].iloc[i])
                    
                    prev_i = i - 1
                    bo_adr_pct = df['adr_14_pct'].iloc[i]
                    bo_adr_abs = df['adr_14_abs'].iloc[i]
                    ext_ema50 = df['ext_ema50'].iloc[i]
                    
                    prev_range = df['range'].iloc[prev_i]
                    prev_adr = df['adr_14_abs'].iloc[prev_i]
                    prev_range_adr_pct = (prev_range / prev_adr * 100) if pd.notna(prev_adr) and prev_adr > 0 else np.nan
                    
                    sim_result = simulate(i, bo_price, bo_adr_pct, bo_adr_abs)
                    active_trade_exit_idx = sim_result['exit_idx']
                    
                    breakouts.append({
                        'bo_idx': i, 'bo_date': df['time'].iloc[i], 'bo_price': bo_price, 'popped_level': popped_level,
                        'sim': sim_result,
                        'pole_start_date': best_ll['ll_date'], 'pole_end_date': ph['ph_date'],
                        'pole_move_pct': best_ll['move_pct'], 'cons_days': (df['time'].iloc[i] - ph['ph_date']).days, 'retracement': retracement * 100,
                        'adr_pct': bo_adr_pct, 'prev_range_adr_pct': prev_range_adr_pct, 'ext_ema50': ext_ema50
                    })
        ph['breakouts'] = breakouts

def best_larsson_sectors() -> list[str]:
    """Returns the list of sectors proven to have the highest historical expectancy for Larsson Line flips."""
    return [
        "Diagnostics & Research",
        "Medical Instruments & Supplies",
        "Computer Hardware",
        "Software - Application",
        "Utilities - Independent Power Producers",
        "Internet Retail",
        "Discount Stores",
        "Residential Construction",
        "Oil & Gas E&P",
        "Other Industrial Metals & Mining",
        "Oil & Gas Midstream",
        "Grocery Stores",
        "Gambling",
        "Leisure",
        "Software - Infrastructure",
        "Internet Content & Information",
        "Waste Management"
    ]

def evaluate_market_regime_for_flip(date, qqq_df, igv_df) -> dict:
    """
    Analyzes index features on a given date to predict if a flip is statistically favorable.
    Based on backtest data:
    1. QQQ > 200 SMA significantly increases avg_R (0.55 vs 0.34)
    2. IGV > 200 SMA massively increases avg_R (0.63 vs 0.14)
    
    Expects qqq_df and igv_df to have 'close' and 'sma200' columns, and date index.
    """
    if date not in qqq_df.index or date not in igv_df.index:
        return {"favorable": False, "reason": "Missing index data"}
        
    qqq_above_200 = qqq_df.loc[date, 'close'] > qqq_df.loc[date, 'sma200']
    igv_above_200 = igv_df.loc[date, 'close'] > igv_df.loc[date, 'sma200']
    
    score = 0
    if qqq_above_200: score += 1
    if igv_above_200: score += 1
    
    favorable = score >= 2
    reason = f"QQQ>200:{qqq_above_200}, IGV>200:{igv_above_200}"
    
    return {
        "favorable": favorable,
        "score": score,
        "reason": reason
    }
