"""Vectorized per-ticker features. Everything downstream reads these columns."""
import numpy as np
import pandas as pd

import config


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c, h, l, v = d["close"], d["high"], d["low"], d["volume"]

    d["sma20"] = c.rolling(20).mean()
    d["sma50"] = c.rolling(50).mean()
    d["sma150"] = c.rolling(150).mean()
    d["sma200"] = c.rolling(200).mean()
    d["ema10"] = c.ewm(span=10, adjust=False).mean()
    d["ema20"] = c.ewm(span=20, adjust=False).mean()   # quality "surfing" check (with ema10)
    d["ema21"] = c.ewm(span=config.HL_MA, adjust=False).mean()  # higher_low_ma pullback EMA (span=HL_MA, now 20)

    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    d["atr14"] = tr.rolling(14).mean()

    d["adr_pct"] = ((h / l - 1.0).rolling(config.ADR_WINDOW).mean() * 100)
    d["vol_avg50"] = v.rolling(config.RVOL_WINDOW).mean()
    d["rvol"] = v / d["vol_avg50"]
    d["dollar_vol"] = (c * v).rolling(config.DOLLARVOL_WINDOW).mean()

    d["chg_pct"] = (c / prev_c - 1.0) * 100
    d["gap_pct"] = (d["open"] / prev_c - 1.0) * 100
    rng = (h - l).replace(0, np.nan)
    d["close_pos"] = (c - l) / rng  # 0 = low of day, 1 = high of day

    d["ret_1m"] = (c / c.shift(21) - 1.0) * 100    # 1/3/6-month returns -> cross-sectional RS rank
    d["ret_3m"] = (c / c.shift(63) - 1.0) * 100
    d["ret_6m"] = (c / c.shift(126) - 1.0) * 100

    d["hi52"] = h.rolling(252, min_periods=60).max()
    lo52 = l.rolling(252, min_periods=60).min()                 # intermediate only — not stored (no downstream reader)
    d["off_hi52_pct"] = (c / d["hi52"] - 1.0) * 100
    d["above_lo52_pct"] = (c / lo52 - 1.0) * 100

    # RSI-14 (Wilder) — backburner entry triggers + the market-environment RSI factor
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi14"] = 100 - 100 / (1 + rs)

    # ADX-14 + DI± (Wilder) — trend-strength bands (chop <20 / trend 25–50 / exhaustion >50)
    up_mv = h.diff()
    dn_mv = -l.diff()
    plus_dm = pd.Series(np.where((up_mv > dn_mv) & (up_mv > 0), up_mv, 0.0), index=d.index)
    minus_dm = pd.Series(np.where((dn_mv > up_mv) & (dn_mv > 0), dn_mv, 0.0), index=d.index)
    atr_w = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().replace(0, np.nan)
    d["di_plus"] = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr_w
    d["di_minus"] = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr_w
    dx = 100 * (d["di_plus"] - d["di_minus"]).abs() / (d["di_plus"] + d["di_minus"]).replace(0, np.nan)
    d["adx14"] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    # NOTE: pivots/swings now come from trendlab.swing_lows / swing_highs (the causal
    # Pine port — a single pivot source shared with the regime engine), not from a
    # centered fractal here. See setups.detect_higher_low_ma / detect_undercut_rally.

    # shrink RAM: every price/indicator column is float64 (8 bytes) by default; float32 (4 bytes)
    # halves per-frame memory with no visible precision loss for charting or setup comparisons.
    f64 = d.select_dtypes("float64").columns
    d[f64] = d[f64].astype("float32")
    
    import scanner_core
    d = scanner_core.calc_larssson_line(d)
    
    return d


def rsi_trigger_price(close: pd.Series, level: float, length: int = 14) -> pd.Series:
    """The price a bar must trade DOWN TO for its Wilder RSI to print `level` — Amir's Pine
    BB-Targets inversion: pOS = close[1] - (len-1)*(rmaU[1]*(100-level)/level - rmaD[1]).
    A LOW <= pOS is an intrabar oversold WICK even when the close's RSI never gets there —
    that's the backburner fire."""
    chg = close.astype(float).diff()
    rma_u = chg.clip(lower=0).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rma_d = (-chg).clip(lower=0).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    x = (100.0 - level) / level
    return (close.shift(1) - (length - 1) * (rma_u.shift(1) * x - rma_d.shift(1))).astype(float)
