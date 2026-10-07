"""Consolidation-shape recognition for continuation setups.

classify_consolidation() fits an upper trendline (through the highs) and a lower
trendline (through the lows) of a window and labels the shape from their slopes:

    rising_triangle : flat top  + rising bottom   (accumulation under resistance)
    pennant         : falling top + rising bottom (symmetrical contraction)
    flag            : flat/down parallel channel  (orderly pullback)
    channel         : rising parallel channel      (rising flag)
    other           : anything else

Slopes are normalized to ATR-per-bar so the tolerance (config.PATTERN_SLOPE_TOL)
is scale-free. Pivots from trendlab are used when >=2 fall in the window (cleaner
envelope); otherwise we regress the raw bar highs/lows, which is fine for the short
(3-20 bar) flags this is used on. Pure/causal: reads only the given window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
import trendlab


def _slope(idx: np.ndarray, price: np.ndarray) -> float:
    """Least-squares slope (price per bar) of a line through the points."""
    if len(idx) < 2:
        return 0.0
    return float(np.polyfit(idx.astype(float), price.astype(float), 1)[0])


def _envelope_slope(d: pd.DataFrame, start: int, end: int, kind: str) -> float:
    """Slope of the high (kind='H') or low (kind='L') trendline across [start,end]."""
    return envelope_line(d, start, end, kind)[0]


def envelope_line(d: pd.DataFrame, start: int, end: int, kind: str) -> tuple[float, float, float]:
    """(slope, intercept, r2) of the high/low trendline across [start,end] — intercept in
    bar-index coordinates so the line extrapolates to a live trigger:
    level(i) = slope * i + intercept. Prefers confirmed swing pivots in-window; falls back to
    regressing the raw bar series."""
    col = "high" if kind == "H" else "low"
    piv = (trendlab.swing_highs if kind == "H" else trendlab.swing_lows)(d, len(d) - start)
    pts = [(i, p) for i, p in piv if start <= i <= end]
    if len(pts) >= 2:
        ix = np.array([i for i, _ in pts], float); pr = np.array([p for _, p in pts], float)
    else:
        ix = np.arange(start, end + 1, dtype=float); pr = d[col].to_numpy(float)[start:end + 1]
    if len(ix) < 2 or np.all(pr == pr[0]):
        return 0.0, (float(pr[0]) if len(pr) else 0.0), 1.0
    slope, intercept = np.polyfit(ix, pr, 1)
    fit = slope * ix + intercept
    ss_tot = float(((pr - pr.mean()) ** 2).sum())
    r2 = 1.0 - float(((pr - fit) ** 2).sum()) / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(intercept), round(r2, 3)


def classify_consolidation(d: pd.DataFrame, start: int, end: int) -> dict:
    """Label the shape of d.iloc[start:end+1]. Returns
    {pattern, hi_slope, lo_slope, tightness} with slopes in ATR/bar and tightness
    = (window high - window low)/last close, %."""
    win = d.iloc[start:end + 1]
    last_close = float(win["close"].iloc[-1])
    hi, lo = float(win["high"].max()), float(win["low"].min())
    tightness = (hi - lo) / last_close * 100 if last_close > 0 else float("inf")
    if len(win) < 3:
        return {"pattern": "other", "hi_slope": 0.0, "lo_slope": 0.0, "tightness": round(tightness, 1)}

    atr = float(win["atr14"].mean()) if "atr14" in win else np.nan
    if not np.isfinite(atr) or atr <= 0:
        atr = max((hi - lo) / len(win), last_close * 0.005)
    hs = _envelope_slope(d, start, end, "H") / atr
    ls = _envelope_slope(d, start, end, "L") / atr

    tol = config.PATTERN_SLOPE_TOL
    flat_top, rising_top = abs(hs) <= tol, hs > tol
    rising_bottom, falling_top = ls > tol, hs < -tol
    flat_bottom, falling_bottom = abs(ls) <= tol, ls < -tol
    parallel = abs(hs - ls) <= tol

    if flat_top and rising_bottom:
        pat = "rising_triangle"
    elif falling_top and rising_bottom:
        pat = "pennant"
    elif falling_top and flat_bottom:
        pat = "falling_triangle"          # descending triangle (bear bias)
    elif rising_top and rising_bottom and ls > hs + tol:
        pat = "wedge_rising"              # converging up-sloped wedge (bearish reversal shape)
    elif falling_top and falling_bottom and hs < ls - tol:
        pat = "wedge_falling"             # converging down-sloped wedge (bullish reversal shape)
    elif parallel and hs <= tol:
        pat = "flag"              # flat or downward parallel channel
    elif parallel:
        pat = "channel"          # rising parallel channel (a rising flag)
    else:
        pat = "other"
    return {"pattern": pat, "hi_slope": round(hs, 3), "lo_slope": round(ls, 3),
            "tightness": round(tightness, 1)}
