"""Multi-timeframe 'situational awareness' state per ticker.

`tf_state(frame)` distills one timeframe's enriched frame into where price sits vs the MAs, the MA
stack/tilt, ATR extension from the 50-SMA, and the EMA-rider + trend direction. `profile()` runs it
across `config.MTF_TFS`. Pure + reusable: serve.py computes it on demand per ticker, build_mtf.py
precomputes it for the universe. The queryable foundation for future market/sector/theme awareness
scores and MTF-conditioned setups (not built yet).
"""
from __future__ import annotations

import pandas as pd

import config
import datastore
import emarider
import trendlab
from indicators import add_indicators

_MAS = ["ema10", "ema20", "sma50", "sma150", "sma200"]   # fast -> slow


def _slope(s: pd.Series, k: int):
    """+1 rising / -1 falling / 0 unknown over k bars."""
    if len(s) <= k or pd.isna(s.iloc[-1]) or pd.isna(s.iloc[-1 - k]):
        return 0
    return 1 if s.iloc[-1] > s.iloc[-1 - k] else -1


def tf_state(d: pd.DataFrame) -> dict | None:
    """One timeframe's enriched frame -> awareness state dict (None if too short)."""
    if d is None or len(d) < 2:
        return None
    last = d.iloc[-1]
    c = float(last["close"])
    ma = {m: (float(last[m]) if (m in d.columns and pd.notna(last[m])) else None) for m in _MAS}
    atr = float(last["atr14"]) if pd.notna(last["atr14"]) else None
    vals = [ma[m] for m in _MAS]
    pairs = [(vals[i], vals[i + 1]) for i in range(len(vals) - 1)
             if vals[i] is not None and vals[i + 1] is not None]
    stack = sum(1 if a >= b else -1 for a, b in pairs)                    # -4..+4 (+4 = perfect bull)
    tilt = sum(_slope(d[m], config.MTF_SLOPE_LOOKBACK) for m in _MAS if m in d.columns)
    ext = lambda m: (round((c - ma[m]) / atr, 2) if (ma.get(m) and atr) else None)
    st = emarider.current_streak(d, config.ER_EMA_LEN, config.ER_ATR_LEN, config.ER_ATR_FRAC,
                                 config.ER_USE_STREAK_THRESH, config.ER_STREAK_ATR_FRAC,
                                 config.ER_USE_ARM_EXIT)
    # current structure segment: up / down / contracting / expanding / rectangle / range
    trend_dir, trend_net, trend_clarity = None, None, None
    if config.REGIME_USE:
        segs = trendlab.segment_chart(d)[0]
        if segs:
            cur = segs[-1].ensure()                   # lazy trendlab metrics
            if cur.kind in ("up", "down"):
                trend_dir, trend_net, trend_clarity = cur.kind, round(cur.net_pct, 1), round(cur.clarity, 2)
            else:                                     # range -> the contraction/expansion subtype
                trend_dir = cur.subtype or cur.kind
    return {
        "close": round(c, 2),
        "gt_ema10": (c > ma["ema10"]) if ma["ema10"] else None,
        "gt_ema20": (c > ma["ema20"]) if ma["ema20"] else None,
        "gt_sma50": (c > ma["sma50"]) if ma["sma50"] else None,
        "gt_sma150": (c > ma["sma150"]) if ma["sma150"] else None,
        "gt_sma200": (c > ma["sma200"]) if ma["sma200"] else None,
        "stack": stack, "tilt": tilt,
        "atr_ext_20": ext("ema20"), "atr_ext_50": ext("sma50"), "atr_ext_200": ext("sma200"),
        "rider_dir": (st["direction"] if st else None),
        "rider_streak": (st["length"] if st else None),
        "rider_touch": (bool(st["near_now"]) if st else None),
        "trend_dir": trend_dir, "trend_net": trend_net, "trend_clarity": trend_clarity,
        "rsi": (round(float(last["rsi14"]), 1)
                if "rsi14" in d.columns and pd.notna(last["rsi14"]) else None),
        "adx": (round(float(last["adx14"]), 1)
                if "adx14" in d.columns and pd.notna(last["adx14"]) else None),
        "adx_band": (_adx_band(float(last["adx14"]))
                     if "adx14" in d.columns and pd.notna(last["adx14"]) else None),
    }


def _adx_band(adx: float) -> str:
    """ADX trend-strength band: chop = range tactics only; exhaustion = late-trend warning."""
    return ("chop" if adx < 20 else "emerging" if adx < 25 else
            "trend" if adx <= 50 else "exhaustion")


def tf_frame(daily_df: pd.DataFrame, df15: pd.DataFrame | None, tf: str, is_future: bool = False):
    """Build + enrich the frame for timeframe `tf`. 1D/2D..6M resample the daily frame; 15m/1h/4h
    resample the 15m frame (futures-aware). Returns an enriched frame or None."""
    cols = ["open", "high", "low", "close", "volume"]
    if tf in datastore.INTRA_TFS or tf == "5m":       # 5m is the base itself (resample returns it as-is)
        if df15 is None or not len(df15):
            return None
        raw = datastore.resample_intraday(df15, tf, futures=is_future)
    elif tf == "1D":
        raw = None if daily_df is None else daily_df[cols]
    else:
        raw = None if daily_df is None else datastore.resample_daily(daily_df[cols], tf)
    if raw is None or len(raw) < 5:
        return None
    return add_indicators(raw)


def profile(daily_df, df15, tfs=None, is_future: bool = False) -> dict:
    """{tf: tf_state} across `tfs` (default config.MTF_TFS) for one ticker."""
    tfs = tfs or config.MTF_TFS
    return {tf: (tf_state(f) if (f := tf_frame(daily_df, df15, tf, is_future)) is not None else None)
            for tf in tfs}
