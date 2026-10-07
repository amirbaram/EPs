"""Ratio / relative-strength engine — a synthetic A÷B instrument that reuses the whole per-frame stack.

A ratio chart (RTY/NQ, NQ/ES, SMH/SPY, a sector-basket ÷ SPY …) is just an OHLC series built by dividing
two instruments bar-by-bar (the way TradingView does it). Once built we enrich it (`add_indicators`) and run
the SAME primitives every other frame gets — `mtf.tf_state` (trend/MA posture) and `sr.sr_levels` (support/
resistance) — so relative strength gets structure, levels, and (later) setups for free.

Used by `market._risk_block` (index-rotation / risk-on-off basket) and the group RS + narrative layers.
Pure / read-only: reuses datastore, indicators, mtf, sr. Does NOT import market (keeps the dep one-way).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import datastore
import mtf
import sr
import universe
from indicators import add_indicators

_OHLC = ["open", "high", "low", "close"]


def _ratio_ohlc(fa: pd.DataFrame, fb: pd.DataFrame) -> pd.DataFrame | None:
    """Bar-by-bar A÷B OHLC on the shared index. open/close = the component ratios; high/low = the max/min of
    the four corner ratios (so the candle is always valid). volume = the numerator's (an activity proxy)."""
    if fa is None or fb is None or not len(fa) or not len(fb):
        return None
    idx = fa.index.intersection(fb.index)
    if len(idx) < 40:
        return None
    A, B = fa.reindex(idx), fb.reindex(idx)
    corners = pd.concat([A[c] / B[c].replace(0, np.nan) for c in _OHLC], axis=1)
    corners = corners.replace([np.inf, -np.inf], np.nan)
    out = pd.DataFrame({
        "open": A["open"] / B["open"].replace(0, np.nan),
        "close": A["close"] / B["close"].replace(0, np.nan),
        "high": corners.max(axis=1),
        "low": corners.min(axis=1),
        "volume": A["volume"] if "volume" in A.columns else 0.0,
    }, index=idx).replace([np.inf, -np.inf], np.nan).dropna(subset=["open", "high", "low", "close"])
    return out if len(out) >= 40 else None


def ratio_frame(a: str, b: str, frames: dict, frames15: dict | None = None,
                live: bool = False, tf: str = "1D", as_of: str | None = None) -> pd.DataFrame | None:
    """Enriched synthetic A÷B frame. live=True builds from the 15m frames (intraday), else the daily frames
    (sliced to as_of). tf resamples the base ratio (2D..6M daily, or 15m..12h intraday)."""
    if live and frames15 is not None:
        raw = _ratio_ohlc(frames15.get(a), frames15.get(b))
        if raw is not None and tf is not None:            # resample_intraday no-ops when tf == the base
            fut = universe.is_future(a) or universe.is_future(b)   # TF, so this is base-agnostic (5m or 15m)
            raw = datastore.resample_intraday(raw, tf, futures=fut)
    else:
        fa, fb = frames.get(a), frames.get(b)
        if as_of is not None:
            fa = fa.loc[:as_of] if fa is not None else None
            fb = fb.loc[:as_of] if fb is not None else None
        raw = _ratio_ohlc(fa, fb)
        if raw is not None and tf not in (None, "1D"):
            raw = datastore.resample_daily(raw, tf)
    if raw is None or len(raw) < 40:
        return None
    return add_indicators(raw)


def basket_series(members: set[str], frames: dict, as_of: str | None = None,
                  min_members: int = 4) -> pd.DataFrame | None:
    """Equal-weight normalized member 'index' as an OHLC frame (each member rebased to 1.0 at its first shared
    bar, then averaged) — a synthetic ETF for groups without one. Reuses the loaded daily frames; no disk hit."""
    cols = []
    for sym in members:
        d = frames.get(sym)
        if d is None or not len(d):
            continue
        c = (d.loc[:as_of] if as_of is not None else d)["close"]
        if len(c) < 60:
            continue
        cols.append(c)
    if len(cols) < min_members:
        return None
    m = pd.concat(cols, axis=1)                          # align on dates (outer)
    m = m[m.notna().sum(axis=1) >= min_members]          # keep bars with enough members
    if len(m) < 60:
        return None
    norm = m.div(m.apply(lambda s: s.loc[s.first_valid_index()]))   # rebase each member to 1.0
    close = norm.mean(axis=1)                             # equal-weight basket level
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 0.0},
                        index=close.index)


def _net_pct(d: pd.DataFrame, bars: int = 21) -> float | None:
    c = d["close"].to_numpy(float)
    if len(c) < 2:
        return None
    c0 = c[-min(len(c), bars + 1)]
    return round((c[-1] / c0 - 1.0) * 100, 2) if c0 else None


def ratio_state(a: str, b: str, frames: dict, frames15: dict | None = None,
                live: bool = False, tf: str = "1D", as_of: str | None = None) -> dict | None:
    """The A÷B ratio's structural relative-strength read: {pair, tf_state posture, S/R levels, net_rs_pct
    over ~1mo}. Scoring into [-1,1] is the caller's job (market._score_state) to keep the dep one-way."""
    d = ratio_frame(a, b, frames, frames15, live, tf, as_of)
    if d is None:
        return None
    st = mtf.tf_state(d)
    if st is None:
        return None
    return {"pair": f"{a}/{b}", "state": st, "sr": sr.sr_levels(d), "net_rs_pct": _net_pct(d)}


def series_state(close_frame: pd.DataFrame, denom: str, frames: dict,
                 as_of: str | None = None) -> dict | None:
    """Relative-strength state of a prebuilt basket (from basket_series) vs a denominator symbol — the
    member-basket analogue of ratio_state, for the 144 fine-grained groups that have no ETF."""
    fb = frames.get(denom)
    if close_frame is None or fb is None:
        return None
    raw = _ratio_ohlc(close_frame, fb.loc[:as_of] if as_of is not None else fb)
    if raw is None:
        return None
    d = add_indicators(raw)
    st = mtf.tf_state(d)
    if st is None:
        return None
    return {"pair": f"basket/{denom}", "state": st, "sr": sr.sr_levels(d), "net_rs_pct": _net_pct(d)}
