"""TRUE EXCHANGE BREADTH for the CLOSE banner internals (Amir 2026-07-06).

The daily/CLOSE breadth block (market._breadth_block) normally samples our ~2000-name active
universe. This module swaps in the REAL NYSE/NASDAQ internals from the manually-extracted
TradingView daily series so the banner shows exchange-official numbers (and the breadth SCORE
is computed from them), for any point-in-time `as_of`.

Sourced from `data/tv/bars_daily_*.parquet` + `data/trin_daily.csv` (manual TV MCP exports —
see DOCS.md §4b). Coverage today: NYSE/NASDAQ net advancers−decliners ($ADD/$ADDQ) and the
Arms index / TRIN ($TRIN, $TRINQ). NOT extractable without driving a live TV chart: %>MA
($MMTW/$MMFI/$MMTH) and new-high/low ($MAHN/$MALN) — those return None here and the caller
self-computes them from our own yfinance daily bars (which is also the TV-removed fallback).

REMOVING TV: set config.USE_TV_BREADTH=False (or delete data/tv/) → internals() returns None →
market._breadth_block reverts to the pure universe-sample computation. No hard dependency.
The LIVE intraday internals (intraday.py) never call this — TV has no realtime feed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

_TV = config.DATA_DIR / "tv"
_TRIN_CSV = config.DATA_DIR / "trin_daily.csv"
# TV symbol → local file, for the optionally-extracted %>MA and new-H/L series. Populate these by
# exporting the symbol from a TV chart to data/tv/bars_daily_<SYM>.parquet; they light up on load.
_PCT_MA = {"pct_20": "MMTW", "pct_50": "MMFI", "pct_200": "MMTH"}   # NYSE % above 20/50/200-day MA
_NHNL = {"nh": "MAHN", "nl": "MALN"}                                # NYSE new highs / new lows
_CACHE: dict = {}


def _daily(sym: str) -> pd.Series | None:
    """Cached 'close' series from data/tv/bars_daily_<sym>.parquet, or None if absent."""
    if sym in _CACHE:
        return _CACHE[sym]
    p = _TV / f"bars_daily_{sym}.parquet"
    s = None
    if p.exists():
        try:
            b = pd.read_parquet(p)
            s = b["close"].copy()
            s.index = pd.to_datetime(s.index)
        except Exception:
            s = None
    _CACHE[sym] = s
    return s


def _trin_nyse() -> pd.Series | None:
    if "trin_nyse" in _CACHE:
        return _CACHE["trin_nyse"]
    s = None
    if _TRIN_CSV.exists():
        try:
            t = pd.read_csv(_TRIN_CSV)
            s = pd.Series(t["close"].values, index=pd.to_datetime(t["date"]))
        except Exception:
            s = None
    _CACHE["trin_nyse"] = s
    return s


def _asof(s: pd.Series | None, as_of) -> float | None:
    """Last value at or before `as_of` (PIT, EOD close read), or None."""
    if s is None or not len(s):
        return None
    sub = s.loc[:as_of] if as_of else s
    if not len(sub):
        return None
    v = sub.iloc[-1]
    return float(v) if pd.notna(v) else None


def _asof_dated(s: pd.Series | None, as_of):
    """(value, source_date) at or before `as_of`, or (None, None)."""
    if s is None or not len(s):
        return None, None
    sub = s.loc[:as_of] if as_of else s
    if not len(sub):
        return None, None
    v = sub.iloc[-1]
    return (float(v) if pd.notna(v) else None), sub.index[-1]


def available() -> bool:
    """True if the toggle is on AND at least the core NYSE net A−D series is on disk."""
    return bool(config.USE_TV_BREADTH) and _daily("ADD") is not None


def internals(as_of: str | None = None) -> dict | None:
    """True exchange internals for `as_of`, or None when disabled / no data (→ caller falls back).
    Keys present only when their source series exists — the caller fills the rest from its sample."""
    if not available():
        return None
    add, src = _asof_dated(_daily("ADD"), as_of)
    if add is None:                                   # core series missing for this date → no true read
        return None
    ref = pd.Timestamp(as_of) if as_of else pd.Timestamp.utcnow().tz_localize(None).normalize()
    if src is not None and (ref - src).days > config.TV_BREADTH_MAX_LAG_DAYS:
        return None                                   # extract too stale for this view → fall back to sample
    addq = _asof(_daily("ADDQ"), as_of)
    out: dict = {"add": add, "addq": addq, "date": (src.date().isoformat() if src is not None else None)}
    out["trin"] = _asof(_trin_nyse(), as_of)
    out["trinq"] = _asof(_daily("TRINQ"), as_of)
    for key, sym in _PCT_MA.items():                  # %>MA (None until $MMTW/$MMFI/$MMTH extracted)
        out[key] = _asof(_daily(sym), as_of)
    for key, sym in _NHNL.items():                    # new H/L (None until $MAHN/$MALN extracted)
        out[key] = _asof(_daily(sym), as_of)
    # breadth SCORE term in [-1,1] from true net A−D (NYSE + NASDAQ half each; tanh-calibrated to ~1σ)
    t = 0.5 * float(np.tanh(add / config.TV_ADD_SCALE))
    if addq is not None:
        t += 0.5 * float(np.tanh(addq / config.TV_ADDQ_SCALE))
    else:
        t = float(np.tanh(add / config.TV_ADD_SCALE))
    out["ad_term"] = float(np.clip(t, -1, 1))
    return out


def clear_cache() -> None:
    _CACHE.clear()


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(internals(sys.argv[1] if len(sys.argv) > 1 else None), indent=1))
