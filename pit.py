"""Point-in-time plumbing shared by serve.py and the awareness validator.

The one rule everything here enforces: at (date D, time T) a reader may only see data that had
CLOSED by T — the prior settled session for anything daily-derived, and only COMPLETED 15m bars
(a bar labeled 09:30 closes at 09:45). serve.py delegates its truncation here so the replay
slider and the offline validator share one implementation.
"""

from __future__ import annotations

import pandas as pd

import config
import datastore
import market
import universe as uni
from indicators import add_indicators


def session_frames15(store: dict, date: str, t: str | None) -> dict | None:
    """The intraday store truncated so `date`'s bars stop at clock time `t` (HH:MM ET) and later days
    are dropped — each frame's latest session becomes `date` (up to t). None if no bar reaches
    `date`. STRICT no-look-ahead: a bar labeled `o` only CLOSES at o+width (5m futures now, was 15m),
    so keep bars with open-time < cutoff (the open-time rule is base-agnostic).
    One searchsorted + iloc view per frame; no copies."""
    if not store:
        return None
    D = pd.Timestamp(date).date()
    hh, mm = (23, 59) if (not t or t == "live") else map(int, t.split(":"))
    cutoff_ts = pd.Timestamp(D) + pd.Timedelta(hours=hh, minutes=mm)
    out = {}
    for sym, f in store.items():
        pos = int(f.index.searchsorted(cutoff_ts, side="left"))   # bars strictly before the cutoff
        if pos > 2 and f.index[pos - 1].date() == D:              # last kept bar COMPLETED on `date`
            out[sym] = f.iloc[:pos]
    return out or None


def prior_session(as_of: str | None, frames: dict) -> str | None:
    """Last settled session strictly BEFORE as_of — the daily anchor for any intraday read."""
    return market._prior_session(as_of, frames)


def resolve(date: str, t: str | None, mode: str | None, frames: dict, store: dict) -> tuple:
    """(frames15, daily_asof) with serve.py's semantics, on an EXPLICIT store (validator/offline
    use; serve keeps its own lazy-loading wrapper). mode="close" or a past date with no clock
    time -> settled EOD read of `date` itself; a clock time -> truncated 15m + prior-settled
    daily anchor."""
    if mode == "close" or not (t and t != "live"):
        return None, date
    return session_frames15(store, date, t), prior_session(date, frames)


def assert_pit(frames15: dict | None, date: str, t: str | None) -> None:
    """Debug guard: raise if any frame contains a bar that had not CLOSED by (date, t)."""
    if not frames15 or not t or t == "live":
        return
    hh, mm = map(int, t.split(":"))
    cutoff = pd.Timestamp(pd.Timestamp(date).date()) + pd.Timedelta(hours=hh, minutes=mm)
    for sym, f in frames15.items():
        if not len(f):
            continue
        off = pd.Timedelta(minutes=datastore._infer_base_min(f.index))   # bar-close = open + its own width
        if f.index[-1] + off > cutoff:
            raise AssertionError(f"look-ahead: {sym} last bar {f.index[-1]} not closed by {date} {t}")


# ---------------------------------------------------------------- offline loaders (validator)

def load_frames(min_bars: int | None = None) -> dict:
    """Enriched daily frames for the active universe (+ curated ETFs + futures), like serve's
    RAM store but headless — for the offline validator. ~20s."""
    min_bars = config.MIN_BARS if min_bars is None else min_bars
    syms = datastore.list_symbols()
    active = uni.active_symbols()
    if active is not None:
        syms = [s for s in syms if s in active]
    frames = {}
    for sym in syms:
        df = datastore.load_bars(sym)
        if df is not None and len(df) >= min_bars:
            frames[sym] = add_indicators(df)
    return frames


def load_frames15(symbols) -> dict:
    """Bulk-load the 15m parquet store for `symbols` (offline validator; serve has FRAMES15)."""
    out = {}
    for sym in symbols:
        df = datastore.load_bars_15m(sym)
        if df is not None and len(df) > 2:
            out[sym] = df
    return out
