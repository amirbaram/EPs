"""dcs_data — data adapter for the DCStruct-Py lab (plan: docs/plan-dcstruct-python-port.md, P0).

LAB-ONLY module family (dcs_*): nothing here is imported by the app. One entry point, `frame()`,
returns a TFFrame: the OHLCV frame for (symbol, timeframe) plus — when available — the aligned 5-minute
CHILD stream with per-parent-bar index windows. The children are what realize the child-feeds-parent
design from the Pine work (2026-07-14): for any TF >= 15m the 5m bars ARE the intrabar data, giving the
engine true intrabar ordering and true within-leg paths instead of Pine's parent-bar approximations.

Data routing (grounded in datastore.py, 2026-07-14):
- equities intraday: datastore.load_bars_5m (Tiingo IEX, RTH 09:30-15:55, 2017+, calibrated volume)
  -> datastore.resample_intraday for 15m/30m/1h/2h/4h (session-anchored) and 8h/12h (continuous).
- futures intraday: datastore.load_bars_15m (LEGACY name — the store holds 5m futures bars, ~60 days,
  24h sessions) -> resample_intraday(futures=True) (continuous grouping for every TF).
- 1D: datastore.load_bars (yfinance, deep history, includes =F futures). 1W: datastore.load_bars_w.
  1M (and 2D/3D/...): datastore.resample_daily over the daily frame.
  Parents come from the daily/weekly stores, NOT resampled from 2017+ 5m — deeper history on purpose
  (a designed improvement over Pine's request.security chart-span limit, proven live 2026-07-14).
- 1m (optional, spot studies only per plan defaults): the ad-hoc data/tiingo/bars_1min/ store.
- research CSVs: data/research_1y/{SYM}_{tf}.csv.gz (e.g. NQF_5m) via load_research() — used by the
  parity harness where a fixed 1-year futures dataset is handy.

Child windows: children of parent bar i are base.iloc[child_lo[i]:child_hi[i]]. For intraday parents
this is exact by construction (resample_intraday labels each parent with its FIRST base bar's time).
For 1D parents the window is the calendar day; empty (lo==hi) before the 5m store starts (2017) — the
engine treats missing children as "no intrabar data" and falls back to parent-bar semantics, exactly
like pine_compat mode. Children for 1W/1M are not provided in v1 (weekly+ decisions don't need
intrabar fidelity; revisit if character on 1W ever matters).
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
import datastore


# Minutes per TF for ladder math. Daily+ use per-instrument session minutes so the ~4-6x rung spacing
# reasons in TRADED time (equities RTH day = 390m; futures day ~ 23h). Calendar TFs beyond that use
# trading-day counts (1W = 5 days, 1M = 21 days).
_EQ_DAY = 390
_FUT_DAY = 1380
_INTRA = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240, "8h": 480, "12h": 720}

# TWO INTERLEAVED rung ladders (Amir 2026-07-15, superseding the single ladder locked 2026-07-14).
# Each ladder keeps ~4-6x spacing WITHIN itself; together they interleave to cover the scales a
# trader actually reads (1m,5m,15m,30m,2h,4h,8h,1D,2D,1W,2W) without any single ladder jumping
# raggedly. A timeframe belongs to exactly one ladder, and parents() walks that ladder.
#
#   A (futures): 1m -> 5m -> 30m -> 2h -> 8h -> 2D -> 2W    (5x, 6x, 4x, 4x, 5.75x, 5x)
#   A (stocks):  1m -> 5m -> 30m -> 2h -> 2D -> 2W          (8h DROPPED — see below)
#   B (both):    15m -> 4h -> 1D -> 1W
#
# Two instrument-specific realities this encodes:
#  - 8h EXISTS FOR FUTURES ONLY. The Globex day is ~23h so 8h = 3 bars/day; an RTH stock day is
#    6.5h, so an "8h" RTH bar is LARGER than the day it lives in and degenerates. Equities skip
#    the rung; 2h -> 2D is 6.5x on RTH, which stays inside the spacing rule anyway.
#  - Ladder B's rungs are spaced for the 23h futures day: 4h -> 1D is 5.75x on futures but only
#    1.6x on an RTH stock day, and 15m -> 4h is 16x on either. B is therefore a COARSE context
#    ladder, not a strict 4-6x chain — kept as Amir specified. Flagged in
#    docs/plan-mtf-timeframes-relationships.md rather than silently "corrected".
LADDER_A_FUT = ["1m", "5m", "30m", "2h", "8h", "2D", "2W"]
LADDER_A_EQ = ["1m", "5m", "30m", "2h", "2D", "2W"]
LADDER_B = ["15m", "4h", "1D", "1W"]

# Back-compat aliases: callers that just want "the main ladder" for an instrument class.
EQ_LADDER = LADDER_A_EQ
FUT_LADDER = LADDER_A_FUT


def _ladders(futures: bool) -> list[list[str]]:
    """Ladder A first (it owns the fallback entry rule), then B."""
    return [LADDER_A_FUT if futures else LADDER_A_EQ, LADDER_B]


def tf_minutes(tf: str, futures: bool = False) -> int:
    """Traded minutes per bar of `tf` — for ladder spacing math, not exact wall-clock."""
    if tf in _INTRA:
        return _INTRA[tf]
    day = _FUT_DAY if futures else _EQ_DAY
    return {"1D": day, "2D": 2 * day, "3D": 3 * day,
            "1W": 5 * day, "2W": 10 * day, "1M": 21 * day}[tf]


def parents(tf: str, futures: bool = False, n: int = 3) -> list[str]:
    """The next `n` parent timeframes above `tf`, walking the ladder that `tf` belongs to.

    A tf that is not itself a rung on either ladder (e.g. 1h) enters at the first rung >= 4x its own
    size, choosing whichever ladder offers the TIGHTEST such fit — so 1h enters ladder B at 4h (4x)
    rather than ladder A at 8h (8x) — then walks that ladder."""
    for lad in _ladders(futures):
        if tf in lad:
            return lad[lad.index(tf) + 1:lad.index(tf) + 1 + n]
    m = tf_minutes(tf, futures)
    best = None
    for lad in _ladders(futures):
        i = next((k for k, r in enumerate(lad) if tf_minutes(r, futures) >= 4 * m), None)
        if i is None:
            continue
        ratio = tf_minutes(lad[i], futures) / m
        if best is None or ratio < best[0]:
            best = (ratio, lad, i)
    if best is None:
        return []
    _, lad, i = best
    return lad[i:i + n]


@dataclass
class TFFrame:
    """One (symbol, timeframe) frame + optional aligned child stream (see module docstring)."""
    symbol: str
    tf: str
    df: pd.DataFrame                       # OHLCV, tz-naive US/Eastern wall-clock DatetimeIndex
    futures: bool
    base: pd.DataFrame | None = None       # the 5m child stream (None when tf is the base / children off)
    child_lo: np.ndarray | None = None     # children of parent i = base.iloc[child_lo[i]:child_hi[i]]
    child_hi: np.ndarray | None = None

    def children(self, i: int) -> pd.DataFrame | None:
        if self.base is None or self.child_lo is None:
            return None
        lo, hi = int(self.child_lo[i]), int(self.child_hi[i])
        return self.base.iloc[lo:hi] if hi > lo else None


def is_futures(symbol: str) -> bool:
    return symbol.endswith("=F") or symbol.upper() in ("NQF", "ESF", "RTYF")


def _base_5m(symbol: str, futures: bool, calibrate: bool) -> pd.DataFrame | None:
    if futures:
        return datastore.load_bars_15m(symbol)     # LEGACY name: the futures 5m store
    return datastore.load_bars_5m(symbol, calibrate=calibrate)


def _intraday_windows(base: pd.DataFrame, parent_index: pd.DatetimeIndex):
    """Exact by construction: each parent bar is labeled with its first base bar's timestamp."""
    lo = base.index.searchsorted(parent_index, side="left")
    hi = np.r_[lo[1:], len(base)]
    return lo.astype(np.int64), hi.astype(np.int64)


def _daily_windows(base: pd.DataFrame, daily_index: pd.DatetimeIndex):
    days = daily_index.normalize()
    lo = base.index.searchsorted(days, side="left")
    hi = base.index.searchsorted(days + pd.Timedelta(days=1), side="left")
    return lo.astype(np.int64), hi.astype(np.int64)


def frame(symbol: str, tf: str = "5m", futures: bool | None = None,
          children: bool = True, calibrate: bool = True) -> TFFrame | None:
    """Load the (symbol, tf) OHLCV frame per the routing table above. Returns None when the backing
    store has no data. `children=True` attaches the 5m stream + per-parent windows for intraday
    tf >= 15m and for 1D (empty windows where 5m coverage hasn't started)."""
    fut = is_futures(symbol) if futures is None else futures
    if tf in ("1m",):
        p = config.DATA_DIR / "tiingo" / "bars_1min" / f"{symbol}{datastore.EXT}"
        df = datastore._read(p) if p.exists() else None      # no public loader for the ad-hoc 1m store
        return TFFrame(symbol, tf, df, fut) if df is not None and len(df) else None
    if tf in _INTRA:
        base = _base_5m(symbol, fut, calibrate)
        if base is None or not len(base):
            return None
        if tf == "5m":
            return TFFrame(symbol, tf, base, fut)
        df = datastore.resample_intraday(base, tf, futures=fut)
        if df is None or not len(df):
            return None
        out = TFFrame(symbol, tf, df, fut)
        if children:
            out.base = base
            out.child_lo, out.child_hi = _intraday_windows(base, df.index)
        return out
    if tf == "1D":
        df = datastore.load_bars(symbol)
        if df is None or not len(df):
            return None
        out = TFFrame(symbol, tf, df, fut)
        if children:
            base = _base_5m(symbol, fut, calibrate)
            if base is not None and len(base):
                out.base = base
                out.child_lo, out.child_hi = _daily_windows(base, df.index)
        return out
    if tf == "1W":
        df = datastore.load_bars_w(symbol)
        return TFFrame(symbol, tf, df, fut) if df is not None and len(df) else None
    # 1M / 2D / 3D / ... — derived from daily
    daily = datastore.load_bars(symbol)
    if daily is None or not len(daily):
        return None
    df = datastore.resample_daily(daily, tf)
    return TFFrame(symbol, tf, df, fut) if df is not None and len(df) else None


def load_research(symbol: str, tf: str) -> pd.DataFrame | None:
    """Fixed research datasets under data/research_1y/ (e.g. NQF_5m.csv.gz, NQF_1m, NQF_1D) — used by
    the parity harness for a stable 1-year futures reference independent of the rolling 60-day store."""
    p = config.DATA_DIR / "research_1y" / f"{symbol}_{tf}.csv.gz"
    if not p.exists():
        return None
    with gzip.open(p, "rt") as fh:
        df = pd.read_csv(fh)
    tcol = next((c for c in df.columns if c.lower() in ("datetime", "date", "time", "timestamp")), df.columns[0])
    df[tcol] = pd.to_datetime(df[tcol])
    df = df.set_index(tcol)
    df.columns = [str(c).lower() for c in df.columns]
    df = df[[c for c in ("open", "high", "low", "close", "volume") if c in df.columns]]
    df.index.name = "datetime"
    return df.sort_index()
