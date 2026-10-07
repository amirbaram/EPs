"""Build a real dcs_mtf.MTFState for ES/NQ from the Databento 1-min store.

Why this module exists (2026-07-15): dcs_data's futures path still points at the legacy Tiingo
futures store — ES had 1,560 bars (~3 months, RTH-only) and NQ had NOTHING. The Databento store
(data/futures_db/) has 16 YEARS of full 23h-Globex 1-min bars for both. Rather than swap the data
source underneath dcs_data/dcs_mtf — shared, parity-tested infrastructure that DCS-5's 153-symbol
batch depends on — this module builds an equivalent MTFState from the better data, reusing
dcs_mtf's OWN alignment helpers so the causal semantics are provably identical.

Extracted from ret3_overnight_test.py so dayfeatures.py can reuse it without duplicating the
builder (one copy = one place for a bug to live).

    m = futures_mtf.stack("ES", tf="2h", parents=("1D", "1W"))
    levels = dcs_levels.parent_levels(m, i)          # real confirmed-pivot S/R, causal
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
import dcs_character
import dcs_core
import dcs_data
import dcs_mtf
import dcs_v5_config as v5cfg

_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def available(fut: str) -> bool:
    return (config.DATA_DIR / "futures_db" / f"{fut}_1m.parquet").exists()


def load_1m(fut: str) -> pd.DataFrame | None:
    p = config.DATA_DIR / "futures_db" / f"{fut}_1m.parquet"
    return pd.read_parquet(p) if p.exists() else None


def resample(d: pd.DataFrame, rule: str) -> pd.DataFrame:
    return d.resample(rule).agg(_AGG).dropna()


# pandas resample rules per TF label. 2h/4h/8h bin from midnight, which for a 23h Globex session
# is the pragmatic choice (the session itself starts 18:00 ET the prior day; exact session-anchored
# binning is a refinement, flagged in docs/plan-mtf-timeframes-relationships.md).
_RULE = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "2h": "2h", "4h": "4h",
         "8h": "8h", "1D": "1D", "2D": "2D", "1W": "1W", "2W": "2W"}


def stack(fut: str, tf: str = "2h", parents: tuple[str, ...] = ("1D", "1W"),
          mult: float = 1.0, eq_frac: float | None = None) -> dcs_mtf.MTFState | None:
    """MTFState for `fut` at `tf` with the given parent TFs, built off Databento 1-min bars.

    eq_frac defaults to v5cfg.EQ_FRAC (G5-approved wiring, Amir via FeaturesStructures 2026-07-16).
    This module hardcoded 0.0 — the v4 legacy value — because it predates the v5 plan (75e2505); the
    v5 intent never reached it since the wiring point sits in a load-bearing file outside the feature
    lane, and nobody noticed because the constant EXISTED and looked wired (v5cfg.EQ_FRAC was in fact
    passed to dcs_core.run nowhere in the codebase). The master plan predicted this exact outcome in
    writing — "eqTol default 0 disables the flat-boundary regimes — v5 should default eq_frac to a
    small nonzero (propose 0.25 ATR, sweepable) or the range-event layer under-fires" — and it
    under-fired across 16 years of data. Pass an explicit eq_frac to override (e.g. 0.0 for a v4-
    comparable run); dcs_mtf.stack()'s own 0.0 default is deliberately NOT changed — that flips the
    legacy 153-symbol equity path, a blast radius nobody has examined."""
    if eq_frac is None:
        eq_frac = v5cfg.EQ_FRAC
    raw = load_1m(fut)
    if raw is None or not len(raw):
        return None
    child_df = resample(raw, _RULE[tf])
    if len(child_df) < dcs_core.ATR_LEN + 2:
        return None
    ccore = dcs_core.run(child_df, mult=mult, eq_frac=eq_frac, pine_compat=False)
    cchar = dcs_character.run(child_df, ccore, pine_compat=False)
    child_frame = dcs_data.TFFrame(symbol=fut, tf=tf, df=child_df, futures=True)
    m = dcs_mtf.MTFState(symbol=fut, tf=tf, child=child_frame, core=ccore, char=cchar)
    child_t = child_df.index.as_unit("ns").asi8

    for ptf in parents:
        pf = resample(raw, _RULE[ptf])
        if len(pf) < dcs_core.ATR_LEN + 2:
            continue
        pcore = dcs_core.run(pf, mult=mult, eq_frac=eq_frac, pine_compat=False)
        pchar = dcs_character.run(pf, pcore, pine_compat=False)
        pframe = dcs_data.TFFrame(symbol=fut, tf=ptf, df=pf, futures=True)
        closed = dcs_mtf._closed_times(pf.index, ptf)
        map_causal = (closed.searchsorted(child_t, side="right") - 1).astype(np.int64)
        map_dev = dcs_mtf._containing(pf.index, ptf, child_t).astype(np.int64)
        m.parents.append(dcs_mtf.ParentView(ptf, pframe, pcore, pchar, map_causal, map_dev))

    n = ccore.n
    rel = np.zeros(n, np.int8)
    for i in range(n):
        code = 0
        for pv in m.parents:
            k = int(pv.map_causal[i])
            if k < 0:
                continue
            pk = int(pv.core.kind[k])
            code = dcs_mtf.relationship(int(ccore.kind[i]), pk)
            if pk != 0:
                break
        rel[i] = code
    m.rel = rel
    return m


def session_positions(m: dcs_mtf.MTFState, start=None, end=None) -> list[tuple[pd.Timestamp, int]]:
    """(session_date, child bar index) for the LAST child bar closed before each RTH open (09:30).
    That is the causal 'what did the overnight leave us with' read for a next-day feature."""
    idx = m.child.df.index
    lo = pd.Timestamp(start).date() if start else idx[30].date()
    hi = pd.Timestamp(end).date() if end else idx[-1].date()
    out = []
    for sess in pd.date_range(lo, hi, freq="B"):
        cutoff = pd.Timestamp(sess) + pd.Timedelta(hours=9, minutes=30)
        pos = idx.searchsorted(cutoff) - 1
        if pos >= 20:
            out.append((pd.Timestamp(sess.date()), int(pos)))
    return out
