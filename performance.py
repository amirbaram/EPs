"""Equal-weighted synthetic performance for sectors and themes across multiple timeframes.

For themes that have a tracking ETF in build_labels.THEME_ETFS, the ETF's own price series
is used as the primary signal (single clean series, institutional consensus on membership).
For everything else — and as a secondary signal alongside the ETF — an equal-weighted
synthetic is computed from the member tickers' bar data (winsorized to cap outliers).

Timeframes
----------
  open_pct        intraday return since today's first 15m bar open (needs 15m bars)
  prev_close_pct  return since prior daily close (needs 15m bars for intraday, or daily)
  w1_pct          1-week return  (~5 trading days, daily bars)
  m1_pct          1-month return (~21 trading days, daily bars)
  m3_pct          3-month return (~63 trading days, daily bars)

Usage
-----
  import performance
  result = performance.all_performance(FRAMES, as_of="2026-06-30")
  # result = {"themes": {theme: {open_pct, prev_close_pct, w1_pct, m1_pct, m3_pct, n, etf}},
  #           "sectors": {sector: {...}}}
"""
from __future__ import annotations

import datetime as dt
from statistics import median

import pandas as pd

import build_labels
import datastore
import labels

# individual return cap before averaging (prevents one earnings gap dominating a group)
WINSOR_INTRADAY = 10.0   # % cap for intraday / prev-close returns
WINSOR_DAILY    = 25.0   # % cap for multi-day returns

# minimum members with valid data to report a group's return
MIN_MEMBERS = 3

# lookback bars for each daily timeframe (d1 = since prior close; wtd added separately below)
_BARS = {"d1": 1, "w1": 5, "m1": 21, "m3": 63}


# ── ETF lookup (theme -> first ETF ticker with data in the bar cache) ─────────

def _etf_for_theme(theme: str, frames: dict) -> str | None:
    for etf in build_labels.THEME_ETFS.get(theme, []):
        if etf in frames:
            return etf
    return None


# ── single-ticker daily returns ───────────────────────────────────────────────

def _daily_returns(df: pd.DataFrame, as_of: str | None) -> dict[str, float | None]:
    """Multi-bar % returns for one ticker's daily frame, as of `as_of` date."""
    d = df.loc[:as_of] if as_of else df
    if d is None or len(d) < 2:
        return {k: None for k in _BARS}
    c = d["close"].to_numpy(float)
    n = len(c)
    out = {}
    for key, bars in _BARS.items():
        if n > bars and c[-1 - bars] > 0:
            out[key] = (c[-1] / c[-1 - bars] - 1) * 100
        else:
            out[key] = None
    out["wtd"] = None                                    # week-to-date: since the last close of the prior ISO week
    cur = d.index[-1].isocalendar()[:2]
    for j in range(n - 2, -1, -1):
        if d.index[j].isocalendar()[:2] != cur:
            base = c[j]
            out["wtd"] = (c[-1] / base - 1) * 100 if base > 0 else None
            break
    return out


def _intraday_returns(df15: pd.DataFrame) -> dict[str, float | None]:
    """Intraday returns from 15m bars: since open + since prior close."""
    if df15 is None or len(df15) < 2:
        return {"open": None, "prev_close": None}
    # today = calendar date of the latest bar
    latest_date = df15.index[-1].date()
    today = df15[df15.index.date == latest_date]
    if today.empty:
        return {"open": None, "prev_close": None}
    last_price = float(today["close"].iloc[-1])
    open_price = float(today["open"].iloc[0])
    open_pct = (last_price / open_price - 1) * 100 if open_price > 0 else None
    # prior session: latest close before today
    prior = df15[df15.index.date < latest_date]
    if prior.empty:
        return {"open": open_pct, "prev_close": None}
    prior_close = float(prior["close"].iloc[-1])
    prev_close_pct = (last_price / prior_close - 1) * 100 if prior_close > 0 else None
    return {"open": open_pct, "prev_close": prev_close_pct}


# ── aggregation helpers ───────────────────────────────────────────────────────

def _wmean(vals: list[float | None], cap: float) -> float | None:
    """Winsorized equal-weighted mean. Returns None if fewer than MIN_MEMBERS valid."""
    clean = [max(-cap, min(cap, v)) for v in vals if v is not None]
    if len(clean) < MIN_MEMBERS:
        return None
    return sum(clean) / len(clean)


def _group_perf(
    symbols: set[str],
    frames: dict,          # sym -> daily enriched DataFrame
    bars15: dict,          # sym -> 15m DataFrame (may be empty)
    as_of: str | None,
) -> dict:
    """Compute all timeframe returns for a group of tickers. Returns a perf dict."""
    daily_rets: dict[str, list[float | None]] = {k: [] for k in _BARS}
    intra_open, intra_prev = [], []
    n_daily = n_intra = 0

    for sym in symbols:
        df = frames.get(sym)
        if df is not None and len(df) >= 2:
            dr = _daily_returns(df, as_of)
            for k in _BARS:
                daily_rets[k].append(dr[k])
            n_daily += 1
        df15 = bars15.get(sym)
        if df15 is not None and len(df15) >= 2:
            ir = _intraday_returns(df15)
            intra_open.append(ir["open"])
            intra_prev.append(ir["prev_close"])
            n_intra += 1

    def _r(v):
        return round(v, 2) if v is not None else None

    return {
        "open_pct":       _r(_wmean(intra_open, WINSOR_INTRADAY)),
        "prev_close_pct": _r(_wmean(intra_prev, WINSOR_INTRADAY)),
        "w1_pct":         _r(_wmean(daily_rets["w1"], WINSOR_DAILY)),
        "m1_pct":         _r(_wmean(daily_rets["m1"], WINSOR_DAILY)),
        "m3_pct":         _r(_wmean(daily_rets["m3"], WINSOR_DAILY)),
        "n": n_daily,
    }


def _etf_perf(etf: str, frames: dict, bars15: dict, as_of: str | None) -> dict:
    """Performance from a single ETF's own price series."""
    df = frames.get(etf)
    if df is None:
        return {}
    dr = _daily_returns(df, as_of)
    df15 = bars15.get(etf)
    ir = _intraday_returns(df15) if df15 is not None else {"open": None, "prev_close": None}
    return {
        "open_pct":       round(ir["open"], 2) if ir["open"] is not None else None,
        "prev_close_pct": round(ir["prev_close"], 2) if ir["prev_close"] is not None else None,
        "w1_pct":         round(dr["w1"], 2) if dr["w1"] is not None else None,
        "m1_pct":         round(dr["m1"], 2) if dr["m1"] is not None else None,
        "m3_pct":         round(dr["m3"], 2) if dr["m3"] is not None else None,
        "n": 1,
    }


# ── per-ticker breakdown ──────────────────────────────────────────────────────

def ticker_perf(sym: str, frames: dict, bars15: dict, as_of: str | None) -> dict | None:
    """All timeframe returns for a single ticker. Returns None if no data."""
    df = frames.get(sym)
    df15 = bars15.get(sym)
    if df is None and df15 is None:
        return None
    def _r(v):
        return round(v, 2) if v is not None else None
    dr = _daily_returns(df, as_of) if df is not None else {k: None for k in _BARS}
    ir = _intraday_returns(df15) if df15 is not None else {"open": None, "prev_close": None}
    return {
        "open_pct":       _r(ir["open"]),
        "prev_close_pct": _r(ir["prev_close"]),
        "d1_pct":         _r(dr["d1"]),
        "wtd_pct":        _r(dr["wtd"]),
        "w1_pct":         _r(dr["w1"]),
        "m1_pct":         _r(dr["m1"]),
        "m3_pct":         _r(dr["m3"]),
    }


def group_tickers(
    symbols: set[str],
    frames: dict,
    bars15: dict,
    as_of: str | None,
) -> list[dict]:
    """Per-ticker performance for every symbol in the group, as a list sorted by m1_pct desc."""
    out = []
    for sym in symbols:
        p = ticker_perf(sym, frames, bars15, as_of)
        if p is not None:
            out.append({"symbol": sym, **p})
    out.sort(key=lambda r: (r["m1_pct"] is not None, r["m1_pct"] or 0), reverse=True)
    return out


# ── public API ────────────────────────────────────────────────────────────────

def _load_bars15(symbols: set[str]) -> dict:
    """Load 15m bars for a set of symbols (only what's cached on disk)."""
    out = {}
    for sym in symbols:
        df = datastore.load_bars_15m(sym)
        if df is not None and len(df):
            out[sym] = df
    return out


def universe_perf(frames: dict, as_of: str | None = None, top_n: int = 50, bars15: dict | None = None) -> dict:
    """
    Per-ticker performance for the full universe.  Daily returns computed in RAM for all
    tickers (fast); 15m bars loaded only for the top+bottom N to add intraday columns.

    Returns
    -------
    {
      "top":    [{"symbol", "w1_pct", "m1_pct", "m3_pct", "open_pct", "prev_close_pct"}, ...],
      "bottom": [...],   # worst performers (same fields)
      "breadth": {"w1": float, "m1": float, "m3": float},   # % of universe > 0
      "total":  int,     # number of tickers with enough data
      "as_of":  str,
    }
    """
    # ── 1. daily returns for every ticker (all in RAM — fast) ─────────────────
    rows = []
    w1_pos = m1_pos = m3_pos = w1_n = m1_n = m3_n = 0
    for sym, df in frames.items():
        dr = _daily_returns(df, as_of)
        if all(v is None for v in dr.values()):
            continue
        rows.append({"symbol": sym, **{f"{k}_pct": dr[k] for k in _BARS}})
        for k, pos_count, n_count in [("w1", "w1_pos", "w1_n"),
                                       ("m1", "m1_pos", "m1_n"),
                                       ("m3", "m3_pos", "m3_n")]:
            if dr[k] is not None:
                if k == "w1":  w1_n += 1;  w1_pos += (1 if dr[k] > 0 else 0)
                elif k == "m1": m1_n += 1; m1_pos += (1 if dr[k] > 0 else 0)
                elif k == "m3": m3_n += 1; m3_pos += (1 if dr[k] > 0 else 0)

    # ── 2. sort by m1, pick top + bottom N ───────────────────────────────────
    def _sort_key(r, col="m1_pct"):
        v = r.get(col)
        return (v is not None, v or 0)

    rows.sort(key=lambda r: _sort_key(r, "m1_pct"), reverse=True)
    top    = rows[:top_n]
    bottom = list(reversed(rows[-top_n:])) if len(rows) >= top_n else []

    # ── 3. load 15m only for the shortlisted tickers (or use a supplied replay-truncated store) ──
    shortlist = {r["symbol"] for r in top + bottom}
    if bars15 is None:
        bars15 = _load_bars15(shortlist)
    def _r(v): return round(v, 2) if v is not None else None
    for row in top + bottom:
        ir = _intraday_returns(bars15.get(row["symbol"])) if bars15.get(row["symbol"]) is not None else {"open": None, "prev_close": None}
        row["open_pct"]       = _r(ir["open"])
        row["prev_close_pct"] = _r(ir["prev_close"])
        for k in _BARS:
            row[f"{k}_pct"] = _r(row.get(f"{k}_pct"))

    breadth = {
        "w1": round(w1_pos / w1_n * 100, 1) if w1_n else None,
        "m1": round(m1_pos / m1_n * 100, 1) if m1_n else None,
        "m3": round(m3_pos / m3_n * 100, 1) if m3_n else None,
    }
    return {"top": top, "bottom": bottom, "breadth": breadth,
            "total": len(rows), "as_of": as_of or dt.date.today().isoformat()}


def all_performance(frames: dict, as_of: str | None = None, bars15: dict | None = None) -> dict:
    """
    Compute performance for all themes and sectors.

    Parameters
    ----------
    frames : dict[str, DataFrame]
        Daily enriched frames already in RAM (e.g. serve.FRAMES).
    as_of : str | None
        ISO date string for backtest mode; None = use all available data.

    Returns
    -------
    {
      "themes":  {theme_name:  {"open_pct", "prev_close_pct", "w1_pct", "m1_pct", "m3_pct",
                                "n", "etf" (str|None), "synth_w1_pct", ...}},
      "sectors": {sector_name: {"open_pct", ..., "n"}},
      "as_of":   str,
    }
    """
    labels.load()
    all_syms_in_frames = set(frames)

    # collect all symbols we'll need 15m bars for
    theme_members: dict[str, set[str]] = {
        t: labels.tickers_in_theme(t) & all_syms_in_frames
        for t in labels.all_themes()
    }
    sector_members: dict[str, set[str]] = {}
    for sym, sec in (labels._SECTOR or {}).items():
        if sym in all_syms_in_frames:
            sector_members.setdefault(sec, set()).add(sym)

    # ETF tickers needed
    etf_map: dict[str, str | None] = {t: _etf_for_theme(t, frames) for t in theme_members}
    all_needed = set.union(*theme_members.values(), *sector_members.values()) if (theme_members or sector_members) else set()
    etf_syms = {e for e in etf_map.values() if e}
    all_needed |= etf_syms

    # intraday store for since-open / since-prev-close. Callers pass the LIVE 5m store (serve's cached
    # _day5_store, calibrated) so the perf panel reads the current session; the _load_bars15 fallback is
    # the legacy (now-frozen) 15m store, kept only for standalone/offline callers.
    if bars15 is None:
        bars15 = _load_bars15(all_needed)

    themes_out = {}
    for theme, members in theme_members.items():
        if not members:
            continue
        etf = etf_map.get(theme)
        synth = _group_perf(members, frames, bars15, as_of)
        if etf:
            ep = _etf_perf(etf, frames, bars15, as_of)
            # primary = ETF; keep synth figures as synth_* for reference
            row = {**ep, "etf": etf,
                   "synth_w1_pct": synth.get("w1_pct"),
                   "synth_m1_pct": synth.get("m1_pct"),
                   "synth_m3_pct": synth.get("m3_pct"),
                   "synth_n": synth.get("n")}
        else:
            row = {**synth, "etf": None}
        themes_out[theme] = row

    sectors_out = {}
    for sec, members in sector_members.items():
        if len(members) < MIN_MEMBERS:
            continue
        sectors_out[sec] = _group_perf(members, frames, bars15, as_of)

    return {
        "themes": themes_out,
        "sectors": sectors_out,
        "as_of": as_of or dt.date.today().isoformat(),
    }
