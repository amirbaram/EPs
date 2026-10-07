"""Precompute the Perf-panel numbers (Sector & Theme Performance) for EVERY past trading day.

What the panel shows, and what this writes (all to data/perf_history/):

  groups.parquet          one row per (date, kind, name): n members + MEDIAN member return for
                          d1 / w1 / m1 / m3 / ytd (%).  kind = "sector" | "theme".
                          Identical math to market.theme_board (the Themes / Sectors / Both views).
  tickers_<YEAR>.parquet  one row per (date, symbol): d1 / wtd / w1 / m1 / m3 / ytd (%).
                          Covers every stock/ETF AND the futures (=F symbols), so it feeds the
                          per-ticker drill-in, the Futures view and the Universe view.

Definitions (same as the live app, performance._daily_returns / market.theme_board):
  d1/w1/m1/m3 = close / close N bars ago - 1 on the symbol's OWN bar series (1/5/21/63 bars)
  wtd         = vs the last close of the prior ISO week
  ytd         = vs the last close of the prior calendar year
  group value = MEDIAN of its members' returns (not mean: one split artifact wrecks a mean);
                groups with < config.GROUP_MIN_MEMBERS valid members are omitted.
NOT stored: the intraday "Open" / "PrevCls" columns - they only exist for the live session.

Known simplification (same as the live app): group membership is TODAY's sectors.csv/themes.csv,
applied to the past. A symbol that stopped trading is carried forward at most 5 sessions, then dropped.

Run:   .venv/bin/python build_perf_history.py              # full rebuild (~1-2 min)
       .venv/bin/python build_perf_history.py --since 2026-09-01   # refresh only recent days (nightly)
Read:  import build_perf_history as bph; bph.load_groups("2025-10-01"); bph.load_tickers("2025-10-01")
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config
import datastore
import labels

OUT_DIR = Path(config.DATA_DIR) / "perf_history"
WINDOWS = {"d1": 1, "w1": 5, "m1": 21, "m3": 63}
METRICS = ["d1", "wtd", "w1", "m1", "m3", "ytd"]
GROUP_METRICS = ["d1", "w1", "m1", "m3", "ytd"]
STALE_LIMIT = 5          # carry a symbol's last value forward at most this many sessions
START_DEFAULT = "2016-01-01"


def symbol_returns(c: pd.Series) -> dict[str, pd.Series]:
    """All six returns (%) for ONE symbol, indexed by that symbol's own bar dates."""
    c = c[np.isfinite(c) & (c > 0)]
    out: dict[str, pd.Series] = {}
    if len(c) < 2:
        return out
    for k, nb in WINDOWS.items():
        out[k] = (c / c.shift(nb) - 1) * 100
    iso = c.index.isocalendar()
    wk = (iso["year"].astype(int) * 100 + iso["week"].astype(int)).to_numpy()
    wk_last = c.groupby(wk).last()                       # each ISO week's last close
    wk_prev = wk_last.shift(1)                           # ... the PRIOR existing week's
    base_w = pd.Series(wk_prev.reindex(wk).to_numpy(), index=c.index)
    out["wtd"] = (c / base_w - 1) * 100
    yr = c.index.year.to_numpy()
    yr_last = c.groupby(yr).last()
    yr_prev = yr_last.shift(1)                           # prior year's last close (empty -> NaN)
    yr_prev = yr_prev.ffill()
    base_y = pd.Series(yr_prev.reindex(yr).to_numpy(), index=c.index)
    out["ytd"] = (c / base_y - 1) * 100
    return out


def compute_tickers(closes: dict[str, pd.Series], dates: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """metric -> wide DataFrame (dates x symbols), float32, rounded to 2dp."""
    cols: dict[str, dict[str, np.ndarray]] = {m: {} for m in METRICS}
    for sym, c in closes.items():
        r = symbol_returns(c)
        for m, s in r.items():
            cols[m][sym] = s.reindex(dates.union(s.index)).ffill(limit=STALE_LIMIT).reindex(dates).to_numpy(np.float32)
    return {m: pd.DataFrame(cols[m], index=dates).round(2) for m in METRICS}


def compute_groups(wide: dict[str, pd.DataFrame], sector: dict[str, str],
                   themes: dict[str, list[str]]) -> pd.DataFrame:
    """Median member return per (date, kind, name); mirrors market.theme_board."""
    members: dict[tuple[str, str], list[str]] = {}
    for sym, sec in sector.items():
        if sec and sym in wide["d1"].columns:
            members.setdefault(("sector", sec), []).append(sym)
    for th, syms in themes.items():
        syms = [s for s in syms if s in wide["d1"].columns]
        if syms:
            members[("theme", th)] = syms
    frames = []
    for (kind, name), syms in members.items():
        if len(syms) < config.GROUP_MIN_MEMBERS:
            continue
        med = {m: wide[m][syms].median(axis=1, skipna=True) for m in GROUP_METRICS}
        cnt = pd.concat([wide[m][syms].notna().sum(axis=1) for m in GROUP_METRICS], axis=1).max(axis=1)
        df = pd.DataFrame(med)
        df["n"] = cnt
        df = df[df["n"] >= config.GROUP_MIN_MEMBERS].dropna(how="all", subset=GROUP_METRICS)
        df.insert(0, "name", name)
        df.insert(0, "kind", kind)
        df.index.name = "date"
        frames.append(df.reset_index())
    g = pd.concat(frames, ignore_index=True)
    g[GROUP_METRICS] = g[GROUP_METRICS].astype("float32").round(2)
    g["n"] = g["n"].astype("int16")
    return g.sort_values(["date", "kind", "name"]).reset_index(drop=True)


def _load_closes(start: str) -> dict[str, pd.Series]:
    closes = {}
    for sym in datastore.list_symbols():
        d = datastore.load_bars(sym)
        if d is None or not len(d):
            continue
        closes[sym] = d["close"].astype(float)
    return closes


def build(since: str | None = None, start: str = START_DEFAULT) -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("loading bars ...", flush=True)
    closes = _load_closes(start)
    spy = closes["SPY"]
    dates = spy.index[spy.index >= pd.Timestamp(start)]
    print(f"  {len(closes)} symbols, {len(dates)} sessions {dates[0].date()} -> {dates[-1].date()} "
          f"({time.time()-t0:.0f}s)", flush=True)

    print("per-ticker returns ...", flush=True)
    wide = compute_tickers(closes, dates)
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    labels.load()
    sector = dict(labels._SECTOR or {})
    themes = {t: sorted(labels.tickers_in_theme(t)) for t in labels.all_themes()}
    print("group medians ...", flush=True)
    groups = compute_groups(wide, sector, themes)
    print(f"  {len(groups)} group-days ({time.time()-t0:.0f}s)", flush=True)

    cut = pd.Timestamp(since) if since else None
    gpath = OUT_DIR / "groups.parquet"
    if cut is not None and gpath.exists():
        old = pd.read_parquet(gpath)
        groups = pd.concat([old[old["date"] < cut], groups[groups["date"] >= cut]], ignore_index=True)
        groups = groups.sort_values(["date", "kind", "name"]).reset_index(drop=True)
    groups.to_parquet(gpath, index=False)

    long = pd.concat({m: wide[m].stack(future_stack=True) for m in METRICS}, axis=1)
    long.index.names = ["date", "symbol"]
    long = long.dropna(how="all").astype("float32").reset_index()
    long["date"] = pd.to_datetime(long["date"])
    for yr, part in long.groupby(long["date"].dt.year):
        if cut is not None and yr < cut.year:
            continue
        p = OUT_DIR / f"tickers_{yr}.parquet"
        if cut is not None and p.exists() and yr == cut.year:
            old = pd.read_parquet(p)
            part = pd.concat([old[old["date"] < cut], part[part["date"] >= cut]], ignore_index=True)
        part.sort_values(["date", "symbol"]).to_parquet(p, index=False, row_group_size=200_000)
    print(f"wrote {OUT_DIR} in {time.time()-t0:.0f}s", flush=True)


# ── readers (for the app) ──────────────────────────────────────────────────────
def load_groups(date: str) -> pd.DataFrame:
    """Sector/theme rows for one trading day (date = 'YYYY-MM-DD'); empty if not built."""
    p = OUT_DIR / "groups.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p, filters=[("date", "==", pd.Timestamp(date))])


def load_tickers(date: str) -> pd.DataFrame:
    """Per-symbol rows (stocks, ETFs, futures) for one trading day."""
    p = OUT_DIR / f"tickers_{str(date)[:4]}.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p, filters=[("date", "==", pd.Timestamp(date))])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", help="only refresh dates >= this (YYYY-MM-DD); default = full rebuild")
    ap.add_argument("--start", default=START_DEFAULT, help="first date to compute")
    a = ap.parse_args()
    build(since=a.since, start=a.start)
