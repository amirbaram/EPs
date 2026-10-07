"""history_store — per-day persisted history of theme/sector scoreboards + per-entity
trend / health state. Contract: specs/2026-07-29-daily-history-store.md (Phase 1).

Per SETTLED session it records, for a curated entity set:
  · market   — health only (health_score.read); dma_state/trend NULL (index trend lives on ES/NQ).
  · ES=F/NQ=F, tickers — dma_state (3-state, 50dma-gated), trend (mtf.tf_state trend_dir),
                          health (mtf.tf_state structural score via market._score_state).
  · themes   — group_scores fields + theme_board medians; health = group_scores score;
                dma_state/trend from the theme's PROXY ETF (NULL if no ETF).
  · sectors  — group_scores fields + theme_board medians; health = group_scores score;
                dma_state/trend NULL (sectors have no proxy ETF — AC-3, Amir 2026-07-29).
Plus a theme_leaders table: per (date, theme) the member tickers ranked by RS.

CAVEATS (binding non-goals):
  NG-2 — theme membership is CURRENT-only (labels.tickers_in_theme has no as-of): historical
         rows apply TODAY's membership to past prices. NOT point-in-time.
  NG-3 — settled-daily only; no intraday "since-09:30 open" column.

Storage (parquet, long-form, partitioned by year):
  data/history/entities/<YYYY>.parquet       one row per (date, kind, id)
  data/history/theme_leaders/<YYYY>.parquet  one row per (date, theme, ticker)

Backfill (monitored):   .venv/bin/python history_store.py --start 2023-01-01
Small-scale test first:  .venv/bin/python history_store.py --start 2026-06-01 --tag hist_test
Nightly append is wired into serve._run_update (fail-open).
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

import config
import datastore
import health_score
import indicators
import labels
import market
import mtf

HIST_DIR = config.DATA_DIR / "history"
ENT_DIR = HIST_DIR / "entities"
LEAD_DIR = HIST_DIR / "theme_leaders"

FUTURES = ["ES=F", "NQ=F"]

# ── DMA state (AC-2): 3-state, 50dma-gated ──────────────────────────────────────
def dma_state_series(close: pd.Series) -> pd.Series:
    """Vectorized 3-state trend over a close series (BULL / BEAR / NEUTRAL / None).

    BULL    = SMA10>SMA20 AND SMA20>SMA50 AND SMA10 rising (SMA10[t] > SMA10[t-1])
    BEAR    = SMA10<SMA20 AND SMA20<SMA50 AND SMA10 falling (SMA10[t] < SMA10[t-1])
    NEUTRAL = otherwise (incl. a 10/20 cross with the 50dma not yet aligned — the chop filter)
    None    = fewer than 50 bars available (SMA50 undefined)
    """
    c = close.astype(float)
    s10, s20, s50 = c.rolling(10).mean(), c.rolling(20).mean(), c.rolling(50).mean()
    rising = (s10 > s10.shift(1)).fillna(False)
    falling = (s10 < s10.shift(1)).fillna(False)
    bull = ((s10 > s20) & (s20 > s50) & rising).fillna(False)
    bear = ((s10 < s20) & (s20 < s50) & falling).fillna(False)
    out = pd.Series(index=c.index, dtype=object)
    out[:] = "NEUTRAL"
    out[bull] = "BULL"
    out[bear] = "BEAR"
    out[s50.isna()] = None                        # <50 bars -> undefined, not NEUTRAL
    return out


def _trend3(st: dict | None) -> str | None:
    """mtf.tf_state trend_dir -> the 3-state field. up->uptrend, down->downtrend, any range
    subtype (contracting/expanding/rectangle/range)->range, unknown->None."""
    if st is None:
        return None
    td = st.get("trend_dir")
    if td == "up":
        return "uptrend"
    if td == "down":
        return "downtrend"
    return "range" if td else None


# ── frame loading (enriched, loaded once, sliced per day) ───────────────────────
def _enriched(symbols: set[str]) -> dict:
    """load_bars + add_indicators for each symbol, once. Frames tf_state / group_scores /
    theme_board / RS all read. Silently drops symbols with no bars."""
    out: dict = {}
    for s in symbols:
        df = datastore.load_bars(s)
        if df is None or len(df) < 2:
            continue
        try:
            out[s] = indicators.add_indicators(df)
        except Exception:
            continue
    return out


def _universe() -> tuple[list[str], list[str], set[str], dict, set[str]]:
    """(themes, sectors_with_min, ticker_rows, theme_etf, all_frame_syms) for the curated Phase-1 set."""
    labels.load()
    themes = sorted(labels.all_themes())
    theme_members: set[str] = set()
    for t in themes:
        theme_members |= labels.tickers_in_theme(t)
    # sectors that clear the min-members floor (group_scores itself also enforces this)
    from collections import Counter
    secct = Counter(v for v in (labels._SECTOR or {}).values() if v)
    sectors = sorted(s for s, n in secct.items() if n >= config.GROUP_MIN_MEMBERS)
    sector_members = {s for s, v in (labels._SECTOR or {}).items() if v and secct[v] >= config.GROUP_MIN_MEMBERS}
    theme_etf = {t: market._group_etf("theme", t) for t in themes}
    theme_etf = {t: e for t, e in theme_etf.items() if e}
    # frames needed: every group member (for accurate group breadth) + tickers + futures + SPY + theme ETFs
    all_syms = set(theme_members) | sector_members | set(FUTURES) | {"SPY"} | set(theme_etf.values())
    ticker_rows = sorted(theme_members)          # per-ticker ROWS = theme members (locked scope)
    return themes, sectors, set(ticker_rows), theme_etf, all_syms


# ── one day's rows ──────────────────────────────────────────────────────────────
TF_TIMEOUT_S = 15   # a whole session (~1881 tickers) takes ~14s; a SINGLE ticker's tf_state is ~7ms,
                    # so >15s is a pathological hang (a trendlab/segment_chart loop on odd data). Cap it,
                    # skip that one ticker (null trend/health), keep the backfill moving. 2000x headroom.


def _sliced_state(frame: pd.DataFrame | None, day: pd.Timestamp,
                  timeout: float | None = None, label: str = "") -> dict | None:
    """tf_state as-of `day` (no look-ahead) — frame sliced to <= day, tf_state reads its last row.

    With `timeout` set AND running on the main thread, a hung tf_state is interrupted (SIGALRM) and
    returns None (logged) instead of freezing the whole backfill. Off the main thread (serve's daemon
    append) SIGALRM is unavailable, so the cap is skipped there — append processes one current day, low
    hang risk, and its thread is daemon/fail-open regardless."""
    if frame is None or not len(frame):
        return None
    d = frame.loc[:day]
    if len(d) < 2:
        return None
    import threading
    if timeout is None or threading.current_thread() is not threading.main_thread():
        return mtf.tf_state(d)
    import signal

    def _h(signum, fr):
        raise TimeoutError()
    old = signal.signal(signal.SIGALRM, _h)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return mtf.tf_state(d)
    except TimeoutError:
        print(f"[history] tf_state TIMEOUT >{timeout}s: {label or '?'} @ {day.date()} "
              f"-> null trend/health (skipped, backfill continues)", flush=True)
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _day_rows(day: pd.Timestamp, frames: dict, themes, sectors, ticker_rows, theme_etf,
              dma_cache: dict) -> tuple[list[dict], list[dict]]:
    """Build (entity_rows, leader_rows) for one settled session."""
    dstr = day.date().isoformat()
    ent: list[dict] = []
    lead: list[dict] = []

    def dma_at(sym: str):
        s = dma_cache.get(sym)
        if s is None:
            return None
        v = s.loc[:day]
        return v.iloc[-1] if len(v) else None

    # market — health only
    try:
        h = health_score.read(dstr)
        mh = h.get("score") if isinstance(h, dict) else None
    except Exception:
        mh = None
    ent.append({"date": dstr, "kind": "market", "id": "MARKET",
                "dma_state": None, "trend": None, "health": mh})

    # ES / NQ — own dma_state + trend + tf_state health
    for f in FUTURES:
        st = _sliced_state(frames.get(f), day, timeout=TF_TIMEOUT_S, label=f)
        ent.append({"date": dstr, "kind": "future", "id": f,
                    "dma_state": dma_at(f),
                    "trend": _trend3(st),
                    "health": (round(market._score_state(st), 3) if st else None)})

    # group scoreboards — ONE pass each for all groups (as-of settled)
    gs = {(r["kind"], r["name"]): r for r in market.group_scores(frames, live=False, as_of=day)}
    tb = {(r["kind"], r["name"]): r for r in market.theme_board(frames, as_of=day)}

    for kind, names in (("theme", themes), ("sector", sectors)):
        for name in names:
            g = gs.get((kind, name))
            b = tb.get((kind, name)) or {}
            if g is None:
                continue                          # group didn't clear min-members on this day
            row = {"date": dstr, "kind": kind, "id": name,
                   "dma_state": None, "trend": None,
                   "health": g.get("score"),
                   "score": g.get("score"), "pct_up": g.get("pct_up"),
                   "pct_50": g.get("pct_50"), "pct_200": g.get("pct_200"),
                   "rs": g.get("rs"), "rs_3m": g.get("rs_3m"), "rs_trend": g.get("rs_trend"),
                   "d1": b.get("d1"), "w1": b.get("w1"), "m1": b.get("m1"),
                   "m3": b.get("m3"), "ytd": b.get("ytd")}
            if kind == "theme":                   # dma/trend from the proxy ETF (NULL if none)
                etf = theme_etf.get(name)
                if etf:
                    st = _sliced_state(frames.get(etf), day, timeout=TF_TIMEOUT_S, label=etf)
                    row["dma_state"] = dma_at(etf)
                    row["trend"] = _trend3(st)
            ent.append(row)

    # per-ticker rows + theme-member leader rankings (RS = member ret_1m - SPY ret_1m, as-of)
    spy = frames.get("SPY")
    spy_r1 = None
    if spy is not None:
        sd = spy.loc[:day]
        if len(sd) and pd.notna(sd["ret_1m"].iloc[-1]):
            spy_r1 = float(sd["ret_1m"].iloc[-1])

    for sym in ticker_rows:
        if sym not in frames:                     # no bars for this member -> skip (no null junk row)
            continue
        st = _sliced_state(frames.get(sym), day, timeout=TF_TIMEOUT_S, label=sym)
        ent.append({"date": dstr, "kind": "ticker", "id": sym,
                    "dma_state": dma_at(sym),
                    "trend": _trend3(st),
                    "health": (round(market._score_state(st), 3) if st else None)})

    for t in themes:
        members = labels.tickers_in_theme(t)
        rows = []
        for sym in members:
            f = frames.get(sym)
            if f is None:
                continue
            fd = f.loc[:day]
            if not len(fd):
                continue
            r1 = fd["ret_1m"].iloc[-1]
            r3 = fd["ret_3m"].iloc[-1]
            if not pd.notna(r1):
                continue
            rs = float(r1) - (spy_r1 or 0.0)
            rows.append((sym, rs, float(r1), float(r3) if pd.notna(r3) else None))
        rows.sort(key=lambda x: -x[1])
        for rank, (sym, rs, r1, r3) in enumerate(rows, 1):
            lead.append({"date": dstr, "theme": t, "ticker": sym, "rank": rank,
                         "rs": round(rs, 2), "ret_1m": round(r1, 2),
                         "ret_3m": (round(r3, 2) if r3 is not None else None)})
    return ent, lead


# ── writing (per-year parquet, dedupe on the day) ───────────────────────────────
def _write_partition(rows: list[dict], base: Path, keys: list[str]) -> None:
    if not rows:
        return
    base.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for year, chunk in df.groupby(df["date"].str[:4]):
        path = base / f"{year}.parquet"
        if path.exists():
            old = pd.read_parquet(path)
            merged = pd.concat([old, chunk], ignore_index=True)
            merged = merged.drop_duplicates(subset=keys, keep="last")
        else:
            merged = chunk
        merged = merged.sort_values(keys).reset_index(drop=True)
        merged.to_parquet(path, index=False)


def _done_dates() -> set[str]:
    """Dates already written to the entities store (for --resume: skip completed sessions)."""
    if not ENT_DIR.exists():
        return set()
    done: set[str] = set()
    for p in ENT_DIR.glob("*.parquet"):
        try:
            done |= set(pd.read_parquet(p, columns=["date"])["date"].unique())
        except Exception:
            continue
    return done


def _sessions(start: str, end: str | None, frames: dict) -> list[pd.Timestamp]:
    """SPY's settled trading days in [start, end] drive the calendar (no weekends/holidays)."""
    spy = frames.get("SPY")
    if spy is None or not len(spy):
        return []
    idx = spy.index
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end) if end else idx[-1]
    return [d for d in idx if lo <= d <= hi]


def build(start: str, end: str | None = None, tag: str = "history_build",
          resume: bool = False, flush_every: int = 25) -> int:
    """Backfill entities + theme_leaders for every settled session in [start, end].

    Crash-safe: flushes to disk every `flush_every` sessions (and at each year boundary), so a
    kill/stall loses at most `flush_every` days of work. resume=True skips sessions already in the
    store — so after a stall you kill + relaunch with --resume and it continues from where it stopped.
    Writes are per-year parquet with dedupe, so a re-done boundary day is harmless."""
    import jobreport
    themes, sectors, ticker_rows, theme_etf, all_syms = _universe()
    frames = _enriched(all_syms)
    dma_cache = {s: dma_state_series(f["close"]) for s, f in frames.items()}
    days = _sessions(start, end, frames)
    if resume:
        done = _done_dates()
        days = [d for d in days if d.date().isoformat() not in done]
    ent_buf: list[dict] = []
    lead_buf: list[dict] = []
    cur_year: str | None = None
    since_flush = 0

    def flush():
        _write_partition(ent_buf, ENT_DIR, ["date", "kind", "id"])
        _write_partition(lead_buf, LEAD_DIR, ["date", "theme", "ticker"])
        ent_buf.clear(); lead_buf.clear()

    with jobreport.Reporter(tag, len(days)) as rep:
        for i, day in enumerate(days):
            yr = f"{day.year}"
            if cur_year is not None and yr != cur_year:   # year boundary -> flush the finished year
                flush(); since_flush = 0
            cur_year = yr
            ent, lead = _day_rows(day, frames, themes, sectors, ticker_rows, theme_etf, dma_cache)
            ent_buf.extend(ent); lead_buf.extend(lead); since_flush += 1
            if since_flush >= flush_every:                 # periodic crash-safe flush
                flush(); since_flush = 0
            rep.tick(i + 1, f"{day.date()} ({len(ent)} ent, {len(lead)} lead)")
        flush()                                            # final partial batch
    return len(days)


def append_day(date: str | None = None) -> int:
    """Append ONE settled session (default: latest settled). Idempotent (dedupe on keys).
    Wrapped fail-open by the serve hook — an exception here must never abort the daily update."""
    themes, sectors, ticker_rows, theme_etf, all_syms = _universe()
    frames = _enriched(all_syms)
    days = _sessions(date or "1900-01-01", date, frames) if date else None
    if not days:
        spy = frames.get("SPY")
        if spy is None or not len(spy):
            return 0
        days = [spy.index[-1]]
    day = days[-1]
    dma_cache = {s: dma_state_series(f["close"]) for s, f in frames.items()}
    ent, lead = _day_rows(day, frames, themes, sectors, ticker_rows, theme_etf, dma_cache)
    _write_partition(ent, ENT_DIR, ["date", "kind", "id"])
    _write_partition(lead, LEAD_DIR, ["date", "theme", "ticker"])
    return len(ent)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=(dt.date.today() - dt.timedelta(days=3 * 365)).isoformat())
    ap.add_argument("--end", default=None)
    ap.add_argument("--tag", default="history_build")
    ap.add_argument("--append", action="store_true", help="append one settled day instead of a backfill")
    ap.add_argument("--resume", action="store_true", help="skip sessions already in the store (restart-safe)")
    a = ap.parse_args()
    if a.append:
        print(f"appended {append_day(a.end)} entity rows")
    else:
        print(f"built {build(a.start, a.end, a.tag, resume=a.resume)} sessions -> {ENT_DIR}")
