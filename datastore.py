"""Local EOD store: one parquet per ticker, bulk init + cheap daily incremental updates.

Design choices:
- auto_adjust=True -> split/dividend adjusted OHLC (pattern logic needs adjusted data).
- Splits/dividends re-base yfinance's whole history, so a tail-merge would seam two price scales:
  update_daily compares the overlap days and refetches a re-based symbol's full history instead
  (and back-adjusts the raw Tiingo 5-min store via repair_tiingo_5m on split-sized factors).
- Resumable: already-cached tickers are skipped on init, so a crashed run just continues.
- A skip-list remembers symbols that return nothing (delisted/bad) so we stop re-asking.
"""
from __future__ import annotations

import datetime as dt
import os
import time
import uuid

import numpy as np
import pandas as pd

import config
import universe as uni

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

COLS = ["open", "high", "low", "close", "volume"]

# parquet if pyarrow/fastparquet present, otherwise pickle — same API either way
try:
    import pyarrow  # noqa: F401
    EXT = ".parquet"
except ImportError:
    try:
        import fastparquet  # noqa: F401
        EXT = ".parquet"
    except ImportError:
        EXT = ".pkl"


def _read(p):
    """Read a cache file. A file can be corrupted by a write that got interrupted mid-stream
    (process kill, crash, disk full) — _write() below is now atomic so that can't happen going
    forward, but defend against any file already in that state: warn and treat it as missing
    rather than letting one bad file take down a whole update/load_frames pass."""
    try:
        return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_pickle(p)
    except Exception as e:
        print(f"  WARNING: {p} is corrupted ({e}) — treating as missing; delete it to force a re-download")
        return None


def _write(df, p):
    """Atomic write: write to a temp file in the same directory, then os.replace() onto the final
    path. os.replace is atomic on the same filesystem, so a kill/crash mid-write can at worst leave
    an orphaned .tmp file — it can never truncate/corrupt the existing good file at `p`."""
    tmp = p.with_name(f"{p.stem}.{uuid.uuid4().hex[:8]}.tmp{p.suffix}")
    try:
        df.to_parquet(tmp) if p.suffix == ".parquet" else df.to_pickle(tmp)
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)


def list_symbols() -> list[str]:
    return sorted(p.stem for p in config.BARS_DIR.glob(f"*{EXT}"))


def _load_skiplist() -> set:
    if config.SKIPLIST_FILE.exists():
        return set(config.SKIPLIST_FILE.read_text().split())
    return set()


def _save_skiplist(s: set) -> None:
    config.SKIPLIST_FILE.write_text("\n".join(sorted(s)))


def bar_path(symbol: str):
    return config.BARS_DIR / f"{symbol}{EXT}"


def load_bars(symbol: str) -> pd.DataFrame | None:
    p = bar_path(symbol)
    if not p.exists():
        return None
    df = _read(p)
    return df if df is not None and len(df) else None


# ---- weekly bars: resampled from daily (W-FRI), cached in BARS_DIR_W -----------

def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    w = df.resample("W-FRI").agg({"open": "first", "high": "max", "low": "min",
                                  "close": "last", "volume": "sum"})
    w = w[w["close"].notna()]
    w.index.name = "date"
    return w


def bar_path_w(symbol: str):
    return config.BARS_DIR_W / f"{symbol}{EXT}"


def load_bars_w(symbol: str) -> pd.DataFrame | None:
    """Weekly frame: use the cached resample if it's at least as fresh as the daily file, else
    (re)build it from daily. The mtime check lets the EOD update skip rebuilding inactive weeklies —
    a stale one is rebuilt lazily the first time it is loaded (only the active set is ever loaded)."""
    p, dp = bar_path_w(symbol), bar_path(symbol)
    if p.exists() and (not dp.exists() or p.stat().st_mtime >= dp.stat().st_mtime):
        df = _read(p)
        if df is not None and len(df):
            return df
    daily = load_bars(symbol)
    if daily is None:
        return None
    w = resample_weekly(daily)
    config.BARS_DIR_W.mkdir(parents=True, exist_ok=True)
    _write(w, p)
    return w if len(w) else None


def build_weekly(symbols: list[str] | None = None) -> None:
    """(Re)build the weekly cache from the daily cache. Cheap — no downloads."""
    config.BARS_DIR_W.mkdir(parents=True, exist_ok=True)
    syms = symbols if symbols is not None else list_symbols()
    for sym in syms:
        daily = load_bars(sym)
        if daily is not None and len(daily):
            _write(resample_weekly(daily), bar_path_w(sym))


# ---- higher daily-derived timeframes (resampled from the daily cache on demand) ------

_OHLCV = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
DAILY_TFS = ["2D", "3D", "1W", "2W", "1M", "3M", "6M"]   # 1D is the raw daily frame
_DAILY_RULE = {"1W": "W-FRI", "2W": "2W-FRI", "1M": "ME", "3M": "QE", "6M": "2QE"}


def resample_daily(daily: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Daily OHLCV -> a higher daily-derived timeframe. 2D/3D group every N *trading* bars
    (anchored at the first bar, so bins are stable under backtest truncation); 1W..6M use calendar
    bins. Bar timestamp = the last day in the period (calendar) / chunk (count-based)."""
    if tf in ("2D", "3D"):
        n = int(tf[0])
        g = np.arange(len(daily)) // n
        r = daily.groupby(g).agg(_OHLCV)
        last = [min((k + 1) * n - 1, len(daily) - 1) for k in range(int(g[-1]) + 1)] if len(daily) else []
        r.index = daily.index[last]
    else:
        r = daily.resample(_DAILY_RULE[tf]).agg(_OHLCV)
    r = r[r["close"].notna()]
    r.index.name = "date"
    return r


# ---- intraday: download 60d of futures 5m bars, build the rest from them ---------------
# NOTE: the "15m" naming below (bar_path_15m/load_bars_15m/fetch_intraday_15m/BARS_DIR_15M) is LEGACY.
# The futures base is now 5m (equities were always 5m via Tiingo); resample_intraday auto-infers width.

INTRA_TFS = ["15m", "30m", "1h", "2h", "4h", "8h", "12h"]
# _INTRA_MIN carries "5m" (the live base for BOTH equities and futures) so resample_intraday can map it;
# INTRA_TFS (the list serve/mtf iterate to decide "is this an intraday TF") stays at 15m+ — 5m is the
# base itself and is handled by the explicit `or tf=="5m"` guards in serve, not this list.
_INTRA_MIN = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240, "8h": 480, "12h": 720}
_RTH_START = pd.Timedelta(hours=9, minutes=30)
_SESSION_MIN = 390      # US regular session = 6.5h = 390 min (26 x 15m bars)


def bar_path_15m(symbol: str):     # LEGACY name: the store now holds 5m futures bars, not 15m
    return config.BARS_DIR_15M / f"{symbol}{EXT}"


def load_bars_15m(symbol: str) -> pd.DataFrame | None:   # LEGACY name: futures 5m base (see module note)
    p = bar_path_15m(symbol)
    if not p.exists():
        return None
    df = _read(p)
    if df is None or not len(df):
        return None
    f64 = df.select_dtypes("float64").columns          # halve RAM: float64 -> float32 (ample precision)
    if len(f64):
        df[f64] = df[f64].astype("float32")
    return df


def bar_path_5m(symbol: str):
    return config.DATA_DIR / "tiingo" / "bars_5min" / f"{symbol}{EXT}"


BARS_DIR_5M_LIVE = config.DATA_DIR / "tiingo" / "bars_5min_live"   # today's RAW-IEX bars, written live


def bar_path_5m_live(symbol: str):
    return BARS_DIR_5M_LIVE / f"{symbol}{EXT}"


def floor_intraday_grid(df: pd.DataFrame, minutes: int = 5) -> pd.DataFrame:
    """Floor an intraday frame's index to the `minutes` grid and collapse any resulting same-slot rows to
    ONE bar, keeping the MAX-VOLUME row (the real print). The Tiingo IEX feed (iex_intraday -> norm_intraday)
    returns 5-min bars whose timestamps carry sub-microsecond FLOAT NOISE (e.g. 09:30:00.000000629), which is
    NOT on the 5-min grid — so a naive merge with the poller's clean :00 bars keeps BOTH and stacks 2-4
    duplicate bars per slot (double-counted volume). Flooring aligns every source to the grid; the max-volume
    keep resolves seed-vs-poller overlaps to the more-complete bar. Used by the live-session seed/merge."""
    if df is None or not len(df):
        return df
    out = df.copy()
    out.index = pd.DatetimeIndex(out.index).floor(f"{minutes}min")
    if out.index.has_duplicates:
        if "volume" in out.columns:
            out = out.sort_values("volume")            # ascending -> the max-volume row is last per slot
        out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


def write_5m_live(symbol: str, df: pd.DataFrame) -> None:
    """Persist today's RAW-IEX 5-min bars to the live OVERLAY so the store is crash-safe and every reader
    (load_bars_5m) sees the CURRENT session — no consumer waits for the EOD backfill. `df` volume MUST be
    raw IEX (uncalibrated), matching the main store's basis. Atomic write (temp + os.replace)."""
    if df is None or not len(df):
        return
    df = floor_intraday_grid(df)                        # belt-and-suspenders: every overlay write is grid-clean
    BARS_DIR_5M_LIVE.mkdir(parents=True, exist_ok=True)  # (also purges any residual off-grid rows on rewrite)
    keep = [c for c in COLS if c in df.columns]
    df = df[keep].copy()
    p = bar_path_5m_live(symbol)
    # MERGE with the existing overlay rather than FULL-REPLACE, so a writer holding fewer / staler bars (the
    # poller _flush vs the background seed vs a forced seed) can't CLOBBER a more-complete overlay (BUGS #51).
    # Union by 5-min slot keeping the MAX-volume row per slot: a filled bar beats a 0-volume placeholder, and
    # floor_intraday_grid dedups. Only affects the crash-safe live overlay (fold_5m_live still keep='last').
    if p.exists():
        try:
            old = _read(p)
            if old is not None and len(old):
                old = old[[c for c in keep if c in old.columns]]
                df = floor_intraday_grid(pd.concat([old, df]))
        except Exception:
            pass
    _write(df, p)


def fold_5m_live(symbol: str | None = None) -> int:
    """EOD: fold the live overlay's (now-settled) bars into the main 5m store and clear the overlay. Per
    symbol idempotent. Returns #symbols folded. Run after the close, before the next session."""
    if not BARS_DIR_5M_LIVE.exists():
        return 0
    syms = [symbol] if symbol else sorted(p.stem for p in BARS_DIR_5M_LIVE.glob(f"*{EXT}"))
    n = 0
    for s in syms:
        lp = bar_path_5m_live(s)
        if not lp.exists():
            continue
        live = _read(lp)
        if live is not None and len(live):
            main = _read(bar_path_5m(s)) if bar_path_5m(s).exists() else None
            merged = live if (main is None or not len(main)) else pd.concat([main, live])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            bar_path_5m(s).parent.mkdir(parents=True, exist_ok=True)
            _write(merged, bar_path_5m(s))
            n += 1
        lp.unlink(missing_ok=True)
    return n


def load_bars_5m(symbol: str, calibrate: bool = True, raw_from=None) -> pd.DataFrame | None:
    """Live intraday BASE frame: the raw Tiingo 5-min RTH store (2017+, 09:30-15:55, IEX prices +
    IEX-sample volume). IEX prints only 2-6% of the tape and the share varies per symbol, so
    `calibrate` (default) scales `volume` by the per-symbol vol_factor (~x22, data/iex_vol_factor.json)
    to put it on CONSOLIDATED scale — the SAME basis the legacy yfinance 15m frames carry. That makes a
    5m frame a drop-in for the 15m RAM store: every downstream RVOL/pace read that assumes
    'frame volume = consolidated' (ep_news.rvol_leaders / live_radar replay path) keeps working, and
    resample_intraday's volume sums stay on consolidated scale. Prices are IEX (untouched, <4bp vs
    consolidated). float32 to halve RAM (parity with load_bars_15m). Absolute volume magnitude is a
    calibrated approximation, not a measurement — label it as such wherever it drives internals.

    Reads the settled main store MERGED with today's live OVERLAY (bars_5min_live/, written by the poller +
    Tiingo gap-backfill, RAW-IEX basis, overlay wins on overlap) — so EVERY reader (app or lab-in-app) sees
    the current session with no per-consumer live plumbing, and it survives a crash.

    `raw_from` (a date/Timestamp): bars at/after it are left RAW (uncalibrated). The LIVE seed passes
    today — FRAMES5's today-portion must stay on the poller's RAW-IEX basis. Calibrating the overlay's
    today-bars into RAM let _flush_live_overlay persist x22 volume back to the overlay (whose max-volume
    merge keeps the inflated row forever), and each mid-session restart compounded another x22; the EOD
    fold then wrote it into the main store PERMANENTLY (2026-07-07/08 corruption, found live 2026-07-09:
    AAPL's 07-08 day-sum read 174 TRILLION shares ~ x22^5.7)."""
    p = bar_path_5m(symbol)
    df = _read(p) if p.exists() else None
    lp = bar_path_5m_live(symbol)
    if lp.exists():
        live = _read(lp)
        if live is not None and len(live):
            if df is None or not len(df):
                df = live
            else:
                df = pd.concat([df, live])
                df = df[~df.index.duplicated(keep="last")].sort_index()
    if df is None or not len(df):
        return None
    if calibrate and "volume" in df.columns:
        try:
            import ep_news
            fac = float(ep_news.vol_factor().get(symbol, 22.0))
        except Exception:
            fac = 22.0
        df = df.copy()
        df["volume"] = df["volume"].astype("float64")
        m = (df.index < pd.Timestamp(raw_from)) if raw_from is not None else slice(None)
        df.loc[m, "volume"] = df.loc[m, "volume"] * fac
    f64 = df.select_dtypes("float64").columns          # halve RAM: float64 -> float32 (ample precision)
    if len(f64):
        df[f64] = df[f64].astype("float32")
    return df


def _normalize_intraday(raw: pd.DataFrame, futures: bool = False) -> pd.DataFrame:
    """Like _normalize but keeps the intraday DatetimeIndex (time-of-day), converts to US/Eastern
    wall-clock, and (for equities) filters to the regular session [09:30, 16:00). Futures trade 24h,
    so their bars are kept as-is."""
    df = _flatten_cols(raw.copy())
    df = df[[c for c in COLS if c in df.columns]].dropna(how="all")
    if "close" in df.columns:
        df = df[df["close"].notna()]
    idx = pd.to_datetime(df.index)
    df.index = idx.tz_convert("America/New_York").tz_localize(None) if idx.tz is not None else idx
    df.index.name = "datetime"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if futures:
        return df                                               # 24h — no session filter
    t = df.index.time
    return df[(t >= dt.time(9, 30)) & (t < dt.time(16, 0))]      # RTH; bars labeled by open time


def _download_intraday_batch(symbols: list[str]) -> dict:
    """One multi-ticker 5m/60-day yfinance call -> {symbol: intraday df} (per-symbol RTH/24h). Reached
    only via fetch_intraday_15m, which serve calls with the FUTURES-only list, so this 5m pull is
    futures-scoped (equities have their own Tiingo 5m base + download_5m_today)."""
    out = {}
    data = yf.download(tickers=" ".join(symbols), period="60d", interval="5m",
                       group_by="ticker", auto_adjust=True, threads=True, progress=False, prepost=False)
    if data is None or data.empty:
        return out
    if len(symbols) == 1:
        df = _normalize_intraday(data, uni.is_future(symbols[0]))
        return {symbols[0]: df} if len(df) else {}
    for sym in symbols:
        if sym in data.columns.get_level_values(0):
            sub = data[sym].dropna(how="all")
            if len(sub):
                df = _normalize_intraday(sub, uni.is_future(sym))
                if len(df):
                    out[sym] = df
    return out


def fetch_intraday_15m(symbols: list[str], progress=None) -> int:
    """Download ~60 days of 5m bars for `symbols` (FUTURES; 24h), one parquet each (full replace —
    intraday is a rolling window). LEGACY name (`_15m`): the interval is 5m now. progress(done, total,
    sym) is called per ticker. Returns #written."""
    if yf is None:
        return 0
    config.BARS_DIR_15M.mkdir(parents=True, exist_ok=True)
    total, written = len(symbols), 0
    for i in range(0, total, config.BATCH_SIZE):
        batch = symbols[i:i + config.BATCH_SIZE]
        try:
            got = _download_intraday_batch(batch)
        except Exception:
            time.sleep(5)
            try:
                got = _download_intraday_batch(batch)
            except Exception:
                got = {}
        for j, sym in enumerate(batch):
            df = got.get(sym)
            if df is not None and len(df):
                _write(df, bar_path_15m(sym))
                written += 1
            if progress:
                progress(i + j + 1, total, sym)
        time.sleep(1.0)
    return written


def download_5m_today(symbols: list[str]) -> dict:
    """Bulk yfinance 5-min RTH bars for TODAY (09:30->now), consolidated volume — used to seed the live 5m
    store with the session-so-far. The Tiingo 5m DISK store only backfills EOD and the live poller has bars
    only since it started, so today's early session exists nowhere until this fills it. period='1d' keeps it
    cheap (one batched download); RTH-filtered + sliced to today. Returns {sym: today 5m df}. Equities only."""
    if yf is None:
        return {}
    out: dict = {}
    today = dt.date.today()
    for i in range(0, len(symbols), config.BATCH_SIZE):
        batch = symbols[i:i + config.BATCH_SIZE]
        try:
            data = yf.download(tickers=" ".join(batch), period="1d", interval="5m", group_by="ticker",
                               auto_adjust=True, threads=True, progress=False, prepost=False)
        except Exception:
            time.sleep(3)
            continue
        if data is None or data.empty:
            continue
        for sym in batch:
            try:
                sub = data[sym].dropna(how="all") if len(batch) > 1 else data
                df = _normalize_intraday(sub, futures=False)          # RTH filter (equities)
                df = df[df.index.normalize() == pd.Timestamp(today)]  # today only
                if len(df):
                    out[sym] = df
            except Exception:
                continue
        time.sleep(0.5)
    return out


def _infer_base_min(idx) -> int:
    """Infer an intraday frame's base bar-width in minutes from its index — the SMALLEST positive gap
    between consecutive bars (robust: lunch/overnight/weekend gaps are larger, so the minimum is the
    true grid step). Sub-30s gaps are dropped as data glitches (a stray near-duplicate timestamp would
    otherwise round to 0 and divide-by-zero downstream); the result floors at 1. Falls back to 15
    (legacy default) only when width can't be inferred; real 5m futures/equity frames infer 5."""
    if idx is None or len(idx) < 2:
        return 15
    gaps = np.diff(idx.values) / np.timedelta64(1, "m")
    gaps = gaps[gaps >= 0.5]                          # >=30s: ignore near-duplicate-timestamp glitches
    return max(1, int(round(float(gaps.min())))) if len(gaps) else 15


def resample_intraday(base: pd.DataFrame, tf: str, futures: bool = False,
                      base_min: int | None = None) -> pd.DataFrame:
    """Build a higher intraday TF from a fixed-width intraday base (`base_min` minutes: 15m legacy,
    or 5m for the Tiingo live base). For equities, 30m/1h/2h/4h are session-anchored (bucket from
    09:30, reset daily) and 8h/12h accumulate continuously. Futures have no RTH session, so EVERY TF
    uses continuous N-bar grouping. Bar timestamp = the first base bar's time of each group.

    Session-anchoring is base-agnostic (it buckets by absolute clock time), so it is CORRECT for any
    base. Only the base-TF early-return and the continuous 8h/12h grouping depend on `base_min`, which
    is AUTO-INFERRED from the frame's own bar spacing when not passed (15m frame -> 15, 5m frame -> 5).
    So every caller is correct for its actual base with no signature change, and this fixes the latent
    8h/12h count-based mislabel on a 5m frame. Pass base_min explicitly to override the inference."""
    if base is None or not len(base):
        return base
    if base_min is None:
        base_min = _infer_base_min(base.index)
    mins = _INTRA_MIN.get(tf, base_min)
    if mins <= base_min:                                      # the base TF (or finer) -> nothing to build
        return base
    idx = base.index
    if mins <= _SESSION_MIN and not futures:                  # fits in a session -> anchor to 09:30 daily
        day_ord = idx.normalize().values.astype("datetime64[D]").astype(np.int64)
        since_open = (idx - idx.normalize()) - _RTH_START     # TimedeltaIndex from 09:30
        bucket = np.asarray(since_open // pd.Timedelta(minutes=mins), dtype=np.int64)
        gid = day_ord * 1000 + bucket
    else:                                                     # continuous N-bar groups across sessions
        n = mins // base_min
        gid = np.arange(len(base)) // n
    _, first = np.unique(gid, return_index=True)              # gid is non-decreasing (time-sorted)
    r = base.groupby(gid, sort=True).agg(_OHLCV)
    r.index = idx[np.sort(first)]
    r = r[r["close"].notna()]
    r.index.name = "datetime"
    return r.sort_index()


def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance returns MultiIndex columns (field, ticker) for single-symbol downloads — collapse to
    the OHLCV field level so single-symbol fetches normalize correctly."""
    if isinstance(df.columns, pd.MultiIndex):
        for lvl in range(df.columns.nlevels):
            if {str(v).lower() for v in df.columns.get_level_values(lvl)} & set(COLS):
                df.columns = [str(v).lower() for v in df.columns.get_level_values(lvl)]
                return df
    df.columns = [str(c).lower() for c in df.columns]
    return df


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    df = _flatten_cols(raw.copy())
    df = df[[c for c in COLS if c in df.columns]].dropna(how="all")
    if "close" in df.columns:
        df = df[df["close"].notna()]   # yfinance can return volume-only rows (NaN OHLC) for a
                                       # provisional/recent day — a bar with no close is unusable
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _settled_end() -> dt.date | None:
    """yfinance `end` clamp so an update NEVER writes today's PARTIAL daily bar: while the US
    session is open (or in the first 15 min after the close, before the closing auction settles),
    stop at yesterday (end is exclusive). After that, None = include today's settled bar."""
    try:
        from zoneinfo import ZoneInfo
        now = dt.datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return None
    if now.weekday() < 5 and dt.time(9, 30) <= now.time() < dt.time(16, 15):
        return now.date()
    return None


def _adjustment_seam(old: pd.DataFrame | None, fresh: pd.DataFrame, tol: float = 0.005) -> float | None:
    """yfinance back-adjusts the WHOLE history on every split/dividend, but update_daily only pulls
    the tail — after an adjustment event the stored history and the fresh tail are on different
    price scales, and a plain concat would print a fake overnight cliff into the store. The few-day
    overlap we already download gives free detection: on shared dates both series must agree.
    Returns the median fresh/stored close ratio on the disagreeing dates (split: far from 1,
    dividend: just under 1) when they differ by more than `tol`, else None."""
    if old is None or fresh is None or not len(fresh):
        return None
    common = old.index.intersection(fresh.index)
    if not len(common):
        return None
    r = (fresh.loc[common, "close"].astype(float) / old.loc[common, "close"].astype(float)).dropna()
    bad = r[(r - 1.0).abs() > tol]
    return float(bad.median()) if len(bad) else None


def repair_tiingo_5m(sym: str) -> bool:
    """Back-adjust the raw (as-traded) Tiingo 5-min store for splits, TradingView-style. The daily
    store is yfinance-adjusted, but the 5m store keeps raw IEX prices, so a split leaves a REAL
    overnight cliff in it. Compare each session's last 5m close against the adjusted daily close:
    that per-session ratio sits near 1 on the current scale, drifts only slowly with dividend
    adjustments, and JUMPS by the split factor across a seam. Rescale every pre-seam segment onto
    the final segment's scale (price *= step, volume /= step), then — if the whole store's tail
    still disagrees with the daily close (store went stale before a split) — rescale everything.
    Returns True if the file was rewritten."""
    path = config.DATA_DIR / "tiingo" / "bars_5min" / f"{sym}{EXT}"
    if not path.exists():
        return False
    df = _read(path)
    daily = load_bars(sym)
    if df is None or not len(df) or daily is None or not len(daily):
        return False
    sess = df["close"].groupby(df.index.normalize()).last()
    r = (daily["close"].reindex(sess.index).astype(float) / sess.astype(float)).dropna()
    if len(r) < 2:
        return False
    out, changed = df, False
    steps = (r / r.shift(1) - 1.0).abs()
    seams = list(r.index[steps > 0.20])                      # scale jumps ≫ IEX-close noise / div drift
    if seams:
        out = df.copy()
        ref = float(r[r.index >= seams[-1]].median())        # final segment = the store's newest scale
        bounds = [r.index[0]] + seams
        days = out.index.normalize()
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            f = float(r[(r.index >= lo) & (r.index < hi)].median()) / ref
            if abs(f - 1.0) < 0.05:
                continue
            m = (days >= lo) & (days < hi)
            for c in ("open", "high", "low", "close"):
                out.loc[m, c] = out.loc[m, c].astype(float) * f
            out.loc[m, "volume"] = out.loc[m, "volume"].astype(float) / f
            changed = True
    # stale-store case: a split AFTER the last stored 5m bar leaves no in-window seam, but recent
    # adjusted-daily closes ≈ current raw scale, so a big tail deviation means the whole store
    # (post-seam-repair, all one scale) needs rescaling onto the current scale.
    f = float(r.iloc[-20:].median())
    if abs(f - 1.0) > 0.20:
        out = out.copy() if out is df else out
        for c in ("open", "high", "low", "close"):
            out[c] = out[c].astype(float) * f
        out["volume"] = out["volume"].astype(float) / f
        changed = True
    if changed:
        _write(out, path)
    return changed


def _download_batch(symbols: list[str], start: dt.date, end: dt.date | None = None) -> dict:
    """One multi-ticker yfinance call -> {symbol: df}."""
    out = {}
    data = yf.download(
        tickers=" ".join(symbols), start=str(start), end=str(end) if end else None,
        group_by="ticker", auto_adjust=True, threads=True, progress=False,
    )
    if data is None or data.empty:
        return out
    if len(symbols) == 1:
        out[symbols[0]] = _normalize(data)
        return out
    for sym in symbols:
        if sym in data.columns.get_level_values(0):
            sub = data[sym].dropna(how="all")
            if len(sub):
                out[sym] = _normalize(sub)
    return out


def init_history(max_tickers: int | None = None) -> None:
    """One-time bulk pull. Safe to re-run: skips what's already cached."""
    if yf is None:
        raise SystemExit("pip install yfinance")
    config.BARS_DIR.mkdir(parents=True, exist_ok=True)
    u = uni.load_universe()
    skip = _load_skiplist()
    todo = [s for s in u["yf_symbol"] if s not in skip and not bar_path(s).exists()]
    if max_tickers:
        todo = todo[:max_tickers]
    start = dt.date.today() - dt.timedelta(days=int(config.HISTORY_YEARS * 365.25))
    print(f"init: {len(todo)} tickers to fetch (start={start})")

    for i in range(0, len(todo), config.BATCH_SIZE):
        batch = todo[i:i + config.BATCH_SIZE]
        try:
            got = _download_batch(batch, start)
        except Exception as e:  # network hiccup: wait & retry once
            print(f"  batch {i}: {e} — retrying in 30s")
            time.sleep(30)
            try:
                got = _download_batch(batch, start)
            except Exception as e2:
                print(f"  batch {i}: failed twice ({e2}), skipping batch")
                continue
        for sym in batch:
            df = got.get(sym)
            if df is None or len(df) < 5:
                skip.add(sym)
            else:
                _write(df, bar_path(sym))
        _save_skiplist(skip)
        done = min(i + config.BATCH_SIZE, len(todo))
        print(f"  {done}/{len(todo)}  (skiplist={len(skip)})")
        time.sleep(1.0)  # be polite, avoid rate limiting
    print("building weekly cache ...")
    build_weekly()
    print("init complete.")


def app_critical_etfs() -> set[str]:
    """ETFs the app's panels structurally depend on — theme reps (build_labels.THEME_ETFS), sectors
    (config.AWARE_SECTOR_ETFS), rotation legs (config.RATIO_BASKET, equities only) and the indices.
    The daily update ALWAYS covers these even when a thin thematic ETF dips below the active-universe
    liquidity floor, so its close can't go stale — a stale close made open_radar print phantom
    overnight gaps and the themes panel read a week old (BUGS 2026-07-09). Futures legs (=F) live in a
    separate store and are excluded. build_labels imports datastore, so import it lazily here."""
    etfs: set[str] = {"SPY", "QQQ", "IWM", "DIA"}
    try:
        import build_labels
        for v in build_labels.THEME_ETFS.values():
            etfs.update(v if isinstance(v, (list, tuple, set)) else [v])
    except Exception:
        pass
    etfs.update(getattr(config, "AWARE_SECTOR_ETFS", []))
    for leg in getattr(config, "RATIO_BASKET", []):
        for x in (leg[0], leg[1]):
            if isinstance(x, str) and not x.endswith("=F"):
                etfs.add(x)
    return etfs


def update_daily(full: bool = False) -> None:
    """Fetch the missing tail for the cached tickers. Normal/daily use (full=False) updates the active
    universe PLUS the app-critical ETFs (app_critical_etfs — always covered so theme/rotation reads
    never go stale). The weekly full refresh (full=True) updates every universe name so membership can
    be re-evaluated (run build_active_universe.py after)."""
    if yf is None:
        raise SystemExit("pip install yfinance")
    config.BARS_DIR.mkdir(parents=True, exist_ok=True)
    u = uni.load_universe()
    skip = _load_skiplist()
    symbols = [s for s in u["yf_symbol"] if s not in skip]
    app_etfs = app_critical_etfs()
    # ALWAYS-COVER set: structural ETFs + the =F futures. Futures are yfinance dailies (uni.FUTURES) but
    # were covered ONLY by the weekly full=True run (app_critical_etfs excludes them as "separate store"
    # — true for their intraday 5m base, NOT for the daily store data/bars/ that FRAMES + charts read).
    # A missed/failed weekly run left every future's daily chart ~26 days stale (Amir 2026-07-28); pull
    # them every daily run like the ETFs. Daily-only — the intraday futures store stays on its own path.
    always = app_etfs | uni.FUTURES_SYMBOLS
    if not full:
        active = uni.active_symbols()                  # daily: the active (liquid) universe ...
        if active is not None:
            keep = set(active) | always               # ... PLUS structural ETFs + futures (never drop)
            symbols = [s for s in symbols if s in keep]
    for s in sorted(always - set(symbols)):            # always-cover names missing from the base universe
        if s not in skip:                              # (theme-rep ETFs like BTF/FOTO, futures) — seed them
            symbols.append(s)

    # global last date across cache decides how far back the pull needs to go
    last_dates = []
    cached = []
    new = []
    for s in symbols:
        df = load_bars(s)
        if df is None:
            new.append(s)
        else:
            cached.append(s)
            last_dates.append(df.index[-1])
    if not cached:
        raise SystemExit("cache empty — run init first")

    oldest_tail = min(last_dates).date()
    start = oldest_tail - dt.timedelta(days=4)  # small overlap, dedupe below
    end = _settled_end()                        # mid-session: settled bars only, never today's partial
    if end:
        print(f"update: US session open — clamping to settled bars (< {end})")
    print(f"update: {len(cached)} cached (pull from {start}), {len(new)} new symbols")

    rebased = []   # symbols whose history was re-based (split/dividend) since our last pull
    for i in range(0, len(cached), config.BATCH_SIZE):
        batch = cached[i:i + config.BATCH_SIZE]
        try:
            got = _download_batch(batch, start, end)
        except Exception as e:
            print(f"  batch {i}: {e} — skipping this run")
            continue
        for sym, fresh in got.items():
            old = load_bars(sym)
            fac = _adjustment_seam(old, fresh)
            if fac is not None:
                rebased.append((sym, fac))   # merging two price scales would print a fake cliff
                continue
            merged = pd.concat([old, fresh])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            merged = merged[merged["close"].notna()]   # never keep a no-close bar (also repairs old)
            _write(merged, bar_path(sym))
        print(f"  {min(i + config.BATCH_SIZE, len(cached))}/{len(cached)}")
        time.sleep(0.5)

    if rebased:
        # a split/dividend re-based these histories on yfinance's side: the tail can't be merged,
        # the whole (cheap, per-symbol) history must be refetched at the new scale. If a batch
        # fails, the stored file keeps its old scale and the seam re-triggers on the next update.
        syms = [s for s, _ in rebased]
        print(f"update: {len(rebased)} re-based (split/dividend) — refetching full history: "
              + " ".join(syms[:15]) + (" ..." if len(syms) > 15 else ""))
        full_start = dt.date.today() - dt.timedelta(days=int(config.HISTORY_YEARS * 365.25))
        for i in range(0, len(syms), config.BATCH_SIZE):
            try:
                got = _download_batch(syms[i:i + config.BATCH_SIZE], full_start, end)
            except Exception as e:
                print(f"  refetch batch {i}: {e} — will self-heal on the next update")
                continue
            for sym, df in got.items():
                if len(df) >= 5:
                    _write(df, bar_path(sym))
        for sym, fac in rebased:
            if abs(fac - 1.0) > 0.03:        # split-sized: the raw 5-min store has a real cliff too
                try:
                    if repair_tiingo_5m(sym):
                        print(f"  {sym}: Tiingo 5-min store back-adjusted (factor {fac:.4g})")
                except Exception as e:
                    print(f"  {sym}: 5m repair failed: {e}")

    # brand-new listings get full (short) history
    if new:
        full_start = dt.date.today() - dt.timedelta(days=int(config.HISTORY_YEARS * 365.25))
        for i in range(0, len(new), config.BATCH_SIZE):
            batch = new[i:i + config.BATCH_SIZE]
            try:
                got = _download_batch(batch, full_start, end)
            except Exception:
                continue
            for sym in batch:
                df = got.get(sym)
                if df is None or len(df) < 5:
                    skip.add(sym)
                else:
                    _write(df, bar_path(sym))
        _save_skiplist(skip)
    # only the active universe's weeklies are ever loaded; the rest rebuild lazily on demand (mtime
    # check in load_bars_w), so skip rebuilding ~4.5k inactive weeklies every EOD update.
    active = uni.active_symbols()
    wk = sorted(active) if active is not None else None
    print(f"rebuilding weekly cache ({len(wk) if wk else 'all'} active) ...")
    build_weekly(wk)
    update_macro()
    print("update complete.")


def update_macro() -> None:
    """Fetch/refresh the market-awareness macro symbols (config.MACRO_SYMBOLS: ^VIX, ^VXN, DX-Y.NYB,
    ^TNX, TLT) into data/bars/ alongside the equity cache, so load_bars() serves them like any ticker.
    Merges onto whatever history is cached (or pulls HISTORY_YEARS on first run). Never fatal."""
    if yf is None or not getattr(config, "MACRO_SYMBOLS", None):
        return
    config.BARS_DIR.mkdir(parents=True, exist_ok=True)
    for sym in config.MACRO_SYMBOLS:
        old = load_bars(sym)
        start = (old.index[-1].date() - dt.timedelta(days=4)) if old is not None else \
            (dt.date.today() - dt.timedelta(days=int(config.HISTORY_YEARS * 365.25)))
        try:
            got = _download_batch([sym], start, _settled_end())
            fresh = got.get(sym)
        except Exception as e:
            print(f"  macro {sym}: {e}")
            continue
        if fresh is None or not len(fresh):
            continue
        if old is not None and _adjustment_seam(old, fresh) is not None:
            # re-based (e.g. a TLT distribution): drop the stored scale, pull the full history
            try:
                full_start = dt.date.today() - dt.timedelta(days=int(config.HISTORY_YEARS * 365.25))
                fresh, old = _download_batch([sym], full_start, _settled_end()).get(sym), None
            except Exception as e:
                print(f"  macro {sym}: refetch failed ({e})")
                continue
            if fresh is None or not len(fresh):
                continue
        merged = fresh if old is None else pd.concat([old, fresh])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        merged = merged[merged["close"].notna()]
        _write(merged, bar_path(sym))


def backfill_history(symbols: list[str], years: int) -> None:
    """Re-download a deep `years` window for `symbols` and REPLACE their cache files (clean — no
    adjustment seam). One-time deepening of the active universe + futures so high TFs have enough bars."""
    if yf is None:
        raise SystemExit("pip install yfinance")
    config.BARS_DIR.mkdir(parents=True, exist_ok=True)
    start = dt.date.today() - dt.timedelta(days=int(years * 365.25))
    print(f"backfill: {len(symbols)} symbols from {start}")
    for i in range(0, len(symbols), config.BATCH_SIZE):
        batch = symbols[i:i + config.BATCH_SIZE]
        try:
            got = _download_batch(batch, start)
        except Exception as e:
            print(f"  batch {i}: {e} — retry in 30s")
            time.sleep(30)
            try:
                got = _download_batch(batch, start)
            except Exception as e2:
                print(f"  batch {i}: failed twice ({e2})")
                continue
        for sym in batch:
            df = got.get(sym)
            if df is not None and len(df) >= 5:
                _write(df, bar_path(sym))      # replace, not merge
        print(f"  {min(i + config.BATCH_SIZE, len(symbols))}/{len(symbols)}")
        time.sleep(1.0)
    print("rebuilding weekly cache ...")
    build_weekly()
    print("backfill complete.")
