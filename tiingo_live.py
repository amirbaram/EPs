"""Tiingo real-time poller — minute-fresh live data for the whole active universe.

Every LIVE_POLL_SEC during US market hours, pull bulk IEX quotes (~100 tickers/request, so the
full universe is ~17 requests/cycle ≈ 10% of the Power hourly budget at 60s cycles) and fold
each quote into the FORMING 5-minute bar of the RAM store (serve.FRAMES5 — the single live
intraday base since Phase 3c). Everything downstream — the awareness map, internals, charts,
the EP radar — sees minute-fresh 5m data with no refactor. Equities only: IEX has no futures
(they keep the yfinance-15m path).

Semantics: the forming bar is PARTIAL by design (that is what "live" means); the replay /
validator paths are untouched — they read the on-disk store through pit.py's strict truncation.
"""

from __future__ import annotations

import datetime as dt
import json
import threading
import time
import urllib.parse
import urllib.request

import pandas as pd

import config

_KEY_FILE = config.DATA_DIR / "tiingo_key.txt"
CHUNK = 100                    # tickers per bulk quote request
STATE = {"running": False, "last_poll": None, "symbols": 0, "errors": 0, "cycles": 0,
         "phase": None, "vol_ref_sum": None, "vol_frozen_cycles": 0, "feed_degraded": False}
# Feed-freeze detector (2026-07-10): a small always-liquid basket whose IEX day-cumulative volume MUST
# grow every RTH cycle. When it stops advancing the trade-data feed is frozen (07-10 open: `volume`
# stuck at placeholder values ~15 min while price flowed) -> DAYVOL can't accumulate -> the rvol/EP
# boards go correctly EMPTY. We flag that so an empty open board reads as "feed warming", not "quiet".
_FEED_REF = ("SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "AMZN")
PREMARKET: dict = {}           # sym -> {last, volume, ts} captured 08:00-09:30 ET (never folded into bars)
DAYVOL: dict = {}              # sym -> latest IEX day-cumulative volume from the quote (raw, uncalibrated)
_PM_DATE = {"d": None}
_DAYVOL_DATE = {"d": None}     # session-date guard: DAYVOL MUST reset each day or it serves a prior
#                                session's full-day volume (a name with no fresh quote keeps its stale
#                                value via the `if vol:` guard) -> cum_rvol blows up 1000x (BUGS 2026-07-08)


def _roll_dayvol(today: str) -> None:
    """Clear the day-volume ledger on a session rollover. Called from BOTH quote handlers so it fires
    whichever runs first today (pre-market capture or the RTH poll)."""
    if _DAYVOL_DATE["d"] != today:
        DAYVOL.clear()
        _DAYVOL_DATE["d"] = today
        STATE.update(vol_ref_sum=None, vol_frozen_cycles=0, feed_degraded=False)  # feed-freeze tracker


def _note_feed_liveness() -> None:
    """Feed-freeze detector, called once per RTH cycle from apply_quotes (so the live poller AND
    sim_open.replay exercise it identically). The reference basket's IEX day-cumulative volume must
    ADVANCE each cycle; when it stalls for FEED_FREEZE_CYCLES cycles the trade-data feed is frozen and
    downstream volume reads (cum_rvol, slot_rvol, EP pace) are not trustworthy. Fail-open + no-op when
    the basket isn't loaded (unit tests with synthetic symbols). DAYVOL is assignment-based (last
    cumulative wins), so a frozen `volume` field keeps the sum flat -> this fires; growth resets it."""
    present = [s for s in _FEED_REF if s in DAYVOL]
    if len(present) < 2:
        return
    cur = sum(float(DAYVOL[s]) for s in present)
    prev = STATE.get("vol_ref_sum")
    if prev is not None:
        STATE["vol_frozen_cycles"] = 0 if cur > prev else int(STATE.get("vol_frozen_cycles") or 0) + 1
    STATE["vol_ref_sum"] = cur
    was = bool(STATE.get("feed_degraded"))
    now = int(STATE.get("vol_frozen_cycles") or 0) >= config.FEED_FREEZE_CYCLES
    STATE["feed_degraded"] = now
    if now != was:                                   # log the transition so the freeze/recovery window
        et = ""                                      # is timestamped in serve.log hands-off (no polling
        try:                                         # by anyone — 2026-07-10 open froze ~15 min upstream)
            from zoneinfo import ZoneInfo
            et = dt.datetime.now(ZoneInfo("America/New_York")).strftime("%H:%M:%S ET")
        except Exception:
            pass
        if now:
            print(f"[feed] {et} volume feed FROZEN — reference basket flat {STATE['vol_frozen_cycles']} "
                  f"cycles (ref_sum={cur:.0f}); RVOL/EP volume reads paused", flush=True)
        else:
            print(f"[feed] {et} volume feed RECOVERED — basket advancing again (ref_sum={cur:.0f})", flush=True)


def _fresh_today(ts_raw, today: str) -> bool:
    """The freshness predicate that gates EVERY day-volume write: True iff the quote's last-sale
    timestamp is TODAY. BUGS-341: pre-open / early-open IEX quotes for names with no prints yet still
    carry YESTERDAY's day-cumulative volume with a null/stale timestamp; ingesting it puts a full
    average day into DAYVOL ~ 1000x pace two bars in. Shared by ingest_day_volume and the PREMARKET
    volume gate so the two never drift apart."""
    return bool(ts_raw) and str(ts_raw)[:10] == today


def ingest_day_volume(sym: str, vol, ts_raw, today: str | None = None) -> bool:
    """The SINGLE writer of DAYVOL (open-hardening plan A, Amir 2026-07-09) — the one choke point that
    used to be a guard duplicated in apply_premarket + apply_quotes. Day-cumulative quote volume enters
    the ledger ONLY with a TODAY timestamp (the freshness contract above). `today` defaults to the real
    session date, so live callers pass nothing and behavior is byte-identical; sim_open passes a past
    tape's date to evaluate the guard against the right session offline. Returns True if written."""
    if today is None:
        today = dt.date.today().isoformat()
    if vol and _fresh_today(ts_raw, today):
        DAYVOL[sym] = float(vol)
        return True
    return False
GAP_SYMS: set = set()          # syms whose forming bar sits ACROSS a gap (old last bar) -> need a targeted
_GAP_LAST: dict = {}           # session re-fetch, not an inflated single-bar append (BUGS #84 part B)


def drain_gaps(cooldown_sec: float = 120.0) -> list[str]:
    """Pop-and-return the symbols apply_quotes flagged as sitting across a DATA GAP (their live overlay's last
    today-bar is older than the immediately-preceding slot, so folding a quote there would append ONE inflated
    bar carrying the whole day-delta and never backfill the missing slots — BUGS #84 part B). serve re-fetches
    each name's session (targeted `_seed_today_session`), landing every missing slot with real per-slot volume.
    A per-symbol cooldown stops a tight retry loop when Tiingo itself is briefly lagging that name."""
    now = time.time()
    out = []
    for s in list(GAP_SYMS):
        GAP_SYMS.discard(s)
        if now - _GAP_LAST.get(s, 0.0) >= cooldown_sec:
            _GAP_LAST[s] = now
            out.append(s)
    return out


def _key() -> str | None:
    try:
        return _KEY_FILE.read_text().strip()
    except OSError:
        return None


def session_phase() -> str | None:
    """'pre' 08:00-09:30 ET (IEX pre-market), 'rth' 09:30-16:00, else None."""
    try:
        from zoneinfo import ZoneInfo
        now = dt.datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return None
    if now.weekday() >= 5:
        return None
    t = now.time()
    if dt.time(8, 0) <= t < dt.time(9, 30):
        return "pre"
    if dt.time(9, 30) <= t < dt.time(16, 0):
        return "rth"
    return None


def _market_open_now() -> bool:
    return session_phase() == "rth"


def apply_premarket(quotes: list[dict], today: str | None = None) -> int:
    """Capture pre-market quotes into PREMARKET (price + IEX pre-market volume). Kept separate
    from the 15m frames — pre-9:30 prints must never appear as session bars. `today` defaults to the
    real session date; sim_open passes a tape's date for offline replay (plan A)."""
    n = 0
    today = today or dt.date.today().isoformat()
    _roll_dayvol(today)
    if _PM_DATE["d"] != today:
        PREMARKET.clear()
        _PM_DATE["d"] = today
    for q in quotes:
        sym = q.get("ticker", "").upper()
        last = q.get("tngoLast") or q.get("last")
        if not sym or last is None:
            continue
        vol = q.get("volume") or 0
        ts = q.get("lastSaleTimestamp") or q.get("timestamp")
        # Price is kept for display; VOLUME only counts when the last sale happened TODAY (see
        # _fresh_today / ingest_day_volume — the stale-quote poisoning root cause, BUGS-341). The
        # pre-market high/low (for open_radar's pm-range read) get the SAME freshness gate — a stale
        # quote's high/low are yesterday's and would print a phantom pre-market range.
        fresh = _fresh_today(ts, today)
        hi, lo = q.get("high"), q.get("low")
        PREMARKET[sym] = {"last": float(last), "volume": float(vol) if fresh else 0.0, "ts": ts,
                          "high": float(hi) if (fresh and hi) else None,
                          "low": float(lo) if (fresh and lo) else None}
        ingest_day_volume(sym, vol, ts, today)       # single DAYVOL writer (plan A)
        n += 1
    return n


def fetch_quotes(symbols: list[str], key: str) -> list[dict]:
    out = []
    for i in range(0, len(symbols), CHUNK):
        chunk = symbols[i:i + CHUNK]
        url = ("https://api.tiingo.com/iex/?" + urllib.parse.urlencode(
            {"tickers": ",".join(s.lower() for s in chunk), "token": key}))
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                out.extend(json.load(r))
        except Exception:
            STATE["errors"] += 1
    return out


def _bar_bucket(ts: pd.Timestamp, bucket_min: int = 15) -> pd.Timestamp:
    """The bucket OPEN time for an ET wall-clock timestamp (naive, matching the store). `bucket_min`
    = bar width in minutes (15 legacy, 5 for the Tiingo 5m live base). Uses .floor (NOT .replace) so
    the NANOSECOND field is cleared too — Tiingo lastSaleTimestamps carry ns precision, and .replace
    leaves ns, which made f.index[-1]==bucket fail every quote -> duplicate off-grid bars per slot."""
    return ts.floor(f"{bucket_min}min")


def apply_quotes(frames: dict, quotes: list[dict], bucket_min: int = 15, today: str | None = None) -> int:
    """Fold quotes into the forming `bucket_min`-minute bar of each RAM frame: update o/h/l/c and set
    the bar's volume so the session cumulative equals the quote's day-total (IEX raw — downstream
    calibrates x vol_factor on read). Appends a new bucket row when a fresh window opens. Returns
    symbols updated. `today` defaults to the real session date; sim_open passes a tape's date (plan A)."""
    n = 0
    today = today or dt.date.today().isoformat()
    _roll_dayvol(today)                          # reset stale day-volume on a session rollover
    for q in quotes:
        sym = q.get("ticker", "").upper()
        f = frames.get(sym)
        last = q.get("tngoLast") or q.get("last")
        ts_raw = q.get("lastSaleTimestamp") or q.get("timestamp")
        day_vol = q.get("volume")
        ingest_day_volume(sym, day_vol, ts_raw, today)   # single DAYVOL writer + freshness (plan A):
        #   stale quotes (no TODAY timestamp) carry yesterday's full day-cumulative volume; ingesting
        #   it put dayv ~ a full average day 2 bars in -> cum_rvol ~ 1/frac ~ 1000x (BUGS-341 flood).
        if f is None or last is None or ts_raw is None or not len(f):
            continue
        try:
            ts = pd.Timestamp(ts_raw).tz_localize(None)
        except Exception:
            continue
        if ts.date() < f.index[-1].date():
            continue                                   # stale print older than the store
        bucket = _bar_bucket(ts, bucket_min)
        last = float(last)
        today_mask = f.index.date == ts.date()
        prior_vol = float(f.loc[today_mask, "volume"].iloc[:-1].sum()) if today_mask.any() else 0.0
        if f.index[-1] == bucket:                      # update the forming bar in place
            f.iloc[-1, f.columns.get_loc("close")] = last
            if last > float(f["high"].iloc[-1]):
                f.iloc[-1, f.columns.get_loc("high")] = last
            if last < float(f["low"].iloc[-1]):
                f.iloc[-1, f.columns.get_loc("low")] = last
            if day_vol is not None:
                prior_vol = float(f.loc[f.index.date == ts.date(), "volume"].iloc[:-1].sum())
                f.iloc[-1, f.columns.get_loc("volume")] = max(float(day_vol) - prior_vol, 0.0)
        elif f.index[-1] < bucket:                     # a new bucket window opened -> append
            prev = bucket - pd.Timedelta(minutes=bucket_min)
            today_idx = f.index[today_mask]
            if not (len(today_idx) and today_idx[-1] == prev):
                # GAP (BUGS #84B): the immediately-preceding slot is missing, so the last today-bar is OLD.
                # Appending here would dump ALL volume since that bar into one inflated forming bar and NEVER
                # backfill 13:35->now -> those slots read 0 -> rvol_leaders drops the name. Defer instead:
                # flag for a targeted session re-fetch (serve.drain_gaps -> _seed_today_session) that fills
                # every missing slot with real per-slot volume; the poller resumes in-place updates after.
                GAP_SYMS.add(sym)
                n += 1
                continue
            row = {c: 0.0 for c in f.columns}
            row.update(open=last, high=last, low=last, close=last)
            if day_vol is not None:
                prior_vol = float(f.loc[f.index.date == ts.date(), "volume"].sum())
                row["volume"] = max(float(day_vol) - prior_vol, 0.0)
            frames[sym] = pd.concat([f, pd.DataFrame([row], index=[bucket])])
        n += 1
    try:
        _note_feed_liveness()            # RTH-only path (poller + sim route pre-market elsewhere)
    except Exception:
        pass
    return n


def start(frames: dict, on_cycle=None, interval: int | None = None,
          bucket_min: int = 5) -> None:
    """Launch the poller thread. `frames` is serve's live RAM store (FRAMES5), updated in place: each RTH
    cycle folds bulk IEX quotes into the forming `bucket_min`-minute bar (5m since Phase 3c — the single
    live intraday base). Equities only — IEX carries no futures. `on_cycle(n)` runs after each RTH cycle
    (serve clears its live caches there)."""
    key = _key()
    if not key or STATE["running"]:
        return
    STATE["running"] = True
    interval = interval or config.LIVE_POLL_SEC

    def loop():
        while STATE["running"]:
            phase = session_phase()
            STATE["phase"] = phase
            if phase and frames:
                syms = list(frames)
                quotes = fetch_quotes(syms, key)
                try:                                   # quote-tape recorder (fail-open): raw
                    import quote_tape                  # batches + state snapshots for offline
                    quote_tape.record_cycle(phase, quotes)      # open-simulation (sim_open.py)
                    quote_tape.maybe_snapshot(DAYVOL, PREMARKET)
                except Exception:
                    pass
                if phase == "pre":
                    n = apply_premarket(quotes)
                else:
                    n = apply_quotes(frames, quotes, bucket_min)
                STATE.update(last_poll=dt.datetime.now().isoformat(timespec="seconds"),
                             symbols=n, cycles=STATE["cycles"] + 1)
                if phase == "rth" and on_cycle:
                    try:
                        on_cycle(n)
                    except Exception:
                        pass
            time.sleep(interval)

    threading.Thread(target=loop, daemon=True, name="tiingo-live").start()


def stop() -> None:
    STATE["running"] = False
