"""Primary interactive dashboard + point-in-time backtester (EOD-only).

    python serve.py        # loads bars into RAM, serves http://127.0.0.1:8780

Pick a date in the browser; the server re-runs setup detection as-of that date
(truncating every ticker at the date, no look-ahead) and the charts show the bars
AFTER it, so you can see how each flagged setup actually resolved. The header also
hosts the live setup-settings panel (⚙), a data-update button (↻), and an in-app
auto-update scheduler — all of which re-run detection on demand.

Speed/RAM lever: SERVE_MIN_DVOL_M env var (or config.SERVE_UNIVERSE_MIN_DVOL_M)
restricts the in-memory universe to names above that average $-volume.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

# --- open-file soft-limit raise (hotfix 2026-07-28): serve.py can inherit the macOS default
# soft RLIMIT_NOFILE of 256 (plain-shell / launchd launch, not the raised interactive limit).
# The auto-pull's yfinance threads=True socket burst (datastore._download_batch, and the
# self-correct pull path below) tips ~256 fds past the ceiling -> EMFILE "too many open files".
# Additive, import-only, stdlib. Idempotent, never lowers, best-effort (a refused raise is non-fatal).
try:
    import resource as _resource
    _nofile_soft, _nofile_hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
    _nofile_want = 65536 if _nofile_hard == _resource.RLIM_INFINITY else min(65536, _nofile_hard)
    if _nofile_soft < _nofile_want:
        _resource.setrlimit(_resource.RLIMIT_NOFILE, (_nofile_want, _nofile_hard))
except Exception:
    pass

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask.json.provider import DefaultJSONProvider

import re

import alerts
import awareness
import config
import datastore
import intraday
import labels
import market
import marketcap
import mtf
import narrative
import patterns
import quality
import ratios
import report
import journal
import performance
import pit
import scan
import sequence
import sr
import settings
import tiingo_live
import setups
import universe as uni
from indicators import add_indicators

class _NumpyJSONProvider(DefaultJSONProvider):
    """Serialize numpy scalars/arrays (frames are float32 since the RAM optimization, so computed
    values like the MTF tilt/trend leak np.float32 into responses) — convert them to Python natives."""
    @staticmethod
    def default(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return DefaultJSONProvider.default(o)


app = Flask(__name__)
app.json = _NumpyJSONProvider(app)

FRAMES: dict[str, pd.DataFrame] = {}   # sym -> enriched full-history frame, held in RAM
FRAMES_W: dict[str, pd.DataFrame] = {}   # sym -> enriched weekly frame
FRAMES15: dict[str, pd.DataFrame] = {}   # FUTURES-ONLY 15m frame (RAM; yfinance, lazy) — 24h instruments
                                         # with no IEX/5m coverage; the futures perf/replay + chart base (3c)
FRAMES5: dict[str, pd.DataFrame] = {}    # sym -> 5m frame (RAM; ~30-session tail, calibrated) — the SINGLE
                                         # live intraday base (equities), poller-fed each cycle (Phase 3c)
UNIVERSE: pd.DataFrame | None = None
LATEST = ""

_SCAN_LOCK = threading.Lock()          # blocks scans while the RAM frames are rebuilt
UPDATE = {"running": False, "phase": "", "last_run": None, "error": None}
_AUTOPULL_DONE: set = set()            # dates auto-refreshed this session (once each -> no pull loops)
DOWNLOAD = {"running": False, "phase": "", "current": 0, "total": 0, "error": None,
            "last_run": None, "universe_sig": None,   # last_run/universe_sig = FULL-pull (manual) cooldown anchor
            "last_fut_auto": None}                     # futures-only 5m auto-refresh cadence timer (decoupled)

# Full-universe detection is ~1-2 min, so memoize it — but PER SETUP, keyed by
# (date, setup, that-setup's-signature). Tuning one setup's knobs changes only its own
# signature, so we recompute just that setup (seconds) and serve the other ~11 from cache,
# instead of rerunning the whole universe. Cleared whenever the underlying frames are rebuilt.
SETUP_CACHE: "OrderedDict[tuple, pd.DataFrame]" = OrderedDict()   # (date, setup, sig) -> that setup's hits
CACHE_MAX = 160                       # ~12 setups x a handful of (date, sig) combos
ALL_SETUPS = list(report.SETUP_LABELS)

# The per-ticker base/regime annotation is the expensive shared cost (trendlab segmentation)
# and is GLOBAL — it only changes with REGIME_* / the date, never with a single setup's knobs.
# Cache it so setup recomputes reuse it instead of re-running regime for every hit.
BASE_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()            # (date, regime_sig) -> {sym: base_row}
BASE_CACHE_MAX = 6                     # each entry is the whole universe's base rows (a few MB)
_MARKET_CACHE: dict = {}               # (as-of date, replay-time) -> market situational-awareness score (cleared on reload)
_GROUPS_CACHE: dict = {}               # (as-of date, labels.sig) -> sector/theme leaderboard; a theme edit
                                       # changes labels.sig so stale scores auto-recompute (also cleared on reload)
_PLAYBOOK_CACHE: dict = {}             # (as-of date, labels.sig) -> natural-language playbook (same invalidation)
_AWARE_CACHE: dict = {}                # (date, t, mode, aware_sig, labels.sig) -> awareness map payload
_AWARE_DAILY: "OrderedDict[tuple, dict]" = OrderedDict()   # (anchor, aware_sig, labels.sig) -> {key: daily_view}
_FRAMES5_CACHE: dict = {}              # date -> {sym: calibrated 5m frame [~30 sessions .. date EOD]} for 5m replay
_AWARE_DAILY_MAX = 8                   # slider scrubs share one anchor -> only overlays recompute

# EMA Rider on a non-1D timeframe: computed on a resampled/intraday frame, isolated from the
# daily per-setup cache. Keyed by (tf, date-or-"latest", ema_rider settings sig).
RIDER_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()           # key -> {"ema_rider_bull":[rows], "ema_rider_bear":[rows]}
RIDER_CACHE_MAX = 48


def _symbol_filter(path: str) -> set[str]:
    """Parse a watchlist file into a set of yf_symbols. Accepts plain text or RTF, comma- or
    whitespace-separated, with or without a TradingView `EXCHANGE:` prefix (e.g. NASDAQ:MU).
    Drops the exchange prefix, upper-cases, and maps '.'->'-' (yf form)."""
    raw = Path(path).read_text(errors="ignore")
    if raw.lstrip().startswith("{\\rtf"):
        raw = re.sub(r"\\[a-zA-Z]+-?\d* ?", " ", raw).replace("{", " ").replace("}", " ")
    out = set()
    for tok in re.split(r"[,\s]+", raw):
        t = tok.strip().upper().split(":")[-1].replace(".", "-")   # drop EXCHANGE: prefix, BRK.B->BRK-B
        if re.fullmatch(r"[A-Z][A-Z0-9-]{0,5}", t):                # ticker-shaped only
            out.add(t)
    return out


def load_frames() -> None:
    """(Re)build the in-RAM daily + weekly frames from the bar cache. Reusable after a
    data update — clears the existing frames first. SERVE_SYMBOLS_FILE restricts the universe
    to a small custom watchlist (fast settings testing)."""
    global UNIVERSE, LATEST
    UNIVERSE = uni.load_universe().set_index("yf_symbol")
    FRAMES.clear()
    FRAMES_W.clear()
    syms = datastore.list_symbols()
    sym_file = os.environ.get("SERVE_SYMBOLS_FILE")
    if sym_file:
        want = _symbol_filter(sym_file)
        missing = sorted(want - set(syms))
        syms = [s for s in syms if s in want]
        print(f"custom universe ({sym_file}): {len(syms)} of {len(want)} requested tickers loaded"
              + (f"; NOT in bar cache: {', '.join(missing)}" if missing else ""))
    elif (active := uni.active_symbols()) is not None:        # liquid subset (build_active_universe.py)
        n0 = len(syms)
        syms = [s for s in syms if s in active]
        print(f"active universe: {len(syms)} of {n0} cached tickers (data/universe_active.csv)")
    min_dvol = float(os.environ.get("SERVE_MIN_DVOL_M", config.SERVE_UNIVERSE_MIN_DVOL_M))
    print(f"loading {len(syms)} tickers into memory (min $vol {min_dvol}M) ...")
    latest = None
    for i, sym in enumerate(syms):
        df = datastore.load_bars(sym)
        if df is None or len(df) < config.MIN_BARS:
            continue
        d = add_indicators(df)
        if min_dvol > 0:
            dv = d["dollar_vol"].iloc[-1]
            if pd.isna(dv) or dv / 1e6 < min_dvol:
                continue
        FRAMES[sym] = d
        wdf = datastore.load_bars_w(sym)            # weekly frame for the 1W setups
        if wdf is not None and len(wdf) >= scan.MIN_BARS_W:
            FRAMES_W[sym] = add_indicators(wdf)
        last = d.index[-1]
        latest = last if latest is None or last > latest else latest
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(syms)}  (kept {len(FRAMES)})")

    # --- SYNTHETIC BASKETS ---
    import ratios
    import labels
    labels.load()
    print("Building synthetic sector/theme baskets for RAM...")
    for sec in set(labels._SECTOR.values()):
        members = {s for s, c in labels._SECTOR.items() if c == sec} & set(FRAMES)
        bs = ratios.basket_series(members, FRAMES, min_members=3)
        if bs is not None:
            FRAMES[f"SYNTH_SEC_{sec.replace(' ', '').replace('&', 'And').replace('-', '')}".upper()] = add_indicators(bs)
            
    for thm in set(labels._SYMS_BY_THEME.keys()):
        members = labels._SYMS_BY_THEME[thm] & set(FRAMES)
        bs = ratios.basket_series(members, FRAMES, min_members=3)
        if bs is not None:
            FRAMES[f"SYNTH_THM_{thm.replace(' ', '').replace('&', 'And').replace('-', '')}".upper()] = add_indicators(bs)
    # -----------------------
    if sym_file:                       # small custom list -> fetch market caps now (quick)
        marketcap.ensure(list(FRAMES), refresh=bool(os.environ.get("SERVE_MCAP_REFRESH")))
    else:
        marketcap.load()               # full universe -> use whatever's already cached
    LATEST = (latest.date().isoformat() if latest is not None else dt.date.today().isoformat())
    try:                               # drop any pre-session scan cache for the current day so it
        import shutil                  # recomputes on the freshly-loaded (complete) frames
        shutil.rmtree(_disk_scan_path(LATEST, "_", "_").parent, ignore_errors=True)
    except Exception:
        pass
    print(f"ready: {len(FRAMES)} daily + {len(FRAMES_W)} weekly in RAM, "
          f"latest {LATEST}  ->  http://127.0.0.1:{os.environ.get('PORT', 8780)}")


def _history_append_safe() -> None:
    """AC-8: append the just-settled session to the daily history store (history_store), off the
    critical path and FAIL-OPEN — an error here must never affect the daily update (NG-5)."""
    try:
        import history_store
        n = history_store.append_day()
        print(f"[history] appended {n} entity rows for the settled session", flush=True)
    except Exception as e:
        print(f"[history] append skipped (non-fatal): {e}", flush=True)


def _run_update(weekly: bool = False) -> None:
    """Background worker: pull the latest EOD data, then rebuild the RAM frames. weekly=True does the
    FULL-universe refresh + re-evaluates active membership (build_active_universe) before reloading."""
    UPDATE.update(running=True, phase="downloading" + (" (full)" if weekly else ""), error=None)
    try:
        if weekly:
            datastore.update_daily(full=True)
            UPDATE["phase"] = "rebuilding universe"
            import build_active_universe
            build_active_universe.build()
        else:
            datastore.update_daily()
        UPDATE["phase"] = "reloading"
        with _SCAN_LOCK:
            load_frames()                  # reloads the (possibly changed) active universe
            SETUP_CACHE.clear()            # frames changed -> every cached scan is now stale
            BASE_CACHE.clear()
            _MARKET_CACHE.clear()
            import tv_breadth; tv_breadth.clear_cache()   # pick up any re-extracted TV daily breadth
            _GROUPS_CACHE.clear()
            _PLAYBOOK_CACHE.clear()
            _AWARE_CACHE.clear()
            _AWARE_DAILY.clear()
            FRAMES15.clear()
            FRAMES5.clear()               # live 5m store -> reseed next live read (poller re-picks it up)
            _FRAMES5_CACHE.clear()        # new day / any split-repair of the 5m store -> drop stale replay frames
            RIDER_CACHE.clear()
        UPDATE.update(phase="done", last_run=dt.datetime.now().isoformat(timespec="seconds"))
        _seed_notif_log()
        threading.Thread(target=precompute, args=(_live_day(),), daemon=True).start()
        try:                                    # fresh daily bars -> fill realized outcomes into the
            import sig_ledger                   # significance ledger (fwd_1d..21d, MFE/MAE, vs-SPY)
            threading.Thread(target=sig_ledger.backfill_outcomes, daemon=True).start()
        except Exception:
            pass
        threading.Thread(target=_history_append_safe, daemon=True).start()   # persist the settled day
    except Exception as e:  # surfaced via /api/update/status
        UPDATE["error"] = str(e)
    finally:
        UPDATE["running"] = False


def start_update(weekly: bool = False) -> bool:
    """Kick off a background update if one isn't already running."""
    if UPDATE["running"]:
        return False
    threading.Thread(target=_run_update, args=(weekly,), daemon=True).start()
    return True


def _live_day() -> str:
    """The date the app anchors 'live' to = TODAY's ET calendar day (Amir 2026-07-08: the picker default and
    the 'Live' / 'back to live' buttons must read TODAY, even pre-market before today's daily bar exists).
    NOT `LATEST` (the newest SETTLED daily bar): today's bar never settles until EOD, so anchoring to LATEST
    made the app open on — and 'Live' snap back to — yesterday. Rolls back over weekends to the most recent
    weekday so a Sat/Sun open shows Friday's session rather than an empty future date. Safe downstream: scans
    slice frames `.loc[:date]` (a today>LATEST date just returns through the last real bar) and
    `_is_partial(today)` splices the live 5m session on top during RTH."""
    d = pd.Timestamp(market._today_et())
    while d.weekday() >= 5:                 # Sat/Sun -> most recent weekday (Friday)
        d -= pd.Timedelta(days=1)
    return d.date().isoformat()


def _session_over_today() -> bool:
    """True from the 16:00 ET close until ET midnight on a weekday — the window where TODAY's session
    is complete and 'today at close' reads (day-type banner, EOD auto-update) should take over."""
    try:
        from zoneinfo import ZoneInfo
        et = dt.datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return False
    return et.weekday() < 5 and et.time() >= dt.time(16, 0)


def _futures_open_now() -> bool:
    """CME Globex ~24x5 (ET): Sun 18:00 -> Fri 17:00, with a daily 17:00-18:00 maintenance break. Gates the
    CHEAP futures 5m auto-refresh so the NQ/ES/… charts stay current OVERNIGHT + pre-market — NOT just equity
    RTH (Amir trades the overnight session; the old `_is_partial` gate froze the 24h futures store outside
    09:30-16:00). yfinance returns no new bar during the break/weekend, so an occasional over-poll is harmless."""
    try:
        from zoneinfo import ZoneInfo
        et = dt.datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return False
    wd, t = et.weekday(), et.time()            # Mon=0 .. Sat=5, Sun=6
    if wd == 5:                                # Saturday: closed all day
        return False
    if wd == 6:                                # Sunday: reopens 18:00 ET
        return t >= dt.time(18, 0)
    if wd == 4 and t >= dt.time(17, 0):        # Friday: closes 17:00 ET
        return False
    return not (dt.time(17, 0) <= t < dt.time(18, 0))    # Mon-Fri: open except the daily maintenance hour


def _universe_sig() -> str:
    """Hash of the loaded symbol set — so a debounce only blocks re-pulls of the SAME universe."""
    return hashlib.md5(",".join(sorted(FRAMES)).encode()).hexdigest()


def _intraday_cooldown_left() -> int:
    """Seconds until another intraday pull is allowed for the CURRENT universe (0 = allowed now).
    Resets whenever the universe changes (different tickers loaded)."""
    if DOWNLOAD["last_run"] is None or DOWNLOAD["universe_sig"] != _universe_sig():
        return 0
    return max(0, int(config.INTRADAY_REFRESH_COOLDOWN_MIN * 60 - (time.time() - DOWNLOAD["last_run"])))


def _run_download_intraday(futures_only: bool = False) -> None:
    """Background worker / manual '⬇ Intraday' + auto-refresh. (1) fetch 60-day 5m bars for the FUTURES
    (24h, no Tiingo coverage) — CHEAP, one batched yfinance call. (2) unless `futures_only`, GAP-FILL today's
    equity 5m session into FRAMES5 from Tiingo (the 'update the missing info' catch-up) — a ~universe per-symbol
    pull, the EXPENSIVE half. The frequent auto-refresh passes futures_only=True (equities are already live via
    the Tiingo 5m poller); the manual button + the slower equity safety-net timer do the full fill. Only a FULL
    pull stamps last_run/universe_sig (the manual-button cooldown anchor), so the 5m futures cadence doesn't
    perpetually debounce the manual catch-up."""
    syms = [s for s in FRAMES if uni.is_future(s)]
    DOWNLOAD.update(running=True, phase="downloading", current=0, total=len(syms), error=None)
    try:
        def prog(done, total, sym):
            DOWNLOAD.update(current=done, phase=f"downloading {sym}")
        n = datastore.fetch_intraday_15m(syms, progress=prog)
        with _SCAN_LOCK:
            RIDER_CACHE.clear()             # new futures intraday bars -> stale rider results
            FRAMES15.clear()                # rebuild the futures 5m RAM store from the fresh bars
        if futures_only:
            DOWNLOAD.update(phase=f"done ({n} futures 5m)")
            return
        m = 0
        try:
            DOWNLOAD.update(phase="filling today's 5m session")
            m = _seed_today_session(force=True)     # equity today-session gap-fill (09:30->now)
            if m:
                with _SCAN_LOCK:                     # fresh session -> recompute the live intraday reads
                    _MARKET_CACHE.clear(); _GROUPS_CACHE.clear(); _AWARE_CACHE.clear(); _PLAYBOOK_CACHE.clear()
        except Exception:
            pass
        DOWNLOAD.update(phase=f"done ({n} futures, {m} equities 5m)", last_run=time.time(),
                        universe_sig=_universe_sig())
    except Exception as e:
        DOWNLOAD["error"] = str(e)
    finally:
        DOWNLOAD["running"] = False


def start_download_intraday(futures_only: bool = False) -> bool:
    if DOWNLOAD["running"]:
        return False
    threading.Thread(target=_run_download_intraday, args=(futures_only,), daemon=True).start()
    return True


@app.get("/")
def index():
    return report.server_shell(default_date=_live_day())   # boot + 'Live' anchor on TODAY, not the settled bar


def _regime_sig() -> str:
    """Signature of the settings the per-ticker base/regime annotation depends on."""
    cur = settings.current()
    return json.dumps({n: cur[n] for n in settings.TUNABLE["regime"]}, sort_keys=True, default=str)


def _disk_scan_path(date: str, setup: str, sig: str):
    import hashlib
    h = hashlib.md5(sig.encode()).hexdigest()[:10]
    return config.DATA_DIR / "scan_cache" / date / f"{setup}_{h}.parquet"


def _prune_scan_cache(keep: int = 12) -> None:
    root = config.DATA_DIR / "scan_cache"
    if not root.exists():
        return
    days = sorted(p for p in root.iterdir() if p.is_dir())
    for p in days[:-keep]:
        for f in p.iterdir():
            f.unlink(missing_ok=True)
        p.rmdir()


def _scan_cached(date: str, warm_only: bool = False) -> "pd.DataFrame | None":
    """Hits for (date, current settings), memoized PER SETUP and backed by a shared per-ticker
    base/regime cache. A setup-knob change recomputes only the affected setups (one pass over
    just those detectors) reusing the cached regime annotation, so it's seconds instead of a
    full rescan. A regime/date change rebuilds the base and every setup. Caller holds _SCAN_LOCK.
    DISK layer: computed per-setup frames persist to data/scan_cache/{date}/ keyed by settings
    signature, so a restart reads yesterday's scan instead of recomputing (~1MB/day).
    warm_only: return None instead of paying a fresh ~60s scan_asof when the date isn't already
    cached (SETUP_CACHE or disk) — used by opportunistic replay annotations that must stay fast."""
    bkey = (date, _regime_sig())
    bases = BASE_CACHE.get(bkey)
    feed = bases if bases is not None else {}    # filled in by scan_asof when computing fresh
    keys = {s: (date, s, settings.setup_sig(s)) for s in ALL_SETUPS}
    missing = [s for s in ALL_SETUPS if keys[s] not in SETUP_CACHE]
    settled = date < LATEST          # the current/mutating day is never disk-cached: its EOD data is
    still = []                       # still filling, so a pre-session parquet would mask today's gaps
    for s in missing:                            # disk before recompute (settled past days only)
        p = _disk_scan_path(date, s, keys[s][2])
        if settled and p.exists():
            try:
                SETUP_CACHE[keys[s]] = pd.read_parquet(p)
                continue
            except Exception:
                pass
        still.append(s)
    missing = still
    if warm_only and missing:
        return None                        # cold date -> skip the ~60s as-of scan (opportunistic replay path)
    if missing:
        if not FRAMES:
            raise ValueError("Server is still booting (bars not loaded yet). Please try again in 30-60 seconds.")
        with settings.apply():             # honor saved threshold overrides for the recompute
            hits = scan.scan_asof(date, FRAMES, UNIVERSE, FRAMES_W, only=set(missing), bases=feed)
        if bases is None:                  # we just built the base map -> cache it
            BASE_CACHE[bkey] = feed
            BASE_CACHE.move_to_end(bkey)
            while len(BASE_CACHE) > BASE_CACHE_MAX:
                BASE_CACHE.popitem(last=False)
        for s in missing:                  # split the pass into per-setup frames (empty df if none)
            SETUP_CACHE[keys[s]] = hits[hits["setup"] == s].copy() if len(hits) else hits
            if settled:                    # never persist the still-filling current day
                try:
                    p = _disk_scan_path(date, s, keys[s][2])
                    p.parent.mkdir(parents=True, exist_ok=True)
                    SETUP_CACHE[keys[s]].to_parquet(p, index=False)
                except Exception:
                    pass                   # disk persistence is best-effort, never blocks the scan
        _prune_scan_cache()
    parts = []
    for s in ALL_SETUPS:                    # assemble in canonical order, refresh LRU
        df = SETUP_CACHE[keys[s]]
        SETUP_CACHE.move_to_end(keys[s])
        parts.append(df)
    while len(SETUP_CACHE) > CACHE_MAX:
        SETUP_CACHE.popitem(last=False)
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


_INTRADAY_BASE_BARS = 12000            # ~5-6 months of 5m — the intraday chart/detector base (resampled up)


def _intraday_base(sym: str):
    """The intraday base frame for charts/detectors, tail-limited; higher TFs (15m/30m/1h/2h/4h/8h/12h)
    resample UP from it via resample_intraday (base AUTO-INFERRED). EQUITIES use the Tiingo 5m store
    (calibrated -> consolidated-scale volume, the same basis the old yfinance-15m frames carried);
    5m->1h == 15m->1h (both session-anchored) and 5m->15m == the old yfinance 15m to <0.004%, so it is a
    near-identical drop-in. FUTURES trade 24h and Tiingo-5m does not cover them, so they use a yfinance
    5m base instead (load_bars_15m is a LEGACY name — the store is 5m now). None when an equity has no
    5m store (those names are dropped from live intraday)."""
    if uni.is_future(sym):
        return datastore.load_bars_15m(sym)             # 24h instruments: yfinance 5m base (legacy name)
    b = datastore.load_bars_5m(sym)                     # deep disk history (lags by a backfill cycle)
    if not FRAMES5 and market._is_partial(None):        # live session but store cold -> warm it once,
        _ensure_live_poller()                           # so charts get today's forming bars like the radar
    live = FRAMES5.get(sym)                             # poller-fed live 5m (disk tail + today's bars)
    if live is not None and len(live):
        if b is None or not len(b):
            b = live
        elif live.index[-1] > b.index[-1]:              # graft today's live bars onto the deep history
            b = pd.concat([b, live[live.index > b.index[-1]]])
            b = b[~b.index.duplicated(keep="last")].sort_index()
    if b is None or not len(b):
        return None
    return b.iloc[-_INTRADAY_BASE_BARS:]


def _tf_frame(sym: str, tf: str, date: str | None):
    """Enriched frame for `sym` on timeframe `tf`. 1D = the RAM daily frame; daily-derived TFs
    (2D..6M) resample the daily frame truncated at `date` (look-ahead-safe); intraday TFs build
    from the Tiingo 5m base (latest snapshot). Returns None if missing / too short."""
    if tf == "1D":
        daily = FRAMES.get(sym)
        return None if daily is None else daily.loc[:date]
    if tf in datastore.INTRA_TFS or tf == "5m":        # 5m = the base itself (resample returns it as-is)
        base5 = _intraday_base(sym)
        if base5 is None:
            return None
        raw = datastore.resample_intraday(base5, tf, futures=uni.is_future(sym))
    else:                                              # 2D..6M daily-derived
        daily = FRAMES.get(sym)
        if daily is None:
            return None
        raw = datastore.resample_daily(daily.loc[:date][["open", "high", "low", "close", "volume"]], tf)
    if raw is None or len(raw) < 5:
        return None
    return add_indicators(raw)


def _rider_scan(tf: str, date: str | None) -> dict:
    """{ema_rider_bull:[rows], ema_rider_bear:[rows]} for `tf`, cached. Reuses the daily base row
    (regime/market_cap/liquidity) + the existing rider detectors + quality/pattern — no detector
    changes. Daily-derived TFs honor the as-of date; intraday is a latest snapshot."""
    intraday = tf in datastore.INTRA_TFS
    eff_date = None if intraday else (date or LATEST)
    key = (tf, "latest" if intraday else eff_date, settings.setup_sig("ema_rider_bull"))
    cached = RIDER_CACHE.get(key)
    if cached is not None:
        RIDER_CACHE.move_to_end(key)
        return cached
    out = {"ema_rider_bull": [], "ema_rider_bear": []}
    active = uni.active_symbols()
    with settings.apply():                             # honor saved ER_* overrides
        for sym in FRAMES:
            if active is not None and sym not in active:
                continue
            tfd = _tf_frame(sym, tf, eff_date)
            if tfd is None:
                continue
            hits = [h for h in (setups.detect_ema_rider_bull(tfd), setups.detect_ema_rider_bear(tfd)) if h]
            if not hits:
                continue
            meta = UNIVERSE.loc[sym] if (UNIVERSE is not None and sym in UNIVERSE.index) else None
            base = scan._base_row(sym, FRAMES[sym].loc[:eff_date], meta)   # daily context
            for hit in hits:
                row = {**base, "tf": tf, **hit}
                row.update(quality.score_hit(tfd, row))
                cb = row.get("cons_bars")
                if "pattern" not in row and cb and int(cb) >= 3:
                    row["pattern"] = patterns.classify_consolidation(tfd, len(tfd) - int(cb), len(tfd) - 1)["pattern"]
                out[hit["setup"]].append(row)
    out = {k: report._clean(v) for k, v in out.items()}   # NaN -> null for the JS table
    RIDER_CACHE[key] = out
    RIDER_CACHE.move_to_end(key)
    while len(RIDER_CACHE) > RIDER_CACHE_MAX:
        RIDER_CACHE.popitem(last=False)
    return out


def precompute(date: str) -> None:
    """Warm the cache for a date off the request path (startup + after an update): the scan
    frames AND the per-row annotations (ext badge / nearest-S/R / rsi-extremes), so the first
    page load after a restart gets the 0.2s path instead of a 20-35s annotation crunch."""
    try:
        t = time.time()
        with _SCAN_LOCK:
            hits = _scan_cached(date)
            tables = {s: (hits[hits["setup"] == s].to_dict("records") if len(hits) else [])
                      for s in ALL_SETUPS}
            _annotate_risk_reward(tables, date)
        print(f"[cache] warmed {date} in {time.time() - t:.0f}s "
              f"({len(SETUP_CACHE)} setup entries, {len(_EXT_CACHE)} annotated symbols)")
        try:                                   # warm the 5m perf store so the first Perf-panel open is
            _perf_bars(date, None)             # fast (~2s), not a ~30s cold _day5_store build on the request path
        except Exception:
            pass
        # warm the HEAVY panels too (riders direct; market/playbook via self-HTTP once the server
        # listens, so the real request path + its caches are exercised) — after a restart these each
        # take minutes cold and serialize on _SCAN_LOCK, so the first user click stalls the whole
        # app (2026-07-08 live QA). Best-effort, sequential, off the request path.
        try:
            t2 = time.time()
            _rider_scan("1D", date)
            print(f"[cache] warmed riders(1D) in {time.time() - t2:.0f}s", flush=True)
        except Exception as e2:
            print(f"[cache] warm riders skipped: {e2}", flush=True)
        import urllib.request as _rq
        port = int(os.environ.get("PORT", 8780))
        for _ in range(60):                                # wait for the listener (boot ordering)
            try:
                _rq.urlopen(f"http://127.0.0.1:{port}/api/data_health", timeout=3)
                break
            except Exception:
                time.sleep(2)
        for ep in ("market", "playbook"):
            try:
                t2 = time.time()
                _rq.urlopen(f"http://127.0.0.1:{port}/api/{ep}", timeout=600)
                print(f"[cache] warmed {ep} in {time.time() - t2:.0f}s", flush=True)
            except Exception as e2:
                print(f"[cache] warm {ep} skipped: {e2}", flush=True)
    except Exception as e:
        print(f"[cache] precompute failed: {e}")


_BB_TF_CACHE: dict = {}
_BB_TFS = ["1W", "1D", "12h", "4h", "1h", "15m", "5m"]   # Amir trades the BB on all of these


def _bb_tf_armed(frame) -> bool:
    if frame is None or len(frame) < 40:
        return False
    try:
        return bool(setups._bb_os_state(frame)[5])
    except Exception:
        return False


def _bb_armed_tfs(sym: str, date: str) -> list[str]:
    """Which timeframes the BackBurner state machine is ARMED on for `sym` (new high standing,
    next oversold wick fires). 1W/1D from RAM frames; 12h..15m resampled from the Tiingo 5m base;
    5m from the Tiingo store tail. Cached per (sym, day)."""
    lvl = settings.current().get("BB_RSI_ENTRY1", config.BB_RSI_ENTRY1)
    key = (sym, date, lvl)                # honor a live RSI-level override in the settings UI
    if key in _BB_TF_CACHE:
        return _BB_TF_CACHE[key]
    out = []
    if _bb_tf_armed(FRAMES_W.get(sym)):
        out.append("1W")
    if _bb_tf_armed(FRAMES.get(sym)):
        out.append("1D")
    if date >= LATEST:
        b5 = _intraday_base(sym)
        if b5 is not None and len(b5) > 120:
            for tf in ("12h", "4h", "1h", "15m"):
                try:
                    fr = datastore.resample_intraday(b5, tf, futures=uni.is_future(sym))
                except Exception:
                    fr = None
                if _bb_tf_armed(fr):
                    out.append(tf)
        if b5 is not None and len(b5) and not uni.is_future(sym):
            try:
                if _bb_tf_armed(b5.iloc[-2200:]):    # ~4 weeks of the merged disk+live 5m base
                    out.append("5m")
            except Exception:
                pass
    _BB_TF_CACHE[key] = out
    if len(_BB_TF_CACHE) > 8000:
        _BB_TF_CACHE.clear()
    return out


def _bb_tf_frame(sym: str, tf: str):
    """The frame the BB machine (and the chart) uses for (symbol, timeframe)."""
    if tf == "1D":
        return FRAMES.get(sym)
    if tf == "1W":
        return FRAMES_W.get(sym)
    if tf == "5m":
        return _intraday_base(sym)   # merged disk+live 5m (equities) OR the yfinance 5m base (futures)
    b5 = _intraday_base(sym)
    if b5 is None:
        return None
    try:
        return datastore.resample_intraday(b5, tf, futures=uni.is_future(sym))
    except Exception:
        return None


@app.get("/api/bb_fires")
def api_bb_fires():
    """Backburner fires for ONE (symbol, timeframe, RSI level) — computed fresh on the SAME frame
    the chart uses, so marker times align exactly. TFs: 5m, 15m, 1h, 4h, 12h, 1D, 1W;
    rsi = the entry trigger level (30/25/20). Each fire carries pOS = the price the wick had to
    touch for RSI to print the level (shown on the chart marker)."""
    sym = (request.args.get("symbol") or "").upper()
    tf = request.args.get("tf") or "1D"
    lvl = float(request.args.get("rsi") or config.BB_RSI_ENTRY1)
    fr = _bb_tf_frame(sym, tf)
    if fr is None or len(fr) < 40:
        return jsonify({"symbol": sym, "tf": tf, "fires": [], "error": f"no {tf} data for {sym}"})
    fires: list = []
    f1, f2, _, p1, p2, armed = setups._bb_os_state(fr, collect=fires, l1=lvl)
    intra = tf not in ("1D", "1W")
    out = [{"date": (int(fr.index[t].tz_localize("UTC").timestamp()) if intra
                     else fr.index[t].date().isoformat()),
            "state": st,
            "pos": round(float(p1[t] if st == "entry1" else p2[t]), 2)} for t, st in fires]
    return jsonify({"symbol": sym, "tf": tf, "rsi": lvl, "fires": out, "n": len(out),
                    "armed": bool(armed)})


_SR_ROW_CACHE: dict = {}
_RATIO_STRUCT_CACHE: dict = {}
SHORT_SETUPS = {"double_top", "head_shoulders", "downtrend", "ema_rider_bear", "rsi_extreme_fade"}


def _nearest_levels(sym: str, date: str):
    """(nearest support, nearest resistance) below/above the close — cached per (sym, day)."""
    key = (sym, date)
    if key not in _SR_ROW_CACHE:
        if len(_SR_ROW_CACHE) > 6000:
            _SR_ROW_CACHE.clear()
        sup = res = None
        d = FRAMES.get(sym)
        if d is not None:
            try:
                srl = sr.sr_levels(d.loc[:date])
                sup = (srl.get("support") or [{}])[0].get("price")
                res = (srl.get("resistance") or [{}])[0].get("price")
            except Exception:
                pass
        _SR_ROW_CACHE[key] = (sup, res)
    return _SR_ROW_CACHE[key]


_ANNOT_LOADED: set = set()


def _annot_path(date: str):
    return config.DATA_DIR / "scan_cache" / date / "annot.parquet"


def _annot_load(date: str) -> None:
    """Warm the ext-badge + nearest-S/R annotation caches from disk. These are computed per
    (symbol, date) on the first /api/scan after a restart — without this, every restart's first
    page load recomputes them all and FEELS like a rescan even though the scan itself is cached."""
    if date in _ANNOT_LOADED:
        return
    _ANNOT_LOADED.add(date)
    p = _annot_path(date)
    if not p.exists():
        return
    try:
        df = pd.read_parquet(p)
        for r in df.itertuples(index=False):
            _EXT_CACHE.setdefault((r.symbol, date), json.loads(r.tfs) if r.tfs else None)
            _SR_ROW_CACHE.setdefault((r.symbol, date),
                                     (None if pd.isna(r.sup) else float(r.sup),
                                      None if pd.isna(r.res) else float(r.res)))
            if hasattr(r, "rsix"):
                _RSIX_CACHE.setdefault((r.symbol, date), json.loads(r.rsix) if r.rsix else None)
    except Exception:
        pass


def _annot_save(date: str) -> None:
    """Best-effort persist of the per-symbol annotations computed for `date` (~50KB/day)."""
    try:
        rows = []
        for (sym, d), tfs in _EXT_CACHE.items():
            if d != date:
                continue
            sup, res = _SR_ROW_CACHE.get((sym, d), (None, None))
            rz = _RSIX_CACHE.get((sym, d))
            rows.append({"symbol": sym, "tfs": json.dumps(tfs) if tfs else None,
                         "sup": sup, "res": res,
                         "rsix": json.dumps(rz) if rz else None})
        if rows:
            p = _annot_path(date)
            p.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(p, index=False)
    except Exception:
        pass


def _annotate_risk_reward(tables: dict, date: str) -> None:
    """$risk / $reward / r-R per row from the S/R ladder (Disabled for speed)."""
    return


# ── Extension Health Badge: ATR-multiple distance from the 50-MA, per timeframe ──────────
_EXT_CACHE: dict = {}


def _ext_kind(sym: str) -> str:
    """Badge tier family — ETFs/indices/futures stretch less than single stocks."""
    return "etf" if (sym in uni.CURATED_ETFS or sym.startswith("^") or uni.is_future(sym)) else "stock"


def _atr_ext50(fr) -> float | None:
    """(close − 50MA) / ATR14 on a frame's last bar. Uses enriched columns when present,
    otherwise computes both on the tail (lean path for resampled badge frames)."""
    if fr is None or len(fr) < 55:
        return None
    last = fr.iloc[-1]
    if "sma50" in fr.columns and "atr14" in fr.columns \
            and pd.notna(last.get("sma50")) and pd.notna(last.get("atr14")):
        ma, atr = float(last["sma50"]), float(last["atr14"])
    else:
        t = fr.iloc[-70:]
        ma = float(t["close"].rolling(50).mean().iloc[-1])
        tr = np.maximum(t["high"] - t["low"],
                        np.maximum((t["high"] - t["close"].shift()).abs(),
                                   (t["low"] - t["close"].shift()).abs()))
        atr = float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
    if not atr or atr <= 0 or np.isnan(ma) or np.isnan(atr):
        return None
    return round((float(last["close"]) - ma) / atr, 1)


def _ext_tfs(sym: str, date: str) -> dict | None:
    """{tf: atr-ext-from-50MA} across config.MTF_TFS for the badge (cached per sym/date).
    Daily-derived TFs honor the as-of date; intraday TFs exist only as the LIVE snapshot
    (60-day 15m window), so they are computed for the latest date alone — no look-ahead."""
    key = (sym, date)
    if key in _EXT_CACHE:
        return _EXT_CACHE[key]
    out, cols = {}, ["open", "high", "low", "close", "volume"]
    daily = FRAMES.get(sym)
    d = daily.loc[:date] if daily is not None else None
    if d is not None:
        v = _atr_ext50(d)                                # enriched fast path
        if v is not None:
            out["1D"] = v
        for tf in config.MTF_TFS:
            if tf not in datastore.INTRA_TFS and tf != "1D":
                try:
                    v = _atr_ext50(datastore.resample_daily(d[cols], tf))
                except Exception:
                    v = None
                if v is not None:
                    out[tf] = v
    if date >= LATEST:                                 # live day (today) OR the settled bar -> intraday MTF context
        df15 = _intraday_base(sym)
        if df15 is not None and len(df15):
            for tf in config.MTF_TFS:
                if tf in datastore.INTRA_TFS:
                    try:
                        v = _atr_ext50(datastore.resample_intraday(df15, tf, futures=uni.is_future(sym)))
                    except Exception:
                        v = None
                    if v is not None:
                        out[tf] = v
    if len(_EXT_CACHE) > 30000:
        _EXT_CACHE.clear()
    _EXT_CACHE[key] = out or None
    return out or None


# ── RSI Historical Extremes (rsi_extremes store): live 1D + persisted summary ────────────
import rsi_extremes as rsix_mod

_RSIX_SUM: dict = {}          # {(sym, tf): summary-state row} from build_rsi_extremes.py
_RSIX_SUM_MTIME = None
_RSIX_CACHE: dict = {}        # (sym, date) -> compact per-TF badge map (or None)
_RSIX_BREADTH = None          # % of actives at all-time RSI zones (capitulation/euphoria read)


def _rsix_summary() -> dict:
    global _RSIX_SUM, _RSIX_SUM_MTIME, _RSIX_BREADTH
    p = config.DATA_DIR / "rsi_extremes" / "summary.parquet"
    try:
        mt = p.stat().st_mtime
    except OSError:
        return _RSIX_SUM
    if mt != _RSIX_SUM_MTIME:
        try:
            df = pd.read_parquet(p)
            _RSIX_SUM = {(r["symbol"], r["tf"]): {k: (None if pd.isna(v) else v)
                                                  for k, v in r.items() if k not in ("symbol", "tf")}
                         for _, r in df.iterrows()}
            d1 = df[df["tf"] == "1D"]
            _RSIX_BREADTH = ({"atl_pct": round(100 * (d1["zone"] == "atl").mean(), 1),
                              "ath_pct": round(100 * (d1["zone"] == "ath").mean(), 1),
                              "n": int(len(d1))} if len(d1) else None)
            _RSIX_SUM_MTIME = mt
        except Exception:
            pass
    return _RSIX_SUM


def _rsix_compact(st: dict | None) -> dict | None:
    """Badge-sized: ALL-TIME zones only (Amir 2026-07-03 — rolling-200 zones were badge noise;
    they still show in the ticker card's rsi-x row). A TF matters when RSI is IN its all-time
    zone or entered/broke one within the recent window."""
    if not st:
        return None
    zone, bs = st.get("zone"), st.get("bars_since")
    zone = zone if zone in ("ath", "atl", "near_ath", "near_atl") else None
    recent = bs is not None and not pd.isna(bs) and bs <= config.RSIX_RECENT_BARS
    if not zone and not recent:
        return None
    return {"z": zone, "b": (int(bs) if recent else None), "r": st.get("rsi"),
            "x": bool(st.get("broke")), "le": st.get("last_event")}


def _rsix_row(sym: str, date: str) -> dict | None:
    """{tf: compact-state} for the ◉ badge: 1D computed LIVE from the RAM frame (as-of date);
    other TFs from the persisted summary (as of its last build; latest date only)."""
    key = (sym, date)
    if key in _RSIX_CACHE:
        return _RSIX_CACHE[key]
    out = {}
    d = FRAMES.get(sym)
    if d is not None and len(d) >= config.RSIX_MIN_HISTORY:
        try:
            c = _rsix_compact(rsix_mod.state(d.loc[:date]))
            if c:
                out["1D"] = c
        except Exception:
            pass
    if date >= LATEST:                                 # live day (today) OR the settled bar -> RSIX summary
        sm = _rsix_summary()
        for tf in config.RSIX_TFS:
            if tf != "1D":
                c = _rsix_compact(sm.get((sym, tf)))
                if c:
                    out[tf] = c
    if len(_RSIX_CACHE) > 30000:
        _RSIX_CACHE.clear()
    _RSIX_CACHE[key] = out or None
    return out or None


def _rsix_full(sym: str) -> dict:
    """Full per-TF states for the ticker card (1D live, rest from the summary build)."""
    out = {}
    d = FRAMES.get(sym)
    if d is not None and len(d) >= config.RSIX_MIN_HISTORY:
        try:
            st = rsix_mod.state(d)
            if st:
                out["1D"] = st
        except Exception:
            pass
    sm = _rsix_summary()
    for tf in config.RSIX_TFS:
        if tf != "1D" and (sym, tf) in sm:
            out[tf] = sm[(sym, tf)]
    return out


def _rsix_chart_marks(fr, cap: int = 40) -> list:
    """Chart markers for the bars whose WICK hit the symbol's ALL-TIME RSI extreme on this
    frame's TF (same intrabar mechanics as the rsi_extreme setups: low <= the price at which
    RSI prints all-time-low+AT_TOL; mirror on highs): [{date, side, rsi, price}] — price = that
    exact trigger price, rsi = the all-time extreme being hit. First bar of each episode only;
    the shifted expanding extremes keep every mark point-in-time."""
    try:
        if fr is None or len(fr) < config.RSIX_MIN_HISTORY:
            return []
        _, ahi, alo, _, _ = rsix_mod.series(fr)
        at = config.RSIX_AT_TOL
        p_lo = rsix_mod.os_trigger_price(fr["close"], (alo + at).clip(lower=2.0, upper=45.0))
        p_hi = rsix_mod.ob_trigger_price(fr["close"], (ahi - at).clip(lower=55.0, upper=98.0))
        low, high = fr["low"].to_numpy(float), fr["high"].to_numpy(float)
        import setups as _st
        out = []
        for side, pser, lvl, ext in (("atl", p_lo, alo, low), ("ath", p_hi, ahi, high)):
            pv, lv = pser.to_numpy(float), lvl.to_numpy(float)
            prev = False
            for t in range(len(pv)):
                f = (not np.isnan(pv[t]) and not np.isnan(lv[t])
                     and (ext[t] <= pv[t] if side == "atl" else ext[t] >= pv[t]))
                if f and not prev:
                    out.append({"date": _st._dt_str(fr, t), "side": side,
                                "rsi": round(float(lv[t]), 1), "price": round(float(pv[t]), 2)})
                prev = f
        out.sort(key=lambda x: str(x["date"] or ""))
        return out[-cap:]
    except Exception:
        return []


# ── Setup Weather: measured awareness→expectancy tilts (T7 study, 43,827 fires, era-robust) ──
# Δ = mean R@20b (fire-close entry, fire-day-low stop) with state ON minus OFF; only
# era-robust cells carried (same sign in >=2 of 3 eras). Source: studies 2026-07-03.
_WEATHER_DELTAS = {
    "capitulation window": {"MOMO": -0.44, "EVENT": -0.11, "MEANREV": -0.43},
    "breadth washed-out":  {"MOMO": -0.37, "EVENT": +0.15, "MEANREV": +0.45},
    "VIX>26 fear":         {"MOMO": 0.0,   "EVENT": +0.25, "MEANREV": +0.38},
    "VIX<16 calm":         {"MOMO": 0.0,   "EVENT": +0.03, "MEANREV": -0.40},
    "QQQ 10>20 rising":    {"MOMO": +0.09, "EVENT": +0.13, "MEANREV": -0.31},
    "SPY ext>=3ATR":       {"MOMO": +0.13, "EVENT": 0.0,   "MEANREV": -0.22},
    "distribution days>=4": {"MOMO": +0.16, "EVENT": +0.12, "MEANREV": +0.24},
}
_WEATHER_SETUPS = {
    "MOMO":    ["QM Breakout", "High Tight Flag", "Stairstep", "Flat Base"],
    "EVENT":   ["Episodic Pivot", "Gappers", "High Volume Close", "Delayed HVC"],
    "MEANREV": ["Backburner"],
}
_WEATHER_CACHE: dict = {}
_CAPWIN_CACHE: dict = {}


def _capitulation_window() -> bool:
    """True within ~42 sessions of a day with >=2% of actives AT their all-time RSI lows —
    the T2-validated capitulation gauge (universal caution state: all families ~-0.4R)."""
    key = LATEST
    if key in _CAPWIN_CACHE:
        return _CAPWIN_CACHE[key]
    hit = False
    try:
        n = at = 0
        counts = None
        for sym, d in FRAMES.items():
            if uni.is_future(sym) or d is None or len(d) < config.RSIX_MIN_HISTORY:
                continue
            r, _, alo, _, _ = rsix_mod.series(d)
            m = ((r <= alo + config.RSIX_AT_TOL) & r.notna() & alo.notna()).iloc[-42:]
            counts = m.astype(float).values + (counts if counts is not None else 0)
            n += 1
        if counts is not None and n > 50:
            hit = bool((100 * counts / n >= 2.0).any())
    except Exception:
        hit = False
    _CAPWIN_CACHE.clear()
    _CAPWIN_CACHE[key] = hit
    return hit


def _weather() -> dict:
    """Current awareness states -> per-family expectancy tilt with NAMED setups (banner panel)."""
    if LATEST in _WEATHER_CACHE:
        return _WEATHER_CACHE[LATEST]
    st, b50 = {}, None
    try:
        q = FRAMES.get("QQQ")
        st["QQQ 10>20 rising"] = bool(q is not None and len(q) > 30
            and q["close"].iloc[-1] > q["ema10"].iloc[-1] > q["ema20"].iloc[-1]
            and q["ema10"].iloc[-1] > q["ema10"].iloc[-6] and q["ema20"].iloc[-1] > q["ema20"].iloc[-6])
        above = tot = 0
        for sym, d in FRAMES.items():
            if uni.is_future(sym) or d is None or len(d) < 60:
                continue
            c, s50 = d["close"].iloc[-1], d["sma50"].iloc[-1]
            if pd.notna(c) and pd.notna(s50):
                tot += 1
                above += c > s50
        b50 = 100 * above / tot if tot else None
        st["breadth washed-out"] = bool(b50 is not None and b50 <= 35)
        spy_f = FRAMES.get("SPY")
        if spy_f is not None:
            ext = float((spy_f["close"].iloc[-1] - spy_f["sma50"].iloc[-1]) / spy_f["atr14"].iloc[-1])
            st["SPY ext>=3ATR"] = ext >= 3
            dn = (spy_f["close"].pct_change() < -0.002) & (spy_f["volume"] > spy_f["volume"].shift(1))
            st["distribution days>=4"] = bool(dn.iloc[-25:].sum() >= 4)
        vx = FRAMES.get("^VIX")
        v = float(vx["close"].iloc[-1]) if vx is not None else None
        st["VIX>26 fear"] = bool(v and v > 26)
        st["VIX<16 calm"] = bool(v and v < 16)
        st["capitulation window"] = _capitulation_window()
    except Exception:
        pass
    fams = {}
    for fam, names in _WEATHER_SETUPS.items():
        score = sum(_WEATHER_DELTAS[k][fam] for k, on in st.items() if on and k in _WEATHER_DELTAS)
        why = [f"{k} {_WEATHER_DELTAS[k][fam]:+.2f}R" for k, on in st.items()
               if on and k in _WEATHER_DELTAS and abs(_WEATHER_DELTAS[k][fam]) >= 0.03]
        fams[fam] = {"score": round(score, 2), "setups": names,
                     "tone": "up" if score >= 0.15 else ("dn" if score <= -0.15 else "flat"),
                     "why": why}
    out = {"states": {k: bool(v) for k, v in st.items()}, "families": fams,
           "b50": (round(b50, 1) if b50 is not None else None),
           "caution": bool(st.get("capitulation window"))}
    _WEATHER_CACHE.clear()
    _WEATHER_CACHE[LATEST] = out
    return out


@app.get("/api/ext")
def api_ext():
    """Extension-health + RSI-extreme data for a comma-list of symbols (badge back-fill):
    {sym: {kind, tfs: {tf: ext}, rsix}} — the client renders the badges goggles-aware."""
    syms = [s for s in (request.args.get("symbols") or "").upper().split(",") if s]
    date = request.args.get("date") or LATEST
    with _SCAN_LOCK:
        out = {s: {"kind": _ext_kind(s), "tfs": _ext_tfs(s, date), "rsix": _rsix_row(s, date)}
               for s in syms[:400]}
    return jsonify(out)


def _plain_tf_frame(sym: str, tf: str, date: str):
    """The (enriched) frame for one symbol on one TF, as everywhere else in the app."""
    if tf == "1D":
        fr = FRAMES.get(sym)
        return fr.loc[:date] if fr is not None else None
    if tf == "1W":
        return FRAMES_W.get(sym)
    return _tf_frame(sym, tf, date)


def _ratio_tf_frame(sym: str, den: str, tf: str, date: str, bars: int | None = None):
    """Enriched RATIO frame (sym / den close) on a TF — setups and charts see the same series.
    `bars` trims both legs first (structure classification doesn't need decades)."""
    num = _plain_tf_frame(sym, tf, date)
    dfr = _plain_tf_frame(den, tf, date)
    if num is None or dfr is None:
        return None
    if bars:
        num, dfr = num.iloc[-bars:], dfr.iloc[-bars:]
    j = num[["open", "high", "low", "close", "volume"]].join(dfr["close"].rename("_dc"), how="inner")
    if len(j) < 60:
        return None
    for c in ("open", "high", "low", "close"):
        j[c] = j[c] / j["_dc"]
    j = j[j["close"].notna()]
    if len(j) < 60:
        return None
    base = float(j["close"].iloc[-1])                  # anchor TODAY = 100: recent structure and
    if base > 0:                                       # S/R live near 100, so rounding/risk math
        for c in ("open", "high", "low", "close"):     # stay meaningful on decades-long ratios
            j[c] = j[c] * (100.0 / base)
    return add_indicators(j.drop(columns="_dc"))


@app.get("/api/tf_setups")
def api_tf_setups():
    """Run EVERY setup detector on ONE (symbol[, /denominator], timeframe) frame as of now —
    'which setups does this chart fit into'. Also returns the frame's nearest S/R so the risk
    math is in the SAME units as the chart (ratio setups get ratio-unit levels)."""
    sym = (request.args.get("symbol") or "").upper()
    den = (request.args.get("den") or "").upper()
    tf = request.args.get("tf") or "1D"
    date = request.args.get("date") or LATEST
    key = (sym, den, tf, date, settings.setup_sig("episodic_pivot"))
    if key in _TF_SETUPS_CACHE:
        return jsonify(_TF_SETUPS_CACHE[key])
    with _SCAN_LOCK:
        fr = _ratio_tf_frame(sym, den, tf, date) if den else _plain_tf_frame(sym, tf, date)
    if fr is None or len(fr) < 60:
        return jsonify({"symbol": sym, "den": den, "tf": tf, "hits": [],
                        "error": f"no {tf} data for {sym}{'/' + den if den else ''}"})
    try:
        with settings.apply():
            hits = setups._run(setups.DETECTORS + setups.MULTI_TF_DETECTORS, fr, None)
        srl = sr.sr_levels(fr)
        sup = (srl.get("support") or [{}])[0].get("price")
        res = (srl.get("resistance") or [{}])[0].get("price")
    except Exception as e:
        return jsonify({"symbol": sym, "den": den, "tf": tf, "hits": [], "error": str(e)})
    out = {"symbol": sym, "den": den, "tf": tf, "hits": awareness._py(hits),
           "sup": sup, "res": res, "px": round(float(fr["close"].iloc[-1]), 4)}
    _TF_SETUPS_CACHE[key] = out
    if len(_TF_SETUPS_CACHE) > 500:
        _TF_SETUPS_CACHE.clear()
    return jsonify(out)


_TF_SETUPS_CACHE: dict = {}


@app.get("/api/setup_hit")
def api_setup_hit():
    """Re-detect ONE setup on the as-of slice at `date` — history charts use this to draw the
    same pattern anatomy (rims/peaks/shoulders/necklines/target) as live scan rows."""
    sym = (request.args.get("symbol") or "").upper()
    setup = request.args.get("setup") or ""
    date = request.args.get("date") or LATEST
    fn = next((f for f, n in setups.SETUP_OF.items() if n == setup), None)
    d = FRAMES.get(sym)
    if fn is None or d is None:
        return jsonify({"error": f"unknown setup/symbol {setup}/{sym}", "hit": None})
    try:
        with settings.apply():
            m = fn(d.loc[:date])
    except Exception as e:
        return jsonify({"error": str(e), "hit": None})
    return jsonify({"symbol": sym, "setup": setup, "date": date, "hit": m})


def _intraday_gappers(date: str) -> list[dict]:
    """RTH-only: the EOD daily bar doesn't exist yet, so detect_gapper (which reads the last daily row) can't
    see today's gap and the Gappers tab shows the PRIOR day. Here we synthesize each liquid name's TODAY daily
    bar from the live 5m session (open = session open, hi/lo running, close = last, volume = calibrated
    day-vol), run the SAME detect_gapper on an EPHEMERAL per-name copy, and return scan-shaped gapper rows so
    the tab populates intraday (Amir 2026-07-07). Never mutates FRAMES; never runs on replay. Cheap: a
    session-open gap pre-filter means only real gap candidates take the add_indicators path."""
    import ep_news
    if not market._is_partial(None):
        return []
    today = pd.Timestamp(market._today_et())
    u = UNIVERSE
    rows: list[dict] = []
    for sym, f5 in list(FRAMES5.items()):            # snapshot: the poller mutates FRAMES5 unlocked
        if uni.is_future(sym) or f5 is None or not len(f5):
            continue
        d = FRAMES.get(sym)
        if d is None or not len(d):
            continue
        t = f5[f5.index.normalize() == today]
        if not len(t):
            continue
        prior_close = float(d["close"].iloc[-1])
        s_open = float(t["open"].iloc[0])
        if prior_close <= 0 or (s_open / prior_close - 1) * 100 < config.GAP_MIN_PCT:
            continue                                 # cheap pre-filter: only real session-open gappers
        dayv = ep_news._live_dayvol(sym)             # calibrated (consolidated) day-vol -> correct rvol vs vol_avg50
        if not dayv:
            dayv = float(t["volume"].sum()) * float(ep_news.vol_factor().get(sym, 22.0))
        synth = pd.DataFrame({"open": [s_open], "high": [float(t["high"].max())],
                              "low": [float(t["low"].min())], "close": [float(t["close"].iloc[-1])],
                              "volume": [dayv]}, index=[today])
        raw = datastore.load_bars(sym)
        if raw is None or not len(raw):
            continue
        raw = raw[raw.index < today]                 # drop any stale/partial today daily row
        try:
            d2 = add_indicators(pd.concat([raw, synth]))
            meta = u.loc[sym] if (u is not None and sym in u.index) else None
            grows = scan._rows_from_enriched(sym, d2, meta, only={"gapper"})
        except Exception:
            continue
        for r in grows:
            r["intraday"] = True                     # live-session synth gapper (approximate, updates intraday)
        rows.extend(grows)
    rows.sort(key=lambda r: -(r.get("gap_pct") or 0))
    return rows


_IG_CACHE = {"t": 0.0, "rows": None}


def _intraday_gappers_cached() -> list[dict]:
    """~1-min cache around _intraday_gappers so /api/scan doesn't recompute the synth-bar pass on every poll.
    Computed OUTSIDE _SCAN_LOCK (in api_scan) so it never extends the lock hold."""
    now = time.time()
    if _IG_CACHE["rows"] is None or now - _IG_CACHE["t"] > 55:   # matches the ~60s UI refresh cadence
        _IG_CACHE["rows"] = _intraday_gappers(LATEST)
        _IG_CACHE["t"] = now
    return _IG_CACHE["rows"]


@app.get("/api/scan")
def api_scan():
    date = request.args.get("date") or LATEST
    spy = FRAMES.get("SPY")
    if spy is not None and len(spy) and date < str(spy.index[0].date()):
        return jsonify({"error": f"no data for {date} — daily history starts "
                                 f"{spy.index[0].date()}", "tables": {}, "charts": {}})
    try:
        # intraday gappers: run the heavy synth-bar + detect_gapper pass OUTSIDE _SCAN_LOCK (per-minute cache)
        # so it doesn't extend the lock hold and starve rvol_leaders/_setup_context, which need the SAME lock
        # (Amir 2026-07-07). Only the cheap swap+concat runs under the lock.
        ig = None
        if market._is_partial(None) and date >= LATEST:
            try:
                ig = _intraday_gappers_cached()
            except Exception:
                ig = None
        with _SCAN_LOCK:                    # cache hit is instant; a miss recomputes + waits out a reload
            hits = _scan_cached(date)
            if ig is not None:              # RTH live: swap the stale prior-day gappers for today's synth gappers
                if hits is not None and len(hits) and "setup" in hits:
                    hits = hits[hits["setup"] != "gapper"]
                if ig:
                    hits = pd.concat([hits, pd.DataFrame(ig)], ignore_index=True) if (hits is not None and len(hits)) else pd.DataFrame(ig)
            tables, charts = report.compute_payload(hits, asof=date, embed_charts=False)
            _annotate_risk_reward(tables, date)
            # backburner tab = the multi-TF workflow: only candidates ARMED somewhere (or firing
            # today), each row showing WHICH timeframes are armed (Amir 2026-07-03)
            keep = []
            for row in tables.get("backburner") or []:
                tfs = _bb_armed_tfs(row["symbol"], date)
                row["armed_tfs"] = " ".join(tfs)
                if tfs or str(row.get("state", "")).startswith("entry"):
                    keep.append(row)
            if "backburner" in tables:
                tables["backburner"] = keep
            # data-freshness: loaded names that actually printed a bar ON `date`, SPY-guarded so a
            # market holiday isn't mistaken for "incomplete". A low ratio on a real trading day means
            # the data is still filling / has a gap -> self-correct by pulling (below, off the lock).
            _ts = pd.Timestamp(date)
            _spy = FRAMES.get("SPY")
            _trading = _spy is not None and len(_spy) and bool((_spy.index.normalize() == _ts).any())
            _on = (sum(1 for f in FRAMES.values() if len(f)
                       and (f.index[-1].normalize() == _ts or bool((f.index.normalize() == _ts).any())))
                   if _trading else 0)
    except Exception as e:  # surface to the browser instead of a blank page
        return jsonify({"error": f"scan error: {e}", "tables": {}, "charts": {}})
    _loaded = len(FRAMES)
    _incomplete = bool(_trading and _loaded and _on < _loaded * 0.90)
    refreshing = False
    # auto-pull only for the current/most-recent day: an EOD pull fills TODAY's still-forming bars but
    # can't backfill an old history gap, so past days get the note (below) without a futile heavy reload.
    if _incomplete and date >= LATEST and date not in _AUTOPULL_DONE and not UPDATE["running"]:
        _AUTOPULL_DONE.add(date)             # self-correct ONCE per date per session (no pull loops)
        refreshing = start_update()          # background pull + reload; the UI polls /api/update/status
    return jsonify({"date": date, "n": int(len(hits)), "tables": tables, "charts": charts,
                    "coverage": {"on_date": _on, "loaded": _loaded, "incomplete": _incomplete,
                                 "refreshing": refreshing, "latest": date >= LATEST}})


@app.post("/api/chart")
def api_chart():
    """Build ONE chart on demand (no upfront embedding). Body: {symbol, tf, date, hit:{...}} —
    the row from the table carries level/trigger/shading; we just render it from the RAM frame."""
    b = request.get_json(silent=True) or {}
    sym, tf, date = b.get("symbol"), b.get("tf", "1D"), b.get("date") or LATEST
    intraday = tf in datastore.INTRA_TFS or tf == "5m"
    with _SCAN_LOCK:                        # don't read frames mid-reload
        if tf == "1D":
            frame = FRAMES.get(sym)
        elif tf == "1W":
            frame = FRAMES_W.get(sym)
        elif tf == "5m":                    # 5m comes from the Tiingo store (15m store can't make it)
            raw5 = _bb_tf_frame(sym, "5m")
            frame = add_indicators(raw5) if raw5 is not None and len(raw5) > 60 else None
        else:                               # 2D..6M resampled, or intraday built from 15m
            frame = _tf_frame(sym, tf, None if intraday else date)
        if frame is None:
            return jsonify({"error": "frame not loaded"})
        den = (b.get("den") or "").upper()             # ratio chart: symbol ÷ denominator close
        if den:
            frame = _ratio_tf_frame(sym, den, tf, date)
            if frame is None:
                return jsonify({"error": f"no ratio frame for {sym}/{den} on {tf}"})
        
        import scanner_core
        frame = scanner_core.calc_larssson_line(frame)
        payload = report._chart_payload(frame, b.get("hit") or {}, asof=None if intraday else date)

        if intraday and payload:                # daily-history line for pre-60-day context
            payload["daily"] = report._daily_context_line(FRAMES.get(sym))
        if payload and not den:                 # ◉ where/when the ALL-TIME RSI extreme was hit
            payload["rsix"] = report._anat(_rsix_chart_marks(frame), intraday)
    return jsonify(payload or {"error": "no chart"})


@app.get("/api/mtf")
def api_mtf():
    """Multi-timeframe awareness profile for ONE ticker (computed on demand). Daily TFs honor the
    as-of date; intraday TFs resample from the latest Tiingo 5m snapshot."""
    sym = request.args.get("symbol")
    date = request.args.get("date") or LATEST
    historical = date < LATEST
    tfs = ["5m"] + config.MTF_TFS                       # 5m goggles row: native from the 5-min base
    with _SCAN_LOCK:
        daily = FRAMES.get(sym)
        if daily is None:
            return jsonify({"error": "frame not loaded", "symbol": sym, "profile": {}})
        if historical:
            # as-of a past date (a history fire): the yfinance 15m store is a rolling ~60d
            # window (today's snapshot beside an old fire = look-ahead), so intraday TFs come
            # from the TIINGO 5-min store instead (~3y, top-liquidity names), truncated at the
            # fire day's close; None when the symbol/date isn't covered. Pass the RAW 5m base so
            # mtf resamples 5m/15m/1h/4h natively (5m→1h == 5m→15m→1h: OHLCV agg is associative).
            base5 = None
            p5 = config.DATA_DIR / "tiingo" / "bars_5min" / f"{sym}{datastore.EXT}"
            if p5.exists():
                try:
                    d5 = datastore._read(p5)
                    if d5 is not None:
                        d5 = d5.loc[:date + " 23:59"].iloc[-12000:]   # ~3 months of 5m to the fire
                        if len(d5) > 200:
                            base5 = d5
                except Exception:
                    base5 = None
        else:
            # LIVE: the overlay-current 5m base (disk + FRAMES5). mtf resamples every TF from it,
            # and the native 5m row IS this base — so the goggles' finest read is truly live.
            base5 = _intraday_base(sym)
        intraday_na = historical and base5 is None
        prof = mtf.profile(daily.loc[:date], base5, tfs=tfs, is_future=uni.is_future(sym))
        # per-TF nearest S/R for the goggles risk/reward columns (validated-setup requirement):
        # the client derives $risk/$reward/r-R from these by goggles direction, and r/R-to-target
        # from the selected row's measured-move target
        for tf in tfs:
            s = prof.get(tf)
            if not s:
                continue
            fr = _tf_frame(sym, tf, date)
            if fr is None or len(fr) < 30:
                continue
            try:
                srl = sr.sr_levels(fr)
                s["sup"] = (srl.get("support") or [{}])[0].get("price")
                s["res"] = (srl.get("resistance") or [{}])[0].get("price")
                s["px"] = round(float(fr["close"].iloc[-1]), 2)
            except Exception:
                pass
    return jsonify({"symbol": sym, "tfs": tfs, "profile": prof,
                    "min_streak": config.ER_MIN_STREAK, "kind": _ext_kind(sym),
                    "asof": date, "historical": historical, "intraday_na": intraday_na})


def _exc_where(e: Exception) -> str:
    """Compact 'Type: msg @ file:line' for a caught endpoint error (and log the full traceback to the
    server log), so a swallowed error points at the exact source line instead of just the bare message."""
    import traceback
    tb = traceback.extract_tb(e.__traceback__)
    traceback.print_exc()
    loc = f"{tb[-1].filename.split('/')[-1]}:{tb[-1].lineno}" if tb else "?"
    return f"{type(e).__name__}: {e} @ {loc}"


@app.get("/api/market")
def api_market():
    """Top-down market situational-awareness score (long/out/short) for the as-of date. Cached per date
    (the breadth pass over all frames is ~0.5s); the cache is cleared whenever the RAM frames reload."""
    date = request.args.get("date") or LATEST
    t = request.args.get("t")                      # replay clock time "HH:MM" (or None/"live" = now)
    mkey = (date, t or "live")
    if mkey not in _MARKET_CACHE:
        try:
            with _SCAN_LOCK:
                f15, dasof = _resolve(date, t)
                if f15 and (not t or t == "live") and date >= LATEST:
                    sess = intraday.latest_session(f15)
                    if sess and sess > date:            # live session in progress
                        date, dasof = sess, pit.prior_session(sess, FRAMES)
                _MARKET_CACHE[mkey] = market.market_read(date, FRAMES, f15, daily_asof=dasof)
        except Exception as e:
            return jsonify({"error": f"market score error: {_exc_where(e)}", "close": None, "live": None})
    out = dict(_MARKET_CACHE[mkey])
    try:                                           # validated health score = the button read
        import health_score                        # (banner_rescore-validated; old composite
        hs_asof = (out.get("close") or {}).get("as_of") or date    # stays as legacy rows)
        out["health"] = health_score.read(hs_asof)
    except Exception as e:
        out["health"] = {"error": str(e)}
    _rsix_summary()                                # refresh breadth if the store was rebuilt
    out["rsix_breadth"] = _RSIX_BREADTH            # % of actives at their all-time RSI zones
    try:
        out["weather"] = _weather()                # measured setup-weather panel (T7 study)
    except Exception:
        pass
    try:
        out["data_health"] = _data_health()        # live 5m warming/rebuilding state (banner caveat)
    except Exception:
        pass
    return jsonify(out)


@app.get("/api/data_health")
def api_data_health():
    """Standalone live-data stability read (warming/rebuilding) so any panel can caveat its numbers
    cheaply without pulling the whole /api/market payload."""
    try:
        return jsonify(_data_health())
    except Exception as e:
        return jsonify({"stable": True, "warming": False, "error": str(e)})


_CROSS_NEWS = {"t": 0.0, "data": None, "running": False}   # cross-instrument news-reaction read, ~5-min cache


def _refresh_cross_news():
    import cross_news
    try:
        _CROSS_NEWS["data"] = cross_news.run()            # analyze proxies -> underlying + write latest.json
        _CROSS_NEWS["t"] = time.time()
    except Exception:
        pass
    finally:
        _CROSS_NEWS["running"] = False


@app.get("/api/cross_news")
def api_cross_news():
    """Cross-instrument news reaction (proxy events -> the UNDERLYING's move; e.g. MSTR selling -> BTC=F).
    NON-blocking: returns the cached read immediately and refreshes in the BACKGROUND when >~5min stale, so
    the ~9-proxy news fetch never blocks the banner. Cold start falls back to the last handoff file."""
    if not _CROSS_NEWS["running"] and (time.time() - _CROSS_NEWS["t"] > 300):
        _CROSS_NEWS["running"] = True
        threading.Thread(target=_refresh_cross_news, daemon=True).start()
    d = _CROSS_NEWS["data"]
    if d is None:                                         # cold: last run's file so the panel isn't blank
        try:
            p = config.DATA_DIR / "cross_news" / "latest.json"
            if p.exists():
                d = json.loads(p.read_text())
        except Exception:
            d = None
    return jsonify(d or {"reads": []})


def _seed_notif_log() -> None:
    """Rebuild the in-RAM grade-A alert log from today's events parquet so a mid-session RESTART
    doesn't wipe the day's alert history from the panel (Amir 2026-07-08)."""
    try:
        import alerts
        p = config.DATA_DIR / "significance_log" / f"{market._today_et()}_events.parquet"
        if not p.exists() or len(alerts.NOTIF_LOG):
            return
        df = pd.read_parquet(p)
        df = df[df["grade"] == "A"].tail(80)
        for _, r in df.iterrows():
            sym = (r.get("text") or "").split(" ")[0]
            alerts.NOTIF_LOG.append({"t": r.get("t"), "sym": sym, "text": r.get("text"),
                                     "side": r.get("side", ""), "kind": r.get("kind", ""),
                                     "grade": r.get("grade", ""), "action": r.get("action", ""),
                                     "why": ""})
        print(f"[alerts] restored {len(alerts.NOTIF_LOG)} grade-A events from today's ledger", flush=True)
    except Exception as e:
        print(f"[alerts] notif-log restore skipped: {e}", flush=True)


@app.get("/api/notifications")
def api_notifications():
    """Recent grade-A NOTIFIED events (mirrors the macOS pings) — the persistent, chart-clickable
    in-app record (Amir 2026-07-08: banners vanish; click-through only opens Script Editor)."""
    import alerts
    return jsonify({"rows": list(alerts.NOTIF_LOG)[::-1]})


@app.get("/api/ep_news")
def api_ep_news():
    """News analysis for an EP name AS-OF its TRIGGER (event) day — the catalyst that drove the episodic
    pivot, which may be a few sessions before today (Amir 2026-07-08: the most relevant news is on the EP
    trigger day, not today). Derives the event date from detect_episodic_pivot's mark (falls back to the
    as-of date for a non-EP pick), pulls that day's Tiingo headlines, and returns the rule-based read:
    catalyst + sentiment + the good-news/bad-reaction TELL (own-vs-peer aware) + the stories."""
    import ep_news
    sym = (request.args.get("sym") or "").upper()
    date = request.args.get("date") or _live_day()
    d = FRAMES.get(sym)
    if not sym or d is None or len(d) < 30:
        return jsonify({"error": f"no daily data for {sym or '(none)'}"})
    d = d.loc[:date]
    ev_date, subtype, ep_age = date, None, None
    try:
        ep = setups.detect_episodic_pivot(d)               # cheap for one symbol; gives the event mark + age
    except Exception:
        ep = None
    if ep and ep.get("marks"):
        ev_date = str(ep["marks"][0]["date"])[:10]         # the EP trigger day (may be a few bars back)
        subtype, ep_age = ep.get("ep_subtype"), ep.get("ep_age", 0)
    move_pct = None                                        # event-day move (close vs prior close) -> reaction tell
    try:
        evrows = d[d.index.normalize() == pd.Timestamp(ev_date)]
        if len(evrows):
            upto = d.loc[:evrows.index[-1]]
            if len(upto) >= 2:
                pc, cc = float(upto["close"].iloc[-2]), float(upto["close"].iloc[-1])
                move_pct = round((cc / pc - 1) * 100, 2) if pc else None
    except Exception:
        pass
    name = _universe_names().get(sym)
    heads = ep_news.news_for(sym, ev_date, days_back=2)    # event day + the two before (catalyst window)
    sig = ep_news.news_signal(sym, heads, move_pct=move_pct, name=name)
    stories = [{"title": h.get("title"), "source": h.get("source"), "published": h.get("published")}
               for h in heads[:12]]
    return jsonify({"symbol": sym, "name": name, "event_date": ev_date, "ep_subtype": subtype,
                    "ep_age": ep_age, "move_pct": move_pct, "n_stories": len(heads),
                    "catalyst": sig.get("catalyst"), "headline": sig.get("headline"),
                    "sent": sig.get("sent"), "conflicting": sig.get("conflicting"),
                    "news_react": sig.get("news_react"), "news_warn": sig.get("news_warn"),
                    "own_headline": sig.get("own_headline"), "peer_headline": sig.get("peer_headline"),
                    "stories": stories})


@app.get("/api/awareness")
def api_awareness():
    """The situational-awareness MAP for a (date, t) moment: per-instrument views (daily sequence +
    S/R ladder + gap/intraday overlay), rotation, and the ranked calls. Point-in-time: daily data
    anchors to the prior settled session on intraday reads; only completed 15m bars are visible."""
    date = request.args.get("date") or LATEST
    t = request.args.get("t")
    mode = request.args.get("mode")
    akey = (date, t or "live", mode or "", settings.aware_sig(), labels.sig())
    if akey not in _AWARE_CACHE:
        try:
            with _SCAN_LOCK:
                f15, anchor = _resolve(date, t, mode)
                if f15 and (not t or t == "live") and date >= LATEST:
                    sess = intraday.latest_session(f15)
                    if sess and sess > date:            # live session in progress (no daily bar yet)
                        date, anchor = sess, pit.prior_session(sess, FRAMES)
                dkey = (anchor, akey[3], akey[4])
                if dkey not in _AWARE_DAILY:
                    _AWARE_DAILY[dkey] = {}
                    while len(_AWARE_DAILY) > _AWARE_DAILY_MAX:
                        _AWARE_DAILY.popitem(last=False)
                payload = awareness.awareness_read(
                    date, t, mode, FRAMES, f15, anchor, daily_cache=_AWARE_DAILY[dkey])
                if not t or t == "live":
                    payload["alerts"] = alerts.recent()
                _AWARE_CACHE[akey] = payload
        except Exception as e:
            return jsonify({"error": f"awareness error: {_exc_where(e)}"})
    return jsonify(_AWARE_CACHE[akey])


_ALERT_EVERY = 3                       # compute the transition snapshot every Nth poll cycle
_FLUSH_EVERY = 5                       # write-through today's completed bars to disk every Nth cycle (~5 min)
_alert_tick = {"n": 0}


def _live_cycle(n_updated: int) -> None:
    """After each live poll: drop only the LIVE cache entries so the next request recomputes on
    minute-fresh bars (replay/close keys stay cached); every _ALERT_EVERY cycles, snapshot the
    regime features and emit transition alerts; every _FLUSH_EVERY cycles, write-through today's
    completed bars to the disk overlay (crash-safe + keeps disk readers live)."""
    with _SCAN_LOCK:
        for cache in (_AWARE_CACHE, _MARKET_CACHE, _GROUPS_CACHE, _PLAYBOOK_CACHE):
            for k in [k for k in cache if (k[1] if len(k) > 1 else "") == "live"]:
                cache.pop(k, None)
    # BUGS #84B heal: names the poller flagged as sitting across a gap get a targeted session re-fetch (off
    # the poll thread) so every missing slot lands with real per-slot volume instead of one inflated bar.
    gaps = tiingo_live.drain_gaps()
    if gaps and not _SESSION_SEED.get("running"):
        def _heal(names):
            try:
                if _seed_today_session(force=True, symbols=names):
                    with _SCAN_LOCK:
                        for cache in (_MARKET_CACHE, _GROUPS_CACHE, _AWARE_CACHE, _PLAYBOOK_CACHE):
                            for k in [k for k in cache if (k[1] if len(k) > 1 else "") == "live"]:
                                cache.pop(k, None)
            except Exception:
                pass
        threading.Thread(target=_heal, args=(gaps,), daemon=True).start()
    if _alert_tick["n"] % _FLUSH_EVERY == 0:           # persist completed bars off the poll thread
        threading.Thread(target=_flush_live_overlay, daemon=True).start()
    _alert_tick["n"] += 1
    if _alert_tick["n"] % _ALERT_EVERY:
        return
    try:
        _alert_snapshot()
    except Exception:
        pass


def _alert_snapshot() -> None:
    """Light regime snapshot off the live 5m store: internals (member pass), sector VWAP sides, SPY
    session state, and the 1h regimes of the key rotation ratios."""
    store = _fresh_frames5() or {}                     # Phase 2: live 5m base (freshness-gated)
    if not store or store.get("SPY") is None:
        return
    labeled = {s for s in store if labels.sector(s) or labels.themes(s)}
    mstats = awareness._member_stats15(store, labeled)
    ib = intraday.aggregate_internals(mstats, FRAMES) if mstats else None
    internals = ({**ib["internals"], "pct_vwap": ib["internals"]["pct_vwap"],
                  "cum_trend": ib["internals"]["cum_trend"]} if ib else None)
    sectors_vwap = {}
    for etf in config.AWARE_SECTOR_ETFS:
        st = intraday._session_stats(store.get(etf)) if store.get(etf) is not None else None
        if st:
            sectors_vwap[etf] = "above" if st["above_vwap"] else "below"
    spy_stats = intraday._session_stats(store["SPY"])
    spy = None
    if spy_stats:
        f = store["SPY"]
        tmask = f.index.date == f.index.date[-1]
        spy = {"vwap_side": "above" if spy_stats["above_vwap"] else "below",
               "last": round(spy_stats["last"], 2),
               "lod": round(float(f.loc[tmask, "low"].min()), 2),
               "hod": round(float(f.loc[tmask, "high"].max()), 2)}
    ratio_regimes = {}
    for a, b in (("QQQ", "SPY"), ("SMH", "SPY"), ("IWM", "SPY")):
        try:
            h1 = ratios.ratio_frame(a, b, {}, store, live=True, tf="1h")
            seq = sequence.sequence_read(h1, tf="1h") if h1 is not None and len(h1) >= 40 else None
            if seq:
                ratio_regimes[f"{a}/{b}"] = seq["regime"]
        except Exception:
            continue
    alerts.diff(alerts.snapshot(internals, sectors_vwap, spy, ratio_regimes))


_FRAMES5_LIVE_BARS = 30 * 78           # ~30 RTH sessions of 5m per symbol held live (decision #3)


def _seed_frames5() -> None:
    """Seed the live 5m RAM store from the Tiingo 5m disk store (calibrated -> consolidated-scale, matching
    FRAMES15's yfinance basis), tail-limited to ~30 sessions; the poller folds today's forming 5m bars on
    top. Idempotent (no-op once populated). Called ONLY from the live 5m path -> replay never pays for it.

    LIVE BASIS INVARIANT (store-corruption root cause, 2026-07-09): FRAMES5's TODAY bars must be RAW
    IEX — the poller, the today-seed and every live volume read (_dayv_checked bar_cons, slot_rvol)
    assume it. `raw_from=today` keeps the crash-recovery overlay's today-bars uncalibrated across a
    mid-session restart. Seeding them x22 made _flush_live_overlay ratchet x22 volume into the overlay
    per restart (write_5m_live's max-volume merge keeps the inflated row forever) and the EOD
    fold_5m_live then poisoned the MAIN store: 07-07 read x22^3, 07-08 x22^5.7 (AAPL 174T shares)."""
    if FRAMES5:
        return
    today = market._today_et()
    for sym in list(FRAMES):
        df = datastore.load_bars_5m(sym, raw_from=today)
        if df is not None and len(df) > 2:
            FRAMES5[sym] = df.iloc[-_FRAMES5_LIVE_BARS:]


_SESSION_SEED = {"date": None, "running": False, "last_auto": None}   # seeded-date + in-flight guard + last auto safety-net run
_SEED_INDEX_FIRST = ("SPY", "QQQ", "IWM", "DIA")     # pinned ahead of dollar-vol in the seed (plan E)


def _seed_today_session(force: bool = False, symbols: list[str] | None = None) -> int:
    """Backfill today's RTH 5m session from yfinance (Smart Pull strategy) instead of Tiingo.
    Equities only. Returns #symbols filled."""
    if not market._is_partial(None):
        return 0
    today = market._today_et()
    if symbols is None and not force and _SESSION_SEED["date"] == today:
        return 0
        
    _SESSION_SEED["running"] = True
    try:
        ts_today = pd.Timestamp(today)
        # 1. Determine Smart Watchlist
        # The user has ~2000 active tickers. We pull top 350 by liquidity plus gappers to stay under yf limits.
        baselines = {}
        eq_syms = [s for s in (symbols or uni.active_symbols()) if not uni.is_future(s)]
        for sym in eq_syms:
            df = FRAMES.get(sym)
            if df is not None and not df.empty:
                baselines[sym] = {'prev_close': df['close'].iloc[-1], 'dollar_vol': df.get('dollar_vol', pd.Series([0])).iloc[-1]}
                
        sorted_liq = sorted(baselines.keys(), key=lambda x: baselines[x]['dollar_vol'], reverse=True)
        smart_watchlist = set(sorted_liq[:350]) # Core highly liquid watchlist
        
        # (Optional: we could do a pre-market sweep for gappers here, but 350 is enough for core radar without hitting limits)
        
        todo = list(smart_watchlist)
        print(f"[yfinance-5m] {dt.datetime.now():%H:%M} Smart Pull starting for {len(todo)} tickers...", flush=True)
        
        # 2. Pull 5m data using datastore's safe batching logic
        fetched_data = datastore.download_5m_today(todo)
        
        n = 0
        for s, df in fetched_data.items():
            datastore.write_5m_live(s, df)             # RAW -> disk overlay
            with _SCAN_LOCK:
                f = FRAMES5.get(s)
                if f is None or not len(f):
                    FRAMES5[s] = df
                else:
                    hist = f[f.index.normalize() < ts_today]
                    today_merged = datastore.floor_intraday_grid(
                        pd.concat([df, f[f.index.normalize() == ts_today]]))
                    FRAMES5[s] = pd.concat([hist, today_merged]).sort_index()
            n += 1
            
        if symbols is None:
            _SESSION_SEED["date"] = today
        print(f"[yfinance-5m] {dt.datetime.now():%H:%M} today-seed done: updated {n} tickers", flush=True)
        return n
    except Exception as e:
        print(f"[yfinance-5m] seed failed: {e}")
        return 0
    finally:
        _SESSION_SEED["running"] = False

def _flush_live_overlay() -> int:
    """Poller write-through: persist today's COMPLETED (raw) 5m bars from FRAMES5 to the disk overlay so the
    store stays current + crash-safe with NO extra API cost (FRAMES5 today bars are already raw IEX). Drops
    the forming (last) bar. Cheap; called every few poll cycles.

    BASIS GUARD (2026-07-09 store corruption): the overlay must stay RAW-IEX. When the poller has a
    quote-basis day-cumulative for the name (DAYVOL, same raw basis), a frame whose today-sum exceeds it
    by far can only be a calibrated/poisoned frame — persisting it would ratchet x22 into the overlay
    (max-volume merge) and, via the EOD fold, into the main store. Skip + count, never write."""
    today = pd.Timestamp(market._today_et())
    n = 0
    for s, f in list(FRAMES5.items()):
        if uni.is_future(s) or f is None or not len(f):
            continue
        td = f[f.index.normalize() == today]
        if len(td) >= 2:                                   # >=2 today bars -> at least one completed
            try:
                dv = tiingo_live.DAYVOL.get(s)
                if dv and float(td["volume"].iloc[:-1].sum()) > 1.5 * float(dv):
                    tiingo_live.STATE["overlay_basis_reject"] = \
                        int(tiingo_live.STATE.get("overlay_basis_reject") or 0) + 1
                    syms = tiingo_live.STATE.setdefault("overlay_basis_reject_syms", {})
                    syms[s] = syms.get(s, 0) + 1          # per-symbol tally (Amir 2026-07-14: which
                    continue                               # names keep tripping this?) -> /api/live/status
                datastore.write_5m_live(s, td.iloc[:-1])   # persist completed (raw); forming bar stays RAM-only
                n += 1
            except Exception:
                continue
    return n


def _data_health() -> dict:
    """Live-data stability read for the UI (Amir 2026-07-07): is the intraday 5m store COMPLETE, or is it
    still WARMING / rebuilding? Motivated by the slow ~universe re-seed that degrades panels for minutes after
    a restart. Cheap — one pass over FRAMES5 index tails. Auto-clears once the today-session seed finishes and
    coverage is high. Signals: _SESSION_SEED (running / not-yet-today = the re-seed) and UPDATE (daily pull).
    Only the LIVE 5m store can be 'warming'; the daily store fills at EOD, so off-hours = stable unless a
    daily pull is running."""
    refreshing = bool(UPDATE.get("running"))
    if not market._is_partial(None):                       # market closed: live 5m isn't the driver
        return {"stable": not refreshing, "warming": refreshing, "seeding": False,
                "refreshing": refreshing, "pct_5m": None, "have": None, "n": None,
                "reason": ("refreshing daily data" if refreshing else "")}
    today = pd.Timestamp(market._today_et())
    # denominator = the ACTIVE equity universe (not just what's in FRAMES5 yet) so % is honest+monotonic
    # while FRAMES5 is still populating post-restart; .get() per symbol is race-safe vs the seed thread
    eq = [s for s in (uni.active_symbols() or ()) if not uni.is_future(s)]
    n = len(eq)

    def _has_today(s):
        f = FRAMES5.get(s)
        return f is not None and len(f) and f.index[-1].normalize() == today
    have = sum(1 for s in eq if _has_today(s))
    pct = round(100 * have / n) if n else 0
    # the today-session seed is in-flight, OR hasn't run yet this session (post-restart gap before _bg fires)
    seeding = bool(_SESSION_SEED.get("running")) or _SESSION_SEED.get("date") != market._today_et()
    warming = bool(seeding or refreshing or (n and pct < 50))   # <50% = deep-incomplete safety net
    if seeding:
        reason = f"seeding today's 5m session — {have}/{n} names ({pct}%)"
    elif refreshing:
        reason = "refreshing daily data"
    elif warming:
        reason = f"5m coverage filling — {have}/{n} names ({pct}%)"
    else:
        reason = ""
    return {"stable": not warming, "warming": warming, "seeding": seeding,
            "refreshing": refreshing, "pct_5m": pct, "have": have, "n": n, "reason": reason}


def _ensure_live_poller() -> dict:
    """Seed the live 5m RAM store (FRAMES5) from disk + today's session (yfinance), and start the Tiingo poller
    folding the forming 5m bar on top — the SINGLE live intraday feed since Phase 3c (equities only; IEX carries
    no futures). Idempotent (seed no-ops once populated; the today-seed no-ops once done for the day; the poller
    no-ops while running)."""
    _seed_frames5()
    # fill today's 09:30->now session in the BACKGROUND (a ~universe yfinance pull) so the first live
    # request doesn't block; the session lands within ~a poll cycle. Guarded against concurrent spawns.
    if (market._is_partial(None) and _SESSION_SEED["date"] != market._today_et()
            and not _SESSION_SEED["running"]):
        def _bg():
            try:
                n = _seed_today_session()                  # owns the running-guard + date bookkeeping
                if n:
                    with _SCAN_LOCK:
                        _MARKET_CACHE.clear(); _GROUPS_CACHE.clear(); _AWARE_CACHE.clear(); _PLAYBOOK_CACHE.clear()
            except Exception:
                pass
        threading.Thread(target=_bg, daemon=True).start()
    if FRAMES5:
        # Instead of tiingo_live, we start a 15-minute yfinance polling thread
        def _yfinance_poller_loop():
            while True:
                time.sleep(15 * 60) # Poll every 15 minutes
                if market._is_partial(None):
                    try:
                        n = _seed_today_session(force=True)
                        if n:
                            with _SCAN_LOCK:
                                _MARKET_CACHE.clear(); _GROUPS_CACHE.clear(); _AWARE_CACHE.clear(); _PLAYBOOK_CACHE.clear()
                            _live_cycle() # Trigger the live UI update
                    except Exception:
                        pass
        import threading
        threading.Thread(target=_yfinance_poller_loop, daemon=True).start()
    return FRAMES5


def _ensure_futures15() -> dict:
    """Populate the RAM 15m store with FUTURES ONLY (ES=F/NQ=F/… trade 24h — no IEX/5m coverage) from the
    yfinance 15m disk (rolling ~60d), for the futures perf/replay path. Equities use the 5m base; the live
    poller feeds FRAMES5, never this. Returns FRAMES15 (futures frames only)."""
    if not FRAMES15:
        for sym in list(FRAMES):
            if not uni.is_future(sym):
                continue
            df = datastore.load_bars_15m(sym)
            if df is not None and len(df) > 2:
                FRAMES15[sym] = df
    return FRAMES15


def _fresh_frames5() -> dict | None:
    """The live 5m RAM store IF the poller has folded today's forming bars (else None). Seeds FRAMES5 +
    starts the 5m poller on demand (only when a live 5m read is actually requested -> replay never pays),
    then probes SPY for a today-dated bar. Warms within one poll cycle of the first live request."""
    _ensure_live_poller()
    spy = FRAMES5.get("SPY")
    if spy is None or not len(spy) or spy.index[-1].date().isoformat() != market._today_et():
        return None
    # FRAMES5 is mutated UNLOCKED by the background seed thread; hand back a shallow SNAPSHOT so a live read
    # during a re-seed can't raise "dict changed size during iteration" while it iterates the store. DataFrames
    # are REPLACED (not mutated in place) on reseed, so a shallow copy is a consistent view. Retry the copy
    # itself past the narrow mutation window (per-symbol writes are ~a Tiingo call apart).
    for _ in range(10):
        try:
            return dict(FRAMES5)
        except RuntimeError:
            continue
    return None


def _session_frames15(date: str, t: str | None) -> dict | None:
    """Replay view for the FUTURES perf panel: the futures 15m store truncated so `date`'s bars stop at `t`
    (HH:MM ET), later days dropped. Equities replay from the 5m store (_asof_frames5). None if no futures 15m
    reaches `date`. Truncation lives in pit.py (shared with the validator)."""
    return pit.session_frames15(_ensure_futures15(), date, t)


def _read_frames5(date: str, t: str | None) -> dict | None:
    """The 5m frame set for a (date, replay-time) request — the intraday BASE for market/awareness
    (Phase 2): a truncated REPLAY view from the deep Tiingo 5m store when t is a clock time, else the
    LIVE poller-fed 5m store while the session is live."""
    if t and t != "live":
        return _asof_frames5(date, t)
    # LIVE while the market is open NOW (a partial session) even if the daily store already carries
    # today's bar (date == LATEST — e.g. a partial daily bar was written); or when viewing a date past
    # the settled store. _fresh_frames5 still returns None until the poller has folded today's first bar.
    if date > LATEST or (date >= LATEST and market._is_partial(None)):
        return _fresh_frames5()
    return None


def _perf_bars(date: str, t: str | None) -> dict | None:
    """Intraday store for the EQUITY performance panels (since-open / since-prev-close), now on the 5m
    base. Replay (clock t): the deep Tiingo 5m store truncated at (date, t). Live: the poller-fed 5m
    store when fresh, else the last settled session from disk. performance._intraday_returns is
    resolution-agnostic (first-bar open, last close, prior-session close), so 5m is a drop-in for 15m.
    Futures are handled separately (they keep the yfinance-15m store — no 5m coverage).

    Live: MERGE the disk 5m (which carries the true 09:30 session open — the same base the CHART reads
    via _intraday_base) with the poller store (freshest last bar), so perf's since-open matches the chart
    instead of reading the poller's first-fold open / a stale last."""
    if t and t != "live":
        return _asof_frames5(date, t)
    live = _fresh_frames5()
    disk = _day5_store(LATEST)                       # true session opens from the disk 5m base
    if not live:
        return disk or None
    if not disk:
        return live
    out = dict(disk)
    for s, lf in live.items():                       # graft the freshest poller bars onto the disk frames
        df = disk.get(s)
        if df is None or not len(df):
            out[s] = lf
        elif len(lf) and lf.index[-1] > df.index[-1]:
            out[s] = pd.concat([df, lf[lf.index > df.index[-1]]])
    return out


_FRAMES5_MIN_DVOL_M = 1.0             # replay 5m universe floor (permissive: thin-name EPs like BLZE ~3M$ qualify)
_FRAMES5_REPLAY_BARS = 30 * 78        # ~30 RTH sessions kept per replay frame — deep enough for 1h/4h posture
_FRAMES5_CACHE_MAX = 6                # bound the per-date replay cache (deep frames -> keep RAM in check)


def _day5_store(date: str) -> dict:
    """Per-date 5m frames from the Tiingo store (calibrated -> consolidated-scale volume), sliced to
    [.. date EOD] and tail-limited so replaying ANY historical date stays cheap to scrub (the t-cut is
    a view). Point-in-time SAFE: the load set is filtered by liquidity as-of the PRIOR settled session
    (never `date`'s own outcome). Loads on demand from disk (deep history — reaches EPs months back),
    cached per date. First load for a date reads a few hundred parquets (~seconds); then instant."""
    if date not in _FRAMES5_CACHE:
        asof = market._prior_session(date, FRAMES) or date
        end = f"{date} 23:59:59"
        out: dict = {}
        for sym in list(FRAMES):
            dd = FRAMES[sym].loc[:asof]
            if not len(dd):
                continue
            dv = dd["dollar_vol"].iloc[-1]
            if pd.isna(dv) or dv / 1e6 < _FRAMES5_MIN_DVOL_M:
                continue                              # illiquid as-of the prior session -> skip the disk read
            f = datastore.load_bars_5m(sym)           # calibrated (IEX x vol_factor = consolidated-scale)
            if f is None or not len(f):
                continue
            f = f.loc[:end]
            if not len(f) or f.index[-1].date().isoformat() < date:
                continue                              # no 5m ON `date` (didn't trade / not covered)
            out[sym] = f.iloc[-_FRAMES5_REPLAY_BARS:]   # ~30 sessions -> deep enough for 1h/4h posture
        _FRAMES5_CACHE[date] = out
        while len(_FRAMES5_CACHE) > _FRAMES5_CACHE_MAX:
            _FRAMES5_CACHE.pop(next(iter(_FRAMES5_CACHE)))   # evict oldest inserted (bound deep-frame RAM)
    return _FRAMES5_CACHE[date]


def _asof_frames5(date: str, t: str | None) -> dict | None:
    """Replay view of the 5m store as-of (date, clock t): the per-date frames truncated so `date`'s bars
    stop at t with strict no-look-ahead (reuses pit.session_frames15 — its open-time<cutoff rule is
    resolution-agnostic, correct at 5m: the 09:30 bar closes 09:35)."""
    store = _day5_store(date)
    return pit.session_frames15(store, date, t) if store else None


def _resolve(date: str, t: str | None, mode: str | None = None):
    """(frames15, daily_asof) for a request. A read is INTRADAY when it asks for a clock time, or is the live
    latest session (and not forced to close mode). For an intraday read ALL daily-derived data (scan/setups,
    RS, momentum, vol/macro, S/R) anchors to the PRIOR SETTLED session — because at time T the session's own
    daily bar has NOT closed yet — and the intraday overlay uses only bars COMPLETED by T (may be empty early,
    e.g. 09:30). A close/EOD read uses the date's own settled bar. Never look-ahead."""
    if mode == "close":
        return None, date
    # INTRADAY when: a clock replay time; OR a date past the settled store; OR the market is OPEN NOW
    # (a live partial session) even if today's daily bar is already in the store (date == LATEST). The
    # last clause is what keeps live internals/awareness alive once a partial daily bar exists.
    is_intraday = (bool(t and t != "live") or date > LATEST
                   or (date >= LATEST and market._is_partial(None)))
    if not is_intraday:
        return None, date                               # viewing a past settled EOD
    return _read_frames5(date, t), market._prior_session(date, FRAMES)   # Phase 2: 5m intraday base


@app.get("/api/groups")
def api_groups():
    """Sector + theme score leaderboard ('which groups are in gear'). Uses the intraday 15m read when
    fresh, else the daily bars. One pass over the universe; cached per date, cleared on frame/15m reload."""
    date = request.args.get("date") or LATEST
    t = request.args.get("t")
    gkey = (date, t or "live", labels.sig())   # theme membership + replay-time feed the scores
    if gkey not in _GROUPS_CACHE:
        try:
            with _SCAN_LOCK:
                f15, dasof = _resolve(date, t)
                _GROUPS_CACHE[gkey] = {"live": f15 is not None,
                                       "groups": market.group_scores(FRAMES, live=f15 is not None,
                                                                     frames15=f15, as_of=dasof)}
        except Exception as e:
            return jsonify({"error": f"groups error: {e}", "groups": []})
    return jsonify(_GROUPS_CACHE[gkey])


@app.get("/api/rs4h")
def api_rs4h():
    """Short-term index RS: 4h ratio-chart structure vs SPY (Amir's ratio workflow).
    Live: 10-min cache with the live 5m store appended; replay: per-date cache."""
    date = request.args.get("date") or LATEST
    live = date >= LATEST
    key = ("rs4h", date, int(time.time() // 600) if live else 0)
    if key not in _GROUPS_CACHE:
        try:
            import index_rs4h
            _GROUPS_CACHE[key] = index_rs4h.read(
                None if live else date, frames15=_fresh_frames5() if live else None)
        except Exception as e:
            return jsonify({"error": f"rs4h error: {e}"})
    return jsonify(_GROUPS_CACHE[key])


@app.get("/api/outlook")
def api_outlook():
    """Fit-free next-month risk outlook (awareness_lab artifact + current dial quintiles).
    Cheap (json + one matrix row) — cached per date."""
    date = request.args.get("date") or LATEST
    okey = ("outlook", date)
    if okey not in _GROUPS_CACHE:
        try:
            import outlook
            _GROUPS_CACHE[okey] = outlook.payload(None if date >= LATEST else date)   # today -> live outlook
        except Exception as e:
            return jsonify({"error": f"outlook error: {e}"})
    return jsonify(_GROUPS_CACHE[okey])


def _nn(v):
    """float32/NaN -> plain float or None (JSON-safe)."""
    return None if v is None or pd.isna(v) else round(float(v), 2)


def _hist_board(date: str) -> list[dict]:
    """Sector+theme rows for a PAST date from data/perf_history (same shape as market.theme_board,
    minus the live 'open' column). [] if the date isn't stored."""
    try:
        import build_perf_history as bph
        g = bph.load_groups(date)
    except Exception:
        return []
    rows = [{"kind": r.kind, "name": r.name, "n": int(r.n), "etf": market._group_etf(r.kind, r.name),
             "open": None, "d1": _nn(r.d1), "w1": _nn(r.w1), "m1": _nn(r.m1), "m3": _nn(r.m3),
             "ytd": _nn(r.ytd)} for r in g.itertuples()]
    rows.sort(key=lambda r: -(r["w1"] if r["w1"] is not None else -1e9))
    return rows


def _hist_tickers(date: str, symbols) -> list[dict] | None:
    """Per-ticker rows (perf_history) for `symbols` on a PAST date, shaped like performance.group_tickers;
    None if the date isn't stored."""
    try:
        import build_perf_history as bph
        df = bph.load_tickers(date)
    except Exception:
        return None
    if df is None or not len(df):
        return None
    df = df[df["symbol"].isin(set(symbols))]
    out = [{"symbol": r.symbol, "open_pct": None, "prev_close_pct": None, "d1_pct": _nn(r.d1),
            "wtd_pct": _nn(r.wtd), "w1_pct": _nn(r.w1), "m1_pct": _nn(r.m1), "m3_pct": _nn(r.m3),
            "ytd_pct": _nn(r.ytd)} for r in df.itertuples()]
    out.sort(key=lambda r: (r["m1_pct"] is not None, r["m1_pct"] or 0), reverse=True)
    return out


@app.get("/api/themeboard")
def api_themeboard():
    """Theme-tracker leaders board: sectors + themes x LIVE-open/1W/1M/3M/YTD MEDIAN member return
    (daily store + the live 5m store for the since-open column). This is the data the merged Perf
    panel renders. Live view is cycle-refreshed; historical (date/t) is cached stably."""
    date = request.args.get("date") or LATEST
    t    = request.args.get("t")
    live = (date >= LATEST and t is None)
    bkey = ("board", "live" if live else (date, t), labels.sig())   # "live" key is dropped each poll cycle
    if bkey not in _GROUPS_CACHE and date < LATEST and t is None:
        rows = _hist_board(date)                                  # precomputed (build_perf_history.py)
        if rows:
            _GROUPS_CACHE[bkey] = {"asof": date, "rows": rows, "source": "perf_history"}
    if bkey not in _GROUPS_CACHE:
        try:
            bars5 = _perf_bars(date, t)                             # outside the lock (may build the 5m store)
            with _SCAN_LOCK:
                _, dasof = _resolve(date, t)
                _GROUPS_CACHE[bkey] = {"asof": dasof or date,
                                       "rows": market.theme_board(FRAMES, as_of=dasof, bars5=bars5)}
        except Exception as e:
            return jsonify({"error": f"themeboard error: {e}", "rows": []})
    return jsonify(_GROUPS_CACHE[bkey])


@app.get("/api/daytype")
def api_daytype():
    """H3 day-type strip. Historical as-of date (time-travel / replay slider) -> the BATCH
    checkpoint probabilities at the latest checkpoint <= t (daytype_live.score_history);
    current session -> the live pre-open read from the frozen production model
    (daytype_live.score_payload). Cheap — cached artifacts, no scan lock."""
    import daytype_live
    sym = request.args.get("sym", "SPY")
    if sym not in daytype_live.SYMS:
        return jsonify({"error": f"sym must be one of {daytype_live.SYMS}"})
    date = request.args.get("date")
    t = request.args.get("t")
    try:
        try:
            from tiingo_live import session_phase
            _rth = session_phase() == "rth"
        except Exception:
            _rth = False
        # POST-CLOSE same ET day (Amir 2026-07-08): after 16:00 the session's 5m data is complete in
        # the live store, so today's read is the live scorer FROZEN AT THE CLOSE — not the pre-open
        # fallback (yesterday's framing) and not score_history (today has no batch row yet -> error).
        # The date rolls at ET midnight, after which the pre-open read for the NEW day takes over.
        _post = (not _rth) and _session_over_today()

        def _live(tag: bool):
            p = daytype_live.score_live(sym, frames15=_fresh_frames5())
            if tag and p.get("mode") == "live-intraday":
                p["session_over"] = True               # UI relabels '● LIVE' -> 'TODAY at CLOSE'
            return jsonify(p)
        # LIVE session TODAY (no replay-time scrub) must be served LIVE even when LATEST==today — else the
        # `date<=LATEST` history branch below swallows today (which has no row in the history parquet) and the
        # whole live day-type strip errors. Past dates + replay-time scrubs (t set) still take the history path.
        if date and date == market._today_et() and not t and (_rth or _post):
            return _live(_post)
        if date and date <= LATEST:
            return jsonify(daytype_live.score_history(sym, date, t))
        if _rth or _post:                              # market open (or just closed), no explicit date
            return _live(_post)                        # live 5m store -> universe breadth
        return jsonify(daytype_live.score_payload(sym))
    except Exception as e:
        return jsonify({"error": str(e)})


@app.get("/api/playbook")
def api_playbook():
    """Natural-language Playbook — long/short candidate groups + named tickers with posture + S/R.
    Reuses the market read + group leaderboard; cached per (date, labels.sig)."""
    date = request.args.get("date") or LATEST
    t = request.args.get("t")
    mode = request.args.get("mode")            # "close" forces the settled EOD read even when intraday is fresh
    tk = t or "live"
    pkey = (date, tk, mode or "", labels.sig())
    if pkey not in _PLAYBOOK_CACHE:
        try:
            with _SCAN_LOCK:
                f15, dasof = _resolve(date, t, mode)              # settled daily anchor (prior session if intraday)
                hits = _scan_cached(dasof)                         # setups from the settled session (no look-ahead)
                mr = market.market_read(date, FRAMES, f15, daily_asof=dasof)
                groups = market.group_scores(FRAMES, live=f15 is not None, frames15=f15, as_of=dasof)
                _PLAYBOOK_CACHE[pkey] = narrative.playbook(mr, groups, FRAMES, f15, as_of=dasof, hits=hits)
        except Exception as e:
            return jsonify({"error": f"playbook error: {e}"})
    return jsonify(_PLAYBOOK_CACHE[pkey])


@app.get("/api/riders")
def api_riders():
    """EMA Rider hits on a chosen timeframe. tf in 1D / 2D..6M / 15m..12h. Intraday needs the 15m
    cache (download button) -> {needs_intraday:true} if absent."""
    tf = request.args.get("tf", "1D")
    date = request.args.get("date") or LATEST
    state = (request.args.get("state") or "all").lower()   # riding|touching|armed|saved|break|all
    if tf in datastore.INTRA_TFS and not config.BARS_DIR_15M.exists():
        return jsonify({"tf": tf, "needs_intraday": True, "tables": {"ema_rider_bull": [], "ema_rider_bear": []}})
    try:
        with _SCAN_LOCK:
            tables = _rider_scan(tf, date)
    except Exception as e:
        return jsonify({"error": f"rider scan error: {e}", "tf": tf, "tables": {}})
    if state != "all":                                     # cheap post-scan filter (cache stays state-agnostic)
        tables = {k: [r for r in v if r.get("state") == state] for k, v in tables.items()}
    tables = {k: report.attach_labels(v) for k, v in tables.items()}   # sector/themes at response time
    return jsonify({"tf": tf, "date": date, "state": state, "tables": tables})


@app.post("/api/reload_labels")
def api_reload_labels():
    """Re-read data/sectors.csv + data/themes.csv after an edit/re-seed. Instant — labels are
    attached at response time, not baked into the cached scans, so no rescan is needed."""
    labels.reload()
    _GROUPS_CACHE.clear()      # group scores depend on theme membership -> drop stale leaderboards
    _PLAYBOOK_CACHE.clear()
    _AWARE_CACHE.clear()       # theme membership feeds the map's baskets/rotation
    _AWARE_DAILY.clear()
    return jsonify({"ok": True, "themes": len(labels.all_themes())})


@app.get("/api/performance/universe")
def api_perf_universe():
    """Top/bottom 50 performers across the full universe + breadth stats. ?date= for backtest; ?t= replays
    since-open as of that clock time."""
    date = request.args.get("date") or LATEST
    t = request.args.get("t")
    b15 = _perf_bars(date, t)
    data = performance.universe_perf(FRAMES, as_of=date, top_n=50, bars15=b15)
    # attach sector + theme labels to each row
    labels.load()
    for row in data["top"] + data["bottom"]:
        sym = row["symbol"]
        row["sector"] = labels.sector(sym) or ""
        row["themes"] = labels.themes(sym)
    return jsonify(data)


@app.get("/api/performance/futures")
def api_perf_futures():
    """Per-ticker performance for all tracked futures symbols. ?date= for backtest mode."""
    import universe as uni
    date   = request.args.get("date") or LATEST
    t      = request.args.get("t")
    syms   = {s for s in uni.FUTURES_SYMBOLS if s in FRAMES}
    bars15 = (_session_frames15(date, t) if (t and t != "live") else performance._load_bars15(syms)) or {}
    tickers = (_hist_tickers(date, syms) if (date < LATEST and not t) else None) \
        or performance.group_tickers(syms, FRAMES, bars15, date)
    # attach friendly name + exchange from FUTURES map
    for row in tickers:
        name, exchange = uni.FUTURES.get(row["symbol"], ("", ""))
        row["name"] = name
        row["exchange"] = exchange
    return jsonify({"tickers": tickers, "as_of": date})


@app.get("/api/performance/group")
def api_perf_group():
    """Per-ticker performance + active setups for one sector or theme.
    ?group=Semiconductors&type=theme&date="""
    group   = request.args.get("group", "")
    gtype   = request.args.get("type", "theme")   # "theme" or "sector"
    date    = request.args.get("date") or LATEST
    t       = request.args.get("t")

    labels.load()
    if gtype == "theme":
        members = labels.tickers_in_theme(group) & set(FRAMES)
    else:
        members = {s for s, sec in (labels._SECTOR or {}).items()
                   if sec == group and s in FRAMES}

    if not members:
        return jsonify({"group": group, "type": gtype, "tickers": [], "as_of": date})

    bars15 = _perf_bars(date, t) or {}

    # per-ticker performance
    tickers = (_hist_tickers(date, members) if (date < LATEST and not t) else None) \
        or performance.group_tickers(members, FRAMES, bars15, date)

    # active setups per ticker from the scan cache (DataFrame with "symbol" + "setup" columns)
    cached = _scan_cached(date)
    sym_setups: dict[str, list[str]] = {}
    if cached is not None and len(cached):
        sub = cached[cached["symbol"].isin(members)][["symbol", "setup"]]
        for _, row in sub.iterrows():
            sym_setups.setdefault(row["symbol"], []).append(row["setup"])

    for row in tickers:
        row["setups"] = sym_setups.get(row["symbol"], [])

    return jsonify({"group": group, "type": gtype, "tickers": tickers, "as_of": date})


@app.get("/api/performance")
def api_performance():
    """Sector + theme performance across 5 timeframes (synthetic equal-weight + ETF where available).
    ?date= for backtest mode (daily TFs); since-open/prev-close use the LIVE 5m store (_perf_bars),
    the SAME source the drill-in (/api/performance/group) reads — so the top-level tracks the current
    session (the legacy load_bars_15m store is frozen) and is fast (the 5m store is cached, no re-read)."""
    date = request.args.get("date") or LATEST
    t    = request.args.get("t")
    data = performance.all_performance(FRAMES, as_of=date, bars15=_perf_bars(date, t) or {})
    return jsonify(data)


@app.get("/api/journal")
def api_journal_list():
    setup = request.args.get("setup") or None
    return jsonify({"entries": journal.list_all(setup), "summary": journal.setups_summary()})


@app.post("/api/journal")
def api_journal_add():
    b = request.json or {}
    sym  = (b.get("symbol") or "").strip().upper()
    date = b.get("date") or LATEST
    if not sym:
        return jsonify({"error": "symbol required"}), 400
    setups = b.get("setups") or [{"setup": b.get("setup", "other"), "setup_date": date}]
    entry = journal.add(sym, date, setups, b.get("tf", "1D"), b.get("comments", ""))
    return jsonify(entry)


@app.put("/api/journal/<int:entry_id>")
def api_journal_update(entry_id):
    b = request.json or {}
    setups = b.get("setups")  # None means don't update setups
    entry = journal.update(entry_id, b.get("comments", ""), setups)
    return jsonify(entry) if entry else (jsonify({"error": "not found"}), 404)


@app.delete("/api/journal/<int:entry_id>")
def api_journal_delete(entry_id):
    return jsonify({"ok": journal.delete(entry_id)})


@app.get("/api/settings")
def api_get_settings():
    return jsonify({"groups": settings.TUNABLE, "values": settings.current(),
                    "defaults": settings.defaults()})


@app.post("/api/settings")
def api_set_settings():
    body = request.get_json(silent=True) or {}
    return jsonify({"ok": True, "values": settings.save(body.get("overrides", {}))})


@app.post("/api/update")
def api_update():
    weekly = (request.args.get("mode") == "weekly")     # weekly = full refresh + universe re-evaluation
    return jsonify({"started": True} if start_update(weekly) else {"error": "already running"})


@app.get("/api/update/status")
def api_update_status():
    return jsonify({**UPDATE, "latest": _live_day()})   # post-update, 'Live' returns to TODAY (not the settled bar)


@app.post("/api/download_intraday")
def api_download_intraday():
    left = _intraday_cooldown_left()        # debounce: block re-pulling the same universe too soon
    if left > 0:
        return jsonify({"cooldown": True, "retry_in": left})
    return jsonify({"started": True} if start_download_intraday() else {"error": "already running"})


@app.get("/api/live/status")
def api_live_status():
    return jsonify(tiingo_live.STATE)


_REGISTRY_CACHE: dict = {}


@app.get("/api/setup_history")
def api_setup_history():
    """Every historical trigger of a setup, from the precomputed registry
    (build_setup_registry.py -> data/setup_registry/{setup}.parquet). Never recomputed here."""
    setup = request.args.get("setup") or ""
    rsi = request.args.get("rsi")
    if setup == "backburner" and rsi and rsi != "30":   # rsi-25 / rsi-20 registry variants
        setup = f"backburner_rsi{int(float(rsi))}"
    path = config.DATA_DIR / "setup_registry" / f"{setup}.parquet"
    if not path.exists():
        return jsonify({"error": f"history not built yet for {setup} — run build_setup_registry.py",
                        "setup": setup, "symbols": []})
    mt = path.stat().st_mtime
    if _REGISTRY_CACHE.get(setup, (None,))[0] != mt:
        df = pd.read_parquet(path)
        by = []
        for sym, g in df.groupby("symbol"):
            g = g.sort_values("date")
            by.append({"symbol": sym, "n": len(g), "last": g["date"].iloc[-1],
                       "fires": [{"date": r["date"], "state": r["state"]}
                                 for _, r in g.iterrows()]})
        by.sort(key=lambda r: r["last"], reverse=True)
        _REGISTRY_CACHE[setup] = (mt, {"setup": setup, "n_fires": len(df),
                                       "n_symbols": len(by), "built": dt.datetime.fromtimestamp(mt)
                                       .strftime("%Y-%m-%d %H:%M"), "symbols": by})
    return jsonify(_REGISTRY_CACHE[setup][1])


_FIRES_CACHE: dict = {}          # setup -> (mtime, {symbol: [{date, state}, ...]})


def _fires_by_symbol(setup: str, path) -> dict:
    """Per-symbol fire lists for one registry parquet, cached by file mtime."""
    mt = path.stat().st_mtime
    hit = _FIRES_CACHE.get(setup)
    if hit is None or hit[0] != mt:
        df = pd.read_parquet(path).sort_values("date")
        by = {str(sym): [{"date": str(r["date"])[:10], "state": str(r["state"])}
                         for _, r in g.iterrows()]
              for sym, g in df.groupby("symbol")}
        _FIRES_CACHE[setup] = (mt, by)
    return _FIRES_CACHE[setup][1]


@app.get("/api/fires")
def api_fires():
    """ALL registries' past triggers for ONE symbol — feeds the chart fire-marks selector.
    Every setup with a registry file is listed (n=0 when the symbol never fired) so the
    checklist is complete; backburner rsi-20/25 variants fold into the rsi-30 base registry."""
    import build_setup_registry as bsr
    sym = (request.args.get("sym") or "").upper()
    if not sym:
        return jsonify({"error": "sym required"})
    out = []
    reg_dir = config.DATA_DIR / "setup_registry"
    for path in sorted(reg_dir.glob("*.parquet")):
        setup = path.stem
        if setup.startswith("backburner_rsi"):
            continue                                   # variants of the same setup
        fires = _fires_by_symbol(setup, path).get(sym, [])
        out.append({"setup": setup,
                    "label": report.SETUP_LABELS.get(setup, setup),
                    "validated": setup in bsr.VALIDATED,
                    "short": setup in SHORT_SETUPS,
                    "n": len(fires), "fires": fires})
    out.sort(key=lambda r: (not r["validated"], r["setup"]))
    return jsonify({"symbol": sym, "setups": out})


def _group_rows(members) -> list[dict]:
    """Per-member performance rows for a theme/sector card, sorted by 1-week strength."""
    rows = []
    for s in sorted(members):
        f = FRAMES.get(s)
        if f is None or len(f) < 23:
            continue
        c = f["close"]
        rr = f.iloc[-1]
        rows.append({"symbol": s, "close": round(float(c.iloc[-1]), 2),
                     "d1": round(float(c.iloc[-1] / c.iloc[-2] - 1) * 100, 1),
                     "w1": round(float(c.iloc[-1] / c.iloc[-6] - 1) * 100, 1),
                     "m1": round(float(c.iloc[-1] / c.iloc[-22] - 1) * 100, 1),
                     "rvol": round(float(rr["rvol"]), 2) if pd.notna(rr.get("rvol")) else None,
                     "dollar_vol_m": (round(float(rr["dollar_vol"]) / 1e6, 1)
                                      if pd.notna(rr.get("dollar_vol")) else None),
                     "ext_tfs": _ext_tfs(s, LATEST), "kind": _ext_kind(s),
                     "rsix": _rsix_row(s, LATEST)})
    rows.sort(key=lambda r: -(r["w1"] if r["w1"] is not None else -999))
    return rows


_SYMBOLS_CACHE: list | None = None


@app.get("/api/symbols")
def api_symbols():
    """The searchable ticker list for the header search box — only symbols actually LOADED into
    FRAMES (those are the ones with a chart + card). {s: symbol, n: name, sec: sector}. FRAMES is
    fixed for the process, so build once and cache."""
    global _SYMBOLS_CACHE
    if _SYMBOLS_CACHE is None:
        out = []
        for s in sorted(FRAMES):
            try:
                nm = UNIVERSE.at[s, "name"] if UNIVERSE is not None and s in UNIVERSE.index else None
                nm = str(nm) if nm is not None and pd.notna(nm) else ""
            except Exception:
                nm = ""
            out.append({"s": s, "n": nm, "sec": labels.sector(s) or ""})
        _SYMBOLS_CACHE = out
    return jsonify(_SYMBOLS_CACHE)


@app.get("/api/ticker")
def api_ticker():
    """One symbol's situational card: sector/theme membership, performance ladder, structural
    read (daily sequence + S/R), today's intraday state when live, and fresh news/8-K/earnings.
    With ?theme=<name>: the THEME's card instead — proxy ETF + every member with performance."""
    import ep_news
    theme = request.args.get("theme")
    sector_etf = (request.args.get("sector_etf") or "").upper()
    if sector_etf:
        members = labels.spdr_members(sector_etf, list(FRAMES))
        rows = _group_rows(members)
        basket = {k: (round(float(np.mean([r[k] for r in rows if r[k] is not None])), 1)
                      if rows else None) for k in ("d1", "w1", "m1")}
        return jsonify({"theme": f"{sector_etf} sector", "etf": sector_etf,
                        "perf": basket, "members": rows[:25], "n_total": len(rows)})
    synth_sec = request.args.get("synth_sec")
    if synth_sec:
        rows = _group_rows([s for s in FRAMES if labels.sector(s) == synth_sec])
        basket = {k: (round(float(np.mean([r[k] for r in rows if r[k] is not None])), 1) if rows else None) for k in ("d1", "w1", "m1")}
        etf_sym = f"SYNTH_SEC_{synth_sec.replace(' ', '').replace('&', 'And').replace('-', '')}".upper()
        return jsonify({"theme": f"{synth_sec} (Synthetic)", "etf": etf_sym, "perf": basket, "members": rows[:30], "n_total": len(rows)})
        
    synth_thm = request.args.get("synth_thm")
    if synth_thm:
        rows = _group_rows(labels.tickers_in_theme(synth_thm))
        basket = {k: (round(float(np.mean([r[k] for r in rows if r[k] is not None])), 1) if rows else None) for k in ("d1", "w1", "m1")}
        etf_sym = f"SYNTH_THM_{synth_thm.replace(' ', '').replace('&', 'And').replace('-', '')}".upper()
        return jsonify({"theme": f"{synth_thm} (Synthetic)", "etf": etf_sym, "perf": basket, "members": rows, "n_total": len(rows)})

    industry = request.args.get("industry")
    if industry:
        rows = _group_rows([s for s in FRAMES if labels.sector(s) == industry])
        basket = {k: (round(float(np.mean([r[k] for r in rows if r[k] is not None])), 1)
                      if rows else None) for k in ("d1", "w1", "m1")}
        return jsonify({"theme": industry, "etf": labels.spdr_of(industry),
                        "perf": basket, "members": rows[:30], "n_total": len(rows)})
    if theme:
        rows = _group_rows(labels.tickers_in_theme(theme))
        basket = {k: (round(float(np.mean([r[k] for r in rows if r[k] is not None])), 1)
                      if rows else None) for k in ("d1", "w1", "m1")}
        return jsonify({"theme": theme, "etf": labels.theme_etf(theme),
                        "perf": basket, "members": rows})
    sym = (request.args.get("symbol") or "").upper()
    d = FRAMES.get(sym)
    if d is None or not len(d):
        return jsonify({"error": f"no data for {sym}"})
    r = d.iloc[-1]

    def ret(nb):
        return (round(float(d["close"].iloc[-1] / d["close"].iloc[-1 - nb] - 1) * 100, 1)
                if len(d) > nb else None)

    perf = {"d1": ret(1), "w1": ret(5), "m1": ret(21), "m3": ret(63), "m6": ret(126),
            "off_ath_pct": (round(float(r["off_ath_pct"]), 1) if pd.notna(r.get("off_ath_pct")) else None),
            "rvol_today": round(float(r["rvol"]), 2) if pd.notna(r.get("rvol")) else None,
            "adr_pct": round(float(r["adr_pct"]), 2) if pd.notna(r.get("adr_pct")) else None,
            "dollar_vol_m": round(float(r["dollar_vol"]) / 1e6, 1) if pd.notna(r.get("dollar_vol")) else None,
            "rsi": round(float(r["rsi14"]), 1) if pd.notna(r.get("rsi14")) else None,
            "close": round(float(r["close"]), 2)}
    try:
        seq = sequence.sequence_read(d)
        srl = sr.sr_levels(d)
        aware = {"regime": seq.get("regime"), "phrase": seq.get("phrase"),
                 "flags": seq.get("flags", []),
                 "resistance": (srl.get("resistance") or [])[:2],
                 "support": (srl.get("support") or [])[:2]}
    except Exception as e:
        aware = {"error": str(e)}
    intra = None
    f15 = _fresh_frames5()
    if f15 and f15.get(sym) is not None:
        st = intraday._session_stats(f15[sym])
        if st:
            intra = {"day_pct": round(st["d"], 2), "vwap_side": "above" if st["above_vwap"] else "below"}
    today = dt.date.today().isoformat()
    heads = ep_news.news_for(sym, today, days_back=4, limit=8)
    news = {"catalyst": ep_news.classify(heads, sym=sym, name=_universe_names().get(sym))["catalyst"], "items": heads[:6],
            "edgar": ep_news.edgar_8k(sym, today, days_back=7),
            "earnings": ep_news.earnings_near(sym, today, days=10)}

    def group_perf(members):
        """Equal-weight basket returns for the symbol's sector/theme peers."""
        out = {}
        for nb, key in ((1, "d1"), (5, "w1"), (21, "m1")):
            vals = []
            for s in members:
                f = FRAMES.get(s)
                if f is not None and len(f) > nb:
                    vals.append(float(f["close"].iloc[-1] / f["close"].iloc[-1 - nb] - 1) * 100)
            out[key] = round(float(np.mean(vals)), 1) if len(vals) >= 3 else None
        out["n"] = len(members)
        return out

    sec = labels.sector(sym)
    sec_perf = None
    if sec:
        peers = [s for s in FRAMES if labels.sector(s) == sec]
        sec_perf = {"name": sec, **group_perf(peers)}
    thm_perf = [{"name": t, **group_perf(list(labels.tickers_in_theme(t)))}
                for t in (labels.themes(sym) or [])]
    def top_leader(members):
        """Strongest peer by 1-week return (the ticker itself excluded)."""
        best, br = None, None
        for s in members:
            if s == sym:
                continue
            f = FRAMES.get(s)
            if f is None or len(f) < 7:
                continue
            r1w = float(f["close"].iloc[-1] / f["close"].iloc[-6] - 1)
            if br is None or r1w > br:
                best, br = s, r1w
        return best

    dens = [{"label": "SPY", "sym": "SPY"}]            # ratio-chart denominators
    spdr = labels.spdr_of(sec) if sec else None
    if spdr and spdr in FRAMES:
        dens.append({"label": f"sector ETF ({spdr})", "sym": spdr})
    for t in (labels.themes(sym) or []):
        te = labels.theme_etf(t)
        if te and te in FRAMES and all(x["sym"] != te for x in dens):
            dens.append({"label": f"{t} ({te})", "sym": te})
    # vs the TOP LEADER (1-week) of each group: industry peers, SPDR sector, every theme
    if sec:
        ld = top_leader([s for s in FRAMES if labels.sector(s) == sec])
        if ld and all(x["sym"] != ld for x in dens):
            dens.append({"label": f"industry leader ({ld})", "sym": ld})
    if spdr:
        ld = top_leader(labels.spdr_members(spdr, list(FRAMES)))
        if ld and all(x["sym"] != ld for x in dens):
            dens.append({"label": f"sector leader ({ld})", "sym": ld})
    for t in (labels.themes(sym) or []):
        ld = top_leader(list(labels.tickers_in_theme(t)))
        if ld and all(x["sym"] != ld for x in dens):
            dens.append({"label": f"{t} leader ({ld})", "sym": ld})
    for r in dens:                                     # current STRUCTURE + extension of each ratio
        ck = (sym, r["sym"], LATEST)
        if ck not in _RATIO_STRUCT_CACHE:
            reg = phr = ext = None
            try:
                fr = _ratio_tf_frame(sym, r["sym"], "1D", LATEST, bars=420)
                if fr is not None and len(fr) >= 60:
                    seq = sequence.sequence_read(fr)
                    reg, phr = seq.get("regime"), seq.get("phrase")
                    ext = _atr_ext50(fr)               # ratio units — badge uses the etf tiers
            except Exception:
                pass
            if len(_RATIO_STRUCT_CACHE) > 4000:
                _RATIO_STRUCT_CACHE.clear()
            _RATIO_STRUCT_CACHE[ck] = (reg, phr, ext)
        r["regime"], r["phrase"], r["ext"] = _RATIO_STRUCT_CACHE[ck]
    return jsonify({"symbol": sym, "sector": sec, "themes": labels.themes(sym),
                    "sector_perf": sec_perf, "theme_perf": thm_perf, "ratio_dens": dens,
                    "perf": perf, "aware": aware, "intraday": intra, "news": news,
                    "ext_tfs": _ext_tfs(sym, LATEST), "kind": _ext_kind(sym),
                    "rsix": _rsix_full(sym), "rsix_compact": _rsix_row(sym, LATEST)})


def _setup_context(hits_date: str, warm_only: bool = False) -> "tuple[dict, dict, dict]":
    """(setmap, floor_map, trig_map) from ONE as-of scan pass.
    setmap:   sym -> ['setup/state', ...] for names in a building/breakout/EP setup — the actionable
              context used to flag RVOL / EP-radar rows (a spike on a name that is ALSO setting up).
    floor_map: sym -> lowered live $vol floor (M$) from config.SETUP_MIN_DVOL_M — a name in a
              thin-friendly EP/gapper setup qualifies at a thinner floor than the generic universe.
    trig_map: sym -> {setup, state, trigger, level} — the BEST setup row per name (EP/gapper breakout
              first) so the significance grader can say 'at/broke trigger X' (plan-ep-significance).
    warm_only (replay): annotate only if that date's scan is already cached — never trigger a fresh
    ~60s as-of scan on an interactive replay read (returns empty maps when cold)."""
    setmap: dict = {}
    floor_map: dict = {}
    trig_map: dict = {}

    def _rank(setup, st):                        # higher = better significance context
        prime = setup in ("episodic_pivot", "gapper")
        return (2 if prime else 1) * 10 + (5 if st == "breakout" else 3 if st in ("building", "watch") else 0)
    try:
        with _SCAN_LOCK:
            hits = _scan_cached(hits_date, warm_only=warm_only)
    except Exception:
        return setmap, floor_map, trig_map
    if hits is not None and len(hits) and "symbol" in hits.columns:
        for _, h in hits.iterrows():
            sym = h["symbol"]; setup = str(h.get("setup", "")); st = str(h.get("state", ""))
            if st in ("building", "breakout") or setup in ("episodic_pivot", "gapper"):
                setmap.setdefault(sym, []).append(f"{setup}/{st or '—'}")
            cur = trig_map.get(sym)
            if cur is None or _rank(setup, st) > _rank(cur["setup"], cur["state"]):
                try:
                    trig = float(h["trigger"]) if pd.notna(h.get("trigger")) else None
                except Exception:
                    trig = None
                trig_map[sym] = {"setup": setup, "state": st, "trigger": trig,
                                 "level": (float(h["level"]) if pd.notna(h.get("level")) else None)}
            f = config.SETUP_MIN_DVOL_M.get(setup)
            if f is not None:
                floor_map[sym] = min(floor_map.get(sym, f), f)
    return setmap, floor_map, trig_map


_UNIV_NAMES: dict = {}
def _universe_names() -> dict:
    """Cached {yf_symbol -> company name} for EP-radar OWN-vs-PEER news attribution."""
    global _UNIV_NAMES
    if not _UNIV_NAMES and UNIVERSE is not None and "name" in UNIVERSE.columns:
        try:
            _UNIV_NAMES = {s: n for s, n in UNIVERSE["name"].items() if isinstance(n, str)}
        except Exception:
            _UNIV_NAMES = {}
    return _UNIV_NAMES


def _grade_rows(rows: list, phase: str, trig_map: dict, frames5: dict | None = None,
                ext_date: str | None = None, replay: bool = False) -> None:
    """Attach the significance read (plan-ep-significance-alerts) to radar/rvol rows in place:
    row['sig'] = {score, grade, action, why, direction}. Composes what's already on the row
    (volume conviction, catalyst, reaction tell) with the scan's setup context (trig_map ->
    at/broke trigger), the live last price (frames5), and the extension RED-tier cap (_ext_tfs).
    UNVALIDATED HEURISTIC — labeled so in the UI until validate_significance.py passes."""
    import significance
    for r in rows:
        sym = r.get("symbol")
        if not sym:
            continue
        ctx = dict(trig_map.get(sym) or {})
        try:                                       # live last price -> trigger-cross detection + ledger entry
            f = (frames5 or {}).get(sym)
            if f is not None and len(f):
                ctx["last"] = float(f["close"].iloc[-1])
                r["last_px"] = ctx["last"]
        except Exception:
            pass
        extended = None
        try:                                       # worst-TF ATR-ext at the RED badge tier caps at B
            tfs = _ext_tfs(sym, ext_date or _live_day()) or {}
            red = config.SIG_EXT_RED.get(_ext_kind(sym), 10.0)
            extended = any(v is not None and v >= red for v in tfs.values()) or None
        except Exception:
            pass
        r["sig"] = significance.grade(r, ctx or None, phase=phase, extended=extended)
        if replay:                                 # replay never carries news -> the 25-pt catalyst
            r["sig"]["news_na"] = True             # component is absent; the chip renders it dimmed


@app.get("/api/ep_radar")
def api_ep_radar():
    """EP radar, REPLAY-aware. Replay (a settled date + clock `t`): EP events as they had formed by
    `t`, read point-in-time from the Tiingo 5m store (base_min=5, daily context anchored to the prior
    session). Live: pre-market (08:00-09:30 ET) overnight gap + calibrated pre-market $vol (Bonde's
    early entry), else RTH forming EPs (15m). News/catalyst is attached in the live RTH path only
    (news is a live API — skipped on replay). Every row carries the significance read (row.sig)."""
    import ep_news
    date = request.args.get("date") or LATEST
    t = request.args.get("t")
    replay = bool(t and t != "live") and date <= LATEST
    phase = "replay" if replay else tiingo_live.session_phase()
    try:
        if replay:
            asof = market._prior_session(date, FRAMES)
            f5 = _asof_frames5(date, t) or {}
            rows = (ep_news.live_radar(FRAMES, f5, min_dvol_m=config.LIVE_EP_MIN_DVOL_M, base_min=5, as_of=asof)
                    if f5 else [])
            setmap, _, trig = _setup_context(asof or date, warm_only=True)  # replay: annotate only if scan already warm
            for r in rows:
                r["setups"] = " · ".join(setmap.get(r["symbol"], [])[:3])
            _grade_rows(rows, "rth", trig, f5, ext_date=asof or date, replay=True)
        elif phase == "pre":
            rows = ep_news.open_radar(FRAMES)
            today = dt.date.today().isoformat()
            setmap, _, trig = _setup_context(LATEST)   # flag gappers that are ALSO in a building/EP setup
            for r in rows:
                r["setups"] = " · ".join(setmap.get(r["symbol"], [])[:3])
            for r in rows[:15]:
                r.update(ep_news.classify(ep_news.news_for(r["symbol"], today),
                                          sym=r["symbol"], name=_universe_names().get(r["symbol"])))
            _grade_rows(rows, "pre", trig)
            import alerts
            import sig_ledger
            sig_ledger.record_events(alerts.radar_events(rows, "pre"))   # E5 pre-market gapper alerts
            sig_ledger.record_rows(rows, "ep_pre")
        else:
            f5 = _fresh_frames5() or {}
            rows = ep_news.radar_with_news(FRAMES, f5, session=market._today_et(), min_dvol_m=config.LIVE_EP_MIN_DVOL_M,
                                           base_min=5, names=_universe_names(), dump=True)
            setmap, _, trig = _setup_context(LATEST)   # setups context was MISSING on the RTH path
            for r in rows:
                r["setups"] = " · ".join(setmap.get(r["symbol"], [])[:3])
            _grade_rows(rows, "rth", trig, f5)
            import alerts
            import sig_ledger
            sig_ledger.record_events(alerts.radar_events(rows, "rth"))  # E1/E3/E4/E6
            sig_ledger.record_rows(rows, "ep")         # the LIVING LEDGER: grades + outcomes accrue daily
    except Exception as e:
        return jsonify({"error": f"ep radar error: {e}", "phase": phase, "rows": []})
    warn = None
    if not replay and tiingo_live.STATE.get("feed_degraded"):     # feed-freeze (see api_rvol_leaders)
        fc = int(tiingo_live.STATE.get("vol_frozen_cycles") or 0)
        warn = f"⚠ live volume feed warming — no volume growth for {fc} polls; EP volume reads pending"
    return jsonify({"phase": phase or "closed", "date": date, "t": t, "rows": rows[:40],
                    **({"warning": warn, "note": warn} if warn else {})})


@app.get("/api/rvol_leaders")
def api_rvol_leaders():
    """Whole-universe INTRADAY RVOL leaders: cumulative-pace RVOL (calibrated) + a best-effort
    slot-spike flag, ranked by pace, each cross-referenced with the session's setup state so a spike
    on a 'building' flat_base/QM/HTF/EP name is highlighted (the high-value alert). REPLAY (a settled
    date + clock `t`): reads the 5m store as-of `t` (base_min=5, point-in-time). LIVE: fresh 15m +
    poller day-volume; returns [] off-session. In-app spike alerts fire only in the live RTH path."""
    import ep_news
    date = request.args.get("date") or LATEST
    t = request.args.get("t")
    replay = bool(t and t != "live") and date <= LATEST
    phase = "replay" if replay else tiingo_live.session_phase()
    try:
        if replay:
            asof = market._prior_session(date, FRAMES)
            f5 = _asof_frames5(date, t) or {}
            if not f5:
                return jsonify({"phase": phase, "date": date, "t": t, "rows": [],
                                "note": f"no 5m data as of {date} {t}"})
            setmap, _, trig = _setup_context(asof or date, warm_only=True)  # replay: annotate only if warm; blanket 1M floor
            rows = ep_news.rvol_leaders(FRAMES, f5, min_dvol_m=1.0, min_pace=1.3, limit=60,
                                        base_min=5, as_of=asof)
        else:
            if phase != "rth":                    # cum-pace RVOL is a LIVE intraday metric — off-session
                return jsonify({"phase": phase or "closed", "date": date, "t": t, "rows": [],
                                "note": "market closed — RVOL leaders runs live during RTH "
                                        "(drag the time-slider to replay a past session)"})
            f5 = _fresh_frames5() or {}
            if not f5:
                return jsonify({"phase": phase or "closed", "rows": [], "note": "no fresh intraday data"})
            setmap, floor_map, trig = _setup_context(LATEST)           # warm live -> per-setup $vol floors
            rows = ep_news.rvol_leaders(FRAMES, f5, session=market._today_et(), min_dvol_m=config.LIVE_RVOL_MIN_DVOL_M,
                                        min_dvol_map=floor_map, min_pace=1.3, limit=60, base_min=5)
        for r in rows:
            r["setups"] = " · ".join(setmap.get(r["symbol"], [])[:3])
        _grade_rows(rows, "rth", trig, f5, ext_date=(asof or date) if replay else None, replay=replay)
        if phase == "rth":                       # live RTH only -> spike + significance event alerts
            import alerts
            import sig_ledger
            sig_ledger.record_events(alerts.rvol_alerts(rows))
            sig_ledger.record_events(alerts.radar_events(rows, "rth"))
            sig_ledger.record_rows(rows, "rvol")
    except Exception as e:
        return jsonify({"error": f"rvol leaders error: {e}", "phase": phase, "rows": []})
    # BOARD-SHAPE WATCHDOG (open-hardening plan C, Amir 2026-07-09): a board full of >100x
    # cum-RVOL names is a data-feed defect, not a market condition (the 09:38 flood put 60
    # names at ~1000x). Warn visibly instead of confidently serving a wrong leaderboard.
    warn = None
    if not replay:
        ext = sum(1 for r in rows if (r.get("cum_rvol") or 0) > 100)
        if ext >= 20:
            warn = (f"⚠ live volume feed suspect — {ext} names read >100× cum RVOL; "
                    "treat this board as unreliable until it clears")
            print(f"[watchdog] {dt.datetime.now():%H:%M} {warn}")
        elif tiingo_live.STATE.get("feed_degraded"):
            # FEED-FREEZE (2026-07-10 open): the trade-data feed's volume stopped advancing, so DAYVOL
            # can't accumulate and this board goes EMPTY — say so, else a blank open reads as "quiet".
            fc = int(tiingo_live.STATE.get("vol_frozen_cycles") or 0)
            warn = (f"⚠ live volume feed warming — no volume growth for {fc} polls; "
                    "RVOL fills in once the feed clears")
    return jsonify({"phase": phase or "closed", "date": date, "t": t, "rows": rows,
                    **({"warning": warn, "note": warn} if warn else {})})


@app.get("/api/download_intraday/status")
def api_download_intraday_status():
    return jsonify({**DOWNLOAD, "retry_in": _intraday_cooldown_left(), "n_tickers": len(FRAMES)})


@app.post("/api/watchlists")
def api_watchlists():
    """Write the current LATEST scan's TradingView watchlist .txt files to output/."""
    try:
        with _SCAN_LOCK:
            hits = _scan_cached(LATEST)
        scan.save_outputs(hits)
    except Exception as e:
        return jsonify({"error": str(e)})
    return jsonify({"ok": True, "n": int(len(hits))})


_FOLD_STATE = {"day": None}            # last date the live 5m overlay was folded into the main store
_EOD_UPDATE = {"day": None}            # last date the close-triggered daily update fired
_PREP_STATE = {"day": None}            # last date the pre-open day-type prep ran (once/weekday)


def _session_bars_complete() -> bool:
    """All of today's session bars are in the live 5m store: SPY (the most reliable IEX name) has the
    final 15:55 RTH slot. Gates the close-triggered fold + daily update — 'as soon as we have all
    bars' (Amir 2026-07-08), not a fixed clock margin."""
    f = FRAMES5.get("SPY")
    if f is None or not len(f):
        return False
    last = f.index[-1]
    today = pd.Timestamp(market._today_et())
    return last.normalize() == today and (last.hour, last.minute) >= (15, 55)


def _scheduler() -> None:
    """In-app auto-update: fire once per day at config.AUTO_UPDATE_TIME (local HH:MM)
    when the toggle is on. Cheap poll every 30s; guarded so it runs at most once/day."""
    last_day = None
    while True:
        time.sleep(30)
        cur = settings.current()
        # EOD, right after the close once the session's bars are all in (>=16:02 ET + SPY has the
        # 15:55 slot; 16:15 hard fallback if the completeness probe never passes): (1) fold today's
        # live 5m overlay into the main store, (2) fire the DAILY update so the banner/health flip to
        # TODAY's close within minutes of the bell instead of waiting for AUTO_UPDATE_TIME (which
        # stays on as the overnight self-healing re-pull). Amir 2026-07-08.
        try:
            from zoneinfo import ZoneInfo
            et = dt.datetime.now(ZoneInfo("America/New_York"))
            _ready = et.time() >= dt.time(16, 15) or (et.time() >= dt.time(16, 2) and _session_bars_complete())
            if et.weekday() < 5 and _ready and _FOLD_STATE["day"] != et.date().isoformat():
                _FOLD_STATE["day"] = et.date().isoformat()
                nf = datastore.fold_5m_live()
                if nf:
                    print(f"[5m] {dt.datetime.now():%H:%M} folded {nf} live-overlay symbols into the main store")
            if (et.weekday() < 5 and _ready and cur.get("AUTO_UPDATE")
                    and _EOD_UPDATE["day"] != et.date().isoformat()):
                if start_update():                     # only stamp on success (an in-flight update retries next tick)
                    _EOD_UPDATE["day"] = et.date().isoformat()
                    print(f"[auto-update] {dt.datetime.now():%Y-%m-%d %H:%M} close-triggered daily update")
        except Exception:
            pass
        # PRE-OPEN DAY-TYPE PREP: compute TODAY's bucket-A (pre-open) day-type features once per weekday
        # at ~DAYTYPE_PREP_TIME_ET so the live read is ready at the bell — was a manual daytype_live.py
        # --prep step (BUGS 2026-07-09). ET-triggered (the scheduler's own clock is local). Best-effort.
        try:
            from zoneinfo import ZoneInfo
            et = dt.datetime.now(ZoneInfo("America/New_York"))
            if market.daily_time_due(et, str(cur.get("DAYTYPE_PREP_TIME_ET", config.DAYTYPE_PREP_TIME_ET)),
                                     _PREP_STATE["day"]):
                _PREP_STATE["day"] = et.date().isoformat()

                def _bg_prep():
                    try:
                        import daytype_live
                        daytype_live.prep()
                        print(f"[daytype] {dt.datetime.now():%H:%M} pre-open --prep done", flush=True)
                    except Exception as e:
                        print(f"[daytype] pre-open prep failed: {e}", flush=True)
                threading.Thread(target=_bg_prep, daemon=True).start()
        except Exception:
            pass
        # FUTURES 5m auto-refresh during FUTURES hours (~24x5, not just equity RTH) — CHEAP (one batched
        # yfinance call), FREQUENT. Own cadence timer (last_fut_auto), decoupled from the equity gap-fill so
        # it runs at the 5m bar cadence (feed delay ~8min) without perpetually debouncing the manual button.
        # Gated on _futures_open_now (was _is_partial = equity RTH) so the 24h futures store stays current
        # OVERNIGHT + pre-market instead of freezing at the prior 16:00 pull.
        if cur.get("INTRADAY_AUTO_REFRESH") and _futures_open_now() and not DOWNLOAD["running"]:
            every = int(cur.get("INTRADAY_REFRESH_EVERY_MIN", 5)) * 60
            lastf = DOWNLOAD.get("last_fut_auto")
            if lastf is None or time.time() - lastf >= every:
                if start_download_intraday(futures_only=True):
                    DOWNLOAD["last_fut_auto"] = time.time()
                    print(f"[intraday] {dt.datetime.now():%H:%M} auto-refreshing futures 5m")
        # EQUITY today-session gap-fill: SLOW safety net (the poller is the live source) — the expensive
        # ~universe Tiingo pull, kept OFF the 5m futures cadence. Runs in its own thread + guard.
        if (cur.get("INTRADAY_AUTO_REFRESH") and market._is_partial(None)
                and not _SESSION_SEED["running"]):
            gevery = config.INTRADAY_EQUITY_GAPFILL_MIN * 60
            glast = _SESSION_SEED.get("last_auto")
            if glast is None or time.time() - glast >= gevery:
                _SESSION_SEED["last_auto"] = time.time()

                def _bg_seed():
                    try:
                        m = _seed_today_session(force=True)
                        if m:
                            with _SCAN_LOCK:
                                _MARKET_CACHE.clear(); _GROUPS_CACHE.clear(); _AWARE_CACHE.clear(); _PLAYBOOK_CACHE.clear()
                    except Exception:
                        pass
                threading.Thread(target=_bg_seed, daemon=True).start()
        if not cur.get("AUTO_UPDATE"):
            continue
        now = dt.datetime.now()
        today = now.date().isoformat()
        if now.strftime("%H:%M") == str(cur.get("AUTO_UPDATE_TIME", "")) and last_day != today:
            last_day = today
            if start_update():
                print(f"[auto-update] {now:%Y-%m-%d %H:%M} triggered daily update")


if __name__ == "__main__":
    load_frames()
    if market.no_restart_window():                        # plan E: loud operator warning on a window restart
        print("⚠ [live-ops] server (re)started INSIDE the 09:15-10:00 ET no-restart window — the board "
              "will be WARMING for a few minutes: indices + top movers seed first, the illiquid tail "
              "follows. Standing rule: avoid restarts here unless the board is actively broken.", flush=True)
    threading.Thread(target=precompute, args=(_live_day(),), daemon=True).start()  # warm the live day in bg
    if market._is_partial(None):                          # during RTH: start the live 5m pipeline NOW
        threading.Thread(target=_ensure_live_poller, daemon=True).start()  # (poller + today-session seed)
    threading.Thread(target=_scheduler, daemon=True).start()
    _seed_notif_log()                      # restore today's grade-A alerts after a restart
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8780)),
            debug=False, threaded=True)
