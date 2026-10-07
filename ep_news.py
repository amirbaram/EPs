"""EP intelligence — news attachment, catalyst classification, and the live intraday EP radar.

Three jobs (Amir's AVAV observation: a $500M-contract EP formed at noon, invisible to the
EOD-daily detector and unexplained without news):

  news_for(sym, date)      — Tiingo news headlines for the symbol around the date (cached to
                             data/news_cache/, so the historical learning loop doesn't re-fetch).
  classify(headlines)      — rule-based catalyst taxonomy: earnings / guidance / contract /
                             fda / m&a / offering / analyst / crypto / sympathy / unknown.
                             Transparent first pass; an LLM judgment layer can sit on top later.
  live_radar(frames, frames15) — intraday EP events forming NOW: today's move vs prior close +
                             gap + cumulative volume vs the time-of-day expected pace (the same
                             U-curve the breakout projection uses). Runs off the 15m store.

The learning loop lives in the validator: historical episodic_pivot events get news_for() +
classify() tags, and `catalyst` becomes a --split dimension — "which news is significant"
becomes a measured table (hit rate / payoff per catalyst type) instead of intuition.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

import config

NEWS_DIR = config.DATA_DIR / "news_cache"
_KEY_FILE = config.DATA_DIR / "tiingo_key.txt"

CATALYSTS = [   # first matching class wins; order = specificity
    ("offering", r"offering|dilut|secondary|shelf|registered direct|convertible note|atm program"),
    ("earnings", r"earnings|quarterly results|q[1-4] (results|revenue)|beats|misses|eps of"),
    ("guidance", r"guidance|outlook|forecast|raises full.year|revenue outlook"),
    ("contract", r"contract|award|order[s]? (worth|valued)|procurement|deal with|partnership"),
    ("fda", r"fda|phase (1|2|3|i{1,3})|trial (data|results)|approval|clearance|breakthrough"),
    ("mna", r"acquir|merger|takeover|buyout|to buy|stake in"),
    ("analyst", r"upgrade|downgrade|initiat|price target|overweight|underweight"),
    ("crypto", r"bitcoin|crypto|ethereum|token|mining"),
]


def _key() -> str | None:
    try:
        return _KEY_FILE.read_text().strip()
    except OSError:
        return None


def news_for(sym: str, date: str, days_back: int = 1, limit: int = 12) -> list[dict]:
    """Headlines for `sym` in [date - days_back, date + 1), cached per (sym, date). Each item:
    {title, source, published}. Empty list when no key / no coverage.

    Tiingo's news archive on this plan only reaches back ~3 months; for older windows the API
    silently returns its OLDEST stories instead of respecting startDate — so results are
    post-filtered to the requested window (out-of-window stories are wrong news, not old news)."""
    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    cache = NEWS_DIR / f"{sym}_{date}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except ValueError:
            pass
    key = _key()
    if not key:
        return []
    start = (pd.Timestamp(date) - pd.Timedelta(days=days_back)).date().isoformat()
    end = (pd.Timestamp(date) + pd.Timedelta(days=1)).date().isoformat()
    url = ("https://api.tiingo.com/tiingo/news?" + urllib.parse.urlencode(
        {"tickers": sym.lower(), "startDate": start, "endDate": end,
         "limit": limit, "token": key}))
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            raw = json.load(r)
        items = [{"title": x.get("title", ""), "source": x.get("source", ""),
                  "published": x.get("publishedDate", "")[:16]} for x in raw
                 if start <= x.get("publishedDate", "")[:10] <= end]
    except Exception:
        return []                                     # transient failure: don't cache
    cache.write_text(json.dumps(items))
    time.sleep(0.15)                                  # gentle pacing for backfill loops
    return items


def archive_floor() -> str:
    """Earliest date the news archive can answer for (~3 months back on this plan)."""
    return (pd.Timestamp.now() - pd.Timedelta(days=88)).date().isoformat()


# ------------------------------------------------- SEC EDGAR 8-K + earnings dates (free, full history)
# Tiingo headlines only reach ~3 months back; EDGAR 8-K filings and the earnings calendar cover the
# deep history for the EP learning loop: "was there a material event / earnings at this EP?"

EDGAR_DIR = config.DATA_DIR / "edgar_cache"
_UA = {"User-Agent": "scan-research amir.baram@gmail.com"}
_CIK: dict | None = None

ITEM_LABELS = {  # the 8-K items that matter for catalyst typing
    "2.02": "earnings results", "1.01": "material agreement", "7.01": "reg FD",
    "8.01": "other event", "5.02": "officer change", "2.01": "acquisition done",
    "1.02": "agreement terminated", "3.01": "listing notice", "4.02": "non-reliance/restatement",
}


def _edgar_get(url: str):
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def _cik_for(sym: str) -> str | None:
    global _CIK
    if _CIK is None:
        EDGAR_DIR.mkdir(parents=True, exist_ok=True)
        f = EDGAR_DIR / "company_tickers.json"
        if f.exists():
            _CIK = json.loads(f.read_text())
        else:
            _CIK = _edgar_get("https://www.sec.gov/files/company_tickers.json")
            f.write_text(json.dumps(_CIK))
    for v in _CIK.values():
        if v["ticker"].upper() == sym.upper():
            return f"{int(v['cik_str']):010d}"
    return None


def edgar_8k(sym: str, date: str, days_back: int = 3) -> list[dict]:
    """8-K filings for `sym` in [date - days_back, date]: [{date, items, labels}]. The full recent
    filing list is cached per symbol (one submissions call covers ~years of filings)."""
    EDGAR_DIR.mkdir(parents=True, exist_ok=True)
    cache = EDGAR_DIR / f"{sym}_8k.json"
    if cache.exists():
        try:
            allf = json.loads(cache.read_text())
        except ValueError:
            allf = None
    else:
        allf = None
    if allf is None:
        cik = _cik_for(sym)
        if not cik:
            return []
        try:
            j = _edgar_get(f"https://data.sec.gov/submissions/CIK{cik}.json")
            rec = j["filings"]["recent"]
            allf = [{"date": rec["filingDate"][i], "items": rec.get("items", [""] * 9999)[i]}
                    for i in range(len(rec["form"])) if rec["form"][i].startswith("8-K")]
        except Exception:
            return []
        cache.write_text(json.dumps(allf))
        time.sleep(0.12)                              # SEC fair-use pacing
    start = (pd.Timestamp(date) - pd.Timedelta(days=days_back)).date().isoformat()
    out = []
    for f in allf:
        if start <= f["date"] <= date:
            labs = [ITEM_LABELS.get(x.strip(), x.strip())
                    for x in (f.get("items") or "").split(",") if x.strip()]
            out.append({"date": f["date"], "items": f.get("items", ""), "labels": labs})
    return out


def earnings_near(sym: str, date: str, days: int = 7) -> str | None:
    """Nearest earnings date in [date - days, date + 1] (yfinance calendar, cached) — asymmetric
    because the question is 'was this EP earnings-driven', i.e. earnings on/just before the move."""
    EDGAR_DIR.mkdir(parents=True, exist_ok=True)
    cache = EDGAR_DIR / f"{sym}_earnings.json"
    if cache.exists():
        try:
            dates = json.loads(cache.read_text())
        except ValueError:
            dates = None
    else:
        dates = None
    if dates is None:
        try:
            import yfinance as yf
            ed = yf.Ticker(sym).get_earnings_dates(limit=100)   # Yahoo caps at 100; oldest-first
            dates = sorted({str(x.date()) for x in ed.index}) if ed is not None and len(ed) else []
        except Exception:
            return None
        cache.write_text(json.dumps(dates))
    t = pd.Timestamp(date)
    best = None
    for ds in dates:
        dd = (t - pd.Timestamp(ds)).days              # positive = earnings BEFORE the hit
        if -1 <= dd <= days and (best is None or abs(dd) < best[0]):
            best = (abs(dd), ds)
    return best[1] if best else None


def classify(headlines: list[dict], sym: str | None = None, name: str | None = None) -> dict:
    """Rule-based catalyst tag over the headline set: {catalyst, headline, n_stories}.
    'sympathy' when stories exist but none match a class; 'unknown' when no stories.
    When `sym`/`name` are given, PREFER the symbol's OWN news (headlines mentioning the ticker/company) over the
    generic market headlines Tiingo mixes in — else a keyword in an UNRELATED story wins the tag (BUGS #31:
    CRNX +99% on drug data showed a 'Samsung Misses…' headline via 'misses'→earnings). Falls back to all
    headlines if none mention the symbol. Backward-compatible: no sym/name → old whole-set behavior."""
    if not headlines:
        return {"catalyst": "unknown", "headline": None, "n_stories": 0}
    pool = [h for h in headlines if _mentions(h["title"], sym, name)] if sym else []
    pool = pool or headlines                                # own news preferred; else the whole set
    text = " ".join(h["title"].lower() for h in pool)
    for cname, pat in CATALYSTS:
        if re.search(pat, text):
            hit = next((h["title"] for h in pool if re.search(pat, h["title"].lower())), pool[0]["title"])
            return {"catalyst": cname, "headline": hit[:140], "n_stories": len(headlines)}
    return {"catalyst": "sympathy", "headline": pool[0]["title"][:140], "n_stories": len(headlines)}


# ---- news SENTIMENT + reaction-divergence (rule-based; tune the lexicons over time) ----------
_SENT_POS = re.compile(r"\b(beat|beats|tops|topped|raise|raises|raised|record|surge|surges|soar|"
                       r"soars|jump|jumps|win|wins|won|award|awarded|approv\w*|breakthrough|upgrade|"
                       r"upgraded|outperform|overweight|expand\w*|grant\w*|clearance|launch\w*|"
                       r"strong|positive|rally|rallies|acquire|acquires|buyout|stake in)\b")
_SENT_NEG = re.compile(r"\b(miss|misses|missed|cut|cuts|lower|lowers|lowered|halt|halts|halted|"
                       r"probe|investigat\w*|downgrade|downgraded|dilut\w*|offering|secondary|shelf|"
                       r"convertible note|lawsuit|sues|sued|recall|warn|warns|warning|delay|delays|"
                       r"delayed|fail|fails|failed|plunge|plunges|slump|weak|underweight|bankrupt\w*|"
                       r"default|resign\w*|subpoena|fraud|short seller|guidance cut)\b")


def sentiment(headlines: list[dict]) -> dict:
    """Rule-based good/bad read over a headline set: {sent, pos, neg, conflicting}. sent in
    good/bad/mixed/neutral; conflicting = both good AND bad language present."""
    pos = sum(1 for h in headlines if _SENT_POS.search(h["title"].lower()))
    neg = sum(1 for h in headlines if _SENT_NEG.search(h["title"].lower()))
    net = pos - neg
    # conflicting only when the MINORITY side is material (>= ~1/3 of the majority) — a lone stray
    # (e.g. a law-firm "investigates the merger" note on a clean takeover) shouldn't flag conflict
    conflicting = bool(pos and neg and min(pos, neg) >= max(1, round(0.34 * max(pos, neg))))
    sent = "mixed" if conflicting else ("good" if net > 0 else ("bad" if net < 0 else "neutral"))
    return {"sent": sent, "pos": pos, "neg": neg, "net": net, "conflicting": conflicting}


def _mentions(title: str, sym: str, name: str | None) -> bool:
    """Does this headline mention THIS ticker / company (its OWN news) vs a theme peer / generic?"""
    tl = title.lower()
    if re.search(rf"(^|[^a-z]){re.escape(sym.lower())}([^a-z]|$)", tl):
        return True
    return bool(name and name.split(",")[0].split()[0].lower() in tl and len(name.split()[0]) > 3)


def news_signal(sym: str, headlines: list[dict], move_pct: float | None = None,
                name: str | None = None) -> dict:
    """Enrich classify() with sentiment + OWN-vs-PEER attribution + a TODAY-ONLY reaction-divergence
    warning: GOOD news + red reaction = bearish tell (big); BAD news + green = bullish tell; conflicting
    headlines are noted. (The 'good-news-no-longer-pumps' history pattern is a deferred lab.)"""
    own_hs = [h for h in headlines if _mentions(h["title"], sym, name)]
    base = classify(own_hs or headlines)                    # catalyst + its headline, from the symbol's OWN news
    s = sentiment(own_hs or headlines)
    peer = next((h["title"] for h in headlines if h not in own_hs), None)
    react, warn = "aligned", ""
    lean = "good" if s["net"] > 0 else ("bad" if s["net"] < 0 else None)   # NET sentiment drives the reaction
    if move_pct is not None and abs(move_pct) >= 1.0 and lean:
        up = move_pct > 0
        if lean == "good" and not up:
            react, warn = "good_news_red", "⚠ GOOD news but price RED — market rejecting the catalyst (bearish tell)"
        elif lean == "bad" and up:
            react, warn = "bad_news_green", "↑ BAD news but price GREEN — resilience (bullish tell)"
    if s["conflicting"]:
        warn = (warn + " · " if warn else "") + "conflicting headlines (good & bad)"
    base.update({"sent": s["sent"], "conflicting": s["conflicting"], "news_react": react,
                 "news_warn": warn, "own_headline": (base.get("headline") if own_hs else None),
                 "peer_headline": (peer[:140] if peer else None)})
    return base                                             # base["headline"] = the OWN catalyst story (classify)


def dump_news_analysis(rows: list[dict], date: str) -> str | None:
    """Write a compact news+state snapshot for MANUAL Claude analysis (the 'hand Claude the info'
    option): per EP ticker with news — its live state + catalyst/sentiment/reaction + the OWN and PEER
    headlines. -> data/news_analysis/{date}.json (+ a stable latest.json pointer). Distilled, not raw
    bars (ties memory live-data-file-for-claude-analysis)."""
    import config
    keep = ("symbol", "name", "move_pct", "gap_pct", "above_vwap", "cum_rvol", "dollar_vol_m",
            "catalyst", "sent", "conflicting", "news_react", "news_warn", "headline",
            "peer_headline", "n_stories", "setups")
    out = [{k: r.get(k) for k in keep} for r in rows if r.get("n_stories")]
    if not out:
        return None
    blob = {"date": date, "generated": pd.Timestamp.now(tz="America/New_York").isoformat(),
            "note": "EP-radar news + live state for manual analysis (good-news-bad-reaction etc.)",
            "tickers": out}
    d = config.DATA_DIR / "news_analysis"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{date}.json").write_text(json.dumps(blob, indent=2, default=str))
    (d / "latest.json").write_text(json.dumps(blob, indent=2, default=str))
    return str(d / "latest.json")


# ---------------------------------------------------------------- live intraday EP radar

# expected fraction of a full day's volume completed by each 15m bar (U-shaped session curve —
# same shape the breakout-stack projection uses in setups._project_last_volume). Hand-typed,
# front-loaded/consolidated-style — kept for the legacy 15m path only.
_VOL_CURVE_15 = [0.10, 0.17, 0.22, 0.27, 0.31, 0.35, 0.38, 0.41, 0.44, 0.47, 0.50, 0.52,
                 0.55, 0.57, 0.60, 0.62, 0.65, 0.67, 0.70, 0.73, 0.76, 0.80, 0.84, 0.89, 0.95, 1.0]

# 5m base (78 RTH slots). Derived empirically from the Tiingo 5m store — median cumulative-volume
# fraction across 15,151 (symbol, session) full days (phase0_verify.py, 2026-07-06). MUST be IEX-basis
# to match the numerator it divides: live `dayv` = tiingo_live.DAYVOL x vol_factor is IEX-shaped, and
# IEX (a continuous lit venue) carries NONE of the open/close auction, so real volume is far more
# back-loaded than the hand-typed 15m curve assumed (18% done by 11:00 vs the old 35%). Using this
# corrects a latent live-mode miscalibration that made morning pace read systematically LOW.
_VOL_CURVE_5 = [0.006, 0.014, 0.021, 0.031, 0.039, 0.049, 0.060, 0.070, 0.080, 0.090, 0.101, 0.110, 0.122,
                0.134, 0.145, 0.156, 0.167, 0.178, 0.190, 0.201, 0.213, 0.225, 0.236, 0.248, 0.259, 0.271,
                0.281, 0.291, 0.302, 0.312, 0.324, 0.334, 0.344, 0.354, 0.365, 0.375, 0.385, 0.395, 0.405,
                0.414, 0.424, 0.433, 0.444, 0.453, 0.463, 0.472, 0.482, 0.491, 0.500, 0.510, 0.520, 0.531,
                0.540, 0.551, 0.562, 0.572, 0.582, 0.594, 0.605, 0.616, 0.627, 0.639, 0.651, 0.662, 0.674,
                0.688, 0.701, 0.714, 0.728, 0.743, 0.759, 0.777, 0.799, 0.821, 0.845, 0.876, 0.920, 1.000]

_VOLF: dict | None = None


def vol_factor() -> dict:
    """Per-symbol IEX->consolidated volume multiplier (data/iex_vol_factor.json, ~22x median).
    IEX prints 1-6% of the tape and the share varies per symbol — raw IEX volume must never be
    compared against consolidated averages without this."""
    global _VOLF
    if _VOLF is None:
        try:
            _VOLF = json.loads((config.DATA_DIR / "iex_vol_factor.json").read_text())
        except (OSError, ValueError):
            _VOLF = {}
    return _VOLF


def _live_dayvol(sym: str) -> float | None:
    """Today's day-cumulative volume in CONSOLIDATED terms, from the live quote ledger (IEX raw x
    calibration factor). None when the poller isn't running / no quote yet."""
    try:
        import tiingo_live
        raw = tiingo_live.DAYVOL.get(sym)
    except Exception:
        return None
    if not raw:
        return None
    return float(raw) * float(vol_factor().get(sym, 22.0))


DAYV_XCHECK_RATIO = 3.0      # quote-basis vs bar-sum-basis disagreement that flips to the bar sum


def _dayv_checked(sym: str, t_bars, live: bool = True) -> tuple[float, bool]:
    """Day-cumulative volume with an UNCONDITIONAL cross-check (open-hardening plan B, Amir
    2026-07-09): the quote-basis day volume is validated against the calibrated bar sum ALWAYS —
    not only above RVOL_SANITY_MAX. The 09:38 flood sat just UNDER the cap (stale quotes put
    yesterday's full day into DAYVOL ~ 1000x pace two bars into the session). When the two bases
    disagree by > DAYV_XCHECK_RATIO the reliable bar sum wins and the event is counted
    (tiingo_live.STATE['dayv_xcheck'] -> /api/live/status). First minutes with an empty/warming
    bar sum can't be validated — the sanity cap still guards those.
    `live=False` (a replay/as_of read) NEVER consults the poller's DAYVOL: that ledger holds the
    CURRENT session's day-volume, and a time-slider replay run while the market is open was
    serving today's leaders on a past day's board (live QA 2026-07-09: the 07-07 10:30 replay
    mirrored the live board). Replay bars are already consolidated-scale, so the bar sum IS the
    right answer there — and the x-factor cross-check would be double-calibrated anyway.
    Returns (dayv_consolidated, live_basis) — live_basis=True means bars are IEX-raw and
    downstream must calibrate x vol_factor (unchanged semantics)."""
    bar_sum = float(t_bars["volume"].sum())
    dayv = _live_dayvol(sym) if live else None
    if dayv is None:
        return bar_sum, False                    # offline/replay: bars already consolidated
    bar_cons = bar_sum * float(vol_factor().get(sym, 22.0))    # live bars are IEX raw
    if bar_cons > 0 and dayv > DAYV_XCHECK_RATIO * bar_cons:
        try:
            import tiingo_live
            tiingo_live.STATE["dayv_xcheck"] = int(tiingo_live.STATE.get("dayv_xcheck") or 0) + 1
        except Exception:
            pass
        return bar_cons, True
    return dayv, True


def live_radar(frames: dict, frames15: dict, min_move: float | None = None, session: str | None = None,
               min_pace: float = 3.0, min_dvol_m: float = 10.0, base_min: int = 15,
               as_of: str | None = None) -> list[dict]:
    """EP events FORMING today: |move| >= EP9M_MIN_CHG (or gap >= EP_MIN_GAP) with cumulative
    volume running >= `min_pace` x the expected pace for this time of day. Returns rows sorted
    by pace; news/catalyst attached for the top rows by the caller (keep API calls modest).

    `frames15` is the base intraday RAM store; `base_min` picks the matching pace curve — 5 for the
    Tiingo 5m base (IEX-shaped _VOL_CURVE_5), 15 for the legacy yfinance 15m base. `as_of` (a date)
    anchors the DAILY reads (vol_avg50/dollar_vol/ret_6m) to that prior settled session for
    point-in-time REPLAY — without it the daily frame's last bar (today) would be look-ahead."""
    min_move = min_move if min_move is not None else config.EP9M_MIN_CHG
    curve = _VOL_CURVE_5 if base_min == 5 else _VOL_CURVE_15
    out = []
    for sym, f15 in frames15.items():
        d = frames.get(sym)
        if d is None or f15 is None or not len(f15):
            continue
        if as_of:
            d = d.loc[:as_of]                        # replay: daily context as-of the prior session
            if not len(d):
                continue
        dv = d["dollar_vol"].iloc[-1]
        if pd.isna(dv) or dv / 1e6 < min_dvol_m:
            continue
        dates = f15.index.date
        today = dates[-1]
        if session is not None and str(today) != session:
            continue          # frame tail is a PRIOR session (no live quotes yet after a restart) —
        #                       yesterday's full day must not read as "today" (2026-07-08 live QA:
        #                       thin names showed frac=1.0 stale boards until their first quote)
        tmask = dates == today
        n_bars = int(tmask.sum())
        if n_bars < 2:
            continue
        prev = f15[~tmask]
        if not len(prev):
            continue
        prev_close = float(prev["close"].iloc[-1])
        t_bars = f15[tmask]
        last = float(t_bars["close"].iloc[-1])
        move = (last / prev_close - 1) * 100
        if abs(move) > 100:
            continue                                  # reverse-split / bad-print artifact, not a move
        o = float(t_bars["open"].iloc[0])
        gap = (o / prev_close - 1) * 100
        if move < min_move and gap < config.EP_MIN_GAP:
            continue
        avg_day_vol = float(d["vol_avg50"].iloc[-1]) if pd.notna(d["vol_avg50"].iloc[-1]) else None
        if not avg_day_vol:
            continue
        frac = curve[min(n_bars, len(curve)) - 1]
        dayv, _live = _dayv_checked(sym, t_bars, live=as_of is None)  # UNCONDITIONAL cross-check (plan B)
        pace = dayv / (avg_day_vol * frac)
        if pace > config.RVOL_SANITY_MAX:
            continue                                 # still absurd after the cross-check -> bad print
        if pace < min_pace:
            continue
        tp = (t_bars["high"].astype(float) + t_bars["low"].astype(float) + t_bars["close"].astype(float)) / 3
        vsum = float(t_bars["volume"].astype(float).sum())
        vwap = float((tp * t_bars["volume"].astype(float)).sum() / vsum) if vsum > 0 else last
        out.append({"symbol": sym, "move_pct": round(move, 1), "gap_pct": round(gap, 1),
                    "vol_pace": round(pace, 1), "shares_m": round(dayv / 1e6, 1),
                    "neglect_6m": (round(float(d["ret_6m"].iloc[-1]), 1)
                                   if pd.notna(d["ret_6m"].iloc[-1]) else None),
                    # dollar_vol_m = rolling AVG $/day (liquidity); live_dvol_m = TODAY's traded $ so far
                    # (day-shares x session VWAP). See rvol_leaders for the rationale (BUGS 2026-07-09).
                    "dollar_vol_m": round(float(dv) / 1e6, 1),
                    "live_dvol_m": round(dayv * vwap / 1e6, 1),
                    "bars": n_bars})
    out.sort(key=lambda r: -r["vol_pace"])
    return out


def open_radar(frames: dict, min_gap: float = 4.0, min_dollar_m: float = 1.0) -> list[dict]:
    """PRE-MARKET EP radar (Bonde's early entry: strong overnight move + strong overnight volume,
    tradeable at the open before any daily bar exists). Reads the poller's 08:00-09:30 IEX quote
    buffer: overnight gap vs yesterday's close + calibrated pre-market dollar volume. IEX carries
    only part of pre-market flow, so treat the volume as a RANKING signal, not a measurement."""
    try:
        import tiingo_live
        pm = dict(tiingo_live.PREMARKET)
    except Exception:
        return []
    out = []
    vf = vol_factor()
    # STALE-CLOSE GUARD (live-verified 2026-07-09 08:15 ET): "yesterday's close" must actually be
    # the LAST SESSION. Theme/rotation ETFs (IXC/COPX/SIL/MSOS) sat at 07-02 daily bars while the
    # universe was on 07-08 — their "overnight gap" was 4 sessions of drift, and the gap x $vol
    # sort then selected exactly the broken names (garbage survivorship). Reference session =
    # newest daily bar across the index ETFs.
    ref = None
    for s in ("SPY", "QQQ", "IWM"):
        f = frames.get(s)
        if f is not None and len(f):
            ref = f.index[-1] if ref is None else max(ref, f.index[-1])
    for sym, q in pm.items():
        d = frames.get(sym)
        if d is None or not len(d):
            continue
        if ref is not None and d.index[-1] != ref:
            continue                                   # stale daily bar -> gap would be fiction
        prev = float(d["close"].iloc[-1])
        if prev <= 0:
            continue
        gap = (q["last"] / prev - 1) * 100
        if abs(gap) < min_gap or abs(gap) > 100:
            continue
        dollar_m = q["volume"] * q["last"] * float(vf.get(sym, 22.0)) / 1e6
        if dollar_m < min_dollar_m:
            continue
        avg_dvol = d["dollar_vol"].iloc[-1]
        ret6 = d["ret_6m"].iloc[-1] if "ret_6m" in d else None
        pm_hi, pm_lo = q.get("high"), q.get("low")     # pre-market swing so far (fresh-gated in the buffer)
        pm_range_pct = (round((float(pm_hi) - float(pm_lo)) / prev * 100, 1)
                        if pm_hi and pm_lo and prev > 0 else None)
        out.append({"symbol": sym, "pm_gap_pct": round(gap, 1),
                    "pm_dollar_m": round(dollar_m, 2),
                    "pm_range_pct": pm_range_pct,
                    "avg_dollar_m": round(float(avg_dvol) / 1e6, 1) if pd.notna(avg_dvol) else None,
                    "neglect_6m": round(float(ret6), 1) if ret6 is not None and pd.notna(ret6) else None})
    out.sort(key=lambda r: -(abs(r["pm_gap_pct"]) * r["pm_dollar_m"]))
    return out


def rvol_leaders(frames: dict, frames15: dict, min_dvol_m: float = 5.0, session: str | None = None,
                 min_pace: float = 1.3, limit: int = 60, base_min: int = 15,
                 as_of: str | None = None, min_dvol_map: dict | None = None) -> list[dict]:
    """Whole-universe INTRADAY RVOL leaders (no EP move/gap filter — just 'who is heavy now').
    Reuses live_radar's calibrated cumulative-pace math:
      cum_rvol  = calibrated day-cumulative volume / (avg daily vol x expected time-of-day fraction)
                  — the reliable 'is the whole day running hot' read.
      slot_rvol = latest 15m bar volume / expected volume for THIS one slot (avg_day_vol x Δfraction)
                  — a best-effort current-bar spike flag (IEX-basis in live -> approximate; exact on
                  replay/yfinance). Leads with cum_rvol; slot is secondary, as flagged to Amir.
    Ranked by cum_rvol desc; above_vwap carries the direction gate for the caller.
    `base_min` picks the pace curve (5 = Tiingo 5m IEX-shaped _VOL_CURVE_5, 15 = legacy 15m).
    `as_of` (a date) anchors the daily reads to that prior settled session for point-in-time replay.
    `min_dvol_map` (sym -> $vol floor M): per-name liquidity floor (a name in a thin-friendly EP/gapper
    setup qualifies below the generic `min_dvol_m`); missing names fall back to `min_dvol_m`."""
    curve = _VOL_CURVE_5 if base_min == 5 else _VOL_CURVE_15
    out = []
    for sym, f15 in frames15.items():
        d = frames.get(sym)
        if d is None or f15 is None or not len(f15):
            continue
        if as_of:
            d = d.loc[:as_of]                        # replay: daily context as-of the prior session
            if not len(d):
                continue
        dv = d["dollar_vol"].iloc[-1]
        floor = min_dvol_map.get(sym, min_dvol_m) if min_dvol_map else min_dvol_m
        if pd.isna(dv) or dv / 1e6 < floor:          # per-setup live floor -> thin-name EPs qualify lower
            continue
        dates = f15.index.date
        today = dates[-1]
        if session is not None and str(today) != session:
            continue          # frame tail is a PRIOR session (no live quotes yet after a restart) —
        #                       yesterday's full day must not read as "today" (2026-07-08 live QA:
        #                       thin names showed frac=1.0 stale boards until their first quote)
        tmask = dates == today
        n_bars = int(tmask.sum())
        if n_bars < 1:
            continue          # FAST OPEN (weekend sweep 2026-07-11): 1-bar rows ARE served — on the
        #                       07-10 tape they were the only signal 09:31-09:36 (CIBR/CRCL, on their
        #                       pre-market volume) while the old 2-bar gate showed nothing until 09:48.
        #                       Safety held at every cutoff on both tapes (0 rows >100×): Plan A gates
        #                       DAYVOL and the Plan-B cross-check works with one bar (poison-verified).
        #                       Rows with <2 bars carry warming=True so the UI marks them as settling.
        prev = f15[~tmask]
        if not len(prev):
            continue
        avg_day_vol = float(d["vol_avg50"].iloc[-1]) if pd.notna(d["vol_avg50"].iloc[-1]) else None
        if not avg_day_vol:
            continue
        t_bars = f15[tmask]
        prev_close = float(prev["close"].iloc[-1])
        last = float(t_bars["close"].iloc[-1])
        move = (last / prev_close - 1) * 100
        if abs(move) > 100:
            continue                                       # split / bad-print artifact
        frac = curve[min(n_bars, len(curve)) - 1]
        dayv, live = _dayv_checked(sym, t_bars, live=as_of is None)   # UNCONDITIONAL cross-check (plan B)
        cum_rvol = dayv / (avg_day_vol * frac) if frac > 0 else None
        if cum_rvol is None or cum_rvol > config.RVOL_SANITY_MAX:  # still absurd after the
            continue                                               # cross-check = real bad print
        if cum_rvol < min_pace:
            continue
        # slot RVOL: latest bar vs the expected volume for just this slot (Δ of the U-curve). When the latest
        # bar is UNFILLED (0 vol — a warming/incomplete live 5m store, or a lagging slot) the slot spike can't
        # be computed; emit None so the UI shows "—" instead of a misleading "×0.0" (Amir 2026-07-07).
        frac_prev = curve[min(n_bars - 1, len(curve)) - 1] if n_bars >= 2 else 0.0
        dfrac = max(frac - frac_prev, 1e-4)
        last_bar = float(t_bars["volume"].iloc[-1])
        if last_bar > 0:
            last_bar_cons = last_bar * float(vol_factor().get(sym, 22.0)) if live else last_bar
            slot_rvol = last_bar_cons / (avg_day_vol * dfrac)
        else:
            slot_rvol = None
        # session VWAP for the direction gate
        tp = (t_bars["high"].astype(float) + t_bars["low"].astype(float) + t_bars["close"].astype(float)) / 3
        vsum = float(t_bars["volume"].astype(float).sum())
        vwap = float((tp * t_bars["volume"].astype(float)).sum() / vsum) if vsum > 0 else last
        out.append({"symbol": sym, "cum_rvol": round(cum_rvol, 1),
                    "slot_rvol": (round(slot_rvol, 1) if slot_rvol is not None else None),
                    "move_pct": round(move, 1), "above_vwap": last > vwap,
                    "shares_m": round(dayv / 1e6, 1),
                    # dollar_vol_m = the name's ROLLING AVG $/day (liquidity, d["dollar_vol"]); NOT today's
                    # traded $ — mislabelled bare "$M" in the UI (BUGS 2026-07-09). live_dvol_m = TODAY's
                    # actual traded $ so far (consolidated day-shares x session VWAP) — the true "how much
                    # money went through today" read that pairs with the avg for a heavy/light call.
                    "dollar_vol_m": round(float(dv) / 1e6, 1),
                    "live_dvol_m": round(dayv * vwap / 1e6, 1),
                    "warming": n_bars < 2,   # 1-bar read (fast open): pace estimate still settling
                    "bars": n_bars})
    out.sort(key=lambda r: -r["cum_rvol"])
    return out[:limit]


def radar_with_news(frames: dict, frames15: dict, top_news: int = 15,
                    names: dict | None = None, dump: bool = False, **kw) -> list[dict]:
    """live_radar + news/catalyst/SENTIMENT/reaction-warning attached to the top rows (bounded API
    usage). `names` (sym->company name) sharpens OWN-vs-PEER attribution; `dump` writes the manual-
    analysis file. Reaction warnings (good-news-red etc.) use each row's today move_pct."""
    rows = live_radar(frames, frames15, **kw)
    today = None
    for r in rows[:top_news]:
        f15 = frames15.get(r["symbol"])
        today = str(f15.index.date[-1]) if f15 is not None and len(f15) else today
        nm = (names or {}).get(r["symbol"])
        if names is not None:
            r["name"] = nm
        if today:
            r.update(news_signal(r["symbol"], news_for(r["symbol"], today), r.get("move_pct"), nm))
        else:
            r["catalyst"] = "unknown"
    if dump and today:
        try:
            dump_news_analysis(rows[:top_news], today)
        except Exception:
            pass
    return rows
