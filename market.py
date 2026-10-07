"""Top-down MARKET situational-awareness score: long / out / short with plain-English reasons.

Combines four weighted blocks into a composite in [-100, +100] -> a 5-band stance:
  1. Trend   — SPY / QQQ / IWM posture (daily + weekly), reusing mtf.tf_state (MA stack, tilt, above-MAs,
               trend clarity); ATR extension from the 50-MA is a *caution* term (don't chase).
  2. Breadth — one pass over the active universe: % above 50/200-SMA, advancers:decliners, new 52wk H vs L.
  3. Vol     — VIX / VXN level + trend (calm & falling = risk-on; high & rising = risk-off).
  4. Macro   — 10-yr yield (^TNX), long bonds (TLT), the dollar (DX-Y.NYB) trend as a context modifier.

Each block returns a score in [-1, +1] and a one-line `why`. Pure/read-only: reuses datastore, indicators,
mtf. Macro/index bars come from the cache (datastore.update_macro keeps the macro symbols fresh).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

import build_labels
import config
import datastore
import intraday
import labels
import mtf
import ratios
import sr
import tv_breadth
import universe
from indicators import add_indicators


def _sr_phrase(d) -> str:
    """Nearest support/resistance as a compact awareness phrase, e.g. '0.3 ATR under ×15 R · 0.8 over ×9 S'."""
    if d is None:
        return ""
    lv = sr.sr_levels(d)
    res, sup = lv.get("resistance"), lv.get("support")
    bits = []
    if res:
        bits.append(f"{res[0]['dist_atr']:.1f} ATR under ×{res[0]['strength']} R")
    if sup:
        bits.append(f"{sup[0]['dist_atr']:.1f} over ×{sup[0]['strength']} S")
    return " · ".join(bits)

_STANCES = [                                   # (min composite, label, color)
    ("strong_long",  "Strong Long",   "#1f9d5b"),
    ("cautious_long", "Cautious Long", "#2fbf8f"),
    ("cautious_short", "Neutral · Stay Out", "#8aa0b8"),   # the band ABOVE cautious_short edge
    ("short",        "Cautious Short", "#e0913a"),
    (None,           "Short",          "#e05a6d"),
]


def _frame(sym: str, as_of: str, weekly: bool, cache: dict):
    """Enriched daily (or weekly) frame for sym, sliced to as_of. Cached per call."""
    key = (sym, weekly)
    if key in cache:
        return cache[key]
    raw = datastore.load_bars(sym)
    if raw is not None and weekly:
        raw = datastore.resample_weekly(raw)
    d = None
    if raw is not None and len(raw) > 30:
        d = add_indicators(raw)
        d = d.loc[:as_of] if as_of else d
        if len(d) < 30:
            d = None
    cache[key] = d
    return d


def _score_state(st) -> float:
    """One tf_state -> posture score ~[-2, +2]: MA stack + rising tilt + above-MAs + trend clarity;
    ATR extension from the 50-MA trims conviction (don't chase an overextended tape)."""
    sc = st["stack"] / 4.0 + (st["tilt"] / 5.0) * 0.5
    if st["gt_sma50"] and st["gt_sma200"]:
        sc += 0.5
    elif st["gt_sma50"] is False and st["gt_sma200"] is False:
        sc -= 0.5
    if st["trend_dir"] == "up":
        sc += 0.3 * (st["trend_clarity"] or 0)
    elif st["trend_dir"] == "down":
        sc -= 0.3 * (st["trend_clarity"] or 0)
    ext = st["atr_ext_50"]
    if ext is not None and abs(ext) > config.MARKET_EXT_CAP:
        sc -= 0.5 * (1 if ext > 0 else -1)
    return float(np.clip(sc, -2, 2))


def _posture(sym: str, as_of: str, cache: dict):
    """A single symbol's DAILY+weekly trend posture in ~[-2, +2] + a readable phrase, from tf_state."""
    states = []
    for weekly in (False, True):
        d = _frame(sym, as_of, weekly, cache)
        st = mtf.tf_state(d) if d is not None else None
        if st is not None:
            states.append((weekly, st))
    if not states:
        return None, f"{sym}: no data"
    score = sum(_score_state(st) for _, st in states) / len(states)
    st = states[0][1]                                              # daily for the phrase
    posn = "above" if st["gt_ema20"] else "below"
    stk = "stacked bull" if st["stack"] >= 3 else ("stacked bear" if st["stack"] <= -3 else "mixed MAs")
    tlt = "rising" if st["tilt"] >= 2 else ("falling" if st["tilt"] <= -2 else "flat")
    why = f"{sym} {posn} 20-EMA, {stk}, {tlt}"
    if len(states) > 1:
        wk = states[1][1]["trend_dir"]
        why += f"; weekly {'up' if wk == 'up' else ('down' if wk == 'down' else 'flat')}"
    ext = st["atr_ext_50"]
    if ext is not None and abs(ext) > config.MARKET_EXT_CAP:
        why += f" ({ext:+.1f}×ATR — overextended)"
    srp = _sr_phrase(_frame(sym, as_of, False, cache))     # daily frame (cached) -> nearest S/R
    if srp:
        why += f"; {srp}"
    return score, why


def _posture_intraday(sym: str, frames15: dict, cache: dict):
    """A symbol's INTRADAY posture (1h + 4h, resampled from 15m) — the live analogue of _posture."""
    df15 = frames15.get(sym)
    if df15 is None or len(df15) < 20:
        return None, f"{sym}: no 15m"
    states = []
    for tf in ("1h", "4h"):
        key = (sym, "i" + tf)
        if key in cache:
            d = cache[key]
        else:
            raw = datastore.resample_intraday(df15, tf)
            d = add_indicators(raw) if (raw is not None and len(raw) > 30) else None
            cache[key] = d
        st = mtf.tf_state(d) if d is not None else None
        if st is not None:
            states.append((tf, st))
    if not states:
        return None, f"{sym}: no intraday"
    score = sum(_score_state(st) for _, st in states) / len(states)
    st = states[0][1]
    posn = "above" if st["gt_ema20"] else "below"
    tlt = "rising" if st["tilt"] >= 2 else ("falling" if st["tilt"] <= -2 else "flat")
    return score, f"{sym} 1h {posn} 20-EMA, {tlt}"


# ---- the four blocks (each -> (score in [-1,1], why) or (None, why) if no data) ----------

def _trend_block(as_of, cache, live=False, frames15=None):
    syms = ("SPY", "QQQ", "IWM", "DIA")     # Dow added 2026-07-05 (Amir: "add Dow to health")
    if live and frames15:
        ps = [(s, *_posture_intraday(s, frames15, cache)) for s in syms]
    else:
        ps = [(s, *_posture(s, as_of, cache)) for s in syms]
    ps = [(s, sc, w) for (s, sc, w) in ps if sc is not None]
    if not ps:
        return None, "no index data"
    score = float(np.clip(np.mean([sc for _, sc, _ in ps]) / 2.0, -1, 1))   # posture ~[-2,2] -> [-1,1]
    return score, "; ".join(w for _, _, w in ps)


def _breadth_block(as_of, frames, live=False, frames15=None):
    if live and frames15:                                          # intraday internals feed the live score
        ib = intraday.intraday_breadth(frames15, frames)
        if ib is None:
            return None, "insufficient intraday"
        return ib["score"], ib["internals"]["why"]
    exch = universe.exchange_map()
    a50 = a200 = adv = dec = nh = nl = n = 0
    nq_n = nq_up = ny_n = ny_up = 0                 # NASDAQ (≈tech) vs NYSE (≈broad) advancers
    for sym, d in frames.items():
        if d is None or not len(d):
            continue
        sub = d.loc[:as_of] if as_of else d
        if not len(sub):
            continue
        r = sub.iloc[-1]
        c = r.get("close")
        if c is None or pd.isna(c):
            continue
        n += 1
        if pd.notna(r.get("sma50")) and c > r["sma50"]:
            a50 += 1
        if pd.notna(r.get("sma200")) and c > r["sma200"]:
            a200 += 1
        ch = r.get("chg_pct")
        up = bool(pd.notna(ch) and ch > 0)
        if pd.notna(ch):
            adv += ch > 0
            dec += ch < 0
            e = exch.get(sym, "")
            if e == "NASDAQ":
                nq_n += 1; nq_up += up
            elif e == "NYSE":
                ny_n += 1; ny_up += up
        oh = r.get("off_hi52_pct")
        if pd.notna(oh):
            nh += oh >= -1.0            # within 1% of the 52wk high
        al = r.get("above_lo52_pct")
        if pd.notna(al):
            nl += al <= 3.0             # within 3% of the 52wk low
    if n < 20:
        return None, "insufficient universe"
    p50, p200 = a50 / n * 100, a200 / n * 100
    ad = (adv / dec) if dec else (adv or 1)
    # sample terms (each ~[-1,1]) — the fallback / self-computed breadth
    t50, t200 = (p50 - 50) / 50, (p200 - 50) / 50
    t_ad, t_nhnl = float(np.tanh(ad - 1)), float(np.tanh((nh - nl) / 25.0))
    why = f"{p50:.0f}% >50-MA, {p200:.0f}% >200-MA, adv:dec {ad:.1f}:1, {nh} new-highs vs {nl} lows"
    if nq_n >= 20 and ny_n >= 20:      # venue split — NASDAQ weaker than NYSE = tech-specific drag
        nyp, nqp = ny_up / ny_n * 100, nq_up / nq_n * 100
        why += f"; NYSE {nyp:.0f}% vs NASDAQ {nqp:.0f}% up"
    # --- true exchange breadth overlay (tv_breadth): swap in real internals per-metric, else keep sample ---
    tvb = tv_breadth.internals(as_of)                      # None if toggle off / no data → pure sample
    if tvb is not None:
        t_ad = tvb["ad_term"]                              # net A−D from $ADD/$ADDQ drives the breadth term
        if tvb.get("pct_50") is not None:
            t50 = (tvb["pct_50"] - 50) / 50
        if tvb.get("pct_200") is not None:
            t200 = (tvb["pct_200"] - 50) / 50
        if tvb.get("nh") is not None and tvb.get("nl") is not None:
            t_nhnl = float(np.tanh((tvb["nh"] - tvb["nl"]) / 25.0))
        add_s = f"NYSE {tvb['add']:+.0f}" + (f" / NASDAQ {tvb['addq']:+.0f}" if tvb.get("addq") is not None else "")
        parts = [f"{add_s} net adv−dec (exchange)"]
        p50s = tvb["pct_50"] if tvb.get("pct_50") is not None else p50    # true %>MA if extracted, else self-computed
        p200s = tvb["pct_200"] if tvb.get("pct_200") is not None else p200
        parts.append(f"{p50s:.0f}% >50-MA, {p200s:.0f}% >200-MA")
        if tvb.get("trin") is not None:
            parts.append("TRIN " + f"{tvb['trin']:.2f}" + (f"/{tvb['trinq']:.2f} (NY/NQ)" if tvb.get("trinq") is not None else " (NYSE)"))
        ref_date = str(as_of)[:10] if as_of else _today_et()          # note when the extract lags the view
        if tvb.get("date") and str(tvb["date"]) != ref_date:
            parts.append(f"breadth as of {tvb['date']}")
        why = "; ".join(parts)
    score = float(np.clip((t50 + t200 + t_ad + t_nhnl) / 4.0, -1, 1))
    return score, why


def _trend_mag(sym, as_of, cache):
    """Continuous trend strength in [-1,+1] (NOT a hard sign): the last close's distance from its own
    20-EMA in ATRs, tanh-squashed by MARKET_MACRO_ATR. +ve = above/rising. A marginal cross ~= 0, a
    sustained trend far from the EMA ~= ±1 — so the macro block nudges rather than slams on tiny moves."""
    d = _frame(sym, as_of, False, cache)
    if d is None:
        return None
    r = d.iloc[-1]
    ema, atr = r.get("ema20"), r.get("atr14")
    if pd.isna(ema) or pd.isna(atr) or atr <= 0:
        return None
    z = (float(r["close"]) - float(ema)) / float(atr)
    return float(np.tanh(z / config.MARKET_MACRO_ATR))


def _vol_block(as_of, cache):
    reasons = []
    scores = []
    levels = {}
    for sym, label in (("^VIX", "VIX"), ("^VXN", "VXN")):
        d = _frame(sym, as_of, False, cache)
        if d is None:
            continue
        r = d.iloc[-1]
        v = float(r["close"])
        levels[label] = v
        lvl = float(np.clip((config.MARKET_VIX_HIGH - v) / (config.MARKET_VIX_HIGH - config.MARKET_VIX_LOW) * 2 - 1, -1, 1))
        rising = pd.notna(r.get("ema10")) and v > r["ema10"]
        sc = float(np.clip(lvl + (-0.4 if rising else 0.4), -1, 1))
        scores.append(sc)
        if sym == "^VIX":
            band = "calm" if v <= config.MARKET_VIX_LOW else ("elevated" if v >= config.MARKET_VIX_HIGH else "moderate")
            # lead with STRUCTURE (Amir 2026-07-05: "recent structures are more telling
            # than the values") — trendlab on the vol index's own daily chart
            struct = ""
            try:
                import trendlab
                if len(d) > 60:
                    w = trendlab.struct_label(int(trendlab.structure_series(d)[-1]))
                    w = {"up": "UPTREND — vol building", "down": "DOWNTREND — vol compressing",
                         "range/contracting": "coiling range",
                         "range/rectangle": "rangebound", "range/expanding": "whipsaw range",
                         "transition": "structure unclear"}.get(w, w)
                    struct = f"vol structure: {w} · "
            except Exception:
                pass
            reasons.append(f"{struct}VIX {v:.1f} ({band}, {'rising' if rising else 'falling'})")
    if not scores:
        return None, "no VIX data"
    why = reasons[0] if reasons else ""
    if "VIX" in levels and "VXN" in levels:      # a wide VXN−VIX spread = tech-specific (NASDAQ) risk
        spread = levels["VXN"] - levels["VIX"]
        if spread >= 6.0:
            why += f"; VXN {levels['VXN']:.0f} (+{spread:.0f} vs VIX — tech risk)"
    return float(np.mean(scores)), why


def _macro_block(as_of, cache, partial=False):
    tnx = _trend_mag("^TNX", as_of, cache)      # yields up = headwind
    tlt = _trend_mag("TLT", as_of, cache)       # bonds up = supportive
    dxy = _trend_mag("DX-Y.NYB", as_of, cache)  # dollar up = headwind
    terms, bits = [], []
    if tnx is not None:
        terms.append(-tnx); bits.append(f"10y yield {'rising' if tnx > 0 else 'falling'}")
    if tlt is not None:
        terms.append(tlt)
    if dxy is not None:
        terms.append(-dxy); bits.append(f"dollar {'firm' if dxy > 0 else 'soft'}")
    if not terms:
        return None, "no macro data"
    sc = float(np.mean(terms))
    if partial:
        sc *= config.MARKET_PARTIAL_DAMP
    why = ", ".join(bits) + (" · intraday (damped)" if partial else "")
    return float(np.clip(sc, -1, 1)), why


def _risk_block(as_of, frames, live=False, frames15=None):
    """Risk-on/off ROTATION: a basket of index/sector ratios (RTY/NQ, NQ/ES, SMH/SPY, XLY/XLP, HG/GC, …).
    Each ratio's posture (mtf.tf_state via ratios) -> [-1,1] * its risk-on sign; the mean is the block score.
    The `why` names the leading risk-on and risk-off contributors — the rotation signal the app was missing."""
    contrib = []
    tf = "1h" if live else "1D"                       # live = intraday ratio posture (1h), else daily
    for a, b, sign, label in config.RATIO_BASKET:
        rs = ratios.ratio_state(a, b, frames, frames15, live=live, tf=tf,
                                as_of=(None if live else as_of))
        if rs is None:
            continue
        s = float(np.clip(_score_state(rs["state"]) / 2.0, -1, 1)) * sign     # posture ~[-2,2] -> [-1,1]
        contrib.append((label, s))
    if not contrib:
        return None, "no ratio data"
    score = float(np.clip(np.mean([s for _, s in contrib]), -1, 1))
    contrib.sort(key=lambda x: -x[1])
    on = [l for l, s in contrib if s > 0.10][:3]
    off = [l for l, s in contrib if s < -0.10][:3]
    bits = []
    if on:
        bits.append("risk-on: " + ", ".join(on))
    if off:
        bits.append("risk-off: " + ", ".join(off))
    return score, ("; ".join(bits) or "mixed/neutral rotation")


def _stance(score: float):
    b = config.MARKET_BANDS
    if score >= b["strong_long"]:
        return _STANCES[0][1], _STANCES[0][2]
    if score >= b["cautious_long"]:
        return _STANCES[1][1], _STANCES[1][2]
    if score >= b["cautious_short"]:
        return _STANCES[2][1], _STANCES[2][2]
    if score >= b["short"]:
        return _STANCES[3][1], _STANCES[3][2]
    return _STANCES[4][1], _STANCES[4][2]


def _is_partial(as_of: str | None) -> bool:
    """True if this is a LIVE partial-bar read: as_of is today AND the US market is currently open."""
    today = dt.date.today().isoformat()
    if as_of and as_of != today:
        return False
    try:
        from zoneinfo import ZoneInfo
        now = dt.datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return False
    return now.weekday() < 5 and dt.time(9, 30) <= now.time() < dt.time(16, 0)


def no_restart_window(now: dt.datetime | None = None) -> bool:
    """True during the 09:15-10:00 ET open window when a server restart degrades the board for
    minutes: a full 5m reseed lags the movers behind the tape, so RVOL/grade surfaces go partial
    (live-ops standing rule — no restarts here unless the board is actively broken; plan E). `now`
    (aware/naive ET) is injectable for tests; defaults to real ET wall-clock. Weekends -> False."""
    if now is None:
        try:
            from zoneinfo import ZoneInfo
            now = dt.datetime.now(ZoneInfo("America/New_York"))
        except Exception:
            return False
    if now.weekday() >= 5:
        return False
    return dt.time(9, 15) <= now.time() < dt.time(10, 0)


def daily_time_due(now: dt.datetime, hhmm: str, last_fired_day: str | None) -> bool:
    """True when a once-per-weekday timed job is due: `now` (ET) is a weekday at/after HH:MM and the
    job hasn't fired yet today (last_fired_day != today). Uses >= (not ==) so a coarse ~30s scheduler
    poll can't miss the target minute. Drives the pre-open day-type prep (serve._scheduler). Malformed
    hhmm -> False (never fire on a bad config value)."""
    if now.weekday() >= 5:
        return False
    try:
        h, m = int(hhmm[:2]), int(hhmm[3:5])
    except (ValueError, IndexError, TypeError):
        return False
    return now.time() >= dt.time(h, m) and last_fired_day != now.date().isoformat()


def market_score(as_of: str | None, frames: dict, live: bool = False, frames15: dict | None = None) -> dict:
    """Composite market stance for as_of. live=True routes the trend + breadth blocks through the 15m
    intraday data (frames15); vol + macro stay on daily. `frames` = {sym: enriched daily frame}."""
    cache: dict = {}
    W = config.MARKET_WEIGHTS
    partial = _is_partial(as_of)
    blocks = []
    for name, fn in (("trend", lambda: _trend_block(as_of, cache, live, frames15)),
                     ("breadth", lambda: _breadth_block(as_of, frames, live, frames15)),
                     ("vol", lambda: _vol_block(as_of, cache)),
                     ("macro", lambda: _macro_block(as_of, cache, partial)),
                     ("risk", lambda: _risk_block(as_of, frames, live, frames15))):
        try:
            sc, why = fn()
        except Exception as e:
            sc, why = None, f"error: {e}"
        blocks.append({"name": name, "weight": W.get(name, 0), "score": (round(sc, 2) if sc is not None else None), "why": why})
    num = sum(b["weight"] * b["score"] for b in blocks if b["score"] is not None)
    den = sum(b["weight"] for b in blocks if b["score"] is not None)
    comp = int(round(100 * num / den)) if den else 0
    stance, color = _stance(comp)
    summary = next((b["why"].split(";")[0] for b in blocks if b["name"] == "trend" and b["score"] is not None), "")
    return {"as_of": as_of, "score": comp, "stance": stance, "color": color,
            "summary": summary, "live": live, "partial": partial, "blocks": blocks}


def _group_etf(kind: str, name: str) -> str | None:
    """The ETF that represents a group (themes have one via build_labels.THEME_ETFS; sectors don't)."""
    if kind == "theme":
        etfs = build_labels.THEME_ETFS.get(name)
        return etfs[0] if etfs else None
    return None


def _member_metric(sym, frames, frames15, live, as_of=None):
    """(up, above50, above200) for a group member: intraday (15m) when live, else the daily bar. `as_of`
    anchors the daily part to a settled session (no look-ahead on a replay). None if no data."""
    d = frames.get(sym)
    if d is None or not len(d):
        return None
    if as_of is not None:
        d = d.loc[:as_of]
        if not len(d):
            return None
    r = d.iloc[-1]
    s50, s200 = r.get("sma50"), r.get("sma200")
    if live and frames15 and sym in frames15:
        st = intraday._session_stats(frames15[sym])
        if st is None:
            return None
        last, up = st["last"], (st["d"] > 0)
    else:
        last = float(r["close"]); ch = r.get("chg_pct"); up = bool(pd.notna(ch) and ch > 0)
    return (1 if up else 0,
            1 if (pd.notna(s50) and last > s50) else 0,
            1 if (pd.notna(s200) and last > s200) else 0)


def _asof_last(d, as_of):
    """The as-of-sliced frame's last row (settled anchor for a replay), or the latest row if as_of is None."""
    if d is None or not len(d):
        return None
    if as_of is not None:
        d = d.loc[:as_of]
    return d.iloc[-1] if len(d) else None


def group_scores(frames: dict, live: bool = False, frames15: dict | None = None, as_of=None) -> list[dict]:
    """Score every sector & theme by member breadth (+ the group ETF's posture for themes) in ONE pass
    over the universe, bucketed by group (O(universe), reuses RAM frames). `as_of` anchors the RS /
    breadth to a settled session (no look-ahead on a replay). Returns a ranked leaderboard."""
    labels.load()
    sym_groups: dict[str, list[tuple[str, str]]] = {}
    for sym, sec in (labels._SECTOR or {}).items():
        if sec:
            sym_groups.setdefault(sym, []).append(("sector", sec))
    for t in labels.all_themes():
        for sym in labels.tickers_in_theme(t):
            sym_groups.setdefault(sym, []).append(("theme", t))
    srr = _asof_last(frames.get("SPY"), as_of)                # benchmark for relative strength
    spy_r1 = float(srr["ret_1m"]) if (srr is not None and pd.notna(srr.get("ret_1m"))) else 0.0
    spy_r3 = float(srr["ret_3m"]) if (srr is not None and pd.notna(srr.get("ret_3m"))) else 0.0
    acc: dict[tuple, list[float]] = {}                        # (kind,name) -> [n, up, a50, a200, sum_r1, sum_r3, n_r]
    for sym, groups in sym_groups.items():
        m = _member_metric(sym, frames, frames15, live, as_of)
        if m is None:
            continue
        rr = _asof_last(frames.get(sym), as_of)
        r1 = r3 = None
        if rr is not None and pd.notna(rr.get("ret_1m")) and pd.notna(rr.get("ret_3m")):
            r1, r3 = float(rr["ret_1m"]), float(rr["ret_3m"])
        for g in groups:
            a = acc.setdefault(g, [0, 0, 0, 0, 0.0, 0.0, 0])
            a[0] += 1; a[1] += m[0]; a[2] += m[1]; a[3] += m[2]
            if r1 is not None:
                a[4] += r1; a[5] += r3; a[6] += 1
    out = []
    for (kind, name), (n, up, a50, a200, sr1, sr3, nr) in acc.items():
        if n < config.GROUP_MIN_MEMBERS:
            continue
        pct_up, p50, p200 = up / n * 100, a50 / n * 100, a200 / n * 100
        bc = float(np.clip(((pct_up - 50) / 50 + (p50 - 50) / 50 + (p200 - 50) / 50) / 3, -1, 1))
        etf = _group_etf(kind, name)
        etf_sc = None
        if etf and etf in frames and len(frames[etf]) > 30:
            ef = frames[etf].loc[:as_of] if as_of is not None else frames[etf]
            st = mtf.tf_state(ef) if len(ef) > 30 else None
            if st is not None:
                etf_sc = _score_state(st) / 2.0
        score = (0.5 * etf_sc + 0.5 * bc) if etf_sc is not None else bc
        # structural relative strength: the group's mean member 1m/3m return minus SPY's (single pass, no
        # ratio frame). rs_trend = accelerating if the 1-month pace outruns the 3-month pace (RS improving).
        rs = rs3 = None
        rs_trend = "flat"
        if nr:
            rs = round(sr1 / nr - spy_r1, 1)
            rs3 = round(sr3 / nr - spy_r3, 1)
            rs_trend = "accel" if rs > rs3 / 3 + 1 else ("fade" if rs < rs3 / 3 - 1 else "flat")
        out.append({"kind": kind, "name": name, "n": n, "score": int(round(100 * score)),
                    "pct_up": round(pct_up), "pct_50": round(p50), "pct_200": round(p200),
                    "rs": rs, "rs_3m": rs3, "rs_trend": rs_trend,
                    "etf_sc": (round(etf_sc * 2, 2) if etf_sc is not None else None),   # ETF posture [-1,1] (divergence check)
                    "etf": etf, "why": f"{p50:.0f}% >50-MA, {pct_up:.0f}% up"
                    + (f", RS {rs:+.0f}% 1mo ({rs_trend})" if rs is not None else "")
                    + (f", ETF {etf} {etf_sc*2:+.1f}" if etf_sc is not None else "")})
    out.sort(key=lambda x: -x["score"])
    return out


def theme_board(frames: dict, as_of=None, bars5=None) -> list[dict]:
    """Theme-tracker style leaders board: every sector & theme x MEDIAN member return over
    Today / 1W / 1M / 3M / YTD (%, from the daily store — Today = the last settled session;
    the app's as-of drives replays). Median, not mean: one reverse-split artifact or a
    +2,000% micro-cap in a 20-name group destroys the mean. One O(universe) pass.

    `bars5` (optional {sym: 5m frame}, serve's live perf store): adds a LIVE median `open` column
    (each member's since-09:30 %, then the group median) so the merged Perf panel tracks the current
    session, not just the last settled one."""
    labels.load()
    sym_groups: dict[str, list[tuple[str, str]]] = {}
    for sym, sec in (labels._SECTOR or {}).items():
        if sec:
            sym_groups.setdefault(sym, []).append(("sector", sec))
    for t in labels.all_themes():
        for sym in labels.tickers_in_theme(t):
            sym_groups.setdefault(sym, []).append(("theme", t))
    WIN = (("d1", 1), ("w1", 5), ("m1", 21), ("m3", 63))
    acc: dict[tuple, dict[str, list[float]]] = {}
    for sym, groups in sym_groups.items():
        d = frames.get(sym)
        if d is None or not len(d):
            continue
        if as_of is not None:
            d = d.loc[:as_of]
        c = d["close"]
        if len(c) < 2 or not np.isfinite(c.iloc[-1]) or c.iloc[-1] <= 0:
            continue
        last = float(c.iloc[-1])
        rets = {}
        for k, nb in WIN:
            if len(c) > nb and np.isfinite(c.iloc[-1 - nb]) and c.iloc[-1 - nb] > 0:
                rets[k] = (last / float(c.iloc[-1 - nb]) - 1) * 100
        prev_yr = c[c.index.year < c.index[-1].year]      # YTD anchor = prior year's last close
        if len(prev_yr) and np.isfinite(prev_yr.iloc[-1]) and prev_yr.iloc[-1] > 0:
            rets["ytd"] = (last / float(prev_yr.iloc[-1]) - 1) * 100
        if bars5 is not None:                              # LIVE since-09:30-open % from the 5m store
            f5 = bars5.get(sym)
            if f5 is not None and len(f5) >= 2:
                d5 = f5[f5.index.normalize().values == f5.index.normalize().values[-1]]   # today's bars
                if len(d5):
                    op = float(d5["open"].iloc[0])
                    if op > 0:
                        rets["open"] = (float(d5["close"].iloc[-1]) / op - 1) * 100
        for g in groups:
            a = acc.setdefault(g, {})
            for k, v in rets.items():
                a.setdefault(k, []).append(v)
    rows = []
    for (kind, name), a in acc.items():
        n = max((len(v) for v in a.values()), default=0)
        if n < config.GROUP_MIN_MEMBERS:
            continue
        r = {"kind": kind, "name": name, "n": n, "etf": _group_etf(kind, name)}
        for k in ("open", "d1", "w1", "m1", "m3", "ytd"):
            r[k] = round(float(np.median(a[k])), 2) if a.get(k) else None
        rows.append(r)
    rows.sort(key=lambda r: -(r["w1"] if r["w1"] is not None else -1e9))
    return rows


def _today_et() -> str:
    try:
        from zoneinfo import ZoneInfo
        return dt.datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    except Exception:
        return dt.date.today().isoformat()


def _settled_asof(as_of: str | None, frames: dict) -> str | None:
    """The 'close' anchor should be the last SETTLED session — not today's partial daily bar while the
    market is open. Returns the newest daily-bar date strictly before today when live; else as_of."""
    if not _is_partial(as_of):
        return as_of
    spy = frames.get("SPY")
    if spy is None or not len(spy):
        return as_of
    today = _today_et()
    prior = [ts for ts in spy.index if ts.date().isoformat() < today]
    return prior[-1].date().isoformat() if prior else as_of


def _prior_session(as_of: str | None, frames: dict) -> str | None:
    """The last daily session STRICTLY before as_of — the point-in-time 'close' anchor while an intraday
    (live or replayed) read for as_of is in progress (so it never look-aheads into as_of's own EOD)."""
    spy = frames.get("SPY")
    if not as_of or spy is None or not len(spy):
        return _settled_asof(as_of, frames)
    prior = [ts for ts in spy.index if ts.date().isoformat() < as_of]
    return prior[-1].date().isoformat() if prior else as_of


def market_read(as_of: str | None, frames: dict, frames15: dict | None = None, daily_asof=None) -> dict:
    """Dual read: a settled 'close' score (always present) + a 'live' intraday score/internals when the 15m
    data has a COMPLETED bar for the session `as_of`. `daily_asof` is the settled anchor for ALL daily-derived
    blocks (vol/macro) — the caller passes the PRIOR session for an intraday/replay read, so at time T you
    only ever see yesterday's settled VIX/macro/closes + today's completed intraday. Never look-ahead."""
    anchor = daily_asof if daily_asof is not None else _settled_asof(as_of, frames)
    close = market_score(anchor, frames, live=False)
    live = internals = None
    if frames15 and intraday.latest_session(frames15) == (as_of or _today_et()):
        live = market_score(anchor, frames, live=True, frames15=frames15)
        ib = intraday.intraday_breadth(frames15, frames)
        internals = ib["internals"] if ib else None
    return {"as_of": as_of, "close": close, "live": live, "internals": internals}
