"""The six setup detectors. Each takes an indicator-enriched DataFrame and returns a
hit dict (or None) describing the setup as of the LAST bar in the frame.

Conventions:
- `trigger` = the price level that turns the setup into a trade.
- `level`   = the reference level the setup is built around (drawn on the chart).
- All detectors are pure: no liquidity filtering here (done in the dashboard).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

import config
import labels
import scanner_core
import geometry
import indicators
import emarider
import framecache
import patterns
import regime
import trendlab

# per-frame EMA-rider memo keyed by frame identity (NOT df.attrs — see framecache.py).
_EMA_RIDER_CACHE = framecache.FrameCache()


def _last(d, col):
    v = d[col].iloc[-1]
    return None if pd.isna(v) else float(v)


def _uptrend_dates(d, ctx) -> dict:
    """ISO date range of the trend segment that triggered the gate (for chart shading)."""
    if ctx is None or ctx.start < 0 or ctx.end < 0:
        return {}
    return {"uptrend_from": d.index[ctx.start].date().isoformat(),
            "uptrend_to": d.index[ctx.end].date().isoformat()}


def _trend_fields(ctx) -> dict:
    """Prior-trend stats carried on a hit for the quality score / table."""
    if ctx is None:
        return {}
    return {"prior_gain_pct": round(ctx.net_pct, 1), "prior_clarity": round(ctx.clarity, 2)}


def _trend_setup(d: pd.DataFrame, want: str) -> dict | None:
    """Classify the stock's CURRENT trend from trendlab's segment engine: fire when the latest
    (open-ended) segment is `want` ('up' or 'down'). Carries the segment's stats + bands."""
    if not config.REGIME_USE:
        return None
    segs, _ = trendlab.segment_chart(d)
    if not segs:
        return None
    s = segs[-1].ensure()                        # the current, still-forming segment (lazy metrics)
    if s.kind != want or s.start < 0:
        return None
    fld = "uptrend" if want == "up" else "downtrend"   # shade green for up, red for down
    return {
        "net_pct": round(s.net_pct, 1),
        "clarity": round(s.clarity, 2),
        "bars": int(s.bars),
        "ann_pct": round(max(min(s.ann_pct, 999.0), -999.0), 1),
        "is_pole": "pole" in s.tags,
        f"{fld}_from": d.index[s.start].date().isoformat(),
        f"{fld}_to": d.index[s.end].date().isoformat(),
    }


def detect_uptrend(d: pd.DataFrame) -> dict | None:
    """All stocks currently in an uptrend (latest trendlab segment = up)."""
    h = _trend_setup(d, "up")
    return {"setup": "uptrend", **h} if h is not None else None


def detect_downtrend(d: pd.DataFrame) -> dict | None:
    """All stocks currently in a downtrend (latest trendlab segment = down)."""
    h = _trend_setup(d, "down")
    return {"setup": "downtrend", **h} if h is not None else None


def _gap_type(d: pd.DataFrame) -> str:
    """TraderLion gap taxonomy on the CONTEXT the gap fires into: breakaway (clears a base),
    runaway (established, non-extended uptrend), exhaustion (late/extended, esp. not held)."""
    row = d.iloc[-1]
    prev = d.iloc[-2]
    ext50 = None
    if pd.notna(prev.get("sma50")) and pd.notna(prev.get("atr14")) and prev["atr14"] > 0:
        ext50 = (float(prev["close"]) - float(prev["sma50"])) / float(prev["atr14"])
    segs = trendlab.segment_chart(d)[0]
    prior = next((s for s in reversed(segs) if s.end < len(d) - 1 and s.kind != "transition"), None)
    if prior is not None and prior.kind == "range":
        r_hi = float(d["high"].iloc[prior.start:prior.end + 1].max())
        if float(row["open"]) > r_hi:
            return "breakaway"                     # gapped clear of a completed base
    if ext50 is not None and ext50 >= 3.0:
        return "exhaustion"                        # gap into a 3-ATR extension = late
    if prior is not None and prior.kind == "up" and (ext50 is None or ext50 < 3.0):
        return "runaway"
    return "common"


def detect_gapper(d: pd.DataFrame) -> dict | None:
    row = d.iloc[-1]
    if pd.isna(row["gap_pct"]) or pd.isna(row["rvol"]):
        return None
    # liquid-leader relaxation (Amir 2026-07-07): a name trading >=$250M/day can't 3x its already-large
    # average, so a hard gap on >=2x volume still qualifies (mirrors the EP detector's leader logic —
    # e.g. WULF +14% on 2.5x, ~$1.7B). Thinner names keep the full 3x bar so no illiquid junk leaks in.
    dv_day = float(row["close"]) * float(row["volume"])     # dollar volume TRADED on the gap day
    leader = dv_day >= config.GAP_LEADER_DVOL_M * 1e6 and float(row["close"]) >= 2.0   # liquid & not penny
    rvol_req = config.GAP_LEADER_RVOL if leader else config.GAP_MIN_RVOL
    if row["gap_pct"] >= config.GAP_MIN_PCT and row["rvol"] >= rvol_req:
        return {
            "setup": "gapper",
            "state": "breakout",                    # the gap day IS the trigger
            "gap_pct": round(float(row["gap_pct"]), 2),
            "rvol": round(float(row["rvol"]), 2),
            "held": bool(row["close"] >= row["open"]),
            "gap_type": _gap_type(d),               # breakaway / runaway / exhaustion / common
            "level": float(d["close"].iloc[-2]),   # prior close = gap fill level
            "trigger": float(row["high"]),
        }
    return None


def _is_ep_event(r, base_dvol: float | None = None) -> str | None:
    """Episodic-pivot event test for one bar: catalyst gap, the objective EP-9M volume filter, or
    a no-gap big-move day (Bonde doesn't require a gap — intraday surges like BFLY 2025-09-18
    count). Liquidity guard = event-day DOLLAR volume rather than a price floor, so sub-$3 names
    with institutional flow qualify; rebalance/quad-witching volume spikes still die on the move
    gates (flat price on huge volume is NOT an EP — Dan Tips). `base_dvol` = median dollar volume
    over the prior 60 bars: dead shells that only trade on the event day are not EPs."""
    if pd.isna(r["rvol"]):
        return None
    if pd.notna(r["chg_pct"]) and r["chg_pct"] < -5:
        return None        # an "event" closing 5%+ BELOW prior close is a failed day or a
                           # reverse-split data artifact (INHD 2025-05-12, Amir verdict), not an EP
    if pd.notna(r["gap_pct"]) and pd.notna(r["chg_pct"]) \
            and r["gap_pct"] >= config.EP_GAP_RETAIN_MIN \
            and r["chg_pct"] < r["gap_pct"] * config.EP_GAP_RETAIN:
        return None        # big gap that CLOSED below half the gap = fade/artifact (SDOT, BULL)
    if base_dvol is not None and base_dvol < config.EP_MIN_BASE_DVOL_M * 1e6:
        return None        # baseline liquidity: the name must actually TRADE before the event
    # Day 1 close-range position quality gate: must close in upper 35% of daily range (filters out Day 1 distribution fades)
    cpos = r.get("close_pos")
    if cpos is None or pd.isna(cpos):
        hi = float(r.get("high", 0))
        lo = float(r.get("low", 0))
        rng = hi - lo
        cpos = (float(r["close"]) - lo) / rng if rng > 1e-4 else 1.0
    if float(cpos) < config.EP_MIN_CLOSE_POS:
        return None
    close = float(r["close"])
    liquid = (close >= config.EP_MIN_PRICE
              and close * float(r["volume"]) >= config.EP_MIN_DVOL_M * 1e6)
    # liquid leaders (avg $vol >= EP_LEADER_DVOL_B): a mega-cap trading 25M+ shares/day cannot
    # 3x its average — MU's +13.8% earnings gap printed rvol 2.6 — so the bar drops to LEADER_RVOL
    leader = (pd.notna(r.get("dollar_vol"))
              and float(r["dollar_vol"]) >= config.EP_LEADER_DVOL_B * 1e9)
    rvol_req = config.EP_LEADER_RVOL if leader else config.EP_MIN_RVOL
    vol_mult = config.EP_LEADER_RVOL if leader else config.EP_VOL_MULT
    if pd.notna(r["gap_pct"]) and r["gap_pct"] >= config.EP_MIN_GAP and r["rvol"] >= rvol_req:
        return "classic"
    if pd.notna(r["chg_pct"]) and liquid:
        if r["chg_pct"] >= config.EP9M_MIN_CHG and float(r["volume"]) >= config.EP_MIN_SHARES \
                and r["rvol"] >= vol_mult:
            return "ep9m"
        if r["chg_pct"] >= config.EP_BIGMOVE_CHG and r["rvol"] >= rvol_req:
            return "bigmove"
    return None


_EP_MARK = {"classic": "EP classic", "ep9m": "EP 9M", "bigmove": "EP big move"}


def detect_episodic_pivot(d: pd.DataFrame) -> dict | None:
    """Episodic Pivot (Bonde playbook). EVENT day (state 'breakout'): a catalyst gap
    (EP_MIN_GAP% + EP_MIN_RVOL) or the EP-9M objective filter (>= EP_MIN_SHARES shares at
    rvol >= EP_VOL_MULT, price >= EP_MIN_PRICE, up day). DELAYED-REACTION watch (state 'watch'):
    for EP_WATCH_BARS after the event while price holds the event-day low — trigger = the nearest
    unbroken resistance from the breakout stack, level = the event-day low. Evidence fields:
    neglect (6-month return INTO the event), held (closed above open on event day)."""
    n = len(d)
    if n < 30:
        return None
    med60 = (d["close"] * d["volume"]).rolling(60, min_periods=10).median().shift(1)
    row = d.iloc[-1]
    sub = _is_ep_event(row, med60.iloc[-1] if pd.notna(med60.iloc[-1]) else None)
    if sub:
        # Anti-clustering debounce: if an EP already fired within the past EP_DEBOUNCE_BARS sessions,
        # today is an intra-run extension of the existing episode rather than a fresh Day 1 ignition.
        lo_deb = max(0, n - 1 - config.EP_DEBOUNCE_BARS)
        prior_ep = any(_is_ep_event(d.iloc[j], med60.iloc[j] if pd.notna(med60.iloc[j]) else None)
                       for j in range(n - 2, lo_deb - 1, -1))
        if not prior_ep:
            neglect = d["ret_6m"].iloc[-2] if n > 2 and pd.notna(d["ret_6m"].iloc[-2]) else None
            return {
                "setup": "episodic_pivot", "state": "breakout", "ep_subtype": sub,
                "marks": [{"date": _dt_str(d, n - 1), "text": _EP_MARK.get(sub, f"EP {sub}"), "pos": "below"}],
                "gap_pct": round(float(row["gap_pct"]), 2) if pd.notna(row["gap_pct"]) else None,
                "rvol": round(float(row["rvol"]), 2) if pd.notna(row["rvol"]) else None,
                "shares_m": round(float(row["volume"]) / 1e6, 1),
                "neglect_6m": round(float(neglect), 1) if neglect is not None else None,
                "held": bool(row["close"] >= row["open"]),
                "level": float(row["low"]),              # event-day low = the line in the sand
                "trigger": float(row["high"]),
            }
    # delayed reaction: find the most recent fresh ignition within the watch window, still above its low
    lo_scan = max(1, n - 1 - config.EP_WATCH_BARS)
    ev = None
    for i in range(lo_scan, n - 1):
        s = _is_ep_event(d.iloc[i], med60.iloc[i] if pd.notna(med60.iloc[i]) else None)
        if s:
            lo_deb = max(0, i - config.EP_DEBOUNCE_BARS)
            is_fresh = not any(_is_ep_event(d.iloc[j], med60.iloc[j] if pd.notna(med60.iloc[j]) else None)
                               for j in range(lo_deb, i))
            if is_fresh:
                ev = (i, s)
    if ev is None:
        return None
    ev_i, sub = ev
    age = int(n - 1 - ev_i)
    ev_open = float(d["open"].iloc[ev_i])
    ev_high = float(d["high"].iloc[ev_i])
    ev_low = float(d["low"].iloc[ev_i])
    ev_close = float(d["close"].iloc[ev_i])
    close = float(row["close"])

    # Hard invalidation: event-day low breached
    if close < ev_low:
        return None                                  # event-day low lost -> EP dead

    # Weak event day: closed red
    if ev_close < ev_open:
        return None                                  # event day faded (not held) -> weak EP, skip watch

    # 48-Hour Retracement Rule (institutional absorption check):
    # Surrendering >50% of Day 1 candle body within 48h flags failed absorption (78.9% historical trap rate)
    d1_body = ev_close - ev_open
    half_retrace = ev_close - (0.5 * max(0.01, d1_body))
    for j in range(ev_i + 1, min(n, ev_i + 3)):
        if float(d["close"].iloc[j]) < half_retrace:
            return None

    # Fade check: gave back >5% of event-day close
    if close < ev_close * 0.95:
        return None                                  # gave back the event-day close -> stale, off watch

    rvol_arr = d["rvol"].to_numpy(float).copy()
    _, proj_frac = _project_last_volume(d)
    if proj_frac is not None and rvol_arr.size and pd.notna(rvol_arr[-1]):
        rvol_arr[-1] = rvol_arr[-1] / proj_frac
    vol_ok_arr = rvol_arr >= config.EP_BREAKOUT_RVOL
    st = _breakout_stack(d, ev_i, n - 1, vol_ok_arr, config.BREAKOUT_STACK_DOWNSWING)
    neglect = d["ret_6m"].iloc[ev_i - 1] if ev_i >= 1 and pd.notna(d["ret_6m"].iloc[ev_i - 1]) else None

    # Lifecycle state machine:
    # 1. Breakout: broke above resistance stack with volume confirmation
    # 2. Phase 2 Momentum (Days 1 to 5): holding above Day 1 close or in upper 25% of Day 1 range -> "riding"
    # 3. Phase 3 Digestion (Days 6 to 21): orderly base/flag above Day 1 low -> "building"
    if st["broke_today"]:
        state = "breakout"
    elif age <= 5 and (close >= ev_close or close >= (ev_close - 0.25 * max(0.01, ev_high - ev_low))):
        state = "riding"
    else:
        state = "building"

    return {
        "setup": "episodic_pivot",
        "state": state,
        "ep_subtype": sub, "ep_age": age,
        "marks": [{"date": _dt_str(d, ev_i), "text": _EP_MARK.get(sub, f"EP {sub}"), "pos": "below"}],
        "gap_pct": round(float(d["gap_pct"].iloc[ev_i]), 2) if pd.notna(d["gap_pct"].iloc[ev_i]) else None,
        "rvol": round(float(row["rvol"]), 2) if pd.notna(row["rvol"]) else None,
        "neglect_6m": round(float(neglect), 1) if neglect is not None else None,
        "held": bool(ev_close >= ev_open),
        "provisional": bool(st["broke_today"] and proj_frac is not None),
        "level": round(ev_low, 2),
        "trigger": st["trigger"],
    }


def _is_hvc_bar(row) -> bool:
    """High-volume-close bar test: heavy volume, closes near the high, up day."""
    if pd.isna(row["rvol"]) or pd.isna(row["close_pos"]):
        return False
    return (row["rvol"] >= config.HVC_MIN_VOL_MULT
            and row["close_pos"] >= config.HVC_CLOSE_RANGE_POS
            and row["chg_pct"] >= config.HVC_MIN_CHANGE_PCT)


def detect_hvc(d: pd.DataFrame) -> dict | None:
    row = d.iloc[-1]
    if not _is_hvc_bar(row):
        return None
    ctx = None
    if config.HVC_REQUIRE_UPTREND and config.REGIME_USE:   # require a real prior uptrend
        ctx = regime.current_regime(d)
        if not ctx.strong_uptrend:
            return None
    return {
        "setup": "hvc",
        "state": "breakout",                   # the high-volume close IS the trigger
        "rvol": round(float(row["rvol"]), 2),
        "chg_pct": round(float(row["chg_pct"]), 2),
        "close_pos": round(float(row["close_pos"]), 2),
        "level": float(row["close"]),          # HVC close = future support ref
        "trigger": float(row["high"]),
        **_trend_fields(ctx),
        **_uptrend_dates(d, ctx),
    }


def detect_flat_base(d: pd.DataFrame) -> dict | None:
    """Flat-topped consolidation: a horizontal resistance (a window high) tested by >=
    FB_MIN_TOUCHES separate bars — so a lone spike never counts — with price coiling below
    it and contained depth, after a prior advance OR a 200-SMA reclaim / bottoming turn.
    Base shape and prior trend use SEPARATE windows. Breakout level = the resistance top.
    (Touches are counted from bar highs, not trendlab pivots, so tight/young bases register.)"""
    n = len(d)
    if n < config.FB_MIN_BARS + 5:
        return None
    close = float(d["close"].iloc[-1])
    h = d["high"].to_numpy(float)
    l = d["low"].to_numpy(float)
    c = d["close"].to_numpy(float)
    hi52 = _last(d, "hi52") or 0.0          # the base must sit near the 52-week high
    near_high = hi52 * (1 - config.FB_NEAR_HIGH / 100)
    tol = config.FB_TOP_TOL / 100
    lo_dist, hi_dist = 1 - config.FB_MAX_DIST_FROM_HIGH / 100, 1 + config.FB_BREAKOUT_TOL / 100
    cand = None
    last = n - 1
    # the base STRUCTURE ends YESTERDAY: today is the TEST bar against the base top. With today
    # included, top = max(high) covered today's own high, so close > top was IMPOSSIBLE — the
    # 'breakout' state never fired once in the setup's history (found in Amir's verification).
    for start in range(max(0, last - config.FB_MAX_BARS), last - config.FB_MIN_BARS + 1, 2):
        hs, ls_, cs = h[start:last], l[start:last], c[start:last]
        top = float(hs.max())
        if top <= 0 or cs[0] <= 0 or top < near_high:
            continue
        if abs(float(cs[-1]) / float(cs[0]) - 1) * 100 > config.FB_MAX_DRIFT:   # base itself flat
            continue
        near = hs >= top * (1 - tol)                           # bars tagging the resistance
        touches = int(near.sum())                              # lone spike -> 1 touch -> rejected
        if touches < config.FB_MIN_TOUCHES:
            continue
        ni = np.nonzero(near)[0]                               # resistance must hold ACROSS the base
        if ni[-1] - ni[0] < config.FB_MIN_TOUCH_SPAN * (last - 1 - start):   # (not bunched at one end)
            continue
        base_low = float(ls_.min())
        if base_low <= 0 or (top - base_low) / top * 100 > config.FB_MAX_DEPTH_PCT:
            continue
        if not (top * lo_dist <= close <= top * hi_dist):     # coiling under the top / breakout day
            continue
        key = (touches, last - start)         # most-tested resistance, then the longer base
        if cand is None or key > cand[0]:
            cand = (key, start, top, base_low, touches)
    if cand is None:
        return None
    _, start, top, base_low, touches = cand

    # prior context on a SEPARATE (earlier) window: advance, OR (loose) 200-SMA reclaim / bottoming
    prior_gain = None
    if config.FB_PRIOR_MODE != "off":
        ps = start - config.FB_PRIOR_LOOKBACK
        if ps >= 0:
            prior_gain = (float(d["close"].iloc[start]) / float(d["close"].iloc[ps]) - 1) * 100
        # backdrop: a strong prior advance (any MA), or — loose — simply above the 200-SMA
        # (flatness/selectivity comes from the drift + touch gates, not a rigid prior-gain window)
        strong = prior_gain is not None and prior_gain >= config.FB_PRIOR_GAIN_PCT
        sma200 = d["sma200"].iloc[-1]
        above200 = pd.notna(sma200) and close > sma200
        if not (strong or (config.FB_PRIOR_MODE == "loose" and above200)):
            return None

    # Stage-2 trend template: above a rising 50 > 200 SMA stack (a true leader, not a laggard)
    if config.FB_REQUIRE_STAGE2:
        s50, s200 = d["sma50"].iloc[-1], d["sma200"].iloc[-1]
        s200_prev = d["sma200"].iloc[-21] if n > 21 else np.nan
        if not (pd.notna(s50) and pd.notna(s200) and pd.notna(s200_prev)
                and close > s50 > s200 and s200 > s200_prev):
            return None

    # volume must dry up across the base (institutions accumulating quietly before the move)
    if config.FB_VOL_CONTRACTION:
        bv = d["volume"].to_numpy(float)[start:]
        half = len(bv) // 2
        if half >= 2 and bv[half:].mean() >= config.FB_VOL_CONTRACTION * bv[:half].mean():
            return None

    # breakout needs VOLUME (spec: 40-50% above average) — a low-volume poke above the top is
    # flagged but not trusted; today's partial bar is projected to a full-session estimate
    state = "building"
    if close > top:
        va = _last(d, "vol_avg50")
        pv, _ = _project_last_volume(d)
        rvol_break = (pv[-1] / va) if va else None
        state = ("breakout" if rvol_break is None or rvol_break >= config.FB_BREAKOUT_RVOL
                 else "breakout_lowvol")
    hit = {
        "setup": "flat_base",
        "state": state,
        "base_days": int(base_bars := n - 1 - start),
        "depth_pct": round((top - base_low) / top * 100, 1),
        "touches": touches,
        "dist_to_trigger_pct": round((top / close - 1) * 100, 2),
        "cons_bars": base_bars,
        "level": round(base_low, 2),
        "trigger": round(top, 2),
    }
    if prior_gain is not None and prior_gain >= config.FB_PRIOR_GAIN_PCT:
        hit["prior_gain_pct"] = round(prior_gain, 1)
        if start - config.FB_PRIOR_LOOKBACK >= 0:   # green-shade the prior advance on the chart
            hit["uptrend_from"] = d.index[start - config.FB_PRIOR_LOOKBACK].date().isoformat()
            hit["uptrend_to"] = d.index[start].date().isoformat()
    return hit


# Typical US-equity intraday CUMULATIVE volume profile: fraction of a full 9:30-16:00 ET session's volume
# that has traded by N minutes into the session. U-shaped (heavy at the open & into the close, light
# midday), so e.g. ~13% is already in by 10:00 (30m) — projecting a partial bar by this curve is far
# tamer and more accurate than linear time (which would say 30/390 = 8%).
_VOL_CURVE = [(0, 0.0), (15, 0.07), (30, 0.13), (60, 0.21), (90, 0.28), (120, 0.34), (150, 0.40),
              (180, 0.45), (210, 0.51), (240, 0.57), (270, 0.63), (300, 0.69), (330, 0.76),
              (360, 0.85), (375, 0.92), (390, 1.0)]


def _session_vol_fraction(last_ts) -> float | None:
    """Fraction of a full session's volume expected to have traded by now — but ONLY when `last_ts` (the
    latest daily bar) is TODAY in ET and the market is currently open (a still-forming partial bar).
    None otherwise (a settled historical bar, pre-market, or after the close) so nothing is projected."""
    if not config.BREAKOUT_VOL_PROJECT:
        return None
    try:
        from zoneinfo import ZoneInfo
        now = dt.datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return None
    if now.weekday() >= 5 or pd.Timestamp(last_ts).date() != now.date():
        return None
    mins = (now - now.replace(hour=9, minute=30, second=0, microsecond=0)).total_seconds() / 60.0
    if mins < 5 or mins >= 390:          # too early to tell / session effectively over (bar ~complete)
        return None
    for (m0, f0), (m1, f1) in zip(_VOL_CURVE, _VOL_CURVE[1:]):
        if mins <= m1:
            return f0 + (f1 - f0) * (mins - m0) / (m1 - m0)
    return None


def _project_last_volume(d: pd.DataFrame) -> tuple[np.ndarray, float | None]:
    """(volume array, fraction|None). When the latest bar is TODAY's partial session, its volume is scaled
    up to a full-session estimate (÷ the intraday-curve fraction) so an intraday breakout's volume is
    comparable to the daily average; fraction is None (and the array unchanged) when the bar is settled."""
    v = d["volume"].to_numpy(float).copy()
    frac = _session_vol_fraction(d.index[-1])
    if frac and v[-1] > 0:
        v[-1] = v[-1] / frac
        return v, frac
    return v, None


def _breakout_stack(d: pd.DataFrame, hi_pos: int, end_pos: int, vol_ok: np.ndarray, dbars: int) -> dict:
    """The consolidation's breakout-level STACK over (hi_pos, end_pos], shared by the QM and HTF breakouts.

    Levels (a LIFO of resistances, nearest on top): the POLE high (bottom, first in) + each confirmed
    swing-pivot high (trendlab.swing_highs — causal, ~1-bar lag) + a synthetic 'previous bar high' when a
    downswing runs >= dbars bars (the stair-step run rule high[i] < high[i-1] + atr*STAIRSTEP_ATR_FRACTION;
    consecutive synthetics REPLACE each other). A level POPS only on a volume-confirmed close above it
    (vol_ok[i]); one bar may pop several. After the pole high pops, a close back below it (in range) resets
    the stack to [pole_high]. Returns the current top (nearest resistance), whether end_pos popped a level,
    and counts. Causal / no look-ahead."""
    high = d["high"].to_numpy(float); close = d["close"].to_numpy(float); atr = d["atr14"].to_numpy(float)
    pole_high = float(high[hi_pos])
    piv = {i for i, _ in trendlab.swing_highs(d, len(d) - (hi_pos + 1)) if hi_pos + 1 <= i <= end_pos}
    frac = config.STAIRSTEP_ATR_FRACTION
    stack: list[tuple[float, str]] = [(pole_high, "pole")]
    dr = 0; n_broken = 0; last_break = -1
    for i in range(hi_pos + 1, end_pos + 1):
        if i in piv:                                            # a new confirmed pivot-high resistance
            stack.append((float(high[i]), "pivot")); dr = 0
        else:
            a = atr[i]
            dr = dr + 1 if (np.isfinite(a) and high[i] < high[i - 1] + a * frac) else 0
            if dr >= dbars:                                     # drawn-out decline w/ no pivot
                lvl = float(high[i - 1])                        # trailing "previous bar's high"
                if stack[-1][1] == "down":
                    stack[-1] = (lvl, "down")                   # replace the prior synthetic
                else:
                    stack.append((lvl, "down"))
        if vol_ok[i]:                                           # volume-confirmed close pops level(s)
            while len(stack) > 1 and close[i] > stack[-1][0]:
                stack.pop(); n_broken += 1; last_break = i
            if len(stack) == 1 and close[i] > stack[-1][0]:     # final break of the pole high
                stack.pop(); n_broken += 1; last_break = i
        if not stack and close[i] < pole_high:                 # fell back into range -> pole high returns
            stack = [(pole_high, "pole")]
    levels = [lv for lv, _ in stack] or [pole_high]
    # trigger = nearest un-broken resistance ABOVE the latest close. A low-volume close that drifts above a
    # level advances the trigger to the next level up (a previous consolidation high, ultimately the flag
    # top) instead of stranding it below price; the drifted-past level stays in the stack and returns as the
    # trigger if price closes back beneath it. Only a VOLUME-confirmed close permanently pops a level. When
    # price sits above every level, the trigger is the top of the flag (pole high) — the terminal breakout.
    close_last = float(close[end_pos])
    above = sorted(lv for lv in levels if lv > close_last)
    trigger = above[0] if above else max(levels)
    return {"trigger": trigger, "broke_today": last_break == end_pos,
            "n_broken": int(n_broken), "n_levels": len(levels), "levels": levels}


def detect_high_tight_flag(d: pd.DataFrame) -> dict | None:
    """High Tight Flag (Traderlion / Deepvue). An explosive POLE (>= HTF_MIN_GAIN_PCT, ~90-120%+) over
    <= HTF_POLE_MAX_BARS bars (~4-8 weeks), then a shallow TIGHT flag that pulls back at most
    HTF_MAX_PULLBACK_PCT off the pole high (tightness near the highs, NO specific shape), and a breakout
    on a volume surge. Pole 'cleanliness' is surfaced as `pole_efficiency` (directional efficiency) and
    only gated when HTF_MIN_EFFICIENCY>0. Strength criteria (MA-stack, 52w-high proximity, ADR) are
    computed and surfaced; each is gated by its own config toggle so we can start loose and tighten from
    results. RS leadership is the rs_rank dashboard filter, not gated here. The trendlab pole gate is
    OFF unless HTF_REQUIRE_REGIME (the official definition doesn't require an uptrend on the pole)."""
    n = len(d)
    if n < config.HTF_LOOKBACK + 10:
        return None
    # the pole high is found EXCLUDING today: on a new-high breakout day the finder used to point
    # at the breakout bar itself -> flag_bars=0 -> None (YPF 2024-12-11, Amir verification)
    look = d.iloc[-(config.HTF_LOOKBACK + 1):-1]
    hi_pos = int(d.index.get_loc(look["high"].idxmax()))
    hi_p = float(d["high"].iloc[hi_pos])
    flag_bars = n - 1 - hi_pos
    if not (config.HTF_FLAG_MIN_BARS <= flag_bars <= config.HTF_FLAG_MAX_BARS):
        return None
    # 1. pole: the launch low within HTF_POLE_MAX_BARS before the high (same finder as QM)
    lowA = d["low"].to_numpy(float)
    mstart = max(0, hi_pos - config.HTF_POLE_MAX_BARS)
    mlow_pos = mstart + int(np.argmin(lowA[mstart:hi_pos + 1]))
    pole_low = float(lowA[mlow_pos])
    if pole_low <= 0:
        return None
    pole_bars = hi_pos - mlow_pos
    gain = (hi_p / pole_low - 1.0) * 100
    if gain < config.HTF_MIN_GAIN_PCT or pole_bars > config.HTF_POLE_MAX_BARS:
        return None
    # 2. pole efficiency = net move / path length over the pole (1.0 = perfectly straight). Surfaced as
    #    evidence; gated only if HTF_MIN_EFFICIENCY>0 (so we calibrate a cutoff from real values).
    closeA = d["close"].to_numpy(float)
    path = float(np.abs(np.diff(closeA[mlow_pos:hi_pos + 1])).sum())
    pole_eff = round((hi_p - pole_low) / path, 2) if path > 0 else None
    if config.HTF_MIN_EFFICIENCY > 0 and (pole_eff is None or pole_eff < config.HTF_MIN_EFFICIENCY):
        return None
    # 3. shallow flag (official depth, no shape req): pullback off the pole HIGH within HTF_MAX_PULLBACK_PCT
    flag_low = float(lowA[hi_pos:].min())
    pullback = (1.0 - flag_low / hi_p) * 100
    if pullback > config.HTF_MAX_PULLBACK_PCT:
        return None
    pullback = round(pullback, 1)
    # 4. strength criteria — compute all; gate each by its own toggle (start loose)
    close = float(closeA[n - 1])
    sma50, sma200 = _last(d, "sma50"), _last(d, "sma200")
    rb = config.HTF_RISING_MA_BARS
    s50p = float(d["sma50"].iloc[-1 - rb]) if n > rb and pd.notna(d["sma50"].iloc[-1 - rb]) else None
    s200p = float(d["sma200"].iloc[-1 - rb]) if n > rb and pd.notna(d["sma200"].iloc[-1 - rb]) else None
    ma_stack = bool(sma50 and sma200 and close > sma50 > sma200
                    and s50p is not None and sma50 > s50p and s200p is not None and sma200 > s200p)
    if config.HTF_REQUIRE_MA_STACK and not ma_stack:
        return None
    off_hi = _last(d, "off_hi52_pct")
    if config.HTF_NEAR_HIGH > 0 and (off_hi is None or off_hi < -config.HTF_NEAR_HIGH):
        return None
    adrA = d["adr_pct"].to_numpy(float)
    pole_adr = float(np.nanmean(adrA[mlow_pos:hi_pos + 1])) if hi_pos > mlow_pos else float("nan")
    if config.HTF_MIN_ADR > 0 and not (pole_adr >= config.HTF_MIN_ADR):
        return None
    # volume dry-up + flag volatility contraction (evidence). Use RAW per-bar range % (not the 20-bar
    # rolling adr_pct, whose window blends pole+flag bars) so contraction<1 = flag tighter than the pole.
    volA = d["volume"].to_numpy(float)
    flag_vol = float(volA[hi_pos + 1:].mean()) if flag_bars >= 1 else float("inf")
    pole_vol = float(volA[mlow_pos:hi_pos + 1].mean())
    rng = (d["high"].to_numpy(float) - lowA) / np.where(closeA > 0, closeA, np.nan) * 100
    pole_rng = float(np.nanmean(rng[mlow_pos:hi_pos + 1])) if hi_pos > mlow_pos else float("nan")
    flag_rng = float(np.nanmean(rng[hi_pos + 1:])) if flag_bars >= 1 else None
    contraction = round(flag_rng / pole_rng, 2) if (flag_rng and pole_rng and pole_rng > 0) else None
    # 5. optional trendlab pole gate (OFF unless HTF_REQUIRE_REGIME); ctx still computed for evidence
    ctx = None
    if config.REGIME_USE:
        ctx = regime.prior_trend_context(d, hi_pos)
        if config.HTF_REQUIRE_REGIME and not (ctx.is_uptrend and ctx.is_pole):
            return None
    # 6. breakout-level stack (shared with QM): the trigger is the top of the stack (nearest resistance).
    #    A volume-confirmed close that pops the top = 'breakout' at any length >= HTF_FLAG_MIN_BARS (3);
    #    otherwise the base is 'too_short' (< the optimal window) or 'optimal' (in it, highlighted).
    rvol = _last(d, "rvol")
    rvol_arr = d["rvol"].to_numpy(float).copy()
    _, proj_frac = _project_last_volume(d)          # project TODAY's partial bar (if live)
    if proj_frac is not None and rvol_arr.size and pd.notna(rvol_arr[-1]):
        rvol_arr[-1] = rvol_arr[-1] / proj_frac     # rvol = vol/avg50 -> scales with the volume projection
        rvol = rvol_arr[-1]
    vol_ok_arr = rvol_arr >= config.HTF_BREAKOUT_RVOL
    st = _breakout_stack(d, hi_pos, n - 1, vol_ok_arr, config.BREAKOUT_STACK_DOWNSWING)
    trigger = st["trigger"]
    provisional = bool(st["broke_today"] and proj_frac is not None)     # today's break used projected volume
    if st["broke_today"]:
        state = "breakout"                  # a level popped today on volume
    elif float(closeA[-1]) > trigger:
        state = "breakout_lowvol"           # closed above the trigger WITHOUT the volume surge
    elif flag_bars < config.HTF_OPTIMAL_MIN_BARS:
        state = "too_short"
    else:
        state = "optimal"                   # coiling under resistance in the optimal window
    pat = patterns.classify_consolidation(d, hi_pos + 1, n - 1)
    idx = d.index
    iso = lambda j: idx[j].date().isoformat()
    return {
        "setup": "high_tight_flag",
        "state": state,
        "provisional": provisional,          # breakout confirmed on TODAY's projected (partial) volume
        "vol_proj": round(proj_frac, 2) if proj_frac is not None else None,   # session fraction used
        "pole_gain_pct": round(gain, 1),
        "pole_bars": int(pole_bars),
        "pole_efficiency": pole_eff,
        "pullback_pct": pullback,
        "flag_bars": int(flag_bars),
        "cons_bars": int(flag_bars),
        "adr_pct": round(pole_adr, 1) if pd.notna(pole_adr) else None,
        "off_hi52_pct": round(off_hi, 1) if off_hi is not None else None,
        "ma_stack": ma_stack,
        "vol_dryup": bool(flag_vol < pole_vol),
        "contraction": contraction,
        "vol_x": round(rvol, 2) if rvol is not None else None,
        "breakouts": st["n_broken"],
        "levels": st["n_levels"],
        "pattern": pat["pattern"],
        "dist_to_trigger_pct": round((trigger / close - 1) * 100, 2),
        "level": round(flag_low, 2),
        "trigger": round(trigger, 2),
        # chart geometry: pole + consolidation shading, and the stack resistance levels (horizontal)
        "htf_pole": [iso(mlow_pos), iso(hi_pos)],
        "htf_cons": [iso(hi_pos), iso(n - 1)],
        "stack_levels": [round(x, 2) for x in st["levels"]],
        **_trend_fields(ctx),
        **_uptrend_dates(d, ctx),
    }


def detect_higher_low_ma(d: pd.DataFrame) -> dict | None:
    """Pullback that UNDERCUTS a rising MA (EMA10 / EMA20 / SMA50) inside a long-term uptrend
    with ascending swing lows. Two states: 'building' = still under the MA (not reclaimed,
    potential entry on the bounce), 'breakout' = closed back above the MA on the latest candle."""
    if len(d) < 210:
        return None
    row = d.iloc[-1]
    sma50, sma200 = _last(d, "sma50"), _last(d, "sma200")
    close, lo = float(row["close"]), float(row["low"])
    # robust uptrend backdrop (a pullback distorts segmentation, so use the MA stack)
    if sma50 is None or sma200 is None or not (close > sma200 and sma50 > sma200):
        return None
    lows = trendlab.swing_lows(d, 60)               # ascending higher-low structure
    if len(lows) < 2:
        return None
    last_piv, prev_piv = float(lows[-1][1]), float(lows[-2][1])
    if last_piv <= prev_piv:
        return None
    # require an undercut of a rising MA; report the deepest pierced (50 > 20 > 10)
    mas = [("50", sma50), ("20", _last(d, "ema20")), ("10", _last(d, "ema10"))]
    pierced = [(nm, v) for nm, v in mas if v is not None and lo < v]
    if not pierced:
        return None
    ma_n, ma_val = pierced[0]
    return {
        "setup": "higher_low_ma",
        "state": "breakout" if close > ma_val else "building",   # reclaimed the MA vs still undercut
        "ma": ma_n,
        "higher_low": round(last_piv, 2),
        "prev_low": round(prev_piv, 2),
        "undercut_pct": round((1 - lo / ma_val) * 100, 2),
        "cons_bars": int(len(d) - 1 - lows[-1][0]),   # pullback length (higher-low pivot -> now)
        "level": round(ma_val, 2),       # the MA being tested (support / stop ref)
        "trigger": float(row["high"]),   # entry = break of this bar's high
    }


def detect_undercut_rally(d: pd.DataFrame) -> dict | None:
    if len(d) < config.UR_LEVEL_LOOKBACK + 5:
        return None
    row = d.iloc[-1]
    # confirmed swing lows as support levels, at least UR_LEVEL_MIN_AGE bars old
    max_idx = len(d) - 1 - config.UR_LEVEL_MIN_AGE
    lows = [p for i, p in trendlab.swing_lows(d, config.UR_LEVEL_LOOKBACK) if i <= max_idx]
    if not lows:
        return None
    lo, close = float(row["low"]), float(row["close"])
    # support level(s) the low pierced today — a shakeout, not a crash (within UR_MAX_UNDERCUT_PCT).
    # We no longer require a reclaim: 'building' = still below, 'breakout' = closed back above.
    cands = [x for x in lows if lo < x and (1 - lo / x) * 100 <= config.UR_MAX_UNDERCUT_PCT]
    if not cands:
        return None
    level = max(cands)                  # the highest support pierced (the level being tested)
    undercut_pct = (1.0 - lo / level) * 100
    # long-term uptrend backdrop (shakeout within an uptrend, not a falling knife)
    if config.REGIME_USE:
        sma50, sma200 = d["sma50"].iloc[-1], d["sma200"].iloc[-1]
        if pd.isna(sma200) or not (close > sma200 and sma50 > sma200):
            return None
    return {
        "setup": "undercut_rally",
        "state": "breakout" if close >= level else "building",   # reclaimed the level vs still below it
        "undercut_pct": round(undercut_pct, 2),
        "reclaim_pct": round((close / level - 1) * 100, 2),
        "rvol": round(float(row["rvol"]), 2) if not pd.isna(row["rvol"]) else None,
        "level": level,
        "trigger": float(row["high"]),
    }


def detect_delayed_hvc(d: pd.DataFrame) -> dict | None:
    """Gapped up in the last few days, consolidated holding the gap, then an HVC today.
    A delayed continuation of the gap's energy after a tight pause."""
    n = len(d)
    if n < config.DHVC_LOOKBACK + 5:
        return None
    row = d.iloc[-1]
    if not _is_hvc_bar(row):                          # today must be the HVC breakout
        return None
    gap = d["gap_pct"].to_numpy(float)
    rvol = d["rvol"].to_numpy(float)
    close = float(row["close"])
    lo_g = max(1, n - 1 - config.DHVC_LOOKBACK)
    hi_g = n - 2 - config.DHVC_MIN_BASE_BARS         # leave >= MIN_BASE_BARS base bars + today
    for g in range(hi_g, lo_g - 1, -1):              # most recent qualifying gap first
        if not (pd.notna(gap[g]) and gap[g] >= config.DHVC_GAP_MIN_PCT):
            continue
        if pd.notna(rvol[g]) and rvol[g] < config.DHVC_GAP_MIN_RVOL:
            continue
        base = d.iloc[g + 1:n - 1]                    # bars between the gap and today
        if len(base) < config.DHVC_MIN_BASE_BARS:
            continue
        gap_close = float(d["close"].iloc[g])
        base_lo, base_hi = float(base["low"].min()), float(base["high"].max())
        if base_lo < gap_close * (1 - config.DHVC_GAP_HOLD_TOL / 100):   # gap got filled
            continue
        depth = (base_hi - base_lo) / base_hi * 100 if base_hi > 0 else 1e9
        if depth > config.DHVC_BASE_MAX_DEPTH:
            continue
        if close <= base_hi:                          # today must break out of the base
            continue
        hit = {
            "setup": "delayed_hvc",
            "state": "breakout",               # the breakout HVC day IS the trigger
            "gap_pct": round(float(gap[g]), 2),
            "base_bars": int(len(base)),
            "rvol": round(float(row["rvol"]), 2),
            "depth_pct": round(depth, 1),
            "cons_bars": int(len(base)),
            "level": round(base_hi, 2),               # the consolidation high we broke
            "trigger": float(row["high"]),
        }
        if config.DHVC_REQUIRE_UPTREND and config.REGIME_USE:
            ctx = regime.current_regime(d)
            if not ctx.strong_uptrend:
                return None
            hit.update(_trend_fields(ctx))
            hit.update(_uptrend_dates(d, ctx))
        return hit
    return None


def _qm_ride(close, low, e10, e20, s50) -> dict | None:
    """Shakeout-aware ride of the rising 10/20-EMA through the consolidation (Qullamaggie surf).
    Models emarider's hold/breach idea (emarider.py) but length-agnostic, with a 50-MA HARD FLOOR
    and close-based 1-bar recovery. Inputs are numpy arrays over the bars AFTER the move high.

      riding         close >= EMA20*(1-band)                 — on/above the line
      pullback-ok    EMA20 zone down to the 50-MA            — a normal pullback (rule: to the 50-MA)
      breakdown      a close BELOW the 50-MA                 — reject (decisive)
      shakeout       a wick undercut of the EMA bought back  — bullish: close held the line same bar,
                     OR a close below the line reclaimed the very next bar (counted, not penalized)

    Requires the EMA stack rising (EMA10>EMA20, both sloping up) and at least QMB_MIN_RIDE_FRAC of
    bars riding (so it is genuinely surfing, not bleeding along the 50-MA). Returns
    {ride_frac, shakeouts, ok} or None on any failed gate."""
    if len(close) == 0 or np.isnan(e10).any() or np.isnan(e20).any() or np.isnan(s50).any():
        return None
    if np.any(close < s50):                               # 50-MA hard floor — a close below rejects
        return None
    # uptrend: fast EMA above slow, and the trend still rising. Gauge "rising" by the stable 50-MA, NOT
    # the 10/20-EMA slope — those naturally flatten/drift down off the post-pole peak during a healthy
    # consolidation, so an EMA-slope test would reject most valid bases.
    if not (e10[-1] > e20[-1] and s50[-1] > s50[0]):
        return None
    band = 1.0 - config.QMB_SURF_BAND / 100.0
    riding = close >= e20 * band
    ride_frac = float(riding.mean())
    if ride_frac < config.QMB_MIN_RIDE_FRAC:
        return None
    if len(riding) >= 2 and not riding[-1] and not riding[-2]:   # two closes below the 20-EMA, unrecovered
        return None                                              # = a broken surf / breakdown (e.g. HPE)
    same_bar = (low < e20) & riding                       # dipped below intrabar, closed back on the line
    next_reclaim = (~riding) & np.concatenate([riding[1:], [False]])   # closed below, next bar reclaimed
    shakeouts = int((same_bar | next_reclaim).sum())
    return {"ride_frac": round(ride_frac, 2), "shakeouts": shakeouts, "ok": True}


def _qm_higher_lows(d: pd.DataFrame, start: int, end: int) -> bool:
    """The contraction range must trend to HIGHER lows: its swing lows are NET higher (last >= first)
    — net-higher, not strictly monotonic, so an intra-range shakeout dip (a lower pivot bought back)
    is allowed, consistent with the bullish-shakeout rule. Lenient when the range has <2 pivot lows."""
    lookback = (len(d) - 1) - start + 2
    lows = [p for i, p in trendlab.swing_lows(d, lookback) if start <= i <= end]
    if len(lows) < 2:
        return True
    return lows[-1] >= lows[0]


def detect_qm_breakout(d: pd.DataFrame) -> dict | None:
    """Qullamaggie momentum continuation. A big move (>=QMB_MIN_MOVE over a few days-weeks within the
    last ~3 months) is the qualifying CONTEXT. The actionable pivot is the consolidation's UPPER
    TRENDLINE — fit robustly so it ignores shakeout spikes — which can be flat (box / ascending
    triangle) or DESCENDING (pennant / symmetrical triangle); the lower boundary is the rising higher
    lows, and the two must converge. Through the base price SURFS the rising 10/20-EMA (pullbacks to the
    50-MA OK, shakeout undercuts bought back fast are bullish; a close below the 50-MA rejects). The
    trigger is the upper line extrapolated to the bar — NOT the pole high. State: `building` while
    coiling under it, `breakout` the day price first closes above it ON VOLUME EXPANSION, dropped after
    (a failed poke-above re-fits next day as an ignored up-shakeout). RS leadership (top ~1-2% over
    1/3/6mo) is the rs_rank dashboard filter, not gated here."""
    n = len(d)
    if n < config.QMB_MOVE_LOOKBACK + config.QMB_CONS_MAX + 30:
        return None
    # move high found EXCLUDING today (same breakout-day blindness as HTF/flat_base: a new-high
    # breakout bar used to become its own reference high -> cons_bars=0 -> None)
    look = d.iloc[-(config.QMB_MOVE_LOOKBACK + config.QMB_CONS_MAX + 1):-1]
    hi_pos = int(d.index.get_loc(look["high"].idxmax()))
    hi_p = float(d["high"].iloc[hi_pos])
    cons_bars = n - 1 - hi_pos
    if not (config.QMB_CONS_MIN <= cons_bars <= config.QMB_CONS_MAX):
        return None
    # 1. qualifying pole: a big move into the high over <= QMB_MOVE_MAX_BARS bars
    mstart = max(0, hi_pos - config.QMB_MOVE_MAX_BARS)
    mlow_pos = mstart + int(np.argmin(d["low"].to_numpy(float)[mstart:hi_pos + 1]))   # the move's launch low
    move_low = float(d["low"].iloc[mlow_pos])
    if move_low <= 0:
        return None
    move = (hi_p / move_low - 1) * 100
    if move < config.QMB_MIN_MOVE:
        return None
    # 2. controlled pole pullback (less is better)
    cons_low = float(d["low"].iloc[hi_pos:].min())
    pullback = (1 - cons_low / hi_p) * 100
    if pullback > config.QMB_MAX_PULLBACK:
        return None
    # 3. breakout-level STACK (shared with HTF): the trigger is the top of the stack (nearest resistance).
    # `building` while coiling under it; `breakout` the day a level is popped on volume expansion.
    closeA = d["close"].to_numpy(float)
    volA, proj_frac = _project_last_volume(d)                           # project TODAY's partial bar (if live)
    win = config.QMB_CONTRACT_WIN
    vbase = volA[max(0, n - 1 - win):n - 1].mean()                      # recent contraction's avg volume
    vol_ok_arr = np.zeros(n, bool)                                      # QM volume rule per bar
    for i in range(hi_pos + 1, n):
        vb = volA[max(0, i - win):i].mean()
        vol_ok_arr[i] = vb > 0 and volA[i] >= config.QMB_BREAKOUT_VOL * vb
    st = _breakout_stack(d, hi_pos, n - 1, vol_ok_arr, config.BREAKOUT_STACK_DOWNSWING)
    trig = st["trigger"]
    state = "breakout" if st["broke_today"] else "building"
    provisional = bool(st["broke_today"] and proj_frac is not None)     # today's break used projected volume
    rend = n - 1
    lo_start = hi_pos + 1 + int(np.argmin(d["low"].to_numpy(float)[hi_pos + 1:n]))   # consolidation low
    # 4. shakeout-aware surf of the rising 10/20-EMA over the consolidation (pole high -> latest bar)
    cons_sl = slice(hi_pos + 1, n)
    ride = _qm_ride(
        closeA[cons_sl], d["low"].to_numpy(float)[cons_sl],
        d["ema10"].to_numpy(float)[cons_sl], d["ema20"].to_numpy(float)[cons_sl],
        d["sma50"].to_numpy(float)[cons_sl])
    if ride is None:
        return None
    # 5. higher lows within the rising-support window (from the consolidation low)
    if not _qm_higher_lows(d, lo_start, rend):
        return None
    # 6. assemble (trigger = top of the breakout stack)
    close = float(closeA[n - 1])
    pat = patterns.classify_consolidation(d, hi_pos + 1, rend)
    vol_x = round(float(volA[n - 1] / vbase), 2) if vbase > 0 else None    # latest bar vol vs recent avg
    idx = d.index
    iso = lambda j: idx[j].date().isoformat()
    return {
        "setup": "qm_breakout",
        "state": state,
        "provisional": provisional,          # breakout confirmed on TODAY's projected (partial) volume
        "vol_proj": round(proj_frac, 2) if proj_frac is not None else None,   # session fraction used
        "move_pct": round(move, 1),
        "cons_bars": int(cons_bars),
        "pullback_pct": round(pullback, 1),
        "ride_frac": ride["ride_frac"],
        "shakeouts": ride["shakeouts"],
        "vol_x": vol_x,
        "breakouts": st["n_broken"],
        "levels": st["n_levels"],
        "pattern": pat["pattern"],
        "dist_to_trigger_pct": round((trig / close - 1) * 100, 2),
        "level": round(float(d["low"].to_numpy(float)[lo_start]), 2),   # consolidation low (support)
        "trigger": round(trig, 2),
        # chart geometry: pole + consolidation shading, and the stack resistance levels (horizontal)
        "qm_pole": [iso(mlow_pos), iso(hi_pos)],
        "qm_cons": [iso(hi_pos + 1), iso(n - 1)],
        "stack_levels": [round(x, 2) for x in st["levels"]],
    }


def _ema_rider_state(d: pd.DataFrame) -> dict | None:
    """The EMA streak ENDING on the latest bar: direction (+1 above / -1 below), length, and
    'holds' = EMA touches after the 1st streak bar (wick within ATR*frac, a wick through counts).
    Ports emaRider.pine via emarider.current_streak. Memoized per frame (framecache) keyed by the
    params, so bull + bear share one pass and a settings change recomputes (mirrors the trendlab cache)."""
    key = (config.ER_EMA_LEN, config.ER_ATR_LEN, config.ER_ATR_FRAC,
           config.ER_USE_STREAK_THRESH, config.ER_STREAK_ATR_FRAC, config.ER_USE_ARM_EXIT)
    cached = _EMA_RIDER_CACHE.get(d)
    if cached is not None and cached[0] == key:
        return cached[1]
    res = emarider.current_streak(
        d, config.ER_EMA_LEN, config.ER_ATR_LEN, config.ER_ATR_FRAC,
        config.ER_USE_STREAK_THRESH, config.ER_STREAK_ATR_FRAC, config.ER_USE_ARM_EXIT)
    _EMA_RIDER_CACHE.put(d, (key, res))
    return res


def _qualifies(st: dict | None) -> bool:
    """A streak-stats dict (from emarider.current_streak) is a real ride."""
    return (st is not None and st["length"] >= config.ER_MIN_STREAK
            and st["holds"] >= config.ER_MIN_HOLDS)


def _ema_rider_hit(d: pd.DataFrame, want: int, state: str, st: dict, shade_to: int) -> dict:
    """Build the hit row from a ride's streak stats `st`. `shade_to` = end bar position of the
    ride to shade (inclusive). `state` in {riding, touching, reversal}."""
    up = want > 0
    row = d.iloc[-1]
    ema_v = st["ema"]
    fld = "uptrend" if up else "downtrend"        # shade the ride green (bull) / red (bear)
    # intraday frames carry a time-of-day -> keep the full timestamp so the chart can shade the
    # exact ride window (daily frames keep the plain date, matching the date-string chart axis)
    intraday = bool((d.index.normalize() != d.index).any())
    iso = (lambda ts: ts.isoformat()) if intraday else (lambda ts: ts.date().isoformat())
    return {
        "setup": "ema_rider_bull" if up else "ema_rider_bear",
        "state": state,
        "ema_len": config.ER_EMA_LEN,
        "streak": st["length"],
        "holds": st["holds"],
        "breaches": st["breaches"],
        "max_wick": st["max_wick"],
        "cons_bars": st["length"],
        "level": round(ema_v, 2) if ema_v is not None else None,   # the EMA = support (bull) / resistance (bear)
        "trigger": float(row["high"] if up else row["low"]),       # continuation entry = break of today's extreme
        f"{fld}_from": iso(d.index[st["start"]]),
        f"{fld}_to": iso(d.index[shade_to]),
    }


def _ema_rider(d: pd.DataFrame, want: int) -> dict | None:
    """Five states for a stock riding the EMA in the `want` direction (+1 above / -1 below):
    riding (clean), touching (wick tagged the EMA zone but closed on-side), armed (closed WRONG-side —
    an exit is armed, the ride still counts), saved (an armed exit just resolved back on-side), break
    (the armed bar's extreme was taken out THIS bar -> the ride flipped, dated back to the arming bar).
    armed/saved/break come from the emaRider.pine arm-exit machine (config.ER_USE_ARM_EXIT); with it off
    the states reduce to riding/touching/break (break = the legacy first-wrong-close flip)."""
    if len(d) < config.ER_MIN_STREAK + config.ER_ATR_LEN + 5:
        return None
    st = _ema_rider_state(d)
    if st is None:
        return None
    arm = config.ER_USE_ARM_EXIT
    if st["direction"] == want and _qualifies(st):                 # still riding our way this bar
        if arm and st.get("armed"):
            state = "armed"                                        # closed wrong-side, exit armed
        elif arm and st.get("event") == "saved":
            state = "saved"                                        # armed exit defended, ride continues
        elif st["near_now"]:
            state = "touching"                                     # wick tagged the EMA, closed on-side
        else:
            state = "riding"
        return _ema_rider_hit(d, want, state, st, len(d) - 1)
    if arm and st["direction"] == -want and st.get("event") == "break":   # the want-ride just BROKE
        broke_at = st["start"]                                     # arming bar (the down-count is dated here)
        prev = emarider.current_streak(                            # the want-ride as it stood at the arming bar
            d.iloc[:broke_at + 1], config.ER_EMA_LEN, config.ER_ATR_LEN, config.ER_ATR_FRAC,
            config.ER_USE_STREAK_THRESH, config.ER_STREAK_ATR_FRAC, arm)
        if prev is not None and prev["direction"] == want and _qualifies(prev):
            return _ema_rider_hit(d, want, "break", prev, broke_at)
    if not arm and st["direction"] == -want and st["length"] == 1:  # legacy: first-wrong-close flip
        prev = emarider.current_streak(                            # the ride as it stood last bar (causal)
            d.iloc[:-1], config.ER_EMA_LEN, config.ER_ATR_LEN, config.ER_ATR_FRAC,
            config.ER_USE_STREAK_THRESH, config.ER_STREAK_ATR_FRAC, arm)
        if prev is not None and prev["direction"] == want and _qualifies(prev):
            return _ema_rider_hit(d, want, "break", prev, len(d) - 2)
    return None


def detect_ema_rider_bull(d: pd.DataFrame) -> dict | None:
    """A long streak of closes ABOVE the EMA that keeps getting bought back at the line."""
    return _ema_rider(d, 1)


def detect_ema_rider_bear(d: pd.DataFrame) -> dict | None:
    """A long streak of closes BELOW the EMA that keeps getting sold at the line."""
    return _ema_rider(d, -1)


# ---- multi-timeframe setups (run on both 1D and 1W) -------------------------

def _bb_os_state(d: pd.DataFrame, collect: list | None = None,
                 l1: float | None = None, l2: float | None = None):
    """Amir's Pine BB-Targets state machine over the frame. Returns (fire1, fire2, run_ath,
    p1, p2, armed) for the CURRENT episode; pass `collect` to also receive every historical fire
    as (bar_index, 'entry1'|'entry2') — used by the review-page sampler. `armed` = a new high
    stands unconsumed (the next oversold wick fires) — the multi-TF backburner watch state.
    `l1`/`l2` override the RSI trigger levels (default 30/20); l2 >= l1 disables entry2."""
    l1 = l1 if l1 is not None else config.BB_RSI_ENTRY1
    l2 = l2 if l2 is not None else config.BB_RSI_ENTRY2
    high = d["high"].to_numpy(float)
    low_a = d["low"].to_numpy(float)
    p1 = indicators.rsi_trigger_price(d["close"], l1).to_numpy(float)
    p2 = (indicators.rsi_trigger_price(d["close"], l2).to_numpy(float)
          if l2 < l1 else np.full(len(d), np.nan))
    armed, ref_ath, run_ath = True, float(high[0]), float(high[0])
    fire1 = fire2 = None
    ep_open = False                                  # between a fire and the next re-arm
    for t in range(1, len(d)):
        run_ath = max(run_ath, float(high[t]))
        if high[t] > ref_ath:
            armed, ep_open = True, False
        if armed and not np.isnan(p1[t]) and low_a[t] <= p1[t]:
            fire1, fire2, ep_open = t, None, True    # the episode's first oversold wick
            ref_ath, armed = run_ath, False          # re-arm only after a NEW high
            if collect is not None:
                collect.append((t, "entry1"))
        elif ep_open and fire2 is None and not np.isnan(p2[t]) and low_a[t] <= p2[t]:
            fire2 = t                                # same episode deepening to entry2
            if collect is not None:
                collect.append((t, "entry2"))
    return fire1, fire2, run_ath, p1, p2, armed


def detect_backburner(d: pd.DataFrame) -> dict | None:
    """Backburner — TWO paths sharing the ATH anchor (Amir trades the TCG oversold-bounce; the
    near-ATH scan is its candidate filter):
    CLASSIC (states building/breakout): at/near the ATH after a strong trendlab uptrend, retraced
    <= BACKBURNER_MAX_RETRAC of that micro up-move — the coiling-at-highs list (unchanged).
    OS ENTRIES (states entry1/entry2): Amir's Pine BB-Targets algo — the fire is the INTRABAR
    oversold WICK: low <= pOS, the exact price at which this bar's RSI prints the OS level
    (close-RSI misses wick touches). State machine: armed -> first wick fires -> disarmed;
    only a new running-ATH re-arms. entry2 = the same episode deepening to the RSI-20 price."""
    if not config.REGIME_USE or len(d) < 60:
        return None
    high = d["high"].to_numpy(float)
    close = float(d["close"].iloc[-1])
    ath_pos = int(high.argmax())
    ath = float(high[ath_pos])

    # ---- OS-entry path (checked first: an entry outranks a coiling row) ----
    fire1, fire2, run_ath, p1, p2, _armed = _bb_os_state(d)
    last = len(d) - 1
    if last in (fire1, fire2):
        rsi_now = (float(d["rsi14"].iloc[-1])
                   if "rsi14" in d.columns and pd.notna(d["rsi14"].iloc[-1]) else None)
        state = "entry2" if fire2 == last else "entry1"
        pos = float(p2[last] if state == "entry2" else p1[last])
        return {
            "setup": "backburner", "state": state,
            "rsi": round(rsi_now, 1) if rsi_now is not None else None,
            "os_price": round(pos, 2),               # the wick trigger (intrabar RSI = level)
            "ath": round(run_ath, 2),
            "retrac_pct": round((run_ath - close) / run_ath * 100, 1),   # off-ATH %
            "level": round(pos, 2),
            "trigger": round(float(d["high"].iloc[-1]), 2),   # entry = break of the fire-bar high
        }

    # ---- classic near-ATH path (unchanged) ----
    if config.BACKBURNER_ATH_MAX_AGE and ath_pos < len(d) - 1 - config.BACKBURNER_ATH_MAX_AGE:
        return None
    ctx = regime.prior_trend_context(d, ath_pos + 1)
    if not ctx.strong_uptrend:
        return None
    up_low = float(d["low"].iloc[ctx.start:ath_pos + 1].min())
    span = ath - up_low
    if span <= 0:
        return None
    retrac = (ath - close) / span * 100
    if retrac > config.BACKBURNER_MAX_RETRAC:
        return None
    rsi_now = (round(float(d["rsi14"].iloc[-1]), 1)
               if "rsi14" in d.columns and pd.notna(d["rsi14"].iloc[-1]) else None)
    return {
        "setup": "backburner",
        # "armed": printed a new ATH on THIS bar — the event that re-arms the OS state machine
        # (Amir: not a trade trigger; the trade is the later entry1/entry2 oversold wick)
        "state": "armed" if ath_pos == len(d) - 1 else "building",
        "rsi": rsi_now,
        "retrac_pct": round(retrac, 1),
        "ath": round(ath, 2),
        "up_low": round(up_low, 2),
        "prior_gain_pct": round(ctx.net_pct, 1),
        "prior_clarity": round(ctx.clarity, 2),
        "cons_bars": int(len(d) - 1 - ath_pos),
        "level": round(up_low + 0.5 * span, 2),
        "trigger": round(ath, 2),
        **_uptrend_dates(d, ctx),
    }


def detect_stairstep(d: pd.DataFrame) -> dict | None:
    """After a strong uptrend, an orderly staircase down: >= STAIRSTEP_MIN_BARS
    consecutive bars each failing to make a meaningful new high (high < previous
    high + threshold), ending at the last bar or one bar back (the reversal)."""
    if not config.REGIME_USE or len(d) < 30:
        return None
    h = d["high"].to_numpy(float)
    atr = d["atr14"].to_numpy(float)
    n = len(d)

    def trailing_run(end: int) -> int:
        run = 0
        for i in range(end, 0, -1):
            if pd.notna(atr[i]) and h[i] < h[i - 1] + atr[i] * config.STAIRSTEP_ATR_FRACTION:
                run += 1
            else:
                break
        return run

    # the run may end on the last bar, or up to STAIRSTEP_MAX_BREAKOUT_BARS back
    # (those trailing bars are allowed to break out above the staircase)
    ends = range(n - 1, n - 2 - config.STAIRSTEP_MAX_BREAKOUT_BARS, -1)
    run, end = max(((trailing_run(e), e) for e in ends if e >= 1), default=(0, n - 1))
    if run < config.STAIRSTEP_MIN_BARS:
        return None
    peak_pos = end - run                               # the high the staircase descends from
    if h[end] >= h[peak_pos]:                           # must net-decline, not a flat top
        return None
    ctx = regime.prior_trend_context(d, peak_pos + 1)  # uptrend into that high
    if not ctx.strong_uptrend:
        return None
    # leadership measured AS OF THE PEAK (before the stepdown), so a strong stock now mid-pullback
    # isn't filtered out for being below its MAs today — the dashboard filters on these for stairstep.
    pclose = float(d["close"].iloc[peak_pos])
    s50p, s200p, adrp = d["sma50"].iloc[peak_pos], d["sma200"].iloc[peak_pos], d["adr_pct"].iloc[peak_pos]
    ath_to_peak = float(d["high"].iloc[:peak_pos + 1].max())
    peak = {
        "above_sma50_at_peak": bool(pclose > s50p) if pd.notna(s50p) else None,
        "above_sma200_at_peak": bool(pclose > s200p) if pd.notna(s200p) else None,
        "off_ath_at_peak": round((pclose / ath_to_peak - 1) * 100, 1) if ath_to_peak > 0 else None,
        "adr_at_peak": round(float(adrp), 2) if pd.notna(adrp) else None,
    }
    return {
        "setup": "stairstep",
        "state": "breakout" if h[-1] > h[-2] else "building",   # broke the prior bar's high vs still stepping
        "step_bars": run,
        "pivot_high": round(float(h[peak_pos]), 2),
        "prior_gain_pct": round(ctx.net_pct, 1),
        "prior_clarity": round(ctx.clarity, 2),
        "cons_bars": int(len(d) - 1 - peak_pos),   # the staircase window from the swing high
        "level": round(float(d["low"].iloc[-1]), 2),
        "trigger": round(float(h[-1]), 2),         # entry = break of the latest bar's high
        **peak,
        **_uptrend_dates(d, ctx),
    }


def _dt_str(d: pd.DataFrame, i) -> str | None:
    """Bar timestamp for chart anatomy: date for daily/weekly bars, full ISO for intraday bars
    (the chart payload converts intraday stamps to the epoch axis)."""
    if i is None or not (0 <= int(i) < len(d)):
        return None
    ts = d.index[int(i)]
    return ts.isoformat() if (ts.hour or ts.minute) else ts.date().isoformat()


def detect_cup_handle(d: pd.DataFrame) -> dict | None:
    """Cup-with-handle continuation (geometry template, chart-verified by Amir's 3 verdict
    passes). Long: trigger = handle high; state building / breakout."""
    if not config.REGIME_USE or len(d) < 90:
        return None
    m = geometry.cup_handle(d)
    if m is None:
        return None
    return {"setup": "cup_handle", "state": m["state"],
            "cup_bars": m.get("cup_bars"), "depth_pct": m.get("depth_pct"),
            "handle_bars": m.get("handle_bars"), "handle_pull_pct": m.get("handle_pull_pct"),
            "target": m.get("target"),
            "l_rim_date": _dt_str(d, m["l_rim"][0]), "r_rim_date": _dt_str(d, m["r_rim"][0]),
            "level": float(m["cup_low"][1]), "trigger": m.get("trigger"),   # cup_low is (bar, price)
            # pattern anatomy on the chart (validated-setup requirement, Amir 2026-07-03)
            "marks": [{"date": _dt_str(d, m["l_rim"][0]), "text": "L-rim", "pos": "above"},
                      {"date": _dt_str(d, m["cup_low"][0]), "text": "cup low", "pos": "below"},
                      {"date": _dt_str(d, m["r_rim"][0]), "text": "R-rim", "pos": "above"}],
            "xlines": [{"price": m.get("target"), "title": "target"}]}


def detect_double_top(d: pd.DataFrame) -> dict | None:
    """Double/triple top reversal (geometry template, chart-verified). SHORT setup: trigger =
    the valley/neckline break; state building / breakdown. Also a warning on longs."""
    if not config.REGIME_USE or len(d) < 60:
        return None
    m = geometry.double_top(d)
    if m is None:
        return None
    return {"setup": "double_top", "state": m["state"], "peaks": m.get("peaks"),
            "vol_lighter_p2": m.get("vol_lighter_p2"), "target": m.get("target"),
            "p1_date": _dt_str(d, m["p1"][0]), "p2_date": _dt_str(d, m["p2"][0]),
            "level": m.get("valley"), "trigger": m.get("valley"),
            "marks": [{"date": _dt_str(d, m["p1"][0]), "text": "P1", "pos": "above"},
                      {"date": _dt_str(d, m["p2"][0]), "text": "P2", "pos": "above"}],
            "xlines": [{"price": m.get("target"), "title": "target"}]}


def detect_head_shoulders(d: pd.DataFrame) -> dict | None:
    """Head & shoulders top (geometry template, chart-verified). SHORT setup: trigger = the
    neckline break; state building / breakdown."""
    if not config.REGIME_USE or len(d) < 60:
        return None
    m = geometry.head_shoulders(d, inverse=False)
    if m is None:
        return None
    return {"setup": "head_shoulders", "state": m["state"], "target": m.get("target"),
            "head_date": _dt_str(d, m["head"][0]), "ls_date": _dt_str(d, m["ls"][0]),
            "rs_date": _dt_str(d, m["rs"][0]),
            "level": m.get("neckline"), "trigger": m.get("neckline"),
            "marks": [{"date": _dt_str(d, m["ls"][0]), "text": "LS", "pos": "above"},
                      {"date": _dt_str(d, m["head"][0]), "text": "HEAD", "pos": "above"},
                      {"date": _dt_str(d, m["rs"][0]), "text": "RS", "pos": "above"}],
            "segline": [{"date": _dt_str(d, m["neck1"][0]), "price": float(m["neck1"][1])},
                        {"date": _dt_str(d, m["neck2"][0]), "price": float(m["neck2"][1])},
                        {"date": _dt_str(d, len(d) - 1), "price": float(m["neckline"])}],
            "xlines": [{"price": m.get("target"), "title": "target"}]}


def detect_inverse_hs(d: pd.DataFrame) -> dict | None:
    """Inverse head & shoulders bottom (geometry template, chart-verified). Long: trigger = the
    neckline reclaim; state building / breakout."""
    if not config.REGIME_USE or len(d) < 60:
        return None
    m = geometry.head_shoulders(d, inverse=True)
    if m is None:
        return None
    return {"setup": "inverse_hs", "state": m["state"], "target": m.get("target"),
            "head_date": _dt_str(d, m["head"][0]), "ls_date": _dt_str(d, m["ls"][0]),
            "rs_date": _dt_str(d, m["rs"][0]),
            "level": m.get("neckline"), "trigger": m.get("neckline"),
            "marks": [{"date": _dt_str(d, m["ls"][0]), "text": "LS", "pos": "below"},
                      {"date": _dt_str(d, m["head"][0]), "text": "HEAD", "pos": "below"},
                      {"date": _dt_str(d, m["rs"][0]), "text": "RS", "pos": "below"}],
            "segline": [{"date": _dt_str(d, m["neck1"][0]), "price": float(m["neck1"][1])},
                        {"date": _dt_str(d, m["neck2"][0]), "price": float(m["neck2"][1])},
                        {"date": _dt_str(d, len(d) - 1), "price": float(m["neckline"])}],
            "xlines": [{"price": m.get("target"), "title": "target"}]}


def _rsix_detect(d: pd.DataFrame, side: str) -> dict | None:
    """RSI Historical Extremes mean-revert (Amir's rsi_extreme.pine port, 2026-07-03): the
    symbol's OWN all-time RSI extreme (prior bars only — blindness rule) defines its personal
    zone; the fire is the intrabar WICK touching the price where RSI prints the zone edge —
    backburner mechanics on a dynamic level. One fire per episode; re-arm on an RSI-50 reclaim.
    Young histories are skipped (an expanding extreme is meaningless on its first months)."""
    if not config.REGIME_USE or len(d) < config.RSIX_MIN_HISTORY:
        return None
    import rsi_extremes as rx
    st = rx.state(d)
    if st is None:
        return None
    fires: list = []
    last_fire, p, _armed = rx.fires(d, side, collect=fires)
    n = len(d) - 1
    fired_recent = last_fire is not None and (n - last_fire) <= config.RSIX_RECENT_BARS
    in_zone = st["zone"] == ("atl" if side == "long" else "ath")
    if not (fired_recent or in_zone):
        return None
    trig = float(p.iloc[-1]) if pd.notna(p.iloc[-1]) else None
    hit = {"setup": "rsi_extreme_revert" if side == "long" else "rsi_extreme_fade",
           "state": "entry1" if last_fire == n else "armed",
           "rsi": st["rsi"],
           "rsi_extreme": st["all_lo"] if side == "long" else st["all_hi"],
           "zone_edge": round((st["all_lo"] + config.RSIX_BUF) if side == "long"
                              else (st["all_hi"] - config.RSIX_BUF), 1),
           "trigger": round(trig, 2) if trig is not None else None,
           "level": round(trig, 2) if trig is not None else None,
           "bars_since_fire": (n - last_fire) if last_fire is not None else None,
           "broke_extreme": st["broke"]}
    if fires:                                        # mark the latest historical fire on the chart
        col = "low" if side == "long" else "high"
        hit["marks"] = {"fire": (int(fires[-1]), round(float(d[col].iloc[fires[-1]]), 2))}
    return hit


def detect_rsi_extreme_revert(d: pd.DataFrame) -> dict | None:
    """LONG mean-revert: wick into the symbol's own ALL-TIME-low RSI zone (capitulation)."""
    return _rsix_detect(d, "long")


def detect_rsi_extreme_fade(d: pd.DataFrame) -> dict | None:
    """SHORT fade: wick into the symbol's own ALL-TIME-high RSI zone (blow-off euphoria)."""
    return _rsix_detect(d, "short")


DETECTORS = [detect_gapper, detect_episodic_pivot, detect_hvc, detect_flat_base,
             detect_high_tight_flag, detect_higher_low_ma, detect_undercut_rally,
             detect_delayed_hvc, detect_qm_breakout, detect_uptrend, detect_downtrend,
             detect_ema_rider_bull, detect_ema_rider_bear,
             detect_double_top, detect_cup_handle, detect_head_shoulders, detect_inverse_hs]
# Temporarily disabled for speed:
# detect_rsi_extreme_revert, detect_rsi_extreme_fade
MULTI_TF_DETECTORS = [detect_backburner, detect_stairstep]

# setup name -> detector, so the server can recompute a SUBSET of setups (per-setup caching)

def detect_ema_cross(d, sym=None, context=None):
    """Evaluates EMA Cross / Larsson Line across absolute and relative baselines."""
    import labels
    import datastore
    import indicators
    import pandas as pd
    import scanner_core
    hits = []
    
    # 0. Determine optimal TF
    sec_name = labels.sector(sym) if sym else "Unknown"
    adr = 0.0
    if len(d) >= 20:
        adr = (d['high'].iloc[-20:] / d['low'].iloc[-20:] - 1).mean() * 100
        
    optimal_tf = "1D"
    if sec_name in [
        "Computer Hardware", "Utilities - Independent Power Producers", "Internet Retail",
        "Residential Construction", "Oil & Gas E&P", "Oil & Gas Midstream", "Internet Content & Information",
        "Waste Management", "Furnishings, Fixtures & Appliances", "Semiconductors", "Insurance - Property & Casualty",
        "REIT - Industrial", "Solar", "Luxury Goods", "Specialty Retail", "Uranium",
        "Financial Data & Stock Exchanges", "Medical Devices", "Electronic Gaming & Multimedia",
        "Consumer Defensive", "Information Technology Services", "Gold", "Building Products & Equipment",
        "Healthcare", "Beverages - Non-Alcoholic", "Conglomerates", "Biotechnology",
        "Building Materials", "Insurance - Life", "Energy", "Steel", "Resorts & Casinos",
        "Staffing & Employment Services", "Auto & Truck Dealerships", "Industrial Distribution",
        "Oil & Gas Integrated", "Tobacco", "Oil & Gas Equipment & Services", "Tools & Accessories",
        "Medical Care Facilities", "Travel Services", "Utilities", "Trucking", "Beverages - Brewers"
    ]:
        optimal_tf = "2D"
    elif sec_name in ["Software - Application", "Software - Infrastructure", "Diagnostics & Research", "Medical Instruments & Supplies", "Drug Manufacturers - Specialty & Generic"]:
        optimal_tf = "1W"
    else:
        if adr < 3.0: optimal_tf = "1W"
        elif adr < 5.0: optimal_tf = "3D"
        elif adr < 8.0: optimal_tf = "2D"
        else: optimal_tf = "2D"

    # Resample if needed
    is_daily_input = len(d) >= 2 and (d.index[-1] - d.index[-2]).days <= 3
    if is_daily_input and optimal_tf != "1D":
        df_target = datastore.resample_daily(d[["open", "high", "low", "close", "volume"]], optimal_tf)
        if df_target is None or len(df_target) < 10:
            df_target = d
            optimal_tf = "1D"
    else:
        df_target = d
        if not is_daily_input:
            optimal_tf = "1W"

    d_abs = scanner_core.calc_larssson_line(df_target)
    
    def check_flip(df_annotated, variant, entity_type):
        if 'larsson_state' not in df_annotated.columns: return
        states = df_annotated['larsson_state'].dropna()
        if len(states) < 2: return
        curr = states.iloc[-1]
        
        if curr != 'yellow':
            return
            
        flipped_idx = None
        for i in range(1, len(states)):
            if states.iloc[-1 - i] != curr:
                flipped_idx = i - 1
                break
        
        if flipped_idx is None:
            flipped_idx = len(states) - 1
            
        if flipped_idx > 5:
            return

        # Group handling
        if flipped_idx == 0:
            group = "Today's Flips"
        else:
            group = "Past Week Flips"

        # EP event handling
        med60 = (df_annotated["close"] * df_annotated["volume"]).rolling(60, min_periods=10).median().shift(1)
        ep_event_found = False
        start_idx = max(0, len(df_annotated) - 1 - flipped_idx - 3)
        end_idx = min(len(df_annotated), len(df_annotated) - 1 - flipped_idx + 4)
        if "rvol" in df_annotated.columns:
            for idx in range(start_idx, end_idx):
                if _is_ep_event(df_annotated.iloc[idx], med60.iloc[idx] if pd.notna(med60.iloc[idx]) else None):
                    ep_event_found = True
                    break

        # Advice formulation
        advice = ""
        if curr == 'yellow':
            advice = "Immediate Entry. "
            if ep_event_found:
                advice = "[EP Event Detected Near Flip] " + advice
            if sec_name in ["Semiconductors", "Software - Application", "Aerospace & Defense"]:
                advice += "EP Confluence strongly favored. "
            elif sec_name == "Biotechnology":
                advice += "Avoid EP (catalyst gaps bleed out). "
                
            if sec_name in ["Semiconductors", "Aerospace & Defense", "Biotechnology", "Internet Content & Information", "Engineering & Construction", "Software - Application"]:
                advice += "Exit on BLUE."
            elif sec_name in ["Software - Infrastructure", "Asset Management", "Specialty Chemicals", "Telecom Services", "Packaged Foods"]:
                advice += "Exit on GREY."
            else:
                advice += "Standard Trade."

        is_favorable = False
        score = 0
        reason = ""
        
        if entity_type == "ticker" and curr == 'yellow' and context and 'frames' in context:
            sec = labels.sector(sym)
            if sec in scanner_core.best_larsson_sectors():
                sector_map = {
                    "Biotechnology": "XBI", "Semiconductors": "SMH",
                    "Semiconductor Equipment & Materials": "SMH",
                    "Software - Application": "IGV", "Software - Infrastructure": "IGV",
                    "Computer Hardware": "QQQ", "Specialty Retail": "XRT"
                }
                etf_sym = sector_map.get(sec, "QQQ")
                
                frames = context['frames']
                if etf_sym in frames:
                    etf_df = frames[etf_sym]
                    date = df_annotated.index[-1]
                    if date in etf_df.index:
                        close = etf_df.loc[date, 'close']
                        sma50 = etf_df['close'].rolling(50).mean().loc[date]
                        sma200 = etf_df['close'].rolling(200).mean().loc[date]
                        sma10 = etf_df['close'].rolling(10).mean().loc[date]
                        sma20 = etf_df['close'].rolling(20).mean().loc[date]
                        
                        above_200 = close > sma200
                        above_50 = close > sma50
                        fast_mom = sma10 > sma20
                        
                        if above_200: score += 1
                        if above_50: score += 1
                        if fast_mom: score += 1
                        
                        is_favorable = (score >= 2)
                        reason = f"{etf_sym} {'above 200' if above_200 else 'below 200'}, {'above 50' if above_50 else 'below 50'}, {'10>20' if fast_mom else '10<20'}"
                        
        hit = {
            "setup": "ema_cross",
            "variant": variant,
            "state": curr,
            "days_since_flip": flipped_idx,
            "entity_type": entity_type,
            "is_favorable": is_favorable,
            "favorability_score": score,
            "favorability_reason": reason,
            "group": group,
            "adr": round(adr, 1),
            "optimal_tf": optimal_tf,
            "advice": advice,
            "tf_override": optimal_tf
        }
        
        if entity_type == "ticker" and context and 'synth_states' in context:
            sec = labels.sector(sym)
            if sec:
                sec_sym = f"SYNTH_SEC_{sec.replace(' ', '').replace('&', 'And').replace('-', '')}".upper()
                if sec_sym in context['synth_states']:
                    hit["sector_state"] = context['synth_states'][sec_sym]
            
            themes = labels.themes(sym)
            if themes:
                thm = themes[0]
                thm_sym = f"SYNTH_THM_{thm.replace(' ', '').replace('&', 'And').replace('-', '')}".upper()
                if thm_sym in context['synth_states']:
                    hit["theme_state"] = context['synth_states'][thm_sym]

        hits.append(hit)

    check_flip(d_abs, "abs", "synth" if (sym and sym.startswith("SYNTH")) else ("ticker" if sym else "synth"))
    
    if context and 'spy_df' in context and context['spy_df'] is not None:
        spy_df = context['spy_df']
        spy_targets = context.get('spy_targets') or {}
        if optimal_tf in spy_targets:
            spy_df_target = spy_targets[optimal_tf]
        elif is_daily_input and optimal_tf != "1D":
            spy_df_target = datastore.resample_daily(spy_df[["open", "high", "low", "close", "volume"]], optimal_tf)
            if spy_df_target is None:
                spy_df_target = spy_df
        else:
            spy_df_target = spy_df
            
        d_rs = df_target.copy()
        for col in ['open', 'high', 'low', 'close']:
            d_rs[col] = d_rs[col] / spy_df_target[col].reindex(d_rs.index, method='ffill')
            
        d_rs = scanner_core.calc_larssson_line(d_rs)
        check_flip(d_rs, "spy", "synth" if (sym and sym.startswith("SYNTH")) else ("ticker" if sym else "synth"))
        
    return hits


CONTEXT_DETECTORS = [detect_ema_cross]
SETUP_OF = {
    detect_gapper: "gapper", detect_episodic_pivot: "episodic_pivot",
    detect_hvc: "hvc", detect_flat_base: "flat_base",
    detect_high_tight_flag: "high_tight_flag", detect_higher_low_ma: "higher_low_ma",
    detect_undercut_rally: "undercut_rally", detect_delayed_hvc: "delayed_hvc",
    detect_qm_breakout: "qm_breakout", detect_uptrend: "uptrend", detect_downtrend: "downtrend",
    detect_ema_rider_bull: "ema_rider_bull", detect_ema_rider_bear: "ema_rider_bear",
    detect_backburner: "backburner", detect_stairstep: "stairstep", detect_ema_cross: "ema_cross",
    detect_cup_handle: "cup_handle", detect_double_top: "double_top",
    detect_head_shoulders: "head_shoulders", detect_inverse_hs: "inverse_hs",
    detect_rsi_extreme_revert: "rsi_extreme_revert", detect_rsi_extreme_fade: "rsi_extreme_fade",
}

def _run(detectors, d: pd.DataFrame, only: set | None) -> list[dict]:

    hits = []
    for fn in detectors:
        if only is not None and SETUP_OF[fn] not in only:
            continue
        try:
            hit = fn(d)
        except Exception:
            hit = None
        if hit:
            hits.append(hit)
    return hits


def run_all(sym: str, d: pd.DataFrame, context: dict | None = None, only: set | None = None) -> list[dict]:
    hits = _run(DETECTORS, d, only)
    if context:
        for fn in CONTEXT_DETECTORS:
            if only is not None and SETUP_OF[fn] not in only: continue
            try:
                hit = fn(d, sym, context)
                if hit:
                    if isinstance(hit, list):
                        hits.extend(hit)
                    else:
                        hits.append(hit)
            except Exception as e:
                pass
    return hits

def run_multi_tf(sym: str, d: pd.DataFrame, context: dict | None = None, only: set | None = None) -> list[dict]:
    return _run(MULTI_TF_DETECTORS, d, only)

def iter_fresh_ep_events(d: pd.DataFrame) -> list[tuple[int, str]]:
    """
    Canonical EP event iterator with O(N) 10-bar debounce.
    Yields (index, subtype) for valid, debounced EP events.
    Used by live detector, registry builder, and ML dataset pipeline.
    """
    if d is None or len(d) < 30:
        return []
        
    med60 = (d["close"] * d["volume"]).rolling(60, min_periods=10).median().shift(1)
    
    valid_eps = []
    last_raw_candidate_idx = -999
    
    for j in range(30, len(d)):
        sub = _is_ep_event(d.iloc[j], med60.iloc[j] if pd.notna(med60.iloc[j]) else None)
        if sub:
            # It's a raw candidate
            if j - last_raw_candidate_idx > config.EP_DEBOUNCE_BARS:
                valid_eps.append((j, sub))
            last_raw_candidate_idx = j
            
    return valid_eps
