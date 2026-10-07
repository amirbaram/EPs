"""Sequence grammar — "what came before" for any instrument, on any timeframe.

Reads the trendlab pivot/segment structure of an enriched OHLC frame (index, sector ETF, theme
basket or ratio; daily or 1h) and classifies the CURRENT moment into a small regime vocabulary
with the events that led to it, an invalidation level, and one human phrase. This is the layer
the old awareness was missing: a 1h uptrend after a base breakout is not the same read as the
same uptrend after a 10-day vertical run into resistance.

    sequence_read(d, lv=None, tf="1D") -> {regime, flags, events, invalidation, confirm,
                                           phrase, meta} | None

Regimes: breakout_fresh, breakdown_fresh, impulse_up, impulse_down, trend_up, trend_down,
distribution, pullback_in_uptrend, bounce_in_downtrend, range, unclear.

Depth/retrace/range measurements anchor on the MAJOR swing (the extreme pivot within
AWARE_MAJOR_LOOKBACK bars), not the last micro-pivot — segment micro-labels flip too fast in
chop for a chart-scale read. All thresholds are AWARE_* config knobs (settings "awareness").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
import mtf
import sr
import trendlab

_BULL_TAGS = {"HH", "HL", "DT HH", "DB HL"}
_BEAR_TAGS = {"LL", "LH", "DT LH", "DB LL"}


def _chain(ptag: list[str]) -> int:
    """Signed length of the trailing consecutive structure chain: +n for a bullish HH/HL run,
    -n for a bearish LL/LH run, 0 if the last tag belongs to neither."""
    if not ptag:
        return 0
    side = _BULL_TAGS if ptag[-1] in _BULL_TAGS else (_BEAR_TAGS if ptag[-1] in _BEAR_TAGS else None)
    if side is None:
        return 0
    n = 0
    for tag in reversed(ptag):
        if tag not in side:
            break
        n += 1
    return n if side is _BULL_TAGS else -n


def _seg_range(d: pd.DataFrame, seg) -> tuple[float, float]:
    hi = float(d["high"].iloc[seg.start:seg.end + 1].max())
    lo = float(d["low"].iloc[seg.start:seg.end + 1].min())
    return hi, lo


def _fmt(x) -> str:
    """Compact price for phrases: 2dp under 100 (ratios print small), else 1dp."""
    if x is None:
        return "?"
    return f"{x:,.2f}" if abs(x) < 100 else f"{x:,.1f}"


def _r(x, nd=2):
    return None if x is None else round(float(x), nd)


def sequence_read(d: pd.DataFrame, lv: dict | None = None, tf: str = "1D",
                  atr_fraction: float | None = None, atr_ref: float | None = None) -> dict | None:
    """Classify the frame's current structure. `lv` = precomputed sr.sr_levels(d) (computed here
    if omitted); `tf` scales the leg-size threshold for intraday frames; `atr_fraction` passes a
    coarser zigzag for close-only basket frames; `atr_ref` (e.g. the DAILY atr for a 1h frame)
    makes depth/height measurements comparable across timeframes."""
    if d is None or len(d) < 40:
        return None
    st = mtf.tf_state(d)
    if st is None:
        return None
    lv = lv if lv is not None else sr.sr_levels(d)
    segs, _swings = trendlab.segment_chart(d, atr_fraction)
    if not segs:
        return None
    pidx, pprice, ptype, ptag = trendlab.pivots(d, atr_fraction)
    close = d["close"].to_numpy(float)
    low = d["low"].to_numpy(float)
    high = d["high"].to_numpy(float)
    c = float(close[-1])
    atr = float(d["atr14"].iloc[-1]) if pd.notna(d["atr14"].iloc[-1]) else 0.0
    if atr <= 0:
        return None
    last = len(d) - 1
    intraday = tf not in ("1D", "1W", "1M")
    leg_min = config.AWARE_LEG_MIN_PCT / (3.0 if intraday else 1.0)
    # depth/height are measured in atr_ref units (pass the DAILY atr for intraday frames so the
    # daily-calibrated budgets apply); without a reference, fall back to frame-ATR ×3 scaling
    atr_d = atr_ref if (atr_ref and atr_ref > 0) else atr * (3.0 if intraday else 1.0)
    pb_max = config.AWARE_PB_MAX_ATR
    consol_max = config.AWARE_CONSOL_MAX_ATR

    cur = segs[-1]
    cur.ensure()
    cur_bars = cur.end - cur.start + 1
    done = [s.ensure() for s in segs if s.kind != "transition" and s is not cur][-6:]

    # ---- MAJOR swing anchors (chart-scale, not the last micro-pivot) ---------------------
    look_from = last - config.AWARE_MAJOR_LOOKBACK
    hi_ks = [k for k in range(len(pidx)) if ptype[k] == trendlab.PIVOT_HIGH and pidx[k] >= look_from]
    lo_ks = [k for k in range(len(pidx)) if ptype[k] == trendlab.PIVOT_LOW and pidx[k] >= look_from]
    maj_hi_k = max(hi_ks, key=lambda k: pprice[k], default=None)
    maj_lo_k = min(lo_ks, key=lambda k: pprice[k], default=None)
    majH = float(pprice[maj_hi_k]) if maj_hi_k is not None else None
    majL = float(pprice[maj_lo_k]) if maj_lo_k is not None else None
    chain = _chain(ptag)
    ext20 = st.get("atr_ext_20")
    gt50 = bool(st.get("gt_sma50"))
    stack = st.get("stack") or 0
    up_ctx = gt50 or stack >= 1                       # broad trend context still constructive
    dn_ctx = (st.get("gt_sma50") is False) and stack <= -1

    # off the major high: how deep, how long, how much of the prior leg given back
    depth_atr = retrace = span_hi = None
    lo_since = None
    if maj_hi_k is not None:
        lo_since = float(low[pidx[maj_hi_k]:].min())
        depth_atr = (majH - lo_since) / atr_d
        span_hi = last - pidx[maj_hi_k]
        base = min((float(pprice[k]) for k in range(len(pidx))
                    if ptype[k] == trendlab.PIVOT_LOW
                    and pidx[maj_hi_k] - 60 <= pidx[k] < pidx[maj_hi_k]), default=None)
        if base is not None and majH > base:
            retrace = (majH - lo_since) / (majH - base)
    height_atr = b_retrace = span_lo = None
    hi_since = None
    if maj_lo_k is not None:
        hi_since = float(high[pidx[maj_lo_k]:].max())
        height_atr = (hi_since - majL) / atr_d
        span_lo = last - pidx[maj_lo_k]
        top = max((float(pprice[k]) for k in range(len(pidx))
                   if ptype[k] == trendlab.PIVOT_HIGH
                   and pidx[maj_lo_k] - 60 <= pidx[k] < pidx[maj_lo_k]), default=None)
        if top is not None and top > majL:
            b_retrace = (hi_since - majL) / (top - majL)

    last_up = next((s for s in reversed(done) if s.kind == "up"), None)
    last_dn = next((s for s in reversed(done) if s.kind == "down"), None)

    # ---- fresh break off a completed range segment --------------------------------------
    def _fresh_break(up: bool):
        for R in reversed(done[-2:]):
            if R.kind != "range" or (R.end - R.start + 1) < config.AWARE_RANGE_MIN_BARS:
                continue
            r_hi, r_lo = _seg_range(d, R)
            level = r_hi if up else r_lo
            beyond = (close[R.end + 1:] > r_hi) if up else (close[R.end + 1:] < r_lo)
            if not beyond.any():
                continue
            b_star = R.end + 1 + int(np.argmax(beyond))
            age = last - b_star
            if age <= config.AWARE_FRESH_BARS and (c > r_hi if up else c < r_lo):
                return R, level, age
        return None

    # also: a fresh close beyond the MAJOR swing band (consolidations the segment FSM chopped up)
    def _fresh_swing_break():
        if majH is not None and c > majH and span_hi is not None and span_hi >= config.AWARE_RANGE_MIN_BARS:
            beyond = close[pidx[maj_hi_k] + 1:] > majH
            b_star = pidx[maj_hi_k] + 1 + int(np.argmax(beyond))
            age = last - b_star
            if beyond.any() and age <= config.AWARE_FRESH_BARS:
                return "up", majH, age, span_hi
        if majL is not None and c < majL and span_lo is not None and span_lo >= config.AWARE_RANGE_MIN_BARS:
            beyond = close[pidx[maj_lo_k] + 1:] < majL
            b_star = pidx[maj_lo_k] + 1 + int(np.argmax(beyond))
            age = last - b_star
            if beyond.any() and age <= config.AWARE_FRESH_BARS:
                return "down", majL, age, span_lo
        return None

    # ---- events + flags gathered regardless of regime ------------------------------------
    events, flags = [], []
    if ext20 is not None and abs(ext20) >= config.AWARE_EXT_ATR:
        flags.append("extended")
        events.append({"type": "extension", "age": 0, "level": None, "mag": _r(ext20),
                       "note": f"{ext20:+.1f} ATR from 20-EMA"})
    near_r = lv["resistance"][0] if lv.get("resistance") else None
    near_s = lv["support"][0] if lv.get("support") else None
    if near_r and near_r["dist_atr"] <= config.AWARE_AT_LEVEL_ATR:
        flags.append("at_resistance")
    if near_s and near_s["dist_atr"] <= config.AWARE_AT_LEVEL_ATR:
        flags.append("at_support")
    w = config.AWARE_RETEST_BARS
    for band, is_sup in [(b, True) for b in (lv.get("support") or [])[:2]] \
                      + [(b, False) for b in (lv.get("resistance") or [])[:2]]:
        if band["strength"] < 3:
            continue
        for i in range(max(0, len(d) - w), len(d)):
            if low[i] <= band["hi"] and high[i] >= band["lo"]:          # bar traded into the band
                if is_sup:
                    ok, bad = close[i] >= band["hi"], close[i] < band["lo"]
                    note = "held" if ok else "failed"
                else:
                    bad, ok = close[i] <= band["lo"], close[i] > band["hi"]
                    note = "broken" if ok else "rejected"
                if ok or bad:
                    events.append({"type": "retest_hold" if ok else "retest_fail",
                                   "age": last - i, "level": band["price"], "mag": band["strength"],
                                   "note": f"{band['kind']} {_fmt(band['price'])} ×{band['strength']} {note}"})
                    break

    def _near_highs() -> bool:
        oh = d["off_hi52_pct"].iloc[-1] if "off_hi52_pct" in d.columns else None
        if oh is not None and pd.notna(oh):
            return float(oh) >= -5.0
        return c >= 0.95 * float(close[-min(60, len(d)):].max())

    meta = {"leg_net_pct": _r(last_up.net_pct if last_up else None, 1),
            "depth_atr": _r(depth_atr), "retrace_pct": _r(retrace), "chain": chain,
            "ext20": _r(ext20), "age": cur_bars,
            "range_hi": None, "range_lo": None, "range_pos": None}

    def _set_band(hi, lo):
        meta.update(range_hi=_r(hi), range_lo=_r(lo),
                    range_pos=_r((c - lo) / (hi - lo)) if hi > lo else None)

    regime = invalidation = confirm = None
    phrase = ""

    # 0 — intraday only: a session gap beyond the prior session's extremes IS the structure
    # break (a gap-and-hold over yesterday's high must not read as "bounce in a 1h downtrend").
    gapbrk = None
    if intraday and isinstance(d.index, pd.DatetimeIndex) and len(d) > 30:
        dates = np.array([ts.date() for ts in d.index])
        today = dates[-1]
        tmask = dates == today
        if tmask.any() and (~tmask).any():
            prev_day = dates[~tmask][-1]
            pmask = dates == prev_day
            y_hi, y_lo = float(high[pmask].max()), float(low[pmask].min())
            o_today = float(d["open"].to_numpy(float)[int(np.argmax(tmask))])
            bars_in = int(tmask.sum())
            if o_today > y_hi and c > y_hi:
                gapbrk = ("up", y_hi, bars_in)
            elif o_today < y_lo and c < y_lo:
                gapbrk = ("down", y_lo, bars_in)
    if gapbrk is not None:
        side, level, bars_in = gapbrk
        regime = "breakout_fresh" if side == "up" else "breakdown_fresh"
        invalidation = level
        events.insert(0, {"type": "gap_break", "age": bars_in - 1, "level": _r(level), "mag": None,
                          "note": f"gapped {'over prior session high' if side == 'up' else 'under prior session low'}"})
        phrase = (f"gapped {'over' if side == 'up' else 'under'} the prior session "
                  f"{'high' if side == 'up' else 'low'} {_fmt(level)} and holding "
                  f"({bars_in} bars); back {'below' if side == 'up' else 'above'} {_fmt(level)} fails")
        meta["age"] = bars_in - 1

    # 1/2 — fresh break (completed range segment, else the major swing band)
    br = _fresh_break(up=True) if regime is None else None
    bd = _fresh_break(up=False) if regime is None and br is None else None
    swb = _fresh_swing_break() if regime is None and br is None and bd is None else None
    if regime is not None:
        pass
    elif br is not None or (swb is not None and swb[0] == "up"):
        R_bars = (br[0].end - br[0].start + 1) if br else swb[3]
        level, age = (br[1], br[2]) if br else (swb[1], swb[2])
        regime, invalidation = "breakout_fresh", level
        events.insert(0, {"type": "breakout", "age": age, "level": _r(level), "mag": None,
                          "note": f"broke a {R_bars}-bar consolidation"})
        room = (f"; next R {_fmt(near_r['price'])} ×{near_r['strength']} {near_r['dist_atr']:.1f} ATR up"
                if near_r else "; no resistance overhead")
        phrase = (f"day-{age} breakout over {_fmt(level)} from a {R_bars}-bar consolidation"
                  f"{room}; back inside < {_fmt(level)} fails")
        meta["age"] = age
    elif bd is not None or (swb is not None and swb[0] == "down"):
        R_bars = (bd[0].end - bd[0].start + 1) if bd else swb[3]
        level, age = (bd[1], bd[2]) if bd else (swb[1], swb[2])
        regime, invalidation = "breakdown_fresh", level
        events.insert(0, {"type": "breakdown", "age": age, "level": _r(level), "mag": None,
                          "note": f"broke a {R_bars}-bar consolidation"})
        phrase = (f"day-{age} breakdown under {_fmt(level)} from a {R_bars}-bar consolidation;"
                  f" reclaim > {_fmt(level)} negates")
        meta["age"] = age

    # 3/4 — impulse leg in progress (fast, young, efficient)
    elif (cur.kind == "up" and cur.slope_atr >= config.AWARE_IMPULSE_SLOPE and cur.er >= 0.5
          and cur_bars <= config.AWARE_IMPULSE_MAX_BARS):
        regime = "impulse_up"
        invalidation = majL if (majL is not None and majL < c) else None
        if cur_bars >= config.AWARE_LATE_BARS or chain >= config.AWARE_LATE_CHAIN:
            flags.append("late")
        phrase = (f"impulse up — {cur_bars} bars +{cur.net_pct:.1f}%"
                  + (", extended" if "extended" in flags else "") + (", late" if "late" in flags else "")
                  + (f"; invalid < {_fmt(invalidation)}" if invalidation else ""))
    elif (cur.kind == "down" and cur.slope_atr <= -config.AWARE_IMPULSE_SLOPE and cur.er >= 0.5
          and cur_bars <= config.AWARE_IMPULSE_MAX_BARS):
        regime = "impulse_down"
        invalidation = majH if (majH is not None and majH > c) else None
        if cur_bars >= config.AWARE_LATE_BARS or chain <= -config.AWARE_LATE_CHAIN:
            flags.append("late")
        phrase = (f"impulse down — {cur_bars} bars {cur.net_pct:.1f}%"
                  + (f"; reclaim > {_fmt(invalidation)} negates" if invalidation else ""))

    # 5/6 — steady trend leg (older/slower than an impulse but clean)
    elif cur.kind == "up" and cur.net_pct > 0 and cur.er >= config.AWARE_TREND_MIN_ER and up_ctx:
        regime = "trend_up"
        invalidation = majL if (majL is not None and majL < c) else None
        if chain >= config.AWARE_LATE_CHAIN:
            flags.append("late")
        phrase = (f"uptrend — {cur_bars} bars +{cur.net_pct:.1f}%, grinding"
                  + (", extended" if "extended" in flags else "")
                  + (f"; invalid < {_fmt(invalidation)}" if invalidation else ""))
    elif cur.kind == "down" and cur.net_pct < 0 and cur.er >= config.AWARE_TREND_MIN_ER and dn_ctx:
        regime = "trend_down"
        invalidation = majH if (majH is not None and majH > c) else None
        phrase = (f"downtrend — {cur_bars} bars {cur.net_pct:.1f}%"
                  + (f"; reclaim > {_fmt(invalidation)} negates" if invalidation else ""))

    # 7 — distribution: consolidating near highs after an advance WITH fresh cracks
    elif (majH is not None and span_hi is not None and span_hi >= config.AWARE_RANGE_MIN_BARS
          and up_ctx and _near_highs() and depth_atr is not None
          and depth_atr <= consol_max
          and (cur.subtype == "expanding"
               or (bool((close[max(0, last - 10):] > majH).any()) and c < majH)          # failed poke
               or (any(t in ("LL", "DB LL") for t in ptag[-2:])                          # fresh crack low
                   and meta.get("range_pos") is None))):                                  # (evaluated below)
        # NOTE: fresh-crack branch re-checks position after _set_band
        r_hi, r_lo = majH, lo_since
        _set_band(r_hi, r_lo)
        crack = any(t in ("LL", "DB LL") for t in ptag[-2:]) and (meta["range_pos"] or 1) <= 0.33
        failed = bool((close[max(0, last - 10):] > majH).any()) and c < majH
        if cur.subtype == "expanding" or failed or crack:
            regime, invalidation, confirm = "distribution", r_hi, r_lo
            why = "widening swings" if cur.subtype == "expanding" else ("failed breakout" if failed else "fresh lower low")
            phrase = (f"distribution risk — {span_hi} bars under {_fmt(r_hi)}, {why};"
                      f" re-break > {_fmt(r_hi)} negates, < {_fmt(r_lo)} confirms")
            meta["age"] = span_hi

    # 8/9 — pullback / bounce at swing scale (short + shallow vs the prior advance/decline)
    if regime is None and (majH is not None and span_hi is not None
                           and 0 < span_hi < config.AWARE_RANGE_MIN_BARS
                           and depth_atr is not None and depth_atr <= pb_max
                           and (retrace is None or retrace <= config.AWARE_PB_MAX_RETRACE)
                           and up_ctx and (last_up is None or last_up.net_pct >= leg_min or majH > c)):
        regime = "pullback_in_uptrend"
        invalidation = min([x for x in (majL, near_s["lo"] if near_s else None) if x is not None],
                           default=None)
        confirm = majH
        sup = f", holds {_fmt(near_s['price'])} ×{near_s['strength']}" if near_s else ""
        phrase = (f"pullback in uptrend — {span_hi} bars, {depth_atr:.1f} ATR deep"
                  + (f" ({retrace:.0%} retrace)" if retrace is not None else "") + sup
                  + (f"; invalid < {_fmt(invalidation)}" if invalidation else "")
                  + (f", resumes > {_fmt(confirm)}" if confirm else ""))
        meta["age"] = span_hi
    if regime is None and (majL is not None and span_lo is not None
                           and 0 < span_lo < config.AWARE_RANGE_MIN_BARS
                           and height_atr is not None and height_atr <= pb_max
                           and (b_retrace is None or b_retrace <= config.AWARE_PB_MAX_RETRACE)
                           and dn_ctx):
        regime = "bounce_in_downtrend"
        invalidation, confirm = majH, majL
        phrase = (f"bounce in downtrend — {span_lo} bars, {height_atr:.1f} ATR off the low"
                  + (f" ({b_retrace:.0%} retrace)" if b_retrace is not None else "")
                  + (f"; short side resumes < {_fmt(confirm)}" if confirm else "")
                  + (f", reclaim > {_fmt(invalidation)} negates" if invalidation else ""))
        meta["age"] = span_lo

    # 10 — consolidation at swing scale (longer sideways band after an advance/decline)
    if regime is None and (majH is not None and majL is not None and span_hi is not None
                           and span_hi >= config.AWARE_RANGE_MIN_BARS and depth_atr is not None
                           and depth_atr <= config.AWARE_CONSOL_MAX_ATR):
        r_hi, r_lo = majH, lo_since
        _set_band(r_hi, r_lo)
        regime, confirm = "range", r_hi
        flags.append("after_advance" if up_ctx else ("after_decline" if dn_ctx else ""))
        flags = [f for f in flags if f]
        kind = {"rectangle": "flat", "contracting": "tightening", "expanding": "widening"}.get(cur.subtype, "")
        phrase = (f"{span_hi}-bar {kind + ' ' if kind else ''}consolidation {_fmt(r_lo)}–{_fmt(r_hi)}"
                  + (" after an advance" if "after_advance" in flags else
                     " after a decline" if "after_decline" in flags else "")
                  + f" (now {int(100 * (meta['range_pos'] or 0))}% up the band);"
                  + f" break > {_fmt(r_hi)} / < {_fmt(r_lo)} decides")
        meta["age"] = span_hi

    if regime is None:
        regime = "unclear"
        phrase = f"no clean structure ({cur.kind}{' ' + cur.subtype if cur.subtype else ''}, {cur_bars} bars)"

    if "at_resistance" in flags and near_r:
        phrase += f"; AT resistance {_fmt(near_r['price'])} ×{near_r['strength']}"
    elif "at_support" in flags and near_s:
        phrase += f"; AT support {_fmt(near_s['price'])} ×{near_s['strength']}"

    return {"regime": regime, "flags": flags, "events": events[:6],
            "invalidation": _r(invalidation), "confirm": _r(confirm),
            "phrase": phrase, "meta": meta}
