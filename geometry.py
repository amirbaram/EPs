"""Pivot-template geometry — the shared engine for pattern detectors.

Every template matches on trendlab's causal zigzag pivots (one source of truth with the charts
and the regime engine), never on raw bars. Detectors in setups.py call these and add their own
liquidity/regime/volume gates; the awareness layer reuses topping_flags() as a cheap warning.

    legs(d)            -> alternating pivot legs [(lo_i, lo_p, hi_i, hi_p, dir), ...]
    leg_retrace(d)     -> health of the CURRENT giveback vs the prior up-leg (38.2/50% rules)
    fit_line(points)   -> (slope, intercept, r2)
    cup_handle(d)      -> cup-with-handle match dict | None
    double_top(d)      -> double/triple-top match dict | None
    head_shoulders(d, inverse=False) -> H&S / inverse-H&S match dict | None
    topping_flags(d)   -> {"double_top": bool, "hs": bool}  (memoized, warning-grade)

Pattern templates run on a COARSER pivot pass (atr_fraction, default config.GEO_ATR_FRACTION)
than the default 0.2 zigzag — the fine pass fragments cup rims and shoulders into micro-swings.
All causal: only confirmed pivots up to the last bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
import framecache
import trendlab

_TOP_CACHE = framecache.FrameCache()


def fit_line(points: list[tuple[int, float]]) -> tuple[float, float, float]:
    """(slope, intercept, r2) through (bar_idx, price) points."""
    if len(points) < 2:
        i, p = (points[0] if points else (0, 0.0))
        return 0.0, float(p), 1.0
    ix = np.array([i for i, _ in points], float)
    pr = np.array([p for _, p in points], float)
    slope, intercept = np.polyfit(ix, pr, 1)
    fit = slope * ix + intercept
    ss_tot = float(((pr - pr.mean()) ** 2).sum())
    r2 = 1.0 - float(((pr - fit) ** 2).sum()) / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(intercept), round(r2, 3)


def _pivots(d: pd.DataFrame, atr_fraction: float | None = None):
    frac = atr_fraction if atr_fraction is not None else config.GEO_ATR_FRACTION
    return trendlab.pivots(d, frac)


def legs(d: pd.DataFrame, lookback: int = 250, atr_fraction: float | None = None) -> list[tuple]:
    """Alternating up/down legs from the coarse pivots within `lookback` bars:
    [(start_i, start_p, end_i, end_p, "up"|"down"), ...] in time order."""
    pidx, pprice, ptype, _ = _pivots(d, atr_fraction)
    cut = len(d) - lookback
    out = []
    for k in range(1, len(pidx)):
        if pidx[k] < cut:
            continue
        direction = "up" if ptype[k] == trendlab.PIVOT_HIGH else "down"
        out.append((int(pidx[k - 1]), float(pprice[k - 1]), int(pidx[k]), float(pprice[k]), direction))
    return out


def leg_retrace(d: pd.DataFrame, atr_fraction: float | None = None) -> dict | None:
    """Health of the current giveback vs the prior UP leg (TCG retracement guide):
    retrace <=38.2% healthy / 38.2-50% gray / >50% reversal risk. Also flags rising volume into
    the retracement (exhaustion tell). None if no completed up-leg."""
    lg = legs(d, atr_fraction=atr_fraction)
    ups = [x for x in lg if x[4] == "up"]
    if not ups:
        return None
    lo_i, lo_p, hi_i, hi_p, _ = ups[-1]
    if hi_p <= lo_p:
        return None
    low_since = float(d["low"].iloc[hi_i:].min())
    retrace = (hi_p - low_since) / (hi_p - lo_p)
    health = ("healthy" if retrace <= 0.382 else "gray" if retrace <= 0.50 else "reversal_risk")
    vol_rising = None
    if "volume" in d.columns and len(d) - 1 > hi_i:
        pb = d["volume"].iloc[hi_i + 1:]
        leg = d["volume"].iloc[lo_i:hi_i + 1]
        if len(pb) >= 2 and len(leg) >= 2:
            vol_rising = bool(pb.mean() > leg.mean())
    return {"leg_lo": round(lo_p, 2), "leg_hi": round(hi_p, 2), "leg_bars": hi_i - lo_i,
            "retrace_pct": round(float(retrace), 3), "health": health,
            "pb_bars": len(d) - 1 - hi_i, "pb_vol_rising": vol_rising}


# ---------------------------------------------------------------- templates
#
# All templates follow Amir's verified rules (data/validation/verdicts_amir_1.md):
# prior-trend CONTEXT gates (continuation vs reversal semantics), TRUE raw-bar extremes for
# rims/peaks/heads (coarse pivots only locate the neighborhoods), depth measured against the
# PRIOR MOVE, and a confirmation/invalidation lifecycle (an unconfirmed top is just similar
# highs in an uptrend).

def _advance_into(d: pd.DataFrame, peak_i: int, lookback: int = 90) -> tuple[float, int] | None:
    """The up-move INTO bar peak_i: (advance_pct, start_i) from the lowest low in the prior
    `lookback` bars to the peak. None if the peak isn't the top of a real advance."""
    lo0 = max(0, peak_i - lookback)
    lows = d["low"].to_numpy(float)[lo0:peak_i + 1]
    if not len(lows):
        return None
    s_rel = int(lows.argmin())
    base = float(lows[s_rel])
    peak = float(d["high"].iloc[peak_i])
    if base <= 0 or peak <= base:
        return None
    return (peak / base - 1) * 100, lo0 + s_rel


def cup_handle(d: pd.DataFrame) -> dict | None:
    """Cup-with-handle (CONTINUATION): prior advance >= CH_PRIOR_ADV into the LEFT RIM (= the
    true high of that move); cup retraces only its upper part (<= CH_MAX_RETRACE of the advance,
    <=1/3 ideal); single-basin U (not V, not multi-hump); RIGHT RIM = the true high of the
    recovery; handle = shallow dip off the right rim. Trigger = handle high."""
    pidx, pprice, ptype, _ = _pivots(d)
    highs_piv = [int(pidx[k]) for k in range(len(pidx)) if ptype[k] == trendlab.PIVOT_HIGH]
    if len(highs_piv) < 2 or len(d) < 80:
        return None
    last = len(d) - 1
    high = d["high"].to_numpy(float)
    low = d["low"].to_numpy(float)
    close = d["close"].to_numpy(float)
    for l_piv in reversed(highs_piv):
        span_all = last - l_piv
        if span_all < config.CH_CUP_MIN + config.CH_HANDLE_MIN:
            continue
        if span_all > config.CH_CUP_MAX + config.CH_HANDLE_MAX:
            break
        # LEFT RIM = true high around the pivot neighborhood
        l_i = l_piv - 3 + int(high[max(0, l_piv - 3):l_piv + 4].argmax())
        l_p = float(high[l_i])
        adv = _advance_into(d, l_i)
        if adv is None or adv[0] < config.CH_PRIOR_ADV:
            continue                                    # continuation pattern needs the move
        if float(high[adv[1]:l_i].max(initial=0)) > l_p:  # rim must BE the move's top
            continue
        adv_pts = l_p - float(low[adv[1]])
        # cup bottom
        seg_lo = low[l_i:last + 1]
        lo_rel = int(seg_lo.argmin())
        cup_low_i = l_i + lo_rel
        cup_low = float(seg_lo[lo_rel])
        depth_pts = l_p - cup_low
        retrace = depth_pts / adv_pts if adv_pts > 0 else 9
        if retrace > config.CH_MAX_RETRACE or (depth_pts / l_p) * 100 < config.CH_DEPTH_MIN:
            continue                                    # too deep vs the move / too trivial
        # RIGHT RIM = true high of the recovery from the bottom
        if cup_low_i + 2 >= last:
            continue
        rec = high[cup_low_i:last + 1]
        r_rel = int(rec.argmax())
        r_i = cup_low_i + r_rel
        r_p = float(rec[r_rel])
        cup_span = r_i - l_i
        if not (config.CH_CUP_MIN <= cup_span <= config.CH_CUP_MAX):
            continue
        if abs(r_p / l_p - 1) * 100 > config.CH_RIM_TOL:
            continue
        if not (cup_span / 4 <= (cup_low_i - l_i) <= 3 * cup_span / 4):
            continue                                    # basin roughly central
        # U not V: enough time near the lows AND a single basin (<=2 interior pivot lows)
        band = cup_low + 0.25 * (min(l_p, r_p) - cup_low)
        if (low[l_i:r_i + 1] <= band).sum() < max(3, cup_span // 8):
            continue
        interior_lows = [k for k in range(len(pidx))
                         if ptype[k] == trendlab.PIVOT_LOW and l_i < pidx[k] < r_i]
        if len(interior_lows) > 2:
            continue                                    # multi-hump correction, not a cup
        # handle: shallow dip off the TRUE right rim
        h_bars = last - r_i
        if not (config.CH_HANDLE_MIN <= h_bars <= config.CH_HANDLE_MAX):
            continue
        h_low = float(low[r_i:last + 1].min())
        h_pull_move = (r_p - h_low) / (r_p - cup_low) if r_p > cup_low else 9
        h_pull_pct = (r_p - h_low) / r_p * 100
        if h_pull_pct < config.CH_HANDLE_PULL_MIN or h_pull_move > 0.5 \
                or h_pull_pct > config.CH_HANDLE_PULL_MAX:
            continue                                    # handle keeps the upper half of the cup
        # the handle's structural high EXCLUDES today — today is the TEST bar. With today included,
        # close > h_high was impossible and the breakout state never fired (same blindness as
        # flat_base / HTF / QMB; found when the registry build produced 0 fires in 600 symbols).
        h_high = float(high[r_i + 1:last].max()) if last > r_i + 1 else r_p
        if h_high > r_p * 1.02:
            continue
        c = float(close[last])
        depth_pct = depth_pts / l_p * 100
        return {"l_rim": (l_i, round(l_p, 2)), "r_rim": (r_i, round(r_p, 2)),
                "cup_low": (cup_low_i, round(cup_low, 2)), "depth_pct": round(depth_pct, 1),
                "retrace_of_move": round(retrace, 2), "prior_adv_pct": round(adv[0], 1),
                "adv_start": adv[1], "cup_bars": cup_span, "handle_bars": h_bars,
                "handle_low": round(h_low, 2), "handle_pull_pct": round(h_pull_pct, 1),
                "trigger": round(max(h_high, r_p * 0.999), 2),
                "target": round(r_p + depth_pts, 2),
                "state": "breakout" if c > h_high else "building"}
    return None


def double_top(d: pd.DataFrame) -> dict | None:
    """Double/triple top (REVERSAL): the peaks must be the ACTUAL TOPS — the two highest highs of
    the lookback with NO intervening/после higher high; prior advance into P1. States: building
    (unconfirmed — just similar running-top highs) / breakdown (close < valley = CONFIRMED).
    Candidates with an intervening higher high belong to the H&S template, not here."""
    look = config.DT_LOOKBACK
    lo0 = max(0, len(d) - look)
    high = d["high"].to_numpy(float)
    low = d["low"].to_numpy(float)
    win = high[lo0:]
    last = len(d) - 1
    # P2 = the LAST time the window max (within tol) was printed; P1 = the first
    top = float(win.max())
    at_top = [lo0 + i for i in range(len(win)) if win[i] >= top * (1 - config.DT_TOL / 100)]
    if len(at_top) < 2:
        return None
    p1_i, p2_i = at_top[0], at_top[-1]
    if p2_i - p1_i < config.DT_MIN_SPACING or last - p2_i > config.DT_MAX_AGE:
        return None
    between = high[p1_i + 1:p2_i]
    if len(between) and float(between.max()) >= min(high[p1_i], high[p2_i]):
        return None                                     # intervening high >= peaks -> H&S land
    adv = _advance_into(d, p1_i)
    if adv is None or adv[0] < config.DT_PRIOR_ADV:
        return None
    valley = float(low[p1_i:p2_i + 1].min())
    p1_p, p2_p = float(high[p1_i]), float(high[p2_i])
    height = (max(p1_p, p2_p) - valley) / max(p1_p, p2_p) * 100
    if height < config.DT_MIN_VALLEY:
        return None
    close = float(d["close"].iloc[-1])
    if close > max(p1_p, p2_p):
        return None                                     # new high -> pattern invalidated
    v = d["volume"].to_numpy(float) if "volume" in d.columns else None
    vol_lighter = (bool(np.nanmean(v[max(0, p2_i - 2):p2_i + 1])
                        < np.nanmean(v[max(0, p1_i - 2):p1_i + 1])) if v is not None else None)
    peaks = min(len([i for i in at_top if i == p1_i or i - p1_i >= config.DT_MIN_SPACING]), 3)
    return {"p1": (p1_i, round(p1_p, 2)), "p2": (p2_i, round(p2_p, 2)), "peaks": peaks,
            "valley": round(valley, 2), "height_pct": round(height, 1),
            "prior_adv_pct": round(adv[0], 1), "vol_lighter_p2": vol_lighter,
            "trigger": round(valley, 2),
            "target": round(valley * (1 - min(height, 40) / 100), 2),
            "state": "breakdown" if close < valley else "building"}


def head_shoulders(d: pd.DataFrame, inverse: bool = False) -> dict | None:
    """(Inverse) head & shoulders — v3 per Amir's pass #2. The HEAD is the TRUE window extreme;
    SHOULDERS must be ADJACENT swings (within HS_ARM_MAX bars each side, arms roughly symmetric);
    the NECKLINE runs through the two raw-bar reaction extremes between LS-head and head-RS (not
    distant pivots), slope-capped. Confirmed on the neckline break; invalidated once the right
    shoulder is taken out."""
    look = config.HS_LOOKBACK
    lo0 = max(0, len(d) - look)
    high = d["high"].to_numpy(float)
    low = d["low"].to_numpy(float)
    last = len(d) - 1
    atr = float(d["atr14"].iloc[-1]) if pd.notna(d["atr14"].iloc[-1]) else None
    if not atr or atr <= 0:
        return None
    if inverse:
        h_i = lo0 + int(low[lo0:].argmin()); h_p = float(low[h_i])
    else:
        h_i = lo0 + int(high[lo0:].argmax()); h_p = float(high[h_i])
    if h_i - lo0 < 5 or last - h_i < 3:
        return None
    pidx, pprice, ptype, _ = _pivots(d)
    want = trendlab.PIVOT_LOW if inverse else trendlab.PIVOT_HIGH
    arm = config.HS_ARM_MAX
    left = [(int(pidx[k]), float(pprice[k])) for k in range(len(pidx))
            if ptype[k] == want and h_i - arm <= pidx[k] < h_i - 2]
    right = [(int(pidx[k]), float(pprice[k])) for k in range(len(pidx))
             if ptype[k] == want and h_i + 2 < pidx[k] <= h_i + arm]
    if not left or not right:
        return None
    # shoulders = the swings ADJACENT to the head (nearest flanking pivots), not the arm extremes
    ls_i, ls_p = max(left, key=lambda x: x[0])
    rs_i, rs_p = min(right, key=lambda x: x[0])
    if last - rs_i > config.HS_MAX_AGE:
        return None
    l_span, r_span = h_i - ls_i, rs_i - h_i
    if not (1 / config.HS_SYM_RATIO <= l_span / max(r_span, 1) <= config.HS_SYM_RATIO):
        return None                                     # arms wildly asymmetric = not the pattern
    beyond = (min(ls_p, rs_p) - h_p) if inverse else (h_p - max(ls_p, rs_p))
    if beyond < config.HS_HEAD_MIN_ATR * atr:
        return None
    if abs(ls_p / rs_p - 1) * 100 > config.HS_SHOULDER_TOL:
        return None
    close = float(d["close"].iloc[-1])
    post_rs = high[rs_i + 1:] if not inverse else low[rs_i + 1:]
    if len(post_rs) and ((not inverse and float(post_rs.max()) > rs_p)
                         or (inverse and float(post_rs.min()) < rs_p)):
        return None                                     # right shoulder taken out = invalidated
    adv = _advance_into(d, h_i)
    if not inverse:
        if adv is None or adv[0] < config.HS_PRIOR_MOVE:
            return None
        variant = "reversal"
    else:
        dec_hi = float(high[max(0, h_i - 90):h_i + 1].max())
        decline = (dec_hi / h_p - 1) * 100 if h_p > 0 else 0
        if decline >= config.HS_PRIOR_MOVE:
            variant = "reversal"
        elif close > h_p and adv is not None and adv[0] >= config.HS_PRIOR_MOVE:
            variant = "continuation"
        else:
            return None
    # NECKLINE through the two reaction extremes adjacent to the head — anchored on bar BODIES
    # (traders draw necklines through the reaction closes, not spike wicks; Amir's pass #2)
    body_hi = np.maximum(d["open"].to_numpy(float), d["close"].to_numpy(float))
    body_lo = np.minimum(d["open"].to_numpy(float), d["close"].to_numpy(float))
    if inverse:
        n1_i = ls_i + int(body_hi[ls_i:h_i + 1].argmax()); n1_p = float(body_hi[n1_i])
        n2_i = h_i + int(body_hi[h_i:rs_i + 1].argmax()); n2_p = float(body_hi[n2_i])
    else:
        n1_i = ls_i + int(body_lo[ls_i:h_i + 1].argmin()); n1_p = float(body_lo[n1_i])
        n2_i = h_i + int(body_lo[h_i:rs_i + 1].argmin()); n2_p = float(body_lo[n2_i])
    if n2_i <= n1_i:
        return None
    slope = (n2_p - n1_p) / (n2_i - n1_i)
    if abs(slope) > config.HS_NECK_SLOPE_ATR * atr:
        return None
    # trigger level = the SHALLOWER (higher) of the two reactions — where the trader draws it
    # (Amir pass #3: SMCI 29.3, SATS/ECHO ~117, CBOE ~275, DLTR "too low" all = the higher trough)
    neckline = max(n1_p, n2_p) if not inverse else max(n1_p, n2_p)
    if (inverse and neckline <= h_p + 0.5 * atr) or (not inverse and neckline >= h_p - 0.5 * atr):
        return None
    depth = abs(neckline - h_p)
    broke = close > neckline if inverse else close < neckline
    return {"ls": (ls_i, round(ls_p, 2)), "head": (h_i, round(h_p, 2)), "rs": (rs_i, round(rs_p, 2)),
            "neck1": (n1_i, round(n1_p, 2)), "neck2": (n2_i, round(n2_p, 2)),
            "neckline": round(float(neckline), 2), "neck_slope": round(float(slope), 4),
            "variant": variant, "trigger": round(float(neckline), 2),
            "target": round(float(neckline + min(depth, 0.4 * neckline)) if inverse
                            else float(neckline - min(depth, 0.4 * neckline)), 2),
            "state": ("breakout" if inverse else "breakdown") if broke else "building"}


def topping_flags(d: pd.DataFrame) -> dict:
    """Cheap memoized warning for LONG hits: is a double-top or H&S template present (building or
    triggered)? A flag, never a gate."""
    cached = _TOP_CACHE.get(d)
    if cached is not None:
        return cached
    dt = double_top(d)
    hs = head_shoulders(d, inverse=False)
    out = {"double_top": bool(dt), "hs": bool(hs)}
    _TOP_CACHE.put(d, out)
    return out
