"""Trend/range segmentation — the regime engine the scanner gates on.

Classifies a chart into a SEQUENCE of segments: up/down trends (with strength +
clarity scores) and ranges (rectangle / contracting / expanding), plus
transitions. Ported from the Pine "SwingsTimes" indicator so pivots and
structure are detected the way that indicator detects them.

Method:
  1. Pivots: causal ATR-fraction swing detector. delta = ATR(14) * ATR_FRACTION.
     In an up-leg, the first bar with low < low[1] - delta confirms the running
     high as a pivot and flips to a down-leg (mirror for lows); outside bars are
     disambiguated by candle color. A pivot is confirmed at the first opposing
     break (~1 bar later, NO centered look-ahead), then tagged HH/LH/HL/LL (or
     DT/DB for equal extremes within delta) vs. the last same-type pivot.
  2. Structure: a tag-driven FSM evaluated EVERY bar against the last high pivot,
     last low pivot, and the live bar — so uptrend/downtrend/range and their
     breaks fire intrabar, without waiting for the next pivot to complete.
        uptrend   = last low is HL AND today's high breaks the last pivot high
        downtrend = mirror;  rectangle = DT high + DB low inside;
        contracting = LH + HL inside;  expanding = HH + LL inside.
  3. Per-segment metrics (unchanged): annualized log-regression slope (strength),
     R^2 + Kaufman efficiency ratio (clarity in [0,1]), ATR-normalized net slope.
  4. Tag 'pole' = a fast, clean, large up-thrust (screening hook).

Causality: every step is left-to-right and uses only confirmed pivots + the
current bar, so truncating at a date yields a valid point-in-time labeling.

Usage:
    python trendlab.py NVDA                  # last 500 bars from data/bars cache
    python trendlab.py NVDA --bars 0         # full history
    python trendlab.py path/to/bars.csv      # any OHLC csv (date,open,high,low,close)
    python trendlab.py NVDA --atr-fraction 0.3 --open
"""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
import datastore
import framecache

# per-frame trendlab memo, keyed by frame identity (NOT df.attrs — see framecache.py: caching on
# .attrs makes pandas deep-copy the payload on every derived op). Evicts when the frame is GC'd.
_TRENDLAB_CACHE = framecache.FrameCache()

# ---- swing detection (Pine "SwingsTimes" port)
ATR_LEN = 14
ATR_FRACTION = config.ATR_FRACTION   # delta = ATR(ATR_LEN) * fraction = min meaningful move

PIVOT_HIGH = 1
PIVOT_LOW = -1

# structure FSM states (0/1/2/5/6/7 match the Pine indicator; 8/9/10 are app-only, no Pine yet)
(S_UNKNOWN, S_UP, S_DOWN, S_RECT, S_EXP, S_CONT, S_DCHAN, S_BRKUP, S_BRKDN
 ) = 0, 1, 2, 5, 6, 7, 8, 9, 10
_STRUCT_KIND = {
    S_UP: ("up", ""), S_DOWN: ("down", ""),
    S_RECT: ("range", "rectangle"), S_EXP: ("range", "expanding"),
    S_CONT: ("range", "contracting"), S_UNKNOWN: ("transition", ""),
    S_DCHAN: ("channel", "descending"),
    S_BRKUP: ("broken", "uptrend"), S_BRKDN: ("broken", "downtrend"),
}

# ---- 'pole' tag: fast, clean, large up-thrust
POLE_MIN_GAIN_PCT = 40.0
POLE_MAX_BARS = 60
POLE_MIN_ER = 0.50


@dataclass
class Swing:
    idx: int          # bar position
    kind: str         # 'H' | 'L'
    price: float
    provisional: bool = False   # active developing extreme, not yet confirmed


@dataclass
class Segment:
    start: int
    end: int                    # inclusive bar positions
    kind: str                   # 'up' | 'down' | 'range' | 'transition'
    subtype: str = ""           # ranges: 'rectangle' | 'contracting' | 'expanding'
    bars: int = 0
    net_pct: float = 0.0        # close-to-close % over the segment
    ann_pct: float = 0.0        # annualized regression slope, %
    slope_atr: float = 0.0      # net progress in ATRs per bar
    r2: float = 0.0
    er: float = 0.0
    clarity: float = 0.0        # (r2 + er) / 2
    tags: list = field(default_factory=list)
    open_ended: bool = False
    _d: object = field(default=None, repr=False, compare=False)     # lazy-metrics source frame
    _atr: object = field(default=None, repr=False, compare=False)   # lazy-metrics ATR array
    _done: bool = field(default=False, repr=False, compare=False)

    def ensure(self) -> "Segment":
        """Compute the regression metrics (r2/clarity/ann/er + net/er pole tag) on FIRST access. A scan
        only reads a couple of recent segments per ticker, so deferring this avoids ~450 log-fits/ticker
        (the dominant scan cost). Idempotent."""
        if not self._done:
            self._done = True
            if self._d is not None:
                _metrics(self, self._d, self._atr)
        return self


def atr_series(d: pd.DataFrame, n: int = ATR_LEN) -> np.ndarray:
    """Wilder's ATR (RMA of true range) — matches Pine's ta.atr."""
    h, l, c = d["high"], d["low"], d["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=1).mean().to_numpy()


def _eval_structure(P_price, P_type, P_tag, cur, hi, lo, delta, swing_dir, run_lo, run_hi,
                    eq_delta=None) -> int:
    """Pine structure FSM for one bar: last two pivots + live (hi,lo) + prior state.
    `delta` = ATR*fraction, the swing-REVERSAL distance (provisional-break lock below).
    `eq_delta` = the double-bottom/top EQUALITY tolerance the pivot TAGGING uses. Kept for the
    signature but the HOLD forgiveness stays on `delta`: hold answers "how far past the anchor
    is a real VIOLATION" (a break concept, reversal scale) — tying it to the equality tolerance
    made the FSM drop trends on 0.2-ATR dips the swing machine tolerates at 0.7 (measured
    2026-07-09: 1h agree .364->.319, flips 4.8->8.9). With one shared knob (the app default)
    delta == eq_delta, so the 2026-07-05 tag/hold alignment still holds there.
    `swing_dir`/`run_lo`/`run_hi` = the DEVELOPING (unconfirmed) swing's direction and running
    extreme, used for the provisional-expanding read below (so the lock holds across the swing,
    not just on the single break bar)."""
    eq_delta = delta if eq_delta is None else eq_delta
    sz = len(P_type)
    if sz < 2:
        return S_UNKNOWN
    if P_type[sz - 1] == PIVOT_HIGH:
        iH, iL = sz - 1, sz - 2
    else:
        iL, iH = sz - 1, sz - 2
    lastHighPrice, lastLowPrice = P_price[iH], P_price[iL]
    lastHighTag, lastLowTag = P_tag[iH], P_tag[iL]

    lastLowIsHL = lastLowTag in ("HL", "DB HL", "~HL")
    uptrendEntry = lastLowIsHL and hi > lastHighPrice and lo >= lastLowPrice
    # hold does NOT require the trend-confirming tag on the far pivot: right after an
    # entry the new extreme is not a confirmed pivot yet, so demanding HH/LL there made
    # the state drop to CONT on any inside bar (Amir 2026-07-05: "an inside bar doesn't
    # break the trend"). Trend now persists until the anchoring pivot is violated.
    # hold tolerates a DOUBLE-BOTTOM undercut of the anchor within `delta` (Amir 2026-07-05,
    # SPY/QQQ 2026-05-19: the pivot was TAGGED 'DB HL' — a higher low — yet the strict
    # lo >= lastLowPrice floor failed on the same within-delta dip and dropped the state to
    # transition; tagging and hold now use the SAME DB tolerance).
    uptrendHold = cur == S_UP and lastLowIsHL and lo >= lastLowPrice - delta

    lastHighIsLH = lastHighTag in ("LH", "DT LH", "~LH")
    downtrendEntry = lastHighIsLH and lo < lastLowPrice and hi <= lastHighPrice
    downtrendHold = cur == S_DOWN and lastHighIsLH and hi <= lastHighPrice + delta

    inside = hi <= lastHighPrice and lo >= lastLowPrice
    isHighDT = lastHighTag == "DT HH" or lastHighTag == "DT LH"
    isLowDB = lastLowTag == "DB HL" or lastLowTag == "DB LL"
    isRect = isHighDT and isLowDB and inside
    isContr = (lastHighTag in ("LH", "DT LH", "~LH")
               and lastLowTag in ("HL", "DB HL", "~HL") and inside)
    isExp = lastHighTag in ("HH", "~HH") and lastLowTag in ("LL", "~LL") and inside

    if uptrendEntry or uptrendHold:
        return S_UP
    if downtrendEntry or downtrendHold:
        return S_DOWN
    # DESCENDING CHANNEL (Amir 2026-07-05): last three CONFIRMED pivots are HH -> LL -> LH,
    # while NOT a confirmed downtrend (that check ran just above). A lower high has CONFIRMED
    # after a lower low made off a higher high — the descending-channel / bull-flag pullback
    # that has not yet broken down into a downtrend. The LH must be a confirmed pivot (Amir:
    # a developing lower high does NOT qualify). Checked before the range/expanding states.
    dchanConfirmed = (P_type[sz - 1] == PIVOT_HIGH and lastHighTag in ("LH", "DT LH", "~LH")
                      and lastLowTag in ("LL", "DB LL", "~LL")
                      and sz >= 3 and P_tag[sz - 3] in ("HH", "~HH"))
    if dchanConfirmed:
        return S_DCHAN
    if isRect:
        return S_RECT
    if isContr:
        return S_CONT
    # BROKEN UPTREND / BROKEN DOWNTREND (Amir 2026-07-05): the widening HH+LL state, split by
    # the DIRECTION of the break. A confirmed higher high with price making a LOWER LOW = an
    # uptrend that broke (broken uptrend); a confirmed lower low with price making a HIGHER HIGH
    # = a downtrend that broke (broken downtrend). Each covers the CONFIRMED pair (isExp, keyed
    # by which extreme is the most-recent pivot) AND the PROVISIONAL break: once the developing
    # swing's running extreme clears the opposite confirmed pivot by more than delta, that new
    # pivot's tag is LOCKED (a running low >delta below the last low can only confirm as an LL;
    # a running high >delta above the last high as an HH), so the broken state is knowable BEFORE
    # the pivot confirms and HOLDS across the swing (SPY 2025-10-10). Causal / no repaint —
    # run_lo/run_hi use only bars up to now.
    brokenUp = lastHighTag in ("HH", "~HH") and (
        (isExp and P_type[sz - 1] == PIVOT_LOW)
        or (swing_dir == -1 and np.isfinite(run_lo) and run_lo < lastLowPrice - delta))
    brokenDown = lastLowTag in ("LL", "~LL") and (
        (isExp and P_type[sz - 1] == PIVOT_HIGH)
        or (swing_dir == 1 and np.isfinite(run_hi) and run_hi > lastHighPrice + delta))
    if brokenUp:
        return S_BRKUP
    if brokenDown:
        return S_BRKDN
    return S_UNKNOWN


def _run(h, l, c, o, atr, atr_fraction, ob_seq=None, reversal_mode="bar", eq_fraction=None):
    """Single causal pass: swing detection + per-bar structure.
    Returns (P_idx, P_price, P_type, P_tag, P_conf, structure[], tail).
    P_conf[k] = the bar whose opposing break locked pivot k in — truncating the
    frame at any t >= P_conf[k] reproduces the pivot identically (causality).
    ob_seq: optional per-bar outside-bar ordering from SUB-BARS (the Pine f_ob_seq,
    2026-07-05 parity fix): +1 = the UP break fired first intrabar, -1 = DOWN first,
    0 = unknown -> fall back to the candle-color heuristic.
    reversal_mode (ST-5g lab study, 2026-07-09 — default 'bar' is the Pine-parity behavior
    and must stay byte-identical): 'bar' = a swing reverses when a bar breaks the PREVIOUS
    bar's extreme by delta; 'extreme' = classic zigzag, the swing reverses when price
    retreats delta from the RUNNING candidate extreme — a gradual counter-move accumulates
    to delta even when no single bar-over-bar step does (NQ 1h Jul 2 20:00: a ~100pt dip in
    -7/-30pt steps was invisible to 'bar' at every fraction).
    eq_fraction (ST-5g, Amir 2026-07-09): the double-top/bottom EQUALITY tolerance, decoupled
    from the reversal fraction — at extreme@0.7 the shared knob called extremes 20+ pts apart
    'doubles'. None = follow atr_fraction (byte-identical default); the lab passes 0.2 so DT/DB
    tagging keeps the app's own semantics regardless of the reversal knob."""
    eq = atr_fraction if eq_fraction is None else eq_fraction
    n = len(h)
    P_idx: list[int] = []
    P_price: list[float] = []
    P_type: list[int] = []
    P_tag: list[str] = []
    P_conf: list[int] = []      # bar where the opposing break locked this pivot in

    def record(t, price, idx, conf):
        # tag vs the ANCHOR: the last same-type pivot that is NOT a forgiven '~' poke. Forgiven
        # pivots must not move the anchor — otherwise repeated ring-deep undercuts each re-anchor
        # and a stair-down decline never prints LL (Amir 2026-07-06 05:30-07:05 5m review).
        # In default mode (eq == atr_fraction) no '~' tag exists, so this walk is identical to
        # "last same-type pivot" — byte-identical behavior (parity test).
        last_price = None
        last_tag = None
        for k in range(len(P_type) - 1, -1, -1):
            if P_type[k] == t and not P_tag[k].startswith("~"):
                last_price, last_tag = P_price[k], P_tag[k]
                break
        if last_price is None:
            tag = "H0" if t == PIVOT_HIGH else "L0"
        else:
            diff = price - last_price
            d_eq, d_rev = atr[idx] * eq, atr[idx] * atr_fraction
            if abs(diff) <= d_eq:                              # equal extreme -> double top/bottom
                if t == PIVOT_HIGH:
                    tag = "DT HH" if last_tag in ("DT HH", "HH") else "DT LH"
                else:
                    tag = "DB LL" if last_tag in ("DB LL", "LL") else "DB HL"
            elif t == PIVOT_HIGH:
                # FORGIVEN ring (ST-5g, corrected per Amir's 2026-07-06 5m marks): only a move
                # BEYOND the prior extreme can be "forgiven" (a stop-run poke), and only when it
                # is COUNTER-trend — a marginal new high in an up-context is simply a (weak) HH,
                # and a high BELOW the prior high is factually an LH regardless of distance.
                up_ctx = last_tag in ("DT HH", "HH")    # H0/L0 carry no class -> ring pokes vs
                                                        # a first pivot stay forgiven, not with-trend
                if diff > 0:
                    tag = "HH" if (up_ctx or diff > d_rev) else "~LH"
                else:
                    tag = "LH"
            else:
                dn_ctx = last_tag in ("DB LL", "LL")
                if diff < 0:
                    tag = "LL" if (dn_ctx or -diff > d_rev) else "~HL"
                else:
                    tag = "HL"
        P_idx.append(idx)
        P_price.append(price)
        P_type.append(t)
        P_tag.append(tag)
        P_conf.append(conf)

    structure = np.zeros(n, dtype=np.int64)
    swingDir = 0
    candH = candL = np.nan
    candHi = candLi = -1
    cur = S_UNKNOWN

    for i in range(n):
        if i == 0:
            continue
        delta = atr[i] * atr_fraction
        ph, pl = h[i - 1], l[i - 1]
        breakUp = h[i] > ph + delta
        breakDown = l[i] < pl - delta
        stepUp = l[i] > pl + delta
        stepDown = h[i] < ph - delta
        isOutside = breakUp and breakDown
        if isOutside and ob_seq is not None and ob_seq[i] != 0:
            sequence = int(ob_seq[i])           # true intrabar order from sub-bars
        else:
            sequence = (-1 if c[i] > o[i] else 1) if isOutside else 0
        rev_delta = delta
        if reversal_mode == "extreme":
            # In extreme mode a single wide bar can both EXTEND the swing and REVERSE it,
            # so intrabar ORDER matters on every bar (not just strict outside bars): sub-bar
            # truth when available, else the candle-color heuristic (red = high first).
            # Reversal tests are evaluated PER-OP against the LIVE candidate inside the op
            # loop below — the 2026-07-09 NQ 5m bug was flags precomputed against the stale
            # pre-bar candidate while the pivot recorded was the freshly extended one
            # (evidence and pivot referred to different levels).
            sequence = int(ob_seq[i]) if ob_seq is not None and ob_seq[i] != 0 \
                else (-1 if c[i] > o[i] else 1)
            # PER-BAR DELTA FLOOR + MARGIN (Amir approved 2026-07-09, margin same day): the
            # reversal threshold on bar i is the bar's own range PLUS the base ATR delta — a
            # lone bar can never round-trip its wick into a swing (NQ 5m 00:35/07:20 phantoms),
            # and clearing the floor by a rounding tie is not a reversal either (Jul 1 04:55 HH
            # cleared the bare floor by 0.25 pts: the swing extreme sat on the bar right before
            # a wide engulfing bar, so "the retrace" was really just that bar's own range).
            # Beyond a bar's own travel, genuine follow-through of the base delta is required.
            rev_delta = max(delta, (h[i] - l[i]) + atr[i] * eq)

        if swingDir == 0:
            swingDir = 1 if stepUp else -1 if stepDown else (1 if c[i] >= c[i - 1] else -1)
            if swingDir == 1:
                candH, candHi, candL, candLi = h[i], i, np.nan, -1
            else:
                candL, candLi, candH, candHi = l[i], i, np.nan, -1
        else:
            # process high & low in the order the (outside-bar) sequence implies
            high_first = sequence == 1 or (sequence == 0 and swingDir == 1)
            for op in (("H", "L") if high_first else ("L", "H")):
                if op == "H":
                    if swingDir == 1:
                        if candHi < 0 or h[i] >= candH:
                            candH, candHi = h[i], i
                    else:                               # swingDir == -1, reversal up?
                        rev = (bool(np.isfinite(candL)) and h[i] > candL + rev_delta) \
                            if reversal_mode == "extreme" else breakUp
                        if rev:
                            if candLi >= 0:
                                record(PIVOT_LOW, candL, candLi, i)
                            swingDir, candH, candHi, candL, candLi = 1, h[i], i, np.nan, -1
                else:
                    if swingDir == -1:
                        if candLi < 0 or l[i] <= candL:
                            candL, candLi = l[i], i
                    else:                               # swingDir == 1, reversal down?
                        rev = (bool(np.isfinite(candH)) and l[i] < candH - rev_delta) \
                            if reversal_mode == "extreme" else breakDown
                        if rev:
                            if candHi >= 0:
                                record(PIVOT_HIGH, candH, candHi, i)
                            swingDir, candL, candLi, candH, candHi = -1, l[i], i, np.nan, -1

        cur = _eval_structure(P_price, P_type, P_tag, cur, h[i], l[i], delta,
                              swingDir, candL, candH, eq_delta=atr[i] * eq)
        structure[i] = cur

    return P_idx, P_price, P_type, P_tag, P_conf, structure, (swingDir, candHi, candH, candLi, candL)


def _metrics(seg: Segment, d: pd.DataFrame, atr: np.ndarray) -> None:
    c = d["close"].to_numpy()[seg.start:seg.end + 1]
    seg.bars = seg.end - seg.start + 1
    if seg.bars < 2 or c[0] <= 0:
        return
    seg.net_pct = (c[-1] / c[0] - 1.0) * 100
    if np.all(np.isfinite(c)) and c.min() > 0:        # the log-linear fit needs positive finite closes
        logc = np.log(c)
        # closed-form degree-1 least-squares (identical to np.polyfit(x, logc, 1) but ~10x faster — no
        # lstsq overhead; _metrics runs ~150k times per scan so this is the dominant cost).
        nbar = len(c)
        xd = np.arange(nbar, dtype=float) - (nbar - 1) / 2.0      # x centered (x = 0..n-1)
        sxx = float((xd * xd).sum())
        yd = logc - float(logc.mean())
        b = float((xd * yd).sum() / sxx) if sxx > 0 else 0.0      # log-slope
        ss_tot = float((yd * yd).sum())
        ss_res = float(((yd - b * xd) ** 2).sum())                # residuals about the fitted line
        seg.r2 = 0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
        seg.ann_pct = (np.exp(np.clip(b * 252, -50, 50)) - 1.0) * 100  # clip: parabolic legs overflow exp
    moves = np.abs(np.diff(c)).sum()
    seg.er = abs(c[-1] - c[0]) / moves if moves > 0 else 0.0
    seg.clarity = (seg.r2 + seg.er) / 2
    mean_atr = float(atr[seg.start:seg.end + 1].mean())
    if mean_atr > 0:
        seg.slope_atr = (c[-1] - c[0]) / max(seg.bars - 1, 1) / mean_atr
    if ("pole" not in seg.tags and seg.kind == "up" and seg.bars <= POLE_MAX_BARS
            and seg.net_pct >= POLE_MIN_GAIN_PCT and seg.er >= POLE_MIN_ER):
        seg.tags.append("pole")


def _segments_from_structure(structure, d, atr) -> list[Segment]:
    """Collapse equal-structure runs into metric-bearing Segments."""
    n = len(structure)
    segments: list[Segment] = []
    a = 0
    for i in range(1, n + 1):
        if i == n or structure[i] != structure[a]:
            kind, subtype = _STRUCT_KIND[int(structure[a])]
            seg = Segment(a, i - 1, kind, subtype=subtype, _d=d, _atr=atr)
            seg.bars = (i - 1) - a + 1          # cheap + needed by _tag_poles/setups before ensure()
            segments.append(seg)               # metrics deferred to seg.ensure() (lazy)
            a = i
    if segments:
        segments[-1].open_ended = True
    return segments


def _tag_poles(segments, P_idx, P_price, P_type) -> None:
    """A pole is a single fast, large up-thrust. Because the structure FSM splits a
    thrust into transition (low->breakout) + up (breakout->break), the segment net%
    understates it — so detect the pole on the raw swing LEG (pivot low -> pivot high)
    and tag whichever 'up' segment holds that high."""
    legs = [(P_idx[k - 1], P_price[k - 1], P_idx[k], P_price[k])
            for k in range(1, len(P_idx))
            if P_type[k] == PIVOT_HIGH and P_type[k - 1] == PIVOT_LOW]
    for seg in segments:
        if seg.kind != "up" or "pole" in seg.tags:
            continue
        for lo_i, lo_p, hi_i, hi_p in legs:
            if (seg.start <= hi_i <= seg.end and lo_p > 0
                    and (hi_i - lo_i) <= POLE_MAX_BARS
                    and (hi_p / lo_p - 1) * 100 >= POLE_MIN_GAIN_PCT):
                seg.tags.append("pole")
                break


def _ob_sequence(d: pd.DataFrame, sub: pd.DataFrame, atr: np.ndarray, fraction: float) -> np.ndarray:
    """Pine f_ob_seq (2026-07-05 parity fix): for each OUTSIDE bar of the parent frame,
    walk its sub-bars chronologically and report which break fired FIRST — +1 up-break
    first, -1 down-break first, 0 = no sub-bars for that span (color fallback).
    Parent index must be period START dates (daily / 2D / weekly resamples are).
    The color heuristic guessed this from the parent candle and mis-ordered e.g. the
    QQQ 2D 2026-06-22 outside bar (Amir's chart); sub-bars are the truth."""
    n = len(d)
    out = np.zeros(n, dtype=np.int8)
    if sub is None or not len(sub):
        return out
    h, l = d["high"].to_numpy(float), d["low"].to_numpy(float)
    sdays = sub.index.normalize()
    sh, sl = sub["high"].to_numpy(float), sub["low"].to_numpy(float)
    so, sc = sub["open"].to_numpy(float), sub["close"].to_numpy(float)
    starts = pd.DatetimeIndex(pd.to_datetime(d.index)).normalize()
    for i in range(1, n):
        delta = atr[i] * fraction
        ph, pl = h[i - 1], l[i - 1]
        if not (h[i] > ph + delta and l[i] < pl - delta):
            continue                                     # not an outside bar
        lo_t = starts[i]
        hi_t = starts[i + 1] if i + 1 < n else None
        m = (sdays >= lo_t) if hi_t is None else ((sdays >= lo_t) & (sdays < hi_t))
        idx = np.where(m)[0]
        for j in idx:
            up = sh[j] > ph + delta
            dn = sl[j] < pl - delta
            if up and not dn:
                out[i] = 1
                break
            if dn and not up:
                out[i] = -1
                break
            if up and dn:                                # both inside one sub-bar: Pine's
                out[i] = 1 if sc[j] >= so[j] else -1     # sub-bar color rule
                break
    return out


def _compute(d: pd.DataFrame, atr_fraction: float | None = None,
             subbars: pd.DataFrame | None = None, reversal_mode: str = "bar",
             eq_fraction: float | None = None) -> dict:
    """Run pivots + structure + segments once; memoize on d.attrs for the default
    fraction so segment_chart / swing_lows / regime all share one pass per frame.
    subbars: optional finer-timeframe OHLC frame for TRUE outside-bar ordering
    (opt-in — results are not memoized so the default path is untouched).
    reversal_mode / eq_fraction: see _run — non-default modes are lab-only, never memoized."""
    use_default = atr_fraction is None and subbars is None and reversal_mode == "bar" \
        and eq_fraction is None
    if use_default:
        cached = _TRENDLAB_CACHE.get(d)
        if cached is not None:
            return cached
    af = ATR_FRACTION if atr_fraction is None else atr_fraction
    atr = atr_series(d)
    ob = _ob_sequence(d, subbars, atr, af) if subbars is not None else None
    P_idx, P_price, P_type, P_tag, P_conf, structure, tail = _run(
        d["high"].to_numpy(float), d["low"].to_numpy(float),
        d["close"].to_numpy(float), d["open"].to_numpy(float), atr, af, ob_seq=ob,
        reversal_mode=reversal_mode, eq_fraction=eq_fraction)

    swings = [Swing(P_idx[k], "H" if P_type[k] == PIVOT_HIGH else "L", P_price[k])
              for k in range(len(P_idx))]
    sd, cHi, cH, cLi, cL = tail            # active developing extreme -> provisional swing
    if sd == 1 and cHi >= 0:
        swings.append(Swing(cHi, "H", cH, provisional=True))
    elif sd == -1 and cLi >= 0:
        swings.append(Swing(cLi, "L", cL, provisional=True))

    segments = _segments_from_structure(structure, d, atr)
    _tag_poles(segments, P_idx, P_price, P_type)
    result = {
        "pidx": P_idx, "pprice": P_price, "ptype": P_type, "ptag": P_tag,
        "pconf": P_conf, "structure": structure, "segments": segments, "swings": swings,
    }
    if use_default:
        _TRENDLAB_CACHE.put(d, result)
    return result


def segment_chart(d: pd.DataFrame, atr_fraction: float | None = None,
                  subbars: pd.DataFrame | None = None) -> tuple[list[Segment], list[Swing]]:
    """OHLC frame -> (ordered segments covering every bar, swings). Entry point
    the scanner/regime layer imports."""
    r = _compute(d, atr_fraction, subbars)
    return r["segments"], r["swings"]


def swing_lows(d: pd.DataFrame, lookback: int) -> list[tuple[int, float]]:
    """Confirmed pivot lows within the last `lookback` bars: [(bar_idx, price), ...]."""
    r = _compute(d)
    cut = len(d) - lookback
    return [(r["pidx"][k], r["pprice"][k]) for k in range(len(r["pidx"]))
            if r["ptype"][k] == PIVOT_LOW and r["pidx"][k] >= cut]


def swing_highs(d: pd.DataFrame, lookback: int) -> list[tuple[int, float]]:
    """Confirmed pivot highs within the last `lookback` bars."""
    r = _compute(d)
    cut = len(d) - lookback
    return [(r["pidx"][k], r["pprice"][k]) for k in range(len(r["pidx"]))
            if r["ptype"][k] == PIVOT_HIGH and r["pidx"][k] >= cut]


def pivots(d: pd.DataFrame, atr_fraction: float | None = None,
           subbars: pd.DataFrame | None = None) -> tuple[list[int], list[float], list[int], list[str]]:
    """Confirmed pivots for the frame: (pidx, pprice, ptype, ptag). ptype uses
    PIVOT_HIGH/PIVOT_LOW; ptag carries the structure chain (H0/L0/HH/LH/HL/LL/DT/DB…).
    Memoized via _compute for the default fraction."""
    r = _compute(d, atr_fraction, subbars)
    return r["pidx"], r["pprice"], r["ptype"], r["ptag"]


def pivots_conf(d: pd.DataFrame, atr_fraction: float | None = None,
                subbars: pd.DataFrame | None = None, reversal_mode: str = "bar",
                eq_fraction: float | None = None
                ) -> tuple[list[int], list[float], list[int], list[str], list[int]]:
    """pivots() plus each pivot's CONFIRMATION bar: (pidx, pprice, ptype, ptag, pconf).
    pconf[k] is the first bar where the pivot is knowable — pivots(d.iloc[:t+1])
    contains pivot k iff t >= pconf[k], so point-in-time studies get every
    confirmation in ONE pass instead of an O(n^2) prefix rescan."""
    r = _compute(d, atr_fraction, subbars, reversal_mode, eq_fraction)
    return r["pidx"], r["pprice"], r["ptype"], r["ptag"], r["pconf"]


def structure_series(d: pd.DataFrame, atr_fraction: float | None = None,
                     subbars: pd.DataFrame | None = None, reversal_mode: str = "bar",
                     eq_fraction: float | None = None) -> np.ndarray:
    """Per-bar structure FSM codes (S_UP/S_DOWN/...). structure[t] is the state a
    prefix truncated at t would end in — see struct_label for the display class."""
    return _compute(d, atr_fraction, subbars, reversal_mode, eq_fraction)["structure"]


def struct_label(code: int) -> str:
    """FSM code -> the segment class string used across the labs ('up', 'down',
    'range/contracting', 'range/expanding', 'range/rectangle', 'transition')."""
    kind, subtype = _STRUCT_KIND[int(code)]
    return kind + (f"/{subtype}" if subtype else "")


# ---------------------------------------------------------------- output

BG = {"up": "rgba(47,191,143,0.10)", "down": "rgba(224,90,109,0.10)",
      "range": "rgba(232,184,75,0.10)", "channel": "rgba(140,110,220,0.12)",
      "broken": "rgba(224,132,60,0.10)", "transition": "rgba(0,0,0,0)"}
BG_POLE = "rgba(47,191,143,0.22)"
MARK = {"up": ("arrowUp", "#2fbf8f"), "down": ("arrowDown", "#e05a6d"),
        "range": ("square", "#e8b84b"), "transition": ("circle", "#6b7a8c")}


def to_table(segments: list[Segment], d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, s in enumerate(segments, 1):
        s.ensure()                                  # lab/CLI shows all segments -> compute their metrics
        label = s.kind + (f"/{s.subtype}" if s.subtype else "")
        rows.append({
            "#": i, "type": label,
            "from": d.index[s.start].date().isoformat(),
            "to": d.index[s.end].date().isoformat(),
            "bars": s.bars, "net_%": round(s.net_pct, 1),
            "ann_%": round(max(min(s.ann_pct, 999.0), -999.0), 1),
            "atr/bar": round(s.slope_atr, 2),
            "r2": round(s.r2, 2), "er": round(s.er, 2),
            "clarity": round(s.clarity, 2),
            "tags": " ".join(s.tags + (["open"] if s.open_ended else [])),
        })
    return pd.DataFrame(rows)


def render_html(symbol: str, d: pd.DataFrame,
                segments: list[Segment], swings: list[Swing],
                table: pd.DataFrame) -> str:
    iso = [i.date().isoformat() for i in d.index]
    candles = [{"time": iso[i], "open": round(float(r.open), 4),
                "high": round(float(r.high), 4), "low": round(float(r.low), 4),
                "close": round(float(r.close), 4)}
               for i, (_, r) in enumerate(d.iterrows())]
    # one bg entry per bar (segments share boundary bars; duplicate times corrupt
    # the chart's time index — later segment wins)
    bar_color = [BG["transition"]] * len(d)
    for s in segments:
        s.ensure()
        color = BG_POLE if "pole" in s.tags else BG[s.kind]
        for i in range(s.start, s.end + 1):
            bar_color[i] = color
    bg = [{"time": iso[i], "value": 1, "color": c} for i, c in enumerate(bar_color)]
    zig = [{"time": iso[s.idx], "value": round(s.price, 4)} for s in swings]
    markers = []
    for i, s in enumerate(segments, 1):
        shape, color = MARK[s.kind]
        text = f"#{i} {s.subtype or s.kind}" + (" POLE" if "pole" in s.tags else "")
        markers.append({"time": iso[s.start], "position": "aboveBar",
                        "shape": shape, "color": color, "text": text})
    seg_rows = table.to_dict("records")
    bounds = [{"a": s.start, "b": s.end} for s in segments]

    html = (HTML_TEMPLATE
            .replace("__SYM__", symbol)
            .replace("__PARAMS__", f"Pine swings · delta {ATR_FRACTION}×ATR({ATR_LEN})")
            .replace("__CANDLES__", json.dumps(candles))
            .replace("__BG__", json.dumps(bg))
            .replace("__ZIG__", json.dumps(zig))
            .replace("__MARKERS__", json.dumps(markers))
            .replace("__SEGS__", json.dumps(seg_rows, default=str))
            .replace("__BOUNDS__", json.dumps(bounds)))
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = config.OUTPUT_DIR / f"trendlab_{symbol}.html"
    out.write_text(html, encoding="utf-8")
    return str(out)


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>trendlab — __SYM__</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
:root{--bg:#0b0f14;--panel:#11161d;--line:#1d2630;--txt:#cfd8e3;--dim:#6b7a8c;
      --up:#2fbf8f;--dn:#e05a6d;--acc:#e8b84b;--mono:'Consolas','Menlo',monospace}
*{box-sizing:border-box;margin:0}
body{background:var(--bg);color:var(--txt);font:14px/1.45 system-ui,sans-serif;
     display:flex;flex-direction:column;height:100vh}
header{display:flex;gap:14px;align-items:baseline;padding:10px 18px;border-bottom:1px solid var(--line)}
header h1{font:600 16px var(--mono);color:var(--acc)}
header .d{color:var(--dim);font:12px var(--mono)}
#chart{flex:1;min-height:0}
#legend{display:flex;gap:18px;padding:6px 18px;color:var(--dim);font:12px var(--mono);
        border-top:1px solid var(--line)}
#legend span::before{content:'';display:inline-block;width:10px;height:10px;margin-right:5px}
#legend .u::before{background:rgba(47,191,143,.45)} #legend .dnn::before{background:rgba(224,90,109,.45)}
#legend .r::before{background:rgba(232,184,75,.45)} #legend .p::before{background:rgba(47,191,143,.9)}
#tbl{max-height:34vh;overflow:auto;border-top:1px solid var(--line)}
table{border-collapse:collapse;width:100%;font:12.5px var(--mono)}
th,td{padding:5px 10px;text-align:right;white-space:nowrap}
th{position:sticky;top:0;background:var(--panel);color:var(--dim);border-bottom:1px solid var(--line)}
th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3),th:nth-child(4),td:nth-child(4),
th:nth-child(12),td:nth-child(12){text-align:left}
tbody tr{cursor:pointer;border-bottom:1px solid #131a22}
tbody tr:hover{background:#16202b}
td.up{color:var(--up)} td.dn{color:var(--dn)}
</style></head><body>
<header><h1>TRENDLAB __SYM__</h1><span class="d">__PARAMS__</span>
<span class="d">click a row to zoom its segment</span></header>
<div id="chart"></div>
<div id="legend"><span class="u">up trend</span><span class="dnn">down trend</span>
<span class="r">range</span><span class="p">pole</span><span>no tint = transition</span></div>
<div id="tbl"></div>
<script>
const CANDLES=__CANDLES__, BGDATA=__BG__, ZIG=__ZIG__, MARKERS=__MARKERS__,
      SEGS=__SEGS__, BOUNDS=__BOUNDS__;
const chart=LightweightCharts.createChart(document.getElementById('chart'),{
  layout:{background:{color:'#0b0f14'},textColor:'#6b7a8c'},
  grid:{vertLines:{color:'#131a22'},horzLines:{color:'#131a22'}},
  rightPriceScale:{borderColor:'#1d2630'},timeScale:{borderColor:'#1d2630'},autoSize:true});
const bgs=chart.addHistogramSeries({priceScaleId:'bg',priceFormat:{type:'volume'},
  lastValueVisible:false,priceLineVisible:false});
chart.priceScale('bg').applyOptions({scaleMargins:{top:0,bottom:0},visible:false});
bgs.setData(BGDATA);
const cs=chart.addCandlestickSeries({upColor:'#2fbf8f',downColor:'#e05a6d',
  wickUpColor:'#2fbf8f',wickDownColor:'#e05a6d',borderVisible:false});
cs.setData(CANDLES); cs.setMarkers(MARKERS);
chart.addLineSeries({color:'rgba(232,184,75,0.85)',lineWidth:1,priceLineVisible:false,
  lastValueVisible:false,crosshairMarkerVisible:false}).setData(ZIG);
chart.timeScale().fitContent();
const cols=Object.keys(SEGS[0]||{});
let h='<table><thead><tr>'+cols.map(c=>'<th>'+c+'</th>').join('')+'</tr></thead><tbody>';
for(let i=0;i<SEGS.length;i++){const x=SEGS[i];
  h+='<tr data-i="'+i+'">'+cols.map(c=>{
    let cls='';if(c==='net_%'||c==='ann_%')cls=x[c]>=0?'up':'dn';
    return '<td class="'+cls+'">'+x[c]+'</td>';}).join('')+'</tr>';}
document.getElementById('tbl').innerHTML=h+'</tbody></table>';
document.querySelectorAll('#tbl tbody tr').forEach(tr=>tr.onclick=()=>{
  const b=BOUNDS[+tr.dataset.i], pad=Math.max(5,Math.round((b.b-b.a)*0.15));
  chart.timeScale().setVisibleLogicalRange({from:b.a-pad,to:b.b+pad});});
</script></body></html>
"""


def load_frame(source: str, bars: int) -> tuple[str, pd.DataFrame]:
    if source.lower().endswith(".csv"):
        df = pd.read_csv(source, parse_dates=["date"], index_col="date")
        df.columns = [c.lower() for c in df.columns]
        name = source.rsplit("/", 1)[-1].removesuffix(".csv")
    else:
        name = source.upper()
        df = datastore.load_bars(name)
        if df is None:
            raise SystemExit(f"{name} not in cache ({config.BARS_DIR}) — run scan.py init, "
                             f"or pass a csv path")
    if bars > 0:
        df = df.iloc[-bars:]
    if len(df) < 40:
        raise SystemExit(f"only {len(df)} bars for {name} — not enough to segment")
    return name, df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("symbol", help="ticker in the bar cache, or a path to an OHLC csv")
    ap.add_argument("--bars", type=int, default=500, help="trailing bars to analyze (0 = all)")
    ap.add_argument("--atr-fraction", type=float, default=ATR_FRACTION,
                    help="swing delta as a fraction of ATR (Pine default 0.2)")
    ap.add_argument("--open", action="store_true", help="open the html when done")
    args = ap.parse_args()

    name, df = load_frame(args.symbol, args.bars)
    segments, swings = segment_chart(df, args.atr_fraction)
    table = to_table(segments, df)
    print(f"\n{name}: {len(df)} bars -> {len(segments)} segments, {len(swings)} swings\n")
    print(table.to_string(index=False))
    out = render_html(name, df, segments, swings, table)
    print(f"\nchart: {out}")
    if args.open:
        webbrowser.open(f"file://{out}")


if __name__ == "__main__":
    sys.exit(main())
