"""SPY (or any symbol's) session day-type classifier: TREND day vs RANGE day.

Three independent methods, all point-in-time (a session is classified from ITS OWN bars plus
PRIOR sessions' ATR only — nothing after the close):

  A  daily-bar heuristics    body/range fraction, close position, range vs prior ATR14.
  B  intraday 5-min structure directional efficiency (net move / total path), one-sidedness vs
                             session VWAP, opening-range escape that never re-enters, close
                             location, range vs prior ATR14.
  C  hybrid                  A and B agree on a trend direction -> trend; both say range ->
                             range; anything else -> mixed (disagreement is the signal).

Labels: trend_up / trend_down / range / mixed.
Daily OHLC + ATR are DERIVED from the 5-min RTH session (one price basis for every method).

    daytype.classify_all(bars5)   -> DataFrame indexed by session date: metrics + label_a/b/c
    daytype.session_metrics(g)    -> dict of one session's raw metrics
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── knobs (study-stage: module constants, not in the ⚙ drawer until graduated) ──────────
OR_BARS = 6            # opening range = first 30 min of 5-min bars
A_TREND_BODY = 0.70    # A: |close-open| / range        (Amir's spec (a); was 0.65)
A_TREND_CPOS = 0.85    # A: close in the top/bottom 15% (Amir's spec (a); was 0.80)
A_TREND_RATR = 1.00    # A: day range >= 1.0x prior ATR14
A_RANGE_RATR = 0.85    # A: day range <= 0.85x prior ATR14 ...
A_RANGE_BODY = 0.45    # ...and small body -> range day
B_TREND_EFF = 0.22     # B: |net| / path-length (calibrated: 5-min path is long — eff ~0.25 is
                       # already the ~90th percentile; famous trend days score 0.30-0.36.
                       # Amir 2026-07-04: relaxed 0.25 -> 0.22, and it is knob-tunable live
                       # in the review page)
B_TREND_SIDE = 0.80    # B: fraction of closes on ONE side of session VWAP (median day is 0.79)
B_TREND_CPOS = 0.72    # B: close position in session range (>= up / <= 1-x down)
B_TREND_RATR = 1.00
B_RANGE_EFF = 0.20
B_RANGE_SIDE = 0.70
B_RANGE_RATR = 1.00
OR_WINDOWS = {"or5": 1, "or15": 3, "or30": 6, "or60": 12}   # opening-range sizes in 5-min bars
ORB_CPOS = 0.70        # ORB trend: day must close in the break direction's top/bottom 30%
ORB_MIN_LEFT = 6       # a break in the last 30 min can't "trend for the rest of the day"
IB_BARS = 12           # Initial Balance = the first hour of 5-min bars
D_EXT = 1.00           # D: IB extended by >= this multiple of IB height, one direction
D_BRK = 0.20           # D: "IB never meaningfully broken" = neither side pushed > this multiple
D_CROSS = 8            # D: this many VWAP crosses = rotation -> range day
D_RANGE_RATR = 0.80    # D: range-day alternative gate: day range < 0.8x prior ATR


def b_metrics_from(g: pd.DataFrame, start_bar: int) -> dict | None:
    """Method-B metrics measured from `start_bar` onward instead of the 09:30 open (Amir
    2026-07-15: "does B read better if we ignore the opening chop?").

    Everything B looks at — path, efficiency, range, close position — uses ONLY the post-start
    window (his call: the day genuinely starts late). VWAP is returned BOTH ways, because which
    one is right is an empirical question, not a design one:
      vwap_side_re  — VWAP rebuilt from start_bar (internally consistent; the late session's own
                      mean price, but not a line any trader has on screen)
      vwap_side_930 — the TRUE session VWAP, still anchored at the real open, with only the
                      post-start closes scored against it (what a trader actually watches)
    At start_bar=0 the two are identical by construction, and this whole function reduces to the
    B half of session_metrics().
    """
    g = g.iloc[start_bar:]
    if len(g) < 6:
        return None
    o = float(g["open"].iloc[0]); c = float(g["close"].iloc[-1])
    hi = float(g["high"].max()); lo = float(g["low"].min())
    rng = hi - lo
    if rng <= 0:
        return None
    cl = g["close"].to_numpy(float)
    path = float(np.abs(np.diff(cl)).sum()) + abs(cl[0] - o)
    return {"eff": (abs(c - o) / path) if path > 0 else 0.0,
            "close_pos": (c - lo) / rng, "range": rng, "up": c >= o}


def _vwap_side(g: pd.DataFrame, start_bar: int, anchor_bar: int) -> float:
    """Fraction of post-`start_bar` closes on ONE side of a VWAP anchored at `anchor_bar`."""
    a = g.iloc[anchor_bar:]
    tp = ((a["high"] + a["low"] + a["close"]) / 3).to_numpy(float)
    v = a["volume"].to_numpy(float)
    vw = (tp * v).cumsum() / np.maximum(v.cumsum(), 1)
    off = start_bar - anchor_bar                      # score only the bars at/after the start
    cl = a["close"].to_numpy(float)[off:]
    vw = vw[off:]
    if not len(cl):
        return float("nan")
    above = float((cl > vw).mean())
    return max(above, 1 - above)


def session_metrics(g: pd.DataFrame) -> dict | None:
    """Raw metrics for ONE session's 5-min RTH bars (no ATR context here)."""
    if len(g) < OR_BARS + 6:
        return None
    o = float(g["open"].iloc[0]); c = float(g["close"].iloc[-1])
    hi = float(g["high"].max()); lo = float(g["low"].min())
    rng = hi - lo
    if rng <= 0:
        return None
    cl = g["close"].to_numpy(float)
    path = float(np.abs(np.diff(cl)).sum()) + abs(cl[0] - o)
    tp = (g["high"] + g["low"] + g["close"]) / 3
    v = g["volume"].to_numpy(float)
    vwap = (tp.to_numpy(float) * v).cumsum() / np.maximum(v.cumsum(), 1)
    above = float((cl > vwap).mean())
    or_hi = float(g["high"].iloc[:OR_BARS].max()); or_lo = float(g["low"].iloc[:OR_BARS].min())
    after = cl[OR_BARS:]
    esc = np.where((after > or_hi) | (after < or_lo))[0]
    or_hold = False
    if esc.size:                       # escaped the opening range — did it ever close back inside?
        back = after[esc[0]:]
        or_hold = bool(((back <= or_hi) & (back >= or_lo)).sum() == 0)
    # Initial Balance (first hour): extension beyond it in IB-height multiples + rotation stats
    ib_ext_up = ib_ext_dn = None; ib_touch_lo = ib_touch_hi = None
    if len(g) >= IB_BARS + 6:
        ib_hi = float(g["high"].iloc[:IB_BARS].max()); ib_lo = float(g["low"].iloc[:IB_BARS].min())
        ib_h = ib_hi - ib_lo
        post_h = float(g["high"].iloc[IB_BARS:].max()); post_l = float(g["low"].iloc[IB_BARS:].min())
        if ib_h > 0:
            ib_ext_up = max(0.0, (post_h - ib_hi) / ib_h)
            ib_ext_dn = max(0.0, (ib_lo - post_l) / ib_h)
            ib_touch_lo = bool(post_l <= ib_lo)      # revisited the LOWER IB extreme after the IB
            ib_touch_hi = bool(post_h >= ib_hi)      # revisited the UPPER IB extreme after the IB
    diff = cl - vwap
    vwap_crosses = int((np.sign(diff[1:]) != np.sign(diff[:-1])).sum())
    # deepest adverse CLOSING retrace vs the day's range, in the day's dominant direction
    if c >= o:
        runmax = np.maximum.accumulate(cl)
        max_pb = float((runmax - cl).max()) / rng
    else:
        runmin = np.minimum.accumulate(cl)
        max_pb = float((cl - runmin).max()) / rng
    return {"open": o, "close": c, "high": hi, "low": lo, "range": rng,
            "body_frac": abs(c - o) / rng, "close_pos": (c - lo) / rng,
            "eff": (abs(c - o) / path) if path > 0 else 0.0,
            "vwap_side": max(above, 1 - above), "or_hold": or_hold,
            "ib_ext_up": ib_ext_up, "ib_ext_dn": ib_ext_dn,
            "ib_touch_lo": ib_touch_lo, "ib_touch_hi": ib_touch_hi,
            "vwap_crosses": vwap_crosses, "max_pb": max_pb,
            "up": c >= o}


def _label_a(m, ratr) -> str:
    if m["body_frac"] >= A_TREND_BODY:
        one_sided = ("up" if (m["close_pos"] >= A_TREND_CPOS and m["up"]) else
                     "down" if (m["close_pos"] <= 1 - A_TREND_CPOS and not m["up"]) else None)
        if one_sided:                       # one-way day: TREND if range expanded, else DRIFT
            return (f"trend_{one_sided}" if ratr >= A_TREND_RATR else f"drift_{one_sided}")
    if ratr <= A_RANGE_RATR and m["body_frac"] <= A_RANGE_BODY:
        return "range"
    return "mixed"


def _label_b(m, ratr) -> str:
    if m["eff"] >= B_TREND_EFF and m["vwap_side"] >= B_TREND_SIDE:
        one_sided = ("up" if (m["close_pos"] >= B_TREND_CPOS and m["up"]) else
                     "down" if (m["close_pos"] <= 1 - B_TREND_CPOS and not m["up"]) else None)
        if one_sided:                       # one-way day: TREND if range expanded, else DRIFT
            return (f"trend_{one_sided}" if ratr >= B_TREND_RATR else f"drift_{one_sided}")
    if (ratr <= B_RANGE_RATR and m["eff"] <= B_RANGE_EFF and m["vwap_side"] <= B_RANGE_SIDE
            and not m["or_hold"]):
        return "range"
    return "mixed"


def _label_c(a: str, b: str) -> str:
    if a == b and a != "mixed":
        return a
    return "mixed"


def _label_d(m, ratr) -> str:
    """Method D — Initial Balance (Amir's spec (c)): trend = IB extended >= D_EXT x its height
    one way AND the OTHER IB extreme never revisited after the first hour; range = IB never
    meaningfully broken, or a small day, or heavy VWAP rotation."""
    if m["ib_ext_up"] is None:
        return "mixed"
    if m["ib_ext_up"] >= D_EXT and not m["ib_touch_lo"]:
        return "trend_up"
    if m["ib_ext_dn"] >= D_EXT and not m["ib_touch_hi"]:
        return "trend_down"
    # DRIFT tier (Amir 2026-07-04, June-30 case): meaningfully broke the IB one way, never
    # revisited the other extreme, closed strong -> one-sided day even if small. Checked
    # BEFORE the range gate so a quiet grind can't be called "range" by the <0.8xATR rule.
    if (m["ib_ext_up"] >= D_BRK and not m["ib_touch_lo"]
            and m["ib_ext_dn"] < D_BRK and m["close_pos"] >= ORB_CPOS):
        return "drift_up"
    if (m["ib_ext_dn"] >= D_BRK and not m["ib_touch_hi"]
            and m["ib_ext_up"] < D_BRK and m["close_pos"] <= 1 - ORB_CPOS):
        return "drift_down"
    if (max(m["ib_ext_up"], m["ib_ext_dn"]) < D_BRK or ratr <= D_RANGE_RATR
            or m["vwap_crosses"] >= D_CROSS):
        return "range"
    return "mixed"


def orb(g: pd.DataFrame, nb: int) -> str | None:
    """Opening-range-breakout TREND test (Amir 2026-07-04): once price CLOSED beyond the
    first-`nb`-bars range, did it hold outside it for the REST of the session and close with
    strength in the break direction? -> 'up' / 'down' / None. Used to sub-type MIXED days."""
    n = len(g)
    if n < nb + ORB_MIN_LEFT:
        return None
    cl = g["close"].to_numpy(float)
    or_hi = float(g["high"].iloc[:nb].max()); or_lo = float(g["low"].iloc[:nb].min())
    brk, j = None, None
    for i in range(nb, n):
        if cl[i] > or_hi:
            brk, j = "up", i; break
        if cl[i] < or_lo:
            brk, j = "down", i; break
    if brk is None or j > n - ORB_MIN_LEFT:
        return None
    hi = float(g["high"].max()); lo = float(g["low"].min()); rng = hi - lo
    if rng <= 0:
        return None
    cpos = (cl[-1] - lo) / rng
    post = cl[j:]
    if brk == "up":
        return "up" if (post >= or_hi).all() and cpos >= ORB_CPOS else None
    return "down" if (post <= or_lo).all() and cpos <= 1 - ORB_CPOS else None


def classify_all(bars5: pd.DataFrame) -> pd.DataFrame:
    """Classify every RTH session in a 5-min frame. Returns a DataFrame indexed by session
    date (str) with the metrics + ratr (range / prior ATR14 of 5-min-derived dailies) +
    label_a / label_b / label_c."""
    rth = bars5.between_time("09:30", "15:59")
    rows = []
    for day, g in rth.groupby(rth.index.date):
        m = session_metrics(g)
        if m:
            m["date"] = str(day)
            for k, nb in OR_WINDOWS.items():           # ORB trend test per opening-range size
                m[f"orb_{k}"] = orb(g, nb)
            rows.append(m)
    df = pd.DataFrame(rows).set_index("date")
    prev_c = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_c).abs(),
                    (df["low"] - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().shift(1)            # PRIOR sessions only
    df["ratr"] = df["range"] / atr
    df = df[df["ratr"].notna()]
    df["label_a"] = [_label_a(m, r) for m, r in zip(df.to_dict("records"), df["ratr"])]
    df["label_b"] = [_label_b(m, r) for m, r in zip(df.to_dict("records"), df["ratr"])]
    df["label_c"] = [_label_c(a, b) for a, b in zip(df["label_a"], df["label_b"])]
    df["label_d"] = [_label_d(m, r) for m, r in zip(df.to_dict("records"), df["ratr"])]
    return df
