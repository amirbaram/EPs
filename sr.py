"""Support/Resistance levels — swing-pivot zones, clustered into strength-weighted levels, classified live.

Ported from sr.pine ("SwingsThreshold"): every confirmed ATR-fraction swing pivot is an S/R zone; on the
latest bar a level below price acts as support, above as resistance. We ENHANCE it — nearby pivots are
merged into one level whose STRENGTH = how many times price reversed/tested there. Reuses trendlab's
pivots (same ATR*0.2 zigzag, framecached), so it runs on any enriched frame (daily / weekly / intraday).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
import trendlab

_EMPTY = {"support": [], "resistance": [], "nearest_support": None, "nearest_resistance": None}


def sr_levels(d: pd.DataFrame, price: float | None = None) -> dict:
    """S/R for the frame's latest bar. Returns the nearest SR_LIVE_COUNT support + resistance levels
    (each {price, lo, hi, kind, strength, dist_atr, dist_pct}) + nearest_support/resistance in ATRs."""
    if d is None or len(d) < 30:
        return dict(_EMPTY)
    n = len(d)
    r = trendlab._compute(d)
    pidx, pprice, ptype = r["pidx"], r["pprice"], r["ptype"]
    high = d["high"].to_numpy(float); low = d["low"].to_numpy(float)
    op = d["open"].to_numpy(float); cl = d["close"].to_numpy(float)
    atr = float(d["atr14"].iloc[-1]) if pd.notna(d["atr14"].iloc[-1]) else None
    if not atr or atr <= 0:
        atr = float(np.nanmean((high - low)[-20:])) or 1.0
    close = float(cl[-1]) if price is None else float(price)
    cut = n - config.SR_LOOKBACK

    # 1. each pivot -> a zone (band from the extreme to the pivot bar's body), keyed by the extreme
    pivs = []
    for k in range(len(pidx)):
        i = int(pidx[k])
        if i < cut or i >= n:
            continue
        if ptype[k] == trendlab.PIVOT_HIGH:
            lo, hi, key = max(float(op[i]), float(cl[i])), float(high[i]), float(high[i])
        else:
            lo, hi, key = float(low[i]), min(float(op[i]), float(cl[i])), float(low[i])
        if hi < lo:
            lo, hi = hi, lo
        pivs.append((key, lo, hi, i))
    if not pivs:
        return dict(_EMPTY)

    # 2. cluster pivots whose key prices sit within SR_CLUSTER_ATR*ATR; strength = tests of the band.
    # Cap each cluster's key-span at `tol` (measure from the cluster's FIRST key, not the last) so a trend's
    # ladder of pivots doesn't single-linkage-chain into one giant band.
    pivs.sort(key=lambda x: x[0])
    tol = config.SR_CLUSTER_ATR * atr
    clusters, cur, start_key = [], [pivs[0]], pivs[0][0]
    for p in pivs[1:]:
        if p[0] - start_key <= tol:
            cur.append(p)
        else:
            clusters.append(cur); cur, start_key = [p], p[0]
    clusters.append(cur)

    levels = []
    for c in clusters:
        first_idx, last_idx = min(p[3] for p in c), max(p[3] for p in c)
        w = [1.0 + (p[3] - cut) / max(1, n - cut) for p in c]        # recency-weighted price
        price_lvl = float(np.average([p[0] for p in c], weights=w))
        lo, hi = min(p[1] for p in c), max(p[2] for p in c)
        tests = len(c)                                               # each pivot = a reversal there
        inside_prev = False                                          # + distinct later re-test episodes
        for j in range(first_idx + 1, n):
            inside = high[j] >= lo and low[j] <= hi
            if inside and not inside_prev:
                tests += 1
            inside_prev = inside
        if tests < config.SR_MIN_STRENGTH:
            continue
        levels.append((price_lvl, lo, hi, min(tests, 25), last_idx))

    # 3. classify live by the latest close; a level whose band CONTAINS price is judged by its center
    support, resistance = [], []
    for price_lvl, lo, hi, strength, _ in levels:
        if close > hi:
            kind = "support"
        elif close < lo:
            kind = "resistance"
        else:
            kind = "resistance" if price_lvl >= close else "support"
        dist = abs(close - price_lvl)
        rec = {"price": round(price_lvl, 2), "lo": round(lo, 2), "hi": round(hi, 2), "kind": kind,
               "strength": int(strength), "dist_atr": round(dist / atr, 2),
               "dist_pct": (round(dist / close * 100, 2) if close else None)}
        (support if kind == "support" else resistance).append(rec)

    support.sort(key=lambda x: x["dist_atr"])                        # nearest first
    resistance.sort(key=lambda x: x["dist_atr"])
    nn = config.SR_LIVE_COUNT
    support, resistance = support[:nn], resistance[:nn]
    return {"support": support, "resistance": resistance,
            "nearest_support": (support[0]["dist_atr"] if support else None),
            "nearest_resistance": (resistance[0]["dist_atr"] if resistance else None)}
