"""dcs_levels — parent swing levels + proximity tags (plan P3; port of v4's f_mtSrDraw selection).

Pure selection over a parent's confirmed pivots: nearest N above + N below a reference price,
deduped within a tolerance — the same levels the v4 chart draws as dotted lines. The CAUSAL variant
only considers pivots already confirmed by the aligned parent bar (events-ledger safe).
"""
from __future__ import annotations

from dataclasses import dataclass

import math


@dataclass
class Level:
    tf: str
    price: float
    is_high: bool
    time: object          # pivot bar time
    side: str             # "above" | "below" (relative to the reference price)
    dist: float


def select(pivots, ref_price: float, n_side: int = 2, tol: float = 0.0, tf: str = "",
           index=None, max_back: int = 12, upto_conf: int | None = None) -> list:
    """Nearest-N-per-side over the last `max_back` pivots (mirrors the Pine selection).
    `upto_conf`: only pivots with conf <= this bar index (causal cut)."""
    pool = [p for p in pivots if upto_conf is None or p.conf <= upto_conf][-max_back:]
    out: list[Level] = []
    used = [False] * len(pool)
    for side in (0, 1):
        for _ in range(n_side):
            best, best_d = -1, math.inf
            for i, p in enumerate(pool):
                if used[i]:
                    continue
                ok = p.price > ref_price if side == 0 else p.price < ref_price
                if not ok:
                    continue
                d = abs(p.price - ref_price)
                if d >= best_d:
                    continue
                if any(used[j] and abs(pool[j].price - p.price) <= tol for j in range(len(pool))):
                    continue
                best, best_d = i, d
            if best < 0:
                continue
            used[best] = True
            p = pool[best]
            out.append(Level(tf=tf, price=p.price, is_high=p.is_high,
                             time=(index[p.idx] if index is not None else p.idx),
                             side="above" if side == 0 else "below", dist=best_d))
    return out


def parent_levels(mtf, i: int, n_side: int = 2, tol_atr: float = 0.25) -> list:
    """All enabled parents' levels at child bar i (CAUSAL: pivots confirmed by the aligned parent
    bar only), deduped per parent within max(0.25*childATR, 0)."""
    ref = float(mtf.child.df["close"].iloc[i])
    a = float(mtf.core.atr[i]) if mtf.core.atr[i] == mtf.core.atr[i] else 0.0
    tol = tol_atr * a
    out = []
    for pv in mtf.parents:
        k = int(pv.map_causal[i])
        if k < 0:
            continue
        out.extend(select(pv.core.pivots, ref, n_side=n_side, tol=tol, tf=pv.tf,
                          index=pv.core.index, upto_conf=k))
    return out


def proximity_tags(levels, ref_price: float, atr: float, near_atr: float = 0.5) -> list:
    """Which parent levels sit within near_atr ATRs of the reference — CAUTION/confluence input."""
    if not (atr == atr and atr > 0):
        return []
    return [lv for lv in levels if abs(lv.price - ref_price) <= near_atr * atr]
