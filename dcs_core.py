"""dcs_core — DCStruct-Py engine core (plan P1; reference = pine/patterns/dc_struct_lib.pine v4).

Faithful port of the library's structStep pivot/regime/trend half (character lands in dcs_character,
P2). One sequential pass per frame — the engine is inherently path-dependent state, so it's a loop,
not vectorized (~180k 5m bars run in seconds; fine for the lab).

Semantics modes (plan D2):
- pine_compat=True — replicate Pine exactly for the parity harness: parent-bar OHLC only, and the
  outside-bar ORDER decided by the close-direction heuristic (red bar -> high printed first; doji ->
  assume swing continuation). All bars are closed offline, so the structural-confirmation gate is
  active on every bar — identical to a freshly loaded Pine chart (the live-bar exemption only ever
  applied to the realtime bar; the known live-vs-reload repaint asymmetry is documented in v3/v4).
- lab mode (default) — when the TFFrame carries 5m children, the outside-bar order comes from the
  TRUE intrabar sequence (which extreme printed first inside the parent bar); bars without children
  fall back to the heuristic. Everything else is identical.

Pinned conventions: ATR length 14 (Wilder RMA, SMA seed — matches ta.atr), regime strings byte-equal
to Pine (unicode arrows included) so the parity diff is a string compare. Pivots are NOT capped at 60
(Pine caps for memory only; the logic never looks deeper than 4 back).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dcs_data import TFFrame

ATR_LEN = 14

# Regime strings — byte-equal to Pine.
R_UNDEF = "UNDEFINED"
R_UP = "UPTREND"
R_DN = "DOWNTREND"
R_CON = "CONTRACTING RANGE"
R_EXP = "EXPANDING RANGE"
R_ASC = "ASCENDING RANGE (EQH+HL)"
R_DES = "DESCENDING RANGE (LH+EQL)"
R_EQH_LL = "RANGE (EQH+LL)"
R_HH_EQL = "RANGE (HH+EQL)"
R_BOX = "FLAT BOX"
R_TR_UP = "TRANSITION ↑ (CHoCH)"
R_TR_DN = "TRANSITION ↓ (CHoCH)"

# devSwing codes (match the Pine tuple contract): 0 none · 1 next leg developing · 2 pulling back ·
# 3 probing for a new high · 4 probing for a new low
DEV_NONE, DEV_LEG, DEV_PULLBACK, DEV_PROBE_HI, DEV_PROBE_LO = 0, 1, 2, 3, 4


def atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = ATR_LEN) -> np.ndarray:
    """ta.atr(n): Wilder RMA of true range, SMA-seeded at bar n-1; NaN before that."""
    m = len(close)
    tr = np.empty(m)
    tr[0] = high[0] - low[0]
    pc = close[:-1]
    tr[1:] = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - pc), np.abs(low[1:] - pc)))
    out = np.full(m, np.nan)
    if m < n:
        return out
    out[n - 1] = tr[:n].mean()
    a = 1.0 / n
    for i in range(n, m):
        out[i] = out[i - 1] + a * (tr[i] - out[i - 1])
    return out


@dataclass
class Pivot:
    idx: int          # bar index of the pivot extreme
    price: float
    is_high: bool
    tag: str          # H/L/HH/HL/LH/LL/EQH/EQL
    conf: int         # bar index where it confirmed


@dataclass
class CoreState:
    """Per-bar engine outputs (arrays of length n) + the confirmed pivot list."""
    n: int
    index: pd.DatetimeIndex
    atr: np.ndarray
    regime: list                 # str per bar
    dev_regime: list             # str | None per bar (provisional, may repaint)
    kind: np.ndarray             # int8: +1 alive uptrend, -1 alive downtrend, 0 none
    legs: np.ndarray             # confirmed impulse legs of the alive trend
    legs_disp: np.ndarray        # display legs (+1 provisional at early-lock genesis)
    dc_dir: np.ndarray           # int8 developing-pivot side: +1 tracking up, -1 down, 0 idle
    origin_idx: np.ndarray       # bar index of trend origin (-1 none)
    origin_price: np.ndarray
    prot: np.ndarray             # protected level (NaN none)
    ext: np.ndarray              # trend extreme so far (NaN none)
    len_atr: np.ndarray          # |ext - origin| / ATR
    dev_swing: np.ndarray        # int8 DEV_* codes
    t_id: np.ndarray
    via_early: np.ndarray        # bool: regime set via early lock THIS bar
    defined: np.ndarray          # bool: trend genesis this bar
    ended: np.ndarray            # bool: protected level breached this bar
    resumed: np.ndarray          # bool: alive trend re-entered its regime this bar
    leg_added: np.ndarray        # bool: confirmed impulse leg added this bar
    ch_up: np.ndarray            # bool: CHoCH up this bar
    ch_dn: np.ndarray
    brk: np.ndarray              # int8 internal-breaks direction
    ext_p: np.ndarray = None     # developing-pivot extreme price per bar (NaN while idle)
    ext_b: np.ndarray = None     # its bar index (-1 while idle)
    pivots: list = field(default_factory=list)   # list[Pivot], chronological

    def pivot_df(self) -> pd.DataFrame:
        return pd.DataFrame([{"idx": p.idx, "time": self.index[p.idx], "price": p.price,
                              "is_high": p.is_high, "tag": p.tag, "conf": p.conf,
                              "conf_time": self.index[p.conf]} for p in self.pivots])


def _hi_first_children(ch: pd.DataFrame):
    """True intrabar order from child bars: did the parent's high print before its low?
    Returns True/False, or None when undecidable (same child bar makes both extremes)."""
    hi = ch["high"].values
    lo = ch["low"].values
    ih = int(np.argmax(hi))
    il = int(np.argmin(lo))
    if ih == il:
        return None
    return ih < il


def run(source, mult: float = 1.0, eq_frac: float = 0.0, early: bool = True,
        pine_compat: bool = False) -> CoreState:
    """Run the engine over a TFFrame (children used in lab mode) or a bare OHLC DataFrame."""
    if isinstance(source, TFFrame):
        df, tfr = source.df, source
    else:
        df, tfr = source, None
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(df)
    atr = atr_wilder(h, l, c)

    st = CoreState(
        n=n, index=df.index, atr=atr,
        regime=[R_UNDEF] * n, dev_regime=[None] * n,
        kind=np.zeros(n, np.int8), legs=np.zeros(n, np.int64), legs_disp=np.zeros(n, np.int64),
        dc_dir=np.zeros(n, np.int8),
        origin_idx=np.full(n, -1, np.int64), origin_price=np.full(n, np.nan),
        prot=np.full(n, np.nan), ext=np.full(n, np.nan), len_atr=np.full(n, np.nan),
        dev_swing=np.zeros(n, np.int8), t_id=np.zeros(n, np.int64),
        via_early=np.zeros(n, bool), defined=np.zeros(n, bool), ended=np.zeros(n, bool),
        resumed=np.zeros(n, bool), leg_added=np.zeros(n, bool),
        ch_up=np.zeros(n, bool), ch_dn=np.zeros(n, bool), brk=np.zeros(n, np.int8),
        ext_p=np.full(n, np.nan), ext_b=np.full(n, -1, np.int64),
    )

    # ── engine state (mirrors the library's var block, same names where practical) ──
    dcdir = 0
    extP = math.nan
    extB = -1
    upE = math.nan
    upB = -1
    dnE = math.nan
    dnB = -1
    pB: list[int] = []
    pP: list[float] = []
    pHi: list[bool] = []
    pTx: list[str] = []
    reg = R_UNDEF
    refHi = math.nan
    refHiBk = True
    refLo = math.nan
    refLoBk = True
    rgHi = math.nan
    rgLo = math.nan
    brk = 0
    kind = 0            # +1/-1/0 (Pine "UP"/"DOWN"/"")
    tid = 0
    legs = 0
    oriB = -1
    oriP = math.nan
    ext2 = math.nan
    prLo = math.nan
    prHi = math.nan
    regSeen = R_UNDEF
    lgLk = False

    def _hi_first(i: int, doji_default: bool) -> bool:
        """Outside-bar order: children when available (lab mode), else the Pine heuristic.
        doji_default is what f_hiFirst returns on close==open: False for the up-leg check (d=1),
        True for the down-leg check (d=-1)."""
        if not pine_compat and tfr is not None:
            ch = tfr.children(i)
            if ch is not None and len(ch) > 1:
                hf = _hi_first_children(ch)
                if hf is not None:
                    return hf
        if c[i] < o[i]:
            return True
        if c[i] > o[i]:
            return False
        return doji_default

    for i in range(n):
        aV = atr[i]
        thr = aV * mult if not math.isnan(aV) else math.nan
        eqT = aV * eq_frac if not math.isnan(aV) else 0.0
        hh1 = i > 0 and h[i] > h[i - 1]
        ll1 = i > 0 and l[i] < l[i - 1]
        pivNew = False
        doCf = False
        cfHi = False
        cfB = -1
        cfP = math.nan

        # ── DC pivot step (structural confirmation active on every closed bar) ──
        if dcdir == 0:
            if math.isnan(upE) or h[i] > upE:
                upE, upB = h[i], i
            if math.isnan(dnE) or l[i] < dnE:
                dnE, dnB = l[i], i
            if not math.isnan(thr):
                if upE - l[i] >= thr and ll1:
                    doCf, cfHi, cfB, cfP = True, True, upB, upE
                    dcdir, extP, extB = -1, l[i], i
                elif h[i] - dnE >= thr and hh1:
                    doCf, cfHi, cfB, cfP = True, False, dnB, dnE
                    dcdir, extP, extB = 1, h[i], i
        elif dcdir == 1:
            nH = h[i] > extP
            if nH:
                extP, extB = h[i], i
            ordU = _hi_first(i, doji_default=False) if nH else True
            if ll1 and ordU and not math.isnan(thr) and extP - l[i] >= thr:
                doCf, cfHi, cfB, cfP = True, True, extB, extP
                dcdir, extP, extB = -1, l[i], i
        else:
            nL = l[i] < extP
            if nL:
                extP, extB = l[i], i
            # Pine: not f_hiFirst(-1) — the LOW must have printed first; doji -> continuation (False)
            ordD = (not _hi_first(i, doji_default=True)) if nL else True
            if hh1 and ordD and not math.isnan(thr) and h[i] - extP >= thr:
                doCf, cfHi, cfB, cfP = True, False, extB, extP
                dcdir, extP, extB = 1, h[i], i

        if doCf:
            prevSame = pP[-2] if len(pP) >= 2 else math.nan
            if cfHi:
                tag = "H" if math.isnan(prevSame) else (
                    "HH" if cfP > prevSame + eqT else "LH" if cfP < prevSame - eqT else "EQH")
            else:
                tag = "L" if math.isnan(prevSame) else (
                    "LL" if cfP < prevSame - eqT else "HL" if cfP > prevSame + eqT else "EQL")
            pB.append(cfB)
            pP.append(cfP)
            pHi.append(cfHi)
            pTx.append(tag)
            st.pivots.append(Pivot(cfB, cfP, cfHi, tag, i))
            pivNew = True

        nP = len(pB)
        # ── recent pivots ──
        rHi = rLo = math.nan
        rHiB = rLoB = -1
        if nP >= 1:
            if pHi[-1]:
                rHi, rHiB = pP[-1], pB[-1]
                if nP >= 2:
                    rLo, rLoB = pP[-2], pB[-2]
            else:
                rLo, rLoB = pP[-1], pB[-1]
                if nP >= 2:
                    rHi, rHiB = pP[-2], pB[-2]

        # ── minimal BOS/CHoCH (TRANSITION states + internal-breaks direction) ──
        if pivNew:
            if pHi[-1]:
                refHi, refHiBk = pP[-1], False
            else:
                refLo, refLoBk = pP[-1], False
            if nP >= 4:
                lh4 = pHi[-1]
                h1v = pP[-1] if lh4 else pP[-2]
                h2v = pP[-3] if lh4 else pP[-4]
                l1v = pP[-2] if lh4 else pP[-1]
                l2v = pP[-4] if lh4 else pP[-3]
                rgHi = max(h1v, h2v)
                rgLo = min(l1v, l2v)
        regExp0 = "EXPANDING" in reg
        regRct0 = "FLAT" in reg
        bHi = rgHi if regRct0 else refHi
        bLo = rgLo if regRct0 else refLo
        chU = chD = False
        if not regExp0 and not refHiBk and not math.isnan(bHi) and h[i] > bHi:
            chU = brk == -1
            brk = 1
            refHiBk = True
        if not regExp0 and not refLoBk and not math.isnan(bLo) and l[i] < bLo:
            chD = brk == 1
            brk = -1
            refLoBk = True

        # ── trend continuity: the protected level IS the CHoCH level — the previous higher-low
        # (uptrend) / lower-high (downtrend). Breaking it ends the trend (Amir 2026-07-14). The
        # level is kept on the MOST RECENT such pivot by the tracking further below. ──
        tDef = tRes = tEnd = False
        if kind == 1 and not math.isnan(prLo) and l[i] < prLo:
            kind, tEnd = 0, True
        if kind == -1 and not math.isnan(prHi) and h[i] > prHi:
            kind, tEnd = 0, True

        # ── regime classification on confirmed pivots ──
        if pivNew and nP >= 4:
            lastHi2 = pHi[-1]
            hA = pP[-1] if lastHi2 else pP[-2]
            hB = pP[-3] if lastHi2 else pP[-4]
            lA = pP[-2] if lastHi2 else pP[-1]
            lB = pP[-4] if lastHi2 else pP[-3]
            hiD = 1 if hA > hB + eqT else -1 if hA < hB - eqT else 0
            loD = 1 if lA > lB + eqT else -1 if lA < lB - eqT else 0
            if hiD == 1 and loD == 1:
                reg = R_UP if lastHi2 else reg
            elif hiD == -1 and loD == -1:
                reg = R_DN if not lastHi2 else reg
            elif hiD == -1 and loD == 1:
                reg = R_CON
            elif hiD == 1 and loD == -1:
                reg = R_EXP
            elif hiD == 0 and loD == 1:
                reg = R_ASC
            elif hiD == -1 and loD == 0:
                reg = R_DES
            elif hiD == 0 and loD == -1:
                reg = R_EQH_LL
            elif hiD == 1 and loD == 0:
                reg = R_HH_EQL
            else:
                reg = R_BOX
        if chU:
            reg = R_TR_UP
        if chD:
            reg = R_TR_DN

        # ── early regime lock via the developing pivot (absorbing outcomes only) ──
        if pivNew:
            lgLk = False
        viaEL = False
        if early and not lgLk and dcdir != 0 and nP >= 3:
            if dcdir == 1:
                lastHigh3 = pP[-2]
                if extP > lastHigh3 + eqT:
                    lA3, lB3 = pP[-1], pP[-3]
                    loD3 = 1 if lA3 > lB3 + eqT else -1 if lA3 < lB3 - eqT else 0
                    reg = R_UP if loD3 == 1 else R_EXP if loD3 == -1 else R_HH_EQL
                    lgLk = viaEL = True
            else:
                lastLow3 = pP[-2]
                if extP < lastLow3 - eqT:
                    hA3, hB3 = pP[-1], pP[-3]
                    hiD3 = 1 if hA3 > hB3 + eqT else -1 if hA3 < hB3 - eqT else 0
                    reg = R_DN if hiD3 == -1 else R_EXP if hiD3 == 1 else R_EQH_LL
                    lgLk = viaEL = True

        # ── alive-trend overrides ──
        if reg == R_HH_EQL and kind == 1:
            reg = R_UP
        if reg == R_EQH_LL and kind == -1:
            reg = R_DN

        # ── trend genesis / resumption (origin offset: 3 back via early lock, else 4) ──
        eUp = reg == R_UP and regSeen != R_UP
        eDn = reg == R_DN and regSeen != R_DN
        if eUp:
            if kind == 1:
                tRes = True
            else:
                tid += 1
                kind = 1
                oo = 3 if viaEL else 4
                oriP = pP[-oo] if nP >= oo else rLo
                oriB = pB[-oo] if nP >= oo else rLoB
                legs = 1 if nP >= oo else 0
                ext2 = h[i]
                tDef = True
            if not math.isnan(rLo):
                prLo = rLo
        if eDn:
            if kind == -1:
                tRes = True
            else:
                tid += 1
                kind = -1
                oo = 3 if viaEL else 4
                oriP = pP[-oo] if nP >= oo else rHi
                oriB = pB[-oo] if nP >= oo else rHiB
                legs = 1 if nP >= oo else 0
                ext2 = l[i]
                tDef = True
            if not math.isnan(rHi):
                prHi = rHi
        legAdd = False
        if pivNew and kind == 1 and nP >= 1 and pHi[-1] and pTx[-1] == "HH":
            legs += 1
            legAdd = True
        if pivNew and kind == -1 and nP >= 1 and not pHi[-1] and pTx[-1] == "LL":
            legs += 1
            legAdd = True
        # invalidation tracks the MOST RECENT opposite pivot (Amir 2026-07-14): update prLo on a new
        # low pivot in an uptrend / prHi on a new high pivot in a downtrend — as soon as the pullback
        # pivot confirms, not only when the next impulse does. Otherwise it lags one pivot and misses
        # the CHoCH break of the nearest higher-low / lower-high.
        if pivNew and kind == 1 and not pHi[-1] and not math.isnan(rLo):
            prLo = rLo
        if pivNew and kind == -1 and pHi[-1] and not math.isnan(rHi):
            prHi = rHi
        if kind == 1:
            ext2 = h[i] if math.isnan(ext2) else max(ext2, h[i])
        if kind == -1:
            ext2 = l[i] if math.isnan(ext2) else min(ext2, l[i])
        regSeen = reg

        legProv = tDef and not pivNew
        legsD = legs + (1 if legProv else 0)
        lenAtr = (abs(ext2 - oriP) / aV) if (kind != 0 and not math.isnan(oriP)
                                             and not math.isnan(ext2) and not math.isnan(aV)
                                             and aV > 0) else math.nan
        devC = (DEV_NONE if dcdir == 0 else
                (DEV_LEG if dcdir == 1 else DEV_PULLBACK) if kind == 1 else
                (DEV_LEG if dcdir == -1 else DEV_PULLBACK) if kind == -1 else
                DEV_PROBE_HI if dcdir == 1 else DEV_PROBE_LO)

        # ── developing regime (provisional, may repaint) ──
        devR = None
        if nP >= 3 and dcdir != 0:
            dHi = dcdir == 1
            hAx = extP if dHi else pP[-1]
            hBx = pP[-2] if dHi else pP[-3]
            lAx = pP[-1] if dHi else extP
            lBx = pP[-3] if dHi else pP[-2]
            hiDd = 1 if hAx > hBx + eqT else -1 if hAx < hBx - eqT else 0
            loDd = 1 if lAx > lBx + eqT else -1 if lAx < lBx - eqT else 0
            if hiDd == 1 and loDd == 1:
                devR = R_UP if dHi else reg
            elif hiDd == -1 and loDd == -1:
                devR = R_DN if not dHi else reg
            elif hiDd == -1 and loDd == 1:
                devR = R_CON
            elif hiDd == 1 and loDd == -1:
                devR = R_EXP
            elif hiDd == 0 and loDd == 1:
                devR = R_ASC
            elif hiDd == -1 and loDd == 0:
                devR = R_DES
            elif hiDd == 0 and loDd == -1:
                devR = R_EQH_LL
            elif hiDd == 1 and loDd == 0:
                devR = R_HH_EQL
            else:
                devR = R_BOX

        # ── record the bar ──
        st.regime[i] = reg
        st.dev_regime[i] = devR
        st.kind[i] = kind
        st.legs[i] = legs if kind != 0 else 0
        st.legs_disp[i] = legsD if kind != 0 else 0
        st.dc_dir[i] = dcdir
        st.origin_idx[i] = oriB if kind != 0 else -1
        st.origin_price[i] = oriP if kind != 0 else math.nan
        st.prot[i] = (prLo if kind == 1 else prHi if kind == -1 else math.nan)
        st.ext[i] = ext2 if kind != 0 else math.nan
        st.len_atr[i] = lenAtr
        st.dev_swing[i] = devC
        st.t_id[i] = tid
        st.via_early[i] = viaEL
        st.defined[i] = tDef
        st.ended[i] = tEnd
        st.resumed[i] = tRes
        st.leg_added[i] = legAdd
        st.ch_up[i] = chU
        st.ch_dn[i] = chD
        st.brk[i] = brk
        st.ext_p[i] = extP
        st.ext_b[i] = extB

    return st
