"""dcs_character — trend-character classifier (plan P2; reference = dc_struct_lib.pine v4 structStep).

Second pass over a dcs_core.CoreState: leg features (pivot geometry EXACT; within-leg ER/convexity
from parent closes in pine_compat mode or from 5m CHILD closes in lab mode — the child-feeds-parent
design), sequence state with genesis backfill, whole-trend ER/R2, the EMA-rider arm-exit machine, and
the PARABOLIC/CLEAN/GRIND/CHANNEL/CHOPPY scorer with the WEAKENING override, CLIMACTIC tag and 3-bar
hysteresis. Thresholds are the flagship's shipped defaults, hardcoded (same as the library).

RIDER: this module carries its OWN verbatim port of DCStruct.riderStep (SAVE-FIRST ordering: while
armed, a close back on-side is checked BEFORE the armed-level break). The app's emarider.py checks
the armed-level break first — on a bar that BOTH breaks the armed extreme intrabar AND closes back
on-side, the Pine suite says SAVED while emarider says BREAK. Both claim to port emaRider.pine, so
one deviates from the original; parity with the live Pine chart wins here, and the divergence is
pinned by tests/test_dcs_character.py for Amir to reconcile app-side later (discovered 2026-07-14).

No S/R-respect measure (no zone book) — identical to the library; those score components never fire.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dcs_data import TFFrame
from dcs_core import CoreState

FORMING = "FORMING"
INSUFF = "INSUFFICIENT DATA"
PARABOLIC = "ACCELERATING"   # constant name kept for code refs; VALUE renamed (Amir 2026-07-14)
CLEAN = "CLEAN"
GRIND = "GRIND"
CHANNEL = "CHANNEL"         # v4-parity VALUE only now (see CHANNEL_V5 below) — same
                             # constant-name-kept-value-renamed pattern as PARABOLIC above
# CHANNEL_V5 = "RHYTHMIC" (Amir APPROVED, FeaturesStructures/addendum-work-split.md #1, 2026-07-16):
# the character label describes CADENCE (CV<=0.35 in leg size/duration/pullback — nothing about
# parallel rails); Stage 2.5's geometry work needs "channel" for the geometric SHAPE
# (cont_type=channel_up/channel_down), which is universal chartist vocabulary not worth fighting.
# v4 stays the live parity reference until v5 is validated and PINE correctly never touched it (a
# Pine chart recomputes fresh every load — no persisted old-label data to translate there). Python's
# dependency is narrower and already has its hook: `run()`'s existing `pine_compat` flag (added to
# reconcile Python with v4's behavior, and `dcs_parity.py` already calls with pine_compat=True) picks
# the value — CHANNEL under pine_compat (matches the live v4 fixtures), CHANNEL_V5 otherwise (the lab
# -mode default every other PY caller uses). No parquet-history mapping shim needed here: this repo's
# data/dcs_lab/ output is fully regenerated on every rebuild (git worktree convention, data/ never
# committed), not appended to incrementally — there is no frozen historical CHANNEL row to reconcile.
CHANNEL_V5 = "RHYTHMIC"
CHOPPY = "CHOPPY"
WEAKENING = "WEAKENING"


def rider(high, low, close, atr, length: int = 20, near_frac: float = 0.5, arm_exit: bool = True):
    """Verbatim port of DCStruct.riderStep (see module docstring for the emarider divergence).
    Returns dict of per-bar arrays: streak, armed, saves, arms, deep, near, ext, ext_max, flip, ema."""
    n = len(close)
    ema = pd.Series(close).ewm(span=length, adjust=False).mean().to_numpy()
    out = {k: np.zeros(n, dtype=(bool if k in ("armed", "flip") else float if k in ("ext", "ext_max") else int))
           for k in ("streak", "armed", "saves", "arms", "deep", "near", "ext", "ext_max", "flip")}
    out["ema"] = ema
    streak = 0
    armed = False
    armLvl = math.nan
    armRun = 0
    saves = arms = deep = near = 0
    brLast = math.nan
    brMax = 0.0
    extMax = 0.0
    for i in range(n):
        a = atr[i]
        ext = (close[i] - ema[i]) / a if (a == a and a > 0) else math.nan
        flip = False
        if streak > 0:
            if armed:
                if close[i] > ema[i]:
                    streak += 1
                    armed = False
                    saves += 1
                    armRun = 0
                elif low[i] < armLvl:
                    armRun += 1
                    streak = -armRun
                    armed = False
                    flip = True
                else:
                    streak += 1
                    armRun += 1
            else:
                if close[i] > ema[i]:
                    streak += 1
                elif arm_exit:
                    streak += 1
                    armed = True
                    armLvl = low[i]
                    armRun = 1
                    arms += 1
                else:
                    streak = -1
                    flip = True
        elif streak < 0:
            if armed:
                if close[i] < ema[i]:
                    streak -= 1
                    armed = False
                    saves += 1
                    armRun = 0
                elif high[i] > armLvl:
                    armRun += 1
                    streak = armRun
                    armed = False
                    flip = True
                else:
                    streak -= 1
                    armRun += 1
            else:
                if close[i] < ema[i]:
                    streak -= 1
                elif arm_exit:
                    streak -= 1
                    armed = True
                    armLvl = high[i]
                    armRun = 1
                    arms += 1
                else:
                    streak = 1
                    flip = True
        else:
            streak = 1 if close[i] > ema[i] else -1
        if flip:
            saves = arms = armRun = 0
            armLvl = math.nan
            brLast = math.nan
            deep = 0
            brMax = 0.0
            near = 0
            extMax = 0.0
        if streak > 1 and low[i] < ema[i]:
            dB = ema[i] - low[i]
            brMax = max(brMax, dB)
            deep = deep + 1 if (brLast == brLast and dB > brLast) else 0
            brLast = dB
        elif streak < -1 and high[i] > ema[i]:
            dB2 = high[i] - ema[i]
            brMax = max(brMax, dB2)
            deep = deep + 1 if (brLast == brLast and dB2 > brLast) else 0
            brLast = dB2
        if a == a:
            if streak > 1 and low[i] <= ema[i] + near_frac * a:
                near += 1
            elif streak < -1 and high[i] >= ema[i] - near_frac * a:
                near += 1
        if ext == ext:
            extMax = max(extMax, abs(ext))
        out["streak"][i] = streak
        out["armed"][i] = armed
        out["saves"][i] = saves
        out["arms"][i] = arms
        out["deep"][i] = deep
        out["near"][i] = near
        out["ext"][i] = ext
        out["ext_max"][i] = extMax
        out["flip"][i] = flip
    return out


def _median(xs):
    n = len(xs)
    if n == 0:
        return math.nan
    s = sorted(xs)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2


def _cv(xs):
    n = len(xs)
    if n < 2:
        return math.nan
    mean = sum(xs) / n
    if mean == 0:
        return math.nan
    ss = sum((x - mean) ** 2 for x in xs)
    return math.sqrt(ss / n) / abs(mean)


def _push8(arr, v):
    arr.append(v)
    if len(arr) > 8:
        arr.pop(0)


def _leg_path(closes) -> tuple:
    """Within-leg ER + convexity over a close series (nan/nan when <3 samples)."""
    n = len(closes)
    if n < 3:
        return math.nan, math.nan
    c0, c1 = closes[0], closes[-1]
    net = c1 - c0
    path = float(np.abs(np.diff(closes)).sum())
    er = abs(net) / path if path > 0 else 0.0
    mI = n // 2
    chord = c0 + net * mI / max(1, n - 1)
    cvx = (closes[mI] - chord) / net if abs(net) > 1e-10 else math.nan
    return er, cvx


@dataclass
class CharState:
    n: int
    label: list                  # active character per bar ("" when no trend alive)
    raw: list                    # raw (pre-hysteresis) label per bar
    clim: np.ndarray             # CLIMACTIC tag
    whole_er: np.ndarray
    whole_r2: np.ndarray
    pb_dep: np.ndarray
    amp_rat: np.ndarray
    er_rat: np.ndarray
    arms_t: np.ndarray           # arm events since trend genesis
    rd: dict = field(default_factory=dict)   # rider arrays


def run(source, core: CoreState, pine_compat: bool = False) -> CharState:
    if isinstance(source, TFFrame):
        df, tfr = source.df, source
    else:
        df, tfr = source, None
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(df)
    atr = core.atr
    rd = rider(h, l, c, atr)
    # rvol (CLIMACTIC input): consolidated volume / SMA50, when volume exists
    if "volume" in df.columns:
        v = df["volume"].to_numpy(float)
        vs = pd.Series(v).rolling(50).mean().to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            rvol = np.where(vs > 0, v / vs, np.nan)
    else:
        rvol = np.full(n, np.nan)
    # cumulative sums for whole-trend ER / R2 (x = absolute bar index, relative-corrected like Pine)
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    cum_tr = np.cumsum(tr)
    cum_c = np.cumsum(c)
    cum_xc = np.cumsum(np.arange(n) * c)
    cum_c2 = np.cumsum(c * c)

    piv = core.pivots
    conf_at = {}                      # bar -> pivot list length AFTER that bar's confirmation
    for k, p in enumerate(piv):
        conf_at[p.conf] = k + 1

    st = CharState(n=n, label=[""] * n, raw=[""] * n, clim=np.zeros(n, bool),
                   whole_er=np.full(n, np.nan), whole_r2=np.full(n, np.nan),
                   pb_dep=np.full(n, np.nan), amp_rat=np.full(n, np.nan),
                   er_rat=np.full(n, np.nan), arms_t=np.zeros(n, np.int64), rd=rd)

    def leg_er_cvx(bar_a: int, bar_b: int):
        if not pine_compat and tfr is not None and tfr.base is not None:
            lo = int(tfr.child_lo[bar_a])
            hi = int(tfr.child_hi[bar_b])
            if hi - lo >= 3:
                return _leg_path(tfr.base["close"].to_numpy(float)[lo:hi])
        return _leg_path(c[bar_a:bar_b + 1])

    # sequence state (reset at genesis)
    iAmp: list = []
    iAmpR: list = []
    iDur: list = []
    iSlp: list = []
    iSlpR: list = []   # raw slope (rawAmp/dur) for drift-free ratios
    iEr: list = []
    pbR: list = []
    lastConv = math.nan
    ovC = ovT = 0
    iOK = iTot = 0
    ampRat = slpRat = erRat = pbDep = math.nan
    armsT = 0
    prevArms = 0
    sub = FORMING
    chal = ""
    chalB = 0
    np_seen = 0
    # pP/pB/pHi mirror piv[:np_seen] (full confirmed-pivot history, never reset on genesis) but are
    # maintained incrementally — appending only the newly-confirmed pivots each bar — instead of
    # rebuilding the whole slice every bar. The old `piv[:np_seen]` rebuild was O(bars x pivot_count):
    # invisible on the equity batch's pivot counts, but at 16yr/5m Globex scale (~1.1M bars, tens of
    # thousands of pivots) it never finished (found 2026-07-15 rebuilding ES/NQ on Databento data).
    pP: list = []
    pB: list = []
    pHi: list = []

    for i in range(n):
        aV = atr[i]
        kind = int(core.kind[i])
        piv_n = conf_at.get(i, np_seen)
        pivNew = piv_n > np_seen
        if pivNew:
            for p in piv[np_seen:piv_n]:
                pP.append(p.price)
                pB.append(p.idx)
                pHi.append(p.is_high)
        np_seen = piv_n
        tDef = bool(core.defined[i])
        viaEL = bool(core.via_early[i])
        nP = np_seen

        # arm-event counter since genesis (edge-detect: the machine's counter resets on flips)
        if tDef:
            armsT = 0
        if kind != 0 and rd["arms"][i] > prevArms:
            armsT += 1
        prevArms = int(rd["arms"][i])

        if tDef:
            iAmp, iAmpR, iDur, iSlp, iSlpR, iEr, pbR = [], [], [], [], [], [], []
            lastConv = math.nan
            ovC = ovT = 0
            iOK = iTot = 0
            ampRat = slpRat = erRat = pbDep = math.nan

        # leg processing: normal freeze = pair (n-2,n-1); genesis backfills the proving swings
        prFrom = (3 if viaEL else 4) if tDef else (2 if (pivNew and kind != 0) else 0)
        if kind != 0 and prFrom >= 2:
            for j in range(prFrom, 1, -1):
                sPv = nP - j
                ePv = nP - j + 1
                if sPv < 0 or ePv > nP - 1:
                    continue
                dirL = 1 if pHi[ePv] else -1
                rawPiv = abs(pP[ePv] - pP[sPv])
                durPiv = float(pB[ePv] - pB[sPv])
                ampPiv = rawPiv / aV if (aV == aV and aV > 0) else math.nan
                slpPiv = ampPiv / durPiv if (ampPiv == ampPiv and durPiv > 0) else math.nan
                lEr, lCvx = leg_er_cvx(pB[sPv], pB[ePv])
                isImp = (dirL == 1 and kind == 1) or (dirL == -1 and kind == -1)
                if isImp:
                    iTot += 1
                    if lEr == lEr:
                        iOK += 1
                    if ampPiv == ampPiv:
                        ok2 = len(iAmp) >= 2   # >=2 PRIOR legs — one prior is not acceleration
                        # amp/slope RATIOS use RAW price sizes, not ATR-normalized (Amir 2026-07-14):
                        # dividing each leg by the ATR at its own time injects ATR drift into the ratio.
                        # iAmp (ATR-normalized) is still kept for absolute-level thresholds (GRIND amp
                        # <= 2 ATR). ER is already unitless.
                        rawSlp = rawPiv / durPiv if durPiv > 0 else math.nan
                        mAr = _median(iAmpR) if ok2 else math.nan
                        mSr = _median(iSlpR) if ok2 else math.nan
                        mE = _median(iEr) if ok2 else math.nan
                        ampRat = rawPiv / mAr if (mAr == mAr and mAr > 0) else math.nan
                        slpRat = rawSlp / mSr if (mSr == mSr and mSr > 0 and rawSlp == rawSlp) else math.nan
                        erRat = lEr / mE if (mE == mE and mE > 0 and lEr == lEr) else math.nan
                        _push8(iAmp, ampPiv)
                        _push8(iAmpR, rawPiv)
                        _push8(iDur, durPiv)
                        if slpPiv == slpPiv:
                            _push8(iSlp, slpPiv)
                        if rawSlp == rawSlp:
                            _push8(iSlpR, rawSlp)
                        if lEr == lEr:
                            _push8(iEr, lEr)
                        if lCvx == lCvx:
                            lastConv = lCvx
                else:
                    liR = iAmpR[-1] if iAmpR else math.nan
                    if liR == liR and liR > 0 and rawPiv > 0:
                        pbDep = rawPiv / liR
                        _push8(pbR, pbDep)
                    if sPv >= 1:
                        mid = (pP[sPv - 1] + pP[sPv]) / 2
                        ovT += 1
                        if (pP[ePv] < mid) if kind == 1 else (pP[ePv] > mid):
                            ovC += 1
        # early-lock genesis: the developing leg is the confirming impulse — provisional push
        if tDef and viaEL and nP >= 1 and core.dc_dir[i] != 0:
            extP = core.ext_p[i]
            rawL = abs(extP - pP[-1])
            durL = float(i - pB[-1])
            ampL = rawL / aV if (aV == aV and aV > 0) else math.nan
            iTot += 1
            if ampL == ampL and rawL > 0:
                _push8(iAmp, ampL)
                _push8(iAmpR, rawL)
                _push8(iDur, max(durL, 1.0))
                if durL > 0:
                    _push8(iSlp, ampL / durL)
                    _push8(iSlpR, rawL / durL)
        # live pullback depth from the developing pivot
        if kind != 0 and not pivNew and nP >= 1:
            dcd = int(core.dc_dir[i])
            if (kind == 1 and dcd == -1) or (kind == -1 and dcd == 1):
                liR = iAmpR[-1] if iAmpR else math.nan
                if liR == liR and liR > 0:
                    pbDep = abs(core.ext_p[i] - pP[-1]) / liR

        # whole-trend ER + R2
        wEr = wR2 = math.nan
        oi = int(core.origin_idx[i])
        if kind != 0 and oi >= 0 and i > oi + 2:
            pathW = cum_tr[i] - cum_tr[oi]
            num = abs(core.ext[i] - core.origin_price[i])
            if pathW > 0:
                wEr = min(1.0, num / pathW)
            nW = i - oi
            Sy = cum_c[i] - cum_c[oi]
            Sic = cum_xc[i] - cum_xc[oi]
            Sy2 = cum_c2[i] - cum_c2[oi]
            a0 = oi + 1
            Sxy = Sic - a0 * Sy
            Sx = nW * (nW - 1) / 2.0
            Sxx = (nW - 1) * nW * (2 * nW - 1) / 6.0
            dX = nW * Sxx - Sx * Sx
            dY = nW * Sy2 - Sy * Sy
            if dX > 0 and dY > 0:
                wR2 = min(1.0, (nW * Sxy - Sx * Sy) ** 2 / (dX * dY))

        # classifier (flagship shipped defaults) + hysteresis
        if tDef:
            sub = FORMING
            chal = ""
            chalB = 0
        rawL2 = FORMING
        clim = False
        if kind != 0:
            ageB = i - oi if oi >= 0 else 0
            if iTot < 2:
                rawL2 = FORMING
            elif iOK / iTot < 0.5:
                rawL2 = INSUFF
            else:
                mAmp = _median(iAmp)
                mEr2 = _median(iEr)
                cvA = _cv(iAmp)
                cvD = _cv(iDur)
                cvP = _cv(pbR)
                ovR = ovC / ovT if ovT > 0 else math.nan
                armR = armsT / max(1.0, ageB)
                streak = int(rd["streak"][i])
                dwell = 100.0 * rd["near"][i] / abs(streak) if streak != 0 else 0.0
                rext = rd["ext"][i] * kind if rd["ext"][i] == rd["ext"][i] else math.nan
                scPar = ((35 if ampRat == ampRat and ampRat >= 1.3 else 0)
                         + (35 if slpRat == slpRat and slpRat >= 1.3 else 0)
                         + (15 if pbDep == pbDep and pbDep <= 0.3 else 0)
                         + (10 if lastConv == lastConv and lastConv <= -0.1 else 0)
                         + (15 if rext == rext and rext >= 2.5 else 0))
                scCln = ((25 if wEr == wEr and wEr >= 0.5 else 0)
                         + (20 if wR2 == wR2 and wR2 >= 0.5 else 0)
                         + (15 if pbDep == pbDep and pbDep <= 0.5 else 0)
                         + (15 if mEr2 == mEr2 and mEr2 >= 0.4 else 0)
                         + (10 if armR <= 0.05 else 0))
                scGr = ((30 if mAmp == mAmp and mAmp <= 2.0 else 0)
                        + (30 if dwell >= 60 else 0)
                        + (20 if abs(streak) >= 15 else 0)
                        + (20 if pbDep == pbDep and pbDep <= 0.4 else 0))
                scCh = ((30 if cvA == cvA and cvA <= 0.35 else 0)
                        + (20 if cvD == cvD and cvD <= 0.35 else 0)
                        + (20 if cvP == cvP and cvP <= 0.35 else 0)
                        + (15 if pbDep == pbDep and 0.5 <= pbDep <= 0.8 else 0)
                        + (15 if wEr == wEr and 0.3 <= wEr <= 0.6 else 0))
                scCp = ((30 if wEr == wEr and wEr <= 0.3 else 0)
                        + (30 if ovR == ovR and ovR >= 0.5 else 0)
                        + (25 if mEr2 == mEr2 and mEr2 <= 0.35 else 0))
                base, best = CHOPPY, scCp
                chan_lbl = CHANNEL if pine_compat else CHANNEL_V5
                for lbl, sc in ((PARABOLIC, scPar), (CLEAN, scCln), (GRIND, scGr), (chan_lbl, scCh)):
                    if sc > best:
                        base, best = lbl, sc
                wk = ((1 if ampRat == ampRat and ampRat <= 0.8 else 0)
                      + (1 if pbDep == pbDep and pbDep >= 0.6 else 0)
                      + (1 if ageB >= 10 and armR >= 0.08 else 0)
                      + (1 if rd["deep"][i] >= 2 else 0)
                      + (1 if erRat == erRat and erRat <= 0.8 else 0))
                rawL2 = WEAKENING if wk >= 3 else base
                clim = (ampRat == ampRat and ampRat >= 2.0
                        and rvol[i] == rvol[i] and rvol[i] >= 1.5)
            immGate = rawL2 in (FORMING, INSUFF) or sub in (FORMING, INSUFF)
            if rawL2 == sub:
                chal = ""
                chalB = 0
            elif immGate:
                sub = rawL2
                chal = ""
                chalB = 0
            else:
                if rawL2 == chal:
                    chalB += 1
                else:
                    chal = rawL2
                    chalB = 1
                if chalB >= 3:
                    sub = rawL2
                    chal = ""
                    chalB = 0

        st.label[i] = sub if kind != 0 else ""
        st.raw[i] = rawL2 if kind != 0 else ""
        st.clim[i] = clim
        st.whole_er[i] = wEr
        st.whole_r2[i] = wR2
        st.pb_dep[i] = pbDep if kind != 0 else math.nan
        st.amp_rat[i] = ampRat if kind != 0 else math.nan
        st.er_rat[i] = erRat if kind != 0 else math.nan
        st.arms_t[i] = armsT

    return st
