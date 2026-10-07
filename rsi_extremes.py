"""RSI Historical Extremes — port of Amir's rsi_extreme.pine (2026-07-03).

A symbol's OWN RSI history defines its personal overbought/oversold zones:
- all-time extremes: expanding max/min of Wilder RSI(RSIX_LEN)
- recent extremes:   rolling RSIX_REC-bar max/min
- zones:             RSIX_BUF RSI-points inside each extreme

Point-in-time note (breakout-day blindness rule): the reference extremes for bar t are built
from bars < t (shift 1). The Pine original compares against a level that already includes the
test bar — self-referential on fresh-extreme days; this port fixes that.

events(d) -> historical zone-entry / extreme-break rows (registry build, chart study)
state(d)  -> live snapshot for the ◉ badge + situational awareness
ob_trigger_price / indicators.rsi_trigger_price -> the price a bar must WICK to for its RSI
to print a level (the intrabar fire, same mechanics as the validated backburner).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def _rsi(close: pd.Series, length: int) -> pd.Series:
    chg = close.astype(float).diff()
    up = chg.clip(lower=0).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    dn = (-chg).clip(lower=0).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    return 100 - 100 / (1 + up / dn)


def series(d: pd.DataFrame):
    """(rsi, all_hi, all_lo, rec_hi, rec_lo) — extremes shifted 1 bar (test bar excluded)."""
    L = config.RSIX_LEN
    r = d["rsi14"].astype(float) if ("rsi14" in d.columns and L == 14) else _rsi(d["close"], L)
    prev = r.shift(1)
    return (r, prev.cummax(), prev.cummin(),
            prev.rolling(config.RSIX_REC, min_periods=L * 2).max(),
            prev.rolling(config.RSIX_REC, min_periods=L * 2).min())


def events(d: pd.DataFrame) -> list[dict]:
    """Every historical zone entry / extreme break: {t, event, rsi, level} sorted by bar."""
    r, ahi, alo, rhi, rlo = series(d)
    buf = config.RSIX_BUF
    rp = r.shift(1)

    def x_over(z):
        return ((r >= z) & (rp < z.shift(1))).to_numpy(dtype=bool)

    def x_under(z):
        return ((r <= z) & (rp > z.shift(1))).to_numpy(dtype=bool)

    spec = [("enter_ath_zone", x_over(ahi - buf), ahi), ("break_ath", x_over(ahi), ahi),
            ("enter_atl_zone", x_under(alo + buf), alo), ("break_atl", x_under(alo), alo),
            ("enter_rechi_zone", x_over(rhi - buf), rhi), ("enter_reclo_zone", x_under(rlo + buf), rlo)]
    out = []
    for ev, mask, lvl in spec:
        lv = lvl.to_numpy(float)
        rv = r.to_numpy(float)
        for t in np.flatnonzero(mask):
            if not np.isnan(lv[t]):
                out.append({"t": int(t), "event": ev,
                            "rsi": round(float(rv[t]), 2), "level": round(float(lv[t]), 2)})
    out.sort(key=lambda x: x["t"])
    return out


def state(d: pd.DataFrame) -> dict | None:
    """Live snapshot: current zone ('ath'|'atl'|'rec_hi'|'rec_lo'|None), broke flag, the
    extreme levels, and bars_since the last all-time-zone event (drives the 'recent' badge)."""
    if d is None or len(d) < config.RSIX_LEN * 3:
        return None
    r, ahi, alo, rhi, rlo = series(d)
    cur, hi, lo = float(r.iloc[-1]), float(ahi.iloc[-1]), float(alo.iloc[-1])
    if np.isnan(cur) or np.isnan(hi):
        return None
    rh = float(rhi.iloc[-1]) if pd.notna(rhi.iloc[-1]) else None
    rl = float(rlo.iloc[-1]) if pd.notna(rlo.iloc[-1]) else None
    buf, at = config.RSIX_BUF, config.RSIX_AT_TOL
    # 'ath'/'atl' = AT the actual extreme (±AT_TOL) — the only states allowed to say "extreme";
    # 'near_*' = inside the buffer zone but not at it (approaching — display-only, never a fire).
    # ROLLING-200 zones (rec_hi/rec_lo) were DROPPED from the read entirely (Amir 2026-07-08 — badge
    # noise; a name at just its 200-bar RSI extreme isn't notable). All-time zones only, everywhere.
    zone = ("ath" if cur >= hi - at else "atl" if cur <= lo + at
            else "near_ath" if cur >= hi - buf else "near_atl" if cur <= lo + buf else None)
    evs = [e for e in events(d) if e["event"] in
           ("enter_ath_zone", "enter_atl_zone", "break_ath", "break_atl")]
    last = evs[-1] if evs else None
    return {"rsi": round(cur, 1), "all_hi": round(hi, 1), "all_lo": round(lo, 1),
            "rec_hi": (round(rh, 1) if rh is not None else None),
            "rec_lo": (round(rl, 1) if rl is not None else None),
            "zone": zone, "broke": bool(cur > hi or cur < lo),
            "last_event": (last["event"] if last else None),
            "bars_since": (len(d) - 1 - last["t"]) if last else None}


def os_trigger_price(close: pd.Series, level: pd.Series | float, length: int | None = None) -> pd.Series:
    """Price a bar must trade DOWN to for RSI to print `level` (Series-level variant of
    indicators.rsi_trigger_price — the extreme zone edge moves with history)."""
    length = length or config.RSIX_LEN
    chg = close.astype(float).diff()
    u = chg.clip(lower=0).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    dn = (-chg).clip(lower=0).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    lvl = level if isinstance(level, pd.Series) else pd.Series(level, index=close.index)
    x = (100.0 - lvl) / lvl
    return (close.shift(1) - (length - 1) * (u.shift(1) * x - dn.shift(1))).astype(float)


def ob_trigger_price(close: pd.Series, level: pd.Series | float, length: int | None = None) -> pd.Series:
    """Price a bar must trade UP to for RSI to print `level` — the overbought mirror:
    g = (len-1)*(rmaD[1]*lvl/(100-lvl) - rmaU[1]); pOB = close[1] + g."""
    length = length or config.RSIX_LEN
    chg = close.astype(float).diff()
    u = chg.clip(lower=0).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    dn = (-chg).clip(lower=0).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    lvl = level if isinstance(level, pd.Series) else pd.Series(level, index=close.index)
    x = lvl / (100.0 - lvl)
    return (close.shift(1) + (length - 1) * (dn.shift(1) * x - u.shift(1))).astype(float)


def fires(d: pd.DataFrame, side: str, collect: list | None = None):
    """Backburner-style state machine on the ALL-TIME zone edge. side='long': fire when the LOW
    wicks the price where RSI = all-time-low + buf; side='short': the HIGH wicks the all-time-high
    − buf price. One fire per episode; re-arm when close-RSI crosses back through 50 (neutral).
    Returns (last_fire_t, trigger_price_series, armed). `collect` gets every historical fire t."""
    r, ahi, alo, _, _ = series(d)
    at = config.RSIX_AT_TOL                      # fire only AT the actual extreme (Amir: the
    if side == "long":                           # buffer zone is "approaching", never a fire)
        lvl = (alo + at).clip(lower=2.0, upper=45.0)
        p = os_trigger_price(d["close"], lvl)
        wick = d["low"].to_numpy(float) <= p.to_numpy(float)
    else:
        lvl = (ahi - at).clip(lower=55.0, upper=98.0)
        p = ob_trigger_price(d["close"], lvl)
        wick = d["high"].to_numpy(float) >= p.to_numpy(float)
    rv = r.to_numpy(float)
    armed, last_fire = True, None
    for t in range(1, len(d)):
        if np.isnan(rv[t]) or np.isnan(p.iloc[t]):
            continue
        if not armed:                                    # re-arm on a close-RSI reclaim of neutral
            if (side == "long" and rv[t] >= 50) or (side == "short" and rv[t] <= 50):
                armed = True
        elif wick[t]:
            last_fire, armed = t, False
            if collect is not None:
                collect.append(t)
    return last_fire, p, armed
