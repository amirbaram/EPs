"""Setup-quality score: blend a few normalized factors into a 0-100 number the
dashboard can sort on, so the best-looking setups float to the top.

Sub-scores (each in [0,1], ramped via config):
  adr      higher ADR (more tradable range)
  trend    quality of the prior up-move (net gain x clarity)
  tight    tightness of the consolidation (tighter = better)
  emasurf  closes "surfing" at/above a rising EMA10 & EMA20 through the base
  voldry   volume drying up across the consolidation

A setup only scores the factors that apply to it (event setups with no
consolidation — gapper, single-bar HVC, undercut — score on adr+trend only); the
composite renormalizes over whatever applied. Pure/causal: reads the hit fields
the detectors attached (cons_bars, prior_gain_pct, ...) plus in-window bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

SUBS = ("adr", "trend", "tight", "emasurf", "voldry", "catalyst")


def _ramp(x, lo, hi):
    """Map x onto [0,1] across lo->hi. Supports an inverted ramp (lo>hi), e.g.
    tightness where a SMALLER range scores higher."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    if lo == hi:
        return 1.0 if x >= lo else 0.0
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


def _first(row, *keys):
    for k in keys:
        v = row.get(k)
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            return v
    return None


def _q_trend(row):
    g = _first(row, "prior_gain_pct", "regime_net_pct")
    cl = _first(row, "prior_clarity", "regime_clarity")
    parts = []
    if g is not None:
        parts.append(_ramp(float(g), *config.Q_TREND_GAIN_RAMP))
    if cl is not None:
        parts.append(min(1.0, max(0.0, float(cl))))
    parts = [p for p in parts if p is not None]
    return sum(parts) / len(parts) if parts else None


def _window(d, row):
    cb = row.get("cons_bars")
    if cb is None or (isinstance(cb, float) and pd.isna(cb)):
        return None
    cb = int(cb)
    return d.iloc[-cb:] if cb >= 3 else None


def _q_tight(d, row):
    win = _window(d, row)
    if win is None:
        return None
    last = float(win["close"].iloc[-1])
    if last <= 0:
        return None
    rng = (float(win["high"].max()) - float(win["low"].min())) / last * 100
    return _ramp(rng, *config.Q_TIGHT_RAMP)


def _q_emasurf(d, row):
    win = _window(d, row)
    if win is None or "ema10" not in win or "ema20" not in win:
        return None
    e10, e20, c = win["ema10"], win["ema20"], win["close"]
    if e10.isna().any() or e20.isna().any():
        return None
    band = 1 - config.Q_EMASURF_BAND / 100
    surf = ((c >= e10 * band) & (c >= e20 * band)).mean()
    rising = e10.iloc[-1] > e10.iloc[0] and e20.iloc[-1] > e20.iloc[0]
    return float(surf) * (1.0 if rising else 0.4)


def _q_voldry(d, row):
    win = _window(d, row)
    if win is None or len(win) < 4:
        return None
    v = win["volume"].to_numpy(float)
    half = len(v) // 2
    first, second = v[:half].mean(), v[half:].mean()
    if not np.isfinite(first) or first <= 0:
        return None
    return _ramp(1.0 - second / first, *config.Q_VOLDRY_RAMP)


def score_hit(d: pd.DataFrame, row: dict) -> dict:
    """Compute the quality score for one hit. `row` = base liquidity columns merged
    with the detector hit; `d` = the matching enriched frame (daily or weekly)."""
    subs = {
        "adr": _ramp(_first(row, "adr_pct"), *config.Q_ADR_RAMP),
        "trend": _q_trend(row),
        "tight": _q_tight(d, row),
        "emasurf": _q_emasurf(d, row),
        "voldry": _q_voldry(d, row),
    }
    if row.get("setup") == "episodic_pivot":
        cpos = row.get("close_pos")
        if (cpos is None or pd.isna(cpos)) and len(d) > 0:
            cpos = d.iloc[-1].get("close_pos")
        rvol = row.get("rvol")
        if (rvol is None or pd.isna(rvol)) and len(d) > 0:
            rvol = d.iloc[-1].get("rvol")
        q_cp = _ramp(float(cpos) if cpos is not None and pd.notna(cpos) else None, *config.Q_EP_CPOS_RAMP)
        q_rv = _ramp(float(rvol) if rvol is not None and pd.notna(rvol) else None, *config.Q_EP_RVOL_RAMP)
        parts = [p for p in (q_cp, q_rv) if p is not None]
        if parts:
            subs["catalyst"] = sum(parts) / len(parts)

    applic = {k: v for k, v in subs.items() if v is not None}
    out = {f"q_{k}": round(v, 2) for k, v in applic.items()}
    if applic:
        wsum = sum(config.Q_WEIGHTS.get(k, 0) for k in applic) or 1.0
        out["quality"] = round(100 * sum(config.Q_WEIGHTS.get(k, 0) * v
                                         for k, v in applic.items()) / wsum, 1)
    else:
        out["quality"] = None
    return out
