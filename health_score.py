"""VALIDATED MARKET-HEALTH SCORE — the banner headline (Amir "wire it", 2026-07-05).

Replaces the hand-weighted conviction composite as the button read. Fit-free equal-weight
mean of the era-robust signals that survived validation (banner_rescore.py, side-by-side
vs the old composite: wins direction hit-rates + fwd5 quintile spread in 25-26, not worse
in 23-24; report at data/validation/banner_rescore_report.txt):

  trend_census   share of SPY/QQQ/IWM/DIA daily trendlab structure up minus down
  persistence    %>20dma quintile -> historical persist21 odds vs base, signed by SPY structure
  inv_risk       outlook dials' mean P(-5% within 21 sessions) vs base, inverted
  med_lean       overnight-environment composite, expanding percentile centered
  credit         HY 5d-change quintile odds vs base, inverted (widening = risk)
  risk appetite  ACCEPTED 2026-07-05 (component lab round 2, Amir): mean of the rotation
                 basket's trendlab signs (RATIO_BASKET, +1 = risk-on, computed fresh) and
                 the NASDAQ-minus-NYSE advancer-share dial (10d mean, expanding percentile).
                 Both halves computed from the app's OWN daily bars — the venue half is EOD
                 breadth, so it is cached to venue_split_series.parquet and RECOMPUTED only
                 when the daily data advances (rides the daily update, NO separate job).
                 Validated: decision-window quintile fwd5 spread +0.66 vs +0.36 ATR, longs
                 better at all horizons, eras not worse.

STANCE BANDS are frozen quantile-matches to the old composite's 23-24 call mix (so call
frequency carried over): strong-long >= +29.3, long >= -5.4, short < -23.8 (re-frozen
2026-07-05 when the risk-appetite dial joined; re-derive via banner_rescore.py --compare
whenever the score is expanded).

CONTEXT LINE (Amir: "worth internalizing is worth expressing"): the score is a CONDITION
read, and its meaning is regime-dependent — in the 23-24 dip-buying tape the WEAKEST health
readings marked the best buys (both scores ranked backwards there). So a weak score is
disambiguated by STRUCTURE: weak health inside intact structure leans washout/opportunity;
weak health + broken structure is the genuinely dangerous combination.

Designed to be EXPANDED: components are additive and independently sourced; add a signal,
re-run banner_rescore validation, re-freeze bands.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config
import datastore
import outlook
import trendlab

IDX = ["SPY", "QQQ", "IWM", "DIA"]
BANDS = {"strong_long": 29.3, "cautious_long": -5.4, "cautious_short": -23.8}
_STANCE = [("strong_long", "Strong Long", "#1d7a4f"), ("cautious_long", "Cautious Long", "#2fbf8f"),
           ("cautious_short", "Neutral / Out", "#6b7a8c"), (None, "Short Bias", "#e05a6d")]
_CACHE: dict = {}


def _struct_census(asof: str | None) -> tuple[float, float, str]:
    """(census [-1,1], spy_sign, spy_word) from daily trendlab states at the asof close."""
    key = ("census", asof)
    if _CACHE.get("ck") == key:
        return _CACHE["cv"]
    tot = spy_sign = 0.0
    spy_word = "?"
    for s in IDX:
        d = datastore.load_bars(s)
        if asof is not None:
            d = d.loc[:asof]
        if d is None or len(d) < 60:
            continue
        st = int(trendlab.structure_series(d)[-1])
        v = 1.0 if st == trendlab.S_UP else (-1.0 if st == trendlab.S_DOWN else 0.0)
        tot += v
        if s == "SPY":
            spy_sign = v
            spy_word = {1: "uptrend", -1: "downtrend"}.get(int(v), "range/transition")
    out = (tot / len(IDX), spy_sign, spy_word)
    _CACHE["ck"], _CACHE["cv"] = key, out
    return out


_VENUE_CACHE = config.DATA_DIR / "features" / "venue_split_series.parquet"


def _venue_series() -> pd.Series:
    """NASDAQ-minus-NYSE advancer share (10d mean, expanding past-only percentile, centered
    to [-1,1]) as a per-session series. This is EOD BREADTH — it only changes when the daily
    bars advance, so it rides the app's daily data update: cached to disk and RECOMPUTED only
    when the cache is behind the latest SPY bar (no separate job). One universe pass on refresh."""
    import universe as uni
    latest = str(datastore.load_bars("SPY").index[-1].date())
    cached = _CACHE.get("venue_series")
    if cached is not None and str(cached.index[-1].date()) >= latest:
        return cached
    try:                                              # disk cache from a prior run/session
        s = pd.read_parquet(_VENUE_CACHE)["x_venue_split"]
        s.index = pd.to_datetime(s.index)
        if str(s.index[-1].date()) >= latest:
            _CACHE["venue_series"] = s
            return s
    except OSError:
        pass
    exch = uni.exchange_map()                          # stale/missing -> recompute one pass
    idx = datastore.load_bars("SPY").index
    nq_up = pd.Series(0.0, index=idx); nq_n = pd.Series(0.0, index=idx)
    ny_up = pd.Series(0.0, index=idx); ny_n = pd.Series(0.0, index=idx)
    for s_ in sorted(uni.active_symbols() or []):
        e = exch.get(s_)
        if e not in ("NASDAQ", "NYSE") or uni.is_future(s_):
            continue
        d = datastore.load_bars(s_)
        if d is None or len(d) < 2:
            continue
        up = (d["close"].pct_change() > 0).astype(float).reindex(idx)
        ok = d["close"].pct_change().notna().astype(float).reindex(idx).fillna(0)
        if e == "NASDAQ":
            nq_up = nq_up.add(up.fillna(0)); nq_n = nq_n.add(ok)
        else:
            ny_up = ny_up.add(up.fillna(0)); ny_n = ny_n.add(ok)
    split = (nq_up / nq_n.replace(0, np.nan) - ny_up / ny_n.replace(0, np.nan)).rolling(10, min_periods=5).mean()
    pct = split.expanding(120).apply(lambda w: (w <= w.iloc[-1]).mean()) * 100
    out = ((pct / 50 - 1).clip(-1, 1)).dropna()
    _CACHE["venue_series"] = out
    try:
        pd.DataFrame({"x_venue_split": out}).to_parquet(_VENUE_CACHE)
    except OSError:
        pass
    return out


def _risk_appetite(asof: str | None) -> float | None:
    """Rotation half (trendlab structure of the RATIO_BASKET charts) + venue-split half,
    both computed from the app's own daily bars. Both halves required (that is what was
    validated) — else the dial sits out."""
    key = ("ra", asof)
    if _CACHE.get("rak") == key:
        return _CACHE["rav"]
    signs = []
    for a, b, sgn, _label in config.RATIO_BASKET:
        da, db = datastore.load_bars(a), datastore.load_bars(b)
        if da is None or db is None:
            continue
        if asof is not None:
            da, db = da.loc[:asof], db.loc[:asof]
        ix = da.index.intersection(db.index)
        if len(ix) < 120:
            continue
        da, db = da.loc[ix], db.loc[ix]
        r = pd.DataFrame({"open": da["open"] / db["open"], "high": da["high"] / db["low"],
                          "low": da["low"] / db["high"], "close": da["close"] / db["close"]}).dropna()
        st = int(trendlab.structure_series(r)[-1])
        signs.append(sgn * (1.0 if st == trendlab.S_UP else (-1.0 if st == trendlab.S_DOWN else 0.0)))
    out = None
    try:
        vs = _venue_series()
        vs = vs.loc[:asof] if asof is not None else vs
        if len(signs) >= 5 and len(vs):
            out = float((np.mean(signs) + float(vs.iloc[-1])) / 2)
    except Exception:
        pass
    _CACHE["rak"], _CACHE["rav"] = key, out
    return out


def _dial_p(art: dict, col: str, v) -> float | None:
    d = art["dials"].get(col)
    if d is None or v is None or not np.isfinite(v):
        return None
    return d["corr5_21"][int(np.searchsorted(d["edges"], v, side="right"))]


def read(asof: str | None = None) -> dict:
    art = outlook._artifact()
    if art is None:
        return {"error": "no awareness_outlook.json"}
    cols = list(art["dials"]) + ["a_u_pct20dma", "a_med_lean"]
    vals, vdate = outlook._values(cols, asof)
    base5 = art["base"]["corr5_21"]

    census, spy_sign, spy_word = _struct_census(asof or vdate)
    comp = {"trend census": census}
    pdial = art["persist_dial"]["a_u_pct20dma"]
    v = vals.get("a_u_pct20dma")
    if v is not None and np.isfinite(v):
        pv = pdial["persist21"][int(np.searchsorted(pdial["edges"], v, side="right"))]
        comp["persistence"] = float(np.clip((pv / art["base"]["persist21"] - 1) * 3, -1, 1) * spy_sign)
    ps = [p for c in art["dials"] if (p := _dial_p(art, c, vals.get(c))) is not None]
    if ps:
        comp["dip risk (inv)"] = float(np.clip(-(np.mean(ps) / base5 - 1) * 2, -1, 1))
    ml = vals.get("a_med_lean")
    if ml is not None and np.isfinite(ml):
        F = outlook._matrix()
        pct = float((F["a_med_lean"].dropna() <= ml).mean() * 100)
        comp["overnight lean"] = float(np.clip(pct / 50 - 1, -1, 1))
    hp = _dial_p(art, "a_hy_chg5", vals.get("a_hy_chg5"))
    if hp is not None:
        comp["credit"] = float(np.clip(-(hp / base5 - 1) * 2, -1, 1))
    ra = _risk_appetite(asof or vdate)
    if ra is not None:
        comp["risk appetite"] = round(ra, 4)
    score = round(100 * float(np.mean(list(comp.values()))), 1)

    stance = color = None
    for band, nm, col_ in _STANCE:
        if band is None or score >= BANDS[band]:
            stance, color = nm, col_
            break

    # regime-context line (the internalization, made conditional on structure)
    if score < BANDS["cautious_long"]:
        if spy_sign >= 0 and census > -0.5:
            ctx = (f"weak health INSIDE intact structure (SPY {spy_word}) — in dip-buying tapes "
                   f"(like 2023-24) readings this weak marked the BEST buys; in trending-down "
                   f"tapes it is danger. Structure says: not broken yet — treat as washout risk, "
                   f"watch for reclaim, don't chase shorts blindly")
        else:
            ctx = (f"weak health AND broken structure (SPY {spy_word}, census "
                   f"{census:+.2f}) — the genuinely dangerous combination; capital "
                   f"preservation first")
    elif score >= BANDS["strong_long"]:
        ctx = ("health strong across the validated dials — condition supports longs "
               "(65% of 5-day windows up vs 62% base in the 25-26 validation)")
    else:
        ctx = ("health mixed — no validated edge from condition alone; let structure and "
               "the day-type read drive")
    return {"score": score, "stance": stance, "color": color, "as_of": vdate,
            "components": {k: round(v, 2) for k, v in comp.items()},
            "context": ctx,
            "why": " · ".join(f"{k} {v:+.2f}" for k, v in comp.items()),
            "note": "validated fit-free health score (banner_rescore.py) — expandable; "
                    "bands frozen to the old composite's call mix"}


if __name__ == "__main__":
    import sys
    print(json.dumps(read(sys.argv[1] if len(sys.argv) > 1 else None), indent=1))
