"""Intraday MARKET INTERNALS computed from the 15-minute bars — a universe-wide, self-computed
TICK / ADD / TRIN / VWAP-breadth (the internals a paid feed sells, derived from our own tape).

Each symbol's 15m frame is classified vs the prior session's close and within today's session; the
aggregate feeds market.py's LIVE score and the dashboard's internals strip. Reuses datastore's 15m store.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _session_stats(df15: pd.DataFrame) -> dict | None:
    """One 15m frame -> today's stats: direction vs prior close, last 15m bar pulse, VWAP side, whether
    the last bar made the session hi/lo, today's volume, and the live last price. None if too short."""
    if df15 is None or len(df15) < 2:
        return None
    dates = df15.index.normalize()
    last_date = dates[-1]
    today = df15[dates.values == last_date]
    if len(today) < 1:
        return None
    last = float(today["close"].iloc[-1])
    prior = df15[dates.values < last_date]
    prev_close = float(prior["close"].iloc[-1]) if len(prior) else float(today["open"].iloc[0])
    d = (last / prev_close - 1.0) if prev_close > 0 else 0.0
    lb = today.iloc[-1]                                            # latest bar
    cc = float(lb["close"])
    # net-TICK pulse = latest price vs the PRIOR bar's close (a real up/down tick). Robust to a freshly
    # -opened forming bar (open==close) which on the 5m base would otherwise zero every symbol -> TICK 0.
    ref = float(today["close"].iloc[-2]) if len(today) >= 2 else float(lb["open"])
    pulse = 1 if cc > ref else (-1 if cc < ref else 0)
    tp = (today["high"].astype(float) + today["low"].astype(float) + today["close"].astype(float)) / 3
    vol = today["volume"].astype(float)
    vsum = float(vol.sum())
    vwap = float((tp * vol).sum() / vsum) if vsum > 0 else last
    hi, lo = float(today["high"].max()), float(today["low"].min())
    sign = np.sign(today["close"].to_numpy(float) - today["open"].to_numpy(float)).astype(int)  # per-bar up/down
    return {"d": d, "pulse": pulse, "above_vwap": last > vwap, "vol": vsum, "last": last,
            "new_hi": float(lb["high"]) >= hi - 1e-9, "new_lo": float(lb["low"]) <= lo + 1e-9,
            "bar_t": today.index, "bar_p": sign}


def intraday_breadth(frames15: dict, frames_daily: dict | None = None) -> dict | None:
    """Aggregate market internals across the 15m frames. `frames_daily` (optional) adds live %>50/200-SMA
    (the live price vs the daily MA). Returns {internals, score in [-1,1]} or None if too few symbols."""
    stats = {}
    for sym, df15 in frames15.items():
        st = _session_stats(df15)
        if st is not None:
            stats[sym] = st
    return aggregate_internals(stats, frames_daily)


def aggregate_internals(stats: dict, frames_daily: dict | None = None) -> dict | None:
    """intraday_breadth's aggregation on PRE-COMPUTED per-symbol _session_stats — so a caller that
    already looped the store (awareness group confirmation) doesn't pay the pass twice."""
    from collections import defaultdict
    adv = dec = 0
    upvol = downvol = 0.0
    tick = new_hi = new_lo = 0
    vwap_above = 0
    a50 = a200 = n_ma = 0
    n = 0
    adline: dict = defaultdict(int)                        # bar timestamp -> net (up−down) across symbols
    for sym, st in stats.items():
        n += 1
        if st["d"] > 0:
            adv += 1; upvol += st["vol"]
        elif st["d"] < 0:
            dec += 1; downvol += st["vol"]
        tick += st["pulse"]
        vwap_above += 1 if st["above_vwap"] else 0
        new_hi += 1 if st["new_hi"] else 0
        new_lo += 1 if st["new_lo"] else 0
        for t, p in zip(st["bar_t"], st["bar_p"]):        # accumulate the cumulative A-D (TICK) line
            adline[t] += int(p)
        if frames_daily is not None:
            dd = frames_daily.get(sym)
            if dd is not None and len(dd):
                r = dd.iloc[-1]
                s50, s200 = r.get("sma50"), r.get("sma200")
                if pd.notna(s50) or pd.notna(s200):
                    n_ma += 1
                    a50 += 1 if (pd.notna(s50) and st["last"] > s50) else 0
                    a200 += 1 if (pd.notna(s200) and st["last"] > s200) else 0
    if n < 20:
        return None
    ad_ratio = (adv / dec) if dec else float(adv or 1)
    trin = ((adv / dec) / (upvol / downvol)) if (dec and upvol and downvol) else None
    pct_up = adv / n * 100
    pct_vwap = vwap_above / n * 100
    # cumulative A-D (TICK) line over the session: net upticks−downticks per bar, cumulated. Its TREND (last
    # ~hour) tells if breadth is BUILDING or FADING even when the snapshot tick looks fine (the −252-cum-TICK
    # divergence). cum_slope feeds the score so intraday deterioration is penalized.
    cum_ad = 0
    cum_trend = "flat"
    cum_slope = 0.0
    if adline:
        times = sorted(adline)
        cum = np.cumsum(np.array([adline[t] for t in times], float))
        cum_ad = int(cum[-1])
        bph = 4                                           # bars/hour — inferred from the A-D grid so the
        if len(times) >= 2:                               # "last hour" window is correct at any base res
            g = np.diff(np.array(times, dtype="datetime64[ns]")) / np.timedelta64(1, "m")
            g = g[g > 0]
            if len(g):
                bph = max(1, int(round(60 / float(g.min()))))   # 15m grid -> 4, 5m grid -> 12
        k = min(len(cum) - 1, bph)                         # ~last hour, resolution-correct
        cum_slope = float(cum[-1] - cum[-1 - k]) if k > 0 else 0.0
        cum_trend = "rising" if cum_slope > 0.05 * n else ("falling" if cum_slope < -0.05 * n else "flat")
    internals = {
        "n": n, "adv": adv, "dec": dec, "ad": adv - dec, "ad_ratio": round(ad_ratio, 2),
        "pct_up": round(pct_up, 1), "trin": (round(trin, 2) if trin is not None else None),
        "tick": tick, "pct_vwap": round(pct_vwap, 1), "new_hi": new_hi, "new_lo": new_lo,
        "cum_ad": cum_ad, "cum_trend": cum_trend,
        "pct_50": (round(a50 / n_ma * 100, 1) if n_ma else None),
        "pct_200": (round(a200 / n_ma * 100, 1) if n_ma else None),
    }
    # score in [-1,1]: breadth %up + net tick + TRIN (bullish<1) + VWAP breadth + net new highs + cum-line trend
    comps = [(pct_up - 50) / 50,
             float(np.tanh(tick / max(1.0, 0.30 * n))),
             (-float(np.tanh(np.log(trin))) if trin and trin > 0 else 0.0),
             (pct_vwap - 50) / 50,
             float(np.tanh((new_hi - new_lo) / max(1.0, 0.10 * n))),
             float(np.tanh(cum_slope / max(1.0, 0.5 * n)))]
    score = float(np.clip(sum(comps) / len(comps), -1, 1))
    internals["why"] = (f"{pct_up:.0f}% up, adv:dec {ad_ratio:.1f}:1, "
                        f"TRIN {internals['trin'] if internals['trin'] is not None else 'n/a'}, "
                        f"TICK {tick:+d}, {pct_vwap:.0f}% >VWAP, {new_hi} new-hi vs {new_lo} lo, "
                        f"cum-AD {cum_ad:+d} ({cum_trend})")
    return {"internals": internals, "score": score}


def latest_session(frames15: dict, probe: str = "SPY"):
    """The date (ISO) of the newest 15m bar for `probe` (or any frame) — used to gauge freshness."""
    df = frames15.get(probe)
    if df is None:
        df = next(iter(frames15.values()), None)
    if df is None or not len(df):
        return None
    return df.index[-1].date().isoformat()
