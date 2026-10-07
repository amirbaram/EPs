"""Situational awareness v2 — the market map.

Builds per-instrument VIEWS (settled daily context + a point-in-time intraday overlay) for the
indices, sector ETFs, theme baskets and ratio pairs, then the rotation map and the ranked calls
(calls.py). A view answers the questions a discretionary trader asks: what is the trend, what
SEQUENCE came before, where are the levels ahead, what did the overnight gap do, and is today
confirming or fading it.

Point-in-time contract: `anchor` is the prior SETTLED session for any intraday read (serve
_resolve provides it); `frames15` contains only bars completed by the replay clock. Daily views
are cached per anchor by the caller (serve keeps an LRU) so slider scrubs only recompute the
intraday overlays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import calls as calls_mod
import config
import datastore
import intraday
import labels
import market
import mtf
import performance
import ratios
import sequence
import sr
from indicators import add_indicators

INDEX_SPECS = [
    {"key": "SPY", "kind": "index", "sym": "SPY", "label": "S&P 500 (ES)"},
    {"key": "QQQ", "kind": "index", "sym": "QQQ", "label": "Nasdaq 100 (NQ)"},
    {"key": "IWM", "kind": "index", "sym": "IWM", "label": "Russell 2000 (RTY)"},
    {"key": "DIA", "kind": "index", "sym": "DIA", "label": "Dow (YM)"},
]

_SECTOR_LABELS = {"XLK": "Tech", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care",
                  "XLY": "Discretionary", "XLP": "Staples", "XLI": "Industrials",
                  "XLB": "Materials", "XLU": "Utilities", "XLRE": "Real Estate",
                  "XLC": "Comm Services", "SMH": "Semis", "KRE": "Regional Banks"}


def sector_specs(frames: dict) -> list[dict]:
    return [{"key": etf, "kind": "sector", "sym": etf,
             "label": _SECTOR_LABELS.get(etf, etf)}
            for etf in config.AWARE_SECTOR_ETFS if etf in frames]


def ratio_specs(frames: dict, frames15: dict | None = None) -> list[dict]:
    out = []
    for a, b, sign, label in config.AWARE_RATIOS:
        if a in frames and b in frames:
            out.append({"key": f"{a}/{b}", "kind": "ratio", "a": a, "b": b,
                        "sign": sign, "label": label})
    return out


def group_members(frames: dict) -> dict[tuple, set]:
    """{(kind, name): members} for every sector-ETF-independent group: the 41 themes and the
    fine-grained industries from sectors.csv (only members loaded in `frames` count)."""
    groups: dict[tuple, set] = {}
    for theme in labels.all_themes():
        m = {s for s in labels.tickers_in_theme(theme) if s in frames}
        if len(m) >= config.GROUP_MIN_MEMBERS:
            groups[("theme", theme)] = m
    ind: dict[str, set] = {}
    for sym in frames:
        sec = labels.sector(sym)
        if sec:
            ind.setdefault(sec, set()).add(sym)
    for name, m in ind.items():
        if len(m) >= config.GROUP_MIN_MEMBERS:
            groups[("industry", name)] = m
    return groups


def _r(x, nd=2):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), nd)


def _py(o):
    """JSON-safe payload: numpy scalars -> python, non-finite -> None (frames are float32, and the
    validator writes payloads straight to disk without Flask's encoder)."""
    if isinstance(o, dict):
        return {k: _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def _slim_levels(lv: dict, n: int = 3) -> dict:
    return {"support": (lv.get("support") or [])[:n],
            "resistance": (lv.get("resistance") or [])[:n]}


# ---------------------------------------------------------------- daily (settled) view

def daily_view(spec: dict, frames: dict, anchor: str) -> dict | None:
    """Settled context on the anchor date: posture + sequence + S/R ladder. Ratio specs build the
    synthetic A÷B frame; basket specs (ETF-less themes) a rebased member index; plain specs slice
    the (already enriched) RAM frame."""
    frac = None
    advisory = False
    if spec["kind"] == "ratio":
        d = ratios.ratio_frame(spec["a"], spec["b"], frames, as_of=anchor)
        if d is not None:
            d = d.iloc[-400:]
    elif spec.get("sym"):
        f = frames.get(spec["sym"])
        d = f.loc[:anchor].iloc[-500:] if f is not None else None
    else:                                              # synthetic basket: o=h=l=c -> ATR ≈ |Δclose|,
        b = ratios.basket_series(spec["members"], frames, as_of=anchor)   # coarser zigzag + advisory S/R
        d = add_indicators(b.iloc[-500:]) if b is not None else None
        frac = config.AWARE_BASKET_ATR_FRACTION
        advisory = True
    if d is None or len(d) < 40:
        return None
    lv = sr.sr_levels(d)
    seq = sequence.sequence_read(d, lv=lv, tf="1D", atr_fraction=frac)
    st = mtf.tf_state(d)
    if seq is None or st is None:
        return None
    out = {"state": st, "seq": seq, "sr": _slim_levels(lv), "close": _r(d["close"].iloc[-1]),
           "atr": _r(float(d["atr14"].iloc[-1])), "asof": str(d.index[-1].date())}
    if advisory:
        out["advisory"] = True
    return out


# ---------------------------------------------------------------- intraday overlay

def _today_slice(f15: pd.DataFrame):
    """(today's bars, prior session's bars) of a truncated 15m frame — the last session is 'today'."""
    dates = f15.index.date
    today = dates[-1]
    tmask = dates == today
    prev = f15[~tmask]
    if not len(prev):
        return f15[tmask], None
    pday = prev.index.date[-1]
    return f15[tmask], prev[prev.index.date == pday]


def intraday_overlay(spec: dict, frames15: dict, daily: dict, date: str) -> dict | None:
    """The point-in-time intraday read on top of a settled daily view: 1h posture+sequence, the
    overnight gap and whether it is holding, today's tape, and the session levels. None when the
    instrument has no 15m data reaching `date` (UI degrades to daily-only)."""
    if not frames15 or daily is None:
        return None
    if spec["kind"] == "ratio":
        h1 = ratios.ratio_frame(spec["a"], spec["b"], {}, frames15, live=True, tf="1h")
        f15 = None
    else:
        f15 = frames15.get(spec["sym"])
        if f15 is None or not len(f15) or str(f15.index[-1].date()) != date:
            return None
        h1 = mtf.tf_frame(None, f15, "1h")
    out: dict = {"h1": None, "gap": None, "day": None, "levels": None}
    if h1 is not None and len(h1) >= 40:
        st1 = mtf.tf_state(h1)
        seq1 = sequence.sequence_read(h1, tf="1h", atr_ref=daily.get("atr"))
        if st1 is not None and seq1 is not None:
            out["h1"] = {"state": st1, "seq": seq1}
    if f15 is None:                                   # ratio overlays: posture only
        return out if out["h1"] is not None else None

    today, prev = _today_slice(f15)
    if not len(today):
        return out if out["h1"] is not None else None
    o = float(today["open"].iloc[0])
    last = float(today["close"].iloc[-1])
    hi, lo = float(today["high"].max()), float(today["low"].min())
    bars = int(len(today))
    atr = daily.get("atr") or 0.0
    y_close = float(prev["close"].iloc[-1]) if prev is not None and len(prev) else None
    y_high = float(prev["high"].max()) if prev is not None and len(prev) else None
    y_low = float(prev["low"].min()) if prev is not None and len(prev) else None

    gap = None
    if y_close and atr:
        gpct = (o / y_close - 1) * 100
        gatr = (o - y_close) / atr
        vs_y = ("above" if o > (y_high or o) else "below" if o < (y_low or o) else "inside")
        settle = max(2, round(30 / datastore._infer_base_min(f15.index)))   # ~30-min settle: 2 bars @15m, 6 @5m
        if bars < settle:
            status = "na"                              # too early — give the gap ~30 min to prove held/faded
        elif gatr >= 0:
            floor = max(o - config.AWARE_GAP_HOLD_ATR * atr, y_close)
            status = "holding" if last >= floor else ("filled" if last < y_close else "faded")
        else:
            cap = min(o + config.AWARE_GAP_HOLD_ATR * atr, y_close)
            status = "holding" if last <= cap else ("filled" if last > y_close else "faded")
        gap = {"pct": _r(gpct), "atr": _r(gatr), "vs_yrange": vs_y, "status": status, "bars": bars}

    tpv = float((today["close"] * today["volume"]).sum())
    vol = float(today["volume"].sum())
    vwap = tpv / vol if vol > 0 else None
    out["day"] = {"last": _r(last),
                  "open_pct": _r((last / o - 1) * 100),
                  "prev_close_pct": _r((last / y_close - 1) * 100) if y_close else None,
                  "range_pos": _r((last - lo) / (hi - lo)) if hi > lo else None,
                  "vwap_side": ("above" if vwap is not None and last >= vwap else
                                "below" if vwap is not None else None),
                  "bars": bars}
    out["gap"] = gap
    out["levels"] = {"open": _r(o), "vwap": _r(vwap), "y_high": _r(y_high),
                     "y_low": _r(y_low), "y_close": _r(y_close)}
    return out


# ---------------------------------------------------------------- descriptive bias

_SEQ_SCORE = {"impulse_up": 1.0, "breakout_fresh": 1.0, "trend_up": 0.8,
              "pullback_in_uptrend": 0.5, "range": 0.0, "unclear": 0.0,
              "distribution": -0.4, "bounce_in_downtrend": -0.5,
              "trend_down": -0.8, "breakdown_fresh": -1.0, "impulse_down": -1.0}


def _seq_score(seq: dict | None, live_price: float | None = None) -> float:
    if not seq:
        return 0.0
    s = _SEQ_SCORE.get(seq["regime"], 0.0)
    if seq["regime"] == "range":
        m = seq.get("meta") or {}
        pos = m.get("range_pos")
        # a live price re-reads the band position in real time (the anchor's close is stale by T)
        if live_price is not None and m.get("range_hi") and m.get("range_lo") is not None \
                and m["range_hi"] > m["range_lo"]:
            pos = min(1.25, max(-0.25, (live_price - m["range_lo"]) / (m["range_hi"] - m["range_lo"])))
        if pos is not None:
            s = (pos - 0.5) * 0.4                      # upper band leans long, lower leans short
    if "extended" in seq.get("flags", ()) and s > 0:
        s *= 0.5
    if "at_resistance" in seq.get("flags", ()):
        s -= 0.15
    elif "at_support" in seq.get("flags", ()):
        s += 0.15
    return s


def bias(view: dict) -> dict:
    """Descriptive per-instrument lean for the card badge (NOT a trade call — calls.py makes
    those): daily sequence + posture, 1h sequence, and the gap behavior."""
    daily, intra = view.get("daily"), view.get("intra")
    live_price = ((intra or {}).get("day") or {}).get("last")
    g = (intra or {}).get("gap")
    big_gap = bool(g and g.get("atr") is not None and g["status"] != "na"
                   and abs(g["atr"]) >= config.AWARE_GAP_MIN_ATR)
    terms: list[tuple[float, float, str]] = []         # (weight, score, tag)
    if daily:
        terms.append((0.45, _seq_score(daily["seq"], live_price), daily["seq"]["regime"]))
        terms.append((0.25, market._score_state(daily["state"]) / 2.0, "posture"))
    if intra and intra.get("h1"):
        hs = _seq_score(intra["h1"]["seq"])
        # a big gap that is HOLDING outranks a lagging hourly label that fights it
        if big_gap and g["status"] == "holding" and hs * g["atr"] < 0:
            hs *= 0.5
        terms.append((0.20, hs, "1h " + intra["h1"]["seq"]["regime"]))
    if g and g.get("atr") is not None and g["status"] != "na":
        if g["status"] == "holding":
            gs = float(np.clip(g["atr"], -1.5, 1.5)) / 1.5
        else:
            gs = -0.3 * np.sign(g["atr"])
        terms.append((0.22 if big_gap else 0.10, gs, f"gap {g['status']}"))
    if not terms:
        return {"side": "neutral", "score": 0.0, "why": "no data"}
    score = sum(w * s for w, s, _ in terms) / sum(w for w, _, _ in terms)
    side = ("long" if score >= 0.35 else "short" if score <= -0.35 else
            "neutral" if abs(score) < 0.15 else "avoid")
    top = max(terms, key=lambda x: abs(x[0] * x[1]))
    return {"side": side, "score": _r(score), "why": top[2]}


# ---------------------------------------------------------------- rotation (strength/weakness map)

def _returns_panel(frames: dict, anchor: str, syms: set) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(closes, daily % returns) for the labeled universe over the rotation window ending at the
    anchor — one aligned matrix so every group's relative-return history is a column-mean away."""
    w = config.AWARE_ROT_WINDOW
    # master calendar = SPY's sessions: per-symbol tails drift when a name misses a day, and the
    # DataFrame union then gives EVERY column NaNs (which silently killed all theme rs_1m/rs_3m)
    spy = frames.get("SPY")
    if spy is None:
        return pd.DataFrame(), pd.DataFrame()
    idx = spy["close"].loc[:anchor].iloc[-(w + 2):].index
    cols = {}
    for s in syms:
        f = frames.get(s)
        if f is None:
            continue
        c = f["close"].loc[:anchor]
        if len(c) >= w + 3:
            cols[s] = c.reindex(idx)
    px = pd.DataFrame(cols, index=idx).astype("float64")
    rets = px.pct_change().iloc[1:] * 100                  # w+1 rows, last = the anchor day itself
    return px, rets


def _rot_row(kind, name, etf, members, px, rets, spy_rets, today_rel, pct_up, pct_vwap,
             ref_today) -> dict | None:
    """One group's rotation entry. `today_rel` = today's group-vs-SPY return (%); the trailing
    distribution comes from the returns panel (close-mode scores the anchor day itself, dropped
    from its own distribution)."""
    if members is not None:
        cols = [s for s in members if s in rets.columns]
        if len(cols) < config.GROUP_MIN_MEMBERS:
            return None
        rrel = rets[cols].mean(axis=1) - spy_rets
        sub = px[cols]
        base = sub.div(sub.bfill().iloc[0]).mean(axis=1)     # rebase on first VALID value: one
                                                             # halted member must not NaN the basket
        leaders = (list((sub.iloc[-1] / sub.iloc[-6] - 1).nlargest(2).index)
                   if len(sub) >= 6 else [])                 # the group's strongest members (5d)
    elif etf in rets.columns:
        rrel = rets[etf] - spy_rets
        base = px[etf]
        leaders = []
    else:
        return None
    spy_px = px["SPY"]
    rs_1m = float((base.iloc[-1] / base.iloc[-22] - 1) * 100
                  - (spy_px.iloc[-1] / spy_px.iloc[-22] - 1) * 100) if len(base) >= 22 else None
    rs_3m = float((base.iloc[-1] / base.iloc[0] - 1) * 100
                  - (spy_px.iloc[-1] / spy_px.iloc[0] - 1) * 100)
    dist = rrel.iloc[:-1] if today_rel is None else rrel     # close mode scores the anchor day
    rot_today = float(rrel.iloc[-1]) if today_rel is None else float(today_rel)
    sd = max(float(dist.std()), config.AWARE_ROT_Z_FLOOR)
    rot_z = (rot_today - float(dist.mean())) / sd
    lead = ("leading" if (rs_1m or 0) >= config.AWARE_ROT_LEAD else
            "lagging" if (rs_1m or 0) <= -config.AWARE_ROT_LEAD else "inline")
    ymove = ("extending" if rot_z >= config.AWARE_ROT_EXT else
             "fading" if rot_z <= -config.AWARE_ROT_EXT else "steady")
    quadrant = {("leading", "extending"): "leading_extending",
                ("leading", "fading"): "leading_fading",
                ("lagging", "extending"): "lagging_improving",
                ("lagging", "fading"): "lagging_breaking",
                }.get((lead, ymove), f"{lead}_{ymove}")
    acting = None
    if pct_up is not None:
        well = pct_up >= 60 and (pct_vwap is None or pct_vwap >= 55)
        poor = pct_up <= 40 or (pct_vwap is not None and pct_vwap <= 40)
        acting = "well" if well else ("poorly" if poor else "mixed")
    narrow = bool(ref_today is not None and ref_today > 0.3 and pct_up is not None and pct_up < 45)
    return {"kind": kind, "name": name, "etf": etf, "n": (len(members) if members else None),
            "leaders": leaders,
            "rs_1m": _r(rs_1m), "rs_3m": _r(rs_3m), "rot_today": _r(rot_today), "rot_z": _r(rot_z),
            "quadrant": quadrant, "acting": acting, "narrow": narrow,
            "pct_up": _r(pct_up, 1), "pct_vwap": _r(pct_vwap, 1)}


def _member_stats15(frames15: dict, syms: set) -> dict:
    """One pass of per-symbol session stats over the labeled universe (~0.5ms/sym; the vectorized
    panel can replace this later without changing the contract)."""
    out = {}
    for s in syms:
        f = frames15.get(s)
        if f is not None:
            st = intraday._session_stats(f)
            if st is not None:
                out[s] = st
    return out


def rotation_map(frames: dict, frames15: dict | None, anchor: str, daily_cache: dict,
                 groups: dict, sector_v: dict, theme_v: dict) -> dict:
    """The strength/weakness map: every group\'s trailing RS vs today\'s rotation z-score, member
    confirmation (acting well/poorly/narrow), and the aggregate internals from the same pass."""
    px, rets = daily_cache["_panel"]
    spy_rets = rets["SPY"]
    mstats = _member_stats15(frames15, set().union(*groups.values())) if frames15 else {}
    spy_today = None
    if frames15 is not None and frames15.get("SPY") is not None:
        st = intraday._session_stats(frames15["SPY"])
        spy_today = st["d"] if st else None

    rows = []
    for etf in config.AWARE_SECTOR_ETFS:                   # sector row = the ETF\'s own series
        today_rel = ref_today = None
        if frames15 is not None and spy_today is not None:
            f = frames15.get(etf)
            st = intraday._session_stats(f) if f is not None else None
            if st:
                ref_today = st["d"]
                today_rel = st["d"] - spy_today
        row = _rot_row("sector", _SECTOR_LABELS.get(etf, etf), etf, None, px, rets, spy_rets,
                       today_rel, None, None, ref_today)
        if row:
            rows.append(row)
    for (kind, name), members in groups.items():
        pct_up = pct_vwap = today_rel = ref_today = None
        if mstats:
            ms = [mstats[s] for s in members if s in mstats]
            if len(ms) >= config.GROUP_MIN_MEMBERS:
                pct_up = 100 * sum(1 for s in ms if s["d"] > 0) / len(ms)
                pct_vwap = 100 * sum(1 for s in ms if s["above_vwap"]) / len(ms)
                ref_today = float(np.median([s["d"] for s in ms]))
                today_rel = ref_today - spy_today if spy_today is not None else None
        else:                                              # close mode: anchor-day member breadth
            cols = [s for s in members if s in rets.columns]
            if len(cols) >= config.GROUP_MIN_MEMBERS:
                pct_up = float((rets[cols].iloc[-1] > 0).mean() * 100)
        etf = labels.theme_etf(name) if kind == "theme" else None   # chartable proxy for the dot
        row = _rot_row(kind, name, etf, members, px, rets, spy_rets, today_rel,
                       pct_up, pct_vwap, ref_today)
        if row:
            rows.append(row)

    # attach the structural read where an instrument view exists (sectors + top themes)
    for row in rows:
        v = sector_v.get(row["etf"]) if row["kind"] == "sector" else theme_v.get(row["name"])
        if v:
            row["seq_regime"] = v["daily"]["seq"]["regime"]
            fl = v["daily"]["seq"]["flags"]
            row["at_level"] = ("at_R" if "at_resistance" in fl else
                               "at_S" if "at_support" in fl else "clear")

    # payload cap: all sectors + themes; industries only the day\'s top/bottom movers
    ind = sorted((r for r in rows if r["kind"] == "industry"), key=lambda r: r["rot_z"] or 0)
    keep = {id(r) for r in ind[:config.AWARE_IND_ROWS] + ind[-config.AWARE_IND_ROWS:]}
    rows = [r for r in rows if r["kind"] != "industry" or id(r) in keep]
    rows.sort(key=lambda r: -(r["rot_z"] or 0))

    internals = None
    if mstats:
        try:
            ib = intraday.aggregate_internals(mstats, frames)
            internals = ({**ib["internals"], "score": round(float(ib["score"]), 2)} if ib else None)
        except Exception:
            internals = None
    return {"groups": rows, "internals": internals}


# ---------------------------------------------------------------- payload assembly

def _instrument_view(spec, frames, frames15, anchor, date, daily_cache: dict) -> dict | None:
    key = spec["key"]
    if key not in daily_cache:
        daily_cache[key] = daily_view(spec, frames, anchor)
    daily = daily_cache[key]
    if daily is None:
        return None
    intra = intraday_overlay(spec, frames15, daily, date) if frames15 else None
    view = {"key": key, "kind": spec["kind"], "label": spec.get("label", key),
            "daily": daily, "intra": intra}
    if spec["kind"] == "ratio":
        view["sign"] = spec.get("sign", 1)
    view["bias"] = bias(view)
    return view


def _theme_specs(frames: dict, groups: dict, daily_cache: dict) -> list[dict]:
    """Instrument specs for the top themes by |trailing RS| (ETF when cached, else basket)."""
    px, rets = daily_cache["_panel"]
    if len(px) < 22:
        return []
    spy_px = px["SPY"]
    spy_1m = (spy_px.iloc[-1] / spy_px.iloc[-22] - 1) * 100
    scored = []
    for (kind, name), members in groups.items():
        if kind != "theme":
            continue
        cols = [s for s in members if s in px.columns]
        if len(cols) < config.GROUP_MIN_MEMBERS:
            continue
        base = px[cols].div(px[cols].iloc[0]).mean(axis=1)
        rs = float((base.iloc[-1] / base.iloc[-22] - 1) * 100 - spy_1m)
        scored.append((abs(rs), name, members))
    scored.sort(key=lambda x: -x[0])
    specs = []
    for _, name, members in scored[:config.AWARE_TOP_THEMES]:
        etf = performance._etf_for_theme(name, frames)
        specs.append({"key": f"theme:{name}", "kind": "theme", "sym": etf,
                      "members": members, "label": name})
    return specs


def awareness_read(date: str, t: str | None, mode: str | None, frames: dict,
                   frames15: dict | None, anchor: str, daily_cache: dict | None = None) -> dict:
    """The full map payload for one (date, t) moment. `anchor` = settled daily anchor from
    serve._resolve (prior session for intraday reads, the date itself for close reads).
    `daily_cache` = caller-owned per-anchor dict so scrubbing recomputes only overlays."""
    daily_cache = daily_cache if daily_cache is not None else {}
    live = bool(frames15)
    if "_groups" not in daily_cache:
        daily_cache["_groups"] = group_members(frames)
    groups = daily_cache["_groups"]
    if "_panel" not in daily_cache:
        syms = set().union(*groups.values()) | set(config.AWARE_SECTOR_ETFS) | {"SPY"}
        daily_cache["_panel"] = _returns_panel(frames, anchor, syms)

    indices = [v for s in INDEX_SPECS
               if (v := _instrument_view(s, frames, frames15, anchor, date, daily_cache))]
    sectors = [v for s in sector_specs(frames)
               if (v := _instrument_view(s, frames, frames15, anchor, date, daily_cache))]
    themes = [v for s in _theme_specs(frames, groups, daily_cache)
              if (v := _instrument_view(s, frames, frames15, anchor, date, daily_cache))]
    ratio_v = [v for s in ratio_specs(frames)
               if (v := _instrument_view(s, frames, frames15, anchor, date, daily_cache))]

    rotation = rotation_map(frames, frames15, anchor, daily_cache, groups,
                            {v["key"]: v for v in sectors}, {v["label"]: v for v in themes})
    payload_wo_calls = {"as_of": date, "t": t, "anchor": anchor, "live": live, "mode": mode or "",
                        "indices": indices, "sectors": sectors, "themes": themes, "ratios": ratio_v,
                        "rotation": rotation}
    try:
        the_calls = calls_mod.evaluate(payload_wo_calls)
    except Exception:
        the_calls = []
    try:
        chip_src = market.market_score(anchor, frames, live=False)
        chip = {"score": chip_src["score"], "stance": chip_src["stance"], "color": chip_src["color"]}
    except Exception:
        chip = None
    data_asof = None
    if frames15 and frames15.get("SPY") is not None and len(frames15["SPY"]):
        _bm = datastore._infer_base_min(frames15["SPY"].index)                  # +base_min => the bar CLOSE time
        data_asof = str(frames15["SPY"].index[-1] + pd.Timedelta(minutes=_bm))  # (15 @15m, 5 @5m)
    return _py({**payload_wo_calls, "calls": the_calls, "chip": chip, "data_asof": data_asof,
                "validation": calls_mod.stats_summary()})
