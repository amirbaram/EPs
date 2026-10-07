"""Natural-language 'Playbook' v3 — a quality-gated, volatility-normalized conviction ranker.

Candidates are drawn whole-universe from group leaders/laggards + the scanner's own setup hits, then:
  1. a HARD tradeability gate drops low-priced / thin / small-cap names (price, $-vol, market cap);
  2. momentum is scored in units of the stock's OWN ADR (move ÷ ADR), so a mega-cap's meaningful move
     competes fairly with a small-cap's noise (the old %-momentum structurally favoured volatile micro-caps);
  3. setups map to a side by TYPE and STATE (rider `riding` = continuation; `touching`/`reversal` = the other
     side; QM/HTF long-only; backburner = long or fade); riders + trend weigh most;
  4. liquidity/size/quality are REWARDED (not just gated); and a per-sector cap keeps the list diversified.

Two modes fall out of `frames15`: LIVE/replay (intraday) and CLOSE/EOD (settled, no future data). Everything
is as-of the read (point-in-time). Descriptive candidate framing only — not buy/sell instructions.
"""
from __future__ import annotations

import math

import pandas as pd

import config
import labels
import marketcap
import mtf
import performance
import sr

# rider states that lean AGAINST the ride (count toward the reversal side). 'saved' = the ride defended
# an armed exit, so it stays a continuation (NOT here); 'break' = the arm-exit flip (replaces 'reversal').
_REVERSING = {"touching", "reversal", "armed", "break"}


def _f(v):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _members(kind: str, name: str) -> set[str]:
    if kind == "theme":
        return labels.tickers_in_theme(name)
    return {s for s, sec in (labels._SECTOR or {}).items() if sec == name}


def _group_maps(groups: list[dict]) -> tuple[dict, dict]:
    """sym -> best-scoring group per side (score>0 = long pool, <0 = short pool)."""
    labels.load()
    longmap: dict = {}
    shortmap: dict = {}
    for g in groups:
        sc = g["score"]
        if sc == 0:
            continue
        tgt = longmap if sc > 0 else shortmap
        for s in _members(g["kind"], g["name"]):
            cur = tgt.get(s)
            if cur is None or (sc > 0 and sc > cur["score"]) or (sc < 0 and sc < cur["score"]):
                tgt[s] = g
    return longmap, shortmap


def _setups_map(hits) -> dict:
    """sym -> {setups, states{setup:state}, quality, price, dvol_m, adr, mcap_b} from the as-of scan hits."""
    out: dict = {}
    if hits is None or not len(hits):
        return out
    for r in hits.itertuples(index=False):
        s = getattr(r, "symbol", None)
        setup = getattr(r, "setup", None)
        if not s or not setup:
            continue
        d = out.get(s)
        if d is None:
            mc = _f(getattr(r, "market_cap", None))
            d = out[s] = {"setups": set(), "states": {}, "quality": None,
                          "price": _f(getattr(r, "close", None)),
                          "dvol_m": _f(getattr(r, "dollar_vol_m", None)),
                          "adr": _f(getattr(r, "adr_pct", None)),
                          "mcap_b": (mc / 1e9 if mc else None)}
        d["setups"].add(setup)
        d["states"][setup] = getattr(r, "state", None)
        q = _f(getattr(r, "quality", None))
        if q is not None and (d["quality"] is None or q > d["quality"]):
            d["quality"] = q
    return out


def _setup_side_weight(setup: str, state, side: str, w: float):
    """(contribution, tag) of one setup to `side`, honoring the rider/trend state logic."""
    rev = state in _REVERSING
    if setup == "ema_rider_bull":                      # bull rider: riding/saved=long; touching/armed/break=short
        if side == "long":
            return (w * (0.2 if rev else 1.0), setup)
        return (w * 0.8, setup + "→short") if rev else (0.0, None)
    if setup == "ema_rider_bear":                      # bear rider: riding/saved=short; touching/armed/break=long
        if side == "short":
            return (w * (0.2 if rev else 1.0), setup)
        return (w * 0.8, setup + "→long") if rev else (0.0, None)
    if setup == "downtrend":
        return (w, setup) if side == "short" else (0.0, None)
    if setup == "backburner":                          # near ATH: a long, or a fade of an extended high-flier
        return (w, setup) if side == "long" else (w * 0.4, setup + " (fade)")
    return (w, setup) if side == "long" else (0.0, None)   # qm_breakout / high_tight_flag / uptrend / … = long-only


def _setup_signals(su: dict, side: str):
    W = config.PLAYBOOK_SETUP_WEIGHTS
    total = 0.0
    tags = []
    states = su.get("states", {})
    for setup in su.get("setups", ()):
        w = W.get(setup, W.get("_default", 0.4))
        contrib, tag = _setup_side_weight(setup, states.get(setup), side, w)
        if contrib > 0:
            total += contrib
            if tag:
                tags.append(tag)
    return total, tags


def _frame_asof(frames: dict, sym: str, as_of):
    """The candidate's frame sliced to the settled anchor (no look-ahead on a replay), or None."""
    d = frames.get(sym)
    if d is None or not len(d):
        return None
    if as_of is not None:
        d = d.loc[:as_of]
    return d if len(d) else None


def _liq(sym: str, frames: dict, su: dict, as_of):
    """(price, $-vol M, ADR%, mcap $B) — from the scan hit when present, else the frame + marketcap cache."""
    price, dvol, adr, mcap_b = su.get("price"), su.get("dvol_m"), su.get("adr"), su.get("mcap_b")
    if price is None or dvol is None or adr is None:
        d = _frame_asof(frames, sym, as_of)
        if d is not None:
            row = d.iloc[-1]
            price = price if price is not None else _f(row.get("close"))
            adr = adr if adr is not None else _f(row.get("adr_pct"))
            if dvol is None:
                dv = _f(row.get("dollar_vol"))
                dvol = dv / 1e6 if dv is not None else None
    if mcap_b is None:
        mc = marketcap.get(sym)
        mcap_b = mc / 1e9 if mc else None
    return price, dvol, adr, mcap_b


def _mom(pr: dict, live: bool) -> float:
    """Recency-weighted signed momentum %. LIVE: today's move comes from the 15m (`prev_close_pct` = since
    yesterday's close incl. the overnight gap, + `open_pct`), since the daily windows are anchored to the
    prior SETTLED session (point-in-time). EOD: settled daily d1/wtd/1w."""
    parts = ([("prev_close_pct", 1.5), ("open_pct", 1.0), ("wtd_pct", 0.5)] if live
             else [("d1_pct", 1.2), ("wtd_pct", 1.0), ("w1_pct", 0.6)])
    num = den = 0.0
    for k, w in parts:
        v = pr.get(k)
        if v is not None:
            num += v * w
            den += w
    return num / den if den else 0.0


def _conv(sym, side, frames, pr, gmap, smap, live, regime, as_of):
    """Per-candidate conviction after the hard tradeability gate. None = filtered out."""
    d = _frame_asof(frames, sym, as_of)
    if d is None:
        return None
    su = smap.get(sym, {})
    price, dvol, adr, mcap_b = _liq(sym, frames, su, as_of)
    if price is None or dvol is None:
        return None
    if price < config.PLAYBOOK_MIN_PRICE or dvol < config.PLAYBOOK_MIN_DVOL_M:
        return None
    if mcap_b is not None and mcap_b < config.PLAYBOOK_MIN_MCAP_B:      # None passes (cache miss ≠ small)
        return None
    setup_score, tags = _setup_signals(su, side) if su else (0.0, [])
    g = gmap.get(sym)
    if setup_score <= 0 and g is None:                                 # need a reason (setup or group)
        return None
    row = d.iloc[-1]
    ema20 = _f(row.get("ema20"))
    above = ema20 is None or price > ema20
    if setup_score <= 0:                                              # group-only: require aligned posture
        if (side == "long") != above:
            return None
    adr_eff = max(adr or 0.0, 1.0)
    sgn = 1.0 if side == "long" else -1.0
    rs = (g.get("rs") or 0.0) if g else 0.0
    conv = (config.PLAYBOOK_W_SETUP * setup_score
            + config.PLAYBOOK_W_RS * sgn * rs / 10.0
            + config.PLAYBOOK_W_MOM * math.tanh((sgn * _mom(pr, live) / adr_eff) / 1.5)   # ADR-normalized
            + config.PLAYBOOK_W_LIQ * (math.log10(max(dvol, 1.0)) - 1.7)                  # ~0 at $50M, + above
            + (0.2 if above == (side == "long") else -0.2)
            + (0.15 if (g and g.get("rs_trend") == ("accel" if side == "long" else "fade")) else 0.0)
            + sgn * regime)                                                              # light regime tilt
    if live:
        op = pr.get("open_pct")
        if op is not None:
            conv += config.PLAYBOOK_W_INTRA * math.tanh((sgn * op / adr_eff) / 1.5)
    q = su.get("quality")
    if q is not None:
        conv += 0.004 * q
    return {"symbol": sym, "conv": conv, "group": g, "tags": tags, "su": su, "pr": pr,
            "price": price, "dvol": round(dvol), "adr": round(adr or 0.0, 1), "mcap_b": mcap_b,
            "developing": setup_score <= 0}


def _note(c: dict, side: str) -> str:
    sym, g, pr = c["symbol"], c["group"], c["pr"]
    head = ("Long " if side == "long" else "Short ") + sym
    if g:
        head += f" ({g['name']}" + (f", RS {g['rs']:+.0f}% {g['rs_trend']}" if g.get("rs") is not None else "") + ")"
    bits = [head, ", ".join(dict.fromkeys(c["tags"])) if c["tags"] else "developing"]
    perf = []
    for lab, key in (("open", "open_pct"), ("d1", "d1_pct"), ("wtd", "wtd_pct"), ("1w", "w1_pct")):
        v = pr.get(key)
        if v is not None:
            perf.append(f"{lab} {v:+.1f}%")
    if perf:
        bits.append(" ".join(perf[:3]))
    st = c.get("st")
    if st is not None:
        tilt = st.get("tilt") or 0
        bits.append(("above" if st.get("gt_ema20") else "below") + " 20-EMA"
                    + (", rising" if tilt >= 2 else (", falling" if tilt <= -2 else "")))
    lv = c.get("sr")
    if lv:
        res, sup = lv.get("resistance"), lv.get("support")
        if side == "long" and res:
            bits.append(f"{res[0]['dist_atr']:.1f} ATR under ×{res[0]['strength']} R")
        elif side == "short" and sup:
            bits.append(f"{sup[0]['dist_atr']:.1f} ATR over ×{sup[0]['strength']} S")
    bits.append(f"${c['price']:.0f} · ${c['dvol']}M/d · ADR {c['adr']}%")
    return " · ".join(bits) + "."


def _side(side, frames, frames15, as_of, gmap, smap, live, regime, top_n):
    pool = set(gmap.keys())
    for sym, su in smap.items():
        if _setup_signals(su, side)[0] > 0:
            pool.add(sym)
    cands = []
    for sym in pool:
        pr = performance.ticker_perf(sym, frames, frames15 or {}, as_of)
        if pr is None:
            continue
        c = _conv(sym, side, frames, pr, gmap, smap, live, regime, as_of)
        if c is not None:
            cands.append(c)
    cands.sort(key=lambda x: -x["conv"])
    for c in cands[:max(6 * top_n, 30)]:                # enrich the finalists (bounded S/R + posture calls)
        d = _frame_asof(frames, c["symbol"], as_of)     # settled anchor (no look-ahead on a replay)
        if d is None:
            continue
        c["st"] = mtf.tf_state(d)
        c["sr"] = lv = sr.sr_levels(d)
        near = lv.get("resistance") if side == "long" else lv.get("support")
        c["conv"] += config.PLAYBOOK_W_ROOM * min((near[0]["dist_atr"] if near else 0.0), 2.0)
    cands.sort(key=lambda x: -x["conv"])
    ideas = []
    per_sec: dict = {}
    for c in cands:
        if "st" not in c:                               # only rank/diversify among enriched finalists
            break
        sec = labels.sector(c["symbol"]) or "?"
        if per_sec.get(sec, 0) >= config.PLAYBOOK_MAX_PER_SECTOR:    # diversification cap
            continue
        per_sec[sec] = per_sec.get(sec, 0) + 1
        g = c["group"]
        ideas.append({"symbol": c["symbol"], "group": (g["name"] if g else None), "sector": sec,
                      "rs": (g.get("rs") if g else None), "tags": list(dict.fromkeys(c["tags"])),
                      "conv": round(c["conv"], 2), "developing": c["developing"], "note": _note(c, side)})
        if len(ideas) >= top_n:
            break
    return ideas


def playbook(market_read: dict, groups: list[dict], frames: dict, frames15: dict | None = None,
             as_of: str | None = None, hits=None, top_n: int | None = None) -> dict:
    """{summary, long_ideas, short_ideas, mode}. Quality-gated, volatility-normalized conviction ranker over
    group + setup candidates. LIVE (frames15 given) uses intraday; CLOSE/EOD (None) uses settled data."""
    top_n = top_n or config.PLAYBOOK_TOP_N
    live = frames15 is not None
    close = market_read.get("close") or {}
    lv = market_read.get("live")
    blocks = close.get("blocks", [])
    risk_why = next((b["why"] for b in blocks if b["name"] == "risk" and b.get("score") is not None), "")
    vol_why = next((b["why"] for b in blocks if b["name"] == "vol" and b.get("score") is not None), "")
    summary = f"{close.get('stance', '?')} ({close.get('score')} close" + (f" / {lv.get('score')} live" if lv else "") + ")."
    summary += f" [{'live' if live else 'close'} mode]"
    if risk_why:
        summary += f" Rotation — {risk_why}."
    if "tech risk" in vol_why:
        summary += " " + vol_why.split(";")[-1].strip() + "."
    regime = max(-0.5, min(0.5, (close.get("score") or 0) / 100.0)) * 0.3    # light long/short tilt by stance
    longmap, shortmap = _group_maps(groups)
    smap = _setups_map(hits)
    return {"summary": summary, "mode": "live" if live else "close",
            "long_ideas": _side("long", frames, frames15, as_of, longmap, smap, live, regime, top_n),
            "short_ideas": _side("short", frames, frames15, as_of, shortmap, smap, live, regime, top_n)}
