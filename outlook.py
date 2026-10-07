"""Fit-free NEXT-MONTH RISK outlook — the app surface of awareness_lab (2026-07-05).

Reads data/features/awareness_outlook.json (per era-robust dial: full-history quintile
edges + historical P(-5%/-3% dip within 21 sessions) per quintile — no fitted weights;
the lab REJECTED fitted models at these horizons) and composes a plain-English payload
for any as-of date: current quintile per dial via the feature matrix (PRE json overlay
for the live/next session), average of the dials' conditional odds vs base -> level,
plus a structure-persistence line (trendlab class + %>20dma quintile -> historical share
of the next month spent in the same structure).

Honesty: the odds are one-dial-at-a-time historical conditionals AVERAGED across dials —
context with receipts, not a joint probability model.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config
import datastore
import trendlab

FDIR = config.DATA_DIR / "features"
ART = FDIR / "awareness_outlook.json"
PRE = FDIR / "daytype_live_pre.json"

_CACHE: dict = {}


def _artifact() -> dict | None:
    try:
        mt = ART.stat().st_mtime
    except OSError:
        return None
    if _CACHE.get("mt") != mt:
        _CACHE.clear()
        _CACHE["mt"] = mt
        _CACHE["art"] = json.loads(ART.read_text())
    return _CACHE["art"]


def _matrix() -> pd.DataFrame:
    if "F" not in _CACHE:
        F = pd.read_parquet(FDIR / "daytype_features.parquet")
        F.index = pd.to_datetime(F.index)
        _CACHE["F"] = F
    return _CACHE["F"]


def _values(cols: list[str], asof: str | None) -> tuple[dict, str]:
    """Latest knowable dial values: matrix row <= asof; when asof is beyond the matrix
    (the live/next session) overlay the pre-open cache."""
    F = _matrix()
    idx = F.index if asof is None else F.index[F.index <= pd.Timestamp(asof)]
    if not len(idx):
        return {}, ""
    row = F.loc[idx[-1]]
    vals = {c: (float(row[c]) if c in row and pd.notna(row[c]) else None) for c in cols}
    vdate = str(idx[-1].date())
    if PRE.exists():
        try:
            pre = json.loads(PRE.read_text())
            if asof is None or pd.Timestamp(pre["date"]) <= pd.Timestamp(asof):
                if pd.Timestamp(pre["date"]) > idx[-1]:
                    feats = pre["syms"]["SPY"]["feats"]
                    for c in cols:
                        if c in feats:
                            vals[c] = float(feats[c])
                    vdate = pre["date"]
        except Exception:
            pass
    rx_needed = [c for c in cols if c.startswith("a_rx_") and vals.get(c) is None]
    if rx_needed:
        vals.update({c: v for c, v in _ratio_vals(asof).items() if c in cols})
    return vals, vdate


def _ratio_vals(asof: str | None) -> dict:
    """Ratio-structure dials computed FRESH from the daily store (latest settled close
    <= asof) — the sidecar's rows are shift(1), and the pre-open cache doesn't carry
    these; a live Monday read must see Friday's close, not Wednesday's."""
    import dayfeatures
    key = ("rx", asof)
    if _CACHE.get("rx_key") == key:
        return _CACHE["rx"]
    out = {}
    for tag, (a, b) in dayfeatures.RATIO_PAIRS.items():
        r = dayfeatures.ratio_frame(a, b)
        if asof is not None:
            r = r.loc[:asof]
        if len(r) < 80:
            continue
        atr = trendlab.atr_series(r)
        m50 = r["close"].rolling(50).mean().iloc[-1]
        if np.isfinite(atr[-1]) and atr[-1] > 0 and np.isfinite(m50):
            out[f"a_rx_{tag}_d50"] = float((r["close"].iloc[-1] - m50) / atr[-1])
    _CACHE["rx_key"] = key
    _CACHE["rx"] = out
    return out


def _quintile(v: float | None, edges: list[float]) -> int | None:
    if v is None or not np.isfinite(v):
        return None
    return int(np.searchsorted(edges, v, side="right"))


def _struct_word(asof: str | None) -> str:
    d = datastore.load_bars("SPY")
    if d is None:
        return "?"
    if asof is not None:
        d = d.loc[:asof]
    if len(d) < 60:
        return "?"
    w = trendlab.struct_label(int(trendlab.structure_series(d)[-1]))
    return {"up": "uptrend", "down": "downtrend"}.get(w, w)


def payload(asof: str | None = None) -> dict:
    art = _artifact()
    if art is None:
        return {"error": "no awareness_outlook.json — run awareness_lab.py --production"}
    dials = art["dials"]
    pc = next(iter(art["persist_dial"]))
    vals, vdate = _values(list(dials) + [pc], asof)
    base5, base3 = art["base"]["corr5_21"], art["base"]["corr3_21"]
    rows, p5s, p3s = [], [], []
    for c, d in dials.items():
        q = _quintile(vals.get(c), d["edges"])
        if q is None:
            continue
        p5, p3 = d["corr5_21"][q], d["corr3_21"][q]
        p5s.append(p5)
        p3s.append(p3)
        ratio = p5 / base5 if base5 else 1.0
        tone = "risky" if ratio >= 1.2 else ("calm" if ratio <= 0.8 else "neutral")
        rows.append({"label": d["label"], "q": q + 1, "p5": round(100 * p5), "tone": tone})
    if not p5s:
        return {"error": "no dial values available"}
    p5m, p3m = float(np.mean(p5s)), float(np.mean(p3s))
    ratio = p5m / base5
    level, color = ("ELEVATED", "dn") if ratio >= 1.25 else (
        ("LOW", "up") if ratio <= 0.75 else ("NORMAL", "dim"))
    risky = [r for r in rows if r["tone"] == "risky"]
    calm = [r for r in rows if r["tone"] == "calm"]
    frag = tailfrag = ""
    if level == "ELEVATED" and risky:
        frag = " — " + " and ".join(r["label"] for r in risky[:2]) + \
            (" flash warning" if len(risky) > 1 else " flashes warning")
    elif level == "LOW" and calm:
        frag = " — " + " and ".join(r["label"] for r in calm[:2]) + " read calm"
    elif level == "NORMAL" and risky:
        tailfrag = " · watch: " + ", ".join(r["label"] for r in risky[:2])
    headline = (f"next-month dip risk {level}{frag}: from days like this a −5% pullback "
                f"started within a month {100 * p5m:.0f}% of the time ({100 * base5:.0f}% base)"
                f" · −3% odds {100 * p3m:.0f}% ({100 * base3:.0f}% base){tailfrag}")
    # structure persistence line
    persist = None
    pd_ = art["persist_dial"][pc]
    q = _quintile(vals.get(pc), pd_["edges"])
    if q is not None:
        pv = pd_["persist21"][q]
        persist = {"cls": _struct_word(asof), "p": round(100 * pv),
                   "base": round(100 * art["base"]["persist21"]), "q": q + 1,
                   "note": f"structure: {_struct_word(asof)} now; with {pd_['label']} in "
                           f"Q{q + 1}, history kept the same structure {100 * pv:.0f}% of "
                           f"the next month ({100 * art['base']['persist21']:.0f}% base)"}
    return {"level": level, "color": color, "headline": headline, "dials": rows,
            "persist": persist, "values_date": vdate,
            "footnote": "historical odds, one dial at a time, averaged over the era-robust "
                        "dials — the lab rejected fitted models at this horizon; "
                        "refreshed by awareness_lab.py --production"}


if __name__ == "__main__":
    import sys
    print(json.dumps(payload(sys.argv[1] if len(sys.argv) > 1 else None), indent=1))
