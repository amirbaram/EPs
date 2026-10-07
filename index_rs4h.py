"""SHORT-TERM INDEX RELATIVE STRENGTH — Amir's 4h ratio-chart workflow (2026-07-05):
"if YM/SPY is in an uptrend and QQQ/SPY ranging, IWM/SPY downtrend, I may long the Dow
if ES rallies or short IWM if SPY falls; if YM/SPY is approaching resistance I'm cautious
even in an uptrend."

Per ratio (QQQ/SPY, IWM/SPY, DIA/SPY): component-OHLC 4h candles (two RTH bars/session,
resampled from the 5-min store; today's forming session appended from the live 15m RAM
store when available) -> trendlab structure word + nearest confirmed ratio pivot
above/below in ratio-ATR (the S/R proximity that turns "leader" into "leader, but don't
chase"). DESCRIPTIVE surface — structure facts + Amir's decision rule in plain English,
no fitted model, no probability claims.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import config
import trendlab

def _sub(sym: str):
    try:
        return pd.read_parquet(STORE / f"{sym}.parquet")
    except OSError:
        return None

STORE = config.DATA_DIR / "tiingo" / "bars_5min"
PAIRS = [("DIA", "Dow"), ("QQQ", "tech"), ("IWM", "small-caps")]
LOOKBACK_SESS = 130                      # ~260 4h bars of structure context
SR_NEAR_ATR = 1.0                        # "approaching" a ratio pivot level

_CACHE: dict = {}


def _bars4h(sym: str, asof: str | None, frames15: dict | None) -> pd.DataFrame:
    """4h RTH candles: 09:30-13:25 + 13:30-15:55 buckets from the 5-min store, with the
    live session (dates beyond the store) appended from the serve 15m RAM store."""
    f = pd.read_parquet(STORE / f"{sym}.parquet")
    if asof is not None:
        f = f.loc[:f"{asof} 23:59"]
    f = f[(f.index.hour * 60 + f.index.minute).to_series(index=f.index).between(570, 955).to_numpy()]
    last_store = f.index[-1].date() if len(f) else None
    if frames15 is not None and sym in frames15 and asof is None:
        g = frames15[sym]
        g = g[(g.index.hour * 60 + g.index.minute >= 570)
              & (g.index.hour * 60 + g.index.minute <= 945)]
        if last_store is not None:
            g = g[g.index.date > last_store]
        if len(g):
            f = pd.concat([f, g[["open", "high", "low", "close"]]])
    if not len(f):
        return pd.DataFrame()
    f = f.iloc[-LOOKBACK_SESS * 78:]
    key = f.index.normalize() + pd.to_timedelta(
        np.where(f.index.hour * 60 + f.index.minute < 810, 570, 810), unit="m")
    return f.groupby(key).agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


def _ratio4h(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    ix = a.index.intersection(b.index)
    r = pd.DataFrame({c: a.loc[ix, c] / b.loc[ix, c] for c in ("open", "high", "low", "close")})
    hi = r.max(axis=1)
    lo = r.min(axis=1)
    r["high"], r["low"] = hi, lo
    return r.dropna()


def _sr(r: pd.DataFrame, atr: float) -> tuple[float | None, float | None]:
    """Nearest CONFIRMED ratio pivot above/below the current value, in ratio-ATR."""
    pidx, pprice, ptype, ptag, pconf = trendlab.pivots_conf(r)
    cur = float(r["close"].iloc[-1])
    res = min((p for p in pprice if p > cur), default=None)
    sup = max((p for p in pprice if p < cur), default=None)
    return (round((res - cur) / atr, 2) if res is not None else None,
            round((cur - sup) / atr, 2) if sup is not None else None)


GRID_SYMS = ["SPY", "QQQ", "IWM", "DIA"]

# ── cascade early-pivot hint (Amir approved 2026-07-05; odds from pivot_cascade.py,
#    matched-base design: P(running extreme is final) at LTF counter-flips, SPY) ──
HINT_STATS = {("5m", "1h", "down"): (74, 26, "~50 min"), ("5m", "1h", "up"): (69, 31, "~45 min"),
              ("1h", "1D", "down"): (86, 14, "~6 hours"), ("1h", "1D", "up"): (77, 23, "~6 hours")}


def _bars1h(sym: str, asof: str | None, frames15: dict | None) -> pd.DataFrame:
    """1h RTH candles (09:30-anchored, 7/session) — same sourcing as _bars4h."""
    f = pd.read_parquet(STORE / f"{sym}.parquet")
    if asof is not None:
        f = f.loc[:f"{asof} 23:59"]
    m = f.index.hour * 60 + f.index.minute
    f = f[(m >= 570) & (m <= 955)]
    last_store = f.index[-1].date() if len(f) else None
    if frames15 is not None and sym in frames15 and asof is None:
        g = frames15[sym]
        gm = g.index.hour * 60 + g.index.minute
        g = g[(gm >= 570) & (gm <= 945)]
        if last_store is not None:
            g = g[g.index.date > last_store]
        if len(g):
            f = pd.concat([f, g[["open", "high", "low", "close"]]])
    if not len(f):
        return pd.DataFrame()
    f = f.iloc[-LOOKBACK_SESS * 78:]
    mins = f.index.hour * 60 + f.index.minute
    key = f.index.normalize() + pd.to_timedelta(570 + ((mins - 570) // 60) * 60, unit="m")
    return f.groupby(key).agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


def _pair_hint(lf: pd.DataFrame, hf: pd.DataFrame, lname: str, hname: str,
               recent: int) -> list[dict]:
    """One cascade check: is the HTF swing running AND did the LTF flip counter-structure
    within the last `recent` LTF bars? Descriptive — the odds come from the frozen study."""
    if len(lf) < 80 or len(hf) < 80:
        return []
    pidx, pprice, ptype, ptag, pconf = trendlab.pivots_conf(hf)
    if not pidx:
        return []
    k = len(pidx) - 1
    dirn = "down" if ptype[k] == trendlab.PIVOT_HIGH else "up"
    import pivot_cascade as pc
    sig_up, sig_dn = pc.ltf_signals(lf)          # deduped: re-arms only after trend break
    arr = sig_up if dirn == "down" else sig_dn
    flip = None
    for j in range(max(1, len(arr) - recent), len(arr)):
        if arr[j] & 1:
            flip = j
    if flip is None:
        return []
    p, fp, lead = HINT_STATS[(lname, hname, dirn)]
    ext = "low" if dirn == "down" else "high"
    run = float(hf["low"].iloc[pconf[k]:].min() if dirn == "down"
                else hf["high"].iloc[pconf[k]:].max())
    text = (f"the {lname} just flipped {'UP' if dirn == 'down' else 'DOWN'} inside the running "
            f"{hname} {dirn}-swing — historically the {hname} {ext} ({run:.2f}) was already in "
            f"{p}% of the time, known {lead} before the {hname} itself confirms "
            f"(wrong {fp}% — early notice, not confirmation)")
    return [{"pair": f"{lname}→{hname}", "dir": dirn, "p": p, "fp": fp, "lead": lead,
             "text": text, "flip_time": str(lf.index[flip])}]


def cascade_hints(asof: str | None = None, frames15: dict | None = None,
                  sym: str = "SPY") -> list[dict]:
    b1 = _bars1h(sym, asof, frames15)
    b5 = pd.read_parquet(STORE / f"{sym}.parquet")
    if asof is not None:
        b5 = b5.loc[:f"{asof} 23:59"]
    b5 = b5.iloc[-6 * 78:]
    out = _pair_hint(b5, b1, "5m", "1h", recent=8)
    out += _pair_hint(b1, _daily_bars(sym, asof), "1h", "1D", recent=3)
    return out


def _daily_bars(sym: str, asof: str | None) -> pd.DataFrame:
    import datastore
    d = datastore.load_bars(sym)
    if d is None:
        return pd.DataFrame()
    return d.loc[:asof] if asof is not None else d


def _bars2d(d: pd.DataFrame) -> pd.DataFrame:
    """2-session bars with CALENDAR-EPOCH parity (Amir 2026-07-05): sessions pair by their
    business-day count since 2000-01-03, so the pairing never shifts when history grows
    and is deterministic across frames. (Holidays offset the count identically every run.)"""
    g = np.busday_count("2000-01-03", d.index.values.astype("datetime64[D]")) // 2
    out = d.groupby(g).agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    starts = d.groupby(g).apply(lambda x: x.index[0])
    out.index = pd.DatetimeIndex(starts.values)
    return out


def _bars1w(d: pd.DataFrame) -> pd.DataFrame:
    """True calendar weeks (W-FRI, TV-parity), indexed by each week's first session."""
    per = d.index.to_period("W-FRI")
    g = d.groupby(per)
    out = g.agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    out.index = pd.DatetimeIndex(g.apply(lambda x: x.index[0]).values)
    return out.dropna()


def structure_grid(asof: str | None = None, frames15: dict | None = None) -> list[dict]:
    """Per index x timeframe trendlab structure word — the read Amir does by eye on the
    4h/1D/2D charts, in one glance. 4h includes the live session; 1D/2D/1W are settled."""
    rows = []
    for sym in GRID_SYMS:
        d = _daily_bars(sym, asof)
        cells = {}
        b4 = _bars4h(sym, asof, frames15)
        for tf, fr in (("4h", b4), ("1D", d), ("2D", _bars2d(d) if len(d) else d),
                       ("1W", _bars1w(d) if len(d) else d)):
            if len(fr) < 60:
                cells[tf] = None
                continue
            sb = _sub(sym) if tf in ("1D", "2D", "1W") else None
            segs, _ = trendlab.segment_chart(fr, subbars=sb)
            cur = segs[-1]
            word = cur.kind + (f"/{cur.subtype}" if cur.subtype else "")
            via = "structure state"
            if word == "transition":
                # Amir 2026-07-05: 'transition' hides the real read. First try the SWING TAGS
                # (his rule: higher lows + lower highs IS contracting per trendlab, even if the
                # live bar hasn't confirmed the FSM range state); else name what broke.
                pidx, pprice, ptype, ptag = trendlab.pivots(fr, subbars=sb)
                lastH = next((ptag[k] for k in range(len(ptag) - 1, -1, -1)
                              if ptype[k] == trendlab.PIVOT_HIGH), None)
                lastL = next((ptag[k] for k in range(len(ptag) - 1, -1, -1)
                              if ptype[k] == trendlab.PIVOT_LOW), None)
                # pivot-ENVELOPE check (Amir's eyeball rule, trendlab pivots only): last high
                # below the recent peak AND last low above the recent trough = narrowing
                env = ""
                hs = [(pidx[k], pprice[k]) for k in range(max(0, len(pidx) - 6), len(pidx))
                      if ptype[k] == trendlab.PIVOT_HIGH]
                ls = [(pidx[k], pprice[k]) for k in range(max(0, len(pidx) - 6), len(pidx))
                      if ptype[k] == trendlab.PIVOT_LOW]
                if len(hs) >= 2 and len(ls) >= 2:
                    delta = float(trendlab.atr_series(fr)[-1]) * trendlab.ATR_FRACTION
                    if (hs[-1][1] < max(p for _, p in hs) - delta
                            and ls[-1][1] > min(p for _, p in ls) + delta):
                        env = "narrowing pivot envelope (highs stepping down off the peak, lows holding above the trough)"
                if lastH in ("LH", "DT LH") and lastL in ("HL", "DB HL"):
                    word, via = "range/contracting", f"swing tags ({lastH}+{lastL})"
                elif lastH in ("DT HH", "DT LH") and lastL in ("DB HL", "DB LL"):
                    word, via = "range/rectangle", f"swing tags ({lastH}+{lastL})"
                elif lastH in ("HH", "DT HH") and lastL in ("LL", "DB LL"):
                    word, via = "range/expanding", f"swing tags ({lastH}+{lastL})"
                elif env:
                    word, via = "range/contracting", env
                elif lastH in ("LH", "DT LH") and lastL in ("LL", "DB LL"):
                    word, via = "rolling-over", f"lower high + lower low ({lastH}+{lastL}) — bearish bias, break not confirmed"
                elif lastH in ("HH", "DT HH") and lastL in ("HL", "DB HL"):
                    word, via = "turning-up", f"higher high + higher low ({lastH}+{lastL}) — bullish bias, break not confirmed"
                else:
                    ref = next((x for x in reversed(segs[:-1]) if x.kind != "transition"), None)
                    if ref is not None and ref.kind == "up":
                        word, via = "broken-up", "uptrend broke — now rangebound"
                    elif ref is not None and ref.kind == "down":
                        word, via = "broken-down", "downtrend broke — now rangebound"
                    else:
                        word, via = "pausing", "no reference structure"
            prev = ""
            if len(segs) > 1:
                p = segs[-2]
                prev = p.kind + (f"/{p.subtype}" if p.subtype else "")
            cells[tf] = {"w": word, "bars": cur.bars, "prev": prev, "via": via,
                         "since": str(fr.index[cur.start].date())}
        rows.append({"sym": sym, **cells})
    return rows


def read(asof: str | None = None, frames15: dict | None = None) -> dict:
    spy = _bars4h("SPY", asof, frames15)
    if len(spy) < 60:
        return {"error": "not enough 4h history"}
    rows = []
    for sym, nick in PAIRS:
        f = _bars4h(sym, asof, frames15)
        r = _ratio4h(f, spy)
        if len(r) < 60:
            continue
        st = trendlab.structure_series(r)
        word = trendlab.struct_label(int(st[-1]))
        word = {"up": "uptrend", "down": "downtrend", "transition": "no clear structure"}.get(word, word)
        atrv = float(trendlab.atr_series(r)[-1])
        res, sup = _sr(r, atrv) if atrv > 0 else (None, None)
        ret10 = float(100 * (r["close"].iloc[-1] / r["close"].iloc[-11] - 1)) if len(r) > 11 else 0.0
        score = (2 if word == "uptrend" else (-2 if word == "downtrend" else 0)) + np.clip(ret10, -1, 1)
        caution = None
        if word == "uptrend" and res is not None and res <= SR_NEAR_ATR:
            caution = f"ratio approaching resistance ({res:.1f} ATR away) — be careful chasing longs"
        elif word == "downtrend" and sup is not None and sup <= SR_NEAR_ATR:
            caution = f"ratio approaching support ({sup:.1f} ATR away) — be careful chasing shorts"
        rows.append({"sym": sym, "nick": nick, "pair": f"{sym}/SPY", "word": word,
                     "ret10": round(ret10, 2), "res_atr": res, "sup_atr": sup,
                     "score": float(score), "caution": caution})
    if not rows:
        return {"error": "no ratio data"}
    rows.sort(key=lambda x: -x["score"])
    lead, lag = rows[0], rows[-1]
    parts = [f"{r['nick']} {r['word']} ({r['pair']})" for r in rows]
    sug = []
    if lead["word"] == "uptrend":
        s = f"on market strength, longs favor {lead['sym']}"
        if lead["caution"]:
            s += f" — but {lead['caution']}"
        sug.append(s)
    if lag["word"] == "downtrend" and lag is not lead:
        s = f"on market weakness, shorts favor {lag['sym']}"
        if lag["caution"]:
            s += f" — but {lag['caution']}"
        sug.append(s)
    if not sug:
        sug.append("no index shows a clear short-term RS edge — trade the tape, not a tilt")
    return {"rows": rows, "asof_bar": str((_bars4h("SPY", asof, frames15)).index[-1]),
            "grid": structure_grid(asof, frames15),
            "hints": cascade_hints(asof, frames15),
            "headline": " · ".join(parts), "suggestion": "; ".join(sug),
            "footnote": "4h ratio candles vs SPY (component-OHLC) — structure + nearest "
                        "confirmed ratio pivot; descriptive read of Amir's ratio workflow, "
                        "not a fitted model"}


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(read(sys.argv[1] if len(sys.argv) > 1 else None), indent=1))
