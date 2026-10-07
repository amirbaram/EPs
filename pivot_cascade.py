"""MTF PIVOT-CASCADE STUDY (Amir 2026-07-04, from the TCG Trend Change Checklist):
does a LOWER-TF trend change tell you the HIGHER-TF's running extreme is already the
final pivot? His example first: 1h down-swing in progress (H confirmed, L not yet) +
5m flips to an UP segment / confirms a HL -> P(the running 1h low is THE pivot low).

METHOD
  HTF state per LTF bar, point-in-time: last HTF pivot k confirmed by the PRIOR HTF
  bar close (pconf from trendlab's single causal pass) sets the swing direction; the
  running extreme is the cummin/cummax of LTF lows/highs from the start of HTF bar
  pconf[k] through the current LTF bar (identical to trendlab's candidate — HTF
  extremes are compositions of their member LTF bars; reconstruction self-checked
  against truncated trendlab runs below).
  OUTCOME Y = the running extreme's PRICE equals the swing's final confirmed pivot
  price (pivot k+1). Y is age-dependent by construction — late in a swing everything
  is "more likely set" — so lift is measured against a base rate MATCHED on
  (swing age in HTF bars) x (ATR-distance of price from the running extreme),
  quartile cells fit on the all-bars base pool per pair/direction.
  SIGNALS (LTF events, both causal at LTF bar close):
    flip  = LTF structure FSM enters up/down (counter to the HTF swing)
    hlc   = LTF pivot confirms HL (during HTF down-swing) / LH (during up-swing)
             — the earlier, weaker hint
  REPORTED per pair/sym/direction/signal: matched lift, false-positive rate
  (signal fires but a new extreme still comes), LEAD TIME vs the HTF's own
  confirmation bar, first-signal-per-swing dedup, era split.

FRAMES  5m store (RTH, Tiingo) -> 1h (09:30-anchored, 7 bars/day) -> daily resample;
        daily->weekly uses the daily yfinance store + W-FRI (same source both sides).

    .venv/bin/python pivot_cascade.py            # full study + self-check
Outputs: data/validation/pivot_cascade_report.txt
         data/features/pivot_cascade_signals.parquet
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import config
import datastore
import trendlab

STORE = config.DATA_DIR / "tiingo" / "bars_5min"
REPORT = config.DATA_DIR / "validation" / "pivot_cascade_report.txt"
OUTP = config.DATA_DIR / "features" / "pivot_cascade_signals.parquet"
SYMS_INTRA = ["SPY", "QQQ", "IWM"]
SYMS_DAILY = ["SPY", "QQQ", "IWM", "DIA"]
MIN_CELL = 50          # matched cell needs this many base bars, else pooled fallback
UP, DOWN = 1, -1


# ---------------------------------------------------------------- frames

def load_5m(sym: str) -> pd.DataFrame:
    f = pd.read_parquet(STORE / f"{sym}.parquet")
    m = (f.index.time >= pd.Timestamp("09:30").time()) & (f.index.time <= pd.Timestamp("15:55").time())
    return f.loc[m]


def resample_1h(f5: pd.DataFrame) -> pd.DataFrame:
    """09:30-anchored RTH hours (bar starts 09:30..15:30 — the 15:30 'hour' is 30min,
    same as a TV RTH hourly chart)."""
    mins = f5.index.hour * 60 + f5.index.minute - 570
    key = f5.index.normalize() + pd.to_timedelta(570 + (mins // 60) * 60, unit="m")
    return _agg(f5, key)


def resample_daily(f5: pd.DataFrame) -> pd.DataFrame:
    return _agg(f5, f5.index.normalize())


def resample_weekly(d: pd.DataFrame) -> pd.DataFrame:
    """Weekly bars indexed by the FIRST session of each week (start-time index is
    what the LTF->HTF mapping needs; aggregation matches swing_scenarios' W-FRI)."""
    per = d.index.to_period("W-FRI")
    g = d.groupby(per)
    w = g.agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    w.index = g.apply(lambda x: x.index[0])
    return w.dropna()


def _agg(f: pd.DataFrame, key) -> pd.DataFrame:
    g = f.groupby(key)
    return g.agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


# ---------------------------------------------------------------- core

def swing_pool(lf: pd.DataFrame, hf: pd.DataFrame) -> pd.DataFrame:
    """One row per LTF bar inside a running HTF swing with a known outcome:
    dir, swing id k, age (HTF bars since the swing's origin extreme), dist
    (ATR-normalized pullback of price from the running extreme), Y, lead
    (HTF bars from this LTF bar to the swing's own confirmation)."""
    pidx, pprice, ptype, ptag, pconf = trendlab.pivots_conf(hf)
    atr_h = trendlab.atr_series(hf)
    lo = lf["low"].to_numpy(float)
    hi = lf["high"].to_numpy(float)
    cl = lf["close"].to_numpy(float)
    t_of_j = np.searchsorted(hf.index.to_numpy(), lf.index.to_numpy(), side="right") - 1
    first_j = np.searchsorted(t_of_j, np.arange(len(hf)), side="left")   # LTF start of each HTF bar

    rows = []
    for k in range(1, len(pidx) - 1):
        cbar, nconf = pconf[k], pconf[k + 1]
        if nconf <= cbar:                                   # both confirmed on one bar
            continue
        d = DOWN if ptype[k] == trendlab.PIVOT_HIGH else UP
        anchor = first_j[cbar]                              # LTF start of the flip bar
        a = first_j[cbar + 1] if cbar + 1 < len(hf) else len(lf)
        b = (first_j[nconf + 1] if nconf + 1 < len(hf) else len(lf)) - 1
        if b < a:
            continue
        ext = (np.minimum if d == DOWN else np.maximum).accumulate(
            (lo if d == DOWN else hi)[anchor:b + 1])[a - anchor:]
        tm1 = t_of_j[a:b + 1] - 1
        av = atr_h[tm1]
        y = ext == pprice[k + 1]
        dist = (cl[a:b + 1] - ext) / av if d == DOWN else (ext - cl[a:b + 1]) / av
        end_j = (first_j[nconf + 1] - 1) if nconf + 1 < len(hf) else len(lf) - 1
        rows.append(pd.DataFrame({
            "j": np.arange(a, b + 1), "k": k, "dir": d,
            "age": t_of_j[a:b + 1] - pidx[k], "dist": dist,
            "y": y.astype(float), "lead": nconf - t_of_j[a:b + 1],
            "lead_ltf": end_j - np.arange(a, b + 1),   # LTF bars until the confirming HTF close
            "ok": np.isfinite(av) & (av > 0)}))
    if not rows:
        return pd.DataFrame()
    P = pd.concat(rows, ignore_index=True)
    P = P[P.ok].drop(columns="ok")
    P["time"] = lf.index.to_numpy()[P.j]
    return P


def ltf_signals(lf: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(sig_up[j], sig_dn[j]) boolean event arrays, 2 signal kinds each:
    bit 1 = structure flip into up/down, bit 2 = counter pivot HL/LH confirmed."""
    st = trendlab.structure_series(lf)
    up = np.zeros(len(lf), dtype=np.int8)
    dn = np.zeros(len(lf), dtype=np.int8)
    # flip = ENTRY into the trend state; leaving the state re-arms (Amir's rule). The
    # trendlab hold fix (2026-07-05) makes exits genuine — an inside bar no longer
    # drops the state, so entry detection alone is dedup enough.
    was = np.roll(st, 1)
    was[0] = trendlab.S_UNKNOWN
    up[(st == trendlab.S_UP) & (was != trendlab.S_UP)] |= 1
    dn[(st == trendlab.S_DOWN) & (was != trendlab.S_DOWN)] |= 1
    lpidx, lpprice, lptype, lptag, lpconf = trendlab.pivots_conf(lf)
    for m in range(len(lpidx)):
        if lptag[m] in ("HL", "DB HL"):
            up[lpconf[m]] |= 2
        elif lptag[m] in ("LH", "DT LH"):
            dn[lpconf[m]] |= 2
    return up, dn


def matched(P: pd.DataFrame, sig_mask: pd.Series) -> dict | None:
    """Signal rows vs base matched on age x dist quartile cells (fit on the pool)."""
    S = P[sig_mask]
    if len(S) < 20:
        return None
    ab = pd.qcut(P.age, 4, labels=False, duplicates="drop")
    db = pd.qcut(P.dist, 4, labels=False, duplicates="drop")
    cell = ab.astype(str) + "|" + db.astype(str)
    grp = P.groupby(cell)["y"].agg(["mean", "size"])
    pooled = P.y.mean()
    base = cell[S.index].map(
        lambda c: grp.at[c, "mean"] if grp.at[c, "size"] >= MIN_CELL else pooled)
    tp = S[S.y == 1]
    first = S.sort_values("j").groupby("k").head(1)
    fbase = cell[first.index].map(
        lambda c: grp.at[c, "mean"] if grp.at[c, "size"] >= MIN_CELL else pooled)
    return {"n": len(S), "swings": S.k.nunique(),
            "p_sig": S.y.mean(), "p_match": base.mean(), "p_raw": pooled,
            "fp": 1 - S.y.mean(),
            "lead_med": tp.lead_ltf.median() if len(tp) else np.nan,
            "lead_1p": (tp.lead >= 1).mean() if len(tp) else np.nan,
            "n1": len(first), "p1": first.y.mean(), "p1_match": fbase.mean(),
            "S": S, "cell": cell}


def fmt(name, m, unit="LTF") -> str:
    if m is None:
        return f"  {name:34s} n<20 — skipped"
    lift = 100 * (m["p_sig"] - m["p_match"])
    l1 = 100 * (m["p1"] - m["p1_match"])
    return (f"  {name:34s} n {m['n']:5d} ({m['swings']:4d} swings)  "
            f"P {100*m['p_sig']:3.0f}% vs matched {100*m['p_match']:3.0f}% "
            f"(raw {100*m['p_raw']:3.0f}%)  lift {lift:+4.1f}pp  FP {100*m['fp']:3.0f}%  "
            f"lead med {m['lead_med']:3.0f} {unit} ({100*m['lead_1p']:2.0f}% "
            f">=1 HTF bar)  |  1st/swing n {m['n1']:4d}: "
            f"{100*m['p1']:3.0f}% vs {100*m['p1_match']:3.0f}% ({l1:+4.1f}pp)")


# ---------------------------------------------------------------- self-check

def self_check(hf: pd.DataFrame, name: str, n_try: int = 300) -> str:
    """Reconstruction == trendlab: truncate the HTF frame at random bars t; the live
    provisional swing (direction + candidate price) must equal (last pconf<=t pivot,
    running extreme since its confirmation bar)."""
    pidx, pprice, ptype, ptag, pconf = trendlab.pivots_conf(hf)
    lo, hi = hf["low"].to_numpy(float), hf["high"].to_numpy(float)
    rng = np.random.default_rng(7)
    bad = tried = 0
    for t in rng.integers(pconf[1] if len(pconf) > 1 else 50, len(hf), n_try):
        r = trendlab._compute(hf.iloc[:t + 1].copy())
        sw = r["swings"][-1] if r["swings"] else None
        if sw is None or not sw.provisional:
            continue
        k = int(np.searchsorted(pconf, t, side="right")) - 1
        if k < 0:
            continue
        tried += 1
        d = DOWN if ptype[k] == trendlab.PIVOT_HIGH else UP
        ext = (lo[pconf[k]:t + 1].min() if d == DOWN else hi[pconf[k]:t + 1].max())
        if (sw.kind != ("L" if d == DOWN else "H")) or not np.isclose(sw.price, ext):
            bad += 1
    return f"  {name}: {tried} truncations, {bad} mismatches"


# ---------------------------------------------------------------- study

UNIT = {"5m->1h": "5m-bars", "1h->daily": "1h-bars", "daily->weekly": "sessions"}


def run_pair(pname, sym, lf, hf, lines, allrows, checks):
    P = swing_pool(lf, hf)
    if P.empty:
        return
    up, dn = ltf_signals(lf)
    P["sig"] = np.where(P.dir == DOWN, up[P.j], dn[P.j])       # counter-direction events
    P["era"] = np.where(pd.to_datetime(P.time) < "2023", "pre23", "23+")
    checks.append(self_check(hf, f"{pname} {sym}"))
    lines.append(f"── {pname} · {sym} · pool {len(P)} LTF bars / "
                 f"{P.k.nunique()} HTF swings ──")
    for d, dname in ((DOWN, "HTF DOWN-swing (is the running LOW final?)"),
                     (UP, "HTF UP-swing (is the running HIGH final?)")):
        sub = P[P.dir == d].reset_index(drop=True)
        if not len(sub):
            continue
        lines.append(f"  {dname}")
        for bit, sname in ((1, "LTF flips counter-structure"),
                           (2, "LTF confirms counter HL/LH")):
            m = matched(sub, (sub.sig & bit) > 0)
            lines.append(fmt(sname, m, UNIT[pname]))
            if m is not None:
                S = m["S"].copy()
                S["pair"], S["sym"], S["signal"] = pname, sym, sname
                S["base_matched"] = m["p_match"]
                allrows.append(S.drop(columns="j"))
                for era in ("pre23", "23+"):
                    me = matched(sub[sub.era == era].reset_index(drop=True),
                                 pd.Series((sub[sub.era == era].sig.to_numpy() & bit) > 0))
                    if me is not None:
                        lines.append(f"    {era:6s}" + fmt("", me, UNIT[pname])[2:])
                # age-bin gradient for the headline pair
                if bit == 1 and d == DOWN and pname == "5m->1h":
                    q = pd.qcut(sub.age, 4, duplicates="drop")
                    sm = (sub.sig & 1) > 0
                    cells = []
                    for b, g in sub.groupby(q):
                        gs = g[sm[g.index]]
                        if len(gs) >= 15:
                            cells.append(f"age {b}: sig {100*gs.y.mean():.0f}% "
                                         f"vs base {100*g.y.mean():.0f}% (n{len(gs)})")
                    lines += ["    by swing age: " + " · ".join(cells)] if cells else []
        lines.append("")


def main() -> None:
    lines = [__doc__.splitlines()[0], ""]
    allrows, checks = [], []
    for sym in SYMS_INTRA:
        f5 = load_5m(sym)
        h1 = resample_1h(f5)
        d1 = resample_daily(f5)
        run_pair("5m->1h", sym, f5, h1, lines, allrows, checks)
        run_pair("1h->daily", sym, h1, d1, lines, allrows, checks)
    for sym in SYMS_DAILY:
        d = datastore.load_bars(sym)
        run_pair("daily->weekly", sym, d, resample_weekly(d), lines, allrows, checks)
    lines += ["── SELF-CHECK: PIT reconstruction vs truncated trendlab runs ──", *checks, ""]
    lines += ["Columns: P = P(running extreme is the final pivot) at signal bars; matched = "
              "base P at the same swing-age x ATR-distance cells; raw = unmatched pool P "
              "(inflated — shown to expose the matching effect); FP = signal fired but a new "
              "extreme still came; lead = LTF bars from the signal until the HTF bar that "
              "confirms the pivot CLOSES (+ share of true signals a full HTF bar early or "
              "more); 1st/swing = deduped to each swing's first signal (events cluster)."]
    txt = "\n".join(lines)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(txt)
    print(txt)
    if allrows:
        A = pd.concat(allrows, ignore_index=True)
        A.to_parquet(OUTP)
        print(f"\n-> {OUTP} ({len(A)} signal rows) · {REPORT}")


if __name__ == "__main__":
    main()
