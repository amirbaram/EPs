"""INDEX RS-DIVERGENCE LAB (Amir 2026-07-04, from TCG_MarketCorrelations): when all the
major indices pull back together and all but one make a LOWER low while one holds a
DOUBLE BOTTOM or HIGHER low, is the relatively strong index the strongest bouncer once
the market bounces — and do divergence events help the day-type classification?

EVENTS (point-in-time): every SPY confirmed pivot LOW (confirmation bar c from
trendlab.pivots_conf — the moment the app first knows the market pullback printed a
low). At bar c each index (SPY/QQQ/IWM/DIA) is classified vs its PRIOR confirmed
pivot low, with the trendlab DT/DB tolerance (delta = ATR x ATR_FRACTION):
    weak   = current pullback low  < prior L - delta   (lower low)
    db     = within +-delta                            (double bottom)
    strong = above prior L + delta                     (higher low)
The "current pullback low" is the index's own just-confirmed L (if confirmed within
R_CONF bars of c) or its RUNNING low since its last swing flip (the live down-swing
candidate — exactly trendlab's, per the pivot_cascade self-check). Indices whose last
L is stale (bounce older than R_CONF bars) drop the event. STRONG class per Amir
includes db ("a double bottom or a higher low").

H1 (the bounce test): on STRICT divergence events (exactly one strong, rest weak),
forward return of the strong index minus the weak indices' mean, in own-ATR units,
at +5/+10/+20 daily bars (+7/+14/+35 hourly) — overall and CONDITIONED on whether SPY
actually bounced (SPY fwd > 0), since the theory speaks to the bounce leg.
CONTROL: is "structure-strong" just trailing relative strength? Same pick-one-index
game on the same events with the trailing-RS leader (past TRAIL-bar ATR return);
report both picks' outperformance vs the other three + agreement rate.

H2 (day-type link, daily events): next-session SPY trend/up/down target rates
(exact daytype_prob label_b + ORB derivation) after divergence vs after uniform-LL
pullbacks vs base.

    .venv/bin/python index_rs_lab.py
Output: data/validation/index_rs_report.txt · data/features/index_rs_events.parquet
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import config
import datastore
import daytype
import trendlab
from pivot_cascade import load_5m, resample_1h

SYMS = ["SPY", "QQQ", "IWM", "DIA"]
R_CONF = 3                       # just-confirmed L still counts as "this pullback"
HORIZONS = {"daily": [5, 10, 20], "1h": [7, 14, 35]}
TRAIL = {"daily": 20, "1h": 35}
REPORT = config.DATA_DIR / "validation" / "index_rs_report.txt"
OUTP = config.DATA_DIR / "features" / "index_rs_events.parquet"


def prep(tf: str) -> dict[str, dict]:
    frames = ({s: datastore.load_bars(s) for s in SYMS} if tf == "daily"
              else {s: resample_1h(load_5m(s)) for s in SYMS})
    common = frames["SPY"].index
    for s in SYMS[1:]:
        common = common.intersection(frames[s].index)
    out = {}
    for s in SYMS:
        f = frames[s].loc[common]
        pidx, pprice, ptype, ptag, pconf = trendlab.pivots_conf(f)
        out[s] = {"f": f, "atr": trendlab.atr_series(f), "lo": f["low"].to_numpy(float),
                  "cl": f["close"].to_numpy(float), "pidx": pidx, "pprice": pprice,
                  "ptype": ptype, "pconf": np.asarray(pconf)}
    return out


def pullback_class(D: dict, c: int) -> tuple[str, float] | None:
    """(class, margin_atr) of the index's current pullback low vs its prior confirmed
    pivot low, knowable at bar c; None if the index isn't in this pullback."""
    kk = int(np.searchsorted(D["pconf"], c, side="right")) - 1
    a = D["atr"][c]
    if kk < 1 or not np.isfinite(a) or a <= 0:
        return None
    delta = a * trendlab.ATR_FRACTION
    if D["ptype"][kk] == trendlab.PIVOT_LOW:
        if c - D["pconf"][kk] > R_CONF:              # bounce already running — stale
            return None
        cur = D["pprice"][kk]
        if kk < 2:
            return None
        prior = D["pprice"][kk - 2]
    else:                                            # live down-swing: running candidate low
        cur = float(D["lo"][D["pconf"][kk]:c + 1].min())
        prior = D["pprice"][kk - 1]
    m = (cur - prior) / a
    cls = "weak" if cur < prior - delta else ("strong" if cur > prior + delta else "db")
    return cls, m


def collect(tf: str) -> pd.DataFrame:
    data = prep(tf)
    S = data["SPY"]
    hzs, trail = HORIZONS[tf], TRAIL[tf]
    n = len(S["f"])
    rows = []
    for k in range(len(S["pidx"])):
        if S["ptype"][k] != trendlab.PIVOT_LOW:
            continue
        c = int(S["pconf"][k])
        if c < trail or c + hzs[0] >= n:
            continue
        cls = {s: pullback_class(data[s], c) for s in SYMS}
        if any(v is None for v in cls.values()):
            continue
        r = {"tf": tf, "time": S["f"].index[c], "c": c}
        for s in SYMS:
            r[f"cls_{s}"], r[f"m_{s}"] = cls[s]
            a = data[s]["atr"][c]
            clo = data[s]["cl"]
            r[f"tr_{s}"] = (clo[c] - clo[c - trail]) / a
            for h in hzs:
                r[f"f{h}_{s}"] = (clo[c + h] - clo[c]) / a if c + h < n else np.nan
        strong = [s for s in SYMS if cls[s][0] in ("strong", "db")]
        weak = [s for s in SYMS if cls[s][0] == "weak"]
        r["n_strong"], r["n_weak"] = len(strong), len(weak)
        r["strict"] = len(strong) == 1 and len(weak) == len(SYMS) - 1
        r["pick_struct"] = (strong[0] if len(strong) == 1 else
                            max(SYMS, key=lambda s: cls[s][1]))
        r["pick_trail"] = max(SYMS, key=lambda s: r[f"tr_{s}"])
        rows.append(r)
    E = pd.DataFrame(rows)
    E["era"] = np.where(pd.to_datetime(E.time) < "2023", "pre23", "23+")
    return E


def outperf(E: pd.DataFrame, pick_col: str, h: int) -> pd.Series:
    """Picked index's fwd ATR return minus the mean of the other three."""
    f = E[[f"f{h}_{s}" for s in SYMS]].to_numpy()
    idx = np.array([SYMS.index(p) for p in E[pick_col]])
    own = f[np.arange(len(E)), idx]
    rest = (f.sum(axis=1) - own) / (len(SYMS) - 1)
    return pd.Series(own - rest, index=E.index)


def block(lines: list, E: pd.DataFrame, tf: str, name: str) -> None:
    hzs = HORIZONS[tf]
    if len(E) < 15:
        lines.append(f"  {name}: n {len(E)} — too few")
        return
    lines.append(f"  {name} (n {len(E)}):")
    for h in hzs:
        ok = E[f"f{h}_SPY"].notna()
        for cond, cn in ((ok, "all"), (ok & (E[f"f{h}_SPY"] > 0), "SPY bounced"),
                         (ok & (E[f"f{h}_SPY"] <= 0), "SPY kept falling")):
            sub = E[cond]
            if len(sub) < 10:
                continue
            st = outperf(sub, "pick_struct", h)
            tr = outperf(sub, "pick_trail", h)
            lines.append(f"    +{h:2d} bars · {cn:16s} n {len(sub):4d}  "
                         f"STRUCT pick {st.mean():+5.2f} ATR (med {st.median():+5.2f}, "
                         f"{100*(st>0).mean():3.0f}% >0)  vs TRAIL pick {tr.mean():+5.2f} "
                         f"(med {tr.median():+5.2f})")
    agree = (E.pick_struct == E.pick_trail).mean()
    lines.append(f"    picks agree {100*agree:.0f}% · strong-index counts: "
                 + " ".join(f"{s}:{(E.pick_struct==s).sum()}" for s in SYMS))


def daytype_link(lines: list, E: pd.DataFrame) -> None:
    f5 = pd.read_parquet(config.DATA_DIR / "tiingo" / "bars_5min" / "SPY.parquet")
    lab = daytype.classify_all(f5)
    orb = lab[[f"orb_{k}" for k in daytype.OR_WINDOWS]]
    tgt = pd.DataFrame({
        "trend": lab.label_b.str.startswith("trend"),
        "up": lab.label_b.isin(["trend_up", "drift_up"]) | ((lab.label_b == "mixed") & orb.eq("up").any(axis=1)),
        "down": lab.label_b.isin(["trend_down", "drift_down"]) | ((lab.label_b == "mixed") & orb.eq("down").any(axis=1)),
    })
    dates = pd.DatetimeIndex(tgt.index)
    nxt = {}
    for _, r in E.iterrows():
        i = dates.searchsorted(pd.Timestamp(r.time), side="right")
        if i < len(dates):
            nxt[r.name] = tgt.iloc[i]
    N = pd.DataFrame(nxt).T
    E2 = E.loc[N.index]
    base = tgt.mean()
    lines.append("── H2 · daily events -> NEXT-SESSION SPY day-type target rates ──")
    lines.append(f"  base rates (all sessions):     trend {100*base.trend:3.0f}%  "
                 f"up {100*base.up:3.0f}%  down {100*base.down:3.0f}%   (n {len(tgt)})")
    for mask, nm in ((pd.Series(True, index=E2.index),
                      "ALL pullback-low events (the fair control)"),
                     (E2.strict, "strict divergence (1 strong, 3 LL)"),
                     ((E2.n_strong >= 1) & (E2.n_weak >= 1), "any divergence"),
                     (E2.n_weak == 4, "uniform LL (no one held)"),
                     (E2.n_strong == 4, "uniform strength (no one broke)")):
        n = N[mask.to_numpy()]
        if len(n) < 12:
            lines.append(f"  {nm:44s} n {len(n):3d} — too few")
            continue
        lines.append(f"  {nm:44s} n {len(n):3d}  trend {100*n.trend.mean():3.0f}%  "
                     f"up {100*n.up.mean():3.0f}%  down {100*n.down.mean():3.0f}%")


def main() -> None:
    lines = ["INDEX RS-DIVERGENCE LAB — does the index that HELD its low bounce strongest, "
             "and do divergence events inform the day type? (TCG_MarketCorrelations)", ""]
    allE = []
    for tf in ("daily", "1h"):
        E = collect(tf)
        allE.append(E)
        combos = E.groupby(["n_strong", "n_weak"]).size()
        lines.append(f"── {tf} · {len(E)} SPY pullback-low events (all 4 indices in the "
                     f"pullback) · strict divergence {int(E.strict.sum())} ──")
        lines.append("  (strong,weak) counts: "
                     + "  ".join(f"({i},{j}):{v}" for (i, j), v in combos.items()))
        block(lines, E[E.strict], tf, "STRICT — Amir's case: 1 holds (HL/DB), other 3 make LLs")
        for era in ("pre23", "23+"):
            block(lines, E[E.strict & (E.era == era)], tf, f"strict · {era}")
        block(lines, E[(E.n_strong >= 1) & (E.n_weak >= 1)], tf,
              "LOOSE — any split (pick = biggest hold-margin)")
        lines.append("")
    daytype_link(lines, allE[0])
    lines += ["", "Outperformance = picked index's forward ATR-return minus the mean of the "
              "other three. STRUCT pick = the index holding above its prior low (margin "
              "tie-break); TRAIL pick = trailing-RS leader on the same events — if the two "
              "columns match, structure adds nothing beyond plain relative strength. "
              "Windows overlap across nearby events; treat n as optimistic."]
    txt = "\n".join(lines)
    REPORT.write_text(txt)
    print(txt)
    A = pd.concat(allE, ignore_index=True)
    A.to_parquet(OUTP)
    print(f"\n-> {OUTP} ({len(A)}) · {REPORT}")


if __name__ == "__main__":
    main()
