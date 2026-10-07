"""PROB V2 stage 1 (Amir approved 2026-07-05): per-bar SESSION features for every RTH 5-min
bar, so day-type probabilities can update every 5 minutes instead of at 9 fixed checkpoints.

One row per (session, bar). Features use bars 0..i only (point-in-time within the day):
  s_move_atr / s_absmove_atr   open->close[i] move in prior-ATR units (signed / abs)
  s_eff                        |move| / cumulative close-to-close path (trendiness)
  s_range_atr / s_cpos         session range so far in ATR / close position in that range
  s_vwap_dist_atr              close - session VWAP, in ATR
  s_vwap_frac                  fraction of closes so far above VWAP
  s_vwap_run                   signed current run of closes on one side of VWAP (bars)
  s_run_max                    longest one-sided VWAP run seen so far today
  s_w1b                        1 after W1b has fired (2 consecutive closes wrong side of
                               VWAP following a >=18-bar one-sided run) — H1 amber input
  s_ib_ext_atr                 signed extension beyond the 1st-hour Initial Balance (ATR);
                               NaN before 10:30, 0 while inside the IB
  s_ib_brk_up / s_ib_brk_dn    IB extremes taken out so far (0/1; NaN before 10:30)
  s_stall                      bars since the session extreme in the move direction
  s_dmove_1h / s_dvwapd_1h /   1-hour DERIVATIVES (value now minus 12 bars ago) of move,
  s_dcpos_1h / s_drange_1h     VWAP distance, close position and range — the trajectory
                               v1 captured by stacking several checkpoint snapshots, and
                               H1's strongest reversal signal family; NaN before 10:30
  s_breadth                    universe %>VWAP at the LATEST checkpoint <= bar close
                               (0945/1000/1030/1130/1230/1330/1430; NaN before 09:45)
  s_breadth_d30                s_breadth - the 10:30 reading (afternoon divergence)
  s_breadth_dhr                s_breadth - the reading one hour earlier — H1 red input

Every numeric feature also gets a BAR-OF-DAY-CONDITIONAL expanding past-percentile column
``*_pct``: the value ranked against ALL PRIOR SESSIONS' value at the SAME bar index (same
(r-1)/(n-1) convention as dayfeatures.py), so "move 0.4 ATR" is judged differently at 09:50
vs 14:30. Only full 78-bar sessions are kept.

    .venv/bin/python daytype_bar_features.py [--sym SPY]
Output: data/features/daytype_bar_features{_sym}.parquet
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import config

FDIR = config.DATA_DIR / "features"
STORE = config.DATA_DIR / "tiingo" / "bars_5min"
NBARS = 78                       # 09:30..15:55 5-min bars; bar i closes at 575+5i minutes
ONESIDED = 18                    # 90-minute one-sided VWAP run (daytype_earlywarn.W1b)
BCOLS = [(585, "c0945_pct_above_vwap_0945"), (600, "c1000_pct_above_vwap_1000"),
         (630, "c_pct_above_vwap_1030"), (690, "c1130_pct_above_vwap_1130"),
         (750, "c1230_pct_above_vwap_1230"), (810, "c1330_pct_above_vwap_1330"),
         (870, "c1430_pct_above_vwap_1430")]


def session_features(op, hi, lo, cl, vol, atr) -> dict:
    """All per-bar features for one 78-bar session (numpy arrays), cumulative/PIT."""
    n = len(cl)
    tp = (hi + lo + cl) / 3
    v = np.maximum(vol, 1.0)
    vwap = np.cumsum(tp * v) / np.cumsum(v)
    move = cl - op
    path = np.cumsum(np.abs(np.diff(np.concatenate([[op], cl]))))
    cum_hi = np.maximum.accumulate(hi)
    cum_lo = np.minimum.accumulate(lo)
    rng = cum_hi - cum_lo
    out = {
        "s_move_atr": move / atr,
        "s_absmove_atr": np.abs(move) / atr,
        "s_eff": np.where(path > 0, np.abs(move) / path, np.nan),
        "s_range_atr": rng / atr,
        "s_cpos": np.where(rng > 0, (cl - cum_lo) / rng, np.nan),
        "s_vwap_dist_atr": (cl - vwap) / atr,
        "s_vwap_frac": np.cumsum(cl > vwap) / np.arange(1, n + 1),
    }
    # signed VWAP-side run, longest run so far, W1b fired-yet flag, stall
    side = np.where(cl > vwap, 1, -1)
    run = np.zeros(n); rmax = np.zeros(n); w1b = np.zeros(n); stall = np.zeros(n)
    r, best, prev_best, fired = 0, 0.0, 0.0, False
    for i in range(n):
        if i and side[i] == side[i - 1]:
            r += 1
        else:
            prev_best = best if i else 0.0
            r = 1
        best = max(best, float(r))
        # W1b: >=2 bars into a new side after the OLD side ran >= ONESIDED bars
        if not fired and r >= 2 and prev_best >= ONESIDED:
            fired = True
        run[i] = side[i] * r
        rmax[i] = best
        w1b[i] = 1.0 if fired else 0.0
        sgn = 1.0 if move[i] >= 0 else -1.0
        stall[i] = i - int(np.argmax(sgn * cl[:i + 1]))
    out["s_vwap_run"] = run
    out["s_run_max"] = rmax
    out["s_w1b"] = w1b
    out["s_stall"] = stall
    # 1-hour deltas (12 bars back, within-session only)
    def d1h(a):
        o = np.full(n, np.nan)
        o[12:] = a[12:] - a[:-12]
        return o
    out["s_dmove_1h"] = d1h(out["s_move_atr"])
    out["s_dvwapd_1h"] = d1h(out["s_vwap_dist_atr"])
    out["s_dcpos_1h"] = d1h(out["s_cpos"])
    out["s_drange_1h"] = d1h(out["s_range_atr"])
    # Initial Balance (bars 0..11, known from the close of bar 11 = 10:30)
    ib_hi = hi[:12].max(); ib_lo = lo[:12].min()
    ext = (np.maximum(cum_hi - ib_hi, 0) - np.maximum(ib_lo - cum_lo, 0)) / atr
    known = np.arange(n) >= 11
    out["s_ib_ext_atr"] = np.where(known, ext, np.nan)
    out["s_ib_brk_up"] = np.where(known, (cum_hi > ib_hi).astype(float), np.nan)
    out["s_ib_brk_dn"] = np.where(known, (cum_lo < ib_lo).astype(float), np.nan)
    return out


def breadth_track(day_row: pd.Series | None) -> dict:
    """Checkpoint breadth carried forward to every bar (same rule as the dev-page cursor
    box): value of the latest checkpoint whose close time <= this bar's close time."""
    t_close = 575 + 5 * np.arange(NBARS)
    br = np.full(NBARS, np.nan); d30 = np.full(NBARS, np.nan); dhr = np.full(NBARS, np.nan)
    if day_row is not None:
        vals = [(t, float(day_row[c])) for t, c in BCOLS
                if c in day_row.index and pd.notna(day_row[c])]
        for j, (t, v) in enumerate(vals):
            m = t_close >= t if j == len(vals) - 1 else \
                (t_close >= t) & (t_close < vals[j + 1][0])
            br[m] = v
            v30 = dict(vals).get(630)
            if v30 is not None and t >= 630:
                d30[m] = v - v30
            prev = dict(vals).get(t - 60)
            if prev is not None:
                dhr[m] = v - prev
    return {"s_breadth": br, "s_breadth_d30": d30, "s_breadth_dhr": dhr}


def build(sym: str) -> pd.DataFrame:
    bars = pd.read_parquet(STORE / f"{sym}.parquet")
    rth = bars.between_time("09:30", "15:59")
    # prior-14-session ATR from the same 5-min-derived dailies daytype.classify_all uses
    import daytype
    daily = daytype.classify_all(bars)
    prev_c = daily["close"].shift(1)
    tr = pd.concat([daily["high"] - daily["low"], (daily["high"] - prev_c).abs(),
                    (daily["low"] - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().shift(1)
    fp = FDIR / ("daytype_features.parquet" if sym == "SPY"
                 else f"daytype_features_{sym}.parquet")
    F = pd.read_parquet(fp) if fp.exists() else pd.read_parquet(
        FDIR / "daytype_features.parquet")

    frames = []
    for d, g in rth.groupby(rth.index.date):
        d = str(d)
        a = atr.get(d, np.nan)
        if len(g) != NBARS or not pd.notna(a) or a <= 0:
            continue
        feats = session_features(g["open"].iloc[0], g["high"].to_numpy(float),
                                 g["low"].to_numpy(float), g["close"].to_numpy(float),
                                 g["volume"].to_numpy(float), float(a))
        feats.update(breadth_track(F.loc[d] if d in F.index else None))
        f = pd.DataFrame(feats)
        f.insert(0, "date", d)
        f.insert(1, "bar", np.arange(NBARS, dtype=np.int16))
        f.insert(2, "mins", (575 + 5 * np.arange(NBARS)).astype(np.int16))
        frames.append(f)
    B = pd.concat(frames, ignore_index=True)

    # bar-of-day-conditional expanding past percentiles (dayfeatures' (r-1)/(n-1) convention,
    # ranked only against prior sessions' value at the SAME bar index)
    scols = [c for c in B.columns if c.startswith("s_")]
    ndays = len(frames)
    for c in scols:
        mat = pd.DataFrame(B[c].to_numpy().reshape(ndays, NBARS))
        r = mat.expanding().rank()
        n = mat.expanding().count()
        pct = (100 * (r - 1) / (n - 1).replace(0, np.nan)).where(mat.notna())
        B[c + "_pct"] = pct.to_numpy().reshape(-1)
    return B


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sym", default="SPY")
    args = ap.parse_args()
    B = build(args.sym)
    sfx = "" if args.sym == "SPY" else f"_{args.sym}"
    out = FDIR / f"daytype_bar_features{sfx}.parquet"
    B.to_parquet(out)
    days = B["date"].nunique()
    print(f"{args.sym}: {len(B)} bar-rows across {days} sessions "
          f"({B['date'].iloc[0]}..{B['date'].iloc[-1]}) -> {out}")


if __name__ == "__main__":
    main()
