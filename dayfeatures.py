"""Day-type FEATURE MATRIX (Amir's trend/range prediction study, 2026-07-04).

One row per SPY session (aligned to daytype.classify_all), columns prefixed by INFORMATION TIME:
  a_*  known BEFORE the open (prior-day / EOD / overnight-published data, external EOD feeds)
  b_*  overnight / pre-market (gap, Globex-proxy inventory, open location vs prior structure)
  c_*  known by ~10:30 (first-hour internals from the universe 5-min store)
Every numeric feature also gets an expanding PAST-percentile column ``*_pct`` (rank among all
history up to and including that day — no future data), per the study's "percentiles, not fixed
thresholds" rule. All features are point-in-time.

Inputs: the Tiingo 5-min stores (RTH + ETH), the daily store, data/trin_daily.csv,
data/tv/bars_daily_{ADD,TICK}.parquet, data/features/squeeze_gex.csv (SqueezeMetrics, lagged
1 day), data/features/hy_oas.csv (FRED), ^VIX/^VIX9D/^VIX3M/^TNX. Intraday TICK/ADD/VOLD/TRIN
trajectories have NO history — reserved for the forward loggers, not faked here.

    .venv/bin/python dayfeatures.py --universe   # heavy stage: %>VWAP@10:30 over ~1,650 names
    .venv/bin/python dayfeatures.py              # everything else + join -> parquet
Output: data/features/daytype_features.parquet
"""
from __future__ import annotations

import argparse
import datetime as dt
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import config
import datastore
import daytype
import universe as uni

STORE = config.DATA_DIR / "tiingo" / "bars_5min"
ETHSTORE = config.DATA_DIR / "tiingo" / "bars_5min_eth"
FDIR = config.DATA_DIR / "features"
OUT = FDIR / "daytype_features.parquet"
UNIV_1030 = FDIR / "univ_1030.parquet"
SPDR = ["XLK", "XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLU"]

# FOMC DECISION days (2nd meeting day; 2020 includes the two emergency cuts; 2026 = published
# schedule). Best-effort hardcode — a wrong date costs one flag, not a lookahead.
FOMC = """2017-02-01 2017-03-15 2017-05-03 2017-06-14 2017-07-26 2017-09-20 2017-11-01 2017-12-13
2018-01-31 2018-03-21 2018-05-02 2018-06-13 2018-08-01 2018-09-26 2018-11-08 2018-12-19
2019-01-30 2019-03-20 2019-05-01 2019-06-19 2019-07-31 2019-09-18 2019-10-30 2019-12-11
2020-01-29 2020-03-03 2020-03-15 2020-04-29 2020-06-10 2020-07-29 2020-09-16 2020-11-05 2020-12-16
2021-01-27 2021-03-17 2021-04-28 2021-06-16 2021-07-28 2021-09-22 2021-11-03 2021-12-15
2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27 2022-09-21 2022-11-02 2022-12-14
2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13
2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18
2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10
2026-01-28 2026-03-18 2026-04-29 2026-06-17 2026-07-29 2026-09-16 2026-10-28 2026-12-09""".split()


def _sess(df_idx) -> pd.DatetimeIndex:
    return pd.to_datetime(df_idx)


def spy_daily_block(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket A features from SPY's own daily bars (derived from the classified sessions)."""
    d = pd.DataFrame(index=df.index)
    rng, cl, op = df["range"], df["close"], df["open"]
    hi, lo = df["high"], df["low"]
    prev_c = cl.shift(1)
    tr = pd.concat([hi - lo, (hi - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    # A1 — range compression (all PRIOR-day: shift(1))
    d["a_nr4"] = (rng == rng.rolling(4).min()).shift(1).fillna(False)
    d["a_nr7"] = (rng == rng.rolling(7).min()).shift(1).fillna(False)
    d["a_inside"] = ((hi < hi.shift(1)) & (lo > lo.shift(1))).shift(1).fillna(False)
    d["a_prev_range_atr"] = (rng / atr).shift(1)
    d["a_range3_vs_atr"] = (rng.rolling(3).mean() / atr).shift(1)
    # A2 — prior day type + coiling
    d["a_prev_type_b"] = df["label_b"].shift(1)
    d["a_prev_type_d"] = df["label_d"].shift(1)
    is_rng = (df["label_d"] == "range").astype(int)
    grp = (is_rng == 0).cumsum()
    d["a_coil_days"] = is_rng.groupby(grp).cumsum().shift(1).fillna(0)
    # A3 — prior close location + consecutive closes
    d["a_prev_cpos"] = df["close_pos"].shift(1)
    up = (cl > prev_c).astype(int)
    runs = up.groupby((up != up.shift()).cumsum()).cumcount() + 1
    d["a_consec_closes"] = (runs * np.where(up == 1, 1, -1)).shift(1)
    # A4 — extension from MAs in ATR
    for n in (20, 50):
        d[f"a_dist_{n}ma_atr"] = ((cl - cl.rolling(n).mean()) / atr).shift(1)
    return d


def universe_daily_block(cal: pd.DatetimeIndex) -> pd.DataFrame:
    """ONE pass over the daily store: breadth, MA census, NH-NL + groups, closing ranges,
    near-high census, $vol-weighted breadth, McClellan. Values are the day's own EOD readings;
    the caller shifts them by 1 into bucket A."""
    import labels
    z = lambda: pd.Series(0.0, index=cal)
    acc = {k: z() for k in
           ("n", "gt20", "gt200", "stack", "inv", "nh", "nl", "nh20", "in5", "in15",
            "upv", "dnv", "adv", "dec", "cr_d", "cr_w", "cr_n")}
    gnh: dict[str, pd.Series] = {}
    for s in sorted(uni.active_symbols() or []):
        if uni.is_future(s):
            continue
        try:
            b = datastore.load_bars(s)
        except Exception:
            continue
        if b is None or len(b) < 60:
            continue
        c, h, l, v = b["close"], b["high"], b["low"], b["volume"]
        ok = c.notna()
        rei = lambda x: x.astype(float).reindex(cal, fill_value=0)
        acc["n"] += rei(ok)
        m20 = c.rolling(20).mean(); m50 = c.rolling(50).mean(); m200 = c.rolling(200).mean()
        acc["gt20"] += rei((c > m20) & m20.notna())
        acc["gt200"] += rei((c > m200) & m200.notna())
        acc["stack"] += rei((c > m20) & (m20 > m50) & (m50 > m200) & m200.notna())
        acc["inv"] += rei((c < m20) & (m20 < m50) & (m50 < m200) & m200.notna())
        hh = c.rolling(252).max().shift(1); ll = c.rolling(252).min().shift(1)
        nh = (c > hh) & hh.notna()
        acc["nh"] += rei(nh)
        acc["nl"] += rei((c < ll) & ll.notna())
        acc["nh20"] += rei((c > c.rolling(20).max().shift(1)))
        off = c / c.rolling(252).max() - 1
        acc["in5"] += rei((off >= -0.05) & hh.notna())
        acc["in15"] += rei((off >= -0.15) & hh.notna())
        dvol = (c * v).fillna(0)
        upd = c > c.shift(1)
        acc["upv"] += dvol.where(upd, 0).reindex(cal, fill_value=0)
        acc["dnv"] += dvol.where(~upd, 0).reindex(cal, fill_value=0)
        acc["adv"] += rei(upd)
        acc["dec"] += rei((c < c.shift(1)))
        rngd = (h - l).replace(0, np.nan)
        acc["cr_d"] += ((c - l) / rngd).fillna(0).reindex(cal, fill_value=0)
        wl = l.rolling(5).min(); wh = h.rolling(5).max()
        acc["cr_w"] += ((c - wl) / (wh - wl).replace(0, np.nan)).fillna(0).reindex(cal, fill_value=0)
        acc["cr_n"] += rei(rngd.notna())
        sec = labels.sector(s)
        if sec:
            gnh[sec] = gnh.get(sec, z()) + rei(nh)
    n = acc["n"].replace(0, np.nan)
    out = pd.DataFrame(index=cal)
    out["u_pct20dma"] = 100 * acc["gt20"] / n
    out["u_pct200dma"] = 100 * acc["gt200"] / n
    out["u_stack_pct"] = 100 * acc["stack"] / n
    out["u_invert_pct"] = 100 * acc["inv"] / n
    out["u_nhnl"] = acc["nh"] - acc["nl"]
    out["u_nh20"] = acc["nh20"]
    out["u_within5_pct"] = 100 * acc["in5"] / n
    out["u_within15_pct"] = 100 * acc["in15"] / n
    out["u_vold_ratio"] = acc["upv"] / acc["dnv"].replace(0, np.nan)
    net = (acc["adv"] - acc["dec"]) / n * 1000        # scaled net advancers
    out["u_mcclellan"] = net.ewm(span=19, adjust=False).mean() - net.ewm(span=39, adjust=False).mean()
    out["u_close_range_d"] = 100 * acc["cr_d"] / acc["cr_n"].replace(0, np.nan)
    out["u_close_range_w"] = 100 * acc["cr_w"] / acc["cr_n"].replace(0, np.nan)
    # group NH: count of groups with >=3 new highs + max streak length across groups
    if gnh:
        G = pd.DataFrame(gnh)
        out["u_groups_nh3"] = (G >= 3).sum(axis=1)
        alive = (G >= 1).astype(int)
        streak = alive.copy()
        for col in alive:
            g = (alive[col] == 0).cumsum()
            streak[col] = alive[col].groupby(g).cumsum()
        out["u_group_nh_maxstreak"] = streak.max(axis=1)
    return out


def dist_days(sym: str, cal) -> pd.Series:
    b = datastore.load_bars(sym)
    down = (b["close"].pct_change() < -0.002) & (b["volume"] > b["volume"].shift(1))
    return down.rolling(25).sum().reindex(cal, method="ffill")


def vol_block(cal: pd.DatetimeIndex) -> pd.DataFrame:
    d = pd.DataFrame(index=cal)
    vix = datastore.load_bars("^VIX")["close"].reindex(cal, method="ffill")
    d["a_vix"] = vix.shift(1)
    d["a_vix_chg1"] = vix.diff(1).shift(1)
    try:
        v9 = datastore.load_bars("^VIX9D")["close"].reindex(cal, method="ffill")
        v3m = datastore.load_bars("^VIX3M")["close"].reindex(cal, method="ffill")
        d["a_vix9d_vix"] = (v9 / vix).shift(1)
        d["a_vix_vix3m"] = (vix / v3m).shift(1)
        d["a_ts_inverted"] = ((vix / v3m) > 1).shift(1)
    except Exception:
        pass
    tnx = datastore.load_bars("^TNX")["close"].reindex(cal, method="ffill")
    d["a_tnx_chg1"] = tnx.diff(1).shift(1)
    return d


def external_block(cal: pd.DatetimeIndex, sym: str = "SPY") -> pd.DataFrame:
    d = pd.DataFrame(index=cal)
    if sym == "QQQ":                                    # NASDAQ internals (TV daily exports)
        if config.DAYTYPE_USE_TV_INTERNALS:             # TV manual export — stale-prone, ~0 skill (see config)
            for isym, cols in (("TRINQ", {"a_trinq": "close"}),
                               ("ADDQ", {"a_addq_close": "close"}),
                               ("TICKQ", {"a_tickq_hi": "high", "a_tickq_lo": "low"})):
                pth = config.DATA_DIR / "tv" / f"bars_daily_{isym}.parquet"
                if pth.exists():
                    b = pd.read_parquet(pth)
                    for k, c in cols.items():
                        d[k] = b[c].reindex(cal, method="ffill").shift(1)
            tq = config.DATA_DIR / "tv" / "bars_daily_TICKQ.parquet"
            if tq.exists():                             # cumulative TICKQ, 5-day slope
                ct = pd.read_parquet(tq)["close"].cumsum()
                d["a_cumtickq_ch5"] = ct.diff(5).reindex(cal, method="ffill").shift(1)
        try:                                            # ^VXN/^VIX (yfinance, self-pullable) — KEPT
            vxn = datastore.load_bars("^VXN")["close"].reindex(cal, method="ffill")
            vix_ = datastore.load_bars("^VIX")["close"].reindex(cal, method="ffill")
            d["a_vxn"] = vxn.shift(1)
            d["a_vxn_vix"] = (vxn / vix_).shift(1)
        except Exception:
            pass
    if sym != "SPY":                                    # cross features vs SPY
        try:
            own = datastore.load_bars(sym)["close"].reindex(cal, method="ffill")
            spyc = datastore.load_bars("SPY")["close"].reindex(cal, method="ffill")
            ratio = own / spyc
            d["a_rel5"] = (100 * ratio.pct_change(5)).shift(1)
            d["a_rel1"] = (100 * ratio.pct_change(1)).shift(1)
        except Exception:
            pass
    if config.DAYTYPE_USE_TV_INTERNALS:                 # NYSE TRIN + ADD/TICK (TV manual export) — stale-prone,
        trin_p = config.DATA_DIR / "trin_daily.csv"     # ~0 skill; OFF drops them from train + live (see config)
        if trin_p.exists():
            trin = pd.read_csv(trin_p, index_col="date")["close"]
            trin.index = pd.to_datetime(trin.index)
            t = trin.reindex(cal, method="ffill").shift(1)
            d["a_trin"] = t
            d["a_trin_hi"] = t > 2.0
            d["a_trin_lo"] = t < 0.6
        for isym, cols in (("ADD", {"a_add_close": "close"}),
                           ("TICK", {"a_tick_hi": "high", "a_tick_lo": "low"})):
            p = config.DATA_DIR / "tv" / f"bars_daily_{isym}.parquet"
            if p.exists():
                b = pd.read_parquet(p)
                for k, c in cols.items():
                    d[k] = b[c].reindex(cal, method="ffill").shift(1)
    gex_p = FDIR / "squeeze_gex.csv"
    if gex_p.exists():
        g = pd.read_csv(gex_p)
        g["date"] = pd.to_datetime(g["date"])
        g = g.set_index("date")
        gex = g["gex"].reindex(cal, method="ffill").shift(1)   # published pre-open next day
        d["a_gex"] = gex
        d["a_gex_neg"] = gex < 0
        d["a_gex_chg"] = g["gex"].diff().reindex(cal, method="ffill").shift(1)
        d["a_dix"] = g["dix"].reindex(cal, method="ffill").shift(1)
    hy_p = FDIR / "hy_oas.csv"
    if hy_p.exists():
        h = pd.read_csv(hy_p)
        h.columns = ["date", "oas"]
        h["oas"] = pd.to_numeric(h["oas"], errors="coerce")
        h["date"] = pd.to_datetime(h["date"])
        oas = h.set_index("date")["oas"].reindex(cal, method="ffill")
        d["a_hy_chg1"] = oas.diff(1).shift(1)
        d["a_hy_chg5"] = oas.diff(5).shift(1)
    hyg_p = FDIR / "hyg_ief.csv"
    if hyg_p.exists():                                  # full-history credit dial (proxy for OAS Δ)
        hy = pd.read_csv(hyg_p, index_col=0, parse_dates=True)
        ratio = (hy["HYG"] / hy["IEF"]).reindex(cal, method="ffill")
        d["a_credit_chg5"] = (100 * ratio.pct_change(5)).shift(1)
    return d


def _wilder_rsi(c: pd.Series, n: int = 14) -> pd.Series:
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def med_block(cal: pd.DatetimeIndex) -> pd.DataFrame:
    """MED bucket-A (TCG Market Environment Dossier, 2026-07-05): prior-day / overnight
    trend states of correlated instruments. PIT rule — shift(1) is applied on each
    instrument's OWN calendar BEFORE reindexing to the session calendar, so BTC's
    weekend bars are carried into Monday and foreign holidays ffill without look-ahead.
      shift(1): DX-Y.NYB, BTC-USD (UTC daily bar closes ~19:00 ET), ^GDAXI (closes
      11:30 ET — same-day is NOT pre-open knowable), leaders, sectors.
      no shift: ^HSI (HK closes ~04:00 ET, same morning IS pre-open knowable; live
      prep may still hold yesterday's row -> stale ffill, never look-ahead).
    Damage-breadth flags: index_rs_lab daily pullback-low events (knowable at the
    confirmation close) flag the NEXT 3 sessions — uniform-LL pullbacks ran next-day
    down 16% vs 7% when one index held (2026-07-05 lab).
    a_med_lean composite: bull-minus-bear sum of the block's components, weights =
    top-vs-bottom-quintile P(up)-delta fit on TRAIN (<2025) SPY targets only (data-
    driven per the dossier spec; weights in data/features/med_composite_weights.json)."""
    import json

    import trendlab

    d = pd.DataFrame(index=cal)

    def aligned(sym, fn, shift=1):
        b = datastore.load_bars(sym)
        if b is None or len(b) < 60:
            return None
        return fn(b).shift(shift).reindex(cal, method="ffill")

    for sym, tag, sh in (("DX-Y.NYB", "dxy", 1), ("BTC-USD", "btc", 1),
                         ("^GDAXI", "dax", 1), ("^HSI", "hsi", 0)):
        r1 = aligned(sym, lambda b: 100 * b["close"].pct_change(1), sh)
        if r1 is None:
            continue
        d[f"a_med_{tag}_ret1"] = r1
        if tag in ("dxy", "btc"):
            d[f"a_med_{tag}_ret5"] = aligned(sym, lambda b: 100 * b["close"].pct_change(5))
            st = aligned(sym, lambda b: pd.Series(
                trendlab.structure_series(b), index=b.index))
            d[f"a_med_{tag}_up"] = st == trendlab.S_UP
            d[f"a_med_{tag}_down"] = st == trendlab.S_DOWN

    # leaders: prior-day trendlab class + RSI>50 counts; SMH kept separate (semis lead)
    upn = pd.Series(0.0, index=cal)
    rsin = pd.Series(0.0, index=cal)
    for sym in ("AAPL", "MSFT", "TSLA", "SMH"):
        st = aligned(sym, lambda b: pd.Series(trendlab.structure_series(b), index=b.index))
        r = aligned(sym, lambda b: _wilder_rsi(b["close"]))
        if st is not None:
            upn += (st == trendlab.S_UP).astype(float)
        if r is not None:
            rsin += (r > 50).astype(float)
    d["a_med_ldr_up_n"] = upn
    d["a_med_ldr_rsi_n"] = rsin
    d["a_med_smh_ret1"] = aligned("SMH", lambda b: 100 * b["close"].pct_change(1))

    xr = {s: aligned(s, lambda b: 100 * b["close"].pct_change(1)) for s in ("XLE", "XLF", "XLV")}
    for s, v in xr.items():
        if v is not None:
            d[f"a_med_{s.lower()}_ret1"] = v
    if xr["XLV"] is not None and xr["XLF"] is not None:
        d["a_med_def_cyc"] = xr["XLV"] - xr["XLF"]        # positive = defensive tone

    try:
        import index_rs_lab
        E = index_rs_lab.collect("daily")
        none_held = pd.Series(False, index=cal)
        someone = pd.Series(False, index=cal)
        for _, r in E.iterrows():
            p = int(cal.searchsorted(pd.Timestamp(r.time), side="right"))
            for q in range(p, min(p + 3, len(cal))):
                if r.n_weak == 4:
                    none_held.iloc[q] = True
                elif r.n_strong >= 1 and r.n_weak >= 1:
                    someone.iloc[q] = True
        d["a_med_dmg_none_held"] = none_held
        d["a_med_dmg_someone_held"] = someone
    except Exception as e:
        print(f"  med damage-breadth skipped: {e}")

    # composite lean (train-only weights; PIT expanding percentiles as inputs)
    try:
        lab = daytype.classify_all(pd.read_parquet(STORE / "SPY.parquet"))
        orb_up = lab[[f"orb_{k}" for k in daytype.OR_WINDOWS]].eq("up").any(axis=1)
        y = (lab.label_b.isin(["trend_up", "drift_up"])
             | ((lab.label_b == "mixed") & orb_up)).astype(float)
        y.index = pd.to_datetime(y.index)               # classify_all indexes by date STRINGS
        y = y.reindex(cal)
        train = pd.Series(cal < pd.Timestamp("2025-01-01"), index=cal)
        comp = pd.Series(0.0, index=cal)
        wts = {}
        for c in list(d.columns):
            s = d[c].astype(float)
            binary = s.dropna().nunique() <= 2
            if binary:
                x = s
            else:
                rk = s.expanding().rank()
                n = s.expanding().count()
                x = ((rk - 1) / (n - 1).replace(0, np.nan)).where(s.notna())
            m = train & x.notna() & y.notna()
            if m.sum() < 200:
                continue
            hi = x[m] >= (1 if binary else 0.8)
            lo = x[m] <= (0 if binary else 0.2)
            if hi.sum() < 30 or lo.sum() < 30:
                continue
            w = float(y[m][hi].mean() - y[m][lo].mean())
            wts[c] = round(w, 4)
            comp = comp.add(w * (x - 0.5), fill_value=0.0)
        d["a_med_lean"] = comp
        (FDIR / "med_composite_weights.json").write_text(json.dumps(wts, indent=1))
    except Exception as e:
        print(f"  med composite skipped: {e}")
    return d


RATIO_PAIRS = {"qqq_spy": ("QQQ", "SPY"), "iwm_spy": ("IWM", "SPY"), "dia_spy": ("DIA", "SPY")}


def ratio_frame(a: str, b: str) -> pd.DataFrame:
    """Component-OHLC ratio candles (Amir's choice 2026-07-05, TV-style symbol division:
    o_A/o_B, h_A/h_B, l_A/l_B, c_A/c_B on shared dates). A ratio's intrabar extremes are
    approximations (the legs' highs aren't simultaneous) — re-ordered so high>=low holds."""
    fa, fb = datastore.load_bars(a), datastore.load_bars(b)
    if fa is None or fb is None:
        return pd.DataFrame()
    ix = fa.index.intersection(fb.index)
    fa, fb = fa.loc[ix], fb.loc[ix]
    r = pd.DataFrame({c: fa[c] / fb[c] for c in ("open", "high", "low", "close")}, index=ix)
    hi = r[["open", "high", "low", "close"]].max(axis=1)
    lo = r[["open", "high", "low", "close"]].min(axis=1)
    r["high"], r["low"] = hi, lo
    return r.dropna()


def ratio_block(cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Cross-index RATIO-CHART STRUCTURE (Amir 2026-07-05: 'these charts structures may
    also inform us of opportunity or caution'). Per QQQ/SPY, IWM/SPY, DIA/SPY — market-level,
    so the same block enters every sym's matrix; all shift(1) on the ratio's own calendar:
      a_rx_*_up / _down   trendlab structure class of the ratio chart
      a_rx_*_d50          close distance from the ratio's 50-SMA in ratio-ATR units
      a_rx_*_ret21        21-day ratio change % (the slower RS trend)

    STATUS (corrected 2026-07-15, Amir): a 2026-07-05 A/B on this exact block showed net -1.4pp
    and it was benched to a sidecar rather than the model. That result should NOT be read as
    "ratio-chart structure doesn't help" — it tested cash ETFs (QQQ/IWM/DIA vs SPY) through the
    OLD trendlab structure engine, before the DC/PIP engine, the two-ladder MTF realignment, and
    real futures data existed. A fair re-test needs: (a) futures legs for IWM/DIA the way ES/NQ
    already replaced SPY/QQQ's cash proxy -- RTY=F (Russell) and YM=F (Dow) are the natural picks,
    but the legacy Tiingo store for both is only ~2.5 months (same shallow-store problem ES/NQ
    had before their Databento pull -- verify-small-before-buying-history applies again here);
    (b) the finalized MTF/relationship ladder work (see docs/plan-mtf-timeframes-relationships.md,
    ST-6) so the structure read matches whatever the rest of the engine settles on. Until both
    exist, this block stays a sidecar (data/features/ratio_features.parquet) -- untested, not
    disproven."""
    import trendlab
    d = pd.DataFrame(index=cal)
    for tag, (a, b) in RATIO_PAIRS.items():
        r = ratio_frame(a, b)
        if len(r) < 80:
            continue
        st = pd.Series(trendlab.structure_series(r), index=r.index).shift(1).reindex(cal, method="ffill")
        d[f"a_rx_{tag}_up"] = st == trendlab.S_UP
        d[f"a_rx_{tag}_down"] = st == trendlab.S_DOWN
        atr = pd.Series(trendlab.atr_series(r), index=r.index)
        d50 = ((r["close"] - r["close"].rolling(50).mean()) / atr).shift(1)
        d[f"a_rx_{tag}_d50"] = d50.reindex(cal, method="ffill")
        d[f"a_rx_{tag}_ret21"] = (100 * r["close"].pct_change(21)).shift(1).reindex(cal, method="ffill")
    return d


def sr_block(cal: pd.DatetimeIndex) -> pd.DataFrame:
    """FUTURES higher-TF support/resistance proximity — RET-3's strongest confirmed finding
    (2026-07-15): when ES/NQ close the overnight session near a REAL confirmed 1D/1W swing pivot,
    the next session is more likely to trend (SPY +2.3pp, QQQ +3.5pp, both clearing their placebo,
    both agreeing on returns too — the only feature in RET-3 where trend-rate and returns agreed
    across both symbols).

    Config here MATCHES the validated one exactly: child 2h, parents 1D/1W, near = within 0.5 ATR
    (dcs_levels' own default). Deliberately NOT re-derived from the new ladder — 2h's ladder-A
    parents are 8h/2D/2W, which is a DIFFERENT test than the one that passed. Changing it would
    silently invalidate the finding this block exists to carry.

    Market-level (futures lead everything), so the same columns enter every symbol's matrix.
    Causal: read at the last 2h bar CLOSED before 09:30, pivots confirmed-only via parent_levels."""
    import dcs_levels
    import futures_mtf
    d = pd.DataFrame(index=cal)
    for fut in ("ES", "NQ"):
        if not futures_mtf.available(fut):
            continue
        m = futures_mtf.stack(fut, tf="2h", parents=("1D", "1W"))
        if m is None:
            continue
        rows = {}
        for sess_date, pos in futures_mtf.session_positions(m):
            atr_v = m.core.atr[pos]
            if not (atr_v == atr_v and atr_v > 0):
                continue
            px = float(m.child.df["close"].iloc[pos])
            levels = dcs_levels.parent_levels(m, pos)
            near = dcs_levels.proximity_tags(levels, px, atr_v)
            dist = min((abs(lv.price - px) / atr_v for lv in levels), default=np.nan)
            rows[sess_date] = (bool(near), dist)
        if not rows:
            continue
        s = pd.Series({k: v[0] for k, v in rows.items()})
        dist = pd.Series({k: v[1] for k, v in rows.items()})
        # values are already as-of the pre-open read for THAT session -> no extra shift needed
        d[f"a_sr_{fut.lower()}_near"] = s.reindex(cal).astype("boolean").fillna(False).astype(bool)
        d[f"a_sr_{fut.lower()}_dist"] = dist.reindex(cal)
    return d


def risk_block(cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Risk-on/off ROTATION postures from the app's health panel (config.RATIO_BASKET) — the
    genuinely-untested half of the health-panel gap (verified 2026-07-15).

    NOT the same as ratio_block: that one tests QQQ/SPY, IWM/SPY, DIA/SPY (index-vs-index) with a
    2026-07-05 A/B showing net -1.4pp on the OLD engine/data (see ratio_block's docstring — that
    result is unconfirmed pending a re-test, not settled). RATIO_BASKET's distinct content is the
    sector/commodity risk-appetite pairs — semis, discretionary-vs-staples, regional banks,
    copper/gold — which have never been in the day-type matrix at all, tested here for the first
    time (2026-07-15 A/B: no material lift on any target).

    Per pair: structure class of the ratio chart + distance from its 50-SMA in ratio-ATR units.
    All shift(1) on the ratio's own calendar (point-in-time: yesterday's close, known this morning)."""
    import trendlab
    d = pd.DataFrame(index=cal)
    for a, b, sign, label in config.RATIO_BASKET:
        tag = f"{a}_{b}".replace("=F", "").replace("-", "").lower()
        try:
            r = ratio_frame(a, b)
        except Exception:
            continue
        if r is None or len(r) < 80:
            continue
        st = pd.Series(trendlab.structure_series(r), index=r.index).shift(1).reindex(cal, method="ffill")
        # `sign` orients the pair so +1 always means RISK-ON (config.RATIO_BASKET's convention)
        d[f"a_rk_{tag}_riskon"] = (st == (trendlab.S_UP if sign > 0 else trendlab.S_DOWN))
        d[f"a_rk_{tag}_riskoff"] = (st == (trendlab.S_DOWN if sign > 0 else trendlab.S_UP))
        atr = pd.Series(trendlab.atr_series(r), index=r.index)
        d50 = ((r["close"] - r["close"].rolling(50).mean()) / atr).shift(1)
        d[f"a_rk_{tag}_d50"] = sign * d50.reindex(cal, method="ffill")
    if len(d.columns):                       # composite: how many pairs are risk-ON right now
        on = [c for c in d.columns if c.endswith("_riskon")]
        d["a_rk_breadth"] = d[on].sum(axis=1) / len(on)
    return d


def calendar_block(cal: pd.DatetimeIndex) -> pd.DataFrame:
    d = pd.DataFrame(index=cal)
    d["a_dow"] = cal.dayofweek
    third_fri = pd.Series([(x.weekday() == 4 and 15 <= x.day <= 21) for x in cal], index=cal)
    d["a_opex"] = third_fri
    d["a_quadwitch"] = third_fri & cal.month.isin([3, 6, 9, 12])
    d["a_month_end"] = pd.Series(cal, index=cal).groupby(cal.to_period("M")).transform("max") == cal
    d["a_quarter_end"] = d["a_month_end"] & cal.month.isin([3, 6, 9, 12])
    d["a_fomc"] = cal.isin(pd.to_datetime(FOMC))
    d["a_nfp"] = pd.Series([(x.weekday() == 4 and x.day <= 7) for x in cal], index=cal)
    return d


def _vwap_arr(g: pd.DataFrame) -> np.ndarray:
    tp = ((g["high"] + g["low"] + g["close"]) / 3).to_numpy(float)
    v = np.maximum(g["volume"].to_numpy(float), 1)
    return (tp * v).cumsum() / v.cumsum()


def spy_intraday_block(df: pd.DataFrame, sym: str = "SPY",
                       extra_bars: dict | None = None) -> pd.DataFrame:
    """Buckets B + C from the SYMBOL's own 5-min bars (RTH + ETH stores) — per session.
    extra_bars (H3 live stage): {SYM: today's 5-min frame, NY-naive index} appended to the
    store frames so TODAY's partial session gets the same features from the same code."""
    def _extend(fr, s, lo="09:30", hi="15:59"):
        """Replace-day semantics: any session present in the extra frame supersedes the
        store's rows for that date — live: appends today; sim: swaps in a truncated day."""
        x = (extra_bars or {}).get(s)
        if x is None or not len(x):
            return fr
        x = x.between_time(lo, hi)
        if not len(x):
            return fr
        days = set(pd.Index(x.index.date))
        if len(fr):
            fr = fr[~pd.Index(fr.index.date).isin(days)]
        return pd.concat([fr, x]).sort_index()
    rth = _extend(pd.read_parquet(STORE / f"{sym}.parquet").between_time("09:30", "15:59"), sym)
    eth_p = ETHSTORE / f"{sym}.parquet"
    eth = pd.read_parquet(eth_p) if eth_p.exists() else None
    if extra_bars is not None and extra_bars.get(sym) is not None:
        eth = _extend(eth if eth is not None else extra_bars[sym].iloc[0:0],
                      sym, "04:00", "09:24")
    other = "SPY" if sym != "SPY" else "QQQ"
    qqq = _extend(pd.read_parquet(STORE / f"{other}.parquet").between_time("09:30", "15:59"),
                  other)
    spdr = {s: _extend(pd.read_parquet(STORE / f"{s}.parquet").between_time("09:30", "15:59"), s)
            for s in SPDR}
    prev_c = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_c).abs(),
                    (df["low"] - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().shift(1)
    rows = []
    prior_va = None                    # (val, vah, poc) of the PRIOR session
    prior_rng = None
    by_day = {str(k): g for k, g in rth.groupby(rth.index.date)}
    eth_by_day = ({str(k): g for k, g in eth.groupby(eth.index.date)} if eth is not None else {})
    vol20: list[float] = []
    pre_rng_hist: list[float] = []
    for d in df.index:
        g = by_day.get(d)
        r: dict = {"date": d}
        # live stage: TODAY's partial session is allowed through with whatever bars exist
        min_bars = 2 if (extra_bars is not None and d == df.index[-1]) else 18
        if g is None or len(g) < min_bars:
            rows.append(r); continue
        a = atr.loc[d] if d in atr.index else np.nan
        op = float(g["open"].iloc[0]); pc = float(prev_c.loc[d]) if pd.notna(prev_c.loc[d]) else None
        # ── B: gap + open location ──
        if pc and pd.notna(a) and a > 0:
            r["b_gap_atr"] = (op - pc) / a
        if prior_rng:
            r["b_open_in_prior_range"] = bool(prior_rng[0] <= op <= prior_rng[1])
        if prior_va:
            r["b_open_in_va"] = bool(prior_va[0] <= op <= prior_va[1])
            r["b_open_vs_poc"] = (op - prior_va[2]) / a if pd.notna(a) and a else None
        # pre-market (08:00+ IEX / 04:00+ TV where covered)
        ge = eth_by_day.get(d)
        if ge is not None and pc:
            pre = ge.between_time("04:00", "09:24")
            if len(pre) >= 6:
                prng = float(pre["high"].max() - pre["low"].min())
                if pd.notna(a) and a > 0:
                    r["b_pre_range_atr"] = prng / a
                if pre_rng_hist:
                    r["b_pre_range_vs_avg"] = prng / (np.mean(pre_rng_hist[-20:]) or np.nan)
                pre_rng_hist.append(prng)
                side = float((pre["close"] > pc).mean())
                r["b_pre_onesided"] = max(side, 1 - side)
        cl2 = g["close"].iloc[:2]
        r["b_open_drive"] = float(abs(cl2.iloc[-1] - op) / (g["high"].iloc[:2].max()
                                  - g["low"].iloc[:2].min() or np.nan))
        # relative move vs the counterpart index at 30m / 11:30 (bps)
        # (added below via the qqq/other frame after the loop)
        # ── checkpoint features at 5m / 15m / 30m (bars 1 / 3 / 6) ──
        clA = g["close"].to_numpy(float)
        vwA = _vwap_arr(g)
        for tag, k in (("05", 1), ("15", 3), ("30", 6),
                       ("1130", 24), ("1230", 36), ("1330", 48), ("1430", 60)):
            if len(g) < k:          # clA[:k] needs exactly k bars — live sessions hit len==k
                continue
            w_hi = float(g["high"].iloc[:k].max()); w_lo = float(g["low"].iloc[:k].min())
            wr = w_hi - w_lo
            r[f"c{tag}_dir"] = float(np.sign(clA[k - 1] - op))
            r[f"c{tag}_rng_atr"] = (wr / a) if pd.notna(a) and a > 0 else None
            if wr > 0:
                r[f"c{tag}_cpos"] = (clA[k - 1] - w_lo) / wr
            pathk = np.abs(np.diff(np.concatenate([[op], clA[:k]]))).sum()
            r[f"c{tag}_eff"] = float(abs(clA[k - 1] - op) / pathk) if pathk else None
            if k >= 2:
                dk = clA[:k] - vwA[:k]
                r[f"c{tag}_vwap_side"] = float(max((dk > 0).mean(), (dk < 0).mean()))
            if pc and abs(op - pc) > 0.01:              # gap hold: how much of the gap survives
                r[f"c{tag}_gap_hold"] = float((clA[k - 1] - pc) / (op - pc))
        # ── C: first hour ──
        vw = _vwap_arr(g)
        cl = g["close"].to_numpy(float)
        h12 = min(12, len(g))
        diff = cl[:h12] - vw[:h12]
        r["c_vwap_crosses_1h"] = int((np.sign(diff[1:]) != np.sign(diff[:-1])).sum())
        r["c_vwap_retouched"] = bool((np.abs(diff[2:]) / cl[2:h12] < 0.0005).any())
        ib_hi = float(g["high"].iloc[:12].max()); ib_lo = float(g["low"].iloc[:12].min())
        if pd.notna(a) and a > 0:
            r["c_ib_range_atr"] = (ib_hi - ib_lo) / a
        brk = np.where((cl[12:] > ib_hi) | (cl[12:] < ib_lo))[0]
        r["c_ib_break_bar"] = int(brk[0] + 12) if brk.size else None
        eff_n = min(12, len(cl) - 1)
        path = np.abs(np.diff(cl[:eff_n + 1])).sum()
        r["c_eff_1h"] = float(abs(cl[eff_n] - op) / path) if path else None
        r["c_range_pos_1030"] = float((cl[h12 - 1] - g["low"].iloc[:h12].min())
                                      / ((g["high"].iloc[:h12].max() - g["low"].iloc[:h12].min()) or np.nan))
        v1h = float(g["volume"].iloc[:12].sum())
        if vol20:
            r["c_volrate_1h"] = v1h / (np.mean(vol20[-20:]) or np.nan)
        vol20.append(v1h)
        # SPY-vs-QQQ agreement at 10:25
        gq = qqq.loc[qqq.index.date.astype(str) == d] if False else None
        # (QQQ lookup via dict below)
        r["_spy_dir_1030"] = np.sign(cl[h12 - 1] - op)
        # value area for the NEXT session's b_ features
        piv = ((g["high"] + g["low"] + g["close"]) / 3).to_numpy(float)
        v = g["volume"].to_numpy(float)
        order = np.argsort(piv)
        pv, vv = piv[order], v[order]
        tot = vv.sum()
        if tot > 0:
            poc_i = int(np.argmax(vv))
            lo_i = hi_i = poc_i
            cap = vv[poc_i]
            while cap < 0.7 * tot and (lo_i > 0 or hi_i < len(vv) - 1):
                nxt_lo = vv[lo_i - 1] if lo_i > 0 else -1
                nxt_hi = vv[hi_i + 1] if hi_i < len(vv) - 1 else -1
                if nxt_hi >= nxt_lo:
                    hi_i += 1; cap += vv[hi_i]
                else:
                    lo_i -= 1; cap += vv[lo_i]
            prior_va = (float(pv[lo_i]), float(pv[hi_i]), float(pv[poc_i]))
        prior_rng = (float(g["low"].min()), float(g["high"].max()))
        rows.append(r)
    out = pd.DataFrame(rows).set_index("date")
    # QQQ direction agreement + sector uniformity at 10:25
    qdir = {}
    for k, gq in qqq.groupby(qqq.index.date):
        if len(gq) >= 12:
            qdir[str(k)] = np.sign(float(gq["close"].iloc[11] - gq["open"].iloc[0]))
    out["c_spy_qqq_agree"] = [
        (qdir.get(d) == s if (d in qdir and pd.notna(s)) else None)
        for d, s in zip(out.index, out["_spy_dir_1030"])]
    for tag, nb in (("1030", 12), ("1230", 36), ("1430", 60)):
        green = {}
        for s, fr in spdr.items():
            for k, gs in fr.groupby(fr.index.date):
                if len(gs) >= nb:
                    green.setdefault(str(k), []).append(
                        float(gs["close"].iloc[nb - 1]) > float(gs["open"].iloc[0]))
        col = "c_sectors_green_1030" if tag == "1030" else f"c{tag}_sectors_green"
        out[col] = [sum(green[d]) if d in green else None for d in out.index]
        if tag == "1030":
            out["c_sector_uniformity"] = [
                (max(sum(green[d]), 11 - sum(green[d])) if d in green and len(green[d]) == 11
                 else None) for d in out.index]
    return out.drop(columns=["_spy_dir_1030"])


def universe_1030_pass() -> None:
    """HEAVY: % of the active universe above its own session VWAP at 09:45 / 10:00 / 10:25."""
    CHK = {"0945": 3, "1000": 6, "1030": 12,
           "1130": 24, "1230": 36, "1330": 48, "1430": 60}
    cnt = {t: {} for t in CHK}
    tot = {t: {} for t in CHK}
    syms = sorted(uni.active_symbols() or [])
    t0 = time.time()
    for i, s in enumerate(syms):
        p = STORE / f"{s}.parquet"
        if uni.is_future(s) or not p.exists():
            continue
        b = pd.read_parquet(p).between_time("09:30", "15:59")
        for k, g in b.groupby(b.index.date):
            if len(g) < 12:
                continue
            d = str(k)
            vw = _vwap_arr(g)
            for tag, nb in CHK.items():
                if len(g) < nb:
                    continue
                tot[tag][d] = tot[tag].get(d, 0) + 1
                if float(g["close"].iloc[nb - 1]) > float(vw[nb - 1]):
                    cnt[tag][d] = cnt[tag].get(d, 0) + 1
        if i % 100 == 0:
            print(f"[{i}/{len(syms)}] {s} ({time.time()-t0:.0f}s)", flush=True)
    out = pd.DataFrame({f"c{'' if tag=='1030' else tag}_pct_above_vwap_{tag}"
                        if tag != "1030" else "c_pct_above_vwap_1030":
                        100 * pd.Series(cnt[tag]) / pd.Series(tot[tag]) for tag in CHK})
    out.index.name = "date"
    FDIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(UNIV_1030)
    print(f"universe checkpoint pass: {len(out)} days -> {UNIV_1030} ({time.time()-t0:.0f}s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", action="store_true", help="run only the heavy universe pass")
    ap.add_argument("--sym", default="SPY")
    args = ap.parse_args()
    if args.universe:
        universe_1030_pass(); return

    df = daytype.classify_all(pd.read_parquet(STORE / f"{args.sym}.parquet"))
    cal = _sess(df.index)
    print(f"{len(df)} sessions")
    blocks = [spy_daily_block(df)]
    print("universe daily pass…")
    ub = universe_daily_block(cal)
    ub_shift = ub.shift(1).add_prefix("a_")            # EOD readings -> known next morning
    blocks.append(ub_shift.set_axis(df.index))
    dd = pd.DataFrame({"a_dd_spy": dist_days("SPY", cal).shift(1).values,
                       "a_dd_qqq": dist_days("QQQ", cal).shift(1).values}, index=df.index)
    blocks.append(dd)
    blocks.append(vol_block(cal).set_axis(df.index))
    blocks.append(external_block(cal, args.sym).set_axis(df.index))
    blocks.append(med_block(cal).set_axis(df.index))
    if args.sym == "SPY":
        # ratio-structure SIDECAR (outlook/lab only, kept out of the day-type model — the
        # 2026-07-05 A/B (-1.4pp) used the old engine/cash-ETF legs and needs a proper re-test,
        # not settled either way; see ratio_block's docstring. Era-robust at the 1-MONTH horizon.)
        rb = ratio_block(cal).set_axis(df.index)
        for c in list(rb.columns):
            if rb[c].dtype != bool:
                sr = rb[c]; rk = sr.expanding().rank(); nn = sr.expanding().count()
                rb[c + "_pct"] = (100 * (rk - 1) / (nn - 1).replace(0, np.nan)).where(sr.notna())
        rb.to_parquet(FDIR / "ratio_features.parquet")
    print("futures S/R block (RET-3 confirmed finding)…")
    blocks.append(sr_block(cal).set_axis(df.index))
    print("risk-on rotation block (health-panel basket)…")
    blocks.append(risk_block(cal).set_axis(df.index))
    blocks.append(calendar_block(cal).set_axis(df.index))
    # realized-vs-implied: yesterday's realized range % vs yesterday's VIX-implied daily move
    vix = datastore.load_bars("^VIX")["close"].reindex(cal, method="ffill")
    rv = (df["range"] / df["close"]).values * 100
    ivd = (vix / np.sqrt(252)).values
    blocks.append(pd.DataFrame({"a_rv_vs_iv": (pd.Series(rv, index=df.index)
                                               / pd.Series(ivd, index=df.index)).shift(1)}))
    print(f"{args.sym} intraday pass…")
    blocks.append(spy_intraday_block(df, args.sym))
    if UNIV_1030.exists():
        u = pd.read_parquet(UNIV_1030)
        blocks.append(u.reindex(df.index))
        print("universe 10:30 metrics attached")
    else:
        print("NOTE: run --universe for c_pct_above_vwap_1030")
    F = pd.concat(blocks, axis=1)
    # expanding past-percentiles for every numeric feature
    num = F.select_dtypes(include=[np.number]).columns
    for c in num:
        s = F[c]
        r = s.expanding().rank()
        n = s.expanding().count()
        F[c + "_pct"] = (100 * (r - 1) / (n - 1).replace(0, np.nan)).where(s.notna())
    FDIR.mkdir(parents=True, exist_ok=True)
    out_p = OUT if args.sym == "SPY" else FDIR / f"daytype_features_{args.sym}.parquet"
    F.to_parquet(out_p)
    print(f"{len(F)} rows x {len(F.columns)} cols -> {out_p}")


if __name__ == "__main__":
    main()
