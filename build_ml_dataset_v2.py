"""
Stage 1: Dataset construction -- daily at-risk panel for the Episodic-Pivot (EP) outcome model.

PURPOSE
    Given an EP that has ALREADY been detected by deterministic code (setups.iter_fresh_ep_events),
    describe it every day as it develops and let a model estimate, from everything known as of that
    day, the probability that the stock goes on to make a +50% / +100% / +150% / +200% move
    (measured from the EP-day close) BEFORE it loses the EP-day low.  The model never detects EPs.

ROW  = one (EP event, as-of trading day).  A row exists while the event is still "alive":
       from age 1 (EP day) until the bar BEFORE the EP-day low is breached, the 250-session horizon
       ends, or the data ends.  Reaching +50% does NOT end the panel (higher thresholds still open).

FEATURE FAMILIES (all strictly as-of; columns prefixed `feature_`)
    event_*     : the EP-day snapshot (gap, rvol, close position, ATR%, dollar volume, subtype,
                  pre-event run-up, distance from 52w high, context ranks on the EP day).
    chk_*       : how the trade has developed through the as-of day (return since EP, MFE/MAE,
                  distance to stop, held/broke EP-day high, which thresholds are already reached).
    chart       : EMA slopes + stacking, distance to ema20/sma50/sma200, Larsson Line state/age/spread,
                  52w/ATH distance, volume (rvol) and its average since EP.
    resistance  : overhead confirmed pivot highs (nearest %, count within +25%, blue-sky flag).
    market      : SPY returns/trend regime; stock relative strength vs SPY.
    sector      : industry basket return (peer mean excl. self), basket RS vs SPY, stock RS vs basket,
                  stock percentile rank inside its industry.
    theme       : same, for the strongest theme the stock belongs to on that date.

LABELS (`label_*`, never predictors)
    label_reach_{50,100,150,200}      1 = threshold reached before stop, 0 = failed (stop / same-bar
                                      ambiguity / full-horizon timeout), NaN = right-censored.
    label_reach_{T}_state             reached | failed_stop | ambiguous_same_bar | timeout | censored
    label_outcome_state               +50% state under the legacy names (target/stop/...).
    NOTE: labels are EVENT-level (constant across an event's rows).  For threshold T, Stage 2 must
    train only on rows where feature_chk_reached_T == 0 (T not yet achieved as of that day).

Provenance (hashes of all dependent code/data, bar inventory) is written next to the parquet.
"""

import pandas as pd
import numpy as np
import os
import hashlib
import json
import sys
import tempfile
import shutil
import pyarrow
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import datastore
import setups
import indicators
import config
import universe
import labels
import trendlab
import scanner_core

DATA_DIR = config.DATA_DIR / "ml_datasets"
os.makedirs(DATA_DIR, exist_ok=True)

HORIZON_SESSIONS = 250
THRESHOLDS = {50: 1.50, 100: 2.00, 150: 2.50, 200: 3.00}   # multiple of EP-day close
PIVOT_LOOKBACK = 250        # sessions of confirmed pivot highs considered resistance
MIN_PEERS = 5               # minimum peers for a basket/rank statistic to be defined
LARSSON_CODE = {"yellow": 1.0, "gray": 0.0, "blue": -1.0}


def get_file_hash(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing required file for provenance: {filepath}")
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def frame_fingerprint(d: pd.DataFrame) -> str:
    """Deterministic SHA-256 of a raw OHLCV frame (index + values)."""
    return hashlib.sha256(pd.util.hash_pandas_object(d, index=True).values.tobytes()).hexdigest()


def inventory_digest(inventory: list[dict]) -> str:
    """Order-independent digest of the bar-input inventory."""
    canon = json.dumps(sorted(inventory, key=lambda r: r["symbol"]), sort_keys=True)
    return hashlib.sha256(canon.encode()).hexdigest()


# --------------------------------------------------------------------------------------
# Market / sector / theme context
# --------------------------------------------------------------------------------------
class ContextBuilder:
    """Cross-sectional context from the loaded equity universe + SPY.

    Everything is a function of data up to each date only (returns over trailing windows, trailing
    MAs).  Basket statistics exclude the stock itself.  Membership is the CURRENT sector/theme label
    snapshot (labels.py) -- a documented limitation (no point-in-time membership available)."""

    def __init__(self, close_panel: pd.DataFrame, spy: pd.DataFrame):
        self.close = close_panel
        self.r1 = (close_panel / close_panel.shift(21) - 1.0) * 100.0
        self.r3 = (close_panel / close_panel.shift(63) - 1.0) * 100.0
        self.spy = self._spy_frame(spy)
        cols = set(close_panel.columns)
        self.industry_members: dict[str, list[str]] = {}
        self.theme_members: dict[str, list[str]] = {}
        for s in close_panel.columns:
            ind = labels.sector(s)
            if ind:
                self.industry_members.setdefault(ind, []).append(s)
        for th in labels.all_themes():
            m = [s for s in labels.tickers_in_theme(th) if s in cols]
            if len(m) > MIN_PEERS:
                self.theme_members[th] = m
        self._sn: dict[tuple, tuple] = {}

    @staticmethod
    def _spy_frame(spy: pd.DataFrame) -> pd.DataFrame:
        c = spy["close"]
        out = pd.DataFrame(index=spy.index)
        out["spy_ret_1m"] = (c / c.shift(21) - 1) * 100
        out["spy_ret_3m"] = (c / c.shift(63) - 1) * 100
        e10 = c.ewm(span=10, adjust=False).mean()
        e20 = c.ewm(span=20, adjust=False).mean()
        s50 = c.rolling(50).mean()
        s200 = c.rolling(200).mean()
        out["spy_above_sma50"] = (c > s50).astype(float).where(s50.notna())
        out["spy_above_sma200"] = (c > s200).astype(float).where(s200.notna())
        out["spy_stacked"] = ((c > e10) & (e10 > e20) & (e20 > s50) & (s50 > s200)).astype(float).where(s200.notna())
        return out

    def _group_sn(self, key: tuple, members: list[str]):
        if key not in self._sn:
            self._sn[key] = (
                self.r1[members].sum(axis=1), self.r1[members].notna().sum(axis=1),
                self.r3[members].sum(axis=1), self.r3[members].notna().sum(axis=1),
            )
        return self._sn[key]

    def _basket(self, key, members, sym):
        """Peer-mean (excl. sym) 1m/3m return and sym's percentile rank (3m) among peers."""
        S1, N1, S3, N3 = self._group_sn(key, members)
        o1, o3 = self.r1[sym], self.r3[sym]
        n1 = N1 - o1.notna().astype(int)
        n3 = N3 - o3.notna().astype(int)
        m1 = ((S1 - o1.fillna(0)) / n1.where(n1 >= MIN_PEERS))
        m3 = ((S3 - o3.fillna(0)) / n3.where(n3 >= MIN_PEERS))
        peers = self.r3[[c for c in members if c != sym]]
        cnt = peers.notna().sum(axis=1)
        rank = peers.lt(o3, axis=0).sum(axis=1) / cnt.where(cnt >= MIN_PEERS)
        rank = rank.where(o3.notna())
        return m1, m3, rank

    def for_symbol(self, sym: str, index: pd.Index) -> pd.DataFrame:
        """Context columns aligned to `index` (NaN where undefined)."""
        out = self.spy.reindex(index).copy()
        if sym not in self.close.columns:
            return out
        o1, o3 = self.r1[sym], self.r3[sym]
        spy1 = self.spy["spy_ret_1m"].reindex(self.close.index)
        spy3 = self.spy["spy_ret_3m"].reindex(self.close.index)
        ctx = pd.DataFrame(index=self.close.index)
        ctx["tk_rs_spy_1m"] = o1 - spy1
        ctx["tk_rs_spy_3m"] = o3 - spy3

        ind = labels.sector(sym)
        mem = self.industry_members.get(ind, []) if ind else []
        if len(mem) > MIN_PEERS:
            m1, m3, rk = self._basket(("ind", ind), mem, sym)
            ctx["ind_ret_1m"], ctx["ind_ret_3m"] = m1, m3
            ctx["ind_rs_spy_3m"] = m3 - spy3
            ctx["ind_rank_3m"] = rk
            ctx["tk_rs_ind_1m"] = o1 - m1
            ctx["tk_rs_ind_3m"] = o3 - m3
            ctx["ind_peer_count"] = len(mem) - 1
        else:
            for c in ("ind_ret_1m", "ind_ret_3m", "ind_rs_spy_3m", "ind_rank_3m", "tk_rs_ind_1m", "tk_rs_ind_3m", "ind_peer_count"):
                ctx[c] = np.nan

        th_list = [t for t in labels.themes(sym) if t in self.theme_members]
        if th_list:
            M1, M3, RK = [], [], []
            for th in th_list:
                m1, m3, rk = self._basket(("th", th), self.theme_members[th], sym)
                M1.append(m1.to_numpy()); M3.append(m3.to_numpy()); RK.append(rk.to_numpy())
            M1, M3, RK = np.array(M1), np.array(M3), np.array(RK)
            filled = np.where(np.isnan(M3), -np.inf, M3)
            best = filled.argmax(axis=0)
            allnan = np.isinf(filled.max(axis=0))
            cols = np.arange(M3.shape[1])
            pick = lambda A: np.where(allnan, np.nan, A[best, cols])
            ctx["theme_ret_1m"] = pick(M1)
            ctx["theme_ret_3m"] = pick(M3)
            ctx["theme_rs_spy_3m"] = ctx["theme_ret_3m"] - spy3
            ctx["theme_rank_3m"] = pick(RK)
            ctx["tk_rs_theme_3m"] = o3 - ctx["theme_ret_3m"]
            
            th_names = np.array(th_list)
            ctx["strongest_theme"] = np.where(allnan, "", th_names[best])
            peer_counts = np.array([len(self.theme_members[t]) - 1 for t in th_list])
            ctx["theme_peer_count"] = np.where(allnan, np.nan, peer_counts[best])
        else:
            for c in ("theme_ret_1m", "theme_ret_3m", "theme_rs_spy_3m", "theme_rank_3m", "tk_rs_theme_3m", "strongest_theme", "theme_peer_count"):
                ctx[c] = np.nan if c != "strongest_theme" else ""
        return pd.concat([out, ctx.reindex(index)], axis=1)


# --------------------------------------------------------------------------------------
# Per-symbol causal chart features (independent of any particular event)
# --------------------------------------------------------------------------------------
def compute_symbol_features(d: pd.DataFrame, ctx: pd.DataFrame | None = None) -> pd.DataFrame:
    """As-of features for every bar of `d`.  Every value at bar t uses data <= t only
    (trailing EMAs/SMAs/rolling windows; confirmed pivots filtered by confirmation bar)."""
    c, h, l = d["close"], d["high"], d["low"]
    F = pd.DataFrame(index=d.index)
    e10 = c.ewm(span=10, adjust=False).mean()
    e20 = c.ewm(span=20, adjust=False).mean()
    s50 = c.rolling(50).mean()
    s200 = c.rolling(200).mean()
    F["dist_ema20_pct"] = (c / e20 - 1) * 100
    F["dist_sma50_pct"] = (c / s50 - 1) * 100
    F["dist_sma200_pct"] = (c / s200 - 1) * 100
    F["ema10_slope_5d_pct"] = (e10 / e10.shift(5) - 1) * 100
    F["ema20_slope_5d_pct"] = (e20 / e20.shift(5) - 1) * 100
    F["sma50_slope_10d_pct"] = (s50 / s50.shift(10) - 1) * 100
    st = ((c > e10).astype(float) + (e10 > e20) + (e20 > s50) + (s50 > s200))
    F["ema_stack_count"] = st.where(s200.notna())
    F["ema_stacked"] = (F["ema_stack_count"] == 4).astype(float).where(s200.notna())
    F["ret_1m_pct"] = (c / c.shift(21) - 1) * 100
    F["ret_3m_pct"] = (c / c.shift(63) - 1) * 100
    F["dist_52w_high_pct"] = (c / c.rolling(252, min_periods=60).max() - 1) * 100
    F["dist_ath_pct"] = (c / c.expanding().max() - 1) * 100

    rng = (h - l).where((h - l) > 0)
    F["close_pos"] = (c - l) / rng
    atr = d["atr14"] if "atr14" in d.columns else pd.Series(np.nan, index=d.index)
    F["atr_pct"] = atr / c * 100
    rvol = d["rvol"] if "rvol" in d.columns else pd.Series(np.nan, index=d.index)
    F["rvol"] = rvol
    F["chg_pct"] = (c / c.shift(1) - 1) * 100

    # Larsson Line (daily): state, age in state, spread, distance to fastest EMA
    L = scanner_core.calc_larssson_line(d[["open", "high", "low", "close", "volume"]])
    code = L["larsson_state"].map(LARSSON_CODE).astype(float)
    run = (code != code.shift()).cumsum()
    F["larsson_state"] = code
    F["larsson_state_age"] = (code.groupby(run).cumcount() + 1).astype(float).where(code.notna())
    F["larsson_spread_pct"] = (L["larsson_ema32"] / L["larsson_ema58"] - 1) * 100
    F["larsson_dist_ema32_pct"] = (c / L["larsson_ema32"] - 1) * 100

    # Overhead resistance from CONFIRMED pivot highs (trendlab single pivot source; usable only
    # from its confirmation bar `pconf`)
    n = len(d)
    r = trendlab._compute(d)
    hp = [(r["pidx"][k], r["pconf"][k], r["pprice"][k]) for k in range(len(r["pidx"]))
          if r["ptype"][k] == trendlab.PIVOT_HIGH]
    near = np.full(n, np.nan); cnt25 = np.full(n, np.nan); blue = np.full(n, np.nan); dlast = np.full(n, np.nan)
    if hp:
        pi = np.array([p[0] for p in hp]); pc = np.array([p[1] for p in hp]); pp = np.array([p[2] for p in hp], float)
        cv = c.to_numpy(float)
        for t in range(n):
            m = (pc <= t) & (pi >= t - PIVOT_LOOKBACK)
            if not m.any():
                continue
            prices = pp[m]
            over = prices[prices > cv[t]]
            blue[t] = 0.0 if over.size else 1.0
            if over.size:
                near[t] = (over.min() / cv[t] - 1) * 100
            cnt25[t] = float(((prices > cv[t]) & (prices <= cv[t] * 1.25)).sum())
            last = pp[m][np.argmax(pc[m])]
            dlast[t] = (cv[t] / last - 1) * 100
    F["overhead_nearest_pivot_pct"] = near
    F["overhead_pivots_within_25pct"] = cnt25
    F["blue_sky_no_overhead_pivot"] = blue
    F["dist_last_pivot_high_pct"] = dlast

    if ctx is not None:
        for col in ctx.columns:
            F["ctx_" + col] = ctx[col].reindex(d.index).to_numpy()
    else:
        for col in ("spy_ret_1m", "spy_ret_3m", "spy_above_sma50", "spy_above_sma200", "spy_stacked",
                    "tk_rs_spy_1m", "tk_rs_spy_3m", "ind_ret_1m", "ind_ret_3m", "ind_rs_spy_3m",
                    "ind_rank_3m", "tk_rs_ind_1m", "tk_rs_ind_3m", "theme_ret_1m", "theme_ret_3m",
                    "theme_rs_spy_3m", "theme_rank_3m", "tk_rs_theme_3m"):
            F["ctx_" + col] = np.nan
    return F


# --------------------------------------------------------------------------------------
# Pure per-event panel
# --------------------------------------------------------------------------------------
def _first_idx(mask: np.ndarray):
    w = np.flatnonzero(mask)
    return int(w[0]) if w.size else None


def build_event_panel(sym: str, d: pd.DataFrame, ep_idx: int, subtype: str,
                      sym_feat: pd.DataFrame | None = None,
                      static: dict | None = None) -> list[dict]:
    """Daily at-risk panel for ONE already-detected EP.  Pure (no I/O).

    d        : daily OHLCV(+indicators) frame; ep_idx = position of the EP day.
    sym_feat : output of compute_symbol_features(d, ctx) (computed here if omitted).
    static   : event-independent categorical info (sector name, themes, ...)."""
    if sym_feat is None:
        sym_feat = compute_symbol_features(d)
    static = static or {}

    hi = d["high"].to_numpy(float)
    lo = d["low"].to_numpy(float)
    cl = d["close"].to_numpy(float)
    n = len(d)

    d1_low, d1_high, d1_close = lo[ep_idx], hi[ep_idx], cl[ep_idx]
    d1_prev_close = cl[ep_idx - 1]
    d1_open = float(d["open"].iloc[ep_idx])
    event_date = d.index[ep_idx]
    event_rng = d1_high - d1_low
    targets = {t: d1_close * m for t, m in THRESHOLDS.items()}

    # ---- forward window (bars after the EP day, up to the horizon) --------------------
    f_lo = lo[ep_idx + 1: ep_idx + 1 + HORIZON_SESSIONS]
    f_hi = hi[ep_idx + 1: ep_idx + 1 + HORIZON_SESSIONS]
    nf = len(f_lo)
    s = _first_idx(f_lo <= d1_low)              # relative index of the stop bar
    horizon_complete = nf == HORIZON_SESSIONS

    # ---- labels per threshold (first passage before stop) -----------------------------
    lab = {}
    for t, price in targets.items():
        j = _first_idx(f_hi >= price)
        if j is not None and (s is None or j < s):
            state, val = "reached", 1.0
            end_dt = d.index[ep_idx + 1 + j]
        elif j is not None and s == j:
            state, val = "ambiguous_same_bar", 0.0
            end_dt = d.index[ep_idx + 1 + s]
        elif s is not None:
            state, val = "failed_stop", 0.0
            end_dt = d.index[ep_idx + 1 + s]
        elif horizon_complete:
            state, val = "timeout", 0.0
            end_dt = d.index[ep_idx + HORIZON_SESSIONS]
        else:
            state, val = "censored", np.nan
            end_dt = None
        lab[t] = (state, val, end_dt)

    legacy = {"reached": "target", "failed_stop": "stop", "ambiguous_same_bar": "ambiguous_same_bar",
              "timeout": "full_horizon_timeout", "censored": "right_censored"}
    outcome_state = legacy[lab[50][0]]

    # ---- panel extent: alive until the bar BEFORE the stop bar / horizon / data end ----
    end_pos = min(n, ep_idx + HORIZON_SESSIONS)
    if s is not None:
        end_pos = min(end_pos, ep_idx + 1 + s)
    n_rows = end_pos - ep_idx
    if n_rows <= 0:
        return []
    pos = np.arange(ep_idx, end_pos)
    age = np.arange(1, n_rows + 1)

    if s is not None:
        resolution = "stop"
        audit_event_to_res = s + 1
        label_end_date = d.index[ep_idx + 1 + s]
    elif horizon_complete:
        resolution = "timeout"
        audit_event_to_res = HORIZON_SESSIONS
        label_end_date = d.index[ep_idx + HORIZON_SESSIONS]
    else:
        resolution = "censored"
        audit_event_to_res = None
        label_end_date = None
    audit_chk_to_res = (audit_event_to_res - (age - 1)) if audit_event_to_res is not None else [None] * n_rows
    observation_end_date = d.index[ep_idx + nf] if nf > 0 else event_date

    res_hi = f_hi[: (s + 1) if s is not None else nf]
    res_lo = f_lo[: (s + 1) if s is not None else nf]
    audit_mfe = (res_hi.max() / d1_close - 1) * 100 if res_hi.size else np.nan
    audit_mae = (res_lo.min() / d1_close - 1) * 100 if res_lo.size else np.nan

    # ---- as-of developing-trade features (forward bars only, cumulative) ---------------
    fwd_hi = hi[ep_idx + 1: end_pos]
    fwd_lo = lo[ep_idx + 1: end_pos]
    fwd_cl = cl[ep_idx + 1: end_pos]
    cm_hi = np.concatenate([[np.nan], np.maximum.accumulate(fwd_hi)]) if fwd_hi.size else np.array([np.nan])
    cm_lo = np.concatenate([[np.nan], np.minimum.accumulate(fwd_lo)]) if fwd_lo.size else np.array([np.nan])
    cm_cl = np.concatenate([[np.nan], np.maximum.accumulate(fwd_cl)]) if fwd_cl.size else np.array([np.nan])
    first = np.arange(n_rows) == 0
    mfe = np.where(first, 0.0, (cm_hi / d1_close - 1) * 100)
    mae = np.where(first, 0.0, (cm_lo / d1_close - 1) * 100)
    cl_now = cl[pos]
    rng_now = hi[pos] - lo[pos]

    cols = {
        "identifier_event_id": f"{sym}_{event_date.date()}_{subtype}",
        "identifier_symbol": sym,
        "identifier_event_date": str(event_date.date()),
        "identifier_as_of_date": d.index[pos].strftime("%Y-%m-%d"),
        "feature_age_sessions": age,
        "feature_event_subtype": subtype,

        # --- EP-day snapshot (static per event)
        "feature_event_gap_pct": (d1_open / d1_prev_close - 1) * 100,
        "feature_event_chg_pct": (d1_close / d1_prev_close - 1) * 100,
        "feature_event_rvol": sym_feat["rvol"].iloc[ep_idx],
        "feature_event_close_pos": sym_feat["close_pos"].iloc[ep_idx],
        "feature_event_atr_pct": sym_feat["atr_pct"].iloc[ep_idx],
        "feature_event_dollar_vol_log10": float(np.log10(max(d1_close * float(d["volume"].iloc[ep_idx]), 1.0))),
        "feature_event_ret_3m_pre_pct": sym_feat["ret_3m_pct"].iloc[ep_idx - 1] if ep_idx >= 1 else np.nan,
        "feature_event_dist_52w_high_pct": sym_feat["dist_52w_high_pct"].iloc[ep_idx],
        "feature_event_larsson_state": sym_feat["larsson_state"].iloc[ep_idx],
        "feature_event_ema_stack_count": sym_feat["ema_stack_count"].iloc[ep_idx],
        "feature_event_ind_rank_3m": sym_feat["ctx_ind_rank_3m"].iloc[ep_idx],
        "feature_event_tk_rs_spy_3m": sym_feat["ctx_tk_rs_spy_3m"].iloc[ep_idx],
        "feature_sector_name": static.get("sector", "Unknown"),
        "feature_spdr_sector": static.get("spdr", "Unknown"),
        "feature_themes": static.get("themes", ""),
        "feature_n_themes": static.get("n_themes", 0),

        # --- developing trade
        "feature_chk_return_since_event_pct": (cl_now / d1_close - 1) * 100,
        "feature_chk_dist_to_d1_low_pct": (cl_now / d1_low - 1) * 100,
        "feature_chk_observed_mfe_pct": mfe,
        "feature_chk_observed_mae_pct": mae,
        "feature_chk_broke_d1_high": np.where(first, 0, cm_hi > d1_high).astype(int),
        "feature_chk_closed_above_d1_high": np.where(first, 0, cm_cl > d1_high).astype(int),
        "feature_chk_range_contraction_vs_event": (rng_now / event_rng) if event_rng > 0 else np.nan,
    }
    for t, price in targets.items():
        cols[f"feature_chk_dist_to_target_{t}_pct"] = (price / cl_now - 1) * 100
        cols[f"feature_chk_reached_{t}"] = np.where(first, 0, cm_hi >= price).astype(int)

    # --- as-of chart / resistance / market / sector / theme features
    sf = sym_feat.iloc[ep_idx:end_pos]
    for name in sf.columns:
        if name in ("ret_3m_pct",):
            pass
        cols["feature_chk_" + name] = sf[name].to_numpy()
    rv = sf["rvol"].to_numpy(float)
    ev_rv = rv.copy(); ev_rv[0] = np.nan
    with np.errstate(all="ignore"):
        cs = np.nancumsum(np.nan_to_num(ev_rv)); cn = np.cumsum(~np.isnan(ev_rv))
        cols["feature_chk_rvol_mean_since_event"] = np.where(cn > 0, cs / np.maximum(cn, 1), np.nan)

    # --- labels / censoring / audit
    for t in THRESHOLDS:
        cols[f"label_reach_{t}"] = lab[t][1]
        cols[f"label_reach_{t}_state"] = lab[t][0]
        cols[f"label_reach_{t}_end_date"] = str(lab[t][2].date()) if lab[t][2] is not None else None
    cols["label_outcome_state"] = outcome_state
    cols["label_resolution"] = resolution
    cols["label_end_date"] = str(label_end_date.date()) if label_end_date is not None else None
    cols["censor_observation_end_date"] = str(observation_end_date.date())
    cols["censor_available_forward_sessions"] = nf
    cols["audit_entry_price"] = d1_close
    cols["audit_stop_price"] = d1_low
    cols["audit_target_price"] = targets[50]
    cols["audit_mfe_pct"] = audit_mfe
    cols["audit_mae_pct"] = audit_mae
    cols["audit_sessions_from_event_to_resolution"] = audit_event_to_res
    cols["audit_sessions_from_checkpoint_to_resolution"] = audit_chk_to_res

    df = pd.DataFrame(cols)
    return df.to_dict("records")


# --------------------------------------------------------------------------------------
# Build driver
# --------------------------------------------------------------------------------------
def build_dataset():
    print("INITIALIZING STAGE 1: CONTEXT-AWARE DAILY AT-RISK PANEL")

    symbols_raw = universe.active_symbols()
    if not symbols_raw:
        print("[!] FATAL: No active universe found. Explicitly failing rather than falling back.")
        sys.exit(1)
    symbols = sorted(s for s in symbols_raw
                     if s not in universe.CURATED_ETFS and s not in universe.FUTURES_SYMBOLS)
    print(f"[*] {len(symbols)} equity symbols requested")

    spy = datastore.load_bars("SPY")
    if spy is None:
        print("[!] FATAL: SPY bars missing; market context unavailable.")
        sys.exit(1)

    # ---- pass 1: inventory + close panel (for cross-sectional sector/theme context) ----
    inventory, closes = [], {}
    cached_bars = {}

    inventory.append({
        "symbol": "SPY", "status": "loaded", "rows": int(len(spy)),
        "min_date": str(spy.index.min().date()), "max_date": str(spy.index.max().date()),
        "ohlcv_sha256": frame_fingerprint(spy)
    })
    max_bar_date = None
    stats = {"symbols_requested": len(symbols), "symbols_missing": 0, "events_found": 0,
             "events_no_rows": 0, "rows_emitted": 0,
             "outcomes_by_threshold": {str(t): {} for t in THRESHOLDS}}
    for i, sym in enumerate(symbols):
        if i and i % 1000 == 0:
            print(f"  pass1 {i}/{len(symbols)}")
        raw = datastore.load_bars(sym)
        if raw is None or len(raw) < 100:
            stats["symbols_missing"] += 1
            inventory.append({"symbol": sym, "status": "missing"})
            continue
        mx = raw.index.max()
        max_bar_date = mx if max_bar_date is None or mx > max_bar_date else max_bar_date
        inventory.append({"symbol": sym, "status": "loaded", "rows": int(len(raw)),
                          "min_date": str(raw.index.min().date()), "max_date": str(mx.date()),
                          "ohlcv_sha256": frame_fingerprint(raw)})
        closes[sym] = raw["close"].astype("float32")
        cached_bars[sym] = raw
    print("[*] Building cross-sectional context (sector/theme baskets)...")
    ctxb = ContextBuilder(pd.DataFrame(closes), spy)

    # ---- pass 2: events -> panels ------------------------------------------------------
    all_records = []
    loaded = [r["symbol"] for r in inventory if r["status"] == "loaded" and r["symbol"] != "SPY"]
    for i, sym in enumerate(loaded):
        if i and i % 500 == 0:
            print(f"  pass2 {i}/{len(loaded)}  rows={stats['rows_emitted']}")
        d = indicators.add_indicators(cached_bars[sym].copy())
        ep_events = setups.iter_fresh_ep_events(d)
        stats["events_found"] += len(ep_events)
        if not ep_events:
            continue
        sym_feat = compute_symbol_features(d, ctxb.for_symbol(sym, d.index))
        sec = labels.sector(sym) or "Unknown"
        ths = labels.themes(sym) or []
        static = {"sector": sec, "spdr": labels.spdr_of(sec) or "Unknown",
                  "themes": "|".join(ths), "n_themes": len(ths)}
        for ep_idx, subtype in ep_events:
            panel = build_event_panel(sym, d, ep_idx, subtype, sym_feat=sym_feat, static=static)
            if not panel:
                stats["events_no_rows"] += 1
                continue
            for t in THRESHOLDS:
                k = panel[0][f"label_reach_{t}_state"]
                stats["outcomes_by_threshold"][str(t)][k] = stats["outcomes_by_threshold"][str(t)].get(k, 0) + 1
            stats["rows_emitted"] += len(panel)
            all_records.extend(panel)

    df_out = pd.DataFrame(all_records)
    cols = list(df_out.columns)
    roles = {
        "identifiers": [c for c in cols if c.startswith("identifier_")],
        "features": [c for c in cols if c.startswith("feature_")],
        "labels": [c for c in cols if c.startswith("label_")],
        "censors": [c for c in cols if c.startswith("censor_")],
        "audit": [c for c in cols if c.startswith("audit_")],
        "categorical_features": ["feature_event_subtype", "feature_sector_name",
                                 "feature_spdr_sector", "feature_themes"],
    }

    tmp_dir = tempfile.mkdtemp(dir=DATA_DIR)
    try:
        tmp_parquet = os.path.join(tmp_dir, "dataset.parquet")
        tmp_json = os.path.join(tmp_dir, "manifest.json")
        tmp_bar_inputs = os.path.join(tmp_dir, "bar_inputs.json")
        df_out.to_parquet(tmp_parquet, index=False)
        with open(tmp_bar_inputs, "w") as f:
            json.dump(inventory, f, indent=2)

        file_hash = get_file_hash(tmp_parquet)
        dataset_id = f"ep_dataset_daily_v4_{file_hash[:12]}"
        final_dir = DATA_DIR / dataset_id
        if final_dir.exists():
            print(f"[!] FATAL: {final_dir} already exists.")
            shutil.rmtree(tmp_dir)
            sys.exit(1)

        manifest = {
            "dataset_id": dataset_id,
            "dataset_schema_version": "v4.0_context_daily_panel",
            "sha256": file_hash,
            "provenance": {
                "script_sha256": get_file_hash(__file__),
                "config_sha256": get_file_hash(config.__file__),
                "setups_sha256": get_file_hash(setups.__file__),
                "indicators_sha256": get_file_hash(indicators.__file__),
                "datastore_sha256": get_file_hash(datastore.__file__),
                "universe_sha256": get_file_hash(universe.__file__),
                "labels_py_sha256": get_file_hash(labels.__file__),
                "sectors_csv_sha256": get_file_hash(labels.SECTORS_FILE),
                "themes_csv_sha256": get_file_hash(labels.THEMES_FILE),
                "trendlab_sha256": get_file_hash(trendlab.__file__),
                "scanner_core_sha256": get_file_hash(scanner_core.__file__),
                "active_universe_file_sha256": get_file_hash(universe.ACTIVE_FILE),
                "bar_inputs_json_sha256": get_file_hash(tmp_bar_inputs),
                "bar_input_inventory_sha256": inventory_digest(inventory),
                "pandas_version": pd.__version__, "numpy_version": np.__version__,
                "pyarrow_version": pyarrow.__version__,
                "max_bar_date_processed": str(max_bar_date.date()) if max_bar_date else None,
            },
            "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_universe": "active_symbols (equities only; ETFs/Futures excluded)",
            "survivorship_caveat": "Conditional on equities present in the CURRENT active universe with cached history; delisted/inactive symbols omitted. Sector/theme membership is the current label snapshot (not point-in-time).",
            "horizon_sessions": HORIZON_SESSIONS,
            "thresholds_multiple_of_ep_close": THRESHOLDS,
            "stop_definition": "Any later bar with low <= EP-day low ends the event (row panel ends the bar before).",
            "panel_type": "Daily at-risk panel; one row per session while the event is alive (no stop breach, within 250 sessions, data available). Reaching +50% does not end the panel.",
            "label_semantics": "label_reach_T is event-level: 1 if high >= T before the stop; 0 if stop / same-bar ambiguity / timeout; NaN if right-censored. Train T-specific models only on rows with feature_chk_reached_T == 0.",
            "ambiguous_bar_policy": "same bar touching stop and threshold => ambiguous_same_bar, treated as 0 (failure) in label_reach_T; state retained.",
            "larsson_note": "Daily-timeframe Larsson Line only (production detector uses sector-optimal 2D/3D/1W resampling; not replicated here).",
            "statistics": stats,
            "column_roles": roles,
            "missing_rates": df_out.isna().mean().to_dict(),
        }
        with open(tmp_json, "w") as f:
            json.dump(manifest, f, indent=2)
        os.rename(tmp_dir, final_dir)
        print(f"\n[*] Stage 1 complete -> {final_dir}  ({dataset_id}); rows={len(df_out)}")
    except Exception as e:
        print(f"[!] Build failed: {e}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    build_dataset()
