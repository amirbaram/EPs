#!/usr/bin/env python3
"""
train_all_pivot_models.py (V2)

Trains, chronologically validates (<2024 Train vs 2025-2026 Out-of-Sample Test),
diagnoses, and persists the 3 additional Structural Swing-Pivot ML Models, and
provides the unified 4-Model inference engine used by Watchlist Studio (`8789`):

1. `uptrend_pivot_high_classifier.pkl` (Model 1 - from `train_uptrend_pivot_model.py`):
   - Identifies Climax Tops / Parabolic Stretch & Multi-Wave Bearish Divergence Highs in Uptrends.
2. `uptrend_pivot_low_classifier.pkl` (Model 2):
   - Identifies buyable Pullback Swing Lows / First MA Support Kisses (EMA20/SMA50) /
     Undercut-and-Reclaims (U&R) with Volume Dry-Up or Hammer Demand during an Uptrend.
3. `downtrend_pivot_high_classifier.pkl` (Model 3):
   - Identifies Bear-Market Rally Tops / Lower-High Rejections into declining EMA20/SMA50
     overhead resistance with RSI(5) vs RSI(14) bear-ceiling compression during a Downtrend.
4. `downtrend_pivot_low_classifier.pkl` (Model 4):
   - Identifies Capitulation / Selling-Climax Washout Bottoms (extreme ADR stretch below
     EMA20/SMA50, 120D p95 downside stretch, climax volume absorption, 18D/35D bullish RSI
     divergence at Low, and 2-bar washout reclaims) during a Downtrend.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import datastore
import indicators
import train_uptrend_pivot_model as tup

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT_DIR = Path("data/models")
UT_HIGH_PATH = OUT_DIR / "uptrend_pivot_high_classifier.pkl"
UT_LOW_PATH = OUT_DIR / "uptrend_pivot_low_classifier.pkl"
DT_HIGH_PATH = OUT_DIR / "downtrend_pivot_high_classifier.pkl"
DT_LOW_PATH = OUT_DIR / "downtrend_pivot_low_classifier.pkl"

UT_HIGH_V4_PATH = OUT_DIR / "uptrend_pivot_high_classifier_v4.pkl"
UT_LOW_V4_PATH = OUT_DIR / "uptrend_pivot_low_classifier_v4.pkl"
DT_HIGH_V4_PATH = OUT_DIR / "downtrend_pivot_high_classifier_v4.pkl"
DT_LOW_V4_PATH = OUT_DIR / "downtrend_pivot_low_classifier_v4.pkl"

_MARKET_SYNC_CACHE: Optional[pd.DataFrame] = None
_B4_INTRADAY_CACHE: Dict[str, pd.DataFrame] = {}


def _get_market_synchrony_df() -> pd.DataFrame:
    """Computes and caches strictly causal daily SPY & QQQ synchrony features (Family B2)."""
    global _MARKET_SYNC_CACHE
    if _MARKET_SYNC_CACHE is not None:
        return _MARKET_SYNC_CACHE

    spy = datastore.load_bars("SPY")
    qqq = datastore.load_bars("QQQ")
    if spy is None or spy.empty:
        _MARKET_SYNC_CACHE = pd.DataFrame()
        return _MARKET_SYNC_CACHE

    sc = spy["close"].astype(float)
    sh = spy["high"].astype(float)
    sl = spy["low"].astype(float)
    _, spy_rsi14_hi, spy_rsi14_lo = compute_rsi_with_extremes(sc, sh, sl, 14)
    s_prev_c = sc.shift(1)
    s_tr = np.maximum(sh - sl, np.maximum((sh - s_prev_c).abs(), (sl - s_prev_c).abs()))
    s_adr20 = np.maximum(s_tr.shift(1).rolling(20, min_periods=5).mean().bfill(), sc * 0.004)
    s_ema20_prev = sc.ewm(span=20, adjust=False).mean().shift(1).bfill()
    spy_ext_ema20_hi = (sh - s_ema20_prev) / s_adr20
    spy_ext_ema20_lo = (sl - s_ema20_prev) / s_adr20
    spy_thrust_5d_adr = (sh - sc.shift(5).bfill()) / s_adr20
    spy_drop_5d_adr = (sc.shift(5).bfill() - sl) / s_adr20

    if qqq is not None and not qqq.empty:
        qc = qqq["close"].astype(float).reindex(spy.index).ffill().bfill()
        qh = qqq["high"].astype(float).reindex(spy.index).ffill().bfill()
        ql = qqq["low"].astype(float).reindex(spy.index).ffill().bfill()
        _, qqq_rsi14_hi, qqq_rsi14_lo = compute_rsi_with_extremes(qc, qh, ql, 14)
    else:
        qqq_rsi14_hi = spy_rsi14_hi
        qqq_rsi14_lo = spy_rsi14_lo

    _MARKET_SYNC_CACHE = pd.DataFrame(
        {
            "spy_close": sc,
            "spy_high": sh,
            "spy_low": sl,
            "spy_adr20": s_adr20,
            "spy_rsi14_high": spy_rsi14_hi,
            "spy_rsi14_low": spy_rsi14_lo,
            "spy_ext_ema20_hi": spy_ext_ema20_hi,
            "spy_ext_ema20_lo": spy_ext_ema20_lo,
            "spy_thrust_5d_adr": spy_thrust_5d_adr,
            "spy_drop_5d_adr": spy_drop_5d_adr,
            "qqq_rsi14_high": qqq_rsi14_hi,
            "qqq_rsi14_low": qqq_rsi14_lo,
        },
        index=spy.index,
    )
    return _MARKET_SYNC_CACHE


def _fast_rsi14_extremes_arr(
    c: np.ndarray, h: np.ndarray, l: np.ndarray, period: int = 14
) -> Tuple[np.ndarray, np.ndarray]:
    """Fast vectorized Wilder's RSI(period) at High and Low on 1D numpy arrays."""
    if len(c) < period + 2:
        return np.full(len(c), 50.0), np.full(len(c), 50.0)
    s_c = pd.Series(c)
    delta = s_c.diff().fillna(0.0)
    alpha = 1.0 / float(period)
    ag = delta.clip(lower=0.0).ewm(alpha=alpha, min_periods=period, adjust=False).mean().to_numpy()
    al = (-delta.clip(upper=0.0)).ewm(alpha=alpha, min_periods=period, adjust=False).mean().to_numpy()
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    prev_ag = np.roll(ag, 1)
    prev_ag[0] = 0.0
    prev_al = np.roll(al, 1)
    prev_al[0] = 0.0

    gh = np.maximum(h - prev_c, 0.0)
    lh = np.maximum(prev_c - h, 0.0)
    agh = (prev_ag * (period - 1.0) + gh) / float(period)
    alh = (prev_al * (period - 1.0) + lh) / float(period)
    rsi_h = np.where(np.isnan(agh), 50.0, 100.0 - 100.0 / (1.0 + agh / (alh + 1e-9)))

    gl = np.maximum(l - prev_c, 0.0)
    ll = np.maximum(prev_c - l, 0.0)
    agl = (prev_ag * (period - 1.0) + gl) / float(period)
    all_ = (prev_al * (period - 1.0) + ll) / float(period)
    rsi_l = np.where(np.isnan(agl), 50.0, 100.0 - 100.0 / (1.0 + agl / (all_ + 1e-9)))
    return rsi_h, rsi_l


def _extract_b4_intraday_daily_summary(symbol: Optional[str]) -> Optional[pd.DataFrame]:
    """Extracts causal daily B4 Sub-Model A+ features from 5-min bars (`data/tiingo/bars_5min/<symbol>.parquet`):
    - 30-min post-peak intrabar pullback (`raw_post_peak_drop`) and post-trough bounce (`raw_post_trough_bounce`)
    - Pace-adjusted daily volume at peak (`pace_vol_hi`) and at trough (`pace_vol_lo`)
    - Developing intrabar wick % (`micro_upper_wick_pct_5m`, `micro_lower_wick_pct_5m`)
    - Multi-Timeframe (`5m, 15m, 30m, 1h, 4h`) Inverse Wilder's RSI(14) OB 75 & OS 25 target hits
    """
    if not symbol or ":" in symbol or "|" in symbol or "/" in symbol:
        return None
    sym = symbol.strip().upper()
    if sym in _B4_INTRADAY_CACHE:
        return _B4_INTRADAY_CACHE[sym]

    df5 = datastore.load_bars_5m(sym, calibrate=True)
    if df5 is None or len(df5) < 78:
        return None

    c5 = df5["close"].to_numpy(dtype=np.float64)
    h5 = df5["high"].to_numpy(dtype=np.float64)
    l5 = df5["low"].to_numpy(dtype=np.float64)
    v5 = df5["volume"].fillna(0.0).to_numpy(dtype=np.float64)
    d5 = df5.index.normalize()

    u_dates, first_idx, counts = np.unique(d5, return_index=True, return_counts=True)
    n_days = len(u_dates)
    max_c = int(counts.max())
    if n_days < 5 or max_c < 2:
        return None

    day_id = np.repeat(np.arange(n_days), counts)
    pos_in_day = np.arange(len(c5)) - np.repeat(first_idx, counts)

    H_mat = np.full((n_days, max_c), -np.inf)
    L_mat = np.full((n_days, max_c), np.inf)
    C_mat = np.full((n_days, max_c), np.nan)
    V_mat = np.zeros((n_days, max_c))

    H_mat[day_id, pos_in_day] = h5
    L_mat[day_id, pos_in_day] = l5
    C_mat[day_id, pos_in_day] = c5
    V_mat[day_id, pos_in_day] = v5

    k_hi = np.argmax(H_mat, axis=1)
    k_lo = np.argmin(L_mat, axis=1)
    day_hi = H_mat[np.arange(n_days), k_hi]
    day_lo = L_mat[np.arange(n_days), k_lo]

    # 30-min (6 5m-bar) post-peak / post-trough reaction window
    offsets = np.arange(7)[None, :]
    hi_win_idx = np.minimum((counts - 1)[:, None], k_hi[:, None] + offsets)
    lo_win_idx = np.minimum((counts - 1)[:, None], k_lo[:, None] + offsets)
    row_idx = np.arange(n_days)[:, None]

    post_peak_min = np.nanmin(np.minimum(L_mat[row_idx, hi_win_idx], C_mat[row_idx, hi_win_idx]), axis=1)
    post_trough_max = np.nanmax(np.maximum(H_mat[row_idx, lo_win_idx], C_mat[row_idx, lo_win_idx]), axis=1)

    raw_post_peak_drop = np.maximum(0.0, day_hi - post_peak_min)
    raw_post_trough_bounce = np.maximum(0.0, post_trough_max - day_lo)

    rng_day = np.maximum(1e-6, day_hi - day_lo)
    micro_up_wick = np.clip((raw_post_peak_drop / rng_day) * 100.0, 0.0, 100.0)
    micro_lo_wick = np.clip((raw_post_trough_bounce / rng_day) * 100.0, 0.0, 100.0)

    cum_V = np.cumsum(V_mat, axis=1)
    k_hi_v = np.minimum(counts - 1, k_hi + 3)
    k_lo_v = np.minimum(counts - 1, k_lo + 3)
    pace_v_hi = cum_V[np.arange(n_days), k_hi_v] / np.clip((k_hi_v + 1) / np.maximum(1.0, counts), 0.10, 1.0)
    pace_v_lo = cum_V[np.arange(n_days), k_lo_v] / np.clip((k_lo_v + 1) / np.maximum(1.0, counts), 0.10, 1.0)

    # Multi-Timeframe (5m, 15m, 30m, 1h, 4h) Inverse Wilder's RSI(14) OB75 / OS25 hit counts
    ob75_hits_intraday = np.zeros(n_days, dtype=np.float64)
    os25_hits_intraday = np.zeros(n_days, dtype=np.float64)
    for k_stride in (1, 3, 6, 13, 39):
        m_len = (len(c5) // k_stride) * k_stride
        if m_len < 15 * k_stride:
            continue
        ck = c5[:m_len].reshape(-1, k_stride)[:, -1]
        hk = h5[:m_len].reshape(-1, k_stride).max(axis=1)
        lk = l5[:m_len].reshape(-1, k_stride).min(axis=1)
        dk = day_id[:m_len].reshape(-1, k_stride)[:, -1]
        rh_k, rl_k = _fast_rsi14_extremes_arr(ck, hk, lk, 14)
        hit_ob_5m = np.repeat(rh_k >= 75.0, k_stride)
        hit_os_5m = np.repeat(rl_k <= 25.0, k_stride)
        if len(hit_ob_5m) < len(c5):
            pad_n = len(c5) - len(hit_ob_5m)
            hit_ob_5m = np.concatenate([hit_ob_5m, np.repeat(hit_ob_5m[-1], pad_n)])
            hit_os_5m = np.concatenate([hit_os_5m, np.repeat(hit_os_5m[-1], pad_n)])
        ob75_hits_intraday += (np.maximum.reduceat(hit_ob_5m.astype(np.int8), first_idx) > 0).astype(np.float64)
        os25_hits_intraday += (np.maximum.reduceat(hit_os_5m.astype(np.int8), first_idx) > 0).astype(np.float64)

    out_df = pd.DataFrame(
        {
            "raw_post_peak_drop": raw_post_peak_drop,
            "raw_post_trough_bounce": raw_post_trough_bounce,
            "micro_upper_wick_5m": micro_up_wick,
            "micro_lower_wick_5m": micro_lo_wick,
            "pace_vol_hi": pace_v_hi,
            "pace_vol_lo": pace_v_lo,
            "intraday_ob75_hits": ob75_hits_intraday,
            "intraday_os25_hits": os25_hits_intraday,
        },
        index=pd.DatetimeIndex(u_dates),
    )
    if len(_B4_INTRADAY_CACHE) < 1200:
        _B4_INTRADAY_CACHE[sym] = out_df
    return out_df


def compute_rsi_with_extremes(
    close: pd.Series, high: pd.Series, low: pd.Series, period: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Computes:
    1. Standard EOD Wilder's RSI (on Close[t])
    2. Exact 'At-the-High' Wilder's RSI (evaluating RSI at High[t] using only prior bars' avg_gain/avg_loss)
    3. Exact 'At-the-Low' Wilder's RSI (evaluating RSI at Low[t] using only prior bars' avg_gain/avg_loss)
    Strictly causal with zero lookahead to EOD close when evaluating at High or Low.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    rs_close = avg_gain / (avg_loss + 1e-9)
    rsi_close = (100.0 - (100.0 / (1.0 + rs_close))).fillna(50.0)

    prev_close = close.shift(1)
    prev_ag = avg_gain.shift(1)
    prev_al = avg_loss.shift(1)

    # At-the-High RSI
    gain_h = (high - prev_close).clip(lower=0.0)
    loss_h = (prev_close - high).clip(lower=0.0)
    ag_h = (prev_ag * (period - 1.0) + gain_h) / period
    al_h = (prev_al * (period - 1.0) + loss_h) / period
    rsi_high = (100.0 - (100.0 / (1.0 + ag_h / (al_h + 1e-9)))).fillna(rsi_close)

    # At-the-Low RSI
    gain_l = (low - prev_close).clip(lower=0.0)
    loss_l = (prev_close - low).clip(lower=0.0)
    ag_l = (prev_ag * (period - 1.0) + gain_l) / period
    al_l = (prev_al * (period - 1.0) + loss_l) / period
    rsi_low = (100.0 - (100.0 / (1.0 + ag_l / (al_l + 1e-9)))).fillna(rsi_close)

    return rsi_close, rsi_high, rsi_low


def compute_all_pivot_features(df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
    """
    Unified vectorized feature extractor for all 4 structural pivot models (V3 + V4 Enhancements B1-B4):
    1. Uptrend Pivot High (UT_HIGH)
    2. Uptrend Pivot Low (UT_LOW)
    3. Downtrend Pivot High (DT_HIGH)
    4. Downtrend Pivot Low (DT_LOW)
    Constructs all new columns in a single dictionary and joins via `pd.concat`
    to avoid DataFrame fragmentation and maximize inference speed.
    """
    b = tup.compute_uptrend_pivot_features(df).copy()
    cols: Dict[str, Any] = {}

    c = b["close"].astype(float)
    h = b["high"].astype(float)
    l = b["low"].astype(float)
    o = b["open"].astype(float)
    v = b["volume"].fillna(0.0).astype(float) if "volume" in b.columns else pd.Series(0.0, index=b.index)
    adr20 = b["adr20"]
    prev_c = c.shift(1)

    # 1. At-the-Low RSI suite (14, 9, 5)
    _, _, rsi14_low = compute_rsi_with_extremes(c, h, l, 14)
    _, _, rsi9_low = compute_rsi_with_extremes(c, h, l, 9)
    _, _, rsi5_low = compute_rsi_with_extremes(c, h, l, 5)

    cols["rsi14_low"] = rsi14_low
    cols["rsi9_low"] = rsi9_low
    cols["rsi5_low"] = rsi5_low
    cols["prev_rsi14_low"] = rsi14_low.shift(1).fillna(50.0)
    cols["rsi14_low_min5"] = rsi14_low.rolling(5, min_periods=1).min()
    cols["rsi5_low_min3"] = rsi5_low.rolling(3, min_periods=1).min()
    cols["rsi14_low_diff3"] = rsi14_low - rsi14_low.shift(3).fillna(50.0)
    cols["rsi14_low_diff5"] = rsi14_low - rsi14_low.shift(5).fillna(50.0)
    cols["consec_rsi30_5d"] = (rsi14_low <= 30.0).rolling(5, min_periods=1).sum()
    cols["consec_rsi25_5d"] = (rsi14_low <= 25.0).rolling(5, min_periods=1).sum()
    cols["consec_rsi5_20_5d"] = (rsi5_low <= 20.0).rolling(5, min_periods=1).sum()

    # Fast vs Slow RSI spread at High (for Bear Rally Ceiling detection) and at Low
    cols["rsi5_vs_rsi14_high_spread"] = b["rsi5_high"] - b["rsi14_high"]
    cols["rsi14_vs_rsi5_low_spread"] = rsi14_low - rsi5_low

    # 2. Moving average anchors (shifted by 1 bar so "At-the-Low" and "At-the-High" are strictly causal intraday)
    ema8_anchor = b["ema8"].shift(1).fillna(b["ema8"])
    ema10_anchor = b["ema10"].shift(1).fillna(b["ema10"])
    ema20_anchor = b["ema20"].shift(1).fillna(b["ema20"])
    sma50_anchor = b["sma50"].shift(1).fillna(b["sma50"])
    sma200 = c.rolling(200, min_periods=50).mean().fillna(b["sma50"])
    sma200_anchor = sma200.shift(1).fillna(sma200)
    cols["sma200"] = sma200
    cols["ema8_anchor"] = ema8_anchor
    cols["ema10_anchor"] = ema10_anchor
    cols["ema20_anchor"] = ema20_anchor
    cols["sma50_anchor"] = sma50_anchor
    cols["sma200_anchor"] = sma200_anchor
    cols["sma20_anchor"] = b["sma20"].shift(1).bfill()
    cols["std20_anchor"] = b["std20"].shift(1).bfill().replace(0, np.nan)

    # Trend Slopes & Multi-Month Regime Consistency
    ema20_slope_5d_adr = (b["ema20"] - b["ema20"].shift(5)) / adr20
    sma50_slope_10d_adr = (b["sma50"] - b["sma50"].shift(10)) / adr20
    cols["ema20_slope_5d_adr"] = ema20_slope_5d_adr
    cols["sma50_slope_10d_adr"] = sma50_slope_10d_adr

    above_sma50_bar = (c >= b["sma50"]).astype(float)
    cols["days_above_sma50_60d"] = above_sma50_bar.shift(1).rolling(60, min_periods=15).mean().fillna(0.5)
    cols["days_below_sma50_60d"] = 1.0 - cols["days_above_sma50_60d"]

    # 3. AT-THE-LOW MA DISTANCES & DOWNSIDE STRETCH (IN ADR UNITS)
    low_dist_ema8_adr = (l - ema8_anchor) / adr20
    low_dist_ema10_adr = (l - ema10_anchor) / adr20
    low_dist_ema20_adr = (l - ema20_anchor) / adr20
    low_dist_sma50_adr = (l - sma50_anchor) / adr20
    low_dist_sma200_adr = (l - sma200_anchor) / adr20

    cols["low_dist_ema8_adr"] = low_dist_ema8_adr
    cols["low_dist_ema10_adr"] = low_dist_ema10_adr
    cols["low_dist_ema20_adr"] = low_dist_ema20_adr
    cols["low_dist_sma50_adr"] = low_dist_sma50_adr
    cols["low_dist_sma200_adr"] = low_dist_sma200_adr

    stretch_below_ema10_adr = (ema10_anchor - l) / adr20
    stretch_below_ema20_adr = (ema20_anchor - l) / adr20
    stretch_below_sma50_adr = (sma50_anchor - l) / adr20
    stretch_below_ema20_max3 = stretch_below_ema20_adr.rolling(3, min_periods=1).max()
    prev_stretch_below_ema20_adr = ((ema20_anchor - prev_c) / adr20).fillna(0.0)

    cols["stretch_below_ema10_adr"] = stretch_below_ema10_adr
    cols["stretch_below_ema20_adr"] = stretch_below_ema20_adr
    cols["stretch_below_sma50_adr"] = stretch_below_sma50_adr
    cols["stretch_below_ema20_max3"] = stretch_below_ema20_max3
    cols["prev_stretch_below_ema20_adr"] = prev_stretch_below_ema20_adr

    # Support Kiss Proximity (for Uptrend Pullback Lows)
    kiss_ema10_dist_adr = low_dist_ema10_adr.abs()
    kiss_ema20_dist_adr = low_dist_ema20_adr.abs()
    kiss_sma50_dist_adr = low_dist_sma50_adr.abs()
    min_support_kiss_adr = np.minimum(kiss_ema20_dist_adr, kiss_sma50_dist_adr)
    cols["kiss_ema10_dist_adr"] = kiss_ema10_dist_adr
    cols["kiss_ema20_dist_adr"] = kiss_ema20_dist_adr
    cols["kiss_sma50_dist_adr"] = kiss_sma50_dist_adr
    cols["min_support_kiss_adr"] = min_support_kiss_adr

    # First Institutional MA Touch Indicators (Uptrend Pullback V2 Feature)
    clean_above_ema20_15d = (l.shift(1) > b["ema20"].shift(1)).astype(float).rolling(15, min_periods=5).sum()
    clean_above_sma50_25d = (l.shift(1) > b["sma50"].shift(1)).astype(float).rolling(25, min_periods=8).sum()
    cols["clean_above_ema20_15d"] = clean_above_ema20_15d
    cols["clean_above_sma50_25d"] = clean_above_sma50_25d
    cols["first_ema20_test_in_15d"] = ((low_dist_ema20_adr <= 0.35) & (low_dist_ema20_adr >= -0.80) & (clean_above_ema20_15d >= 11)).astype(float)
    cols["first_sma50_test_in_25d"] = ((low_dist_sma50_adr <= 0.45) & (low_dist_sma50_adr >= -0.90) & (clean_above_sma50_25d >= 18)).astype(float)

    # Overhead Resistance Kiss & Rejection (for Downtrend Bear Rally Highs V2)
    kiss_falling_ema20_adr = ((ema20_anchor - h).abs() / adr20)
    kiss_falling_sma50_adr = ((sma50_anchor - h).abs() / adr20)
    min_overhead_kiss_adr = np.minimum(kiss_falling_ema20_adr, kiss_falling_sma50_adr)
    cols["kiss_falling_ema20_adr"] = kiss_falling_ema20_adr
    cols["kiss_falling_sma50_adr"] = kiss_falling_sma50_adr
    cols["min_overhead_kiss_adr"] = min_overhead_kiss_adr
    cols["high_dist_sma50_adr"] = (h - sma50_anchor) / adr20
    cols["high_dist_sma200_adr"] = (h - sma200_anchor) / adr20
    prev_40d_high_s5 = h.shift(5).rolling(40, min_periods=10).max().bfill()
    cols["prev_40d_high_s5"] = prev_40d_high_s5
    cols["dist_below_40d_high_adr"] = ((prev_40d_high_s5 - h) / adr20).fillna(2.0)

    # Bollinger Z-score at Low[t] & 120D Self-Normalized Downside Stretch
    bb_z_low = ((l - b["sma20"].shift(1)) / b["std20"].shift(1)).clip(-6.0, 4.0).fillna(0.0)
    cols["bb_z_low"] = bb_z_low
    cols["bb_z_low_min3"] = bb_z_low.rolling(3, min_periods=1).min()

    roll_p95_down20 = stretch_below_ema20_adr.shift(1).rolling(120, min_periods=25).quantile(0.95).clip(lower=1.5).bfill()
    cols["roll_p95_down20"] = roll_p95_down20
    cols["rel_down_ema20_p95"] = stretch_below_ema20_adr / roll_p95_down20

    # Prior-bar rolling anchors for exact intrabar sweep recomputation
    cols["roll_p95_ema10"] = b["ext_ema10_adr"].shift(1).rolling(120, min_periods=25).quantile(0.95).clip(lower=1.5).bfill()
    cols["roll_p95_ema20"] = b["ext_ema20_adr"].shift(1).rolling(120, min_periods=25).quantile(0.95).clip(lower=2.0).bfill()
    cols["prev_ext10_max2"] = b["ext_ema10_adr"].shift(1).rolling(2, min_periods=1).max().fillna(-99.0)
    cols["prev_ext20_max2"] = b["ext_ema20_adr"].shift(1).rolling(2, min_periods=1).max().fillna(-99.0)
    cols["prev_bb_z_hi_max2"] = b["bb_z_high"].shift(1).rolling(2, min_periods=1).max().fillna(-99.0)
    cols["prev_rsi14_hi_max4"] = b["rsi14_high"].shift(1).rolling(4, min_periods=1).max().fillna(50.0)
    cols["prev_rsi5_hi_max2"] = b["rsi5_high"].shift(1).rolling(2, min_periods=1).max().fillna(50.0)
    cols["rsi14_hi_shift3"] = b["rsi14_high"].shift(3).fillna(50.0)
    cols["rsi14_hi_shift5"] = b["rsi14_high"].shift(5).fillna(50.0)
    cols["prev_consec_rsi70_4d"] = (b["rsi14_high"] >= 70.0).astype(float).shift(1).rolling(4, min_periods=1).sum().fillna(0.0)
    cols["prev_consec_rsi75_4d"] = (b["rsi14_high"] >= 75.0).astype(float).shift(1).rolling(4, min_periods=1).sum().fillna(0.0)
    cols["prev_consec_rsi80_4d"] = (b["rsi5_high"] >= 80.0).astype(float).shift(1).rolling(4, min_periods=1).sum().fillna(0.0)
    cols["prev_wave_high_18"] = h.shift(3).rolling(16, min_periods=4).max().bfill()
    cols["prev_wave_rsi_18"] = b["rsi14_high"].shift(3).rolling(16, min_periods=4).max().fillna(50.0)
    cols["prev_wave_high_35"] = h.shift(4).rolling(32, min_periods=8).max().bfill()
    cols["prev_wave_rsi_35"] = b["rsi14_high"].shift(4).rolling(32, min_periods=8).max().fillna(50.0)
    cols["prev_wave_ext_35"] = b["ext_ema20_adr"].shift(4).rolling(32, min_periods=8).max().fillna(0.0)

    cols["prev_stretch20_max2"] = stretch_below_ema20_adr.shift(1).rolling(2, min_periods=1).max().fillna(-99.0)
    cols["prev_bb_z_lo_min2"] = bb_z_low.shift(1).rolling(2, min_periods=1).min().fillna(99.0)
    cols["prev_rsi14_lo_min4"] = rsi14_low.shift(1).rolling(4, min_periods=1).min().fillna(50.0)
    cols["prev_rsi5_lo_min2"] = rsi5_low.shift(1).rolling(2, min_periods=1).min().fillna(50.0)
    cols["rsi14_lo_shift3"] = rsi14_low.shift(3).fillna(50.0)
    cols["rsi14_lo_shift5"] = rsi14_low.shift(5).fillna(50.0)
    cols["prev_consec_rsi30_4d"] = (rsi14_low <= 30.0).astype(float).shift(1).rolling(4, min_periods=1).sum().fillna(0.0)
    cols["prev_consec_rsi25_4d"] = (rsi14_low <= 25.0).astype(float).shift(1).rolling(4, min_periods=1).sum().fillna(0.0)
    cols["prev_consec_rsi5_20_4d"] = (rsi5_low <= 20.0).astype(float).shift(1).rolling(4, min_periods=1).sum().fillna(0.0)

    # 4. MULTI-DAY PULLBACK / WATERFALL VELOCITY TO LOW[t] (IN ADR UNITS)
    drop_1d_adr = (prev_c - l) / adr20
    drop_2d_adr = (c.shift(2) - l) / adr20
    drop_3d_adr = (h.shift(1).rolling(3, min_periods=1).max() - l) / adr20
    drop_5d_adr = (h.shift(1).rolling(5, min_periods=1).max() - l) / adr20
    drop_10d_adr = (h.shift(1).rolling(10, min_periods=2).max() - l) / adr20
    drop_15d_adr = (h.shift(1).rolling(15, min_periods=3).max() - l) / adr20
    drop_20d_adr = (h.shift(1).rolling(20, min_periods=5).max() - l) / adr20
    drop_40d_adr = (h.shift(1).rolling(40, min_periods=10).max() - l) / adr20

    cols["drop_1d_adr"] = drop_1d_adr
    cols["drop_2d_adr"] = drop_2d_adr
    cols["drop_3d_adr"] = drop_3d_adr
    cols["drop_5d_adr"] = drop_5d_adr
    cols["drop_10d_adr"] = drop_10d_adr
    cols["drop_15d_adr"] = drop_15d_adr
    cols["drop_20d_adr"] = drop_20d_adr
    cols["drop_40d_adr"] = drop_40d_adr
    cols["waterfall_accel"] = (drop_3d_adr / 3.0) / np.maximum(0.15, drop_10d_adr / 10.0)

    # Prior Rally Leg Context (Uptrend Pullback Retracement Ratio)
    prior_20d_high = h.shift(1).rolling(20, min_periods=5).max()
    prior_40d_low = l.shift(5).rolling(40, min_periods=10).min()
    prior_rally_adr = ((prior_20d_high - prior_40d_low) / adr20).clip(lower=1.0)
    cols["prior_rally_20d_adr"] = prior_rally_adr
    cols["pullback_retracement_ratio"] = (drop_15d_adr / prior_rally_adr).clip(0.0, 2.5).fillna(0.5)

    # Prior Selloff Leg Context (Downtrend Bounce Retracement Ratio)
    prior_20d_low = l.shift(1).rolling(20, min_periods=5).min()
    prior_40d_high = h.shift(5).rolling(40, min_periods=10).max()
    prior_selloff_adr = ((prior_40d_high - prior_20d_low) / adr20).clip(lower=1.0)
    cols["prior_selloff_20d_adr"] = prior_selloff_adr
    cols["bounce_retracement_ratio"] = (b["thrust_10d_adr"] / prior_selloff_adr).clip(0.0, 2.5).fillna(0.5)

    # 5. Uninterrupted Down-Leg from EMA10 / EMA20
    low_arr = l.to_numpy()
    high_arr = h.to_numpy()
    ema10_arr = b["ema10"].to_numpy()
    ema20_arr = b["ema20"].to_numpy()
    adr_arr = adr20.to_numpy()
    n = len(b)

    bars_below_ema10 = np.zeros(n)
    bars_below_ema20 = np.zeros(n)
    down_leg_adr_10 = np.zeros(n)
    down_leg_adr_20 = np.zeros(n)
    anchor_hi_10 = high_arr[0]
    anchor_hi_20 = high_arr[0]
    cnt_b10 = 0
    cnt_b20 = 0

    for i in range(1, n):
        if high_arr[i - 1] >= ema10_arr[i - 1] * 0.995:
            cnt_b10 = 1
            anchor_hi_10 = high_arr[i - 1]
        else:
            cnt_b10 += 1
            if high_arr[i - 1] > anchor_hi_10:
                anchor_hi_10 = high_arr[i - 1]

        if high_arr[i - 1] >= ema20_arr[i - 1] * 0.992:
            cnt_b20 = 1
            anchor_hi_20 = high_arr[i - 1]
        else:
            cnt_b20 += 1
            if high_arr[i - 1] > anchor_hi_20:
                anchor_hi_20 = high_arr[i - 1]

        bars_below_ema10[i] = cnt_b10
        bars_below_ema20[i] = cnt_b20
        cur_adr = max(1e-6, adr_arr[i])
        down_leg_adr_10[i] = max(0.0, (anchor_hi_10 - low_arr[i]) / cur_adr)
        down_leg_adr_20[i] = max(0.0, (anchor_hi_20 - low_arr[i]) / cur_adr)

    cols["bars_below_ema10"] = bars_below_ema10
    cols["bars_below_ema20"] = bars_below_ema20
    cols["down_leg_adr_from_ema10"] = down_leg_adr_10
    cols["down_leg_adr_from_ema20"] = down_leg_adr_20

    # Down-day streaks & consecutive lower lows
    down_bar = (c < prev_c).astype(float)
    cols["down_days_5d"] = down_bar.shift(1).rolling(5, min_periods=1).sum() + (l < prev_c).astype(float)
    cols["down_days_10d"] = down_bar.shift(1).rolling(10, min_periods=1).sum() + (l < prev_c).astype(float)

    ll_bar = (l < l.shift(1)).astype(int)
    consec_ll = np.zeros(n)
    run_ll = 0
    for i, val in enumerate(ll_bar.to_numpy()):
        run_ll = (run_ll + 1) if val == 1 else 0
        consec_ll[i] = run_ll
    cols["consec_ll"] = consec_ll

    cols["open_to_low_adr"] = (o - l) / adr20
    cols["gap_down_on_oversold"] = (-b["gap_adr"]).clip(lower=0.0) * prev_stretch_below_ema20_adr.clip(lower=0.0)

    # Distance to 8D / 10D / 20D Low
    roll_l8 = l.rolling(8, min_periods=2).min()
    roll_l10_prev = l.shift(1).rolling(10, min_periods=2).min().bfill()
    roll_l20_prev = l.shift(1).rolling(20, min_periods=5).min().bfill()
    dist_from_8d_low_adr = (l - roll_l8) / adr20
    cols["dist_from_8d_low_adr"] = dist_from_8d_low_adr
    cols["undercut_10d_low_adr"] = (roll_l10_prev - l) / adr20
    cols["undercut_20d_low_adr"] = (roll_l20_prev - l) / adr20

    # 6. MULTI-WAVE BULLISH DIVERGENCE AT LOW (18-DAY & 35-DAY)
    prev_wave_low_18 = l.shift(3).rolling(16, min_periods=4).min()
    prev_wave_rsi_low_18 = rsi14_low.shift(3).rolling(16, min_periods=4).min()
    cols["prev_wave_low_18"] = prev_wave_low_18.bfill()
    cols["prev_wave_rsi_low_18"] = prev_wave_rsi_low_18.fillna(50.0)
    is_nl_18 = l <= prev_wave_low_18 * 1.015
    cols["bullish_div_low_18d"] = (is_nl_18 & (rsi14_low > prev_wave_rsi_low_18 + 1.5)).astype(float)
    cols["rsi_bull_div_delta_18d"] = np.where(is_nl_18, (rsi14_low - prev_wave_rsi_low_18).clip(lower=0.0), 0.0)

    prev_wave_low_35 = l.shift(4).rolling(32, min_periods=8).min()
    prev_wave_rsi_low_35 = rsi14_low.shift(4).rolling(32, min_periods=8).min()
    cols["prev_wave_low_35"] = prev_wave_low_35.bfill()
    cols["prev_wave_rsi_low_35"] = prev_wave_rsi_low_35.fillna(50.0)
    is_nl_35 = l <= prev_wave_low_35 * 1.015
    cols["bullish_div_low_35d"] = (is_nl_35 & (rsi14_low > prev_wave_rsi_low_35 + 2.0)).astype(float)
    cols["rsi_bull_div_delta_35d"] = np.where(is_nl_35, (rsi14_low - prev_wave_rsi_low_35).clip(lower=0.0), 0.0)

    # 7. INTRADAY DEMAND / HAMMER / UNDERCUT & RECLAIM / WASHOUT FEATURES (V2)
    rng = (h - l).clip(lower=1e-6)
    bounce_from_low_adr = (c - l) / adr20
    cols["lower_wick_pct"] = ((np.minimum(o, c) - l) / rng) * 100.0
    cols["bounce_from_low_adr"] = bounce_from_low_adr
    cols["green_reclaim_bar"] = ((l <= roll_l8 * 1.008) & (c > o)).astype(float)
    cols["hammer_vol_score"] = b["rvol"] * bounce_from_low_adr
    cols["capitulation_vol_stretch"] = b["rvol"] * stretch_below_ema20_adr.clip(lower=0.0)

    # V2 Volume Dry-Up on Pullback vs Climax Washout Score
    vol_ma20 = v.shift(1).rolling(20, min_periods=5).mean().replace(0, np.nan)
    vol_3d_ratio = (v.rolling(3, min_periods=1).mean() / vol_ma20).fillna(1.0).clip(0.1, 10.0)
    cols["pullback_vol_dryup"] = vol_3d_ratio
    cols["support_confluence_score"] = bounce_from_low_adr / (0.25 + min_support_kiss_adr)

    climax_washout_score = b["rvol"] * b["range_to_adr"] * stretch_below_ema20_adr.clip(lower=0.0)
    cols["climax_washout_score"] = climax_washout_score
    cols["seller_exhaustion_wick"] = bounce_from_low_adr * stretch_below_ema20_adr.clip(lower=0.0)
    cols["day2_washout_reclaim"] = climax_washout_score.shift(1).fillna(0.0) * ((c - o) / adr20).clip(lower=0.0)

    # V2 Overhead Falling-MA Rejection (for Downtrend Pivot Highs)
    rejected_ema20 = (kiss_falling_ema20_adr <= 0.45) & (c < ema20_anchor) & (b["fade_from_high_adr"] >= 0.35)
    rejected_sma50 = (kiss_falling_sma50_adr <= 0.55) & (c < sma50_anchor) & (b["fade_from_high_adr"] >= 0.35)
    cols["rejected_at_falling_ma"] = (rejected_ema20 | rejected_sma50).astype(float)
    cols["overhead_rejection_score"] = b["fade_from_high_adr"] / (0.25 + min_overhead_kiss_adr)

    # 2-Bar Bottom Reclaim & Undercut-and-Reclaim (U&R) of EMA20 / SMA50 / 10D Low
    cols["two_bar_bottom_reclaim"] = drop_5d_adr.shift(1).clip(lower=0.0) * ((c - o) / adr20).clip(lower=0.0)
    cols["failed_low_undercut"] = (
        (((l - l.shift(1)).abs() / adr20) <= 0.65)
        & (c > prev_c)
        & (drop_10d_adr >= 2.0)
    ).astype(float)
    cols["ma_undercut_reclaim"] = (
        (((l <= ema20_anchor) & (c >= ema20_anchor)) | ((l <= sma50_anchor) & (c >= sma50_anchor)))
    ).astype(float)

    # Breakdown Ignition Filter (Day-1 high-volume gap-down closing weak out of an unextended base is a breakdown, NOT a buyable dip!)
    is_fresh_breakdown = (
        (b["gap_adr"] <= -1.5)
        & (b["rvol"] >= 1.75)
        & ((b["close_pos"] <= 0.45) | (prev_stretch_below_ema20_adr < 1.25))
    )
    cols["is_fresh_breakdown"] = is_fresh_breakdown

    # =========================================================================
    # V3 MUTUALLY EXCLUSIVE STRUCTURAL REGIME & DIRECTIONAL MASKS
    # =========================================================================
    is_primary_uptrend = (
        (b["ema20"] >= b["sma50"])
        | ((c >= b["sma50"]) & ((sma50_slope_10d_adr >= -0.05) | (b["ema10"] >= b["sma50"])))
    )
    is_primary_downtrend = ~is_primary_uptrend
    cols["is_primary_uptrend"] = is_primary_uptrend
    cols["is_primary_downtrend"] = is_primary_downtrend

    net_above_ema8 = (h - ema8_anchor) > (ema8_anchor - l)
    net_below_ema8 = ~net_above_ema8
    cols["net_above_ema8"] = net_above_ema8

    # 1. UPTREND PULLBACK REGIME (for Model 2: Uptrend Pivot Low)
    is_uptrend_pullback_regime = is_primary_uptrend & (prior_rally_adr >= 2.2)
    cols["is_uptrend_pullback_regime"] = is_uptrend_pullback_regime
    cols["is_near_uptrend_low"] = (
        is_uptrend_pullback_regime
        & (~is_fresh_breakdown)
        & net_below_ema8
        & (low_dist_ema10_adr <= 0.25)
        & ((rsi14_low <= 55.0) | (low_dist_ema20_adr <= 0.15))
        & (dist_from_8d_low_adr <= 0.45)
        & (drop_15d_adr >= 2.0)
    )

    # 2. DOWNTREND REGIME (for Model 3: Downtrend Pivot High & Model 4: Downtrend Pivot Low)
    cols["is_downtrend_regime"] = is_primary_downtrend
    roll_h8 = h.rolling(8, min_periods=2).max()

    cols["is_near_downtrend_high"] = (
        is_primary_downtrend
        & (~b["is_fresh_ignition"])
        & net_above_ema8
        & (b["ext_ema10_adr"] >= -0.25)
        & (b["rsi14_high"] >= 48.0)
        & (((roll_h8 - h) / adr20) <= 0.45)
        & (b["thrust_10d_adr"] >= 2.0)
    )

    cols["is_near_downtrend_low"] = (
        is_primary_downtrend
        & (~is_fresh_breakdown)
        & net_below_ema8
        & (low_dist_ema10_adr < -0.20)
        & (rsi14_low <= 44.0)
        & (dist_from_8d_low_adr <= 0.45)
        & ((stretch_below_ema20_max3 >= 1.4) | (drop_10d_adr >= 2.6))
    )

    b["is_near_local_high"] = (
        b["is_near_local_high"]
        & is_primary_uptrend
        & net_above_ema8
    )

    # =========================================================================
    # V4 ENHANCEMENTS B1–B4 (ZERO LOOKAHEAD CAUSAL FEATURE FAMILIES)
    # =========================================================================
    c_arr = c.to_numpy(dtype=np.float64)
    v_arr = v.to_numpy(dtype=np.float64)
    ref_px_arr = np.where(net_above_ema8.to_numpy(), high_arr, low_arr)

    # --- B1: Structural Overhead Supply & Support Memory + Inverse Wilder's RSI (OB 75 / OS 25) ---
    # 1a. Confirmed prior structural swing high/low (bar i-3 confirmed as 6-bar local extremum at i-1)
    prior_sh = h.shift(3).where(h.shift(3) >= h.shift(1).rolling(6, min_periods=3).max()).ffill().fillna(h.shift(1).bfill())
    prior_sl = l.shift(3).where(l.shift(3) <= l.shift(1).rolling(6, min_periods=3).min()).ffill().fillna(l.shift(1).bfill())
    cols["prior_swing_high"] = prior_sh
    cols["prior_swing_low"] = prior_sl
    cols["dist_to_prior_swing_high_adr"] = ((ref_px_arr - prior_sh.to_numpy()) / adr_arr).clip(-15.0, 15.0)
    cols["dist_to_prior_swing_low_adr"] = ((ref_px_arr - prior_sl.to_numpy()) / adr_arr).clip(-15.0, 15.0)

    # 1b. Causal 60-day peak & trough Anchored VWAP + 60-day Volume Density within +/- 0.5 ADR
    W = 60
    h_pad = np.concatenate([np.repeat(high_arr[0], W), high_arr[:-1]])
    l_pad = np.concatenate([np.repeat(low_arr[0], W), low_arr[:-1]])
    c_pad = np.concatenate([np.repeat(c_arr[0], W), c_arr[:-1]])
    v_eff = np.where(v_arr > 0, v_arr, 1.0)
    v_pad = np.concatenate([np.repeat(v_eff[0], W), v_eff[:-1]])
    dvol_pad = c_pad * v_pad

    h_win = sliding_window_view(h_pad, W)
    l_win = sliding_window_view(l_pad, W)
    dvol_win = sliding_window_view(dvol_pad, W)

    band_lo = (ref_px_arr - 0.5 * adr_arr)[:, None]
    band_hi = (ref_px_arr + 0.5 * adr_arr)[:, None]
    overlaps = (l_win <= band_hi) & (h_win >= band_lo)
    ov_dens = (dvol_win * overlaps).sum(axis=1) / np.maximum(1e-9, dvol_win.sum(axis=1))
    cols["overhead_volume_density_60d"] = np.clip(ov_dens, 0.0, 1.0)

    k_hi_rel = np.argmax(h_win, axis=1)
    k_lo_rel = np.argmin(l_win, axis=1)
    tp_pad = (h_pad + l_pad + c_pad) / 3.0
    pv_pad = tp_pad * v_pad
    cum_pv = np.concatenate([[0.0], np.cumsum(pv_pad)])
    cum_v = np.concatenate([[0.0], np.cumsum(v_pad)])
    end_idx = np.arange(W, W + n)
    start_hi = np.arange(n) + k_hi_rel
    start_lo = np.arange(n) + k_lo_rel
    avwap_hi = (cum_pv[end_idx] - cum_pv[start_hi]) / np.maximum(1e-9, cum_v[end_idx] - cum_v[start_hi])
    avwap_lo = (cum_pv[end_idx] - cum_pv[start_lo]) / np.maximum(1e-9, cum_v[end_idx] - cum_v[start_lo])
    cols["avwap_60d_high"] = avwap_hi
    cols["avwap_60d_low"] = avwap_lo
    cols["avwap_from_60d_high_dist_adr"] = np.clip((ref_px_arr - avwap_hi) / adr_arr, -15.0, 15.0)
    cols["avwap_from_60d_low_dist_adr"] = np.clip((ref_px_arr - avwap_lo) / adr_arr, -15.0, 15.0)

    # 1c. Exact Piecewise Inverse Wilder's RSI(14) OB 75 & OS 25 Target Price Distance (in ADR)
    ob75_px = indicators.rsi_trigger_price(c, 75.0, 14).bfill().fillna(c)
    os25_px = indicators.rsi_trigger_price(c, 25.0, 14).bfill().fillna(c)
    cols["ob75_px"] = ob75_px
    cols["os25_px"] = os25_px
    cols["inv_rsi75_dist_adr"] = ((h - ob75_px) / adr20).clip(-12.0, 12.0).fillna(0.0)
    cols["inv_rsi25_dist_adr"] = ((os25_px - l) / adr20).clip(-12.0, 12.0).fillna(0.0)

    # --- B2: Market (SPY / QQQ) & Sector RS Synchrony Features ---
    mkt_df = _get_market_synchrony_df()
    if not mkt_df.empty and isinstance(b.index, pd.DatetimeIndex):
        mkt_aligned = mkt_df.reindex(b.index).ffill().bfill()
        spy_rsi14_hi = mkt_aligned["spy_rsi14_high"].fillna(50.0)
        spy_rsi14_lo = mkt_aligned["spy_rsi14_low"].fillna(50.0)
        qqq_rsi14_hi = mkt_aligned["qqq_rsi14_high"].fillna(50.0)
        qqq_rsi14_lo = mkt_aligned["qqq_rsi14_low"].fillna(50.0)
        spy_ext_20 = np.where(
            net_above_ema8,
            mkt_aligned["spy_ext_ema20_hi"].fillna(0.0),
            mkt_aligned["spy_ext_ema20_lo"].fillna(0.0),
        )
        spy_th5 = mkt_aligned["spy_thrust_5d_adr"].fillna(0.0)
        spy_dr5 = mkt_aligned["spy_drop_5d_adr"].fillna(0.0)
        spy_h_s = mkt_aligned["spy_high"].replace(0, np.nan).bfill()
        spy_l_s = mkt_aligned["spy_low"].replace(0, np.nan).bfill()

        cols["spy_rsi14_high"] = spy_rsi14_hi
        cols["spy_rsi14_low"] = spy_rsi14_lo
        cols["spy_ext_ema20_adr"] = spy_ext_20
        cols["qqq_rsi14_high"] = qqq_rsi14_hi
        cols["qqq_rsi14_low"] = qqq_rsi14_lo
        cols["spy_thrust_5d_adr"] = spy_th5
        cols["spy_drop_5d_adr"] = spy_dr5

        own_move_5d_hi = ((h - c.shift(5).bfill()) / adr20).fillna(0.0)
        own_move_5d_lo = ((c.shift(5).bfill() - l) / adr20).fillna(0.0)
        cols["rs_vs_spy_5d_adr"] = np.where(
            net_above_ema8,
            own_move_5d_hi - spy_th5,
            own_move_5d_lo - spy_dr5,
        )

        rs_hi = (h / spy_h_s).ffill().bfill()
        rs_lo = (l / spy_l_s).ffill().bfill()
        prev_rs_h20 = rs_hi.shift(3).rolling(16, min_periods=4).max().bfill()
        prev_rs_l20 = rs_lo.shift(3).rolling(16, min_periods=4).min().bfill()
        prev_h20_w = h.shift(3).rolling(16, min_periods=4).max().bfill()
        prev_l20_w = l.shift(3).rolling(16, min_periods=4).min().bfill()
        rs_div_hi = np.where(
            (h >= prev_h20_w * 0.985) & (rs_hi < prev_rs_h20 * 0.985),
            1.0,
            np.where(rs_hi >= prev_rs_h20, -1.0, 0.0),
        )
        rs_div_lo = np.where(
            (l <= prev_l20_w * 1.015) & (rs_lo > prev_rs_l20 * 1.015),
            1.0,
            np.where(rs_lo <= prev_rs_l20, -1.0, 0.0),
        )
        cols["rs_div_at_pivot"] = np.where(net_above_ema8, rs_div_hi, rs_div_lo)
    else:
        cols["spy_rsi14_high"] = 50.0
        cols["spy_rsi14_low"] = 50.0
        cols["spy_ext_ema20_adr"] = 0.0
        cols["qqq_rsi14_high"] = 50.0
        cols["qqq_rsi14_low"] = 50.0
        cols["spy_thrust_5d_adr"] = 0.0
        cols["spy_drop_5d_adr"] = 0.0
        cols["rs_vs_spy_5d_adr"] = 0.0
        cols["rs_div_at_pivot"] = 0.0

    # --- B3: Developing Weekly (W-FRI) Confluence (Strictly Causal up to day t) ---
    if isinstance(b.index, pd.DatetimeIndex) and len(b) >= 15:
        w_c = c.resample("W-FRI").last().ffill()
        w_h = h.resample("W-FRI").max().ffill()
        w_l = l.resample("W-FRI").min().ffill()
        w_delta = w_c.diff().fillna(0.0)
        w_ag14 = w_delta.clip(lower=0.0).ewm(alpha=1.0 / 14.0, min_periods=3, adjust=False).mean()
        w_al14 = (-w_delta.clip(upper=0.0)).ewm(alpha=1.0 / 14.0, min_periods=3, adjust=False).mean()
        w_ema10 = w_c.ewm(span=10, adjust=False).mean()
        w_prev_c = w_c.shift(1).bfill()
        w_tr = np.maximum(w_h - w_l, np.maximum((w_h - w_prev_c).abs(), (w_l - w_prev_c).abs()))
        w_adr = np.maximum(w_tr.rolling(12, min_periods=2).mean().bfill(), w_c * 0.015)

        w_up_arr = (w_c > w_prev_c).to_numpy(dtype=int)
        w_dn_arr = (w_c < w_prev_c).to_numpy(dtype=int)
        w_up_str = np.zeros(len(w_c))
        w_dn_str = np.zeros(len(w_c))
        ru = rd = 0
        for idx_w in range(len(w_c)):
            ru = (ru + 1) if w_up_arr[idx_w] else 0
            rd = (rd + 1) if w_dn_arr[idx_w] else 0
            w_up_str[idx_w] = ru
            w_dn_str[idx_w] = rd

        w_state = pd.DataFrame(
            {
                "prev_w_c": w_c.shift(1),
                "prev_w_ag14": w_ag14.shift(1),
                "prev_w_al14": w_al14.shift(1),
                "prev_w_ema10": w_ema10.shift(1),
                "prev_w_adr": w_adr.shift(1),
                "prev_w_up": pd.Series(w_up_str, index=w_c.index).shift(1).fillna(0.0),
                "prev_w_dn": pd.Series(w_dn_str, index=w_c.index).shift(1).fillna(0.0),
            }
        )
        w_state.index = w_c.index.to_period("W-FRI")
        wk_per = b.index.to_period("W-FRI")
        mapped_w = w_state.reindex(wk_per)
        mapped_w.index = b.index

        wtd_hi = h.groupby(wk_per).cummax()
        wtd_lo = l.groupby(wk_per).cummin()
        wtd_hi_prev = wtd_hi.groupby(wk_per).shift(1).fillna(-np.inf)
        wtd_lo_prev = wtd_lo.groupby(wk_per).shift(1).fillna(np.inf)

        pw_c = mapped_w["prev_w_c"].bfill().fillna(c)
        pw_ag = mapped_w["prev_w_ag14"].bfill().fillna(0.0)
        pw_al = mapped_w["prev_w_al14"].bfill().fillna(0.0)
        pw_ema10 = mapped_w["prev_w_ema10"].bfill().fillna(c)
        pw_adr = np.maximum(1e-6, mapped_w["prev_w_adr"].bfill().fillna(adr20 * 2.2))
        pw_up = mapped_w["prev_w_up"].fillna(0.0)
        pw_dn = mapped_w["prev_w_dn"].fillna(0.0)

        gh_w = (wtd_hi - pw_c).clip(lower=0.0)
        lh_w = (pw_c - wtd_hi).clip(lower=0.0)
        agh_w = (pw_ag * 13.0 + gh_w) / 14.0
        alh_w = (pw_al * 13.0 + lh_w) / 14.0
        w_rsi14_hi = (100.0 - 100.0 / (1.0 + agh_w / (alh_w + 1e-9))).fillna(b["rsi14_high"])

        gl_w = (wtd_lo - pw_c).clip(lower=0.0)
        ll_w = (pw_c - wtd_lo).clip(lower=0.0)
        agl_w = (pw_ag * 13.0 + gl_w) / 14.0
        all_w = (pw_al * 13.0 + ll_w) / 14.0
        w_rsi14_lo = (100.0 - 100.0 / (1.0 + agl_w / (all_w + 1e-9))).fillna(rsi14_low)

        cols["prev_w_c"] = pw_c
        cols["prev_w_ag14"] = pw_ag
        cols["prev_w_al14"] = pw_al
        cols["prev_w_ema10"] = pw_ema10
        cols["prev_w_adr"] = pw_adr
        cols["wtd_high_prev"] = wtd_hi_prev
        cols["wtd_low_prev"] = wtd_lo_prev
        cols["weekly_rsi14_high"] = w_rsi14_hi
        cols["weekly_rsi14_low"] = w_rsi14_lo
        cols["weekly_ext_ema10w_adr"] = np.where(
            net_above_ema8,
            (wtd_hi - pw_ema10) / pw_adr,
            (wtd_lo - pw_ema10) / pw_adr,
        )
        cols["consec_up_weeks"] = pw_up + (wtd_hi > pw_c).astype(float)
        cols["consec_down_weeks"] = pw_dn + (wtd_lo < pw_c).astype(float)
    else:
        cols["prev_w_c"] = prev_c.bfill()
        cols["prev_w_ag14"] = 0.0
        cols["prev_w_al14"] = 0.0
        cols["prev_w_ema10"] = sma50_anchor
        cols["prev_w_adr"] = adr20 * 2.2
        cols["wtd_high_prev"] = -np.inf
        cols["wtd_low_prev"] = np.inf
        cols["weekly_rsi14_high"] = b["rsi14_high"]
        cols["weekly_rsi14_low"] = rsi14_low
        cols["weekly_ext_ema10w_adr"] = np.where(
            net_above_ema8,
            b["ext_sma50_adr"] / 2.2,
            low_dist_sma50_adr / 2.2,
        )
        cols["consec_up_weeks"] = (b["up_days_10d"] / 2.0).clip(0.0, 10.0)
        cols["consec_down_weeks"] = (cols["down_days_10d"] / 2.0).clip(0.0, 10.0)

    # --- B4: Real-Time Intrabar Partial-Rejection (Sub-Model A+) & Multi-TF (5m..1w) Inv-RSI Confluence ---
    # Deterministic daily/weekly MTF baseline + 5-min intraday enrichment when available
    daily_ob_proxy = (
        (b["rsi5_high"] >= 80.0).astype(float) * 2.0
        + (b["rsi9_high"] >= 76.0).astype(float) * 2.0
        + (b["rsi14_high"] >= 75.0).astype(float) * 2.0
        + (cols["weekly_rsi14_high"] >= 75.0).astype(float)
    ) / 7.0
    daily_os_proxy = (
        (rsi5_low <= 20.0).astype(float) * 2.0
        + (rsi9_low <= 24.0).astype(float) * 2.0
        + (rsi14_low <= 25.0).astype(float) * 2.0
        + (cols["weekly_rsi14_low"] <= 25.0).astype(float)
    ) / 7.0

    micro_fade_default = b["fade_from_high_adr"].clip(0.0, 1.25)
    micro_bounce_default = bounce_from_low_adr.clip(0.0, 1.25)
    pace_rvol_default = b["rvol"].clip(0.1, 15.0)
    micro_up_wick_default = b["upper_wick_pct"].clip(0.0, 100.0)
    micro_lo_wick_default = cols["lower_wick_pct"].clip(0.0, 100.0)

    b4_5m = _extract_b4_intraday_daily_summary(symbol) if symbol else None
    if b4_5m is not None and isinstance(b.index, pd.DatetimeIndex):
        b4_aligned = b4_5m.reindex(b.index.normalize())
        b4_aligned.index = b.index
        has_5m = b4_aligned["raw_post_peak_drop"].notna()

        mf_5m = (b4_aligned["raw_post_peak_drop"] / adr20).clip(0.0, 1.5)
        mb_5m = (b4_aligned["raw_post_trough_bounce"] / adr20).clip(0.0, 1.5)
        pr_hi_5m = (b4_aligned["pace_vol_hi"] / vol_ma20).clip(0.1, 15.0)
        pr_lo_5m = (b4_aligned["pace_vol_lo"] / vol_ma20).clip(0.1, 15.0)
        pr_5m = np.where(net_above_ema8, pr_hi_5m, pr_lo_5m)

        mtf_ob_5m = (
            b4_aligned["intraday_ob75_hits"]
            + (b["rsi14_high"] >= 75.0).astype(float)
            + (cols["weekly_rsi14_high"] >= 75.0).astype(float)
        ) / 7.0
        mtf_os_5m = (
            b4_aligned["intraday_os25_hits"]
            + (rsi14_low <= 25.0).astype(float)
            + (cols["weekly_rsi14_low"] <= 25.0).astype(float)
        ) / 7.0

        cols["micro_fade_adr"] = np.where(has_5m, mf_5m, micro_fade_default)
        cols["micro_bounce_adr"] = np.where(has_5m, mb_5m, micro_bounce_default)
        cols["pace_rvol"] = np.where(has_5m & pd.notna(pr_5m), pr_5m, pace_rvol_default)
        cols["micro_upper_wick_pct"] = np.where(has_5m, b4_aligned["micro_upper_wick_5m"], micro_up_wick_default)
        cols["micro_lower_wick_pct"] = np.where(has_5m, b4_aligned["micro_lower_wick_5m"], micro_lo_wick_default)
        cols["mtf_ob75_confluence"] = np.where(has_5m, mtf_ob_5m, daily_ob_proxy)
        cols["mtf_os25_confluence"] = np.where(has_5m, mtf_os_5m, daily_os_proxy)
    else:
        cols["micro_fade_adr"] = micro_fade_default
        cols["micro_bounce_adr"] = micro_bounce_default
        cols["pace_rvol"] = pace_rvol_default
        cols["micro_upper_wick_pct"] = micro_up_wick_default
        cols["micro_lower_wick_pct"] = micro_lo_wick_default
        cols["mtf_ob75_confluence"] = daily_ob_proxy
        cols["mtf_os25_confluence"] = daily_os_proxy

    return pd.concat([b, pd.DataFrame(cols, index=b.index)], axis=1)


def label_uptrend_pivot_lows(b: pd.DataFrame) -> pd.Series:
    """
    V2 Ground-Truth Uptrend Swing Pivot Low (`1`):
    1. Strictly inside `is_primary_uptrend` and not a fresh Day-1 breakdown (`~is_fresh_breakdown`).
    2. `Low[t]` pulled back into or below EMA10 support zone (`low_dist_ema10_adr <= 0.35`, `drop_15d_adr >= 2.0`).
    3. `Low[t]` is within `0.25 ADR` of the local 17-bar window minimum `[t-8, t+8]`
       AND future 8 bars (`t+1..t+8`) never undercut `Low[t] - 0.25 * ADR20[t]`.
    4. Immediate turn initiation within 5 bars (`fut_max_h5 - curr_l >= 1.5 * ADR20[t]`)
       plus genuine 10-bar upswing resumption (`fut_max_h10 - curr_l >= 2.8 * ADR20[t]` and `fut_max_c10 - curr_l >= 1.8 * ADR20[t]`).
    """
    h = b["high"].to_numpy()
    l = b["low"].to_numpy()
    c = b["close"].to_numpy()
    adr = b["adr20"].to_numpy()
    drop15 = b["drop_15d_adr"].to_numpy()
    low_d10 = b["low_dist_ema10_adr"].to_numpy()
    ut_mask = b["is_primary_uptrend"].to_numpy()
    fresh_bd = b["is_fresh_breakdown"].to_numpy()
    n = len(b)
    labels = np.zeros(n, dtype=int)

    for i in range(10, n - 10):
        if (not ut_mask[i]) or fresh_bd[i] or drop15[i] < 2.0 or low_d10[i] > 0.35:
            continue
        curr_l = l[i]
        curr_adr = adr[i]
        if curr_adr <= 0 or np.isnan(curr_l):
            continue
        past_min = np.min(l[max(0, i - 8) : i + 1])
        fut_min_8 = np.min(l[i + 1 : i + 9])
        if (curr_l - past_min) > 0.25 * curr_adr:
            continue
        if (curr_l - fut_min_8) > 0.25 * curr_adr:
            continue
        fut_max_h5 = np.max(h[i + 1 : i + 6])
        fut_max_h10 = np.max(h[i + 1 : i + 11])
        fut_max_c10 = np.max(c[i + 1 : i + 11])
        if (
            (fut_max_h5 - curr_l) >= 1.5 * curr_adr
            and (fut_max_h10 - curr_l) >= 2.8 * curr_adr
            and (fut_max_c10 - curr_l) >= 1.8 * curr_adr
        ):
            labels[i] = 1

    return pd.Series(labels, index=b.index)


def label_downtrend_pivot_highs(b: pd.DataFrame) -> pd.Series:
    """
    V2 Ground-Truth Downtrend Swing Pivot High (`1` - Bear Rally Top / Lower High Rejection):
    1. Strictly inside `is_primary_downtrend` and not a fresh Day-1 bullish ignition (`~is_fresh_ignition`).
    2. Preceded by a counter-trend bounce (`thrust_10d_adr >= 2.0`, `ext_ema10_adr >= -0.25`).
    3. `High[t]` is within `0.25 ADR` of the local 17-bar window peak `[t-8, t+8]`
       AND future 8 bars (`t+1..t+8`) never exceed `High[t] + 0.25 * ADR20[t]`.
    4. Immediate rollover initiation within 5 bars (`curr_h - fut_min_l5 >= 1.5 * ADR20[t]`)
       plus 10-bar downtrend resumption (`curr_h - fut_min_l10 >= 2.8 * ADR20[t]` and `curr_h - fut_min_c10 >= 1.8 * ADR20[t]`).
    """
    h = b["high"].to_numpy()
    l = b["low"].to_numpy()
    c = b["close"].to_numpy()
    adr = b["adr20"].to_numpy()
    th10 = b["thrust_10d_adr"].to_numpy()
    ext10 = b["ext_ema10_adr"].to_numpy()
    dt_mask = b["is_primary_downtrend"].to_numpy()
    fresh_ig = b["is_fresh_ignition"].to_numpy()
    n = len(b)
    labels = np.zeros(n, dtype=int)

    for i in range(10, n - 10):
        if (not dt_mask[i]) or fresh_ig[i] or th10[i] < 2.0 or ext10[i] < -0.25:
            continue
        curr_h = h[i]
        curr_adr = adr[i]
        if curr_adr <= 0 or np.isnan(curr_h):
            continue
        past_max = np.max(h[max(0, i - 8) : i + 1])
        fut_max_8 = np.max(h[i + 1 : i + 9])
        if (past_max - curr_h) > 0.25 * curr_adr:
            continue
        if (fut_max_8 - curr_h) > 0.25 * curr_adr:
            continue
        fut_min_l5 = np.min(l[i + 1 : i + 6])
        fut_min_l10 = np.min(l[i + 1 : i + 11])
        fut_min_c10 = np.min(c[i + 1 : i + 11])
        if (
            (curr_h - fut_min_l5) >= 1.5 * curr_adr
            and (curr_h - fut_min_l10) >= 2.8 * curr_adr
            and (curr_h - fut_min_c10) >= 1.8 * curr_adr
        ):
            labels[i] = 1

    return pd.Series(labels, index=b.index)


def label_downtrend_pivot_lows(b: pd.DataFrame) -> pd.Series:
    """
    V2 Ground-Truth Downtrend Swing Pivot Low (`1` - Capitulation / Selling Climax Bottom):
    1. Strictly inside `is_primary_downtrend` and not a fresh Day-1 unextended breakdown (`~is_fresh_breakdown`).
    2. Preceded by a real waterfall drop (`drop_10d_adr >= 2.8` or `stretch_below_ema20_max3 >= 1.5`).
    3. `Low[t]` is within `0.25 ADR` of the local 17-bar window minimum `[t-8, t+8]`
       AND future 8 bars (`t+1..t+8`) never undercut `Low[t] - 0.25 * ADR20[t]`.
    4. Immediate snapback initiation within 5 bars (`fut_max_h5 - curr_l >= 1.5 * ADR20[t]`)
       plus 10-bar relief rally (`fut_max_h10 - curr_l >= 2.8 * ADR20[t]` and `fut_max_c10 - curr_l >= 1.8 * ADR20[t]`).
    """
    h = b["high"].to_numpy()
    l = b["low"].to_numpy()
    c = b["close"].to_numpy()
    adr = b["adr20"].to_numpy()
    drop10 = b["drop_10d_adr"].to_numpy()
    str20 = b["stretch_below_ema20_max3"].to_numpy()
    dt_mask = b["is_primary_downtrend"].to_numpy()
    fresh_bd = b["is_fresh_breakdown"].to_numpy()
    n = len(b)
    labels = np.zeros(n, dtype=int)

    for i in range(10, n - 10):
        if (not dt_mask[i]) or fresh_bd[i] or (drop10[i] < 2.8 and str20[i] < 1.5):
            continue
        curr_l = l[i]
        curr_adr = adr[i]
        if curr_adr <= 0 or np.isnan(curr_l):
            continue
        past_min = np.min(l[max(0, i - 8) : i + 1])
        fut_min_8 = np.min(l[i + 1 : i + 9])
        if (curr_l - past_min) > 0.25 * curr_adr:
            continue
        if (curr_l - fut_min_8) > 0.25 * curr_adr:
            continue
        fut_max_h5 = np.max(h[i + 1 : i + 6])
        fut_max_h10 = np.max(h[i + 1 : i + 11])
        fut_max_c10 = np.max(c[i + 1 : i + 11])
        if (
            (fut_max_h5 - curr_l) >= 1.5 * curr_adr
            and (fut_max_h10 - curr_l) >= 2.8 * curr_adr
            and (fut_max_c10 - curr_l) >= 1.8 * curr_adr
        ):
            labels[i] = 1

    return pd.Series(labels, index=b.index)


# ==============================================================================
# V3 BASELINE FEATURE LISTS (DO NOT MODIFY — USED BY EXISTING V3 MODELS)
# ==============================================================================

# 1. UPTREND PIVOT LOW (Pullback Buy / Support Kiss / U&R)
UT_LOW_AT_LOW_FEATURES = [
    "rsi14_low",
    "rsi9_low",
    "rsi5_low",
    "prev_rsi14_low",
    "rsi14_low_min5",
    "rsi5_low_min3",
    "rsi14_low_diff3",
    "rsi14_low_diff5",
    "rsi14_vs_rsi5_low_spread",
    "low_dist_ema8_adr",
    "low_dist_ema10_adr",
    "low_dist_ema20_adr",
    "low_dist_sma50_adr",
    "low_dist_sma200_adr",
    "kiss_ema10_dist_adr",
    "kiss_ema20_dist_adr",
    "kiss_sma50_dist_adr",
    "min_support_kiss_adr",
    "first_ema20_test_in_15d",
    "first_sma50_test_in_25d",
    "days_above_sma50_60d",
    "bb_z_low",
    "bb_z_low_min3",
    "drop_1d_adr",
    "drop_2d_adr",
    "drop_3d_adr",
    "drop_5d_adr",
    "drop_10d_adr",
    "drop_15d_adr",
    "drop_20d_adr",
    "prior_rally_20d_adr",
    "pullback_retracement_ratio",
    "bars_below_ema10",
    "down_leg_adr_from_ema10",
    "down_days_5d",
    "down_days_10d",
    "consec_ll",
    "dist_from_8d_low_adr",
    "undercut_10d_low_adr",
    "ema20_slope_5d_adr",
    "sma50_slope_10d_adr",
    "gap_adr",
    "open_to_low_adr",
    "adr20_pct",
]

UT_LOW_FULL_BAR_FEATURES = UT_LOW_AT_LOW_FEATURES + [
    "rsi14",
    "lower_wick_pct",
    "close_pos",
    "bounce_from_low_adr",
    "range_to_adr",
    "green_reclaim_bar",
    "two_bar_bottom_reclaim",
    "failed_low_undercut",
    "ma_undercut_reclaim",
    "support_confluence_score",
    "pullback_vol_dryup",
    "rvol",
    "hammer_vol_score",
]

# 2. DOWNTREND PIVOT HIGH (Bear Rally Top / Overhead Supply Rejection)
DT_HIGH_AT_HIGH_FEATURES = [
    "rsi14_high",
    "rsi9_high",
    "rsi5_high",
    "prev_rsi14_high",
    "rsi14_high_max5",
    "rsi5_high_max3",
    "rsi14_high_diff3",
    "rsi14_high_diff5",
    "rsi5_vs_rsi14_high_spread",
    "ext_ema8_adr",
    "ext_ema10_adr",
    "ext_ema20_adr",
    "ext_sma50_adr",
    "high_dist_sma50_adr",
    "high_dist_sma200_adr",
    "kiss_falling_ema20_adr",
    "kiss_falling_sma50_adr",
    "min_overhead_kiss_adr",
    "dist_below_40d_high_adr",
    "days_below_sma50_60d",
    "ext_ema10_adr_max3",
    "ext_ema20_adr_max3",
    "bb_z_high",
    "bb_z_high_max3",
    "bars_since_ema10_touch",
    "leg_adr_from_ema10",
    "thrust_1d_adr",
    "thrust_2d_adr",
    "thrust_3d_adr",
    "thrust_5d_adr",
    "thrust_10d_adr",
    "thrust_20d_adr",
    "prior_selloff_20d_adr",
    "bounce_retracement_ratio",
    "dist_from_10d_high_adr",
    "breakout_above_10d_adr",
    "ema20_slope_5d_adr",
    "sma50_slope_10d_adr",
    "up_days_5d",
    "up_days_10d",
    "consec_hh",
    "gap_adr",
    "open_to_high_adr",
    "adr20_pct",
]

DT_HIGH_FULL_BAR_FEATURES = DT_HIGH_AT_HIGH_FEATURES + [
    "rsi14",
    "upper_wick_pct",
    "close_pos",
    "fade_from_high_adr",
    "range_to_adr",
    "red_reversal_bar",
    "two_bar_top_reversal",
    "failed_high_retest",
    "rejected_at_falling_ma",
    "overhead_rejection_score",
    "rvol",
    "reversal_vol_score",
]

# 3. DOWNTREND PIVOT LOW (Capitulation Washout / Selling Climax Bottom)
DT_LOW_AT_LOW_FEATURES = [
    "rsi14_low",
    "rsi9_low",
    "rsi5_low",
    "prev_rsi14_low",
    "rsi14_low_min5",
    "rsi5_low_min3",
    "rsi14_low_diff3",
    "rsi14_low_diff5",
    "rsi14_vs_rsi5_low_spread",
    "consec_rsi30_5d",
    "consec_rsi25_5d",
    "consec_rsi5_20_5d",
    "stretch_below_ema10_adr",
    "stretch_below_ema20_adr",
    "stretch_below_sma50_adr",
    "stretch_below_ema20_max3",
    "prev_stretch_below_ema20_adr",
    "low_dist_sma200_adr",
    "days_below_sma50_60d",
    "bb_z_low",
    "bb_z_low_min3",
    "rel_down_ema20_p95",
    "bars_below_ema10",
    "bars_below_ema20",
    "down_leg_adr_from_ema10",
    "down_leg_adr_from_ema20",
    "drop_1d_adr",
    "drop_2d_adr",
    "drop_3d_adr",
    "drop_5d_adr",
    "drop_10d_adr",
    "drop_20d_adr",
    "drop_40d_adr",
    "waterfall_accel",
    "dist_from_8d_low_adr",
    "undercut_10d_low_adr",
    "undercut_20d_low_adr",
    "gap_adr",
    "open_to_low_adr",
    "gap_down_on_oversold",
    "down_days_5d",
    "down_days_10d",
    "consec_ll",
    "bullish_div_low_18d",
    "rsi_bull_div_delta_18d",
    "bullish_div_low_35d",
    "rsi_bull_div_delta_35d",
    "adr20_pct",
]

DT_LOW_FULL_BAR_FEATURES = DT_LOW_AT_LOW_FEATURES + [
    "rsi14",
    "lower_wick_pct",
    "close_pos",
    "bounce_from_low_adr",
    "range_to_adr",
    "green_reclaim_bar",
    "two_bar_bottom_reclaim",
    "failed_low_undercut",
    "rvol",
    "hammer_vol_score",
    "capitulation_vol_stretch",
    "climax_washout_score",
    "seller_exhaustion_wick",
    "day2_washout_reclaim",
]

# ==============================================================================
# V4 ENHANCED FEATURE FAMILIES (B1, B2, B3, B4 + INVERSE WILDER'S RSI)
# ==============================================================================
B1_HIGH_FEATURES = [
    "dist_to_prior_swing_high_adr",
    "dist_to_prior_swing_low_adr",
    "avwap_from_60d_high_dist_adr",
    "avwap_from_60d_low_dist_adr",
    "overhead_volume_density_60d",
    "inv_rsi75_dist_adr",
]
B1_LOW_FEATURES = [
    "dist_to_prior_swing_high_adr",
    "dist_to_prior_swing_low_adr",
    "avwap_from_60d_high_dist_adr",
    "avwap_from_60d_low_dist_adr",
    "overhead_volume_density_60d",
    "inv_rsi25_dist_adr",
]
B2_SYNC_FEATURES = [
    "spy_rsi14_high",
    "spy_rsi14_low",
    "spy_ext_ema20_adr",
    "qqq_rsi14_high",
    "qqq_rsi14_low",
    "rs_vs_spy_5d_adr",
    "rs_div_at_pivot",
]
B3_WEEKLY_FEATURES = [
    "weekly_rsi14_high",
    "weekly_rsi14_low",
    "weekly_ext_ema10w_adr",
    "consec_up_weeks",
    "consec_down_weeks",
]
B4_HIGH_A_PLUS_ADDONS = [
    "micro_fade_adr",
    "pace_rvol",
    "micro_upper_wick_pct",
    "mtf_ob75_confluence",
]
B4_LOW_A_PLUS_ADDONS = [
    "micro_bounce_adr",
    "pace_rvol",
    "micro_lower_wick_pct",
    "mtf_os25_confluence",
]

# V4 Feature Sets for Model 1 (Uptrend Pivot High)
UT_HIGH_V4_AT_HIGH_FEATURES = tup.AT_HIGH_FEATURES + B1_HIGH_FEATURES + B2_SYNC_FEATURES + B3_WEEKLY_FEATURES + ["mtf_ob75_confluence"]
UT_HIGH_V4_A_PLUS_FEATURES = UT_HIGH_V4_AT_HIGH_FEATURES + ["micro_fade_adr", "pace_rvol", "micro_upper_wick_pct"]
UT_HIGH_V4_FULL_BAR_FEATURES = tup.FULL_BAR_FEATURES + B1_HIGH_FEATURES + B2_SYNC_FEATURES + B3_WEEKLY_FEATURES + B4_HIGH_A_PLUS_ADDONS

# V4 Feature Sets for Model 2 (Uptrend Pivot Low)
UT_LOW_V4_AT_LOW_FEATURES = UT_LOW_AT_LOW_FEATURES + B1_LOW_FEATURES + B2_SYNC_FEATURES + B3_WEEKLY_FEATURES + ["mtf_os25_confluence"]
UT_LOW_V4_A_PLUS_FEATURES = UT_LOW_V4_AT_LOW_FEATURES + ["micro_bounce_adr", "pace_rvol", "micro_lower_wick_pct"]
UT_LOW_V4_FULL_BAR_FEATURES = UT_LOW_FULL_BAR_FEATURES + B1_LOW_FEATURES + B2_SYNC_FEATURES + B3_WEEKLY_FEATURES + B4_LOW_A_PLUS_ADDONS

# V4 Feature Sets for Model 3 (Downtrend Pivot High)
DT_HIGH_V4_AT_HIGH_FEATURES = DT_HIGH_AT_HIGH_FEATURES + B1_HIGH_FEATURES + B2_SYNC_FEATURES + B3_WEEKLY_FEATURES + ["mtf_ob75_confluence"]
DT_HIGH_V4_A_PLUS_FEATURES = DT_HIGH_V4_AT_HIGH_FEATURES + ["micro_fade_adr", "pace_rvol", "micro_upper_wick_pct"]
DT_HIGH_V4_FULL_BAR_FEATURES = DT_HIGH_FULL_BAR_FEATURES + B1_HIGH_FEATURES + B2_SYNC_FEATURES + B3_WEEKLY_FEATURES + B4_HIGH_A_PLUS_ADDONS

# V4 Feature Sets for Model 4 (Downtrend Pivot Low)
DT_LOW_V4_AT_LOW_FEATURES = DT_LOW_AT_LOW_FEATURES + B1_LOW_FEATURES + B2_SYNC_FEATURES + B3_WEEKLY_FEATURES + ["mtf_os25_confluence"]
DT_LOW_V4_A_PLUS_FEATURES = DT_LOW_V4_AT_LOW_FEATURES + ["micro_bounce_adr", "pace_rvol", "micro_lower_wick_pct"]
DT_LOW_V4_FULL_BAR_FEATURES = DT_LOW_FULL_BAR_FEATURES + B1_LOW_FEATURES + B2_SYNC_FEATURES + B3_WEEKLY_FEATURES + B4_LOW_A_PLUS_ADDONS


def _fit_lgb_xgb_pair(
    X_tr: pd.DataFrame, y_tr: np.ndarray, spw: float, n_est: int = 280
) -> Tuple[lgb.LGBMClassifier, XGBClassifier]:
    m_lgb = lgb.LGBMClassifier(
        n_estimators=n_est,
        max_depth=5,
        num_leaves=26,
        learning_rate=0.035,
        min_child_samples=60,
        scale_pos_weight=spw,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    m_xgb = XGBClassifier(
        n_estimators=n_est,
        max_depth=4,
        learning_rate=0.035,
        scale_pos_weight=spw,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )
    m_lgb.fit(X_tr, y_tr)
    m_xgb.fit(X_tr, y_tr)
    return m_lgb, m_xgb


def train_and_evaluate_v4_model(
    name: str,
    df_regime: pd.DataFrame,
    target_col: str,
    mask_col: str,
    v3_ex_cols: List[str],
    v3_fb_cols: List[str],
    v4_ex_cols: List[str],
    v4_ap_cols: List[str],
    v4_fb_cols: List[str],
    spw_extreme: float = 3.2,
    spw_a_plus: float = 3.0,
    spw_full: float = 2.8,
) -> Dict[str, Any]:
    """
    Runs full chronological OOS ablation across Enhancement Families B1..B5 (<2025 Train vs 2025-2026 OOS Test)
    and returns the trained V4 Tri-Path (Sub-Model A + Sub-Model A+ + Sub-Model B) + OOF Isotonic Calibrated bundle.
    """
    print("\n" + "=" * 96)
    print(f"🚀 V4 ABLATION & TRAINING (B1–B5 + INVERSE-RSI, STRICT OOS 2025–2026): {name}")
    print("=" * 96)

    pos_count = int(df_regime[target_col].sum())
    base_rate = float(df_regime[target_col].mean())
    print(f"Regime Bars: {len(df_regime):,} | Positive Pivots: {pos_count:,} ({base_rate*100:.2f}% base rate)")

    train_df = df_regime[df_regime["date_dt"] < "2025-01-01"].reset_index(drop=True)
    test_df = df_regime[df_regime["date_dt"] >= "2025-01-01"].reset_index(drop=True)
    print(f"Strict Chronological Split -> Train (<2025-01-01): {len(train_df):,} | Out-of-Sample Test (2025-2026): {len(test_df):,}")

    y_tr = train_df[target_col].to_numpy()
    y_te = test_df[target_col].to_numpy()
    mask_tr = train_df[mask_col].to_numpy()
    mask_te = test_df[mask_col].to_numpy()

    def _eval_cols(cols_ex: List[str], cols_fb: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Xtr_e = train_df[cols_ex].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        Xte_e = test_df[cols_ex].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        Xtr_f = train_df[cols_fb].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        Xte_f = test_df[cols_fb].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        le, xe = _fit_lgb_xgb_pair(Xtr_e, y_tr, spw_extreme, n_est=240)
        lf, xf = _fit_lgb_xgb_pair(Xtr_f, y_tr, spw_full, n_est=240)
        pe = 0.55 * le.predict_proba(Xte_e)[:, 1] + 0.45 * xe.predict_proba(Xte_e)[:, 1]
        pf = 0.55 * lf.predict_proba(Xte_f)[:, 1] + 0.45 * xf.predict_proba(Xte_f)[:, 1]
        pd_comb = 0.25 * np.minimum(pe, pf) + 0.75 * np.maximum(pe, pf)
        return np.where(mask_te, pe, 0.0), np.where(mask_te, pf, 0.0), np.where(mask_te, pd_comb, 0.0)

    is_hi = "high" in target_col or "high" in mask_col
    b1_cols = B1_HIGH_FEATURES if is_hi else B1_LOW_FEATURES

    # 1. V3 Baseline
    p_v3_ex, p_v3_fb, p_v3_dual = _eval_cols(v3_ex_cols, v3_fb_cols)
    # 2. + B1 (Overhead Supply, AVWAP, Swings, Inverse-RSI)
    p_b1_ex, p_b1_fb, p_b1_dual = _eval_cols(v3_ex_cols + b1_cols, v3_fb_cols + b1_cols)
    # 3. + B1 + B2 (Market SPY/QQQ & RS Synchrony)
    p_b2_ex, p_b2_fb, p_b2_dual = _eval_cols(
        v3_ex_cols + b1_cols + B2_SYNC_FEATURES,
        v3_fb_cols + b1_cols + B2_SYNC_FEATURES,
    )
    # 4. + B1 + B2 + B3 (Developing Weekly W-FRI Confluence)
    p_b3_ex, p_b3_fb, p_b3_dual = _eval_cols(
        v3_ex_cols + b1_cols + B2_SYNC_FEATURES + B3_WEEKLY_FEATURES,
        v3_fb_cols + b1_cols + B2_SYNC_FEATURES + B3_WEEKLY_FEATURES,
    )

    # 5. Train Full V4 Tri-Path Ensemble: Sub-Model A (At-Extreme) + Sub-Model A+ (Intrabar Micro-Rejection) + Sub-Model B (Full Bar)
    X_tr_ex = train_df[v4_ex_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_te_ex = test_df[v4_ex_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_tr_ap = train_df[v4_ap_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_te_ap = test_df[v4_ap_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_tr_fb = train_df[v4_fb_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_te_fb = test_df[v4_fb_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    lgb_ex, xgb_ex = _fit_lgb_xgb_pair(X_tr_ex, y_tr, spw_extreme, n_est=280)
    lgb_ap, xgb_ap = _fit_lgb_xgb_pair(X_tr_ap, y_tr, spw_a_plus, n_est=280)
    lgb_fb, xgb_fb = _fit_lgb_xgb_pair(X_tr_fb, y_tr, spw_full, n_est=280)

    p_ex_te_raw = 0.55 * lgb_ex.predict_proba(X_te_ex)[:, 1] + 0.45 * xgb_ex.predict_proba(X_te_ex)[:, 1]
    p_ap_te_raw = 0.55 * lgb_ap.predict_proba(X_te_ap)[:, 1] + 0.45 * xgb_ap.predict_proba(X_te_ap)[:, 1]
    p_fb_te_raw = 0.55 * lgb_fb.predict_proba(X_te_fb)[:, 1] + 0.45 * xgb_fb.predict_proba(X_te_fb)[:, 1]

    # Combine Sub-Model A, Sub-Model A+, and Sub-Model B
    p_ex_plus_te = 0.45 * p_ex_te_raw + 0.55 * p_ap_te_raw
    p_v4_dual_raw = 0.25 * np.minimum(p_ex_plus_te, p_fb_te_raw) + 0.75 * np.maximum(p_ex_plus_te, p_fb_te_raw)

    p_v4_ex = np.where(mask_te, p_ex_te_raw, 0.0)
    p_v4_ap = np.where(mask_te, p_ap_te_raw, 0.0)
    p_v4_fb = np.where(mask_te, p_fb_te_raw, 0.0)
    p_v4_dual = np.where(mask_te, p_v4_dual_raw, 0.0)

    # 6. B5: Out-of-Fold Isotonic Probability Calibration (strictly within <2025-01-01 training data)
    kf = KFold(n_splits=4, shuffle=False)
    oof_dual = np.zeros(len(train_df), dtype=np.float64)
    for tr_idx, val_idx in kf.split(X_tr_fb):
        le_k, xe_k = _fit_lgb_xgb_pair(X_tr_ex.iloc[tr_idx], y_tr[tr_idx], spw_extreme, n_est=160)
        la_k, xa_k = _fit_lgb_xgb_pair(X_tr_ap.iloc[tr_idx], y_tr[tr_idx], spw_a_plus, n_est=160)
        lf_k, xf_k = _fit_lgb_xgb_pair(X_tr_fb.iloc[tr_idx], y_tr[tr_idx], spw_full, n_est=160)
        pe_v = 0.55 * le_k.predict_proba(X_tr_ex.iloc[val_idx])[:, 1] + 0.45 * xe_k.predict_proba(X_tr_ex.iloc[val_idx])[:, 1]
        pa_v = 0.55 * la_k.predict_proba(X_tr_ap.iloc[val_idx])[:, 1] + 0.45 * xa_k.predict_proba(X_tr_ap.iloc[val_idx])[:, 1]
        pf_v = 0.55 * lf_k.predict_proba(X_tr_fb.iloc[val_idx])[:, 1] + 0.45 * xf_k.predict_proba(X_tr_fb.iloc[val_idx])[:, 1]
        pep_v = 0.45 * pe_v + 0.55 * pa_v
        oof_dual[val_idx] = 0.25 * np.minimum(pep_v, pf_v) + 0.75 * np.maximum(pep_v, pf_v)

    iso_cal = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    valid_oof = mask_tr.astype(bool)
    iso_cal.fit(oof_dual[valid_oof], y_tr[valid_oof])
    p_v4_cal = np.where(mask_te, iso_cal.predict(p_v4_dual_raw), 0.0)

    brier_raw = brier_score_loss(y_te[mask_te], p_v4_dual_raw[mask_te])
    brier_cal = brier_score_loss(y_te[mask_te], p_v4_cal[mask_te])

    print("\n  📊 STRICT OOS (2025-2026) STEPWISE ABLATION TABLE:")
    for lbl, pr_ex, pr_du in [
        ("1. V3 Baseline (Daily OHLCV Only)", p_v3_ex, p_v3_dual),
        ("2. + B1 (Swings, 60D AVWAP, Vol Density, Inv-RSI)", p_b1_ex, p_b1_dual),
        ("3. + B2 (SPY/QQQ Synchrony & RS Divergence)", p_b2_ex, p_b2_dual),
        ("4. + B3 (Developing Weekly W-FRI Confluence)", p_b3_ex, p_b3_dual),
        ("5. + B4 Sub-Model A+ (Intrabar Micro-Rej + MTF)", p_v4_ap, p_v4_dual),
    ]:
        auc_du = roc_auc_score(y_te, pr_du)
        ap_du = average_precision_score(y_te, pr_du)
        ap_ex = average_precision_score(y_te, pr_ex)
        p50 = precision_score(y_te, (pr_du >= 0.50).astype(int), zero_division=0)
        r50 = recall_score(y_te, (pr_du >= 0.50).astype(int), zero_division=0)
        p60 = precision_score(y_te, (pr_du >= 0.60).astype(int), zero_division=0)
        r60 = recall_score(y_te, (pr_du >= 0.60).astype(int), zero_division=0)
        print(
            f"  {lbl:50s} | ROC-AUC: {auc_du:.4f} | Dual PR-AUC: {ap_du:.4f} | Sub-A/A+ PR-AUC: {ap_ex:.4f} "
            f"| @50% P/R: {p50*100:.1f}%/{r50*100:.1f}% | @60% P/R: {p60*100:.1f}%/{r60*100:.1f}%"
        )

    print(
        f"  6. + B5 OOF Isotonic Calibration                   | Masked Brier Loss: {brier_raw:.4f} (raw) -> {brier_cal:.4f} (calibrated) "
        f"| Empirical Precision @ Score=50%: {float(iso_cal.predict([0.50])[0])*100:.1f}%, @ Score=65%: {float(iso_cal.predict([0.65])[0])*100:.1f}%"
    )

    imp_fb = pd.Series(lgb_fb.feature_importances_, index=v4_fb_cols).sort_values(ascending=False)
    print(f"\n  Top 12 V4 Features ({name}):")
    for k, val in imp_fb.head(12).items():
        print(f"    {k:34s}: {val}")

    return {
        "name": name,
        "version": "4.0",
        "mask_col": mask_col,
        "lgb_extreme": lgb_ex,
        "xgb_extreme": xgb_ex,
        "lgb_a_plus": lgb_ap,
        "xgb_a_plus": xgb_ap,
        "lgb_full": lgb_fb,
        "xgb_full": xgb_fb,
        "iso_calibrator": iso_cal,
        "at_extreme_features": v4_ex_cols,
        "a_plus_features": v4_ap_cols,
        "full_bar_features": v4_fb_cols,
        "default_threshold_pct": 50.0,
        "trained_date": "2026-10-14",
        "train_cutoff": "2024-12-31",
        "oos_start": "2025-01-01",
        "n_train_samples": len(train_df),
        "n_oos_samples": len(test_df),
        "n_samples": len(df_regime),
    }


def train_and_evaluate_dual_model(
    name: str,
    df_regime: pd.DataFrame,
    target_col: str,
    mask_col: str,
    at_extreme_cols: List[str],
    full_bar_cols: List[str],
    spw_extreme: float = 2.6,
    spw_full: float = 2.4,
) -> Dict[str, Any]:
    print("\n" + "=" * 88)
    print(f"🚀 TRAINING & VALIDATING (V3 STRICT OOS): {name}")
    print("=" * 88)

    pos_count = int(df_regime[target_col].sum())
    base_rate = float(df_regime[target_col].mean())
    print(f"Regime Bars: {len(df_regime):,} | Positive Pivots: {pos_count:,} ({base_rate*100:.2f}% base rate)")

    train_df = df_regime[df_regime["date_dt"] < "2025-01-01"]
    test_df = df_regime[df_regime["date_dt"] >= "2025-01-01"]
    print(f"Strict Chronological Split -> Train (<2025-01-01): {len(train_df):,} | Out-of-Sample Test (2025-2026): {len(test_df):,}")

    X_tr_ex = train_df[at_extreme_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_tr = train_df[target_col].to_numpy()
    X_te_ex = test_df[at_extreme_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_te = test_df[target_col].to_numpy()

    X_tr_fb = train_df[full_bar_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_te_fb = test_df[full_bar_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mask_te = test_df[mask_col].to_numpy()

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=400))
    lr.fit(X_tr_ex, y_tr)
    p_lr = np.where(mask_te, lr.predict_proba(X_te_ex)[:, 1], 0.0)

    lgb_ex, xgb_ex = _fit_lgb_xgb_pair(X_tr_ex, y_tr, spw_extreme, n_est=260)
    p_ex_raw = 0.55 * lgb_ex.predict_proba(X_te_ex)[:, 1] + 0.45 * xgb_ex.predict_proba(X_te_ex)[:, 1]
    p_ex = np.where(mask_te, p_ex_raw, 0.0)

    lgb_fb, xgb_fb = _fit_lgb_xgb_pair(X_tr_fb, y_tr, spw_full, n_est=260)
    p_fb_raw = 0.55 * lgb_fb.predict_proba(X_te_fb)[:, 1] + 0.45 * xgb_fb.predict_proba(X_te_fb)[:, 1]
    p_fb = np.where(mask_te, p_fb_raw, 0.0)

    p_dual = np.where(mask_te, 0.25 * np.minimum(p_ex_raw, p_fb_raw) + 0.75 * np.maximum(p_ex_raw, p_fb_raw), 0.0)

    for lbl, probs in [
        ("Baseline (Logistic Regression At-Extreme)", p_lr),
        ("Sub-Model A (Pure At-Extreme Stretch/Support)", p_ex),
        ("Sub-Model B (Full Bar + Wick/Reclaim + Vol)", p_fb),
        ("Dual-Path Combined Ensemble (V3)", p_dual),
    ]:
        auc_v = roc_auc_score(y_te, probs)
        pr_v = average_precision_score(y_te, probs)
        p50 = precision_score(y_te, (probs >= 0.50).astype(int), zero_division=0)
        r50 = recall_score(y_te, (probs >= 0.50).astype(int), zero_division=0)
        p60 = precision_score(y_te, (probs >= 0.60).astype(int), zero_division=0)
        r60 = recall_score(y_te, (probs >= 0.60).astype(int), zero_division=0)
        print(
            f"  {lbl:46s} | ROC-AUC: {auc_v:.4f} | PR-AUC: {pr_v:.4f} "
            f"| @50% P/R: {p50*100:.1f}%/{r50*100:.1f}% | @60% P/R: {p60*100:.1f}%/{r60*100:.1f}%"
        )

    imp_fb = pd.Series(lgb_fb.feature_importances_, index=full_bar_cols).sort_values(ascending=False)
    print(f"\n  Top 10 Features ({name}):")
    for k, v in imp_fb.head(10).items():
        print(f"    {k:32s}: {v}")

    return {
        "name": name,
        "version": "3.0",
        "mask_col": mask_col,
        "lgb_extreme": lgb_ex,
        "xgb_extreme": xgb_ex,
        "lgb_full": lgb_fb,
        "xgb_full": xgb_fb,
        "at_extreme_features": at_extreme_cols,
        "full_bar_features": full_bar_cols,
        "trained_date": "2026-10-14",
        "train_cutoff": "2024-12-31",
        "oos_start": "2025-01-01",
        "n_train_samples": len(train_df),
        "n_oos_samples": len(test_df),
        "n_samples": len(df_regime),
    }


def score_dual_model_on_features(
    b: pd.DataFrame, model_bundle: Dict[str, Any], mask_col: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates a Dual-Path (V3) or Tri-Path (V4 with Sub-Model A+) model bundle on precomputed feature DataFrame `b`.
    Returns `(prob_dual, prob_extreme, prob_full)`.
    """
    ex_cols = model_bundle.get("at_extreme_features") or model_bundle.get("at_high_features")
    fb_cols = model_bundle["full_bar_features"]

    lgb_ex = model_bundle.get("lgb_extreme") or model_bundle.get("lgb_high") or model_bundle.get("lgb_at_high")
    xgb_ex = model_bundle.get("xgb_extreme") or model_bundle.get("xgb_high") or model_bundle.get("xgb_at_high")
    lgb_fb = model_bundle["lgb_full"]
    xgb_fb = model_bundle["xgb_full"]

    X_ex = b[ex_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_fb = b[fb_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    p_ex_raw = 0.55 * lgb_ex.predict_proba(X_ex)[:, 1] + 0.45 * xgb_ex.predict_proba(X_ex)[:, 1]
    p_fb_raw = 0.55 * lgb_fb.predict_proba(X_fb)[:, 1] + 0.45 * xgb_fb.predict_proba(X_fb)[:, 1]

    # V4 Sub-Model A+ (Real-Time Intrabar Partial-Rejection Path)
    if "lgb_a_plus" in model_bundle and "xgb_a_plus" in model_bundle and "a_plus_features" in model_bundle:
        ap_cols = model_bundle["a_plus_features"]
        X_ap = b[ap_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        p_ap_raw = (
            0.55 * model_bundle["lgb_a_plus"].predict_proba(X_ap)[:, 1]
            + 0.45 * model_bundle["xgb_a_plus"].predict_proba(X_ap)[:, 1]
        )
        p_ex_combined = 0.45 * p_ex_raw + 0.55 * p_ap_raw
    else:
        p_ex_combined = p_ex_raw

    p_dual_raw = 0.25 * np.minimum(p_ex_combined, p_fb_raw) + 0.75 * np.maximum(p_ex_combined, p_fb_raw)

    mask = b[mask_col].to_numpy()
    return (
        np.where(mask, p_dual_raw, 0.0),
        np.where(mask, p_ex_combined, 0.0),
        np.where(mask, p_fb_raw, 0.0),
    )


def score_calibrated_probability(
    prob_dual: np.ndarray, model_bundle: Optional[Dict[str, Any]]
) -> np.ndarray:
    """Applies the Out-of-Fold Isotonic Calibrator (B5) if present in `model_bundle`, returning empirical precision."""
    if not model_bundle or "iso_calibrator" not in model_bundle or model_bundle["iso_calibrator"] is None:
        return prob_dual
    iso = model_bundle["iso_calibrator"]
    nz = prob_dual > 0.0
    out = np.zeros_like(prob_dual, dtype=np.float64)
    if np.any(nz):
        out[nz] = iso.predict(prob_dual[nz])
    return out


def main() -> None:
    t0 = time.time()
    rs_df = pd.read_parquet("data/simulations/rs_universe_scores.parquet")
    sort_col = "rs_score" if "rs_score" in rs_df.columns else rs_df.columns[0]
    top_rs = rs_df.sort_values(sort_col, ascending=False)["symbol"].head(400).tolist()
    bot_rs = rs_df.sort_values(sort_col, ascending=True)["symbol"].head(200).tolist()
    ep_df = pd.read_parquet("data/simulations/optimized_ep_strategy_trades.parquet")
    ep_syms = ep_df["symbol"].unique().tolist()[:250]
    etf_syms = sorted(datastore.app_critical_etfs())
    must_include = [
        "TWLO", "PLTR", "DINO", "MRNA", "PENG", "ANF", "FROG", "PTC",
        "WK", "CCJ", "GRAL", "TEAM", "SBS", "AMZN", "NVDA", "SMH",
        "XBI", "QQQ", "SPY", "XLK", "XLE", "RSP", "IWM", "DIA", "MDY",
    ]
    symbols = list(dict.fromkeys(must_include + etf_syms + top_rs + bot_rs + ep_syms))[:850]
    print(f"Extracting V4 unified 4-regime pivot features (including B1–B4 + Inverse-RSI) across {len(symbols)} symbols...")

    ut_high_frames = []
    ut_low_frames = []
    dt_high_frames = []
    dt_low_frames = []

    for idx_s, sym in enumerate(symbols):
        df = datastore.load_bars(sym)
        if df is None or len(df) < 120:
            continue
        b = compute_all_pivot_features(df, symbol=sym)
        meta = pd.DataFrame(
            {
                "is_ut_high": tup.label_pivot_highs(b),
                "is_ut_low": label_uptrend_pivot_lows(b),
                "is_dt_high": label_downtrend_pivot_highs(b),
                "is_dt_low": label_downtrend_pivot_lows(b),
                "symbol": sym,
                "date": b.index,
            },
            index=b.index,
        )
        b = pd.concat([b, meta], axis=1)
        valid = b.iloc[60:-10]

        c_ut_high = valid[valid["is_uptrend_bar"]].copy()
        if not c_ut_high.empty:
            ut_high_frames.append(c_ut_high)

        c_ut_low = valid[valid["is_uptrend_pullback_regime"]].copy()
        if not c_ut_low.empty:
            ut_low_frames.append(c_ut_low)

        c_dt = valid[valid["is_downtrend_regime"]].copy()
        if not c_dt.empty:
            dt_high_frames.append(c_dt)
            dt_low_frames.append(c_dt)

        if (idx_s + 1) % 200 == 0:
            print(f"  Processed {idx_s + 1}/{len(symbols)} symbols ({time.time() - t0:.1f}s)...")

    df_ut_high = pd.concat(ut_high_frames, ignore_index=True)
    df_ut_high["date_dt"] = pd.to_datetime(df_ut_high["date"])

    df_ut_low = pd.concat(ut_low_frames, ignore_index=True)
    df_ut_low["date_dt"] = pd.to_datetime(df_ut_low["date"])

    df_dt_high = pd.concat(dt_high_frames, ignore_index=True)
    df_dt_high["date_dt"] = pd.to_datetime(df_dt_high["date"])

    df_dt_low = pd.concat(dt_low_frames, ignore_index=True)
    df_dt_low["date_dt"] = pd.to_datetime(df_dt_low["date"])

    # Train & Evaluate all 4 V4 models (saving to *_v4.pkl, preserving existing V3 *.pkl models!)
    m_ut_high_v4 = train_and_evaluate_v4_model(
        "Model 1: Uptrend Pivot High V4 (Climax / Multi-Wave Top)",
        df_ut_high,
        "is_ut_high",
        "is_near_local_high",
        tup.AT_HIGH_FEATURES,
        tup.FULL_BAR_FEATURES,
        UT_HIGH_V4_AT_HIGH_FEATURES,
        UT_HIGH_V4_A_PLUS_FEATURES,
        UT_HIGH_V4_FULL_BAR_FEATURES,
        spw_extreme=3.2,
        spw_a_plus=2.9,
        spw_full=2.6,
    )
    m_ut_low_v4 = train_and_evaluate_v4_model(
        "Model 2: Uptrend Pivot Low V4 (Pullback Buy / Support Reclaim)",
        df_ut_low,
        "is_ut_low",
        "is_near_uptrend_low",
        UT_LOW_AT_LOW_FEATURES,
        UT_LOW_FULL_BAR_FEATURES,
        UT_LOW_V4_AT_LOW_FEATURES,
        UT_LOW_V4_A_PLUS_FEATURES,
        UT_LOW_V4_FULL_BAR_FEATURES,
        spw_extreme=3.5,
        spw_a_plus=3.3,
        spw_full=3.1,
    )
    m_dt_high_v4 = train_and_evaluate_v4_model(
        "Model 3: Downtrend Pivot High V4 (Bear Rally Rejection / Lower High)",
        df_dt_high,
        "is_dt_high",
        "is_near_downtrend_high",
        DT_HIGH_AT_HIGH_FEATURES,
        DT_HIGH_FULL_BAR_FEATURES,
        DT_HIGH_V4_AT_HIGH_FEATURES,
        DT_HIGH_V4_A_PLUS_FEATURES,
        DT_HIGH_V4_FULL_BAR_FEATURES,
        spw_extreme=3.6,
        spw_a_plus=3.4,
        spw_full=3.2,
    )
    m_dt_low_v4 = train_and_evaluate_v4_model(
        "Model 4: Downtrend Pivot Low V4 (Capitulation Washout / Climax Bottom)",
        df_dt_low,
        "is_dt_low",
        "is_near_downtrend_low",
        DT_LOW_AT_LOW_FEATURES,
        DT_LOW_FULL_BAR_FEATURES,
        DT_LOW_V4_AT_LOW_FEATURES,
        DT_LOW_V4_A_PLUS_FEATURES,
        DT_LOW_V4_FULL_BAR_FEATURES,
        spw_extreme=3.4,
        spw_a_plus=3.2,
        spw_full=3.0,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(m_ut_high_v4, UT_HIGH_V4_PATH)
    joblib.dump(m_ut_low_v4, UT_LOW_V4_PATH)
    joblib.dump(m_dt_high_v4, DT_HIGH_V4_PATH)
    joblib.dump(m_dt_low_v4, DT_LOW_V4_PATH)
    print(f"\n✅ Saved all 4 V4 strict-OOS models (*_v4.pkl, preserving V3 *.pkl models) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

