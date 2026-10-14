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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import datastore
import train_uptrend_pivot_model as tup

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

OUT_DIR = Path("data/models")
UT_HIGH_PATH = OUT_DIR / "uptrend_pivot_high_classifier.pkl"
UT_LOW_PATH = OUT_DIR / "uptrend_pivot_low_classifier.pkl"
DT_HIGH_PATH = OUT_DIR / "downtrend_pivot_high_classifier.pkl"
DT_LOW_PATH = OUT_DIR / "downtrend_pivot_low_classifier.pkl"


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


def compute_all_pivot_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unified vectorized feature extractor for all 4 structural pivot models:
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
    cols["dist_below_40d_high_adr"] = ((h.shift(5).rolling(40, min_periods=10).max() - h) / adr20).fillna(2.0)

    # Bollinger Z-score at Low[t] & 120D Self-Normalized Downside Stretch
    bb_z_low = ((l - b["sma20"].shift(1)) / b["std20"].shift(1)).clip(-6.0, 4.0).fillna(0.0)
    cols["bb_z_low"] = bb_z_low
    cols["bb_z_low_min3"] = bb_z_low.rolling(3, min_periods=1).min()

    roll_p95_down20 = stretch_below_ema20_adr.shift(1).rolling(120, min_periods=25).quantile(0.95).clip(lower=1.5).bfill()
    cols["rel_down_ema20_p95"] = stretch_below_ema20_adr / roll_p95_down20

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
    is_nl_18 = l <= prev_wave_low_18 * 1.015
    cols["bullish_div_low_18d"] = (is_nl_18 & (rsi14_low > prev_wave_rsi_low_18 + 1.5)).astype(float)
    cols["rsi_bull_div_delta_18d"] = np.where(is_nl_18, (rsi14_low - prev_wave_rsi_low_18).clip(lower=0.0), 0.0)

    prev_wave_low_35 = l.shift(4).rolling(32, min_periods=8).min()
    prev_wave_rsi_low_35 = rsi14_low.shift(4).rolling(32, min_periods=8).min()
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
    # Primary Uptrend vs Primary Downtrend are strictly complementary (~is_primary_uptrend == is_primary_downtrend)
    is_primary_uptrend = (
        (b["ema20"] >= b["sma50"])
        | ((c >= b["sma50"]) & ((sma50_slope_10d_adr >= -0.05) | (b["ema10"] >= b["sma50"])))
    )
    is_primary_downtrend = ~is_primary_uptrend
    cols["is_primary_uptrend"] = is_primary_uptrend
    cols["is_primary_downtrend"] = is_primary_downtrend

    # Net excursion vs EMA8 separates High-candidates from Low-candidates on wide volatile bars
    net_above_ema8 = (h - ema8_anchor) > (ema8_anchor - l)
    net_below_ema8 = ~net_above_ema8

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

    # Near a Downtrend Bounce High (for Model 3: Downtrend Pivot High)
    cols["is_near_downtrend_high"] = (
        is_primary_downtrend
        & (~b["is_fresh_ignition"])
        & net_above_ema8
        & (b["ext_ema10_adr"] >= -0.25)
        & (b["rsi14_high"] >= 48.0)
        & (((roll_h8 - h) / adr20) <= 0.45)
        & (b["thrust_10d_adr"] >= 2.0)
    )

    # Near a Downtrend Waterfall Low (for Model 4: Downtrend Pivot Low)
    cols["is_near_downtrend_low"] = (
        is_primary_downtrend
        & (~is_fresh_breakdown)
        & net_below_ema8
        & (low_dist_ema10_adr < -0.20)
        & (rsi14_low <= 44.0)
        & (dist_from_8d_low_adr <= 0.45)
        & ((stretch_below_ema20_max3 >= 1.4) | (drop_10d_adr >= 2.6))
    )

    # Also refine `is_near_local_high` (Model 1: Uptrend Pivot High) so it strictly requires `is_primary_uptrend` and `net_above_ema8`
    b["is_near_local_high"] = (
        b["is_near_local_high"]
        & is_primary_uptrend
        & net_above_ema8
    )

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
# V2 FEATURE LISTS FOR THE 3 NEW MODELS
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

    # Baseline Logistic Regression
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=400))
    lr.fit(X_tr_ex, y_tr)
    p_lr = np.where(mask_te, lr.predict_proba(X_te_ex)[:, 1], 0.0)

    lgb_ex = lgb.LGBMClassifier(
        n_estimators=260,
        max_depth=5,
        num_leaves=26,
        learning_rate=0.035,
        min_child_samples=60,
        scale_pos_weight=spw_extreme,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    xgb_ex = XGBClassifier(
        n_estimators=260,
        max_depth=4,
        learning_rate=0.035,
        scale_pos_weight=spw_extreme,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )
    lgb_ex.fit(X_tr_ex, y_tr)
    xgb_ex.fit(X_tr_ex, y_tr)
    p_ex_raw = 0.55 * lgb_ex.predict_proba(X_te_ex)[:, 1] + 0.45 * xgb_ex.predict_proba(X_te_ex)[:, 1]
    p_ex = np.where(mask_te, p_ex_raw, 0.0)

    lgb_fb = lgb.LGBMClassifier(
        n_estimators=260,
        max_depth=5,
        num_leaves=26,
        learning_rate=0.035,
        min_child_samples=60,
        scale_pos_weight=spw_full,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    xgb_fb = XGBClassifier(
        n_estimators=260,
        max_depth=4,
        learning_rate=0.035,
        scale_pos_weight=spw_full,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )
    lgb_fb.fit(X_tr_fb, y_tr)
    xgb_fb.fit(X_tr_fb, y_tr)
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

    # IMPORTANT: Save strictly the < 2025-01-01 trained weights (DO NOT refit on 2025-2026!)
    # so that all 2025-2026 chart predictions remain 100% Out-of-Sample!
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
    Evaluates a Dual-Path model bundle on precomputed feature DataFrame `b`.
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
    p_dual_raw = 0.25 * np.minimum(p_ex_raw, p_fb_raw) + 0.75 * np.maximum(p_ex_raw, p_fb_raw)

    mask = b[mask_col].to_numpy()
    return (
        np.where(mask, p_dual_raw, 0.0),
        np.where(mask, p_ex_raw, 0.0),
        np.where(mask, p_fb_raw, 0.0),
    )


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
    print(f"Extracting V3 unified 4-regime pivot features across {len(symbols)} symbols...")

    ut_low_frames = []
    dt_high_frames = []
    dt_low_frames = []

    for sym in symbols:
        df = datastore.load_bars(sym)
        if df is None or len(df) < 120:
            continue
        b = compute_all_pivot_features(df)
        meta = pd.DataFrame(
            {
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

        c_ut_low = valid[valid["is_uptrend_pullback_regime"]].copy()
        if not c_ut_low.empty:
            ut_low_frames.append(c_ut_low)

        c_dt = valid[valid["is_downtrend_regime"]].copy()
        if not c_dt.empty:
            dt_high_frames.append(c_dt)
            dt_low_frames.append(c_dt)

    df_ut_low = pd.concat(ut_low_frames, ignore_index=True)
    df_ut_low["date_dt"] = pd.to_datetime(df_ut_low["date"])

    df_dt_high = pd.concat(dt_high_frames, ignore_index=True)
    df_dt_high["date_dt"] = pd.to_datetime(df_dt_high["date"])

    df_dt_low = pd.concat(dt_low_frames, ignore_index=True)
    df_dt_low["date_dt"] = pd.to_datetime(df_dt_low["date"])

    # Train & Evaluate all 3 V3 models
    m_ut_low = train_and_evaluate_dual_model(
        "Model 2: Uptrend Pivot Low (Pullback Buy / Support Reclaim)",
        df_ut_low,
        "is_ut_low",
        "is_near_uptrend_low",
        UT_LOW_AT_LOW_FEATURES,
        UT_LOW_FULL_BAR_FEATURES,
        spw_extreme=3.5,
        spw_full=3.1,
    )
    m_dt_high = train_and_evaluate_dual_model(
        "Model 3: Downtrend Pivot High (Bear Rally Rejection / Lower High)",
        df_dt_high,
        "is_dt_high",
        "is_near_downtrend_high",
        DT_HIGH_AT_HIGH_FEATURES,
        DT_HIGH_FULL_BAR_FEATURES,
        spw_extreme=3.6,
        spw_full=3.2,
    )
    m_dt_low = train_and_evaluate_dual_model(
        "Model 4: Downtrend Pivot Low (Capitulation Washout / Climax Bottom)",
        df_dt_low,
        "is_dt_low",
        "is_near_downtrend_low",
        DT_LOW_AT_LOW_FEATURES,
        DT_LOW_FULL_BAR_FEATURES,
        spw_extreme=3.4,
        spw_full=3.0,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(m_ut_low, UT_LOW_PATH)
    joblib.dump(m_dt_high, DT_HIGH_PATH)
    joblib.dump(m_dt_low, DT_LOW_PATH)
    print(f"\n✅ Saved all 3 V3 strict-OOS models in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
