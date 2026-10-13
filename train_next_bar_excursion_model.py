"""
Episodic Pivot (EP) Strategy System — Next-Bar Excursion & Sequence Modeling Pipeline
=====================================================================================
Predicts next-bar Maximum Favorable Excursion (MFE %), Maximum Adverse Excursion (MAE %),
and the first-hit sequence probability P(MFE First vs MAE First) using intraday 5-minute
bars across the universe of ~2,000 tickers.
"""

from __future__ import annotations
import os
import sys
import time
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, accuracy_score, brier_score_loss
import lightgbm as lgb

import datastore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_DEFAULT_PATH = Path("data/ml_datasets/next_bar_excursions.parquet")
MODEL_DEFAULT_PATH = Path("data/models/ep_next_bar_excursion_model.pkl")

FEATURE_COLS = [
    # Bar Anatomy (Current Bar t)
    "feat_close_pos", "feat_range_pct", "feat_body_pct", "feat_body_dir",
    "feat_upper_wick", "feat_lower_wick", "feat_gap_pct", "feat_intraday_ret",
    # Momentum & Multibar Returns
    "feat_ret_1d", "feat_ret_3d", "feat_ret_5d", "feat_ret_10d", "feat_ret_20d",
    # Moving Averages & Trend Structure
    "feat_dist_ema8", "feat_dist_ema21", "feat_dist_sma50", "feat_dist_sma200",
    "feat_ema8_ema21_spread", "feat_ema21_sma50_spread", "feat_trend_alignment",
    # Volatility & Bands
    "feat_atr14_pct", "feat_atr5_pct", "feat_atr_ratio", "feat_range_to_atr",
    "feat_bb_width", "feat_bb_pct", "feat_hist_vol_20",
    # Oscillators & Extrema
    "feat_rsi14", "feat_rsi5", "feat_stoch_k_14", "feat_dist_h20", "feat_dist_l20",
    # Volume Dynamics
    "feat_rvol", "feat_rvol_5d", "feat_vol_chg_1d", "feat_log_dollar_vol",
    # Intraday 5m Microstructure of Current Day t
    "feat_curr_first_hit", "feat_curr_mfe_time", "feat_curr_mae_time",
    "feat_curr_or_ret", "feat_curr_lh_ret", "feat_curr_dist_vwap",
    # Market Context (SPY) & Relative Strength
    "feat_spy_ret_1d", "feat_spy_ret_5d", "feat_rel_str_spy_1d", "feat_rel_str_spy_5d",
    # Calendar
    "feat_day_of_week"
]

_SPY_CONTEXT_CACHE: pd.DataFrame | None = None

def compute_spy_market_context() -> pd.DataFrame:
    """Precomputes daily SPY market context for O(1) alignment."""
    daily_spy = datastore.load_bars("SPY")
    if daily_spy is not None and len(daily_spy) > 0:
        c = daily_spy["close"].astype(np.float32)
        h = daily_spy["high"].astype(np.float32)
        l = daily_spy["low"].astype(np.float32)
        ret1 = (c - c.shift(1)) / c.shift(1)
        ret5 = (c - c.shift(5)) / c.shift(5)
        cp = (c - l) / (h - l).replace(0, np.nan)
        ema21 = c.ewm(span=21, adjust=False).mean()
        dist21 = (c - ema21) / c
        dates = daily_spy.index.strftime("%Y-%m-%d")
        df_spy = pd.DataFrame({
            "feat_spy_ret_1d": ret1.values,
            "feat_spy_ret_5d": ret5.values,
            "feat_spy_close_pos": cp.values,
            "feat_spy_dist_ema21": dist21.values
        }, index=dates).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return df_spy

    spy_b5 = datastore.load_bars_5m("SPY", calibrate=False)
    if spy_b5 is None or len(spy_b5) == 0:
        return pd.DataFrame()

    dates_5m = spy_b5.index.strftime("%Y-%m-%d").to_numpy()
    unique_dates, idx_starts, counts = np.unique(dates_5m, return_index=True, return_counts=True)
    closes_5m = spy_b5["close"].to_numpy(dtype=np.float32)
    highs_5m = spy_b5["high"].to_numpy(dtype=np.float32)
    lows_5m = spy_b5["low"].to_numpy(dtype=np.float32)

    spy_closes = np.array([closes_5m[idx_starts[i] + counts[i] - 1] for i in range(len(unique_dates))], dtype=np.float32)
    spy_highs = np.array([np.max(highs_5m[idx_starts[i]:idx_starts[i] + counts[i]]) for i in range(len(unique_dates))], dtype=np.float32)
    spy_lows = np.array([np.min(lows_5m[idx_starts[i]:idx_starts[i] + counts[i]]) for i in range(len(unique_dates))], dtype=np.float32)

    spy_df = pd.DataFrame({
        "spy_close": spy_closes,
        "spy_high": spy_highs,
        "spy_low": spy_lows
    }, index=unique_dates)

    c = spy_df["spy_close"]
    ret1 = (c - c.shift(1)) / c.shift(1)
    ret5 = (c - c.shift(5)) / c.shift(5)
    cp = (c - spy_df["spy_low"]) / (spy_df["spy_high"] - spy_df["spy_low"]).replace(0, np.nan)
    ema21 = c.ewm(span=21, adjust=False).mean()
    dist21 = (c - ema21) / c

    res = pd.DataFrame({
        "feat_spy_ret_1d": ret1,
        "feat_spy_ret_5d": ret5,
        "feat_spy_close_pos": cp,
        "feat_spy_dist_ema21": dist21
    }, index=unique_dates).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return res

def get_spy_market_context() -> pd.DataFrame:
    """Returns cached SPY market context for instant reuse across tickers and inference calls."""
    global _SPY_CONTEXT_CACHE
    if _SPY_CONTEXT_CACHE is None or len(_SPY_CONTEXT_CACHE) == 0:
        _SPY_CONTEXT_CACHE = compute_spy_market_context()
    return _SPY_CONTEXT_CACHE

def extract_ticker_features_and_targets(
    sym: str,
    b5: pd.DataFrame | None = None,
    spy_ctx: pd.DataFrame | None = None,
    drop_unlabeled: bool = True
) -> pd.DataFrame | None:
    """Extracts zero-lookahead features and 5m ground-truth targets for a given ticker."""
    if b5 is None:
        b5 = datastore.load_bars_5m(sym, calibrate=False)
    if b5 is None or len(b5) < 156:
        return None

    if spy_ctx is None:
        spy_ctx = get_spy_market_context()

    dates_5m = b5.index.strftime("%Y-%m-%d").to_numpy()
    highs = b5["high"].to_numpy(dtype=np.float32)
    lows = b5["low"].to_numpy(dtype=np.float32)
    opens = b5["open"].to_numpy(dtype=np.float32)
    closes = b5["close"].to_numpy(dtype=np.float32)
    vols = b5["volume"].to_numpy(dtype=np.float32)

    unique_dates, idx_starts, counts = np.unique(dates_5m, return_index=True, return_counts=True)
    n_days = len(unique_dates)
    if n_days < 25:
        return None

    day_opens = np.empty(n_days, dtype=np.float32)
    day_highs = np.empty(n_days, dtype=np.float32)
    day_lows = np.empty(n_days, dtype=np.float32)
    day_closes = np.empty(n_days, dtype=np.float32)
    day_vols = np.empty(n_days, dtype=np.float32)
    day_vwaps = np.empty(n_days, dtype=np.float32)
    seq_first = np.empty(n_days, dtype=np.float32)
    mfe_time = np.empty(n_days, dtype=np.float32)
    mae_time = np.empty(n_days, dtype=np.float32)
    or_ret = np.empty(n_days, dtype=np.float32)
    lh_ret = np.empty(n_days, dtype=np.float32)

    for i in range(n_days):
        s = idx_starts[i]
        c_len = counts[i]
        sub_h = highs[s:s+c_len]
        sub_l = lows[s:s+c_len]
        sub_o = opens[s:s+c_len]
        sub_c = closes[s:s+c_len]
        sub_v = vols[s:s+c_len]

        max_h = float(np.max(sub_h))
        min_l = float(np.min(sub_l))
        tot_v = float(np.sum(sub_v))

        day_opens[i] = sub_o[0]
        day_highs[i] = max_h
        day_lows[i] = min_l
        day_closes[i] = sub_c[-1]
        day_vols[i] = tot_v
        day_vwaps[i] = (np.sum(sub_c * sub_v) / tot_v) if tot_v > 0 else sub_c[-1]

        first_h = int(np.argmax(sub_h))
        first_l = int(np.argmin(sub_l))

        mfe_time[i] = float(first_h) / max(1, c_len - 1)
        mae_time[i] = float(first_l) / max(1, c_len - 1)

        # Ground truth sequence resolution from 5m bars:
        if first_h < first_l:
            seq_first[i] = 1.0  # High (MFE) touched before Low (MAE)
        elif first_l < first_h:
            seq_first[i] = 0.0  # Low (MAE) touched before High (MFE)
        else:
            # Both extremes occurred inside the exact same 5-minute candle
            o_bar = sub_o[first_h]
            c_bar = sub_c[first_h]
            # If bar opened at the day's high, High was touched first
            if abs(o_bar - max_h) <= 1e-4 and max_h > min_l:
                seq_first[i] = 1.0
            # If bar opened at the day's low, Low was touched first
            elif abs(o_bar - min_l) <= 1e-4 and max_h > min_l:
                seq_first[i] = 0.0
            # Bull candle: open -> low wick -> rally to high -> close near high (Low first)
            elif c_bar > o_bar:
                seq_first[i] = 0.0
            # Bear candle: open -> high wick -> drop to low -> close near low (High first)
            elif c_bar < o_bar:
                seq_first[i] = 1.0
            else:
                # Doji / flat: whichever extreme is closer to open (default 0.0)
                dist_to_h = abs(max_h - o_bar)
                dist_to_l = abs(o_bar - min_l)
                seq_first[i] = 1.0 if dist_to_h < dist_to_l else 0.0

        idx_or = min(5, c_len - 1)
        or_ret[i] = (sub_c[idx_or] - sub_o[0]) / sub_o[0] if sub_o[0] > 0 else 0.0
        idx_lh = max(0, c_len - 12)
        lh_ret[i] = (sub_c[-1] - sub_o[idx_lh]) / sub_o[idx_lh] if sub_o[idx_lh] > 0 else 0.0

    df_days = pd.DataFrame({
        "open": day_opens, "high": day_highs, "low": day_lows, "close": day_closes, "volume": day_vols,
        "vwap": day_vwaps, "first_hit": seq_first, "mfe_time": mfe_time, "mae_time": mae_time,
        "or_ret": or_ret, "lh_ret": lh_ret
    }, index=pd.to_datetime(unique_dates))

    c = df_days["close"]
    h = df_days["high"]
    l = df_days["low"]
    o = df_days["open"]
    v = df_days["volume"]

    hl_span = (h - l).replace(0, np.nan)
    close_pos = (c - l) / hl_span
    range_pct = (h - l) / c
    body_pct = (c - o).abs() / hl_span
    body_dir = (c - o) / hl_span
    upper_wick = (h - np.maximum(o, c)) / hl_span
    lower_wick = (np.minimum(o, c) - l) / hl_span
    gap_pct = (o - c.shift(1)) / c.shift(1)
    intraday_ret = (c - o) / o

    ret_1d = (c - c.shift(1)) / c.shift(1)
    ret_3d = (c - c.shift(3)) / c.shift(3)
    ret_5d = (c - c.shift(5)) / c.shift(5)
    ret_10d = (c - c.shift(10)) / c.shift(10)
    ret_20d = (c - c.shift(20)) / c.shift(20)

    ema8 = c.ewm(span=8, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    sma50 = c.rolling(50, min_periods=10).mean()
    sma200 = c.rolling(200, min_periods=20).mean()

    dist_ema8 = (c - ema8) / c
    dist_ema21 = (c - ema21) / c
    dist_sma50 = (c - sma50) / c
    dist_sma200 = (c - sma200) / c
    ema8_ema21_spread = (ema8 - ema21) / c
    ema21_sma50_spread = (ema21 - sma50) / c
    trend_alignment = ((ema8 > ema21) & (ema21 > sma50)).astype(np.float32)

    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=5).mean()
    atr5 = tr.rolling(5, min_periods=2).mean()
    atr20 = tr.rolling(20, min_periods=10).mean()
    atr14_pct = atr14 / c
    atr_ratio = atr5 / atr20.replace(0, np.nan)
    range_to_atr = (h - l) / atr14.replace(0, np.nan)

    bb_mid = c.rolling(20, min_periods=5).mean()
    bb_std = c.rolling(20, min_periods=5).std()
    bb_w = (4 * bb_std) / c
    bb_pct = (c - (bb_mid - 2 * bb_std)) / (4 * bb_std).replace(0, np.nan)

    log_ret = np.log(c / c.shift(1).replace(0, np.nan))
    hist_vol_20 = log_ret.rolling(20, min_periods=5).std() * np.sqrt(252)

    delta = c.diff()
    up = delta.clip(lower=0)
    dn = -delta.clip(upper=0)
    roll_up14 = up.ewm(com=13, adjust=False).mean()
    roll_dn14 = dn.ewm(com=13, adjust=False).mean()
    rsi14 = 100.0 - (100.0 / (1.0 + roll_up14 / roll_dn14.replace(0, np.nan)))

    roll_up5 = up.ewm(com=4, adjust=False).mean()
    roll_dn5 = dn.ewm(com=4, adjust=False).mean()
    rsi5 = 100.0 - (100.0 / (1.0 + roll_up5 / roll_dn5.replace(0, np.nan)))

    h20 = h.rolling(20, min_periods=5).max()
    l20 = l.rolling(20, min_periods=5).min()
    dist_h20 = (c - h20) / h20
    dist_l20 = (c - l20) / l20

    v_sma20 = v.rolling(20, min_periods=5).mean()
    rvol = v / v_sma20.replace(0, np.nan)
    rvol_5d = v.rolling(5, min_periods=2).mean() / v_sma20.replace(0, np.nan)
    vol_chg_1d = (v - v.shift(1)) / (v.shift(1) + 1.0)
    log_dvol = np.log1p(c * v)

    curr_dist_vwap = (c - df_days["vwap"]) / df_days["vwap"].replace(0, np.nan)

    # Next-bar targets (from day t+1)
    target_next_mfe = (df_days["high"].shift(-1) - c) / c * 100.0
    target_next_mae = (df_days["low"].shift(-1) - c) / c * 100.0
    target_next_mfe_first = df_days["first_hit"].shift(-1)
    target_next_ret = (c.shift(-1) - c) / c * 100.0

    res_df = pd.DataFrame({
        "symbol": sym,
        "date": unique_dates,
        "close_price": c.values,
        "feat_close_pos": close_pos.values,
        "feat_range_pct": range_pct.values,
        "feat_body_pct": body_pct.values,
        "feat_body_dir": body_dir.values,
        "feat_upper_wick": upper_wick.values,
        "feat_lower_wick": lower_wick.values,
        "feat_gap_pct": gap_pct.values,
        "feat_intraday_ret": intraday_ret.values,
        "feat_ret_1d": ret_1d.values,
        "feat_ret_3d": ret_3d.values,
        "feat_ret_5d": ret_5d.values,
        "feat_ret_10d": ret_10d.values,
        "feat_ret_20d": ret_20d.values,
        "feat_dist_ema8": dist_ema8.values,
        "feat_dist_ema21": dist_ema21.values,
        "feat_dist_sma50": dist_sma50.values,
        "feat_dist_sma200": dist_sma200.values,
        "feat_ema8_ema21_spread": ema8_ema21_spread.values,
        "feat_ema21_sma50_spread": ema21_sma50_spread.values,
        "feat_trend_alignment": trend_alignment.values,
        "feat_atr14_pct": atr14_pct.values,
        "feat_atr5_pct": (atr5 / c).values,
        "feat_atr_ratio": atr_ratio.values,
        "feat_range_to_atr": range_to_atr.values,
        "feat_bb_width": bb_w.values,
        "feat_bb_pct": bb_pct.values,
        "feat_hist_vol_20": hist_vol_20.values,
        "feat_rsi14": rsi14.values,
        "feat_rsi5": rsi5.values,
        "feat_stoch_k_14": ((c - l20) / (h20 - l20).replace(0, np.nan)).values,
        "feat_dist_h20": dist_h20.values,
        "feat_dist_l20": dist_l20.values,
        "feat_rvol": rvol.values,
        "feat_rvol_5d": rvol_5d.values,
        "feat_vol_chg_1d": vol_chg_1d.values,
        "feat_log_dollar_vol": log_dvol.values,
        "feat_curr_first_hit": df_days["first_hit"].values,
        "feat_curr_mfe_time": df_days["mfe_time"].values,
        "feat_curr_mae_time": df_days["mae_time"].values,
        "feat_curr_or_ret": df_days["or_ret"].values,
        "feat_curr_lh_ret": df_days["lh_ret"].values,
        "feat_curr_dist_vwap": curr_dist_vwap.values,
        "feat_day_of_week": df_days.index.dayofweek.values,
        # Targets
        "target_next_mfe": target_next_mfe.values,
        "target_next_mae": target_next_mae.values,
        "target_next_mfe_first": target_next_mfe_first.values,
        "target_next_ret": target_next_ret.values
    })

    # Merge SPY market context if provided
    if spy_ctx is not None and len(spy_ctx) > 0:
        res_df = res_df.merge(spy_ctx, left_on="date", right_index=True, how="left")
        res_df["feat_rel_str_spy_1d"] = res_df["feat_ret_1d"] - res_df["feat_spy_ret_1d"].fillna(0.0)
        res_df["feat_rel_str_spy_5d"] = res_df["feat_ret_5d"] - res_df["feat_spy_ret_5d"].fillna(0.0)
    else:
        res_df["feat_spy_ret_1d"] = 0.0
        res_df["feat_spy_ret_5d"] = 0.0
        res_df["feat_rel_str_spy_1d"] = res_df["feat_ret_1d"]
        res_df["feat_rel_str_spy_5d"] = res_df["feat_ret_5d"]

    # Drop rows based on drop_unlabeled
    if drop_unlabeled:
        res_df = res_df.iloc[:-1].copy()
        res_df = res_df.dropna(subset=["target_next_mfe", "target_next_mae", "target_next_mfe_first", "feat_ret_1d"])
    else:
        # Keep latest bar for live inference; drop early rows where indicators warm up
        res_df = res_df.dropna(subset=["feat_ret_1d"]).copy()

    # Sanity filter: minimum price $1.00 to filter unadjusted penny stock artifacts
    res_df = res_df[res_df["close_price"] >= 1.0].copy()

    # Sanitize infinities across all numeric columns
    numeric_cols = res_df.select_dtypes(include=[np.number]).columns
    res_df[numeric_cols] = res_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return res_df

def extract_ticker_features_from_daily(
    sym: str,
    daily_df: pd.DataFrame | None = None,
    spy_ctx: pd.DataFrame | None = None,
    drop_unlabeled: bool = True
) -> pd.DataFrame | None:
    """Extracts zero-lookahead features from daily bars when 5m intraday bars are not available."""
    if daily_df is None:
        daily_df = datastore.load_bars(sym)
    if daily_df is None or len(daily_df) < 25:
        return None

    if spy_ctx is None:
        spy_ctx = get_spy_market_context()

    df = daily_df.copy()
    c = df["close"].astype(np.float32)
    h = df["high"].astype(np.float32)
    l = df["low"].astype(np.float32)
    o = df["open"].astype(np.float32)
    v = df["volume"].astype(np.float32)

    hl_span = (h - l).replace(0, np.nan)
    close_pos = (c - l) / hl_span
    range_pct = (h - l) / c
    body_pct = (c - o).abs() / hl_span
    body_dir = (c - o) / hl_span
    upper_wick = (h - np.maximum(o, c)) / hl_span
    lower_wick = (np.minimum(o, c) - l) / hl_span
    gap_pct = (o - c.shift(1)) / c.shift(1)
    intraday_ret = (c - o) / o

    ret_1d = (c - c.shift(1)) / c.shift(1)
    ret_3d = (c - c.shift(3)) / c.shift(3)
    ret_5d = (c - c.shift(5)) / c.shift(5)
    ret_10d = (c - c.shift(10)) / c.shift(10)
    ret_20d = (c - c.shift(20)) / c.shift(20)

    ema8 = c.ewm(span=8, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    sma50 = c.rolling(50, min_periods=10).mean()
    sma200 = c.rolling(200, min_periods=20).mean()

    dist_ema8 = (c - ema8) / c
    dist_ema21 = (c - ema21) / c
    dist_sma50 = (c - sma50) / c
    dist_sma200 = (c - sma200) / c
    ema8_ema21_spread = (ema8 - ema21) / c
    ema21_sma50_spread = (ema21 - sma50) / c
    trend_alignment = ((ema8 > ema21) & (ema21 > sma50)).astype(np.float32)

    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=5).mean()
    atr5 = tr.rolling(5, min_periods=2).mean()
    atr20 = tr.rolling(20, min_periods=10).mean()
    atr14_pct = atr14 / c
    atr_ratio = atr5 / atr20.replace(0, np.nan)
    range_to_atr = (h - l) / atr14.replace(0, np.nan)

    bb_mid = c.rolling(20, min_periods=5).mean()
    bb_std = c.rolling(20, min_periods=5).std()
    bb_w = (4 * bb_std) / c
    bb_pct = (c - (bb_mid - 2 * bb_std)) / (4 * bb_std).replace(0, np.nan)

    log_ret = np.log(c / c.shift(1).replace(0, np.nan))
    hist_vol_20 = log_ret.rolling(20, min_periods=5).std() * np.sqrt(252)

    delta = c.diff()
    up = delta.clip(lower=0)
    dn = -delta.clip(upper=0)
    roll_up14 = up.ewm(com=13, adjust=False).mean()
    roll_dn14 = dn.ewm(com=13, adjust=False).mean()
    rsi14 = 100.0 - (100.0 / (1.0 + roll_up14 / roll_dn14.replace(0, np.nan)))

    roll_up5 = up.ewm(com=4, adjust=False).mean()
    roll_dn5 = dn.ewm(com=4, adjust=False).mean()
    rsi5 = 100.0 - (100.0 / (1.0 + roll_up5 / roll_dn5.replace(0, np.nan)))

    h20 = h.rolling(20, min_periods=5).max()
    l20 = l.rolling(20, min_periods=5).min()
    dist_h20 = (c - h20) / h20
    dist_l20 = (c - l20) / l20

    v_sma20 = v.rolling(20, min_periods=5).mean()
    rvol = v / v_sma20.replace(0, np.nan)
    rvol_5d = v.rolling(5, min_periods=2).mean() / v_sma20.replace(0, np.nan)
    vol_chg_1d = (v - v.shift(1)) / (v.shift(1) + 1.0)
    log_dvol = np.log1p(c * v)

    dates = df.index.strftime("%Y-%m-%d")
    target_next_mfe = (h.shift(-1) - c) / c * 100.0
    target_next_mae = (l.shift(-1) - c) / c * 100.0
    target_next_ret = (c.shift(-1) - c) / c * 100.0

    res_df = pd.DataFrame({
        "symbol": sym,
        "date": dates,
        "close_price": c.values,
        "feat_close_pos": close_pos.values,
        "feat_range_pct": range_pct.values,
        "feat_body_pct": body_pct.values,
        "feat_body_dir": body_dir.values,
        "feat_upper_wick": upper_wick.values,
        "feat_lower_wick": lower_wick.values,
        "feat_gap_pct": gap_pct.values,
        "feat_intraday_ret": intraday_ret.values,
        "feat_ret_1d": ret_1d.values,
        "feat_ret_3d": ret_3d.values,
        "feat_ret_5d": ret_5d.values,
        "feat_ret_10d": ret_10d.values,
        "feat_ret_20d": ret_20d.values,
        "feat_dist_ema8": dist_ema8.values,
        "feat_dist_ema21": dist_ema21.values,
        "feat_dist_sma50": dist_sma50.values,
        "feat_dist_sma200": dist_sma200.values,
        "feat_ema8_ema21_spread": ema8_ema21_spread.values,
        "feat_ema21_sma50_spread": ema21_sma50_spread.values,
        "feat_trend_alignment": trend_alignment.values,
        "feat_atr14_pct": atr14_pct.values,
        "feat_atr5_pct": (atr5 / c).values,
        "feat_atr_ratio": atr_ratio.values,
        "feat_range_to_atr": range_to_atr.values,
        "feat_bb_width": bb_w.values,
        "feat_bb_pct": bb_pct.values,
        "feat_hist_vol_20": hist_vol_20.values,
        "feat_rsi14": rsi14.values,
        "feat_rsi5": rsi5.values,
        "feat_stoch_k_14": ((c - l20) / (h20 - l20).replace(0, np.nan)).values,
        "feat_dist_h20": dist_h20.values,
        "feat_dist_l20": dist_l20.values,
        "feat_rvol": rvol.values,
        "feat_rvol_5d": rvol_5d.values,
        "feat_vol_chg_1d": vol_chg_1d.values,
        "feat_log_dollar_vol": log_dvol.values,
        "feat_curr_first_hit": 0.5,
        "feat_curr_mfe_time": 0.5,
        "feat_curr_mae_time": 0.5,
        "feat_curr_or_ret": 0.0,
        "feat_curr_lh_ret": 0.0,
        "feat_curr_dist_vwap": 0.0,
        "feat_day_of_week": pd.to_datetime(dates).dayofweek.values,
        "target_next_mfe": target_next_mfe.values,
        "target_next_mae": target_next_mae.values,
        "target_next_mfe_first": np.nan,
        "target_next_ret": target_next_ret.values
    })

    if spy_ctx is not None and len(spy_ctx) > 0:
        res_df = res_df.merge(spy_ctx, left_on="date", right_index=True, how="left")
        res_df["feat_rel_str_spy_1d"] = res_df["feat_ret_1d"] - res_df["feat_spy_ret_1d"].fillna(0.0)
        res_df["feat_rel_str_spy_5d"] = res_df["feat_ret_5d"] - res_df["feat_spy_ret_5d"].fillna(0.0)
    else:
        res_df["feat_spy_ret_1d"] = 0.0
        res_df["feat_spy_ret_5d"] = 0.0
        res_df["feat_rel_str_spy_1d"] = res_df["feat_ret_1d"]
        res_df["feat_rel_str_spy_5d"] = res_df["feat_ret_5d"]

    if drop_unlabeled:
        res_df = res_df.iloc[:-1].copy()
        res_df = res_df.dropna(subset=["target_next_mfe", "target_next_mae", "feat_ret_1d"])
    else:
        res_df = res_df.dropna(subset=["feat_ret_1d"]).copy()

    res_df = res_df[res_df["close_price"] >= 1.0].copy()
    numeric_cols = res_df.select_dtypes(include=[np.number]).columns
    res_df[numeric_cols] = res_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return res_df

def build_next_bar_dataset(
    symbols: list[str] | None = None,
    max_workers: int = 10,
    output_path: Path | str = DATASET_DEFAULT_PATH,
    min_bars: int = 156
) -> pd.DataFrame:
    """Extracts features and targets across universe of tickers and saves to Parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if symbols is None:
        p_5m = Path("data/tiingo/bars_5min")
        symbols = sorted([f.stem for f in p_5m.glob("*.parquet")])

    logger.info(f"Building Next-Bar Excursions dataset across {len(symbols)} tickers...")
    t0 = time.time()

    # Precalculate SPY context
    spy_ctx = compute_spy_market_context()
    logger.info(f"Loaded SPY market context for {len(spy_ctx)} dates.")

    dfs = []
    processed = 0

    def _worker(s: str) -> pd.DataFrame | None:
        try:
            return extract_ticker_features_and_targets(s, spy_ctx=spy_ctx)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(_worker, symbols):
            processed += 1
            if res is not None and len(res) > 0:
                dfs.append(res)
            if processed % 200 == 0 or processed == len(symbols):
                logger.info(f"Progress: {processed}/{len(symbols)} tickers processed ({len(dfs)} valid).")

    if not dfs:
        raise RuntimeError("No valid data extracted across symbols.")

    full_df = pd.concat(dfs, ignore_index=True)
    t1 = time.time()
    logger.info(f"Extraction complete in {t1 - t0:.1f}s: {len(full_df):,} total observations across {len(dfs)} tickers.")

    # Cast feature columns to float32
    for col in FEATURE_COLS:
        if col in full_df.columns:
            full_df[col] = pd.to_numeric(full_df[col], errors="coerce").fillna(0.0).astype(np.float32)

    full_df.to_parquet(output_path, compression="snappy", index=False)
    logger.info(f"✅ Saved clean dataset to {output_path} ({output_path.stat().st_size / (1024**2):.1f} MB)")
    return full_df

def train_next_bar_models(
    dataset_path: Path | str = DATASET_DEFAULT_PATH,
    out_model_path: Path | str = MODEL_DEFAULT_PATH,
    split_date: str = "2023-01-01",
    sample_n: int | None = None
) -> dict:
    """Trains MFE Regressor, MAE Regressor, and First-Hit Sequence Classifier."""
    dataset_path = Path(dataset_path)
    out_model_path = Path(out_model_path)
    out_model_path.parent.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run dataset build first.")

    logger.info(f"Loading dataset from {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    logger.info(f"Loaded {len(df):,} total samples.")

    # Clean extreme outlier spikes
    df = df[
        (df["target_next_mfe"] >= -50.0) & (df["target_next_mfe"] <= 150.0) &
        (df["target_next_mae"] >= -100.0) & (df["target_next_mae"] <= 50.0) &
        (df["target_next_mfe"] >= df["target_next_mae"])
    ].copy()

    # Fill NA on numeric features
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)

    # Optional subsample for fast training or prototyping
    if sample_n is not None and len(df) > sample_n:
        logger.info(f"Subsampling to {sample_n:,} rows for accelerated training...")
        df = df.sample(n=sample_n, random_state=42).copy()

    # Chronological Split
    df["date_dt"] = pd.to_datetime(df["date"])
    train_mask = df["date_dt"] < pd.Timestamp(split_date)
    test_mask = df["date_dt"] >= pd.Timestamp(split_date)

    X_train = df.loc[train_mask, FEATURE_COLS]
    y_mfe_train = df.loc[train_mask, "target_next_mfe"]
    y_mae_train = df.loc[train_mask, "target_next_mae"]
    y_seq_train = df.loc[train_mask, "target_next_mfe_first"]

    X_test = df.loc[test_mask, FEATURE_COLS]
    y_mfe_test = df.loc[test_mask, "target_next_mfe"]
    y_mae_test = df.loc[test_mask, "target_next_mae"]
    y_seq_test = df.loc[test_mask, "target_next_mfe_first"]

    logger.info(f"In-Sample Train (< {split_date}): {len(X_train):,} samples")
    logger.info(f"Out-of-Sample Test (>= {split_date}): {len(X_test):,} samples")

    # 1. Next-Bar MFE Regressor
    logger.info("Training Model 1: Next-Bar MFE Regressor (Expected Peak Gain %)...")
    pipe_mfe = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.03, max_depth=6,
            num_leaves=31, subsample=0.85, colsample_bytree=0.85,
            random_state=42, verbose=-1, n_jobs=-1
        ))
    ])
    pipe_mfe.fit(X_train, y_mfe_train)
    pred_mfe_test = pipe_mfe.predict(X_test)
    mfe_mae = float(mean_absolute_error(y_mfe_test, pred_mfe_test))
    mfe_r2 = float(r2_score(y_mfe_test, pred_mfe_test))
    logger.info(f"  • OOS Next-Bar MFE MAE:  {mfe_mae:.2f}% | R^2: {mfe_r2:.3f}")
    logger.info(f"  • Actual Mean MFE: +{y_mfe_test.mean():.2f}% vs Predicted: +{pred_mfe_test.mean():.2f}%")

    # 2. Next-Bar MAE Quantile Regressor (alpha=0.30)
    logger.info("Training Model 2: Next-Bar MAE Regressor (Drawdown Risk Cushion %)...")
    pipe_mae = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", lgb.LGBMRegressor(
            objective="quantile", alpha=0.30,
            n_estimators=200, learning_rate=0.03, max_depth=6,
            num_leaves=31, subsample=0.85,
            random_state=42, verbose=-1, n_jobs=-1
        ))
    ])
    pipe_mae.fit(X_train, y_mae_train)
    pred_mae_test = pipe_mae.predict(X_test)
    mae_mae = float(mean_absolute_error(y_mae_test, pred_mae_test))
    mae_coverage = float((y_mae_test >= pred_mae_test).mean() * 100.0)
    logger.info(f"  • OOS Next-Bar MAE MAE:  {mae_mae:.2f}% | Coverage: {mae_coverage:.1f}%")
    logger.info(f"  • Actual Mean MAE: {y_mae_test.mean():.2f}% vs Predicted: {pred_mae_test.mean():.2f}%")

    # 3. Next-Bar Sequence Classifier (P(High hit before Low))
    logger.info("Training Model 3: Next-Bar First-Hit Sequence Classifier (P(MFE First))...")
    pipe_seq = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.03, max_depth=6,
            num_leaves=31, subsample=0.85, colsample_bytree=0.85,
            random_state=42, verbose=-1, n_jobs=-1
        ))
    ])
    pipe_seq.fit(X_train, y_seq_train)
    probs_seq_test = pipe_seq.predict_proba(X_test)[:, 1]
    seq_auc = float(roc_auc_score(y_seq_test, probs_seq_test))
    seq_acc = float(accuracy_score(y_seq_test, probs_seq_test >= 0.5))
    seq_brier = float(brier_score_loss(y_seq_test, probs_seq_test))
    logger.info(f"  • OOS Next-Bar Sequence ROC-AUC:   {seq_auc:.3f}")
    logger.info(f"  • OOS Next-Bar Sequence Accuracy:  {seq_acc * 100.0:.1f}%")
    logger.info(f"  • OOS Next-Bar Sequence Brier:     {seq_brier:.4f}")

    # Feature Importances
    mfe_booster = pipe_mfe.named_steps["regressor"].booster_
    imp_mfe = dict(zip(FEATURE_COLS, mfe_booster.feature_importance(importance_type="gain")))
    top_mfe = sorted(imp_mfe.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info(f"  • Top MFE Features: {', '.join(k for k, _ in top_mfe)}")

    seq_booster = pipe_seq.named_steps["classifier"].booster_
    imp_seq = dict(zip(FEATURE_COLS, seq_booster.feature_importance(importance_type="gain")))
    top_seq = sorted(imp_seq.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info(f"  • Top Sequence Features: {', '.join(k for k, _ in top_seq)}")

    bundle = {
        "mfe_model": pipe_mfe,
        "mae_model": pipe_mae,
        "seq_model": pipe_seq,
        "feature_cols": FEATURE_COLS,
        "train_date": time.strftime("%Y-%m-%d"),
        "oos_metrics": {
            "mfe_mae": round(mfe_mae, 3),
            "mfe_r2": round(mfe_r2, 4),
            "mae_mae": round(mae_mae, 3),
            "mae_coverage_pct": round(mae_coverage, 1),
            "seq_roc_auc": round(seq_auc, 4),
            "seq_accuracy_pct": round(seq_acc * 100.0, 2),
            "seq_brier": round(seq_brier, 4)
        },
        "description": "Next-Bar Excursion Regressors (MFE & MAE) & First-Hit Sequence Classifier"
    }

    joblib.dump(bundle, out_model_path)
    logger.info(f"✅ Successfully saved model bundle to {out_model_path}")
    return bundle

def main():
    parser = argparse.ArgumentParser(description="Next-Bar Excursion and First-Hit Sequence Model Pipeline")
    parser.add_argument("--build-dataset", action="store_true", help="Build full dataset across all ~2000 tickers")
    parser.add_argument("--train", action="store_true", help="Train models on existing or freshly built dataset")
    parser.add_argument("--dataset-path", type=str, default=str(DATASET_DEFAULT_PATH))
    parser.add_argument("--model-path", type=str, default=str(MODEL_DEFAULT_PATH))
    parser.add_argument("--sample-size", type=int, default=None, help="Sample size for training (optional)")
    parser.add_argument("--workers", type=int, default=10, help="Number of extraction threads")
    args = parser.parse_args()

    if not args.build_dataset and not args.train:
        # Default behavior: build dataset if not exists, then train
        if not Path(args.dataset_path).exists():
            build_next_bar_dataset(output_path=args.dataset_path, max_workers=args.workers)
        train_next_bar_models(dataset_path=args.dataset_path, out_model_path=args.model_path, sample_n=args.sample_size)
    else:
        if args.build_dataset:
            build_next_bar_dataset(output_path=args.dataset_path, max_workers=args.workers)
        if args.train:
            train_next_bar_models(dataset_path=args.dataset_path, out_model_path=args.model_path, sample_n=args.sample_size)

if __name__ == "__main__":
    main()
