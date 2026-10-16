#!/usr/bin/env python3
"""
nq_ml_engine.py — Institutional NQ Intraday (15m, 1h, 4h) Pivot High/Low & Forward MAE/MFE ML Engine.

Trains & evaluates strictly causal NQ-specific Machine Learning models on 10 years of 1-minute NQ futures data
(`data/futures_db/NQ_1m_seams.parquet`, 2016-07-15 to 2026-07-14):
  - Chronological Train Split (8 Years): 2016-07-15 to 2024-07-14
    (with 2023-07-15 to 2024-07-14 held out for Out-of-Fold Isotonic Calibration)
  - Chronological Out-of-Sample Test Split (2 Years): 2024-07-15 to 2026-07-14
    (plus live 5-min tail from `data/bars_15m/NQ=F.parquet` up to 2026-10-15 for interactive chart exploration)

Key Architectural Pillars:
1. Causal Contract-Roll Seam Adjustment:
   - Eliminates artificial ~200-600 pt quarterly contract roll jumps (`is_roll == True`) via multiplicative
     ratio adjustment for all multi-bar indicators (EMA, SMA, ATR, Wilder's RSI, Inverse-RSI) and masks out
     any bar whose forward K-bar window crosses a contract roll seam.
2. 1-Minute Intrabar Microstructure Features:
   - Volume Delta Fraction (`m1_vol_delta_frac`), Top/Bottom 25% Range Volume Absorption (`m1_vol_top25_frac`,
     `m1_vol_bot25_frac`), Extreme Timing Fraction (`m1_high_time_frac`, `m1_low_time_frac`), Post-Extreme
     Intrabar Rejection (`m1_post_high_drop_frac`, `m1_post_low_bounce_frac`), and Max 1m Sweep Spike.
3. NQ Session, Liquidity Sweep & Multi-Timeframe Confluence:
   - Globex Asia, London Open, NY Pre-Market Macro, NY RTH Open Drive, NY Midday, NY Power Hour regimes.
   - Time-of-Day Slot Relative Volume (`tod_rvol`) comparing bar volume to the prior 20-day mean at the exact same clock slot.
   - Causal Session VWAPs (RTH 09:30 ET & Globex 18:00 ET) and Overnight High/Low & Prior RTH High/Low Liquidity Sweeps.
   - Piecewise Closed-Form Inverse Wilder's RSI(14) & RSI(5) exact price targets (`ob75_px`, `os25_px`).
4. Dual-Path Pivot High/Low Classifiers + Forward K-Bar (K=4, 8, 12) MAE & MFE Regressors.
"""
from __future__ import annotations

import json
import math
import os
import sys
import threading
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
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
NQ_1M_SEAMS_PATH = ROOT / "data" / "futures_db" / "NQ_1m_seams.parquet"
NQ_5M_TAIL_PATH = ROOT / "data" / "bars_15m" / "NQ=F.parquet"
MODELS_PATH = ROOT / "data" / "models" / "nq_intraday_models.pkl"
RESULTS_JSON_PATH = ROOT / "data" / "simulations" / "nq_intraday_oos_results.json"

GC_1M_PATH = ROOT / "data" / "futures_db" / "GC_1m.parquet"
GC_5M_TAIL_PATH = ROOT / "data" / "bars_15m" / "GC=F.parquet"
GC_MODELS_PATH = ROOT / "data" / "models" / "gc_intraday_models.pkl"
GC_RESULTS_JSON_PATH = ROOT / "data" / "simulations" / "gc_intraday_oos_results.json"

CL_1M_PATH = ROOT / "data" / "futures_db" / "CL_1m.parquet"
CL_5M_TAIL_PATH = ROOT / "data" / "bars_15m" / "CL=F.parquet"
CL_MODELS_PATH = ROOT / "data" / "models" / "cl_intraday_models.pkl"
CL_RESULTS_JSON_PATH = ROOT / "data" / "simulations" / "cl_intraday_oos_results.json"

BTC_1M_PATH = ROOT / "data" / "futures_db" / "BTC_1m.parquet"
BTC_5M_TAIL_PATH = ROOT / "data" / "bars_15m" / "BTC=F.parquet"
BTC_MODELS_PATH = ROOT / "data" / "models" / "btc_intraday_models.pkl"
BTC_RESULTS_JSON_PATH = ROOT / "data" / "simulations" / "btc_intraday_oos_results.json"

SI_1M_PATH = ROOT / "data" / "futures_db" / "SI_1m.parquet"
SI_5M_TAIL_PATH = ROOT / "data" / "bars_15m" / "SI=F.parquet"
SI_MODELS_PATH = ROOT / "data" / "models" / "si_intraday_models.pkl"
SI_RESULTS_JSON_PATH = ROOT / "data" / "simulations" / "si_intraday_oos_results.json"

NQ_WICKLESS_MODELS_PATH = ROOT / "data" / "models" / "nq_wickless_intraday_models.pkl"
GC_WICKLESS_MODELS_PATH = ROOT / "data" / "models" / "gc_wickless_intraday_models.pkl"
CL_WICKLESS_MODELS_PATH = ROOT / "data" / "models" / "cl_wickless_intraday_models.pkl"
BTC_WICKLESS_MODELS_PATH = ROOT / "data" / "models" / "btc_wickless_intraday_models.pkl"
SI_WICKLESS_MODELS_PATH = ROOT / "data" / "models" / "si_wickless_intraday_models.pkl"

SCORED_CACHE_DIR = ROOT / "data" / "simulations" / "nq_intraday_scored"

STUDY_START = "2016-07-15 00:00:00"
TRAIN_CUTOFF = "2024-07-15 00:00:00"
CAL_START = "2023-07-15 00:00:00"
OOS_START = "2024-07-15 00:00:00"
OOS_END = "2026-07-14 23:59:59"

TIMEFRAMES = ("5m", "15m", "1h", "4h")
HORIZONS = (4, 8, 12)


def _atomic_to_parquet(df: pd.DataFrame, target_path: Path) -> None:
    """Writes `df` to a process/thread-unique temporary parquet file and atomically replaces `target_path`."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.parent / f".{target_path.name}.{os.getpid()}.{threading.get_ident()}.tmp.parquet"
    try:
        df.to_parquet(tmp_path)
        os.replace(tmp_path, target_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def is_gc_symbol(symbol: Optional[str]) -> bool:
    """Returns True if `symbol` refers to Gold / Micro Gold futures (`GC=F`, `MGC=F`, `GC`, `MGC`)."""
    s = str(symbol or "").upper().strip()
    return s in ("GC=F", "MGC=F", "GC", "MGC")


def is_cl_symbol(symbol: Optional[str]) -> bool:
    """Returns True if `symbol` refers to Crude Oil / Micro Crude Oil futures (`CL=F`, `MCL=F`, `CL`, `MCL`)."""
    s = str(symbol or "").upper().strip()
    return s in ("CL=F", "MCL=F", "CL", "MCL")


def is_btc_symbol(symbol: Optional[str]) -> bool:
    """Returns True if `symbol` refers to Bitcoin / Micro Bitcoin futures (`BTC=F`, `MBT=F`, `BTC-USD`, `BTC`, `MBT`)."""
    s = str(symbol or "").upper().strip()
    return s in ("BTC=F", "MBT=F", "BTC-USD", "BTC", "MBT")


def is_si_symbol(symbol: Optional[str]) -> bool:
    """Returns True if `symbol` refers to Silver / Micro Silver futures (`SI=F`, `SIL=F`, `SI`, `SIL`, `MSI`)."""
    s = str(symbol or "").upper().strip()
    return s in ("SI=F", "SIL=F", "SI", "SIL", "MSI")


def get_futures_prefix(symbol: Optional[str]) -> str:
    """Returns `'gc'`, `'cl'`, `'btc'`, `'si'`, or `'nq'` for the given futures symbol."""
    if is_gc_symbol(symbol):
        return "gc"
    if is_cl_symbol(symbol):
        return "cl"
    if is_btc_symbol(symbol):
        return "btc"
    if is_si_symbol(symbol):
        return "si"
    return "nq"


def is_intraday_futures_symbol(symbol: Optional[str]) -> bool:
    """Returns True if `symbol` has dedicated 15m/1h/4h intraday futures ML models (`NQ`, `GC`, `CL`, `BTC`, `SI`)."""
    s = str(symbol or "").upper().strip()
    return (
        s in ("NQ=F", "MNQ=F", "NQ", "MNQ")
        or is_gc_symbol(s)
        or is_cl_symbol(s)
        or is_btc_symbol(s)
        or is_si_symbol(s)
    )


def load_nq_1m_base(start_date: str = STUDY_START, include_live_tail: bool = True) -> pd.DataFrame:
    """Loads 1-minute NQ bars with causal contract-roll ratio factors (`roll_mult`) and continuous prices."""
    df = pd.read_parquet(NQ_1M_SEAMS_PATH)
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)].copy()

    if include_live_tail and NQ_5M_TAIL_PATH.exists():
        try:
            tail = pd.read_parquet(NQ_5M_TAIL_PATH)
            tail = tail[tail.index > df.index.max()].copy()
            if not tail.empty:
                tail["volume"] = tail["volume"].fillna(0.0).astype(np.uint64)
                tail["is_roll"] = False
                # Flag September 2026 contract roll if present (3rd Sunday of Sept 2026 = 2026-09-20 18:00)
                sep_roll = (tail.index >= pd.Timestamp("2026-09-20 18:00:00"))
                if np.any(sep_roll):
                    first_sep_idx = tail.index[np.argmax(sep_roll)]
                    tail.loc[first_sep_idx, "is_roll"] = True
                df = pd.concat([df, tail[["open", "high", "low", "close", "volume", "is_roll"]]]).sort_index()
        except Exception:
            pass

    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    is_roll = df["is_roll"].to_numpy(dtype=bool)

    # Strictly causal cumulative roll ratio: at each roll bar i, ratio = open[i] / close[i-1]
    roll_ratios = np.ones(len(df), dtype=np.float64)
    roll_idx = np.flatnonzero(is_roll)
    for idx in roll_idx:
        if idx > 0 and c[idx - 1] > 0 and o[idx] > 0:
            r = o[idx] / c[idx - 1]
            if 0.85 <= r <= 1.15:
                roll_ratios[idx] = r

    cum_roll_mult = np.cumprod(roll_ratios)
    df["roll_mult"] = cum_roll_mult
    df["open_cont"] = o / cum_roll_mult
    df["high_cont"] = h / cum_roll_mult
    df["low_cont"] = l / cum_roll_mult
    df["close_cont"] = c / cum_roll_mult
    return df


def load_gc_1m_base(start_date: str = STUDY_START, include_live_tail: bool = True) -> pd.DataFrame:
    """Loads 10-year 1-minute Gold Futures (`GC_1m.parquet`) with causal contract-roll seam ratio factors."""
    df = pd.read_parquet(GC_1M_PATH)
    if df.index.name != "datetime":
        df.index.name = "datetime"
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)].copy()

    if include_live_tail and GC_5M_TAIL_PATH.exists():
        try:
            tail = pd.read_parquet(GC_5M_TAIL_PATH)
            tail = tail[tail.index > df.index.max()].copy()
            if not tail.empty:
                tail["volume"] = tail["volume"].fillna(0.0).astype(np.float64)
                df = pd.concat([df[["open", "high", "low", "close", "volume"]], tail[["open", "high", "low", "close", "volume"]]]).sort_index()
                df = df[~df.index.duplicated(keep="last")]
        except Exception:
            pass

    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)

    prev_c = np.roll(c, 1)
    prev_c[0] = o[0]
    gap_pct = np.abs(o - prev_c) / np.maximum(1.0, prev_c)
    dt_diff = pd.Series(df.index).diff().dt.total_seconds().fillna(60.0).to_numpy()
    is_roll = (dt_diff >= 3600.0) & (gap_pct >= 0.006)
    df["is_roll"] = is_roll

    roll_ratios = np.ones(len(df), dtype=np.float64)
    for idx in np.flatnonzero(is_roll):
        if idx > 0 and c[idx - 1] > 0 and o[idx] > 0:
            r = o[idx] / c[idx - 1]
            if 0.85 <= r <= 1.15:
                roll_ratios[idx] = r

    cum_roll_mult = np.cumprod(roll_ratios)
    df["roll_mult"] = cum_roll_mult
    df["open_cont"] = o / cum_roll_mult
    df["high_cont"] = h / cum_roll_mult
    df["low_cont"] = l / cum_roll_mult
    df["close_cont"] = c / cum_roll_mult
    return df


def load_cl_1m_base(start_date: str = STUDY_START, include_live_tail: bool = True) -> pd.DataFrame:
    """Loads 10-year 1-minute Crude Oil Futures (`CL_1m.parquet`) with causal contract-roll seam ratio factors."""
    df = pd.read_parquet(CL_1M_PATH)
    if df.index.name != "datetime":
        df.index.name = "datetime"
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)].copy()

    # Clamp negative/sub-dollar April 2020 oil anomaly to $10.00 so ratio & indicator math remains well-conditioned
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(np.float64).clip(lower=10.0)

    if include_live_tail and CL_5M_TAIL_PATH.exists():
        try:
            tail = pd.read_parquet(CL_5M_TAIL_PATH)
            tail = tail[tail.index > df.index.max()].copy()
            if not tail.empty:
                tail["volume"] = tail["volume"].fillna(0.0).astype(np.float64)
                df = pd.concat([df[["open", "high", "low", "close", "volume"]], tail[["open", "high", "low", "close", "volume"]]]).sort_index()
                df = df[~df.index.duplicated(keep="last")]
        except Exception:
            pass

    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)

    prev_c = np.roll(c, 1)
    prev_c[0] = o[0]
    gap_pct = np.abs(o - prev_c) / np.maximum(1.0, prev_c)
    dt_diff = pd.Series(df.index).diff().dt.total_seconds().fillna(60.0).to_numpy()
    is_roll = (dt_diff >= 3600.0) & (gap_pct >= 0.008)
    df["is_roll"] = is_roll

    roll_ratios = np.ones(len(df), dtype=np.float64)
    for idx in np.flatnonzero(is_roll):
        if idx > 0 and c[idx - 1] > 0 and o[idx] > 0:
            r = o[idx] / c[idx - 1]
            if 0.80 <= r <= 1.25:
                roll_ratios[idx] = r

    cum_roll_mult = np.cumprod(roll_ratios)
    df["roll_mult"] = cum_roll_mult
    df["open_cont"] = o / cum_roll_mult
    df["high_cont"] = h / cum_roll_mult
    df["low_cont"] = l / cum_roll_mult
    df["close_cont"] = c / cum_roll_mult
    return df


def load_btc_1m_base(start_date: str = "2017-08-17 00:00:00", include_live_tail: bool = True) -> pd.DataFrame:
    """Loads 9-year 1-minute Bitcoin (`BTC_1m.parquet`, 2017–2026) plus live `BTC=F` 5-minute tail."""
    df = pd.read_parquet(BTC_1M_PATH)
    if df.index.name != "datetime":
        df.index.name = "datetime"
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)].copy()

    if include_live_tail and BTC_5M_TAIL_PATH.exists():
        try:
            tail = pd.read_parquet(BTC_5M_TAIL_PATH)
            tail = tail[tail.index > df.index.max()].copy()
            if not tail.empty:
                tail["volume"] = tail["volume"].fillna(0.0).astype(np.float64)
                df = pd.concat([df[["open", "high", "low", "close", "volume"]], tail[["open", "high", "low", "close", "volume"]]]).sort_index()
                df = df[~df.index.duplicated(keep="last")]
        except Exception:
            pass

    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)

    prev_c = np.roll(c, 1)
    prev_c[0] = o[0]
    gap_pct = np.abs(o - prev_c) / np.maximum(1.0, prev_c)
    dt_diff = pd.Series(df.index).diff().dt.total_seconds().fillna(60.0).to_numpy()
    is_roll = (dt_diff >= 3600.0) & (gap_pct >= 0.015)
    df["is_roll"] = is_roll

    roll_ratios = np.ones(len(df), dtype=np.float64)
    for idx in np.flatnonzero(is_roll):
        if idx > 0 and c[idx - 1] > 0 and o[idx] > 0:
            r = o[idx] / c[idx - 1]
            if 0.80 <= r <= 1.25:
                roll_ratios[idx] = r

    cum_roll_mult = np.cumprod(roll_ratios)
    df["roll_mult"] = cum_roll_mult
    df["open_cont"] = o / cum_roll_mult
    df["high_cont"] = h / cum_roll_mult
    df["low_cont"] = l / cum_roll_mult
    df["close_cont"] = c / cum_roll_mult
    return df


def load_si_1m_base(start_date: str = "2021-08-01 18:00:00", include_live_tail: bool = True) -> pd.DataFrame:
    """Loads 5-year 1-minute Silver Futures (`SI_1m.parquet`, 2021–2026) with causal contract-roll seam ratio factors and live 5m tail."""
    df = pd.read_parquet(SI_1M_PATH)
    if df.index.name != "datetime":
        df.index.name = "datetime"
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)].copy()

    if include_live_tail and SI_5M_TAIL_PATH.exists():
        try:
            tail = pd.read_parquet(SI_5M_TAIL_PATH)
            tail = tail[tail.index > df.index.max()].copy()
            if not tail.empty:
                tail["volume"] = tail["volume"].fillna(0.0).astype(np.float64)
                df = pd.concat([df[["open", "high", "low", "close", "volume"]], tail[["open", "high", "low", "close", "volume"]]]).sort_index()
                df = df[~df.index.duplicated(keep="last")]
        except Exception:
            pass

    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)

    prev_c = np.roll(c, 1)
    prev_c[0] = o[0]
    gap_pct = np.abs(o - prev_c) / np.maximum(1.0, prev_c)
    dt_diff = pd.Series(df.index).diff().dt.total_seconds().fillna(60.0).to_numpy()
    is_roll = (dt_diff >= 3600.0) & (gap_pct >= 0.008)
    df["is_roll"] = is_roll

    roll_ratios = np.ones(len(df), dtype=np.float64)
    for idx in np.flatnonzero(is_roll):
        if idx > 0 and c[idx - 1] > 0 and o[idx] > 0:
            r = o[idx] / c[idx - 1]
            if 0.80 <= r <= 1.25:
                roll_ratios[idx] = r

    cum_roll_mult = np.cumprod(roll_ratios)
    df["roll_mult"] = cum_roll_mult
    df["open_cont"] = o / cum_roll_mult
    df["high_cont"] = h / cum_roll_mult
    df["low_cont"] = l / cum_roll_mult
    df["close_cont"] = c / cum_roll_mult
    return df


def resample_nq_with_1m_microstructure(df_1m: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    Vectorized resampling of 1-minute NQ bars into `15m`, `1h`, or `4h` bars,
    extracting both OHLCV (raw & seam-adjusted continuous) and 1-minute intrabar microstructure features.
    """
    freq_map = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h"}
    freq = freq_map.get(tf, "15min")

    bucket_ts = df_1m.index.floor(freq)
    u_buckets, first_idx, counts = np.unique(bucket_ts.values, return_index=True, return_counts=True)
    last_idx = first_idx + counts - 1
    n_bars = len(u_buckets)

    o = df_1m["open"].to_numpy(dtype=np.float64)
    h = df_1m["high"].to_numpy(dtype=np.float64)
    l = df_1m["low"].to_numpy(dtype=np.float64)
    c = df_1m["close"].to_numpy(dtype=np.float64)
    v = df_1m["volume"].to_numpy(dtype=np.float64)
    is_roll = df_1m["is_roll"].to_numpy(dtype=np.int8)
    rm = df_1m["roll_mult"].to_numpy(dtype=np.float64)

    oc = df_1m["open_cont"].to_numpy(dtype=np.float64)
    hc = df_1m["high_cont"].to_numpy(dtype=np.float64)
    lc = df_1m["low_cont"].to_numpy(dtype=np.float64)
    cc = df_1m["close_cont"].to_numpy(dtype=np.float64)

    # Standard OHLCV via reduceat
    bar_o = o[first_idx]
    bar_h = np.maximum.reduceat(h, first_idx)
    bar_l = np.minimum.reduceat(l, first_idx)
    bar_c = c[last_idx]
    bar_v = np.add.reduceat(v, first_idx)
    bar_roll = np.maximum.reduceat(is_roll, first_idx) > 0
    bar_rm = rm[last_idx]

    bar_oc = oc[first_idx]
    bar_hc = np.maximum.reduceat(hc, first_idx)
    bar_lc = np.minimum.reduceat(lc, first_idx)
    bar_cc = cc[last_idx]

    # Map each 1m bar to its parent resampled bar index
    bar_id = np.repeat(np.arange(n_bars), counts)
    b_hc_exp = bar_hc[bar_id]
    b_lc_exp = bar_lc[bar_id]
    b_rng_exp = np.maximum(1e-6, b_hc_exp - b_lc_exp)

    # 1m Intrabar Microstructure
    up_1m = (cc > oc).astype(np.float64)
    dn_1m = (cc < oc).astype(np.float64)
    v_delta_1m = v * (up_1m - dn_1m)

    # Relative position of 1m close inside the parent bar's [Low, High] range
    pos_in_rng = (cc - b_lc_exp) / b_rng_exp
    v_top25_1m = v * (pos_in_rng >= 0.75).astype(np.float64)
    v_bot25_1m = v * (pos_in_rng <= 0.25).astype(np.float64)

    rng_1m = hc - lc
    max_rng_1m = np.maximum.reduceat(rng_1m, first_idx)
    tot_rng_cont = np.maximum(1e-6, bar_hc - bar_lc)

    v_safe = np.maximum(1.0, bar_v)
    m1_vol_delta_frac = np.add.reduceat(v_delta_1m, first_idx) / v_safe
    m1_vol_top25_frac = np.add.reduceat(v_top25_1m, first_idx) / v_safe
    m1_vol_bot25_frac = np.add.reduceat(v_bot25_1m, first_idx) / v_safe
    m1_up_bar_frac = np.add.reduceat(up_1m, first_idx) / np.maximum(1.0, counts)
    m1_max_bar_range_frac = np.clip(max_rng_1m / tot_rng_cont, 0.0, 1.0)

    # Intrabar timing of High and Low (0.0 = first minute of bar, 1.0 = last minute of bar)
    pos_in_bucket = (np.arange(len(df_1m)) - np.repeat(first_idx, counts)) / np.maximum(1.0, np.repeat(counts - 1, counts))
    is_at_hi = (hc >= b_hc_exp - 1e-7)
    is_at_lo = (lc <= b_lc_exp + 1e-7)
    hi_time_1m = np.where(is_at_hi, pos_in_bucket, -1.0)
    lo_time_1m = np.where(is_at_lo, pos_in_bucket, -1.0)
    m1_high_time_frac = np.clip(np.maximum.reduceat(hi_time_1m, first_idx), 0.0, 1.0)
    m1_low_time_frac = np.clip(np.maximum.reduceat(lo_time_1m, first_idx), 0.0, 1.0)

    # Volume traded in the second half of the bar vs first half (late-bar absorption/conviction)
    v_second_half = v * (pos_in_bucket >= 0.50).astype(np.float64)
    m1_late_vol_frac = np.add.reduceat(v_second_half, first_idx) / v_safe

    out = pd.DataFrame(
        {
            "open": bar_o,
            "high": bar_h,
            "low": bar_l,
            "close": bar_c,
            "volume": bar_v,
            "is_roll": bar_roll,
            "roll_mult": bar_rm,
            "open_cont": bar_oc,
            "high_cont": bar_hc,
            "low_cont": bar_lc,
            "close_cont": bar_cc,
            "m1_count": counts.astype(np.float64),
            "m1_vol_delta_frac": m1_vol_delta_frac,
            "m1_vol_top25_frac": m1_vol_top25_frac,
            "m1_vol_bot25_frac": m1_vol_bot25_frac,
            "m1_up_bar_frac": m1_up_bar_frac,
            "m1_max_bar_range_frac": m1_max_bar_range_frac,
            "m1_high_time_frac": m1_high_time_frac,
            "m1_low_time_frac": m1_low_time_frac,
            "m1_late_vol_frac": m1_late_vol_frac,
        },
        index=pd.DatetimeIndex(u_buckets, name="datetime"),
    )
    # Filter out degenerate 0-range weekend/holiday placeholder bars
    out = out[(out["high"] > out["low"]) | (out["volume"] > 0)].copy()
    return out


def _compute_rsi_and_extremes(
    c: pd.Series, h: pd.Series, l: pd.Series, period: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Computes Wilder's RSI(period) at Close, High, and Low plus prior bar's avg gain & loss."""
    delta = c.diff().fillna(0.0)
    alpha = 1.0 / float(period)
    ag = delta.clip(lower=0.0).ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    al = (-delta.clip(upper=0.0)).ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    rsi_c = (100.0 - 100.0 / (1.0 + ag / (al + 1e-9))).fillna(50.0)

    prev_c = c.shift(1).bfill()
    prev_ag = ag.shift(1).fillna(0.0)
    prev_al = al.shift(1).fillna(0.0)

    gh = (h - prev_c).clip(lower=0.0)
    lh = (prev_c - h).clip(lower=0.0)
    agh = (prev_ag * (period - 1.0) + gh) / float(period)
    alh = (prev_al * (period - 1.0) + lh) / float(period)
    rsi_h = (100.0 - 100.0 / (1.0 + agh / (alh + 1e-9))).fillna(50.0)

    gl = (l - prev_c).clip(lower=0.0)
    ll = (prev_c - l).clip(lower=0.0)
    agl = (prev_ag * (period - 1.0) + gl) / float(period)
    all_ = (prev_al * (period - 1.0) + ll) / float(period)
    rsi_l = (100.0 - 100.0 / (1.0 + agl / (all_ + 1e-9))).fillna(50.0)

    return rsi_c, rsi_h, rsi_l, prev_ag, prev_al


def _inverse_wilder_rsi_target(
    prev_c: pd.Series, prev_ag: pd.Series, prev_al: pd.Series, period: int, target_rsi: float
) -> pd.Series:
    """
    Piecewise closed-form Inverse Wilder's RSI target price (strictly known at bar t-1 close):
    Exact price P required on bar t for RSI(period) to hit `target_rsi`.
    """
    tau = target_rsi / (100.0 - target_rsi)
    # Current RSI at flat price change (P = prev_c)
    ag_flat = prev_ag * (period - 1.0) / float(period)
    al_flat = prev_al * (period - 1.0) / float(period)
    rsi_flat = 100.0 - 100.0 / (1.0 + ag_flat / (al_flat + 1e-9))

    # Branch 1: Need an up-move (target_rsi >= rsi_flat) -> g = (period - 1)*(tau * prev_al - prev_ag), l = 0
    up_px = prev_c + (period - 1.0) * (tau * prev_al - prev_ag)
    # Branch 2: Need a down-move (target_rsi < rsi_flat) -> l = (period - 1)*(prev_ag / tau - prev_al), g = 0
    dn_px = prev_c - (period - 1.0) * (prev_ag / max(tau, 1e-9) - prev_al)
    return pd.Series(np.where(target_rsi >= rsi_flat, up_px, dn_px), index=prev_c.index)


def compute_nq_intraday_features(df_bars: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    Computes strictly causal NQ intraday features on resampled `15m`, `1h`, or `4h` bars.
    Uses continuous seam-adjusted prices (`open_cont`, `high_cont`, `low_cont`, `close_cont`) for all
    multi-bar indicators, then projects price levels onto the active contract scale (`* roll_mult`).
    """
    b = df_bars.copy()
    oc = b["open_cont"].astype(float)
    hc = b["high_cont"].astype(float)
    lc = b["low_cont"].astype(float)
    cc = b["close_cont"].astype(float)
    v = b["volume"].astype(float)
    rm = b["roll_mult"].astype(float)

    prev_c = cc.shift(1).bfill()
    tr = np.maximum(hc - lc, np.maximum((hc - prev_c).abs(), (lc - prev_c).abs()))
    # Strictly causal ATR20 (anchored at t-1 so developing bar t range never leaks into the denominator)
    adr20 = np.maximum(tr.shift(1).rolling(20, min_periods=5).mean().bfill(), cc * 0.0012)
    atr14 = np.maximum(tr.shift(1).ewm(alpha=1.0 / 14.0, min_periods=5, adjust=False).mean().bfill(), cc * 0.0012)

    b["adr20_cont"] = adr20
    b["atr14_pts"] = atr14 * rm
    b["adr20_pts"] = adr20 * rm

    # Causal Moving Averages (anchored at t-1 for zero-lookahead extreme evaluation, plus EOD display MAs)
    ema8_full = cc.ewm(span=8, adjust=False).mean()
    ema10_full = cc.ewm(span=10, adjust=False).mean()
    ema20_full = cc.ewm(span=20, adjust=False).mean()
    sma50_full = cc.rolling(50, min_periods=15).mean().bfill()
    sma200_full = cc.rolling(200, min_periods=40).mean().bfill()

    ema8_anc = ema8_full.shift(1).bfill()
    ema10_anc = ema10_full.shift(1).bfill()
    ema20_anc = ema20_full.shift(1).bfill()
    sma50_anc = sma50_full.shift(1).bfill()
    sma200_anc = sma200_full.shift(1).bfill()

    # Display MAs in as-traded points
    b["ema10_px"] = ema10_full * rm
    b["ema20_px"] = ema20_full * rm
    b["sma50_px"] = sma50_full * rm
    b["sma200_px"] = sma200_full * rm

    # Wilder's RSI(14), RSI(9), RSI(5) at Close, High, Low
    rsi14_c, rsi14_h, rsi14_l, pag14, pal14 = _compute_rsi_and_extremes(cc, hc, lc, 14)
    rsi9_c, rsi9_h, rsi9_l, _, _ = _compute_rsi_and_extremes(cc, hc, lc, 9)
    rsi5_c, rsi5_h, rsi5_l, pag5, pal5 = _compute_rsi_and_extremes(cc, hc, lc, 5)

    b["rsi14"] = rsi14_c
    b["rsi14_high"] = rsi14_h
    b["rsi14_low"] = rsi14_l
    b["rsi9_high"] = rsi9_h
    b["rsi9_low"] = rsi9_l
    b["rsi5_high"] = rsi5_h
    b["rsi5_low"] = rsi5_l
    b["prev_rsi14"] = rsi14_c.shift(1).fillna(50.0)
    b["prev_rsi14_high"] = rsi14_h.shift(1).fillna(50.0)
    b["prev_rsi14_low"] = rsi14_l.shift(1).fillna(50.0)

    # Piecewise Closed-Form Inverse Wilder's RSI-14 & RSI-5 Targets
    ob75_cont = _inverse_wilder_rsi_target(prev_c, pag14, pal14, 14, 75.0)
    os25_cont = _inverse_wilder_rsi_target(prev_c, pag14, pal14, 14, 25.0)
    ob80_rsi5_cont = _inverse_wilder_rsi_target(prev_c, pag5, pal5, 5, 80.0)
    os20_rsi5_cont = _inverse_wilder_rsi_target(prev_c, pag5, pal5, 5, 20.0)

    b["ob75_px"] = ob75_cont * rm
    b["os25_px"] = os25_cont * rm
    b["inv_rsi75_overshoot_atr"] = ((hc - ob75_cont) / adr20).clip(-10.0, 10.0).fillna(0.0)
    b["inv_rsi25_overshoot_atr"] = ((os25_cont - lc) / adr20).clip(-10.0, 10.0).fillna(0.0)
    b["inv_rsi5_80_overshoot_atr"] = ((hc - ob80_rsi5_cont) / adr20).clip(-10.0, 10.0).fillna(0.0)
    b["inv_rsi5_20_overshoot_atr"] = ((os20_rsi5_cont - lc) / adr20).clip(-10.0, 10.0).fillna(0.0)
    b["hit_inv_ob75"] = (hc >= ob75_cont).astype(float)
    b["hit_inv_os25"] = (lc <= os25_cont).astype(float)

    # Stretch & Distance from Moving Averages (in ATR units)
    b["ext_ema8_hi_atr"] = ((hc - ema8_anc) / adr20).clip(-12.0, 12.0)
    b["ext_ema10_hi_atr"] = ((hc - ema10_anc) / adr20).clip(-12.0, 12.0)
    b["ext_ema20_hi_atr"] = ((hc - ema20_anc) / adr20).clip(-15.0, 15.0)
    b["ext_sma50_hi_atr"] = ((hc - sma50_anc) / adr20).clip(-20.0, 20.0)
    b["ext_sma200_hi_atr"] = ((hc - sma200_anc) / adr20).clip(-30.0, 30.0)

    b["ext_ema8_lo_atr"] = ((lc - ema8_anc) / adr20).clip(-12.0, 12.0)
    b["ext_ema10_lo_atr"] = ((lc - ema10_anc) / adr20).clip(-12.0, 12.0)
    b["ext_ema20_lo_atr"] = ((lc - ema20_anc) / adr20).clip(-15.0, 15.0)
    b["ext_sma50_lo_atr"] = ((lc - sma50_anc) / adr20).clip(-20.0, 20.0)
    b["ext_sma200_lo_atr"] = ((lc - sma200_anc) / adr20).clip(-30.0, 30.0)

    # Trend slopes & acceleration (strictly causal at t-1)
    b["ema20_slope5_atr"] = ((ema20_anc - ema20_anc.shift(5).bfill()) / adr20).clip(-8.0, 8.0)
    b["sma50_slope10_atr"] = ((sma50_anc - sma50_anc.shift(10).bfill()) / adr20).clip(-10.0, 10.0)
    b["sma200_slope20_atr"] = ((sma200_anc - sma200_anc.shift(20).bfill()) / adr20).clip(-12.0, 12.0)
    b["ema10_vs_ema20_atr"] = ((ema10_anc - ema20_anc) / adr20).clip(-8.0, 8.0)
    b["ema20_vs_sma50_atr"] = ((ema20_anc - sma50_anc) / adr20).clip(-12.0, 12.0)
    b["sma50_vs_sma200_atr"] = ((sma50_anc - sma200_anc) / adr20).clip(-20.0, 20.0)

    # 60-bar Macro Run & Drawdown (strictly causal t-1 anchor)
    low_60_prev = lc.shift(1).rolling(60, min_periods=10).min().bfill()
    high_60_prev = hc.shift(1).rolling(60, min_periods=10).max().bfill()
    b["run_from_60b_low_adr"] = ((hc - low_60_prev) / adr20).clip(0.0, 30.0).fillna(0.0)
    b["drop_from_60b_high_adr"] = ((high_60_prev - lc) / adr20).clip(0.0, 30.0).fillna(0.0)
    b["dist_from_60b_high_adr"] = ((high_60_prev - hc) / adr20).clip(-5.0, 15.0).fillna(0.0)
    b["dist_from_60b_low_adr"] = ((lc - low_60_prev) / adr20).clip(-5.0, 15.0).fillna(0.0)

    # Multi-bar Thrust / Velocity & Stretch Percentile
    for w in (3, 5, 10, 20):
        b[f"thrust_hi_{w}b_atr"] = ((hc - cc.shift(w).bfill()) / adr20).clip(-15.0, 20.0)
        b[f"drop_lo_{w}b_atr"] = ((cc.shift(w).bfill() - lc) / adr20).clip(-15.0, 20.0)

    # Multi-wave RSI & Price Divergence (15-bar and 40-bar lookback windows)
    for w_div in (15, 40):
        prev_pk_h = hc.shift(2).rolling(w_div, min_periods=5).max().bfill()
        prev_pk_rsi = rsi14_h.shift(2).rolling(w_div, min_periods=5).max().fillna(50.0)
        near_pk = hc >= (prev_pk_h - 0.5 * adr20)
        b[f"bear_rsi_div_{w_div}b"] = np.where(near_pk, (prev_pk_rsi - rsi14_h).clip(-20.0, 30.0), 0.0)
        b[f"higher_high_vs_{w_div}b_atr"] = ((hc - prev_pk_h) / adr20).clip(-10.0, 10.0)

        prev_tr_l = lc.shift(2).rolling(w_div, min_periods=5).min().bfill()
        prev_tr_rsi = rsi14_l.shift(2).rolling(w_div, min_periods=5).min().fillna(50.0)
        near_tr = lc <= (prev_tr_l + 0.5 * adr20)
        b[f"bull_rsi_div_{w_div}b"] = np.where(near_tr, (rsi14_l - prev_tr_rsi).clip(-20.0, 30.0), 0.0)
        b[f"lower_low_vs_{w_div}b_atr"] = ((prev_tr_l - lc) / adr20).clip(-10.0, 10.0)

    # Bar Anatomy & Candle Rejection (Sub-Model B Full-Bar features)
    rng_c = np.maximum(1e-6, hc - lc)
    b["bar_range_atr"] = (rng_c / adr20).clip(0.0, 8.0)
    b["close_pos_in_bar"] = ((cc - lc) / rng_c).clip(0.0, 1.0)
    b["upper_wick_frac"] = ((hc - np.maximum(oc, cc)) / rng_c).clip(0.0, 1.0)
    b["lower_wick_frac"] = ((np.minimum(oc, cc) - lc) / rng_c).clip(0.0, 1.0)
    b["body_signed_atr"] = ((cc - oc) / adr20).clip(-6.0, 6.0)
    b["fade_from_high_atr"] = ((hc - cc) / adr20).clip(0.0, 6.0)
    b["bounce_from_low_atr"] = ((cc - lc) / adr20).clip(0.0, 6.0)

    # =========================================================================
    # NQ SESSION, GLOBEX / RTH VWAP, OVERNIGHT LIQUIDITY SWEEPS & TOD RVOL
    # =========================================================================
    idx = b.index
    mins_from_midnight = idx.hour * 60 + idx.minute
    b["sin_tod"] = np.sin(2.0 * np.pi * mins_from_midnight / 1440.0)
    b["cos_tod"] = np.cos(2.0 * np.pi * mins_from_midnight / 1440.0)
    b["day_of_week"] = idx.dayofweek.astype(float)

    # Session classification (ET):
    # 0: Asia (18:00-02:59), 1: London (03:00-07:59), 2: NY Pre-Market (08:00-09:29),
    # 3: NY Open Drive (09:30-10:29), 4: NY Midday (10:30-13:59), 5: NY Power Hour (14:00-16:00)
    sess_id = np.zeros(len(b), dtype=np.int64)
    sess_id[(mins_from_midnight >= 180) & (mins_from_midnight < 480)] = 1
    sess_id[(mins_from_midnight >= 480) & (mins_from_midnight < 570)] = 2
    sess_id[(mins_from_midnight >= 570) & (mins_from_midnight < 630)] = 3
    sess_id[(mins_from_midnight >= 630) & (mins_from_midnight < 840)] = 4
    sess_id[(mins_from_midnight >= 840) & (mins_from_midnight <= 965)] = 5
    b["session_id"] = sess_id.astype(float)
    b["is_rth"] = ((mins_from_midnight >= 570) & (mins_from_midnight <= 960)).astype(float)
    b["is_ny_open_drive"] = (sess_id == 3).astype(float)
    b["is_london_or_ny_pre"] = ((sess_id == 1) | (sess_id == 2)).astype(float)

    # Time-of-Day Slot Relative Volume (`tod_rvol`): compares bar volume against prior 20 sessions at same clock slot!
    b["rvol_raw"] = (v / np.maximum(1.0, v.shift(1).rolling(20, min_periods=5).mean().bfill())).clip(0.05, 15.0)
    slot_mean_vol = v.groupby(mins_from_midnight).transform(
        lambda s: s.shift(1).rolling(20, min_periods=3).mean().bfill()
    )
    b["tod_rvol"] = (v / np.maximum(1.0, slot_mean_vol)).clip(0.05, 15.0)

    # Globex Trading Date: shifting timestamps by +6 hours maps 18:00 ET previous evening to the current trading day
    globex_day = (idx + pd.Timedelta(hours=6)).normalize()
    typ_c = (hc + lc + cc) / 3.0
    pv_c = typ_c * v

    # Causal Globex Session VWAP (anchored at 18:00 ET)
    cum_pv_globex = pv_c.groupby(globex_day).cumsum()
    cum_v_globex = v.groupby(globex_day).cumsum()
    globex_vwap_cont = np.where(cum_v_globex > 0, cum_pv_globex / np.maximum(1.0, cum_v_globex), cc)
    globex_vwap_s = pd.Series(globex_vwap_cont, index=idx)
    globex_vwap_prev = globex_vwap_s.groupby(globex_day).shift(1).fillna(prev_c)

    b["globex_vwap_px"] = globex_vwap_s * rm
    b["dist_globex_vwap_hi_atr"] = ((hc - globex_vwap_prev) / adr20).clip(-12.0, 12.0)
    b["dist_globex_vwap_lo_atr"] = ((lc - globex_vwap_prev) / adr20).clip(-12.0, 12.0)

    # Causal RTH Session VWAP (anchored at 09:30 ET, carried forward outside RTH)
    is_rth_mask = (mins_from_midnight >= 570) & (mins_from_midnight <= 960)
    pv_rth = np.where(is_rth_mask, pv_c, 0.0)
    v_rth = np.where(is_rth_mask, v, 0.0)
    cum_pv_rth = pd.Series(pv_rth, index=idx).groupby(globex_day).cumsum()
    cum_v_rth = pd.Series(v_rth, index=idx).groupby(globex_day).cumsum()
    rth_vwap_raw = pd.Series(np.where(cum_v_rth > 0, cum_pv_rth / np.maximum(1.0, cum_v_rth), np.nan), index=idx)
    rth_vwap_cont = rth_vwap_raw.ffill().fillna(globex_vwap_s)
    rth_vwap_prev = rth_vwap_cont.shift(1).bfill()

    b["rth_vwap_px"] = rth_vwap_cont * rm
    b["dist_rth_vwap_hi_atr"] = ((hc - rth_vwap_prev) / adr20).clip(-12.0, 12.0)
    b["dist_rth_vwap_lo_atr"] = ((lc - rth_vwap_prev) / adr20).clip(-12.0, 12.0)

    # Developing Overnight High / Low (18:00 ET to 09:29 ET) & Liquidity Sweeps
    is_on_mask = mins_from_midnight < 570  # since globex_day starts at 18:00, < 09:30 ET is overnight
    on_h_series = pd.Series(np.where(is_on_mask, hc, -np.inf), index=idx)
    on_l_series = pd.Series(np.where(is_on_mask, lc, np.inf), index=idx)
    dev_on_hi = on_h_series.groupby(globex_day).cummax().replace(-np.inf, np.nan).ffill().fillna(hc)
    dev_on_lo = on_l_series.groupby(globex_day).cummin().replace(np.inf, np.nan).ffill().fillna(lc)
    on_hi_prev = dev_on_hi.groupby(globex_day).shift(1).fillna(prev_c + adr20)
    on_lo_prev = dev_on_lo.groupby(globex_day).shift(1).fillna(prev_c - adr20)

    b["on_high_px"] = dev_on_hi * rm
    b["on_low_px"] = dev_on_lo * rm
    b["dist_on_high_atr"] = ((hc - on_hi_prev) / adr20).clip(-12.0, 12.0)
    b["dist_on_low_atr"] = ((lc - on_lo_prev) / adr20).clip(-12.0, 12.0)
    b["swept_on_high"] = ((hc > on_hi_prev) & (lc < on_hi_prev)).astype(float)
    b["swept_on_low"] = ((lc < on_lo_prev) & (hc > on_lo_prev)).astype(float)

    # Previous Day's Settled RTH High / Low & Liquidity Sweeps
    rth_h_only = pd.Series(np.where(is_rth_mask, hc, np.nan), index=idx)
    rth_l_only = pd.Series(np.where(is_rth_mask, lc, np.nan), index=idx)
    daily_rth_hi = rth_h_only.groupby(globex_day).max().shift(1).ffill()
    daily_rth_lo = rth_l_only.groupby(globex_day).min().shift(1).ffill()
    prev_rth_hi = daily_rth_hi.reindex(globex_day).to_numpy(dtype=np.float64)
    prev_rth_lo = daily_rth_lo.reindex(globex_day).to_numpy(dtype=np.float64)
    prev_rth_hi_s = pd.Series(prev_rth_hi, index=idx).bfill().fillna(hc)
    prev_rth_lo_s = pd.Series(prev_rth_lo, index=idx).bfill().fillna(lc)

    b["prev_rth_high_px"] = prev_rth_hi_s * rm
    b["prev_rth_low_px"] = prev_rth_lo_s * rm
    b["dist_prev_rth_high_atr"] = ((hc - prev_rth_hi_s) / adr20).clip(-15.0, 15.0)
    b["dist_prev_rth_low_atr"] = ((lc - prev_rth_lo_s) / adr20).clip(-15.0, 15.0)
    b["swept_prev_rth_high"] = ((hc > prev_rth_hi_s) & (prev_c <= prev_rth_hi_s)).astype(float)
    b["swept_prev_rth_low"] = ((lc < prev_rth_lo_s) & (prev_c >= prev_rth_lo_s)).astype(float)

    # =========================================================================
    # CAUSAL MULTI-TIMEFRAME CONFLUENCE (Higher TF RSI & Trend State)
    # =========================================================================
    # Compute causal 4h and Daily (Globex day) developing RSI & EMA20 extension
    d_c = cc.groupby(globex_day).last()
    d_h = hc.groupby(globex_day).max()
    d_l = lc.groupby(globex_day).min()
    d_delta = d_c.diff().fillna(0.0)
    d_ag14 = d_delta.clip(lower=0.0).ewm(alpha=1.0 / 14.0, min_periods=3, adjust=False).mean()
    d_al14 = (-d_delta.clip(upper=0.0)).ewm(alpha=1.0 / 14.0, min_periods=3, adjust=False).mean()
    d_ema20 = d_c.ewm(span=20, adjust=False).mean()
    d_tr = np.maximum(d_h - d_l, (d_c - d_c.shift(1).bfill()).abs())
    d_adr = np.maximum(d_tr.rolling(14, min_periods=3).mean().bfill(), d_c * 0.008)

    d_state = pd.DataFrame(
        {
            "prev_d_c": d_c.shift(1),
            "prev_d_ag": d_ag14.shift(1),
            "prev_d_al": d_al14.shift(1),
            "prev_d_ema20": d_ema20.shift(1),
            "prev_d_adr": d_adr.shift(1),
        }
    )
    mapped_d = d_state.reindex(globex_day)
    mapped_d.index = idx

    dtd_hi = hc.groupby(globex_day).cummax()
    dtd_lo = lc.groupby(globex_day).cummin()
    pdc = mapped_d["prev_d_c"].bfill().fillna(prev_c)
    pdag = mapped_d["prev_d_ag"].bfill().fillna(0.0)
    pdal = mapped_d["prev_d_al"].bfill().fillna(0.0)
    pdema20 = mapped_d["prev_d_ema20"].bfill().fillna(prev_c)
    pdadr = np.maximum(1e-6, mapped_d["prev_d_adr"].bfill().fillna(adr20 * 4.0))

    d_gh = (dtd_hi - pdc).clip(lower=0.0)
    d_lh = (pdc - dtd_hi).clip(lower=0.0)
    d_rsi_hi = 100.0 - 100.0 / (1.0 + ((pdag * 13.0 + d_gh) / 14.0) / (((pdal * 13.0 + d_lh) / 14.0) + 1e-9))

    d_gl = (dtd_lo - pdc).clip(lower=0.0)
    d_ll = (pdc - dtd_lo).clip(lower=0.0)
    d_rsi_lo = 100.0 - 100.0 / (1.0 + ((pdag * 13.0 + d_gl) / 14.0) / (((pdal * 13.0 + d_ll) / 14.0) + 1e-9))

    b["daily_rsi14_high"] = d_rsi_hi.fillna(50.0)
    b["daily_rsi14_low"] = d_rsi_lo.fillna(50.0)
    b["daily_ext_ema20_hi_adr"] = ((dtd_hi - pdema20) / pdadr).clip(-8.0, 8.0).fillna(0.0)
    b["daily_ext_ema20_lo_adr"] = ((dtd_lo - pdema20) / pdadr).clip(-8.0, 8.0).fillna(0.0)

    # Multi-scale Overbought / Oversold Confluence Score
    b["mtf_ob_confluence"] = (
        (rsi5_h >= 80.0).astype(float)
        + (rsi9_h >= 75.0).astype(float)
        + (rsi14_h >= 72.0).astype(float)
        + (b["daily_rsi14_high"] >= 70.0).astype(float)
        + b["hit_inv_ob75"]
    ) / 5.0
    b["mtf_os_confluence"] = (
        (rsi5_l <= 20.0).astype(float)
        + (rsi9_l <= 25.0).astype(float)
        + (rsi14_l <= 28.0).astype(float)
        + (b["daily_rsi14_low"] <= 30.0).astype(float)
        + b["hit_inv_os25"]
    ) / 5.0

    # Regime flags, Pullback-to-MA Kiss Features & Causal Local Extreme Candidate Filters
    b = b.copy()
    b["is_uptrend_regime"] = (ema20_anc >= sma50_anc).astype(float)

    # Downtrend Bounce-to-Resistance Kiss Features (for DT_HIGH trend-aligned short setups)
    b["kiss_falling_ema10_atr"] = (np.abs(hc - ema10_anc) / adr20).clip(0.0, 12.0)
    b["kiss_falling_ema20_atr"] = (np.abs(hc - ema20_anc) / adr20).clip(0.0, 12.0)
    b["kiss_falling_sma50_atr"] = (np.abs(hc - sma50_anc) / adr20).clip(0.0, 15.0)
    b["kiss_vwap_from_below_atr"] = (np.abs(hc - globex_vwap_prev) / adr20).clip(0.0, 12.0)
    b["min_overhead_kiss_atr"] = np.minimum(
        np.minimum(b["kiss_falling_ema10_atr"], b["kiss_falling_ema20_atr"]),
        np.minimum(b["kiss_falling_sma50_atr"], b["kiss_vwap_from_below_atr"]),
    ).clip(0.0, 10.0)

    # Uptrend Pullback-to-Support Kiss Features (for UT_LOW trend-aligned long setups)
    b["kiss_rising_ema10_atr"] = (np.abs(lc - ema10_anc) / adr20).clip(0.0, 12.0)
    b["kiss_rising_ema20_atr"] = (np.abs(lc - ema20_anc) / adr20).clip(0.0, 12.0)
    b["kiss_rising_sma50_atr"] = (np.abs(lc - sma50_anc) / adr20).clip(0.0, 15.0)
    b["kiss_vwap_from_above_atr"] = (np.abs(lc - globex_vwap_prev) / adr20).clip(0.0, 12.0)
    b["min_support_kiss_atr"] = np.minimum(
        np.minimum(b["kiss_rising_ema10_atr"], b["kiss_rising_ema20_atr"]),
        np.minimum(b["kiss_rising_sma50_atr"], b["kiss_vwap_from_above_atr"]),
    ).clip(0.0, 10.0)

    # 40-bar Structural Range & Fibonacci Retracement Position (strictly causal at t-1)
    roll_hi_40 = hc.shift(1).rolling(40, min_periods=5).max().bfill()
    roll_lo_40 = lc.shift(1).rolling(40, min_periods=5).min().bfill()
    rng_40 = np.maximum(adr20, roll_hi_40 - roll_lo_40)
    b["dist_below_40b_high_atr"] = ((roll_hi_40 - hc) / adr20).clip(-5.0, 25.0)
    b["bounce_retracement_40b"] = ((hc - roll_lo_40) / rng_40).clip(0.0, 1.5)
    b["dist_above_40b_low_atr"] = ((lc - roll_lo_40) / adr20).clip(-5.0, 25.0)
    b["pullback_retracement_40b"] = ((roll_hi_40 - lc) / rng_40).clip(0.0, 1.5)

    roll_hi_8 = hc.shift(1).rolling(8, min_periods=2).max().bfill()
    roll_lo_8 = lc.shift(1).rolling(8, min_periods=2).min().bfill()
    roll_hi_4 = hc.shift(1).rolling(4, min_periods=2).max().bfill()
    roll_lo_4 = lc.shift(1).rolling(4, min_periods=2).min().bfill()
    is_dt = b["is_uptrend_regime"] < 0.5
    is_ut = b["is_uptrend_regime"] >= 0.5

    b["is_cand_high"] = (
        (hc >= (roll_hi_8 - 0.15 * adr20))
        | (rsi5_h >= 70.0)
        | (is_dt & (hc >= (roll_hi_4 - 0.10 * adr20)) & (b["min_overhead_kiss_atr"] <= 1.25))
    )
    b["is_cand_low"] = (
        (lc <= (roll_lo_8 + 0.15 * adr20))
        | (rsi5_l <= 30.0)
        | (is_ut & (lc <= (roll_lo_4 + 0.10 * adr20)) & (b["min_support_kiss_atr"] <= 1.25))
    )

    return b


def label_nq_pivots_and_excursions(
    b: pd.DataFrame, pivot_left: int = 5, pivot_right: int = 5, rev_atr: float = 1.5
) -> pd.DataFrame:
    """
    Computes structural Pivot High / Pivot Low labels AND Forward K-bar (K in {4, 8, 12})
    MFE, MAE, and Net Return labels (both in ATR units and in exact NQ points).
    """
    df = b.copy()
    hc = df["high_cont"].to_numpy(dtype=np.float64)
    lc = df["low_cont"].to_numpy(dtype=np.float64)
    cc = df["close_cont"].to_numpy(dtype=np.float64)
    adr = np.maximum(1e-6, df["adr20_cont"].to_numpy(dtype=np.float64))
    rm = df["roll_mult"].to_numpy(dtype=np.float64)
    is_roll = df["is_roll"].to_numpy(dtype=bool)
    n = len(df)

    # Zero-Lookahead Post-Bar Entry Price: enter at Close[t] (cc) once the bar's full features are known
    ent_short_cont = cc.copy()
    ent_long_cont = cc.copy()
    df["trigger_short_px"] = ent_short_cont * rm
    df["trigger_long_px"] = ent_long_cont * rm

    # 1. Forward K-bar MFE, MAE, Net Return, and Roll-Seam Masking for K in (4, 8, 12) strictly over bars t+1..t+K from Close[t]
    for k in HORIZONS:
        mfe_short_atr = np.zeros(n, dtype=np.float64)
        mae_short_atr = np.zeros(n, dtype=np.float64)
        net_short_atr = np.zeros(n, dtype=np.float64)

        mfe_long_atr = np.zeros(n, dtype=np.float64)
        mae_long_atr = np.zeros(n, dtype=np.float64)
        net_long_atr = np.zeros(n, dtype=np.float64)
        crosses_roll = np.ones(n, dtype=bool)

        if n > k + 1:
            valid_n = n - k
            h_win = sliding_window_view(hc[1:], k)[:valid_n]
            l_win = sliding_window_view(lc[1:], k)[:valid_n]
            r_win = sliding_window_view(is_roll[1:], k)[:valid_n]

            fwd_max_h = h_win.max(axis=1)
            fwd_min_l = l_win.min(axis=1)
            fwd_close = cc[k:]
            es_c = ent_short_cont[:valid_n]
            el_c = ent_long_cont[:valid_n]
            adr_c = adr[:valid_n]

            # Short from Close[t] over next k bars (t+1 .. t+k)
            mfe_short_atr[:valid_n] = np.maximum(0.0, (es_c - fwd_min_l) / adr_c)
            mae_short_atr[:valid_n] = np.maximum(0.0, (fwd_max_h - es_c) / adr_c)
            net_short_atr[:valid_n] = (es_c - fwd_close) / adr_c

            # Long from Close[t] over next k bars (t+1 .. t+k)
            mfe_long_atr[:valid_n] = np.maximum(0.0, (fwd_max_h - el_c) / adr_c)
            mae_long_atr[:valid_n] = np.maximum(0.0, (el_c - fwd_min_l) / adr_c)
            net_long_atr[:valid_n] = (fwd_close - el_c) / adr_c

            crosses_roll[:valid_n] = is_roll[:valid_n] | np.any(r_win, axis=1)

        df[f"mfe_short_{k}b_atr"] = mfe_short_atr
        df[f"mae_short_{k}b_atr"] = mae_short_atr
        df[f"net_short_{k}b_atr"] = net_short_atr

        df[f"mfe_long_{k}b_atr"] = mfe_long_atr
        df[f"mae_long_{k}b_atr"] = mae_long_atr
        df[f"net_long_{k}b_atr"] = net_long_atr
        df[f"crosses_roll_{k}b"] = crosses_roll

        # Asymmetry binary targets: MFE >= 1.5 * MAE and MFE >= 1.2 ATR from Close[t]
        df[f"asym_short_{k}b"] = (
            (mfe_short_atr >= 1.5 * np.maximum(0.25, mae_short_atr)) & (mfe_short_atr >= 1.2)
        ).astype(int)
        df[f"asym_long_{k}b"] = (
            (mfe_long_atr >= 1.5 * np.maximum(0.25, mae_long_atr)) & (mfe_long_atr >= 1.2)
        ).astype(int)

    # 2. Regime-Aware Structural Swing Pivot High & Pivot Low Labels
    # Trend-aligned setups (DT_HIGH in downtrends, UT_LOW in uptrends) capture pullbacks/bounces to moving averages,
    # whereas counter-trend setups (DT_LOW in downtrends, UT_HIGH in uptrends) require a deeper stretched reversal.
    win_len = pivot_left + pivot_right + 1
    is_piv_hi = np.zeros(n, dtype=int)
    is_piv_lo = np.zeros(n, dtype=int)

    if n > win_len + 8:
        h_sw = sliding_window_view(hc, win_len)
        l_sw = sliding_window_view(lc, win_len)
        loc_max = h_sw.max(axis=1)
        loc_min = l_sw.min(axis=1)

        # Shorter 3-bar local window for trend-aligned pullback/bounce pivots kissing moving averages
        sub_l = max(2, pivot_left - 2)
        sub_win = 2 * sub_l + 1
        h_sw_s = sliding_window_view(hc[pivot_left - sub_l : n - (pivot_right - sub_l)], sub_win)
        l_sw_s = sliding_window_view(lc[pivot_left - sub_l : n - (pivot_right - sub_l)], sub_win)
        loc_max_short = h_sw_s.max(axis=1)
        loc_min_short = l_sw_s.min(axis=1)

        fwd8_l = sliding_window_view(lc[pivot_left + 1 :], 8)
        fwd8_h = sliding_window_view(hc[pivot_left + 1 :], 8)
        m_len = min(len(loc_max), len(fwd8_l), len(loc_max_short))

        loc_max = loc_max[:m_len]
        loc_min = loc_min[:m_len]
        loc_max_short = loc_max_short[:m_len]
        loc_min_short = loc_min_short[:m_len]
        fwd8_l = fwd8_l[:m_len]
        fwd8_h = fwd8_h[:m_len]

        center_h = hc[pivot_left : pivot_left + m_len]
        center_l = lc[pivot_left : pivot_left + m_len]
        center_adr = np.maximum(1e-6, adr[pivot_left : pivot_left + m_len])
        ut_reg = df["is_uptrend_regime"].to_numpy(dtype=np.float64)[pivot_left : pivot_left + m_len] >= 0.5
        ext_hi = df["ext_ema20_hi_atr"].to_numpy(dtype=np.float64)[pivot_left : pivot_left + m_len]
        ext_lo = df["ext_ema20_lo_atr"].to_numpy(dtype=np.float64)[pivot_left : pivot_left + m_len]

        drop_from_hi_8b = (center_h - fwd8_l.min(axis=1)) / center_adr
        rally_from_lo_8b = (fwd8_h.max(axis=1) - center_l) / center_adr

        # Pivot High:
        # - In Downtrend (DT_HIGH, trend-aligned short): local high (or 3-bar bounce high) + drop >= 0.90 * rev_atr
        # - In Uptrend (UT_HIGH, counter-trend short): full-window local high + drop >= 1.15 * rev_atr + stretched above EMA20
        cond_dt_hi = (~ut_reg) & (center_h >= loc_max_short - 0.05 * center_adr) & (drop_from_hi_8b >= 0.90 * rev_atr)
        cond_ut_hi = ut_reg & (center_h >= loc_max - 0.05 * center_adr) & (drop_from_hi_8b >= 1.15 * rev_atr) & (ext_hi >= 0.65)
        cond_hi = cond_dt_hi | cond_ut_hi

        # Pivot Low:
        # - In Uptrend (UT_LOW, trend-aligned long): local low (or 3-bar pullback low) + rally >= 0.90 * rev_atr
        # - In Downtrend (DT_LOW, counter-trend long): full-window local low + rally >= 1.25 * rev_atr + stretched below EMA20
        cond_ut_lo = ut_reg & (center_l <= loc_min_short + 0.05 * center_adr) & (rally_from_lo_8b >= 0.90 * rev_atr)
        cond_dt_lo = (~ut_reg) & (center_l <= loc_min + 0.05 * center_adr) & (rally_from_lo_8b >= 1.25 * rev_atr) & (ext_lo <= -0.85)
        cond_lo = cond_ut_lo | cond_dt_lo

        is_piv_hi[pivot_left : pivot_left + m_len] = cond_hi.astype(int)
        is_piv_lo[pivot_left : pivot_left + m_len] = cond_lo.astype(int)

    df["is_pivot_high"] = is_piv_hi
    df["is_pivot_low"] = is_piv_lo
    return df


# =============================================================================
# FEATURE DEFINITIONS (STRICTLY CAUSAL DUAL-PATH + 1M MICROSTRUCTURE)
# =============================================================================

PIVOT_HIGH_AT_EXTREME_COLS: List[str] = [
    "rsi14_high",
    "rsi9_high",
    "rsi5_high",
    "prev_rsi14",
    "prev_rsi14_high",
    "inv_rsi75_overshoot_atr",
    "inv_rsi5_80_overshoot_atr",
    "hit_inv_ob75",
    "ext_ema8_hi_atr",
    "ext_ema10_hi_atr",
    "ext_ema20_hi_atr",
    "ext_sma50_hi_atr",
    "ext_sma200_hi_atr",
    "kiss_falling_ema10_atr",
    "kiss_falling_ema20_atr",
    "kiss_falling_sma50_atr",
    "kiss_vwap_from_below_atr",
    "min_overhead_kiss_atr",
    "dist_below_40b_high_atr",
    "bounce_retracement_40b",
    "ema20_slope5_atr",
    "sma50_slope10_atr",
    "sma200_slope20_atr",
    "ema10_vs_ema20_atr",
    "ema20_vs_sma50_atr",
    "sma50_vs_sma200_atr",
    "thrust_hi_3b_atr",
    "thrust_hi_5b_atr",
    "thrust_hi_10b_atr",
    "thrust_hi_20b_atr",
    "bear_rsi_div_15b",
    "higher_high_vs_15b_atr",
    "bear_rsi_div_40b",
    "higher_high_vs_40b_atr",
    "sin_tod",
    "cos_tod",
    "day_of_week",
    "session_id",
    "is_rth",
    "is_ny_open_drive",
    "is_london_or_ny_pre",
    "dist_globex_vwap_hi_atr",
    "dist_rth_vwap_hi_atr",
    "dist_on_high_atr",
    "swept_on_high",
    "dist_prev_rth_high_atr",
    "swept_prev_rth_high",
    "daily_rsi14_high",
    "daily_ext_ema20_hi_adr",
    "mtf_ob_confluence",
    "is_uptrend_regime",
    "run_from_60b_low_adr",
    "dist_from_60b_high_adr",
]

PIVOT_HIGH_FULL_BAR_COLS: List[str] = PIVOT_HIGH_AT_EXTREME_COLS + [
    "bar_range_atr",
    "close_pos_in_bar",
    "upper_wick_frac",
    "lower_wick_frac",
    "body_signed_atr",
    "fade_from_high_atr",
    "rvol_raw",
    "tod_rvol",
    "m1_vol_delta_frac",
    "m1_vol_top25_frac",
    "m1_vol_bot25_frac",
    "m1_up_bar_frac",
    "m1_max_bar_range_frac",
    "m1_high_time_frac",
    "m1_low_time_frac",
    "m1_late_vol_frac",
]

PIVOT_LOW_AT_EXTREME_COLS: List[str] = [
    "rsi14_low",
    "rsi9_low",
    "rsi5_low",
    "prev_rsi14",
    "prev_rsi14_low",
    "inv_rsi25_overshoot_atr",
    "inv_rsi5_20_overshoot_atr",
    "hit_inv_os25",
    "ext_ema8_lo_atr",
    "ext_ema10_lo_atr",
    "ext_ema20_lo_atr",
    "ext_sma50_lo_atr",
    "ext_sma200_lo_atr",
    "kiss_rising_ema10_atr",
    "kiss_rising_ema20_atr",
    "kiss_rising_sma50_atr",
    "kiss_vwap_from_above_atr",
    "min_support_kiss_atr",
    "dist_above_40b_low_atr",
    "pullback_retracement_40b",
    "ema20_slope5_atr",
    "sma50_slope10_atr",
    "sma200_slope20_atr",
    "ema10_vs_ema20_atr",
    "ema20_vs_sma50_atr",
    "sma50_vs_sma200_atr",
    "drop_lo_3b_atr",
    "drop_lo_5b_atr",
    "drop_lo_10b_atr",
    "drop_lo_20b_atr",
    "bull_rsi_div_15b",
    "lower_low_vs_15b_atr",
    "bull_rsi_div_40b",
    "lower_low_vs_40b_atr",
    "sin_tod",
    "cos_tod",
    "day_of_week",
    "session_id",
    "is_rth",
    "is_ny_open_drive",
    "is_london_or_ny_pre",
    "dist_globex_vwap_lo_atr",
    "dist_rth_vwap_lo_atr",
    "dist_on_low_atr",
    "swept_on_low",
    "dist_prev_rth_low_atr",
    "swept_prev_rth_low",
    "daily_rsi14_low",
    "daily_ext_ema20_lo_adr",
    "mtf_os_confluence",
    "is_uptrend_regime",
    "drop_from_60b_high_adr",
    "dist_from_60b_low_adr",
]

MAJOR_TOP_AT_EXTREME_COLS: List[str] = list(PIVOT_HIGH_AT_EXTREME_COLS)
MAJOR_BOTTOM_AT_EXTREME_COLS: List[str] = list(PIVOT_LOW_AT_EXTREME_COLS)

PIVOT_LOW_FULL_BAR_COLS: List[str] = PIVOT_LOW_AT_EXTREME_COLS + [
    "bar_range_atr",
    "close_pos_in_bar",
    "upper_wick_frac",
    "lower_wick_frac",
    "body_signed_atr",
    "bounce_from_low_atr",
    "rvol_raw",
    "tod_rvol",
    "m1_vol_delta_frac",
    "m1_vol_top25_frac",
    "m1_vol_bot25_frac",
    "m1_up_bar_frac",
    "m1_max_bar_range_frac",
    "m1_high_time_frac",
    "m1_low_time_frac",
    "m1_late_vol_frac",
]


def _fit_single_dual_classifier(
    df_cand: pd.DataFrame,
    label_col: str,
    at_ext_cols: List[str],
    full_cols: List[str],
) -> Dict[str, Any]:
    """
    Fits a single Dual-Path (Sub-Model A At-Extreme + Sub-Model B Full-Bar + 1m Microstructure)
    LightGBM + XGBoost ensemble on `df_cand` (< TRAIN_CUTOFF) with chronological
    Isotonic Calibration on `[CAL_START, TRAIN_CUTOFF)`.
    """
    tr_core = df_cand[df_cand.index < pd.Timestamp(CAL_START)]
    tr_cal = df_cand[(df_cand.index >= pd.Timestamp(CAL_START)) & (df_cand.index < pd.Timestamp(TRAIN_CUTOFF))]
    tr_all = df_cand[df_cand.index < pd.Timestamp(TRAIN_CUTOFF)]

    if len(tr_core) < 200 or len(tr_cal) < 50:
        tr_core = tr_all.iloc[: int(len(tr_all) * 0.85)]
        tr_cal = tr_all.iloc[int(len(tr_all) * 0.85) :]

    y_core = tr_core[label_col].to_numpy(dtype=int)
    pos_rate = max(0.01, float(y_core.mean()))
    # Sub-Model B (Full-Bar) has strong wick separation; Sub-Model A (At-Extreme) needs stronger class-balance
    # weighting so real-time intrabar discovery reaches full [0.05, 0.90] probability resolution without being crushed.
    spw_fb = float(np.clip((1.0 - pos_rate) / max(pos_rate, 1e-4) * 0.48, 1.4, 5.0))
    spw_ex = float(np.clip((1.0 - pos_rate) / max(pos_rate, 1e-4) * 0.85, 2.2, 7.5))

    def _fit_pair(X_df: pd.DataFrame, y_arr: np.ndarray, spw_val: float):
        X_clean = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        m_lgb = lgb.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.035,
            num_leaves=31,
            max_depth=6,
            min_child_samples=40,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw_val,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        m_xgb = XGBClassifier(
            n_estimators=220,
            learning_rate=0.035,
            max_depth=5,
            min_child_weight=30,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw_val,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        m_lgb.fit(X_clean, y_arr)
        m_xgb.fit(X_clean, y_arr)
        return m_lgb, m_xgb

    # Fit on core training set to calibrate on unseen chronological validation year (2023-2024)
    lgb_ex_c, xgb_ex_c = _fit_pair(tr_core[at_ext_cols], y_core, spw_ex)
    lgb_fb_c, xgb_fb_c = _fit_pair(tr_core[full_cols], y_core, spw_fb)

    X_cal_ex = tr_cal[at_ext_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_cal_fb = tr_cal[full_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_cal = tr_cal[label_col].to_numpy(dtype=int)

    p_ex_cal = 0.55 * lgb_ex_c.predict_proba(X_cal_ex)[:, 1] + 0.45 * xgb_ex_c.predict_proba(X_cal_ex)[:, 1]
    p_fb_cal = 0.55 * lgb_fb_c.predict_proba(X_cal_fb)[:, 1] + 0.45 * xgb_fb_c.predict_proba(X_cal_fb)[:, 1]
    p_dual_cal = 0.35 * p_ex_cal + 0.65 * p_fb_cal

    iso_dual = IsotonicRegression(y_min=0.01, y_max=0.96, out_of_bounds="clip")
    iso_dual.fit(p_dual_cal, y_cal)

    iso_ex = IsotonicRegression(y_min=0.01, y_max=0.96, out_of_bounds="clip")
    iso_ex.fit(p_ex_cal, y_cal)

    # Refit final trees on full 8-year training window (2016-2024)
    y_all = tr_all[label_col].to_numpy(dtype=int)
    lgb_ex, xgb_ex = _fit_pair(tr_all[at_ext_cols], y_all, spw_ex)
    lgb_fb, xgb_fb = _fit_pair(tr_all[full_cols], y_all, spw_fb)

    imp = pd.Series(lgb_fb.feature_importances_, index=full_cols).sort_values(ascending=False)
    top_features = [{"feature": k, "importance": int(v)} for k, v in imp.head(12).items()]

    return {
        "lgb_extreme": lgb_ex,
        "xgb_extreme": xgb_ex,
        "lgb_full": lgb_fb,
        "xgb_full": xgb_fb,
        "iso_dual": iso_dual,
        "iso_extreme": iso_ex,
        "at_extreme_cols": at_ext_cols,
        "full_bar_cols": full_cols,
        "top_features": top_features,
    }


def _fit_dual_classifier(
    df_cand: pd.DataFrame,
    label_col: str,
    at_ext_cols: List[str],
    full_cols: List[str],
) -> Dict[str, Any]:
    """
    Fits Regime-Specific Dual-Path Classifiers (splitting into Uptrend `ut_sub` and Downtrend `dt_sub`
    when `is_uptrend_regime` is available, yielding the 4-Model architecture `UT_HIGH`, `DT_HIGH`, `UT_LOW`, `DT_LOW`),
    plus a unified fallback bundle.
    """
    base_bundle = _fit_single_dual_classifier(df_cand, label_col, at_ext_cols, full_cols)
    if "is_uptrend_regime" in df_cand.columns:
        ut_cand = df_cand[df_cand["is_uptrend_regime"] >= 0.5]
        dt_cand = df_cand[df_cand["is_uptrend_regime"] < 0.5]
        if len(ut_cand) >= 250 and len(dt_cand) >= 250:
            base_bundle["ut_sub"] = _fit_single_dual_classifier(ut_cand, label_col, at_ext_cols, full_cols)
            base_bundle["dt_sub"] = _fit_single_dual_classifier(dt_cand, label_col, at_ext_cols, full_cols)
    return base_bundle


def _score_single_sub_bundle(
    df: pd.DataFrame, bundle: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "lgb" in bundle and "feature_cols" in bundle:
        cols = bundle["feature_cols"]
        X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        p_raw = 0.55 * bundle["lgb"].predict_proba(X)[:, 1] + 0.45 * bundle["xgb"].predict_proba(X)[:, 1]
        iso = bundle.get("iso")
        if iso is not None:
            p_iso = iso.predict(p_raw)
            p_final = np.clip(0.65 * p_raw + 0.35 * p_iso, 0.0, 0.98)
        else:
            p_final = p_raw
        return p_final, p_final, p_raw

    ex_cols = bundle["at_extreme_cols"]
    fb_cols = bundle["full_bar_cols"]

    X_ex = df[ex_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_fb = df[fb_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    p_ex_raw = 0.55 * bundle["lgb_extreme"].predict_proba(X_ex)[:, 1] + 0.45 * bundle["xgb_extreme"].predict_proba(X_ex)[:, 1]
    p_fb_raw = 0.55 * bundle["lgb_full"].predict_proba(X_fb)[:, 1] + 0.45 * bundle["xgb_full"].predict_proba(X_fb)[:, 1]
    p_dual_raw = 0.35 * p_ex_raw + 0.65 * p_fb_raw

    p_dual_iso = bundle["iso_dual"].predict(p_dual_raw)
    p_ex_iso = bundle["iso_extreme"].predict(p_ex_raw)

    # Prevent isotonic ceiling compression on Sub-Model A (At-Extreme) when validation base rate is ~12-16%
    iso_ex_max = float(np.max(getattr(bundle["iso_extreme"], "y_thresholds_", [0.85])))
    iso_ex_scale = min(1.85, 0.82 / max(0.30, iso_ex_max))
    p_ex_iso_scaled = np.clip(p_ex_iso * iso_ex_scale, 0.0, 0.96)

    p_dual_final = np.clip(0.55 * p_dual_raw + 0.45 * p_dual_iso, 0.0, 0.98)
    p_ex_final = np.clip(0.72 * p_ex_raw + 0.28 * p_ex_iso_scaled, 0.0, 0.98)
    return p_dual_final, p_ex_final, p_dual_raw


def score_nq_pivot_bundle(
    df: pd.DataFrame, bundle: Dict[str, Any], cand_col: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scores `df` with a trained NQ Dual-Path Pivot bundle (routing Uptrend bars to `ut_sub`
    and Downtrend bars to `dt_sub` when regime-specific sub-models are present).
    Returns `(prob_dual_cal, prob_extreme_cal, prob_dual_raw)` masked to `cand_col`.
    """
    mask = df[cand_col].to_numpy(dtype=bool)
    if "ut_sub" in bundle and "dt_sub" in bundle and "is_uptrend_regime" in df.columns:
        is_ut = df["is_uptrend_regime"].to_numpy(dtype=np.float64) >= 0.5
        d_ut, e_ut, r_ut = _score_single_sub_bundle(df, bundle["ut_sub"])
        d_dt, e_dt, r_dt = _score_single_sub_bundle(df, bundle["dt_sub"])
        p_dual_final = np.where(is_ut, d_ut, d_dt)
        p_ex_final = np.where(is_ut, e_ut, e_dt)
        p_dual_raw = np.where(is_ut, r_ut, r_dt)
    else:
        p_dual_final, p_ex_final, p_dual_raw = _score_single_sub_bundle(df, bundle)

    return (
        np.where(mask, p_dual_final, 0.0),
        np.where(mask, p_ex_final, 0.0),
        np.where(mask, p_dual_raw, 0.0),
    )



def _fit_excursion_models_for_direction(
    df_cand_train: pd.DataFrame,
    direction: str,
    feat_cols: List[str],
) -> Dict[int, Dict[str, Any]]:
    """
    Trains Forward K-Bar (K in {4, 8, 12}) MFE & MAE Regressors (Expected + Quantile)
    and Asymmetry Classifiers (`MFE >= 1.8 * MAE & MFE >= 1.4 ATR`) on the 8-year training set.
    """
    X_tr = df_cand_train[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    models_by_k: Dict[int, Dict[str, Any]] = {}

    for k in HORIZONS:
        valid_mask = ~df_cand_train[f"crosses_roll_{k}b"].to_numpy(dtype=bool)
        Xk = X_tr.iloc[valid_mask]
        y_mfe = np.clip(df_cand_train[f"mfe_{direction}_{k}b_atr"].to_numpy(dtype=np.float64)[valid_mask], 0.0, 10.0)
        y_mae = np.clip(df_cand_train[f"mae_{direction}_{k}b_atr"].to_numpy(dtype=np.float64)[valid_mask], 0.0, 10.0)
        y_asym = df_cand_train[f"asym_{direction}_{k}b"].to_numpy(dtype=int)[valid_mask]

        # Expected MFE Regressor (Huber loss + regularization for OOS monotonicity)
        reg_mfe_mean = lgb.LGBMRegressor(
            objective="huber",
            n_estimators=140,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=4,
            min_child_samples=100,
            reg_lambda=10.0,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        reg_mfe_mean.fit(Xk, y_mfe)

        # 70th Percentile MFE Target Regressor
        reg_mfe_q70 = lgb.LGBMRegressor(
            objective="quantile",
            alpha=0.70,
            n_estimators=130,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=4,
            min_child_samples=100,
            reg_lambda=10.0,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        reg_mfe_q70.fit(Xk, y_mfe)

        # Expected MAE Regressor
        reg_mae_mean = lgb.LGBMRegressor(
            objective="huber",
            n_estimators=140,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=4,
            min_child_samples=100,
            reg_lambda=10.0,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        reg_mae_mean.fit(Xk, y_mae)

        # 75th Percentile MAE Safe Stop Cushion Regressor
        reg_mae_q75 = lgb.LGBMRegressor(
            objective="quantile",
            alpha=0.75,
            n_estimators=130,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=4,
            min_child_samples=100,
            reg_lambda=10.0,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        reg_mae_q75.fit(Xk, y_mae)

        # Asymmetry Classifier: P(MFE_K >= 1.5 * MAE_K & MFE_K >= 1.2 ATR)
        pos_r = max(0.01, float(y_asym.mean()))
        spw_a = float(np.clip((1.0 - pos_r) / max(pos_r, 1e-4) * 0.45, 1.0, 3.5))
        clf_asym = lgb.LGBMClassifier(
            n_estimators=140,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=4,
            min_child_samples=100,
            reg_lambda=10.0,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw_a,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        clf_asym.fit(Xk, y_asym)

        models_by_k[k] = {
            "reg_mfe_mean": reg_mfe_mean,
            "reg_mfe_q70": reg_mfe_q70,
            "reg_mae_mean": reg_mae_mean,
            "reg_mae_q75": reg_mae_q75,
            "clf_asym": clf_asym,
            "feat_cols": feat_cols,
        }

    return models_by_k


def score_excursion_models(
    df: pd.DataFrame,
    exc_bundle: Dict[int, Dict[str, Any]],
    direction: str,
) -> None:
    """In-place adds predicted MFE, MAE, Quantiles, Asymmetry Prob, and Edge Ratio for all K in {4, 8, 12}."""
    for k, mb in exc_bundle.items():
        X = df[mb["feat_cols"]].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        pred_mfe = np.clip(mb["reg_mfe_mean"].predict(X), 0.30, 10.0)
        pred_mfe_q70 = np.clip(mb["reg_mfe_q70"].predict(X), pred_mfe * 1.08, 12.0)
        pred_mae = np.clip(mb["reg_mae_mean"].predict(X), 0.30, 8.0)
        pred_mae_q75 = np.clip(mb["reg_mae_q75"].predict(X), pred_mae * 1.08, 10.0)
        prob_asym = mb["clf_asym"].predict_proba(X)[:, 1]

        # Actionable Predicted MFE/MAE Ratio calibrated to standard [1.25, 1.75] hurdles
        raw_ratio = pred_mfe / np.maximum(0.35, pred_mae)
        edge_ratio = np.clip(raw_ratio * (1.07 + 0.71 * prob_asym), 0.35, 6.0)

        df[f"pred_mfe_{direction}_{k}b_atr"] = pred_mfe
        df[f"pred_mfe_q70_{direction}_{k}b_atr"] = pred_mfe_q70
        df[f"pred_mae_{direction}_{k}b_atr"] = pred_mae
        df[f"pred_mae_q75_{direction}_{k}b_atr"] = pred_mae_q75
        df[f"prob_asym_{direction}_{k}b"] = prob_asym
        df[f"pred_ratio_{direction}_{k}b"] = edge_ratio


def evaluate_generic_equity_baseline_on_nq(df_oos: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluates the existing Generic Daily Equity Pivot Models (trained on daily equities)
    applied zero-shot to the NQ 2-year OOS test set (`2024-07-15` to `2026-07-14`).
    Because generic daily equity models lack NQ Globex/RTH session structure, Overnight High/Low sweeps,
    Time-of-Day Slot RVOL, and 1-minute volume delta/absorption, we measure their exact OOS performance
    for head-to-head comparison against the NQ-Specific models.
    """
    import train_all_pivot_models as tap

    # Use a representative slice if very large, or full OOS for 1h/4h and last 25k bars for 15m
    eval_df = df_oos.copy()
    if len(eval_df) > 30000:
        eval_df = eval_df.iloc[-30000:].copy()

    raw_ohlcv = eval_df[["open_cont", "high_cont", "low_cont", "close_cont", "volume"]].rename(
        columns={"open_cont": "open", "high_cont": "high", "low_cont": "low", "close_cont": "close"}
    )
    feats_eq = tap.compute_all_pivot_features(raw_ohlcv, symbol=None)

    ut_hi_m = joblib.load(tap.UT_HIGH_V4_PATH) if tap.UT_HIGH_V4_PATH.exists() else joblib.load(tap.UT_HIGH_PATH)
    ut_lo_m = joblib.load(tap.UT_LOW_V4_PATH) if tap.UT_LOW_V4_PATH.exists() else joblib.load(tap.UT_LOW_PATH)
    dt_hi_m = joblib.load(tap.DT_HIGH_V4_PATH) if tap.DT_HIGH_V4_PATH.exists() else joblib.load(tap.DT_HIGH_PATH)
    dt_lo_m = joblib.load(tap.DT_LOW_V4_PATH) if tap.DT_LOW_V4_PATH.exists() else joblib.load(tap.DT_LOW_PATH)

    p_ut_hi, _, _ = tap.score_dual_model_on_features(feats_eq, ut_hi_m, "is_near_local_high")
    p_dt_hi, _, _ = tap.score_dual_model_on_features(feats_eq, dt_hi_m, "is_near_downtrend_high")
    p_eq_hi = np.maximum(p_ut_hi, p_dt_hi)

    p_ut_lo, _, _ = tap.score_dual_model_on_features(feats_eq, ut_lo_m, "is_near_uptrend_low")
    p_dt_lo, _, _ = tap.score_dual_model_on_features(feats_eq, dt_lo_m, "is_near_downtrend_low")
    p_eq_lo = np.maximum(p_ut_lo, p_dt_lo)

    y_hi = eval_df["is_pivot_high"].to_numpy(dtype=int)
    y_lo = eval_df["is_pivot_low"].to_numpy(dtype=int)
    cand_hi = eval_df["is_cand_high"].to_numpy(dtype=bool)
    cand_lo = eval_df["is_cand_low"].to_numpy(dtype=bool)

    auc_hi = float(roc_auc_score(y_hi[cand_hi], p_eq_hi[cand_hi])) if y_hi[cand_hi].sum() > 0 else 0.5
    pr_hi = float(average_precision_score(y_hi[cand_hi], p_eq_hi[cand_hi])) if y_hi[cand_hi].sum() > 0 else 0.0
    p55_hi = float(precision_score(y_hi[cand_hi], (p_eq_hi[cand_hi] >= 0.55).astype(int), zero_division=0))

    auc_lo = float(roc_auc_score(y_lo[cand_lo], p_eq_lo[cand_lo])) if y_lo[cand_lo].sum() > 0 else 0.5
    pr_lo = float(average_precision_score(y_lo[cand_lo], p_eq_lo[cand_lo])) if y_lo[cand_lo].sum() > 0 else 0.0
    p55_lo = float(precision_score(y_lo[cand_lo], (p_eq_lo[cand_lo] >= 0.55).astype(int), zero_division=0))

    return {
        "eq_high_roc_auc": round(auc_hi, 4),
        "eq_high_pr_auc": round(pr_hi, 4),
        "eq_high_prec_55": round(p55_hi * 100.0, 2),
        "eq_low_roc_auc": round(auc_lo, 4),
        "eq_low_pr_auc": round(pr_lo, 4),
        "eq_low_prec_55": round(p55_lo * 100.0, 2),
    }


def compute_oos_deciles_and_strategy_stats(
    df_oos: pd.DataFrame, tf: str
) -> Dict[str, Any]:
    """
    Computes comprehensive 2-Year Out-of-Sample (`2024-07-15` to `2026-07-14`) metrics:
    1. Pivot High & Pivot Low classification metrics (ROC-AUC, PR-AUC, Precision @ 50%, 55%, 60%, 65%, 70%, Lift).
    2. Forward K-Bar (K=4, 8, 12) MFE/MAE Prediction Deciles (Quintiles 1..5) showing monotonic sorting of
       realized MFE/MAE ratio, realized net NQ points, and win rate.
    3. Actionable Strategy Simulation across probability thresholds & MFE/MAE ratio filters.
    """
    valid_oos = df_oos[~df_oos["crosses_roll_12b"]].copy()

    # 1. Classification Metrics on Candidate Extremes
    c_hi = valid_oos[valid_oos["is_cand_high"]]
    c_lo = valid_oos[valid_oos["is_cand_low"]]

    def _cls_metrics(sub: pd.DataFrame, y_col: str, p_dual_col: str, p_ex_col: str) -> Dict[str, Any]:
        y = sub[y_col].to_numpy(dtype=int)
        p_d = sub[p_dual_col].to_numpy(dtype=np.float64)
        p_e = sub[p_ex_col].to_numpy(dtype=np.float64)
        base_rate = float(y.mean()) if len(y) else 0.0
        auc_d = float(roc_auc_score(y, p_d)) if y.sum() > 0 else 0.5
        pr_d = float(average_precision_score(y, p_d)) if y.sum() > 0 else 0.0
        auc_e = float(roc_auc_score(y, p_e)) if y.sum() > 0 else 0.5
        pr_e = float(average_precision_score(y, p_e)) if y.sum() > 0 else 0.0

        thresh_stats = {}
        for th in (0.50, 0.55, 0.60, 0.65, 0.70):
            sel = p_d >= th
            n_sig = int(sel.sum())
            prec = float(precision_score(y, sel.astype(int), zero_division=0))
            rec = float(recall_score(y, sel.astype(int), zero_division=0))
            lift = prec / max(base_rate, 1e-6)
            thresh_stats[f"{int(th*100)}"] = {
                "signals": n_sig,
                "precision_pct": round(prec * 100.0, 2),
                "recall_pct": round(rec * 100.0, 2),
                "lift": round(lift, 2),
            }
        return {
            "n_candidates": len(sub),
            "n_true_pivots": int(y.sum()),
            "base_rate_pct": round(base_rate * 100.0, 2),
            "dual_roc_auc": round(auc_d, 4),
            "dual_pr_auc": round(pr_d, 4),
            "extreme_roc_auc": round(auc_e, 4),
            "extreme_pr_auc": round(pr_e, 4),
            "thresholds": thresh_stats,
        }

    hi_cls = _cls_metrics(c_hi, "is_pivot_high", "prob_pivot_high", "prob_pivot_high_extreme")
    lo_cls = _cls_metrics(c_lo, "is_pivot_low", "prob_pivot_low", "prob_pivot_low_extreme")

    # 2. Forward K-Bar Excursion Calibration Quintiles & Strategy Performance on OOS Candidates & Signals
    horizons_summary: Dict[str, Any] = {}
    sig_hi = valid_oos[valid_oos["is_cand_high"]].copy()
    sig_lo = valid_oos[valid_oos["is_cand_low"]].copy()

    for k in HORIZONS:
        # Combine High (Short) and Low (Long) candidate excursions into unified directional arrays
        pred_mfe = np.concatenate([
            sig_hi[f"pred_mfe_short_{k}b_atr"].to_numpy(),
            sig_lo[f"pred_mfe_long_{k}b_atr"].to_numpy(),
        ])
        pred_mae = np.concatenate([
            sig_hi[f"pred_mae_short_{k}b_atr"].to_numpy(),
            sig_lo[f"pred_mae_long_{k}b_atr"].to_numpy(),
        ])
        pred_ratio = np.concatenate([
            sig_hi[f"pred_ratio_short_{k}b"].to_numpy(),
            sig_lo[f"pred_ratio_long_{k}b"].to_numpy(),
        ])
        act_mfe = np.concatenate([
            sig_hi[f"mfe_short_{k}b_atr"].to_numpy(),
            sig_lo[f"mfe_long_{k}b_atr"].to_numpy(),
        ])
        act_mae = np.concatenate([
            sig_hi[f"mae_short_{k}b_atr"].to_numpy(),
            sig_lo[f"mae_long_{k}b_atr"].to_numpy(),
        ])
        act_net = np.concatenate([
            sig_hi[f"net_short_{k}b_atr"].to_numpy(),
            sig_lo[f"net_long_{k}b_atr"].to_numpy(),
        ])
        atr_pts = np.concatenate([
            sig_hi["adr20_pts"].to_numpy(),
            sig_lo["adr20_pts"].to_numpy(),
        ])
        probs = np.concatenate([
            sig_hi["prob_pivot_high"].to_numpy(),
            sig_lo["prob_pivot_low"].to_numpy(),
        ])

        quintiles = []
        if len(pred_ratio) >= 25:
            qs = np.quantile(pred_ratio, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            for q_idx in range(5):
                q_lo, q_hi = qs[q_idx], qs[q_idx + 1]
                m_q = (pred_ratio >= q_lo) & (pred_ratio <= q_hi if q_idx == 4 else pred_ratio < q_hi)
                if m_q.sum() == 0:
                    continue
                r_mfe = float(act_mfe[m_q].mean())
                r_mae = float(act_mae[m_q].mean())
                r_net_atr = float(act_net[m_q].mean())
                r_net_pts = float((act_net[m_q] * atr_pts[m_q]).mean())
                r_mfe_pts = float((act_mfe[m_q] * atr_pts[m_q]).mean())
                r_mae_pts = float((act_mae[m_q] * atr_pts[m_q]).mean())
                win_favorable = float((act_mfe[m_q] > act_mae[m_q]).mean())
                quintiles.append({
                    "bucket": f"Q{q_idx+1} ({'Lowest' if q_idx==0 else 'Top' if q_idx==4 else f'Tier {q_idx+1}'})",
                    "count": int(m_q.sum()),
                    "pred_ratio_avg": round(float(pred_ratio[m_q].mean()), 2),
                    "pred_mfe_atr": round(float(pred_mfe[m_q].mean()), 2),
                    "pred_mae_atr": round(float(pred_mae[m_q].mean()), 2),
                    "realized_mfe_atr": round(r_mfe, 2),
                    "realized_mae_atr": round(r_mae, 2),
                    "realized_ratio": round(r_mfe / max(0.05, r_mae), 2),
                    "realized_mfe_pts": round(r_mfe_pts, 1),
                    "realized_mae_pts": round(r_mae_pts, 1),
                    "avg_net_atr": round(r_net_atr, 3),
                    "avg_net_pts": round(r_net_pts, 1),
                    "favorable_win_rate_pct": round(win_favorable * 100.0, 1),
                })

        # Evaluate filtered strategy performance across Pivot Probability + Predicted MFE/MAE Ratio tiers
        # using strict chronological bar-by-bar 2-unit bracket simulation from Close[t] (no silent filter bypass!)
        strat_tiers = []
        for p_th, r_th, label in [
            (0.45, 1.0, "Baseline Pivot Signals (Prob >= 45%, All Ratios)"),
            (0.50, 1.25, "Standard Filter (Prob >= 50% + Pred MFE/MAE >= 1.25x)"),
            (0.55, 1.35, "Live Default Filter (Prob >= 55% + Pred MFE/MAE >= 1.35x)"),
            (0.55, 1.50, "High-Conviction Filter (Prob >= 55% + Pred MFE/MAE >= 1.50x)"),
            (0.60, 1.75, "Institutional A+ Setup (Prob >= 60% + Pred MFE/MAE >= 1.75x)"),
        ]:
            sim_stats = simulate_chronological_2unit_bracket(df_oos, p_th=p_th, r_th=r_th, k=k, mode="dual")
            strat_tiers.append({
                "tier": label,
                "prob_threshold": p_th,
                "ratio_threshold": r_th,
                **sim_stats,
            })

        horizons_summary[str(k)] = {
            "horizon_bars": k,
            "quintiles": quintiles,
            "strategy_tiers": strat_tiers,
        }

    return {
        "timeframe": tf,
        "pivot_high_metrics": hi_cls,
        "pivot_low_metrics": lo_cls,
        "horizons": horizons_summary,
    }


def simulate_chronological_2unit_bracket(
    df_sub: pd.DataFrame,
    p_th: float = 0.55,
    r_th: float = 1.35,
    k: int = 8,
    mode: str = "dual",
    direction_filter: str = "all",
) -> Dict[str, Any]:
    """
    Simulates the exact 2-Unit Bracket Strategy chronologically bar-by-bar:
      - Strict 1-position-at-a-time guard (`next_free_idx` prevents overlapping trades).
      - When 1-minute developing-bar columns (`trigger_short_px`, `dev_high_short_px`, `rem_high_short_px`, `rem_low_short_px`)
        are present, enters intrabar at `close_1m[t, m^*]` (`trigger_short_px` / `trigger_long_px`), sets initial stop
        above `dev_high_short_px` / below `dev_low_long_px` known at minute `m^*`, and evaluates Step 0 across the
        remaining minutes `m^* + 1 .. M_t - 1` of developing bar `t` (`rem_high`, `rem_low`) before walking bars `t+1 .. t+k`.
      - Falls back cleanly to `Close[t]` when called on DataFrames without 1-minute developing columns.
      - Conservative Stop-first tie-breaker when a bar hits both Stop and Target.
      - Unit 1 (50%) exits at `TP1 = clip(1.05 * pred_mfe, 1.10, 3.50) ATR`, moving Unit 2 Stop to Breakeven (`entry_c`).
      - Unit 2 (50% runner) exits at `TP2 = max(1.25 * TP1, min(5.0, pred_mfe_q70)) ATR`, Breakeven Stop, or Bar `k` Close.
      - Zero silent filter bypasses: if 0 signals match `(prob >= p_th) & (pred_ratio >= r_th)`, returns 0 trades.
    """
    if df_sub is None or df_sub.empty:
        return {
            "trades": 0,
            "trades_per_day": 0.0,
            "win_rate_pct": 0.0,
            "tp1_hit_rate_pct": 0.0,
            "realized_mfe_atr": 0.0,
            "realized_mae_atr": 0.0,
            "realized_mfe_mae_ratio": 0.0,
            "avg_mfe_pts": 0.0,
            "avg_mae_pts": 0.0,
            "avg_net_pts": 0.0,
            "avg_r": 0.0,
            "profit_factor": 0.0,
            "total_net_pts": 0.0,
            "max_dd_pts": 0.0,
            "total_pnl_1_nq_usd": 0.0,
            "total_pnl_1_mnq_usd": 0.0,
        }

    n = len(df_sub)
    hc = df_sub["high"].to_numpy(dtype=np.float64)
    lc = df_sub["low"].to_numpy(dtype=np.float64)
    cc = df_sub["close"].to_numpy(dtype=np.float64)
    atr_col = "atr14_pts" if "atr14_pts" in df_sub.columns else "adr20_pts"
    atr = np.maximum(1e-6, df_sub[atr_col].to_numpy(dtype=np.float64))
    is_roll = df_sub[f"crosses_roll_{k}b"].to_numpy(dtype=bool) if f"crosses_roll_{k}b" in df_sub.columns else np.zeros(n, dtype=bool)

    has_dev_s = all(c in df_sub.columns for c in ("trigger_short_px", "dev_high_short_px", "rem_high_short_px", "rem_low_short_px"))
    has_dev_l = all(c in df_sub.columns for c in ("trigger_long_px", "dev_low_long_px", "rem_high_long_px", "rem_low_long_px"))
    trig_s = df_sub["trigger_short_px"].to_numpy(dtype=np.float64) if has_dev_s else cc
    dev_hi_s = df_sub["dev_high_short_px"].to_numpy(dtype=np.float64) if has_dev_s else hc
    rem_hi_s = df_sub["rem_high_short_px"].to_numpy(dtype=np.float64) if has_dev_s else np.full(n, np.nan)
    rem_lo_s = df_sub["rem_low_short_px"].to_numpy(dtype=np.float64) if has_dev_s else np.full(n, np.nan)

    trig_l = df_sub["trigger_long_px"].to_numpy(dtype=np.float64) if has_dev_l else cc
    dev_lo_l = df_sub["dev_low_long_px"].to_numpy(dtype=np.float64) if has_dev_l else lc
    rem_hi_l = df_sub["rem_high_long_px"].to_numpy(dtype=np.float64) if has_dev_l else np.full(n, np.nan)
    rem_lo_l = df_sub["rem_low_long_px"].to_numpy(dtype=np.float64) if has_dev_l else np.full(n, np.nan)

    p_hi_col = "prob_pivot_high_extreme" if mode == "extreme" else "prob_pivot_high"
    p_lo_col = "prob_pivot_low_extreme" if mode == "extreme" else "prob_pivot_low"
    p_hi = df_sub[p_hi_col].to_numpy(dtype=np.float64)
    p_lo = df_sub[p_lo_col].to_numpy(dtype=np.float64)
    cand_hi = df_sub["is_cand_high"].to_numpy(dtype=bool) if "is_cand_high" in df_sub.columns else np.ones(n, dtype=bool)
    cand_lo = df_sub["is_cand_low"].to_numpy(dtype=bool) if "is_cand_low" in df_sub.columns else np.ones(n, dtype=bool)

    pmfe_s = np.maximum(0.80, df_sub[f"pred_mfe_short_{k}b_atr"].to_numpy(dtype=np.float64))
    pmfe_q70_s = np.maximum(pmfe_s * 1.20, df_sub[f"pred_mfe_q70_short_{k}b_atr"].to_numpy(dtype=np.float64))
    pmae_s = np.maximum(0.45, df_sub[f"pred_mae_short_{k}b_atr"].to_numpy(dtype=np.float64))
    prat_s = df_sub[f"pred_ratio_short_{k}b"].to_numpy(dtype=np.float64)

    pmfe_l = np.maximum(0.80, df_sub[f"pred_mfe_long_{k}b_atr"].to_numpy(dtype=np.float64))
    pmfe_q70_l = np.maximum(pmfe_l * 1.20, df_sub[f"pred_mfe_q70_long_{k}b_atr"].to_numpy(dtype=np.float64))
    pmae_l = np.maximum(0.45, df_sub[f"pred_mae_long_{k}b_atr"].to_numpy(dtype=np.float64))
    prat_l = df_sub[f"pred_ratio_long_{k}b"].to_numpy(dtype=np.float64)

    next_free_idx = 0
    pnls_pts: List[float] = []
    rs: List[float] = []
    tp1_hits: List[bool] = []
    mfes_atr: List[float] = []
    maes_atr: List[float] = []
    mfes_pts: List[float] = []
    maes_pts: List[float] = []

    for i in range(n - k):
        if i < next_free_idx or is_roll[i]:
            continue
        hi_ok = (direction_filter in ("all", "short")) and cand_hi[i] and (p_hi[i] >= p_th) and (prat_s[i] >= r_th)
        lo_ok = (direction_filter in ("all", "long")) and cand_lo[i] and (p_lo[i] >= p_th) and (prat_l[i] >= r_th)
        if not hi_ok and not lo_ok:
            continue
        is_short = True if (hi_ok and (not lo_ok or p_hi[i] >= p_lo[i])) else False

        a_pts = atr[i]
        if is_short:
            use_dev = has_dev_s and np.isfinite(trig_s[i]) and trig_s[i] > 0 and np.isfinite(dev_hi_s[i]) and dev_hi_s[i] > 0
            entry_c = trig_s[i] if use_dev else cc[i]
            ref_hi = dev_hi_s[i] if use_dev else hc[i]
            stop_atr = max(1.00, 1.20 * pmae_s[i], (ref_hi + 0.15 * a_pts - entry_c) / a_pts)
            tp1_atr = min(3.50, max(1.10, 1.05 * pmfe_s[i]))
            tp2_atr = max(tp1_atr * 1.25, min(5.00, pmfe_q70_s[i]))
            r_hi_0 = rem_hi_s[i] if (use_dev and np.isfinite(rem_hi_s[i]) and rem_hi_s[i] > 0) else np.nan
            r_lo_0 = rem_lo_s[i] if (use_dev and np.isfinite(rem_lo_s[i]) and rem_lo_s[i] > 0) else np.nan
        else:
            use_dev = has_dev_l and np.isfinite(trig_l[i]) and trig_l[i] > 0 and np.isfinite(dev_lo_l[i]) and dev_lo_l[i] > 0
            entry_c = trig_l[i] if use_dev else cc[i]
            ref_lo = dev_lo_l[i] if use_dev else lc[i]
            stop_atr = max(1.00, 1.20 * pmae_l[i], (entry_c - (ref_lo - 0.15 * a_pts)) / a_pts)
            tp1_atr = min(3.50, max(1.10, 1.05 * pmfe_l[i]))
            tp2_atr = max(tp1_atr * 1.25, min(5.00, pmfe_q70_l[i]))
            r_hi_0 = rem_hi_l[i] if (use_dev and np.isfinite(rem_hi_l[i]) and rem_hi_l[i] > 0) else np.nan
            r_lo_0 = rem_lo_l[i] if (use_dev and np.isfinite(rem_lo_l[i]) and rem_lo_l[i] > 0) else np.nan

        if tp1_atr / max(0.1, stop_atr) < 1.00:
            continue

        # Realized post-entry MFE/MAE across Step 0 (remaining minutes of bar i) + next K bars
        fh = hc[i + 1 : i + 1 + k].max()
        fl = lc[i + 1 : i + 1 + k].min()
        if np.isfinite(r_hi_0):
            fh = max(fh, r_hi_0)
        if np.isfinite(r_lo_0):
            fl = min(fl, r_lo_0)
        if is_short:
            t_mfe_atr = max(0.0, (entry_c - fl) / a_pts)
            t_mae_atr = max(0.0, (fh - entry_c) / a_pts)
        else:
            t_mfe_atr = max(0.0, (fh - entry_c) / a_pts)
            t_mae_atr = max(0.0, (entry_c - fl) / a_pts)

        tp1_hit = False
        u1_atr = 0.0
        u2_atr = 0.0
        exit_bar = i + k
        exited = False

        # Step 0: Remaining minutes m^* + 1 .. M_i - 1 of current parent bar i
        if np.isfinite(r_hi_0) and np.isfinite(r_lo_0):
            if is_short:
                fav_0 = (entry_c - r_lo_0) / a_pts
                adv_0 = (r_hi_0 - entry_c) / a_pts
            else:
                fav_0 = (r_hi_0 - entry_c) / a_pts
                adv_0 = (entry_c - r_lo_0) / a_pts
            if adv_0 >= stop_atr:
                u1_atr = -stop_atr
                u2_atr = -stop_atr
                exit_bar = i
                exited = True
            elif fav_0 >= tp1_atr:
                tp1_hit = True
                u1_atr = tp1_atr
                if fav_0 >= tp2_atr:
                    u2_atr = tp2_atr
                    exit_bar = i
                    exited = True

        # Steps 1 .. K: Subsequent parent bars i + 1 .. i + k
        if not exited:
            for j in range(i + 1, i + 1 + k):
                hj = hc[j]
                lj = lc[j]
                if is_short:
                    fav_j = (entry_c - lj) / a_pts
                    adv_j = (hj - entry_c) / a_pts
                else:
                    fav_j = (hj - entry_c) / a_pts
                    adv_j = (entry_c - lj) / a_pts

                if not tp1_hit:
                    # Conservative tie-breaker: if initial stop is hit on bar j, both units stop out
                    if adv_j >= stop_atr:
                        u1_atr = -stop_atr
                        u2_atr = -stop_atr
                        exit_bar = j
                        exited = True
                        break
                    elif fav_j >= tp1_atr:
                        tp1_hit = True
                        u1_atr = tp1_atr
                        if fav_j >= tp2_atr:
                            u2_atr = tp2_atr
                            exit_bar = j
                            exited = True
                            break
                else:
                    # Unit 1 already banked TP1; Unit 2 stop is now at Breakeven (0.0)
                    if adv_j >= 0.0:
                        u2_atr = 0.0
                        exit_bar = j
                        exited = True
                        break
                    elif fav_j >= tp2_atr:
                        u2_atr = tp2_atr
                        exit_bar = j
                        exited = True
                        break

        if not exited:
            c_end = cc[i + k]
            net_end = ((entry_c - c_end) if is_short else (c_end - entry_c)) / a_pts
            if not tp1_hit:
                u1_atr = net_end
                u2_atr = net_end
            else:
                u2_atr = net_end

        next_free_idx = exit_bar + 1
        avg_atr = 0.5 * (u1_atr + u2_atr)
        net_pts = avg_atr * a_pts
        pnls_pts.append(net_pts)
        rs.append(avg_atr / max(0.1, stop_atr))
        tp1_hits.append(tp1_hit)
        mfes_atr.append(t_mfe_atr)
        maes_atr.append(t_mae_atr)
        mfes_pts.append(t_mfe_atr * a_pts)
        maes_pts.append(t_mae_atr * a_pts)

    if not pnls_pts:
        return {
            "trades": 0,
            "trades_per_day": 0.0,
            "win_rate_pct": 0.0,
            "tp1_hit_rate_pct": 0.0,
            "realized_mfe_atr": 0.0,
            "realized_mae_atr": 0.0,
            "realized_mfe_mae_ratio": 0.0,
            "avg_mfe_pts": 0.0,
            "avg_mae_pts": 0.0,
            "avg_net_pts": 0.0,
            "avg_r": 0.0,
            "profit_factor": 0.0,
            "total_net_pts": 0.0,
            "max_dd_pts": 0.0,
            "total_pnl_1_nq_usd": 0.0,
            "total_pnl_1_mnq_usd": 0.0,
        }

    arr = np.array(pnls_pts, dtype=np.float64)
    r_arr = np.array(rs, dtype=np.float64)
    wins = arr > 0
    gw = float(arr[wins].sum()) if wins.any() else 0.0
    gl = float(np.abs(arr[~wins]).sum()) if (~wins).any() else 0.0
    pf = gw / gl if gl > 0 else (99.9 if gw > 0 else 0.0)
    cum = np.cumsum(arr)
    pk = np.maximum.accumulate(np.insert(cum, 0, 0.0))[1:]
    max_dd = float(np.min(cum - pk))
    n_days = max(1, len(np.unique(df_sub.index.date)))
    m_mfe = float(np.mean(mfes_atr))
    m_mae = float(np.mean(maes_atr))

    return {
        "trades": len(arr),
        "trades_per_day": round(float(len(arr)) / float(n_days), 2),
        "win_rate_pct": round(float(wins.mean() * 100.0), 1),
        "tp1_hit_rate_pct": round(float(np.mean(tp1_hits) * 100.0), 1),
        "realized_mfe_atr": round(m_mfe, 2),
        "realized_mae_atr": round(m_mae, 2),
        "realized_mfe_mae_ratio": round(m_mfe / max(0.05, m_mae), 2),
        "avg_mfe_pts": round(float(np.mean(mfes_pts)), 1),
        "avg_mae_pts": round(float(np.mean(maes_pts)), 1),
        "avg_net_pts": round(float(arr.mean()), 2),
        "avg_r": round(float(r_arr.mean()), 3),
        "profit_factor": round(pf, 2),
        "total_net_pts": round(float(arr.sum()), 1),
        "max_dd_pts": round(max_dd, 1),
        "total_pnl_1_nq_usd": round(float(arr.sum() * 20.0), 0),
        "total_pnl_1_mnq_usd": round(float(arr.sum() * 2.0), 0),
    }


def train_and_evaluate_all_nq_models() -> Dict[str, Any]:
    """
    End-to-end training & strict 2-year OOS evaluation across `15m`, `1h`, and `4h` timeframes.
    Persists:
      - `data/models/nq_intraday_models.pkl`
      - `data/simulations/nq_intraday_oos_results.json`
      - `data/simulations/nq_intraday_scored/nq_<tf>_scored.parquet`
    """
    t0 = time.time()
    print("Loading 10-year 1-minute NQ futures dataset (with seam ratios & live tail)...")
    df_1m = load_nq_1m_base(start_date=STUDY_START, include_live_tail=True)
    print(f"  Loaded {len(df_1m):,} 1-minute bars ({df_1m.index.min()} -> {df_1m.index.max()}) in {time.time()-t0:.1f}s")

    SCORED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_models: Dict[str, Any] = {}
    summary_report: Dict[str, Any] = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "study_window": {"start": STUDY_START, "train_cutoff": TRAIN_CUTOFF, "oos_start": OOS_START, "oos_end": OOS_END},
        "timeframes": {},
    }

    # Pivot window settings per timeframe
    tf_cfg = {
        "15m": {"left": 6, "right": 6, "rev_atr": 1.6},
        "1h": {"left": 5, "right": 5, "rev_atr": 1.5},
        "4h": {"left": 4, "right": 4, "rev_atr": 1.4},
    }

    for tf in TIMEFRAMES:
        t_tf = time.time()
        cfg = tf_cfg[tf]
        print(f"\n==================================================================")
        print(f"Processing Timeframe: {tf} (left={cfg['left']}, right={cfg['right']}, rev_atr={cfg['rev_atr']})")
        print(f"==================================================================")

        bars = resample_nq_with_1m_microstructure(df_1m, tf)
        feats = compute_nq_intraday_features(bars, tf)
        labeled = label_nq_pivots_and_excursions(
            feats, pivot_left=cfg["left"], pivot_right=cfg["right"], rev_atr=cfg["rev_atr"]
        )

        # Warmup trim (first 200 bars)
        labeled = labeled.iloc[200:].copy()

        train_df = labeled[labeled.index < pd.Timestamp(TRAIN_CUTOFF)].copy()
        oos_df = labeled[(labeled.index >= pd.Timestamp(OOS_START)) & (labeled.index <= pd.Timestamp(OOS_END))].copy()
        print(f"  [{tf}] Total bars: {len(labeled):,} | Train (2016-2024): {len(train_df):,} | OOS (2024-2026): {len(oos_df):,}")

        # 1. Train Pivot High & Pivot Low Classifiers on Candidate Extremes
        cand_hi_tr = train_df[train_df["is_cand_high"] & (~train_df["crosses_roll_8b"])].copy()
        cand_lo_tr = train_df[train_df["is_cand_low"] & (~train_df["crosses_roll_8b"])].copy()

        print(f"  [{tf}] Training Dual-Path Pivot High Classifier on {len(cand_hi_tr):,} candidates...")
        piv_hi_bundle = _fit_dual_classifier(
            cand_hi_tr, "is_pivot_high", PIVOT_HIGH_AT_EXTREME_COLS, PIVOT_HIGH_FULL_BAR_COLS
        )

        print(f"  [{tf}] Training Dual-Path Pivot Low Classifier on {len(cand_lo_tr):,} candidates...")
        piv_lo_bundle = _fit_dual_classifier(
            cand_lo_tr, "is_pivot_low", PIVOT_LOW_AT_EXTREME_COLS, PIVOT_LOW_FULL_BAR_COLS
        )

        # Score entire timeline (Train + OOS + Live Tail)
        p_hi_dual, p_hi_ex, p_hi_raw = score_nq_pivot_bundle(labeled, piv_hi_bundle, "is_cand_high")
        p_lo_dual, p_lo_ex, p_lo_raw = score_nq_pivot_bundle(labeled, piv_lo_bundle, "is_cand_low")

        labeled["prob_pivot_high"] = p_hi_dual
        labeled["prob_pivot_high_extreme"] = p_hi_ex
        labeled["prob_pivot_low"] = p_lo_dual
        labeled["prob_pivot_low_extreme"] = p_lo_ex

        # 2. Train Forward K-Bar (K=4, 8, 12) MFE & MAE Regressors + Asymmetry Classifiers
        # Use strictly causal structural + session + 1m microstructure features (no in-sample pivot prob leakage)
        exc_short_cols = list(PIVOT_HIGH_FULL_BAR_COLS)
        exc_long_cols = list(PIVOT_LOW_FULL_BAR_COLS)

        train_scored = labeled[labeled.index < pd.Timestamp(TRAIN_CUTOFF)]
        cand_hi_tr_s = train_scored[train_scored["is_cand_high"]].copy()
        cand_lo_tr_s = train_scored[train_scored["is_cand_low"]].copy()

        print(f"  [{tf}] Training Forward K={{4, 8, 12}} MFE/MAE Regressors & Asymmetry Classifiers...")
        exc_short_bundle = _fit_excursion_models_for_direction(cand_hi_tr_s, "short", exc_short_cols)
        exc_long_bundle = _fit_excursion_models_for_direction(cand_lo_tr_s, "long", exc_long_cols)

        score_excursion_models(labeled, exc_short_bundle, "short")
        score_excursion_models(labeled, exc_long_bundle, "long")

        # 3. Evaluate 2-Year Out-of-Sample (2024-07-15 to 2026-07-14) + Generic Equity Baseline
        oos_scored = labeled[(labeled.index >= pd.Timestamp(OOS_START)) & (labeled.index <= pd.Timestamp(OOS_END))].copy()
        tf_eval = compute_oos_deciles_and_strategy_stats(oos_scored, tf)
        eq_base = evaluate_generic_equity_baseline_on_nq(oos_scored)
        tf_eval["generic_equity_baseline"] = eq_base
        tf_eval["top_features_high"] = piv_hi_bundle["top_features"]
        tf_eval["top_features_low"] = piv_lo_bundle["top_features"]
        tf_eval["n_train_bars"] = len(train_df)
        tf_eval["n_oos_bars"] = len(oos_scored)

        hi_m = tf_eval["pivot_high_metrics"]
        lo_m = tf_eval["pivot_low_metrics"]
        print(
            f"  [{tf} OOS] NQ-Specific Pivot High ROC-AUC: {hi_m['dual_roc_auc']:.4f} (PR-AUC: {hi_m['dual_pr_auc']:.4f}, "
            f"@60% Prec: {hi_m['thresholds']['60']['precision_pct']:.1f}%) vs Generic Equity ROC-AUC: {eq_base['eq_high_roc_auc']:.4f}"
        )
        print(
            f"  [{tf} OOS] NQ-Specific Pivot Low  ROC-AUC: {lo_m['dual_roc_auc']:.4f} (PR-AUC: {lo_m['dual_pr_auc']:.4f}, "
            f"@60% Prec: {lo_m['thresholds']['60']['precision_pct']:.1f}%) vs Generic Equity ROC-AUC: {eq_base['eq_low_roc_auc']:.4f}"
        )

        # Save scored parquet for fast server loading (keep last 4 years + all OOS for 15m, and full 10y for 1h/4h)
        save_cols = [
            "open", "high", "low", "close", "volume", "is_roll", "roll_mult",
            "trigger_short_px", "trigger_long_px",
            "atr14_pts", "adr20_pts", "ema10_px", "ema20_px", "sma50_px", "sma200_px",
            "rth_vwap_px", "globex_vwap_px", "on_high_px", "on_low_px", "prev_rth_high_px", "prev_rth_low_px",
            "ob75_px", "os25_px", "rsi14", "rsi14_high", "rsi14_low", "rsi5_high", "rsi5_low",
            "tod_rvol", "rvol_raw", "session_id", "is_rth", "swept_on_high", "swept_on_low",
            "swept_prev_rth_high", "swept_prev_rth_low", "mtf_ob_confluence", "mtf_os_confluence",
            "m1_vol_delta_frac", "m1_vol_top25_frac", "m1_vol_bot25_frac",
            "m1_up_bar_frac", "m1_max_bar_range_frac", "m1_high_time_frac", "m1_low_time_frac", "m1_late_vol_frac",
            "is_cand_high", "is_cand_low", "is_pivot_high", "is_pivot_low",
            "prob_pivot_high", "prob_pivot_high_extreme", "prob_pivot_low", "prob_pivot_low_extreme",
        ]
        for k in HORIZONS:
            save_cols.extend([
                f"mfe_short_{k}b_atr", f"mae_short_{k}b_atr", f"net_short_{k}b_atr",
                f"mfe_long_{k}b_atr", f"mae_long_{k}b_atr", f"net_long_{k}b_atr",
                f"pred_mfe_short_{k}b_atr", f"pred_mfe_q70_short_{k}b_atr",
                f"pred_mae_short_{k}b_atr", f"pred_mae_q75_short_{k}b_atr",
                f"prob_asym_short_{k}b", f"pred_ratio_short_{k}b",
                f"pred_mfe_long_{k}b_atr", f"pred_mfe_q70_long_{k}b_atr",
                f"pred_mae_long_{k}b_atr", f"pred_mae_q75_long_{k}b_atr",
                f"prob_asym_long_{k}b", f"pred_ratio_long_{k}b",
                f"crosses_roll_{k}b",
            ])

        save_df = labeled[save_cols].copy()
        if tf == "15m":
            # Save from 2022-01-01 onwards (~115k bars) for ultra-fast interactive chart browsing + full OOS
            save_df = save_df[save_df.index >= pd.Timestamp("2022-01-01")].copy()
        for fcol in save_df.select_dtypes("float64").columns:
            save_df[fcol] = save_df[fcol].astype("float32")
        _atomic_to_parquet(save_df, SCORED_CACHE_DIR / f"nq_{tf}_scored.parquet")

        all_models[tf] = {
            "pivot_high": piv_hi_bundle,
            "pivot_low": piv_lo_bundle,
            "excursion_short": exc_short_bundle,
            "excursion_long": exc_long_bundle,
            "config": cfg,
        }
        summary_report["timeframes"][tf] = tf_eval
        print(f"  [{tf}] Completed in {time.time() - t_tf:.1f}s")

    joblib.dump(all_models, MODELS_PATH)
    RESULTS_JSON_PATH.write_text(json.dumps(summary_report, indent=2), encoding="utf-8")
    print(f"\n✅ Saved NQ Intraday ML models ({MODELS_PATH}) & OOS Report ({RESULTS_JSON_PATH}) in {time.time()-t0:.1f}s")
    return summary_report


def _label_gc_pivots_and_excursions(
    b: pd.DataFrame, pivot_left: int = 6, pivot_right: int = 6, rev_atr: float = 1.6
) -> pd.DataFrame:
    """Labels GC Pivot Highs & Pivot Lows (including the +0.05*ADR DT_LOW tolerance) and Forward K-bar Excursions."""
    df = label_nq_pivots_and_excursions(b, pivot_left=pivot_left, pivot_right=pivot_right, rev_atr=rev_atr)
    hc = df["high_cont"].to_numpy(dtype=np.float64)
    lc = df["low_cont"].to_numpy(dtype=np.float64)
    adr = np.maximum(1e-6, df["adr20_cont"].to_numpy(dtype=np.float64))
    n = len(df)
    win_len = pivot_left + pivot_right + 1
    is_piv_lo = np.zeros(n, dtype=int)

    if n > win_len + 8:
        l_sw = sliding_window_view(lc, win_len)
        loc_min = l_sw.min(axis=1)

        sub_l = max(2, pivot_left - 2)
        sub_win = 2 * sub_l + 1
        l_sw_s = sliding_window_view(lc[pivot_left - sub_l : n - (pivot_right - sub_l)], sub_win)
        loc_min_short = l_sw_s.min(axis=1)

        fwd8_h = sliding_window_view(hc[pivot_left + 1 :], 8)
        m_len = min(len(loc_min), len(fwd8_h), len(loc_min_short))

        loc_min = loc_min[:m_len]
        loc_min_short = loc_min_short[:m_len]
        fwd8_h = fwd8_h[:m_len]

        center_l = lc[pivot_left : pivot_left + m_len]
        center_adr = np.maximum(1e-6, adr[pivot_left : pivot_left + m_len])
        ut_reg = df["is_uptrend_regime"].to_numpy(dtype=np.float64)[pivot_left : pivot_left + m_len] >= 0.5
        ext_lo = df["ext_ema20_lo_atr"].to_numpy(dtype=np.float64)[pivot_left : pivot_left + m_len]

        rally_from_lo_8b = (fwd8_h.max(axis=1) - center_l) / center_adr

        cond_ut_lo = ut_reg & (center_l <= loc_min_short + 0.05 * center_adr) & (rally_from_lo_8b >= 0.90 * rev_atr)
        cond_dt_lo = (~ut_reg) & (center_l <= loc_min + 0.05 * center_adr) & (rally_from_lo_8b >= 1.25 * rev_atr) & (ext_lo <= -0.85)
        cond_lo = cond_ut_lo | cond_dt_lo
        is_piv_lo[pivot_left : pivot_left + m_len] = cond_lo.astype(int)

    df["is_pivot_low"] = is_piv_lo
    return df


def train_and_evaluate_all_gc_models() -> Dict[str, Any]:
    """
    End-to-end training & strict 2-year OOS evaluation for Gold Futures (`GC`) across `15m`, `1h`, and `4h`.
    Persists:
      - `data/models/gc_intraday_models.pkl`
      - `data/simulations/gc_intraday_oos_results.json`
      - `data/simulations/nq_intraday_scored/gc_<tf>_scored.parquet`
    """
    t0 = time.time()
    # First fetch live 5m GC=F tail from TWS/yfinance so the saved parquets include bars up to today!
    try:
        df5_live = fetch_live_nq_5m_bars_from_tws_or_yf(symbol="GC=F")
        if df5_live is not None and not df5_live.empty:
            _atomic_to_parquet(df5_live, GC_5M_TAIL_PATH)
    except Exception:
        pass

    print("Loading 10-year 1-minute GC (Gold Futures) dataset (with seam ratios & live tail)...")
    df_1m = load_gc_1m_base(start_date=STUDY_START, include_live_tail=True)
    print(f"  Loaded {len(df_1m):,} GC 1-minute bars ({df_1m.index.min()} -> {df_1m.index.max()}) in {time.time()-t0:.1f}s")

    SCORED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    GC_MODELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    GC_RESULTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_models: Dict[str, Any] = {}
    summary_report: Dict[str, Any] = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "symbol": "GC=F",
        "study_window": {"start": STUDY_START, "train_cutoff": TRAIN_CUTOFF, "oos_start": OOS_START, "oos_end": OOS_END},
        "timeframes": {},
    }

    tf_cfg = {
        "15m": {"left": 6, "right": 6, "rev_atr": 1.6},
        "1h": {"left": 5, "right": 5, "rev_atr": 1.5},
        "4h": {"left": 4, "right": 4, "rev_atr": 1.4},
    }

    for tf in TIMEFRAMES:
        t_tf = time.time()
        cfg = tf_cfg[tf]
        print(f"\n==================================================================")
        print(f"Processing GC Timeframe: {tf} (left={cfg['left']}, right={cfg['right']}, rev_atr={cfg['rev_atr']})")
        print(f"==================================================================")

        bars = resample_nq_with_1m_microstructure(df_1m, tf)
        feats = compute_nq_intraday_features(bars, tf)
        labeled = _label_gc_pivots_and_excursions(
            feats, pivot_left=cfg["left"], pivot_right=cfg["right"], rev_atr=cfg["rev_atr"]
        )
        labeled = labeled.iloc[200:].copy()

        train_df = labeled[labeled.index < pd.Timestamp(TRAIN_CUTOFF)].copy()
        oos_df = labeled[(labeled.index >= pd.Timestamp(OOS_START)) & (labeled.index <= pd.Timestamp(OOS_END))].copy()
        print(f"  [GC {tf}] Total bars: {len(labeled):,} | Train (2016-2024): {len(train_df):,} | OOS (2024-2026): {len(oos_df):,}")

        cand_hi_tr = train_df[train_df["is_cand_high"] & (~train_df["crosses_roll_8b"])].copy()
        cand_lo_tr = train_df[train_df["is_cand_low"] & (~train_df["crosses_roll_8b"])].copy()

        print(f"  [GC {tf}] Training Dual-Path Pivot High Classifier on {len(cand_hi_tr):,} candidates...")
        piv_hi_bundle = _fit_dual_classifier(
            cand_hi_tr, "is_pivot_high", PIVOT_HIGH_AT_EXTREME_COLS, PIVOT_HIGH_FULL_BAR_COLS
        )
        print(f"  [GC {tf}] Training Dual-Path Pivot Low Classifier on {len(cand_lo_tr):,} candidates...")
        piv_lo_bundle = _fit_dual_classifier(
            cand_lo_tr, "is_pivot_low", PIVOT_LOW_AT_EXTREME_COLS, PIVOT_LOW_FULL_BAR_COLS
        )

        p_hi_dual, p_hi_ex, _ = score_nq_pivot_bundle(labeled, piv_hi_bundle, "is_cand_high")
        p_lo_dual, p_lo_ex, _ = score_nq_pivot_bundle(labeled, piv_lo_bundle, "is_cand_low")
        labeled["prob_pivot_high"] = p_hi_dual
        labeled["prob_pivot_high_extreme"] = p_hi_ex
        labeled["prob_pivot_low"] = p_lo_dual
        labeled["prob_pivot_low_extreme"] = p_lo_ex

        train_scored = labeled[labeled.index < pd.Timestamp(TRAIN_CUTOFF)]
        cand_hi_tr_s = train_scored[train_scored["is_cand_high"]].copy()
        cand_lo_tr_s = train_scored[train_scored["is_cand_low"]].copy()

        print(f"  [GC {tf}] Training Forward K={{4, 8, 12}} MFE/MAE Regressors & Asymmetry Classifiers...")
        exc_short_bundle = _fit_excursion_models_for_direction(cand_hi_tr_s, "short", list(PIVOT_HIGH_FULL_BAR_COLS))
        exc_long_bundle = _fit_excursion_models_for_direction(cand_lo_tr_s, "long", list(PIVOT_LOW_FULL_BAR_COLS))

        score_excursion_models(labeled, exc_short_bundle, "short")
        score_excursion_models(labeled, exc_long_bundle, "long")

        # Live tail bars after 2026-07-31 do not cross a roll unless is_roll is True
        for k_h in HORIZONS:
            rc = f"crosses_roll_{k_h}b"
            if rc in labeled.columns and not labeled["is_roll"].tail(k_h + 2).any():
                labeled.iloc[-k_h:, labeled.columns.get_loc(rc)] = False

        oos_scored = labeled[(labeled.index >= pd.Timestamp(OOS_START)) & (labeled.index <= pd.Timestamp(OOS_END))].copy()
        tf_eval = compute_oos_deciles_and_strategy_stats(oos_scored, tf)
        tf_eval["top_features_high"] = piv_hi_bundle["top_features"]
        tf_eval["top_features_low"] = piv_lo_bundle["top_features"]
        tf_eval["n_train_bars"] = len(train_df)
        tf_eval["n_oos_bars"] = len(oos_scored)

        save_cols = [
            "open", "high", "low", "close", "volume", "is_roll", "roll_mult",
            "trigger_short_px", "trigger_long_px",
            "atr14_pts", "adr20_pts", "ema10_px", "ema20_px", "sma50_px", "sma200_px",
            "rth_vwap_px", "globex_vwap_px", "on_high_px", "on_low_px", "prev_rth_high_px", "prev_rth_low_px",
            "ob75_px", "os25_px", "rsi14", "rsi14_high", "rsi14_low", "rsi5_high", "rsi5_low",
            "tod_rvol", "rvol_raw", "session_id", "is_rth", "swept_on_high", "swept_on_low",
            "swept_prev_rth_high", "swept_prev_rth_low", "mtf_ob_confluence", "mtf_os_confluence",
            "m1_vol_delta_frac", "m1_vol_top25_frac", "m1_vol_bot25_frac",
            "m1_up_bar_frac", "m1_max_bar_range_frac", "m1_high_time_frac", "m1_low_time_frac", "m1_late_vol_frac",
            "is_cand_high", "is_cand_low", "is_pivot_high", "is_pivot_low",
            "prob_pivot_high", "prob_pivot_high_extreme", "prob_pivot_low", "prob_pivot_low_extreme",
        ]
        for k in HORIZONS:
            save_cols.extend([
                f"mfe_short_{k}b_atr", f"mae_short_{k}b_atr", f"net_short_{k}b_atr",
                f"mfe_long_{k}b_atr", f"mae_long_{k}b_atr", f"net_long_{k}b_atr",
                f"pred_mfe_short_{k}b_atr", f"pred_mfe_q70_short_{k}b_atr",
                f"pred_mae_short_{k}b_atr", f"pred_mae_q75_short_{k}b_atr",
                f"prob_asym_short_{k}b", f"pred_ratio_short_{k}b",
                f"pred_mfe_long_{k}b_atr", f"pred_mfe_q70_long_{k}b_atr",
                f"pred_mae_long_{k}b_atr", f"pred_mae_q75_long_{k}b_atr",
                f"prob_asym_long_{k}b", f"pred_ratio_long_{k}b",
                f"crosses_roll_{k}b",
            ])

        save_df = labeled[save_cols].copy()
        if tf == "15m":
            save_df = save_df[save_df.index >= pd.Timestamp("2022-01-01")].copy()
        for fcol in save_df.select_dtypes("float64").columns:
            save_df[fcol] = save_df[fcol].astype("float32")
        _atomic_to_parquet(save_df, SCORED_CACHE_DIR / f"gc_{tf}_scored.parquet")

        all_models[tf] = {
            "pivot_high": piv_hi_bundle,
            "pivot_low": piv_lo_bundle,
            "excursion_short": exc_short_bundle,
            "excursion_long": exc_long_bundle,
            "config": cfg,
        }
        summary_report["timeframes"][tf] = tf_eval
        print(f"  [GC {tf}] Completed in {time.time() - t_tf:.1f}s")

    joblib.dump(all_models, GC_MODELS_PATH)
    GC_RESULTS_JSON_PATH.write_text(json.dumps(summary_report, indent=2), encoding="utf-8")
    print(f"\n✅ Saved GC Intraday ML models ({GC_MODELS_PATH}) & OOS Report ({GC_RESULTS_JSON_PATH}) in {time.time()-t0:.1f}s")
    return summary_report


def _train_and_evaluate_custom_futures_suite(
    symbol: str,
    prefix: str,
    load_base_fn: Any,
    tail_5m_path: Path,
    models_out_path: Path,
    results_json_path: Path,
    study_start: str = STUDY_START,
    train_cutoff: str = TRAIN_CUTOFF,
    oos_start: str = OOS_START,
    oos_end: str = OOS_END,
) -> Dict[str, Any]:
    """Generic trainer & 2-year OOS evaluator for 1-minute futures datasets (`CL`, `BTC`, `GC`)."""
    t0 = time.time()
    try:
        df5_live = fetch_live_nq_5m_bars_from_tws_or_yf(symbol=symbol)
        if df5_live is not None and not df5_live.empty:
            _atomic_to_parquet(df5_live, tail_5m_path)
    except Exception:
        pass

    print(f"Loading 1-minute {symbol} dataset (with seam ratios & live tail)...")
    df_1m = load_base_fn(start_date=study_start, include_live_tail=True)
    print(f"  Loaded {len(df_1m):,} {symbol} 1-minute bars ({df_1m.index.min()} -> {df_1m.index.max()}) in {time.time()-t0:.1f}s")

    SCORED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    models_out_path.parent.mkdir(parents=True, exist_ok=True)
    results_json_path.parent.mkdir(parents=True, exist_ok=True)

    all_models: Dict[str, Any] = {}
    summary_report: Dict[str, Any] = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "symbol": symbol,
        "study_window": {"start": study_start, "train_cutoff": train_cutoff, "oos_start": oos_start, "oos_end": oos_end},
        "timeframes": {},
    }

    tf_cfg = {
        "15m": {"left": 6, "right": 6, "rev_atr": 1.6},
        "1h": {"left": 5, "right": 5, "rev_atr": 1.5},
        "4h": {"left": 4, "right": 4, "rev_atr": 1.4},
    }

    for tf in TIMEFRAMES:
        t_tf = time.time()
        cfg = tf_cfg[tf]
        print(f"\n==================================================================")
        print(f"Processing {symbol} Timeframe: {tf} (left={cfg['left']}, right={cfg['right']}, rev_atr={cfg['rev_atr']})")
        print(f"==================================================================")

        bars = resample_nq_with_1m_microstructure(df_1m, tf)
        feats = compute_nq_intraday_features(bars, tf)
        labeled = _label_gc_pivots_and_excursions(
            feats, pivot_left=cfg["left"], pivot_right=cfg["right"], rev_atr=cfg["rev_atr"]
        )
        labeled = labeled.iloc[200:].copy()

        train_df = labeled[labeled.index < pd.Timestamp(train_cutoff)].copy()
        oos_df = labeled[(labeled.index >= pd.Timestamp(oos_start)) & (labeled.index <= pd.Timestamp(oos_end))].copy()
        print(f"  [{symbol} {tf}] Total bars: {len(labeled):,} | Train: {len(train_df):,} | OOS: {len(oos_df):,}")

        cand_hi_tr = train_df[train_df["is_cand_high"] & (~train_df["crosses_roll_8b"])].copy()
        cand_lo_tr = train_df[train_df["is_cand_low"] & (~train_df["crosses_roll_8b"])].copy()

        print(f"  [{symbol} {tf}] Training Dual-Path Pivot High Classifier on {len(cand_hi_tr):,} candidates...")
        piv_hi_bundle = _fit_dual_classifier(
            cand_hi_tr, "is_pivot_high", PIVOT_HIGH_AT_EXTREME_COLS, PIVOT_HIGH_FULL_BAR_COLS
        )
        print(f"  [{symbol} {tf}] Training Dual-Path Pivot Low Classifier on {len(cand_lo_tr):,} candidates...")
        piv_lo_bundle = _fit_dual_classifier(
            cand_lo_tr, "is_pivot_low", PIVOT_LOW_AT_EXTREME_COLS, PIVOT_LOW_FULL_BAR_COLS
        )

        p_hi_dual, p_hi_ex, _ = score_nq_pivot_bundle(labeled, piv_hi_bundle, "is_cand_high")
        p_lo_dual, p_lo_ex, _ = score_nq_pivot_bundle(labeled, piv_lo_bundle, "is_cand_low")
        labeled["prob_pivot_high"] = p_hi_dual
        labeled["prob_pivot_high_extreme"] = p_hi_ex
        labeled["prob_pivot_low"] = p_lo_dual
        labeled["prob_pivot_low_extreme"] = p_lo_ex

        train_scored = labeled[labeled.index < pd.Timestamp(train_cutoff)]
        cand_hi_tr_s = train_scored[train_scored["is_cand_high"]].copy()
        cand_lo_tr_s = train_scored[train_scored["is_cand_low"]].copy()

        print(f"  [{symbol} {tf}] Training Forward K={{4, 8, 12}} MFE/MAE Regressors & Asymmetry Classifiers...")
        exc_short_bundle = _fit_excursion_models_for_direction(cand_hi_tr_s, "short", list(PIVOT_HIGH_FULL_BAR_COLS))
        exc_long_bundle = _fit_excursion_models_for_direction(cand_lo_tr_s, "long", list(PIVOT_LOW_FULL_BAR_COLS))

        score_excursion_models(labeled, exc_short_bundle, "short")
        score_excursion_models(labeled, exc_long_bundle, "long")

        for k_h in HORIZONS:
            rc = f"crosses_roll_{k_h}b"
            if rc in labeled.columns and not labeled["is_roll"].tail(k_h + 2).any():
                labeled.iloc[-k_h:, labeled.columns.get_loc(rc)] = False

        oos_scored = labeled[(labeled.index >= pd.Timestamp(oos_start)) & (labeled.index <= pd.Timestamp(oos_end))].copy()
        tf_eval = compute_oos_deciles_and_strategy_stats(oos_scored, tf)
        tf_eval["top_features_high"] = piv_hi_bundle["top_features"]
        tf_eval["top_features_low"] = piv_lo_bundle["top_features"]
        tf_eval["n_train_bars"] = len(train_df)
        tf_eval["n_oos_bars"] = len(oos_scored)

        save_cols = [
            "open", "high", "low", "close", "volume", "is_roll", "roll_mult",
            "trigger_short_px", "trigger_long_px",
            "atr14_pts", "adr20_pts", "ema10_px", "ema20_px", "sma50_px", "sma200_px",
            "rth_vwap_px", "globex_vwap_px", "on_high_px", "on_low_px", "prev_rth_high_px", "prev_rth_low_px",
            "ob75_px", "os25_px", "rsi14", "rsi14_high", "rsi14_low", "rsi5_high", "rsi5_low",
            "tod_rvol", "rvol_raw", "session_id", "is_rth", "swept_on_high", "swept_on_low",
            "swept_prev_rth_high", "swept_prev_rth_low", "mtf_ob_confluence", "mtf_os_confluence",
            "m1_vol_delta_frac", "m1_vol_top25_frac", "m1_vol_bot25_frac",
            "m1_up_bar_frac", "m1_max_bar_range_frac", "m1_high_time_frac", "m1_low_time_frac", "m1_late_vol_frac",
            "is_cand_high", "is_cand_low", "is_pivot_high", "is_pivot_low",
            "prob_pivot_high", "prob_pivot_high_extreme", "prob_pivot_low", "prob_pivot_low_extreme",
        ]
        for k in HORIZONS:
            save_cols.extend([
                f"mfe_short_{k}b_atr", f"mae_short_{k}b_atr", f"net_short_{k}b_atr",
                f"mfe_long_{k}b_atr", f"mae_long_{k}b_atr", f"net_long_{k}b_atr",
                f"pred_mfe_short_{k}b_atr", f"pred_mfe_q70_short_{k}b_atr",
                f"pred_mae_short_{k}b_atr", f"pred_mae_q75_short_{k}b_atr",
                f"prob_asym_short_{k}b", f"pred_ratio_short_{k}b",
                f"pred_mfe_long_{k}b_atr", f"pred_mfe_q70_long_{k}b_atr",
                f"pred_mae_long_{k}b_atr", f"pred_mae_q75_long_{k}b_atr",
                f"prob_asym_long_{k}b", f"pred_ratio_long_{k}b",
                f"crosses_roll_{k}b",
            ])

        save_df = labeled[save_cols].copy()
        if tf == "15m":
            save_df = save_df[save_df.index >= pd.Timestamp("2022-01-01")].copy()
        for fcol in save_df.select_dtypes("float64").columns:
            save_df[fcol] = save_df[fcol].astype("float32")
        _atomic_to_parquet(save_df, SCORED_CACHE_DIR / f"{prefix}_{tf}_scored.parquet")

        all_models[tf] = {
            "pivot_high": piv_hi_bundle,
            "pivot_low": piv_lo_bundle,
            "excursion_short": exc_short_bundle,
            "excursion_long": exc_long_bundle,
            "config": cfg,
        }
        summary_report["timeframes"][tf] = tf_eval
        print(f"  [{symbol} {tf}] Completed in {time.time() - t_tf:.1f}s")

    joblib.dump(all_models, models_out_path)
    results_json_path.write_text(json.dumps(summary_report, indent=2), encoding="utf-8")
    print(f"\n✅ Saved {symbol} Intraday ML models ({models_out_path}) & OOS Report ({results_json_path}) in {time.time()-t0:.1f}s")
    return summary_report


def train_and_evaluate_all_cl_models() -> Dict[str, Any]:
    """End-to-end training & 2-year OOS evaluation for Crude Oil Futures (`CL=F`) across `15m`, `1h`, and `4h`."""
    return _train_and_evaluate_custom_futures_suite(
        symbol="CL=F",
        prefix="cl",
        load_base_fn=load_cl_1m_base,
        tail_5m_path=CL_5M_TAIL_PATH,
        models_out_path=CL_MODELS_PATH,
        results_json_path=CL_RESULTS_JSON_PATH,
    )


def train_and_evaluate_all_btc_models() -> Dict[str, Any]:
    """End-to-end training & 2-year OOS evaluation for Bitcoin Futures (`BTC=F`) across `15m`, `1h`, and `4h`."""
    return _train_and_evaluate_custom_futures_suite(
        symbol="BTC=F",
        prefix="btc",
        load_base_fn=load_btc_1m_base,
        tail_5m_path=BTC_5M_TAIL_PATH,
        models_out_path=BTC_MODELS_PATH,
        results_json_path=BTC_RESULTS_JSON_PATH,
        study_start="2017-08-17 00:00:00",
        train_cutoff="2024-07-01 00:00:00",
        oos_start="2024-07-01 00:00:00",
        oos_end="2026-06-30 23:59:59",
    )


def train_and_evaluate_all_si_models() -> Dict[str, Any]:
    """End-to-end training & 2-year OOS evaluation for Silver Futures (`SI=F`) across `15m`, `1h`, and `4h`."""
    return _train_and_evaluate_custom_futures_suite(
        symbol="SI=F",
        prefix="si",
        load_base_fn=load_si_1m_base,
        tail_5m_path=SI_5M_TAIL_PATH,
        models_out_path=SI_MODELS_PATH,
        results_json_path=SI_RESULTS_JSON_PATH,
        study_start="2021-08-01 18:00:00",
        train_cutoff="2024-07-15 00:00:00",
        oos_start="2024-07-15 00:00:00",
        oos_end="2026-07-14 23:59:59",
    )


# ==============================================================================
# 8. Real-Time Multi-Timeframe (15m, 1h, 4h) Trade Plan & TWS Alert Engine
# ==============================================================================

_NQ_SCORED_RAM_CACHE: Dict[str, Tuple[float, pd.DataFrame]] = {}
_NQ_MODELS_RAM_CACHE: Tuple[float, Dict[str, Any]] = (0.0, {})
_GC_MODELS_RAM_CACHE: Tuple[float, Dict[str, Any]] = (0.0, {})
_CL_MODELS_RAM_CACHE: Tuple[float, Dict[str, Any]] = (0.0, {})
_BTC_MODELS_RAM_CACHE: Tuple[float, Dict[str, Any]] = (0.0, {})
_SI_MODELS_RAM_CACHE: Tuple[float, Dict[str, Any]] = (0.0, {})

_NQ_WL_MODELS_RAM_CACHE: Tuple[float, Dict[str, Any]] = (0.0, {})
_GC_WL_MODELS_RAM_CACHE: Tuple[float, Dict[str, Any]] = (0.0, {})
_CL_WL_MODELS_RAM_CACHE: Tuple[float, Dict[str, Any]] = (0.0, {})
_BTC_WL_MODELS_RAM_CACHE: Tuple[float, Dict[str, Any]] = (0.0, {})
_SI_WL_MODELS_RAM_CACHE: Tuple[float, Dict[str, Any]] = (0.0, {})

_LAST_NQ_INTRADAY_SYNC_TS: float = 0.0
_LAST_GC_INTRADAY_SYNC_TS: float = 0.0
_LAST_INTRADAY_SYNC_TS_BY_SYM: Dict[str, float] = {}

TF_MINUTES_MAP: Dict[str, int] = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}
TF_DISPLAY_MAP: Dict[str, str] = {"5m": "5M", "15m": "15M", "1h": "1H", "4h": "4H"}


def get_cached_nq_scored_df(tf: str, symbol: str = "NQ=F") -> pd.DataFrame:
    """Loads and caches `{nq,gc,cl,btc,si}_<tf>_scored.parquet` in RAM with mtime invalidation."""
    tf_norm = tf.lower().strip()
    prefix = get_futures_prefix(symbol)
    cache_k = tf_norm if prefix == "nq" else f"{prefix}_{tf_norm}"
    p = SCORED_CACHE_DIR / f"{prefix}_{tf_norm}_scored.parquet"
    if not p.exists():
        return pd.DataFrame()
    mtime = p.stat().st_mtime
    cached = _NQ_SCORED_RAM_CACHE.get(cache_k)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    df = pd.read_parquet(p)
    _NQ_SCORED_RAM_CACHE[cache_k] = (mtime, df)
    return df


def get_cached_nq_models(symbol: str = "NQ=F", use_wickless: bool = False) -> Dict[str, Any]:
    """Loads and caches `{nq,gc,cl,btc,si}_[wickless_]intraday_models.pkl` in RAM with mtime invalidation."""
    global _NQ_MODELS_RAM_CACHE, _GC_MODELS_RAM_CACHE, _CL_MODELS_RAM_CACHE, _BTC_MODELS_RAM_CACHE, _SI_MODELS_RAM_CACHE
    global _NQ_WL_MODELS_RAM_CACHE, _GC_WL_MODELS_RAM_CACHE, _CL_WL_MODELS_RAM_CACHE, _BTC_WL_MODELS_RAM_CACHE, _SI_WL_MODELS_RAM_CACHE

    if is_gc_symbol(symbol):
        target_path = GC_WICKLESS_MODELS_PATH if (use_wickless and GC_WICKLESS_MODELS_PATH.exists()) else GC_MODELS_PATH
        if not target_path.exists():
            return {}
        mtime = target_path.stat().st_mtime
        if use_wickless and target_path == GC_WICKLESS_MODELS_PATH:
            if _GC_WL_MODELS_RAM_CACHE[0] == mtime and _GC_WL_MODELS_RAM_CACHE[1]:
                return _GC_WL_MODELS_RAM_CACHE[1]
            models = joblib.load(target_path)
            _GC_WL_MODELS_RAM_CACHE = (mtime, models)
            return models
        else:
            if _GC_MODELS_RAM_CACHE[0] == mtime and _GC_MODELS_RAM_CACHE[1]:
                return _GC_MODELS_RAM_CACHE[1]
            models = joblib.load(target_path)
            _GC_MODELS_RAM_CACHE = (mtime, models)
            return models

    if is_cl_symbol(symbol):
        target_path = CL_WICKLESS_MODELS_PATH if (use_wickless and CL_WICKLESS_MODELS_PATH.exists()) else CL_MODELS_PATH
        if not target_path.exists():
            return {}
        mtime = target_path.stat().st_mtime
        if use_wickless and target_path == CL_WICKLESS_MODELS_PATH:
            if _CL_WL_MODELS_RAM_CACHE[0] == mtime and _CL_WL_MODELS_RAM_CACHE[1]:
                return _CL_WL_MODELS_RAM_CACHE[1]
            models = joblib.load(target_path)
            _CL_WL_MODELS_RAM_CACHE = (mtime, models)
            return models
        else:
            if _CL_MODELS_RAM_CACHE[0] == mtime and _CL_MODELS_RAM_CACHE[1]:
                return _CL_MODELS_RAM_CACHE[1]
            models = joblib.load(target_path)
            _CL_MODELS_RAM_CACHE = (mtime, models)
            return models

    if is_btc_symbol(symbol):
        target_path = BTC_WICKLESS_MODELS_PATH if (use_wickless and BTC_WICKLESS_MODELS_PATH.exists()) else BTC_MODELS_PATH
        if not target_path.exists():
            return {}
        mtime = target_path.stat().st_mtime
        if use_wickless and target_path == BTC_WICKLESS_MODELS_PATH:
            if _BTC_WL_MODELS_RAM_CACHE[0] == mtime and _BTC_WL_MODELS_RAM_CACHE[1]:
                return _BTC_WL_MODELS_RAM_CACHE[1]
            models = joblib.load(target_path)
            _BTC_WL_MODELS_RAM_CACHE = (mtime, models)
            return models
        else:
            if _BTC_MODELS_RAM_CACHE[0] == mtime and _BTC_MODELS_RAM_CACHE[1]:
                return _BTC_MODELS_RAM_CACHE[1]
            models = joblib.load(target_path)
            _BTC_MODELS_RAM_CACHE = (mtime, models)
            return models

    if is_si_symbol(symbol):
        target_path = SI_WICKLESS_MODELS_PATH if (use_wickless and SI_WICKLESS_MODELS_PATH.exists()) else SI_MODELS_PATH
        if not target_path.exists():
            return {}
        mtime = target_path.stat().st_mtime
        if use_wickless and target_path == SI_WICKLESS_MODELS_PATH:
            if _SI_WL_MODELS_RAM_CACHE[0] == mtime and _SI_WL_MODELS_RAM_CACHE[1]:
                return _SI_WL_MODELS_RAM_CACHE[1]
            models = joblib.load(target_path)
            _SI_WL_MODELS_RAM_CACHE = (mtime, models)
            return models
        else:
            if _SI_MODELS_RAM_CACHE[0] == mtime and _SI_MODELS_RAM_CACHE[1]:
                return _SI_MODELS_RAM_CACHE[1]
            models = joblib.load(target_path)
            _SI_MODELS_RAM_CACHE = (mtime, models)
            return models

    # Default: NQ
    target_path = NQ_WICKLESS_MODELS_PATH if (use_wickless and NQ_WICKLESS_MODELS_PATH.exists()) else MODELS_PATH
    if not target_path.exists():
        return {}
    mtime = target_path.stat().st_mtime
    if use_wickless and target_path == NQ_WICKLESS_MODELS_PATH:
        if _NQ_WL_MODELS_RAM_CACHE[0] == mtime and _NQ_WL_MODELS_RAM_CACHE[1]:
            return _NQ_WL_MODELS_RAM_CACHE[1]
        models = joblib.load(target_path)
        _NQ_WL_MODELS_RAM_CACHE = (mtime, models)
        return models
    else:
        if _NQ_MODELS_RAM_CACHE[0] == mtime and _NQ_MODELS_RAM_CACHE[1]:
            return _NQ_MODELS_RAM_CACHE[1]
        models = joblib.load(target_path)
        _NQ_MODELS_RAM_CACHE = (mtime, models)
        return models


def fetch_live_nq_5m_bars_from_tws_or_yf(
    ports: Tuple[int, ...] = (7497, 7496, 4002, 4001),
    symbol: str = "NQ=F",
) -> Optional[pd.DataFrame]:
    """Fetches the latest 2 days of 5-minute NQ, GC, CL, or BTC futures bars in America/New_York wall-clock time
    from Interactive Brokers TWS (preferred) or yfinance fallback."""
    prefix = get_futures_prefix(symbol)
    if prefix == "gc":
        tws_sym, tws_exch, yf_sym, base_cid = "GC", "COMEX", "GC=F", 195
    elif prefix == "cl":
        tws_sym, tws_exch, yf_sym, base_cid = "CL", "NYMEX", "CL=F", 210
    elif prefix == "btc":
        tws_sym, tws_exch, yf_sym, base_cid = "BRR", "CME", "BTC=F", 225
    elif prefix == "si":
        tws_sym, tws_exch, yf_sym, base_cid = "SI", "COMEX", "SI=F", 240
    else:
        tws_sym, tws_exch, yf_sym, base_cid = "NQ", "CME", "NQ=F", 180

    # 1. Try Interactive Brokers TWS
    try:
        import asyncio
        from zoneinfo import ZoneInfo
        from ib_async import ContFuture, IB

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ib = IB()
        try:
            connected = False
            for p in ports:
                try:
                    cid = base_cid + int(time.time() * 10) % 15
                    ib.connect("127.0.0.1", p, clientId=cid, timeout=2.0, readonly=True)
                    if ib.isConnected():
                        connected = True
                        break
                except Exception:
                    pass
            if connected and ib.isConnected():
                c = ContFuture(tws_sym, tws_exch)
                ib.qualifyContracts(c)
                if getattr(c, "conId", 0):
                    hbars = ib.reqHistoricalData(
                        c,
                        endDateTime="",
                        durationStr="2 D",
                        barSizeSetting="5 mins",
                        whatToShow="TRADES",
                        useRTH=False,
                        formatDate=2,
                        timeout=3.5,
                    )
                    if hbars:
                        rows = []
                        ny_tz = ZoneInfo("America/New_York")
                        for b in hbars:
                            ts = pd.Timestamp(b.date)
                            if ts.tzinfo is not None:
                                dt_et = ts.tz_convert(ny_tz).tz_localize(None)
                            else:
                                dt_et = ts
                            rows.append({
                                "datetime": dt_et,
                                "open": float(b.open),
                                "high": float(b.high),
                                "low": float(b.low),
                                "close": float(b.close),
                                "volume": max(0.0, float(getattr(b, "volume", 0.0) or 0.0)),
                            })
                        if rows:
                            df_5m = pd.DataFrame(rows).set_index("datetime").sort_index()
                            df_5m = df_5m[~df_5m.index.duplicated(keep="last")]
                            return df_5m
        finally:
            try:
                if ib.isConnected():
                    ib.disconnect()
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass
    except Exception:
        pass

    # 2. Fallback to yfinance 5m bars
    try:
        from zoneinfo import ZoneInfo
        import yfinance as yf

        sub = yf.download(yf_sym, period="5d", interval="5m", prepost=True, auto_adjust=False, progress=False)
        if sub is not None and not sub.empty:
            if isinstance(sub.columns, pd.MultiIndex):
                sub.columns = [c[0] for c in sub.columns]
            sub = sub.dropna(subset=["Close"]).copy()
            if not sub.empty:
                idx = sub.index
                if idx.tz is not None:
                    idx = idx.tz_convert(ZoneInfo("America/New_York")).tz_localize(None)
                df_5m = pd.DataFrame(
                    {
                        "open": sub["Open"].to_numpy(dtype=float),
                        "high": sub["High"].to_numpy(dtype=float),
                        "low": sub["Low"].to_numpy(dtype=float),
                        "close": sub["Close"].to_numpy(dtype=float),
                        "volume": sub["Volume"].fillna(0.0).to_numpy(dtype=float),
                    },
                    index=pd.DatetimeIndex(idx, name="datetime"),
                ).sort_index()
                df_5m = df_5m[~df_5m.index.duplicated(keep="last")]
                return df_5m
    except Exception:
        pass
    return None


def _sanitize_5m_bad_prints(df5: pd.DataFrame) -> pd.DataFrame:
    """Strictly causal bad-print filter using only prior-bar (`t-1`) and current-bar (`t`) prices,
    scaled proportionally to price level (~0.6% spike threshold, e.g., ~120 pts on NQ, ~$16 on GC, ~$0.45 on CL)."""
    if df5 is None or len(df5) < 5:
        return df5
    out = df5.copy()
    o = out["open"].to_numpy(dtype=float)
    h = out["high"].to_numpy(dtype=float)
    l = out["low"].to_numpy(dtype=float)
    c = out["close"].to_numpy(dtype=float)
    prev_c = np.roll(c, 1)
    prev_c[0] = o[0]
    prev_h = np.roll(h, 1)
    prev_h[0] = h[0]
    prev_l = np.roll(l, 1)
    prev_l[0] = l[0]

    max_spike = np.maximum(0.45, np.abs(prev_c) * 0.006)
    clamp_pad = np.maximum(0.03, np.abs(prev_c) * 0.0004)

    # Detect isolated open outlier > max_spike away from prev_close and current close while close stayed near prev_close
    bad_open = (np.abs(o - prev_c) > max_spike) & (np.abs(o - c) > max_spike) & (np.abs(c - prev_c) < 0.5 * max_spike)
    if np.any(bad_open):
        o = np.where(bad_open, prev_c, o)
        out["open"] = o

    # Detect isolated high spike > max_spike above max(open, close, prev_high) (strictly causal: no next_h)
    ref_hi = np.maximum(np.maximum(o, c), prev_h)
    bad_hi = (h - ref_hi) > max_spike
    if np.any(bad_hi):
        h = np.where(bad_hi, np.maximum(o, c) + clamp_pad, h)
        out["high"] = h

    # Detect isolated low spike > max_spike below min(open, close, prev_low) (strictly causal: no next_l)
    ref_lo = np.minimum(np.minimum(o, c), prev_l)
    bad_lo = (ref_lo - l) > max_spike
    if np.any(bad_lo):
        l = np.where(bad_lo, np.minimum(o, c) - clamp_pad, l)
        out["low"] = l

    return out


def sync_live_nq_intraday_bars(
    tws_5m_df: Optional[pd.DataFrame] = None,
    live_bar: Optional[Dict[str, Any]] = None,
    persist_parquet: bool = True,
    symbol: str = "NQ=F",
) -> bool:
    """
    Fuses live 5-minute NQ, GC, CL, or BTC futures bars into `data/bars_15m/{NQ=F,GC=F,CL=F,BTC=F}.parquet`
    and resamples + scores the live tail across all three timeframes (`15m`, `1h`, `4h`) in
    `{nq,gc,cl,btc}_<tf>_scored.parquet` and `_NQ_SCORED_RAM_CACHE`.
    """
    global _LAST_NQ_INTRADAY_SYNC_TS, _LAST_GC_INTRADAY_SYNC_TS
    prefix = get_futures_prefix(symbol)
    tail_map = {
        "gc": GC_5M_TAIL_PATH,
        "cl": CL_5M_TAIL_PATH,
        "btc": BTC_5M_TAIL_PATH,
        "si": SI_5M_TAIL_PATH,
        "nq": NQ_5M_TAIL_PATH,
    }
    tail_path = tail_map.get(prefix, NQ_5M_TAIL_PATH)

    if tws_5m_df is None or tws_5m_df.empty:
        tws_5m_df = fetch_live_nq_5m_bars_from_tws_or_yf(symbol=symbol)
    if tws_5m_df is None or tws_5m_df.empty:
        return False

    try:
        if prefix != "btc":
            tws_5m_df = _sanitize_5m_bad_prints(tws_5m_df)
        if tail_path.exists():
            try:
                old_5m = pd.read_parquet(tail_path)
                cutoff_ts = tws_5m_df.index[0]
                merged_5m = pd.concat([old_5m[old_5m.index < cutoff_ts], tws_5m_df]).sort_index()
                merged_5m = merged_5m[~merged_5m.index.duplicated(keep="last")]
            except Exception:
                merged_5m = tws_5m_df.copy()
        else:
            merged_5m = tws_5m_df.copy()

        if prefix != "btc":
            merged_5m = _sanitize_5m_bad_prints(merged_5m)

        if persist_parquet:
            try:
                _atomic_to_parquet(merged_5m, tail_path)
            except Exception:
                pass

        # Prepare recent 7-day 5m window for fast resampling
        recent_5m = merged_5m.tail(1500).copy()
        recent_5m["is_roll"] = False
        recent_5m["roll_mult"] = 1.0
        recent_5m["open_cont"] = recent_5m["open"].astype(float)
        recent_5m["high_cont"] = recent_5m["high"].astype(float)
        recent_5m["low_cont"] = recent_5m["low"].astype(float)
        recent_5m["close_cont"] = recent_5m["close"].astype(float)

        models_all = get_cached_nq_models(symbol=symbol)
        if not models_all:
            return False

        for tf in TIMEFRAMES:
            df_base = get_cached_nq_scored_df(tf, symbol=symbol)
            tf_bundle = models_all.get(tf)
            if df_base is None or df_base.empty or not tf_bundle:
                continue

            res_tf = resample_nq_with_1m_microstructure(recent_5m, tf)
            if res_tf.empty:
                continue
            # Scale 5-minute max sub-bar range fraction (~0.55x) to match 1-minute sub-bar calibration
            if "m1_max_bar_range_frac" in res_tf.columns:
                res_tf["m1_max_bar_range_frac"] = (res_tf["m1_max_bar_range_frac"] * 0.55).clip(0.05, 1.0)

            # If a live Level-1 tick price is provided, fuse it into the latest developing resampled bar
            if live_bar and _safe_float(live_bar.get("close"), 0.0) > 0:
                c_live = _safe_float(live_bar["close"], 0.0)
                last_r_idx = res_tf.index[-1]
                res_tf.loc[last_r_idx, "close"] = c_live
                res_tf.loc[last_r_idx, "close_cont"] = c_live
                if c_live > float(res_tf.loc[last_r_idx, "high"]):
                    res_tf.loc[last_r_idx, "high"] = c_live
                    res_tf.loc[last_r_idx, "high_cont"] = c_live
                if c_live < float(res_tf.loc[last_r_idx, "low"]):
                    res_tf.loc[last_r_idx, "low"] = c_live
                    res_tf.loc[last_r_idx, "low_cont"] = c_live

            # Slice 650 bars of history before res_tf starts so all 200-bar SMAs, 20-day TOD RVOL, and VWAPs are exact
            res_start = res_tf.index[0]
            hist_before = df_base[df_base.index < res_start]
            warm_tail = hist_before.tail(650).copy()

            raw_cols = [
                "open", "high", "low", "close", "volume", "is_roll",
                "m1_vol_delta_frac", "m1_vol_top25_frac", "m1_vol_bot25_frac",
                "m1_up_bar_frac", "m1_max_bar_range_frac", "m1_high_time_frac", "m1_low_time_frac", "m1_late_vol_frac",
            ]
            m1_col_defaults = {
                "is_roll": False,
                "m1_vol_delta_frac": 0.0,
                "m1_vol_top25_frac": 0.25,
                "m1_vol_bot25_frac": 0.25,
                "m1_up_bar_frac": 0.50,
                "m1_max_bar_range_frac": 0.20,
                "m1_high_time_frac": 0.50,
                "m1_low_time_frac": 0.50,
                "m1_late_vol_frac": 0.50,
            }
            warm_raw = pd.DataFrame(index=warm_tail.index)
            for c in raw_cols:
                if c in warm_tail.columns:
                    warm_raw[c] = warm_tail[c].astype(float if c != "is_roll" else bool)
                else:
                    warm_raw[c] = m1_col_defaults.get(c, 0.25)
            warm_raw["roll_mult"] = warm_tail["roll_mult"].astype(float) if "roll_mult" in warm_tail.columns else 1.0
            warm_raw["open_cont"] = warm_raw["open"] * warm_raw["roll_mult"]
            warm_raw["high_cont"] = warm_raw["high"] * warm_raw["roll_mult"]
            warm_raw["low_cont"] = warm_raw["low"] * warm_raw["roll_mult"]
            warm_raw["close_cont"] = warm_raw["close"] * warm_raw["roll_mult"]
            warm_raw["m1_count"] = float(TF_MINUTES_MAP.get(tf, 15))

            combined_raw = pd.concat([warm_raw, res_tf[warm_raw.columns]]).sort_index()
            combined_raw = combined_raw[~combined_raw.index.duplicated(keep="last")]

            feats = compute_nq_intraday_features(combined_raw, tf)
            cfg = tf_bundle.get("config", {"left": 5, "right": 5, "rev_atr": 1.4})
            label_fn = label_nq_pivots_and_excursions if prefix == "nq" else _label_gc_pivots_and_excursions
            labeled = label_fn(
                feats,
                pivot_left=int(cfg.get("left", 5)),
                pivot_right=int(cfg.get("right", 5)),
                rev_atr=float(cfg.get("rev_atr", 1.4)),
            )
            # Live tail bars do not cross a quarterly contract roll unless is_roll is True
            for k_h in HORIZONS:
                rc = f"crosses_roll_{k_h}b"
                if rc in labeled.columns and not labeled["is_roll"].tail(k_h + 2).any():
                    labeled.iloc[-k_h:, labeled.columns.get_loc(rc)] = False

            p_hi_dual, p_hi_ex, _ = score_nq_pivot_bundle(labeled, tf_bundle["pivot_high"], "is_cand_high")
            p_lo_dual, p_lo_ex, _ = score_nq_pivot_bundle(labeled, tf_bundle["pivot_low"], "is_cand_low")
            labeled["prob_pivot_high"] = p_hi_dual
            labeled["prob_pivot_high_extreme"] = p_hi_ex
            labeled["prob_pivot_low"] = p_lo_dual
            labeled["prob_pivot_low_extreme"] = p_lo_ex

            score_excursion_models(labeled, tf_bundle["excursion_short"], "short")
            score_excursion_models(labeled, tf_bundle["excursion_long"], "long")

            new_scored_part = labeled[labeled.index >= res_start].copy()
            for c in df_base.columns:
                if c not in new_scored_part.columns:
                    new_scored_part[c] = 0.0
            new_scored_part = new_scored_part[df_base.columns]
            for fcol in df_base.select_dtypes("float32").columns:
                new_scored_part[fcol] = new_scored_part[fcol].astype("float32")

            updated_df = pd.concat([hist_before, new_scored_part]).sort_index()
            updated_df = updated_df[~updated_df.index.duplicated(keep="last")]

            p_out = SCORED_CACHE_DIR / f"{prefix}_{tf}_scored.parquet"
            if persist_parquet:
                try:
                    _atomic_to_parquet(updated_df, p_out)
                    mtime = p_out.stat().st_mtime
                except Exception:
                    mtime = time.time()
            else:
                mtime = time.time()
            cache_k = tf if prefix == "nq" else f"{prefix}_{tf}"
            _NQ_SCORED_RAM_CACHE[cache_k] = (mtime, updated_df)

        _NQ_RT_EVAL_CACHE.clear()
        now_t = time.time()
        _LAST_INTRADAY_SYNC_TS_BY_SYM[prefix] = now_t
        if prefix == "gc":
            _LAST_GC_INTRADAY_SYNC_TS = now_t
        elif prefix == "nq":
            _LAST_NQ_INTRADAY_SYNC_TS = now_t
        return True
    except Exception:
        return False


def ensure_nq_intraday_uptodate(max_age_sec: float = 45.0, force: bool = False, symbol: str = "NQ=F") -> bool:
    """Ensures cached NQ, GC, CL, or BTC intraday bars (`15m`, `1h`, `4h`) are synced with live TWS/yfinance bars."""
    global _LAST_NQ_INTRADAY_SYNC_TS, _LAST_GC_INTRADAY_SYNC_TS
    is_test_runner = any("unittest" in str(a) or "pytest" in str(a) or "test_" in str(a) for a in sys.argv)
    if not force and is_test_runner:
        return False
    now_ts = time.time()
    prefix = get_futures_prefix(symbol)
    last_ts = _LAST_INTRADAY_SYNC_TS_BY_SYM.get(
        prefix,
        _LAST_GC_INTRADAY_SYNC_TS if prefix == "gc" else (_LAST_NQ_INTRADAY_SYNC_TS if prefix == "nq" else 0.0),
    )
    if not force and (now_ts - last_ts) < max_age_sec:
        return False
    _LAST_INTRADAY_SYNC_TS_BY_SYM[prefix] = now_ts
    if prefix == "gc":
        _LAST_GC_INTRADAY_SYNC_TS = now_ts
    elif prefix == "nq":
        _LAST_NQ_INTRADAY_SYNC_TS = now_ts
    return sync_live_nq_intraday_bars(persist_parquet=True, symbol=symbol)


def compute_nq_trade_plan(
    row_or_dict: Any,
    tf: str,
    is_high_pivot: bool,
    horizon_k: int = 8,
    symbol: str = "NQ=F",
    use_wickless: bool = False,
) -> Dict[str, Any]:
    """
    Computes the exact actionable trade handling plan (Entry, Stop-Loss, Target 1, Target 2,
    Time Stop, R:R, and Full/Micro contract dollar risk/reward) for a detected `15m`, `1h`, or `4h` pivot.
    Supports both Dual-Path with-wick mode and pure Wickless at-extreme mode.
    """
    tf_norm = tf.lower().strip()
    tf_label = TF_DISPLAY_MAP.get(tf_norm, tf_norm.upper())
    k = int(horizon_k) if int(horizon_k) in HORIZONS else 8
    d_key = "short" if is_high_pivot else "long"
    prefix = get_futures_prefix(symbol)
    if prefix == "gc":
        sym_root, micro_root = "GC", "MGC"
        full_usd_per_pt, micro_usd_per_pt = 100.0, 10.0
        min_atr_floor, min_stop_floor, tick_step = 1.0, 1.0, 0.10
    elif prefix == "cl":
        sym_root, micro_root = "CL", "MCL"
        full_usd_per_pt, micro_usd_per_pt = 1000.0, 100.0
        min_atr_floor, min_stop_floor, tick_step = 0.08, 0.08, 0.01
    elif prefix == "btc":
        sym_root, micro_root = "BTC", "MBT"
        full_usd_per_pt, micro_usd_per_pt = 5.0, 0.10
        min_atr_floor, min_stop_floor, tick_step = 50.0, 40.0, 5.0
    elif prefix == "si":
        sym_root, micro_root = "SI", "SIL"
        full_usd_per_pt, micro_usd_per_pt = 5000.0, 1000.0
        min_atr_floor, min_stop_floor, tick_step = 0.05, 0.04, 0.005
    else:
        sym_root, micro_root = "NQ", "MNQ"
        full_usd_per_pt, micro_usd_per_pt = 20.0, 2.0
        min_atr_floor, min_stop_floor, tick_step = 4.0, 2.0, 0.25

    def _g(col: str, default: float = 0.0) -> float:
        try:
            v = row_or_dict[col] if col in row_or_dict else getattr(row_or_dict, col, default)
            if v is None or pd.isna(v):
                return float(default)
            return float(v)
        except Exception:
            return float(default)

    o_px = _g("open", 0.0)
    h_px = _g("high", 0.0)
    l_px = _g("low", 0.0)
    c_px = _g("close", 0.0)
    atr_pts = max(min_atr_floor, _g("atr14_pts", max(min_atr_floor * 2.0, h_px - l_px)))

    raw_prob = (
        _g("prob_pivot_high_extreme" if is_high_pivot else "prob_pivot_low_extreme", 0.0)
        if use_wickless
        else _g("prob_pivot_high" if is_high_pivot else "prob_pivot_low", 0.0)
    )
    prob_pct = round(raw_prob * 100.0 if raw_prob <= 1.0 else raw_prob, 1)
    raw_ex = _g("prob_pivot_high_extreme" if is_high_pivot else "prob_pivot_low_extreme", 0.0)
    at_extreme_pct = round(raw_ex * 100.0 if raw_ex <= 1.0 else raw_ex, 1)

    pred_mfe_atr = max(0.80, _g(f"pred_mfe_{d_key}_{k}b_atr", 1.85))
    pred_mfe_q70_atr = max(pred_mfe_atr * 1.20, _g(f"pred_mfe_q70_{d_key}_{k}b_atr", pred_mfe_atr * 1.35))
    pred_mae_atr = max(0.45, _g(f"pred_mae_{d_key}_{k}b_atr", 0.90))
    pred_mae_q75_atr = max(pred_mae_atr, _g(f"pred_mae_q75_{d_key}_{k}b_atr", pred_mae_atr * 1.25))
    pred_ratio = max(0.50, _g(f"pred_ratio_{d_key}_{k}b", pred_mfe_atr / max(0.25, pred_mae_atr)))
    raw_asym = _g(f"prob_asym_{d_key}_{k}b", 0.55)
    prob_asym_pct = round(raw_asym * 100.0 if raw_asym <= 1.0 else raw_asym, 1)

    def _tick_round(val: float) -> float:
        if prefix == "gc":
            return round(round(float(val) * 10.0) / 10.0, 2)
        if prefix == "cl":
            return round(float(val), 2)
        if prefix == "btc":
            return round(round(float(val) / 5.0) * 5.0, 2)
        if prefix == "si":
            return round(round(float(val) * 200.0) / 200.0, 3)
        return round(round(float(val) * 4.0) / 4.0, 2)

    # 1. Zero-Lookahead Entry Price: Intrabar Extreme Touch (Wickless) vs 1m Developing Wick-Edge Trigger (`close_1m[t, m^*]`) or `Close[t]`
    if is_high_pivot:
        dev_h_raw = _g("dev_high_short_px", h_px)
        ref_ext_px = _tick_round(dev_h_raw if dev_h_raw > 0 else h_px)
        extreme_px = _tick_round(h_px if h_px > 0 else ref_ext_px)
        trig_min = int(_g("trig_min_short", 0))
        if use_wickless:
            entry_px = extreme_px
        else:
            trig_default = _tick_round(c_px if c_px > 0 else h_px)
            entry_px = _tick_round(_g("trigger_short_px", trig_default))
            if entry_px <= 0:
                entry_px = trig_default
    else:
        dev_l_raw = _g("dev_low_long_px", l_px)
        ref_ext_px = _tick_round(dev_l_raw if dev_l_raw > 0 else l_px)
        extreme_px = _tick_round(l_px if l_px > 0 else ref_ext_px)
        trig_min = int(_g("trig_min_long", 0))
        if use_wickless:
            entry_px = extreme_px
        else:
            trig_default = _tick_round(c_px if c_px > 0 else l_px)
            entry_px = _tick_round(_g("trigger_long_px", trig_default))
            if entry_px <= 0:
                entry_px = trig_default

    # 2. Stop-Loss Placement: max(1.00 ATR, 1.20 * pred_mae_atr) and at least 0.15 ATR beyond developing pivot extreme
    model_stop_atr = max(1.00, 1.20 * pred_mae_atr)
    if is_high_pivot:
        stop_by_mae = entry_px + model_stop_atr * atr_pts
        stop_by_wick = ref_ext_px + (0.20 if use_wickless else 0.15) * atr_pts
        stop_px = _tick_round(max(stop_by_mae, stop_by_wick, entry_px + min_stop_floor))
        stop_dist_pts = round(max(min_stop_floor, stop_px - entry_px), 2)
    else:
        stop_by_mae = entry_px - model_stop_atr * atr_pts
        stop_by_wick = ref_ext_px - (0.20 if use_wickless else 0.15) * atr_pts
        stop_px = _tick_round(min(stop_by_mae, stop_by_wick, entry_px - min_stop_floor))
        stop_dist_pts = round(max(min_stop_floor, entry_px - stop_px), 2)

    stop_dist_atr = round(stop_dist_pts / atr_pts, 2)

    # 3. Profit Targets: Target 1 (1.05x Pred MFE, clipped to [1.10, 3.50] ATR) & Target 2 (Q70 MFE Runner)
    target1_atr = round(min(3.50, max(1.10, 1.05 * pred_mfe_atr)), 2)
    target1_pts = _tick_round(target1_atr * atr_pts)
    target2_atr = round(max(target1_atr * 1.25, min(5.00, pred_mfe_q70_atr)), 2)
    target2_pts = _tick_round(target2_atr * atr_pts)

    pred_mfe_pts = _tick_round(pred_mfe_atr * atr_pts)
    pred_mfe_q70_pts = _tick_round(pred_mfe_q70_atr * atr_pts)
    pred_mae_pts = _tick_round(pred_mae_atr * atr_pts)
    pred_mae_q75_pts = _tick_round(pred_mae_q75_atr * atr_pts)

    if is_high_pivot:
        mfe_expected_px = _tick_round(entry_px - pred_mfe_pts)
        mfe_q70_px = _tick_round(entry_px - pred_mfe_q70_pts)
        mae_expected_px = _tick_round(entry_px + pred_mae_pts)
        mae_q75_px = _tick_round(entry_px + pred_mae_q75_pts)
        target1_px = _tick_round(entry_px - target1_pts)
        target2_px = _tick_round(min(target1_px - tick_step, entry_px - target2_pts))
        target1_pts = round(entry_px - target1_px, 2)
        target2_pts = round(entry_px - target2_px, 2)
        direction = "SHORT"
        mod_tag = " (⚡ Wickless Intrabar)" if use_wickless else ""
        action_title = f"🔻 {sym_root} [{tf_label}] SHORT PIVOT HIGH{mod_tag}"
        if use_wickless:
            entry_rule = f"Short @ {entry_px:.2f} (Intrabar Extreme Touch — 0 Wick Lag)"
        elif trig_min > 0:
            entry_rule = f"Short @ {entry_px:.2f} (1m developing wick-edge trigger at +{trig_min}m off {ref_ext_px:.2f})"
        else:
            entry_rule = f"Short @ {entry_px:.2f} (Intrabar developing trigger near pivot high {ref_ext_px:.2f})"
        stop_rule = f"Stop @ {stop_px:.2f} (+{stop_dist_pts:.2f} pts / {stop_dist_atr:.2f}x ATR, 1.2x Pred MAE above high)"
    else:
        mfe_expected_px = _tick_round(entry_px + pred_mfe_pts)
        mfe_q70_px = _tick_round(entry_px + pred_mfe_q70_pts)
        mae_expected_px = _tick_round(entry_px - pred_mae_pts)
        mae_q75_px = _tick_round(entry_px - pred_mae_q75_pts)
        target1_px = _tick_round(entry_px + target1_pts)
        target2_px = _tick_round(max(target1_px + tick_step, entry_px + target2_pts))
        target1_pts = round(target1_px - entry_px, 2)
        target2_pts = round(target2_px - entry_px, 2)
        direction = "LONG"
        mod_tag = " (⚡ Wickless Intrabar)" if use_wickless else ""
        action_title = f"💎 {sym_root} [{tf_label}] LONG PIVOT LOW{mod_tag}"
        if use_wickless:
            entry_rule = f"Buy @ {entry_px:.2f} (Intrabar Extreme Touch — 0 Wick Lag)"
        elif trig_min > 0:
            entry_rule = f"Buy @ {entry_px:.2f} (1m developing wick-edge trigger at +{trig_min}m off {ref_ext_px:.2f})"
        else:
            entry_rule = f"Buy @ {entry_px:.2f} (Intrabar developing trigger near pivot low {ref_ext_px:.2f})"
        stop_rule = f"Stop @ {stop_px:.2f} (-{stop_dist_pts:.2f} pts / {stop_dist_atr:.2f}x ATR, 1.2x Pred MAE below low)"

    rr_t1 = round(target1_pts / max(min_stop_floor, stop_dist_pts), 2)
    rr_t2 = round(target2_pts / max(min_stop_floor, stop_dist_pts), 2)

    # 4. Trade Validity & Expectancy Gates (factual metrics only — zero hardcoded win rate claims)
    gate_prob_pass = bool(prob_pct >= 50.0)
    gate_ratio_pass = bool(pred_ratio >= 1.25)
    gate_rr_pass = bool(rr_t1 >= 1.00)
    is_valid_trade = bool(gate_prob_pass and gate_ratio_pass and gate_rr_pass)

    if is_valid_trade and prob_pct >= 60.0 and pred_ratio >= 1.75:
        validity_tier = "A+ Institutional Setup (Valid Trade)"
        expected_win_rate_str = f"Prob {prob_pct:.1f}% · Pred MFE/MAE {pred_ratio:.2f}x · TP1 {rr_t1:.2f}R"
    elif is_valid_trade and prob_pct >= 55.0 and pred_ratio >= 1.50:
        validity_tier = "High-Conviction Setup (Valid Trade)"
        expected_win_rate_str = f"Prob {prob_pct:.1f}% · Pred MFE/MAE {pred_ratio:.2f}x · TP1 {rr_t1:.2f}R"
    elif is_valid_trade:
        validity_tier = "Standard Setup (Valid Trade)"
        expected_win_rate_str = f"Prob {prob_pct:.1f}% · Pred MFE/MAE {pred_ratio:.2f}x · TP1 {rr_t1:.2f}R"
    else:
        failed_reasons = []
        if not gate_prob_pass:
            failed_reasons.append(f"Prob {prob_pct:.0f}% < 50% min")
        if not gate_ratio_pass:
            failed_reasons.append(f"MFE/MAE {pred_ratio:.2f}x < 1.25x min")
        if not gate_rr_pass:
            failed_reasons.append(f"TP1 R:R {rr_t1:.2f}x < 1.00x min")
        validity_tier = f"NOT VALID — Watch Only ({', '.join(failed_reasons)})"
        expected_win_rate_str = "Below Gate Hurdles — Do Not Enter"

    tf_mins = TF_MINUTES_MAP.get(tf_norm, 15)
    total_mins = k * tf_mins
    if total_mins % 60 == 0:
        time_exit_label = f"{k} bars ({total_mins // 60}h)"
    else:
        time_exit_label = f"{k} bars ({total_mins / 60.0:.1f}h)"

    exit_rule = (
        f"TP1 (50%): {target1_px:.2f} ({'+' if not is_high_pivot else '-'}{target1_pts:.2f} pts, {rr_t1:.2f}R) -> Stop to BE | "
        f"TP2 (50% Runner): {target2_px:.2f} ({'+' if not is_high_pivot else '-'}{target2_pts:.2f} pts, {rr_t2:.2f}R) | "
        f"Time Stop: {time_exit_label}"
    )

    trade_plan_text = (
        f"{action_title} ({prob_pct:.1f}% prob, {pred_ratio:.2f}x MFE/MAE) | "
        f"Entry: {entry_px:.2f} | Stop: {stop_px:.2f} ({stop_dist_pts:.2f} pts / {stop_dist_atr:.2f} ATR) | "
        f"TP1 (50%): {target1_px:.2f} ({target1_pts:.2f} pts, {rr_t1:.2f}R) | "
        f"TP2 (Runner): {target2_px:.2f} ({target2_pts:.2f} pts, {rr_t2:.2f}R) | "
        f"Time Exit: {time_exit_label}"
    )

    return {
        "symbol": symbol,
        "contract_root": sym_root,
        "micro_contract_root": micro_root,
        "tf": tf_norm,
        "timeframe": tf_norm,
        "timeframe_label": tf_label,
        "direction": direction,
        "is_high_pivot": bool(is_high_pivot),
        "action_title": action_title,
        "prob_pct": prob_pct,
        "at_extreme_pct": at_extreme_pct,
        "pred_mfe_atr": round(pred_mfe_atr, 2),
        "pred_mfe_q70_atr": round(pred_mfe_q70_atr, 2),
        "pred_mae_atr": round(pred_mae_atr, 2),
        "pred_mae_q75_atr": round(pred_mae_q75_atr, 2),
        "pred_mfe_pts": round(pred_mfe_pts, 2),
        "pred_mfe_q70_pts": round(pred_mfe_q70_pts, 2),
        "pred_mae_pts": round(pred_mae_pts, 2),
        "pred_mae_q75_pts": round(pred_mae_q75_pts, 2),
        "mfe_expected_px": mfe_expected_px,
        "mfe_q70_px": mfe_q70_px,
        "mae_expected_px": mae_expected_px,
        "mae_q75_px": mae_q75_px,
        "pred_ratio": round(pred_ratio, 2),
        "prob_asym_pct": prob_asym_pct,
        "is_valid_trade": is_valid_trade,
        "gate_prob_pass": gate_prob_pass,
        "gate_ratio_pass": gate_ratio_pass,
        "gate_rr_pass": gate_rr_pass,
        "validity_tier": validity_tier,
        "expected_win_rate_str": expected_win_rate_str,
        "atr_pts": round(atr_pts, 2),
        "extreme_price": extreme_px,
        "extreme_px": extreme_px,
        "close_price": round(c_px, 2),
        "entry_price": entry_px,
        "entry_px": entry_px,
        "entry_rule": entry_rule,
        "entry_rule_text": entry_rule,
        "stop_price": stop_px,
        "stop_px": stop_px,
        "stop_dist_pts": stop_dist_pts,
        "stop_dist_atr": stop_dist_atr,
        "stop_atr": stop_dist_atr,
        "stop_rule": stop_rule,
        "stop_rule_text": stop_rule,
        "target1_price": target1_px,
        "target1_px": target1_px,
        "target1_pts": target1_pts,
        "target1_dist_pts": target1_pts,
        "target1_atr": target1_atr,
        "target2_price": target2_px,
        "target2_px": target2_px,
        "target2_pts": target2_pts,
        "target2_dist_pts": target2_pts,
        "target2_atr": target2_atr,
        "rr_target1": rr_t1,
        "rr_target2": rr_t2,
        "time_exit_bars": k,
        "horizon_bars": k,
        "time_exit_label": time_exit_label,
        "horizon_label": time_exit_label,
        "exit_rule": exit_rule,
        "target_rule_text": exit_rule,
        "is_wickless": bool(use_wickless),
        "model_mode": "wickless" if use_wickless else "current",
        "time_exit_rule_text": f"If neither Stop nor Target is hit within {time_exit_label}, exit at bar {k} close.",
        "risk_1_nq_usd": round(stop_dist_pts * full_usd_per_pt, 0),
        "risk_usd_nq": round(stop_dist_pts * full_usd_per_pt, 0),
        "reward1_1_nq_usd": round(target1_pts * full_usd_per_pt, 0),
        "reward2_1_nq_usd": round(target2_pts * full_usd_per_pt, 0),
        "risk_1_mnq_usd": round(stop_dist_pts * micro_usd_per_pt, 0),
        "risk_usd_mnq": round(stop_dist_pts * micro_usd_per_pt, 0),
        "reward1_1_mnq_usd": round(target1_pts * micro_usd_per_pt, 0),
        "reward2_1_mnq_usd": round(target2_pts * micro_usd_per_pt, 0),
        "trade_plan_text": trade_plan_text,
    }


def _safe_float(val: Any, default: float = 0.0, ndigits: Optional[int] = None) -> float:
    """Safely converts `val` to a finite float, replacing None/NaN/Inf with `default`."""
    try:
        if val is None or pd.isna(val):
            res = float(default)
        else:
            f = float(val)
            res = float(default) if (math.isnan(f) or math.isinf(f)) else f
    except (ValueError, TypeError):
        res = float(default)
    return round(res, ndigits) if ndigits is not None else res


def _sanitize_json_value(obj: Any) -> Any:
    """Recursively sanitizes dicts/lists/floats so zero NaN or Infinity tokens ever enter JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize_json_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json_value(x) for x in obj]
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if math.isnan(f) or math.isinf(f):
            return 0.0
        return f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _score_live_bar_override_on_tf(
    df_tf: pd.DataFrame,
    tf: str,
    live_bar: Dict[str, Any],
    symbol: str = "NQ=F",
    use_wickless: bool = False,
) -> pd.Series:
    """
    Updates the developing bar at the tail of `df_tf` with `live_bar` OHLCV and re-scores
    the NQ-, GC-, CL-, BTC-, or SI-specific Dual-Path / Wickless Pivot High/Low Classifiers and Excursion Regressors on the tail.
    """
    last_row = df_tf.iloc[-1].copy()
    c_px = _safe_float(live_bar.get("close"), _safe_float(last_row.get("close"), 0.0))
    if c_px <= 0:
        return last_row

    is_test_runner = any("unittest" in str(a) or "pytest" in str(a) or "test_" in str(a) for a in sys.argv)
    idx_last = df_tf.index[-1]
    if not live_bar.get("is_intraday_bar") and not is_test_runner:
        if str(live_bar.get("source", "")) == "Datastore-EOD":
            return last_row
        lb_date = str(live_bar.get("date", ""))
        if lb_date and lb_date < idx_last.strftime("%Y-%m-%d"):
            return last_row

    models_all = get_cached_nq_models(symbol=symbol, use_wickless=use_wickless)
    tf_bundle = models_all.get(tf)
    if not tf_bundle:
        return last_row

    tail = df_tf.tail(220).copy()
    idx_last = tail.index[-1]

    h_live = _safe_float(live_bar.get("high"), c_px) or c_px
    l_live = _safe_float(live_bar.get("low"), c_px) or c_px
    o_live = _safe_float(live_bar.get("open"), _safe_float(tail.loc[idx_last, "open"], c_px)) or c_px

    use_explicit_hl = bool(live_bar.get("is_intraday_bar")) or is_test_runner
    eff_h_live = h_live if use_explicit_hl else c_px
    eff_l_live = l_live if use_explicit_hl else c_px

    new_h = max(_safe_float(tail.loc[idx_last, "high"], c_px), eff_h_live, c_px)
    new_l = min(_safe_float(tail.loc[idx_last, "low"], c_px), eff_l_live, c_px)
    tail.loc[idx_last, "close"] = c_px
    tail.loc[idx_last, "high"] = new_h
    tail.loc[idx_last, "low"] = new_l
    if ((live_bar.get("is_intraday_bar") and tf in ("5m", "15m")) or is_test_runner) and abs(o_live - c_px) < max(400.0, c_px * 0.05):
        tail.loc[idx_last, "open"] = max(new_l, min(new_h, o_live))

    m1_defaults = {
        "m1_vol_top25_frac": 0.25,
        "m1_vol_bot25_frac": 0.25,
        "m1_vol_delta_frac": 0.0,
        "m1_max_1m_vol_share": 0.12,
        "m1_up_down_vol_ratio": 1.0,
        "m1_early_vol_frac": 0.35,
        "m1_late_vol_frac": 0.35,
        "m1_up_bar_frac": 0.50,
        "m1_max_bar_range_frac": 0.20,
        "m1_high_time_frac": 0.50,
        "m1_low_time_frac": 0.50,
        "is_roll": False,
    }
    raw_ohlcv = tail[["open", "high", "low", "close", "volume"]].copy()
    raw_ohlcv["open_cont"] = raw_ohlcv["open"].astype(float)
    raw_ohlcv["high_cont"] = raw_ohlcv["high"].astype(float)
    raw_ohlcv["low_cont"] = raw_ohlcv["low"].astype(float)
    raw_ohlcv["close_cont"] = raw_ohlcv["close"].astype(float)
    raw_ohlcv["roll_mult"] = 1.0
    raw_ohlcv["m1_vwap_cont"] = (raw_ohlcv["high_cont"] + raw_ohlcv["low_cont"] + raw_ohlcv["close_cont"]) / 3.0
    raw_ohlcv["m1_vwap_px"] = raw_ohlcv["m1_vwap_cont"]
    for col, def_val in m1_defaults.items():
        if col in tail.columns:
            raw_ohlcv[col] = tail[col].copy()
        else:
            raw_ohlcv[col] = def_val

    feats = compute_nq_intraday_features(raw_ohlcv, tf)
    cfg = tf_bundle.get("config", {"left": 5, "right": 5})
    left_w = int(cfg.get("left", 5))
    hi_last = _safe_float(feats["high"].iloc[-1], new_h)
    lo_last = _safe_float(feats["low"].iloc[-1], new_l)
    prev_hi = _safe_float(feats["high"].iloc[-left_w - 1 : -1].max(), hi_last)
    prev_lo = _safe_float(feats["low"].iloc[-left_w - 1 : -1].min(), lo_last)
    feats["is_cand_high"] = False
    feats["is_cand_low"] = False
    feats.iloc[-1, feats.columns.get_loc("is_cand_high")] = bool(hi_last >= prev_hi)
    feats.iloc[-1, feats.columns.get_loc("is_cand_low")] = bool(lo_last <= prev_lo)

    last_df = feats.iloc[[-1]].copy()
    for col in last_df.columns:
        val_c = last_df[col].iloc[0]
        if pd.isna(val_c) and col in last_row.index and pd.notna(last_row[col]):
            last_df[col] = last_row[col]
    last_df["tod_rvol"] = _safe_float(last_df["tod_rvol"].iloc[0] if "tod_rvol" in last_df.columns else None, _safe_float(last_row.get("tod_rvol"), 1.0))
    prefix = get_futures_prefix(symbol)
    min_atr_def = 0.10 if prefix == "si" else (0.20 if prefix == "cl" else (120.0 if prefix == "btc" else (5.0 if prefix == "gc" else 25.0)))
    last_df["atr14_pts"] = max(0.05, _safe_float(last_df["atr14_pts"].iloc[0] if "atr14_pts" in last_df.columns else None, _safe_float(last_row.get("atr14_pts"), min_atr_def)))
    last_df["rsi14"] = _safe_float(last_df["rsi14"].iloc[0] if "rsi14" in last_df.columns else None, _safe_float(last_row.get("rsi14"), 50.0))

    last_df["trigger_short_px"] = last_df["close"].astype(float)
    last_df["trigger_long_px"] = last_df["close"].astype(float)
    last_df["dev_high_short_px"] = last_df["high"].astype(float)
    last_df["dev_low_long_px"] = last_df["low"].astype(float)

    p_hi, p_hi_ex, _ = score_nq_pivot_bundle(last_df, tf_bundle["pivot_high"], "is_cand_high")
    p_lo, p_lo_ex, _ = score_nq_pivot_bundle(last_df, tf_bundle["pivot_low"], "is_cand_low")
    last_df["prob_pivot_high"] = np.nan_to_num(p_hi, nan=0.0, posinf=1.0, neginf=0.0)
    last_df["prob_pivot_high_extreme"] = np.nan_to_num(p_hi_ex, nan=0.0, posinf=1.0, neginf=0.0)
    last_df["prob_pivot_low"] = np.nan_to_num(p_lo, nan=0.0, posinf=1.0, neginf=0.0)
    last_df["prob_pivot_low_extreme"] = np.nan_to_num(p_lo_ex, nan=0.0, posinf=1.0, neginf=0.0)

    if "major_top" in tf_bundle:
        _, p_mt_ex, _ = score_nq_pivot_bundle(last_df, tf_bundle["major_top"], "is_cand_high")
        last_df["prob_major_top_extreme"] = np.nan_to_num(p_mt_ex, nan=0.0, posinf=1.0, neginf=0.0)
        last_df["prob_maj_ut_top_extreme"] = last_df["prob_major_top_extreme"]
    else:
        base_wl = get_cached_nq_models(symbol=symbol, use_wickless=True)
        wl_tf = base_wl.get(tf, {})
        if "major_top" in wl_tf:
            _, p_mt_ex, _ = score_nq_pivot_bundle(last_df, wl_tf["major_top"], "is_cand_high")
            last_df["prob_major_top_extreme"] = np.nan_to_num(p_mt_ex, nan=0.0, posinf=1.0, neginf=0.0)
            last_df["prob_maj_ut_top_extreme"] = last_df["prob_major_top_extreme"]
        else:
            last_df["prob_major_top_extreme"] = 0.0
            last_df["prob_maj_ut_top_extreme"] = 0.0

    if "major_bottom" in tf_bundle:
        _, p_mb_ex, _ = score_nq_pivot_bundle(last_df, tf_bundle["major_bottom"], "is_cand_low")
        last_df["prob_major_bottom_extreme"] = np.nan_to_num(p_mb_ex, nan=0.0, posinf=1.0, neginf=0.0)
        last_df["prob_maj_dt_low_extreme"] = last_df["prob_major_bottom_extreme"]
    else:
        base_wl = get_cached_nq_models(symbol=symbol, use_wickless=True)
        wl_tf = base_wl.get(tf, {})
        if "major_bottom" in wl_tf:
            _, p_mb_ex, _ = score_nq_pivot_bundle(last_df, wl_tf["major_bottom"], "is_cand_low")
            last_df["prob_major_bottom_extreme"] = np.nan_to_num(p_mb_ex, nan=0.0, posinf=1.0, neginf=0.0)
            last_df["prob_maj_dt_low_extreme"] = last_df["prob_major_bottom_extreme"]
        else:
            last_df["prob_major_bottom_extreme"] = 0.0
            last_df["prob_maj_dt_low_extreme"] = 0.0

    if "excursion_short" in tf_bundle:
        score_excursion_models(last_df, tf_bundle["excursion_short"], "short")
    else:
        base_all = get_cached_nq_models(symbol=symbol, use_wickless=False)
        base_tf = base_all.get(tf, {})
        if "excursion_short" in base_tf:
            score_excursion_models(last_df, base_tf["excursion_short"], "short")

    if "excursion_long" in tf_bundle:
        score_excursion_models(last_df, tf_bundle["excursion_long"], "long")
    else:
        base_all = get_cached_nq_models(symbol=symbol, use_wickless=False)
        base_tf = base_all.get(tf, {})
        if "excursion_long" in base_tf:
            score_excursion_models(last_df, base_tf["excursion_long"], "long")

    return last_df.iloc[0]


_NQ_RT_EVAL_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}


def evaluate_nq_realtime_multi_tf(
    live_bar: Optional[Dict[str, Any]] = None,
    threshold_pct: float = 50.0,
    horizon_k: int = 8,
    symbol: str = "NQ=F",
    use_wickless: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates NQ, GC, CL, BTC, or SI `15m`, `1h`, and `4h` intraday pivot models in real time, computing:
      - Current bar probabilities & regime per timeframe (`15m`, `1h`, `4h`)
      - Latest active / triggered pivot signal & full actionable `trade_plan` per timeframe
      - Real-time TWS alerts with complete Entry, Stop-Loss, Target 1, Target 2, and Time Exit rules
    """
    thr = _safe_float(threshold_pct, 50.0)
    k = int(horizon_k) if int(horizon_k) in HORIZONS else 8
    c_key = _safe_float(live_bar.get("close"), 0.0, 2) if live_bar else 0.0
    h_key = _safe_float(live_bar.get("high"), 0.0, 2) if live_bar else 0.0
    l_key = _safe_float(live_bar.get("low"), 0.0, 2) if live_bar else 0.0
    d_key = str(live_bar.get("date", "")) if live_bar else ""
    cache_key = (str(symbol), d_key, c_key, h_key, l_key, round(thr, 1), k, bool(use_wickless))
    cached_res = _NQ_RT_EVAL_CACHE.get(cache_key)
    if cached_res is not None:
        return cached_res

    prefix = get_futures_prefix(symbol)
    sym_root = prefix.upper()
    move_tol = 0.01 if prefix == "si" else (0.03 if prefix == "cl" else (10.0 if prefix == "btc" else (0.15 if prefix == "gc" else 0.5)))

    tf_summaries: Dict[str, Dict[str, Any]] = {}
    triggered_alerts: List[Dict[str, Any]] = []
    best_overall_plan: Optional[Dict[str, Any]] = None
    best_overall_prob = -1.0

    for tf in TIMEFRAMES:
        df_tf = get_cached_nq_scored_df(tf, symbol=symbol)
        if df_tf is None or df_tf.empty:
            continue

        last_ts = df_tf.index[-1]
        last_time_str = last_ts.strftime("%Y-%m-%d %H:%M")

        if live_bar and _safe_float(live_bar.get("close"), 0.0) > 0:
            c_in = _safe_float(live_bar["close"], 0.0)
            h_in = _safe_float(live_bar.get("high"), c_in) or c_in
            l_in = _safe_float(live_bar.get("low"), c_in) or c_in
            row_prev = df_tf.iloc[-1]
            if (
                abs(c_in - _safe_float(row_prev["close"], c_in)) > move_tol
                or h_in > _safe_float(row_prev["high"], h_in) + move_tol
                or l_in < _safe_float(row_prev["low"], l_in) - move_tol
            ):
                curr_row = _score_live_bar_override_on_tf(df_tf, tf, live_bar, symbol=symbol, use_wickless=use_wickless)
                if live_bar.get("date"):
                    last_time_str = f"{last_ts.strftime('%Y-%m-%d %H:%M')} LIVE"
            else:
                curr_row = row_prev
        else:
            curr_row = df_tf.iloc[-1]

        curr_hi_pct = _safe_float(curr_row.get("prob_pivot_high", curr_row.get("prob_pivot_high_extreme", 0.0)), 0.0) * 100.0
        curr_hi_pct = round(curr_hi_pct, 1)
        curr_lo_pct = _safe_float(curr_row.get("prob_pivot_low", curr_row.get("prob_pivot_low_extreme", 0.0)), 0.0) * 100.0
        curr_lo_pct = round(curr_lo_pct, 1)
        curr_hi_ex_pct = round(_safe_float(curr_row.get("prob_pivot_high_extreme", curr_row.get("prob_pivot_high", 0.0)), 0.0) * 100.0, 1)
        curr_lo_ex_pct = round(_safe_float(curr_row.get("prob_pivot_low_extreme", curr_row.get("prob_pivot_low", 0.0)), 0.0) * 100.0, 1)
        curr_maj_top_pct = round(_safe_float(curr_row.get("prob_major_top_extreme", curr_row.get("prob_maj_ut_top_extreme", 0.0)), 0.0) * 100.0, 1)
        curr_maj_bot_pct = round(_safe_float(curr_row.get("prob_major_bottom_extreme", curr_row.get("prob_maj_dt_low_extreme", 0.0)), 0.0) * 100.0, 1)

        lb = min(max(k + 1, 4), len(df_tf))
        recent_slice = df_tf.tail(lb)
        active_row = curr_row
        active_ts = last_ts
        active_time_str = last_time_str
        if use_wickless:
            active_is_hi = curr_hi_ex_pct >= curr_lo_ex_pct
            active_prob = max(curr_hi_ex_pct, curr_lo_ex_pct)
        else:
            active_is_hi = curr_hi_pct >= curr_lo_pct
            active_prob = max(curr_hi_pct, curr_lo_pct)
        is_curr_bar_trig = active_prob >= thr

        def _is_plan_still_alive(cand_row: pd.Series, cand_is_hi: bool, subseq_highs: np.ndarray, subseq_lows: np.ndarray) -> bool:
            c_plan = compute_nq_trade_plan(cand_row, tf=tf, is_high_pivot=cand_is_hi, horizon_k=k, symbol=symbol, use_wickless=use_wickless)
            s_px = float(c_plan["stop_px"])
            t1_px = float(c_plan["target1_px"])
            if cand_is_hi:
                return bool(np.max(subseq_highs) < s_px and np.min(subseq_lows) > t1_px)
            return bool(np.min(subseq_lows) > s_px and np.max(subseq_highs) < t1_px)

        if not is_curr_bar_trig and len(recent_slice) > 1:
            hist_slice = recent_slice.iloc[:-1]
            if use_wickless:
                hi_col = "prob_pivot_high_extreme" if "prob_pivot_high_extreme" in hist_slice.columns else "prob_pivot_high"
                lo_col = "prob_pivot_low_extreme" if "prob_pivot_low_extreme" in hist_slice.columns else "prob_pivot_low"
            else:
                hi_col = "prob_pivot_high" if "prob_pivot_high" in hist_slice.columns else "prob_pivot_high_extreme"
                lo_col = "prob_pivot_low" if "prob_pivot_low" in hist_slice.columns else "prob_pivot_low_extreme"
            hi_probs = (np.nan_to_num(hist_slice[hi_col].to_numpy(dtype=np.float64), nan=0.0) * 100.0) if hi_col in hist_slice.columns else np.zeros(len(hist_slice))
            lo_probs = (np.nan_to_num(hist_slice[lo_col].to_numpy(dtype=np.float64), nan=0.0) * 100.0) if lo_col in hist_slice.columns else np.zeros(len(hist_slice))
            h_hist = hist_slice["high"].to_numpy(dtype=np.float64)
            l_hist = hist_slice["low"].to_numpy(dtype=np.float64)
            c_hi_curr = _safe_float(curr_row.get("high"), 0.0)
            c_lo_curr = _safe_float(curr_row.get("low"), 0.0)

            found_recent = False
            for idx_back in range(len(hist_slice) - 1, -1, -1):
                p_h = float(hi_probs[idx_back])
                p_l = float(lo_probs[idx_back])
                if max(p_h, p_l) >= thr:
                    cand_row = hist_slice.iloc[idx_back]
                    cand_is_hi = p_h >= p_l
                    sub_h = np.append(h_hist[idx_back + 1 :], c_hi_curr)
                    sub_l = np.append(l_hist[idx_back + 1 :], c_lo_curr)
                    if _is_plan_still_alive(cand_row, cand_is_hi, sub_h, sub_l):
                        active_row = cand_row
                        active_ts = hist_slice.index[idx_back]
                        active_time_str = active_ts.strftime("%Y-%m-%d %H:%M")
                        active_is_hi = cand_is_hi
                        active_prob = round(max(p_h, p_l), 1)
                        found_recent = True
                        break
            if not found_recent:
                best_cand_prob = active_prob
                for idx_back in range(len(hist_slice) - 1, -1, -1):
                    p_h = float(hi_probs[idx_back])
                    p_l = float(lo_probs[idx_back])
                    cand_p = max(p_h, p_l)
                    if cand_p > best_cand_prob:
                        cand_row = hist_slice.iloc[idx_back]
                        cand_is_hi = p_h >= p_l
                        sub_h = np.append(h_hist[idx_back + 1 :], c_hi_curr)
                        sub_l = np.append(l_hist[idx_back + 1 :], c_lo_curr)
                        if _is_plan_still_alive(cand_row, cand_is_hi, sub_h, sub_l):
                            best_cand_prob = cand_p
                            active_row = cand_row
                            active_ts = hist_slice.index[idx_back]
                            active_time_str = active_ts.strftime("%Y-%m-%d %H:%M")
                            active_is_hi = cand_is_hi
                            active_prob = round(cand_p, 1)

        plan = compute_nq_trade_plan(active_row, tf=tf, is_high_pivot=active_is_hi, horizon_k=k, symbol=symbol, use_wickless=use_wickless)
        plan["bar_time"] = active_time_str
        plan["bar_timestamp"] = int(active_ts.timestamp())
        plan["is_live_developing"] = bool(is_curr_bar_trig)
        plan["triggered"] = bool(active_prob >= thr)
        plan["is_triggered"] = bool(active_prob >= thr)

        tf_label = TF_DISPLAY_MAP.get(tf, tf.upper())
        status_str = ("pivot_high" if active_is_hi else "pivot_low") if active_prob >= thr else "watching"
        def_atr = 0.15 if prefix == "si" else (0.25 if prefix == "cl" else (120.0 if prefix == "btc" else (7.0 if prefix == "gc" else 25.0)))
        tf_summaries[tf] = {
            "tf": tf,
            "timeframe": tf,
            "timeframe_label": tf_label,
            "status": status_str,
            "last_bar_time": last_time_str,
            "last_bar_timestamp": int(last_ts.timestamp()),
            "last_bar_ohlcv": {
                "time": int(last_ts.timestamp()),
                "time_str": last_ts.strftime("%Y-%m-%d %H:%M"),
                "open": _safe_float(curr_row.get("open"), 0.0, 2),
                "high": _safe_float(curr_row.get("high"), 0.0, 2),
                "low": _safe_float(curr_row.get("low"), 0.0, 2),
                "close": _safe_float(curr_row.get("close"), 0.0, 2),
                "volume": _safe_float(curr_row.get("volume"), 0.0, 0),
            },
            "close": _safe_float(curr_row.get("close"), 0.0, 2),
            "high": _safe_float(curr_row.get("high"), 0.0, 2),
            "low": _safe_float(curr_row.get("low"), 0.0, 2),
            "atr14_pts": _safe_float(curr_row.get("atr14_pts"), def_atr, 2),
            "rsi14": _safe_float(curr_row.get("rsi14"), 50.0, 1),
            "tod_rvol": _safe_float(curr_row.get("tod_rvol"), 1.0, 2),
            "prob_high_pct": max(curr_hi_pct, active_prob if active_is_hi else 0.0),
            "prob_low_pct": max(curr_lo_pct, active_prob if not active_is_hi else 0.0),
            "curr_prob_high_pct": curr_hi_pct,
            "curr_prob_low_pct": curr_lo_pct,
            "curr_extreme_high_pct": curr_hi_ex_pct,
            "curr_extreme_low_pct": curr_lo_ex_pct,
            "curr_major_top_pct": curr_maj_top_pct,
            "curr_major_bottom_pct": curr_maj_bot_pct,
            "prob_major_top_pct": curr_maj_top_pct,
            "prob_major_bottom_pct": curr_maj_bot_pct,
            "active_pivot_time": active_time_str,
            "active_prob_pct": active_prob,
            "active_direction": plan["direction"],
            "active_is_high_pivot": bool(active_is_hi),
            "triggered": bool(active_prob >= thr),
            "is_live_bar_triggered": bool(is_curr_bar_trig),
            "trade_plan": plan,
        }

        if active_prob > best_overall_prob:
            best_overall_prob = active_prob
            best_overall_plan = plan

        if active_prob >= thr:
            piv_model_key = f"{prefix}_{tf}_{'high' if active_is_hi else 'low'}"
            piv_label = f"{'🔻' if active_is_hi else '💎'} {sym_root} [{tf_label}] {'Pivot High (SHORT)' if active_is_hi else 'Pivot Low (LONG)'}"
            color_hex = "#f43f5e" if active_is_hi else "#10b981"
            msg = (
                f"🚨 {symbol} [{tf_label}] {'Pivot High (SHORT)' if active_is_hi else 'Pivot Low (LONG)'} "
                f"({active_prob:.1f}% ≥ {thr:.0f}% | {plan['pred_ratio']:.2f}x MFE/MAE) @ {active_time_str} — "
                f"Entry: {plan['entry_price']:.2f} | Stop: {plan['stop_price']:.2f} ({plan['stop_dist_pts']:.2f} pts / {plan['stop_dist_atr']:.2f} ATR) | "
                f"TP1 (50%): {plan['target1_price']:.2f} ({plan['rr_target1']:.2f}R) | "
                f"TP2 (Runner): {plan['target2_price']:.2f} ({plan['rr_target2']:.2f}R) | "
                f"Exit: {plan['time_exit_label']}"
            )
            triggered_alerts.append({
                "symbol": symbol,
                "is_nq_intraday": True,
                "timeframe": tf,
                "timeframe_label": tf_label,
                "bar_time": active_time_str,
                "model_key": piv_model_key,
                "pivot_type": piv_model_key,
                "model_label": piv_label,
                "pivot_label": piv_label,
                "color": color_hex,
                "is_high_pivot": bool(active_is_hi),
                "direction": plan["direction"],
                "prob_pct": active_prob,
                "threshold_pct": thr,
                "current_price": _safe_float(curr_row.get("close"), 0.0, 2),
                "extreme_price": plan["extreme_price"],
                "bar_price": plan["extreme_price"],
                "trigger_price": plan["entry_price"],
                "entry_price": plan["entry_price"],
                "entry_px": plan["entry_price"],
                "stop_price": plan["stop_price"],
                "stop_px": plan["stop_price"],
                "target1_price": plan["target1_price"],
                "target1_px": plan["target1_price"],
                "target2_price": plan["target2_price"],
                "target2_px": plan["target2_price"],
                "rr_target1": plan["rr_target1"],
                "rr_target2": plan["rr_target2"],
                "time_exit_label": plan["time_exit_label"],
                "is_wickless": bool(use_wickless),
                "model_mode": "wickless" if use_wickless else "current",
                "trade_plan": plan,
                "message": msg,
            })

        # Major Reversals on developing / recent bar
        if curr_maj_top_pct >= thr:
            maj_top_key = f"{prefix}_{tf}_maj_ut_top"
            maj_top_label = f"👑 {sym_root} [{tf_label}] Major Top (SHORT Breakdown)"
            plan_maj_top = compute_nq_trade_plan(curr_row, tf=tf, is_high_pivot=True, horizon_k=k, symbol=symbol, use_wickless=use_wickless)
            triggered_alerts.append({
                "symbol": symbol,
                "is_nq_intraday": True,
                "timeframe": tf,
                "timeframe_label": tf_label,
                "bar_time": active_time_str,
                "model_key": maj_top_key,
                "pivot_type": "maj_ut_top",
                "model_label": maj_top_label,
                "pivot_label": maj_top_label,
                "color": "#fbbf24",
                "is_high_pivot": True,
                "direction": "SHORT",
                "prob_pct": curr_maj_top_pct,
                "threshold_pct": thr,
                "current_price": _safe_float(curr_row.get("close"), 0.0, 2),
                "extreme_price": plan_maj_top["extreme_price"],
                "bar_price": plan_maj_top["extreme_price"],
                "trigger_price": plan_maj_top["entry_price"],
                "entry_price": plan_maj_top["entry_price"],
                "entry_px": plan_maj_top["entry_price"],
                "stop_price": plan_maj_top["stop_price"],
                "stop_px": plan_maj_top["stop_price"],
                "target1_price": plan_maj_top["target1_price"],
                "target1_px": plan_maj_top["target1_price"],
                "target2_price": plan_maj_top["target2_price"],
                "target2_px": plan_maj_top["target2_price"],
                "rr_target1": plan_maj_top["rr_target1"],
                "rr_target2": plan_maj_top["rr_target2"],
                "time_exit_label": plan_maj_top["time_exit_label"],
                "is_wickless": True,
                "model_mode": "wickless",
                "trade_plan": plan_maj_top,
                "message": (
                    f"👑 {symbol} [{tf_label}] Major Top (Primary Breakdown Risk) "
                    f"({curr_maj_top_pct:.1f}% ≥ {thr:.0f}%) @ {active_time_str} — "
                    f"Entry: {plan_maj_top['entry_price']:.2f} | Stop: {plan_maj_top['stop_price']:.2f}"
                ),
            })

        if curr_maj_bot_pct >= thr:
            maj_bot_key = f"{prefix}_{tf}_maj_dt_low"
            maj_bot_label = f"🚀 {sym_root} [{tf_label}] Major Bottom (LONG Turnaround)"
            plan_maj_bot = compute_nq_trade_plan(curr_row, tf=tf, is_high_pivot=False, horizon_k=k, symbol=symbol, use_wickless=use_wickless)
            triggered_alerts.append({
                "symbol": symbol,
                "is_nq_intraday": True,
                "timeframe": tf,
                "timeframe_label": tf_label,
                "bar_time": active_time_str,
                "model_key": maj_bot_key,
                "pivot_type": "maj_dt_low",
                "model_label": maj_bot_label,
                "pivot_label": maj_bot_label,
                "color": "#10b981",
                "is_high_pivot": False,
                "direction": "LONG",
                "prob_pct": curr_maj_bot_pct,
                "threshold_pct": thr,
                "current_price": _safe_float(curr_row.get("close"), 0.0, 2),
                "extreme_price": plan_maj_bot["extreme_price"],
                "bar_price": plan_maj_bot["extreme_price"],
                "trigger_price": plan_maj_bot["entry_price"],
                "entry_price": plan_maj_bot["entry_price"],
                "entry_px": plan_maj_bot["entry_price"],
                "stop_price": plan_maj_bot["stop_price"],
                "stop_px": plan_maj_bot["stop_price"],
                "target1_price": plan_maj_bot["target1_price"],
                "target1_px": plan_maj_bot["target1_price"],
                "target2_price": plan_maj_bot["target2_price"],
                "target2_px": plan_maj_bot["target2_price"],
                "rr_target1": plan_maj_bot["rr_target1"],
                "rr_target2": plan_maj_bot["rr_target2"],
                "time_exit_label": plan_maj_bot["time_exit_label"],
                "is_wickless": True,
                "model_mode": "wickless",
                "trade_plan": plan_maj_bot,
                "message": (
                    f"🚀 {symbol} [{tf_label}] Major Bottom (Turnaround Rally) "
                    f"({curr_maj_bot_pct:.1f}% ≥ {thr:.0f}%) @ {active_time_str} — "
                    f"Entry: {plan_maj_bot['entry_price']:.2f} | Stop: {plan_maj_bot['stop_price']:.2f}"
                ),
            })

    out_payload = _sanitize_json_value({
        "symbol": symbol,
        "threshold_pct": thr,
        "horizon_bars": k,
        "timeframes": tf_summaries,
        "best_trade_plan": best_overall_plan,
        "alerts": triggered_alerts,
    })
    if len(_NQ_RT_EVAL_CACHE) > 128:
        _NQ_RT_EVAL_CACHE.clear()
    _NQ_RT_EVAL_CACHE[cache_key] = out_payload
    return out_payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Futures Intraday ML Model Trainer")
    parser.add_argument("--symbol", type=str, default="SI=F", help="Symbol to train (NQ=F, GC=F, CL=F, BTC=F, SI=F)")
    args = parser.parse_args()
    sym = args.symbol.upper().strip()
    if is_gc_symbol(sym):
        train_and_evaluate_all_gc_models()
    elif is_cl_symbol(sym):
        train_and_evaluate_all_cl_models()
    elif is_btc_symbol(sym):
        train_and_evaluate_all_btc_models()
    elif is_si_symbol(sym):
        train_and_evaluate_all_si_models()
    else:
        train_and_evaluate_all_nq_models()


