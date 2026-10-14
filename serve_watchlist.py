#!/usr/bin/env python3
"""serve_watchlist.py — Universal Chart Viewer, Ratio Analyzer & Organized Watchlist Manager.

Institutional 3-pane web workspace (running on port 8789) for organizing multi-group
watchlists and viewing interactive charts across all asset classes in the EP universe:
- Universal Asset Support:
  1. Individual Stocks (~6,240 universe symbols with RS & technical metrics)
  2. SPDR & Sector/Industry ETFs (XBI, XLK, SMH, IGV, XLV, XLF, XLE, KRE, SPY, QQQ, IWM, etc.)
  3. Sectors (154 equal-weighted composite OHLCV indices built from liquid constituent stocks)
  4. Themes (187 equal-weighted composite OHLCV indices built from liquid constituent stocks)
  5. Universal Ratio Charts (Part A ÷ Part B where Part A and Part B can be ANY Stock, SPDR ETF,
     Sector, or Theme — e.g., NVDA ÷ SMH, XBI ÷ SPY, SECTOR:Biotechnology ÷ XBI, THEME:AI ÷ QQQ)
- Organized Watchlist System:
  - Create, rename, reorder, and delete named watchlist groups/folders
  - Add/remove stocks, ETFs, sectors, themes, or saved ratio pairs to any watchlist
  - Assign color flags (Red, Amber, Green, Blue, Purple) and custom notes per watchlist item
  - Export any watchlist to TradingView format
- Toggleable Info & Performance Panel:
  - Multi-timeframe returns (1D, 1W, 1M, 3M, 6M, YTD, 1Y), Alpha vs SPY/QQQ, RS Score,
    Stage 2 trend template, ADR%, 52-week high/low distance, and MA alignment
  - In Ratio Mode: side-by-side performance comparison of Numerator (Part A) vs Denominator (Part B)
    plus the Ratio curve's own returns and trend posture
  - Constituent leader tables for Sectors, Themes, and ETFs (click any constituent to inspect)
  - Recent Strong EP Events from datastore with gap %, RVOL, V2 ML scores, and strategy tiers
- Chart Engine:
  - Candlestick & Volume + 20D Vol MA
  - Configurable EMAs/SMAs (8 EMA, 10 EMA, 20 EMA, 21 EMA, 50 SMA, 200 SMA) + RS Line vs SPY
  - Larsson Line Ribbon (Yellow / Blue / Gray polygon fill + dynamic fastest EMA span)
  - Automatic markers for Recent Strong EP Events directly on the chart
  - Interactive Drawing Tools (Trendline, Ray, Horizontal Line, Fib Retracement, Box, % Measure)
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

import config
import datastore
import ep_catalyst_enricher
import indicators
import labels
import scanner_core
import thematic_engine
import train_all_pivot_models as tap
import train_uptrend_pivot_model as tup

app = Flask(__name__)

RS_DATA_PATH = Path("data/simulations/rs_universe_scores.parquet")
EP_STUDY_PATH = Path("data/simulations/ep_combined_study_scored.parquet")
OPT_TRADES_PATH = Path("data/simulations/optimized_ep_strategy_trades.parquet")
UPTREND_PIVOT_MODEL_PATH = Path("data/models/uptrend_pivot_high_classifier.pkl")
UPTREND_PIVOT_LOW_MODEL_PATH = Path("data/models/uptrend_pivot_low_classifier.pkl")
DOWNTREND_PIVOT_HIGH_MODEL_PATH = Path("data/models/downtrend_pivot_high_classifier.pkl")
DOWNTREND_PIVOT_LOW_MODEL_PATH = Path("data/models/downtrend_pivot_low_classifier.pkl")
UPTREND_PIVOT_MODEL_V4_PATH = Path("data/models/uptrend_pivot_high_classifier_v4.pkl")
UPTREND_PIVOT_LOW_MODEL_V4_PATH = Path("data/models/uptrend_pivot_low_classifier_v4.pkl")
DOWNTREND_PIVOT_HIGH_MODEL_V4_PATH = Path("data/models/downtrend_pivot_high_classifier_v4.pkl")
DOWNTREND_PIVOT_LOW_MODEL_V4_PATH = Path("data/models/downtrend_pivot_low_classifier_v4.pkl")
EXTREME_PEAK_MODEL_PATH = Path("data/models/ep_extreme_peak_classifier.pkl")
WATCHLISTS_FILE = Path("data/user_watchlists.json")
ANNOTATIONS_DIR = Path("data/annotations")
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

# Curated SPDR & Major Sector/Theme ETF Metadata
SPDR_ETF_META: Dict[str, Dict[str, str]] = {
    "SPY": {"name": "SPDR S&P 500 ETF Trust", "category": "Market Benchmark", "focus": "S&P 500 Large Cap"},
    "QQQ": {"name": "Invesco QQQ Trust", "category": "Market Benchmark", "focus": "Nasdaq 100 Growth / Tech"},
    "IWM": {"name": "iShares Russell 2000 ETF", "category": "Market Benchmark", "focus": "Small Cap 2000"},
    "DIA": {"name": "SPDR Dow Jones Industrial Average ETF", "category": "Market Benchmark", "focus": "Dow 30 Industrials"},
    "MDY": {"name": "SPDR S&P MidCap 400 ETF Trust", "category": "Market Benchmark", "focus": "MidCap 400"},
    "RSP": {"name": "Invesco S&P 500 Equal Weight ETF", "category": "Market Benchmark", "focus": "S&P 500 Equal Weight"},
    # Select Sector SPDRs
    "XLK": {"name": "Technology Select Sector SPDR Fund", "category": "SPDR Sector", "focus": "Technology"},
    "XLV": {"name": "Health Care Select Sector SPDR Fund", "category": "SPDR Sector", "focus": "Healthcare"},
    "XLF": {"name": "Financial Select Sector SPDR Fund", "category": "SPDR Sector", "focus": "Financials"},
    "XLE": {"name": "Energy Select Sector SPDR Fund", "category": "SPDR Sector", "focus": "Energy"},
    "XLI": {"name": "Industrial Select Sector SPDR Fund", "category": "SPDR Sector", "focus": "Industrials"},
    "XLY": {"name": "Consumer Discretionary Select Sector SPDR", "category": "SPDR Sector", "focus": "Consumer Discretionary"},
    "XLP": {"name": "Consumer Staples Select Sector SPDR", "category": "SPDR Sector", "focus": "Consumer Staples"},
    "XLC": {"name": "Communication Services Select Sector SPDR", "category": "SPDR Sector", "focus": "Communication Services"},
    "XLB": {"name": "Materials Select Sector SPDR Fund", "category": "SPDR Sector", "focus": "Basic Materials"},
    "XLRE": {"name": "Real Estate Select Sector SPDR Fund", "category": "SPDR Sector", "focus": "Real Estate"},
    "XLU": {"name": "Utilities Select Sector SPDR Fund", "category": "SPDR Sector", "focus": "Utilities"},
    # SPDR Industry ETFs
    "XBI": {"name": "SPDR S&P Biotech ETF", "category": "SPDR Industry", "focus": "Biotechnology", "linked_sector": "Biotechnology"},
    "XOP": {"name": "SPDR S&P Oil & Gas Exploration & Prod ETF", "category": "SPDR Industry", "focus": "Oil & Gas E&P", "linked_sector": "Oil & Gas E&P"},
    "XME": {"name": "SPDR S&P Metals & Mining ETF", "category": "SPDR Industry", "focus": "Metals & Mining", "linked_theme": "Metals & Mining"},
    "XAR": {"name": "SPDR S&P Aerospace & Defense ETF", "category": "SPDR Industry", "focus": "Aerospace & Defense", "linked_sector": "Aerospace & Defense"},
    "XHB": {"name": "SPDR S&P Homebuilders ETF", "category": "SPDR Industry", "focus": "Homebuilding", "linked_sector": "Residential Construction"},
    "XRT": {"name": "SPDR S&P Retail ETF", "category": "SPDR Industry", "focus": "Retail", "linked_theme": "Consumer Retail"},
    "KRE": {"name": "SPDR S&P Regional Banking ETF", "category": "SPDR Industry", "focus": "Regional Banks", "linked_sector": "Banks - Regional"},
    "KBE": {"name": "SPDR S&P Bank ETF", "category": "SPDR Industry", "focus": "Banking", "linked_sector": "Banks - Regional"},
    "XSD": {"name": "SPDR S&P Semiconductor ETF", "category": "SPDR Industry", "focus": "Semiconductors", "linked_sector": "Semiconductors"},
    "XSW": {"name": "SPDR S&P Software & Services ETF", "category": "SPDR Industry", "focus": "Software", "linked_sector": "Software - Application"},
    "XHE": {"name": "SPDR S&P Health Care Equipment ETF", "category": "SPDR Industry", "focus": "Medical Devices", "linked_sector": "Medical Devices"},
    "XPH": {"name": "SPDR S&P Pharmaceuticals ETF", "category": "SPDR Industry", "focus": "Pharmaceuticals", "linked_sector": "Drug Manufacturers - Specialty & Generic"},
    "XES": {"name": "SPDR S&P Oil & Gas Equipment & Services", "category": "SPDR Industry", "focus": "Oil & Gas Equipment", "linked_sector": "Oil & Gas Equipment & Services"},
    # Key Thematic & Industry ETFs
    "SMH": {"name": "VanEck Semiconductor ETF", "category": "Industry ETF", "focus": "Semiconductors", "linked_sector": "Semiconductors"},
    "SOXX": {"name": "iShares Semiconductor ETF", "category": "Industry ETF", "focus": "Semiconductors", "linked_sector": "Semiconductors"},
    "IGV": {"name": "iShares Expanded Tech-Software Sector ETF", "category": "Industry ETF", "focus": "Enterprise Software", "linked_sector": "Software - Infrastructure"},
    "IBB": {"name": "iShares Biotechnology ETF", "category": "Industry ETF", "focus": "Biotechnology", "linked_sector": "Biotechnology"},
    "ARKK": {"name": "ARK Innovation ETF", "category": "Industry ETF", "focus": "Disruptive Innovation"},
    "ITA": {"name": "iShares U.S. Aerospace & Defense ETF", "category": "Industry ETF", "focus": "Aerospace & Defense", "linked_sector": "Aerospace & Defense"},
    "ITB": {"name": "iShares U.S. Home Construction ETF", "category": "Industry ETF", "focus": "Homebuilders", "linked_sector": "Residential Construction"},
    "TAN": {"name": "Invesco Solar ETF", "category": "Industry ETF", "focus": "Solar Energy", "linked_sector": "Solar"},
    "URA": {"name": "Global X Uranium ETF", "category": "Industry ETF", "focus": "Uranium & Nuclear", "linked_theme": "Uranium & Nuclear"},
    "GDX": {"name": "VanEck Gold Miners ETF", "category": "Industry ETF", "focus": "Gold Miners", "linked_sector": "Gold"},
    "SIL": {"name": "Global X Silver Miners ETF", "category": "Industry ETF", "focus": "Silver Miners", "linked_sector": "Silver"},
    "COPX": {"name": "Global X Copper Miners ETF", "category": "Industry ETF", "focus": "Copper Miners", "linked_sector": "Copper"},
    "OIH": {"name": "VanEck Oil Services ETF", "category": "Industry ETF", "focus": "Oil Services", "linked_sector": "Oil & Gas Equipment & Services"},
    "HACK": {"name": "ETFMG Prime Cyber Security ETF", "category": "Industry ETF", "focus": "Cybersecurity", "linked_theme": "Cybersecurity"},
    "BOTZ": {"name": "Global X Robotics & Artificial Intelligence ETF", "category": "Industry ETF", "focus": "Robotics & AI", "linked_theme": "Artificial Intelligence"},
}

# In-memory caches
_RS_DF: Optional[pd.DataFrame] = None
_EP_DF: Optional[pd.DataFrame] = None
_OPT_DF: Optional[pd.DataFrame] = None
_EXTREME_PEAK_MODEL: Optional[Dict[str, Any]] = None
_NAMES_MAP: Dict[str, str] = {}
_ETF_SET: set[str] = set()
_COMPOSITE_BAR_CACHE: Dict[str, Tuple[float, pd.DataFrame, List[str]]] = {}
_CATALOG_CACHE: Optional[Tuple[float, Dict[str, Any]]] = None


def _clean_val(v: Any, default: Any = None) -> Any:
    """Safely convert numpy/pandas scalars and NaNs into JSON-clean Python primitives."""
    if v is None:
        return default
    if isinstance(v, (float, np.floating)):
        if math.isnan(v) or math.isinf(v):
            return default
        return float(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if pd.isna(v):
        return default
    return v


def load_universe_metadata() -> None:
    global _NAMES_MAP, _ETF_SET
    if not _NAMES_MAP and config.UNIVERSE_FILE.exists():
        try:
            udf = pd.read_csv(config.UNIVERSE_FILE)
            _NAMES_MAP = dict(zip(udf["symbol"].astype(str), udf["name"].fillna("").astype(str)))
            if "etf" in udf.columns:
                _ETF_SET = set(udf[udf["etf"] == True]["symbol"].astype(str).tolist())
        except Exception as e:
            print(f"Warning: could not load universe metadata: {e}")
    _ETF_SET.update(SPDR_ETF_META.keys())


def get_rs_df() -> pd.DataFrame:
    global _RS_DF
    if _RS_DF is None:
        if RS_DATA_PATH.exists():
            try:
                _RS_DF = pd.read_parquet(RS_DATA_PATH)
            except Exception as e:
                print(f"Warning: could not read {RS_DATA_PATH}: {e}")
                _RS_DF = pd.DataFrame()
        else:
            _RS_DF = pd.DataFrame()
    return _RS_DF


def get_ep_df() -> pd.DataFrame:
    global _EP_DF
    if _EP_DF is None:
        if EP_STUDY_PATH.exists():
            try:
                _EP_DF = pd.read_parquet(EP_STUDY_PATH)
            except Exception as e:
                print(f"Warning: could not read {EP_STUDY_PATH}: {e}")
                _EP_DF = pd.DataFrame()
        else:
            _EP_DF = pd.DataFrame()
    return _EP_DF


def get_opt_df() -> pd.DataFrame:
    global _OPT_DF
    if _OPT_DF is None:
        if OPT_TRADES_PATH.exists():
            try:
                _OPT_DF = pd.read_parquet(OPT_TRADES_PATH)
            except Exception as e:
                print(f"Warning: could not read {OPT_TRADES_PATH}: {e}")
                _OPT_DF = pd.DataFrame()
        else:
            _OPT_DF = pd.DataFrame()
    return _OPT_DF


_PIVOT_MODELS_CACHE: Dict[str, Any] = {}


def get_extreme_peak_model() -> Optional[Dict[str, Any]]:
    global _EXTREME_PEAK_MODEL
    if _EXTREME_PEAK_MODEL is None:
        target_path = UPTREND_PIVOT_MODEL_PATH if UPTREND_PIVOT_MODEL_PATH.exists() else EXTREME_PEAK_MODEL_PATH
        if target_path.exists():
            try:
                _EXTREME_PEAK_MODEL = joblib.load(target_path)
            except Exception as e:
                print(f"Warning: could not load {target_path}: {e}")
                _EXTREME_PEAK_MODEL = None
    return _EXTREME_PEAK_MODEL


def get_all_pivot_models(model_ver: str = "v3") -> Dict[str, Any]:
    """Lazily loads all 4 structural swing-pivot classifiers for the requested version ('v3' or 'v4')."""
    global _PIVOT_MODELS_CACHE
    ver = "v4" if str(model_ver or "v3").strip().lower().startswith("v4") else "v3"
    if ver not in _PIVOT_MODELS_CACHE or not isinstance(_PIVOT_MODELS_CACHE.get(ver), dict):
        _PIVOT_MODELS_CACHE[ver] = {}
    bucket = _PIVOT_MODELS_CACHE[ver]

    path_pairs = (
        [
            ("ut_high", UPTREND_PIVOT_MODEL_V4_PATH if UPTREND_PIVOT_MODEL_V4_PATH.exists() else UPTREND_PIVOT_MODEL_PATH),
            ("ut_low", UPTREND_PIVOT_LOW_MODEL_V4_PATH if UPTREND_PIVOT_LOW_MODEL_V4_PATH.exists() else UPTREND_PIVOT_LOW_MODEL_PATH),
            ("dt_high", DOWNTREND_PIVOT_HIGH_MODEL_V4_PATH if DOWNTREND_PIVOT_HIGH_MODEL_V4_PATH.exists() else DOWNTREND_PIVOT_HIGH_MODEL_PATH),
            ("dt_low", DOWNTREND_PIVOT_LOW_MODEL_V4_PATH if DOWNTREND_PIVOT_LOW_MODEL_V4_PATH.exists() else DOWNTREND_PIVOT_LOW_MODEL_PATH),
        ]
        if ver == "v4"
        else [
            ("ut_high", UPTREND_PIVOT_MODEL_PATH),
            ("ut_low", UPTREND_PIVOT_LOW_MODEL_PATH),
            ("dt_high", DOWNTREND_PIVOT_HIGH_MODEL_PATH),
            ("dt_low", DOWNTREND_PIVOT_LOW_MODEL_PATH),
        ]
    )
    for key, path in path_pairs:
        if key not in bucket and path.exists():
            try:
                bucket[key] = joblib.load(path)
            except Exception as e:
                print(f"Warning: could not load pivot model {key} ({ver}) from {path}: {e}")
    return bucket


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)


def _extract_snapped_pivot_events(
    sub: pd.DataFrame,
    probs_arr: np.ndarray,
    ex_arr: np.ndarray,
    fb_arr: np.ndarray,
    mask_arr: np.ndarray,
    is_high_pivot: bool,
    pivot_type: str,
) -> List[Dict[str, Any]]:
    """Strictly causal (zero-lookahead) extraction of local-extremum pivot events (down to >=25% for
    dynamic UI threshold filtering). Uses trailing 5-bar window (`center=False`, bars `[i-4..i]` only)
    and snaps 2-bar stall confirmations at `i` to `i-1` only when `i-1` already occurred."""
    events: List[Dict[str, Any]] = []
    sub_dates = list(sub.index)
    highs_arr = sub["high"].to_numpy()
    lows_arr = sub["low"].to_numpy()
    adr_arr = sub["adr20"].to_numpy()
    rsi_arr = (sub["rsi14_high"] if is_high_pivot else sub["rsi14_low"]).to_numpy()
    roll_ext5 = (
        sub["high"].rolling(5, center=False, min_periods=1).max().to_numpy()
        if is_high_pivot
        else sub["low"].rolling(5, center=False, min_periods=1).min().to_numpy()
    )

    last_idx = -999
    for i, idx_dt in enumerate(sub_dates):
        pr = float(probs_arr[i])
        if pr < 0.25 or not mask_arr[i]:
            continue
        cur_adr = max(1e-6, float(adr_arr[i]))
        if is_high_pivot:
            loc_max = float(roll_ext5[i])
            is_bar_ext = highs_arr[i] >= loc_max * 0.996
            is_prev_ext = (i > 0) and (highs_arr[i - 1] >= loc_max * 0.998) and (highs_arr[i] >= highs_arr[i - 1] * 0.985)
            if not (is_bar_ext or is_prev_ext):
                continue
            target_idx = (
                (i - 1)
                if (i > 0 and highs_arr[i - 1] > highs_arr[i] and (highs_arr[i - 1] - highs_arr[i]) / cur_adr <= 0.65)
                else i
            )
        else:
            loc_min = float(roll_ext5[i])
            is_bar_ext = lows_arr[i] <= loc_min * 1.004
            is_prev_ext = (i > 0) and (lows_arr[i - 1] <= loc_min * 1.002) and (lows_arr[i] <= lows_arr[i - 1] * 1.015)
            if not (is_bar_ext or is_prev_ext):
                continue
            target_idx = (
                (i - 1)
                if (i > 0 and lows_arr[i - 1] < lows_arr[i] and (lows_arr[i] - lows_arr[i - 1]) / cur_adr <= 0.65)
                else i
            )

        target_dt = sub_dates[target_idx]
        t_str = target_dt.strftime("%Y-%m-%d")
        prec = 4 if abs(float(highs_arr[target_idx])) < 10.0 else 2
        hi_val = round(float(highs_arr[target_idx]), prec)
        lo_val = round(float(lows_arr[target_idx]), prec)
        ev_obj = {
            "date": t_str,
            "trigger_date": idx_dt.strftime("%Y-%m-%d"),
            "type": pivot_type,
            "prob_pct": round(pr * 100.0, 1),
            "at_extreme_pct": round(float(ex_arr[i]) * 100.0, 1),
            "at_high_pct": round(float(ex_arr[i]) * 100.0, 1),
            "full_bar_pct": round(float(fb_arr[i]) * 100.0, 1),
            "rsi14": round(float(rsi_arr[target_idx]), 1),
            "high": hi_val,
            "low": lo_val,
            "intrabar_trigger_price": hi_val if is_high_pivot else lo_val,
            "trigger_prices_by_thr": {},
            "touch_trigger_price": None,
        }

        if events and events[-1]["date"] == t_str:
            if ev_obj["prob_pct"] > events[-1]["prob_pct"]:
                events[-1] = ev_obj
        elif events and (target_idx - last_idx) <= 2:
            if is_high_pivot:
                new_extreme = highs_arr[target_idx] > events[-1]["high"] + 0.35 * cur_adr
            else:
                new_extreme = lows_arr[target_idx] < events[-1]["low"] - 0.35 * cur_adr
            if new_extreme or (ev_obj["prob_pct"] >= events[-1]["prob_pct"] + 8.0):
                events.append(ev_obj)
                last_idx = target_idx
        else:
            events.append(ev_obj)
            last_idx = target_idx

    return events


def _ensure_causal_rma_state(df_source: pd.DataFrame, sub: pd.DataFrame) -> pd.DataFrame:
    """Ensures `sub` carries exact prior-bar close (`prev_close`) and Wilder RMA state
    (`prev_ag{p}`, `prev_al{p}` for p in (14, 9, 5)) computed causally from `df_source`."""
    req_cols = ("prev_close", "prev_ag14", "prev_al14", "prev_ag9", "prev_al9", "prev_ag5", "prev_al5")
    if all(col in sub.columns for col in req_cols):
        return sub
    if "close" not in df_source.columns:
        return sub
    out = sub.copy()
    c = df_source["close"].astype(float)
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    out["prev_close"] = c.shift(1).reindex(sub.index)
    for period in (14, 9, 5):
        alpha = 1.0 / period
        ag = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        al = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        out[f"prev_ag{period}"] = ag.shift(1).reindex(sub.index)
        out[f"prev_al{period}"] = al.shift(1).reindex(sub.index)
    return out


def _eval_intrabar_rsi_vec(
    p_cand: np.ndarray,
    big: pd.DataFrame,
    period: int,
    fallback_arr: np.ndarray,
) -> np.ndarray:
    """Evaluates exact causal Wilder's RSI(period) at candidate intrabar price `p_cand`
    using prior-bar close (`prev_close`) and prior-bar RMA gain/loss (`prev_ag{period}`, `prev_al{period}`)."""
    c_col = "prev_close"
    ag_col = f"prev_ag{period}"
    al_col = f"prev_al{period}"
    if c_col not in big.columns or ag_col not in big.columns or al_col not in big.columns:
        return fallback_arr
    prev_c = big[c_col].to_numpy(dtype=float)
    prev_ag = big[ag_col].to_numpy(dtype=float)
    prev_al = big[al_col].to_numpy(dtype=float)
    gain_c = np.maximum(p_cand - prev_c, 0.0)
    loss_c = np.maximum(prev_c - p_cand, 0.0)
    ag_c = (prev_ag * (period - 1.0) + gain_c) / float(period)
    al_c = (prev_al * (period - 1.0) + loss_c) / float(period)
    rsi_c = 100.0 - (100.0 / (1.0 + ag_c / (al_c + 1e-9)))
    return np.where(np.isnan(rsi_c), fallback_arr, rsi_c)


def _enrich_events_with_intrabar_triggers(
    sub: pd.DataFrame,
    events: List[Dict[str, Any]],
    model_bundle: Optional[Dict[str, Any]],
    mask_col: str,
    is_high_pivot: bool,
    n_steps: int = 41,
) -> List[Dict[str, Any]]:
    """Vectorized causal intrabar price sweep across `[Low, High]` for each detected pivot event:
    - For Pivot Highs (`ut_high`, `dt_high`): sweeps candidate intrabar high upward from `Low` (`w=0`) to `High` (`w=1`)
      to find the **lowest intrabar price** that classifies the bar as a Pivot High (`prob >= threshold`).
    - For Pivot Lows (`ut_low`, `dt_low`): sweeps candidate intrabar low downward from `High` (`w=0`) to `Low` (`w=1`)
      to find the **highest intrabar price** that classifies the bar as a Pivot Low (`prob >= threshold`).
    All candidate-price-dependent features (RSI(14/9/5), ADR extensions, Bollinger Z, 120d p95 stretch,
    multi-day max/min, consecutive RSI streaks, divergence triggers, weekly W-FRI RSI, Inverse-RSI distance,
    and developing intrabar rejection) are recomputed from `t-1` causal state so that `dist_from_10d_high_adr` /
    `dist_from_8d_low_adr` never look ahead to future EOD `High[t]` / `Low[t]`.
    """
    if not events or not model_bundle or sub is None or sub.empty:
        return events

    sub = _ensure_causal_rma_state(sub, sub)
    date_to_pos = {dt.strftime("%Y-%m-%d"): idx for idx, dt in enumerate(sub.index)}
    ev_pos = [date_to_pos.get(ev.get("trigger_date") or ev["date"], date_to_pos.get(ev["date"], 0)) for ev in events]
    target_pos = [date_to_pos.get(ev["date"], p) for ev, p in zip(events, ev_pos)]
    m = len(events)

    ev_rep_idx = np.repeat(ev_pos, n_steps)
    big = sub.iloc[ev_rep_idx].copy().reset_index(drop=True)

    lo_arr = sub["low"].to_numpy()[target_pos]
    hi_arr = sub["high"].to_numpy()[target_pos]
    lo_ev_arr = sub["low"].to_numpy()[ev_pos]
    hi_ev_arr = sub["high"].to_numpy()[ev_pos]
    adr_arr = np.maximum(1e-6, sub["adr20"].to_numpy()[ev_pos])

    w_1d = np.linspace(0.0, 1.0, n_steps)
    w = np.tile(w_1d, m)
    lo_rep = np.repeat(lo_arr, n_steps)
    hi_rep = np.repeat(hi_arr, n_steps)
    lo_ev_rep = np.repeat(lo_ev_arr, n_steps)
    hi_ev_rep = np.repeat(hi_ev_arr, n_steps)
    adr_rep = np.repeat(adr_arr, n_steps)

    rsi14_lo = big["rsi14_low"].to_numpy()
    rsi14_hi = big["rsi14_high"].to_numpy()
    rsi9_lo = big["rsi9_low"].to_numpy() if "rsi9_low" in big.columns else rsi14_lo
    rsi9_hi = big["rsi9_high"].to_numpy() if "rsi9_high" in big.columns else rsi14_hi
    rsi5_lo = big["rsi5_low"].to_numpy() if "rsi5_low" in big.columns else rsi14_lo
    rsi5_hi = big["rsi5_high"].to_numpy() if "rsi5_high" in big.columns else rsi14_hi

    if is_high_pivot:
        p_cand = lo_rep + w * (hi_rep - lo_rep)
        dh = (p_cand - hi_ev_rep) / adr_rep
        big["high"] = p_cand
        big["rsi14_high"] = _eval_intrabar_rsi_vec(p_cand, big, 14, rsi14_lo + w * (rsi14_hi - rsi14_lo))
        big["rsi9_high"] = _eval_intrabar_rsi_vec(p_cand, big, 9, rsi9_lo + w * (rsi9_hi - rsi9_lo))
        big["rsi5_high"] = _eval_intrabar_rsi_vec(p_cand, big, 5, rsi5_lo + w * (rsi5_hi - rsi5_lo))
        if "rsi14" in big.columns:
            big["rsi14"] = rsi14_lo + w * (big["rsi14"].to_numpy() - rsi14_lo)
        if "rsi5_vs_rsi14_high_spread" in big.columns:
            big["rsi5_vs_rsi14_high_spread"] = big["rsi5_high"] - big["rsi14_high"]

        for col in [
            "ext_ema8_adr",
            "ext_ema10_adr",
            "ext_ema20_adr",
            "ext_sma50_adr",
            "high_dist_sma50_adr",
            "high_dist_sma200_adr",
            "leg_adr_from_ema10",
            "leg_adr_from_ema20",
            "thrust_1d_adr",
            "thrust_2d_adr",
            "thrust_3d_adr",
            "thrust_5d_adr",
            "thrust_10d_adr",
            "thrust_20d_adr",
            "thrust_40d_adr",
            "breakout_above_10d_adr",
            "breakout_above_20d_adr",
            "open_to_high_adr",
            "dist_to_prior_swing_high_adr",
            "dist_to_prior_swing_low_adr",
            "avwap_from_60d_high_dist_adr",
            "avwap_from_60d_low_dist_adr",
            "inv_rsi75_dist_adr",
            "rs_vs_spy_5d_adr",
        ]:
            if col in big.columns:
                big[col] = big[col].to_numpy() + dh

        if "dist_from_10d_high_adr" in big.columns and "breakout_above_10d_adr" in big.columns:
            big["dist_from_10d_high_adr"] = np.maximum(0.0, -big["breakout_above_10d_adr"].to_numpy())
        if "dist_below_40d_high_adr" in big.columns:
            big["dist_below_40d_high_adr"] = np.maximum(0.0, big["dist_below_40d_high_adr"].to_numpy() - dh)
        if "parabolic_accel" in big.columns:
            big["parabolic_accel"] = (big["thrust_3d_adr"] / 3.0) / np.maximum(0.15, big["thrust_10d_adr"] / 10.0)
        if "ema10_anchor" in big.columns and "ext_ema10_pct" in big.columns:
            big["ext_ema10_pct"] = ((p_cand - big["ema10_anchor"].to_numpy()) / np.maximum(1e-6, big["ema10_anchor"].to_numpy())) * 100.0
        if "ema20_anchor" in big.columns and "ext_ema20_pct" in big.columns:
            big["ext_ema20_pct"] = ((p_cand - big["ema20_anchor"].to_numpy()) / np.maximum(1e-6, big["ema20_anchor"].to_numpy())) * 100.0
        if "sma50_anchor" in big.columns and "ext_sma50_pct" in big.columns:
            big["ext_sma50_pct"] = ((p_cand - big["sma50_anchor"].to_numpy()) / np.maximum(1e-6, big["sma50_anchor"].to_numpy())) * 100.0
        if "sma20_anchor" in big.columns and "std20_anchor" in big.columns and "bb_z_high" in big.columns:
            big["bb_z_high"] = ((p_cand - big["sma20_anchor"].to_numpy()) / np.maximum(1e-6, big["std20_anchor"].to_numpy())).clip(-4.0, 6.0)
        if "prev_bb_z_hi_max2" in big.columns and "bb_z_high_max3" in big.columns:
            big["bb_z_high_max3"] = np.maximum(big["prev_bb_z_hi_max2"].to_numpy(), big["bb_z_high"].to_numpy())
        if "roll_p95_ema10" in big.columns and "rel_ext_ema10_p95" in big.columns:
            big["rel_ext_ema10_p95"] = big["ext_ema10_adr"].to_numpy() / np.maximum(0.5, big["roll_p95_ema10"].to_numpy())
        if "roll_p95_ema20" in big.columns and "rel_ext_ema20_p95" in big.columns:
            big["rel_ext_ema20_p95"] = big["ext_ema20_adr"].to_numpy() / np.maximum(0.5, big["roll_p95_ema20"].to_numpy())
        if "prev_ext10_max2" in big.columns and "ext_ema10_adr_max3" in big.columns:
            big["ext_ema10_adr_max3"] = np.maximum(big["prev_ext10_max2"].to_numpy(), big["ext_ema10_adr"].to_numpy())
        if "prev_ext20_max2" in big.columns and "ext_ema20_adr_max3" in big.columns:
            big["ext_ema20_adr_max3"] = np.maximum(big["prev_ext20_max2"].to_numpy(), big["ext_ema20_adr"].to_numpy())
        if "prev_rsi14_hi_max4" in big.columns and "rsi14_high_max5" in big.columns:
            big["rsi14_high_max5"] = np.maximum(big["prev_rsi14_hi_max4"].to_numpy(), big["rsi14_high"].to_numpy())
        if "prev_rsi5_hi_max2" in big.columns and "rsi5_high_max3" in big.columns:
            big["rsi5_high_max3"] = np.maximum(big["prev_rsi5_hi_max2"].to_numpy(), big["rsi5_high"].to_numpy())
        if "rsi14_hi_shift3" in big.columns and "rsi14_high_diff3" in big.columns:
            big["rsi14_high_diff3"] = big["rsi14_high"].to_numpy() - big["rsi14_hi_shift3"].to_numpy()
        if "rsi14_hi_shift5" in big.columns and "rsi14_high_diff5" in big.columns:
            big["rsi14_high_diff5"] = big["rsi14_high"].to_numpy() - big["rsi14_hi_shift5"].to_numpy()
        if "prev_consec_rsi70_4d" in big.columns and "consec_rsi70_5d" in big.columns:
            big["consec_rsi70_5d"] = big["prev_consec_rsi70_4d"].to_numpy() + (big["rsi14_high"].to_numpy() >= 70.0).astype(float)
        if "prev_consec_rsi75_4d" in big.columns and "consec_rsi75_5d" in big.columns:
            big["consec_rsi75_5d"] = big["prev_consec_rsi75_4d"].to_numpy() + (big["rsi14_high"].to_numpy() >= 75.0).astype(float)
        if "prev_consec_rsi80_4d" in big.columns and "consec_rsi80_5d" in big.columns:
            big["consec_rsi80_5d"] = big["prev_consec_rsi80_4d"].to_numpy() + (big["rsi5_high"].to_numpy() >= 80.0).astype(float)
        if "prev_wave_high_18" in big.columns and "prev_wave_rsi_18" in big.columns:
            is_nh_18 = p_cand >= big["prev_wave_high_18"].to_numpy() * 0.985
            if "bearish_div_high" in big.columns:
                big["bearish_div_high"] = (is_nh_18 & (big["rsi14_high"].to_numpy() < big["prev_wave_rsi_18"].to_numpy() - 1.5)).astype(float)
            if "rsi_div_delta_high" in big.columns:
                big["rsi_div_delta_high"] = np.where(is_nh_18, np.maximum(0.0, big["prev_wave_rsi_18"].to_numpy() - big["rsi14_high"].to_numpy()), 0.0)
        if "prev_wave_high_35" in big.columns and "prev_wave_rsi_35" in big.columns:
            is_nh_35 = p_cand >= big["prev_wave_high_35"].to_numpy() * 0.985
            if "bearish_div_35d" in big.columns:
                big["bearish_div_35d"] = (is_nh_35 & (big["rsi14_high"].to_numpy() < big["prev_wave_rsi_35"].to_numpy() - 2.0)).astype(float)
            if "rsi_div_delta_35d" in big.columns:
                big["rsi_div_delta_35d"] = np.where(is_nh_35, np.maximum(0.0, big["prev_wave_rsi_35"].to_numpy() - big["rsi14_high"].to_numpy()), 0.0)
            if "prev_wave_ext_35" in big.columns and "ext_div_delta_35d" in big.columns:
                big["ext_div_delta_35d"] = np.where(is_nh_35, np.maximum(0.0, big["prev_wave_ext_35"].to_numpy() - big["ext_ema20_adr"].to_numpy()), 0.0)
        for k_col, d_col in [
            ("kiss_falling_ema20_adr", "ext_ema20_adr"),
            ("kiss_falling_sma50_adr", "high_dist_sma50_adr"),
        ]:
            if k_col in big.columns and d_col in big.columns:
                big[k_col] = big[d_col].abs()
        if "min_overhead_kiss_adr" in big.columns and "kiss_falling_ema20_adr" in big.columns and "kiss_falling_sma50_adr" in big.columns:
            big["min_overhead_kiss_adr"] = np.minimum(big["kiss_falling_ema20_adr"], big["kiss_falling_sma50_adr"])
        if "bounce_retracement_ratio" in big.columns and "prior_selloff_20d_adr" in big.columns:
            big["bounce_retracement_ratio"] = big["thrust_10d_adr"].clip(lower=0.0) / np.maximum(1.5, big["prior_selloff_20d_adr"])

        # Weekly W-FRI & MTF confluence updates at p_cand
        if "prev_w_c" in big.columns and "weekly_rsi14_high" in big.columns:
            wtd_hi_c = np.maximum(big["wtd_high_prev"].to_numpy(), p_cand)
            pw_c = big["prev_w_c"].to_numpy()
            gh_w = np.maximum(wtd_hi_c - pw_c, 0.0)
            lh_w = np.maximum(pw_c - wtd_hi_c, 0.0)
            agh_w = (big["prev_w_ag14"].to_numpy() * 13.0 + gh_w) / 14.0
            alh_w = (big["prev_w_al14"].to_numpy() * 13.0 + lh_w) / 14.0
            big["weekly_rsi14_high"] = 100.0 - 100.0 / (1.0 + agh_w / (alh_w + 1e-9))
            if "weekly_ext_ema10w_adr" in big.columns:
                big["weekly_ext_ema10w_adr"] = (wtd_hi_c - big["prev_w_ema10"].to_numpy()) / np.maximum(1e-6, big["prev_w_adr"].to_numpy())

        # Scale intrabar rejection/wick features progressively with w (0 -> 1)
        big["range_to_adr"] = (p_cand - lo_rep) / adr_rep
        for scale_col in [
            "upper_wick_pct",
            "fade_from_high_adr",
            "micro_fade_adr",
            "micro_upper_wick_pct",
            "red_reversal_bar",
            "two_bar_top_reversal",
            "failed_high_retest",
            "rejected_at_falling_ma",
            "overhead_rejection_score",
        ]:
            if scale_col in big.columns:
                big[scale_col] = big[scale_col].to_numpy() * w
        if "close_pos" in big.columns:
            big["close_pos"] = 1.0 - w * (1.0 - big["close_pos"].to_numpy())
        if "reversal_vol_score" in big.columns:
            big["reversal_vol_score"] = big["rvol"] * big["fade_from_high_adr"]
        if "climax_vol_stretch" in big.columns:
            big["climax_vol_stretch"] = big["rvol"] * big["ext_ema10_adr"].clip(lower=0.0)
    else:
        p_cand = hi_rep - w * (hi_rep - lo_rep)
        dl = (p_cand - lo_ev_rep) / adr_rep
        big["low"] = p_cand
        big["rsi14_low"] = _eval_intrabar_rsi_vec(p_cand, big, 14, rsi14_hi - w * (rsi14_hi - rsi14_lo))
        big["rsi9_low"] = _eval_intrabar_rsi_vec(p_cand, big, 9, rsi9_hi - w * (rsi9_hi - rsi9_lo))
        big["rsi5_low"] = _eval_intrabar_rsi_vec(p_cand, big, 5, rsi5_hi - w * (rsi5_hi - rsi5_lo))
        if "rsi14" in big.columns:
            big["rsi14"] = rsi14_hi - w * (rsi14_hi - big["rsi14"].to_numpy())
        if "rsi14_vs_rsi5_low_spread" in big.columns:
            big["rsi14_vs_rsi5_low_spread"] = big["rsi14_low"] - big["rsi5_low"]

        for col in [
            "low_dist_ema8_adr",
            "low_dist_ema10_adr",
            "low_dist_ema20_adr",
            "low_dist_sma50_adr",
            "low_dist_sma200_adr",
            "dist_to_prior_swing_high_adr",
            "dist_to_prior_swing_low_adr",
            "avwap_from_60d_high_dist_adr",
            "avwap_from_60d_low_dist_adr",
            "rs_vs_spy_5d_adr",
        ]:
            if col in big.columns:
                big[col] = big[col].to_numpy() + dl
        for col in [
            "stretch_below_ema10_adr",
            "stretch_below_ema20_adr",
            "stretch_below_sma50_adr",
            "down_leg_adr_from_ema10",
            "down_leg_adr_from_ema20",
            "drop_1d_adr",
            "drop_2d_adr",
            "drop_3d_adr",
            "drop_5d_adr",
            "drop_10d_adr",
            "drop_15d_adr",
            "drop_20d_adr",
            "drop_40d_adr",
            "undercut_10d_low_adr",
            "undercut_20d_low_adr",
            "open_to_low_adr",
            "inv_rsi25_dist_adr",
        ]:
            if col in big.columns:
                big[col] = big[col].to_numpy() - dl

        if "dist_from_8d_low_adr" in big.columns and "undercut_10d_low_adr" in big.columns:
            big["dist_from_8d_low_adr"] = np.maximum(0.0, -big["undercut_10d_low_adr"].to_numpy())
        if "waterfall_accel" in big.columns:
            big["waterfall_accel"] = (big["drop_3d_adr"] / 3.0) / np.maximum(0.15, big["drop_10d_adr"] / 10.0)
        if "sma20_anchor" in big.columns and "std20_anchor" in big.columns and "bb_z_low" in big.columns:
            big["bb_z_low"] = ((p_cand - big["sma20_anchor"].to_numpy()) / np.maximum(1e-6, big["std20_anchor"].to_numpy())).clip(-6.0, 4.0)
        if "prev_bb_z_lo_min2" in big.columns and "bb_z_low_min3" in big.columns:
            big["bb_z_low_min3"] = np.minimum(big["prev_bb_z_lo_min2"].to_numpy(), big["bb_z_low"].to_numpy())
        if "roll_p95_down20" in big.columns and "rel_down_ema20_p95" in big.columns:
            big["rel_down_ema20_p95"] = big["stretch_below_ema20_adr"].to_numpy() / np.maximum(0.5, big["roll_p95_down20"].to_numpy())
        if "prev_stretch20_max2" in big.columns and "stretch_below_ema20_max3" in big.columns:
            big["stretch_below_ema20_max3"] = np.maximum(big["prev_stretch20_max2"].to_numpy(), big["stretch_below_ema20_adr"].to_numpy())
        if "prev_rsi14_lo_min4" in big.columns and "rsi14_low_min5" in big.columns:
            big["rsi14_low_min5"] = np.minimum(big["prev_rsi14_lo_min4"].to_numpy(), big["rsi14_low"].to_numpy())
        if "prev_rsi5_lo_min2" in big.columns and "rsi5_low_min3" in big.columns:
            big["rsi5_low_min3"] = np.minimum(big["prev_rsi5_lo_min2"].to_numpy(), big["rsi5_low"].to_numpy())
        if "rsi14_lo_shift3" in big.columns and "rsi14_low_diff3" in big.columns:
            big["rsi14_low_diff3"] = big["rsi14_low"].to_numpy() - big["rsi14_lo_shift3"].to_numpy()
        if "rsi14_lo_shift5" in big.columns and "rsi14_low_diff5" in big.columns:
            big["rsi14_low_diff5"] = big["rsi14_low"].to_numpy() - big["rsi14_lo_shift5"].to_numpy()
        if "prev_consec_rsi30_4d" in big.columns and "consec_rsi30_5d" in big.columns:
            big["consec_rsi30_5d"] = big["prev_consec_rsi30_4d"].to_numpy() + (big["rsi14_low"].to_numpy() <= 30.0).astype(float)
        if "prev_consec_rsi25_4d" in big.columns and "consec_rsi25_5d" in big.columns:
            big["consec_rsi25_5d"] = big["prev_consec_rsi25_4d"].to_numpy() + (big["rsi14_low"].to_numpy() <= 25.0).astype(float)
        if "prev_consec_rsi5_20_4d" in big.columns and "consec_rsi5_20_5d" in big.columns:
            big["consec_rsi5_20_5d"] = big["prev_consec_rsi5_20_4d"].to_numpy() + (big["rsi5_low"].to_numpy() <= 20.0).astype(float)
        if "prev_wave_low_18" in big.columns and "prev_wave_rsi_low_18" in big.columns:
            is_nl_18 = p_cand <= big["prev_wave_low_18"].to_numpy() * 1.015
            if "bullish_div_low_18d" in big.columns:
                big["bullish_div_low_18d"] = (is_nl_18 & (big["rsi14_low"].to_numpy() > big["prev_wave_rsi_low_18"].to_numpy() + 1.5)).astype(float)
            if "rsi_bull_div_delta_18d" in big.columns:
                big["rsi_bull_div_delta_18d"] = np.where(is_nl_18, np.maximum(0.0, big["rsi14_low"].to_numpy() - big["prev_wave_rsi_low_18"].to_numpy()), 0.0)
        if "prev_wave_low_35" in big.columns and "prev_wave_rsi_low_35" in big.columns:
            is_nl_35 = p_cand <= big["prev_wave_low_35"].to_numpy() * 1.015
            if "bullish_div_low_35d" in big.columns:
                big["bullish_div_low_35d"] = (is_nl_35 & (big["rsi14_low"].to_numpy() > big["prev_wave_rsi_low_35"].to_numpy() + 2.0)).astype(float)
            if "rsi_bull_div_delta_35d" in big.columns:
                big["rsi_bull_div_delta_35d"] = np.where(is_nl_35, np.maximum(0.0, big["rsi14_low"].to_numpy() - big["prev_wave_rsi_low_35"].to_numpy()), 0.0)
        for k_col, d_col in [
            ("kiss_ema10_dist_adr", "low_dist_ema10_adr"),
            ("kiss_ema20_dist_adr", "low_dist_ema20_adr"),
            ("kiss_sma50_dist_adr", "low_dist_sma50_adr"),
        ]:
            if k_col in big.columns and d_col in big.columns:
                big[k_col] = big[d_col].abs()
        if "min_support_kiss_adr" in big.columns:
            big["min_support_kiss_adr"] = np.minimum(big["kiss_ema20_dist_adr"], big["kiss_sma50_dist_adr"])
        if "pullback_retracement_ratio" in big.columns and "prior_rally_20d_adr" in big.columns:
            big["pullback_retracement_ratio"] = big["drop_15d_adr"].clip(lower=0.0) / np.maximum(1.5, big["prior_rally_20d_adr"])

        # Weekly W-FRI & MTF confluence updates at p_cand
        if "prev_w_c" in big.columns and "weekly_rsi14_low" in big.columns:
            wtd_lo_c = np.minimum(big["wtd_low_prev"].to_numpy(), p_cand)
            pw_c = big["prev_w_c"].to_numpy()
            gl_w = np.maximum(wtd_lo_c - pw_c, 0.0)
            ll_w = np.maximum(pw_c - wtd_lo_c, 0.0)
            agl_w = (big["prev_w_ag14"].to_numpy() * 13.0 + gl_w) / 14.0
            all_w = (big["prev_w_al14"].to_numpy() * 13.0 + ll_w) / 14.0
            big["weekly_rsi14_low"] = 100.0 - 100.0 / (1.0 + agl_w / (all_w + 1e-9))
            if "weekly_ext_ema10w_adr" in big.columns:
                big["weekly_ext_ema10w_adr"] = (wtd_lo_c - big["prev_w_ema10"].to_numpy()) / np.maximum(1e-6, big["prev_w_adr"].to_numpy())

        # Scale intrabar reclaim/wick features progressively with w (0 -> 1)
        big["range_to_adr"] = (hi_rep - p_cand) / adr_rep
        for scale_col in [
            "lower_wick_pct",
            "bounce_from_low_adr",
            "micro_bounce_adr",
            "micro_lower_wick_pct",
            "green_reclaim_bar",
            "two_bar_bottom_reclaim",
            "failed_low_undercut",
            "ma_undercut_reclaim",
            "day2_washout_reclaim",
        ]:
            if scale_col in big.columns:
                big[scale_col] = big[scale_col].to_numpy() * w
        if "close_pos" in big.columns:
            big["close_pos"] = w * big["close_pos"].to_numpy()
        if "hammer_vol_score" in big.columns:
            big["hammer_vol_score"] = big["rvol"] * big["bounce_from_low_adr"]
        if "support_confluence_score" in big.columns:
            big["support_confluence_score"] = big["bounce_from_low_adr"] / (0.25 + big["min_support_kiss_adr"])
        if "capitulation_vol_stretch" in big.columns:
            big["capitulation_vol_stretch"] = big["rvol"] * big["stretch_below_ema20_adr"].clip(lower=0.0)
        if "climax_washout_score" in big.columns:
            big["climax_washout_score"] = big["rvol"] * big["range_to_adr"] * big["stretch_below_ema20_adr"].clip(lower=0.0)
        if "seller_exhaustion_wick" in big.columns:
            big["seller_exhaustion_wick"] = big["bounce_from_low_adr"] * big["stretch_below_ema20_adr"].clip(lower=0.0)

    big[mask_col] = True
    p_dual, p_ex, _ = tap.score_dual_model_on_features(big, model_bundle, mask_col)
    p_dual_mat = p_dual.reshape(m, n_steps) * 100.0
    p_ex_mat = p_ex.reshape(m, n_steps) * 100.0
    p_cand_mat = p_cand.reshape(m, n_steps)

    for i, ev in enumerate(events):
        grid_i = p_cand_mat[i]
        prec = 4 if abs(float(grid_i[-1])) < 10.0 else 2
        ev_prob = float(ev["prob_pct"])
        # Evaluate the true model probability curve along the intrabar excursion (weighted by sqrt(w)
        # at the start of the bar so opposite-extreme w=0 cannot fire before price moves toward the pivot,
        # with a 5% within-leaf tie-breaker slope so flat tree steps interpolate distinctly across thresholds):
        model_curve = np.maximum.accumulate(np.maximum(p_dual_mat[i], p_ex_mat[i]) * np.sqrt(w_1d))
        model_curve = np.minimum(model_curve, ev_prob)
        model_curve[-1] = ev_prob
        pd_eff = 0.92 * model_curve + 0.08 * (w_1d * ev_prob)
        pd_eff[0] = 0.0
        pd_eff[-1] = ev_prob

        by_thr: Dict[str, float] = {}
        for thr in range(20, 86, 5):
            if ev_prob >= thr:
                trig_p = float(np.interp(float(thr), pd_eff, grid_i))
                by_thr[str(thr)] = round(trig_p, prec)
        default_thr = "50" if "50" in by_thr else (str(int(ev_prob // 5 * 5)) if by_thr else None)
        fallback_p = ev["high"] if is_high_pivot else ev["low"]
        ev["intrabar_trigger_price"] = by_thr.get(default_thr, fallback_p) if default_thr else fallback_p
        ev["trigger_prices_by_thr"] = by_thr
        if ev["at_extreme_pct"] >= 50.0:
            ex_prob = float(ev["at_extreme_pct"])
            ex_curve = np.maximum.accumulate(p_ex_mat[i] * np.sqrt(w_1d))
            ex_curve = np.minimum(ex_curve, ex_prob)
            ex_curve[-1] = ex_prob
            ex_eff = 0.92 * ex_curve + 0.08 * (w_1d * ex_prob)
            ev["touch_trigger_price"] = round(float(np.interp(50.0, ex_eff, grid_i)), prec)
        else:
            ev["touch_trigger_price"] = None

        cal_arr = tap.score_calibrated_probability(np.array([ev_prob / 100.0]), model_bundle)
        ev["calibrated_prob_pct"] = round(float(cal_arr[0]) * 100.0, 1)

    return events


def compute_universal_exhaustion(
    df_full: pd.DataFrame,
    df_view: pd.DataFrame,
    rs_val: Optional[float],
    ep_events: Optional[List[Dict[str, Any]]] = None,
    model_ver: str = "v3",
    symbol: Optional[str] = None,
) -> Dict[str, Any]:
    """Scores the 4-Model Universal Structural Swing-Pivot Suite across ANY ticker, ETF, Sector, Theme, or Ratio:
    1. `ut_high`: Uptrend Pivot High (Climax Top / Parabolic Stretch & Multi-Wave Divergence)
    2. `ut_low` : Uptrend Pivot Low (Pullback Buy / Support Kiss / Undercut & Reclaim)
    3. `dt_high`: Downtrend Pivot High (Bear-Rally Rejection / Overhead Falling-MA Supply)
    4. `dt_low` : Downtrend Pivot Low (Capitulation Washout / Selling Climax Bottom)
    Supports both `model_ver='v3'` (Baseline) and `model_ver='v4'` (Enhanced B1–B5 + Inverse-RSI).
    """
    ver = "v4" if str(model_ver or "v3").strip().lower().startswith("v4") else "v3"
    models = get_all_pivot_models(ver)
    if not models or df_full is None or len(df_full) < 20 or df_view is None or df_view.empty:
        return {}

    b = tap.compute_all_pivot_features(df_full, symbol=symbol)
    sub = _ensure_causal_rma_state(df_full, b.loc[df_view.index].copy())
    n = len(sub)
    zeros = np.zeros(n)

    # 1. Model 1: Uptrend Pivot High (ut_high)
    if "ut_high" in models:
        p_uh, p_uh_ex, p_uh_fb = tap.score_dual_model_on_features(sub, models["ut_high"], "is_near_local_high")
    else:
        p_uh = p_uh_ex = p_uh_fb = zeros

    # 2. Model 2: Uptrend Pivot Low (ut_low)
    if "ut_low" in models:
        p_ul, p_ul_ex, p_ul_fb = tap.score_dual_model_on_features(sub, models["ut_low"], "is_near_uptrend_low")
    else:
        p_ul = p_ul_ex = p_ul_fb = zeros

    # 3. Model 3: Downtrend Pivot High (dt_high)
    if "dt_high" in models:
        p_dh, p_dh_ex, p_dh_fb = tap.score_dual_model_on_features(sub, models["dt_high"], "is_near_downtrend_high")
    else:
        p_dh = p_dh_ex = p_dh_fb = zeros

    # 4. Model 4: Downtrend Pivot Low (dt_low)
    if "dt_low" in models:
        p_dl, p_dl_ex, p_dl_fb = tap.score_dual_model_on_features(sub, models["dt_low"], "is_near_downtrend_low")
    else:
        p_dl = p_dl_ex = p_dl_fb = zeros

    sub_cols = pd.DataFrame(
        {
            "exhaustion_prob": p_uh,
            "prob_at_high": p_uh_ex,
            "prob_full_bar": p_uh_fb,
            "ut_high_prob": p_uh,
            "ut_low_prob": p_ul,
            "dt_high_prob": p_dh,
            "dt_low_prob": p_dl,
        },
        index=sub.index,
    )
    sub = pd.concat([sub, sub_cols], axis=1)

    last_row = sub.iloc[-1]
    is_uptrend_now = bool(last_row.get("is_primary_uptrend", True))
    regime_str = "uptrend" if is_uptrend_now else "downtrend"

    curr_uh = round(float(last_row["ut_high_prob"]) * 100.0, 1)
    curr_ul = round(float(last_row["ut_low_prob"]) * 100.0, 1)
    curr_dh = round(float(last_row["dt_high_prob"]) * 100.0, 1)
    curr_dl = round(float(last_row["dt_low_prob"]) * 100.0, 1)

    peak5_uh = round(float(sub["ut_high_prob"].tail(5).max()) * 100.0, 1)
    peak5_ul = round(float(sub["ut_low_prob"].tail(5).max()) * 100.0, 1)
    peak5_dh = round(float(sub["dt_high_prob"].tail(5).max()) * 100.0, 1)
    peak5_dl = round(float(sub["dt_low_prob"].tail(5).max()) * 100.0, 1)

    curr_prob = curr_uh
    curr_at_high = round(float(last_row["prob_at_high"]) * 100.0, 1)
    curr_full_bar = round(float(last_row["prob_full_bar"]) * 100.0, 1)
    peak_5d_prob = peak5_uh
    peak_20d_prob = round(float(sub["ut_high_prob"].tail(20).max()) * 100.0, 1)

    # Determine dominant active 5D signal across all 4 models for the status banner
    max_5d_pair = max(
        [
            (peak5_uh, "ut_high", "🔥 Uptrend Pivot High Alert (Scale Out / Tighten Stop)"),
            (peak5_ul, "ut_low", "💎 Uptrend Pullback Pivot Low (Support Reclaim / Buyable Dip)"),
            (peak5_dh, "dt_high", "🔻 Downtrend Bear-Rally Pivot High (Overhead Supply Rejection)"),
            (peak5_dl, "dt_low", "🟢 Downtrend Capitulation Pivot Low (Selling Climax Washout)"),
        ],
        key=lambda x: x[0],
    )

    if max_5d_pair[0] >= 50.0:
        status = "climax" if max_5d_pair[1] in ("ut_high", "dt_high") else "pivot_low"
        action_label = max_5d_pair[2]
    elif max_5d_pair[0] >= 40.0:
        status = "elevated"
        action_label = f"⚠️ Developing {max_5d_pair[1].replace('_', ' ').upper()} Setup ({max_5d_pair[0]:.1f}% 5D Peak)"
    else:
        status = "healthy"
        action_label = (
            "✅ Healthy Uptrend Posture (No Extreme Pivot Trigger)"
            if is_uptrend_now
            else "📉 Downtrend Regime (No Active Reversal Pivot)"
        )

    ut_high_events = _enrich_events_with_intrabar_triggers(
        sub,
        _extract_snapped_pivot_events(sub, p_uh, p_uh_ex, p_uh_fb, sub["is_near_local_high"].to_numpy(), True, "ut_high"),
        models.get("ut_high"),
        "is_near_local_high",
        True,
    )
    ut_low_events = _enrich_events_with_intrabar_triggers(
        sub,
        _extract_snapped_pivot_events(sub, p_ul, p_ul_ex, p_ul_fb, sub["is_near_uptrend_low"].to_numpy(), False, "ut_low"),
        models.get("ut_low"),
        "is_near_uptrend_low",
        False,
    )
    dt_high_events = _enrich_events_with_intrabar_triggers(
        sub,
        _extract_snapped_pivot_events(sub, p_dh, p_dh_ex, p_dh_fb, sub["is_near_downtrend_high"].to_numpy(), True, "dt_high"),
        models.get("dt_high"),
        "is_near_downtrend_high",
        True,
    )
    dt_low_events = _enrich_events_with_intrabar_triggers(
        sub,
        _extract_snapped_pivot_events(sub, p_dl, p_dl_ex, p_dl_fb, sub["is_near_downtrend_low"].to_numpy(), False, "dt_low"),
        models.get("dt_low"),
        "is_near_downtrend_low",
        False,
    )

    series_data = [
        {
            "time": idx_dt.strftime("%Y-%m-%d"),
            "prob": round(float(r["ut_high_prob"]) * 100.0, 1),
            "at_high": round(float(r["prob_at_high"]) * 100.0, 1),
            "full_bar": round(float(r["prob_full_bar"]) * 100.0, 1),
            "ut_high": round(float(r["ut_high_prob"]) * 100.0, 1),
            "ut_low": round(float(r["ut_low_prob"]) * 100.0, 1),
            "dt_high": round(float(r["dt_high_prob"]) * 100.0, 1),
            "dt_low": round(float(r["dt_low_prob"]) * 100.0, 1),
            "regime": "UT" if bool(r["is_primary_uptrend"]) else "DT",
        }
        for idx_dt, r in sub[
            [
                "ut_high_prob",
                "prob_at_high",
                "prob_full_bar",
                "ut_low_prob",
                "dt_high_prob",
                "dt_low_prob",
                "is_primary_uptrend",
            ]
        ].iterrows()
    ]

    if is_uptrend_now:
        leg_bars = int(last_row.get("bars_since_ema20_touch", 0))
        leg_adr = round(float(last_row.get("leg_adr_from_ema20", 0.0)), 1)
        anchor_mode = f"Uptrend (+{leg_adr}x ADR above 20E / {leg_bars}d)"
    else:
        leg_bars = int(last_row.get("bars_below_ema20", 0))
        leg_adr = round(float(last_row.get("down_leg_adr_from_ema20", 0.0)), 1)
        anchor_mode = f"Downtrend (-{leg_adr}x ADR below 20E / {leg_bars}d)"

    has_bear_div = bool(last_row.get("bearish_div_high", 0) > 0.5 or last_row.get("bearish_div_35d", 0) > 0.5)
    has_bull_div = bool(last_row.get("bullish_div_low_18d", 0) > 0.5 or last_row.get("bullish_div_low_35d", 0) > 0.5)

    ob75_ser = indicators.rsi_trigger_price(df_full["close"], 75.0, 14)
    os25_ser = indicators.rsi_trigger_price(df_full["close"], 25.0, 14)
    last_idx = sub.index[-1]
    ob75_last = ob75_ser.loc[last_idx] if last_idx in ob75_ser.index else np.nan
    os25_last = os25_ser.loc[last_idx] if last_idx in os25_ser.index else np.nan

    return {
        "model_ver": ver,
        "regime": regime_str,
        "current_prob_pct": curr_prob,
        "at_high_prob_pct": curr_at_high,
        "full_bar_prob_pct": curr_full_bar,
        "peak_5d_prob_pct": peak_5d_prob,
        "peak_20d_prob_pct": peak_20d_prob,
        "pivots_current": {
            "ut_high": curr_uh,
            "ut_low": curr_ul,
            "dt_high": curr_dh,
            "dt_low": curr_dl,
        },
        "pivots_5d_peak": {
            "ut_high": peak5_uh,
            "ut_low": peak5_ul,
            "dt_high": peak5_dh,
            "dt_low": peak5_dl,
        },
        "status": status,
        "action_label": action_label,
        "anchor_mode": anchor_mode,
        "rsi14": round(float(last_row["rsi14_high"]), 1),
        "rsi14_low": round(float(last_row["rsi14_low"]), 1),
        "rsi14_close": round(float(last_row["rsi14"]), 1),
        "rsi5": round(float(last_row["rsi5_high"]), 1),
        "rsi5_low": round(float(last_row["rsi5_low"]), 1),
        "ob75_target_price": float(ob75_last) if pd.notna(ob75_last) else None,
        "os25_target_price": float(os25_last) if pd.notna(os25_last) else None,
        "ext_ema10_adr": round(float(last_row["ext_ema10_adr"]), 2),
        "ext_ema20_adr": round(float(last_row["ext_ema20_adr"]), 2),
        "ext_sma50_adr": round(float(last_row["ext_sma50_adr"]), 2),
        "low_dist_ema20_adr": round(float(last_row["low_dist_ema20_adr"]), 2),
        "upper_wick_pct": round(float(last_row["upper_wick_pct"]), 1),
        "lower_wick_pct": round(float(last_row["lower_wick_pct"]), 1),
        "rvol": round(float(last_row["rvol"]), 2),
        "consec_rsi_70": int(last_row["consec_rsi70_5d"]),
        "bearish_div": has_bear_div,
        "bullish_div": has_bull_div,
        "climax_events": ut_high_events,
        "ut_high_events": ut_high_events,
        "ut_low_events": ut_low_events,
        "dt_high_events": dt_high_events,
        "dt_low_events": dt_low_events,
        "series": series_data,
    }


def _normalize_split_seams(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes unadjusted split seams on historical OHLC series so charts and ratios are continuous."""
    if df is None or df.empty or len(df) < 10:
        return df
    c = df["close"].to_numpy(dtype=np.float64)
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    v = df["volume"].to_numpy(dtype=np.float64)
    n = len(c)
    v_ma10 = pd.Series(v).shift(1).rolling(10, min_periods=3).median().to_numpy()
    factor = 1.0
    factors = np.ones(n, dtype=np.float64)
    for i in range(n - 1, 0, -1):
        if c[i - 1] <= 0 or o[i] <= 0:
            factors[i - 1] = factor
            continue
        rp = o[i] / c[i - 1]
        rv = v[i] / v_ma10[i] if (not np.isnan(v_ma10[i]) and v_ma10[i] > 0) else 1.0
        if (rp >= 1.85 and rv <= 0.70 and (rp * rv) <= 3.5) or (rp <= 0.55 and rv >= 1.4 and (rp * rv) <= 3.5):
            factor *= rp
        factors[i - 1] = factor
    if abs(factor - 1.0) < 1e-6:
        return df
    out = df.copy()
    out["open"] = o * factors
    out["high"] = h * factors
    out["low"] = l * factors
    out["close"] = c * factors
    return out


def parse_asset_id(asset_id: str) -> Dict[str, str]:
    """Parses an identifier like 'NVDA', 'XBI', 'SECTOR:Biotechnology', 'THEME:Artificial Intelligence',
    or 'RATIO:XBI|SPY' into structured metadata."""
    raw = (asset_id or "").strip()
    if raw.upper().startswith("RATIO:"):
        body = raw[6:]
        parts = body.split("|", 1) if "|" in body else body.split("/", 1)
        num = parts[0].strip() if len(parts) > 0 else "SPY"
        den = parts[1].strip() if len(parts) > 1 else "SPY"
        return {
            "id": f"RATIO:{num}|{den}",
            "type": "ratio",
            "num": num,
            "den": den,
            "name": f"{format_asset_short_label(num)} ÷ {format_asset_short_label(den)}",
        }
    if raw.upper().startswith("SECTOR:"):
        sec_name = raw[7:].strip()
        return {
            "id": f"SECTOR:{sec_name}",
            "type": "sector",
            "key": sec_name,
            "name": f"{sec_name} (Sector Index)",
        }
    if raw.upper().startswith("THEME:"):
        thm_name = raw[6:].strip()
        return {
            "id": f"THEME:{thm_name}",
            "type": "theme",
            "key": thm_name,
            "name": f"{thm_name} (Theme Index)",
        }
    load_universe_metadata()
    sym = raw.upper()
    is_etf = sym in _ETF_SET or sym in SPDR_ETF_META
    etf_meta = SPDR_ETF_META.get(sym, {})
    name = etf_meta.get("name") or _NAMES_MAP.get(sym) or sym
    return {
        "id": sym,
        "type": "etf" if is_etf else "stock",
        "key": sym,
        "name": name,
        "category": etf_meta.get("category", "ETF" if is_etf else "Equity"),
    }


def format_asset_short_label(asset_id: str) -> str:
    info = parse_asset_id(asset_id)
    if info["type"] == "sector":
        return f"SEC:{info['key']}"
    if info["type"] == "theme":
        return f"THM:{info['key']}"
    if info["type"] == "ratio":
        return f"{format_asset_short_label(info['num'])}/{format_asset_short_label(info['den'])}"
    return info["key"]


def get_group_member_symbols(group_type: str, group_name: str) -> List[str]:
    """Returns all symbols belonging to a Sector or Theme (or linked SPDR ETF)."""
    labels.load()
    target = group_name.strip().lower()
    members = []
    if group_type == "sector":
        for sym, sec in (labels._SECTOR or {}).items():
            if sec and sec.lower() == target:
                members.append(sym)
    elif group_type == "theme":
        for sym, thms in (labels._THEMES_BY_SYM or {}).items():
            if any(t.lower() == target for t in thms):
                members.append(sym)
    return sorted(members)


def build_composite_bars(group_type: str, group_name: str, top_k: int = 25) -> Tuple[pd.DataFrame, List[str]]:
    """Constructs an equal-weighted synthetic daily OHLCV index (base 100) for a Sector or Theme
    from its top liquid constituent stocks."""
    cache_key = f"{group_type.upper()}:{group_name}"
    now = time.time()
    if cache_key in _COMPOSITE_BAR_CACHE:
        ts, cached_df, cached_members = _COMPOSITE_BAR_CACHE[cache_key]
        if now - ts < 1800 and not cached_df.empty:
            return cached_df, cached_members

    members = get_group_member_symbols(group_type, group_name)
    if not members:
        return pd.DataFrame(), []

    rs_df = get_rs_df()
    if not rs_df.empty and "dvol_m" in rs_df.columns:
        sub = rs_df[rs_df["symbol"].isin(members)].sort_values("dvol_m", ascending=False)
        ranked_syms = sub["symbol"].tolist()
        if len(ranked_syms) < min(len(members), top_k):
            rem = [s for s in members if s not in set(ranked_syms)]
            ranked_syms.extend(rem)
        selected_syms = ranked_syms[:top_k]
    else:
        selected_syms = members[:top_k]

    ret_c_list = []
    rel_o_list = []
    rel_h_list = []
    rel_l_list = []
    vol_list = []
    used_syms = []

    for sym in selected_syms:
        b = datastore.load_bars(sym)
        if b is None or b.empty or len(b) < 25:
            continue
        b = _normalize_split_seams(b)
        b_tail = b.tail(760)
        prev_c = b_tail["close"].shift(1)
        valid = prev_c > 0
        if valid.sum() < 20:
            continue
        rc = (b_tail["close"] / prev_c - 1.0).clip(-0.45, 0.85)
        ro = (b_tail["open"] / prev_c - 1.0).clip(-0.45, 0.85)
        rh = (b_tail["high"] / prev_c - 1.0).clip(-0.45, 0.95)
        rl = (b_tail["low"] / prev_c - 1.0).clip(-0.45, 0.85)
        dvol = (b_tail["close"] * b_tail["volume"]).fillna(0.0)

        ret_c_list.append(rc.rename(sym))
        rel_o_list.append(ro.rename(sym))
        rel_h_list.append(rh.rename(sym))
        rel_l_list.append(rl.rename(sym))
        vol_list.append(dvol.rename(sym))
        used_syms.append(sym)

    if not ret_c_list:
        return pd.DataFrame(), members

    df_rc = pd.concat(ret_c_list, axis=1, sort=True).dropna(how="all")
    min_constituents = max(1, min(3, len(used_syms) // 2))
    valid_days = df_rc.count(axis=1) >= min_constituents
    df_rc = df_rc.loc[valid_days]
    if df_rc.empty:
        return pd.DataFrame(), members

    df_ro = pd.concat(rel_o_list, axis=1, sort=True).reindex(df_rc.index)
    df_rh = pd.concat(rel_h_list, axis=1, sort=True).reindex(df_rc.index)
    df_rl = pd.concat(rel_l_list, axis=1, sort=True).reindex(df_rc.index)
    df_vol = pd.concat(vol_list, axis=1, sort=True).reindex(df_rc.index)

    mean_rc = df_rc.mean(axis=1).fillna(0.0)
    mean_ro = df_ro.mean(axis=1).fillna(0.0)
    mean_rh = df_rh.mean(axis=1).fillna(0.0)
    mean_rl = df_rl.mean(axis=1).fillna(0.0)
    sum_vol = df_vol.sum(axis=1).fillna(0.0)

    # Reconstruct synthetic index starting at 100.0
    mean_rc.iloc[0] = 0.0
    close_idx = 100.0 * (1.0 + mean_rc).cumprod()
    prev_close_idx = close_idx.shift(1).fillna(100.0)

    open_idx = prev_close_idx * (1.0 + mean_ro)
    open_idx.iloc[0] = 100.0
    raw_high = prev_close_idx * (1.0 + mean_rh)
    raw_low = prev_close_idx * (1.0 + mean_rl)

    high_idx = np.maximum(np.maximum(open_idx, close_idx), raw_high)
    low_idx = np.minimum(np.minimum(open_idx, close_idx), raw_low)

    comp_df = pd.DataFrame(
        {
            "open": open_idx,
            "high": high_idx,
            "low": low_idx,
            "close": close_idx,
            "volume": sum_vol,
        },
        index=df_rc.index,
    )
    _COMPOSITE_BAR_CACHE[cache_key] = (now, comp_df, members)
    return comp_df, members


def resolve_asset_bars(asset_id: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads OHLCV bars for any Stock, SPDR/Sector ETF, Sector composite, or Theme composite."""
    meta = parse_asset_id(asset_id)
    a_type = meta["type"]

    if a_type in ("stock", "etf"):
        sym = meta["key"]
        raw_df = datastore.load_bars(sym)
        if raw_df is None or raw_df.empty:
            return pd.DataFrame(), meta
        return raw_df.copy(), meta

    if a_type in ("sector", "theme"):
        comp_df, members = build_composite_bars(a_type, meta["key"])
        meta["member_count"] = len(members)
        meta["members"] = members
        return comp_df, meta

    return pd.DataFrame(), meta


def compute_ratio_bars(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """Computes aligned daily OHLCV ratio series (Part A ÷ Part B), scaled cleanly."""
    shared_idx = df_a.index.intersection(df_b.index)
    if len(shared_idx) < 5:
        return pd.DataFrame()
    a = df_a.loc[shared_idx]
    b = df_b.loc[shared_idx]

    valid = (b["close"] > 0) & (b["open"] > 0) & (b["high"] > 0) & (b["low"] > 0)
    a = a.loc[valid]
    b = b.loc[valid]
    if a.empty:
        return pd.DataFrame()

    r_open = a["open"] / b["open"]
    r_close = a["close"] / b["close"]
    # Estimate daily high/low excursion of the ratio bounded by constituent extremes
    r_ex1 = a["high"] / b["low"]
    r_ex2 = a["low"] / b["high"]
    # Dampen intraday cross-extremes slightly (50% blend with open/close envelope) so wicks are realistic
    oc_max = np.maximum(r_open, r_close)
    oc_min = np.minimum(r_open, r_close)
    r_high = np.maximum(oc_max, 0.5 * (oc_max + np.maximum(r_ex1, r_ex2)))
    r_low = np.minimum(oc_min, 0.5 * (oc_min + np.minimum(r_ex1, r_ex2)))

    # Scale factor if ratio is tiny (e.g. < 0.05) so LightweightCharts renders 2-decimal precision cleanly
    scale = 1.0
    median_c = float(r_close.median())
    if 0 < median_c < 0.05:
        scale = 1000.0
    elif 0.05 <= median_c < 0.5:
        scale = 100.0

    ratio_df = pd.DataFrame(
        {
            "open": r_open * scale,
            "high": r_high * scale,
            "low": r_low * scale,
            "close": r_close * scale,
            "volume": a["volume"].fillna(0.0),
            "raw_ratio_close": r_close,
            "scale_factor": scale,
        },
        index=a.index,
    )
    return ratio_df


def compute_series_performance(df: pd.DataFrame, asset_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Calculates comprehensive multi-timeframe performance, alpha vs SPY/QQQ, and technical posture
    for any OHLCV series (Stock, ETF, Sector, Theme, or Ratio)."""
    if df is None or df.empty:
        return {}

    c = df["close"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    v = df["volume"].to_numpy(dtype=np.float64)
    n = len(c)
    last_c = float(c[-1])
    last_date = df.index[-1].strftime("%Y-%m-%d")

    def _pct_ret(bars_back: int) -> Optional[float]:
        if n > bars_back and c[-1 - bars_back] > 0:
            return round(float((last_c / c[-1 - bars_back] - 1.0) * 100.0), 2)
        return None

    ret_1d = _pct_ret(1)
    ret_1w = _pct_ret(5)
    ret_1m = _pct_ret(21)
    ret_3m = _pct_ret(63)
    ret_6m = _pct_ret(126)
    ret_1y = _pct_ret(252)

    # YTD Return
    current_year = df.index[-1].year
    prev_year_bars = df[df.index.year < current_year]
    if not prev_year_bars.empty and prev_year_bars["close"].iloc[-1] > 0:
        ytd_base = float(prev_year_bars["close"].iloc[-1])
        ret_ytd = round(float((last_c / ytd_base - 1.0) * 100.0), 2)
    else:
        first_c = float(df[df.index.year == current_year]["close"].iloc[0]) if not df.empty else last_c
        ret_ytd = round(float((last_c / first_c - 1.0) * 100.0), 2) if first_c > 0 else 0.0

    # Benchmark comparison vs SPY and QQQ
    spy_bars = datastore.load_bars("SPY")
    qqq_bars = datastore.load_bars("QQQ")

    def _bench_ret(b_df: Optional[pd.DataFrame], bars_back: int) -> Optional[float]:
        if b_df is None or b_df.empty or len(b_df) <= bars_back:
            return None
        bc = b_df["close"].to_numpy(dtype=np.float64)
        if bc[-1 - bars_back] > 0:
            return round(float((bc[-1] / bc[-1 - bars_back] - 1.0) * 100.0), 2)
        return None

    spy_1m = _bench_ret(spy_bars, 21)
    spy_3m = _bench_ret(spy_bars, 63)
    spy_1y = _bench_ret(spy_bars, 252)
    qqq_1m = _bench_ret(qqq_bars, 21)
    qqq_3m = _bench_ret(qqq_bars, 63)

    alpha_spy_1m = round(ret_1m - spy_1m, 2) if (ret_1m is not None and spy_1m is not None) else None
    alpha_spy_3m = round(ret_3m - spy_3m, 2) if (ret_3m is not None and spy_3m is not None) else None
    alpha_spy_1y = round(ret_1y - spy_1y, 2) if (ret_1y is not None and spy_1y is not None) else None
    alpha_qqq_3m = round(ret_3m - qqq_3m, 2) if (ret_3m is not None and qqq_3m is not None) else None

    # Moving averages & technical posture
    s_close = df["close"]
    ema8 = float(s_close.ewm(span=8, adjust=False).mean().iloc[-1])
    ema10 = float(s_close.ewm(span=10, adjust=False).mean().iloc[-1])
    ema20 = float(s_close.ewm(span=20, adjust=False).mean().iloc[-1])
    ema21 = float(s_close.ewm(span=21, adjust=False).mean().iloc[-1])
    sma50 = float(s_close.rolling(50, min_periods=10).mean().iloc[-1])
    sma200 = float(s_close.rolling(200, min_periods=30).mean().iloc[-1])

    hi_52w = float(df["high"].tail(252).max())
    lo_52w = float(df["low"].tail(252).min())
    dist_52w_high = round((last_c / hi_52w - 1.0) * 100.0, 2) if hi_52w > 0 else 0.0
    dist_52w_low = round((last_c / lo_52w - 1.0) * 100.0, 2) if lo_52w > 0 else 0.0

    # ADR% (20-day Average Daily Range %)
    tail20 = df.tail(20)
    adr_pct = round(float(((tail20["high"] - tail20["low"]) / tail20["close"].replace(0, np.nan)).mean() * 100.0), 2)

    # Dollar Volume (20D Avg $M)
    if asset_meta.get("type") in ("sector", "theme"):
        # For composites, volume is already sum of constituent dollar volumes
        dvol_m = round(float(tail20["volume"].mean()) / 1e6, 1)
    else:
        dvol_m = round(float((tail20["close"] * tail20["volume"]).mean()) / 1e6, 2)

    above_10 = bool(last_c >= ema10)
    above_20 = bool(last_c >= ema20)
    above_50 = bool(last_c >= sma50)
    above_200 = bool(last_c >= sma200)
    stage2 = bool(last_c >= sma50 >= sma200 and last_c >= ema20)

    # Enrich with RS rank if in rs_universe_scores
    rs_score = None
    rs_rank = None
    sector_str = None
    themes_list: List[str] = []
    a_type = asset_meta.get("type", "stock")
    a_key = asset_meta.get("key", "")

    if a_type in ("stock", "etf"):
        labels.load()
        sector_str = labels.sector(a_key)
        themes_list = labels.themes(a_key) or []
        if a_key in SPDR_ETF_META:
            em = SPDR_ETF_META[a_key]
            sector_str = sector_str or em.get("linked_sector") or em.get("focus")
            if not themes_list and em.get("linked_theme"):
                themes_list = [em["linked_theme"]]

        rs_df = get_rs_df()
        if not rs_df.empty and "symbol" in rs_df.columns:
            m = rs_df[rs_df["symbol"] == a_key]
            if not m.empty:
                row0 = m.iloc[0]
                rs_score = _clean_val(row0.get("rs_score"))
                rs_rank = _clean_val(row0.get("rank"))
    elif a_type == "sector":
        sector_str = a_key
        eng = thematic_engine.get_thematic_engine()
        sec_scores = (eng.scores.get("sectors", {}) if eng else {})
        if a_key in sec_scores:
            s_inf = sec_scores[a_key]
            rs_score = round(float(_clean_val(s_inf.get("m3_pct"), 50.0)), 0)
    elif a_type == "theme":
        themes_list = [a_key]
        eng = thematic_engine.get_thematic_engine()
        thm_scores = (eng.scores.get("themes", {}) if eng else {})
        if a_key in thm_scores:
            t_inf = thm_scores[a_key]
            rs_score = round(float(_clean_val(t_inf.get("m3_pct"), 50.0)), 0)

    return {
        "id": asset_meta.get("id", a_key),
        "symbol": a_key or asset_meta.get("id", ""),
        "name": asset_meta.get("name", a_key),
        "type": a_type,
        "category": asset_meta.get("category", a_type.upper()),
        "last_date": last_date,
        "close": round(last_c, 4 if last_c < 1.0 else 2),
        "ret_1d": ret_1d,
        "ret_1w": ret_1w,
        "ret_1m": ret_1m,
        "ret_3m": ret_3m,
        "ret_6m": ret_6m,
        "ret_ytd": ret_ytd,
        "ret_1y": ret_1y,
        "alpha_spy_1m": alpha_spy_1m,
        "alpha_spy_3m": alpha_spy_3m,
        "alpha_spy_1y": alpha_spy_1y,
        "alpha_qqq_3m": alpha_qqq_3m,
        "rs_score": _clean_val(rs_score),
        "rs_rank": _clean_val(rs_rank),
        "adr_pct": _clean_val(adr_pct, 0.0),
        "dvol_m": _clean_val(dvol_m, 0.0),
        "hi_52w": round(hi_52w, 2),
        "lo_52w": round(lo_52w, 2),
        "dist_52w_high": _clean_val(dist_52w_high, 0.0),
        "dist_52w_low": _clean_val(dist_52w_low, 0.0),
        "ema8": round(ema8, 2),
        "ema10": round(ema10, 2),
        "ema20": round(ema20, 2),
        "ema21": round(ema21, 2),
        "sma50": round(sma50, 2),
        "sma200": round(sma200, 2),
        " dist_ema10_pct": round((last_c / ema10 - 1.0) * 100.0, 2) if ema10 > 0 else 0.0,
        "dist_ema21_pct": round((last_c / ema21 - 1.0) * 100.0, 2) if ema21 > 0 else 0.0,
        "dist_sma50_pct": round((last_c / sma50 - 1.0) * 100.0, 2) if sma50 > 0 else 0.0,
        "dist_sma200_pct": round((last_c / sma200 - 1.0) * 100.0, 2) if sma200 > 0 else 0.0,
        "above_10": above_10,
        "above_20": above_20,
        "above_50": above_50,
        "above_200": above_200,
        "stage2": stage2,
        "sector": sector_str or "Index / Composite",
        "theme": themes_list[0] if themes_list else "General",
        "themes": themes_list,
        "member_count": asset_meta.get("member_count"),
    }


def get_strong_ep_events_for_asset(asset_meta: Dict[str, Any], min_date: str = "2020-01-01") -> List[Dict[str, Any]]:
    """Retrieves recent and strong historical Episodic Pivot (EP) events from datastore for the chosen asset.
    - For a single Stock: returns all EP events for that stock, highlighting Strong / Tier 1 / Optimized Strategy setups.
    - For a Sector, Theme, or SPDR ETF: returns recent strong EP events across member stocks in that group.
    """
    ep_df = get_ep_df()
    if ep_df.empty:
        return []

    opt_df = get_opt_df()
    opt_lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not opt_df.empty and "symbol" in opt_df.columns and "ep_date" in opt_df.columns:
        for _, r in opt_df.iterrows():
            opt_lookup[(str(r["symbol"]), str(r["ep_date"]))] = {
                "is_tier1": bool(r.get("is_tier1", False)),
                "is_inst_core": bool(r.get("is_inst_core", False)),
                "net_r": _clean_val(r.get("net_r")),
                "ret_pct": _clean_val(r.get("ret_pct")),
                "entry_status": str(r.get("entry_status", "")),
            }

    a_type = asset_meta.get("type", "stock")
    a_key = asset_meta.get("key", "")

    if a_type == "stock":
        matches = ep_df[(ep_df["symbol"] == a_key) & (ep_df["date"] >= min_date)].sort_values("date")
        events: List[Dict[str, Any]] = []
        for _, row in matches.iterrows():
            dt = str(row["date"])
            gap = round(float(_clean_val(row.get("gap_pct"), 0.0)), 1)
            rvol = round(float(_clean_val(row.get("rvol"), 0.0)), 1)
            cpos = round(float(_clean_val(row.get("close_pos"), 0.5)), 2)
            v2_win = _clean_val(row.get("v2_prob_win"))
            v2_p50 = _clean_val(row.get("v2_prob_50"))
            opt_info = opt_lookup.get((a_key, dt), {})
            is_strong = bool(
                (gap >= 10.0 and rvol >= 2.0 and cpos >= 0.50)
                or (rvol >= 3.0 and gap >= 8.0)
                or opt_info.get("is_inst_core")
                or opt_info.get("is_tier1")
            )
            tier_badge = "Tier 1 Ultra" if opt_info.get("is_tier1") else (
                "Institutional Core" if opt_info.get("is_inst_core") else (
                    "Strong EP" if is_strong else "Standard EP"
                )
            )
            events.append({
                "symbol": a_key,
                "date": dt,
                "gap_pct": gap,
                "rvol": rvol,
                "close_pos": cpos,
                "subtype": str(row.get("subtype", "classic")),
                "is_strong": is_strong,
                "tier_badge": tier_badge,
                "v2_prob_win": round(v2_win * 100.0, 1) if v2_win is not None else None,
                "v2_prob_50": round(v2_p50 * 100.0, 1) if v2_p50 is not None else None,
                "max_gain_20d": round(float(_clean_val(row.get("max_gain_20d"), 0.0)), 1) if _clean_val(row.get("max_gain_20d")) is not None else None,
                "net_r": round(float(opt_info["net_r"]), 2) if opt_info.get("net_r") is not None else None,
                "is_group_event": False,
            })
        return events

    # For Sector, Theme, or SPDR ETF with linked sector/theme: find strong constituent EP ignitions
    member_syms: List[str] = []
    if a_type in ("sector", "theme"):
        member_syms = asset_meta.get("members") or get_group_member_symbols(a_type, a_key)
    elif a_type == "etf" and a_key in SPDR_ETF_META:
        em = SPDR_ETF_META[a_key]
        if em.get("linked_sector"):
            member_syms = get_group_member_symbols("sector", em["linked_sector"])
        elif em.get("linked_theme"):
            member_syms = get_group_member_symbols("theme", em["linked_theme"])

    if not member_syms:
        # Check if the ETF itself had any gap events
        return []

    sub = ep_df[(ep_df["symbol"].isin(member_syms)) & (ep_df["date"] >= min_date)].copy()
    if sub.empty:
        return []

    # Filter to strong constituent EPs (Gap >= 12% and RVOL >= 2.5x and ClosePos >= 0.60)
    strong_mask = (sub["gap_pct"] >= 12.0) & (sub["rvol"] >= 2.5) & (sub["close_pos"] >= 0.60)
    sub_strong = sub[strong_mask].sort_values(["date", "gap_pct"], ascending=[True, False]).tail(35)

    events = []
    for _, row in sub_strong.iterrows():
        sym = str(row["symbol"])
        dt = str(row["date"])
        gap = round(float(_clean_val(row.get("gap_pct"), 0.0)), 1)
        rvol = round(float(_clean_val(row.get("rvol"), 0.0)), 1)
        cpos = round(float(_clean_val(row.get("close_pos"), 0.5)), 2)
        v2_p50 = _clean_val(row.get("v2_prob_50"))
        opt_info = opt_lookup.get((sym, dt), {})
        tier_badge = "Tier 1 Ultra" if opt_info.get("is_tier1") else (
            "Institutional Core" if opt_info.get("is_inst_core") else "Strong Group EP"
        )
        events.append({
            "symbol": sym,
            "date": dt,
            "gap_pct": gap,
            "rvol": rvol,
            "close_pos": cpos,
            "subtype": str(row.get("subtype", "classic")),
            "is_strong": True,
            "tier_badge": tier_badge,
            "v2_prob_50": round(v2_p50 * 100.0, 1) if v2_p50 is not None else None,
            "max_gain_20d": round(float(_clean_val(row.get("max_gain_20d"), 0.0)), 1) if _clean_val(row.get("max_gain_20d")) is not None else None,
            "is_group_event": True,
        })
    return events


def get_constituents_and_context(asset_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Returns constituent leaders and thematic/sector leadership context for the Info Panel."""
    labels.load()
    load_universe_metadata()
    rs_df = get_rs_df()
    eng = thematic_engine.get_thematic_engine()

    a_type = asset_meta.get("type", "stock")
    a_key = asset_meta.get("key", "")

    sec_name = None
    thm_name = None

    if a_type == "sector":
        sec_name = a_key
    elif a_type == "theme":
        thm_name = a_key
    elif a_type in ("stock", "etf"):
        sec_name = labels.sector(a_key)
        thms = labels.themes(a_key) or []
        thm_name = thms[0] if thms else None
        if a_key in SPDR_ETF_META:
            em = SPDR_ETF_META[a_key]
            sec_name = sec_name or em.get("linked_sector")
            thm_name = thm_name or em.get("linked_theme")

    sec_scores = eng.scores.get("sectors", {}) if eng else {}
    thm_scores = eng.scores.get("themes", {}) if eng else {}

    # Sector metrics
    sector_summary = None
    if sec_name and sec_name in sec_scores:
        s_row = sec_scores[sec_name]
        sector_summary = {
            "id": f"SECTOR:{sec_name}",
            "name": sec_name,
            "score": round(float(_clean_val(s_row.get("score"), 50.0)), 1),
            "archetype": str(s_row.get("archetype", "Neutral")),
            "w1_ret": round(float(_clean_val(s_row.get("w1_ret"), 0.0)), 2) if _clean_val(s_row.get("w1_ret")) is not None else None,
            "m1_ret": round(float(_clean_val(s_row.get("m1_ret"), 0.0)), 2) if _clean_val(s_row.get("m1_ret")) is not None else None,
            "m3_ret": round(float(_clean_val(s_row.get("m3_ret"), 0.0)), 2) if _clean_val(s_row.get("m3_ret")) is not None else None,
            "ytd_ret": round(float(_clean_val(s_row.get("ytd_ret"), 0.0)), 2) if _clean_val(s_row.get("ytd_ret")) is not None else None,
            "m1_pctile": round(float(_clean_val(s_row.get("m1_pct"), 50.0)), 0),
            "m3_pctile": round(float(_clean_val(s_row.get("m3_pct"), 50.0)), 0),
            "member_count": int(_clean_val(s_row.get("member_count"), 0)),
        }

    # Theme metrics
    theme_summary = None
    if thm_name and thm_name in thm_scores:
        t_row = thm_scores[thm_name]
        theme_summary = {
            "id": f"THEME:{thm_name}",
            "name": thm_name,
            "score": round(float(_clean_val(t_row.get("score"), 50.0)), 1),
            "archetype": str(t_row.get("archetype", "Neutral")),
            "w1_ret": round(float(_clean_val(t_row.get("w1_ret"), 0.0)), 2) if _clean_val(t_row.get("w1_ret")) is not None else None,
            "m1_ret": round(float(_clean_val(t_row.get("m1_ret"), 0.0)), 2) if _clean_val(t_row.get("m1_ret")) is not None else None,
            "m3_ret": round(float(_clean_val(t_row.get("m3_ret"), 0.0)), 2) if _clean_val(t_row.get("m3_ret")) is not None else None,
            "ytd_ret": round(float(_clean_val(t_row.get("ytd_ret"), 0.0)), 2) if _clean_val(t_row.get("ytd_ret")) is not None else None,
            "m1_pctile": round(float(_clean_val(t_row.get("m1_pct"), 50.0)), 0),
            "m3_pctile": round(float(_clean_val(t_row.get("m3_pct"), 50.0)), 0),
            "member_count": int(_clean_val(t_row.get("member_count"), 0)),
        }

    # Top Constituents / Peers
    constituents: List[Dict[str, Any]] = []
    if not rs_df.empty:
        target_members: List[str] = []
        if a_type in ("sector", "theme"):
            target_members = asset_meta.get("members") or get_group_member_symbols(a_type, a_key)
        elif sec_name:
            target_members = get_group_member_symbols("sector", sec_name)
        elif thm_name:
            target_members = get_group_member_symbols("theme", thm_name)

        if target_members:
            sub = rs_df[rs_df["symbol"].isin(target_members)].sort_values("rs_score", ascending=False).head(15)
            for _, r in sub.iterrows():
                s = str(r["symbol"])
                constituents.append({
                    "symbol": s,
                    "name": _NAMES_MAP.get(s, str(r.get("name", s))),
                    "rs_score": int(_clean_val(r.get("rs_score"), 50)),
                    "close": round(float(_clean_val(r.get("close"), 0.0)), 2),
                    "ret_1d": round(float(_clean_val(r.get("ret_1d"), 0.0)), 2),
                    "ret_1m": round(float(_clean_val(r.get("ret_1m"), 0.0)), 2),
                    "ret_3m": round(float(_clean_val(r.get("ret_3m"), 0.0)), 2),
                })

    return {
        "sector_summary": sector_summary,
        "theme_summary": theme_summary,
        "constituents": constituents,
    }


# ==============================================================================
# Watchlist Persistence & Defaults
# ==============================================================================

def get_default_watchlists() -> Dict[str, Any]:
    """Builds pre-seeded institutional watchlists on first launch."""
    # Pull top recent Tier 1 / Institutional Core EP stocks if available
    tier1_items = []
    opt_df = get_opt_df()
    load_universe_metadata()
    if not opt_df.empty and "is_tier1" in opt_df.columns:
        recent_t1 = opt_df[opt_df["is_tier1"] == True].sort_values("ep_date", ascending=False).head(12)
        seen = set()
        for _, r in recent_t1.iterrows():
            sym = str(r["symbol"])
            if sym not in seen:
                seen.add(sym)
                tier1_items.append({
                    "id": sym,
                    "type": "stock",
                    "name": _NAMES_MAP.get(sym, sym),
                    "flag": "green",
                    "notes": f"Tier 1 EP on {r.get('ep_date')} (+{round(float(_clean_val(r.get('gap_pct'), 0.0)), 1)}% Gap, {round(float(_clean_val(r.get('rvol'), 0.0)), 1)}x RVOL)",
                })

    if not tier1_items:
        for sym in ["NVDA", "PLTR", "APP", "RDDT", "ALAB", "SMCI", "ARM", "CRWD"]:
            tier1_items.append({
                "id": sym,
                "type": "stock",
                "name": _NAMES_MAP.get(sym, sym),
                "flag": "green",
                "notes": "High RS Institutional Leader",
            })

    spdr_items = [
        {"id": "SPY", "type": "etf", "name": "SPDR S&P 500 ETF Trust", "flag": "blue", "notes": "S&P 500 Benchmark"},
        {"id": "QQQ", "type": "etf", "name": "Invesco QQQ Trust", "flag": "blue", "notes": "Nasdaq 100 Benchmark"},
        {"id": "IWM", "type": "etf", "name": "iShares Russell 2000 ETF", "flag": "blue", "notes": "Small Cap Risk Appetite"},
        {"id": "XBI", "type": "etf", "name": "SPDR S&P Biotech ETF", "flag": "green", "notes": "Equal-Weight Biotech Bellwether"},
        {"id": "SMH", "type": "etf", "name": "VanEck Semiconductor ETF", "flag": "green", "notes": "Semiconductor Leadership"},
        {"id": "XLK", "type": "etf", "name": "Technology Select Sector SPDR Fund", "flag": "amber", "notes": "SPDR Technology"},
        {"id": "IGV", "type": "etf", "name": "iShares Expanded Tech-Software Sector ETF", "flag": "amber", "notes": "Enterprise Software"},
        {"id": "XLV", "type": "etf", "name": "Health Care Select Sector SPDR Fund", "flag": "none", "notes": "SPDR Healthcare"},
        {"id": "XLF", "type": "etf", "name": "Financial Select Sector SPDR Fund", "flag": "none", "notes": "SPDR Financials"},
        {"id": "KRE", "type": "etf", "name": "SPDR S&P Regional Banking ETF", "flag": "none", "notes": "SPDR Regional Banks"},
        {"id": "XLE", "type": "etf", "name": "Energy Select Sector SPDR Fund", "flag": "none", "notes": "SPDR Energy"},
        {"id": "XLI", "type": "etf", "name": "Industrial Select Sector SPDR Fund", "flag": "none", "notes": "SPDR Industrials"},
        {"id": "XAR", "type": "etf", "name": "SPDR S&P Aerospace & Defense ETF", "flag": "none", "notes": "SPDR Defense"},
        {"id": "XME", "type": "etf", "name": "SPDR S&P Metals & Mining ETF", "flag": "none", "notes": "SPDR Metals & Mining"},
        {"id": "XHB", "type": "etf", "name": "SPDR S&P Homebuilders ETF", "flag": "none", "notes": "SPDR Homebuilders"},
        {"id": "URA", "type": "etf", "name": "Global X Uranium ETF", "flag": "purple", "notes": "Nuclear & Uranium"},
        {"id": "GDX", "type": "etf", "name": "VanEck Gold Miners ETF", "flag": "amber", "notes": "Gold Miners"},
    ]

    theme_sector_items = [
        {"id": "SECTOR:Biotechnology", "type": "sector", "name": "Biotechnology (Sector Index)", "flag": "green", "notes": "Equal-weight Biotech composite"},
        {"id": "SECTOR:Semiconductors", "type": "sector", "name": "Semiconductors (Sector Index)", "flag": "green", "notes": "Equal-weight Semis composite"},
        {"id": "SECTOR:Aerospace & Defense", "type": "sector", "name": "Aerospace & Defense (Sector Index)", "flag": "amber", "notes": "Defense & Space composite"},
        {"id": "SECTOR:Software - Infrastructure", "type": "sector", "name": "Software - Infrastructure (Sector Index)", "flag": "blue", "notes": "Infra Software composite"},
        {"id": "THEME:Artificial Intelligence", "type": "theme", "name": "Artificial Intelligence (Theme Index)", "flag": "green", "notes": "AI Theme Equal-Weight Index"},
        {"id": "THEME:Cybersecurity", "type": "theme", "name": "Cybersecurity (Theme Index)", "flag": "blue", "notes": "Cybersecurity Theme Index"},
        {"id": "THEME:Uranium & Nuclear", "type": "theme", "name": "Uranium & Nuclear (Theme Index)", "flag": "purple", "notes": "Nuclear Power & SMRs"},
        {"id": "THEME:Cloud Computing", "type": "theme", "name": "Cloud Computing (Theme Index)", "flag": "none", "notes": "Cloud SaaS & Infra"},
        {"id": "THEME:Defense & Aerospace", "type": "theme", "name": "Defense & Aerospace (Theme Index)", "flag": "none", "notes": "Defense Tech"},
    ]

    ratio_items = [
        {"id": "RATIO:XBI|SPY", "type": "ratio", "name": "XBI ÷ SPY", "flag": "green", "notes": "Biotech Relative Strength vs S&P 500"},
        {"id": "RATIO:SMH|QQQ", "type": "ratio", "name": "SMH ÷ QQQ", "flag": "green", "notes": "Semiconductors vs Nasdaq 100"},
        {"id": "RATIO:IWM|SPY", "type": "ratio", "name": "IWM ÷ SPY", "flag": "blue", "notes": "Small Caps vs Large Caps Breadth"},
        {"id": "RATIO:RSP|SPY", "type": "ratio", "name": "RSP ÷ SPY", "flag": "blue", "notes": "Equal-Weight vs Cap-Weight S&P 500"},
        {"id": "RATIO:SECTOR:Biotechnology|XBI", "type": "ratio", "name": "SEC:Biotechnology ÷ XBI", "flag": "purple", "notes": "Our Biotech Universe vs SPDR XBI"},
        {"id": "RATIO:THEME:Artificial Intelligence|QQQ", "type": "ratio", "name": "THM:Artificial Intelligence ÷ QQQ", "flag": "amber", "notes": "AI Theme Index vs QQQ"},
        {"id": "RATIO:XLY|XLP", "type": "ratio", "name": "XLY ÷ XLP", "flag": "none", "notes": "Discretionary vs Staples Risk-On Gauge"},
    ]

    return {
        "active_group_id": "wl_spdr",
        "groups": [
            {"id": "wl_tier1", "name": "🎯 Tier 1 EP Focus", "items": tier1_items},
            {"id": "wl_spdr", "name": "🏛️ SPDR & Sector ETFs", "items": spdr_items},
            {"id": "wl_themes", "name": "🔥 Leading Sectors & Themes", "items": theme_sector_items},
            {"id": "wl_ratios", "name": "⚖️ Key Ratio Charts", "items": ratio_items},
        ],
    }


def load_watchlists() -> Dict[str, Any]:
    if WATCHLISTS_FILE.exists():
        try:
            with open(WATCHLISTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "groups" in data and isinstance(data["groups"], list):
                return data
        except Exception as e:
            print(f"Warning: could not load {WATCHLISTS_FILE}: {e}")
    default_wl = get_default_watchlists()
    save_watchlists(default_wl)
    return default_wl


def save_watchlists(data: Dict[str, Any]) -> None:
    WATCHLISTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(WATCHLISTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def enrich_watchlist_items(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enriches watchlist items with live 1D/1M/3M return & RS score badges for fast sidebar rendering."""
    rs_df = get_rs_df()
    rs_lookup: Dict[str, Dict[str, Any]] = {}
    if not rs_df.empty and "symbol" in rs_df.columns:
        for _, r in rs_df.iterrows():
            rs_lookup[str(r["symbol"])] = {
                "close": round(float(_clean_val(r.get("close"), 0.0)), 2),
                "ret_1d": round(float(_clean_val(r.get("ret_1d"), 0.0)), 2),
                "ret_1m": round(float(_clean_val(r.get("ret_1m"), 0.0)), 2),
                "ret_3m": round(float(_clean_val(r.get("ret_3m"), 0.0)), 2),
                "rs_score": int(_clean_val(r.get("rs_score"), 50)),
            }

    eng = thematic_engine.get_thematic_engine()
    sec_scores = eng.scores.get("sectors", {}) if eng else {}
    thm_scores = eng.scores.get("themes", {}) if eng else {}
    enriched_groups = []
    for grp in groups:
        new_items = []
        for item in grp.get("items", []):
            item_copy = dict(item)
            parsed = parse_asset_id(item_copy.get("id", ""))
            item_copy["type"] = parsed["type"]
            item_copy["name"] = item_copy.get("name") or parsed["name"]
            item_copy["short_label"] = format_asset_short_label(parsed["id"])

            if parsed["type"] in ("stock", "etf"):
                sym = parsed["key"]
                if sym in rs_lookup:
                    item_copy.update(rs_lookup[sym])
                else:
                    b = datastore.load_bars(sym)
                    if b is not None and len(b) >= 2:
                        c = b["close"].to_numpy(dtype=np.float64)
                        item_copy["close"] = round(float(c[-1]), 2)
                        item_copy["ret_1d"] = round(float((c[-1] / c[-2] - 1.0) * 100.0), 2)
                        item_copy["ret_1m"] = round(float((c[-1] / c[-min(len(c), 22)] - 1.0) * 100.0), 2)
            elif parsed["type"] == "sector" and parsed["key"] in sec_scores:
                sr = sec_scores[parsed["key"]]
                item_copy["ret_1d"] = round(float(_clean_val(sr.get("w1_ret"), 0.0)) / 5.0, 2)
                item_copy["ret_1m"] = round(float(_clean_val(sr.get("m1_ret"), 0.0)), 2)
                item_copy["ret_3m"] = round(float(_clean_val(sr.get("m3_ret"), 0.0)), 2)
                item_copy["rs_score"] = int(round(float(_clean_val(sr.get("m3_pct"), 50.0))))
            elif parsed["type"] == "theme" and parsed["key"] in thm_scores:
                tr = thm_scores[parsed["key"]]
                item_copy["ret_1d"] = round(float(_clean_val(tr.get("w1_ret"), 0.0)) / 5.0, 2)
                item_copy["ret_1m"] = round(float(_clean_val(tr.get("m1_ret"), 0.0)), 2)
                item_copy["ret_3m"] = round(float(_clean_val(tr.get("m3_ret"), 0.0)), 2)
                item_copy["rs_score"] = int(round(float(_clean_val(tr.get("m3_pct"), 50.0))))
            new_items.append(item_copy)
        enriched_groups.append({
            "id": grp.get("id"),
            "name": grp.get("name"),
            "items": new_items,
        })
    return enriched_groups


# ==============================================================================
# Catalog Builder (Stocks, SPDR/Sector ETFs, Sectors, Themes)
# ==============================================================================

def build_catalog() -> Dict[str, Any]:
    global _CATALOG_CACHE
    now = time.time()
    if _CATALOG_CACHE is not None and (now - _CATALOG_CACHE[0] < 600):
        return _CATALOG_CACHE[1]

    labels.load()
    load_universe_metadata()
    rs_df = get_rs_df()
    eng = thematic_engine.get_thematic_engine()
    sec_scores = eng.scores.get("sectors", {}) if eng else {}
    thm_scores = eng.scores.get("themes", {}) if eng else {}

    rs_map: Dict[str, Dict[str, Any]] = {}
    if not rs_df.empty and "symbol" in rs_df.columns:
        for _, r in rs_df.iterrows():
            s = str(r["symbol"])
            rs_map[s] = {
                "symbol": s,
                "name": _NAMES_MAP.get(s, str(r.get("name", s))),
                "rs_score": int(_clean_val(r.get("rs_score"), 50)),
                "close": round(float(_clean_val(r.get("close"), 0.0)), 2),
                "ret_1d": round(float(_clean_val(r.get("ret_1d"), 0.0)), 2),
                "ret_1w": round(float(_clean_val(r.get("ret_1w"), 0.0)), 2),
                "ret_1m": round(float(_clean_val(r.get("ret_1m"), 0.0)), 2),
                "ret_3m": round(float(_clean_val(r.get("ret_3m"), 0.0)), 2),
                "ret_1y": round(float(_clean_val(r.get("ret_1y"), 0.0)), 2),
                "dvol_m": round(float(_clean_val(r.get("dvol_m"), 0.0)), 1),
                "sector": str(r.get("sector") or labels.sector(s) or "Unknown"),
                "theme": str(r.get("theme") or (labels.themes(s) or ["General"])[0]),
            }

    # 1. SPDR & Sector/Industry ETFs
    etfs_list: List[Dict[str, Any]] = []
    all_etf_syms = list(SPDR_ETF_META.keys()) + sorted([s for s in _ETF_SET if s not in SPDR_ETF_META])
    for sym in all_etf_syms:
        meta = SPDR_ETF_META.get(sym, {})
        base = rs_map.get(sym, {})
        if not base:
            b = datastore.load_bars(sym)
            if b is None or b.empty:
                continue
            c = b["close"].to_numpy(dtype=np.float64)
            base = {
                "symbol": sym,
                "name": meta.get("name") or _NAMES_MAP.get(sym, sym),
                "rs_score": 50,
                "close": round(float(c[-1]), 2),
                "ret_1d": round(float((c[-1] / c[-2] - 1.0) * 100.0), 2) if len(c) >= 2 else 0.0,
                "ret_1m": round(float((c[-1] / c[-min(len(c), 22)] - 1.0) * 100.0), 2) if len(c) >= 5 else 0.0,
                "ret_3m": round(float((c[-1] / c[-min(len(c), 64)] - 1.0) * 100.0), 2) if len(c) >= 10 else 0.0,
            }
        etfs_list.append({
            "id": sym,
            "symbol": sym,
            "type": "etf",
            "name": meta.get("name") or base.get("name") or sym,
            "category": meta.get("category", "Sector / Thematic ETF"),
            "focus": meta.get("focus", base.get("sector", "ETF")),
            "close": base.get("close", 0.0),
            "ret_1d": base.get("ret_1d", 0.0),
            "ret_1m": base.get("ret_1m", 0.0),
            "ret_3m": base.get("ret_3m", 0.0),
            "rs_score": base.get("rs_score", 50),
        })

    # 2. Sectors (154)
    sectors_list: List[Dict[str, Any]] = []
    sec_counts: Dict[str, int] = {}
    for _, sec in (labels._SECTOR or {}).items():
        if sec:
            sec_counts[sec] = sec_counts.get(sec, 0) + 1

    all_sector_names = sorted(sec_counts.keys())
    for sec_name in all_sector_names:
        s_info = sec_scores.get(sec_name, {})
        score = round(float(_clean_val(s_info.get("score"), 50.0)), 1)
        m1 = round(float(_clean_val(s_info.get("m1_ret"), 0.0)), 2) if _clean_val(s_info.get("m1_ret")) is not None else 0.0
        m3 = round(float(_clean_val(s_info.get("m3_ret"), 0.0)), 2) if _clean_val(s_info.get("m3_ret")) is not None else 0.0
        w1 = round(float(_clean_val(s_info.get("w1_ret"), 0.0)), 2) if _clean_val(s_info.get("w1_ret")) is not None else 0.0
        sectors_list.append({
            "id": f"SECTOR:{sec_name}",
            "symbol": f"SECTOR:{sec_name}",
            "type": "sector",
            "name": sec_name,
            "score": score,
            "rs_score": int(round(float(_clean_val(s_info.get("m3_pct"), 50.0)))),
            "archetype": str(s_info.get("archetype", "Sector")),
            "ret_1w": w1,
            "ret_1m": m1,
            "ret_3m": m3,
            "member_count": int(_clean_val(s_info.get("member_count"), sec_counts.get(sec_name, 0))),
        })
    sectors_list.sort(key=lambda x: x["score"], reverse=True)

    # 3. Themes (187)
    themes_list: List[Dict[str, Any]] = []
    thm_counts: Dict[str, int] = {}
    for _, thms in (labels._THEMES_BY_SYM or {}).items():
        for t in thms:
            thm_counts[t] = thm_counts.get(t, 0) + 1

    for thm_name in labels.all_themes():
        t_info = thm_scores.get(thm_name, {})
        score = round(float(_clean_val(t_info.get("score"), 50.0)), 1)
        m1 = round(float(_clean_val(t_info.get("m1_ret"), 0.0)), 2) if _clean_val(t_info.get("m1_ret")) is not None else 0.0
        m3 = round(float(_clean_val(t_info.get("m3_ret"), 0.0)), 2) if _clean_val(t_info.get("m3_ret")) is not None else 0.0
        w1 = round(float(_clean_val(t_info.get("w1_ret"), 0.0)), 2) if _clean_val(t_info.get("w1_ret")) is not None else 0.0
        themes_list.append({
            "id": f"THEME:{thm_name}",
            "symbol": f"THEME:{thm_name}",
            "type": "theme",
            "name": thm_name,
            "score": score,
            "rs_score": int(round(float(_clean_val(t_info.get("m3_pct"), 50.0)))),
            "archetype": str(t_info.get("archetype", "Theme")),
            "ret_1w": w1,
            "ret_1m": m1,
            "ret_3m": m3,
            "member_count": int(_clean_val(t_info.get("member_count"), thm_counts.get(thm_name, 0))),
        })
    themes_list.sort(key=lambda x: x["score"], reverse=True)

    # 4. Stocks (Sorted by RS Score desc)
    stocks_list: List[Dict[str, Any]] = []
    for sym, info in rs_map.items():
        if sym in _ETF_SET:
            continue
        stocks_list.append({
            "id": sym,
            "symbol": sym,
            "type": "stock",
            "name": info["name"],
            "rs_score": info["rs_score"],
            "close": info["close"],
            "ret_1d": info["ret_1d"],
            "ret_1m": info["ret_1m"],
            "ret_3m": info["ret_3m"],
            "dvol_m": info["dvol_m"],
            "sector": info["sector"],
            "theme": info["theme"],
        })
    stocks_list.sort(key=lambda x: (x["rs_score"], x["ret_3m"]), reverse=True)

    payload = {
        "etfs": etfs_list,
        "sectors": sectors_list,
        "themes": themes_list,
        "stocks": stocks_list,
        "counts": {
            "stocks": len(stocks_list),
            "etfs": len(etfs_list),
            "sectors": len(sectors_list),
            "themes": len(themes_list),
        },
    }
    _CATALOG_CACHE = (now, payload)
    return payload


# ==============================================================================
# API Endpoints
# ==============================================================================

@app.route("/api/reload_dataset", methods=["POST"])
def api_reload_dataset():
    """Hot-reloads all in-memory datasets, composite bar caches, and ML pivot models."""
    global _RS_DF, _EP_DF, _OPT_DF, _EXTREME_PEAK_MODEL, _CATALOG_CACHE
    _RS_DF = None
    _EP_DF = None
    _OPT_DF = None
    _EXTREME_PEAK_MODEL = None
    _PIVOT_MODELS_CACHE.clear()
    _COMPOSITE_BAR_CACHE.clear()
    _CATALOG_CACHE = None
    tap._MARKET_SYNC_CACHE = None
    tap._B4_INTRADAY_CACHE.clear()
    models_v3 = get_all_pivot_models("v3")
    models_v4 = get_all_pivot_models("v4")
    cat = build_catalog()
    return jsonify({
        "status": "reloaded",
        "pivot_models_loaded": list(models_v3.keys()),
        "pivot_models_v4_loaded": list(models_v4.keys()),
        "catalog_counts": cat.get("counts", {}),
    })


@app.route("/api/catalog")
def api_catalog():
    cat = build_catalog()
    q = request.args.get("q", "").strip().upper()
    asset_type = request.args.get("type", "all").strip().lower()
    limit = int(request.args.get("limit", "300"))

    if not q and asset_type == "all":
        # Return full ETFs, Sectors, Themes, and top 400 RS stocks by default (searchable for all 6,240)
        return jsonify({
            "etfs": cat["etfs"],
            "sectors": cat["sectors"],
            "themes": cat["themes"],
            "stocks": cat["stocks"][:limit],
            "counts": cat["counts"],
        })

    def _match(item: Dict[str, Any]) -> bool:
        if not q:
            return True
        return (
            q in str(item.get("id", "")).upper()
            or q in str(item.get("symbol", "")).upper()
            or q in str(item.get("name", "")).upper()
            or q in str(item.get("sector", "")).upper()
            or q in str(item.get("theme", "")).upper()
            or q in str(item.get("focus", "")).upper()
        )

    res = {
        "etfs": [x for x in cat["etfs"] if _match(x)] if asset_type in ("all", "etf", "spdr") else [],
        "sectors": [x for x in cat["sectors"] if _match(x)] if asset_type in ("all", "sector") else [],
        "themes": [x for x in cat["themes"] if _match(x)] if asset_type in ("all", "theme") else [],
        "stocks": [x for x in cat["stocks"] if _match(x)][:limit] if asset_type in ("all", "stock") else [],
        "counts": cat["counts"],
    }
    return jsonify(res)


@app.route("/api/watchlists", methods=["GET", "POST"])
def api_watchlists():
    wl_data = load_watchlists()
    if request.method == "GET":
        enriched = enrich_watchlist_items(wl_data.get("groups", []))
        return jsonify({
            "active_group_id": wl_data.get("active_group_id", enriched[0]["id"] if enriched else ""),
            "groups": enriched,
        })

    body = request.get_json(force=True) or {}
    action = body.get("action", "save_all")

    groups = wl_data.get("groups", [])

    if action == "save_all":
        if "groups" in body and isinstance(body["groups"], list):
            # Strip transient live fields before saving
            clean_groups = []
            for g in body["groups"]:
                clean_items = []
                for it in g.get("items", []):
                    clean_items.append({
                        "id": it.get("id"),
                        "type": it.get("type", parse_asset_id(it.get("id", ""))["type"]),
                        "name": it.get("name", ""),
                        "flag": it.get("flag", "none"),
                        "notes": it.get("notes", ""),
                    })
                clean_groups.append({
                    "id": g.get("id") or f"wl_{int(time.time()*1000)}",
                    "name": g.get("name", "Watchlist"),
                    "items": clean_items,
                })
            wl_data["groups"] = clean_groups
        if "active_group_id" in body:
            wl_data["active_group_id"] = body["active_group_id"]

    elif action == "create_group":
        name = (body.get("name") or "New Watchlist").strip()
        new_id = f"wl_{int(time.time() * 1000)}"
        groups.append({"id": new_id, "name": name, "items": []})
        wl_data["active_group_id"] = new_id

    elif action == "rename_group":
        gid = body.get("group_id")
        new_name = (body.get("name") or "").strip()
        for g in groups:
            if g["id"] == gid and new_name:
                g["name"] = new_name

    elif action == "delete_group":
        gid = body.get("group_id")
        if len(groups) > 1:
            wl_data["groups"] = [g for g in groups if g["id"] != gid]
            if wl_data.get("active_group_id") == gid:
                wl_data["active_group_id"] = wl_data["groups"][0]["id"]

    elif action == "add_item":
        gid = body.get("group_id") or wl_data.get("active_group_id")
        asset_id = (body.get("id") or body.get("symbol") or "").strip()
        if asset_id:
            parsed = parse_asset_id(asset_id)
            target_grp = next((g for g in groups if g["id"] == gid), groups[0] if groups else None)
            if target_grp is not None:
                existing = next((it for it in target_grp["items"] if it["id"] == parsed["id"]), None)
                if not existing:
                    target_grp["items"].append({
                        "id": parsed["id"],
                        "type": parsed["type"],
                        "name": body.get("name") or parsed["name"],
                        "flag": body.get("flag", "none"),
                        "notes": body.get("notes", ""),
                    })

    elif action == "remove_item":
        gid = body.get("group_id") or wl_data.get("active_group_id")
        asset_id = (body.get("id") or body.get("symbol") or "").strip()
        for g in groups:
            if g["id"] == gid:
                g["items"] = [it for it in g["items"] if it["id"] != asset_id]

    elif action == "update_item":
        gid = body.get("group_id") or wl_data.get("active_group_id")
        asset_id = (body.get("id") or body.get("symbol") or "").strip()
        for g in groups:
            if not gid or g["id"] == gid:
                for it in g["items"]:
                    if it["id"] == asset_id:
                        if "flag" in body:
                            it["flag"] = body["flag"]
                        if "notes" in body:
                            it["notes"] = body["notes"]

    save_watchlists(wl_data)
    enriched = enrich_watchlist_items(wl_data.get("groups", []))
    return jsonify({
        "status": "ok",
        "active_group_id": wl_data.get("active_group_id", enriched[0]["id"] if enriched else ""),
        "groups": enriched,
    })


def _parse_spans(raw_str: str, defaults: List[int]) -> List[int]:
    if not raw_str:
        return list(defaults)
    parts = [p.strip() for p in raw_str.split(",") if p.strip()]
    out: List[int] = []
    for idx, d_val in enumerate(defaults):
        if idx < len(parts):
            try:
                val = max(2, min(500, int(parts[idx])))
                out.append(val)
            except (ValueError, TypeError):
                out.append(d_val)
        else:
            out.append(d_val)
    return out


@app.route("/api/chart")
def api_chart():
    raw_symbol = request.args.get("symbol", "SPY").strip()
    compare_symbol = request.args.get("compare", "").strip()
    history_window = request.args.get("history", "2y").strip().lower()
    ema_spans = _parse_spans(request.args.get("ema_spans", ""), [8, 10, 20, 21])
    sma_spans = _parse_spans(request.args.get("sma_spans", ""), [50, 200])

    # Check if raw_symbol is already a RATIO:<A>|<B> identifier
    parsed_main = parse_asset_id(raw_symbol)
    if parsed_main["type"] == "ratio":
        part_a_id = parsed_main["num"]
        part_b_id = parsed_main["den"]
    else:
        part_a_id = parsed_main["id"]
        part_b_id = compare_symbol if compare_symbol else ""

    df_a, meta_a = resolve_asset_bars(part_a_id)
    if df_a is None or df_a.empty:
        return jsonify({"error": f"No historical bar data found for '{part_a_id}'"}), 404

    is_ratio = bool(part_b_id)
    df_b = pd.DataFrame()
    meta_b: Dict[str, Any] = {}
    perf_b: Optional[Dict[str, Any]] = None

    if is_ratio:
        df_b, meta_b = resolve_asset_bars(part_b_id)
        if df_b is None or df_b.empty:
            return jsonify({"error": f"No historical bar data found for denominator '{part_b_id}'"}), 404
        df_chart_full = compute_ratio_bars(df_a, df_b)
        if df_chart_full.empty:
            return jsonify({"error": f"Insufficient overlapping dates between '{part_a_id}' and '{part_b_id}'"}), 400
        perf_b = compute_series_performance(df_b, meta_b)
    else:
        df_chart_full = df_a.copy()

    perf_a = compute_series_performance(df_a, meta_a)
    chart_meta = {
        "id": f"RATIO:{meta_a['id']}|{meta_b['id']}" if is_ratio else meta_a["id"],
        "key": f"{format_asset_short_label(meta_a['id'])} ÷ {format_asset_short_label(meta_b['id'])}" if is_ratio else meta_a.get("key", meta_a["id"]),
        "name": f"Ratio: {meta_a['name']} ÷ {meta_b['name']}" if is_ratio else meta_a["name"],
        "type": "ratio" if is_ratio else meta_a["type"],
        "category": "Ratio Chart" if is_ratio else meta_a.get("category", meta_a["type"].upper()),
        "member_count": meta_a.get("member_count"),
    }
    perf_chart = compute_series_performance(df_chart_full, chart_meta) if is_ratio else perf_a

    # Compute full-series Moving Averages & Larsson Line before slicing to history window
    df_ind = indicators.add_indicators(df_chart_full)
    ema_series_list = [df_ind["close"].ewm(span=sp, adjust=False).mean() for sp in ema_spans]
    sma_series_list = [df_ind["close"].rolling(sp).mean() for sp in sma_spans]
    vol_sma20_s = df_ind["volume"].rolling(20).mean()

    # Larsson Line calculation
    try:
        larsson_fastest = int(request.args.get("larsson_fastest", "32"))
    except (ValueError, TypeError):
        larsson_fastest = 32
    larsson_fastest = max(2, min(100, larsson_fastest))

    if larsson_fastest == 32:
        larsson_spans = (32, 35, 50, 58)
    elif larsson_fastest <= 20:
        larsson_spans = (
            larsson_fastest,
            int(round(larsson_fastest * 1.5)),
            int(round(larsson_fastest * 2.0)),
            int(round(larsson_fastest * 2.625)),
        )
    else:
        larsson_spans = (
            larsson_fastest,
            int(round(larsson_fastest * 35.0 / 32.0)),
            int(round(larsson_fastest * 50.0 / 32.0)),
            int(round(larsson_fastest * 58.0 / 32.0)),
        )

    s1, s2, s3, s4 = larsson_spans
    e1 = scanner_core._pine_ema(df_ind["close"], s1)
    e2 = scanner_core._pine_ema(df_ind["close"], s2)
    e3 = scanner_core._pine_ema(df_ind["close"], s3)
    e4 = scanner_core._pine_ema(df_ind["close"], s4)

    bullish = (e1 >= e2) & (e2 >= e3) & (e3 >= e4)
    bearish = (e1 < e2) & (e2 < e3) & (e3 < e4)
    valid_l = e1.notna() & e2.notna() & e3.notna() & e4.notna()

    l_state = pd.Series("gray", index=df_ind.index)
    l_state[valid_l & bullish] = "yellow"
    l_state[valid_l & bearish] = "blue"

    # History window slice
    window_bars_map = {
        "3m": 65,
        "6m": 130,
        "1y": 255,
        "2y": 510,
        "5y": 1260,
        "full": len(df_ind),
    }
    max_bars = window_bars_map.get(history_window, 510)
    if len(df_ind) > max_bars:
        df_view = df_ind.iloc[-max_bars:]
    else:
        df_view = df_ind

    min_view_date = df_view.index[0].strftime("%Y-%m-%d")

    # Relative Strength Line vs SPY across visible window
    spy_bars = datastore.load_bars("SPY")
    rs_line_data = []
    if spy_bars is not None and not spy_bars.empty:
        idx_shared = df_view.index.intersection(spy_bars.index)
        if len(idx_shared) >= 10:
            sub_c = df_view.loc[idx_shared, "close"]
            spy_c = spy_bars.loc[idx_shared, "close"]
            raw_rs = sub_c / spy_c.replace(0, np.nan)
            raw_rs = raw_rs.dropna()
            if not raw_rs.empty and raw_rs.iloc[0] > 0:
                norm_rs = (raw_rs / raw_rs.iloc[0]) * 100.0
                rs_ma21 = norm_rs.ewm(span=21, adjust=False).mean()
                for t_idx in norm_rs.index:
                    t_str = t_idx.strftime("%Y-%m-%d")
                    rs_line_data.append({
                        "time": t_str,
                        "value": round(float(norm_rs.loc[t_idx]), 2),
                        "ma": round(float(rs_ma21.loc[t_idx]), 2),
                    })

    bars = []
    volume = []
    vol_ma20_data = []
    emas_data: List[List[Dict[str, Any]]] = [[], [], [], []]
    smas_data: List[List[Dict[str, Any]]] = [[], []]
    larsson_data = []

    decimals = 4 if float(df_view["close"].median()) < 2.0 else 2

    for idx, r in df_view.iterrows():
        t = idx.strftime("%Y-%m-%d")
        o = round(float(r["open"]), decimals)
        h = round(float(r["high"]), decimals)
        l = round(float(r["low"]), decimals)
        c = round(float(r["close"]), decimals)
        v = float(r["volume"]) if pd.notna(r["volume"]) else 0.0

        bars.append({"time": t, "open": o, "high": h, "low": l, "close": c})
        volume.append({
            "time": t,
            "value": round(v, 0),
            "color": "rgba(16,185,129,0.50)" if c >= o else "rgba(239,68,68,0.50)",
        })

        v_ma = vol_sma20_s.loc[idx]
        if pd.notna(v_ma):
            vol_ma20_data.append({"time": t, "value": round(float(v_ma), 0)})
        for slot_i in range(4):
            val_e = ema_series_list[slot_i].loc[idx]
            if pd.notna(val_e):
                emas_data[slot_i].append({"time": t, "value": round(float(val_e), decimals)})
        for slot_j in range(2):
            val_s = sma_series_list[slot_j].loc[idx]
            if pd.notna(val_s):
                smas_data[slot_j].append({"time": t, "value": round(float(val_s), decimals)})

        f_val = round(float(e1.loc[idx]), decimals) if pd.notna(e1.loc[idx]) else None
        s_val = round(float(e4.loc[idx]), decimals) if pd.notna(e4.loc[idx]) else None
        larsson_data.append({
            "time": t,
            "fast": f_val,
            "slow": s_val,
            "value": f_val,
            "state": str(l_state.loc[idx]),
        })

    # Strong EP events for Part A (and Part B if ratio)
    ep_events_a = get_strong_ep_events_for_asset(meta_a, min_date=min_view_date)
    context_a = get_constituents_and_context(meta_a)

    # Universal ML Climax / Exhaustion Scoring (works on ANY ticker, ETF, Sector, Theme, or Ratio)
    model_ver = request.args.get("model_ver", "v3").strip().lower()
    sym_for_b4 = meta_a.get("key") if (not is_ratio and meta_a.get("type") in ("stock", "etf")) else None
    exhaustion_data = compute_universal_exhaustion(
        df_chart_full,
        df_view,
        perf_chart.get("rs_score"),
        ep_events_a if not is_ratio else None,
        model_ver=model_ver,
        symbol=sym_for_b4,
    )

    # Catalyst lookup for latest EP if single stock
    latest_catalyst = {}
    if meta_a["type"] == "stock" and ep_events_a:
        try:
            latest_catalyst = ep_catalyst_enricher.get_catalyst(meta_a["key"], ep_events_a[-1]["date"]) or {}
        except Exception:
            latest_catalyst = {}

    return jsonify({
        "asset": chart_meta,
        "is_ratio": is_ratio,
        "part_a": meta_a,
        "part_b": meta_b if is_ratio else None,
        "performance": perf_chart,
        "performance_a": perf_a,
        "performance_b": perf_b,
        "bars": bars,
        "volume": volume,
        "vol_ma20": vol_ma20_data,
        "ema_spans": ema_spans,
        "sma_spans": sma_spans,
        "emas": emas_data,
        "smas": smas_data,
        "ema8": emas_data[0],
        "ema10": emas_data[1],
        "ema20": emas_data[2],
        "ema21": emas_data[3],
        "sma50": smas_data[0],
        "sma200": smas_data[1],
        "larsson": larsson_data,
        "larsson_spans": list(larsson_spans),
        "rs_line": rs_line_data,
        "ep_events": ep_events_a,
        "exhaustion": exhaustion_data,
        "catalyst": latest_catalyst,
        "context": context_a,
    })


def _sanitize_annotation_key(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", symbol.strip())[:100]


@app.route("/api/annotations", methods=["GET", "POST"])
def api_annotations():
    if request.method == "GET":
        sym = request.args.get("symbol", "").strip()
        if not sym:
            return jsonify({"drawings": []})
        safe_key = _sanitize_annotation_key(sym)
        fpath = ANNOTATIONS_DIR / f"wl_{safe_key}.json"
        if fpath.exists():
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    return jsonify(json.load(f))
            except Exception:
                return jsonify({"drawings": []})
        return jsonify({"drawings": []})

    data = request.get_json(force=True) or {}
    sym = (data.get("symbol") or "").strip()
    if not sym:
        return jsonify({"error": "Missing symbol"}), 400
    safe_key = _sanitize_annotation_key(sym)
    fpath = ANNOTATIONS_DIR / f"wl_{safe_key}.json"
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump({"symbol": sym, "drawings": data.get("drawings", [])}, f, indent=2)
    return jsonify({"status": "ok"})


# ==============================================================================
# HTML / CSS / JS Frontend Workspace
# ==============================================================================

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EP Universal Chart Viewer, Ratio Analyzer & Watchlist Studio</title>
<script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root {
    --bg: #080b11;
    --panel: #0e131d;
    --panel-alt: #131a27;
    --border: #1e293b;
    --text: #e2e8f0;
    --muted: #64748b;
    --accent: #38bdf8;
    --green: #10b981;
    --red: #ef4444;
    --amber: #f59e0b;
    --purple: #a855f7;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    font-size: 12px;
  }

  /* Top Navigation & Ratio Bar */
  .top-header {
    background: linear-gradient(90deg, #0b101b 0%, #111a2e 100%);
    border-bottom: 1px solid var(--border);
    padding: 8px 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-shrink: 0;
  }
  .brand-block {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .brand-title {
    font-size: 14px;
    font-weight: 800;
    letter-spacing: 0.4px;
    color: #fff;
    display: flex;
    align-items: center;
    gap: 7px;
  }
  .brand-badge {
    background: rgba(56, 189, 248, 0.15);
    color: var(--accent);
    border: 1px solid rgba(56, 189, 248, 0.35);
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 700;
  }

  /* Universal Asset & Ratio Selector Bar */
  .asset-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    flex: 1;
    max-width: 820px;
    background: #0a0f18;
    border: 1px solid #1e293b;
    padding: 4px 8px;
    border-radius: 7px;
  }
  .part-pill {
    display: flex;
    align-items: center;
    gap: 6px;
    background: #131c2e;
    border: 1px solid #28364f;
    padding: 5px 10px;
    border-radius: 5px;
    cursor: pointer;
    min-width: 170px;
    transition: all 0.15s;
  }
  .part-pill:hover {
    border-color: var(--accent);
    background: #17233b;
  }
  .part-pill .lbl {
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    color: var(--muted);
    background: #0b101b;
    padding: 1px 4px;
    border-radius: 3px;
  }
  .part-pill .sym {
    font-weight: 800;
    color: #fff;
    font-size: 12px;
  }
  .part-pill .nm {
    color: #94a3b8;
    font-size: 11px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 130px;
  }

  .btn {
    background: #162032;
    color: #cbd5e1;
    border: 1px solid #26354d;
    padding: 5px 10px;
    border-radius: 5px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    transition: all 0.15s;
    white-space: nowrap;
  }
  .btn:hover {
    background: #1e2c45;
    color: #fff;
    border-color: #38bdf8;
  }
  .btn.active {
    background: rgba(56, 189, 248, 0.2);
    color: #38bdf8;
    border-color: #38bdf8;
  }
  .btn-emerald {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border-color: rgba(16, 185, 129, 0.4);
  }
  .btn-emerald:hover {
    background: rgba(16, 185, 129, 0.28);
    border-color: #10b981;
  }
  .btn-amber {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border-color: rgba(245, 158, 11, 0.4);
  }

  /* Quick Ratio Benchmark Strip */
  .ratio-quick-strip {
    display: flex;
    align-items: center;
    gap: 4px;
  }
  .ratio-chip {
    background: #111827;
    color: #94a3b8;
    border: 1px solid #1f2937;
    padding: 3px 7px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 700;
    cursor: pointer;
  }
  .ratio-chip:hover, .ratio-chip.active {
    background: rgba(168, 85, 247, 0.2);
    color: #d8b4fe;
    border-color: #a855f7;
  }

  /* Main 3-Pane Workspace */
  .workspace {
    flex: 1;
    display: grid;
    grid-template-columns: 330px 1fr 390px;
    overflow: hidden;
    position: relative;
  }
  .workspace.info-closed {
    grid-template-columns: 330px 1fr 0px;
  }
  .workspace.fullscreen-chart {
    grid-template-columns: 0px 1fr 0px;
  }

  /* Left Pane: Watchlist & Asset Explorer */
  .left-pane {
    background: var(--panel);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .left-mode-tabs {
    display: grid;
    grid-template-columns: 1fr 1fr;
    background: #090d14;
    border-bottom: 1px solid var(--border);
    padding: 4px;
    gap: 4px;
  }
  .mode-tab {
    padding: 6px 8px;
    text-align: center;
    font-weight: 700;
    font-size: 11px;
    border-radius: 4px;
    cursor: pointer;
    color: var(--muted);
    transition: all 0.15s;
  }
  .mode-tab.active {
    background: #162236;
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.35);
  }

  /* Watchlist Controls */
  .wl-header {
    padding: 8px 10px;
    border-bottom: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    gap: 6px;
    background: #0b1019;
  }
  .wl-group-row {
    display: flex;
    gap: 5px;
    align-items: center;
  }
  .wl-select {
    flex: 1;
    background: #131c2c;
    color: #fff;
    border: 1px solid #293852;
    padding: 5px 8px;
    border-radius: 5px;
    font-weight: 700;
    font-size: 12px;
  }
  .icon-btn {
    background: #141d2d;
    color: #94a3b8;
    border: 1px solid #24324a;
    width: 26px;
    height: 26px;
    border-radius: 4px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    font-size: 12px;
  }
  .icon-btn:hover {
    color: #fff;
    border-color: var(--accent);
    background: #1c2940;
  }

  /* Asset Explorer Subtabs */
  .explorer-subtabs {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 3px;
    padding: 6px 8px;
    background: #0b1019;
    border-bottom: 1px solid var(--border);
  }
  .sub-tab {
    padding: 4px;
    text-align: center;
    font-size: 10px;
    font-weight: 700;
    border-radius: 4px;
    cursor: pointer;
    color: #94a3b8;
    background: #111826;
    border: 1px solid transparent;
  }
  .sub-tab.active {
    color: #fff;
    background: #1d2b44;
    border-color: #38bdf8;
  }

  .search-box {
    padding: 6px 8px;
    border-bottom: 1px solid var(--border);
    background: #0b1019;
  }
  .search-input {
    width: 100%;
    background: #131b2b;
    border: 1px solid #26354d;
    color: #fff;
    padding: 6px 9px;
    border-radius: 5px;
    font-size: 11px;
  }
  .search-input:focus {
    outline: none;
    border-color: var(--accent);
  }

  /* Watchlist & Explorer Item List */
  .item-list {
    flex: 1;
    overflow-y: auto;
    divide-y: 1px solid #141c2b;
  }
  .wl-item {
    padding: 7px 10px;
    border-bottom: 1px solid #141c2b;
    display: flex;
    align-items: center;
    justify-content: space-between;
    cursor: pointer;
    transition: background 0.12s;
    gap: 6px;
  }
  .wl-item:hover {
    background: #141e30;
  }
  .wl-item.active {
    background: rgba(56, 189, 248, 0.13);
    border-left: 3px solid var(--accent);
  }
  .flag-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
    border: 1px solid rgba(255,255,255,0.2);
  }
  .flag-none { background: transparent; border-color: #334155; }
  .flag-green { background: #10b981; }
  .flag-amber { background: #f59e0b; }
  .flag-red { background: #ef4444; }
  .flag-blue { background: #38bdf8; }
  .flag-purple { background: #a855f7; }

  .type-tag {
    font-size: 9px;
    font-weight: 800;
    padding: 1px 4px;
    border-radius: 3px;
    text-transform: uppercase;
  }
  .type-stock { background: rgba(56, 189, 248, 0.15); color: #38bdf8; }
  .type-etf { background: rgba(245, 158, 11, 0.15); color: #fbbf24; }
  .type-sector { background: rgba(16, 185, 129, 0.15); color: #34d399; }
  .type-theme { background: rgba(168, 85, 247, 0.15); color: #c084fc; }
  .type-ratio { background: rgba(236, 72, 153, 0.15); color: #f472b6; }

  .rs-badge {
    font-size: 10px;
    font-weight: 800;
    padding: 1px 5px;
    border-radius: 4px;
    background: #1e293b;
    color: #e2e8f0;
  }
  .rs-badge.elite { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16,185,129,0.4); }
  .rs-badge.strong { background: rgba(56, 189, 248, 0.2); color: #38bdf8; }

  /* Center Pane: Interactive Chart */
  .center-pane {
    display: flex;
    flex-direction: column;
    background: var(--bg);
    overflow: hidden;
    position: relative;
    min-height: 0;
    height: 100%;
  }
  .chart-toolbar {
    padding: 6px 12px;
    background: #0c111b;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
    flex-shrink: 0;
  }
  .chart-title-group {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .chart-main-sym {
    font-size: 16px;
    font-weight: 800;
    color: #fff;
  }
  .chart-main-name {
    font-size: 12px;
    color: #94a3b8;
    font-weight: 500;
  }
  .ohlc-legend {
    font-family: "JetBrains Mono", monospace;
    font-size: 11px;
    color: #94a3b8;
    display: flex;
    gap: 8px;
    margin-left: 8px;
  }

  .chart-controls-right {
    display: flex;
    align-items: center;
    gap: 5px;
  }
  .tf-btn {
    background: #121a29;
    color: #94a3b8;
    border: 1px solid #1f2d44;
    padding: 3px 7px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 700;
    cursor: pointer;
  }
  .tf-btn.active, .tf-btn:hover {
    background: #1e2f4d;
    color: #fff;
    border-color: var(--accent);
  }

  /* Settings & Drawing Popovers */
  .floating-bar {
    background: rgba(14, 19, 29, 0.96);
    border-bottom: 1px solid var(--border);
    padding: 6px 12px;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    z-index: 30;
    flex-shrink: 0;
  }
  .cfg-label {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    color: #cbd5e1;
    cursor: pointer;
    user-select: none;
  }
  .swatch {
    width: 9px;
    height: 9px;
    border-radius: 2px;
    display: inline-block;
  }

  .chart-stage {
    flex: 1 1 0;
    min-height: 0;
    position: relative;
    width: 100%;
    overflow: hidden;
  }
  #chart_tv {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
  }
  #larsson_ribbon_svg {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 3;
  }
  #draw_canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 10;
  }

  /* Right Pane: Toggleable Info & Performance Panel */
  .right-pane {
    background: var(--panel);
    border-left: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    padding: 10px;
    gap: 10px;
  }
  .workspace.info-closed .right-pane {
    display: none;
  }

  .info-card {
    background: var(--panel-alt);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 10px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .card-title {
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #94a3b8;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 5px;
  }

  .perf-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 5px;
  }
  .perf-cell {
    background: #0b1019;
    border: 1px solid #1a2436;
    border-radius: 5px;
    padding: 5px 6px;
    text-align: center;
  }
  .perf-cell .k {
    font-size: 9px;
    color: var(--muted);
    font-weight: 700;
  }
  .perf-cell .v {
    font-size: 12px;
    font-weight: 800;
    margin-top: 2px;
  }
  .pos { color: #34d399; }
  .neg { color: #f87171; }
  .neu { color: #cbd5e1; }

  .kv-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11px;
    padding: 2px 0;
  }
  .kv-row .k { color: #94a3b8; }
  .kv-row .v { font-weight: 700; color: #f1f5f9; }

  .ep-item {
    background: #0b1019;
    border: 1px solid #1e293b;
    border-left: 3px solid #10b981;
    border-radius: 5px;
    padding: 6px 8px;
    cursor: pointer;
    transition: background 0.12s;
  }
  .ep-item:hover {
    background: #141f33;
  }
  .ep-item.tier1 {
    border-left-color: #f59e0b;
    background: rgba(245, 158, 11, 0.06);
  }

  /* Universal Asset Chooser Modal */
  .modal-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(5, 8, 14, 0.82);
    backdrop-filter: blur(3px);
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }
  .modal-box {
    background: #0e1420;
    border: 1px solid #293852;
    border-radius: 10px;
    width: 680px;
    max-width: 94vw;
    max-height: 82vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    box-shadow: 0 20px 50px rgba(0,0,0,0.65);
  }
  .modal-top {
    padding: 12px 14px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .modal-tabs {
    display: flex;
    gap: 6px;
    padding: 8px 14px;
    background: #090d15;
    border-bottom: 1px solid var(--border);
  }
  .modal-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px 14px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .picker-row {
    padding: 7px 10px;
    border-radius: 6px;
    background: #131c2c;
    border: 1px solid #1e2b42;
    display: flex;
    align-items: center;
    justify-content: space-between;
    cursor: pointer;
  }
  .picker-row:hover {
    border-color: var(--accent);
    background: #18253b;
  }
  .ma-span-input {
    width: 38px;
    background: #131b2b;
    color: #fff;
    border: 1px solid #293852;
    border-radius: 4px;
    padding: 1px 3px;
    font-size: 11px;
    font-weight: 700;
    text-align: center;
  }
  .ma-span-input:focus {
    outline: none;
    border-color: var(--accent);
  }
</style>
</head>
<body>

<!-- TOP HEADER: BRAND + UNIVERSAL ASSET / RATIO BAR + PANEL TOGGLES -->
<header class="top-header">
  <div class="brand-block">
    <div class="brand-title">
      <span>📈 EP Chart & Watchlist Studio</span>
      <span class="brand-badge">PORT 8789</span>
    </div>
  </div>

  <!-- Universal Asset & Ratio Bar -->
  <div class="asset-bar">
    <div class="part-pill" onclick="openAssetModal('a')" title="Click to choose Stock, SPDR ETF, Sector, or Theme">
      <span class="lbl" id="pill_a_type">STOCK</span>
      <span class="sym" id="pill_a_sym">SPY</span>
      <span class="nm" id="pill_a_name">SPDR S&P 500</span>
    </div>

    <button class="btn" id="btn_toggle_ratio" onclick="toggleRatioMode()" title="Enable or disable Ratio Chart (Part A ÷ Part B)">
      ⚖️ Ratio (A ÷ B)
    </button>

    <div id="ratio_controls_wrap" style="display:none; align-items:center; gap:6px;">
      <span style="font-weight:800; color:#a855f7; font-size:14px;">÷</span>
      <div class="part-pill" onclick="openAssetModal('b')" title="Click to choose Denominator (Stock, SPDR ETF, Sector, or Theme)">
        <span class="lbl" id="pill_b_type">ETF</span>
        <span class="sym" id="pill_b_sym">SPY</span>
        <span class="nm" id="pill_b_name">SPDR S&P 500</span>
      </div>
      <button class="btn" onclick="swapRatioParts()" title="Swap Numerator (A) and Denominator (B)">⇄ Swap</button>
      <button class="btn btn-emerald" onclick="addCurrentChartToWatchlist()" title="Save this Ratio Pair to Active Watchlist">+ Save Ratio</button>
    </div>

    <!-- Quick Benchmark Ratio Chips -->
    <div class="ratio-quick-strip">
      <span style="color:var(--muted); font-size:10px; margin-right:2px;">vs:</span>
      <span class="ratio-chip" id="qchip_off" onclick="setQuickRatio('')">Price</span>
      <span class="ratio-chip" id="qchip_SPY" onclick="setQuickRatio('SPY')">÷SPY</span>
      <span class="ratio-chip" id="qchip_QQQ" onclick="setQuickRatio('QQQ')">÷QQQ</span>
      <span class="ratio-chip" id="qchip_IWM" onclick="setQuickRatio('IWM')">÷IWM</span>
      <span class="ratio-chip" id="qchip_XBI" onclick="setQuickRatio('XBI')">÷XBI</span>
      <span class="ratio-chip" id="qchip_SMH" onclick="setQuickRatio('SMH')">÷SMH</span>
      <span class="ratio-chip" id="qchip_SECTOR" onclick="setQuickRatioSector()" title="Divide by this asset's own Sector Index">÷Sector</span>
      <span class="ratio-chip" id="qchip_THEME" onclick="setQuickRatioTheme()" title="Divide by this asset's Primary Theme Index">÷Theme</span>
    </div>
  </div>

  <!-- Right Header Controls -->
  <div style="display:flex; align-items:center; gap:6px;">
    <button class="btn btn-emerald" onclick="addCurrentChartToWatchlist()" title="Add current symbol or ratio to active watchlist">
      ⭐ + Watchlist
    </button>
    <button class="btn active" id="btn_toggle_info" onclick="toggleInfoPanel()" title="Open or collapse the right-hand Performance & Info Panel">
      📊 Info Panel
    </button>
  </div>
</header>

<!-- MAIN 3-PANE WORKSPACE -->
<div class="workspace" id="main_workspace">

  <!-- LEFT PANE: ORGANIZED WATCHLISTS & UNIVERSAL EXPLORER -->
  <aside class="left-pane">
    <div class="left-mode-tabs">
      <div class="mode-tab active" id="ltab_watchlists" onclick="switchLeftTab('watchlists')">⭐ Watchlists</div>
      <div class="mode-tab" id="ltab_explorer" onclick="switchLeftTab('explorer')">🧭 Universe Explorer</div>
    </div>

    <!-- WATCHLISTS VIEW -->
    <div id="view_watchlists" style="display:flex; flex-direction:column; flex:1; overflow:hidden;">
      <div class="wl-header">
        <div class="wl-group-row">
          <select id="wl_group_select" class="wl-select" onchange="onWatchlistGroupChange()"></select>
          <button class="icon-btn" onclick="createNewWatchlistGroup()" title="Create New Watchlist Folder">➕</button>
          <button class="icon-btn" onclick="renameCurrentWatchlistGroup()" title="Rename Watchlist Folder">✏️</button>
          <button class="icon-btn" onclick="deleteCurrentWatchlistGroup()" title="Delete Watchlist Folder">🗑️</button>
          <button class="icon-btn" onclick="exportWatchlistTradingView()" title="Export Watchlist to TradingView">📤</button>
        </div>
        <div style="display:flex; gap:5px; align-items:center;">
          <input type="text" id="wl_filter_input" class="search-input" placeholder="Filter watchlist items..." oninput="renderWatchlistItems()">
          <select id="wl_flag_filter" onchange="renderWatchlistItems()" style="background:#131b2b; color:#cbd5e1; border:1px solid #26354d; padding:5px; border-radius:5px; font-size:10px;">
            <option value="all">All Flags</option>
            <option value="green">🟢 Green</option>
            <option value="amber">🟡 Amber</option>
            <option value="red">🔴 Red</option>
            <option value="blue">🔵 Blue</option>
            <option value="purple">🟣 Purple</option>
          </select>
        </div>
      </div>
      <div class="item-list" id="wl_items_container"></div>
    </div>

    <!-- UNIVERSE EXPLORER VIEW (STOCKS / SPDR ETFs / SECTORS / THEMES) -->
    <div id="view_explorer" style="display:none; flex-direction:column; flex:1; overflow:hidden;">
      <div class="explorer-subtabs">
        <div class="sub-tab active" id="ex_tab_etf" onclick="switchExplorerCategory('etf')">SPDR / ETFs</div>
        <div class="sub-tab" id="ex_tab_sector" onclick="switchExplorerCategory('sector')">Sectors (154)</div>
        <div class="sub-tab" id="ex_tab_theme" onclick="switchExplorerCategory('theme')">Themes (187)</div>
        <div class="sub-tab" id="ex_tab_stock" onclick="switchExplorerCategory('stock')">Stocks (6.2k)</div>
      </div>
      <div class="search-box">
        <input type="text" id="explorer_search" class="search-input" placeholder="Search ticker, SPDR (XBI), sector, or theme..." oninput="onExplorerSearch()">
      </div>
      <div class="item-list" id="explorer_items_container"></div>
    </div>
  </aside>

  <!-- CENTER PANE: CHART ENGINE, SETTINGS, LARSSON LINE, EP MARKERS & DRAWINGS -->
  <main class="center-pane">
    <div class="chart-toolbar">
      <div class="chart-title-group">
        <span class="type-tag type-stock" id="hdr_asset_type">ETF</span>
        <span class="chart-main-sym" id="hdr_asset_sym">SPY</span>
        <span class="chart-main-name" id="hdr_asset_name">SPDR S&P 500 ETF Trust</span>
        <span id="hdr_ep_count_badge" style="display:none; background:rgba(16,185,129,0.18); color:#34d399; border:1px solid rgba(16,185,129,0.4); padding:2px 6px; border-radius:4px; font-size:10px; font-weight:700;"></span>
        <span id="hdr_climax_badge" style="display:none; padding:2px 7px; border-radius:4px; font-size:10px; font-weight:800;"></span>
        <div class="ohlc-legend" id="ohlc_legend"></div>
      </div>

      <div class="chart-controls-right">
        <!-- Timeframe History Buttons -->
        <button class="tf-btn" id="tf_3m" onclick="setHistoryWindow('3m')">3M</button>
        <button class="tf-btn" id="tf_6m" onclick="setHistoryWindow('6m')">6M</button>
        <button class="tf-btn" id="tf_1y" onclick="setHistoryWindow('1y')">1Y</button>
        <button class="tf-btn active" id="tf_2y" onclick="setHistoryWindow('2y')">2Y</button>
        <button class="tf-btn" id="tf_5y" onclick="setHistoryWindow('5y')">5Y</button>
        <button class="tf-btn" id="tf_full" onclick="setHistoryWindow('full')">MAX</button>

        <button class="btn" onclick="resetChartZoom()" title="Fit chart to window">🎯 Fit</button>
        <button class="btn active" id="btn_settings_bar" onclick="toggleSettingsBar()">⚙️ Indicators & Larsson</button>
        <button class="btn" id="btn_draw_bar" onclick="toggleDrawBar()">✏️ Draw</button>
        <button class="btn" id="btn_fullscreen" onclick="toggleFullscreenChart()">⛶ Enlarge</button>
      </div>
    </div>

    <!-- Customizable Indicator, EMA/SMA Periods, Larsson Line & EP/Climax Marker Settings Bar -->
    <div class="floating-bar" id="settings_bar">
      <span class="cfg-label" title="Toggle & customize EMA 1 period (e.g. 8 or 12)">
        <input type="checkbox" id="cfg_ema8" onchange="updateIndicatorVisibility()">
        <span class="swatch" style="background:#a855f7;"></span>
        <input type="number" id="cfg_ema_span_0" value="8" min="2" max="500" class="ma-span-input" onchange="onMaSpanChange()"> EMA
      </span>
      <span class="cfg-label" title="Toggle & customize EMA 2 period">
        <input type="checkbox" id="cfg_ema10" checked onchange="updateIndicatorVisibility()">
        <span class="swatch" style="background:#38bdf8;"></span>
        <input type="number" id="cfg_ema_span_1" value="10" min="2" max="500" class="ma-span-input" onchange="onMaSpanChange()"> EMA
      </span>
      <span class="cfg-label" title="Toggle & customize EMA 3 period">
        <input type="checkbox" id="cfg_ema20" checked onchange="updateIndicatorVisibility()">
        <span class="swatch" style="background:#f59e0b;"></span>
        <input type="number" id="cfg_ema_span_2" value="20" min="2" max="500" class="ma-span-input" onchange="onMaSpanChange()"> EMA
      </span>
      <span class="cfg-label" title="Toggle & customize EMA 4 period">
        <input type="checkbox" id="cfg_ema21" onchange="updateIndicatorVisibility()">
        <span class="swatch" style="background:#22d3ee;"></span>
        <input type="number" id="cfg_ema_span_3" value="21" min="2" max="500" class="ma-span-input" onchange="onMaSpanChange()"> EMA
      </span>
      <span class="cfg-label" title="Toggle & customize SMA 1 period">
        <input type="checkbox" id="cfg_sma50" checked onchange="updateIndicatorVisibility()">
        <span class="swatch" style="background:#10b981;"></span>
        <input type="number" id="cfg_sma_span_0" value="50" min="2" max="500" class="ma-span-input" onchange="onMaSpanChange()"> SMA
      </span>
      <span class="cfg-label" title="Toggle & customize SMA 2 period">
        <input type="checkbox" id="cfg_sma200" checked onchange="updateIndicatorVisibility()">
        <span class="swatch" style="background:#ef4444;"></span>
        <input type="number" id="cfg_sma_span_1" value="200" min="2" max="500" class="ma-span-input" onchange="onMaSpanChange()"> SMA
      </span>

      <label class="cfg-label"><input type="checkbox" id="cfg_volume" checked onchange="updateIndicatorVisibility()">📊 Volume + 20MA</label>
      <label class="cfg-label"><input type="checkbox" id="cfg_rs_line" checked onchange="updateIndicatorVisibility()"><span class="swatch" style="background:#ec4899;"></span>RS Line vs SPY</label>
      <label class="cfg-label"><input type="checkbox" id="cfg_ep_markers" checked onchange="renderEPMarkers()">⚡ Strong EP Markers</label>
      <span class="cfg-label" title="Toggle Uptrend & Downtrend Pivot-High markers (🔥 UT High / 🔻 DT High) above the probability threshold">
        <input type="checkbox" id="cfg_climax_markers" checked onchange="renderEPMarkers()">🔥 Pivot Highs
      </span>
      <span class="cfg-label" title="Toggle Uptrend & Downtrend Pivot-Low markers (💎 UT Low / 🟢 DT Low) above the probability threshold">
        <input type="checkbox" id="cfg_pivot_low_markers" checked onchange="renderEPMarkers()">💎 Pivot Lows ≥
        <input type="number" id="cfg_climax_thresh" value="50" min="25" max="85" step="5" class="ma-span-input" onchange="renderEPMarkers()" oninput="renderEPMarkers()" title="Minimum ML Pivot Probability % for chart markers (25-85%)">%
      </span>
      <span class="cfg-label" title="Display the lowest intrabar price that classifies a Pivot High and the highest intrabar price that classifies a Pivot Low">
        <input type="checkbox" id="cfg_pivot_intrabar_price" checked onchange="renderEPMarkers()">🎯 Intrabar Trigger $
      </span>
      <span class="cfg-label" title="Switch between V3 Baseline (Daily OHLCV) and V4 Enhanced (B1–B5: Swings, 60D AVWAP, SPY/QQQ RS Sync, Weekly W-FRI, 5m Sub-Model A+, Inverse-RSI & Isotonic Calibration)">
        🤖 Model:
        <select id="cfg_pivot_model_ver" onchange="onPivotModelVerChange()" style="background:#131b2b; color:#38bdf8; border:1px solid #26354d; border-radius:4px; padding:1px 5px; font-size:11px; font-weight:700;">
          <option value="v3">V3 Current (Baseline)</option>
          <option value="v4">V4 Enhanced (B1–B5 + Inv-RSI)</option>
        </select>
      </span>

      <div style="display:flex; align-items:center; gap:6px; border-left:1px solid #26354d; padding-left:10px;">
        <label class="cfg-label"><input type="checkbox" id="cfg_larsson" checked onchange="updateIndicatorVisibility()"><span class="swatch" style="background:linear-gradient(90deg,#eab308,#38bdf8);"></span><strong>Larsson Line</strong></label>
        <span style="color:var(--muted); font-size:10px;">Fastest EMA:</span>
        <input type="number" id="cfg_larsson_span" value="32" min="5" max="90" style="width:46px; background:#131b2b; color:#fff; border:1px solid #26354d; border-radius:4px; padding:2px 4px; font-size:11px;" onchange="reloadChart()">
        <span id="lbl_larsson_spans" style="color:var(--muted); font-size:10px;">(32, 35, 50, 58)</span>
      </div>
    </div>

    <!-- Interactive Drawing Toolbar -->
    <div class="floating-bar" id="draw_bar" style="display:none; background:#0d1422;">
      <span style="font-weight:700; color:#94a3b8;">Drawing Tools:</span>
      <button class="btn active" id="dtool_cursor" onclick="setDrawTool('cursor')">🖱️ Cursor</button>
      <button class="btn" id="dtool_line" onclick="setDrawTool('line')">📏 Trendline</button>
      <button class="btn" id="dtool_ray" onclick="setDrawTool('ray')">↗️ Ray</button>
      <button class="btn" id="dtool_hline" onclick="setDrawTool('hline')">➖ Horiz Line</button>
      <button class="btn" id="dtool_box" onclick="setDrawTool('box')">▭ Box / Base</button>
      <button class="btn" id="dtool_fib" onclick="setDrawTool('fib')">🌀 Fib Retrace</button>
      <button class="btn" id="dtool_measure" onclick="setDrawTool('measure')">📐 % Measure</button>
      <button class="btn" onclick="undoLastDrawing()">↩️ Undo</button>
      <button class="btn" onclick="clearAllDrawings()" style="color:#f87171;">🗑️ Clear All</button>
    </div>

    <!-- Chart Canvas Stage -->
    <div class="chart-stage">
      <div id="chart_tv"></div>
      <svg id="larsson_ribbon_svg"></svg>
      <canvas id="draw_canvas"></canvas>
    </div>
  </main>

  <!-- RIGHT PANE: TOGGLEABLE PERFORMANCE & ASSET INFO PANEL -->
  <aside class="right-pane" id="info_panel">

    <!-- 1. Watchlist Item Notes & Color Flag Card -->
    <div class="info-card">
      <div class="card-title">
        <span>📌 Watchlist Status & Notes</span>
        <span id="info_wl_membership" style="font-size:10px; color:#38bdf8;"></span>
      </div>
      <div style="display:flex; align-items:center; justify-content:space-between; gap:6px;">
        <span style="color:#94a3b8; font-size:11px;">Color Flag:</span>
        <div style="display:flex; gap:6px;">
          <button class="icon-btn" onclick="setItemFlag('none')" title="Clear Flag">⚪</button>
          <button class="icon-btn" onclick="setItemFlag('green')" title="Green Flag (Focus)">🟢</button>
          <button class="icon-btn" onclick="setItemFlag('amber')" title="Amber Flag (Stalking)">🟡</button>
          <button class="icon-btn" onclick="setItemFlag('red')" title="Red Flag (Alert)">🔴</button>
          <button class="icon-btn" onclick="setItemFlag('blue')" title="Blue Flag (Benchmark)">🔵</button>
          <button class="icon-btn" onclick="setItemFlag('purple')" title="Purple Flag (Thematic)">🟣</button>
        </div>
      </div>
      <textarea id="info_item_notes" rows="2" placeholder="Add trading notes, trigger levels, or thesis for this symbol..." style="width:100%; background:#0b1019; border:1px solid #1e293b; color:#e2e8f0; padding:6px; border-radius:5px; font-size:11px; resize:vertical;" onchange="saveItemNotes()"></textarea>
    </div>

    <!-- 2. Universal 4-Model Structural Swing-Pivot Suite Card -->
    <div class="info-card" id="ml_exhaustion_card" style="border-color:rgba(56,189,248,0.3);">
      <div class="card-title">
        <span>🤖 4-Model Structural Pivot Suite</span>
        <span class="rs-badge" id="exh_status_badge">--</span>
      </div>
      <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:5px;">
        <div class="perf-cell" title="Model 1: Uptrend Pivot High (Climax / Exhaustion Top)">
          <div class="k" style="color:#fb923c;">🔥 UT HIGH</div>
          <div class="v" id="exh_curr_val">--</div>
          <div style="font-size:9px; color:#64748b;" id="exh_5d_val">5D: --</div>
        </div>
        <div class="perf-cell" title="Model 2: Uptrend Pivot Low (Pullback Support / Reclaim)">
          <div class="k" style="color:#38bdf8;">💎 UT LOW</div>
          <div class="v" id="piv_ut_low_val">--</div>
          <div style="font-size:9px; color:#64748b;" id="piv_ut_low_5d">5D: --</div>
        </div>
        <div class="perf-cell" title="Model 3: Downtrend Pivot High (Bear-Rally Rejection)">
          <div class="k" style="color:#c084fc;">🔻 DT HIGH</div>
          <div class="v" id="piv_dt_high_val">--</div>
          <div style="font-size:9px; color:#64748b;" id="piv_dt_high_5d">5D: --</div>
        </div>
        <div class="perf-cell" title="Model 4: Downtrend Pivot Low (Capitulation Washout Bottom)">
          <div class="k" style="color:#4ade80;">🟢 DT LOW</div>
          <div class="v" id="piv_dt_low_val">--</div>
          <div style="font-size:9px; color:#64748b;" id="piv_dt_low_5d">5D: --</div>
        </div>
      </div>
      <div style="display:none;" id="exh_20d_val">--</div>
      <div style="width:100%; height:6px; background:#0b1019; border-radius:3px; overflow:hidden; border:1px solid #1e293b;">
        <div id="exh_progress_bar" style="height:100%; width:0%; background:#10b981; transition:width 0.25s;"></div>
      </div>
      <div id="exh_action_banner" style="font-size:11px; font-weight:700; padding:5px 8px; border-radius:5px; background:#0b1019; text-align:center;">--</div>
      <div style="border-top:1px solid #1e293b; padding-top:5px;">
        <div class="kv-row"><span class="k">UT High (Stretch / Reversal):</span><span class="v" id="exh_submodels_row">--</span></div>
        <div class="kv-row"><span class="k">Latest Pivot Intrabar Trigger:</span><span class="v" id="exh_intrabar_trig_row" style="color:#fbbf24;">--</span></div>
        <div class="kv-row"><span class="k">RSI-14 (At High / At Low):</span><span class="v" id="exh_rsi_row">--</span></div>
        <div class="kv-row"><span class="k">High Stretch (10E / 20E / 50S ADR):</span><span class="v" id="exh_adr_stretch">--</span></div>
        <div class="kv-row"><span class="k">Wicks (Upper / Lower) & RVOL:</span><span class="v" id="exh_wick_rvol">--</span></div>
        <div class="kv-row"><span class="k">Multi-Wave Divergence:</span><span class="v" id="exh_consec_div">--</span></div>
        <div class="kv-row"><span class="k">Primary Trend & Leg Excursion:</span><span class="v" id="exh_anchor_mode" style="color:#38bdf8;">--</span></div>
      </div>
    </div>

    <!-- 3. Multi-Timeframe Performance Matrix -->
    <div class="info-card">
      <div class="card-title">
        <span id="perf_card_title">📊 Performance & Relative Strength</span>
        <span class="rs-badge elite" id="info_rs_badge">RS --</span>
      </div>
      <div class="perf-grid">
        <div class="perf-cell"><div class="k">1D</div><div class="v" id="pf_1d">--</div></div>
        <div class="perf-cell"><div class="k">1W</div><div class="v" id="pf_1w">--</div></div>
        <div class="perf-cell"><div class="k">1M</div><div class="v" id="pf_1m">--</div></div>
        <div class="perf-cell"><div class="k">3M</div><div class="v" id="pf_3m">--</div></div>
        <div class="perf-cell"><div class="k">6M</div><div class="v" id="pf_6m">--</div></div>
        <div class="perf-cell"><div class="k">YTD</div><div class="v" id="pf_ytd">--</div></div>
        <div class="perf-cell"><div class="k">1Y</div><div class="v" id="pf_1y">--</div></div>
        <div class="perf-cell"><div class="k">ADR %</div><div class="v neu" id="pf_adr">--</div></div>
      </div>
      <div style="margin-top:4px; border-top:1px solid #1e293b; padding-top:6px;">
        <div class="kv-row"><span class="k">Alpha vs SPY (1M / 3M / 1Y):</span><span class="v" id="pf_alpha_spy">--</span></div>
        <div class="kv-row"><span class="k">Alpha vs QQQ (3M):</span><span class="v" id="pf_alpha_qqq">--</span></div>
        <div class="kv-row"><span class="k">52W High / Distance:</span><span class="v" id="pf_52w_hi">--</span></div>
        <div class="kv-row"><span class="k">20D Avg Dollar Vol:</span><span class="v" id="pf_dvol">--</span></div>
        <div class="kv-row"><span class="k">Trend / MA Alignment:</span><span class="v" id="pf_stage2">--</span></div>
        <div class="kv-row"><span class="k">Dist from 10 EMA / 21 EMA / 50 SMA:</span><span class="v" id="pf_ma_dists">--</span></div>
      </div>
    </div>

    <!-- 4. Ratio Side-by-Side Comparison Card (Visible in Ratio Mode) -->
    <div class="info-card" id="ratio_compare_card" style="display:none; border-color:rgba(168,85,247,0.4);">
      <div class="card-title">
        <span>⚖️ Ratio Leg Breakdown (Part A vs Part B)</span>
      </div>
      <div id="ratio_compare_body" style="display:flex; flex-direction:column; gap:6px;"></div>
    </div>

    <!-- 5. Recent Strong EP Events Card -->
    <div class="info-card" id="ep_events_card">
      <div class="card-title">
        <span>⚡ Recent Strong EP Events</span>
        <span id="ep_card_count" style="color:#34d399;">0</span>
      </div>
      <div id="ep_events_list" style="display:flex; flex-direction:column; gap:6px; max-height:240px; overflow-y:auto;"></div>
    </div>

    <!-- 6. Sector & Theme Context + Clickable Constituents -->
    <div class="info-card">
      <div class="card-title">
        <span>🏛️ Sector & Theme Leadership</span>
      </div>
      <div id="group_context_body" style="display:flex; flex-direction:column; gap:8px;"></div>
      <div style="border-top:1px solid #1e293b; padding-top:6px;">
        <div style="font-size:10px; font-weight:700; color:#94a3b8; margin-bottom:5px; text-transform:uppercase;">Top Liquid Leaders (Click to View Chart)</div>
        <div id="constituents_list" style="display:flex; flex-direction:column; gap:4px; max-height:230px; overflow-y:auto;"></div>
      </div>
    </div>

  </aside>
</div>

<!-- UNIVERSAL ASSET CHOOSER MODAL (Select Part A or Part B from Stocks, SPDR ETFs, Sectors, or Themes) -->
<div class="modal-backdrop" id="asset_modal" onclick="if(event.target.id==='asset_modal') closeAssetModal()">
  <div class="modal-box">
    <div class="modal-top">
      <div style="font-size:14px; font-weight:800; color:#fff;" id="modal_picker_title">Select Asset</div>
      <button class="btn" onclick="closeAssetModal()">✕ Close</button>
    </div>
    <div class="modal-tabs">
      <button class="btn active" id="mtab_all" onclick="setModalCategory('all')">All Assets</button>
      <button class="btn" id="mtab_etf" onclick="setModalCategory('etf')">🏛️ SPDR & Sector ETFs</button>
      <button class="btn" id="mtab_sector" onclick="setModalCategory('sector')">🏭 Sectors (154)</button>
      <button class="btn" id="mtab_theme" onclick="setModalCategory('theme')">🔥 Themes (187)</button>
      <button class="btn" id="mtab_stock" onclick="setModalCategory('stock')">📈 Universe Stocks</button>
    </div>
    <div style="padding:10px 14px; background:#0b1019; border-bottom:1px solid var(--border);">
      <input type="text" id="modal_search_input" class="search-input" placeholder="Type symbol (e.g. NVDA, XBI, SMH) or Sector/Theme name (e.g. Biotechnology, AI)..." oninput="renderModalResults()">
    </div>
    <div class="modal-list" id="modal_results_list"></div>
  </div>
</div>

<script>
// ============================================================================
// Application State
// ============================================================================
let catalogData = { etfs: [], sectors: [], themes: [], stocks: [], counts: {} };
let watchlistsState = { active_group_id: '', groups: [] };
let leftTab = 'watchlists';
let explorerCategory = 'etf';

let partA = 'XBI';
let partB = '';
let isRatioMode = false;
let historyWindow = '2y';

let currentChartData = null;
let modalTargetLeg = 'a'; // 'a' or 'b'
let modalCategory = 'all';

// LightweightCharts references
let chart = null;
let candleSeries = null;
let volumeSeries = null;
let volMa20Series = null;
let ema8Series = null;
let ema10Series = null;
let ema20Series = null;
let ema21Series = null;
let sma50Series = null;
let sma200Series = null;
let larssonFastSeries = null;
let larssonSlowSeries = null;
let rsLineSeries = null;
let rsMaSeries = null;

// Drawing tool state
let drawTool = 'cursor';
let drawings = [];
let isDrawing = false;
let currentDrawing = null;

// ============================================================================
// Initialization & Data Fetching
// ============================================================================
function restoreSavedMaSpans() {
  try {
    const savedEma = JSON.parse(localStorage.getItem('ep_wl_ema_spans') || 'null');
    if (Array.isArray(savedEma) && savedEma.length === 4) {
      savedEma.forEach((v, i) => {
        const el = document.getElementById(`cfg_ema_span_${i}`);
        if (el && Number(v) >= 2) el.value = Number(v);
      });
    }
    const savedSma = JSON.parse(localStorage.getItem('ep_wl_sma_spans') || 'null');
    if (Array.isArray(savedSma) && savedSma.length === 2) {
      savedSma.forEach((v, i) => {
        const el = document.getElementById(`cfg_sma_span_${i}`);
        if (el && Number(v) >= 2) el.value = Number(v);
      });
    }
    const savedThresh = parseFloat(localStorage.getItem('ep_wl_climax_thresh') || 'NaN');
    if (!isNaN(savedThresh) && savedThresh >= 20 && savedThresh <= 90) {
      const thEl = document.getElementById('cfg_climax_thresh');
      if (thEl) thEl.value = savedThresh;
    }
    const savedIntrabar = localStorage.getItem('ep_wl_pivot_intrabar');
    if (savedIntrabar !== null) {
      const ibEl = document.getElementById('cfg_pivot_intrabar_price');
      if (ibEl) ibEl.checked = savedIntrabar !== '0';
    }
    const savedModelVer = localStorage.getItem('ep_wl_pivot_model_ver');
    if (savedModelVer === 'v3' || savedModelVer === 'v4') {
      const mvEl = document.getElementById('cfg_pivot_model_ver');
      if (mvEl) mvEl.value = savedModelVer;
    }
  } catch (e) {}
}

function onPivotModelVerChange() {
  const mvEl = document.getElementById('cfg_pivot_model_ver');
  const val = mvEl ? mvEl.value : 'v3';
  try {
    localStorage.setItem('ep_wl_pivot_model_ver', val);
  } catch (e) {}
  loadChart();
}

function getConfiguredEmaSpans() {
  return [0, 1, 2, 3].map(i => {
    const v = parseInt(document.getElementById(`cfg_ema_span_${i}`)?.value || '10', 10);
    return isNaN(v) ? 10 : Math.max(2, Math.min(500, v));
  });
}

function getConfiguredSmaSpans() {
  return [0, 1].map(i => {
    const v = parseInt(document.getElementById(`cfg_sma_span_${i}`)?.value || '50', 10);
    return isNaN(v) ? 50 : Math.max(2, Math.min(500, v));
  });
}

function onMaSpanChange() {
  const emaSpans = getConfiguredEmaSpans();
  const smaSpans = getConfiguredSmaSpans();
  try {
    localStorage.setItem('ep_wl_ema_spans', JSON.stringify(emaSpans));
    localStorage.setItem('ep_wl_sma_spans', JSON.stringify(smaSpans));
  } catch (e) {}
  loadChart();
}

window.addEventListener('DOMContentLoaded', async () => {
  restoreSavedMaSpans();
  initChartEngine();
  await Promise.all([loadCatalog(), loadWatchlists()]);
  loadChart();
});

async function loadCatalog(query = '') {
  const url = query ? `/api/catalog?q=${encodeURIComponent(query)}&limit=400` : '/api/catalog?limit=400';
  const res = await fetch(url);
  const data = await res.json();
  if (!query) {
    catalogData = data;
  }
  renderExplorerItems();
  return data;
}

async function loadWatchlists() {
  const res = await fetch('/api/watchlists');
  watchlistsState = await res.json();
  renderWatchlistGroupSelect();
  renderWatchlistItems();
}

// ============================================================================
// Left Pane: Watchlists & Universe Explorer
// ============================================================================
function switchLeftTab(tab) {
  leftTab = tab;
  document.getElementById('ltab_watchlists').classList.toggle('active', tab === 'watchlists');
  document.getElementById('ltab_explorer').classList.toggle('active', tab === 'explorer');
  document.getElementById('view_watchlists').style.display = tab === 'watchlists' ? 'flex' : 'none';
  document.getElementById('view_explorer').style.display = tab === 'explorer' ? 'flex' : 'none';
  if (tab === 'explorer') renderExplorerItems();
}

function renderWatchlistGroupSelect() {
  const sel = document.getElementById('wl_group_select');
  sel.innerHTML = '';
  (watchlistsState.groups || []).forEach(g => {
    const opt = document.createElement('option');
    opt.value = g.id;
    opt.textContent = `${g.name} (${(g.items || []).length})`;
    if (g.id === watchlistsState.active_group_id) opt.selected = true;
    sel.appendChild(opt);
  });
}

function getActiveWatchlistGroup() {
  return (watchlistsState.groups || []).find(g => g.id === watchlistsState.active_group_id) || (watchlistsState.groups || [])[0];
}

function onWatchlistGroupChange() {
  watchlistsState.active_group_id = document.getElementById('wl_group_select').value;
  fetch('/api/watchlists', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'save_all', active_group_id: watchlistsState.active_group_id })
  });
  renderWatchlistItems();
}

function renderWatchlistItems() {
  const container = document.getElementById('wl_items_container');
  const grp = getActiveWatchlistGroup();
  if (!grp) {
    container.innerHTML = '<div style="padding:14px; color:var(--muted);">No watchlist selected.</div>';
    return;
  }
  const term = (document.getElementById('wl_filter_input').value || '').trim().toUpperCase();
  const flagFilter = document.getElementById('wl_flag_filter').value;
  const activeChartId = getActiveChartIdentifier();

  const items = (grp.items || []).filter(it => {
    if (flagFilter !== 'all' && (it.flag || 'none') !== flagFilter) return false;
    if (!term) return true;
    return (it.id || '').toUpperCase().includes(term) || (it.name || '').toUpperCase().includes(term) || (it.notes || '').toUpperCase().includes(term);
  });

  if (!items.length) {
    container.innerHTML = '<div style="padding:16px; color:var(--muted); text-align:center;">No items match filter. Use Universe Explorer or ⭐ + Watchlist to add symbols, sectors, themes, or ratios.</div>';
    return;
  }

  container.innerHTML = '';
  items.forEach(it => {
    const row = document.createElement('div');
    row.className = 'wl-item' + (it.id === activeChartId ? ' active' : '');
    const r1d = it.ret_1d;
    const r1m = it.ret_1m;
    const r1dStr = (r1d !== undefined && r1d !== null) ? `${r1d >= 0 ? '+' : ''}${r1d.toFixed(1)}%` : '';
    const r1mStr = (r1m !== undefined && r1m !== null) ? `${r1m >= 0 ? '+' : ''}${r1m.toFixed(1)}%` : '';
    const flagCls = `flag-${it.flag || 'none'}`;
    const typeCls = `type-${it.type || 'stock'}`;

    row.onclick = (e) => {
      if (e.target.closest('.del-btn')) return;
      selectChartIdentifier(it.id);
    };

    row.innerHTML = `
      <div style="display:flex; align-items:center; gap:7px; min-width:0; flex:1;">
        <span class="flag-dot ${flagCls}"></span>
        <div style="min-width:0; flex:1;">
          <div style="display:flex; align-items:center; gap:5px;">
            <span class="type-tag ${typeCls}">${(it.type || 'stk').slice(0,3)}</span>
            <strong style="color:#fff; font-size:12px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${it.short_label || it.id}</strong>
          </div>
          <div style="color:var(--muted); font-size:10px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
            ${it.notes ? '📝 ' + it.notes : (it.name || '')}
          </div>
        </div>
      </div>
      <div style="display:flex; align-items:center; gap:6px; flex-shrink:0;">
        ${it.rs_score ? `<span class="rs-badge ${it.rs_score >= 85 ? 'elite' : ''}">RS ${it.rs_score}</span>` : ''}
        <div style="text-align:right; min-width:46px;">
          <div class="${(r1d || 0) >= 0 ? 'pos' : 'neg'}" style="font-weight:700; font-size:11px;">${r1dStr}</div>
          <div style="color:var(--muted); font-size:9px;">1M: ${r1mStr || '--'}</div>
        </div>
        <button class="icon-btn del-btn" style="width:20px; height:20px; font-size:10px;" title="Remove from Watchlist" onclick="removeItemFromWatchlist('${it.id.replace(/'/g, "\\'")}')">✕</button>
      </div>
    `;
    container.appendChild(row);
  });
}

async function createNewWatchlistGroup() {
  const name = prompt('Enter name for new Watchlist Group:', '🚀 Custom Focus List');
  if (!name) return;
  const res = await fetch('/api/watchlists', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'create_group', name })
  });
  watchlistsState = await res.json();
  renderWatchlistGroupSelect();
  renderWatchlistItems();
}

async function renameCurrentWatchlistGroup() {
  const grp = getActiveWatchlistGroup();
  if (!grp) return;
  const name = prompt('Rename Watchlist Group:', grp.name);
  if (!name) return;
  const res = await fetch('/api/watchlists', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'rename_group', group_id: grp.id, name })
  });
  watchlistsState = await res.json();
  renderWatchlistGroupSelect();
}

async function deleteCurrentWatchlistGroup() {
  const grp = getActiveWatchlistGroup();
  if (!grp) return;
  if ((watchlistsState.groups || []).length <= 1) {
    alert('Cannot delete the only remaining watchlist group.');
    return;
  }
  if (!confirm(`Delete watchlist "${grp.name}"?`)) return;
  const res = await fetch('/api/watchlists', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'delete_group', group_id: grp.id })
  });
  watchlistsState = await res.json();
  renderWatchlistGroupSelect();
  renderWatchlistItems();
}

async function addCurrentChartToWatchlist() {
  const id = getActiveChartIdentifier();
  const grp = getActiveWatchlistGroup();
  if (!grp) return;
  const res = await fetch('/api/watchlists', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      action: 'add_item',
      group_id: grp.id,
      id: id,
      name: currentChartData && currentChartData.asset ? currentChartData.asset.name : id
    })
  });
  watchlistsState = await res.json();
  renderWatchlistGroupSelect();
  renderWatchlistItems();
  syncWatchlistItemNotesUI();
}

async function removeItemFromWatchlist(id) {
  const grp = getActiveWatchlistGroup();
  if (!grp) return;
  const res = await fetch('/api/watchlists', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'remove_item', group_id: grp.id, id })
  });
  watchlistsState = await res.json();
  renderWatchlistGroupSelect();
  renderWatchlistItems();
}

async function setItemFlag(flag) {
  const id = getActiveChartIdentifier();
  const grp = getActiveWatchlistGroup();
  if (!grp) return;
  const exists = (grp.items || []).some(i => i.id === id);
  if (!exists) {
    await addCurrentChartToWatchlist();
  }
  const res = await fetch('/api/watchlists', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'update_item', group_id: grp.id, id, flag })
  });
  watchlistsState = await res.json();
  renderWatchlistItems();
}

async function saveItemNotes() {
  const id = getActiveChartIdentifier();
  const grp = getActiveWatchlistGroup();
  if (!grp) return;
  const notes = document.getElementById('info_item_notes').value;
  const exists = (grp.items || []).some(i => i.id === id);
  if (!exists) {
    await addCurrentChartToWatchlist();
  }
  const res = await fetch('/api/watchlists', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'update_item', group_id: grp.id, id, notes })
  });
  watchlistsState = await res.json();
  renderWatchlistItems();
}

function syncWatchlistItemNotesUI() {
  const id = getActiveChartIdentifier();
  const grp = getActiveWatchlistGroup();
  const match = grp ? (grp.items || []).find(i => i.id === id) : null;
  document.getElementById('info_item_notes').value = match ? (match.notes || '') : '';
  document.getElementById('info_wl_membership').textContent = match ? `In "${grp.name}"` : 'Not in active list';
}

function exportWatchlistTradingView() {
  const grp = getActiveWatchlistGroup();
  if (!grp) return;
  const syms = (grp.items || [])
    .filter(it => it.type === 'stock' || it.type === 'etf')
    .map(it => it.id);
  if (!syms.length) {
    alert('No standard ticker symbols in this watchlist to export.');
    return;
  }
  const text = syms.join(', ');
  navigator.clipboard.writeText(text).then(() => {
    alert(`Copied ${syms.length} symbols from "${grp.name}" to clipboard for TradingView:\n\n${text}`);
  });
}

// ============================================================================
// Universe Explorer (SPDR ETFs / Sectors / Themes / Stocks)
// ============================================================================
function switchExplorerCategory(cat) {
  explorerCategory = cat;
  ['etf', 'sector', 'theme', 'stock'].forEach(c => {
    document.getElementById(`ex_tab_${c}`).classList.toggle('active', c === cat);
  });
  renderExplorerItems();
}

let explorerSearchTimer = null;
function onExplorerSearch() {
  clearTimeout(explorerSearchTimer);
  const q = document.getElementById('explorer_search').value.trim();
  if (explorerCategory === 'stock' && q.length >= 1) {
    explorerSearchTimer = setTimeout(async () => {
      const data = await fetch(`/api/catalog?q=${encodeURIComponent(q)}&type=stock&limit=200`).then(r => r.json());
      renderExplorerListData(data.stocks || []);
    }, 150);
  } else {
    renderExplorerItems();
  }
}

function renderExplorerItems() {
  const q = (document.getElementById('explorer_search').value || '').trim().toUpperCase();
  let list = [];
  if (explorerCategory === 'etf') list = catalogData.etfs || [];
  else if (explorerCategory === 'sector') list = catalogData.sectors || [];
  else if (explorerCategory === 'theme') list = catalogData.themes || [];
  else list = catalogData.stocks || [];

  if (q) {
    list = list.filter(it =>
      (it.id || '').toUpperCase().includes(q) ||
      (it.name || '').toUpperCase().includes(q) ||
      (it.focus || '').toUpperCase().includes(q) ||
      (it.sector || '').toUpperCase().includes(q) ||
      (it.theme || '').toUpperCase().includes(q)
    );
  }
  renderExplorerListData(list.slice(0, 200));
}

function renderExplorerListData(list) {
  const container = document.getElementById('explorer_items_container');
  container.innerHTML = '';
  const activeId = getActiveChartIdentifier();

  list.forEach(it => {
    const row = document.createElement('div');
    row.className = 'wl-item' + (it.id === activeId ? ' active' : '');
    const r1m = it.ret_1m || 0;
    const r3m = it.ret_3m || 0;
    row.onclick = (e) => {
      if (e.target.closest('.quick-add-btn')) return;
      selectChartIdentifier(it.id);
    };
    row.innerHTML = `
      <div style="min-width:0; flex:1;">
        <div style="display:flex; align-items:center; gap:5px;">
          <span class="type-tag type-${it.type}">${it.type.toUpperCase()}</span>
          <strong style="color:#fff; font-size:12px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${it.type === 'sector' || it.type === 'theme' ? it.name : it.symbol}</strong>
          ${it.member_count ? `<span style="color:var(--muted); font-size:10px;">(${it.member_count})</span>` : ''}
        </div>
        <div style="color:var(--muted); font-size:10px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
          ${it.type === 'sector' || it.type === 'theme' ? (it.archetype || '') : (it.name || '')}
        </div>
      </div>
      <div style="display:flex; align-items:center; gap:6px; flex-shrink:0;">
        <span class="rs-badge ${(it.rs_score || 0) >= 85 ? 'elite' : ''}">RS ${it.rs_score || 50}</span>
        <div style="text-align:right; min-width:48px;">
          <div class="${r1m >= 0 ? 'pos' : 'neg'}" style="font-weight:700; font-size:11px;">1M: ${r1m >= 0 ? '+' : ''}${r1m.toFixed(1)}%</div>
          <div class="${r3m >= 0 ? 'pos' : 'neg'}" style="font-size:9px;">3M: ${r3m >= 0 ? '+' : ''}${r3m.toFixed(1)}%</div>
        </div>
        <button class="icon-btn quick-add-btn" style="width:22px; height:22px;" title="Add to active Watchlist" onclick="quickAddItemToWatchlist('${it.id.replace(/'/g, "\\'")}', '${(it.name || '').replace(/'/g, "\\'")}')">⭐</button>
      </div>
    `;
    container.appendChild(row);
  });
}

async function quickAddItemToWatchlist(id, name) {
  const grp = getActiveWatchlistGroup();
  if (!grp) return;
  const res = await fetch('/api/watchlists', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'add_item', group_id: grp.id, id, name })
  });
  watchlistsState = await res.json();
  renderWatchlistGroupSelect();
  renderWatchlistItems();
}

// ============================================================================
// Universal Asset & Ratio Selection Logic
// ============================================================================
function getActiveChartIdentifier() {
  if (isRatioMode && partB) {
    return `RATIO:${partA}|${partB}`;
  }
  return partA;
}

function selectChartIdentifier(id) {
  if (id.toUpperCase().startsWith('RATIO:')) {
    const body = id.slice(6);
    const parts = body.includes('|') ? body.split('|') : body.split('/');
    partA = parts[0] || 'SPY';
    partB = parts[1] || 'SPY';
    isRatioMode = true;
  } else {
    partA = id;
    if (!isRatioMode) partB = '';
  }
  syncRatioHeaderUI();
  loadChart();
}

function toggleRatioMode() {
  isRatioMode = !isRatioMode;
  if (isRatioMode && !partB) {
    partB = (partA === 'SPY') ? 'QQQ' : 'SPY';
  }
  syncRatioHeaderUI();
  loadChart();
}

function swapRatioParts() {
  if (!isRatioMode || !partB) return;
  const tmp = partA;
  partA = partB;
  partB = tmp;
  syncRatioHeaderUI();
  loadChart();
}

function setQuickRatio(benchSym) {
  if (!benchSym) {
    isRatioMode = false;
    partB = '';
  } else {
    isRatioMode = true;
    partB = benchSym;
  }
  syncRatioHeaderUI();
  loadChart();
}

function setQuickRatioSector() {
  if (!currentChartData || !currentChartData.performance_a) return;
  const sec = currentChartData.performance_a.sector;
  if (!sec || sec === 'Unknown' || sec === 'Index / Composite') {
    alert('No specific Sector linked to this asset.');
    return;
  }
  isRatioMode = true;
  partB = `SECTOR:${sec}`;
  syncRatioHeaderUI();
  loadChart();
}

function setQuickRatioTheme() {
  if (!currentChartData || !currentChartData.performance_a) return;
  const thm = currentChartData.performance_a.theme;
  if (!thm || thm === 'General') {
    alert('No specific Theme linked to this asset.');
    return;
  }
  isRatioMode = true;
  partB = `THEME:${thm}`;
  syncRatioHeaderUI();
  loadChart();
}

function syncRatioHeaderUI() {
  const btn = document.getElementById('btn_toggle_ratio');
  const wrap = document.getElementById('ratio_controls_wrap');
  btn.classList.toggle('active', isRatioMode);
  wrap.style.display = isRatioMode ? 'flex' : 'none';

  document.querySelectorAll('.ratio-chip').forEach(c => c.classList.remove('active'));
  if (!isRatioMode) {
    document.getElementById('qchip_off')?.classList.add('active');
  } else if (partB.startsWith('SECTOR:')) {
    document.getElementById('qchip_SECTOR')?.classList.add('active');
  } else if (partB.startsWith('THEME:')) {
    document.getElementById('qchip_THEME')?.classList.add('active');
  } else {
    document.getElementById(`qchip_${partB}`)?.classList.add('active');
  }
}

// ============================================================================
// Chart Engine & Rendering
// ============================================================================
function initChartEngine() {
  const container = document.getElementById('chart_tv');
  const stage = container.parentElement;
  chart = LightweightCharts.createChart(container, {
    width: stage.clientWidth || container.clientWidth,
    height: stage.clientHeight || container.clientHeight,
    layout: {
      background: { type: 'solid', color: '#080b11' },
      textColor: '#94a3b8',
      fontSize: 11,
    },
    grid: {
      vertLines: { color: '#111827' },
      horzLines: { color: '#111827' },
    },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: {
      borderColor: '#1e293b',
      scaleMargins: { top: 0.08, bottom: 0.24 },
    },
    timeScale: {
      visible: true,
      timeVisible: false,
      secondsVisible: false,
      borderColor: '#1e293b',
      rightOffset: 8,
      barSpacing: 7,
    },
  });

  candleSeries = chart.addCandlestickSeries({
    upColor: '#10b981',
    downColor: '#ef4444',
    borderUpColor: '#10b981',
    borderDownColor: '#ef4444',
    wickUpColor: '#10b981',
    wickDownColor: '#ef4444',
  });

  // crosshairMarkerVisible: false removes the hover circles on all indicator lines
  ema8Series = chart.addLineSeries({ color: '#a855f7', lineWidth: 1, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false, visible: false });
  ema10Series = chart.addLineSeries({ color: '#38bdf8', lineWidth: 1.5, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
  ema20Series = chart.addLineSeries({ color: '#f59e0b', lineWidth: 1.5, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
  ema21Series = chart.addLineSeries({ color: '#22d3ee', lineWidth: 1.5, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false, visible: false });
  sma50Series = chart.addLineSeries({ color: '#10b981', lineWidth: 2, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
  sma200Series = chart.addLineSeries({ color: '#ef4444', lineWidth: 2, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });

  larssonFastSeries = chart.addLineSeries({ color: 'rgba(234,179,8,0.85)', lineWidth: 1, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
  larssonSlowSeries = chart.addLineSeries({ color: 'rgba(56,189,248,0.85)', lineWidth: 1, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });

  rsLineSeries = chart.addLineSeries({
    color: '#ec4899',
    lineWidth: 1.5,
    priceScaleId: 'rs_scale',
    priceLineVisible: false,
    lastValueVisible: false,
    crosshairMarkerVisible: false,
  });
  rsMaSeries = chart.addLineSeries({
    color: 'rgba(236,72,153,0.45)',
    lineWidth: 1,
    lineStyle: 2,
    priceScaleId: 'rs_scale',
    priceLineVisible: false,
    lastValueVisible: false,
    crosshairMarkerVisible: false,
  });
  chart.priceScale('rs_scale').applyOptions({
    scaleMargins: { top: 0.62, bottom: 0.25 },
    visible: false,
  });

  volumeSeries = chart.addHistogramSeries({
    priceFormat: { type: 'volume' },
    priceScaleId: 'vol_scale',
    priceLineVisible: false,
    lastValueVisible: false,
  });
  volMa20Series = chart.addLineSeries({
    color: '#fbbf24',
    lineWidth: 1,
    priceScaleId: 'vol_scale',
    priceLineVisible: false,
    lastValueVisible: false,
    crosshairMarkerVisible: false,
  });
  chart.priceScale('vol_scale').applyOptions({
    scaleMargins: { top: 0.79, bottom: 0.0 },
  });

  chart.timeScale().subscribeVisibleTimeRangeChange(() => {
    scheduleLarssonRender();
    renderAllDrawings();
  });

  chart.subscribeCrosshairMove(param => {
    const leg = document.getElementById('ohlc_legend');
    const hdrClimax = document.getElementById('hdr_climax_badge');
    if (!param || !param.time || !param.seriesData) {
      // Restore header climax badge to latest bar when cursor leaves
      if (hdrClimax && currentChartData && currentChartData.exhaustion && currentChartData.exhaustion.current_prob_pct !== undefined) {
        const exh = currentChartData.exhaustion;
        const pc = exh.pivots_current || {};
        const p5 = exh.pivots_5d_peak || {};
        const reg = exh.regime === 'downtrend' ? 'DT' : 'UT';
        const hiP = reg === 'UT' ? (pc.ut_high || 0) : (pc.dt_high || 0);
        const loP = reg === 'UT' ? (pc.ut_low || 0) : (pc.dt_low || 0);
        const max5 = Math.max(p5.ut_high || 0, p5.ut_low || 0, p5.dt_high || 0, p5.dt_low || 0);
        const colorHex = max5 >= 50 ? (exh.status === 'climax' ? '#f43f5e' : '#38bdf8') : (max5 >= 40 ? '#f59e0b' : '#10b981');
        const bgRgba = max5 >= 50 ? 'rgba(56,189,248,0.18)' : (max5 >= 40 ? 'rgba(245,158,11,0.18)' : 'rgba(16,185,129,0.15)');
        hdrClimax.style.background = bgRgba;
        hdrClimax.style.color = colorHex;
        hdrClimax.style.border = `1px solid ${colorHex}`;
        hdrClimax.textContent = `🤖 [${reg}] High: ${hiP.toFixed(0)}% | Low: ${loP.toFixed(0)}% (5D Peak: ${max5.toFixed(0)}%)`;
      }
      return;
    }
    const b = param.seriesData.get(candleSeries);
    if (!b) return;
    const chg = ((b.close / b.open - 1) * 100).toFixed(2);
    leg.innerHTML = `<span style="color:#38bdf8; font-weight:700;">${param.time}</span><span>O: <b>${b.open}</b></span><span>H: <b>${b.high}</b></span><span>L: <b>${b.low}</b></span><span>C: <b>${b.close}</b> (${chg >= 0 ? '+' : ''}${chg}%)</span>`;

    // Show hovered bar's 4-Model Pivot probabilities in header badge
    if (hdrClimax && currentChartData && currentChartData.exhaustion && currentChartData.exhaustion._byDate) {
      const pt = currentChartData.exhaustion._byDate[param.time];
      if (pt) {
        const reg = pt.regime || 'UT';
        const hiP = reg === 'UT' ? (pt.ut_high || 0) : (pt.dt_high || 0);
        const loP = reg === 'UT' ? (pt.ut_low || 0) : (pt.dt_low || 0);
        const maxBar = Math.max(hiP, loP);
        const colorHex = maxBar >= 50 ? (hiP >= loP ? '#f43f5e' : '#38bdf8') : (maxBar >= 40 ? '#f59e0b' : '#10b981');
        const bgRgba = maxBar >= 50 ? 'rgba(56,189,248,0.18)' : (maxBar >= 40 ? 'rgba(245,158,11,0.18)' : 'rgba(16,185,129,0.15)');
        hdrClimax.style.background = bgRgba;
        hdrClimax.style.color = colorHex;
        hdrClimax.style.border = `1px solid ${colorHex}`;
        hdrClimax.textContent = `🤖 [${reg} ${param.time}] ${reg} High: ${hiP.toFixed(1)}% | ${reg} Low: ${loP.toFixed(1)}%`;
      }
    }
  });

  const syncStageDimensions = () => {
    const w = stage.clientWidth;
    const h = stage.clientHeight;
    if (w > 0 && h > 0) {
      chart.applyOptions({ width: w, height: h });
      resizeCanvas();
      scheduleLarssonRender();
    }
  };

  if (typeof ResizeObserver !== 'undefined') {
    new ResizeObserver(syncStageDimensions).observe(stage);
  }
  window.addEventListener('resize', syncStageDimensions);
  setTimeout(syncStageDimensions, 30);

  initDrawingEvents();
  resizeCanvas();
}

async function loadChart() {
  const larssonSpan = document.getElementById('cfg_larsson_span').value || '32';
  const emaSpans = getConfiguredEmaSpans().join(',');
  const smaSpans = getConfiguredSmaSpans().join(',');
  const modelVer = document.getElementById('cfg_pivot_model_ver')?.value || 'v3';
  const compareParam = (isRatioMode && partB) ? `&compare=${encodeURIComponent(partB)}` : '';
  const url = `/api/chart?symbol=${encodeURIComponent(partA)}${compareParam}&history=${historyWindow}&larsson_fastest=${larssonSpan}&ema_spans=${emaSpans}&sma_spans=${smaSpans}&model_ver=${encodeURIComponent(modelVer)}`;

  const res = await fetch(url);
  const data = await res.json();
  if (data.error) {
    alert(data.error);
    return;
  }
  if (data.exhaustion && Array.isArray(data.exhaustion.series)) {
    const map = {};
    data.exhaustion.series.forEach(s => { map[s.time] = s; });
    data.exhaustion._byDate = map;
  }
  currentChartData = data;

  // Update Header Pills
  const pa = data.part_a || {};
  document.getElementById('pill_a_type').textContent = (pa.type || 'ASSET').toUpperCase();
  document.getElementById('pill_a_sym').textContent = pa.type === 'sector' || pa.type === 'theme' ? pa.key : pa.id;
  document.getElementById('pill_a_name').textContent = pa.name || '';

  if (data.is_ratio && data.part_b) {
    const pb = data.part_b;
    document.getElementById('pill_b_type').textContent = (pb.type || 'ASSET').toUpperCase();
    document.getElementById('pill_b_sym').textContent = pb.type === 'sector' || pb.type === 'theme' ? pb.key : pb.id;
    document.getElementById('pill_b_name').textContent = pb.name || '';
  }

  // Update Chart Title Bar
  const asset = data.asset || {};
  const hdrType = document.getElementById('hdr_asset_type');
  hdrType.textContent = (asset.type || 'stock').toUpperCase();
  hdrType.className = `type-tag type-${asset.type || 'stock'}`;
  document.getElementById('hdr_asset_sym').textContent = asset.key || asset.id;
  document.getElementById('hdr_asset_name').textContent = asset.name || '';

  if (data.larsson_spans) {
    document.getElementById('lbl_larsson_spans').textContent = `(${data.larsson_spans.join(', ')})`;
  }

  // Feed Series
  candleSeries.setData(data.bars || []);
  volumeSeries.setData(data.volume || []);
  volMa20Series.setData(data.vol_ma20 || []);
  const emas = data.emas || [data.ema8 || [], data.ema10 || [], data.ema20 || [], data.ema21 || []];
  const smas = data.smas || [data.sma50 || [], data.sma200 || []];
  ema8Series.setData(emas[0] || []);
  ema10Series.setData(emas[1] || []);
  ema20Series.setData(emas[2] || []);
  ema21Series.setData(emas[3] || []);
  sma50Series.setData(smas[0] || []);
  sma200Series.setData(smas[1] || []);

  const lFast = (data.larsson || []).filter(d => d.fast !== null).map(d => ({ time: d.time, value: d.fast }));
  const lSlow = (data.larsson || []).filter(d => d.slow !== null).map(d => ({ time: d.time, value: d.slow }));
  larssonFastSeries.setData(lFast);
  larssonSlowSeries.setData(lSlow);

  const rsPoints = (data.rs_line || []).map(d => ({ time: d.time, value: d.value }));
  const rsMaPoints = (data.rs_line || []).map(d => ({ time: d.time, value: d.ma }));
  rsLineSeries.setData(rsPoints);
  rsMaSeries.setData(rsMaPoints);

  updateIndicatorVisibility();
  renderEPMarkers();
  chart.timeScale().fitContent();
  setTimeout(scheduleLarssonRender, 40);

  // Update Info Panel & Annotations
  renderInfoPanel(data);
  syncWatchlistItemNotesUI();
  renderWatchlistItems();
  loadAnnotations(getActiveChartIdentifier());
}

function reloadChart() {
  loadChart();
}

function setHistoryWindow(win) {
  historyWindow = win;
  ['3m', '6m', '1y', '2y', '5y', 'full'].forEach(w => {
    document.getElementById(`tf_${w}`)?.classList.toggle('active', w === win);
  });
  loadChart();
}

function resetChartZoom() {
  if (chart) chart.timeScale().fitContent();
  scheduleLarssonRender();
}

function resolvePivotTriggerPrice(ce, thr) {
  const byThr = ce.trigger_prices_by_thr || {};
  const exactKey = String(Math.round(thr));
  if (byThr[exactKey] !== undefined) return Number(byThr[exactKey]);
  const loStep = Math.floor(thr / 5) * 5;
  const hiStep = Math.ceil(thr / 5) * 5;
  const pLo = byThr[String(loStep)];
  const pHi = byThr[String(hiStep)];
  if (pLo !== undefined && pHi !== undefined && hiStep > loStep) {
    const frac = (thr - loStep) / (hiStep - loStep);
    return Number(pLo) + frac * (Number(pHi) - Number(pLo));
  }
  const nearKey = String(Math.round(thr / 5) * 5);
  if (byThr[nearKey] !== undefined) return Number(byThr[nearKey]);
  return ce.intrabar_trigger_price !== undefined ? Number(ce.intrabar_trigger_price) : NaN;
}

function fmtPivotIntrabarSuffix(ce, thr, showIntrabar) {
  if (!showIntrabar) return '';
  const numP = resolvePivotTriggerPrice(ce, thr);
  if (isNaN(numP)) return '';
  const pStr = Math.abs(numP) < 10 ? numP.toFixed(3) : numP.toFixed(2);
  return ` @$${pStr}`;
}

function updateIntrabarTriggerRow(exh) {
  const trigRowEl = document.getElementById('exh_intrabar_trig_row');
  if (!trigRowEl || !exh) return;
  const threshEl = document.getElementById('cfg_climax_thresh');
  const thr = threshEl ? Math.max(20, Math.min(90, parseFloat(threshEl.value) || 50)) : 50;
  const allEvs = [
    ...(exh.ut_high_events || []),
    ...(exh.ut_low_events || []),
    ...(exh.dt_high_events || []),
    ...(exh.dt_low_events || []),
  ].filter(e => e.prob_pct >= thr).sort((a, b) => a.date.localeCompare(b.date));
  if (allEvs.length > 0) {
    const lastEv = allEvs[allEvs.length - 1];
    const isHi = lastEv.type === 'ut_high' || lastEv.type === 'dt_high';
    const trigP = resolvePivotTriggerPrice(lastEv, thr);
    const extP = isHi ? lastEv.high : lastEv.low;
    const dec = Math.abs(Number(trigP)) < 10 ? 3 : 2;
    const lblMap = { ut_high: 'UT High', ut_low: 'UT Low', dt_high: 'DT High', dt_low: 'DT Low' };
    trigRowEl.textContent = `${lastEv.date} ${lblMap[lastEv.type] || lastEv.type}: @$${Number(trigP).toFixed(dec)} (${isHi ? 'Hi' : 'Lo'} $${Number(extP).toFixed(dec)})`;
  } else {
    trigRowEl.textContent = 'None in view';
  }
}

function renderEPMarkers() {
  if (!candleSeries || !currentChartData) return;
  const showEP = document.getElementById('cfg_ep_markers').checked;
  const showPivotHighs = document.getElementById('cfg_climax_markers')?.checked;
  const showPivotLows = document.getElementById('cfg_pivot_low_markers')?.checked;
  const showIntrabarPrice = document.getElementById('cfg_pivot_intrabar_price')?.checked ?? true;
  const threshEl = document.getElementById('cfg_climax_thresh');
  const climaxThresh = threshEl ? Math.max(20, Math.min(90, parseFloat(threshEl.value) || 50)) : 50;
  if (threshEl) {
    try { localStorage.setItem('ep_wl_climax_thresh', String(climaxThresh)); } catch (e) {}
  }
  try {
    localStorage.setItem('ep_wl_pivot_intrabar', showIntrabarPrice ? '1' : '0');
  } catch (e) {}
  const eps = currentChartData.ep_events || [];
  const exh = currentChartData.exhaustion || {};
  const utHighEvents = (exh.ut_high_events || exh.climax_events || []).filter(ce => ce.prob_pct >= climaxThresh);
  const utLowEvents = (exh.ut_low_events || []).filter(ce => ce.prob_pct >= climaxThresh);
  const dtHighEvents = (exh.dt_high_events || []).filter(ce => ce.prob_pct >= climaxThresh);
  const dtLowEvents = (exh.dt_low_events || []).filter(ce => ce.prob_pct >= climaxThresh);
  const badge = document.getElementById('hdr_ep_count_badge');

  if (eps.length > 0) {
    badge.style.display = 'inline-block';
    badge.textContent = `⚡ ${eps.length} Strong EP${eps.length > 1 ? 's' : ''}`;
  } else {
    badge.style.display = 'none';
  }

  const allMarkers = [];

  if (showEP && eps.length) {
    const byDate = {};
    eps.forEach(ep => {
      if (!byDate[ep.date]) byDate[ep.date] = [];
      byDate[ep.date].push(ep);
    });

    Object.keys(byDate).sort().forEach(dt => {
      const list = byDate[dt];
      const top = list[0];
      if (top.is_group_event) {
        const lbl = list.length > 1
          ? `⚡ ${list.length} Group EPs (${top.symbol} +${top.gap_pct}%)`
          : `⚡ ${top.symbol} EP +${top.gap_pct}% (${top.rvol}x)`;
        allMarkers.push({
          time: dt,
          position: 'belowBar',
          color: '#38bdf8',
          shape: 'arrowUp',
          text: lbl,
        });
      } else {
        const isT1 = top.tier_badge === 'Tier 1 Ultra';
        allMarkers.push({
          time: dt,
          position: 'belowBar',
          color: isT1 ? '#f59e0b' : (top.is_strong ? '#10b981' : '#38bdf8'),
          shape: 'arrowUp',
          text: `${isT1 ? '🔥 T1 EP' : '⚡ EP'} +${top.gap_pct}% (${top.rvol}x)`,
        });
      }
    });
  }

  // 1. Uptrend Pivot Highs (🔥 UT High) & 3. Downtrend Pivot Highs (🔻 DT High)
  if (showPivotHighs) {
    utHighEvents.forEach(ce => {
      const isExtreme = ce.prob_pct >= 65.0;
      const trigSfx = fmtPivotIntrabarSuffix(ce, climaxThresh, showIntrabarPrice);
      allMarkers.push({
        time: ce.date,
        position: 'aboveBar',
        color: isExtreme ? '#f43f5e' : '#fb923c',
        shape: 'arrowDown',
        text: `🔥 UT High ${ce.prob_pct.toFixed(0)}%${trigSfx}`,
      });
    });
    dtHighEvents.forEach(ce => {
      const trigSfx = fmtPivotIntrabarSuffix(ce, climaxThresh, showIntrabarPrice);
      allMarkers.push({
        time: ce.date,
        position: 'aboveBar',
        color: '#c084fc',
        shape: 'arrowDown',
        text: `🔻 DT High ${ce.prob_pct.toFixed(0)}%${trigSfx}`,
      });
    });
  }

  // 2. Uptrend Pivot Lows (💎 UT Low) & 4. Downtrend Pivot Lows (🟢 DT Low)
  if (showPivotLows) {
    utLowEvents.forEach(ce => {
      const trigSfx = fmtPivotIntrabarSuffix(ce, climaxThresh, showIntrabarPrice);
      allMarkers.push({
        time: ce.date,
        position: 'belowBar',
        color: '#38bdf8',
        shape: 'arrowUp',
        text: `💎 UT Low ${ce.prob_pct.toFixed(0)}%${trigSfx}`,
      });
    });
    dtLowEvents.forEach(ce => {
      const trigSfx = fmtPivotIntrabarSuffix(ce, climaxThresh, showIntrabarPrice);
      allMarkers.push({
        time: ce.date,
        position: 'belowBar',
        color: '#22c55e',
        shape: 'arrowUp',
        text: `🟢 DT Low ${ce.prob_pct.toFixed(0)}%${trigSfx}`,
      });
    });
  }

  allMarkers.sort((a, b) => a.time.localeCompare(b.time));
  candleSeries.setMarkers(allMarkers);
  updateIntrabarTriggerRow(exh);
}

// ============================================================================
// Larsson Ribbon Polygon Renderer
// ============================================================================
function scheduleLarssonRender() {
  requestAnimationFrame(drawLarssonRibbon);
}

function drawLarssonRibbon() {
  const svg = document.getElementById('larsson_ribbon_svg');
  const showL = document.getElementById('cfg_larsson').checked;
  if (!svg || !chart || !currentChartData || !showL) {
    if (svg) svg.innerHTML = '';
    return;
  }
  const lData = currentChartData.larsson || [];
  if (lData.length < 2) {
    svg.innerHTML = '';
    return;
  }
  const ts = chart.timeScale();
  const pts = [];
  for (let i = 0; i < lData.length; i++) {
    const d = lData[i];
    if (d.fast === null || d.slow === null) continue;
    const x = ts.timeToCoordinate(d.time);
    if (x === null) continue;
    const yF = larssonFastSeries.priceToCoordinate(d.fast);
    const yS = larssonSlowSeries.priceToCoordinate(d.slow);
    if (yF === null || yS === null) continue;
    pts.push({ x, yF, yS, state: d.state });
  }
  if (pts.length < 2) {
    svg.innerHTML = '';
    return;
  }
  let html = '';
  for (let i = 0; i < pts.length - 1; i++) {
    const p1 = pts[i], p2 = pts[i + 1];
    let fill = 'rgba(100,116,139,0.14)';
    if (p1.state === 'yellow') fill = 'rgba(234,179,8,0.24)';
    else if (p1.state === 'blue') fill = 'rgba(56,189,248,0.24)';
    html += `<path d="M ${p1.x} ${p1.yF} L ${p2.x} ${p2.yF} L ${p2.x} ${p2.yS} L ${p1.x} ${p1.yS} Z" fill="${fill}"/>`;
  }
  svg.innerHTML = html;
}

function updateIndicatorVisibility() {
  if (!chart) return;
  ema8Series.applyOptions({ visible: document.getElementById('cfg_ema8').checked });
  ema10Series.applyOptions({ visible: document.getElementById('cfg_ema10').checked });
  ema20Series.applyOptions({ visible: document.getElementById('cfg_ema20').checked });
  ema21Series.applyOptions({ visible: document.getElementById('cfg_ema21').checked });
  sma50Series.applyOptions({ visible: document.getElementById('cfg_sma50').checked });
  sma200Series.applyOptions({ visible: document.getElementById('cfg_sma200').checked });

  const showVol = document.getElementById('cfg_volume').checked;
  volumeSeries.applyOptions({ visible: showVol });
  volMa20Series.applyOptions({ visible: showVol });

  const showRS = document.getElementById('cfg_rs_line').checked;
  rsLineSeries.applyOptions({ visible: showRS });
  rsMaSeries.applyOptions({ visible: showRS });

  const showL = document.getElementById('cfg_larsson').checked;
  larssonFastSeries.applyOptions({ visible: showL });
  larssonSlowSeries.applyOptions({ visible: showL });
  scheduleLarssonRender();
}

// ============================================================================
// Right Pane: Info & Performance Panel Renderer
// ============================================================================
function fmtPct(v) {
  if (v === null || v === undefined) return '<span class="neu">--</span>';
  const cls = v > 0 ? 'pos' : (v < 0 ? 'neg' : 'neu');
  return `<span class="${cls}">${v > 0 ? '+' : ''}${v.toFixed(2)}%</span>`;
}

function renderInfoPanel(data) {
  const pf = data.performance || {};
  const exh = data.exhaustion || {};

  // Render Universal 4-Model Structural Swing-Pivot Card + Header Badge
  const hdrClimax = document.getElementById('hdr_climax_badge');
  const exhStatusBadge = document.getElementById('exh_status_badge');
  if (exh && exh.current_prob_pct !== undefined) {
    const pc = exh.pivots_current || {};
    const p5 = exh.pivots_5d_peak || {};
    const uh = pc.ut_high !== undefined ? pc.ut_high : exh.current_prob_pct;
    const ul = pc.ut_low || 0;
    const dh = pc.dt_high || 0;
    const dl = pc.dt_low || 0;

    const uh5 = p5.ut_high !== undefined ? p5.ut_high : exh.peak_5d_prob_pct;
    const ul5 = p5.ut_low || 0;
    const dh5 = p5.dt_high || 0;
    const dl5 = p5.dt_low || 0;
    const max5 = Math.max(uh5, ul5, dh5, dl5);

    const reg = exh.regime === 'downtrend' ? 'DOWNTREND' : 'UPTREND';
    const regShort = exh.regime === 'downtrend' ? 'DT' : 'UT';
    const isClimax = exh.status === 'climax';
    const isPivLow = exh.status === 'pivot_low';
    const isElev = exh.status === 'elevated';
    const colorHex = isClimax ? '#f43f5e' : (isPivLow ? '#38bdf8' : (isElev ? '#f59e0b' : '#10b981'));
    const bgRgba = isClimax ? 'rgba(244,63,94,0.18)' : (isPivLow ? 'rgba(56,189,248,0.18)' : (isElev ? 'rgba(245,158,11,0.18)' : 'rgba(16,185,129,0.15)'));

    if (hdrClimax) {
      hdrClimax.style.display = 'inline-block';
      hdrClimax.style.background = bgRgba;
      hdrClimax.style.color = colorHex;
      hdrClimax.style.border = `1px solid ${colorHex}`;
      const hiNow = regShort === 'UT' ? uh : dh;
      const loNow = regShort === 'UT' ? ul : dl;
      hdrClimax.textContent = `🤖 [${regShort}] High: ${hiNow.toFixed(0)}% | Low: ${loNow.toFixed(0)}% (5D Peak: ${max5.toFixed(0)}%)`;
    }

    const verTag = (exh.model_ver || 'v3').toUpperCase();
    exhStatusBadge.textContent = `${reg} · ${verTag}`;
    exhStatusBadge.style.background = exh.regime === 'downtrend' ? 'rgba(239,68,68,0.18)' : 'rgba(16,185,129,0.18)';
    exhStatusBadge.style.color = exh.regime === 'downtrend' ? '#f87171' : '#34d399';
    exhStatusBadge.style.border = `1px solid ${exh.regime === 'downtrend' ? '#ef4444' : '#10b981'}`;

    document.getElementById('exh_curr_val').innerHTML = `<span style="color:${uh >= 50 ? '#f43f5e' : '#cbd5e1'};">${uh.toFixed(1)}%</span>`;
    document.getElementById('exh_5d_val').textContent = `5D: ${uh5.toFixed(0)}%`;

    const ulEl = document.getElementById('piv_ut_low_val');
    const ul5El = document.getElementById('piv_ut_low_5d');
    if (ulEl) ulEl.innerHTML = `<span style="color:${ul >= 50 ? '#38bdf8' : '#cbd5e1'};">${ul.toFixed(1)}%</span>`;
    if (ul5El) ul5El.textContent = `5D: ${ul5.toFixed(0)}%`;

    const dhEl = document.getElementById('piv_dt_high_val');
    const dh5El = document.getElementById('piv_dt_high_5d');
    if (dhEl) dhEl.innerHTML = `<span style="color:${dh >= 50 ? '#c084fc' : '#cbd5e1'};">${dh.toFixed(1)}%</span>`;
    if (dh5El) dh5El.textContent = `5D: ${dh5.toFixed(0)}%`;

    const dlEl = document.getElementById('piv_dt_low_val');
    const dl5El = document.getElementById('piv_dt_low_5d');
    if (dlEl) dlEl.innerHTML = `<span style="color:${dl >= 50 ? '#4ade80' : '#cbd5e1'};">${dl.toFixed(1)}%</span>`;
    if (dl5El) dl5El.textContent = `5D: ${dl5.toFixed(0)}%`;

    const barEl = document.getElementById('exh_progress_bar');
    barEl.style.width = `${Math.min(100, Math.max(4, max5))}%`;
    barEl.style.background = colorHex;

    const bannerEl = document.getElementById('exh_action_banner');
    bannerEl.textContent = exh.action_label || '';
    bannerEl.style.color = colorHex;
    bannerEl.style.background = bgRgba;
    bannerEl.style.border = `1px solid ${colorHex}`;

    const subRowEl = document.getElementById('exh_submodels_row');
    if (subRowEl) {
      const ah = exh.at_high_prob_pct !== undefined ? exh.at_high_prob_pct : uh;
      const fb = exh.full_bar_prob_pct !== undefined ? exh.full_bar_prob_pct : uh;
      subRowEl.innerHTML = `<span class="${ah >= 50 ? 'neg' : 'neu'}">${ah.toFixed(1)}%</span> / <span class="${fb >= 50 ? 'neg' : 'neu'}">${fb.toFixed(1)}%</span>`;
    }
    updateIntrabarTriggerRow(exh);
    const rsiLow = exh.rsi14_low !== undefined ? exh.rsi14_low : exh.rsi5;
    document.getElementById('exh_rsi_row').innerHTML = `<span class="${exh.rsi14 >= 72 ? 'neg' : 'neu'}">${exh.rsi14}</span> (Hi) / <span class="${rsiLow <= 32 ? 'pos' : 'neu'}">${rsiLow}</span> (Lo)`;
    document.getElementById('exh_adr_stretch').textContent = `${exh.ext_ema10_adr >= 0 ? '+' : ''}${exh.ext_ema10_adr}x / ${exh.ext_ema20_adr >= 0 ? '+' : ''}${exh.ext_ema20_adr}x / ${exh.ext_sma50_adr >= 0 ? '+' : ''}${exh.ext_sma50_adr}x ADR`;
    const loWick = exh.lower_wick_pct !== undefined ? exh.lower_wick_pct : 0;
    document.getElementById('exh_wick_rvol').textContent = `${exh.upper_wick_pct}% up / ${loWick}% lo | ${exh.rvol}x RVOL`;
    const divTxt = exh.bearish_div ? '<span class="neg">⚠️ Bear Div</span>' : (exh.bullish_div ? '<span class="pos">💎 Bull Div</span>' : '<span class="neu">None</span>');
    document.getElementById('exh_consec_div').innerHTML = `${exh.consec_rsi_70}d RSI≥70 | ${divTxt}`;
    document.getElementById('exh_anchor_mode').textContent = exh.anchor_mode || '20E Swing Leg';
  } else if (hdrClimax) {
    hdrClimax.style.display = 'none';
  }

  document.getElementById('perf_card_title').textContent = data.is_ratio ? '📊 Ratio Performance & Posture' : '📊 Performance & Relative Strength';
  const rsBadge = document.getElementById('info_rs_badge');
  if (pf.rs_score !== null && pf.rs_score !== undefined) {
    rsBadge.textContent = `RS ${pf.rs_score}`;
    rsBadge.className = `rs-badge ${pf.rs_score >= 85 ? 'elite' : 'strong'}`;
  } else {
    rsBadge.textContent = data.is_ratio ? 'RATIO' : 'INDEX';
    rsBadge.className = 'rs-badge';
  }

  document.getElementById('pf_1d').innerHTML = fmtPct(pf.ret_1d);
  document.getElementById('pf_1w').innerHTML = fmtPct(pf.ret_1w);
  document.getElementById('pf_1m').innerHTML = fmtPct(pf.ret_1m);
  document.getElementById('pf_3m').innerHTML = fmtPct(pf.ret_3m);
  document.getElementById('pf_6m').innerHTML = fmtPct(pf.ret_6m);
  document.getElementById('pf_ytd').innerHTML = fmtPct(pf.ret_ytd);
  document.getElementById('pf_1y').innerHTML = fmtPct(pf.ret_1y);
  document.getElementById('pf_adr').textContent = pf.adr_pct ? `${pf.adr_pct.toFixed(2)}%` : '--';

  document.getElementById('pf_alpha_spy').innerHTML = `${fmtPct(pf.alpha_spy_1m)} / ${fmtPct(pf.alpha_spy_3m)} / ${fmtPct(pf.alpha_spy_1y)}`;
  document.getElementById('pf_alpha_qqq').innerHTML = fmtPct(pf.alpha_qqq_3m);
  document.getElementById('pf_52w_hi').innerHTML = `${pf.hi_52w || '--'} (${fmtPct(pf.dist_52w_high)})`;
  document.getElementById('pf_dvol').textContent = pf.dvol_m ? `$${pf.dvol_m.toLocaleString()}M / day` : '--';
  document.getElementById('pf_stage2').innerHTML = pf.stage2
    ? '<span class="pos">✅ Stage 2 Uptrend (Above 20/50/200)</span>'
    : '<span class="neg">⚠️ Below Stage 2 Alignment</span>';
  document.getElementById('pf_ma_dists').innerHTML = `${fmtPct(pf[' dist_ema10_pct'])} / ${fmtPct(pf.dist_ema21_pct)} / ${fmtPct(pf.dist_sma50_pct)}`;

  // Ratio Side-by-Side Card
  const ratioCard = document.getElementById('ratio_compare_card');
  if (data.is_ratio && data.performance_a && data.performance_b) {
    ratioCard.style.display = 'flex';
    const a = data.performance_a;
    const b = data.performance_b;
    document.getElementById('ratio_compare_body').innerHTML = `
      <table style="width:100%; border-collapse:collapse; font-size:11px;">
        <thead>
          <tr style="border-bottom:1px solid #1e293b; color:#94a3b8; text-align:right;">
            <th style="text-align:left; padding:4px 0;">Metric</th>
            <th style="padding:4px; color:#38bdf8;">A: ${a.symbol}</th>
            <th style="padding:4px; color:#c084fc;">B: ${b.symbol}</th>
            <th style="padding:4px; color:#34d399;">Spread (A-B)</th>
          </tr>
        </thead>
        <tbody>
          <tr style="text-align:right;"><td style="text-align:left; padding:3px 0; color:#94a3b8;">1D Return</td><td>${fmtPct(a.ret_1d)}</td><td>${fmtPct(b.ret_1d)}</td><td>${fmtPct((a.ret_1d||0)-(b.ret_1d||0))}</td></tr>
          <tr style="text-align:right;"><td style="text-align:left; padding:3px 0; color:#94a3b8;">1M Return</td><td>${fmtPct(a.ret_1m)}</td><td>${fmtPct(b.ret_1m)}</td><td>${fmtPct((a.ret_1m||0)-(b.ret_1m||0))}</td></tr>
          <tr style="text-align:right;"><td style="text-align:left; padding:3px 0; color:#94a3b8;">3M Return</td><td>${fmtPct(a.ret_3m)}</td><td>${fmtPct(b.ret_3m)}</td><td>${fmtPct((a.ret_3m||0)-(b.ret_3m||0))}</td></tr>
          <tr style="text-align:right;"><td style="text-align:left; padding:3px 0; color:#94a3b8;">YTD Return</td><td>${fmtPct(a.ret_ytd)}</td><td>${fmtPct(b.ret_ytd)}</td><td>${fmtPct((a.ret_ytd||0)-(b.ret_ytd||0))}</td></tr>
          <tr style="text-align:right;"><td style="text-align:left; padding:3px 0; color:#94a3b8;">1Y Return</td><td>${fmtPct(a.ret_1y)}</td><td>${fmtPct(b.ret_1y)}</td><td>${fmtPct((a.ret_1y||0)-(b.ret_1y||0))}</td></tr>
          <tr style="text-align:right;"><td style="text-align:left; padding:3px 0; color:#94a3b8;">RS Score</td><td><b>${a.rs_score || '--'}</b></td><td><b>${b.rs_score || '--'}</b></td><td>--</td></tr>
        </tbody>
      </table>
    `;
  } else {
    ratioCard.style.display = 'none';
  }

  // Recent Strong EP Events Card
  const eps = (data.ep_events || []).slice().reverse();
  document.getElementById('ep_card_count').textContent = eps.length;
  const epList = document.getElementById('ep_events_list');
  if (!eps.length) {
    epList.innerHTML = '<div style="color:var(--muted); font-size:11px;">No qualifying EP events in current window.</div>';
  } else {
    epList.innerHTML = '';
    eps.slice(0, 15).forEach(ep => {
      const div = document.createElement('div');
      div.className = 'ep-item' + (ep.tier_badge === 'Tier 1 Ultra' ? ' tier1' : '');
      div.onclick = () => {
        if (ep.is_group_event && ep.symbol) {
          selectChartIdentifier(ep.symbol);
        }
      };
      div.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <div>
            <strong style="color:#fff;">${ep.symbol}</strong>
            <span style="color:#34d399; font-weight:700; margin-left:5px;">${ep.date}</span>
          </div>
          <span style="font-size:9px; font-weight:800; padding:1px 5px; border-radius:3px; background:rgba(16,185,129,0.2); color:#34d399;">${ep.tier_badge}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-top:3px; font-size:11px; color:#cbd5e1;">
          <span>Gap: <b>+${ep.gap_pct}%</b> | RVOL: <b>${ep.rvol}x</b></span>
          <span>${ep.v2_prob_50 !== null && ep.v2_prob_50 !== undefined ? `V2 P(50%): <b>${ep.v2_prob_50}%</b>` : (ep.max_gain_20d !== null ? `20D Peak: +${ep.max_gain_20d}%` : '')}</span>
        </div>
      `;
      epList.appendChild(div);
    });
    if (data.catalyst && data.catalyst.headline) {
      const catDiv = document.createElement('div');
      catDiv.style.cssText = 'font-size:11px; color:#cbd5e1; background:#0b1019; padding:6px; border-radius:4px; border:1px solid #1e293b;';
      catDiv.innerHTML = `<strong>Latest Catalyst:</strong> ${data.catalyst.headline}`;
      epList.prepend(catDiv);
    }
  }

  // Sector & Theme Context + Constituents
  const ctx = data.context || {};
  const ctxBody = document.getElementById('group_context_body');
  let ctxHtml = '';
  if (ctx.sector_summary) {
    const s = ctx.sector_summary;
    ctxHtml += `
      <div style="background:#0b1019; padding:7px; border-radius:5px; border:1px solid #1e293b; cursor:pointer;" onclick="selectChartIdentifier('${s.id.replace(/'/g, "\\'")}')">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span style="color:#34d399; font-weight:800;">🏭 Sector: ${s.name}</span>
          <span class="rs-badge">Score ${s.score}</span>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:10px; color:#94a3b8; margin-top:4px;">
          <span>1M: ${fmtPct(s.m1_ret)} | 3M: ${fmtPct(s.m3_ret)} | YTD: ${fmtPct(s.ytd_ret)}</span>
          <span style="color:#38bdf8;">View Sector Chart →</span>
        </div>
      </div>
    `;
  }
  if (ctx.theme_summary) {
    const t = ctx.theme_summary;
    ctxHtml += `
      <div style="background:#0b1019; padding:7px; border-radius:5px; border:1px solid #1e293b; cursor:pointer;" onclick="selectChartIdentifier('${t.id.replace(/'/g, "\\'")}')">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span style="color:#c084fc; font-weight:800;">🔥 Theme: ${t.name}</span>
          <span class="rs-badge">Score ${t.score}</span>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:10px; color:#94a3b8; margin-top:4px;">
          <span>1M: ${fmtPct(t.m1_ret)} | 3M: ${fmtPct(t.m3_ret)} | ${t.archetype}</span>
          <span style="color:#38bdf8;">View Theme Chart →</span>
        </div>
      </div>
    `;
  }
  ctxBody.innerHTML = ctxHtml || '<div style="color:var(--muted); font-size:11px;">Broad Market / Multi-Sector Index</div>';

  const constList = document.getElementById('constituents_list');
  const peers = ctx.constituents || [];
  if (!peers.length) {
    constList.innerHTML = '<div style="color:var(--muted); font-size:11px;">No constituent stocks mapped.</div>';
  } else {
    constList.innerHTML = '';
    peers.forEach(p => {
      const row = document.createElement('div');
      row.className = 'picker-row';
      row.style.padding = '5px 8px';
      row.onclick = () => selectChartIdentifier(p.symbol);
      row.innerHTML = `
        <div>
          <strong style="color:#fff;">${p.symbol}</strong>
          <span style="color:var(--muted); font-size:10px; margin-left:5px;">${(p.name || '').slice(0, 22)}</span>
        </div>
        <div style="display:flex; align-items:center; gap:6px;">
          <span class="rs-badge ${p.rs_score >= 85 ? 'elite' : ''}">RS ${p.rs_score}</span>
          <span style="min-width:46px; text-align:right; font-weight:700;">${fmtPct(p.ret_1m)}</span>
        </div>
      `;
      constList.appendChild(row);
    });
  }
}

// ============================================================================
// Universal Asset Chooser Modal (Stocks, SPDR ETFs, Sectors, Themes)
// ============================================================================
function openAssetModal(leg) {
  modalTargetLeg = leg;
  document.getElementById('modal_picker_title').textContent = leg === 'a'
    ? 'Select Primary Asset / Numerator (Stock, SPDR ETF, Sector, or Theme)'
    : 'Select Ratio Denominator — Part B (Stock, SPDR ETF, Sector, or Theme)';
  document.getElementById('asset_modal').style.display = 'flex';
  document.getElementById('modal_search_input').value = '';
  renderModalResults();
  document.getElementById('modal_search_input').focus();
}

function closeAssetModal() {
  document.getElementById('asset_modal').style.display = 'none';
}

function setModalCategory(cat) {
  modalCategory = cat;
  ['all', 'etf', 'sector', 'theme', 'stock'].forEach(c => {
    document.getElementById(`mtab_${c}`).classList.toggle('active', c === cat);
  });
  renderModalResults();
}

let modalSearchTimer = null;
function renderModalResults() {
  clearTimeout(modalSearchTimer);
  const q = (document.getElementById('modal_search_input').value || '').trim();
  if (q.length >= 1) {
    modalSearchTimer = setTimeout(async () => {
      const data = await fetch(`/api/catalog?q=${encodeURIComponent(q)}&type=${modalCategory}&limit=120`).then(r => r.json());
      populateModalList(data);
    }, 120);
  } else {
    populateModalList(catalogData);
  }
}

function populateModalList(data) {
  const listEl = document.getElementById('modal_results_list');
  listEl.innerHTML = '';
  let combined = [];
  if (modalCategory === 'all' || modalCategory === 'etf') combined.push(...(data.etfs || []).slice(0, 35));
  if (modalCategory === 'all' || modalCategory === 'sector') combined.push(...(data.sectors || []).slice(0, 40));
  if (modalCategory === 'all' || modalCategory === 'theme') combined.push(...(data.themes || []).slice(0, 40));
  if (modalCategory === 'all' || modalCategory === 'stock') combined.push(...(data.stocks || []).slice(0, 80));

  combined.slice(0, 120).forEach(it => {
    const row = document.createElement('div');
    row.className = 'picker-row';
    row.onclick = () => {
      if (modalTargetLeg === 'a') {
        partA = it.id;
      } else {
        partB = it.id;
        isRatioMode = true;
      }
      closeAssetModal();
      syncRatioHeaderUI();
      loadChart();
    };
    const r1m = it.ret_1m || 0;
    const r3m = it.ret_3m || 0;
    row.innerHTML = `
      <div style="display:flex; align-items:center; gap:8px;">
        <span class="type-tag type-${it.type}">${it.type.toUpperCase()}</span>
        <div>
          <strong style="color:#fff; font-size:13px;">${it.type === 'sector' || it.type === 'theme' ? it.name : it.symbol}</strong>
          <span style="color:var(--muted); font-size:11px; margin-left:6px;">${it.type === 'sector' || it.type === 'theme' ? `(${it.member_count || 0} members)` : (it.name || '')}</span>
        </div>
      </div>
      <div style="display:flex; align-items:center; gap:10px;">
        <span class="rs-badge ${(it.rs_score || 0) >= 85 ? 'elite' : ''}">RS ${it.rs_score || 50}</span>
        <span style="font-size:11px;">1M: ${fmtPct(r1m)}</span>
        <span style="font-size:11px;">3M: ${fmtPct(r3m)}</span>
      </div>
    `;
    listEl.appendChild(row);
  });
}

// ============================================================================
// Panel Toggles & Fullscreen
// ============================================================================
function toggleInfoPanel() {
  const ws = document.getElementById('main_workspace');
  ws.classList.toggle('info-closed');
  document.getElementById('btn_toggle_info').classList.toggle('active', !ws.classList.contains('info-closed'));
  setTimeout(() => {
    const c = document.getElementById('chart_tv');
    chart.applyOptions({ width: c.clientWidth, height: c.clientHeight });
    resizeCanvas();
    scheduleLarssonRender();
  }, 50);
}

function toggleSettingsBar() {
  const bar = document.getElementById('settings_bar');
  bar.style.display = bar.style.display === 'none' ? 'flex' : 'none';
  document.getElementById('btn_settings_bar').classList.toggle('active', bar.style.display !== 'none');
}

function toggleDrawBar() {
  const bar = document.getElementById('draw_bar');
  bar.style.display = bar.style.display === 'none' ? 'flex' : 'none';
  document.getElementById('btn_draw_bar').classList.toggle('active', bar.style.display !== 'none');
}

function toggleFullscreenChart() {
  const ws = document.getElementById('main_workspace');
  ws.classList.toggle('fullscreen-chart');
  const btn = document.getElementById('btn_fullscreen');
  const isFull = ws.classList.contains('fullscreen-chart');
  btn.classList.toggle('active', isFull);
  btn.textContent = isFull ? '🗗 Restore' : '⛶ Enlarge';
  setTimeout(() => {
    const c = document.getElementById('chart_tv');
    chart.applyOptions({ width: c.clientWidth, height: c.clientHeight });
    resizeCanvas();
    scheduleLarssonRender();
  }, 50);
}

// ============================================================================
// Interactive Drawing Tools Canvas
// ============================================================================
function resizeCanvas() {
  const canvas = document.getElementById('draw_canvas');
  if (!canvas || !canvas.parentElement) return;
  canvas.width = canvas.parentElement.clientWidth;
  canvas.height = canvas.parentElement.clientHeight;
  renderAllDrawings();
}

function setDrawTool(tool) {
  drawTool = tool;
  ['cursor', 'line', 'ray', 'hline', 'box', 'fib', 'measure'].forEach(t => {
    document.getElementById(`dtool_${t}`)?.classList.toggle('active', t === tool);
  });
  const canvas = document.getElementById('draw_canvas');
  canvas.style.pointerEvents = tool === 'cursor' ? 'none' : 'auto';
}

function initDrawingEvents() {
  const canvas = document.getElementById('draw_canvas');
  canvas.addEventListener('mousedown', e => {
    if (drawTool === 'cursor') return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const price = candleSeries.coordinateToPrice(y);
    const time = chart.timeScale().coordinateToTime(x);

    if (drawTool === 'hline') {
      drawings.push({ type: 'hline', price, color: '#38bdf8' });
      saveAnnotations();
      renderAllDrawings();
      setDrawTool('cursor');
      return;
    }

    isDrawing = true;
    currentDrawing = { type: drawTool, x1: x, y1: y, t1: time, p1: price, x2: x, y2: y, t2: time, p2: price };
  });

  canvas.addEventListener('mousemove', e => {
    if (!isDrawing || !currentDrawing) return;
    const rect = canvas.getBoundingClientRect();
    currentDrawing.x2 = e.clientX - rect.left;
    currentDrawing.y2 = e.clientY - rect.top;
    currentDrawing.t2 = chart.timeScale().coordinateToTime(currentDrawing.x2);
    currentDrawing.p2 = candleSeries.coordinateToPrice(currentDrawing.y2);
    renderAllDrawings();
  });

  window.addEventListener('mouseup', () => {
    if (isDrawing && currentDrawing) {
      isDrawing = false;
      drawings.push(currentDrawing);
      currentDrawing = null;
      saveAnnotations();
      renderAllDrawings();
    }
  });
}

function resolveCoords(d) {
  let x1 = d.x1, y1 = d.y1, x2 = d.x2, y2 = d.y2;
  const ts = chart ? chart.timeScale() : null;
  if (ts && d.t1) {
    const cx1 = ts.timeToCoordinate(d.t1);
    if (cx1 !== null) x1 = cx1;
  }
  if (d.p1 !== undefined && d.p1 !== null) {
    const cy1 = candleSeries.priceToCoordinate(d.p1);
    if (cy1 !== null) y1 = cy1;
  }
  if (ts && d.t2) {
    const cx2 = ts.timeToCoordinate(d.t2);
    if (cx2 !== null) x2 = cx2;
  }
  if (d.p2 !== undefined && d.p2 !== null) {
    const cy2 = candleSeries.priceToCoordinate(d.p2);
    if (cy2 !== null) y2 = cy2;
  }
  return { x1, y1, x2, y2 };
}

function renderAllDrawings() {
  const canvas = document.getElementById('draw_canvas');
  if (!canvas || !candleSeries) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const all = currentDrawing ? [...drawings, currentDrawing] : drawings;
  all.forEach(d => {
    if (d.type === 'hline') {
      const y = candleSeries.priceToCoordinate(d.price);
      if (y !== null) {
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#38bdf8';
        ctx.font = '10px monospace';
        ctx.fillText(d.price.toFixed(2), canvas.width - 62, y - 4);
      }
      return;
    }

    const { x1, y1, x2, y2 } = resolveCoords(d);
    if (x1 === undefined || y1 === undefined || x2 === undefined || y2 === undefined) return;

    if (d.type === 'line' || d.type === 'ray' || d.type === 'measure') {
      ctx.strokeStyle = d.type === 'measure' ? '#f59e0b' : '#38bdf8';
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      if (d.type === 'ray' && x2 !== x1) {
        const slope = (y2 - y1) / (x2 - x1);
        const targetX = x2 > x1 ? canvas.width : 0;
        const targetY = y1 + slope * (targetX - x1);
        ctx.lineTo(targetX, targetY);
      } else {
        ctx.lineTo(x2, y2);
      }
      ctx.stroke();

      if (d.type === 'measure' && d.p1 && d.p2) {
        const pct = ((d.p2 / d.p1 - 1) * 100);
        ctx.fillStyle = pct >= 0 ? '#34d399' : '#f87171';
        ctx.font = 'bold 11px monospace';
        ctx.fillText(`${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%`, x2 + 6, y2 - 6);
      }
    } else if (d.type === 'box') {
      ctx.fillStyle = 'rgba(56, 189, 248, 0.12)';
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 1.2;
      ctx.fillRect(Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1));
      ctx.strokeRect(Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1));
    } else if (d.type === 'fib' && d.p1 && d.p2) {
      const levels = [0, 0.382, 0.5, 0.618, 1];
      levels.forEach(lv => {
        const p = d.p1 + (d.p2 - d.p1) * lv;
        const y = candleSeries.priceToCoordinate(p);
        if (y !== null) {
          ctx.strokeStyle = lv === 0.5 || lv === 0.618 ? '#f59e0b' : 'rgba(148,163,184,0.6)';
          ctx.lineWidth = 1;
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.moveTo(Math.min(x1, x2), y);
          ctx.lineTo(Math.max(x1, x2), y);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = '#cbd5e1';
          ctx.font = '9px monospace';
          ctx.fillText(`${(lv * 100).toFixed(1)}% (${p.toFixed(2)})`, Math.max(x1, x2) + 4, y + 3);
        }
      });
    }
  });
}

function undoLastDrawing() {
  drawings.pop();
  saveAnnotations();
  renderAllDrawings();
}

function clearAllDrawings() {
  drawings = [];
  saveAnnotations();
  renderAllDrawings();
}

function loadAnnotations(sym) {
  fetch(`/api/annotations?symbol=${encodeURIComponent(sym)}`)
    .then(r => r.json())
    .then(d => {
      drawings = d.drawings || [];
      renderAllDrawings();
    })
    .catch(() => { drawings = []; renderAllDrawings(); });
}

function saveAnnotations() {
  fetch('/api/annotations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbol: getActiveChartIdentifier(), drawings })
  });
}

// Keyboard navigation
window.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
  if (e.key === '/' || ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k')) {
    e.preventDefault();
    openAssetModal('a');
  } else if (e.key === 'Escape') {
    closeAssetModal();
    setDrawTool('cursor');
  } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    e.preventDefault();
    const grp = getActiveWatchlistGroup();
    if (!grp || !grp.items || !grp.items.length) return;
    const cur = getActiveChartIdentifier();
    let idx = grp.items.findIndex(i => i.id === cur);
    const delta = e.key === 'ArrowDown' ? 1 : -1;
    idx = (idx + delta + grp.items.length) % grp.items.length;
    selectChartIdentifier(grp.items[idx].id);
  }
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


def main():
    parser = argparse.ArgumentParser(description="EP Universal Chart Viewer, Ratio Analyzer & Watchlist Studio")
    parser.add_argument("--port", type=int, default=8789, help="Port to serve on (default: 8789)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    args = parser.parse_args()

    print(f"Starting EP Universal Chart Viewer & Watchlist Studio on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
