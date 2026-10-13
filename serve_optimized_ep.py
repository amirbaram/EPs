#!/usr/bin/env python3
"""serve_optimized_ep.py — Institutional Optimized EP Strategy Server (Port 8787).

Dedicated application demonstrating the optimized Episodic Pivot strategy:
1. Quality Filters:
   - Chart to the Left: Quiet base (runup <= 5%), within 25% of 52W high, RS >= 80, tight consolidation.
   - Day 1 Catalyst: Upper wick <= 20%, close position >= 0.70, RVOL >= 2.5x, no OpEx/Quad distortion.
   - Market Health: SPY above 50 SMA and 20-day momentum >= -2% at breakout entry.
   - Top Winning Sectors & Themes only.
2. Execution Rules:
   - Entry: Day 1 High Breakout strictly within 20 trading days.
   - Position Sizing: V2 Conviction-Weighted at breakout (0.75x to 1.50x).
   - Stop Loss: ML 20 MAE level below Day 1 Close.
   - Trailing Exits: Matched Moving Average trailing exit (50 SMA for Compounders, 20 EMA for Momentum, 10 EMA for Swings).

Usage:
  .venv/bin/python serve_optimized_ep.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

# Ensure repository root is on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import datastore
import ep_catalyst_enricher
import indicators
import scanner_core
from zoneinfo import ZoneInfo
from ep_ml_engine import engine as ml_engine
import thematic_engine

DATA_PATH = ROOT / "data" / "simulations" / "optimized_ep_strategy_trades.parquet"
EP_SCORED_PATH = ROOT / "data" / "simulations" / "ep_combined_study_scored.parquet"
RS_SCORED_PATH = ROOT / "data" / "simulations" / "rs_universe_scores.parquet"
PORT = 8787
EXTREME_PEAK_MODEL_PATH = ROOT / "data" / "models" / "ep_extreme_peak_classifier.pkl"
INTRADAY_CACHE_PATH = ROOT / "data" / "cache" / "intraday_watchlist_status.json"
MONITOR_SETTINGS_FILE = ROOT / "data" / "cache" / "monitor_settings.json"
_EXTREME_PEAK_MODEL = None


def apply_intraday_cache_to_watchlist(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Overlays real-time intraday price, high, exhaustion, and alert states from monitor cache."""
    if not INTRADAY_CACHE_PATH.exists():
        return items
    try:
        data = json.loads(INTRADAY_CACHE_PATH.read_text(encoding="utf-8"))
        positions = data.get("positions", {})
        if not positions:
            return items

        for it in items:
            sym = it.get("symbol")
            if sym in positions:
                p = positions[sym]
                if "current_price" in p:
                    it["current_price"] = p["current_price"]
                if "intraday_high" in p:
                    it["intraday_high"] = p["intraday_high"]
                if "intraday_low" in p:
                    it["intraday_low"] = p["intraday_low"]
                if "intraday_open" in p:
                    it["intraday_open"] = p["intraday_open"]
                if "unrealized_r" in p:
                    it["unrealized_r"] = p["unrealized_r"]
                if "unrealized_pct" in p:
                    it["unrealized_pct"] = p["unrealized_pct"]
                if "peak_intraday_r" in p:
                    it["peak_intraday_r"] = p["peak_intraday_r"]
                if "exhaustion_prob" in p:
                    it["current_high_exhaustion_prob"] = p["exhaustion_prob"]
                if "exhaustion_pct" in p:
                    it["current_high_exhaustion_pct"] = p["exhaustion_pct"]
                if "alert_state" in p:
                    it["alert_state"] = p["alert_state"]
                if "alert_action" in p:
                    it["alert_action"] = p["alert_action"]
                    if p.get("is_alert"):
                        it["climax_action"] = p["alert_action"]
                if "is_alert" in p:
                    it["is_alert"] = p["is_alert"]
                if "last_bar_time" in p:
                    it["last_bar_time"] = p["last_bar_time"]
                if "recommended_ma_price" in p:
                    it["recommended_ma_price"] = p["recommended_ma_price"]
                if "recommended_ma_buffer_pct" in p:
                    it["recommended_ma_buffer_pct"] = p["recommended_ma_buffer_pct"]
    except Exception:
        pass
    return items


def get_extreme_peak_model():
    """Loads and caches the calibrated LightGBM + XGBoost extreme peak classifier."""
    global _EXTREME_PEAK_MODEL
    if _EXTREME_PEAK_MODEL is None and EXTREME_PEAK_MODEL_PATH.exists():
        try:
            _EXTREME_PEAK_MODEL = joblib.load(EXTREME_PEAK_MODEL_PATH)
        except Exception:
            pass
    return _EXTREME_PEAK_MODEL


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates standard Wilder's RSI."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).copy()
    loss = (-delta.where(delta < 0, 0.0)).copy()
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def predict_exhaustion_for_bars(bars: pd.DataFrame, entry_px: float, stop_px: float, rs_val: float = 75.0, v2_conv: float = 0.15) -> float:
    """Predicts the probability that the current candle's high is an extreme climax peak before trailing exit."""
    model_data = get_extreme_peak_model()
    if model_data is None or bars is None or len(bars) < 20:
        return 0.0
    try:
        cal_lgb = model_data.get("cal_lgb")
        cal_xgb = model_data.get("cal_xgb")
        feature_cols = model_data.get("feature_cols", [])

        b = bars.copy()
        b["ema10"] = b["close"].ewm(span=10, adjust=False).mean()
        b["ema20"] = b["close"].ewm(span=20, adjust=False).mean()
        b["sma50"] = b["close"].rolling(window=50, min_periods=20).mean()
        b["vol_sma50"] = b["volume"].rolling(window=50, min_periods=20).mean()

        prev_close = b["close"].shift(1)
        tr = np.maximum(b["high"] - b["low"], np.maximum((b["high"] - prev_close).abs(), (b["low"] - prev_close).abs()))
        b["adr20"] = tr.rolling(window=20, min_periods=10).mean()
        b["adr20_pct"] = (b["adr20"] / b["close"]) * 100.0

        b["rsi14"] = compute_rsi(b["close"], 14)
        b["rsi9"] = compute_rsi(b["close"], 9)
        b["rsi5"] = compute_rsi(b["close"], 5)
        b["rsi14_max5"] = b["rsi14"].rolling(5).max()
        b["rsi14_max10"] = b["rsi14"].rolling(10).max()
        b["rsi14_diff3"] = b["rsi14"] - b["rsi14"].shift(3)

        b["high_roll14"] = b["high"].rolling(14).max()
        b["prev_high_roll14"] = b["high"].shift(14).rolling(14).max()
        b["rsi_roll14"] = b["rsi14"].rolling(14).max()
        b["prev_rsi_roll14"] = b["rsi14"].shift(14).rolling(14).max()

        row = b.iloc[-1]
        curr_high = float(row["high"])
        curr_close = float(row["close"])
        curr_open = float(row["open"])
        curr_low = float(row["low"])
        curr_vol = float(row["volume"])

        ema10 = float(row["ema10"])
        ema20 = float(row["ema20"])
        sma50 = float(row["sma50"]) if pd.notna(row["sma50"]) else ema20
        vol_sma50 = float(row["vol_sma50"]) if pd.notna(row["vol_sma50"]) and row["vol_sma50"] > 0 else 1.0
        adr20 = float(row["adr20"]) if pd.notna(row["adr20"]) and row["adr20"] > 0 else max(0.1, curr_close * 0.03)

        candle_range = max(0.001, curr_high - curr_low)
        upper_wick = max(0.0, curr_high - max(curr_open, curr_close))
        upper_wick_pct = (upper_wick / candle_range) * 100.0
        close_pos = (curr_close - curr_low) / candle_range
        range_to_adr = candle_range / adr20

        price_making_hh = (row["high_roll14"] >= row["prev_high_roll14"])
        rsi_making_lh = (row["rsi_roll14"] < row["prev_rsi_roll14"] - 2.0)
        rsi_bearish_divergence = int(price_making_hh and rsi_making_lh)
        rsi_divergence_delta = float(row["prev_rsi_roll14"] - row["rsi_roll14"]) if rsi_bearish_divergence else 0.0

        ext_ema10_pct = ((curr_high - ema10) / ema10) * 100.0
        ext_ema20_pct = ((curr_high - ema20) / ema20) * 100.0
        ext_sma50_pct = ((curr_high - sma50) / sma50) * 100.0
        ext_ema10_adr = (curr_high - ema10) / adr20
        ext_ema20_adr = (curr_high - ema20) / adr20
        ext_sma50_adr = (curr_high - sma50) / adr20
        rvol = curr_vol / vol_sma50
        volume_churn = rvol / (abs(curr_close / curr_open - 1.0) * 100.0 + 0.5)

        gain_3d = ((curr_close / float(b["close"].iloc[-4])) - 1.0) * 100.0 if len(b) >= 4 else 0.0
        gain_5d = ((curr_close / float(b["close"].iloc[-6])) - 1.0) * 100.0 if len(b) >= 6 else 0.0
        consec_rsi_70 = int((b["rsi14"].iloc[-3:] >= 70.0).sum()) if len(b) >= 3 else 0
        consec_rsi_75 = int((b["rsi14"].iloc[-3:] >= 75.0).sum()) if len(b) >= 3 else 0

        risk_dollar = max(0.01, entry_px - stop_px)
        gain_since_entry = ((curr_high - entry_px) / entry_px) * 100.0
        current_r = (curr_high - entry_px) / risk_dollar

        feats = {
            "rsi14": float(row["rsi14"]), "rsi9": float(row["rsi9"]), "rsi5": float(row["rsi5"]),
            "rsi14_max5": float(row["rsi14_max5"]), "rsi14_max10": float(row["rsi14_max10"]),
            "rsi14_diff3": float(row["rsi14_diff3"]), "rsi_bearish_divergence": rsi_bearish_divergence,
            "rsi_divergence_delta": rsi_divergence_delta, "consec_rsi_70": consec_rsi_70, "consec_rsi_75": consec_rsi_75,
            "ext_ema10_pct": ext_ema10_pct, "ext_ema20_pct": ext_ema20_pct, "ext_sma50_pct": ext_sma50_pct,
            "ext_ema10_adr": ext_ema10_adr, "ext_ema20_adr": ext_ema20_adr, "ext_sma50_adr": ext_sma50_adr,
            "upper_wick_pct": upper_wick_pct, "close_pos": close_pos, "range_to_adr": range_to_adr,
            "rvol": rvol, "volume_churn": volume_churn,
            "gain_since_entry": gain_since_entry, "current_r": current_r,
            "gain_3d": gain_3d, "gain_5d": gain_5d,
            "adr20_pct": float(row["adr20_pct"]),
            "rs_score": rs_val, "v2_conviction": v2_conv
        }

        df_row = pd.DataFrame([feats]).reindex(columns=feature_cols, fill_value=0.0)
        p_lgb = float(cal_lgb.predict_proba(df_row)[0][1]) if cal_lgb is not None else 0.0
        p_xgb = float(cal_xgb.predict_proba(df_row)[0][1]) if cal_xgb is not None else 0.0
        return float(0.55 * p_lgb + 0.45 * p_xgb)
    except Exception:
        return 0.0


def classify_institutional_archetype(row_data: Dict[str, Any], thm_eval: Dict[str, Any], rs: float, stage2: bool) -> Tuple[str, str, str, str]:
    """Classifies ticker into one of 7 institutional archetypes and determines optimal position management."""
    theme = thm_eval.get("theme", "Unknown")
    sector = thm_eval.get("sector", "Unknown")
    thm_score = float(thm_eval.get("theme_info", {}).get("score", 50.0))
    sec_score = float(thm_eval.get("sector_info", {}).get("score", 50.0))
    rvol = float(row_data.get("rvol", 1.0) or 1.0)
    gap = float(row_data.get("gap_pct", 0.0) or 0.0)
    c_pos = float(row_data.get("close_pos", 0.5) or 0.5)

    TOP_SECTORS = {"Specialty Industrial Machinery", "Semiconductors", "Industrials", "Biotechnology", "Computer Hardware", "Technology", "Financial Services", "Capital Markets", "Healthcare", "Medical Devices"}
    TOP_THEMES = {"Semiconductors", "Biotechnology", "Software - Infrastructure", "Computer Hardware", "Health Information Services", "Medical Devices", "Software - Application", "Cloud", "Artificial Intelligence"}
    TOP_CLUSTERS = {"Computer Hardware", "Semiconductors", "Biotechnology", "Semiconductor Equipment", "Bitcoin Miners", "Quantum Computing", "Uranium & Nuclear", "Clean Energy", "Software - Application", "Software - Infrastructure", "Aerospace & Defense", "Cloud", "Artificial Intelligence"}
    HB_THEMES = {"Bitcoin Miners", "Quantum Computing", "Semiconductor Equipment", "Clean Energy", "Data Centers", "Solar", "Nuclear & Uranium"}

    if theme in HB_THEMES or (gap >= 12.0 and rvol >= 3.5):
        return ("⚡ High Beta Momentum", "ema10", "10 EMA Fast Velocity Trail", "Explosive velocity momentum. Strict 10 EMA trailing locks in rapid gains before deep retracements.")
    if (sec_score < 35 and thm_score < 35) and (rvol >= 6.0 and c_pos >= 0.80):
        return ("🎯 Idiosyncratic Alpha", "ema10", "10 EMA Fast Velocity Trail", "Massive idiosyncratic catalyst overrides broad group drag. Fast 10 EMA trailing locks in spike gains.")
    if (c_pos >= 0.65 and rvol >= 2.5 and gap >= 5.0 and (sector in TOP_SECTORS or theme in TOP_THEMES or thm_score >= 60)):
        return ("👑 Pinnacle Elite", "sma50", "50 SMA Institutional Baseline", "Apex institutional quality. Wide 50 SMA baseline allows multi-month trend maturation without premature shakeout.")
    if (sector in TOP_CLUSTERS or theme in TOP_CLUSTERS or rs >= 90 or stage2):
        return ("🚀 Multi-Quarter Leaders", "sma50", "50 SMA Institutional Baseline", "Secular institutional accumulation leader. Trail exclusively on rising 50 SMA; harvest 50% on ML exhaustion.")
    if (rvol >= 4.0 and gap >= 6.0 and c_pos >= 0.65):
        return ("⭐ Institutional Sweet Spot", "ema20", "20 EMA Intermediate Momentum Trail", "High volume surge sweet spot. Trails smoothly above the rising 20 EMA intermediate absorption cushion.")
    if (thm_score >= 65 or sec_score >= 65 or rs >= 80) and rvol >= 2.5:
        return ("🌟 Emerging Leaders", "ema20", "20 EMA Intermediate Momentum Trail", "High relative strength leadership trend. 20 EMA strikes optimal balance between profit retention and runner longevity.")
    if c_pos >= 0.65:
        return ("🛡️ Conservative Swing", "ema10", "10 EMA Fast Velocity Trail", "Solid structural breakout. 10 EMA trailing locks in steady swing gains.")
    return ("🏛️ Institutional Core", "ema20", "20 EMA Intermediate Momentum Trail", "Standard institutional episodic pivot. 20 EMA balanced intermediate trailing exit.")


app = Flask(__name__)
_DATASET: Optional[pd.DataFrame] = None
_WATCHLIST: Optional[List[Dict[str, Any]]] = None
_MARKET_LOOKUP: Optional[Dict[str, Dict[str, Any]]] = None


def get_market_lookup() -> Dict[str, Dict[str, Any]]:
    """Precomputes SPY, QQQ, and Tech Leadership (QQQ/SPY) indicators for all trading dates."""
    global _MARKET_LOOKUP
    if _MARKET_LOOKUP is not None:
        return _MARKET_LOOKUP
    spy = datastore.load_bars("SPY")
    qqq = datastore.load_bars("QQQ")
    lookup: Dict[str, Dict[str, Any]] = {}
    if spy is not None and qqq is not None:
        common = spy.index.intersection(qqq.index)
        spy_c = spy.loc[common].copy()
        qqq_c = qqq.loc[common].copy()
        spy_c["sma50"] = spy_c["close"].rolling(50, min_periods=20).mean()
        spy_c["sma200"] = spy_c["close"].rolling(200, min_periods=50).mean()
        spy_c["ret20"] = spy_c["close"].pct_change(20) * 100.0

        qqq_c["sma50"] = qqq_c["close"].rolling(50, min_periods=20).mean()
        qqq_c["sma200"] = qqq_c["close"].rolling(200, min_periods=50).mean()
        qqq_c["ret20"] = qqq_c["close"].pct_change(20) * 100.0
        qqq_c["ema10"] = qqq_c["close"].ewm(span=10, adjust=False).mean()
        qqq_c["ema20"] = qqq_c["close"].ewm(span=20, adjust=False).mean()

        ratio = qqq_c["close"] / spy_c["close"]
        ratio_e10 = ratio.ewm(span=10, adjust=False).mean()
        ratio_e20 = ratio.ewm(span=20, adjust=False).mean()

        d_strs = common.strftime("%Y-%m-%d").tolist()
        for i, ds in enumerate(d_strs):
            lookup[ds] = {
                "spy_above_50": bool(spy_c["close"].iloc[i] >= spy_c["sma50"].iloc[i]),
                "spy_ret20": round(float(spy_c["ret20"].iloc[i]), 1) if pd.notna(spy_c["ret20"].iloc[i]) else 0.0,
                "qqq_above_50": bool(qqq_c["close"].iloc[i] >= qqq_c["sma50"].iloc[i]),
                "qqq_ret20": round(float(qqq_c["ret20"].iloc[i]), 1) if pd.notna(qqq_c["ret20"].iloc[i]) else 0.0,
                "qqq_bull_stack": bool(qqq_c["close"].iloc[i] >= qqq_c["ema10"].iloc[i] >= qqq_c["ema20"].iloc[i]),
                "tech_lead": bool(ratio.iloc[i] >= ratio_e10.iloc[i] >= ratio_e20.iloc[i]),
            }
    _MARKET_LOOKUP = lookup
    return _MARKET_LOOKUP


def safe_float(val: Any, default: float = 0.0) -> float:
    if val is None or pd.isna(val):
        return default
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except Exception:
        return default


def get_dataset(force_reload: bool = False) -> pd.DataFrame:
    """Loads and caches the optimized strategy dataset."""
    global _DATASET
    if _DATASET is None or force_reload:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Optimized trades dataset not found at {DATA_PATH}")
        df = pd.read_parquet(DATA_PATH)
        # Ensure helper columns
        if "is_tier1" not in df.columns:
            df["is_tier1"] = True
        if "tier" not in df.columns:
            df["tier"] = df["is_tier1"].map({True: "Tier 1: Ultra-Elite", False: "Tier 2: Institutional Core"})
        _DATASET = df
    return _DATASET


def get_active_watchlist(force_reload: bool = False) -> List[Dict[str, Any]]:
    """Scans and caches active in-flight trades and coiling setups from the last 20-25 trading days."""
    global _WATCHLIST
    if _WATCHLIST is not None and not force_reload:
        return apply_intraday_cache_to_watchlist([dict(x) for x in _WATCHLIST])

    if not EP_SCORED_PATH.exists():
        return []

    df_ep = pd.read_parquet(EP_SCORED_PATH)
    df_ep["date"] = pd.to_datetime(df_ep["date"])

    rs_map: Dict[str, float] = {}
    stage2_map: Dict[str, bool] = {}
    if RS_SCORED_PATH.exists():
        try:
            rs_df = pd.read_parquet(RS_SCORED_PATH)
            rs_map = dict(zip(rs_df["symbol"], rs_df["rs_score"]))
            stage2_map = dict(zip(rs_df["symbol"], rs_df["stage2"]))
        except Exception:
            pass

    max_dt = df_ep["date"].max()
    cutoff_dt = max_dt - pd.Timedelta(days=35)
    recent = df_ep[df_ep["date"] >= cutoff_dt].copy()
    recent = recent.sort_values(by="date", ascending=False)

    eng = thematic_engine.get_thematic_engine()
    mkt_lookup = get_market_lookup()

    items: List[Dict[str, Any]] = []

    for _, row in recent.iterrows():
        sym = str(row["symbol"]).strip().upper()
        ep_dt = row["date"].strftime("%Y-%m-%d")
        rs = rs_map.get(sym, safe_float(row.get("rs_score", 50.0), 50.0))
        stage2 = stage2_map.get(sym, bool(row.get("stage2", False)))
        gap_pct = safe_float(row.get("gap_pct", 0.0))
        rvol = safe_float(row.get("rvol", 0.0))

        # Filter criteria: RS >= 70 OR Stage 2 OR Gap >= 8% with RVOL >= 2.5x
        if not ((rs >= 70.0) or stage2 or (gap_pct >= 8.0 and rvol >= 2.5)):
            continue

        bars = datastore.load_bars(sym)
        if bars is None or bars.empty:
            continue

        ep_locs = np.where(bars.index.strftime("%Y-%m-%d") == ep_dt)[0]
        if len(ep_locs) == 0:
            continue
        ep_idx = int(ep_locs[0])

        d1_high = float(row["high"]) if "high" in row and pd.notna(row["high"]) else float(bars["high"].iloc[ep_idx])
        d1_low = float(row["low"]) if "low" in row and pd.notna(row["low"]) else float(bars["low"].iloc[ep_idx])
        d1_close = float(row["close"]) if "close" in row and pd.notna(row["close"]) else float(bars["close"].iloc[ep_idx])
        d1_open = float(bars["open"].iloc[ep_idx])

        # Pre-EP features (Chart to Left)
        pre_bars = bars.iloc[:ep_idx]
        if not pre_bars.empty:
            pre_20 = pre_bars.iloc[-20:]
            d0_close = float(pre_bars["close"].iloc[-1])
            c20_min = float(pre_20["low"].min())
            c20_max = float(pre_20["high"].max())
            pre_consol_tightness = round(((c20_max - c20_min) / d0_close) * 100.0, 1) if d0_close > 0 else 0.0
            pre_runup = round(((d0_close - float(pre_20["close"].iloc[0])) / float(pre_20["close"].iloc[0])) * 100.0, 1) if len(pre_20) > 0 and float(pre_20["close"].iloc[0]) > 0 else 0.0
            hi_52w = float(pre_bars.iloc[-250:]["high"].max())
            pre_dist_52w = round(((hi_52w - d0_close) / hi_52w) * 100.0, 1) if hi_52w > 0 else 0.0
        else:
            pre_consol_tightness = 0.0
            pre_runup = 0.0
            pre_dist_52w = 0.0

        d1_rng = max(0.01, d1_high - d1_low)
        day1_upper_wick = round(((d1_high - max(d1_open, d1_close)) / d1_rng) * 100.0, 1)
        close_pos_val = round(max(0.0, min(1.0, (d1_close - d1_low) / d1_rng)), 2)

        fwd = bars.iloc[ep_idx + 1:]
        curr_close = float(bars["close"].iloc[-1])
        trigger_px = round(d1_high + 0.05, 2)
        stop_px = round(max(d1_low, trigger_px * 0.927), 2)
        risk_pts = max(0.01, trigger_px - stop_px)

        broke_out = False
        breakout_date = None
        breakout_idx = None
        stopped_out = False

        for b_i in range(len(fwd)):
            bar = fwd.iloc[b_i]
            b_dt = fwd.index[b_i].strftime("%Y-%m-%d")
            if not broke_out:
                if float(bar["low"]) < d1_low:
                    stopped_out = True
                    break
                if float(bar["high"]) >= trigger_px:
                    broke_out = True
                    breakout_date = b_dt
                    breakout_idx = b_i
            else:
                if float(bar["low"]) <= stop_px:
                    stopped_out = True
                    break

        if stopped_out and not broke_out:
            continue

        # V2 Rolling Conviction & Win Probability
        eval_fwd = breakout_idx if (broke_out and breakout_idx is not None) else min(len(fwd), 2)
        v2_feats = ml_engine.compute_rolling_features(sym, ep_idx, ep_idx + eval_fwd)
        v2_preds = ml_engine.predict_rolling(v2_feats) if v2_feats else {}
        v2_50 = float(v2_preds.get("prob_50", 0.0))
        v2_win = round(float(1.0 - (v2_50 * 0.4)), 3) if v2_50 > 0 else 0.50

        # Thematic Engine Evaluation
        thm_eval = eng.evaluate_symbol(sym)
        theme_name = thm_eval.get("theme", "Unknown")
        sector_name = thm_eval.get("sector", "Unknown")
        theme_score = round(float(thm_eval.get("theme_info", {}).get("score", 50.0)), 1)
        sector_score = round(float(thm_eval.get("sector_info", {}).get("score", 50.0)), 1)
        thm_m1 = round(float(thm_eval.get("theme_info", {}).get("m1_ret", 0.0)), 1)
        sec_m1 = round(float(thm_eval.get("sector_info", {}).get("m1_ret", 0.0)), 1)
        thm_arch = thm_eval.get("primary_archetype", "Leading Trend")

        # Market Health (SPY & QQQ) on active trigger date (or EP date if pending)
        active_mkt_dt = breakout_date if broke_out else ep_dt
        m_info = mkt_lookup.get(active_mkt_dt, {})
        spy_50 = m_info.get("spy_above_50", True)
        spy_ret20 = m_info.get("spy_ret20", 0.0)
        qqq_50 = m_info.get("qqq_above_50", True)
        qqq_ret20 = m_info.get("qqq_ret20", 0.0)
        qqq_bull = m_info.get("qqq_bull_stack", True)
        tech_lead = m_info.get("tech_lead", True)

        # Classify Institutional Archetype & Matched Trailing MA
        arch_name, rec_ma_key, rec_ma_name, plan_desc = classify_institutional_archetype(row, thm_eval, rs, stage2)

        # Compute moving averages on latest bar
        ema10_val = float(bars["close"].ewm(span=10, adjust=False).mean().iloc[-1])
        ema20_val = float(bars["close"].ewm(span=20, adjust=False).mean().iloc[-1])
        sma50_val = float(bars["close"].rolling(50, min_periods=20).mean().iloc[-1])
        rec_ma_px = sma50_val if rec_ma_key == "sma50" else (ema10_val if rec_ma_key == "ema10" else ema20_val)
        rec_ma_buf = round(((curr_close - rec_ma_px) / rec_ma_px) * 100.0, 1) if rec_ma_px > 0 else 0.0

        # Predict ML Climax Exhaustion on Current Candle High
        exh_prob = predict_exhaustion_for_bars(bars, trigger_px, stop_px, rs_val=rs, v2_conv=v2_50)
        exh_pct = round(exh_prob * 100.0, 1)
        climax_th = 0.55 if rec_ma_key == "sma50" else 0.45
        climax_th_pct = 55.0 if rec_ma_key == "sma50" else 45.0
        sell_half_rule = f"Sell 50% partial profit into extreme high when ML Exhaustion ≥ 50.0% (locks peak R, retains remainder on {rec_ma_name})"
        exit_rule = f"Close below {rec_ma_name.split(' ')[0]} triggers full exit of position at market close"

        # Suggested Position Size according to Setup Quality & Institutional Trade Rules
        if v2_50 >= 0.25 and (arch_name == "👑 Pinnacle Elite" or rs >= 85.0):
            sug_mult = 1.50
            sug_label = "1.50x Max Allocation"
            sug_desc = f"Apex institutional quality: Pinnacle Elite / High RS ({rs:.0f}) + High V2 Conviction P(+50%): {v2_50*100:.1f}% ≥ 25%."
        elif v2_50 >= 0.20 or rs >= 80.0:
            sug_mult = 1.25
            sug_label = "1.25x Enhanced Allocation"
            sug_desc = f"Solid institutional setup: Strong Conviction P(+50%): {v2_50*100:.1f}% and High RS ({rs:.0f}) leadership."
        elif v2_50 >= 0.10:
            sug_mult = 1.00
            sug_label = "1.00x Base Allocation"
            sug_desc = f"Standard institutional EP allocation: 1.0 R risk budget under {arch_name} rules."
        else:
            sug_mult = 0.75
            sug_label = "0.75x Defensive Allocation"
            sug_desc = f"Defensive / speculative setup: Lower V2 Conviction ({v2_50*100:.1f}% < 10%). Size down to contain downside."

        # Fundamental Catalyst & SEC Filings are loaded on-demand by /api/chart for selected setup
        cat_dossier = None

        if broke_out and not stopped_out:
            unr_r = round((curr_close - trigger_px) / risk_pts, 2)
            unr_pct = round(((curr_close / trigger_px) - 1.0) * 100.0, 1)
            if exh_prob >= climax_th:
                climax_action = f"🚨 SELL 50% PARTIAL PROFIT NOW (Current High Exhaustion: {exh_pct}% ≥ {climax_th_pct:.0f}%) — Retain 50% Runner on {rec_ma_name}"
            elif exh_prob >= 0.40:
                climax_action = f"⚠️ ELEVATED CLIMAX RISK ({exh_pct}%) — Approaching {climax_th_pct:.0f}% Sell-Half Threshold"
            else:
                climax_action = f"🟢 HOLD RUNNER (Current High Exhaustion: {exh_pct}% < {climax_th_pct:.0f}% Trigger Threshold)"

            items.append({
                "symbol": sym,
                "ep_date": ep_dt,
                "status": "Active Long",
                "trigger_date": breakout_date,
                "trigger_price": trigger_px,
                "current_price": round(curr_close, 2),
                "stop_price": stop_px,
                "unrealized_r": unr_r,
                "unrealized_pct": unr_pct,
                "dist_to_trigger": 0.0,
                "rs_score": round(rs, 1),
                "stage2": stage2,
                "gap_pct": round(gap_pct, 1),
                "rvol": round(rvol, 1),
                "close_pos": close_pos_val,
                "day1_upper_wick_pct": day1_upper_wick,
                "pre_dist_52w_high": pre_dist_52w,
                "pre_runup_20d": pre_runup,
                "pre_consol_tightness_20d": pre_consol_tightness,
                "dynamic_stop_loss_pct": round(((trigger_px - stop_px) / trigger_px) * 100.0, 1),
                "v2_breakout": round(v2_50, 4),
                "v2_conviction_50": round(v2_50, 4),
                "v2_win_prob": v2_win,
                "suggested_size_mult": sug_mult,
                "suggested_size_label": sug_label,
                "suggested_size_desc": sug_desc,
                "catalyst": cat_dossier,
                "days_in_trade": len(fwd) - breakout_idx if breakout_idx is not None else 1,
                "days_coiling": 0,
                "trade_type": "Trade 1: Base EP Delayed Breakout (Days 2-5)",
                "archetype": arch_name,
                "recommended_ma_name": rec_ma_name,
                "recommended_ma_key": rec_ma_key,
                "recommended_ma_price": round(rec_ma_px, 2),
                "recommended_ma_buffer_pct": rec_ma_buf,
                "current_high_exhaustion_prob": round(exh_prob, 3),
                "current_high_exhaustion_pct": exh_pct,
                "climax_threshold": climax_th,
                "climax_threshold_pct": climax_th_pct,
                "climax_sell_half_rule": sell_half_rule,
                "climax_action": climax_action,
                "trailing_exit_rule": exit_rule,
                "position_management_plan": plan_desc,
                "sector": sector_name,
                "theme": theme_name,
                "theme_score": theme_score,
                "sector_score": sector_score,
                "thm_m1": thm_m1,
                "sec_m1": sec_m1,
                "thematic_archetype": thm_arch,
                "spy_above_50_entry": spy_50,
                "spy_ret_20d_entry": spy_ret20,
                "qqq_above_50_entry": qqq_50,
                "qqq_ret_20d_entry": qqq_ret20,
                "qqq_bull_stack": qqq_bull,
                "qqq_spy_ratio_bull": tech_lead,
            })
        elif not broke_out and not stopped_out and len(fwd) <= 20:
            dist_to_trigger = round(((trigger_px / curr_close) - 1.0) * 100.0, 1)
            climax_action = "⏳ Pending Breakout Trigger (Monitors at Breakout High)"
            items.append({
                "symbol": sym,
                "ep_date": ep_dt,
                "status": "Coiling Setup",
                "trigger_date": "Pending",
                "trigger_price": trigger_px,
                "current_price": round(curr_close, 2),
                "stop_price": stop_px,
                "unrealized_r": 0.0,
                "unrealized_pct": 0.0,
                "dist_to_trigger": dist_to_trigger,
                "rs_score": round(rs, 1),
                "stage2": stage2,
                "gap_pct": round(gap_pct, 1),
                "rvol": round(rvol, 1),
                "close_pos": close_pos_val,
                "day1_upper_wick_pct": day1_upper_wick,
                "pre_dist_52w_high": pre_dist_52w,
                "pre_runup_20d": pre_runup,
                "pre_consol_tightness_20d": pre_consol_tightness,
                "dynamic_stop_loss_pct": round(((trigger_px - stop_px) / trigger_px) * 100.0, 1),
                "v2_breakout": round(v2_50, 4),
                "v2_conviction_50": round(v2_50, 4),
                "v2_win_prob": v2_win,
                "suggested_size_mult": sug_mult,
                "suggested_size_label": sug_label,
                "suggested_size_desc": sug_desc,
                "catalyst": cat_dossier,
                "days_in_trade": 0,
                "days_coiling": len(fwd),
                "trade_type": "Trade 1: Base EP Delayed Breakout (Days 2-5)",
                "archetype": arch_name,
                "recommended_ma_name": rec_ma_name,
                "recommended_ma_key": rec_ma_key,
                "recommended_ma_price": round(rec_ma_px, 2),
                "recommended_ma_buffer_pct": rec_ma_buf,
                "current_high_exhaustion_prob": round(exh_prob, 3),
                "current_high_exhaustion_pct": exh_pct,
                "climax_threshold": climax_th,
                "climax_threshold_pct": climax_th_pct,
                "climax_sell_half_rule": sell_half_rule,
                "climax_action": climax_action,
                "trailing_exit_rule": exit_rule,
                "position_management_plan": plan_desc,
                "sector": sector_name,
                "theme": theme_name,
                "theme_score": theme_score,
                "sector_score": sector_score,
                "thm_m1": thm_m1,
                "sec_m1": sec_m1,
                "thematic_archetype": thm_arch,
                "spy_above_50_entry": spy_50,
                "spy_ret_20d_entry": spy_ret20,
                "qqq_above_50_entry": qqq_50,
                "qqq_ret_20d_entry": qqq_ret20,
                "qqq_bull_stack": qqq_bull,
                "qqq_spy_ratio_bull": tech_lead,
            })

    _WATCHLIST = items
    return apply_intraday_cache_to_watchlist([dict(x) for x in _WATCHLIST])


@app.route("/api/reload_dataset", methods=["GET", "POST"])
def api_reload_dataset():
    """Hot-reload endpoint for daily automation updater."""
    df = get_dataset(force_reload=True)
    wl = get_active_watchlist(force_reload=True)
    return jsonify({
        "status": "ok",
        "trades_count": len(df),
        "tier1_count": int(df["is_tier1"].sum()),
        "max_date": str(df["entry_date"].max()) if not df.empty else "",
        "watchlist_count": len(wl),
        "active_count": sum(1 for x in wl if x["status"] == "Active Long"),
        "coiling_count": sum(1 for x in wl if x["status"] == "Coiling Setup"),
    })


def apply_progressive_exposure(df: pd.DataFrame, base_r_col: str, progressive_mode: str) -> pd.Series:
    """Applies progressive exposure models chronologically."""
    if progressive_mode == "none" or df.empty:
        return df[base_r_col]

    # Must be sorted by entry_date chronologically
    df_sorted = df.sort_values(by=["entry_date", "symbol"]).copy()
    base_r = df_sorted[base_r_col].values
    new_r = []

    if progressive_mode == "ladder":
        # Classic 3-Tier Ladder: 0.5x -> 1.0x -> 1.5x
        state = 0
        consec_losses = 0
        mults = [0.50, 1.00, 1.50]
        for r_val in base_r:
            new_r.append(r_val * mults[state])
            if r_val > 0:
                consec_losses = 0
                state = min(2, state + 1)
            else:
                consec_losses += 1
                state = 0 if consec_losses >= 2 else max(0, state - 1)
    elif progressive_mode == "qtr_defense":
        # Quarter-Size Defense Ladder: 0.25x on 2 losses -> 0.5x -> 1.0x -> 1.5x
        state = 1  # Start at 0.50x recovery
        consec_losses = 0
        mults = [0.25, 0.50, 1.00, 1.50]
        for r_val in base_r:
            new_r.append(r_val * mults[state])
            if r_val > 0:
                consec_losses = 0
                state = min(3, state + 1)
            else:
                consec_losses += 1
                state = 0 if consec_losses >= 2 else max(0, state - 1)
    elif progressive_mode == "confluence":
        # Confluence (V2 Conviction x Progressive State)
        state = 1
        consec_losses = 0
        mults = [0.60, 1.00, 1.40]
        for i, r_val in enumerate(base_r):
            new_r.append(r_val * mults[state])
            if r_val > 0:
                consec_losses = 0
                state = min(2, state + 1)
            else:
                consec_losses += 1
                state = 0 if consec_losses >= 2 else max(0, state - 1)
    elif progressive_mode == "qtr_confluence":
        # Confluence + 0.25x Defense (V2 Conviction x 0.25x Floor on 2 Losses)
        state = 2  # Start at 1.00x standard
        consec_losses = 0
        mults = [0.25, 0.60, 1.00, 1.40]
        for i, r_val in enumerate(base_r):
            new_r.append(r_val * mults[state])
            if r_val > 0:
                consec_losses = 0
                state = min(3, state + 1)
            else:
                consec_losses += 1
                state = 0 if consec_losses >= 2 else max(0, state - 1)
    elif progressive_mode == "momentum":
        # Rolling 5-Trade Streak
        recent = []
        for i, r_val in enumerate(base_r):
            if len(recent) < 5:
                m = 1.0
            else:
                wr = sum(1 for x in recent[-5:] if x > 0) / 5.0
                m = 1.40 if wr >= 0.60 else (0.60 if wr <= 0.20 else 1.00)
            new_r.append(r_val * m)
            recent.append(r_val)
    elif progressive_mode == "equity_ma":
        # Equity curve 10-MA filter
        eq_curve = [0.0]
        for i, r_val in enumerate(base_r):
            if len(eq_curve) >= 10:
                ma = float(np.mean(eq_curve[-10:]))
                m = 1.25 if eq_curve[-1] >= ma else 0.50
            else:
                m = 1.00
            eff = r_val * m
            new_r.append(eff)
            eq_curve.append(eq_curve[-1] + eff)
    else:
        return df[base_r_col]

    df_sorted["prog_r"] = new_r
    return df_sorted.loc[df.index, "prog_r"]


def filter_dataset(
    tier: str = "all",
    archetype: str = "all",
    exit_mode: str = "adaptive",
    sizing: str = "v2",
    min_v2: float = 0.0,
    min_rs: float = 0.0,
    progressive: str = "none",
    stop_mode: str = "structural",
    climax: str = "none",
) -> Tuple[pd.DataFrame, str, str]:
    """Filters dataset according to active UI parameters and selects the active R column."""
    df = get_dataset().copy()

    # 1. Tier Filter
    if tier == "tier1":
        df = df[df["is_tier1"] == True]

    # 2. Archetype Filter
    if archetype and archetype != "all":
        col = f"is_{archetype.lower().replace(' ', '_')}"
        if col in df.columns:
            df = df[df[col] == True]

    # 3. Minimum V2 Conviction Filter
    if min_v2 > 0.0:
        v2_col = "v2_win_prob" if "v2_win_prob" in df.columns else ("v2_breakout" if "v2_breakout" in df.columns else "ml_50")
        df = df[df[v2_col].fillna(0.0) >= min_v2]

    # 4. Minimum RS Score Filter
    if min_rs > 0.0:
        rs_col = "rs_score_breakout" if "rs_score_breakout" in df.columns else "rs_score"
        df = df[df[rs_col].fillna(50.0) >= min_rs]

    # 5. Resolve Active R and Exit Columns based on exit_mode, sizing, stop_mode, and climax harvesting
    is_ml_stop = (stop_mode == "ml_adaptive")
    is_climax = (climax == "partial_50")

    if is_ml_stop:
        if is_climax:
            if exit_mode == "sma50":
                base_r_col = "climax_blended_v2_r_ml_stop_sma50" if sizing == "v2" else "climax_blended_r_ml_stop_sma50"
                ma_label = "50 SMA + 50% Climax Take (ML Stop)"
            elif exit_mode == "ema20":
                base_r_col = "climax_blended_v2_r_ml_stop_ema20" if sizing == "v2" else "climax_blended_r_ml_stop_ema20"
                ma_label = "20 EMA + 50% Climax Take (ML Stop)"
            elif exit_mode == "ema10":
                base_r_col = "ml_stop_ema10_v2_r" if sizing == "v2" else "ml_stop_ema10_r"
                ma_label = "10 EMA (ML Stop)"
            else:  # adaptive
                base_r_col = "climax_blended_v2_r_ml_stop_adaptive" if sizing == "v2" else "climax_blended_r_ml_stop_adaptive"
                ma_label = "Adaptive MA + 50% Climax Take (ML Stop)"
        else:
            if exit_mode == "sma50":
                base_r_col = "ml_stop_sma50_v2_r" if sizing == "v2" else "ml_stop_sma50_r"
                ma_label = "50 SMA (ML Stop)"
            elif exit_mode == "ema20":
                base_r_col = "ml_stop_ema20_v2_r" if sizing == "v2" else "ml_stop_ema20_r"
                ma_label = "20 EMA (ML Stop)"
            elif exit_mode == "ema10":
                base_r_col = "ml_stop_ema10_v2_r" if sizing == "v2" else "ml_stop_ema10_r"
                ma_label = "10 EMA (ML Stop)"
            else:  # adaptive
                base_r_col = "ml_stop_adaptive_v2_r" if sizing == "v2" else "ml_stop_adaptive_r"
                ma_label = "Adaptive MA (ML Stop)"
    else:  # structural stop
        if is_climax:
            if exit_mode == "sma50":
                base_r_col = "climax_blended_v2_r_sma50" if sizing == "v2" else "climax_blended_r_sma50"
                ma_label = "50 SMA + 50% Climax Take"
            elif exit_mode == "ema20":
                base_r_col = "climax_blended_v2_r_ema20" if sizing == "v2" else "climax_blended_r_ema20"
                ma_label = "20 EMA + 50% Climax Take"
            elif exit_mode == "ema10":
                base_r_col = "ema10_v2_r" if sizing == "v2" else "ema10_r"
                ma_label = "10 EMA"
            else:  # adaptive
                base_r_col = "climax_blended_v2_r_adaptive" if sizing == "v2" else "climax_blended_r_adaptive"
                ma_label = "Adaptive MA + 50% Climax Take"
        else:
            if exit_mode == "sma50":
                base_r_col = "sma50_v2_r" if sizing == "v2" else "sma50_r"
                ma_label = "50 SMA"
            elif exit_mode == "ema20":
                base_r_col = "ema20_v2_r" if sizing == "v2" else "ema20_r"
                ma_label = "20 EMA"
            elif exit_mode == "ema10":
                base_r_col = "ema10_v2_r" if sizing == "v2" else "ema10_r"
                ma_label = "10 EMA"
            else:  # adaptive
                base_r_col = "adaptive_v2_sized_r" if sizing == "v2" else "adaptive_r"
                ma_label = "Adaptive MA"

    if base_r_col not in df.columns:
        base_r_col = "adaptive_v2_sized_r" if sizing == "v2" else "adaptive_r"

    # 6. Apply Progressive Exposure if active
    if progressive != "none":
        df["active_calc_r"] = apply_progressive_exposure(df, base_r_col, progressive)
        r_col = "active_calc_r"
    else:
        r_col = base_r_col

    return df, r_col, ma_label


def compute_kpis(df: pd.DataFrame, r_col: str) -> Dict[str, Any]:
    """Calculates summary KPIs over filtered trades."""
    n = len(df)
    if n == 0:
        return {
            "trades_count": 0, "win_rate_pct": 0.0, "ev_r": 0.0, "total_r": 0.0,
            "profit_factor": 0.0, "payoff_ratio": 0.0, "avg_win_r": 0.0, "avg_loser_r": 0.0,
            "big_wins_count": 0, "home_runs_count": 0, "max_drawdown_r": 0.0
        }

    wins = df[df[r_col] > 0]
    losses = df[df[r_col] <= 0]
    wr = (len(wins) / n) * 100.0
    ev = float(df[r_col].mean())
    tot = float(df[r_col].sum())
    gp = float(wins[r_col].sum())
    gl = float(abs(losses[r_col].sum()))
    pf = (gp / gl) if gl > 0 else (99.0 if gp > 0 else 0.0)
    avg_w = float(wins[r_col].mean()) if not wins.empty else 0.0
    avg_l = float(losses[r_col].mean()) if not losses.empty else 0.0
    payoff = abs(avg_w / avg_l) if avg_l != 0 else (99.0 if avg_w > 0 else 0.0)

    # Historical drawdown calculation
    cumsum = df[r_col].cumsum()
    max_dd = float((cumsum - cumsum.cummax()).min())

    return {
        "trades_count": n,
        "win_rate_pct": round(wr, 1),
        "ev_r": round(ev, 3),
        "total_r": round(tot, 1),
        "profit_factor": round(pf, 2),
        "payoff_ratio": round(payoff, 2),
        "avg_win_r": round(avg_w, 2),
        "avg_loser_r": round(avg_l, 2),
        "big_wins_count": int((df[r_col] >= 2.0).sum()),
        "home_runs_count": int((df[r_col] >= 5.0).sum()),
        "max_drawdown_r": round(max_dd, 1),
    }


@app.route("/api/kpis")
def api_kpis():
    """Returns dynamic strategy KPIs according to query parameters."""
    tier = request.args.get("tier", "all")
    archetype = request.args.get("archetype", "all")
    exit_mode = request.args.get("exit_mode", "adaptive")
    sizing = request.args.get("sizing", "v2")
    min_v2 = float(request.args.get("min_v2", 0.0))
    min_rs = float(request.args.get("min_rs", 0.0))
    prog = request.args.get("progressive", "none")
    stop_mode = request.args.get("stop_mode", "structural")
    climax = request.args.get("climax", "none")

    df_filtered, r_col, ma_label = filter_dataset(tier, archetype, exit_mode, sizing, min_v2, min_rs, prog, stop_mode, climax)
    kpis = compute_kpis(df_filtered, r_col)
    kpis["active_r_col"] = r_col
    kpis["active_ma_label"] = ma_label
    return jsonify(kpis)


@app.route("/api/trades")
def api_trades():
    """Returns the list of trades matching the current filter set."""
    tier = request.args.get("tier", "all")
    archetype = request.args.get("archetype", "all")
    exit_mode = request.args.get("exit_mode", "adaptive")
    sizing = request.args.get("sizing", "v2")
    min_v2 = float(request.args.get("min_v2", 0.0))
    min_rs = float(request.args.get("min_rs", 0.0))
    prog = request.args.get("progressive", "none")
    sort_by = request.args.get("sort_by", "r_multiple")
    sort_dir = request.args.get("sort_dir", "desc")
    stop_mode = request.args.get("stop_mode", "structural")
    climax = request.args.get("climax", "none")

    df_filtered, r_col, ma_label = filter_dataset(tier, archetype, exit_mode, sizing, min_v2, min_rs, prog, stop_mode, climax)
    if df_filtered.empty:
        return jsonify({"trades": [], "count": 0})

    is_ml_stop = (stop_mode == "ml_adaptive")
    records = []
    for _, r in df_filtered.iterrows():
        if is_ml_stop:
            active_stop_px = safe_float(r.get("ml_stop_price", r["stop_price"]), 0.0)
            active_risk_pct = safe_float(r.get("ml_stop_pct", 6.0), 6.0)
            if exit_mode == "ema10":
                active_exit_px = r.get("ml_stop_ema10_exit_price", r.get("ml_stop_adaptive_exit_price", r["exit_price"]))
                active_exit_dt = r.get("ml_stop_ema10_exit_date", r.get("ml_stop_adaptive_exit_date", r["exit_date"]))
                active_reason = r.get("ml_stop_ema10_exit_reason", "Close Below 10 EMA (ML Stop)")
                active_days = r.get("ml_stop_ema10_hold_days", r["holding_days"])
                active_ret = r.get("ml_stop_ema10_ret_pct", r["return_pct"])
            elif exit_mode == "ema20":
                active_exit_px = r.get("ml_stop_ema20_exit_price", r.get("ml_stop_adaptive_exit_price", r["exit_price"]))
                active_exit_dt = r.get("ml_stop_ema20_exit_date", r.get("ml_stop_adaptive_exit_date", r["exit_date"]))
                active_reason = r.get("ml_stop_ema20_exit_reason", "Close Below 20 EMA (ML Stop)")
                active_days = r.get("ml_stop_ema20_hold_days", r["holding_days"])
                active_ret = r.get("ml_stop_ema20_ret_pct", r["return_pct"])
            elif exit_mode == "sma50":
                active_exit_px = r.get("ml_stop_sma50_exit_price", r.get("ml_stop_adaptive_exit_price", r["exit_price"]))
                active_exit_dt = r.get("ml_stop_sma50_exit_date", r.get("ml_stop_adaptive_exit_date", r["exit_date"]))
                active_reason = r.get("ml_stop_sma50_exit_reason", "Close Below 50 SMA (ML Stop)")
                active_days = r.get("ml_stop_sma50_hold_days", r["holding_days"])
                active_ret = r.get("ml_stop_sma50_ret_pct", r["return_pct"])
            else:
                active_exit_px = r.get("ml_stop_adaptive_exit_price", r["exit_price"])
                active_exit_dt = r.get("ml_stop_adaptive_exit_date", r["exit_date"])
                active_reason = r.get("ml_stop_adaptive_exit_reason", r["exit_reason"])
                active_days = r.get("ml_stop_adaptive_hold_days", r["holding_days"])
                active_ret = r.get("ml_stop_adaptive_ret_pct", r["return_pct"])
        else:
            active_stop_px = safe_float(r["stop_price"], 0.0)
            active_risk_pct = safe_float(r.get("initial_risk_pct", 10.0), 10.0)
            if exit_mode == "ema10":
                active_exit_px = r.get("ema10_exit_price", r.get("adaptive_exit_price", r["exit_price"]))
                active_exit_dt = r.get("ema10_exit_date", r.get("adaptive_exit_date", r["exit_date"]))
                active_reason = r.get("ema10_exit_reason", "Close Below 10 EMA")
                active_days = r.get("ema10_hold_days", r["holding_days"])
                active_ret = r.get("ema10_ret_pct", r.get("adaptive_ret_pct", r["return_pct"]))
            elif exit_mode == "ema20":
                active_exit_px = r.get("ema20_exit_price", r.get("adaptive_exit_price", r["exit_price"]))
                active_exit_dt = r.get("ema20_exit_date", r.get("adaptive_exit_date", r["exit_date"]))
                active_reason = r.get("ema20_exit_reason", "Close Below 20 EMA")
                active_days = r.get("ema20_hold_days", r["holding_days"])
                active_ret = r.get("ema20_ret_pct", r.get("adaptive_ret_pct", r["return_pct"]))
            elif exit_mode == "sma50":
                active_exit_px = r.get("sma50_exit_price", r.get("adaptive_exit_price", r["exit_price"]))
                active_exit_dt = r.get("sma50_exit_date", r.get("adaptive_exit_date", r["exit_date"]))
                active_reason = r.get("sma50_exit_reason", "Close Below 50 SMA")
                active_days = r.get("sma50_hold_days", r["holding_days"])
                active_ret = r.get("sma50_ret_pct", r.get("adaptive_ret_pct", r["return_pct"]))
            else:
                active_exit_px = r.get("adaptive_exit_price", r["exit_price"])
                active_exit_dt = r.get("adaptive_exit_date", r["exit_date"])
                active_reason = r.get("adaptive_exit_reason", r["exit_reason"])
                active_days = r.get("adaptive_holding_days", r["holding_days"])
                active_ret = r.get("adaptive_ret_pct", r["return_pct"])

        r_val = safe_float(r[r_col], 0.0)
        v2_val = safe_float(r.get("v2_breakout", r.get("ml_50", 0.0)), 0.0)
        rs_val = safe_float(r.get("rs_score_breakout", r.get("rs_score", 50.0)), 50.0)

        # Resolve Climax Harvest Details for this trade & exit mode
        if exit_mode == "ema20":
            has_climax = bool(r.get("has_climax_20", False))
            climax_dt = r.get("climax_date_20")
            climax_px = safe_float(r.get("climax_price_20", 0.0))
            climax_r = safe_float(r.get("climax_r_20", 0.0))
        elif exit_mode == "sma50":
            has_climax = bool(r.get("has_climax_50", False))
            climax_dt = r.get("climax_date_50")
            climax_px = safe_float(r.get("climax_price_50", 0.0))
            climax_r = safe_float(r.get("climax_r_50", 0.0))
        else:  # adaptive or ema10
            has_climax = bool(r.get("has_climax_adaptive", False))
            climax_dt = r.get("climax_date_adaptive")
            climax_px = safe_float(r.get("climax_price_adaptive", 0.0))
            climax_r = safe_float(r.get("climax_r_adaptive", 0.0))

        records.append({
            "symbol": r["symbol"],
            "ep_date": str(r["ep_date"])[:10],
            "entry_date": str(r["entry_date"])[:10],
            "exit_date": str(active_exit_dt)[:10],
            "archetype": r.get("primary_archetype", "Core"),
            "sector": r.get("sector", "Unknown"),
            "theme": r.get("theme", "Unknown"),
            "v2_size_mult": safe_float(r.get("v2_size_mult", 1.0), 1.0),
            "ml_50": round(v2_val, 3),
            "v2_breakout": round(v2_val, 3),
            "v2_win_prob": round(safe_float(r.get("v2_win_prob", 0.5), 0.5), 3),
            "v2_conviction_50": round(safe_float(r.get("v2_conviction_50", 0.0), 0.0), 3),
            "rs_score": round(rs_val, 1),
            "rs_score_breakout": round(rs_val, 1),
            "stage2_breakout": bool(r.get("stage2_breakout", r.get("stage2", False))),
            "thm_m1_breakout": round(safe_float(r.get("thm_m1_breakout", 0.0), 0.0), 1),
            "sec_m1_breakout": round(safe_float(r.get("sec_m1_breakout", 0.0), 0.0), 1),
            "entry_price": round(safe_float(r["entry_price"], 0.0), 2),
            "stop_price": round(active_stop_px, 2),
            "ml_stop_price": round(safe_float(r.get("ml_stop_price", r["stop_price"]), 0.0), 2),
            "ml_stop_pct": round(safe_float(r.get("ml_stop_pct", active_risk_pct), 6.0), 1),
            "exit_price": round(safe_float(active_exit_px, 0.0), 2),
            "r_multiple": round(r_val, 2),
            "return_pct": round(safe_float(active_ret, 0.0), 2),
            "holding_days": int(safe_float(active_days, 1)),
            "exit_reason": active_reason,
            "exit_ma": ma_label,
            "tier": r.get("tier", "Core"),
            "is_tier1": bool(r.get("is_tier1", False)),
            "trade_type": "Trade 1: Base EP Delayed Breakout (Days 2-5)",
            "archetype": r.get("primary_archetype", "General"),
            "recommended_ma_name": ma_label,
            "recommended_ma_key": "sma50" if "50" in ma_label else ("ema10" if "10" in ma_label else "ema20"),
            "current_high_exhaustion_pct": round(safe_float(r.get("climax_prob_adaptive" if exit_mode=="adaptive" else f"climax_prob_{exit_mode}"), 0.0) * 100.0, 1),
            "climax_sell_half_rule": "Sell 50% partial profit into extreme high when ML Exhaustion ≥ 50.0%",
            "climax_action": f"🚨 Harvested 50% @ ${climax_px:.2f} on {climax_dt} (+{climax_r:.2f}R)" if has_climax else "None (Pure Trailing MA)",
            "trailing_exit_rule": f"Close below {ma_label.split(' ')[0]} triggers full exit of position",
            "position_management_plan": f"Matched institutional strategy: {r.get('primary_archetype', 'Core')} trailed with {ma_label}.",
            # Climax Harvest Fields
            "has_climax": has_climax,
            "climax_date": str(climax_dt)[:10] if climax_dt and str(climax_dt) not in ["None", ""] else None,
            "climax_price": round(climax_px, 2) if climax_px > 0 else 0.0,
            "climax_r": round(climax_r, 2),
            # Dossier fields
            "pre_dist_52w_high": round(safe_float(r.get("pre_dist_52w_high", 0.0), 0.0), 1),
            "pre_runup_20d": round(safe_float(r.get("pre_runup_20d", 0.0), 0.0), 1),
            "pre_consol_tightness_20d": round(safe_float(r.get("pre_consol_tightness_20d", 0.0), 0.0), 1),
            "gap_pct": round(safe_float(r.get("gap_pct", 0.0), 0.0), 1),
            "rvol": round(safe_float(r.get("rvol", 0.0), 0.0), 1),
            "close_pos": round(safe_float(r.get("close_pos", 0.0), 0.0), 2),
            "day1_upper_wick_pct": round(safe_float(r.get("day1_upper_wick_pct", 0.0), 0.0), 1),
            "dynamic_stop_loss_pct": round(safe_float(active_risk_pct, 7.3), 1),
            "days_to_breakout": int(safe_float(r.get("days_to_breakout", 1), 1)),
            "spy_above_50_entry": bool(r.get("spy_above_50_entry", True)),
            "spy_ret_20d_entry": round(safe_float(r.get("spy_ret_20d_entry", 0.0), 0.0), 1),
            "qqq_above_50_entry": bool(r.get("qqq_above_50_entry", True)),
            "qqq_ret_20d_entry": round(safe_float(r.get("qqq_ret_20d_entry", 0.0), 0.0), 1),
            "qqq_bull_stack": bool(r.get("qqq_bull_stack", True)),
            "qqq_spy_ratio_bull": bool(r.get("qqq_spy_ratio_bull", True)),
            "theme_score": round(safe_float(r.get("theme_score", 50.0), 50.0), 1),
            "sector_score": round(safe_float(r.get("sector_score", 50.0), 50.0), 1),
            "thematic_archetype": str(r.get("thematic_archetype", "Leading Trend")),
            "peak_r": round(safe_float(r.get("peak_r", 0.0), 0.0), 2),
        })

    # Sort
    asc = (sort_dir.lower() == "asc")
    if sort_by in ["r_multiple", "ml_50", "return_pct", "holding_days", "entry_price", "v2_size_mult", "rs_score_breakout", "v2_breakout"]:
        records.sort(key=lambda x: safe_float(x.get(sort_by, 0.0)), reverse=not asc)
    elif sort_by in ["ep_date", "entry_date", "exit_date", "symbol", "archetype"]:
        records.sort(key=lambda x: str(x.get(sort_by, "")), reverse=not asc)
    else:
        records.sort(key=lambda x: safe_float(x.get("r_multiple", 0.0)), reverse=True)

    return jsonify({"trades": records, "count": len(records)})


@app.route("/api/watchlist")
def api_watchlist():
    """Returns active in-flight trades and coiling setups from the last 20-25 trading days."""
    status_filter = request.args.get("status", "all").strip().lower()
    min_rs = float(request.args.get("min_rs", 0.0))
    sort_by = request.args.get("sort_by", "unrealized_r")
    sort_dir = request.args.get("sort_dir", "desc")

    items = get_active_watchlist()
    filtered = []
    for it in items:
        if status_filter == "active" and it["status"] != "Active Long":
            continue
        if status_filter == "coiling" and it["status"] != "Coiling Setup":
            continue
        if min_rs > 0 and it["rs_score"] < min_rs:
            continue
        filtered.append(it)

    asc = (sort_dir.lower() == "asc")
    if sort_by in ["unrealized_r", "unrealized_pct", "dist_to_trigger", "rs_score", "gap_pct", "rvol", "trigger_price", "current_price"]:
        filtered.sort(key=lambda x: safe_float(x.get(sort_by, 0.0)), reverse=not asc)
    elif sort_by in ["ep_date", "trigger_date", "symbol", "status"]:
        filtered.sort(key=lambda x: str(x.get(sort_by, "")), reverse=not asc)
    else:
        filtered.sort(key=lambda x: safe_float(x.get("unrealized_r", 0.0)), reverse=True)

    n_active = sum(1 for x in items if x["status"] == "Active Long")
    n_coiling = sum(1 for x in items if x["status"] == "Coiling Setup")

    intra_meta = {}
    if INTRADAY_CACHE_PATH.exists():
        try:
            cdata = json.loads(INTRADAY_CACHE_PATH.read_text(encoding="utf-8"))
            intra_meta = {
                "market_open": cdata.get("market_open", False),
                "timestamp_et": cdata.get("timestamp_et", ""),
                "alerts_count": cdata.get("alerts_count", 0),
                "active_count": cdata.get("active_count", n_active),
            }
        except Exception:
            pass

    return jsonify({
        "items": filtered,
        "count": len(filtered),
        "total_active": n_active,
        "total_coiling": n_coiling,
        "intraday_meta": intra_meta,
    })


@app.route("/api/intraday_watchlist")
def api_intraday_watchlist():
    """Returns real-time intraday active watchlist status, alerts, and positions telemetry."""
    if not INTRADAY_CACHE_PATH.exists():
        return jsonify({
            "status": "no_cache",
            "market_open": False,
            "active_count": 0,
            "alerts_count": 0,
            "positions": {},
        })
    try:
        data = json.loads(INTRADAY_CACHE_PATH.read_text(encoding="utf-8"))
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/monitor_settings", methods=["GET", "POST"])
def api_monitor_settings():
    """Reads or updates intraday monitor interval and rate-limit settings."""
    if request.method == "POST":
        data = request.get_json() or {}
        interval = int(data.get("poll_interval_sec", data.get("interval", 300)))
        interval = max(60, min(3600, interval))
        payload = {
            "interval_sec": interval,
            "poll_interval_sec": interval,
            "interval_min": round(interval / 60.0, 1),
            "poll_interval_min": round(interval / 60.0, 1),
            "recommended_interval_sec": 300,
            "rate_limit_guidance": "5 to 15-minute polling is optimal for institutional daily EP tracking while preventing Yahoo Finance IP throttling.",
            "updated_at": dt.datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S ET"),
            "suggestion": "5 to 15-minute polling is optimal for institutional daily EP tracking while preventing Yahoo Finance IP throttling."
        }
        MONITOR_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        MONITOR_SETTINGS_FILE.write_text(json.dumps(payload, indent=2))
        return jsonify({"status": "ok", "poll_interval_sec": interval, "settings": payload})

    if MONITOR_SETTINGS_FILE.exists():
        try:
            d = json.loads(MONITOR_SETTINGS_FILE.read_text(encoding="utf-8"))
            sec = d.get("poll_interval_sec", d.get("interval_sec", 300))
            d["poll_interval_sec"] = sec
            d["interval_sec"] = sec
            d["recommended_interval_sec"] = 300
            d["rate_limit_guidance"] = d.get("rate_limit_guidance", "5 to 15-minute polling is optimal for institutional daily EP tracking while preventing Yahoo Finance IP throttling.")
            return jsonify(d)
        except Exception:
            pass
    return jsonify({
        "interval_sec": 300,
        "poll_interval_sec": 300,
        "interval_min": 5.0,
        "poll_interval_min": 5.0,
        "recommended_interval_sec": 300,
        "rate_limit_guidance": "5 to 15-minute polling is optimal for institutional daily EP tracking while preventing Yahoo Finance IP throttling.",
        "updated_at": "",
        "suggestion": "5 to 15-minute polling is optimal for institutional daily EP tracking while preventing Yahoo Finance IP throttling."
    })


@app.route("/api/run_monitor_cycle", methods=["POST"])
def api_run_monitor_cycle():
    """Triggers an immediate intraday evaluation cycle."""
    try:
        import intraday_watchlist_monitor
        res = intraday_watchlist_monitor.run_intraday_cycle(force=True)
        return jsonify({"status": "ok", "result": res})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chart")
def api_chart():
    """Generates TradingView Lightweight Charts payload with candlesticks, full EMAs, Larsson ribbon, and catalyst."""
    symbol = request.args.get("symbol", "").strip().upper()
    ep_date = request.args.get("ep_date", "").strip()
    entry_date = request.args.get("entry_date", "").strip()
    exit_date = request.args.get("exit_date", "").strip()
    exit_mode = request.args.get("exit_mode", "adaptive")
    history_mode = request.args.get("history", "setup").strip().lower()

    try:
        larsson_fastest = int(request.args.get("larsson_fastest", "32"))
    except (ValueError, TypeError):
        larsson_fastest = 32
    larsson_fastest = max(3, min(80, larsson_fastest))

    if not symbol:
        return jsonify({"error": "Symbol required"}), 400

    bars = datastore.load_bars(symbol)
    if bars is None or bars.empty:
        return jsonify({"error": f"No bar data available for {symbol}"}), 404

    bars = bars.copy()
    bars["ema8"] = bars["close"].ewm(span=8, adjust=False).mean()
    bars["ema10"] = bars["close"].ewm(span=10, adjust=False).mean()
    bars["ema20"] = bars["close"].ewm(span=20, adjust=False).mean()
    bars["ema21"] = bars["close"].ewm(span=21, adjust=False).mean()
    bars["sma50"] = bars["close"].rolling(50, min_periods=20).mean()
    bars["sma200"] = bars["close"].rolling(200, min_periods=50).mean()
    bars["vol_ma20"] = bars["volume"].rolling(20, min_periods=5).mean()

    # Larsson Line Ribbon calculation
    if larsson_fastest == 32:
        larsson_spans = (32, 35, 50, 58)
    elif larsson_fastest <= 20:
        larsson_spans = (larsson_fastest, int(round(larsson_fastest * 1.5)), int(round(larsson_fastest * 2.0)), int(round(larsson_fastest * 2.625)))
    else:
        larsson_spans = (larsson_fastest, int(round(larsson_fastest * 35.0 / 32.0)), int(round(larsson_fastest * 50.0 / 32.0)), int(round(larsson_fastest * 58.0 / 32.0)))

    s1, s2, s3, s4 = larsson_spans
    e1 = bars["close"].ewm(span=s1, adjust=False).mean()
    e2 = bars["close"].ewm(span=s2, adjust=False).mean()
    e3 = bars["close"].ewm(span=s3, adjust=False).mean()
    e4 = bars["close"].ewm(span=s4, adjust=False).mean()
    bullish = (e1 >= e2) & (e2 >= e3) & (e3 >= e4)
    bearish = (e1 < e2) & (e2 < e3) & (e3 < e4)
    valid_l = e1.notna() & e2.notna() & e3.notna() & e4.notna()

    bars["larsson_fast"] = e1
    bars["larsson_slow"] = e4
    bars["larsson_state"] = "gray"
    bars.loc[valid_l & bullish, "larsson_state"] = "yellow"
    bars.loc[valid_l & bearish, "larsson_state"] = "blue"

    # Slice according to history depth setting
    if history_mode == "full":
        view_bars = bars.copy()
    elif history_mode == "2y":
        view_bars = bars.tail(504).copy()
    elif history_mode == "1y":
        view_bars = bars.tail(252).copy()
    elif history_mode == "6m":
        view_bars = bars.tail(126).copy()
    else:  # "setup"
        ep_matches = np.where(bars.index.strftime("%Y-%m-%d") == ep_date)[0]
        if len(ep_matches) > 0:
            ep_idx = int(ep_matches[0])
            start_idx = max(0, ep_idx - 65)  # 65 bars prior (chart to left)
            if exit_date and exit_date not in ["None", "Pending", ""]:
                end_matches = np.where(bars.index.strftime("%Y-%m-%d") == exit_date)[0]
                end_idx = min(len(bars), int(end_matches[0]) + 30) if len(end_matches) > 0 else min(len(bars), ep_idx + 180)
            else:
                end_idx = len(bars)
            view_bars = bars.iloc[start_idx:end_idx].copy()
        else:
            view_bars = bars.tail(250).copy()

    # Format Candlesticks & Indicators
    candle_data = []
    volume_data = []
    vol_ma20_data = []
    ema8_data = []
    ema10_data = []
    ema20_data = []
    ema21_data = []
    sma50_data = []
    sma200_data = []
    larsson_fast_data = []
    larsson_slow_data = []

    for t, r in view_bars.iterrows():
        t_str = t.strftime("%Y-%m-%d")
        c = float(r["close"])
        o = float(r["open"])
        candle_data.append({
            "time": t_str, "open": o, "high": float(r["high"]), "low": float(r["low"]), "close": c
        })
        volume_data.append({
            "time": t_str, "value": float(r["volume"]),
            "color": "rgba(34, 197, 94, 0.4)" if c >= o else "rgba(239, 68, 68, 0.4)"
        })
        if pd.notna(r.get("vol_ma20")):
            vol_ma20_data.append({"time": t_str, "value": round(float(r["vol_ma20"]), 0)})
        if pd.notna(r.get("ema8")):
            ema8_data.append({"time": t_str, "value": round(float(r["ema8"]), 2)})
        if pd.notna(r.get("ema10")):
            ema10_data.append({"time": t_str, "value": round(float(r["ema10"]), 2)})
        if pd.notna(r.get("ema20")):
            ema20_data.append({"time": t_str, "value": round(float(r["ema20"]), 2)})
        if pd.notna(r.get("ema21")):
            ema21_data.append({"time": t_str, "value": round(float(r["ema21"]), 2)})
        if pd.notna(r.get("sma50")):
            sma50_data.append({"time": t_str, "value": round(float(r["sma50"]), 2)})
        if pd.notna(r.get("sma200")):
            sma200_data.append({"time": t_str, "value": round(float(r["sma200"]), 2)})
        if pd.notna(r.get("larsson_fast")):
            larsson_fast_data.append({"time": t_str, "value": round(float(r["larsson_fast"]), 2), "state": r.get("larsson_state", "gray")})
        if pd.notna(r.get("larsson_slow")):
            larsson_slow_data.append({"time": t_str, "value": round(float(r["larsson_slow"]), 2), "state": r.get("larsson_state", "gray")})

    # Fetch Fundamental Catalyst Dossier & SEC EDGAR Filings
    try:
        cat_dossier = ep_catalyst_enricher.get_catalyst_dossier(symbol, ep_date)
    except Exception:
        cat_dossier = None

    return jsonify({
        "symbol": symbol,
        "candles": candle_data,
        "volume": volume_data,
        "vol_ma20": vol_ma20_data,
        "ema8": ema8_data,
        "ema10": ema10_data,
        "ema20": ema20_data,
        "ema21": ema21_data,
        "sma50": sma50_data,
        "sma200": sma200_data,
        "larsson_fast": larsson_fast_data,
        "larsson_slow": larsson_slow_data,
        "larsson_spans": list(larsson_spans),
        "catalyst": cat_dossier,
        "history_mode": history_mode,
        "ep_date": ep_date,
        "entry_date": entry_date,
        "exit_date": exit_date,
    })


@app.route("/")
def index():
    """Renders the single-page application dashboard."""
    return render_template_string(HTML_TEMPLATE)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Institutional Optimized EP Strategy Explorer (Port 8787)</title>
  <script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    :root {
      --bg: #090c13;
      --card: #111520;
      --card-border: #1e2638;
      --accent: #38bdf8;
      --accent-hover: #0284c7;
      --green: #22c55e;
      --red: #ef4444;
      --amber: #f59e0b;
      --purple: #c084fc;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --border: #1e293b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column; overflow: hidden; }

    /* Top Navigation & KPI Header */
    .top-header { background: #0c101a; border-bottom: 1px solid var(--border); padding: 8px 16px; flex-shrink: 0; }
    .header-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .logo-area { display: flex; align-items: center; gap: 10px; }
    .logo-title { font-size: 15px; font-weight: 700; color: #fff; letter-spacing: 0.3px; }
    .logo-badge { background: rgba(56, 189, 248, 0.15); color: var(--accent); border: 1px solid rgba(56, 189, 248, 0.3); font-size: 11px; padding: 2px 7px; border-radius: 4px; font-weight: 600; }

    /* KPI Summary Row */
    .kpi-grid { display: grid; grid-template-columns: repeat(7, 1fr); gap: 8px; }
    .kpi-card { background: var(--card); border: 1px solid var(--card-border); padding: 6px 10px; border-radius: 6px; }
    .kpi-lbl { font-size: 10px; color: var(--text-muted); text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; }
    .kpi-val { font-size: 15px; font-weight: 700; color: #fff; margin-top: 2px; }
    .kpi-val.pos { color: var(--green); }
    .kpi-val.neg { color: var(--red); }

    /* Filter Controls Bar */
    .filter-bar { background: #0f1422; border-bottom: 1px solid var(--border); padding: 8px 16px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap; flex-shrink: 0; }
    .ctl-group { display: flex; align-items: center; gap: 6px; }
    .ctl-label { font-size: 11px; color: var(--text-muted); font-weight: 600; text-transform: uppercase; }
    
    .btn-toggle { background: #1a2234; border: 1px solid #27344d; color: var(--text-muted); padding: 4px 10px; font-size: 11px; border-radius: 4px; cursor: pointer; transition: all 0.15s; font-weight: 600; }
    .btn-toggle:hover { background: #232e46; color: #fff; }
    .btn-toggle.active { background: rgba(56, 189, 248, 0.2); border-color: var(--accent); color: var(--accent); }
    .btn-toggle.active-tier1 { background: rgba(34, 197, 94, 0.2); border-color: var(--green); color: var(--green); }

    select.ctl-select { background: #1a2234; border: 1px solid #27344d; color: #fff; font-size: 11px; padding: 4px 8px; border-radius: 4px; outline: none; }
    select.ctl-select:focus { border-color: var(--accent); }

    /* Main 3-Pane Layout */
    .workspace { flex: 1; display: flex; min-height: 0; overflow: hidden; position: relative; }

    /* Left Pane: Table */
    .pane-table { flex: 0 0 380px; width: 380px; min-width: 380px; border-right: 1px solid var(--border); display: flex; flex-direction: column; background: #0b0e17; z-index: 5; }
    .table-header { padding: 6px 12px; background: #0e1320; border-bottom: 1px solid var(--border); font-size: 11px; font-weight: 600; color: var(--text-muted); display: flex; justify-content: space-between; flex-shrink: 0; }
    .table-scroll { flex: 1; overflow-y: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 11px; }
    th { position: sticky; top: 0; background: #131929; color: var(--text-muted); font-weight: 600; text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--border); z-index: 2; cursor: pointer; }
    th.tl { text-align: left; }
    td { padding: 6px 8px; border-bottom: 1px solid #141b2a; text-align: right; white-space: nowrap; }
    td.tl { text-align: left; }
    tr { cursor: pointer; transition: background 0.1s; }
    tr:hover { background: #172033; }
    tr.selected { background: #1e293b; border-left: 3px solid var(--accent); }
    .badge-arch { font-size: 9px; padding: 1px 4px; border-radius: 3px; font-weight: 600; background: #1e293b; color: #cbd5e1; }
    .badge-sz { font-size: 9px; padding: 1px 4px; border-radius: 3px; font-weight: 700; }
    .sz-15 { background: rgba(34, 197, 94, 0.2); color: var(--green); }
    .sz-125 { background: rgba(56, 189, 248, 0.2); color: var(--accent); }
    .sz-10 { background: rgba(148, 163, 184, 0.2); color: #cbd5e1; }
    .sz-075 { background: rgba(239, 68, 68, 0.2); color: var(--red); }

    /* Center Pane: Chart */
    .pane-chart { flex: 1; border-right: 1px solid var(--border); display: flex; flex-direction: column; background: #090c13; min-width: 0; overflow: hidden; position: relative; }
    .chart-top-bar { padding: 5px 12px; background: #0c101a; border-bottom: 1px solid var(--border); display: flex; flex-direction: column; gap: 4px; flex-shrink: 0; min-height: 56px; }
    .chart-header-row { display: flex; justify-content: space-between; align-items: center; gap: 8px; width: 100%; min-width: 0; }
    .chart-sub-row { display: flex; justify-content: space-between; align-items: center; gap: 8px; width: 100%; min-width: 0; border-top: 1px solid rgba(255, 255, 255, 0.04); padding-top: 3px; }
    .chart-meta-group { display: flex; align-items: center; gap: 8px; min-width: 0; flex-shrink: 1; }
    .chart-ticker { font-size: 16px; font-weight: 800; color: #fff; letter-spacing: 0.5px; flex-shrink: 0; }
    .chart-sub { font-size: 11px; color: var(--text-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 280px; }
    .chart-nav-group { display: flex; align-items: center; gap: 6px; flex-shrink: 0; }
    .btn-nav { background: #131929; border: 1px solid #27344d; color: #cbd5e1; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 4px; cursor: pointer; display: inline-flex; align-items: center; gap: 3px; transition: all 0.15s ease; }
    .btn-nav:hover:not(:disabled) { background: #1e293b; border-color: var(--accent); color: #fff; }
    .btn-nav:disabled { opacity: 0.35; cursor: not-allowed; }
    .nav-stock-select { max-width: 220px; font-size: 11px; font-weight: 600; padding: 2px 7px; background: #131929; border: 1px solid #27344d; color: #fff; border-radius: 4px; outline: none; text-overflow: ellipsis; }
    .nav-counter { font-size: 10.5px; font-weight: 700; color: var(--accent); background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.25); padding: 2px 6px; border-radius: 4px; white-space: nowrap; }
    .fs-hint { font-size: 10px; color: #64748b; margin-left: 6px; white-space: nowrap; }
    .chart-controls-group { display: flex; align-items: center; gap: 8px; font-size: 11px; position: relative; flex-shrink: 0; }
    .btn-fullscreen-active { background: rgba(56, 189, 248, 0.15) !important; border-color: var(--accent) !important; color: var(--accent) !important; font-weight: 700 !important; }
    .btn-fullscreen-active:hover { background: rgba(56, 189, 248, 0.3) !important; }
    .chart-legends-bar { font-size: 10.5px; display: flex; gap: 8px; align-items: center; margin-left: 4px; white-space: nowrap; }
    .chart-container { flex: 1; min-height: 0; width: 100%; height: 100%; position: relative; overflow: hidden; }

    /* Right Pane: Dossier */
    .pane-dossier { flex: 0 0 350px; width: 350px; min-width: 350px; background: #0b0e17; overflow-y: auto; overflow-x: hidden; padding: 12px; display: flex; flex-direction: column; gap: 10px; z-index: 10; position: relative; box-shadow: -2px 0 10px rgba(0, 0, 0, 0.5); }
    .d-card { background: var(--card); border: 1px solid var(--card-border); border-radius: 6px; padding: 10px; }
    .d-card-title { font-size: 11px; font-weight: 700; color: var(--accent); text-transform: uppercase; margin-bottom: 8px; display: flex; justify-content: space-between; }
    .d-row { display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 5px; }
    .d-row:last-child { margin-bottom: 0; }
    .d-lbl { color: var(--text-muted); }
    .d-val { font-weight: 600; color: #fff; }
    .d-val.pos { color: var(--green); }
    .d-val.neg { color: var(--red); }
    .d-val.hl { color: var(--accent); font-weight: 700; }
    .badge-status { font-size: 10px; padding: 2px 6px; border-radius: 4px; font-weight: 700; }
    .status-active { background: rgba(34, 197, 94, 0.2); color: var(--green); border: 1px solid rgba(34, 197, 94, 0.4); }
    .status-coiling { background: rgba(245, 158, 11, 0.2); color: var(--amber); border: 1px solid rgba(245, 158, 11, 0.4); }
    @keyframes alertPulse {
      0% { box-shadow: 0 0 0 0 rgba(192, 132, 252, 0.7); }
      70% { box-shadow: 0 0 0 8px rgba(192, 132, 252, 0); }
      100% { box-shadow: 0 0 0 0 rgba(192, 132, 252, 0); }
    }
    .row-alert-pulse {
      background: rgba(192, 132, 252, 0.15) !important;
      border-left: 3px solid #c084fc !important;
    }
    .badge-alert-pulse {
      background: #c084fc;
      color: #0b0f19;
      font-weight: 800;
      font-size: 9px;
      padding: 2px 6px;
      border-radius: 4px;
      display: inline-block;
      animation: alertPulse 1.5s infinite;
      margin-left: 4px;
    }
    .badge-elevated {
      background: rgba(245, 158, 11, 0.2);
      color: #fbbf24;
      border: 1px solid rgba(245, 158, 11, 0.4);
      font-weight: 700;
      font-size: 9px;
      padding: 2px 5px;
      border-radius: 4px;
      margin-left: 4px;
    }
    .monitor-indicator {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 14px;
      font-size: 11px;
      font-weight: 700;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.1);
      color: var(--text-muted);
    }
    .monitor-indicator.live {
      background: rgba(34, 197, 94, 0.12);
      border-color: rgba(34, 197, 94, 0.3);
      color: #4ade80;
    }
    .monitor-indicator.alerting {
      background: rgba(192, 132, 252, 0.2);
      border-color: #c084fc;
      color: #e9d5ff;
      animation: alertPulse 1.5s infinite;
    }

    /* Fullscreen Chart Mode */
    .pane-chart.fullscreen {
      position: fixed !important;
      inset: 0 !important;
      width: 100vw !important;
      height: 100vh !important;
      z-index: 9999 !important;
      background: #090c13 !important;
    }
    .pane-chart.fullscreen .chart-top-bar {
      display: flex !important;
      flex-direction: row !important;
      justify-content: space-between !important;
      align-items: center !important;
      padding: 10px 24px !important;
      background: #0c101a !important;
      border-bottom: 1px solid #1e293b !important;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6) !important;
      min-height: 52px !important;
    }
    .pane-chart.fullscreen .chart-header-row { width: auto !important; gap: 16px !important; }
    .pane-chart.fullscreen .chart-sub-row { width: auto !important; gap: 16px !important; border-top: none !important; padding-top: 0 !important; }
    .pane-chart.fullscreen .chart-ticker { font-size: 22px !important; }
    .pane-chart.fullscreen .chart-sub { max-width: 500px !important; font-size: 12px !important; }
    .pane-chart.fullscreen .nav-stock-select { max-width: 340px !important; font-size: 12px !important; padding: 4px 10px !important; }
    .pane-chart.fullscreen .btn-nav { font-size: 12px !important; padding: 4px 12px !important; }
    .pane-chart.fullscreen .btn-toggle { font-size: 12px !important; padding: 4px 12px !important; }
    .pane-chart.fullscreen .chart-legends-bar { font-size: 11.5px !important; gap: 12px !important; }
    .pane-chart.fullscreen .fs-hint { display: inline-block !important; }

    /* Popover Settings Panels */
    .settings-popover {
      position: absolute;
      background: #0f172a;
      border: 1px solid #1e293b;
      border-radius: 8px;
      box-shadow: 0 12px 30px rgba(0, 0, 0, 0.8);
      z-index: 100;
      padding: 12px 14px;
      width: 320px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .popover-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 12px;
      font-weight: 700;
      color: #fff;
      border-bottom: 1px solid #1e293b;
      padding-bottom: 6px;
    }
    .popover-close {
      background: none;
      border: none;
      color: var(--text-muted);
      font-size: 16px;
      cursor: pointer;
      line-height: 1;
    }
    .popover-close:hover { color: #fff; }
    .popover-sec-title {
      font-size: 10px;
      font-weight: 700;
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 4px;
    }
    .popover-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 11px;
      margin-bottom: 4px;
    }
    .popover-row label {
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      color: #cbd5e1;
      font-size: 11px;
    }
    .suggestion-box {
      background: rgba(56, 189, 248, 0.08);
      border: 1px solid rgba(56, 189, 248, 0.25);
      border-radius: 6px;
      padding: 8px;
      font-size: 10.5px;
      color: #cbd5e1;
      line-height: 1.4;
    }
    .filing-item {
      background: #131826;
      border: 1px solid #202b3f;
      border-radius: 4px;
      padding: 5px 8px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 10.5px;
      margin-bottom: 4px;
    }
    .filing-link {
      color: #38bdf8;
      text-decoration: none;
      font-weight: 700;
      font-size: 10.5px;
      flex-shrink: 0;
    }
    .filing-link:hover { text-decoration: underline; }
  </style>
</head>
<body>

  <!-- Top Header with Real-Time Strategy KPIs -->
  <div class="top-header">
    <div class="header-row">
      <div class="logo-area">
        <span class="logo-title">⚡ INSTITUTIONAL OPTIMIZED EP EXPLORER</span>
        <span class="logo-badge">PORT 8787</span>
        <span style="font-size:11px; color:var(--text-muted);">20-Day Breakout · Dynamic V2 Sizing · Matched MA Trailing Exits</span>
      </div>
      <div style="display:flex; align-items:center; gap:8px;">
        <button class="btn-toggle active" id="btn_mode_closed" onclick="setMainMode('closed')">📜 Closed Strategy Trades (947)</button>
        <button class="btn-toggle" id="btn_mode_watchlist" onclick="setMainMode('watchlist')">🎯 Active & Coiling Watchlist (<span id="btn_wl_count">55</span>)</button>
        <button class="btn-toggle" onclick="reloadDataset()">🔄 Reload Dataset</button>
      </div>
    </div>
    <div class="kpi-grid">
      <div class="kpi-card"><div class="kpi-lbl" id="kpi_lbl_1">QUALIFIED TRADES</div><div class="kpi-val" id="kpi_val_1">-</div></div>
      <div class="kpi-card"><div class="kpi-lbl" id="kpi_lbl_2">STRATEGY WIN RATE</div><div class="kpi-val pos" id="kpi_val_2">-</div></div>
      <div class="kpi-card"><div class="kpi-lbl" id="kpi_lbl_3">EXPECTANCY (EV)</div><div class="kpi-val pos" id="kpi_val_3">-</div></div>
      <div class="kpi-card"><div class="kpi-lbl" id="kpi_lbl_4">TOTAL REALIZED P&L</div><div class="kpi-val pos" id="kpi_val_4">-</div></div>
      <div class="kpi-card"><div class="kpi-lbl" id="kpi_lbl_5">PROFIT FACTOR</div><div class="kpi-val pos" id="kpi_val_5">-</div></div>
      <div class="kpi-card"><div class="kpi-lbl" id="kpi_lbl_6">PAYOFF RATIO</div><div class="kpi-val" id="kpi_val_6">-</div></div>
      <div class="kpi-card"><div class="kpi-lbl" id="kpi_lbl_7">MAX DRAWDOWN</div><div class="kpi-val neg" id="kpi_val_7">-</div></div>
    </div>
  </div>

  <!-- Filter & Mode Bar -->
  <div class="filter-bar">
    <!-- Closed Strategy Mode Filters -->
    <div id="filters_closed" style="display:flex; align-items:center; gap:12px; flex-wrap:wrap;">
      <div class="ctl-group">
        <span class="ctl-label">PORTFOLIO TIER:</span>
        <button class="btn-toggle active" id="btn_tier_all" onclick="setTier('all')">🛡️ Institutional Core (947 Trades)</button>
        <button class="btn-toggle" id="btn_tier_1" onclick="setTier('tier1')">🎯 Tier 1 Ultra-Elite (341 Trades)</button>
      </div>

      <div class="ctl-group" style="margin-left: 8px;">
        <span class="ctl-label">TRAILING EXIT MA:</span>
        <select class="ctl-select" id="sel_exit_mode" onchange="onFilterChange()">
          <option value="adaptive" selected>🤖 Adaptive Match (By Archetype)</option>
          <option value="ema20">📈 Pure 20 EMA Trailing Exit</option>
          <option value="sma50">🏛️ Pure 50 SMA Institutional Baseline</option>
          <option value="ema10">⚡ Pure 10 EMA Fast Trailing Exit</option>
        </select>
      </div>

      <div class="ctl-group">
        <span class="ctl-label">SIZING:</span>
        <select class="ctl-select" id="sel_sizing" onchange="onFilterChange()">
          <option value="v2" selected>⚖️ V2 Conviction Weighted (0.75x - 1.50x)</option>
          <option value="flat">Flat 1.0 R (Unweighted)</option>
        </select>
      </div>

      <div class="ctl-group">
        <span class="ctl-label">ARCHETYPE:</span>
        <select class="ctl-select" id="sel_archetype" onchange="onFilterChange()">
          <option value="all" selected>All 7 Archetypes</option>
          <option value="Pinnacle Elite">👑 Pinnacle Elite</option>
          <option value="Emerging Leaders">🌟 Emerging Leaders</option>
          <option value="Institutional Sweet Spot">⭐ Institutional Sweet Spot</option>
          <option value="Multi Quarter Leaders">🚀 Multi-Quarter Leaders</option>
          <option value="Conservative Swing">🛡️ Conservative Swing</option>
          <option value="High Beta Momentum">⚡ High Beta Momentum</option>
          <option value="Idiosyncratic Alpha">🎯 Idiosyncratic Alpha</option>
        </select>
      </div>

      <div class="ctl-group">
        <span class="ctl-label">MIN V2:</span>
        <select class="ctl-select" id="sel_min_v2" onchange="onFilterChange()">
          <option value="0.0" selected>All</option>
          <option value="0.20">Exclude Drag (&ge; 0.20)</option>
          <option value="0.25">High Conviction (&ge; 0.25)</option>
          <option value="0.35">Elite Only (&ge; 0.35)</option>
        </select>
      </div>

      <div class="ctl-group">
        <span class="ctl-label">MIN RS:</span>
        <select class="ctl-select" id="sel_min_rs" onchange="onFilterChange()">
          <option value="0.0" selected>All</option>
          <option value="80.0">&ge; 80 (Leaders)</option>
          <option value="85.0">&ge; 85 (High RS)</option>
          <option value="90.0">&ge; 90 (Elite)</option>
          <option value="95.0">&ge; 95 (Top 5%)</option>
        </select>
      </div>

      <div class="ctl-group">
        <span class="ctl-label">STOP LOSS:</span>
        <select class="ctl-select" id="sel_stop_mode" onchange="onFilterChange()">
          <option value="structural" selected>🛡️ Structural Stop (D1 Low)</option>
          <option value="ml_adaptive">🤖 ML Adaptive Stop (20D MAE Bounded 4.5% - D1 Low)</option>
        </select>
      </div>

      <div class="ctl-group">
        <span class="ctl-label">CLIMAX HARVEST:</span>
        <select class="ctl-select" id="sel_climax" onchange="onFilterChange()">
          <option value="none" selected>Off (Pure Trailing MA)</option>
          <option value="partial_50">⭐ 50% Extreme High Harvest (ML Climax)</option>
        </select>
      </div>

      <div class="ctl-group">
        <span class="ctl-label">PROGRESSIVE EXPOSURE:</span>
        <select class="ctl-select" id="sel_prog" onchange="onFilterChange()">
          <option value="none" selected>Off (Static V2 Sizing)</option>
          <option value="qtr_confluence">⭐ Recommended: Confluence + 0.25x Defense (PF 1.81, MaxDD -22.5R)</option>
          <option value="qtr_defense">🛡️ Quarter-Size Defense Ladder (PF 1.82, MaxDD -21.1R)</option>
          <option value="confluence">🔥 V2 x Progressive State (+280.1R, MaxDD -22.5R)</option>
          <option value="momentum">⚡ Rolling 5-Trade Momentum (+310.2R, PF 1.62)</option>
          <option value="ladder">🪜 Classic 3-Tier Ladder (MaxDD -24.8R, PF 1.64)</option>
          <option value="equity_ma">📈 Equity Curve 10-MA Governor (+305.4R, PF 1.68)</option>
        </select>
      </div>
    </div>

    <!-- Active Watchlist Mode Filters -->
    <div id="filters_watchlist" style="display:none; align-items:center; gap:12px; flex-wrap:wrap;">
      <div class="ctl-group">
        <span class="ctl-label">STATUS:</span>
        <select class="ctl-select" id="sel_wl_status" onchange="loadWatchlistData()">
          <option value="all" selected>All Watchlist Setups</option>
          <option value="active">🟢 Active Long Positions Only</option>
          <option value="coiling">⏳ Coiling Breakout Setups Only</option>
        </select>
      </div>

      <div class="ctl-group">
        <span class="ctl-label">MIN RS RATING:</span>
        <select class="ctl-select" id="sel_wl_min_rs" onchange="loadWatchlistData()">
          <option value="0.0" selected>All RS Scores</option>
          <option value="70.0">&ge; 70 (Broad Leaders)</option>
          <option value="80.0">&ge; 80 (RS Leaders)</option>
          <option value="90.0">&ge; 90 (Elite Top Decile)</option>
        </select>
      </div>

      <div class="ctl-group">
        <span class="ctl-label">STAGE 2 POSTURE:</span>
        <select class="ctl-select" id="sel_wl_stage2" onchange="loadWatchlistData()">
          <option value="all" selected>All Postures</option>
          <option value="stage2">Stage 2 Bullish Only (Price > 50 > 200)</option>
        </select>
      </div>

      <div style="display:flex; align-items:center; gap:8px; margin-left:auto; position:relative;">
        <button class="btn-toggle" id="btn_monitor_settings" onclick="toggleMonitorSettings()" title="Configure intraday monitor polling interval & rate limit safety">⚙️ Detector Settings</button>
        <div id="monitor_pill" class="monitor-indicator">
          <span id="monitor_dot" style="display:inline-block; width:7px; height:7px; border-radius:50%; background:#94a3b8;"></span>
          <span id="monitor_text">⚡ Intraday Monitor: Standby</span>
        </div>

        <!-- Detector Settings Popover -->
        <div id="monitor_settings_panel" class="settings-popover" style="display:none; top:36px; right:0;">
          <div class="popover-header">
            <span>⚡ Intraday Exhaustion Detector</span>
            <button class="popover-close" onclick="toggleMonitorSettings()">&times;</button>
          </div>
          <div>
            <div class="popover-sec-title">POLLING INTERVAL &amp; RATE-LIMIT PROTECTION</div>
            <div class="popover-row">
              <span>Check Interval:</span>
              <select id="sel_monitor_interval" class="ctl-select" onchange="saveMonitorInterval(this.value)">
                <option value="300" selected>5 Min (Recommended: Safe &amp; Responsive)</option>
                <option value="600">10 Min (Standard Institutional)</option>
                <option value="900">15 Min (Defensive / Minimal Footprint)</option>
                <option value="120">2 Min (Fast Scalping / Elevated IP Risk)</option>
              </select>
            </div>
            <div class="suggestion-box" style="margin-top:6px;">
              💡 <strong>Institutional Guidance</strong>: YFinance imposes hourly IP rate blocks if polled continuously under 1 minute. A <strong>5 to 15 minute</strong> interval allows high-fidelity detection of intraday exhaustion spikes while guaranteeing 100% uninterrupted uptime across 40+ tickers.
            </div>
          </div>
          <div style="border-top:1px solid #1e293b; padding-top:8px;">
            <div class="popover-sec-title">ON-DEMAND MANUAL REFRESH</div>
            <button class="btn-toggle" style="width:100%; justify-content:center; display:flex;" onclick="runMonitorCycleNow()">⚡ Run Check Cycle Now</button>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Workspace -->
  <div class="workspace">
    <!-- Left Pane: Leaderboard Table -->
    <div class="pane-table">
      <div class="table-header">
        <span id="tbl_count_lbl">QUALIFIED TRADES</span>
        <span id="tbl_sort_lbl" style="font-size:10px; color:var(--accent); font-weight:700;">SORT: REALIZED R ▼</span>
      </div>
      <div class="table-scroll">
        <table>
          <thead id="tbl_head">
            <tr>
              <th class="tl" onclick="sortTable('symbol')" title="Sort by Symbol">TICKER</th>
              <th class="tl" onclick="sortTable('archetype')" title="Sort by Archetype">ARCHETYPE</th>
              <th onclick="sortTable('rs_score_breakout')" title="Sort by Breakout RS">RS</th>
              <th onclick="sortTable('v2_breakout')" title="Sort by Breakout V2 Conviction">V2</th>
              <th onclick="sortTable('v2_size_mult')" title="Sort by Position Size Multiplier">SIZE</th>
              <th onclick="sortTable('return_pct')" title="Sort by Return %">RET %</th>
              <th onclick="sortTable('r_multiple')" title="Sort by Realized R Multiple">RESULT (R)</th>
              <th onclick="sortTable('entry_date')" style="color:var(--accent);" title="Sort by Entry Date">DATE ⬍</th>
            </tr>
          </thead>
          <tbody id="trade_rows"></tbody>
        </table>
      </div>
    </div>

    <!-- Center Pane: TradingView Chart -->
    <div class="pane-chart" id="pane_chart">
      <div class="chart-top-bar" id="chart_top_bar">
        <!-- Top Row: Ticker Info (Left) & Controls/Fullscreen (Right) -->
        <div class="chart-header-row">
          <div class="chart-meta-group">
            <span class="chart-ticker" id="c_ticker">-</span>
            <span id="c_badge_status" class="badge-status status-active" style="display:none;">Active</span>
            <span id="c_badge_arch" class="badge-arch" style="display:none; font-size:9.5px; padding:2px 5px;">-</span>
            <span class="chart-sub" id="c_meta" title="">-</span>
          </div>
          <div class="chart-controls-group">
            <div class="ctl-group" style="gap:4px;">
              <span class="ctl-label" style="font-size:10px;">HISTORY:</span>
              <select id="sel_chart_history" class="ctl-select" onchange="onChartHistoryChange()" style="font-size:10px; padding:2px 6px;">
                <option value="setup" selected>Setup Focus (-40D/+60D)</option>
                <option value="6m">6 Months</option>
                <option value="1y">1 Year</option>
                <option value="2y">2 Years</option>
                <option value="full">Full History</option>
              </select>
            </div>
            <button class="btn-toggle" id="btn_chart_settings" onclick="toggleChartSettings()" title="Configure Moving Averages, Larsson Ribbon &amp; Indicators" style="font-size:10.5px; padding:2px 7px;">⚙️ Indicators</button>
            <button class="btn-toggle btn-fullscreen-active" id="btn_chart_fullscreen" onclick="toggleChartFullscreen()" title="Toggle Fullscreen Chart" style="font-size:10.5px; padding:2px 8px;">⛶ Fullscreen</button>
          </div>
        </div>

        <!-- Sub Row: Navigation Controls (Left) & Price Line Legends (Right) -->
        <div class="chart-sub-row">
          <div class="chart-nav-group">
            <button class="btn-nav" id="btn_nav_prev" onclick="selectPreviousTrade()" title="Previous Stock (← arrow key)">◀ Prev</button>
            <select id="sel_nav_stock" class="ctl-select nav-stock-select" onchange="onNavStockSelect(this.value)" title="Select Stock from Current List"></select>
            <button class="btn-nav" id="btn_nav_next" onclick="selectNextTrade()" title="Next Stock (→ arrow key)">Next ▶</button>
            <span class="nav-counter" id="lbl_nav_counter" title="Position in current filtered list">1 / 54</span>
            <span id="fs_keyboard_hint" class="fs-hint" style="display:none;">(← / → browse · Esc exit)</span>
          </div>
          <div class="chart-legends-bar">
            <span id="c_trigger_legend" style="color:#22c55e;">── Breakout</span>
            <span id="c_stop_legend" style="color:#ef4444;">── Stop</span>
            <span id="c_ma_legend" style="color:#0284c7;">── Trail MA</span>
            <span id="c_climax_legend" style="color:#c084fc; display:none;">── 50% Climax</span>
          </div>
        </div>

        <!-- Chart Settings Popover -->
        <div id="chart_settings_panel" class="settings-popover" style="display:none; top:36px; right:12px;">
          <div class="popover-header">
            <span>⚙️ Chart Indicators &amp; Ribbons</span>
            <button class="popover-close" onclick="toggleChartSettings()">&times;</button>
          </div>
            <div>
              <div class="popover-sec-title">📈 MOVING AVERAGES</div>
              <div class="popover-row"><label><input type="checkbox" id="chk_ema8" onchange="toggleIndicatorSeries()"> <span style="color:#ec4899;">8 EMA (Fast Momentum)</span></label></div>
              <div class="popover-row"><label><input type="checkbox" id="chk_ema10" onchange="toggleIndicatorSeries()"> <span style="color:#f59e0b;">10 EMA (Fast Velocity Trail)</span></label></div>
              <div class="popover-row"><label><input type="checkbox" id="chk_ema20" onchange="toggleIndicatorSeries()"> <span style="color:#38bdf8;">20 EMA (Swing Baseline)</span></label></div>
              <div class="popover-row"><label><input type="checkbox" id="chk_ema21" onchange="toggleIndicatorSeries()"> <span style="color:#6366f1;">21 EMA (Dynamic Trail Anchor)</span></label></div>
              <div class="popover-row"><label><input type="checkbox" id="chk_sma50" onchange="toggleIndicatorSeries()"> <span style="color:#a855f7;">50 SMA (Institutional Baseline)</span></label></div>
              <div class="popover-row"><label><input type="checkbox" id="chk_sma200" onchange="toggleIndicatorSeries()"> <span style="color:#cbd5e1;">200 SMA (Long-term Trend)</span></label></div>
              <div class="popover-row"><label><input type="checkbox" id="chk_vol_ma20" checked onchange="toggleIndicatorSeries()"> <span style="color:#38bdf8;">20-Day Volume MA</span></label></div>
            </div>
            <div style="border-top:1px solid #1e293b; padding-top:8px;">
              <div class="popover-sec-title">🟡 LARSSON LINE RIBBON</div>
              <div class="popover-row">
                <label><input type="checkbox" id="chk_larsson" onchange="toggleIndicatorSeries()"> <span style="color:#facc15; font-weight:700;">Larsson Line Ribbon</span></label>
              </div>
              <div class="popover-row" style="background:#111827; padding:4px 6px; border-radius:4px; margin-top:4px;">
                <span>Fastest EMA:</span>
                <input type="number" id="cfg_larsson_fastest" value="32" min="3" max="80" style="width:42px; background:#1e293b; color:#38bdf8; border:1px solid #334155; border-radius:3px; padding:2px; font-size:11px; text-align:center; font-weight:700;" onchange="onLarssonFastestChange()">
                <span id="lbl_larsson_spans" style="color:#64748b; font-size:10px;">(32, 35, 50, 58)</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="chart-container" id="chart_div"></div>
    </div>

    <!-- Right Pane: Institutional Trade Dossier -->
    <div class="pane-dossier">
      <!-- Card 1: Setup Execution & Return -->
      <div class="d-card" id="card_exec">
        <div class="d-card-title"><span id="d_card1_title">🛡️ 1. Setup Status &amp; Execution</span><span id="d_r_badge" class="badge-sz sz-15">+0.0R</span></div>
        <div class="d-row"><span class="d-lbl">Trade Type:</span><span class="d-val hl" id="d_trade_type">Trade 1 Base EP Delayed Breakout</span></div>
        <div class="d-row"><span class="d-lbl">Trader Archetype:</span><span class="d-val hl" id="d_archetype">-</span></div>
        <div class="d-row"><span class="d-lbl" id="d_lbl_entry">Entry (D1 Breakout):</span><span class="d-val hl" id="d_entry">-</span></div>
        <div class="d-row"><span class="d-lbl" id="d_lbl_stop">Stop Loss / Invalidation:</span><span class="d-val" id="d_stop" style="color:var(--red);">-</span></div>
        <div class="d-row"><span class="d-lbl">Initial Risk %:</span><span class="d-val" id="d_risk_pct">-</span></div>
        <div class="d-row"><span class="d-lbl">Suggested Size:</span><span class="d-val hl" id="d_size_mult">-</span></div>
        <div id="d_size_desc_row" style="background:#0f172a; padding:6px 8px; border-radius:4px; border:1px solid #1e293b; font-size:10px; line-height:1.35; margin-bottom:5px;">
          <div style="font-weight:700; color:var(--accent); margin-bottom:2px;" id="d_size_title">1.00x Base Allocation</div>
          <div id="d_size_desc" style="color:#94a3b8;">Standard institutional EP allocation: 1.0 R risk budget.</div>
        </div>
        <div class="d-row"><span class="d-lbl" id="d_lbl_exit">Current / Exit Price:</span><span class="d-val" id="d_exit">-</span></div>
        <div class="d-row"><span class="d-lbl" id="d_lbl_exit_reason">Unrealized / Exit Result:</span><span class="d-val" id="d_exit_reason" style="color:var(--amber);">-</span></div>
        <div class="d-row"><span class="d-lbl" id="d_lbl_hold">Holding Period:</span><span class="d-val" id="d_hold">-</span></div>
        <div class="d-row"><span class="d-lbl" id="d_lbl_peak_r">Peak Realized R:</span><span class="d-val pos" id="d_peak_r">-</span></div>
      </div>

      <!-- Card 2: Position Management & ML Exhaustion -->
      <div class="d-card" id="card_pos_mgmt">
        <div class="d-card-title"><span>🤖 2. Position Management &amp; ML Exhaustion</span></div>
        <div class="d-row"><span class="d-lbl">Recommended Trailing MA:</span><span class="d-val hl" id="d_rec_ma">-</span></div>
        <div class="d-row"><span class="d-lbl">Current MA Price &amp; Buffer:</span><span class="d-val" id="d_ma_level">-</span></div>
        <div class="d-row"><span class="d-lbl">ML Exhaustion (Current High):</span><span class="d-val hl" id="d_exh_score">-</span></div>
        <div class="d-row"><span class="d-lbl">Sell-Half Trigger Rule:</span><span class="d-val" id="d_sell_half_rule" style="color:var(--purple); font-size:10.5px; text-align:right; max-width:210px;">-</span></div>
        <div class="d-row"><span class="d-lbl">Harvest Action / Signal:</span><span class="d-val" id="d_harvest_action">-</span></div>
        <div class="d-row"><span class="d-lbl">Trailing Exit Rule:</span><span class="d-val" id="d_exit_rule" style="font-size:10.5px; color:var(--text-muted); text-align:right; max-width:210px;">-</span></div>
        <div class="d-row"><span class="d-lbl">Strategy Rationale:</span><span class="d-val" id="d_plan_desc" style="font-size:10px; color:#cbd5e1; text-align:right; max-width:210px;">-</span></div>
      </div>

      <!-- Card 3: Fundamental Catalyst & SEC EDGAR Filings -->
      <div class="d-card" id="card_catalyst">
        <div class="d-card-title">
          <span style="color:#c084fc;">📰 3. Fundamental Catalyst &amp; SEC EDGAR</span>
          <span id="cat_tier_badge" class="badge-status status-active" style="background:#2e1065; color:#c4b5fd; border:1px solid #7c3aed; font-size:9.5px;">Tier-1 Catalyst</span>
        </div>
        <div style="display:flex; flex-direction:column; gap:6px;">
          <div style="background:#131722; border-left:3px solid #8b5cf6; padding:6px 8px; border-radius:4px;">
            <div style="font-size:9.5px; color:#94a3b8; text-transform:uppercase; font-weight:700;" id="cat_category">Catalyst Category</div>
            <div style="font-size:11.5px; font-weight:700; color:#fff; margin-top:2px;" id="cat_headline">Headline Loading...</div>
          </div>
          <div style="font-size:10.5px; color:#cbd5e1; line-height:1.4; background:#0b0e14; padding:6px 8px; border-radius:4px; border:1px solid #1e283d;">
            <div style="font-size:9.5px; color:#38bdf8; font-weight:700; margin-bottom:2px; text-transform:uppercase;">🏛️ Institutional Repricing Trajectory:</div>
            <div id="cat_analysis">Institutional volume surge triggering structural repricing.</div>
          </div>
          <div>
            <div style="font-size:9.5px; color:var(--text-muted); text-transform:uppercase; font-weight:700; margin-bottom:4px; display:flex; justify-content:space-between;">
              <span>Official SEC EDGAR Filings (T-3 to T+1):</span>
              <span id="filing_count_badge" style="color:#38bdf8;">0 Filings</span>
            </div>
            <div id="filings_list" style="display:flex; flex-direction:column; gap:3px;">
              <div style="color:var(--text-muted); font-style:italic; font-size:10px;">No official SEC filings in window.</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Card 4: Relative Strength (RS) & Technical Posture -->
      <div class="d-card" id="card_rs_posture">
        <div class="d-card-title"><span>🏆 4. Relative Strength (RS) &amp; Posture</span></div>
        <div class="d-row"><span class="d-lbl">Breakout RS Rating:</span><span class="d-val hl" id="d_rs">-</span></div>
        <div class="d-row"><span class="d-lbl">Stage 2 Posture:</span><span class="d-val pos" id="d_stage2">-</span></div>
        <div class="d-row"><span class="d-lbl">Distance to 52W High:</span><span class="d-val" id="d_dist_52w">-</span></div>
        <div class="d-row"><span class="d-lbl">20D Base Tightness:</span><span class="d-val" id="d_base_tight">-</span></div>
        <div class="d-row"><span class="d-lbl">Pre-EP 20D Runup:</span><span class="d-val" id="d_pre_runup">-</span></div>
      </div>

      <!-- Card 5: Sector & Theme Momentum Backdrop -->
      <div class="d-card" id="card_sec_theme">
        <div class="d-card-title"><span>🌊 5. Sector &amp; Theme Momentum Backdrop</span></div>
        <div class="d-row"><span class="d-lbl">Theme Momentum:</span><span class="d-val" id="d_thm_mom">-</span></div>
        <div class="d-row"><span class="d-lbl">Sector Momentum:</span><span class="d-val" id="d_sec_mom">-</span></div>
        <div class="d-row"><span class="d-lbl">Thematic Archetype:</span><span class="d-val hl" id="d_thm_arch">-</span></div>
      </div>

      <!-- Card 6: Day 1 EP Catalyst Quality -->
      <div class="d-card" id="card_ep_metrics">
        <div class="d-card-title"><span>⚡ 6. Day 1 EP Catalyst Metrics</span></div>
        <div class="d-row"><span class="d-lbl">Day 1 Gap %:</span><span class="d-val pos" id="d_gap">-</span></div>
        <div class="d-row"><span class="d-lbl">Day 1 RVOL:</span><span class="d-val hl" id="d_rvol">-</span></div>
        <div class="d-row"><span class="d-lbl">Close Position in Range:</span><span class="d-val" id="d_close_pos">-</span></div>
        <div class="d-row"><span class="d-lbl">Upper Selling Wick %:</span><span class="d-val" id="d_upper_wick">-</span></div>
        <div class="d-row"><span class="d-lbl">V2 Rolling Conviction P(+50%):</span><span class="d-val hl" id="d_v2_prob">-</span></div>
      </div>

      <!-- Card 7: Market Health at Entry (SPY & QQQ) -->
      <div class="d-card" id="card_mkt_health">
        <div class="d-card-title"><span>🌐 7. Market Health at Entry (SPY &amp; QQQ)</span></div>
        <div class="d-row"><span class="d-lbl">SPY Above 50 SMA:</span><span class="d-val pos" id="d_spy_50">-</span></div>
        <div class="d-row"><span class="d-lbl">SPY 20D Momentum:</span><span class="d-val" id="d_spy_mom">-</span></div>
        <div class="d-row"><span class="d-lbl">QQQ Above 50 SMA:</span><span class="d-val pos" id="d_qqq_50">-</span></div>
        <div class="d-row"><span class="d-lbl">QQQ 20D Momentum:</span><span class="d-val" id="d_qqq_mom">-</span></div>
        <div class="d-row"><span class="d-lbl">QQQ 10/20 EMA Regime:</span><span class="d-val" id="d_qqq_regime">-</span></div>
        <div class="d-row"><span class="d-lbl">Tech Leadership (QQQ/SPY):</span><span class="d-val hl" id="d_tech_lead">-</span></div>
        <div class="d-row"><span class="d-lbl" id="d_lbl_days_breakout">Days to Breakout:</span><span class="d-val" id="d_days_breakout">-</span></div>
      </div>
    </div>
  </div>

  <script>
    let mainMode = 'closed'; // 'closed' or 'watchlist'
    let activeTier = 'all';
    let currentTrades = [];
    let currentWatchlist = [];
    let selectedTrade = null;
    let currentSortBy = 'r_multiple';
    let currentSortDir = 'desc';
    let chart, candleSeries, volumeSeries, maSeries, triggerLine, stopLine, climaxLine;
    let ema8Series, ema10Series, ema20Series, ema21Series, sma50Series, sma200Series, volMa20Series;
    let larssonFastSeries, larssonSlowSeries;
    let currentChartData = null;
    let currentPollIntervalSec = 300;
    let pollTimerId = null;
    let isFullscreen = false;

    window.onload = function() {
      initChart();
      initIndicatorSettings();
      loadMonitorSettings();
      loadData();
      preloadWatchlistCount();
      resetWatchlistPolling();

      window.addEventListener('keydown', (e) => {
        if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA')) {
          if (e.key === 'Escape') {
            e.target.blur();
            if (isFullscreen) toggleChartFullscreen();
          }
          return;
        }
        if (e.key === 'Escape') {
          if (isFullscreen) toggleChartFullscreen();
          const cs = document.getElementById('chart_settings_panel');
          if (cs) cs.style.display = 'none';
          const ms = document.getElementById('monitor_settings_panel');
          if (ms) ms.style.display = 'none';
        } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
          e.preventDefault();
          selectPreviousTrade();
        } else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
          e.preventDefault();
          selectNextTrade();
        }
      });
    };

    function initChart() {
      const el = document.getElementById('chart_div');
      chart = LightweightCharts.createChart(el, {
        width: el.clientWidth || 800,
        height: el.clientHeight || 500,
        layout: { background: { color: '#090c13' }, textColor: '#94a3b8' },
        grid: { vertLines: { color: '#141a29' }, horzLines: { color: '#141a29' } },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        rightPriceScale: {
          borderColor: '#1e293b',
          autoScale: true,
          scaleMargins: { top: 0.08, bottom: 0.22 }
        },
        timeScale: { borderColor: '#1e293b' }
      });

      candleSeries = chart.addCandlestickSeries({
        upColor: '#22c55e', downColor: '#ef4444',
        borderUpColor: '#22c55e', borderDownColor: '#ef4444',
        wickUpColor: '#22c55e', wickDownColor: '#ef4444'
      });

      volumeSeries = chart.addHistogramSeries({
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume_scale',
      });

      // Confine volume strictly to bottom 18% so it never goes high into candles
      chart.priceScale('volume_scale').applyOptions({
        scaleMargins: { top: 0.82, bottom: 0.0 }
      });

      // Indicator Lines
      volMa20Series = chart.addLineSeries({
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume_scale',
        color: '#38bdf8',
        lineWidth: 1.5,
        title: 'Vol 20 MA',
        priceLineVisible: false
      });

      ema8Series = chart.addLineSeries({ color: '#ec4899', lineWidth: 1, title: '8 EMA', priceLineVisible: false });
      ema10Series = chart.addLineSeries({ color: '#f59e0b', lineWidth: 1, title: '10 EMA', priceLineVisible: false });
      ema20Series = chart.addLineSeries({ color: '#38bdf8', lineWidth: 1, title: '20 EMA', priceLineVisible: false });
      ema21Series = chart.addLineSeries({ color: '#6366f1', lineWidth: 1, title: '21 EMA', priceLineVisible: false });
      sma50Series = chart.addLineSeries({ color: '#a855f7', lineWidth: 1, title: '50 SMA', priceLineVisible: false });
      sma200Series = chart.addLineSeries({ color: '#cbd5e1', lineWidth: 1, title: '200 SMA', priceLineVisible: false });

      // Larsson Ribbon Lines
      larssonFastSeries = chart.addLineSeries({ color: '#facc15', lineWidth: 1.5, title: 'Larsson Fast', priceLineVisible: false });
      larssonSlowSeries = chart.addLineSeries({ color: '#0284c7', lineWidth: 1.5, title: 'Larsson Slow', priceLineVisible: false });

      // Active Strategy Trailing MA
      maSeries = chart.addLineSeries({ color: '#38bdf8', lineWidth: 2 });

      // Robust ResizeObserver ensures canvas always matches container without overflowing
      const ro = new ResizeObserver(entries => {
        if (!entries || entries.length === 0) return;
        const rect = entries[0].contentRect;
        if (rect.width > 0 && rect.height > 0) {
          chart.applyOptions({ width: Math.floor(rect.width), height: Math.floor(rect.height) });
        }
      });
      ro.observe(el);
    }

    function initIndicatorSettings() {
      const saved = JSON.parse(localStorage.getItem('ep_chart_indicators') || '{}');
      const list = ['chk_ema8', 'chk_ema10', 'chk_ema20', 'chk_ema21', 'chk_sma50', 'chk_sma200', 'chk_vol_ma20', 'chk_larsson'];
      list.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
          if (saved[id] !== undefined) {
            el.checked = saved[id];
          } else if (id === 'chk_vol_ma20') {
            el.checked = true;
          } else {
            el.checked = false;
          }
        }
      });
      const savedFastest = localStorage.getItem('ep_larsson_fastest');
      if (savedFastest && document.getElementById('cfg_larsson_fastest')) {
        document.getElementById('cfg_larsson_fastest').value = savedFastest;
      }
    }

    function toggleIndicatorSeries() {
      if (!currentChartData) return;
      const states = {};
      const map = [
        { id: 'chk_ema8', series: ema8Series, data: currentChartData.ema8 },
        { id: 'chk_ema10', series: ema10Series, data: currentChartData.ema10 },
        { id: 'chk_ema20', series: ema20Series, data: currentChartData.ema20 },
        { id: 'chk_ema21', series: ema21Series, data: currentChartData.ema21 },
        { id: 'chk_sma50', series: sma50Series, data: currentChartData.sma50 },
        { id: 'chk_sma200', series: sma200Series, data: currentChartData.sma200 },
        { id: 'chk_vol_ma20', series: volMa20Series, data: currentChartData.vol_ma20 },
        { id: 'chk_larsson', series: larssonFastSeries, data: currentChartData.larsson_fast, slowSeries: larssonSlowSeries, slowData: currentChartData.larsson_slow }
      ];

      map.forEach(item => {
        const el = document.getElementById(item.id);
        const checked = el ? el.checked : false;
        states[item.id] = checked;
        if (item.series) {
          item.series.setData(checked && item.data ? item.data : []);
        }
        if (item.slowSeries) {
          item.slowSeries.setData(checked && item.slowData ? item.slowData : []);
        }
      });

      localStorage.setItem('ep_chart_indicators', JSON.stringify(states));
    }

    function toggleChartFullscreen() {
      isFullscreen = !isFullscreen;
      const p = document.getElementById('pane_chart');
      const b = document.getElementById('btn_chart_fullscreen');
      const hint = document.getElementById('fs_keyboard_hint');
      if (p) p.classList.toggle('fullscreen', isFullscreen);
      if (b) {
        b.textContent = isFullscreen ? '🗗 Minimize' : '⛶ Fullscreen';
        b.title = isFullscreen ? 'Exit Fullscreen (Esc)' : 'Toggle Fullscreen Chart';
      }
      if (hint) {
        hint.style.display = isFullscreen ? 'inline' : 'none';
      }
      setTimeout(() => {
        if (chart) {
          const el = document.getElementById('chart_div');
          chart.applyOptions({ width: el.clientWidth, height: el.clientHeight });
        }
      }, 80);
    }

    function toggleChartSettings() {
      const p = document.getElementById('chart_settings_panel');
      if (!p) return;
      p.style.display = (p.style.display === 'none' || !p.style.display) ? 'flex' : 'none';
    }

    function toggleMonitorSettings() {
      const p = document.getElementById('monitor_settings_panel');
      if (!p) return;
      p.style.display = (p.style.display === 'none' || !p.style.display) ? 'flex' : 'none';
    }

    async function loadMonitorSettings() {
      try {
        const res = await fetch('/api/monitor_settings');
        if (res.ok) {
          const data = await res.json();
          currentPollIntervalSec = data.poll_interval_sec || 300;
          const sel = document.getElementById('sel_monitor_interval');
          if (sel) sel.value = String(currentPollIntervalSec);
          resetWatchlistPolling();
        }
      } catch (e) {}
    }

    async function saveMonitorInterval(val) {
      try {
        const sec = parseInt(val, 10);
        const res = await fetch('/api/monitor_settings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ poll_interval_sec: sec })
        });
        if (res.ok) {
          const data = await res.json();
          currentPollIntervalSec = data.poll_interval_sec;
          resetWatchlistPolling();
          alert(`Detector interval updated to ${Math.round(currentPollIntervalSec / 60)} minutes.`);
        }
      } catch (e) {
        alert('Failed to update monitor settings: ' + e.message);
      }
    }

    async function runMonitorCycleNow() {
      try {
        const res = await fetch('/api/run_monitor_cycle', { method: 'POST' });
        if (res.ok) {
          const data = await res.json();
          alert(`Check cycle complete: ${data.alerts_count} alert(s) found out of ${data.active_count} active setups.`);
          loadWatchlistData();
        }
      } catch (e) {
        alert('Failed to run cycle: ' + e.message);
      }
    }

    function resetWatchlistPolling() {
      if (pollTimerId) clearInterval(pollTimerId);
      pollTimerId = setInterval(() => {
        if (mainMode === 'watchlist') {
          loadWatchlistData();
        }
      }, currentPollIntervalSec * 1000);
    }

    function onChartHistoryChange() {
      if (selectedTrade) {
        selectTrade(selectedTrade);
      }
    }

    function onLarssonFastestChange() {
      const val = document.getElementById('cfg_larsson_fastest').value;
      localStorage.setItem('ep_larsson_fastest', val);
      if (selectedTrade) {
        selectTrade(selectedTrade);
      }
    }

    function setMainMode(mode) {
      mainMode = mode;
      document.getElementById('btn_mode_closed').className = (mode === 'closed') ? 'btn-toggle active' : 'btn-toggle';
      document.getElementById('btn_mode_watchlist').className = (mode === 'watchlist') ? 'btn-toggle active' : 'btn-toggle';

      if (mode === 'closed') {
        document.getElementById('filters_closed').style.display = 'flex';
        document.getElementById('filters_watchlist').style.display = 'none';
        currentSortBy = 'r_multiple';
        currentSortDir = 'desc';
        updateTableHeader();
        loadData();
      } else {
        document.getElementById('filters_closed').style.display = 'none';
        document.getElementById('filters_watchlist').style.display = 'flex';
        currentSortBy = 'unrealized_r';
        currentSortDir = 'desc';
        updateTableHeader();
        loadWatchlistData();
      }
    }

    function updateTableHeader() {
      const thead = document.getElementById('tbl_head');
      if (mainMode === 'closed') {
        thead.innerHTML = `
          <tr>
            <th class="tl" onclick="sortTable('symbol')" title="Sort by Symbol">TICKER</th>
            <th class="tl" onclick="sortTable('archetype')" title="Sort by Archetype">ARCHETYPE</th>
            <th onclick="sortTable('rs_score_breakout')" title="Sort by Breakout RS">RS</th>
            <th onclick="sortTable('v2_breakout')" title="Sort by Breakout V2 Conviction">V2</th>
            <th onclick="sortTable('v2_size_mult')" title="Sort by Position Size Multiplier">SIZE</th>
            <th onclick="sortTable('return_pct')" title="Sort by Return %">RET %</th>
            <th onclick="sortTable('r_multiple')" title="Sort by Realized R Multiple">RESULT (R)</th>
            <th onclick="sortTable('entry_date')" style="color:var(--accent);" title="Sort by Entry Date">DATE ⬍</th>
          </tr>
        `;
      } else {
        thead.innerHTML = `
          <tr>
            <th class="tl" onclick="sortWatchlistTable('symbol')" title="Sort by Symbol">TICKER</th>
            <th class="tl" onclick="sortWatchlistTable('status')" title="Sort by Status">STATUS</th>
            <th onclick="sortWatchlistTable('trigger_price')" title="Sort by Trigger Price">TRIGGER</th>
            <th onclick="sortWatchlistTable('current_price')" title="Sort by Current Price">CURRENT</th>
            <th onclick="sortWatchlistTable('stop_price')" title="Sort by Stop Price">STOP</th>
            <th onclick="sortWatchlistTable('unrealized_r')" title="Sort by Gain / Proximity">GAIN / DIST</th>
            <th onclick="sortWatchlistTable('rs_score')" title="Sort by RS Score">RS</th>
            <th onclick="sortWatchlistTable('ep_date')" style="color:var(--accent);" title="Sort by EP Date">EP DATE ⬍</th>
          </tr>
        `;
      }
    }

    function setTier(t) {
      activeTier = t;
      document.getElementById('btn_tier_all').className = (t === 'all') ? 'btn-toggle active' : 'btn-toggle';
      document.getElementById('btn_tier_1').className = (t === 'tier1') ? 'btn-toggle active-tier1' : 'btn-toggle';
      loadData();
    }

    function onFilterChange() {
      loadData();
    }

    function sortTable(col) {
      if (currentSortBy === col) {
        currentSortDir = (currentSortDir === 'desc') ? 'asc' : 'desc';
      } else {
        currentSortBy = col;
        currentSortDir = (col === 'entry_date' || col === 'symbol' || col === 'archetype') ? 'asc' : 'desc';
      }

      const sortLbl = document.getElementById('tbl_sort_lbl');
      if (sortLbl) {
        const dirArrow = currentSortDir === 'desc' ? '▼' : '▲';
        const cleanName = col.replace('_breakout', '').replace('_multiple', '').toUpperCase();
        sortLbl.textContent = `SORT: ${cleanName} ${dirArrow}`;
      }

      currentTrades.sort((a, b) => {
        let vA = a[col];
        let vB = b[col];
        if (typeof vA === 'string' || col === 'entry_date' || col === 'ep_date' || col === 'symbol' || col === 'archetype') {
          vA = String(vA || '');
          vB = String(vB || '');
          return currentSortDir === 'asc' ? vA.localeCompare(vB) : vB.localeCompare(vA);
        }
        vA = Number(vA) || 0;
        vB = Number(vB) || 0;
        return currentSortDir === 'asc' ? (vA - vB) : (vB - vA);
      });

      renderClosedTable();
      populateNavStockSelect();
      if (currentTrades.length > 0) {
        const matched = currentTrades.find(t => selectedTrade && t.symbol === selectedTrade.symbol && t.entry_date === selectedTrade.entry_date);
        selectTrade(matched || currentTrades[0]);
      }
    }

    function sortWatchlistTable(col) {
      if (currentSortBy === col) {
        currentSortDir = (currentSortDir === 'desc') ? 'asc' : 'desc';
      } else {
        currentSortBy = col;
        currentSortDir = (col === 'ep_date' || col === 'symbol' || col === 'status') ? 'asc' : 'desc';
      }

      const sortLbl = document.getElementById('tbl_sort_lbl');
      if (sortLbl) {
        const dirArrow = currentSortDir === 'desc' ? '▼' : '▲';
        sortLbl.textContent = `SORT: ${col.toUpperCase()} ${dirArrow}`;
      }

      currentWatchlist.sort((a, b) => {
        let vA = a[col];
        let vB = b[col];
        if (typeof vA === 'string' || col === 'ep_date' || col === 'symbol' || col === 'status' || col === 'archetype') {
          vA = String(vA || '');
          vB = String(vB || '');
          return currentSortDir === 'asc' ? vA.localeCompare(vB) : vB.localeCompare(vA);
        }
        vA = Number(vA) || 0;
        vB = Number(vB) || 0;
        return currentSortDir === 'asc' ? (vA - vB) : (vB - vA);
      });

      renderWatchlistTable();
      populateNavStockSelect();
      if (currentWatchlist.length > 0) {
        const matched = currentWatchlist.find(t => selectedTrade && t.symbol === selectedTrade.symbol && t.ep_date === selectedTrade.ep_date);
        selectTrade(matched || currentWatchlist[0]);
      }
    }

    async function preloadWatchlistCount() {
      try {
        const res = await fetch('/api/watchlist');
        if (res.ok) {
          const data = await res.json();
          document.getElementById('btn_wl_count').textContent = data.count || 0;
        }
      } catch (e) {}
    }

    async function loadData() {
      const exitMode = document.getElementById('sel_exit_mode').value;
      const sizing = document.getElementById('sel_sizing').value;
      const arch = document.getElementById('sel_archetype').value;
      const minV2 = document.getElementById('sel_min_v2').value;
      const minRS = document.getElementById('sel_min_rs') ? document.getElementById('sel_min_rs').value : 0;
      const prog = document.getElementById('sel_prog') ? document.getElementById('sel_prog').value : 'none';
      const stopMode = document.getElementById('sel_stop_mode') ? document.getElementById('sel_stop_mode').value : 'structural';
      const climax = document.getElementById('sel_climax') ? document.getElementById('sel_climax').value : 'none';

      const prevSelectedSym = selectedTrade ? selectedTrade.symbol : null;
      const prevSelectedEntry = selectedTrade ? selectedTrade.entry_date : null;

      const q = `tier=${activeTier}&exit_mode=${exitMode}&sizing=${sizing}&archetype=${arch}&min_v2=${minV2}&min_rs=${minRS}&progressive=${prog}&stop_mode=${stopMode}&climax=${climax}&sort_by=${currentSortBy}&sort_dir=${currentSortDir}`;

      try {
        // 1. Fetch KPIs
        const kpiRes = await fetch(`/api/kpis?${q}`);
        if (!kpiRes.ok) throw new Error(`HTTP ${kpiRes.status} from /api/kpis`);
        const kpis = await kpiRes.json();
        
        document.getElementById('kpi_lbl_1').textContent = 'QUALIFIED TRADES';
        document.getElementById('kpi_val_1').textContent = kpis.trades_count.toLocaleString();
        document.getElementById('kpi_lbl_2').textContent = 'STRATEGY WIN RATE';
        document.getElementById('kpi_val_2').textContent = kpis.win_rate_pct + '%';
        document.getElementById('kpi_lbl_3').textContent = 'EXPECTANCY (EV)';
        document.getElementById('kpi_val_3').textContent = (kpis.ev_r >= 0 ? '+' : '') + kpis.ev_r.toFixed(3) + ' R';
        document.getElementById('kpi_lbl_4').textContent = 'TOTAL REALIZED P&L';
        document.getElementById('kpi_val_4').textContent = (kpis.total_r >= 0 ? '+' : '') + kpis.total_r.toFixed(1) + ' R';
        document.getElementById('kpi_lbl_5').textContent = 'PROFIT FACTOR';
        document.getElementById('kpi_val_5').textContent = kpis.profit_factor.toFixed(2);
        document.getElementById('kpi_lbl_6').textContent = 'PAYOFF RATIO';
        document.getElementById('kpi_val_6').textContent = kpis.payoff_ratio.toFixed(2);
        document.getElementById('kpi_lbl_7').textContent = 'MAX DRAWDOWN';
        document.getElementById('kpi_val_7').textContent = kpis.max_drawdown_r.toFixed(1) + ' R';

        // 2. Fetch Trades
        const tRes = await fetch(`/api/trades?${q}`);
        if (!tRes.ok) throw new Error(`HTTP ${tRes.status} from /api/trades`);
        const data = await tRes.json();
        currentTrades = data.trades || [];
        document.getElementById('tbl_count_lbl').textContent = `QUALIFIED TRADES (${currentTrades.length})`;
        renderClosedTable();
        populateNavStockSelect();

        if (currentTrades.length > 0) {
          const matched = currentTrades.find(t => t.symbol === prevSelectedSym && t.entry_date === prevSelectedEntry);
          selectTrade(matched || currentTrades[0]);
        } else {
          document.getElementById('trade_rows').innerHTML = '<tr><td colspan="8" style="text-align:center; padding:20px; color:var(--text-muted);">No trades match the current filter criteria.</td></tr>';
        }
      } catch (err) {
        console.error("Error loading optimized EP data:", err);
        document.getElementById('tbl_count_lbl').textContent = "LOAD ERROR";
        document.getElementById('trade_rows').innerHTML = `<tr><td colspan="8" style="text-align:center; padding:20px; color:var(--red);">Failed to load trades: ${err.message}</td></tr>`;
      }
    }

    async function loadWatchlistData() {
      const statusFilter = document.getElementById('sel_wl_status').value;
      const minRS = document.getElementById('sel_wl_min_rs').value;
      const stage2Filter = document.getElementById('sel_wl_stage2').value;

      const q = `status=${statusFilter}&min_rs=${minRS}&sort_by=${currentSortBy}&sort_dir=${currentSortDir}`;

      try {
        const res = await fetch(`/api/watchlist?${q}`);
        if (!res.ok) throw new Error(`HTTP ${res.status} from /api/watchlist`);
        const data = await res.json();
        let items = data.items || [];
        if (stage2Filter === 'stage2') {
          items = items.filter(x => x.stage2);
        }
        currentWatchlist = items;

        // Update Intraday Monitor telemetry pill
        const pill = document.getElementById('monitor_pill');
        const dot = document.getElementById('monitor_dot');
        const mText = document.getElementById('monitor_text');
        const iMeta = data.intraday_meta || {};
        if (pill && dot && mText) {
          if (iMeta.market_open) {
            if (iMeta.alerts_count > 0) {
              pill.className = 'monitor-indicator alerting';
              dot.style.background = '#c084fc';
              mText.textContent = `🚨 ${iMeta.alerts_count} INTRADAY SELL HALF ALERT(S)`;
            } else {
              pill.className = 'monitor-indicator live';
              dot.style.background = '#22c55e';
              mText.textContent = `⚡ Intraday Monitor: Live (${iMeta.timestamp_et ? iMeta.timestamp_et.split(' ')[1] + ' ET' : 'Open'})`;
            }
          } else {
            pill.className = 'monitor-indicator';
            dot.style.background = '#94a3b8';
            mText.textContent = `⚪ Intraday Monitor: Market Closed`;
          }
        }

        const activeCount = items.filter(x => x.status === 'Active Long').length;
        const coilingCount = items.filter(x => x.status === 'Coiling Setup').length;
        const topGain = items.length > 0 ? Math.max(...items.map(x => x.unrealized_r)) : 0.0;
        const avgGain = activeCount > 0 ? (items.filter(x => x.status === 'Active Long').reduce((a, b) => a + b.unrealized_pct, 0) / activeCount) : 0.0;
        const avgRS = items.length > 0 ? (items.reduce((a, b) => a + b.rs_score, 0) / items.length) : 0.0;
        const stage2Pct = items.length > 0 ? (items.filter(x => x.stage2).length / items.length * 100) : 0.0;

        document.getElementById('kpi_lbl_1').textContent = 'TOTAL CANDIDATES';
        document.getElementById('kpi_val_1').textContent = items.length;
        document.getElementById('kpi_lbl_2').textContent = 'ACTIVE LONGS';
        document.getElementById('kpi_val_2').textContent = activeCount;
        document.getElementById('kpi_lbl_3').textContent = 'COILING SETUPS';
        document.getElementById('kpi_val_3').textContent = coilingCount;
        document.getElementById('kpi_lbl_4').textContent = 'TOP UNREALIZED GAIN';
        document.getElementById('kpi_val_4').textContent = `+${topGain.toFixed(2)} R`;
        document.getElementById('kpi_lbl_5').textContent = 'AVG ACTIVE GAIN';
        document.getElementById('kpi_val_5').textContent = `+${avgGain.toFixed(1)}%`;
        document.getElementById('kpi_lbl_6').textContent = 'AVG RS RATING';
        document.getElementById('kpi_val_6').textContent = avgRS.toFixed(1);
        document.getElementById('kpi_lbl_7').textContent = 'STAGE 2 RATIO';
        document.getElementById('kpi_val_7').textContent = `${stage2Pct.toFixed(0)}%`;

        document.getElementById('tbl_count_lbl').textContent = `WATCHLIST SETUPS (${items.length})`;
        renderWatchlistTable();
        populateNavStockSelect();

        if (currentWatchlist.length > 0) {
          selectTrade(currentWatchlist[0]);
        } else {
          document.getElementById('trade_rows').innerHTML = '<tr><td colspan="8" style="text-align:center; padding:20px; color:var(--text-muted);">No watchlist candidates match the selected filters.</td></tr>';
        }
      } catch (err) {
        console.error("Error loading watchlist data:", err);
        document.getElementById('trade_rows').innerHTML = `<tr><td colspan="8" style="text-align:center; padding:20px; color:var(--red);">Failed to load watchlist: ${err.message}</td></tr>`;
      }
    }

    function renderClosedTable() {
      const tbody = document.getElementById('trade_rows');
      tbody.innerHTML = '';
      currentTrades.forEach((tr, i) => {
        const row = document.createElement('tr');
        if (selectedTrade && selectedTrade.symbol === tr.symbol && selectedTrade.entry_date === tr.entry_date) {
          row.className = 'selected';
        }

        let szClass = 'sz-10';
        if (tr.v2_size_mult >= 1.5) szClass = 'sz-15';
        else if (tr.v2_size_mult >= 1.25) szClass = 'sz-125';
        else if (tr.v2_size_mult < 1.0) szClass = 'sz-075';

        const rColor = tr.r_multiple >= 0 ? 'var(--green)' : 'var(--red)';
        const retColor = tr.return_pct >= 0 ? 'var(--green)' : 'var(--red)';
        const rsVal = tr.rs_score_breakout || tr.rs_score || 50;
        const v2Pct = Math.round((tr.v2_breakout || tr.ml_50 || 0) * 100);
        const rsColor = rsVal >= 90 ? '#38bdf8' : (rsVal >= 80 ? '#22c55e' : '#94a3b8');

        row.innerHTML = `
          <td class="tl"><strong style="color:#fff;">${tr.symbol}</strong></td>
          <td class="tl"><span class="badge-arch">${tr.archetype}</span></td>
          <td><span style="color:${rsColor}; font-weight:600;">${rsVal}</span></td>
          <td><span style="color:#c084fc; font-weight:600;">${v2Pct}%</span></td>
          <td><span class="badge-sz ${szClass}">${tr.v2_size_mult}x</span></td>
          <td style="color:${retColor}">${tr.return_pct > 0 ? '+' : ''}${tr.return_pct}%</td>
          <td style="color:${rColor}; font-weight:700;">${tr.r_multiple > 0 ? '+' : ''}${tr.r_multiple.toFixed(2)} R</td>
          <td style="color:var(--accent); font-size:10px; font-weight:600;">${tr.entry_date}</td>
        `;
        row.onclick = () => selectTrade(tr);
        tbody.appendChild(row);
      });
    }

    function renderWatchlistTable() {
      const tbody = document.getElementById('trade_rows');
      tbody.innerHTML = '';
      currentWatchlist.forEach((it, i) => {
        const row = document.createElement('tr');
        const isAlert = it.is_alert || (it.alert_state && it.alert_state.includes('🚨')) || (it.current_high_exhaustion_pct >= (it.climax_threshold_pct || 50.0));
        const isElevated = (!isAlert && (it.alert_state === '⚠️ ELEVATED_WATCH' || it.current_high_exhaustion_pct >= ((it.climax_threshold_pct || 50.0) - 5.0)));

        if (selectedTrade && selectedTrade.symbol === it.symbol && selectedTrade.ep_date === it.ep_date) {
          row.className = isAlert ? 'selected row-alert-pulse' : 'selected';
        } else if (isAlert) {
          row.className = 'row-alert-pulse';
        }

        const isActive = (it.status === 'Active Long');
        let statusBadge = isActive ? '<span class="badge-status status-active">Active Long</span>' : '<span class="badge-status status-coiling">Coiling Setup</span>';
        if (isAlert) {
          statusBadge += `<span class="badge-alert-pulse" title="${it.alert_action || 'ML Climax Exhaustion Detected'}">🚨 SELL HALF</span>`;
        } else if (isElevated) {
          statusBadge += `<span class="badge-elevated" title="${it.alert_action || 'Elevated Exhaustion'}">⚠️ WATCH</span>`;
        }
        
        let gainCol = '';
        if (isActive) {
          const rColor = it.unrealized_r >= 0 ? 'var(--green)' : 'var(--red)';
          const exhPct = it.current_high_exhaustion_pct !== undefined ? it.current_high_exhaustion_pct : 0.0;
          gainCol = `<span style="color:${rColor}; font-weight:700;">+${it.unrealized_r.toFixed(2)}R (+${it.unrealized_pct}%)</span><div style="font-size:9.5px; color:${isAlert ? '#c084fc' : (isElevated ? '#fbbf24' : 'var(--text-muted)')}; font-weight:600;">Exh: ${exhPct}%</div>`;
        } else {
          gainCol = `<span style="color:var(--amber); font-weight:600;">${it.dist_to_trigger.toFixed(1)}% to Trigger</span>`;
        }

        const rsColor = it.rs_score >= 90 ? '#38bdf8' : (it.rs_score >= 80 ? '#22c55e' : '#94a3b8');
        const archBadge = `<span class="badge-arch" style="font-size:9.5px; font-weight:600;">${it.archetype ? it.archetype.replace(/^[^\\s]+\\s*/, '') : 'Core'}</span>`;

        row.innerHTML = `
          <td class="tl"><strong style="color:#fff;">${it.symbol}</strong></td>
          <td class="tl">${archBadge}</td>
          <td class="tl">${statusBadge}</td>
          <td>$${it.trigger_price.toFixed(2)}</td>
          <td style="color:#fff; font-weight:600;">$${it.current_price.toFixed(2)}</td>
          <td>${gainCol}</td>
          <td><span style="color:${rsColor}; font-weight:600;">${it.rs_score}</span></td>
          <td style="color:var(--accent); font-size:10px; font-weight:600;">${it.ep_date}</td>
        `;
        row.onclick = () => selectTrade(it);
        tbody.appendChild(row);
      });
    }

    function getActiveList() {
      return (mainMode === 'watchlist') ? currentWatchlist : currentTrades;
    }

    function populateNavStockSelect() {
      const list = getActiveList();
      const sel = document.getElementById('sel_nav_stock');
      if (!sel) return;
      sel.innerHTML = '';
      if (!list || list.length === 0) {
        const opt = document.createElement('option');
        opt.value = '-1';
        opt.textContent = 'No stocks available';
        sel.appendChild(opt);
        updateNavControls();
        return;
      }

      list.forEach((t, i) => {
        const opt = document.createElement('option');
        opt.value = i;
        if (mainMode === 'watchlist') {
          const isAct = (t.status === 'Active Long');
          const perf = isAct ? `+${t.unrealized_r.toFixed(2)}R` : `${t.dist_to_trigger.toFixed(1)}% dist`;
          const statusShort = isAct ? 'Active' : 'Coiling';
          opt.textContent = `${i + 1}. ${t.symbol} · ${statusShort} (${perf}) · RS:${t.rs_score}`;
        } else {
          const rSign = t.r_multiple >= 0 ? '+' : '';
          const archClean = t.archetype ? t.archetype.replace(/^[^\s]+\s*/, '') : 'EP';
          opt.textContent = `${i + 1}. ${t.symbol} · ${t.entry_date} (${rSign}${t.r_multiple.toFixed(2)}R) · ${archClean}`;
        }
        sel.appendChild(opt);
      });
      updateNavControls();
    }

    function updateNavControls() {
      const list = getActiveList();
      const sel = document.getElementById('sel_nav_stock');
      const lbl = document.getElementById('lbl_nav_counter');
      const btnPrev = document.getElementById('btn_nav_prev');
      const btnNext = document.getElementById('btn_nav_next');

      if (!list || list.length === 0 || !selectedTrade) {
        if (lbl) lbl.textContent = '0 / 0';
        if (btnPrev) btnPrev.disabled = true;
        if (btnNext) btnNext.disabled = true;
        return;
      }

      const curIdx = list.findIndex(t => t.symbol === selectedTrade.symbol && (t.entry_date === selectedTrade.entry_date || t.ep_date === selectedTrade.ep_date));
      const activeIdx = curIdx >= 0 ? curIdx : 0;

      if (lbl) lbl.textContent = `${activeIdx + 1} / ${list.length}`;
      if (btnPrev) btnPrev.disabled = (activeIdx <= 0);
      if (btnNext) btnNext.disabled = (activeIdx >= list.length - 1);
      if (sel && sel.value != activeIdx) {
        sel.value = activeIdx;
      }
    }

    function onNavStockSelect(idxVal) {
      const list = getActiveList();
      const idx = parseInt(idxVal, 10);
      if (list && idx >= 0 && idx < list.length) {
        selectTrade(list[idx]);
        scrollTableRowIntoView(idx);
      }
    }

    function selectPreviousTrade() {
      const list = getActiveList();
      if (!list || list.length === 0 || !selectedTrade) return;
      const curIdx = list.findIndex(t => t.symbol === selectedTrade.symbol && (t.entry_date === selectedTrade.entry_date || t.ep_date === selectedTrade.ep_date));
      if (curIdx > 0) {
        selectTrade(list[curIdx - 1]);
        scrollTableRowIntoView(curIdx - 1);
      }
    }

    function selectNextTrade() {
      const list = getActiveList();
      if (!list || list.length === 0 || !selectedTrade) return;
      const curIdx = list.findIndex(t => t.symbol === selectedTrade.symbol && (t.entry_date === selectedTrade.entry_date || t.ep_date === selectedTrade.ep_date));
      if (curIdx >= 0 && curIdx < list.length - 1) {
        selectTrade(list[curIdx + 1]);
        scrollTableRowIntoView(curIdx + 1);
      }
    }

    function scrollTableRowIntoView(idx) {
      const tbody = document.getElementById('trade_rows');
      if (tbody && tbody.children && tbody.children[idx]) {
        tbody.children[idx].scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      }
    }

    async function selectTrade(tr) {
      selectedTrade = tr;
      if (mainMode === 'closed') {
        renderClosedTable();
      } else {
        renderWatchlistTable();
      }

      const isWl = (mainMode === 'watchlist' || tr.status !== undefined);
      const isActiveLong = (tr.status === 'Active Long');
      const isCoiling = (tr.status === 'Coiling Setup');

      // Update Chart header
      document.getElementById('c_ticker').textContent = tr.symbol;
      const statusBadge = document.getElementById('c_badge_status');
      const archBadge = document.getElementById('c_badge_arch');
      const cMeta = document.getElementById('c_meta');

      if (isWl) {
        if (statusBadge) {
          statusBadge.textContent = tr.status || 'Active';
          statusBadge.className = isActiveLong ? 'badge-status status-active' : 'badge-status status-coiling';
          statusBadge.style.display = 'inline-block';
        }
        if (archBadge) {
          archBadge.textContent = tr.archetype ? tr.archetype.replace(/^[^\s]+\s*/, '') : 'Core';
          archBadge.style.display = 'inline-block';
        }
        const fullMeta = `EP: ${tr.ep_date} · RS: ${tr.rs_score} · ${tr.sector || tr.theme || 'Tech'}`;
        cMeta.textContent = fullMeta;
        cMeta.title = `${tr.symbol} · ${tr.status} · ${tr.archetype} · ${fullMeta}`;
      } else {
        if (statusBadge) {
          const rStr = (tr.r_multiple >= 0 ? '+' : '') + tr.r_multiple.toFixed(2) + ' R';
          statusBadge.textContent = rStr;
          statusBadge.className = tr.r_multiple >= 0 ? 'badge-status status-active' : 'badge-status status-coiling';
          statusBadge.style.display = 'inline-block';
        }
        if (archBadge) {
          archBadge.textContent = tr.archetype ? tr.archetype.replace(/^[^\s]+\s*/, '') : 'Core';
          archBadge.style.display = 'inline-block';
        }
        const fullMeta = `${tr.sector} · Entry: ${tr.entry_date} · Sizing: ${tr.v2_size_mult}x`;
        cMeta.textContent = fullMeta;
        cMeta.title = `${tr.symbol} · ${tr.archetype} · ${fullMeta}`;
      }

      updateNavControls();

      // Update Trade Classification & Setup Status
      document.getElementById('d_trade_type').textContent = tr.trade_type || 'Trade 1: Base EP Delayed Breakout (Days 2-5)';
      document.getElementById('d_archetype').textContent = tr.archetype || '👑 Pinnacle Elite';

      // Update Dossier Card 1
      const rBadge = document.getElementById('d_r_badge');
      const sugLabel = tr.suggested_size_label || (tr.v2_size_mult ? `${tr.v2_size_mult}x Allocation` : '1.00x Base Allocation');
      const sugDesc = tr.suggested_size_desc || (tr.position_management_plan || 'Standard institutional EP allocation: 1.0 R risk budget under institutional rules.');

      document.getElementById('d_size_mult').textContent = sugLabel;
      document.getElementById('d_size_title').textContent = sugLabel;
      document.getElementById('d_size_desc').textContent = sugDesc;

      if (isWl) {
        if (isActiveLong) {
          rBadge.textContent = `+${tr.unrealized_r.toFixed(2)} R`;
          rBadge.className = 'badge-sz sz-15';
          document.getElementById('d_card1_title').textContent = '🟢 1. Active Long Position';
          document.getElementById('d_lbl_entry').textContent = 'Entry (Breakout):';
          document.getElementById('d_entry').textContent = `$${tr.trigger_price.toFixed(2)} (${tr.trigger_date})`;
          document.getElementById('d_lbl_stop').textContent = 'Stop Loss:';
          document.getElementById('d_stop').textContent = `$${tr.stop_price.toFixed(2)} (-${tr.dynamic_stop_loss_pct}%)`;
          document.getElementById('d_risk_pct').textContent = `${tr.dynamic_stop_loss_pct}%`;
          document.getElementById('d_lbl_exit').textContent = 'Current Price:';
          document.getElementById('d_exit').textContent = `$${tr.current_price.toFixed(2)}`;
          document.getElementById('d_lbl_exit_reason').textContent = 'Unrealized Gain:';
          document.getElementById('d_exit_reason').textContent = `+${tr.unrealized_pct}% (+${tr.unrealized_r.toFixed(2)} R)`;
          document.getElementById('d_lbl_hold').textContent = 'Days in Trade:';
          document.getElementById('d_hold').textContent = `${tr.days_in_trade} trading days`;
          document.getElementById('d_lbl_peak_r').textContent = 'Unrealized R:';
          document.getElementById('d_peak_r').textContent = `+${tr.unrealized_r.toFixed(2)} R`;
        } else {
          rBadge.textContent = `${tr.dist_to_trigger.toFixed(1)}% away`;
          rBadge.className = 'badge-sz sz-125';
          document.getElementById('d_card1_title').textContent = '⏳ 1. Coiling Breakout Setup';
          document.getElementById('d_lbl_entry').textContent = 'Trigger (D1 High + 0.05):';
          document.getElementById('d_entry').textContent = `$${tr.trigger_price.toFixed(2)}`;
          document.getElementById('d_lbl_stop').textContent = 'Invalidation (D1 Low):';
          document.getElementById('d_stop').textContent = `$${tr.stop_price.toFixed(2)} (-${tr.dynamic_stop_loss_pct}%)`;
          document.getElementById('d_risk_pct').textContent = `${tr.dynamic_stop_loss_pct}%`;
          document.getElementById('d_lbl_exit').textContent = 'Current Price:';
          document.getElementById('d_exit').textContent = `$${tr.current_price.toFixed(2)}`;
          document.getElementById('d_lbl_exit_reason').textContent = 'Setup Status:';
          document.getElementById('d_exit_reason').textContent = `Coiling (${tr.dist_to_trigger.toFixed(1)}% to Trigger)`;
          document.getElementById('d_lbl_hold').textContent = 'Days Since EP:';
          document.getElementById('d_hold').textContent = `${tr.days_coiling} trading days`;
          document.getElementById('d_lbl_peak_r').textContent = 'Status:';
          document.getElementById('d_peak_r').textContent = 'Awaiting Trigger';
        }
      } else {
        rBadge.textContent = `${tr.r_multiple > 0 ? '+' : ''}${tr.r_multiple.toFixed(2)} R`;
        rBadge.className = `badge-sz ${tr.r_multiple >= 0 ? 'sz-15' : 'sz-075'}`;
        const isMlStop = (!isWl && document.getElementById('sel_stop_mode') && document.getElementById('sel_stop_mode').value === 'ml_adaptive');
        document.getElementById('d_card1_title').textContent = isMlStop ? '🤖 1. ML Adaptive Stop Trade Execution' : '🛡️ 1. Trade Execution & Return';
        document.getElementById('d_lbl_entry').textContent = 'Entry (D1 Breakout):';
        document.getElementById('d_entry').textContent = `$${tr.entry_price.toFixed(2)} (${tr.entry_date})`;
        document.getElementById('d_lbl_stop').textContent = isMlStop ? 'Stop Loss (🤖 20D MAE):' : 'Stop Loss (🛡️ D1 Low):';
        document.getElementById('d_stop').textContent = `$${tr.stop_price.toFixed(2)} (-${tr.dynamic_stop_loss_pct}%)`;
        document.getElementById('d_risk_pct').textContent = `${tr.dynamic_stop_loss_pct}%`;
        document.getElementById('d_lbl_exit').textContent = 'Exit Price & Date:';
        document.getElementById('d_exit').textContent = `$${tr.exit_price.toFixed(2)} (${tr.exit_date})`;
        document.getElementById('d_lbl_exit_reason').textContent = 'Exit Reason:';
        document.getElementById('d_exit_reason').textContent = tr.exit_reason;
        document.getElementById('d_lbl_hold').textContent = 'Holding Period:';
        document.getElementById('d_hold').textContent = `${tr.holding_days} trading days`;
        document.getElementById('d_lbl_peak_r').textContent = 'Peak Realized R:';
        document.getElementById('d_peak_r').textContent = `+${tr.peak_r.toFixed(2)} R`;
      }

      // Update Card 2: Position Management & ML Exhaustion Plan
      const recMaName = tr.recommended_ma_name || (tr.exit_ma || 'Adaptive MA');
      document.getElementById('d_rec_ma').textContent = recMaName;

      if (isWl) {
        if (tr.recommended_ma_price) {
          const bufSign = tr.recommended_ma_buffer_pct >= 0 ? '+' : '';
          document.getElementById('d_ma_level').textContent = `$${tr.recommended_ma_price.toFixed(2)} (${bufSign}${tr.recommended_ma_buffer_pct.toFixed(1)}% buffer)`;
          document.getElementById('d_ma_level').className = `d-val ${tr.recommended_ma_buffer_pct >= 0 ? 'pos' : 'neg'}`;
        } else {
          document.getElementById('d_ma_level').textContent = 'Tracking Entry';
        }

        const exhPct = tr.current_high_exhaustion_pct !== undefined ? tr.current_high_exhaustion_pct : 0.0;
        let exhClass = 'pos';
        let exhTag = 'Normal';
        if (exhPct >= (tr.climax_threshold_pct || 50.0)) {
          exhClass = 'neg';
          exhTag = '🚨 CLIMAX EXHAUSTION';
        } else if (exhPct >= 40.0) {
          exhClass = 'hl';
          exhTag = '⚠️ Elevated';
        }
        document.getElementById('d_exh_score').textContent = `${exhPct.toFixed(1)}% (${exhTag})`;
        document.getElementById('d_exh_score').className = `d-val ${exhClass}`;

        document.getElementById('d_sell_half_rule').textContent = tr.climax_sell_half_rule || `Sell 50% partial profit at ≥ 50.0% Exhaustion`;
        document.getElementById('d_harvest_action').textContent = tr.climax_action || `🟢 HOLD RUNNER`;
        document.getElementById('d_harvest_action').className = `d-val ${exhPct >= 50.0 ? 'neg' : 'pos'}`;
        document.getElementById('d_exit_rule').textContent = tr.trailing_exit_rule || `Close below ${recMaName} triggers exit`;
        document.getElementById('d_plan_desc').textContent = tr.position_management_plan || '-';
      } else {
        document.getElementById('d_ma_level').textContent = `Exited @ $${tr.exit_price.toFixed(2)}`;
        const hasClimax = tr.has_climax;
        document.getElementById('d_exh_score').textContent = hasClimax ? `🚨 Triggered (${tr.climax_date})` : `None (Pure Trailing MA)`;
        document.getElementById('d_sell_half_rule').textContent = `Sell 50% partial profit at ≥ 50.0% ML Exhaustion into extreme high`;
        document.getElementById('d_harvest_action').textContent = hasClimax ? `🚨 Harvested 50% @ $${tr.climax_price.toFixed(2)} (+${tr.climax_r.toFixed(2)}R)` : `Full Exit on Trailing MA`;
        document.getElementById('d_harvest_action').className = `d-val ${hasClimax ? 'hl' : 'pos'}`;
        document.getElementById('d_exit_rule').textContent = `Exited on ${tr.exit_reason}`;
        document.getElementById('d_plan_desc').textContent = `Matched institutional strategy: ${tr.archetype} trailed with ${recMaName}.`;
      }

      // Update Dossier Cards 4 & 5: RS & Sector/Theme
      document.getElementById('d_pre_runup').textContent = `${(tr.pre_runup_20d || 0) > 0 ? '+' : ''}${tr.pre_runup_20d || 0}% (Base)`;
      document.getElementById('d_base_tight').textContent = `${tr.pre_consol_tightness_20d || 0}% consolidation`;
      document.getElementById('d_dist_52w').textContent = `-${tr.pre_dist_52w_high || 0}% from 52W High`;
      document.getElementById('d_rs').textContent = `${tr.rs_score_breakout || tr.rs_score || 50} (RS Score)`;
      document.getElementById('d_stage2').textContent = (tr.stage2_breakout || tr.stage2) ? 'YES (Bullish Posture)' : 'NO';

      const thmName = tr.theme || 'Tech';
      const thmM1 = tr.thm_m1 !== undefined ? tr.thm_m1 : (tr.thm_m1_breakout || 0);
      const thmScore = tr.theme_score || tr.thm_score || '';
      document.getElementById('d_thm_mom').textContent = `${thmName} (${thmM1 > 0 ? '+' : ''}${thmM1}% 1M${thmScore ? ', Score: ' + thmScore : ''})`;

      const secName = tr.sector || 'Equities';
      const secM1 = tr.sec_m1 !== undefined ? tr.sec_m1 : (tr.sec_m1_breakout || 0);
      const secScore = tr.sector_score || tr.sec_score || '';
      document.getElementById('d_sec_mom').textContent = `${secName} (${secM1 > 0 ? '+' : ''}${secM1}% 1M${secScore ? ', Score: ' + secScore : ''})`;

      const archName = tr.thematic_archetype || tr.primary_archetype || 'Leading Trend';
      document.getElementById('d_thm_arch').textContent = archName;

      // Card 6: Day 1 EP Catalyst Quality
      document.getElementById('d_gap').textContent = `+${tr.gap_pct || 0}%`;
      document.getElementById('d_rvol').textContent = `${tr.rvol || 0}x volume`;
      document.getElementById('d_close_pos').textContent = `${tr.close_pos || 0.5}`;
      document.getElementById('d_upper_wick').textContent = `${tr.day1_upper_wick_pct || 0}%`;

      const winP = (tr.v2_win_prob !== undefined && tr.v2_win_prob !== null) ? tr.v2_win_prob : 0.50;
      const conv50 = (tr.v2_conviction_50 !== undefined && tr.v2_conviction_50 !== null) ? tr.v2_conviction_50 : (tr.v2_breakout || tr.ml_50 || 0);
      document.getElementById('d_v2_prob').textContent = `${Math.round(winP * 100)}% Win | P(+50%): ${(conv50 * 100).toFixed(1)}%`;

      // Card 7: Market Health at Entry (SPY & QQQ)
      const spy50 = (tr.spy_above_50_entry !== undefined) ? tr.spy_above_50_entry : true;
      document.getElementById('d_spy_50').textContent = spy50 ? 'YES (Tailwind)' : 'NO (Headwind)';
      document.getElementById('d_spy_50').className = `d-val ${spy50 ? 'pos' : 'neg'}`;

      const spyMom = tr.spy_ret_20d_entry !== undefined ? tr.spy_ret_20d_entry : 0;
      document.getElementById('d_spy_mom').textContent = `${spyMom > 0 ? '+' : ''}${spyMom}%`;
      document.getElementById('d_spy_mom').className = `d-val ${spyMom >= 0 ? 'pos' : 'neg'}`;

      const qqq50 = (tr.qqq_above_50_entry !== undefined) ? tr.qqq_above_50_entry : true;
      document.getElementById('d_qqq_50').textContent = qqq50 ? 'YES (Tailwind)' : 'NO (Headwind)';
      document.getElementById('d_qqq_50').className = `d-val ${qqq50 ? 'pos' : 'neg'}`;

      const qqqMom = tr.qqq_ret_20d_entry !== undefined ? tr.qqq_ret_20d_entry : 0;
      document.getElementById('d_qqq_mom').textContent = `${qqqMom > 0 ? '+' : ''}${qqqMom}%`;
      document.getElementById('d_qqq_mom').className = `d-val ${qqqMom >= 0 ? 'pos' : 'neg'}`;

      const qqqBull = tr.qqq_bull_stack;
      document.getElementById('d_qqq_regime').textContent = (qqqBull === true) ? 'Bull Stack (10 > 20 EMA)' : (qqqBull === false ? 'Pullback Divergence' : 'Neutral Alignment');

      const techLead = tr.qqq_spy_ratio_bull;
      document.getElementById('d_tech_lead').textContent = (techLead === true) ? '✓ Tech Leading (QQQ/SPY Bull)' : (techLead === false ? 'Broad Market / Defensive' : 'Neutral Ratio');

      if (isWl) {
        document.getElementById('d_lbl_days_breakout').textContent = isActiveLong ? 'Days in Trade:' : 'Days Coiling:';
        document.getElementById('d_days_breakout').textContent = isActiveLong ? `${tr.days_in_trade} days` : `${tr.days_coiling} days`;
      } else {
        document.getElementById('d_lbl_days_breakout').textContent = 'Days to Breakout:';
        document.getElementById('d_days_breakout').textContent = `${tr.days_to_breakout} days after EP`;
      }

      // Fetch Bars & Draw Chart
      const exitMode = document.getElementById('sel_exit_mode') ? document.getElementById('sel_exit_mode').value : 'adaptive';
      const historyMode = document.getElementById('sel_chart_history') ? document.getElementById('sel_chart_history').value : 'setup';
      const lFastest = document.getElementById('cfg_larsson_fastest') ? document.getElementById('cfg_larsson_fastest').value : '32';
      const entryDt = isWl ? (tr.trigger_date && tr.trigger_date !== 'Pending' ? tr.trigger_date : tr.ep_date) : tr.entry_date;
      const exitDt = isWl ? 'None' : tr.exit_date;

      const chartUrl = `/api/chart?symbol=${tr.symbol}&ep_date=${tr.ep_date}&entry_date=${entryDt}&exit_date=${exitDt}&exit_mode=${exitMode}&history=${historyMode}&larsson_fastest=${lFastest}`;
      const res = await fetch(chartUrl);
      const cData = await res.json();
      currentChartData = cData;

      candleSeries.setData(cData.candles);
      volumeSeries.setData(cData.volume);

      let maPoints = cData.ema20;
      let maName = '20 EMA';
      let maColor = '#38bdf8';

      const recKey = tr.recommended_ma_key || 'ema20';
      if (exitMode === 'sma50' || (exitMode === 'adaptive' && recKey === 'sma50')) {
        maPoints = cData.sma50;
        maName = '50 SMA (Institutional Baseline)';
        maColor = '#c084fc';
      } else if (exitMode === 'ema10' || (exitMode === 'adaptive' && recKey === 'ema10')) {
        maPoints = cData.ema10;
        maName = '10 EMA (Fast Velocity Trail)';
        maColor = '#f59e0b';
      } else {
        maPoints = cData.ema20;
        maName = '20 EMA (Intermediate Momentum)';
        maColor = '#38bdf8';
      }

      maSeries.setData(maPoints);
      maSeries.applyOptions({ color: maColor });
      document.getElementById('c_ma_legend').textContent = `── Trailing MA: ${maName}`;
      document.getElementById('c_ma_legend').style.color = maColor;

      // Update Spans & Indicator series
      if (cData.larsson_spans) {
        document.getElementById('lbl_larsson_spans').textContent = `(${cData.larsson_spans.join(', ')})`;
      }
      toggleIndicatorSeries();

      // Card 3: Fundamental Catalyst & SEC EDGAR Filings
      const cat = cData.catalyst || tr.catalyst;
      if (cat) {
        document.getElementById('cat_category').textContent = cat.category || 'Institutional Event';
        document.getElementById('cat_headline').textContent = cat.headline || `${tr.symbol} Repricing Event`;
        document.getElementById('cat_tier_badge').textContent = cat.tier || 'Tier-1 Catalyst';
        document.getElementById('cat_analysis').textContent = cat.analysis || `${tr.symbol} experienced an institutional volume surge.`;

        const filings = cat.sec_filings || [];
        document.getElementById('filing_count_badge').textContent = `${filings.length} Filing${filings.length === 1 ? '' : 's'}`;
        const filingsList = document.getElementById('filings_list');
        if (filings.length === 0) {
          filingsList.innerHTML = '<div style="color:var(--text-muted); font-style:italic; font-size:10px; padding:3px 0;">No official SEC filings within 3 days. Catalyst driven by press release or earnings.</div>';
        } else {
          filingsList.innerHTML = filings.map(f => {
            const is8K = f.form && (f.form.includes('8-K') || f.form.includes('6-K'));
            const badgeCol = is8K ? '#8b5cf6' : '#0284c7';
            const itemsText = f.items ? `· Items: <b>${f.items}</b>` : '';
            const linkHtml = f.url ? `<a href="${f.url}" target="_blank" rel="noopener noreferrer" class="filing-link" title="Open official SEC filing on EDGAR archive">View SEC EDGAR ↗</a>` : '';
            return `
              <div class="filing-item">
                <div style="min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; margin-right:8px;">
                  <span class="badge-status status-active" style="background:${badgeCol}; color:#fff; font-size:9.5px; padding:1px 5px; border-radius:3px;">Form ${f.form}</span>
                  <span style="color:#cbd5e1; margin-left:6px; font-weight:600;">${f.date}</span>
                  <span style="color:#94a3b8; margin-left:4px; font-size:10px;">${itemsText}</span>
                </div>
                ${linkHtml}
              </div>
            `;
          }).join('');
        }
      } else {
        document.getElementById('cat_category').textContent = 'EP Catalyst';
        document.getElementById('cat_headline').textContent = `${tr.symbol} Episodic Repricing`;
        document.getElementById('cat_tier_badge').textContent = 'Tier-1 Institutional Repricing';
        document.getElementById('cat_analysis').textContent = `${tr.symbol} volume surge triggering structural repricing.`;
        document.getElementById('filing_count_badge').textContent = '0 Filings';
        document.getElementById('filings_list').innerHTML = '<div style="color:var(--text-muted); font-style:italic; font-size:10px;">Catalyst data loading...</div>';
      }

      // Price lines
      if (triggerLine) candleSeries.removePriceLine(triggerLine);
      if (stopLine) candleSeries.removePriceLine(stopLine);
      if (climaxLine) {
        candleSeries.removePriceLine(climaxLine);
        climaxLine = null;
      }

      const triggerPx = isWl ? tr.trigger_price : tr.entry_price;
      const isMlStop = (!isWl && document.getElementById('sel_stop_mode') && document.getElementById('sel_stop_mode').value === 'ml_adaptive');
      const stopPx = tr.stop_price;
      const stopColor = isMlStop ? '#f59e0b' : '#ef4444';
      const stopTitle = isWl ? (isActiveLong ? `Stop Loss: $${stopPx.toFixed(2)}` : `Invalidation: $${stopPx.toFixed(2)}`) : (isMlStop ? `🤖 ML 20 MAE Stop: $${stopPx.toFixed(2)} (-${tr.dynamic_stop_loss_pct}%)` : `🛡️ Structural Stop: $${stopPx.toFixed(2)} (-${tr.dynamic_stop_loss_pct}%)`);

      triggerLine = candleSeries.createPriceLine({
        price: triggerPx, color: '#22c55e', lineWidth: 2, lineStyle: 2,
        title: isWl ? (isActiveLong ? `Breakout Entry: $${triggerPx.toFixed(2)}` : `Breakout Trigger: $${triggerPx.toFixed(2)}`) : `Breakout Entry: $${triggerPx.toFixed(2)} (${tr.v2_size_mult}x)`
      });

      stopLine = candleSeries.createPriceLine({
        price: stopPx, color: stopColor, lineWidth: 2, lineStyle: 2,
        title: stopTitle
      });

      const climaxLegend = document.getElementById('c_climax_legend');
      if (isWl && isActiveLong && (tr.current_high_exhaustion_pct >= (tr.climax_threshold_pct || 50.0) || tr.is_alert)) {
        const peakPrice = tr.intraday_high || tr.current_price;
        climaxLine = candleSeries.createPriceLine({
          price: peakPrice, color: '#c084fc', lineWidth: 2, lineStyle: 2,
          title: `🚨 INTRADAY SELL HALF: $${peakPrice.toFixed(2)} (${tr.current_high_exhaustion_pct}% Exhaustion)`
        });
        if (climaxLegend) {
          climaxLegend.style.display = 'inline';
          climaxLegend.textContent = `── 🚨 INTRADAY SELL HALF (${tr.current_high_exhaustion_pct}%)`;
          climaxLegend.style.color = '#c084fc';
        }
      } else if (!isWl && tr.has_climax && tr.climax_price > 0 && document.getElementById('sel_climax') && document.getElementById('sel_climax').value === 'partial_50') {
        climaxLine = candleSeries.createPriceLine({
          price: tr.climax_price, color: '#c084fc', lineWidth: 2, lineStyle: 2,
          title: `50% Climax Take: $${tr.climax_price.toFixed(2)} (+${tr.climax_r.toFixed(2)}R)`
        });
        if (climaxLegend) {
          climaxLegend.style.display = 'inline';
          climaxLegend.textContent = `── 50% Climax Take`;
          climaxLegend.style.color = '#c084fc';
        }
      } else {
        if (climaxLegend) climaxLegend.style.display = 'none';
      }

      // Markers
      let markers = [];
      if (isWl) {
        if (isActiveLong && tr.trigger_date && tr.trigger_date !== 'Pending') {
          markers.push({ time: tr.trigger_date, position: 'belowBar', color: '#22c55e', shape: 'arrowUp', text: `BUY @ $${tr.trigger_price.toFixed(2)} (+${tr.unrealized_pct}%)` });
        } else {
          markers.push({ time: tr.ep_date, position: 'belowBar', color: '#f59e0b', shape: 'arrowUp', text: `EP Catalyst (+${tr.gap_pct}%)` });
        }
      } else {
        const isStopLoss = (tr.r_multiple < 0 || (tr.exit_reason && tr.exit_reason.toLowerCase().includes('stop')));
        const exitLabel = isStopLoss ? `STOP HIT @ $${tr.exit_price.toFixed(2)} (${tr.r_multiple.toFixed(2)}R)` : `EXIT (${maName}) @ $${tr.exit_price.toFixed(2)} (+${tr.r_multiple.toFixed(2)}R)`;
        markers = [
          { time: tr.entry_date, position: 'belowBar', color: '#22c55e', shape: 'arrowUp', text: `BUY @ $${tr.entry_price.toFixed(2)} (${tr.v2_size_mult}x)` },
          { time: tr.exit_date, position: 'aboveBar', color: isStopLoss ? '#ef4444' : '#22c55e', shape: 'arrowDown', text: exitLabel }
        ];
        if (tr.has_climax && tr.climax_date && tr.climax_date !== 'None' && tr.climax_date !== tr.exit_date && document.getElementById('sel_climax') && document.getElementById('sel_climax').value === 'partial_50') {
          markers.push({
            time: tr.climax_date,
            position: 'aboveBar',
            color: '#c084fc',
            shape: 'arrowDown',
            text: `CLIMAX HARVEST (50%): $${tr.climax_price.toFixed(2)} (+${tr.climax_r.toFixed(2)}R)`
          });
        }
      }
      candleSeries.setMarkers(markers);

      chart.priceScale('right').applyOptions({
        autoScale: true,
        scaleMargins: { top: 0.08, bottom: 0.22 }
      });
      chart.priceScale('volume_scale').applyOptions({
        scaleMargins: { top: 0.82, bottom: 0.0 }
      });
      chart.timeScale().fitContent();
    }

    async function reloadDataset() {
      const res = await fetch('/api/reload_dataset', { method: 'POST' });
      const data = await res.json();
      alert(`Dataset reloaded: ${data.trades_count} closed trades (${data.tier1_count} Tier 1), ${data.watchlist_count} watchlist candidates.`);
      preloadWatchlistCount();
      if (mainMode === 'closed') {
        loadData();
      } else {
        loadWatchlistData();
      }
    }
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    print(f"🚀 Launching Institutional Optimized EP Explorer on port {PORT}...")
    print(f"URL: http://127.0.0.1:{PORT}")
    app.run(host="127.0.0.1", port=PORT, debug=False)

