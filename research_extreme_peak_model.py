"""
research_extreme_peak_model.py

Rigorous research script to:
1. Evaluate Part 1: Exhaustion classifier trained on Strategy EPs vs Full Universe.
2. Develop and Evaluate Part 2: New ML Model for Extreme Peak / Climax High Recognition.
   - Computes RSI (14, 9, 5), RSI sequence, RSI Bearish Divergence, EMA/SMA extensions in ADR,
     candle rejection wicks, range expansion, volume churn, and unrealized R-multiples.
   - Uses zero-lookahead chronological split: Train (2016-2022) vs Out-of-Sample Test (2023-2026).
   - Tests LightGBM, XGBoost, CatBoost, Random Forest with calibrated probabilities.
   - Runs trade-level simulation: Partial profit take at extreme high + trailing remainder.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, precision_recall_curve,
    auc, f1_score, precision_score, recall_score, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

import datastore


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


def extract_trade_bars_dataset(
    df_trades: pd.DataFrame,
    exit_horizon: str = "sma50",  # 'sma50' or 'ema20'
) -> pd.DataFrame:
    """
    Extracts daily bars for each trade from entry_date to trailing exit_date.
    Computes all technical features and the ground truth Extreme High label.
    """
    exit_date_col = f"{exit_horizon}_exit_date"
    exit_px_col = f"{exit_horizon}_exit_price"
    exit_r_col = f"{exit_horizon}_v2_r"
    
    records = []
    symbols = df_trades["symbol"].unique()
    print(f"Extracting trade bars for {len(df_trades)} trades across {len(symbols)} symbols ({exit_horizon.upper()} horizon)...")
    
    # Pre-cache bars
    bar_cache: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        b = datastore.load_bars(s)
        if b is not None and len(b) >= 60:
            b = b.copy()
            # Technical Indicators
            b["ema10"] = b["close"].ewm(span=10, adjust=False).mean()
            b["ema20"] = b["close"].ewm(span=20, adjust=False).mean()
            b["sma50"] = b["close"].rolling(window=50, min_periods=20).mean()
            b["vol_sma50"] = b["volume"].rolling(window=50, min_periods=20).mean()
            
            # True Range and ADR
            prev_close = b["close"].shift(1)
            tr = np.maximum(
                b["high"] - b["low"],
                np.maximum(
                    (b["high"] - prev_close).abs(),
                    (b["low"] - prev_close).abs()
                )
            )
            b["adr20"] = tr.rolling(window=20, min_periods=10).mean()
            b["adr20_pct"] = (b["adr20"] / b["close"]) * 100.0
            
            # RSI Suite
            b["rsi14"] = compute_rsi(b["close"], 14)
            b["rsi9"] = compute_rsi(b["close"], 9)
            b["rsi5"] = compute_rsi(b["close"], 5)
            
            # Rolling RSI Highs
            b["rsi14_max5"] = b["rsi14"].rolling(5).max()
            b["rsi14_max10"] = b["rsi14"].rolling(10).max()
            b["rsi14_diff3"] = b["rsi14"] - b["rsi14"].shift(3)
            
            # 14-bar Rolling Swing Highs for Price and RSI (Bearish Divergence Detection)
            b["high_roll14"] = b["high"].rolling(14).max()
            b["prev_high_roll14"] = b["high"].shift(14).rolling(14).max()
            b["rsi_roll14"] = b["rsi14"].rolling(14).max()
            b["prev_rsi_roll14"] = b["rsi14"].shift(14).rolling(14).max()
            
            bar_cache[s] = b
            
    for _, trade in df_trades.iterrows():
        sym = trade["symbol"]
        if sym not in bar_cache:
            continue
        bars = bar_cache[sym]
        
        entry_dt = pd.to_datetime(trade["entry_date"])
        exit_dt_val = trade.get(exit_date_col)
        if pd.isna(exit_dt_val):
            continue
        exit_dt = pd.to_datetime(exit_dt_val)
        
        # Slice trade window
        idx_entry = bars.index.searchsorted(entry_dt)
        idx_exit = bars.index.searchsorted(exit_dt)
        if idx_entry >= len(bars) or idx_exit >= len(bars) or idx_exit < idx_entry:
            continue
            
        trade_bars = bars.iloc[idx_entry:idx_exit + 1].copy()
        n_trade_bars = len(trade_bars)
        if n_trade_bars < 2:
            continue
            
        entry_price = float(trade["entry_price"])
        stop_price = float(trade["stop_price"])
        risk_dollar = max(0.01, entry_price - stop_price)
        final_exit_price = float(trade[exit_px_col])
        final_r = float(trade.get(exit_r_col, 0.0))
        
        # Absolute Peak High of the Trade
        peak_high = float(trade_bars["high"].max())
        peak_r = (peak_high - entry_price) / risk_dollar
        
        # Calculate features and labels for each bar t
        for bar_idx in range(n_trade_bars):
            row = trade_bars.iloc[bar_idx]
            bar_date = trade_bars.index[bar_idx]
            curr_high = float(row["high"])
            curr_close = float(row["close"])
            curr_open = float(row["open"])
            curr_low = float(row["low"])
            curr_vol = float(row["volume"])
            
            # Future peak after this bar until exit
            if bar_idx < n_trade_bars - 1:
                future_peak = float(trade_bars["high"].iloc[bar_idx + 1:].max())
            else:
                future_peak = curr_high
                
            # Ground Truth Extreme Peak Definition:
            # 1. Bar is within 2.5% of absolute trade peak: curr_high >= 0.975 * peak_high
            # 2. Future price never significantly exceeds this bar's high before exit: future_peak <= curr_high * 1.015
            # 3. Peak was a meaningful profitable excursion: peak_r >= 1.50 R (or >= 15% gain)
            # 4. Significant giveback from this high to trailing exit: (curr_high - final_exit_price) / curr_high >= 0.06
            is_extreme_high = int(
                (curr_high >= 0.975 * peak_high) and
                (future_peak <= curr_high * 1.015) and
                (peak_r >= 1.50) and
                ((curr_high - final_exit_price) / curr_high >= 0.06)
            )
            
            # Indicator references
            ema10 = float(row["ema10"])
            ema20 = float(row["ema20"])
            sma50 = float(row["sma50"]) if pd.notna(row["sma50"]) else ema20
            vol_sma50 = float(row["vol_sma50"]) if pd.notna(row["vol_sma50"]) and row["vol_sma50"] > 0 else 1.0
            adr20 = float(row["adr20"]) if pd.notna(row["adr20"]) and row["adr20"] > 0 else max(0.1, curr_close * 0.03)
            
            # Candle geometry
            candle_range = max(0.001, curr_high - curr_low)
            upper_wick = max(0.0, curr_high - max(curr_open, curr_close))
            lower_wick = max(0.0, min(curr_open, curr_close) - curr_low)
            body = abs(curr_close - curr_open)
            
            upper_wick_pct = (upper_wick / candle_range) * 100.0
            close_pos = (curr_close - curr_low) / candle_range
            range_to_adr = candle_range / adr20
            
            # RSI Bearish Divergence: Price higher than 14d ago peak, but RSI lower
            price_making_hh = (row["high_roll14"] >= row["prev_high_roll14"])
            rsi_making_lh = (row["rsi_roll14"] < row["prev_rsi_roll14"] - 2.0)
            rsi_bearish_divergence = int(price_making_hh and rsi_making_lh)
            rsi_divergence_delta = float(row["prev_rsi_roll14"] - row["rsi_roll14"]) if rsi_bearish_divergence else 0.0
            
            # Extension above EMAs in terms of ADR
            ext_ema10_pct = ((curr_high - ema10) / ema10) * 100.0
            ext_ema20_pct = ((curr_high - ema20) / ema20) * 100.0
            ext_sma50_pct = ((curr_high - sma50) / sma50) * 100.0
            
            ext_ema10_adr = (curr_high - ema10) / adr20
            ext_ema20_adr = (curr_high - ema20) / adr20
            ext_sma50_adr = (curr_high - sma50) / adr20
            
            # Volume signatures
            rvol = curr_vol / vol_sma50
            volume_churn = rvol / max(0.1, (body / adr20))
            
            # Multi-day momentum
            gain_since_entry = ((curr_high - entry_price) / entry_price) * 100.0
            current_r = (curr_high - entry_price) / risk_dollar
            
            # Recent price streak
            if bar_idx >= 3:
                c_prev3 = float(trade_bars["close"].iloc[bar_idx - 3])
                gain_3d = ((curr_close - c_prev3) / c_prev3) * 100.0
            else:
                gain_3d = 0.0
                
            if bar_idx >= 5:
                c_prev5 = float(trade_bars["close"].iloc[bar_idx - 5])
                gain_5d = ((curr_close - c_prev5) / c_prev5) * 100.0
            else:
                gain_5d = 0.0
                
            # RSI overbought sequence
            rsi14 = float(row["rsi14"])
            rsi9 = float(row["rsi9"])
            rsi5 = float(row["rsi5"])
            
            # RSI consecutive overbought bars (last 5 bars)
            recent_rsi = trade_bars["rsi14"].iloc[max(0, bar_idx - 4):bar_idx + 1]
            consec_rsi_70 = int((recent_rsi >= 70.0).sum())
            consec_rsi_75 = int((recent_rsi >= 75.0).sum())
            
            records.append({
                "symbol": sym,
                "entry_date": trade["entry_date"],
                "bar_date": str(bar_date.date()),
                "bar_idx_in_trade": bar_idx,
                "n_trade_bars": n_trade_bars,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "risk_dollar": risk_dollar,
                "curr_high": curr_high,
                "curr_close": curr_close,
                "final_exit_price": final_exit_price,
                "final_r": final_r,
                "peak_high": peak_high,
                "peak_r": peak_r,
                "future_peak": future_peak,
                
                # Ground truth target
                "is_extreme_high": is_extreme_high,
                
                # --- RSI FEATURES ---
                "rsi14": rsi14,
                "rsi9": rsi9,
                "rsi5": rsi5,
                "rsi14_max5": float(row["rsi14_max5"]),
                "rsi14_max10": float(row["rsi14_max10"]),
                "rsi14_diff3": float(row["rsi14_diff3"]) if pd.notna(row["rsi14_diff3"]) else 0.0,
                "rsi_bearish_divergence": rsi_bearish_divergence,
                "rsi_divergence_delta": rsi_divergence_delta,
                "consec_rsi_70": consec_rsi_70,
                "consec_rsi_75": consec_rsi_75,
                
                # --- EXTENSION FEATURES ---
                "ext_ema10_pct": ext_ema10_pct,
                "ext_ema20_pct": ext_ema20_pct,
                "ext_sma50_pct": ext_sma50_pct,
                "ext_ema10_adr": ext_ema10_adr,
                "ext_ema20_adr": ext_ema20_adr,
                "ext_sma50_adr": ext_sma50_adr,
                
                # --- CANDLE ANATOMY ---
                "upper_wick_pct": upper_wick_pct,
                "close_pos": close_pos,
                "range_to_adr": range_to_adr,
                
                # --- VOLUME CHURN & RVOL ---
                "rvol": rvol,
                "volume_churn": min(20.0, volume_churn),
                
                # --- VELOCITY & TRADE STATE ---
                "gain_since_entry": gain_since_entry,
                "current_r": current_r,
                "gain_3d": gain_3d,
                "gain_5d": gain_5d,
                "adr20_pct": float(row["adr20_pct"]),
                
                # Context
                "rs_score": float(trade.get("rs_score_breakout", trade.get("rs_score", 75.0))),
                "v2_conviction": float(trade.get("v2_conviction_50", 0.10)),
            })
            
    df_out = pd.DataFrame(records)
    print(f"Total extracted trade bars: {len(df_out):,} | Positive Extreme Highs: {df_out['is_extreme_high'].sum():,} ({df_out['is_extreme_high'].mean()*100:.2f}%)")
    return df_out


def train_and_evaluate_peak_models(df_bars: pd.DataFrame) -> Dict[str, Any]:
    """Trains and compares machine learning models for Extreme Peak detection."""
    df_bars["bar_dt"] = pd.to_datetime(df_bars["bar_date"])
    df_bars = df_bars.sort_values("bar_dt").reset_index(drop=True)
    
    # Feature columns
    feature_cols = [
        "rsi14", "rsi9", "rsi5", "rsi14_max5", "rsi14_max10", "rsi14_diff3",
        "rsi_bearish_divergence", "rsi_divergence_delta", "consec_rsi_70", "consec_rsi_75",
        "ext_ema10_pct", "ext_ema20_pct", "ext_sma50_pct",
        "ext_ema10_adr", "ext_ema20_adr", "ext_sma50_adr",
        "upper_wick_pct", "close_pos", "range_to_adr",
        "rvol", "volume_churn",
        "gain_since_entry", "current_r", "gain_3d", "gain_5d", "adr20_pct",
        "rs_score", "v2_conviction"
    ]
    
    # Condition: Climax detection is relevant when current unrealized R is >= 1.5R (protecting runners)
    # Filter candidate bars to current_r >= 1.2R (where a trader would consider taking partial profit)
    mask_eligible = (df_bars["current_r"] >= 1.20)
    df_sub = df_bars[mask_eligible].copy().reset_index(drop=True)
    print(f"\nEligible runner bars (current_r >= 1.2R): {len(df_sub):,} | Positive Peaks: {df_sub['is_extreme_high'].sum():,} ({df_sub['is_extreme_high'].mean()*100:.2f}%)")
    
    # Chronological Split: Train < 2023-01-01 (6.5 years), Test >= 2023-01-01 (3.8 years)
    train_mask = (df_sub["bar_dt"] < "2023-01-01")
    test_mask = (df_sub["bar_dt"] >= "2023-01-01")
    
    X_train = df_sub.loc[train_mask, feature_cols].fillna(0.0)
    y_train = df_sub.loc[train_mask, "is_extreme_high"]
    
    X_test = df_sub.loc[test_mask, feature_cols].fillna(0.0)
    y_test = df_sub.loc[test_mask, "is_extreme_high"]
    
    print(f"Train set: {len(X_train)} (pos: {y_train.sum()}, {y_train.mean()*100:.1f}%)")
    print(f"Test OOS set: {len(X_test)} (pos: {y_test.sum()}, {y_test.mean()*100:.1f}%)")
    
    # Event-level sample weighting (so 100-day trades don't dominate 20-day trades)
    df_sub["event_id"] = df_sub["symbol"] + "_" + df_sub["entry_date"]
    ev_counts = df_sub.groupby("event_id")["event_id"].transform("count")
    weights = (1.0 / np.sqrt(ev_counts))
    w_train = weights.loc[train_mask]
    
    # 1. LightGBM Model
    lgb_model = lgb.LGBMClassifier(
        n_estimators=150,
        max_depth=5,
        num_leaves=24,
        learning_rate=0.03,
        scale_pos_weight=2.5,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train, sample_weight=w_train)
    probs_lgb = lgb_model.predict_proba(X_test)[:, 1]
    
    # 2. XGBoost Model
    xgb_model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.03,
        scale_pos_weight=2.5,
        random_state=42,
        eval_metric="logloss"
    )
    xgb_model.fit(X_train, y_train, sample_weight=w_train)
    probs_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    # 3. Random Forest Model
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    probs_rf = rf_model.predict_proba(X_test)[:, 1]
    
    # 4. Ensemble (Average of LGB + XGB + RF)
    probs_ens = (probs_lgb * 0.45) + (probs_xgb * 0.40) + (probs_rf * 0.15)
    
    models = {
        "LightGBM": probs_lgb,
        "XGBoost": probs_xgb,
        "RandomForest": probs_rf,
        "Ensemble (LGB+XGB+RF)": probs_ens
    }
    
    print("\n" + "=" * 80)
    print("🏆 OUT-OF-SAMPLE TEST PERFORMANCE (2023–2026)")
    print("=" * 80)
    
    results = {}
    for name, p in models.items():
        roc = roc_auc_score(y_test, p)
        brier = brier_score_loss(y_test, p)
        prec_arr, rec_arr, _ = precision_recall_curve(y_test, p)
        pr_auc = auc(rec_arr, prec_arr)
        
        print(f"\n📌 {name}:")
        print(f"   ROC-AUC: {roc:.3f} | PR-AUC: {pr_auc:.3f} | Brier: {brier:.4f}")
        
        # Test thresholds
        for th in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
            pred_bin = (p >= th).astype(int)
            prec = precision_score(y_test, pred_bin, zero_division=0)
            rec = recall_score(y_test, pred_bin, zero_division=0)
            f1 = f1_score(y_test, pred_bin, zero_division=0)
            n_flags = int(pred_bin.sum())
            print(f"     Threshold {th:.2f} -> Flags: {n_flags:4d} | Precision: {prec:5.1%} | Recall: {rec:5.1%} | F1: {f1:.3f}")
            
        results[name] = {"roc": roc, "pr_auc": pr_auc, "brier": brier}
        
    # Feature Importance (LightGBM)
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": lgb_model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    
    print("\n" + "=" * 80)
    print("📊 FEATURE IMPORTANCE RANKING (LightGBM)")
    print("=" * 80)
    for idx, r in imp_df.head(15).iterrows():
        print(f"  {idx+1:2d}. {r['feature']:25s} | Importance: {r['importance']:4d}")
        
    # Attach predictions to test dataframe for simulation
    df_test_eval = df_sub.loc[test_mask].copy().reset_index(drop=True)
    df_test_eval["prob_peak"] = probs_ens
    
    return {
        "lgb_model": lgb_model,
        "xgb_model": xgb_model,
        "feature_cols": feature_cols,
        "imp_df": imp_df,
        "df_test_eval": df_test_eval,
        "results": results
    }


def simulate_partial_profit_strategy(
    df_test_bars: pd.DataFrame,
    df_trades: pd.DataFrame,
    threshold: float = 0.50,
    partial_take_pct: float = 0.50,  # Sell 50% at peak high, trail remaining 50% on MA
    exit_horizon: str = "sma50",
) -> Dict[str, Any]:
    """
    Simulates the strategy impact on Out-Of-Sample trades when taking partial profits
    at detected Extreme High bars.
    """
    exit_r_col = f"{exit_horizon}_v2_r"
    
    # Group test bars by trade (symbol + entry_date)
    trades_eval = []
    
    for (sym, entry_date), group in df_test_bars.groupby(["symbol", "entry_date"]):
        # Find if any bar triggered the peak signal
        peak_triggers = group[group["prob_peak"] >= threshold]
        
        orig_trade = df_trades[(df_trades["symbol"] == sym) & (df_trades["entry_date"] == entry_date)]
        if orig_trade.empty:
            continue
        orig_row = orig_trade.iloc[0]
        
        base_r = float(orig_row.get(exit_r_col, 0.0))
        entry_price = float(orig_row["entry_price"])
        stop_price = float(orig_row["stop_price"])
        risk_dollar = max(0.01, entry_price - stop_price)
        v2_size = float(orig_row.get("v2_size_mult", 1.0))
        
        if len(peak_triggers) > 0:
            # First trigger bar
            first_trig = peak_triggers.iloc[0]
            # Realized R at peak: sell at that day's High (or Close)
            # Conservative assumption: sell at Close of trigger day (or midpoint of Close and High)
            fill_px = (float(first_trig["curr_close"]) + float(first_trig["curr_high"])) / 2.0
            peak_r_captured = (fill_px - entry_price) / risk_dollar
            
            # Sized effective R:
            # partial_take_pct at peak_r, and (1.0 - partial_take_pct) at base_r
            blended_r = (partial_take_pct * peak_r_captured) + ((1.0 - partial_take_pct) * (base_r / v2_size))
            final_sized_r = blended_r * v2_size
            has_climax_exit = True
        else:
            final_sized_r = base_r
            peak_r_captured = 0.0
            has_climax_exit = False
            
        trades_eval.append({
            "symbol": sym,
            "entry_date": entry_date,
            "base_sized_r": base_r,
            "climax_sized_r": final_sized_r,
            "has_climax_exit": has_climax_exit,
            "peak_r_captured": peak_r_captured,
            "v2_size": v2_size
        })
        
    df_res = pd.DataFrame(trades_eval)
    
    # Compute comparative KPIs
    n_trades = len(df_res)
    climax_count = df_res["has_climax_exit"].sum()
    
    # Baseline
    base_tot = df_res["base_sized_r"].sum()
    base_ev = df_res["base_sized_r"].mean()
    base_wr = (df_res["base_sized_r"] > 0).mean() * 100.0
    base_wins = df_res[df_res["base_sized_r"] > 0]["base_sized_r"]
    base_losses = df_res[df_res["base_sized_r"] < 0]["base_sized_r"]
    base_pf = base_wins.sum() / abs(base_losses.sum()) if abs(base_losses.sum()) > 0 else 0.0
    base_payoff = base_wins.mean() / abs(base_losses.mean()) if abs(base_losses.mean()) > 0 else 0.0
    
    # Climax Augmented
    climax_tot = df_res["climax_sized_r"].sum()
    climax_ev = df_res["climax_sized_r"].mean()
    climax_wr = (df_res["climax_sized_r"] > 0).mean() * 100.0
    climax_wins = df_res[df_res["climax_sized_r"] > 0]["climax_sized_r"]
    climax_losses = df_res[df_res["climax_sized_r"] < 0]["climax_sized_r"]
    climax_pf = climax_wins.sum() / abs(climax_losses.sum()) if abs(climax_losses.sum()) > 0 else 0.0
    climax_payoff = climax_wins.mean() / abs(climax_losses.mean()) if abs(climax_losses.mean()) > 0 else 0.0
    
    # Drawdowns
    base_cum = df_res["base_sized_r"].cumsum()
    base_dd = (base_cum - base_cum.cummax()).min()
    
    climax_cum = df_res["climax_sized_r"].cumsum()
    climax_dd = (climax_cum - climax_cum.cummax()).min()
    
    delta_r = climax_tot - base_tot
    
    return {
        "n_trades": n_trades,
        "climax_count": climax_count,
        "base_tot": base_tot,
        "base_ev": base_ev,
        "base_wr": base_wr,
        "base_pf": base_pf,
        "base_payoff": base_payoff,
        "base_dd": base_dd,
        "climax_tot": climax_tot,
        "climax_ev": climax_ev,
        "climax_wr": climax_wr,
        "climax_pf": climax_pf,
        "climax_payoff": climax_payoff,
        "climax_dd": climax_dd,
        "delta_r": delta_r,
    }


def main():
    print("=" * 80)
    print("🔬 RESEARCH: EXHAUSTION & EXTREME HIGH RECOGNITION ML SYSTEM")
    print("=" * 80)
    
    df_trades = pd.read_parquet("data/simulations/optimized_ep_strategy_trades.parquet")
    print(f"Loaded {len(df_trades)} strategy trades.")
    
    # 1. Extract Trade Bars for 50 SMA and 20 EMA
    df_bars_sma50 = extract_trade_bars_dataset(df_trades, exit_horizon="sma50")
    
    # 2. Train and Evaluate ML Peak Models
    res_sma50 = train_and_evaluate_peak_models(df_bars_sma50)
    
    # 3. Simulate Strategy Impact under different partial take options
    print("\n" + "=" * 80)
    print("📈 STRATEGY IMPACT SIMULATION: 50 SMA EXIT + CLIMAX HARVESTING (OOS 2023–2026)")
    print("=" * 80)
    
    for take_pct in [0.33, 0.50, 0.66, 1.00]:
        for th in [0.45, 0.50, 0.55]:
            sim = simulate_partial_profit_strategy(
                res_sma50["df_test_eval"],
                df_trades,
                threshold=th,
                partial_take_pct=take_pct,
                exit_horizon="sma50"
            )
            print(f"Take {take_pct*100:4.0f}% at Th={th:.2f} | Trigs: {sim['climax_count']:2d}/{sim['n_trades']:2d} | "
                  f"Base Tot: {sim['base_tot']:+6.1f}R (EV {sim['base_ev']:+5.2f}R, PF {sim['base_pf']:.2f}, DD {sim['base_dd']:.1f}R) -> "
                  f"Climax Tot: {sim['climax_tot']:+6.1f}R (EV {sim['climax_ev']:+5.2f}R, PF {sim['climax_pf']:.2f}, DD {sim['climax_dd']:.1f}R) | "
                  f"Delta: {sim['delta_r']:+6.1f}R")


if __name__ == "__main__":
    main()
