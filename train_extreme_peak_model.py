"""
train_extreme_peak_model.py

Trains and persists the institutional Extreme Peak / Climax High Recognition ML model:
- Learns the probability that an active trade's daily high represents the terminal climax
  before the trailing moving average exit is hit.
- Saves model pipeline to data/models/ep_extreme_peak_classifier.pkl.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_score, recall_score, f1_score

import research_extreme_peak_model as rep

OUT_MODEL_PATH = Path("data/models/ep_extreme_peak_classifier.pkl")

def main():
    print("=" * 80)
    print("🚀 TRAINING INSTITUTIONAL EXTREME PEAK / CLIMAX HARVESTING ML MODEL")
    print("=" * 80)
    
    df_trades = pd.read_parquet("data/simulations/optimized_ep_strategy_trades.parquet")
    print(f"Loaded {len(df_trades)} strategy trades.")
    
    # Extract trade bars across both horizons to ensure rich generalization across short and long holding periods
    print("Extracting trade bars for 20 EMA and 50 SMA horizons...")
    df_bars_20 = rep.extract_trade_bars_dataset(df_trades, exit_horizon="ema20")
    df_bars_50 = rep.extract_trade_bars_dataset(df_trades, exit_horizon="sma50")
    
    df_bars_20["horizon"] = "ema20"
    df_bars_50["horizon"] = "sma50"
    
    df_bars = pd.concat([df_bars_20, df_bars_50], ignore_index=True)
    df_bars["bar_dt"] = pd.to_datetime(df_bars["bar_date"])
    df_bars = df_bars.sort_values("bar_dt").reset_index(drop=True)
    
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
    
    # Train condition: runner bar where current_r >= 1.20
    mask_elig = (df_bars["current_r"] >= 1.20)
    df_train = df_bars[mask_elig].copy().reset_index(drop=True)
    
    print(f"Total eligible training bars: {len(df_train):,} | Positive Extreme Highs: {df_train['is_extreme_high'].sum():,} ({df_train['is_extreme_high'].mean()*100:.2f}%)")
    
    X = df_train[feature_cols].fillna(0.0)
    y = df_train["is_extreme_high"]
    
    # Event-level weights
    df_train["event_id"] = df_train["symbol"] + "_" + df_train["entry_date"] + "_" + df_train["horizon"]
    ev_counts = df_train.groupby("event_id")["event_id"].transform("count")
    sample_weights = (1.0 / np.sqrt(ev_counts))
    
    # Base Estimators for Calibrated Voting Ensemble
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=150,
        max_depth=5,
        num_leaves=24,
        learning_rate=0.03,
        scale_pos_weight=2.2,
        random_state=42,
        verbose=-1
    )
    
    xgb_clf = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.03,
        scale_pos_weight=2.2,
        random_state=42,
        eval_metric="logloss"
    )
    
    print("Fitting calibrated LightGBM...")
    cal_lgb = CalibratedClassifierCV(estimator=lgb_clf, method="isotonic", cv=3)
    cal_lgb.fit(X, y, sample_weight=sample_weights)
    
    print("Fitting calibrated XGBoost...")
    cal_xgb = CalibratedClassifierCV(estimator=xgb_clf, method="isotonic", cv=3)
    cal_xgb.fit(X, y, sample_weight=sample_weights)
    
    # Package Model
    model_payload = {
        "cal_lgb": cal_lgb,
        "cal_xgb": cal_xgb,
        "feature_cols": feature_cols,
        "threshold_ema20": 0.45,
        "threshold_sma50": 0.55,
        "trained_date": "2026-10-13",
        "n_samples": len(X),
        "target_desc": "Extreme High / Terminal Climax before Trailing Exit",
    }
    
    OUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_payload, OUT_MODEL_PATH)
    print(f"✅ Successfully saved Extreme Peak Classifier to {OUT_MODEL_PATH}")
    
    # Quick sanity check prediction
    probs_lgb = cal_lgb.predict_proba(X.iloc[:10])[:, 1]
    probs_xgb = cal_xgb.predict_proba(X.iloc[:10])[:, 1]
    blend = 0.55 * probs_lgb + 0.45 * probs_xgb
    print(f"Sample test predictions: {blend[:5].round(3)}")


if __name__ == "__main__":
    main()
