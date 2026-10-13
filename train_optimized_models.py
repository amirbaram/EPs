"""
Production Retraining Pipeline for:
1. 20-Day MAE Stop-Loss Quantile Regressor (ep_delayed_stop_model.pkl)
2. Breakout V2 Conviction Classifier (ep_v2_breakout_conviction.pkl)
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import joblib

ROOT = Path(__file__).resolve().parent

def run_retraining():
    print("=" * 80)
    print("🚀 INSTITUTIONAL ML RETRAINING PIPELINE")
    print("=" * 80)
    
    in_path = ROOT / "data" / "simulations" / "breakout_v2_rs_study.parquet"
    print(f"📖 Loading baseline simulated breakout trades from: {in_path}")
    df = pd.read_parquet(in_path)
    print(f"Loaded {len(df):,} breakouts.")
    
    # 1. Feature selection: 28 rich zero-lookahead institutional features
    features = [
        "gap_pct", "rvol", "dvol_m", "close_pos", "adr_pct",
        "day1_body_pct", "day1_upper_wick_pct", "prev_day_gap_up",
        "breakout_rvol", "breakout_gap_pct", "shelf_pullback_pct", "days_to_breakout",
        "rs_score_breakout", "dist_52w_breakout", "stage2_breakout",
        "pre_consol_tightness_20d", "pre_runup_20d", "pre_runup_65d", "pre_volume_dryup_5d",
        "pre_above_sma50", "pre_above_sma200", "pre_sma50_slope",
        "sec_m1_breakout", "sec_m3_breakout", "thm_m1_breakout", "thm_m3_breakout",
        "spy_ret_20d_ep", "spy_ret_20d_entry", "initial_risk_pct"
    ]
    features = [c for c in features if c in df.columns]
    print(f"Selected {len(features)} institutional features.")
    
    # Clean features
    X = df[features].copy()
    for col in features:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
        
    df["entry_dt"] = pd.to_datetime(df["entry_date"])
    df["year"] = df["entry_dt"].dt.year
    
    # -------------------------------------------------------------
    # MODEL 1: 20-DAY MAE STOP-LOSS MODEL (QUANTILE REGRESSION)
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("1. TRAINING 20-DAY MAE STOP-LOSS MODEL (alpha=0.90 QUANTILE)")
    print("-" * 80)
    
    # We train on WINNING trades (R >= 1.0) so the model learns the adverse excursion needed before winning
    # mae_pct is positive (e.g. 4.5% drawdown)
    win_mask = df["r_multiple"] >= 1.0
    print(f"Total winning trades in universe: {win_mask.sum():,} ({win_mask.mean()*100:.1f}%)")
    
    # Train OOS validation first (< 2023 train, >= 2023 test)
    tr_m = win_mask & (df["year"] < 2023)
    te_m = df["year"] >= 2023
    
    stop_model_val = lgb.LGBMRegressor(
        objective="quantile",
        alpha=0.90, # 90% coverage of winning adverse excursions
        n_estimators=120,
        learning_rate=0.03,
        num_leaves=16,
        min_child_samples=15,
        subsample=0.85,
        random_state=42,
        verbose=-1
    )
    stop_model_val.fit(X.loc[tr_m], df.loc[tr_m, "mae_pct"])
    
    pred_mae_te = stop_model_val.predict(X.loc[te_m])
    clamped_stop_te = np.clip(pred_mae_te, 4.5, df.loc[te_m, "initial_risk_pct"].values)
    print(f"OOS (2023+) Validation:")
    print(f"  • Predicted Stop Distance: Mean = {clamped_stop_te.mean():.2f}% | Median = {np.median(clamped_stop_te):.2f}%")
    print(f"  • Shakeout rate on OOS Big Winners (R>=2): {(df.loc[te_m & (df['r_multiple'] >= 2.0), 'mae_pct'] >= clamped_stop_te[df.loc[te_m, 'r_multiple'] >= 2.0]).mean()*100:.1f}%")
    
    # Now train Full Production Stop Model on all historical winners
    stop_model_prod = lgb.LGBMRegressor(
        objective="quantile",
        alpha=0.90,
        n_estimators=150,
        learning_rate=0.03,
        num_leaves=16,
        min_child_samples=15,
        subsample=0.85,
        random_state=42,
        verbose=-1
    )
    stop_model_prod.fit(X.loc[win_mask], df.loc[win_mask, "mae_pct"])
    
    # Predict dynamic stop for ALL trades
    pred_mae_all = stop_model_prod.predict(X)
    # Bounded stop logic:
    # 1. Minimum stop distance: 4.5% (protects monster runners like CAPR, ASTS, TSLA)
    # 2. Maximum stop distance: Day 1 Low / Shelf Low distance (never wider than structural support)
    df["ml_stop_pct"] = np.clip(pred_mae_all, 4.5, df["initial_risk_pct"].values)
    df["ml_stop_price"] = df["entry_price"] * (1.0 - df["ml_stop_pct"] / 100.0)
    
    stop_out_path = ROOT / "data" / "models" / "ep_delayed_stop_model.pkl"
    joblib.dump({
        "model": stop_model_prod,
        "feature_cols": features,
        "alpha": 0.90,
        "min_stop_pct": 4.5,
        "description": "20-Day MAE Stop-Loss Quantile Regressor (alpha=0.90, clamped 4.5% to D1 Low)"
    }, stop_out_path)
    print(f"✅ Saved Production Stop-Loss Model to: {stop_out_path}")
    
    # -------------------------------------------------------------
    # MODEL 2: BREAKOUT V2 CONVICTION CLASSIFIER
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("2. TRAINING BREAKOUT V2 CONVICTION CLASSIFIER")
    print("-" * 80)
    
    # Target: Did the stock deliver a strong institutional run?
    # We train on MFE >= +50% as well as Trade Win with calibrated probabilities
    y_target_50 = (df["mfe_pct"] >= 50.0).astype(int)
    y_target_win = (df["r_multiple"] > 0).astype(int)
    
    print(f"Universe Base Rates -> MFE >= +50%: {y_target_50.mean()*100:.1f}% ({y_target_50.sum():,} events) | Win: {y_target_win.mean()*100:.1f}%")
    
    # OOS Validation (< 2023 train, >= 2023 test)
    tr_idx = df["year"] < 2023
    te_idx = df["year"] >= 2023
    
    base_xgb = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric="logloss"
    )
    
    # Train calibrated model
    calib_val = CalibratedClassifierCV(estimator=base_xgb, method="isotonic", cv=5)
    calib_val.fit(X.loc[tr_idx], y_target_50.loc[tr_idx])
    
    p_val_te = calib_val.predict_proba(X.loc[te_idx])[:, 1]
    auc_val = roc_auc_score(y_target_50.loc[te_idx], p_val_te)
    brier_val = brier_score_loss(y_target_50.loc[te_idx], p_val_te)
    print(f"OOS (2023+) Validation -> ROC-AUC: {auc_val:.3f} | Brier: {brier_val:.4f}")
    
    # Train Full Production V2 Model
    base_xgb_prod = xgb.XGBClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric="logloss"
    )
    calib_prod = CalibratedClassifierCV(estimator=base_xgb_prod, method="isotonic", cv=5)
    calib_prod.fit(X, y_target_50)
    
    # Also train Win Classifier
    base_xgb_win = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric="logloss"
    )
    calib_win = CalibratedClassifierCV(estimator=base_xgb_win, method="isotonic", cv=5)
    calib_win.fit(X, y_target_win)
    
    # Predict conviction scores on dataset
    df["v2_conviction_50"] = np.round(calib_prod.predict_proba(X)[:, 1], 4)
    df["v2_win_prob"] = np.round(calib_win.predict_proba(X)[:, 1], 4)
    
    # Update v2_breakout with the calibrated model
    df["v2_breakout"] = df["v2_conviction_50"]
    
    v2_out_path = ROOT / "data" / "models" / "ep_v2_breakout_conviction.pkl"
    joblib.dump({
        "model_50": calib_prod,
        "model_win": calib_win,
        "feature_cols": features,
        "description": "Calibrated Institutional Breakout V2 Conviction Classifier (P(+50%) and P(Win))"
    }, v2_out_path)
    print(f"✅ Saved Production V2 Conviction Model to: {v2_out_path}")
    
    # Save enriched dataset
    df.to_parquet(in_path, index=False)
    print(f"✅ Updated {in_path} with new ml_stop_price, ml_stop_pct, v2_conviction_50, and v2_win_prob.")
    print("=" * 80)

if __name__ == "__main__":
    run_retraining()
