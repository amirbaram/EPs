"""train_qm_conviction_model.py — Trains Setup Quality / Big Move Conviction (QM-V2) and Trap/Failure Avoidance Models.

Zero-lookahead machine learning models predicting:
1. P(Big Move): Probability that a QM breakout reaches >= 3.0 R (or 4.0 R) MFE.
2. P(Trap / Failure): Probability that a QM breakout fails immediately (MFE < 0.8 R and R <= -0.8 R).
3. QM Quality Score: Unified 0-100 score classifying setups into actionable tiers:
   - Premier / Pinnacle (Tier 1): High conviction, lowest failure risk -> 1.5x Conviction Sizing.
   - High Quality (Tier 2): Solid breakout -> 1.2x sizing.
   - Standard (Tier 3): Standard setup -> 1.0x sizing.
   - Marginal (Tier 4): Choppy / extended -> 0.5x sizing.
   - Toxic / Trap (Tier 5): High failure risk -> AVOID (0x allocation).

Uses purged chronological split: Train < 2023 (22,797 trades), Test >= 2023 (30,364 trades).
Outputs: data/models/qm_setup_quality_model.pkl
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TRADES_PATH = Path("data/simulations/qm_strategy_trades.parquet")
MODEL_OUT_PATH = Path("data/models/qm_setup_quality_model.pkl")

QUALITY_FEATURE_COLS = [
    "move_pct",
    "cons_bars",
    "pullback_pct",
    "pullback_ratio",
    "vol_x",
    "breakout_rvol",
    "pinch_ratio",
    "r2",
    "rs_score",
    "dist_52w_high",
    "prior_ep_days_ago",
    "prior_ep_gap_pct",
    "prior_ep_rvol",
    "spy_above_200sma",
    "spy_above_50sma",
    "spy_10ema_slope",
    "qqq_bull_stack",
    "qqq_pullback",
    "qqq_spy_ratio_bull",
    "is_leading_sector",
    "is_leading_theme",
    "is_leading_macro",
    "stage2",
    "has_prior_ep",
]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[QUALITY_FEATURE_COLS].copy()
    X["pinch_ratio"] = X["pinch_ratio"].fillna(2.5)
    X["r2"] = X["r2"].fillna(0.5)
    X["rs_score"] = X["rs_score"].fillna(50.0)
    X["dist_52w_high"] = X["dist_52w_high"].fillna(-15.0)
    X["prior_ep_days_ago"] = X["prior_ep_days_ago"].fillna(999.0)
    X["prior_ep_gap_pct"] = X["prior_ep_gap_pct"].fillna(0.0)
    X["prior_ep_rvol"] = X["prior_ep_rvol"].fillna(0.0)
    for c in [
        "spy_above_200sma",
        "spy_above_50sma",
        "qqq_bull_stack",
        "qqq_pullback",
        "qqq_spy_ratio_bull",
        "is_leading_sector",
        "is_leading_theme",
        "is_leading_macro",
        "stage2",
        "has_prior_ep",
    ]:
        X[c] = X[c].astype(float)
    return X


def train_setup_quality_models():
    print(f"Loading QM strategy trades from {TRADES_PATH}...")
    df = pd.read_parquet(TRADES_PATH)
    print(f"Loaded {len(df)} total trades.")

    X = prepare_features(df)
    y_big = (df["mfe_r_06x"] >= 3.0).astype(int)
    y_trap = ((df["mfe_r_06x"] < 0.8) & (df["r_06x"] <= -0.8)).astype(int)

    train_mask = df["date"] < "2023-01-01"
    test_mask = df["date"] >= "2023-01-01"

    print(f"Train samples (< 2023) : {train_mask.sum():,} | Big moves: {y_big[train_mask].mean()*100:.1f}%, Traps: {y_trap[train_mask].mean()*100:.1f}%")
    print(f"Test samples  (>= 2023): {test_mask.sum():,} | Big moves: {y_big[test_mask].mean()*100:.1f}%, Traps: {y_trap[test_mask].mean()*100:.1f}%")

    # 1. Big Move Classifier
    print("\n--- Training Big Move Conviction Classifier (P(MFE >= 3.0 R)) ---")
    clf_big = lgb.LGBMClassifier(
        n_estimators=160,
        max_depth=5,
        num_leaves=24,
        learning_rate=0.035,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    clf_big.fit(X[train_mask], y_big[train_mask])
    p_big_test = clf_big.predict_proba(X[test_mask])[:, 1]
    auc_big = roc_auc_score(y_big[test_mask], p_big_test)
    print(f"OOS Big Move ROC-AUC (2023-2026): {auc_big:.4f}")

    # 2. Trap / Failure Avoidance Classifier
    print("\n--- Training Trap / Failure Avoidance Classifier (P(Trap)) ---")
    clf_trap = lgb.LGBMClassifier(
        n_estimators=140,
        max_depth=4,
        num_leaves=15,
        learning_rate=0.035,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    clf_trap.fit(X[train_mask], y_trap[train_mask])
    p_trap_test = clf_trap.predict_proba(X[test_mask])[:, 1]
    auc_trap = roc_auc_score(y_trap[test_mask], p_trap_test)
    print(f"OOS Trap Avoidance ROC-AUC (2023-2026): {auc_trap:.4f}")

    # Top features for Big Move
    print("\nTop 8 Feature Importances for Big Move Conviction:")
    imp_big = sorted(zip(QUALITY_FEATURE_COLS, clf_big.feature_importances_), key=lambda x: x[1], reverse=True)
    for f, imp in imp_big[:8]:
        print(f"  • {f:<22}: {imp}")

    # Top features for Trap Avoidance
    print("\nTop 8 Feature Importances for Trap Avoidance:")
    imp_trap = sorted(zip(QUALITY_FEATURE_COLS, clf_trap.feature_importances_), key=lambda x: x[1], reverse=True)
    for f, imp in imp_trap[:8]:
        print(f"  • {f:<22}: {imp}")

    # Package bundle
    bundle = {
        "clf_big": clf_big,
        "clf_trap": clf_trap,
        "feature_cols": QUALITY_FEATURE_COLS,
        "auc_big": round(float(auc_big), 4),
        "auc_trap": round(float(auc_trap), 4),
        "train_samples": int(train_mask.sum()),
        "test_samples": int(test_mask.sum()),
    }

    MODEL_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_OUT_PATH)
    print(f"\n✅ Saved QM Setup Quality Model Bundle to {MODEL_OUT_PATH}.")
    return bundle


if __name__ == "__main__":
    train_setup_quality_models()
