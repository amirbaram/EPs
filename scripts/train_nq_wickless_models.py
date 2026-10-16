"""
Dedicated Training Script: NQ / MNQ Wickless Intraday Pivot Models (15m, 1h, 4h).
Trains strictly on At-Extreme features (Zero Wick / Zero Close Leakage), saving:
- data/models/nq_wickless_intraday_models.pkl
- data/simulations/nq_wickless_intraday_oos_results.json
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nq_ml_engine

STUDY_START = "2016-07-15 00:00:00"
TRAIN_CUTOFF = "2024-07-15 00:00:00"
CAL_START = "2023-07-15 00:00:00"
OOS_START = "2024-07-15 00:00:00"
OOS_END = "2026-07-14 23:59:59"

OUT_MODELS_PATH = ROOT / "data" / "models" / "nq_wickless_intraday_models.pkl"
OUT_RESULTS_PATH = ROOT / "data" / "simulations" / "nq_wickless_intraday_oos_results.json"


def fit_wickless_classifier(
    df_cand: pd.DataFrame,
    label_col: str,
    feature_cols: List[str],
    name: str,
) -> Dict[str, Any]:
    """Fits LightGBM + XGBoost ensemble strictly on at-extreme features with isotonic calibration."""
    tr_core = df_cand[df_cand.index < pd.Timestamp(CAL_START)]
    tr_cal = df_cand[(df_cand.index >= pd.Timestamp(CAL_START)) & (df_cand.index < pd.Timestamp(TRAIN_CUTOFF))]
    tr_all = df_cand[df_cand.index < pd.Timestamp(TRAIN_CUTOFF)]

    y_core = tr_core[label_col].to_numpy(dtype=int)
    pos_rate = max(0.01, float(y_core.mean()))
    spw = float(np.clip((1.0 - pos_rate) / max(pos_rate, 1e-4) * 0.85, 2.0, 7.5))

    def _fit_pair(X_df: pd.DataFrame, y_arr: np.ndarray):
        X_clean = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        m_lgb = lgb.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.035,
            num_leaves=31,
            max_depth=6,
            min_child_samples=40,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
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
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        m_lgb.fit(X_clean, y_arr)
        m_xgb.fit(X_clean, y_arr)
        return m_lgb, m_xgb

    lgb_c, xgb_c = _fit_pair(tr_core[feature_cols], y_core)
    X_cal = tr_cal[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_cal = tr_cal[label_col].to_numpy(dtype=int)
    p_cal = 0.55 * lgb_c.predict_proba(X_cal)[:, 1] + 0.45 * xgb_c.predict_proba(X_cal)[:, 1]

    iso = IsotonicRegression(y_min=0.01, y_max=0.96, out_of_bounds="clip")
    iso.fit(p_cal, y_cal)

    y_all = tr_all[label_col].to_numpy(dtype=int)
    lgb_all, xgb_all = _fit_pair(tr_all[feature_cols], y_all)

    imp = pd.Series(lgb_all.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top_features = [{"feature": k, "importance": int(v)} for k, v in imp.head(12).items()]

    return {
        "name": name,
        "lgb": lgb_all,
        "xgb": xgb_all,
        "iso": iso,
        "feature_cols": feature_cols,
        "top_features": top_features,
    }


def main():
    t_start = time.time()
    print("=" * 80)
    print("TRAINING NQ / MNQ WICKLESS INTRADAY PIVOT MODELS (15M, 1H, 4H)")
    print("=" * 80)

    print("Loading NQ continuous 1m dataset...")
    df_1m = nq_ml_engine.load_nq_1m_base(start_date=STUDY_START, include_live_tail=True)
    print(f"Loaded {len(df_1m):,} 1m bars in {time.time()-t_start:.1f}s")

    all_models: Dict[str, Any] = {}
    report: Dict[str, Any] = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "model_type": "wickless_intraday_at_extreme",
        "study_window": {"start": STUDY_START, "train_cutoff": TRAIN_CUTOFF, "oos_start": OOS_START, "oos_end": OOS_END},
        "timeframes": {},
    }

    tf_cfgs = {
        "15m": {"left": 6, "right": 6, "rev_atr": 1.6},
        "1h": {"left": 5, "right": 5, "rev_atr": 1.5},
        "4h": {"left": 4, "right": 4, "rev_atr": 1.4},
    }

    for tf, cfg in tf_cfgs.items():
        t0 = time.time()
        print(f"\nProcessing Timeframe: {tf} (Wickless At-Extreme)...")
        bars = nq_ml_engine.resample_nq_with_1m_microstructure(df_1m, tf)
        feats = nq_ml_engine.compute_nq_intraday_features(bars, tf)
        labeled = nq_ml_engine.label_nq_pivots_and_excursions(
            feats, pivot_left=cfg["left"], pivot_right=cfg["right"], rev_atr=cfg["rev_atr"]
        )
        labeled = labeled.iloc[200:].copy()

        train_df = labeled[labeled.index < pd.Timestamp(TRAIN_CUTOFF)].copy()
        oos_df = labeled[(labeled.index >= pd.Timestamp(OOS_START)) & (labeled.index <= pd.Timestamp(OOS_END))].copy()

        high_cols = [c for c in nq_ml_engine.PIVOT_HIGH_AT_EXTREME_COLS if c in labeled.columns]
        low_cols = [c for c in nq_ml_engine.PIVOT_LOW_AT_EXTREME_COLS if c in labeled.columns]

        cand_hi_tr = train_df[train_df["is_cand_high"] & (~train_df["crosses_roll_8b"])].copy()
        cand_lo_tr = train_df[train_df["is_cand_low"] & (~train_df["crosses_roll_8b"])].copy()

        print(f"  [{tf}] Training Wickless Pivot High ({len(high_cols)} features on {len(cand_hi_tr):,} candidates)...")
        m_hi = fit_wickless_classifier(cand_hi_tr, "is_pivot_high", high_cols, f"NQ_{tf}_Wickless_High")

        print(f"  [{tf}] Training Wickless Pivot Low ({len(low_cols)} features on {len(cand_lo_tr):,} candidates)...")
        m_lo = fit_wickless_classifier(cand_lo_tr, "is_pivot_low", low_cols, f"NQ_{tf}_Wickless_Low")

        all_models[tf] = {
            "pivot_high": m_hi,
            "pivot_low": m_lo,
            "feature_cols_high": high_cols,
            "feature_cols_low": low_cols,
            "config": cfg,
        }

        # Score OOS to record metrics
        oos_hi = oos_df[oos_df["is_cand_high"] & (~oos_df["crosses_roll_8b"])].copy()
        oos_lo = oos_df[oos_df["is_cand_low"] & (~oos_df["crosses_roll_8b"])].copy()

        X_hi = oos_hi[high_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        p_hi = 0.55 * m_hi["lgb"].predict_proba(X_hi)[:, 1] + 0.45 * m_hi["xgb"].predict_proba(X_hi)[:, 1]
        y_hi = oos_hi["is_pivot_high"].to_numpy(dtype=int)

        X_lo = oos_lo[low_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        p_lo = 0.55 * m_lo["lgb"].predict_proba(X_lo)[:, 1] + 0.45 * m_lo["xgb"].predict_proba(X_lo)[:, 1]
        y_lo = oos_lo["is_pivot_low"].to_numpy(dtype=int)

        from sklearn.metrics import roc_auc_score, average_precision_score
        auc_hi = float(roc_auc_score(y_hi, p_hi))
        pr_hi = float(average_precision_score(y_hi, p_hi))
        auc_lo = float(roc_auc_score(y_lo, p_lo))
        pr_lo = float(average_precision_score(y_lo, p_lo))

        print(f"  [{tf}] OOS High AUC: {auc_hi:.4f} | Low AUC: {auc_lo:.4f} in {time.time()-t0:.1f}s")
        report["timeframes"][tf] = {
            "high": {"auc": auc_hi, "pr_auc": pr_hi, "top_features": m_hi["top_features"][:6]},
            "low": {"auc": auc_lo, "pr_auc": pr_lo, "top_features": m_lo["top_features"][:6]},
        }

    OUT_MODELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(all_models, OUT_MODELS_PATH)
    OUT_RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n✅ Successfully saved dedicated Wickless models to {OUT_MODELS_PATH}")
    print(f"✅ Successfully saved report to {OUT_RESULTS_PATH}")
    print(f"Total time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
