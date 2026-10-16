"""
Empirical Comparison: MNQ 15m Pivot Models With vs. Without Wick Features.

Compares:
1. Model WITH Wicks (Full-Bar: upper/lower wick fraction, fade from high/bounce from low, close position, 1m order flow)
2. Model WITHOUT Wicks (Pure At-Extreme: RSI exhaustion, MA extensions, multi-wave divergence, session context, overnight sweeps)
3. Intrabar Execution Simulation: Evaluates the price and R:R edge of entering intrabar at the extreme vs. waiting for the 15m bar close.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, brier_score_loss

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nq_ml_engine

STUDY_START = "2021-08-01 18:00:00"
TRAIN_CUTOFF = "2024-07-15 00:00:00"
CAL_START = "2023-07-15 00:00:00"
OOS_START = "2024-07-15 00:00:00"
OOS_END = "2026-07-14 23:59:59"


def fit_model_bundle(
    df_cand: pd.DataFrame,
    label_col: str,
    feature_cols: List[str],
    model_name: str,
) -> Dict[str, Any]:
    """Fits LightGBM + XGBoost ensemble with chronological Isotonic calibration."""
    tr_core = df_cand[df_cand.index < pd.Timestamp(CAL_START)]
    tr_cal = df_cand[(df_cand.index >= pd.Timestamp(CAL_START)) & (df_cand.index < pd.Timestamp(TRAIN_CUTOFF))]
    tr_all = df_cand[df_cand.index < pd.Timestamp(TRAIN_CUTOFF)]

    y_core = tr_core[label_col].to_numpy(dtype=int)
    pos_rate = max(0.01, float(y_core.mean()))
    # Scale positive class so raw probabilities span full 0.10 to 0.85 range (like daily stock models)
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

    # Refit on all training data
    y_all = tr_all[label_col].to_numpy(dtype=int)
    lgb_all, xgb_all = _fit_pair(tr_all[feature_cols], y_all)

    # Feature importances
    imp = pd.Series(lgb_all.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top_features = [{"feature": k, "importance": int(v)} for k, v in imp.head(10).items()]

    return {
        "name": model_name,
        "lgb": lgb_all,
        "xgb": xgb_all,
        "iso": iso,
        "feature_cols": feature_cols,
        "top_features": top_features,
    }


def score_bundle(df: pd.DataFrame, bundle: Dict[str, Any], cand_mask_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """Scores a model bundle on a dataframe, returning (p_raw, p_iso)."""
    cols = bundle["feature_cols"]
    X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    p_raw = 0.55 * bundle["lgb"].predict_proba(X)[:, 1] + 0.45 * bundle["xgb"].predict_proba(X)[:, 1]
    p_iso = bundle["iso"].predict(p_raw)
    
    if cand_mask_col in df.columns:
        m = df[cand_mask_col].to_numpy(dtype=bool)
        p_raw = np.where(m, p_raw, 0.0)
        p_iso = np.where(m, p_iso, 0.0)
    return p_raw, p_iso


def run_comparison():
    print("=" * 80)
    print("EMPIRICAL COMPARISON: MNQ 15M PIVOT MODELS WITH VS. WITHOUT WICK FEATURES")
    print("=" * 80)

    print("\n1. Loading and preparing NQ 15m feature dataset from 1-minute base...")
    t0 = time.time()
    df_1m = nq_ml_engine.load_nq_1m_base(start_date=STUDY_START, include_live_tail=True)
    b15 = nq_ml_engine.resample_nq_with_1m_microstructure(df_1m, "15m")
    f15 = nq_ml_engine.compute_nq_intraday_features(b15, "15m")
    labeled = nq_ml_engine.label_nq_pivots_and_excursions(f15, pivot_left=6, pivot_right=6, rev_atr=1.6)
    labeled = labeled.iloc[200:].copy()  # warmup trim
    print(f"   Prepared {len(labeled):,} 15m bars ({labeled.index.min()} -> {labeled.index.max()}) in {time.time()-t0:.2f}s")

    train_df = labeled[labeled.index < pd.Timestamp(TRAIN_CUTOFF)].copy()
    oos_df = labeled[(labeled.index >= pd.Timestamp(OOS_START)) & (labeled.index <= pd.Timestamp(OOS_END))].copy()
    print(f"   Train window: < {TRAIN_CUTOFF} ({len(train_df):,} bars)")
    print(f"   OOS test window: {OOS_START} -> {OOS_END} ({len(oos_df):,} bars)")

    # Define Feature Sets
    # -------------------------------------------------------------
    # WITHOUT WICKS: Strictly observable the instant price reaches the extreme
    high_cols_no_wick = [c for c in nq_ml_engine.PIVOT_HIGH_AT_EXTREME_COLS if c in labeled.columns]
    low_cols_no_wick = [c for c in nq_ml_engine.PIVOT_LOW_AT_EXTREME_COLS if c in labeled.columns]

    # WITH WICKS: Full-bar features including wicks, close position, fade, and 1m absorption
    high_cols_with_wick = [c for c in nq_ml_engine.PIVOT_HIGH_FULL_BAR_COLS if c in labeled.columns]
    low_cols_with_wick = [c for c in nq_ml_engine.PIVOT_LOW_FULL_BAR_COLS if c in labeled.columns]

    wick_features_high = [c for c in high_cols_with_wick if c not in high_cols_no_wick]
    wick_features_low = [c for c in low_cols_with_wick if c not in low_cols_no_wick]

    print("\n2. Feature Set Audit:")
    print(f"   Wickless High features: {len(high_cols_no_wick)} cols (RSI stretch, MA extensions, divergences, session VWAPs, ON sweeps)")
    print(f"   With-Wick High features: {len(high_cols_with_wick)} cols (+{len(wick_features_high)} wick/close/order-flow cols)")
    print(f"   Wick/Close features included in With-Wick model: {wick_features_high}")

    # 3. Train Pivot High Models
    # -------------------------------------------------------------
    print("\n3. Training Pivot High Models (Candidate Highs)...")
    cand_hi_tr = train_df[train_df["is_cand_high"] & (~train_df["crosses_roll_8b"])].copy()
    print(f"   Training candidate high bars: {len(cand_hi_tr):,} (True pivots: {cand_hi_tr['is_pivot_high'].sum():,})")

    t0 = time.time()
    model_hi_nowick = fit_model_bundle(cand_hi_tr, "is_pivot_high", high_cols_no_wick, "Pivot High (WITHOUT Wick)")
    print(f"   ✓ Trained Wickless Pivot High in {time.time()-t0:.1f}s")

    t0 = time.time()
    model_hi_wick = fit_model_bundle(cand_hi_tr, "is_pivot_high", high_cols_with_wick, "Pivot High (WITH Wick)")
    print(f"   ✓ Trained With-Wick Pivot High in {time.time()-t0:.1f}s")

    # 4. Train Pivot Low Models
    # -------------------------------------------------------------
    print("\n4. Training Pivot Low Models (Candidate Lows)...")
    cand_lo_tr = train_df[train_df["is_cand_low"] & (~train_df["crosses_roll_8b"])].copy()
    print(f"   Training candidate low bars: {len(cand_lo_tr):,} (True pivots: {cand_lo_tr['is_pivot_low'].sum():,})")

    t0 = time.time()
    model_lo_nowick = fit_model_bundle(cand_lo_tr, "is_pivot_low", low_cols_no_wick, "Pivot Low (WITHOUT Wick)")
    print(f"   ✓ Trained Wickless Pivot Low in {time.time()-t0:.1f}s")

    t0 = time.time()
    model_lo_wick = fit_model_bundle(cand_lo_tr, "is_pivot_low", low_cols_with_wick, "Pivot Low (WITH Wick)")
    print(f"   ✓ Trained With-Wick Pivot Low in {time.time()-t0:.1f}s")

    # 5. Score Out-of-Sample Test Set
    # -------------------------------------------------------------
    print("\n5. Scoring Out-of-Sample (OOS) Test Set (2024-2026)...")
    oos_cand_hi = oos_df[oos_df["is_cand_high"] & (~oos_df["crosses_roll_8b"])].copy()
    oos_cand_lo = oos_df[oos_df["is_cand_low"] & (~oos_df["crosses_roll_8b"])].copy()

    # Scores at Bar Close (full bar info present)
    p_hi_nowick_raw, p_hi_nowick_iso = score_bundle(oos_cand_hi, model_hi_nowick, "is_cand_high")
    p_hi_wick_raw, p_hi_wick_iso = score_bundle(oos_cand_hi, model_hi_wick, "is_cand_high")

    p_lo_nowick_raw, p_lo_nowick_iso = score_bundle(oos_cand_lo, model_lo_nowick, "is_cand_low")
    p_lo_wick_raw, p_lo_wick_iso = score_bundle(oos_cand_lo, model_lo_wick, "is_cand_low")

    y_hi = oos_cand_hi["is_pivot_high"].to_numpy(dtype=int)
    y_lo = oos_cand_lo["is_pivot_low"].to_numpy(dtype=int)

    base_hi = y_hi.mean() * 100.0
    base_lo = y_lo.mean() * 100.0

    print(f"\n   OOS Candidate Highs: {len(oos_cand_hi):,} (Base Pivot Rate: {base_hi:.2f}%)")
    print(f"   OOS Candidate Lows:  {len(oos_cand_lo):,} (Base Pivot Rate: {base_lo:.2f}%)")

    # 6. Comparative Evaluation at Bar Close
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("A. EVALUATION AT 15M BAR CLOSE (When Wick is Fully Formed)")
    print("=" * 80)

    def print_metrics(name: str, y_true: np.ndarray, p_pred: np.ndarray, base_rate: float):
        auc = roc_auc_score(y_true, p_pred)
        pr_auc = average_precision_score(y_true, p_pred)
        brier = brier_score_loss(y_true, p_pred)
        print(f"\n   --- {name} ---")
        print(f"   ROC-AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f} (Base: {base_rate/100:.4f}) | Brier Score: {brier:.4f}")
        print(f"   {'Threshold':<12} | {'Signals':<8} | {'Precision':<10} | {'Recall':<8} | {'Lift':<6}")
        print(f"   {'-'*12}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*6}")
        for th in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
            mask = (p_pred >= th)
            n_sig = int(mask.sum())
            if n_sig > 0:
                prec = float(precision_score(y_true, mask, zero_division=0)) * 100.0
                rec = float(recall_score(y_true, mask, zero_division=0)) * 100.0
                lift = prec / base_rate
                print(f"   >= {int(th*100):2d}%        | {n_sig:<8} | {prec:6.2f}%    | {rec:5.2f}%  | {lift:4.2f}x")
            else:
                print(f"   >= {int(th*100):2d}%        | 0        | N/A        | 0.00%   | N/A")

    print("\n[PIVOT HIGH CLASSIFIERS - AT BAR CLOSE (RAW SCORE)]")
    print_metrics("PIVOT HIGH: WITHOUT WICK (Pure At-Extreme)", y_hi, p_hi_nowick_raw, base_hi)
    print_metrics("PIVOT HIGH: WITH WICK (Full-Bar + Wick)", y_hi, p_hi_wick_raw, base_hi)

    print("\n[PIVOT LOW CLASSIFIERS - AT BAR CLOSE (RAW SCORE)]")
    print_metrics("PIVOT LOW: WITHOUT WICK (Pure At-Extreme)", y_lo, p_lo_nowick_raw, base_lo)
    print_metrics("PIVOT LOW: WITH WICK (Full-Bar + Wick)", y_lo, p_lo_wick_raw, base_lo)

    # 7. INTRABAR SIMULATION: What happens at the exact moment of extreme touch?
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("B. INTRABAR SIMULATION: AT-THE-TOUCH EVALUATION (Before Wick Exists)")
    print("=" * 80)
    print("   At the moment price touches a new high intrabar:")
    print("   - Price is at High[t], Close[t] == High[t]")
    print("   - upper_wick_frac == 0.0, fade_from_high_atr == 0.0, close_pos_in_bar == 1.0")
    print("   - A model trained WITH wicks will see 0 wick and 0 fade, so its probability drops!")

    # Simulate intrabar touch features on oos_cand_hi
    df_touch_hi = oos_cand_hi.copy()
    df_touch_hi["upper_wick_frac"] = 0.0
    df_touch_hi["fade_from_high_atr"] = 0.0
    df_touch_hi["close_pos_in_bar"] = 1.0
    df_touch_hi["lower_wick_frac"] = 0.0
    df_touch_hi["m1_vol_top25_frac"] = 0.25
    df_touch_hi["m1_high_time_frac"] = 1.0

    # Score both models under intrabar touch conditions
    p_touch_hi_nowick_raw, _ = score_bundle(df_touch_hi, model_hi_nowick, "is_cand_high")
    p_touch_hi_wick_raw, _ = score_bundle(df_touch_hi, model_hi_wick, "is_cand_high")

    print("\n   [PIVOT HIGH EVALUATION AT INTRABAR TOUCH (RAW SCORE)]")
    print_metrics("WICKLESS MODEL AT TOUCH (Immune to missing wick)", y_hi, p_touch_hi_nowick_raw, base_hi)
    print_metrics("WITH-WICK MODEL AT TOUCH (Degraded by missing wick)", y_hi, p_touch_hi_wick_raw, base_hi)

    # 8. EXECUTION EDGE & THE 'WICK TAX' (Price Slippage Analysis)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("C. EXECUTION SLIPPAGE & THE 'WICK TAX' ANALYSIS")
    print("=" * 80)
    
    # Calculate the exact distance between the high/low and the 15m close
    true_piv_hi = oos_cand_hi[oos_cand_hi["is_pivot_high"].to_numpy(dtype=bool)].copy()
    wick_pts_hi = (true_piv_hi["high"] - true_piv_hi["close"]).to_numpy()
    atr_hi = true_piv_hi["atr14_pts"].to_numpy()
    wick_atr_hi = wick_pts_hi / np.maximum(atr_hi, 1e-4)

    print(f"\n   True Pivot Highs in OOS: {len(true_piv_hi):,} events")
    print(f"   Average Upper Wick on True Pivot Highs: {np.mean(wick_pts_hi):.2f} NQ points ({np.mean(wick_atr_hi):.2f}x ATR)")
    print(f"   Median Upper Wick on True Pivot Highs:  {np.median(wick_pts_hi):.2f} NQ points")
    print(f"   75th %ile Upper Wick:                   {np.percentile(wick_pts_hi, 75):.2f} NQ points")
    print(f"   Dollar Value per MNQ contract ($2/pt):  ${np.mean(wick_pts_hi)*2.0:.2f} average slippage per trade")
    print(f"   Dollar Value per NQ contract ($20/pt):  ${np.mean(wick_pts_hi)*20.0:.2f} average slippage per trade")

    print("\n   CRITICAL EXECUTION COMPARISON:")
    print(f"   If you wait for the 15m bar to CLOSE to confirm the wick:")
    print(f"   -> You enter SHORT on average {np.mean(wick_pts_hi):.2f} points LOWER than the high!")
    print(f"   -> Your stop must be above the high, so your risk buffer is widened by +{np.mean(wick_pts_hi):.2f} points!")
    print(f"   If you enter INTRABAR at the touch using the Wickless model:")
    print(f"   -> You enter right at or within 1-2 points of the high, capturing the entire {np.mean(wick_pts_hi):.2f} pt wick as profit!")

    # 9. FORWARD MFE & MAE TRADE OUTCOMES (Entering Intrabar vs Bar Close)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("D. FORWARD TRADE OUTCOMES (Next 8 Bars): INTRABAR TOUCH VS BAR CLOSE ENTRY")
    print("=" * 80)

    # Threshold >= 55%
    mask_sig_hi = (p_touch_hi_nowick_raw >= 0.55)
    sig_hi_df = oos_cand_hi[mask_sig_hi].copy()

    wick_val_pts = (sig_hi_df["high"] - sig_hi_df["close"]).to_numpy()
    atr_val = sig_hi_df["atr14_pts"].to_numpy()

    mfe_close_pts = sig_hi_df["mfe_short_8b_atr"].to_numpy() * atr_val
    mae_close_pts = sig_hi_df["mae_short_8b_atr"].to_numpy() * atr_val

    mfe_intrabar_pts = mfe_close_pts + wick_val_pts
    mae_intrabar_pts = np.maximum(0.0, mae_close_pts - wick_val_pts)

    print(f"\n   Signals analyzed (Wickless Pivot High Prob >= 55%): {len(sig_hi_df):,} trades")
    print(f"\n   {'Metric':<36} | {'Intrabar Entry (No-Wick)':<25} | {'Bar Close Entry (With-Wick)'}")
    print(f"   {'-'*36}-+-{'-'*25}-+-{'-'*28}")
    print(f"   {'Average Realized MFE (Target)':<36} | {np.mean(mfe_intrabar_pts):8.2f} pts              | {np.mean(mfe_close_pts):8.2f} pts")
    print(f"   {'Average Realized MAE (Adverse)':<36} | {np.mean(mae_intrabar_pts):8.2f} pts              | {np.mean(mae_close_pts):8.2f} pts")
    print(f"   {'Realized MFE / MAE Ratio':<36} | {np.mean(mfe_intrabar_pts)/max(0.1, np.mean(mae_intrabar_pts)):8.2f}x                | {np.mean(mfe_close_pts)/max(0.1, np.mean(mae_close_pts)):8.2f}x")
    print(f"   {'Favorable Win Rate (MFE >= 2x MAE)':<36} | {(mfe_intrabar_pts >= 2.0*mae_intrabar_pts).mean()*100:7.2f}%               | {(mfe_close_pts >= 2.0*mae_close_pts).mean()*100:7.2f}%")
    print(f"   {'Net Edge (MFE - MAE)':<36} | {np.mean(mfe_intrabar_pts - mae_intrabar_pts):+8.2f} pts              | {np.mean(mfe_close_pts - mae_close_pts):+8.2f} pts")
    print(f"   {'Dollar Net Edge / MNQ Contract':<36} | ${np.mean(mfe_intrabar_pts - mae_intrabar_pts)*2.0:+8.2f}                 | ${np.mean(mfe_close_pts - mae_close_pts)*2.0:+8.2f}")
    print(f"   {'Dollar Net Edge / NQ Contract':<36} | ${np.mean(mfe_intrabar_pts - mae_intrabar_pts)*20.0:+8.2f}                 | ${np.mean(mfe_close_pts - mae_close_pts)*20.0:+8.2f}")

    # Top features comparison
    print("\n" + "=" * 80)
    print("E. TOP FEATURE IMPORTANCE COMPARISON")
    print("=" * 80)
    print("\n   Top Features in WICKLESS Model (Identifies extreme before wick):")
    for i, it in enumerate(model_hi_nowick["top_features"][:8], 1):
        print(f"     {i}. {it['feature']:<30} (Importance: {it['importance']})")

    print("\n   Top Features in WITH-WICK Model (Relies on formed wick):")
    for i, it in enumerate(model_hi_wick["top_features"][:8], 1):
        print(f"     {i}. {it['feature']:<30} (Importance: {it['importance']})")


if __name__ == "__main__":
    run_comparison()
