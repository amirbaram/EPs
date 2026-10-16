"""
scripts/train_all_futures_wickless_models.py
=============================================
Complete Retraining Pipeline for all 5 Futures Tickers with Wickless Pivot Detections:
  - NQ=F (E-mini / Micro Nasdaq-100)
  - GC=F (Gold / Micro Gold)
  - CL=F (Crude Oil / Micro Crude Oil)
  - BTC=F (Bitcoin / Micro Bitcoin)
  - SI=F (Silver / Micro Silver)

Trains strictly on At-Extreme features (Zero Wick / Zero Close Rejection Leakage)
across 15m, 1h, and 4h timeframes, evaluates on 2-year Out-of-Sample test set,
and writes:
  - data/models/{prefix}_wickless_intraday_models.pkl
  - data/simulations/{prefix}_wickless_intraday_oos_results.json
  - Updates prob_pivot_high_extreme & prob_pivot_low_extreme in data/simulations/nq_intraday_scored/{prefix}_{tf}_scored.parquet
"""

from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nq_ml_engine

FUTURES_SPECS = [
    {
        "symbol": "NQ=F",
        "prefix": "nq",
        "loader": nq_ml_engine.load_nq_1m_base,
        "study_start": "2016-07-15 00:00:00",
        "train_cutoff": "2024-07-15 00:00:00",
        "cal_start": "2023-07-15 00:00:00",
        "oos_start": "2024-07-15 00:00:00",
        "oos_end": "2026-07-14 23:59:59",
    },
    {
        "symbol": "GC=F",
        "prefix": "gc",
        "loader": nq_ml_engine.load_gc_1m_base,
        "study_start": "2016-07-15 00:00:00",
        "train_cutoff": "2024-07-15 00:00:00",
        "cal_start": "2023-07-15 00:00:00",
        "oos_start": "2024-07-15 00:00:00",
        "oos_end": "2026-07-14 23:59:59",
    },
    {
        "symbol": "CL=F",
        "prefix": "cl",
        "loader": nq_ml_engine.load_cl_1m_base,
        "study_start": "2016-07-15 00:00:00",
        "train_cutoff": "2024-07-15 00:00:00",
        "cal_start": "2023-07-15 00:00:00",
        "oos_start": "2024-07-15 00:00:00",
        "oos_end": "2026-07-14 23:59:59",
    },
    {
        "symbol": "BTC=F",
        "prefix": "btc",
        "loader": nq_ml_engine.load_btc_1m_base,
        "study_start": "2017-08-17 00:00:00",
        "train_cutoff": "2024-07-01 00:00:00",
        "cal_start": "2023-07-01 00:00:00",
        "oos_start": "2024-07-01 00:00:00",
        "oos_end": "2026-06-30 23:59:59",
    },
    {
        "symbol": "SI=F",
        "prefix": "si",
        "loader": nq_ml_engine.load_si_1m_base,
        "study_start": "2021-08-01 18:00:00",
        "train_cutoff": "2024-07-15 00:00:00",
        "cal_start": "2023-07-15 00:00:00",
        "oos_start": "2024-07-15 00:00:00",
        "oos_end": "2026-07-14 23:59:59",
    },
]


def label_futures_major_reversals_vec(df: pd.DataFrame, tf: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Labels Major Macro Trend Reversal Tops (is_major_top) and Bottoms (is_major_bottom)
    using strictly causal anchor and zero-lookahead forward confirmation.
    """
    hc = df["high_cont"].to_numpy(dtype=np.float64)
    lc = df["low_cont"].to_numpy(dtype=np.float64)
    cc = df["close_cont"].to_numpy(dtype=np.float64)
    adr = np.maximum(1e-6, df["adr20_cont"].to_numpy(dtype=np.float64))
    n = len(df)

    if tf == "15m":
        w_macro, k_rev = 16, 24
        run_req, rev_req, sust_req = 2.6, 3.0, 2.0
    elif tf == "1h":
        w_macro, k_rev = 12, 20
        run_req, rev_req, sust_req = 2.6, 2.8, 1.8
    else:  # 4h
        w_macro, k_rev = 8, 16
        run_req, rev_req, sust_req = 2.6, 2.8, 1.8

    is_maj_top = np.zeros(n, dtype=int)
    is_maj_bot = np.zeros(n, dtype=int)

    valid_len = n - w_macro - k_rev
    if valid_len <= 0:
        return is_maj_top, is_maj_bot

    from numpy.lib.stride_tricks import sliding_window_view
    prior_hc_win = sliding_window_view(hc[:-k_rev], w_macro)[:-1]
    prior_lc_win = sliding_window_view(lc[:-k_rev], w_macro)[:-1]

    centers_h = hc[w_macro : n - k_rev]
    centers_l = lc[w_macro : n - k_rev]
    centers_adr = adr[w_macro : n - k_rev]

    fwd_hc_win = sliding_window_view(hc[w_macro + 1 :], k_rev)[:valid_len]
    fwd_lc_win = sliding_window_view(lc[w_macro + 1 :], k_rev)[:valid_len]
    fwd_close = cc[w_macro + k_rev : n]

    # TOP: Prior macro run up + deep subsequent reversal breakdown + not breached + sustained
    prior_low = prior_lc_win.min(axis=1)
    prior_max_h = prior_hc_win.max(axis=1)
    prior_run = (centers_h - prior_low) / centers_adr
    is_loc_hi = centers_h >= prior_max_h - 0.10 * centers_adr

    fwd_min_l = fwd_lc_win.min(axis=1)
    fwd_max_h = fwd_hc_win.max(axis=1)
    fwd_drop = (centers_h - fwd_min_l) / centers_adr
    top_not_breached = fwd_max_h <= centers_h + 0.20 * centers_adr
    top_sustained = (centers_h - fwd_close) / centers_adr >= sust_req

    top_cond = is_loc_hi & (prior_run >= run_req) & (fwd_drop >= rev_req) & top_not_breached & top_sustained
    is_maj_top[w_macro : n - k_rev] = top_cond.astype(int)

    # BOTTOM: Prior macro decline + deep subsequent turnaround rally + not breached + sustained
    prior_high = prior_hc_win.max(axis=1)
    prior_min_l = prior_lc_win.min(axis=1)
    prior_drop = (prior_high - centers_l) / centers_adr
    is_loc_lo = centers_l <= prior_min_l + 0.10 * centers_adr

    fwd_rally = (fwd_max_h - centers_l) / centers_adr
    bot_not_breached = fwd_min_l >= centers_l - 0.20 * centers_adr
    bot_sustained = (fwd_close - centers_l) / centers_adr >= sust_req

    bot_cond = is_loc_lo & (prior_drop >= run_req) & (fwd_rally >= rev_req) & bot_not_breached & bot_sustained
    is_maj_bot[w_macro : n - k_rev] = bot_cond.astype(int)

    return is_maj_top, is_maj_bot


def fit_wickless_classifier(
    df_cand: pd.DataFrame,
    label_col: str,
    feature_cols: List[str],
    name: str,
    cal_start: str,
    train_cutoff: str,
) -> Dict[str, Any]:
    """Fits LightGBM + XGBoost ensemble strictly on at-extreme features with isotonic calibration."""
    tr_core = df_cand[df_cand.index < pd.Timestamp(cal_start)]
    tr_cal = df_cand[(df_cand.index >= pd.Timestamp(cal_start)) & (df_cand.index < pd.Timestamp(train_cutoff))]
    tr_all = df_cand[df_cand.index < pd.Timestamp(train_cutoff)]

    y_core = tr_core[label_col].to_numpy(dtype=int)
    pos_rate = max(0.01, float(y_core.mean()))
    spw = float(np.clip((1.0 - pos_rate) / max(pos_rate, 1e-4) * 0.85, 2.0, 7.5))

    def _fit_pair(X_df: pd.DataFrame, y_arr: np.ndarray) -> Tuple[lgb.LGBMClassifier, XGBClassifier]:
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


def train_single_future(spec: Dict[str, Any]) -> Dict[str, Any]:
    sym = spec["symbol"]
    prefix = spec["prefix"]
    t_start = time.time()
    print("\n" + "=" * 75)
    print(f"TRAINING COMPLETE WICKLESS ML SUITE: {sym} (prefix: {prefix})")
    print("=" * 75)

    print(f"Loading continuous 1m dataset for {sym}...")
    df_1m = spec["loader"](start_date=spec["study_start"], include_live_tail=True)
    print(f"  Loaded {len(df_1m):,} 1m bars in {time.time()-t_start:.1f}s")

    all_models: Dict[str, Any] = {}
    report: Dict[str, Any] = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "symbol": sym,
        "prefix": prefix,
        "model_type": "wickless_intraday_at_extreme",
        "study_window": {
            "start": spec["study_start"],
            "train_cutoff": spec["train_cutoff"],
            "oos_start": spec["oos_start"],
            "oos_end": spec["oos_end"],
        },
        "timeframes": {},
    }

    tf_cfgs = {
        "15m": {"left": 6, "right": 6, "rev_atr": 1.6},
        "1h": {"left": 5, "right": 5, "rev_atr": 1.5},
        "4h": {"left": 4, "right": 4, "rev_atr": 1.4},
    }

    for tf, cfg in tf_cfgs.items():
        t0 = time.time()
        print(f"\n  [{sym} {tf}] Processing Wickless At-Extreme (left={cfg['left']}, right={cfg['right']}, rev={cfg['rev_atr']})...")
        bars = nq_ml_engine.resample_nq_with_1m_microstructure(df_1m, tf)
        feats = nq_ml_engine.compute_nq_intraday_features(bars, tf)

        if prefix == "nq":
            labeled = nq_ml_engine.label_nq_pivots_and_excursions(
                feats, pivot_left=cfg["left"], pivot_right=cfg["right"], rev_atr=cfg["rev_atr"]
            )
        else:
            labeled = nq_ml_engine._label_gc_pivots_and_excursions(
                feats, pivot_left=cfg["left"], pivot_right=cfg["right"], rev_atr=cfg["rev_atr"]
            )
        top_arr, bot_arr = label_futures_major_reversals_vec(feats, tf)
        labeled["is_major_top"] = top_arr
        labeled["is_major_bottom"] = bot_arr
        labeled = labeled.iloc[200:].copy()

        train_df = labeled[labeled.index < pd.Timestamp(spec["train_cutoff"])].copy()
        oos_df = labeled[
            (labeled.index >= pd.Timestamp(spec["oos_start"])) & (labeled.index <= pd.Timestamp(spec["oos_end"]))
        ].copy()

        high_cols = [c for c in nq_ml_engine.PIVOT_HIGH_AT_EXTREME_COLS if c in labeled.columns]
        low_cols = [c for c in nq_ml_engine.PIVOT_LOW_AT_EXTREME_COLS if c in labeled.columns]

        cand_hi_tr = train_df[train_df["is_cand_high"] & (~train_df["crosses_roll_8b"])].copy()
        cand_lo_tr = train_df[train_df["is_cand_low"] & (~train_df["crosses_roll_8b"])].copy()

        print(f"    Training Wickless Pivot High ({len(high_cols)} feats on {len(cand_hi_tr):,} cands)...")
        m_hi = fit_wickless_classifier(
            cand_hi_tr, "is_pivot_high", high_cols, f"{prefix.upper()}_{tf}_Wickless_High",
            spec["cal_start"], spec["train_cutoff"]
        )

        print(f"    Training Wickless Pivot Low ({len(low_cols)} feats on {len(cand_lo_tr):,} cands)...")
        m_lo = fit_wickless_classifier(
            cand_lo_tr, "is_pivot_low", low_cols, f"{prefix.upper()}_{tf}_Wickless_Low",
            spec["cal_start"], spec["train_cutoff"]
        )

        print(f"    Training Wickless Major Top ({len(high_cols)} feats on {len(cand_hi_tr):,} cands)...")
        m_maj_top = fit_wickless_classifier(
            cand_hi_tr, "is_major_top", high_cols, f"{prefix.upper()}_{tf}_Wickless_Major_Top",
            spec["cal_start"], spec["train_cutoff"]
        )

        print(f"    Training Wickless Major Bottom ({len(low_cols)} feats on {len(cand_lo_tr):,} cands)...")
        m_maj_bot = fit_wickless_classifier(
            cand_lo_tr, "is_major_bottom", low_cols, f"{prefix.upper()}_{tf}_Wickless_Major_Bottom",
            spec["cal_start"], spec["train_cutoff"]
        )

        all_models[tf] = {
            "pivot_high": m_hi,
            "pivot_low": m_lo,
            "major_top": m_maj_top,
            "major_bottom": m_maj_bot,
            "feature_cols_high": high_cols,
            "feature_cols_low": low_cols,
            "feature_cols_major_top": high_cols,
            "feature_cols_major_bottom": low_cols,
            "config": cfg,
        }

        # Evaluate OOS metrics
        oos_hi = oos_df[oos_df["is_cand_high"] & (~oos_df["crosses_roll_8b"])].copy()
        oos_lo = oos_df[oos_df["is_cand_low"] & (~oos_df["crosses_roll_8b"])].copy()

        X_hi = oos_hi[high_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        p_hi = 0.55 * m_hi["lgb"].predict_proba(X_hi)[:, 1] + 0.45 * m_hi["xgb"].predict_proba(X_hi)[:, 1]
        y_hi = oos_hi["is_pivot_high"].to_numpy(dtype=int)

        X_lo = oos_lo[low_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        p_lo = 0.55 * m_lo["lgb"].predict_proba(X_lo)[:, 1] + 0.45 * m_lo["xgb"].predict_proba(X_lo)[:, 1]
        y_lo = oos_lo["is_pivot_low"].to_numpy(dtype=int)

        p_mt = 0.55 * m_maj_top["lgb"].predict_proba(X_hi)[:, 1] + 0.45 * m_maj_top["xgb"].predict_proba(X_hi)[:, 1]
        y_mt = oos_hi["is_major_top"].to_numpy(dtype=int)

        p_mb = 0.55 * m_maj_bot["lgb"].predict_proba(X_lo)[:, 1] + 0.45 * m_maj_bot["xgb"].predict_proba(X_lo)[:, 1]
        y_mb = oos_lo["is_major_bottom"].to_numpy(dtype=int)

        auc_hi = float(roc_auc_score(y_hi, p_hi)) if len(np.unique(y_hi)) > 1 else 0.5
        pr_hi = float(average_precision_score(y_hi, p_hi)) if len(np.unique(y_hi)) > 1 else 0.0
        auc_lo = float(roc_auc_score(y_lo, p_lo)) if len(np.unique(y_lo)) > 1 else 0.5
        pr_lo = float(average_precision_score(y_lo, p_lo)) if len(np.unique(y_lo)) > 1 else 0.0

        auc_mt = float(roc_auc_score(y_mt, p_mt)) if len(np.unique(y_mt)) > 1 else 0.5
        pr_mt = float(average_precision_score(y_mt, p_mt)) if len(np.unique(y_mt)) > 1 else 0.0
        auc_mb = float(roc_auc_score(y_mb, p_mb)) if len(np.unique(y_mb)) > 1 else 0.5
        pr_mb = float(average_precision_score(y_mb, p_mb)) if len(np.unique(y_mb)) > 1 else 0.0

        print(f"    [{sym} {tf}] OOS High AUC: {auc_hi:.4f} | Low AUC: {auc_lo:.4f} | MajTop AUC: {auc_mt:.4f} | MajBot AUC: {auc_mb:.4f} in {time.time()-t0:.1f}s")
        report["timeframes"][tf] = {
            "high": {"auc": auc_hi, "pr_auc": pr_hi, "top_features": m_hi["top_features"][:6]},
            "low": {"auc": auc_lo, "pr_auc": pr_lo, "top_features": m_lo["top_features"][:6]},
            "major_top": {"auc": auc_mt, "pr_auc": pr_mt, "top_features": m_maj_top["top_features"][:6]},
            "major_bottom": {"auc": auc_mb, "pr_auc": pr_mb, "top_features": m_maj_bot["top_features"][:6]},
        }

        # Score full labeled dataset with the new models to update scored cache
        X_all_hi = labeled[high_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_all_lo = labeled[low_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        p_all_hi_raw = 0.55 * m_hi["lgb"].predict_proba(X_all_hi)[:, 1] + 0.45 * m_hi["xgb"].predict_proba(X_all_hi)[:, 1]
        p_all_lo_raw = 0.55 * m_lo["lgb"].predict_proba(X_all_lo)[:, 1] + 0.45 * m_lo["xgb"].predict_proba(X_all_lo)[:, 1]
        p_all_hi = np.clip(0.70 * p_all_hi_raw + 0.30 * m_hi["iso"].predict(p_all_hi_raw), 0.0, 0.98)
        p_all_lo = np.clip(0.70 * p_all_lo_raw + 0.30 * m_lo["iso"].predict(p_all_lo_raw), 0.0, 0.98)

        p_all_mt_raw = 0.55 * m_maj_top["lgb"].predict_proba(X_all_hi)[:, 1] + 0.45 * m_maj_top["xgb"].predict_proba(X_all_hi)[:, 1]
        p_all_mb_raw = 0.55 * m_maj_bot["lgb"].predict_proba(X_all_lo)[:, 1] + 0.45 * m_maj_bot["xgb"].predict_proba(X_all_lo)[:, 1]
        p_all_mt = np.clip(0.70 * p_all_mt_raw + 0.30 * m_maj_top["iso"].predict(p_all_mt_raw), 0.0, 0.98)
        p_all_mb = np.clip(0.70 * p_all_mb_raw + 0.30 * m_maj_bot["iso"].predict(p_all_mb_raw), 0.0, 0.98)

        mask_hi = labeled["is_cand_high"].to_numpy(dtype=bool)
        mask_lo = labeled["is_cand_low"].to_numpy(dtype=bool)
        labeled["prob_pivot_high_extreme"] = np.where(mask_hi, p_all_hi, 0.0)
        labeled["prob_pivot_low_extreme"] = np.where(mask_lo, p_all_lo, 0.0)
        labeled["prob_major_top_extreme"] = np.where(mask_hi, p_all_mt, 0.0)
        labeled["prob_major_bottom_extreme"] = np.where(mask_lo, p_all_mb, 0.0)
        labeled["prob_maj_ut_top_extreme"] = labeled["prob_major_top_extreme"]
        labeled["prob_maj_dt_low_extreme"] = labeled["prob_major_bottom_extreme"]

        # Update the stored scored cache parquet
        cached_p = nq_ml_engine.SCORED_CACHE_DIR / f"{prefix}_{tf}_scored.parquet"
        if cached_p.exists():
            try:
                cur_df = pd.read_parquet(cached_p)
                idx_common = cur_df.index.intersection(labeled.index)
                cur_df.loc[idx_common, "prob_pivot_high_extreme"] = labeled.loc[idx_common, "prob_pivot_high_extreme"].astype("float32")
                cur_df.loc[idx_common, "prob_pivot_low_extreme"] = labeled.loc[idx_common, "prob_pivot_low_extreme"].astype("float32")
                cur_df.loc[idx_common, "prob_major_top_extreme"] = labeled.loc[idx_common, "prob_major_top_extreme"].astype("float32")
                cur_df.loc[idx_common, "prob_major_bottom_extreme"] = labeled.loc[idx_common, "prob_major_bottom_extreme"].astype("float32")
                cur_df.loc[idx_common, "prob_maj_ut_top_extreme"] = labeled.loc[idx_common, "prob_maj_ut_top_extreme"].astype("float32")
                cur_df.loc[idx_common, "prob_maj_dt_low_extreme"] = labeled.loc[idx_common, "prob_maj_dt_low_extreme"].astype("float32")
                nq_ml_engine._atomic_to_parquet(cur_df, cached_p)
                print(f"    Updated prob_pivot_... and prob_major_... extreme in {cached_p.name}")
            except Exception as e:
                print(f"    Notice: could not update existing cache {cached_p.name}: {e}")

    out_models_p = ROOT / "data" / "models" / f"{prefix}_wickless_intraday_models.pkl"
    out_results_p = ROOT / "data" / "simulations" / f"{prefix}_wickless_intraday_oos_results.json"
    out_models_p.parent.mkdir(parents=True, exist_ok=True)
    out_results_p.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(all_models, out_models_p)
    out_results_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  ✅ Saved {sym} Wickless models ({out_models_p.name}) & OOS Report in {time.time()-t_start:.1f}s")
    return report


def main() -> None:
    t_global = time.time()
    print("=" * 80)
    print("STARTING COMPLETE RETRAINING FOR ALL 5 FUTURES TICKERS (WICKLESS MODELS)")
    print("  Tickers: NQ=F, GC=F, CL=F, BTC=F, SI=F across 15m, 1h, and 4h")
    print("=" * 80)

    summary = {}
    for spec in FUTURES_SPECS:
        summary[spec["symbol"]] = train_single_future(spec)

    print("\n" + "=" * 80)
    print(f"✅ COMPLETE RETRAINING FINISHED FOR ALL 5 FUTURES IN {time.time() - t_global:.1f}s")
    for sym, res in summary.items():
        tfs = res.get("timeframes", {})
        line = " | ".join(
            f"{tf}: H={v['high']['auc']:.3f} L={v['low']['auc']:.3f} MajT={v.get('major_top', {}).get('auc', 0.0):.3f} MajB={v.get('major_bottom', {}).get('auc', 0.0):.3f}"
            for tf, v in tfs.items()
        )
        print(f"  • {sym:6s} -> {line}")
    print("=" * 80)


if __name__ == "__main__":
    main()
