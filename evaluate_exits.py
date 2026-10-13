"""Evaluate Advanced ML Exit Models:
1. Model 1: 1-Year Climax Exit Exhaustion Classifier (ep_exhaustion_classifier.pkl)
   - Evaluated on active runner setups with confirmed gain.
   - Evaluates ROC-AUC, Accuracy (threshold >= 0.50), Precision, Recall, F1.
2. Model 2: Dynamic Trailing Stop Quantile Regressor (ep_dynamic_trailer.pkl)
   - Evaluated on runner drawdowns using 22 TrendLab confirmed swing, parabolic, and volume features.
   - Evaluates MAE, Quantile (Pinball) Loss (alpha=0.30), Empirical Coverage, and Quantile distribution.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

import datastore
import indicators
from ep_ml_engine import engine

DEFAULT_CACHE_PATH = Path("data/ml_datasets/amir_rolling/dataset_rolling_exit_features.parquet")
ROLLING_DATASET_PATH = Path("data/ml_datasets/amir_rolling/dataset_rolling.parquet")
EXH_MODEL_PATH = Path("data/models/ep_exhaustion_classifier.pkl")
STOP_MODEL_PATH = Path("data/models/ep_dynamic_trailer.pkl")


def extract_exit_features_dataset(
    min_ret: float = 0.20,
    max_samples: int | None = None,
    out_path: Path | None = DEFAULT_CACHE_PATH,
) -> pd.DataFrame:
    """Extracts the 22 TrendLab swing, parabolic, and volume features for candidate runner rows."""
    if not ROLLING_DATASET_PATH.exists():
        raise FileNotFoundError(f"Source rolling dataset not found: {ROLLING_DATASET_PATH}")

    print(f"[*] Loading rolling events from {ROLLING_DATASET_PATH}...")
    df = pd.read_parquet(ROLLING_DATASET_PATH)
    candidates = df[df["feature_ret_since_ep"] >= min_ret].copy()
    if max_samples:
        candidates = candidates.head(max_samples)

    print(f"[*] Extracting 22 TrendLab exit features across {len(candidates)} runner rows...")
    t0 = time.time()

    cached_bars: dict[str, pd.DataFrame | None] = {}
    for sym in candidates["symbol"].unique():
        b_df = datastore.load_bars(sym)
        if b_df is not None:
            b_df = indicators.add_indicators(b_df)
            b_df["ema10"] = b_df["close"].ewm(span=10, adjust=False).mean()
            b_df["ema20"] = b_df["close"].ewm(span=20, adjust=False).mean()
            b_df["ema50"] = b_df["close"].ewm(span=50, adjust=False).mean()
            cached_bars[sym] = b_df

    pipe_exh = joblib.load(EXH_MODEL_PATH)
    cols = list(pipe_exh.feature_names_in_)

    feat_rows = []
    for idx, row in candidates.iterrows():
        sym = row["symbol"]
        b_df = cached_bars.get(sym)
        if b_df is None:
            continue
        ts_entry = pd.Timestamp(row["entry_date"])
        ts_eval = pd.Timestamp(row["date"])
        if ts_entry not in b_df.index or ts_eval not in b_df.index:
            continue
        d1_idx = b_df.index.get_loc(ts_entry)
        t_idx = b_df.index.get_loc(ts_eval)
        if isinstance(d1_idx, (slice, np.ndarray)):
            d1_idx = d1_idx[0] if isinstance(d1_idx, np.ndarray) else d1_idx.start
        if isinstance(t_idx, (slice, np.ndarray)):
            t_idx = t_idx[0] if isinstance(t_idx, np.ndarray) else t_idx.start

        f = engine.compute_rolling_features(sym, d1_idx, t_idx, df=b_df)
        if f is not None:
            f_dict = {c: f.get(c, 0.0) for c in cols}
            f_dict["symbol"] = sym
            f_dict["date"] = row["date"]
            f_dict["entry_date"] = row["entry_date"]
            f_dict["label_exhaustion"] = row.get("label_exhaustion", 0)
            f_dict["forward_mae_pct"] = row.get("forward_mae_pct", np.nan)
            f_dict["target_50"] = row.get("target_50", np.nan)
            f_dict["feature_ret_since_ep"] = row["feature_ret_since_ep"]
            feat_rows.append(f_dict)

    out_df = pd.DataFrame(feat_rows)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(out_path, index=False)
        print(f"[*] Cached {len(out_df)} exit feature rows to {out_path} ({time.time() - t0:.1f}s)")
    return out_df


def load_or_extract_features(
    recompute: bool = False,
    cache_path: Path = DEFAULT_CACHE_PATH,
    min_ret: float = 0.20,
    max_samples: int | None = None,
) -> pd.DataFrame:
    """Loads 22 exit features from parquet cache if present, otherwise computes them."""
    if not recompute and cache_path.exists():
        print(f"[*] Loading precomputed 22 exit features from cache: {cache_path}")
        df = pd.read_parquet(cache_path)
        if max_samples:
            df = df.head(max_samples)
        return df
    return extract_exit_features_dataset(min_ret=min_ret, max_samples=max_samples, out_path=cache_path)


def evaluate(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Evaluate Advanced ML Exit Models (Exhaustion & Dynamic Trailer)")
    parser.add_argument("--recompute", action="store_true", help="Force recomputation of the 22 exit features from bar datastore")
    parser.add_argument("--cache-path", type=str, default=str(DEFAULT_CACHE_PATH), help="Path to parquet exit features cache")
    parser.add_argument("--min-ret", type=float, default=0.20, help="Minimum return threshold for runner evaluation (default: 0.20)")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap evaluation on maximum N samples")
    args = parser.parse_args(argv)

    cache_path = Path(args.cache_path)
    df = load_or_extract_features(
        recompute=args.recompute,
        cache_path=cache_path,
        min_ret=args.min_ret,
        max_samples=args.max_samples,
    )

    if not EXH_MODEL_PATH.exists() or not STOP_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model files in data/models: {EXH_MODEL_PATH} / {STOP_MODEL_PATH}")

    pipe_exh = joblib.load(EXH_MODEL_PATH)
    pipe_stop = joblib.load(STOP_MODEL_PATH)
    feature_cols = list(pipe_exh.feature_names_in_)

    metrics_result: dict = {"exhaustion": {}, "dynamic_trailer": {}}

    print("\n" + "=" * 80)
    print("🎯 ADVANCED ML EXITS EVALUATION REPORT")
    print(f"Features: 22 TrendLab confirmed swing, parabolic, and volume features")
    print(f"Dataset:  {len(df):,} runner observations evaluated")
    print("=" * 80)

    if len(df) == 0:
        print("[-] No runner observations found matching criteria.")
        return metrics_result

    # -------------------------------------------------------------------------
    # 1. EVALUATION: CLIMAX EXHAUSTION CLASSIFIER
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("📊 1. MODEL 1: 1-YEAR CLIMAX EXIT CLASSIFIER (ep_exhaustion_classifier.pkl)")
    print("-" * 80)

    for ret_thresh in [0.20, 0.30]:
        df_sub = df[df["feature_ret_since_ep"] >= ret_thresh].copy()
        if len(df_sub) == 0:
            print(f"Cohort: Unrealized Gain >= {ret_thresh * 100:.0f}% (No observations in sample)\n")
            continue
        if "label_exhaustion" not in df_sub.columns:
            print(f"Cohort: Unrealized Gain >= {ret_thresh * 100:.0f}% (N = {len(df_sub):,}) - 'label_exhaustion' column not present\n")
            continue

        X_exh = df_sub[feature_cols]
        y_exh = df_sub["label_exhaustion"].to_numpy().astype(int)
        probs = pipe_exh.predict_proba(X_exh)[:, 1]
        preds = (probs >= 0.50).astype(int)

        n_pos = int(y_exh.sum())
        n_neg = len(y_exh) - n_pos
        pct_pos = (n_pos / len(df_sub) * 100.0) if len(df_sub) > 0 else 0.0

        has_both_classes = df_sub["label_exhaustion"].nunique() > 1
        if has_both_classes:
            auc = roc_auc_score(y_exh, probs)
            bal_acc = balanced_accuracy_score(y_exh, preds)
            auc_str = f"{auc:.3f}"
            bal_acc_str = f"{bal_acc * 100:.1f}%"
        else:
            auc = float("nan")
            bal_acc = float("nan")
            auc_str = "N/A (single class in sample)"
            bal_acc_str = "N/A"

        acc = accuracy_score(y_exh, preds)
        prec = precision_score(y_exh, preds, zero_division=0)
        rec = recall_score(y_exh, preds, zero_division=0)
        f1 = f1_score(y_exh, preds, zero_division=0)

        mean_pos = f"{probs[y_exh == 1].mean():.3f}" if n_pos > 0 else "N/A"
        mean_neg = f"{probs[y_exh == 0].mean():.3f}" if n_neg > 0 else "N/A"

        print(f"Cohort: Unrealized Gain >= {ret_thresh * 100:.0f}% (N = {len(df_sub):,}, Exhaustion Events = {n_pos:,} / {len(df_sub):,} [{pct_pos:.1f}%])")
        print(f"  • ROC-AUC Score:      {auc_str}")
        print(f"  • Accuracy (p>=0.50): {acc * 100:.1f}%  (Balanced: {bal_acc_str})")
        print(f"  • Precision / Recall: {prec * 100:.1f}% / {rec * 100:.1f}%  (F1: {f1:.3f})")
        print(f"  • Mean Prob Predicted: Exhaustion={mean_pos} vs Normal={mean_neg}\n")

        cohort_key = f"gain_ge_{int(ret_thresh * 100)}"
        metrics_result["exhaustion"][cohort_key] = {
            "n": len(df_sub),
            "n_pos": n_pos,
            "roc_auc": auc,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }

    # -------------------------------------------------------------------------
    # 2. EVALUATION: DYNAMIC TRAILING STOP QUANTILE REGRESSOR
    # -------------------------------------------------------------------------
    print("-" * 80)
    print("📊 2. MODEL 2: DYNAMIC TRAILING STOP QUANTILE REGRESSOR (ep_dynamic_trailer.pkl)")
    print("-" * 80)

    df_stop_eval = df.dropna(subset=["forward_mae_pct"]).copy()
    if len(df_stop_eval) > 0:
        X_stop = df_stop_eval[feature_cols]
        preds_stop = pipe_stop.predict(X_stop)
        actual_mae_pct = np.abs(df_stop_eval["forward_mae_pct"].to_numpy()) * 100.0

        mae = mean_absolute_error(actual_mae_pct, preds_stop)
        alpha = 0.30
        diff = actual_mae_pct - preds_stop
        pinball_loss = np.maximum(alpha * diff, (alpha - 1.0) * diff).mean()
        coverage_pct = (actual_mae_pct <= preds_stop).mean() * 100.0

        print(f"Runner Sample Size:     {len(df_stop_eval):,} observations")
        print(f"Quantile Target Alpha:  0.30 (30th percentile adverse excursion buffer)")
        print(f"  • Mean Absolute Error: {mae:.2f}% (percentage points)")
        print(f"  • Pinball Loss:        {pinball_loss:.3f}")
        print(f"  • Empirical Coverage:  {coverage_pct:.1f}% (Actual adverse excursion stayed within buffer)")

        print("\nQuantile Distribution Comparison (Actual Drawdown vs Predicted Buffer):")
        pctiles = [10, 25, 50, 75, 90]
        act_pcts = np.percentile(actual_mae_pct, pctiles)
        pred_pcts = np.percentile(preds_stop, pctiles)
        print(f"  {'Percentile':<12} | {'Actual MAE (%)':<16} | {'Predicted Buffer (%)':<22}")
        print("  " + "-" * 54)
        for p, a_val, pr_val in zip(pctiles, act_pcts, pred_pcts):
            print(f"  {p:>2}th %ile     | {a_val:>13.2f}% | {pr_val:>19.2f}%")

        metrics_result["dynamic_trailer"] = {
            "n": len(df_stop_eval),
            "mae": mae,
            "pinball_loss": pinball_loss,
            "empirical_coverage": coverage_pct,
        }
    else:
        print("[-] No observations with valid 'forward_mae_pct' found.")

    print("\n" + "=" * 80)
    print("✅ Advanced ML Exits evaluation completed successfully.")
    print("=" * 80)
    return metrics_result


if __name__ == "__main__":
    evaluate()
