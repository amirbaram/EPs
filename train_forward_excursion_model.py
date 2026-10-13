import time
import datastore
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, accuracy_score
import lightgbm as lgb

def train_forward_excursion_model():
    print("=" * 80)
    print("🚀 TRAINING FORWARD 20-SESSION MAX EXCURSION MODEL")
    print("=" * 80)

    parquet_path = Path("data/ml_datasets/amir_rolling/dataset_rolling.parquet")
    if not parquet_path.exists():
        print(f"[-] Missing dataset: {parquet_path}")
        return

    df = pd.read_parquet(parquet_path)
    print(f"[*] Loaded {len(df):,} rolling observations from {parquet_path}")

    # Compute Forward 20-day MFE, MAE, and First-Arrival Sequence for all rows
    print("[*] Computing forward 20-session MFE, MAE & sequence targets from datastore...")
    t0 = time.time()
    mfe_20d = np.full(len(df), np.nan)
    mae_20d = np.full(len(df), np.nan)
    seq_mfe_first = np.full(len(df), np.nan)
    barrier_2r = np.full(len(df), np.nan)

    sym_groups = df.groupby("symbol").indices
    for sym, idxs in sym_groups.items():
        bars = datastore.load_bars(sym)
        if bars is None or len(bars) == 0:
            continue
        date_to_pos = {d: i for i, d in enumerate(bars.index.strftime("%Y-%m-%d"))}
        closes = bars["close"].to_numpy(dtype=float)
        highs = bars["high"].to_numpy(dtype=float)
        lows = bars["low"].to_numpy(dtype=float)
        opens = bars["open"].to_numpy(dtype=float)
        n_bars = len(bars)

        sym_dates = df["date"].iloc[idxs].values
        sym_dists = df["feature_dist_to_d1_low"].iloc[idxs].values
        for k, d_str in enumerate(sym_dates):
            pos = date_to_pos.get(d_str)
            if pos is not None and pos + 1 < n_bars:
                f_end = min(n_bars, pos + 21)
                c = closes[pos]
                if c > 0:
                    sub_h = highs[pos+1:f_end]
                    sub_l = lows[pos+1:f_end]
                    mfe_20d[idxs[k]] = (np.max(sub_h) - c) / c * 100.0
                    mae_20d[idxs[k]] = (np.min(sub_l) - c) / c * 100.0
                    if len(sub_h) > 0:
                        t_mfe = int(np.argmax(sub_h))
                        t_mae = int(np.argmin(sub_l))
                        if t_mfe < t_mae:
                            seq_mfe_first[idxs[k]] = 1.0
                        elif t_mae < t_mfe:
                            seq_mfe_first[idxs[k]] = 0.0
                        else:
                            sub_o = opens[pos+1:f_end]
                            sub_c = closes[pos+1:f_end]
                            seq_mfe_first[idxs[k]] = 1.0 if sub_c[t_mfe] < sub_o[t_mfe] else 0.0

                    # Option B: Target 2.0R vs 1.0R Stop Loss barrier hit within 20 sessions
                    dist_low = sym_dists[k]
                    if pd.notna(dist_low) and dist_low > 0.02 and dist_low < 0.50:
                        risk_pct = dist_low
                        target_2r_px = c * (1.0 + 2.0 * risk_pct)
                        stop_px = c * (1.0 - risk_pct)
                        t_target_2r = np.where(sub_h >= target_2r_px)[0]
                        t_stop = np.where(sub_l <= stop_px)[0]
                        first_target_2r = t_target_2r[0] if len(t_target_2r) > 0 else 999
                        first_stop = t_stop[0] if len(t_stop) > 0 else 999
                        barrier_2r[idxs[k]] = 1.0 if first_target_2r < first_stop else 0.0

    df["target_mfe_20d"] = mfe_20d
    df["target_mae_20d"] = mae_20d
    df["target_mfe_first"] = seq_mfe_first
    df["target_barrier_2r"] = barrier_2r
    t1 = time.time()
    print(f"[*] 20D targets computed in {t1 - t0:.2f}s")

    # Filter to valid observations in the first 20 sessions post-EP
    df_clean = df[
        (df["feature_days_since_ep"] >= 1) &
        (df["feature_days_since_ep"] <= 20) &
        df["target_mfe_20d"].notna() &
        df["target_mae_20d"].notna() &
        df["target_mfe_first"].notna()
    ].copy()
    print(f"[*] Clean dataset: {len(df_clean):,} observations across days 1–20 post-EP")

    feature_cols = [
        "feature_days_since_ep", "feature_ret_since_ep", "feature_dist_to_d1_low",
        "feature_tightness_1m", "feature_tightness_since_ep", "feature_digestion_ratio",
        "feature_gap_pct", "feature_rvol", "feature_close_pos", "feature_is_novel_vol_9m",
        "feature_is_inst_sweet_spot", "feature_is_blue_sky", "feature_dist_to_overhead_pct",
        "feature_ind_rank_3m", "feature_tk_rs_spy_3m", "feature_is_qullamagi_linear"
    ]

    # Fill NA on numeric features cleanly
    for c in feature_cols:
        df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce").fillna(0.0)

    # Chronological In-Sample / Out-of-Sample Split
    df_clean["date_dt"] = pd.to_datetime(df_clean["date"])
    train_mask = df_clean["date_dt"] < pd.Timestamp("2023-01-01")
    test_mask = df_clean["date_dt"] >= pd.Timestamp("2023-01-01")

    X_train = df_clean.loc[train_mask, feature_cols]
    y_mfe_train = df_clean.loc[train_mask, "target_mfe_20d"]
    y_mae_train = df_clean.loc[train_mask, "target_mae_20d"]
    y_seq_train = df_clean.loc[train_mask, "target_mfe_first"]

    X_test = df_clean.loc[test_mask, feature_cols]
    y_mfe_test = df_clean.loc[test_mask, "target_mfe_20d"]
    y_mae_test = df_clean.loc[test_mask, "target_mae_20d"]
    y_seq_test = df_clean.loc[test_mask, "target_mfe_first"]

    print(f"[*] Train set (< 2023): {len(X_train):,} samples | Test set (>= 2023): {len(X_test):,} samples")

    # 1. Train Model: Forward 20D Maximum Favorable Excursion (MFE) Regressor
    print("\n[*] Training Forward 20D MFE Regressor (Expected Peak Gain %)...")
    pipe_mfe = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.03, max_depth=5,
            num_leaves=20, min_child_samples=30, subsample=0.85,
            colsample_bytree=0.85, random_state=42, verbose=-1
        ))
    ])
    pipe_mfe.fit(X_train, y_mfe_train)

    pred_mfe_test = pipe_mfe.predict(X_test)
    mfe_mae_test = mean_absolute_error(y_mfe_test, pred_mfe_test)
    mfe_r2_test = r2_score(y_mfe_test, pred_mfe_test)
    print(f"  • OOS 20D MFE Mean Absolute Error: {mfe_mae_test:.2f}%")
    print(f"  • OOS 20D MFE R^2 Score:           {mfe_r2_test:.3f}")
    print(f"  • Actual Mean MFE: +{y_mfe_test.mean():.2f}% vs Predicted: +{pred_mfe_test.mean():.2f}%")

    # 2. Train Model: Forward 20D Maximum Adverse Excursion (MAE) Quantile Regressor
    print("\n[*] Training Forward 20D MAE Regressor (Drawdown Risk Cushion %)...")
    pipe_mae = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", lgb.LGBMRegressor(
            objective="quantile", alpha=0.30,
            n_estimators=180, learning_rate=0.03, max_depth=4,
            num_leaves=15, min_child_samples=30, subsample=0.85,
            random_state=42, verbose=-1
        ))
    ])
    pipe_mae.fit(X_train, y_mae_train)

    pred_mae_test = pipe_mae.predict(X_test)
    mae_diff_test = mean_absolute_error(y_mae_test, pred_mae_test)
    coverage = (y_mae_test >= pred_mae_test).mean() * 100.0
    print(f"  • OOS 20D MAE Mean Absolute Error: {mae_diff_test:.2f}%")
    print(f"  • OOS 20D MAE Quantile Coverage:   {coverage:.1f}%")

    # 3. Train Model: Forward 20D Sequence Classifier (Probability MFE First)
    print("\n[*] Training Forward 20D Sequence Classifier (P(MFE First vs MAE First))...")
    pipe_seq = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", lgb.LGBMClassifier(
            n_estimators=180, learning_rate=0.03, max_depth=4,
            num_leaves=16, min_child_samples=30, subsample=0.85,
            colsample_bytree=0.85, random_state=42, verbose=-1
        ))
    ])
    pipe_seq.fit(X_train, y_seq_train)

    probs_seq_test = pipe_seq.predict_proba(X_test)[:, 1]
    auc_seq = roc_auc_score(y_seq_test, probs_seq_test)
    acc_seq = accuracy_score(y_seq_test, probs_seq_test >= 0.5)
    print(f"  • OOS 20D Sequence ROC AUC:       {auc_seq:.3f}")
    print(f"  • OOS 20D Sequence Accuracy:      {acc_seq * 100:.1f}%")

    # 4. Train Model: Forward 20D Option B Barrier Classifier (P(2.0R Target Before 1.0R Stop))
    print("\n[*] Training Forward 20D Option B Barrier Classifier (P(2.0R Target Before Stop))...")
    barrier_tr_m = train_mask & df_clean["target_barrier_2r"].notna()
    barrier_te_m = test_mask & df_clean["target_barrier_2r"].notna()
    X_barrier_train = df_clean.loc[barrier_tr_m, feature_cols]
    y_barrier_train = df_clean.loc[barrier_tr_m, "target_barrier_2r"]
    X_barrier_test = df_clean.loc[barrier_te_m, feature_cols]
    y_barrier_test = df_clean.loc[barrier_te_m, "target_barrier_2r"]

    pipe_barrier = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", lgb.LGBMClassifier(
            n_estimators=180, learning_rate=0.03, max_depth=4,
            num_leaves=16, min_child_samples=30, subsample=0.85,
            colsample_bytree=0.85, random_state=42, verbose=-1
        ))
    ])
    pipe_barrier.fit(X_barrier_train, y_barrier_train)

    probs_barrier_test = pipe_barrier.predict_proba(X_barrier_test)[:, 1]
    auc_barrier = roc_auc_score(y_barrier_test, probs_barrier_test)
    acc_barrier = accuracy_score(y_barrier_test, probs_barrier_test >= 0.5)
    top10_barrier = y_barrier_test[probs_barrier_test >= np.percentile(probs_barrier_test, 90)].mean() * 100
    top5_barrier = y_barrier_test[probs_barrier_test >= np.percentile(probs_barrier_test, 95)].mean() * 100
    base_rate = y_barrier_test.mean() * 100

    print(f"  • OOS 20D Barrier ROC AUC:        {auc_barrier:.4f}")
    print(f"  • OOS 20D Barrier Accuracy:       {acc_barrier * 100:.1f}%")
    print(f"  • Base Win Rate (2R Before Stop): {base_rate:.1f}%")
    print(f"  • Top 10% Conviction Win Rate:    {top10_barrier:.1f}% (Lift: {top10_barrier/base_rate:.2f}x)")
    print(f"  • Top 5% Conviction Win Rate:     {top5_barrier:.1f}% (Lift: {top5_barrier/base_rate:.2f}x)")

    # Save packaged model bundle
    out_dir = Path("data/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "ep_forward_excursion_regressor.pkl"

    bundle = {
        "mfe_model": pipe_mfe,
        "mae_model": pipe_mae,
        "seq_model": pipe_seq,
        "barrier_model": pipe_barrier,
        "feature_cols": feature_cols,
        "train_date": "2026-10-10",
        "description": "Forward 20-Session Maximum Excursion Regressors (MFE & MAE), Sequence Classifier (P(MFE First)), and Barrier Classifier (P(2R Before Stop))"
    }
    joblib.dump(bundle, out_file)
    print(f"\n✅ Successfully saved forward excursion model bundle to: {out_file}")
    print("=" * 80)

if __name__ == "__main__":
    train_forward_excursion_model()
