"""Train Continuation ML Models:
1. Model 1: Cost-Sensitive Continuation / Undercut & Reclaim (U&R) Qualification Classifier
   (Saved to data/models/ep_ur_reentry_classifier.pkl)
2. Model 2: Continuation Trailing Stop Quantile Regressor
   (Saved to data/models/ep_continuation_dynamic_trailer.pkl)
"""
import os
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import datastore
import indicators
import scanner_core


def extract_continuation_datasets(ribbon_spans=(8, 12, 16, 21)):
    print("🚀 Extracting Continuation Datasets from EP Universe (6,501 events)...")
    study_path = Path("data/simulations/ep_combined_study_scored.parquet")
    backup_path = Path("data/simulations/ep_combined_study_scored_pre_dual_track.parquet")
    source_path = backup_path if backup_path.exists() else study_path
    df_ep = pd.read_parquet(source_path)

    ur_dataset = []
    trailer_dataset = []

    t0 = time.time()
    for i, (_, row) in enumerate(df_ep.iterrows()):
        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i}/{len(df_ep)} EPs ({time.time() - t0:.1f}s)...")

        sym = row["symbol"]
        date_str = str(row["date"])
        raw = datastore.load_bars(sym)
        if raw is None or len(raw) < 30:
            continue

        d = raw.copy()
        if "rvol" not in d.columns:
            d = indicators.add_indicators(d)
        d = d.loc[:, ~d.columns.duplicated()]

        d = scanner_core.calc_intermediate_ribbon(d, spans=ribbon_spans)
        if "sma50" not in d.columns:
            d["sma50"] = d["close"].rolling(50).mean()
        if "vol20" not in d.columns:
            d["vol20"] = d["volume"].rolling(20).mean()

        ts = pd.Timestamp(date_str)
        if ts not in d.index:
            continue
        pos = d.index.get_loc(ts)
        if isinstance(pos, (slice, np.ndarray)):
            pos = pos[0] if isinstance(pos, np.ndarray) else pos.start
        if pos >= len(d) - 2:
            continue

        fwd = d.iloc[pos:min(len(d), pos + 250)].copy()
        if len(fwd) < 5:
            continue

        d1 = fwd.iloc[0]
        d1_close = float(d1["close"])
        d1_low = float(d1["low"])
        d1_high = float(d1["high"])
        risk1_pct = (d1_close - d1_low) / d1_close * 100.0 if d1_close > 0 else 11.1

        # Simulate Trade 1
        t1_stopped = False
        t1_exit_bar = None
        t1_peak = d1_close
        seen_bull = False

        for b in range(1, len(fwd)):
            c = float(fwd["close"].iloc[b])
            l = float(fwd["low"].iloc[b])
            h = float(fwd["high"].iloc[b])
            r_state = fwd["ribbon_state"].iloc[b]
            if h > t1_peak:
                t1_peak = h
            if pd.notna(r_state) and r_state in ["yellow", "gray"]:
                seen_bull = True

            if l <= d1_low:
                t1_stopped = True
                t1_exit_bar = b
                break
            if seen_bull and pd.notna(r_state) and r_state == "blue":
                t1_exit_bar = b
                break
            sma50 = fwd["sma50"].iloc[b]
            if seen_bull and pd.notna(sma50) and c < sma50:
                t1_exit_bar = b
                break

        if t1_exit_bar is None:
            t1_exit_bar = len(fwd) - 1

        # Evaluate Track 1 U&R Candidates
        if t1_stopped and 1 <= t1_exit_bar <= 10:
            shakeout_low = float(fwd["low"].iloc[t1_exit_bar])
            controlled_ur = (shakeout_low >= d1_low * 0.85)

            ur_signal_bar = None
            if controlled_ur:
                ur_end_search = min(len(fwd), t1_exit_bar + 16)
                for b in range(t1_exit_bar + 1, ur_end_search):
                    l_b = float(fwd["low"].iloc[b])
                    c_b = float(fwd["close"].iloc[b])
                    if l_b < shakeout_low:
                        shakeout_low = l_b
                    if shakeout_low < d1_low * 0.85:
                        controlled_ur = False
                        break
                    if c_b >= d1_low:
                        ur_signal_bar = b
                        break

            if controlled_ur and ur_signal_bar is not None and ur_signal_bar + 1 < len(fwd):
                entry_bar = ur_signal_bar + 1
                t2_entry = float(fwd["open"].iloc[entry_bar])
                t2_stop = round(shakeout_low, 2)
                risk_pct = (t2_entry - t2_stop) / t2_entry * 100.0

                if 1.0 <= risk_pct <= 35.0:
                    risk_pts = t2_entry - t2_stop
                    target_15pct_price = t2_entry * 1.15
                    target_3r_price = t2_entry + 3.0 * risk_pts

                    # Target: hitting +15% gain OR +3R before hitting stop loss
                    hit_target = 0
                    stopped_out = False
                    hold_end = min(len(fwd), entry_bar + 80)

                    # Check day 1 on entry bar
                    day1_l = float(fwd["low"].iloc[entry_bar])
                    day1_h = float(fwd["high"].iloc[entry_bar])
                    if day1_l <= t2_stop:
                        stopped_out = True
                    elif day1_h >= target_15pct_price or day1_h >= target_3r_price:
                        hit_target = 1

                    if not stopped_out and hit_target == 0:
                        for fb in range(entry_bar + 1, hold_end):
                            fl = float(fwd["low"].iloc[fb])
                            fh = float(fwd["high"].iloc[fb])
                            if fl <= t2_stop:
                                stopped_out = True
                                break
                            if fh >= target_15pct_price or fh >= target_3r_price:
                                hit_target = 1
                                break

                    # Signal bar features (zero lookahead)
                    sig_c = float(fwd["close"].iloc[ur_signal_bar])
                    sig_sma50 = float(fwd["sma50"].iloc[ur_signal_bar]) if pd.notna(fwd["sma50"].iloc[ur_signal_bar]) else sig_c
                    sig_vol = float(fwd["volume"].iloc[ur_signal_bar])
                    sig_vol20 = float(fwd["vol20"].iloc[ur_signal_bar]) if pd.notna(fwd["vol20"].iloc[ur_signal_bar]) else sig_vol
                    sig_rvol = float(fwd["rvol"].iloc[ur_signal_bar]) if pd.notna(fwd["rvol"].iloc[ur_signal_bar]) else 1.0
                    sig_comp = float(fwd["ribbon_compression"].iloc[ur_signal_bar]) if pd.notna(fwd["ribbon_compression"].iloc[ur_signal_bar]) else 0.0

                    undercut_depth_pct = (d1_low - shakeout_low) / d1_low * 100.0
                    days_to_reclaim = float(ur_signal_bar - t1_exit_bar)
                    reclaim_vol_ratio = sig_vol / sig_vol20 if sig_vol20 > 0 else 1.0
                    dist_to_sma50 = (sig_c - sig_sma50) / sig_sma50 if sig_sma50 > 0 else 0.0
                    sma50_cushion = (shakeout_low - sig_sma50) / sig_sma50 if sig_sma50 > 0 else 0.0
                    drop_from_peak = (t1_peak - shakeout_low) / t1_peak * 100.0 if t1_peak > 0 else 0.0

                    sig_date = fwd.index[ur_signal_bar].strftime("%Y-%m-%d")
                    ur_dataset.append({
                        "symbol": sym,
                        "date": sig_date,
                        "entry_date": fwd.index[entry_bar].strftime("%Y-%m-%d"),
                        "feature_undercut_depth_pct": undercut_depth_pct,
                        "feature_days_to_reclaim": days_to_reclaim,
                        "feature_reclaim_volume_ratio": reclaim_vol_ratio,
                        "feature_dist_to_sma50": dist_to_sma50,
                        "feature_sma50_cushion": sma50_cushion,
                        "feature_ribbon_compression": sig_comp,
                        "feature_drop_from_peak": drop_from_peak,
                        "feature_rvol": sig_rvol,
                        "target_win": hit_target,
                    })

        # Evaluate Continuation Trades (Track 1 & Track 2) for Model 2 (Dynamic Trailer)
        # We simulate continuation trades that develop into runners (clearing >= +2.5R)
        # to record the distribution of pullback MAE from peak.
        seen_cons = False
        reentry_signal_bar = None
        cons_low = float(fwd["low"].iloc[t1_exit_bar])
        curr_peak = t1_peak

        search_end = min(len(fwd), t1_exit_bar + 66)
        for b in range(t1_exit_bar, search_end):
            l = float(fwd["low"].iloc[b])
            h = float(fwd["high"].iloc[b])
            c = float(fwd["close"].iloc[b])
            s = fwd["ribbon_state"].iloc[b]
            if h > curr_peak:
                curr_peak = h
                cons_low = l
            elif l < cons_low:
                cons_low = l
            if pd.notna(s) and s in ["blue", "gray"]:
                seen_cons = True
            pullback_pct = (curr_peak - cons_low) / curr_peak * 100.0 if curr_peak > 0 else 0.0
            if seen_cons and b > t1_exit_bar and pd.notna(s) and s == "yellow":
                reentry_signal_bar = b
                break
            if not seen_cons and b > t1_exit_bar + 2 and pullback_pct >= 5.0 and pd.notna(s) and s == "yellow":
                prev_5d_high = float(fwd["high"].iloc[max(0, b - 5):b].max())
                if c > prev_5d_high:
                    reentry_signal_bar = b
                    break

        # If continuation trade enters
        if reentry_signal_bar is not None and reentry_signal_bar + 1 < len(fwd):
            c_sig = float(fwd["close"].iloc[reentry_signal_bar])
            sma50_sig = fwd["sma50"].iloc[reentry_signal_bar] if "sma50" in fwd.columns else None
            held_50 = pd.isna(sma50_sig) or c_sig >= sma50_sig
            elapsed_days = int(reentry_signal_bar - t1_exit_bar)
            allowed_cons = (elapsed_days <= 65 if held_50 else elapsed_days <= 45)
            drop_fp = (curr_peak - cons_low) / curr_peak * 100.0 if curr_peak > 0 else 0.0

            if allowed_cons and drop_fp <= 55.0:
                swing5_low = float(fwd["low"].iloc[max(0, reentry_signal_bar - 4):reentry_signal_bar + 1].min())
                t2_stop = round(swing5_low, 2)
                entry_bar = reentry_signal_bar + 1
                t2_entry = float(fwd["open"].iloc[entry_bar])
                risk_pct = (t2_entry - t2_stop) / t2_entry * 100.0

                if 1.0 <= risk_pct <= 35.0:
                    risk_pts = t2_entry - t2_stop
                    trade_peak = t2_entry
                    run_bars = fwd.iloc[entry_bar:]

                    for tb_idx in range(len(run_bars)):
                        b_abs = entry_bar + tb_idx
                        bh = float(fwd["high"].iloc[b_abs])
                        bl = float(fwd["low"].iloc[b_abs])
                        bc = float(fwd["close"].iloc[b_abs])
                        if bh > trade_peak:
                            trade_peak = bh

                        # Check if unrealized R clears >= +2.5R
                        unrealized_r = (trade_peak - t2_entry) / risk_pts
                        if unrealized_r >= 2.5 and tb_idx >= 3:
                            # Forward MAE from peak over the next 20 sessions or till trade exit
                            fwd_window = fwd.iloc[b_abs + 1:min(len(fwd), b_abs + 21)]
                            if len(fwd_window) >= 3:
                                min_low = float(fwd_window["low"].min())
                                mae_pct = (trade_peak - min_low) / trade_peak * 100.0

                                # Extract continuation runner features
                                ema21_col = f"ribbon_ema{ribbon_spans[3]}"
                                e21 = float(fwd[ema21_col].iloc[b_abs]) if ema21_col in fwd.columns else bc
                                sma50_val = float(fwd["sma50"].iloc[b_abs]) if pd.notna(fwd["sma50"].iloc[b_abs]) else bc
                                comp_val = float(fwd["ribbon_compression"].iloc[b_abs]) if pd.notna(fwd["ribbon_compression"].iloc[b_abs]) else 0.0
                                v20 = float(fwd["vol20"].iloc[b_abs]) if pd.notna(fwd["vol20"].iloc[b_abs]) else 1.0
                                v_curr = float(fwd["volume"].iloc[b_abs])

                                trailer_dataset.append({
                                    "symbol": sym,
                                    "date": fwd.index[b_abs].strftime("%Y-%m-%d"),
                                    "feature_r_multiple": unrealized_r,
                                    "feature_dist_ribbon_ema21": (bc - e21) / e21 if e21 > 0 else 0.0,
                                    "feature_ribbon_compression": comp_val,
                                    "feature_dist_50sma": (bc - sma50_val) / sma50_val if sma50_val > 0 else 0.0,
                                    "feature_vol_contraction": v_curr / v20 if v20 > 0 else 1.0,
                                    "feature_days_in_trade": float(tb_idx),
                                    "feature_drawdown_from_peak": (bc - trade_peak) / trade_peak * 100.0,
                                    "target_mae_pct": max(1.0, min(45.0, mae_pct)),
                                })

                        if bl <= t2_stop:
                            break

    df_ur = pd.DataFrame(ur_dataset)
    df_trailer = pd.DataFrame(trailer_dataset)
    print(f"Extraction complete in {time.time() - t0:.1f}s.")
    print(f"  Track 1 U&R Dataset: {len(df_ur)} records, Target Win Rate: {df_ur['target_win'].mean()*100:.1f}%")
    print(f"  Continuation Trailer Dataset: {len(df_trailer)} records, Mean Target MAE: {df_trailer['target_mae_pct'].mean():.2f}%")
    return df_ur, df_trailer


def train_ur_classifier(df_ur: pd.DataFrame):
    print("\n" + "=" * 80)
    print("🧠 TRAINING MODEL 1: U&R QUALIFICATION CLASSIFIER")
    print("=" * 80)
    features = [
        "feature_undercut_depth_pct",
        "feature_days_to_reclaim",
        "feature_reclaim_volume_ratio",
        "feature_dist_to_sma50",
        "feature_sma50_cushion",
        "feature_ribbon_compression",
        "feature_drop_from_peak",
        "feature_rvol",
    ]

    df_ur["date_idx"] = pd.to_datetime(df_ur["date"])
    df_ur = df_ur.sort_values("date_idx").reset_index(drop=True)

    # Chronological Split
    train_mask = df_ur["date_idx"] < pd.Timestamp("2021-01-01")
    cal_mask = (df_ur["date_idx"] >= pd.Timestamp("2021-01-01")) & (df_ur["date_idx"] < pd.Timestamp("2023-01-01"))
    test_mask = df_ur["date_idx"] >= pd.Timestamp("2023-01-01")

    train_df = df_ur[train_mask]
    cal_df = df_ur[cal_mask]
    test_df = df_ur[test_mask]

    print(f"Split sizes: Train={len(train_df)}, Calibration={len(cal_df)}, Test={len(test_df)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(train_df[features])
    y_train = train_df["target_win"]

    X_cal_s = scaler.transform(cal_df[features])
    y_cal = cal_df["target_win"]

    X_test_s = scaler.transform(test_df[features])
    y_test = test_df["target_win"]

    # Train base LightGBM with cost-sensitive class balancing
    base_lgb = lgb.LGBMClassifier(
        n_estimators=160,
        max_depth=4,
        learning_rate=0.04,
        class_weight="balanced",
        min_child_samples=15,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbose=-1,
    )
    base_lgb.fit(X_train_s, y_train)

    # Leakage-free isotonic probability calibration on calibration cohort
    calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(base_lgb), method="isotonic")
    calibrated.fit(X_cal_s, y_cal)

    serving_pipeline = Pipeline([
        ("scaler", scaler),
        ("classifier", calibrated),
    ])

    # Out-of-sample Evaluation
    preds_test_prob = serving_pipeline.predict_proba(test_df[features])[:, 1]
    auc_score = roc_auc_score(y_test, preds_test_prob)
    print(f"Out-of-sample Test AUC: {auc_score:.3f}")

    # Top decile precision
    top_decile = test_df.copy()
    top_decile["prob"] = preds_test_prob
    high_prob = top_decile[top_decile["prob"] >= 0.45]
    print(f"High-Conviction (>0.45 prob) Win Rate: {high_prob['target_win'].mean()*100:.1f}% vs Baseline: {test_df['target_win'].mean()*100:.1f}%")

    out_path = Path("data/models/ep_ur_reentry_classifier.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(serving_pipeline, out_path)
    print(f"✅ Saved Model 1 to {out_path}")
    return serving_pipeline


def train_continuation_trailer(df_trailer: pd.DataFrame):
    print("\n" + "=" * 80)
    print("🧠 TRAINING MODEL 2: CONTINUATION TRAILING STOP QUANTILE REGRESSOR")
    print("=" * 80)
    features = [
        "feature_r_multiple",
        "feature_dist_ribbon_ema21",
        "feature_ribbon_compression",
        "feature_dist_50sma",
        "feature_vol_contraction",
        "feature_days_in_trade",
        "feature_drawdown_from_peak",
    ]

    df_trailer["date_idx"] = pd.to_datetime(df_trailer["date"])
    df_trailer = df_trailer.sort_values("date_idx").reset_index(drop=True)

    train_mask = df_trailer["date_idx"] < pd.Timestamp("2022-01-01")
    test_mask = df_trailer["date_idx"] >= pd.Timestamp("2022-01-01")

    train_df = df_trailer[train_mask]
    test_df = df_trailer[test_mask]

    print(f"Split sizes: Train={len(train_df)}, Test={len(test_df)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(train_df[features])
    y_train = train_df["target_mae_pct"]

    X_test_s = scaler.transform(test_df[features])
    y_test = test_df["target_mae_pct"]

    # Asymmetric quantile regression: 10th percentile buffer from peak
    # To protect profits tightly without prematurely triggering on micro-noise
    reg = lgb.LGBMRegressor(
        objective="quantile",
        alpha=0.15,
        n_estimators=180,
        learning_rate=0.03,
        max_depth=4,
        min_child_samples=20,
        subsample=0.85,
        random_state=42,
        verbose=-1,
    )
    reg.fit(X_train_s, y_train)

    pipeline = Pipeline([
        ("scaler", scaler),
        ("regressor", reg),
    ])

    test_preds = pipeline.predict(test_df[features])
    print(f"Test Predicted 15th percentile MAE Buffer: Mean={test_preds.mean():.2f}%, Min={test_preds.min():.2f}%, Max={test_preds.max():.2f}%")

    out_path = Path("data/models/ep_continuation_dynamic_trailer.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)
    print(f"✅ Saved Model 2 to {out_path}")
    return pipeline


if __name__ == "__main__":
    df_ur, df_trailer = extract_continuation_datasets()
    train_ur_classifier(df_ur)
    train_continuation_trailer(df_trailer)
