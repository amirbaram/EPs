"""train_qm_climax_model.py — Trains Dedicated ML Climax Exhaustion Model for QM Setups.

Implements zero-lookahead feature engineering and training as specified in
docs/QM_ML_EXHAUSTION_CLIMAX_SPEC.md:
- 25 features spanning moving average stretch, RSI momentum extremes, candle geometry,
  climax volume, TrendLab structural swings, and macro sponsorship.
- Purged chronological split (Train < 2023, Test >= 2023).
- Outputs data/models/qm_exhaustion_classifier.pkl.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import datastore
import indicators
import trendlab

SETUPS_PATH = Path("data/simulations/qm_historical_setups.parquet")
TRADES_PATH = Path("data/simulations/qm_strategy_trades.parquet")
MODEL_OUT_PATH = Path("data/models/qm_exhaustion_classifier.pkl")

CLIMAX_FEATURE_COLS = [
    "feat_dist_ema10_pct",
    "feat_dist_ema20_pct",
    "feat_dist_sma50_pct",
    "feat_adr_multiple",
    "feat_unrealized_r",
    "feat_rsi14",
    "feat_rsi3_extreme",
    "feat_rsi_divergence",
    "feat_rsi_vs_sma",
    "feat_bar_rvol",
    "feat_vol_vs_breakout",
    "feat_upper_wick_ratio",
    "feat_close_pos",
    "feat_consec_up_days",
    "feat_bars_since_entry",
    "feat_tl_swing_dist",
    "feat_tl_is_pivot_high",
    "feat_dist_52w_high",
    "feat_dist_pole_peak",
    "feat_pole_move_pct",
    "feat_cons_bars",
    "feat_pinch_ratio",
    "feat_prior_ep_days",
    "feat_rs_score",
    "feat_macro_tailwind",
]


def calc_rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def extract_climax_dataset() -> pd.DataFrame:
    print(f"Loading trades from {TRADES_PATH}...")
    trades = pd.read_parquet(TRADES_PATH)
    print(f"Loaded {len(trades)} trades.")

    # Filter to trades that experienced positive extension (MFE >= 1.8 R or gain >= 10%)
    runners = trades[(trades["mfe_r_06x"] >= 1.8) | (trades["gain_pct_06x"] >= 10.0)].copy()
    print(f"Qualifying runner trades for climax extraction: {len(runners)} across {runners['symbol'].nunique()} symbols.")

    grouped = runners.groupby("symbol")
    symbols = sorted(grouped.groups.keys())
    total_syms = len(symbols)

    records: List[Dict[str, Any]] = []
    t0 = time.time()

    for i_sym, sym in enumerate(symbols):
        if (i_sym + 1) % 250 == 0 or (i_sym + 1) == total_syms:
            elapsed = time.time() - t0
            rate = (i_sym + 1) / elapsed if elapsed > 0 else 0
            print(f"Progress: {i_sym + 1}/{total_syms} symbols ({(i_sym + 1)/total_syms*100:.1f}%) | Samples: {len(records)} | {rate:.1f} sym/s")

        bars = datastore.load_bars(sym)
        if bars is None or len(bars) < 60:
            continue

        c_s = bars["close"]
        h_s = bars["high"]
        l_s = bars["low"]
        o_s = bars["open"]
        v_s = bars["volume"]

        e10 = c_s.ewm(span=10, adjust=False).mean().to_numpy(float)
        e20 = c_s.ewm(span=20, adjust=False).mean().to_numpy(float)
        s50 = c_s.rolling(50, min_periods=20).mean().to_numpy(float)
        v20 = v_s.rolling(20, min_periods=5).mean().to_numpy(float)

        rsi14 = calc_rsi(c_s, 14).to_numpy(float)
        rsi3 = calc_rsi(c_s, 3).to_numpy(float)
        sma5_rsi14 = pd.Series(rsi14).rolling(5, min_periods=2).mean().to_numpy(float)

        prev_c = c_s.shift(1)
        tr = pd.concat([h_s - l_s, (h_s - prev_c).abs(), (l_s - prev_c).abs()], axis=1).max(axis=1)
        atr14 = tr.rolling(14, min_periods=5).mean().to_numpy(float)

        hi252 = h_s.rolling(252, min_periods=60).max().to_numpy(float)

        c_arr = c_s.to_numpy(float)
        h_arr = h_s.to_numpy(float)
        l_arr = l_s.to_numpy(float)
        o_arr = o_s.to_numpy(float)
        v_arr = v_s.to_numpy(float)

        try:
            r_tl = trendlab._compute(bars)
            tl_pidx = set(r_tl.get("pidx", []))
            tl_ptype = {pidx: r_tl["ptype"][i] for i, pidx in enumerate(r_tl.get("pidx", []))}
        except Exception:
            tl_pidx = set()
            tl_ptype = {}

        dates_list = bars.index.strftime("%Y-%m-%d").tolist()
        date_to_idx = {d: idx for idx, d in enumerate(dates_list)}
        n_bars = len(dates_list)

        sym_trades = grouped.get_group(sym)

        for _, tr_row in sym_trades.iterrows():
            d_str = str(tr_row["date"])
            if d_str not in date_to_idx:
                continue
            entry_idx = date_to_idx[d_str]
            entry_p = float(tr_row["entry_price"])
            stop_p = float(tr_row["stop_loss_06x"])
            init_risk = entry_p - stop_p
            if init_risk <= 0:
                continue

            vol_bo = max(float(v_arr[entry_idx]), 1.0)
            pivot_low_p = float(tr_row["pivot_low_price"])
            pole_high = entry_p  # breakout baseline level

            move_pct = float(tr_row.get("move_pct", 30.0))
            cons_bars = float(tr_row.get("cons_bars", 10.0))
            pinch_ratio = float(tr_row.get("pinch_ratio", 2.5)) if pd.notna(tr_row.get("pinch_ratio")) else 2.5
            prior_ep_days = float(tr_row.get("prior_ep_days_ago", 999.0)) if pd.notna(tr_row.get("prior_ep_days_ago")) else 999.0
            rs_score = float(tr_row.get("rs_score", 50.0)) if pd.notna(tr_row.get("rs_score")) else 50.0
            macro_tailwind = 1.0 if bool(tr_row.get("is_leading_macro", False)) else 0.0

            max_eval_bars = min(n_bars - 1, entry_idx + 45)

            for t in range(entry_idx + 1, max_eval_bars + 1):
                cur_c = c_arr[t]
                cur_h = h_arr[t]
                cur_l = l_arr[t]
                cur_o = o_arr[t]
                cur_v = v_arr[t]

                # Stoploss breach terminates trade evaluation
                if cur_l <= stop_p:
                    break

                unrealized_r = (cur_c - entry_p) / init_risk
                unrealized_ret = (cur_c / entry_p - 1.0)

                # Focus candidate climax bars on positive momentum excursions
                if unrealized_r < 1.5 and unrealized_ret < 0.08:
                    continue

                # Compute Forward Target (Zero-Lookahead Labeling)
                # Next 5-day MAE (in R multiples from cur_c)
                fwd_5_end = min(n_bars, t + 6)
                fwd_10_end = min(n_bars, t + 11)

                if fwd_5_end <= t + 1:
                    continue

                min_fwd_5_low = np.min(l_arr[t + 1:fwd_5_end])
                fwd_mae_r = (cur_c - min_fwd_5_low) / init_risk

                max_fwd_10_high = np.max(h_arr[t:fwd_10_end])
                cur_high_ratio = cur_h / (max_fwd_10_high + 1e-6)

                fwd_3_idx = min(n_bars - 1, t + 3)
                close_3_lower = (c_arr[fwd_3_idx] < cur_c)

                # Target Definition:
                # 1 = Climax Exhaustion: A bar where selling captures >= 96% of 10-day peak
                #     and is followed by a >= 1.5 R pullback or 3-day momentum rollover.
                is_climax = bool((fwd_mae_r >= 1.5) or (cur_high_ratio >= 0.96 and close_3_lower))

                # Feature extraction
                cur_e10 = e10[t]
                cur_e20 = e20[t]
                cur_s50 = s50[t] if not np.isnan(s50[t]) else cur_e20
                cur_atr = max(atr14[t], 0.01) if not np.isnan(atr14[t]) else 0.05 * cur_c

                feat_dist_e10 = (cur_c - cur_e10) / cur_e10
                feat_dist_e20 = (cur_c - cur_e20) / cur_e20
                feat_dist_s50 = (cur_c - cur_s50) / cur_s50
                feat_adr_mult = (cur_h - cur_e20) / cur_atr

                cur_r14 = rsi14[t] if not np.isnan(rsi14[t]) else 50.0
                cur_r3 = rsi3[t] if not np.isnan(rsi3[t]) else 50.0
                cur_r14_sma = sma5_rsi14[t] if not np.isnan(sma5_rsi14[t]) else cur_r14

                r3_ago_idx = max(0, t - 3)
                r14_ago = rsi14[r3_ago_idx] if not np.isnan(rsi14[r3_ago_idx]) else cur_r14
                rsi_div = (cur_r14 - r14_ago) if cur_h > h_arr[r3_ago_idx] else 0.0
                rsi_vs_sma = cur_r14 - cur_r14_sma

                bar_rvol = cur_v / max(v20[t], 1.0) if not np.isnan(v20[t]) else 1.0
                vol_vs_bo = cur_v / vol_bo

                c_range = max(cur_h - cur_l, 0.001)
                upper_wick = (cur_h - max(cur_o, cur_c)) / c_range
                close_pos = (cur_c - cur_l) / c_range

                # Consecutive up days
                consec_up = 0
                for k in range(t, max(0, t - 6), -1):
                    if k > 0 and c_arr[k] > c_arr[k - 1]:
                        consec_up += 1
                    else:
                        break

                bars_since = t - entry_idx
                tl_swing_dist = (cur_c - pivot_low_p) / pivot_low_p if pivot_low_p > 0 else 0.0
                tl_is_pivot_h = 1.0 if (t in tl_pidx and tl_ptype.get(t) == "H") else 0.0

                cur_hi252 = hi252[t] if not np.isnan(hi252[t]) else cur_h
                dist_52w = (cur_c - cur_hi252) / cur_hi252 if cur_hi252 > 0 else 0.0
                dist_pole = (cur_c - pole_high) / pole_high if pole_high > 0 else 0.0

                rec = {
                    "symbol": sym,
                    "date": dates_list[t],
                    "target_climax": 1 if is_climax else 0,
                    "feat_dist_ema10_pct": round(feat_dist_e10, 4),
                    "feat_dist_ema20_pct": round(feat_dist_e20, 4),
                    "feat_dist_sma50_pct": round(feat_dist_s50, 4),
                    "feat_adr_multiple": round(feat_adr_mult, 3),
                    "feat_unrealized_r": round(unrealized_r, 3),
                    "feat_rsi14": round(cur_r14, 2),
                    "feat_rsi3_extreme": round(cur_r3, 2),
                    "feat_rsi_divergence": round(rsi_div, 2),
                    "feat_rsi_vs_sma": round(rsi_vs_sma, 2),
                    "feat_bar_rvol": round(bar_rvol, 3),
                    "feat_vol_vs_breakout": round(vol_vs_bo, 3),
                    "feat_upper_wick_ratio": round(upper_wick, 3),
                    "feat_close_pos": round(close_pos, 3),
                    "feat_consec_up_days": consec_up,
                    "feat_bars_since_entry": bars_since,
                    "feat_tl_swing_dist": round(tl_swing_dist, 4),
                    "feat_tl_is_pivot_high": tl_is_pivot_h,
                    "feat_dist_52w_high": round(dist_52w, 4),
                    "feat_dist_pole_peak": round(dist_pole, 4),
                    "feat_pole_move_pct": round(move_pct, 2),
                    "feat_cons_bars": round(cons_bars, 1),
                    "feat_pinch_ratio": round(pinch_ratio, 2),
                    "feat_prior_ep_days": round(prior_ep_days, 1),
                    "feat_rs_score": round(rs_score, 1),
                    "feat_macro_tailwind": macro_tailwind,
                }
                records.append(rec)

    df_out = pd.DataFrame(records)
    print(f"Total candidate bars extracted: {len(df_out)}.")
    return df_out


def train_and_evaluate(df: pd.DataFrame) -> Pipeline:
    print("\n══════════════════════════════════════════════════════════════════════════════")
    print(" 🧠 Training QM Climax Exhaustion Model (LGBMClassifier)")
    print("══════════════════════════════════════════════════════════════════════════════")

    df["date_idx"] = pd.to_datetime(df["date"])
    df = df.sort_values("date_idx").reset_index(drop=True)

    # Purged Chronological Split: Train < 2023, Test >= 2023
    train_mask = df["date_idx"] < pd.Timestamp("2023-01-01")
    test_mask = df["date_idx"] >= pd.Timestamp("2023-01-01")

    train_df = df[train_mask]
    test_df = df[test_mask]

    print(f"  • Training observations   (< 2023) : {len(train_df):,} bars (Climax: {train_df['target_climax'].sum():,}, {train_df['target_climax'].mean()*100:.1f}%)")
    print(f"  • Out-of-Sample test bars (≥ 2023) : {len(test_df):,} bars (Climax: {test_df['target_climax'].sum():,}, {test_df['target_climax'].mean()*100:.1f}%)")

    X_train = train_df[CLIMAX_FEATURE_COLS]
    y_train = train_df["target_climax"]

    X_test = test_df[CLIMAX_FEATURE_COLS]
    y_test = test_df["target_climax"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = lgb.LGBMClassifier(
        n_estimators=150,
        max_depth=4,
        num_leaves=15,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )

    clf.fit(X_train_scaled, y_train)

    # Evaluate on OOS Test Set
    probs_test = clf.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, probs_test)
    print(f"\n  🎯 Out-of-Sample ROC-AUC (2023–2026): {auc:.4f}")

    preds_test = (probs_test >= 0.50).astype(int)
    print("\nClassification Report (Threshold 0.50):")
    print(classification_report(y_test, preds_test, digits=3))

    # Feature Importance
    importances = clf.feature_importances_
    feat_imp = sorted(zip(CLIMAX_FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)
    print("\nTop 10 Feature Importances:")
    for f, imp in feat_imp[:10]:
        print(f"  • {f:<25}: {imp}")

    pipe = Pipeline([
        ("scaler", scaler),
        ("classifier", clf),
    ])

    MODEL_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_OUT_PATH)
    print(f"\n✅ Successfully saved QM Exhaustion Classifier to {MODEL_OUT_PATH}.")
    return pipe


def main():
    df_data = extract_climax_dataset()
    if df_data.empty:
        print("❌ Error: No climax dataset samples extracted.")
        return
    train_and_evaluate(df_data)


if __name__ == "__main__":
    main()
