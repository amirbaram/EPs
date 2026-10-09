import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
import lightgbm as lgb
import shutil

def train_cost_sensitive_reentry():
    print("🚀 TRAINING COST-SENSITIVE TRADE 2/3 RE-ENTRY MULTI-LEG CLASSIFIER")
    parquet_file = Path("data/ml_datasets/reentry/dataset_reentry.parquet")
    if not parquet_file.exists():
        print(f"Dataset {parquet_file} not found.")
        return
        
    df = pd.read_parquet(parquet_file)
    df = df.dropna(subset=["target_15pct_win"]).copy()
    features = [c for c in df.columns if c.startswith("feature_")]
    print(f"Training on {len(df)} reentry samples using {len(features)} features...")
    
    # Chronological sort for panel data to avoid temporal leakage
    df["date_idx"] = pd.to_datetime(df["date"])
    df = df.sort_values("date_idx")
    
    # Inverse-duration weighting + Asymmetric Winner Scaling
    df["event_id"] = df["symbol"] + "_" + df["entry_date"]
    event_counts = df.groupby("event_id")["event_id"].transform("count")
    
    # Asymmetric sample weight:
    # For winners (target_15pct_win == 1), scale sample weight by (forward_max_20d / 0.10) with minimum 1.0 and cap at 10.0
    win_bonus = np.clip(df["forward_max_20d"] / 0.10, 1.0, 10.0)
    df["sample_weight"] = (1.0 / event_counts) * np.where(df["target_15pct_win"] == 1, win_bonus, 1.0)
    
    # In-sample (train) < 2023-01-01, Out-of-sample >= 2023-01-01
    train_df = df[df["date_idx"] < pd.Timestamp("2023-01-01")].copy()
    test_df = df[df["date_idx"] >= pd.Timestamp("2023-01-01")].copy()
    
    print(f"In-sample train: {len(train_df)}, Out-of-sample test: {len(test_df)}")
    
    scaler = StandardScaler()
    X_train_tf = scaler.fit_transform(train_df[features])
    y_train = train_df["target_15pct_win"]
    w_train = train_df["sample_weight"]
    
    split_idx = int(len(train_df) * 0.8)
    X_base = X_train_tf[:split_idx]
    y_base = y_train.iloc[:split_idx]
    w_base = w_train.iloc[:split_idx]
    
    X_calib = X_train_tf[split_idx:]
    y_calib = y_train.iloc[split_idx:]
    w_calib = w_train.iloc[split_idx:]
    
    # Base LightGBM Classifier
    lgb_base = lgb.LGBMClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.05, 
        class_weight='balanced', random_state=42, verbose=-1
    )
    lgb_base.fit(X_base, y_base, sample_weight=w_base)
    
    # Calibrated Classifier on unseen in-sample holdout
    calibrated_model = CalibratedClassifierCV(estimator=FrozenEstimator(lgb_base), method='isotonic')
    calibrated_model.fit(X_calib, y_calib, sample_weight=w_calib)
    
    # Test set metrics
    X_test_tf = scaler.transform(test_df[features])
    y_test = test_df["target_15pct_win"]
    preds_test = calibrated_model.predict_proba(X_test_tf)[:, 1]
    
    big_winners = (test_df["target_15pct_win"] == 1) & (test_df["forward_max_20d"] >= 0.30)
    recall_big = (preds_test[big_winners] >= 0.40).mean()
    print(f"OOS Big Winners (>=30% MFE) Recall at 0.40 cutoff: {recall_big*100:.1f}%")
    print(f"OOS Mean Prob on Big Winners: {preds_test[big_winners].mean():.3f} vs Losers: {preds_test[y_test == 0].mean():.3f}")
    
    serving_pipeline = Pipeline([("scaler", scaler), ("classifier", calibrated_model)])
    
    out_path = Path("data/models/ep_reentry_classifier.pkl")
    backup_path = Path("data/models/ep_reentry_classifier_v1_backup.pkl")
    if out_path.exists() and not backup_path.exists():
        shutil.copy(out_path, backup_path)
        print(f"Backed up old model to {backup_path}")
        
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(serving_pipeline, out_path)
    print(f"✅ Successfully Saved Cost-Sensitive Re-Entry Model to {out_path}")

if __name__ == "__main__":
    train_cost_sensitive_reentry()
