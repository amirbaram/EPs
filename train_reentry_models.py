import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    pass

def train_reentry_models():
    print("🚀 TRAINING TRADE 2/3 RE-ENTRY MULTI-LEG CLASSIFIER")
    parquet_file = Path("data/ml_datasets/reentry/dataset_reentry.parquet")
    if not parquet_file.exists():
        print(f"Dataset {parquet_file} not found. Run build_reentry_dataset.py first.")
        return
        
    df = pd.read_parquet(parquet_file)
    df = df.dropna(subset=["target_15pct_win"])
    features = [c for c in df.columns if c.startswith("feature_")]
    print(f"Training on {len(df)} reentry pullback samples using {len(features)} features...")
    
    # Chronological sort for panel data to avoid leakage
    df["date_idx"] = pd.to_datetime(df["date"])
    df = df.sort_values("date_idx")
    
    # Inverse-duration weighting
    df["event_id"] = df["symbol"] + "_" + df["entry_date"]
    event_counts = df.groupby("event_id")["event_id"].transform("count")
    df["sample_weight"] = 1.0 / event_counts
    
    train_df = df[df["date_idx"] < pd.Timestamp("2023-01-01")].copy()
    test_df = df[df["date_idx"] >= pd.Timestamp("2023-01-01")].copy()
    
    scaler = StandardScaler()
    X_train_tf = scaler.fit_transform(train_df[features])
    y_train = train_df["target_15pct_win"]
    w_train = train_df["sample_weight"]
    
    # [FIX]: Chronological Split for calibrator to prevent Test-Set Leakage
    split_idx = int(len(train_df) * 0.8)
    X_base = X_train_tf[:split_idx]
    y_base = y_train.iloc[:split_idx]
    w_base = w_train.iloc[:split_idx]
    
    X_calib = X_train_tf[split_idx:]
    y_calib = y_train.iloc[split_idx:]
    w_calib = w_train.iloc[split_idx:]
    
    X_test_tf = scaler.transform(test_df[features])
    y_test = test_df["target_15pct_win"]
    
    # Base Model
    lgb_base = lgb.LGBMClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.05, 
        class_weight='balanced', random_state=42
    )
    lgb_base.fit(X_base, y_base, sample_weight=w_base)
    
    # Calibrate on unseen training holdout
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated_model = CalibratedClassifierCV(estimator=FrozenEstimator(lgb_base), method='isotonic')
    except ImportError:
        calibrated_model = CalibratedClassifierCV(estimator=lgb_base, method='isotonic', cv="prefit")
        
    calibrated_model.fit(X_calib, y_calib, sample_weight=w_calib)
    
    serving_pipeline = Pipeline([("scaler", scaler), ("classifier", calibrated_model)])
    
    out_path = Path("data/models/ep_reentry_classifier.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(serving_pipeline, out_path)
    print(f"✅ Saved Trade 2/3 Re-Entry Model to {out_path}")

if __name__ == "__main__":
    train_reentry_models()
