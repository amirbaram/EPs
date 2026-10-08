import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
import lightgbm as lgb
import os

def train_reentry_model():
    print("🚀 TRAINING RE-ENTRY ML MODEL (TRADE 2 & 3)")
    df = pd.read_parquet("data/ml_datasets/reentry/dataset_reentry.parquet")
    
    features = [c for c in df.columns if c.startswith("feature_")]
    
    # Chronological sort for panel data to avoid leakage
    df["date_idx"] = pd.to_datetime(df["date"])
    df = df.sort_values("date_idx")
    
    # Inverse-duration weighting to prevent survivorship bias from long pullbacks
    df["event_id"] = df["symbol"] + "_" + df["entry_date"]
    event_counts = df.groupby("event_id").size()
    df["sample_weight"] = df["event_id"].map(event_counts).apply(lambda x: 1.0 / x)
    
    train_df = df[df["date_idx"] < pd.Timestamp("2023-01-01")].copy()
    test_df = df[df["date_idx"] >= pd.Timestamp("2023-01-01")].copy()
    
    scaler = StandardScaler()
    
    X_train_tf = scaler.fit_transform(train_df[features])
    y_train = train_df["target_15pct_win"]
    w_train = train_df["sample_weight"]
    
    X_test_tf = scaler.transform(test_df[features])
    y_test = test_df["target_15pct_win"]
    
    # Base Model
    lgb_base = lgb.LGBMClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.05, 
        class_weight='balanced', random_state=42
    )
    
    lgb_base.fit(X_train_tf, y_train, sample_weight=w_train)
    
    # Calibrate
    calibrated_model = CalibratedClassifierCV(estimator=FrozenEstimator(lgb_base), method='isotonic', )
    calibrated_model.fit(X_test_tf, y_test)
    
    serving_pipeline = Pipeline([
        ("scaler", scaler),
        ("classifier", calibrated_model)
    ])
    
    os.makedirs("data/models", exist_ok=True)
    joblib.dump(serving_pipeline, "data/models/ep_reentry_classifier.pkl")
    print("✅ Saved Re-Entry Classifier.")
    
    # Evaluate
    preds = serving_pipeline.predict_proba(X_test_tf)[:, 1]
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, preds)
    print(f"Test AUC: {auc:.3f}")

if __name__ == "__main__":
    train_reentry_model()
