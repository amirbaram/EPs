import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import lightgbm as lgb
from sklearn.base import clone

def train_advanced_exits():
    print("🚀 TRAINING ADVANCED ML EXITS (DYNAMIC STOPLOSS & EXHAUSTION)")
    parquet_file = Path("data/ml_datasets/amir_rolling/dataset_rolling.parquet")
    if not parquet_file.exists(): return
    
    df = pd.read_parquet(parquet_file)

    # [FIX]: Calculate inverse-duration weights
    df["date_idx"] = pd.to_datetime(df["date"])
    if "entry_date" in df.columns:
        df["event_id"] = df["symbol"] + "_" + df["entry_date"]
    else:
        df["event_id"] = df["symbol"] + "_" + (df["date_idx"] - pd.to_timedelta(df["feature_days_since_ep"], unit='D')).dt.strftime("%Y-%m-%d")
        
    event_counts = df.groupby("event_id")["event_id"].transform("count")
    df["sample_weight"] = 1.0 / event_counts

    features = [c for c in df.columns if c.startswith("feature_")]
    num_features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat_features = [c for c in features if c not in num_features]
    
    preprocessor_base = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
    ])
    
    out_dir = Path("data/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Chronological sort & train split to eliminate leakage
    df = df.sort_values("date_idx")
    train_df = df[df["date_idx"] < pd.Timestamp("2023-01-01")].copy()
    test_df = df[df["date_idx"] >= pd.Timestamp("2023-01-01")].copy()
    print(f"In-sample train: {len(train_df)}, Out-of-sample test: {len(test_df)}")

    # MODEL 1: EXHAUSTION CLASSIFIER (Only on runners +20%)
    print("Training Exhaustion Classifier...")
    preprocessor_exh = clone(preprocessor_base)
    train_runners = train_df[train_df["feature_ret_since_ep"] >= 0.20].copy()
    test_runners = test_df[test_df["feature_ret_since_ep"] >= 0.20].copy()
    if "label_exhaustion" in train_runners.columns and len(train_runners) > 0:
        X_train_tf = preprocessor_exh.fit_transform(train_runners[features])
        y_exh = train_runners["label_exhaustion"]
        w_exh = train_runners["sample_weight"]
        
        lgb_exhaustion = lgb.LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            class_weight='balanced', random_state=42, verbose=-1
        )
        lgb_exhaustion.fit(X_train_tf, y_exh, sample_weight=w_exh)
        
        pipe_exh = Pipeline([("preprocessor", preprocessor_exh), ("classifier", lgb_exhaustion)])
        if len(test_runners) > 0:
            from sklearn.metrics import roc_auc_score
            y_test_exh = test_runners["label_exhaustion"]
            probs_exh = pipe_exh.predict_proba(test_runners[features])[:, 1]
            print(f"  OOS Exhaustion ROC-AUC (2023+): {roc_auc_score(y_test_exh, probs_exh):.3f}")
            
        joblib.dump(pipe_exh, out_dir / "ep_exhaustion_classifier.pkl")
        print("  ✅ Saved Exhaustion Classifier")
    
    # MODEL 2: DYNAMIC STOPLOSS REGRESSOR (Point-in-Time Trailing Stop on active runners, ret >= 0.05)
    print("Training Dynamic Trailing Stop Regressor...")
    preprocessor_stop = clone(preprocessor_base)
    # Train causally on all active runner states (unrealized gain >= 5%), without target_50 lookahead conditioning
    train_runners = train_df[train_df["feature_ret_since_ep"] >= 0.05].dropna(subset=["forward_mae_pct"]).copy()
    test_runners = test_df[test_df["feature_ret_since_ep"] >= 0.05].dropna(subset=["forward_mae_pct"]).copy()
    
    if len(train_runners) > 0:
        X_stop_tf = preprocessor_stop.fit_transform(train_runners[features])
        y_stop = train_runners["forward_mae_pct"]
        w_stop = train_runners["sample_weight"]
        
        lgb_stop = lgb.LGBMRegressor(
            objective="quantile", alpha=0.10,
            n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1
        )
        lgb_stop.fit(X_stop_tf, y_stop, sample_weight=w_stop)
        
        pipe_stop = Pipeline([("preprocessor", preprocessor_stop), ("regressor", lgb_stop)])
        if len(test_runners) > 0:
            pred_stop = pipe_stop.predict(test_runners[features])
            mae_diff = (test_runners["forward_mae_pct"] - pred_stop).abs().mean()
            print(f"  OOS Dynamic Stop 10th %ile MAE Error: {mae_diff:.4f}")
            
        joblib.dump(pipe_stop, out_dir / "ep_dynamic_trailer.pkl")
        print("  ✅ Saved Dynamic Trailing Stop Regressor")

if __name__ == "__main__":
    train_advanced_exits()
