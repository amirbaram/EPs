import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import lightgbm as lgb

def train_advanced_exits():
    print("🚀 TRAINING ADVANCED EXITS (PARTIALS & TRAILING STOPS)")
    df = pd.read_parquet("data/ml_datasets/amir_rolling/dataset_rolling.parquet")
    
    features = [c for c in df.columns if c.startswith("feature_")]
    num_features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat_features = [c for c in features if c not in num_features]
    
    preprocessor = ColumnTransformer([
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
    ])

    # ==========================================
    # MODEL 1: EXHAUSTION CLASSIFIER (PARTIALS)
    # ==========================================
    print("\n[*] Training Peak Exhaustion Classifier...")
    df_runners = df[df["feature_ret_since_ep"] >= 0.30].copy()
    if "label_exhaustion" in df_runners.columns:
        X_train_tf = preprocessor.fit_transform(df_runners[features])
        y_exh = df_runners["label_exhaustion"]
        
        lgb_exhaustion = lgb.LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05, 
            class_weight='balanced', random_state=42
        )
        lgb_exhaustion.fit(X_train_tf, y_exh)
        
        pipe_exh = Pipeline([("preprocessor", preprocessor), ("classifier", lgb_exhaustion)])
        joblib.dump(pipe_exh, "data/models/ep_exhaustion_classifier.pkl")
        print("  ✅ Saved Exhaustion Classifier")

    # ==========================================
    # MODEL 2: DYNAMIC TRAILING STOP (HAZARD)
    # ==========================================
    print("\n[*] Training Dynamic Trailing Stop (Trend Hazard)...")
    df_winners = df[df["target_50"] == 1].copy()
    df_winners = df_winners.dropna(subset=["forward_mae_pct"])
    
    if "forward_mae_pct" in df_winners.columns and len(df_winners) > 0:
        X_stop_tf = preprocessor.fit_transform(df_winners[features])
        y_stop = df_winners["forward_mae_pct"]
        
        # Quantile regression to predict the 10th percentile bound
        lgb_stop = lgb.LGBMRegressor(
            objective='quantile', alpha=0.10, 
            n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42
        )
        lgb_stop.fit(X_stop_tf, y_stop)
        
        pipe_stop = Pipeline([("preprocessor", preprocessor), ("regressor", lgb_stop)])
        joblib.dump(pipe_stop, "data/models/ep_dynamic_trailer.pkl")
        print("  ✅ Saved Dynamic Trailing Stop Regressor")

if __name__ == "__main__":
    train_advanced_exits()
