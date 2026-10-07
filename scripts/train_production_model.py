import os
import glob
import json
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

def run_production_training():
    print("🚀 STAGE 3: PRODUCTION MODEL TRAINING")
    
    # 1. Find the latest dataset
    data_dir = Path("data/ml_datasets")
    folders = sorted([f for f in data_dir.glob("ep_dataset_daily_v4_*") if f.is_dir()])
    if not folders:
        print("[!] No dataset found. Run Stage 1 first.")
        return
    
    latest_folder = folders[-1]
    parquet_file = latest_folder / "dataset.parquet"
    print(f"[*] Loading dataset: {latest_folder.name}")
    
    df = pd.read_parquet(parquet_file)
    print(f"[*] Raw rows: {len(df)}")
    
    # 2. Filter unresolved panels for +50% target
    df = df[df["feature_chk_reached_50"] == 0].copy()
    
    # 3. Target setup: train on all valid (resolved) history
    df = df[df["label_reach_50"].notna()].copy()
    df["target"] = df["label_reach_50"].astype(int)
    print(f"[*] Production training rows (Full History): {len(df)}")
    
    # 4. Feature Schema extraction
    features = [c for c in df.columns if c.startswith("feature_")]
    num_features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat_features = [c for c in features if c not in num_features]
    
    schema = {
        "features_ordered": features,
        "num_features": num_features,
        "cat_features": cat_features
    }
    
    # 5. Build the Champion Pipeline (XGBoost)
    preprocessor = ColumnTransformer([
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
    ])
    
    # Base XGBoost model from Stage 2
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, 
        max_depth=4, 
        learning_rate=0.05, 
        random_state=42, 
        use_label_encoder=False, 
        eval_metric="logloss"
    )
    
    # Wrap in CalibratedClassifierCV for perfectly scaled production probabilities
    # We use cv=5 to train across all data without wasting a holdout set
    calibrated_xgb = CalibratedClassifierCV(xgb_model, method="isotonic", cv=5)
    
    prod_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", calibrated_xgb)
    ])
    
    # 6. Fit the Model
    print("[*] Fitting Calibrated XGBoost on entire historical dataset...")
    X = df[features]
    y = df["target"]
    prod_pipeline.fit(X, y)
    
    # 7. Export Artifacts
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / "ep_xgboost_prod.pkl"
    schema_path = models_dir / "feature_schema.json"
    
    joblib.dump(prod_pipeline, model_path)
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
        
    print(f"[*] ✅ Model successfully exported to: {model_path}")
    print(f"[*] ✅ Feature schema exported to: {schema_path}")
    print("🚀 PRODUCTION TRAINING COMPLETE. The model is ready for live inference.")

if __name__ == "__main__":
    run_production_training()
