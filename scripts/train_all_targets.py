import json
import joblib
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

def train_all():
    print("🚀 TRAINING ALL THRESHOLD MODELS (+50%, +100%, +150%, +200%)")
    
    data_dir = Path("data/ml_datasets")
    folders = sorted([f for f in data_dir.glob("ep_dataset_daily_v4_*") if f.is_dir()])
    parquet_file = folders[-1] / "dataset.parquet"
    df = pd.read_parquet(parquet_file)
    
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    targets = [50, 100, 150, 200]
    
    for t in targets:
        print(f"\n[*] Training Model for Target: +{t}%")
        
        # 1. Filter unresolved panels for THIS specific target
        df_t = df[df[f"feature_chk_reached_{t}"] == 0].copy()
        
        # 2. Drop censored outcomes
        df_t = df_t[df_t[f"label_reach_{t}"].notna()].copy()
        df_t["target"] = df_t[f"label_reach_{t}"].astype(int)
        
        print(f"    -> Valid Rows: {len(df_t)} | Hits: {df_t['target'].sum()} | Base Rate: {df_t['target'].mean():.2%}")
        
        # 3. Features
        features = [c for c in df.columns if c.startswith("feature_")]
        num_features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
        cat_features = [c for c in features if c not in num_features]
        
        preprocessor = ColumnTransformer([
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
        ])
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, 
            random_state=42, use_label_encoder=False, eval_metric="logloss"
        )
        
        calibrated_xgb = CalibratedClassifierCV(xgb_model, method="isotonic", cv=5)
        
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", calibrated_xgb)
        ])
        
        # 4. Train
        X = df_t[features]
        y = df_t["target"]
        pipeline.fit(X, y)
        
        # 5. Save
        model_path = models_dir / f"ep_xgboost_{t}_prod.pkl"
        joblib.dump(pipeline, model_path)
        print(f"    -> ✅ Saved to {model_path}")

if __name__ == "__main__":
    train_all()
