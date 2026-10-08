import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

def train_rolling_models():
    print("🚀 TRAINING ROLLING MULTI-DAY MODELS (DAY 1 TO DAY 20)")
    
    parquet_file = Path("data/ml_datasets/amir_rolling/dataset_rolling.parquet")
    if not parquet_file.exists():
        print("Dataset not found!")
        return
        
    df = pd.read_parquet(parquet_file)
    print(f"[*] Total Rolling Daily Evaluation Vectors: {len(df)}")
    
    df["date_idx"] = pd.to_datetime(df["date"])
    
    train_df = df[df["date_idx"] < pd.Timestamp("2023-01-01")].copy()
    test_df = df[df["date_idx"] >= pd.Timestamp("2023-01-01")].copy()
    
    features = [c for c in df.columns if c.startswith("feature_")]
    num_features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat_features = [c for c in features if c not in num_features]
    
    preprocessor = ColumnTransformer([
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
    ])
    
    targets = [50, 100, 150, 200]
    
    for t in targets:
        print(f"\n===========================================")
        print(f"[*] Training 'Rolling Predictor' for Target: +{t}%")
        
        y_train = train_df[f"target_{t}"]
        y_test = test_df[f"target_{t}"]
        
        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        if num_pos == 0: continue
        scale_pos = num_neg / num_pos
        print(f"  Base Rate (Train): {y_train.mean():.2%} | Applied Up-weight scale: {scale_pos:.1f}x")
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.05, 
            
            random_state=42, use_label_encoder=False, eval_metric="logloss"
        )
        
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", xgb_model)
        ])
        
        pipeline.fit(train_df[features], y_train)
        
        if len(test_df) > 0:
            probs = pipeline.predict_proba(test_df[features])[:, 1]
            test_df[f"prob_{t}"] = probs
            
            threshold = np.percentile(probs, 90)
            top_decile = test_df[test_df[f"prob_{t}"] >= threshold]
            
            auc = roc_auc_score(y_test, probs)
            ap = average_precision_score(y_test, probs)
            
            print(f"  [OOS Test Performance (2023+)]")
            print(f"  ROC-AUC: {auc:.3f}")
            print(f"  PR-AUC: {ap:.3f}")
            
        out_path = Path(f"data/models/amir_rolling_xgboost_{t}.pkl")
        joblib.dump(pipeline, out_path)
        print(f"  ✅ Saved Rolling Model to {out_path}")

if __name__ == "__main__":
    train_rolling_models()
