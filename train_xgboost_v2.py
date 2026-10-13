import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

def train_v2_models():
    print("🚀 TRAINING V2 ADVANCED INSIGHTS MODELS")
    
    parquet_file = Path("data/ml_datasets/amir_spec/dataset_v2.parquet")
    if not parquet_file.exists():
        print("Dataset not found!")
        return
        
    df = pd.read_parquet(parquet_file)
    print(f"[*] Total Day-5 Entry Events: {len(df)}")
    
    df["date"] = pd.to_datetime(df["entry_date"])
    
    # Use exact same chronological split as V1
    train_df = df[df["date"] < pd.Timestamp("2023-01-01")].copy()
    test_df = df[df["date"] >= pd.Timestamp("2023-01-01")].copy()
    
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
        print(f"[*] Training 'V2 Advanced Ranker' for Target: +{t}%")
        
        y_train = train_df[f"target_{t}"]
        y_test = test_df[f"target_{t}"]
        
        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        if num_pos == 0: continue
        scale_pos = num_neg / num_pos
        print(f"  Base Rate (Train): {y_train.mean():.2%} | Applied Up-weight scale: {scale_pos:.1f}x")
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05, 
            scale_pos_weight=scale_pos,
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
            print(f"  Test Base Rate: {y_test.mean():.2%}")
            print(f"  Top 10% Alpha Rank Win Rate: {top_decile[f'target_{t}'].mean():.2%}")
            print(f"  ROC-AUC: {auc:.3f}")
            print(f"  PR-AUC: {ap:.3f}")
            
        out_path = Path(f"data/models/amir_spec_xgboost_v2_{t}.pkl")
        joblib.dump(pipeline, out_path)
        print(f"  ✅ Saved V2 Ranker to {out_path}")

if __name__ == "__main__":
    train_v2_models()
