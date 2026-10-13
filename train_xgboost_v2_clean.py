import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

def train_v2_clean():
    print("🚀 TRAINING V2 ADVANCED INSIGHTS MODELS (CLEANED - NO LONG HISTORY REQS)")
    
    parquet_file = Path("data/ml_datasets/amir_spec/dataset_v2.parquet")
    df = pd.read_parquet(parquet_file)
    df["date"] = pd.to_datetime(df["entry_date"])
    
    train_df = df[df["date"] < pd.Timestamp("2023-01-01")].copy()
    test_df = df[df["date"] >= pd.Timestamp("2023-01-01")].copy()
    
    # We drop the features that require too much history and hurt the model
    drop_features = ["feature_dist_200sma", "feature_neglect_6m"]
    
    features = [c for c in df.columns if c.startswith("feature_") and c not in drop_features]
    num_features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat_features = [c for c in features if c not in num_features]
    
    preprocessor = ColumnTransformer([
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
    ])
    
    y_train = train_df["target_100"]
    y_test = test_df["target_100"]
    
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos = num_neg / num_pos
    
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
    
    probs = pipeline.predict_proba(test_df[features])[:, 1]
    auc = roc_auc_score(y_test, probs)
    ap = average_precision_score(y_test, probs)
    
    print(f"  [OOS Test Performance (2023+)]")
    print(f"  ROC-AUC: {auc:.3f}")
    print(f"  PR-AUC: {ap:.3f}")

if __name__ == "__main__":
    train_v2_clean()
