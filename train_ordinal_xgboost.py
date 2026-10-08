import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

def train_ordinal_model():
    print("🚀 TRAINING ROLLING MULTI-DAY ORDINAL MODEL (DAY 1 TO DAY 20)")
    
    parquet_file = Path("data/ml_datasets/amir_rolling/dataset_rolling.parquet")
    if not parquet_file.exists():
        print("Dataset not found!")
        return
        
    df = pd.read_parquet(parquet_file)
    print(f"[*] Total Rolling Daily Evaluation Vectors: {len(df)}")
    
    df["date_idx"] = pd.to_datetime(df["date"])
    
    df = df.dropna(subset=["target_50"])
    print(f"[*] Total Vectors after dropping right-censored paths: {len(df)}")

    # [FIX]: Eliminate panel-data Duration/Survivorship bias via inverse sample weighting
    df["event_id"] = df["symbol"] + "_" + df["entry_date"]
    event_counts = df.groupby("event_id").size()
    df["sample_weight"] = df["event_id"].map(event_counts).apply(lambda x: 1.0 / x)

    # Create Ordinal Class (Max Tier Reached)
    df["ordinal_class"] = 0
    df.loc[df["target_50"] == 1, "ordinal_class"] = 1
    df.loc[df["target_100"] == 1, "ordinal_class"] = 2
    df.loc[df["target_150"] == 1, "ordinal_class"] = 3
    df.loc[df["target_200"] == 1, "ordinal_class"] = 4
    
    train_df = df[df["date_idx"] < pd.Timestamp("2023-01-01")].copy()
    test_df = df[df["date_idx"] >= pd.Timestamp("2023-01-01")].copy()
    
    features = [c for c in df.columns if c.startswith("feature_")]
    num_features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat_features = [c for c in features if c not in num_features]
    
    preprocessor = ColumnTransformer([
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
    ])
    
    y_train = train_df["ordinal_class"]
    y_test = test_df["ordinal_class"]
    w_train = train_df["sample_weight"]
    
    # [FIX]: Transform data FIRST so CalibratedClassifierCV can natively slice sample_weight
    X_train_tf = preprocessor.fit_transform(train_df[features])
    if len(test_df) > 0:
        X_test_tf = preprocessor.transform(test_df[features])
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.05, 
        random_state=42, use_label_encoder=False,
        objective="multi:softprob",
        num_class=5
    )
    
    calibrated_model = CalibratedClassifierCV(
        estimator=xgb_model, 
        method='isotonic', 
        cv=5 
    )
    
    calibrated_model.fit(X_train_tf, y_train, sample_weight=w_train)
    
    # Repackage into a seamless pipeline for inference
    from sklearn.pipeline import Pipeline
    serving_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", calibrated_model)
    ])
    
    if len(test_df) > 0:
        probs = serving_pipeline.predict_proba(test_df[features])
        
        import numpy as np
        prob_50 = np.clip(probs[:, 1:].sum(axis=1), 0, 1)
        prob_100 = np.clip(probs[:, 2:].sum(axis=1), 0, 1)
        prob_150 = np.clip(probs[:, 3:].sum(axis=1), 0, 1)
        prob_200 = np.clip(probs[:, 4], 0, 1)
        
        # ... (Metrics printing remains identical) ...
        
    out_path = Path("data/models/amir_rolling_xgboost_ordinal.pkl")
    import joblib
    joblib.dump(serving_pipeline, out_path)
    print(f"  ✅ Saved Ordinal Calibrated Model to {out_path}")

if __name__ == "__main__":
    train_ordinal_model()
