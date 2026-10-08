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
    print("🚀 TRAINING AMIR ORIGINAL SPEC ORDINAL MODEL (DAY-5 ENTRY)")
    
    parquet_file = Path("data/ml_datasets/amir_spec/dataset.parquet")
    if not parquet_file.exists():
        print("Dataset not found!")
        return
        
    df = pd.read_parquet(parquet_file)
    print(f"[*] Total Day-5 Entry Events: {len(df)}")
    
    df["date_idx"] = pd.to_datetime(df["entry_date"])
    
    # Drop rows that are right-censored (have NaNs in the target column)
    # Only drop if the trade didn't finish its first milestone
    df = df.dropna(subset=["target_50"])
    print(f"[*] Total Vectors after dropping right-censored paths: {len(df)}")

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
    
    print(f"[*] Training 'Alpha Ranker Ordinal Predictor'")
    print(f"  Class 0 (Failed):  {(y_train == 0).sum()} ({((y_train == 0).mean()*100):.1f}%)")
    print(f"  Class 1 (+50%):    {(y_train == 1).sum()} ({((y_train == 1).mean()*100):.1f}%)")
    print(f"  Class 2 (+100%):   {(y_train == 2).sum()} ({((y_train == 2).mean()*100):.1f}%)")
    print(f"  Class 3 (+150%):   {(y_train == 3).sum()} ({((y_train == 3).mean()*100):.1f}%)")
    print(f"  Class 4 (+200%):   {(y_train == 4).sum()} ({((y_train == 4).mean()*100):.1f}%)")
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.05, 
        random_state=42, use_label_encoder=False,
        objective="multi:softprob",
        num_class=5
    )
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", xgb_model)
    ])
    
    calibrated_model = CalibratedClassifierCV(
        estimator=pipeline, 
        method='isotonic', 
        cv=5 
    )
    
    calibrated_model.fit(train_df[features], y_train)
    
    if len(test_df) > 0:
        probs = calibrated_model.predict_proba(test_df[features])
        
        import numpy as np
        prob_50 = np.clip(probs[:, 1:].sum(axis=1), 0, 1)
        prob_100 = np.clip(probs[:, 2:].sum(axis=1), 0, 1)
        prob_150 = np.clip(probs[:, 3:].sum(axis=1), 0, 1)
        prob_200 = np.clip(probs[:, 4], 0, 1)
        
        y_test_50 = (y_test >= 1).astype(int)
        y_test_100 = (y_test >= 2).astype(int)
        y_test_150 = (y_test >= 3).astype(int)
        y_test_200 = (y_test >= 4).astype(int)
        
        print(f"  [OOS Test Performance (2023+) | Brier Scores]")
        print(f"  +50% Target:  Brier={brier_score_loss(y_test_50, prob_50):.4f} | ROC-AUC={roc_auc_score(y_test_50, prob_50):.3f}")
        print(f"  +100% Target: Brier={brier_score_loss(y_test_100, prob_100):.4f} | ROC-AUC={roc_auc_score(y_test_100, prob_100):.3f}")
        print(f"  +150% Target: Brier={brier_score_loss(y_test_150, prob_150):.4f} | ROC-AUC={roc_auc_score(y_test_150, prob_150):.3f}")
        print(f"  +200% Target: Brier={brier_score_loss(y_test_200, prob_200):.4f} | ROC-AUC={roc_auc_score(y_test_200, prob_200):.3f}")
        
    out_path = Path("data/models/amir_spec_xgboost_ordinal.pkl")
    joblib.dump(calibrated_model, out_path)
    print(f"  ✅ Saved Ordinal Calibrated Model to {out_path}")

if __name__ == "__main__":
    train_ordinal_model()
