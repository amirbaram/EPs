import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_ep_toxicity_filter():
    print("🚀 TRAINING ISOLATION FOREST TOXICITY FILTER (UNSUPERVISED)")
    
    df = pd.read_parquet("data/ml_datasets/amir_spec/dataset.parquet")
    
    # Select purely Day-1 descriptive characteristics
    features = [
        "feature_gap_pct", "feature_rvol", "feature_close_pos", 
        "feature_is_novel_vol_9m"
    ]
    
    X_train = df[features].fillna(0)
    
    # contamination=0.03 assumes the 3% weirdest setups historically were toxic
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('iso_forest', IsolationForest(n_estimators=200, contamination=0.03, random_state=42))
    ])
    
    pipeline.fit(X_train)
    out_path = Path("data/models/ep_toxicity_filter.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)
    
    preds = pipeline.predict(X_train)
    print(f"[*] Flagged {(preds == -1).sum()} historical setups as toxic flow.")

if __name__ == "__main__":
    train_ep_toxicity_filter()
