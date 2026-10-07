import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json

def analyze_higher_targets():
    print("🚀 DEEP DIVE: HIGHER TARGETS (+100%, +150%, +200%)")
    
    data_dir = Path("data/ml_datasets")
    folders = sorted([f for f in data_dir.glob("ep_dataset_daily_v4_*") if f.is_dir()])
    parquet_file = folders[-1] / "dataset.parquet"
    df_raw = pd.read_parquet(parquet_file)
    
    with open("data/models/feature_schema.json", "r") as f:
        schema = json.load(f)
    features = schema["features_ordered"]
    
    targets = [100, 150, 200]
    
    for t in targets:
        print(f"\n===========================================")
        print(f"[*] Analyzing Target: +{t}%")
        
        # 1. Filter unresolved panels for THIS target
        df = df_raw[df_raw[f"feature_chk_reached_{t}"] == 0].copy()
        
        # 2. Drop censored outcomes
        df = df[df[f"label_reach_{t}"].notna()].copy()
        df["target"] = df[f"label_reach_{t}"].astype(int)
        
        # 3. Test Set (OOS)
        df["as_of_date"] = pd.to_datetime(df["identifier_as_of_date"])
        df_test = df[df["as_of_date"] >= pd.Timestamp("2024-01-01")].copy()
        
        if len(df_test) == 0:
            print("  Not enough test data.")
            continue
            
        # 4. Predict
        model_path = f"data/models/ep_xgboost_{t}_prod.pkl"
        if not Path(model_path).exists():
            print(f"  Model {model_path} not found.")
            continue
            
        model = joblib.load(model_path)
        df_test["predicted_prob"] = model.predict_proba(df_test[features])[:, 1]
        
        # 5. Top 10% Analysis
        threshold = np.percentile(df_test["predicted_prob"], 90)
        top_decile = df_test[df_test["predicted_prob"] >= threshold].copy()
        
        print(f"  Total Test Rows: {len(df_test)} | Base Rate: {df_test['target'].mean():.2%}")
        print(f"  Top 10% Threshold: Prob >= {threshold:.4f}")
        print(f"  Top 10% Win Rate (Accuracy): {top_decile['target'].mean():.2%}")
        
        # 6. Latency Analysis
        avg_age = top_decile['feature_age_sessions'].mean()
        avg_ret = top_decile['feature_chk_return_since_event_pct'].mean()
        
        print(f"\n  [Latency Profile of Top 10%]")
        print(f"  - Avg Age when Signal Fires: {avg_age:.1f} days")
        print(f"  - Avg Return already achieved: +{avg_ret:.1f}%")
        
        up_40_plus = top_decile[top_decile['feature_chk_return_since_event_pct'] >= 40]
        print(f"  - Signals firing when ALREADY UP > 40%: {len(up_40_plus)/len(top_decile):.1%}")

if __name__ == "__main__":
    analyze_higher_targets()
