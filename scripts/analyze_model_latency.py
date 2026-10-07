import pandas as pd
import numpy as np
import joblib
from pathlib import Path

def analyze_latency():
    print("🚀 ANALYZING MODEL PREDICTION LATENCY...")
    
    # 1. Load dataset
    data_dir = Path("data/ml_datasets")
    folders = sorted([f for f in data_dir.glob("ep_dataset_daily_v4_*") if f.is_dir()])
    parquet_file = folders[-1] / "dataset.parquet"
    df = pd.read_parquet(parquet_file)
    
    # 2. Filter exactly as we did for Test Evaluation (OOS: 2024+)
    df = df[df["feature_chk_reached_50"] == 0].copy()
    df = df[df["label_reach_50"].notna()].copy()
    df["as_of_date"] = pd.to_datetime(df["identifier_as_of_date"])
    df = df[df["as_of_date"] >= pd.Timestamp("2024-01-01")].copy()
    
    # 3. Load Model and Schema
    model = joblib.load("data/models/ep_xgboost_prod.pkl")
    import json
    with open("data/models/feature_schema.json", "r") as f:
        schema = json.load(f)
    features = schema["features_ordered"]
    
    # 4. Predict
    df["predicted_prob"] = model.predict_proba(df[features])[:, 1]
    
    # 5. Extract Top Decile
    threshold = np.percentile(df["predicted_prob"], 90)
    top_decile = df[df["predicted_prob"] >= threshold].copy()
    
    # 6. Analyze "Current Return" at the time of prediction
    # If the target is +50% from audit_entry_price, how far along are we?
    # Actually, the feature row represents the state at the END of that session.
    # What was the close on that day? It should be in `feature_close` or we can approximate.
    # Wait, the dataset might not have `feature_close`. Let's check available price features.
    price_cols = [c for c in df.columns if "close" in c or "price" in c]
    # audit_entry_price, audit_target_price, audit_stop_price are available.
    
    # We can infer how far it has moved if we know the daily close, but let's check `feature_age_sessions` first.
    print(f"Top 10% Threshold: {threshold:.4f}")
    print(f"Total Top 10% Signals: {len(top_decile)}")
    
    age_dist = top_decile["feature_age_sessions"].value_counts().sort_index()
    print("\n[+] Top 10% Signals by EP Age (Sessions since event):")
    for age in range(1, 11):
        count = age_dist.get(age, 0)
        pct = count / len(top_decile)
        print(f"  Day {age}: {count} signals ({pct:.1%})")
        
    print(f"  Day 11-20: {age_dist[(age_dist.index > 10) & (age_dist.index <= 20)].sum()} signals")
    print(f"  Day 21+: {age_dist[age_dist.index > 20].sum()} signals")
    
    # Let's check what the base rate is for Day 1
    day1_all = df[df["feature_age_sessions"] == 1]
    day1_top = top_decile[top_decile["feature_age_sessions"] == 1]
    print(f"\n[+] Day 1 Specifics:")
    print(f"  Total Day 1 EPs in Test Set: {len(day1_all)}")
    print(f"  How many Day 1 EPs got Top 10% score? {len(day1_top)} ({len(day1_top)/len(day1_all):.1%})")
    print(f"  Win Rate of Day 1 Top 10% signals: {day1_top['label_reach_50'].mean():.1%}")

if __name__ == "__main__":
    analyze_latency()
