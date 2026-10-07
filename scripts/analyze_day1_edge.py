import pandas as pd
import numpy as np
import joblib
from pathlib import Path

data_dir = Path("data/ml_datasets")
folders = sorted([f for f in data_dir.glob("ep_dataset_daily_v4_*") if f.is_dir()])
parquet_file = folders[-1] / "dataset.parquet"
df = pd.read_parquet(parquet_file)

df = df[df["feature_chk_reached_50"] == 0].copy()
df = df[df["label_reach_50"].notna()].copy()
df["as_of_date"] = pd.to_datetime(df["identifier_as_of_date"])
df = df[df["as_of_date"] >= pd.Timestamp("2024-01-01")].copy()

model = joblib.load("data/models/ep_xgboost_prod.pkl")
import json
with open("data/models/feature_schema.json", "r") as f:
    schema = json.load(f)
features = schema["features_ordered"]

df["predicted_prob"] = model.predict_proba(df[features])[:, 1]

# ISOLATE DAY 1 ONLY
day1 = df[df["feature_age_sessions"] == 1].copy()

print(f"Total Day 1 Events in Test Set: {len(day1)}")
print(f"Base Rate (Win Rate) of all Day 1s: {day1['label_reach_50'].mean():.2%}")

# Get the Top 10% OF DAY 1 EVENTS
threshold = np.percentile(day1["predicted_prob"], 90)
day1_top10 = day1[day1["predicted_prob"] >= threshold]

print(f"Top 10% threshold FOR DAY 1 specifically: {threshold:.4f}")
print(f"Win Rate of Top 10% Day 1s: {day1_top10['label_reach_50'].mean():.2%}")

# Let's check Bottom 10%
bottom_threshold = np.percentile(day1["predicted_prob"], 10)
day1_bottom10 = day1[day1["predicted_prob"] <= bottom_threshold]
print(f"Win Rate of Bottom 10% Day 1s: {day1_bottom10['label_reach_50'].mean():.2%}")

