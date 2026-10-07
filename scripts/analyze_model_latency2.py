import pandas as pd
import numpy as np
import joblib
from pathlib import Path

print("🚀 DEEP DIVE: PREDICTION LATENCY & PRICE LOCATION")

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
threshold = np.percentile(df["predicted_prob"], 90)
top_decile = df[df["predicted_prob"] >= threshold].copy()

print(f"\n[+] Analysis of Top 10% Signals (Prob >= {threshold:.4f}):")
print(f"Total signals: {len(top_decile)}")
print(f"Average Age of EP (sessions): {top_decile['feature_age_sessions'].mean():.1f} days")
print(f"Average Return since EP Day 1: {top_decile['feature_chk_return_since_event_pct'].mean():.1f}%")
print(f"Average Distance left to +50% target: {top_decile['feature_chk_dist_to_target_50_pct'].mean():.1f}%")
print(f"Average MFE (Max Favorable Excursion) achieved so far: {top_decile['feature_chk_observed_mfe_pct'].mean():.1f}%")

print("\n[+] Are we just predicting on stocks that are already up 45%?")
up_40_plus = top_decile[top_decile['feature_chk_return_since_event_pct'] >= 40]
print(f"Signals where stock is ALREADY UP 40%+ from entry: {len(up_40_plus)} ({len(up_40_plus)/len(top_decile):.1%})")

up_20_to_40 = top_decile[(top_decile['feature_chk_return_since_event_pct'] >= 20) & (top_decile['feature_chk_return_since_event_pct'] < 40)]
print(f"Signals where stock is UP 20% to 40%: {len(up_20_to_40)} ({len(up_20_to_40)/len(top_decile):.1%})")

down_or_flat = top_decile[top_decile['feature_chk_return_since_event_pct'] <= 5]
print(f"Signals where stock is FLAT or DOWN (<= 5%): {len(down_or_flat)} ({len(down_or_flat)/len(top_decile):.1%})")

