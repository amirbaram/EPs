import pandas as pd
df = pd.read_parquet("data/simulations/ep_combined_study_scored.parquet").dropna(subset=["ml_50"])

print("\n--- Example of High Score ---")
high_50 = df[df["ml_50"] > 0.60]
if not high_50.empty:
    ex = high_50.sort_values("ml_50", ascending=False).iloc[10] # get a fun one
    print(f"Ticker: {ex['symbol']} on {ex['date']}")
    print(f"Model gave {ex['ml_50']*100:.1f}% probability for +50% target.")
    print(f"Model gave {ex['ml_100']*100:.1f}% probability for +100% target.")
    print(f"Did it hit +50%? {ex['hit_50pct']}. Max Gain was: {ex['max_gain']:.1f}%")
