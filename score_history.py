import pandas as pd
import numpy as np
from pathlib import Path
from ep_ml_engine import engine

df = pd.read_parquet("data/simulations/ep_combined_study.parquet")

# We will batch infer.
ml_50 = []
ml_100 = []
ml_150 = []
ml_200 = []

# pre-load datastore cache if possible
import datastore

for i, row in df.iterrows():
    sym = row["symbol"]
    d_str = row["date"]
    
    bars = datastore.load_bars(sym)
    
    prob_50 = np.nan
    prob_100 = np.nan
    prob_150 = np.nan
    prob_200 = np.nan
    
    if bars is not None:
        try:
            d1_idx = bars.index.get_loc(d_str)
            feats = engine.compute_features(sym, d1_idx)
            if feats is not None:
                preds = engine.predict(feats)
                prob_50 = preds["prob_50"]
                prob_100 = preds["prob_100"]
                prob_150 = preds["prob_150"]
                prob_200 = preds["prob_200"]
        except Exception as e:
            pass
            
    ml_50.append(prob_50)
    ml_100.append(prob_100)
    ml_150.append(prob_150)
    ml_200.append(prob_200)

df["ml_50"] = ml_50
df["ml_100"] = ml_100
df["ml_150"] = ml_150
df["ml_200"] = ml_200

out_path = Path("data/simulations/ep_combined_study_scored.parquet")
df.to_parquet(out_path)
print(f"Saved scored historical dataset to {out_path}")
