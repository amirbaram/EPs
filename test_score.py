import pandas as pd
from ep_ml_engine import engine
import datastore

df = pd.read_parquet("data/simulations/ep_combined_study.parquet").head(5)
for i, row in df.iterrows():
    sym = row["symbol"]
    d_str = row["date"]
    bars = datastore.load_bars(sym)
    if bars is not None:
        try:
            d1_idx = bars.index.get_loc(d_str)
            feats = engine.compute_features(sym, d1_idx)
            print(f"{sym} {d_str} feats: {bool(feats)}")
        except Exception as e:
            print(f"ERROR on {sym} {d_str}: {repr(e)}")
