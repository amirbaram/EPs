import datastore
import pandas as pd
from ep_ml_engine import engine

bars = datastore.load_bars("PTC")
dt_str = "2026-10-05"
print(f"{dt_str} in index? {dt_str in bars.index}")
try:
    d1_idx = bars.index.get_loc(dt_str)
    print("d1_idx:", d1_idx)
    ml_feats = engine.compute_features("PTC", d1_idx)
    print("ml_feats:", bool(ml_feats))
except Exception as e:
    print("Error:", repr(e))
