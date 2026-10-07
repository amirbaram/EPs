import datastore
from ep_ml_engine import engine

sym = "FLY"
dt_str = "2026-09-18"
bars = datastore.load_bars(sym)
try:
    d1_idx = bars.index.get_loc(dt_str)
    feats = engine.compute_features(sym, d1_idx)
    print("Feats:", feats)
except Exception as e:
    print("Error:", repr(e))
