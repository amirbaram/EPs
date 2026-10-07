import datastore
from ep_ml_engine import engine

sym = "GGAL"
dt_str = "2025-09-22"
bars = datastore.load_bars(sym)
try:
    d1_idx = bars.index.get_loc(dt_str)
    feats = engine.compute_features(sym, d1_idx)
    print("Feats:", feats)
except Exception as e:
    print("Error:", repr(e))
