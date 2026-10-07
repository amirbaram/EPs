import serve_ep_tracker
events = serve_ep_tracker.get_tracker_data(force_rescan=False)
from ep_ml_engine import engine
import datastore
for ev in events[:10]:
    sym = ev["symbol"]
    dt_str = ev.get("date") or ev.get("event_date")
    bars = datastore.load_bars(sym)
    if bars is not None and dt_str in bars.index:
        d1_idx = bars.index.get_loc(dt_str)
        bars_since = len(bars) - 1 - d1_idx
        dfwd = max(1, min(5, bars_since))
        ml_feats = engine.compute_features(sym, d1_idx, days_forward=dfwd)
        print(f"{sym} dfwd={dfwd} feats={bool(ml_feats)}")
