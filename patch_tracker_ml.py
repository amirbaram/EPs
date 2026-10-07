import re

with open("serve_ep_tracker.py", "r") as f:
    code = f.read()

search = """def api_tracker_events():
    events = get_tracker_data(force_rescan=False)
    return jsonify({"events": events, "count": len(events)})"""

replace = """def api_tracker_events():
    events = get_tracker_data(force_rescan=False)
    from ep_ml_engine import engine
    import datastore
    import pandas as pd
    
    for ev in events:
        sym = ev["symbol"]
        dt_str = ev["date"]
        bars = datastore.load_bars(sym)
        ev["ml_50"] = None
        ev["ml_150"] = None
        if bars is not None and dt_str in bars.index:
            try:
                d1_idx = bars.index.get_loc(dt_str)
                ml_feats = engine.compute_features(sym, d1_idx)
                if ml_feats:
                    preds = engine.predict(ml_feats)
                    ev["ml_50"] = preds["prob_50"]
                    ev["ml_150"] = preds["prob_150"]
            except Exception:
                pass

    return jsonify({"events": events, "count": len(events)})"""

if search in code:
    code = code.replace(search, replace)
    print("tracker patched!")
else:
    print("search failed!")

with open("serve_ep_tracker.py", "w") as f:
    f.write(code)

