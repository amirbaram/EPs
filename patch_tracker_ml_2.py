with open("serve_ep_tracker.py", "r") as f:
    code = f.read()

search = """ml_feats = engine.compute_features(sym, d1_idx)"""
replace = """bars_since = len(bars) - 1 - d1_idx
                dfwd = max(1, min(5, bars_since))
                ml_feats = engine.compute_features(sym, d1_idx, days_forward=dfwd)"""
code = code.replace(search, replace)

search2 = """ml_feats = engine.compute_features(features.get("symbol"), d1_idx)"""
replace2 = """bars_since = len(bars) - 1 - d1_idx
                dfwd = max(1, min(5, bars_since))
                ml_feats = engine.compute_features(features.get("symbol"), d1_idx, days_forward=dfwd)"""
code = code.replace(search2, replace2)

with open("serve_ep_tracker.py", "w") as f:
    f.write(code)

