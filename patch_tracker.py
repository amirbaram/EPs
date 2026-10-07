import re

with open("serve_ep_tracker.py", "r") as f:
    code = f.read()

# Replace api_ai_top_picks loop
search_str = """        pred = ep_predictor.get_predictions(features)
        score = pred["prob_hit_100"] + pred["prob_clean_run"] - pred["prob_consolidation"]
        
        if pred["archetype"] == "Toxic Traps":
            score -= 100
            
        scored_events.append({
            "symbol": features.get("symbol"),
            "date": dt_str,
            "days_old": features["days_since_ep"],
            "archetype": pred["archetype"],
            "score": round(score, 1),
            "prob_100": pred["prob_hit_100"],
            "prob_consol": pred["prob_consolidation"],"""

replacement_str = """        import datastore
        from ep_ml_engine import engine
        
        prob_100 = 0.0
        prob_50 = 0.0
        ml_150 = 0.0
        
        # ML Engine Inference
        bars = datastore.load_bars(features.get("symbol"))
        if bars is not None and dt_str in bars.index:
            try:
                d1_idx = bars.index.get_loc(dt_str)
                ml_feats = engine.compute_features(features.get("symbol"), d1_idx)
                if ml_feats:
                    preds = engine.predict(ml_feats)
                    prob_50 = preds["prob_50"]
                    prob_100 = preds["prob_100"]
                    ml_150 = preds["prob_150"]
            except Exception:
                pass
                
        pred = ep_predictor.get_predictions(features)
        archetype = pred["archetype"]
        
        # Blend ML Score with archetype rules (ML takes precedence for score)
        score = (prob_100 * 100) + (prob_50 * 50)
        
        if archetype == "Toxic Traps":
            score -= 100
            
        scored_events.append({
            "symbol": features.get("symbol"),
            "date": dt_str,
            "days_old": features["days_since_ep"],
            "archetype": f"{archetype} (AI: {int(prob_100*100)}%)",
            "score": round(score, 1),
            "prob_100": round(prob_100 * 100, 1),
            "prob_consol": round(ml_150 * 100, 1),"""

code = code.replace(search_str, replacement_str)
with open("serve_ep_tracker.py", "w") as f:
    f.write(code)
print("serve_ep_tracker.py patched!")
