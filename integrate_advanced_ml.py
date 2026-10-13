with open("ep_ml_engine.py", "r") as f:
    text = f.read()

# __init__ patch
old_init = """        v2_path = base / "amir_rolling_xgboost_ordinal.pkl"
        if v2_path.exists():
            self.v2_ordinal = joblib.load(v2_path)
        else:
            self.v2_ordinal = None
            
        self.spy_df = datastore.load_bars("SPY")"""

new_init = """        v2_path = base / "amir_rolling_xgboost_ordinal.pkl"
        if v2_path.exists():
            self.v2_ordinal = joblib.load(v2_path)
        else:
            self.v2_ordinal = None
            
        tox_path = base / "ep_toxicity_filter.pkl"
        if tox_path.exists():
            self.toxicity_filter = joblib.load(tox_path)
        else:
            self.toxicity_filter = None
            
        stop_path = base / "ep_dynamic_stoploss_q05.pkl"
        if stop_path.exists():
            self.dynamic_stop = joblib.load(stop_path)
        else:
            self.dynamic_stop = None
            
        self.spy_df = datastore.load_bars("SPY")"""
text = text.replace(old_init, new_init)

# predict patch
old_predict = """        return {
            "prob_50": float(probs[1:].sum()),
            "prob_100": float(probs[2:].sum()),
            "prob_150": float(probs[3:].sum()),
            "prob_200": float(probs[4])
        }"""
        
new_predict = """        res = {
            "prob_50": float(probs[1:].sum()),
            "prob_100": float(probs[2:].sum()),
            "prob_150": float(probs[3:].sum()),
            "prob_200": float(probs[4])
        }
        if self.toxicity_filter:
            tox_features = ["feature_gap_pct", "feature_rvol", "feature_close_pos", "feature_is_novel_vol_9m"]
            # Just incase rolling doesn't have it exactly, we take what we have
            tox_df = df_feat[[c for c in tox_features if c in df_feat.columns]].fillna(0)
            if len(tox_df.columns) == len(tox_features):
                res["is_toxic"] = bool(self.toxicity_filter.predict(tox_df)[0] == -1)
                
        if self.dynamic_stop:
            stop_feats = df_feat.select_dtypes(include=[np.number])
            # The LGBM model expects the exact same columns used in training.
            # We wrap in try-except in case feature sets misalign during inference
            try:
                res["dynamic_stop_loss_pct"] = float(self.dynamic_stop.predict(stop_feats)[0])
            except Exception:
                pass
                
        return res"""

text = text.replace(old_predict, new_predict)

with open("ep_ml_engine.py", "w") as f:
    f.write(text)
