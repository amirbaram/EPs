from __future__ import annotations
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
import datastore
import indicators
import labels
import trendlab

class EPMLEngine:
    def __init__(self):
        self.models = {}
        self.rolling_models = {}
        
        base = Path("data/models")
        
        v1_path = base / "amir_spec_xgboost_ordinal.pkl"
        self.v1_ordinal = joblib.load(v1_path) if v1_path.exists() else None
            
        v2_path = base / "amir_rolling_xgboost_ordinal.pkl"
        self.v2_ordinal = joblib.load(v2_path) if v2_path.exists() else None
            
        tox_path = base / "ep_toxicity_filter.pkl"
        self.toxicity_filter = joblib.load(tox_path) if tox_path.exists() else None
            
        stop_path = base / "ep_dynamic_stoploss_q05.pkl"
        self.dynamic_stop = joblib.load(stop_path) if stop_path.exists() else None
            
        self.spy_df = datastore.load_bars("SPY")
        
        # [FIX]: Load Context Cache EXACTLY ONCE into an O(1) Lookup Dictionary
        self.context_cache = {}
        try:
            with open("data/cache/tracker_eps.json", "r") as f:
                tracker = json.load(f)
                for ev in tracker:
                    self.context_cache[(ev["symbol"], ev["event_date"])] = {
                        "ind_rank_3m": ev.get("sec_m1_pctile", np.nan),
                        "tk_rs_spy_3m": ev.get("tk_rs_spy_3m", np.nan),
                        "strongest_theme": ev.get("theme", "")
                    }
        except Exception:
            pass

    def compute_features(self, sym: str, d1_idx: int, days_forward: int = 5) -> dict | None:
        df = datastore.load_bars(sym)
        if df is None or len(df) <= d1_idx + (days_forward - 1): return None
        df = df.copy()
        if 'rvol' not in df.columns: df = indicators.add_indicators(df)
        df = df.loc[:, ~df.columns.duplicated()]
        df["ema10"] = df["close"].ewm(span=10, adjust=False).mean()
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

        d5_idx = d1_idx + (days_forward - 1)
        d1_low = df["low"].iloc[d1_idx]
        d1_close = df["close"].iloc[d1_idx]
        d1_vol = df["volume"].iloc[d1_idx]
        
        if df["low"].iloc[d1_idx:d5_idx+1].min() < d1_low: return None
            
        d5_close = df["close"].iloc[d5_idx]
        pre_20 = df.iloc[max(0, d1_idx-20):d1_idx]
        tightness = (pre_20["high"].max() - pre_20["low"].min()) / pre_20["low"].min() if len(pre_20) > 0 and pre_20["low"].min() > 0 else 0
        is_linear = int((df["ema10"].iloc[d5_idx] > df["ema20"].iloc[d5_idx]) and (df["ema20"].iloc[d5_idx] > df["ema50"].iloc[d5_idx]))
        pre_189_vol = df["volume"].iloc[max(0, d1_idx-189):d1_idx]
        is_novel_vol = int(d1_vol > pre_189_vol.max()) if len(pre_189_vol) > 0 else 0
        rvol = df["rvol"].iloc[d1_idx]
        close_pos = df["close_pos"].iloc[d1_idx]
        is_inst_sweet_spot = int(rvol >= 5.0 and close_pos >= 0.65)
        
        d2_d5 = df.iloc[d1_idx+1:d5_idx+1]
        up_vol = d2_d5[d2_d5["close"] > d2_d5["open"]]["volume"].sum()
        dn_vol = d2_d5[d2_d5["close"] <= d2_d5["open"]]["volume"].sum()
        digestion_ratio = up_vol / (dn_vol + 1)
        
        # [FIX]: Use trendlab instead of global shifts for accurate historical pivots
        lookback = df.iloc[max(0, d5_idx-250):d5_idx+1]
        r = trendlab._compute(lookback)
        hp = [(r["pidx"][k], r["pconf"][k], r["pprice"][k]) for k in range(len(r["pidx"])) if r["ptype"][k] == trendlab.PIVOT_HIGH]
        valid_pivots_list = [pprice for pidx, pconf, pprice in hp if pconf <= len(lookback) - 1]
        
        valid_pivots = pd.Series(valid_pivots_list, dtype=float)
        overhead = valid_pivots[valid_pivots > d5_close]
        is_blue_sky = 1 if len(overhead) == 0 else 0
        dist_to_overhead = 1.0 if len(overhead) == 0 else (overhead.min() - d5_close) / d5_close
            
        # [FIX]: Fast O(1) Cache Lookup completely removes I/O Bottleneck
        ev_date = str(df.index[d1_idx].date())
        c_data = self.context_cache.get((sym, ev_date), {
            "ind_rank_3m": np.nan, "tk_rs_spy_3m": np.nan, "strongest_theme": ""
        })

        return {
            "feature_gap_pct": df["gap_pct"].iloc[d1_idx], "feature_rvol": rvol, "feature_close_pos": close_pos,
            "feature_tightness_1m": tightness, "feature_is_qullamagi_linear": is_linear, "feature_is_novel_vol_9m": is_novel_vol,
            "feature_is_inst_sweet_spot": is_inst_sweet_spot, "feature_digestion_ratio": digestion_ratio,
            "feature_day5_ret_vs_day1": (d5_close - d1_close) / d1_close if d1_close > 0 else 0,
            "feature_is_blue_sky": is_blue_sky, "feature_dist_to_overhead_pct": dist_to_overhead,
            "feature_ind_rank_3m": c_data["ind_rank_3m"], "feature_tk_rs_spy_3m": c_data["tk_rs_spy_3m"],
            "feature_strongest_theme": c_data["strongest_theme"]
        }

    def compute_rolling_features(self, sym: str, d1_idx: int, t_idx: int) -> dict | None:
        df = datastore.load_bars(sym)
        if df is None or t_idx >= len(df) or t_idx < d1_idx: return None
        
        df = df.copy()
        if 'rvol' not in df.columns: df = indicators.add_indicators(df)
        df = df.loc[:, ~df.columns.duplicated()]
        df["ema10"] = df["close"].ewm(span=10, adjust=False).mean()
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

        d1_low = df["low"].iloc[d1_idx]
        d1_close = df["close"].iloc[d1_idx]
        d1_vol = df["volume"].iloc[d1_idx]
        gap_pct = df["gap_pct"].iloc[d1_idx]
        rvol = df["rvol"].iloc[d1_idx]
        close_pos = df["close_pos"].iloc[d1_idx]
        
        if df["low"].iloc[d1_idx:t_idx+1].min() < d1_low: return None
        
        t_close = df["close"].iloc[t_idx]
        days_since_ep = t_idx - d1_idx
        pre_20 = df.iloc[max(0, d1_idx-20):d1_idx]
        tightness_1m = (pre_20["high"].max() - pre_20["low"].min()) / pre_20["low"].min() if len(pre_20) > 0 and pre_20["low"].min() > 0 else 0
        
        pre_189_vol = df["volume"].iloc[max(0, d1_idx-189):d1_idx]
        is_novel_vol = int(d1_vol > pre_189_vol.max()) if len(pre_189_vol) > 0 else 0
        is_inst_sweet_spot = int(rvol >= 5.0 and close_pos >= 0.65)
        
        d1_to_t = df.iloc[d1_idx:t_idx+1]
        tightness_since_ep = (d1_to_t["high"].max() - d1_to_t["low"].min()) / d1_to_t["low"].min() if len(d1_to_t) > 1 and d1_to_t["low"].min() > 0 else 0.0
        
        d2_t = df.iloc[d1_idx+1:t_idx+1]
        if len(d2_t) > 0:
            up_vol = d2_t[d2_t["close"] > d2_t["open"]]["volume"].sum()
            dn_vol = d2_t[d2_t["close"] <= d2_t["open"]]["volume"].sum()
            digestion_ratio = up_vol / (dn_vol + 1)
        else:
            digestion_ratio = 1.0
            
        ret_since_ep = (t_close - d1_close) / d1_close if d1_close > 0 else 0
        dist_to_d1_low = (t_close - d1_low) / d1_low if d1_low > 0 else 0
        
        lookback = df.iloc[max(0, t_idx-250):t_idx+1]
        r = trendlab._compute(lookback)
        hp = [(r["pidx"][k], r["pconf"][k], r["pprice"][k]) for k in range(len(r["pidx"])) if r["ptype"][k] == trendlab.PIVOT_HIGH]
        valid_pivots_list = [pprice for pidx, pconf, pprice in hp if pconf <= len(lookback) - 1]
        
        valid_pivots = pd.Series(valid_pivots_list, dtype=float)
        overhead = valid_pivots[valid_pivots > t_close]
        is_blue_sky = 1 if len(overhead) == 0 else 0
        dist_to_overhead = 1.0 if len(overhead) == 0 else (overhead.min() - t_close) / t_close
        
        ev_date = str(df.index[d1_idx].date())
        c_data = self.context_cache.get((sym, ev_date), {
            "ind_rank_3m": np.nan, "tk_rs_spy_3m": np.nan, "strongest_theme": ""
        })
            
        return {
            "feature_days_since_ep": days_since_ep,
            "feature_tightness_1m": tightness_1m,
            "feature_gap_pct": gap_pct, "feature_rvol": rvol, "feature_close_pos": close_pos,
            "feature_is_novel_vol_9m": is_novel_vol, "feature_is_inst_sweet_spot": is_inst_sweet_spot,
            "feature_tightness_since_ep": tightness_since_ep, "feature_digestion_ratio": digestion_ratio,
            "feature_ret_since_ep": ret_since_ep, "feature_dist_to_d1_low": dist_to_d1_low,
            "feature_is_blue_sky": is_blue_sky, "feature_dist_to_overhead_pct": dist_to_overhead,
            "feature_ind_rank_3m": c_data["ind_rank_3m"], "feature_tk_rs_spy_3m": c_data["tk_rs_spy_3m"],
            "feature_strongest_theme": c_data["strongest_theme"],
            "feature_is_qullamagi_linear": int(df["ema10"].iloc[t_idx] > df["ema20"].iloc[t_idx] and df["ema20"].iloc[t_idx] > df["ema50"].iloc[t_idx]),
        }

    def _execute_predictions(self, model, features: dict, is_v1: bool = False) -> dict:
        if not model: return {"prob_50": 0.0, "prob_100": 0.0, "prob_150": 0.0, "prob_200": 0.0}
        
        df_feat = pd.DataFrame([features])
        probs = model.predict_proba(df_feat)[0]
        
        # [FIX]: The Target Catch-22 Override
        curr_ret = features.get("feature_day5_ret_vs_day1", features.get("feature_ret_since_ep", 0.0))
        
        res = {
            "prob_50": 1.0 if curr_ret >= 0.50 else min(1.0, float(probs[1:].sum())),
            "prob_100": 1.0 if curr_ret >= 1.00 else min(1.0, float(probs[2:].sum())),
            "prob_150": 1.0 if curr_ret >= 1.50 else min(1.0, float(probs[3:].sum())),
            "prob_200": 1.0 if curr_ret >= 2.00 else min(1.0, float(probs[4]))
        }
        
        if self.toxicity_filter:
            tox_features = ["feature_gap_pct", "feature_rvol", "feature_close_pos", "feature_is_novel_vol_9m"]
            tox_dict = {c: float(df_feat[c].iloc[0]) if c in df_feat.columns else 0.0 for c in tox_features}
            tox_df = pd.DataFrame([tox_dict])
            res["is_toxic"] = bool(self.toxicity_filter.predict(tox_df)[0] == -1)
                
        if self.dynamic_stop:
            try:
                res["dynamic_stop_loss_pct"] = float(self.dynamic_stop.predict(df_feat)[0])
            except Exception:
                pass
                
        return res

    def predict(self, features: dict) -> dict:
        return self._execute_predictions(self.v1_ordinal, features, is_v1=True)
        
    def predict_rolling(self, features: dict) -> dict:
        return self._execute_predictions(self.v2_ordinal, features, is_v1=False)

engine = EPMLEngine()
