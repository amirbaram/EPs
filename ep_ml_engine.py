from __future__ import annotations
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
import datastore
import indicators
import labels

class EPMLEngine:
    def __init__(self):
        self.models = {}
        self._load_models()
        self.spy_df = datastore.load_bars("SPY")

    def _load_models(self):
        targets = [50, 100, 150, 200]
        base = Path("data/models")
        for t in targets:
            model_path = base / f"amir_spec_xgboost_{t}.pkl"
            if model_path.exists():
                self.models[t] = joblib.load(model_path)
            else:
                print(f"[ML Engine] Warning: Model for +{t}% not found.")

    def compute_features(self, sym: str, d1_idx: int, days_forward: int = 5) -> dict | None:
        """
        Given the day-1 event index, computes the Amir Spec Day-5 features.
        Returns None if the setup failed to survive to Day 5 or data is insufficient.
        """
        df = datastore.load_bars(sym)
        if df is None or len(df) <= d1_idx + (days_forward - 1):
            return None
            
        df = df.copy()
        if 'rvol' not in df.columns:
            df = indicators.add_indicators(df)
            
        df = df.loc[:, ~df.columns.duplicated()]
        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
        
        df["is_pivot_high"] = (
            (df["high"] > df["high"].shift(1)) & 
            (df["high"] > df["high"].shift(2)) & 
            (df["high"] > df["high"].shift(-1)) & 
            (df["high"] > df["high"].shift(-2))
        )

        d5_idx = d1_idx + (days_forward - 1)
        
        d1_low = df["low"].iloc[d1_idx]
        d1_close = df["close"].iloc[d1_idx]
        d1_vol = df["volume"].iloc[d1_idx]
        
        survival_window = df["low"].iloc[d1_idx:d5_idx+1]
        if survival_window.min() < d1_low:
            return None # Failed before Day 5
            
        d5_close = df["close"].iloc[d5_idx]
        
        pre_20 = df.iloc[max(0, d1_idx-20):d1_idx]
        tightness = (pre_20["high"].max() - pre_20["low"].min()) / pre_20["low"].min() if len(pre_20) > 0 and pre_20["low"].min() > 0 else 0
        
        d5_ema10 = df["ema10"].iloc[d5_idx]
        d5_ema20 = df["ema20"].iloc[d5_idx]
        d5_ema50 = df["ema50"].iloc[d5_idx]
        is_linear = int((d5_ema10 > d5_ema20) and (d5_ema20 > d5_ema50))
        
        pre_189_vol = df["volume"].iloc[max(0, d1_idx-189):d1_idx]
        is_novel_vol = int(d1_vol > pre_189_vol.max()) if len(pre_189_vol) > 0 else 0
        
        rvol = df["rvol"].iloc[d1_idx]
        close_pos = df["close_pos"].iloc[d1_idx]
        is_inst_sweet_spot = int(rvol >= 5.0 and close_pos >= 0.65)
        
        d2_d5 = df.iloc[d1_idx+1:d5_idx+1]
        up_vol = d2_d5[d2_d5["close"] > d2_d5["open"]]["volume"].sum()
        dn_vol = d2_d5[d2_d5["close"] <= d2_d5["open"]]["volume"].sum()
        digestion_ratio = up_vol / (dn_vol + 1)
        
        lookback_pivots = df.iloc[max(0, d5_idx-250):d1_idx]
        valid_pivots = lookback_pivots[lookback_pivots["is_pivot_high"]]["high"]
        
        overhead = valid_pivots[valid_pivots > d5_close]
        if len(overhead) == 0:
            is_blue_sky = 1
            dist_to_overhead = 1.0 
        else:
            is_blue_sky = 0
            dist_to_overhead = (overhead.min() - d5_close) / d5_close
            
        # Context building (we can approximate the simple ones to avoid ContextBuilder overhead for single lookups)
        # We need ind_rank_3m, tk_rs_spy_3m, strongest_theme
        from build_ml_dataset import ContextBuilder
        ctxb = ContextBuilder(pd.DataFrame({sym: df["close"]}), self.spy_df)
        c_data = ctxb.for_symbol(sym, df.index).iloc[d5_idx]
        
        return {
            "feature_gap_pct": df["gap_pct"].iloc[d1_idx],
            "feature_rvol": rvol,
            "feature_close_pos": close_pos,
            "feature_tightness_1m": tightness,
            "feature_is_qullamagi_linear": is_linear,
            "feature_is_novel_vol_9m": is_novel_vol,
            "feature_is_inst_sweet_spot": is_inst_sweet_spot,
            "feature_digestion_ratio": digestion_ratio,
            "feature_day5_ret_vs_day1": (d5_close - d1_close) / d1_close if d1_close > 0 else 0,
            "feature_is_blue_sky": is_blue_sky,
            "feature_dist_to_overhead_pct": dist_to_overhead,
            "feature_ind_rank_3m": c_data["ind_rank_3m"],
            "feature_tk_rs_spy_3m": c_data["tk_rs_spy_3m"],
            "feature_strongest_theme": c_data["strongest_theme"],
        }
        
    def predict(self, features: dict) -> dict:
        if not self.models:
            return {"prob_50": 0.0, "prob_100": 0.0, "prob_150": 0.0, "prob_200": 0.0}
            
        # The XGBoost models expect a DataFrame with the exact columns used during training.
        # We also need to map categorical features correctly (easiest way is to pass a dataframe so the preprocessor handles it)
        df_feat = pd.DataFrame([features])
        
        res = {}
        for t, model in self.models.items():
            res[f"prob_{t}"] = float(model.predict_proba(df_feat)[0, 1])
            
        return res

engine = EPMLEngine()
