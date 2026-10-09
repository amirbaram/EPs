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
            
        stop_path = base / "ep_dynamic_trailer.pkl"
        self.dynamic_stop = joblib.load(stop_path) if stop_path.exists() else None
        
        exh_path = base / "ep_exhaustion_classifier.pkl"
        self.exhaustion_classifier = joblib.load(exh_path) if exh_path.exists() else None
            
        reentry_path = base / "ep_reentry_classifier.pkl"
        self.reentry_classifier = joblib.load(reentry_path) if reentry_path.exists() else None

        ur_path = base / "ep_ur_reentry_classifier.pkl"
        self.ur_reentry_classifier = joblib.load(ur_path) if ur_path.exists() else None

        cont_stop_path = base / "ep_continuation_dynamic_trailer.pkl"
        self.continuation_trailer = joblib.load(cont_stop_path) if cont_stop_path.exists() else None

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
                
        if hasattr(self, 'exhaustion_classifier') and self.exhaustion_classifier:
            try:
                # Exhaustion is only meaningful if stock is up
                if curr_ret >= 0.20:
                    probs_exh = self.exhaustion_classifier.predict_proba(df_feat)[0]
                    res["prob_exhaustion"] = float(probs_exh[1])
            except Exception:
                pass
                
        return res

    def predict(self, features: dict) -> dict:
        return self._execute_predictions(self.v1_ordinal, features, is_v1=True)
        
    def predict_rolling(self, features: dict) -> dict:
        return self._execute_predictions(self.v2_ordinal, features, is_v1=False)

    def compute_reentry_features(self, sym: str, ep_idx: int, t_idx: int, ribbon_spans=(8, 12, 16, 21)) -> dict | None:
        df = datastore.load_bars(sym)
        if df is None or t_idx >= len(df) or t_idx < ep_idx: return None

        df = df.copy()
        df["ema10"] = df["close"].ewm(span=10, adjust=False).mean()
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["sma50"] = df["close"].rolling(50).mean()
        df["vol20"] = df["volume"].rolling(20).mean()

        import scanner_core
        ribbon_df = scanner_core.calc_intermediate_ribbon(df, spans=ribbon_spans)
        df["ribbon_state"] = ribbon_df["ribbon_state"]
        df["ribbon_compression"] = ribbon_df["ribbon_compression"]
        df["ribbon_bandwidth"] = ribbon_df["ribbon_bandwidth"]
        s1, _, _, s4 = ribbon_spans
        df[f"ribbon_ema{s1}"] = ribbon_df[f"ribbon_ema{s1}"]
        df[f"ribbon_ema{s4}"] = ribbon_df[f"ribbon_ema{s4}"]

        c = float(df["close"].iloc[t_idx])
        sma50 = float(df["sma50"].iloc[t_idx]) if pd.notna(df["sma50"].iloc[t_idx]) else 0.0
        peak_high = float(df["high"].iloc[ep_idx:t_idx].max()) if t_idx > ep_idx else c
        drawdown_from_peak = (c - peak_high) / peak_high if peak_high > 0 else 0.0

        ema10 = float(df["ema10"].iloc[t_idx]) if pd.notna(df["ema10"].iloc[t_idx]) else c
        ema20 = float(df["ema20"].iloc[t_idx]) if pd.notna(df["ema20"].iloc[t_idx]) else c
        vol20 = float(df["vol20"].iloc[t_idx]) if pd.notna(df["vol20"].iloc[t_idx]) else 1.0
        vol = float(df["volume"].iloc[t_idx]) if pd.notna(df["volume"].iloc[t_idx]) else 0.0

        prev_state = df["ribbon_state"].iloc[t_idx - 1] if t_idx > 0 else None
        curr_state = df["ribbon_state"].iloc[t_idx]
        is_yellow_flip = int(curr_state == "yellow" and prev_state != "yellow")

        e_fast = float(df[f"ribbon_ema{s1}"].iloc[t_idx]) if pd.notna(df[f"ribbon_ema{s1}"].iloc[t_idx]) else c
        e_slow = float(df[f"ribbon_ema{s4}"].iloc[t_idx]) if pd.notna(df[f"ribbon_ema{s4}"].iloc[t_idx]) else c
        comp = float(df["ribbon_compression"].iloc[t_idx]) if pd.notna(df["ribbon_compression"].iloc[t_idx]) else 0.0
        bw = float(df["ribbon_bandwidth"].iloc[t_idx]) if pd.notna(df["ribbon_bandwidth"].iloc[t_idx]) else 0.0

        return {
            "feature_days_since_ep": t_idx - ep_idx,
            "feature_drawdown_from_peak": drawdown_from_peak,
            "feature_dist_10ema": (c - ema10) / ema10 if ema10 > 0 else 0.0,
            "feature_dist_20ema": (c - ema20) / ema20 if ema20 > 0 else 0.0,
            "feature_dist_50sma": (c - sma50) / sma50 if sma50 > 0 else 0.0,
            "feature_vol_contraction": vol / vol20 if vol20 > 0 else 1.0,
            "feature_is_yellow_flip": is_yellow_flip,
            "feature_ribbon_compression": comp,
            "feature_ribbon_bandwidth": bw,
            "feature_dist_ribbon_ema8": (c - e_fast) / e_fast if e_fast > 0 else 0.0,
            "feature_dist_ribbon_ema21": (c - e_slow) / e_slow if e_slow > 0 else 0.0,
        }

    def predict_reentry(self, features: dict) -> dict:
        if not hasattr(self, 'reentry_classifier') or self.reentry_classifier is None:
            return {"prob_win": 0.0, "predicted_win": False}
        df_feat = pd.DataFrame([features])
        if hasattr(self.reentry_classifier, "named_steps") and "scaler" in self.reentry_classifier.named_steps:
            cols = list(self.reentry_classifier.named_steps["scaler"].feature_names_in_)
            df_feat = df_feat.reindex(columns=cols, fill_value=0.0)
        elif hasattr(self.reentry_classifier, "feature_names_in_"):
            cols = list(self.reentry_classifier.feature_names_in_)
            df_feat = df_feat.reindex(columns=cols, fill_value=0.0)
        try:
            probs = self.reentry_classifier.predict_proba(df_feat)[0]
            prob_win = float(probs[1]) if len(probs) > 1 else float(probs[0])
            return {
                "prob_win": round(prob_win, 3),
                "predicted_win": bool(prob_win >= 0.5)
            }
        except Exception:
            return {"prob_win": 0.0, "predicted_win": False}

    def compute_ur_reentry_features(
        self,
        sym: str,
        ep_idx: int,
        t1_exit_bar: int,
        ur_signal_bar: int,
        shakeout_low: float,
        ribbon_spans=(8, 12, 16, 21),
        df: pd.DataFrame | None = None
    ) -> dict | None:
        if df is None:
            df = datastore.load_bars(sym)
        if df is None:
            return None

        # Auto-adjust if caller passed relative bar offsets from ep_idx (e.g. from an EP forward slice)
        if ep_idx > 0 and ur_signal_bar < ep_idx and t1_exit_bar < ep_idx:
            t1_exit_bar = ep_idx + t1_exit_bar
            ur_signal_bar = ep_idx + ur_signal_bar

        if ur_signal_bar >= len(df) or ur_signal_bar < ep_idx:
            return None

        df = df.copy()
        if "rvol" not in df.columns:
            df = indicators.add_indicators(df)
        df = df.loc[:, ~df.columns.duplicated()]

        if "sma50" not in df.columns:
            df["sma50"] = df["close"].rolling(50).mean()
        if "vol20" not in df.columns:
            df["vol20"] = df["volume"].rolling(20).mean()

        import scanner_core
        if "ribbon_compression" not in df.columns:
            ribbon_df = scanner_core.calc_intermediate_ribbon(df, spans=ribbon_spans)
            df["ribbon_compression"] = ribbon_df["ribbon_compression"]

        d1_low = float(df["low"].iloc[ep_idx])
        t1_peak = float(df["high"].iloc[ep_idx:t1_exit_bar + 1].max())

        sig_c = float(df["close"].iloc[ur_signal_bar])
        sig_sma50 = float(df["sma50"].iloc[ur_signal_bar]) if pd.notna(df["sma50"].iloc[ur_signal_bar]) else sig_c
        sig_vol = float(df["volume"].iloc[ur_signal_bar])
        sig_vol20 = float(df["vol20"].iloc[ur_signal_bar]) if pd.notna(df["vol20"].iloc[ur_signal_bar]) else sig_vol
        sig_rvol = float(df["rvol"].iloc[ur_signal_bar]) if pd.notna(df["rvol"].iloc[ur_signal_bar]) else 1.0
        sig_comp = float(df["ribbon_compression"].iloc[ur_signal_bar]) if pd.notna(df["ribbon_compression"].iloc[ur_signal_bar]) else 0.0

        undercut_depth_pct = (d1_low - shakeout_low) / d1_low * 100.0 if d1_low > 0 else 0.0
        days_to_reclaim = float(ur_signal_bar - t1_exit_bar)
        reclaim_vol_ratio = sig_vol / sig_vol20 if sig_vol20 > 0 else 1.0
        dist_to_sma50 = (sig_c - sig_sma50) / sig_sma50 if sig_sma50 > 0 else 0.0
        sma50_cushion = (shakeout_low - sig_sma50) / sig_sma50 if sig_sma50 > 0 else 0.0
        drop_from_peak = (t1_peak - shakeout_low) / t1_peak * 100.0 if t1_peak > 0 else 0.0

        return {
            "feature_undercut_depth_pct": undercut_depth_pct,
            "feature_days_to_reclaim": days_to_reclaim,
            "feature_reclaim_volume_ratio": reclaim_vol_ratio,
            "feature_dist_to_sma50": dist_to_sma50,
            "feature_sma50_cushion": sma50_cushion,
            "feature_ribbon_compression": sig_comp,
            "feature_drop_from_peak": drop_from_peak,
            "feature_rvol": sig_rvol,
        }

    def predict_ur_reentry(self, features: dict, prob_threshold: float = 0.40) -> dict:
        if not hasattr(self, 'ur_reentry_classifier') or self.ur_reentry_classifier is None:
            return {"prob_win": 0.5, "is_qualified": True}
        df_feat = pd.DataFrame([features])
        if hasattr(self.ur_reentry_classifier, "named_steps") and "scaler" in self.ur_reentry_classifier.named_steps:
            cols = list(self.ur_reentry_classifier.named_steps["scaler"].feature_names_in_)
            df_feat = df_feat.reindex(columns=cols, fill_value=0.0)
        elif hasattr(self.ur_reentry_classifier, "feature_names_in_"):
            cols = list(self.ur_reentry_classifier.feature_names_in_)
            df_feat = df_feat.reindex(columns=cols, fill_value=0.0)
        try:
            probs = self.ur_reentry_classifier.predict_proba(df_feat)[0]
            prob_win = float(probs[1]) if len(probs) > 1 else float(probs[0])
            return {
                "prob_win": round(prob_win, 3),
                "is_qualified": bool(prob_win >= prob_threshold),
            }
        except Exception:
            return {"prob_win": 0.5, "is_qualified": True}

    def compute_continuation_trailer_features(
        self,
        df: pd.DataFrame,
        b_idx: int,
        entry_price: float,
        trade_peak: float,
        risk_pts: float,
        ribbon_spans=(8, 12, 16, 21),
        entry_bar: int | None = None,
    ) -> dict | None:
        if b_idx >= len(df) or risk_pts <= 0:
            return None
        c = float(df["close"].iloc[b_idx])
        v = float(df["volume"].iloc[b_idx])
        v20 = float(df["vol20"].iloc[b_idx]) if "vol20" in df.columns and pd.notna(df["vol20"].iloc[b_idx]) else v
        sma50 = float(df["sma50"].iloc[b_idx]) if "sma50" in df.columns and pd.notna(df["sma50"].iloc[b_idx]) else c

        ema21_col = f"ribbon_ema{ribbon_spans[3]}"
        e21 = float(df[ema21_col].iloc[b_idx]) if ema21_col in df.columns and pd.notna(df[ema21_col].iloc[b_idx]) else c
        comp = float(df["ribbon_compression"].iloc[b_idx]) if "ribbon_compression" in df.columns and pd.notna(df["ribbon_compression"].iloc[b_idx]) else 0.0

        unrealized_r = (trade_peak - entry_price) / risk_pts
        days_in_trade = float(b_idx - entry_bar) if entry_bar is not None and b_idx >= entry_bar else float(min(120, b_idx))
        return {
            "feature_r_multiple": unrealized_r,
            "feature_dist_ribbon_ema21": (c - e21) / e21 if e21 > 0 else 0.0,
            "feature_ribbon_compression": comp,
            "feature_dist_50sma": (c - sma50) / sma50 if sma50 > 0 else 0.0,
            "feature_vol_contraction": v / v20 if v20 > 0 else 1.0,
            "feature_days_in_trade": days_in_trade,
            "feature_drawdown_from_peak": (c - trade_peak) / trade_peak * 100.0 if trade_peak > 0 else 0.0,
        }

    def predict_continuation_trailer(self, features: dict) -> float:
        if not hasattr(self, 'continuation_trailer') or self.continuation_trailer is None:
            return 12.0
        df_feat = pd.DataFrame([features])
        if hasattr(self.continuation_trailer, "named_steps") and "scaler" in self.continuation_trailer.named_steps:
            cols = list(self.continuation_trailer.named_steps["scaler"].feature_names_in_)
            df_feat = df_feat.reindex(columns=cols, fill_value=0.0)
        try:
            pred = float(self.continuation_trailer.predict(df_feat)[0])
            return round(max(3.0, min(35.0, pred)), 2)
        except Exception:
            return 12.0

engine = EPMLEngine()
