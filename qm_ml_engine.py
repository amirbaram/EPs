"""qm_ml_engine.py — Dedicated Machine Learning Engine for QM Momentum Setups.

Implements zero-lookahead feature extraction and inference for the Institutional QM ML Triad:
1. QM-V2: Setup Quality / Big Move Conviction & Trap Avoidance Model.
2. QM-Stop: Dynamic Quantile Regression Stop Loss Model.
3. QM-Climax: Parabolic Exhaustion Detection & Partial Scaling Model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

CLIMAX_MODEL_PATH = Path("data/models/qm_exhaustion_classifier.pkl")
QUALITY_MODEL_PATH = Path("data/models/qm_setup_quality_model.pkl")
STOP_MODEL_PATH = Path("data/models/qm_dynamic_stop_model.pkl")

CLIMAX_FEATURE_COLS = [
    "feat_dist_ema10_pct",
    "feat_dist_ema20_pct",
    "feat_dist_sma50_pct",
    "feat_adr_multiple",
    "feat_unrealized_r",
    "feat_rsi14",
    "feat_rsi3_extreme",
    "feat_rsi_divergence",
    "feat_rsi_vs_sma",
    "feat_bar_rvol",
    "feat_vol_vs_breakout",
    "feat_upper_wick_ratio",
    "feat_close_pos",
    "feat_consec_up_days",
    "feat_bars_since_entry",
    "feat_tl_swing_dist",
    "feat_tl_is_pivot_high",
    "feat_dist_52w_high",
    "feat_dist_pole_peak",
    "feat_pole_move_pct",
    "feat_cons_bars",
    "feat_pinch_ratio",
    "feat_prior_ep_days",
    "feat_rs_score",
    "feat_macro_tailwind",
]

QUALITY_FEATURE_COLS = [
    "move_pct",
    "cons_bars",
    "pullback_pct",
    "pullback_ratio",
    "vol_x",
    "breakout_rvol",
    "pinch_ratio",
    "r2",
    "rs_score",
    "dist_52w_high",
    "prior_ep_days_ago",
    "prior_ep_gap_pct",
    "prior_ep_rvol",
    "spy_above_200sma",
    "spy_above_50sma",
    "spy_10ema_slope",
    "qqq_bull_stack",
    "qqq_pullback",
    "qqq_spy_ratio_bull",
    "is_leading_sector",
    "is_leading_theme",
    "is_leading_macro",
    "stage2",
    "has_prior_ep",
]

STOP_FEATURE_COLS = [
    "move_pct",
    "cons_bars",
    "pullback_pct",
    "pullback_ratio",
    "pinch_ratio",
    "r2",
    "vol_x",
    "breakout_rvol",
    "risk_pct_06x",
    "risk_pct_10x",
    "dist_52w_high",
    "spy_above_200sma",
    "spy_above_50sma",
    "spy_10ema_slope",
    "qqq_bull_stack",
]


class QMMLEngine:
    """Unified Inference Engine for QM Machine Learning Triad."""

    _instance: Optional["QMMLEngine"] = None

    def __init__(
        self,
        climax_path: Union[str, Path] = CLIMAX_MODEL_PATH,
        quality_path: Union[str, Path] = QUALITY_MODEL_PATH,
        stop_path: Union[str, Path] = STOP_MODEL_PATH,
    ):
        self.climax_path = Path(climax_path)
        self.quality_path = Path(quality_path)
        self.stop_path = Path(stop_path)

        # Climax components
        self.pipeline = None
        self._scaler = None
        self._clf = None

        # Quality components
        self.clf_big = None
        self.clf_trap = None

        # Stop components
        self.stop_regressor = None

        self.load_models()

    @classmethod
    def get_engine(cls) -> "QMMLEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_models(self) -> bool:
        success = True
        # 1. Climax model
        if self.climax_path.exists():
            try:
                self.pipeline = joblib.load(self.climax_path)
                if hasattr(self.pipeline, "named_steps"):
                    self._scaler = self.pipeline.named_steps.get("scaler")
                    self._clf = self.pipeline.named_steps.get("classifier")
            except Exception as e:
                print(f"Warning: Failed loading climax model: {e}")
                success = False

        # 2. Quality model
        if self.quality_path.exists():
            try:
                q_bundle = joblib.load(self.quality_path)
                if isinstance(q_bundle, dict):
                    self.clf_big = q_bundle.get("clf_big")
                    self.clf_trap = q_bundle.get("clf_trap")
            except Exception as e:
                print(f"Warning: Failed loading quality model: {e}")
                success = False

        # 3. Dynamic stop model
        if self.stop_path.exists():
            try:
                s_bundle = joblib.load(self.stop_path)
                if isinstance(s_bundle, dict):
                    self.stop_regressor = s_bundle.get("regressor")
            except Exception as e:
                print(f"Warning: Failed loading dynamic stop model: {e}")
                success = False

        return success

    # -------------------------------------------------------------------------
    # 1. Climax Exhaustion Prediction
    # -------------------------------------------------------------------------
    def predict_climax_proba(self, features: Union[Dict[str, Any], pd.DataFrame]) -> float:
        """Returns P(Exhaustion) in [0.0, 1.0]."""
        if self.pipeline is None and not self.load_models():
            return 0.0

        if isinstance(features, dict):
            if self._scaler is not None and self._clf is not None:
                feat_arr = np.array([float(features.get(col, 0.0)) for col in CLIMAX_FEATURE_COLS], dtype=np.float32)
                scaled = (feat_arr - self._scaler.mean_) / self._scaler.scale_
                return float(self._clf.predict_proba(scaled.reshape(1, -1))[0, 1])
            df_in = pd.DataFrame([{col: float(features.get(col, 0.0)) for col in CLIMAX_FEATURE_COLS}])
        else:
            df_in = features[CLIMAX_FEATURE_COLS]

        try:
            probs = self.pipeline.predict_proba(df_in)
            return float(probs[0, 1]) if probs.ndim == 2 else float(probs[1])
        except Exception:
            return 0.0

    def predict_proba_fast(self, feat_arr: np.ndarray) -> float:
        """Vectorized numpy inference for climax exhaustion (sub-millisecond)."""
        if self._scaler is None or self._clf is None:
            if not self.load_models():
                return 0.0
        scaled = (feat_arr - self._scaler.mean_) / self._scaler.scale_
        return float(self._clf.predict_proba(scaled.reshape(1, -1))[0, 1])

    # -------------------------------------------------------------------------
    # 2. Setup Quality & Trap Avoidance Prediction (QM-V2)
    # -------------------------------------------------------------------------
    def predict_setup_quality(self, features: Union[Dict[str, Any], pd.DataFrame, pd.Series]) -> Dict[str, Any]:
        """Evaluates setup coil quality and failure risk, returning score, probabilities and tier."""
        if not features or (isinstance(features, dict) and (len(features) == 0 or float(features.get("move_pct", 0.0) or 0.0) <= 0)):
            return {
                "qm_score": 50.0,
                "p_big": 0.25,
                "p_trap": 0.23,
                "tier": "Standard",
                "sizing": 1.0,
                "recommendation": "STANDARD BREAKOUT — STANDARD ALLOCATION (1.0x)",
            }

        if self.clf_big is None or self.clf_trap is None:
            if not self.load_models():
                return {
                    "qm_score": 50.0,
                    "p_big": 0.25,
                    "p_trap": 0.23,
                    "tier": "Standard",
                    "sizing": 1.0,
                    "recommendation": "STANDARD BREAKOUT — STANDARD ALLOCATION (1.0x)",
                }

        row: Dict[str, float] = {}
        for col in QUALITY_FEATURE_COLS:
            val = features.get(col, 0.0) if hasattr(features, "get") else getattr(features, col, 0.0)
            if val is None or pd.isna(val):
                if col == "pinch_ratio":
                    val = 2.5
                elif col == "r2":
                    val = 0.5
                elif col == "rs_score":
                    val = 50.0
                elif col == "dist_52w_high":
                    val = -15.0
                elif col == "prior_ep_days_ago":
                    val = 999.0
                else:
                    val = 0.0
            row[col] = float(val)

        # Derive ratios if missing from raw setup row
        if row.get("pullback_ratio", 0.0) == 0.0:
            m_p = float(features.get("move_pct", 0.0) or 0.0) if hasattr(features, "get") else 0.0
            pb_p = float(features.get("pullback_pct", 0.0) or 0.0) if hasattr(features, "get") else 0.0
            row["pullback_ratio"] = (pb_p / m_p) if m_p > 0 else 0.5

        if row.get("breakout_rvol", 0.0) == 0.0:
            vx = float(features.get("vol_x", 1.0) or 1.0) if hasattr(features, "get") else 1.0
            row["breakout_rvol"] = vx

        df_in = pd.DataFrame([row])
        try:
            p_big = float(self.clf_big.predict_proba(df_in)[0, 1])
            p_trap = float(self.clf_trap.predict_proba(df_in)[0, 1])
        except Exception:
            p_big, p_trap = 0.25, 0.23

        score = float(np.clip((p_big / (p_big + p_trap + 1e-5)) * 100.0, 0.0, 100.0))

        if score >= 75.0 and p_trap < 0.22:
            tier = "Premier"
            sizing = 1.50
            rec = "PREMIER BREAKOUT — FULL CONVICTION (1.5x)"
        elif score >= 60.0 and p_trap < 0.28:
            tier = "High Quality"
            sizing = 1.20
            rec = "HIGH QUALITY SETUP — OVERWEIGHT (1.2x)"
        elif score >= 40.0 and p_trap < 0.35:
            tier = "Standard"
            sizing = 1.00
            rec = "STANDARD BREAKOUT — STANDARD ALLOCATION (1.0x)"
        elif p_trap >= 0.35 or score < 25.0:
            tier = "Toxic / Trap"
            sizing = 0.00
            rec = "TOXIC SETUP AVOIDED — HIGH FAILURE RISK"
        else:
            tier = "Marginal"
            sizing = 0.50
            rec = "MARGINAL SETUP — DEFENSIVE SIZING (0.5x)"

        return {
            "qm_score": round(score, 1),
            "p_big": round(p_big, 3),
            "p_trap": round(p_trap, 3),
            "tier": tier,
            "sizing": sizing,
            "recommendation": rec,
        }

    # -------------------------------------------------------------------------
    # 3. Dynamic Quantile Regression Stop Loss Prediction (QM-Stop)
    # -------------------------------------------------------------------------
    def predict_dynamic_stop(
        self,
        features: Union[Dict[str, Any], pd.DataFrame, pd.Series],
        entry_price: float,
        pivot_low_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Predicts volatility-informed dynamic quantile stop loss and structural boundaries."""
        if self.stop_regressor is None and not self.load_models():
            default_risk_pct = 5.0
            stop_px = entry_price * (1.0 - default_risk_pct / 100.0)
            return {
                "pred_mae_pct": round(default_risk_pct, 2),
                "dynamic_stop_price": round(stop_px, 2),
                "risk_pct": round(default_risk_pct, 2),
            }

        row: Dict[str, float] = {}
        for col in STOP_FEATURE_COLS:
            val = features.get(col, 0.0) if hasattr(features, "get") else getattr(features, col, 0.0)
            if val is None or pd.isna(val):
                if col == "pinch_ratio":
                    val = 2.5
                elif col == "r2":
                    val = 0.5
                elif col == "dist_52w_high":
                    val = -15.0
                elif col == "risk_pct_06x":
                    val = 5.0
                elif col == "risk_pct_10x":
                    val = 8.5
                else:
                    val = 0.0
            row[col] = float(val)

        df_in = pd.DataFrame([row])
        try:
            pred_mae = float(np.clip(self.stop_regressor.predict(df_in)[0], 0.015, 0.10))
        except Exception:
            pred_mae = 0.045

        raw_stop_px = entry_price * (1.0 - pred_mae)

        # Structural Boundaries:
        # Never wider than 0.6x Pivot Low; never tighter than 1.5% below entry
        if pivot_low_price is not None and 0 < pivot_low_price < entry_price:
            full_pivot = entry_price - pivot_low_price
            stop_06x = entry_price - 0.60 * full_pivot
        else:
            stop_06x = entry_price * 0.95

        bounded_stop_px = min(entry_price * 0.985, max(raw_stop_px, stop_06x))
        effective_risk_pct = (entry_price - bounded_stop_px) / entry_price * 100.0

        return {
            "pred_mae_pct": round(pred_mae * 100.0, 2),
            "dynamic_stop_price": round(bounded_stop_px, 2),
            "risk_pct": round(effective_risk_pct, 2),
        }

    # -------------------------------------------------------------------------
    # 4. Comprehensive Setup Guidance (For Fresh Setups & Dossier)
    # -------------------------------------------------------------------------
    def get_setup_guidance(
        self,
        features: Union[Dict[str, Any], pd.DataFrame, pd.Series],
        entry_price: float,
        pivot_low_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Provides complete institutional trade execution guidance for fresh or historical setups."""
        if isinstance(features, dict) and features.get("qm_tier"):
            tier = features["qm_tier"]
            score = float(features.get("qm_score", 50.0))
            p_big = float(features.get("p_big_move", features.get("p_big", 0.25)))
            p_trap = float(features.get("p_trap", 0.23))
            sizing = float(features.get("conviction_sizing", features.get("sizing", 1.0)))
            rec = features.get("ml_recommendation") or features.get("recommendation")
            if not rec:
                if tier == "Premier":
                    rec = "PREMIER BREAKOUT — FULL CONVICTION (1.5x)"
                elif tier == "High Quality":
                    rec = "HIGH QUALITY SETUP — OVERWEIGHT (1.2x)"
                elif tier == "Toxic / Trap":
                    rec = "TOXIC SETUP AVOIDED — HIGH FAILURE RISK"
                elif tier == "Marginal":
                    rec = "MARGINAL SETUP — DEFENSIVE SIZING (0.5x)"
                else:
                    rec = "STANDARD BREAKOUT — STANDARD ALLOCATION (1.0x)"
            quality = {
                "qm_score": round(score, 1),
                "p_big": round(p_big, 3),
                "p_trap": round(p_trap, 3),
                "tier": tier,
                "sizing": sizing,
                "recommendation": rec,
            }
        else:
            quality = self.predict_setup_quality(features)

        stop_info = self.predict_dynamic_stop(features, entry_price, pivot_low_price)

        init_risk = entry_price - stop_info["dynamic_stop_price"]
        climax_target_25r = entry_price + 2.5 * init_risk if init_risk > 0 else entry_price * 1.10
        climax_target_35r = entry_price + 3.5 * init_risk if init_risk > 0 else entry_price * 1.15

        return {
            "quality": quality,
            "stop_loss": stop_info,
            "targets": {
                "hurdle_25r": round(climax_target_25r, 2),
                "hurdle_35r": round(climax_target_35r, 2),
                "scale_out_plan": "Sell 50% partial into Climax High (P >= 0.45 or +2.5R), move stop to Breakeven, trail 50% runner on 10 EMA.",
            },
        }

    # -------------------------------------------------------------------------
    # 5. High-Performance Vectorized Batch Enrichment
    # -------------------------------------------------------------------------
    def enrich_setups_with_ml(self, df: pd.DataFrame, trades_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Enriches an entire setups dataframe (breakouts + building setups) with ML quality tiers & predictions."""
        if df.empty:
            return df

        out = df.copy()

        # Step 1: Map from existing strategy trades lookup if available
        mapping = {}
        if trades_df is not None and not trades_df.empty:
            cols = [c for c in ["qm_tier", "qm_score", "p_big_move", "p_trap", "conviction_sizing", "pred_mae_pct", "stop_loss_dynamic", "risk_pct_dynamic"] if c in trades_df.columns]
            if cols:
                mapping = trades_df.set_index(["symbol", "date"])[cols].to_dict("index")
        elif Path("data/simulations/qm_strategy_trades.parquet").exists():
            try:
                tdf = pd.read_parquet("data/simulations/qm_strategy_trades.parquet")
                cols = [c for c in ["qm_tier", "qm_score", "p_big_move", "p_trap", "conviction_sizing", "pred_mae_pct", "stop_loss_dynamic", "risk_pct_dynamic"] if c in tdf.columns]
                if cols:
                    mapping = tdf.set_index(["symbol", "date"])[cols].to_dict("index")
            except Exception:
                mapping = {}

        tiers = [None] * len(out)
        scores = [None] * len(out)
        p_bigs = [None] * len(out)
        p_traps = [None] * len(out)
        sizings = [None] * len(out)

        missing_indices = []
        sym_list = out["symbol"].astype(str).tolist()
        date_list = out["date"].astype(str).tolist()

        for idx, (sym, dt) in enumerate(zip(sym_list, date_list)):
            val = mapping.get((sym, dt))
            if val and val.get("qm_tier"):
                tiers[idx] = str(val["qm_tier"])
                scores[idx] = float(val["qm_score"]) if val.get("qm_score") is not None else 50.0
                p_bigs[idx] = float(val["p_big_move"]) if val.get("p_big_move") is not None else 0.25
                p_traps[idx] = float(val["p_trap"]) if val.get("p_trap") is not None else 0.23
                sizings[idx] = float(val["conviction_sizing"]) if val.get("conviction_sizing") is not None else 1.0
            else:
                missing_indices.append(idx)

        # Step 2: Vectorized prediction for remaining setups (e.g. building / fresh setups)
        if missing_indices and self.clf_big is not None and self.clf_trap is not None:
            sub = out.iloc[missing_indices].copy()
            if "pullback_ratio" not in sub.columns and "pullback_pct" in sub.columns and "move_pct" in sub.columns:
                m_pct = pd.to_numeric(sub["move_pct"], errors="coerce").fillna(0.0)
                pb_pct = pd.to_numeric(sub["pullback_pct"], errors="coerce").fillna(0.0)
                sub["pullback_ratio"] = np.where(m_pct > 0, pb_pct / m_pct, 0.5)

            if "breakout_rvol" not in sub.columns and "vol_x" in sub.columns:
                sub["breakout_rvol"] = pd.to_numeric(sub["vol_x"], errors="coerce").fillna(1.0)

            X_q = pd.DataFrame(index=sub.index)
            for col in QUALITY_FEATURE_COLS:
                if col in sub.columns:
                    s_col = sub[col]
                    if s_col.dtype == object or s_col.dtype == bool:
                        X_q[col] = s_col.astype(bool).astype(float).fillna(0.0)
                    else:
                        fill_v = 999.0 if col == "prior_ep_days_ago" else 0.0
                        X_q[col] = pd.to_numeric(s_col, errors="coerce").fillna(fill_v)
                elif col == "pinch_ratio":
                    X_q[col] = 2.5
                elif col == "r2":
                    X_q[col] = 0.5
                elif col == "rs_score":
                    X_q[col] = 50.0
                elif col == "dist_52w_high":
                    X_q[col] = -15.0
                elif col == "prior_ep_days_ago":
                    X_q[col] = 999.0
                else:
                    X_q[col] = 0.0

            X_q["pinch_ratio"] = X_q["pinch_ratio"].fillna(2.5)
            X_q["r2"] = X_q["r2"].fillna(0.5)
            X_q["rs_score"] = X_q["rs_score"].fillna(50.0)
            X_q["dist_52w_high"] = X_q["dist_52w_high"].fillna(-15.0)
            X_q["prior_ep_days_ago"] = X_q["prior_ep_days_ago"].fillna(999.0)

            try:
                p_b = self.clf_big.predict_proba(X_q)[:, 1]
                p_t = self.clf_trap.predict_proba(X_q)[:, 1]
                sc = np.clip((p_b / (p_b + p_t + 1e-5)) * 100.0, 0.0, 100.0)

                for i, orig_idx in enumerate(missing_indices):
                    s_val = sc[i]
                    pt_val = p_t[i]
                    if s_val >= 75.0 and pt_val < 0.22:
                        t_str = "Premier"
                        sz = 1.50
                    elif s_val >= 60.0 and pt_val < 0.28:
                        t_str = "High Quality"
                        sz = 1.20
                    elif s_val >= 40.0 and pt_val < 0.35:
                        t_str = "Standard"
                        sz = 1.00
                    elif pt_val >= 0.35 or s_val < 25.0:
                        t_str = "Toxic / Trap"
                        sz = 0.00
                    else:
                        t_str = "Marginal"
                        sz = 0.50

                    tiers[orig_idx] = t_str
                    scores[orig_idx] = round(float(s_val), 1)
                    p_bigs[orig_idx] = round(float(p_b[i]), 3)
                    p_traps[orig_idx] = round(float(pt_val), 3)
                    sizings[orig_idx] = sz
            except Exception as e:
                print(f"Warning: Batch quality prediction failed: {e}")

        # Default fallback for any remaining
        for idx in range(len(out)):
            if tiers[idx] is None:
                tiers[idx] = "Standard"
                scores[idx] = 50.0
                p_bigs[idx] = 0.25
                p_traps[idx] = 0.23
                sizings[idx] = 1.0

        out["qm_tier"] = tiers
        out["qm_score"] = scores
        out["p_big_move"] = p_bigs
        out["p_trap"] = p_traps
        out["conviction_sizing"] = sizings
        return out


def get_qm_ml_engine() -> QMMLEngine:
    return QMMLEngine.get_engine()
