"""
Unit and regression tests for ML Climax Exit and Dynamic Trailing Stop.
Tests cover zero-lookahead feature extraction, model inference, and dossier trade simulation.
Seed: 20261010
Date: 2026-10-10
"""
import unittest
import pandas as pd
import numpy as np
import os
from pathlib import Path
import serve_ep
from ep_ml_engine import EPMLEngine


class TestClimaxTrailer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = EPMLEngine()

    def test_parquet_climax_trailer_columns(self):
        """Verifies ep_combined_study_scored.parquet contains all required simulation columns."""
        path = Path("data/simulations/ep_combined_study_scored.parquet")
        self.assertTrue(path.exists(), "ep_combined_study_scored.parquet must exist")
        df = pd.read_parquet(path)
        required_cols = ["t1_climax_r", "t1_climax_triggered", "t1_dyn_r", "t1_climax_dyn_r"]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing column {col} in parquet")
        self.assertTrue((df["t1_climax_triggered"].dtype == bool) or (df["t1_climax_triggered"].dtype == np.bool_))
        self.assertGreater(df["t1_climax_triggered"].sum(), 0, "Expected positive climax triggers")

    def test_rolling_features_and_inference(self):
        """Tests that compute_rolling_features computes swing & parabolic features and models infer without error."""
        # Create synthetic price series with an EP at bar 30 and an explosive uptrend
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        prices = [10.0] * 30
        # Day 1 EP bar
        prices.append(12.0)
        # 29 bars of steady upward momentum
        for i in range(1, 30):
            prices.append(12.0 + i * 0.5)

        closes = np.array(prices, dtype=float)
        highs = closes + 0.3
        lows = closes - 0.3
        lows[30] = 11.0 # Day 1 low
        opens = closes - 0.1
        vols = np.full(len(closes), 1_000_000.0)
        vols[30] = 5_000_000.0 # High volume EP

        df = pd.DataFrame({
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vols,
            "rvol": np.full(len(closes), 2.0),
            "gap_pct": np.full(len(closes), 10.0),
            "close_pos": np.full(len(closes), 0.85),
            "adr_pts": np.full(len(closes), 0.5)
        }, index=dates)

        # Compute rolling features at bar 50 relative to EP at bar 30
        features = self.engine.compute_rolling_features("SYNTH", 30, 50, df=df)
        self.assertIsNotNone(features)

        # Check required climax & trailer features are present
        required_features = [
            "feature_days_since_ep", "feature_ret_since_ep", "feature_unrealized_r",
            "feature_gain_adr", "feature_drawdown_from_peak", "feature_dist_10ema",
            "feature_dist_20ema", "feature_dist_50sma", "feature_rvol",
            "feature_rvol_pivot_high", "feature_last_swing_retrace", "feature_mean_swing_retrace",
            "feature_swing_vol_ratio", "feature_parabolic_bars", "feature_thrust_accel",
            "feature_consecutive_gaps", "feature_gap_count_5d", "feature_gap_pct",
            "feature_close_pos", "feature_is_novel_vol_9m", "feature_is_blue_sky",
            "feature_dist_to_overhead_pct"
        ]
        for f in required_features:
            self.assertIn(f, features, f"Missing feature {f}")

        # Check prediction functions
        climax_prob = self.engine.predict_climax(features)
        self.assertIsInstance(climax_prob, float)
        self.assertTrue(0.0 <= climax_prob <= 1.0)

        trailer_buf = self.engine.predict_dynamic_trailer(features)
        self.assertIsInstance(trailer_buf, float)
        self.assertGreater(trailer_buf, 0.0)

    def test_compute_dossier_climax_and_trailer(self):
        """Verifies compute_dossier executes climax partial taking and dynamic trailing stop when requested."""
        dates = pd.date_range("2024-01-01", periods=80, freq="B")
        # Build an EP at bar 0 followed by huge 100% gain, climax blow-off, and pullback
        closes = [20.0]
        # Run from 20 to 45 over 40 bars (more than +100% gain, > +10R)
        for i in range(1, 40):
            closes.append(20.0 + i * 0.7)
        # Climax bar at bar 40 with parabolic extension
        closes.append(50.0)
        # Sharp pullback breaking trailing stop
        for i in range(41, 80):
            closes.append(max(25.0, 50.0 - (i - 40) * 1.2))

        closes = np.array(closes, dtype=float)
        highs = closes + 0.5
        lows = closes - 0.5
        lows[0] = 18.5 # Day 1 low (risk = $1.50)
        opens = closes - 0.1
        vols = np.full(len(closes), 2_000_000.0)

        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes, "volume": vols,
            "rvol": np.full(len(closes), 3.0), "gap_pct": np.full(len(closes), 15.0),
            "close_pos": np.full(len(closes), 0.90), "adr_pts": np.full(len(closes), 1.0)
        }, index=dates)

        # Baseline run (neither climax nor trailer active)
        dossier_base = serve_ep.compute_dossier("TEST", "2024-01-01", df, 0, None, use_dynamic_trailer=False, use_climax_exit=False)
        t1_base = dossier_base["trade_1"]
        self.assertFalse(t1_base.get("climax_exit_triggered", False))
        self.assertFalse(t1_base.get("dynamic_trailer_active", False))

        # Climax exit run
        dossier_climax = serve_ep.compute_dossier("TEST", "2024-01-01", df, 0, None, use_dynamic_trailer=False, use_climax_exit=True)
        t1_climax = dossier_climax["trade_1"]
        self.assertTrue(t1_climax["climax_exit_triggered"], "Expected climax exit to trigger on parabolic move")
        self.assertIsNotNone(t1_climax["climax_exit_price"])
        self.assertIsNotNone(t1_climax["climax_partial_r"])
        self.assertGreater(t1_climax["climax_partial_r"], 5.0)

        # Dynamic trailer run
        dossier_trail = serve_ep.compute_dossier("TEST", "2024-01-01", df, 0, None, use_dynamic_trailer=True, use_climax_exit=False)
        t1_trail = dossier_trail["trade_1"]
        self.assertTrue(t1_trail["dynamic_trailer_active"], "Expected dynamic trailer to activate")
        self.assertIsNotNone(t1_trail["dynamic_trailer_stop"])
        self.assertEqual(t1_trail["exit_reason"], "ML Trailing Stop Hit")

    def test_immediate_stop_edge_case(self):
        """Verifies immediate Day 1 low breach halts trade cleanly without activating climax or trailer."""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        closes = np.array([20.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0], dtype=float)
        lows = np.array([19.0, 16.5, 15.5, 14.5, 13.5, 12.5, 11.5, 10.5, 9.5, 8.5], dtype=float)
        highs = closes + 0.5
        opens = closes + 0.2
        vols = np.full(len(closes), 1_000_000.0)

        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes, "volume": vols,
            "rvol": np.full(len(closes), 2.0), "gap_pct": np.full(len(closes), 10.0),
            "close_pos": np.full(len(closes), 0.8), "adr_pts": np.full(len(closes), 1.0)
        }, index=dates)

        dossier = serve_ep.compute_dossier("STOP_TEST", "2024-01-01", df, 0, None, use_dynamic_trailer=True, use_climax_exit=True)
        t1 = dossier["trade_1"]
        self.assertTrue(t1["stopped_out"])
        self.assertEqual(t1["exit_reason"], "Day 1 Low Stop Hit")
        self.assertFalse(t1["climax_exit_triggered"])
        self.assertFalse(t1["dynamic_trailer_active"])
        self.assertLess(t1["trade_r"], 0.0)

    def test_sub_hurdle_consolidation_edge_case(self):
        """Verifies trade fluctuating below the +3.0R activation hurdle never activates trailing stop."""
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        # Entry 20.0, stop 18.0 (risk 2.0 pts). To reach +3.0R price needs to hit 26.0.
        # Fluctuate between 20.0 and 23.0 (+1.5R max)
        closes = [20.0]
        for i in range(1, 30):
            closes.append(20.0 + (i % 5) * 0.6)
        closes = np.array(closes, dtype=float)
        highs = closes + 0.2
        lows = closes - 0.2
        lows[0] = 18.0
        opens = closes
        vols = np.full(len(closes), 1_000_000.0)

        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes, "volume": vols,
            "rvol": np.full(len(closes), 1.5), "gap_pct": np.full(len(closes), 8.0),
            "close_pos": np.full(len(closes), 0.7), "adr_pts": np.full(len(closes), 0.8)
        }, index=dates)

        dossier = serve_ep.compute_dossier("SUB_HURDLE", "2024-01-01", df, 0, None, use_dynamic_trailer=True, use_climax_exit=True)
        t1 = dossier["trade_1"]
        self.assertFalse(t1["climax_exit_triggered"])
        self.assertFalse(t1["dynamic_trailer_active"])

    def test_zero_lookahead_invariance(self):
        """Strict causality test: rolling features at bar t must be identical whether computed on full series or series truncated at t."""
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        np.random.seed(42)
        base_trend = np.linspace(20.0, 35.0, 60)
        c = base_trend + np.random.uniform(0.1, 0.5, 60)
        c[10] = 22.0 # EP bar
        df_full = pd.DataFrame({
            "open": c, "high": c * 1.02, "low": c * 0.98, "close": c,
            "volume": np.random.uniform(500_000, 2_000_000, 60),
            "rvol": np.full(60, 2.0), "gap_pct": np.full(60, 10.0),
            "close_pos": np.full(60, 0.85), "adr_pts": np.full(60, 1.0)
        }, index=dates)
        df_full.iloc[10, df_full.columns.get_loc("low")] = 20.0
        # Ensure all subsequent lows stay above 20.0
        df_full.iloc[11:, df_full.columns.get_loc("low")] = np.maximum(df_full.iloc[11:]["low"], 20.5)

        t_eval = 40
        df_trunc = df_full.iloc[:t_eval + 1].copy()

        feats_full = self.engine.compute_rolling_features("TEST", 10, t_eval, df=df_full)
        feats_trunc = self.engine.compute_rolling_features("TEST", 10, t_eval, df=df_trunc)

        self.assertIsNotNone(feats_full)
        self.assertIsNotNone(feats_trunc)

        numeric_keys = [k for k in feats_full if isinstance(feats_full[k], (int, float, np.floating, np.integer))]
        for k in numeric_keys:
            v_full = float(feats_full[k])
            v_trunc = float(feats_trunc[k])
            if np.isnan(v_full):
                self.assertTrue(np.isnan(v_trunc), f"Mismatch NaN on feature {k}")
            else:
                self.assertAlmostEqual(v_full, v_trunc, places=4, msg=f"Lookahead leak detected on feature {k}")

    def test_sub_sma50_ep_hold_no_premature_breakdown(self):
        """Regression test: verifies that an EP occurring below the 50 SMA does not prematurely exit on Day 2/3.
        Seed: 20261010, Date: 2026-10-10.
        """
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        # Stock starts at $10.00 with 50 SMA at $15.00 (below 50 SMA)
        # It runs up from $10.00 to $14.00, never breaking Day 1 low ($9.00)
        closes = [10.0 + i * 0.15 for i in range(30)]
        closes = np.array(closes, dtype=float)
        highs = closes + 0.3
        lows = closes - 0.3
        lows[0] = 9.0 # Day 1 low
        opens = closes - 0.1
        vols = np.full(len(closes), 1_000_000.0)

        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes, "volume": vols,
            "rvol": np.full(len(closes), 2.0), "gap_pct": np.full(len(closes), 10.0),
            "close_pos": np.full(len(closes), 0.85), "adr_pts": np.full(len(closes), 0.5),
            "sma50": np.full(len(closes), 15.0), # 50 SMA remains above price
            "ribbon_state": ["yellow"] * len(closes) # Bullish ribbon
        }, index=dates)

        dossier = serve_ep.compute_dossier("SUB_SMA50", "2024-01-01", df, 0, None)
        t1 = dossier["trade_1"]
        # Must NOT exit on Day 2 or Day 3 with "50 SMA Breakdown"
        self.assertFalse(t1["stopped_out"])
        self.assertNotEqual(t1["exit_reason"], "50 SMA Breakdown")
        self.assertGreaterEqual(t1["hold_days"], 25)

    def test_model_inference_with_missing_and_partial_features(self):
        """Regression test: verifies predict_climax and predict_dynamic_trailer handle empty and partial features cleanly.
        Seed: 20261010, Date: 2026-10-10.
        """
        # Empty features
        p_empty = self.engine.predict_climax({})
        self.assertIsInstance(p_empty, float)
        self.assertTrue(0.0 <= p_empty <= 1.0)

        b_empty = self.engine.predict_dynamic_trailer({})
        self.assertIsInstance(b_empty, float)
        self.assertGreater(b_empty, 0.0)

        # Partial features
        partial = {"feature_unrealized_r": 3.5, "feature_dist_20ema": 0.25}
        p_part = self.engine.predict_climax(partial)
        self.assertIsInstance(p_part, float)
        self.assertTrue(0.0 <= p_part <= 1.0)

        b_part = self.engine.predict_dynamic_trailer(partial)
        self.assertIsInstance(b_part, float)
        self.assertGreater(b_part, 0.0)


if __name__ == "__main__":
    unittest.main()

