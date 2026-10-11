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

    def test_evaluate_exits_harmonization_regression(self):
        """Regression test (seed: 20261010, Date: 2026-10-10):
        Verifies evaluate_exits.py harmonizes the 22 TrendLab confirmed swing, parabolic,
        and volume features without ValueError, verifying both exit models evaluate cleanly.
        """
        import evaluate_exits
        # Test loading/extracting with sample limit
        df_exits = evaluate_exits.load_or_extract_features(max_samples=25)
        self.assertIsNotNone(df_exits)
        self.assertGreater(len(df_exits), 0)

        required_cols = list(self.engine.exhaustion_classifier.feature_names_in_)
        for col in required_cols:
            self.assertIn(col, df_exits.columns, f"Missing required exit feature {col}")

        # Check model predictions on harmonized features
        X = df_exits[required_cols]
        probs = self.engine.exhaustion_classifier.predict_proba(X)[:, 1]
        self.assertEqual(len(probs), len(df_exits))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

        preds_stop = self.engine.dynamic_stop.predict(X)
        self.assertEqual(len(preds_stop), len(df_exits))
        self.assertTrue(np.all(preds_stop > 0.0))

        # Programmatic CLI execution verification
        metrics = evaluate_exits.evaluate(argv=["--max-samples", "25"])
        self.assertIn("exhaustion", metrics)
        self.assertIn("dynamic_trailer", metrics)
        self.assertGreater(metrics["dynamic_trailer"]["mae"], 0.0)
        self.assertGreater(metrics["dynamic_trailer"]["empirical_coverage"], 0.0)

    def test_configurable_climax_and_trailer_regression(self):
        """Regression test (seed: 20261010, Date: 2026-10-10):
        Verifies configurable climax partial profit % (e.g. 25% vs 75%),
        configurable R-hurdle thresholds (e.g. 2.0R vs 10.0R),
        and explicit dynamic trailer percentage visibility per ticker.
        """
        dates = pd.date_range("2024-01-01", periods=80, freq="B")
        closes = [20.0]
        for i in range(1, 40):
            closes.append(20.0 + i * 0.7)
        closes.append(50.0)
        for i in range(41, 80):
            closes.append(max(25.0, 50.0 - (i - 40) * 1.2))

        closes = np.array(closes, dtype=float)
        highs = closes + 0.5
        lows = closes - 0.5
        lows[0] = 18.5
        opens = closes - 0.1
        vols = np.full(len(closes), 2_000_000.0)

        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes, "volume": vols,
            "rvol": np.full(len(closes), 3.0), "gap_pct": np.full(len(closes), 15.0),
            "close_pos": np.full(len(closes), 0.90), "adr_pts": np.full(len(closes), 1.0)
        }, index=dates)

        # 1. Configurable Climax Partial % (25% vs 75%)
        dossier_25 = serve_ep.compute_dossier("TEST", "2024-01-01", df, 0, None,
                                              use_dynamic_trailer=False, use_climax_exit=True,
                                              climax_partial_pct=0.25, climax_r_hurdle=3.0)
        t1_25 = dossier_25["trade_1"]
        self.assertTrue(t1_25["climax_exit_triggered"])
        self.assertEqual(t1_25["climax_partial_pct"], 25.0)
        self.assertEqual(t1_25["climax_r_hurdle"], 3.0)

        dossier_75 = serve_ep.compute_dossier("TEST", "2024-01-01", df, 0, None,
                                              use_dynamic_trailer=False, use_climax_exit=True,
                                              climax_partial_pct=0.75, climax_r_hurdle=3.0)
        t1_75 = dossier_75["trade_1"]
        self.assertTrue(t1_75["climax_exit_triggered"])
        self.assertEqual(t1_75["climax_partial_pct"], 75.0)

        # Verify exact weighted blend: selling 25% vs 75% reflects the respective weighting
        self.assertNotEqual(t1_75["trade_r"], t1_25["trade_r"])
        self.assertAlmostEqual(t1_25["trade_r"], 0.25 * t1_25["climax_partial_r"] + 0.75 * 10.4, places=1)
        self.assertAlmostEqual(t1_75["trade_r"], 0.75 * t1_75["climax_partial_r"] + 0.25 * 10.4, places=1)

        # 2. Configurable R-Hurdle: Hurdle higher than peak unrealized R prevents climax exit
        dossier_high_hurdle = serve_ep.compute_dossier("TEST", "2024-01-01", df, 0, None,
                                                      use_dynamic_trailer=False, use_climax_exit=True,
                                                      climax_r_hurdle=50.0)
        self.assertFalse(dossier_high_hurdle["trade_1"]["climax_exit_triggered"])

        # 3. Dynamic Trailer Visibility: Percentage & Stop Level
        dossier_trail = serve_ep.compute_dossier("TEST", "2024-01-01", df, 0, None,
                                                use_dynamic_trailer=True, use_climax_exit=False,
                                                climax_r_hurdle=3.0)
        t1_trail = dossier_trail["trade_1"]
        self.assertTrue(t1_trail["dynamic_trailer_active"])
        self.assertIsNotNone(t1_trail["dynamic_trailer_stop"])
        self.assertIsNotNone(t1_trail["dynamic_trailer_pct"])
        self.assertGreater(t1_trail["dynamic_trailer_pct"], 0.0)
        self.assertEqual(t1_trail["exit_reason"], "ML Trailing Stop Hit")

        # 4. API Endpoints with Flask Test Client
        client = serve_ep.app.test_client()
        resp_ev = client.get("/api/events?climax_partial_pct=0.33&climax_r_hurdle=4.0")
        self.assertEqual(resp_ev.status_code, 200)
        data_ev = resp_ev.get_json()
        self.assertIn("events", data_ev)
        self.assertIn("kpis", data_ev)

        # 5. Dynamic Trailer honoring ev_row pre-scored values
        ev_mock = {"dynamic_stop_loss_pct": -0.095}
        dossier_mock = serve_ep.compute_dossier("TEST", "2024-01-01", df, 0, ev_mock,
                                                use_dynamic_trailer=True, use_climax_exit=False)
        self.assertAlmostEqual(dossier_mock["trade_1"]["dynamic_trailer_pct"], 9.5, places=1)

    def test_delayed_breakout_trade1_and_table_events_regression(self):
        """Verifies Delayed Breakout Trade 1 dossier payload, entry date, stop sanity, and /api/events strat_r."""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        # Day 0 (2024-01-01): EP day, Open 10.0, Low 9.5, High 12.0, Close 11.5
        # Day 1 (2024-01-02): Breakout bar, Open 11.8, Low 11.0, High 12.5 (breaches 12.0), Close 12.2
        # Day 2+ (2024-01-03+): Continuation bars
        df_breakout = pd.DataFrame({
            "open": [10.0, 11.8, 12.2, 12.5, 12.8, 13.0, 13.2, 13.5, 13.8, 14.0],
            "high": [12.0, 12.5, 12.8, 13.0, 13.5, 13.8, 14.0, 14.2, 14.5, 14.8],
            "low": [9.5, 11.0, 11.8, 12.0, 12.3, 12.6, 12.8, 13.0, 13.3, 13.5],
            "close": [11.5, 12.2, 12.6, 12.9, 13.3, 13.6, 13.9, 14.1, 14.3, 14.6],
            "volume": [1_000_000] * 10,
            "rvol": [3.0] * 10,
            "gap_pct": [10.0] * 10,
            "close_pos": [0.8] * 10
        }, index=dates)

        # 1. Delayed Breakout with successful entry
        dossier_delayed = serve_ep.compute_dossier("TEST", "2024-01-01", df_breakout, 0, None, t1_mode="delayed")
        t1_del = dossier_delayed["trade_1"]
        self.assertFalse(t1_del["skipped"])
        self.assertEqual(t1_del["entry_mode"], "delayed")
        self.assertEqual(t1_del["entry_date"], "2024-01-02")
        self.assertEqual(t1_del["entry_price"], 12.0)
        self.assertLessEqual(t1_del["stop_price"], t1_del["entry_price"])
        self.assertGreaterEqual(t1_del["stop_price"], 9.5)

        # 2. Delayed Breakout when Day 1 Low is breached (Cancelled / Defense Active)
        df_cancelled = df_breakout.copy()
        df_cancelled.loc[dates[1], "low"] = 9.0  # Breaches Day 1 low (9.5)
        dossier_canc = serve_ep.compute_dossier("TEST", "2024-01-01", df_cancelled, 0, None, t1_mode="delayed")
        t1_canc = dossier_canc["trade_1"]
        self.assertTrue(t1_canc["skipped"])
        self.assertEqual(t1_canc["entry_mode"], "delayed")
        self.assertIsNone(t1_canc["entry_date"])
        self.assertIn("Cancelled / Defense Active", t1_canc["exit_reason"])
        self.assertEqual(t1_canc["trade_r"], 0.0)

        # 3. Standard Day 1 Close Entry
        dossier_d1 = serve_ep.compute_dossier("TEST", "2024-01-01", df_breakout, 0, None, t1_mode="d1_close")
        t1_d1 = dossier_d1["trade_1"]
        self.assertFalse(t1_d1["skipped"])
        self.assertEqual(t1_d1["entry_mode"], "d1_close")
        self.assertEqual(t1_d1["entry_date"], "2024-01-01")
        self.assertEqual(t1_d1["entry_price"], 11.5)

        # 4. Delayed Breakout with use_trade1=False
        dossier_not1 = serve_ep.compute_dossier("TEST", "2024-01-01", df_breakout, 0, None, t1_mode="delayed", use_trade1=False)
        self.assertTrue(dossier_not1["trade_1"]["skipped"])
        self.assertEqual(dossier_not1["trade_1"]["entry_mode"], "delayed")
        self.assertIsNone(dossier_not1["trade_1"]["entry_date"])

        # 5. Stop sanity calculation with positive / scaled dynamic stop loss pct
        ev_pos_stop = {"dynamic_stop_loss_pct": 0.08}
        dossier_pos = serve_ep.compute_dossier("TEST", "2024-01-01", df_breakout, 0, ev_pos_stop, t1_mode="delayed", use_ml_stop=True)
        self.assertLess(dossier_pos["trade_1"]["stop_price"], dossier_pos["trade_1"]["entry_price"])
        self.assertGreaterEqual(dossier_pos["trade_1"]["stop_price"], 9.5)

        # 6. HTML template verification for Card 4 IDs and Events table header
        self.assertIn('id="lbl_tr_entry"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="lbl_tr_add"', serve_ep.HTML_TEMPLATE)
        self.assertIn('>Strategy R</th>', serve_ep.HTML_TEMPLATE)
        self.assertIn('colspan="15"', serve_ep.HTML_TEMPLATE)

        # 7. /api/events endpoint returns strat_r and strat_ret for active mode
        client = serve_ep.app.test_client()
        resp = client.get("/api/events?t1_mode=delayed")
        self.assertEqual(resp.status_code, 200)
        ev_data = resp.get_json()
        self.assertIn("events", ev_data)
        if len(ev_data["events"]) > 0:
            first_ev = ev_data["events"][0]
            self.assertIn("strat_r", first_ev)
            self.assertIn("strat_ret", first_ev)
            self.assertTrue(isinstance(first_ev["strat_r"], (int, float)))
            self.assertTrue(isinstance(first_ev["strat_ret"], (int, float)))

    def test_v2_filter_card_and_drawing_tools_regression(self):
        """Verifies V2 filter endpoint, chart header metrics, V2 card elements, and drawing tools overlay."""
        client = serve_ep.app.test_client()

        # 1. Verify /api/chart returns non-zero gap_pct and rvol for DELL
        resp = client.get("/api/chart?symbol=DELL&date=2026-02-27")
        self.assertEqual(resp.status_code, 200)
        c_data = resp.get_json()
        self.assertIn("gap_pct", c_data)
        self.assertIn("rvol", c_data)
        self.assertGreater(c_data["gap_pct"], 0.0)
        self.assertGreater(c_data["rvol"], 0.0)

        # 2. Verify /api/events with v2 filter (e.g. v2=20 -> ml_50 >= 0.20)
        resp_v2 = client.get("/api/events?v2=20")
        self.assertEqual(resp_v2.status_code, 200)
        v2_events = resp_v2.get_json().get("events", [])
        for ev in v2_events[:20]:
            if ev.get("ml_50") is not None:
                self.assertGreaterEqual(ev["ml_50"], 0.20)

        # 3. Verify HTML template contains V2 card, filter dropdown, and drawing canvas
        self.assertIn('id="f_v2"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="card_v2_ml"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="chart_draw_toolbar"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="draw_canvas"', serve_ep.HTML_TEMPLATE)
        self.assertIn('btn_tool_measure', serve_ep.HTML_TEMPLATE)
        self.assertIn('btn_tool_line', serve_ep.HTML_TEMPLATE)
        self.assertIn('btn_tool_text', serve_ep.HTML_TEMPLATE)
        self.assertIn('btn_tool_arrow', serve_ep.HTML_TEMPLATE)

    def test_full_history_larsson_and_filtered_setups_regression(self):
        """Verifies full history bar loading, 10 EMA, auto-proportional Larsson ribbon, and filter-aware ticker setups."""
        client = serve_ep.app.test_client()

        # 1. Full history loading: verifies bars span back to inception (>500 bars), not just 40 bars
        resp = client.get("/api/chart?symbol=DELL&date=2026-02-27")
        self.assertEqual(resp.status_code, 200)
        c_data = resp.get_json()
        self.assertIn("bars", c_data)
        self.assertGreater(len(c_data["bars"]), 500)
        self.assertIn("event_index", c_data)
        self.assertGreater(c_data["event_index"], 40)

        # 2. 10 EMA returned and matched with bars
        self.assertIn("ema10", c_data)
        self.assertGreater(len(c_data["ema10"]), 500)
        self.assertIn("time", c_data["ema10"][0])
        self.assertIn("value", c_data["ema10"][0])

        # 3. Larsson line calculation with default fastest 8 -> spans [8, 12, 16, 21]
        self.assertIn("larsson", c_data)
        self.assertEqual(c_data["larsson_spans"], [8, 12, 16, 21])
        states = {x["state"] for x in c_data["larsson"]}
        self.assertTrue("yellow" in states or "gray" in states)

        # 4. Larsson line calculation with custom fastest 10 -> spans [10, 15, 20, 26]
        resp_custom = client.get("/api/chart?symbol=DELL&date=2026-02-27&larsson_fastest=10")
        self.assertEqual(resp_custom.status_code, 200)
        c_custom = resp_custom.get_json()
        self.assertEqual(c_custom["larsson_spans"], [10, 15, 20, 26])

        # 5. Larsson line calculation with custom fastest 32 (traditional) -> spans [32, 35, 50, 58]
        resp_trad = client.get("/api/chart?symbol=DELL&date=2026-02-27&larsson_fastest=32")
        self.assertEqual(resp_trad.status_code, 200)
        c_trad = resp_trad.get_json()
        self.assertEqual(c_trad["larsson_spans"], [32, 35, 50, 58])

        # 6. Filter-aware setups per ticker
        resp_unfiltered = client.get("/api/chart?symbol=NVDA&date=2024-05-23")
        c_unf = resp_unfiltered.get_json()
        unf_count = len(c_unf.get("ticker_history", []))

        resp_filtered = client.get("/api/chart?symbol=NVDA&date=2024-05-23&v2=50")
        c_filt = resp_filtered.get_json()
        filt_count = len(c_filt.get("ticker_history", []))
        self.assertLessEqual(filt_count, unf_count)

        # 7. HTML controls present
        self.assertIn('id="chk_ema10"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="chk_larsson"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="cfg_larsson_fastest"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="lbl_larsson_spans"', serve_ep.HTML_TEMPLATE)

    def test_larsson_ribbon_svg_and_fast_slow_readout_regression(self):
        """Verifies Larsson Line SVG ribbon structure, fast/slow boundaries, and readout values."""
        client = serve_ep.app.test_client()
        resp = client.get("/api/chart?symbol=DELL&date=2026-02-27&larsson_fastest=32")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        self.assertIn("larsson", data)
        self.assertEqual(data["larsson_spans"], [32, 35, 50, 58])
        l_bars = data["larsson"]
        self.assertGreater(len(l_bars), 100)

        # Check structure of larsson bar items
        last_item = l_bars[-1]
        self.assertIn("time", last_item)
        self.assertIn("fast", last_item)
        self.assertIn("slow", last_item)
        self.assertIn("value", last_item)
        self.assertIn("state", last_item)
        self.assertIn(last_item["state"], ["yellow", "blue", "gray"])
        self.assertIsInstance(last_item["fast"], (int, float))
        self.assertIsInstance(last_item["slow"], (int, float))

        # Check frontend SVG ribbon elements and legend values in HTML template
        self.assertIn('id="larsson_ribbon_svg"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="larsson_ribbon_group"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="lbl_larsson_vals"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="larsson_val_fast"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="larsson_val_slow"', serve_ep.HTML_TEMPLATE)
        self.assertIn('function renderLarssonRibbon(', serve_ep.HTML_TEMPLATE)
        self.assertIn('function updateLarssonLegendValues(', serve_ep.HTML_TEMPLATE)

    def test_chart_settings_panel_and_marking_toggles_regression(self):
        """Verifies settings panel, EMA 20/50 toggles, T1/T2/T3 markings, and EP event markers."""
        client = serve_ep.app.test_client()
        resp = client.get("/api/chart?symbol=DELL&date=2026-02-27&v2=50")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        # 1. API chart payload verification
        self.assertIn("all_ticker_events", data)
        self.assertIsInstance(data["all_ticker_events"], list)
        self.assertGreater(len(data["all_ticker_events"]), 0)
        first_ep = data["all_ticker_events"][0]
        self.assertIn("date", first_ep)
        self.assertIn("gap_pct", first_ep)
        self.assertIn("rvol", first_ep)
        self.assertIn("max_gain", first_ep)
        self.assertIn("is_qualified", first_ep)
        self.assertIn("disqualifications", first_ep)

        # 2. Moving Average toggles in HTML
        self.assertIn('id="chk_ema20"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="chk_sma50"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="chk_settings_ema20"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="chk_settings_sma50"', serve_ep.HTML_TEMPLATE)

        # 3. Settings panel trigger buttons and floating panel
        self.assertIn('id="btn_chart_settings"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="btn_tool_settings"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="chart_settings_panel"', serve_ep.HTML_TEMPLATE)
        self.assertIn('toggleChartSettings()', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="chk_settings_show_reasons"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="hud_ep_reason"', serve_ep.HTML_TEMPLATE)
        self.assertIn('toggleShowFilterReasons()', serve_ep.HTML_TEMPLATE)

        # 4. Trade marking checkboxes
        self.assertIn('id="chk_mark_t1"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="chk_mark_t2"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="chk_mark_t3"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="chk_mark_eps"', serve_ep.HTML_TEMPLATE)
        self.assertIn('renderChartMarkersAndLines', serve_ep.HTML_TEMPLATE)

        # 5. Exit parameter dropdowns inside settings panel
        self.assertIn('id="cfg_settings_climax_pct"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="cfg_settings_climax_hurdle"', serve_ep.HTML_TEMPLATE)

        # 6. Controls hidden when chart is not enlarged to prevent overflow
        self.assertIn('class="chart-enlarged-only-controls"', serve_ep.HTML_TEMPLATE)
        self.assertIn('.chart-enlarged-only-controls', serve_ep.HTML_TEMPLATE)

    def test_daily_exhaustion_hover_and_settings_toggle_regression(self):
        """Verifies daily ML exhaustion calculation, API payload, HUD elements, and settings toggle."""
        client = serve_ep.app.test_client()
        resp = client.get("/api/chart?symbol=DELL&date=2026-02-27")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        # 1. API payload contains exhaustion dictionary
        self.assertIn("exhaustion", data)
        self.assertIsInstance(data["exhaustion"], dict)
        if len(data["exhaustion"]) > 0:
            first_day = next(iter(data["exhaustion"].values()))
            self.assertIn("prob", first_day)
            self.assertIn("pct", first_day)
            self.assertIn("level", first_day)

        # 2. HTML elements for hover readout & crosshair HUD
        self.assertIn('id="lbl_hover_exhaustion"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="val_hover_exhaustion"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="chart_hover_hud"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="chk_settings_show_exhaustion"', serve_ep.HTML_TEMPLATE)

        # 3. JavaScript handler functions
        self.assertIn('function toggleShowExhaustion(', serve_ep.HTML_TEMPLATE)
        self.assertIn('function updateExhaustionHover(', serve_ep.HTML_TEMPLATE)

    def test_forward_excursion_ml_and_prev_gap_filters_regression(self):
        """Verifies 20-session forward excursion model, prev-gap filtering in API/UI, and chart marker toggles."""
        client = serve_ep.app.test_client()

        # 1. Forward excursion model inference
        from ep_ml_engine import engine
        import datastore
        bars = datastore.load_bars("NVDA")
        self.assertIsNotNone(bars)
        pos = bars.index.get_loc("2023-05-25")
        feats = engine.compute_rolling_features("NVDA", pos, pos + 5, df=bars)
        self.assertIsNotNone(feats)
        exc = engine.predict_forward_excursion(feats)
        self.assertIn("pred_mfe_20d", exc)
        self.assertIn("pred_mae_20d", exc)
        self.assertIn("prob_mfe_first", exc)
        self.assertIsInstance(exc["pred_mfe_20d"], float)
        self.assertIsInstance(exc["pred_mae_20d"], float)
        self.assertIsInstance(exc["prob_mfe_first"], float)
        self.assertGreaterEqual(exc["prob_mfe_first"], 0.0)
        self.assertLessEqual(exc["prob_mfe_first"], 1.0)
        self.assertGreater(exc["pred_mfe_20d"], 0.0)

        rolling_preds = engine.predict_rolling(feats)
        self.assertIn("forward_mfe_20d", rolling_preds)
        self.assertIn("forward_mae_20d", rolling_preds)
        self.assertIn("prob_mfe_first", rolling_preds)

        # 2. API chart returns forward excursion
        resp = client.get("/api/chart?symbol=NVDA&date=2023-05-25")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("forward_mfe_20d", data)
        self.assertIn("forward_mae_20d", data)
        self.assertIn("prob_mfe_first", data)
        self.assertIn("vol_ma20", data)
        self.assertIsInstance(data["vol_ma20"], list)
        self.assertGreater(len(data["vol_ma20"]), 0)
        self.assertIsNotNone(data["forward_mfe_20d"])

        # 3. Prior day gap filtering in /api/events
        ev_all_resp = client.get("/api/events")
        self.assertEqual(ev_all_resp.status_code, 200)
        all_events = ev_all_resp.get_json()["events"]
        self.assertGreater(len(all_events), 0)

        ev_nogap_resp = client.get("/api/events?prev_gap=no_gap")
        self.assertEqual(ev_nogap_resp.status_code, 200)
        nogap_events = ev_nogap_resp.get_json()["events"]
        self.assertGreater(len(nogap_events), 0)
        self.assertLess(len(nogap_events), len(all_events))
        for ev in nogap_events[:50]:
            self.assertFalse(ev.get("prev_day_gap_up", False))

        ev_gapup_resp = client.get("/api/events?prev_gap=gap_up")
        self.assertEqual(ev_gapup_resp.status_code, 200)
        gapup_events = ev_gapup_resp.get_json()["events"]
        self.assertGreater(len(gapup_events), 0)
        for ev in gapup_events[:50]:
            self.assertTrue(ev.get("prev_day_gap_up", False))

        # 4. Booster condition cond_no_prev_gap
        booster_resp = client.get("/api/events?cond_no_prev_gap=1")
        self.assertEqual(booster_resp.status_code, 200)
        booster_events = booster_resp.get_json()["events"]
        self.assertEqual(len(booster_events), len(nogap_events))

        # 5. UI elements in HTML template
        self.assertIn('id="f_prev_gap"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="b_no_prev_gap"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="chk_mark_all_eps"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="chk_mark_filt_eps"', serve_ep.HTML_TEMPLATE)
        self.assertIn('onToggleAllEpsChange()', serve_ep.HTML_TEMPLATE)
        self.assertIn('onToggleFiltEpsChange()', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="ai_v2_mfe"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="ai_v2_mae"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="ai_v2_seq"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="row_v2_seq_20d"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="v2_mfe_20d_val"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="v2_mae_20d_val"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="v2_seq_20d_val"', serve_ep.HTML_TEMPLATE)
        self.assertIn('id="chk_settings_vol_ma20"', serve_ep.HTML_TEMPLATE)
        self.assertIn('toggleVolMa20()', serve_ep.HTML_TEMPLATE)

    def test_predictive_targets_when_stopped_out_regression(self):
        """Verifies that ML predictive targets are computed and displayed even if trade stopped out."""
        from ep_ml_engine import engine
        import datastore
        bars = datastore.load_bars("BB")
        self.assertIsNotNone(bars)
        d1_idx = bars.index.get_loc("2016-09-28")
        
        # 1. Without allow_stopped, low breach returns None
        feats_strict = engine.compute_rolling_features("BB", d1_idx, d1_idx + 5, df=bars, allow_stopped=False)
        self.assertIsNone(feats_strict)

        # 2. With allow_stopped=True, features are computed and flagged is_stopped_out=True
        feats_allowed = engine.compute_rolling_features("BB", d1_idx, d1_idx + 5, df=bars, allow_stopped=True)
        self.assertIsNotNone(feats_allowed)
        self.assertTrue(feats_allowed.get("is_stopped_out", False))

        # 3. Predictions produce full valid numbers
        preds = engine.predict_rolling(feats_allowed)
        self.assertIn("prob_50", preds)
        self.assertIn("prob_100", preds)
        self.assertIn("forward_mfe_20d", preds)
        self.assertIn("forward_mae_20d", preds)
        self.assertTrue(preds.get("is_stopped_out", False))
        self.assertIsInstance(preds["prob_50"], float)
        self.assertIsInstance(preds["forward_mfe_20d"], float)

        # 4. /api/ml_rolling endpoint returns probs and is_stopped flag
        client = serve_ep.app.test_client()
        resp = client.get("/api/ml_rolling?symbol=BB&event_date=2016-09-28&days_forward=5")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("probs", data)
        self.assertTrue(data.get("is_stopped", False))
        self.assertIn("prob_50", data["probs"])
        self.assertIn("forward_mfe_20d", data["probs"])
        self.assertNotIn("stopped_out", data)

        # 5. HTML template contains the STOPPED badge and retains target numbers
        self.assertIn('Day ${days} <span style="background:#ef4444; color:#fff; padding:1px 5px; border-radius:4px; font-size:9px; margin-left:4px; font-weight:bold;">STOPPED</span>', serve_ep.HTML_TEMPLATE)

    def test_compute_daily_exhaustion_on_stopped_out_trade(self):
        """Regression test: verifies compute_daily_exhaustion returns valid numeric exhaustion
        probabilities and is_stopped=True on bars after Day 1 Low is breached.
        """
        import serve_ep
        import datastore
        import indicators
        bars = datastore.load_bars("BB")
        self.assertIsNotNone(bars)
        d = indicators.add_indicators(bars)
        d1_idx = d.index.get_loc("2016-09-28")

        result = serve_ep.compute_daily_exhaustion(d, d1_idx, max_bars=10)
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

        # BB breached Day 1 low within 10 days
        stopped_days = [date_str for date_str, info in result.items() if info.get("is_stopped")]
        self.assertGreater(len(stopped_days), 0, "BB should have stopped out days in forward window")
        for s_dt in stopped_days:
            info = result[s_dt]
            self.assertTrue(info["is_stopped"])
            self.assertIn("prob", info)
            self.assertIn("pct", info)
            self.assertNotEqual(info["level"], "Stopped")
            self.assertIn(info["level"], ["Low", "Elevated", "High", "🚨 Climax"])
            self.assertGreaterEqual(info["prob"], 0.0)
            self.assertLessEqual(info["prob"], 1.0)

    def test_ticker_p_recent_eps_in_dataset_and_chart(self):
        """Regression test: verifies ticker P includes recent Sept 18 and Sept 24, 2026
        Delayed Reaction EP (DRE) events in ep_combined_study_scored.parquet and /api/chart.
        """
        import serve_ep
        import pandas as pd
        from pathlib import Path

        # 1. Dataset verification
        df = serve_ep.get_df()
        self.assertFalse(df.empty)
        p_rows = df[df["symbol"] == "P"].set_index("date")
        self.assertIn("2026-09-18", p_rows.index, "P 2026-09-18 must exist in dataset")
        self.assertIn("2026-09-24", p_rows.index, "P 2026-09-24 must exist in dataset")

        row_18 = p_rows.loc["2026-09-18"]
        row_24 = p_rows.loc["2026-09-24"]
        self.assertEqual(row_18["subtype"], "dre")
        self.assertEqual(row_24["subtype"], "dre")
        self.assertTrue(row_18["t1_delayed_has_entry"])
        self.assertTrue(row_24["t1_delayed_has_entry"])
        self.assertGreater(float(row_18["t1_delayed_r"]), 0.0)
        self.assertGreater(float(row_24["t1_delayed_r"]), 0.0)

        # 2. Client API verification
        client = serve_ep.app.test_client()
        resp = client.get("/api/chart?symbol=P&date=2026-09-18")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        all_dates = [e["date"] for e in data.get("all_ticker_events", [])]
        self.assertIn("2026-09-18", all_dates)
        self.assertIn("2026-09-24", all_dates)

        # 3. Dossier verification for Delayed Breakout
        tr1 = data.get("dossier", {}).get("trade_1", {})
        self.assertFalse(tr1.get("skipped", True))
        self.assertAlmostEqual(float(tr1.get("entry_price")), 108.31, places=1)

    def test_api_events_recent_days_filter(self):
        """Regression test: verifies /api/events?recent_days=N correctly filters events
        to only those occurring within the most recent N days of the latest dataset session.
        """
        import serve_ep
        import pandas as pd

        client = serve_ep.app.test_client()

        # 1. Full baseline
        resp_all = client.get("/api/events?preset=all")
        self.assertEqual(resp_all.status_code, 200)
        data_all = resp_all.get_json()
        n_all = len(data_all["events"])
        self.assertGreater(n_all, 1000)

        # 2. Recent 30 days
        resp_30 = client.get("/api/events?preset=all&recent_days=30")
        self.assertEqual(resp_30.status_code, 200)
        data_30 = resp_30.get_json()
        ev_30 = data_30["events"]
        self.assertGreater(len(ev_30), 0)
        self.assertLess(len(ev_30), n_all)

        max_dt = max(e["date"] for e in ev_30)
        min_dt_30 = min(e["date"] for e in ev_30)
        cutoff_30 = (pd.to_datetime(max_dt) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        for e in ev_30:
            self.assertGreaterEqual(e["date"], cutoff_30, f"Event {e['symbol']} on {e['date']} is older than {cutoff_30}")

        # 3. Recent 5 days
        resp_5 = client.get("/api/events?preset=all&recent_days=5")
        self.assertEqual(resp_5.status_code, 200)
        data_5 = resp_5.get_json()
        ev_5 = data_5["events"]
        self.assertGreater(len(ev_5), 0)
        self.assertLessEqual(len(ev_5), len(ev_30))

        cutoff_5 = (pd.to_datetime(max_dt) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        for e in ev_5:
            self.assertGreaterEqual(e["date"], cutoff_5)

        # 4. Safe fallback on non-integer/invalid value
        resp_inv = client.get("/api/events?preset=all&recent_days=invalid")
        self.assertEqual(resp_inv.status_code, 200)
        self.assertEqual(len(resp_inv.get_json()["events"]), n_all)

    def test_api_chart_adr14_response_and_hover_payload(self):
        """Regression test: verifies that /api/chart returns 14-day ADR pts and pct payload for hover display.
        Seed: 20261011, Date: 2026-10-11.
        """
        import serve_ep
        client = serve_ep.app.test_client()
        df = serve_ep.get_df()
        self.assertFalse(df.empty)

        # Pick the first valid symbol and date from DF
        sample_row = df.iloc[0]
        sym = sample_row["symbol"]
        dt = sample_row["date"]

        resp = client.get(f"/api/chart?symbol={sym}&date={dt}")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("adr14", data)
        self.assertIsInstance(data["adr14"], dict)

        # Ensure that date exists in adr14 map and contains pts and pct
        if dt in data["adr14"]:
            val = data["adr14"][dt]
            self.assertIn("pts", val)
            self.assertIn("pct", val)
            self.assertGreater(val["pts"], 0.0)
            self.assertGreater(val["pct"], 0.0)

    def test_historical_catalyst_and_sec_filings_enrichment(self):
        """Regression test: verifies historical EP catalyst classification and SEC EDGAR filing attachment in /api/chart.
        Seed: 20261011, Date: 2026-10-11.
        """
        import serve_ep
        import ep_catalyst_enricher
        client = serve_ep.app.test_client()

        # 1. Verify ep_catalyst_enricher direct lookup
        cat = ep_catalyst_enricher.get_catalyst_dossier("SMMT", "2026-09-03", sector="Biotechnology", theme="Biotechnology")
        self.assertIsInstance(cat, dict)
        self.assertEqual(cat["symbol"], "SMMT")
        self.assertIn("category", cat)
        self.assertIn("headline", cat)
        self.assertIn("sec_filings", cat)
        self.assertIsInstance(cat["sec_filings"], list)

        # 2. Verify /api/chart returns dossier with catalyst and filings
        resp = client.get("/api/chart?symbol=SMMT&date=2026-09-03")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("dossier", data)
        self.assertIn("catalyst", data["dossier"])
        d_cat = data["dossier"]["catalyst"]
        self.assertIn("category", d_cat)
        self.assertIn("headline", d_cat)
        self.assertIn("tier", d_cat)
        self.assertIn("analysis", d_cat)
        self.assertIn("sec_filings", d_cat)


if __name__ == "__main__":
    unittest.main()



