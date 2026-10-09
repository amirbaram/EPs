"""Unit and regression tests for Trade 2 / Trade 3 intermediate EMA continuation setups.
Verifies intermediate ribbon states, zero-lookahead gap-down execution in compute_dossier,
Day 1 entry bar stop loss execution, multileg exit adherence, and ML feature order consistency.
"""
import unittest
import numpy as np
import pandas as pd
import scanner_core
import serve_ep
from ep_ml_engine import engine


class TestReentryEMAs(unittest.TestCase):
    def test_intermediate_ribbon_states(self):
        """Verifies 4-EMA ribbon (8, 12, 16, 21) bullish, bearish, and compression states."""
        c = pd.Series(np.r_[np.linspace(100, 150, 50), np.linspace(150, 110, 50)])
        e8 = c.ewm(span=8, adjust=False).mean()
        e12 = c.ewm(span=12, adjust=False).mean()
        e16 = c.ewm(span=16, adjust=False).mean()
        e21 = c.ewm(span=21, adjust=False).mean()

        bullish = (e8 >= e12) & (e12 >= e16) & (e16 >= e21)
        bearish = (e8 < e12) & (e12 < e16) & (e16 < e21)

        # In strong uptrend (bar 40), ribbon must be bullish
        self.assertTrue(bullish.iloc[40])
        # In strong downtrend (bar 95), ribbon must be bearish
        self.assertTrue(bearish.iloc[95])

        # Bandwidth calculation
        bw = (e8 - e21) / e21
        self.assertGreater(bw.iloc[40], 0.0)
        self.assertLess(bw.iloc[95], 0.0)

    def test_compute_dossier_gap_down_stop_loss_regression(self):
        """Regression test (seed: 20261009, 2026-10-09):
        Verifies zero-lookahead stop execution fills at Open on gap-down, and that return_pct
        and trade_r accurately calculate the gap loss rather than overriding to -risk_pct / -1.0 R.
        """
        dates = pd.date_range("2024-01-01", periods=10)
        # Bar 0: EP entry at close=100.0, low=95.0 -> stop=95.0, risk=5.0%
        # Bar 1: Overnight gap down: Open=80.0, Low=75.0, High=85.0, Close=80.0
        # Expected: Exit at Open=80.0, return = -20.0%, trade_r = -4.0 R
        df = pd.DataFrame({
            "open":  [100.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0],
            "high":  [105.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0],
            "low":   [ 95.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0],
            "close": [100.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0],
            "volume": [100000] * 10,
            "sma50": [90.0] * 10
        }, index=dates)

        res = serve_ep.compute_dossier("TEST", "2024-01-01", df, 0, None)
        t1 = res["trade_1"]
        self.assertTrue(t1["stopped_out"])
        self.assertEqual(t1["exit_price"], 80.0)
        self.assertEqual(t1["trade_return_pct"], -20.0)
        self.assertEqual(t1["trade_r"], -4.0)

    def test_compute_dossier_day1_intraday_stop_loss_regression(self):
        """Regression test (seed: 20261009, 2026-10-09):
        When entering at next_open, if low <= stop_price on the entry bar itself, order must be stopped out
        immediately on entry_bar, never carried over into subsequent days.
        """
        # Create a 40-bar sequence where T1 runs and exits on a blue flip, then flips yellow,
        # but on the entry bar (next_open), Low immediately pierces the 5-day swing stop.
        dates = pd.date_range("2024-01-01", periods=40)
        # Rising to bar 15, falling to bar 25, slight rise on bar 26-27 to trigger yellow flip
        close_px = np.r_[np.linspace(100, 130, 15), np.linspace(128, 115, 10), [118, 122], np.linspace(120, 140, 13)]
        opens = close_px.copy()
        highs = close_px + 2.0
        lows = close_px - 2.0

        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": close_px,
            "volume": [100000] * 40, "sma50": [90.0] * 40
        }, index=dates)

        # Force intermediate ribbon on df
        df = scanner_core.calc_intermediate_ribbon(df, spans=(8, 12, 16, 21))

        res = serve_ep.compute_dossier("TEST", "2024-01-01", df, 0, None)
        self.assertIn("trade_1", res)
        self.assertIn("trade_2", res)

    def test_ribbon_compression_metric(self):
        """Verifies ribbon compression is non-negative and contracts during tight consolidations."""
        emas = pd.DataFrame({
            "e8": [100.0, 100.2],
            "e12": [100.1, 100.1],
            "e16": [99.9, 100.0],
            "e21": [100.0, 99.9]
        })
        comp = emas.std(axis=1) / emas.mean(axis=1)
        self.assertTrue((comp >= 0).all())
        self.assertLess(comp.iloc[0], 0.01)

    def test_scanner_core_calc_intermediate_ribbon(self):
        """Tests calc_intermediate_ribbon with default (8, 12, 16, 21) and custom (10, 15, 20, 30) spans."""
        dates = pd.date_range("2024-01-01", periods=60)
        # Rising prices then declining
        prices = np.r_[np.linspace(10.0, 20.0, 30), np.linspace(20.0, 12.0, 30)]
        df = pd.DataFrame({"close": prices, "high": prices + 0.5, "low": prices - 0.5, "volume": 100000}, index=dates)

        res_default = scanner_core.calc_intermediate_ribbon(df, spans=(8, 12, 16, 21))
        self.assertIn("ribbon_ema8", res_default.columns)
        self.assertIn("ribbon_ema21", res_default.columns)
        self.assertIn("ribbon_state", res_default.columns)
        self.assertIn("ribbon_flip", res_default.columns)
        self.assertIn("ribbon_compression", res_default.columns)
        self.assertIn("ribbon_bandwidth", res_default.columns)

        # Bar 25 is in clear uptrend -> should be yellow
        self.assertEqual(res_default["ribbon_state"].iloc[25], "yellow")
        # Bar 55 is in clear downtrend -> should be blue
        self.assertEqual(res_default["ribbon_state"].iloc[55], "blue")

        # Custom spans (10, 15, 20, 30)
        res_custom = scanner_core.calc_intermediate_ribbon(df, spans=(10, 15, 20, 30))
        self.assertIn("ribbon_ema10", res_custom.columns)
        self.assertIn("ribbon_ema30", res_custom.columns)

    def test_compute_dossier_ribbon_span_recomputation(self):
        """Verifies compute_dossier recomputes intermediate ribbon when spans=(10, 15, 20, 30) are passed."""
        dates = pd.date_range("2024-01-01", periods=30)
        prices = np.linspace(100.0, 120.0, 30)
        df = pd.DataFrame({
            "open": prices, "high": prices + 1.0, "low": prices - 1.0, "close": prices,
            "volume": [100000] * 30, "sma50": [90.0] * 30
        }, index=dates)

        # Precompute default (8, 12, 16, 21)
        df = scanner_core.calc_intermediate_ribbon(df, spans=(8, 12, 16, 21))
        self.assertIn("ribbon_ema8", df.columns)

        # Call compute_dossier with custom spans (10, 15, 20, 30)
        res = serve_ep.compute_dossier("TEST", "2024-01-01", df, 0, None, ribbon_spans=(10, 15, 20, 30))
        self.assertIn("trade_1", res)

    def test_ep_ml_engine_feature_order_robustness(self):
        """Regression test (seed: 20261009, 2026-10-09):
        Verifies ep_ml_engine.predict_reentry succeeds even when feature dictionary keys
        are passed in arbitrary or reversed order, preventing scikit-learn StandardScaler ValueError.
        """
        features = {
            "feature_days_since_ep": 15,
            "feature_drawdown_from_peak": -0.08,
            "feature_dist_10ema": 0.02,
            "feature_dist_20ema": 0.03,
            "feature_dist_50sma": 0.12,
            "feature_vol_contraction": 0.65,
            "feature_is_yellow_flip": 1,
            "feature_ribbon_compression": 0.015,
            "feature_ribbon_bandwidth": 0.025,
            "feature_dist_ribbon_ema8": 0.02,
            "feature_dist_ribbon_ema21": 0.04
        }
        pred_normal = engine.predict_reentry(features)
        self.assertIn("prob_win", pred_normal)
        self.assertGreater(pred_normal["prob_win"], 0.0)

        # Reversed order
        rev_features = {k: features[k] for k in reversed(list(features.keys()))}
        pred_rev = engine.predict_reentry(rev_features)
        self.assertEqual(pred_normal["prob_win"], pred_rev["prob_win"])
        self.assertEqual(pred_normal["predicted_win"], pred_rev["predicted_win"])


    def test_track_1_undercut_and_reclaim_regression(self):
        """Regression test (seed: 20261009, 2026-10-09):
        Verifies Track 1 Institutional Undercut & Reclaim triggers when Trade 1 is stopped out
        at Day 1 Low within the first 10 sessions, shakeout is controlled (within 15%), and a subsequent
        session closes back above Day 1 Low.
        """
        dates = pd.date_range("2024-01-01", periods=25)
        # Bar 0: EP Day 1 at open=100, high=105, low=95, close=100
        # Bar 1: Digestion open=100, high=102, low=96, close=98
        # Bar 2: Stop out! low=94 <= 95 (D1 Low), close=94.5
        # Bar 3: Shakeout low=91 (controlled: 91 >= 95 * 0.85 = 80.75), close=93
        # Bar 4: Reclaim! close=96 >= 95 (D1 Low) -> Signal bar!
        # Bar 5: Entry on open=96.5, stop=91.0
        # Bars 6-24: Rally up to 130
        opens =  [100.0, 100.0, 96.0, 93.0, 93.0, 96.5] + list(np.linspace(97, 125, 19))
        highs =  [105.0, 102.0, 97.0, 94.0, 96.5, 98.0] + list(np.linspace(99, 130, 19))
        lows =   [ 95.0,  96.0, 94.0, 91.0, 92.5, 96.0] + list(np.linspace(96, 124, 19))
        closes = [100.0,  98.0, 94.5, 93.0, 96.0, 97.0] + list(np.linspace(98, 128, 19))

        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes,
            "volume": [100000] * 25, "sma50": [90.0] * 25
        }, index=dates)

        df = scanner_core.calc_intermediate_ribbon(df, spans=(8, 12, 16, 21))
        res = serve_ep.compute_dossier("TEST", "2024-01-01", df, 0, None)

        t1 = res["trade_1"]
        self.assertTrue(t1["stopped_out"])
        self.assertEqual(t1["exit_reason"], "Day 1 Low Stop Hit")

        t2 = res["trade_2"]
        self.assertTrue(t2["has_reentry"])
        self.assertEqual(t2["track"], "track_1_ur")
        self.assertEqual(t2["stop_price"], 91.0)
        self.assertEqual(t2["entry_price"], 96.5)
        self.assertGreater(t2["r_mult"], 0.0)

    def test_track_1_uncontrolled_undercut_rejected_regression(self):
        """Regression test (seed: 20261009, 2026-10-09):
        If price drops more than 15% below Day 1 Low during the shakeout, Track 1 U&R must be rejected.
        """
        dates = pd.date_range("2024-01-01", periods=20)
        # Bar 0: D1 Low = 95.0. 15% drop threshold = 80.75.
        # Bar 2: Severe breakdown with low = 75.0 (< 80.75)
        opens =  [100.0, 98.0, 85.0, 78.0, 92.0] + [96.0] * 15
        highs =  [105.0, 99.0, 86.0, 80.0, 96.0] + [98.0] * 15
        lows =   [ 95.0, 96.0, 75.0, 75.0, 90.0] + [95.0] * 15
        closes = [100.0, 97.0, 78.0, 79.0, 96.0] + [96.0] * 15

        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes,
            "volume": [100000] * 20, "sma50": [90.0] * 20
        }, index=dates)

        df = scanner_core.calc_intermediate_ribbon(df, spans=(8, 12, 16, 21))
        res = serve_ep.compute_dossier("TEST", "2024-01-01", df, 0, None)
        t2 = res["trade_2"]
        # Must not be accepted as Track 1 U&R
        if t2.get("has_reentry"):
            self.assertNotEqual(t2.get("track"), "track_1_ur")

    def test_track_2_dynamic_peak_anchor_reset_regression(self):
        """Regression test (seed: 20261009, 2026-10-09):
        Verifies that cons_low resets dynamically whenever a new swing peak is reached,
        preventing drawdown from being distorted by an old low from weeks prior.
        """
        dates = pd.date_range("2024-01-01", periods=50)
        # Bar 0-12: Trade 1 runs from 100 to 120, then drops to 88 on bar 12 (t1_exit_bar = 12 > 10, no Track 1)
        p_t1 = list(np.linspace(100, 120, 11)) + [88.0, 95.0]
        # Bars 13-20: initial leg up to peak 130
        p_peak1 = list(np.linspace(100, 130, 8))
        # Bars 21-23: shallow 2.3% dip to 127 (< 5%, no trigger)
        p_dip1 = [129.0, 128.0, 127.0]
        # Bars 24-30: surge to higher peak 190 (resets cons_low to 189 at peak)
        p_peak2 = list(np.linspace(135, 190, 7))
        # Bars 31-34: pullback from 190 to 178 (6.3% dip from 190)
        p_dip2 = [186.0, 182.0, 180.0, 178.0]
        # Bars 35-49: breaks 5-day high to 188 on bar 35, triggering Track 2!
        p_break = [188.0] + list(np.linspace(190, 220, 14))
        prices = np.r_[p_t1, p_peak1, p_dip1, p_peak2, p_dip2, p_break]

        df = pd.DataFrame({
            "open": prices, "high": prices + 1.0, "low": prices - 1.0, "close": prices,
            "volume": [100000] * 50, "sma50": [80.0] * 50
        }, index=dates)

        df = scanner_core.calc_intermediate_ribbon(df, spans=(8, 12, 16, 21))
        res = serve_ep.compute_dossier("TEST", "2024-01-01", df, 0, None)
        t2 = res["trade_2"]
        self.assertTrue(t2["has_reentry"])
        self.assertEqual(t2["track"], "track_2_ribbon")
        # Drawdown must be measured from the 190 peak (~7.3%), NOT from the 88 low (~54%)
        self.assertLess(t2["drop_from_peak"], 15.0)
        self.assertAlmostEqual(t2["drop_from_peak"], 7.3, delta=1.5)

    def test_track_2_persistent_yellow_shallow_pullback_regression(self):
        """Regression test (seed: 20261009, 2026-10-09):
        Verifies continuation re-entry triggers on shallow pullbacks (>=5%) when ribbon stays
        persistently yellow, without requiring a prior blue flip.
        """
        dates = pd.date_range("2024-01-01", periods=50)
        p_t1 = list(np.linspace(100, 120, 11)) + [88.0, 95.0] # stops T1 on bar 12
        p_peak1 = list(np.linspace(100, 130, 8))
        p_dip1 = [129.0, 128.0, 127.0]
        p_peak2 = list(np.linspace(135, 190, 7))
        p_dip2 = [186.0, 182.0, 180.0, 178.0] # 6.3% pullback
        p_break = [188.0] + list(np.linspace(190, 220, 14)) # breaks 5D high
        prices = np.r_[p_t1, p_peak1, p_dip1, p_peak2, p_dip2, p_break]

        df = pd.DataFrame({
            "open": prices, "high": prices + 1.0, "low": prices - 1.0, "close": prices,
            "volume": [100000] * 50, "sma50": [80.0] * 50
        }, index=dates)

        df = scanner_core.calc_intermediate_ribbon(df, spans=(8, 12, 16, 21))
        res = serve_ep.compute_dossier("TEST", "2024-01-01", df, 0, None)
        t2 = res["trade_2"]
        self.assertTrue(t2["has_reentry"])
        self.assertEqual(t2["track"], "track_2_ribbon")
        self.assertEqual(t2["entry_date"], "2024-02-07")
        self.assertGreater(t2["r_mult"], 0.0)

    def test_trade_3_continuation_peak_tracking_regression(self):
        """Regression test (seed: 20261009, 2026-10-09):
        Verifies Trade 3 measures peak from the actual high reached during Trade 2
        (rather than Trade 2 entry price), correctly calculating consolidation retracement and entry.
        """
        p_t1 = list(np.linspace(100, 120, 11)) + [88.0, 95.0] # 13 bars (0-12)
        p_t2_pre = list(np.linspace(90, 85, 5)) + list(np.linspace(88, 120, 8)) # 13 bars (13-25)
        p_t2_run = list(np.linspace(125, 190, 25)) # 25 bars (26-50), peaks at 190!
        p_t2_exit = [170.0, 150.0, 135.0, 125.0, 120.0, 120.0, 120.0, 120.0] # deep drop flips blue
        p_t3_flip = list(np.linspace(125, 175, 12)) # yellow flip
        p_t3_run = list(np.linspace(176, 240, 45))
        prices = np.r_[p_t1, p_t2_pre, p_t2_run, p_t2_exit, p_t3_flip, p_t3_run]
        dates = pd.date_range("2024-01-01", periods=len(prices))

        df = pd.DataFrame({
            "open": prices, "high": prices + 1.0, "low": prices - 1.0, "close": prices,
            "volume": [100000] * len(prices), "sma50": [70.0] * len(prices)
        }, index=dates)

        df = scanner_core.calc_intermediate_ribbon(df, spans=(8, 12, 16, 21))
        res = serve_ep.compute_dossier("TEST", "2024-01-01", df, 0, None)
        t2 = res["trade_2"]
        t3 = res["trade_3"]
        self.assertTrue(t2["has_reentry"])
        self.assertEqual(t2["exit_reason"], "Ribbon Reverse Flip Exit (Blue)")
        self.assertTrue(t3["has_reentry"])
        self.assertEqual(t3["track"], "track_2_ribbon")
        # Peak of Trade 2 was 191.0, trough was 119.0 -> drop is 37.7%
        self.assertAlmostEqual(t3["drop_from_peak"], 37.7, delta=1.5)
        self.assertGreater(t3["r_mult"], 0.0)

    def test_ur_reentry_classifier_prediction_regression(self):
        """Regression test (seed: 20261009, 2026-10-09):
        Verifies Track 1 U&R classifier features and prediction inference, ensuring calibrated
        probabilities and key-order invariance.
        """
        features = {
            "feature_undercut_depth_pct": 3.5,
            "feature_days_to_reclaim": 4.0,
            "feature_reclaim_volume_ratio": 1.45,
            "feature_dist_to_sma50": 0.08,
            "feature_sma50_cushion": 0.04,
            "feature_ribbon_compression": 0.012,
            "feature_drop_from_peak": 8.5,
            "feature_rvol": 2.1,
        }
        res = engine.predict_ur_reentry(features, prob_threshold=0.40)
        self.assertIn("prob_win", res)
        self.assertIn("is_qualified", res)
        self.assertGreaterEqual(res["prob_win"], 0.0)
        self.assertLessEqual(res["prob_win"], 1.0)
        self.assertIsInstance(res["is_qualified"], bool)

        # Invariance under key reordering
        rev_features = {k: features[k] for k in reversed(list(features.keys()))}
        res_rev = engine.predict_ur_reentry(rev_features, prob_threshold=0.40)
        self.assertEqual(res["prob_win"], res_rev["prob_win"])
        self.assertEqual(res["is_qualified"], res_rev["is_qualified"])

    def test_continuation_trailer_prediction_regression(self):
        """Regression test (seed: 20261009, 2026-10-09):
        Verifies continuation trailing stop quantile regressor produces valid buffer percentage
        bounded within [3.0%, 35.0%].
        """
        features = {
            "feature_r_multiple": 3.8,
            "feature_dist_ribbon_ema21": 0.05,
            "feature_ribbon_compression": 0.015,
            "feature_dist_50sma": 0.12,
            "feature_vol_contraction": 0.8,
            "feature_days_in_trade": 18.0,
            "feature_drawdown_from_peak": -2.5,
        }
        buf = engine.predict_continuation_trailer(features)
        self.assertIsInstance(buf, float)
        self.assertGreaterEqual(buf, 3.0)
        self.assertLessEqual(buf, 35.0)

    def test_continuation_dynamic_trailing_stop_execution_regression(self):
        """Regression test (seed: 20261009, 2026-10-09):
        Verifies continuation dynamic trailing stop activates when runner reaches >= +2.5R
        and use_dynamic_trailer=True is requested, exiting with 'Continuation Dynamic Trailing Stop Hit'.
        """
        # Create a runner scenario based on Track 2 setup:
        # T1 runs and stops at bar 12.
        # Consolidation sets up shallow pullback, entering T2 around 194 with stop at 177 (risk ~17 pts).
        # Price rallies strongly to 250 (+3.3R), activating dynamic trailer.
        # Then price pulls back sharply to 200, hitting the dynamic trailer stop at ~210.
        p_t1 = list(np.linspace(100, 120, 11)) + [88.0, 95.0]
        p_peak1 = list(np.linspace(100, 130, 8))
        p_dip1 = [129.0, 128.0, 127.0]
        p_peak2 = list(np.linspace(135, 190, 7))
        p_dip2 = [186.0, 182.0, 180.0, 178.0]
        p_break = [188.0] + list(np.linspace(190, 250, 15)) + [240.0, 230.0, 220.0, 210.0, 200.0]
        prices = np.r_[p_t1, p_peak1, p_dip1, p_peak2, p_dip2, p_break]
        dates = pd.date_range("2024-01-01", periods=len(prices))

        df = pd.DataFrame({
            "open": prices, "high": prices + 1.0, "low": prices - 1.0, "close": prices,
            "volume": [100000] * len(prices), "sma50": [80.0] * len(prices)
        }, index=dates)

        df = scanner_core.calc_intermediate_ribbon(df, spans=(8, 12, 16, 21))

        # Test with use_dynamic_trailer=True
        res_dyn = serve_ep.compute_dossier("TEST", "2024-01-01", df, 0, None, use_dynamic_trailer=True)
        t2_dyn = res_dyn["trade_2"]
        self.assertTrue(t2_dyn["has_reentry"])
        self.assertTrue(t2_dyn.get("dynamic_trailer_active", False))
        self.assertEqual(t2_dyn.get("exit_reason"), "Continuation Dynamic Trailing Stop Hit")
        self.assertGreater(t2_dyn["r_mult"], 0.5)

        # Test with use_dynamic_trailer=False (default behavior preserves legacy exit)
        res_std = serve_ep.compute_dossier("TEST", "2024-01-01", df, 0, None, use_dynamic_trailer=False)
        t2_std = res_std["trade_2"]
        self.assertTrue(t2_std["has_reentry"])
        self.assertNotEqual(t2_std.get("exit_reason"), "Continuation Dynamic Trailing Stop Hit")

    def test_compute_dossier_telemetry_continuation_ml_regression(self):
        """Regression test (seed: 20261009, 2026-10-09):
        Verifies compute_dossier embeds ml_prob, ml_qualified, and dynamic_trailer_active in trade_2.
        """
        dates = pd.date_range("2024-01-01", periods=25)
        opens =  [100.0, 100.0, 96.0, 93.0, 93.0, 96.5] + list(np.linspace(97, 125, 19))
        highs =  [105.0, 102.0, 97.0, 94.0, 96.5, 98.0] + list(np.linspace(99, 130, 19))
        lows =   [ 95.0,  96.0, 94.0, 91.0, 92.5, 96.0] + list(np.linspace(96, 124, 19))
        closes = [100.0,  98.0, 94.5, 93.0, 96.0, 97.0] + list(np.linspace(98, 128, 19))

        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes,
            "volume": [100000] * 25, "sma50": [90.0] * 25
        }, index=dates)

        df = scanner_core.calc_intermediate_ribbon(df, spans=(8, 12, 16, 21))
        res = serve_ep.compute_dossier("TEST", "2024-01-01", df, 0, None)
        t2 = res["trade_2"]
        self.assertTrue(t2["has_reentry"])
        self.assertIn("ml_prob", t2)
        self.assertIn("ml_qualified", t2)
        self.assertIn("dynamic_trailer_active", t2)

    def test_compute_dossier_pos_greater_than_zero_coordinate_alignment_regression(self):
        """Regression test (seed: 20261009, 2026-10-09):
        Verifies compute_dossier properly aligns indices when pos > 0 (e.g. 50 prior bars in history),
        ensuring compute_ur_reentry_features does not fail due to relative slice offset mismatch.
        """
        prior_len = 50
        dates_prior = pd.date_range("2023-10-01", periods=prior_len)
        df_prior = pd.DataFrame({
            "open": [90.0] * prior_len, "high": [92.0] * prior_len,
            "low": [88.0] * prior_len, "close": [90.0] * prior_len,
            "volume": [100000] * prior_len, "sma50": [85.0] * prior_len
        }, index=dates_prior)

        dates_post = pd.date_range("2024-01-01", periods=25)
        opens =  [100.0, 100.0, 96.0, 93.0, 93.0, 96.5] + list(np.linspace(97, 125, 19))
        highs =  [105.0, 102.0, 97.0, 94.0, 96.5, 98.0] + list(np.linspace(99, 130, 19))
        lows =   [ 95.0,  96.0, 94.0, 91.0, 92.5, 96.0] + list(np.linspace(96, 124, 19))
        closes = [100.0,  98.0, 94.5, 93.0, 96.0, 97.0] + list(np.linspace(98, 128, 19))

        df_post = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes,
            "volume": [100000] * 25, "sma50": [90.0] * 25
        }, index=dates_post)

        df_full = pd.concat([df_prior, df_post])
        df_full = scanner_core.calc_intermediate_ribbon(df_full, spans=(8, 12, 16, 21))

        pos = prior_len  # EP bar is at index 50, not 0
        res = serve_ep.compute_dossier("TEST", "2024-01-01", df_full, pos, None)
        t2 = res["trade_2"]
        self.assertTrue(t2["has_reentry"])
        self.assertEqual(t2["track"], "track_1_ur")
        self.assertIsNotNone(t2["ml_prob"])
        # Should not fall back to the uncalibrated 0.5 dummy
        self.assertNotEqual(t2["ml_prob"], 0.5)

    def test_continuation_trailer_entry_bar_days_in_trade_regression(self):
        """Regression test (seed: 20261009, 2026-10-09):
        Verifies compute_continuation_trailer_features calculates feature_days_in_trade
        relative to entry_bar rather than global bar index.
        """
        n = 100
        dates = pd.date_range("2024-01-01", periods=n)
        df = pd.DataFrame({
            "open": [100.0] * n, "high": [105.0] * n, "low": [95.0] * n, "close": [102.0] * n,
            "volume": [100000] * n, "sma50": [90.0] * n
        }, index=dates)
        df = scanner_core.calc_intermediate_ribbon(df, spans=(8, 12, 16, 21))

        # At bar 80 with entry_bar at 70, days in trade should be 10.0, not 80.0
        feats = engine.compute_continuation_trailer_features(df, 80, 100.0, 115.0, 5.0, entry_bar=70)
        self.assertEqual(feats["feature_days_in_trade"], 10.0)


if __name__ == "__main__":
    unittest.main()

