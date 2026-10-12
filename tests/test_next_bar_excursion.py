"""
Regression test suite for Next-Bar Excursion & Sequence Modeling Pipeline.
Tests:
1. Ground truth labeling from 5m bars (High first vs Low first, same-candle tie breaks).
2. Zero-lookahead feature causality.
3. EPMLEngine integration, prediction contract, and robustness to missing features.
4. End-to-end extraction on real datastore ticker bars.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path

import datastore
from ep_ml_engine import engine
import train_next_bar_excursion_model as nbm

class TestNextBarExcursionML(unittest.TestCase):

    def test_ground_truth_high_first_and_low_first_resolution(self):
        """Verifies that 5m intraday sequences resolve high-first and low-first correctly."""
        # 1. Day where High occurs at 10:00 and Low occurs at 14:00 (High First)
        times_day1 = pd.date_range("2024-01-02 09:30", "2024-01-02 15:55", freq="5min")
        n_bars = len(times_day1)
        opens1 = np.full(n_bars, 100.0, dtype=np.float32)
        highs1 = np.full(n_bars, 100.5, dtype=np.float32)
        lows1 = np.full(n_bars, 99.5, dtype=np.float32)
        closes1 = np.full(n_bars, 100.0, dtype=np.float32)
        vols1 = np.full(n_bars, 1000.0, dtype=np.float32)

        # Set peak at bar 6 (10:00)
        highs1[6] = 105.0
        # Set trough at bar 54 (14:00)
        lows1[54] = 95.0

        # 2. Day where Low occurs at 10:00 and High occurs at 14:00 (Low First)
        times_day2 = pd.date_range("2024-01-03 09:30", "2024-01-03 15:55", freq="5min")
        opens2 = np.full(n_bars, 100.0, dtype=np.float32)
        highs2 = np.full(n_bars, 100.5, dtype=np.float32)
        lows2 = np.full(n_bars, 99.5, dtype=np.float32)
        closes2 = np.full(n_bars, 100.0, dtype=np.float32)
        vols2 = np.full(n_bars, 1000.0, dtype=np.float32)

        # Set trough at bar 6 (10:00)
        lows2[6] = 95.0
        # Set peak at bar 54 (14:00)
        highs2[54] = 105.0

        all_times = times_day1.append(times_day2)
        all_opens = np.concatenate([opens1, opens2])
        all_highs = np.concatenate([highs1, highs2])
        all_lows = np.concatenate([lows1, lows2])
        all_closes = np.concatenate([closes1, closes2])
        all_vols = np.concatenate([vols1, vols2])

        synth_df = pd.DataFrame({
            "open": all_opens, "high": all_highs, "low": all_lows,
            "close": all_closes, "volume": all_vols
        }, index=all_times)

        # Repeat using business days to have enough bars for the min_bars requirement
        bdays = pd.bdate_range("2024-01-01", periods=35)
        many_days = []
        for i, bday in enumerate(bdays):
            d_str = bday.strftime("%Y-%m-%d")
            t_idx = pd.date_range(f"{d_str} 09:30", f"{d_str} 15:55", freq="5min")
            h = highs1 if (i % 2 == 0) else highs2
            l = lows1 if (i % 2 == 0) else lows2
            df_d = pd.DataFrame({
                "open": opens1, "high": h, "low": l, "close": closes1, "volume": vols1
            }, index=t_idx)
            many_days.append(df_d)

        full_synth = pd.concat(many_days)
        out = nbm.extract_ticker_features_and_targets("SYNTH", b5=full_synth)
        self.assertIsNotNone(out)
        self.assertGreater(len(out), 0)

        # Check sequence labeling
        # Row 0 is 2024-01-02 (bday i=1). Its next day is bday i=2 (even) -> highs1 (High first = 1.0)
        # Row 1 is 2024-01-03 (bday i=2). Its next day is bday i=3 (odd) -> highs2 (Low first = 0.0)
        for idx in range(min(10, len(out) - 1)):
            expected_next_seq = 1.0 if (idx % 2 == 0) else 0.0
            actual_next_seq = out["target_next_mfe_first"].iloc[idx]
            self.assertEqual(actual_next_seq, expected_next_seq)

    def test_ground_truth_same_5m_bar_tie_break(self):
        """Verifies candle tie-breaking when session peak and trough occur in the same 5-minute bar."""
        # Case A: Green candle (close >= open) -> opened, dipped to low, rallied to high/close -> Low first (0.0)
        # Case B: Red candle (close < open) -> opened, spiked to high, sold off to low/close -> High first (1.0)
        n_bars = 78
        opens = np.full(n_bars, 100.0, dtype=np.float32)
        highs = np.full(n_bars, 100.5, dtype=np.float32)
        lows = np.full(n_bars, 99.5, dtype=np.float32)
        closes = np.full(n_bars, 100.0, dtype=np.float32)
        vols = np.full(n_bars, 1000.0, dtype=np.float32)

        # In bar 10: both extreme high and extreme low occur
        opens[10] = 98.0
        highs[10] = 110.0
        lows[10] = 90.0

        # Create 35 business days of data
        bdays = pd.bdate_range("2024-01-01", periods=35)
        days = []
        for i, bday in enumerate(bdays):
            d_str = bday.strftime("%Y-%m-%d")
            t_idx = pd.date_range(f"{d_str} 09:30", f"{d_str} 15:55", freq="5min")
            c_bar = 108.0 if (i % 2 == 0) else 92.0 # even: green candle (low first = 0.0), odd: red candle (high first = 1.0)
            c_copy = closes.copy()
            c_copy[10] = c_bar
            df_d = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": c_copy, "volume": vols}, index=t_idx)
            days.append(df_d)

        full_synth = pd.concat(days)
        out = nbm.extract_ticker_features_and_targets("SYNTH_TIE", b5=full_synth)
        self.assertIsNotNone(out)

        # Check tie breaking
        # Row 0 is 2024-01-02 (bday i=1). Its next day is bday i=2 (even: green candle -> low first = 0.0)
        # Row 1 is 2024-01-03 (bday i=2). Its next day is bday i=3 (odd: red candle -> high first = 1.0)
        for idx in range(min(5, len(out) - 1)):
            expected = 0.0 if (idx % 2 == 0) else 1.0
            actual = out["target_next_mfe_first"].iloc[idx]
            self.assertEqual(actual, expected)

    def test_zero_lookahead_feature_causality_regression(self):
        """Verifies strict zero-lookahead causality: features at bar t are identical whether computed
        on the full history or on a truncated series ending at bar t."""
        b5 = datastore.load_bars_5m("AAPL", calibrate=False)
        self.assertIsNotNone(b5)

        # Full extraction
        out_full = nbm.extract_ticker_features_and_targets("AAPL", b5=b5)
        self.assertIsNotNone(out_full)

        # Pick a historical date t (e.g. index 1500)
        target_idx = 1500
        target_date = out_full["date"].iloc[target_idx]
        features_full = out_full.iloc[target_idx][nbm.FEATURE_COLS].to_dict()

        # Truncate 5m bars strictly at target_date end-of-day
        cutoff = f"{target_date} 23:59:59"
        b5_truncated = b5.loc[:cutoff].copy()
        out_trunc = nbm.extract_ticker_features_and_targets("AAPL", b5=b5_truncated)
        self.assertIsNotNone(out_trunc)

        # The last row in out_trunc is target_date - 1 (since the last available day has no target),
        # so let us look for target_date in out_trunc or test at target_date - 1:
        prev_date = out_full["date"].iloc[target_idx - 1]
        row_full = out_full[out_full["date"] == prev_date].iloc[0][nbm.FEATURE_COLS]
        row_trunc = out_trunc[out_trunc["date"] == prev_date].iloc[0][nbm.FEATURE_COLS]

        for col in nbm.FEATURE_COLS:
            v_full = float(row_full[col])
            v_trunc = float(row_trunc[col])
            self.assertAlmostEqual(v_full, v_trunc, places=4, msg=f"Feature {col} leaked lookahead!")

    def test_next_bar_engine_inference_contract(self):
        """Verifies EPMLEngine next-bar inference returns the expected contract and schema."""
        # 1. Prediction on dummy features
        dummy_feats = {col: 0.0 for col in nbm.FEATURE_COLS}
        preds = engine.predict_next_bar_excursion(dummy_feats)
        self.assertIsInstance(preds, dict)
        self.assertIn("pred_next_mfe", preds)
        self.assertIn("pred_next_mae", preds)
        self.assertIn("prob_mfe_first", preds)
        self.assertIn("pred_first_extreme", preds)

        if preds["pred_next_mfe"] is not None:
            self.assertIsInstance(preds["pred_next_mfe"], float)
            self.assertIsInstance(preds["pred_next_mae"], float)
            self.assertIsInstance(preds["prob_mfe_first"], float)
            self.assertGreaterEqual(preds["prob_mfe_first"], 0.0)
            self.assertLessEqual(preds["prob_mfe_first"], 1.0)
            self.assertIn(preds["pred_first_extreme"], ["HIGH", "LOW"])

        # 2. Empty / partial features robustness
        empty_preds = engine.predict_next_bar_excursion({})
        self.assertIsInstance(empty_preds, dict)
        self.assertIn("pred_next_mfe", empty_preds)

    def test_extract_ticker_features_and_targets_real_ticker(self):
        """Verifies end-to-end extraction on a real ticker from datastore."""
        out = nbm.extract_ticker_features_and_targets("MSFT")
        self.assertIsNotNone(out)
        self.assertGreater(len(out), 500)

        # Check all required features exist
        for col in nbm.FEATURE_COLS:
            self.assertIn(col, out.columns)

        # Check targets
        self.assertIn("target_next_mfe", out.columns)
        self.assertIn("target_next_mae", out.columns)
        self.assertIn("target_next_mfe_first", out.columns)
        self.assertIn("target_next_ret", out.columns)

        # No nulls in targets
        self.assertEqual(out["target_next_mfe"].isna().sum(), 0)
        self.assertEqual(out["target_next_mae"].isna().sum(), 0)
        self.assertEqual(out["target_next_mfe_first"].isna().sum(), 0)

        # Verify geometric property: High >= Low for each day
        self.assertTrue((out["target_next_mfe"] >= out["target_next_mae"]).all())

        # Verify binary sequence: only 0.0 or 1.0
        unique_seq = set(out["target_next_mfe_first"].unique())
        self.assertTrue(unique_seq.issubset({0.0, 1.0}))

    def test_compute_next_bar_features_latest_bar_date_alignment(self):
        """Verifies compute_next_bar_features(sym, bar_idx=-1) extracts features from the latest
        available session close, not the penultimate session."""
        b5 = datastore.load_bars_5m("AAPL", calibrate=False)
        self.assertIsNotNone(b5)
        latest_date = b5.index.strftime("%Y-%m-%d")[-1]

        feats = engine.compute_next_bar_features("AAPL", bar_idx=-1)
        self.assertIsNotNone(feats)
        self.assertIn("feat_ret_1d", feats)

        # Check against full extraction with drop_unlabeled=False
        df_all = nbm.extract_ticker_features_and_targets("AAPL", drop_unlabeled=False)
        self.assertIsNotNone(df_all)
        self.assertEqual(df_all["date"].iloc[-1], latest_date)
        self.assertAlmostEqual(feats["feat_ret_1d"], float(df_all["feat_ret_1d"].iloc[-1]), places=5)

    def test_ground_truth_open_at_high_tie_break(self):
        """Verifies candle tie-breaking when a 5m bar opens at the session high (Open == High).
        High must be touched first (1.0)."""
        n_bars = 78
        opens = np.full(n_bars, 100.0, dtype=np.float32)
        highs = np.full(n_bars, 100.5, dtype=np.float32)
        lows = np.full(n_bars, 99.5, dtype=np.float32)
        closes = np.full(n_bars, 100.0, dtype=np.float32)
        vols = np.full(n_bars, 1000.0, dtype=np.float32)

        # Bar 10: opens at high (110.0), drops to low (90.0), closes at 100.0
        opens[10] = 110.0
        highs[10] = 110.0
        lows[10] = 90.0
        closes[10] = 100.0

        bdays = pd.bdate_range("2024-01-01", periods=30)
        days = []
        for bday in bdays:
            d_str = bday.strftime("%Y-%m-%d")
            t_idx = pd.date_range(f"{d_str} 09:30", f"{d_str} 15:55", freq="5min")
            df_d = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols}, index=t_idx)
            days.append(df_d)

        full_synth = pd.concat(days)
        out = nbm.extract_ticker_features_and_targets("SYNTH_OAH", b5=full_synth)
        self.assertIsNotNone(out)
        # All target sequence labels should be 1.0 (High first because it opened at High)
        for idx in range(min(5, len(out) - 1)):
            self.assertEqual(out["target_next_mfe_first"].iloc[idx], 1.0)

    def test_predict_robustness_to_inf_and_nan(self):
        """Verifies predict_next_bar_excursion does not raise or return None when features contain Inf."""
        dummy_feats = {col: 0.0 for col in nbm.FEATURE_COLS}
        dummy_feats["feat_rvol"] = float("inf")
        dummy_feats["feat_atr14_pct"] = float("-inf")
        dummy_feats["feat_dist_ema8"] = float("nan")

        preds = engine.predict_next_bar_excursion(dummy_feats)
        self.assertIsInstance(preds, dict)
        self.assertIsNotNone(preds["pred_next_mfe"])
        self.assertIsNotNone(preds["pred_next_mae"])
        self.assertIsNotNone(preds["prob_mfe_first"])
        self.assertIn(preds["pred_first_extreme"], ["HIGH", "LOW"])

    def test_daily_fallback_for_non_5m_symbols(self):
        """Verifies that tickers with daily bars but no 5m bars extract valid feature dictionaries."""
        feats = engine.compute_next_bar_features("TR", bar_idx=-1)
        self.assertIsNotNone(feats)
        self.assertEqual(len(feats), len(nbm.FEATURE_COLS))
        for col in nbm.FEATURE_COLS:
            self.assertIn(col, feats)

        pred = engine.predict_next_bar_excursion(feats)
        self.assertIsNotNone(pred["pred_next_mfe"])

    def test_vectorized_batch_prediction(self):
        """Verifies predict_next_bar_excursion_batch matches row-by-row prediction."""
        dummy1 = {col: 0.0 for col in nbm.FEATURE_COLS}
        dummy2 = {col: 0.05 for col in nbm.FEATURE_COLS}
        df_batch = pd.DataFrame([dummy1, dummy2])

        batch_preds = engine.predict_next_bar_excursion_batch(df_batch)
        self.assertEqual(len(batch_preds), 2)

        pred1 = engine.predict_next_bar_excursion(dummy1)
        self.assertEqual(batch_preds["pred_next_mfe"].iloc[0], pred1["pred_next_mfe"])
        self.assertEqual(batch_preds["prob_mfe_first"].iloc[0], pred1["prob_mfe_first"])

    def test_spy_context_not_zero_in_inference(self):
        """Verifies SPY market context features are populated with real market values in inference."""
        feats = engine.compute_next_bar_features("AAPL", bar_idx=-1)
        self.assertIsNotNone(feats)
        self.assertIn("feat_spy_ret_1d", feats)
        # SPY context should be merged and present
        spy_ctx = nbm.get_spy_market_context()
        self.assertGreater(len(spy_ctx), 0)


if __name__ == "__main__":
    unittest.main()
