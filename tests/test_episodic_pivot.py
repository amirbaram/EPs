"""Tests for Episodic Pivot (EP) setup detection and quality gates.

Seed: 2026-10-04 (EP research and Day 1 close-range quality gate).
"""
import unittest
import pandas as pd
import numpy as np

import config
import setups

class TestEpisodicPivot(unittest.TestCase):
    def test_close_pos_gate_rejects_fading_gap(self):
        """Bar with gap & high volume that closes in lower range (distribution fade) must be rejected."""
        # Gap +15%, ran to +30%, but faded to close at 112 (low 110, high 130 -> close_pos = 2/20 = 0.10)
        bar = pd.Series({
            "open": 115.0,
            "high": 130.0,
            "low": 110.0,
            "close": 112.0,
            "volume": 2_000_000,
            "gap_pct": 15.0,
            "chg_pct": 12.0,
            "rvol": 4.5,
            "close_pos": 0.10,
        })
        base_dvol = 5_000_000.0  # $5M prior median $vol
        sub = setups._is_ep_event(bar, base_dvol)
        self.assertIsNone(sub, "Expected fade candle with close_pos=0.10 to be rejected as EP")

    def test_close_pos_gate_accepts_strong_closing_gap(self):
        """Bar with gap & high volume that closes in upper range (strong accumulation) must pass."""
        # Gap +15%, ran to +30%, closed at 128 (low 114, high 130 -> close_pos = 14/16 = 0.875)
        bar = pd.Series({
            "open": 115.0,
            "high": 130.0,
            "low": 114.0,
            "close": 128.0,
            "volume": 2_000_000,
            "gap_pct": 15.0,
            "chg_pct": 28.0,
            "rvol": 4.5,
            "close_pos": 0.875,
        })
        base_dvol = 5_000_000.0
        sub = setups._is_ep_event(bar, base_dvol)
        self.assertEqual(sub, "classic", "Expected strong close candle with close_pos=0.875 to pass as classic EP")

    def test_detect_episodic_pivot_lifecycle(self):
        """Full synthetic frame test: strong close emits breakout, weak close does not."""
        dates = pd.date_range("2026-06-01", periods=65, freq="B")
        # 64 normal bars
        opens = [100.0] * 64
        highs = [102.0] * 64
        lows = [98.0] * 64
        closes = [100.0] * 64
        volumes = [500_000] * 64
        rvols = [1.0] * 64
        gap_pcts = [0.0] * 64
        chg_pcts = [0.0] * 64
        close_pos = [0.50] * 64

        # Bar 65 (Event day with strong close)
        opens.append(112.0)
        highs.append(125.0)
        lows.append(110.0)
        closes.append(123.0)  # close_pos = (123-110)/(125-110) = 13/15 = 0.867
        volumes.append(2_500_000)
        rvols.append(5.0)
        gap_pcts.append(12.0)
        chg_pcts.append(23.0)
        close_pos.append(0.867)

        d_strong = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes,
            "volume": volumes, "rvol": rvols, "gap_pct": gap_pcts, "chg_pct": chg_pcts,
            "close_pos": close_pos, "ret_6m": [15.0] * 65, "atr14": [2.0] * 65
        }, index=dates)

        hit = setups.detect_episodic_pivot(d_strong)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["state"], "breakout")
        self.assertEqual(hit["ep_subtype"], "classic")
        self.assertEqual(hit["level"], 110.0)
        self.assertEqual(hit["trigger"], 125.0)

        # Now test with bar 65 having a weak close (close_pos = 0.20)
        closes[-1] = 113.0  # close_pos = (113-110)/15 = 0.20
        close_pos[-1] = 0.20
        d_weak = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes,
            "volume": volumes, "rvol": rvols, "gap_pct": gap_pcts, "chg_pct": chg_pcts,
            "close_pos": close_pos, "ret_6m": [15.0] * 65, "atr14": [2.0] * 65
        }, index=dates)

        hit_weak = setups.detect_episodic_pivot(d_weak)
        self.assertIsNone(hit_weak, "Weak closing gap should not trigger EP")

    def test_ep_anti_clustering_debounce(self):
        """Second high-volume day within debounce window must not trigger a new Day 1 event."""
        dates = pd.date_range("2026-06-01", periods=66, freq="B")
        opens = [100.0] * 64
        highs = [102.0] * 64
        lows = [98.0] * 64
        closes = [100.0] * 64
        volumes = [500_000] * 64
        rvols = [1.0] * 64
        gap_pcts = [0.0] * 64
        chg_pcts = [0.0] * 64
        close_pos = [0.50] * 64

        # Bar 65 (Day 1 Catalyst ignition)
        opens.append(112.0)
        highs.append(125.0)
        lows.append(110.0)
        closes.append(123.0)
        volumes.append(2_500_000)
        rvols.append(5.0)
        gap_pcts.append(12.0)
        chg_pcts.append(23.0)
        close_pos.append(0.867)

        # Bar 66 (Day 2 High-volume continuation day: 12M shares, RVOL 4.0)
        opens.append(124.0)
        highs.append(135.0)
        lows.append(122.0)
        closes.append(133.0)
        volumes.append(12_000_000)
        rvols.append(4.0)
        gap_pcts.append(1.0)
        chg_pcts.append(8.0)
        close_pos.append(0.846)

        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes,
            "volume": volumes, "rvol": rvols, "gap_pct": gap_pcts, "chg_pct": chg_pcts,
            "close_pos": close_pos, "ret_6m": [15.0] * 66, "atr14": [2.0] * 66
        }, index=dates)

        # Bar 65 by itself triggers Day 1 breakout
        hit_day1 = setups.detect_episodic_pivot(df.iloc[:65])
        self.assertIsNotNone(hit_day1)
        self.assertEqual(hit_day1["state"], "breakout")
        self.assertNotIn("ep_age", hit_day1)  # Day 1 has no ep_age (today is event)

        # Bar 66 must NOT trigger as a brand new Day 1 event (no new ignition);
        # it is debounced and anchored back to Bar 65
        hit_day2 = setups.detect_episodic_pivot(df.iloc[:66])
        self.assertIsNotNone(hit_day2)
        # Should be anchored to the original event (ep_age = 1)
        self.assertEqual(hit_day2.get("ep_age"), 1)
        self.assertEqual(hit_day2["level"], 110.0)  # Anchored to Day 1 low
        self.assertEqual(hit_day2["state"], "breakout")  # Broke above Day 1 high with volume

    def test_ep_state_riding_momentum(self):
        """Holding in upper range without breaking out emits state riding."""
        dates = pd.date_range("2026-06-01", periods=66, freq="B")
        opens = [100.0] * 64 + [112.0, 123.0]
        highs = [102.0] * 64 + [125.0, 124.5]  # inside Day 1 high (125)
        lows = [98.0] * 64 + [110.0, 121.0]   # holding well above Day 1 low
        closes = [100.0] * 64 + [123.0, 123.5] # holding above Day 1 close
        volumes = [500_000] * 64 + [2_500_000, 600_000]
        rvols = [1.0] * 64 + [5.0, 1.2]
        gap_pcts = [0.0] * 64 + [12.0, 0.0]
        chg_pcts = [0.0] * 64 + [23.0, 0.4]
        close_pos = [0.50] * 64 + [0.867, 0.71]

        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes,
            "volume": volumes, "rvol": rvols, "gap_pct": gap_pcts, "chg_pct": chg_pcts,
            "close_pos": close_pos, "ret_6m": [15.0] * 66, "atr14": [2.0] * 66
        }, index=dates)

        hit = setups.detect_episodic_pivot(df)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["state"], "riding")
        self.assertEqual(hit["ep_age"], 1)

    def test_ep_state_48h_retracement_rejection(self):
        """Surrendering >50% of Day 1 body on Day 2 must reject the episode (failed absorption)."""
        dates = pd.date_range("2026-06-01", periods=66, freq="B")
        opens = [100.0] * 64 + [112.0, 120.0]
        highs = [102.0] * 64 + [125.0, 122.0]
        lows = [98.0] * 64 + [110.0, 113.0]
        # Day 1: open 112, close 123 (body = 11 pts). 50% retrace = 123 - 5.5 = 117.5.
        # Day 2: fades hard and closes at 114 (< 117.5 -> violated 48h absorption).
        closes = [100.0] * 64 + [123.0, 114.0]
        volumes = [500_000] * 64 + [2_500_000, 1_000_000]
        rvols = [1.0] * 64 + [5.0, 2.0]
        gap_pcts = [0.0] * 64 + [12.0, -2.4]
        chg_pcts = [0.0] * 64 + [23.0, -7.3]
        close_pos = [0.50] * 64 + [0.867, 0.11]

        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes,
            "volume": volumes, "rvol": rvols, "gap_pct": gap_pcts, "chg_pct": chg_pcts,
            "close_pos": close_pos, "ret_6m": [15.0] * 66, "atr14": [2.0] * 66
        }, index=dates)

        hit_day2 = setups.detect_episodic_pivot(df)
        self.assertIsNone(hit_day2, "Day 2 closing below 50% of Day 1 body must fail absorption and be dropped")

    def test_ep_state_building_digestion(self):
        """Episode past Day 5 consolidating in an orderly base emits state building."""
        dates = pd.date_range("2026-06-01", periods=72, freq="B")
        opens = [100.0] * 64 + [112.0] + [122.0] * 7
        highs = [102.0] * 64 + [125.0] + [124.0] * 7
        lows = [98.0] * 64 + [110.0] + [120.0] * 7
        closes = [100.0] * 64 + [123.0] + [122.0] * 7  # holding cleanly between 120 and 124
        volumes = [500_000] * 64 + [2_500_000] + [400_000] * 7
        rvols = [1.0] * 64 + [5.0] + [0.8] * 7
        gap_pcts = [0.0] * 64 + [12.0] + [0.0] * 7
        chg_pcts = [0.0] * 64 + [23.0] + [0.0] * 7
        close_pos = [0.50] * 64 + [0.867] + [0.50] * 7

        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes,
            "volume": volumes, "rvol": rvols, "gap_pct": gap_pcts, "chg_pct": chg_pcts,
            "close_pos": close_pos, "ret_6m": [15.0] * 72, "atr14": [2.0] * 72
        }, index=dates)

        hit = setups.detect_episodic_pivot(df)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["state"], "building")
        self.assertEqual(hit["ep_age"], 7)

    def test_ep_quality_catalyst_scoring(self):
        """EP quality score incorporates catalyst sub-score (close_pos and rvol)."""
        import quality
        d_dummy = pd.DataFrame({"close": [100.0, 110.0]})
        row_elite = {
            "setup": "episodic_pivot", "adr_pct": 5.0, "prior_gain_pct": 50.0, "prior_clarity": 0.8,
            "close_pos": 0.90, "rvol": 8.0
        }
        res_elite = quality.score_hit(d_dummy, row_elite)
        self.assertIn("q_catalyst", res_elite)
        self.assertEqual(res_elite["q_catalyst"], 1.0)  # Max catalyst quality

        row_weak = {
            "setup": "episodic_pivot", "adr_pct": 5.0, "prior_gain_pct": 50.0, "prior_clarity": 0.8,
            "close_pos": 0.65, "rvol": 3.0
        }
        res_weak = quality.score_hit(d_dummy, row_weak)
        self.assertIn("q_catalyst", res_weak)
        self.assertEqual(res_weak["q_catalyst"], 0.0)  # Min catalyst quality
        self.assertGreater(res_elite["quality"], res_weak["quality"])

if __name__ == "__main__":
    unittest.main()

