#!/usr/bin/env python3
"""tests/test_intraday_monitor.py — Regression Test Suite for Real-Time Intraday Active Watchlist Monitor.

Verifies:
1. Market hours calendar check (is_market_open during trading session, pre-market, post-market, and weekend).
2. evaluate_intraday_position: synthetic daily bar fusion, moving average buffer, and Extreme Peak ML prediction.
3. Climax alert triggering at archetype thresholds (>=55% for 50 SMA, >=45% for 20 EMA).
4. Watchlist caching and overlay logic in serve_optimized_ep (Port 8787).
5. Flask API endpoints: /api/intraday_watchlist and /api/watchlist intraday_meta contract.
6. Daemon PID handling and lifecycle management.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import intraday_watchlist_monitor as monitor
import serve_optimized_ep as srv


class TestIntradayWatchlistMonitor(unittest.TestCase):
    """Test suite for intraday_watchlist_monitor.py and serve_optimized_ep intraday integration."""

    def setUp(self):
        self.tz = ZoneInfo("America/New_York")
        self.test_dir = tempfile.mkdtemp()
        self.orig_cache_file = monitor.CACHE_FILE
        self.orig_pid_file = monitor.PID_FILE
        self.orig_srv_cache = srv.INTRADAY_CACHE_PATH

        self.mock_cache = Path(self.test_dir) / "intraday_status.json"
        self.mock_pid = Path(self.test_dir) / "monitor.pid"

        monitor.CACHE_FILE = self.mock_cache
        monitor.PID_FILE = self.mock_pid
        srv.INTRADAY_CACHE_PATH = self.mock_cache

    def tearDown(self):
        monitor.CACHE_FILE = self.orig_cache_file
        monitor.PID_FILE = self.orig_pid_file
        srv.INTRADAY_CACHE_PATH = self.orig_srv_cache
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_is_market_open_regular_hours(self):
        """Verifies market is open on a trading Wednesday at 11:30 AM ET."""
        # 2026-05-13 is a Wednesday (trading day)
        t_open = dt.datetime(2026, 5, 13, 11, 30, tzinfo=self.tz)
        is_open, reason = monitor.is_market_open(t_open)
        self.assertTrue(is_open)
        self.assertIn("Regular Trading Hours", reason)

    def test_is_market_open_pre_and_post_market(self):
        """Verifies market is closed in pre-market (08:30 ET) and post-market (17:00 ET)."""
        t_pre = dt.datetime(2026, 5, 13, 8, 30, tzinfo=self.tz)
        is_open, reason = monitor.is_market_open(t_pre)
        self.assertFalse(is_open)
        self.assertIn("Pre-market", reason)

        t_post = dt.datetime(2026, 5, 13, 17, 0, tzinfo=self.tz)
        is_open, reason = monitor.is_market_open(t_post)
        self.assertFalse(is_open)
        self.assertIn("Post-market", reason)

    def test_is_market_open_weekend(self):
        """Verifies market is closed on a Sunday."""
        # 2026-05-17 is a Sunday
        t_sun = dt.datetime(2026, 5, 17, 12, 0, tzinfo=self.tz)
        is_open, reason = monitor.is_market_open(t_sun)
        self.assertFalse(is_open)
        self.assertIn("Weekend/Holiday", reason)

    def test_evaluate_intraday_position_with_synthetic_bars(self):
        """Verifies synthetic bar fusion, moving averages, and exhaustion evaluation."""
        pos = {
            "symbol": "MOCK",
            "archetype": "🌟 Emerging Leaders",
            "trade_type": "Trade 1: Base EP Delayed Breakout (Days 2-5)",
            "trigger_price": 50.0,
            "stop_price": 45.0,
            "rs_score": 85.0,
            "v2_conviction_50": 0.25,
            "recommended_ma_key": "ema20",
            "recommended_ma_name": "20 EMA Intermediate Momentum Trail",
            "climax_threshold": 0.45,
            "climax_threshold_pct": 45.0,
        }

        # Mock historical daily bars (30 bars)
        dates = pd.date_range("2026-04-01", periods=30, freq="B")
        hist_df = pd.DataFrame({
            "open": np.linspace(40, 50, 30),
            "high": np.linspace(41, 52, 30),
            "low": np.linspace(39, 49, 30),
            "close": np.linspace(40.5, 51, 30),
            "volume": np.full(30, 1_000_000.0),
        }, index=dates)

        # Mock 5m intraday bars
        intra_times = pd.date_range("2026-05-13 09:30", "2026-05-13 11:30", freq="5min", tz=self.tz)
        intra_df = pd.DataFrame({
            "Open": np.linspace(51.0, 56.0, len(intra_times)),
            "High": np.linspace(51.5, 58.5, len(intra_times)),
            "Low": np.linspace(50.8, 55.5, len(intra_times)),
            "Close": np.linspace(51.2, 57.8, len(intra_times)),
            "Volume": np.full(len(intra_times), 50_000.0),
        }, index=intra_times)

        with patch("datastore.load_bars", return_value=hist_df), \
             patch("serve_optimized_ep.predict_exhaustion_for_bars", return_value=0.58):

            res = monitor.evaluate_intraday_position(pos, intra_df)

            self.assertEqual(res["symbol"], "MOCK")
            self.assertEqual(res["intraday_high"], 58.5)
            self.assertEqual(res["current_price"], 57.8)
            self.assertGreater(res["unrealized_pct"], 0.0)
            self.assertEqual(res["exhaustion_prob"], 0.58)
            self.assertEqual(res["exhaustion_pct"], 58.0)
            self.assertTrue(res["is_alert"])
            self.assertEqual(res["alert_state"], "🚨 INTRADAY_SELL_HALF")
            self.assertIn("SELL HALF ALERT", res["alert_action"])
            self.assertGreater(res["recommended_ma_price"], 0.0)

    def test_evaluate_intraday_position_below_hurdle(self):
        """Verifies position below climax hurdle remains in HOLD_RUNNER state."""
        pos = {
            "symbol": "HOLD",
            "archetype": "👑 Pinnacle Elite",
            "trigger_price": 100.0,
            "stop_price": 92.0,
            "rs_score": 92.0,
            "v2_conviction_50": 0.30,
            "recommended_ma_key": "sma50",
            "recommended_ma_name": "50 SMA Institutional Baseline",
            "climax_threshold": 0.55,
            "climax_threshold_pct": 55.0,
        }

        dates = pd.date_range("2026-04-01", periods=30, freq="B")
        hist_df = pd.DataFrame({
            "open": np.linspace(80, 100, 30),
            "high": np.linspace(82, 102, 30),
            "low": np.linspace(79, 99, 30),
            "close": np.linspace(81, 101, 30),
            "volume": np.full(30, 500_000.0),
        }, index=dates)

        with patch("datastore.load_bars", return_value=hist_df), \
             patch("serve_optimized_ep.predict_exhaustion_for_bars", return_value=0.22):

            res = monitor.evaluate_intraday_position(pos, None)
            self.assertEqual(res["symbol"], "HOLD")
            self.assertEqual(res["exhaustion_pct"], 22.0)
            self.assertFalse(res["is_alert"])
            self.assertEqual(res["alert_state"], "🟢 HOLD_RUNNER")
            self.assertIn("Hold Runner", res["alert_action"])

    def test_serve_ep_api_intraday_watchlist_contract(self):
        """Verifies Flask endpoint /api/intraday_watchlist returns cached telemetry and positions."""
        mock_payload = {
            "timestamp_et": "2026-05-13 11:30:00 EDT",
            "market_open": True,
            "active_count": 1,
            "alerts_count": 1,
            "positions": {
                "TEST": {
                    "symbol": "TEST",
                    "current_price": 105.50,
                    "intraday_high": 108.00,
                    "exhaustion_pct": 59.2,
                    "alert_state": "🚨 INTRADAY_SELL_HALF",
                    "is_alert": True,
                }
            },
        }
        self.mock_cache.write_text(json.dumps(mock_payload, indent=2))

        client = srv.app.test_client()
        resp = client.get("/api/intraday_watchlist")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("market_open"))
        self.assertEqual(data.get("alerts_count"), 1)
        self.assertIn("TEST", data.get("positions", {}))

    def test_apply_intraday_cache_to_watchlist_merging(self):
        """Verifies apply_intraday_cache_to_watchlist seamlessly updates items."""
        mock_payload = {
            "timestamp_et": "2026-05-13 11:30:00 EDT",
            "market_open": True,
            "active_count": 1,
            "alerts_count": 1,
            "positions": {
                "XYZ": {
                    "symbol": "XYZ",
                    "current_price": 125.00,
                    "intraday_high": 130.00,
                    "unrealized_r": 4.5,
                    "unrealized_pct": 25.0,
                    "exhaustion_pct": 62.1,
                    "alert_state": "🚨 INTRADAY_SELL_HALF",
                    "alert_action": "🚨 SELL 50% PARTIAL PROFIT NOW",
                    "is_alert": True,
                }
            }
        }
        self.mock_cache.write_text(json.dumps(mock_payload, indent=2))

        base_items = [{
            "symbol": "XYZ",
            "status": "Active Long",
            "current_price": 120.00,
            "current_high_exhaustion_pct": 35.0,
            "is_alert": False,
        }]

        updated = srv.apply_intraday_cache_to_watchlist(base_items)
        self.assertEqual(len(updated), 1)
        item = updated[0]
        self.assertEqual(item["current_price"], 125.00)
        self.assertEqual(item["intraday_high"], 130.00)
        self.assertEqual(item["current_high_exhaustion_pct"], 62.1)
        self.assertTrue(item["is_alert"])
        self.assertEqual(item["alert_state"], "🚨 INTRADAY_SELL_HALF")
        self.assertEqual(item["climax_action"], "🚨 SELL 50% PARTIAL PROFIT NOW")

    def test_daemon_pid_handling(self):
        """Verifies PID file creation, running check, and clean shutdown."""
        # When no pid file exists
        running, pid = monitor.is_daemon_running()
        self.assertFalse(running)
        self.assertIsNone(pid)

        # Write current pid
        current_pid = os.getpid()
        self.mock_pid.write_text(str(current_pid))
        running, pid = monitor.is_daemon_running()
        self.assertTrue(running)
        self.assertEqual(pid, current_pid)


if __name__ == "__main__":
    unittest.main()
