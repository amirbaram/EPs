#!/usr/bin/env python3
"""tests/test_daily_update.py — Unit & Regression Tests for Daily Universe Retrieval Orchestrator."""

import datetime as dt
import os
import unittest
from unittest.mock import patch, MagicMock
from zoneinfo import ZoneInfo

import daily_update
import trading_calendar


class TestDailyUpdateOrchestrator(unittest.TestCase):
    def test_get_dataset_timestamps(self):
        """Verifies dataset timestamps inspection returns status for all 4 apps and datastore."""
        ts = daily_update.get_dataset_timestamps()
        self.assertIn("spy_bars_latest", ts)
        self.assertIn("rs_db_latest", ts)
        self.assertIn("ep_study_latest", ts)
        self.assertIn("qm_setups_latest", ts)
        self.assertIn("opt_strategy_latest", ts)
        self.assertIn("rs_sessions_count", ts)
        self.assertIn("ep_events_count", ts)
        self.assertIn("qm_setups_count", ts)
        self.assertIn("opt_strategy_count", ts)

        # Datastore and parquets should have non-empty dates
        self.assertIsNotNone(ts["spy_bars_latest"])
        self.assertGreater(ts["rs_sessions_count"], 0)
        self.assertGreater(ts["ep_events_count"], 0)
        self.assertGreater(ts["qm_setups_count"], 0)
        self.assertGreater(ts["opt_strategy_count"], 0)

    def test_get_next_run_time_weekday_schedule(self):
        """Verifies calculation of next run time on trading days and over weekends."""
        tz = ZoneInfo("America/New_York")

        # 1. Tuesday morning at 10:00 ET -> should schedule Tuesday 16:45 ET
        tue_morning = dt.datetime(2026, 10, 13, 10, 0, tzinfo=tz)
        next_run = daily_update.get_next_run_time("16:45", now_et=tue_morning)
        self.assertEqual(next_run.year, 2026)
        self.assertEqual(next_run.month, 10)
        self.assertEqual(next_run.day, 13)
        self.assertEqual(next_run.hour, 16)
        self.assertEqual(next_run.minute, 45)

        # 2. Tuesday evening at 17:30 ET -> should schedule Wednesday 16:45 ET
        tue_evening = dt.datetime(2026, 10, 13, 17, 30, tzinfo=tz)
        next_run = daily_update.get_next_run_time("16:45", now_et=tue_evening)
        self.assertEqual(next_run.day, 14)
        self.assertEqual(next_run.hour, 16)
        self.assertEqual(next_run.minute, 45)

        # 3. Friday evening at 18:00 ET -> should schedule Monday 16:45 ET (skip Sat/Sun)
        fri_evening = dt.datetime(2026, 10, 16, 18, 0, tzinfo=tz)
        next_run = daily_update.get_next_run_time("16:45", now_et=fri_evening)
        self.assertEqual(next_run.day, 19)  # Oct 19 is Monday
        self.assertEqual(next_run.weekday(), 0)  # Monday

    def test_reload_server_offline_handling(self):
        """Verifies reload_server gracefully handles offline servers without exceptions."""
        res = daily_update.reload_server(59999, "Dummy Offline Server")
        self.assertEqual(res["status"], "offline")
        self.assertEqual(res["port"], 59999)

    def test_daemon_lifecycle_checks(self):
        """Verifies PID file detection and cleanup."""
        # Non-existent or fake PID
        fake_pid = 999999
        daily_update.PID_FILE.write_text(str(fake_pid))
        try:
            running, pid = daily_update.is_daemon_running()
            # Since fake_pid is not a running process, is_daemon_running should detect stale PID and return False
            self.assertFalse(running)
            self.assertFalse(daily_update.PID_FILE.exists())
        finally:
            daily_update.PID_FILE.unlink(missing_ok=True)

    @patch("daily_update.datastore.update_daily")
    @patch("daily_update.update_rs_database.update_database")
    @patch("daily_update.ep_updater.update_pipeline")
    @patch("daily_update.update_qm_database.run_update")
    @patch("daily_update.sim_optimized_ep_strategy.run_optimization")
    @patch("daily_update.reload_server")
    def test_run_pipeline_orchestration(
        self,
        mock_reload,
        mock_opt,
        mock_qm,
        mock_ep,
        mock_rs,
        mock_bars,
    ):
        """Verifies proper execution order and summary payload of run_pipeline."""
        mock_rs.return_value = True
        mock_qm.return_value = True
        mock_reload.return_value = {"status": "reloaded"}

        res = daily_update.run_pipeline(full_bars=False, force=False, skip_download=False)

        self.assertTrue(res["success"])
        self.assertIn("bars_download", res["steps"])
        self.assertIn("rs_database", res["steps"])
        self.assertIn("ep_study", res["steps"])
        self.assertIn("qm_setups", res["steps"])
        self.assertIn("optimized_ep_strategy", res["steps"])
        self.assertEqual(len(res["server_reloads"]), len(daily_update.TARGET_SERVERS))
        self.assertIn("8787", res["server_reloads"])

        # Verify calls occurred in correct sequence
        mock_bars.assert_called_once_with(full=False)
        mock_rs.assert_called_once_with(force=False)
        mock_ep.assert_called_once_with(skip_download=True, full_download=False, dry_run=False)
        mock_qm.assert_called_once_with(force=False)
        mock_opt.assert_called_once()
        self.assertEqual(mock_reload.call_count, len(daily_update.TARGET_SERVERS))


if __name__ == "__main__":
    unittest.main()
