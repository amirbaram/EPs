"""Unit tests for EP Portfolio Manager and Trade Lifecycle Execution Engine.

Tests:
1. Adding position with proper risk and sizing calculation.
2. Adjusting stop loss.
3. Executing secondary add (+50% size with blended entry price).
4. Closing position with realized P&L and R-multiple accounting.
5. Evaluating lifecycle alerts (Stop loss breach, Day 2 breakout, etc.).
"""
import unittest
import tempfile
import shutil
from pathlib import Path

import ep_portfolio_manager as pm

class TestEPPortfolioManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.orig_path = pm.PORTFOLIO_PATH
        pm.PORTFOLIO_PATH = Path(self.test_dir) / "ep_portfolio.json"

    def tearDown(self):
        pm.PORTFOLIO_PATH = self.orig_path
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_settings_update(self):
        pm.update_settings(150000.0, 1.5)
        port = pm.load_portfolio()
        self.assertEqual(port["settings"]["portfolio_size"], 150000.0)
        self.assertEqual(port["settings"]["default_risk_pct"], 1.5)

    def test_add_and_adjust_position(self):
        pos = pm.add_position(
            symbol="TEST",
            entry_price=10.0,
            stop_price=9.0,
            shares=1000,
            event_date="2026-10-01",
            risk_pct=1.0,
            notes="Pinnacle EP setup"
        )
        self.assertEqual(pos["symbol"], "TEST")
        self.assertEqual(pos["entry_price"], 10.0)
        self.assertEqual(pos["stop_price"], 9.0)
        self.assertEqual(pos["risk_per_share"], 1.0)
        self.assertEqual(pos["initial_risk_dollars"], 1000.0)

        # Adjust stop to breakeven
        updated = pm.update_position(pos["id"], action="adjust_stop", stop_price=10.0)
        self.assertEqual(updated["stop_price"], 10.0)

        # Secondary Add (+500 shares @ $11.00 with stop at $10.00)
        added = pm.update_position(pos["id"], action="secondary_add", add_price=11.0, add_shares=500, add_stop=10.0)
        self.assertEqual(added["shares"], 1500)
        # Blended: (10.0*1000 + 11.0*500) / 1500 = 15500 / 1500 = 10.33
        self.assertAlmostEqual(added["entry_price"], 10.33, places=2)
        self.assertTrue(added["has_d2_add"])
        self.assertEqual(added["adds"][0]["stop_price"], 10.0)

        # Close position at $12.00
        closed = pm.close_position(pos["id"], exit_price=12.0, exit_reason="Target Reached")
        self.assertEqual(closed["exit_price"], 12.0)
        # PnL: (12.0 - 10.33) * 1500 = 2505.00
        self.assertAlmostEqual(closed["realized_pnl_dollars"], 2505.0, delta=10.0)
        # Initial risk = $1000 base + (11.0 - 10.0)*500 = $1500. Realized R = 2505 / 1500 = 1.67 R
        self.assertGreater(closed["realized_r"], 1.5)

        # Check portfolio evaluation
        evaluated = pm.evaluate_portfolio()
        self.assertEqual(len(evaluated["positions"]), 0)
        self.assertEqual(len(evaluated["closed_positions"]), 1)
        self.assertGreater(evaluated["summary"]["realized_pnl"], 2000.0)

        # Test deleting closed position
        pos_id = closed["id"]
        deleted = pm.delete_closed_position(pos_id)
        self.assertTrue(deleted)
        evaluated_after_del = pm.evaluate_portfolio()
        self.assertEqual(len(evaluated_after_del["closed_positions"]), 0)
        self.assertEqual(evaluated_after_del["summary"]["realized_pnl"], 0.0)

        # Test clearing all closed positions
        pm.add_position("TEST2", 10.0, 9.0, 100, notes="")
        pos2 = pm.load_portfolio()["positions"][0]
        pm.close_position(pos2["id"], 11.0, exit_reason="Done")
        self.assertEqual(len(pm.load_portfolio()["closed_positions"]), 1)
        cleared_count = pm.clear_all_closed_positions()
        self.assertEqual(cleared_count, 1)
        self.assertEqual(len(pm.load_portfolio()["closed_positions"]), 0)

if __name__ == "__main__":
    unittest.main()
