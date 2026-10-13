#!/usr/bin/env python3
"""tests/test_optimized_ep_viewer.py — Regression & API Contract Suite for Port 8787 App."""

import unittest
from serve_optimized_ep import app, get_dataset


class TestOptimizedEPViewer(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.df = get_dataset()

    def test_dataset_loaded_and_valid(self):
        """Verifies dataset has trades and necessary columns."""
        self.assertGreater(len(self.df), 0)
        self.assertIn("is_tier1", self.df.columns)
        self.assertIn("v2_size_mult", self.df.columns)
        self.assertIn("adaptive_r", self.df.columns)
        self.assertIn("adaptive_v2_sized_r", self.df.columns)

    def test_index_route(self):
        """Verifies HTML template loads and contains essential UI elements."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8")
        self.assertIn("INSTITUTIONAL OPTIMIZED EP EXPLORER", html)
        self.assertIn("PORT 8787", html)
        self.assertIn("btn_tier_1", html)
        self.assertIn("sel_exit_mode", html)
        self.assertIn("sel_sizing", html)

    def test_api_kpis_contract(self):
        """Verifies /api/kpis returns valid numerical stats."""
        tier1_n = int((self.df["is_tier1"] == True).sum())
        total_n = len(self.df)

        # Tier 1 Sniper
        resp = self.client.get("/api/kpis?tier=tier1&exit_mode=adaptive&sizing=v2")
        self.assertEqual(resp.status_code, 200)
        data = resp.json
        self.assertEqual(data["trades_count"], tier1_n)
        self.assertGreater(data["win_rate_pct"], 40.0)
        self.assertGreater(data["ev_r"], 0.20)
        self.assertGreater(data["profit_factor"], 1.4)

        # Core Suite
        resp2 = self.client.get("/api/kpis?tier=all&exit_mode=ema20&sizing=v2")
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.json
        self.assertEqual(data2["trades_count"], total_n)
        self.assertGreater(data2["win_rate_pct"], 40.0)
        self.assertGreater(data2["total_r"], 50.0)

        # Test min_rs filter
        resp_rs = self.client.get("/api/kpis?tier=all&min_rs=90.0")
        self.assertEqual(resp_rs.status_code, 200)
        self.assertLessEqual(resp_rs.json["trades_count"], total_n)

    def test_api_trades_contract(self):
        """Verifies /api/trades returns trade records with correct structure."""
        tier1_n = int((self.df["is_tier1"] == True).sum())
        resp = self.client.get("/api/trades?tier=tier1&exit_mode=adaptive&sizing=v2")
        self.assertEqual(resp.status_code, 200)
        data = resp.json
        self.assertIn("trades", data)
        self.assertEqual(data["count"], tier1_n)
        first = data["trades"][0]
        self.assertIn("symbol", first)
        self.assertIn("entry_price", first)
        self.assertIn("stop_price", first)
        self.assertIn("r_multiple", first)
        self.assertIn("v2_size_mult", first)
        self.assertIn("rs_score_breakout", first)
        self.assertIn("v2_breakout", first)

    def test_api_chart_contract(self):
        """Verifies /api/chart returns candlesticks and indicator series."""
        sample_sym = str(self.df["symbol"].iloc[0])
        ep_date = str(self.df["ep_date"].iloc[0])[:10]
        entry_date = str(self.df["entry_date"].iloc[0])[:10]
        exit_date = str(self.df["exit_date"].iloc[0])[:10]

        url = f"/api/chart?symbol={sample_sym}&ep_date={ep_date}&entry_date={entry_date}&exit_date={exit_date}&exit_mode=adaptive"
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        data = resp.json
        self.assertIn("candles", data)
        self.assertIn("volume", data)
        self.assertIn("ema10", data)
        self.assertIn("ema20", data)
        self.assertIn("sma50", data)
        self.assertGreater(len(data["candles"]), 10)

    def test_progressive_exposure_and_date_sorting(self):
        """Verifies progressive exposure and date sorting parameters."""
        # Progressive exposure
        resp = self.client.get("/api/kpis?progressive=confluence&exit_mode=sma50")
        self.assertEqual(resp.status_code, 200)
        data = resp.json
        self.assertGreater(data["total_r"], 140.0)

        # Test Quarter-Size Defense Ladder (0.25x on 2 losses)
        resp_qtr = self.client.get("/api/kpis?progressive=qtr_defense&exit_mode=ema20")
        self.assertEqual(resp_qtr.status_code, 200)
        qtr_data = resp_qtr.json
        self.assertGreater(qtr_data["profit_factor"], 1.70)
        self.assertGreater(qtr_data["max_drawdown_r"], -25.0)  # Slashed from -48.2R down to -21.1R

        # Test Confluence + 0.25x Defense
        resp_qtr_conf = self.client.get("/api/kpis?progressive=qtr_confluence&exit_mode=ema20")
        self.assertEqual(resp_qtr_conf.status_code, 200)
        conf_data = resp_qtr_conf.json
        self.assertGreater(conf_data["profit_factor"], 1.80)
        self.assertGreater(conf_data["total_r"], 110.0)
        self.assertGreater(conf_data["max_drawdown_r"], -25.0)

        # Date sorting
        resp_date = self.client.get("/api/trades?sort_by=entry_date&sort_dir=asc")
        self.assertEqual(resp_date.status_code, 200)
        trades = resp_date.json["trades"]
        self.assertEqual(trades[0]["entry_date"], "2016-09-30")

        # Exit mode switching updates exit_date and exit_price
        resp_sma50 = self.client.get("/api/trades?exit_mode=sma50")
        self.assertEqual(resp_sma50.status_code, 200)
        t_sma50 = [t for t in resp_sma50.json["trades"] if t["symbol"] == "SHOP"][0]
        self.assertEqual(t_sma50["exit_date"], "2017-06-27")
        self.assertEqual(t_sma50["exit_price"], 8.7)

    def test_api_reload_dataset(self):
        """Verifies /api/reload_dataset reloads dataset and returns status."""
        tier1_n = int((self.df["is_tier1"] == True).sum())
        total_n = len(self.df)
        resp = self.client.post("/api/reload_dataset")
        self.assertEqual(resp.status_code, 200)
        data = resp.json
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["trades_count"], total_n)
        self.assertEqual(data["tier1_count"], tier1_n)
        self.assertIn("watchlist_count", data)

    def test_api_watchlist_contract(self):
        """Verifies /api/watchlist returns active in-flight trades and coiling setups."""
        resp = self.client.get("/api/watchlist")
        self.assertEqual(resp.status_code, 200)
        data = resp.json
        self.assertIn("items", data)
        self.assertGreater(data["count"], 0)
        self.assertGreater(data["total_active"], 0)
        self.assertGreater(data["total_coiling"], 0)

        # Check item structure
        first = data["items"][0]
        self.assertIn("symbol", first)
        self.assertIn("status", first)
        self.assertIn(first["status"], ["Active Long", "Coiling Setup"])
        self.assertIn("trigger_price", first)
        self.assertIn("current_price", first)
        self.assertIn("stop_price", first)
        self.assertIn("rs_score", first)
        self.assertIn("stage2", first)

        # Test status filter: active
        resp_act = self.client.get("/api/watchlist?status=active")
        self.assertEqual(resp_act.status_code, 200)
        for it in resp_act.json["items"]:
            self.assertEqual(it["status"], "Active Long")

        # Test status filter: coiling
        resp_coil = self.client.get("/api/watchlist?status=coiling")
        self.assertEqual(resp_coil.status_code, 200)
        for it in resp_coil.json["items"]:
            self.assertEqual(it["status"], "Coiling Setup")


if __name__ == "__main__":
    unittest.main()
