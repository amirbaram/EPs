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

        # Verify sizing and institutional fields in active/coiling items
        first_act = [x for x in data["items"] if x["status"] == "Active Long"][0]
        self.assertIn("suggested_size_mult", first_act)
        self.assertIn("suggested_size_label", first_act)
        self.assertIn("suggested_size_desc", first_act)
        self.assertIn("current_high_exhaustion_pct", first_act)
        self.assertIn("climax_threshold_pct", first_act)
        self.assertIn("climax_sell_half_rule", first_act)
        self.assertIn("climax_action", first_act)
        self.assertIn("trailing_exit_rule", first_act)
        self.assertIn("position_management_plan", first_act)
        self.assertIn("is_tier1", first_act)
        self.assertIn("is_institutional_core", first_act)
        self.assertIn("tier", first_act)
        self.assertIn("ml_stop_price", first_act)
        self.assertIn("ml_stop_pct", first_act)
        self.assertGreater(first_act["ml_stop_price"], 0.0)
        self.assertGreater(first_act["ml_stop_pct"], 0.0)

        first_coil = [x for x in data["items"] if x["status"] == "Coiling Setup"][0]
        self.assertIn("ml_stop_price", first_coil)
        self.assertIn("ml_stop_pct", first_coil)
        self.assertGreater(first_coil["ml_stop_price"], 0.0)
        self.assertGreater(first_coil["ml_stop_pct"], 0.0)

        # Test portfolio tier filtering: core and tier1
        self.assertIn("total_core", data)
        self.assertIn("total_tier1", data)
        self.assertGreater(data["total_core"], 0)
        self.assertGreater(data["total_tier1"], 0)

        resp_core = self.client.get("/api/watchlist?tier=core")
        self.assertEqual(resp_core.status_code, 200)
        self.assertEqual(len(resp_core.json["items"]), data["total_core"])
        for it in resp_core.json["items"]:
            self.assertTrue(it["is_institutional_core"])

        resp_t1 = self.client.get("/api/watchlist?tier=tier1")
        self.assertEqual(resp_t1.status_code, 200)
        self.assertEqual(len(resp_t1.json["items"]), data["total_tier1"])
        for it in resp_t1.json["items"]:
            self.assertTrue(it["is_tier1"])

    def test_api_monitor_settings_contract(self):
        """Verifies GET and POST for /api/monitor_settings."""
        # GET
        resp = self.client.get("/api/monitor_settings")
        self.assertEqual(resp.status_code, 200)
        data = resp.json
        self.assertIn("poll_interval_sec", data)
        self.assertIn("recommended_interval_sec", data)
        self.assertIn("rate_limit_guidance", data)
        self.assertEqual(data["recommended_interval_sec"], 300)

        # POST valid
        resp_post = self.client.post("/api/monitor_settings", json={"poll_interval_sec": 600})
        self.assertEqual(resp_post.status_code, 200)
        self.assertEqual(resp_post.json["poll_interval_sec"], 600)

        # Restore default 300
        self.client.post("/api/monitor_settings", json={"poll_interval_sec": 300})

    def test_chart_indicators_history_and_larsson_contract(self):
        """Verifies /api/chart returns extended indicators, history depth, and Larsson spans."""
        sample_sym = str(self.df["symbol"].iloc[0])
        ep_date = str(self.df["ep_date"].iloc[0])[:10]
        entry_date = str(self.df["entry_date"].iloc[0])[:10]
        exit_date = str(self.df["exit_date"].iloc[0])[:10]

        # Test history=6m and larsson_fastest=32
        url = f"/api/chart?symbol={sample_sym}&ep_date={ep_date}&entry_date={entry_date}&exit_date={exit_date}&history=6m&larsson_fastest=32"
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        data = resp.json
        self.assertIn("vol_ma20", data)
        self.assertIn("ema8", data)
        self.assertIn("ema21", data)
        self.assertIn("larsson_fast", data)
        self.assertIn("larsson_slow", data)
        self.assertIn("larsson_spans", data)
        self.assertEqual(data["larsson_spans"], [32, 35, 50, 58])
        self.assertEqual(data["history_mode"], "6m")

        # Test history=1y and custom larsson_fastest=25
        url_1y = f"/api/chart?symbol={sample_sym}&ep_date={ep_date}&entry_date={entry_date}&exit_date={exit_date}&history=1y&larsson_fastest=25"
        resp_1y = self.client.get(url_1y)
        self.assertEqual(resp_1y.status_code, 200)
        data_1y = resp_1y.json
        self.assertEqual(data_1y["larsson_spans"], [25, 27, 39, 45])
        self.assertGreater(len(data_1y["candles"]), len(data["candles"]))

    def test_index_ui_elements_extended(self):
        """Verifies all new HTML UI elements for settings, indicators, and dossier cards exist."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8")
        self.assertIn("btn_monitor_settings", html)
        self.assertIn("monitor_settings_panel", html)
        self.assertIn("sel_monitor_interval", html)
        self.assertIn("btn_chart_settings", html)
        self.assertIn("btn_chart_fullscreen", html)
        self.assertIn("sel_chart_history", html)
        self.assertIn("chart_settings_panel", html)
        self.assertIn("card_catalyst", html)
        self.assertIn("card_rs_posture", html)
        self.assertIn("card_sec_theme", html)
        self.assertIn("d_size_desc", html)
        self.assertIn("btn_nav_prev", html)
        self.assertIn("btn_nav_next", html)
        self.assertIn("sel_nav_stock", html)
        self.assertIn("lbl_nav_counter", html)
        self.assertIn("fs_keyboard_hint", html)
        self.assertIn("btn_wl_tier_all", html)
        self.assertIn("btn_wl_tier_core", html)
        self.assertIn("btn_wl_tier_1", html)
        self.assertIn("c_badge_tier", html)
        self.assertIn("d_tier", html)
        self.assertIn("row_ml_stop", html)
        self.assertIn("d_ml_stop", html)
        self.assertIn("c_ml_stop_legend", html)


if __name__ == "__main__":
    unittest.main()

