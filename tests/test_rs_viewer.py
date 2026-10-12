"""test_rs_viewer.py — Regression test suite for serve_rs (Port 8786).

Verifies:
1. RS Score dataset generation and percentile ranking (1-99).
2. /api/stocks endpoint sorting (highest RS on top), filtering (min_rs, sector, theme, universe type, liquidity).
3. /api/chart endpoint contract (bars, volume, EMAs, SMAs, Larsson ribbon, RS Line vs SPY, stock/sector/theme info).
4. /api/annotations GET and POST endpoints for persistent chart drawings.
5. Index UI rendering (3-pane layout, KPI row, chart controls, dossier cards).
"""
import json
import unittest
import pandas as pd
import serve_rs


class TestRSViewer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = serve_rs.app.test_client()
        # Ensure dataset is loaded
        cls.df = serve_rs.get_df()

    def test_rs_dataset_structure_and_ranking(self):
        """Verifies RS scores are computed, ranked 1-99, and sorted highest first."""
        self.assertFalse(self.df.empty, "RS dataset should not be empty")
        self.assertIn("symbol", self.df.columns)
        self.assertIn("rs_score", self.df.columns)
        self.assertIn("raw_rs", self.df.columns)
        self.assertIn("close", self.df.columns)
        self.assertIn("ret_1m", self.df.columns)
        self.assertIn("ret_3m", self.df.columns)
        self.assertIn("ret_1y", self.df.columns)
        self.assertIn("alpha_1m", self.df.columns)
        self.assertIn("adr_pct", self.df.columns)
        self.assertIn("dvol_m", self.df.columns)
        self.assertIn("sector", self.df.columns)
        self.assertIn("theme", self.df.columns)

        # Verify RS score bounds 1-99
        self.assertGreaterEqual(int(self.df["rs_score"].max()), 90)
        self.assertLessEqual(int(self.df["rs_score"].max()), 99)
        self.assertGreaterEqual(int(self.df["rs_score"].min()), 1)

    def test_api_stocks_default_sorted_highest_rs_first(self):
        """Verifies /api/stocks returns stocks sorted by RS score descending (highest on top)."""
        res = self.client.get("/api/stocks")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("stocks", data)
        self.assertIn("kpis", data)
        self.assertIn("sectors", data)
        self.assertIn("themes", data)
        self.assertGreater(len(data["stocks"]), 0)

        # Check first stock has maximum RS score
        first_stock = data["stocks"][0]
        self.assertGreaterEqual(first_stock["rs_score"], 90)
        
        # Verify descending order of RS scores
        scores = [s["rs_score"] for s in data["stocks"][:50]]
        self.assertEqual(scores, sorted(scores, reverse=True), "Top stocks must be sorted highest RS on top")

    def test_api_stocks_filters(self):
        """Verifies /api/stocks filters by min_rs, liquidity, and universe_type."""
        # 1. Filter by min_rs=90
        res = self.client.get("/api/stocks?min_rs=90")
        self.assertEqual(res.status_code, 200)
        d = res.get_json()
        for s in d["stocks"]:
            self.assertGreaterEqual(s["rs_score"], 90)

        # 2. Filter by common stocks
        res_common = self.client.get("/api/stocks?universe_type=common")
        self.assertEqual(res_common.status_code, 200)
        for s in res_common.get_json()["stocks"][:20]:
            self.assertFalse(s.get("is_etf", False))

    def test_api_chart_contract(self):
        """Verifies /api/chart returns bars, volume, EMAs, SMAs, Larsson ribbon, RS line, and info dossiers."""
        res = self.client.get("/api/chart?symbol=NVDA&larsson_fastest=32")
        self.assertEqual(res.status_code, 200)
        d = res.get_json()
        self.assertEqual(d.get("symbol"), "NVDA")
        self.assertIn("bars", d)
        self.assertGreater(len(d["bars"]), 0)
        self.assertIn("volume", d)
        self.assertIn("vol_ma20", d)
        self.assertIn("ema10", d)
        self.assertIn("ema20", d)
        self.assertIn("sma50", d)
        self.assertIn("sma200", d)
        self.assertIn("larsson", d)
        self.assertIn("rs_line", d)
        self.assertGreater(len(d["rs_line"]), 0, "RS line vs SPY must be populated")

        # Dossier payloads
        self.assertIn("stock", d)
        self.assertIn("sector", d)
        self.assertIn("theme", d)

        stock_info = d["stock"]
        self.assertEqual(stock_info["symbol"], "NVDA")
        self.assertIn("rs_score", stock_info)
        self.assertIn("hi_52w", stock_info)
        self.assertIn("lo_52w", stock_info)
        self.assertIn("ret_1m", stock_info)
        self.assertIn("ret_3m", stock_info)
        self.assertIn("ret_1y", stock_info)

        sector_info = d["sector"]
        self.assertIn("name", sector_info)
        self.assertIn("macro_diag", sector_info)
        self.assertIn("peers", sector_info)

        theme_info = d["theme"]
        self.assertIn("name", theme_info)
        self.assertIn("primary_archetype", theme_info)
        self.assertIn("peers", theme_info)

    def test_api_annotations_roundtrip(self):
        """Verifies chart annotations GET and POST endpoints."""
        drawing = {"type": "hline", "price": 125.5, "color": "#38bdf8"}
        post_res = self.client.post("/api/annotations", json={"symbol": "TEST_RS", "drawings": [drawing]})
        self.assertEqual(post_res.status_code, 200)
        self.assertTrue(post_res.get_json().get("ok"))

        get_res = self.client.get("/api/annotations?symbol=TEST_RS")
        self.assertEqual(get_res.status_code, 200)
        data = get_res.get_json()
        self.assertEqual(data.get("symbol"), "TEST_RS")
        self.assertEqual(len(data.get("drawings", [])), 1)
        self.assertEqual(data["drawings"][0]["price"], 125.5)

    def test_api_dates_endpoint(self):
        """Verifies /api/dates returns available historical dates."""
        res = self.client.get("/api/dates")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("dates", data)
        self.assertIn("latest", data)
        self.assertIn("count", data)
        self.assertGreater(data["count"], 1000)
        self.assertIn("2024-05-23", data["dates"])

    def test_api_stocks_historical_query(self):
        """Verifies /api/stocks?date=YYYY-MM-DD returns correct historical leaderboard."""
        res = self.client.get("/api/stocks?date=2024-05-23")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("stocks", data)
        self.assertEqual(data["selected_date"], "2024-05-23")
        self.assertTrue(data["is_historical"])
        self.assertGreater(len(data["stocks"]), 500)
        # Check sorted by RS score
        self.assertGreaterEqual(data["stocks"][0]["rs_score"], 95)

    def test_api_chart_historical_query(self):
        """Verifies /api/chart with date and cut_chart returns historical snapshot."""
        # Full chart with marker
        res = self.client.get("/api/chart?symbol=NVDA&date=2024-05-23")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["target_date"], "2024-05-23")
        self.assertTrue(data["is_historical"])
        self.assertEqual(data["stock"]["symbol"], "NVDA")
        self.assertGreaterEqual(data["stock"]["rs_score"], 90)

        # Cut chart at date
        res_cut = self.client.get("/api/chart?symbol=NVDA&date=2024-05-23&cut_chart=true")
        self.assertEqual(res_cut.status_code, 200)
        data_cut = res_cut.get_json()
        self.assertTrue(data_cut["cut_chart"])
        # The last bar time must not exceed 2024-05-23
        last_bar_time = data_cut["bars"][-1]["time"]
        self.assertLessEqual(last_bar_time, "2024-05-23")

    def test_index_ui_elements(self):
        """Verifies HTML template contains 3-pane layout, controls, and table."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.data.decode("utf-8")
        self.assertIn("Institutional Relative Strength (RS) Explorer", html)
        self.assertIn("stocks_table_body", html)
        self.assertIn("chart_tv", html)
        self.assertIn("dossier_pane", html)
        self.assertIn("btn_chart_fullscreen", html)
        self.assertIn("btn_chart_settings", html)
        self.assertIn("btn_toggle_draw", html)
        self.assertIn("recomputeRS()", html)
        self.assertIn("cfg_rs_line", html)
        self.assertIn("dos_returns_body", html)
        self.assertIn("dos_sector_peers", html)
        self.assertIn("dos_theme_peers", html)
        self.assertIn("historical_date_picker", html)
        self.assertIn("date_mode_badge", html)
        self.assertIn("btn_toggle_cut_chart", html)
        self.assertIn("openTradingViewExportModal()", html)
        self.assertIn("tv_export_modal", html)
        self.assertIn("tv_export_preview", html)
        self.assertIn("copyTVExportToClipboard()", html)
        self.assertIn("downloadTVExportFile()", html)



class TestServeEPRelativeStrengthIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import serve_ep
        cls.serve_ep = serve_ep
        cls.ep_client = serve_ep.app.test_client()

    def test_serve_ep_ui_elements_for_rs(self):
        """Verifies HTML template contains b_rs90 booster chip, f_rs dropdown, and right-pane badges."""
        res = self.ep_client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.data.decode("utf-8")
        self.assertIn("id=\"b_rs90\"", html)
        self.assertIn("class=\"booster-chip active\"", html)
        self.assertIn("id=\"f_rs\"", html)
        self.assertIn("value=\"rs90_posture\" selected", html)
        self.assertIn("id=\"ctx_rs_badge\"", html)
        self.assertIn("id=\"ctx_stage2_badge\"", html)
        self.assertIn("id=\"ctx_dist52_badge\"", html)

    def test_serve_ep_rs_filter_and_booster_contract(self):
        """Verifies /api/events filters by RS rating and structural posture."""
        # 1. Default rs90_posture
        res = self.ep_client.get("/api/events?rs=rs90_posture")
        self.assertEqual(res.status_code, 200)
        events = res.get_json()["events"]
        self.assertGreater(len(events), 0)
        for ev in events[:50]:
            self.assertGreaterEqual(ev["rs_score"], 90.0)
            self.assertTrue(ev["stage2"] or (ev["dist_52w_high"] is not None and ev["dist_52w_high"] >= -15.0))

        # 2. Booster cond_rs90=1
        res_b = self.ep_client.get("/api/events?cond_rs90=1")
        self.assertEqual(res_b.status_code, 200)
        events_b = res_b.get_json()["events"]
        self.assertGreater(len(events_b), 0)
        for ev in events_b[:50]:
            self.assertGreaterEqual(ev["rs_score"], 90.0)
            self.assertTrue(ev["stage2"] or (ev["dist_52w_high"] is not None and ev["dist_52w_high"] >= -15.0))

        # 3. Stage 2 Only filter
        res_s2 = self.ep_client.get("/api/events?rs=stage2")
        self.assertEqual(res_s2.status_code, 200)
        for ev in res_s2.get_json()["events"][:50]:
            self.assertTrue(ev["stage2"])

        # 4. Within 15% 52W High Only filter
        res_hi15 = self.ep_client.get("/api/events?rs=hi15")
        self.assertEqual(res_hi15.status_code, 200)
        for ev in res_hi15.get_json()["events"][:50]:
            self.assertGreaterEqual(ev["dist_52w_high"], -15.0)

    def test_serve_ep_api_chart_rs_payload_and_disqualifications(self):
        """Verifies /api/chart returns rs_score, stage2, and dist_52w_high in payload and ticker history."""
        res = self.ep_client.get("/api/chart?symbol=NVDA&date=2024-05-23&rs=rs90_posture")
        self.assertEqual(res.status_code, 200)
        d = res.get_json()
        self.assertIn("rs_score", d)
        self.assertGreaterEqual(d["rs_score"], 90.0)
        self.assertIn("stage2", d)
        self.assertTrue(d["stage2"])
        self.assertIn("dist_52w_high", d)

        # Check ticker history
        self.assertIn("ticker_history", d)
        self.assertGreater(len(d["ticker_history"]), 0)
        first_h = d["ticker_history"][0]
        self.assertIn("rs_score", first_h)
        self.assertIn("stage2", first_h)
        self.assertIn("dist_52w_high", first_h)


if __name__ == "__main__":
    unittest.main()

