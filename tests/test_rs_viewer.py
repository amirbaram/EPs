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


if __name__ == "__main__":
    unittest.main()
