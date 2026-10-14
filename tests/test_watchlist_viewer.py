import json
import tempfile
import unittest
from pathlib import Path
import serve_watchlist


class TestWatchlistViewer(unittest.TestCase):
    def setUp(self):
        serve_watchlist.app.config["TESTING"] = True
        self.client = serve_watchlist.app.test_client()
        self._orig_wl_file = serve_watchlist.WATCHLISTS_FILE
        self._tmp_dir = tempfile.TemporaryDirectory()
        serve_watchlist.WATCHLISTS_FILE = Path(self._tmp_dir.name) / "test_watchlists.json"

    def tearDown(self):
        serve_watchlist.WATCHLISTS_FILE = self._orig_wl_file
        self._tmp_dir.cleanup()

    def test_index_html_renders_all_core_controls(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("EP Chart & Watchlist Studio", html)
        self.assertIn("toggleRatioMode()", html)
        self.assertIn("toggleInfoPanel()", html)
        self.assertIn("cfg_larsson", html)
        self.assertIn("cfg_ep_markers", html)
        self.assertIn("cfg_climax_markers", html)
        self.assertIn("cfg_pivot_low_markers", html)
        self.assertIn("cfg_climax_thresh", html)
        self.assertIn("cfg_ema_span_0", html)
        self.assertIn("crosshairMarkerVisible: false", html)
        self.assertIn("ml_exhaustion_card", html)
        self.assertIn("piv_ut_low_val", html)
        self.assertIn("piv_dt_high_val", html)
        self.assertIn("piv_dt_low_val", html)
        self.assertIn("setDrawTool", html)

    def test_catalog_endpoint_includes_stocks_spdr_sectors_themes(self):
        resp = self.client.get("/api/catalog?limit=50")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("etfs", data)
        self.assertIn("sectors", data)
        self.assertIn("themes", data)
        self.assertIn("stocks", data)
        self.assertGreater(len(data["etfs"]), 20)
        self.assertGreater(len(data["sectors"]), 100)
        self.assertGreater(len(data["themes"]), 100)
        self.assertGreater(len(data["stocks"]), 10)

        etf_syms = {x["symbol"] for x in data["etfs"]}
        self.assertIn("XBI", etf_syms)
        self.assertIn("XLK", etf_syms)
        self.assertIn("SPY", etf_syms)

    def test_watchlists_crud_operations(self):
        # 1. Initial GET creates default watchlists
        r1 = self.client.get("/api/watchlists")
        self.assertEqual(r1.status_code, 200)
        d1 = r1.get_json()
        self.assertGreaterEqual(len(d1["groups"]), 4)

        # 2. Create a new watchlist group
        r2 = self.client.post(
            "/api/watchlists",
            data=json.dumps({"action": "create_group", "name": "Biotech & Ratios"}),
            content_type="application/json",
        )
        self.assertEqual(r2.status_code, 200)
        d2 = r2.get_json()
        new_gid = d2["active_group_id"]
        self.assertTrue(any(g["name"] == "Biotech & Ratios" for g in d2["groups"]))

        # 3. Add a SPDR ETF, a Sector, and a Ratio chart to the new group
        for item_id in ["XBI", "SECTOR:Biotechnology", "RATIO:XBI|SPY"]:
            r_add = self.client.post(
                "/api/watchlists",
                data=json.dumps({"action": "add_item", "group_id": new_gid, "id": item_id}),
                content_type="application/json",
            )
            self.assertEqual(r_add.status_code, 200)

        # 4. Update item flag and notes
        r_upd = self.client.post(
            "/api/watchlists",
            data=json.dumps({
                "action": "update_item",
                "group_id": new_gid,
                "id": "XBI",
                "flag": "green",
                "notes": "Key Biotech Breakout Level",
            }),
            content_type="application/json",
        )
        d_upd = r_upd.get_json()
        grp = next(g for g in d_upd["groups"] if g["id"] == new_gid)
        xbi_item = next(it for it in grp["items"] if it["id"] == "XBI")
        self.assertEqual(xbi_item["flag"], "green")
        self.assertEqual(xbi_item["notes"], "Key Biotech Breakout Level")

        # 5. Remove item
        r_rem = self.client.post(
            "/api/watchlists",
            data=json.dumps({"action": "remove_item", "group_id": new_gid, "id": "XBI"}),
            content_type="application/json",
        )
        d_rem = r_rem.get_json()
        grp_after = next(g for g in d_rem["groups"] if g["id"] == new_gid)
        self.assertFalse(any(it["id"] == "XBI" for it in grp_after["items"]))

    def test_chart_stock_spdr_sector_theme_and_ratio(self):
        # 1. Single Stock (NVDA) with custom EMA spans (e.g. 12 EMA instead of 8 EMA)
        r_stk = self.client.get("/api/chart?symbol=NVDA&history=1y&larsson_fastest=32&ema_spans=12,10,20,21&sma_spans=50,150")
        self.assertEqual(r_stk.status_code, 200)
        d_stk = r_stk.get_json()
        self.assertFalse(d_stk["is_ratio"])
        self.assertGreater(len(d_stk["bars"]), 100)
        self.assertEqual(d_stk["larsson_spans"], [32, 35, 50, 58])
        self.assertEqual(d_stk["ema_spans"], [12, 10, 20, 21])
        self.assertEqual(d_stk["sma_spans"], [50, 150])
        self.assertEqual(len(d_stk["emas"]), 4)
        self.assertIn("ret_1m", d_stk["performance"])
        self.assertIsInstance(d_stk["ep_events"], list)
        self.assertIn("exhaustion", d_stk)
        self.assertIn("current_prob_pct", d_stk["exhaustion"])
        self.assertGreaterEqual(d_stk["exhaustion"]["current_prob_pct"], 0.0)
        self.assertLessEqual(d_stk["exhaustion"]["current_prob_pct"], 100.0)

        # 2. SPDR Sector ETF (XBI) — verifies universal ML exhaustion works without an EP event
        r_etf = self.client.get("/api/chart?symbol=XBI&history=1y")
        self.assertEqual(r_etf.status_code, 200)
        d_etf = r_etf.get_json()
        self.assertEqual(d_etf["part_a"]["type"], "etf")
        self.assertGreater(len(d_etf["bars"]), 100)
        self.assertIn("current_prob_pct", d_etf["exhaustion"])

        # 3. Sector Composite Index (SECTOR:Semiconductors)
        r_sec = self.client.get("/api/chart?symbol=SECTOR:Semiconductors&history=1y")
        self.assertEqual(r_sec.status_code, 200)
        d_sec = r_sec.get_json()
        self.assertEqual(d_sec["part_a"]["type"], "sector")
        self.assertGreater(len(d_sec["bars"]), 50)
        self.assertIn("current_prob_pct", d_sec["exhaustion"])

        # 4. Theme Composite Index (THEME:Artificial Intelligence)
        r_thm = self.client.get("/api/chart?symbol=THEME:Artificial%20Intelligence&history=1y")
        self.assertEqual(r_thm.status_code, 200)
        d_thm = r_thm.get_json()
        self.assertEqual(d_thm["part_a"]["type"], "theme")
        self.assertGreater(len(d_thm["bars"]), 50)

        # 5. Ratio Chart: Sector vs SPDR ETF (SECTOR:Biotechnology ÷ XBI)
        r_rat = self.client.get("/api/chart?symbol=SECTOR:Biotechnology&compare=XBI&history=1y")
        self.assertEqual(r_rat.status_code, 200)
        d_rat = r_rat.get_json()
        self.assertTrue(d_rat["is_ratio"])
        self.assertIsNotNone(d_rat["performance_a"])
        self.assertIsNotNone(d_rat["performance_b"])
        self.assertGreater(len(d_rat["bars"]), 50)
        self.assertIn("current_prob_pct", d_rat["exhaustion"])

        # 6. 4-Model Structural Pivot Suite Detection on TWLO & MRNA
        r_twlo = self.client.get("/api/chart?symbol=TWLO&history=1y")
        self.assertEqual(r_twlo.status_code, 200)
        ex_twlo = r_twlo.get_json()["exhaustion"]
        self.assertIn("pivots_current", ex_twlo)
        self.assertIn("pivots_5d_peak", ex_twlo)
        self.assertIn("ut_high_events", ex_twlo)
        self.assertIn("ut_low_events", ex_twlo)
        self.assertIn("dt_high_events", ex_twlo)
        self.assertIn("dt_low_events", ex_twlo)
        ut_hi_50 = [e for e in ex_twlo["ut_high_events"] if e["prob_pct"] >= 50.0]
        ut_lo_50 = [e for e in ex_twlo["ut_low_events"] if e["prob_pct"] >= 50.0]
        dt_lo_50 = [e for e in ex_twlo["dt_low_events"] if e["prob_pct"] >= 50.0]
        self.assertGreaterEqual(len(ut_hi_50), 4)
        self.assertGreaterEqual(len(ut_lo_50), 2)
        self.assertGreaterEqual(len(dt_lo_50), 1)

        r_mrna = self.client.get("/api/chart?symbol=MRNA&history=1y")
        self.assertEqual(r_mrna.status_code, 200)
        ex_mrna = r_mrna.get_json()["exhaustion"]
        dt_hi_50 = [e for e in ex_mrna["dt_high_events"] if e["prob_pct"] >= 50.0]
        self.assertGreaterEqual(len(dt_hi_50), 1)

    def test_annotations_save_and_load(self):
        sample_drawings = [{"type": "hline", "price": 125.5, "color": "#38bdf8"}]
        r_post = self.client.post(
            "/api/annotations",
            data=json.dumps({"symbol": "RATIO:XBI|SPY", "drawings": sample_drawings}),
            content_type="application/json",
        )
        self.assertEqual(r_post.status_code, 200)

        r_get = self.client.get("/api/annotations?symbol=RATIO:XBI|SPY")
        self.assertEqual(r_get.status_code, 200)
        self.assertEqual(len(r_get.get_json()["drawings"]), 1)

    def test_reload_dataset_and_causal_pivot_extraction(self):
        import numpy as np
        import pandas as pd
        import daily_update
        import datastore

        # 1. Verify RSP and SPDR ETFs are in datastore.app_critical_etfs()
        crit = datastore.app_critical_etfs()
        for sym in ("SPY", "QQQ", "RSP", "IWM", "XBI", "SMH", "XLK", "XLE"):
            self.assertIn(sym, crit)

        # 2. Verify daily_update covers port 8789 and all 4 pivot models
        ports = [p for p, _ in daily_update.TARGET_SERVERS]
        self.assertIn(8789, ports)
        self.assertEqual(len(daily_update.PIVOT_MODEL_PATHS), 4)

        # 3. Verify /api/reload_dataset endpoint works
        r_rel = self.client.post("/api/reload_dataset")
        self.assertEqual(r_rel.status_code, 200)
        rel_json = r_rel.get_json()
        self.assertEqual(rel_json["status"], "reloaded")
        self.assertEqual(len(rel_json["pivot_models_loaded"]), 4)

        # 4. Verify _extract_snapped_pivot_events is strictly causal (future bars cannot suppress bar t)
        dates = pd.date_range("2026-01-01", periods=10, freq="B")
        sub_t = pd.DataFrame(
            {
                "high": [100.0, 101.0, 102.0, 104.0, 106.0, 120.0, 125.0, 122.0, 121.0, 120.0],
                "low": [98.0, 99.0, 100.0, 102.0, 104.0, 105.0, 118.0, 117.0, 116.0, 115.0],
                "adr20": [2.0] * 10,
                "rsi14_high": [65.0, 68.0, 70.0, 74.0, 78.0, 85.0, 88.0, 75.0, 72.0, 70.0],
                "rsi14_low": [45.0] * 10,
            },
            index=dates,
        )
        probs = np.array([0.1, 0.1, 0.1, 0.1, 0.55, 0.1, 0.1, 0.1, 0.1, 0.1])
        mask = np.array([True] * 10)
        # Truncated at index 4 (bar 4 is live rightmost bar) vs full 10 bars (where bar 5 & 6 jump to 120, 125)
        ev_live = serve_watchlist._extract_snapped_pivot_events(sub_t.iloc[:5], probs[:5], probs[:5], probs[:5], mask[:5], True, "ut_high")
        ev_full = serve_watchlist._extract_snapped_pivot_events(sub_t, probs, probs, probs, mask, True, "ut_high")
        self.assertEqual(len(ev_live), 1)
        self.assertEqual(ev_live[0]["date"], ev_full[0]["date"])
        self.assertEqual(ev_live[0]["prob_pct"], ev_full[0]["prob_pct"])


if __name__ == "__main__":
    unittest.main()

