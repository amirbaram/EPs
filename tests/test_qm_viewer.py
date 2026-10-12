"""tests/test_qm_viewer.py — Regression Test Suite for QM Viewer & EP Cross-Referencing System.

Verifies:
1. QM setup detection geometry (pole, consolidation, surfing, higher lows).
2. EP cross-referencing proximity calculation (prior igniter <=60d, post booster <=60d).
3. EP badge classification (Ignited Pole, Boosted Breakout, Pure Technical).
4. serve_qm.py Flask API contracts (/api/events, /api/chart, /api/reload_dataset).
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import setups
import indicators
from mine_qm_setups import _match_eps_for_setup
import serve_qm


class TestQMViewer(unittest.TestCase):

    def _generate_synthetic_qm_series(self, breakout: bool = True) -> pd.DataFrame:
        """Generates a clean synthetic OHLCV dataframe forming a qualifying QM pole + base."""
        dates = pd.date_range("2024-01-01", periods=160, freq="B")
        n = len(dates)

        close = np.full(n, 50.0)
        open_p = np.full(n, 50.0)
        high = np.full(n, 50.5)
        low = np.full(n, 49.5)
        volume = np.full(n, 1_000_000.0)

        # Baseline drift bars 0..100
        for i in range(1, 100):
            close[i] = close[i - 1] * (1.0 + 0.001)
            open_p[i] = close[i - 1]
            high[i] = max(open_p[i], close[i]) + 0.4
            low[i] = min(open_p[i], close[i]) - 0.4

        # Explosive Momentum Pole: bars 100..120 (+50% move over 20 bars)
        for i in range(100, 121):
            close[i] = close[i - 1] * 1.021
            open_p[i] = close[i - 1]
            high[i] = close[i] + 0.5
            low[i] = open_p[i] - 0.2
            volume[i] = 3_000_000.0

        # Consolidation Base: bars 121..140 (20 bars tight surfing with higher lows)
        hi_p = high[120]
        base_low = hi_p * 0.92  # 8% shallow pullback
        for i in range(121, 140):
            # Ascending higher lows
            progress = (i - 121) / 20.0
            l_val = base_low + progress * (hi_p * 0.04)
            c_val = l_val + 0.8
            open_p[i] = c_val - 0.3
            close[i] = c_val
            high[i] = c_val + 0.4
            low[i] = l_val
            volume[i] = 800_000.0  # volume contracting

        # Bar 140: Breakout or Building
        if breakout:
            open_p[140] = close[139]
            close[140] = hi_p + 1.5  # clear pop above pole high
            high[140] = close[140] + 0.5
            low[140] = open_p[140] - 0.2
            volume[140] = 2_500_000.0  # volume expansion >= 1.3x baseline
        else:
            open_p[140] = close[139]
            close[140] = close[139] * 0.995
            high[140] = close[140] + 0.3
            low[140] = close[140] - 0.3
            volume[140] = 700_000.0

        df = pd.DataFrame({
            "open": open_p[:141],
            "high": high[:141],
            "low": low[:141],
            "close": close[:141],
            "volume": volume[:141],
        }, index=dates[:141])

        df = indicators.add_indicators(df)
        return df

    def test_qm_geometry_breakout_detection(self):
        """Verifies detect_qm_breakout flags synthetic volume-expanded breakout."""
        df = self._generate_synthetic_qm_series(breakout=True)
        res = setups.detect_qm_breakout(df)
        self.assertIsNotNone(res, "QM breakout should trigger on synthetic volume-expanded breakout")
        self.assertEqual(res["setup"], "qm_breakout")
        self.assertEqual(res["state"], "breakout")
        self.assertGreaterEqual(res["move_pct"], 30.0)
        self.assertGreaterEqual(res["cons_bars"], 8)
        self.assertLessEqual(res["pullback_pct"], 50.0)
        self.assertIn("qm_pole", res)
        self.assertIn("qm_cons", res)
        self.assertEqual(len(res["qm_pole"]), 2)
        self.assertEqual(len(res["qm_cons"]), 2)

    def test_qm_geometry_building_detection(self):
        """Verifies detect_qm_breakout returns 'building' while coiling under resistance."""
        df = self._generate_synthetic_qm_series(breakout=False)
        res = setups.detect_qm_breakout(df)
        self.assertIsNotNone(res, "QM setup should be detected in building state")
        self.assertEqual(res["state"], "building")
        self.assertGreaterEqual(res["ride_frac"], setups.config.QMB_MIN_RIDE_FRAC)

    def test_ep_cross_reference_proximity_matching(self):
        """Verifies EP proximity delta, days ago, days later, and badges."""
        qm_date = "2024-05-15"
        ep_list = [
            # Prior EP 25 days before
            {
                "symbol": "TEST",
                "date": "2024-04-20",
                "dt": pd.to_datetime("2024-04-20"),
                "gap_pct": 14.5,
                "rvol": 4.2,
                "close_pos": 0.88,
                "subtype": "classic",
            },
            # Post EP 15 days later
            {
                "symbol": "TEST",
                "date": "2024-05-30",
                "dt": pd.to_datetime("2024-05-30"),
                "gap_pct": 21.0,
                "rvol": 5.1,
                "close_pos": 0.95,
                "subtype": "classic",
            },
        ]

        meta = _match_eps_for_setup("TEST", qm_date, ep_list)
        self.assertTrue(meta["has_prior_ep"])
        self.assertEqual(meta["prior_ep_date"], "2024-04-20")
        self.assertEqual(meta["prior_ep_days_ago"], 25)
        self.assertEqual(meta["prior_ep_gap_pct"], 14.5)
        self.assertEqual(meta["prior_ep_rvol"], 4.2)

        self.assertTrue(meta["has_post_ep"])
        self.assertEqual(meta["post_ep_date"], "2024-05-30")
        self.assertEqual(meta["post_ep_days_later"], 15)
        self.assertEqual(meta["post_ep_gap_pct"], 21.0)
        self.assertEqual(meta["post_ep_rvol"], 5.1)

        self.assertEqual(meta["ep_badge"], "EP-Ignited & Boosted")

    def test_ep_badge_classification_pure_technical(self):
        """Verifies setup without EP in [-60, +60] window receives Pure Technical QM badge."""
        qm_date = "2024-05-15"
        ep_list = [
            # Distant EP 120 days ago
            {
                "symbol": "TEST",
                "date": "2024-01-15",
                "dt": pd.to_datetime("2024-01-15"),
                "gap_pct": 10.0,
                "rvol": 3.0,
                "close_pos": 0.80,
                "subtype": "classic",
            }
        ]

        meta = _match_eps_for_setup("TEST", qm_date, ep_list)
        self.assertFalse(meta["has_prior_ep"])
        self.assertFalse(meta["has_post_ep"])
        self.assertEqual(meta["ep_badge"], "Pure Technical QM")

    def test_serve_qm_flask_api_contracts(self):
        """Verifies serve_qm Flask app routes /api/events and /api/reload_dataset."""
        client = serve_qm.app.test_client()

        # Reload dataset endpoint
        res = client.get("/api/reload_dataset")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "ok")

        # Events endpoint
        res_ev = client.get("/api/events?state=all&ep_link=all")
        self.assertEqual(res_ev.status_code, 200)
        ev_data = res_ev.get_json()
        self.assertIn("events", ev_data)
        self.assertIn("kpis", ev_data)
        self.assertIn("total", ev_data["kpis"])
        self.assertIn("breakouts", ev_data["kpis"])
        self.assertIn("building", ev_data["kpis"])

        # Index page verifies dedicated volume scale and autoScale reset logic
        res_idx = client.get("/")
        self.assertEqual(res_idx.status_code, 200)
        self.assertIn(b"QM MOMENTUM VIEWER", res_idx.data)
        self.assertIn(b"priceScaleId: 'volume'", res_idx.data)
        self.assertIn(b"resetChartScale()", res_idx.data)
        self.assertIn(b"autoScale: true", res_idx.data)

    def test_serve_qm_chart_api_contract(self):
        """Verifies /api/chart endpoint returns bars, indicators, positive levels, and setup metadata."""
        client = serve_qm.app.test_client()
        # Fetch first event to test chart endpoint
        res_ev = client.get("/api/events?limit=1")
        self.assertEqual(res_ev.status_code, 200)
        ev_data = res_ev.get_json()
        if ev_data.get("events"):
            ev = ev_data["events"][0]
            res_chart = client.get(f"/api/chart?symbol={ev['symbol']}&date={ev['date']}")
            self.assertEqual(res_chart.status_code, 200)
            c_data = res_chart.get_json()
            self.assertEqual(c_data["symbol"], ev["symbol"])
            self.assertIn("bars", c_data)
            self.assertIn("volume", c_data)
            self.assertIn("ema10", c_data)
            self.assertIn("ema20", c_data)
            self.assertIn("sma50", c_data)
            self.assertIn("sma200", c_data)
            self.assertIn("setup", c_data)
            self.assertIn("sector_theme", c_data)
            self.assertIn("sector", c_data["sector_theme"])
            self.assertIn("theme", c_data["sector_theme"])
            self.assertIn("macro_status", c_data["sector_theme"])
            self.assertGreater(len(c_data["bars"]), 0)
            self.assertGreater(c_data["setup"]["level"], 0)
            self.assertGreater(c_data["setup"]["trigger"], 0)

    def test_serve_qm_group_stocks_api_contract(self):
        """Verifies /api/group_stocks endpoint returns constituent stocks and metrics."""
        client = serve_qm.app.test_client()
        res = client.get("/api/group_stocks?group=Semiconductors&kind=theme")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("group", data)
        self.assertIn("kind", data)
        self.assertIn("stocks", data)
        self.assertIn("count", data)
        self.assertIn("info", data)

        # Check HTML contains the sector/theme backdrop card and leaderboard modal
        res_idx = client.get("/")
        self.assertEqual(res_idx.status_code, 200)
        self.assertIn(b"card_sec_thm", res_idx.data)
        self.assertIn(b"group_stocks_modal", res_idx.data)
        self.assertIn(b"openGroupStocksModal", res_idx.data)

    def test_serve_qm_macro_flow_filters(self):
        """Verifies filtering by leading sectors, leading themes, and emerging themes."""
        client = serve_qm.app.test_client()

        # 1. Macro Filter: leading_sectors
        res_secs = client.get("/api/events?macro=leading_sectors")
        self.assertEqual(res_secs.status_code, 200)
        d_secs = res_secs.get_json()
        self.assertIn("leading_sectors", d_secs)
        self.assertGreater(len(d_secs["leading_sectors"]), 0)
        for ev in d_secs["events"][:20]:
            self.assertTrue(ev["is_leading_sector"])

        # 2. Macro Filter: leading_themes
        res_thms = client.get("/api/events?macro=leading_themes")
        self.assertEqual(res_thms.status_code, 200)
        d_thms = res_thms.get_json()
        self.assertIn("leading_themes", d_thms)
        self.assertGreater(len(d_thms["leading_themes"]), 0)
        for ev in d_thms["events"][:20]:
            self.assertTrue(ev["is_leading_theme"])

        # 3. Macro Filter: emerging_themes
        res_emrg = client.get("/api/events?macro=emerging_themes")
        self.assertEqual(res_emrg.status_code, 200)
        d_emrg = res_emrg.get_json()
        self.assertIn("emerging_themes", d_emrg)
        self.assertGreater(len(d_emrg["emerging_themes"]), 0)
        for ev in d_emrg["events"][:20]:
            self.assertTrue(ev["is_emerging_theme"])

        # 4. Sector & Theme Dropdown Special Values
        res_drop_sec = client.get("/api/events?sector=leading_sectors")
        self.assertEqual(res_drop_sec.status_code, 200)
        self.assertEqual(len(res_drop_sec.get_json()["events"]), len(d_secs["events"]))

        res_drop_emrg = client.get("/api/events?theme=emerging_themes")
        self.assertEqual(res_drop_emrg.status_code, 200)
        self.assertEqual(len(res_drop_emrg.get_json()["events"]), len(d_emrg["events"]))

        # 5. UI Elements
        res_idx = client.get("/")
        self.assertEqual(res_idx.status_code, 200)
        self.assertIn(b"f_macro_leading_sectors", res_idx.data)
        self.assertIn(b"f_macro_leading_themes", res_idx.data)
        self.assertIn(b"f_macro_emerging_themes", res_idx.data)
        self.assertIn(b"kpi_thematic_lead", res_idx.data)
        self.assertIn(b"onSectorChange()", res_idx.data)
        self.assertIn(b"onThemeChange()", res_idx.data)


if __name__ == "__main__":
    unittest.main()


