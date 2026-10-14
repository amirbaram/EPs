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
import config
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

    def test_chart_settings_larsson_and_draw_tools(self):
        """Verifies chart settings panel, Larsson line computation, drawing tools, and enlarge button in serve_qm."""
        client = serve_qm.app.test_client()

        # 1. Verify /api/chart returns larsson data, spans, and vol_ma20
        res = client.get("/api/chart?symbol=NVDA&date=2024-05-23&larsson_fastest=32")
        self.assertEqual(res.status_code, 200)
        d = res.get_json()
        self.assertIn("larsson", d)
        self.assertIn("larsson_spans", d)
        self.assertEqual(d["larsson_spans"], [32, 35, 50, 58])
        self.assertIn("vol_ma20", d)
        self.assertGreater(len(d["larsson"]), 0)
        l_first = d["larsson"][-1]
        self.assertIn("fast", l_first)
        self.assertIn("slow", l_first)
        self.assertIn("state", l_first)
        self.assertIn(l_first["state"], ["yellow", "blue", "gray"])

        # 2. Verify /api/annotations GET and POST
        test_drawings = [
            {"type": "line", "t1": "2024-05-20", "p1": 900.0, "t2": "2024-05-23", "p2": 1050.0}
        ]
        res_post = client.post("/api/annotations", json={"symbol": "TEST_QM", "drawings": test_drawings})
        self.assertEqual(res_post.status_code, 200)
        self.assertTrue(res_post.get_json().get("ok"))

        res_get = client.get("/api/annotations?symbol=TEST_QM")
        self.assertEqual(res_get.status_code, 200)
        d_anno = res_get.get_json()
        self.assertEqual(d_anno.get("symbol"), "TEST_QM")
        self.assertEqual(len(d_anno.get("drawings", [])), 1)
        self.assertEqual(d_anno["drawings"][0]["type"], "line")

        # 3. Verify UI Elements in index
        res_idx = client.get("/")
        self.assertEqual(res_idx.status_code, 200)
        html = res_idx.data.decode("utf-8")
        self.assertIn("btn_chart_settings", html)
        self.assertIn("btn_toggle_draw", html)
        self.assertIn("btn_chart_fullscreen", html)
        self.assertIn("chart_settings_panel", html)
        self.assertIn("chart_draw_toolbar", html)
        self.assertIn("larsson_ribbon_svg", html)
        self.assertIn("draw_canvas", html)
        self.assertIn("cfg_larsson_fastest", html)
        self.assertIn("toggleChartSettings()", html)
        self.assertIn("toggleDrawToolbar()", html)
        self.assertIn("toggleChartFullscreen()", html)

    def test_enlarged_chart_stock_list_and_navigation(self):
        """Verifies enlarged chart stock list chooser button, modal, and next/prev buttons."""
        client = serve_qm.app.test_client()
        res_idx = client.get("/")
        self.assertEqual(res_idx.status_code, 200)
        html = res_idx.data.decode("utf-8")

        # 1. Enlarged controls & navigation buttons in chart header
        self.assertIn("chart-enlarged-only-controls", html)
        self.assertIn("btn_enlarged_stock_list", html)
        self.assertIn("btn_chart_prev_ticker", html)
        self.assertIn("btn_chart_next_ticker", html)
        self.assertIn("lbl_enlarged_stock_count", html)

        # 2. Stock chooser modal elements
        self.assertIn("enlarged_stock_list_modal", html)
        self.assertIn("esl_search", html)
        self.assertIn("esl_table_body", html)
        self.assertIn("esl_count", html)
        self.assertIn("esl-table", html)

        # 3. JavaScript helper functions
        self.assertIn("openEnlargedStockListModal()", html)
        self.assertIn("closeEnlargedStockListModal", html)
        self.assertIn("renderEnlargedStockList", html)
        self.assertIn("stepFilteredStock(-1)", html)
        self.assertIn("stepFilteredStock(1)", html)
        self.assertIn("onEslRowClick", html)
        self.assertIn("updateEnlargedStockCount()", html)

    def test_qm_rs_score_and_posture_filtering(self):
        """Verifies RS score and Stage 2 / Near-ATH posture filtering, KPIs, and dossier metadata."""
        client = serve_qm.app.test_client()

        # 1. Default API call: rs_filter=rs90_stage2_or_15
        res_default = client.get("/api/events")
        self.assertEqual(res_default.status_code, 200)
        data_default = res_default.get_json()
        events_default = data_default.get("events", [])
        self.assertGreater(len(events_default), 0)

        # Verify that all events returned satisfy RS >= 90 and posture (Stage 2 OR within 15% 52W high)
        for ev in events_default[:50]:
            rs = ev.get("rs_score")
            s2 = ev.get("stage2")
            d52 = ev.get("dist_52w_high")
            self.assertIsNotNone(rs, f"Event {ev['symbol']} missing rs_score")
            self.assertGreaterEqual(rs, 90.0)
            self.assertTrue(bool(s2) or (d52 is not None and d52 >= -15.0),
                            f"Event {ev['symbol']} fails posture requirement: stage2={s2}, dist_52w={d52}")

        kpis = data_default.get("kpis", {})
        self.assertIn("rs_leader_pct", kpis)
        self.assertEqual(kpis["rs_leader_pct"], 100.0)

        # 2. Unfiltered API call: rs_filter=all
        res_all = client.get("/api/events?rs_filter=all")
        self.assertEqual(res_all.status_code, 200)
        data_all = res_all.get_json()
        self.assertGreater(data_all["kpis"]["total"], data_default["kpis"]["total"])

        # 3. Stage 2 Only API call
        res_s2 = client.get("/api/events?rs_filter=stage2_only")
        self.assertEqual(res_s2.status_code, 200)
        events_s2 = res_s2.get_json().get("events", [])
        for ev in events_s2[:30]:
            self.assertTrue(ev.get("stage2"), f"Event {ev['symbol']} should be stage2")

        # 4. Chart endpoint includes RS technicals in setup payload
        res_chart = client.get("/api/chart?symbol=NVDA&date=2024-05-23")
        self.assertEqual(res_chart.status_code, 200)
        d_chart = res_chart.get_json()
        setup = d_chart.get("setup", {})
        self.assertIn("rs_score", setup)
        self.assertIn("stage2", setup)
        self.assertIn("dist_52w_high", setup)

        # 5. UI Elements present in HTML
        res_idx = client.get("/")
        self.assertEqual(res_idx.status_code, 200)
        html = res_idx.data.decode("utf-8")
        self.assertIn("sel_rs", html)
        self.assertIn("rs90_stage2_or_15", html)
        self.assertIn("kpi_rs_leader", html)
        self.assertIn("card_rs_technicals", html)
        self.assertIn("dos_rs_score", html)
        self.assertIn("dos_stage2", html)
        self.assertIn("dos_dist_52w", html)
        self.assertIn("dos_rs_posture_status", html)

    def test_qm_breakout_strategy_mode_and_trade_info(self):
        """Verifies Institutional QM Breakout Strategy filtering, dynamic trade KPIs, and trade characteristics payload."""
        client = serve_qm.app.test_client()

        # 1. Verify /api/events with strategy_mode=1
        res = client.get("/api/events?strategy_mode=1")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("strategy_mode"))
        kpis = data.get("strategy_kpis")
        self.assertIsNotNone(kpis, "Missing strategy_kpis payload")
        self.assertGreater(kpis.get("total_trades", 0), 0)
        self.assertIn("win_rate", kpis)
        self.assertIn("profit_factor", kpis)
        self.assertIn("expectancy_r", kpis)
        self.assertIn("total_r", kpis)
        self.assertIn("sized_total_r", kpis)
        self.assertIn("avg_win_r", kpis)
        self.assertIn("avg_loss_r", kpis)
        self.assertIn("win_loss_ratio", kpis)
        self.assertIn("max_win_r", kpis)
        self.assertIn("max_loss_r", kpis)
        self.assertIn("avg_bars", kpis)

        # All events returned in strategy mode must be breakouts
        events = data.get("events", [])
        self.assertGreater(len(events), 0)
        for ev in events[:30]:
            self.assertEqual(ev.get("state"), "breakout")

        # 2. Verify specific strategy filter configuration: Leading macro + SPY 200 SMA + 0.6x Stop + Dynamic Sizing
        res_lead = client.get("/api/events?strategy_mode=1&strat_macro=leading&strat_spy=spy200&strat_stop=06x&strat_sizing=dynamic&strat_rs=rs90_lead")
        self.assertEqual(res_lead.status_code, 200)
        lead_data = res_lead.get_json()
        lead_kpis = lead_data.get("strategy_kpis")
        self.assertIsNotNone(lead_kpis)
        self.assertGreaterEqual(lead_kpis["total_trades"], 100)
        self.assertGreater(lead_kpis["expectancy_r"], 0.0)
        self.assertGreater(lead_kpis["profit_factor"], 1.0)
        self.assertGreater(lead_kpis["sized_total_r"], 0.0)

        # 3. Verify /api/chart returns trade_info object for breakout setups
        sample_ev = events[0]
        sym = sample_ev["symbol"]
        dt = sample_ev["date"]
        res_chart = client.get(f"/api/chart?symbol={sym}&date={dt}")
        self.assertEqual(res_chart.status_code, 200)
        d_chart = res_chart.get_json()
        setup = d_chart.get("setup", {})
        self.assertIn("trade_info", setup, f"Setup for {sym} {dt} missing trade_info")
        ti = setup["trade_info"]
        self.assertIn("entry_price", ti)
        self.assertIn("stop_loss_06x", ti)
        self.assertIn("risk_pct_06x", ti)
        self.assertIn("r_06x", ti)
        self.assertIn("sized_r_06x", ti)
        self.assertIn("pos_size_mult", ti)
        self.assertIn("s_ep", ti)
        self.assertIn("s_pole", ti)
        self.assertIn("s_rvol", ti)
        self.assertIn("s_pb", ti)
        self.assertIn("exit_date_06x", ti)
        self.assertIn("exit_price_06x", ti)
        self.assertIn("exit_reason_06x", ti)
        self.assertIn("spy_above_200sma", ti)
        self.assertIn("qqq_bull_stack", ti)
        self.assertIn("qqq_pullback", ti)

        # 4. Verify UI Elements in index HTML
        res_idx = client.get("/")
        self.assertEqual(res_idx.status_code, 200)
        html = res_idx.data.decode("utf-8")
        self.assertIn("btn_strategy_toggle", html)
        self.assertIn("btn_strategy_settings", html)
        self.assertIn("strategy_kpi_bar", html)
        self.assertIn("strategy_settings_modal", html)
        self.assertIn("card_strategy_trade", html)
        self.assertIn("dos_strat_entry", html)
        self.assertIn("dos_strat_stop", html)
        self.assertIn("dos_strat_r", html)
        self.assertIn("dos_strat_sized_r", html)
        self.assertIn("dos_strat_s_ep", html)
        self.assertIn("dos_strat_s_pole", html)
        self.assertIn("dos_strat_s_rvol", html)
        self.assertIn("dos_strat_s_pb", html)
        self.assertIn("toggleStrategyMode()", html)
        self.assertIn("openStrategySettingsModal()", html)
        self.assertIn("applyStrategySettings()", html)
        self.assertIn("updateStrategyKPIs", html)

    def test_large_cap_momentum_setups_and_symbol_search_regression(self):
        """Verifies institutional large-cap momentum setups (15-25% poles), 3-7 bars cons, and symbol search RS bypass."""
        import mine_qm_setups

        # 1. Verify QMB config: min_move 15%, 3-month pole duration (63 bars), and 3-bar min consolidation
        self.assertEqual(config.QMB_MIN_MOVE, 15.0)
        self.assertEqual(config.QMB_MOVE_MAX_BARS, 63)
        self.assertEqual(config.QMB_CONS_MIN, 3)

        # 2. Verify GOOGL scans multiple historical setups across 2016-2026
        googl_setups = mine_qm_setups.scan_single_symbol("GOOGL")
        self.assertGreaterEqual(len(googl_setups), 40, "GOOGL should yield >=40 historical setups with 3-7 bars and 3-mo pole")
        bo_setups = [s for s in googl_setups if s["state"] == "breakout"]
        self.assertGreaterEqual(len(bo_setups), 10, "GOOGL should yield >=10 historical breakout episodes")
        short_cons = [s for s in googl_setups if 3 <= s["cons_bars"] <= 7]
        self.assertGreaterEqual(len(short_cons), 15, "GOOGL should yield >=15 short 3-7 bar consolidations")

        # 3. Verify /api/events with search=GOOGL and cons_bars=3-7
        client = serve_qm.app.test_client()

        res_events = client.get("/api/events?cons_bars=3-7")
        self.assertEqual(res_events.status_code, 200)

        # Check UI template has min_move 15% option and cons_bars 3-7 option
        res_idx = client.get("/")
        html = res_idx.data.decode("utf-8")
        self.assertIn('value="15"', html)
        self.assertIn('value="20"', html)
        self.assertIn('value="3-7"', html)

    def test_googl_april_2026_pole_absolute_high_regression(self):
        """Regression test: verifies GOOGL April 2026 setup (pole starting 2026-03-30 into mid-April)

        is detected. Previously, an older peak from Feb 3, 2026 (348.55) masked the fresh rally
        leg due to global 108-bar idxmax(). With launch-trough anchoring, the absolute high achieved
        since the launch low within the valid consolidation window is properly identified.
        """
        import datastore
        import mine_qm_setups

        df = datastore.load_bars("GOOGL")
        self.assertIsNotNone(df, "GOOGL bars must be present in datastore")
        df = indicators.add_indicators(df)

        # 1. Verify building state on 2026-04-24
        loc_apr24 = df.index.get_loc("2026-04-24")
        res_build = setups.detect_qm_breakout(df.iloc[: loc_apr24 + 1])
        self.assertIsNotNone(res_build, "GOOGL on 2026-04-24 should be detected as building QM setup")
        self.assertEqual(res_build["state"], "building")
        self.assertEqual(res_build["qm_pole"][0], "2026-03-30")
        self.assertEqual(res_build["qm_pole"][1], "2026-04-17")
        self.assertGreaterEqual(res_build["move_pct"], 25.0)
        self.assertGreaterEqual(res_build["ride_frac"], setups.config.QMB_MIN_RIDE_FRAC)

        # 2. Verify breakout state on 2026-04-30
        loc_apr30 = df.index.get_loc("2026-04-30")
        res_break = setups.detect_qm_breakout(df.iloc[: loc_apr30 + 1])
        self.assertIsNotNone(res_break, "GOOGL on 2026-04-30 should be detected as breakout QM setup")
        self.assertEqual(res_break["state"], "breakout")
        self.assertEqual(res_break["qm_pole"][0], "2026-03-30")

        # 3. Verify scan_single_symbol captures the April 2026 episode with launch-trough debouncing
        googl_episodes = mine_qm_setups.scan_single_symbol("GOOGL")
        apr_episodes = [
            e for e in googl_episodes
            if e["qm_pole_start"] == "2026-03-30" and "2026-04" in e["date"]
        ]
        self.assertEqual(len(apr_episodes), 1, "Grinding daily highs should debounce into exactly 1 April episode")
        self.assertEqual(apr_episodes[0]["date"], "2026-04-30")
        self.assertEqual(apr_episodes[0]["state"], "breakout")

    def test_bb_squeeze_and_linear_regression_filters_and_multi_setup_toggles_regression(self):
        """Verifies BB Squeeze (pinch_ratio) and Pole R2 filters in /api/events,
        multi-setup payload in /api/chart, and UI controls for toggling multiple setups.
        """
        client = serve_qm.app.test_client()

        # 1. Verify /api/events returns r2 and pinch_ratio fields
        res = client.get("/api/events?limit=10")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("events", data)
        self.assertGreater(len(data["events"]), 0)
        ev0 = data["events"][0]
        self.assertIn("r2", ev0)
        self.assertIn("pinch_ratio", ev0)

        # 2. Verify BB Squeeze filtering
        res_tight = client.get("/api/events?bb_pinch=tight")
        self.assertEqual(res_tight.status_code, 200)
        d_tight = res_tight.get_json()
        for ev in d_tight["events"][:20]:
            if ev.get("pinch_ratio") is not None:
                self.assertLessEqual(ev["pinch_ratio"], 2.5)

        # 3. Verify Pole R2 filtering
        res_r2 = client.get("/api/events?r2=0.80")
        self.assertEqual(res_r2.status_code, 200)
        d_r2 = res_r2.get_json()
        for ev in d_r2["events"][:20]:
            if ev.get("r2") is not None:
                self.assertGreaterEqual(ev["r2"], 0.80)

        # 4. Verify /api/chart returns symbol_setups for multi-setup rendering
        sym = ev0["symbol"]
        res_chart = client.get(f"/api/chart?symbol={sym}&date={ev0['date']}")
        self.assertEqual(res_chart.status_code, 200)
        chart_data = res_chart.get_json()
        self.assertIn("symbol_setups", chart_data)
        self.assertIsInstance(chart_data["symbol_setups"], list)
        self.assertIn("r2", chart_data["setup"])
        self.assertIn("pinch_ratio", chart_data["setup"])

        # 5. Verify UI template contains BB Squeeze & R2 dropdowns, checkboxes, and multi-setup canvas overlays
        res_idx = client.get("/")
        self.assertEqual(res_idx.status_code, 200)
        html = res_idx.data.decode("utf-8")
        self.assertIn('id="sel_bb_pinch"', html)
        self.assertIn('id="sel_r2"', html)
        self.assertIn('id="header_select_all"', html)
        self.assertIn('toggleSetupCheckbox', html)
        self.assertIn('SETUP_PALETTE', html)
        self.assertIn('selectedSetupKeys', html)
        self.assertIn('dos_r2', html)
        self.assertIn('dos_pinch_ratio', html)

    def test_strategy_mode_bb_squeeze_and_r2_filtering_regression(self):
        """Verifies Strategy Mode integrates Bollinger Band squeeze and Pole R2 filters into trade simulation KPIs and event filtering."""
        client = serve_qm.app.test_client()

        # 1. Baseline strategy mode
        res_base = client.get("/api/events?strategy_mode=1&strat_macro=all&strat_spy=all&strat_rs=all")
        self.assertEqual(res_base.status_code, 200)
        data_base = res_base.get_json()
        self.assertIn("strategy_kpis", data_base)
        kpi_base = data_base["strategy_kpis"]
        self.assertGreater(kpi_base["total_trades"], 0)

        # 2. Strategy mode with Tight BB Squeeze (<= 2.5) and Pole R2 (>= 0.65)
        res_opt = client.get("/api/events?strategy_mode=1&strat_macro=all&strat_spy=all&strat_rs=all&strat_bb_pinch=tight&strat_r2=0.65")
        self.assertEqual(res_opt.status_code, 200)
        data_opt = res_opt.get_json()
        self.assertIn("strategy_kpis", data_opt)
        kpi_opt = data_opt["strategy_kpis"]
        self.assertEqual(kpi_opt["strat_bb_pinch"], "tight")
        self.assertEqual(kpi_opt["strat_r2"], "0.65")
        self.assertGreater(kpi_opt["total_trades"], 0)

        # Verify that tighter BB pinch + R2 >= 0.65 yields positive expectancy, strong win/loss ratio and filtered subset
        self.assertGreater(kpi_opt["expectancy_r"], 0.0)
        self.assertGreater(kpi_opt["profit_factor"], 1.0)
        self.assertGreaterEqual(kpi_opt["win_loss_ratio"], 2.0)
        self.assertLess(kpi_opt["total_trades"], kpi_base["total_trades"])

        # Verify events returned adhere to filters
        for ev in data_opt["events"][:20]:
            if ev.get("pinch_ratio") is not None:
                self.assertLessEqual(ev["pinch_ratio"], 2.5)
            if ev.get("r2") is not None:
                self.assertGreaterEqual(ev["r2"], 0.65)

        # 3. Verify Strategy Settings modal has both controls
        res_ui = client.get("/")
        html = res_ui.data.decode("utf-8")
        self.assertIn('id="cfg_strat_bb_pinch"', html)
        self.assertIn('id="cfg_strat_r2"', html)

    def test_leading_sector_and_theme_filters_regression(self):
        """Verifies that filtering by leading_sectors or leading_themes returns positive matches in both normal and strategy mode."""
        client = serve_qm.app.test_client()

        # 1. Normal mode: leading sectors
        res_sec = client.get("/api/events?sector=leading_sectors")
        self.assertEqual(res_sec.status_code, 200)
        d_sec = res_sec.get_json()
        self.assertGreater(d_sec["kpis"]["total"], 0, "Filtering sector=leading_sectors in normal mode must return > 0 results")
        self.assertGreater(len(d_sec["events"]), 0)

        # 2. Normal mode: leading themes
        res_thm = client.get("/api/events?theme=leading_themes")
        self.assertEqual(res_thm.status_code, 200)
        d_thm = res_thm.get_json()
        self.assertGreater(d_thm["kpis"]["total"], 0, "Filtering theme=leading_themes in normal mode must return > 0 results")
        self.assertGreater(len(d_thm["events"]), 0)

        # 3. Strategy mode: leading sectors
        res_strat_sec = client.get("/api/events?strategy_mode=1&sector=leading_sectors&strat_macro=all&strat_spy=all&strat_rs=all")
        self.assertEqual(res_strat_sec.status_code, 200)
        d_strat_sec = res_strat_sec.get_json()
        self.assertGreater(d_strat_sec["kpis"]["total"], 0, "Filtering sector=leading_sectors in strategy mode must return > 0 results")
        self.assertGreater(d_strat_sec["strategy_kpis"]["total_trades"], 0)

        # 4. Strategy mode: leading themes
        res_strat_thm = client.get("/api/events?strategy_mode=1&theme=leading_themes&strat_macro=all&strat_spy=all&strat_rs=all")
        self.assertEqual(res_strat_thm.status_code, 200)
        d_strat_thm = res_strat_thm.get_json()
        self.assertGreater(d_strat_thm["kpis"]["total"], 0, "Filtering theme=leading_themes in strategy mode must return > 0 results")
        self.assertGreater(d_strat_thm["strategy_kpis"]["total_trades"], 0)

    def test_strategy_mode_trailing_exit_models_regression(self):
        """Verifies Strategy Mode supports multiple trailing exit models (10ema, 20ema, 50ema, staged, balanced) and reflects in KPIs and Dossier."""
        client = serve_qm.app.test_client()

        # 1. 10 EMA Trailing Exit
        res_10 = client.get("/api/events?strategy_mode=1&strat_macro=all&strat_spy=all&strat_rs=all&strat_exit=10ema")
        self.assertEqual(res_10.status_code, 200)
        d_10 = res_10.get_json()
        self.assertIn("strategy_kpis", d_10)
        kpi_10 = d_10["strategy_kpis"]
        self.assertEqual(kpi_10["strat_exit"], "10ema")
        self.assertGreater(kpi_10["total_trades"], 0)

        # 2. 50 EMA Trailing Exit (Expectancy & bars held should be higher than 10 EMA)
        res_50 = client.get("/api/events?strategy_mode=1&strat_macro=all&strat_spy=all&strat_rs=all&strat_exit=50ema")
        self.assertEqual(res_50.status_code, 200)
        d_50 = res_50.get_json()
        self.assertIn("strategy_kpis", d_50)
        kpi_50 = d_50["strategy_kpis"]
        self.assertEqual(kpi_50["strat_exit"], "50ema")
        self.assertGreater(kpi_50["expectancy_r"], kpi_10["expectancy_r"])
        self.assertGreater(kpi_50["avg_bars"], kpi_10["avg_bars"])

        # 3. Staged Scaling (33/33/34)
        res_staged = client.get("/api/events?strategy_mode=1&strat_macro=all&strat_spy=all&strat_rs=all&strat_exit=staged")
        self.assertEqual(res_staged.status_code, 200)
        d_staged = res_staged.get_json()
        kpi_staged = d_staged["strategy_kpis"]
        self.assertEqual(kpi_staged["strat_exit"], "staged")
        self.assertGreater(kpi_staged["win_rate"], kpi_50["win_rate"])
        self.assertGreater(kpi_staged["expectancy_r"], kpi_10["expectancy_r"])

        # 4. Balanced Scaling (50/50)
        res_bal = client.get("/api/events?strategy_mode=1&strat_macro=all&strat_spy=all&strat_rs=all&strat_exit=balanced")
        self.assertEqual(res_bal.status_code, 200)
        d_bal = res_bal.get_json()
        kpi_bal = d_bal["strategy_kpis"]
        self.assertEqual(kpi_bal["strat_exit"], "balanced")
        self.assertGreater(kpi_bal["expectancy_r"], kpi_10["expectancy_r"])

        # 5. Verify /api/chart returns multi-exit trade_info fields for breakout trades
        ev0 = d_10["events"][0]
        res_chart = client.get(f"/api/chart?symbol={ev0['symbol']}&date={ev0['date']}")
        self.assertEqual(res_chart.status_code, 200)
        chart_data = res_chart.get_json()
        if "trade_info" in chart_data["setup"]:
            ti = chart_data["setup"]["trade_info"]
            self.assertIn("r_e10_06x", ti)
            self.assertIn("r_e20_06x", ti)
            self.assertIn("r_e50_06x", ti)
            self.assertIn("r_staged_06x", ti)
            self.assertIn("r_balanced_06x", ti)

        # 6. Verify UI contains Exit Model toolbar dropdown and settings modal selector
        res_ui = client.get("/")
        html = res_ui.data.decode("utf-8")
        self.assertIn('id="sel_strat_exit"', html)
        self.assertIn('id="cfg_strat_exit"', html)
        self.assertIn('onStratExitChange', html)

    def test_trader_archetypes_and_ml_climax_exits_regression(self):
        """Verifies Trader Archetype presets, ML Climax partial scaling models, and UI integration."""
        client = serve_qm.app.test_client()

        # 1. Test ML Climax Swing Exit KPI
        res_cswing = client.get("/api/events?strategy_mode=1&strat_macro=all&strat_spy=all&strat_rs=all&strat_exit=climax_swing")
        self.assertEqual(res_cswing.status_code, 200)
        d_cswing = res_cswing.get_json()
        kpi_cswing = d_cswing["strategy_kpis"]
        self.assertEqual(kpi_cswing["strat_exit"], "climax_swing")
        self.assertGreater(kpi_cswing["total_trades"], 0)
        self.assertGreater(kpi_cswing["expectancy_r"], 0.0)
        self.assertGreater(kpi_cswing["profit_factor"], 1.0)

        # 2. Test ML Climax Hybrid Exit KPI
        res_chybrid = client.get("/api/events?strategy_mode=1&strat_macro=all&strat_spy=all&strat_rs=all&strat_exit=climax_hybrid")
        self.assertEqual(res_chybrid.status_code, 200)
        d_chybrid = res_chybrid.get_json()
        kpi_chybrid = d_chybrid["strategy_kpis"]
        self.assertEqual(kpi_chybrid["strat_exit"], "climax_hybrid")
        self.assertGreater(kpi_chybrid["total_trades"], 0)
        self.assertGreater(kpi_chybrid["expectancy_r"], 0.0)

        # 3. Test ML Climax Trend Exit KPI (1.0x stop)
        res_ctrend = client.get("/api/events?strategy_mode=1&strat_macro=all&strat_spy=all&strat_rs=all&strat_stop=10x&strat_exit=climax_trend")
        self.assertEqual(res_ctrend.status_code, 200)
        d_ctrend = res_ctrend.get_json()
        kpi_ctrend = d_ctrend["strategy_kpis"]
        self.assertEqual(kpi_ctrend["strat_exit"], "climax_trend")
        self.assertEqual(kpi_ctrend["strat_stop"], "10x")
        self.assertGreater(kpi_ctrend["total_trades"], 0)

        # 4. Verify /api/chart returns ML Climax metrics in trade_info
        ev0 = d_cswing["events"][0]
        res_chart = client.get(f"/api/chart?symbol={ev0['symbol']}&date={ev0['date']}")
        self.assertEqual(res_chart.status_code, 200)
        chart_data = res_chart.get_json()
        if "trade_info" in chart_data["setup"]:
            ti = chart_data["setup"]["trade_info"]
            self.assertIn("r_climax_swing_06x", ti)
            self.assertIn("r_climax_hybrid_06x", ti)
            self.assertIn("r_climax_trend_06x", ti)
            self.assertIn("climax_date_06x", ti)
            self.assertIn("climax_price_06x", ti)

        # 5. Verify UI elements for Archetypes and Climax Exits
        res_ui = client.get("/")
        html = res_ui.data.decode("utf-8")
        self.assertIn('id="sel_trader_archetype"', html)
        self.assertIn('id="cfg_trader_archetype"', html)
        self.assertIn('id="grp_strat_archetype" style="display:flex;', html)
        self.assertIn('applyArchetypePreset', html)
        self.assertIn('onTraderArchetypeChange', html)
        self.assertIn('climax_swing', html)
        self.assertIn('climax_hybrid', html)
        self.assertIn('climax_trend', html)
        self.assertIn("'climax_swing': 'climax_swing'", html)
        self.assertIn("strategySettings.bb_pinch = 'all'", html)
        self.assertIn("strategySettings.fresh_ep = '30d'", html)
        self.assertIn("strategySettings.rvol = '1.5x'", html)

        # 6. Verify KPI Synchronization and Archetype Presets
        res_hv = client.get("/api/events?strategy_mode=1&strat_macro=leading&strat_spy=spy200&strat_stop=06x&strat_exit=climax_swing&strat_sizing=dynamic&strat_rs=rs90_lead&strat_bb_pinch=tight")
        self.assertEqual(res_hv.status_code, 200)
        d_hv = res_hv.get_json()
        self.assertEqual(d_hv["kpis"]["total"], d_hv["strategy_kpis"]["total_trades"])
        self.assertGreater(d_hv["strategy_kpis"]["total_trades"], 0)
        self.assertGreater(d_hv["strategy_kpis"]["expectancy_r"], 0.0)

        res_ih = client.get("/api/events?strategy_mode=1&strat_macro=leading&strat_spy=spy200&strat_stop=06x&strat_exit=climax_hybrid&strat_sizing=dynamic&strat_rs=rs90_lead&strat_fresh_ep=30d&strat_rvol=1.5x")
        self.assertEqual(res_ih.status_code, 200)
        d_ih = res_ih.get_json()
        self.assertEqual(d_ih["kpis"]["total"], d_ih["strategy_kpis"]["total_trades"])
        self.assertGreater(d_ih["strategy_kpis"]["total_trades"], 0)

        # 7. Verify strat_rs=all synchronization
        res_all_rs = client.get("/api/events?strategy_mode=1&strat_rs=all")
        self.assertEqual(res_all_rs.status_code, 200)
        d_all_rs = res_all_rs.get_json()
        self.assertEqual(d_all_rs["kpis"]["total"], d_all_rs["strategy_kpis"]["total_trades"])

        # 8. Verify toolbar macro chips in strategy mode synchronize total counts
        res_chip_sec = client.get("/api/events?strategy_mode=1&macro=leading_sectors")
        self.assertEqual(res_chip_sec.status_code, 200)
        d_chip_sec = res_chip_sec.get_json()
        self.assertEqual(d_chip_sec["kpis"]["total"], d_chip_sec["strategy_kpis"]["total_trades"])

    def test_qm_ml_triad_and_tier_filtering_regression(self):
        """Verifies ML Setup Quality (QM-V2) tier filters, Dynamic Quantile Stop (QM-Stop),
        and Institutional Triad synergy in serve_qm API and UI.
        """
        client = serve_qm.app.test_client()

        # 1. Verify ML Tier Filtering: Premier
        res_prem = client.get("/api/events?strategy_mode=1&strat_tier=premier")
        self.assertEqual(res_prem.status_code, 200)
        d_prem = res_prem.get_json()
        self.assertIn("events", d_prem)
        self.assertGreater(len(d_prem["events"]), 0)
        for ev in d_prem["events"]:
            self.assertEqual(ev.get("qm_tier"), "Premier")

        # 2. Verify ML Tier Filtering: High Quality (Premier + High Quality)
        res_hq = client.get("/api/events?strategy_mode=1&strat_tier=high_quality")
        self.assertEqual(res_hq.status_code, 200)
        d_hq = res_hq.get_json()
        self.assertGreater(len(d_hq["events"]), len(d_prem["events"]))
        for ev in d_hq["events"]:
            self.assertIn(ev.get("qm_tier"), ["Premier", "High Quality"])

        # 3. Verify ML Tier Filtering: Avoid Traps (excludes Toxic / Trap)
        res_at = client.get("/api/events?strategy_mode=1&strat_tier=avoid_traps")
        self.assertEqual(res_at.status_code, 200)
        d_at = res_at.get_json()
        self.assertGreater(len(d_at["events"]), len(d_hq["events"]))
        for ev in d_at["events"]:
            self.assertNotEqual(ev.get("qm_tier"), "Toxic / Trap")

        # 4. Verify Dynamic Quantile Stop Loss KPI
        res_dyn = client.get("/api/events?strategy_mode=1&strat_stop=dynamic&strat_exit=climax_swing")
        self.assertEqual(res_dyn.status_code, 200)
        d_dyn = res_dyn.get_json()
        kpi_dyn = d_dyn["strategy_kpis"]
        self.assertEqual(kpi_dyn["strat_stop"], "dynamic")
        self.assertGreater(kpi_dyn["total_trades"], 0)
        self.assertGreater(kpi_dyn["expectancy_r"], 0.5)

        # 5. Verify Institutional Triad Synergistic Setup
        res_triad = client.get("/api/events?strategy_mode=1&strat_stop=dynamic&strat_exit=triad_optimal&strat_tier=premier&strat_sizing=dynamic")
        self.assertEqual(res_triad.status_code, 200)
        d_triad = res_triad.get_json()
        kpi_triad = d_triad["strategy_kpis"]
        self.assertEqual(kpi_triad["strat_exit"], "triad_optimal")
        self.assertGreaterEqual(kpi_triad["win_rate"], 50.0)
        self.assertGreater(kpi_triad["expectancy_r"], 1.0)
        self.assertGreater(kpi_triad["profit_factor"], 5.0)

        # 6. Verify UI markup includes ML quality tiers, dynamic stop, targets, and triad preset
        res_ui = client.get("/")
        html = res_ui.data.decode("utf-8")
        self.assertIn('id="sel_strat_tier"', html)
        self.assertIn('id="cfg_strat_tier"', html)
        self.assertIn('id="card_ml_guidance"', html)
        self.assertIn('id="dos_ml_hurdle_25r"', html)
        self.assertIn('id="dos_ml_hurdle_35r"', html)
        self.assertIn('id="dos_ml_plan"', html)
        self.assertIn('institutional_triad', html)
        self.assertIn('triad_optimal', html)
        self.assertIn('onStratTierChange', html)
        self.assertIn('💎 PREMIER', html)

        # 7. Verify /api/chart returns ML Setup Quality and Guidance
        ev0 = d_prem["events"][0]
        res_chart = client.get(f"/api/chart?symbol={ev0['symbol']}&date={ev0['date']}")
        self.assertEqual(res_chart.status_code, 200)
        chart_data = res_chart.get_json()
        setup = chart_data.get("setup", {})
        self.assertTrue("trade_info" in setup or "ml_guidance" in setup)
        if "trade_info" in setup:
            ti = setup["trade_info"]
            self.assertIn("qm_tier", ti)
            self.assertIn("p_big_move", ti)
            self.assertIn("p_trap", ti)
            self.assertIn("r_triad_optimal", ti)

        # 8. Verify Setup Mode (strategy_mode=0) ML Tier Filtering
        res_sm0_prem = client.get("/api/events?strategy_mode=0&strat_tier=premier")
        self.assertEqual(res_sm0_prem.status_code, 200)
        d_sm0_prem = res_sm0_prem.get_json()
        self.assertGreater(len(d_sm0_prem["events"]), 0)
        for ev in d_sm0_prem["events"]:
            self.assertEqual(ev.get("qm_tier"), "Premier")

        res_sm0_trap = client.get("/api/events?strategy_mode=0&strat_tier=avoid_traps")
        self.assertEqual(res_sm0_trap.status_code, 200)
        d_sm0_trap = res_sm0_trap.get_json()
        self.assertGreater(len(d_sm0_trap["events"]), len(d_sm0_prem["events"]))
        for ev in d_sm0_trap["events"]:
            self.assertNotEqual(ev.get("qm_tier"), "Toxic / Trap")

        # 9. Verify Fresh Setups (state=building) ML Tier Filtering
        res_bld_prem = client.get("/api/events?state=building&strat_tier=premier")
        self.assertEqual(res_bld_prem.status_code, 200)
        d_bld_prem = res_bld_prem.get_json()
        self.assertGreater(len(d_bld_prem["events"]), 0)
        for ev in d_bld_prem["events"]:
            self.assertEqual(ev.get("state"), "building")
            self.assertEqual(ev.get("qm_tier"), "Premier")

        res_bld_trap = client.get("/api/events?state=building&strat_tier=avoid_traps")
        self.assertEqual(res_bld_trap.status_code, 200)
        d_bld_trap = res_bld_trap.get_json()
        for ev in d_bld_trap["events"]:
            self.assertEqual(ev.get("state"), "building")
            self.assertNotEqual(ev.get("qm_tier"), "Toxic / Trap")

        # 10. Verify Fresh Building Setup Chart Pricing and ML Guidance Targets
        bld_ev = d_bld_prem["events"][0]
        res_bld_chart = client.get(f"/api/chart?symbol={bld_ev['symbol']}&date={bld_ev['date']}")
        self.assertEqual(res_bld_chart.status_code, 200)
        bld_c_data = res_bld_chart.get_json()
        bld_setup = bld_c_data.get("setup", {})
        self.assertIn("ml_guidance", bld_setup)
        ml_g = bld_setup["ml_guidance"]
        self.assertIn("quality", ml_g)
        self.assertEqual(ml_g["quality"]["tier"], "Premier")
        self.assertIn("targets", ml_g)
        self.assertIn("hurdle_25r", ml_g["targets"])
        self.assertIn("hurdle_35r", ml_g["targets"])
        self.assertIn("scale_out_plan", ml_g["targets"])

        # Verify dynamic stop price is strictly anchored to the setup price, NOT the 2026 last bar price
        stop_p = ml_g["stop_loss"]["dynamic_stop_price"]
        trig_p = bld_setup["trigger"]
        self.assertLess(stop_p, trig_p, "Dynamic stop price must be strictly below setup trigger price")
        self.assertGreater(stop_p, trig_p * 0.85, "Dynamic stop price should remain within reasonable distance of setup")


if __name__ == "__main__":
    unittest.main()



