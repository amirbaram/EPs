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
        self.assertIn("cfg_pivot_intrabar_price", html)
        self.assertIn("fmtPivotIntrabarSuffix", html)
        self.assertIn("exh_intrabar_trig_row", html)
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

        # 6. 4-Model Structural Pivot Suite Detection + Intrabar Trigger Prices on TWLO & MRNA
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

        # Verify intrabar trigger price bounds:
        # - For Pivot Highs: low <= intrabar_trigger_price <= high (and triggers before high on wide bars)
        # - For Pivot Lows : low <= intrabar_trigger_price <= high (and triggers above low on wide bars)
        for e in ut_hi_50:
            self.assertIn("intrabar_trigger_price", e)
            self.assertIn("trigger_prices_by_thr", e)
            self.assertGreaterEqual(e["intrabar_trigger_price"], e["low"] - 0.01)
            self.assertLessEqual(e["intrabar_trigger_price"], e["high"] + 0.01)
        self.assertTrue(any(e["intrabar_trigger_price"] < e["high"] for e in ut_hi_50))

        for e in ut_lo_50:
            self.assertIn("intrabar_trigger_price", e)
            self.assertIn("trigger_prices_by_thr", e)
            self.assertGreaterEqual(e["intrabar_trigger_price"], e["low"] - 0.01)
            self.assertLessEqual(e["intrabar_trigger_price"], e["high"] + 0.01)
        self.assertTrue(any(e["intrabar_trigger_price"] > e["low"] for e in ut_lo_50))

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

    def test_piecewise_inverse_wilder_rsi_and_intrabar_evaluation(self):
        import numpy as np
        import pandas as pd
        import datastore
        import indicators
        import rsi_extremes
        import train_all_pivot_models as tap

        def exact_wilder_rsi(series_c: pd.Series, length: int) -> pd.Series:
            chg = series_c.astype(float).diff()
            u = chg.clip(lower=0.0).ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
            d = (-chg).clip(lower=0.0).ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
            return 100.0 - 100.0 / (1.0 + u / d)

        # Build a price series that crosses deep oversold (< 15) and deep overbought (> 85)
        # so both RSI[t-1] <= R and RSI[t-1] > R regimes are exercised for every target R.
        idx = pd.date_range("2025-01-01", periods=80, freq="B")
        vals = [100.0]
        for i in range(1, 80):
            if i < 20:
                vals.append(vals[-1] * (1.035 if i % 4 != 0 else 0.992))  # surge to RSI > 85
            elif i < 45:
                vals.append(vals[-1] * (0.962 if i % 5 != 0 else 1.008))  # plunge to RSI < 15
            elif i < 65:
                vals.append(vals[-1] * (1.032 if i % 4 != 0 else 0.990))  # second surge > 80
            else:
                vals.append(vals[-1] * (0.968 if i % 4 != 0 else 1.005))  # second plunge < 20
        close_s = pd.Series(vals, index=idx, dtype=float)

        for length in (14, 9, 5):
            base_rsi = exact_wilder_rsi(close_s, length)
            prev_rsi = base_rsi.shift(1)
            for target_lvl in (20.0, 25.0, 30.0, 50.0, 70.0, 75.0, 80.0):
                p_ind = indicators.rsi_trigger_price(close_s, target_lvl, length=length)
                p_os = rsi_extremes.os_trigger_price(close_s, target_lvl, length=length)
                p_ob = rsi_extremes.ob_trigger_price(close_s, target_lvl, length=length)

                # Warmup bars must be NaN
                self.assertTrue(p_ind.iloc[:length].isna().all())
                np.testing.assert_allclose(p_ind.iloc[length:].to_numpy(), p_os.iloc[length:].to_numpy(), rtol=0, atol=1e-12)
                np.testing.assert_allclose(p_ind.iloc[length:].to_numpy(), p_ob.iloc[length:].to_numpy(), rtol=0, atol=1e-12)

                # Ensure both regimes (RSI[t-1] <= R and RSI[t-1] > R) are present
                valid_prev = prev_rsi.iloc[length + 1 :]
                self.assertTrue((valid_prev <= target_lvl).any(), f"Missing RSI[t-1] <= {target_lvl} regime for len={length}")
                self.assertTrue((valid_prev > target_lvl).any(), f"Missing RSI[t-1] > {target_lvl} regime for len={length}")

                # Round-trip check at every bar t >= length + 1:
                for t in range(length + 1, len(close_s)):
                    c_mod = close_s.iloc[: t + 1].copy()
                    c_mod.iloc[t] = p_ind.iloc[t]
                    rsi_at_target = float(exact_wilder_rsi(c_mod, length).iloc[t])
                    self.assertLess(
                        abs(rsi_at_target - target_lvl),
                        1e-9,
                        f"Round-trip failed at t={t}, len={length}, target={target_lvl}: got {rsi_at_target}",
                    )

            # Causality check: truncating future bars never alters trigger price at bar t
            p_full = indicators.rsi_trigger_price(close_s, 75.0, length=length)
            p_trunc = indicators.rsi_trigger_price(close_s.iloc[:40], 75.0, length=length)
            np.testing.assert_allclose(
                p_full.iloc[length:40].to_numpy(),
                p_trunc.iloc[length:40].to_numpy(),
                rtol=0,
                atol=1e-12,
            )

        # Verify exact intrabar Wilder RSI evaluation on straddling bars vs linear interpolation
        df_twlo = datastore.load_bars("TWLO")
        self.assertIsNotNone(df_twlo)
        df_sub = df_twlo.tail(160).copy()
        b_feat = tap.compute_all_pivot_features(df_sub)
        sub_rma = serve_watchlist._ensure_causal_rma_state(df_sub, b_feat.tail(60).copy())

        # Pick a bar that straddles prev_close (low < prev_close < high)
        straddle_mask = (sub_rma["low"] < sub_rma["prev_close"]) & (sub_rma["high"] > sub_rma["prev_close"])
        self.assertTrue(straddle_mask.any())
        pos_idx = int(np.flatnonzero(straddle_mask.to_numpy())[-1])
        bar_dt = sub_rma.index[pos_idx]
        full_loc = df_sub.index.get_loc(bar_dt)

        lo_v = float(sub_rma["low"].iloc[pos_idx])
        hi_v = float(sub_rma["high"].iloc[pos_idx])
        grid_p = np.linspace(lo_v, hi_v, 21)
        big_test = pd.concat([sub_rma.iloc[[pos_idx]]] * 21, ignore_index=True)

        for period in (14, 9, 5):
            rsi_vec = serve_watchlist._eval_intrabar_rsi_vec(grid_p, big_test, period, np.zeros(21))
            # Compare against exact bar-by-bar Wilder RSI when close[t] is set to each candidate price
            for k, p_cand in enumerate(grid_p):
                c_copy = df_sub["close"].iloc[: full_loc + 1].astype(float).copy()
                c_copy.iloc[-1] = float(p_cand)
                chg = c_copy.diff()
                ag = float(chg.clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().iloc[-1])
                al = float((-chg.clip(upper=0.0)).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().iloc[-1])
                expected_rsi = 100.0 - (100.0 / (1.0 + ag / (al + 1e-9)))
                self.assertLess(abs(float(rsi_vec[k]) - expected_rsi), 1e-9)

        # Verify compute_universal_exhaustion includes exact ob75_target_price and os25_target_price
        ex_out = serve_watchlist.compute_universal_exhaustion(df_sub, df_sub.tail(60), 80.0)
        self.assertIn("ob75_target_price", ex_out)
        self.assertIn("os25_target_price", ex_out)
        self.assertIsNotNone(ex_out["ob75_target_price"])
        self.assertIsNotNone(ex_out["os25_target_price"])
        self.assertGreater(ex_out["ob75_target_price"], ex_out["os25_target_price"])
        for tgt_lvl, tgt_key in ((75.0, "ob75_target_price"), (25.0, "os25_target_price")):
            c_last = df_sub["close"].astype(float).copy()
            c_last.iloc[-1] = float(ex_out[tgt_key])
            rsi_last = float(exact_wilder_rsi(c_last, 14).iloc[-1])
            self.assertLess(abs(rsi_last - tgt_lvl), 1e-9)

        # Edge cases: Series with RangeIndex, ndarray levels, out-of-domain levels (<=0 or >=100), and rsi_extremes.fires
        p_scalar = indicators.rsi_trigger_price(close_s, 75.0, length=14)
        p_range_ser = indicators.rsi_trigger_price(close_s, pd.Series([75.0] * len(close_s)), length=14)
        p_ndarray = indicators.rsi_trigger_price(close_s, np.full(len(close_s), 75.0), length=14)
        np.testing.assert_allclose(p_scalar.iloc[14:].to_numpy(), p_range_ser.iloc[14:].to_numpy(), rtol=0, atol=1e-12)
        np.testing.assert_allclose(p_scalar.iloc[14:].to_numpy(), p_ndarray.iloc[14:].to_numpy(), rtol=0, atol=1e-12)
        self.assertTrue(indicators.rsi_trigger_price(close_s, 0.0, length=14).isna().all())
        self.assertTrue(indicators.rsi_trigger_price(close_s, 100.0, length=14).isna().all())
        self.assertTrue(rsi_extremes.os_trigger_price(close_s, 0.0, length=14).isna().all())
        self.assertTrue(rsi_extremes.ob_trigger_price(close_s, 100.0, length=14).isna().all())
        _, p_fire_long, _ = rsi_extremes.fires(df_twlo.tail(250), "long")
        _, p_fire_short, _ = rsi_extremes.fires(df_twlo.tail(250), "short")
        self.assertTrue(p_fire_long.notna().any())
        self.assertTrue(p_fire_short.notna().any())

        # Verify 2-bar snapped stall events (date != trigger_date) align causal t-1 RMA state to target_pos (date)
        ex_252 = serve_watchlist.compute_universal_exhaustion(df_twlo, df_twlo.tail(252), 80.0)
        snapped_evs = [e for e in ex_252["ut_high_events"] if e.get("trigger_date") and e["date"] != e["trigger_date"]]
        self.assertGreaterEqual(len(snapped_evs), 1)
        for sev in snapped_evs:
            self.assertGreaterEqual(sev["intrabar_trigger_price"], sev["low"] - 0.01)
            self.assertLessEqual(sev["intrabar_trigger_price"], sev["high"] + 0.01)

    def test_v3_vs_v4_model_selector_and_threshold_intrabar_trigger_monotonicity(self):
        # 1. Verify UI selector and interpolation helpers exist in HTML
        resp_html = self.client.get("/")
        self.assertEqual(resp_html.status_code, 200)
        html = resp_html.get_data(as_text=True)
        self.assertIn("cfg_pivot_model_ver", html)
        self.assertIn("onPivotModelVerChange()", html)
        self.assertIn("resolvePivotTriggerPrice", html)
        self.assertIn("updateIntrabarTriggerRow", html)

        # 2. Verify both V3 and V4 model files exist on disk (V3 preserved untouched, V4 added alongside)
        for p_v3, p_v4 in [
            (serve_watchlist.UPTREND_PIVOT_MODEL_PATH, serve_watchlist.UPTREND_PIVOT_MODEL_V4_PATH),
            (serve_watchlist.UPTREND_PIVOT_LOW_MODEL_PATH, serve_watchlist.UPTREND_PIVOT_LOW_MODEL_V4_PATH),
            (serve_watchlist.DOWNTREND_PIVOT_HIGH_MODEL_PATH, serve_watchlist.DOWNTREND_PIVOT_HIGH_MODEL_V4_PATH),
            (serve_watchlist.DOWNTREND_PIVOT_LOW_MODEL_PATH, serve_watchlist.DOWNTREND_PIVOT_LOW_MODEL_V4_PATH),
        ]:
            self.assertTrue(p_v3.exists(), f"Missing preserved V3 model {p_v3}")
            self.assertTrue(p_v4.exists(), f"Missing V4 model {p_v4}")

        m_v3 = serve_watchlist.get_all_pivot_models("v3")
        m_v4 = serve_watchlist.get_all_pivot_models("v4")
        self.assertEqual(set(m_v3.keys()), {"ut_high", "ut_low", "dt_high", "dt_low"})
        self.assertEqual(set(m_v4.keys()), {"ut_high", "ut_low", "dt_high", "dt_low"})
        for k in ("ut_high", "ut_low", "dt_high", "dt_low"):
            self.assertEqual(m_v4[k].get("version"), "4.0")
            self.assertIn("lgb_a_plus", m_v4[k])
            self.assertIn("xgb_a_plus", m_v4[k])
            self.assertIn("iso_calibrator", m_v4[k])

        # 3. Query /api/chart on TWLO with model_ver=v3 vs model_ver=v4
        r_v3 = self.client.get("/api/chart?symbol=TWLO&history=2y&model_ver=v3")
        r_v4 = self.client.get("/api/chart?symbol=TWLO&history=2y&model_ver=v4")
        self.assertEqual(r_v3.status_code, 200)
        self.assertEqual(r_v4.status_code, 200)
        ex_v3 = r_v3.get_json()["exhaustion"]
        ex_v4 = r_v4.get_json()["exhaustion"]
        self.assertEqual(ex_v3["model_ver"], "v3")
        self.assertEqual(ex_v4["model_ver"], "v4")

        # V3 and V4 must produce distinct probability series on TWLO
        s_v3 = [pt["ut_high"] for pt in ex_v3["series"]]
        s_v4 = [pt["ut_high"] for pt in ex_v4["series"]]
        self.assertEqual(len(s_v3), len(s_v4))
        self.assertNotEqual(s_v3, s_v4)

        # 4. Verify strict intrabar trigger price monotonicity across thresholds (35% vs 50% vs 65%)
        for ex in (ex_v3, ex_v4):
            hi_evs_65 = [
                e
                for e in (ex["ut_high_events"] + ex["dt_high_events"])
                if e["prob_pct"] >= 65.0 and (e["high"] - e["low"]) >= 0.50
            ]
            lo_evs_65 = [
                e
                for e in (ex["ut_low_events"] + ex["dt_low_events"])
                if e["prob_pct"] >= 65.0 and (e["high"] - e["low"]) >= 0.50
            ]
            self.assertGreaterEqual(len(hi_evs_65), 1)
            self.assertGreaterEqual(len(lo_evs_65), 1)

            for e in hi_evs_65:
                by_t = e["trigger_prices_by_thr"]
                self.assertIn("35", by_t)
                self.assertIn("50", by_t)
                self.assertIn("65", by_t)
                self.assertLess(by_t["35"], by_t["50"], f"High pivot {e['date']} 35% must trigger below 50%: {by_t}")
                self.assertLess(by_t["50"], by_t["65"], f"High pivot {e['date']} 50% must trigger below 65%: {by_t}")
                self.assertLessEqual(by_t["65"], e["high"] + 1e-6)
                self.assertGreaterEqual(by_t["35"], e["low"] - 1e-6)

            for e in lo_evs_65:
                by_t = e["trigger_prices_by_thr"]
                self.assertIn("35", by_t)
                self.assertIn("50", by_t)
                self.assertIn("65", by_t)
                self.assertGreater(by_t["35"], by_t["50"], f"Low pivot {e['date']} 35% must trigger above 50%: {by_t}")
                self.assertGreater(by_t["50"], by_t["65"], f"Low pivot {e['date']} 50% must trigger above 65%: {by_t}")
                self.assertGreaterEqual(by_t["65"], e["low"] - 1e-6)
                self.assertLessEqual(by_t["35"], e["high"] + 1e-6)


if __name__ == "__main__":
    unittest.main()


