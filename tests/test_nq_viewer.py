import unittest
import numpy as np
import pandas as pd

import nq_ml_engine as nqe
import serve_nq


class TestNQIntradayEngineAndServer(unittest.TestCase):
    def setUp(self):
        self.client = serve_nq.app.test_client()

    def test_summary_endpoint_and_oos_superiority(self):
        resp = self.client.get("/api/summary")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("timeframes", data)
        for tf in ("15m", "1h", "4h"):
            self.assertIn(tf, data["timeframes"])
            tf_d = data["timeframes"][tf]
            hi_auc = tf_d["pivot_high_metrics"]["dual_roc_auc"]
            lo_auc = tf_d["pivot_low_metrics"]["dual_roc_auc"]
            eq_hi_auc = tf_d["generic_equity_baseline"]["eq_high_roc_auc"]
            eq_lo_auc = tf_d["generic_equity_baseline"]["eq_low_roc_auc"]

            # NQ-specific models must substantially outperform generic daily equity models on OOS
            self.assertGreater(hi_auc, 0.78)
            self.assertGreater(lo_auc, 0.80)
            self.assertGreater(hi_auc, eq_hi_auc + 0.10)
            self.assertGreater(lo_auc, eq_lo_auc + 0.10)

            # Quintiles Q1..Q5 must be populated and monotonically sorted by predicted MFE/MAE ratio
            q_list = tf_d["horizons"]["8"]["quintiles"]
            self.assertEqual(len(q_list), 5)
            self.assertGreater(q_list[-1]["pred_ratio_avg"], q_list[0]["pred_ratio_avg"] + 0.25)
            self.assertGreater(q_list[-1]["realized_ratio"], 0.75)

    def test_signals_endpoint_filtering(self):
        for tf in ("15m", "1h", "4h"):
            resp = self.client.get(f"/api/signals?tf={tf}&horizon=8&prob_th=0.55&ratio_th=1.30&period=oos")
            self.assertEqual(resp.status_code, 200)
            payload = resp.get_json()
            self.assertEqual(payload["tf"], tf)
            self.assertGreater(payload["kpis"]["total_signals"], 10)
            self.assertGreater(payload["kpis"]["realized_mfe_mae_ratio"], 0.75)
            self.assertIn("chronological_executed_trades", payload["kpis"])
            sig = payload["signals"][0]
            self.assertIn("trigger_px", sig)
            self.assertIn("target_px", sig)
            self.assertIn("stop_px", sig)
            self.assertGreaterEqual(sig["prob_pct"], 55.0)

    def test_chart_endpoint_and_index_ui(self):
        resp_ui = self.client.get("/")
        self.assertEqual(resp_ui.status_code, 200)
        html = resp_ui.get_data(as_text=True)
        self.assertIn("NQ Intraday Pivot High/Low & Forward MAE/MFE Prediction Studio", html)
        self.assertIn("chk-non-actionable", html)
        self.assertIn("inspectBarByTimestamp", html)

        # Default /api/chart loads ALL history (59,677 bars on 1h)
        resp_all = self.client.get("/api/chart?tf=1h&horizon=8&prob_th=0.55&ratio_th=1.30")
        self.assertEqual(resp_all.status_code, 200)
        all_data = resp_all.get_json()
        self.assertGreater(len(all_data["candles"]), 50000)

        # Verify Sep 18, 2026 16:00 ET (blocked by quarterly contract roll) and 03:00 ET (below 55% prob threshold)
        # have explicit non-actionable explanations in /api/signals?include_non_actionable=1 and /api/bar
        resp_sig = self.client.get("/api/signals?tf=1h&horizon=8&prob_th=0.55&ratio_th=1.30&period=oos&include_non_actionable=1")
        self.assertEqual(resp_sig.status_code, 200)
        sig_payload = resp_sig.get_json()
        sep18_16 = [s for s in sig_payload["signals"] if s["timestamp"] == "2026-09-18 16:00"]
        self.assertEqual(len(sep18_16), 1)
        self.assertFalse(sep18_16[0]["is_actionable"])
        self.assertEqual(sep18_16[0]["status_code"], "BLOCKED_CONTRACT_ROLL")
        self.assertIn("Contract-Roll Seam Guard", sep18_16[0]["non_actionable_reason"])

        below_prob = [s for s in sig_payload["signals"] if s["status_code"] == "BELOW_PROB_THRESH"]
        self.assertGreater(len(below_prob), 10)
        self.assertFalse(below_prob[0]["is_actionable"])
        self.assertIn("below the 55% threshold", below_prob[0]["non_actionable_reason"])

        # Verify Sub-Model A (At-Extreme, mode=extreme) produces rich actionable signals across 15m, 1h, and 4h
        # and enters directly on pivot discovery (trigger_px == entry_px)
        for tf_check in ("15m", "1h", "4h"):
            resp_ex = self.client.get(f"/api/signals?tf={tf_check}&horizon=8&prob_th=0.55&ratio_th=1.20&period=oos&mode=extreme")
            self.assertEqual(resp_ex.status_code, 200)
            ex_data = resp_ex.get_json()
            self.assertGreater(ex_data["kpis"]["total_signals"], 40, f"Expected rich Sub-Model A signals on {tf_check}")

        resp_bar = self.client.get("/api/bar?tf=1h&horizon=8&prob_th=0.55&ratio_th=1.30&ts=2026-09-18%2016:00")
        self.assertEqual(resp_bar.status_code, 200)
        bar_sig = resp_bar.get_json()["signal"]
        self.assertFalse(bar_sig["is_actionable"])
        self.assertEqual(len(bar_sig["gates"]), 3)

        resp_chart = self.client.get("/api/chart?tf=1h&horizon=8&prob_th=0.55&ratio_th=1.30&bars=300")
        self.assertEqual(resp_chart.status_code, 200)
        c_data = resp_chart.get_json()
        self.assertGreater(len(c_data["candles"]), 100)
        self.assertGreater(len(c_data["ema20"]), 100)
        self.assertGreater(len(c_data["vwap"]), 100)

    def test_causal_feature_invariance_to_future_shocks(self):
        # Verify that mutating future bars (t > 120) has ZERO effect on causal features at t <= 120
        df_1h = pd.read_parquet(nqe.SCORED_CACHE_DIR / "nq_1h_scored.parquet").iloc[:200].copy()
        base_bars = pd.DataFrame({
            "open": df_1h["open"],
            "high": df_1h["high"],
            "low": df_1h["low"],
            "close": df_1h["close"],
            "volume": df_1h["volume"],
            "is_roll": False,
            "roll_mult": 1.0,
            "open_cont": df_1h["open"],
            "high_cont": df_1h["high"],
            "low_cont": df_1h["low"],
            "close_cont": df_1h["close"],
            "m1_count": 60.0,
            "m1_vol_delta_frac": df_1h["m1_vol_delta_frac"],
            "m1_vol_top25_frac": df_1h["m1_vol_top25_frac"],
            "m1_vol_bot25_frac": df_1h["m1_vol_bot25_frac"],
            "m1_up_bar_frac": 0.5,
            "m1_max_bar_range_frac": 0.2,
            "m1_high_time_frac": 0.5,
            "m1_low_time_frac": 0.5,
            "m1_late_vol_frac": 0.5,
        }, index=df_1h.index)

        f1 = nqe.compute_nq_intraday_features(base_bars, "1h")
        shocked = base_bars.copy()
        shocked.iloc[121:, shocked.columns.get_loc("high_cont")] *= 1.50
        shocked.iloc[121:, shocked.columns.get_loc("close_cont")] *= 1.50
        f2 = nqe.compute_nq_intraday_features(shocked, "1h")

        for col in ("rsi14_high", "inv_rsi75_overshoot_atr", "dist_rth_vwap_hi_atr", "dist_on_high_atr", "tod_rvol"):
            np.testing.assert_allclose(
                f1[col].iloc[:121].to_numpy(),
                f2[col].iloc[:121].to_numpy(),
                rtol=1e-5,
                atol=1e-5,
            )

    def test_zero_lookahead_excursions_and_chronological_2unit_bracket(self):
        """Verifies zero-lookahead post-Close[t] excursions, causal bad-print sanitization, and chronological 2-unit simulation."""
        df_1h = pd.read_parquet(nqe.SCORED_CACHE_DIR / "nq_1h_scored.parquet").iloc[:160].copy()
        base_bars = pd.DataFrame({
            "open": df_1h["open"],
            "high": df_1h["high"],
            "low": df_1h["low"],
            "close": df_1h["close"],
            "volume": df_1h["volume"],
            "is_roll": False,
            "roll_mult": 1.0,
            "open_cont": df_1h["open"],
            "high_cont": df_1h["high"],
            "low_cont": df_1h["low"],
            "close_cont": df_1h["close"],
            "m1_count": 60.0,
            "m1_vol_delta_frac": df_1h["m1_vol_delta_frac"],
            "m1_vol_top25_frac": df_1h["m1_vol_top25_frac"],
            "m1_vol_bot25_frac": df_1h["m1_vol_bot25_frac"],
            "m1_up_bar_frac": 0.5,
            "m1_max_bar_range_frac": 0.2,
            "m1_high_time_frac": 0.5,
            "m1_low_time_frac": 0.5,
            "m1_late_vol_frac": 0.5,
        }, index=df_1h.index)
        feats = nqe.compute_nq_intraday_features(base_bars, "1h")
        lab1 = nqe.label_nq_pivots_and_excursions(feats, pivot_left=5, pivot_right=5, rev_atr=1.5)

        # 1. Trigger prices must equal Close[t] (zero intrabar High[t]/Low[t] lookahead)
        np.testing.assert_allclose(lab1["trigger_short_px"].to_numpy(), lab1["close"].to_numpy(), rtol=1e-6)
        np.testing.assert_allclose(lab1["trigger_long_px"].to_numpy(), lab1["close"].to_numpy(), rtol=1e-6)

        # 2. Mutating High[t] and Low[t] at bar 50 while keeping Close[50] and atr14_cont[50] fixed
        # must NOT change bar 50's forward K-bar excursions (proving zero bar-t wick contamination)
        feats_mod = feats.copy()
        feats_mod.iloc[50, feats_mod.columns.get_loc("high_cont")] += 250.0
        feats_mod.iloc[50, feats_mod.columns.get_loc("low_cont")] -= 250.0
        lab2 = nqe.label_nq_pivots_and_excursions(feats_mod, pivot_left=5, pivot_right=5, rev_atr=1.5)
        for col in ("mfe_short_8b_atr", "mae_short_8b_atr", "net_short_8b_atr", "mfe_long_8b_atr", "mae_long_8b_atr", "net_long_8b_atr"):
            self.assertAlmostEqual(float(lab1[col].iloc[50]), float(lab2[col].iloc[50]), places=6)

        # 3. _sanitize_5m_bad_prints must be strictly causal (changing bar t+1 never alters bar t)
        df5_a = pd.DataFrame({
            "open": [70.0, 70.1, 70.2, 70.1, 70.0, 70.1],
            "high": [70.2, 70.3, 72.5, 70.3, 70.2, 70.3],
            "low": [69.9, 70.0, 70.0, 70.0, 69.9, 70.0],
            "close": [70.1, 70.2, 70.1, 70.0, 70.1, 70.2],
            "volume": [100.0] * 6,
        })
        df5_b = df5_a.copy()
        df5_b.iloc[3, df5_b.columns.get_loc("high")] = 75.0
        san_a = nqe._sanitize_5m_bad_prints(df5_a)
        san_b = nqe._sanitize_5m_bad_prints(df5_b)
        self.assertLess(float(san_a["high"].iloc[2]), 71.0)
        self.assertAlmostEqual(float(san_a["high"].iloc[2]), float(san_b["high"].iloc[2]), places=6)

        # 4. simulate_chronological_2unit_bracket must return 0 trades when filter matches 0 bars (no silent fallback)
        empty_sim = nqe.simulate_chronological_2unit_bracket(df_1h, p_th=0.999, r_th=9.99, k=8)
        self.assertEqual(empty_sim["trades"], 0)
        self.assertEqual(empty_sim["total_net_pts"], 0.0)

        # 5. compute_nq_trade_plan must not contain hardcoded fake win-rate strings
        plan = nqe.compute_nq_trade_plan(df_1h.iloc[-1], "15m", is_high_pivot=True, horizon_k=8, symbol="NQ=F")
        self.assertNotIn("~70% Win Rate", plan["expected_win_rate_str"])
        self.assertNotIn("~69% Win Rate", plan["expected_win_rate_str"])
        self.assertNotIn("~67% Win Rate", plan["expected_win_rate_str"])

        # 6. simulate_chronological_2unit_bracket must evaluate Step 0 (remaining 1m bars of developing bar i)
        idx_sim = pd.date_range("2025-01-02 09:30", periods=12, freq="15min", tz="America/New_York")
        df_dev = pd.DataFrame({
            "open": [20000.0] * 12,
            "high": [20050.0] + [20005.0] * 11,
            "low": [19990.0] + [19995.0] * 11,
            "close": [20000.0] * 12,
            "atr14_pts": [20.0] * 12,
            "is_cand_high": [True] + [False] * 11,
            "is_cand_low": [False] * 12,
            "prob_pivot_high": [0.70] + [0.10] * 11,
            "prob_pivot_high_extreme": [0.70] + [0.10] * 11,
            "prob_pivot_low": [0.10] * 12,
            "prob_pivot_low_extreme": [0.10] * 12,
            "pred_mfe_short_8b_atr": [2.0] * 12,
            "pred_mfe_q70_short_8b_atr": [2.8] * 12,
            "pred_mae_short_8b_atr": [0.8] * 12,
            "pred_ratio_short_8b": [2.5] * 12,
            "pred_mfe_long_8b_atr": [1.5] * 12,
            "pred_mfe_q70_long_8b_atr": [2.0] * 12,
            "pred_mae_long_8b_atr": [1.0] * 12,
            "pred_ratio_long_8b": [1.5] * 12,
            # Triggered intrabar at 20018.0 off developing high 20020.0 (0.10 ATR wick),
            # then remaining 1m bars of bar 0 spiked to 20050.0 (+1.60 ATR adverse -> hits 1.00 ATR stop in Step 0!)
            "trigger_short_px": [20018.0] + [20000.0] * 11,
            "dev_high_short_px": [20020.0] + [20005.0] * 11,
            "rem_high_short_px": [20050.0] + [np.nan] * 11,
            "rem_low_short_px": [19995.0] + [np.nan] * 11,
            "trig_min_short": [5] + [0] * 11,
        }, index=idx_sim)
        sim_step0_stop = nqe.simulate_chronological_2unit_bracket(df_dev, p_th=0.55, r_th=1.35, k=8)
        self.assertEqual(sim_step0_stop["trades"], 1)
        self.assertLess(sim_step0_stop["total_net_pts"], 0.0)
        self.assertAlmostEqual(sim_step0_stop["avg_r"], -1.0, places=2)

        # Now if remaining 1m bars of bar 0 stayed below 20021.0 and dropped to 19960.0 (+2.90 ATR favorable -> hits TP1 & TP2 in Step 0!)
        df_dev_win = df_dev.copy()
        df_dev_win.loc[idx_sim[0], "rem_high_short_px"] = 20021.0
        df_dev_win.loc[idx_sim[0], "rem_low_short_px"] = 19960.0
        sim_step0_win = nqe.simulate_chronological_2unit_bracket(df_dev_win, p_th=0.55, r_th=1.35, k=8)
        self.assertEqual(sim_step0_win["trades"], 1)
        self.assertGreater(sim_step0_win["total_net_pts"], 40.0)
        self.assertEqual(sim_step0_win["tp1_hit_rate_pct"], 100.0)

        plan_dev = nqe.compute_nq_trade_plan(df_dev_win.iloc[0], "15m", is_high_pivot=True, horizon_k=8, symbol="NQ=F")
        self.assertAlmostEqual(plan_dev["entry_price"], 20018.0, places=2)
        self.assertIn("+5m", plan_dev["entry_rule"])


if __name__ == "__main__":
    unittest.main()

