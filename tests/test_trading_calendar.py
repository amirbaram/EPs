"""Unit and regression tests for trading_calendar.py."""

import unittest
import pandas as pd
import trading_calendar


class TestTradingCalendar(unittest.TestCase):
    def test_quad_witching_dates(self):
        # 3rd Friday of March, June, September, December
        self.assertTrue(trading_calendar.is_quad_witching("2024-03-15"))
        self.assertTrue(trading_calendar.is_quad_witching("2024-06-21"))
        self.assertTrue(trading_calendar.is_quad_witching("2024-09-20"))
        self.assertTrue(trading_calendar.is_quad_witching("2024-12-20"))
        self.assertFalse(trading_calendar.is_quad_witching("2024-01-19"))  # Monthly OpEx, but not Quad Witching

    def test_monthly_opex_and_holiday_thursday_shift(self):
        # Normal 3rd Friday
        self.assertTrue(trading_calendar.is_monthly_opex("2024-01-19"))
        self.assertTrue(trading_calendar.is_monthly_opex("2024-02-16"))

        # Good Friday in 2022 was April 15 (market closed). OpEx shifted to Thursday April 14.
        self.assertTrue(trading_calendar.is_monthly_opex("2022-04-14"))
        self.assertFalse(trading_calendar.is_monthly_opex("2022-04-15"))

        # Good Friday in 2025 was April 18 (market closed). OpEx shifted to Thursday April 17.
        self.assertTrue(trading_calendar.is_monthly_opex("2025-04-17"))
        self.assertFalse(trading_calendar.is_monthly_opex("2025-04-18"))

    def test_russell_reconstitution(self):
        # 4th Friday of June
        self.assertTrue(trading_calendar.is_russell_reconstitution("2021-06-25"))
        self.assertTrue(trading_calendar.is_russell_reconstitution("2022-06-24"))
        self.assertTrue(trading_calendar.is_russell_reconstitution("2023-06-23"))
        self.assertTrue(trading_calendar.is_russell_reconstitution("2024-06-28"))
        self.assertTrue(trading_calendar.is_russell_reconstitution("2025-06-27"))
        self.assertTrue(trading_calendar.is_russell_reconstitution("2026-06-26"))

        # Not Russell
        self.assertFalse(trading_calendar.is_russell_reconstitution("2024-06-21"))  # 3rd Friday (Quad Witching)

    def test_quarter_and_month_ends(self):
        # Quarter ends
        self.assertTrue(trading_calendar.is_quarter_end("2024-03-28"))  # Good Friday was March 29, 2024! So March 28 was quarter end
        self.assertTrue(trading_calendar.is_quarter_end("2024-06-28"))
        self.assertTrue(trading_calendar.is_quarter_end("2024-09-30"))
        self.assertTrue(trading_calendar.is_quarter_end("2024-12-31"))

        # Month ends
        self.assertTrue(trading_calendar.is_month_end("2024-01-31"))
        self.assertTrue(trading_calendar.is_month_end("2024-02-29"))

    def test_get_calendar_anomaly_type(self):
        desc = trading_calendar.get_calendar_anomaly_type("2024-06-21")
        self.assertEqual(desc, "Quad Witching")

        # 2024-06-28 was both Russell Reconstitution and Quarter-End!
        desc_dual = trading_calendar.get_calendar_anomaly_type("2024-06-28")
        self.assertIn("Russell Reconstitution", desc_dual)
        self.assertIn("Quarter-End", desc_dual)

    def test_annotate_dataframe(self):
        df = pd.DataFrame({"date": ["2024-01-19", "2024-06-21", "2024-06-28", "2024-05-10"]})
        annotated = trading_calendar.annotate_calendar_events(df)
        self.assertTrue(annotated.loc[0, "is_opex"])
        self.assertTrue(annotated.loc[1, "is_quad_witching"])
        self.assertTrue(annotated.loc[2, "is_russell_recon"])
        self.assertFalse(annotated.loc[3, "is_mechanical_volume_date"])


    def test_serve_ep_calendar_filters_and_api_contract(self):
        import serve_ep
        client = serve_ep.app.test_client()

        # 1. Clean filter excludes mechanical volume anomalies (unless gap >= 8%)
        resp_all = client.get("/api/events?preset=all")
        self.assertEqual(resp_all.status_code, 200)
        n_all = len(resp_all.get_json()["events"])

        resp_clean = client.get("/api/events?preset=all&calendar=clean")
        self.assertEqual(resp_clean.status_code, 200)
        ev_clean = resp_clean.get_json()["events"]
        n_clean = len(ev_clean)
        self.assertLess(n_clean, n_all, "Clean filter should exclude mechanical anomaly setups")

        for e in ev_clean:
            if e.get("is_mechanical_volume_date"):
                self.assertGreaterEqual(float(e.get("gap_pct", 0) or 0), 8.0, "Mechanical dates in clean set must have catalyst gap >= 8%")

        # 2. Quad witching filter
        resp_quad = client.get("/api/events?preset=all&calendar=quad")
        self.assertEqual(resp_quad.status_code, 200)
        ev_quad = resp_quad.get_json()["events"]
        self.assertGreater(len(ev_quad), 0)
        for e in ev_quad:
            self.assertTrue(e.get("is_quad_witching"))

    def test_ticker_p_calendar_disqualification_on_quad_witching(self):
        import serve_ep
        client = serve_ep.app.test_client()

        resp = client.get("/api/chart?symbol=P&date=2026-09-18")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        # Sept 18, 2026 was Quad Witching
        self.assertTrue(data.get("is_mechanical_volume_date"))
        self.assertIn("Quad Witching", data.get("calendar_anomaly", ""))

        # P 2026-09-18 had a 1.3% gap, so it should be flagged as disqualified due to Volume Anomaly Date
        t_hist = [h for h in data.get("ticker_history", []) if h["date"] == "2026-09-18"]
        self.assertGreater(len(t_hist), 0)
        p_hist = t_hist[0]
        self.assertFalse(p_hist["is_qualified"])
        self.assertTrue(any("Volume Anomaly Date" in r for r in p_hist["disqualifications"]))


if __name__ == "__main__":
    unittest.main()

