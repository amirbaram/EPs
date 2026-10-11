"""Regression tests for on-chart drawings disk persistence, trade marking defaults, and EP qualification accuracy."""
import json
import unittest
from pathlib import Path

from serve_ep import app, ANNOTATIONS_DIR


class TestAnnotationsAndEPQualification(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_trade_markings_off_by_default(self):
        """Verifies that Trade 1, Trade 2, and Trade 3 markings checkboxes are unchecked by default in HTML."""
        res = self.client.get("/")
        html = res.data.decode("utf-8")
        
        self.assertIn('id="chk_mark_t1"', html)
        self.assertIn('id="chk_mark_t2"', html)
        self.assertIn('id="chk_mark_t3"', html)
        
        # None of them should have checked attribute
        self.assertNotIn('id="chk_mark_t1" checked', html)
        self.assertNotIn('id="chk_mark_t2" checked', html)
        self.assertNotIn('id="chk_mark_t3" checked', html)

    def test_clean_json_boolean_integrity(self):
        """Verifies clean_json preserves genuine bool types and does not convert True/False to 1/0."""
        from serve_ep import get_df
        # Test clean_json directly via an endpoint that uses clean_json
        res = self.client.get("/api/chart?symbol=PENG&date=2026-07-08")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        
        # Find an event with disqualifications
        all_evs = data.get("all_ticker_events", [])
        disqualified = [e for e in all_evs if e.get("disqualifications")]
        qualified = [e for e in all_evs if not e.get("disqualifications")]
        
        self.assertTrue(len(disqualified) > 0)
        self.assertTrue(len(qualified) > 0)
        
        # Verify strict boolean type in JSON
        for d in disqualified:
            self.assertIs(d["is_qualified"], False)
        for q in qualified:
            self.assertIs(q["is_qualified"], True)

    def test_annotations_disk_persistence_roundtrip(self):
        """Verifies on-chart user drawings are saved to disk under data/annotations/{sym}.json and reloaded."""
        test_sym = "TESTANNOT"
        target_file = ANNOTATIONS_DIR / f"{test_sym}.json"
        
        if target_file.exists():
            target_file.unlink()

        try:
            sample_drawings = [
                {
                    "type": "line",
                    "t1": "2026-01-05",
                    "p1": 42.50,
                    "t2": "2026-01-20",
                    "p2": 58.00,
                    "x1_rel": 0.2,
                    "y1_rel": 0.4,
                    "x2_rel": 0.6,
                    "y2_rel": 0.1
                },
                {
                    "type": "text",
                    "time": "2026-01-20",
                    "price": 58.00,
                    "text": "Institutional Breakout Shelf"
                }
            ]

            # 1. Save drawings via POST
            post_res = self.client.post("/api/annotations", data=json.dumps({
                "symbol": test_sym,
                "drawings": sample_drawings
            }), content_type="application/json")
            self.assertEqual(post_res.status_code, 200)
            self.assertTrue(post_res.get_json().get("success"))

            # 2. Verify file exists on disk
            self.assertTrue(target_file.exists(), f"Expected annotations file at {target_file}")
            with open(target_file, "r", encoding="utf-8") as f:
                disk_data = json.load(f)
            self.assertEqual(disk_data["symbol"], test_sym)
            self.assertEqual(len(disk_data["drawings"]), 2)

            # 3. Reload drawings via GET
            get_res = self.client.get(f"/api/annotations?symbol={test_sym}")
            self.assertEqual(get_res.status_code, 200)
            get_data = get_res.get_json()
            self.assertEqual(get_data["symbol"], test_sym)
            self.assertEqual(len(get_data["drawings"]), 2)
            self.assertEqual(get_data["drawings"][1]["text"], "Institutional Breakout Shelf")

            # 4. Clear drawings (POST empty array)
            clear_res = self.client.post("/api/annotations", data=json.dumps({
                "symbol": test_sym,
                "drawings": []
            }), content_type="application/json")
            self.assertEqual(clear_res.status_code, 200)
            self.assertFalse(target_file.exists(), "Empty drawings should remove file from disk")

            # 5. GET after clear returns empty array
            get_after_clear = self.client.get(f"/api/annotations?symbol={test_sym}")
            self.assertEqual(get_after_clear.status_code, 200)
            self.assertEqual(get_after_clear.get_json()["drawings"], [])

        finally:
            if target_file.exists():
                target_file.unlink()


if __name__ == "__main__":
    unittest.main()
