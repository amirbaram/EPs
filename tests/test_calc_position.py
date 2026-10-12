import unittest
import math
from calc_position import calculate_position

class TestPositionCalculator(unittest.TestCase):
    def test_long_position_sizing(self):
        # Entry 100, Stop 95, Max loss 500 -> 500 / 5 = 100 shares
        res = calculate_position(100.0, 95.0, 500.0)
        self.assertEqual(res["direction"], "LONG")
        self.assertEqual(res["allowed_shares"], 100)
        self.assertAlmostEqual(res["risk_per_share"], 5.0)
        self.assertAlmostEqual(res["total_position_value"], 10000.0)
        self.assertAlmostEqual(res["actual_risk"], 500.0)
        self.assertAlmostEqual(res["r_targets"]["1R"][0], 105.0)
        self.assertAlmostEqual(res["r_targets"]["2R"][0], 110.0)
        self.assertAlmostEqual(res["r_targets"]["3R"][0], 115.0)

    def test_short_position_sizing(self):
        # Entry 50, Stop 52.50, Max loss 250 -> 250 / 2.5 = 100 shares
        res = calculate_position(50.0, 52.50, 250.0)
        self.assertEqual(res["direction"], "SHORT")
        self.assertEqual(res["allowed_shares"], 100)
        self.assertAlmostEqual(res["risk_per_share"], 2.5)
        self.assertAlmostEqual(res["total_position_value"], 5000.0)
        self.assertAlmostEqual(res["actual_risk"], 250.0)
        self.assertAlmostEqual(res["r_targets"]["1R"][0], 47.50)

    def test_rounding_down(self):
        # Entry 10.0, Stop 9.0 -> $1 risk. Max loss 75.50 -> 75 shares
        res = calculate_position(10.0, 9.0, 75.50)
        self.assertEqual(res["allowed_shares"], 75)
        self.assertAlmostEqual(res["actual_risk"], 75.0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            calculate_position(100.0, 100.0, 500.0)
        with self.assertRaises(ValueError):
            calculate_position(0.0, 95.0, 500.0)
        with self.assertRaises(ValueError):
            calculate_position(100.0, -5.0, 500.0)
        with self.assertRaises(ValueError):
            calculate_position(100.0, 95.0, -100.0)

if __name__ == '__main__':
    unittest.main()
