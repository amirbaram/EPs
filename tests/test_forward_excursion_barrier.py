"""Regression tests for Forward Excursion Regressors and Option B Barrier Model."""
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import ep_ml_engine

class TestForwardExcursionBarrier(unittest.TestCase):
    def test_model_bundle_components(self):
        eng = ep_ml_engine.EPMLEngine()
        self.assertIsNotNone(eng.forward_excursion_model, "Forward excursion model bundle must exist")
        self.assertIn("mfe_model", eng.forward_excursion_model)
        self.assertIn("mae_model", eng.forward_excursion_model)
        self.assertIn("seq_model", eng.forward_excursion_model)
        self.assertIn("barrier_model", eng.forward_excursion_model)
        self.assertIn("feature_cols", eng.forward_excursion_model)

    def test_prediction_output_fields(self):
        eng = ep_ml_engine.EPMLEngine()
        dummy_features = {c: 0.0 for c in eng.forward_excursion_model["feature_cols"]}
        dummy_features["feature_days_since_ep"] = 5
        dummy_features["feature_dist_to_d1_low"] = 0.08
        dummy_features["feature_gap_pct"] = 12.0
        dummy_features["feature_rvol"] = 3.5

        preds = eng.predict_forward_excursion(dummy_features)
        self.assertIn("pred_mfe_20d", preds)
        self.assertIn("pred_mae_20d", preds)
        self.assertIn("prob_mfe_first", preds)
        self.assertIn("prob_barrier_2r", preds)

        self.assertIsInstance(preds["pred_mfe_20d"], float)
        self.assertIsInstance(preds["pred_mae_20d"], float)
        self.assertIsInstance(preds["prob_mfe_first"], float)
        self.assertIsInstance(preds["prob_barrier_2r"], float)

        self.assertGreaterEqual(preds["prob_barrier_2r"], 0.0)
        self.assertLessEqual(preds["prob_barrier_2r"], 1.0)

if __name__ == "__main__":
    unittest.main()
