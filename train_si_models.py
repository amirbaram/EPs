"""
train_si_models.py — Silver Futures (SI=F / SIL=F) Intraday ML Pivot & Excursion Model Trainer
=============================================================================================
Trains 15m, 1h, and 4h Dual-Path Structural Pivot High/Low Classifiers,
Forward K={4, 8, 12} MAE/MFE Regressors, and Asymmetry Classifiers on 5-year 1-minute Silver data,
evaluating strictly out-of-sample on the 2-year 2024–2026 test set.
"""

from __future__ import annotations

import sys
import time
import nq_ml_engine


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("Starting Silver Futures (SI=F) 15m / 1h / 4h ML Training Pipeline")
    print("=" * 70)
    report = nq_ml_engine.train_and_evaluate_all_si_models()
    print("=" * 70)
    print(f"✅ All Silver Futures (SI=F) models trained & evaluated in {time.time() - t0:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
