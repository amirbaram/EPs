# ML Evaluation Contract

## 1. Architectural Phases
The pipeline will not be monolithic. It is strictly partitioned into three independent stages:
1. **Dataset Construction:** Produces an immutable, versioned Parquet file + manifest.
2. **Model Evaluation:** Runs purged walk-forward folds, preprocessing, calibration, and diagnostics. Produces reports, no production models.
3. **Final Training/Export:** Trains the final champion model and saves the inference artifacts (preprocessor, calibrator, model, feature schema).

## 2. Dataset Construction Invariants (Stage 1)
- **Canonical Event Population:** Uses the exact 10-bar debounced episode generator from `setups.py` (not raw `_is_ep_event` calls).
- **RVOL Parity:** Must use the 50-day rolling mean (matching `setups.py`), explicitly deciding whether the event day is included (Yes, `setups.py` does not shift the volume window for the event-day calculation, wait, we must strictly mirror the production detector).
- **Missing Data:** Uncalculated or unavailable historical indicators (e.g., `sma200`) will be encoded as `np.nan`, not `0`.
- **Decision & Execution Timestamp:** 
  - Features are observed at the **Day-1 Close**.
  - Hypothetical execution is at the **Day-1 Close** (matching the historical `serve_ep.py` backtester), treating Day-1 Close as the executable entry price.
- **Ambiguous Bar Policy:** If a single day's OHLC touches both the target and the stop, we apply the conservative convention: the stop is assumed to be hit first. This acts as an absolute lower bound on performance.

## 3. Right-Censoring & Outcome States
We will not immediately collapse forward paths into binary labels. The dataset will record rich outcome states based on a strict **250-session horizon**:
- `target`: Hit target (+50%) before stop.
- `stop`: Hit Day-1-low stop before target.
- `full_horizon_timeout`: Reached 250 sessions without hitting target or stop.
- `right_censored`: Data ends before 250 sessions (e.g., 2026 events) and neither target nor stop was hit.
- `ambiguous_same_bar`: Both target and stop hit on the same day (counted as `stop` for binary targets, but flagged).
- `missing_history` / `delisted_or_unavailable`.

*Note on Survivorship Bias:* Until Phase 2 data acquisition retrieves delisted symbols, all outputs must be explicitly labeled: *"Conditional probabilities among symbols surviving into the current cached universe."*

## 4. Evaluation Contract (Stage 2)
- **Objective:** Establish walk-forward discrimination, probability calibration, and economic lift independently. "Calibrated AUC" is discarded as a mixed metric.
- **Walk-Forward Folds:** 
  - Split strictly on whole dates. 
  - Purge rule: `label_end_date < test_start`.
  - Expanding folds (e.g., Train $\le$ 2020 $\rightarrow$ Calibrate 2021 $\rightarrow$ Test 2022).
- **Three-Population Calibration:** Model fitting (Train), Hyperparameter/Calibration (Validation), and Final Measurement (Test) will use three strictly segregated datasets.
- **Baseline:** Regularized Logistic Regression (requires strict training-only preprocessing: median imputation, missingness flags, standardization, one-hot encoding).
- **Challenger:** Unweighted XGBoost / Histogram Gradient Boosting.
- **Metrics:** Brier score, log loss, calibration curves, PR-AUC, top-decile lift, and net Expected Value (R-multiple).
