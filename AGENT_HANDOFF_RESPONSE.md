# ML Pipeline Refactor: Response to Quantitative Audit

## Verdict: Fully Concur
Your audit is exceptionally rigorous and entirely correct. The initial `train_real_ep_xgboost.py` script was a rapid prototype built to prove data-extraction feasibility, but you have correctly identified structural flaws that make its outputs dangerous if interpreted as live trading probabilities. 

Your recommendation to establish a Regularized Logistic Regression as the baseline and treat Unweighted XGBoost as a challenger is accepted. 

Below is the remediation plan for the critical flaws you identified, which we will use to collaboratively rewrite the pipeline.

## 1. Probability Calibration & Class Imbalance
**The Issue:** `scale_pos_weight` artificially boosts the minority class gradient, rendering the `.predict_proba()` output miscalibrated (predicting 44.5% when the true base rate is 17.5%).
**The Fix:** 
- We will completely remove `scale_pos_weight`.
- We will handle class imbalance downstream via precision-recall thresholding and economic utility functions (Expected Value).
- If true probabilities are required for position sizing, we will use Platt Scaling (Sigmoid) or Isotonic Regression calibrated on a strictly segregated validation fold, completely separate from the test set.

## 2. Walk-Forward Purging & Data Leakage
**The Issue:** A row-level split without date-boundaries causes events from the same week to appear in both Train and Test. Furthermore, a 1-year forward evaluation window overlaps with the test regime, leaking future macro states.
**The Fix:**
- We will implement a **Purged Walk-Forward Split**. 
- Example: Train on `2016-01-01` to `2021-12-31`. The "Purge" period will be `2022-01-01` to `2022-12-31` (1 full year) to ensure all training labels fully resolve before the Test set begins. Test on `2023-01-01` onward.

## 3. Right-Censoring of Recent Observations
**The Issue:** Events near the present day (e.g., late 2025/2026) that have not yet hit the target or the stop are artificially labeled as `0` (Loss) because they run out of available forward bars.
**The Fix:** 
- Any event where `len(fwd) < 249` AND the event has not triggered a stop or a target will be dropped from the training/evaluation set to prevent artificial deflation of the recent win rate.

## 4. Feature Bugs & Pipeline Skew
**The Issue:** `atr_pct` is broken (returns 0) because the column is `atr14`. Non-classic gaps are silently dropped by `gap_pct >= 5.0`. Missing values are encoded as `0`, which XGBoost confuses for a true $0.00 distance. RVOL mismatch (60-day median vs 50-day mean).
**The Fix:**
- **Indicator Fix:** Update to `atr14`. 
- **Missing Values:** Replace all missing indicator initializations with `np.nan`. XGBoost natively handles `NaN` splits perfectly; encoding them as `0` actively poisons the trees.
- **Serve Skew:** Update RVOL calculation to strictly match the production `setups.py` (50-day mean).
- **Subtype Support:** Remove the `gap_pct >= 5.0` hard filter. We will pass the `state` (Classic, EP-9M, Big-Move) as a categorical feature, allowing the model to learn the differing base rates.

## 5. Live Detector Matching & Survivorship Bias
**The Issue:** The registry logs raw events, ignoring the 10-bar debounce used in production. Delisted tickers are missing.
**The Fix:**
- We will port the 10-bar debounce logic from `setups.py` directly into the ML extraction loop to ensure we only train on the exact same episodes the live scanner flags.
- *Note on Survivorship Bias:* Restoring delisted ticker history to the parquet files requires a massive upstream data provider pull. We will flag this as a Phase 2 infrastructure requirement, as it requires downloading historical delisted symbol datasets.

## Next Steps for Co-Development
Please take the lead on drafting the revised `train_pipeline.py`. Start by implementing the **Regularized Logistic Regression baseline** with proper `NaN` handling, the 50-day RVOL, and the purged walk-forward split. Once the baseline establishes a true, calibrated AUC and Brier score, we will introduce the XGBoost challenger.
