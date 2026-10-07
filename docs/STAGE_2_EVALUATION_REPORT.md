# Stage 2: ML Model Evaluation Report

## 1. Overview
Stage 2 evaluates the walk-forward predictability of the Episodic Pivot context dataset.
Following the evaluation contract, we test the probability of an EP achieving a +50% move before hitting the EP-day low stop. 

## 2. Experimental Design
- **Target:** `label_reach_50` (Binary: 1 if hit, 0 if stop/timeout)
- **Purge Rule:** The training set explicitly drops any daily feature row where the event's *final outcome date* (`label_reach_50_end_date`) bleeds into the validation or test split.
- **Data Splits:**
  - **Train:** Events resolving before 2023-01-01
  - **Validation:** Events resolving between 2023-01-01 and 2024-01-01
  - **Test (Blind OOS):** Events starting after 2024-01-01

## 3. Models Evaluated
1. **Baseline:** Regularized Logistic Regression (L2, C=0.1)
   - Preprocessing: Median imputation, Standardization, One-Hot Encoding.
2. **Challenger:** Unweighted XGBoost (`max_depth=4`, `lr=0.05`, `n_estimators=200`)
   - Preprocessing: Native missing value handling, Dense One-Hot Encoding for categoricals.

## 4. Evaluation Metrics
We measure discrimination, calibration, and Expected Value (EV). EV is measured in "R-multiples".
- **Global EV:** The average R-multiple of all trades in the test set.
- **Top 10% EV:** The average R-multiple of trades specifically identified by the model in its top decile of confidence.


## 5. Results (Blind OOS: 2024–2026)

### Data Funnel
* **Total Rows (All Events & Days):** 467,409
* **Unresolved Rows:** 265,946 (The model only sees days where the outcome is still unknown)
* **Final Rows (Uncensored):** 256,548
* **Base Rate (Global probability of hitting +50%):** 35.51%
* **Test Set Size (OOS rows):** 107,331

### Baseline (Logistic Regression)
* **Brier Score:** 0.2299
* **Log Loss:** 0.7155
* **PR-AUC:** 0.6139
* **Global Expected Value:** 1.63 R
* **Top 10% Expected Value:** 3.30 R
* **Top 10% Win Rate:** 74.58%

### Challenger (Unweighted XGBoost)
* **Brier Score:** 0.2043 *(Lower is better)*
* **Log Loss:** 0.5997 *(Lower is better)*
* **PR-AUC:** 0.6529 *(Higher is better)*
* **Global Expected Value:** 1.63 R
* **Top 10% Expected Value:** **3.58 R**
* **Top 10% Win Rate:** **76.11%**

## 6. Verdict
The predictive features injected during Stage 1 have proven exceptionally robust out-of-sample.
1. The **XGBoost Challenger comfortably outperforms the Logistic baseline** across all calibration and discrimination metrics (Log Loss, Brier, PR-AUC).
2. The model exhibits massive economic lift in the top decile. While a random day of an unresolved EP has a 35.51% chance of hitting +50%, **if the XGBoost model ranks it in the top 10% of confidence, the win rate skyrockets to 76.11%**, producing an expected value of **3.58 R** per trade.

The pipeline is ready for final production training and integration into the dashboard.
