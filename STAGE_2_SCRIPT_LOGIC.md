# Stage 2 Evaluation Script Architecture

The script `scripts/evaluate_stage2.py` implements the contractual Stage 2 evaluation for predicting whether an Episodic Pivot reaches the +50% target before the stop limit.

## 1. Purged Walk-Forward Splits
To ensure absolute avoidance of future leakage, the walk-forward split uses the newly implemented `label_reach_50_end_date`.

```python
train_mask = (df["as_of_date"] < val_start) & (df["end_date"] < val_start)
```
* **Why this is critical:** An event occurring in December 2022 might not hit its target/stop until February 2023. If we allow that event into the Training set, we are letting the model "see" the 2023 outcome while predicting on the 2023 Validation set. The `train_mask` purges any event whose outcome is not fully resolved *before* the start of the validation period.

## 2. Unresolved Panel Filtering
```python
df = df[df["feature_chk_reached_50"] == 0].copy()
```
The model is predicting whether an event *will* hit the target. If it already hit the target yesterday, the probability is mechanically 100%. We correctly filter the dataset to only rows where the outcome is still unknown as of the observation date.

## 3. Data Preprocessing & Model Selection
In strict compliance with the quant reviewer's instructions:
1. **Baseline: Regularized Logistic Regression (L2 Penalty)**
   - Preprocessed with Median Imputation, Standardization, and One-Hot Encoding for categorical context.
2. **Challenger: XGBoost (Unweighted)**
   - Evaluated as a strict out-of-the-box challenger without upweighting classes.

## 4. Expected Value (R-Multiple) Optimization
Rather than evaluating solely on Log Loss or PR-AUC, the script calculates the economic lift (Expected Value).
- **R-multiple mapping:** `+R` (Target Price - Entry) / (Entry - Stop) for successes; `-1.0` for failures.
- **Top 10% Decile Extraction:** Isolates the top decile of confidence probabilities and measures the average realized P&L of those specific executions, compared against the global average.
