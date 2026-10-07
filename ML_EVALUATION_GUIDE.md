# ML Pipeline Evaluation Guide

## 1. Core Architecture
The `train_real_ep_xgboost.py` script is a production-ready machine learning pipeline designed to train an XGBoost ranking engine on 10 years of historical Episodic Pivots. 

It executes in two distinct stages:
1. **Feature Extraction (O(N) Optimization):** It parses the database of historical EPs, loads their raw price bars, calculates deep technical indicators, and determines the forward outcome.
2. **Model Training (Out-Of-Sample):** It splits the data chronologically, balances the classes, trains the `XGBClassifier`, and evaluates the results on unseen future data.

## 2. Efficiency & Checkpointing Optimizations (Deep Analysis)
A naive loop over 12,000 EPs would load a stock's 10-year history and calculate its 200-day moving average *for every single EP event*, leading to $O(N)$ bar-loading overhead. 
* **The Optimization:** The script groups the raw EP registry by `symbol` (`eps.groupby('symbol')`). It loads `datastore.load_bars(sym)` and computes indicators **exactly once per ticker**. It then iterates through all EP dates for that specific ticker in memory. This reduces I/O and computation time by roughly 80%.
* **The Checkpointing:** Feature extraction can take a while. The script incrementally flushes rows to `data/ml_features_checkpoint.csv` via `f_out.flush()`. If the script hits an error or is manually interrupted, the next run reads the checkpoint, identifies the `processed_symbols`, and resumes exactly where it left off.

## 3. The Feature Engineering
The model is currently designed to learn from structural micro-features:
- `gap_pct`: The exact percentage of the overnight gap.
- `rvol`: Relative volume based on a trailing 60-day median (the institutional footprint).
- `close_pos`: The position of the close relative to the daily range (measures intraday absorption).
- `dist_20`, `dist_50`, `dist_200`: The percentage distance between the closing price and the major moving averages. This allows the model to detect over-extended exhaustion gaps versus tightly coiled breakouts.
- `atr_pct`: The percentage volatility, helping the model normalize expected moves across different asset classes.

## 4. The Target Label
The model trains on a binary classification target: `target_hit`.
- **Label 1 (Win):** The stock hit a +50% gain from the Day 1 close *before* it ever dropped below the Day 1 Low.
- **Label 0 (Loss):** The stock hit the Day 1 Low stop before reaching a +50% gain.

## 5. Overcoming Class Imbalance
Because high-quality Pinnacle Elite setups that run +50% are statistically rare (a 15-20% base rate), a naive ML model will just predict "0" for every trade to achieve 85% accuracy.
* **The Fix:** The script dynamically calculates the ratio of failures to successes (`num_neg / num_pos`) and passes it into the XGBoost `scale_pos_weight` parameter. This forces the algorithm to mathematically care much more about identifying the rare winners, rather than just maximizing raw accuracy by spamming "0".

## 6. Execution Instructions & Absolute Paths for Future Agents
If you are an agent tasked with evaluating or running this code, you MUST use the following absolute paths:

- **The Main Script:** `/Users/amirbaram/Downloads/scan/train_real_ep_xgboost.py`
- **The Python Environment:** `/Users/amirbaram/Downloads/scan/.venv/bin/python` (Always use this virtual environment when executing).
- **The Raw EP Registry (Input):** `/Users/amirbaram/Downloads/scan/data/setup_registry/episodic_pivot.parquet`
- **The Checkpoint File (Output):** `/Users/amirbaram/Downloads/scan/data/ml_features_checkpoint.csv`

**To run the pipeline:**
```bash
cd /Users/amirbaram/Downloads/scan
.venv/bin/python train_real_ep_xgboost.py
```
