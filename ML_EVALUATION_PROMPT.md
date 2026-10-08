# Deep Think AI Evaluation: Episodic Pivot ML Architecture

## Instructions for Gemini Advanced / Deep Think
Please evaluate our Machine Learning architecture for Episodic Pivot (EP) stock trading setups. You will be reviewing the codebase via our GitHub repository. Your goal is to identify mathematical flaws, data leakage, poor target framing, and suggest structural improvements to our dataset generation and XGBoost pipelines.

---

### 1. Context: What is an Episodic Pivot (EP)?
An Episodic Pivot is a massive, unexpected gap-up in a stock's price driven by a fundamental catalyst (usually earnings). Institutions use this liquidity event to begin accumulating massive positions, which can drive the stock up +100% to +300% over the next 3 to 12 months (Post-Earnings Announcement Drift).

### 2. What We Have Learned from 10 Years of Historical Testing
Through rigorous backtesting of our uncurated universe (23,000+ EPs), we have discovered:
1. **The Day 1 Base Setup:** Buying blindly at the Day 1 Close and placing a wide structural stop-loss strictly at the Day 1 Low yields a highly consistent, positive Expectancy (+2.66 R) with a win rate of ~40-52% for top-tier setups.
2. **The Day 2 Asymmetry (DEP / Bonde DRE):** For specific "Delayed Reaction" setups, waiting for the stock to break the Day 2 High and placing an ultra-tight stop at the Day 2 Low produces an absurdly massive right-tail Expectancy (+121.13 R). However, the win rate plummets to ~18% due to constant whipsawing, making it psychologically brutal to trade.
3. **The Limit of Human Heuristics:** We originally used hardcoded heuristic rule engines to grade setups (e.g., adding +10% probability if it held the 5-day moving average). These failed to capture complex, non-linear interactions between Base Tightness, Market Theme state, and Volume Digestion, prompting our shift to Machine Learning.

### 3. What We Are Trying to Accomplish with ML
We want our XGBoost classification models to dynamically evaluate a developing EP chart and output the exact, highly calibrated real-world probability that the stock will hit progressive scale-out targets (+50%, +100%, +150%, +200% from the Day 1 Close price) before it hits the structural stop-loss (the Day 1 Low). 

We currently run two competing architectures:
- **V1 (Fixed Day-5 Snapshot):** Evaluates the chart strictly as of the Close of Day 5.
- **V2 (Rolling Timeline):** A panel-data model that re-evaluates the chart every single day (Days 1 to 20), calculating dynamic base structure features (e.g., `feature_tightness_since_ep`) as the consolidation develops.

### 4. Key Files to Evaluate
Please read and evaluate the following core files in the repository:
1. `build_ml_dataset.py` (V1 Data Builder): Look closely at the target definition and censorship logic (`label_reach_T is event-level`). We recently discovered that excluding rows that already hit the target by Day 5 warped the training populations and broke monotonicity across targets.
2. `build_rolling_dataset.py` (V2 Data Builder): Look at the forward-looking target loop evaluating if price hits the target before hitting the Day 1 Low stop.
3. `ep_ml_engine.py`: Review `compute_rolling_features()`. Are we introducing forward-looking data leakage when calculating `ema10/20/50` or `tightness_since_ep`?
4. `train_amir_spec.py` & `train_rolling_xgboost.py`: Review the XGBoost pipelines. We recently discovered that utilizing SMOTE / `scale_pos_weight` artificially inflated the `.predict_proba()` outputs. We have removed it, but how should we properly handle the massive class imbalance (only ~2-5% of stocks reach the +200% target) while ensuring the probability output remains strictly calibrated to real-world base rates?

### 5. Your Deliverables
1. **Data Leakage Audit:** Point out any look-ahead bias in our feature generation.
2. **Target Definition Audit:** Criticize our handling of right-censored data and stop-loss logic.
3. **Probability Calibration:** Suggest specific mathematical implementations (e.g., Platt Scaling, Isotonic Regression, or alternative XGBoost objective functions) to ensure our low-base-rate minority classes predict accurate, un-inflated probabilities.
