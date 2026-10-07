# ML Model Development Plan: The Next Evolution

## 1. Viability & Expected Results vs. Current System

**The Current System (Empirical Rule-Based Matrix):**
Our current system is a rigid lookup table. We found historical "sweet spots" (e.g., RVOL > 5.0x, ClosePos > 0.65) by looking at the entire 10-year dataset at once. 
* **The Flaw:** Hard boundaries. An EP with 4.9x RVOL gets zero credit, while one with 5.0x gets a massive upgrade. Furthermore, because we looked at the *entire* decade of data to create the rules, it suffers from **Lookahead Bias / In-Sample Overfitting**. The results are structurally optimistic.

**The True ML Model (XGBoost / Random Forest):**
Training a model on the first 7 years (2016–2022) and testing it blindly on the last 3 years (2023–2026) is the gold standard of quantitative finance.
* **The Viability:** Extremely viable. With 6,501 historical EP events, we have a robust, statistically significant dataset perfectly sized for gradient-boosting algorithms (XGBoost, LightGBM). Tabular ML models excel at finding non-linear relationships in financial data.
* **Expected Results:** The true ML model will likely output a *lower* theoretical P&L on paper than our current in-sample backtest, but it will be **infinitely more robust and realistic for future trading**. It will smooth out hard boundaries (treating 4.9x RVOL and 5.0x almost identically) and automatically weigh combinations of factors we can't manually calculate (e.g., "High RVOL is great for Tech, but actually bearish for small-cap Biotech if the gap is >40%").

---

## 2. Comprehensive Feature Engineering (What the Model Needs to Learn)

To predict if an EP will double or collapse, the model must ingest a highly granular matrix of features that captures the micro-structure of the stock and the macro-structure of the market.

### A. Pre-Catalyst Stock History (The Coil)
* **Market Cap & Float / Short Interest:** Nimble small-cap squeeze vs. heavy mega-cap drift.
* **Price Architecture:** Distance from 52-Week High/Low, All-Time Highs.
* **Historical DNA:** Has this specific stock successfully triggered a massive EP in the past? (Repeat offenders often run again).
* **6-Month Neglect Score:** Has the stock been dead/consolidating for months? (High explosive potential).
* **Pre-EP ATR (Volatility):** How wild is the stock normally?

### B. Day 1 Catalyst Mechanics (The Ignition & Our Insights)
* **Gap % & Intraday Expansion %:** How big was the jump, and how much did it push intraday?
* **RVOL & Absolute Dollar Volume:** The true footprint of institutional liquidity.
* **Close Position (ClosePos):** Where did it close relative to the daily range?
* **Expectancy Boosters (Binary Flags):** The model will be explicitly fed our empirical insights as features: `is_novel_9m_volume`, `is_inst_sweet_spot`, `is_elite_close_pos`, `held_48h_absorption`, `held_5d_support`.

### C. Top-Down Sector & Theme Context (The Wind)
* **Sector / Theme Percentiles (1W, 1M, 3M, 6M):** Is the sector accelerating or dragging?
* **Absolute Thematic Returns:** What is the actual % return of the theme over the last quarter?
* **Thematic Breadth / Clustering:** How many *other* stocks in this exact same theme have triggered an EP in the last 14 days? (Detects massive institutional sector rotation).

### D. Market Regime & Recent Success Rates (The Environment)
* **EP Global Success Rate:** What is the win rate of the last 20 EPs across the entire market? (If the last 20 failed, the model learns we are in a toxic, trap-heavy regime).
* **Broad Market Breadth:** Percentage of all active US equities trading above their 50-day and 200-day SMAs.
* **Index Health (SPY/QQQ/IWM):** Distance to 20/50 SMAs, and VIX (Fear Gauge) levels.

### E. Dynamic Time-Series Features (The Evolution - For Day 5, 10, 20 updates)
*(To solve the flatlining probability issue you spotted on SMMT)*
* **Distance to 10 EMA / 20 EMA / 50 SMA:** Is the stock stretched +20% above its moving average, or resting directly on it?
* **Cumulative Volume Digestion:** Ratio of volume on green days vs red days since the EP.
* **Current Price vs Day 1 Close:** Is it holding the structural baseline?

---

## 3. Development & Integration Plan

### Phase 1: Data Preparation & The Out-of-Sample Split
1. **Universe Filtering (The Actionable Archetypes):** You are absolutely right. If we restrict this strictly to the 747 Pinnacle Elite setups, the model is completely blind to the incredibly lucrative Emerging Leaders, Institutional Sweet Spots, and Multi-Quarter Leaders. We will filter out the "Toxic Crap & Gap Traps", but we will feed the model **ALL actionable setups across the 4 main playbooks**. This ensures the model learns the unique nuances of each setup type (e.g., how a Sweet Spot behaves differently than a Compounder).
2. We strictly split the actionable events chronologically to prevent time-leakage:
   - **Train Set (60%):** 2016 – 2021 (The model learns the patterns here).
   - **Validation Set (20%):** 2022 (Used to tune hyper-parameters and prevent overfitting during a brutal bear market).
   - **Holdout Test Set (20%):** 2023 – 2026 (The blind test. The model has never seen this data. If it performs well here, we have a holy grail).

### Phase 2: Model Architecture & Training
1. **Target Definitions:** We train multiple binary classification models.
   - *Model A (The Home Run):* Predicts if the stock hits +100% before hitting the Day 1 Low stop.
   - *Model B (The Trap):* Predicts if the stock hits the Day 1 Low stop within 5 days.
2. **Algorithm:** We utilize **XGBoost**. It handles missing values, prevents overfitting via early stopping, and handles non-linear financial data better than Deep Learning neural networks.

### Phase 3: Interpretability (SHAP Values)
1. We cannot trade a "black box". If the model says "Buy", we need to know *why*.
2. We integrate **SHAP (SHapley Additive exPlanations)**. The model will output exact reasons for its score (e.g., *"Score is 85% because RVOL is massive (+20%), Theme is surging (+15%), but penalized because broader market is weak (-5%)."*)

### Phase 4: App Integration
1. The model is saved as a serialized `.pkl` or `.onnx` file.
2. The Live EP Tracker (Port 8783) loads this model into memory on startup.
3. Every day, the app feeds the live features of the active EPs into the model, generating real-time, dynamically shifting probabilities that update based on distance to the moving averages, completely solving the "flatlining" issue.
