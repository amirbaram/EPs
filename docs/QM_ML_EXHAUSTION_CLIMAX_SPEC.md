# Quantitative Blueprint: ML Climax Exhaustion Detection & Strategy Refinement for QM Setups (Port 8785)
## Technical Analysis, Architectural Code Audit, and Implementation Specification for Autonomous Trading Agents

> **Target Systems:** [`serve_qm.py`](file:///Users/amirbaram/Documents/code/EPs/serve_qm.py), [`build_qm_strategy_dataset.py`](file:///Users/amirbaram/Documents/code/EPs/build_qm_strategy_dataset.py), [`run_exact_ema_study.py`](file:///Users/amirbaram/Documents/code/EPs/run_exact_ema_study.py), [`update_qm_database.py`](file:///Users/amirbaram/Documents/code/EPs/update_qm_database.py)  
> **Target Dataset:** [`data/simulations/qm_strategy_trades.parquet`](file:///Users/amirbaram/Documents/code/EPs/data/simulations/qm_strategy_trades.parquet) (53,147 Breakout Trades)  
> **Context:** Cross-pollination of empirical findings from the Episodic Pivot (EP) Strategy System (`serve_ep.py`, `ep_ml_engine.py`, `serve_optimized_ep.py`) to the Qullamaggie Momentum Continuation (QM) Strategy System.  
> **Status:** Production Specification & Research Roadmap.

---

## 1. Executive Summary & Core Verdict

The **QM Strategy on Port 8785** models Qullamaggie momentum continuation setups across 53,147 historical breakouts over a 10-year period. While the setup geometry, TrendLab pivot low stop models ($0.6\times$ and $1.0\times$), and multi-factor dynamic conviction sizing ($S \in [0.50, 2.50]$) are well-structured, **the current trailing exit framework suffers from a severe structural ceiling**:

1. **Low Win Rate (24.4% to 31.8%):** Because exits rely solely on a *closing price violation* below the 10 EMA, 20 EMA, or 50 EMA, profitable trades that surge $+2.0\text{ R}$ to $+6.0\text{ R}$ into parabolic momentum frequently round-trip back to breakeven or small losses during inevitable pullbacks before triggering an exit.
2. **Profit Factor Compression:** The broad universe strategy produces modest profit factors of $1.11$ to $1.32$ under 100% EMA trailing. In contrast, when scaling out (Staged 33/33/34 or Balanced 50/50), drawdowns drop by up to $41\%$ (from $-372.0\text{ R}$ to $-220.4\text{ R}$).
3. **The Missing Piece — ML Climax Exhaustion Partial Profit Taking:**  
   **YES, the QM strategy will benefit immensely from its own dedicated Machine Learning Climax Exhaustion Detector.**
   In the institutional EP trading engine, the 1-Year Climax Exit model (`ep_exhaustion_classifier.pkl`) is the primary driver of peak expectancy. It locks in a **50% partial profit** when exhaustion probability reaches $\ge 0.50$, locking in peak R while retaining a risk-free 50% runner on the 50 SMA.
   Kristjan Qullamaggie himself explicitly preaches this rule:
   > *"Sell 1/3 to 1/2 of your position into strength after 3 to 5 days of momentum, then move stop to breakeven and trail the remainder on the 10 EMA or 20 EMA."*

Currently, the 8785 strategy models **zero partial profit taking into strength**—it only trails moving averages. Adding an ML Climax model will bridge the gap between empirical backtesting and actual Pinnacle Elite execution.

---

## 2. In-Depth Codebase Audit & Architectural Issues Found

### 2.1 Issue 1: Breakout Day Close Entry Lookahead / Slippage Bias
* **Location:** [`build_qm_strategy_dataset.py`](file:///Users/amirbaram/Documents/code/EPs/build_qm_strategy_dataset.py), Line 305:
  ```python
  entry_p = float(close_arr[idx])
  ```
* **The Problem:** The simulation enters trades at the **Close of the Breakout Day** (`close_arr[idx]`).
  - If a stock triggers at $\$50.00$ intraday and surges $+10\%$ to close at $\$55.00$, entering at $\$55.00$ creates severe execution distortion. The stop distance (e.g. to pivot low $\$45.00$) expands from $\$5.00$ to $\$10.00$, doubling initial dollar risk, cutting position size by $50\%$, and drastically reducing realized R-multiples.
  - Conversely, if a stock gapped above trigger at open and faded into the close, entering at the close assumes you could buy at a price that wasn't available when the breakout triggered.
* **EP Insight & Recommendation:** In the EP strategy, entering at Day 1 close was eliminated in favor of **Delayed Breakout Entry** (stop-buy placed 0.05 above trigger/high).  
  **Fix for QM:** The simulation should support:
  ```python
  # Realistic Intraday Stop-Buy Entry:
  trigger_px = float(setup["trigger"])
  # If open gapped above trigger, enter at open; otherwise enter at trigger + 0.05
  realistic_entry_p = max(float(open_arr[idx]), trigger_px + 0.05)
  ```

---

### 2.2 Issue 2: EMA Trailing Activation Dead Zone (Stranded 250-Bar Trades)
* **Location:** [`run_exact_ema_study.py`](file:///Users/amirbaram/Documents/code/EPs/run_exact_ema_study.py), Lines 55–64 & [`build_qm_strategy_dataset.py`](file:///Users/amirbaram/Documents/code/EPs/build_qm_strategy_dataset.py), Lines 198–207:
  ```python
  # 2. EMA Trailing Exit: Activated once EMA slopes upward and price is >= 1% above EMA
  if cur_ema > prev_ema and (cur_close / cur_ema - 1.0) >= 0.01:
      trail_active = True
      
  if trail_active:
      if cur_close < cur_ema:
          exit_p = cur_close
          ...
  ```
* **The Problem:** If a breakout triggers, but the stock moves sluggishly (e.g., gains $+0.8\%$ and never closes $\ge 1.0\%$ above the EMA) and then drifts sideways or downward *without* breaching the initial stop loss, **`trail_active` is NEVER set to `True`**.
  - The trade remains trapped in a simulation "dead zone" for **250 trading sessions**, finally exiting at `max_holding_period` even though price has traded below the 10 EMA for months!
* **Fix:** `trail_active` should activate immediately after Day 2 or once price is above the EMA, or if `cur_close < cur_ema` occurs after bar 3, trigger an exit regardless of whether the $+1\%$ hurdle was cleared.

---

### 2.3 Issue 3: Pipeline Disconnect Between Miner, Strategy Builder, and Study
* **Location:** [`build_qm_strategy_dataset.py`](file:///Users/amirbaram/Documents/code/EPs/build_qm_strategy_dataset.py), [`run_exact_ema_study.py`](file:///Users/amirbaram/Documents/code/EPs/run_exact_ema_study.py), [`update_qm_database.py`](file:///Users/amirbaram/Documents/code/EPs/update_qm_database.py).
* **The Problem:**
  1. `build_qm_strategy_dataset.py` currently only generates baseline 10 EMA columns (`r_06x`, `r_10x`). It does **not** compute `r_e10_06x`, `r_e20_06x`, `r_e50_06x`, `r_staged_06x`, or `r_balanced_06x`.
  2. `run_exact_ema_study.py` was executed as a one-off standalone script outputting to `exact_ema_study_results.parquet`.
  3. `update_qm_database.py` (which runs on automated daily updates) calls `mine_qm_setups.py`, but **never executes `build_qm_strategy_dataset.py`**.
  4. If an engineer re-runs `build_qm_strategy_dataset.py`, it overwrites `data/simulations/qm_strategy_trades.parquet` and **deletes all the multi-EMA and scaling columns**, breaking the Strategy Mode UI in `serve_qm.py`!
* **Fix:** Consolidate `run_exact_ema_study.py` directly into `build_qm_strategy_dataset.py` so that all 114 columns (including staged and balanced scaling) are computed in a single unified script, and wire it into `update_qm_database.py`.

---

## 3. Why QM Demands an ML Climax / Exhaustion Model

In the EP system, Episodic Pivots are fundamental shocks with massive quarterly drift. QM setups, by contrast, are **continuation patterns occurring after a stock has already rallied $+30\%$ to $+100\%+$**.

```
Price
  ▲                                           [BLOW-OFF / CLIMAX HIGH]
  │                                           🚨 SELL 33% - 50% PARTIAL PROFIT
  │                                                     ▲
  │                                                    ┌┴┐  Exhaustion Wick
  │                                                   ┌┘ └┐
  │                                                  ┌┘   └┐
  │                                ┌─────────────────┘     └───► 10 EMA Pullback
  │                               ┌┘                                 (Hold 50% Runner)
  │                   ┌───────────┘ Breakout
  │            ┌──────┘ Base
  │     Pole  ┌┘
  │    ┌──────┘
  │   ┌┘
  └───┴────────────────────────────────────────────────────────────────────────► Time
      Day 0       Day 10-25       Day 26 (Entry)   Day 29-33 (Climax)
```

### Empirical Justification:
1. **The Extension Climax:** After breaking out from a tight base, momentum leaders typically make an explosive $3$ to $5$-day push ($+15\%$ to $+40\%$). At the peak, the stock is stretched $4\times$ to $6\times\text{ ADR}$ above its 20 EMA with extreme RSI/volume.
2. **The Inevitable Mean Reversion:** Because momentum traders and algorithms take profits, the stock pulls back to test its rising 10 EMA or 20 EMA. A stock that was $+4.5\text{ R}$ at day 4 pulls back to $+1.2\text{ R}$ at the 10 EMA.
3. **The EMA Trailing Trap:** If 100% of the position is held until a close below the 10 EMA, the trader gives back **$70\%$ of the peak gain** on every single trade!
4. **The Mathematical Solution:**
   - **Sell 50% at Climax High:** Bank $+4.5\text{ R} \times 0.50 = +2.25\text{ R}$ locked in cash.
   - **Hold 50% Runner on 20 EMA or 50 SMA:** If the stock resumes its trend into a multi-month runner, the remaining 50% captures the $+10\text{ R}$ to $+20\text{ R}$ move. If it fails and stops out at breakeven ($0\text{ R}$), the net trade result is still **$+2.25\text{ R}$**, turning a losing/scratch trade into a solid winner.

---

## 4. Feature Engineering Specification for QM Climax Detector

The other session should engineer a feature vector $X_t \in \mathbb{R}^{24}$ computed on each open trade bar $t \in [t_{\text{entry}} + 1, t_{\text{exit}}]$ to predict the probability that current bar high is within $\le 3$ bars of the trade's terminal peak.

### 4.1 Group 1: Moving Average, ADR Stretch & RSI Momentum Extremes
| Feature Name | Mathematical Formula | Institutional Rationale |
|---|---|---|
| `feat_dist_ema10_pct` | $(Close_t - \text{EMA}_{10,t}) / \text{EMA}_{10,t}$ | Measures short-term extension above primary momentum moving average. |
| `feat_dist_ema20_pct` | $(Close_t - \text{EMA}_{20,t}) / \text{EMA}_{20,t}$ | Measures intermediate extension. Values $> 15\%$ indicate extreme stretch. |
| `feat_dist_sma50_pct` | $(Close_t - \text{SMA}_{50,t}) / \text{SMA}_{50,t}$ | Baseline trend extension. |
| `feat_adr_multiple` | $(High_t - \text{EMA}_{20,t}) / \text{ADR}_{14,t}$ | How many daily ADRs price is stretched above 20 EMA. $> 4.0\times$ is climax danger. |
| `feat_unrealized_r` | $(Close_t - Entry) / (Entry - Stop)$ | Current open R-multiple. Exhaustion model should only fire when $\ge +2.5\text{ R}$. |
| `feat_rsi14` | $\text{RSI}(Close, 14)$ | Standard oscillator level. Values $\ge 80.0$ indicate intense buying pressure. |
| `feat_rsi3_extreme` | $\text{RSI}(Close, 3)$ | **Ultra-fast 3-period RSI.** In fast 3–5 day breakouts, $\text{RSI}_3 \ge 90.0$ pinpoints blow-off tops with high precision. |
| `feat_rsi_divergence` | $\text{RSI}_{14,t} - \text{RSI}_{14, t-3} \text{ where } High_t > High_{t-3}$ | **Bearish momentum divergence.** Price prints higher high while internal momentum decelerates. |
| `feat_rsi_vs_sma` | $\text{RSI}_{14,t} - \text{SMA}_5(\text{RSI}_{14})_t$ | Measures the rate of oscillator exhaustion relative to its rolling baseline. |

> **💡 Crucial Research Note on RSI in Momentum Breakouts:**  
> In the baseline EP model (`ep_exhaustion_classifier.pkl`), standard $\text{RSI}_{14}$ was intentionally omitted from the primary 22 features because fundamental EP gap runners experience sustained multi-month drift where standard $\text{RSI}_{14}$ stays pinned between 75 and 85 without reversing, causing naive models to exit far too early.  
> **However, for QM momentum breakouts, RSI is highly valuable when implemented correctly:**  
> 1. Rather than relying on standard $\text{RSI}_{14} \ge 70$ (which triggers prematurely), the model must use **extreme overbought thresholds ($\text{RSI}_3 \ge 90$ or $\text{RSI}_{14} \ge 82$)**;  
> 2. It must combine RSI with **Volume Exhaustion (`feat_bar_rvol` $\ge 2.5\times$)** and **Upper Wick Rejection (`feat_upper_wick_ratio` $\ge 0.35$)**;  
> 3. It must evaluate **Momentum Divergence (`feat_rsi_divergence`)** to detect when buying energy has depleted despite a marginal new high.

### 4.2 Group 2: Candle Geometry & Climax Volume
| Feature Name | Mathematical Formula | Institutional Rationale |
|---|---|---|
| `feat_bar_rvol` | $Volume_t / \text{SMA}_{20}(Volume)_t$ | Volume surge into highs. Climax volume is typically $\ge 2.5\times$ average. |
| `feat_vol_vs_breakout` | $Volume_t / Volume_{\text{breakout}}$ | If day 4 volume exceeds breakout day volume on an extended candle, it signals a churn/exhaustion day. |
| `feat_upper_wick_ratio` | $(High_t - \max(Open_t, Close_t)) / (High_t - Low_t)$ | Upper shadow ratio. $> 0.40$ indicates severe intraday rejection and supply dumping. |
| `feat_close_pos` | $(Close_t - Low_t) / (High_t - Low_t)$ | Candle range position. Closing near the low after a run indicates exhaustion. |
| `feat_consec_up_days` | Count of consecutive bars where $Close_k > Close_{k-1}$ | 4 to 6 consecutive up bars without a pause signals parabolic exhaustion. |
| `feat_bars_since_entry` | $t - t_{\text{entry}}$ | Cycle timing. QM momentum moves typically climax between bars 3 and 7. |

### 4.3 Group 3: TrendLab Swings & Structural Momentum
| Feature Name | Mathematical Formula | Institutional Rationale |
|---|---|---|
| `feat_tl_swing_dist` | $(Close_t - PivotLow_{\text{base}}) / PivotLow_{\text{base}}$ | Net advance from consolidation launch low. |
| `feat_tl_is_pivot_high` | $1 \text{ if bar } t \text{ is TrendLab confirmed pivot high else } 0$ | Confirmed swing peak detection. |
| `feat_dist_52w_high` | $(Close_t - High_{52w}) / High_{52w}$ | Proximity to 52-week high (Blue Sky breakout vs overhead supply trap). |
| `feat_dist_pole_peak` | $(Close_t - PolePeak) / PolePeak$ | Advance beyond initial momentum pole peak. |

### 4.4 Group 4: Setup Heritage & Macro Sponsorship
| Feature Name | Mathematical Formula | Institutional Rationale |
|---|---|---|
| `feat_pole_move_pct` | Initial momentum pole magnitude (`move_pct`) | Monster poles ($\ge 50\%$) have larger secondary legs than small poles ($15\%$). |
| `feat_cons_bars` | Consolidation base duration (`cons_bars`) | Longer bases (20–40 bars) sustain longer runs than short flags (5–8 bars). |
| `feat_pinch_ratio` | Bollinger Band pinch ratio (`pinch_ratio`) | Tight squeezes ($< 2.5$) explode with greater velocity. |
| `feat_prior_ep_days` | Days since prior EP catalyst (`prior_ep_days_ago`) | Fresh EP setups have institutional drift, extending run duration. |
| `feat_rs_score` | Relative Strength Rating ($1 - 99$) | High RS leaders ($\ge 90$) withstand pullbacks; low RS names collapse on first dip. |
| `feat_macro_tailwind` | $1 \text{ if Leading Sector or Theme else } 0$ | Macro sponsorship prevents premature breakdown. |

---

## 5. Machine Learning Modeling & Training Specification

### 5.1 Training Target Formulation (Zero-Lookahead)
For each bar $t$ where open profit is $\ge +2.5\text{ R}$ (or $+15\%$):
- Compute forward maximum favorable excursion ($MFE_{t+10}$) and forward maximum adverse excursion ($MAE_{t+10}$).
- **Binary Target $Y_t = 1$ (Climax Exhaustion Event):**
  $$Y_t = 1 \iff \left( MAE_{t+5} \ge 1.5\text{ R} \right) \lor \left( High_t \ge 0.96 \times \max_{k \in [t, t+10]}(High_k) \text{ and } Close_{t+3} < Close_t \right)$$
  *(In plain English: A bar where taking profits captures $\ge 96\%$ of the 10-day peak before a sharp pullback of $\ge 1.5\text{ R}$ occurs).*

### 5.2 Model Architecture
- **Algorithm:** Calibrated Gradient Boosted Decision Trees (`XGBClassifier` or `LGBMClassifier`).
- **Objective:** `binary:logistic`, evaluating Area Under the ROC Curve (AUC) and Precision-Recall Curve.
- **Cross-Validation:** **Purged Walk-Forward Time-Series Split** (5 folds across 2016–2026) to prevent any temporal lookahead leakage.
- **Probability Calibration:** Isotonic regression or Platt scaling to ensure output probabilities $P(\text{Exhaustion}) \in [0.0, 1.0]$ reflect true historical frequencies.

---

## 6. Implementation Guidelines by Trader Archetype & Ticker Setup

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   QM HYBRID CLIMAX + TRAILING SCALE-OUT MATRIX                        │
├─────────────────────────┬──────────────────────┬───────────────────────────────────────┤
│ Trader Archetype        │ Climax Exit Rule     │ Trailing Exit on Remainder            │
├─────────────────────────┼──────────────────────┼───────────────────────────────────────┤
│ 1. High-Velocity Swing  │ Sell 50% @ P ≥ 0.45  │ Trail 50% on 10 EMA (or Breakout LOD) │
│    (Qullamaggie Pure)   │ (Hurdle: ≥ +2.5 R)   │ Fast capital turnover, max Sharpe     │
├─────────────────────────┼──────────────────────┼───────────────────────────────────────┤
│ 2. Institutional Hybrid │ Sell 33% @ P ≥ 0.50  │ Trail 33% on 10 EMA, 34% on 20 EMA    │
│    (Scale-Out)          │ (Hurdle: ≥ +3.0 R)   │ Balanced cashflow, min drawdown       │
├─────────────────────────┼──────────────────────┼───────────────────────────────────────┤
│ 3. Position / Trend     │ Sell 50% @ P ≥ 0.65  │ Trail 50% on 50 SMA / EMA             │
│    (Monster Hunter)     │ (Hurdle: ≥ +5.0 R)   │ Maximizes multi-month +15R runners    │
└─────────────────────────┴──────────────────────┴───────────────────────────────────────┘
```

### 6.1 By Ticker Setup Characteristics
1. **Fresh EP Igniters / Boosters (`prior_ep_days_ago <= 30`):**
   - High institutional post-earnings drift.
   - Do **NOT** sell too early. Require $P(\text{Exhaustion}) \ge 0.60$ and open profit $\ge +4.0\text{ R}$. Trail the remainder on the **20 EMA** or **50 SMA**.
2. **High-Beta Momentum & Monster Poles (`move_pct >= 50%`, Small/Mid-Caps):**
   - Violent multi-day bursts followed by brutal mean-reversion.
   - Aggressive climax trigger: Sell $50\%$ when $P(\text{Exhaustion}) \ge 0.45$ (or $\ge 3.0\times\text{ ADR}$ extension). Trail remainder closely on **10 EMA**.
3. **Institutional Core Leaders (`rs_score >= 90`, Large/Mega-Caps):**
   - Slower, sustained trends surfing the 20 EMA and 50 SMA.
   - Climax trigger should be conservative ($P \ge 0.60, \ge +4.0\text{ R}$), with $50\%$ runner trailed on the **50 SMA**.

---

## 7. Concrete Action Items for the Other Session Agent

The agent working on the QM strategy should execute the following steps in sequence:

1. **Fix Execution Slippage:** Update `build_qm_strategy_dataset.py` to use `max(open_p, trigger_p + 0.05)` instead of `close_p` on breakout day.
2. **Fix Trailing Dead Zone:** In `simulate_ema_exit`, ensure that trades do not hold for 250 bars if price falls below the trailing EMA, even if the initial $+1\%$ hurdle was missed.
3. **Unify Strategy Builder:** Merge `run_exact_ema_study.py` logic into `build_qm_strategy_dataset.py` so all 114 columns (including staged and balanced scaling) are computed in one reproducible script.
4. **Build Feature Extractor:** Create `qm_ml_engine.py` implementing the 24 climax features specified in Section 4.
5. **Train `qm_exhaustion_classifier.pkl`:** Train on the 53,147 breakout trades using temporal purged cross-validation.
6. **Simulate Climax Scaling:** Re-run strategy simulation with a new exit model: `strat_exit=climax_hybrid` (50% partial at Climax, 50% on 20 EMA / 50 SMA).
7. **Surface in UI:** Add Climax Exit markers (purple warning stars and profit badges) to the LightweightChart in `serve_qm.py` and add `strat_exit=climax_hybrid` to the toolbar dropdown on Port 8785.
8. **Verify Gate:** Run `bash scripts/check.sh` and ensure all tests in `tests/test_qm_viewer.py` pass cleanly.

---
*Authored by Antigravity Quantitative Trading Systems Architecture Team for amirbaram/EPs.*
