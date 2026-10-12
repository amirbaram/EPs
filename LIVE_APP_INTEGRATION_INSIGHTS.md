# Live App Integration Insights & Fixes Roadmap
**Document Version:** 1.0  
**Date:** October 9, 2026  
**Target Application:** `serve_ep_tracker.py` (Port 8783) and Live EP Scanner  
**Reference Benchmark:** `serve_ep.py` (Port 8782) & `ep_ml_engine.py`

---

## 1. Executive Summary & Objective

This document consolidates all critical quantitative findings, algorithmic calibrations, mathematical fixes, and architectural adjustments discovered and proven in the Episodic Pivot (EP) Historical Review engine (`serve_ep.py`). 

When integrating these capabilities into the **Live Tracker (`serve_ep_tracker.py`)** and daily scanner pipelines, this document serves as the mandatory specification to ensure zero regressions, consistent trade management logic, and perfectly aligned mathematics.

---

## 2. Quantitative Insights & Benchmark Expectancy

### A. The Trend-Following Win Rate Reality
- **Historical Illusion (Unstopped Market Drift):** Earlier raw data displayed 54%–56% win rates. This occurred because static parquet columns (`ret_60d > 0`) measured unstopped 60-day price drift. Over 54% of setups have slight positive drift (+1% to +3%), but in active trading they chop around, violate moving averages, and get stopped out.
- **Calibrated Trade Execution Win Rate:**
  - **Pinnacle Elite Setups:** **38%–39% Win Rate** (61%–62% losses/scratches).
  - **Broad EP Universe (Unfiltered):** **28.2% Win Rate** (71.8% losses/scratches).
- **The Core Asymmetric Edge:**
  - The strategy generates an Expected Value of **+3.46 R to +4.23 R per trade** with a Profit Factor of **~9.5**.
  - This edge does **not** rely on high win rates. It comes from an asymmetric **15:1 win/loss payoff ratio**:
    - **Average Winner:** **+9.9 R to +12.0 R** (runners hitting multi-quarter compounder targets).
    - **Average Loss:** **-0.7 R to -0.8 R** (losses strictly truncated at Day 1 Low or ratcheted higher by dynamic stops).

### B. Setup Performance Spectrum (Calibrated Benchmarks)
| Configuration | Universe | Win Rate | Expectancy (EV) | Avg Win | Avg Loss | Profit Factor | Max Drawdown | Total P&L |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Mechanical Baseline** (All ML OFF) | Pinnacle Elite | **39.2%** | **+3.46 R** | +9.9 R | -0.7 R | 9.55 | -9.00 R | +2,614.4 R |
| **AI Targets Only** | Pinnacle Elite | **39.2%** | **+3.57 R** | +10.1 R | -0.7 R | 9.85 | -9.00 R | +2,695.8 R |
| **Multi-Leg Baseline** (No ML Filter) | Pinnacle Elite | **42.4%** | **+4.74 R** | +12.5 R | -1.0 R | 9.20 | -11.40 R | +3,582.2 R |
| **Full AI Optimal** (All ML ON) | Pinnacle Elite | **39.5%** | **+4.23 R** | **+12.0 R** | **-0.8 R** | **9.43** | **-10.80 R** | **+3,194.8 R** |
| **Mechanical Baseline** | Full Market (6,501 EPs) | **28.2%** | **+2.03 R** | +9.1 R | -0.8 R | 6.78 | -64.77 R | +13,219.3 R |
| **Full AI Optimal** | Full Market (6,501 EPs) | **28.4%** | **+2.51 R** | +11.1 R | -0.9 R | 7.15 | -61.20 R | +16,295.9 R |

---

## 3. Mandatory Backend Calculations & Bug Fixes for Live App

When porting or updating live tracking in `serve_ep_tracker.py`, the following four code-level bug fixes must be replicated:

### 1. Avoid Query Parameter Shadowing in Simulation Loops
- **The Bug:** Checking `if outcome == "Gap & Crap Trap"` where `outcome` is the HTTP query parameter (`request.args.get("outcome")`). When the parameter is `None`, the check fails silently, bypassing all setup penalties and treating every positive price drift as a winner.
- **The Fix:** Always evaluate the per-row classification:
  ```python
  ev_outcome = r_ev.get("outcome", "")
  if ev_outcome == "Gap & Crap Trap" or r_ev.get("breached_d1_low_5d", False):
      t1_r = -1.0
  elif ev_outcome == "Fade to Black / Churn":
      t1_r = -0.5
  ...
  ```

### 2. Full Re-Entry Feature Dictionary in `ep_ml_engine.py`
- **The Bug:** `compute_features()` and `compute_rolling_features()` computed the 6 re-entry features but omitted them from the returned dictionary, causing `predict_rolling()` to evaluate the classifier on `0.0` values.
- **The Fix:** Ensure all 7 features required by the re-entry scaler are present in the dictionary:
  ```python
  reentry_feats = [
      "feature_days_since_ep",
      "feature_drawdown_from_peak",
      "feature_dist_10ema",
      "feature_dist_20ema",
      "feature_dist_50sma",
      "feature_vol_contraction",
      "feature_is_yellow_flip"
  ]
  ```

### 3. Dynamic Stop Loss Mathematical Consistency in Dossiers
- **The Bug:** Setting `t1_ret = -risk1_pct` whenever `t1_stopped` was True. This overwrote the dynamic trailing stop and falsely reported a full -1.0 R loss (-5.0%) even when the stop had ratcheted to -1.2%.
- **The Fix:** Base trade return strictly on the achieved exit price:
  ```python
  t1_ret = (t1_exit_price / d1_close - 1.0) * 100.0
  t1_r = t1_ret / risk1_pct if risk1_pct > 0 else 0.0
  ```

### 4. Multi-Leg (Trade 2 & Trade 3) Yellow Re-Entry Architecture
- **Rules of Engagement:**
  1. **Consolidation Duration:** Maximum 45 sessions in Blue/Gray Larsson state.
  2. **Retracement Gate:** Pullback low must hold within 55% of the prior peak high.
  3. **Trigger:** Daily close re-flipping back to Yellow (Fast Larsson ribbon alignment).
  4. **Stop Loss:** 5-day swing low (`swing5_low`) or 32 EMA.
  5. **ML Quality Gate:** When `use_multileg_ml` is enabled, reject any candidate with `prob_reentry < 0.40`.

---

## 4. UI / UX & Interactive Control Standards for Live App

To match the operational capabilities of the Historical Review app, `serve_ep_tracker.py` should implement:

### 1. Dynamic Strategy Execution Bar
Provide the top control bar with 4 live toggles and 2 quick-presets:
- `⚙️ ML Targets (Exhaustion)`: Toggles climax profit-taking (`prob_exhaustion > 0.80`).
- `🛡️ ML Trailing Stop (10th %ile MAE)`: Toggles daily ratchet stop based on predicted MAE.
- `🔄 Multi-Leg Execution (Trade 2 & 3)`: Toggles second/third leg tracking on Yellow flips.
- `🤖 Filter Multi-Leg by ML (≥40%)`: Toggles machine learning entry conviction gate.
- **Quick Presets:** `[Mechanical Baseline]` and `[AI-Enhanced Optimal]`.

### 2. Dossier Multi-Leg Card States
The right-hand trade card in the live tracker must clearly distinguish 3 states:
1. **Multi-Leg OFF:** Neutral indicator explaining that only Trade 1 is being tracked.
2. **ML Vetoed:** Amber pill stating `Vetoed by ML Re-Entry Model (X.X% < 40.0% threshold)`.
3. **Active Trade 2 / 3:** Full execution details (Entry Price, Stop Price, Risk %, Consolidation Days, Retracement %, Current R-multiple, Combined Net R).

### 3. Decommissioning Legacy Components
- **Toxic Flow Warning:** Permanently disabled and replaced with the pure technical trap archetype (`violated_48h` or `close_pos < 0.50`).
- **V1 Snapshot Model:** Deprecated in favor of the dynamic V2 Rolling Model with multi-day timeline progression.

---

## 5. Live Tracking Edge Cases & Pre-Flight Checklist

Before deploying updates to the live tracker or automated alerts:

1. **Intraday Bar vs EOD Bar Timing:**
   - EOD metrics (like `close_pos` and `rvol`) change throughout the session until 16:00 ET.
   - For live tracking during market hours, dynamic stops and ML targets must evaluate against the latest live composite bar.
2. **Corporate Actions & Dividends:**
   - Always verify adjusted close consistency across yfinance bars to avoid artificial Yellow flip triggers caused by dividend drops or splits.
3. **Multi-Leg Peak Anchoring:**
   - When a stock enters Trade 2, ensure the peak high is anchored from the original Day 1 EP surge, NOT just the local Trade 2 swing.
4. **Zero Hardcoded Metrics:**
   - All banner KPIs, win rates, and dossier stats must calculate dynamically from the active event subset and user selections.
