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

### A. The Real-Bar Trade Execution Reality (Zero-Lookahead Audited)
- **Historical Inflation Traps Eliminated:**
  - **48H Violation Deletion Trap:** Earlier figures displayed 43.8% win rates and +1.30R to +1.49R EV because `~violated_48h` was applied at Day 1 screening time. This secretly deleted 230 Day 2–3 absorption failures (-0.78R avg, 7.0% WR), artificially inflating edge. In reality, screening is strictly causal (Day 1 RVOL, gap, close position, sector/theme momentum).
  - **Trade Ledger Distortion:** Summing `t1_r + t2_r` into a single event combined distinct trades into one, erasing Trade 1 losses when Trade 2 succeeded and dividing profit by events (6,500) rather than actual trades executed (8,705).
- **Audited Ground-Truth Benchmarks:**
  - **Pinnacle Elite Trade 1 (Baseline Mechanical):** **35.2% Win Rate**, **+0.74 R EV**, **PF 2.15** ($N=953$).
  - **Pinnacle Elite (Day 3 Defensive Exit):** **33.5% Win Rate**, **+0.66 R EV**, **PF 2.08** (exits Day 3 close on absorption breach).
  - **Pinnacle Elite (Day 5 Progressive Pyramiding):** **34.6% Win Rate**, **+0.90 R EV**, **PF 2.24** (audited 2-tranche P&L math).
  - **Conservative Swing (Strong Close: Delayed 5D Breakout):** **36.2% Win Rate**, **+0.43 R EV**, **PF 1.70** ($N=4,927$, avoids 992 Day 2 gap-down traps).
  - **Bonde DRE (Weak Day 1 Close: Delayed Reaction):** **31.0%–35.9% Win Rate**, **+0.71 R to +0.86 R EV**, **PF 2.02–2.20** ($N=4,297$, filters 62% of toxic fade traps).
  - **Broad Market EPs (Full Per-Trade Portfolio Ledger):** **31.9% Win Rate**, **+0.46 R EV**, **PF 1.74** ($N=8,705$ executed trades, $+4,025.3\text{ R}$ total P&L).

### B. Setup Performance Spectrum (Audited Benchmarks)
| Configuration | Strategy / Archetype | Win Rate | Expectancy (EV) | Profit Factor | Sample ($N$) | Total P&L | Key Trade Management Rule |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **Pinnacle Baseline** | 💎 Pinnacle Elite (T1) | **35.2%** | **+0.74 R** | 2.15 | 953 trades | +709.7 R | Day 1 Close Entry, Hard Stop Day 1 Low, 50 SMA Trail |
| **Pinnacle + D3 Defense** | 💎 Pinnacle Elite (D3 Exit) | **33.5%** | **+0.66 R** | 2.08 | 953 trades | +632.3 R | Cuts trade at Day 3 Close if upper 50% body breached |
| **Pinnacle + Pyramiding** | 💎 Pinnacle Elite (Add) | **34.6%** | **+0.90 R** | 2.24 | 953 trades | +855.4 R | +50% size tranche on high-conviction Day 5 close |
| **Pinnacle Continuation** | ♻️ Pinnacle (Trade 2 Only) | **30.0%** | **+0.34 R** | 1.68 | 313 trades | +105.6 R | Re-entry on yellow flip after orderly consolidation |
| **Pinnacle Multi-Trade** | 💎 Pinnacle (Honest Ledger) | **33.9%** | **+0.64 R** | 2.06 | 1,266 trades | +815.4 R | Full portfolio combining T1 and T2 executed legs |
| **Bonde DRE (Day 2)** | ⚡ Bonde Delayed Reaction | **31.0%** | **+0.86 R** | 2.20 | 1,281 trades | +1,101.7 R | Weak Day 1 Close; stop-buy on Day 2 D1 High breakout |
| **Bonde DRE (Day 3)** | ⚡ Bonde Delayed Reaction | **35.9%** | **+0.71 R** | 2.02 | 840 trades | +596.4 R | Weak Day 1 Close; stop-buy on Day 3 D1 High breakout |
| **Conservative Swing** | 🛡️ Conservative Swing | **36.2%** | **+0.43 R** | 1.70 | 4,927 trades | +2,095.5 R | Strong Close; waits up to 5 days for D1 High breakout |
| **Full EP Portfolio** | 🌐 Broad Market (All Legs) | **31.9%** | **+0.46 R** | 1.74 | 8,705 trades | +4,025.3 R | Unpacked distinct trade ledger across full 10-year history |

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

---

## 7. Verified Point-in-Time Real-Bar Backtests: DRE & Conservative Swing

### A. Weak Day 1 Closes: The Bonde DRE Strategy (2015–2026 Universe)
- **Problem:** Entering on Day 1 Close when a gap closes weak ($\text{ClosePos} < 0.65$) is mathematically toxic (62% collapse through Day 1 Low).
- **Solution (Bonde DRE):** Wait for price to hold Day 1 Low and trigger a breakout above Day 1 High within a 1–5 session window.
- **Empirical Results (15,655 Debounced Setups):**
  - **1-Day Wait (Day 2 Breakout):** Filters **7,665 traps** (saved from loss!) | 2,909 trades | **Win Rate 30.2%** | **EV +0.86 R** | **Profit Factor 2.20** | **Avg Stop 7.4%** | Total P&L **+2,491.0 R**.
  - **3-Day Wait:** Filters **9,257 traps** | 4,297 trades | **Win Rate 31.0%** | **EV +0.71 R** | **Profit Factor 2.02** | **Avg Stop 7.9%** | Total P&L **+3,038.3 R**.
  - **5-Day Wait:** Filters **9,698 traps** | 4,711 trades | **Win Rate 31.4%** | **EV +0.67 R** | **Profit Factor 1.98** | **Avg Stop 8.1%** | Total P&L **+3,173.1 R**.
- **Archetype Assignment:** `⚡ Delayed Reaction (DRE)` | Sizing: **$1.0\text{ R}$ on breakout**.

### B. Strong Day 1 Closes: Delayed Breakout vs. Immediate Close Entry
- **Problem:** Traders wanting higher win rate and lower shakeout churn on classical EPs.
- **Solution:** Place stop-buy at Day 1 High instead of entering market on Day 1 close.
- **Empirical Results (6,501 Setups):**
  - **Immediate Day 1 Close Entry:** 6,501 trades | **Win Rate 32.4%** | **EV +0.40 R** | **Profit Factor 1.65** | Total P&L **+2,603.0 R**.
  - **Delayed Entry (5-Day Breakout Window):** 4,927 trades | **Win Rate 36.2% (+3.8% boost)** | **EV +0.43 R** | **Profit Factor 1.70** | Total P&L **+2,095.5 R** | **Filters out 992 Day 2 gap-and-crap traps**.
- **Archetype Assignment:** `🛡️ Conservative Swing / Low-Churn` | Sizing: **$1.0\text{ R}$ on Day 2–5 breakout**.
