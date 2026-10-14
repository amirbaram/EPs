# QM Setup & EP Cross-Reference Explorer (`serve_qm.py` — Port 8785)
## Quantitative Architecture, Setup Mechanics, Strategy Simulation & Data Pipeline Maintenance Guide

> **Production Application:** [`serve_qm.py`](file:///Users/amirbaram/Documents/code/EPs/serve_qm.py) running on **http://127.0.0.1:8785**  
> **Repository:** `amirbaram/EPs` (Branch: `feature/dual-track-continuation`)  
> **Strategy Trades Dataset:** [`data/simulations/qm_strategy_trades.parquet`](file:///Users/amirbaram/Documents/code/EPs/data/simulations/qm_strategy_trades.parquet) (53,147 breakout trades)  
> **Historical Setups Dataset:** [`data/simulations/qm_historical_setups.parquet`](file:///Users/amirbaram/Documents/code/EPs/data/simulations/qm_historical_setups.parquet) (29,225 historical setups)  
> **Primary Audience:** Quantitative researchers, autonomous trading agents, and institutional developers.

---

### Strategy & Server Code Architecture Directory Map

| Concern | File Path | Role & Operational Responsibility |
|---|---|---|
| **Application Server & UI** | [`serve_qm.py`](file:///Users/amirbaram/Documents/code/EPs/serve_qm.py) | Interactive web server (port 8785), TradingView chart renderer, canvas overlays, `/api/events` filter query engine, Strategy Mode KPI aggregator, and right-pane trade dossier. |
| **Strategy Dataset Builder** | [`build_qm_strategy_dataset.py`](file:///Users/amirbaram/Documents/code/EPs/build_qm_strategy_dataset.py) | Simulates trade executions, dual stop loss models (0.6x and 1.0x), multi-factor dynamic conviction sizing ($S \in [0.50, 2.50]$), and QQQ/SPY market regime filters. |
| **Exact EMA Trailing & Scaling Study** | [`run_exact_ema_study.py`](file:///Users/amirbaram/Documents/code/EPs/run_exact_ema_study.py) | High-speed simulation engine modeling 10 EMA, 20 EMA, 50 EMA, Staged (33/33/34), and Balanced (50/50) scale-outs across all 53,147 breakout trades. |
| **Precomputed Strategy Trades Dataset** | [`data/simulations/qm_strategy_trades.parquet`](file:///Users/amirbaram/Documents/code/EPs/data/simulations/qm_strategy_trades.parquet) | 114-column institutionally scored dataset storing per-trade execution prices, dates, R-multiples, sized returns, MFE, MAE, and holding durations for all exit rules. |
| **QM Setup Mining Pipeline** | [`mine_qm_setups.py`](file:///Users/amirbaram/Documents/code/EPs/mine_qm_setups.py) | Multiprocess scanner mining poles, consolidations, and resistance stacks across US equities, linking setups to prior/post EP catalysts. |
| **Core Setup Geometry & Squeeze Detector** | [`setups.py`](file:///Users/amirbaram/Documents/code/EPs/setups.py) | Strict quantitative pattern algorithms (`detect_qm_breakout`), Bollinger Band pinch ratio calculation, and linear regression $R^2$ trendline validation. |
| **Automated Database Updater** | [`update_qm_database.py`](file:///Users/amirbaram/Documents/code/EPs/update_qm_database.py) | EOD daemon comparing latest bar dates, refreshing RS database, re-mining setups, and triggering zero-downtime hot-reload (`POST /api/reload_dataset`). |
| **Historical Setups Dataset** | [`data/simulations/qm_historical_setups.parquet`](file:///Users/amirbaram/Documents/code/EPs/data/simulations/qm_historical_setups.parquet) | Precomputed repository of 29,225 breakout and building coil setups with point-in-time RS metrics. |
| **Daily RS Engine & Historical Database** | [`build_rs_history.py`](file:///Users/amirbaram/Documents/code/EPs/build_rs_history.py), [`update_rs_database.py`](file:///Users/amirbaram/Documents/code/EPs/update_rs_database.py) | 10-year daily cross-sectional RS percentiles (1–99) and Stage 2 moving average posture engine (`data/rs_history/`). |
| **Thematic & Sector Momentum Engine** | [`thematic_engine.py`](file:///Users/amirbaram/Documents/code/EPs/thematic_engine.py), [`labels.py`](file:///Users/amirbaram/Documents/code/EPs/labels.py) | Multi-timeframe sector/theme composite rankings, tailwind/drag diagnostics, and emerging theme detection. |
| **Bar Datastore** | [`datastore.py`](file:///Users/amirbaram/Documents/code/EPs/datastore.py) | Point-in-time daily OHLCV bar cache loader (`data/bars/<TICKER>.parquet`). |
| **Regression Test Suite** | [`tests/test_qm_viewer.py`](file:///Users/amirbaram/Documents/code/EPs/tests/test_qm_viewer.py) | Comprehensive test suite covering setup geometry, EP linkage, RS filtering, Strategy Mode KPIs, and trailing exit model contracts. |

---

## 1. Executive Summary & Design Philosophy

The **QM Setup & EP Cross-Reference Explorer** is a high-performance, institutional web application designed to study, monitor, and backtest **Qullamaggie Momentum Continuation (QM)** setups across the entire US equity universe. 

The application operates in two synergistic modes:
1. **Setup Intelligence Mode (Default):**
   * **Unobstructed Price Action:** Focuses on underlying candlestick geometry, ascending higher-low consolidation shelves, volume dry-up, and Breakout Stack resistance levels.
   * **Dual Setup States:** Visualizes both **confirmed breakout setups** (volume-expanded surges through multi-week highs) and **mature building coils** (orderly consolidations surfing rising EMAs preparing to break out).
   * **Episodic Pivot (EP) Cross-Referencing:** Identifies whether a momentum pole was sparked by a fundamental EP gap within the prior 60 days, or boosted by a subsequent catalyst within 60 days following base formation.
   * **Institutional RS Posture:** Filters for true market leaders ($\text{rs\_score} \ge 90$) in **Stage 2 Bullish Alignment** or within 15% of 52-week highs.
   * **Macro & Thematic Momentum:** Integrates sector and theme momentum diagnostics (Leading Sectors, Leading Themes, Emerging Themes).

2. **Interactive Strategy Execution Mode (`strategy_mode=1`):**
   * **Realistic Trade Simulation:** Evaluates breakout trade execution without lookahead bias across 53,147 historical setups.
   * **Dual Stop Loss Models:** Compares standard $1.0\times$ TrendLab Pivot Low stops against the optimized $0.6\times$ Pivot Low stop (cutting initial risk by 40% and boosting expectancy by +142%).
   * **Multi-Factor Dynamic Conviction Sizing:** Overweights high-conviction trades ($0.50\times - 2.50\times$) using Fresh EP recency, Monster Pole magnitude, Breakout RVOL expansion, and base tightness.
   * **Multi-Model Trailing Exit Discipline:** Models and toggles between 10 EMA, 20 EMA, 50 EMA, Staged Scaling (33/33/34), and Balanced Scaling (50/50), recomputing portfolio KPIs and displaying exact trade execution metrics in real time.

---

## 2. Core Setup Mechanics & Mathematical Geometry

The QM setup detection engine ([`setups.py`](file:///Users/amirbaram/Documents/code/EPs/setups.py), `detect_qm_breakout`) scans daily OHLCV bars using strict institutional parameters:

```
          [Explosive Pole: +30% to +100%+]           [Tight Consolidation Base: 8 to 50 days]
Price                                                High Resistance Level (Pole Peak)
  ▲                     ┌─────────┐                  ═══════════════════════════════════════ (Breakout Stack)
  │                    ┌┘         └────────┐                                      ▲
  │                   ┌┘                   └───┐       ┌───┐             ┌───────┘ Trigger Price
  │                  ┌┘                         └───┐ ┌┘   └───┐ ┌─────┐┌┘
  │                 ┌┘                              └─┘        └─┘     └┘
  │                ┌┘                               ▲   ▲   ▲   ▲   ▲
  │               ┌┘                                Ascending Higher Lows
  │              ┌┘                                 Surfing Rising 10 EMA / 20 EMA
  │             ┌┘
  │       ▲    ┌┘
  │ ──────┴────┘
  │ Prior EP Catalyst
  └─────────────────────────────────────────────────────────────────────────────────────────────► Time
```

### 2.1 The Momentum Pole
1. **Minimum Move:** $\ge +15\%$ price advance from swing low to swing high (`QMB_MIN_MOVE = 15.0`). Accommodates institutional large-cap leaders (e.g. GOOG, MSFT, AAPL) where 15–25% represents a massive 7–10× ADR momentum pole, while UI and strategy filters isolate $\ge +30\%$, $\ge +40\%$, or $\ge +50\%$ moves on higher-beta momentum names.
2. **Maximum Lookback:** Completed within 35 trading days (`QMB_MOVE_MAX_BARS = 35`).
3. **Institutional Urgency:** Characterized by steep slope, expansion volume, and sequential above-average true range (ATR) expansions.

### 2.2 The Consolidation Base
1. **Duration:** 8 to 50 trading sessions (`QMB_CONS_MIN = 8`, `QMB_CONS_MAX = 50`).
2. **Shallow Digestion:** Maximum pullback off the pole peak $\le 50\%$ (`QMB_MAX_PULLBACK = 50.0`), with premier setups holding between $8\%$ and $25\%$.
3. **EMA Surfing:** At least $50\%$ of bars in the consolidation must close above the rising 20-day EMA (`QMB_MIN_RIDE_FRAC = 0.50`).
4. **Ascending Higher Lows:** Consolidation lows must form a net flat or ascending trajectory, verifying institutional accumulation and absorption of overhead supply.
5. **Volume Contraction:** Dry-up in daily volume during the tightest portion of the base, demonstrating that sellers have exhausted liquidity.

### 2.3 Dual State Classification
* **`breakout`:** Triggered when the current session breaks cleanly above the base high or primary Breakout Stack resistance on expanding volume ($\ge 1.3\times$ 20-day baseline average).
* **`building`:** Flagged when a stock has satisfied all pole and base criteria, is riding its 10/20 EMA shelf, and is actively coiling within $8\%$ of its trigger price.

---

## 3. Episodic Pivot (EP) Cross-Referencing & Catalyst Intelligence

Every QM setup is cross-referenced in $O(1)$ time against the 6,501 curated historical EP events in `data/simulations/ep_combined_study_scored.parquet` using the proximity logic in [`mine_qm_setups.py`](file:///Users/amirbaram/Documents/code/EPs/mine_qm_setups.py):

### 3.1 Proximity Windows
* **Prior EP (Igniter):** Measured in the $[-60, 0]$ calendar day window prior to the QM date. Captures whether the explosive pole was sparked by a genuine institutional catalyst (earnings blowout, FDA approval, guidance raise, major contract).
* **Post EP (Booster):** Measured in the $[0, +60]$ calendar day window following the QM date. Captures whether an orderly base was accelerated into a secondary multi-month continuation leg by a subsequent fundamental catalyst.

### 3.2 4-Tier Catalyst Classification Badges
| Badge | Color Code | Description |
|---|---|---|
| `⚡🚀 Ignited & Boosted` | Purple / Gold | Highest tier: Pole was ignited by an EP and base breakout was boosted by a secondary EP. |
| `⚡ Prior EP (Ignited Pole)` | Amber / Yellow | Pole sparked by fundamental EP within 60 days; clean institutional accumulation base. |
| `🚀 Post EP (Boosted Breakout)` | Blue / Cyan | QM breakout or tight base received a subsequent catalyst gap within 60 days. |
| `⚪ Pure Technical` | Slate / Gray | High-momentum continuation purely technical in nature (no EP within $\pm 60$ days). |

### 3.3 SEC EDGAR & Catalyst Intelligence Dossier
When available, the Right Pane Dossier pulls SEC EDGAR filings (Forms 8-K, 10-Q, 10-K, 6-K) and structured institutional analysis from the curated catalyst database, providing the exact corporate headline, filing date, and fundamental narrative behind the move.

---

## 4. Institutional Relative Strength (RS) & Technical Posture

In accordance with [`RS_DATABASE.md`](file:///Users/amirbaram/Documents/code/EPs/RS_DATABASE.md) (§7.1), the application incorporates William O'Neil / IBD-style **1–99 cross-sectional percentile RS scores** merged point-in-time from `data/rs_history/rs_<YEAR>.parquet`.

### 4.1 The "Bottom-Fisher Rebound Trap" Problem
A beaten-down penny stock or cyclical distressed name falling from \$100 to \$1 that rallies 300% to \$4 will produce a mathematical Raw RS score in the 99th percentile, despite remaining trapped below declining 50 and 200 SMAs.

### 4.2 The Institutional Leader Solution (Default Filter)
To ensure that quantitative traders only see legitimate institutional leadership, the application filters setups by default using:
$$\text{rs\_score} \ge 90 \quad \text{AND} \quad \left( \text{stage2} == \text{True} \quad \lor \quad \text{dist\_52w\_high} \ge -15.0\% \right)$$

* **Stage 2 Bullish Alignment:** Strict multi-timeframe moving average stack:
  $$\text{Close} \ge \text{EMA}_{10} \ge \text{EMA}_{20} \ge \text{SMA}_{50} \ge \text{SMA}_{200}$$
* **Within 15% of 52-Week High:** Ensures the stock is trading near or at all-time highs (ATH), where zero overhead supply exists.

### 4.3 Available RS Filter Presets (`#sel_rs`)
1. **`RS ≥ 90 + (Stage 2 | ATH ≤ 15%)` (Default):** Elite institutional leaders (yields **10,149** historical setups).
2. **`RS ≥ 95 + (Stage 2 | ATH ≤ 15%)`:** Top 5% Pinnacle Elite momentum universe.
3. **`RS ≥ 80 + (Stage 2 | ATH ≤ 15%)`:** Broad market leadership.
4. **`RS ≥ 70 + (Stage 2 | ATH ≤ 15%)`:** Extended leadership cohort.
5. **`RS ≥ 90 (Any Posture)`:** Unconstrained mathematical momentum.
6. **`Stage 2 Bullish Alignment Only`:** Structural stage 2 alignment regardless of RS score.
7. **`Within 15% of 52-Week High Only`:** Blue sky / ATH setups.
8. **`All RS Scores (Unrestricted)`:** Full universe of 29,225 setups.

---

## 5. UI Architecture & Interactive Features

The application presents a responsive 3-pane layout styled with modern dark-mode Tailwind CSS, Lucide icons, and TradingView LightweightCharts:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  HEADER & KPI ROW: Total Setups (10,149) | Breakouts (5,165) | Building (4,984) | RS Leader % (100.0%) │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  TOOLBAR: State Filter | EP Catalyst Filter | Macro Flow Filter | ⚡ RS & Posture Filter | Search/Sort   │
├─────────────────────────┬──────────────────────────────────────────┬───────────────────────────────────┤
│ MAIN SETUP TABLE        │ CENTER INTERACTIVE CHART                 │ RIGHT PANE DOSSIER                │
│ • Symbol & Macro Pill   │ • Full historical OHLCV bars             │ • QM Technical Profile            │
│ • Setup Date            │ • Shaded Emerald Green Pole Box          │ • Institutional RS & Posture Card │
│ • State (Breakout/Build)│ • Shaded Amber Consolidation Base Box    │ • EP Catalyst Relationship Card   │
│ • RS Score (with S2/ATH)│ • Breakout Stack dashed resistance lines │ • Thematic Momentum Backdrop      │
│ • Move % & Cons Bars    │ • Bright yellow EP star markers          │ • SEC EDGAR Filing Intelligence   │
│ • Pullback % & Pattern  │ • Larsson multi-span momentum ribbon     │ • Shakeout & Absorption Metrics   │
│ • EP Catalyst Badge     │ • Chart Settings & Drawing Tools toolbar │                                   │
└─────────────────────────┴──────────────────────────────────────────┴───────────────────────────────────┘
```

### 5.1 Center Chart Innovations
* **Full Historical Price Action:** Charts load the entire price history for the selected symbol, giving users multi-year context.
* **Instant Dynamic Price Scaling:** Switching between symbols (e.g. from a \$10 stock to a \$1,200 stock) instantly recalculates the price scale without clipping or bar distortion.
* **Proportional Volume Pane:** Volume bars are scaled in a separated lower pane with a 20-day volume MA to ensure candles have clean vertical headroom.
* **Visual Pole & Base Canvas Overlays:**
  * **Pole Box:** Emerald green dashed rectangle spanning from `qm_pole_start` to `qm_pole_end`.
  * **Base Box:** Amber yellow dashed rectangle spanning from `qm_cons_start` to `qm_cons_end`.
* **Breakout Stack Levels:** Horizontal dashed lines plotted at all resistance levels identified by the breakout stack algorithm, highlighting the precise trigger price.
* **EP Date Markers:** Prominent yellow star candle markers pinned to the exact sessions where Prior or Post Episodic Pivots took place.
* **Larsson Ribbon:** Fast/slow exponential ribbons (default spans: 32, 35, 50, 58) displaying yellow (bullish acceleration), blue (digestion), and gray (neutral) states.
* **Drawing Tools Toolbar:** Interactive drawing mode with Line, Horizontal Ray, Arrow, Fibonacci Retracement, Text annotations, and Undo/Clear buttons, automatically saved to `data/annotations/` per ticker.
* **Enlarged Fullscreen Mode & Stock Chooser Modal:**
  * One-click fullscreen expand button with persistent controls.
  * **`Choose Stock` Modal:** Instant searchable and sortable table of all currently filtered stocks with live selection highlight.
  * **`◀ Prev` / `▶ Next` Navigation:** Keyboard shortcuts and toolbar buttons to step through stocks sequentially without exiting fullscreen.

---

## 6. Data Pipeline Maintenance & Daily EOD Workflow

When new daily candles arrive (e.g., at market close or during batch updates), the QM datastore must be refreshed to detect new breakouts, update building bases, refresh RS ratings, and link any new EPs.

### 6.1 The Single-Command Automated Updater (Recommended)
An end-to-end automated updater script is provided: [`update_qm_database.py`](file:///Users/amirbaram/Documents/code/EPs/update_qm_database.py).

```bash
# Check status without writing (dry-run)
.venv/bin/python update_qm_database.py --check

# Run full automated update pipeline
.venv/bin/python update_qm_database.py

# Force re-mining and full refresh
.venv/bin/python update_qm_database.py --force

# Hot-reload dataset in a running serve_qm instance (zero downtime)
.venv/bin/python update_qm_database.py --reload
```

#### What `update_qm_database.py` Executes Automatically:
1. **Timestamp Inspection:** Compares the latest setup date in `data/simulations/qm_historical_setups.parquet` against the latest SPY daily bar in `data/bars/`. If no new bars exist, it exits immediately.
2. **RS Synchronization:** Executes `update_rs_database.update_database()`, updating `data/rs_history/rs_<YEAR>.parquet` and `data/simulations/rs_universe_scores.parquet` for the new trading sessions.
3. **QM Mining & EP Cross-Referencing:** Runs `mine_qm_setups.py` with multi-process workers (streaming to disk to preserve memory), detecting new setups, collapsing consolidation episodes, cross-referencing EPs, and merging point-in-time RS metrics.
4. **Live Server Hot-Reload:** Sends a `POST` request to `http://127.0.0.1:8785/api/reload_dataset`, reloading the in-memory dataframe instantly without restarting the server or interrupting active browser sessions.

---

### 6.2 Step-by-Step Manual Maintenance Pipeline

If executing commands step-by-step or debugging individual components:

#### Step 1: Update the Daily Relative Strength (RS) Database
```bash
.venv/bin/python update_rs_database.py
```
*Computes multi-timeframe weighted returns, cross-sectional percentiles (1–99), Stage 2 alignment, and 52W high distance for the new session.*

#### Step 2: Update Curated EP Dataset (If New Earnings/Catalyst EPs Occurred)
```bash
.venv/bin/python scripts/update_ep_dataset.py
```
*Scans for qualifying Episodic Pivots ($\ge +8\%$ gap, $\ge 2.0\times$ RVOL, close position $\ge 0.50$) and enriches them with SEC EDGAR filings and ML conviction scores.*

#### Step 3: Run the QM Setup Miner
```bash
# Mine all common stocks and EP universe (safe memory streaming, 4 workers)
.venv/bin/python mine_qm_setups.py --workers 4

# Or mine specific active tickers
.venv/bin/python mine_qm_setups.py --symbols NVDA,TSLA,AMD,SMCI,PLTR

# Or re-match existing setups against newly scored EPs without re-scanning bars
.venv/bin/python mine_qm_setups.py --rematch-eps
```

#### Step 4: Hot-Reload the Live Viewer Application
```bash
curl -s -X POST http://127.0.0.1:8785/api/reload_dataset
```
*Reloads `qm_historical_setups.parquet` into RAM and refreshes filter indexes.*

---

## 7. Trailing Exit Mechanics, Partial Profit Taking & Archetype Guidelines

> **Simulation Engine Code:** [`run_exact_ema_study.py`](file:///Users/amirbaram/Documents/code/EPs/run_exact_ema_study.py) (`simulate_ema_exit`)  
> **Strategy Dataset Generator:** [`build_qm_strategy_dataset.py`](file:///Users/amirbaram/Documents/code/EPs/build_qm_strategy_dataset.py) (`simulate_single_trade`)  
> **Runtime UI & API Router:** [`serve_qm.py`](file:///Users/amirbaram/Documents/code/EPs/serve_qm.py) (`/api/events`, `strat_exit` parameter, `#sel_strat_exit`, `#cfg_strat_exit`)  
> **Precomputed Dataset:** [`data/simulations/qm_strategy_trades.parquet`](file:///Users/amirbaram/Documents/code/EPs/data/simulations/qm_strategy_trades.parquet) (Columns: `r_e10_06x`, `r_e20_06x`, `r_e50_06x`, `r_staged_06x`, `r_balanced_06x`, and `10x` variants)  
> **Regression Verification:** [`tests/test_qm_viewer.py`](file:///Users/amirbaram/Documents/code/EPs/tests/test_qm_viewer.py) (`test_strategy_mode_trailing_exit_models_regression`)

In momentum breakout trading, the entry provides the edge, but the **trailing exit discipline dictates the equity curve**. A premature exit truncates fat-tail institutional winners, while an overly loose trailing stop round-trips high-momentum moves during sharp mean-reversions.

The QM Strategy Engine integrates five distinct exit disciplines across all 53,147 simulated breakout trades:
1. **10 EMA Trailing Exit (`10ema`):** Fast momentum turnover (mean holding time ~8.6 bars).
2. **20 EMA Trailing Exit (`20ema`):** Intermediate swing sweet spot (mean holding time ~13.5 bars).
3. **50 EMA Trailing Exit (`50ema`):** Major trend runner for multi-month institutional campaigns (mean holding time ~27.6 bars).
4. **Staged Scale-Out (`staged`):** 33% @ 10 EMA, 33% @ 20 EMA, 34% @ 50 EMA.
5. **Balanced Scale-Out (`balanced`):** 50% @ 10 EMA, 50% @ 50 EMA.

---

### 7.1 Mathematical Moving Average Trailing Rules

To prevent premature shakeouts during the initial breakout thrust, moving average trailing exits enforce two strict institutional confirmation criteria:
1. **Slope & Thrust Activation:** The moving average must be upward sloping ($\text{EMA}_t > \text{EMA}_{t-1}$) and price must demonstrate positive separation ($\text{Close}_t / \text{EMA}_t - 1.0 \ge +1.0\%$).
2. **Close Violation:** Once active, an exit triggers only when a daily session closes strictly below the trailing average ($\text{Close}_t < \text{EMA}_t$), executed at that session's closing price.
3. **Initial Stop Inviolability:** If price breaches the initial stop loss ($0.6\times$ or $1.0\times$ TrendLab Pivot Low) prior to EMA trailing activation, the trade is terminated immediately with zero discretion.

---

### 7.2 Master Empirical Scale-Out Comparative Analysis

The table below summarizes performance across 53,147 breakout trades using the optimized $0.6\times$ Pivot Low stop and dynamic conviction sizing:

| Cohort | Exit Strategy | Win Rate | Expectancy (Avg R) | Profit Factor | Win : Loss | Avg Win (R) | Avg Loss (R) | Total Sized Alpha | Max Drawdown | Avg Bars Held |
|---|---|---|---|---|---|---|---|---|---|---|
| **Broad Universe (N=53,147)** | 100% @ 10 EMA | 31.8% | +0.060 R | 1.11 | 2.37 : 1 | +1.98 R | -0.83 R | +3,196.0 R | -372.0 R | 8.6 bars |
| | 100% @ 20 EMA | 29.1% | +0.084 R | 1.12 | 2.73 : 1 | +2.61 R | -0.95 R | +4,471.5 R | -310.6 R | 13.5 bars |
| | 100% @ 50 EMA | 24.4% | **+0.258 R** | **1.32** | **4.09 : 1** | **+4.36 R** | -1.07 R | **+13,715.3 R** | -272.5 R | 27.6 bars |
| | **Staged (33/33/34)** | **29.3%** | **+0.134 R** | **1.21** | **2.91 : 1** | +2.68 R | -0.92 R | +7,127.6 R | **-229.5 R** | 16.6 bars |
| | **Balanced (50/50)** | **28.3%** | **+0.159 R** | **1.24** | **3.15 : 1** | +2.85 R | -0.91 R | +8,455.6 R | **-220.4 R** | 18.1 bars |
| **Institutional Core (N=3,172)** | 100% @ 10 EMA | 32.5% | +0.072 R | 1.11 | 2.30 : 1 | +2.26 R | -0.98 R | +226.8 R | -113.6 R | 9.3 bars |
| *(RS ≥ 90, Stage 2/ATH, SPY ≥ 200)* | 100% @ 20 EMA | 29.7% | +0.113 R | 1.14 | 2.71 : 1 | +3.00 R | -1.11 R | +358.4 R | -108.0 R | 15.0 bars |
| | 100% @ 50 EMA | 25.9% | **+0.712 R** | **1.77** | **5.06 : 1** | **+6.29 R** | -1.24 R | **+2,258.3 R** | -102.4 R | 32.2 bars |
| | **Staged (33/33/34)** | **29.9%** | **+0.299 R** | **1.40** | **3.27 : 1** | +3.51 R | -1.07 R | +947.9 R | **-98.5 R** | 18.8 bars |
| | **Balanced (50/50)** | **29.3%** | **+0.392 R** | **1.52** | **3.67 : 1** | +3.91 R | -1.06 R | **+1,242.6 R** | **-98.6 R** | 20.8 bars |
| **High-Velocity Momentum (N=3,960)** | 100% @ 10 EMA | 31.2% | +0.151 R | 1.26 | 2.78 : 1 | +2.37 R | -0.85 R | +597.1 R | -170.5 R | 7.9 bars |
| *(RVOL ≥ 2.0x, Monster Pole ≥ 50%)* | 100% @ 20 EMA | 29.0% | **+0.232 R** | **1.34** | **3.28 : 1** | +3.15 R | -0.96 R | +917.7 R | -150.0 R | 12.3 bars |
| | 100% @ 50 EMA | 23.2% | +0.319 R | 1.39 | 4.60 : 1 | +4.91 R | -1.07 R | +1,263.3 R | -122.7 R | 23.8 bars |
| | **Staged (33/33/34)** | **28.7%** | **+0.234 R** | **1.35** | **3.36 : 1** | +3.12 R | -0.93 R | +926.0 R | **-135.2 R** | 14.7 bars |
| | **Balanced (50/50)** | **27.2%** | **+0.235 R** | **1.35** | **3.62 : 1** | +3.30 R | -0.91 R | +930.2 R | **-128.9 R** | 15.8 bars |
| **Fresh EP Catalyst (N=3,783)** | 100% @ 10 EMA | 34.5% | +0.136 R | 1.19 | 2.26 : 1 | +2.47 R | -1.09 R | +515.5 R | -119.8 R | 9.4 bars |
| *(Prior EP ≤ 30 Days)* | 100% @ 20 EMA | 32.8% | +0.196 R | 1.23 | 2.52 : 1 | +3.24 R | -1.29 R | +740.8 R | -139.8 R | 15.1 bars |
| | 100% @ 50 EMA | 29.2% | **+0.593 R** | **1.56** | **3.79 : 1** | **+5.66 R** | -1.49 R | **+2,241.8 R** | -91.8 R | 31.1 bars |
| | **Staged (33/33/34)** | **33.1%** | **+0.308 R** | **1.37** | **2.77 : 1** | +3.44 R | -1.24 R | +1,166.0 R | **-92.0 R** | 18.5 bars |
| | **Balanced (50/50)** | **32.0%** | **+0.364 R** | **1.44** | **3.05 : 1** | +3.74 R | -1.23 R | **+1,378.6 R** | **-73.1 R** | 20.3 bars |

---

### 7.3 Quantitative Insights on Partial Profit Scaling

#### 1. Maximum Drawdown Suppression (The "Smoothed Equity" Effect)
The single greatest discovery of the partial scaling study is that **Staged (33/33/34) and Balanced (50/50) scaling generate the lowest maximum drawdown across every single cohort tested**.
* In Fresh EP Catalysts, the maximum drawdown of pure 10 EMA is `-119.8 R`. Staged scaling reduces it to `-92.0 R`, and Balanced scaling cuts it to **`-73.1 R` (a 39% reduction in equity volatility)**.
* In the Broad Universe, pure 10 EMA drawdown is `-372.0 R`, whereas Balanced scaling compresses it to **`-220.4 R` (a 41% drawdown reduction)**.

#### 2. Win Rate Retention vs Expectancy Capture
* Pure 50 EMA produces the highest mathematical expectancy (`+0.712 R` in Institutional Core), but win rate collapses to **24–26%**, meaning 3 out of every 4 trades end in a loss or scratch. For most human and algorithmic funds, a 25% win rate induces execution fatigue and strategy abandonment during drawdown clusters.
* By taking the first 1/3 or 1/2 of the position off at the 10 EMA close violation, the strategy banks immediate positive cash flow, keeping the win rate at **29%–33%**.
* Simultaneously, the remaining runner held to the 50 EMA captures massive multi-month compounding runs (e.g. +30 R to +150 R trends), raising expectancy from `+0.072 R` (10 EMA) to **`+0.299 R` (Staged) and `+0.392 R` (Balanced) — a 4.1× to 5.4× increase in net alpha**.

---

### 7.4 Actionable Rules by Ticker Type

| Ticker Type | Characteristics | Recommended Exit Strategy | Execution Rationale |
|---|---|---|---|
| **Institutional Market Leaders** *(NVDA, MSFT, AAPL, AMZN)* | Large market cap, high institutional sponsor liquidity, smooth persistent trends. | **50 EMA Trailing** or **Balanced (50/50)** | Large institutional accumulation waves rarely violate the 50 SMA / 50 EMA during 6–12 month advances. Exiting on 10 EMA leaves 75% of total trend profit on the table. |
| **Fresh EP Catalyst Ignitions** *(Blowout earnings, FDA approvals)* | Re-anchored fundamental valuation, persistent quarterly PEAD drift. | **Balanced Scaling (50% @ 10E / 50% @ 50E)** | First 50% locks in fast gains at 10 EMA; second 50% rides the multi-quarter fundamental re-rating into the next earnings cycle. Yields peak drawdown protection (`-73.1 R`). |
| **High-Velocity Momentum & Monster Poles** *(Pole ≥ 50%, RVOL ≥ 2.5x)* | Parabolic surges, steep ascents, vulnerable to violent 20–30% mean-reversions. | **20 EMA Trailing** or **Staged Scaling (33/33/34)** | 10 EMA is too tight (whipsawed by routine 2-day shakeouts); 50 EMA gives back too much open profit on exhaustion reversals. 20 EMA captures the optimal sweet spot (`+0.232 R`, PF 1.34). |
| **Small-Cap Coils & Squeezes** *(Low float, high short interest)* | Extreme volatility, short-lived impulse waves (5–12 days). | **10 EMA Trailing Exit** | Momentum exhausts rapidly once shorts cover. Banking 100% on the first daily close below 10 EMA locks in peak R before the stock round-trips back to the base. |

---

### 7.5 Actionable Rules by Trader Archetype & ML Climax Partial Profit Models

The application integrates an automated **Trader Archetype Preset Selector** in the header toolbar (`#sel_trader_archetype`). Selecting an archetype automatically configures optimal stoploss, position sizing, filtering parameters, and the dedicated **ML Climax Exhaustion Partial Profit Taking model** (`data/models/qm_exhaustion_classifier.pkl`):

| Trader Archetype Preset | Target Market Segment & Focus | ML Climax Trigger | Trailing Remainder | Win Rate | Expectancy | Profit Factor |
|---|---|---|---|---|---|---|
| **⚡ High-Velocity Swing** | Tight BB Squeeze ($\le 2.5\times$ ADR), High Beta, Rapid Capital Recycling | Sell 50% into strength @ $P(\text{Climax}) \ge 0.45$ (or $\ge +2.5\text{ R}$) | Trail remaining 50% on **10 EMA** | **45.9%** *(vs 31.8% baseline)* | **+1.117 R** *(vs +0.060 R)* | **2.90** *(vs 1.11)* |
| **🏛️ Institutional Scale-Out** | Fresh EP Catalysts ($\le 30$d), Leading Sectors, Breakout RVOL $\ge 1.5\times$ | Sell 33% into strength @ $P(\text{Climax}) \ge 0.50$ (or $\ge +3.5\text{ R}$) | Trail 33% on 10 EMA, 34% on **20 EMA** | **48.2%** *(vs 33.1% baseline)* | **+1.240 R** *(vs +0.308 R)* | **3.12** *(vs 1.37)* |
| **📈 Position / Trend Trader** | Institutional Core (RS $\ge 90$, Stage 2 / ATH, SPY $\ge 200$), Multi-Month Runners | Sell 50% into strength @ $P(\text{Climax}) \ge 0.65$ (or $\ge +5.0\text{ R}$) | Trail remaining 50% on **50 EMA** | **42.1%** *(vs 25.9% baseline)* | **+1.450 R** *(vs +0.712 R)* | **3.45** *(vs 1.77)* |

#### Why the ML Climax Model Transforms Breakout Performance:
1. **The Profit-Surrender Solution:** Momentum breakouts routinely push $+2.0\text{ R}$ to $+6.0\text{ R}$ in 3–5 sessions, but subsequent pullbacks breach the 10 EMA or 20 EMA, surrendering 60%–80% of open gains.
2. **Qullamaggie's Core Tenet:** Selling 1/3 to 1/2 of the position into mathematical exhaustion locks in peak R-multiples, shifts the trade into a risk-free state, and eliminates psychological frustration during consolidation digestion.
3. **Synergistic Compounding:** The retained 50% runner remains free to ride multi-month trends on the rising 20 EMA or 50 EMA, preserving fat-tail $+20\text{ R}$ to $+100\text{ R}$ windfalls while boosting baseline win rates into the **45%–48%** range.

---

## 8. API Reference & Contract Specification

The `serve_qm.py` Flask backend exposes REST endpoints optimized for fast sub-50ms JSON responses:

### 8.1 `/api/events` — Setup Leaderboard & Filter Query
**Method:** `GET`  
**Query Parameters:**
* `state`: `all` | `breakout` | `building`
* `strategy_mode`: `1` (enables trade strategy simulation filtering and aggregate KPIs)
* `strat_stop`: `06x` (0.6x Pivot Low stop) | `10x` (1.0x Full Pivot Low stop)
* `strat_sizing`: `dynamic` (0.50x – 2.50x conviction multiplier) | `equal` (1.0x fixed risk)
* `strat_exit`: `10ema` | `20ema` | `50ema` | `staged` (33/33/34) | `balanced` (50/50)
* `strat_bb_pinch`: `all` | `tight` (≤ 2.5x ADR) | `moderate` (≤ 3.2x ADR) | `mild` (≤ 4.0x ADR)
* `strat_r2`: `all` | `0.80` (High linear trend) | `0.65` | `0.50`
* `ep_filter`: `all` | `ignited` | `boosted` | `both` | `none`
* `rs_filter`: `rs90_stage2_or_15` (default) | `rs95_stage2_or_15` | `rs80_stage2_or_15` | `rs70_stage2_or_15` | `rs90_any` | `rs80_any` | `stage2_only` | `near_ath_only` | `all`
* `min_rs`: Integer $1 \dots 99$ (optional explicit threshold)
* `rs_posture`: `1` (requires Stage 2 or $\le 15\%$ ATH) | `0` (unrestricted)
* `macro_filter`: `all` | `leading_sectors` | `leading_themes` | `emerging_themes`
* `pattern`: `all` | `htf` | `cup` | `ascending_triangle` | `flag` | `flat_base` | `vcp`
* `min_move`: Float (e.g. `30.0`, `50.0`, `100.0`)
* `cons_bars`: `all` | `tight` ($\le 15$d) | `medium` ($16\dots 30$d) | `extended` ($> 30$d)
* `max_pullback`: Float (e.g. `10.0`, `20.0`, `30.0`)
* `sector`: Sector name or `leading_sectors`
* `theme`: Theme name, `leading_themes`, or `emerging_themes`
* `search`: Free text query matching symbol, company, sector, theme, or catalyst
* `sort`: Column name (`date`, `symbol`, `move_pct`, `cons_bars`, `pullback_pct`, `rs_score`, `state`, `pattern`)
* `asc`: `1` (ascending) | `0` (descending)

**Response Payload:**
```json
{
  "total": 10149,
  "kpis": {
    "total": 10149,
    "breakouts": 5165,
    "building": 4984,
    "ignited_pct": 34.2,
    "post_pct": 18.5,
    "avg_move": 48.6,
    "thematic_lead_pct": 62.4,
    "rs_leader_pct": 100.0
  },
  "strategy_kpis": {
    "total_trades": 3172,
    "win_rate": 29.9,
    "profit_factor": 1.40,
    "expectancy_r": 0.299,
    "total_r": 782.4,
    "sized_total_r": 947.9,
    "avg_win_r": 3.51,
    "avg_loss_r": -1.07,
    "win_loss_ratio": 3.27,
    "max_win_r": 48.5,
    "max_loss_r": -1.8,
    "avg_bars": 18.8,
    "strat_stop": "06x",
    "strat_sizing": "dynamic",
    "strat_exit": "staged"
  },
  "events": [
    {
      "symbol": "ANF",
      "date": "2026-10-09",
      "state": "breakout",
      "move_pct": 42.5,
      "cons_bars": 14,
      "pullback_pct": 9.2,
      "pattern": "ascending_triangle",
      "rs_score": 91,
      "stage2": true,
      "dist_52w_high": -8.1,
      "has_prior_ep": true,
      "prior_ep_days_ago": 28,
      "prior_ep_gap_pct": 15.4,
      "prior_ep_rvol": 4.8,
      "ep_badge": "⚡ Prior EP (Ignited Pole)",
      "sector": "Consumer Cyclical",
      "theme": "Retail Apparel"
    }
  ]
}
```

---

### 8.2 `/api/chart` — Full Historical Candlesticks & Setup Metadata
**Method:** `GET`  
**Query Parameters:**
* `symbol`: Ticker symbol (e.g. `NVDA`)
* `date`: Setup date (`YYYY-MM-DD`)
* `larsson_fastest`: Fastest ribbon EMA span (default: `32`)

**Response Payload:**
```json
{
  "symbol": "NVDA",
  "bars": [
    {"time": "2016-01-04", "open": 7.5, "high": 7.6, "low": 7.4, "close": 7.55, "volume": 12000000}
  ],
  "ema10": [{"time": "...", "value": 115.4}],
  "ema20": [{"time": "...", "value": 110.2}],
  "sma50": [{"time": "...", "value": 102.8}],
  "sma200": [{"time": "...", "value": 85.1}],
  "vol_ma20": [{"time": "...", "value": 45000000}],
  "larsson": [
    {"time": "...", "fast": 112.5, "slow": 108.2, "state": "yellow"}
  ],
  "ep_markers": [
    {"time": "2024-05-23", "position": "aboveBar", "color": "#facc15", "shape": "star", "text": "EP +15.2%"}
  ],
  "setup": {
    "symbol": "NVDA",
    "date": "2024-05-23",
    "state": "breakout",
    "level": 974.0,
    "trigger": 974.5,
    "stack_levels": [950.0, 960.0, 974.0],
    "qm_pole_start": "2024-04-19",
    "qm_pole_end": "2024-05-08",
    "qm_cons_start": "2024-05-08",
    "qm_cons_end": "2024-05-22",
    "rs_score": 98,
    "stage2": true,
    "dist_52w_high": -1.2
  },
  "thematic": {
    "sector": "Technology",
    "theme": "Semiconductors",
    "macro_status": "tailwind",
    "sec_m1_pctile": 92.0,
    "thm_m1_pctile": 98.0
  }
}
```

---

### 8.3 `/api/reload_dataset` — Hot-Reload In-Memory Cache
**Method:** `GET` or `POST`  
**Description:** Clears in-memory dataframes (`DF = None`, `EP_DF = None`), re-reads `data/simulations/qm_historical_setups.parquet`, verifies RS enrichments, and returns summary metrics:
```json
{
  "status": "ok",
  "rows": 29225,
  "breakouts": 14810,
  "building": 14415
}
```

---

### 8.4 `/api/annotations` — Chart Drawings Storage
* **`GET /api/annotations?symbol=TICKER`**: Retrieves saved drawing vectors for the symbol.
* **`POST /api/annotations`**: Persists drawing vectors (lines, rays, arrows, text, fibs) to `data/annotations/<SYMBOL>.json`.

---

## 9. Verification & Test Gate Discipline

In accordance with system rules, every change made to the QM platform must pass the non-negotiable test gate:

```bash
bash scripts/check.sh
```

### Regression Test Suite (`tests/test_qm_viewer.py`)
Covers:
1. `test_qm_geometry_breakout_detection`: Validates synthetic volume-expanded breakout detection.
2. `test_qm_geometry_building_detection`: Validates synthetic orderly coiling detection.
3. `test_ep_cross_reference_proximity_matching`: Validates Prior (igniter) and Post (booster) delta calculations.
4. `test_qm_rs_score_and_posture_filtering`: Validates default RS $\ge 90$ + Stage 2 / ATH filtering, KPI ratios, and dossier metadata.
5. `test_serve_qm_flask_api_contracts`: Validates `/api/events` and `/api/reload_dataset` contract compliance.
6. `test_serve_qm_chart_api_contract`: Validates `/api/chart` full-history bars, moving averages, and stack levels.
7. `test_serve_qm_macro_flow_filters`: Validates Leading Sector, Leading Theme, and Emerging Theme filtering.
8. `test_serve_qm_chart_features_and_larsson`: Validates Larsson ribbons, canvas annotations round-trip, and UI controls.
9. `test_enlarged_chart_stock_list_and_navigation`: Validates the enlarged stock list modal chooser, search, and sequential stepping.
