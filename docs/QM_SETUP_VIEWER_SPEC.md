# QM Setup Viewer & EP Cross-Referencing System: Technical Specification & Implementation Guide

## 1. Executive Summary & Objective

The goal is to build a high-performance interactive historical setup viewer in `/Users/amirbaram/Documents/code/EPs` (dedicated server, e.g., `serve_qm.py` on port **8785**), modeled directly on the architecture and visual polish of the modern `serve_ep.py` (port 8782).

This application allows discretionary traders and researchers to:
1. **Explore Historical QM (Qullamaggie Momentum Continuation) Setups:** Inspect historical setups across US common stocks, visualizing the momentum pole, consolidation coiling, moving average surfing (EMA10/EMA20/SMA50), and breakout stack trigger levels.
2. **Highlight EP (Episodic Pivot) Relationships:** Cross-reference every QM setup against the historical Episodic Pivot dataset to immediately identify whether a QM setup was ignited by a prior EP (within 60 days before) or subsequently fueled by an EP (within 60 days after).
3. **Pure Setup Visualizer (Zero Simulated Trades):** Unlike `serve_ep.py`, this app does **NOT** simulate trade executions, stops, trailing stops, or R-multiples. It is dedicated to pure pattern recognition, setup inspection, breakout geometry, and catalyst alignment.

---

## 2. Technical Definition of the QM Setup (from `Downloads/scan`)

The QM setup is Kristjan Qullamaggie's momentum continuation setup. The core logic resides in `/Users/amirbaram/Downloads/scan/setups.py` (`detect_qm_breakout`) and is governed by parameters in `/Users/amirbaram/Downloads/scan/config.py`.

### A. Mathematical Criteria & Rules

| Step | Rule | Logic & Config Thresholds | Plain-English Meaning |
|---|---|---|---|
| **1. Momentum Pole** | `move >= QMB_MIN_MOVE` | Low-to-high gain $\ge 30.0\%$ over $\le 35$ bars (`QMB_MOVE_MAX_BARS`) within the last 63 bars (`QMB_MOVE_LOOKBACK`). | The stock experienced explosive institutional accumulation (the "pole"). |
| **2. Orderly Consolidation** | `QMB_CONS_MIN <= cons_bars <= QMB_CONS_MAX` | Consolidation duration between **8 and 45 trading sessions** from the pole high. | Stock pauses to digest gains; avoids V-shaped reversals. |
| **3. Controlled Pullback** | `pullback <= QMB_MAX_PULLBACK` | Maximum drawdown from pole high $\le 50.0\%$ (`(1 - cons_low / hi_p) * 100`). | Sellers cannot push price down significantly; buyers support the base. |
| **4. EMA Surfing** | `_qm_ride(close, low, e10, e20, s50)` | EMA10 > EMA20 and both sloping upward. Price closes within 3.0% of EMA20 (`QMB_SURF_BAND`). $\ge 50\%$ of consolidation bars must ride EMA20 (`QMB_MIN_RIDE_FRAC`). **No close below 50 SMA**. | Institutional accumulation keeps price glued to rising short-term moving averages. |
| **5. Higher Lows Structure** | `_qm_higher_lows(d, lo_start, rend)` | Swing lows from the consolidation low to the latest bar must form ascending higher lows. | Supply is drying up; buyers step in at progressively higher prices. |
| **6. Volume Contraction** | `QMB_CONTRACT_WIN = 10` | Average volume over the prior 10 bars contracts compared to the pole volume. | Selling volume dries up as the base coils tightly. |
| **7. Breakout Stack & Trigger** | `_breakout_stack(...)` | Identifies local swing high resistance levels in the base. The **trigger** is the nearest resistance level. | A dynamic horizontal line or trendline defining the actionable breakout point. |
| **8. Breakout Bar** | `volA >= QMB_BREAKOUT_VOL * vb` | Breakout bar volume $\ge 1.3\times$ recent consolidation average volume (`QMB_BREAKOUT_VOL = 1.3`). State switches from `building` to `breakout`. | Confirmed expansion of price and volume out of the coil. |
| **9. Pattern Classification** | `patterns.classify_consolidation(...)` | Least-squares fit of upper and lower trendlines: `flag`, `pennant`, `rising_triangle`, `channel`. | Visual geometric classification of the base. |

### B. Core Functions to Extract / Copy
The new app requires these exact functions:
1. `detect_qm_breakout(d: pd.DataFrame)` (from `setups.py` lines 816–906)
2. `_qm_ride(...)` (from `setups.py` lines 769–803)
3. `_qm_higher_lows(...)` (from `setups.py` lines 805–814)
4. `_breakout_stack(...)` (from `setups.py` lines 468–525)
5. `_project_last_volume(...)` (from `setups.py` lines 448–466)
6. `patterns.classify_consolidation(...)` (from `patterns.py`)
7. `indicators.add_indicators(df)` (from `indicators.py`)

---

## 3. EP (Episodic Pivot) Cross-Referencing Logic

In `Documents/code/EPs`, the historical Episodic Pivot dataset lives at:
`data/simulations/ep_combined_study_scored.parquet` (6,501 curated EP events across 10 years).

### A. Proximity Correlation Windows
For every detected QM setup on ticker `SYMBOL` at date `T_QM`:
1. **Prior EP Catalyst ($[-60, 0]$ days before $T_{QM}$):**
   - Did an EP occur between $T_{QM} - 60\text{ days}$ and $T_{QM}$?
   - **Meaning:** The entire QM pole and subsequent base were directly ignited by an institutional Episodic Pivot!
   - Attributes captured: `prior_ep_date`, `prior_ep_days_ago`, `prior_ep_gap_pct`, `prior_ep_rvol`, `prior_ep_subtype` (Pinnacle Elite vs Classic), `prior_ep_catalyst`.
2. **Post EP Catalyst ($[0, +60]$ days after $T_{QM}$):**
   - Did an EP occur between $T_{QM}$ and $T_{QM} + 60\text{ days}$?
   - **Meaning:** A fresh institutional earnings or news gap occurred immediately following the QM consolidation, providing secondary rocket fuel.
   - Attributes captured: `post_ep_date`, `post_ep_days_later`, `post_ep_gap_pct`, `post_ep_rvol`, `post_ep_subtype`.

### B. Classification Badges for UI
- ⚡ **EP-Ignited Pole:** Prior EP within 45 trading days. (Highest conviction institutional momentum continuation).
- 🚀 **EP-Boosted Breakout:** Post EP within 20 trading days of breakout.
- ⚪ **Pure Technical QM:** No EP within 60 days (technical momentum without major catalyst gap).

---

## 4. Modern UI & Server Architecture (Modeled on 8782 in `Documents/code/EPs`)

The new application (`serve_qm.py`) should run on port **8785** and follow the exact 3-pane layout of `serve_ep.py`:

```
+---------------------------------------------------------------------------------------------------------+
| [⚡ QM MOMENTUM VIEWER]   Total: 3,420 | Breakouts: 1,180 | Building: 2,240 | EP-Ignited: 42.8%          |
+---------------------------------------------------------------------------------------------------------+
| [FILTERS: State | EP Link | Min Move % | Cons Days | Pattern | Sector/Theme | Search Symbol]            |
+------------------------------------+-------------------------------------------+------------------------+
| MAIN TABLE (Sortable / Filterable) | CENTER PANE: FULL-HEIGHT CHART           | RIGHT PANE: DOSSIER    |
| - Symbol & Date                    | - Candlesticks (OHLC)                     | 1. QM Technical Profile|
| - State (Breakout / Building)      | - EMA 10 (Blue), EMA 20 (Yellow)          |    - Pole Gain & Bars  |
| - Pole Move % & Duration           | - SMA 50 (Red), SMA 200 (Gray)            |    - Cons Length & Low |
| - Cons Bars & Pullback %           | - Volume Bars                             |    - Ride Frac & Bands |
| - Pattern Shape                    | - QM Overlays:                            |    - Breakout Vol Ratio|
| - EP Linkage Badges:               |   * Shaded Pole Box (Green)               | 2. EP Catalyst Linkage |
|   * [⚡ Prior EP: 24d ago (+18%)]   |   * Shaded Consolidation Box (Yellow)     |    * Prior EP Date/Gap |
|   * [🚀 Post EP: 12d later (+22%)] |   * Horizontal Breakout Stack Lines       |    * Catalyst Headline |
|                                    |   * Vertical Marker on Prior/Post EP Bar  | 3. Sector & Theme RS   |
+------------------------------------+-------------------------------------------+------------------------+
```

### Key UI Features (Using LightweightCharts + Tailwind CSS):
- **TradingView / LightweightCharts Candlestick Engine:** Ultra-fast, interactive zooming, panning, and tooltips.
- **QM Geometry Overlays:**
  - `qm_pole`: Shaded rectangle covering low-to-high pole bars.
  - `qm_cons`: Shaded rectangle covering the consolidation range.
  - `stack_levels`: Horizontal dashed or solid lines showing the resistance levels and top breakout trigger.
- **EP Catalyst Marker:** A distinct bright cyan or purple marker on the candlestick bar where an Episodic Pivot occurred, allowing the trader to visually link the catalyst to the consolidation.
- **No Simulated Trades:** No trade R-multiples, trade exit lines, or simulated profit charts. Focus entirely on the setup quality, geometry, and catalyst alignment.

---

## 5. Mining & Dataset Generation Pipeline (`mine_qm_setups.py`)

To ensure the web app launches instantly with zero lag, historical QM setups should be precomputed into a clean Parquet dataset:
`data/simulations/qm_historical_setups.parquet`.

### Precomputation Steps:
1. Load common stock bars from `data/bars/*.parquet`.
2. Compute indicators: `indicators.add_indicators(df)`.
3. Detect QM setups using `detect_qm_breakout(df.iloc[:i])`.
4. Debounce building states: Collapse consecutive daily building bars into unique consolidation episodes, recording:
   - Initial breakout date (if broken out).
   - Peak consolidation date / terminal building date.
   - Geometry: pole start, pole end, cons start, cons end, levels.
5. Cross-reference `data/simulations/ep_combined_study_scored.parquet`:
   - Match by `symbol` and compute date delta to find closest prior EP and closest post EP.
6. Save to `data/simulations/qm_historical_setups.parquet`.

---

## 6. Prompt for the New Session Agent

Copy and paste the exact prompt below into your new session:

```markdown
You are a quantitative systems developer working in `/Users/amirbaram/Documents/code/EPs` (active git branch: `feature/dual-track-continuation`).

TASK:
Build a new standalone server application in `Documents/code/EPs` (suggested file: `serve_qm.py` running on port 8785) to view historical QM (Qullamaggie Momentum Continuation) setups, highlighting Episodic Pivot (EP) setups that occurred recently before or after each QM setup, without simulated trades.

CRITICAL WORKSPACE RULES:
1. All new code, datasets, scripts, and tests must reside exclusively in `/Users/amirbaram/Documents/code/EPs`.
2. Do NOT run commands or modify files in `Downloads/scan` (you may only read from it as reference if needed).
3. Strictly use `.venv/bin/python` in `/Users/amirbaram/Documents/code/EPs`.
4. Ensure the fast gate `bash scripts/check.sh` remains 100% green at all times.

STEP 1: REVIEW THE SPECIFICATION & CODEBASE
1. Read the full specification in `docs/QM_SETUP_VIEWER_SPEC.md`.
2. Review the QM setup detection logic in `setups.py` (`detect_qm_breakout`, `_qm_ride`, `_qm_higher_lows`, `_breakout_stack`) and `config.py` (`QMB_*` parameters).
3. Review the modern UI architecture in `serve_ep.py` (port 8782): inspect its 3-pane layout (table on left, LightweightCharts candlestick chart in center, dossier card on right, Tailwind CSS, Lucide icons, `/api/events` and `/api/chart` endpoints).

STEP 2: PRECOMPUTE HISTORICAL QM SETUPS & EP CROSS-REFERENCE
1. Create a mining script `mine_qm_setups.py` in `Documents/code/EPs`:
   - Iterate through historical daily bars in `data/bars/*.parquet`.
   - Run `detect_qm_breakout` to capture historical QM breakout and mature building setups.
   - Cross-reference with `data/simulations/ep_combined_study_scored.parquet` to calculate prior EP proximity (within 60 days before) and post EP proximity (within 60 days after).
   - Save the precomputed dataset to `data/simulations/qm_historical_setups.parquet`.
2. Test the mining script on a liquid subset first, then generate the full dataset.

STEP 3: BUILD THE QM SETUP VIEWER SERVER (`serve_qm.py`)
1. Implement `serve_qm.py` (running on port 8785):
   - Fast endpoint `/api/events`: filterable by state (`breakout` vs `building`), EP linkage (Has Prior EP, Has Post EP, No EP), pattern shape, move %, consolidation length, sector, theme, and date range.
   - Fast endpoint `/api/chart`: returns OHLCV candlestick bars, moving averages (EMA10, EMA20, SMA50), QM geometry overlays (shaded pole box, shaded consolidation box, breakout stack resistance lines), and EP markers.
   - Comprehensive Dossier card: QM Technical Profile + EP Catalyst Linkage Card + Sector/Theme backdrop.
   - No simulated trades: omit trade simulations, R-multiples, and stops.

STEP 4: VERIFICATION & REGRESSION TESTING
1. Add regression tests in `tests/` verifying QM detection and EP linkage.
2. Run `bash scripts/check.sh` and ensure all tests pass (0 failures).
3. Launch `serve_qm.py` on port 8785 and smoke test live endpoints.
4. Document the new application in `README.md` and `DOCS.md`.
```
