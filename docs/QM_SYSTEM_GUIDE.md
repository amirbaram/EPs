# QM Setup & EP Cross-Reference Explorer (`serve_qm.py` — Port 8785)
## Quantitative Architecture, Setup Mechanics & Data Pipeline Maintenance Guide

> **Production Application:** `serve_qm.py` running on **http://127.0.0.1:8785**  
> **Repository:** `amirbaram/EPs` (Branch: `feature/dual-track-continuation`)  
> **Mining Pipeline:** [`mine_qm_setups.py`](file:///Users/amirbaram/Documents/code/EPs/mine_qm_setups.py)  
> **Automated Updater:** [`update_qm_database.py`](file:///Users/amirbaram/Documents/code/EPs/update_qm_database.py)  
> **Precomputed Dataset:** `data/simulations/qm_historical_setups.parquet` (29,225 historical setups)  
> **Primary Audience:** Quantitative researchers, autonomous trading agents, and institutional developers.

---

## 1. Executive Summary & Design Philosophy

The **QM Setup & EP Cross-Reference Explorer** is a high-performance, institutional web application designed to study and monitor **Qullamaggie Momentum Continuation (QM)** setups across the entire US equity universe. 

Unlike the core EP Strategy Backtester (`serve_ep.py` on port 8782), which executes simulated trades, tracks stop losses, and models dynamic R-multiples, the 8785 application is a **pure setup viewer and technical pattern intelligence platform**:
* **No Simulated Trades:** Omits artificial stops, exits, and PnL curves, providing an unobstructed view of the underlying price action and base structure.
* **Dual Setup States:** Captures both **confirmed breakout setups** (volume-expanded breakouts above multi-week resistance) and **mature building coils** (orderly consolidations surfing rising EMAs with higher lows preparing to break out).
* **Episodic Pivot (EP) Cross-Referencing:** Quantifies whether a QM momentum pole was ignited by a high-conviction fundamental EP gap within the prior 60 days, or boosted by a subsequent catalyst gap within 60 days after base formation.
* **Institutional Relative Strength (RS) & Posture Filtering:** Integrates the 10-year daily RS database (`data/rs_history/`, 7.1M records) to screen for market leaders ($\text{rs\_score} \ge 90$) in **Stage 2 Bullish Alignment** or within 15% of 52-week highs.
* **Macro & Thematic Momentum:** Links setups to real-time sector and theme percentiles (Leading Sectors, Leading Themes, Emerging Themes).
* **Interactive Charting Suite:** Features TradingView-style LightweightCharts with canvas-shaded pole and base overlays, Breakout Stack resistance levels, Larsson multi-span ribbons, persistent drawing tools, and an enlarged chart stock chooser modal with sequential stepping.

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

## 7. API Reference & Contract Specification

The `serve_qm.py` Flask backend exposes REST endpoints optimized for fast sub-50ms JSON responses:

### 7.1 `/api/events` — Setup Leaderboard & Filter Query
**Method:** `GET`  
**Query Parameters:**
* `state`: `all` | `breakout` | `building`
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

### 7.2 `/api/chart` — Full Historical Candlesticks & Setup Metadata
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

### 7.3 `/api/reload_dataset` — Hot-Reload In-Memory Cache
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

### 7.4 `/api/annotations` — Chart Drawings Storage
* **`GET /api/annotations?symbol=TICKER`**: Retrieves saved drawing vectors for the symbol.
* **`POST /api/annotations`**: Persists drawing vectors (lines, rays, arrows, text, fibs) to `data/annotations/<SYMBOL>.json`.

---

## 8. Verification & Test Gate Discipline

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
