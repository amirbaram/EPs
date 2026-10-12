# Institutional Relative Strength (RS) Database Specification & Guide

> **For Autonomous Agents & Developers Working in `amirbaram/EPs`**  
> **Status:** Production-grade Institutional Database  
> **Location:** `data/rs_history/rs_<YEAR>.parquet` & `data/simulations/rs_universe_scores.parquet`  
> **Engine & Query API:** [`build_rs_history.py`](file:///Users/amirbaram/Documents/code/EPs/build_rs_history.py)  
> **Automated Updater:** [`update_rs_database.py`](file:///Users/amirbaram/Documents/code/EPs/update_rs_database.py)  
> **Explorer Dashboard:** [`serve_rs.py`](file:///Users/amirbaram/Documents/code/EPs/serve_rs.py) (Port 8786)

---

## 1. Overview & Purpose

The **Institutional Relative Strength (RS) Database** precomputes daily William O'Neil / Investor's Business Daily (IBD) style **1–99 cross-sectional Relative Strength percentile ratings** and multi-timeframe posture metrics for the entire US equity universe (common stocks and ETFs) across 10+ years of history (2016–present, **2,519+ trading sessions**).

### Why This Database Exists
1. **Zero Lookahead Bias in Backtesting:** When evaluating an Episodic Pivot (EP), Quarter-Muncher (QM), or Dual-Track continuation setup from a past year (e.g., May 2024 or November 2021), an agent can instantly lookup the exact RS rating and technical posture as of that specific historical session.
2. **Sub-10ms Lookup Latency:** Single-day market-wide queries execute in **~5 to 7 milliseconds** using PyArrow predicate pushdown without loading entire yearly partitions.
3. **Decoupled Architecture:** Eliminates expensive on-the-fly indicator calculations in dashboards, scanners, and ML feature extractors.

---

## 2. Directory Layout & File Storage

```
data/
├── rs_history/
│   ├── rs_2016.parquet      # 70k+ rows (Oct-Dec 2016)
│   ├── rs_2017.parquet      # 383k rows
│   ├── rs_2018.parquet      # 403k rows
│   ├── rs_2019.parquet      # 427k rows
│   ├── rs_2020.parquet      # 450k rows
│   ├── rs_2021.parquet      # 496k rows
│   ├── rs_2022.parquet      # 537k rows
│   ├── rs_2023.parquet      # 759k rows
│   ├── rs_2024.parquet      # 1.30M rows
│   ├── rs_2025.parquet      # 1.38M rows
│   └── rs_2026.parquet      # 886k+ rows (Jan-Oct 2026)
└── simulations/
    └── rs_universe_scores.parquet  # Latest active session snapshot (~2,085 active rows)
```

* **Storage Footprint:** ~50–60 MB total for the full 10-year dataset (~7.1 million records).
* **Compression:** Zstandard (`zstd`) with 100,000-row parquet row groups for fast date-partitioned predicate pushdown.

---

## 3. Mathematical Formula & Methodology

### 3.1 Four-Quarter Weighted Price Momentum
For each symbol on each trading day $t$:
* $R_{1d} = (C_t / C_{t-1} - 1) \times 100$
* $R_{1w} = (C_t / C_{t-5} - 1) \times 100$
* $R_{1m} = (C_t / C_{t-21} - 1) \times 100$
* $R_{q1} = R_{3m} = (C_t / C_{t-63} - 1) \times 100$ *(40% weight — most recent quarter)*
* $R_{q2} = R_{6m} = (C_t / C_{t-126} - 1) \times 100$ *(20% weight)*
* $R_{q3} = R_{9m} = (C_t / C_{t-189} - 1) \times 100$ *(20% weight)*
* $R_{q4} = R_{1y} = (C_t / C_{t-252} - 1) \times 100$ *(20% weight)*

$$\text{Raw RS} = 0.40 \cdot R_{3m} + 0.20 \cdot R_{6m} + 0.20 \cdot R_{9m} + 0.20 \cdot R_{1y}$$

*(Graceful fallback for newer listings: if $n < 252$ bars, $R_{1y}$ defaults to $R_{9m}$; if $n < 189$, $R_{9m}$ defaults to $R_{6m}$; if $n < 126$, $R_{6m}$ defaults to $R_{3m}$.)*

### 3.2 Cross-Sectional Percentile Ranking (1 to 99)
On each trading date $t$, active securities are partitioned into **Common Stocks** and **ETFs** to prevent leveraged ETFs from distorting equity percentiles:

$$\text{rs\_score} = \text{round}\left( \text{rank}_{\text{pct}}(\text{Raw RS}) \times 98 + 1 \right) \in [1, 99]$$

* **Rank 99:** Top 1% market-wide price performers.
* **Rank 90–98:** Institutional Leader cohort (top decile).
* **Rank 1–20:** Lagging performers / severe downtrends.

### 3.3 Alpha vs. SPY Benchmark
* $\text{alpha\_1m} = R_{1m} - R_{1m}^{\text{SPY}}$
* $\text{alpha\_3m} = R_{3m} - R_{3m}^{\text{SPY}}$
* $\text{alpha\_1y} = R_{1y} - R_{1y}^{\text{SPY}}$

---

## 4. Schema & Data Dictionary

| Column | Type | Description |
|---|---|---|
| `date` | `datetime64[ns]` | Trading session date (e.g. `2024-05-23`). |
| `symbol` | `string` | Ticker symbol (e.g. `NVDA`). |
| `name` | `string` | Company or fund description. |
| `is_etf` | `bool` | `True` for ETFs, `False` for common stocks. |
| `close` | `float32` | Split-adjusted session close price. |
| `rs_score` | `int16` | **Cross-sectional Relative Strength rating (1 to 99).** |
| `rank` | `int16` | Absolute market rank on this session (`1` = highest RS score). |
| `raw_rs` | `float32` | 4-quarter weighted momentum return (%). |
| `ret_1d` | `float32` | 1-session return (%). |
| `ret_1w` | `float32` | 5-session (1-week) return (%). |
| `ret_1m` | `float32` | 21-session (1-month) return (%). |
| `ret_3m` | `float32` | 63-session (3-month / Q1) return (%). |
| `ret_6m` | `float32` | 126-session (6-month / Q2) return (%). |
| `ret_1y` | `float32` | 252-session (1-year / Q4) return (%). |
| `alpha_1m` | `float32` | 1-month excess return vs SPY benchmark (%). |
| `alpha_3m` | `float32` | 3-month excess return vs SPY benchmark (%). |
| `alpha_1y` | `float32` | 1-year excess return vs SPY benchmark (%). |
| `hi_52w` | `float32` | Trailing 252-day highest intraday high. |
| `lo_52w` | `float32` | Trailing 252-day lowest intraday low. |
| `dist_52w_high`| `float32` | Percentage distance from 52-week high (`(close / hi_52w - 1) * 100`). |
| `adr_pct` | `float32` | 20-day Average Daily Range percentage (`(High - Low) / Close * 100`). |
| `avg_vol_50` | `int32` | 50-day average share volume. |
| `dvol_m` | `float32` | 20-day average dollar volume in millions (`(Close * Volume) / 1M`). |
| `above_10` | `bool` | Close $\ge$ 10 EMA. |
| `above_20` | `bool` | Close $\ge$ 20 EMA. |
| `above_50` | `bool` | Close $\ge$ 50 SMA. |
| `above_200` | `bool` | Close $\ge$ 200 SMA. |
| `stage2` | `bool` | **Stage 2 Bullish Stack:** `Close >= 10 EMA >= 20 EMA >= 50 SMA >= 200 SMA`. |
| `sector` | `string` | Institutional sector classification. |
| `theme` | `string` | Primary thematic classification. |
| `themes` | `string` | All comma-separated thematic tags. |

---

## 5. Python API & Code Recipes

### Recipe 1: Load Market Leaderboard for a Specific Day (<7ms)
```python
from build_rs_history import load_rs_history

# Load exact state of the market on May 23, 2024
df = load_rs_history("2024-05-23")

# Top 10 Common Stocks by RS Score
leaders = df[~df["is_etf"]].head(10)
print(leaders[["rank", "symbol", "rs_score", "close", "ret_3m", "sector"]])
```

### Recipe 2: Load Ticker's Full Historical RS Trajectory
```python
from build_rs_history import load_symbol_rs_history

# Retrieve entire 10-year daily RS history for a stock
nvda_rs = load_symbol_rs_history("NVDA")
print(nvda_rs[["date", "close", "rs_score", "rank", "ret_1m", "ret_1y"]].tail())
```

### Recipe 3: Zero-Lookahead EP Filter (RS $\ge$ 90 Before Breakout)
```python
import build_rs_history

def get_pre_breakout_rs(symbol: str, breakout_date: str) -> int:
    """Retrieve RS rating as of the day prior to breakout."""
    ts = build_rs_history.load_symbol_rs_history(symbol)
    if ts.empty:
        return 50
    prior_sessions = ts[ts["date"] < breakout_date]
    if prior_sessions.empty:
        return 50
    return int(prior_sessions.iloc[-1]["rs_score"])

# Example: Check RS score of a setup before Day 1
score = get_pre_breakout_rs("NVDA", "2024-05-23")
print(f"NVDA RS rating before breakout: {score}")
```

### Recipe 4: PyArrow Direct Predicate Pushdown (High-Performance Scans)
```python
import pyarrow.dataset as ds
from pathlib import Path

dataset = ds.dataset(Path("data/rs_history"), format="parquet")

# Scan all records where RS >= 95 and stage2 is True in 2024
table = dataset.to_table(
    filter=(
        (ds.field("rs_score") >= 95) &
        (ds.field("stage2") == True) &
        (ds.field("date") >= "2024-01-01") &
        (ds.field("date") <= "2024-12-31")
    ),
    columns=["date", "symbol", "rs_score", "close", "sector"]
)
df_scanned = table.to_pandas()
```

---

## 6. How to Update the Database

When new daily OHLCV bars are added to `data/bars/` (e.g. nightly via automated download or manual fetch):

### Method A: Automated Incremental Update (Recommended)
Run [`update_rs_database.py`](file:///Users/amirbaram/Documents/code/EPs/update_rs_database.py). It automatically compares the latest date in `data/rs_history/` with the latest bars in `datastore.load_bars("SPY")`, updates only the new trading sessions, and refreshes `data/simulations/rs_universe_scores.parquet`:

```bash
.venv/bin/python update_rs_database.py
```

* **Dry-Run Check:**
  ```bash
  .venv/bin/python update_rs_database.py --check
  ```
* **Force Rebuild of Current Year:**
  ```bash
  .venv/bin/python update_rs_database.py --force
  ```

### Method B: Direct Script Invocation
```bash
# Rebuild starting from a specific date:
.venv/bin/python build_rs_history.py --since 2026-10-01

# Full 10-year rebuild from 2016-01-01 (takes ~2.5 min):
.venv/bin/python build_rs_history.py
```

---

## 7. Common Pitfalls & How to Avoid Them

### 1. The "Bottom-Fisher" Rebound Trap (e.g. REPL)
* **The Trap:** A beaten-down stock (like a biotech) that collapses from \$8 down to \$1.50 and then bounces to \$13.40 will produce a massive mathematical return (+700% in 6 months), awarding it an RS score of 97–99 even though its structural multi-year chart is in a secular decline.
* **The Solution:** Always pair `rs_score >= 90` with posture filters:
  * `stage2 == True` (verifies pristine $C \ge 10 \ge 20 \ge 50 \ge 200$ moving average alignment).
  * `dist_52w_high >= -15.0` (guarantees the stock is within 15% of its 52-week highs).

### 2. Unadjusted Split Seams
* The pipeline includes built-in split seam normalization (`_normalize_split_seams_series`). However, if an unfamiliar stock exhibits an abnormal $R_{1m} > 1000\%$ return without news, verify its bar series in `data/bars/<SYMBOL>.parquet` against split records.

---

## 8. Verification & Gate Requirement

Whenever modifying `build_rs_history.py`, `update_rs_database.py`, or `serve_rs.py`:
```bash
bash scripts/check.sh
```
* Byte-compiles all core modules including `build_rs_history.py` and `serve_rs.py`.
* Executes all unit and regression tests in [`tests/test_rs_viewer.py`](file:///Users/amirbaram/Documents/code/EPs/tests/test_rs_viewer.py).
* Must exit with code `0` before any task is considered complete.
