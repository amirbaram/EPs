# EMA Cross (Larsson Line) Integration Plan

This document outlines the efficient, scalable architecture to introduce the 4-EMA Larsson Line (referred to as "EMA Cross") into the scanner app, adhering to caching, UI, and performance constraints.

## 1. Engine & Core Calculations (`setups.py` & `scanner_core.py`)
Instead of computing all Relative Strength (RS) ratios across the entire universe inside `indicators.py` (which would bloat memory and execution time), we will compute the EMA Cross at the exact moment of setup evaluation.

**New Detectors in `setups.py`:**
We will write a unified detector `detect_ema_cross` that leverages `scanner_core.calc_larssson_line` across multiple baselines:
- `abs`: Absolute price flip.
- `rs_spy`: Ticker vs SPY flip.
- `rs_sector`: Ticker vs Sector flip.
- `rs_theme`: Ticker vs Theme flip.

**Efficient RS Pass in `scan.py`:**
Since `setups.py`'s `run_all(d)` currently only takes the ticker's dataframe, we will add an optional `context_frames` dict argument (containing `spy_df`, the ticker's `sector_df`, and `theme_df`) that `scan.py` populates during its loop.

## 2. Hit Generation & Disk Caching
When a ticker triggers any of these conditions (either today or in the past 5 days), it will generate a hit dictionary stored in `hits_{date}.csv`. 
To support the UI's requirement of separating "flips today" from "flips in the past week" and "yellow vs blue", the setup hit will carry metadata:
- `setup`: `"ema_cross"`
- `variant`: `"abs"`, `"spy"`, `"sector"`, `"theme"`
- `state`: `"yellow"` or `"blue"`
- `days_since_flip`: `0` (flipped today) up to `4` (flipped within past week)
- `entity_type`: `"ticker"`, `"spdr_etf"`, `"synthetic_sector"`

*Caching Benefit:* By emitting these as standard hits during the nightly `scan.py` run, the scanner app UI will load them instantly from the `.csv` cache without recalculating the heavy 4-EMAs on demand.

## 3. UI Integration (`report.py` & `index.html`)
The combinatorial complexity (Tickers vs SPY, Sectors vs SPY, Yellow vs Blue, Today vs Week) would crowd the main sidebar if added as raw setups.

**New UI Filtering:**
We will add a dedicated interface grouping for the EMA Cross.
- **Filters:** 
  - Scope: `[Tickers, SPDR ETFs, Sectors]`
  - Baseline: `[Absolute, vs SPY, vs Sector, vs Theme]`
  - Flip Timing: `[Flipped Today, Flipped Past Week]`
  - Direction: `[To Yellow, To Blue]`
- **Future-proofing Combinations:** The frontend will read the flat cached JSON payload. Since the payload has states for both Absolute and RS, the UI can easily filter intersections (e.g., "Show me Tickers that flipped Yellow vs SPY AND are Absolute Yellow").

## 4. Work Breakdown (The Steps)
1. **Core Data Plumbing:** 
   - Modify `ratios.py` or `scan.py` to construct and cache synthetic Sector and Theme baskets during the daily scan pass.
2. **Setup Detectors:**
   - Implement `detect_ema_cross` in `setups.py`.
   - Update `scan.py` to inject the context DataFrames into the setup detectors.
3. **Dashboard Formatting (`report.py`):**
   - Register the new `ema_cross` setup.
   - Group the hits cleanly in `compute_payload()`.
4. **UI Refactoring (`index.html` / `serve.py`):**
   - Build the dropdown/filtering logic to traverse the EMA Cross variations.

**Validation Gate:**
As per `AGENTS.md`, after Stage 2, I will verify the hits via a dry run and run `check.sh`.
