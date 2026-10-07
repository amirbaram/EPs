# Larsson Line Advanced Analysis & Research Plan

This document outlines the systematic plan to analyze advanced structural, technical, and entry/exit characteristics for the Larsson Line strategy. All tests will be executed on a **per-sector** basis, and insights will be meticulously recorded here as they are discovered.

## Phase 1: Shorts (Blue Flips) Baseline Discovery
Before diving into advanced micro-optimizations for longs, we will establish the baseline viability of shorting.
1. **Sector Baseline Scan:** Run the foundational backtest across all sectors for Blue Flips to determine which sectors yield a positive expectancy for shorting.
2. **Regime Identification:** Identify the "Combo B" equivalent for shorts (e.g., Sector ETF < 200 SMA, 10 SMA < 20 SMA).
3. **Decision Gate:** If shorting proves mathematically viable in specific sectors, Phase 2 advanced tests will be mirrored for Blue Flips.

## Phase 2: Longs (Yellow Flips) Advanced Deep Dive
We will run targeted, historical backtests (strictly avoiding lookahead bias) to evaluate the following dimensions for Yellow Flips in our top-performing sectors:

### A. Prior State Influence
* **Hypothesis:** Reversals (`Blue -> [Grey] -> Yellow`) perform differently than trend continuations (`Yellow -> Grey -> Yellow`).
* **Test:** Segment backtest results by prior state and compare Win Rate, Average R, and Max Drawdown.

### B. Better Entries (Pullbacks)
* **Hypothesis:** Entering immediately on a flip is inferior to waiting for a structural pullback.
* **Test 1 (RSI):** Track P&L if the entry is delayed until RSI becomes oversold (e.g., < 40) while the state remains yellow.
* **Test 2 (EMA Proximity):** Track P&L if the entry is delayed until the price touches/approaches the 20 or 50 SMA.
* **Test 3 (Continuation):** Track P&L if the entry requires a higher-high *after* the pullback.

### C. Catalyst Confluence (Episodic Pivots)
* **Hypothesis:** Yellow flips accompanied by an Episodic Pivot (EP) event (massive gap + volume) have a significantly higher expectancy.
* **Test:** Cross-reference flip dates with `detect_episodic_pivot` hits (window: +/- 5 days) and measure performance divergence.

### D. Structural Breakouts & Ranges
* **Hypothesis:** Flips that occur inside a chop zone/range fail more often; waiting for a breakout improves win rate.
* **Test:** Measure the 20-day price range prior to the flip. If in a tight range, require a structural breakout (or breakout + pullback) before triggering the entry.

### E. Timeframe Resonance (Daily vs Weekly & ADR)
* **Hypothesis:** High ADR (volatile) stocks perform better on weekly flips, while low ADR stocks perform better on daily flips.
* **Test:** Run the backtest using weekly bars vs daily bars and correlate the outperformance with the ticker's historical ADR at the time of the flip.

### F. Exit Optimization
* **Hypothesis:** Exiting strictly on a Blue flip captures larger trends but suffers deeper drawdowns than exiting on a Grey flip.
* **Test:** Compare the equity curves of "Exit on Grey" vs "Exit on Blue" across all top sectors.

## Phase 3: Synthesis & Implementation
1. **Compile Insights:** Document all findings per sector in a dedicated knowledge base.
2. **Scanner Integration:** Update `setups.py` and the UI to flag the absolute highest-expectancy setups (e.g., "Combo B + EP Event + Pullback Entry").
3. **Dashboard Upgrades:** Add columns for new predictive factors (e.g., Prior State, EP Confluence, Daily/Weekly ADR match).
