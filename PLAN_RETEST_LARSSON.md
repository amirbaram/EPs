# Plan: Re-evaluate Larsson Setup Insights

## 1. Context & Motivation
The previous tests and analysis (`LARSSON_ANALYSIS.md` and `ADVANCED_EXITS_ANALYSIS.md`) were generated before the complete structural rewrite of the `themes.csv` and `sectors.csv` taxonomy. While some previous tests attempted to pull strict Finviz industries directly, the thematic mappings (e.g., Photonics, AI, etc.) were severely flawed due to the old regex keyword system. Now that the app possesses a pristine, 1-to-1 mapping of 143 Finviz industries and deterministic ETF-based themes, the entire historical simulation suite must be re-swept to find the true mathematical edge.

## 2. Tests to be Redone
- **Baseline Expectancy by Sector & Theme (Longs & Shorts):** We need to rerun the EV (Expected Value) and Win Rate of buying Yellow Flips and shorting Blue Flips across all newly defined sectors and themes.
- **Multi-Timeframe Dominance (1D vs 2D vs 1W):** We need to recalculate which timeframe is optimal for the new sector classifications.
- **Advanced Exits Validation:** We need to re-verify if holding until the "Blue Flip" is still mathematically superior to partial profit taking under the new strict definitions.

## 3. Subagent Execution Plan
A dedicated quantitative subagent (using the `pro` model) will be dispatched to:
1. Write a unified simulation script that imports `labels` and iterates over all active symbols, testing the `EMA_Cross` setup.
2. Group and aggregate trade outcomes (R-multiples, hold times, win rates) using the exact strings now populated in `labels._SECTOR` and `labels.themes()`.
3. Output the findings into a new master artifact: `UPDATED_LARSSON_INSIGHTS.md`.

## 4. Next Steps (Main Agent)
Once the subagent completes its quantitative sweep, I will take the newly generated `UPDATED_LARSSON_INSIGHTS.md` and use it to surgically patch the hardcoded sector and theme arrays inside `setups.py` (specifically `detect_ema_cross` and any related logic), ensuring the live scanner engine strictly relies on the verified mathematical edges of the new taxonomy.
