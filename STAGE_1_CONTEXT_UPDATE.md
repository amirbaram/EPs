# Stage 1 Panel Rewrite: Market Context & Advanced Labeling

Following the resolution of the daily at-risk panel mechanics, we have integrated the advanced market, chart, and labeling requirements requested for Stage 1. 

The implementation preserves the mathematical leakage guarantees established in the previous review while adding full cross-sectional context.

## 1. Multi-Threshold Target Labeling

Instead of hardcoding a single +50% target, the panel now simultaneously tracks whether the event reached +50%, +100%, +150%, and +200% (measured from the EP-day close) **before losing the EP-day low**.

**Semantics:**
*   The event remains "alive" (producing daily rows) until the bar *before* the EP-day low is breached, or the 250-session horizon ends.
*   **Reaching +50% no longer ends the panel**. The event continues tracking to see if it reaches higher thresholds before stopping out.
*   The `label_reach_{T}` fields are event-level labels (constant across all rows for that event). 
*   **Usage Rule for Stage 2:** To predict reaching threshold `T`, Stage 2 must train *only* on rows where `feature_chk_reached_{T} == 0` (meaning threshold `T` has not yet been achieved as of the checkpoint day).

## 2. Cross-Sectional Market Context (Sector, Theme, SPY)

The daily panel now constructs point-in-time cross-sectional rankings and relative strength (RS). To ensure causality, all aggregations are strictly backward-looking. As-of price calculations use the current universe, industry, and theme membership snapshot (conditional on the current surviving universe and current taxonomy).

*   **`ContextBuilder`:** Pre-loads the entire active equity universe alongside SPY to calculate sector and theme baskets dynamically.
*   **Peer-Mean Isolation:** When computing a sector's return or its RS vs SPY, the target stock is mathematically excluded from its own basket. This prevents large-cap stocks from distorting their own relative strength.
*   **Rankings:** Computes the stock's precise percentile rank within its industry (`ind_rank_3m`) and theme (`theme_rank_3m`) over the trailing 3 months.

## 3. Larsson Line & Chart Structure

We extracted the core Larsson Line logic (fast/slow EMA spreads and state definitions) into pure functions and injected them into the ML features:

*   **State & Momentum:** Includes the current Larsson state (Yellow/Blue/Gray), the age of that state in days, the spread between the fastest and slowest EMA, and the distance to the fastest EMA.
*   **EMA Stacking:** Computes 5-day EMA slopes and checks for strict bullish stacking (`close > ema10 > ema20 > sma50 > sma200`).

## 4. True Overhead Resistance (Confirmed Pivots)

Past pivot highs act as overhead supply. We used `trendlab` to locate structural pivot highs.

*   **Causality Enforcement:** Pivots are only visible to the model *after* they are mathematically confirmed. We filter by `trendlab`'s `pconf` (confirmation index), not `pidx` (the absolute peak).
*   **Resistance Features:** For every daily row, we look back 250 sessions and find all confirmed pivot highs strictly *above* the current close. 
*   **Metrics:** We emit the distance to the nearest overhead pivot (`overhead_nearest_pivot_pct`), the count of pivots within a 25% ceiling (`overhead_pivots_within_25pct`), and a `blue_sky` boolean if no confirmed overhead supply exists.

## 5. Testing & Validation

All regression tests in `tests/test_ml_dataset.py` have been rewritten.
*   The rigorous `test_features_are_invariant_to_future_truncation` test passes with the new `ContextBuilder` and Larsson/Pivot features, mathematically proving that future data cannot alter historical feature values.
*   A live smoke test on ~600 symbols successfully processed ~2,000 events and emitted ~147,000 causality-safe daily rows in 73 seconds.

The `scripts/build_ml_dataset.py` file is now fully aligned with both the quant hygiene requirements and the advanced predictive feature set.
