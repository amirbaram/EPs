# Stage 1 Dataset Refactor: Response to Final Audit

Thank you for the rigorous final audit. You caught critical discrepancies in the debounce logic and universe selection, and accurately diagnosed that a Day-1 static model fundamentally fails the user's requirement to classify trades based on *developing* conditions.

I have completely rewritten `/Users/amirbaram/Downloads/scan/scripts/build_ml_dataset.py`.

Here is how the P0 and P1 blocking issues were resolved:

## P0: Evolving Conditions (The Pooled Checkpoint Dataset)
You were correct: a static Day-1 model cannot learn from 48-hour absorption. 
The script now builds a **pooled checkpoint dataset**. 
- It generates rows for `age_sessions` $\in \{1, 2, 3, 5, 10\}$.
- Features (`rvol`, `dist_50`, `close_pos`) are captured *as of* the specific checkpoint date (`as_of_date`). 
- **Leakage Prevention:** If a trade resolves (hits target or stop) *before* a checkpoint is reached (e.g., stops out on Day 2), it is dropped from the Day 3, 5, and 10 checkpoints. The target probability is strictly: "Conditional on remaining unresolved through this checkpoint, what is the probability of success?"

## P0: Debounce Parity
The script no longer uses a naive `last_ep_idx` distance check. 
It now mirrors the `setups.py` logic perfectly: it gathers all raw candidates via `_is_ep_event`, and suppresses any candidate if *any* raw candidate occurred in the preceding 10 bars (regardless of whether the preceding one was suppressed or accepted).

## P0: Stale Registry Universe
The script no longer relies on `episodic_pivot.parquet` to define the universe. It pulls the true `universe.active_symbols()`. The manifest explicitly records the survivorship bias caveat.

## P1: Outcomes, Censors, and Data Integrity
- **Stops:** Changed from `< d1_low` to `<= d1_low` for exact parity with the historical backtester.
- **Rich Outcomes:** Instead of binary 1/0, it records `target`, `stop`, `full_horizon_timeout`, `right_censored`, and `ambiguous_same_bar` (which defaults to the conservative stop assumption).
- **Economic Fields:** Added `sessions_to_resolution`, `mfe_pct` (Maximum Favorable Excursion), and `mae_pct` (Maximum Adverse Excursion) calculated strictly over the resolved slice.
- **Missing Data:** Uncalculated MAs safely default to `np.nan`.
- **Versioning:** Output is saved as `ep_dataset_pooled_v1_{timestamp}.parquet`. A JSON manifest is written atomically, recording the SHA256 content hash, exact schema, configurations, and extraction provenance.

The dataset boundary is now rock solid. Please evaluate the new `build_ml_dataset.py`.
