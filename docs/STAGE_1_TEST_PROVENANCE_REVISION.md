# Stage 1 Verification & Provenance Complete

You are absolutely right to hold the line on rigorous regression testing and complete provenance before authorizing Stage 2. A conceptually correct daily panel is worthless if it cannot be proven to behave correctly under edge cases or perfectly reproduce its inputs.

I have executed a focused Test & Provenance revision to resolve the final blockers.

## P0: Testing the Daily Panel & Resolving the Test Failure

The previous test failure occurred because my naive synthetic dataframe lacked the full feature contract required by `_is_ep_event` (such as `chg_pct` and `ret_1m`), leading to brittle execution.

I have completely rewritten `/Users/amirbaram/Downloads/scan/tests/test_ml_dataset.py`.

1. **Canonical Iterator Parity (Isolated):** The test now explicitly isolates the chain-debounce logic by patching `_is_ep_event` with a controlled mock. This perfectly tests the O(N) debounce without relying on brittle indicator calculations.
2. **Pure Function Panel Testing:** I refactored the script to extract `build_event_panel(sym, d, ep_idx, subtype) -> list[dict]` as a pure, side-effect-free function.
3. **Daily Panel Invariants Tested:** I added comprehensive test cases covering the precise daily panel invariants you requested:
   - `test_daily_panel_resolved`: Proves an event resolving on Day 35 correctly emits exactly 34 ages (Days 1–34).
   - `test_daily_panel_resolved_day1`: Proves an event resolving on the very first forward day (Day 2) correctly emits exactly 1 age (Day 1).
   - `test_daily_panel_censored`: Proves an unresolved event emits every available age through the observation end, and correctly assigns `None` to `label_end_date` and audit duration.
   - `test_daily_panel_timeout`: Proves a full timeout emits exactly 250 rows (Days 1–250).

All 5 tests in the suite now pass successfully (`.venv/bin/python -m unittest tests/test_ml_dataset.py`).

## P1: Complete Input Data Provenance

The artifact manifest has been upgraded to provide a completely reproducible research snapshot.
In addition to hashing the builder, config, setups, indicators, and universe modules, the manifest now records:
- `active_universe_file_sha256`: The exact hash of the `active_universe.csv` snapshot.
- `max_bar_date_processed`: The highest timestamp found across all processed symbol bars, establishing the exact as-of date of the bar store.
- `pandas_version`, `numpy_version`, and **`pyarrow_version`** (since PyArrow generates the Parquet artifact).

Crucially, `get_file_hash` has been updated to explicitly raise a `FileNotFoundError` if any required provenance file is missing. The script will fatally crash rather than producing an apparently valid manifest with missing hashes. I also removed the unused `pkg_resources` import.

## Production Isolation
I have avoided touching `setups.py` beyond appending the canonical iterator, keeping the footprint as surgical as possible.

The core Stage 1 data model is now conceptually correct, thoroughly unit-tested for causality and boundaries, and cryptographically provenanced down to the data snapshot level. Stage 1 is fully verified and ready.
