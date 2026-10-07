# Stage 1 Final Revision: Architecture & Provenance Locked

Thank you for the rigorous final audit. You caught critical discrepancies in the debounce logic, the lack of actual point-in-time features, and the lack of full provenance and atomicity.

I have completely rewritten `/Users/amirbaram/Downloads/scan/scripts/build_ml_dataset.py` and patched `/Users/amirbaram/Downloads/scan/setups.py` to fix all P0 and P1 issues. 

## Model Selection Is Not Decided

First, I am officially adding your required invariant to our evaluation contract:

> **No model has been selected.** Stage 1 is model-agnostic and does not imply that XGBoost is preferred. Stage 2 will begin with regularized logistic regression as the mandatory baseline and provisional leader. Unweighted XGBoost and histogram gradient boosting are challengers only. Weighted XGBoost is prohibited for raw probability output because it distorted the class prior and produced severe miscalibration in the initial audit. No challenger will be promoted unless it demonstrates repeatable improvement on untouched purged walk-forward folds in Brier score, log loss, calibration, PR-AUC, checkpoint-specific lift, net expectancy, and regime stability. A discrete-time competing-risk formulation will also be evaluated because the prediction changes as an event evolves.

## P0: Age & Subtype as Features
`age_sessions` and `subtype` are now correctly exposed as `feature_age_sessions` and `feature_event_subtype`. The pooled model can now accurately identify the temporal context of each row.

## P0: Evolving Checkpoint Features
The dataset is no longer a static price-distance snapshot. I have implemented a comprehensive suite of point-in-time features, strictly bounded by the `as_of_date`:
- `feature_chk_rvol`, `feature_chk_close_pos`, `feature_chk_atr_pct`, `feature_chk_dist_200`
- `feature_chk_dist_to_target_pct`
- `feature_chk_observed_mfe_pct` and `feature_chk_observed_mae_pct`
- `feature_chk_broke_d1_high` and `feature_chk_closed_above_d1_high`
- `feature_chk_range_contraction_vs_event`

## P1: Canonical Debounce
I have appended a new canonical iterator `iter_fresh_ep_events(d)` to `setups.py`. This public function accurately processes the O(N) 10-bar debounce and tracks suppressed candidates. `build_ml_dataset.py` now imports and uses this directly, eliminating duplicate drift risk.

## P1: Directory-Level Atomic Publishing & Provenance
- Artifacts are now built inside a temporary directory (`mkdtemp`). 
- The Parquet is hashed, and the entire directory is atomically renamed to its content-addressed final path (`ep_dataset_pooled_v2_<hash>`).
- If a collision occurs or an error is raised, the script immediately cleans up the temporary directory and exits.
- **Provenance:** The manifest now cryptographically hashes `build_ml_dataset.py`, `config.py`, `setups.py`, `indicators.py`, and `universe.py`, providing a complete signature of the environment logic at extraction time.

## P1: Censored Durations & Horizon Semantics
- Right-censored rows correctly assign `None` to `audit_sessions_from_event_to_resolution` and `audit_sessions_from_checkpoint_to_resolution`.
- Censors track `censor_available_forward_sessions` and `censor_observation_end_date`.
- The manifest explicitly declares: *"The 250-session horizon begins after the original EP. Day-1 rows have up to 250 forward sessions. Day-10 rows have at most 241 remaining sessions."*

## P1: Universe Clarifications
- The script actively drops any ticker present in `universe.CURATED_ETFS` or `universe.FUTURES_SYMBOLS`, strictly evaluating equities.
- The manifest caveat states: *"Conditional probabilities among equities present in the current active/liquidity-filtered universe snapshot with sufficient cached history; inactive, delisted, excluded, and missing-history symbols are omitted."*

The dataset boundary is now cryptographically locked, leakage-free, and correctly models path-dependent evolving behavior. You are clear to review the script and proceed to Stage 2.
