# AGENTS.md — Episodic Pivot (EP) Strategy System (`amirbaram/EPs`)

Institutional **Episodic Pivot (EP)** trading system, machine learning models, and real-time dashboard suite (Python 3.13).
Located at `/Users/amirbaram/Documents/code/EPs`.

## Primary Applications & Ports
```bash
.venv/bin/python serve_ep.py          # Historical Review & Strategy Backtester → http://127.0.0.1:8782
.venv/bin/python serve_ep_tracker.py  # Live Multi-Quarter EP Tracker & Telemetry → http://127.0.0.1:8783
```
Deps: Python 3.13 virtual environment located at `.venv/`. Use `.venv/bin/python` for all execution.

## The Gate — Run After EVERY Change (Non-Negotiable)
```bash
bash scripts/check.sh                 # Byte-compiles core EP modules + runs tests/ (fast, offline, zero deps)
```
- Green gate is required before calling any change done. It exits non-zero on the first failure.
- Every bug fix adds a permanent regression test in `tests/`.

## Core System Architecture & Trade Execution Rules
1. **Trade 1 Execution — Delayed Breakout Only (No Day 1 Close Entry):**
   - Entering on Day 1 Close is fundamentally **invalid** for live Pinnacle Elite trading because critical criteria (the 48-Hour Upper Body Absorption rule and Day 5 V2 rolling model) cannot be known on Day 1 without lookahead bias.
   - The strategy deploys **Delayed Breakout Entry** on Days 2 to 5: stop-buy order placed 0.05 above Day 1 High once 48H holding is verified.
   - If price breaches Day 1 Low before breakout, the order is cancelled immediately with **zero capital risked** (bypasses 535 gap-down traps).
2. **Day 5 V2 Conviction Pyramiding:**
   - Adds +50% position size on Day 5 if the rolling ordinal model conviction holds ($P(+50\%) \ge 0.20$) and price holds above initial breakout level.
3. **1-Year ML Climax Partial Profit Take:**
   - Trained up to 250 trading days (1 full calendar year) using zero-lookahead confirmed TrendLab swings, retracement ratios, up/down swing volume, pivot RVOL spikes, and consecutive gap-ups.
   - Triggers a **50% partial profit take** when exhaustion probability reaches $\ge 0.50$, locking in peak R while retaining the remaining 50% runner on the 50 SMA baseline.
4. **ML Dynamic Trailing Stop:**
   - Activates once unrealized profit clears $\ge +3.0\text{ R}$ (or $5.0\times\text{ ADR}$).
   - Trails using a 30th percentile MAE quantile regression buffer ($\alpha = 0.30$, ~16.8% buffer) anchored below the rising 21 EMA and 5-day swing shelf.
5. **Trade 2 & 3 Multi-Leg Continuations:**
   - **Track 1 / Track A:** Institutional Undercut & Reclaim (U&R) and High Tight Flag (HTF) 10 EMA pullbacks holding $\le 18\%$ consolidation.
   - **Track 2 / Track B:** Secondary Base Intermediate Ribbon (8, 12, 16, 21) breakout after Blue/Gray digestion.

## Directory & File Map
| Concern | File |
|---|---|
| Historical Review Server & Strategy Engine | `serve_ep.py` (port 8782) |
| Live Multi-Quarter Tracker Server | `serve_ep_tracker.py` (port 8783) |
| Machine Learning Engine & Feature Extractors | `ep_ml_engine.py` |
| Technical Indicators & Intermediate Ribbons | `scanner_core.py`, `indicators.py` |
| Local Datastore Bar Loading | `datastore.py` |
| Sector & Theme Classifications | `labels.py`, `thematic_engine.py` |
| Playbook Handbook PDF Generator | `generate_handbook_pdf.py` |
| Model Weights (`data/models/`) | `ep_exhaustion_classifier.pkl`, `ep_dynamic_trailer.pkl`, `ep_ur_reentry_classifier.pkl`, `ep_continuation_dynamic_trailer.pkl`, `amir_rolling_xgboost_ordinal.pkl` |
| Scored Historical Dataset | `data/simulations/ep_combined_study_scored.parquet` |
| Regression Test Suite | `tests/` |

## Git Discipline
- Repository is `amirbaram/EPs` located at `/Users/amirbaram/Documents/code/EPs`.
- Do not commit or push to `main`. Commit only to your working branch when explicitly requested by Amir.
