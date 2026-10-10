# Episodic Pivot (EP) Strategy System (`amirbaram/EPs`)

Institutional-grade **Episodic Pivot (EP)** trading system, point-in-time backtester, real-time live execution tracker, and zero-lookahead machine learning models (Python 3.13).

---

## 🚀 Quickstart & Server Applications

Run all commands using the local `.venv` environment:

```bash
# 1. EP Historical Review & Backtester (Interactive Dossier & Portfolio Simulation)
.venv/bin/python serve_ep.py          # Primary App → http://127.0.0.1:8782

# 2. Live EP Multi-Quarter Tracker & Rolling Telemetry
.venv/bin/python serve_ep_tracker.py  # Live Tracker → http://127.0.0.1:8783
```

### The Non-Negotiable Test Gate
Before completing any changes or opening a new session, run:
```bash
bash scripts/check.sh                 # Byte-compiles core modules + runs regression suite (tests/)
```
- Exits non-zero on the first failure.
- Every bug fix adds a permanent regression test in `tests/`.

---

## 🎯 Core Trading Strategy & Execution Architecture

### 1. Trade 1: Delayed Breakout Entry (Zero-Lookahead)
- **Why Day 1 Close Entry is Eliminated:** Entering at Day 1 Close is invalid in real trading because critical filtering criteria (the 48-Hour Upper Body Absorption Rule and the Day 5 V2 Rolling ML Model) require multi-day price action. Furthermore, buying Day 1 Close forces the trader to absorb overnight gap-down distribution traps (over 535 gap collapses historically).
- **Execution Rule:** On Days 2 to 5 following a verified Episodic Pivot, a stop-buy order is placed **0.05 above the Day 1 High** once the 48-Hour absorption condition is met.
- **Capital Protection:** If price violates Day 1 Low before breaking out, the stop-buy is immediately cancelled with **zero capital risked**.

### 2. Day 5 V2 Conviction Pyramiding
- On Day 5 post-EP, the rolling ordinal ML model assesses the probability of achieving a $+50\%$ or $+100\%$ advance.
- If $P(+50\%) \ge 0.20$ and the stock is holding above the breakout pivot, an additional **+50% position size** is pyramided into the trade, with the blended stop loss moved to break-even / local shelf support.

### 3. 1-Year ML Climax Exit (Partial Profit Protection)
- Trained across horizons up to 250 trading days (1 full calendar year) using zero-lookahead confirmed TrendLab swings, retracement ratios, volume accumulation vs. distribution, relative volume spikes at pivot highs, and consecutive gap-ups.
- When exhaustion probability reaches $\ge 0.50$, the system executes a **50% partial profit take**, securing peak gains while allowing the remaining 50% runner to trail along institutional support.

### 4. ML Dynamic Trailing Stop
- Activates once open profit exceeds **$+3.0\text{ R}$** (or $5.0\times\text{ ADR}$).
- Uses a calibrated 30th percentile MAE quantile regression model ($\alpha = 0.30$, ~16.8% volatility cushion) anchored beneath the rising 21 EMA and 5-day swing shelf, drastically reducing maximum drawdown while letting multi-baggers compound.

### 5. Multi-Leg Continuations (Trade 2 & Trade 3)
- **Track 1 (Institutional Undercut & Reclaim - U&R):** Detects controlled shakeouts below Day 1 Low within the first 10 sessions that quickly reclaim Day 1 levels on high relative volume.
- **Track 2 (Secondary Base Intermediate Ribbon):** Deploys an intermediate 4-EMA ribbon `(8, 12, 16, 21)` to identify secondary consolidations and base breakouts during weeks 3 to 13 post-EP.

---

## 📊 Empirical Performance & Comparative Analysis (Pinnacle Elite)

### 1. Head-to-Head Strategy Performance (252 Delayed Breakout Trades)

Every trade below uses zero-lookahead Delayed Breakout entries:

| Strategy Configuration | Trades | Win Rate | Avg Win | Avg Loss | EV (R) | Total Net R | Profit Factor | Max Drawdown | Calmar Ratio |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Mechanical Baseline (50 SMA, No ML)** | 252 | 41.3% | +2.98 R | -0.97 R | **+0.665 R** | **+167.5 R** | 2.17 | -19.0 R | 8.83 |
| **2. + New ML Dynamic Trailing Stop** | 252 | **46.0%** | +1.39 R | **-0.54 R** | +0.350 R | +88.3 R | 2.21 | **-9.6 R (-49.5%)** | 9.17 |
| **3. + New ML Climax Exit (50% Partial)** | 252 | **47.2%** | +1.33 R | **-0.59 R** | +0.320 R | +80.5 R | 2.03 | **-9.9 R (-47.9%)** | 8.17 |
| **4. + Both New MLs (Climax + Trailer)** | 252 | **46.0%** | +1.40 R | **-0.54 R** | +0.357 R | +89.8 R | 2.23 | **-9.6 R (-49.5%)** | **9.33** |
| **5. ⚡ AI-Optimal (Pyramid + Both MLs)** | 252 | **44.4%** | +1.62 R | **-0.56 R** | +0.409 R | +103.1 R | **2.32** | **-11.2 R (-50.4%)**| 9.18 |

*(Note: Pyramiding alone without ML exits experienced a **-22.6 R** drawdown. AI-Optimal cuts peak drawdown by more than half to **-11.2 R**).*

### 2. How the New MLs Compare to Previous Implementations

| Dimension | Previous ML Implementation | New ML Implementation | Measured Impact |
|---|---|---|---|
| **Climax Training Horizon** | Capped at 60 trading days | Expanded to **250 trading days (1 full year)** | Detects late-stage exhaustion pushes across multi-month runs. |
| **Climax Feature Set** | Static return thresholds (`ret >= 20%`) | **TrendLab confirmed swings**, swing volume ratios, pivot RVOL spikes, consecutive gap-ups | Grounded in causal market structure rather than arbitrary price levels. |
| **Climax Action** | Full position exit (or bypassed in UI) | **50% Partial Profit Take** (leaves 50% runner on 50 SMA baseline) | Secures peak open gains while still participating in multi-baggers. |
| **Trailing Stop Hurdle** | +2.0 R (choked off normal pullbacks) | **+3.0 R / 5.0× ADR** | Gives winners enough breathing room during early base-building. |
| **Quantile Buffer** | 10th percentile MAE (~8% cushion) | **30th percentile MAE (~16.8% cushion)** anchored below 21 EMA / 5-day shelf | Avoids premature shakeouts during institutional support retests. |
| **Drawdown Reduction** | 0% reduction (old Day 1 ML DD: -15.5 R vs -15.2 R) | **-49.5% reduction** (-9.6 R vs -19.0 R) | Cuts downside risk and portfolio volatility in half. |
| **Average Loss** | -0.97 R (full stops) | **-0.54 R (-44% reduction)** | Open profits are actively protected from round-tripping. |

### 3. Structural Trade-Off: Risk-Adjusted Quality vs. Unconstrained Outliers

- **Sharpe & Drawdown Protection:** For traders seeking lower equity curve volatility, higher win rate (+44.4%–47.2%), and halved drawdowns (-9.6 R vs -19.0 R), **AI-Optimal with ML Stops** delivers superior risk-adjusted quality (Calmar 9.33 vs 8.83).
- **Right-Tail Outlier Capture:** The **Mechanical 50 SMA Baseline** captures unconstrained 20R–30R mega-runners (like SMCI or NVDA), producing higher raw EV (+0.665 R vs +0.409 R), but requires absorbing twice the drawdown depth (-19.0 R) and taking full losses on failed consolidations.

## 📂 Repository Structure

- `serve_ep.py`: Historical Review interactive dashboard and portfolio backtester (port 8782).
- `serve_ep_tracker.py`: Live multi-quarter EP tracking dashboard and rolling telemetry (port 8783).
- `ep_ml_engine.py`: Machine learning feature engineering, rolling inference, and model definitions.
- `scanner_core.py`: Technical indicators, intermediate ribbon computation `(8, 12, 16, 21)`, and swing metrics.
- `generate_handbook_pdf.py`: Automated compiler for the Strategy Handbook PDF (`docs/EP_Pinnacle_Elite_Strategy_Handbook.pdf`).
- `data/models/`: Serialized ML model binaries:
  - `amir_rolling_xgboost_ordinal.pkl`: V2 Rolling Conviction Model.
  - `ep_exhaustion_classifier.pkl`: 1-Year Climax Exit Classifier.
  - `ep_dynamic_trailer.pkl`: Dynamic Trailing Stop Quantile Regressor.
  - `ep_ur_reentry_classifier.pkl`: Trade 2 U&R Re-entry Classifier.
  - `ep_continuation_dynamic_trailer.pkl`: Continuation Trailing Regressor.
- `data/simulations/ep_combined_study_scored.parquet`: Full 10-year scored dataset of 6,501 historical EP events.
- `tests/`: Automated regression test suite.
- `scripts/check.sh`: Local CI gate script.
