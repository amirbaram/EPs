# Episodic Pivot (EP): Code Analysis, Playbook Synthesis & Research Plan

## 1. Dissection of Current EP Detection & Lifecycle in the Code

### 1.1 Detection Mechanics (`setups.py`)
In the existing codebase, EP detection is governed by two core functions in [`setups.py`](file:///Users/amirbaram/Downloads/scan/setups.py):

1. **Event Day Detector: `_is_ep_event(row, base_dvol)`** (lines 131–168)
   Checks whether a single bar qualifies as an EP across three distinct paths:
   - **`classic`**: Gap up $\ge 6.0\%$ (`EP_MIN_GAP`) with RVOL $\ge 3.0$ (`EP_MIN_RVOL`), relaxed to $2.0$ for mega-caps with daily dollar volume $\ge \$1\text{B}$.
   - **`ep9m`**: Raw volume $\ge 9,000,000$ shares (`EP_MIN_SHARES`), day change $\ge 4\%$ (`EP9M_MIN_CHG`), RVOL $\ge 3.0$, and day dollar volume $\ge \$10\text{M}$.
   - **`bigmove`**: Large intraday surge without gap: day change $\ge 10.0\%$ (`EP_BIGMOVE_CHG`), RVOL $\ge 3.0$, and day dollar volume $\ge \$10\text{M}$.

   **Existing Safety Gates:**
   - Rejects reverse splits / crash days: `chg_pct < -5%` rejected.
   - Gap retention: If `gap_pct >= 25%`, the bar must retain at least $50\%$ of the gap (`chg_pct >= gap_pct * 0.5`).
   - Prior baseline liquidity: Prior 60-day median dollar volume must be $\ge \$0.5\text{M}$ (`base_dvol >= 0.5M`) to weed out dead shell companies that only trade on promotion days.

2. **Lifecycle & State Machine: `detect_episodic_pivot(d)`** (lines 172–238)
   Operates across two states:
   - **State `breakout` (Day 1 / Event Day)**:
     - Triggers when `_is_ep_event` fires on the latest bar (`d.iloc[-1]`).
     - Emits `level = low` (line in the sand / invalidation stop) and `trigger = high`.
     - Records `held = (close >= open)` and `neglect_6m` (6-month return prior to the event).
   - **State `watch` / Delayed-Reaction `breakout` (Days 2 to 21)**:
     - Scans back up to 21 bars (`EP_WATCH_BARS = 21`).
     - **Survival rules**:
       - `close < event_low`: Event is dead (dropped immediately).
       - `event_close < event_open`: Event day was a red candle (faded), skipped.
       - `close < event_close * 0.95`: Gave back $>5\%$ of event close, off watch.
     - Builds a breakout resistance stack (`_breakout_stack`). If price breaks above the stack with volume $\ge 1.5\text{x}$ RVOL, transitions to state `breakout`, otherwise remains in `watch`.

---

## 2. Gaps Between Current Code and the Institutional EP Playbook

| Playbook Principle | Current Code Status | Limitation / Vulnerability |
|---|---|---|
| **Intraday Bar Shape (Day 1)** | Only checks `close >= open` | Ignores where in the range the bar closed. An EP opening $+20\%$, running to $+35\%$, and closing at $+21\%$ (closing at the bottom $7\%$ of day range) passes `close >= open`, but is institutional distribution ("Gap & Crap"). |
| **Volume Cluster / Duplicate Fires** | `ep9m` fires independently on every high-volume day | Once a high-beta stock heats up (e.g. `PLTR`, `MARA`), `ep9m` triggers on 5 consecutive days during a secondary run, corrupting the definition of a *pivot*. |
| **Phase 2 (Days 2–5) Momentum Check** | Direct leap from Day 1 to a generic 21-bar watch | Misses the **48-Hour Retracement Rule**: institutional absorption failure occurs if $>50\%$ of the Day 1 body is given back within 48h. |
| **The 4 Outcome Archetypes** | Binary (`breakout` vs `watch`) | Does not classify into Runaway Trend, High-Tight Digestion, Stagnation, or Trap. |
| **Sector & Theme Differentiation** | Uniform rules across all symbols | Biotechs (high dilution/offering risk) are treated identically to SaaS (high institutional PEAD drift) and Semis (multi-quarter guidance ramps). |
| **Relative Strength Context** | Only checks symbol's prior 6-month return (`neglect_6m`) | Completely ignores Sector & Theme relative strength vs SPY prior to and during the event. |

---

## 3. Concrete Code Improvements to Propose

1. **Closing Range Position (`close_pos`) Gate**:
   - Require `close_pos = (close - low) / (high - low) >= 0.65` on Day 1. High-quality EPs must close in the upper third of their daily range, proving buyers absorbed late-day profit taking.
2. **Fresh Pivot / Anti-Clustering Gate**:
   - An EP event cannot trigger if an EP already fired within the past 10–15 sessions, unless the new event represents a new leg higher breaking out of a distinct consolidation.
3. **Phase 2 (Days 2–5) Momentum Lifecycle States**:
   - State `momentum_thrust`: Days 2–5 holding above Day 1 close or consolidating in upper $25\%$ of Day 1 range.
   - State `digestion_flag`: Days 3–10 forming a tight flag holding $>50\%$ of Day 1 candle.
   - State `failed_absorption`: Breaches Day 1 low or loses $>50\%$ of Day 1 body within 48 hours.
4. **Sector-Tailored Calibration**:
   - **Biotech / Pharma**: Enforce higher dollar volume ($\ge \$25\text{M}$ instead of $\$10\text{M}$) and Day 1 close in top $20\%$ (`close_pos >= 0.80`) to filter out toxic ATM warrant/dilution traps.
   - **Enterprise Software & Semis**: Integrate 10/20 EMA trailing mode once past Day 5 (riding the PEAD wave).
5. **Sector & Theme Tailwind Filter**:
   - Score bonus / higher conviction when the ticker's Sector and Theme have a positive 1-month return vs SPY (`m1_pct > spy_m1_pct`) at the time of the event.

---

## 4. Empirical Research Methodology Across the Active Universe

To prove these dynamics rigorously before making software edits, we conduct a historical simulation study across the 2,057 active symbols using our cached historical daily bars (`data/bars/`) and precomputed performance history (`data/perf_history/`).

### Simulation Dimensions & Metrics:
1. **Event Population**:
   - Scan all active universe symbols from 2020 through 2026.
   - Extract every valid EP event date, subtype (`classic`, `ep9m`, `bigmove`), Day 1 gap %, RVOL, Day $Vol, and `close_pos`.
2. **Context & Environment Stamping**:
   - Stamp Sector & Themes via `labels.sector(sym)` and `labels.themes(sym)`.
   - Stamp pre-event relative strength: Ticker, Sector, and Theme returns ($1\text{W}$, $1\text{M}$, $3\text{M}$) vs SPY as of the event date (from `data/perf_history/groups.parquet` and `tickers_*.parquet`).
3. **Follow-Through Measurement**:
   - **Phase 2 (Day 2 to 5)**:
     - Day 2–5 Max Favorable Excursion ($\text{MFE}_5$) and Max Adverse Excursion ($\text{MAE}_5$).
     - Day 2 Close vs Day 1 Body ($50\%$ retracement violation rate).
     - Breached Day 1 Low within 5 days (Trap rate).
   - **Phase 3 (Week 2 to 12 / Days 10, 20, 60)**:
     - Cumulative return at Day 10, 20, and 60.
     - Maximum gain within 60 days.
     - Ratio of trades holding above rising 10 EMA / 20 EMA.
4. **Outcome Archetype Classification**:
   - **Runaway Monster**: $\text{Gain}_{20} \ge +30\%$ without ever closing below Day 1 low.
   - **High-Tight Digestion**: Max pullback $\le 10\%$, breaks to new highs within 15 days.
   - **Fade / Stagnation**: Chopped within $\pm 5\%$ of Day 1 close, drifted back to pre-gap levels.
   - **Gap & Crap Trap**: Closed below Day 1 low within 5 sessions.
5. **Sector & Theme Expectancy Matrix**:
   - Win Rate, Expectancy ($E[R]$), and Trap Rate broken down by Finviz industry sectors and thematic baskets.
6. **Market Regime Correlation**:
   - EP success rate under bull markets (e.g. 2020–2021, 2023–2024) vs chop/bear markets (2022).
