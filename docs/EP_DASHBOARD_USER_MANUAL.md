# Episodic Pivot (EP) Dashboard User Manual

This manual provides a comprehensive guide on how to utilize the two primary EP applications, how to configure the system for different trader archetypes, and deep insights into the progressive exposure sizing models tested over the 10-year dataset.

---

## 1. App Ecosystem Overview

### App 1: The Historical Institutional Review (Port 8780 / 8782)
* **Purpose:** Point-in-time historical backtesting and structural review.
* **How to use it:** 
  - Select a historical EP event from the left panel. The app immediately loads the chart and multi-leg trade dossier *as of that exact date*.
  - Use this app to build your pattern recognition. Study how the 50-day SMA baseline, 48-Hour Absorption Gate, and 20 EMA trailing stops look in hindsight across the greatest runners of the last 10 years.
  - **New Feature:** Scroll through the **"🤖 AI Probabilistic Evolution"** timeline on the right side to watch how the AI prediction engine's +100% target probability evolved dynamically day-by-day based on structural holds.

### App 2: The Live EP Follow-Through Tracker (Port 8783)
* **Purpose:** Real-time portfolio management, live alerts, and daily execution planning.
* **How to use it:**
  - Open the app each morning or after the market close to review the **Active Pipeline** of tracked EPs.
  - The engine will automatically flag which setups are ready for a Secondary Breakout Add, which are hovering near their Day 1 Low stops, and which have breached the 50 SMA baseline.
  - **New Feature:** Click on any live setup in the pipeline to immediately see its real-time **"🤖 AI PREDICTION & TARGETS"** probabilities, including warnings if the setup has devolved into a Toxic Trap.

---

## 2. Trader Archetypes & System Configuration

Depending on your psychological tolerance for drawdown, win rate, and your ultimate compounding goals, you should configure the system differently. 

Here are the three primary Trader Archetypes and the optimal settings for each based on the 10-year study:

### Archetype A: The Aggressive Compounder (Maximum EV, Higher Heat)
* **Goal:** Maximize total compounding upside (+1,900 R+ over 10 years) by riding multi-quarter trends.
* **Win Rate Expectation:** 34% - 38%
* **Expected Worst Drawdown:** -26.0 R
* **Settings & Rules:**
  - **Initial Risk:** 1.0 R (Scale to 1.5 R for Institutional Sweet Spot RVOL 5.0x+).
  - **Progressive Sizing:** **Streak-Based Multiplier (1.0 R to 1.35 R on hot streaks, 0.5 R on cold streaks).**
  - **Secondary Adds:** Add +50% size immediately upon Day 2 High Breakout.
  - **Exits:** Pure 50-day SMA trailing stop. NO partials at 2R or 3R. Let the runners run to +20R, +30R. 

### Archetype B: The Balanced Operator (High Win Rate, Smoothed Equity Curve)
* **Goal:** High consistency, smooth psychological equity curve, low drawdown. 
* **Win Rate Expectation:** 52% - 54%
* **Expected Worst Drawdown:** -8.0 R to -9.0 R
* **Settings & Rules:**
  - **Initial Risk:** 1.0 R Flat (No scaling).
  - **Progressive Sizing:** Static Sizing (No streak multipliers).
  - **Secondary Adds:** Only add size if the 48-Hour Upper-Body Absorption Gate is formally passed on Day 3.
  - **Exits:** **Partial Exits (P3_Partial_3R)**. Sell 1/3 of the position at +3.0 R to lock in profit, and trail the remaining 2/3 strictly against the 20 EMA (not the 50 SMA).
  - **Stops:** Move stop to Breakeven once the stock clears +3.0 R. (Do not move it before 3R, or you will get chopped out of 15% of your winners).

### Archetype C: The Defensive Sniper (Highest Expectancy, Fewest Trades)
* **Goal:** Only take the absolute safest, highest-probability "Pinnacle Elite" setups.
* **Win Rate Expectation:** ~60% - 70%
* **Expected Worst Drawdown:** -4.5 R
* **Settings & Rules:**
  - **Initial Risk:** 0.5 R for all trades (Half-heat).
  - **Progressive Sizing:** None.
  - **Secondary Adds:** None.
  - **Filters:** Only execute if the stock meets the **Idiosyncratic Alpha Protocol** (RVOL > 8.0x, DVol > $150M, ClosePos > 0.85) or has the **5D Support Held** booster activated. Ignore everything else.
  - **Exits:** Sell 1/2 at +2.0 R, trail remainder on the 10 EMA for a quick exit.

---

## 3. Models of Progressive Exposure (Empirical Insights)

During our 10-year research, we simulated multiple models of progressive exposure to answer the question: *How do we optimally size up during hot markets without getting obliterated during sector rotations?*

Here is what we tested and the insights uncovered:

### Model 1: Flat Static Sizing (The Baseline)
* **Mechanics:** Risk exactly 1.0 R on every single EP, regardless of recent win/loss streaks.
* **Insights:** Produced a respectable **+577.1 R** over 10 years with a $-8.83 R$ drawdown. However, it completely failed to capitalize on the massive cluster of +300% runners that occurred during the 2020/2021 post-COVID tech boom and the 2023 AI surge. It leaves massive alpha on the table.

### Model 2: The Equity-Curve Moving Average Crossover (Macro Filter)
* **Mechanics:** Trade 1.0 R when the portfolio equity curve is above its 20-trade moving average. Cut risk to 0.5 R when the equity curve drops below the moving average.
* **Insights:** This model proved to be **too slow**. Because EPs are explosive and clustered, by the time the equity curve dropped below its MA to signal a sizing reduction, the drawdown was usually already over. Furthermore, it kept risk at 0.5 R right as the *next* wave of massive winners was igniting, crippling the recovery phase.

### Model 3: Streak-Based Multiplier (The Winner)
* **Mechanics:** 
  - Base Risk: 1.0 R.
  - After 2 consecutive wins, scale risk up by 15% (1.15 R).
  - After 3 consecutive wins, scale risk to 1.35 R.
  - After 3 consecutive losses, cut risk aggressively to 0.50 R.
  - After 5 consecutive losses, cut risk to 0.25 R (Capital preservation).
* **Insights:** This model generated an astonishing **+1,989.3 R** over the 10-year period (more than 3x the baseline profit) with a drawdown of only **-26.67 R**.
* **Why it works:** Episodic Pivots are notoriously "streaky." They occur in dense clusters when market momentum aligns. The streak-multiplier naturally "presses" the bet exactly when the market is rewarding the setup, and instantly throttles down exposure at the first sign of a regime shift or sector-wide rotation.

### Final Recommendation on Exposure
The optimal progressive exposure protocol is the **Streak-Based Multiplier combined with the RVOL Sweet Spot Boost**.
1. Rely on the streak-multiplier to automatically manage your base risk (0.5 R during cold streaks, 1.0 R standard, 1.35 R during hot streaks).
2. Override the base risk with a **+25% to +50% size boost** on individual trades that hit the Institutional Sweet Spot (RVOL 5.0x - 7.0x) or print a Novel 9M Volume day, as these have dramatically higher expected values (+3.96 R) and are less likely to trap.
