# Comprehensive Research Plan: Pinnacle Elite Predictive Engine & Progressive Exposure

## 1. Executive Summary & Prime Objective
The objective of this research is to transform the Setup Scanner into an advanced, predictive EP intelligence engine. The user has explicitly rejected broad, uncurated simulations that include penny stocks and low-expectancy traps. Therefore, this research will strictly build upon the proven **Pinnacle Elite** foundation to achieve the following:

1. **Advanced Classification & Prediction**: Predict trade outcomes dynamically. Classify setups as "highly promising" vs. "warning sign / trap" using evolving conditions (ADR expansion, Theme state, Base depth). Provide actionable probabilities for multi-bagger potential vs. failure.
2. **Context-Aware Trade Management**: Evaluate multiple entry archetypes (including E1 Close, E2 Breakout, and **E3 Bonde Delayed Entry**) paired strictly with context-appropriate stoplosses (Candle Low vs. Pivot Low) and dynamic exits (High-R partials, ADR extensions, moving average climax stretches).
3. **Comprehensive Progressive Exposure**: Move beyond simple streak-based sizing. Develop robust, dynamic risk-scaling systems that respond to portfolio equity heat, thematic cluster momentum, and broader market regimes.
4. **Strict Alignment with Prior Learnings**: We will *not* regress. The baseline dataset will strictly enforce prior empirical truths (e.g., no premature breakeven at 1R/1.5R, severe headwind vetos, high RVOL requirements).

---

## 2. Core Rule 0: Strict Baseline Curation (The "Pinnacle Elite" Gate)
Prior research proved that analyzing the entire universe of raw gaps yields a "Garbage In, Garbage Out" result with unacceptably low EV (~0.15R) and massive drawdowns.
**Before any probabilistic modeling or trade management simulation occurs, the dataset MUST be filtered by the Pinnacle Elite baseline and Qullamaggie/Bonde criteria:**
- **Price & Liquidity Minimums**: No penny stocks or illiquid traps.
- **EP Type Stratification**: Every EP will be strictly separated and modeled independently by its catalyst architecture (e.g., **9M** morning spikes, **DEP** delayed earnings plays, **EOD** end-of-day momentum, **Bonde DRE** delayed reaction entries). We will not mix 9M intraday dynamics with EOD closing dynamics.
- **Close Position (ClosePos)**: $\ge 0.65$ for standard EOD entries (waived for specific Bonde DRE setups which wait for Day 3+ confirmation).
- **Qullamaggie & Bonde Base Context**: Setups must emerge from a recognizable structural base (e.g., 3-6 month flat base, high tight flag). Extended exhaustion gaps are vetoed.
- **Relative Volume (RVOL)**: Must meet extreme surge thresholds.
- **Severe Headwind Veto**: Setups triggering under a declining 50/200 SMA or in the bottom 35th percentile of thematic momentum are instantly discarded from the primary research universe.
- **48-Hour Institutional Absorption**: Setups must prove themselves within 48 hours or be flagged as traps.

---

## 3. Research Dimensions & Parameter Space

### Dimension A: Entry Archetypes & Structural Stoplosses
Different entries require different invalidation points. We will evaluate:
1. **E1: Day 1 Close Entry (Anticipatory)**
   - *Stop 1a (Candle Low)*: Immediate invalidation below the gap candle low.
   - *Stop 1b (SwingsTimes Pivot Low)*: Wider structural stop accommodating Day 2 backfills.
2. **E2: Day 2 Breakout (Confirmation)**
   - *Stop 2a (Day 1 Low)*: Standard wide base stop.
   - *Stop 2b (Day 2 Low)*: Tighter stop for immediate momentum validation.
3. **E3: Bonde Delayed Entry (Days 3-21 Flag Breakout)**
   - *Stop 3a (Flag Low)*: Ultra-tight stop placed at the low of the consolidation flag, offering extreme R-asymmetry.
   - *Stop 3b (Master Day 1 Low)*: Wider guardrail.

### Dimension B: Dynamic Trade Management & Exits
We will discard premature breakeven strategies (which destroy profitability via whipsaws) and test advanced exit logic tailored to market conditions:
1. **Condition-Based Profit Taking**: Predicting the exact moment to take partials based on evolving momentum decay (e.g., price stretching $>2.5 \times \text{ADR}$ over the 10 EMA, or a 3-day parabolic climax), rather than blind static R-multiples.
2. **High-R Milestone Partials**: Scaling out at $+3.0R$, $+5.0R$, and $+10.0R$ when structural conditions support holding.
3. **ADR Multiples**: Taking profits when price extends $+3.0 \times \text{ADR}$ or $+5.0 \times \text{ADR}$ from entry.
4. **Moving Average Extensions (Climax Exits)**: Taking profits when price is excessively stretched above the 50 SMA (e.g., $> 40\%$ or $> 3.5 \times \text{ADR}$) or the fast 10 EMA (Qullamaggie blow-off rule).
5. **Context-Aware Breakeven**: Moving to breakeven *only* after structural higher-lows are formed or significant $+3.0R$ milestones are crossed.

### Dimension C: Condition Evolving & Classification Features
To predict outcomes, we will track how conditions evolve post-EP:
1. **Market Regime (Qullamaggie)**: $QQQ > 10\text{ EMA} > 20\text{ EMA}$ (Risk-On) vs. Chop/Risk-Off.
2. **Sector & Theme Regimes (Larsson Variants)**: 
   - Fast (5/10/20/40), Standard (10/21/50/200), and Slow (20/50/100/200) EMA stacks.
3. **Dynamic ADR Evolution**: Tracking $\text{ADR}_t / \text{ADR}_0$ to detect volatility expansion vs. contraction.
4. **Prior Base Quality**: Dormant flat base (neglect) vs. extended exhaustion.

### Dimension D: Comprehensive Progressive Exposure
Testing dynamic position sizing systems:
1. **Thematic Cluster Heat**: Sizing up when multiple EPs trigger simultaneously in the same leading sector (e.g., AI, Biotech).
2. **Portfolio Equity Curve Scaling**: Increasing risk per trade (e.g., $0.1\% \to 0.5\% \to 1.0\%$) only when the portfolio equity curve is trading above its own 10 EMA.
3. **Win-Streak / Loss-Streak Multipliers**: Aggressive throttle-down (halving risk) after 2 consecutive losses; scaling back up upon the first structural win.

---

## 4. Multi-Horizon Probability Modeling Methodology

We will build a classification model to provide live, dynamic probabilities for the trader:
1. **Feature Vector Extraction at Time $t$**:
   - Capturing Market Regime, Theme State, Base Type, Volume Pace, and ADR Evolution for every day $t$ post-EP.
2. **Outcome Targets (Prediction Goals)**:
   - $P(\text{Highly Promising: Reaches } +3\text{R or Doubles} \mid \text{Conditions at } t)$
   - $P(\text{Warning Sign / Trap: Reaches stop before } +1\text{R} \mid \text{Conditions at } t)$
3. **Output Generation**:
   - The engine will classify trades into buckets (e.g., "A+ Setup: 78% probability of $+3R$") or issue warnings (e.g., "Warning: ADR contracting under declining Sector 10 EMA; 85% trap probability").

---

## 5. Review & Approval Gate (MANDATORY)

> [!WARNING]
> **AGENT DIRECTIVE**: Do **NOT** execute this plan, write simulation code, or run backtests until the user explicitly approves this artifact.

**To the User:**
Please review this comprehensive research plan. It focuses on:
1. Strict filtering (no penny stocks, Pinnacle Elite base only).
2. Advanced classification for predicting promising setups vs. traps.
3. Bonde delayed entries and context-specific stoplosses.
4. Advanced exits (High-R, ADR extensions, MA stretches).
5. Comprehensive progressive exposure.

**Do I have your confirmation to proceed with building the classification and simulation engine based strictly on this plan?**
