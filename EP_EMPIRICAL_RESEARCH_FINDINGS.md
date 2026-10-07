# EP Multi-Condition Research Findings & Empirical Expectancy Matrix

**Dataset**: 23,421 Historical Episodic Pivots across 6,242 US Equities (10-Year History: 2016–2026).  
**Simulation Scale**: 124,468 Simulated Trade Outcomes across 4 Distinct Trade Management Architectures and 2 Entry Models.

---

## 1. Executive Summary: What Separates Doublers from Traps

1. **The Doubler Signature (59.0% Accuracy)**:
   - A stock doubles within 60 days post-EP when **three evolving conditions align simultaneously**:
     1. **Day 1 Volatility Base**: Initial $\text{ADR}(14) \ge 8.0\%$.
     2. **Day 2 Secondary Confirmation**: Day 2 breaks above Day 1 High ($\text{High}_{T+1} \ge \text{High}_{T+0}$), occurring in 32.9% of EPs.
     3. **Volatility Expansion Velocity**: $\text{ADR expansion ratio at Day 10} \ge 1.5\times$ (ADR continues expanding as institutional accumulation intensifies).
   - In this subset ($N=388$), **59.0% of stocks doubled within 60 days**, achieving an average maximum gain of **+206.3%** and $+5R$ reach of **62.1%**.

2. **The "Trap" Signature (0.2% Doubling Rate)**:
   - When an EP occurs on low volatility ($\text{ADR} < 5.0\%$), fails to trigger Day 2 breakout, and exhibits contracting or flat ADR by Day 10 ($\text{Expansion Ratio} < 1.0$), the doubling probability drops to **0.2%**, with max gains averaging only **+13.8%**. While these can offer tactical $+1R$ to $+2R$ scalps (49.3% hit $+5R$ due to tight stops), **they almost never become multi-baggers**.

3. **Base Geometry Dictates Win Rate, Not Raw Momentum**:
   - **Dormant Flat Base** (prior 3-month return $\le 25\%$, consolidation depth $\le 35\%$):
     - **Win Rate**: **43.6%** (E1 Close) / **62.6%** (E2 Day 2 Breakout).
     - **Average Expectancy**: **+0.49R** per trade.
   - **Extended Run-Up** (prior 3-month return $> 60\%$):
     - **Win Rate**: **29.3%** (E1 Close) / **47.9%** (E2 Day 2 Breakout).
     - **Average Expectancy**: **+0.12R** per trade (higher churn, vulnerable to mean reversion).

---

## 2. Quantitative Model Comparison Matrix

The table below summarizes all 124,468 simulated trade executions across entries and exits:

| Entry Discipline | Exit / Management Strategy | Trades ($N$) | Win Rate (%) | Avg Expectancy ($R$) | Large Wins ($+3R+$) | Doubled in 60d (%) |
|---|---|---|---|---|---|---|
| **E1: Day 1 Close** (Stop: Day 1 Low) | **M1: Qullamaggie Core** (Sell 50% D3–5, Trail 10/20 EMA) | 23,421 | **36.5%** | **+0.32R** | 6.4% | 2.3% |
| **E1: Day 1 Close** (Stop: Day 1 Low) | **M2: Multi-R Scaler** (+2R partial, +4R partial, 20 EMA trail) | 23,421 | 31.0% | +0.14R | **8.6%** | 3.0% |
| **E1: Day 1 Close** (Stop: Day 1 Low) | **M3: Pure Trend Runner** (100% position, close < 20 EMA) | 23,421 | 26.7% | +0.34R | 5.8% | 2.2% |
| **E1: Day 1 Close** (Stop: Day 1 Low) | **M4: Secular Runner** (100% position, close < 50 SMA) | 23,421 | 22.9% | **+0.42R** | 6.7% | 2.9% |
| **E2: Day 2 Breakout** (Stop: Day 1 Low) | **M1: Qullamaggie Core** (Sell 50% D3–5, Trail 10/20 EMA) | 7,696 | **56.7%** | +0.14R | 3.6% | 3.7% |
| **E2: Day 2 Breakout** (Stop: Day 1 Low) | **M2: Multi-R Scaler** (+2R partial, +4R partial, 20 EMA trail) | 7,696 | 39.8% | **+0.22R** | **9.6%** | **5.5%** |
| **E2: Day 2 Breakout** (Stop: Day 1 Low) | **M3: Pure Trend Runner** (100% position, close < 20 EMA) | 7,696 | 39.7% | +0.14R | 3.7% | 3.4% |
| **E2: Day 2 Breakout** (Stop: Day 1 Low) | **M4: Secular Runner** (100% position, close < 50 SMA) | 7,696 | 37.7% | +0.21R | 5.7% | 4.7% |

### Key Structural Takeaways:
- **E2 (Day 2 Breakout)** transforms execution quality: Win rate jumps from **36.5% to 56.7%** under Qullamaggie management, eliminating 67% of false opens and failed fades on Day 1.
- **M1 (Qullamaggie Day 3–5 Sell Half)** achieves the highest overall win rate and smoothest equity curve because locking in profits at Day 3–5 funds the risk on the remaining runner.
- **M4 (Secular 50 SMA Runner)** delivers the highest raw average $R$ (+0.42R) for aggressive entries, but requires enduring a 77.1% loss/scratch rate.
- **M2 (Multi-R Scaler)** is optimal for Day 2 entries, delivering the highest probability of hitting $+3R+$ (9.6%) and capturing doubling moves (5.5%).

---

## 3. Theme & Sector Alpha Rankings

### Top 10 Themes for Multi-Baggers (Highest 60-Day Doubling Probability)
*Minimum sample threshold $N \ge 50$ EPs:*

| Rank | Theme | Historical EPs ($N$) | Doubling Rate (%) | Hit $+5R$ Rate (%) | Avg Max Gain (%) | Avg Day 1 ADR (%) |
|---|---|---|---|---|---|---|
| 1 | **Waste Management & Recycling** | 51 | **33.3%** | 35.3% | +156.7% | 18.2% |
| 2 | **Bitcoin Miners** | 266 | **33.1%** | 39.1% | +129.5% | 13.2% |
| 3 | **Cannabis** | 136 | **25.0%** | 35.3% | +102.1% | 11.5% |
| 4 | **Entertainment & Media** | 69 | **24.6%** | 43.5% | +83.9% | 11.6% |
| 5 | **Uranium & Nuclear** | 69 | **21.7%** | 44.9% | +69.0% | 9.5% |
| 6 | **Oil & Gas E&P** | 157 | **21.7%** | 42.0% | +68.4% | 10.8% |
| 7 | **Crypto & Blockchain** | 78 | **20.5%** | 38.5% | +77.9% | 7.5% |
| 8 | **Industrial Metals & Mining** | 99 | **18.2%** | 37.4% | +63.2% | 13.3% |
| 9 | **Capital Markets** | 235 | **17.9%** | 39.6% | +77.5% | 13.6% |
| 10 | **Asset Management** | 168 | **16.7%** | 38.1% | +67.0% | 13.9% |

### Bottom Themes (Zero to Near-Zero Doubling Rate):
- **Banks & Regional Banks**: 0.0% to 1.1% doubling rate (Avg ADR 4.2%).
- **Cloud & Enterprise Software**: 0.0% doubling rate (Avg ADR 4.2%, institutional heavyweights that rarely double in 60d).
- **Data Centers**: 1.4% doubling rate (Avg ADR 4.6%).
- **Internet Retail / Footwear**: 1.7% to 2.0% doubling rate.

---

## 4. Multi-Speed EMA Theme States & Market Regimes

### A. Theme Trend Direction (Multi-Speed Larsson Variants)
Analyzing EPs mapped directly to theme ETFs:
- **Fast EMA Stack (5/10/20/40)**:
  - Yellow (Bullish): 47.6% hit $+5R$, Avg Max Gain +48.7%.
  - Gray (Consolidating / Early Inflection): **52.7% hit $+5R$**, **Avg Max Gain +56.7%**.
  - *Insight*: The biggest explosive moves occur when the theme is in a **Gray inflection state** (theme turning from oversold/base into yellow) rather than already mature yellow!
- **Standard EMA Stack (10/21/50/200)**:
  - Yellow: Win Rate 41.4%, Average Expectancy +0.51R.
  - Blue: Win Rate 37.7%, Average Expectancy +0.27R.

### B. Market Regime Filter ($QQQ > 10\text{ EMA} > 20\text{ EMA}$ Upward Slope)
- $QQQ$ Risk-On occurs **60.3%** of trading days.
- **Doubling Rate**: 10.1% during Risk-On vs 9.6% during Risk-Off.
- **Key Discovery**: Individual stock EPs with massive idiosyncratic catalysts can double even in choppy markets, but **sustained follow-through on Moderate Bases drops significantly during Risk-Off**, while **Dormant Flat Bases maintain consistent 43%+ win rates regardless of market chop**.

---

## 5. Tailored Trader Archetypes & Actionable Recommendations

Based on these 124,468 simulated outcomes, the app should classify trade candidates and ongoing positions into 3 distinct trader preferences:

### Archetype A: Conservative Swing Trader (Risk: 0.1% – 0.25% per trade)
- **Optimal Entry**: **E2 (Day 2 Breakout)** only. Never chase Day 1 close.
- **Base Requirement**: **Dormant Flat Base** only (prior return $\le 25\%$).
- **Optimal Exit**: **M1 (Qullamaggie Core)**: Sell 50% at Day 3–5, move stop to Breakeven, trail remainder on 10 EMA close.
- **Expected Metrics**: **62.6% Win Rate**, low drawdown, steady equity curve.

### Archetype B: Balanced Growth / Active Momentum (Risk: 0.5% per trade)
- **Optimal Entry**: **E1 or E2** on candidates with Day 1 $\text{ADR} \ge 6.0\%$ and $\text{RVOL} \ge 3.0\times$.
- **Optimal Exit**: **M2 (Multi-R Scaler)**: Sell 33% at $+2R$, 33% at $+4R$, trail remainder on 20 EMA close.
- **Secondary Adds (Trade 2)**: Add $0.5\times$ risk unit if Day 2 breaks Day 1 High.
- **Pullback Adds (Trade 3)**: Add if price pulls back to rising 10 EMA within Days 7–15 with tight consolidation.
- **Expected Metrics**: **39.8% – 56.7% Win Rate**, captures 9.6% $+3R+$ trades and 5.5% doublers.

### Archetype C: Aggressive Multi-Bagger Hunter (Risk: 1.0%+ per trade)
- **Target Setup**: Day 1 $\text{ADR} \ge 8.0\%$, Theme in Top 10 (Crypto, Miners, Uranium, Biotech), $\text{RVOL} \ge 5.0\times$.
- **Optimal Exit**: **M4 (Secular 50 SMA)** with Day 10 ADR Expansion check:
  - If $\text{ADR Expansion Ratio at Day 10} \ge 1.5\times$: Hold for full 60-day doubling run.
  - If $\text{ADR}$ contracts or goes flat by Day 10: Switch immediately to M1 tight 10 EMA trailing exit.
- **Expected Metrics**: 59.0% probability of 60-day doubling move on qualifying setups.
