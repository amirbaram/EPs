# Empirical Research Findings: Multi-Stoploss Entries & Comprehensive Trade Management

**Dataset**: 23,421 Historical Episodic Pivots across 6,242 US Equities (2016–2026).  
**Scale**: **466,952 simulated trade outcomes** across 6 Entry/Stoploss disciplines and 8 Trade Management systems.

---

## 1. Stop-Loss Structure Analysis: Candle Low vs. Pivot Low vs. Tight Flag Low

| Entry Archetype | Stop-Loss Basis | Sample ($N$) | Win Rate (%) | Scratch Rate (%) | Avg Expectancy ($R$) | Hit $+3R+$ (%) | 60d Doubler (%) |
|---|---|---|---|---|---|---|---|
| **E1: Day 1 Close** | **Candle Low** (Day 1 Low) | 186,776 | 25.9% | 6.8% | **+0.05R** | **8.1%** | **2.5%** |
| **E1: Day 1 Close** | **Prior Pivot Low** (SwingsTimes) | 106,400 | **41.0%** | 10.5% | +0.05R | 2.6% | 1.7% |
| **E2: Day 2 Breakout** | **Day 1 Low** (Master Support) | 61,552 | **38.9%** | 8.2% | +0.17R | 5.8% | **4.2%** |
| **E2: Day 2 Breakout** | **Day 2 Low** (Tight Intraday) | 55,855 | 28.3% | 7.9% | **+0.35R** | **11.4%** | 2.4% |
| **E3: Bonde Delayed Flag**| **Consolidation Flag Low** (Tight 3%–6%) | 26,157 | **55.5%** | 6.4% | **+0.64R** | **8.2%** | 1.3% |
| **E3: Bonde Delayed Flag**| **Master Day 1 Low** (Wide Guardrail) | 29,144 | **61.5%** | 6.5% | **+0.61R** | 7.3% | 2.1% |

### Key Structural Takeaways:
1. **The Candle Low vs Prior Pivot Low Trade-Off**:
   - For Day 1 entries, using the **Prior Pivot Low** dramatically elevates the win rate from **25.9% to 41.0%** (+15.1% win rate increase) by absorbing intraday shakeouts and opening auction backfills.
   - However, because the stop distance is wider, the $R$-multiple leverage is smaller: only 2.6% of trades reach $+3R+$ (vs 8.1% on Candle Low).
   - **Conclusion**: Conservative traders seeking high win rates and low churn benefit from the **Prior Pivot Low**, whereas aggressive compounding traders require the tighter **Candle Low**.
2. **The Power of Bonde Delayed Flag Entries (E3)**:
   - Bonde Delayed Entries achieve the **highest mathematical expectancy in the entire dataset**: **+0.61R to +0.64R per trade**, with win rates consistently above **55%–61%**.
   - Placing the stop at the **tight flag low** delivers the highest expectancy (+0.64R) because the risk unit is shrunk to 3%–6%, generating massive asymmetric $R$-multiples on subsequent breakouts.

---

## 2. Trade Management Strategy Comparison

*Trimmed distribution ($-2.0R \le R \le +30.0R$) across all 58,000+ trades per model:*

| Strategy Model | Core Mechanics | Win Rate (%) | Median $R$ | Avg Expectancy ($R$) | Hit $+3R+$ (%) | Doubler Capture (%) |
|---|---|---|---|---|---|---|
| **TM_HighR_Ladder** | Sell 25% @ $+3R$, 25% @ $+5R$ (BE @ $+3R$), trail 20 EMA | 34.2% | -1.00R | **+0.26R** | **12.4%** | **3.0%** |
| **TM_Apex_Asym** | Sell 20% @ $+4R$, $+8R$, $+12R$ (BE @ $+3R$), trail 50 SMA | 35.2% | -0.48R | **+0.21R** | 7.7% | 2.7% |
| **TM_Premature_BE1p5R** | Control: Move BE @ $+1.5R$, sell 33% @ $+2R$ | 36.3% | -1.00R | +0.20R | 5.4% | 2.6% |
| **TM_ADR_Targets** | Sell 33% @ $+3\times\text{ADR}$, 33% @ $+5\times\text{ADR}$ (BE @ $+3R$) | 36.2% | -1.00R | +0.19R | 6.7% | 2.7% |
| **TM_Qulla_BE3R** | Sell 50% Day 3–5 into strength, BE @ $+3R$, trail 20 EMA | **40.3%** | **-0.21R** | +0.14R | 4.6% | 2.4% |
| **TM_Secular_SMA50** | 0% partials, BE @ $+3R$, trail 50 SMA close | 31.6% | -0.49R | +0.12R | 5.4% | 2.7% |
| **TM_SMA50_Climax_Ext** | Scale 50% on $>40\%$ stretch above 50 SMA, trail 20 EMA | 34.4% | -0.24R | +0.11R | 4.3% | 1.7% |
| **TM_Pure_EMA20_BE3R** | 0% partials, BE @ $+3R$, trail 20 EMA close | 33.5% | -0.28R | +0.10R | 4.5% | 2.0% |

### Key Strategy Takeaways:
1. **The High-R Ladder Dominance**:
   - `TM_HighR_Ladder` (+3R and +5R partials with BE locked at +3R) delivers the **highest expectancy (+0.26R)** and the **highest proportion of $+3R+$ trades (12.4%)**.
   - By waiting until $+3.0R$ to take the first partial and move the stop to breakeven, it avoids the premature scratch whipsaw while locking in meaningful gains before trailing the 20 EMA.
2. **ADR Multiples Performance**:
   - `TM_ADR_Targets` (+3x ADR and +5x ADR) performs strongly (+0.19R), providing an adaptive milestone that automatically flexes to high-volatility names (e.g. Biotech, Miners) versus low-volatility large caps.
3. **Qullamaggie Day 3–5 Smoothing**:
   - `TM_Qulla_BE3R` delivers the **highest win rate (40.3%)** and the **lowest median loss (-0.21R)**, providing the smoothest equity curve for active swing traders.

---

## 3. Bonde Delayed Flag Breakout (E3) Strategy Breakdown

When isolating Bonde Delayed Breakouts ($N=55,301$ trade simulations), the performance jumps significantly across all models:

| Strategy on Bonde Delayed Flag (E3) | Win Rate (%) | Avg Expectancy ($R$) | Hit $+3R+$ (%) | Doubler Rate (%) |
|---|---|---|---|---|
| **TM_HighR_Ladder** | 54.2% | **+0.89R** | **16.2%** | **2.3%** |
| **TM_ADR_Targets** | **65.0%** | +0.75R | 7.1% | 2.0% |
| **TM_Apex_Asym** | 60.4% | +0.66R | 7.3% | 1.9% |
| **TM_Secular_SMA50** | 59.9% | +0.66R | 6.4% | 1.9% |
| **TM_Qulla_BE3R** | **66.7%** | +0.45R | 3.1% | 1.6% |

- In a **Bullish Theme Regime (Yellow)**, Bonde Delayed Entries achieve a **65.3% Win Rate** and **+0.75R average expectancy**.
- In an **Inflecting/Reversal Theme Regime (Blue-to-Yellow / Early Inflection)**, expectancy surges to **+1.27R per trade** (66.7% win rate) as the stock catches the full thematic accumulation wave from the ground floor.
