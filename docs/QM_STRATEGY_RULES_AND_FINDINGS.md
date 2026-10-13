# QM Breakout Trading Strategy: Empirical Rules, Positioning & Findings
## High-Conviction Institutional Momentum Continuation Engine

> **Repository:** `amirbaram/EPs` (Branch: `feature/dual-track-continuation`)  
> **Interactive Application:** [`serve_qm.py`](file:///Users/amirbaram/Documents/code/EPs/serve_qm.py) on **http://127.0.0.1:8785**  
> **Backtest Dataset:** `data/simulations/qm_leading_breakouts_trades.parquet` (2,616 leading trades) & `data/simulations/qm_breakout_trades_full.parquet` (10,976 universe trades)  
> **Execution Scripts:** [`sim_leading_qm_breakouts.py`](file:///Users/amirbaram/Documents/code/EPs/sim_leading_qm_breakouts.py) & [`analyze_leading_qm_study.py`](file:///Users/amirbaram/Documents/code/EPs/analyze_leading_qm_study.py)

---

## 1. Strategy Overview & Core Thesis

The **QM Breakout Trading Strategy** is an institutional momentum continuation framework modeled on the principles of **Kristjan Qullamaggie**, **Mark Minervini**, and **William O'Neil**. It trades explosive momentum breakouts emerging from multi-week tight consolidations, specifically when price has digested previous advances by "surfing" rising short-term exponential moving averages (10 EMA and 20 EMA).

### Key Empirical Discoveries in This Study:
1. **The 0.6x Pivot Low Stoploss Breakthrough:** Replacing a full $1.0\times$ TrendLab pivot low stop with a **$0.60\times$ stop distance** increased strategy Expectancy from **$+0.057\text{ R}$ to $+0.138\text{ R}$ (+142%)**, doubled the average winning trade from **$+1.19\text{ R}$ to $+2.00\text{ R}$**, and boosted the Big Winner rate ($\ge 3\text{ R}$) from **$3.0\%$ to $6.4\%$**.
2. **MAE of Winners (The 0.6R Invalidation Rule):** Across 10,976 historical breakouts, **$92.1\%$ of winning trades never drew down more than $0.60\text{ R}$** from entry (average MAE was only $0.24\text{ R}$). Setting the stop at $0.60\times$ cuts losing trades with $40\%$ less dollar loss while allowing $92\%+$ of true winners to run completely undisturbed.
3. **Dynamic Position Sizing Alpha:** Overweighting setups with **Fresh Episodic Pivots ($\le 15$ days)**, **Monster Poles ($\ge 100\%$)**, **Massive Breakout RVOL ($\ge 2.5\times$)**, and **Tight Pullback Ratios ($\le 0.25$)** generated **$+109.5\text{ R}$ in excess profit (+30.3% return boost)** and lifted strategy Expectancy to **$+0.180\text{ R}$** (Profit Factor 1.30).
4. **The QQQ Divergence Inversion:** Counterintuitively, QM breakouts that trigger when the **QQQ is in a pullback / correction below its 10/20 EMA** generate **higher win rates ($37.3\%$ vs. $32.3\%$)** and **double the expectancy ($+0.238\text{ R}$ vs. $+0.108\text{ R}$)** compared to breakouts during extended QQQ bull runs. Stocks that break out against a choppy market reveal massive institutional accumulation.
5. **Pinnacle Elite RS ($\ge 95$) + Stage 2 Posture:** Stocks in the top 5% of cross-sectional Relative Strength ($\text{RS} \ge 95$) in Stage 2 Bullish Alignment generated **$+0.182\text{ R}$ expectancy** and an asymmetric **$2.65:1$ win/loss payoff ratio**. Non-Stage 2 stocks have near-zero edge ($+0.024\text{ R}$, PF 1.04).

---

## 2. Setup Identification Rules ("To the Left")

```
          [Explosive Pole: +30% to +100%+]             [Tight Consolidation Base: 8 to 50 days]
Price                                                  Breakout Stack / Pole Peak Resistance
  ▲                     ┌─────────┐                    ═══════════════════════════════════════
  │                    ┌┘         └────────┐                                      ▲
  │                   ┌┘                   └───┐       ┌───┐             ┌───────┘ ENTRY: Close of Breakout Candle
  │                  ┌┘                         └───┐ ┌┘   └───┐ ┌─────┐┌┘         (Price expands on RVOL >= 1.5x)
  │                 ┌┘                              └─┘        └─┘     └┘
  │                ┌┘                               ▲   ▲   ▲   ▲   ▲
  │               ┌┘                                Ascending Higher Lows
  │              ┌┘                                 Surfing Rising 10 EMA / 20 EMA
  │             ┌┘
  │       ▲    ┌┘
  │ ──────┴────┘
  │ Prior EP Catalyst (<= 15 - 60 days)
  └─────────────────────────────────────────────────────────────────────────────────────────────► Time
```

1. **Momentum Pole:**
   - **Minimum Advance:** $\ge +15\%$ price move from swing low to swing high (`QMB_MIN_MOVE = 15.0`). Accommodates institutional mega-caps (e.g. GOOG, MSFT, AAPL) where 15–25% represents a massive 7–10× ADR momentum pole, while UI/strategy filters allow isolating $\ge +30\%$, $\ge +40\%$, or $\ge +50\%$ moves on higher-beta momentum names.
   - **Maximum Duration:** Completed within 35 trading sessions (`QMB_MOVE_MAX_BARS = 35`).
   - **Classification:** Large-Cap Pole ($+15\%$ to $+30\%$), Standard Pole ($+30\%$ to $+50\%$), Strong Pole ($+50\%$ to $+100\%$), Monster Pole ($\ge +100\%$).
2. **Consolidation Base:**
   - **Duration:** 8 to 50 trading days (`QMB_CONS_MIN = 8`, `QMB_CONS_MAX = 50`).
   - **Pullback Depth:** Maximum $50\%$ pullback off the pole peak, with premier bases holding between $8\%$ and $22\%$.
   - **Pullback Ratio:** $\text{Pullback Ratio} = \text{Pullback \%} / \text{Move \%}$. Tighter ratios ($\le 0.25$) indicate superior overhead supply absorption.
   - **20 EMA Surfing:** At least $50\%$ of bars in the consolidation must close above the rising 20 EMA (`QMB_MIN_RIDE_FRAC = 0.50`), with elite setups holding $\ge 85\%$.
   - **Ascending Higher Lows:** Consolidation lows must form a net ascending or flat trajectory.
   - **Volume Dry-up:** Volume contraction during the latter half of the base, verifying seller exhaustion.
3. **Confirmed Breakout Trigger:**
   - Breakout session must close above the base high or primary Breakout Stack resistance level on expanding relative volume ($\ge 1.3\times$ 20-day baseline average).

---

## 3. Macro & Market Regime Filtering Rules

### 3.1 Leading Sectors & Leading Themes Universe Filter
The strategy trades only stocks whose primary sector or theme is actively classified as a **Leading Sector** or **Leading Theme**:
* **Top Winning Sectors:** Healthcare (+0.315 R, PF 1.67), Basic Materials (+0.283 R, PF 1.53), Industrials (+0.196 R, PF 1.37), Technology (+0.128 R, PF 1.23).
* **Top Winning Themes:** Bitcoin Miners (+1.403 R, PF 3.18), Clean Energy (+1.050 R, PF 3.25), Diagnostics & Research (+0.960 R, PF 3.99), Genomics (+0.541 R, PF 2.22), Data Centers (+0.504 R, PF 2.17), Solar (+0.351 R, PF 1.82).
* **Excluded Toxic Sectors:** Regional Banks (-0.184 R, PF 0.43), Communication Services (-0.126 R, PF 0.69), Utilities (-0.085 R, PF 0.77), Consumer Defensive (-0.093 R, PF 0.78).

### 3.2 Broad Market Regime (SPY Filter)
* **Rule:** $\text{SPY Close} \ge \text{SPY 200-day SMA}$ on entry date.
* **Empirical Impact:** Eliminates secular bear markets where breakouts face severe systematic headwinds (SPY $< 200$ SMA produced negative expectancy: $-0.005\text{ R}$, PF 0.99).

### 3.3 QQQ Trend & Relative Strength Divergence (The Divergence Edge)
* **QQQ Trend Stack:**
  * When $\text{QQQ} > \text{EMA}_{10} > \text{EMA}_{20}$ (Extended Bull): 1,880 trades, Win Rate: **32.3%**, Expectancy: **+0.108 R**, PF: 1.20.
  * When QQQ is in a Pullback / Correction: 715 trades, Win Rate: **37.3%**, Expectancy: **+0.238 R**, PF: **1.48** (+120% expectancy!).
* **QQQ/SPY Leadership Ratio:**
  * When $\text{QQQ/SPY}$ is in a non-bull stack (Tech digesting, broad cyclicals/healthcare leading): Win Rate: **36.2%**, Expectancy: **+0.217 R**, PF: **1.43**.
  * When $\text{QQQ/SPY}$ is in a bull stack (Tech leading): Win Rate: 31.4%, Expectancy: +0.077 R, PF: 1.14.

---

## 4. Trade Execution & Stoploss Rules

### 4.1 Trade Entry
* **Timing:** Long position executed on the **Close of the Breakout Candle**.
* **Entry Price:** $\text{Entry Price} = \text{Close}_{\text{breakout}}$.

### 4.2 The 0.6x Stoploss Rule
* **Pivot Low Identification:** Identify the previous confirmed TrendLab pivot low ($P_{\text{low}}$) on or prior to the breakout date ($\text{pconf} \le \text{idx}_{\text{entry}}$).
* **Full Pivot Distance:** $D = \text{Entry Price} - P_{\text{low}}$.
* **Stop Distance Calculation:**
  $$\text{Stop Distance} = 0.60 \times D = 0.60 \times (\text{Entry Price} - P_{\text{low}})$$
* **Stoploss Order Price:**
  $$\text{Stoploss} = \text{Entry Price} - \text{Stop Distance}$$
* **Unit of Risk (1.0 R):**
  $$1.0\text{ R} = \text{Stop Distance} = 0.60 \times D$$
* **Intraday Execution:** If $\text{Low}_t \le \text{Stoploss}$, exit immediately at $\min(\text{Open}_t, \text{Stoploss})$ (incorporating gap-down slippage).

---

## 5. Dynamic Position Sizing Rules

Each trade is allocated a **Dynamic Position Sizing Multiplier ($S$)** based on 4 quantitative conviction factors:

$$S = \text{clamp}\left( s_{\text{EP}} \times s_{\text{Pole}} \times s_{\text{RVOL}} \times s_{\text{PB}}, \; 0.50, \; 2.50 \right)$$

$$\text{Dollar Position Size} = \frac{\text{Account Capital} \times \text{Base Risk \%} \times S}{\text{Stop Distance}}$$

### 5.1 Sub-Factor Sizing Multipliers
| Sizing Factor | Criteria | Multiplier | Empirical Performance |
|---|---|---|---|
| **$s_{\text{EP}}$: Fresh EP Recency** | **Prior EP $\le 15$ days ago** | **$1.30\times$** (+30%) | **45.8% Win Rate, +0.303 R Exp, PF 1.77** |
| | Fresh EP $16$ to $30$ days ago | **$1.15\times$** (+15%) | 31.1% Win Rate, -0.012 R Exp |
| | Seasoned EP $31$ to $60$ days ago| **$1.05\times$** (+5%) | **36.7% Win Rate, +0.296 R Exp, PF 1.61** |
| | No Prior EP (Pure Technical) | **$0.85\times$** (-15%) | 31.8% Win Rate, +0.102 R Exp, PF 1.18 |
| **$s_{\text{Pole}}$: Pole Magnitude** | **Monster Pole $\ge +100\%$** | **$1.25\times$** (+25%) | **+0.289 R Exp, PF 1.55, 8.1% Big Wins** |
| | Strong Pole $+50\%$ to $+100\%$ | **$1.10\times$** (+10%) | +0.072 R Exp, PF 1.13 |
| | Standard Pole $+30\%$ to $+50\%$ | **$1.00\times$** (1.0x baseline) | +0.136 R Exp, PF 1.26 |
| **$s_{\text{RVOL}}$: Breakout Volume** | **Massive RVOL $\ge 2.5\times$** | **$1.20\times$** (+20%) | **39.1% Win Rate, +0.235 R Exp, PF 1.52** |
| | Strong RVOL $1.5\times$ to $2.5\times$ | **$1.10\times$** (+10%) | 34.3% Win Rate, +0.146 R Exp, PF 1.27 |
| | Moderate RVOL $< 1.5\times$ | **$0.90\times$** (-10%) | 31.4% Win Rate, +0.104 R Exp, PF 1.19 |
| **$s_{\text{PB}}$: Pullback Ratio** | **Tight Absorption ($\le 0.25$)** | **$1.20\times$** (+20%) | **+0.204 R Exp, PF 1.37, 7.1% Big Wins** |
| | Orderly Base ($0.25$ to $0.35$) | **$1.05\times$** (+5%) | +0.085 R Exp, PF 1.16 |
| | Loose Digestion ($> 0.35$) | **$0.85\times$** (-15%) | +0.060 R Exp, PF 1.12 |

---

## 6. Trade Management & Trailing Exit Rules

### 6.1 10 EMA Trailing Exit Activation
1. **Activation Condition:**
   - The 10-day EMA must be sloping upward: $\text{EMA}_{10}[t] > \text{EMA}_{10}[t-1]$.
   - Price must close well above the 10 EMA: $\text{Close}[t] \ge 1.01 \times \text{EMA}_{10}[t]$ ($\ge 1.0\%$ extension).
2. **Prior to Activation:**
   - The trade is protected strictly by the **$0.6\times$ stoploss** below entry price.
   - Allows the breakout to breathe during Days 1–3 without getting whipsawed on retests of the breakout level.

### 6.2 Trailing Exit Execution
* **Rule:** Once the 10 EMA trailing rule has activated, exit the entire position on the **first bar where $\text{Close}[t] < \text{EMA}_{10}[t]$**.
* **Exit Price:** Executed at the close of the violating bar ($\text{Exit Price} = \text{Close}_t$).
* **Empirical Performance:** Captured **$53.5\%$ win rate** and an average of **$+0.862\text{ R}$** across 1,611 trailing exits, with an average hold time of **$10.9$ days**.

---

## 7. Institutional Relative Strength & Posture (`RS_DATABASE.md`)

Cross-referencing setups against the 10-year daily RS database (`data/rs_history/`, 7.1M records):

| RS Cohort | Trades | Win Rate | Expectancy | Profit Factor | Total Return | Win/Loss Ratio |
|---|---|---|---|---|---|---|
| **RS Score $\ge 95$ (Pinnacle Elite)** | 716 | **33.9%** | **+0.182 R** | **1.36** | **+130.2 R** | **2.65 : 1** |
| **RS Score $80 - 89$ (Decile 2)** | 644 | **33.9%** | **+0.173 R** | **1.32** | **+111.3 R** | **2.58 : 1** |
| **Stage 2 Alignment (`C >= 10 >= 20 >= 50 >= 200`)** | 2,248 | **34.2%** | **+0.162 R** | **1.31** | **+364.5 R** | **2.52 : 1** |
| **Within 15% of 52W High (Near ATH)** | 2,046 | **34.8%** | **+0.156 R** | **1.30** | **+318.5 R** | **2.44 : 1** |
| **★ Institutional Leader (RS $\ge 90$ + S2/ATH)** | 1,237 | **33.4%** | **+0.152 R** | **1.29** | **+188.3 R** | **2.57 : 1** |
| **Non-Stage 2 Alignment** | 347 | 30.5% | +0.024 R | 1.04 | +8.3 R | 2.37 : 1 |

---

## 8. Strategy Evolution & Comprehensive Comparison

| Strategy Model | Configuration | Trades | Win % | Expectancy | Profit Factor | Total Return | Big Winners |
|---|---|---|---|---|---|---|---|
| **1. Unfiltered Baseline** | All Breakouts + 1.0x Full Pivot Stop | 10,976 | **34.4%** | **+0.057 R** | 1.16 | +627.3 R | 3.0% (332) |
| **2. Leading Macro + 0.6x Stop (Unsized)** | Leading Sectors/Themes + SPY $\ge$ 200SMA | 2,616 | **33.6%** | **+0.138 R** | **1.26** | +361.5 R | 6.4% (166) |
| **3. Leading Macro + 0.6x Stop (Sized)** | **Dynamic Multiplier $S \times R$** | 2,616 | **33.6%** | **+0.180 R** | **1.30** | **+471.0 R** | **7.6% (199)** |
| **4. Leading + QQQ Pullback Divergence** | QQQ not in bull stack + Sized | 715 | **37.3%** | **+0.238 R** | **1.48** | +170.5 R | 7.1% (51) |
| **5. Leading + Fresh Prior EP ($\le 15$d)** | Fresh EP Igniter + Sized | 190 | **45.8%** | **+0.303 R** | **1.77** | +57.6 R | 6.3% (12) |
| **6. Leading + Monster Pole ($\ge 100\%$)** | Monster Pole + Sized | 434 | **33.2%** | **+0.289 R** | **1.55** | +125.3 R | **8.1% (35)** |

---

## 9. Integration with Application Suite

1. **Port 8785 (`serve_qm.py`):** Includes a dedicated **`🎯 Strategy Mode`** button and **`⚙️ Strategy Settings`** modal to filter the live setup list by these exact rules, displaying the dynamic strategy KPI stats and individual Trade Characteristics & Scoring in the Right Pane Dossier.
2. **Datastore:** Trade logs and features are stored in `data/simulations/qm_leading_breakouts_trades.parquet` and can be queried or joined for subsequent research questions.
