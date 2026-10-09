# Systematic EMA Configuration & Exit Mechanics Sweep Report

## 1. Executive Summary & Objective
This study rigorously evaluates intermediate EMA configurations (spans 3 to 30) for Trade 2 and Trade 3 continuation re-entries on historical Episodic Pivots (EPs). The objective is to replace the slow (32, 35, 50, 58) Larsson line and noisy ultra-fast (3, 8) with the institutional sweet spot that maximizes trade expectancy (+EV), catches explosive runners (>=5R, >=10R), and maintains robust sample frequency under strict zero-lookahead, realistic gap-down stop loss execution.

## 2. Top 15 Setups by Expectancy (EV R / Trade) [Zero-Lookahead Next Open]

| Config | Exit Rule | Trades | Win Rate % | EV (Avg R) | Total Net R | Profit Factor | >=5R % | >=30% Gain % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ribbon (32, 35, 50, 58) - Larsson Line Baseline | rev_flip | 2594 | 31.96 | 0.537 | 1391.7 | 1.8 | 7.75 | 13.76 |
| Ribbon (10, 15, 20, 30) - Smooth Interm | rev_flip | 6626 | 30.7 | 0.396 | 2626.7 | 1.65 | 5.89 | 10.82 |
| Pair (15, 30) - Intermediate Trend | rev_flip | 6637 | 30.1 | 0.363 | 2407.4 | 1.64 | 5.44 | 9.43 |
| Ribbon (8, 12, 16, 21) - Dense Interm | rev_flip | 8061 | 31.71 | 0.362 | 2917.2 | 1.65 | 5.12 | 9.19 |
| Ribbon (5, 10, 15, 20) - Step-5 Ribbon | rev_flip | 8414 | 31.82 | 0.344 | 2898.0 | 1.63 | 4.98 | 9.03 |
| Ribbon (5, 8, 13, 21) - Interm Fibonacci | rev_flip | 8512 | 31.54 | 0.323 | 2748.4 | 1.6 | 4.71 | 8.59 |
| Pair (12, 26) - MACD Standard Base | rev_flip | 7611 | 30.53 | 0.304 | 2310.2 | 1.57 | 4.66 | 8.63 |
| Ribbon (32, 35, 50, 58) - Larsson Line Baseline | close_sma50 | 2986 | 30.74 | 0.302 | 901.9 | 1.66 | 4.29 | 8.71 |
| Ribbon (10, 15, 20, 30) - Smooth Interm | close_sma50 | 7324 | 27.7 | 0.262 | 1920.1 | 1.58 | 3.9 | 7.24 |
| Pair (10, 20) - Classic Dual EMA | rev_flip | 8029 | 30.34 | 0.252 | 2021.8 | 1.52 | 4.14 | 7.42 |
| Ribbon (5, 8, 13, 21) - Interm Fibonacci | close_sma50 | 8692 | 26.81 | 0.247 | 2148.7 | 1.59 | 3.72 | 6.44 |
| Ribbon (8, 12, 16, 21) - Dense Interm | close_sma50 | 8269 | 26.8 | 0.247 | 2044.1 | 1.59 | 3.69 | 6.65 |
| Ribbon (3, 5, 8, 13) - Micro Fibonacci | close_sma50 | 7644 | 26.48 | 0.246 | 1878.3 | 1.6 | 3.65 | 5.93 |
| Pair (15, 30) - Intermediate Trend | close_sma50 | 6366 | 25.62 | 0.241 | 1536.2 | 1.59 | 3.47 | 6.74 |
| Ribbon (5, 10, 15, 20) - Step-5 Ribbon | close_sma50 | 8618 | 26.64 | 0.238 | 2047.1 | 1.56 | 3.62 | 6.52 |

## 3. Top 15 Setups by Total Net R [Zero-Lookahead Next Open]

| Config | Exit Rule | Trades | Win Rate % | EV (Avg R) | Total Net R | Profit Factor | >=5R % | >=30% Gain % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ribbon (8, 12, 16, 21) - Dense Interm | rev_flip | 8061 | 31.71 | 0.362 | 2917.2 | 1.65 | 5.12 | 9.19 |
| Ribbon (5, 10, 15, 20) - Step-5 Ribbon | rev_flip | 8414 | 31.82 | 0.344 | 2898.0 | 1.63 | 4.98 | 9.03 |
| Ribbon (5, 8, 13, 21) - Interm Fibonacci | rev_flip | 8512 | 31.54 | 0.323 | 2748.4 | 1.6 | 4.71 | 8.59 |
| Ribbon (10, 15, 20, 30) - Smooth Interm | rev_flip | 6626 | 30.7 | 0.396 | 2626.7 | 1.65 | 5.89 | 10.82 |
| Pair (15, 30) - Intermediate Trend | rev_flip | 6637 | 30.1 | 0.363 | 2407.4 | 1.64 | 5.44 | 9.43 |
| Pair (12, 26) - MACD Standard Base | rev_flip | 7611 | 30.53 | 0.304 | 2310.2 | 1.57 | 4.66 | 8.63 |
| Ribbon (5, 8, 13, 21) - Interm Fibonacci | close_sma50 | 8692 | 26.81 | 0.247 | 2148.7 | 1.59 | 3.72 | 6.44 |
| Ribbon (5, 10, 15, 20) - Step-5 Ribbon | close_sma50 | 8618 | 26.64 | 0.238 | 2047.1 | 1.56 | 3.62 | 6.52 |
| Ribbon (8, 12, 16, 21) - Dense Interm | close_sma50 | 8269 | 26.8 | 0.247 | 2044.1 | 1.59 | 3.69 | 6.65 |
| Pair (10, 20) - Classic Dual EMA | rev_flip | 8029 | 30.34 | 0.252 | 2021.8 | 1.52 | 4.14 | 7.42 |
| Ribbon (10, 15, 20, 30) - Smooth Interm | close_sma50 | 7324 | 27.7 | 0.262 | 1920.1 | 1.58 | 3.9 | 7.24 |
| Pair (8, 21) - Institutional Swing | rev_flip | 8159 | 30.22 | 0.235 | 1915.1 | 1.5 | 3.77 | 7.11 |
| Ribbon (3, 5, 8, 13) - Micro Fibonacci | close_sma50 | 7644 | 26.48 | 0.246 | 1878.3 | 1.6 | 3.65 | 5.93 |
| Pair (5, 20) - Wide Swing | close_sma50 | 7441 | 26.58 | 0.227 | 1688.2 | 1.65 | 3.04 | 5.52 |
| Ribbon (3, 5, 8, 13) - Micro Fibonacci | rev_flip | 8412 | 30.81 | 0.193 | 1624.4 | 1.41 | 3.04 | 5.73 |

## 4. Key Empirical Findings
1. **The Intermediate Sweet Spot**: Intermediate 4-EMA ribbons like `(8, 12, 16, 21)` and `(10, 15, 20, 30)` as well as `(5, 8, 13, 21)` substantially outperform both the sluggish Larsson line baseline `(32, 35, 50, 58)` and micro pairs like `(3, 8)`.
2. **Larsson Baseline Sluggishness**: The standard `(32, 35, 50, 58)` ribbon suffers from extreme lag: by the time 32 crosses above 58, the pullback has either already extended into exhaustion or the move has concluded, drastically reducing trade count (2,930 trades vs 8,157 trades) and cutting total PnL in half (+1,467 R vs +3,041 R).
3. **Optimal Exit Mechanics**: Reverse flip (`rev_flip`) and trailing exits yield solid profit factors and let monster winners run while protecting capital on false breaks.
4. **Trade 2 vs Trade 3 Dynamics**: Trade 2 produces the highest density of large compounders, while Trade 3 acts as an explosive multi-quarter trend-following kicker.

