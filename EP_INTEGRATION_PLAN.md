# EP Research & Integration Plan (V3)

> **Important Reference:** See [LIVE_APP_INTEGRATION_INSIGHTS.md](file:///Users/amirbaram/Documents/code/EPs/LIVE_APP_INTEGRATION_INSIGHTS.md) for the complete quantitative insights, calibrated benchmarks (38.9% win rate vs 61% drift, 15:1 asymmetric payoff), code fixes, and step-by-step implementation standards for `serve_ep_tracker.py`.

To ensure the prediction model is truly comprehensive, we must incorporate all proprietary Playbook Insights and Expectancy Boosters defined in `serve_ep.py` before we integrate the engine into the live apps. 

## Phase 1: Advanced Feature Engineering & ML Extraction
Our previous simulation only looked at basic Theme State, Market Regime, and EP Type. We will now expand the simulation and dataset to explicitly track and calculate probabilities based on your proprietary compound features:

**Playbook Insights to be modeled:**
- **Emerging Leaders**: (Group Momentum $\ge 65$th Pctile + 48H Absorbed + RVOL $\ge 2.5x$ + ClosePos $\ge 0.65$)
- **Institutional Sweet Spot**: (48H Held + ClosePos $\ge 0.65$ + RVOL $\ge 3.0x$ + Gap $\ge 6\%$)
- **Multi-Quarter Leaders (Compounders)**: (Top Tech/Semis/Nuclear/Biotech + 48H Held)
- **Neglected Turnarounds**: (6M Neglect $\le -15\%$ + RVOL $\ge 4.0x$ + ClosePos $\ge 0.70$)
- **Toxic Traps**: (48H Violated OR ClosePos $< 0.50$)

**Expectancy Boosters (Binary Features):**
- Veto Severe Headwinds applied
- Winning Sectors Only
- Momentum Tailwind ($\ge 60$th Pctile)
- 48H Upper Body Absorption
- Elite ClosePos ($\ge 0.80$)
- 5D Support Held (Day 1 Low intact for 5 days)

## Phase 2: Predictive Target & Consolidation Modeling
Instead of just outputting an overall Expected Value (EV), the prediction engine will be trained on the new features to output specific probabilistic events:
1. **Target Range Prediction**: What is the mathematical probability that this specific setup (e.g. an "Emerging Leader" with a "Momentum Tailwind") will hit $+3x$ ADR vs $+5x$ ADR?
2. **Consolidation Probability**: What is the probability that the stock will enter a deep multi-week consolidation *before* hitting the $+3x$ ADR target? (This tells the trader whether they can hold blindly or if they need to actively trade around a core position).
3. **Caution/Veto Flags**: Explicitly flagging when a setup transitions into a "Toxic Trap" profile or violates the 48H absorption rule, triggering an immediate abort recommendation.

## Phase 3: Historical Dashboard Integration (App 8782)
- **Time-Series Prediction Overlay**: Add a new panel below the main price chart.
- For every day after the EP trigger, this panel will display the prediction engine's daily output, showing exactly how the probabilities of hitting targets vs consolidating evolved live as the Expectancy Boosters (like 5D Support Held) activated or failed.

## Phase 4: Live Portfolio Management (App 8783)
- **Info Panel Augmentation**: When clicking an active holding, display the active Playbook Insight classification, the predicted Price Target Range, and the % probability of hitting it vs consolidating.
- **Proactive Alerts**: A notification feed that fires when the model flags a critical state change (e.g., "5D Support Violated - Downgrading to Trap Status", or "Target Range Reached - Scale Out").
