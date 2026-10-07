# Larsson Line Analysis Insights

## Phase 1: Shorts (Blue Flips) Baseline Discovery

**Execution Date:** 2026-10-03
**Test Parameters:**
- Entry: On the close of a fresh flip to strictly BLUE.
- Stoploss: Placed at the most recent Pivot High prior to the flip.
- Exit Strategy: Exit when the state turns strictly GREY or YELLOW (i.e., no longer blue).
- Data: Entire US Market (2057 tickers).

### Results Overview
The baseline analysis for shorting strictly on Blue flips shows an overwhelmingly **negative expectancy** across almost all sectors. The upward drift of the equity market severely penalizes naive shorting. 

**The few marginally positive sectors:**
1. **Capital Markets:** +21.13 R (Avg R: 0.05, Win Rate: 22.4%, 411 Trades)
2. **Luxury Goods:** +16.06 R (Avg R: 0.25, Win Rate: 26.2%, 65 Trades)
3. **Thermal Coal:** +9.47 R (Avg R: 0.34, Win Rate: 25.0%, 28 Trades)
4. **Chemicals:** +8.67 R (Avg R: 0.07, Win Rate: 26.4%, 121 Trades)
5. **Oil & Gas Drilling:** +5.86 R (Avg R: 0.07, Win Rate: 29.2%, 89 Trades)

**The worst-performing sectors:**
1. **Biotechnology:** -417.85 R
2. **Software - Infrastructure:** -316.61 R
3. **Semiconductors:** -271.02 R
4. **Banks - Regional:** -264.20 R
5. **Software - Application:** -202.65 R

### Key Insight
Shorting purely on a blue flip is mathematical suicide in high-growth/tech sectors (Biotech, Software, Semis). To make shorting viable, we *must* apply a strict structural filter. 

**Next Steps for Shorts:**
Just as we did for Yellow Flips (Combo B), we need to test if shorting only when the overarching Sector ETF is in a severe downtrend (e.g. Sector ETF < 200 SMA & 10 SMA < 20 SMA) can push the expectancy positive.
