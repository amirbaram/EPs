"""analyze_leading_qm_study.py — Quantitative Analysis of Leading QM Breakouts with 0.6x Stoploss & Dynamic Position Sizing.

Answers:
1. Performance stats with 0.6x stoploss and dynamic sizing.
2. Comparison: Unsized R vs Sized R.
3. Analysis of QQQ > EMA10 > EMA20 regime impact on win rate and expectancy.
4. Analysis of QQQ/SPY relative strength regime impact (Tech leading vs lagging).
5. Analysis of Stock RS Score & Technical Posture (per RS_DATABASE.md).
6. Patterns among winners vs losers across sectors, themes, catalysts, and geometry.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

DATA_PATH = Path("data/simulations/qm_leading_breakouts_trades.parquet")


def compute_metrics(df: pd.DataFrame, label: str = "", r_col: str = "r_multiple") -> dict:
    n = len(df)
    if n == 0:
        return {"name": label, "count": 0}
    r = df[r_col].to_numpy(float)
    wins = df[df[r_col] > 0]
    losses = df[df[r_col] <= 0]
    scratches = df[df[r_col].abs() <= 0.3]
    big_wins = df[df[r_col] >= 3.0]
    monster_wins = df[df[r_col] >= 5.0]
    full_losses = df[df[r_col] <= -0.9]
    
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = (n_wins / n) * 100.0
    
    sum_win_r = wins[r_col].sum()
    sum_loss_r = abs(losses[r_col].sum())
    profit_factor = (sum_win_r / sum_loss_r) if sum_loss_r > 0 else np.nan
    
    avg_r = float(np.mean(r))
    med_r = float(np.median(r))
    std_r = float(np.std(r)) if n > 1 else 0.0
    
    avg_win = float(wins[r_col].mean()) if n_wins > 0 else 0.0
    avg_loss = float(losses[r_col].mean()) if n_losses > 0 else 0.0
    win_loss_ratio = (avg_win / abs(avg_loss)) if avg_loss != 0 else np.nan
    
    expectancy = (win_rate / 100.0 * avg_win) + ((1.0 - win_rate / 100.0) * avg_loss)
    avg_gain_pct = float(df["gain_pct"].mean())
    avg_bars = float(df["bars_in_trade"].mean())
    avg_bars_win = float(wins["bars_in_trade"].mean()) if n_wins > 0 else 0.0
    avg_bars_loss = float(losses["bars_in_trade"].mean()) if n_losses > 0 else 0.0
    
    return {
        "name": label,
        "count": n,
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2) if pd.notna(profit_factor) else 0.0,
        "avg_r": round(avg_r, 3),
        "med_r": round(med_r, 3),
        "std_r": round(std_r, 3),
        "expectancy": round(expectancy, 3),
        "total_r": round(float(np.sum(r)), 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "win_loss_ratio": round(win_loss_ratio, 2) if pd.notna(win_loss_ratio) else 0.0,
        "big_win_pct": round((len(big_wins) / n) * 100.0, 1),
        "monster_win_pct": round((len(monster_wins) / n) * 100.0, 1),
        "full_loss_pct": round((len(full_losses) / n) * 100.0, 1),
        "scratch_pct": round((len(scratches) / n) * 100.0, 1),
        "avg_gain_pct": round(avg_gain_pct, 2),
        "avg_bars": round(avg_bars, 1),
        "avg_bars_win": round(avg_bars_win, 1),
        "avg_bars_loss": round(avg_bars_loss, 1),
        "max_r": round(float(np.max(r)), 2),
        "min_r": round(float(np.min(r)), 2),
    }


def print_table(rows: list[dict], title: str, sort_key: str = "expectancy", ascending: bool = False):
    print("=" * 115)
    print(f"  {title.upper()}")
    print("=" * 115)
    if not rows:
        print("No data.")
        return
    valid_rows = [r for r in rows if r["count"] > 0]
    if sort_key:
        valid_rows = sorted(valid_rows, key=lambda x: x.get(sort_key, 0), reverse=not ascending)
        
    hdr = f"{'Cohort':<36} | {'Count':>5} | {'Win%':>6} | {'Avg R':>7} | {'Exp (R)':>7} | {'PF':>5} | {'Tot R':>8} | {'Win/Loss':>8} | {'BigWin%':>7} | {'Days':>5}"
    print(hdr)
    print("-" * 115)
    for r in valid_rows:
        line = (
            f"{r['name'][:36]:<36} | {r['count']:>5} | {r['win_rate']:>5.1f}% | {r['avg_r']:>+7.3f} | "
            f"{r['expectancy']:>+7.3f} | {r['profit_factor']:>5.2f} | {r['total_r']:>+8.1f} | "
            f"{r['win_loss_ratio']:>8.2f} | {r['big_win_pct']:>6.1f}% | {r['avg_bars']:>5.1f}"
        )
        print(line)
    print("-" * 115)
    print()


def main():
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found.")
        sys.exit(1)
        
    df = pd.read_parquet(DATA_PATH)
    print("==========================================================================================")
    print(f"📊 LEADING QM BREAKOUT STRATEGY STUDY ({len(df):,} TRADES)")
    print("   Filters: Leading Sectors & Themes | SPY >= 200 SMA")
    print("   Stoploss: 0.6x Distance to Previous TrendLab Pivot Low")
    print("   Exit: First Close Below 10 EMA (Once 10 EMA is Rising & Price Closed Well Above)")
    print("==========================================================================================")
    
    # 1. Baseline Performance (Unsized vs Sized)
    base_unsized = compute_metrics(df, "1. Unsized Baseline (1.0R Base)", r_col="r_multiple")
    base_sized = compute_metrics(df, "2. Position-Sized Model (Dynamic S*R)", r_col="sized_r_multiple")
    
    print("\n[1] HEADLINE TRADING PERFORMANCE:")
    print(f"  • Total Trades Executed       : {base_unsized['count']:,}")
    print(f"  • Win Rate                    : {base_unsized['win_rate']:.1f}% ({df['is_win'].sum():,} wins / {(~df['is_win']).sum():,} losses)")
    print(f"  • Unsized Strategy Return     : {base_unsized['total_r']:+,.1f} R (Expectancy: {base_unsized['expectancy']:+.3f} R, PF: {base_unsized['profit_factor']:.2f})")
    print(f"  • Position-Sized Total Return : {base_sized['total_r']:+,.1f} R (Expectancy: {base_sized['expectancy']:+.3f} R, PF: {base_sized['profit_factor']:.2f})")
    print(f"  • Sizing Multiplier Alpha     : Sizing generated {base_sized['total_r'] - base_unsized['total_r']:+,.1f} R in excess profit (+{(base_sized['total_r']/base_unsized['total_r']-1)*100:.1f}% boost!)")
    print(f"  • Average Winner / Loser      : {base_unsized['avg_win']:+.2f} R / {base_unsized['avg_loss']:+.2f} R (Win/Loss Ratio: {base_unsized['win_loss_ratio']:.2f} : 1)")
    print(f"  • Big Winners (>= 3.0 R)      : {base_unsized['big_win_pct']:.1f}% ({df['is_big_win'].sum():,} trades, up to {base_unsized['max_r']:+.2f} R)")
    print(f"  • Monster Winners (>= 5.0 R)  : {base_unsized['monster_win_pct']:.1f}% ({df['is_monster_win'].sum():,} trades)")
    print(f"  • Average Duration            : {base_unsized['avg_bars']:.1f} days (Winners: {base_unsized['avg_bars_win']:.1f}d, Losers: {base_unsized['avg_bars_loss']:.1f}d)")
    
    print(f"\n[2] EXIT REASON BREAKDOWN:")
    for reason, grp in df.groupby("exit_reason"):
        st = compute_metrics(grp, reason)
        print(f"  • {reason:<22}: {st['count']:>5} trades ({st['count']/len(df)*100:>5.1f}%) | Win%: {st['win_rate']:>5.1f}% | Avg R: {st['avg_r']:>+6.3f} R | Exp: {st['expectancy']:>+6.3f} R | Avg Days: {st['avg_bars']:>4.1f}d")

    # 2. QQQ Trend Regime Analysis
    print("\n[3] QQQ TREND REGIME: QQQ > EMA10 > EMA20 ANALYSIS:")
    qqq_rows = []
    df_qqq = df[df["qqq_bull_stack"].notna()]
    if len(df_qqq) > 0:
        qqq_rows.append(compute_metrics(df_qqq[df_qqq["qqq_bull_stack"] == True], "QQQ Bull Stack (QQQ > EMA10 > EMA20)"))
        qqq_rows.append(compute_metrics(df_qqq[df_qqq["qqq_bull_stack"] == False], "QQQ Non-Bull Stack (Pullback/Correction)"))
        print_table(qqq_rows, "QQQ Trend Regime (QQQ > EMA10 > EMA20)", sort_key="expectancy")
        
    # 3. QQQ/SPY Relative Strength Regime Analysis
    print("\n[4] QQQ/SPY RELATIVE STRENGTH REGIME: (QQQ/SPY) > EMA10 > EMA20:")
    ratio_rows = []
    df_ratio = df[df["qqq_spy_ratio_bull"].notna()]
    if len(df_ratio) > 0:
        ratio_rows.append(compute_metrics(df_ratio[df_ratio["qqq_spy_ratio_bull"] == True], "QQQ/SPY Bull Stack (Tech/Growth Leading Market)"))
        ratio_rows.append(compute_metrics(df_ratio[df_ratio["qqq_spy_ratio_bull"] == False], "QQQ/SPY Non-Bull (Tech Lagging Market)"))
        # Combined QQQ + QQQ/SPY
        ratio_rows.append(compute_metrics(
            df_ratio[(df_ratio["qqq_bull_stack"] == True) & (df_ratio["qqq_spy_ratio_bull"] == True)],
            "★ DUAL BULL: QQQ Trend + QQQ/SPY Tech Leadership"
        ))
        ratio_rows.append(compute_metrics(
            df_ratio[(df_ratio["qqq_bull_stack"] == False) & (df_ratio["qqq_spy_ratio_bull"] == False)],
            "DUAL BEAR: QQQ Correcting & Tech Lagging"
        ))
        print_table(ratio_rows, "QQQ/SPY Relative Strength & Tech Leadership Regime", sort_key="expectancy")

    # 4. Institutional RS Score & Posture (RS_DATABASE.md)
    print("\n[5] INSTITUTIONAL RS SCORE & TECHNICAL POSTURE (RS_DATABASE.md):")
    rs_rows = []
    df_rs = df[df["rs_score"].notna()]
    if len(df_rs) > 0:
        rs_rows.append(compute_metrics(df_rs[df_rs["rs_score"] >= 95], "RS Score >= 95 (Pinnacle Elite)"))
        rs_rows.append(compute_metrics(df_rs[(df_rs["rs_score"] >= 90) & (df_rs["rs_score"] < 95)], "RS Score 90 - 94 (Decile 1)"))
        rs_rows.append(compute_metrics(df_rs[(df_rs["rs_score"] >= 80) & (df_rs["rs_score"] < 90)], "RS Score 80 - 89 (Decile 2)"))
        rs_rows.append(compute_metrics(df_rs[(df_rs["rs_score"] >= 70) & (df_rs["rs_score"] < 80)], "RS Score 70 - 79 (Decile 3)"))
        rs_rows.append(compute_metrics(df_rs[df_rs["rs_score"] < 70], "RS Score < 70 (Non-Leaders)"))
        rs_rows.append(compute_metrics(df_rs[df_rs["stage2"] == True], "Stage 2 Bullish Alignment (C>=10>=20>=50>=200)"))
        rs_rows.append(compute_metrics(df_rs[df_rs["stage2"] == False], "Non-Stage 2 Alignment"))
        rs_rows.append(compute_metrics(df_rs[df_rs["dist_52w_high"] >= -15.0], "Within 15% of 52W High (Near ATH)"))
        rs_rows.append(compute_metrics(df_rs[df_rs["dist_52w_high"] < -20.0], "Deep off 52W High (> 20% off ATH)"))
        rs_rows.append(compute_metrics(df_rs[df_rs["rs_leader"] == True], "★ Institutional Leader (RS>=90 + S2/ATH)"))
        rs_rows.append(compute_metrics(df_rs[df_rs["rs_leader"] == False], "Non-Leader Setup"))
        print_table(rs_rows, "Institutional RS Score & Posture Impact", sort_key="expectancy")

    # 5. Position Sizing Component Analysis
    print("\n[6] POSITION SIZING COMPONENT ATTRIBUTION:")
    size_rows = []
    # EP recency
    size_rows.append(compute_metrics(df[(df["has_prior_ep"]) & (df["prior_ep_days_ago"] <= 15)], "Fresh EP <= 15d (Sizing: 1.30x)"))
    size_rows.append(compute_metrics(df[(df["has_prior_ep"]) & (df["prior_ep_days_ago"] > 15) & (df["prior_ep_days_ago"] <= 30)], "Fresh EP 16-30d (Sizing: 1.15x)"))
    size_rows.append(compute_metrics(df[(df["has_prior_ep"]) & (df["prior_ep_days_ago"] > 30) & (df["prior_ep_days_ago"] <= 60)], "Seasoned EP 31-60d (Sizing: 1.05x)"))
    size_rows.append(compute_metrics(df[~df["has_prior_ep"]], "No Prior EP (Sizing: 0.85x)"))
    
    # Pole size
    size_rows.append(compute_metrics(df[df["move_pct"] >= 100.0], "Monster Pole >= 100% (Sizing: 1.25x)"))
    size_rows.append(compute_metrics(df[(df["move_pct"] >= 50.0) & (df["move_pct"] < 100.0)], "Strong Pole 50-100% (Sizing: 1.10x)"))
    size_rows.append(compute_metrics(df[df["move_pct"] < 50.0], "Standard Pole 30-50% (Sizing: 1.00x)"))
    
    # RVOL
    size_rows.append(compute_metrics(df[df["breakout_rvol"] >= 2.5], "Massive RVOL >= 2.5x (Sizing: 1.20x)"))
    size_rows.append(compute_metrics(df[(df["breakout_rvol"] >= 1.5) & (df["breakout_rvol"] < 2.5)], "Strong RVOL 1.5-2.5x (Sizing: 1.10x)"))
    size_rows.append(compute_metrics(df[df["breakout_rvol"] < 1.5], "Moderate RVOL < 1.5x (Sizing: 0.90x)"))
    
    # Pullback ratio
    size_rows.append(compute_metrics(df[df["pullback_ratio"] <= 0.25], "Tight Ratio <= 0.25 (Sizing: 1.20x)"))
    size_rows.append(compute_metrics(df[(df["pullback_ratio"] > 0.25) & (df["pullback_ratio"] <= 0.35)], "Medium Ratio 0.25-0.35 (Sizing: 1.05x)"))
    size_rows.append(compute_metrics(df[df["pullback_ratio"] > 0.35], "Loose Ratio > 0.35 (Sizing: 0.85x)"))
    print_table(size_rows, "Position Sizing Component Attribution", sort_key="expectancy")

    # 6. Sector and Theme Breakdown in Leading Universe
    sec_rows = [compute_metrics(grp, sec) for sec, grp in df.groupby("sector") if len(grp) >= 30]
    print_table(sec_rows, "Leading Sectors Performance Breakdown (Min 30 trades)", sort_key="expectancy")
    
    thm_rows = [compute_metrics(grp, thm) for thm, grp in df.groupby("theme") if len(grp) >= 20 and thm != "General"]
    top_thms = sorted(thm_rows, key=lambda x: x["expectancy"], reverse=True)[:15]
    print_table(top_thms, "Top 15 Leading Themes by Expectancy (Min 20 trades)", sort_key="expectancy")

    # 7. Comparison Table: Strategy Evolution
    print("==========================================================================================")
    print("  🏆 STRATEGY FORMULA COMPARISONS: FROM UNFILTERED TO HIGH-CONVICTION PINNACLE")
    print("==========================================================================================")
    evol_rows = []
    evol_rows.append(compute_metrics(df, "1. Leading Universe + 0.6x Stop (Unsized)", r_col="r_multiple"))
    evol_rows.append(compute_metrics(df, "2. Leading Universe + 0.6x Stop (Sized S*R)", r_col="sized_r_multiple"))
    
    # + QQQ Bull Stack
    f_qqq = df[df["qqq_bull_stack"] == True]
    evol_rows.append(compute_metrics(f_qqq, "3. Leading + 0.6x Stop + QQQ Bull Stack", r_col="sized_r_multiple"))
    
    # + QQQ/SPY Tech Leadership
    f_tech = df[(df["qqq_bull_stack"] == True) & (df["qqq_spy_ratio_bull"] == True)]
    evol_rows.append(compute_metrics(f_tech, "4. Leading + QQQ Bull + QQQ/SPY Tech Lead", r_col="sized_r_multiple"))
    
    # + Institutional Leader (RS >= 90 & Stage 2 / ATH)
    f_lead = df[
        (df["qqq_bull_stack"] == True) &
        (df["qqq_spy_ratio_bull"] == True) &
        (df["rs_leader"] == True)
    ]
    evol_rows.append(compute_metrics(f_lead, "5. Tech Lead + RS>=90 Leader Posture", r_col="sized_r_multiple"))
    
    # + Prior EP Catalyst
    f_pinnacle = df[
        (df["qqq_bull_stack"] == True) &
        (df["qqq_spy_ratio_bull"] == True) &
        (df["rs_leader"] == True) &
        (df["has_prior_ep"] == True)
    ]
    evol_rows.append(compute_metrics(f_pinnacle, "6. ★ PINNACLE HIGH-CONVICTION (All Filters + EP)", r_col="sized_r_multiple"))
    
    print_table(evol_rows, "Evolution to High-Conviction Pinnacle QM Strategy", sort_key="expectancy")


if __name__ == "__main__":
    main()
