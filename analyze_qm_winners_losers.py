"""analyze_qm_winners_losers.py — Deep Quantitative Pattern Analysis of QM Breakout Trades.

Analyzes 10,976 simulated QM breakout trades from `data/simulations/qm_breakout_trades_full.parquet`:
1. Full trading performance statistics (Win rate, Expectancy, Profit Factor, Drawdown, Exit reasons).
2. Sector & Theme performance breakdowns.
3. Market Health correlation (SPY trend alignment, moving averages, 21-day returns).
4. Institutional Relative Strength & Technical Posture impact (RS deciles, Stage 2, 52W high distance).
5. Setup Geometry "To the Left" (Pole move %, consolidation days, pullback %, ride fraction, patterns, volume contraction, breakout candle qualities).
6. Forward Trade Dynamics "To the Right" (MFE capture efficiency, MAE drawdown tolerance of winners, post EP catalyst boost).
7. Top Alpha Features: Identifies specific filter combinations that maximize Win Rate and Expectancy while eliminating the largest fraction of losers.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

DATA_PATH = Path("data/simulations/qm_breakout_trades_full.parquet")


def compute_stats(df: pd.DataFrame, label: str = "") -> dict:
    n = len(df)
    if n == 0:
        return {"name": label, "count": 0}
    r = df["r_multiple"].to_numpy(float)
    wins = df[df["r_multiple"] > 0]
    losses = df[df["r_multiple"] <= 0]
    scratches = df[df["r_multiple"].abs() <= 0.3]
    big_wins = df[df["r_multiple"] >= 3.0]
    monster_wins = df[df["r_multiple"] >= 5.0]
    full_losses = df[df["r_multiple"] <= -0.9]
    
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = (n_wins / n) * 100.0
    
    sum_win_r = wins["r_multiple"].sum()
    sum_loss_r = abs(losses["r_multiple"].sum())
    profit_factor = (sum_win_r / sum_loss_r) if sum_loss_r > 0 else np.nan
    
    avg_r = float(np.mean(r))
    med_r = float(np.median(r))
    std_r = float(np.std(r)) if n > 1 else 0.0
    
    avg_win = float(wins["r_multiple"].mean()) if n_wins > 0 else 0.0
    avg_loss = float(losses["r_multiple"].mean()) if n_losses > 0 else 0.0
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


def print_table(rows: list[dict], title: str, sort_key: str = "avg_r", ascending: bool = False):
    print("=" * 110)
    print(f"  {title.upper()}")
    print("=" * 110)
    if not rows:
        print("No data.")
        return
    
    # Sort
    valid_rows = [r for r in rows if r["count"] > 0]
    if sort_key:
        valid_rows = sorted(valid_rows, key=lambda x: x.get(sort_key, 0), reverse=not ascending)
        
    hdr = f"{'Cohort':<32} | {'Count':>6} | {'Win%':>6} | {'Avg R':>7} | {'Exp (R)':>7} | {'PF':>5} | {'Tot R':>8} | {'Win/Loss':>8} | {'BigWin%':>7} | {'Days':>5}"
    print(hdr)
    print("-" * 110)
    for r in valid_rows:
        line = (
            f"{r['name'][:32]:<32} | {r['count']:>6} | {r['win_rate']:>5.1f}% | {r['avg_r']:>+7.3f} | "
            f"{r['expectancy']:>+7.3f} | {r['profit_factor']:>5.2f} | {r['total_r']:>+8.1f} | "
            f"{r['win_loss_ratio']:>8.2f} | {r['big_win_pct']:>6.1f}% | {r['avg_bars']:>5.1f}"
        )
        print(line)
    print("-" * 110)
    print()


def main():
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found. Please run sim_qm_breakouts_study.py first.")
        sys.exit(1)
        
    df = pd.read_parquet(DATA_PATH)
    print("==========================================================================================")
    print(f"📊 QM BREAKOUT STRATEGY BACKTEST RESULTS ({len(df):,} TRADES ACROSS US EQUITIES)")
    print("   Trade Rule: Buy Close of QM Breakout | Initial Stop: Below TrendLab Pivot Low")
    print("   Exit Rule: On First Close Below 10 EMA (Once 10 EMA is Rising & Price Closed Well Above)")
    print("==========================================================================================")
    
    # 1. Overall Baseline Stats
    baseline = compute_stats(df, "Baseline (All Trades)")
    print(f"\n[1] OVERALL PERFORMANCE SUMMARY:")
    print(f"  • Total Trades Executed   : {baseline['count']:,}")
    print(f"  • Win Rate                : {baseline['win_rate']:.1f}% ({df['is_win'].sum():,} wins / {(~df['is_win']).sum():,} losses)")
    print(f"  • Profit Factor           : {baseline['profit_factor']:.2f}")
    print(f"  • Expectancy per Trade    : {baseline['expectancy']:+.3f} R")
    print(f"  • Total Strategy Return   : {baseline['total_r']:+,.1f} R")
    print(f"  • Average Trade R         : {baseline['avg_r']:+.3f} R (Median: {baseline['med_r']:+.3f} R, Std: {baseline['std_r']:.3f})")
    print(f"  • Average Winner          : {baseline['avg_win']:+.2f} R (Avg Gain: +{df[df['r_multiple'] > 0]['gain_pct'].mean():.1f}%)")
    print(f"  • Average Loser           : {baseline['avg_loss']:+.2f} R (Avg Loss: {df[df['r_multiple'] <= 0]['gain_pct'].mean():.1f}%)")
    print(f"  • Win / Loss Ratio        : {baseline['win_loss_ratio']:.2f} : 1")
    print(f"  • Big Winners (>= 3.0 R)  : {baseline['big_win_pct']:.1f}% ({df['is_big_win'].sum():,} trades)")
    print(f"  • Monster Winners (>=5.0R): {baseline['monster_win_pct']:.1f}% ({sum(df['r_multiple'] >= 5.0):,} trades)")
    print(f"  • Full Losses (<= -0.9 R) : {baseline['full_loss_pct']:.1f}% ({df['is_full_loss'].sum():,} trades)")
    print(f"  • Scratches (+/- 0.3 R)   : {baseline['scratch_pct']:.1f}% ({df['is_scratch'].sum():,} trades)")
    print(f"  • Max Winner / Max Loser  : {baseline['max_r']:+.2f} R / {baseline['min_r']:+.2f} R")
    print(f"  • Average Duration        : {baseline['avg_bars']:.1f} days (Winners: {baseline['avg_bars_win']:.1f}d, Losers: {baseline['avg_bars_loss']:.1f}d)")
    
    # Exit Reason Breakdown
    print(f"\n[2] EXIT REASON BREAKDOWN:")
    for reason, grp in df.groupby("exit_reason"):
        st = compute_stats(grp, reason)
        print(f"  • {reason:<22}: {st['count']:>5} trades ({st['count']/len(df)*100:>5.1f}%) | Win%: {st['win_rate']:>5.1f}% | Avg R: {st['avg_r']:>+6.3f} R | Exp: {st['expectancy']:>+6.3f} R | Avg Days: {st['avg_bars']:>4.1f}d")
        
    # Trailing Stop Activation Analysis
    act_trades = df[df["trail_activated"] == True]
    non_act_trades = df[df["trail_activated"] == False]
    print(f"\n[3] 10 EMA TRAILING STOP ACTIVATION IMPACT:")
    st_act = compute_stats(act_trades, "10 EMA Trail Activated")
    st_non = compute_stats(non_act_trades, "Never Activated (Stopped Out First)")
    print(f"  • Trail Activated         : {st_act['count']:>5} trades ({st_act['count']/len(df)*100:.1f}%) | Win%: {st_act['win_rate']:.1f}% | Avg R: {st_act['avg_r']:+.3f} R | Exp: {st_act['expectancy']:+.3f} R | PF: {st_act['profit_factor']:.2f}")
    print(f"  • Never Activated         : {st_non['count']:>5} trades ({st_non['count']/len(df)*100:.1f}%) | Win%: {st_non['win_rate']:.1f}% | Avg R: {st_non['avg_r']:+.3f} R | Exp: {st_non['expectancy']:+.3f} R | PF: {st_non['profit_factor']:.2f}")

    # 4. Sector Performance
    sec_rows = [compute_stats(grp, sec) for sec, grp in df.groupby("sector") if len(grp) >= 50]
    print_table(sec_rows, "Performance by Sector (Min 50 trades)", sort_key="expectancy")
    
    # 5. Top Themes
    thm_rows = [compute_stats(grp, thm) for thm, grp in df.groupby("theme") if len(grp) >= 30 and thm != "General"]
    top_thms = sorted(thm_rows, key=lambda x: x["expectancy"], reverse=True)[:15]
    print_table(top_thms, "Top 15 Themes by Expectancy (Min 30 trades)", sort_key="expectancy")
    worst_thms = sorted(thm_rows, key=lambda x: x["expectancy"])[:10]
    print_table(worst_thms, "Bottom 10 Themes by Expectancy (Min 30 trades)", sort_key="expectancy", ascending=True)

    # 6. Market Health Context (SPY)
    mkt_rows = []
    # Filter to trades where SPY context exists
    df_mkt = df[df["spy_above_10ema"].notna()].copy()
    if len(df_mkt) > 0:
        mkt_rows.append(compute_stats(df_mkt[df_mkt["spy_trend_aligned"] == True], "SPY Trend Aligned (C >= 10 >= 20 >= 50 >= 200)"))
        mkt_rows.append(compute_stats(df_mkt[df_mkt["spy_trend_aligned"] == False], "SPY Non-Aligned"))
        mkt_rows.append(compute_stats(df_mkt[df_mkt["spy_above_50sma"] == True], "SPY Above 50 SMA"))
        mkt_rows.append(compute_stats(df_mkt[df_mkt["spy_above_50sma"] == False], "SPY Below 50 SMA (Market Correction)"))
        mkt_rows.append(compute_stats(df_mkt[df_mkt["spy_above_200sma"] == True], "SPY Above 200 SMA (Bull Market)"))
        mkt_rows.append(compute_stats(df_mkt[df_mkt["spy_above_200sma"] == False], "SPY Below 200 SMA (Bear Market)"))
        mkt_rows.append(compute_stats(df_mkt[df_mkt["spy_return_21d"] > 0], "SPY 1-Month Return > 0%"))
        mkt_rows.append(compute_stats(df_mkt[df_mkt["spy_return_21d"] <= 0], "SPY 1-Month Return <= 0%"))
        mkt_rows.append(compute_stats(df_mkt[df_mkt["spy_dist_52w_high"] >= -5.0], "SPY Near ATH (<= 5% off High)"))
        mkt_rows.append(compute_stats(df_mkt[df_mkt["spy_dist_52w_high"] < -15.0], "SPY Deep Pullback (> 15% off High)"))
        print_table(mkt_rows, "Market Health Context (SPY Posture on Breakout Day)", sort_key="expectancy")

    # 7. Institutional Relative Strength & Technical Posture
    rs_rows = []
    df_rs = df[df["rs_score"].notna()].copy()
    if len(df_rs) > 0:
        rs_rows.append(compute_stats(df_rs[df_rs["rs_score"] >= 95], "RS Score >= 95 (Pinnacle Elite)"))
        rs_rows.append(compute_stats(df_rs[(df_rs["rs_score"] >= 90) & (df_rs["rs_score"] < 95)], "RS Score 90 - 94 (Decile 1)"))
        rs_rows.append(compute_stats(df_rs[(df_rs["rs_score"] >= 80) & (df_rs["rs_score"] < 90)], "RS Score 80 - 89 (Decile 2)"))
        rs_rows.append(compute_stats(df_rs[(df_rs["rs_score"] >= 70) & (df_rs["rs_score"] < 80)], "RS Score 70 - 79 (Decile 3)"))
        rs_rows.append(compute_stats(df_rs[(df_rs["rs_score"] >= 50) & (df_rs["rs_score"] < 70)], "RS Score 50 - 69 (Midpack)"))
        rs_rows.append(compute_stats(df_rs[df_rs["rs_score"] < 50], "RS Score < 50 (Laggards)"))
        rs_rows.append(compute_stats(df_rs[df_rs["stage2"] == True], "Stock Stage 2 Bullish Alignment"))
        rs_rows.append(compute_stats(df_rs[df_rs["stage2"] == False], "Stock Non-Stage 2"))
        rs_rows.append(compute_stats(df_rs[df_rs["dist_52w_high"] >= -15.0], "Within 15% of 52W High (Near ATH)"))
        rs_rows.append(compute_stats(df_rs[df_rs["dist_52w_high"] < -25.0], "Deep off 52W High (> 25% off High)"))
        rs_rows.append(compute_stats(df_rs[df_rs["rs_leader"] == True], "★ Institutional Leader (RS>=90 & S2/ATH)"))
        rs_rows.append(compute_stats(df_rs[df_rs["rs_leader"] == False], "Non-Leader Setup"))
        print_table(rs_rows, "Institutional RS Score & Posture Performance", sort_key="expectancy")

    # 8. Setup "To the Left" — Geometry & Catalysts
    left_rows = []
    # Prior EP
    left_rows.append(compute_stats(df[df["has_prior_ep"] == True], "Has Prior EP Catalyst (<= 60d)"))
    left_rows.append(compute_stats(df[df["has_prior_ep"] == False], "No Prior EP (Pure Technical)"))
    left_rows.append(compute_stats(df[(df["has_prior_ep"] == True) & (df["prior_ep_days_ago"] <= 20)], "Prior EP Fresh (<= 20d ago)"))
    left_rows.append(compute_stats(df[(df["has_prior_ep"] == True) & (df["prior_ep_days_ago"] > 20)], "Prior EP Seasoned (21 - 60d ago)"))
    left_rows.append(compute_stats(df[(df["has_prior_ep"] == True) & (df["prior_ep_gap_pct"] >= 15.0)], "Prior EP Gap >= +15%"))
    left_rows.append(compute_stats(df[(df["has_prior_ep"] == True) & (df["prior_ep_rvol"] >= 3.0)], "Prior EP RVOL >= 3.0x"))
    
    # Pole Move %
    left_rows.append(compute_stats(df[df["move_pct"] >= 100.0], "Monster Pole: Move >= +100%"))
    left_rows.append(compute_stats(df[(df["move_pct"] >= 50.0) & (df["move_pct"] < 100.0)], "Strong Pole: Move +50% to +100%"))
    left_rows.append(compute_stats(df[(df["move_pct"] >= 30.0) & (df["move_pct"] < 50.0)], "Standard Pole: Move +30% to +50%"))
    
    # Consolidation Length
    left_rows.append(compute_stats(df[df["cons_bars"] <= 12], "Tight Base (8 - 12 days)"))
    left_rows.append(compute_stats(df[(df["cons_bars"] > 12) & (df["cons_bars"] <= 25)], "Medium Base (13 - 25 days)"))
    left_rows.append(compute_stats(df[df["cons_bars"] > 25], "Extended Base (> 25 days)"))
    
    # Pullback Depth
    left_rows.append(compute_stats(df[df["pullback_pct"] <= 12.0], "Shallow Digestion (Pullback <= 12%)"))
    left_rows.append(compute_stats(df[(df["pullback_pct"] > 12.0) & (df["pullback_pct"] <= 22.0)], "Moderate Pullback (12% - 22%)"))
    left_rows.append(compute_stats(df[df["pullback_pct"] > 22.0], "Deep Pullback (> 22%)"))
    
    # Pullback to Move ratio
    left_rows.append(compute_stats(df[df["pullback_ratio"] <= 0.35], "Tight Absorption (Pullback / Move <= 0.35)"))
    left_rows.append(compute_stats(df[df["pullback_ratio"] > 0.50], "Loose Digestion (Pullback / Move > 0.50)"))
    
    # Ride Fraction
    left_rows.append(compute_stats(df[df["ride_frac"] >= 0.85], "Dominant EMA Ride (Ride Frac >= 85%)"))
    left_rows.append(compute_stats(df[df["ride_frac"] < 0.65], "Loose EMA Surfing (Ride Frac < 65%)"))
    
    # Base Volume
    left_rows.append(compute_stats(df[df["vol_x"] <= 1.0], "Volume Dry-up in Base (Vol X <= 1.0)"))
    left_rows.append(compute_stats(df[df["vol_x"] > 1.5], "Heavy Churn in Base (Vol X > 1.5)"))
    
    # Breakout Candle Quality
    left_rows.append(compute_stats(df[df["breakout_close_pos"] >= 0.80], "Power Close: Close in top 20% of range"))
    left_rows.append(compute_stats(df[df["breakout_close_pos"] < 0.60], "Weak Close: Close in lower 60% of range"))
    left_rows.append(compute_stats(df[df["breakout_rvol"] >= 2.0], "Explosive Breakout RVOL >= 2.0x"))
    left_rows.append(compute_stats(df[df["breakout_gap_pct"] >= 2.0], "Breakout Day Gap >= +2.0%"))
    
    # Initial Risk %
    left_rows.append(compute_stats(df[df["risk_pct"] <= 6.0], "Tight Risk: Pivot Low <= 6%"))
    left_rows.append(compute_stats(df[(df["risk_pct"] > 6.0) & (df["risk_pct"] <= 10.0)], "Standard Risk: Pivot Low 6% - 10%"))
    left_rows.append(compute_stats(df[df["risk_pct"] > 12.0], "Wide Risk: Pivot Low > 12%"))
    
    # Pattern Shape
    for pat, grp in df.groupby("pattern"):
        if len(grp) >= 100:
            left_rows.append(compute_stats(grp, f"Pattern: {pat}"))
            
    print_table(left_rows, "Setup 'To The Left' — Geometry, Catalysts & Candle Quality", sort_key="expectancy")

    # 9. Chart "To the Right" — Forward Excursion & Acceleration
    right_rows = []
    # Post EP
    right_rows.append(compute_stats(df[df["has_post_ep"] == True], "Received Post EP Boost (<= 60d)"))
    right_rows.append(compute_stats(df[df["has_post_ep"] == False], "No Post EP"))
    
    # Peak MFE reached
    mfe_vals = df["max_mfe_r"].to_numpy(float)
    mae_vals = df["max_mae_r"].to_numpy(float)
    winners = df[df["r_multiple"] > 0]
    losers = df[df["r_multiple"] <= 0]
    
    print("==========================================================================================")
    print("  📈 FORWARD EXCURSION & TRADE DYNAMICS ('TO THE RIGHT')")
    print("==========================================================================================")
    print(f"  • Average Peak MFE Reached : {df['max_mfe_r'].mean():.2f} R (Avg Peak Gain: +{df['max_mfe_pct'].mean():.1f}%)")
    print(f"  • Median Peak MFE Reached  : {df['max_mfe_r'].median():.2f} R (Median Peak Gain: +{df['max_mfe_pct'].median():.1f}%)")
    print(f"  • Trades reaching >= +2.0R : {(df['max_mfe_r'] >= 2.0).mean()*100:.1f}% ({sum(df['max_mfe_r'] >= 2.0):,} trades)")
    print(f"  • Trades reaching >= +3.0R : {(df['max_mfe_r'] >= 3.0).mean()*100:.1f}% ({sum(df['max_mfe_r'] >= 3.0):,} trades)")
    print(f"  • Trades reaching >= +5.0R : {(df['max_mfe_r'] >= 5.0).mean()*100:.1f}% ({sum(df['max_mfe_r'] >= 5.0):,} trades)")
    print(f"  • Average MFE of Winners   : {winners['max_mfe_r'].mean():.2f} R (vs Realized: {winners['r_multiple'].mean():.2f} R)")
    print(f"  • 10 EMA Capture Efficiency: {winners['capture_ratio'].mean()*100:.1f}% of peak MFE locked in by 10 EMA exit")
    print(f"  • Average MAE of Winners   : {winners['max_mae_r'].mean():.2f} R (Deepest drawdown winners suffered before running)")
    print(f"    - Winners with MAE <= 0.3R: {(winners['max_mae_r'] <= 0.3).mean()*100:.1f}% (immediate liftoff with zero heat!)")
    print(f"    - Winners with MAE <= 0.6R: {(winners['max_mae_r'] <= 0.6).mean()*100:.1f}% (almost never approach initial stop)")
    print("------------------------------------------------------------------------------------------\n")

    # 10. Information Gain & Feature Importance (Separating Winners vs Losers)
    print("==========================================================================================")
    print("  🎯 STATISTICAL SEPARATION: FEATURES THAT DIFFERENTIATE WINNERS FROM LOSERS")
    print("==========================================================================================")
    
    numeric_features = [
        ("rs_score", "Institutional Relative Strength (1-99)"),
        ("dist_52w_high", "Distance to 52-Week High (%)"),
        ("risk_pct", "Initial Stop Loss Distance (%)"),
        ("breakout_close_pos", "Breakout Candle Close Position (0-1)"),
        ("breakout_rvol", "Breakout Day Relative Volume (x)"),
        ("breakout_range_atr", "Breakout Candle Range / ATR"),
        ("pullback_ratio", "Base Pullback / Move Ratio"),
        ("ride_frac", "Consolidation 20 EMA Ride Fraction"),
        ("vol_x", "Consolidation Volume Ratio"),
        ("cons_bars", "Consolidation Duration (Days)"),
        ("move_pct", "Pole Move Gain (%)"),
        ("dist_10ema", "Distance to 10 EMA on Breakout (%)"),
        ("spy_return_21d", "SPY 21-Day Trend Momentum (%)"),
        ("spy_dist_52w_high", "SPY Distance to 52W High (%)"),
    ]
    
    feature_ranking = []
    for col, desc in numeric_features:
        valid = df.dropna(subset=[col, "r_multiple"])
        if len(valid) < 200: continue
        w_vals = valid[valid["r_multiple"] > 0][col].to_numpy(float)
        l_vals = valid[valid["r_multiple"] <= 0][col].to_numpy(float)
        
        t_stat, p_val = stats.ttest_ind(w_vals, l_vals, equal_var=False)
        corr, _ = stats.pointbiserialr(valid["is_win"], valid[col])
        feature_ranking.append({
            "col": col,
            "desc": desc,
            "win_mean": float(np.mean(w_vals)),
            "loss_mean": float(np.mean(l_vals)),
            "diff": float(np.mean(w_vals) - np.mean(l_vals)),
            "t_stat": t_stat,
            "p_val": p_val,
            "corr": corr,
        })
        
    feature_ranking = sorted(feature_ranking, key=lambda x: abs(x["t_stat"]), reverse=True)
    
    print(f"{'Feature Description':<40} | {'Winner Mean':>12} | {'Loser Mean':>12} | {'T-Statistic':>12} | {'P-Value':>10}")
    print("-" * 95)
    for f in feature_ranking:
        sig = "***" if f["p_val"] < 0.001 else ("**" if f["p_val"] < 0.01 else ("*" if f["p_val"] < 0.05 else ""))
        print(f"{f['desc']:<40} | {f['win_mean']:>12.2f} | {f['loss_mean']:>12.2f} | {f['t_stat']:>+12.2f} | {f['p_val']:>9.2e} {sig}")
    print("-" * 95)
    print("Significance: *** p < 0.001 | ** p < 0.01 | * p < 0.05\n")

    # 11. Multi-Condition Alpha Filter Models: Comparing Strategy Presets
    print("==========================================================================================")
    print("  🏆 STRATEGY FORMULA COMPARISONS: FROM UNFILTERED BASELINE TO PINNACLE ELITE")
    print("==========================================================================================")
    
    combos = []
    combos.append(compute_stats(df, "1. Unfiltered Baseline (All Trades)"))
    
    # Filter 2: Market Health Only
    f_mkt = df[(df["spy_above_50sma"] == True) & (df["spy_10ema_slope"] > 0)]
    combos.append(compute_stats(f_mkt, "2. Healthy Market Only (SPY > 50SMA & 10EMA Rising)"))
    
    # Filter 3: Institutional Leaders (RS >= 90 & Stage2 or ATH<=15%)
    f_rs = df[df["rs_leader"] == True]
    combos.append(compute_stats(f_rs, "3. Institutional Leaders (RS>=90 + S2/ATH)"))
    
    # Filter 4: Prior EP Catalyst Only
    f_ep = df[df["has_prior_ep"] == True]
    combos.append(compute_stats(f_ep, "4. Catalyzed QM (Has Prior EP <= 60d)"))
    
    # Filter 5: Power Breakout Candle Quality
    f_candle = df[(df["breakout_close_pos"] >= 0.70) & (df["breakout_rvol"] >= 1.5)]
    combos.append(compute_stats(f_candle, "5. Power Candle (Close Pos >= 70% & RVOL >= 1.5x)"))
    
    # Filter 6: Tight Base & Absorption
    f_base = df[(df["pullback_pct"] <= 20.0) & (df["ride_frac"] >= 0.75) & (df["risk_pct"] <= 10.0)]
    combos.append(compute_stats(f_base, "6. Tight Absorption Base (PB<=20%, Ride>=75%, Risk<=10%)"))
    
    # Filter 7: The Sweet Spot Formula (RS Leader + Healthy Market + Tight Base)
    f_combo = df[
        (df["rs_leader"] == True) &
        (df["spy_above_50sma"] == True) &
        (df["pullback_pct"] <= 22.0) &
        (df["risk_pct"] <= 11.0)
    ]
    combos.append(compute_stats(f_combo, "7. Elite Institutional Filter (Leader + Mkt + Tight Base)"))
    
    # Filter 8: Pinnacle Elite (Elite Filter + Prior EP Catalyst + Power Candle)
    f_pinnacle = df[
        (df["rs_leader"] == True) &
        (df["spy_above_50sma"] == True) &
        (df["pullback_pct"] <= 20.0) &
        (df["risk_pct"] <= 10.0) &
        (df["has_prior_ep"] == True) &
        (df["breakout_close_pos"] >= 0.65)
    ]
    combos.append(compute_stats(f_pinnacle, "8. ★ PINNACLE ELITE TRIPLE-CROWN (All Filters)"))
    
    print_table(combos, "Strategy Evolution: Baseline vs Engineered Filter Presets", sort_key="expectancy")


if __name__ == "__main__":
    main()
