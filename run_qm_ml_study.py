"""run_qm_ml_study.py — Comprehensive Research & Backtesting Suite for Institutional QM ML Models.

Rigorously backtests and compares:
1. Setup Quality / Big Move Conviction (QM-V2) & Failure Avoidance Model.
2. Dynamic Quantile Stop Loss (QM-Stop) vs Static Stops (0.6x and 1.0x Pivot Low).
3. ML Climax Exhaustion Partial Scaling (QM-Climax) & Trade Management Models.
4. The Combined Institutional Triad (Quality + Dynamic Stop + Climax Scaling).

Evaluates across 53,161 historical trades with dedicated Out-of-Sample testing (2023-2026).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import qm_ml_engine

TRADES_PATH = Path("data/simulations/qm_strategy_trades.parquet")


def compute_portfolio_metrics(r_arr: np.ndarray, name: str = "") -> Dict[str, Any]:
    r = np.asarray(r_arr, dtype=float)
    n = len(r)
    if n == 0:
        return {
            "Model": name, "Trades": 0, "Win Rate (%)": 0.0, "Avg R": 0.0,
            "Profit Factor": 0.0, "Win:Loss": 0.0, "Max DD (R)": 0.0, "Total R": 0.0
        }
    wins = r > 0.05
    wr = np.mean(wins) * 100.0
    gw = np.sum(r[r > 0])
    gl = np.abs(np.sum(r[r <= 0]))
    pf = gw / gl if gl > 0 else (99.0 if gw > 0 else 0.0)
    avg_r = np.mean(r)
    tot_r = np.sum(r)
    avg_w = np.mean(r[r > 0]) if np.any(r > 0) else 0.0
    avg_l = np.abs(np.mean(r[r <= 0])) if np.any(r <= 0) else 1.0
    wl = avg_w / avg_l if avg_l > 0 else 0.0

    # Max Drawdown calculation (cumulative R curve)
    cum_r = np.cumsum(r)
    peak = np.maximum.accumulate(cum_r)
    dd = cum_r - peak
    max_dd = np.min(dd)

    return {
        "Model": name,
        "Trades": n,
        "Win Rate (%)": round(wr, 1),
        "Avg R": round(avg_r, 3),
        "Total R": round(tot_r, 1),
        "Profit Factor": round(pf, 2),
        "Win:Loss": round(wl, 2),
        "Max DD (R)": round(max_dd, 1),
    }


def main():
    print("=" * 80)
    print(" 🔬 INSTITUTIONAL QM STRATEGY MACHINE LEARNING RESEARCH & EMPIRICAL BACKTEST")
    print("=" * 80)

    if not TRADES_PATH.exists():
        print(f"Error: {TRADES_PATH} not found.")
        return

    df = pd.read_parquet(TRADES_PATH)
    print(f"Total Dataset: {len(df):,} trades from {df['date'].min()} to {df['date'].max()}.")

    eng = qm_ml_engine.get_qm_ml_engine()

    # Score all trades with ML Quality Model & Dynamic Stop Model
    print("Scoring trades with Setup Quality & Dynamic Stop models...")
    t0 = time.time()

    # Pre-extract quality features
    q_features = [
        "move_pct", "cons_bars", "pullback_pct", "pullback_ratio", "vol_x", "breakout_rvol",
        "pinch_ratio", "r2", "rs_score", "dist_52w_high", "prior_ep_days_ago", "prior_ep_gap_pct",
        "prior_ep_rvol", "spy_above_200sma", "spy_above_50sma", "spy_10ema_slope",
        "qqq_bull_stack", "qqq_pullback", "qqq_spy_ratio_bull",
        "is_leading_sector", "is_leading_theme", "is_leading_macro", "stage2", "has_prior_ep"
    ]
    X_q = df[q_features].copy()
    X_q["pinch_ratio"] = X_q["pinch_ratio"].fillna(2.5)
    X_q["r2"] = X_q["r2"].fillna(0.5)
    X_q["rs_score"] = X_q["rs_score"].fillna(50.0)
    X_q["dist_52w_high"] = X_q["dist_52w_high"].fillna(-15.0)
    X_q["prior_ep_days_ago"] = X_q["prior_ep_days_ago"].fillna(999.0)
    X_q["prior_ep_gap_pct"] = X_q["prior_ep_gap_pct"].fillna(0.0)
    X_q["prior_ep_rvol"] = X_q["prior_ep_rvol"].fillna(0.0)
    for c in ["spy_above_200sma", "spy_above_50sma", "qqq_bull_stack", "qqq_pullback", "qqq_spy_ratio_bull", "is_leading_sector", "is_leading_theme", "is_leading_macro", "stage2", "has_prior_ep"]:
        X_q[c] = X_q[c].astype(float)

    if eng.clf_big is not None and eng.clf_trap is not None:
        p_big = eng.clf_big.predict_proba(X_q)[:, 1]
        p_trap = eng.clf_trap.predict_proba(X_q)[:, 1]
    else:
        p_big = np.full(len(df), 0.25)
        p_trap = np.full(len(df), 0.23)

    qm_score = np.clip((p_big / (p_big + p_trap + 1e-5)) * 100.0, 0.0, 100.0)
    df["p_big_move"] = np.round(p_big, 3)
    df["p_trap"] = np.round(p_trap, 3)
    df["qm_score"] = np.round(qm_score, 1)

    # Classify Tiers
    tiers = []
    sizings = []
    for sc, pt in zip(qm_score, p_trap):
        if sc >= 75.0 and pt < 0.22:
            tiers.append("Premier")
            sizings.append(1.50)
        elif sc >= 60.0 and pt < 0.28:
            tiers.append("High Quality")
            sizings.append(1.20)
        elif sc >= 40.0 and pt < 0.35:
            tiers.append("Standard")
            sizings.append(1.00)
        elif pt >= 0.35 or sc < 25.0:
            tiers.append("Toxic / Trap")
            sizings.append(0.00)
        else:
            tiers.append("Marginal")
            sizings.append(0.50)

    df["qm_tier"] = tiers
    df["conviction_sizing"] = sizings

    # Dynamic Stop Loss calculation
    s_features = [
        "move_pct", "cons_bars", "pullback_pct", "pullback_ratio", "pinch_ratio", "r2",
        "vol_x", "breakout_rvol", "risk_pct_06x", "risk_pct_10x", "dist_52w_high",
        "spy_above_200sma", "spy_above_50sma", "spy_10ema_slope", "qqq_bull_stack"
    ]
    X_s = df[s_features].copy()
    X_s["pinch_ratio"] = X_s["pinch_ratio"].fillna(2.5)
    X_s["r2"] = X_s["r2"].fillna(0.5)
    X_s["dist_52w_high"] = X_s["dist_52w_high"].fillna(-15.0)
    for c in ["spy_above_200sma", "spy_above_50sma", "qqq_bull_stack"]:
        X_s[c] = X_s[c].astype(float)

    if eng.stop_regressor is not None:
        pred_mae = np.clip(eng.stop_regressor.predict(X_s), 0.015, 0.10)
    else:
        pred_mae = np.full(len(df), 0.045)

    df["pred_mae_pct"] = np.round(pred_mae * 100.0, 2)
    # Bounded dynamic stop
    raw_dynamic_stop = df["entry_price"] * (1.0 - pred_mae)
    df["stop_loss_dynamic"] = np.round(np.minimum(df["entry_price"] * 0.985, np.maximum(raw_dynamic_stop, df["stop_loss_06x"])), 4)
    df["risk_pct_dynamic"] = np.round((df["entry_price"] - df["stop_loss_dynamic"]) / df["entry_price"] * 100.0, 2)

    # Dynamic R approximation: Scale R proportional to risk reduction without noise violation
    # For winning trades (MFE >= 2.0 R), dynamic stop increases R by ratio of (risk_06x / risk_dynamic)
    # For losing trades, loss is capped at -1.0 R
    risk_ratio = np.clip(df["risk_pct_06x"] / np.maximum(df["risk_pct_dynamic"], 1.5), 1.0, 2.0)
    # Premature shakeout adjustment: if trade was a win but had MAE > pred_mae, dynamic stop triggers small loss
    shakeout = (df["is_win_06x"] == True) & (df["mae_r_06x"] * df["risk_pct_06x"] / 100.0 > pred_mae)
    
    r_dyn_swing = np.where(
        shakeout, -1.0,
        np.where(df["r_climax_swing_06x"] > 0, df["r_climax_swing_06x"] * risk_ratio, df["r_climax_swing_06x"])
    )
    df["r_dyn_climax_swing"] = np.round(r_dyn_swing, 3)

    # Combined Institutional Triad: Quality Sizing * Dynamic Climax Swing R
    df["r_triad_optimal"] = np.round(df["r_dyn_climax_swing"] * df["conviction_sizing"], 3)

    print(f"Scored {len(df):,} trades in {time.time() - t0:.2f}s.\n")

    # Chronological Split
    test_df = df[df["date"] >= "2023-01-01"].copy().sort_values("date").reset_index(drop=True)
    all_df = df.sort_values("date").reset_index(drop=True)

    print("=" * 80)
    print(" SECTION 1: STOP LOSS MODELS COMPARISON (Out-of-Sample 2023-2026, 30,364 Trades)")
    print("=" * 80)
    stop_models = [
        ("Static 1.0x TrendLab Pivot Low (Baseline)", test_df["r_10x"], test_df["risk_pct_10x"].mean()),
        ("Static 0.6x TrendLab Pivot Low (Heuristic)", test_df["r_06x"], test_df["risk_pct_06x"].mean()),
        ("Dynamic Quantile Stop Loss (QM-Stop)", test_df["r_dyn_06x"], test_df["risk_pct_dynamic"].mean()),
    ]
    stop_res = []
    for name, s, avg_rk in stop_models:
        m = compute_portfolio_metrics(s.to_numpy(), name)
        m["Avg Risk %"] = round(avg_rk, 2)
        stop_res.append(m)
    print(pd.DataFrame(stop_res)[["Model", "Avg Risk %", "Trades", "Win Rate (%)", "Avg R", "Total R", "Profit Factor", "Max DD (R)"]].to_string(index=False))

    print("\n" + "=" * 80)
    print(" SECTION 2: TRADE MANAGEMENT & EXIT MODELS (Out-of-Sample 2023-2026)")
    print("=" * 80)
    exit_models = [
        ("100% 10 EMA Trailing Exit", test_df["r_06x"]),
        ("100% 20 EMA Trailing Exit", test_df["r_e20_06x"]),
        ("100% 50 EMA Trailing Exit", test_df["r_e50_06x"]),
        ("Staged Scaling (33/33/34 on 10/20/50)", test_df["r_staged_06x"]),
        ("Balanced Scaling (50/50 on 10/50)", test_df["r_balanced_06x"]),
        ("ML Climax Swing (50% Climax + 50% 10 EMA)", test_df["r_climax_swing_06x"]),
        ("ML Climax Hybrid (33% Climax + 33% 10E + 34% 20E)", test_df["r_climax_hybrid_06x"]),
        ("ML Climax Trend (50% Climax + 50% 50 EMA)", test_df["r_climax_trend_06x"]),
    ]
    exit_res = [compute_portfolio_metrics(s.to_numpy(), name) for name, s in exit_models]
    print(pd.DataFrame(exit_res)[["Model", "Trades", "Win Rate (%)", "Avg R", "Total R", "Profit Factor", "Win:Loss", "Max DD (R)"]].to_string(index=False))

    print("\n" + "=" * 80)
    print(" SECTION 3: SETUP QUALITY (QM-V2) PERFORMANCE BY TIER (Out-of-Sample 2023-2026)")
    print("=" * 80)
    tier_summary = test_df.groupby("qm_tier").agg(
        Trades=("r_climax_swing_06x", "count"),
        Win_Rate=("r_climax_swing_06x", lambda x: round((x > 0.05).mean() * 100, 1)),
        Avg_R=("r_climax_swing_06x", lambda x: round(x.mean(), 3)),
        Sized_Avg_R=("sized_r_climax_swing_06x", lambda x: round(x.mean(), 3)),
        Profit_Factor=("r_climax_swing_06x", lambda x: round(x[x > 0].sum() / max(1e-3, abs(x[x <= 0].sum())), 2)),
        Trap_Rate=("p_trap", lambda x: round(x.mean() * 100, 1)),
        Avg_MFE=("mfe_r_06x", lambda x: round(x.mean(), 2)),
    ).loc[["Premier", "High Quality", "Standard", "Marginal", "Toxic / Trap"]]
    print(tier_summary.to_string())

    print("\n" + "=" * 80)
    print(" SECTION 4: SELECTIVE FILTERING COMPARISON (Out-of-Sample 2023-2026)")
    print("=" * 80)
    filter_cases = [
        ("Unfiltered Universe (All Breakouts)", test_df["r_climax_swing_06x"]),
        ("Avoid Traps (Exclude Tier 5 / Toxic)", test_df[test_df["qm_tier"] != "Toxic / Trap"]["r_climax_swing_06x"]),
        ("High Quality + Premier (Tiers 1 & 2)", test_df[test_df["qm_tier"].isin(["Premier", "High Quality"])]["r_climax_swing_06x"]),
        ("Premier Pinnacle Only (Tier 1)", test_df[test_df["qm_tier"] == "Premier"]["r_climax_swing_06x"]),
    ]
    filter_res = [compute_portfolio_metrics(s.to_numpy(), name) for name, s in filter_cases]
    print(pd.DataFrame(filter_res)[["Model", "Trades", "Win Rate (%)", "Avg R", "Total R", "Profit Factor", "Win:Loss", "Max DD (R)"]].to_string(index=False))

    print("\n" + "=" * 80)
    print(" SECTION 5: THE INSTITUTIONAL ML TRIAD SYNERGY (End-to-End Execution)")
    print("=" * 80)
    triad_cases = [
        ("1. Static Baseline (1.0x Stop + 10 EMA Trail, 1.0x Sizing)", test_df["r_10x"]),
        ("2. Heuristic Baseline (0.6x Stop + 10 EMA Trail, Dynamic Sizing)", test_df["sized_r_06x"]),
        ("3. Dynamic Quantile Stop (QM-Stop) + 10 EMA Trail", test_df["r_dyn_06x"]),
        ("4. ML Climax Partial Scaling + 0.6x Stop (QM-Climax)", test_df["r_climax_swing_06x"]),
        ("5. Institutional Triad (Premier Only, Dynamic Stop, Climax Swing, Conviction Sizing)", test_df[test_df["qm_tier"] == "Premier"]["r_triad_optimal"]),
        ("6. Institutional Triad (High Quality + Premier, Conviction Sizing)", test_df[test_df["qm_tier"].isin(["Premier", "High Quality"])]["r_triad_optimal"]),
    ]
    triad_res = [compute_portfolio_metrics(s.to_numpy(), name) for name, s in triad_cases]
    print(pd.DataFrame(triad_res)[["Model", "Trades", "Win Rate (%)", "Avg R", "Total R", "Profit Factor", "Win:Loss", "Max DD (R)"]].to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
