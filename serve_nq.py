#!/usr/bin/env python3
"""
serve_nq.py — Institutional NQ Intraday (15m, 1h, 4h) Pivot High/Low & Forward MAE/MFE Explorer (Port 8790).

Presents the empirical 10-year study (8-Year Train: 2016-2024 vs 2-Year Out-of-Sample Test: 2024-2026)
of NQ-specific Machine Learning models across 15m, 1h, and 4h timeframes:
  - Dual-Path Pivot High & Pivot Low Classifiers (Sub-Model A Real-Time At-Extreme + Sub-Model B Full-Bar & 1m Microstructure)
  - Forward K-Bar (K = 4, 8, 12 bars) MFE & MAE Regressors, Quantile Target/Stop Buffers, and Asymmetry Classifiers
  - Head-to-head comparison against Generic Daily Equity Pivot Models applied zero-shot to NQ
  - Interactive TradingView LightweightCharts viewer loading ALL history with Session VWAP, Overnight High/Low sweeps,
    Pivot High/Low markers, explicit Non-Actionable gate explanations (Contract Roll Seam guard, Prob threshold,
    MFE/MAE Ratio threshold), click-any-bar inspector, and visual Forward K-Bar Predicted MFE Target & MAE Stop Cushion projections.

Usage:
  .venv/bin/python serve_nq.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nq_ml_engine as nqe

PORT = 8790
app = Flask(__name__)

_DF_CACHE: Dict[str, pd.DataFrame] = {}
_DF_MTIME: Dict[str, float] = {}
_CHART_STATIC_CACHE: Dict[str, Dict[str, Any]] = {}
_CHART_STATIC_MTIME: Dict[str, float] = {}
_SUMMARY_CACHE: Optional[Dict[str, Any]] = None

SESSION_NAMES = {
    0: "Asia Globex (18:00-03:00)",
    1: "London Open (03:00-08:00)",
    2: "NY Pre-Mkt Macro (08:00-09:30)",
    3: "NY RTH Open Drive (09:30-10:30)",
    4: "NY Midday (10:30-14:00)",
    5: "NY Power Hour (14:00-16:00)",
}


def _get_scored_df(tf: str) -> pd.DataFrame:
    tf_clean = tf if tf in ("15m", "1h", "4h") else "1h"
    try:
        nqe.ensure_nq_intraday_uptodate(max_age_sec=35.0)
    except Exception:
        pass
    p = nqe.SCORED_CACHE_DIR / f"nq_{tf_clean}_scored.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Scored NQ parquet not found at {p}. Run `.venv/bin/python nq_ml_engine.py` first.")
    mtime = p.stat().st_mtime
    if tf_clean not in _DF_CACHE or _DF_MTIME.get(tf_clean) != mtime:
        df = pd.read_parquet(p)
        _DF_CACHE[tf_clean] = df
        _DF_MTIME[tf_clean] = mtime
    return _DF_CACHE[tf_clean]


def _get_chart_static_series(tf: str) -> Dict[str, Any]:
    """Precomputes and caches the full-history candle and indicator series for fast (<50ms) full-history chart loads."""
    tf_clean = tf if tf in ("15m", "1h", "4h") else "1h"
    df = _get_scored_df(tf_clean)
    mtime = _DF_MTIME.get(tf_clean, 0.0)
    if tf_clean not in _CHART_STATIC_CACHE or _CHART_STATIC_MTIME.get(tf_clean) != mtime:
        _CHART_STATIC_MTIME[tf_clean] = mtime
        u_ts = df.index.as_unit("s").asi8.tolist()
        o_arr = np.round(df["open"].to_numpy(dtype=float), 2).tolist()
        h_arr = np.round(df["high"].to_numpy(dtype=float), 2).tolist()
        l_arr = np.round(df["low"].to_numpy(dtype=float), 2).tolist()
        c_arr = np.round(df["close"].to_numpy(dtype=float), 2).tolist()
        v_arr = df["volume"].fillna(0).to_numpy(dtype=int).tolist()

        candles = [
            {"time": t, "open": o, "high": h, "low": l, "close": c, "volume": v}
            for t, o, h, l, c, v in zip(u_ts, o_arr, h_arr, l_arr, c_arr, v_arr)
        ]

        def _build_line(col_name: str) -> List[Dict[str, Any]]:
            if col_name not in df.columns:
                return []
            s = df[col_name]
            mask = s.notna().to_numpy()
            ts_sub = [u_ts[i] for i in np.flatnonzero(mask)]
            val_sub = np.round(s.to_numpy(dtype=float)[mask], 2).tolist()
            return [{"time": t, "value": v} for t, v in zip(ts_sub, val_sub)]

        _CHART_STATIC_CACHE[tf_clean] = {
            "u_ts": u_ts,
            "candles": candles,
            "ema20": _build_line("ema20_px"),
            "sma50": _build_line("sma50_px"),
            "vwap": _build_line("rth_vwap_px"),
            "on_high": _build_line("on_high_px"),
            "on_low": _build_line("on_low_px"),
        }
    return _CHART_STATIC_CACHE[tf_clean]


def _get_summary_report() -> Dict[str, Any]:
    global _SUMMARY_CACHE
    if _SUMMARY_CACHE is None:
        if nqe.RESULTS_JSON_PATH.exists():
            _SUMMARY_CACHE = json.loads(nqe.RESULTS_JSON_PATH.read_text(encoding="utf-8"))
        else:
            _SUMMARY_CACHE = {"timeframes": {}}
    return _SUMMARY_CACHE


def _build_signal_record(
    ts: pd.Timestamp,
    row: pd.Series,
    direction: str,
    k: int,
    mode: str,
    prob_th: float = 0.55,
    ratio_th: float = 1.30,
) -> Dict[str, Any]:
    is_short = direction == "short"
    prob_col = (
        ("prob_pivot_high_extreme" if mode == "extreme" else "prob_pivot_high")
        if is_short
        else ("prob_pivot_low_extreme" if mode == "extreme" else "prob_pivot_low")
    )
    prob = float(row.get(prob_col, 0.0))
    prob_dual = float(row.get("prob_pivot_high" if is_short else "prob_pivot_low", 0.0))
    prob_ext = float(row.get("prob_pivot_high_extreme" if is_short else "prob_pivot_low_extreme", 0.0))
    atr_pts = max(1.0, float(row.get("adr20_pts", row.get("atr14_pts", 25.0))))
    trig_px = float(row.get("trigger_short_px" if is_short else "trigger_long_px", row["close"]))

    pred_mfe_atr = float(row.get(f"pred_mfe_{direction}_{k}b_atr", 1.5))
    pred_mfe_q70_atr = float(row.get(f"pred_mfe_q70_{direction}_{k}b_atr", pred_mfe_atr * 1.15))
    pred_mae_atr = float(row.get(f"pred_mae_{direction}_{k}b_atr", 1.0))
    pred_mae_q75_atr = float(row.get(f"pred_mae_q75_{direction}_{k}b_atr", pred_mae_atr * 1.15))
    pred_ratio = float(row.get(f"pred_ratio_{direction}_{k}b", pred_mfe_atr / max(0.35, pred_mae_atr)))
    prob_asym = float(row.get(f"prob_asym_{direction}_{k}b", 0.5))

    act_mfe_atr = float(row.get(f"mfe_{direction}_{k}b_atr", 0.0))
    act_mae_atr = float(row.get(f"mae_{direction}_{k}b_atr", 0.0))
    act_net_atr = float(row.get(f"net_{direction}_{k}b_atr", 0.0))

    pred_mfe_pts = pred_mfe_atr * atr_pts
    pred_mae_pts = pred_mae_atr * atr_pts
    act_mfe_pts = act_mfe_atr * atr_pts
    act_mae_pts = act_mae_atr * atr_pts
    act_net_pts = act_net_atr * atr_pts

    if is_short:
        target_px = trig_px - pred_mfe_atr * atr_pts
        target_q70_px = trig_px - pred_mfe_q70_atr * atr_pts
        stop_px = trig_px + pred_mae_q75_atr * atr_pts
    else:
        target_px = trig_px + pred_mfe_atr * atr_pts
        target_q70_px = trig_px + pred_mfe_q70_atr * atr_pts
        stop_px = trig_px - pred_mae_q75_atr * atr_pts

    sess_id = int(row.get("session_id", 0))
    is_true_piv = bool(row.get("is_pivot_high" if is_short else "is_pivot_low", 0) == 1)
    crosses_roll = bool(row.get(f"crosses_roll_{k}b", False))
    is_roll_bar = bool(row.get("is_roll", False))

    # Evaluate the 3 Actionability Gates
    gate_prob_pass = prob >= prob_th - 1e-9
    gate_ratio_pass = pred_ratio >= ratio_th - 1e-9
    gate_roll_pass = not crosses_roll
    is_actionable = bool(gate_prob_pass and gate_ratio_pass and gate_roll_pass)

    reasons: List[str] = []
    short_reasons: List[str] = []
    if not gate_roll_pass:
        reasons.append(
            f"Blocked by Quarterly Contract-Roll Seam Guard: forward {k}-bar holding window crosses a quarterly NQ contract roll (basis jump)."
        )
        short_reasons.append(f"Contract Roll in {k}B")
    if not gate_prob_pass:
        diff_pct = (prob_th - prob) * 100.0
        reasons.append(
            f"ML Pivot Probability ({prob * 100.0:.1f}%) is below the {prob_th * 100.0:.0f}% threshold (missed by -{diff_pct:.1f}%)."
        )
        short_reasons.append(f"Prob {prob * 100.0:.1f}% < {prob_th * 100.0:.0f}%")
    if not gate_ratio_pass:
        diff_r = ratio_th - pred_ratio
        reasons.append(
            f"Predicted {k}-Bar MFE/MAE Ratio ({pred_ratio:.2f}x) is below the {ratio_th:.2f}x asymmetry hurdle (missed by -{diff_r:.2f}x)."
        )
        short_reasons.append(f"MFE/MAE {pred_ratio:.2f}x < {ratio_th:.2f}x")

    if is_actionable:
        status_code = "ACTIONABLE"
        status_label = "ACTIONABLE"
        non_actionable_reason = "Passed all 3 gates: ML Pivot Probability, Predicted MFE/MAE Ratio, and Contract-Roll Seam Guard."
        non_actionable_short = "Actionable"
    else:
        if not gate_roll_pass and gate_prob_pass:
            status_code = "BLOCKED_CONTRACT_ROLL"
        elif not gate_prob_pass and gate_ratio_pass:
            status_code = "BELOW_PROB_THRESH"
        elif not gate_ratio_pass and gate_prob_pass:
            status_code = "BELOW_RATIO_THRESH"
        else:
            status_code = "BELOW_THRESHOLDS"
        status_label = "NON-ACTIONABLE"
        non_actionable_reason = " | ".join(reasons)
        non_actionable_short = " + ".join(short_reasons)

    gates = [
        {
            "name": "Gate 1: ML Pivot Probability",
            "passed": gate_prob_pass,
            "value": f"{prob * 100.0:.1f}%",
            "required": f">= {prob_th * 100.0:.0f}%",
            "note": (
                f"Pass (+{(prob - prob_th) * 100.0:.1f}% above hurdle)"
                if gate_prob_pass
                else f"Failed (-{(prob_th - prob) * 100.0:.1f}% below hurdle; Dual={prob_dual * 100.0:.1f}%, At-Extreme={prob_ext * 100.0:.1f}%)"
            ),
        },
        {
            "name": f"Gate 2: Pred {k}-Bar MFE/MAE Ratio",
            "passed": gate_ratio_pass,
            "value": f"{pred_ratio:.2f}x",
            "required": f">= {ratio_th:.2f}x",
            "note": (
                f"Pass (+{pred_ratio - ratio_th:.2f}x above hurdle)"
                if gate_ratio_pass
                else f"Failed (-{ratio_th - pred_ratio:.2f}x below hurdle)"
            ),
        },
        {
            "name": f"Gate 3: {k}-Bar Contract-Roll Seam Guard",
            "passed": gate_roll_pass,
            "value": "CLEAR" if gate_roll_pass else f"CROSSES ROLL ({k}B)",
            "required": "No Roll in Window",
            "note": (
                f"No quarterly contract roll within next {k} bars"
                if gate_roll_pass
                else f"Forward {k}-bar window crosses quarterly contract roll seam (basis jump)"
            ),
        },
    ]

    # Simulated bracket P&L (conservative tie-breaking if both target and stop are reached in K-bar window)
    tgt_atr = min(pred_mfe_atr * 1.05, 3.5)
    stp_atr = max(pred_mae_atr * 1.20, 1.0)
    hit_t = act_mfe_atr >= tgt_atr
    hit_s = act_mae_atr >= stp_atr
    if hit_t and not hit_s:
        r_out = tgt_atr
    elif hit_s:
        r_out = -stp_atr
    else:
        r_out = act_net_atr
    sim_bracket_pts = round(r_out * atr_pts, 1)

    return {
        "timestamp": ts.strftime("%Y-%m-%d %H:%M"),
        "unix_ts": int(ts.timestamp()),
        "direction": "SHORT" if is_short else "LONG",
        "label": "PIVOT HIGH (Short)" if is_short else "PIVOT LOW (Long)",
        "session": SESSION_NAMES.get(sess_id, "Globex"),
        "trigger_px": round(trig_px, 2),
        "open_px": round(float(row["open"]), 2),
        "high_px": round(float(row["high"]), 2),
        "low_px": round(float(row["low"]), 2),
        "close_px": round(float(row["close"]), 2),
        "atr_pts": round(atr_pts, 1),
        "prob_pct": round(prob * 100.0, 1),
        "prob_dual_pct": round(prob_dual * 100.0, 1),
        "prob_extreme_pct": round(prob_ext * 100.0, 1),
        "pred_mfe_atr": round(pred_mfe_atr, 2),
        "pred_mfe_pts": round(pred_mfe_pts, 1),
        "pred_mae_atr": round(pred_mae_atr, 2),
        "pred_mae_pts": round(pred_mae_pts, 1),
        "pred_ratio": round(pred_ratio, 2),
        "prob_asym_pct": round(prob_asym * 100.0, 1),
        "target_px": round(target_px, 2),
        "target_q70_px": round(target_q70_px, 2),
        "stop_px": round(stop_px, 2),
        "act_mfe_atr": round(act_mfe_atr, 2),
        "act_mfe_pts": round(act_mfe_pts, 1),
        "act_mae_atr": round(act_mae_atr, 2),
        "act_mae_pts": round(act_mae_pts, 1),
        "act_ratio": round(act_mfe_atr / max(0.08, act_mae_atr), 2),
        "act_net_atr": round(act_net_atr, 2),
        "act_net_pts": round(act_net_pts, 1),
        "sim_bracket_pts": sim_bracket_pts,
        "is_true_pivot": is_true_piv,
        "is_actionable": is_actionable,
        "crosses_roll": crosses_roll,
        "is_roll_bar": is_roll_bar,
        "status_code": status_code,
        "status_label": status_label,
        "non_actionable_reason": non_actionable_reason,
        "non_actionable_short": non_actionable_short,
        "gates": gates,
        "swept_on": bool(row.get("swept_on_high" if is_short else "swept_on_low", 0) > 0),
        "swept_prev_rth": bool(row.get("swept_prev_rth_high" if is_short else "swept_prev_rth_low", 0) > 0),
        "tod_rvol": round(float(row.get("tod_rvol", 1.0)), 2),
        "rsi14_ext": round(float(row.get("rsi14_high" if is_short else "rsi14_low", 50.0)), 1),
        "m1_vol_delta": round(float(row.get("m1_vol_delta_frac", 0.0)) * 100.0, 1),
        "m1_absorption_pct": round(
            float(row.get("m1_vol_top25_frac" if is_short else "m1_vol_bot25_frac", 0.0)) * 100.0, 1
        ),
    }


@app.route("/api/summary")
def api_summary():
    return jsonify(_get_summary_report())


@app.route("/api/bar")
def api_bar():
    """Inspects any bar on the chart by Unix timestamp (or ISO timestamp) and explains why it was or wasn't actionable."""
    tf = request.args.get("tf", "1h")
    k = int(request.args.get("horizon", 8))
    if k not in (4, 8, 12):
        k = 8
    prob_th = float(request.args.get("prob_th", 0.55))
    ratio_th = float(request.args.get("ratio_th", 1.30))
    mode = request.args.get("mode", "dual")
    ts_arg = request.args.get("ts", "").strip()
    direction_arg = request.args.get("direction", "auto").lower()

    df = _get_scored_df(tf)
    if not ts_arg:
        pos = len(df) - 1
    else:
        try:
            if ts_arg.isdigit():
                target_dt = pd.to_datetime(int(ts_arg), unit="s")
            else:
                target_dt = pd.Timestamp(ts_arg)
            pos = int(df.index.searchsorted(target_dt))
            if pos >= len(df):
                pos = len(df) - 1
            elif pos > 0 and abs((df.index[pos - 1] - target_dt).total_seconds()) < abs((df.index[pos] - target_dt).total_seconds()):
                pos = pos - 1
        except Exception:
            pos = len(df) - 1

    ts = df.index[pos]
    row = df.iloc[pos]
    p_hi_col = "prob_pivot_high_extreme" if mode == "extreme" else "prob_pivot_high"
    p_lo_col = "prob_pivot_low_extreme" if mode == "extreme" else "prob_pivot_low"
    p_hi = float(row.get(p_hi_col, 0.0))
    p_lo = float(row.get(p_lo_col, 0.0))

    if direction_arg in ("short", "high"):
        chosen_dir = "short"
    elif direction_arg in ("long", "low"):
        chosen_dir = "long"
    else:
        # Pick direction with higher pivot probability (or confirmed swing pivot flag)
        if p_hi >= p_lo:
            chosen_dir = "short"
        else:
            chosen_dir = "long"

    rec = _build_signal_record(ts, row, chosen_dir, k, mode, prob_th=prob_th, ratio_th=ratio_th)
    rec_alt = _build_signal_record(
        ts, row, "long" if chosen_dir == "short" else "short", k, mode, prob_th=prob_th, ratio_th=ratio_th
    )
    return jsonify({
        "tf": tf,
        "horizon": k,
        "signal": rec,
        "alt_signal": rec_alt,
    })


@app.route("/api/signals")
def api_signals():
    tf = request.args.get("tf", "1h")
    k = int(request.args.get("horizon", 8))
    if k not in (4, 8, 12):
        k = 8
    prob_th = float(request.args.get("prob_th", 0.55))
    ratio_th = float(request.args.get("ratio_th", 1.30))
    mode = request.args.get("mode", "dual")
    period = request.args.get("period", "oos")
    direction_filter = request.args.get("direction", "all")
    include_non_actionable = request.args.get("include_non_actionable", "0") == "1"

    df = _get_scored_df(tf)
    if period == "oos":
        sub_df = df[df.index >= pd.Timestamp(nqe.OOS_START)].copy()
    elif period == "train":
        sub_df = df[df.index < pd.Timestamp(nqe.OOS_START)].copy()
    else:
        sub_df = df.copy()

    p_hi_col = "prob_pivot_high_extreme" if mode == "extreme" else "prob_pivot_high"
    p_lo_col = "prob_pivot_low_extreme" if mode == "extreme" else "prob_pivot_low"
    roll_col = f"crosses_roll_{k}b"

    # Actionable masks (pass Prob + Ratio + No Roll Seam)
    act_hi_mask = (sub_df[p_hi_col] >= prob_th) & (sub_df[f"pred_ratio_short_{k}b"] >= ratio_th) & (~sub_df[roll_col])
    act_lo_mask = (sub_df[p_lo_col] >= prob_th) & (sub_df[f"pred_ratio_long_{k}b"] >= ratio_th) & (~sub_df[roll_col])

    # Non-actionable candidate masks (Roll-blocked signals OR Near-miss / structural swing pivots with prob >= max(0.38, prob_th - 0.10))
    near_prob_floor = max(0.38, prob_th - 0.10)
    non_act_hi_mask = (~act_hi_mask) & (
        ((sub_df[p_hi_col] >= prob_th) & (sub_df[f"pred_ratio_short_{k}b"] >= ratio_th) & sub_df[roll_col])
        | ((sub_df[p_hi_col] >= near_prob_floor) & ((sub_df["is_pivot_high"] == 1) | (sub_df[p_hi_col] >= prob_th)))
    )
    non_act_lo_mask = (~act_lo_mask) & (
        ((sub_df[p_lo_col] >= prob_th) & (sub_df[f"pred_ratio_long_{k}b"] >= ratio_th) & sub_df[roll_col])
        | ((sub_df[p_lo_col] >= near_prob_floor) & ((sub_df["is_pivot_low"] == 1) | (sub_df[p_lo_col] >= prob_th)))
    )

    actionable_signals: List[Dict[str, Any]] = []
    non_actionable_signals: List[Dict[str, Any]] = []

    if direction_filter in ("all", "short"):
        for ts, row in sub_df[act_hi_mask].iterrows():
            actionable_signals.append(_build_signal_record(ts, row, "short", k, mode, prob_th=prob_th, ratio_th=ratio_th))
        for ts, row in sub_df[non_act_hi_mask].iterrows():
            non_actionable_signals.append(_build_signal_record(ts, row, "short", k, mode, prob_th=prob_th, ratio_th=ratio_th))

    if direction_filter in ("all", "long"):
        for ts, row in sub_df[act_lo_mask].iterrows():
            actionable_signals.append(_build_signal_record(ts, row, "long", k, mode, prob_th=prob_th, ratio_th=ratio_th))
        for ts, row in sub_df[non_act_lo_mask].iterrows():
            non_actionable_signals.append(_build_signal_record(ts, row, "long", k, mode, prob_th=prob_th, ratio_th=ratio_th))

    actionable_signals.sort(key=lambda x: x["unix_ts"], reverse=True)
    non_actionable_signals.sort(key=lambda x: x["unix_ts"], reverse=True)

    # Compute chronological 2-unit bracket simulation across the exact filtered signals
    chrono_sim = nqe.simulate_chronological_2unit_bracket(
        sub_df,
        p_th=prob_th,
        r_th=ratio_th,
        k=k,
        mode=mode,
        direction_filter=direction_filter,
    )

    # Compute aggregate KPI metrics strictly over actionable signals
    n_sig = len(actionable_signals)
    if n_sig > 0:
        true_piv_rate = sum(1 for s in actionable_signals if s["is_true_pivot"]) / n_sig * 100.0
        fav_win_rate = sum(1 for s in actionable_signals if s["act_mfe_atr"] > s["act_mae_atr"]) / n_sig * 100.0
        avg_pred_ratio = float(np.mean([s["pred_ratio"] for s in actionable_signals]))
        avg_mfe_atr = float(np.mean([s["act_mfe_atr"] for s in actionable_signals]))
        avg_mae_atr = float(np.mean([s["act_mae_atr"] for s in actionable_signals]))
        avg_mfe_pts = float(np.mean([s["act_mfe_pts"] for s in actionable_signals]))
        avg_mae_pts = float(np.mean([s["act_mae_pts"] for s in actionable_signals]))

        kpis = {
            "total_signals": n_sig,
            "chronological_executed_trades": chrono_sim["trades"],
            "non_actionable_count": len(non_actionable_signals),
            "roll_blocked_count": sum(1 for s in non_actionable_signals if s["status_code"] == "BLOCKED_CONTRACT_ROLL"),
            "near_miss_count": sum(1 for s in non_actionable_signals if s["status_code"] != "BLOCKED_CONTRACT_ROLL"),
            "true_pivot_rate_pct": round(true_piv_rate, 1),
            "favorable_win_rate_pct": round(fav_win_rate, 1),
            "bracket_win_rate_pct": chrono_sim["win_rate_pct"],
            "tp1_hit_rate_pct": chrono_sim["tp1_hit_rate_pct"],
            "avg_r_multiple": chrono_sim["avg_r"],
            "avg_pred_ratio": round(avg_pred_ratio, 2),
            "realized_mfe_atr": round(avg_mfe_atr, 2),
            "realized_mae_atr": round(avg_mae_atr, 2),
            "realized_mfe_mae_ratio": round(avg_mfe_atr / max(0.05, avg_mae_atr), 2),
            "avg_mfe_pts": round(avg_mfe_pts, 1),
            "avg_mae_pts": round(avg_mae_pts, 1),
            "avg_net_pts": chrono_sim["avg_net_pts"],
            "profit_factor": chrono_sim["profit_factor"],
            "max_dd_pts": chrono_sim["max_dd_pts"],
            "total_net_pts": round(chrono_sim["total_net_pts"], 0),
            "total_pnl_nq_usd": round(chrono_sim["total_pnl_1_nq_usd"], 0),
            "total_pnl_mnq_usd": round(chrono_sim["total_pnl_1_mnq_usd"], 0),
        }
    else:
        kpis = {
            "total_signals": 0,
            "chronological_executed_trades": 0,
            "non_actionable_count": len(non_actionable_signals),
            "roll_blocked_count": sum(1 for s in non_actionable_signals if s["status_code"] == "BLOCKED_CONTRACT_ROLL"),
            "near_miss_count": sum(1 for s in non_actionable_signals if s["status_code"] != "BLOCKED_CONTRACT_ROLL"),
            "true_pivot_rate_pct": 0.0,
            "favorable_win_rate_pct": 0.0,
            "bracket_win_rate_pct": 0.0,
            "tp1_hit_rate_pct": 0.0,
            "tp2_hit_rate_pct": 0.0,
            "avg_r_multiple": 0.0,
            "avg_pred_ratio": 0.0,
            "realized_mfe_atr": 0.0,
            "realized_mae_atr": 0.0,
            "realized_mfe_mae_ratio": 0.0,
            "avg_mfe_pts": 0.0,
            "avg_mae_pts": 0.0,
            "avg_net_pts": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_r": 0.0,
            "total_net_pts": 0.0,
            "total_pnl_nq_usd": 0.0,
            "total_pnl_mnq_usd": 0.0,
        }

    combined = actionable_signals + non_actionable_signals if include_non_actionable else actionable_signals
    combined.sort(key=lambda x: x["unix_ts"], reverse=True)

    return jsonify({
        "tf": tf,
        "horizon": k,
        "prob_th": prob_th,
        "ratio_th": ratio_th,
        "mode": mode,
        "period": period,
        "kpis": kpis,
        "signals": combined[:800],
        "non_actionable_signals": non_actionable_signals[:400],
    })


@app.route("/api/chart")
def api_chart():
    tf = request.args.get("tf", "1h")
    k = int(request.args.get("horizon", 8))
    if k not in (4, 8, 12):
        k = 8
    prob_th = float(request.args.get("prob_th", 0.55))
    ratio_th = float(request.args.get("ratio_th", 1.30))
    mode = request.args.get("mode", "dual")
    focus_ts = request.args.get("focus_ts", "")
    bars_arg = request.args.get("bars", "all").strip().lower()
    show_non_actionable = request.args.get("show_non_actionable", "1") != "0"

    df = _get_scored_df(tf)
    static = _get_chart_static_series(tf)

    if bars_arg in ("all", "", "full", "max", "0"):
        start_i = 0
        end_i = len(df)
        sub = df
        candles = static["candles"]
        ema20_line = static["ema20"]
        sma50_line = static["sma50"]
        vwap_line = static["vwap"]
        on_high_line = static["on_high"]
        on_low_line = static["on_low"]
        ts_list = static["u_ts"]
    else:
        n_bars = max(150, int(bars_arg))
        if focus_ts:
            try:
                if focus_ts.isdigit():
                    target_dt = pd.to_datetime(int(focus_ts), unit="s")
                else:
                    target_dt = pd.Timestamp(focus_ts)
                pos = int(df.index.searchsorted(target_dt))
                start_i = max(0, pos - int(n_bars * 0.65))
                end_i = min(len(df), start_i + n_bars)
            except Exception:
                start_i = max(0, len(df) - n_bars)
                end_i = len(df)
        else:
            start_i = max(0, len(df) - n_bars)
            end_i = len(df)

        sub = df.iloc[start_i:end_i]
        ts_list = static["u_ts"][start_i:end_i]
        candles = static["candles"][start_i:end_i]
        min_u = ts_list[0] if ts_list else 0
        max_u = ts_list[-1] if ts_list else 0
        ema20_line = [p for p in static["ema20"] if min_u <= p["time"] <= max_u]
        sma50_line = [p for p in static["sma50"] if min_u <= p["time"] <= max_u]
        vwap_line = [p for p in static["vwap"] if min_u <= p["time"] <= max_u]
        on_high_line = [p for p in static["on_high"] if min_u <= p["time"] <= max_u]
        on_low_line = [p for p in static["on_low"] if min_u <= p["time"] <= max_u]

    p_hi_col = "prob_pivot_high_extreme" if mode == "extreme" else "prob_pivot_high"
    p_lo_col = "prob_pivot_low_extreme" if mode == "extreme" else "prob_pivot_low"
    roll_col = f"crosses_roll_{k}b"
    near_prob_floor = max(0.38, prob_th - 0.10)

    p_hi_s = sub[p_hi_col]
    p_lo_s = sub[p_lo_col]
    r_hi_s = sub[f"pred_ratio_short_{k}b"]
    r_lo_s = sub[f"pred_ratio_long_{k}b"]
    roll_s = sub[roll_col]

    # Candidate masks (vectorized over sub)
    hi_act_mask = (p_hi_s >= prob_th) & (r_hi_s >= ratio_th) & (~roll_s)
    lo_act_mask = (p_lo_s >= prob_th) & (r_lo_s >= ratio_th) & (~roll_s)

    if show_non_actionable:
        hi_non_act_mask = (~hi_act_mask) & (
            ((p_hi_s >= prob_th) & (r_hi_s >= ratio_th) & roll_s)
            | ((p_hi_s >= near_prob_floor) & ((sub["is_pivot_high"] == 1) | (p_hi_s >= prob_th)))
        )
        lo_non_act_mask = (~lo_act_mask) & (
            ((p_lo_s >= prob_th) & (r_lo_s >= ratio_th) & roll_s)
            | ((p_lo_s >= near_prob_floor) & ((sub["is_pivot_low"] == 1) | (p_lo_s >= prob_th)))
        )
        cand_mask = hi_act_mask | lo_act_mask | hi_non_act_mask | lo_non_act_mask
    else:
        hi_non_act_mask = pd.Series(False, index=sub.index)
        lo_non_act_mask = pd.Series(False, index=sub.index)
        cand_mask = hi_act_mask | lo_act_mask

    markers: List[Dict[str, Any]] = []
    signals_in_view: List[Dict[str, Any]] = []

    cand_indices = np.flatnonzero(cand_mask.to_numpy())
    for idx_i in cand_indices:
        ts = sub.index[idx_i]
        row = sub.iloc[idx_i]
        u_ts = ts_list[idx_i]

        # High candidate
        if bool(hi_act_mask.iloc[idx_i]) or bool(hi_non_act_mask.iloc[idx_i]):
            sig = _build_signal_record(ts, row, "short", k, mode, prob_th=prob_th, ratio_th=ratio_th)
            end_bar_idx = min(len(ts_list) - 1, int(idx_i) + k)
            sig["horizon_end_ts"] = ts_list[end_bar_idx]
            signals_in_view.append(sig)
            if sig["is_actionable"]:
                markers.append({
                    "time": u_ts,
                    "position": "aboveBar",
                    "color": "#f43f5e",
                    "shape": "arrowDown",
                    "text": f"PIV HI {sig['prob_pct']:.0f}% | MFE/MAE {sig['pred_ratio']:.1f}x",
                })
            elif sig["status_code"] == "BLOCKED_CONTRACT_ROLL":
                markers.append({
                    "time": u_ts,
                    "position": "aboveBar",
                    "color": "#f59e0b",
                    "shape": "circle",
                    "text": f"⛔ PIV HI {sig['prob_pct']:.0f}% (Non-Actionable: {sig['non_actionable_short']})",
                })
            else:
                markers.append({
                    "time": u_ts,
                    "position": "aboveBar",
                    "color": "#94a3b8",
                    "shape": "circle",
                    "text": f"⚠️ PIV HI {sig['prob_pct']:.0f}% (Non-Actionable: {sig['non_actionable_short']})",
                })

        # Low candidate
        if bool(lo_act_mask.iloc[idx_i]) or bool(lo_non_act_mask.iloc[idx_i]):
            sig = _build_signal_record(ts, row, "long", k, mode, prob_th=prob_th, ratio_th=ratio_th)
            end_bar_idx = min(len(ts_list) - 1, int(idx_i) + k)
            sig["horizon_end_ts"] = ts_list[end_bar_idx]
            signals_in_view.append(sig)
            if sig["is_actionable"]:
                markers.append({
                    "time": u_ts,
                    "position": "belowBar",
                    "color": "#10b981",
                    "shape": "arrowUp",
                    "text": f"PIV LO {sig['prob_pct']:.0f}% | MFE/MAE {sig['pred_ratio']:.1f}x",
                })
            elif sig["status_code"] == "BLOCKED_CONTRACT_ROLL":
                markers.append({
                    "time": u_ts,
                    "position": "belowBar",
                    "color": "#f59e0b",
                    "shape": "circle",
                    "text": f"⛔ PIV LO {sig['prob_pct']:.0f}% (Non-Actionable: {sig['non_actionable_short']})",
                })
            else:
                markers.append({
                    "time": u_ts,
                    "position": "belowBar",
                    "color": "#94a3b8",
                    "shape": "circle",
                    "text": f"⚠️ PIV LO {sig['prob_pct']:.0f}% (Non-Actionable: {sig['non_actionable_short']})",
                })

    return jsonify({
        "tf": tf,
        "horizon": k,
        "total_bars": len(candles),
        "first_bar_time": df.index[start_i].strftime("%Y-%m-%d %H:%M") if len(candles) else "",
        "last_bar_time": df.index[end_i - 1].strftime("%Y-%m-%d %H:%M") if len(candles) else "",
        "candles": candles,
        "ema20": ema20_line,
        "sma50": sma50_line,
        "vwap": vwap_line,
        "on_high": on_high_line,
        "on_low": on_low_line,
        "markers": markers,
        "signals_in_view": signals_in_view,
    })


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NQ Intraday ML Studio — Pivot High/Low & Forward MAE/MFE Explorer (Port 8790)</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root {
    --bg: #0b0f17;
    --panel: #111827;
    --panel-alt: #161f33;
    --border: #1e293b;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --green: #10b981;
    --red: #f43f5e;
    --amber: #f59e0b;
    --purple: #a855f7;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
    font-size: 13px;
    line-height: 1.45;
  }
  header {
    background: linear-gradient(90deg, #0f172a 0%, #111827 100%);
    border-bottom: 1px solid var(--border);
    padding: 12px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .brand-badge {
    background: rgba(56, 189, 248, 0.15);
    color: var(--accent);
    border: 1px solid rgba(56, 189, 248, 0.4);
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 12px;
    letter-spacing: 0.5px;
  }
  .brand h1 {
    font-size: 16px;
    font-weight: 700;
    color: #f8fafc;
  }
  .brand-sub {
    color: var(--muted);
    font-size: 11.5px;
  }
  .controls-bar {
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    padding: 10px 20px;
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    align-items: center;
  }
  .ctrl-group {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .ctrl-label {
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
  }
  .btn-group {
    display: inline-flex;
    background: #0b0f17;
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
  }
  .btn-group button {
    background: transparent;
    color: var(--muted);
    border: none;
    padding: 5px 10px;
    font-size: 11.5px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
  }
  .btn-group button.active {
    background: var(--accent);
    color: #0b0f17;
  }
  .btn-group button:hover:not(.active) {
    color: var(--text);
    background: rgba(255,255,255,0.04);
  }
  select, input[type="range"] {
    background: #0b0f17;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 12px;
  }
  .val-pill {
    background: rgba(56, 189, 248, 0.12);
    color: var(--accent);
    padding: 2px 7px;
    border-radius: 4px;
    font-weight: 700;
    font-size: 12px;
    min-width: 44px;
    text-align: center;
  }
  .kpi-strip {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px;
    padding: 12px 20px;
    background: #0d1320;
    border-bottom: 1px solid var(--border);
  }
  .kpi-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 9px 12px;
  }
  .kpi-title {
    color: var(--muted);
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
  }
  .kpi-val {
    font-size: 17px;
    font-weight: 700;
    margin-top: 2px;
  }
  .kpi-sub {
    font-size: 11px;
    color: var(--muted);
    margin-top: 1px;
  }
  .pos { color: var(--green); }
  .neg { color: var(--red); }
  .acc { color: var(--accent); }
  .amb { color: var(--amber); }

  .main-grid {
    display: grid;
    grid-template-columns: 1fr 400px;
    gap: 14px;
    padding: 14px 20px;
  }
  .main-grid.fullscreen-chart {
    grid-template-columns: 1fr !important;
  }
  .main-grid.fullscreen-chart .inspector-card {
    display: none !important;
  }
  @media (max-width: 1200px) {
    .main-grid { grid-template-columns: 1fr; }
  }
  .chart-card, .inspector-card, .section-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }
  .card-header {
    padding: 10px 14px;
    background: var(--panel-alt);
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
  }
  .card-title {
    font-weight: 700;
    font-size: 13px;
    color: #f8fafc;
  }
  .legend-row {
    display: flex;
    gap: 12px;
    font-size: 11px;
    color: var(--muted);
    align-items: center;
    flex-wrap: wrap;
  }
  .dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 4px;
  }
  #tv-chart {
    width: 100%;
    height: 500px;
  }
  .main-grid.fullscreen-chart #tv-chart {
    height: 75vh;
  }
  .inspector-body {
    padding: 12px 14px;
  }
  .sig-banner {
    padding: 9px 12px;
    border-radius: 6px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-weight: 700;
  }
  .sig-banner.short {
    background: rgba(244, 63, 94, 0.14);
    border: 1px solid rgba(244, 63, 94, 0.4);
    color: #fda4af;
  }
  .sig-banner.long {
    background: rgba(16, 185, 129, 0.14);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #6ee7b7;
  }
  .gate-box {
    border-radius: 6px;
    padding: 8px 10px;
    margin-bottom: 10px;
    font-size: 11.5px;
  }
  .gate-box.pass {
    background: rgba(16, 185, 129, 0.10);
    border: 1px solid rgba(16, 185, 129, 0.38);
  }
  .gate-box.fail {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.45);
  }
  .gate-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 8px;
    padding: 4px 0;
    border-top: 1px solid rgba(255,255,255,0.06);
    font-size: 11px;
  }
  .detail-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 10px;
  }
  .detail-box {
    background: #0b0f17;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 7px 9px;
  }
  .detail-box .lbl {
    font-size: 10px;
    color: var(--muted);
    text-transform: uppercase;
  }
  .detail-box .val {
    font-size: 13px;
    font-weight: 700;
    margin-top: 2px;
  }
  .tabs-bar {
    display: flex;
    gap: 8px;
    padding: 0 20px;
    margin-top: 4px;
  }
  .tab-btn {
    background: var(--panel);
    color: var(--muted);
    border: 1px solid var(--border);
    border-bottom: none;
    padding: 9px 16px;
    border-radius: 8px 8px 0 0;
    font-weight: 700;
    font-size: 12.5px;
    cursor: pointer;
  }
  .tab-btn.active {
    background: var(--panel-alt);
    color: var(--accent);
    border-color: var(--accent);
  }
  .tab-panel {
    display: none;
    padding: 0 20px 24px 20px;
  }
  .tab-panel.active {
    display: block;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }
  th, td {
    padding: 8px 10px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }
  th {
    background: #0f172a;
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    position: sticky;
    top: 0;
    z-index: 2;
  }
  tbody tr {
    cursor: pointer;
    transition: background 0.12s;
  }
  tbody tr:hover {
    background: rgba(56, 189, 248, 0.08);
  }
  tbody tr.selected-row {
    background: rgba(56, 189, 248, 0.16);
  }
  tbody tr.non-actionable-row {
    background: rgba(245, 158, 11, 0.05);
  }
  .badge {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 10.5px;
    font-weight: 700;
  }
  .badge-short { background: rgba(244, 63, 94, 0.18); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.35); }
  .badge-long { background: rgba(16, 185, 129, 0.18); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); }
  .badge-sweep { background: rgba(168, 85, 247, 0.18); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.35); margin-right: 3px; }
  .badge-true { background: rgba(56, 189, 248, 0.15); color: #38bdf8; }
  .badge-act { background: rgba(16, 185, 129, 0.18); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
  .badge-roll { background: rgba(245, 158, 11, 0.20); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.45); }
  .badge-miss { background: rgba(148, 163, 184, 0.16); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.35); }
  .verdict-box {
    background: linear-gradient(135deg, rgba(16,185,129,0.10) 0%, rgba(56,189,248,0.10) 100%);
    border: 1px solid rgba(16,185,129,0.35);
    border-radius: 8px;
    padding: 14px 18px;
    margin: 12px 20px 4px 20px;
  }
  .verdict-title {
    font-weight: 700;
    font-size: 13.5px;
    color: #34d399;
    margin-bottom: 4px;
  }
  .two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-top: 12px;
  }
  @media (max-width: 1000px) {
    .two-col { grid-template-columns: 1fr; }
  }
  .table-scroll {
    max-height: 460px;
    overflow-y: auto;
  }
</style>
</head>
<body>

<header>
  <div class="brand">
    <span class="brand-badge">NQ FUTURES ML · PORT 8790</span>
    <div>
      <h1>NQ Intraday Pivot High/Low & Forward MAE/MFE Prediction Studio</h1>
      <div class="brand-sub">10-Year 1-Minute NQ Study (Train: 2016–2024 · Strict Out-of-Sample Test: 2024–2026) · Full 10Y History Loaded · Click Any Bar/Marker for Actionability Gate Breakdown</div>
    </div>
  </div>
  <div class="legend-row">
    <span><i class="dot" style="background:#38bdf8"></i>EMA 20</span>
    <span><i class="dot" style="background:#f59e0b"></i>SMA 50</span>
    <span><i class="dot" style="background:#a855f7"></i>RTH VWAP</span>
    <span><i class="dot" style="background:#10b981"></i>Actionable Pivot Low / MFE Target</span>
    <span><i class="dot" style="background:#f43f5e"></i>Actionable Pivot High / MAE Stop</span>
    <span><i class="dot" style="background:#f59e0b"></i>⛔ Blocked by Contract Roll</span>
    <span><i class="dot" style="background:#94a3b8"></i>⚠️ Sub-Threshold Swing Pivot</span>
  </div>
</header>

<div class="controls-bar">
  <div class="ctrl-group">
    <span class="ctrl-label">Timeframe:</span>
    <div class="btn-group" id="tf-group">
      <button data-val="15m">15m</button>
      <button data-val="1h" class="active">1h</button>
      <button data-val="4h">4h</button>
    </div>
  </div>

  <div class="ctrl-group">
    <span class="ctrl-label">Forward Horizon (K):</span>
    <div class="btn-group" id="k-group">
      <button data-val="4">Next 4 Bars</button>
      <button data-val="8" class="active">Next 8 Bars</button>
      <button data-val="12">Next 12 Bars</button>
    </div>
  </div>

  <div class="ctrl-group">
    <span class="ctrl-label">Sample Window:</span>
    <div class="btn-group" id="period-group">
      <button data-val="oos" class="active">2Y OOS (2024–2026)</button>
      <button data-val="train">Train (2016–2024)</button>
      <button data-val="all">All (10Y)</button>
    </div>
  </div>

  <div class="ctrl-group">
    <span class="ctrl-label">Model Mode:</span>
    <div class="btn-group" id="mode-group">
      <button data-val="dual" class="active">Dual-Path (Full + 1m Micro)</button>
      <button data-val="extreme">At-Extreme (Sub-Model A)</button>
    </div>
  </div>

  <div class="ctrl-group">
    <span class="ctrl-label">Direction:</span>
    <select id="dir-select">
      <option value="all">All Pivots (Long & Short)</option>
      <option value="long">Pivot Lows Only (Long)</option>
      <option value="short">Pivot Highs Only (Short)</option>
    </select>
  </div>

  <div class="ctrl-group">
    <span class="ctrl-label">Min Pivot Prob:</span>
    <input type="range" id="prob-slider" min="0.40" max="0.75" step="0.05" value="0.55">
    <span class="val-pill" id="prob-val">55%</span>
  </div>

  <div class="ctrl-group">
    <span class="ctrl-label">Min Pred MFE/MAE:</span>
    <input type="range" id="ratio-slider" min="0.80" max="2.50" step="0.10" value="1.30">
    <span class="val-pill" id="ratio-val">1.30x</span>
  </div>

  <div class="ctrl-group">
    <label style="display:inline-flex;align-items:center;gap:5px;cursor:pointer;font-size:11.5px;color:#fbbf24;font-weight:700;" title="Show non-actionable pivots on the chart and in the table with explicit explanations (e.g. Sep 18 16:00 blocked by Quarterly Contract Roll Seam, Sep 18 03:00 Prob 50.3% < 55%)">
      <input type="checkbox" id="chk-non-actionable" checked>
      <span>🔍 Show Non-Actionable & Why</span>
    </label>
  </div>
</div>

<div class="kpi-strip" id="kpi-strip">
  <div class="kpi-card">
    <div class="kpi-title">Actionable Signals</div>
    <div class="kpi-val acc" id="kpi-signals">--</div>
    <div class="kpi-sub" id="kpi-piv-rate">True Pivot Rate: --%</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">NQ-Spec vs Equity AUC</div>
    <div class="kpi-val pos" id="kpi-auc">--</div>
    <div class="kpi-sub" id="kpi-auc-sub">vs Generic Equity: --</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Favorable Excursion Win%</div>
    <div class="kpi-val pos" id="kpi-win">--</div>
    <div class="kpi-sub" id="kpi-bwin">Bracket Win Rate: --%</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Realized MFE / MAE</div>
    <div class="kpi-val acc" id="kpi-ratio">--</div>
    <div class="kpi-sub" id="kpi-pred-ratio">Predicted Ratio: --x</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Avg Realized MFE</div>
    <div class="kpi-val pos" id="kpi-mfe">--</div>
    <div class="kpi-sub" id="kpi-mfe-atr">-- ATR</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Avg Realized MAE</div>
    <div class="kpi-val neg" id="kpi-mae">--</div>
    <div class="kpi-sub" id="kpi-mae-atr">-- ATR</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Avg Net Expectancy</div>
    <div class="kpi-val pos" id="kpi-net-pts">--</div>
    <div class="kpi-sub" id="kpi-pf">Profit Factor: --</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Simulated Net P&L (1 NQ)</div>
    <div class="kpi-val pos" id="kpi-pnl-nq">--</div>
    <div class="kpi-sub" id="kpi-pnl-mnq">1 MNQ ($2/pt): --</div>
  </div>
</div>

<div class="verdict-box">
  <div class="verdict-title">Empirical Verdict: Does Training NQ-Specific Intraday Models (2016–2024 Train → 2024–2026 OOS) Outperform Generic Models?</div>
  <div style="color:var(--text);font-size:12.5px;">
    <strong>YES — by a wide statistical margin across all three timeframes (`15m`, `1h`, `4h`).</strong>
    Generic daily equity pivot models applied zero-shot to intraday NQ achieve only <strong>0.51–0.72 ROC-AUC</strong> because equities only trade 09:30–16:00 RTH and lack 23-hour Globex session cyclicality, Overnight High/Low (`18:00–09:29 ET`) liquidity sweeps, Time-of-Day Slot RVOL (`tod_rvol`), and 1-minute intrabar volume absorption. Training specifically on 8 years of seam-adjusted 1-minute NQ data (`2016–2024`) lifts strict 2-year Out-of-Sample (`2024–2026`) accuracy to <strong>0.80–0.88 ROC-AUC</strong> (with <strong>60%–88% precision</strong> at high-conviction thresholds), while the Forward $K$-Bar MAE/MFE regressors monotonically separate top-quintile setups (`2.3x–2.7x` realized MFE/MAE ratio, `72%–78%` favorable excursion win rate).
  </div>
</div>

<div class="main-grid" id="main-grid">
  <div class="chart-card">
    <div class="card-header">
      <div>
        <div class="card-title" id="chart-title">NQ Futures Interactive Chart — Full 10-Year History Loaded (Click Any Candle or Signal Row to Inspect Actionability)</div>
        <div style="font-size:11.5px;color:var(--muted);" id="chart-sub">Loaded all history bars · Click any bar on the chart (e.g. Sep 18) to inspect its 3-Gate Actionability Checklist</div>
      </div>
      <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
        <span class="ctrl-label">Chart Zoom:</span>
        <div class="btn-group" id="zoom-group">
          <button onclick="setChartZoom('signal')" id="zbtn-signal" class="active" title="Zoom around currently selected signal">🔍 Signal</button>
          <button onclick="setChartZoom('1m')" id="zbtn-1m">1M</button>
          <button onclick="setChartZoom('3m')" id="zbtn-3m">3M</button>
          <button onclick="setChartZoom('1y')" id="zbtn-1y">1Y</button>
          <button onclick="setChartZoom('oos')" id="zbtn-oos">2Y OOS</button>
          <button onclick="setChartZoom('all')" id="zbtn-all" title="Fit all 10 years of history in view">All History (10Y)</button>
        </div>
        <button onclick="toggleFullscreenNqChart()" id="btn-nq-fullscreen" style="background:#0b0f17;color:var(--accent);border:1px solid rgba(56,189,248,0.45);padding:5px 10px;border-radius:6px;font-size:11.5px;font-weight:700;cursor:pointer;">⛶ Enlarge</button>
      </div>
    </div>
    <div id="tv-chart"></div>
  </div>

  <div class="inspector-card">
    <div class="card-header">
      <div class="card-title">Selected Pivot & Actionability Gate Inspector</div>
      <span class="badge badge-true" id="insp-horizon-badge">Next 8 Bars</span>
    </div>
    <div class="inspector-body" id="inspector-body">
      <div style="color:var(--muted);padding:20px 0;text-align:center;">
        Select any Pivot High or Pivot Low signal from the table below (or click any candle/marker on the chart) to inspect its 3-Gate Actionability Checklist & Forward K-Bar MFE/MAE projection.
      </div>
    </div>
  </div>
</div>

<div class="tabs-bar">
  <button class="tab-btn active" onclick="switchTab('signals')">1. Out-of-Sample Pivot Signal Explorer (Click Row to View on Chart)</button>
  <button class="tab-btn" onclick="switchTab('comparison')">2. NQ-Specific vs. Generic Equity Models & Feature Importance</button>
  <button class="tab-btn" onclick="switchTab('calibration')">3. Forward K-Bar MAE/MFE Calibration Quintiles (Q1–Q5) & Strategy Tiers</button>
</div>

<div class="tab-panel active" id="tab-signals">
  <div class="section-card">
    <div class="card-header">
      <div class="card-title">NQ Intraday Pivot High & Pivot Low Signals (Actionable & Non-Actionable Gate Explanations)</div>
      <div style="color:var(--muted);font-size:11.5px;" id="signals-table-sub">Click any row (or any bar on the chart) to center the chart and inspect its Actionability Gate Breakdown & MFE/MAE Zones</div>
    </div>
    <div class="table-scroll">
      <table>
        <thead>
          <tr>
            <th>Timestamp (ET)</th>
            <th>Direction</th>
            <th>Actionability & Gate Explanation</th>
            <th>Session Regime</th>
            <th>Trigger Px</th>
            <th>Pivot Prob</th>
            <th>Liquidity Sweeps & 1m Micro</th>
            <th>Pred MFE (K Bars)</th>
            <th>Pred MAE (K Bars)</th>
            <th>Pred Ratio</th>
            <th>Realized MFE</th>
            <th>Realized MAE</th>
            <th>Realized Ratio</th>
            <th>Sim Bracket P&L</th>
          </tr>
        </thead>
        <tbody id="signals-tbody"></tbody>
      </table>
    </div>
  </div>
</div>

<div class="tab-panel" id="tab-comparison">
  <div class="section-card" style="padding:14px;">
    <h3 style="margin-bottom:10px;color:#f8fafc;">Head-to-Head OOS Comparison: NQ-Specific ML (Trained 2016–2024) vs. Generic Daily Equity ML (2024–2026 OOS)</h3>
    <table>
      <thead>
        <tr>
          <th>Timeframe</th>
          <th>OOS Bars (2024–2026)</th>
          <th>Model Direction</th>
          <th>Base Rate</th>
          <th>Generic Equity Model ROC-AUC</th>
          <th>NQ-Specific Model ROC-AUC</th>
          <th>NQ-Specific PR-AUC</th>
          <th>Prec @ 55% Prob</th>
          <th>Prec @ 60% Prob</th>
          <th>Prec @ 70% Prob</th>
          <th>Precision Lift @ 65%</th>
        </tr>
      </thead>
      <tbody id="comp-tbody"></tbody>
    </table>

    <div class="two-col">
      <div class="section-card">
        <div class="card-header"><div class="card-title">Top 12 NQ Features — Pivot High Classifier (Short)</div></div>
        <table>
          <thead><tr><th>Rank</th><th>Feature Name</th><th>Split Importance</th></tr></thead>
          <tbody id="feat-hi-tbody"></tbody>
        </table>
      </div>
      <div class="section-card">
        <div class="card-header"><div class="card-title">Top 12 NQ Features — Pivot Low Classifier (Long)</div></div>
        <table>
          <thead><tr><th>Rank</th><th>Feature Name</th><th>Split Importance</th></tr></thead>
          <tbody id="feat-lo-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<div class="tab-panel" id="tab-calibration">
  <div class="section-card" style="padding:14px;">
    <h3 style="margin-bottom:10px;color:#f8fafc;">Out-of-Sample Forward K-Bar Predicted MFE/MAE Ratio Quintiles (Q1 Lowest → Q5 Top)</h3>
    <table>
      <thead>
        <tr>
          <th>Quintile Bucket</th>
          <th>OOS Count</th>
          <th>Avg Pred MFE/MAE</th>
          <th>Pred MFE (ATR)</th>
          <th>Pred MAE (ATR)</th>
          <th>Realized MFE (ATR / Pts)</th>
          <th>Realized MAE (ATR / Pts)</th>
          <th>Realized MFE/MAE Ratio</th>
          <th>Favorable Excursion Win%</th>
          <th>Avg Net Excursion (Pts)</th>
        </tr>
      </thead>
      <tbody id="quintiles-tbody"></tbody>
    </table>

    <h3 style="margin:18px 0 10px 0;color:#f8fafc;">Actionable OOS Bracket Strategy Tiers (Combining Pivot Probability + Predicted MFE/MAE Ratio Filter)</h3>
    <table>
      <thead>
        <tr>
          <th>Strategy Conviction Tier</th>
          <th>OOS Trades</th>
          <th>Bracket Win%</th>
          <th>Realized MFE (ATR / Pts)</th>
          <th>Realized MAE (ATR / Pts)</th>
          <th>Realized MFE/MAE</th>
          <th>Avg Net Pts / Trade</th>
          <th>Profit Factor</th>
          <th>Total Net Pts</th>
          <th>1 NQ Contract ($20/pt)</th>
          <th>1 MNQ Contract ($2/pt)</th>
        </tr>
      </thead>
      <tbody id="tiers-tbody"></tbody>
    </table>
  </div>
</div>

<script>
let state = {
  tf: '1h',
  horizon: 8,
  period: 'oos',
  mode: 'dual',
  direction: 'all',
  prob_th: 0.55,
  ratio_th: 1.30,
  show_non_actionable: true,
  zoom_mode: 'signal',
  focus_ts: '',
  summary: null,
  signals: [],
  selectedSignal: null,
  chartData: null,
};

let chart = null;
let candleSeries = null;
let ema20Series = null;
let sma50Series = null;
let vwapSeries = null;
let onHighSeries = null;
let onLowSeries = null;
let projEntrySeries = null;
let projTargetSeries = null;
let projStopSeries = null;

function initChart() {
  const container = document.getElementById('tv-chart');
  chart = LightweightCharts.createChart(container, {
    layout: { background: { type: 'solid', color: '#0b0f17' }, textColor: '#94a3b8' },
    grid: { vertLines: { color: '#161f33' }, horzLines: { color: '#161f33' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: '#1e293b' },
    timeScale: { borderColor: '#1e293b', timeVisible: true, secondsVisible: false, minBarSpacing: 0.02 },
  });

  candleSeries = chart.addCandlestickSeries({
    upColor: '#10b981', downColor: '#f43f5e',
    borderUpColor: '#10b981', borderDownColor: '#f43f5e',
    wickUpColor: '#10b981', wickDownColor: '#f43f5e',
  });

  ema20Series = chart.addLineSeries({ color: '#38bdf8', lineWidth: 1, title: 'EMA20' });
  sma50Series = chart.addLineSeries({ color: '#f59e0b', lineWidth: 1, title: 'SMA50' });
  vwapSeries = chart.addLineSeries({ color: '#a855f7', lineWidth: 1, lineStyle: 2, title: 'RTH VWAP' });
  onHighSeries = chart.addLineSeries({ color: 'rgba(244,63,94,0.45)', lineWidth: 1, lineStyle: 3, title: 'ON High' });
  onLowSeries = chart.addLineSeries({ color: 'rgba(16,185,129,0.45)', lineWidth: 1, lineStyle: 3, title: 'ON Low' });

  // Forward K-bar projection series for the currently inspected signal
  projEntrySeries = chart.addLineSeries({ color: '#38bdf8', lineWidth: 2, lineStyle: 2, title: 'Trigger Entry' });
  projTargetSeries = chart.addLineSeries({ color: '#10b981', lineWidth: 3, title: 'Pred MFE Target' });
  projStopSeries = chart.addLineSeries({ color: '#f43f5e', lineWidth: 3, title: 'Pred MAE Stop' });

  // Clicking ANY candle or marker on the chart inspects that bar and explains why it was or wasn't actionable!
  chart.subscribeClick(async (param) => {
    if (!param || !param.time) return;
    const clickedTs = typeof param.time === 'number' ? param.time : null;
    if (!clickedTs) return;
    await inspectBarByTimestamp(clickedTs);
  });

  window.addEventListener('resize', () => {
    chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
  });
}

async function inspectBarByTimestamp(unixTs, preferredDir = 'auto') {
  const qs = new URLSearchParams({
    tf: state.tf,
    horizon: state.horizon,
    prob_th: state.prob_th,
    ratio_th: state.ratio_th,
    mode: state.mode,
    ts: String(unixTs),
    direction: preferredDir,
  });
  const res = await fetch(`/api/bar?${qs.toString()}`);
  const data = await res.json();
  if (data && data.signal) {
    state.selectedSignal = data.signal;
    state.selectedAltSignal = data.alt_signal || null;
    renderInspector(state.selectedSignal, state.selectedAltSignal);
    if (state.chartData && state.chartData.candles) {
      drawSignalProjectionOnChart(state.selectedSignal, state.chartData.candles);
    }
  }
}

function toggleFullscreenNqChart() {
  const grid = document.getElementById('main-grid');
  const btn = document.getElementById('btn-nq-fullscreen');
  const container = document.getElementById('tv-chart');
  const isFs = grid.classList.toggle('fullscreen-chart');
  btn.textContent = isFs ? '🗗 Restore' : '⛶ Enlarge';
  setTimeout(() => {
    if (chart && container) {
      chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
    }
  }, 50);
}

function fmtUsd(v) {
  const s = Math.abs(v).toLocaleString('en-US', { maximumFractionDigits: 0 });
  return (v >= 0 ? '+$' : '-$') + s;
}

function renderInspector(sig, altSig = null) {
  const el = document.getElementById('inspector-body');
  document.getElementById('insp-horizon-badge').textContent = `Next ${state.horizon} Bars (${state.tf})`;
  if (!sig) {
    el.innerHTML = `<div style="color:var(--muted);padding:20px 0;text-align:center;">No signal selected. Click any row in the Signal Explorer table below or click any bar on the chart.</div>`;
    return;
  }
  const isShort = sig.direction === 'SHORT';
  const netCls = sig.act_net_pts >= 0 ? 'pos' : 'neg';
  const sweepTags = [];
  if (sig.swept_on) sweepTags.push('Overnight H/L Sweep');
  if (sig.swept_prev_rth) sweepTags.push('Prev RTH H/L Sweep');
  const sweepStr = sweepTags.length ? sweepTags.join(' + ') : 'Standard Structural Extreme';

  const isAct = sig.is_actionable !== false;
  const gates = sig.gates || [];
  const gateRowsHtml = gates.map(g => `
    <div class="gate-row">
      <div>
        <span style="font-weight:700;color:${g.passed ? 'var(--green)' : 'var(--amber)'};">${g.passed ? '✅' : '❌'} ${g.name}</span>
        <div style="color:var(--muted);font-size:10.5px;">${g.note || ''}</div>
      </div>
      <div style="text-align:right;white-space:nowrap;">
        <div style="font-weight:800;color:${g.passed ? 'var(--green)' : '#fbbf24'};">${g.value}</div>
        <div style="font-size:10px;color:var(--muted);">Req: ${g.required}</div>
      </div>
    </div>
  `).join('');

  const altBtnHtml = altSig ? `
    <button onclick="inspectBarByTimestamp(${sig.unix_ts}, '${isShort ? 'long' : 'short'}')" style="background:#0b0f17;color:var(--accent);border:1px solid var(--border);border-radius:4px;padding:2px 7px;font-size:10.5px;cursor:pointer;">
      Switch to ${isShort ? 'Pivot Low (Long)' : 'Pivot High (Short)'} (${altSig.prob_pct.toFixed(1)}%)
    </button>
  ` : '';

  el.innerHTML = `
    <div class="sig-banner ${isShort ? 'short' : 'long'}">
      <span>🕒 ${sig.timestamp} ET · ${sig.label}</span>
      <span>Prob: ${sig.prob_pct.toFixed(1)}%</span>
    </div>

    <div class="gate-box ${isAct ? 'pass' : 'fail'}">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
        <strong style="color:${isAct ? 'var(--green)' : '#fbbf24'};font-size:12px;">
          ${isAct ? '✅ ACTIONABLE SIGNAL (All 3 Gates Passed)' : '⛔ NOT ACTIONABLE — Why Filtered Out:'}
        </strong>
        ${altBtnHtml}
      </div>
      <div style="color:var(--text);font-size:11px;margin-bottom:6px;">
        ${sig.non_actionable_reason || ''}
      </div>
      ${gateRowsHtml}
    </div>

    <div style="font-size:11.5px;color:var(--muted);margin-bottom:8px;">
      Session: <strong style="color:var(--text)">${sig.session}</strong> · ${sweepStr}
    </div>
    <div class="detail-grid">
      <div class="detail-box">
        <div class="lbl">Actionable Trigger Px</div>
        <div class="val acc">${sig.trigger_px.toFixed(2)}</div>
      </div>
      <div class="detail-box">
        <div class="lbl">Bar ATR (20)</div>
        <div class="val">${sig.atr_pts.toFixed(1)} pts</div>
      </div>
      <div class="detail-box">
        <div class="lbl">Pred MFE Target (${state.horizon}B)</div>
        <div class="val pos">${sig.target_px.toFixed(2)} (+${sig.pred_mfe_pts.toFixed(1)}p / ${sig.pred_mfe_atr.toFixed(2)} ATR)</div>
      </div>
      <div class="detail-box">
        <div class="lbl">Pred MAE Cushion (${state.horizon}B)</div>
        <div class="val neg">${sig.stop_px.toFixed(2)} (-${sig.pred_mae_pts.toFixed(1)}p / ${sig.pred_mae_atr.toFixed(2)} ATR)</div>
      </div>
      <div class="detail-box">
        <div class="lbl">Predicted MFE / MAE</div>
        <div class="val amb">${sig.pred_ratio.toFixed(2)}x (Asym: ${sig.prob_asym_pct.toFixed(0)}%)</div>
      </div>
      <div class="detail-box">
        <div class="lbl">1m Micro Absorption</div>
        <div class="val">${sig.m1_absorption_pct.toFixed(1)}% (ΔVol ${sig.m1_vol_delta >= 0 ? '+' : ''}${sig.m1_vol_delta.toFixed(1)}%)</div>
      </div>
      <div class="detail-box">
        <div class="lbl">Realized ${state.horizon}B MFE / MAE</div>
        <div class="val ${sig.act_mfe_atr >= sig.act_mae_atr ? 'pos' : 'neg'}">
          +${sig.act_mfe_pts.toFixed(1)}p / -${sig.act_mae_pts.toFixed(1)}p (${sig.act_ratio.toFixed(2)}x)
        </div>
      </div>
      <div class="detail-box">
        <div class="lbl">Realized ${state.horizon}B Net P&L</div>
        <div class="val ${netCls}">${sig.act_net_pts >= 0 ? '+' : ''}${sig.act_net_pts.toFixed(1)} pts (${fmtUsd(sig.act_net_pts * 20)})</div>
      </div>
    </div>
    <div style="font-size:11px;color:var(--muted);border-top:1px solid var(--border);padding-top:6px;">
      True Structural Pivot Outcome: <strong style="color:${sig.is_true_pivot ? 'var(--green)' : 'var(--amber)'}">${sig.is_true_pivot ? 'CONFIRMED SWING PIVOT' : 'MINOR REACTION'}</strong> · TOD Slot RVOL: <strong>${sig.tod_rvol.toFixed(2)}x</strong> · Extreme RSI(14): <strong>${sig.rsi14_ext.toFixed(1)}</strong>
    </div>
  `;
}

function drawSignalProjectionOnChart(sig, candles) {
  if (!sig || !candles || !candles.length) {
    projEntrySeries.setData([]);
    projTargetSeries.setData([]);
    projStopSeries.setData([]);
    return;
  }
  const startTs = sig.unix_ts;
  let idx = candles.findIndex(c => c.time >= startTs);
  if (idx < 0) idx = candles.length - 1;
  const endIdx = Math.min(candles.length - 1, idx + state.horizon);

  const entryPts = [];
  const targetPts = [];
  const stopPts = [];
  for (let i = idx; i <= endIdx; i++) {
    const t = candles[i].time;
    entryPts.push({ time: t, value: sig.trigger_px });
    targetPts.push({ time: t, value: sig.target_px });
    stopPts.push({ time: t, value: sig.stop_px });
  }
  projEntrySeries.setData(entryPts);
  projTargetSeries.setData(targetPts);
  projStopSeries.setData(stopPts);
}

function applyChartZoom() {
  if (!chart || !state.chartData || !state.chartData.candles || !state.chartData.candles.length) return;
  const candles = state.chartData.candles;
  const n = candles.length;
  const mode = state.zoom_mode || 'signal';

  if (mode === 'all') {
    chart.timeScale().fitContent();
    return;
  }
  if (mode === 'signal' && state.selectedSignal) {
    const targetTs = state.selectedSignal.unix_ts;
    let idx = candles.findIndex(c => c.time >= targetTs);
    if (idx < 0) idx = n - 1;
    const span = state.tf === '15m' ? 180 : (state.tf === '1h' ? 140 : 100);
    chart.timeScale().setVisibleLogicalRange({
      from: Math.max(0, idx - Math.round(span * 0.65)),
      to: Math.min(n - 1, idx + Math.round(span * 0.35)),
    });
    return;
  }

  const lastTs = candles[n - 1].time;
  let lookbackSec = 90 * 86400;
  if (mode === '1m') lookbackSec = 30 * 86400;
  else if (mode === '3m') lookbackSec = 90 * 86400;
  else if (mode === '1y') lookbackSec = 365 * 86400;
  else if (mode === 'oos') lookbackSec = 820 * 86400;

  const startTs = lastTs - lookbackSec;
  let fromIdx = candles.findIndex(c => c.time >= startTs);
  if (fromIdx < 0) fromIdx = 0;
  chart.timeScale().setVisibleLogicalRange({ from: fromIdx, to: n - 1 });
}

function setChartZoom(zMode) {
  state.zoom_mode = zMode;
  document.querySelectorAll('#zoom-group button').forEach(b => b.classList.remove('active'));
  const btn = document.getElementById(`zbtn-${zMode}`);
  if (btn) btn.classList.add('active');
  applyChartZoom();
}

async function loadSummary() {
  const res = await fetch('/api/summary');
  state.summary = await res.json();
  renderSummaryTables();
}

function renderSummaryTables() {
  if (!state.summary || !state.summary.timeframes) return;
  const compTbody = document.getElementById('comp-tbody');
  compTbody.innerHTML = '';

  for (const tf of ['15m', '1h', '4h']) {
    const tfData = state.summary.timeframes[tf];
    if (!tfData) continue;
    const eq = tfData.generic_equity_baseline || {};
    const hi = tfData.pivot_high_metrics;
    const lo = tfData.pivot_low_metrics;

    compTbody.innerHTML += `
      <tr>
        <td><strong>${tf}</strong></td>
        <td>${tfData.n_oos_bars.toLocaleString()}</td>
        <td><span class="badge badge-short">Pivot High (Short)</span></td>
        <td>${hi.base_rate_pct.toFixed(1)}%</td>
        <td class="neg">${(eq.eq_high_roc_auc || 0.5).toFixed(4)}</td>
        <td class="pos"><strong>${hi.dual_roc_auc.toFixed(4)}</strong></td>
        <td>${hi.dual_pr_auc.toFixed(4)}</td>
        <td>${hi.thresholds['55'].precision_pct.toFixed(1)}%</td>
        <td class="pos"><strong>${hi.thresholds['60'].precision_pct.toFixed(1)}%</strong></td>
        <td class="pos"><strong>${hi.thresholds['70'].precision_pct.toFixed(1)}%</strong></td>
        <td class="acc"><strong>${hi.thresholds['65'].lift.toFixed(2)}x</strong></td>
      </tr>
      <tr>
        <td><strong>${tf}</strong></td>
        <td>${tfData.n_oos_bars.toLocaleString()}</td>
        <td><span class="badge badge-long">Pivot Low (Long)</span></td>
        <td>${lo.base_rate_pct.toFixed(1)}%</td>
        <td class="neg">${(eq.eq_low_roc_auc || 0.5).toFixed(4)}</td>
        <td class="pos"><strong>${lo.dual_roc_auc.toFixed(4)}</strong></td>
        <td>${lo.dual_pr_auc.toFixed(4)}</td>
        <td>${lo.thresholds['55'].precision_pct.toFixed(1)}%</td>
        <td class="pos"><strong>${lo.thresholds['60'].precision_pct.toFixed(1)}%</strong></td>
        <td class="pos"><strong>${lo.thresholds['70'].precision_pct.toFixed(1)}%</strong></td>
        <td class="acc"><strong>${lo.thresholds['65'].lift.toFixed(2)}x</strong></td>
      </tr>
    `;
  }

  const curTf = state.summary.timeframes[state.tf];
  if (!curTf) return;

  // Update KPI AUC card
  const avgNqAuc = ((curTf.pivot_high_metrics.dual_roc_auc + curTf.pivot_low_metrics.dual_roc_auc) / 2).toFixed(4);
  const eqB = curTf.generic_equity_baseline || {};
  const avgEqAuc = (((eqB.eq_high_roc_auc || 0.5) + (eqB.eq_low_roc_auc || 0.5)) / 2).toFixed(4);
  document.getElementById('kpi-auc').textContent = avgNqAuc;
  document.getElementById('kpi-auc-sub').textContent = `vs Generic Equity: ${avgEqAuc}`;

  // Feature tables
  const fHi = document.getElementById('feat-hi-tbody');
  const fLo = document.getElementById('feat-lo-tbody');
  fHi.innerHTML = (curTf.top_features_high || []).map((f, i) => `<tr><td>#${i+1}</td><td><code>${f.feature}</code></td><td>${f.importance}</td></tr>`).join('');
  fLo.innerHTML = (curTf.top_features_low || []).map((f, i) => `<tr><td>#${i+1}</td><td><code>${f.feature}</code></td><td>${f.importance}</td></tr>`).join('');

  // Quintiles & Tiers for current horizon
  const hData = (curTf.horizons || {})[String(state.horizon)];
  if (hData) {
    const qTbody = document.getElementById('quintiles-tbody');
    qTbody.innerHTML = (hData.quintiles || []).map(q => `
      <tr>
        <td><strong>${q.bucket}</strong></td>
        <td>${q.count.toLocaleString()}</td>
        <td class="acc"><strong>${q.pred_ratio_avg.toFixed(2)}x</strong></td>
        <td>${q.pred_mfe_atr.toFixed(2)} ATR</td>
        <td>${q.pred_mae_atr.toFixed(2)} ATR</td>
        <td class="pos">${q.realized_mfe_atr.toFixed(2)} ATR (${q.realized_mfe_pts.toFixed(1)} pts)</td>
        <td class="neg">${q.realized_mae_atr.toFixed(2)} ATR (${q.realized_mae_pts.toFixed(1)} pts)</td>
        <td class="${q.realized_ratio >= 1.2 ? 'pos' : ''}"><strong>${q.realized_ratio.toFixed(2)}x</strong></td>
        <td class="${q.favorable_win_rate_pct >= 55 ? 'pos' : ''}"><strong>${q.favorable_win_rate_pct.toFixed(1)}%</strong></td>
        <td class="${q.avg_net_pts >= 0 ? 'pos' : 'neg'}"><strong>${q.avg_net_pts >= 0 ? '+' : ''}${q.avg_net_pts.toFixed(1)} pts</strong></td>
      </tr>
    `).join('');

    const tTbody = document.getElementById('tiers-tbody');
    tTbody.innerHTML = (hData.strategy_tiers || []).map(t => `
      <tr>
        <td><strong>${t.tier}</strong></td>
        <td>${t.trades.toLocaleString()}</td>
        <td class="pos"><strong>${t.win_rate_pct.toFixed(1)}%</strong></td>
        <td class="pos">${t.realized_mfe_atr.toFixed(2)} ATR (${t.avg_mfe_pts.toFixed(1)} pts)</td>
        <td class="neg">${t.realized_mae_atr.toFixed(2)} ATR (${t.avg_mae_pts.toFixed(1)} pts)</td>
        <td class="acc"><strong>${t.realized_mfe_mae_ratio.toFixed(2)}x</strong></td>
        <td class="${t.avg_net_pts >= 0 ? 'pos' : 'neg'}"><strong>${t.avg_net_pts >= 0 ? '+' : ''}${t.avg_net_pts.toFixed(1)} pts</strong></td>
        <td class="${t.profit_factor >= 1.2 ? 'pos' : ''}"><strong>${t.profit_factor.toFixed(2)}</strong></td>
        <td class="${t.total_net_pts >= 0 ? 'pos' : 'neg'}"><strong>${t.total_net_pts >= 0 ? '+' : ''}${t.total_net_pts.toLocaleString()} pts</strong></td>
        <td class="${t.total_pnl_1_nq_usd >= 0 ? 'pos' : 'neg'}"><strong>${fmtUsd(t.total_pnl_1_nq_usd)}</strong></td>
        <td class="${t.total_pnl_1_mnq_usd >= 0 ? 'pos' : 'neg'}">${fmtUsd(t.total_pnl_1_mnq_usd)}</td>
      </tr>
    `).join('');
  }
}

async function loadSignalsAndChart(keepFocus = false) {
  if (!keepFocus) state.focus_ts = '';
  const qs = new URLSearchParams({
    tf: state.tf,
    horizon: state.horizon,
    prob_th: state.prob_th,
    ratio_th: state.ratio_th,
    mode: state.mode,
    period: state.period,
    direction: state.direction,
    include_non_actionable: state.show_non_actionable ? '1' : '0',
  });

  const sigRes = await fetch(`/api/signals?${qs.toString()}`);
  const sigData = await sigRes.json();
  state.signals = sigData.signals || [];

  // Update KPI strip
  const k = sigData.kpis || {};
  document.getElementById('kpi-signals').textContent = (k.total_signals || 0).toLocaleString();
  document.getElementById('kpi-piv-rate').textContent = `True Pivot: ${(k.true_pivot_rate_pct || 0).toFixed(1)}% · Roll-Blocked: ${k.roll_blocked_count || 0}`;
  document.getElementById('kpi-win').textContent = `${(k.favorable_win_rate_pct || 0).toFixed(1)}%`;
  document.getElementById('kpi-bwin').textContent = `Bracket Win Rate: ${(k.bracket_win_rate_pct || 0).toFixed(1)}%`;
  document.getElementById('kpi-ratio').textContent = `${(k.realized_mfe_mae_ratio || 0).toFixed(2)}x`;
  document.getElementById('kpi-pred-ratio').textContent = `Avg Pred Ratio: ${(k.avg_pred_ratio || 0).toFixed(2)}x`;
  document.getElementById('kpi-mfe').textContent = `+${(k.avg_mfe_pts || 0).toFixed(1)} pts`;
  document.getElementById('kpi-mfe-atr').textContent = `${(k.realized_mfe_atr || 0).toFixed(2)} ATR`;
  document.getElementById('kpi-mae').textContent = `-${(k.avg_mae_pts || 0).toFixed(1)} pts`;
  document.getElementById('kpi-mae-atr').textContent = `${(k.realized_mae_atr || 0).toFixed(2)} ATR`;

  const netEl = document.getElementById('kpi-net-pts');
  netEl.textContent = `${(k.avg_net_pts || 0) >= 0 ? '+' : ''}${(k.avg_net_pts || 0).toFixed(1)} pts`;
  netEl.className = `kpi-val ${(k.avg_net_pts || 0) >= 0 ? 'pos' : 'neg'}`;
  document.getElementById('kpi-pf').textContent = `Profit Factor: ${(k.profit_factor || 0).toFixed(2)}`;

  const pnlEl = document.getElementById('kpi-pnl-nq');
  pnlEl.textContent = fmtUsd(k.total_pnl_nq_usd || 0);
  pnlEl.className = `kpi-val ${(k.total_pnl_nq_usd || 0) >= 0 ? 'pos' : 'neg'}`;
  document.getElementById('kpi-pnl-mnq').textContent = `1 MNQ ($2/pt): ${fmtUsd(k.total_pnl_mnq_usd || 0)}`;

  renderSignalsTable();
  if (state.signals.length > 0 && !keepFocus) {
    const firstAct = state.signals.find(s => s.is_actionable !== false) || state.signals[0];
    state.selectedSignal = firstAct;
    state.focus_ts = String(firstAct.unix_ts);
  }
  await loadChart();
}

function renderSignalsTable() {
  const tbody = document.getElementById('signals-tbody');
  tbody.innerHTML = state.signals.map((s, idx) => {
    const isShort = s.direction === 'SHORT';
    const isAct = s.is_actionable !== false;
    const badges = [];
    if (s.swept_on) badges.push('<span class="badge badge-sweep">ON Sweep</span>');
    if (s.swept_prev_rth) badges.push('<span class="badge badge-sweep">Prev RTH Sweep</span>');
    badges.push(`<span style="color:var(--muted);font-size:11px;">Abs ${s.m1_absorption_pct.toFixed(0)}% · RVOL ${s.tod_rvol.toFixed(1)}x</span>`);
    const simPts = s.sim_bracket_pts || s.act_net_pts;

    let statusHtml = '';
    if (isAct) {
      statusHtml = `<span class="badge badge-act">✅ ACTIONABLE</span>`;
    } else if (s.status_code === 'BLOCKED_CONTRACT_ROLL') {
      statusHtml = `<span class="badge badge-roll" title="${s.non_actionable_reason}">⛔ NON-ACTIONABLE: ${s.non_actionable_short}</span>`;
    } else {
      statusHtml = `<span class="badge badge-miss" title="${s.non_actionable_reason}">⚠️ NON-ACTIONABLE: ${s.non_actionable_short}</span>`;
    }

    const isSelected = state.selectedSignal && state.selectedSignal.unix_ts === s.unix_ts && state.selectedSignal.direction === s.direction;
    return `
      <tr onclick="selectSignal(${idx})" id="sig-row-${idx}" class="${isSelected ? 'selected-row' : ''} ${!isAct ? 'non-actionable-row' : ''}">
        <td><strong>${s.timestamp}</strong></td>
        <td><span class="badge ${isShort ? 'badge-short' : 'badge-long'}">${s.direction}</span></td>
        <td>${statusHtml}</td>
        <td>${s.session}</td>
        <td><strong>${s.trigger_px.toFixed(2)}</strong></td>
        <td class="${isAct ? 'acc' : 'amb'}"><strong>${s.prob_pct.toFixed(1)}%</strong></td>
        <td>${badges.join(' ')}</td>
        <td class="pos">+${s.pred_mfe_pts.toFixed(1)}p (${s.pred_mfe_atr.toFixed(2)} ATR)</td>
        <td class="neg">-${s.pred_mae_pts.toFixed(1)}p (${s.pred_mae_atr.toFixed(2)} ATR)</td>
        <td class="amb"><strong>${s.pred_ratio.toFixed(2)}x</strong></td>
        <td class="pos">+${s.act_mfe_pts.toFixed(1)}p (${s.act_mfe_atr.toFixed(2)} ATR)</td>
        <td class="neg">-${s.act_mae_pts.toFixed(1)}p (${s.act_mae_atr.toFixed(2)} ATR)</td>
        <td class="${s.act_ratio >= 1.3 ? 'pos' : ''}"><strong>${s.act_ratio.toFixed(2)}x</strong></td>
        <td class="${simPts >= 0 ? 'pos' : 'neg'}"><strong>${simPts >= 0 ? '+' : ''}${simPts.toFixed(1)}p (${fmtUsd(simPts * 20)})</strong></td>
      </tr>
    `;
  }).join('');
}

async function selectSignal(idx) {
  const sig = state.signals[idx];
  if (!sig) return;
  state.selectedSignal = sig;
  state.focus_ts = String(sig.unix_ts);
  document.querySelectorAll('#signals-tbody tr').forEach(tr => tr.classList.remove('selected-row'));
  const rowEl = document.getElementById(`sig-row-${idx}`);
  if (rowEl) rowEl.classList.add('selected-row');

  // All history is already loaded on the chart — update inspector, draw projection, and pan/zoom smoothly!
  renderInspector(state.selectedSignal);
  if (state.chartData && state.chartData.candles) {
    drawSignalProjectionOnChart(state.selectedSignal, state.chartData.candles);
    state.zoom_mode = 'signal';
    document.querySelectorAll('#zoom-group button').forEach(b => b.classList.remove('active'));
    const zBtn = document.getElementById('zbtn-signal');
    if (zBtn) zBtn.classList.add('active');
    applyChartZoom();
  } else {
    await loadChart();
  }
}

async function loadChart() {
  const qs = new URLSearchParams({
    tf: state.tf,
    horizon: state.horizon,
    prob_th: state.prob_th,
    ratio_th: state.ratio_th,
    mode: state.mode,
    bars: 'all',
    show_non_actionable: state.show_non_actionable ? '1' : '0',
  });
  const res = await fetch(`/api/chart?${qs.toString()}`);
  const data = await res.json();
  state.chartData = data;

  candleSeries.setData(data.candles || []);
  ema20Series.setData(data.ema20 || []);
  sma50Series.setData(data.sma50 || []);
  vwapSeries.setData(data.vwap || []);
  onHighSeries.setData(data.on_high || []);
  onLowSeries.setData(data.on_low || []);
  candleSeries.setMarkers(data.markers || []);

  const subEl = document.getElementById('chart-sub');
  if (subEl && data.total_bars) {
    subEl.textContent = `Loaded all ${data.total_bars.toLocaleString()} ${state.tf} bars (${data.first_bar_time} → ${data.last_bar_time} ET) · Click any bar/marker on chart to inspect why it was or wasn't actionable`;
  }

  if (state.selectedSignal) {
    renderInspector(state.selectedSignal);
    drawSignalProjectionOnChart(state.selectedSignal, data.candles || []);
  } else if (data.signals_in_view && data.signals_in_view.length > 0) {
    state.selectedSignal = data.signals_in_view[data.signals_in_view.length - 1];
    renderInspector(state.selectedSignal);
    drawSignalProjectionOnChart(state.selectedSignal, data.candles || []);
  } else {
    renderInspector(null);
    drawSignalProjectionOnChart(null, []);
  }
  applyChartZoom();
}

function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach((b, i) => {
    const map = ['signals', 'comparison', 'calibration'];
    b.classList.toggle('active', map[i] === name);
  });
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.getElementById(`tab-${name}`).classList.add('active');
}

function bindControls() {
  const bindGroup = (id, key) => {
    document.querySelectorAll(`#${id} button`).forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll(`#${id} button`).forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state[key] = key === 'horizon' ? parseInt(btn.dataset.val, 10) : btn.dataset.val;
        renderSummaryTables();
        loadSignalsAndChart(false);
      });
    });
  };
  bindGroup('tf-group', 'tf');
  bindGroup('k-group', 'horizon');
  bindGroup('period-group', 'period');
  bindGroup('mode-group', 'mode');

  document.getElementById('dir-select').addEventListener('change', e => {
    state.direction = e.target.value;
    loadSignalsAndChart(false);
  });

  const pSlider = document.getElementById('prob-slider');
  pSlider.addEventListener('input', e => {
    state.prob_th = parseFloat(e.target.value);
    document.getElementById('prob-val').textContent = `${Math.round(state.prob_th * 100)}%`;
  });
  pSlider.addEventListener('change', () => loadSignalsAndChart(false));

  const rSlider = document.getElementById('ratio-slider');
  rSlider.addEventListener('input', e => {
    state.ratio_th = parseFloat(e.target.value);
    document.getElementById('ratio-val').textContent = `${state.ratio_th.toFixed(2)}x`;
  });
  rSlider.addEventListener('change', () => loadSignalsAndChart(false));

  const chkNonAct = document.getElementById('chk-non-actionable');
  if (chkNonAct) {
    chkNonAct.addEventListener('change', e => {
      state.show_non_actionable = e.target.checked;
      loadSignalsAndChart(true);
    });
  }
}

window.addEventListener('DOMContentLoaded', async () => {
  initChart();
  bindControls();
  await loadSummary();
  await loadSignalsAndChart(false);
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


if __name__ == "__main__":
    print(f"Starting NQ Intraday Pivot & Forward MAE/MFE Explorer on http://127.0.0.1:{PORT}")
    app.run(host="127.0.0.1", port=PORT, debug=False)
