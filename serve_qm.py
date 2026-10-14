#!/usr/bin/env python3
"""serve_qm.py — Historical QM (Qullamaggie Momentum Continuation) Setup & EP Cross-Reference Explorer.

Institutional 3-pane web workspace (running on port 8785) dedicated to pure pattern recognition,
momentum continuation geometry, and Episodic Pivot (EP) catalyst alignment:
- Left Pane: Sortable & Filterable Setup Table (State, Move %, Base length, Pullback, EP Link)
- Center Pane: Interactive LightweightCharts Candlestick Engine with EMA10/20, SMA50/200,
  shaded QM Pole box, shaded QM Base box, Breakout Stack resistance levels, and EP markers
- Right Pane: Comprehensive Setup Dossier (QM Technical Profile, EP Catalyst Linkage, Sector/Theme)
- Zero Simulated Trades: Pure setup visualizer without trade simulations, stops, or R-multiples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

import config
import datastore
import ep_catalyst_enricher
import indicators
import labels
import qm_ml_engine
import scanner_core
import thematic_engine

app = Flask(__name__)

DATA_PATH = Path("data/simulations/qm_historical_setups.parquet")
EP_STUDY_PATH = Path("data/simulations/ep_combined_study_scored.parquet")
STRATEGY_TRADES_PATH = Path("data/simulations/qm_strategy_trades.parquet")
ANNOTATIONS_DIR = Path("data/annotations")
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

DF: Optional[pd.DataFrame] = None
EP_DF: Optional[pd.DataFrame] = None
STRATEGY_DF: Optional[pd.DataFrame] = None
STRATEGY_LOOKUP: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None


def _enrich_with_rs_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """Enriches QM setups with institutional Relative Strength (RS), Stage 2, and 52W High distance."""
    rs_dir = Path("data/rs_history")
    if not rs_dir.exists() or df.empty:
        return df
    rs_files = sorted(rs_dir.glob("rs_*.parquet"))
    if not rs_files:
        return df
    try:
        cols = ["date", "symbol", "rs_score", "stage2", "dist_52w_high"]
        rs_dfs = [pd.read_parquet(p, columns=cols) for p in rs_files]
        all_rs = pd.concat(rs_dfs, ignore_index=True)
        all_rs["date"] = all_rs["date"].dt.strftime("%Y-%m-%d")
        for c in ["rs_score", "stage2", "dist_52w_high"]:
            if c in df.columns:
                df = df.drop(columns=[c])
        enriched = df.merge(all_rs, on=["symbol", "date"], how="left")
        return enriched
    except Exception as e:
        print(f"Warning: Failed to enrich setups with RS metrics: {e}")
        return df


def get_df() -> pd.DataFrame:
    global DF
    if DF is None:
        if DATA_PATH.exists():
            print(f"Loading QM historical setups dataset from {DATA_PATH}...")
            DF = pd.read_parquet(DATA_PATH)
            DF["theme"] = DF["theme"].fillna("General")
            DF["themes"] = DF["themes"].fillna("General")
            DF["sector"] = DF["sector"].fillna("Unknown")
            DF["pattern"] = DF["pattern"].fillna("other")
            DF["ep_badge"] = DF["ep_badge"].fillna("Pure Technical QM")
            if "rs_score" not in DF.columns or "stage2" not in DF.columns or "dist_52w_high" not in DF.columns:
                DF = _enrich_with_rs_technicals(DF)
            if "qm_tier" not in DF.columns:
                s_df = get_strategy_df()
                DF = qm_ml_engine.get_qm_ml_engine().enrich_setups_with_ml(DF, s_df)
            print(f"Loaded {len(DF)} QM historical setups.")
        else:
            print(f"Warning: {DATA_PATH} not found. Running with empty dataset.")
            DF = pd.DataFrame()
    return DF


def get_ep_df() -> pd.DataFrame:
    global EP_DF
    if EP_DF is None:
        if EP_STUDY_PATH.exists():
            EP_DF = pd.read_parquet(EP_STUDY_PATH)
        else:
            EP_DF = pd.DataFrame()
    return EP_DF


def get_strategy_df() -> pd.DataFrame:
    global STRATEGY_DF, STRATEGY_LOOKUP
    if STRATEGY_DF is None:
        if STRATEGY_TRADES_PATH.exists():
            print(f"Loading QM strategy trades dataset from {STRATEGY_TRADES_PATH}...")
            STRATEGY_DF = pd.read_parquet(STRATEGY_TRADES_PATH)
            recs = STRATEGY_DF.to_dict(orient="records")
            STRATEGY_LOOKUP = {(str(r["symbol"]), str(r["date"])): r for r in recs}
            print(f"Loaded {len(STRATEGY_DF)} QM breakout strategy trades into lookup.")
        else:
            print(f"Warning: {STRATEGY_TRADES_PATH} not found.")
            STRATEGY_DF = pd.DataFrame()
            STRATEGY_LOOKUP = {}
    return STRATEGY_DF


@app.route("/api/reload_dataset", methods=["GET", "POST"])
def api_reload_dataset():
    global DF, EP_DF, STRATEGY_DF, STRATEGY_LOOKUP
    DF = None
    EP_DF = None
    STRATEGY_DF = None
    STRATEGY_LOOKUP = None
    df = get_df()
    _ = get_strategy_df()
    return jsonify({
        "status": "ok",
        "rows": len(df),
        "breakouts": int((df["state"] == "breakout").sum()) if not df.empty else 0,
        "building": int((df["state"] == "building").sum()) if not df.empty else 0,
        "strategy_trades": len(STRATEGY_LOOKUP) if STRATEGY_LOOKUP else 0,
    })


@app.route("/api/group_stocks")
def api_group_stocks():
    group = request.args.get("group", "").strip()
    kind = request.args.get("kind", "theme").strip()
    if not group:
        return jsonify({"error": "Missing group"}), 400
    eng = thematic_engine.get_thematic_engine()
    return jsonify(eng.get_group_stocks(group, kind))


def get_thematic_macro_sets():
    eng = thematic_engine.get_thematic_engine()
    if not eng:
        return set(), set(), set()
    summ = eng.get_emerging_summary()
    lead_secs = set([s["name"] for s in summ.get("emerging_sectors", [])])
    lead_thms = set([t["name"] for t in summ.get("all_themes_ranked", []) if t.get("score", 0) >= 65.0])
    emerg_thms = set([
        t["name"] for t in summ.get("emerging_themes", [])
        if "Velocity Surge" in t.get("archetypes", [])
        or "Acceleration Cascade" in t.get("archetypes", [])
        or t.get("badge_type") in ["surge", "acceleration"]
        or t.get("score", 0) >= 70.0
    ])
    return lead_secs, lead_thms, emerg_thms


@app.route("/api/events")
def api_events():
    df = get_df()
    if df.empty:
        default_kpis = {
            "total": 0,
            "breakouts": 0,
            "building": 0,
            "prior_pct": 0.0,
            "post_pct": 0.0,
            "ignited_pct": 0.0,
            "avg_move": 0.0,
        }
        return jsonify({"events": [], "kpis": default_kpis, "sectors": [], "themes": [], "patterns": []})

    q = df.copy()
    _ = get_strategy_df()

    # Strategy Mode Override
    strategy_mode = request.args.get("strategy_mode", "0") in ["1", "true", "True"]
    strategy_kpis = None

    if strategy_mode:
        strat_df = get_strategy_df()
        if not strat_df.empty:
            sq = strat_df.copy()

            # 1. Macro Universe Filter
            strat_macro = request.args.get("strat_macro", "").strip()
            macro_param = request.args.get("macro", "").strip()
            # If toolbar macro chip was clicked and strat_macro is default/leading, respect toolbar macro chip
            if macro_param and macro_param != "all" and (not strat_macro or strat_macro in ["default", "leading"]):
                strat_macro = macro_param
            elif not strat_macro:
                strat_macro = "leading"

            if strat_macro == "leading":
                sq = sq[sq["is_leading_macro"] == True]
            elif strat_macro == "leading_sectors":
                sq = sq[sq["is_leading_sector"] == True]
            elif strat_macro == "leading_themes":
                sq = sq[sq["is_leading_theme"] == True]
            elif strat_macro == "emerging_themes":
                sq = sq[sq["is_emerging_theme"] == True]

            # 2. Market Trend (SPY) Filter
            strat_spy = request.args.get("strat_spy", "spy200").strip()
            if strat_spy == "spy200":
                sq = sq[sq["spy_above_200sma"] == True]
            elif strat_spy == "spy50":
                sq = sq[sq["spy_above_50sma"] == True]

            # 3. RS & Posture Filter
            strat_rs = request.args.get("strat_rs", "rs90_lead").strip()
            if strat_rs == "rs90_lead":
                sq = sq[sq["rs_leader"] == True]
            elif strat_rs == "rs95":
                sq = sq[(sq["rs_score"] >= 95) & ((sq["stage2"] == True) | (sq["dist_52w_high"] >= -15.0))]
            elif strat_rs == "rs80":
                sq = sq[(sq["rs_score"] >= 80) & ((sq["stage2"] == True) | (sq["dist_52w_high"] >= -15.0))]

            # 4. Fresh EP Recency Filter
            strat_fresh_ep = request.args.get("strat_fresh_ep", "all").strip()
            if strat_fresh_ep == "15d":
                sq = sq[sq["has_prior_ep"] & (sq["prior_ep_days_ago"] <= 15)]
            elif strat_fresh_ep == "30d":
                sq = sq[sq["has_prior_ep"] & (sq["prior_ep_days_ago"] <= 30)]
            elif strat_fresh_ep == "has_ep":
                sq = sq[sq["has_prior_ep"] == True]

            # 5. Breakout RVOL Filter
            strat_rvol = request.args.get("strat_rvol", "all").strip()
            if strat_rvol == "1.5x":
                sq = sq[sq["breakout_rvol"] >= 1.5]
            elif strat_rvol == "2.0x":
                sq = sq[sq["breakout_rvol"] >= 2.0]
            elif strat_rvol == "2.5x":
                sq = sq[sq["breakout_rvol"] >= 2.5]

            # 6. QQQ Regime Filter
            strat_qqq = request.args.get("strat_qqq", "all").strip()
            if strat_qqq == "pullback":
                sq = sq[sq["qqq_pullback"] == True]
            elif strat_qqq == "bull_stack":
                sq = sq[sq["qqq_bull_stack"] == True]

            # 7. Tech Leadership Filter
            strat_tech_lead = request.args.get("strat_tech_lead", "all").strip()
            if strat_tech_lead == "bull_stack":
                sq = sq[sq["qqq_spy_ratio_bull"] == True]

            # 8. Bollinger Band Squeeze (Pinch Ratio) Filter (modal or toolbar)
            strat_bb_pinch = request.args.get("strat_bb_pinch", "").strip().lower()
            if not strat_bb_pinch or strat_bb_pinch == "all":
                strat_bb_pinch = request.args.get("bb_pinch", "all").strip().lower()
            if strat_bb_pinch == "tight" and "pinch_ratio" in sq.columns:
                sq = sq[sq["pinch_ratio"] <= 2.5]
            elif strat_bb_pinch == "moderate" and "pinch_ratio" in sq.columns:
                sq = sq[sq["pinch_ratio"] <= 3.2]
            elif strat_bb_pinch == "mild" and "pinch_ratio" in sq.columns:
                sq = sq[sq["pinch_ratio"] <= 4.0]

            # 9. Pole Linearity (R^2) Filter (modal or toolbar)
            strat_r2 = request.args.get("strat_r2", "").strip()
            if not strat_r2 or strat_r2 == "all":
                strat_r2 = request.args.get("r2", "all").strip()
            if strat_r2 == "0.80" and "r2" in sq.columns:
                sq = sq[sq["r2"] >= 0.80]
            elif strat_r2 == "0.65" and "r2" in sq.columns:
                sq = sq[sq["r2"] >= 0.65]
            elif strat_r2 == "0.50" and "r2" in sq.columns:
                sq = sq[sq["r2"] >= 0.50]

            # 10. Toolbar Filters applied to Strategy Trades
            # Pattern
            pat_f = request.args.get("pattern", "all").strip()
            if pat_f != "all" and pat_f and "pattern" in sq.columns:
                sq = sq[sq["pattern"] == pat_f]

            # Min Move %
            min_move = request.args.get("min_move", "").strip()
            if min_move and "move_pct" in sq.columns:
                try:
                    sq = sq[sq["move_pct"] >= float(min_move)]
                except (ValueError, TypeError):
                    pass

            # Consolidation Bars
            cons_f = request.args.get("cons_bars", "all").strip()
            if "cons_bars" in sq.columns:
                if cons_f == "3-7":
                    sq = sq[(sq["cons_bars"] >= 3) & (sq["cons_bars"] <= 7)]
                elif cons_f == "8-15":
                    sq = sq[(sq["cons_bars"] >= 8) & (sq["cons_bars"] <= 15)]
                elif cons_f == "16-25":
                    sq = sq[(sq["cons_bars"] >= 16) & (sq["cons_bars"] <= 25)]
                elif cons_f == "26-45":
                    sq = sq[(sq["cons_bars"] >= 26) & (sq["cons_bars"] <= 45)]

            # Max Pullback %
            max_pb = request.args.get("max_pullback", "").strip()
            if max_pb and "pullback_pct" in sq.columns:
                try:
                    sq = sq[sq["pullback_pct"] <= float(max_pb)]
                except (ValueError, TypeError):
                    pass

            # Sector & Theme
            sec_f = request.args.get("sector", "all").strip()
            if sec_f == "leading_sectors":
                sq = sq[sq["is_leading_sector"] == True]
            elif sec_f != "all" and sec_f and "sector" in sq.columns:
                sq = sq[(sq["sector"] == sec_f) | (sq.get("theme", pd.Series(dtype=object)) == sec_f) | sq.get("themes", pd.Series(dtype=object)).astype(str).str.contains(sec_f, case=False, na=False, regex=False)]

            thm_f = request.args.get("theme", "all").strip()
            if thm_f == "leading_themes":
                sq = sq[sq["is_leading_theme"] == True]
            elif thm_f == "emerging_themes":
                sq = sq[sq["is_emerging_theme"] == True]
            elif thm_f != "all" and thm_f:
                sq = sq[sq.get("themes", pd.Series(dtype=object)).astype(str).str.contains(thm_f, case=False, na=False, regex=False) | (sq.get("theme", pd.Series(dtype=object)) == thm_f) | (sq.get("sector", pd.Series(dtype=object)) == thm_f)]

            # Symbol Search
            search_q = request.args.get("search", "").strip().upper()
            if search_q and "symbol" in sq.columns:
                sq = sq[sq["symbol"].str.contains(search_q, case=False, na=False)]

            # 10. Setup Quality & Machine Learning Tier Filter
            strat_tier = request.args.get("strat_tier", "").strip().lower()
            if not strat_tier or strat_tier == "all":
                strat_tier = request.args.get("tier", "all").strip().lower()
            if "qm_tier" in sq.columns:
                if strat_tier == "premier":
                    sq = sq[sq["qm_tier"] == "Premier"]
                elif strat_tier == "high_quality":
                    sq = sq[sq["qm_tier"].isin(["Premier", "High Quality"])]
                elif strat_tier == "avoid_traps":
                    sq = sq[sq["qm_tier"] != "Toxic / Trap"]

            # 11. Sizing, Stoploss, and Trailing Exit Models for Stats
            strat_stop = request.args.get("strat_stop", "06x").strip()
            strat_sizing = request.args.get("strat_sizing", "dynamic").strip()
            strat_exit = request.args.get("strat_exit", "10ema").strip().lower()

            if strat_exit == "triad_optimal":
                r_key = "r_triad_optimal"
                sized_r_key = "sized_r_triad_optimal"
                win_key = "is_win_triad_optimal"
                bars_key = "bars_triad_optimal"
            elif strat_stop == "dynamic":
                if strat_exit in ["climax_swing", "climax_hybrid"]:
                    r_key = "r_dyn_climax_swing"
                    sized_r_key = "sized_r_dyn_climax_swing"
                    win_key = "is_win_dyn_climax_swing"
                    bars_key = "bars_dyn_climax_swing"
                else:
                    r_key = "r_dyn_06x"
                    sized_r_key = "sized_r_dyn_06x"
                    win_key = "is_win_dyn_06x"
                    bars_key = "bars_dyn_06x"
            else:
                exit_map = {
                    "10ema": "e10",
                    "20ema": "e20",
                    "50ema": "e50",
                    "staged": "staged",
                    "balanced": "balanced",
                    "climax_swing": "climax_swing",
                    "climax_hybrid": "climax_hybrid",
                    "climax_trend": "climax_trend",
                }
                exit_pfx = exit_map.get(strat_exit, "e10")
                stop_sfx = "06x" if strat_stop == "06x" else "10x"

                r_key = f"r_{exit_pfx}_{stop_sfx}"
                sized_r_key = f"sized_r_{exit_pfx}_{stop_sfx}"
                win_key = f"is_win_{exit_pfx}_{stop_sfx}"
                bars_key = f"bars_{exit_pfx}_{stop_sfx}"

            if r_key not in sq.columns:
                r_key = f"r_{stop_sfx}" if f"r_{stop_sfx}" in sq.columns else "r_06x"
            if sized_r_key not in sq.columns:
                sized_r_key = f"sized_r_{stop_sfx}" if f"sized_r_{stop_sfx}" in sq.columns else "sized_r_06x"
            if win_key not in sq.columns:
                win_key = f"is_win_{stop_sfx}" if f"is_win_{stop_sfx}" in sq.columns else "is_win_06x"
            if bars_key not in sq.columns:
                bars_key = f"bars_{stop_sfx}" if f"bars_{stop_sfx}" in sq.columns else "bars_06x"

            eff_r = sq[sized_r_key] if strat_sizing == "dynamic" else sq[r_key]
            n_trades = len(sq)
            wins = int(sq[win_key].sum()) if n_trades > 0 else 0
            losses = n_trades - wins
            win_rate = (wins / n_trades * 100.0) if n_trades > 0 else 0.0

            gross_win = float(eff_r[eff_r > 0].sum()) if n_trades > 0 else 0.0
            gross_loss = float(abs(eff_r[eff_r <= 0].sum())) if n_trades > 0 else 0.0
            pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)
            exp_r = round(float(eff_r.mean()), 3) if n_trades > 0 else 0.0
            tot_r = round(float(sq[r_key].sum()), 1) if n_trades > 0 else 0.0
            sized_tot_r = round(float(sq[sized_r_key].sum()), 1) if n_trades > 0 else 0.0

            avg_win = round(float(eff_r[eff_r > 0].mean()), 2) if wins > 0 else 0.0
            avg_loss = round(float(eff_r[eff_r <= 0].mean()), 2) if losses > 0 else 0.0
            wl_ratio = round(avg_win / abs(avg_loss), 2) if avg_loss != 0 else 0.0
            max_win = round(float(eff_r.max()), 1) if n_trades > 0 else 0.0
            max_loss = round(float(eff_r.min()), 1) if n_trades > 0 else 0.0
            avg_bars = round(float(sq[bars_key].mean()), 1) if n_trades > 0 else 0.0

            strategy_kpis = {
                "total_trades": n_trades,
                "win_rate": round(win_rate, 1),
                "profit_factor": pf,
                "expectancy_r": exp_r,
                "total_r": tot_r,
                "sized_total_r": sized_tot_r,
                "avg_win_r": avg_win,
                "avg_loss_r": avg_loss,
                "win_loss_ratio": wl_ratio,
                "max_win_r": max_win,
                "max_loss_r": max_loss,
                "avg_bars": avg_bars,
                "strat_stop": strat_stop,
                "strat_sizing": strat_sizing,
                "strat_exit": strat_exit,
                "strat_tier": strat_tier,
                "strat_macro": strat_macro,
                "strat_spy": strat_spy,
                "strat_rs": strat_rs,
                "strat_bb_pinch": strat_bb_pinch,
                "strat_r2": strat_r2,
            }

            # Filter events to only the qualifying breakout trades
            valid_keys = set(zip(sq["symbol"].astype(str), sq["date"].astype(str)))
            q = q[(q["state"] == "breakout") & q.apply(lambda r: (str(r["symbol"]), str(r["date"])) in valid_keys, axis=1)]

    lead_secs, lead_thms, emerg_thms = get_thematic_macro_sets()

    # Filters applied only in standard Setup Mode (in Strategy Mode, all filtering is already executed on sq)
    if not strategy_mode:
        # 1. State Filter
        state_f = request.args.get("state", "all")
        if state_f in ["breakout", "building"]:
            q = q[q["state"] == state_f]

        # 1b. ML Setup Quality Tier Filter (Applies to all setups & fresh building setups)
        strat_tier = request.args.get("strat_tier", "").strip().lower()
        if not strat_tier or strat_tier == "all":
            strat_tier = request.args.get("tier", "all").strip().lower()
        if strat_tier and strat_tier != "all" and "qm_tier" in q.columns:
            if strat_tier == "premier":
                q = q[q["qm_tier"] == "Premier"]
            elif strat_tier == "high_quality":
                q = q[q["qm_tier"].isin(["Premier", "High Quality"])]
            elif strat_tier == "avoid_traps":
                q = q[q["qm_tier"] != "Toxic / Trap"]

        # 2. EP Catalyst Link Filter
        ep_link_f = request.args.get("ep_link", "all")
        if ep_link_f == "prior":
            q = q[q["has_prior_ep"]]
        elif ep_link_f == "post":
            q = q[q["has_post_ep"]]
        elif ep_link_f == "ignited":
            q = q[q["ep_badge"].str.contains("Ignited", case=False, na=False)]
        elif ep_link_f == "boosted":
            q = q[q["ep_badge"].str.contains("Boosted", case=False, na=False)]
        elif ep_link_f == "both":
            q = q[q["has_prior_ep"] & q["has_post_ep"]]
        elif ep_link_f == "none":
            q = q[~q["has_prior_ep"] & ~q["has_post_ep"]]

        # 3. Pattern Filter
        pat_f = request.args.get("pattern", "all")
        if pat_f != "all" and pat_f:
            q = q[q["pattern"] == pat_f]

        # 4. Move % Filter
        min_move = request.args.get("min_move", "")
        if min_move:
            try:
                val = float(min_move)
                q = q[q["move_pct"] >= val]
            except (ValueError, TypeError):
                pass

        # 5. Consolidation Bars Filter
        cons_f = request.args.get("cons_bars", "all")
        if cons_f == "3-7":
            q = q[(q["cons_bars"] >= 3) & (q["cons_bars"] <= 7)]
        elif cons_f == "8-15":
            q = q[(q["cons_bars"] >= 8) & (q["cons_bars"] <= 15)]
        elif cons_f == "16-25":
            q = q[(q["cons_bars"] >= 16) & (q["cons_bars"] <= 25)]
        elif cons_f == "26-45":
            q = q[(q["cons_bars"] >= 26) & (q["cons_bars"] <= 45)]

        # 6. Pullback % Filter
        max_pb = request.args.get("max_pullback", "")
        if max_pb:
            try:
                val = float(max_pb)
                q = q[q["pullback_pct"] <= val]
            except (ValueError, TypeError):
                pass

        # 6b. Bollinger Band Squeeze (Pinch Ratio) Filter
        bb_pinch_f = request.args.get("bb_pinch", "all").strip().lower()
        if bb_pinch_f in ["tight", "moderate", "mild"] and "pinch_ratio" in q.columns:
            if bb_pinch_f == "tight":
                q = q[q["pinch_ratio"].notna() & (q["pinch_ratio"] <= 2.5)]
            elif bb_pinch_f == "moderate":
                q = q[q["pinch_ratio"].notna() & (q["pinch_ratio"] <= 3.2)]
            elif bb_pinch_f == "mild":
                q = q[q["pinch_ratio"].notna() & (q["pinch_ratio"] <= 4.0)]

        # 6c. Pole Linear Regression R2 Filter
        r2_f = request.args.get("r2", "all").strip()
        if r2_f not in ["all", ""] and "r2" in q.columns:
            try:
                min_r2 = float(r2_f)
                q = q[q["r2"].notna() & (q["r2"] >= min_r2)]
            except (ValueError, TypeError):
                pass

        # 7. Thematic / Macro Flow Filter (Leading Sectors, Leading Themes, Emerging Themes)
        macro_f = request.args.get("macro", "all")
        if macro_f == "leading_sectors":
            q = q[q["sector"].isin(lead_secs) | q["theme"].isin(lead_secs) | q["themes"].apply(lambda x: any(s in str(x) for s in lead_secs) if pd.notna(x) else False)]
        elif macro_f == "leading_themes":
            q = q[q["theme"].isin(lead_thms) | q["themes"].apply(lambda x: any(t in str(x) for t in lead_thms) if pd.notna(x) else False)]
        elif macro_f == "emerging_themes":
            q = q[q["theme"].isin(emerg_thms) | q["themes"].apply(lambda x: any(t in str(x) for t in emerg_thms) if pd.notna(x) else False)]

        # 8. Sector & Theme Filters
        sec_f = request.args.get("sector", "all")
        if sec_f == "leading_sectors":
            q = q[q["sector"].isin(lead_secs) | q["theme"].isin(lead_secs) | q["themes"].apply(lambda x: any(s in str(x) for s in lead_secs) if pd.notna(x) else False)]
        elif sec_f != "all" and sec_f:
            q = q[(q["sector"] == sec_f) | (q["theme"] == sec_f) | q["themes"].astype(str).str.contains(sec_f, case=False, na=False, regex=False)]

        thm_f = request.args.get("theme", "all")
        if thm_f == "leading_themes":
            q = q[q["theme"].isin(lead_thms) | q["themes"].apply(lambda x: any(t in str(x) for t in lead_thms) if pd.notna(x) else False)]
        elif thm_f == "emerging_themes":
            q = q[q["theme"].isin(emerg_thms) | q["themes"].apply(lambda x: any(t in str(x) for t in emerg_thms) if pd.notna(x) else False)]
        elif thm_f != "all" and thm_f:
            q = q[q["themes"].astype(str).str.contains(thm_f, case=False, na=False, regex=False) | (q["theme"] == thm_f) | (q["sector"] == thm_f)]

        # 9. Search Symbol
        search_q = request.args.get("search", "").strip().upper()
        if search_q:
            q = q[q["symbol"].str.contains(search_q, case=False, na=False)]

        # Relative Strength (RS) Score & Posture Filter
        # Default: RS >= 90 for stage 2 bullish (stage2 == True) or within 15% 52w high (dist_52w_high >= -15.0)
        rs_f = request.args.get("rs_filter", "rs90_stage2_or_15").strip()
        if search_q and rs_f in ["rs90_stage2_or_15", "default"]:
            pass
        elif rs_f in ["rs90_stage2_or_15", "default"]:
            q = q[(q["rs_score"] >= 90) & ((q["stage2"] == True) | (q["dist_52w_high"] >= -15.0))]
        elif rs_f == "rs95_stage2_or_15":
            q = q[(q["rs_score"] >= 95) & ((q["stage2"] == True) | (q["dist_52w_high"] >= -15.0))]
        elif rs_f == "rs80_stage2_or_15":
            q = q[(q["rs_score"] >= 80) & ((q["stage2"] == True) | (q["dist_52w_high"] >= -15.0))]
        elif rs_f == "rs70_stage2_or_15":
            q = q[(q["rs_score"] >= 70) & ((q["stage2"] == True) | (q["dist_52w_high"] >= -15.0))]
        elif rs_f == "rs90_any":
            q = q[q["rs_score"] >= 90]
        elif rs_f == "rs80_any":
            q = q[q["rs_score"] >= 80]
        elif rs_f == "stage2_only":
            q = q[q["stage2"] == True]
        elif rs_f == "near_ath_only":
            q = q[q["dist_52w_high"] >= -15.0]
        elif rs_f == "all":
            pass

        min_rs = request.args.get("min_rs", "").strip()
        if min_rs:
            try:
                val_rs = float(min_rs)
                req_posture = request.args.get("rs_posture", "1") in ["1", "true", "True"]
                if req_posture:
                    q = q[(q["rs_score"] >= val_rs) & ((q["stage2"] == True) | (q["dist_52w_high"] >= -15.0))]
                else:
                    q = q[q["rs_score"] >= val_rs]
            except (ValueError, TypeError):
                pass

    # Sort default by date descending
    sort_col = request.args.get("sort", "date")
    sort_asc = request.args.get("asc", "0") == "1"
    if sort_col in q.columns:
        q = q.sort_values(sort_col, ascending=sort_asc)
    else:
        q = q.sort_values("date", ascending=False)

    total_count = len(q)
    bks = int((q["state"] == "breakout").sum())
    bld = int((q["state"] == "building").sum())
    prior_cnt = int(q["has_prior_ep"].sum())
    post_cnt = int(q["has_post_ep"].sum())
    ignited_cnt = int(q["ep_badge"].str.contains("Ignited", case=False, na=False).sum())
    avg_move = round(float(q["move_pct"].mean()), 1) if not q.empty else 0.0

    lead_sec_cnt = int((q["sector"].isin(lead_secs) | q["theme"].isin(lead_secs) | q["themes"].apply(lambda x: any(s in str(x) for s in lead_secs) if pd.notna(x) else False)).sum())
    lead_thm_cnt = int((q["theme"].isin(lead_thms) | q["themes"].apply(lambda x: any(t in str(x) for t in lead_thms) if pd.notna(x) else False)).sum())
    emerg_thm_cnt = int((q["theme"].isin(emerg_thms) | q["themes"].apply(lambda x: any(t in str(x) for t in emerg_thms) if pd.notna(x) else False)).sum())
    rs_lead_cnt = int(((q["rs_score"] >= 90) & ((q["stage2"] == True) | (q["dist_52w_high"] >= -15.0))).sum()) if not q.empty else 0

    kpis = {
        "total": total_count,
        "breakouts": bks,
        "building": bld,
        "prior_pct": round((prior_cnt / total_count * 100), 1) if total_count > 0 else 0.0,
        "post_pct": round((post_cnt / total_count * 100), 1) if total_count > 0 else 0.0,
        "ignited_pct": round((ignited_cnt / total_count * 100), 1) if total_count > 0 else 0.0,
        "avg_move": avg_move,
        "leading_sectors_count": lead_sec_cnt,
        "leading_themes_count": lead_thm_cnt,
        "emerging_themes_count": emerg_thm_cnt,
        "thematic_lead_pct": round((max(lead_thm_cnt, lead_sec_cnt) / total_count * 100), 1) if total_count > 0 else 0.0,
        "rs_leader_pct": round((rs_lead_cnt / total_count * 100), 1) if total_count > 0 else 0.0,
    }

    # Distinct categories for filter dropdowns
    all_sectors = sorted([s for s in df["sector"].dropna().unique() if s and s != "Unknown"])
    all_themes = sorted([t for t in df["theme"].dropna().unique() if t and t != "General"])
    all_patterns = sorted([p for p in df["pattern"].dropna().unique() if p])

    # Convert records for JSON
    events_list = []
    # Cap to 2,000 for instant UI rendering
    for _, row in q.head(2000).iterrows():
        st_levels = row.get("stack_levels")
        if isinstance(st_levels, np.ndarray):
            st_levels = st_levels.tolist()
        elif not isinstance(st_levels, (list, tuple)):
            st_levels = []

        sec_val = str(row.get("sector", ""))
        thm_val = str(row.get("theme", ""))
        thms_val = str(row.get("themes", ""))
        is_lead_sec = (sec_val in lead_secs) or (thm_val in lead_secs) or any(s in thms_val for s in lead_secs)
        is_lead_thm = (thm_val in lead_thms) or any(t in thms_val for t in lead_thms)
        is_emerg_thm = (thm_val in emerg_thms) or any(t in thms_val for t in emerg_thms)

        sym_str = str(row["symbol"])
        date_str = str(row["date"])
        strat_trade = STRATEGY_LOOKUP.get((sym_str, date_str)) if STRATEGY_LOOKUP else None

        qm_tier = None
        qm_score = None
        p_big_move = None
        p_trap = None
        if strat_trade:
            if "qm_tier" in strat_trade and pd.notna(strat_trade["qm_tier"]):
                qm_tier = str(strat_trade["qm_tier"])
            if "qm_score" in strat_trade and pd.notna(strat_trade["qm_score"]):
                qm_score = round(float(strat_trade["qm_score"]), 1)
            if "p_big_move" in strat_trade and pd.notna(strat_trade["p_big_move"]):
                p_big_move = round(float(strat_trade["p_big_move"]), 3)
            if "p_trap" in strat_trade and pd.notna(strat_trade["p_trap"]):
                p_trap = round(float(strat_trade["p_trap"]), 3)
        elif "qm_tier" in row and pd.notna(row.get("qm_tier")):
            qm_tier = str(row["qm_tier"])
            if "qm_score" in row and pd.notna(row.get("qm_score")):
                qm_score = round(float(row["qm_score"]), 1)
            if "p_big_move" in row and pd.notna(row.get("p_big_move")):
                p_big_move = round(float(row["p_big_move"]), 3)
            if "p_trap" in row and pd.notna(row.get("p_trap")):
                p_trap = round(float(row["p_trap"]), 3)

        rec = {
            "symbol": sym_str,
            "date": date_str,
            "state": str(row["state"]),
            "rs_score": int(row["rs_score"]) if pd.notna(row.get("rs_score")) else None,
            "stage2": bool(row.get("stage2", False)) if pd.notna(row.get("stage2")) else False,
            "dist_52w_high": round(float(row["dist_52w_high"]), 1) if pd.notna(row.get("dist_52w_high")) else None,
            "move_pct": round(float(row["move_pct"]), 1),
            "cons_bars": int(row["cons_bars"]),
            "pullback_pct": round(float(row["pullback_pct"]), 1),
            "r2": round(float(row["r2"]), 3) if ("r2" in row and pd.notna(row.get("r2"))) else None,
            "pinch_ratio": round(float(row["pinch_ratio"]), 2) if ("pinch_ratio" in row and pd.notna(row.get("pinch_ratio"))) else None,
            "ride_frac": round(float(row["ride_frac"]), 2) if pd.notna(row.get("ride_frac")) else 0.0,
            "shakeouts": int(row["shakeouts"]) if pd.notna(row.get("shakeouts")) else 0,
            "vol_x": round(float(row["vol_x"]), 2) if pd.notna(row.get("vol_x")) else None,
            "pattern": str(row["pattern"]),
            "level": round(float(row["level"]), 2),
            "trigger": round(float(row["trigger"]), 2),
            "dist_to_trigger_pct": round(float(row["dist_to_trigger_pct"]), 2),
            "qm_pole_start": str(row.get("qm_pole_start", "")),
            "qm_pole_end": str(row.get("qm_pole_end", "")),
            "qm_cons_start": str(row.get("qm_cons_start", "")),
            "qm_cons_end": str(row.get("qm_cons_end", "")),
            "stack_levels": [round(float(x), 2) for x in st_levels],
            "sector": sec_val,
            "theme": thm_val,
            "is_leading_sector": is_lead_sec,
            "is_leading_theme": is_lead_thm,
            "is_emerging_theme": is_emerg_thm,
            "has_prior_ep": bool(row.get("has_prior_ep", False)),
            "prior_ep_date": str(row["prior_ep_date"]) if pd.notna(row.get("prior_ep_date")) and row.get("prior_ep_date") else None,
            "prior_ep_days_ago": int(row["prior_ep_days_ago"]) if pd.notna(row.get("prior_ep_days_ago")) and row.get("prior_ep_days_ago") is not None else None,
            "prior_ep_gap_pct": round(float(row["prior_ep_gap_pct"]), 1) if pd.notna(row.get("prior_ep_gap_pct")) and row.get("prior_ep_gap_pct") is not None else None,
            "prior_ep_rvol": round(float(row["prior_ep_rvol"]), 1) if pd.notna(row.get("prior_ep_rvol")) and row.get("prior_ep_rvol") is not None else None,
            "prior_ep_subtype": str(row.get("prior_ep_subtype", "")) if pd.notna(row.get("prior_ep_subtype")) else None,
            "prior_ep_catalyst": str(row.get("prior_ep_catalyst", "")) if pd.notna(row.get("prior_ep_catalyst")) else "",
            "has_post_ep": bool(row.get("has_post_ep", False)),
            "post_ep_date": str(row["post_ep_date"]) if pd.notna(row.get("post_ep_date")) and row.get("post_ep_date") else None,
            "post_ep_days_later": int(row["post_ep_days_later"]) if pd.notna(row.get("post_ep_days_later")) and row.get("post_ep_days_later") is not None else None,
            "post_ep_gap_pct": round(float(row["post_ep_gap_pct"]), 1) if pd.notna(row.get("post_ep_gap_pct")) and row.get("post_ep_gap_pct") is not None else None,
            "post_ep_rvol": round(float(row["post_ep_rvol"]), 1) if pd.notna(row.get("post_ep_rvol")) and row.get("post_ep_rvol") is not None else None,
            "post_ep_subtype": str(row.get("post_ep_subtype", "")) if pd.notna(row.get("post_ep_subtype")) else None,
            "post_ep_catalyst": str(row.get("post_ep_catalyst", "")) if pd.notna(row.get("post_ep_catalyst")) else "",
            "ep_badge": str(row.get("ep_badge", "Pure Technical QM")),
            "qm_tier": qm_tier,
            "qm_score": qm_score,
            "p_big_move": p_big_move,
            "p_trap": p_trap,
        }
        events_list.append(rec)

    return jsonify({
        "events": events_list,
        "kpis": kpis,
        "strategy_kpis": strategy_kpis,
        "strategy_mode": strategy_mode,
        "sectors": all_sectors,
        "themes": all_themes,
        "patterns": all_patterns,
        "leading_sectors": sorted(list(lead_secs)),
        "leading_themes": sorted(list(lead_thms)),
        "emerging_themes": sorted(list(emerg_thms)),
    })


@app.route("/api/annotations", methods=["GET", "POST"])
def api_annotations():
    if request.method == "GET":
        sym = request.args.get("symbol", "").upper().strip()
        if not sym:
            return jsonify({"symbol": "", "drawings": []})
        file_path = ANNOTATIONS_DIR / f"{sym}.json"
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return jsonify(data)
            except Exception as e:
                return jsonify({"symbol": sym, "drawings": [], "error": str(e)})
        return jsonify({"symbol": sym, "drawings": []})

    data = request.get_json(silent=True) or {}
    sym = data.get("symbol", "").upper().strip()
    drawings = data.get("drawings", [])
    if not sym:
        return jsonify({"ok": False, "error": "Missing symbol"}), 400

    file_path = ANNOTATIONS_DIR / f"{sym}.json"
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"symbol": sym, "drawings": drawings}, f, indent=2)
        return jsonify({"ok": True, "count": len(drawings)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/chart")
def api_chart():
    sym = request.args.get("symbol", "").strip().upper()
    date_str = request.args.get("date", "").strip()

    if not sym:
        return jsonify({"error": "Missing symbol"}), 400

    df_full = datastore.load_bars(sym)
    if df_full is None or df_full.empty:
        return jsonify({"error": f"No bar data for {sym}"}), 404

    df = indicators.add_indicators(df_full)

    # Moving averages
    ema10_s = df["ema10"]
    ema20_s = df["ema20"]
    sma50_s = df["sma50"]
    sma200_s = df["sma200"] if "sma200" in df.columns else df["close"].rolling(200).mean()

    # Larsson Line Ribbon calculation
    try:
        larsson_fastest = int(request.args.get("larsson_fastest", "32"))
    except (ValueError, TypeError):
        larsson_fastest = 32
    larsson_fastest = max(2, min(100, larsson_fastest))

    if larsson_fastest == 32:
        larsson_spans = (32, 35, 50, 58)
    elif larsson_fastest <= 20:
        s1 = larsson_fastest
        s2 = int(round(larsson_fastest * 1.5))
        s3 = int(round(larsson_fastest * 2.0))
        s4 = int(round(larsson_fastest * 2.625))
        larsson_spans = (s1, s2, s3, s4)
    else:
        s1 = larsson_fastest
        s2 = int(round(larsson_fastest * 35.0 / 32.0))
        s3 = int(round(larsson_fastest * 50.0 / 32.0))
        s4 = int(round(larsson_fastest * 58.0 / 32.0))
        larsson_spans = (s1, s2, s3, s4)

    s1, s2, s3, s4 = larsson_spans
    e1 = scanner_core._pine_ema(df['close'], s1)
    e2 = scanner_core._pine_ema(df['close'], s2)
    e3 = scanner_core._pine_ema(df['close'], s3)
    e4 = scanner_core._pine_ema(df['close'], s4)

    bullish = (e1 >= e2) & (e2 >= e3) & (e3 >= e4)
    bearish = (e1 < e2) & (e2 < e3) & (e3 < e4)
    valid_l = e1.notna() & e2.notna() & e3.notna() & e4.notna()

    l_state = pd.Series('gray', index=df.index)
    l_state[valid_l & bullish] = 'yellow'
    l_state[valid_l & bearish] = 'blue'

    vol_sma20 = df["volume"].rolling(20).mean()

    # Determine event row
    dataset = get_df()
    setup_row = None
    if not dataset.empty:
        matches = dataset[(dataset["symbol"] == sym) & (dataset["date"] == date_str)]
        if not matches.empty:
            setup_row = matches.iloc[0].to_dict()

    # Event date index in dataframe
    try:
        if date_str:
            target_dt = pd.to_datetime(date_str)
            pos_matches = np.where(df.index == target_dt)[0]
            pos = int(pos_matches[0]) if len(pos_matches) > 0 else (len(df) - 1)
        else:
            pos = len(df) - 1
            date_str = df.index[pos].strftime("%Y-%m-%d")
    except Exception:
        pos = len(df) - 1
        date_str = df.index[pos].strftime("%Y-%m-%d")

    # Return full historical bars as requested
    sub = df

    bars = []
    volume = []
    vol_ma20_data = []
    ema10_data = []
    ema20_data = []
    sma50_data = []
    sma200_data = []
    larsson_data = []

    for i, (idx, r) in enumerate(sub.iterrows()):
        t = idx.strftime("%Y-%m-%d")
        o, h, l, c = float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])
        v = float(r["volume"]) if pd.notna(r["volume"]) else 0
        bars.append({"time": t, "open": o, "high": h, "low": l, "close": c})
        volume.append({
            "time": t,
            "value": v,
            "color": "rgba(16,185,129,0.55)" if c >= o else "rgba(239,68,68,0.55)",
        })

        if pd.notna(vol_sma20.iloc[i]):
            vol_ma20_data.append({"time": t, "value": round(float(vol_sma20.iloc[i]), 0)})
        if pd.notna(ema10_s.loc[idx]):
            ema10_data.append({"time": t, "value": round(float(ema10_s.loc[idx]), 2)})
        if pd.notna(ema20_s.loc[idx]):
            ema20_data.append({"time": t, "value": round(float(ema20_s.loc[idx]), 2)})
        if pd.notna(sma50_s.loc[idx]):
            sma50_data.append({"time": t, "value": round(float(sma50_s.loc[idx]), 2)})
        if pd.notna(sma200_s.loc[idx]):
            sma200_data.append({"time": t, "value": round(float(sma200_s.loc[idx]), 2)})

        f_val = round(float(e1.iloc[i]), 2) if pd.notna(e1.iloc[i]) else None
        s_val = round(float(e4.iloc[i]), 2) if pd.notna(e4.iloc[i]) else None
        larsson_data.append({
            "time": t,
            "fast": f_val,
            "slow": s_val,
            "value": f_val,
            "state": str(l_state.iloc[i]),
        })

    # Fetch all historical EPs for this symbol for markers
    ep_df = get_ep_df()
    symbol_eps = []
    if not ep_df.empty:
        sym_matches = ep_df[ep_df["symbol"] == sym].sort_values("date")
        for _, ep_row in sym_matches.iterrows():
            symbol_eps.append({
                "date": str(ep_row["date"]),
                "gap_pct": round(float(ep_row.get("gap_pct", 0.0)), 1),
                "rvol": round(float(ep_row.get("rvol", 0.0)), 1),
                "close_pos": round(float(ep_row.get("close_pos", 0.5)), 2),
                "subtype": str(ep_row.get("subtype", "classic")),
            })

    # Sector & Theme Momentum Backdrop evaluation
    sec_name = str(setup_row.get("sector") if setup_row and pd.notna(setup_row.get("sector")) else (labels.sector(sym) or "Unknown"))
    thm_list = labels.themes(sym) or []
    primary_thm = str(setup_row.get("theme") if setup_row and pd.notna(setup_row.get("theme")) else (thm_list[0] if thm_list else "General"))
    themes_str = str(setup_row.get("themes") if setup_row and pd.notna(setup_row.get("themes")) else (", ".join(thm_list) if thm_list else primary_thm))

    eng = thematic_engine.get_thematic_engine()
    thm_eval = eng.evaluate_symbol(sym) if eng else {}
    sec_info = thm_eval.get("sector_info", {})
    thm_info = thm_eval.get("theme_info", {})
    sec_m1 = float(sec_info.get("m1_pct")) if pd.notna(sec_info.get("m1_pct")) else None
    sec_m3 = float(sec_info.get("m3_pct")) if pd.notna(sec_info.get("m3_pct")) else None
    thm_m1 = float(thm_info.get("m1_pct")) if pd.notna(thm_info.get("m1_pct")) else None

    # Macro diagnosis
    if (thm_m1 is not None and thm_m1 >= 70) or (sec_m1 is not None and sec_m1 >= 75):
        macro_diag = "Strong Institutional Tailwind"
        macro_desc = "Top decile money flow sponsorship. High probability of multi-quarter momentum continuation."
        macro_status = "tailwind"
    elif (thm_m1 is not None and thm_m1 <= 30) or (sec_m1 is not None and sec_m1 <= 25):
        macro_diag = "Institutional Drag / Headwind"
        macro_desc = "Lacking sector/theme momentum sponsorship. Prone to early churn or false breakout fader."
        macro_status = "drag"
    else:
        macro_diag = "Neutral / Market-In-Line Flow"
        macro_desc = "Sector and theme momentum aligned with broader market baseline."
        macro_status = "neutral"

    sector_theme_payload = {
        "sector": sec_name,
        "theme": primary_thm,
        "themes": themes_str,
        "sec_m1_pctile": round(sec_m1, 1) if sec_m1 is not None else None,
        "sec_m3_pctile": round(sec_m3, 1) if sec_m3 is not None else None,
        "thm_m1_pctile": round(thm_m1, 1) if thm_m1 is not None else None,
        "macro_diag": macro_diag,
        "macro_desc": macro_desc,
        "macro_status": macro_status,
        "composite_score": thm_eval.get("composite_score", 50.0),
        "primary_archetype": thm_eval.get("primary_archetype", "Market Baseline"),
        "is_veto": thm_eval.get("is_veto", False),
    }

    # Fundamental Catalyst & SEC EDGAR Intelligence
    catalyst_dossier = None
    ep_target_date = setup_row.get("prior_ep_date") or setup_row.get("post_ep_date") if setup_row else None
    if ep_target_date:
        try:
            catalyst_dossier = ep_catalyst_enricher.get_catalyst_dossier(sym, str(ep_target_date), sector=sec_name, theme=primary_thm)
        except Exception:
            pass

    # Build clean setup payload
    st_levels = setup_row.get("stack_levels", []) if setup_row else []
    if isinstance(st_levels, np.ndarray):
        st_levels = st_levels.tolist()
    elif not isinstance(st_levels, (list, tuple)):
        st_levels = []

    setup_payload = {
        "symbol": sym,
        "date": date_str,
        "state": str(setup_row.get("state", "building")) if setup_row else "building",
        "rs_score": int(setup_row["rs_score"]) if setup_row and pd.notna(setup_row.get("rs_score")) else None,
        "stage2": bool(setup_row.get("stage2", False)) if setup_row and pd.notna(setup_row.get("stage2")) else False,
        "dist_52w_high": round(float(setup_row["dist_52w_high"]), 1) if setup_row and pd.notna(setup_row.get("dist_52w_high")) else None,
        "move_pct": round(float(setup_row.get("move_pct", 0.0)), 1) if setup_row else 0.0,
        "cons_bars": int(setup_row.get("cons_bars", 0)) if setup_row else 0,
        "pullback_pct": round(float(setup_row.get("pullback_pct", 0.0)), 1) if setup_row else 0.0,
        "r2": round(float(setup_row["r2"]), 3) if setup_row and "r2" in setup_row and pd.notna(setup_row.get("r2")) else None,
        "pinch_ratio": round(float(setup_row["pinch_ratio"]), 2) if setup_row and "pinch_ratio" in setup_row and pd.notna(setup_row.get("pinch_ratio")) else None,
        "ride_frac": round(float(setup_row.get("ride_frac", 0.0)), 2) if setup_row else 0.0,
        "shakeouts": int(setup_row.get("shakeouts", 0)) if setup_row else 0,
        "vol_x": round(float(setup_row.get("vol_x", 1.0)), 2) if (setup_row and pd.notna(setup_row.get("vol_x"))) else None,
        "pattern": str(setup_row.get("pattern", "other")) if setup_row else "other",
        "level": round(float(setup_row.get("level", 0.0)), 2) if setup_row else round(float(df["low"].iloc[pos]), 2),
        "trigger": round(float(setup_row.get("trigger", 0.0)), 2) if setup_row else round(float(df["high"].iloc[pos]), 2),
        "dist_to_trigger_pct": round(float(setup_row.get("dist_to_trigger_pct", 0.0)), 2) if setup_row else 0.0,
        "qm_pole_start": str(setup_row.get("qm_pole_start", "")) if setup_row else "",
        "qm_pole_end": str(setup_row.get("qm_pole_end", "")) if setup_row else "",
        "qm_cons_start": str(setup_row.get("qm_cons_start", "")) if setup_row else "",
        "qm_cons_end": str(setup_row.get("qm_cons_end", "")) if setup_row else "",
        "stack_levels": [round(float(x), 2) for x in st_levels],
        "sector": sec_name,
        "theme": primary_thm,
        "themes": themes_str,
        "has_prior_ep": bool(setup_row.get("has_prior_ep", False)) if setup_row else False,
        "prior_ep_date": str(setup_row.get("prior_ep_date")) if setup_row and pd.notna(setup_row.get("prior_ep_date")) and setup_row.get("prior_ep_date") else None,
        "prior_ep_days_ago": int(setup_row["prior_ep_days_ago"]) if setup_row and pd.notna(setup_row.get("prior_ep_days_ago")) and setup_row.get("prior_ep_days_ago") is not None else None,
        "prior_ep_gap_pct": round(float(setup_row["prior_ep_gap_pct"]), 1) if setup_row and pd.notna(setup_row.get("prior_ep_gap_pct")) and setup_row.get("prior_ep_gap_pct") is not None else None,
        "prior_ep_rvol": round(float(setup_row["prior_ep_rvol"]), 1) if setup_row and pd.notna(setup_row.get("prior_ep_rvol")) and setup_row.get("prior_ep_rvol") is not None else None,
        "prior_ep_subtype": str(setup_row.get("prior_ep_subtype", "")) if setup_row and pd.notna(setup_row.get("prior_ep_subtype")) else None,
        "prior_ep_catalyst": str(setup_row.get("prior_ep_catalyst", "")) if setup_row and pd.notna(setup_row.get("prior_ep_catalyst")) else "",
        "has_post_ep": bool(setup_row.get("has_post_ep", False)) if setup_row else False,
        "post_ep_date": str(setup_row.get("post_ep_date")) if setup_row and pd.notna(setup_row.get("post_ep_date")) and setup_row.get("post_ep_date") else None,
        "post_ep_days_later": int(setup_row["post_ep_days_later"]) if setup_row and pd.notna(setup_row.get("post_ep_days_later")) and setup_row.get("post_ep_days_later") is not None else None,
        "post_ep_gap_pct": round(float(setup_row["post_ep_gap_pct"]), 1) if setup_row and pd.notna(setup_row.get("post_ep_gap_pct")) and setup_row.get("post_ep_gap_pct") is not None else None,
        "post_ep_rvol": round(float(setup_row["post_ep_rvol"]), 1) if setup_row and pd.notna(setup_row.get("post_ep_rvol")) and setup_row.get("post_ep_rvol") is not None else None,
        "post_ep_subtype": str(setup_row.get("post_ep_subtype", "")) if setup_row and pd.notna(setup_row.get("post_ep_subtype")) else None,
        "post_ep_catalyst": str(setup_row.get("post_ep_catalyst", "")) if setup_row and pd.notna(setup_row.get("post_ep_catalyst")) else "",
        "ep_badge": str(setup_row.get("ep_badge", "Pure Technical QM")) if setup_row else "Pure Technical QM",
    }

    # Fetch all setups for this symbol for multi-setup visualization
    symbol_setups = []
    if not dataset.empty:
        sym_matches = dataset[dataset["symbol"] == sym].sort_values("date")
        for _, s_row in sym_matches.iterrows():
            st_levs = s_row.get("stack_levels", [])
            if isinstance(st_levs, np.ndarray):
                st_levs = st_levs.tolist()
            elif not isinstance(st_levs, (list, tuple)):
                st_levs = []
            symbol_setups.append({
                "symbol": sym,
                "date": str(s_row["date"]),
                "state": str(s_row.get("state", "building")),
                "level": round(float(s_row.get("level", 0.0)), 2),
                "trigger": round(float(s_row.get("trigger", 0.0)), 2),
                "dist_to_trigger_pct": round(float(s_row.get("dist_to_trigger_pct", 0.0)), 2),
                "qm_pole_start": str(s_row.get("qm_pole_start", "")),
                "qm_pole_end": str(s_row.get("qm_pole_end", "")),
                "qm_cons_start": str(s_row.get("qm_cons_start", "")),
                "qm_cons_end": str(s_row.get("qm_cons_end", "")),
                "stack_levels": [round(float(x), 2) for x in st_levs],
                "r2": round(float(s_row["r2"]), 3) if ("r2" in s_row and pd.notna(s_row.get("r2"))) else None,
                "pinch_ratio": round(float(s_row["pinch_ratio"]), 2) if ("pinch_ratio" in s_row and pd.notna(s_row.get("pinch_ratio"))) else None,
                "move_pct": round(float(s_row.get("move_pct", 0.0)), 1),
                "cons_bars": int(s_row.get("cons_bars", 0)),
                "pullback_pct": round(float(s_row.get("pullback_pct", 0.0)), 1),
                "pattern": str(s_row.get("pattern", "other")),
                "ep_badge": str(s_row.get("ep_badge", "Pure Technical QM")),
            })

    # Enrich with Strategy Trade Execution & Conviction Sizing if breakout
    _ = get_strategy_df()
    if STRATEGY_LOOKUP:
        tr = STRATEGY_LOOKUP.get((str(sym), str(date_str)))
        if tr:
            setup_payload["trade_info"] = {
                "entry_price": round(float(tr["entry_price"]), 2),
                "pivot_low_price": round(float(tr["pivot_low_price"]), 2),
                "pivot_low_date": str(tr["pivot_low_date"]),
                "full_pivot_dist": round(float(tr["full_pivot_dist"]), 2),
                "stop_loss_06x": round(float(tr["stop_loss_06x"]), 2),
                "risk_pct_06x": round(float(tr["risk_pct_06x"]), 1),
                "exit_date_06x": str(tr["exit_date_06x"]),
                "exit_price_06x": round(float(tr["exit_price_06x"]), 2),
                "exit_reason_06x": str(tr["exit_reason_06x"]),
                "bars_06x": int(tr["bars_06x"]),
                "r_06x": round(float(tr.get("r_e10_06x", tr.get("r_06x", 0.0))), 2),
                "sized_r_06x": round(float(tr.get("sized_r_e10_06x", tr.get("sized_r_06x", 0.0))), 2),
                "gain_pct_06x": round(float(tr["gain_pct_06x"]), 1),
                "mfe_r_06x": round(float(tr["mfe_r_06x"]), 2),
                "mae_r_06x": round(float(tr["mae_r_06x"]), 2),
                "is_win_06x": bool(tr.get("is_win_e10_06x", tr.get("is_win_06x", False))),
                # Trailing Exits (0.6x Stop)
                "r_e10_06x": round(float(tr.get("r_e10_06x", tr.get("r_06x", 0.0))), 2),
                "sized_r_e10_06x": round(float(tr.get("sized_r_e10_06x", tr.get("sized_r_06x", 0.0))), 2),
                "is_win_e10_06x": bool(tr.get("is_win_e10_06x", tr.get("is_win_06x", False))),
                "bars_e10_06x": int(tr.get("bars_e10_06x", tr.get("bars_06x", 0))),
                "r_e20_06x": round(float(tr.get("r_e20_06x", 0.0)), 2),
                "sized_r_e20_06x": round(float(tr.get("sized_r_e20_06x", 0.0)), 2),
                "is_win_e20_06x": bool(tr.get("is_win_e20_06x", False)),
                "bars_e20_06x": int(tr.get("bars_e20_06x", 0)),
                "r_e50_06x": round(float(tr.get("r_e50_06x", 0.0)), 2),
                "sized_r_e50_06x": round(float(tr.get("sized_r_e50_06x", 0.0)), 2),
                "is_win_e50_06x": bool(tr.get("is_win_e50_06x", False)),
                "bars_e50_06x": int(tr.get("bars_e50_06x", 0)),
                "r_staged_06x": round(float(tr.get("r_staged_06x", 0.0)), 2),
                "sized_r_staged_06x": round(float(tr.get("sized_r_staged_06x", 0.0)), 2),
                "is_win_staged_06x": bool(tr.get("is_win_staged_06x", False)),
                "bars_staged_06x": int(tr.get("bars_staged_06x", 0)),
                "r_balanced_06x": round(float(tr.get("r_balanced_06x", 0.0)), 2),
                "sized_r_balanced_06x": round(float(tr.get("sized_r_balanced_06x", 0.0)), 2),
                "is_win_balanced_06x": bool(tr.get("is_win_balanced_06x", False)),
                "bars_balanced_06x": int(tr.get("bars_balanced_06x", 0)),
                # Dedicated ML Climax Partial Scaling Exits (0.6x Stop)
                "r_climax_swing_06x": round(float(tr.get("r_climax_swing_06x", 0.0)), 2),
                "sized_r_climax_swing_06x": round(float(tr.get("sized_r_climax_swing_06x", 0.0)), 2),
                "is_win_climax_swing_06x": bool(tr.get("is_win_climax_swing_06x", False)),
                "bars_climax_swing_06x": int(tr.get("bars_climax_swing_06x", 0)),
                "climax_date_06x": str(tr["climax_date_06x"]) if pd.notna(tr.get("climax_date_06x")) and tr.get("climax_date_06x") else None,
                "climax_price_06x": round(float(tr["climax_price_06x"]), 2) if pd.notna(tr.get("climax_price_06x")) and tr.get("climax_price_06x") is not None else None,
                "climax_r_06x": round(float(tr["climax_r_06x"]), 2) if pd.notna(tr.get("climax_r_06x")) and tr.get("climax_r_06x") is not None else None,
                "prob_exhaustion_06x": round(float(tr["prob_exhaustion_06x"]), 3) if pd.notna(tr.get("prob_exhaustion_06x")) and tr.get("prob_exhaustion_06x") is not None else None,
                "r_climax_hybrid_06x": round(float(tr.get("r_climax_hybrid_06x", 0.0)), 2),
                "sized_r_climax_hybrid_06x": round(float(tr.get("sized_r_climax_hybrid_06x", 0.0)), 2),
                "is_win_climax_hybrid_06x": bool(tr.get("is_win_climax_hybrid_06x", False)),
                "bars_climax_hybrid_06x": int(tr.get("bars_climax_hybrid_06x", 0)),
                "r_climax_trend_06x": round(float(tr.get("r_climax_trend_06x", 0.0)), 2),
                "sized_r_climax_trend_06x": round(float(tr.get("sized_r_climax_trend_06x", 0.0)), 2),
                "is_win_climax_trend_06x": bool(tr.get("is_win_climax_trend_06x", False)),
                "bars_climax_trend_06x": int(tr.get("bars_climax_trend_06x", 0)),
                # 1.0x Full Pivot Stoploss Trade
                "stop_loss_10x": round(float(tr["stop_loss_10x"]), 2),
                "risk_pct_10x": round(float(tr["risk_pct_10x"]), 1),
                "exit_date_10x": str(tr["exit_date_10x"]),
                "exit_price_10x": round(float(tr["exit_price_10x"]), 2),
                "exit_reason_10x": str(tr["exit_reason_10x"]),
                "bars_10x": int(tr["bars_10x"]),
                "r_10x": round(float(tr.get("r_e10_10x", tr.get("r_10x", 0.0))), 2),
                "sized_r_10x": round(float(tr.get("sized_r_e10_10x", tr.get("sized_r_10x", 0.0))), 2),
                "gain_pct_10x": round(float(tr["gain_pct_10x"]), 1),
                "mfe_r_10x": round(float(tr["mfe_r_10x"]), 2),
                "mae_r_10x": round(float(tr["mae_r_10x"]), 2),
                "is_win_10x": bool(tr.get("is_win_e10_10x", tr.get("is_win_10x", False))),
                # Trailing Exits (1.0x Stop)
                "r_e10_10x": round(float(tr.get("r_e10_10x", tr.get("r_10x", 0.0))), 2),
                "sized_r_e10_10x": round(float(tr.get("sized_r_e10_10x", tr.get("sized_r_10x", 0.0))), 2),
                "is_win_e10_10x": bool(tr.get("is_win_e10_10x", tr.get("is_win_10x", False))),
                "bars_e10_10x": int(tr.get("bars_e10_10x", tr.get("bars_10x", 0))),
                "r_e20_10x": round(float(tr.get("r_e20_10x", 0.0)), 2),
                "sized_r_e20_10x": round(float(tr.get("sized_r_e20_10x", 0.0)), 2),
                "is_win_e20_10x": bool(tr.get("is_win_e20_10x", False)),
                "bars_e20_10x": int(tr.get("bars_e20_10x", 0)),
                "r_e50_10x": round(float(tr.get("r_e50_10x", 0.0)), 2),
                "sized_r_e50_10x": round(float(tr.get("sized_r_e50_10x", 0.0)), 2),
                "is_win_e50_10x": bool(tr.get("is_win_e50_10x", False)),
                "bars_e50_10x": int(tr.get("bars_e50_10x", 0)),
                "r_staged_10x": round(float(tr.get("r_staged_10x", 0.0)), 2),
                "sized_r_staged_10x": round(float(tr.get("sized_r_staged_10x", 0.0)), 2),
                "is_win_staged_10x": bool(tr.get("is_win_staged_10x", False)),
                "bars_staged_10x": int(tr.get("bars_staged_10x", 0)),
                "r_balanced_10x": round(float(tr.get("r_balanced_10x", 0.0)), 2),
                "sized_r_balanced_10x": round(float(tr.get("sized_r_balanced_10x", 0.0)), 2),
                "is_win_balanced_10x": bool(tr.get("is_win_balanced_10x", False)),
                "bars_balanced_10x": int(tr.get("bars_balanced_10x", 0)),
                # Dedicated ML Climax Partial Scaling Exits (1.0x Stop)
                "r_climax_swing_10x": round(float(tr.get("r_climax_swing_10x", 0.0)), 2),
                "sized_r_climax_swing_10x": round(float(tr.get("sized_r_climax_swing_10x", 0.0)), 2),
                "is_win_climax_swing_10x": bool(tr.get("is_win_climax_swing_10x", False)),
                "bars_climax_swing_10x": int(tr.get("bars_climax_swing_10x", 0)),
                "climax_date_10x": str(tr["climax_date_10x"]) if pd.notna(tr.get("climax_date_10x")) and tr.get("climax_date_10x") else None,
                "climax_price_10x": round(float(tr["climax_price_10x"]), 2) if pd.notna(tr.get("climax_price_10x")) and tr.get("climax_price_10x") is not None else None,
                "climax_r_10x": round(float(tr["climax_r_10x"]), 2) if pd.notna(tr.get("climax_r_10x")) and tr.get("climax_r_10x") is not None else None,
                "prob_exhaustion_10x": round(float(tr["prob_exhaustion_10x"]), 3) if pd.notna(tr.get("prob_exhaustion_10x")) and tr.get("prob_exhaustion_10x") is not None else None,
                "r_climax_hybrid_10x": round(float(tr.get("r_climax_hybrid_10x", 0.0)), 2),
                "sized_r_climax_hybrid_10x": round(float(tr.get("sized_r_climax_hybrid_10x", 0.0)), 2),
                "is_win_climax_hybrid_10x": bool(tr.get("is_win_climax_hybrid_10x", False)),
                "bars_climax_hybrid_10x": int(tr.get("bars_climax_hybrid_10x", 0)),
                "r_climax_trend_10x": round(float(tr.get("r_climax_trend_10x", 0.0)), 2),
                "sized_r_climax_trend_10x": round(float(tr.get("sized_r_climax_trend_10x", 0.0)), 2),
                "is_win_climax_trend_10x": bool(tr.get("is_win_climax_trend_10x", False)),
                "bars_climax_trend_10x": int(tr.get("bars_climax_trend_10x", 0)),
                "pos_size_mult": round(float(tr["pos_size_mult"]), 2),
                "s_ep": round(float(tr["s_ep"]), 2),
                "s_pole": round(float(tr["s_pole"]), 2),
                "s_rvol": round(float(tr["s_rvol"]), 2),
                "s_pb": round(float(tr["s_pb"]), 2),
                "spy_above_200sma": bool(tr["spy_above_200sma"]),
                "spy_above_50sma": bool(tr["spy_above_50sma"]),
                "qqq_bull_stack": bool(tr["qqq_bull_stack"]),
                "qqq_pullback": bool(tr["qqq_pullback"]),
                "qqq_spy_ratio_bull": bool(tr["qqq_spy_ratio_bull"]),
                "is_leading_macro": bool(tr["is_leading_macro"]),
                "is_leading_sector": bool(tr["is_leading_sector"]),
                "is_leading_theme": bool(tr["is_leading_theme"]),
                # Institutional ML Triad Fields
                "qm_score": round(float(tr.get("qm_score", 50.0)), 1) if pd.notna(tr.get("qm_score")) else None,
                "qm_tier": str(tr.get("qm_tier", "Standard")) if pd.notna(tr.get("qm_tier")) else "Standard",
                "p_big_move": round(float(tr.get("p_big_move", 0.25)), 3) if pd.notna(tr.get("p_big_move")) else None,
                "p_trap": round(float(tr.get("p_trap", 0.23)), 3) if pd.notna(tr.get("p_trap")) else None,
                "conviction_sizing": round(float(tr.get("conviction_sizing", 1.0)), 2) if pd.notna(tr.get("conviction_sizing")) else 1.0,
                "ml_recommendation": str(tr.get("ml_recommendation", "")) if pd.notna(tr.get("ml_recommendation")) else "",
                "pred_mae_pct": round(float(tr.get("pred_mae_pct", 4.5)), 2) if pd.notna(tr.get("pred_mae_pct")) else None,
                "stop_loss_dynamic": round(float(tr.get("stop_loss_dynamic", 0.0)), 2) if pd.notna(tr.get("stop_loss_dynamic")) else None,
                "risk_pct_dynamic": round(float(tr.get("risk_pct_dynamic", 4.5)), 2) if pd.notna(tr.get("risk_pct_dynamic")) else None,
                "r_dyn_06x": round(float(tr.get("r_dyn_06x", 0.0)), 2) if pd.notna(tr.get("r_dyn_06x")) else 0.0,
                "sized_r_dyn_06x": round(float(tr.get("sized_r_dyn_06x", 0.0)), 2) if pd.notna(tr.get("sized_r_dyn_06x")) else 0.0,
                "is_win_dyn_06x": bool(tr.get("is_win_dyn_06x", False)),
                "r_dyn_climax_swing": round(float(tr.get("r_dyn_climax_swing", 0.0)), 2) if pd.notna(tr.get("r_dyn_climax_swing")) else 0.0,
                "sized_r_dyn_climax_swing": round(float(tr.get("sized_r_dyn_climax_swing", 0.0)), 2) if pd.notna(tr.get("sized_r_dyn_climax_swing")) else 0.0,
                "is_win_dyn_climax_swing": bool(tr.get("is_win_dyn_climax_swing", False)),
                "r_triad_optimal": round(float(tr.get("r_triad_optimal", 0.0)), 2) if pd.notna(tr.get("r_triad_optimal")) else 0.0,
                "sized_r_triad_optimal": round(float(tr.get("sized_r_triad_optimal", 0.0)), 2) if pd.notna(tr.get("sized_r_triad_optimal")) else 0.0,
                "is_win_triad_optimal": bool(tr.get("is_win_triad_optimal", False)),
            }

    # Generate Real-Time ML Guidance for any setup (fresh or historical)
    try:
        if setup_payload.get("trade_info"):
            cur_p = float(setup_payload["trade_info"]["entry_price"])
            p_low = float(setup_payload["trade_info"]["pivot_low_price"])
        else:
            cur_p = float(setup_payload.get("trigger", df["close"].iloc[pos])) if setup_payload else float(df["close"].iloc[pos])
            p_low = float(setup_payload.get("level", df["low"].iloc[pos])) if setup_payload else float(df["low"].iloc[pos])
        setup_payload["ml_guidance"] = qm_ml_engine.get_qm_ml_engine().get_setup_guidance(setup_row if setup_row is not None else {}, cur_p, p_low)
    except Exception:
        pass

    return jsonify({
        "symbol": sym,
        "date": date_str,
        "bars": bars,
        "volume": volume,
        "vol_ma20": vol_ma20_data,
        "ema10": ema10_data,
        "ema20": ema20_data,
        "sma50": sma50_data,
        "sma200": sma200_data,
        "larsson": larsson_data,
        "larsson_spans": list(larsson_spans),
        "setup": setup_payload,
        "symbol_setups": symbol_setups,
        "sector_theme": sector_theme_payload,
        "catalyst": catalyst_dossier,
        "all_eps": symbol_eps,
    })


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>QM Momentum Continuation & EP Explorer</title>
<script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root {
    --bg: #0b0e14; --surface: #151922; --surface-2: #1e2533; --border: #232936;
    --text: #e2e8f0; --muted: #7b849b; --accent: #3b82f6;
    --green: #10b981; --red: #ef4444; --gold: #f59e0b; --purple: #a855f7; --cyan: #06b6d4;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
    font-size: 13px; height: 100vh; display: flex; flex-direction: column; overflow: hidden;
  }
  header {
    background: var(--surface); border-bottom: 1px solid var(--border);
    padding: 8px 18px; display: flex; justify-content: space-between; align-items: center; flex-shrink: 0;
  }
  .title { font-size: 16px; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 8px; }
  .badge { background: #1e293b; color: var(--gold); border: 1px solid #334155; padding: 2px 7px; border-radius: 4px; font-size: 11px; }

  .reload-btn {
    background: #1e293b; color: #38bdf8; border: 1px solid #0284c7;
    padding: 5px 12px; border-radius: 6px; font-size: 11.5px; font-weight: 600; cursor: pointer;
    transition: all 0.15s ease; display: flex; align-items: center; gap: 6px;
  }
  .reload-btn:hover { background: #0369a1; color: #fff; }

  /* KPI Cards */
  .kpi-row {
    display: flex; gap: 10px; padding: 7px 16px; background: #0f131c; border-bottom: 1px solid var(--border); flex-shrink: 0;
  }
  .kpi-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
    padding: 6px 12px; flex: 1; min-width: 110px; cursor: default;
  }
  .kpi-label { font-size: 10.5px; color: var(--muted); text-transform: uppercase; font-weight: 600; margin-bottom: 2px; }
  .kpi-val { font-size: 15px; font-weight: 700; color: #fff; }
  .kpi-sub { font-size: 10.5px; color: var(--muted); margin-top: 2px; }

  /* Toolbar */
  .toolbar {
    background: var(--surface); border-bottom: 1px solid var(--border); padding: 6px 16px;
    display: flex; flex-wrap: wrap; gap: 10px; align-items: center; font-size: 12px; flex-shrink: 0;
  }
  .filter-group { display: flex; align-items: center; gap: 5px; }
  .filter-group label { color: var(--muted); font-size: 10.5px; text-transform: uppercase; font-weight: 600; }
  select, input {
    background: #0f131a; color: #e2e8f0; border: 1px solid var(--border);
    padding: 4px 8px; border-radius: 4px; font-size: 12px; outline: none;
  }
  select:focus, input:focus { border-color: var(--accent); }

  .btn-filter-chip {
    background: #182030; color: #94a3b8; border: 1px solid #2a3449;
    padding: 3px 9px; border-radius: 12px; font-size: 11px; font-weight: 600; cursor: pointer;
    transition: all 0.15s ease;
  }
  .btn-filter-chip:hover { background: #232d42; color: #e2e8f0; border-color: #3b4968; }
  .btn-filter-chip.active {
    background: #2563eb; color: #fff; border-color: #60a5fa; box-shadow: 0 0 8px rgba(37,99,235,0.4);
  }

  /* 3-Pane Workspace */
  .workspace { display: flex; flex: 1; min-height: 0; overflow: hidden; position: relative; }
  
  /* Left Pane: Table */
  .table-pane { flex: 1.1; min-width: 440px; min-height: 0; overflow: auto; border-right: 1px solid var(--border); }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th {
    background: #11151f; color: var(--muted); font-weight: 600; text-align: right;
    padding: 7px 9px; border-bottom: 1px solid var(--border); position: sticky; top: 0; z-index: 10;
    cursor: pointer; user-select: none;
  }
  th.tl { text-align: left; }
  td { padding: 6px 9px; border-bottom: 1px solid #1a202c; text-align: right; white-space: nowrap; }
  td.tl { text-align: left; }
  tr { cursor: pointer; transition: background 0.1s; }
  tr:hover { background: #1a2233; }
  tr.selected { background: #1e293b; outline: 1px solid var(--accent); }
  .setup-cb { cursor: pointer; accent-color: #38bdf8; width: 14px; height: 14px; vertical-align: middle; }
  .color-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }

  /* Center Pane: Full-Height Chart */
  .chart-pane { flex: 1.5; min-width: 480px; min-height: 0; display: flex; flex-direction: column; background: #0c0f16; border-right: 1px solid var(--border); position: relative; overflow: hidden; }

  .chart-header {
    background: #11151f; border-bottom: 1px solid var(--border);
    padding: 6px 14px; display: flex; justify-content: space-between; align-items: center; flex-shrink: 0;
    gap: 8px;
  }
  .chart-title-box {
    display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: 13.5px;
    min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; padding-right: 8px;
  }
  .chart-controls { display: flex; align-items: center; gap: 6px; flex-shrink: 0; }

  .btn-chart-enlarge {
    background: #1e293b; color: #38bdf8; border: 1px solid #0284c7;
    padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 700;
    cursor: pointer; display: inline-flex; align-items: center; gap: 4px;
    transition: all 0.15s ease;
    white-space: nowrap;
    flex-shrink: 0;
  }
  .btn-chart-enlarge:hover {
    background: #0284c7; color: #fff;
  }
  .btn-chart-enlarge.active {
    background: #0284c7; color: #fff; border-color: #38bdf8;
    box-shadow: 0 0 8px rgba(56, 189, 248, 0.4);
  }
  .btn-chart-expand-main {
    background: #0369a1;
    color: #f0f9ff;
    border-color: #38bdf8;
  }
  .btn-chart-expand-main:hover {
    background: #0284c7;
  }

  /* Chart header controls visible only when enlarged */
  .chart-enlarged-only-controls {
    display: none !important;
  }
  .workspace.fullscreen-chart-mode .chart-enlarged-only-controls {
    display: inline-flex !important;
    align-items: center;
    gap: 6px;
    flex-wrap: nowrap;
  }
  .workspace.fullscreen-chart-mode #btn_enlarged_stock_list {
    background: #0284c7 !important;
    color: #fff !important;
    border-color: #38bdf8 !important;
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.4);
  }

  /* Enlarged Stock List Table */
  .esl-table { width: 100%; border-collapse: collapse; font-size: 11.5px; }
  .esl-table th {
    padding: 7px 10px; background: #161e2e; color: #94a3b8; font-weight: 700;
    text-transform: uppercase; font-size: 10px; letter-spacing: 0.5px;
    border-bottom: 1px solid #283548; user-select: none;
  }
  .esl-table th.tl { text-align: left; }
  .esl-table td { padding: 6px 10px; border-bottom: 1px solid #1a2333; }
  .esl-table td.tl { text-align: left; }
  .esl-table tr { cursor: pointer; transition: background 0.12s ease; }
  .esl-table tr:hover { background: rgba(56, 189, 248, 0.08); }
  .esl-table tr.selected { background: rgba(2, 132, 199, 0.22) !important; border-left: 3px solid #38bdf8; }

  /* Chart Settings Floating Panel */
  .chart-settings-panel {
    position: absolute; top: 10px; right: 14px; z-index: 60;
    width: 290px; background: rgba(15, 23, 42, 0.95); border: 1px solid #334155;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.7); border-radius: 8px;
    backdrop-filter: blur(10px); overflow: hidden; display: flex; flex-direction: column;
    animation: fadeIn 0.15s ease-out;
  }
  .settings-panel-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 8px 12px; background: #111827; border-bottom: 1px solid #1f2937;
  }
  .panel-close-btn {
    background: transparent; border: none; color: #94a3b8; font-size: 16px;
    font-weight: 700; cursor: pointer; line-height: 1; padding: 2px 6px; border-radius: 4px;
  }
  .panel-close-btn:hover { color: #fff; background: #334155; }
  .settings-panel-body {
    padding: 10px 12px; display: flex; flex-direction: column; gap: 10px;
    max-height: 480px; overflow-y: auto;
  }
  .settings-section {
    border-bottom: 1px solid #1e293b; padding-bottom: 8px;
  }
  .settings-section:last-child { border-bottom: none; padding-bottom: 0; }
  .settings-sec-title {
    font-size: 9.5px; font-weight: 800; color: #64748b; letter-spacing: 0.5px;
    margin-bottom: 6px; text-transform: uppercase;
  }
  .settings-row {
    display: flex; align-items: center; margin-bottom: 4px;
  }
  .settings-row:last-child { margin-bottom: 0; }
  .toggle-control {
    display: inline-flex; align-items: center; gap: 6px; cursor: pointer; user-select: none;
  }
  .toggle-label {
    font-weight: 600;
  }

  /* Chart Drawing Toolbar */
  .chart-draw-toolbar {
    position: absolute; top: 10px; left: 10px; z-index: 50;
    background: rgba(15, 23, 42, 0.92); border: 1px solid #334155;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.6); border-radius: 6px;
    padding: 3px 6px; display: flex; align-items: center; gap: 4px;
    backdrop-filter: blur(8px);
  }
  .draw-btn {
    background: #1e293b; color: #cbd5e1; border: 1px solid #334155;
    padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;
    cursor: pointer; display: inline-flex; align-items: center; gap: 4px;
    transition: all 0.15s ease;
  }
  .draw-btn:hover { background: #334155; color: #fff; }
  .draw-btn.active {
    background: #0284c7; color: #fff; border-color: #38bdf8;
    box-shadow: 0 0 8px rgba(56, 189, 248, 0.5);
  }
  .draw-btn-danger:hover { background: #991b1b; border-color: #ef4444; color: #fff; }

  .chart-container-rel { flex: 1; min-height: 0; position: relative; width: 100%; overflow: hidden; }
  #chart_tv { width: 100%; height: 100%; min-height: 0; }
  #chart_canvas_overlay {
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none; z-index: 6;
  }

  .chart-hud {
    position: absolute; top: 10px; left: 14px; z-index: 20;
    background: rgba(15, 23, 42, 0.9); backdrop-filter: blur(8px);
    border: 1px solid #334155; border-radius: 6px; padding: 4px 10px;
    font-size: 11.5px; color: #e2e8f0; pointer-events: none; display: flex; gap: 12px;
  }

  /* Right Pane: Dossier */
  .dossier-pane {
    flex: 0 0 380px; width: 380px; max-width: 420px; min-height: 0; overflow-y: auto;
    background: var(--surface); padding: 14px; display: flex; flex-direction: column; gap: 12px;
  }

  .dossier-card {
    background: #0f131c; border: 1px solid var(--border); border-radius: 6px;
    padding: 12px; display: flex; flex-direction: column; gap: 8px;
  }
  .dossier-title {
    font-size: 11.5px; font-weight: 700; color: var(--gold); text-transform: uppercase;
    display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #1a2233; padding-bottom: 4px;
  }
  .dossier-row { display: flex; justify-content: space-between; align-items: center; font-size: 12px; }
  .dossier-label { color: var(--muted); }
  .dossier-val { font-weight: 600; color: #f8fafc; }

  /* Badges & Pills */
  .pill-breakout { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #059669; padding: 2px 6px; border-radius: 4px; font-size: 10.5px; font-weight: 700; text-transform: uppercase; }
  .pill-building { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #d97706; padding: 2px 6px; border-radius: 4px; font-size: 10.5px; font-weight: 700; text-transform: uppercase; }
  
  .badge-ep-ignited { background: rgba(6, 182, 212, 0.15); color: #22d3ee; border: 1px solid #0891b2; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: 700; }
  .badge-ep-boosted { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid #9333ea; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: 700; }
  .badge-ep-both { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid #059669; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: 700; }
  .badge-ep-tech { background: #1e293b; color: #94a3b8; border: 1px solid #334155; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: 600; }

  /* Fullscreen Mode */
  .workspace.fullscreen-chart-mode {
    position: fixed !important;
    inset: 0 !important;
    z-index: 9999 !important;
    width: 100vw !important;
    height: 100vh !important;
    background: #0b0e14 !important;
    overflow: hidden !important;
  }
  .workspace.fullscreen-chart-mode .table-pane { display: none !important; }
  .workspace.fullscreen-chart-mode .chart-pane {
    flex: 1 !important;
    min-width: 0 !important;
    height: 100vh !important;
    border-right: 1px solid var(--border) !important;
    position: relative !important;
  }
  .workspace.fullscreen-chart-mode .dossier-pane {
    flex: 0 0 420px !important;
    width: 420px !important;
    max-width: 460px !important;
    height: 100vh !important;
    overflow-y: auto !important;
  }

  /* Sector/Theme Context & Momentum Backdrop */
  .ctx-row { display: flex; flex-direction: column; gap: 6px; font-size: 12px; }
  .ctx-item { display: flex; justify-content: space-between; align-items: center; }
  .ctx-tag { font-weight: 600; color: #f1f5f9; }
  .ctx-pctiles { display: flex; gap: 4px; }
  .pctile-badge { font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 3px; }
  .pctile-high { background: rgba(16, 185, 129, 0.2); color: #34d399; }
  .pctile-mid { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
  .pctile-low { background: rgba(239, 68, 68, 0.2); color: #f87171; }
  .diag-pill {
    padding: 5px 9px; border-radius: 4px; font-size: 11px; font-weight: 600; line-height: 1.4;
  }
  .diag-tailwind { background: rgba(16, 185, 129, 0.12); color: #34d399; border-left: 3px solid #10b981; }
  .diag-neutral { background: rgba(148, 163, 184, 0.12); color: #cbd5e1; border-left: 3px solid #64748b; }
  .diag-drag { background: rgba(239, 68, 68, 0.12); color: #f87171; border-left: 3px solid #ef4444; }

  /* Stock Leaderboard Modal */
  .modal-overlay {
    display: none; position: fixed; inset: 0; background: rgba(0, 0, 0, 0.82);
    z-index: 100005 !important; justify-content: center; align-items: center; padding: 20px;
    backdrop-filter: blur(5px);
  }
  .modal-overlay.active { display: flex; }
  .modal-box {
    background: #121722; border: 1px solid #2d3748; border-radius: 8px; width: 100%; max-width: 840px;
    max-height: 85vh; display: flex; flex-direction: column; box-shadow: 0 10px 30px rgba(0,0,0,0.85);
  }
  .modal-header {
    padding: 12px 18px; border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center; background: #171d2b;
  }
  .modal-title { font-size: 14px; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 8px; }
  .modal-close {
    background: none; border: none; color: var(--muted); font-size: 20px; cursor: pointer;
  }
  .modal-close:hover { color: #fff; }
  .modal-body {
    padding: 16px 20px; overflow-y: auto; line-height: 1.6; font-size: 12px; color: #cbd5e1;
  }

  /* Strategy Mode Toolbar Buttons */
  .btn-strategy {
    background: #0f172a; color: #94a3b8; border: 1px solid #334155;
    padding: 4px 10px; border-radius: 6px; font-size: 11.5px; font-weight: 700;
    cursor: pointer; display: inline-flex; align-items: center; gap: 5px; transition: all 0.2s ease;
  }
  .btn-strategy:hover { background: #1e293b; color: #e2e8f0; border-color: #64748b; }
  .btn-strategy.active {
    background: linear-gradient(135deg, #0369a1 0%, #0284c7 100%);
    color: #fff; border-color: #38bdf8;
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.4);
  }
  .btn-strategy-settings {
    background: #0f172a; color: #cbd5e1; border: 1px solid #334155;
    padding: 4px 8px; border-radius: 6px; font-size: 12.5px; cursor: pointer;
    transition: all 0.2s ease;
  }
  .btn-strategy-settings:hover { background: #1e293b; color: #38bdf8; border-color: #0284c7; }

  /* Strategy KPI Ribbon */
  .strategy-kpi-bar {
    display: flex; gap: 8px; padding: 7px 16px; background: linear-gradient(90deg, #07101e 0%, #0b172a 100%);
    border-bottom: 2px solid #0284c7; flex-shrink: 0; overflow-x: auto;
  }
  .strat-kpi-card {
    background: rgba(15, 23, 42, 0.85); border: 1px solid #1e3a5f; border-radius: 6px;
    padding: 5px 10px; flex: 1; min-width: 105px;
  }
  .strat-kpi-label { font-size: 9.5px; color: #93c5fd; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; margin-bottom: 2px; }
  .strat-kpi-val { font-size: 14.5px; font-weight: 800; color: #fff; }
  .strat-kpi-sub { font-size: 9.5px; color: #64748b; margin-top: 1px; }

  /* Strategy Settings Modal Elements */
  .settings-group {
    background: #111827; border: 1px solid #1f2937; border-radius: 6px; padding: 10px 12px;
  }
  .settings-group-title {
    font-size: 11.5px; font-weight: 700; color: #38bdf8; margin-bottom: 4px; display: flex; align-items: center; justify-content: space-between;
  }
  .settings-group-desc { font-size: 10.5px; color: var(--muted); margin-bottom: 6px; line-height: 1.35; }
  .cfg-select {
    width: 100%; padding: 5px 8px; background: #0b0f19; border: 1px solid #374151; color: #f3f4f6; border-radius: 4px; font-size: 11.5px; outline: none;
  }
</style>
</head>
<body>

<header>
  <div class="title">
    <span>⚡ QM MOMENTUM VIEWER</span>
    <span class="badge">QULLAMAGGIE CONTINUATION</span>
    <span style="font-size:12px; color:var(--muted); font-weight:400; margin-left:8px;">EP Catalyst Alignment & Pure Setup Explorer</span>
  </div>
  <div style="display:flex; align-items:center; gap:10px;">
    <button class="reload-btn" onclick="reloadDataset()">🔄 Reload Dataset</button>
    <span style="font-size:11px; color:#64748b; font-family:monospace;">PORT 8785</span>
  </div>
</header>

<!-- KPI Ribbon -->
<div class="kpi-row" id="kpi_row">
  <div class="kpi-card">
    <div class="kpi-label">TOTAL SETUPS</div>
    <div class="kpi-val" id="kpi_total">--</div>
    <div class="kpi-sub">Mined historical bases</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">BREAKOUTS</div>
    <div class="kpi-val" style="color:#34d399;" id="kpi_breakouts">--</div>
    <div class="kpi-sub">Volume-confirmed breaks</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">BUILDING</div>
    <div class="kpi-val" style="color:#fbbf24;" id="kpi_building">--</div>
    <div class="kpi-sub">Mature coiling bases</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">EP-IGNITED POLE</div>
    <div class="kpi-val" style="color:#38bdf8;" id="kpi_ignited">--%</div>
    <div class="kpi-sub">Prior EP within 45d</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">POST-EP BOOSTED</div>
    <div class="kpi-val" style="color:#c084fc;" id="kpi_post">--%</div>
    <div class="kpi-sub">Catalyst gap within 20d</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">AVG POLE MOVE</div>
    <div class="kpi-val" style="color:#e2e8f0;" id="kpi_avg_move">--%</div>
    <div class="kpi-sub">Launch to pole peak</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">THEMATIC LEADER</div>
    <div class="kpi-val" style="color:#c084fc;" id="kpi_thematic_lead">--%</div>
    <div class="kpi-sub">Leading / Emerging group</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">RS LEADER (≥90)</div>
    <div class="kpi-val" style="color:#38bdf8;" id="kpi_rs_leader">--%</div>
    <div class="kpi-sub">Stage 2 / ≤15% 52W High</div>
  </div>
</div>

<!-- Filter Toolbar -->
<div class="toolbar">
  <div class="filter-group">
    <label>⚡ RS &amp; Posture:</label>
    <select id="sel_rs" onchange="fetchEvents()" style="background:#0f172a; border-color:#0284c7; color:#38bdf8; font-weight:700;">
      <option value="rs90_stage2_or_15" selected>⚡ RS ≥ 90 (Stage 2 or ≤15% 52W High) — Default</option>
      <option value="rs95_stage2_or_15">🔥 RS ≥ 95 (Stage 2 or ≤15% 52W High)</option>
      <option value="rs80_stage2_or_15">📈 RS ≥ 80 (Stage 2 or ≤15% 52W High)</option>
      <option value="rs70_stage2_or_15">📊 RS ≥ 70 (Stage 2 or ≤15% 52W High)</option>
      <option value="rs90_any">RS ≥ 90 (Any Posture)</option>
      <option value="rs80_any">RS ≥ 80 (Any Posture)</option>
      <option value="stage2_only">Stage 2 Bullish Stack Only</option>
      <option value="near_ath_only">Within 15% 52W High Only</option>
      <option value="all">All RS Scores &amp; Postures</option>
    </select>
  </div>

  <div class="filter-group">
    <label>State:</label>
    <button class="btn-filter-chip active" id="f_state_all" onclick="setStateFilter('all')">All</button>
    <button class="btn-filter-chip" id="f_state_breakout" onclick="setStateFilter('breakout')">Breakouts</button>
    <button class="btn-filter-chip" id="f_state_building" onclick="setStateFilter('building')">Building</button>
  </div>

  <div class="filter-group">
    <label>EP Catalyst:</label>
    <button class="btn-filter-chip active" id="f_ep_all" onclick="setEpFilter('all')">All</button>
    <button class="btn-filter-chip" id="f_ep_ignited" onclick="setEpFilter('ignited')">⚡ Ignited Pole</button>
    <button class="btn-filter-chip" id="f_ep_boosted" onclick="setEpFilter('boosted')">🚀 Boosted</button>
    <button class="btn-filter-chip" id="f_ep_both" onclick="setEpFilter('both')">⚡🚀 Both</button>
    <button class="btn-filter-chip" id="f_ep_none" onclick="setEpFilter('none')">⚪ Pure Technical</button>
  </div>

  <div class="filter-group">
    <label>Thematic Flow:</label>
    <button class="btn-filter-chip active" id="f_macro_all" onclick="setMacroFilter('all')">All</button>
    <button class="btn-filter-chip" id="f_macro_leading_sectors" onclick="setMacroFilter('leading_sectors')">🏭 Leading Sectors</button>
    <button class="btn-filter-chip" id="f_macro_leading_themes" onclick="setMacroFilter('leading_themes')">🚀 Leading Themes</button>
    <button class="btn-filter-chip" id="f_macro_emerging_themes" onclick="setMacroFilter('emerging_themes')">🌟 Emerging Themes</button>
  </div>

  <div class="filter-group">
    <label>Pattern:</label>
    <select id="sel_pattern" onchange="fetchEvents()">
      <option value="all">All Patterns</option>
    </select>
  </div>

  <div class="filter-group">
    <label>Min Move:</label>
    <select id="sel_move" onchange="fetchEvents()">
      <option value="">All Moves (≥ +15%)</option>
      <option value="15">≥ +15% (Large Cap Momentum)</option>
      <option value="20">≥ +20%</option>
      <option value="30">≥ +30% (Standard QM)</option>
      <option value="40">≥ +40%</option>
      <option value="50">≥ +50%</option>
      <option value="75">≥ +75%</option>
      <option value="100">≥ +100%</option>
    </select>
  </div>

  <div class="filter-group">
    <label>Cons Bars:</label>
    <select id="sel_cons" onchange="fetchEvents()">
      <option value="all">All Lengths (3–45 bars)</option>
      <option value="3-7">⚡ 3 – 7 bars (Quick Flags / High Velocity)</option>
      <option value="8-15">8 – 15 bars (Tight)</option>
      <option value="16-25">16 – 25 bars (Medium)</option>
      <option value="26-45">26 – 45 bars (Extended)</option>
    </select>
  </div>

  <div class="filter-group">
    <label>Max Pullback:</label>
    <select id="sel_pullback" onchange="fetchEvents()">
      <option value="">All</option>
      <option value="15">≤ 15%</option>
      <option value="25">≤ 25%</option>
      <option value="35">≤ 35%</option>
      <option value="50">≤ 50%</option>
    </select>
  </div>

  <div class="filter-group">
    <label>BB Squeeze:</label>
    <select id="sel_bb_pinch" onchange="fetchEvents()">
      <option value="all" selected>All</option>
      <option value="tight">⚡ Tight (≤ 2.5x ADR)</option>
      <option value="moderate">Moderate (≤ 3.2x ADR)</option>
      <option value="mild">Mild (≤ 4.0x ADR)</option>
    </select>
  </div>

  <div class="filter-group">
    <label>Pole R²:</label>
    <select id="sel_r2" onchange="fetchEvents()">
      <option value="all" selected>All</option>
      <option value="0.80">🔥 High (R² ≥ 0.80)</option>
      <option value="0.65">Standard (R² ≥ 0.65)</option>
      <option value="0.50">Permissive (R² ≥ 0.50)</option>
    </select>
  </div>

  <div class="filter-group">
    <label>Sector:</label>
    <select id="sel_sector" onchange="onSectorChange()">
      <option value="all">All Sectors</option>
      <option value="leading_sectors">🏭 All Leading Sectors (Score ≥ 65)</option>
    </select>
  </div>

  <div class="filter-group">
    <label>Theme:</label>
    <select id="sel_theme" onchange="onThemeChange()">
      <option value="all">All Themes</option>
      <option value="leading_themes">🚀 All Leading Themes (Score ≥ 65)</option>
      <option value="emerging_themes">🌟 All Emerging Themes (Velocity Surge)</option>
    </select>
  </div>

  <div class="filter-group">
    <input type="text" id="inp_search" placeholder="Search symbol..." style="width:120px;" oninput="onSearchInput()">
  </div>

  <div class="filter-group" id="grp_strat_tier" style="display:flex; align-items:center; gap:5px;">
    <label style="color:#10b981; font-weight:700; font-size:11px;">ML Quality:</label>
    <select id="sel_strat_tier" onchange="onStratTierChange()" style="font-size:11px; border-color:#059669; background:#0f172a; color:#34d399; font-weight:600; border-radius:4px; padding:3px 6px;">
      <option value="all" selected>🌐 All Setups (Unfiltered Universe)</option>
      <option value="avoid_traps">🛡️ Broad System (Avoid Traps — 173 Trades)</option>
      <option value="high_quality">⭐ Active Balanced (Tiers 1-2 — 58 Trades)</option>
      <option value="premier">💎 Apex Conviction (Premier Tier 1 — 16 Trades)</option>
    </select>
  </div>

  <div class="filter-group" id="grp_strat_archetype" style="display:flex; align-items:center; gap:5px;">
    <label style="color:#c084fc; font-weight:700; font-size:11px;">Archetype:</label>
    <select id="sel_trader_archetype" onchange="onTraderArchetypeChange()" style="font-size:11px; border-color:#9333ea; background:#0f172a; color:#c084fc; font-weight:600; border-radius:4px; padding:3px 6px;">
      <option value="custom">Custom Configuration</option>
      <option value="institutional_triad">👑 Institutional Triad: All Trades (518 Trades)</option>
      <option value="institutional_triad_balanced">⭐ Institutional Triad: Active Balanced (58 Trades)</option>
      <option value="institutional_triad_apex">💎 Institutional Triad: Apex Conviction (16 Trades)</option>
      <option value="high_velocity" selected>⚡ High-Velocity Swing (Tight Stops, Climax Partial, Fast Turnover)</option>
      <option value="institutional_hybrid">🏛️ Institutional Scale-Out (Multi-Leg Staged, RS Leaders)</option>
      <option value="position_trend">📈 Position / Trend (Wider Stops, 50 EMA Runners)</option>
    </select>
  </div>

  <div class="filter-group" id="grp_strat_exit" style="display:none; align-items:center; gap:5px;">
    <label style="color:#38bdf8; font-weight:700; font-size:11px;">Exit Model:</label>
    <select id="sel_strat_exit" onchange="onStratExitChange()" style="font-size:11px; border-color:#0284c7; background:#0f172a; color:#38bdf8; font-weight:600; border-radius:4px; padding:3px 6px;">
      <option value="triad_optimal">👑 Institutional Triad (Dynamic Stop + Climax + Conviction)</option>
      <option value="climax_swing" selected>🎯 ML Climax Swing (50% Climax @ P≥0.45, 50% 10 EMA)</option>
      <option value="climax_hybrid">🎯 ML Climax Hybrid (33% Climax @ P≥0.50, 33% 10E, 34% 20E)</option>
      <option value="climax_trend">🎯 ML Climax Trend (50% Climax @ P≥0.65, 50% 50 EMA)</option>
      <option value="10ema">10 EMA Trailing</option>
      <option value="20ema">20 EMA Trailing</option>
      <option value="50ema">50 EMA Trailing</option>
      <option value="staged">Staged Scale (33/33/34)</option>
      <option value="balanced">Balanced Scale (50/50)</option>
    </select>
  </div>

  <div class="filter-group" style="margin-left:auto; display:flex; align-items:center; gap:5px;">
    <button class="btn-strategy" id="btn_strategy_toggle" onclick="toggleStrategyMode()" title="Filter to setups passing QM strategy rules and show aggregate trade stats">
      <span>🎯</span> <span id="strategy_mode_lbl">Strategy Mode: OFF</span>
    </button>
    <button class="btn-strategy-settings" id="btn_strategy_settings" onclick="openStrategySettingsModal()" title="Configure QM Strategy Filtering &amp; Sizing Rules">
      ⚙️
    </button>
  </div>

  <button class="btn-filter-chip" onclick="resetFilters()">Reset Filters</button>
</div>

<!-- Strategy Performance KPI Banner (Visible when Strategy Mode is Active) -->
<div class="strategy-kpi-bar" id="strategy_kpi_bar" style="display:none;">
  <div class="strat-kpi-card" style="border-color:#0284c7;">
    <div class="strat-kpi-label">TOTAL TRADES</div>
    <div class="strat-kpi-val" id="skpi_trades" style="color:#38bdf8;">--</div>
    <div class="strat-kpi-sub" id="skpi_sub_trades">Filtered breakouts</div>
  </div>
  <div class="strat-kpi-card" style="border-color:#059669;">
    <div class="strat-kpi-label">WIN RATE</div>
    <div class="strat-kpi-val" id="skpi_winrate" style="color:#34d399;">--%</div>
    <div class="strat-kpi-sub" id="skpi_sub_winrate">Trailing 10 EMA</div>
  </div>
  <div class="strat-kpi-card" style="border-color:#0284c7;">
    <div class="strat-kpi-label">PROFIT FACTOR</div>
    <div class="strat-kpi-val" id="skpi_pf" style="color:#38bdf8;">--</div>
    <div class="strat-kpi-sub">Gross Win / Gross Loss</div>
  </div>
  <div class="strat-kpi-card" style="border-color:#d97706;">
    <div class="strat-kpi-label">EXPECTANCY (R)</div>
    <div class="strat-kpi-val" id="skpi_expectancy" style="color:#fbbf24;">-- R</div>
    <div class="strat-kpi-sub" id="skpi_sub_expectancy">Per Trade Mean</div>
  </div>
  <div class="strat-kpi-card" style="border-color:#059669;">
    <div class="strat-kpi-label">TOTAL RETURN</div>
    <div class="strat-kpi-val" id="skpi_total_r" style="color:#34d399;">-- R</div>
    <div class="strat-kpi-sub" id="skpi_sub_total_r">Unsized Realized</div>
  </div>
  <div class="strat-kpi-card" style="border-color:#7c3aed;">
    <div class="strat-kpi-label">SIZED RETURN</div>
    <div class="strat-kpi-val" id="skpi_sized_r" style="color:#c084fc;">-- R</div>
    <div class="strat-kpi-sub" id="skpi_sub_sized_r">Dynamic Multiplier</div>
  </div>
  <div class="strat-kpi-card" style="border-color:#0284c7;">
    <div class="strat-kpi-label">WIN / LOSS RATIO</div>
    <div class="strat-kpi-val" id="skpi_win_loss" style="color:#38bdf8;">-- : 1</div>
    <div class="strat-kpi-sub" id="skpi_sub_win_loss">Avg Win vs Avg Loss</div>
  </div>
  <div class="strat-kpi-card" style="border-color:#0284c7;">
    <div class="strat-kpi-label">MAX WIN / MAX LOSS</div>
    <div class="strat-kpi-val" id="skpi_max_win" style="font-size:13px; color:#e2e8f0;">-- R / -- R</div>
    <div class="strat-kpi-sub" id="skpi_sub_max_win">Holding: -- bars</div>
  </div>
</div>

<!-- 3-Pane Workspace -->
<div class="workspace" id="main_workspace">

  <!-- Left: Main Table -->
  <div class="table-pane" id="table_pane">
    <div style="display:flex; justify-content:space-between; align-items:center; padding:6px 10px; background:#11151f; border-bottom:1px solid var(--border); position:sticky; top:0; z-index:15;">
      <span style="font-weight:700; font-size:11.5px; color:#94a3b8;" id="table_setups_count">QM Setups</span>
      <div style="display:flex; gap:6px;">
        <button class="btn-filter-chip" onclick="selectAllSetups()" style="padding:2px 8px; font-size:10.5px;">Select All</button>
        <button class="btn-filter-chip" onclick="clearSelectedSetups()" style="padding:2px 8px; font-size:10.5px;">Clear</button>
      </div>
    </div>
    <table>
      <thead>
        <tr>
          <th style="width:28px; text-align:center;">
            <input type="checkbox" id="header_select_all" onchange="toggleSelectAll(this.checked)" title="Select/Deselect All Visible Setups" class="setup-cb">
          </th>
          <th style="width:38px; text-align:center;">#</th>
          <th class="tl" onclick="sortTable('symbol')">Symbol</th>
          <th class="tl" onclick="sortTable('date')">Date</th>
          <th class="tl" onclick="sortTable('state')">State</th>
          <th onclick="sortTable('rs_score')" style="color:#38bdf8; cursor:pointer;" title="1-99 Institutional Relative Strength Score & Posture">RS ↕</th>
          <th onclick="sortTable('move_pct')">Pole Move</th>
          <th onclick="sortTable('cons_bars')">Cons</th>
          <th onclick="sortTable('pullback_pct')">Pullback</th>
          <th onclick="sortTable('r2')" title="Pole Linearity R² (0.00-1.00)">R²</th>
          <th onclick="sortTable('pinch_ratio')" title="Base Bollinger Band Squeeze (Min BB Width / ADR20)">BB Squeeze</th>
          <th class="tl" onclick="sortTable('pattern')">Pattern</th>
          <th class="tl" onclick="sortTable('ep_badge')">EP Catalyst Link</th>
        </tr>
      </thead>
      <tbody id="events_tbody">
        <tr><td colspan="13" style="text-align:center; padding:30px; color:var(--muted);">Loading QM setups...</td></tr>
      </tbody>
    </table>
  </div>

  <!-- Center: Chart Pane -->
  <div class="chart-pane" id="chart_pane">
    <div class="chart-header">
      <div class="chart-title-box">
        <span id="chart_sym" style="color:#38bdf8;">--</span>
        <span id="chart_date" style="font-size:12px; color:var(--muted); font-weight:400;">--</span>
        <span id="chart_state_pill"></span>
        <span id="chart_ep_badge"></span>
      </div>
      <div class="chart-controls">
        <div class="chart-enlarged-only-controls">
          <button id="btn_enlarged_stock_list" class="btn-chart-enlarge" onclick="openEnlargedStockListModal()" title="Choose a different stock from current list (Ctrl/Cmd+K)">
            📋 Choose Stock (<span id="lbl_enlarged_stock_count">0</span>)
          </button>
          <div style="display:inline-flex; align-items:center; gap:2px;">
            <button id="btn_chart_prev_ticker" class="btn-chart-enlarge" onclick="stepFilteredStock(-1)" title="Previous setup in list (Arrow Left / Arrow Up)" style="padding:3px 8px;">◀ Prev</button>
            <button id="btn_chart_next_ticker" class="btn-chart-enlarge" onclick="stepFilteredStock(1)" title="Next setup in list (Arrow Right / Arrow Down)" style="padding:3px 8px;">Next ▶</button>
          </div>
        </div>
        <button id="btn_chart_settings" class="btn-chart-enlarge" onclick="toggleChartSettings()" title="Configure Chart Displays, Markers, and Parameters">⚙️ Settings</button>
        <button id="btn_toggle_draw" class="btn-chart-enlarge" onclick="toggleDrawToolbar()" title="Toggle on-chart drawing tools (Measure, Line, Text, Arrow)">✏️ Tools</button>
        <button class="btn-chart-enlarge" onclick="resetChartScale()" title="Auto-scale Y-axis and reset view">⟲ Reset</button>
        <button id="btn_chart_fullscreen" class="btn-chart-enlarge btn-chart-expand-main" onclick="toggleChartFullscreen()" title="Toggle full-screen chart display (Esc to exit)" style="flex-shrink:0;">⛶ Enlarge</button>
      </div>
    </div>

    <div class="chart-container-rel">
      <div id="chart_hud" class="chart-hud" style="display:none;"></div>
      <div id="chart_tv"></div>
      <canvas id="chart_canvas_overlay"></canvas>

      <svg id="larsson_ribbon_svg" style="position:absolute; inset:0; width:100%; height:100%; pointer-events:none; z-index:12; overflow:hidden;">
        <g id="larsson_ribbon_group"></g>
      </svg>
      <svg id="draw_canvas" style="position:absolute; inset:0; width:100%; height:100%; pointer-events:none; z-index:20; overflow:hidden;">
        <defs>
          <marker id="arrowhead" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">
            <polygon points="0 0, 9 3.5, 0 7" fill="#38bdf8" />
          </marker>
        </defs>
        <g id="draw_elements"></g>
        <g id="draw_preview"></g>
      </svg>

      <!-- Floating Chart Drawing Toolbar -->
      <div id="chart_draw_toolbar" class="chart-draw-toolbar" style="display:none;">
        <button type="button" class="draw-btn active" id="btn_tool_cursor" onclick="setDrawTool('cursor')" title="Pan &amp; Zoom Chart (Normal Mode)"><span style="font-size:12px;">🖐️</span> Pan</button>
        <button type="button" class="draw-btn" id="btn_tool_measure" onclick="setDrawTool('measure')" title="Measure Price % &amp; Bars (Click 2 points)"><span style="font-size:12px;">📏</span> Measure</button>
        <button type="button" class="draw-btn" id="btn_tool_line" onclick="setDrawTool('line')" title="Trendline (Click start &amp; end)"><span style="font-size:12px;">📈</span> Line</button>
        <button type="button" class="draw-btn" id="btn_tool_text" onclick="setDrawTool('text')" title="Text Annotation (Click to place note)"><span style="font-size:12px;">🔤</span> Text</button>
        <button type="button" class="draw-btn" id="btn_tool_arrow" onclick="setDrawTool('arrow')" title="Arrow Pointer (Click tail &amp; head)"><span style="font-size:12px;">↗️</span> Arrow</button>
        <div style="width:1px; height:18px; background:#334155; margin:0 2px;"></div>
        <button type="button" class="draw-btn" id="btn_tool_settings" onclick="toggleChartSettings()" title="Configure Chart Displays, Markings &amp; Parameters"><span style="font-size:12px;">⚙️</span> Settings</button>
        <div style="width:1px; height:18px; background:#334155; margin:0 2px;"></div>
        <button type="button" class="draw-btn draw-btn-danger" onclick="clearAllDrawings()" title="Clear all drawings on chart"><span style="font-size:12px;">🗑️</span> Clear</button>
      </div>

      <!-- Floating Chart Settings Panel -->
      <div id="chart_settings_panel" class="chart-settings-panel" style="display:none;">
        <div class="settings-panel-header">
          <span style="font-weight:700; color:#f8fafc; font-size:12px; display:flex; align-items:center; gap:6px;">
            <span>⚙️</span> Chart &amp; Indicator Settings
          </span>
          <button type="button" class="panel-close-btn" onclick="toggleChartSettings()" title="Close Settings">&times;</button>
        </div>
        <div class="settings-panel-body">
          <div class="settings-section">
            <div class="settings-sec-title">📈 MOVING AVERAGES &amp; RIBBONS</div>
            <div class="settings-row">
              <label class="toggle-control" style="margin:0; font-size:11px;">
                <input type="checkbox" id="chk_settings_ema10" checked onchange="toggleEma10()">
                <span class="toggle-label" style="color:#38bdf8;">10 EMA (Momentum)</span>
              </label>
            </div>
            <div class="settings-row">
              <label class="toggle-control" style="margin:0; font-size:11px;">
                <input type="checkbox" id="chk_settings_ema20" checked onchange="toggleEma20()">
                <span class="toggle-label" style="color:#eab308;">20 EMA (Swing baseline)</span>
              </label>
            </div>
            <div class="settings-row">
              <label class="toggle-control" style="margin:0; font-size:11px;">
                <input type="checkbox" id="chk_settings_sma50" checked onchange="toggleSma50()">
                <span class="toggle-label" style="color:#ef4444;">50 SMA (Institutional trend)</span>
              </label>
            </div>
            <div class="settings-row">
              <label class="toggle-control" style="margin:0; font-size:11px;">
                <input type="checkbox" id="chk_settings_sma200" checked onchange="toggleSma200()">
                <span class="toggle-label" style="color:#64748b;">200 SMA (Long-term anchor)</span>
              </label>
            </div>
            <div class="settings-row">
              <label class="toggle-control" style="margin:0; font-size:11px;">
                <input type="checkbox" id="chk_settings_vol_ma20" checked onchange="toggleVolMa20()">
                <span class="toggle-label" style="color:#38bdf8;">20 Day Volume MA</span>
              </label>
            </div>
            <div class="settings-row" style="margin-top:2px;">
              <label class="toggle-control" style="margin:0; font-size:11px;">
                <input type="checkbox" id="chk_settings_larsson" checked onchange="toggleLarsson()">
                <span class="toggle-label" style="color:#facc15; font-weight:700;">Larsson Line Ribbon</span>
              </label>
            </div>
            <div class="settings-row" style="margin-top:6px; justify-content:space-between; align-items:center; background:#111827; padding:4px 8px; border-radius:4px; border:1px solid #1f2937;">
              <div style="display:flex; align-items:center; gap:6px;">
                <span style="font-size:11px; color:#cbd5e1; font-weight:600;">Fastest EMA:</span>
                <input type="number" id="cfg_larsson_fastest" value="32" min="3" max="80" style="width:42px; background:#1e293b; color:#38bdf8; border:1px solid #334155; border-radius:3px; padding:2px 4px; font-size:11px; text-align:center; font-weight:700;" onchange="onLarssonFastestChange()">
              </div>
              <div style="display:flex; align-items:center; gap:6px;">
                <span id="lbl_larsson_spans" style="color:#64748b; font-size:10px; font-weight:600;">(32, 35, 50, 58)</span>
                <span id="lbl_larsson_vals" style="font-size:11px; font-weight:700; display:inline-flex; gap:6px;">
                  <span id="larsson_val_fast" style="color:#facc15;">—</span>
                  <span id="larsson_val_slow" style="color:#38bdf8;">—</span>
                </span>
              </div>
            </div>
          </div>

          <div class="settings-section">
            <div class="settings-sec-title">📐 QM GEOMETRY &amp; VISUALS</div>
            <div class="settings-row">
              <label class="toggle-control" style="margin:0; font-size:11px;" title="Shaded Emerald Pole Box depicting impulse move">
                <input type="checkbox" id="chk_settings_pole_box" checked onchange="togglePoleBox()">
                <span class="toggle-label" style="color:#34d399;">QM Pole Box</span>
              </label>
            </div>
            <div class="settings-row">
              <label class="toggle-control" style="margin:0; font-size:11px;" title="Shaded Amber Base Box depicting consolidation surfing">
                <input type="checkbox" id="chk_settings_cons_box" checked onchange="toggleConsBox()">
                <span class="toggle-label" style="color:#fbbf24;">QM Base Box</span>
              </label>
            </div>
            <div class="settings-row">
              <label class="toggle-control" style="margin:0; font-size:11px;" title="Horizontal Trigger and Support/Resistance price lines">
                <input type="checkbox" id="chk_settings_price_lines" checked onchange="togglePriceLines()">
                <span class="toggle-label" style="color:#a855f7;">Breakout &amp; Stack Price Lines</span>
              </label>
            </div>
            <div class="settings-row">
              <label class="toggle-control" style="margin:0; font-size:11px;" title="Markers for Prior and Post Episodic Pivots">
                <input type="checkbox" id="chk_settings_ep_markers" checked onchange="toggleEpMarkers()">
                <span class="toggle-label" style="color:#06b6d4;">EP Catalyst Markers</span>
              </label>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Right: Dossier Pane -->
  <div class="dossier-pane" id="dossier_pane">
    <!-- Quick Title Card -->
    <div class="dossier-card">
      <div class="dossier-title">
        <span>OVERVIEW</span>
        <span id="dos_status_pill"></span>
      </div>
      <div class="dossier-row">
        <span class="dossier-label">Symbol / Date:</span>
        <span class="dossier-val" id="dos_sym_date">--</span>
      </div>
      <div class="dossier-row">
        <span class="dossier-label">Sector:</span>
        <span class="dossier-val" id="dos_sector" style="cursor:pointer; text-decoration:underline dotted; color:#38bdf8;" title="Click to view Sector Stock Leaderboard" onclick="openGroupStocksModal(this.textContent, 'sector')">--</span>
      </div>
      <div class="dossier-row">
        <span class="dossier-label">Themes:</span>
        <span class="dossier-val" id="dos_themes" style="font-size:11px; max-width:220px; text-align:right; cursor:pointer; text-decoration:underline dotted; color:#c084fc;" title="Click to view Theme Stock Leaderboard" onclick="openGroupStocksModal(this.textContent, 'theme')">--</span>
      </div>
    </div>

    <!-- Dedicated ML Setup Quality & Dynamic Trade Execution Card -->
    <div class="dossier-card" id="card_ml_guidance" style="border-color:#10b981; background:linear-gradient(180deg, #071914 0%, #061210 100%); display:none;">
      <div class="dossier-title">
        <span style="color:#34d399; font-weight:700;">🧠 ML SETUP QUALITY &amp; GUIDANCE</span>
        <span id="dos_ml_tier_badge" class="pctile-badge pctile-high">--</span>
      </div>

      <div style="background:#04140e; border:1px solid #064e3b; border-radius:6px; padding:6px 10px; margin-bottom:6px;">
        <div style="font-size:11px; font-weight:700; color:#6ee7b7;" id="dos_ml_rec_text">--</div>
      </div>

      <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-bottom:6px;">
        <div style="background:#082218; border:1px solid #047857; border-radius:6px; padding:5px 8px; text-align:center;">
          <div style="font-size:9.5px; color:#a7f3d0; text-transform:uppercase; font-weight:700;">QM Quality Score</div>
          <div style="font-size:16px; font-weight:800; color:#34d399;" id="dos_ml_score">-- / 100</div>
          <div style="font-size:9.5px; color:#6ee7b7;" id="dos_ml_sizing">--x Conviction</div>
        </div>
        <div style="background:#082218; border:1px solid #047857; border-radius:6px; padding:5px 8px; text-align:center;">
          <div style="font-size:9.5px; color:#a7f3d0; text-transform:uppercase; font-weight:700;">Dynamic Stop Loss</div>
          <div style="font-size:15px; font-weight:800; color:#f87171;" id="dos_ml_stop">$--</div>
          <div style="font-size:9.5px; color:#fca5a5;" id="dos_ml_risk">--% risk</div>
        </div>
      </div>

      <div style="display:flex; justify-content:space-between; font-size:10.5px; color:#cbd5e1; background:#051911; padding:5px 8px; border-radius:4px; border:1px solid #064e3b;">
        <span>P(Big Move ≥ 3R): <strong style="color:#34d399;" id="dos_ml_p_big">--%</strong></span>
        <span>P(Trap Failure): <strong style="color:#f87171;" id="dos_ml_p_trap">--%</strong></span>
      </div>

      <!-- Trade Management & Scaling Guidance -->
      <div style="margin-top:6px; background:#071a13; border:1px solid #064e3b; border-radius:6px; padding:6px 8px;">
        <div style="font-size:9.5px; color:#a7f3d0; text-transform:uppercase; font-weight:700; margin-bottom:4px;">🎯 Pre-Trade Hurdle Targets (At Entry via Dynamic Stop)</div>
        <div style="display:flex; justify-content:space-between; font-size:10.5px; color:#e2e8f0; margin-bottom:4px;">
          <span>Target 1 (+2.5R Partial): <strong style="color:#38bdf8;" id="dos_ml_hurdle_25r">$--</strong></span>
          <span>Target 2 (+3.5R Climax): <strong style="color:#a855f7;" id="dos_ml_hurdle_35r">$--</strong></span>
        </div>
        <div style="font-size:9.5px; color:#6ee7b7; line-height:1.3;" id="dos_ml_plan">Sell 50% partial into Climax High (P ≥ 0.45 or +2.5R), move stop to Breakeven, trail 50% runner on 10 EMA.</div>
      </div>
    </div>

    <!-- QM Breakout Strategy Trade Execution & Conviction Sizing Card -->
    <div class="dossier-card" id="card_strategy_trade" style="border-color:#0284c7; background:linear-gradient(180deg, #0d1527 0%, #0b111e 100%); display:none;">
      <div class="dossier-title">
        <span style="color:#38bdf8; font-weight:700;">🎯 QM STRATEGY TRADE EXECUTION</span>
        <span id="dos_strat_result_badge" class="pctile-badge pctile-high">--</span>
      </div>

      <!-- Execution Grid -->
      <div style="background:#070d18; border:1px solid #1e293b; border-radius:6px; padding:8px 10px; display:flex; flex-direction:column; gap:4px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span style="color:var(--muted); font-size:11px;">Breakout Entry:</span>
          <span style="font-weight:700; color:#fff;" id="dos_strat_entry">--</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span style="color:var(--muted); font-size:11px;">Stop Loss Level:</span>
          <span style="font-weight:700; color:#f87171;" id="dos_strat_stop">--</span>
        </div>
        <div id="dos_strat_partial_row" style="display:none; justify-content:space-between; align-items:center; background:rgba(147, 51, 234, 0.15); border:1px solid #7c3aed; padding:3px 6px; border-radius:4px;">
          <span style="color:#d8b4fe; font-size:10.5px; font-weight:700;">🎯 Climax Partial:</span>
          <span style="font-weight:700; color:#c084fc; font-size:11px;" id="dos_strat_partial_exit">--</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span style="color:var(--muted); font-size:11px;" id="dos_strat_exit_lbl">Trailing Exit:</span>
          <span style="font-weight:700; color:#38bdf8;" id="dos_strat_exit">--</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span style="color:var(--muted); font-size:11px;">Exit Reason:</span>
          <span style="font-size:11px; color:#cbd5e1;" id="dos_strat_exit_reason">--</span>
        </div>
      </div>

      <!-- Return & Excursion Performance -->
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px;">
        <div style="background:#0d1829; border:1px solid #1e3a5f; border-radius:6px; padding:6px 8px; text-align:center;">
          <div style="font-size:9.5px; color:#93c5fd; text-transform:uppercase; font-weight:700;">Realized Return (1.0x)</div>
          <div style="font-size:15px; font-weight:800;" id="dos_strat_r">-- R</div>
          <div style="font-size:10px; color:var(--muted);" id="dos_strat_gain_pct">--% gain</div>
        </div>
        <div style="background:#1b142f; border:1px solid #4c1d95; border-radius:6px; padding:6px 8px; text-align:center;">
          <div style="font-size:9.5px; color:#c084fc; text-transform:uppercase; font-weight:700;">Sized Return (Dynamic)</div>
          <div style="font-size:15px; font-weight:800; color:#c084fc;" id="dos_strat_sized_r">-- R</div>
          <div style="font-size:10px; color:#ddd6fe;" id="dos_strat_size_mult">Conviction: --x</div>
        </div>
      </div>

      <div style="display:flex; justify-content:space-between; font-size:10.5px; color:var(--muted); padding:2px 2px;">
        <span>Peak MFE: <strong style="color:#34d399;" id="dos_strat_mfe">-- R</strong></span>
        <span>Max Drawdown MAE: <strong style="color:#f87171;" id="dos_strat_mae">-- R</strong></span>
        <span>Holding: <strong style="color:#e2e8f0;" id="dos_strat_bars">-- bars</strong></span>
      </div>

      <!-- Multi-Factor Conviction Sizing Breakdown -->
      <div style="background:#080e1b; border:1px solid #1e293b; border-radius:6px; padding:8px 10px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
          <span style="font-size:10.5px; font-weight:700; color:#c084fc;">⚡ CONVICTION SIZING FACTORS</span>
          <span style="font-size:10.5px; font-weight:700; color:#fff;" id="dos_strat_total_mult_badge">1.00x Base</span>
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:4px; font-size:10px;">
          <div style="background:#111928; padding:4px 6px; border-radius:4px; display:flex; justify-content:space-between;">
            <span style="color:var(--muted);">Fresh EP:</span>
            <strong id="dos_strat_s_ep" style="color:#38bdf8;">1.00x</strong>
          </div>
          <div style="background:#111928; padding:4px 6px; border-radius:4px; display:flex; justify-content:space-between;">
            <span style="color:var(--muted);">Pole Move:</span>
            <strong id="dos_strat_s_pole" style="color:#34d399;">1.00x</strong>
          </div>
          <div style="background:#111928; padding:4px 6px; border-radius:4px; display:flex; justify-content:space-between;">
            <span style="color:var(--muted);">Breakout RVOL:</span>
            <strong id="dos_strat_s_rvol" style="color:#f59e0b;">1.00x</strong>
          </div>
          <div style="background:#111928; padding:4px 6px; border-radius:4px; display:flex; justify-content:space-between;">
            <span style="color:var(--muted);">Pullback Tight:</span>
            <strong id="dos_strat_s_pb" style="color:#a855f7;">1.00x</strong>
          </div>
        </div>
      </div>

      <!-- Market Regime & Alignment Checklist -->
      <div style="font-size:11px; color:#cbd5e1; display:flex; flex-direction:column; gap:3px; background:#070d18; border:1px solid #1e293b; border-radius:6px; padding:6px 8px;">
        <div style="display:flex; justify-content:space-between;">
          <span style="color:var(--muted); font-size:10.5px;">Market Trend (SPY):</span>
          <span id="dos_strat_spy_status">--</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
          <span style="color:var(--muted); font-size:10.5px;">QQQ 10/20 EMA Regime:</span>
          <span id="dos_strat_qqq_status">--</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
          <span style="color:var(--muted); font-size:10.5px;">Tech Leadership (QQQ/SPY):</span>
          <span id="dos_strat_tech_status">--</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
          <span style="color:var(--muted); font-size:10.5px;">Macro Universe:</span>
          <span id="dos_strat_macro_status">--</span>
        </div>
      </div>
    </div>

    <!-- Sector & Theme Momentum Backdrop -->
    <div class="dossier-card" id="card_sec_thm">
      <div class="dossier-title">
        <span>🌐 SECTOR &amp; THEME BACKDROP</span>
        <span id="ctx_diag_badge" class="pctile-badge pctile-high">Tailwind</span>
      </div>
      <div class="ctx-row">
        <div class="ctx-item">
          <div><span style="color:var(--muted); font-size:10px; text-transform:uppercase;">Sector: </span><span class="ctx-tag" id="ctx_sec_name" style="cursor:pointer; text-decoration:underline dotted; color:#38bdf8;" title="Click to view Sector Stock Leaderboard" onclick="openGroupStocksModal(this.textContent, 'sector')">Technology</span></div>
          <div class="ctx-pctiles">
            <span class="pctile-badge pctile-high" id="ctx_sec_m1">1M: --th</span>
            <span class="pctile-badge pctile-high" id="ctx_sec_m3">3M: --th</span>
          </div>
        </div>
        <div class="ctx-item">
          <div><span style="color:var(--muted); font-size:10px; text-transform:uppercase;">Theme: </span><span class="ctx-tag" id="ctx_thm_name" style="cursor:pointer; text-decoration:underline dotted; color:#c084fc;" title="Click to view Theme Stock Leaderboard" onclick="openGroupStocksModal(this.textContent, 'theme')">Semiconductors</span></div>
          <div class="ctx-pctiles">
            <span class="pctile-badge pctile-high" id="ctx_thm_m1">1M: --th</span>
          </div>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; font-size:11px; margin-top:2px; color:var(--muted);">
          <span>Archetype: <strong id="ctx_archetype" style="color:#e2e8f0;">Market Baseline</strong></span>
          <span>Composite: <strong id="ctx_composite" style="color:#38bdf8;">50.0</strong></span>
        </div>
        <div class="diag-pill diag-tailwind" id="ctx_diag_desc">
          Top decile institutional money flow sponsorship. High probability of multi-quarter continuation.
        </div>
      </div>
    </div>

    <!-- Institutional Relative Strength & Posture Card -->
    <div class="dossier-card" id="card_rs_technicals" style="border-color:#0284c7;">
      <div class="dossier-title">
        <span>⚡ INSTITUTIONAL RELATIVE STRENGTH</span>
        <span id="dos_rs_badge" class="pctile-badge pctile-high">RS --</span>
      </div>
      <div class="dossier-row">
        <span class="dossier-label">Cross-Sectional RS Score:</span>
        <span class="dossier-val" id="dos_rs_score" style="font-weight:700; color:#38bdf8;">--</span>
      </div>
      <div class="dossier-row">
        <span class="dossier-label">Stage 2 Bullish Stack:</span>
        <span class="dossier-val" id="dos_stage2">--</span>
      </div>
      <div class="dossier-row">
        <span class="dossier-label">Distance from 52W High:</span>
        <span class="dossier-val" id="dos_dist_52w">--</span>
      </div>
      <div class="dossier-row">
        <span class="dossier-label">Posture Status:</span>
        <span class="dossier-val" id="dos_rs_posture_status">--</span>
      </div>
    </div>

    <!-- QM Technical Profile -->
    <div class="dossier-card">
      <div class="dossier-title">
        <span>QM TECHNICAL PROFILE</span>
        <span id="dos_pattern" style="color:#38bdf8;">--</span>
      </div>
      <div class="dossier-row">
        <span class="dossier-label">Momentum Pole Move:</span>
        <span class="dossier-val" style="color:#34d399;" id="dos_pole_move">--</span>
      </div>
      <div class="dossier-row">
        <span class="dossier-label">Pole Dates:</span>
        <span class="dossier-val" style="font-family:monospace; font-size:11px;" id="dos_pole_dates">--</span>
      </div>
      <div class="dossier-row">
        <span class="dossier-label">Consolidation Duration:</span>
        <span class="dossier-val" id="dos_cons_bars">--</span>
      </div>
      <div class="dossier-row">
        <span class="dossier-label">Max Base Pullback:</span>
        <span class="dossier-val" id="dos_pullback">--</span>
      </div>
      <div class="dossier-row">
        <span class="dossier-label">EMA20 Surfing Fraction:</span>
        <span class="dossier-val" id="dos_ride_frac">--</span>
      </div>
      <div class="dossier-row">
        <span class="dossier-label">Intraday Shakeouts:</span>
        <span class="dossier-val" id="dos_shakeouts">--</span>
      </div>
      <div class="dossier-row">
        <span class="dossier-label">Breakout Volume Surge:</span>
        <span class="dossier-val" id="dos_vol_x">--</span>
      </div>
      <div class="dossier-row">
        <span class="dossier-label">Consolidation Low (Support):</span>
        <span class="dossier-val" style="color:#f87171;" id="dos_level">--</span>
      </div>
      <div class="dossier-row">
        <span class="dossier-label">Trigger Resistance:</span>
        <span class="dossier-val" style="color:#34d399;" id="dos_trigger">--</span>
      </div>
      <div class="dossier-row">
        <span class="dossier-label">Resistance Stack:</span>
        <span class="dossier-val" style="font-family:monospace; font-size:11px;" id="dos_stack">--</span>
      </div>
      <div class="dossier-row">
        <span class="dossier-label">Pole Linearity (R²):</span>
        <span class="dossier-val" style="color:#38bdf8; font-weight:700;" id="dos_r2">--</span>
      </div>
      <div class="dossier-row">
        <span class="dossier-label">BB Squeeze (Pinch Ratio):</span>
        <span class="dossier-val" style="color:#fbbf24; font-weight:700;" id="dos_pinch_ratio">--</span>
      </div>
    </div>

    <!-- EP Catalyst Linkage Card -->
    <div class="dossier-card" id="dos_ep_card" style="border-color:#0284c7;">
      <div class="dossier-title">
        <span>⚡ EP CATALYST LINKAGE</span>
        <span id="dos_ep_badge">--</span>
      </div>
      
      <!-- Prior EP Section -->
      <div id="dos_prior_ep_sec" style="display:flex; flex-direction:column; gap:4px; margin-top:2px;">
        <div style="font-size:11px; font-weight:700; color:#38bdf8;">⚡ Prior EP Igniter:</div>
        <div class="dossier-row">
          <span class="dossier-label">Proximity:</span>
          <span class="dossier-val" style="color:#38bdf8;" id="dos_prior_days">--</span>
        </div>
        <div class="dossier-row">
          <span class="dossier-label">EP Date:</span>
          <span class="dossier-val" id="dos_prior_date">--</span>
        </div>
        <div class="dossier-row">
          <span class="dossier-label">Gap % / RVOL:</span>
          <span class="dossier-val" id="dos_prior_stats">--</span>
        </div>
        <div class="dossier-row">
          <span class="dossier-label">Subtype / Catalyst:</span>
          <span class="dossier-val" style="font-size:11px; color:#cbd5e1; text-align:right;" id="dos_prior_cat">--</span>
        </div>
      </div>

      <!-- Post EP Section -->
      <div id="dos_post_ep_sec" style="display:flex; flex-direction:column; gap:4px; margin-top:8px; border-top:1px dashed #334155; padding-top:6px;">
        <div style="font-size:11px; font-weight:700; color:#c084fc;">🚀 Post EP Continuation Rocket:</div>
        <div class="dossier-row">
          <span class="dossier-label">Proximity:</span>
          <span class="dossier-val" style="color:#c084fc;" id="dos_post_days">--</span>
        </div>
        <div class="dossier-row">
          <span class="dossier-label">EP Date:</span>
          <span class="dossier-val" id="dos_post_date">--</span>
        </div>
        <div class="dossier-row">
          <span class="dossier-label">Gap % / RVOL:</span>
          <span class="dossier-val" id="dos_post_stats">--</span>
        </div>
      </div>

      <!-- Catalyst Intelligence Section -->
      <div id="dos_cat_intel_sec" style="display:none; flex-direction:column; gap:6px; margin-top:8px; border-top:1px dashed #334155; padding-top:6px;">
        <div style="font-size:11px; font-weight:700; color:#38bdf8;">🏛️ SEC EDGAR &amp; Catalyst Intelligence:</div>
        <div style="font-size:11px; font-weight:600; color:#fff;" id="dos_cat_headline">--</div>
        <div style="font-size:10.5px; color:#cbd5e1; background:#0b0e14; padding:6px 8px; border-radius:4px; border:1px solid #1e283d;" id="dos_cat_analysis">--</div>
        <div id="dos_cat_links" style="display:flex; flex-wrap:wrap; gap:6px; font-size:10.5px; margin-top:2px;"></div>
      </div>

      <!-- Pure Technical Section -->
      <div id="dos_pure_tech_sec" style="display:none; font-size:11.5px; color:#94a3b8; line-height:1.4;">
        ⚪ <strong>Pure Technical QM:</strong> This stock coiled and surfed the moving averages based on pure market accumulation without a major corporate Episodic Pivot earnings/catalyst gap in the ±60-day window.
      </div>
    </div>
  </div>

</div>

<!-- Sector & Theme Stock Leaderboard Modal -->
<div class="modal-overlay" id="group_stocks_modal" onclick="closeGroupStocks(event)">
  <div class="modal-box" onclick="event.stopPropagation()">
    <div class="modal-header">
      <div class="modal-title">
        <span id="gs_title">📊 Stock Leaderboard</span>
        <span class="badge" id="gs_badge" style="font-size:11px; margin-left:8px; background:#1e293b; color:#cbd5e1;">0 Stocks</span>
      </div>
      <button class="modal-close" onclick="closeGroupStocks()">&times;</button>
    </div>
    <div class="modal-body" id="gs_body" style="padding:16px;">
      <div style="text-align:center; padding:30px; color:var(--muted);">Loading constituent stocks...</div>
    </div>
  </div>
</div>

<!-- Filtered QM Setups Modal for Enlarged Chart Navigation -->
<div class="modal-overlay" id="enlarged_stock_list_modal" onclick="closeEnlargedStockListModal(event)">
  <div class="modal-box" style="width:1120px; max-width:96vw; height:85vh; display:flex; flex-direction:column;" onclick="event.stopPropagation()">
    <div class="modal-header" style="padding:10px 18px; display:flex; justify-content:space-between; align-items:center; background:#161d2b; border-bottom:1px solid #232f42;">
      <div style="display:flex; align-items:center; gap:12px; flex:1; flex-wrap:wrap;">
        <span style="font-size:15px; font-weight:700; color:#fff; display:flex; align-items:center; gap:6px;">
          📋 Filtered QM Setups (<span id="esl_count" style="color:#38bdf8;">0</span>)
        </span>
        <div style="position:relative; width:280px;">
          <input type="text" id="esl_search" placeholder="Quick search symbol, pattern, sector, theme..." 
                 oninput="filterEnlargedStockList(this.value)" 
                 onkeydown="if(event.key==='Enter'){ const r=document.querySelector('#esl_table_body tr'); if(r) r.click(); }"
                 style="width:100%; background:#0c1017; border:1px solid #334155; border-radius:4px; padding:5px 10px 5px 28px; color:#fff; font-size:12px; outline:none;">
          <span style="position:absolute; left:8px; top:6px; font-size:12px; color:#64748b;">🔍</span>
        </div>
        <span style="font-size:11px; color:#94a3b8;" id="esl_filter_summary">Current Active Filters</span>
      </div>
      <button class="modal-close" onclick="closeEnlargedStockListModal()" style="font-size:22px; line-height:1; padding:2px 8px; cursor:pointer;">&times;</button>
    </div>
    <div style="background:#0f172a; padding:6px 18px; border-bottom:1px solid #1e293b; font-size:11px; color:#94a3b8; display:flex; justify-content:space-between; align-items:center;">
      <span>Click any row to switch chart and dossier immediately. Active setup is highlighted.</span>
      <span>Tip: Press <kbd style="background:#1e293b; padding:1px 4px; border-radius:3px; color:#cbd5e1; border:1px solid #334155;">Esc</kbd> to close · <kbd style="background:#1e293b; padding:1px 4px; border-radius:3px; color:#cbd5e1; border:1px solid #334155;">Enter</kbd> to select first · ◀ / ▶ to step</span>
    </div>
    <div style="flex:1; overflow-y:auto; padding:0; background:#0e121a;">
      <table class="esl-table">
        <thead style="position:sticky; top:0; z-index:10; background:#161e2e; border-bottom:1px solid #283548;">
          <tr>
            <th class="tl" onclick="sortEslBy('symbol')" style="cursor:pointer;" title="Sort by Symbol">Symbol ↕</th>
            <th class="tl" onclick="sortEslBy('date')" style="cursor:pointer;" title="Sort by Date">Date ↕</th>
            <th class="tl" onclick="sortEslBy('state')" style="cursor:pointer; width:90px;" title="Sort by State">State ↕</th>
            <th onclick="sortEslBy('rs_score')" style="cursor:pointer; color:#38bdf8;" title="Sort by Relative Strength Score & Posture">RS ↕</th>
            <th onclick="sortEslBy('move_pct')" style="cursor:pointer;" title="Sort by Pole Move %">Move % ↕</th>
            <th onclick="sortEslBy('cons_bars')" style="cursor:pointer;" title="Sort by Consolidation Bars">Cons ↕</th>
            <th onclick="sortEslBy('pullback_pct')" style="cursor:pointer;" title="Sort by Pullback %">Pullback ↕</th>
            <th class="tl" onclick="sortEslBy('pattern')" style="cursor:pointer;" title="Sort by Pattern Shape">Pattern ↕</th>
            <th class="tl" title="EP Relationship">EP Catalyst Badge</th>
            <th class="tl" onclick="sortEslBy('sector')" style="cursor:pointer;" title="Sort by Sector">Sector ↕</th>
            <th class="tl" onclick="sortEslBy('theme')" style="cursor:pointer;" title="Sort by Theme">Theme ↕</th>
          </tr>
        </thead>
        <tbody id="esl_table_body">
          <!-- Dynamically populated -->
        </tbody>
      </table>
    </div>
  </div>
</div>

<!-- Strategy Settings Modal -->
<div class="modal-overlay" id="strategy_settings_modal" onclick="closeStrategySettingsModal(event)">
  <div class="modal-box" style="width:680px; max-width:92vw; max-height:88vh; display:flex; flex-direction:column;" onclick="event.stopPropagation()">
    <div class="modal-header" style="background:#0f172a; border-bottom:1px solid #1e293b; padding:12px 18px;">
      <div class="modal-title" style="color:#38bdf8; font-size:14px; display:flex; align-items:center; gap:8px;">
        <span>🎯 Institutional QM Strategy Filtering &amp; Sizing Rules</span>
      </div>
      <button class="modal-close" onclick="closeStrategySettingsModal()">&times;</button>
    </div>
    <div class="modal-body" style="background:#090d16; padding:16px 20px; display:flex; flex-direction:column; gap:12px; overflow-y:auto;">
      <!-- Trader Archetype Quick Presets -->
      <div class="settings-group" style="background:#1e1b4b; border:1px solid #6366f1; border-radius:6px; padding:10px;">
        <div class="settings-group-title" style="color:#c7d2fe;">
          <span>⚡ Trader Archetype Presets</span>
        </div>
        <div class="settings-group-desc" style="color:#a5b4fc;">Instantly configure institutional strategy settings and filters for your trading style.</div>
        <select id="cfg_trader_archetype" class="cfg-select" onchange="applyArchetypePreset(this.value)" style="border-color:#818cf8; background:#0f172a; color:#e0e7ff; font-weight:600;">
          <option value="custom">Custom Configuration</option>
          <option value="institutional_triad">👑 Institutional Triad: All Trades (518 Trades)</option>
          <option value="institutional_triad_balanced">⭐ Institutional Triad: Active Balanced (58 Trades)</option>
          <option value="institutional_triad_apex">💎 Institutional Triad: Apex Conviction (16 Trades)</option>
          <option value="high_velocity" selected>⚡ High-Velocity Swing (Tight Stops, Climax Partial, Fast Turnover)</option>
          <option value="institutional_hybrid">🏛️ Institutional Scale-Out (Multi-Leg Staged, RS Leaders, Macro Tailwinds)</option>
          <option value="position_trend">📈 Position / Trend (Wider Stops, 50 EMA Runners, Major Themes)</option>
        </select>
      </div>

      <!-- ML Setup Quality Tier -->
      <div class="settings-group">
        <div class="settings-group-title">
          <span>🧠 Setup Quality &amp; Failure Avoidance (QM-V2)</span>
        </div>
        <div class="settings-group-desc">Filters low-expectancy traps and prioritizes high-probability monster breakout runners.</div>
        <select id="cfg_strat_tier" class="cfg-select">
          <option value="all" selected>🌐 All Setups (Unfiltered Universe — 518 Trades)</option>
          <option value="avoid_traps">🛡️ Broad System (Avoid Toxic Traps — 173 Trades)</option>
          <option value="high_quality">⭐ Active Balanced (Tiers 1 &amp; 2: 65.5% WR, +1.68 R, PF 10.6 — 58 Trades)</option>
          <option value="premier">💎 Apex Conviction (Premier Tier 1: 64.3% WR, +2.15 R, PF 13.1 — 16 Trades)</option>
        </select>
      </div>

      <!-- Macro / Universe -->
      <div class="settings-group">
        <div class="settings-group-title">
          <span>🏭 Macro Universe Sponsorship</span>
        </div>
        <div class="settings-group-desc">Filter breakout candidates by institutional sector and theme momentum leadership (Score ≥ 65).</div>
        <select id="cfg_strat_macro" class="cfg-select">
          <option value="leading" selected>Leading Sectors &amp; Leading Themes (Score ≥ 65) — Recommended (+471.0 R)</option>
          <option value="leading_sectors">Leading Sectors Only</option>
          <option value="leading_themes">Leading Themes Only</option>
          <option value="all">Unrestricted Universe (All Sectors &amp; Themes)</option>
        </select>
      </div>

      <!-- Market Health (SPY) -->
      <div class="settings-group">
        <div class="settings-group-title">
          <span>📈 Market Regime Filter (SPY)</span>
        </div>
        <div class="settings-group-desc">Bypasses long trades in macro bear markets to avoid low-liquidity whipsaws.</div>
        <select id="cfg_strat_spy" class="cfg-select">
          <option value="spy200" selected>SPY ≥ 200 SMA (Bull Market Baseline) — Recommended</option>
          <option value="spy50">SPY ≥ 50 SMA (Strict Momentum Alignment)</option>
          <option value="all">All Market Regimes (No SPY Filter)</option>
        </select>
      </div>

      <!-- Stop Loss Mechanics -->
      <div class="settings-group">
        <div class="settings-group-title">
          <span>🛑 Stop Loss Mechanics</span>
        </div>
        <div class="settings-group-desc">Dynamic Quantile Stop (QM-Stop) tailors risk to volatility (~3.5% risk, PF 3.02) vs fixed pivot stops.</div>
        <select id="cfg_strat_stop" class="cfg-select">
          <option value="dynamic">🎯 Dynamic Quantile Regression Stop (QM-Stop: ~3.5% Risk, PF 3.02)</option>
          <option value="06x" selected>0.6x Distance to Previous TrendLab Pivot Low — Optimized (+0.180 R, PF 1.30)</option>
          <option value="10x">1.0x Full Distance to Previous Pivot Low — Classical (+0.057 R, PF 1.09)</option>
        </select>
      </div>

      <!-- Trailing Exit Strategy -->
      <div class="settings-group">
        <div class="settings-group-title">
          <span>🏁 Trailing Exit Model</span>
        </div>
        <div class="settings-group-desc">Dynamic exit discipline across moving averages, staged scaling, and ML climax exhaustion.</div>
        <select id="cfg_strat_exit" class="cfg-select">
          <option value="triad_optimal">👑 Institutional Triad (Dynamic Stop + Climax Partial + Conviction Sizing)</option>
          <option value="climax_swing" selected>🎯 ML Climax Swing (50% Climax @ P≥0.45, 50% 10 EMA Trailing)</option>
          <option value="climax_hybrid">🎯 ML Climax Hybrid (33% Climax @ P≥0.50, 33% 10E, 34% 20E)</option>
          <option value="climax_trend">🎯 ML Climax Trend (50% Climax @ P≥0.65, 50% 50 EMA Trailing)</option>
          <option value="10ema">10 EMA Trailing Exit (Fast / Momentum Turnover — 8.6 bars)</option>
          <option value="20ema">20 EMA Trailing Exit (Intermediate Sweet Spot — 13.5 bars)</option>
          <option value="50ema">50 EMA Trailing Exit (Major Trend Runner — Peak Compounding)</option>
          <option value="staged">Staged Scaling (33% @ 10E / 33% @ 20E / 34% @ 50E — Lowest Drawdown)</option>
          <option value="balanced">Balanced Scaling (50% @ 10E / 50% @ 50E — High Win Rate + Big Runners)</option>
        </select>
      </div>

      <!-- Position Sizing Model -->
      <div class="settings-group">
        <div class="settings-group-title">
          <span>⚖️ Position Sizing Multiplier (S)</span>
        </div>
        <div class="settings-group-desc">Dynamic multi-factor conviction sizing overweights Fresh EPs, Monster Poles, RVOL, and Tight Pullbacks.</div>
        <select id="cfg_strat_sizing" class="cfg-select">
          <option value="dynamic" selected>Dynamic Multi-Factor Sizing (0.50x – 2.50x) — (+471.0 R Total Alpha)</option>
          <option value="equal">Equal Weighting (1.0x Fixed Unit Risk) — (+361.5 R)</option>
        </select>
      </div>

      <!-- Relative Strength Posture -->
      <div class="settings-group">
        <div class="settings-group-title">
          <span>⚡ Relative Strength (RS) &amp; Posture</span>
        </div>
        <div class="settings-group-desc">Institutional leadership posture from daily cross-sectional RS history database.</div>
        <select id="cfg_strat_rs" class="cfg-select">
          <option value="rs90_lead" selected>RS ≥ 90 + (Stage 2 Bullish Stack OR ≤15% 52W High) — Default</option>
          <option value="rs95">RS ≥ 95 Elite Leaders (Stage 2 / ≤15% 52W High) — Peak Win/Loss (2.65:1)</option>
          <option value="rs80">RS ≥ 80 (Stage 2 / ≤15% 52W High)</option>
          <option value="all">All RS Scores &amp; Postures</option>
        </select>
      </div>

      <!-- Fresh EP Alignment -->
      <div class="settings-group">
        <div class="settings-group-title">
          <span>🚀 Fresh EP Recency Alignment</span>
        </div>
        <div class="settings-group-desc">Episodic Pivot catalyst ignition preceding QM momentum pole.</div>
        <select id="cfg_strat_fresh_ep" class="cfg-select">
          <option value="all" selected>All Breakouts (Sizing automatically scales 0.85x – 1.30x)</option>
          <option value="15d">Fresh EP within ≤ 15 Days (+30% Sizing Bonus)</option>
          <option value="30d">Fresh EP within ≤ 30 Days (+15% Sizing Bonus)</option>
          <option value="has_ep">Any Prior EP Catalyst Link</option>
        </select>
      </div>

      <!-- Breakout RVOL Filter -->
      <div class="settings-group">
        <div class="settings-group-title">
          <span>📊 Breakout RVOL Expansion</span>
        </div>
        <div class="settings-group-desc">Volume explosion on breakout confirmation day relative to 20-day SMA.</div>
        <select id="cfg_strat_rvol" class="cfg-select">
          <option value="all" selected>All Volume Levels (Sizing automatically scales 0.90x – 1.20x)</option>
          <option value="1.5x">RVOL ≥ 1.5x Base Volume</option>
          <option value="2.0x">RVOL ≥ 2.0x Base Volume</option>
          <option value="2.5x">RVOL ≥ 2.5x Institutional Explosion (+20% Sizing Bonus)</option>
        </select>
      </div>

      <!-- Nasdaq 100 (QQQ) Trend & Divergence -->
      <div class="settings-group">
        <div class="settings-group-title">
          <span>🌐 Nasdaq 100 (QQQ) Market Regime</span>
        </div>
        <div class="settings-group-desc">Setups during QQQ pullbacks show superior +0.238 R expectancy due to market divergence.</div>
        <select id="cfg_strat_qqq" class="cfg-select">
          <option value="all" selected>All QQQ Market Regimes — Default</option>
          <option value="pullback">Pullback Divergence (QQQ &lt; 10 or 20 EMA) — Highest Edge (+0.238 R, PF 1.48)</option>
          <option value="bull_stack">Extended Bull Stack (QQQ &gt; 10 EMA &gt; 20 EMA)</option>
        </select>
      </div>

      <!-- Tech Leadership (QQQ/SPY Ratio) -->
      <div class="settings-group">
        <div class="settings-group-title">
          <span>💻 Tech Leadership (QQQ/SPY Ratio)</span>
        </div>
        <div class="settings-group-desc">Relative strength of tech equities against broad S&amp;P 500 index.</div>
        <select id="cfg_strat_tech_lead" class="cfg-select">
          <option value="all" selected>All Tech Leadership Regimes — Default</option>
          <option value="bull_stack">(QQQ/SPY) &gt; 10 EMA &gt; 20 EMA (Tech Dominance, +0.201 R)</option>
        </select>
      </div>

      <!-- Bollinger Band Squeeze (Pinch Ratio) -->
      <div class="settings-group">
        <div class="settings-group-title">
          <span>🎯 Bollinger Band Squeeze (Pinch Ratio)</span>
        </div>
        <div class="settings-group-desc">Measures contraction of Bollinger Bands within the base. Tight pinch (≤2.5) yields +0.098 R avg expectancy (+63% vs baseline).</div>
        <select id="cfg_strat_bb_pinch" class="cfg-select">
          <option value="all" selected>All Base Consolidations (No Squeeze Filter)</option>
          <option value="tight">Tight Pinch (≤ 2.5) — High Expansion Potential (+0.098 R, PF 1.17)</option>
          <option value="moderate">Moderate Pinch (≤ 3.2)</option>
          <option value="mild">Mild Pinch (≤ 4.0)</option>
        </select>
      </div>

      <!-- Pole Linearity (R² Regression) -->
      <div class="settings-group">
        <div class="settings-group-title">
          <span>📏 Pole Linearity (R² Regression)</span>
        </div>
        <div class="settings-group-desc">Measures cleanliness of the momentum pole. When combined with BB Pinch ≤ 2.5, R² ≥ 0.65 delivers +0.141 R (+135% edge, PF 1.24).</div>
        <select id="cfg_strat_r2" class="cfg-select">
          <option value="all" selected>All Pole Trajectories (No R² Filter)</option>
          <option value="0.80">Ultra Linear (R² ≥ 0.80) — Institutional Staircase</option>
          <option value="0.65">Strong Trend (R² ≥ 0.65) — Clean Upward Drive</option>
          <option value="0.50">Moderate Trend (R² ≥ 0.50)</option>
        </select>
      </div>

      <div style="display:flex; justify-content:space-between; margin-top:8px; padding-top:10px; border-top:1px solid #1f2937;">
        <button class="btn-filter-chip" onclick="resetStrategySettings()">Reset to Defaults</button>
        <button class="reload-btn" onclick="applyStrategySettings()">✓ Apply Settings &amp; Filter</button>
      </div>
    </div>
  </div>
</div>

<script>
// State
let allEvents = [];
let currentEvent = null;
let currentChartData = null;
let currentSort = { col: 'date', asc: false };
let activeStateFilter = 'all';
let activeEpFilter = 'all';
let activeMacroFilter = 'all';
let searchDebounceTimer = null;
let selectedSetupKeys = new Set();

const SETUP_PALETTE = [
  { stroke: '#10b981', fill: 'rgba(16, 185, 129, 0.12)', text: '#34d399', name: 'Emerald' },
  { stroke: '#38bdf8', fill: 'rgba(56, 189, 248, 0.12)', text: '#38bdf8', name: 'Sky' },
  { stroke: '#f59e0b', fill: 'rgba(245, 158, 11, 0.12)', text: '#fbbf24', name: 'Amber' },
  { stroke: '#a855f7', fill: 'rgba(168, 85, 247, 0.12)', text: '#c084fc', name: 'Purple' },
  { stroke: '#f43f5e', fill: 'rgba(244, 63, 94, 0.12)', text: '#fb7185', name: 'Rose' },
  { stroke: '#06b6d4', fill: 'rgba(6, 182, 212, 0.12)', text: '#22d3ee', name: 'Cyan' },
  { stroke: '#e879f9', fill: 'rgba(232, 121, 249, 0.12)', text: '#f0abfc', name: 'Fuchsia' },
  { stroke: '#fb923c', fill: 'rgba(251, 146, 60, 0.12)', text: '#fdba74', name: 'Orange' },
];

// Strategy Mode & Settings State
let strategyMode = false;
let selectedTraderArchetype = 'high_velocity';
let strategySettings = {
  macro: 'leading',
  spy: 'spy200',
  stop: '06x',
  sizing: 'dynamic',
  exit: 'climax_swing',
  tier: 'all',
  rs: 'rs90_lead',
  fresh_ep: 'all',
  rvol: 'all',
  qqq: 'all',
  tech_lead: 'all',
  bb_pinch: 'all',
  r2: 'all'
};

function applyArchetypePreset(preset) {
  selectedTraderArchetype = preset;
  const selArch = document.getElementById('sel_trader_archetype');
  if (selArch) selArch.value = preset;
  const cfgArch = document.getElementById('cfg_trader_archetype');
  if (cfgArch) cfgArch.value = preset;

  if (preset === 'institutional_triad') {
    // Institutional Triad: Dynamic Quantile Stop (QM-Stop), ML Climax Swing, Conviction Sizing, All Universe (518 Trades)
    strategySettings.macro = 'leading';
    strategySettings.spy = 'spy200';
    strategySettings.stop = 'dynamic';
    strategySettings.exit = 'triad_optimal';
    strategySettings.sizing = 'dynamic';
    strategySettings.tier = 'all';
    strategySettings.rs = 'rs90_lead';
    strategySettings.fresh_ep = 'all';
    strategySettings.rvol = 'all';
    strategySettings.qqq = 'all';
    strategySettings.tech_lead = 'all';
    strategySettings.bb_pinch = 'all';
    strategySettings.r2 = 'all';
  } else if (preset === 'institutional_triad_balanced') {
    // Institutional Triad (Active Balanced): Tiers 1-2 (58 Trades, 65.5% WR, +1.68 R, 10.6 PF)
    strategySettings.macro = 'leading';
    strategySettings.spy = 'spy200';
    strategySettings.stop = 'dynamic';
    strategySettings.exit = 'triad_optimal';
    strategySettings.sizing = 'dynamic';
    strategySettings.tier = 'high_quality';
    strategySettings.rs = 'rs90_lead';
    strategySettings.fresh_ep = 'all';
    strategySettings.rvol = 'all';
    strategySettings.qqq = 'all';
    strategySettings.tech_lead = 'all';
    strategySettings.bb_pinch = 'all';
    strategySettings.r2 = 'all';
  } else if (preset === 'institutional_triad_apex') {
    // Institutional Triad (Apex Conviction): Premier Tier 1 (16 Trades, 64.3% WR, +2.15 R, 13.1 PF)
    strategySettings.macro = 'leading';
    strategySettings.spy = 'spy200';
    strategySettings.stop = 'dynamic';
    strategySettings.exit = 'triad_optimal';
    strategySettings.sizing = 'dynamic';
    strategySettings.tier = 'premier';
    strategySettings.rs = 'rs90_lead';
    strategySettings.fresh_ep = 'all';
    strategySettings.rvol = 'all';
    strategySettings.qqq = 'all';
    strategySettings.tech_lead = 'all';
    strategySettings.bb_pinch = 'all';
    strategySettings.r2 = 'all';
  } else if (preset === 'high_velocity') {
    // High-Velocity Swing: 0.6x Stop, ML Climax Partial (P>=0.45) + 10 EMA Trailing, Dynamic Sizing, Leading Macro, All BB Pinch
    strategySettings.macro = 'leading';
    strategySettings.spy = 'spy200';
    strategySettings.stop = '06x';
    strategySettings.exit = 'climax_swing';
    strategySettings.sizing = 'dynamic';
    strategySettings.tier = 'avoid_traps';
    strategySettings.rs = 'rs90_lead';
    strategySettings.fresh_ep = 'all';
    strategySettings.rvol = 'all';
    strategySettings.qqq = 'all';
    strategySettings.tech_lead = 'all';
    strategySettings.bb_pinch = 'all';
    strategySettings.r2 = 'all';
  } else if (preset === 'institutional_hybrid') {
    // Institutional Scale-Out: 0.6x Stop, Multi-Leg Staged (P>=0.50 + Staged 10/20 EMA), RS 90 Leaders, Fresh EP <= 30d, RVOL >= 1.5x
    strategySettings.macro = 'leading';
    strategySettings.spy = 'spy200';
    strategySettings.stop = '06x';
    strategySettings.exit = 'climax_hybrid';
    strategySettings.sizing = 'dynamic';
    strategySettings.tier = 'all';
    strategySettings.rs = 'rs90_lead';
    strategySettings.fresh_ep = '30d';
    strategySettings.rvol = '1.5x';
    strategySettings.qqq = 'all';
    strategySettings.tech_lead = 'all';
    strategySettings.bb_pinch = 'all';
    strategySettings.r2 = 'all';
  } else if (preset === 'position_trend') {
    // Position / Trend: 1.0x Full Pivot Stop, Major Trend Climax (P>=0.65 + 50 EMA Trailing), Dynamic Sizing, Leading Macro, RS 90
    strategySettings.macro = 'leading';
    strategySettings.spy = 'spy200';
    strategySettings.stop = '10x';
    strategySettings.exit = 'climax_trend';
    strategySettings.sizing = 'dynamic';
    strategySettings.tier = 'all';
    strategySettings.rs = 'rs90_lead';
    strategySettings.fresh_ep = 'all';
    strategySettings.rvol = 'all';
    strategySettings.qqq = 'all';
    strategySettings.tech_lead = 'all';
    strategySettings.bb_pinch = 'all';
    strategySettings.r2 = 'all';
  }

  // Update modal fields
  const selMacro = document.getElementById('cfg_strat_macro');
  const selSpy = document.getElementById('cfg_strat_spy');
  const selStop = document.getElementById('cfg_strat_stop');
  const selExit = document.getElementById('cfg_strat_exit');
  const selSizing = document.getElementById('cfg_strat_sizing');
  const selRs = document.getElementById('cfg_strat_rs');
  const selFresh = document.getElementById('cfg_strat_fresh_ep');
  const selRvol = document.getElementById('cfg_strat_rvol');
  const selQqq = document.getElementById('cfg_strat_qqq');
  const selTech = document.getElementById('cfg_strat_tech_lead');
  const selBb = document.getElementById('cfg_strat_bb_pinch');
  const selR2 = document.getElementById('cfg_strat_r2');
  const selTier = document.getElementById('cfg_strat_tier');

  if (selMacro) selMacro.value = strategySettings.macro;
  if (selSpy) selSpy.value = strategySettings.spy;
  if (selStop) selStop.value = strategySettings.stop;
  if (selExit) selExit.value = strategySettings.exit;
  if (selSizing) selSizing.value = strategySettings.sizing;
  if (selRs) selRs.value = strategySettings.rs;
  if (selFresh) selFresh.value = strategySettings.fresh_ep;
  if (selRvol) selRvol.value = strategySettings.rvol;
  if (selQqq) selQqq.value = strategySettings.qqq;
  if (selTech) selTech.value = strategySettings.tech_lead;
  if (selBb) selBb.value = strategySettings.bb_pinch;
  if (selR2) selR2.value = strategySettings.r2;
  if (selTier) selTier.value = strategySettings.tier || 'all';

  // Sync toolbar fields
  const selExitToolbar = document.getElementById('sel_strat_exit');
  if (selExitToolbar) selExitToolbar.value = strategySettings.exit;
  const selBbToolbar = document.getElementById('sel_bb_pinch');
  if (selBbToolbar) selBbToolbar.value = strategySettings.bb_pinch;
  const selR2Toolbar = document.getElementById('sel_r2');
  if (selR2Toolbar) selR2Toolbar.value = strategySettings.r2;
  const selTierToolbar = document.getElementById('sel_strat_tier');
  if (selTierToolbar) selTierToolbar.value = strategySettings.tier || 'all';
}

function onTraderArchetypeChange() {
  const sel = document.getElementById('sel_trader_archetype');
  if (sel) {
    applyArchetypePreset(sel.value);
    if (!strategyMode) {
      toggleStrategyMode();
    } else {
      fetchEvents();
    }
  }
}

function toggleStrategyMode() {
  strategyMode = !strategyMode;
  const btn = document.getElementById('btn_strategy_toggle');
  const lbl = document.getElementById('strategy_mode_lbl');
  const stratKpis = document.getElementById('strategy_kpi_bar');
  const normKpis = document.getElementById('kpi_row');
  const grpExit = document.getElementById('grp_strat_exit');
  const grpArch = document.getElementById('grp_strat_archetype');

  if (strategyMode) {
    if (btn) btn.classList.add('active');
    if (lbl) lbl.textContent = '🎯 Strategy Mode: ACTIVE';
    if (stratKpis) stratKpis.style.display = 'flex';
    if (normKpis) normKpis.style.display = 'none';
    if (grpExit) grpExit.style.display = 'flex';
    if (grpArch) grpArch.style.display = 'flex';
  } else {
    if (btn) btn.classList.remove('active');
    if (lbl) lbl.textContent = 'Strategy Mode: OFF';
    if (stratKpis) stratKpis.style.display = 'none';
    if (normKpis) normKpis.style.display = 'flex';
    if (grpExit) grpExit.style.display = 'none';
    if (grpArch) grpArch.style.display = 'flex';
  }
  fetchEvents();
}

function onStratExitChange() {
  const sel = document.getElementById('sel_strat_exit');
  if (sel) {
    strategySettings.exit = sel.value;
    const cfg = document.getElementById('cfg_strat_exit');
    if (cfg) cfg.value = sel.value;
    // Mark as custom if manually modified away from archetype preset
    checkCustomArchetype();
  }
  fetchEvents();
}

function onStratTierChange() {
  const sel = document.getElementById('sel_strat_tier');
  if (sel) {
    strategySettings.tier = sel.value;
    const cfg = document.getElementById('cfg_strat_tier');
    if (cfg) cfg.value = sel.value;
    checkCustomArchetype();
  }
  fetchEvents();
}

function checkCustomArchetype() {
  const isInstTriadApex = (
    strategySettings.stop === 'dynamic' &&
    strategySettings.exit === 'triad_optimal' &&
    strategySettings.sizing === 'dynamic' &&
    strategySettings.macro === 'leading' &&
    strategySettings.tier === 'premier' &&
    strategySettings.bb_pinch === 'all' &&
    strategySettings.fresh_ep === 'all' &&
    strategySettings.rvol === 'all'
  );
  const isInstTriadBalanced = (
    strategySettings.stop === 'dynamic' &&
    strategySettings.exit === 'triad_optimal' &&
    strategySettings.sizing === 'dynamic' &&
    strategySettings.macro === 'leading' &&
    strategySettings.tier === 'high_quality' &&
    strategySettings.bb_pinch === 'all' &&
    strategySettings.fresh_ep === 'all' &&
    strategySettings.rvol === 'all'
  );
  const isInstTriadAll = (
    strategySettings.stop === 'dynamic' &&
    strategySettings.exit === 'triad_optimal' &&
    strategySettings.sizing === 'dynamic' &&
    strategySettings.macro === 'leading' &&
    (strategySettings.tier === 'all' || strategySettings.tier === 'avoid_traps') &&
    strategySettings.bb_pinch === 'all' &&
    strategySettings.fresh_ep === 'all' &&
    strategySettings.rvol === 'all'
  );
  const isHighVelocity = (
    strategySettings.stop === '06x' &&
    strategySettings.exit === 'climax_swing' &&
    strategySettings.sizing === 'dynamic' &&
    strategySettings.macro === 'leading' &&
    strategySettings.tier === 'avoid_traps' &&
    strategySettings.bb_pinch === 'all' &&
    strategySettings.fresh_ep === 'all' &&
    strategySettings.rvol === 'all'
  );
  const isInstHybrid = (
    strategySettings.stop === '06x' &&
    strategySettings.exit === 'climax_hybrid' &&
    strategySettings.sizing === 'dynamic' &&
    strategySettings.macro === 'leading' &&
    strategySettings.tier === 'all' &&
    strategySettings.fresh_ep === '30d' &&
    strategySettings.rvol === '1.5x'
  );
  const isPosTrend = (
    strategySettings.stop === '10x' &&
    strategySettings.exit === 'climax_trend' &&
    strategySettings.sizing === 'dynamic' &&
    strategySettings.macro === 'leading' &&
    strategySettings.tier === 'all' &&
    strategySettings.fresh_ep === 'all' &&
    strategySettings.rvol === 'all'
  );

  let detected = 'custom';
  if (isInstTriadApex) detected = 'institutional_triad_apex';
  else if (isInstTriadBalanced) detected = 'institutional_triad_balanced';
  else if (isInstTriadAll) detected = 'institutional_triad';
  else if (isHighVelocity) detected = 'high_velocity';
  else if (isInstHybrid) detected = 'institutional_hybrid';
  else if (isPosTrend) detected = 'position_trend';

  selectedTraderArchetype = detected;
  const selArch = document.getElementById('sel_trader_archetype');
  if (selArch) selArch.value = detected;
  const cfgArch = document.getElementById('cfg_trader_archetype');
  if (cfgArch) cfgArch.value = detected;
}

function openStrategySettingsModal() {
  const selMacro = document.getElementById('cfg_strat_macro');
  const selSpy = document.getElementById('cfg_strat_spy');
  const selStop = document.getElementById('cfg_strat_stop');
  const selExit = document.getElementById('cfg_strat_exit');
  const selSizing = document.getElementById('cfg_strat_sizing');
  const selRs = document.getElementById('cfg_strat_rs');
  const selFresh = document.getElementById('cfg_strat_fresh_ep');
  const selRvol = document.getElementById('cfg_strat_rvol');
  const selQqq = document.getElementById('cfg_strat_qqq');
  const selTech = document.getElementById('cfg_strat_tech_lead');
  const selBb = document.getElementById('cfg_strat_bb_pinch');
  const selR2 = document.getElementById('cfg_strat_r2');
  const selTier = document.getElementById('cfg_strat_tier');
  const selArch = document.getElementById('cfg_trader_archetype');

  if (selArch) selArch.value = selectedTraderArchetype;
  if (selMacro) selMacro.value = strategySettings.macro;
  if (selSpy) selSpy.value = strategySettings.spy;
  if (selStop) selStop.value = strategySettings.stop;
  if (selExit) selExit.value = strategySettings.exit || 'climax_swing';
  if (selSizing) selSizing.value = strategySettings.sizing;
  if (selRs) selRs.value = strategySettings.rs;
  if (selFresh) selFresh.value = strategySettings.fresh_ep;
  if (selRvol) selRvol.value = strategySettings.rvol;
  if (selQqq) selQqq.value = strategySettings.qqq;
  if (selTech) selTech.value = strategySettings.tech_lead;
  if (selBb) selBb.value = strategySettings.bb_pinch;
  if (selR2) selR2.value = strategySettings.r2;
  if (selTier) selTier.value = strategySettings.tier || 'all';

  const m = document.getElementById('strategy_settings_modal');
  if (m) m.classList.add('active');
}

function closeStrategySettingsModal(e) {
  if (e && e.target && e.target.id !== 'strategy_settings_modal' && !e.target.classList.contains('modal-close')) return;
  const m = document.getElementById('strategy_settings_modal');
  if (m) m.classList.remove('active');
}

function applyStrategySettings() {
  strategySettings.macro = document.getElementById('cfg_strat_macro').value;
  strategySettings.spy = document.getElementById('cfg_strat_spy').value;
  strategySettings.stop = document.getElementById('cfg_strat_stop').value;
  if (document.getElementById('cfg_strat_exit')) strategySettings.exit = document.getElementById('cfg_strat_exit').value;
  strategySettings.sizing = document.getElementById('cfg_strat_sizing').value;
  strategySettings.rs = document.getElementById('cfg_strat_rs').value;
  strategySettings.fresh_ep = document.getElementById('cfg_strat_fresh_ep').value;
  strategySettings.rvol = document.getElementById('cfg_strat_rvol').value;
  strategySettings.qqq = document.getElementById('cfg_strat_qqq').value;
  strategySettings.tech_lead = document.getElementById('cfg_strat_tech_lead').value;
  if (document.getElementById('cfg_strat_bb_pinch')) strategySettings.bb_pinch = document.getElementById('cfg_strat_bb_pinch').value;
  if (document.getElementById('cfg_strat_r2')) strategySettings.r2 = document.getElementById('cfg_strat_r2').value;
  if (document.getElementById('cfg_strat_tier')) strategySettings.tier = document.getElementById('cfg_strat_tier').value;

  const selExit = document.getElementById('sel_strat_exit');
  if (selExit) selExit.value = strategySettings.exit;
  const selBb = document.getElementById('sel_bb_pinch');
  if (selBb) selBb.value = strategySettings.bb_pinch;
  const selR2 = document.getElementById('sel_r2');
  if (selR2) selR2.value = strategySettings.r2;
  const selTier = document.getElementById('sel_strat_tier');
  if (selTier) selTier.value = strategySettings.tier;

  checkCustomArchetype();
  closeStrategySettingsModal();

  if (!strategyMode) {
    toggleStrategyMode();
  } else {
    fetchEvents();
  }
}

function resetStrategySettings() {
  applyArchetypePreset('high_velocity');
  openStrategySettingsModal();
}

function updateStrategyKPIs(k) {
  if (!k) return;
  const trEl = document.getElementById('skpi_trades');
  if (trEl) trEl.textContent = (k.total_trades || 0).toLocaleString();
  const wrEl = document.getElementById('skpi_winrate');
  if (wrEl) wrEl.textContent = (k.win_rate != null ? k.win_rate.toFixed(1) : '0.0') + '%';
  const subWrEl = document.getElementById('skpi_sub_winrate');
  const exitLabels = {
    'triad_optimal': '👑 Institutional Triad (Dynamic + Climax Swing)',
    'climax_swing': 'ML Climax Swing (P≥0.45 + 10 EMA)',
    'climax_hybrid': 'ML Climax Hybrid (P≥0.50 + Staged)',
    'climax_trend': 'ML Climax Trend (P≥0.65 + 50 EMA)',
    '10ema': '10 EMA Trailing Exit',
    '20ema': '20 EMA Trailing Exit',
    '50ema': '50 EMA Trailing Exit',
    'staged': 'Staged Scale (33/33/34)',
    'balanced': 'Balanced Scale (50/50)',
  };
  const exLbl = exitLabels[k.strat_exit] || '10 EMA Exit';
  const stopLbl = k.strat_stop === 'dynamic' ? 'Dynamic ML Stop' : (k.strat_stop === '06x' ? '0.6x Stop' : '1.0x Stop');
  if (subWrEl) subWrEl.textContent = stopLbl + ' • ' + exLbl;
  const pfEl = document.getElementById('skpi_pf');
  if (pfEl) pfEl.textContent = k.profit_factor != null ? k.profit_factor.toFixed(2) : '--';
  const expEl = document.getElementById('skpi_expectancy');
  if (expEl) expEl.textContent = (k.expectancy_r > 0 ? '+' : '') + (k.expectancy_r != null ? k.expectancy_r.toFixed(3) : '0.0') + ' R';
  const totREl = document.getElementById('skpi_total_r');
  if (totREl) totREl.textContent = (k.total_r > 0 ? '+' : '') + (k.total_r != null ? k.total_r.toFixed(1) : '0.0') + ' R';
  const sizedREl = document.getElementById('skpi_sized_r');
  if (sizedREl) sizedREl.textContent = (k.sized_total_r > 0 ? '+' : '') + (k.sized_total_r != null ? k.sized_total_r.toFixed(1) : '0.0') + ' R';
  const subSizedEl = document.getElementById('skpi_sub_sized_r');
  if (subSizedEl) subSizedEl.textContent = (k.strat_sizing === 'dynamic' ? 'Dynamic Multiplier' : '1.0x Equal Weight');
  const wlEl = document.getElementById('skpi_win_loss');
  if (wlEl) wlEl.textContent = (k.win_loss_ratio != null ? k.win_loss_ratio.toFixed(2) : '--') + ' : 1';
  const subWlEl = document.getElementById('skpi_sub_win_loss');
  if (subWlEl) subWlEl.textContent = `Avg +${k.avg_win_r || 0}R vs ${k.avg_loss_r || 0}R`;
  const maxWEl = document.getElementById('skpi_max_win');
  if (maxWEl) maxWEl.textContent = `+${k.max_win_r || 0} R / ${k.max_loss_r || 0} R`;
  const subMaxWEl = document.getElementById('skpi_sub_max_win');
  if (subMaxWEl) subMaxWEl.textContent = `Avg Hold: ${k.avg_bars || 0} bars`;
}

// Chart state
let chart = null;
let candleSeries = null;
let volumeSeries = null;
let volMa20Series = null;
let ema10Series = null;
let ema20Series = null;
let sma50Series = null;
let sma200Series = null;
let larssonFastSeries = null;
let larssonSlowSeries = null;
let activePriceLines = [];

let showPoleBox = true;
let showConsBox = true;
let showPriceLines = true;
let showEpMarkers = true;

/* --- On-Chart Drawing Tools (Measure, Line, Text, Arrow) with Disk Persistence --- */
let currentDrawTool = 'cursor';
let drawings = [];
let drawStart = null;
let currentChartSymbol = null;

function saveCurrentDrawings() {
  const sym = currentChartSymbol || (currentEvent ? currentEvent.symbol : null);
  if (!sym) return;
  fetch('/api/annotations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbol: sym, drawings: drawings })
  }).then(r => r.json()).catch(err => {
    console.error('Failed to save drawings to disk:', err);
  });
}

function loadDrawingsForSymbol(sym) {
  if (!sym) return;
  currentChartSymbol = sym;
  fetch(`/api/annotations?symbol=${encodeURIComponent(sym)}`)
    .then(r => r.json())
    .then(res => {
      if (currentChartSymbol === sym) {
        drawings = (res && Array.isArray(res.drawings)) ? res.drawings : [];
        renderAllDrawings();
      }
    })
    .catch(err => {
      console.warn('Could not load drawings for ' + sym, err);
      drawings = [];
      renderAllDrawings();
    });
}

function toggleDrawToolbar() {
  const tb = document.getElementById('chart_draw_toolbar');
  if (!tb) return;
  const isHidden = (tb.style.display === 'none' || !tb.style.display);
  tb.style.display = isHidden ? 'flex' : 'none';
  const b = document.getElementById('btn_toggle_draw');
  if (b) b.classList.toggle('active', isHidden);
}

function setDrawTool(tool) {
  currentDrawTool = tool;
  document.querySelectorAll('.draw-btn').forEach(b => b.classList.remove('active'));
  const btn = document.getElementById('btn_tool_' + tool);
  if (btn) btn.classList.add('active');

  const canvas = document.getElementById('draw_canvas');
  if (!canvas) return;
  if (tool === 'cursor') {
    canvas.style.pointerEvents = 'none';
    canvas.style.cursor = 'default';
    drawStart = null;
    clearDrawPreview();
  } else {
    canvas.style.pointerEvents = 'auto';
    canvas.style.cursor = 'crosshair';
  }
}

function clearAllDrawings() {
  drawings = [];
  drawStart = null;
  clearDrawPreview();
  renderAllDrawings();
  saveCurrentDrawings();
}

function clearDrawPreview() {
  const prev = document.getElementById('draw_preview');
  if (prev) prev.innerHTML = '';
}

function getCanvasCoords(e) {
  const canvas = document.getElementById('draw_canvas');
  if (!canvas) return { x: 0, y: 0, price: null, time: null, w: 1, h: 1 };
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  const price = candleSeries ? candleSeries.coordinateToPrice(y) : null;
  let time = chart ? chart.timeScale().coordinateToTime(x) : null;
  if (time) {
    if (typeof time === 'object' && time.year) {
      const mm = String(time.month).padStart(2, '0');
      const dd = String(time.day).padStart(2, '0');
      time = `${time.year}-${mm}-${dd}`;
    } else if (typeof time === 'number') {
      const dt = new Date(time * 1000);
      if (!isNaN(dt.getTime())) {
        time = dt.toISOString().split('T')[0];
      }
    }
  }
  return { x, y, price, time, w: rect.width || 1, h: rect.height || 1 };
}

function coordForTime(t, fallbackX) {
  if (!chart || !t) return fallbackX;
  try {
    const cx = chart.timeScale().timeToCoordinate(t);
    return cx != null ? cx : fallbackX;
  } catch(e) {
    return fallbackX;
  }
}

function coordForPrice(p, fallbackY) {
  if (!candleSeries || p == null) return fallbackY;
  try {
    const cy = candleSeries.priceToCoordinate(p);
    return cy != null ? cy : fallbackY;
  } catch(e) {
    return fallbackY;
  }
}

function onCanvasMouseDown(e) {
  if (currentDrawTool === 'cursor') return;
  const pt = getCanvasCoords(e);

  if (currentDrawTool === 'text') {
    const note = prompt('Enter chart note / annotation:');
    if (note && note.trim()) {
      drawings.push({
        type: 'text',
        time: pt.time,
        price: pt.price,
        x_rel: pt.w > 0 ? pt.x / pt.w : 0.5,
        y_rel: pt.h > 0 ? pt.y / pt.h : 0.5,
        text: note.trim()
      });
      renderAllDrawings();
      saveCurrentDrawings();
    }
    setDrawTool('cursor');
    return;
  }

  if (!drawStart) {
    drawStart = pt;
  } else {
    commitDrawing(drawStart, pt, currentDrawTool);
    drawStart = null;
    clearDrawPreview();
    setDrawTool('cursor');
  }
}

function onCanvasMouseMove(e) {
  if (currentDrawTool === 'cursor' || !drawStart) return;
  const pt = getCanvasCoords(e);
  renderPreviewDrawing(drawStart, pt, currentDrawTool);
}

function commitDrawing(p1, p2, tool) {
  const w = p1.w || 1000;
  const h = p1.h || 600;
  drawings.push({
    type: tool,
    t1: p1.time,
    p1: p1.price,
    t2: p2.time,
    p2: p2.price,
    x1_rel: p1.x / w,
    y1_rel: p1.y / h,
    x2_rel: p2.x / w,
    y2_rel: p2.y / h
  });
  renderAllDrawings();
  saveCurrentDrawings();
}

function renderPreviewDrawing(p1, p2, tool) {
  const prevG = document.getElementById('draw_preview');
  if (!prevG) return;
  prevG.innerHTML = buildDrawingSvg(tool, p1.x, p1.y, p2.x, p2.y, p1.price, p2.price, true);
}

function buildDrawingSvg(tool, x1, y1, x2, y2, price1, price2, isPreview) {
  if (tool === 'line') {
    return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#38bdf8" stroke-width="2" stroke-dasharray="${isPreview?'4 4':'none'}"/>
            <circle cx="${x1}" cy="${y1}" r="3.5" fill="#38bdf8"/>
            <circle cx="${x2}" cy="${y2}" r="3.5" fill="#38bdf8"/>`;
  }
  if (tool === 'arrow') {
    return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#38bdf8" stroke-width="2.5" marker-end="url(#arrowhead)"/>
            <circle cx="${x1}" cy="${y1}" r="3" fill="#38bdf8"/>`;
  }
  if (tool === 'measure') {
    const rx = Math.min(x1, x2);
    const ry = Math.min(y1, y2);
    const rw = Math.abs(x2 - x1);
    const rh = Math.abs(y2 - y1);
    const deltaP = (price1 != null && price2 != null) ? (price2 - price1) : 0;
    const pct = (price1 != null && price1 > 0 && price2 != null) ? ((price2 - price1) / price1 * 100) : 0;
    const isUp = deltaP >= 0;
    const boxColor = isUp ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)';
    const strokeColor = isUp ? '#10b981' : '#ef4444';
    const tagBg = isUp ? '#065f46' : '#991b1b';
    const tagX = rx + rw / 2;
    const tagY = ry + rh / 2;
    const sign = isUp ? '+' : '';
    const label = `${sign}${pct.toFixed(2)}% (${sign}$${deltaP.toFixed(2)})`;

    return `<rect x="${rx}" y="${ry}" width="${rw}" height="${rh}" fill="${boxColor}" stroke="${strokeColor}" stroke-dasharray="3 3" stroke-width="1.5"/>
            <line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${strokeColor}" stroke-width="1.5"/>
            <circle cx="${x1}" cy="${y1}" r="3.5" fill="${strokeColor}"/>
            <circle cx="${x2}" cy="${y2}" r="3.5" fill="${strokeColor}"/>
            <g transform="translate(${tagX}, ${tagY})">
              <rect x="-75" y="-13" width="150" height="24" rx="4" fill="${tagBg}" stroke="${strokeColor}" stroke-width="1"/>
              <text x="0" y="3" text-anchor="middle" fill="#ffffff" font-size="11" font-weight="700" font-family="sans-serif">${label}</text>
            </g>`;
  }
  return '';
}

function renderAllDrawings() {
  const container = document.getElementById('draw_elements');
  const canvas = document.getElementById('draw_canvas');
  if (!container || !canvas) return;
  const rect = canvas.getBoundingClientRect();
  const w = rect.width, h = rect.height;
  if (w <= 0 || h <= 0) return;

  let html = '';
  drawings.forEach(d => {
    if (d.type === 'text') {
      const x = coordForTime(d.time, d.x_rel * w);
      const y = coordForPrice(d.price, d.y_rel * h);
      const textLen = (d.text || '').length;
      const boxW = Math.max(70, textLen * 7 + 16);
      html += `<g transform="translate(${x}, ${y})">
                 <rect x="0" y="-20" width="${boxW}" height="24" rx="4" fill="#0f172a" stroke="#38bdf8" stroke-width="1.5"/>
                 <text x="8" y="-4" fill="#f8fafc" font-size="11" font-weight="600" font-family="sans-serif">${d.text}</text>
                 <circle cx="0" cy="0" r="3" fill="#38bdf8"/>
               </g>`;
    } else {
      const x1 = coordForTime(d.t1, d.x1_rel * w);
      const y1 = coordForPrice(d.p1, d.y1_rel * h);
      const x2 = coordForTime(d.t2, d.x2_rel * w);
      const y2 = coordForPrice(d.p2, d.y2_rel * h);
      html += buildDrawingSvg(d.type, x1, y1, x2, y2, d.p1, d.p2, false);
    }
  });
  container.innerHTML = html;
}

// Init Chart
function initChart() {
  const container = document.getElementById('chart_tv');
  container.innerHTML = '';

  chart = LightweightCharts.createChart(container, {
    width: container.clientWidth,
    height: container.clientHeight,
    layout: {
      background: { color: '#0c0f16' },
      textColor: '#94a3b8',
      fontSize: 11,
      fontFamily: 'monospace',
    },
    grid: {
      vertLines: { color: '#161d2b' },
      horzLines: { color: '#161d2b' },
    },
    crosshair: {
      mode: LightweightCharts.CrosshairMode.Normal,
      vertLine: { color: '#475569', width: 1, style: 3 },
      horzLine: { color: '#475569', width: 1, style: 3 },
    },
    rightPriceScale: {
      borderColor: '#232936',
      autoScale: true,
      scaleMargins: { top: 0.08, bottom: 0.22 },
    },
    timeScale: {
      borderColor: '#232936',
      timeVisible: true,
      secondsVisible: false,
      visible: true,
      borderVisible: true,
    },
  });

  volumeSeries = chart.addHistogramSeries({
    priceFormat: { type: 'volume' },
    priceScaleId: 'volume',
    priceLineVisible: false,
    lastValueVisible: false,
  });

  volMa20Series = chart.addLineSeries({
    color: '#38bdf8',
    lineWidth: 1,
    title: 'Vol MA20',
    priceScaleId: 'volume',
    priceLineVisible: false,
    lastValueVisible: false,
  });

  chart.priceScale('volume').applyOptions({
    scaleMargins: { top: 0.85, bottom: 0.0 },
  });

  larssonFastSeries = chart.addLineSeries({
    lineWidth: 1,
    color: 'rgba(250, 204, 21, 0.001)',
    title: 'Larsson Fast',
    priceLineVisible: false,
    lastValueVisible: true,
    crosshairMarkerVisible: false,
  });
  larssonSlowSeries = chart.addLineSeries({
    lineWidth: 1,
    color: 'rgba(56, 189, 248, 0.001)',
    title: 'Larsson Slow',
    priceLineVisible: false,
    lastValueVisible: true,
    crosshairMarkerVisible: false,
  });

  candleSeries = chart.addCandlestickSeries({
    upColor: '#10b981',
    downColor: '#ef4444',
    borderVisible: false,
    wickUpColor: '#10b981',
    wickDownColor: '#ef4444',
  });

  ema10Series = chart.addLineSeries({ color: '#38bdf8', lineWidth: 1.5, title: 'EMA10' });
  ema20Series = chart.addLineSeries({ color: '#eab308', lineWidth: 1.5, title: 'EMA20' });
  sma50Series = chart.addLineSeries({ color: '#ef4444', lineWidth: 1.5, title: 'SMA50' });
  sma200Series = chart.addLineSeries({ color: '#64748b', lineWidth: 1.5, title: 'SMA200' });

  // Crosshair move HUD
  chart.subscribeCrosshairMove(param => {
    const hud = document.getElementById('chart_hud');
    if (!param.time || !param.seriesPrices || !param.seriesPrices.get(candleSeries)) {
      hud.style.display = 'none';
      return;
    }
    const ohlc = param.seriesPrices.get(candleSeries);
    const vol = param.seriesPrices.get(volumeSeries) || 0;
    const e10 = param.seriesPrices.get(ema10Series);
    const e20 = param.seriesPrices.get(ema20Series);
    const s50 = param.seriesPrices.get(sma50Series);

    hud.style.display = 'flex';
    hud.innerHTML = `
      <span><strong>${param.time}</strong></span>
      <span>O: ${ohlc.open.toFixed(2)}</span>
      <span>H: ${ohlc.high.toFixed(2)}</span>
      <span>L: ${ohlc.low.toFixed(2)}</span>
      <span>C: <strong style="color:${ohlc.close >= ohlc.open ? '#34d399' : '#f87171'}">${ohlc.close.toFixed(2)}</strong></span>
      <span>Vol: ${(vol / 1000).toFixed(0)}K</span>
      ${e10 ? `<span style="color:#38bdf8;">E10: ${e10.toFixed(2)}</span>` : ''}
      ${e20 ? `<span style="color:#eab308;">E20: ${e20.toFixed(2)}</span>` : ''}
      ${s50 ? `<span style="color:#ef4444;">S50: ${s50.toFixed(2)}</span>` : ''}
    `;

    if (window._lastLarssonData) {
      const lItem = window._lastLarssonData.find(x => x.time === param.time);
      if (lItem) updateLarssonLegendValues(lItem);
    }
  });

  // Time scale scroll/zoom listener to redraw canvas shaded boxes, drawings, and larsson ribbon
  chart.timeScale().subscribeVisibleLogicalRangeChange(() => {
    drawCanvasOverlays();
    renderAllDrawings();
    scheduleLarssonRender();
  });
  chart.timeScale().subscribeVisibleTimeRangeChange(() => {
    scheduleLarssonRender();
  });

  const containerRel = document.querySelector('.chart-container-rel');
  if (containerRel) {
    containerRel.addEventListener('wheel', () => scheduleLarssonRender(), { passive: true });
    containerRel.addEventListener('touchmove', () => scheduleLarssonRender(), { passive: true });
    containerRel.addEventListener('mousemove', (e) => {
      if (e.buttons > 0) scheduleLarssonRender();
    });
    containerRel.addEventListener('mouseup', () => scheduleLarssonRender());
  }

  const drawCanvas = document.getElementById('draw_canvas');
  if (drawCanvas) {
    drawCanvas.addEventListener('mousedown', onCanvasMouseDown);
    drawCanvas.addEventListener('mousemove', onCanvasMouseMove);
  }

  window.addEventListener('resize', () => {
    if (chart && container) {
      chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
      drawCanvasOverlays();
      renderAllDrawings();
      scheduleLarssonRender();
    }
  });

  const roTarget = document.querySelector('.chart-container-rel') || container;
  if (window.ResizeObserver && roTarget) {
    const ro = new ResizeObserver(entries => {
      for (const entry of entries) {
        const cr = entry.contentRect;
        if (cr.width > 0 && cr.height > 0 && chart) {
          chart.applyOptions({ width: Math.round(cr.width), height: Math.round(cr.height) });
          requestAnimationFrame(() => {
            drawCanvasOverlays();
            renderAllDrawings();
            scheduleLarssonRender();
          });
        }
      }
    });
    ro.observe(roTarget);
  }
}

function drawCanvasOverlays() {
  const canvas = document.getElementById('chart_canvas_overlay');
  if (!canvas || !chart || !candleSeries || !currentChartData) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.parentElement.clientWidth;
  const h = canvas.height = canvas.parentElement.clientHeight;
  ctx.clearRect(0, 0, w, h);

  const timeScale = chart.timeScale();
  const allSymSetups = currentChartData.symbol_setups || (currentChartData.setup ? [currentChartData.setup] : []);
  if (!allSymSetups.length) return;

  // Filter to setups selected by user toggles, or default to current setup if none selected for this symbol
  let setupsToDraw = allSymSetups.filter(s => selectedSetupKeys.has(`${s.symbol}_${s.date}`));
  if (setupsToDraw.length === 0 && currentChartData.setup) {
    setupsToDraw = [currentChartData.setup];
  }

  // Draw original full-height background visualization for setups
  setupsToDraw.forEach((setup, idx) => {
    const isCurrent = (currentEvent && currentEvent.symbol === setup.symbol && currentEvent.date === setup.date);

    // 1. Shaded Pole Area (Emerald Green vertical strip from top 0 to bottom h)
    if (showPoleBox && setup.qm_pole_start && setup.qm_pole_end) {
      const x1 = timeScale.timeToCoordinate(setup.qm_pole_start);
      const x2 = timeScale.timeToCoordinate(setup.qm_pole_end);
      if (x1 !== null && x2 !== null) {
        const left = Math.min(x1, x2);
        const width = Math.max(8, Math.abs(x2 - x1));
        ctx.fillStyle = isCurrent ? 'rgba(16, 185, 129, 0.12)' : 'rgba(16, 185, 129, 0.06)';
        ctx.fillRect(left, 0, width, h);
        ctx.strokeStyle = isCurrent ? 'rgba(16, 185, 129, 0.45)' : 'rgba(16, 185, 129, 0.25)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.strokeRect(left, 0, width, h);
        ctx.setLineDash([]);

        // Label at top
        ctx.fillStyle = '#34d399';
        ctx.font = 'bold 11px monospace';
        let poleLbl = `POLE +${setup.move_pct}%`;
        if (setup.r2 != null) poleLbl += ` (R² ${setup.r2.toFixed(2)})`;
        ctx.fillText(poleLbl, left + 6, 26);
      }
    }

    // 2. Shaded Consolidation / Base Area (Amber vertical strip from top 0 to bottom h)
    const cStart = setup.qm_cons_start || setup.qm_pole_end;
    const cEnd = setup.qm_cons_end || setup.date;
    if (showConsBox && cStart && cEnd) {
      const x1 = timeScale.timeToCoordinate(cStart);
      const x2 = timeScale.timeToCoordinate(cEnd);
      if (x1 !== null && x2 !== null) {
        const left = Math.min(x1, x2);
        const width = Math.max(12, Math.abs(x2 - x1));
        ctx.fillStyle = isCurrent ? 'rgba(245, 158, 11, 0.12)' : 'rgba(245, 158, 11, 0.06)';
        ctx.fillRect(left, 0, width, h);
        ctx.strokeStyle = isCurrent ? 'rgba(245, 158, 11, 0.45)' : 'rgba(245, 158, 11, 0.25)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.strokeRect(left, 0, width, h);
        ctx.setLineDash([]);

        // Label at top
        ctx.fillStyle = '#fbbf24';
        ctx.font = 'bold 11px monospace';
        let baseLbl = `BASE (${setup.cons_bars}d, -${setup.pullback_pct}%)`;
        if (setup.pinch_ratio != null) baseLbl += ` [Pinch ${setup.pinch_ratio.toFixed(1)}x]`;
        ctx.fillText(baseLbl, left + 6, 44);
      }
    }

    // 3. Marking the Breakout (Vertical line and top badge)
    const isBk = (setup.state === 'breakout');
    const bkDate = setup.date || setup.qm_cons_end;
    if (bkDate) {
      const xBk = timeScale.timeToCoordinate(bkDate);
      if (xBk !== null && xBk >= 0 && xBk <= w) {
        ctx.save();
        ctx.strokeStyle = isBk ? 'rgba(16, 185, 129, 0.85)' : 'rgba(245, 158, 11, 0.85)';
        ctx.lineWidth = isCurrent ? 2 : 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(xBk, 0);
        ctx.lineTo(xBk, h);
        ctx.stroke();
        ctx.restore();

        // Breakout pill at top of vertical line
        const badgeText = isBk 
          ? `🚀 BREAKOUT ${setup.trigger ? '$' + setup.trigger.toFixed(2) : ''}`
          : `⏳ COILING`;
        ctx.font = 'bold 10.5px monospace';
        const textW = ctx.measureText(badgeText).width;
        const pillW = textW + 14;
        const pillH = 18;
        const pillX = Math.max(4, Math.min(w - pillW - 4, xBk - pillW / 2));
        const pillY = 6;

        ctx.fillStyle = 'rgba(15, 23, 42, 0.92)';
        ctx.strokeStyle = isBk ? '#10b981' : '#f59e0b';
        ctx.lineWidth = 1;
        ctx.beginPath();
        if (ctx.roundRect) {
          ctx.roundRect(pillX, pillY, pillW, pillH, 4);
        } else {
          ctx.rect(pillX, pillY, pillW, pillH);
        }
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = isBk ? '#34d399' : '#fbbf24';
        ctx.fillText(badgeText, pillX + 7, pillY + 13);
      }
    }

    // 4. Horizontal Trigger Line across consolidation into breakout
    if (setup.trigger && setup.trigger > 0) {
      const yTrig = candleSeries.priceToCoordinate(setup.trigger);
      if (yTrig !== null && yTrig >= 0 && yTrig <= h) {
        const xStart = cStart ? timeScale.timeToCoordinate(cStart) : 0;
        const xEnd = bkDate ? timeScale.timeToCoordinate(bkDate) : w;
        if (xStart !== null && xEnd !== null) {
          ctx.save();
          ctx.strokeStyle = '#10b981';
          ctx.lineWidth = isCurrent ? 1.5 : 1;
          ctx.setLineDash([5, 3]);
          ctx.beginPath();
          ctx.moveTo(Math.max(0, xStart), yTrig);
          ctx.lineTo(Math.min(w, xEnd + 25), yTrig);
          ctx.stroke();
          ctx.restore();
        }
      }
    }
  });
}

// Fetch Events List with active filters
function fetchEvents() {
  const p = new URLSearchParams();
  p.set('state', activeStateFilter);
  p.set('ep_link', activeEpFilter);
  p.set('macro', activeMacroFilter);

  const pat = document.getElementById('sel_pattern').value;
  if (pat !== 'all') p.set('pattern', pat);

  const mv = document.getElementById('sel_move').value;
  if (mv) p.set('min_move', mv);

  const cons = document.getElementById('sel_cons').value;
  if (cons !== 'all') p.set('cons_bars', cons);

  const pb = document.getElementById('sel_pullback').value;
  if (pb) p.set('max_pullback', pb);

  const bbPinchEl = document.getElementById('sel_bb_pinch');
  if (bbPinchEl && bbPinchEl.value !== 'all') p.set('bb_pinch', bbPinchEl.value);

  const r2El = document.getElementById('sel_r2');
  if (r2El && r2El.value !== 'all') p.set('r2', r2El.value);

  const sec = document.getElementById('sel_sector').value;
  if (sec !== 'all') p.set('sector', sec);

  const thm = document.getElementById('sel_theme').value;
  if (thm !== 'all') p.set('theme', thm);

  const rsEl = document.getElementById('sel_rs');
  if (rsEl) p.set('rs_filter', rsEl.value);

  const tierEl = document.getElementById('sel_strat_tier');
  if (tierEl && tierEl.value !== 'all') p.set('strat_tier', tierEl.value);

  const srch = document.getElementById('inp_search').value.trim();
  if (srch) p.set('search', srch);

  p.set('sort', currentSort.col);
  p.set('asc', currentSort.asc ? '1' : '0');

  if (strategyMode) {
    p.set('strategy_mode', '1');
    p.set('strat_macro', strategySettings.macro);
    p.set('strat_spy', strategySettings.spy);
    p.set('strat_stop', strategySettings.stop);
    p.set('strat_sizing', strategySettings.sizing);

    const curTier = document.getElementById('sel_strat_tier')?.value;
    if (curTier) strategySettings.tier = curTier;
    p.set('strat_tier', strategySettings.tier || 'all');

    const curExit = document.getElementById('sel_strat_exit')?.value;
    if (curExit) strategySettings.exit = curExit;
    p.set('strat_exit', strategySettings.exit || '10ema');

    p.set('strat_rs', strategySettings.rs);
    p.set('strat_fresh_ep', strategySettings.fresh_ep);
    p.set('strat_rvol', strategySettings.rvol);
    p.set('strat_qqq', strategySettings.qqq);
    p.set('strat_tech_lead', strategySettings.tech_lead);

    const curBb = document.getElementById('sel_bb_pinch')?.value;
    if (curBb && curBb !== 'all') strategySettings.bb_pinch = curBb;
    const curR2 = document.getElementById('sel_r2')?.value;
    if (curR2 && curR2 !== 'all') strategySettings.r2 = curR2;

    p.set('strat_bb_pinch', strategySettings.bb_pinch);
    p.set('strat_r2', strategySettings.r2);
  }

  fetch('/api/events?' + p.toString())
    .then(r => r.json())
    .then(data => {
      allEvents = data.events || [];
      updateKPIs(data.kpis || {});
      if (data.strategy_kpis) {
        updateStrategyKPIs(data.strategy_kpis);
      }
      populateDropdowns(
        data.sectors || [],
        data.themes || [],
        data.patterns || [],
        data.leading_sectors || [],
        data.leading_themes || [],
        data.emerging_themes || []
      );
      renderTable();
      updateEnlargedStockCount();
      const eslModal = document.getElementById('enlarged_stock_list_modal');
      if (eslModal && eslModal.classList.contains('active')) {
        renderEnlargedStockList();
      }
      if (allEvents.length > 0) {
        const matched = currentEvent ? allEvents.find(e => e.symbol === currentEvent.symbol && e.date === currentEvent.date) : null;
        selectEvent(matched || allEvents[0]);
      } else {
        clearDossier();
      }
    })
    .catch(err => {
      console.error('Failed to fetch events:', err);
    });
}

function updateKPIs(k) {
  document.getElementById('kpi_total').textContent = (k.total || 0).toLocaleString();
  document.getElementById('kpi_breakouts').textContent = (k.breakouts || 0).toLocaleString();
  document.getElementById('kpi_building').textContent = (k.building || 0).toLocaleString();
  document.getElementById('kpi_ignited').textContent = (k.ignited_pct || 0).toFixed(1) + '%';
  document.getElementById('kpi_post').textContent = (k.post_pct || 0).toFixed(1) + '%';
  document.getElementById('kpi_avg_move').textContent = '+' + (k.avg_move || 0).toFixed(1) + '%';
  if (document.getElementById('kpi_thematic_lead')) {
    document.getElementById('kpi_thematic_lead').textContent = (k.thematic_lead_pct != null ? k.thematic_lead_pct.toFixed(1) : '0.0') + '%';
  }
  if (document.getElementById('kpi_rs_leader')) {
    document.getElementById('kpi_rs_leader').textContent = (k.rs_leader_pct != null ? k.rs_leader_pct.toFixed(1) : '0.0') + '%';
  }
}

function populateDropdowns(sectors, themes, patterns, leadingSectors, leadingThemes, emergingThemes) {
  const leadSecSet = new Set(leadingSectors || []);
  const leadThmSet = new Set(leadingThemes || []);
  const emergThmSet = new Set(emergingThemes || []);

  const selSec = document.getElementById('sel_sector');
  if (selSec.options.length <= 2) {
    const sortedSecs = [...sectors].sort((a, b) => {
      const aL = leadSecSet.has(a) ? 1 : 0;
      const bL = leadSecSet.has(b) ? 1 : 0;
      if (aL !== bL) return bL - aL;
      return a.localeCompare(b);
    });

    sortedSecs.forEach(s => {
      const opt = document.createElement('option');
      opt.value = s;
      if (leadSecSet.has(s)) {
        opt.textContent = `🏭 ${s} (Leading)`;
        opt.style.color = '#38bdf8';
        opt.style.fontWeight = '600';
      } else {
        opt.textContent = s;
      }
      selSec.appendChild(opt);
    });
  }

  const selThm = document.getElementById('sel_theme');
  if (selThm.options.length <= 3) {
    const sortedThms = [...themes].sort((a, b) => {
      const aRank = emergThmSet.has(a) ? 2 : (leadThmSet.has(a) ? 1 : 0);
      const bRank = emergThmSet.has(b) ? 2 : (leadThmSet.has(b) ? 1 : 0);
      if (aRank !== bRank) return bRank - aRank;
      return a.localeCompare(b);
    });

    sortedThms.forEach(t => {
      const opt = document.createElement('option');
      opt.value = t;
      if (emergThmSet.has(t)) {
        opt.textContent = `🌟 ${t} (Emerging)`;
        opt.style.color = '#fbbf24';
        opt.style.fontWeight = '600';
      } else if (leadThmSet.has(t)) {
        opt.textContent = `🚀 ${t} (Leading)`;
        opt.style.color = '#c084fc';
        opt.style.fontWeight = '600';
      } else {
        opt.textContent = t;
      }
      selThm.appendChild(opt);
    });
  }

  const selPat = document.getElementById('sel_pattern');
  if (selPat.options.length <= 1) {
    patterns.forEach(p => {
      const opt = document.createElement('option');
      opt.value = p; opt.textContent = p.replace('_', ' ').toUpperCase();
      selPat.appendChild(opt);
    });
  }
}

function renderTable() {
  const tbody = document.getElementById('events_tbody');
  tbody.innerHTML = '';

  const cntEl = document.getElementById('table_setups_count');
  if (cntEl) cntEl.textContent = `${allEvents.length} Setups`;

  if (allEvents.length === 0) {
    let msg = 'No matching QM setups found.';
    const curPinch = document.getElementById('sel_bb_pinch')?.value;
    if (curPinch && curPinch !== 'all') {
      msg += `<div style="margin-top:8px; font-size:11px; color:#38bdf8;">💡 <b>Tip:</b> BB Squeeze is set to <i>${curPinch}</i>. Set <b>BB Squeeze to "All"</b> to allow the ML model to evaluate volatility dynamically.</div>`;
    }
    tbody.innerHTML = `<tr><td colspan="13" style="text-align:center; padding:35px 20px; color:var(--muted); line-height:1.6;">${msg}</td></tr>`;
    updateHeaderCheckbox();
    return;
  }

  allEvents.forEach((ev, idx) => {
    const tr = document.createElement('tr');
    tr.id = `row_${ev.symbol}_${ev.date}`;
    tr.onclick = () => selectEvent(ev);

    const isBk = (ev.state === 'breakout');
    const statePill = isBk ? '<span class="pill-breakout">BREAKOUT</span>' : '<span class="pill-building">BUILDING</span>';
    
    // Relative Strength (RS) score & posture badge
    let rsPill = '·';
    if (ev.rs_score != null) {
      let rsColor = '#94a3b8';
      let rsSub = '';
      if (ev.rs_score >= 90) {
        rsColor = '#38bdf8';
        if (ev.stage2) rsSub = ' <span style="color:#10b981; font-size:9.5px; font-weight:700;" title="Stage 2 Bullish Stack">S2</span>';
        else if (ev.dist_52w_high != null && ev.dist_52w_high >= -15.0) rsSub = ' <span style="color:#f59e0b; font-size:9.5px; font-weight:700;" title="Within 15% of 52W High">ATH</span>';
      } else if (ev.rs_score >= 80) {
        rsColor = '#34d399';
      }
      rsPill = `<span style="font-weight:700; color:${rsColor};">${ev.rs_score}</span>${rsSub}`;
    }

    // EP badge pill
    let epBadgeHtml = '<span class="badge-ep-tech">⚪ Pure Technical</span>';
    if (ev.has_prior_ep && ev.has_post_ep) {
      epBadgeHtml = `<span class="badge-ep-both">⚡🚀 Ignited (${ev.prior_ep_days_ago}d) & Boosted</span>`;
    } else if (ev.has_prior_ep) {
      epBadgeHtml = `<span class="badge-ep-ignited">⚡ Prior EP (${ev.prior_ep_days_ago}d ago, +${ev.prior_ep_gap_pct}%)</span>`;
    } else if (ev.has_post_ep) {
      epBadgeHtml = `<span class="badge-ep-boosted">🚀 Post EP (${ev.post_ep_days_later}d later, +${ev.post_ep_gap_pct}%)</span>`;
    }

    // Macro flow badge
    let macroBadge = '';
    if (ev.is_emerging_theme) {
      macroBadge = ' <span class="badge" style="background:#78350f; color:#fde68a; font-size:9.5px; margin-left:4px;" title="Emerging Theme: Velocity Surge / Acceleration">🌟 Emerging</span>';
    } else if (ev.is_leading_theme) {
      macroBadge = ' <span class="badge" style="background:#4c1d95; color:#ddd6fe; font-size:9.5px; margin-left:4px;" title="Leading Theme: Score ≥ 65">🚀 Leading</span>';
    } else if (ev.is_leading_sector) {
      macroBadge = ' <span class="badge" style="background:#1e3a8a; color:#bfdbfe; font-size:9.5px; margin-left:4px;" title="Leading Sector: Score ≥ 65">🏭 Leading</span>';
    }

    // Machine Learning Quality Tier badge
    let mlPill = '';
    if (ev.qm_tier === 'Premier') {
      mlPill = ' <span class="badge" style="background:#065f46; color:#6ee7b7; font-size:9.5px; font-weight:700; margin-left:4px;" title="ML Setup Quality: Premier Pinnacle (Score ≥ 75, Failure Risk < 22%)">💎 PREMIER</span>';
    } else if (ev.qm_tier === 'High Quality') {
      mlPill = ' <span class="badge" style="background:#064e3b; color:#a7f3d0; font-size:9.5px; font-weight:700; margin-left:4px;" title="ML Setup Quality: High Quality Setup (Score ≥ 60)">⭐ HIGH</span>';
    } else if (ev.qm_tier === 'Toxic / Trap') {
      mlPill = ' <span class="badge" style="background:#7f1d1d; color:#fca5a5; font-size:9.5px; font-weight:700; margin-left:4px;" title="ML Trap Warning: Toxic Failure Risk ≥ 35%">⚠️ TRAP</span>';
    }

    const k = `${ev.symbol}_${ev.date}`;
    const isChecked = selectedSetupKeys.has(k) ? 'checked' : '';
    const pal = SETUP_PALETTE[idx % SETUP_PALETTE.length];
    const r2Str = (ev.r2 != null) ? `<span style="font-weight:700; color:#38bdf8;">${ev.r2.toFixed(2)}</span>` : '<span style="color:#64748b;">-</span>';
    const pinchStr = (ev.pinch_ratio != null) ? `<span style="font-weight:700; color:#fbbf24;">${ev.pinch_ratio.toFixed(2)}x</span>` : '<span style="color:#64748b;">-</span>';

    tr.innerHTML = `
      <td style="text-align:center;" onclick="event.stopPropagation()">
        <input type="checkbox" class="setup-cb" id="cb_${ev.symbol}_${ev.date}" ${isChecked} onchange="toggleSetupCheckbox('${ev.symbol}', '${ev.date}', this.checked)">
      </td>
      <td style="text-align:center; font-weight:700; color:${pal.stroke}; white-space:nowrap;">
        <span class="color-dot" style="background:${pal.stroke};"></span>#${idx + 1}
      </td>
      <td class="tl"><strong style="color:#38bdf8;">${ev.symbol}</strong>${macroBadge}</td>
      <td class="tl" style="font-family:monospace; color:#94a3b8;">${ev.date}</td>
      <td class="tl">${statePill}${mlPill}</td>
      <td style="text-align:center;">${rsPill}</td>
      <td style="color:#34d399; font-weight:600;">+${ev.move_pct.toFixed(1)}%</td>
      <td>${ev.cons_bars}d</td>
      <td style="color:${ev.pullback_pct > 25 ? '#f87171' : '#cbd5e1'}">-${ev.pullback_pct.toFixed(1)}%</td>
      <td style="text-align:center;">${r2Str}</td>
      <td style="text-align:center;">${pinchStr}</td>
      <td class="tl" style="font-size:11px; color:#cbd5e1;">${ev.pattern}</td>
      <td class="tl">${epBadgeHtml}</td>
    `;
    tbody.appendChild(tr);
  });

  updateHeaderCheckbox();
}

function toggleSetupCheckbox(sym, date, checked) {
  const k = `${sym}_${date}`;
  if (checked) {
    selectedSetupKeys.add(k);
  } else {
    selectedSetupKeys.delete(k);
  }
  updateHeaderCheckbox();

  if (currentChartSymbol === sym) {
    drawCanvasOverlays();
  } else if (checked) {
    const ev = allEvents.find(e => e.symbol === sym && e.date === date);
    if (ev) selectEvent(ev);
  }
}

function updateHeaderCheckbox() {
  const hcb = document.getElementById('header_select_all');
  if (!hcb) return;
  if (!allEvents || allEvents.length === 0) {
    hcb.checked = false;
    hcb.indeterminate = false;
  } else {
    const visibleKeys = allEvents.map(e => `${e.symbol}_${e.date}`);
    const selectedCount = visibleKeys.filter(k => selectedSetupKeys.has(k)).length;
    if (selectedCount === visibleKeys.length && visibleKeys.length > 0) {
      hcb.checked = true;
      hcb.indeterminate = false;
    } else if (selectedCount === 0) {
      hcb.checked = false;
      hcb.indeterminate = false;
    } else {
      hcb.checked = false;
      hcb.indeterminate = true;
    }
  }
}

function toggleSelectAll(checked) {
  allEvents.forEach(ev => {
    const k = `${ev.symbol}_${ev.date}`;
    if (checked) selectedSetupKeys.add(k);
    else selectedSetupKeys.delete(k);
  });
  document.querySelectorAll('.setup-cb').forEach(cb => { cb.checked = checked; });
  updateHeaderCheckbox();
  drawCanvasOverlays();
}

function selectAllSetups() { toggleSelectAll(true); }
function clearSelectedSetups() { toggleSelectAll(false); }

function selectEvent(ev) {
  currentEvent = ev;
  document.querySelectorAll('#events_tbody tr').forEach(r => r.classList.remove('selected'));
  const activeRow = document.getElementById(`row_${ev.symbol}_${ev.date}`);
  if (activeRow) activeRow.classList.add('selected');

  const k = `${ev.symbol}_${ev.date}`;
  if (!selectedSetupKeys.has(k)) {
    if (currentChartSymbol !== ev.symbol) {
      selectedSetupKeys.clear();
      document.querySelectorAll('.setup-cb').forEach(cb => { cb.checked = false; });
    }
    selectedSetupKeys.add(k);
    const cb = document.getElementById(`cb_${ev.symbol}_${ev.date}`);
    if (cb) cb.checked = true;
  }
  updateHeaderCheckbox();

  updateEnlargedStockCount();
  loadChart(ev.symbol, ev.date);
}

function loadChart(sym, dateStr) {
  currentChartSymbol = sym;
  loadDrawingsForSymbol(sym);

  const lFastest = document.getElementById('cfg_larsson_fastest') ? document.getElementById('cfg_larsson_fastest').value : '32';
  fetch(`/api/chart?symbol=${encodeURIComponent(sym)}&date=${encodeURIComponent(dateStr)}&larsson_fastest=${encodeURIComponent(lFastest)}`)
    .then(r => r.json())
    .then(data => {
      currentChartData = data;
      renderChart(data);
      renderDossier(data.setup, data.all_eps || [], data.sector_theme, data.catalyst);
    })
    .catch(err => {
      console.error('Failed to load chart:', err);
    });
}

function applySetupPriceLines(setup) {
  activePriceLines.forEach(pl => {
    try { candleSeries.removePriceLine(pl); } catch (e) {}
  });
  activePriceLines = [];

  if (!showPriceLines || !setup) return;

  if (setup.trigger && setup.trigger > 0) {
    activePriceLines.push(candleSeries.createPriceLine({
      price: setup.trigger, color: '#10b981', lineWidth: 2, lineStyle: 0, axisLabelVisible: true, title: `Trigger $${setup.trigger}`
    }));
  }
  if (setup.level && setup.level > 0) {
    activePriceLines.push(candleSeries.createPriceLine({
      price: setup.level, color: '#ef4444', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: `Base Low $${setup.level}`
    }));
  }
  (setup.stack_levels || []).forEach(lvl => {
    if (lvl && lvl > 0 && lvl !== setup.trigger && lvl !== setup.level) {
      activePriceLines.push(candleSeries.createPriceLine({
        price: lvl, color: '#f59e0b', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: `Stack $${lvl}`
      }));
    }
  });

  if (setup.trade_info) {
    const ti = setup.trade_info;
    const curStop = (typeof strategySettings !== 'undefined' && strategySettings.stop) ? strategySettings.stop : '06x';
    const isStop10 = (curStop === '10x');
    let stopP = ti.stop_loss_06x;
    let stopLbl = '0.6x';
    if (curStop === 'dynamic' && ti.stop_loss_dynamic && ti.stop_loss_dynamic > 0) {
      stopP = ti.stop_loss_dynamic;
      stopLbl = 'Dynamic';
    } else if (isStop10) {
      stopP = ti.stop_loss_10x;
      stopLbl = '1.0x';
    }
    if (stopP && stopP > 0) {
      activePriceLines.push(candleSeries.createPriceLine({
        price: stopP, color: '#f87171', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: `Stop (${stopLbl}) $${stopP.toFixed(2)}`
      }));
    }
    const exitP = isStop10 ? ti.exit_price_10x : ti.exit_price_06x;
    if (exitP && exitP > 0) {
      activePriceLines.push(candleSeries.createPriceLine({
        price: exitP, color: '#38bdf8', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: `Exit $${exitP.toFixed(2)}`
      }));
    }
    const curExit = (typeof strategySettings !== 'undefined' && strategySettings.exit) ? strategySettings.exit : '10ema';
    const isClimaxModel = curExit.startsWith('climax_') || curExit === 'triad_optimal';
    if (isClimaxModel) {
      const cP = isStop10 ? ti.climax_price_10x : ti.climax_price_06x;
      if (cP && cP > 0) {
        activePriceLines.push(candleSeries.createPriceLine({
          price: cP, color: '#c084fc', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: `Climax $${cP.toFixed(2)}`
        }));
      }
    }
  }
}

function applyCandleMarkers(data) {
  if (!candleSeries || !data) return;
  if (!showEpMarkers) {
    candleSeries.setMarkers([]);
    return;
  }
  const setup = data.setup || {};
  const isBk = (setup.state === 'breakout');
  const markers = [];
  (data.all_eps || []).forEach(ep => {
    const isPrior = (ep.date === setup.prior_ep_date);
    const isPost = (ep.date === setup.post_ep_date);

    if (isPrior) {
      markers.push({
        time: ep.date,
        position: 'belowBar',
        color: '#06b6d4',
        shape: 'arrowUp',
        text: `⚡ PRIOR EP: +${ep.gap_pct}% (RVOL ${ep.rvol}x)`
      });
    } else if (isPost) {
      markers.push({
        time: ep.date,
        position: 'aboveBar',
        color: '#a855f7',
        shape: 'arrowDown',
        text: `🚀 POST EP: +${ep.gap_pct}% (RVOL ${ep.rvol}x)`
      });
    } else {
      markers.push({
        time: ep.date,
        position: 'belowBar',
        color: '#475569',
        shape: 'circle',
        text: `EP +${ep.gap_pct}%`
      });
    }
  });

  if (setup.date) {
    markers.push({
      time: setup.date,
      position: isBk ? 'aboveBar' : 'belowBar',
      color: isBk ? '#10b981' : '#f59e0b',
      shape: 'circle',
      text: isBk ? '🚀 QM BREAKOUT' : '⏳ QM COIL'
    });
  }

  if (setup.trade_info) {
    const ti = setup.trade_info;
    const curStop = (typeof strategySettings !== 'undefined' && strategySettings.stop) ? strategySettings.stop : '06x';
    const curExit = (typeof strategySettings !== 'undefined' && strategySettings.exit) ? strategySettings.exit : 'climax_swing';
    const isClimaxModel = curExit.startsWith('climax_') || curExit === 'triad_optimal';

    if (isClimaxModel) {
      const isStop10 = (curStop === '10x');
      const cDate = isStop10 ? ti.climax_date_10x : ti.climax_date_06x;
      const cR = isStop10 ? ti.climax_r_10x : ti.climax_r_06x;
      const cP = isStop10 ? ti.climax_price_10x : ti.climax_price_06x;
      const cProb = isStop10 ? ti.prob_exhaustion_10x : ti.prob_exhaustion_06x;
      if (cDate && cP) {
        markers.push({
          time: cDate,
          position: 'aboveBar',
          color: '#c084fc',
          shape: 'arrowDown',
          text: `🎯 ML CLIMAX: +${cR}R ($${cP.toFixed(2)}) P=${cProb}`
        });
      }
    }
  }

  markers.sort((a, b) => a.time.localeCompare(b.time));
  candleSeries.setMarkers(markers);
}

function renderChart(data) {
  if (!chart) initChart();

  // Clear previous price lines
  activePriceLines.forEach(pl => {
    try { candleSeries.removePriceLine(pl); } catch (e) {}
  });
  activePriceLines = [];

  // Set Data
  candleSeries.setData(data.bars || []);
  volumeSeries.setData(data.volume || []);
  if (volMa20Series) volMa20Series.setData(data.vol_ma20 || []);
  ema10Series.setData(data.ema10 || []);
  ema20Series.setData(data.ema20 || []);
  sma50Series.setData(data.sma50 || []);
  sma200Series.setData(data.sma200 || []);

  window._lastLarssonData = data.larsson || [];
  updateNativeLarssonSeries(window._lastLarssonData);
  scheduleLarssonRender();
  updateLarssonLegendValues();

  if (data.larsson_spans) {
    const spEl = document.getElementById('lbl_larsson_spans');
    if (spEl) spEl.textContent = `(${data.larsson_spans.join(', ')})`;
  }

  const setup = data.setup;

  // Header Title
  document.getElementById('chart_sym').textContent = setup.symbol;
  document.getElementById('chart_date').textContent = setup.date;
  const isBk = (setup.state === 'breakout');
  document.getElementById('chart_state_pill').innerHTML = isBk ? '<span class="pill-breakout">BREAKOUT</span>' : '<span class="pill-building">BUILDING</span>';
  document.getElementById('chart_ep_badge').innerHTML = setup.ep_badge ? `<span class="badge-ep-ignited">${setup.ep_badge}</span>` : '';

  // Resistance Stack Price Lines
  applySetupPriceLines(setup);

  // Candlestick Markers
  applyCandleMarkers(data);

  // Set visible range around consolidation & pole and auto-scale price axis to new ticker
  resetChartScale();
}

function resetChartScale() {
  if (!chart || !currentChartData) return;
  const setup = currentChartData.setup;
  const data = currentChartData;

  const container = document.getElementById('chart_tv');
  if (container && container.clientWidth > 0 && container.clientHeight > 0) {
    chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
  }

  if (data.bars && data.bars.length > 0) {
    let evIdx = data.bars.findIndex(b => b.time === setup.date);
    if (evIdx < 0) evIdx = data.bars.length - 1;

    let poleStartIdx = setup.qm_pole_start ? data.bars.findIndex(b => b.time === setup.qm_pole_start) : -1;
    let priorEpIdx = setup.prior_ep_date ? data.bars.findIndex(b => b.time === setup.prior_ep_date) : -1;

    let leadIdx = evIdx;
    if (poleStartIdx >= 0 && poleStartIdx < leadIdx) leadIdx = poleStartIdx;
    if (priorEpIdx >= 0 && priorEpIdx < leadIdx) leadIdx = priorEpIdx;

    const fromIdx = Math.max(0, leadIdx - 15);
    const toIdx = Math.min(data.bars.length - 1, evIdx + 25);

    chart.timeScale().setVisibleLogicalRange({ from: fromIdx, to: toIdx });
  } else {
    chart.timeScale().fitContent();
  }

  // Force price scale to autoScale strictly to the new ticker's price range
  chart.priceScale('right').applyOptions({
    autoScale: true,
    scaleMargins: { top: 0.08, bottom: 0.22 },
  });
  chart.priceScale('volume').applyOptions({
    scaleMargins: { top: 0.85, bottom: 0.0 },
  });

  requestAnimationFrame(drawCanvasOverlays);
  setTimeout(drawCanvasOverlays, 60);
  setTimeout(drawCanvasOverlays, 180);
}

function renderDossier(s, eps, st, cat) {
  document.getElementById('dos_status_pill').innerHTML = (s.state === 'breakout')
    ? '<span class="pill-breakout">BREAKOUT</span>' : '<span class="pill-building">BUILDING</span>';
  document.getElementById('dos_sym_date').textContent = `${s.symbol} • ${s.date}`;
  document.getElementById('dos_sector').textContent = s.sector || 'Unknown';
  document.getElementById('dos_themes').textContent = s.themes || s.theme || 'General';

  // ML Setup Quality & Dynamic Trade Execution Card
  const mlCard = document.getElementById('card_ml_guidance');
  if (mlCard) {
    const ti = s ? (s.trade_info || (s.ml_guidance ? {
      qm_tier: s.ml_guidance.quality?.tier,
      qm_score: s.ml_guidance.quality?.qm_score,
      p_big_move: s.ml_guidance.quality?.p_big,
      p_trap: s.ml_guidance.quality?.p_trap,
      conviction_sizing: s.ml_guidance.quality?.sizing,
      ml_recommendation: s.ml_guidance.quality?.recommendation,
      stop_loss_dynamic: s.ml_guidance.stop_loss?.dynamic_stop_price,
      risk_pct_dynamic: s.ml_guidance.stop_loss?.risk_pct
    } : null)) : null;

    if (ti && (ti.qm_tier || ti.qm_score !== undefined)) {
      mlCard.style.display = 'flex';
      const tierBadge = document.getElementById('dos_ml_tier_badge');
      if (tierBadge) {
        if (ti.qm_tier === 'Premier') {
          tierBadge.className = 'pctile-badge pctile-high';
          tierBadge.textContent = '💎 PREMIER PINNACLE';
        } else if (ti.qm_tier === 'High Quality') {
          tierBadge.className = 'pctile-badge pctile-high';
          tierBadge.textContent = '⭐ HIGH QUALITY';
        } else if (ti.qm_tier === 'Toxic / Trap') {
          tierBadge.className = 'pctile-badge pctile-low';
          tierBadge.textContent = '⚠️ TOXIC TRAP';
        } else {
          tierBadge.className = 'pctile-badge pctile-mid';
          tierBadge.textContent = ti.qm_tier || 'Standard';
        }
      }

      document.getElementById('dos_ml_rec_text').textContent = ti.ml_recommendation || 'Standard Breakout Setup';
      document.getElementById('dos_ml_score').textContent = `${(ti.qm_score !== null && ti.qm_score !== undefined) ? ti.qm_score.toFixed(1) : '50.0'} / 100`;
      document.getElementById('dos_ml_sizing').textContent = `${(ti.conviction_sizing !== null && ti.conviction_sizing !== undefined) ? ti.conviction_sizing.toFixed(2) : '1.00'}x Conviction`;

      if (ti.stop_loss_dynamic && ti.stop_loss_dynamic > 0) {
        document.getElementById('dos_ml_stop').textContent = `$${ti.stop_loss_dynamic.toFixed(2)}`;
        document.getElementById('dos_ml_risk').textContent = `-${ti.risk_pct_dynamic ? ti.risk_pct_dynamic.toFixed(1) : '3.5'}% risk`;
      } else {
        document.getElementById('dos_ml_stop').textContent = '--';
        document.getElementById('dos_ml_risk').textContent = '--';
      }

      document.getElementById('dos_ml_p_big').textContent = (ti.p_big_move !== null && ti.p_big_move !== undefined) ? `${(ti.p_big_move * 100).toFixed(1)}%` : '--';
      document.getElementById('dos_ml_p_trap').textContent = (ti.p_trap !== null && ti.p_trap !== undefined) ? `${(ti.p_trap * 100).toFixed(1)}%` : '--';

      const tgts = s?.ml_guidance?.targets;
      const el25r = document.getElementById('dos_ml_hurdle_25r');
      const el35r = document.getElementById('dos_ml_hurdle_35r');
      const elPlan = document.getElementById('dos_ml_plan');
      if (el25r) el25r.textContent = (tgts && tgts.hurdle_25r) ? `$${tgts.hurdle_25r.toFixed(2)}` : '--';
      if (el35r) el35r.textContent = (tgts && tgts.hurdle_35r) ? `$${tgts.hurdle_35r.toFixed(2)}` : '--';
      if (elPlan && tgts && tgts.scale_out_plan) elPlan.textContent = tgts.scale_out_plan;
    } else {
      mlCard.style.display = 'none';
    }
  }

  // QM Breakout Strategy Trade Execution & Sizing Card
  const stratCard = document.getElementById('card_strategy_trade');
  if (stratCard) {
    if (s && s.trade_info) {
      stratCard.style.display = 'flex';
      const ti = s.trade_info;
      const curStop = (typeof strategySettings !== 'undefined' && strategySettings.stop) ? strategySettings.stop : '06x';
      const curExit = (typeof strategySettings !== 'undefined' && strategySettings.exit) ? strategySettings.exit : '10ema';
      const exitMap = {
        '10ema': 'e10',
        '20ema': 'e20',
        '50ema': 'e50',
        'staged': 'staged',
        'balanced': 'balanced',
        'climax_swing': 'climax_swing',
        'climax_hybrid': 'climax_hybrid',
        'climax_trend': 'climax_trend',
        'triad_optimal': 'triad_optimal'
      };
      const isStop10 = (curStop === '10x');
      const stSfx = isStop10 ? '10x' : '06x';

      let stopPrice = ti.stop_loss_06x;
      let riskPct = ti.risk_pct_06x;
      let stopLabel = '0.6x';
      if (curStop === 'dynamic' && ti.stop_loss_dynamic && ti.stop_loss_dynamic > 0) {
        stopPrice = ti.stop_loss_dynamic;
        riskPct = ti.risk_pct_dynamic || (((ti.entry_price - stopPrice) / ti.entry_price) * 100);
        stopLabel = 'Dynamic Stop';
      } else if (curStop === '10x') {
        stopPrice = ti.stop_loss_10x;
        riskPct = ti.risk_pct_10x;
        stopLabel = '1.0x';
      }

      let realizedR = isStop10 ? ti.r_10x : ti.r_06x;
      let sizedR = isStop10 ? ti.sized_r_10x : ti.sized_r_06x;
      let barsHeld = isStop10 ? ti.bars_10x : ti.bars_06x;
      let isWin = isStop10 ? ti.is_win_10x : ti.is_win_06x;
      let gainPct = isStop10 ? ti.gain_pct_10x : ti.gain_pct_06x;

      if (curExit === 'triad_optimal') {
        realizedR = (ti.r_triad_optimal != null) ? ti.r_triad_optimal : ((ti.r_dyn_climax_swing != null) ? ti.r_dyn_climax_swing : ti.r_climax_swing_06x);
        sizedR = (ti.sized_r_triad_optimal != null) ? ti.sized_r_triad_optimal : ((ti.sized_r_dyn_climax_swing != null) ? ti.sized_r_dyn_climax_swing : ti.sized_r_climax_swing_06x);
        barsHeld = (ti.bars_triad_optimal != null) ? ti.bars_triad_optimal : ti.bars_climax_swing_06x;
        isWin = (ti.is_win_triad_optimal != null) ? ti.is_win_triad_optimal : ti.is_win_climax_swing_06x;
      } else if (curExit.startsWith('climax_')) {
        const kR = `r_${curExit}_${stSfx}`;
        const kSized = `sized_r_${curExit}_${stSfx}`;
        const kBars = `bars_${curExit}_${stSfx}`;
        const kWin = `is_win_${curExit}_${stSfx}`;
        realizedR = (ti[kR] != null) ? ti[kR] : (isStop10 ? ti.r_10x : ti.r_06x);
        sizedR = (ti[kSized] != null) ? ti[kSized] : (isStop10 ? ti.sized_r_10x : ti.sized_r_06x);
        barsHeld = (ti[kBars] != null) ? ti[kBars] : (isStop10 ? ti.bars_10x : ti.bars_06x);
        isWin = (ti[kWin] != null) ? ti[kWin] : (isStop10 ? ti.is_win_10x : ti.is_win_06x);
      } else {
        const exitMap = { '10ema': 'e10', '20ema': 'e20', '50ema': 'e50', 'staged': 'staged', 'balanced': 'balanced' };
        const exPfx = exitMap[curExit] || 'e10';
        const kR = `r_${exPfx}_${stSfx}`;
        const kSized = `sized_r_${exPfx}_${stSfx}`;
        const kBars = `bars_${exPfx}_${stSfx}`;
        const kWin = `is_win_${exPfx}_${stSfx}`;
        realizedR = (ti[kR] != null) ? ti[kR] : (isStop10 ? ti.r_10x : ti.r_06x);
        sizedR = (ti[kSized] != null) ? ti[kSized] : (isStop10 ? ti.sized_r_10x : ti.sized_r_06x);
        barsHeld = (ti[kBars] != null) ? ti[kBars] : (isStop10 ? ti.bars_10x : ti.bars_06x);
        isWin = (ti[kWin] != null) ? ti[kWin] : (isStop10 ? ti.is_win_10x : ti.is_win_06x);
      }

      const exitPrice = isStop10 ? ti.exit_price_10x : ti.exit_price_06x;
      const exitDate = isStop10 ? ti.exit_date_10x : ti.exit_date_06x;
      const exitReason = isStop10 ? ti.exit_reason_10x : ti.exit_reason_06x;
      const mfeR = isStop10 ? ti.mfe_r_10x : ti.mfe_r_06x;
      const maeR = isStop10 ? ti.mae_r_10x : ti.mae_r_06x;

      const isClimaxModel = curExit.startsWith('climax_') || curExit === 'triad_optimal';
      const cDate = isStop10 ? ti.climax_date_10x : ti.climax_date_06x;
      const cP = isStop10 ? ti.climax_price_10x : ti.climax_price_06x;
      const cR = isStop10 ? ti.climax_r_10x : ti.climax_r_06x;
      const cProb = isStop10 ? ti.prob_exhaustion_10x : ti.prob_exhaustion_06x;

      const badgeEl = document.getElementById('dos_strat_result_badge');
      if (badgeEl) {
        if (isWin) {
          badgeEl.className = 'pctile-badge pctile-high';
          badgeEl.textContent = `🟢 WIN (${realizedR > 0 ? '+' : ''}${realizedR.toFixed(2)} R)`;
        } else {
          badgeEl.className = 'pctile-badge pctile-low';
          badgeEl.textContent = `🔴 LOSS (${realizedR.toFixed(2)} R)`;
        }
      }

      document.getElementById('dos_strat_entry').textContent = `$${ti.entry_price.toFixed(2)}`;
      document.getElementById('dos_strat_stop').textContent = `$${stopPrice.toFixed(2)} (-${riskPct.toFixed(1)}%) [${stopLabel}]`;

      const partialRow = document.getElementById('dos_strat_partial_row');
      const exitLbl = document.getElementById('dos_strat_exit_lbl');

      if (isClimaxModel && cDate && cP) {
        if (partialRow) {
          partialRow.style.display = 'flex';
          document.getElementById('dos_strat_partial_exit').textContent = `50% @ $${cP.toFixed(2)} (+${cR}R) on ${cDate} (P=${cProb})`;
        }
        if (exitLbl) exitLbl.textContent = 'Runner Exit (50%):';
        document.getElementById('dos_strat_exit').textContent = `$${exitPrice.toFixed(2)} on ${exitDate}`;
      } else {
        if (partialRow) partialRow.style.display = 'none';
        if (exitLbl) {
          exitLbl.textContent = curExit === '20ema' ? '20 EMA Exit (100%):' : (curExit === '50ema' ? '50 EMA Exit (100%):' : '10 EMA Exit (100%):');
        }
        document.getElementById('dos_strat_exit').textContent = `$${exitPrice.toFixed(2)} on ${exitDate}`;
      }

      let humanExit = '10 EMA Close Violation';
      if (curExit === '20ema') humanExit = '20 EMA Close Violation';
      else if (curExit === '50ema') humanExit = '50 EMA Close Violation';
      else if (curExit === 'staged') humanExit = 'Staged Scaling (33% @ 10E / 33% @ 20E / 34% @ 50E)';
      else if (curExit === 'balanced') humanExit = 'Balanced Scaling (50% @ 10E / 50% @ 50E)';
      else if (curExit === 'climax_swing') humanExit = 'ML Climax Swing (50% Climax @ P≥0.45, 50% 10 EMA)';
      else if (curExit === 'climax_hybrid') humanExit = 'ML Climax Hybrid (33% Climax @ P≥0.50, 33% 10E, 34% 20E)';
      else if (curExit === 'climax_trend') humanExit = 'ML Climax Trend (50% Climax @ P≥0.65, 50% 50 EMA)';
      else if (curExit === 'triad_optimal') humanExit = '👑 Institutional Triad (Dynamic Stop + Climax Partial + Runner)';

      if (exitReason && exitReason.includes('stop')) humanExit = 'Initial Stop Loss Breached';
      else if (exitReason && exitReason.includes('max')) humanExit = 'Max Holding (250 bars)';
      document.getElementById('dos_strat_exit_reason').textContent = humanExit;

      const rEl = document.getElementById('dos_strat_r');
      rEl.textContent = `${realizedR > 0 ? '+' : ''}${realizedR.toFixed(2)} R`;
      rEl.style.color = realizedR > 0 ? '#34d399' : '#f87171';
      document.getElementById('dos_strat_gain_pct').textContent = `${gainPct > 0 ? '+' : ''}${gainPct.toFixed(1)}% gain`;

      const sizedEl = document.getElementById('dos_strat_sized_r');
      sizedEl.textContent = `${sizedR > 0 ? '+' : ''}${sizedR.toFixed(2)} R`;
      sizedEl.style.color = sizedR > 0 ? '#c084fc' : '#f87171';
      document.getElementById('dos_strat_size_mult').textContent = `Conviction: ${ti.pos_size_mult.toFixed(2)}x`;

      document.getElementById('dos_strat_mfe').textContent = `+${mfeR.toFixed(2)} R`;
      document.getElementById('dos_strat_mae').textContent = `-${maeR.toFixed(2)} R`;
      document.getElementById('dos_strat_bars').textContent = `${barsHeld} bars`;

      // Sizing Factors
      document.getElementById('dos_strat_total_mult_badge').textContent = `${ti.pos_size_mult.toFixed(2)}x Sizing`;
      document.getElementById('dos_strat_s_ep').textContent = `${ti.s_ep.toFixed(2)}x`;
      document.getElementById('dos_strat_s_pole').textContent = `${ti.s_pole.toFixed(2)}x`;
      document.getElementById('dos_strat_s_rvol').textContent = `${ti.s_rvol.toFixed(2)}x`;
      document.getElementById('dos_strat_s_pb').textContent = `${ti.s_pb.toFixed(2)}x`;

      // Regime Checklist
      document.getElementById('dos_strat_spy_status').innerHTML = ti.spy_above_200sma
        ? '<span style="color:#10b981; font-weight:700;">✓ Bull Market (SPY ≥ 200 SMA)</span>'
        : '<span style="color:#f87171; font-weight:700;">✗ Bear Market (SPY &lt; 200 SMA)</span>';

      let qqqText = '<span style="color:#94a3b8;">Market Baseline</span>';
      if (ti.qqq_pullback) qqqText = '<span style="color:#fbbf24; font-weight:700;">Pullback Divergence (+0.238 R)</span>';
      else if (ti.qqq_bull_stack) qqqText = '<span style="color:#38bdf8; font-weight:700;">Bull Stack (+0.108 R)</span>';
      document.getElementById('dos_strat_qqq_status').innerHTML = qqqText;

      document.getElementById('dos_strat_tech_status').innerHTML = ti.qqq_spy_ratio_bull
        ? '<span style="color:#38bdf8; font-weight:700;">✓ Tech Leading ((QQQ/SPY) Bull)</span>'
        : '<span style="color:#94a3b8;">Tech Lagging / Neutral</span>';

      document.getElementById('dos_strat_macro_status').innerHTML = ti.is_leading_macro
        ? '<span style="color:#10b981; font-weight:700;">✓ Leading Sector/Theme (Score ≥ 65)</span>'
        : '<span style="color:#94a3b8;">Standard Universe</span>';
    } else {
      stratCard.style.display = 'none';
    }
  }

  // Sector & Theme Momentum Backdrop
  if (st) {
    document.getElementById('ctx_sec_name').textContent = st.sector || 'Unknown';
    const sec1El = document.getElementById('ctx_sec_m1');
    if (st.sec_m1_pctile != null) {
      sec1El.textContent = `1M: ${Math.round(st.sec_m1_pctile)}th`;
      sec1El.className = 'pctile-badge ' + (st.sec_m1_pctile >= 70 ? 'pctile-high' : (st.sec_m1_pctile >= 30 ? 'pctile-mid' : 'pctile-low'));
      sec1El.style.display = 'inline-block';
    } else { sec1El.style.display = 'none'; }

    const sec3El = document.getElementById('ctx_sec_m3');
    if (st.sec_m3_pctile != null) {
      sec3El.textContent = `3M: ${Math.round(st.sec_m3_pctile)}th`;
      sec3El.className = 'pctile-badge ' + (st.sec_m3_pctile >= 70 ? 'pctile-high' : (st.sec_m3_pctile >= 30 ? 'pctile-mid' : 'pctile-low'));
      sec3El.style.display = 'inline-block';
    } else { sec3El.style.display = 'none'; }

    document.getElementById('ctx_thm_name').textContent = st.theme || 'General';
    const thm1El = document.getElementById('ctx_thm_m1');
    if (st.thm_m1_pctile != null) {
      thm1El.textContent = `1M: ${Math.round(st.thm_m1_pctile)}th`;
      thm1El.className = 'pctile-badge ' + (st.thm_m1_pctile >= 70 ? 'pctile-high' : (st.thm_m1_pctile >= 30 ? 'pctile-mid' : 'pctile-low'));
      thm1El.style.display = 'inline-block';
    } else { thm1El.style.display = 'none'; }

    document.getElementById('ctx_archetype').textContent = st.primary_archetype || 'Market Baseline';
    document.getElementById('ctx_composite').textContent = st.composite_score != null ? st.composite_score.toFixed(1) : '50.0';

    const badgeEl = document.getElementById('ctx_diag_badge');
    badgeEl.className = 'pctile-badge ' + (st.macro_status === 'tailwind' ? 'pctile-high' : (st.macro_status === 'drag' ? 'pctile-low' : 'pctile-mid'));
    badgeEl.textContent = (st.macro_status === 'tailwind' ? '🟢 ' : (st.macro_status === 'drag' ? '🔴 ' : '🟡 ')) + st.macro_diag;

    const descEl = document.getElementById('ctx_diag_desc');
    descEl.className = 'diag-pill ' + (st.macro_status === 'tailwind' ? 'diag-tailwind' : (st.macro_status === 'drag' ? 'diag-drag' : 'diag-neutral'));
    descEl.textContent = st.macro_desc;
  }

  // Institutional Relative Strength & Posture Card
  const rsScoreEl = document.getElementById('dos_rs_score');
  const rsBadgeEl = document.getElementById('dos_rs_badge');
  const stage2El = document.getElementById('dos_stage2');
  const dist52wEl = document.getElementById('dos_dist_52w');
  const postureEl = document.getElementById('dos_rs_posture_status');

  if (s.rs_score != null) {
    if (rsScoreEl) rsScoreEl.textContent = `${s.rs_score} / 99`;
    if (rsBadgeEl) {
      if (s.rs_score >= 90) {
        rsBadgeEl.className = 'pctile-badge pctile-high';
        rsBadgeEl.textContent = `RS ${s.rs_score} 🔥 Top Decile`;
      } else if (s.rs_score >= 80) {
        rsBadgeEl.className = 'pctile-badge pctile-high';
        rsBadgeEl.textContent = `RS ${s.rs_score} Strong`;
      } else {
        rsBadgeEl.className = 'pctile-badge pctile-mid';
        rsBadgeEl.textContent = `RS ${s.rs_score}`;
      }
    }
  } else {
    if (rsScoreEl) rsScoreEl.textContent = '·';
    if (rsBadgeEl) {
      rsBadgeEl.className = 'pctile-badge pctile-low';
      rsBadgeEl.textContent = 'RS ·';
    }
  }

  if (stage2El) {
    stage2El.innerHTML = s.stage2
      ? '<span style="color:#10b981; font-weight:700;">✓ Bullish (C ≥ 10 ≥ 20 ≥ 50 ≥ 200)</span>'
      : '<span style="color:#94a3b8;">✗ Neutral / Non-Stage 2</span>';
  }

  if (dist52wEl) {
    if (s.dist_52w_high != null) {
      const isAth15 = s.dist_52w_high >= -15.0;
      dist52wEl.innerHTML = `<span style="color:${isAth15 ? '#34d399' : '#cbd5e1'}; font-weight:${isAth15 ? '700' : '400'};">${s.dist_52w_high > 0 ? '+' : ''}${s.dist_52w_high.toFixed(1)}% ${isAth15 ? '🏆 (Within 15% ATH)' : ''}</span>`;
    } else {
      dist52wEl.textContent = '·';
    }
  }

  if (postureEl) {
    const isQual = (s.rs_score != null && s.rs_score >= 90) && (s.stage2 || (s.dist_52w_high != null && s.dist_52w_high >= -15.0));
    if (isQual) {
      postureEl.innerHTML = '<span class="diag-pill diag-tailwind" style="display:inline-block; font-size:10px; padding:2px 8px;">🌟 Institutional Leader (RS ≥ 90 + S2/ATH)</span>';
    } else if (s.rs_score != null && s.rs_score >= 90) {
      postureEl.innerHTML = '<span class="diag-pill diag-neutral" style="display:inline-block; font-size:10px; padding:2px 8px;">⚠️ High RS but Lacks Stage 2 / Near-ATH</span>';
    } else {
      postureEl.innerHTML = '<span style="color:var(--muted); font-size:11px;">Standard Momentum Setup</span>';
    }
  }

  document.getElementById('dos_pattern').textContent = (s.pattern || 'other').toUpperCase();
  document.getElementById('dos_pole_move').textContent = `+${s.move_pct.toFixed(1)}%`;
  document.getElementById('dos_pole_dates').textContent = `${s.qm_pole_start} → ${s.qm_pole_end}`;
  document.getElementById('dos_cons_bars').textContent = `${s.cons_bars} trading sessions`;
  document.getElementById('dos_pullback').textContent = `-${s.pullback_pct.toFixed(1)}% off high`;
  document.getElementById('dos_ride_frac').textContent = `${(s.ride_frac * 100).toFixed(0)}% riding 20 EMA`;
  document.getElementById('dos_shakeouts').textContent = `${s.shakeouts} wick undercuts bought back`;
  document.getElementById('dos_vol_x').textContent = s.vol_x ? `${s.vol_x.toFixed(2)}x base avg` : 'Normal';
  document.getElementById('dos_level').textContent = `$${s.level.toFixed(2)}`;
  document.getElementById('dos_trigger').textContent = `$${s.trigger.toFixed(2)} (${s.dist_to_trigger_pct > 0 ? '+' : ''}${s.dist_to_trigger_pct.toFixed(2)}%)`;
  document.getElementById('dos_stack').textContent = (s.stack_levels && s.stack_levels.length > 0) ? s.stack_levels.map(x => `$${x}`).join(', ') : `$${s.trigger}`;
  const dosR2 = document.getElementById('dos_r2');
  if (dosR2) dosR2.textContent = (s.r2 != null) ? s.r2.toFixed(3) : '·';
  const dosPinch = document.getElementById('dos_pinch_ratio');
  if (dosPinch) dosPinch.textContent = (s.pinch_ratio != null) ? `${s.pinch_ratio.toFixed(2)}x ADR` : '·';

  // EP Catalyst Linkage Card
  document.getElementById('dos_ep_badge').textContent = s.ep_badge;

  const priorSec = document.getElementById('dos_prior_ep_sec');
  const postSec = document.getElementById('dos_post_ep_sec');
  const pureSec = document.getElementById('dos_pure_tech_sec');

  if (s.has_prior_ep) {
    priorSec.style.display = 'flex';
    document.getElementById('dos_prior_days').textContent = `Ignited ${s.prior_ep_days_ago} calendar days before QM`;
    document.getElementById('dos_prior_date').textContent = s.prior_ep_date;
    document.getElementById('dos_prior_stats').textContent = `+${s.prior_ep_gap_pct}% gap • ${s.prior_ep_rvol}x RVOL`;
    document.getElementById('dos_prior_cat').textContent = s.prior_ep_catalyst || s.prior_ep_subtype || 'EP Catalyst';
  } else {
    priorSec.style.display = 'none';
  }

  if (s.has_post_ep) {
    postSec.style.display = 'flex';
    document.getElementById('dos_post_days').textContent = `Secondary catalyst ${s.post_ep_days_later} days after QM`;
    document.getElementById('dos_post_date').textContent = s.post_ep_date;
    document.getElementById('dos_post_stats').textContent = `+${s.post_ep_gap_pct}% gap • ${s.post_ep_rvol}x RVOL`;
  } else {
    postSec.style.display = 'none';
  }

  // SEC EDGAR & Catalyst Intelligence
  const catSec = document.getElementById('dos_cat_intel_sec');
  if (cat && (cat.headline || (cat.sec_filings && cat.sec_filings.length > 0) || cat.narrative)) {
    if (catSec) {
      catSec.style.display = 'flex';
      document.getElementById('dos_cat_headline').textContent = cat.headline || (cat.catalyst_type ? cat.catalyst_type.toUpperCase() : 'Corporate Catalyst');
      document.getElementById('dos_cat_analysis').textContent = cat.institutional_trajectory || cat.narrative || 'Institutional volume repricing in progress.';
      const linksEl = document.getElementById('dos_cat_links');
      linksEl.innerHTML = '';
      if (cat.sec_filings && cat.sec_filings.length > 0) {
        cat.sec_filings.slice(0, 3).forEach(f => {
          const a = document.createElement('a');
          a.href = f.url || '#';
          a.target = '_blank';
          a.style.color = '#38bdf8';
          a.style.textDecoration = 'underline';
          a.textContent = `📄 ${f.form} (${f.filing_date || ''})`;
          linksEl.appendChild(a);
        });
      }
    }
  } else {
    if (catSec) catSec.style.display = 'none';
  }

  if (!s.has_prior_ep && !s.has_post_ep) {
    pureSec.style.display = 'block';
  } else {
    pureSec.style.display = 'none';
  }
}

function openGroupStocksModal(name, kind) {
  if (!name || name === 'General' || name === 'Unknown') return;
  document.getElementById('gs_title').textContent = `📊 ${name} (${kind.toUpperCase()}) — Stock Leaderboard`;
  document.getElementById('gs_badge').textContent = 'Loading...';
  document.getElementById('gs_body').innerHTML = '<div style="text-align:center; padding:30px; color:var(--muted);">Fetching constituent stocks and performance...</div>';
  document.getElementById('group_stocks_modal').classList.add('active');

  fetch(`/api/group_stocks?group=${encodeURIComponent(name)}&kind=${kind}`)
    .then(r => r.json())
    .then(data => {
      if (!data || !data.stocks) {
        document.getElementById('gs_body').innerHTML = `<div style="padding:20px; color:#f87171;">No stocks found for ${name}.</div>`;
        return;
      }
      document.getElementById('gs_badge').textContent = `${data.count} Stocks`;
      const info = data.info || {};
      const scoreCol = (info.score || 50) >= 80 ? '#10b981' : ((info.score || 50) >= 65 ? '#3b82f6' : '#f59e0b');

      document.getElementById('gs_body').innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; border-bottom:1px solid #232d42; padding-bottom:10px;">
          <div>
            <div style="font-size:14px; font-weight:700; color:#fff;">${name}</div>
            <div style="font-size:11px; color:#94a3b8;">${kind.toUpperCase()} · Score: <b style="color:${scoreCol};">${info.score || '·'}</b> · ${info.primary_archetype || ''}</div>
          </div>
          <div style="font-size:11px; color:var(--muted);">
            Ranked by 1-week momentum. Click "Filter 🔍" to isolate setups.
          </div>
        </div>

        <table style="width:100%; border-collapse:collapse; font-size:11.5px;">
          <thead>
            <tr style="background:#11151f; color:var(--muted); font-size:10.5px; border-bottom:1px solid var(--border);">
              <th style="padding:6px 8px; text-align:left;">#</th>
              <th style="padding:6px 8px; text-align:left;">Symbol</th>
              <th style="padding:6px 8px; text-align:right;">Price</th>
              <th style="padding:6px 8px; text-align:right;">D1 %</th>
              <th style="padding:6px 8px; text-align:right;">1W %</th>
              <th style="padding:6px 8px; text-align:right;">1M %</th>
              <th style="padding:6px 8px; text-align:right;">3M %</th>
              <th style="padding:6px 8px; text-align:right;">YTD %</th>
              <th style="padding:6px 8px; text-align:center;">Action</th>
            </tr>
          </thead>
          <tbody>
            ${data.stocks.map((s, idx) => {
              const c1w = s.w1 != null && s.w1 >= 0 ? '#34d399' : '#f87171';
              const cm1 = s.m1 != null && s.m1 >= 0 ? '#34d399' : '#f87171';
              const cytd = s.ytd != null && s.ytd >= 0 ? '#fbbf24' : '#cbd5e1';
              return `<tr style="border-bottom:1px solid #1a202c; cursor:pointer;" onclick="selectStockFromModal('${s.symbol}')">
                <td style="padding:6px 8px; text-align:left; color:var(--muted);">${idx + 1}</td>
                <td style="padding:6px 8px; text-align:left; font-weight:700; color:#38bdf8;">${s.symbol}</td>
                <td style="padding:6px 8px; text-align:right; color:#cbd5e1;">${s.price != null ? '$' + s.price.toFixed(2) : '·'}</td>
                <td style="padding:6px 8px; text-align:right; color:${s.d1 != null && s.d1 >= 0 ? '#34d399' : '#f87171'};">${s.d1 != null ? (s.d1 >= 0 ? '+' : '') + s.d1.toFixed(1) + '%' : '·'}</td>
                <td style="padding:6px 8px; text-align:right; font-weight:700; color:${c1w};">${s.w1 != null ? (s.w1 >= 0 ? '+' : '') + s.w1.toFixed(1) + '%' : '·'}</td>
                <td style="padding:6px 8px; text-align:right; color:${cm1};">${s.m1 != null ? (s.m1 >= 0 ? '+' : '') + s.m1.toFixed(1) + '%' : '·'}</td>
                <td style="padding:6px 8px; text-align:right; color:${s.m3 != null && s.m3 >= 0 ? '#34d399' : '#f87171'};">${s.m3 != null ? (s.m3 >= 0 ? '+' : '') + s.m3.toFixed(1) + '%' : '·'}</td>
                <td style="padding:6px 8px; text-align:right; font-weight:700; color:${cytd};">${s.ytd != null ? (s.ytd >= 0 ? '+' : '') + s.ytd.toFixed(1) + '%' : '·'}</td>
                <td style="padding:6px 8px; text-align:center;"><button class="btn-filter-chip" style="padding:2px 8px; font-size:10px;" onclick="event.stopPropagation(); selectStockFromModal('${s.symbol}')">Filter 🔍</button></td>
              </tr>`;
            }).join('')}
          </tbody>
        </table>
      `;
    })
    .catch(e => {
      document.getElementById('gs_body').innerHTML = `<div style="padding:20px; color:#f87171;">Error loading stocks: ${e.message}</div>`;
    });
}

function closeGroupStocks(e) {
  if (e && e.target && e.target.id !== 'group_stocks_modal' && !e.target.classList.contains('modal-close')) return;
  document.getElementById('group_stocks_modal').classList.remove('active');
}

function selectStockFromModal(sym) {
  closeGroupStocks();
  const searchInput = document.getElementById('inp_search');
  if (searchInput) searchInput.value = sym;
  fetchEvents();
}

// Enlarged Chart Stock Chooser Modal & Step Navigation
let eslFilterText = '';

function updateEnlargedStockCount() {
  const lbl = document.getElementById('lbl_enlarged_stock_count');
  if (!lbl) return;
  const total = allEvents ? allEvents.length : 0;
  if (!total) {
    lbl.textContent = '0';
    return;
  }
  let idx = -1;
  if (currentEvent) {
    idx = allEvents.findIndex(e => e.symbol === currentEvent.symbol && e.date === currentEvent.date);
  }
  if (idx !== -1) {
    lbl.textContent = `${idx + 1}/${total}`;
  } else {
    lbl.textContent = `${total}`;
  }
}

function openEnlargedStockListModal() {
  const modal = document.getElementById('enlarged_stock_list_modal');
  if (!modal) return;
  modal.classList.add('active');
  const searchInput = document.getElementById('esl_search');
  if (searchInput) searchInput.value = '';
  eslFilterText = '';
  renderEnlargedStockList('');
  setTimeout(() => {
    if (searchInput) searchInput.focus();
    const selRow = document.querySelector('#esl_table_body tr.selected');
    if (selRow) selRow.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }, 60);
}

function closeEnlargedStockListModal(e) {
  if (e && e.target && e.target.id !== 'enlarged_stock_list_modal' && !e.target.classList.contains('modal-close')) return;
  const modal = document.getElementById('enlarged_stock_list_modal');
  if (modal) modal.classList.remove('active');
}

function filterEnlargedStockList(val) {
  eslFilterText = (val || '').trim().toLowerCase();
  renderEnlargedStockList(eslFilterText);
}

function renderEnlargedStockList(query) {
  const tbody = document.getElementById('esl_table_body');
  if (!tbody) return;
  const countEl = document.getElementById('esl_count');
  const totalCount = allEvents ? allEvents.length : 0;
  updateEnlargedStockCount();

  const q = (query != null ? query : eslFilterText || '').toLowerCase().trim();
  let list = allEvents || [];
  if (q) {
    list = list.filter(e => 
      (e.symbol && e.symbol.toLowerCase().includes(q)) ||
      (e.date && e.date.includes(q)) ||
      (e.state && e.state.toLowerCase().includes(q)) ||
      (e.pattern && e.pattern.toLowerCase().includes(q)) ||
      (e.sector && e.sector.toLowerCase().includes(q)) ||
      (e.theme && e.theme.toLowerCase().includes(q))
    );
  }

  if (countEl) {
    countEl.textContent = q ? `${list.length} / ${totalCount}` : totalCount.toLocaleString();
  }

  const sumEl = document.getElementById('esl_filter_summary');
  if (sumEl) {
    sumEl.textContent = `State: ${activeStateFilter.toUpperCase()} · EP: ${activeEpFilter.toUpperCase()} · Macro: ${activeMacroFilter}`;
  }

  if (!list.length) {
    tbody.innerHTML = '<tr><td colspan="11" style="text-align:center; padding:30px; color:var(--muted);">No matching QM setups found.</td></tr>';
    return;
  }

  const rows = list.map(ev => {
    const isSel = (currentEvent && ev.symbol === currentEvent.symbol && ev.date === currentEvent.date);
    const isBk = (ev.state === 'breakout');
    const statePill = isBk ? '<span class="pill-breakout">BREAKOUT</span>' : '<span class="pill-building">BUILDING</span>';

    // RS pill
    let rsPill = '<span style="color:#64748b; font-size:11px;">--</span>';
    if (ev.rs_score != null) {
      const rsVal = Math.round(ev.rs_score);
      const isLeader = (rsVal >= 90);
      const color = isLeader ? '#38bdf8' : (rsVal >= 70 ? '#94a3b8' : '#64748b');
      const fw = isLeader ? 'bold' : 'normal';
      let postBadges = '';
      if (ev.stage2) postBadges += ' <span style="font-size:9px; background:#065f46; color:#a7f3d0; padding:1px 3px; border-radius:2px;" title="Stage 2 Bullish Alignment">S2</span>';
      if (ev.dist_52w_high != null && ev.dist_52w_high >= -15.0) postBadges += ' <span style="font-size:9px; background:#1e3a8a; color:#bfdbfe; padding:1px 3px; border-radius:2px;" title="Within 15% of 52W High">ATH</span>';
      rsPill = `<span style="color:${color}; font-weight:${fw}; font-family:monospace;">${rsVal}</span>${postBadges}`;
    }

    // EP badge pill
    let epBadgeHtml = '<span class="badge-ep-tech">⚪ Pure Technical</span>';
    if (ev.has_prior_ep && ev.has_post_ep) {
      epBadgeHtml = `<span class="badge-ep-both">⚡🚀 Ignited & Boosted</span>`;
    } else if (ev.has_prior_ep) {
      epBadgeHtml = `<span class="badge-ep-ignited">⚡ Prior EP (${ev.prior_ep_days_ago}d ago, +${ev.prior_ep_gap_pct}%)</span>`;
    } else if (ev.has_post_ep) {
      epBadgeHtml = `<span class="badge-ep-boosted">🚀 Post EP (${ev.post_ep_days_later}d later, +${ev.post_ep_gap_pct}%)</span>`;
    }

    // Macro flow badge
    let macroBadge = '';
    if (ev.is_emerging_theme) {
      macroBadge = ' <span class="badge" style="background:#78350f; color:#fde68a; font-size:9.5px; margin-left:4px;" title="Emerging Theme">🌟 Emerging</span>';
    } else if (ev.is_leading_theme) {
      macroBadge = ' <span class="badge" style="background:#4c1d95; color:#ddd6fe; font-size:9.5px; margin-left:4px;" title="Leading Theme">🚀 Leading</span>';
    } else if (ev.is_leading_sector) {
      macroBadge = ' <span class="badge" style="background:#1e3a8a; color:#bfdbfe; font-size:9.5px; margin-left:4px;" title="Leading Sector">🏭 Leading</span>';
    }

    return `<tr class="${isSel ? 'selected' : ''}" onclick="onEslRowClick('${ev.symbol}','${ev.date}')">
      <td class="tl"><strong style="color:#38bdf8;">${ev.symbol}</strong>${macroBadge}</td>
      <td class="tl" style="font-family:monospace; color:#94a3b8;">${ev.date}</td>
      <td class="tl">${statePill}</td>
      <td style="text-align:center;">${rsPill}</td>
      <td style="color:#34d399; font-weight:600;">+${(ev.move_pct || 0).toFixed(1)}%</td>
      <td>${ev.cons_bars}d</td>
      <td style="color:${(ev.pullback_pct || 0) > 25 ? '#f87171' : '#cbd5e1'}">-${(ev.pullback_pct || 0).toFixed(1)}%</td>
      <td class="tl" style="font-size:11px; color:#cbd5e1;">${ev.pattern || '·'}</td>
      <td class="tl">${epBadgeHtml}</td>
      <td class="tl" style="max-width:120px; overflow:hidden; text-overflow:ellipsis; color:#38bdf8;">${ev.sector || '·'}</td>
      <td class="tl" style="max-width:120px; overflow:hidden; text-overflow:ellipsis; color:#c084fc;">${ev.theme || '·'}</td>
    </tr>`;
  }).join('');

  tbody.innerHTML = rows;
}

function onEslRowClick(sym, dt) {
  const ev = allEvents.find(e => e.symbol === sym && e.date === dt);
  if (ev) {
    selectEvent(ev);
    const row = document.getElementById(`row_${sym}_${dt}`);
    if (row) row.scrollIntoView({ block: 'nearest' });
  }
  closeEnlargedStockListModal();
}

function sortEslBy(col) {
  sortTable(col);
  setTimeout(() => {
    renderEnlargedStockList();
  }, 100);
}

function stepFilteredStock(delta) {
  if (!allEvents || !allEvents.length) return;
  let idx = -1;
  if (currentEvent) {
    idx = allEvents.findIndex(e => e.symbol === currentEvent.symbol && e.date === currentEvent.date);
  }
  if (idx === -1) {
    idx = (delta > 0) ? 0 : allEvents.length - 1;
  } else {
    idx = (idx + delta + allEvents.length) % allEvents.length;
  }
  const nextEv = allEvents[idx];
  if (nextEv) {
    selectEvent(nextEv);
    const row = document.getElementById(`row_${nextEv.symbol}_${nextEv.date}`);
    if (row) row.scrollIntoView({ block: 'nearest' });
    updateEnlargedStockCount();

    const modal = document.getElementById('enlarged_stock_list_modal');
    if (modal && modal.classList.contains('active')) {
      renderEnlargedStockList();
      setTimeout(() => {
        const selRow = document.querySelector('#esl_table_body tr.selected');
        if (selRow) selRow.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      }, 40);
    }
  }
}

function clearDossier() {
  document.getElementById('dos_sym_date').textContent = '--';
  document.getElementById('dos_sector').textContent = '--';
  document.getElementById('dos_themes').textContent = '--';
  document.getElementById('dos_pole_move').textContent = '--';
  if (document.getElementById('dos_rs_score')) document.getElementById('dos_rs_score').textContent = '--';
  if (document.getElementById('dos_rs_badge')) {
    document.getElementById('dos_rs_badge').className = 'pctile-badge pctile-low';
    document.getElementById('dos_rs_badge').textContent = 'RS --';
  }
  if (document.getElementById('dos_stage2')) document.getElementById('dos_stage2').textContent = '--';
  if (document.getElementById('dos_dist_52w')) document.getElementById('dos_dist_52w').textContent = '--';
  if (document.getElementById('dos_rs_posture_status')) document.getElementById('dos_rs_posture_status').textContent = '--';
  if (document.getElementById('dos_r2')) document.getElementById('dos_r2').textContent = '--';
  if (document.getElementById('dos_pinch_ratio')) document.getElementById('dos_pinch_ratio').textContent = '--';
  const stratCard = document.getElementById('card_strategy_trade');
  if (stratCard) stratCard.style.display = 'none';
  const mlCard = document.getElementById('card_ml_guidance');
  if (mlCard) mlCard.style.display = 'none';
}

// Filter Actions
function setStateFilter(st) {
  activeStateFilter = st;
  ['all', 'breakout', 'building'].forEach(s => {
    const btn = document.getElementById(`f_state_${s}`);
    if (btn) btn.classList.toggle('active', s === st);
  });
  fetchEvents();
}

function setEpFilter(ep) {
  activeEpFilter = ep;
  ['all', 'ignited', 'boosted', 'both', 'none'].forEach(e => {
    const btn = document.getElementById(`f_ep_${e}`);
    if (btn) btn.classList.toggle('active', e === ep);
  });
  fetchEvents();
}

function setMacroFilter(mf) {
  activeMacroFilter = mf;
  ['all', 'leading_sectors', 'leading_themes', 'emerging_themes'].forEach(m => {
    const btn = document.getElementById(`f_macro_${m}`);
    if (btn) btn.classList.toggle('active', m === mf);
  });
  if (mf === 'leading_sectors') {
    document.getElementById('sel_sector').value = 'leading_sectors';
    document.getElementById('sel_theme').value = 'all';
    strategySettings.macro = 'leading_sectors';
  } else if (mf === 'leading_themes') {
    document.getElementById('sel_theme').value = 'leading_themes';
    document.getElementById('sel_sector').value = 'all';
    strategySettings.macro = 'leading_themes';
  } else if (mf === 'emerging_themes') {
    document.getElementById('sel_theme').value = 'emerging_themes';
    document.getElementById('sel_sector').value = 'all';
    strategySettings.macro = 'emerging_themes';
  } else {
    document.getElementById('sel_sector').value = 'all';
    document.getElementById('sel_theme').value = 'all';
    strategySettings.macro = 'all';
  }
  const cfgMacro = document.getElementById('cfg_strat_macro');
  if (cfgMacro) cfgMacro.value = strategySettings.macro;
  fetchEvents();
}

function onSectorChange() {
  const v = document.getElementById('sel_sector').value;
  if (v === 'leading_sectors') {
    activeMacroFilter = 'leading_sectors';
  } else {
    activeMacroFilter = 'all';
  }
  ['all', 'leading_sectors', 'leading_themes', 'emerging_themes'].forEach(m => {
    const btn = document.getElementById(`f_macro_${m}`);
    if (btn) btn.classList.toggle('active', m === activeMacroFilter);
  });
  fetchEvents();
}

function onThemeChange() {
  const v = document.getElementById('sel_theme').value;
  if (v === 'leading_themes') {
    activeMacroFilter = 'leading_themes';
  } else if (v === 'emerging_themes') {
    activeMacroFilter = 'emerging_themes';
  } else {
    activeMacroFilter = 'all';
  }
  ['all', 'leading_sectors', 'leading_themes', 'emerging_themes'].forEach(m => {
    const btn = document.getElementById(`f_macro_${m}`);
    if (btn) btn.classList.toggle('active', m === activeMacroFilter);
  });
  fetchEvents();
}

function onSearchInput() {
  clearTimeout(searchDebounceTimer);
  searchDebounceTimer = setTimeout(fetchEvents, 300);
}

function resetFilters() {
  activeStateFilter = 'all';
  activeEpFilter = 'all';
  activeMacroFilter = 'all';
  ['all', 'breakout', 'building'].forEach(s => {
    const btn = document.getElementById(`f_state_${s}`);
    if (btn) btn.classList.toggle('active', s === 'all');
  });
  ['all', 'ignited', 'boosted', 'both', 'none'].forEach(e => {
    const btn = document.getElementById(`f_ep_${e}`);
    if (btn) btn.classList.toggle('active', e === 'all');
  });
  ['all', 'leading_sectors', 'leading_themes', 'emerging_themes'].forEach(m => {
    const btn = document.getElementById(`f_macro_${m}`);
    if (btn) btn.classList.toggle('active', m === 'all');
  });
  if (document.getElementById('sel_rs')) document.getElementById('sel_rs').value = 'rs90_stage2_or_15';
  document.getElementById('sel_pattern').value = 'all';
  document.getElementById('sel_move').value = '';
  document.getElementById('sel_cons').value = 'all';
  document.getElementById('sel_pullback').value = '';
  document.getElementById('sel_sector').value = 'all';
  document.getElementById('sel_theme').value = 'all';
  if (document.getElementById('sel_bb_pinch')) document.getElementById('sel_bb_pinch').value = 'all';
  if (document.getElementById('sel_r2')) document.getElementById('sel_r2').value = 'all';
  if (document.getElementById('sel_strat_exit')) document.getElementById('sel_strat_exit').value = '10ema';
  if (document.getElementById('cfg_strat_exit')) document.getElementById('cfg_strat_exit').value = '10ema';
  strategySettings.exit = '10ema';
  document.getElementById('inp_search').value = '';
  fetchEvents();
}

function sortTable(col) {
  if (currentSort.col === col) {
    currentSort.asc = !currentSort.asc;
  } else {
    currentSort.col = col;
    currentSort.asc = (col === 'symbol' || col === 'pattern' || col === 'pinch_ratio');
  }
  fetchEvents();
}

// Indicator Toggles
// Chart Settings & Indicator Toggles
function toggleChartSettings() {
  const p = document.getElementById('chart_settings_panel');
  if (!p) return;
  const isHidden = (p.style.display === 'none' || !p.style.display);
  p.style.display = isHidden ? 'flex' : 'none';
  const b = document.getElementById('btn_chart_settings');
  if (b) b.classList.toggle('active', isHidden);
  const bTool = document.getElementById('btn_tool_settings');
  if (bTool) bTool.classList.toggle('active', isHidden);
}

function toggleEma10() {
  const el = document.getElementById('chk_settings_ema10') || document.getElementById('chk_ema10');
  const on = el ? el.checked : true;
  if (ema10Series) ema10Series.applyOptions({ visible: on });
}
function toggleEma20() {
  const el = document.getElementById('chk_settings_ema20') || document.getElementById('chk_ema20');
  const on = el ? el.checked : true;
  if (ema20Series) ema20Series.applyOptions({ visible: on });
}
function toggleSma50() {
  const el = document.getElementById('chk_settings_sma50') || document.getElementById('chk_sma50');
  const on = el ? el.checked : true;
  if (sma50Series) sma50Series.applyOptions({ visible: on });
}
function toggleSma200() {
  const el = document.getElementById('chk_settings_sma200') || document.getElementById('chk_sma200');
  const on = el ? el.checked : true;
  if (sma200Series) sma200Series.applyOptions({ visible: on });
}
function toggleVolMa20() {
  const el = document.getElementById('chk_settings_vol_ma20');
  const on = el ? el.checked : true;
  if (volMa20Series) volMa20Series.applyOptions({ visible: on });
}
function togglePoleBox() {
  const el = document.getElementById('chk_settings_pole_box') || document.getElementById('chk_pole_box');
  showPoleBox = el ? el.checked : true;
  drawCanvasOverlays();
}
function toggleConsBox() {
  const el = document.getElementById('chk_settings_cons_box') || document.getElementById('chk_cons_box');
  showConsBox = el ? el.checked : true;
  drawCanvasOverlays();
}
function togglePriceLines() {
  const el = document.getElementById('chk_settings_price_lines');
  showPriceLines = el ? el.checked : true;
  if (currentChartData && currentChartData.setup) {
    applySetupPriceLines(currentChartData.setup);
  }
}
function toggleEpMarkers() {
  const el = document.getElementById('chk_settings_ep_markers');
  showEpMarkers = el ? el.checked : true;
  if (currentChartData) {
    applyCandleMarkers(currentChartData);
  }
}

// Larsson Ribbon Renderer
let _larssonRaf = null;
function scheduleLarssonRender() {
  if (_larssonRaf) cancelAnimationFrame(_larssonRaf);
  _larssonRaf = requestAnimationFrame(() => {
    _larssonRaf = null;
    renderLarssonRibbon();
  });
}

function updateNativeLarssonSeries(larssonData) {
  if (!larssonFastSeries || !larssonSlowSeries) return;
  const isEnabled = document.getElementById('chk_settings_larsson') ? document.getElementById('chk_settings_larsson').checked : true;
  if (!isEnabled || !larssonData || !larssonData.length) {
    larssonFastSeries.setData([]);
    larssonSlowSeries.setData([]);
    return;
  }
  const fastData = [];
  const slowData = [];
  for (let i = 0; i < larssonData.length; i++) {
    const item = larssonData[i];
    if (item.fast != null) {
      fastData.push({ time: item.time, value: item.fast });
    }
    if (item.slow != null) {
      slowData.push({ time: item.time, value: item.slow });
    }
  }
  larssonFastSeries.setData(fastData);
  larssonSlowSeries.setData(slowData);
}

function renderLarssonRibbon(larssonData) {
  if (larssonData) {
    window._lastLarssonData = larssonData;
  } else {
    larssonData = window._lastLarssonData;
  }
  const group = document.getElementById('larsson_ribbon_group');
  if (!group) return;
  const isEnabled = document.getElementById('chk_settings_larsson') ? document.getElementById('chk_settings_larsson').checked : true;
  if (!isEnabled || !larssonData || !larssonData.length || !chart || !candleSeries) {
    group.innerHTML = '';
    return;
  }

  const container = document.getElementById('chart_tv');
  const svg = document.getElementById('larsson_ribbon_svg');
  if (svg && container) {
    svg.setAttribute('width', container.clientWidth);
    svg.setAttribute('height', container.clientHeight);
    svg.style.width = container.clientWidth + 'px';
    svg.style.height = container.clientHeight + 'px';
  }

  const logicalRange = chart.timeScale().getVisibleLogicalRange();
  const fromIdx = logicalRange ? Math.max(0, Math.floor(logicalRange.from) - 8) : 0;
  const toIdx = logicalRange ? Math.min(larssonData.length - 1, Math.ceil(logicalRange.to) + 8) : larssonData.length - 1;

  let segments = [];
  let currentSeg = null;

  for (let i = fromIdx; i <= toIdx; i++) {
    const item = larssonData[i];
    if (!item || item.fast == null || item.slow == null) {
      if (currentSeg && currentSeg.pts.length >= 2) segments.push(currentSeg);
      currentSeg = null;
      continue;
    }
    const x = chart.timeScale().timeToCoordinate(item.time);
    const yFast = candleSeries.priceToCoordinate(item.fast);
    const ySlow = candleSeries.priceToCoordinate(item.slow);
    if (x == null || yFast == null || ySlow == null) continue;

    const pt = { x, yFast, ySlow, state: item.state };

    if (!currentSeg) {
      currentSeg = { state: item.state, pts: [pt] };
    } else if (currentSeg.state === item.state) {
      currentSeg.pts.push(pt);
    } else {
      currentSeg.pts.push(pt);
      if (currentSeg.pts.length >= 2) segments.push(currentSeg);
      const prevPt = currentSeg.pts[currentSeg.pts.length - 2];
      currentSeg = { state: item.state, pts: [prevPt, pt] };
    }
  }
  if (currentSeg && currentSeg.pts.length >= 2) {
    segments.push(currentSeg);
  }

  const styleConfig = {
    yellow: { fill: 'rgba(250, 204, 21, 0.35)', strokeFast: '#facc15', strokeSlow: '#ca8a04', strokeWidth: '2.0' },
    blue:   { fill: 'rgba(56, 189, 248, 0.30)', strokeFast: '#38bdf8', strokeSlow: '#0284c7', strokeWidth: '2.0' },
    gray:   { fill: 'rgba(148, 163, 184, 0.20)', strokeFast: '#94a3b8', strokeSlow: '#64748b', strokeWidth: '1.5' }
  };

  let svgHtml = '';
  segments.forEach(seg => {
    const cfg = styleConfig[seg.state] || styleConfig.gray;
    const pts = seg.pts;
    if (pts.length < 2) return;

    let pathD = `M ${pts[0].x.toFixed(1)} ${pts[0].yFast.toFixed(1)}`;
    for (let j = 1; j < pts.length; j++) {
      pathD += ` L ${pts[j].x.toFixed(1)} ${pts[j].yFast.toFixed(1)}`;
    }
    pathD += ` L ${pts[pts.length - 1].x.toFixed(1)} ${pts[pts.length - 1].ySlow.toFixed(1)}`;
    for (let j = pts.length - 2; j >= 0; j--) {
      pathD += ` L ${pts[j].x.toFixed(1)} ${pts[j].ySlow.toFixed(1)}`;
    }
    pathD += ' Z';

    let fastLineD = `M ${pts[0].x.toFixed(1)} ${pts[0].yFast.toFixed(1)}`;
    let slowLineD = `M ${pts[0].x.toFixed(1)} ${pts[0].ySlow.toFixed(1)}`;
    for (let j = 1; j < pts.length; j++) {
      fastLineD += ` L ${pts[j].x.toFixed(1)} ${pts[j].yFast.toFixed(1)}`;
      slowLineD += ` L ${pts[j].x.toFixed(1)} ${pts[j].ySlow.toFixed(1)}`;
    }

    svgHtml += `<path d="${pathD}" fill="${cfg.fill}" />`;
    svgHtml += `<path d="${fastLineD}" fill="none" stroke="${cfg.strokeFast}" stroke-width="${cfg.strokeWidth}" stroke-linejoin="round" stroke-linecap="round" />`;
    svgHtml += `<path d="${slowLineD}" fill="none" stroke="${cfg.strokeSlow}" stroke-width="${cfg.strokeWidth}" stroke-linejoin="round" stroke-linecap="round" />`;
  });

  group.innerHTML = svgHtml;
}

function updateLarssonLegendValues(item) {
  const elFast = document.getElementById('larsson_val_fast');
  const elSlow = document.getElementById('larsson_val_slow');
  if (!elFast || !elSlow) return;
  if (!item && window._lastLarssonData && window._lastLarssonData.length) {
    for (let i = window._lastLarssonData.length - 1; i >= 0; i--) {
      if (window._lastLarssonData[i].fast != null && window._lastLarssonData[i].slow != null) {
        item = window._lastLarssonData[i];
        break;
      }
    }
  }
  if (item && item.fast != null && item.slow != null) {
    elFast.textContent = item.fast.toFixed(2);
    elSlow.textContent = item.slow.toFixed(2);
    if (item.state === 'yellow') {
      elFast.style.color = '#facc15';
      elSlow.style.color = '#facc15';
    } else if (item.state === 'blue') {
      elFast.style.color = '#38bdf8';
      elSlow.style.color = '#38bdf8';
    } else {
      elFast.style.color = '#94a3b8';
      elSlow.style.color = '#94a3b8';
    }
  }
}

function onLarssonFastestChange() {
  const el = document.getElementById('cfg_larsson_fastest');
  let val = parseInt(el.value, 10);
  if (isNaN(val) || val < 2) val = 32;
  if (val > 100) val = 100;
  el.value = val;
  if (currentEvent) {
    loadChart(currentEvent.symbol, currentEvent.date);
  }
}

function toggleLarsson() {
  const el = document.getElementById('chk_settings_larsson');
  const isEnabled = el ? el.checked : true;
  updateNativeLarssonSeries(window._lastLarssonData);
  scheduleLarssonRender();
}

function toggleChartFullscreen() {
  const ws = document.getElementById('main_workspace');
  const btn = document.getElementById('btn_chart_fullscreen');
  if (!ws) return;
  const isFull = ws.classList.toggle('fullscreen-chart-mode');
  if (btn) {
    btn.innerHTML = isFull ? '🗗 Minimize' : '⛶ Enlarge';
  }
  const drawToolbar = document.getElementById('chart_draw_toolbar');
  if (drawToolbar) {
    drawToolbar.style.display = isFull ? 'flex' : 'none';
  }
  updateEnlargedStockCount();
  if (!isFull) {
    const p = document.getElementById('chart_settings_panel');
    if (p) p.style.display = 'none';
    const b = document.getElementById('btn_chart_settings');
    if (b) b.classList.remove('active');
    const bTool = document.getElementById('btn_tool_settings');
    if (bTool) bTool.classList.remove('active');
  }
  const updateLayout = () => {
    const container = document.getElementById('chart_tv');
    if (chart && container) {
      chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
    }
    requestAnimationFrame(() => {
      drawCanvasOverlays();
      renderAllDrawings();
      scheduleLarssonRender();
    });
  };
  setTimeout(updateLayout, 50);
  setTimeout(updateLayout, 200);
}
const toggleFullscreen = toggleChartFullscreen;

function reloadDataset() {
  fetch('/api/reload_dataset')
    .then(r => r.json())
    .then(data => {
      alert(`Dataset reloaded: ${data.rows} setups (${data.breakouts} breakouts, ${data.building} building).`);
      fetchEvents();
    });
}

// Keyboard Navigation
window.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    const eslModal = document.getElementById('enlarged_stock_list_modal');
    if (eslModal && eslModal.classList.contains('active')) {
      closeEnlargedStockListModal();
      return;
    }
    const p = document.getElementById('chart_settings_panel');
    if (p && p.style.display !== 'none') {
      p.style.display = 'none';
      const b = document.getElementById('btn_chart_settings');
      if (b) b.classList.remove('active');
      const bTool = document.getElementById('btn_tool_settings');
      if (bTool) bTool.classList.remove('active');
      return;
    }
    if (currentDrawTool !== 'cursor') {
      setDrawTool('cursor');
      return;
    }
    const ws = document.getElementById('main_workspace');
    if (ws && ws.classList.contains('fullscreen-chart-mode')) {
      toggleChartFullscreen();
      return;
    }
  }

  // Ctrl/Cmd + K opens stock chooser modal
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
    e.preventDefault();
    openEnlargedStockListModal();
    return;
  }

  if (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'SELECT') return;
  if (!allEvents || allEvents.length === 0 || !currentEvent) return;

  const ws = document.getElementById('main_workspace');
  const isEnlarged = ws && ws.classList.contains('fullscreen-chart-mode');

  if (e.key === 'ArrowRight' || (isEnlarged && e.key === 'ArrowDown')) {
    e.preventDefault();
    stepFilteredStock(1);
  } else if (e.key === 'ArrowLeft' || (isEnlarged && e.key === 'ArrowUp')) {
    e.preventDefault();
    stepFilteredStock(-1);
  } else if (!isEnlarged && e.key === 'ArrowDown') {
    e.preventDefault();
    stepFilteredStock(1);
  } else if (!isEnlarged && e.key === 'ArrowUp') {
    e.preventDefault();
    stepFilteredStock(-1);
  }
});

// Boot
window.addEventListener('DOMContentLoaded', () => {
  initChart();
  fetchEvents();
});
</script>

</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="QM Historical Setup & EP Explorer Server")
    parser.add_argument("--port", type=int, default=8785, help="Port to listen on (default: 8785)")
    args = parser.parse_args()

    get_df()
    get_ep_df()

    print("=================================================================")
    print(f"🚀 QM Setup & EP Explorer Server running at: http://127.0.0.1:{args.port}")
    print("=================================================================")
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
