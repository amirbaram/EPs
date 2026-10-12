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
import thematic_engine

app = Flask(__name__)

DATA_PATH = Path("data/simulations/qm_historical_setups.parquet")
EP_STUDY_PATH = Path("data/simulations/ep_combined_study_scored.parquet")

DF: Optional[pd.DataFrame] = None
EP_DF: Optional[pd.DataFrame] = None


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


@app.route("/api/reload_dataset", methods=["GET", "POST"])
def api_reload_dataset():
    global DF, EP_DF
    DF = None
    EP_DF = None
    df = get_df()
    return jsonify({
        "status": "ok",
        "rows": len(df),
        "breakouts": int((df["state"] == "breakout").sum()) if not df.empty else 0,
        "building": int((df["state"] == "building").sum()) if not df.empty else 0,
    })


@app.route("/api/group_stocks")
def api_group_stocks():
    group = request.args.get("group", "").strip()
    kind = request.args.get("kind", "theme").strip()
    if not group:
        return jsonify({"error": "Missing group"}), 400
    eng = thematic_engine.get_thematic_engine()
    return jsonify(eng.get_group_stocks(group, kind))


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

    # 1. State Filter
    state_f = request.args.get("state", "all")
    if state_f in ["breakout", "building"]:
        q = q[q["state"] == state_f]

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
    if cons_f == "8-15":
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

    # 7. Sector & Theme Filters
    sec_f = request.args.get("sector", "all")
    if sec_f != "all" and sec_f:
        q = q[q["sector"] == sec_f]

    thm_f = request.args.get("theme", "all")
    if thm_f != "all" and thm_f:
        q = q[q["themes"].str.contains(thm_f, case=False, na=False) | (q["theme"] == thm_f)]

    # 8. Search Symbol
    search_q = request.args.get("search", "").strip().upper()
    if search_q:
        q = q[q["symbol"].str.contains(search_q, case=False, na=False)]

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

    kpis = {
        "total": total_count,
        "breakouts": bks,
        "building": bld,
        "prior_pct": round((prior_cnt / total_count * 100), 1) if total_count > 0 else 0.0,
        "post_pct": round((post_cnt / total_count * 100), 1) if total_count > 0 else 0.0,
        "ignited_pct": round((ignited_cnt / total_count * 100), 1) if total_count > 0 else 0.0,
        "avg_move": avg_move,
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

        rec = {
            "symbol": str(row["symbol"]),
            "date": str(row["date"]),
            "state": str(row["state"]),
            "move_pct": round(float(row["move_pct"]), 1),
            "cons_bars": int(row["cons_bars"]),
            "pullback_pct": round(float(row["pullback_pct"]), 1),
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
            "sector": str(row.get("sector", "")),
            "theme": str(row.get("theme", "")),
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
        }
        events_list.append(rec)

    return jsonify({
        "events": events_list,
        "kpis": kpis,
        "sectors": all_sectors,
        "themes": all_themes,
        "patterns": all_patterns,
    })


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
    ema10_data = []
    ema20_data = []
    sma50_data = []
    sma200_data = []

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

        if pd.notna(ema10_s.loc[idx]):
            ema10_data.append({"time": t, "value": round(float(ema10_s.loc[idx]), 2)})
        if pd.notna(ema20_s.loc[idx]):
            ema20_data.append({"time": t, "value": round(float(ema20_s.loc[idx]), 2)})
        if pd.notna(sma50_s.loc[idx]):
            sma50_data.append({"time": t, "value": round(float(sma50_s.loc[idx]), 2)})
        if pd.notna(sma200_s.loc[idx]):
            sma200_data.append({"time": t, "value": round(float(sma200_s.loc[idx]), 2)})

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
        "move_pct": round(float(setup_row.get("move_pct", 0.0)), 1) if setup_row else 0.0,
        "cons_bars": int(setup_row.get("cons_bars", 0)) if setup_row else 0,
        "pullback_pct": round(float(setup_row.get("pullback_pct", 0.0)), 1) if setup_row else 0.0,
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

    return jsonify({
        "symbol": sym,
        "date": date_str,
        "bars": bars,
        "volume": volume,
        "ema10": ema10_data,
        "ema20": ema20_data,
        "sma50": sma50_data,
        "sma200": sma200_data,
        "setup": setup_payload,
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
  .workspace { display: flex; flex: 1; overflow: hidden; position: relative; }
  
  /* Left Pane: Table */
  .table-pane { flex: 1.1; min-width: 440px; overflow: auto; border-right: 1px solid var(--border); }
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

  /* Center Pane: Full-Height Chart */
  .chart-pane { flex: 1.5; min-width: 500px; min-height: 0; display: flex; flex-direction: column; background: #0c0f16; border-right: 1px solid var(--border); position: relative; overflow: hidden; }

  .chart-header {
    background: #11151f; border-bottom: 1px solid var(--border);
    padding: 6px 14px; display: flex; justify-content: space-between; align-items: center; flex-shrink: 0;
  }
  .chart-title-box { display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: 14px; }
  .chart-controls { display: flex; align-items: center; gap: 6px; }

  .chart-container-rel { flex: 1; position: relative; width: 100%; height: 100%; overflow: hidden; }
  #chart_tv { width: 100%; height: 100%; }
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
    flex: 0 0 380px; width: 380px; max-width: 420px; overflow-y: auto;
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
  .workspace.fullscreen-chart-mode .table-pane { display: none !important; }
  .workspace.fullscreen-chart-mode .chart-pane { flex: 1 !important; }

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
</div>

<!-- Filter Toolbar -->
<div class="toolbar">
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
    <label>Pattern:</label>
    <select id="sel_pattern" onchange="fetchEvents()">
      <option value="all">All Patterns</option>
    </select>
  </div>

  <div class="filter-group">
    <label>Min Move:</label>
    <select id="sel_move" onchange="fetchEvents()">
      <option value="">All Moves</option>
      <option value="30">≥ +30%</option>
      <option value="40">≥ +40%</option>
      <option value="50">≥ +50%</option>
      <option value="75">≥ +75%</option>
      <option value="100">≥ +100%</option>
    </select>
  </div>

  <div class="filter-group">
    <label>Cons Bars:</label>
    <select id="sel_cons" onchange="fetchEvents()">
      <option value="all">All Lengths</option>
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
    <label>Sector:</label>
    <select id="sel_sector" onchange="fetchEvents()">
      <option value="all">All Sectors</option>
    </select>
  </div>

  <div class="filter-group">
    <label>Theme:</label>
    <select id="sel_theme" onchange="fetchEvents()">
      <option value="all">All Themes</option>
    </select>
  </div>

  <div class="filter-group">
    <input type="text" id="inp_search" placeholder="Search symbol..." style="width:120px;" oninput="onSearchInput()">
  </div>

  <button class="btn-filter-chip" onclick="resetFilters()" style="margin-left:auto;">Reset Filters</button>
</div>

<!-- 3-Pane Workspace -->
<div class="workspace" id="main_workspace">

  <!-- Left: Main Table -->
  <div class="table-pane" id="table_pane">
    <table>
      <thead>
        <tr>
          <th class="tl" onclick="sortTable('symbol')">Symbol</th>
          <th class="tl" onclick="sortTable('date')">Date</th>
          <th class="tl" onclick="sortTable('state')">State</th>
          <th onclick="sortTable('move_pct')">Pole Move</th>
          <th onclick="sortTable('cons_bars')">Cons</th>
          <th onclick="sortTable('pullback_pct')">Pullback</th>
          <th class="tl" onclick="sortTable('pattern')">Pattern</th>
          <th class="tl" onclick="sortTable('ep_badge')">EP Catalyst Link</th>
        </tr>
      </thead>
      <tbody id="events_tbody">
        <tr><td colspan="8" style="text-align:center; padding:30px; color:var(--muted);">Loading QM setups...</td></tr>
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
        <label class="btn-filter-chip active" style="font-size:10px;"><input type="checkbox" id="chk_ema10" checked onchange="toggleEma10()" style="display:none;"> EMA 10</label>
        <label class="btn-filter-chip active" style="font-size:10px;"><input type="checkbox" id="chk_ema20" checked onchange="toggleEma20()" style="display:none;"> EMA 20</label>
        <label class="btn-filter-chip active" style="font-size:10px;"><input type="checkbox" id="chk_sma50" checked onchange="toggleSma50()" style="display:none;"> SMA 50</label>
        <label class="btn-filter-chip active" style="font-size:10px;"><input type="checkbox" id="chk_sma200" checked onchange="toggleSma200()" style="display:none;"> SMA 200</label>
        <label class="btn-filter-chip active" style="font-size:10px;"><input type="checkbox" id="chk_pole_box" checked onchange="togglePoleBox()" style="display:none;"> Pole Box</label>
        <label class="btn-filter-chip active" style="font-size:10px;"><input type="checkbox" id="chk_cons_box" checked onchange="toggleConsBox()" style="display:none;"> Base Box</label>
        <button class="btn-filter-chip" onclick="resetChartScale()" title="Auto-scale Y-axis and reset view">⟲ Reset Scale</button>
        <button class="btn-filter-chip" onclick="toggleFullscreen()" title="Enlarge Chart View">⛶ Fullscreen</button>
      </div>
    </div>

    <div class="chart-container-rel">
      <div id="chart_hud" class="chart-hud" style="display:none;"></div>
      <div id="chart_tv"></div>
      <canvas id="chart_canvas_overlay"></canvas>
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

<script>
// State
let allEvents = [];
let currentEvent = null;
let currentChartData = null;
let currentSort = { col: 'date', asc: false };
let activeStateFilter = 'all';
let activeEpFilter = 'all';
let searchDebounceTimer = null;

// Chart state
let chart = null;
let candleSeries = null;
let volumeSeries = null;
let ema10Series = null;
let ema20Series = null;
let sma50Series = null;
let sma200Series = null;
let activePriceLines = [];

let showPoleBox = true;
let showConsBox = true;

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
    },
  });

  volumeSeries = chart.addHistogramSeries({
    priceFormat: { type: 'volume' },
    priceScaleId: 'volume',
    priceLineVisible: false,
    lastValueVisible: false,
  });

  chart.priceScale('volume').applyOptions({
    scaleMargins: { top: 0.85, bottom: 0.0 },
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
  });

  // Time scale scroll/zoom listener to redraw canvas shaded boxes
  chart.timeScale().subscribeVisibleLogicalRangeChange(drawCanvasOverlays);
  chart.timeScale().subscribeVisibleTimeRangeChange(drawCanvasOverlays);

  window.addEventListener('resize', () => {
    if (chart && container) {
      chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
      drawCanvasOverlays();
    }
  });
}

function drawCanvasOverlays() {
  const canvas = document.getElementById('chart_canvas_overlay');
  if (!canvas || !chart || !candleSeries || !currentChartData) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.parentElement.clientWidth;
  const h = canvas.height = canvas.parentElement.clientHeight;
  ctx.clearRect(0, 0, w, h);

  const setup = currentChartData.setup;
  if (!setup) return;
  const timeScale = chart.timeScale();

  // 1. Shaded Pole Box (Emerald Green)
  if (showPoleBox && setup.qm_pole_start && setup.qm_pole_end) {
    const x1 = timeScale.timeToCoordinate(setup.qm_pole_start);
    const x2 = timeScale.timeToCoordinate(setup.qm_pole_end);
    if (x1 !== null && x2 !== null) {
      const left = Math.min(x1, x2);
      const width = Math.max(8, Math.abs(x2 - x1));
      ctx.fillStyle = 'rgba(16, 185, 129, 0.12)';
      ctx.fillRect(left, 0, width, h);
      ctx.strokeStyle = 'rgba(16, 185, 129, 0.45)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(left, 0, width, h);
      ctx.setLineDash([]);
      
      // Label
      ctx.fillStyle = '#34d399';
      ctx.font = 'bold 11px monospace';
      ctx.fillText(`POLE +${setup.move_pct}%`, left + 6, 26);
    }
  }

  // 2. Shaded Base Box (Amber/Yellow)
  if (showConsBox && setup.qm_cons_start && setup.qm_cons_end) {
    const x1 = timeScale.timeToCoordinate(setup.qm_cons_start);
    const x2 = timeScale.timeToCoordinate(setup.qm_cons_end);
    if (x1 !== null && x2 !== null) {
      const left = Math.min(x1, x2);
      const width = Math.max(12, Math.abs(x2 - x1));
      ctx.fillStyle = 'rgba(245, 158, 11, 0.12)';
      ctx.fillRect(left, 0, width, h);
      ctx.strokeStyle = 'rgba(245, 158, 11, 0.45)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(left, 0, width, h);
      ctx.setLineDash([]);
      
      // Label
      ctx.fillStyle = '#fbbf24';
      ctx.font = 'bold 11px monospace';
      ctx.fillText(`BASE (${setup.cons_bars}d, -${setup.pullback_pct}%)`, left + 6, 42);
    }
  }
}

// Fetch Events List with active filters
function fetchEvents() {
  const p = new URLSearchParams();
  p.set('state', activeStateFilter);
  p.set('ep_link', activeEpFilter);

  const pat = document.getElementById('sel_pattern').value;
  if (pat !== 'all') p.set('pattern', pat);

  const mv = document.getElementById('sel_move').value;
  if (mv) p.set('min_move', mv);

  const cons = document.getElementById('sel_cons').value;
  if (cons !== 'all') p.set('cons_bars', cons);

  const pb = document.getElementById('sel_pullback').value;
  if (pb) p.set('max_pullback', pb);

  const sec = document.getElementById('sel_sector').value;
  if (sec !== 'all') p.set('sector', sec);

  const thm = document.getElementById('sel_theme').value;
  if (thm !== 'all') p.set('theme', thm);

  const srch = document.getElementById('inp_search').value.trim();
  if (srch) p.set('search', srch);

  p.set('sort', currentSort.col);
  p.set('asc', currentSort.asc ? '1' : '0');

  fetch('/api/events?' + p.toString())
    .then(r => r.json())
    .then(data => {
      allEvents = data.events || [];
      updateKPIs(data.kpis || {});
      populateDropdowns(data.sectors || [], data.themes || [], data.patterns || []);
      renderTable();
      if (allEvents.length > 0) {
        selectEvent(allEvents[0]);
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
}

function populateDropdowns(sectors, themes, patterns) {
  const selSec = document.getElementById('sel_sector');
  if (selSec.options.length <= 1) {
    sectors.forEach(s => {
      const opt = document.createElement('option');
      opt.value = s; opt.textContent = s;
      selSec.appendChild(opt);
    });
  }

  const selThm = document.getElementById('sel_theme');
  if (selThm.options.length <= 1) {
    themes.forEach(t => {
      const opt = document.createElement('option');
      opt.value = t; opt.textContent = t;
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

  if (allEvents.length === 0) {
    tbody.innerHTML = '<tr><td colspan="8" style="text-align:center; padding:30px; color:var(--muted);">No matching QM setups found.</td></tr>';
    return;
  }

  allEvents.forEach((ev, idx) => {
    const tr = document.createElement('tr');
    tr.id = `row_${ev.symbol}_${ev.date}`;
    tr.onclick = () => selectEvent(ev);

    const isBk = (ev.state === 'breakout');
    const statePill = isBk ? '<span class="pill-breakout">BREAKOUT</span>' : '<span class="pill-building">BUILDING</span>';
    
    // EP badge pill
    let epBadgeHtml = '<span class="badge-ep-tech">⚪ Pure Technical</span>';
    if (ev.has_prior_ep && ev.has_post_ep) {
      epBadgeHtml = `<span class="badge-ep-both">⚡🚀 Ignited (${ev.prior_ep_days_ago}d) & Boosted</span>`;
    } else if (ev.has_prior_ep) {
      epBadgeHtml = `<span class="badge-ep-ignited">⚡ Prior EP (${ev.prior_ep_days_ago}d ago, +${ev.prior_ep_gap_pct}%)</span>`;
    } else if (ev.has_post_ep) {
      epBadgeHtml = `<span class="badge-ep-boosted">🚀 Post EP (${ev.post_ep_days_later}d later, +${ev.post_ep_gap_pct}%)</span>`;
    }

    tr.innerHTML = `
      <td class="tl"><strong style="color:#38bdf8;">${ev.symbol}</strong></td>
      <td class="tl" style="font-family:monospace; color:#94a3b8;">${ev.date}</td>
      <td class="tl">${statePill}</td>
      <td style="color:#34d399; font-weight:600;">+${ev.move_pct.toFixed(1)}%</td>
      <td>${ev.cons_bars}d</td>
      <td style="color:${ev.pullback_pct > 25 ? '#f87171' : '#cbd5e1'}">-${ev.pullback_pct.toFixed(1)}%</td>
      <td class="tl" style="font-size:11px; color:#cbd5e1;">${ev.pattern}</td>
      <td class="tl">${epBadgeHtml}</td>
    `;
    tbody.appendChild(tr);
  });
}

function selectEvent(ev) {
  currentEvent = ev;
  document.querySelectorAll('#events_tbody tr').forEach(r => r.classList.remove('selected'));
  const activeRow = document.getElementById(`row_${ev.symbol}_${ev.date}`);
  if (activeRow) activeRow.classList.add('selected');

  loadChart(ev.symbol, ev.date);
}

function loadChart(sym, dateStr) {
  fetch(`/api/chart?symbol=${encodeURIComponent(sym)}&date=${encodeURIComponent(dateStr)}`)
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
  ema10Series.setData(data.ema10 || []);
  ema20Series.setData(data.ema20 || []);
  sma50Series.setData(data.sma50 || []);
  sma200Series.setData(data.sma200 || []);

  const setup = data.setup;

  // Header Title
  document.getElementById('chart_sym').textContent = setup.symbol;
  document.getElementById('chart_date').textContent = setup.date;
  const isBk = (setup.state === 'breakout');
  document.getElementById('chart_state_pill').innerHTML = isBk ? '<span class="pill-breakout">BREAKOUT</span>' : '<span class="pill-building">BUILDING</span>';
  document.getElementById('chart_ep_badge').innerHTML = setup.ep_badge ? `<span class="badge-ep-ignited">${setup.ep_badge}</span>` : '';

  // Resistance Stack Price Lines
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

  // Candlestick Markers: Prior EP, Post EP, and QM Event
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

  // Setup Date Marker
  markers.push({
    time: setup.date,
    position: isBk ? 'aboveBar' : 'belowBar',
    color: isBk ? '#10b981' : '#f59e0b',
    shape: 'circle',
    text: isBk ? '🚀 QM BREAKOUT' : '⏳ QM COIL'
  });

  markers.sort((a, b) => a.time.localeCompare(b.time));
  candleSeries.setMarkers(markers);

  // Set visible range around consolidation & pole and auto-scale price axis to new ticker
  resetChartScale();
}

function resetChartScale() {
  if (!chart || !currentChartData) return;
  const setup = currentChartData.setup;
  const data = currentChartData;

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

function clearDossier() {
  document.getElementById('dos_sym_date').textContent = '--';
  document.getElementById('dos_sector').textContent = '--';
  document.getElementById('dos_themes').textContent = '--';
  document.getElementById('dos_pole_move').textContent = '--';
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

function onSearchInput() {
  clearTimeout(searchDebounceTimer);
  searchDebounceTimer = setTimeout(fetchEvents, 300);
}

function resetFilters() {
  activeStateFilter = 'all';
  activeEpFilter = 'all';
  ['all', 'breakout', 'building'].forEach(s => {
    const btn = document.getElementById(`f_state_${s}`);
    if (btn) btn.classList.toggle('active', s === 'all');
  });
  ['all', 'ignited', 'boosted', 'both', 'none'].forEach(e => {
    const btn = document.getElementById(`f_ep_${e}`);
    if (btn) btn.classList.toggle('active', e === 'all');
  });
  document.getElementById('sel_pattern').value = 'all';
  document.getElementById('sel_move').value = '';
  document.getElementById('sel_cons').value = 'all';
  document.getElementById('sel_pullback').value = '';
  document.getElementById('sel_sector').value = 'all';
  document.getElementById('sel_theme').value = 'all';
  document.getElementById('inp_search').value = '';
  fetchEvents();
}

function sortTable(col) {
  if (currentSort.col === col) {
    currentSort.asc = !currentSort.asc;
  } else {
    currentSort.col = col;
    currentSort.asc = (col === 'symbol' || col === 'pattern');
  }
  fetchEvents();
}

// Indicator Toggles
function toggleEma10() {
  const on = document.getElementById('chk_ema10').checked;
  ema10Series.applyOptions({ visible: on });
}
function toggleEma20() {
  const on = document.getElementById('chk_ema20').checked;
  ema20Series.applyOptions({ visible: on });
}
function toggleSma50() {
  const on = document.getElementById('chk_sma50').checked;
  sma50Series.applyOptions({ visible: on });
}
function toggleSma200() {
  const on = document.getElementById('chk_sma200').checked;
  sma200Series.applyOptions({ visible: on });
}
function togglePoleBox() {
  showPoleBox = document.getElementById('chk_pole_box').checked;
  drawCanvasOverlays();
}
function toggleConsBox() {
  showConsBox = document.getElementById('chk_cons_box').checked;
  drawCanvasOverlays();
}

function toggleFullscreen() {
  const ws = document.getElementById('main_workspace');
  ws.classList.toggle('fullscreen-chart-mode');
  setTimeout(() => {
    const container = document.getElementById('chart_tv');
    if (chart && container) {
      chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
      drawCanvasOverlays();
    }
  }, 100);
}

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
  if (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'SELECT') return;
  if (!allEvents || allEvents.length === 0 || !currentEvent) return;

  const curIdx = allEvents.findIndex(x => x.symbol === currentEvent.symbol && x.date === currentEvent.date);
  if (curIdx === -1) return;

  if (e.key === 'ArrowDown' && curIdx < allEvents.length - 1) {
    e.preventDefault();
    selectEvent(allEvents[curIdx + 1]);
    const row = document.getElementById(`row_${allEvents[curIdx + 1].symbol}_${allEvents[curIdx + 1].date}`);
    if (row) row.scrollIntoView({ block: 'nearest' });
  } else if (e.key === 'ArrowUp' && curIdx > 0) {
    e.preventDefault();
    selectEvent(allEvents[curIdx - 1]);
    const row = document.getElementById(`row_${allEvents[curIdx - 1].symbol}_${allEvents[curIdx - 1].date}`);
    if (row) row.scrollIntoView({ block: 'nearest' });
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
