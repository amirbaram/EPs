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
import scanner_core
import thematic_engine

app = Flask(__name__)

DATA_PATH = Path("data/simulations/qm_historical_setups.parquet")
EP_STUDY_PATH = Path("data/simulations/ep_combined_study_scored.parquet")
ANNOTATIONS_DIR = Path("data/annotations")
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

DF: Optional[pd.DataFrame] = None
EP_DF: Optional[pd.DataFrame] = None


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

    lead_secs, lead_thms, emerg_thms = get_thematic_macro_sets()

    # 7. Thematic / Macro Flow Filter (Leading Sectors, Leading Themes, Emerging Themes)
    macro_f = request.args.get("macro", "all")
    if macro_f == "leading_sectors":
        q = q[q["sector"].isin(lead_secs)]
    elif macro_f == "leading_themes":
        q = q[q["theme"].isin(lead_thms) | q["themes"].apply(lambda x: any(t in str(x) for t in lead_thms) if pd.notna(x) else False)]
    elif macro_f == "emerging_themes":
        q = q[q["theme"].isin(emerg_thms) | q["themes"].apply(lambda x: any(t in str(x) for t in emerg_thms) if pd.notna(x) else False)]

    # 8. Sector & Theme Filters
    sec_f = request.args.get("sector", "all")
    if sec_f == "leading_sectors":
        q = q[q["sector"].isin(lead_secs)]
    elif sec_f != "all" and sec_f:
        q = q[q["sector"] == sec_f]

    thm_f = request.args.get("theme", "all")
    if thm_f == "leading_themes":
        q = q[q["theme"].isin(lead_thms) | q["themes"].apply(lambda x: any(t in str(x) for t in lead_thms) if pd.notna(x) else False)]
    elif thm_f == "emerging_themes":
        q = q[q["theme"].isin(emerg_thms) | q["themes"].apply(lambda x: any(t in str(x) for t in emerg_thms) if pd.notna(x) else False)]
    elif thm_f != "all" and thm_f:
        q = q[q["themes"].str.contains(thm_f, case=False, na=False) | (q["theme"] == thm_f)]

    # 9. Search Symbol
    search_q = request.args.get("search", "").strip().upper()
    if search_q:
        q = q[q["symbol"].str.contains(search_q, case=False, na=False)]

    # Relative Strength (RS) Score & Posture Filter
    # Default: RS >= 90 for stage 2 bullish (stage2 == True) or within 15% 52w high (dist_52w_high >= -15.0)
    rs_f = request.args.get("rs_filter", "rs90_stage2_or_15").strip()
    if rs_f in ["rs90_stage2_or_15", "default"]:
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

    lead_sec_cnt = int(q["sector"].isin(lead_secs).sum())
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
        is_lead_sec = sec_val in lead_secs
        is_lead_thm = (thm_val in lead_thms) or any(t in thms_val for t in lead_thms)
        is_emerg_thm = (thm_val in emerg_thms) or any(t in thms_val for t in emerg_thms)

        rec = {
            "symbol": str(row["symbol"]),
            "date": str(row["date"]),
            "state": str(row["state"]),
            "rs_score": int(row["rs_score"]) if pd.notna(row.get("rs_score")) else None,
            "stage2": bool(row.get("stage2", False)) if pd.notna(row.get("stage2")) else False,
            "dist_52w_high": round(float(row["dist_52w_high"]), 1) if pd.notna(row.get("dist_52w_high")) else None,
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
        }
        events_list.append(rec)

    return jsonify({
        "events": events_list,
        "kpis": kpis,
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
        "vol_ma20": vol_ma20_data,
        "ema10": ema10_data,
        "ema20": ema20_data,
        "sma50": sma50_data,
        "sma200": sma200_data,
        "larsson": larsson_data,
        "larsson_spans": list(larsson_spans),
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
          <th onclick="sortTable('rs_score')" style="color:#38bdf8; cursor:pointer;" title="1-99 Institutional Relative Strength Score & Posture">RS ↕</th>
          <th onclick="sortTable('move_pct')">Pole Move</th>
          <th onclick="sortTable('cons_bars')">Cons</th>
          <th onclick="sortTable('pullback_pct')">Pullback</th>
          <th class="tl" onclick="sortTable('pattern')">Pattern</th>
          <th class="tl" onclick="sortTable('ep_badge')">EP Catalyst Link</th>
        </tr>
      </thead>
      <tbody id="events_tbody">
        <tr><td colspan="9" style="text-align:center; padding:30px; color:var(--muted);">Loading QM setups...</td></tr>
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
  p.set('macro', activeMacroFilter);

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

  const rsEl = document.getElementById('sel_rs');
  if (rsEl) p.set('rs_filter', rsEl.value);

  const srch = document.getElementById('inp_search').value.trim();
  if (srch) p.set('search', srch);

  p.set('sort', currentSort.col);
  p.set('asc', currentSort.asc ? '1' : '0');

  fetch('/api/events?' + p.toString())
    .then(r => r.json())
    .then(data => {
      allEvents = data.events || [];
      updateKPIs(data.kpis || {});
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

  if (allEvents.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" style="text-align:center; padding:30px; color:var(--muted);">No matching QM setups found.</td></tr>';
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

    tr.innerHTML = `
      <td class="tl"><strong style="color:#38bdf8;">${ev.symbol}</strong>${macroBadge}</td>
      <td class="tl" style="font-family:monospace; color:#94a3b8;">${ev.date}</td>
      <td class="tl">${statePill}</td>
      <td style="text-align:center;">${rsPill}</td>
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
  } else if (mf === 'leading_themes') {
    document.getElementById('sel_theme').value = 'leading_themes';
    document.getElementById('sel_sector').value = 'all';
  } else if (mf === 'emerging_themes') {
    document.getElementById('sel_theme').value = 'emerging_themes';
    document.getElementById('sel_sector').value = 'all';
  } else {
    document.getElementById('sel_sector').value = 'all';
    document.getElementById('sel_theme').value = 'all';
  }
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
