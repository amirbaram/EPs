#!/usr/bin/env python3
"""serve_rs.py — Relative Strength (RS) Universe Explorer & Technical Workspace.

Institutional 3-pane web workspace (running on port 8786) dedicated to calculating
and exploring Relative Strength (RS) scores across the full universe of stocks:
- Left Pane: Sortable & Filterable Stock Table (Rank, Symbol, Name, RS Score, Returns, ADR, $Vol, Sector, Theme)
  Sorted by RS Score (highest on top) by default.
- Center Pane: Interactive LightweightCharts Candlestick Engine with EMA10/20, SMA50/200,
  Larsson Ribbon, Volume + 20-MA, Relative Strength (RS) Line vs SPY, drawing tools, and fullscreen mode.
- Right Pane: Comprehensive Multi-Asset Dossier (Stock RS & Technical Posture, Sector Performance & Institutional Flow,
  Theme Performance & Thematic Archetype, and Peer Group Leaders).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
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

DATA_PATH = Path("data/simulations/rs_universe_scores.parquet")
EP_STUDY_PATH = Path("data/simulations/ep_combined_study_scored.parquet")
ANNOTATIONS_DIR = Path("data/annotations")
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

DF: Optional[pd.DataFrame] = None
EP_DF: Optional[pd.DataFrame] = None
NAMES_MAP: Dict[str, str] = {}
ETF_MAP: Dict[str, bool] = {}


def load_universe_metadata():
    global NAMES_MAP, ETF_MAP
    if not NAMES_MAP and config.UNIVERSE_FILE.exists():
        try:
            udf = pd.read_csv(config.UNIVERSE_FILE)
            NAMES_MAP = dict(zip(udf["symbol"], udf["name"]))
            ETF_MAP = dict(zip(udf["symbol"], udf.get("etf", False)))
        except Exception as e:
            print(f"Warning: could not load universe metadata: {e}")


def get_ep_df() -> pd.DataFrame:
    global EP_DF
    if EP_DF is None:
        if EP_STUDY_PATH.exists():
            try:
                EP_DF = pd.read_parquet(EP_STUDY_PATH)
            except Exception as e:
                print(f"Warning: could not load EP study dataset: {e}")
                EP_DF = pd.DataFrame()
        else:
            EP_DF = pd.DataFrame()
    return EP_DF


def compute_universe_rs(force: bool = False) -> pd.DataFrame:
    """Computes standard IBD/O'Neil style Relative Strength (RS) scores across the universe.
    Weighted Performance = 40% 3M + 20% 6M + 20% 9M + 20% 1Y.
    Percentile Rank = 1 to 99 across active universe.
    Caches result to data/simulations/rs_universe_scores.parquet.
    """
    global DF
    if not force and DATA_PATH.exists():
        try:
            print(f"Loading cached RS universe scores from {DATA_PATH}...")
            DF = pd.read_parquet(DATA_PATH)
            print(f"Loaded {len(DF)} stocks with RS scores.")
            return DF
        except Exception as e:
            print(f"Failed to read cache {DATA_PATH}: {e}. Recomputing...")

    print("Computing Relative Strength (RS) scores across universe...")
    t0 = time.time()
    labels.load()
    load_universe_metadata()

    spy_bars = datastore.load_bars("SPY")
    if spy_bars is None or spy_bars.empty:
        spy_last_date = pd.Timestamp.now().strftime("%Y-%m-%d")
        spy_ret_1m = spy_ret_3m = spy_ret_1y = 0.0
    else:
        spy_last_date = spy_bars.index[-1].strftime("%Y-%m-%d")
        spy_c = spy_bars["close"].to_numpy(dtype=np.float64)
        spy_ret_1m = (spy_c[-1] / spy_c[-21] - 1.0) * 100 if len(spy_c) >= 21 else 0.0
        spy_ret_3m = (spy_c[-1] / spy_c[-63] - 1.0) * 100 if len(spy_c) >= 63 else 0.0
        spy_ret_1y = (spy_c[-1] / spy_c[-252] - 1.0) * 100 if len(spy_c) >= 252 else 0.0

    symbols = datastore.list_symbols()

    def _worker(sym: str) -> Optional[Dict[str, Any]]:
        try:
            df = datastore.load_bars(sym)
            if df is None or len(df) < 20:
                return None
            c = df["close"].to_numpy(dtype=np.float64)
            v = df["volume"].to_numpy(dtype=np.float64)
            h = df["high"].to_numpy(dtype=np.float64)
            l = df["low"].to_numpy(dtype=np.float64)
            n = len(c)
            last_date = df.index[-1].strftime("%Y-%m-%d")
            last_close = c[-1]

            r_1d = (c[-1] / c[-2] - 1.0) * 100 if n >= 2 else 0.0
            r_1w = (c[-1] / c[-5] - 1.0) * 100 if n >= 5 else 0.0
            r_1m = (c[-1] / c[-21] - 1.0) * 100 if n >= 21 else 0.0
            r_3m = (c[-1] / c[-63] - 1.0) * 100 if n >= 63 else 0.0
            r_6m = (c[-1] / c[-126] - 1.0) * 100 if n >= 126 else r_3m
            r_9m = (c[-1] / c[-189] - 1.0) * 100 if n >= 189 else r_6m
            r_1y = (c[-1] / c[-252] - 1.0) * 100 if n >= 252 else r_9m

            # Weighted RS: 40% 3M, 20% 6M, 20% 9M, 20% 1Y
            raw_rs = 0.40 * r_3m + 0.20 * r_6m + 0.20 * r_9m + 0.20 * r_1y

            alpha_1m = r_1m - spy_ret_1m
            alpha_3m = r_3m - spy_ret_3m
            alpha_1y = r_1y - spy_ret_1y

            hi_window = min(252, n)
            hi_52w = float(np.max(h[-hi_window:]))
            lo_52w = float(np.min(l[-hi_window:]))
            dist_52w_high = (last_close / hi_52w - 1.0) * 100 if hi_52w > 0 else 0.0

            daily_rng = (h[-min(20, n):] - l[-min(20, n):]) / c[-min(20, n):] * 100
            adr_pct = float(np.mean(daily_rng))

            vol_50 = float(np.mean(v[-min(50, n):]))
            dvol_20 = float(np.mean(c[-min(20, n):] * v[-min(20, n):])) / 1e6

            # Moving averages for posture
            ema10 = float(pd.Series(c).ewm(span=10, adjust=False).mean().iloc[-1])
            ema20 = float(pd.Series(c).ewm(span=20, adjust=False).mean().iloc[-1])
            sma50 = float(pd.Series(c).rolling(50).mean().iloc[-1]) if n >= 50 else ema20
            sma200 = float(pd.Series(c).rolling(200).mean().iloc[-1]) if n >= 200 else sma50

            above_10 = bool(last_close >= ema10)
            above_20 = bool(last_close >= ema20)
            above_50 = bool(last_close >= sma50)
            above_200 = bool(last_close >= sma200)
            stage2 = bool(last_close >= ema10 >= ema20 >= sma50 >= sma200)

            sec = labels.sector(sym) or "Unknown"
            thms = labels.themes(sym) or []
            thm = thms[0] if thms else "General"
            is_etf = bool(ETF_MAP.get(sym, False))

            return {
                "symbol": sym,
                "name": NAMES_MAP.get(sym, sym),
                "is_etf": is_etf,
                "last_date": last_date,
                "close": round(float(last_close), 2),
                "ret_1d": round(float(r_1d), 2),
                "ret_1w": round(float(r_1w), 2),
                "ret_1m": round(float(r_1m), 2),
                "ret_3m": round(float(r_3m), 2),
                "ret_6m": round(float(r_6m), 2),
                "ret_1y": round(float(r_1y), 2),
                "alpha_1m": round(float(alpha_1m), 2),
                "alpha_3m": round(float(alpha_3m), 2),
                "alpha_1y": round(float(alpha_1y), 2),
                "raw_rs": float(raw_rs),
                "adr_pct": round(adr_pct, 2),
                "avg_vol_50": int(vol_50),
                "dvol_m": round(dvol_20, 2),
                "hi_52w": round(hi_52w, 2),
                "lo_52w": round(lo_52w, 2),
                "dist_52w_high": round(dist_52w_high, 2),
                "above_10": above_10,
                "above_20": above_20,
                "above_50": above_50,
                "above_200": above_200,
                "stage2": stage2,
                "sector": sec,
                "theme": thm,
                "themes": ", ".join(thms) if thms else thm,
                "bars_count": n,
            }
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(_worker, symbols))

    valid = [r for r in results if r is not None]
    if not valid:
        print("Warning: no valid symbols found.")
        DF = pd.DataFrame()
        return DF

    df_calc = pd.DataFrame(valid)

    # Filter active tickers within 15 calendar days of SPY's last date
    active_cutoff = pd.to_datetime(spy_last_date) - pd.Timedelta(days=15)
    df_calc["last_dt"] = pd.to_datetime(df_calc["last_date"])
    df_active = df_calc[df_calc["last_dt"] >= active_cutoff].copy()

    # Percentile ranking 1 to 99
    common_mask = ~df_active["is_etf"]
    df_active["rs_score"] = 50
    if common_mask.sum() > 0:
        df_active.loc[common_mask, "rs_score"] = (
            df_active.loc[common_mask, "raw_rs"].rank(pct=True) * 98 + 1
        ).round().astype(int)

    etf_mask = df_active["is_etf"]
    if etf_mask.sum() > 0:
        df_active.loc[etf_mask, "rs_score"] = (
            df_active.loc[etf_mask, "raw_rs"].rank(pct=True) * 98 + 1
        ).round().astype(int)

    # Sort highest RS score on top
    df_active = df_active.sort_values("rs_score", ascending=False).reset_index(drop=True)
    df_active["rank"] = df_active.index + 1

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_to_save = df_active.drop(columns=["last_dt"])
    df_to_save.to_parquet(DATA_PATH)
    DF = df_to_save
    print(f"Saved {len(DF)} stocks with RS scores to {DATA_PATH} in {time.time()-t0:.2f}s.")
    return DF


def get_df() -> pd.DataFrame:
    global DF
    if DF is None:
        DF = compute_universe_rs(force=False)
    return DF


@app.route("/api/recompute_rs", methods=["GET", "POST"])
def api_recompute_rs():
    global DF
    DF = compute_universe_rs(force=True)
    return jsonify({
        "status": "ok",
        "stocks_count": len(DF),
        "max_date": str(DF["last_date"].max()) if not DF.empty else "",
        "top_stocks": DF[["rank", "symbol", "rs_score", "close"]].head(5).to_dict(orient="records"),
    })


@app.route("/api/stocks")
def api_stocks():
    df = get_df()
    if df.empty:
        return jsonify({
            "stocks": [],
            "total_count": 0,
            "kpis": {
                "total": 0, "rs90_count": 0, "rs80_count": 0,
                "stage2_count": 0, "near_high_count": 0,
                "spy_1m": 0.0, "spy_3m": 0.0, "spy_1y": 0.0
            },
            "sectors": [], "themes": []
        })

    # SPY benchmark stats
    spy_bars = datastore.load_bars("SPY")
    if spy_bars is not None and not spy_bars.empty:
        sc = spy_bars["close"].to_numpy(dtype=np.float64)
        spy_1m = round(float((sc[-1] / sc[-21] - 1.0) * 100), 2) if len(sc) >= 21 else 0.0
        spy_3m = round(float((sc[-1] / sc[-63] - 1.0) * 100), 2) if len(sc) >= 63 else 0.0
        spy_1y = round(float((sc[-1] / sc[-252] - 1.0) * 100), 2) if len(sc) >= 252 else 0.0
    else:
        spy_1m = spy_3m = spy_1y = 0.0

    q = df.copy()

    # Universe type filter
    u_type = request.args.get("universe_type", "common").strip().lower()
    if u_type == "common":
        q = q[~q["is_etf"]]
    elif u_type == "etf":
        q = q[q["is_etf"]]

    # Min RS score filter
    min_rs = request.args.get("min_rs", "").strip()
    if min_rs.isdigit():
        q = q[q["rs_score"] >= int(min_rs)]

    # Min Dollar Volume filter
    min_dvol = request.args.get("min_dvol", "").strip()
    if min_dvol:
        try:
            q = q[q["dvol_m"] >= float(min_dvol)]
        except ValueError:
            pass

    # Sector filter
    sector_filt = request.args.get("sector", "").strip()
    if sector_filt and sector_filt != "all":
        q = q[q["sector"] == sector_filt]

    # Theme filter
    theme_filt = request.args.get("theme", "").strip()
    if theme_filt and theme_filt != "all":
        q = q[(q["theme"] == theme_filt) | q["themes"].str.contains(theme_filt, na=False)]

    # Trend Posture filter
    trend = request.args.get("trend", "all").strip().lower()
    if trend == "stage2":
        q = q[q["stage2"]]
    elif trend == "above_50":
        q = q[q["above_50"]]
    elif trend == "above_200":
        q = q[q["above_200"]]
    elif trend == "near_high":
        q = q[q["dist_52w_high"] >= -15.0]

    # Search query
    search = request.args.get("search", "").strip().upper()
    if search:
        q = q[q["symbol"].str.contains(search, na=False) | q["name"].str.upper().str.contains(search, na=False)]

    # Sorting
    sort_by = request.args.get("sort_by", "rs_score").strip()
    sort_dir = request.args.get("sort_dir", "desc").strip().lower()
    ascending = (sort_dir == "asc")
    if sort_by in q.columns:
        q = q.sort_values(sort_by, ascending=ascending)
    else:
        q = q.sort_values("rs_score", ascending=False)

    total_count = len(q)
    rs90_count = int((q["rs_score"] >= 90).sum())
    rs80_count = int((q["rs_score"] >= 80).sum())
    stage2_count = int(q["stage2"].sum())
    near_high_count = int((q["dist_52w_high"] >= -15.0).sum())

    kpis = {
        "total": total_count,
        "rs90_count": rs90_count,
        "rs80_count": rs80_count,
        "stage2_count": stage2_count,
        "near_high_count": near_high_count,
        "spy_1m": spy_1m,
        "spy_3m": spy_3m,
        "spy_1y": spy_1y,
    }

    all_sectors = sorted([s for s in df["sector"].dropna().unique() if s and s != "Unknown"])
    all_themes = sorted([t for t in df["theme"].dropna().unique() if t and t != "General"])

    # Cap to top 1500 for snappy UI rendering
    top_stocks = q.head(1500).to_dict(orient="records")

    return jsonify({
        "stocks": top_stocks,
        "total_count": total_count,
        "kpis": kpis,
        "sectors": all_sectors,
        "themes": all_themes,
    })


@app.route("/api/chart")
def api_chart():
    sym = request.args.get("symbol", "").strip().upper()
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

    # Larsson Ribbon calculation
    try:
        larsson_fastest = int(request.args.get("larsson_fastest", "32"))
    except (ValueError, TypeError):
        larsson_fastest = 32
    larsson_fastest = max(2, min(100, larsson_fastest))

    if larsson_fastest == 32:
        larsson_spans = (32, 35, 50, 58)
    elif larsson_fastest <= 20:
        larsson_spans = (larsson_fastest, int(round(larsson_fastest * 1.5)),
                         int(round(larsson_fastest * 2.0)), int(round(larsson_fastest * 2.625)))
    else:
        larsson_spans = (larsson_fastest,
                         int(round(larsson_fastest * 35.0 / 32.0)),
                         int(round(larsson_fastest * 50.0 / 32.0)),
                         int(round(larsson_fastest * 58.0 / 32.0)))

    s1, s2, s3, s4 = larsson_spans
    e1 = scanner_core._pine_ema(df["close"], s1)
    e2 = scanner_core._pine_ema(df["close"], s2)
    e3 = scanner_core._pine_ema(df["close"], s3)
    e4 = scanner_core._pine_ema(df["close"], s4)

    bullish = (e1 >= e2) & (e2 >= e3) & (e3 >= e4)
    bearish = (e1 < e2) & (e2 < e3) & (e3 < e4)
    valid_l = e1.notna() & e2.notna() & e3.notna() & e4.notna()

    l_state = pd.Series("gray", index=df.index)
    l_state[valid_l & bullish] = "yellow"
    l_state[valid_l & bearish] = "blue"

    vol_sma20 = df["volume"].rolling(20).mean()

    # Relative Strength Line vs SPY calculation
    spy_bars = datastore.load_bars("SPY")
    rs_line_data = []
    if spy_bars is not None and not spy_bars.empty:
        idx_shared = df.index.intersection(spy_bars.index)
        if len(idx_shared) >= 20:
            sub_c = df.loc[idx_shared, "close"]
            spy_c = spy_bars.loc[idx_shared, "close"]
            raw_rs = (sub_c / spy_c)
            # Rebase to 100 at starting bar
            norm_rs = (raw_rs / raw_rs.iloc[0]) * 100.0
            rs_ma21 = norm_rs.ewm(span=21, adjust=False).mean()
            for t_idx in idx_shared:
                t_str = t_idx.strftime("%Y-%m-%d")
                rs_line_data.append({
                    "time": t_str,
                    "value": round(float(norm_rs.loc[t_idx]), 2),
                    "ma": round(float(rs_ma21.loc[t_idx]), 2),
                })

    bars = []
    volume = []
    vol_ma20_data = []
    ema10_data = []
    ema20_data = []
    sma50_data = []
    sma200_data = []
    larsson_data = []

    for i, (idx, r) in enumerate(df.iterrows()):
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

    # Historical EPs for markers
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

    # Stock info from RS dataset
    dataset = get_df()
    stock_row = None
    if not dataset.empty:
        matches = dataset[dataset["symbol"] == sym]
        if not matches.empty:
            stock_row = matches.iloc[0].to_dict()

    if not stock_row:
        # Fallback if not in precomputed table
        last_c = float(df["close"].iloc[-1])
        stock_row = {
            "symbol": sym,
            "name": NAMES_MAP.get(sym, sym),
            "rank": 0,
            "rs_score": 50,
            "close": round(last_c, 2),
            "ret_1d": 0.0,
            "ret_1w": 0.0,
            "ret_1m": 0.0,
            "ret_3m": 0.0,
            "ret_6m": 0.0,
            "ret_1y": 0.0,
            "alpha_1m": 0.0,
            "alpha_3m": 0.0,
            "alpha_1y": 0.0,
            "adr_pct": round(float(df["adr_pct"].iloc[-1]), 2) if "adr_pct" in df.columns else 3.0,
            "avg_vol_50": int(df["volume"].rolling(50).mean().iloc[-1]) if len(df) >= 50 else 100000,
            "dvol_m": round(float((df["close"] * df["volume"]).rolling(20).mean().iloc[-1]) / 1e6, 2) if len(df) >= 20 else 10.0,
            "hi_52w": round(float(df["high"].tail(252).max()), 2),
            "lo_52w": round(float(df["low"].tail(252).min()), 2),
            "dist_52w_high": round(float((last_c / df["high"].tail(252).max() - 1.0) * 100), 2),
            "above_10": bool(last_c >= ema10_s.iloc[-1]),
            "above_20": bool(last_c >= ema20_s.iloc[-1]),
            "above_50": bool(last_c >= sma50_s.iloc[-1]),
            "above_200": bool(last_c >= sma200_s.iloc[-1]),
            "stage2": bool(last_c >= ema10_s.iloc[-1] >= ema20_s.iloc[-1] >= sma50_s.iloc[-1] >= sma200_s.iloc[-1]),
            "sector": labels.sector(sym) or "Unknown",
            "theme": (labels.themes(sym) or ["General"])[0],
            "themes": ", ".join(labels.themes(sym) or ["General"]),
        }

    sec_name = str(stock_row.get("sector") or (labels.sector(sym) or "Unknown"))
    thms_list = labels.themes(sym) or []
    primary_thm = str(stock_row.get("theme") or (thms_list[0] if thms_list else "General"))

    # Thematic engine evaluation
    eng = thematic_engine.get_thematic_engine()
    thm_eval = eng.evaluate_symbol(sym) if eng else {}
    sec_info = thm_eval.get("sector_info", {})
    thm_info = thm_eval.get("theme_info", {})
    sec_m1 = float(sec_info.get("m1_pct")) if pd.notna(sec_info.get("m1_pct")) else None
    sec_m3 = float(sec_info.get("m3_pct")) if pd.notna(sec_info.get("m3_pct")) else None
    thm_m1 = float(thm_info.get("m1_pct")) if pd.notna(thm_info.get("m1_pct")) else None

    if (thm_m1 is not None and thm_m1 >= 70) or (sec_m1 is not None and sec_m1 >= 75):
        macro_diag = "Strong Institutional Tailwind"
        macro_desc = "Top decile institutional flow sponsorship. High probability of multi-quarter momentum continuation."
        macro_status = "tailwind"
    elif (thm_m1 is not None and thm_m1 <= 30) or (sec_m1 is not None and sec_m1 <= 25):
        macro_diag = "Institutional Drag / Headwind"
        macro_desc = "Lacking momentum sponsorship. High churn rate; institutional selling on strength."
        macro_status = "drag"
    else:
        macro_diag = "Market-In-Line Flow"
        macro_desc = "Sector and theme momentum moving in-line with broader market indices."
        macro_status = "neutral"

    # Find Top Sector and Theme Peers
    sec_peers = []
    thm_peers = []
    if not dataset.empty:
        if sec_name and sec_name != "Unknown":
            s_match = dataset[(dataset["sector"] == sec_name) & (dataset["symbol"] != sym)]
            sec_peers = s_match.sort_values("rs_score", ascending=False).head(5)[
                ["symbol", "name", "rs_score", "close", "ret_1m", "ret_3m"]
            ].to_dict(orient="records")

        if primary_thm and primary_thm != "General":
            t_match = dataset[(dataset["theme"] == primary_thm) & (dataset["symbol"] != sym)]
            thm_peers = t_match.sort_values("rs_score", ascending=False).head(5)[
                ["symbol", "name", "rs_score", "close", "ret_1m", "ret_3m"]
            ].to_dict(orient="records")

    sector_payload = {
        "name": sec_name,
        "m1_pctile": round(sec_m1, 1) if sec_m1 is not None else 50.0,
        "m3_pctile": round(sec_m3, 1) if sec_m3 is not None else 50.0,
        "w1_ret": sec_info.get("w1_ret"),
        "m1_ret": sec_info.get("m1_ret"),
        "m3_ret": sec_info.get("m3_ret"),
        "ytd_ret": sec_info.get("ytd_ret"),
        "member_count": sec_info.get("member_count", 0),
        "macro_diag": macro_diag,
        "macro_desc": macro_desc,
        "macro_status": macro_status,
        "peers": sec_peers,
    }

    theme_payload = {
        "name": primary_thm,
        "themes_str": stock_row.get("themes", primary_thm),
        "m1_pctile": round(thm_m1, 1) if thm_m1 is not None else 50.0,
        "w1_ret": thm_info.get("w1_ret"),
        "m1_ret": thm_info.get("m1_ret"),
        "m3_ret": thm_info.get("m3_ret"),
        "ytd_ret": thm_info.get("ytd_ret"),
        "member_count": thm_info.get("member_count", 0),
        "primary_archetype": thm_eval.get("primary_archetype", "Market Baseline"),
        "badge_type": thm_eval.get("badge_type", "neutral"),
        "composite_score": thm_eval.get("composite_score", 50.0),
        "peers": thm_peers,
    }

    # Fundamental Catalyst info if any EP occurred
    catalyst_dossier = None
    if symbol_eps:
        recent_ep = symbol_eps[-1]["date"]
        try:
            catalyst_dossier = ep_catalyst_enricher.get_catalyst_dossier(
                sym, recent_ep, sector=sec_name, theme=primary_thm
            )
        except Exception:
            pass

    return jsonify({
        "symbol": sym,
        "bars": bars,
        "volume": volume,
        "vol_ma20": vol_ma20_data,
        "ema10": ema10_data,
        "ema20": ema20_data,
        "sma50": sma50_data,
        "sma200": sma200_data,
        "larsson": larsson_data,
        "larsson_spans": list(larsson_spans),
        "rs_line": rs_line_data,
        "eps": symbol_eps,
        "stock": stock_row,
        "sector": sector_payload,
        "theme": theme_payload,
        "catalyst": catalyst_dossier,
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


@app.route("/api/group_stocks")
def api_group_stocks():
    group = request.args.get("group", "").strip()
    kind = request.args.get("kind", "theme").strip()
    if not group:
        return jsonify({"error": "Missing group"}), 400
    eng = thematic_engine.get_thematic_engine()
    if not eng:
        return jsonify({"group": group, "stocks": []})
    return jsonify(eng.get_group_stocks(group, kind))


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Institutional Relative Strength (RS) Universe Explorer</title>
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
  .title { font-size: 15.5px; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 8px; }
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
  .kpi-label { font-size: 10px; color: var(--muted); text-transform: uppercase; font-weight: 700; margin-bottom: 2px; }
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
  .table-pane { flex: 1.15; min-width: 460px; overflow: auto; border-right: 1px solid var(--border); }
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
  .chart-pane { flex: 1.45; min-width: 480px; min-height: 0; display: flex; flex-direction: column; background: #0c0f16; border-right: 1px solid var(--border); position: relative; overflow: hidden; }

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
  .btn-chart-enlarge:hover { background: #0284c7; color: #fff; }
  .btn-chart-enlarge.active {
    background: #0284c7; color: #fff; border-color: #38bdf8;
    box-shadow: 0 0 8px rgba(56, 189, 248, 0.4);
  }

  .chart-enlarged-only-controls { display: none !important; }
  .workspace.fullscreen-chart-mode .chart-enlarged-only-controls {
    display: inline-flex !important;
    align-items: center;
    gap: 6px;
    flex-wrap: nowrap;
  }

  /* Settings Panel */
  .chart-settings-panel {
    position: absolute; top: 10px; right: 14px; z-index: 60;
    width: 290px; background: rgba(15, 23, 42, 0.95); border: 1px solid #334155;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.7); border-radius: 8px;
    backdrop-filter: blur(10px); overflow: hidden; display: flex; flex-direction: column;
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
  .settings-section { border-bottom: 1px solid #1e293b; padding-bottom: 8px; }
  .settings-section:last-child { border-bottom: none; padding-bottom: 0; }
  .settings-sec-title {
    font-size: 9.5px; font-weight: 800; color: #64748b; letter-spacing: 0.5px;
    margin-bottom: 6px; text-transform: uppercase;
  }
  .settings-row { display: flex; align-items: center; margin-bottom: 4px; }
  .settings-row:last-child { margin-bottom: 0; }
  .toggle-control { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; user-select: none; }

  /* Draw Toolbar */
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
  #draw_canvas {
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none; z-index: 6;
  }
  #larsson_ribbon_svg {
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none; z-index: 4;
  }

  .chart-hud {
    position: absolute; top: 10px; left: 14px; z-index: 20;
    background: rgba(15, 23, 42, 0.9); backdrop-filter: blur(8px);
    border: 1px solid #334155; border-radius: 6px; padding: 4px 10px;
    font-size: 11.5px; color: #e2e8f0; pointer-events: none; display: flex; gap: 12px;
  }

  /* Right Pane: Dossier */
  .dossier-pane {
    flex: 0 0 390px; width: 390px; max-width: 440px; overflow-y: auto;
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

  /* RS Badges */
  .pill-rs-elite { background: rgba(16, 185, 129, 0.25); color: #34d399; border: 1px solid #059669; padding: 2px 7px; border-radius: 4px; font-size: 11px; font-weight: 800; }
  .pill-rs-leader { background: rgba(6, 182, 212, 0.2); color: #22d3ee; border: 1px solid #0891b2; padding: 2px 7px; border-radius: 4px; font-size: 11px; font-weight: 800; }
  .pill-rs-mid { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #d97706; padding: 2px 7px; border-radius: 4px; font-size: 11px; font-weight: 800; }
  .pill-rs-lag { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid #dc2626; padding: 2px 7px; border-radius: 4px; font-size: 11px; font-weight: 700; }

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

  /* 52w Range Bar */
  .range-bar-box {
    width: 100%; height: 8px; background: #1e293b; border-radius: 4px; position: relative; margin: 6px 0;
  }
  .range-bar-fill {
    height: 100%; background: linear-gradient(90deg, #3b82f6, #10b981); border-radius: 4px;
  }
  .range-bar-thumb {
    position: absolute; top: -3px; width: 6px; height: 14px; background: #fff; border-radius: 2px;
    box-shadow: 0 0 6px rgba(255,255,255,0.8); transform: translateX(-50%);
  }

  /* Peer Chips */
  .peer-item {
    display: flex; justify-content: space-between; align-items: center;
    padding: 4px 6px; background: #151d2c; border: 1px solid #243046; border-radius: 4px;
    cursor: pointer; transition: all 0.12s ease;
  }
  .peer-item:hover { background: #1e2b42; border-color: #38bdf8; }

  /* Modal Chooser */
  .modal-overlay {
    position: fixed; inset: 0; background: rgba(0,0,0,0.7); z-index: 10000;
    display: flex; align-items: center; justify-content: center; backdrop-filter: blur(4px);
  }
  .modal-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    width: 580px; max-height: 80vh; display: flex; flex-direction: column; overflow: hidden;
    box-shadow: 0 10px 40px rgba(0,0,0,0.8);
  }
  .modal-header {
    padding: 10px 14px; background: #11151f; border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
  }
  .modal-body { padding: 12px; overflow-y: auto; flex: 1; }
</style>
</head>
<body>

<header>
  <div class="title">
    <span>Institutional Relative Strength (RS) Explorer</span>
    <span class="badge">Port 8786</span>
    <span class="badge" style="color:#38bdf8; border-color:#0284c7;">IBD 1-99 System</span>
  </div>
  <div style="display:flex; align-items:center; gap:10px;">
    <button class="reload-btn" onclick="recomputeRS()" title="Re-scan and recalculate RS for all tickers in universe">
      <span>🔄</span> Recalculate RS
    </button>
  </div>
</header>

<!-- KPI Summary Row -->
<div class="kpi-row">
  <div class="kpi-card">
    <div class="kpi-label">Active Stocks</div>
    <div class="kpi-val" id="kpi_total">0</div>
    <div class="kpi-sub">In Universe</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">RS 90+ Leaders</div>
    <div class="kpi-val" id="kpi_rs90" style="color:var(--green);">0</div>
    <div class="kpi-sub">Top Decile (IBD Elite)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Stage 2 Bullish Stack</div>
    <div class="kpi-val" id="kpi_stage2" style="color:var(--cyan);">0</div>
    <div class="kpi-sub">C > 10 > 20 > 50 > 200</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Near 52W High (≤15%)</div>
    <div class="kpi-val" id="kpi_near_high" style="color:var(--gold);">0</div>
    <div class="kpi-sub">Institutional Leadership</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">SPY Benchmark</div>
    <div class="kpi-val" id="kpi_spy_bench" style="font-size:12.5px;">1M: 0.0% • 3M: 0.0%</div>
    <div class="kpi-sub" id="kpi_spy_sub">1Y: 0.0%</div>
  </div>
</div>

<!-- Filter Toolbar -->
<div class="toolbar">
  <div class="filter-group">
    <label>Search:</label>
    <input type="text" id="filter_search" placeholder="Ticker or Name..." oninput="onFilterChange()" style="width:130px;">
  </div>
  <div class="filter-group">
    <label>Universe:</label>
    <select id="filter_universe" onchange="onFilterChange()">
      <option value="common" selected>Common Stocks</option>
      <option value="etf">ETFs Only</option>
      <option value="all">All Securities</option>
    </select>
  </div>
  <div class="filter-group">
    <label>Min RS:</label>
    <div style="display:flex; gap:4px;">
      <button class="btn-filter-chip active" id="chip_rs_all" onclick="setRSChip('')">All</button>
      <button class="btn-filter-chip" id="chip_rs_70" onclick="setRSChip('70')">70+</button>
      <button class="btn-filter-chip" id="chip_rs_80" onclick="setRSChip('80')">80+</button>
      <button class="btn-filter-chip" id="chip_rs_90" onclick="setRSChip('90')">90+</button>
      <button class="btn-filter-chip" id="chip_rs_95" onclick="setRSChip('95')">95+</button>
    </div>
  </div>
  <div class="filter-group">
    <label>Liquidity:</label>
    <select id="filter_dvol" onchange="onFilterChange()">
      <option value="" selected>All $Vol</option>
      <option value="5">$5M+ / day</option>
      <option value="20">$20M+ / day</option>
      <option value="50">$50M+ / day</option>
      <option value="100">$100M+ / day</option>
    </select>
  </div>
  <div class="filter-group">
    <label>Trend:</label>
    <select id="filter_trend" onchange="onFilterChange()">
      <option value="all" selected>All Postures</option>
      <option value="stage2">Stage 2 Bullish Stack</option>
      <option value="above_50">Above 50 SMA</option>
      <option value="above_200">Above 200 SMA</option>
      <option value="near_high">Within 15% 52W High</option>
    </select>
  </div>
  <div class="filter-group">
    <label>Sector:</label>
    <select id="filter_sector" onchange="onFilterChange()" style="max-width:140px;">
      <option value="all">All Sectors</option>
    </select>
  </div>
  <div class="filter-group">
    <label>Theme:</label>
    <select id="filter_theme" onchange="onFilterChange()" style="max-width:140px;">
      <option value="all">All Themes</option>
    </select>
  </div>
</div>

<!-- 3-Pane Workspace -->
<div class="workspace" id="main_workspace">

  <!-- Left Pane: Stock Table -->
  <div class="table-pane">
    <table>
      <thead>
        <tr>
          <th onclick="changeSort('rank')" style="width:36px;">#</th>
          <th class="tl" onclick="changeSort('symbol')">Symbol</th>
          <th class="tl" onclick="changeSort('name')">Name</th>
          <th onclick="changeSort('rs_score')">RS</th>
          <th onclick="changeSort('close')">Price</th>
          <th onclick="changeSort('ret_1d')">1D %</th>
          <th onclick="changeSort('ret_1w')">1W %</th>
          <th onclick="changeSort('ret_1m')">1M %</th>
          <th onclick="changeSort('ret_3m')">3M %</th>
          <th onclick="changeSort('ret_1y')">1Y %</th>
          <th onclick="changeSort('adr_pct')">ADR%</th>
          <th onclick="changeSort('dvol_m')">$Vol(M)</th>
          <th class="tl" onclick="changeSort('sector')">Sector</th>
          <th class="tl" onclick="changeSort('theme')">Theme</th>
        </tr>
      </thead>
      <tbody id="stocks_table_body"></tbody>
    </table>
  </div>

  <!-- Center Pane: Chart -->
  <div class="chart-pane">
    <div class="chart-header">
      <div class="chart-title-box">
        <span id="chart_symbol" style="color:#fff; font-size:15px;">NVDA</span>
        <span id="chart_name" style="color:var(--muted); font-size:12px; font-weight:normal;">NVIDIA Corp</span>
        <span id="chart_rs_pill" class="pill-rs-elite">RS 99</span>
        <span id="chart_stage2_badge" class="badge" style="color:var(--cyan); border-color:#0891b2; font-size:10px;">STAGE 2</span>
      </div>
      <div class="chart-controls">
        <div class="chart-enlarged-only-controls">
          <button id="btn_enlarged_stock_list" class="btn-chart-enlarge" onclick="openStockPickerModal()">
            📋 Choose Stock (<span id="lbl_enlarged_stock_count">0</span>)
          </button>
          <button class="btn-chart-enlarge" onclick="stepStock(-1)">◀ Prev</button>
          <button class="btn-chart-enlarge" onclick="stepStock(1)">Next ▶</button>
        </div>
        <button id="btn_chart_settings" class="btn-chart-enlarge" onclick="toggleSettingsPanel()">⚙️ Settings</button>
        <button id="btn_toggle_draw" class="btn-chart-enlarge" onclick="toggleDrawToolbar()">✏️ Tools</button>
        <button class="btn-chart-enlarge" onclick="resetChart()">⟲ Reset</button>
        <button id="btn_chart_fullscreen" class="btn-chart-enlarge" onclick="toggleFullscreen()">⛶ Enlarge</button>
      </div>
    </div>

    <!-- Floating Draw Toolbar -->
    <div class="chart-draw-toolbar" id="chart_draw_toolbar" style="display:none;">
      <button class="draw-btn active" id="btn_tool_cursor" onclick="setDrawTool('cursor')">👆 Pan</button>
      <button class="draw-btn" id="btn_tool_line" onclick="setDrawTool('line')">📏 Line</button>
      <button class="draw-btn" id="btn_tool_hline" onclick="setDrawTool('hline')">━ H-Line</button>
      <button class="draw-btn" id="btn_tool_measure" onclick="setDrawTool('measure')">📐 %Δ</button>
      <button class="draw-btn draw-btn-danger" onclick="clearDrawings()">🗑️ Clear</button>
    </div>

    <!-- Chart Settings Floating Panel -->
    <div class="chart-settings-panel" id="chart_settings_panel" style="display:none;">
      <div class="settings-panel-header">
        <span style="font-weight:700; color:#e2e8f0; font-size:12px;">Chart Display Settings</span>
        <button class="panel-close-btn" onclick="toggleSettingsPanel()">✕</button>
      </div>
      <div class="settings-panel-body">
        <div class="settings-section">
          <div class="settings-sec-title">Moving Averages</div>
          <div class="settings-row"><label class="toggle-control"><input type="checkbox" id="cfg_ema10" checked onchange="updateVisibility()"><span style="color:#38bdf8;">EMA 10 (Cyan)</span></label></div>
          <div class="settings-row"><label class="toggle-control"><input type="checkbox" id="cfg_ema20" checked onchange="updateVisibility()"><span style="color:#eab308;">EMA 20 (Gold)</span></label></div>
          <div class="settings-row"><label class="toggle-control"><input type="checkbox" id="cfg_sma50" checked onchange="updateVisibility()"><span style="color:#ef4444;">SMA 50 (Red)</span></label></div>
          <div class="settings-row"><label class="toggle-control"><input type="checkbox" id="cfg_sma200" checked onchange="updateVisibility()"><span style="color:#64748b;">SMA 200 (Gray)</span></label></div>
        </div>
        <div class="settings-section">
          <div class="settings-sec-title">Relative Strength (RS) Line</div>
          <div class="settings-row"><label class="toggle-control"><input type="checkbox" id="cfg_rs_line" checked onchange="updateVisibility()"><span style="color:#a855f7;">RS Line vs SPY + 21 EMA</span></label></div>
        </div>
        <div class="settings-section">
          <div class="settings-sec-title">Larsson Line Ribbon</div>
          <div class="settings-row"><label class="toggle-control"><input type="checkbox" id="cfg_larsson" checked onchange="updateVisibility()"><span>Larsson Ribbon (Bullish yellow / Bearish blue)</span></label></div>
          <div style="display:flex; align-items:center; gap:6px; margin-top:6px;">
            <span style="color:var(--muted); font-size:11px;">Fastest Span:</span>
            <input type="number" id="cfg_larsson_fastest" value="32" min="5" max="100" style="width:55px;" onchange="onLarssonSpanChange()">
          </div>
        </div>
        <div class="settings-section">
          <div class="settings-sec-title">Markers & Volume</div>
          <div class="settings-row"><label class="toggle-control"><input type="checkbox" id="cfg_volume" checked onchange="updateVisibility()"><span>Volume Histogram + 20-MA</span></label></div>
          <div class="settings-row"><label class="toggle-control"><input type="checkbox" id="cfg_ep_markers" checked onchange="updateVisibility()"><span>EP Historical Catalyst Markers</span></label></div>
        </div>
      </div>
    </div>

    <!-- Chart Rel Container -->
    <div class="chart-container-rel">
      <div id="chart_tv"></div>
      <svg id="larsson_ribbon_svg"></svg>
      <canvas id="draw_canvas"></canvas>
      <div class="chart-hud" id="chart_hud" style="display:none;"></div>
    </div>
  </div>

  <!-- Right Pane: Comprehensive Dossier -->
  <div class="dossier-pane" id="dossier_pane">

    <!-- Card 1: Stock RS & Technical Posture -->
    <div class="dossier-card">
      <div class="dossier-title">
        <span>Stock RS Profile</span>
        <span id="dos_rank" class="badge">#1</span>
      </div>
      <div style="display:flex; align-items:baseline; justify-content:space-between; margin-top:2px;">
        <div>
          <span id="dos_sym" style="font-size:18px; font-weight:800; color:#fff;">NVDA</span>
          <div id="dos_company_name" style="font-size:11px; color:var(--muted);">NVIDIA Corporation</div>
        </div>
        <div style="text-align:right;">
          <span id="dos_price" style="font-size:18px; font-weight:800; color:#fff;">$0.00</span>
          <div id="dos_chg_1d" style="font-size:11px; font-weight:700;">+0.0%</div>
        </div>
      </div>

      <!-- 52w Range -->
      <div style="margin-top:4px;">
        <div style="display:flex; justify-content:space-between; font-size:10.5px; color:var(--muted);">
          <span>52W Lo: <strong id="dos_lo52">$0.00</strong></span>
          <span id="dos_dist_hi" style="color:#38bdf8;">0.0% off High</span>
          <span>52W Hi: <strong id="dos_hi52">$0.00</strong></span>
        </div>
        <div class="range-bar-box">
          <div class="range-bar-fill" id="dos_range_fill" style="width:50%;"></div>
          <div class="range-bar-thumb" id="dos_range_thumb" style="left:50%;"></div>
        </div>
      </div>

      <!-- Returns & Alpha vs SPY Table -->
      <table style="margin-top:6px; font-size:11px;">
        <thead>
          <tr style="background:#111827;">
            <th class="tl" style="padding:3px 6px;">Horizon</th>
            <th style="padding:3px 6px;">Stock Return</th>
            <th style="padding:3px 6px;">SPY Return</th>
            <th style="padding:3px 6px;">Alpha Spread</th>
          </tr>
        </thead>
        <tbody id="dos_returns_body"></tbody>
      </table>

      <!-- Volatility & Alignment -->
      <div style="display:flex; flex-direction:column; gap:4px; margin-top:4px;">
        <div class="dossier-row">
          <span class="dossier-label">20D ADR:</span>
          <span class="dossier-val" id="dos_adr">0.0%</span>
        </div>
        <div class="dossier-row">
          <span class="dossier-label">20D Dollar Volume:</span>
          <span class="dossier-val" id="dos_dvol">$0.0M / day</span>
        </div>
        <div class="dossier-row">
          <span class="dossier-label">Trend Alignment:</span>
          <span class="dossier-val" id="dos_alignment">Stage 2 Bullish</span>
        </div>
      </div>
    </div>

    <!-- Card 2: Sector Context & Institutional Flow -->
    <div class="dossier-card">
      <div class="dossier-title">
        <span>Sector Context</span>
        <span id="dos_sec_badge" class="badge" style="color:#38bdf8;">Leading</span>
      </div>
      <div style="font-size:13.5px; font-weight:700; color:#fff;" id="dos_sector_name">Technology</div>
      <div style="font-size:11px; color:var(--muted);" id="dos_sector_diag">Strong Institutional Tailwind</div>

      <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:6px; text-align:center; margin-top:4px;">
        <div style="background:#131a26; padding:4px; border-radius:4px;">
          <div style="font-size:9.5px; color:var(--muted);">1W</div>
          <div style="font-weight:700;" id="dos_sec_1w">0.0%</div>
        </div>
        <div style="background:#131a26; padding:4px; border-radius:4px;">
          <div style="font-size:9.5px; color:var(--muted);">1M (Pct)</div>
          <div style="font-weight:700;" id="dos_sec_1m">50th</div>
        </div>
        <div style="background:#131a26; padding:4px; border-radius:4px;">
          <div style="font-size:9.5px; color:var(--muted);">3M (Pct)</div>
          <div style="font-weight:700;" id="dos_sec_3m">50th</div>
        </div>
        <div style="background:#131a26; padding:4px; border-radius:4px;">
          <div style="font-size:9.5px; color:var(--muted);">YTD</div>
          <div style="font-weight:700;" id="dos_sec_ytd">0.0%</div>
        </div>
      </div>

      <div style="font-size:10.5px; font-weight:700; color:var(--muted); text-transform:uppercase; margin-top:6px;">Top Sector RS Leaders:</div>
      <div id="dos_sector_peers" style="display:flex; flex-direction:column; gap:4px;"></div>
    </div>

    <!-- Card 3: Theme Context & Momentum Archetype -->
    <div class="dossier-card">
      <div class="dossier-title">
        <span>Thematic Momentum</span>
        <span id="dos_thm_badge" class="badge" style="color:var(--gold);">Surge</span>
      </div>
      <div style="font-size:13.5px; font-weight:700; color:#fff;" id="dos_theme_name">Artificial Intelligence</div>
      <div style="font-size:11px; color:var(--muted);" id="dos_theme_archetype">Multi-TF Power Cluster</div>

      <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:6px; text-align:center; margin-top:4px;">
        <div style="background:#131a26; padding:4px; border-radius:4px;">
          <div style="font-size:9.5px; color:var(--muted);">1W</div>
          <div style="font-weight:700;" id="dos_thm_1w">0.0%</div>
        </div>
        <div style="background:#131a26; padding:4px; border-radius:4px;">
          <div style="font-size:9.5px; color:var(--muted);">1M (Pct)</div>
          <div style="font-weight:700;" id="dos_thm_1m">50th</div>
        </div>
        <div style="background:#131a26; padding:4px; border-radius:4px;">
          <div style="font-size:9.5px; color:var(--muted);">3M</div>
          <div style="font-weight:700;" id="dos_thm_3m">0.0%</div>
        </div>
        <div style="background:#131a26; padding:4px; border-radius:4px;">
          <div style="font-size:9.5px; color:var(--muted);">Score</div>
          <div style="font-weight:700; color:var(--cyan);" id="dos_thm_score">50.0</div>
        </div>
      </div>

      <div style="font-size:10.5px; font-weight:700; color:var(--muted); text-transform:uppercase; margin-top:6px;">Top Theme RS Leaders:</div>
      <div id="dos_theme_peers" style="display:flex; flex-direction:column; gap:4px;"></div>
    </div>

    <!-- Card 4: Historical EP Catalysts (if any) -->
    <div class="dossier-card" id="dos_ep_card" style="display:none;">
      <div class="dossier-title">
        <span>Historical EP Catalysts</span>
        <span class="badge" style="color:var(--green); border-color:#059669;">Episodic Pivot</span>
      </div>
      <div id="dos_ep_content" style="font-size:11.5px; display:flex; flex-direction:column; gap:6px;"></div>
    </div>

  </div>
</div>

<!-- Stock Picker Modal for Enlarged Mode -->
<div class="modal-overlay" id="stock_picker_modal" style="display:none;" onclick="closeStockPickerModal(event)">
  <div class="modal-card" onclick="event.stopPropagation()">
    <div class="modal-header">
      <span style="font-weight:700; color:#fff; font-size:13px;">Choose Stock (<span id="modal_stock_count">0</span> filtered)</span>
      <button class="panel-close-btn" onclick="document.getElementById('stock_picker_modal').style.display='none'">✕</button>
    </div>
    <div style="padding:8px 12px; background:#0f131a; border-bottom:1px solid var(--border);">
      <input type="text" id="modal_search" placeholder="Type to filter..." style="width:100%;" oninput="renderModalStocks()">
    </div>
    <div class="modal-body" id="modal_stock_list"></div>
  </div>
</div>

<script>
let allStocks = [];
let filteredStocks = [];
let currentStock = null;
let currentChartData = null;
let selectedMinRS = '';
let currentSortBy = 'rs_score';
let currentSortDir = 'desc';

// Chart globals
let chart = null;
let candleSeries = null;
let volumeSeries = null;
let volMa20Series = null;
let ema10Series = null;
let ema20Series = null;
let sma50Series = null;
let sma200Series = null;
let rsLineSeries = null;
let rsMaSeries = null;
let larssonFastSeries = null;
let larssonSlowSeries = null;

// Drawing globals
let drawTool = 'cursor';
let drawings = [];
let currentDrawing = null;
let isDrawing = false;

// API calls
function loadStocks() {
  const uType = document.getElementById('filter_universe').value;
  const search = document.getElementById('filter_search').value.trim();
  const dvol = document.getElementById('filter_dvol').value;
  const trend = document.getElementById('filter_trend').value;
  const sec = document.getElementById('filter_sector').value;
  const thm = document.getElementById('filter_theme').value;

  const url = `/api/stocks?universe_type=${encodeURIComponent(uType)}` +
    `&min_rs=${encodeURIComponent(selectedMinRS)}` +
    `&min_dvol=${encodeURIComponent(dvol)}` +
    `&trend=${encodeURIComponent(trend)}` +
    `&sector=${encodeURIComponent(sec)}` +
    `&theme=${encodeURIComponent(thm)}` +
    `&search=${encodeURIComponent(search)}` +
    `&sort_by=${encodeURIComponent(currentSortBy)}` +
    `&sort_dir=${encodeURIComponent(currentSortDir)}`;

  fetch(url)
    .then(r => r.json())
    .then(data => {
      filteredStocks = data.stocks || [];
      updateKPIs(data.kpis);
      populateDropdowns(data.sectors, data.themes);
      renderTable();
      if (!currentStock && filteredStocks.length > 0) {
        selectStock(filteredStocks[0].symbol);
      }
    })
    .catch(err => console.error('Failed to load stocks:', err));
}

function updateKPIs(k) {
  if (!k) return;
  document.getElementById('kpi_total').textContent = k.total.toLocaleString();
  document.getElementById('kpi_rs90').textContent = k.rs90_count.toLocaleString();
  document.getElementById('kpi_stage2').textContent = k.stage2_count.toLocaleString();
  document.getElementById('kpi_near_high').textContent = k.near_high_count.toLocaleString();
  document.getElementById('kpi_spy_bench').textContent = `1M: ${k.spy_1m > 0 ? '+' : ''}${k.spy_1m}% • 3M: ${k.spy_3m > 0 ? '+' : ''}${k.spy_3m}%`;
  document.getElementById('kpi_spy_sub').textContent = `1Y: ${k.spy_1y > 0 ? '+' : ''}${k.spy_1y}% Benchmark`;
}

function populateDropdowns(sectors, themes) {
  const secSel = document.getElementById('filter_sector');
  const thmSel = document.getElementById('filter_theme');
  if (secSel.options.length <= 1 && sectors) {
    sectors.forEach(s => {
      const opt = document.createElement('option');
      opt.value = s; opt.textContent = s;
      secSel.appendChild(opt);
    });
  }
  if (thmSel.options.length <= 1 && themes) {
    themes.forEach(t => {
      const opt = document.createElement('option');
      opt.value = t; opt.textContent = t;
      thmSel.appendChild(opt);
    });
  }
}

function renderTable() {
  const tbody = document.getElementById('stocks_table_body');
  tbody.innerHTML = '';

  filteredStocks.forEach((s, idx) => {
    const tr = document.createElement('tr');
    tr.id = `stock_row_${s.symbol}`;
    if (currentStock && currentStock.symbol === s.symbol) tr.classList.add('selected');

    let rsClass = 'pill-rs-lag';
    if (s.rs_score >= 90) rsClass = 'pill-rs-elite';
    else if (s.rs_score >= 80) rsClass = 'pill-rs-leader';
    else if (s.rs_score >= 70) rsClass = 'pill-rs-mid';

    const d1Color = s.ret_1d >= 0 ? '#34d399' : '#f87171';
    const w1Color = s.ret_1w >= 0 ? '#34d399' : '#f87171';
    const m1Color = s.ret_1m >= 0 ? '#34d399' : '#f87171';
    const m3Color = s.ret_3m >= 0 ? '#34d399' : '#f87171';
    const y1Color = s.ret_1y >= 0 ? '#34d399' : '#f87171';

    tr.innerHTML = `
      <td style="color:var(--muted);">${s.rank || (idx + 1)}</td>
      <td class="tl"><strong style="color:#fff;">${s.symbol}</strong></td>
      <td class="tl" style="max-width:140px; overflow:hidden; text-overflow:ellipsis;" title="${s.name}">${s.name}</td>
      <td><span class="${rsClass}">${s.rs_score}</span></td>
      <td>$${s.close.toFixed(2)}</td>
      <td style="color:${d1Color};">${s.ret_1d > 0 ? '+' : ''}${s.ret_1d.toFixed(1)}%</td>
      <td style="color:${w1Color};">${s.ret_1w > 0 ? '+' : ''}${s.ret_1w.toFixed(1)}%</td>
      <td style="color:${m1Color};">${s.ret_1m > 0 ? '+' : ''}${s.ret_1m.toFixed(1)}%</td>
      <td style="color:${m3Color};">${s.ret_3m > 0 ? '+' : ''}${s.ret_3m.toFixed(1)}%</td>
      <td style="color:${y1Color};">${s.ret_1y > 0 ? '+' : ''}${s.ret_1y.toFixed(1)}%</td>
      <td>${s.adr_pct.toFixed(1)}%</td>
      <td>$${s.dvol_m.toFixed(1)}M</td>
      <td class="tl" style="color:var(--muted); max-width:110px; overflow:hidden; text-overflow:ellipsis;">${s.sector}</td>
      <td class="tl" style="color:var(--muted); max-width:110px; overflow:hidden; text-overflow:ellipsis;">${s.theme}</td>
    `;

    tr.onclick = () => selectStock(s.symbol);
    tbody.appendChild(tr);
  });
}

function selectStock(sym) {
  if (!sym) return;
  document.querySelectorAll('#stocks_table_body tr').forEach(r => r.classList.remove('selected'));
  const row = document.getElementById(`stock_row_${sym}`);
  if (row) {
    row.classList.add('selected');
    row.scrollIntoView({ block: 'nearest' });
  }

  const lFastest = document.getElementById('cfg_larsson_fastest').value || 32;
  fetch(`/api/chart?symbol=${encodeURIComponent(sym)}&larsson_fastest=${encodeURIComponent(lFastest)}`)
    .then(r => r.json())
    .then(data => {
      if (data.error) {
        console.error(data.error);
        return;
      }
      currentStock = data.stock;
      currentChartData = data;
      renderChart(data);
      renderDossier(data);
      loadAnnotations(sym);
      updateHeaderInfo(data);
    })
    .catch(err => console.error('Failed to load chart:', err));
}

function updateHeaderInfo(d) {
  document.getElementById('chart_symbol').textContent = d.symbol;
  document.getElementById('chart_name').textContent = d.stock.name || '';
  const pill = document.getElementById('chart_rs_pill');
  pill.textContent = `RS ${d.stock.rs_score}`;
  pill.className = d.stock.rs_score >= 90 ? 'pill-rs-elite' : (d.stock.rs_score >= 80 ? 'pill-rs-leader' : 'pill-rs-mid');
  const s2Badge = document.getElementById('chart_stage2_badge');
  s2Badge.style.display = d.stock.stage2 ? 'inline-block' : 'none';
  document.getElementById('lbl_enlarged_stock_count').textContent = filteredStocks.length;
}

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
      scaleMargins: { top: 0.08, bottom: 0.28 },
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
    scaleMargins: { top: 0.82, bottom: 0.0 },
  });

  // Relative strength line vs SPY (overlaid in bottom pane)
  rsLineSeries = chart.addLineSeries({
    color: '#a855f7',
    lineWidth: 1.5,
    title: 'RS vs SPY',
    priceScaleId: 'rs_scale',
    priceLineVisible: false,
    lastValueVisible: true,
  });
  rsMaSeries = chart.addLineSeries({
    color: '#fbbf24',
    lineWidth: 1,
    title: 'RS 21EMA',
    priceScaleId: 'rs_scale',
    priceLineVisible: false,
    lastValueVisible: false,
  });
  chart.priceScale('rs_scale').applyOptions({
    scaleMargins: { top: 0.70, bottom: 0.16 },
  });

  larssonFastSeries = chart.addLineSeries({
    lineWidth: 1, color: 'rgba(250, 204, 21, 0.001)',
    priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false
  });
  larssonSlowSeries = chart.addLineSeries({
    lineWidth: 1, color: 'rgba(56, 189, 248, 0.001)',
    priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false
  });

  candleSeries = chart.addCandlestickSeries({
    upColor: '#10b981', downColor: '#ef4444', borderVisible: false,
    wickUpColor: '#10b981', wickDownColor: '#ef4444'
  });

  ema10Series = chart.addLineSeries({ color: '#38bdf8', lineWidth: 1.5, title: 'EMA10' });
  ema20Series = chart.addLineSeries({ color: '#eab308', lineWidth: 1.5, title: 'EMA20' });
  sma50Series = chart.addLineSeries({ color: '#ef4444', lineWidth: 1.5, title: 'SMA50' });
  sma200Series = chart.addLineSeries({ color: '#64748b', lineWidth: 1.5, title: 'SMA200' });

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
    const rsVal = param.seriesPrices.get(rsLineSeries);

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
      ${rsVal ? `<span style="color:#a855f7;">RS: ${rsVal.toFixed(1)}</span>` : ''}
    `;
  });

  chart.timeScale().subscribeVisibleLogicalRangeChange(() => {
    scheduleLarssonRender();
    renderAllDrawings();
  });

  const canvas = document.getElementById('draw_canvas');
  canvas.addEventListener('mousedown', onCanvasMouseDown);
  canvas.addEventListener('mousemove', onCanvasMouseMove);

  window.addEventListener('resize', () => {
    if (chart && container) {
      chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
      resizeCanvas();
      scheduleLarssonRender();
    }
  });
  resizeCanvas();
}

function renderChart(data) {
  if (!chart) initChart();

  candleSeries.setData(data.bars);
  volumeSeries.setData(data.volume);
  volMa20Series.setData(data.vol_ma20);
  ema10Series.setData(data.ema10);
  ema20Series.setData(data.ema20);
  sma50Series.setData(data.sma50);
  sma200Series.setData(data.sma200);

  if (data.rs_line && data.rs_line.length > 0) {
    rsLineSeries.setData(data.rs_line.map(x => ({ time: x.time, value: x.value })));
    rsMaSeries.setData(data.rs_line.map(x => ({ time: x.time, value: x.ma })));
  } else {
    rsLineSeries.setData([]);
    rsMaSeries.setData([]);
  }

  larssonFastSeries.setData(data.larsson.map(x => ({ time: x.time, value: x.fast || 0 })));
  larssonSlowSeries.setData(data.larsson.map(x => ({ time: x.time, value: x.slow || 0 })));

  window._currentLarssonData = data.larsson;
  scheduleLarssonRender();

  // Markers for historical EPs
  if (data.eps && data.eps.length > 0 && document.getElementById('cfg_ep_markers').checked) {
    const markers = data.eps.map(ep => ({
      time: ep.date,
      position: 'aboveBar',
      color: '#10b981',
      shape: 'arrowDown',
      text: `EP +${ep.gap_pct}%`,
    }));
    candleSeries.setMarkers(markers);
  } else {
    candleSeries.setMarkers([]);
  }

  chart.timeScale().fitContent();
  updateVisibility();
}

function renderDossier(data) {
  const s = data.stock;
  const sec = data.sector;
  const thm = data.theme;

  document.getElementById('dos_rank').textContent = `#${s.rank || '-'}`;
  document.getElementById('dos_sym').textContent = s.symbol;
  document.getElementById('dos_company_name').textContent = s.name || s.symbol;
  document.getElementById('dos_price').textContent = `$${s.close.toFixed(2)}`;
  const chgEl = document.getElementById('dos_chg_1d');
  chgEl.textContent = `${s.ret_1d > 0 ? '+' : ''}${s.ret_1d.toFixed(2)}%`;
  chgEl.style.color = s.ret_1d >= 0 ? '#34d399' : '#f87171';

  document.getElementById('dos_lo52').textContent = `$${s.lo_52w.toFixed(2)}`;
  document.getElementById('dos_hi52').textContent = `$${s.hi_52w.toFixed(2)}`;
  document.getElementById('dos_dist_hi').textContent = `${s.dist_52w_high > 0 ? '+' : ''}${s.dist_52w_high.toFixed(1)}% off High`;

  // Range bar positioning
  const rangeSpan = Math.max(0.01, s.hi_52w - s.lo_52w);
  const pct = Math.min(100, Math.max(0, (s.close - s.lo_52w) / rangeSpan * 100));
  document.getElementById('dos_range_fill').style.width = `${pct}%`;
  document.getElementById('dos_range_thumb').style.left = `${pct}%`;

  // Multi-Timeframe Return & Alpha Table
  const retTable = document.getElementById('dos_returns_body');
  retTable.innerHTML = '';
  const horizons = [
    { label: '1-Day', ret: s.ret_1d, spy: null, alpha: null },
    { label: '1-Week', ret: s.ret_1w, spy: null, alpha: null },
    { label: '1-Month', ret: s.ret_1m, spy: (s.ret_1m - s.alpha_1m), alpha: s.alpha_1m },
    { label: '3-Month', ret: s.ret_3m, spy: (s.ret_3m - s.alpha_3m), alpha: s.alpha_3m },
    { label: '6-Month', ret: s.ret_6m, spy: null, alpha: null },
    { label: '1-Year', ret: s.ret_1y, spy: (s.ret_1y - s.alpha_1y), alpha: s.alpha_1y },
  ];

  horizons.forEach(h => {
    const tr = document.createElement('tr');
    const stCol = h.ret >= 0 ? '#34d399' : '#f87171';
    const spyCol = (h.spy !== null && h.spy >= 0) ? '#94a3b8' : '#64748b';
    const aCol = (h.alpha !== null && h.alpha >= 0) ? '#38bdf8' : '#f87171';

    tr.innerHTML = `
      <td class="tl" style="color:var(--muted); padding:3px 6px;">${h.label}</td>
      <td style="color:${stCol}; font-weight:600; padding:3px 6px;">${h.ret > 0 ? '+' : ''}${h.ret.toFixed(1)}%</td>
      <td style="color:${spyCol}; padding:3px 6px;">${h.spy !== null ? (h.spy > 0 ? '+' : '') + h.spy.toFixed(1) + '%' : '-'}</td>
      <td style="color:${aCol}; font-weight:700; padding:3px 6px;">${h.alpha !== null ? (h.alpha > 0 ? '+' : '') + h.alpha.toFixed(1) + '%' : '-'}</td>
    `;
    retTable.appendChild(tr);
  });

  document.getElementById('dos_adr').textContent = `${s.adr_pct.toFixed(2)}% ($${(s.close * s.adr_pct / 100).toFixed(2)})`;
  document.getElementById('dos_dvol').textContent = `$${s.dvol_m.toFixed(1)}M / day`;
  const alignEl = document.getElementById('dos_alignment');
  if (s.stage2) {
    alignEl.innerHTML = '<span style="color:#22d3ee; font-weight:700;">★ Full Stage 2 Stack</span>';
  } else if (s.above_50) {
    alignEl.innerHTML = '<span style="color:#34d399;">Above 50 SMA</span>';
  } else {
    alignEl.innerHTML = '<span style="color:#f87171;">Below 50 SMA (Consolidating)</span>';
  }

  // Sector Card
  document.getElementById('dos_sector_name').textContent = sec.name || 'Unknown';
  document.getElementById('dos_sector_diag').textContent = sec.macro_desc || sec.macro_diag;
  document.getElementById('dos_sec_1w').textContent = sec.w1_ret !== undefined && sec.w1_ret !== null ? `${sec.w1_ret > 0 ? '+' : ''}${sec.w1_ret.toFixed(1)}%` : '-';
  document.getElementById('dos_sec_1m').textContent = `${sec.m1_pctile.toFixed(0)}th %ile`;
  document.getElementById('dos_sec_3m').textContent = `${sec.m3_pctile.toFixed(0)}th %ile`;
  document.getElementById('dos_sec_ytd').textContent = sec.ytd_ret !== undefined && sec.ytd_ret !== null ? `${sec.ytd_ret > 0 ? '+' : ''}${sec.ytd_ret.toFixed(1)}%` : '-';

  const secBadge = document.getElementById('dos_sec_badge');
  secBadge.textContent = sec.macro_status ? sec.macro_status.toUpperCase() : 'NEUTRAL';
  secBadge.style.color = sec.macro_status === 'tailwind' ? 'var(--green)' : (sec.macro_status === 'drag' ? 'var(--red)' : '#38bdf8');

  const secPeersContainer = document.getElementById('dos_sector_peers');
  secPeersContainer.innerHTML = '';
  if (sec.peers && sec.peers.length > 0) {
    sec.peers.forEach(p => {
      const div = document.createElement('div');
      div.className = 'peer-item';
      div.onclick = () => selectStock(p.symbol);
      div.innerHTML = `
        <div style="display:flex; align-items:center; gap:6px;">
          <strong style="color:#fff;">${p.symbol}</strong>
          <span style="color:var(--muted); font-size:11px; max-width:120px; overflow:hidden; text-overflow:ellipsis;">${p.name || ''}</span>
        </div>
        <div style="display:flex; align-items:center; gap:6px;">
          <span class="pill-rs-elite" style="font-size:10px; padding:1px 5px;">RS ${p.rs_score}</span>
          <span style="font-weight:600; color:#cbd5e1;">$${p.close.toFixed(2)}</span>
        </div>
      `;
      secPeersContainer.appendChild(div);
    });
  } else {
    secPeersContainer.innerHTML = '<div style="color:var(--muted); font-size:11px;">No other peers in sector.</div>';
  }

  // Theme Card
  document.getElementById('dos_theme_name').textContent = thm.name || 'General';
  document.getElementById('dos_theme_archetype').textContent = thm.primary_archetype || 'Market Baseline';
  document.getElementById('dos_thm_1w').textContent = thm.w1_ret !== undefined && thm.w1_ret !== null ? `${thm.w1_ret > 0 ? '+' : ''}${thm.w1_ret.toFixed(1)}%` : '-';
  document.getElementById('dos_thm_1m').textContent = `${thm.m1_pctile.toFixed(0)}th %ile`;
  document.getElementById('dos_thm_3m').textContent = thm.m3_ret !== undefined && thm.m3_ret !== null ? `${thm.m3_ret > 0 ? '+' : ''}${thm.m3_ret.toFixed(1)}%` : '-';
  document.getElementById('dos_thm_score').textContent = thm.composite_score.toFixed(1);

  const thmBadge = document.getElementById('dos_thm_badge');
  thmBadge.textContent = thm.badge_type.toUpperCase();
  thmBadge.style.color = thm.badge_type === 'surge' || thm.badge_type === 'power' ? 'var(--green)' : 'var(--gold)';

  const thmPeersContainer = document.getElementById('dos_theme_peers');
  thmPeersContainer.innerHTML = '';
  if (thm.peers && thm.peers.length > 0) {
    thm.peers.forEach(p => {
      const div = document.createElement('div');
      div.className = 'peer-item';
      div.onclick = () => selectStock(p.symbol);
      div.innerHTML = `
        <div style="display:flex; align-items:center; gap:6px;">
          <strong style="color:#fff;">${p.symbol}</strong>
          <span style="color:var(--muted); font-size:11px; max-width:120px; overflow:hidden; text-overflow:ellipsis;">${p.name || ''}</span>
        </div>
        <div style="display:flex; align-items:center; gap:6px;">
          <span class="pill-rs-elite" style="font-size:10px; padding:1px 5px;">RS ${p.rs_score}</span>
          <span style="font-weight:600; color:#cbd5e1;">$${p.close.toFixed(2)}</span>
        </div>
      `;
      thmPeersContainer.appendChild(div);
    });
  } else {
    thmPeersContainer.innerHTML = '<div style="color:var(--muted); font-size:11px;">No other peers in theme.</div>';
  }

  // EP Catalyst Card
  const epCard = document.getElementById('dos_ep_card');
  const epContent = document.getElementById('dos_ep_content');
  if (data.eps && data.eps.length > 0) {
    epCard.style.display = 'flex';
    let epHtml = '';
    data.eps.slice(-3).reverse().forEach(ep => {
      epHtml += `
        <div style="background:#131a26; padding:6px; border-radius:4px; border-left:3px solid #10b981;">
          <div style="display:flex; justify-content:space-between; font-weight:700;">
            <span style="color:#34d399;">${ep.date}</span>
            <span>+${ep.gap_pct}% Gap (${ep.rvol}x Vol)</span>
          </div>
          <div style="color:var(--muted); font-size:11px; margin-top:2px;">Subtype: ${ep.subtype.toUpperCase()}</div>
        </div>
      `;
    });
    if (data.catalyst && data.catalyst.headline) {
      epHtml += `<div style="font-size:11px; color:#cbd5e1; margin-top:4px;"><strong>Catalyst:</strong> ${data.catalyst.headline}</div>`;
    }
    epContent.innerHTML = epHtml;
  } else {
    epCard.style.display = 'none';
  }
}

// Larsson Ribbon Render
function scheduleLarssonRender() {
  requestAnimationFrame(drawLarssonRibbon);
}

function drawLarssonRibbon() {
  const svg = document.getElementById('larsson_ribbon_svg');
  if (!svg || !chart || !window._currentLarssonData || !document.getElementById('cfg_larsson').checked) {
    if (svg) svg.innerHTML = '';
    return;
  }
  const timeScale = chart.timeScale();
  const data = window._currentLarssonData;
  if (!data || !data.length) return;

  const points = [];
  data.forEach(d => {
    const x = timeScale.timeToCoordinate(d.time);
    if (x === null) return;
    const yF = larssonFastSeries.priceToCoordinate(d.fast);
    const yS = larssonSlowSeries.priceToCoordinate(d.slow);
    if (yF === null || yS === null) return;
    points.push({ x, yF, yS, state: d.state });
  });

  if (points.length < 2) {
    svg.innerHTML = '';
    return;
  }

  let pathsHtml = '';
  for (let i = 0; i < points.length - 1; i++) {
    const p1 = points[i];
    const p2 = points[i + 1];
    let fill = 'rgba(100, 116, 139, 0.12)';
    if (p1.state === 'yellow') fill = 'rgba(234, 179, 8, 0.22)';
    else if (p1.state === 'blue') fill = 'rgba(56, 189, 248, 0.22)';

    const d = `M ${p1.x} ${p1.yF} L ${p2.x} ${p2.yF} L ${p2.x} ${p2.yS} L ${p1.x} ${p1.yS} Z`;
    pathsHtml += `<path d="${d}" fill="${fill}" />`;
  }
  svg.innerHTML = pathsHtml;
}

// Visibility toggles
function updateVisibility() {
  if (!chart) return;
  ema10Series.applyOptions({ visible: document.getElementById('cfg_ema10').checked });
  ema20Series.applyOptions({ visible: document.getElementById('cfg_ema20').checked });
  sma50Series.applyOptions({ visible: document.getElementById('cfg_sma50').checked });
  sma200Series.applyOptions({ visible: document.getElementById('cfg_sma200').checked });

  const showRS = document.getElementById('cfg_rs_line').checked;
  rsLineSeries.applyOptions({ visible: showRS });
  rsMaSeries.applyOptions({ visible: showRS });

  volumeSeries.applyOptions({ visible: document.getElementById('cfg_volume').checked });
  volMa20Series.applyOptions({ visible: document.getElementById('cfg_volume').checked });

  scheduleLarssonRender();
}

function onLarssonSpanChange() {
  if (currentStock) selectStock(currentStock.symbol);
}

function resetChart() {
  if (chart) chart.timeScale().fitContent();
}

function toggleSettingsPanel() {
  const p = document.getElementById('chart_settings_panel');
  p.style.display = p.style.display === 'none' ? 'flex' : 'none';
}

function toggleDrawToolbar() {
  const tb = document.getElementById('chart_draw_toolbar');
  tb.style.display = tb.style.display === 'none' ? 'flex' : 'none';
}

function toggleFullscreen() {
  const ws = document.getElementById('main_workspace');
  ws.classList.toggle('fullscreen-chart-mode');
  const btn = document.getElementById('btn_chart_fullscreen');
  btn.classList.toggle('active');
  btn.textContent = ws.classList.contains('fullscreen-chart-mode') ? '🗗 Standard' : '⛶ Enlarge';

  setTimeout(() => {
    const container = document.getElementById('chart_tv');
    chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
    resizeCanvas();
    chart.timeScale().fitContent();
    scheduleLarssonRender();
  }, 50);
}

// Navigation & Filters
function onFilterChange() {
  loadStocks();
}

function setRSChip(val) {
  selectedMinRS = val;
  document.querySelectorAll('.btn-filter-chip').forEach(c => c.classList.remove('active'));
  if (val === '') document.getElementById('chip_rs_all').classList.add('active');
  else if (val === '70') document.getElementById('chip_rs_70').classList.add('active');
  else if (val === '80') document.getElementById('chip_rs_80').classList.add('active');
  else if (val === '90') document.getElementById('chip_rs_90').classList.add('active');
  else if (val === '95') document.getElementById('chip_rs_95').classList.add('active');
  loadStocks();
}

function changeSort(col) {
  if (currentSortBy === col) {
    currentSortDir = (currentSortDir === 'desc') ? 'asc' : 'desc';
  } else {
    currentSortBy = col;
    currentSortDir = (col === 'symbol' || col === 'name' || col === 'sector' || col === 'theme') ? 'asc' : 'desc';
  }
  loadStocks();
}

function stepStock(delta) {
  if (!filteredStocks.length || !currentStock) return;
  const idx = filteredStocks.findIndex(s => s.symbol === currentStock.symbol);
  if (idx === -1) return;
  let nextIdx = idx + delta;
  if (nextIdx < 0) nextIdx = filteredStocks.length - 1;
  if (nextIdx >= filteredStocks.length) nextIdx = 0;
  selectStock(filteredStocks[nextIdx].symbol);
}

// Modal Chooser
function openStockPickerModal() {
  document.getElementById('stock_picker_modal').style.display = 'flex';
  document.getElementById('modal_search').value = '';
  renderModalStocks();
  document.getElementById('modal_search').focus();
}

function closeStockPickerModal(e) {
  if (e.target.id === 'stock_picker_modal') {
    document.getElementById('stock_picker_modal').style.display = 'none';
  }
}

function renderModalStocks() {
  const container = document.getElementById('modal_stock_list');
  const term = document.getElementById('modal_search').value.trim().toUpperCase();
  const list = filteredStocks.filter(s => !term || s.symbol.includes(term) || (s.name && s.name.toUpperCase().includes(term)));
  document.getElementById('modal_stock_count').textContent = list.length;

  container.innerHTML = '';
  list.slice(0, 100).forEach(s => {
    const div = document.createElement('div');
    div.className = 'peer-item';
    div.style.marginBottom = '4px';
    div.onclick = () => {
      selectStock(s.symbol);
      document.getElementById('stock_picker_modal').style.display = 'none';
    };
    div.innerHTML = `
      <div>
        <strong style="color:#fff; font-size:13px;">${s.symbol}</strong>
        <span style="color:var(--muted); font-size:11px; margin-left:6px;">${s.name || ''}</span>
      </div>
      <div style="display:flex; align-items:center; gap:8px;">
        <span class="pill-rs-elite">RS ${s.rs_score}</span>
        <span style="color:#cbd5e1; font-weight:700;">$${s.close.toFixed(2)}</span>
      </div>
    `;
    container.appendChild(div);
  });
}

// Drawing tool canvas
function resizeCanvas() {
  const canvas = document.getElementById('draw_canvas');
  if (!canvas) return;
  canvas.width = canvas.parentElement.clientWidth;
  canvas.height = canvas.parentElement.clientHeight;
  renderAllDrawings();
}

function setDrawTool(t) {
  drawTool = t;
  document.querySelectorAll('.draw-btn').forEach(b => b.classList.remove('active'));
  const btn = document.getElementById(`btn_tool_${t}`);
  if (btn) btn.classList.add('active');
  const canvas = document.getElementById('draw_canvas');
  canvas.style.pointerEvents = (t === 'cursor') ? 'none' : 'auto';
}

function onCanvasMouseDown(e) {
  if (drawTool === 'cursor') return;
  const rect = e.target.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  if (drawTool === 'hline') {
    const price = candleSeries.coordinateToPrice(y);
    drawings.push({ type: 'hline', price, color: '#38bdf8' });
    saveAnnotations();
    renderAllDrawings();
    setDrawTool('cursor');
    return;
  }

  isDrawing = true;
  currentDrawing = {
    type: drawTool,
    x1: x, y1: y,
    t1: chart.timeScale().coordinateToTime(x),
    p1: candleSeries.coordinateToPrice(y),
    x2: x, y2: y,
  };
}

function onCanvasMouseMove(e) {
  if (!isDrawing || !currentDrawing) return;
  const rect = e.target.getBoundingClientRect();
  currentDrawing.x2 = e.clientX - rect.left;
  currentDrawing.y2 = e.clientY - rect.top;
  renderAllDrawings();
}

window.addEventListener('mouseup', () => {
  if (isDrawing && currentDrawing) {
    isDrawing = false;
    currentDrawing.t2 = chart.timeScale().coordinateToTime(currentDrawing.x2);
    currentDrawing.p2 = candleSeries.coordinateToPrice(currentDrawing.y2);
    drawings.push(currentDrawing);
    currentDrawing = null;
    saveAnnotations();
    renderAllDrawings();
  }
});

function renderAllDrawings() {
  const canvas = document.getElementById('draw_canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const timeScale = chart ? chart.timeScale() : null;

  drawings.forEach(d => {
    if (d.type === 'hline') {
      const y = candleSeries.priceToCoordinate(d.price);
      if (y !== null) {
        ctx.strokeStyle = d.color || '#38bdf8';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(0, y); ctx.lineTo(canvas.width, y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#38bdf8';
        ctx.font = '10px monospace';
        ctx.fillText(`$${d.price.toFixed(2)}`, canvas.width - 55, y - 4);
      }
    } else if (d.type === 'line' || d.type === 'measure') {
      let x1 = d.x1, y1 = d.y1, x2 = d.x2, y2 = d.y2;
      if (timeScale && d.t1 && d.p1) {
        const cx1 = timeScale.timeToCoordinate(d.t1);
        const cy1 = candleSeries.priceToCoordinate(d.p1);
        if (cx1 !== null && cy1 !== null) { x1 = cx1; y1 = cy1; }
      }
      if (timeScale && d.t2 && d.p2) {
        const cx2 = timeScale.timeToCoordinate(d.t2);
        const cy2 = candleSeries.priceToCoordinate(d.p2);
        if (cx2 !== null && cy2 !== null) { x2 = cx2; y2 = cy2; }
      }

      ctx.strokeStyle = d.type === 'measure' ? '#f59e0b' : '#38bdf8';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
      ctx.stroke();

      if (d.type === 'measure' && d.p1 && d.p2) {
        const pct = ((d.p2 / d.p1 - 1.0) * 100);
        ctx.fillStyle = pct >= 0 ? '#34d399' : '#f87171';
        ctx.font = '11px monospace';
        ctx.fillText(`${pct > 0 ? '+' : ''}${pct.toFixed(2)}%`, x2 + 5, y2 - 5);
      }
    }
  });

  // Current drawing preview
  if (currentDrawing) {
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(currentDrawing.x1, currentDrawing.y1);
    ctx.lineTo(currentDrawing.x2, currentDrawing.y2);
    ctx.stroke();
  }
}

function clearDrawings() {
  drawings = [];
  saveAnnotations();
  renderAllDrawings();
}

function loadAnnotations(sym) {
  fetch(`/api/annotations?symbol=${encodeURIComponent(sym)}`)
    .then(r => r.json())
    .then(data => {
      drawings = data.drawings || [];
      renderAllDrawings();
    })
    .catch(() => { drawings = []; renderAllDrawings(); });
}

function saveAnnotations() {
  if (!currentStock) return;
  fetch('/api/annotations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbol: currentStock.symbol, drawings: drawings })
  });
}

function recomputeRS() {
  if (!confirm('Re-calculate Relative Strength scores across the universe?')) return;
  fetch('/api/recompute_rs', { method: 'POST' })
    .then(r => r.json())
    .then(() => {
      alert('Relative strength dataset recalculated successfully.');
      loadStocks();
    });
}

// Keyboard shortcuts
window.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
    e.preventDefault();
    stepStock(1);
  } else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
    e.preventDefault();
    stepStock(-1);
  } else if (e.key === '/' || (e.ctrlKey && e.key === 'k') || (e.metaKey && e.key === 'k')) {
    e.preventDefault();
    openStockPickerModal();
  } else if (e.key === 'Escape') {
    document.getElementById('stock_picker_modal').style.display = 'none';
    document.getElementById('chart_settings_panel').style.display = 'none';
    document.getElementById('chart_draw_toolbar').style.display = 'none';
  }
});

// Boot
window.addEventListener('DOMContentLoaded', () => {
  initChart();
  loadStocks();
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


def main():
    parser = argparse.ArgumentParser(description="Relative Strength (RS) Universe Explorer")
    parser.add_argument("--port", type=int, default=8786, help="Port to serve on (default: 8786)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--recompute", action="store_true", help="Force recomputation of RS universe dataset")
    args = parser.parse_args()

    if args.recompute:
        compute_universe_rs(force=True)

    print(f"Starting Relative Strength (RS) Universe Explorer on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
