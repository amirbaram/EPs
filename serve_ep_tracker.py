"""EP Live Tracker & Follow-Through Monitor (Port 8783).

Tracks current-market episodic pivots across their confirmation lifecycle:
- Today's Fresh EPs (Day 1 Ignition)
- Past Week EPs (Follow-Through Pipeline across 4 Confirmation Windows)
- Active PEAD Runners (Holding Day 1 Stop Loss)
- Emerging Sectors & Themes Momentum Radar (Dynamic 4-TF composite + 10-Yr EP Edge)
- Actionable Trade Management & Position Sizing Guidance (Progressive Exposure)
- Interactive Lightweight Chart with Entry, Stop, and Add Trigger lines

Run: .venv/bin/python serve_ep_tracker.py --port 8783
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, Response

import datastore
import indicators
import labels
import scanner_core
import setups
import universe
import thematic_engine
import ep_portfolio_manager as pm
import ep_premarket

app = Flask(__name__)

CACHE_PATH = Path("data/cache/tracker_eps.json")
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
CATALYST_PATH = Path("data/ep_catalysts.json")
FILINGS_CACHE_PATH = Path("data/cache/ep_sec_filings.json")

def load_ep_catalysts() -> dict:
    if CATALYST_PATH.exists():
        try:
            with open(CATALYST_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def load_sec_filings() -> dict:
    if FILINGS_CACHE_PATH.exists():
        try:
            with open(FILINGS_CACHE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

TOP_SECTORS = [
    "Computer Hardware", "Biotechnology", "Healthcare", "Technology", "Semiconductors",
    "Industrials", "Capital Markets", "Financial Services", "Specialty Industrial Machinery"
]
TOP_THEMES = [
    "Semiconductors", "Biotechnology", "Software - Infrastructure", "Computer Hardware",
    "Bitcoin Miners", "Quantum Computing", "Uranium & Nuclear", "Artificial Intelligence",
    "Communication Equipment", "Solar", "Technology"
]

TRACKER_DATA = []

def scan_recent_eps(days_back: int = 14) -> list[dict]:
    """Scans the active universe for qualifying Pinnacle Elite EPs within the last N sessions."""
    symbols = list(universe.active_symbols())
    results = []
    engine = thematic_engine.get_thematic_engine()
    
    # Load universe metadata for market cap ($B)
    mcap_map = {}
    try:
        uni_df = pd.read_csv("data/universe_active.csv").set_index("yf_symbol")
        if "mcap_m" in uni_df.columns:
            mcap_map = (uni_df["mcap_m"] / 1000.0).dropna().to_dict()
    except Exception:
        pass
    
    print(f"Scanning {len(symbols)} active universe symbols for recent Pinnacle EPs...")
    for sym in symbols:
        raw = datastore.load_bars(sym)
        if raw is None or len(raw) < 50:
            continue
        d = indicators.add_indicators(raw)
        d = scanner_core.calc_larssson_line(d)
        if d is None or len(d) < 30:
            continue
        n = len(d)
        
        # Precompute 60-day dollar volume median once per symbol
        med60 = (d["close"] * d["volume"]).rolling(60, min_periods=10).median().shift(1)
        
        # Scan window
        scan_len = min(days_back, n - 30)
        for i in range(n - scan_len, n):
            row = d.iloc[i]
            ev_dt = d.index[i].strftime("%Y-%m-%d")
            m60_val = med60.iloc[i] if pd.notna(med60.iloc[i]) else None
            
            gap = round(float(row["gap_pct"]), 1) if pd.notna(row["gap_pct"]) else 0.0
            rvol = round(float(row["rvol"]), 1) if pd.notna(row["rvol"]) else 0.0
            c_pos = round((float(row["close"]) - float(row["low"])) / max(0.001, float(row["high"]) - float(row["low"])), 2)
            dvol_m = round(float(row["close"]) * float(row["volume"]) / 1e6, 1)
            vol_shares = float(row["volume"])
            
            sub = setups._is_ep_event(row, m60_val)
            
            # Baseline eligibility for classical EP (requires c_pos >= 0.65)
            is_classical_ep = bool(sub) and ((gap >= 5.0 or (sub == "ep9m" and dvol_m >= 50.0)) and rvol >= 2.5 and c_pos >= 0.65 and dvol_m >= 15.0)
            
            # Pradeep Bonde Delayed Reaction EP (DRE):
            # Day 1 had high catalyst energy (Gap >= 5.0% or 9M volume) and RVOL >= 2.5x, $Vol >= $15M,
            # but closed weak (c_pos < 0.65 or red close). Forward bars must hold Day 1 Low!
            fwd = d.iloc[i:]
            bars_since = len(fwd) - 1
            held_d1_low = float(fwd["low"].min()) >= float(row["low"])
            is_dre = (not is_classical_ep) and (gap >= 5.0 or vol_shares >= 9_000_000) and rvol >= 2.5 and dvol_m >= 15.0 and (c_pos < 0.65) and held_d1_low
            
            if not (is_classical_ep or is_dre):
                continue
            
            if is_dre and not sub:
                sub = "dre"
            
            # Novel 9M Volume check: first time volume >= 9M in prior 60 sessions
            is_novel_9m = False
            if vol_shares >= 9_000_000:
                prior_60 = d.iloc[max(0, i - 60):i]
                if len(prior_60) > 0 and (prior_60["volume"] < 9_000_000).all():
                    is_novel_9m = True

            # Market cap in $B
            mcap_b = round(mcap_map.get(sym, 0.0), 2)
            is_cap10 = (0.0 < mcap_b <= 10.0)
            
            # Lazy thematic evaluation for confirmed EP candidates
            thm_eval = engine.evaluate_symbol(sym)
            theme_score = float(thm_eval.get("composite_score", 50.0))
            archetype = str(thm_eval.get("primary_archetype", "Neutral Flow"))
            is_veto = bool(thm_eval.get("is_veto", False))
            theme_badge = str(thm_eval.get("badge_type", "neutral"))
            sec_name = str(thm_eval.get("sector", "Unknown"))
            thm_name = str(thm_eval.get("theme", "General"))
            is_tailwind = (theme_score >= 65.0) and not is_veto

            sec_info = thm_eval.get("sector_info", {})
            thm_info = thm_eval.get("theme_info", {})
            sec_pctiles = {"1w": sec_info.get("w1_pct", 50), "1m": sec_info.get("m1_pct", 50), "3m": sec_info.get("m3_pct", 50), "ytd": sec_info.get("ytd_pct", 50)}
            thm_pctiles = {"1w": thm_info.get("w1_pct", 50), "1m": thm_info.get("m1_pct", 50), "3m": thm_info.get("m3_pct", 50), "ytd": thm_info.get("ytd_pct", 50)}

            if is_veto:
                diagnosis = f"Severe Headwind: {thm_name} is in bottom 35th percentile. Historical trap rate is 37.5%."
            elif is_tailwind:
                diagnosis = f"Institutional Tailwind: {thm_name} exhibits top-tier momentum ({archetype}, Score: {theme_score}/100)."
            else:
                diagnosis = f"Neutral market-in-line momentum flow ({thm_name}). Standard baseline expectancy."
            
            # Follow-through tracking
            curr_bar = fwd.iloc[-1]
            curr_px = float(curr_bar["close"])
            d1_close = float(row["close"])
            d1_low = float(row["low"])
            d1_high = float(row["high"])
            d1_open = float(row["open"])
            
            ret_pct = round((curr_px / d1_close - 1.0) * 100.0, 1)
            risk_pct = round((d1_close - d1_low) / d1_close * 100.0, 1)
            curr_r = round(ret_pct / risk_pct, 2) if risk_pct > 0 else 0.0
            
            # 48h absorption gate: evaluated at end of 48-hour window (Day 3 close) or current price
            d1_body = d1_close - d1_open
            half_body = d1_close - 0.5 * max(0.01, d1_body)
            
            if bars_since >= 2:
                d3_close = float(fwd.iloc[2]["close"])
                held_48h = (d3_close >= half_body) or (curr_px >= d1_close)
            elif bars_since == 1:
                held_48h = (float(fwd.iloc[1]["close"]) >= half_body) or (curr_px >= d1_close)
            else:
                held_48h = True
            
            d2_add = float(fwd.iloc[1]["high"]) > d1_high if bars_since >= 1 else False
            
            # Determine lifecycle status & suggested trade action
            if not held_d1_low:
                status = "Stopped Out / Invalidation"
                stage_badge = "badge-stopped"
                action_headline = "DEFENSIVE STOP TRIGGERED"
                action_plan = f"Price breached Day 1 low stop (${d1_low:.2f}). Position closed for strict risk control (-1.0 R). Add ticker to watchlist for Trade 2 Yellow Re-Entry."
            elif is_dre:
                dre_risk_pct = round((d1_high - d1_low) / d1_high * 100.0, 1)
                if curr_px >= d1_high:
                    status = "Delayed Reaction Breakout (DRE)"
                    stage_badge = "badge-dre-triggered"
                    action_headline = "⚡ DRE TRIGGER CONFIRMED: BREAKOUT ABOVE DAY 1 HIGH"
                    action_plan = f"Broke above Day 1 High (${d1_high:.2f}) after messy Day 1 digestion! Bonde Delayed Reaction Entry confirmed.\n• Stop Loss: Day 1 Low (${d1_low:.2f}) or recent tight shelf (${float(fwd['low'].min()):.2f}).\n• Quantitative Edge: 10-year verified real-bar backtest delivers +0.86 R (Day 2) to +0.71 R (Day 3) EV, 2.20 PF, 7.4% stop distance, avoiding 62% of toxic fade traps."
                else:
                    status = "Delayed Reaction Watchlist (DRE)"
                    stage_badge = "badge-dre-watch"
                    action_headline = "⚡ DRE WATCH: DIGESTING ABOVE DAY 1 LOW"
                    action_plan = f"Stock had massive volume/gap on Day 1 but closed weak (ClosePos {c_pos:.2f}). Holding Day 1 Low (${d1_low:.2f}).\n• Action: Place stop-buy order at ${d1_high:.2f} (Day 1 High).\n• Risk: Hard stop at ${d1_low:.2f} (-{dre_risk_pct:.1f}% risk). Wait for breakout trigger."
            elif curr_px >= d1_high:
                if bars_since == 1:
                    status = "Day 2 Follow-Through (+Add Active)"
                    stage_badge = "badge-add"
                    action_headline = "SECONDARY ADD CONFIRMED"
                    action_plan = f"Crossed Day 1 High (${d1_high:.2f}). Secondary add triggered! Scale position to 1.5x heat. Keep stop locked at Day 1 Low (${d1_low:.2f})."
                else:
                    status = f"PEAD Trend Active (Day {bars_since + 1})"
                    stage_badge = "badge-runner"
                    action_headline = "BREAKOUT AT NEW HIGHS — PEAD RUNNER"
                    action_plan = f"Trading above Day 1 High (${d1_high:.2f}) at ${curr_px:.2f} (+{curr_r:.2f} R). Base + Secondary Add in profit. Trail stop along 20 EMA."
            elif bars_since >= 2 and not held_48h and curr_px < d1_close:
                status = "48H Absorption Violated"
                stage_badge = "badge-violated"
                action_headline = "HIGH RISK OF TRAP — DEFENSE ACTIVE"
                action_plan = f"Failed to hold upper 50% body on Day 3 (${half_body:.2f}). Cancel secondary adds. Keep hard stop at ${d1_low:.2f}. Prepare to take scratch/breakeven."
            elif bars_since == 0:
                status = "Day 1 Ignition (Fresh Today)"
                stage_badge = "badge-fresh"
                action_headline = "ACTIVE DAY 1 BUY SIGNAL"
                action_plan = f"Enter on close or Day 2 open. Hard Stop: ${d1_low:.2f} (-{risk_pct:.1f}% risk). Secondary Add: Prepare to add 50% if price crosses ${d1_high:.2f}."
            elif bars_since == 1:
                if d2_add:
                    status = "Day 2 Follow-Through (+Add Active)"
                    stage_badge = "badge-add"
                    action_headline = "SECONDARY ADD CONFIRMED"
                    action_plan = f"Crossed Day 1 High (${d1_high:.2f}). Secondary add triggered! Scale position to 1.5x heat. Keep stop locked at Day 1 Low (${d1_low:.2f})."
                else:
                    status = "Day 2 Inside Range (Digesting)"
                    stage_badge = "badge-digest"
                    action_headline = "HOLDING DAY 1 POSITION"
                    action_plan = f"Price digesting inside Day 1 range. Hold base position with stop at ${d1_low:.2f}. Add arms if price breaks above ${d1_high:.2f}."
            elif bars_since in [2, 3]:
                status = "48H Body Absorbed (Strong Support)"
                stage_badge = "badge-absorbed"
                action_headline = "48H GATE CONFIRMED — HOLD FOR PEAD"
                action_plan = f"Held upper 50% body through Day 3 (${half_body:.2f}). Institutional absorption confirmed. Hold for multi-quarter PEAD drift. Trail out on first Larsson Blue Flip."
            elif curr_px >= d1_close:
                status = f"PEAD Trend Active (Day {bars_since + 1})"
                stage_badge = "badge-runner"
                action_headline = "PEAD RUNNER IN PROGRESS"
                action_plan = f"Holding in profit (+{curr_r:.2f} R). Trail stop along dynamic 20 EMA / swing lows. Exit on first daily Larsson Line Blue Flip."
            else:
                status = f"Consolidating Above Stop (Day {bars_since + 1})"
                stage_badge = "badge-digest"
                action_headline = "BASE CONSOLIDATION ACTIVE"
                action_plan = f"Testing post-breakout base ({curr_r:.2f} R). Hard stop intact at ${d1_low:.2f}. Watch for 20 EMA support and secondary pivot."

            # 50-Day SMA Institutional Baseline
            sma50_val = float(d["sma50"].iloc[-1]) if "sma50" in d.columns and pd.notna(d["sma50"].iloc[-1]) else None
            above_sma50 = bool(curr_px >= sma50_val) if sma50_val is not None else True

            # Prior 60-bar drift before EP for Turnaround classification
            prior_close_pct = 0.0
            if i >= 60:
                prior_px = float(d["close"].iloc[i-60])
                ep_base_px = float(d["close"].iloc[i-1]) if i > 0 else prior_px
                prior_close_pct = (ep_base_px / prior_px - 1.0) * 100.0 if prior_px > 0 else 0.0

            # The 6 Institutional Trader Archetypes (+ Idiosyncratic Alpha + DRE + Headwind):
            is_idiosyncratic = bool(is_veto and (rvol >= 6.0 and dvol_m >= 75.0 and c_pos >= 0.80 and held_48h))
            is_severe_headwind = bool(is_veto and not is_idiosyncratic)
            is_pinnacle_elite = bool(is_classical_ep and held_48h and c_pos >= 0.65 and rvol >= 2.5 and dvol_m >= 15.0 and is_tailwind and above_sma50)
            is_compounder = bool((is_cap10 or thm_name in ["Semiconductors", "Artificial Intelligence", "Biotechnology", "Computer Hardware", "Uranium & Nuclear", "Quantum Computing", "Software - Infrastructure"]) and held_48h and above_sma50)
            is_sweet_spot = bool(rvol >= 4.5 and gap >= 6.0 and c_pos >= 0.65 and held_48h)
            is_emerging_leader = bool((sec_pctiles.get("1m", 50) >= 65 or thm_pctiles.get("1m", 50) >= 65 or theme_score >= 65) and rvol >= 2.5 and held_48h)
            is_turnaround = bool((prior_close_pct <= -15.0 or not above_sma50) and rvol >= 3.5 and c_pos >= 0.70)
            is_high_beta = bool(thm_name in ["Bitcoin Miners", "Quantum Computing", "Solar", "Communication Equipment"] or (gap >= 12.0 and rvol >= 3.5))

            if is_idiosyncratic:
                trader_archetype = "🎯 Idiosyncratic Alpha"
                archetype_key = "idiosyncratic"
                archetype_badge = "chip-tailwind"
                archetype_sizing = "0.50 R (Strict Half-Heat)"
                archetype_plan_note = "Blowout catalyst turnover overrides sector lag. Max heat 0.50R, Day 2 breakout adds prohibited. Stop locked at Day 1 Low."
            elif is_severe_headwind:
                trader_archetype = "🚫 Severe Headwind (Trap Avoid)"
                archetype_key = "headwind"
                archetype_badge = "chip-veto"
                archetype_sizing = "0.0 R (DO NOT TRADE / AVOID)"
                archetype_plan_note = "Bottom 35th percentile sector headwind. 37.5% empirical trap rate. Institutions use gap liquidity to sell. Preserve capital."
            elif is_dre:
                trader_archetype = "⚡ Delayed Reaction (DRE)"
                archetype_key = "dre"
                archetype_badge = "chip-gold"
                archetype_sizing = "1.0 R on D1 High Breakout"
                archetype_plan_note = "Pradeep Bonde DRE. Messy Day 1 close digested; enter on breakout above Day 1 High stop-buy. Stop at Day 1 Low."
            elif is_pinnacle_elite:
                trader_archetype = "💎 Pinnacle Elite"
                archetype_key = "pinnacle"
                archetype_badge = "chip-tailwind"
                archetype_sizing = "1.0 R Initial → 1.5 R on Day 5 Add"
                archetype_plan_note = "Apex Institutional Quality. 35.2% Win Rate, +0.74R Baseline EV (+0.90R Pyramiding), PF 2.15. Zero-lookahead audited edge."
            elif is_compounder:
                trader_archetype = "🚀 Multi-Quarter Compounder"
                archetype_key = "compounder"
                archetype_badge = "chip-tailwind"
                archetype_sizing = "1.0 R to 1.25 R"
                archetype_plan_note = "Secular institutional accumulation. Trail exclusively with 50 SMA baseline. Disable early climax exits."
            elif is_sweet_spot:
                trader_archetype = "⭐ Institutional Sweet Spot"
                archetype_key = "sweet"
                archetype_badge = "chip-gold"
                archetype_sizing = "1.25 R to 1.50 R (Size Boost)"
                archetype_plan_note = "RVOL sweet spot (≥5x) cuts max drawdown in half (-6.0R) and accelerates EV to +1.68R+."
            elif is_emerging_leader:
                trader_archetype = "🌟 Emerging Leader"
                archetype_key = "emerging"
                archetype_badge = "chip-purple"
                archetype_sizing = "1.0 R (Velocity Runner)"
                archetype_plan_note = "Top 35% relative group momentum. Fast velocity; lock breakeven at +2.0R, scale 1/3 into climax."
            elif is_turnaround:
                trader_archetype = "🔄 Neglected Turnaround"
                archetype_key = "turnaround"
                archetype_badge = "chip-red"
                archetype_sizing = "0.50 R to 0.75 R"
                archetype_plan_note = "Deep value accumulation off lows. Trapped overhead supply exists; take 50% profit at +3.0R resistance."
            elif is_high_beta:
                trader_archetype = "⚡ High-Beta Momentum"
                archetype_key = "high_beta"
                archetype_badge = "chip-cyan"
                archetype_sizing = "0.50 R (High Volatility)"
                archetype_plan_note = "High volatility runner. Use tight trailing stop (10 EMA or ML MAE). Never let +2.0R turn into a loss."
            else:
                trader_archetype = "📊 Quality Swing Runner"
                archetype_key = "swing"
                archetype_badge = "chip-neutral"
                archetype_sizing = "1.0 R Baseline"
                archetype_plan_note = "Standard EP baseline. Respect Day 1 low stop loss and trail along 20 EMA."

            # Archetype guidance injection into action plan
            archetype_header = f"【ARCHETYPE: {trader_archetype.upper()}】\n• Suggested Sizing: {archetype_sizing}\n• Playbook Mandate: {archetype_plan_note}\n\n"
            action_plan = archetype_header + action_plan

            if is_idiosyncratic:
                is_veto = False
                theme_badge = "idiosyncratic"
                action_headline = f"🎯 IDIOSYNCRATIC ALPHA: {action_headline}"
            elif is_severe_headwind:
                action_headline = f"⚠️ SEVERE HEADWIND: AVOID / PRESERVE CAPITAL"
            elif is_tailwind:
                action_headline = f"🚀 TAILWIND ACTIVE: {action_headline}"

            # 50 SMA baseline notes
            if sma50_val is not None:
                if above_sma50:
                    action_plan += f"\n\n🏛️ 50-DAY SMA BASELINE: Price is holding above institutional baseline (${sma50_val:.2f}). 10-year data shows 74.1% win rate and 23.28 PF when holding 50 SMA."
                else:
                    action_plan += f"\n\n⚠️ BELOW 50-DAY SMA BASELINE: Price breached baseline (${sma50_val:.2f}). Trapped overhead supply active; tighten trailing stop."

            if rvol >= 5.0:
                action_plan += f"\n\n🔥 SWEET SPOT VOLUME ACCELERATION: RVOL {rvol:.1f}x meets institutional sweet spot. Empirical drawdown is cut in half (-6.0 R). Sizing: Eligible for 1.25 R to 1.50 R heat."

            if is_novel_9m:
                action_plan += f"\n\n🔥 NOVEL 9M VOLUME DETECTED: First 9M+ share day in >60 sessions. 10-year study delivers +1.92 R EV (+102% edge boost)."

            # Trade 2 & 3 Yellow Re-Entry evaluation
            t2_has_reentry = False
            t2_in_pullback = False
            t2_ml_vetoed = False
            t2_prob_reentry = None
            t2_cons_days = 0
            t2_drop_from_peak = 0.0
            t2_entry_date = None
            t2_entry_price = None
            t2_stop_price = None
            t2_risk_pct = None
            t2_curr_r = None
            t2_status_desc = "No Re-Entry Active"

            # Intermediate ribbon (8, 12, 16, 21) for continuation setup detection
            e8 = fwd["close"].ewm(span=8, adjust=False).mean()
            e12 = fwd["close"].ewm(span=12, adjust=False).mean()
            e16 = fwd["close"].ewm(span=16, adjust=False).mean()
            e21 = fwd["close"].ewm(span=21, adjust=False).mean()
            ribbon_bullish = (e8 >= e12) & (e12 >= e16) & (e16 >= e21)
            ribbon_bearish = (e8 < e12) & (e12 < e16) & (e16 < e21)
            ribbon_state = pd.Series("gray", index=fwd.index)
            ribbon_state.loc[ribbon_bullish] = "yellow"
            ribbon_state.loc[ribbon_bearish] = "blue"
            fwd["ribbon_state"] = ribbon_state

            t1_peak = float(fwd["high"].max())
            seen_cons = False
            cons_days = 0
            reentry_bar = None
            t2_track_name = "base_ribbon"

            # Track A: Early High Tight Flag (HTF) Check (Days 3 to 25)
            htf_signal = None
            htf_stop_px = None
            pole_h = 0.0
            pole_b = None
            for b in range(1, min(len(fwd), 16)):
                h = float(fwd["high"].iloc[b])
                if (h / d1_close - 1.0) >= 0.12 and h > pole_h:
                    pole_h = h
                    pole_b = b
            if pole_b is not None:
                for b in range(pole_b + 1, min(len(fwd), pole_b + 16)):
                    l = float(fwd["low"].iloc[b])
                    c = float(fwd["close"].iloc[b])
                    pullback = (pole_h - l) / pole_h * 100.0 if pole_h > 0 else 0.0
                    if pullback > 18.0:
                        break
                    prev_h = float(fwd["high"].iloc[max(0, b - 4):b].max())
                    if c > prev_h:
                        htf_signal = b
                        htf_stop_px = round(float(fwd["low"].iloc[max(pole_b, b - 4):b + 1].min()), 2)
                        break

            if htf_signal is not None:
                t2_has_reentry = True
                t2_track_name = "htf_continuation"
                t2_entry_date = fwd.index[htf_signal].strftime("%Y-%m-%d")
                t2_entry_price = round(float(fwd["close"].iloc[htf_signal]), 2)
                t2_stop_price = htf_stop_px
                t2_risk_pct = round((t2_entry_price - t2_stop_price) / t2_entry_price * 100.0, 1) if t2_entry_price > 0 else 5.0
                t2_cons_days = int(htf_signal - pole_b)
                t2_drop_from_peak = round((pole_h - t2_stop_price) / pole_h * 100.0, 1)
                if t2_risk_pct > 0:
                    t2_curr_r = round((curr_px - t2_entry_price) / (t2_entry_price - t2_stop_price), 2)
                t2_status_desc = f"Active HTF Flag Breakout (+{t2_curr_r:.2f} R)"
            else:
                # Track B: Secondary Base Ribbon Re-Entry
                cons_low = float(fwd["low"].iloc[0])
                for b in range(1, len(fwd)):
                    l_b = float(fwd["low"].iloc[b])
                    s_b = fwd["ribbon_state"].iloc[b]
                    if l_b < cons_low:
                        cons_low = l_b
                    if pd.notna(s_b) and s_b in ["blue", "gray"]:
                        seen_cons = True
                        cons_days += 1
                    if seen_cons and b >= 5 and pd.notna(s_b) and s_b == "yellow" and reentry_bar is None:
                        reentry_bar = b
                        break

                if reentry_bar is not None and cons_days <= 45:
                    drop_from_peak = round((t1_peak - cons_low) / t1_peak * 100.0, 1) if t1_peak > 0 else 0.0
                    if drop_from_peak <= 45.0:
                        t2_has_reentry = True
                        t2_entry_date = fwd.index[reentry_bar].strftime("%Y-%m-%d")
                        t2_entry_price = round(float(fwd["close"].iloc[reentry_bar]), 2)
                        swing5_low = round(float(fwd["low"].iloc[max(0, reentry_bar - 5):reentry_bar + 1].min()), 2)
                        t2_stop_price = swing5_low
                        t2_risk_pct = round((t2_entry_price - t2_stop_price) / t2_entry_price * 100.0, 1) if t2_entry_price > 0 else 5.0
                        t2_cons_days = cons_days
                        t2_drop_from_peak = drop_from_peak
                        if t2_risk_pct > 0:
                            t2_curr_r = round((curr_px - t2_entry_price) / (t2_entry_price - t2_stop_price), 2)
                        
                        try:
                            from ep_ml_engine import engine as ml_engine
                            re_feats = ml_engine.compute_rolling_features(sym, i, i + reentry_bar)
                            if re_feats is not None:
                                re_preds = ml_engine.predict_rolling(re_feats)
                                t2_prob_reentry = round(float(re_preds.get("prob_reentry", 0.0)), 3)
                                if t2_prob_reentry < 0.35:
                                    t2_ml_vetoed = True
                        except Exception:
                            pass

                        if t2_ml_vetoed:
                            t2_status_desc = f"ML Vetoed ({t2_prob_reentry*100:.1f}% < 35% threshold)"
                        else:
                            t2_status_desc = f"Active Base Ribbon Re-Entry (+{t2_curr_r:.2f} R)"
                elif seen_cons and reentry_bar is None:
                    drop_from_peak = round((t1_peak - cons_low) / t1_peak * 100.0, 1) if t1_peak > 0 else 0.0
                    swing5_low = round(float(fwd["low"].iloc[max(0, len(fwd) - 5):].min()), 2)
                    t2_in_pullback = True
                    t2_cons_days = cons_days
                    t2_drop_from_peak = drop_from_peak
                    t2_stop_price = swing5_low
                    t2_status_desc = f"Pullback in Progress (Day {cons_days}, Retraced {drop_from_peak}%, Shelf ${swing5_low:.2f})"

            t2_info = {
                "has_reentry": t2_has_reentry,
                "in_pullback": t2_in_pullback,
                "ml_vetoed": t2_ml_vetoed,
                "prob_reentry": t2_prob_reentry,
                "cons_days": t2_cons_days,
                "drop_from_peak": t2_drop_from_peak,
                "entry_date": t2_entry_date,
                "entry_price": t2_entry_price,
                "stop_price": t2_stop_price,
                "risk_pct": t2_risk_pct,
                "curr_r": t2_curr_r,
                "status_desc": t2_status_desc
            }

            results.append({
                "symbol": sym,
                "event_date": ev_dt,
                "subtype": sub,
                "gap_pct": gap,
                "rvol": rvol,
                "close_pos": c_pos,
                "dvol_m": dvol_m,
                "entry_price": d1_close,
                "stop_price": d1_low,
                "add_price": d1_high,
                "curr_price": curr_px,
                "risk_pct": risk_pct,
                "ret_pct": ret_pct,
                "curr_r": curr_r,
                "bars_since": bars_since,
                "status": status,
                "stage_badge": stage_badge,
                "action_headline": action_headline,
                "action_plan": action_plan,
                "held_d1_low": held_d1_low,
                "held_48h": held_48h,
                "d2_add": d2_add,
                "is_dre": is_dre,
                "is_novel_9m": is_novel_9m,
                "mcap_b": mcap_b,
                "is_cap10": is_cap10,
                "sector": sec_name,
                "theme": thm_name,
                "theme_score": theme_score,
                "theme_archetype": archetype,
                "theme_badge": theme_badge,
                "trader_archetype": trader_archetype,
                "archetype_key": archetype_key,
                "archetype_badge": archetype_badge,
                "archetype_sizing": archetype_sizing,
                "archetype_plan_note": archetype_plan_note,
                "t2_info": t2_info,
                "is_thematic_veto": is_veto,
                "is_thematic_tailwind": is_tailwind,
                "is_idiosyncratic": is_idiosyncratic,
                "above_sma50": above_sma50,
                "sma50_val": round(sma50_val, 2) if sma50_val is not None else None,
                "theme_diagnosis": diagnosis,
                "sec_pctiles": sec_pctiles,
                "thm_pctiles": thm_pctiles
            })
            
    # Sort descending by event date, then dollar volume
    results.sort(key=lambda x: (x["event_date"], x["dvol_m"]), reverse=True)
    print(f"Scan complete. Found {len(results)} Pinnacle EPs.")
    return results

def get_tracker_data(force_rescan: bool = False):
    global TRACKER_DATA
    if force_rescan or not TRACKER_DATA:
        if CACHE_PATH.exists() and not force_rescan:
            try:
                with open(CACHE_PATH, "r") as f:
                    TRACKER_DATA = json.load(f)
                if TRACKER_DATA and ("trader_archetype" not in TRACKER_DATA[0] or "t2_info" not in TRACKER_DATA[0]):
                    print("Cache missing trader_archetype or t2_info - forcing auto-rescan...")
                    TRACKER_DATA = []
                else:
                    print(f"Loaded {len(TRACKER_DATA)} cached tracker events.")
            except Exception as e:
                print(f"Failed to load cache: {e}")
        if not TRACKER_DATA or force_rescan:
            TRACKER_DATA = scan_recent_eps(days_back=14)

        # Enrich with catalysts and SEC filings
        catalysts = load_ep_catalysts()
        sec_filings = load_sec_filings()
        for ev in TRACKER_DATA:
            sym = ev["symbol"]
            cat = catalysts.get(sym, {})
            ev["catalyst_headline"] = cat.get("headline", f"{sym} EP Volume & Price Repricing")
            ev["catalyst_category"] = cat.get("category", f"{ev.get('sector', 'Corporate')} Inflection")
            ev["catalyst_tier"] = cat.get("quality_tier", "Tier-1 Institutional Repricing")
            ev["catalyst_analysis"] = cat.get("analysis", f"{sym} demonstrated massive institutional volume turnover (${ev.get('dvol_m', 0):.1f}M, RVOL {ev.get('rvol', 0):.1f}x) triggering structural repricing.")
            ev["catalyst_news"] = cat.get("news_summary", f"{sym} news and corporate announcements surrounding {ev.get('event_date')}.")
            ev["sec_filings"] = sec_filings.get(sym, [])

        with open(CACHE_PATH, "w") as f:
            json.dump(TRACKER_DATA, f, indent=2)
    return TRACKER_DATA

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Episodic Pivot (EP) Live Tracker & Follow-Through Monitor</title>
<script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root {
    --bg: #0b0e14; --surface: #151922; --border: #232936;
    --text: #e2e8f0; --muted: #7b849b; --accent: #3b82f6;
    --green: #10b981; --red: #ef4444; --gold: #f59e0b; --purple: #8b5cf6; --cyan: #38bdf8;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
    font-size: 13px; height: 100vh; display: flex; flex-direction: column; overflow: hidden;
  }
  header {
    background: var(--surface); border-bottom: 1px solid var(--border);
    padding: 10px 18px; display: flex; justify-content: space-between; align-items: center; flex-shrink: 0;
  }
  .title { font-size: 15px; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 8px; }
  .badge { background: #0284c7; color: #fff; padding: 2px 7px; border-radius: 4px; font-size: 11px; font-weight: 600; }
  
  .btn-emerging {
    background: #2e1065; border: 1px solid #7c3aed; color: #c4b5fd; padding: 5px 14px; border-radius: 5px;
    font-size: 12px; font-weight: 700; cursor: pointer; display: flex; align-items: center; gap: 6px; transition: all 0.15s;
  }
  .btn-emerging:hover { background: #4c1d95; color: #fff; box-shadow: 0 0 10px rgba(124,58,237,0.4); }

  .btn-export {
    background: #0f2338; border: 1px solid #0284c7; color: #38bdf8; padding: 5px 12px; border-radius: 5px;
    font-size: 11.5px; font-weight: 700; cursor: pointer; display: flex; align-items: center; gap: 5px; transition: all 0.15s;
  }
  .btn-export:hover { background: #0369a1; color: #fff; box-shadow: 0 0 10px rgba(2,132,199,0.4); }

  .btn-refresh {
    background: #1e293b; border: 1px solid #334155; color: #cbd5e1; padding: 5px 12px; border-radius: 5px;
    font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.15s;
  }
  .btn-refresh:hover { background: #334155; color: #fff; }

  /* View Filter Bar */
  .view-bar {
    background: #0f141f; border-bottom: 1px solid var(--border); padding: 8px 16px;
    display: flex; gap: 8px; align-items: center; flex-shrink: 0;
  }
  .view-tab {
    background: #182030; border: 1px solid #28354d; color: #94a3b8; padding: 5px 14px;
    border-radius: 5px; font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.15s;
  }
  .view-tab:hover { background: #222d42; color: #fff; }
  .view-tab.active { background: var(--accent); border-color: var(--accent); color: #fff; }

  /* Expectancy Booster Bar */
  .booster-bar {
    background: #0c101a; border-bottom: 1px solid var(--border);
    padding: 6px 16px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; flex-shrink: 0;
  }
  .booster-label { font-size: 11px; font-weight: 700; color: var(--cyan); text-transform: uppercase; margin-right: 4px; display: flex; align-items: center; gap: 4px; }
  .booster-chip {
    background: #151b29; color: #94a3b8; border: 1px solid #242e42;
    padding: 4px 10px; border-radius: 14px; font-size: 11px; font-weight: 600; cursor: pointer;
    display: flex; align-items: center; gap: 6px; transition: all 0.15s ease; user-select: none;
  }
  .booster-chip:hover { background: #1c2538; color: #e2e8f0; border-color: #384666; }
  .booster-chip.active {
    background: rgba(16, 185, 129, 0.18); color: #34d399; border-color: #059669; box-shadow: 0 0 8px rgba(5,150,105,0.3);
  }
  .booster-pill {
    font-size: 10px; font-weight: 700; padding: 1px 6px; border-radius: 4px;
    background: rgba(0,0,0,0.35); color: #fbbf24;
  }
  .booster-reset {
    background: none; border: 1px dashed #475569; color: #94a3b8; padding: 3px 8px;
    border-radius: 12px; font-size: 10px; cursor: pointer; margin-left: 4px;
  }
  .booster-reset:hover { color: #fff; border-color: #94a3b8; background: #1e293b; }
  .btn-guide-toggle {
    background: #1e1b4b; border: 1px solid #4338ca; color: #c7d2fe; padding: 4px 11px;
    border-radius: 12px; font-size: 10.5px; font-weight: 700; cursor: pointer; margin-left: auto;
    display: flex; align-items: center; gap: 5px; transition: all 0.15s;
  }
  .btn-guide-toggle:hover { background: #312e81; color: #fff; box-shadow: 0 0 8px rgba(67,56,202,0.4); }

  /* Impact Diagnostics Collapsible Drawer / Panel */
  .impact-guide-panel {
    display: none; background: #0a0d14; border-bottom: 2px solid #4338ca; padding: 14px 18px;
    flex-shrink: 0; font-size: 11.5px; color: #cbd5e1; line-height: 1.5;
  }
  .impact-guide-panel.open { display: block; }

  /* Active Ticker Full-Width Context Bar */
  .active-ticker-bar {
    background: #0d121c; border-bottom: 1px solid var(--border);
    padding: 6px 16px; display: flex; justify-content: space-between; align-items: center; gap: 14px;
    flex-shrink: 0; min-height: 38px;
  }
  .active-ticker-left {
    display: flex; align-items: center; gap: 12px; min-width: 0; flex-shrink: 1; overflow: hidden;
  }
  .active-ticker-sym {
    font-size: 14px; font-weight: 800; color: #fff; letter-spacing: 0.5px;
    background: #182235; border: 1px solid #29384f; padding: 2px 8px; border-radius: 4px;
    display: flex; align-items: center; gap: 6px; flex-shrink: 0;
  }
  .active-ticker-details {
    font-size: 11.5px; color: #cbd5e1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .active-ticker-badges {
    display: flex; align-items: center; gap: 6px; flex-shrink: 0; overflow-x: auto; scrollbar-width: none;
  }
  .active-ticker-badges::-webkit-scrollbar { display: none; }

  /* Main Workspace */
  .workspace { display: flex; flex: 1; overflow: hidden; }

  /* Left Table Pane */
  .table-pane { width: 45%; border-right: 1px solid var(--border); overflow-y: auto; background: var(--surface); }
  table { width: 100%; border-collapse: collapse; font-size: 11.5px; }
  th {
    background: #11151f; color: var(--muted); text-align: right; padding: 8px 10px;
    font-weight: 600; font-size: 10.5px; position: sticky; top: 0; z-index: 10;
    border-bottom: 1px solid var(--border);
  }
  th.tl { text-align: left; }
  td { padding: 8px 10px; border-bottom: 1px solid #1a202c; text-align: right; }
  td.tl { text-align: left; }
  tr:hover { background: #1a2233; cursor: pointer; }
  tr.selected { background: #1e293b !important; border-left: 3px solid var(--accent); }

  /* Stage Badges */
  .badge-fresh { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); padding: 2px 7px; border-radius: 12px; font-weight: 700; font-size: 10px; white-space: nowrap; }
  .badge-add { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); padding: 2px 7px; border-radius: 12px; font-weight: 700; font-size: 10px; white-space: nowrap; }
  .badge-digest { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); padding: 2px 7px; border-radius: 12px; font-weight: 600; font-size: 10px; white-space: nowrap; }
  .badge-absorbed { background: rgba(16, 185, 129, 0.12); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.25); padding: 2px 7px; border-radius: 12px; font-weight: 600; font-size: 10px; white-space: nowrap; }
  .badge-runner { background: rgba(139, 92, 246, 0.15); color: #c084fc; border: 1px solid rgba(139, 92, 246, 0.3); padding: 2px 7px; border-radius: 12px; font-weight: 700; font-size: 10px; white-space: nowrap; }
  .badge-violated { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); padding: 2px 7px; border-radius: 12px; font-weight: 600; font-size: 10px; white-space: nowrap; }
  .badge-stopped { background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.35); padding: 2px 7px; border-radius: 12px; font-weight: 700; font-size: 10px; white-space: nowrap; }
  .badge-dre-triggered { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #d97706; padding: 2px 7px; border-radius: 12px; font-weight: 700; font-size: 10px; white-space: nowrap; }
  .badge-dre-watch { background: rgba(139, 92, 246, 0.15); color: #c4b5fd; border: 1px solid #7c3aed; padding: 2px 7px; border-radius: 12px; font-weight: 600; font-size: 10px; white-space: nowrap; }

  /* Theme Chips & Micro-Pills */
  .score-chip {
    display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 12px; font-size: 10px; font-weight: 600; white-space: nowrap; flex-shrink: 0; line-height: 1.2;
  }
  .chip-tailwind { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
  .chip-neutral { background: rgba(148, 163, 184, 0.12); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.25); }
  .chip-veto { background: rgba(239, 68, 68, 0.18); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
  .chip-cyan { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
  .chip-gold { background: rgba(245, 158, 11, 0.18); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.35); }
  .chip-purple { background: rgba(168, 85, 247, 0.18); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }
  .chip-red { background: rgba(239, 68, 68, 0.18); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }

  /* Top Strategy Config Bar & Controls */
  .strategy-config-bar {
    background: #0d121c; border-bottom: 1px solid var(--border); padding: 8px 16px;
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; flex-shrink: 0;
  }
  .toggle-control {
    display: inline-flex; align-items: center; gap: 6px; cursor: pointer; font-size: 11.5px;
    user-select: none; color: #cbd5e1;
  }
  .toggle-control input[type="checkbox"] {
    accent-color: #0284c7; width: 14px; height: 14px; cursor: pointer;
  }
  .strat-btn {
    background: #182030; border: 1px solid #28354d; color: #94a3b8; padding: 4px 10px;
    border-radius: 4px; font-size: 11px; font-weight: 600; cursor: pointer; transition: all 0.15s;
  }
  .strat-btn:hover { background: #222d42; color: #fff; }
  .strat-btn.active-strat {
    background: #0369a1; border-color: #38bdf8; color: #fff; box-shadow: 0 0 8px rgba(56, 189, 248, 0.3);
  }
  .tp-row { display: flex; justify-content: space-between; align-items: center; }

  /* Center Chart Pane */
  .chart-pane { flex: 1; display: flex; flex-direction: column; overflow: hidden; background: #0c0f16; border-right: 1px solid var(--border); min-height: 0; min-width: 0; }
  .chart-header {
    background: #0f131d; padding: 6px 14px; border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center; gap: 12px;
    min-height: 38px; flex-shrink: 0; overflow: hidden;
  }
  .chart-header-left {
    display: flex; align-items: baseline; gap: 10px; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex-shrink: 1;
  }
  .chart-title { font-size: 13.5px; font-weight: 800; color: #fff; letter-spacing: 0.3px; flex-shrink: 0; }
  .chart-sub { font-size: 11px; color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .chart-badges-wrap {
    display: flex; align-items: center; gap: 6px; flex-shrink: 0; overflow-x: auto; scrollbar-width: none;
  }
  .chart-badges-wrap::-webkit-scrollbar { display: none; }
  #chart-container { flex: 1; width: 100%; height: 100%; min-height: 0; min-width: 0; position: relative; overflow: hidden; }

  /* Right Action Plan Pane */
  .plan-pane { width: 28%; overflow-y: auto; background: var(--surface); padding: 12px; display: flex; flex-direction: column; gap: 12px; }
  .card {
    background: #0f141f; border: 1px solid var(--border); border-radius: 6px; padding: 12px;
    display: flex; flex-direction: column; gap: 8px;
  }
  .card-title { font-size: 11px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; display: flex; justify-content: space-between; align-items: center; }
  
  .plan-action-box {
    background: #172033; border-left: 4px solid var(--accent); padding: 10px 12px; border-radius: 4px;
    display: flex; flex-direction: column; gap: 4px;
  }
  .action-headline { font-size: 12px; font-weight: 700; color: #fff; }
  .action-desc { font-size: 11px; color: #cbd5e1; line-height: 1.45; }

  .level-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  .level-box { background: #131a28; border: 1px solid #1e283d; border-radius: 4px; padding: 6px 8px; }
  .level-lbl { font-size: 9.5px; color: var(--muted); text-transform: uppercase; font-weight: 600; }
  .level-val { font-size: 13px; font-weight: 700; color: #fff; margin-top: 2px; }

  .windows-list { display: flex; flex-direction: column; gap: 5px; }
  .win-row {
    background: #111724; border: 1px solid #1c2538; border-radius: 4px; padding: 6px 8px;
    display: flex; justify-content: space-between; align-items: center; font-size: 11px;
  }
  .win-lbl { color: #cbd5e1; font-weight: 600; }
  .win-val { font-weight: 700; font-size: 10.5px; }

  /* Emerging Drawer (Slide-Over) */
  .drawer-overlay {
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.7); z-index: 1000;
    justify-content: flex-end;
  }
  .drawer-overlay.active { display: flex; }
  .drawer-panel {
    width: 680px; max-width: 90vw; height: 100vh; background: #0f141f; border-left: 1px solid #2d3748;
    display: flex; flex-direction: column; box-shadow: -10px 0 30px rgba(0,0,0,0.8);
  }
  .drawer-header {
    background: #141b29; padding: 14px 18px; border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
  }
  .drawer-title { font-size: 15px; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 8px; }
  .drawer-close { background: none; border: none; color: var(--muted); font-size: 22px; cursor: pointer; }
  .drawer-close:hover { color: #fff; }
  .drawer-tabs {
    display: flex; background: #111724; border-bottom: 1px solid var(--border); padding: 6px 12px; gap: 6px;
  }
  .d-tab {
    background: #172033; border: 1px solid #232d42; color: #94a3b8; padding: 6px 12px;
    border-radius: 4px; font-size: 11px; font-weight: 600; cursor: pointer;
  }
  .d-tab.active { background: #6d28d9; border-color: #8b5cf6; color: #fff; }
  .drawer-content { flex: 1; overflow-y: auto; padding: 14px; }
  
  /* Emerging Radar Table */
  .radar-table { width: 100%; border-collapse: collapse; font-size: 11.5px; }
  .radar-table th { background: #161e2e; color: var(--muted); padding: 7px 10px; font-size: 10.5px; text-align: right; }
  .radar-table th.tl { text-align: left; }
  .radar-table td { padding: 8px 10px; border-bottom: 1px solid #1c2538; text-align: right; }
  .radar-table td.tl { text-align: left; }
  .radar-table tr:hover { background: #1a2336; }
  .score-bar-bg { width: 80px; height: 6px; background: #1f293d; border-radius: 3px; display: inline-block; vertical-align: middle; margin-left: 6px; overflow: hidden; }
  .score-bar-fill { height: 100%; border-radius: 3px; }

  /* Nav View Switcher */
  .nav-btn {
    background: #182030; border: 1px solid #28354d; color: #94a3b8; padding: 5px 13px;
    border-radius: 6px; font-size: 11.5px; font-weight: 700; cursor: pointer; transition: all 0.15s;
    display: flex; align-items: center; gap: 6px; text-decoration: none;
  }
  .nav-btn:hover { background: #222d42; color: #fff; }
  .nav-btn.active { background: #0284c7; border-color: #38bdf8; color: #fff; box-shadow: 0 0 10px rgba(2,132,199,0.3); }

  /* Portfolio View Container & Scorecards */
  .portfolio-container {
    flex: 1; min-height: 0; display: flex; flex-direction: column; overflow-y: auto; background: var(--bg);
  }
  .portfolio-header-kpi {
    background: #0d121c; border-bottom: 1px solid var(--border); padding: 12px 18px;
    display: flex; gap: 12px; align-items: center; flex-wrap: wrap; flex-shrink: 0;
  }
  .kpi-card {
    background: #151b28; border: 1px solid #222c3f; border-radius: 8px; padding: 9px 14px;
    min-width: 135px; flex: 1; display: flex; flex-direction: column; gap: 2px;
  }
  .kpi-title { font-size: 10.5px; font-weight: 600; color: var(--muted); text-transform: uppercase; }
  .kpi-val { font-size: 17px; font-weight: 800; color: #fff; }
  .kpi-sub { font-size: 10.5px; color: var(--muted); }

  /* Execution Alerts Bar */
  .alerts-container {
    padding: 10px 18px 4px 18px; display: flex; flex-direction: column; gap: 8px; flex-shrink: 0;
  }
  .alert-card {
    border-radius: 6px; padding: 10px 14px; display: flex; justify-content: space-between; align-items: center;
    font-size: 12px; font-weight: 500; line-height: 1.4;
  }
  .alert-card.danger { background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; color: #fca5a5; }
  .alert-card.action { background: rgba(245, 158, 11, 0.15); border: 1px solid #f59e0b; color: #fde68a; }
  .alert-card.warning { background: rgba(59, 130, 246, 0.15); border: 1px solid #3b82f6; color: #bfdbfe; }
  .alert-card.info { background: rgba(56, 189, 248, 0.15); border: 1px solid #0284c7; color: #bae6fd; }
  .alert-card.success { background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #a7f3d0; }
  .alert-btn {
    padding: 4px 10px; border-radius: 4px; font-size: 11px; font-weight: 700; cursor: pointer; border: none;
    margin-left: 10px; white-space: nowrap;
  }
  .alert-btn.red { background: #ef4444; color: #fff; }
  .alert-btn.gold { background: #f59e0b; color: #000; }
  .alert-btn.blue { background: #3b82f6; color: #fff; }

  /* Portfolio Tables Pane */
  .portfolio-section {
    padding: 10px 18px 24px 18px; display: flex; flex-direction: column; gap: 16px;
  }
  .sec-heading {
    font-size: 13px; font-weight: 700; color: #cbd5e1; display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 8px;
  }
  .port-table {
    width: 100%; border-collapse: collapse; background: #131824; border: 1px solid var(--border); border-radius: 6px;
    font-size: 11.5px; overflow: hidden;
  }
  .port-table th {
    background: #0f1420; color: var(--muted); padding: 9px 10px; font-weight: 600; font-size: 10.5px; text-align: right;
    border-bottom: 1px solid var(--border);
  }
  .port-table th.tl { text-align: left; }
  .port-table td { padding: 9px 10px; border-bottom: 1px solid #1b2333; text-align: right; }
  .port-table td.tl { text-align: left; }
  .port-table tr:hover { background: #1a2233; }

  /* Modals */
  .modal-overlay {
    display: none; position: fixed; inset: 0; background: rgba(0, 0, 0, 0.75); z-index: 2000;
    align-items: center; justify-content: center; backdrop-filter: blur(2px);
  }
  .modal-overlay.open { display: flex; }
  .modal-box {
    background: #131824; border: 1px solid #2a3449; border-radius: 8px; width: 480px; max-width: 95vw;
    box-shadow: 0 10px 30px rgba(0,0,0,0.6); overflow: hidden; display: flex; flex-direction: column;
  }
  .modal-head {
    background: #171f2f; border-bottom: 1px solid #242f44; padding: 12px 16px;
    display: flex; justify-content: space-between; align-items: center;
  }
  .modal-head-title { font-size: 13px; font-weight: 700; color: #fff; }
  .modal-body { padding: 16px; display: flex; flex-direction: column; gap: 12px; font-size: 12px; }
  .modal-foot {
    background: #0f1420; border-top: 1px solid #242f44; padding: 10px 16px;
    display: flex; justify-content: flex-end; gap: 10px;
  }
  .input-label { font-size: 11px; color: var(--muted); font-weight: 600; margin-bottom: 4px; }
  .input-field {
    background: #0a0d14; border: 1px solid #28354d; color: #fff; padding: 6px 10px;
    border-radius: 4px; font-size: 12px; width: 100%; box-sizing: border-box; font-family: monospace;
  }
  .input-field:focus { outline: none; border-color: #0284c7; }
  .input-row { display: flex; gap: 12px; }
  .input-col { flex: 1; }
</style>
</head>
<body>

<header>
  <div style="display:flex; align-items:center; gap:12px;">
    <div class="title">
      <span>⚡ EP Follow-Through Tracker</span>
      <span class="badge">Port 8783</span>
    </div>
    <div style="display:flex; gap:6px; margin-left:10px;">
      <button class="nav-btn active" id="nav_btn_scanner" onclick="switchMainView('scanner')">
        📊 EP Scanner &amp; Catalysts
      </button>
      <button class="nav-btn" id="nav_btn_premarket" onclick="switchMainView('premarket')" style="border-color:#f59e0b; color:#fbbf24;">
        🌅 Premarket Radar <span id="premarket_nav_badge" class="badge" style="background:#f59e0b; color:#000; display:inline-block; padding:1px 5px; font-size:10px; margin-left:4px;">0</span>
      </button>
      <button class="nav-btn" id="nav_btn_portfolio" onclick="switchMainView('portfolio')" style="border-color:#0284c7; color:#38bdf8;">
        💼 EP Portfolio Manager <span id="portfolio_nav_badge" class="badge" style="background:#ef4444; color:#fff; display:none; padding:1px 5px; font-size:10px; margin-left:4px;">0</span>
      </button>
    </div>
  </div>
  <div style="display:flex; align-items:center; gap:8px;">
    <button class="btn-emerging" id="btn_pm_update_header" onclick="updatePremarketScan()" style="background:#78350f; border-color:#d97706; color:#fef3c7;" title="Pull 15m yfinance intraday bars & EDGAR 8-K filings for in-play sectors and themes">
      ⚡ Premarket Update
    </button>
    <button class="btn-emerging" onclick="openEmergingDrawer()">
      🌐 Emerging Themes <span class="badge" style="background:#8b5cf6;" id="em_badge_top">Leaderboard</span>
    </button>
    <button class="btn-emerging" onclick="onAiTopPicksClick()" style="border-color:#10b981; color:#34d399; margin-right:8px;" id="btn_top_ai_modal" title="Filter left table to AI Top Picks & open pipeline report">
      🏆 AI Top Picks (<span id="btn_cnt_top_ai">0</span>)
    </button>
    <button class="btn-export" onclick="downloadTradingViewList()" title="Export current/all tickers formatted for TradingView Watchlist">
      📋 TradingView List
    </button>
    <button class="btn-export" onclick="downloadAiJson()" title="Export rich JSON with catalyst, SEC filings, and technical metrics for AI prompts">
      🤖 AI Agent JSON
    </button>
    <button class="btn-export" onclick="downloadPremarketAiJson()" style="border-color:#f59e0b; color:#fbbf24;" title="Export rich Premarket JSON with early tactical triggers, 15m ORB levels, VWAP reclaim status, and SEC filings for AI prompt">
      🌅 Premarket JSON
    </button>
    <span style="font-size:11.5px; color:var(--muted); margin-left:4px;">Market: <b>Live</b></span>
    <button class="btn-refresh" onclick="refreshScan()">🔄 Rescan</button>
  </div>
</header>

<div id="scanner_view" style="display:flex; flex-direction:column; flex:1; overflow:hidden;">
<div class="view-bar">
  <span style="font-size:11px; color:var(--muted); font-weight:700; text-transform:uppercase;">View Pipeline:</span>
  <button class="view-tab active" id="tab_all" onclick="setView('all')">⭐ All Tracked EPs (<span id="cnt_all">0</span>)</button>
  <button class="view-tab" id="tab_top_ai" onclick="setView('top_ai')" style="border-color:#10b981; color:#34d399;">🏆 AI Top Ranked (<span id="cnt_top_ai">0</span>)</button>
  <button class="view-tab" id="tab_today" onclick="setView('today')">🔥 Today's Fresh EPs (<span id="cnt_today">0</span>)</button>
  <button class="view-tab" id="tab_dre" onclick="setView('dre')" style="border-color:#f59e0b; color:#fbbf24;">⚡ Delayed Reaction EPs (<span id="cnt_dre">0</span>)</button>
  <button class="view-tab" id="tab_week" onclick="setView('week')">📅 Past Week Pipeline (<span id="cnt_week">0</span>)</button>
  <button class="view-tab" id="tab_active" onclick="setView('active')">🟢 Active Holding Stop (<span id="cnt_active">0</span>)</button>
  <button class="view-tab" id="tab_idiosyncratic" onclick="setView('idiosyncratic')" style="border-color:#10b981; color:#6ee7b7;">🎯 Idiosyncratic Alpha (<span id="cnt_idiosyncratic">0</span>)</button>
  
  <div style="margin-left:auto; display:flex; align-items:center; gap:6px;">
    <label style="font-size:11px; color:var(--muted); font-weight:700; text-transform:uppercase;">Trader Archetype:</label>
    <select id="f_trader_archetype" onchange="onArchetypeFilterChange()" style="background:#131826; border:1px solid #28354d; color:#38bdf8; font-weight:700; padding:4px 8px; border-radius:4px; font-size:11.5px; cursor:pointer;">
      <option value="">All 6 Institutional Archetypes</option>
      <option value="pinnacle">💎 Pinnacle Elite (Apex Quality)</option>
      <option value="conservative">🛡️ Conservative Swing (Delayed Breakout Confirmation)</option>
      <option value="compounder">🚀 Multi-Quarter Compounder (Secular Leaders)</option>
      <option value="emerging">🌟 Emerging Leader (Velocity &amp; Power)</option>
      <option value="sweet">⭐ Institutional Sweet Spot (Volume Surge)</option>
      <option value="turnaround">🔄 Neglected Turnaround (Deep Value)</option>
      <option value="high_beta">⚡ High-Beta Momentum (Fast Sprinter)</option>
      <option value="dre">⚡ Delayed Reaction (DRE - Weak Day 1 Close)</option>
      <option value="idiosyncratic">🎯 Idiosyncratic Alpha (Mega-Catalyst)</option>
      <option value="headwind">🚫 Severe Headwind (Trap Avoidance)</option>
    </select>
  </div>
</div>

<!-- Strategy Execution Rules & Trade Management Bar -->
<div class="strategy-config-bar">
  <div style="display:flex; align-items:center; gap:8px;">
    <span style="font-weight:700; color:#f8fafc; font-size:12px; display:flex; align-items:center; gap:5px;">
      ⚙️ Strategy Execution:
    </span>
    <span style="color:#38bdf8; font-size:11px; font-weight:700; background:#0c2340; border:1px solid #0284c7; padding:2px 8px; border-radius:10px;" id="strategy_mode_badge">AI-Enhanced Optimal</span>
  </div>
  
  <div style="display:flex; align-items:center; gap:14px; flex-wrap:wrap;">
    <!-- Toggle 0: Trade 1 Entry -->
    <label class="toggle-control" title="When ON: takes Trade 1 on Day 1 close with stop at Day 1 Low. When OFF: skips Trade 1 entirely and only trades continuation legs (Trade 2 & 3).">
      <input type="checkbox" id="cfg_trade1" checked onchange="onStrategyConfigChange()">
      <span class="toggle-label">🚀 Trade 1 Entry</span>
    </label>

    <!-- Toggle 1: ML Profit Target -->
    <label class="toggle-control" title="When ON: exits trade on momentum climax when ML Exhaustion probability > 0.80. When OFF: pure trend rider exiting on structural 50 SMA breakdown.">
      <input type="checkbox" id="cfg_ml_targets" checked onchange="onStrategyConfigChange()">
      <span class="toggle-label">🎯 ML Profit Target (Exhaustion)</span>
    </label>

    <!-- Toggle 2: ML Trailing Stop -->
    <label class="toggle-control" title="When ON: dynamically trails stop based on 10th percentile MAE model once +2.0R hurdle is reached. When OFF: hard stop fixed at Day 1 Low.">
      <input type="checkbox" id="cfg_ml_stop" checked onchange="onStrategyConfigChange()">
      <span class="toggle-label">🛑 ML Dynamic Stop (+2.0R Hurdle)</span>
    </label>

    <!-- Toggle 3: Multi-Leg (Trade 2 & 3) -->
    <label class="toggle-control" title="When ON: monitors for 2nd and 3rd leg yellow re-entries after initial trade exit. When OFF: evaluates Trade 1 only.">
      <input type="checkbox" id="cfg_multileg" checked onchange="onStrategyConfigChange()">
      <span class="toggle-label">♻️ Multi-Leg (Trade 2 &amp; 3)</span>
    </label>

    <!-- Toggle 4: Multi-Leg ML Filter -->
    <label class="toggle-control" title="When ON: only takes re-entries confirmed by Cost-Sensitive ML Re-Entry model (Prob >= 35%). When OFF: takes all mechanical technical re-entries.">
      <input type="checkbox" id="cfg_multileg_ml" checked onchange="onStrategyConfigChange()">
      <span class="toggle-label">🤖 Multi-Leg ML Filter (≥35%)</span>
    </label>

    <!-- Quick Presets -->
    <div style="display:flex; gap:6px;">
      <button class="strat-btn" id="btn_strat_baseline" onclick="setStrategyPreset('baseline')" title="Raw mechanical execution of Trade 1 (No ML)">Mechanical Baseline</button>
      <button class="strat-btn active-strat" id="btn_strat_optimal" onclick="setStrategyPreset('optimal')" title="All ML features ON: Dynamic Trailing Stop, Exhaustion Climax Exit, and ML-Filtered Multi-Leg">AI-Enhanced Optimal</button>
    </div>
  </div>
</div>

<!-- Interactive Expectancy Boosters Bar -->
<div class="booster-bar">
  <span class="booster-label">⚡ Expectancy Boosters:</span>
  <div class="booster-chip" id="b_veto_headwind" onclick="toggleBooster('veto_headwind')" title="Hard Veto: Filters out setups occurring in sectors or themes in bottom 35th percentile (severe multi-month headwind). Cuts trap rate from 38% to 15% and slashes max drawdown from -26R to -8R.">
    <span>🚫 Veto Severe Headwinds</span>
    <span class="booster-pill">Trap -23% | MaxDD -8R</span>
  </div>
  <div class="booster-chip" id="b_novel_9m" onclick="toggleBooster('novel_9m')" title="Novel 9M Volume: First 9M+ share volume day in 60 sessions. 10-year study delivers +1.92 R EV vs +0.95 R for repeat volume names.">
    <span>🔥 Novel 9M Volume</span>
    <span class="booster-pill">EV +1.92R (+102%)</span>
  </div>
  <div class="booster-chip" id="b_cap10" onclick="toggleBooster('cap10')" title="Bonde CAP 10x10: Small/Mid-Cap (< $10B). Multi-quarter compounders have smaller market caps that allow for explosive institutional multiple expansion.">
    <span>💎 CAP 10×10 (&lt; $10B)</span>
    <span class="booster-pill">High Multibagger EV</span>
  </div>
  <div class="booster-chip" id="b_rvol_sweet" onclick="toggleBooster('rvol_sweet')" title="Sweet Spot RVOL: Requires RVOL ≥ 5.0x. Cuts empirical drawdown in half (-6.0R) and surges EV to +1.68R.">
    <span>🔥 Sweet Spot RVOL (≥5.0x)</span>
    <span class="booster-pill">EV +1.68R | MaxDD -6R</span>
  </div>
  <div class="booster-chip" id="b_48h" onclick="toggleBooster('48h')" title="48H Body Absorption: Requires holding the upper 50% of Day 1 candle body through Day 3. Slashes Gap & Crap trap rate from 48% to 22% and adds +18% win rate.">
    <span>🛡️ 48H Upper Body Absorption</span>
    <span class="booster-pill">Trap -22% | Win +18%</span>
  </div>
  <div class="booster-chip" id="b_elite_close" onclick="toggleBooster('elite_close')" title="Elite Close: Requires Day 1 close in top 20% of range (ClosePos ≥ 0.80). Adds +0.69R EV and boosts follow-through by +9.4%.">
    <span>⭐ Elite ClosePos (≥0.80)</span>
    <span class="booster-pill">EV +0.69R | Low Churn</span>
  </div>
  <div class="booster-chip" id="b_sma50" onclick="toggleBooster('sma50')" title="Institutional Baseline: Requires price to be trading above rising 50-day SMA. 10-year study shows 74.1% win rate and 23.28 profit factor for compounders.">
    <span>🏛️ Above 50-Day SMA</span>
    <span class="booster-pill">PF 23.3 | Win 74%</span>
  </div>
  <div class="booster-chip" id="b_tailwind" onclick="toggleBooster('tailwind')" title="Thematic Tailwind: Requires Sector/Theme score ≥ 65th percentile (Power Clusters or Velocity Surges). Adds +12% win rate and +45% higher portfolio P&L.">
    <span>🟢 Thematic Tailwind (≥65)</span>
    <span class="booster-pill">Win +12% | P&L +45%</span>
  </div>
  <button class="booster-reset" onclick="resetBoosters()">Clear Boosters</button>
  <button class="btn-guide-toggle" onclick="toggleImpactGuide()" id="btn_toggle_guide">
    📖 Strategy Impact Guide &amp; Stat Explanations ▾
  </button>
</div>

<!-- Strategy Impact Guide & Quant Diagnostics Panel (Collapsible) -->
<div class="impact-guide-panel" id="impact_guide">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px;">
    <div>
      <div style="font-size:13px; font-weight:700; color:#c7d2fe; display:flex; align-items:center; gap:8px;">
        <span>📊 Quantitative Expectancy Guide: Zero-Lookahead Ground Truth Attribution</span>
        <span class="badge" style="background:#4338ca; color:#fff;">10-Year Empirical Study (6,500 EPs)</span>
      </div>
      <div style="font-size:11px; color:#94a3b8; margin-top:3px;">
        Raw big gappers without filtering deliver a modest baseline (<b>32.4% Win Rate, +0.40 R EV</b>; 67.6% fail or get stopped). Asymmetric alpha comes from disciplined execution filters that eliminate capital-destructive traps and scale into institutional compounders:
      </div>
    </div>
    <button class="modal-close" style="font-size:18px; line-height:1; cursor:pointer;" onclick="toggleImpactGuide()">&times;</button>
  </div>
  <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap:10px;">
    <div style="background:#111522; border-left:3px solid #ef4444; border-radius:4px; padding:10px 12px;">
      <div style="font-weight:700; color:#f87171; font-size:11.5px; display:flex; justify-content:space-between;">
        <span>1. 🚫 Veto Severe Headwinds</span>
        <span style="color:#fbbf24; font-size:10px;">MaxDD: -26R → -8R</span>
      </div>
      <div style="color:#cbd5e1; font-size:11px; margin-top:4px; line-height:1.4;">
        <b>Statistical Impact:</b> Win Rate <b>+8.6%</b> | Drawdown slashed from <b>-26.0 R to -8.0 R</b> | Cuts negative EV drag of <b>-0.85 R</b> per trade.<br>
        <span style="color:#94a3b8;"><b>Mechanism:</b> Outcast sectors (bottom 35th percentile) suffer broad institutional outflows. EPs here trap 37.5% of the time as institutions use gap liquidity to sell existing inventory.</span>
      </div>
    </div>

    <div style="background:#111522; border-left:3px solid #f59e0b; border-radius:4px; padding:10px 12px;">
      <div style="font-weight:700; color:#fbbf24; font-size:11.5px; display:flex; justify-content:space-between;">
        <span>2. 🔥 Sweet Spot RVOL (≥5.0x)</span>
        <span style="color:#34d399; font-size:10px;">EV: +0.40R → +1.68R</span>
      </div>
      <div style="color:#cbd5e1; font-size:11px; margin-top:4px; line-height:1.4;">
        <b>Statistical Impact:</b> Expected Value jumps to <b>+1.68 R</b> | Max Drawdown cut in half to <b>-6.0 R</b> | Eliminates 65% of low-momentum churn.<br>
        <span style="color:#94a3b8;"><b>Mechanism:</b> 5x+ volume creates true supply vacuums. Massive institutional demand cannot be satisfied in 1 session, causing multi-week Post-Earnings Announcement Drift (PEAD).</span>
      </div>
    </div>

    <div style="background:#111522; border-left:3px solid #10b981; border-radius:4px; padding:10px 12px;">
      <div style="font-weight:700; color:#34d399; font-size:11.5px; display:flex; justify-content:space-between;">
        <span>3. 🛡️ 48H Upper Body Absorption</span>
        <span style="color:#38bdf8; font-size:10px;">Trap Rate: 48% → 22%</span>
      </div>
      <div style="color:#cbd5e1; font-size:11px; margin-top:4px; line-height:1.4;">
        <b>Statistical Impact:</b> Slashes Gap &amp; Crap trap rate from <b>48% to 22%</b> | Win Rate jumps to <b>43.8%+</b>.<br>
        <span style="color:#94a3b8;"><b>Mechanism:</b> Holding the upper 50% of the Day 1 body through Day 3 proves that aggressive buyers are absorbing profit-taking. Stocks failing this gate usually announce dilutive secondaries (ATM offerings).</span>
      </div>
    </div>

    <div style="background:#111522; border-left:3px solid #38bdf8; border-radius:4px; padding:10px 12px;">
      <div style="font-weight:700; color:#38bdf8; font-size:11.5px; display:flex; justify-content:space-between;">
        <span>4. ⭐ Elite ClosePos (≥0.80)</span>
        <span style="color:#c084fc; font-size:10px;">Follow-Through: +9.4%</span>
      </div>
      <div style="color:#cbd5e1; font-size:11px; margin-top:4px; line-height:1.4;">
        <b>Statistical Impact:</b> Adds <b>+0.69 R</b> EV per trade | Reduces early shakeout rate.<br>
        <span style="color:#94a3b8;"><b>Mechanism:</b> Closing in the top 20% of the daily high-low range confirms that institutions were aggressively bidding into the 4 PM close, eliminating upper-wick rejection traps.</span>
      </div>
    </div>

    <div style="background:#111522; border-left:3px solid #8b5cf6; border-radius:4px; padding:10px 12px;">
      <div style="font-weight:700; color:#c084fc; font-size:11.5px; display:flex; justify-content:space-between;">
        <span>5. 🏛️ Above 50-Day SMA Baseline</span>
        <span style="color:#34d399; font-size:10px;">Win Rate: 74.1% | PF 23.3</span>
      </div>
      <div style="color:#cbd5e1; font-size:11px; margin-top:4px; line-height:1.4;">
        <b>Statistical Impact:</b> 10-Year Win Rate <b>74.1%</b> | Multi-Quarter Compounder Profit Factor <b>23.28</b>.<br>
        <span style="color:#94a3b8;"><b>Mechanism:</b> Stocks above their 50 SMA have no immediate trapped overhead supply. Sellers are in profit and willing to let runners ride, removing overhead resistance.</span>
      </div>
    </div>

    <div style="background:#111522; border-left:3px solid #06b6d4; border-radius:4px; padding:10px 12px;">
      <div style="font-weight:700; color:#22d3ee; font-size:11.5px; display:flex; justify-content:space-between;">
        <span>6. 🟢 Thematic Tailwind (Score ≥65)</span>
        <span style="color:#fbbf24; font-size:10px;">Net P&amp;L: +45% Higher</span>
      </div>
      <div style="color:#cbd5e1; font-size:11px; margin-top:4px; line-height:1.4;">
        <b>Statistical Impact:</b> Win Rate <b>+12.0%</b> | Total Portfolio Gain <b>+45% higher</b>.<br>
        <span style="color:#94a3b8;"><b>Mechanism:</b> Macro and sector capital rotation creates a tide that lifts all boats. Even mediocre single-stock EPS reports can drift higher for months if the broader theme is in a Multi-TF Power Cluster.</span>
      </div>
    </div>

    <div style="background:#111522; border-left:3px solid #ec4899; border-radius:4px; padding:10px 12px;">
      <div style="font-weight:700; color:#f472b6; font-size:11.5px; display:flex; justify-content:space-between;">
        <span>7. 🔥 Novel 9M Volume Ignition</span>
        <span style="color:#34d399; font-size:10px;">EV: +0.95R → +1.92R</span>
      </div>
      <div style="color:#cbd5e1; font-size:11px; margin-top:4px; line-height:1.4;">
        <b>Statistical Impact:</b> Expected Value jumps from <b>+0.95 R to +1.92 R</b> (+102% edge boost) when volume crosses 9M shares for the first time in 60 sessions.<br>
        <span style="color:#94a3b8;"><b>Mechanism:</b> Bonde 9M novelty creates sudden awareness. Mega-caps routinely trade 9M, but when a mid-cap trades 9M out of nowhere, an institutional sponsor is establishing a foundational footprint.</span>
      </div>
    </div>

    <div style="background:#111522; border-left:3px solid #eab308; border-radius:4px; padding:10px 12px;">
      <div style="font-weight:700; color:#facc15; font-size:11.5px; display:flex; justify-content:space-between;">
        <span>8. 💎 Bonde CAP 10×10 (&lt; $10B)</span>
        <span style="color:#38bdf8; font-size:10px;">Multibagger Incubator</span>
      </div>
      <div style="color:#cbd5e1; font-size:11px; margin-top:4px; line-height:1.4;">
        <b>Statistical Impact:</b> Concentrates <b>88.4% of all +100% to +358% compounders</b>.<br>
        <span style="color:#94a3b8;"><b>Mechanism:</b> Stocks under $10B market cap have room for 5x to 10x PEAD drift before institutional ownership saturation caps multiple expansion.</span>
      </div>
    </div>

    <div style="background:#111522; border-left:3px solid #f59e0b; border-radius:4px; padding:10px 12px;">
      <div style="font-weight:700; color:#fbbf24; font-size:11.5px; display:flex; justify-content:space-between;">
        <span>9. ⚡ Bonde DRE (Weak Day 1 Close)</span>
        <span style="color:#34d399; font-size:10px;">EV: +0.86R | PF 2.20</span>
      </div>
      <div style="color:#cbd5e1; font-size:11px; margin-top:4px; line-height:1.4;">
        <b>Statistical Impact:</b> Filters out <b>9,698 traps (62% of weak-close gaps)</b> | Delivers <b>+0.86 R EV</b> (Day 2 breakout) to <b>+0.71 R EV</b> (Day 3) | <b>2.20 PF</b> | Tighter stop (<b>7.4%</b> vs 10.9%).<br>
        <span style="color:#94a3b8;"><b>Mechanism:</b> When a catalyst gaps but closes weak, waiting for Day 1 High breakout proves absorption and avoids immediate Day 2 gap-down collapses.</span>
      </div>
    </div>

    <div style="background:#111522; border-left:3px solid #38bdf8; border-radius:4px; padding:10px 12px;">
      <div style="font-weight:700; color:#38bdf8; font-size:11.5px; display:flex; justify-content:space-between;">
        <span>10. 🛡️ Conservative Swing (Delayed Strong Entry)</span>
        <span style="color:#34d399; font-size:10px;">Win Rate: 36.2% | PF 1.70</span>
      </div>
      <div style="color:#cbd5e1; font-size:11px; margin-top:4px; line-height:1.4;">
        <b>Statistical Impact:</b> Lifts Win Rate from <b>32.4% to 36.2%</b> (+3.8% edge) | Automatically avoids <b>992 Day 2 gap-and-crap traps</b>.<br>
        <span style="color:#94a3b8;"><b>Mechanism:</b> Rather than buying Day 1 close into potential overnight reversal risk, conservative traders enter on Day 1 High breakout confirmation on Day 2–5.</span>
      </div>
    </div>
  </div>
</div>

<!-- Full-Width Active Ticker Context Bar (Above Panels) -->
<div class="active-ticker-bar" id="active_ticker_bar">
  <div class="active-ticker-left">
    <div class="active-ticker-sym" id="top_sym">Select Ticker</div>
    <div class="active-ticker-details" id="top_details">Click any stock in the table to display its confirmation chart and trade plan</div>
  </div>
  <div class="active-ticker-badges" id="top_badges"></div>
</div>

<div class="workspace">
  <!-- Left: Event Table -->
  <div class="table-pane">
    <div id="filter_stat_banner" style="background:#131826; border-bottom:1px solid var(--border); padding:8px 14px; font-size:11.5px; color:#cbd5e1;">
      <!-- Dynamically filled by updateLiveStatsBanner -->
    </div>
    <table id="ep-table">
      <thead>
        <tr>
          <th class="tl">Date</th>
          <th class="tl">Symbol</th>
          <th class="tl">Trader Archetype</th>
          <th class="tl">Group Momentum</th>
          <th class="tl">Stage / Status</th>
          <th>Gap %</th>
          <th>RVOL</th>
          <th>ClosePos</th>
          <th>$Traded</th>
          <th>P&amp;L %</th>
          <th>Open R</th>
          <th title="AI Probability for +50% target (Day 5 Live Inference)">AI 50</th>
          <th title="AI Probability for +150% target (Day 5 Live Inference)">AI 150</th>
        </tr>
      </thead>
      <tbody id="table-body">
        <tr><td colspan="13" style="text-align:center; padding:20px; color:var(--muted);">Loading live EP pipeline...</td></tr>
      </tbody>
    </table>
  </div>

  <!-- Center: Interactive Candlestick Chart -->
  <div class="chart-pane">
    <div id="chart-container"></div>
  </div>

  <!-- Right: Trade Management Plan & Confirmation Windows -->
  <div class="plan-pane" id="plan_pane">
    <!-- Card 1: Actionable Trade Execution Guidance -->
    <div class="card" id="card_action" style="border-color:#0284c7;">
      <div class="card-title">
        <span style="color:#38bdf8;">🎯 Suggested Trade Management</span>
        <span id="plan_badge" class="badge-fresh">Active</span>
      </div>
      <div class="plan-action-box">
        <div class="action-headline" id="action_headline">Evaluating EP...</div>
        <div class="action-desc" id="action_desc">Select a stock to view suggested trade plan.</div>
      </div>
      <div class="level-grid">
        <div class="level-box">
          <div class="level-lbl">Entry Level</div>
          <div class="level-val" style="color:var(--green);" id="val_entry">$0.00</div>
        </div>
        <div class="level-box">
          <div class="level-lbl">D1 Stop (Risk %)</div>
          <div class="level-val" style="color:var(--red);" id="val_stop">$0.00 (-0.0%)</div>
        </div>
        <div class="level-box">
          <div class="level-lbl">Secondary Add (+50%)</div>
          <div class="level-val" style="color:var(--cyan);" id="val_add">$0.00</div>
        </div>
        <div class="level-box">
          <div class="level-lbl">Current P&amp;L (R)</div>
          <div class="level-val" style="color:var(--gold);" id="val_pnl">+0.00 R</div>
        </div>
      </div>
      <div style="margin-top:12px;">
        <button class="btn-add-portfolio" id="btn_card_add_port" onclick="openAddToPortfolioModal()" style="width:100%; background:#0284c7; border:none; color:#fff; padding:8px 12px; border-radius:6px; font-weight:700; font-size:12px; cursor:pointer; display:flex; align-items:center; justify-content:center; gap:6px; transition:all 0.15s;">
          💼 Add to Live Portfolio
        </button>
      </div>
    </div>

    
    <!-- Card 1.2: AI Prediction & Alerts -->
    <div class="card" id="card_ai_prediction" style="border-color:#10b981;">
      <div class="card-title">
        <span style="color:#34d399;">🤖 AI Probabilities (V2 Rolling)</span>
      </div>
      <div style="font-size:11.5px; display:flex; flex-direction:column; gap:8px;">
        
        <!-- V2 Rolling -->
        <div style="display:flex; flex-direction:column; background:#0f172a; padding:8px; border-radius:6px; border:1px solid #38bdf8;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#38bdf8; font-weight:bold;">V2 Rolling Timeline</span>
                <span style="color:#e2e8f0; font-weight:bold; background:#0369a1; padding:2px 6px; border-radius:10px; font-size:10px;" id="ai_v2_day_badge">Day X</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding-bottom:2px;">
              <span style="color:var(--muted);">+50% Target:</span>
              <span id="ai_v2_50" style="font-weight:700; color:#cbd5e1;">--%</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding-bottom:2px;">
              <span style="color:var(--muted);">+100% Target:</span>
              <span id="ai_v2_100" style="font-weight:700; color:var(--green);">--%</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding-bottom:2px;">
              <span style="color:var(--muted);">+150% Target:</span>
              <span id="ai_v2_150" style="font-weight:700; color:var(--cyan);">--%</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding-bottom:2px;">
              <span style="color:var(--muted);">+200% Target:</span>
              <span id="ai_v2_200" style="font-weight:700; color:#a855f7;">--%</span>
            </div>
        </div>
        <div id="ai_adv_warnings_v2" style="margin-top: 5px; font-weight: bold; font-size: 11px; display: none;">
             <div id="ai_dynamic_stop_v2" style="color: #f59e0b; padding: 4px; border: 1px solid #f59e0b; border-radius: 4px; display: none; text-align: center;">🤖 ML Trailing Stop (10th %ile MAE): <span id="ai_stop_val_v2"></span></div>
             <div id="ai_exhaustion_v2" style="color: #8b5cf6; padding: 4px; border: 1px solid #8b5cf6; border-radius: 4px; display: none; text-align: center; margin-top: 5px;">🔥 EXHAUSTION: <span id="ai_exh_val_v2"></span></div>
             <div id="ai_reentry_v2" style="color: #34d399; padding: 4px; border: 1px solid #34d399; border-radius: 4px; display: none; text-align: center; margin-top: 5px;">♻️ ML TRADE 2/3 RE-ENTRY: <span id="ai_reentry_val_v2"></span></div>
        </div>

      </div>
    </div>

    <!-- Card 1.4: Trade 2 & 3: Yellow Re-Entry (Multi-Leg PEAD) Dossier -->
    <div class="card" id="card_trade2_exec" style="border-color:#38bdf8;">
      <div class="card-title">
        <span style="color:#38bdf8;">🔄 Trade 2 &amp; 3: Yellow Re-Entry</span>
        <span id="t2_status_pill" class="score-chip chip-cyan" style="font-size:10px;">Evaluating...</span>
      </div>
      <div id="t2_details_box" style="display:flex; flex-direction:column; gap:6px; font-size:11px;">
        <div class="tp-row"><span style="color:var(--muted);">Re-Entry Status:</span><span id="t2_status_val" style="font-weight:700; color:#fff;">--</span></div>
        <div class="tp-row"><span style="color:var(--muted);">Re-Entry Trigger:</span><span id="t2_entry_val" style="font-weight:700; color:var(--green);">$0.00</span></div>
        <div class="tp-row"><span style="color:var(--muted);">Stop Loss (5D Shelf):</span><span id="t2_stop_val" style="font-weight:700; color:var(--red);">$0.00 (-0.0%)</span></div>
        <div class="tp-row"><span style="color:var(--muted);">Consolidation &amp; Retracement:</span><span id="t2_cons_val" style="color:#cbd5e1;">0 sessions (Pullback 0.0%)</span></div>
        <div class="tp-row" id="row_t2_ml" style="display:none;"><span style="color:var(--muted);">ML Re-Entry Conviction:</span><span id="t2_ml_val" style="font-weight:700; color:var(--cyan);">--%</span></div>
        <div class="tp-row"><span style="color:var(--muted);">Trade 2 Open P&amp;L:</span><span id="t2_pnl_val" style="font-weight:700; color:var(--gold);">+0.00 R</span></div>
      </div>
      <div id="t2_none_msg" style="display:none; color:var(--muted); font-size:11px; font-style:italic; line-height:1.4;">
        No Second-Leg Re-Entry detected (did not re-flip Yellow within 45 sessions or retracement exceeded 55%).
      </div>
    </div>

    <!-- Card 1.3: Strategy Quality & Expectancy Audit -->
    <div class="card" id="card_quality_audit">
      <div class="card-title">
        <span>⚡ Expectancy &amp; Booster Audit</span>
        <span id="qual_score_badge" class="score-chip chip-tailwind">Evaluating...</span>
      </div>
      <div id="qual_checklist" style="display:flex; flex-direction:column; gap:6px; font-size:11px;">
        <div style="color:var(--muted); font-style:italic;">Select a stock to audit booster compliance.</div>
      </div>
    </div>

    <!-- Card 1.5: Thematic Momentum & Archetype Diagnostic -->
    <div class="card" id="card_thematic">
      <div class="card-title">
        <span>🌐 Sector &amp; Thematic Radar</span>
        <span id="thm_badge" class="score-chip chip-tailwind">Score: 85</span>
      </div>
      <div style="font-size:11.5px; display:flex; flex-direction:column; gap:6px;">
        <div style="display:flex; justify-content:space-between;">
          <span style="color:var(--muted);">Industry Sector:</span>
          <span id="thm_sec_name" style="font-weight:700; color:#38bdf8; cursor:pointer; text-decoration:underline dotted;" title="Click to view Sector Stock Leaderboard" onclick="openGroupStocks(this.textContent, 'sector')">Technology</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
          <span style="color:var(--muted);">Primary Theme:</span>
          <span id="thm_theme_name" style="font-weight:700; color:#c084fc; cursor:pointer; text-decoration:underline dotted;" title="Click to view Theme Stock Leaderboard" onclick="openGroupStocks(this.textContent, 'theme')">Semiconductors</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
          <span style="color:var(--muted);">Archetype:</span>
          <span id="thm_archetype" style="font-weight:700; color:var(--purple);">Multi-TF Power Cluster</span>
        </div>
        <div id="thm_diag_desc" style="background:#131a28; border-left:3px solid var(--accent); padding:6px 10px; border-radius:4px; font-size:11px; line-height:1.4; color:#cbd5e1; margin-top:2px;">
          Evaluating momentum context...
        </div>
      </div>
    </div>

    <!-- Card 1.8: Fundamental Catalyst & SEC 8-K/6-K Filings Audit -->
    <div class="card" id="card_catalyst" style="border-color:#8b5cf6;">
      <div class="card-title">
        <span style="color:#c084fc;">📰 Fundamental Catalyst &amp; SEC Filings</span>
        <span id="cat_tier_badge" class="score-chip" style="background:#2e1065; color:#c4b5fd; border:1px solid #7c3aed;">Tier-1 Catalyst</span>
      </div>
      <div style="display:flex; flex-direction:column; gap:8px;">
        <div style="background:#131722; border-left:3px solid #8b5cf6; padding:8px 10px; border-radius:4px;">
          <div style="font-size:10px; color:#94a3b8; text-transform:uppercase; font-weight:700;" id="cat_category">Catalyst Category</div>
          <div style="font-size:12px; font-weight:700; color:#fff; margin-top:2px;" id="cat_headline">Headline Loading...</div>
        </div>
        <div style="font-size:11px; color:#cbd5e1; line-height:1.45; background:#0b0e14; padding:8px 10px; border-radius:4px; border:1px solid #1e283d;">
          <div style="font-size:10px; color:#38bdf8; font-weight:700; margin-bottom:4px; text-transform:uppercase;">🏛️ Institutional Trajectory Analysis:</div>
          <div id="cat_analysis">Deep analysis loading...</div>
        </div>
        <div>
          <div style="font-size:10px; color:var(--muted); text-transform:uppercase; font-weight:700; margin-bottom:4px; display:flex; justify-content:space-between;">
            <span>Official SEC EDGAR Filings (T-3 to T+0):</span>
            <span id="filing_count_badge" style="color:#38bdf8;">0 Filings</span>
          </div>
          <div id="filings_list" style="display:flex; flex-direction:column; gap:4px; font-size:10.5px;">
            <div style="color:var(--muted); font-style:italic;">No material SEC filings found in window.</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Card 2: 4 Confirmation Windows Audit -->
    <div class="card">
      <div class="card-title">
        <span>⏳ 4 Confirmation Windows Audit</span>
        <span style="font-size:10px; color:var(--muted);">Progress Tracker</span>
      </div>
      <div class="windows-list">
        <div class="win-row">
          <span class="win-lbl">Window 1: Day 1 Volume &amp; Range</span>
          <span class="win-val" id="win1_val" style="color:var(--green);">✓ Elite Surge</span>
        </div>
        <div class="win-row">
          <span class="win-lbl">Window 2: Day 2 Gap &amp; Go (+Add)</span>
          <span class="win-val" id="win2_val">Pending</span>
        </div>
        <div class="win-row">
          <span class="win-lbl">Window 3: Day 3 (48H Absorption)</span>
          <span class="win-val" id="win3_val">Pending</span>
        </div>
        <div class="win-row">
          <span class="win-lbl">Window 4: Day 5 Leg Resolution</span>
          <span class="win-val" id="win4_val">Pending</span>
        </div>
      </div>
    </div>

    <!-- Card 3: Progressive Exposure Sizing Guide -->
    <div class="card">
      <div class="card-title">
        <span>⚖️ Position Sizing (Progressive Exposure)</span>
        <span style="font-size:10px; color:var(--gold);">Max DD Cut -48%</span>
      </div>
      <div style="font-size:11px; color:#cbd5e1; line-height:1.45;">
        • <b>Normal Regime:</b> Risk exactly <b>1.0 R</b> on Day 1 entry.<br/>
        • <b>Loss-Streak Defense:</b> If on <b>2 consecutive losses</b>, throttle heat to <b>0.50 R</b>. If 3 losses, cut to <b>0.25 R</b>.<br/>
        • <b>Hot-Streak Scale:</b> If on <b>2 consecutive winners</b>, expand to <b>1.35 R</b> to aggressively compound.
      </div>
    </div>
  </div>
</div>
</div> <!-- end #scanner_view -->

<!-- DEDICATED EP PORTFOLIO MANAGER VIEW -->
<div id="portfolio_view" class="portfolio-container" style="display:none;">
  <!-- Portfolio KPI Scorecard Banner -->
  <div class="portfolio-header-kpi">
    <div class="kpi-card" style="border-color:#38bdf8;">
      <div class="kpi-title">Account Equity</div>
      <div class="kpi-val" id="kpi_equity" style="color:#38bdf8;">$100,000.00</div>
      <div class="kpi-sub" id="kpi_equity_sub">Initial: $100,000.00</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Cash Available</div>
      <div class="kpi-val" id="kpi_cash" style="color:#34d399;">$100,000.00</div>
      <div class="kpi-sub" id="kpi_cash_sub">100.0% unallocated</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Invested Capital</div>
      <div class="kpi-val" id="kpi_invested">$0.00</div>
      <div class="kpi-sub" id="kpi_open_count">0 Open Positions</div>
    </div>
    <div class="kpi-card" style="border-color:#f59e0b;">
      <div class="kpi-title">Open Portfolio Heat</div>
      <div class="kpi-val" id="kpi_heat" style="color:#fbbf24;">0.00 R</div>
      <div class="kpi-sub" id="kpi_heat_sub">$0.00 at risk (0.0%)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Unrealized P&amp;L</div>
      <div class="kpi-val" id="kpi_unrealized">+0.00 R</div>
      <div class="kpi-sub" id="kpi_unrealized_sub">+$0.00 Open</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Realized P&amp;L &amp; Win Rate</div>
      <div class="kpi-val" id="kpi_realized">+0.00 R</div>
      <div class="kpi-sub" id="kpi_win_rate">0 Trades (0.0% Win)</div>
    </div>
    <button onclick="openPortfolioSettingsModal()" style="background:#1e293b; border:1px solid #334155; color:#cbd5e1; border-radius:6px; padding:10px 14px; font-weight:700; font-size:12px; cursor:pointer; height:fit-content; margin-left:auto;">
      ⚙️ Portfolio Settings
    </button>
  </div>

  <!-- Real-Time Automated Execution Alerts Ribbon -->
  <div class="alerts-container" id="portfolio_alerts_container">
    <div style="background:#0f1523; border:1px solid #1e293b; border-radius:6px; padding:10px 16px; color:#94a3b8; font-size:12px; display:flex; align-items:center; gap:8px;">
      <span>🛡️</span>
      <span>No execution alerts pending. All open positions are respecting risk parameters.</span>
    </div>
  </div>

  <!-- Holdings & Closed Trades Tables -->
  <div class="portfolio-section">
    <!-- Active Holdings -->
    <div>
      <div class="sec-heading">
        <div style="display:flex; align-items:center; gap:8px;">
          <span>💼 Active EP Positions (<span id="open_pos_count">0</span>)</span>
          <span style="font-size:11px; font-weight:400; color:var(--muted);">1.0 R Risk per Trade | Invalidation strictly at Day 1 Low</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
          <button class="nav-btn" onclick="openAddTickerModal()" style="background:#0284c7; border-color:#38bdf8; color:#fff; font-size:11.5px; padding:4px 10px; display:flex; align-items:center; gap:5px;">
            <span>➕</span> Add Ticker to Portfolio
          </button>
        </div>
      </div>
      <div style="overflow-x:auto;">
        <table class="port-table">
          <thead>
            <tr>
              <th class="tl">Symbol</th>
              <th class="tl">Sector &amp; Theme</th>
              <th>Entry Date</th>
              <th>Entry Px</th>
              <th>Current Px</th>
              <th>Shares</th>
              <th>Invested ($)</th>
              <th>Active Stop</th>
              <th>Open (R)</th>
              <th>Unrealized P&amp;L</th>
              <th class="tl">Lifecycle Status</th>
              <th style="text-align:center;">Actions</th>
            </tr>
          </thead>
          <tbody id="port_positions_body">
            <tr>
              <td colspan="12" style="text-align:center; padding:30px; color:var(--muted);">
                No active positions in portfolio.<br>
                <span style="font-size:11px; color:#38bdf8; margin-top:4px; display:inline-block;">Go to "EP Scanner &amp; Catalysts" and click "Add to Live Portfolio" on any setup.</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Closed Trades History -->
    <div style="margin-top:10px;">
      <div class="sec-heading">
        <div style="display:flex; align-items:center; gap:8px;">
          <span>📜 Closed Trades Ledger (<span id="closed_pos_count">0</span>)</span>
          <span style="font-size:11px; font-weight:400; color:var(--muted);">Full Historical R-Multiple Audit</span>
        </div>
        <div>
          <button class="nav-btn" onclick="clearAllClosedPositions()" style="background:#271216; border-color:#991b1b; color:#fca5a5; font-size:10.5px; padding:3px 8px; cursor:pointer;" title="Clear entire closed trades history and reset tally">
            🗑️ Clear History
          </button>
        </div>
      </div>
      <div style="overflow-x:auto;">
        <table class="port-table">
          <thead>
            <tr>
              <th class="tl">Symbol</th>
              <th class="tl">Sector &amp; Theme</th>
              <th>Entry Date</th>
              <th>Exit Date</th>
              <th>Hold Days</th>
              <th>Entry Px</th>
              <th>Exit Px</th>
              <th>Shares</th>
              <th>Net R</th>
              <th>Realized P&amp;L ($)</th>
              <th class="tl">Exit Trigger / Reason</th>
              <th style="text-align:center;">Action</th>
            </tr>
          </thead>
          <tbody id="port_closed_body">
            <tr>
              <td colspan="12" style="text-align:center; padding:20px; color:var(--muted);">
                No closed trades yet.
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<!-- DEDICATED PREMARKET RADAR & EARLY INTRADAY TACTICAL VIEW -->
<div id="premarket_view" class="portfolio-container" style="display:none;">
  <!-- Premarket Header KPI Strip -->
  <div class="portfolio-header-kpi">
    <div class="kpi-card" style="border-color:#f59e0b;">
      <div class="kpi-title">Candidate Universe</div>
      <div class="kpi-val" id="pm_kpi_scanned" style="color:#fbbf24;">0</div>
      <div class="kpi-sub">Top Themes, Sectors &amp; Core EPs</div>
    </div>
    <div class="kpi-card" style="border-color:#10b981;">
      <div class="kpi-title">Potential Elite EPs</div>
      <div class="kpi-val" id="pm_kpi_elite" style="color:#34d399;">0</div>
      <div class="kpi-sub">Gap &ge; +4%, Proj RVOL &ge; 2.0x</div>
    </div>
    <div class="kpi-card" style="border-color:#38bdf8;">
      <div class="kpi-title">Overnight Gappers</div>
      <div class="kpi-val" id="pm_kpi_gappers" style="color:#38bdf8;">0</div>
      <div class="kpi-sub">Watchlist Movers (Awaiting Pace)</div>
    </div>
    <div class="kpi-card" style="border-color:#8b5cf6;">
      <div class="kpi-title">Leading Emerging Themes</div>
      <div class="kpi-val" id="pm_kpi_themes" style="font-size:12px; color:#c4b5fd; font-weight:700; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">Loading...</div>
      <div class="kpi-sub">Top Institutional Capital Flow</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Feed Status</div>
      <div class="kpi-val" id="pm_kpi_status" style="font-size:14px; color:#94a3b8;">Live 15m Feed</div>
      <div class="kpi-sub" id="pm_kpi_updated">Awaiting scan</div>
    </div>
  </div>

  <!-- Premarket Action Bar -->
  <div style="background:#0f1420; border-bottom:1px solid var(--border); padding:8px 18px; display:flex; justify-content:space-between; align-items:center; flex-shrink:0;">
    <div style="display:flex; align-items:center; gap:10px;">
      <span style="font-size:12px; font-weight:700; color:#fbbf24;">🌅 Early Detection Radar:</span>
      <span style="font-size:11.5px; color:#cbd5e1;">Premarket 15m Price &amp; Volume Pacing + SEC EDGAR 8-K Intelligence</span>
    </div>
    <div style="display:flex; align-items:center; gap:8px;">
      <button class="btn-refresh" id="pm_btn_rescan" onclick="updatePremarketScan()" style="background:#2e1065; border-color:#7c3aed; color:#c4b5fd; font-weight:700;">
        ⚡ Premarket Update / Rescan
      </button>
      <button class="btn-export" onclick="downloadPremarketAiJson()">
        🤖 Export Premarket AI JSON
      </button>
      <button class="btn-export" onclick="copyPremarketPrompt()" style="border-color:#10b981; color:#34d399;">
        📋 Copy AI Prompt
      </button>
    </div>
  </div>

  <!-- Premarket Content Sections -->
  <div class="portfolio-section">
    <!-- SECTION 1: Potential Elite EPs Forming Today -->
    <div>
      <div class="sec-heading">
        <span style="color:#34d399; display:flex; align-items:center; gap:6px;">
          <span>🔥 Potential Elite EPs Forming Today (Early Tactical Execution Pipeline)</span>
          <span class="badge" style="background:#059669;" id="pm_badge_elite_count">0</span>
        </span>
        <span style="font-size:11px; color:var(--muted); font-weight:500;">
          Historical Day 1 Open-to-Close Mean Gain: <b>+10.39%</b> | Win Rate: <b>81.9%</b> vs 61.7% EOD Close
        </span>
      </div>
      <div style="background:#131824; border:1px solid var(--border); border-radius:6px; overflow:hidden;">
        <table class="port-table">
          <thead>
            <tr>
              <th class="tl">Symbol &amp; Sector / Theme</th>
              <th>Prior Close</th>
              <th>Premarket Px</th>
              <th>Overnight Gap %</th>
              <th>Premarket Vol</th>
              <th>Projected RVOL</th>
              <th>SEC 8-K / Catalyst Info</th>
              <th class="tl">Recommended Early Tactic</th>
              <th>Max Cap Cap</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody id="pm_elite_tbody">
            <tr>
              <td colspan="12" style="text-align:center; padding:25px; color:var(--muted);">
                Click <b>"⚡ Premarket Update"</b> to pull live 15m intraday feeds for in-play sectors and themes.
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- SECTION 2: Overnight Gappers & Watchlist Movers -->
    <div>
      <div class="sec-heading">
        <span style="color:#fbbf24; display:flex; align-items:center; gap:6px;">
          <span>⚡ Overnight Gappers &amp; Watchlist Movers (Awaiting Volume Pace)</span>
          <span class="badge" style="background:#b45309;" id="pm_badge_gappers_count">0</span>
        </span>
        <span style="font-size:11px; color:var(--muted); font-weight:500;">
          ⚠️ <b>64.5% Blind Gapper Trap</b>: Without projected RVOL &ge; 2.5x and theme tailwinds, morning gaps fade into red candles.
        </span>
      </div>
      <div style="background:#131824; border:1px solid var(--border); border-radius:6px; overflow:hidden;">
        <table class="port-table">
          <thead>
            <tr>
              <th class="tl">Symbol &amp; Theme</th>
              <th>Prior Close</th>
              <th>Premarket Px</th>
              <th>Overnight Gap %</th>
              <th>Premarket Vol</th>
              <th>Projected RVOL</th>
              <th>SEC 8-K Filing</th>
              <th class="tl">Strategy Insight &amp; Condition Diagnosis</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody id="pm_gappers_tbody">
            <tr>
              <td colspan="8" style="text-align:center; padding:25px; color:var(--muted);">
                No overnight gappers loaded.
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- SECTION 3: Tactical Playbooks & AI Prompt Station -->
    <div style="display:grid; grid-template-columns:1fr 1fr 1.2fr; gap:12px; margin-top:4px;">
      <!-- Playbook 1: 15m ORB -->
      <div class="card" style="border-color:#0284c7;">
        <div class="card-title" style="color:#38bdf8;">
          <span>🎯 Tactical Playbook 1: 15m ORB</span>
          <span class="badge" style="background:#0284c7;">Gaps +4% to +12%</span>
        </div>
        <div style="font-size:11.5px; color:#cbd5e1; line-height:1.5;">
          <b>Why it works:</b> Lets the initial 09:30–09:45 EST 15m bar print to absorb early emotional volatility.
          <ul style="margin:6px 0 0 16px;">
            <li><b>Entry Trigger:</b> Buy-Stop Limit 1¢ above 15m High.</li>
            <li><b>Hard Invalidation:</b> Stop-Loss 1¢ below 15m Low.</li>
            <li><b>Asymmetric Edge:</b> Average risk is only <b>2.8%</b> (vs 8.2% at EOD close), enabling greater share count and massive R-multiple expansion.</li>
          </ul>
        </div>
      </div>

      <!-- Playbook 2: Morning Washout & VWAP Reclaim -->
      <div class="card" style="border-color:#8b5cf6;">
        <div class="card-title" style="color:#c084fc;">
          <span>🌊 Tactical Playbook 2: VWAP Reclaim</span>
          <span class="badge" style="background:#6d28d9;">Extreme Gaps &gt; +12%</span>
        </div>
        <div style="font-size:11.5px; color:#cbd5e1; line-height:1.5;">
          <b>Why it works:</b> Big gappers face heavy premarket profit-taking at the open. Chasing at 09:30 is suicide.
          <ul style="margin:6px 0 0 16px;">
            <li><b>Wait for Washout:</b> Let early sellers flush the stock down during 09:30–09:45.</li>
            <li><b>Entry Trigger:</b> 5-minute candle close reclaiming session VWAP from below on expanding volume.</li>
            <li><b>Hard Stop:</b> Swing low of the morning washout flush. Completely avoids the "Gap-and-Crap" faders.</li>
          </ul>
        </div>
      </div>

      <!-- Station 3: External AI Agent Station -->
      <div class="card" style="border-color:#10b981;">
        <div class="card-title" style="color:#34d399;">
          <span>🤖 AI Agent Premarket Execution Station</span>
          <button class="booster-reset" onclick="copyPremarketPrompt()" style="border-color:#10b981; color:#34d399; font-weight:700;">
            📋 Copy Prompt
          </button>
        </div>
        <div style="font-size:11px; color:#94a3b8; margin-bottom:4px;">
          Pass this prompt along with <a href="/api/export/premarket_json" target="_blank" style="color:#38bdf8; text-decoration:none;">/api/export/premarket_json</a> to external AI agents:
        </div>
        <textarea id="pm_prompt_text" readonly style="width:100%; height:90px; background:#0b0e14; border:1px solid #1e293b; color:#cbd5e1; font-size:10px; font-family:monospace; padding:6px; border-radius:4px; resize:none;">You are the Lead Quantitative Execution Trader specializing in Premarket Episodic Pivot (EP) Ignitions and Opening Range Breakouts (ORB).
Evaluate this morning's premarket candidate cohort, eliminate the 64.5% "Gap-and-Crap" fader traps, and construct a precise, execution-ready Intraday Opening Game Plan (09:15 – 10:15 EST) for high-expectancy capital deployment using the 15m ORB and VWAP Reclaim protocols with asymmetric biotech risk caps.</textarea>
      </div>
    </div>
  </div>
</div>

<!-- Modal 1: Add to Live Portfolio -->
<div class="modal-overlay" id="modal_add_portfolio" onclick="closeModalOnOutside(event, 'modal_add_portfolio')">
  <div class="modal-box" onclick="event.stopPropagation()">
    <div class="modal-head">
      <div class="modal-head-title">💼 Add EP Position to Live Portfolio</div>
      <button class="drawer-close" onclick="closeModal('modal_add_portfolio')">&times;</button>
    </div>
    <div class="modal-body">
      <div id="m_add_input_ticker_row" style="margin-bottom:10px;">
        <div class="input-label">Ticker Symbol (Enter any stock, e.g. INTR, STNE, PTC, RXO, GME)</div>
        <div style="display:flex; gap:8px;">
          <input type="text" id="m_add_ticker_input" class="input-field" placeholder="e.g. STNE" style="text-transform:uppercase; font-weight:700;" onkeydown="if(event.key==='Enter')fetchCustomTickerQuote()">
          <button class="nav-btn" onclick="fetchCustomTickerQuote()" style="background:#0284c7; color:#fff; border-color:#38bdf8; padding:0 14px; font-weight:700;">Lookup</button>
        </div>
      </div>

      <div style="background:#0b0e14; border:1px solid #1e293b; border-radius:6px; padding:10px 12px; display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
        <div>
          <span style="font-size:15px; font-weight:800; color:#fff;" id="m_add_sym">XYZ</span>
          <span style="font-size:11px; color:var(--muted); margin-left:6px;" id="m_add_comp">Company Name</span>
        </div>
        <div style="font-size:11px; color:#38bdf8; font-weight:700;" id="m_add_meta">Sector &bull; Theme</div>
      </div>

      <div class="input-row">
        <div class="input-col">
          <div class="input-label">Entry Price ($)</div>
          <input type="number" step="0.01" id="m_add_entry" class="input-field" oninput="recalcAddModal()">
        </div>
        <div class="input-col">
          <div class="input-label">Stop Loss ($) [Day 1 Low]</div>
          <input type="number" step="0.01" id="m_add_stop" class="input-field" oninput="recalcAddModal()">
        </div>
      </div>

      <div class="input-row">
        <div class="input-col">
          <div class="input-label">Risk % of Portfolio (R)</div>
          <input type="number" step="0.1" min="0.1" max="10.0" id="m_add_risk_pct" class="input-field" oninput="recalcAddModal()">
        </div>
        <div class="input-col">
          <div class="input-label">Dollar Risk ($)</div>
          <input type="text" id="m_add_risk_dollars" class="input-field" readonly style="color:#f87171; background:#11151f;">
        </div>
      </div>

      <div class="input-row">
        <div class="input-col">
          <div class="input-label">Suggested Shares</div>
          <input type="number" step="1" id="m_add_shares" class="input-field" oninput="recalcAddFromShares()">
        </div>
        <div class="input-col">
          <div class="input-label">Total Capital Required ($)</div>
          <input type="text" id="m_add_capital" class="input-field" readonly style="color:#38bdf8; background:#11151f;">
        </div>
      </div>

      <div id="m_add_gap_warning" style="display:none; margin-bottom:12px; font-size:11px; padding:8px 12px; border-radius:6px; font-weight:600; line-height:1.4;"></div>

      <div>
        <div class="input-label">Trade Execution Notes / Rationale</div>
        <input type="text" id="m_add_notes" class="input-field" placeholder="e.g. Grade A Catalyst in Power Cluster">
      </div>
    </div>
    <div class="modal-foot">
      <button class="view-tab" onclick="closeModal('modal_add_portfolio')">Cancel</button>
      <button class="nav-btn active" onclick="submitAddToPortfolio()" style="background:#0284c7; border-color:#38bdf8; color:#fff;">
        Confirm &amp; Execute Position
      </button>
    </div>
  </div>
</div>

<!-- Modal 2: Modify Position (Entry, Stop Loss, Position Size) -->
<div class="modal-overlay" id="modal_adjust_stop" onclick="closeModalOnOutside(event, 'modal_adjust_stop')">
  <div class="modal-box" onclick="event.stopPropagation()">
    <div class="modal-head">
      <div class="modal-head-title">⚙️ Modify Position: <span id="m_adj_sym">XYZ</span></div>
      <button class="drawer-close" onclick="closeModal('modal_adjust_stop')">&times;</button>
    </div>
    <div class="modal-body">
      <input type="hidden" id="m_adj_pos_id">
      <div style="font-size:11.5px; color:var(--muted); line-height:1.4; margin-bottom:8px;">
        Current Px: <b id="m_adj_curr_px" style="color:#34d399;">$0.00</b> | Stop: <b id="m_adj_curr_stop" style="color:#ef4444;">$0.00</b> | Open R: <b id="m_adj_curr_r" style="color:#38bdf8;">0.0 R</b>
      </div>
      
      <div class="input-row">
        <div class="input-col">
          <div class="input-label">Entry Price ($)</div>
          <input type="number" step="0.01" id="m_adj_entry_input" class="input-field" oninput="recalcModifyModal()">
        </div>
        <div class="input-col">
          <div class="input-label">Stop Loss ($)</div>
          <input type="number" step="0.01" id="m_adj_new_stop" class="input-field" oninput="recalcModifyModal()">
        </div>
      </div>

      <div class="input-row">
        <div class="input-col">
          <div class="input-label">Position Size (Shares)</div>
          <input type="number" step="1" min="1" id="m_adj_shares_input" class="input-field" oninput="recalcModifyModal()">
        </div>
        <div class="input-col">
          <div class="input-label">Total Capital Invested ($)</div>
          <input type="text" id="m_adj_capital_val" class="input-field" readonly style="color:#38bdf8; background:#11151f;">
        </div>
      </div>

      <div class="input-row">
        <div class="input-col">
          <div class="input-label">Calculated Dollar Risk ($)</div>
          <input type="text" id="m_adj_risk_val" class="input-field" readonly style="color:#f87171; background:#11151f;">
        </div>
        <div class="input-col">
          <div class="input-label">Risk Distance (%)</div>
          <input type="text" id="m_adj_dist_val" class="input-field" readonly style="color:#fbbf24; background:#11151f;">
        </div>
      </div>

      <div>
        <div class="input-label">Quick Stop Rules (10-Yr Study Insights):</div>
        <div style="display:flex; gap:6px; flex-wrap:wrap; margin-top:4px;">
          <button class="booster-chip" onclick="setQuickStop('be')" id="btn_quick_be">🛡️ Breakeven (<span id="q_be_val">$0.00</span>)</button>
          <button class="booster-chip" onclick="setQuickStop('ema20')" id="btn_quick_ema20">📉 20 EMA (<span id="q_ema20_val">$0.00</span>)</button>
          <button class="booster-chip" onclick="setQuickStop('sma50')" id="btn_quick_sma50">🏛️ 50 SMA (<span id="q_sma50_val">$0.00</span>)</button>
        </div>
      </div>
      <div id="m_adj_tranches_section" style="display:none; margin-top:10px; background:#0b0e14; border:1px solid #1e293b; border-radius:6px; padding:8px 10px;">
        <div style="font-size:11px; font-weight:700; color:#c084fc; margin-bottom:6px;">⚡ Executed Add Tranches &amp; Stops:</div>
        <div id="m_adj_tranches_list" style="display:flex; flex-direction:column; gap:6px;"></div>
      </div>
      <div style="margin-top:8px;">
        <div class="input-label">Adjustment Reason / Notes</div>
        <input type="text" id="m_adj_notes" class="input-field" placeholder="e.g. Adjusted position size and trailed stop to rising 20 EMA">
      </div>
    </div>
    <div class="modal-foot">
      <button class="view-tab" onclick="closeModal('modal_adjust_stop')">Cancel</button>
      <button class="nav-btn active" onclick="submitModifyPosition()" style="background:#0284c7; border-color:#38bdf8; color:#fff;">Save Position Changes</button>
    </div>
  </div>
</div>

<!-- Modal 3: Secondary Add (+50% Size) -->
<div class="modal-overlay" id="modal_secondary_add" onclick="closeModalOnOutside(event, 'modal_secondary_add')">
  <div class="modal-box" onclick="event.stopPropagation()">
    <div class="modal-head">
      <div class="modal-head-title">⚡ Day 2 Secondary Add (+50%) for <span id="m_add_sec_sym">XYZ</span></div>
      <button class="drawer-close" onclick="closeModal('modal_secondary_add')">&times;</button>
    </div>
    <div class="modal-body">
      <input type="hidden" id="m_add_sec_pos_id">
      <div style="background:#0b0e14; border:1px solid #1e293b; border-radius:6px; padding:10px 12px; font-size:11.5px; color:#cbd5e1; line-height:1.45;">
        • Price broke above Day 1 High (<b id="m_sec_d1_high" style="color:#38bdf8;">$0.00</b>), validating institutional accumulation.<br>
        • Canonical Rule: Scale <b>+50% size</b> at breakout and lock stop to protect capital.
      </div>
      <div class="input-row">
        <div class="input-col">
          <div class="input-label">Add Fill Price ($)</div>
          <input type="number" step="0.01" id="m_sec_price" class="input-field" oninput="recalcSecondaryAddModal()">
        </div>
        <div class="input-col">
          <div class="input-label">Add Stop Loss ($)</div>
          <input type="number" step="0.01" id="m_sec_stop" class="input-field" oninput="recalcSecondaryAddModal()">
        </div>
      </div>
      <div class="input-row">
        <div class="input-col">
          <div class="input-label">Add Shares (+50% of initial)</div>
          <input type="number" step="1" id="m_sec_shares" class="input-field" oninput="recalcSecondaryAddModal()">
        </div>
        <div class="input-col">
          <div class="input-label">Add Risk ($ &amp; %)</div>
          <input type="text" id="m_sec_risk_feedback" class="input-field" readonly style="color:#f87171; background:#11151f;">
        </div>
      </div>
      <div>
        <div class="input-label">Notes</div>
        <input type="text" id="m_sec_notes" class="input-field" value="Day 2 breakout above D1 High confirmed">
      </div>
    </div>
    <div class="modal-foot">
      <button class="view-tab" onclick="closeModal('modal_secondary_add')">Cancel</button>
      <button class="nav-btn active" onclick="submitSecondaryAdd()" style="background:#f59e0b; color:#000; border-color:#fbbf24;">
        Execute Secondary Add
      </button>
    </div>
  </div>
</div>

<!-- Modal 4: Close Position -->
<div class="modal-overlay" id="modal_close_trade" onclick="closeModalOnOutside(event, 'modal_close_trade')">
  <div class="modal-box" onclick="event.stopPropagation()">
    <div class="modal-head">
      <div class="modal-head-title">✕ Close EP Position: <span id="m_close_sym">XYZ</span></div>
      <button class="drawer-close" onclick="closeModal('modal_close_trade')">&times;</button>
    </div>
    <div class="modal-body">
      <input type="hidden" id="m_close_pos_id">
      <div class="input-row">
        <div class="input-col">
          <div class="input-label">Exit Price ($)</div>
          <input type="number" step="0.01" id="m_close_px" class="input-field">
        </div>
        <div class="input-col">
          <div class="input-label">Exit Date</div>
          <input type="date" id="m_close_date" class="input-field">
        </div>
      </div>
      <div>
        <div class="input-label">Exit Trigger / Strategy Reason</div>
        <select id="m_close_reason" class="input-field" style="background:#0a0d14;">
          <option value="🛑 Stop Loss Hit">🛑 Stop Loss Hit (Day 1 Low breached)</option>
          <option value="🔵 Larsson Blue Flip Exit">🔵 Larsson Blue Flip Exit (Trend momentum broken)</option>
          <option value="📉 20 EMA Close Violation">📉 20 EMA Close Violation (Swing exit)</option>
          <option value="🏛️ 50 SMA Institutional Violation">🏛️ 50 SMA Institutional Violation (Multi-quarter exit)</option>
          <option value="🎯 Profit Target / De-risking">🎯 Profit Target / Discretionary De-risking</option>
          <option value="Manual Exit">Manual Exit / Discretionary</option>
        </select>
      </div>
      <div>
        <div class="input-label">Trade Review Notes</div>
        <input type="text" id="m_close_notes" class="input-field" placeholder="e.g. Captured runner; exited on first daily blue flip">
      </div>
    </div>
    <div class="modal-foot">
      <button class="view-tab" onclick="closeModal('modal_close_trade')">Cancel</button>
      <button class="nav-btn active" onclick="submitCloseTrade()" style="background:#ef4444; border-color:#f87171; color:#fff;">
        Confirm Close Trade
      </button>
    </div>
  </div>
</div>

<!-- Modal 5: Portfolio Settings -->
<div class="modal-overlay" id="modal_portfolio_settings" onclick="closeModalOnOutside(event, 'modal_portfolio_settings')">
  <div class="modal-box" onclick="event.stopPropagation()">
    <div class="modal-head">
      <div class="modal-head-title">⚙️ EP Portfolio Configuration</div>
      <button class="drawer-close" onclick="closeModal('modal_portfolio_settings')">&times;</button>
    </div>
    <div class="modal-body">
      <div>
        <div class="input-label">Total Portfolio Capital ($)</div>
        <input type="number" step="1000" id="m_set_port_size" class="input-field">
      </div>
      <div>
        <div class="input-label">Default Risk per Trade (% of Portfolio = 1.0 R)</div>
        <input type="number" step="0.1" min="0.1" max="10.0" id="m_set_risk_pct" class="input-field">
        <div style="font-size:10.5px; color:var(--muted); margin-top:3px;">
          e.g. 1.0% on $100,000 = $1,000 max risk per setup.
        </div>
      </div>
    </div>
    <div class="modal-foot">
      <button class="view-tab" onclick="closeModal('modal_portfolio_settings')">Cancel</button>
      <button class="nav-btn active" onclick="submitPortfolioSettings()">Save Settings</button>
    </div>
  </div>
</div>

<!-- Modal 6: Ticker Insights & TradingView Alert Setup Generator -->
<div class="modal-overlay" id="modal_ticker_insights" onclick="closeModalOnOutside(event, 'modal_ticker_insights')">
  <div class="modal-box" style="max-width:680px;" onclick="event.stopPropagation()">
    <div class="modal-head">
      <div>
        <div class="modal-head-title">📊 <span id="m_ins_sym">XYZ</span> — Quantitative Insights &amp; Alert Setup</div>
        <div style="font-size:11px; color:var(--muted);" id="m_ins_comp">Company Name</div>
      </div>
      <button class="drawer-close" onclick="closeModal('modal_ticker_insights')">&times;</button>
    </div>
    <div class="modal-body" style="max-height:75vh; overflow-y:auto;">
      
      <!-- Summary Chips -->
      <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px;">
        <span class="badge" id="m_ins_badge_stage" style="background:#0284c7; padding:4px 8px; font-size:11px;">Stage</span>
        <span class="badge" id="m_ins_badge_theme" style="background:#2e1065; color:#c4b5fd; padding:4px 8px; font-size:11px;">Theme</span>
        <span class="badge" id="m_ins_badge_risk" style="background:#1e293b; color:#fbbf24; padding:4px 8px; font-size:11px;">R-Multiple</span>
      </div>

      <!-- Key Quantitative Levels -->
      <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:8px; margin-bottom:14px; background:#0b0e14; border:1px solid #1e293b; padding:10px; border-radius:6px;">
        <div>
          <div style="font-size:10px; color:var(--muted);">Entry Price</div>
          <div style="font-size:14px; font-weight:700; color:#fff;" id="m_ins_entry">$0.00</div>
        </div>
        <div>
          <div style="font-size:10px; color:var(--muted);">Current Stop</div>
          <div style="font-size:14px; font-weight:700; color:#ef4444;" id="m_ins_stop">$0.00</div>
        </div>
        <div>
          <div style="font-size:10px; color:var(--muted);">Day 1 High (Add)</div>
          <div style="font-size:14px; font-weight:700; color:#38bdf8;" id="m_ins_d1_high">$0.00</div>
        </div>
        <div>
          <div style="font-size:10px; color:var(--muted);">Target +3R</div>
          <div style="font-size:14px; font-weight:700; color:#34d399;" id="m_ins_target_3r">$0.00</div>
        </div>
      </div>

      <!-- Insights & Execution Guide -->
      <div style="background:#0a0d14; border-left:3px solid #38bdf8; padding:12px 14px; border-radius:4px; margin-bottom:14px; font-size:12px; line-height:1.5; color:#cbd5e1;">
        <div style="font-weight:700; color:#fff; margin-bottom:6px; display:flex; align-items:center; gap:6px;">
          <span>🎯 Handling Plan (Based on 10-Yr Study Insights):</span>
        </div>
        <div id="m_ins_guide_content" style="white-space:pre-line;">
          • Loading quantitative playbook...
        </div>
      </div>

      <!-- TradingView Pre-Configured Alerts Generator -->
      <div>
        <div style="font-weight:700; font-size:12.5px; color:#fff; margin-bottom:8px; display:flex; align-items:center; gap:6px;">
          <span>🔔 Recommended TradingView Alerts:</span>
          <span style="font-size:10.5px; color:var(--muted); font-weight:400;">(Copy &amp; paste directly into TradingView)</span>
        </div>

        <div style="display:flex; flex-direction:column; gap:8px;" id="m_ins_tv_alerts_list">
          <!-- Populated dynamically via JS -->
        </div>
      </div>

    </div>
    <div class="modal-foot">
      <button class="view-tab" onclick="closeModal('modal_ticker_insights')">Close</button>
      <button class="nav-btn" onclick="openModifyFromInsights()" style="background:#0284c7; border-color:#38bdf8; color:#fff;">
        ⚙️ Modify Holding
      </button>
    </div>
  </div>
</div>

<!-- Slide-Over Drawer: Emerging Sectors & Themes -->
<div class="drawer-overlay" id="drawer_overlay" onclick="closeDrawerOnOutside(event)">
  <div class="drawer-panel" onclick="event.stopPropagation()">
    <div class="drawer-header">
      <div>
        <div class="drawer-title">🌐 Emerging Sectors &amp; Thematic Momentum Engine</div>
        <div style="font-size:11px; color:var(--muted); margin-top:2px;">
          Multi-Timeframe Percentile Ranking (1W, 1M, 3M, YTD) + 10-Yr EP Incubator Edge Tracking
        </div>
      </div>
      <button class="drawer-close" onclick="closeEmergingDrawer()">&times;</button>
    </div>
    
    <div class="drawer-tabs">
      <button class="d-tab active" id="dtab_themes" onclick="switchDrawerTab('themes')">🔥 Top Emerging Themes</button>
      <button class="d-tab" id="dtab_sectors" onclick="switchDrawerTab('sectors')">🏭 Top Emerging Sectors</button>
      <button class="d-tab" id="dtab_all_themes" onclick="switchDrawerTab('all_themes')">🌐 All Themes Leaderboard</button>
      <button class="d-tab" id="dtab_all_sectors" onclick="switchDrawerTab('all_sectors')">🏛️ All Sectors Leaderboard</button>
      <button class="d-tab" id="dtab_veto" onclick="switchDrawerTab('veto')" style="border-color:#ef4444; color:#fca5a5;">🚫 Severe Headwind Veto</button>
      <button class="d-tab" id="dtab_guide" onclick="switchDrawerTab('guide')">💡 Archetypes Guide</button>
    </div>

    <div class="drawer-content" id="drawer_body">
      <div style="text-align:center; padding:30px; color:var(--muted);">Loading emerging momentum board...</div>
    </div>
  </div>
</div>

<!-- Modal 7: AI Top Ranked EPs -->
<div class="modal-overlay" id="modal_ai_top_picks" onclick="closeModalOnOutside(event, 'modal_ai_top_picks')">
  <div class="modal-box" style="max-width:800px; background:#0f172a;" onclick="event.stopPropagation()">
    <div class="modal-head">
      <div>
        <div class="modal-head-title">🏆 AI Predictive Engine — Live Pipeline Analysis</div>
        <div style="font-size:11px; color:var(--muted);">Dynamically ranked by combined AI probability scores · Click any card to select &amp; chart</div>
      </div>
      <div style="display:flex; align-items:center; gap:8px;">
        <button style="background:#0284c7; color:#fff; border:none; padding:4px 10px; border-radius:4px; font-size:11.5px; font-weight:700; cursor:pointer;" onclick="closeModal('modal_ai_top_picks'); setView('top_ai');">📋 Filter Left Table</button>
        <button class="modal-close" onclick="closeModal('modal_ai_top_picks')">✕</button>
      </div>
    </div>
    
    <div class="modal-body" style="padding:15px; color:#e2e8f0; font-size:13px; max-height:70vh; overflow-y:auto;" id="ai_top_picks_content">
      <div style="text-align:center; padding:40px; color:#94a3b8;">Processing Live ML Inference...</div>
    </div>
  </div>
</div>

<script>
let currentView = 'all';
let allEvents = [];
let selectedEvent = null;
let chart = null, candleSeries = null, volumeSeries = null, ema20Series = null;
let activeLines = [];
let emergingBoard = null;
let currentDrawerTab = 'themes';
let drawerSortKey = 'score';
let drawerSortAsc = false;

let stratConfig = {
  use_trade1: true,
  use_ml_targets: true,
  use_ml_stop: true,
  use_multileg: true,
  use_multileg_ml: true
};

function onStrategyConfigChange() {
  const t1 = document.getElementById('cfg_trade1');
  const mt = document.getElementById('cfg_ml_targets');
  const ms = document.getElementById('cfg_ml_stop');
  const ml = document.getElementById('cfg_multileg');
  const mm = document.getElementById('cfg_multileg_ml');

  if (t1) stratConfig.use_trade1 = t1.checked;
  if (mt) stratConfig.use_ml_targets = mt.checked;
  if (ms) stratConfig.use_ml_stop = ms.checked;
  if (ml) stratConfig.use_multileg = ml.checked;
  if (mm) stratConfig.use_multileg_ml = mm.checked;

  const badge = document.getElementById('strategy_mode_badge');
  const btnBase = document.getElementById('btn_strat_baseline');
  const btnOpt = document.getElementById('btn_strat_optimal');
  
  const isBaseline = stratConfig.use_trade1 && !stratConfig.use_ml_targets && !stratConfig.use_ml_stop && !stratConfig.use_multileg && !stratConfig.use_multileg_ml;
  const isOptimal = stratConfig.use_trade1 && stratConfig.use_ml_targets && stratConfig.use_ml_stop && stratConfig.use_multileg && stratConfig.use_multileg_ml;

  if (isBaseline) {
    if (badge) badge.textContent = "Mechanical Baseline";
    if (btnBase) btnBase.classList.add('active-strat');
    if (btnOpt) btnOpt.classList.remove('active-strat');
  } else if (isOptimal) {
    if (badge) badge.textContent = "AI-Enhanced Optimal";
    if (btnOpt) btnOpt.classList.add('active-strat');
    if (btnBase) btnBase.classList.remove('active-strat');
  } else if (!stratConfig.use_trade1 && stratConfig.use_multileg) {
    if (badge) badge.textContent = "Continuation Only (Legs 2+3)";
    if (btnBase) btnBase.classList.remove('active-strat');
    if (btnOpt) btnOpt.classList.remove('active-strat');
  } else {
    if (badge) badge.textContent = "Custom Configuration";
    if (btnBase) btnBase.classList.remove('active-strat');
    if (btnOpt) btnOpt.classList.remove('active-strat');
  }

  renderTable();
  if (selectedEvent) selectEvent(selectedEvent);
}

function setStrategyPreset(preset) {
  const t1 = document.getElementById('cfg_trade1');
  const mt = document.getElementById('cfg_ml_targets');
  const ms = document.getElementById('cfg_ml_stop');
  const ml = document.getElementById('cfg_multileg');
  const mm = document.getElementById('cfg_multileg_ml');

  if (preset === 'baseline') {
    if (t1) t1.checked = true;
    if (mt) mt.checked = false;
    if (ms) ms.checked = false;
    if (ml) ml.checked = false;
    if (mm) mm.checked = false;
  } else if (preset === 'optimal') {
    if (t1) t1.checked = true;
    if (mt) mt.checked = true;
    if (ms) ms.checked = true;
    if (ml) ml.checked = true;
    if (mm) mm.checked = true;
  }
  onStrategyConfigChange();
}

function onArchetypeFilterChange() {
  renderTable();
}

let activeBoosters = {
  veto_headwind: false,
  novel_9m: false,
  cap10: false,
  rvol_sweet: false,
  '48h': false,
  elite_close: false,
  sma50: false,
  tailwind: false
};

function toggleBooster(name) {
  activeBoosters[name] = !activeBoosters[name];
  const chip = document.getElementById(`b_${name}`);
  if (chip) {
    if (activeBoosters[name]) chip.classList.add('active');
    else chip.classList.remove('active');
  }
  renderTable();
}

function resetBoosters() {
  for (const k in activeBoosters) {
    activeBoosters[k] = false;
    const chip = document.getElementById(`b_${k}`);
    if (chip) chip.classList.remove('active');
  }
  renderTable();
}

function toggleImpactGuide() {
  const panel = document.getElementById('impact_guide');
  panel.classList.toggle('open');
  const btn = document.getElementById('btn_toggle_guide');
  btn.textContent = panel.classList.contains('open') ? '📖 Close Strategy Impact Guide ▴' : '📖 Strategy Impact Guide & Stat Explanations ▾';
}

function updateLiveStatsBanner() {
  const list = getFilteredEvents();
  const banner = document.getElementById('filter_stat_banner');
  if (!banner) return;
  
  const count = list.length;
  let activeBoosterCount = Object.values(activeBoosters).filter(Boolean).length;
  
  let avgRvol = count > 0 ? (list.reduce((acc, e) => acc + (e.rvol || 0), 0) / count).toFixed(1) : '0';
  let activeHold = list.filter(e => e.held_d1_low).length;
  let holdPct = count > 0 ? ((activeHold / count) * 100).toFixed(0) : '0';

  // True empirical calculations from active filtered events and strategy configuration
  let executedTrades = [];
  for (let e of list) {
    if (stratConfig && stratConfig.use_trade1) {
      let t1_r = 0.0;
      if (!e.held_d1_low) {
        t1_r = -1.0;
      } else {
        t1_r = (e.curr_r !== null && e.curr_r !== undefined) ? Number(e.curr_r) : 0.0;
        if (stratConfig.use_ml_stop && t1_r >= 2.0) {
          t1_r = Math.max(0.0, t1_r);
        }
      }
      executedTrades.push(t1_r);
    }

    if (stratConfig && stratConfig.use_multileg && e.t2_info) {
      const t2 = e.t2_info;
      const isVetoed = stratConfig.use_multileg_ml && t2.ml_vetoed;
      if (t2.has_reentry && !isVetoed) {
        let t2_r = (t2.curr_r !== null && t2.curr_r !== undefined) ? Number(t2.curr_r) : 0.0;
        executedTrades.push(t2_r);
      }
    }
  }

  let realWinRate = '0.0';
  let realEV = '0.00';
  let realDrawdown = '0.0';
  let totalNetR = '0.0';
  let numTrades = executedTrades.length;

  if (numTrades > 0) {
    let wins = executedTrades.filter(r => r > 0);
    realWinRate = ((wins.length / numTrades) * 100).toFixed(1);
    let sumR = executedTrades.reduce((acc, r) => acc + r, 0);
    realEV = (sumR / numTrades).toFixed(2);
    totalNetR = sumR.toFixed(1);

    let cum = 0;
    let peak = 0;
    let maxDd = 0;
    for (let r of executedTrades) {
      cum += r;
      if (cum > peak) peak = cum;
      let dd = cum - peak;
      if (dd < maxDd) maxDd = dd;
    }
    realDrawdown = maxDd.toFixed(1);
  }

  const evColor = Number(realEV) >= 0 ? 'var(--green)' : 'var(--red)';
  const ddColor = Number(realDrawdown) >= -10 ? 'var(--green)' : 'var(--gold)';
  const netColor = Number(totalNetR) >= 0 ? 'var(--green)' : 'var(--red)';

  banner.innerHTML = `
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
      <div>
        <span style="font-weight:700; color:#fff;">Showing ${count} of ${allEvents.length} EPs</span>
        <span style="color:var(--muted); font-size:11px; margin-left:6px;">(${activeBoosterCount} Active Booster${activeBoosterCount === 1 ? '' : 's'})</span>
      </div>
      <div style="display:flex; gap:12px; font-size:11px; flex-wrap:wrap;">
        <span>Avg RVOL: <b style="color:var(--cyan);">${avgRvol}x</b></span>
        <span>Holding Stop: <b style="color:var(--green);">${activeHold}/${count} (${holdPct}%)</b></span>
        <span>Active Trades: <b style="color:#fff;">${numTrades}</b></span>
        <span>Expectancy: <b style="color:${evColor};">${Number(realEV) >= 0 ? '+' : ''}${realEV} R</b></span>
        <span>Max Drawdown: <b style="color:${ddColor};">${realDrawdown} R</b></span>
        <span>Win Rate: <b style="color:var(--cyan);">${realWinRate}%</b></span>
        <span>Total Net: <b style="color:${netColor};">${Number(totalNetR) >= 0 ? '+' : ''}${totalNetR} R</b></span>
      </div>
    </div>
  `;
}

function initChart() {
  if (typeof LightweightCharts === 'undefined') {
    setTimeout(initChart, 150);
    return;
  }
  const container = document.getElementById('chart-container');
  if (!container || chart) return;

  chart = LightweightCharts.createChart(container, {
    layout: { background: { color: '#0c0f16' }, textColor: '#cbd5e1' },
    grid: { vertLines: { color: '#1a2233' }, horzLines: { color: '#1a2233' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: {
      borderColor: '#232936',
      autoScale: true,
      scaleMargins: { top: 0.08, bottom: 0.25 },
      entireTextOnly: true
    },
    timeScale: {
      borderColor: '#232936',
      timeVisible: true,
      visible: true,
      borderVisible: true,
      fixLeftEdge: false,
      fixRightEdge: false,
      rightOffset: 12,
      shiftVisibleRangeOnNewBar: true
    }
  });
  candleSeries = chart.addCandlestickSeries({
    upColor: '#10b981', downColor: '#ef4444',
    borderUpColor: '#10b981', borderDownColor: '#ef4444',
    wickUpColor: '#10b981', wickDownColor: '#ef4444',
    priceFormat: { type: 'price', precision: 2, minMove: 0.01 }
  });
  volumeSeries = chart.addHistogramSeries({
    priceFormat: { type: 'volume' },
    priceScaleId: 'volume',
    priceLineVisible: false
  });
  chart.priceScale('volume').applyOptions({
    scaleMargins: { top: 0.84, bottom: 0.0 }
  });
  ema20Series = chart.addLineSeries({ color: '#38bdf8', lineWidth: 1.5, title: '20 EMA' });

  const ro = new ResizeObserver(entries => {
    for (const entry of entries) {
      const w = Math.floor(entry.contentRect.width);
      const h = Math.floor(entry.contentRect.height);
      if (w > 0 && h > 0) {
        chart.applyOptions({ width: w, height: h });
      }
    }
  });
  ro.observe(container);

  if (selectedEvent) {
    loadChartData(selectedEvent);
  }
}

function setView(view) {
  currentView = view;
  document.querySelectorAll('.view-tab').forEach(b => b.classList.remove('active'));
  const tabEl = document.getElementById(`tab_${view}`);
  if (tabEl) tabEl.classList.add('active');
  if (view === 'top_ai') {
    const archFilter = document.getElementById('f_trader_archetype') || document.getElementById('sel_archetype_filter');
    if (archFilter && archFilter.value !== '') {
      archFilter.value = '';
    }
  }
  renderTable();
}

function fetchEvents() {
  fetch('/api/tracker_events')
    .then(r => r.json())
    .then(data => {
      allEvents = Array.isArray(data) ? data : (data.events || []);
      document.getElementById('cnt_all').textContent = allEvents.length;
      const topAiCnt = document.getElementById('cnt_top_ai');
      const activeAiList = allEvents.filter(e => e.held_d1_low && (e.ml_50 || 0) >= 0.20 && !(e.is_thematic_veto && !e.is_idiosyncratic));
      if (topAiCnt) topAiCnt.textContent = activeAiList.length;
      const btnTopAiCnt = document.getElementById('btn_cnt_top_ai');
      if (btnTopAiCnt) btnTopAiCnt.textContent = activeAiList.length;
      document.getElementById('cnt_today').textContent = allEvents.filter(e => e.bars_since === 0).length;
      const dreCnt = document.getElementById('cnt_dre');
      if (dreCnt) dreCnt.textContent = allEvents.filter(e => e.is_dre).length;
      document.getElementById('cnt_week').textContent = allEvents.filter(e => e.bars_since <= 5).length;
      document.getElementById('cnt_active').textContent = allEvents.filter(e => e.held_d1_low).length;
      document.getElementById('cnt_idiosyncratic').textContent = allEvents.filter(e => e.is_idiosyncratic).length;
      renderTable();
      if (allEvents.length > 0 && !selectedEvent) {
        selectEvent(allEvents[0]);
      }
    })
    .catch(err => {
      console.error("Failed to load tracker events:", err);
      const tbody = document.getElementById('table-body');
      if (tbody) {
        tbody.innerHTML = `<tr><td colspan="12" style="text-align:center; padding:20px; color:var(--red);">Error loading events: ${err.message}. Please refresh page.</td></tr>`;
      }
    });
}

function renderTable() {
  const list = getFilteredEvents();
  updateLiveStatsBanner();

  const tbody = document.getElementById('table-body');
  if (!list.length) {
    tbody.innerHTML = '<tr><td colspan="13" style="text-align:center; padding:25px; color:var(--muted);">No matching EPs for active view, archetype &amp; booster filters.<br><span style="font-size:11px; color:#38bdf8; margin-top:4px; display:inline-block;">Click "Clear Boosters" or adjust filter conditions above to expand search.</span></td></tr>';
    return;
  }

  tbody.innerHTML = list.map(e => {
    const isSel = selectedEvent && selectedEvent.symbol === e.symbol && selectedEvent.event_date === e.event_date;
    const pColor = e.ret_pct >= 0 ? 'var(--green)' : 'var(--red)';
    const rColor = e.curr_r >= 0 ? 'var(--green)' : 'var(--red)';
    
    let chipClass = 'chip-neutral';
    if (e.is_thematic_veto) chipClass = 'chip-veto';
    else if (e.is_thematic_tailwind) chipClass = 'chip-tailwind';

    const novelBadge = e.is_novel_9m ? '<span style="color:#f472b6; font-size:10px; margin-left:3px;" title="Novel 9M Volume Day (First in 60d)">🔥9M</span>' : '';
    const dreBadge = e.is_dre ? '<span style="color:#fbbf24; font-size:10px; margin-left:3px;" title="Delayed Reaction EP">⚡DRE</span>' : '';

    return `<tr class="${isSel ? 'selected' : ''}" onclick="onRowClick('${e.symbol}', '${e.event_date}')">
      <td class="tl">${e.event_date}</td>
      <td class="tl" style="font-weight:700; color:#fff;">${e.symbol}${novelBadge}${dreBadge}</td>
      <td class="tl"><span class="score-chip ${e.archetype_badge || 'chip-neutral'}">${e.trader_archetype || '📊 Quality Swing'}</span></td>
      <td class="tl">
        <span class="score-chip ${chipClass}" title="${e.theme_archetype} (${e.theme})">${e.theme_score}</span>
        <span style="font-size:10.5px; color:#c084fc; margin-left:4px; text-decoration:underline dotted; cursor:pointer;" onclick="event.stopPropagation(); openGroupStocks('${e.theme}', 'theme')" title="Click to view ${e.theme} Stock Leaderboard">${e.theme}</span>
      </td>
      <td class="tl"><span class="${e.stage_badge}">${e.status}</span></td>
      <td>+${e.gap_pct}%</td>
      <td>${e.rvol}x</td>
      <td>${e.close_pos}</td>
      <td>$${e.dvol_m}M</td>
      <td style="color:${pColor}; font-weight:600;">${e.ret_pct >= 0 ? '+' : ''}${e.ret_pct}%</td>
      <td style="color:${rColor}; font-weight:700;">${e.curr_r >= 0 ? '+' : ''}${e.curr_r} R</td>
      <td style="${e.ml_50 > 0.4 ? 'background-color:rgba(34, 197, 94, 0.2); font-weight:bold; color:#22c55e;' : ''}">${e.ml_50 != null ? (e.ml_50 * 100).toFixed(1) + '%' : '<span style="color:#64748b; font-size:10px;">Day 5</span>'}</td>
      <td style="${e.ml_150 > 0.15 ? 'background-color:rgba(34, 197, 94, 0.2); font-weight:bold; color:#22c55e;' : ''}">${e.ml_150 != null ? (e.ml_150 * 100).toFixed(1) + '%' : '<span style="color:#64748b; font-size:10px;">Day 5</span>'}</td>
    </tr>`;
  }).join('');
}

function onRowClick(sym, dt) {
  const ev = allEvents.find(e => e.symbol === sym && e.event_date === dt);
  if (ev) selectEvent(ev);
}

function selectEvent(ev) {
  selectedEvent = ev;
  renderTable();

  const topSym = document.getElementById('top_sym');
  if (topSym) topSym.textContent = `${ev.symbol}`;
  const topDetails = document.getElementById('top_details');
  if (topDetails) {
    topDetails.textContent = `EP ${ev.event_date} · +${ev.gap_pct}% Gap · ${ev.rvol}x Vol · CP ${ev.close_pos}${ev.mcap_b ? ' · $' + ev.mcap_b + 'B' : ''} · ${ev.sector} (${ev.theme})`;
  }
  
  let chipClass = 'chip-neutral';
  if (ev.is_thematic_veto) chipClass = 'chip-veto';
  else if (ev.is_thematic_tailwind) chipClass = 'chip-tailwind';

  let extraBadges = '';
  if (ev.is_dre) {
    extraBadges += `<span class="score-chip" style="background:rgba(245,158,11,0.2); color:#fbbf24; border:1px solid #d97706;" title="Delayed Reaction EP">⚡ DRE</span>`;
  }
  if (ev.is_novel_9m) {
    extraBadges += `<span class="score-chip" style="background:rgba(236,72,153,0.18); color:#f472b6; border:1px solid #db2777;" title="Novel 9M Volume (+1.92R Historical EV)">🔥 Novel 9M</span>`;
  }
  if (ev.is_cap10) {
    extraBadges += `<span class="score-chip" style="background:rgba(234,179,8,0.18); color:#facc15; border:1px solid #ca8a04;" title="CAP 10x10 ($${ev.mcap_b}B Market Cap)">💎 Cap 10×10</span>`;
  }
  if (ev.is_idiosyncratic) {
    extraBadges += `<span class="score-chip" style="background:rgba(16,185,129,0.18); color:#34d399; border:1px solid #059669;" title="Idiosyncratic Alpha (+0.50R EV)">🎯 Alpha</span>`;
  }
  if (ev.rvol >= 5.0) {
    extraBadges += `<span class="score-chip" style="background:rgba(245,158,11,0.18); color:#fbbf24; border:1px solid #b45309;" title="RVOL Sweet Spot (${ev.rvol.toFixed(1)}x)">🔥 ${ev.rvol.toFixed(1)}x RVOL</span>`;
  }
  if (ev.above_sma50 && ev.sma50_val !== null) {
    extraBadges += `<span class="score-chip" style="background:rgba(99,102,241,0.18); color:#a5b4fc; border:1px solid #6366f1;" title="Above 50 SMA ($${ev.sma50_val.toFixed(2)})">🏛️ >50 SMA</span>`;
  }

  const badgesHtml = `
    <span class="score-chip ${chipClass}" title="Theme Score: ${ev.theme_score} | Archetype: ${ev.theme_archetype}">${ev.theme_score} · ${ev.theme_archetype}</span>
    <span class="${ev.stage_badge}">${ev.status}</span>
    ${extraBadges}
  `;
  const topBadges = document.getElementById('top_badges');
  if (topBadges) topBadges.innerHTML = badgesHtml;
  const chartBadges = document.getElementById('chart_badges');
  if (chartBadges) chartBadges.innerHTML = badgesHtml;

  // Update Trade Plan Card
  let effectiveHeadline = ev.action_headline;
  let effectivePlan = ev.action_plan;
  let stopText = `$${ev.stop_price.toFixed(2)} (-${ev.risk_pct.toFixed(1)}%)`;

  if (!stratConfig.use_trade1) {
    effectiveHeadline = "SKIPPED TRADE 1 (CONTINUATION LEGS ONLY)";
    effectivePlan = "Strategy is configured to skip Day 1 ignition and focus strictly on Trade 2 & 3 yellow re-entries on lower volatility.";
    document.getElementById('plan_badge').className = "badge-digest";
    document.getElementById('plan_badge').textContent = "Skipped T1";
  } else {
    document.getElementById('plan_badge').className = ev.stage_badge;
    document.getElementById('plan_badge').textContent = ev.status;
  }

  // If ML dynamic stop is enabled and stock has reached +2.0R hurdle, ratchet stop to breakeven or trailing MAE
  if (stratConfig.use_ml_stop && ev.curr_r >= 2.0 && ev.held_d1_low) {
    const beStop = ev.entry_price;
    stopText = `$${beStop.toFixed(2)} (Ratcheted Breakeven / +2.0R Hurdle Met)`;
    effectivePlan += `\n\n🛡️ ML DYNAMIC STOP ACTIVE: +2.0R hurdle achieved. Risk eliminated; hard stop moved from Day 1 Low to Breakeven ($${beStop.toFixed(2)}).`;
  }

  // If ML profit target (exhaustion) is active and prob_exhaustion is high
  if (stratConfig.use_ml_targets && ev.prob_exhaustion && ev.prob_exhaustion > 0.80) {
    effectiveHeadline = "🎯 ML EXHAUSTION CLIMAX PROFIT TARGET TRIGGERED";
    effectivePlan = `ML Exhaustion probability ${(ev.prob_exhaustion*100).toFixed(1)}% indicates high probability of sharp mean reversion. Harvest 50% to 100% of open profit at current market.\n\n` + effectivePlan;
  }

  document.getElementById('action_headline').textContent = effectiveHeadline;
  document.getElementById('action_desc').textContent = effectivePlan;
  document.getElementById('val_entry').textContent = `$${ev.entry_price.toFixed(2)}`;
  document.getElementById('val_stop').textContent = stopText;
  document.getElementById('val_add').textContent = `$${ev.add_price.toFixed(2)}`;
  
  const rSign = ev.curr_r >= 0 ? '+' : '';
  const pnlEl = document.getElementById('val_pnl');
  pnlEl.textContent = `${rSign}${ev.curr_r.toFixed(2)} R (${ev.ret_pct >= 0 ? '+' : ''}${ev.ret_pct.toFixed(1)}%)`;
  pnlEl.style.color = ev.curr_r >= 0 ? 'var(--green)' : 'var(--red)';

  // Update Trade 2 & 3 Yellow Re-Entry Dossier Card
  const t2Pill = document.getElementById('t2_status_pill');
  const t2Box = document.getElementById('t2_details_box');
  const t2None = document.getElementById('t2_none_msg');
  const t2 = ev.t2_info || {};

  if (!stratConfig.use_multileg) {
    if (t2Pill) { t2Pill.className = 'score-chip chip-neutral'; t2Pill.textContent = 'Multi-Leg OFF'; }
    if (t2Box) t2Box.style.display = 'none';
    if (t2None) {
      t2None.style.display = 'block';
      t2None.textContent = 'Multi-Leg monitoring is disabled in top strategy config bar. Enable to track Trade 2 & 3 yellow re-entries.';
    }
  } else if (stratConfig.use_multileg_ml && t2.ml_vetoed) {
    if (t2Pill) { t2Pill.className = 'score-chip chip-veto'; t2Pill.textContent = 'ML Vetoed'; }
    if (t2Box) t2Box.style.display = 'none';
    if (t2None) {
      t2None.style.display = 'block';
      t2None.innerHTML = `⚠️ Re-entry signal was vetoed by ML Re-Entry Model (Conviction <b>${t2.prob_reentry ? (t2.prob_reentry*100).toFixed(1) : '0'}%</b> &lt; 35% hurdle). High risk of bull trap / secondary fakeout.`;
    }
  } else if (t2.has_reentry) {
    if (t2Pill) { t2Pill.className = 'score-chip chip-tailwind'; t2Pill.textContent = `Active (+${t2.curr_r !== null ? t2.curr_r : 0} R)`; }
    if (t2Box) t2Box.style.display = 'flex';
    if (t2None) t2None.style.display = 'none';
    
    document.getElementById('t2_status_val').textContent = `Yellow Flip Triggered (${t2.entry_date})`;
    document.getElementById('t2_entry_val').textContent = `$${t2.entry_price ? t2.entry_price.toFixed(2) : '-'}`;
    document.getElementById('t2_stop_val').textContent = `$${t2.stop_price ? t2.stop_price.toFixed(2) : '-'} (-${t2.risk_pct ? t2.risk_pct.toFixed(1) : '-'}%)`;
    document.getElementById('t2_cons_val').textContent = `${t2.cons_days} sessions (Retraced -${t2.drop_from_peak}%)`;
    
    const mlRow = document.getElementById('row_t2_ml');
    if (t2.prob_reentry !== null && mlRow) {
      mlRow.style.display = 'flex';
      document.getElementById('t2_ml_val').textContent = `${(t2.prob_reentry * 100).toFixed(1)}%`;
    } else if (mlRow) {
      mlRow.style.display = 'none';
    }
    
    const pnlEl2 = document.getElementById('t2_pnl_val');
    const r2 = t2.curr_r !== null ? t2.curr_r : 0;
    pnlEl2.textContent = `${r2 >= 0 ? '+' : ''}${r2.toFixed(2)} R`;
    pnlEl2.style.color = r2 >= 0 ? 'var(--green)' : 'var(--red)';
  } else if (t2.in_pullback) {
    if (t2Pill) { t2Pill.className = 'score-chip chip-gold'; t2Pill.textContent = `Pullback (Day ${t2.cons_days})`; }
    if (t2Box) t2Box.style.display = 'flex';
    if (t2None) t2None.style.display = 'none';
    
    document.getElementById('t2_status_val').textContent = `Consolidating (Shelf Support Watch)`;
    document.getElementById('t2_entry_val').textContent = `Waiting for Yellow Flip`;
    document.getElementById('t2_stop_val').textContent = `Anticipated Stop: $${t2.stop_price ? t2.stop_price.toFixed(2) : '-'}`;
    document.getElementById('t2_cons_val').textContent = `${t2.cons_days} sessions (Retraced -${t2.drop_from_peak}%)`;
    const mlRow = document.getElementById('row_t2_ml');
    if (mlRow) mlRow.style.display = 'none';
    document.getElementById('t2_pnl_val').textContent = `Pending Entry`;
    document.getElementById('t2_pnl_val').style.color = 'var(--muted)';
  } else {
    if (t2Pill) { t2Pill.className = 'score-chip chip-neutral'; t2Pill.textContent = 'No Re-Entry'; }
    if (t2Box) t2Box.style.display = 'none';
    if (t2None) {
      t2None.style.display = 'block';
      t2None.textContent = 'No Second-Leg Re-Entry detected (did not re-flip Yellow within 45 sessions or retracement exceeded 55%).';
    }
  }

  // Update Quality & Booster Audit Card
  let passCount = 0;
  const auditItems = [];

  if (!ev.is_thematic_veto || ev.is_idiosyncratic) {
    passCount++;
    auditItems.push(`<div style="display:flex; justify-content:space-between; align-items:center;"><span style="color:#34d399;">✓ Headwind Veto Cleared</span><span style="color:var(--muted); font-size:10px;">Score: ${ev.theme_score}</span></div>`);
  } else {
    auditItems.push(`<div style="display:flex; justify-content:space-between; align-items:center;"><span style="color:#f87171;">✗ Severe Headwind Drag</span><span style="color:#ef4444; font-size:10px; font-weight:700;">Trap Rate 38%</span></div>`);
  }

  if (ev.rvol >= 5.0) {
    passCount++;
    auditItems.push(`<div style="display:flex; justify-content:space-between; align-items:center;"><span style="color:#34d399;">✓ Sweet Spot RVOL (${ev.rvol.toFixed(1)}x)</span><span style="color:#fbbf24; font-size:10px; font-weight:700;">EV +3.96R</span></div>`);
  } else {
    auditItems.push(`<div style="display:flex; justify-content:space-between; align-items:center;"><span style="color:#94a3b8;">ℹ️ Standard RVOL (${ev.rvol.toFixed(1)}x)</span><span style="color:var(--muted); font-size:10px;">Target ≥ 5.0x</span></div>`);
  }

  if (ev.bars_since < 2) {
    auditItems.push(`<div style="display:flex; justify-content:space-between; align-items:center;"><span style="color:#38bdf8;">⏳ 48H Gate Pending</span><span style="color:var(--muted); font-size:10px;">Day ${ev.bars_since + 1} of 3</span></div>`);
  } else if (ev.held_48h) {
    passCount++;
    auditItems.push(`<div style="display:flex; justify-content:space-between; align-items:center;"><span style="color:#34d399;">✓ 48H Body Absorbed</span><span style="color:#34d399; font-size:10px; font-weight:700;">Trap -22%</span></div>`);
  } else {
    auditItems.push(`<div style="display:flex; justify-content:space-between; align-items:center;"><span style="color:#f87171;">✗ 48H Retrace Violated</span><span style="color:#ef4444; font-size:10px; font-weight:700;">Trap Risk 74%</span></div>`);
  }

  if (ev.close_pos >= 0.80) {
    passCount++;
    auditItems.push(`<div style="display:flex; justify-content:space-between; align-items:center;"><span style="color:#34d399;">✓ Elite ClosePos (${ev.close_pos.toFixed(2)})</span><span style="color:#38bdf8; font-size:10px; font-weight:700;">EV +0.69R</span></div>`);
  } else {
    auditItems.push(`<div style="display:flex; justify-content:space-between; align-items:center;"><span style="color:#94a3b8;">ℹ️ Standard Close (${ev.close_pos.toFixed(2)})</span><span style="color:var(--muted); font-size:10px;">Target ≥ 0.80</span></div>`);
  }

  if (ev.is_novel_9m) {
    passCount++;
    auditItems.push(`<div style="display:flex; justify-content:space-between; align-items:center;"><span style="color:#f472b6;">✓ Novel 9M Volume Ignition</span><span style="color:#34d399; font-size:10px; font-weight:700;">EV +1.92R</span></div>`);
  }

  if (ev.is_cap10) {
    passCount++;
    auditItems.push(`<div style="display:flex; justify-content:space-between; align-items:center;"><span style="color:#facc15;">✓ CAP 10×10 Incubator ($${ev.mcap_b}B)</span><span style="color:#38bdf8; font-size:10px; font-weight:700;">Multibagger Fit</span></div>`);
  }

  if (ev.above_sma50) {
    passCount++;
    auditItems.push(`<div style="display:flex; justify-content:space-between; align-items:center;"><span style="color:#34d399;">✓ Above 50 SMA (${ev.sma50_val ? '$' + ev.sma50_val.toFixed(2) : 'Holding'})</span><span style="color:#c084fc; font-size:10px; font-weight:700;">Win Rate 74%</span></div>`);
  } else {
    auditItems.push(`<div style="display:flex; justify-content:space-between; align-items:center;"><span style="color:#f87171;">✗ Below 50 SMA (${ev.sma50_val ? '$' + ev.sma50_val.toFixed(2) : '-'})</span><span style="color:#ef4444; font-size:10px;">Overhead Supply</span></div>`);
  }

  if (ev.theme_score >= 65 || ev.is_thematic_tailwind) {
    passCount++;
    auditItems.push(`<div style="display:flex; justify-content:space-between; align-items:center;"><span style="color:#34d399;">✓ Thematic Tailwind (${ev.theme_score}/100)</span><span style="color:#34d399; font-size:10px; font-weight:700;">Win +12%</span></div>`);
  } else {
    auditItems.push(`<div style="display:flex; justify-content:space-between; align-items:center;"><span style="color:#94a3b8;">ℹ️ Neutral / Lagging Theme (${ev.theme_score}/100)</span><span style="color:var(--muted); font-size:10px;">Target ≥ 65</span></div>`);
  }

  let gradeText = 'Grade C (Caution)';
  let gradeClass = 'chip-veto';
  if (passCount >= 6) {
    gradeText = `Grade A Elite (${passCount}/8 Boosters)`;
    gradeClass = 'chip-tailwind';
  } else if (passCount >= 3) {
    gradeText = `Grade B Momentum (${passCount}/8 Boosters)`;
    gradeClass = 'chip-neutral';
  }

  const qBadge = document.getElementById('qual_score_badge');
  if (qBadge) {
    qBadge.className = `score-chip ${gradeClass}`;
    qBadge.textContent = gradeText;
  }
  const qList = document.getElementById('qual_checklist');
  if (qList) {
    qList.innerHTML = auditItems.join('');
  }

  // Update Thematic Card
  const thmBadge = document.getElementById('thm_badge');
  thmBadge.className = `score-chip ${chipClass}`;
  thmBadge.textContent = `Score: ${ev.theme_score}/100`;
  document.getElementById('thm_sec_name').textContent = ev.sector;
  document.getElementById('thm_theme_name').textContent = ev.theme;
  document.getElementById('thm_archetype').textContent = ev.theme_archetype;
  document.getElementById('thm_diag_desc').textContent = ev.theme_diagnosis;

  // Update Catalyst & SEC Filings Card
  document.getElementById('cat_category').textContent = ev.catalyst_category || `${ev.sector} Structural Event`;
  document.getElementById('cat_headline').textContent = ev.catalyst_headline || `${ev.symbol} Institutional Repricing`;
  document.getElementById('cat_analysis').textContent = ev.catalyst_analysis || `${ev.symbol} experienced an episodic surge with $${ev.dvol_m}M traded.`;
  document.getElementById('cat_tier_badge').textContent = ev.catalyst_tier || 'Tier-1 Catalyst';

  const filingsList = document.getElementById('filings_list');
  const filingsCountBadge = document.getElementById('filing_count_badge');
  const filings = ev.sec_filings || [];
  filingsCountBadge.textContent = `${filings.length} Filing${filings.length === 1 ? '' : 's'}`;

  if (!filings.length) {
    filingsList.innerHTML = '<div style="color:var(--muted); font-style:italic; padding:4px 0;">No official SEC filings within 3 days. Primary catalyst driven by press release or institutional analyst upgrade.</div>';
  } else {
    filingsList.innerHTML = filings.map(f => {
      const is8K = f.form.includes('8-K') || f.form.includes('6-K');
      const badgeCol = is8K ? '#8b5cf6' : '#0284c7';
      const itemsText = f.items ? `· Items: <b>${f.items}</b>` : '';
      return `
        <div style="background:#131826; border:1px solid #202b3f; border-radius:4px; padding:5px 8px; display:flex; justify-content:space-between; align-items:center;">
          <div>
            <span class="badge" style="background:${badgeCol}; color:#fff; font-size:9.5px; padding:1px 5px;">Form ${f.form}</span>
            <span style="color:#cbd5e1; margin-left:6px;">${f.date}</span>
            <span style="color:#94a3b8; margin-left:4px; font-size:10px;">${itemsText}</span>
          </div>
          <a href="${f.url}" target="_blank" rel="noopener noreferrer" style="color:#38bdf8; text-decoration:none; font-weight:700; font-size:10.5px;" title="View official SEC filing on EDGAR archive">
            View SEC EDGAR ↗
          </a>
        </div>
      `;
    }).join('');
  }

  // Update 4 Windows Stepper
  const w1 = document.getElementById('win1_val');
  w1.textContent = `Gap +${ev.gap_pct}%, RVOL ${ev.rvol}x, Pos ${ev.close_pos}`;
  w1.style.color = ev.close_pos >= 0.65 ? 'var(--green)' : 'var(--red)';

  const w2 = document.getElementById('win2_val');
  if (ev.bars_since >= 1) {
    w2.textContent = ev.d2_add ? 'Triggered Add ✓ (Broke High)' : 'Inside Range (Held Stop)';
    w2.style.color = ev.d2_add ? 'var(--cyan)' : 'var(--gold)';
  } else {
    w2.textContent = 'Pending (Opens Tomorrow)';
    w2.style.color = 'var(--muted)';
  }

  const w3 = document.getElementById('win3_val');
  if (ev.bars_since >= 2) {
    w3.textContent = ev.held_48h ? 'Upper Body Absorbed ✓' : 'Violated (>50% lost) ✗';
    w3.style.color = ev.held_48h ? 'var(--green)' : 'var(--red)';
  } else {
    w3.textContent = 'Pending Day 3 Close';
    w3.style.color = 'var(--muted)';
  }

  const w4 = document.getElementById('win4_val');
  if (ev.bars_since >= 4) {
    w4.textContent = ev.held_d1_low ? `Riding (+${ev.curr_r} R) ✓` : 'Stop Loss Hit ✗';
    w4.style.color = ev.held_d1_low ? 'var(--green)' : 'var(--red)';
  } else {
    w4.textContent = `Developing (Day ${ev.bars_since + 1} of 5)`;
    w4.style.color = 'var(--muted)';
  }

  // [FIX]: Real-world day is 1-based (bars_since + 1). Rolling API offset is 0-based (bars_since).
  let display_day = ev.bars_since + 1;
  let fetch_offset = ev.bars_since > 0 ? ev.bars_since : 0; 
  
  document.getElementById('ai_v2_day_badge').textContent = `Day ${display_day}`;
  document.getElementById('ai_v2_50').textContent = '...';
  document.getElementById('ai_v2_100').textContent = '...';
  document.getElementById('ai_v2_150').textContent = '...';
  document.getElementById('ai_v2_200').textContent = '...';
  
  const formatProb = (p) => p >= 1.0 ? '<span style="color:#10b981; font-weight:800;">MET ✓</span>' : (p * 100).toFixed(1) + '%';

  fetch(`/api/ml_rolling?symbol=${ev.symbol}&event_date=${ev.event_date}&days_forward=${fetch_offset}`)
    .then(res => res.json())
    .then(data => {
        if (data.stopped_out || data.error) {
            document.getElementById('ai_v2_50').textContent = 'STOP';
            document.getElementById('ai_v2_100').textContent = 'STOP';
            document.getElementById('ai_v2_150').textContent = 'STOP';
            document.getElementById('ai_v2_200').textContent = 'STOP';
            return;
        }
        if (data.probs) {
            document.getElementById('ai_v2_50').innerHTML = formatProb(data.probs.prob_50);
            document.getElementById('ai_v2_100').innerHTML = formatProb(data.probs.prob_100);
            document.getElementById('ai_v2_150').innerHTML = formatProb(data.probs.prob_150);
            document.getElementById('ai_v2_200').innerHTML = formatProb(data.probs.prob_200);
            
            let adv_warnings = false;
            
            if (data.probs.dynamic_stop_loss_pct != null) {
                document.getElementById('ai_dynamic_stop_v2').style.display = 'block';
                document.getElementById('ai_stop_val_v2').textContent = (data.probs.dynamic_stop_loss_pct * 100).toFixed(1) + '%';
                adv_warnings = true;
            } else {
                document.getElementById('ai_dynamic_stop_v2').style.display = 'none';
            }

            if (data.probs.prob_reentry != null && data.probs.prob_reentry > 0.40) {
                document.getElementById('ai_reentry_v2').style.display = 'block';
                let reProb = (data.probs.prob_reentry * 100).toFixed(1) + '%';
                if (data.probs.prob_reentry > 0.60) { reProb += ' (BUY TRIGGER)'; }
                document.getElementById('ai_reentry_val_v2').textContent = reProb;
                adv_warnings = true;
            } else {
                document.getElementById('ai_reentry_v2').style.display = 'none';
            }
            
            if (data.probs.prob_exhaustion != null) {
                document.getElementById('ai_exhaustion_v2').style.display = 'block';
                let exhProb = (data.probs.prob_exhaustion * 100).toFixed(1) + '%';
                if (data.probs.prob_exhaustion > 0.70) { exhProb += ' (SELL TRIGGER)'; }
                document.getElementById('ai_exh_val_v2').textContent = exhProb;
                adv_warnings = true;
            } else {
                document.getElementById('ai_exhaustion_v2').style.display = 'none';
            }
            
            if (adv_warnings) {
                document.getElementById('ai_adv_warnings_v2').style.display = 'block';
            } else {
                document.getElementById('ai_adv_warnings_v2').style.display = 'none';
            }
        }
    });

  // Load Chart
  loadChartData(ev);
}

function loadChartData(ev) {
  if (!chart || !candleSeries) return;
  fetch(`/api/chart?symbol=${ev.symbol}&event_date=${ev.event_date}`)
    .then(r => r.json())
    .then(data => {
      if (!data || !data.bars || !data.bars.length || !candleSeries) return;

      activeLines.forEach(l => {
        try { candleSeries.removePriceLine(l); } catch(e){}
      });
      activeLines = [];

      candleSeries.setData(data.bars);
      volumeSeries.setData(data.volume);
      if (data.ema20) ema20Series.setData(data.ema20);

      activeLines.push(candleSeries.createPriceLine({
        price: parseFloat(ev.entry_price.toFixed(2)),
        color: '#10b981', lineWidth: 1, lineStyle: 0,
        title: `Entry ($${ev.entry_price.toFixed(2)})`
      }));
      activeLines.push(candleSeries.createPriceLine({
        price: parseFloat(ev.stop_price.toFixed(2)),
        color: '#ef4444', lineWidth: 2, lineStyle: 2,
        title: `Stop ($${ev.stop_price.toFixed(2)})`
      }));
      activeLines.push(candleSeries.createPriceLine({
        price: parseFloat(ev.add_price.toFixed(2)),
        color: '#38bdf8', lineWidth: 1, lineStyle: 2,
        title: `Add ($${ev.add_price.toFixed(2)})`
      }));

      candleSeries.setMarkers([
        { time: ev.event_date, position: 'belowBar', color: '#10b981', shape: 'arrowUp', text: `EP Ignition +${ev.gap_pct}%` }
      ]);

      chart.priceScale('right').applyOptions({ autoScale: true, scaleMargins: { top: 0.08, bottom: 0.25 } });
      chart.priceScale('volume').applyOptions({ scaleMargins: { top: 0.84, bottom: 0.0 } });
      chart.timeScale().fitContent();
      chart.timeScale().applyOptions({ rightOffset: 12 });
      chart.timeScale().scrollToPosition(12, false);
    })
    .catch(err => console.error("Error loading chart data:", err));
}

function refreshScan() {
  const btn = document.querySelector('.btn-refresh');
  btn.textContent = 'Scanning...';
  fetch('/api/rescan')
    .then(r => r.json())
    .then(d => {
      btn.textContent = '🔄 Rescan';
      fetchEvents();
      fetchEmergingBoard();
    });
}

function getFilteredEvents() {
  let list = allEvents;
  if (currentView === 'today') list = list.filter(e => e.bars_since === 0);
  else if (currentView === 'dre') list = list.filter(e => e.is_dre);
  else if (currentView === 'week') list = list.filter(e => e.bars_since <= 5);
  else if (currentView === 'active') list = list.filter(e => e.held_d1_low);
  else if (currentView === 'idiosyncratic') list = list.filter(e => e.is_idiosyncratic);
  else if (currentView === 'top_ai') {
    list = list.filter(e => e.held_d1_low && (e.ml_50 !== null && e.ml_50 !== undefined && e.ml_50 >= 0.20) && !(e.is_thematic_veto && !e.is_idiosyncratic))
               .sort((a,b) => {
                 const sa = ((a.ml_50||0)*1.0) + ((a.ml_100||0)*1.5) + ((a.ml_150||0)*2.0) + ((a.ml_200||0)*2.5);
                 const sb = ((b.ml_50||0)*1.0) + ((b.ml_100||0)*1.5) + ((b.ml_150||0)*2.0) + ((b.ml_200||0)*2.5);
                 return sb - sa;
               });
  }

  // Trader Archetype Dropdown Filter
  const archFilter = document.getElementById('f_trader_archetype') || document.getElementById('sel_archetype_filter');
  const archVal = archFilter ? archFilter.value : 'all';
  if (archVal && archVal !== 'all') {
    if (archVal === 'conservative') {
      list = list.filter(e => (e.close_pos >= 0.65 || !e.is_dre) && e.held_d1_low);
    } else {
      list = list.filter(e => e.archetype_key === archVal);
    }
  }

  // Continuation Mode Only (Trade 1 OFF, Multi-Leg ON)
  if (stratConfig && !stratConfig.use_trade1 && stratConfig.use_multileg) {
    list = list.filter(e => {
      const t2 = e.t2_info;
      if (!t2) return false;
      if (stratConfig.use_multileg_ml && t2.ml_vetoed) return false;
      return t2.has_reentry || t2.in_pullback;
    });
  }

  if (activeBoosters.veto_headwind) {
    list = list.filter(e => !e.is_thematic_veto || e.is_idiosyncratic);
  }
  if (activeBoosters.novel_9m) {
    list = list.filter(e => e.is_novel_9m);
  }
  if (activeBoosters.cap10) {
    list = list.filter(e => e.is_cap10);
  }
  if (activeBoosters.rvol_sweet) {
    list = list.filter(e => e.rvol >= 5.0);
  }
  if (activeBoosters['48h']) {
    list = list.filter(e => e.bars_since < 2 || e.held_48h);
  }
  if (activeBoosters.elite_close) {
    list = list.filter(e => e.close_pos >= 0.80);
  }
  if (activeBoosters.sma50) {
    list = list.filter(e => e.above_sma50);
  }
  if (activeBoosters.tailwind) {
    list = list.filter(e => e.theme_score >= 65 || e.is_thematic_tailwind);
  }
  return list;
}

function downloadTradingViewList() {
  const list = getFilteredEvents();
  if (!list.length) {
    alert("No EP setups to export in current view!");
    return;
  }
  // TradingView watchlist format: comma-separated or newline list of tickers
  const tvText = list.map(e => e.symbol).join(String.fromCharCode(44, 10));
  const blob = new Blob([tvText], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `tradingview_ep_watch_${currentView}_${new Date().toISOString().slice(0,10)}.txt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function downloadAiJson() {
  const list = getFilteredEvents();
  if (!list.length) {
    alert("No EP setups to export in current view!");
    return;
  }
  
  const payload = {
    metadata: {
      generated_at: new Date().toISOString(),
      pipeline_view: currentView,
      total_count: list.length,
      market_date: "2026-10-02",
      prompt_injection_guidance: "This JSON contains quantitative episodic pivot (EP) institutional repricing setups. Each ticker includes market microstructure footprints (Gap, RVOL, $Volume, Close Position), 4-horizon group momentum percentiles, 4-stage confirmation windows audit, official SEC EDGAR filings links, and institutional catalyst trajectory analysis."
    },
    setups: list.map(e => ({
      symbol: e.symbol,
      event_date: e.event_date,
      days_since_ignition: e.bars_since,
      status: e.status,
      microstructure: {
        gap_pct: e.gap_pct,
        rvol: e.rvol,
        close_position: e.close_pos,
        dollar_volume_millions: e.dvol_m,
        subtype: e.subtype
      },
      trade_levels: {
        entry_price: e.entry_price,
        day1_stop_loss: e.stop_price,
        day2_secondary_add: e.add_price,
        current_price: e.curr_price,
        risk_pct: e.risk_pct,
        open_return_pct: e.ret_pct,
        open_r_multiple: e.curr_r
      },
      catalyst_intelligence: {
        headline: e.catalyst_headline || "",
        category: e.catalyst_category || "",
        tier: e.catalyst_tier || "",
        institutional_analysis: e.catalyst_analysis || "",
        news_summary: e.catalyst_news || "",
        sec_edgar_filings: (e.sec_filings || []).map(f => ({
          date: f.date,
          form: f.form,
          items: f.items,
          url: f.url
        }))
      },
      thematic_radar: {
        sector: e.sector,
        theme: e.theme,
        theme_score: e.theme_score,
        archetype: e.theme_archetype,
        is_tailwind: e.is_thematic_tailwind,
        is_veto: e.is_thematic_veto,
        is_idiosyncratic: e.is_idiosyncratic,
        sector_percentiles: e.sec_pctiles,
        theme_percentiles: e.thm_pctiles
      },
      confirmation_windows: {
        window_1_day1_surge: e.close_pos >= 0.65 ? "CONFIRMED" : "WEAK",
        window_2_day2_add: e.d2_add ? "TRIGGERED_ADD" : (e.bars_since >= 1 ? "INSIDE_RANGE" : "PENDING"),
        window_3_48h_absorption: e.held_48h ? "CONFIRMED_ABSORPTION" : (e.bars_since >= 2 ? "VIOLATED" : "PENDING"),
        window_4_day5_resolution: e.held_d1_low ? "ACTIVE_RUNNER" : "STOPPED_OUT"
      },
      suggested_action: {
        headline: e.action_headline,
        execution_plan: e.action_plan
      }
    }))
  };

  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `ep_ai_agent_payload_${currentView}_${new Date().toISOString().slice(0,10)}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/* Emerging Sectors & Themes Drawer Functions */
function openEmergingDrawer() {
  document.getElementById('drawer_overlay').classList.add('active');
  if (!emergingBoard) {
    fetchEmergingBoard();
  } else {
    renderDrawerTab();
  }
}

function closeEmergingDrawer() {
  document.getElementById('drawer_overlay').classList.remove('active');
}

function closeDrawerOnOutside(e) {
  if (e.target.id === 'drawer_overlay') closeEmergingDrawer();
}

function switchDrawerTab(tab) {
  currentDrawerTab = tab;
  document.querySelectorAll('.d-tab').forEach(b => b.classList.remove('active'));
  document.getElementById(`dtab_${tab}`).classList.add('active');
  renderDrawerTab();
}

function setDrawerSort(key) {
  if (drawerSortKey === key) {
    drawerSortAsc = !drawerSortAsc;
  } else {
    drawerSortKey = key;
    drawerSortAsc = (key === 'name'); // default alphabetical asc, numeric desc
  }
  renderDrawerTab();
}

function fetchEmergingBoard() {
  fetch('/api/emerging_board')
    .then(r => r.json())
    .then(data => {
      emergingBoard = data;
      renderDrawerTab();
    })
    .catch(e => console.error("Failed to fetch emerging board:", e));
}

function renderDrawerTab() {
  const container = document.getElementById('drawer_body');
  if (!emergingBoard) {
    container.innerHTML = '<div style="text-align:center; padding:30px; color:var(--muted);">Loading emerging data...</div>';
    return;
  }

  if (currentDrawerTab === 'themes') {
    const list = emergingBoard.emerging_themes || emergingBoard.all_themes_ranked || [];
    container.innerHTML = `
      <div style="margin-bottom:12px; font-size:12px; color:#cbd5e1;">
        Displaying current top emerging themes ranked by composite 4-TF momentum and historical 10-Yr EP incubator edge bonus.
      </div>
      <table class="radar-table">
        <thead>
          <tr>
            <th class="tl">#</th>
            <th class="tl">Theme Cluster</th>
            <th>Composite</th>
            <th>1W</th>
            <th>1M</th>
            <th>3M</th>
            <th>YTD</th>
            <th class="tl">Archetype</th>
            <th>Incubator</th>
          </tr>
        </thead>
        <tbody>
          ${list.map((t, idx) => {
            const barW = Math.max(5, Math.min(100, t.score));
            const barCol = t.score >= 80 ? '#10b981' : (t.score >= 65 ? '#3b82f6' : '#f59e0b');
            return `<tr onclick="openGroupStocks('${t.name}', 'theme')" style="cursor:pointer;" title="Click to view all ${t.name} constituent stocks">
              <td class="tl" style="color:var(--muted);">${idx + 1}</td>
              <td class="tl" style="font-weight:700; color:#fff;">
                <span>${t.name}</span>
                <span style="font-size:9.5px; color:#38bdf8; margin-left:6px;">🔍 Stocks</span>
              </td>
              <td style="font-weight:700; color:${barCol};">
                ${t.score}
                <div class="score-bar-bg"><div class="score-bar-fill" style="width:${barW}%; background:${barCol};"></div></div>
              </td>
              <td style="color:${t.w1_pct >= 70 ? 'var(--green)' : '#cbd5e1'}">${t.w1_pct}th</td>
              <td style="color:${t.m1_pct >= 70 ? 'var(--green)' : '#cbd5e1'}">${t.m1_pct}th</td>
              <td style="color:${t.m3_pct >= 70 ? 'var(--green)' : '#cbd5e1'}">${t.m3_pct}th</td>
              <td style="color:${t.ytd_pct >= 70 ? 'var(--green)' : '#cbd5e1'}">${t.ytd_pct}th</td>
              <td class="tl"><span class="badge" style="background:#2e1065; color:#c4b5fd; font-size:9.5px;">${t.primary_archetype}</span></td>
              <td style="color:var(--gold); font-weight:700;">+${t.bonus} pts</td>
            </tr>`;
          }).join('')}
        </tbody>
      </table>
    `;
  } else if (currentDrawerTab === 'sectors') {
    const list = emergingBoard.emerging_sectors || emergingBoard.all_sectors_ranked || [];
    container.innerHTML = `
      <div style="margin-bottom:12px; font-size:12px; color:#cbd5e1;">
        Displaying top Finviz industry sectors leading institutional liquidity flow. <b>Click any row to inspect member stocks!</b>
      </div>
      <table class="radar-table">
        <thead>
          <tr>
            <th class="tl">#</th>
            <th class="tl">Sector Name</th>
            <th>Composite</th>
            <th>1W</th>
            <th>1M</th>
            <th>3M</th>
            <th>YTD</th>
            <th class="tl">Archetype</th>
            <th>Incubator</th>
          </tr>
        </thead>
        <tbody>
          ${list.map((s, idx) => {
            const barW = Math.max(5, Math.min(100, s.score));
            const barCol = s.score >= 80 ? '#10b981' : (s.score >= 65 ? '#3b82f6' : '#f59e0b');
            return `<tr onclick="openGroupStocks('${s.name}', 'sector')" style="cursor:pointer;" title="Click to view all ${s.name} constituent stocks">
              <td class="tl" style="color:var(--muted);">${idx + 1}</td>
              <td class="tl" style="font-weight:700; color:#fff;">
                <span>${s.name}</span>
                <span style="font-size:9.5px; color:#38bdf8; margin-left:6px;">🔍 Stocks</span>
              </td>
              <td style="font-weight:700; color:${barCol};">
                ${s.score}
                <div class="score-bar-bg"><div class="score-bar-fill" style="width:${barW}%; background:${barCol};"></div></div>
              </td>
              <td style="color:${s.w1_pct >= 70 ? 'var(--green)' : '#cbd5e1'}">${s.w1_pct}th</td>
              <td style="color:${s.m1_pct >= 70 ? 'var(--green)' : '#cbd5e1'}">${s.m1_pct}th</td>
              <td style="color:${s.m3_pct >= 70 ? 'var(--green)' : '#cbd5e1'}">${s.m3_pct}th</td>
              <td style="color:${s.ytd_pct >= 70 ? 'var(--green)' : '#cbd5e1'}">${s.ytd_pct}th</td>
              <td class="tl"><span class="badge" style="background:#082f49; color:#38bdf8; font-size:9.5px;">${s.primary_archetype}</span></td>
              <td style="color:var(--gold); font-weight:700;">+${s.bonus} pts</td>
            </tr>`;
          }).join('')}
        </tbody>
      </table>
    `;
  } else if (currentDrawerTab === 'all_themes' || currentDrawerTab === 'all_sectors') {
    const isTheme = currentDrawerTab === 'all_themes';
    const rawList = isTheme ? (emergingBoard.all_themes_ranked || []) : (emergingBoard.all_sectors_ranked || []);
    const kind = isTheme ? 'theme' : 'sector';
    const label = isTheme ? 'Themes' : 'Sectors';

    // Sort list according to drawerSortKey and drawerSortAsc
    const list = [...rawList].sort((a, b) => {
      let vA = a[drawerSortKey];
      let vB = b[drawerSortKey];
      if (typeof vA === 'string') {
        return drawerSortAsc ? vA.localeCompare(vB) : vB.localeCompare(vA);
      }
      vA = vA != null ? vA : -999999;
      vB = vB != null ? vB : -999999;
      return drawerSortAsc ? vA - vB : vB - vA;
    });

    const sortArrow = (k) => drawerSortKey === k ? (drawerSortAsc ? ' ▲' : ' ▼') : '';
    const thStyle = (k) => `cursor:pointer; ${drawerSortKey === k ? 'color:var(--blue); font-weight:700;' : ''}`;

    container.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; font-size:12px; color:#cbd5e1;">
        <div>Complete Institutional Leaderboard: <b>${list.length} ${label}</b> ranked across all timeframes. Click any column header to sort.</div>
        <div style="font-size:11px; color:#94a3b8;">Sorted by: <b>${drawerSortKey.toUpperCase()}</b> (${drawerSortAsc ? 'Asc' : 'Desc'})</div>
      </div>
      <table class="radar-table">
        <thead>
          <tr>
            <th class="tl" onclick="setDrawerSort('score')" style="cursor:pointer;">#</th>
            <th class="tl" onclick="setDrawerSort('name')" style="${thStyle('name')}">${label.slice(0, -1)} Name${sortArrow('name')}</th>
            <th onclick="setDrawerSort('member_count')" style="${thStyle('member_count')}">N${sortArrow('member_count')}</th>
            <th onclick="setDrawerSort('score')" style="${thStyle('score')}">Score${sortArrow('score')}</th>
            <th onclick="setDrawerSort('w1_ret')" style="${thStyle('w1_ret')}">1W %${sortArrow('w1_ret')}</th>
            <th onclick="setDrawerSort('m1_ret')" style="${thStyle('m1_ret')}">1M %${sortArrow('m1_ret')}</th>
            <th onclick="setDrawerSort('m3_ret')" style="${thStyle('m3_ret')}">3M %${sortArrow('m3_ret')}</th>
            <th onclick="setDrawerSort('ytd_ret')" style="${thStyle('ytd_ret')}">YTD %${sortArrow('ytd_ret')}</th>
            <th class="tl">Archetype</th>
            <th onclick="setDrawerSort('bonus')" style="${thStyle('bonus')}">Incubator${sortArrow('bonus')}</th>
          </tr>
        </thead>
        <tbody>
          ${list.map((row, idx) => {
            const barW = Math.max(5, Math.min(100, row.score));
            const barCol = row.score >= 80 ? '#10b981' : (row.score >= 65 ? '#3b82f6' : (row.score >= 50 ? '#f59e0b' : '#ef4444'));
            
            const fmtRet = (ret, pct) => {
              if (ret == null) return '<span style="color:var(--muted);">-</span>';
              const col = ret >= 0 ? 'var(--green)' : 'var(--red)';
              const sign = ret > 0 ? '+' : '';
              return `<div style="color:${col}; font-weight:600;">${sign}${ret.toFixed(1)}%</div>
                      <div style="font-size:9.5px; color:#94a3b8;">${pct != null ? pct + 'th' : ''}</div>`;
            };

            const archBg = row.badge_type === 'veto' ? 'rgba(239,68,68,0.2)' : (row.badge_type === 'power' ? '#2e1065' : (row.badge_type === 'surge' ? '#064e3b' : '#1e293b'));
            const archCol = row.badge_type === 'veto' ? '#f87171' : (row.badge_type === 'power' ? '#c4b5fd' : (row.badge_type === 'surge' ? '#6ee7b7' : '#cbd5e1'));

            return `<tr onclick="openGroupStocks('${row.name}', '${kind}')" style="cursor:pointer;" title="Click to inspect all ${row.name} constituent stocks">
              <td class="tl" style="color:var(--muted);">${idx + 1}</td>
              <td class="tl" style="font-weight:700; color:#fff;">
                <span>${row.name}</span>
                <span style="font-size:9.5px; color:#38bdf8; margin-left:6px;">🔍 Stocks</span>
              </td>
              <td style="color:var(--muted); font-size:11px;">${row.member_count || '-'}</td>
              <td style="font-weight:700; color:${barCol};">
                ${row.score}
                <div class="score-bar-bg"><div class="score-bar-fill" style="width:${barW}%; background:${barCol};"></div></div>
              </td>
              <td>${fmtRet(row.w1_ret, row.w1_pct)}</td>
              <td>${fmtRet(row.m1_ret, row.m1_pct)}</td>
              <td>${fmtRet(row.m3_ret, row.m3_pct)}</td>
              <td>${fmtRet(row.ytd_ret, row.ytd_pct)}</td>
              <td class="tl"><span class="badge" style="background:${archBg}; color:${archCol}; font-size:9.5px;">${row.primary_archetype}</span></td>
              <td style="color:${row.bonus > 0 ? 'var(--gold)' : 'var(--muted)'}; font-weight:700;">+${row.bonus} pts</td>
            </tr>`;
          }).join('')}
        </tbody>
      </table>
    `;
  } else if (currentDrawerTab === 'veto') {
    const list = emergingBoard.headwind_veto || [];
    container.innerHTML = `
      <div style="margin-bottom:12px; font-size:12px; color:#fca5a5; background:rgba(239,68,68,0.15); border:1px solid #dc2626; padding:10px 12px; border-radius:6px; line-height:1.4;">
        <b>⚠️ Mandatory Hard Veto Protocol:</b> Groups lagging in the bottom 35th percentile across 1M and 3M have a <b>37.5% failure rate</b> on EPs. The strategy applies a strict risk penalty or avoids taking breakout adds. <b>Click any row to inspect member stocks.</b>
      </div>
      <table class="radar-table">
        <thead>
          <tr>
            <th class="tl">#</th>
            <th class="tl">Group Name</th>
            <th class="tl">Type</th>
            <th>Score</th>
            <th>1M Pctile</th>
            <th>3M Pctile</th>
            <th class="tl">Status</th>
          </tr>
        </thead>
        <tbody>
          ${list.map((v, idx) => `<tr onclick="openGroupStocks('${v.name}', '${v.kind || "theme"}')" style="cursor:pointer;" title="Click to view all ${v.name} constituent stocks">
            <td class="tl" style="color:var(--muted);">${idx + 1}</td>
            <td class="tl" style="font-weight:700; color:#f87171;">
              <span>${v.name}</span>
              <span style="font-size:9.5px; color:#fca5a5; margin-left:6px;">🔍 Stocks</span>
            </td>
            <td class="tl" style="color:var(--muted); text-transform:capitalize;">${v.kind || 'group'}</td>
            <td style="font-weight:700; color:#ef4444;">${v.score}</td>
            <td style="color:#ef4444;">${v.m1_pct}th</td>
            <td style="color:#ef4444;">${v.m3_pct}th</td>
            <td class="tl"><span class="badge" style="background:rgba(239,68,68,0.2); color:#f87171; font-size:9.5px;">Hard Veto</span></td>
          </tr>`).join('')}
        </tbody>
      </table>
    `;
  } else if (currentDrawerTab === 'guide') {
    container.innerHTML = `
      <div style="display:flex; flex-direction:column; gap:14px; font-size:12px; color:#cbd5e1; line-height:1.5;">
        <div style="background:#131a28; border-left:4px solid #10b981; padding:10px 14px; border-radius:4px;">
          <h4 style="color:#34d399; font-size:13px; margin-bottom:4px;">1. ⚡ Velocity Surge (1W ≥ 85th, 3M ≤ 50th)</h4>
          <p><b>Market Phenomenon:</b> Sudden thematic ignition or inflection point emerging from dormancy (e.g., fresh regulatory approval or new tech catalyst). Highest early asymmetric R-multiple potential.</p>
          <p style="margin-top:4px; color:#94a3b8;"><b>Execution Rule:</b> Take full 1.0 R size immediately; set Day 1 stop loss tight. Prepare for aggressive Day 2 secondary add.</p>
        </div>

        <div style="background:#131a28; border-left:4px solid #8b5cf6; padding:10px 14px; border-radius:4px;">
          <h4 style="color:#c084fc; font-size:13px; margin-bottom:4px;">2. 🚀 Multi-TF Power Cluster (All 4 TFs ≥ 70th)</h4>
          <p><b>Market Phenomenon:</b> Undisputed institutional leadership and secular multi-quarter accumulation (e.g. Semiconductors in AI bull run). Win rate jumps to 44.8%.</p>
          <p style="margin-top:4px; color:#94a3b8;"><b>Execution Rule:</b> Ideal for multi-quarter PEAD drift. Hold through shallow consolidations and trail stops along daily 20 EMA.</p>
        </div>

        <div style="background:#131a28; border-left:4px solid #38bdf8; padding:10px 14px; border-radius:4px;">
          <h4 style="color:#38bdf8; font-size:13px; margin-bottom:4px;">3. 📈 Acceleration Cascade (1W > 1M > 3M > YTD)</h4>
          <p><b>Market Phenomenon:</b> Clean upward momentum gradient where short-term flow is accelerating relative to long-term baseline. Institutional buying is steadily intensifying.</p>
          <p style="margin-top:4px; color:#94a3b8;"><b>Execution Rule:</b> High confidence for follow-through. Look for second-leg yellow flips on any subsequent consolidation.</p>
        </div>

        <div style="background:#131a28; border-left:4px solid #f59e0b; padding:10px 14px; border-radius:4px;">
          <h4 style="color:#fbbf24; font-size:13px; margin-bottom:4px;">4. 🏛️ Top Board Persistence (1M ≥ 80th, 3M ≥ 80th)</h4>
          <p><b>Market Phenomenon:</b> Sustained residency in the top deciles of performance. Demonstrates durable institutional capital sponsorship.</p>
          <p style="margin-top:4px; color:#94a3b8;"><b>Execution Rule:</b> Low retrace risk. Day 3 retrace gate passes in >82% of cases.</p>
        </div>

        <div style="background:rgba(239,68,68,0.1); border-left:4px solid #ef4444; padding:10px 14px; border-radius:4px;">
          <h4 style="color:#f87171; font-size:13px; margin-bottom:4px;">5. 🚫 Severe Headwind Veto (1M < 35th AND 3M < 35th)</h4>
          <p><b>Market Phenomenon:</b> Outcast sectors facing institutional distribution or capital flight. EPs here suffer from frequent gap-and-crap traps (37.5% failure).</p>
          <p style="margin-top:4px; color:#fca5a5;"><b>Execution Rule:</b> VETO or cut risk by 50%. Strictly enforce stop at Day 1 low.</p>
        </div>
      </div>
    `;
  }
}

function openGroupStocks(name, kind) {
  openEmergingDrawer();
  const container = document.getElementById('drawer_body');
  container.innerHTML = `
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; border-bottom:1px solid var(--border); padding-bottom:10px;">
      <button class="btn-refresh" onclick="switchDrawerTab('${kind === 'theme' ? 'themes' : 'sectors'}')">← Back to ${kind === 'theme' ? 'Themes' : 'Sectors'}</button>
      <div style="font-size:14px; font-weight:700; color:#fff;">${name} (${kind.toUpperCase()})</div>
      <div style="color:var(--muted); font-size:11px;">Loading stock leaderboard...</div>
    </div>
    <div style="text-align:center; padding:30px; color:var(--muted);">Fetching constituent stocks and performance...</div>
  `;

  fetch(`/api/group_stocks?group=${encodeURIComponent(name)}&kind=${kind}`)
    .then(r => r.json())
    .then(data => {
      if (!data || !data.stocks) {
        container.innerHTML = `<div style="padding:20px; color:var(--red);">No stocks found for ${name}.</div>`;
        return;
      }
      const info = data.info || {};
      const scoreCol = (info.score || 50) >= 80 ? '#10b981' : ((info.score || 50) >= 65 ? '#3b82f6' : '#f59e0b');

      container.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; border-bottom:1px solid var(--border); padding-bottom:10px;">
          <button class="btn-refresh" onclick="switchDrawerTab('${kind === 'theme' ? 'themes' : 'sectors'}')">← Back to ${kind === 'theme' ? 'Themes' : 'Sectors'}</button>
          <div style="text-align:center;">
            <div style="font-size:15px; font-weight:700; color:#fff;">${name}</div>
            <div style="font-size:11px; color:#94a3b8;">${kind.toUpperCase()} · Score: <b style="color:${scoreCol};">${info.score || '·'}</b> · ${info.primary_archetype || ''}</div>
          </div>
          <span class="badge" style="background:#1e293b; color:#cbd5e1;">${data.count} Stocks</span>
        </div>

        <div style="margin-bottom:10px; font-size:11.5px; color:#94a3b8;">
          Constituent stocks ranked by 1-week momentum. Click any stock to load chart.
        </div>

        <table class="radar-table">
          <thead>
            <tr>
              <th class="tl">#</th>
              <th class="tl">Symbol</th>
              <th>Price</th>
              <th>D1 %</th>
              <th>1W %</th>
              <th>1M %</th>
              <th>3M %</th>
              <th>YTD %</th>
              <th>EP Hist</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            ${data.stocks.map((s, idx) => {
              const c1w = s.w1 != null && s.w1 >= 0 ? 'var(--green)' : 'var(--red)';
              const cm1 = s.m1 != null && s.m1 >= 0 ? 'var(--green)' : 'var(--red)';
              const cytd = s.ytd != null && s.ytd >= 0 ? 'var(--gold)' : '#cbd5e1';
              const epBadge = s.ep_count > 0 ? `<span class="badge" style="background:#047857; color:#6ee7b7; font-size:9px;">${s.ep_count} EPs</span>` : `<span style="color:var(--muted);">-</span>`;
              return `<tr onclick="loadStockChart('${s.symbol}')">
                <td class="tl" style="color:var(--muted);">${idx + 1}</td>
                <td class="tl" style="font-weight:700; color:#fff;">${s.symbol}</td>
                <td style="color:#cbd5e1;">${s.price != null ? '$' + s.price.toFixed(2) : '·'}</td>
                <td style="color:${s.d1 != null && s.d1 >= 0 ? 'var(--green)' : 'var(--red)'};">${s.d1 != null ? (s.d1 >= 0 ? '+' : '') + s.d1.toFixed(1) + '%' : '·'}</td>
                <td style="font-weight:700; color:${c1w};">${s.w1 != null ? (s.w1 >= 0 ? '+' : '') + s.w1.toFixed(1) + '%' : '·'}</td>
                <td style="color:${cm1};">${s.m1 != null ? (s.m1 >= 0 ? '+' : '') + s.m1.toFixed(1) + '%' : '·'}</td>
                <td style="color:${s.m3 != null && s.m3 >= 0 ? 'var(--green)' : 'var(--red)'};">${s.m3 != null ? (s.m3 >= 0 ? '+' : '') + s.m3.toFixed(1) + '%' : '·'}</td>
                <td style="font-weight:700; color:${cytd};">${s.ytd != null ? (s.ytd >= 0 ? '+' : '') + s.ytd.toFixed(1) + '%' : '·'}</td>
                <td>${epBadge}</td>
                <td><button class="btn-refresh" style="padding:2px 8px; font-size:10px;" onclick="event.stopPropagation(); loadStockChart('${s.symbol}')">Chart 📈</button></td>
              </tr>`;
            }).join('')}
          </tbody>
        </table>
      `;
    })
    .catch(e => {
      container.innerHTML = `<div style="padding:20px; color:var(--red);">Error loading stocks: ${e.message}</div>`;
    });
}

function loadStockChart(sym) {
  const ev = allEvents.find(e => e.symbol === sym);
  if (ev) {
    selectEvent(ev);
  } else {
    if (!chart || !candleSeries) return;
    fetch(`/api/chart?symbol=${sym}`)
      .then(r => r.json())
      .then(data => {
        if (!data || !data.bars || !data.bars.length || !candleSeries) return;
        activeLines.forEach(l => {
          try { candleSeries.removePriceLine(l); } catch(e){}
        });
        activeLines = [];
        candleSeries.setData(data.bars);
        volumeSeries.setData(data.volume);
        if (data.ema20) ema20Series.setData(data.ema20);
        candleSeries.setMarkers([]);
        document.getElementById('chart_sym').textContent = `${sym} — Stock Chart`;
        document.getElementById('chart_details').textContent = `${sym} constituent chart`;
        document.getElementById('chart_badges').innerHTML = `<span class="score-chip chip-neutral">${sym}</span>`;
        chart.timeScale().fitContent();
        chart.timeScale().applyOptions({ rightOffset: 12 });
        chart.timeScale().scrollToPosition(12, false);
      })
      .catch(e => console.error("Error loading stock chart:", e));
  }
}

// ==========================================
// EP PORTFOLIO MANAGER & LIFECYCLE CONTROLS
// ==========================================
let currentMainView = 'scanner';
let portfolioData = null;
let premarketData = null;

function switchMainView(view) {
  currentMainView = view;
  const scanView = document.getElementById('scanner_view');
  const portView = document.getElementById('portfolio_view');
  const pmView = document.getElementById('premarket_view');
  const btnScan = document.getElementById('nav_btn_scanner');
  const btnPort = document.getElementById('nav_btn_portfolio');
  const btnPm = document.getElementById('nav_btn_premarket');

  if (view === 'portfolio') {
    if (scanView) scanView.style.display = 'none';
    if (pmView) pmView.style.display = 'none';
    if (portView) portView.style.display = 'flex';
    if (btnScan) btnScan.classList.remove('active');
    if (btnPm) btnPm.classList.remove('active');
    if (btnPort) btnPort.classList.add('active');
    fetchPortfolio();
    window.location.hash = '#portfolio';
  } else if (view === 'premarket') {
    if (scanView) scanView.style.display = 'none';
    if (portView) portView.style.display = 'none';
    if (pmView) pmView.style.display = 'flex';
    if (btnScan) btnScan.classList.remove('active');
    if (btnPort) btnPort.classList.remove('active');
    if (btnPm) btnPm.classList.add('active');
    fetchPremarketData(false);
    window.location.hash = '#premarket';
  } else {
    if (portView) portView.style.display = 'none';
    if (pmView) pmView.style.display = 'none';
    if (scanView) scanView.style.display = 'flex';
    if (btnPort) btnPort.classList.remove('active');
    if (btnPm) btnPm.classList.remove('active');
    if (btnScan) btnScan.classList.add('active');
    window.location.hash = '#scanner';
    if (chart) chart.timeScale().fitContent();
  }
}

// Premarket Radar Client Operations
function fetchPremarketData(force = false) {
  fetch(`/api/premarket/data?force=${force ? '1' : '0'}`)
    .then(r => r.json())
    .then(data => {
      premarketData = data;
      renderPremarketKPIs(data.metadata);
      renderPremarketTables(data);
    })
    .catch(e => console.error("Error fetching premarket data:", e));
}

function updatePremarketScan() {
  const btnHeader = document.getElementById('btn_pm_update_header');
  const btnPanel = document.getElementById('pm_btn_rescan');
  if (btnHeader) btnHeader.textContent = "⏳ Pulling 15m Bars...";
  if (btnPanel) btnPanel.textContent = "⏳ Scanning In-Play Sectors & SEC 8-K...";

  fetch('/api/premarket/scan', { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      premarketData = data;
      renderPremarketKPIs(data.metadata);
      renderPremarketTables(data);
      if (btnHeader) btnHeader.textContent = "⚡ Premarket Update";
      if (btnPanel) btnPanel.textContent = "⚡ Premarket Update / Rescan";
      if (currentMainView !== 'premarket') {
        switchMainView('premarket');
      }
    })
    .catch(err => {
      alert("Error scanning premarket: " + err.message);
      if (btnHeader) btnHeader.textContent = "⚡ Premarket Update";
      if (btnPanel) btnPanel.textContent = "⚡ Premarket Update / Rescan";
    });
}

function renderPremarketKPIs(meta) {
  if (!meta) return;
  const elScanned = document.getElementById('pm_kpi_scanned');
  const elElite = document.getElementById('pm_kpi_elite');
  const elGappers = document.getElementById('pm_kpi_gappers');
  const elThemes = document.getElementById('pm_kpi_themes');
  const elUpdated = document.getElementById('pm_kpi_updated');
  const badgeNav = document.getElementById('premarket_nav_badge');
  const badgeElite = document.getElementById('pm_badge_elite_count');
  const badgeGappers = document.getElementById('pm_badge_gappers_count');

  if (elScanned) elScanned.textContent = meta.total_scanned || 0;
  if (elElite) elElite.textContent = meta.potential_elite_eps_count || 0;
  if (elGappers) elGappers.textContent = meta.overnight_gappers_count || 0;
  if (badgeElite) badgeElite.textContent = meta.potential_elite_eps_count || 0;
  if (badgeGappers) badgeGappers.textContent = meta.overnight_gappers_count || 0;
  if (badgeNav) {
    badgeNav.textContent = meta.potential_elite_eps_count || 0;
    badgeNav.style.display = (meta.potential_elite_eps_count > 0) ? 'inline-block' : 'none';
  }
  if (elThemes && meta.leading_themes) {
    elThemes.textContent = meta.leading_themes.slice(0, 3).join(", ");
  }
  if (elUpdated) elUpdated.textContent = meta.timestamp || "Live";
}

function renderPremarketTables(data) {
  if (!data) return;
  const eliteBody = document.getElementById('pm_elite_tbody');
  const gappersBody = document.getElementById('pm_gappers_tbody');

  // 1. Potential Elite EPs
  if (eliteBody) {
    if (!data.potential_elite_eps || !data.potential_elite_eps.length) {
      eliteBody.innerHTML = `<tr><td colspan="12" style="text-align:center; padding:25px; color:var(--muted);">No potential Elite EPs currently qualify (requires Gap &ge; +4%, Proj RVOL &ge; 2.0x, or material 8-K in a Tailwind theme).</td></tr>`;
    } else {
      let html = '';
      data.potential_elite_eps.forEach(item => {
        const gapColor = item.gap_pct >= 0 ? '#34d399' : '#f87171';
        const rvolBadge = item.projected_rvol >= 5.0 
          ? `<span class="badge" style="background:#f59e0b; color:#000; font-weight:800;">${item.projected_rvol}x Sweet Spot</span>`
          : `<span class="badge" style="background:#0284c7;">${item.projected_rvol}x</span>`;
        const tacticBadge = item.early_tactic.includes("ORB")
          ? `<span class="badge" style="background:#0284c7; color:#fff;">🎯 ${item.early_tactic}</span>`
          : `<span class="badge" style="background:#7c3aed; color:#fff;">🌊 ${item.early_tactic}</span>`;
        const capBadge = item.is_biotech
          ? `<span class="badge" style="background:rgba(239,68,68,0.25); color:#fca5a5; border:1px solid #ef4444;">12.5% Biotech Cap</span>`
          : `<span class="badge" style="background:rgba(56,189,248,0.2); color:#38bdf8;">25.0% Standard</span>`;

        html += `
          <tr>
            <td class="tl">
              <span style="font-weight:800; font-size:12.5px; color:#fff; cursor:pointer;" onclick="inspectPremarketStock('${item.symbol}')">${item.symbol}</span>
              <div style="font-size:10.5px; color:var(--muted);">${item.sector} • <span style="color:#c4b5fd;">${item.theme}</span> (${item.theme_score})</div>
            </td>
            <td>$${item.prior_close.toFixed(2)}</td>
            <td style="font-weight:700; color:#fff;">$${item.premarket_price.toFixed(2)}</td>
            <td style="font-weight:800; color:${gapColor}; font-size:12px;">${item.gap_pct >= 0 ? '+' : ''}${item.gap_pct.toFixed(2)}%</td>
            <td style="font-size:11px;">${item.premarket_vol.toLocaleString()} <span style="color:var(--muted); font-size:10px;">/ ${(item.avg_20d_vol/1e3).toFixed(0)}k</span></td>
            <td>${rvolBadge}</td>
            <td style="font-size:10.5px; color:#cbd5e1; max-width:180px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;" title="${item.sec_8k_filings} | ${item.catalyst_category}">
              <div style="font-weight:600; color:#38bdf8;">${item.catalyst_category}</div>
              <div style="color:var(--muted); font-size:10px;">${item.sec_8k_filings}</div>
            </td>
            <td class="tl" style="font-size:11px; max-width:220px;">
              ${tacticBadge}
              <div style="font-size:10px; color:var(--muted); margin-top:2px;">${item.tactic_rule}</div>
            </td>
            <td>${capBadge}</td>
            <td>
              <div style="display:flex; gap:4px; justify-content:flex-end;">
                <button class="alert-btn blue" onclick="addPremarketToPortfolio(${JSON.stringify(item).replace(/"/g, '&quot;')})" title="Add position to Live Portfolio pre-populated with tactical stop">+ Port</button>
                <button class="alert-btn" style="background:#1e293b; color:#cbd5e1;" onclick="inspectPremarketStock('${item.symbol}')">Chart</button>
              </div>
            </td>
          </tr>
        `;
      });
      eliteBody.innerHTML = html;
    }
  }

  // 2. Overnight Gappers
  if (gappersBody) {
    if (!data.overnight_gappers || !data.overnight_gappers.length) {
      gappersBody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding:20px; color:var(--muted);">No other overnight gappers detected.</td></tr>`;
    } else {
      let html = '';
      data.overnight_gappers.forEach(item => {
        const gapColor = item.gap_pct >= 0 ? '#fbbf24' : '#f87171';
        html += `
          <tr>
            <td class="tl">
              <span style="font-weight:800; font-size:12px; color:#cbd5e1; cursor:pointer;" onclick="inspectPremarketStock('${item.symbol}')">${item.symbol}</span>
              <div style="font-size:10px; color:var(--muted);">${item.theme} (${item.theme_score})</div>
            </td>
            <td>$${item.prior_close.toFixed(2)}</td>
            <td style="font-weight:700;">$${item.premarket_price.toFixed(2)}</td>
            <td style="font-weight:700; color:${gapColor}; font-size:11.5px;">${item.gap_pct >= 0 ? '+' : ''}${item.gap_pct.toFixed(2)}%</td>
            <td style="font-size:11px;">${item.premarket_vol.toLocaleString()}</td>
            <td>${item.projected_rvol}x</td>
            <td style="font-size:10.5px; color:var(--muted);">${item.sec_8k_filings}</td>
            <td class="tl" style="font-size:11px; color:#cbd5e1;">
              ${item.tactic_rule}
            </td>
            <td>
              <button class="alert-btn" style="background:#1e293b; color:#cbd5e1;" onclick="inspectPremarketStock('${item.symbol}')">Chart</button>
            </td>
          </tr>
        `;
      });
      gappersBody.innerHTML = html;
    }
  }
}

function addPremarketToPortfolio(item) {
  const tacticalStopDistance = item.early_tactic.includes("ORB") ? 0.03 : 0.045; // 3% ORB stop vs 4.5% VWAP stop
  const ev = {
    symbol: item.symbol,
    company: item.symbol,
    sector: item.sector,
    theme: item.theme,
    close: item.premarket_price,
    d1_close: item.premarket_price,
    d1_low: item.premarket_price * (1.0 - tacticalStopDistance),
    quality_tier: item.is_tailwind ? 'A' : 'B',
    event_date: item.timestamp ? item.timestamp.split(' ')[0] : '2026-10-05',
    early_tactic: item.early_tactic
  };
  openAddPortfolioModal(ev);
  const notesEl = document.getElementById('m_add_notes');
  if (notesEl) {
    notesEl.value = `Premarket Early Ignition: ${item.early_tactic} | Gap: ${item.gap_pct}% | Proj RVOL: ${item.projected_rvol}x`;
  }
}

function inspectPremarketStock(sym) {
  switchMainView('scanner');
  loadStockChart(sym);
}

function downloadPremarketAiJson() {
  window.open('/api/export/premarket_json', '_blank');
}

function copyPremarketPrompt() {
  const masterPrompt = `You are the Lead Quantitative Execution Trader specializing in Premarket Episodic Pivot (EP) Ignitions and Opening Range Breakouts (ORB).

You have been provided with:
1. The Premarket EP Tactical Knowledge Base (documenting the +10.39% Day 1 expansion vs the 64.5% Blind Gapper Trap, 15m ORB rules, and VWAP Reclaim mechanics).
2. A live JSON export of today's premarket candidates from the institutional scanner (/api/export/premarket_json).

### YOUR PRIME OBJECTIVE
Evaluate this morning's premarket candidate cohort, eliminate the 64.5% "Gap-and-Crap" fader traps, and construct a precise, execution-ready Intraday Opening Game Plan (09:15 – 10:15 EST) for high-expectancy capital deployment.

---

### MANDATORY TRIAGE & EXECUTION PROTOCOLS

1. COHORT SEGREGATION (SEPARATING ELITE CANDIDATES FROM BLIND GAPPERS):
   - Group A: "POTENTIAL ELITE EP (EARLY TACTICAL ACTION)"
     * Criteria: Overnight Gap ≥ +4.0% (ideally +5.0% to +15.0%), Projected RVOL ≥ 2.5x (or premarket volume pacing ≥ 5% of 20-day average), Theme Score ≥ 65.0 (Tailwind), and No Thematic Veto.
   - Group B: "OVERNIGHT GAPPER / MOVER (VOLUME CONFIRMATION WATCH)"
     * Criteria: Gap ≥ +3.0% but Projected RVOL < 2.0x, or Theme Score < 65.0. CHASE STRICTLY PROHIBITED AT OPEN.
   - Group C: "GAP-DOWN STOP LOSS VIOLATION (DEFENSIVE EXIT)"
     * Criteria: Any active portfolio holding trading below its invalidation stop in premarket. Exit immediately at 09:30 Market-on-Open (MOO).

2. TACTICAL INTRADAY ENTRY PROTOCOL:
   - For Gaps +4.0% to +12.0%: RECOMMEND 15-MINUTE OPENING RANGE BREAKOUT (15m ORB):
     * Do NOT buy at 09:30:00! Let the 09:30–09:45 15m candle establish the initial balance.
     * ORDER 1: Buy-Stop Limit placed 1 cent above the 15m High.
     * ORDER 2: Hard Stop-Loss placed 1 cent below the 15m Low.
     * Note: This reduces risk distance from the EOD baseline (~8.2%) to ~2.8%–3.5%, unlocking massive R-multiple leverage.
   - For Extreme Gaps (> +12.0%): RECOMMEND MORNING WASHOUT & VWAP RECLAIM:
     * Never chase at the open. Expect aggressive premarket profit-taking in the first 5–15 minutes.
     * Wait for morning flush to establish an exhaustion low and price to curl back up.
     * ORDER: Buy on 5-minute candle close reclaiming VWAP from below, with Hard Stop placed at the morning washout low.

3. ASYMMETRIC BIOTECH & TECH CAPITAL ALLOCATION:
   - Mega-Cap Tech / Software / Semis: Position size based on 1.0 R risk, capped at 25.0% portfolio equity weight.
   - Biotechnology / Speculative Genomics: Position size based on 1.0 R risk, strictly capped at 12.5% portfolio equity weight to protect fund capital against binary gap-downs.

4. REQUIRED OUTPUT DOSSIER FORMAT:
   - SECTION 1: TODAY'S PREMARKET APEX CANDIDATES (Ranked by institutional conviction).
     For each: Ticker, Catalyst Classification (Archetype 1 vs 2), Technical Metrics (Gap %, Premarket Vol, Projected RVOL, Theme Score), Exact Opening Order Ticket (15m ORB or VWAP Reclaim with specific price thresholds and stop distance), and Asymmetric Capital Weight Cap.
   - SECTION 2: OVERNIGHT GAPPERS ON WATCH (Conditions required for mid-day activation).
   - SECTION 3: DEFENSIVE RISK ALERTS (Any gap-down threats or sector headwinds).`;

  navigator.clipboard.writeText(masterPrompt)
    .then(() => alert("✅ Premarket AI Agent Prompt copied to clipboard!"))
    .catch(() => alert("Please copy manually from the station box below."));
}

function fetchPortfolio() {
  fetch('/api/portfolio')
    .then(r => r.json())
    .then(data => {
      portfolioData = data;
      renderPortfolioKPIs(data.summary);
      renderPortfolioAlerts(data.alerts);
      renderPortfolioHoldings(data.positions);
      renderPortfolioClosed(data.closed_positions);
      updateNavAlertBadge(data.summary.active_alerts_count);
    })
    .catch(e => console.error("Error fetching portfolio:", e));
}

function updateNavAlertBadge(count) {
  const badge = document.getElementById('portfolio_nav_badge');
  if (badge) {
    if (count > 0) {
      badge.textContent = count;
      badge.style.display = 'inline-block';
    } else {
      badge.style.display = 'none';
    }
  }
}

function renderPortfolioKPIs(sum) {
  if (!sum) return;
  document.getElementById('kpi_equity').textContent = `$${sum.account_equity.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
  document.getElementById('kpi_equity_sub').textContent = `Starting Capital: $${sum.portfolio_size.toLocaleString()}`;

  document.getElementById('kpi_cash').textContent = `$${sum.cash_available.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
  const unallocPct = sum.account_equity > 0 ? ((sum.cash_available / sum.account_equity) * 100).toFixed(1) : 0;
  document.getElementById('kpi_cash_sub').textContent = `${unallocPct}% unallocated cash`;

  document.getElementById('kpi_invested').textContent = `$${sum.total_invested.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
  document.getElementById('kpi_open_count').textContent = `${sum.open_count} Active Position${sum.open_count === 1 ? '' : 's'}`;
  document.getElementById('open_pos_count').textContent = sum.open_count;

  document.getElementById('kpi_heat').textContent = `${sum.open_heat_r >= 0 ? '+' : ''}${sum.open_heat_r.toFixed(2)} R`;
  document.getElementById('kpi_heat_sub').textContent = `$${sum.open_heat_dollars.toLocaleString()} at risk (${sum.open_heat_pct}%)`;

  const unpCol = sum.unrealized_pnl >= 0 ? 'var(--green)' : 'var(--red)';
  document.getElementById('kpi_unrealized').style.color = unpCol;
  document.getElementById('kpi_unrealized').textContent = `${sum.unrealized_r >= 0 ? '+' : ''}${sum.unrealized_r.toFixed(2)} R`;
  document.getElementById('kpi_unrealized_sub').textContent = `${sum.unrealized_pnl >= 0 ? '+' : ''}$${sum.unrealized_pnl.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})} Open`;

  const rpCol = sum.realized_pnl >= 0 ? 'var(--green)' : 'var(--red)';
  document.getElementById('kpi_realized').style.color = rpCol;
  document.getElementById('kpi_realized').textContent = `${sum.realized_r >= 0 ? '+' : ''}${sum.realized_r.toFixed(2)} R`;
  document.getElementById('kpi_win_rate').textContent = `${sum.closed_count} Closed (${sum.win_rate}% Win)`;
  document.getElementById('closed_pos_count').textContent = sum.closed_count;
}

function renderPortfolioAlerts(alerts) {
  const container = document.getElementById('portfolio_alerts_container');
  if (!container) return;

  if (!alerts || !alerts.length) {
    container.innerHTML = `
      <div style="background:#0f1523; border:1px solid #1e293b; border-radius:6px; padding:10px 16px; color:#94a3b8; font-size:12px; display:flex; align-items:center; gap:8px;">
        <span>🛡️</span>
        <span>No execution alerts pending. All open positions are respecting risk parameters.</span>
      </div>`;
    return;
  }

  container.innerHTML = alerts.map(a => {
    let actionBtn = '';
    if (a.suggested_action === 'close') {
      actionBtn = `<button class="alert-btn red" onclick="openCloseTradeModal('${a.pos_id}')">Exit Trade Now</button>`;
    } else if (a.suggested_action === 'secondary_add') {
      actionBtn = `<button class="alert-btn gold" onclick="openSecondaryAddModal('${a.pos_id}')">Execute +50% Add</button>`;
    } else if (a.suggested_action === 'adjust_stop') {
      actionBtn = `<button class="alert-btn blue" onclick="openAdjustStopModal('${a.pos_id}')">Trail Stop Loss</button>`;
    }

    return `
      <div class="alert-card ${a.type}">
        <div style="display:flex; align-items:center; gap:10px;">
          <span style="font-weight:700;">${a.title}:</span>
          <span>${a.message}</span>
        </div>
        <div>${actionBtn}</div>
      </div>`;
  }).join('');
}

function renderPortfolioHoldings(positions) {
  const tbody = document.getElementById('port_positions_body');
  if (!tbody) return;

  if (!positions || !positions.length) {
    tbody.innerHTML = `
      <tr>
        <td colspan="12" style="text-align:center; padding:30px; color:var(--muted);">
          No active positions in portfolio.<br>
          <span style="font-size:11px; color:#38bdf8; margin-top:4px; display:inline-block;">Go to "EP Scanner &amp; Catalysts" and click "Add to Live Portfolio" on any setup.</span>
        </td>
      </tr>`;
    return;
  }

  tbody.innerHTML = positions.map(p => {
    const pCol = p.unrealized_pnl_dollars >= 0 ? 'var(--green)' : 'var(--red)';
    const rCol = p.open_r >= 0 ? 'var(--green)' : 'var(--red)';

    return `
      <tr style="cursor:pointer;" onclick="openTickerInsightsModal('${p.id}')" title="Click to view handling plan & TradingView alert setup">
        <td class="tl" style="font-weight:800; color:#fff; font-size:13px;">
          <span style="color:#38bdf8; text-decoration:underline;">${p.symbol}</span>
        </td>
        <td class="tl">
          <div style="font-weight:600; color:#cbd5e1;">${p.company}</div>
          <div style="font-size:10.5px; color:var(--muted);">${p.sector} &bull; <span style="color:#c084fc;">${p.theme}</span></div>
        </td>
        <td>${p.entry_date}</td>
        <td>$${p.entry_price.toFixed(2)}</td>
        <td style="font-weight:700; color:#fff;">$${p.current_price.toFixed(2)}</td>
        <td>${p.shares.toLocaleString()}</td>
        <td>$${p.invested_dollars.toLocaleString()}</td>
        <td style="color:#ef4444; font-weight:700;">$${p.stop_price.toFixed(2)}</td>
        <td style="color:${rCol}; font-weight:700; font-size:12.5px;">${p.open_r >= 0 ? '+' : ''}${p.open_r.toFixed(2)} R</td>
        <td style="color:${pCol}; font-weight:600;">${p.unrealized_pnl_dollars >= 0 ? '+' : ''}$${p.unrealized_pnl_dollars.toFixed(2)} (${p.unrealized_pnl_pct >= 0 ? '+' : ''}${p.unrealized_pnl_pct.toFixed(1)}%)</td>
        <td class="tl"><span class="badge" style="background:#1e293b; color:#cbd5e1;">${p.status_stage}</span></td>
        <td style="text-align:center;" onclick="event.stopPropagation()">
          <div style="display:flex; gap:4px; justify-content:center;">
            <button onclick="openAdjustStopModal('${p.id}')" style="background:#0f2338; border:1px solid #0284c7; color:#38bdf8; padding:3px 6px; border-radius:4px; font-size:10px; cursor:pointer;" title="Modify Entry, Stop Loss, or Size">⚙️ Edit</button>
            <button onclick="openSecondaryAddModal('${p.id}')" style="background:#2e1065; border:1px solid #7c3aed; color:#c4b5fd; padding:3px 6px; border-radius:4px; font-size:10px; cursor:pointer;" title="Secondary Add (+50% size)">+ Add</button>
            <button onclick="openCloseTradeModal('${p.id}')" style="background:#3b1212; border:1px solid #dc2626; color:#fca5a5; padding:3px 6px; border-radius:4px; font-size:10px; cursor:pointer;" title="Close Position">✕ Close</button>
          </div>
        </td>
      </tr>`;
  }).join('');
}

function renderPortfolioClosed(closed) {
  const tbody = document.getElementById('port_closed_body');
  if (!tbody) return;

  if (!closed || !closed.length) {
    tbody.innerHTML = `
      <tr>
        <td colspan="11" style="text-align:center; padding:20px; color:var(--muted);">
          No closed trades yet.
        </td>
      </tr>`;
    return;
  }

  tbody.innerHTML = closed.map(c => {
    const pCol = c.realized_pnl_dollars >= 0 ? 'var(--green)' : 'var(--red)';
    const rCol = c.realized_r >= 0 ? 'var(--green)' : 'var(--red)';

    return `
      <tr>
        <td class="tl" style="font-weight:700; color:#fff;">${c.symbol}</td>
        <td class="tl" style="font-size:10.5px; color:var(--muted);">${c.sector} &bull; ${c.theme}</td>
        <td>${c.entry_date}</td>
        <td>${c.exit_date}</td>
        <td>${c.hold_days}d</td>
        <td>$${c.entry_price.toFixed(2)}</td>
        <td>$${c.exit_price.toFixed(2)}</td>
        <td>${c.shares.toLocaleString()}</td>
        <td style="color:${rCol}; font-weight:700;">${c.realized_r >= 0 ? '+' : ''}${c.realized_r.toFixed(2)} R</td>
        <td style="color:${pCol}; font-weight:600;">${c.realized_pnl_dollars >= 0 ? '+' : ''}$${c.realized_pnl_dollars.toFixed(2)}</td>
        <td class="tl"><span style="font-size:10.5px; color:#cbd5e1;">${c.exit_reason}</span></td>
        <td style="text-align:center;">
          <button onclick="deleteClosedTrade('${c.id}')" style="background:#2d1519; border:1px solid #7f1d1d; color:#fca5a5; padding:2px 6px; border-radius:4px; font-size:10px; cursor:pointer;" title="Delete this closed trade from history">✕ Delete</button>
        </td>
      </tr>`;
  }).join('');
}

function deleteClosedTrade(posId) {
  if (!confirm("Are you sure you want to delete this closed trade from history and tally?")) return;
  fetch('/api/portfolio/delete_closed_position', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id: posId })
  })
  .then(r => r.json())
  .then(res => {
    if (res.error) {
      alert("Error deleting closed trade: " + res.error);
    } else {
      fetchPortfolio();
    }
  })
  .catch(err => alert("Failed to delete closed trade: " + err.message));
}

function clearAllClosedPositions() {
  if (!confirm("Are you sure you want to delete ALL closed trades from history? This will reset the realized tally to 0.")) return;
  fetch('/api/portfolio/clear_closed_positions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({})
  })
  .then(r => r.json())
  .then(res => {
    if (res.error) {
      alert("Error clearing closed history: " + res.error);
    } else {
      fetchPortfolio();
    }
  })
  .catch(err => alert("Failed to clear closed history: " + err.message));
}

// ==========================================
// MODAL CONTROLLERS & FORM HANDLERS
// ==========================================
function closeModal(modalId) {
  const el = document.getElementById(modalId);
  if (el) el.classList.remove('open');
}

function closeModalOnOutside(event, modalId) {
  if (event.target.id === modalId) closeModal(modalId);
}

function openAddTickerModal() {
  currentAddModalEvent = null;
  document.getElementById('m_add_input_ticker_row').style.display = 'block';
  document.getElementById('m_add_ticker_input').value = '';
  document.getElementById('m_add_sym').textContent = 'ENTER TICKER';
  document.getElementById('m_add_comp').textContent = 'Lookup ticker below';
  document.getElementById('m_add_meta').textContent = 'General';
  document.getElementById('m_add_entry').value = '';
  document.getElementById('m_add_stop').value = '';
  document.getElementById('m_add_shares').value = '100';
  document.getElementById('m_add_capital').value = '$0.00';
  document.getElementById('m_add_risk_dollars').value = '$0.00';
  const defRisk = (portfolioData && portfolioData.settings && portfolioData.settings.default_risk_pct) || 1.0;
  document.getElementById('m_add_risk_pct').value = defRisk.toFixed(1);
  document.getElementById('m_add_notes').value = 'Discretionary / Portfolio Entry';
  document.getElementById('modal_add_portfolio').classList.add('open');
  setTimeout(() => document.getElementById('m_add_ticker_input').focus(), 150);
}

function fetchCustomTickerQuote() {
  const sym = (document.getElementById('m_add_ticker_input').value || '').trim().toUpperCase();
  if (!sym) {
    alert("Please enter a ticker symbol.");
    return;
  }
  fetch(`/api/symbol_quote?symbol=${encodeURIComponent(sym)}`)
    .then(r => r.json())
    .then(data => {
      if (data.error) {
        alert("Error looking up " + sym + ": " + data.error);
        return;
      }
      currentAddModalEvent = {
        symbol: data.symbol,
        company: data.company,
        sector: data.sector,
        theme: data.theme,
        event_date: data.date,
        close: data.close,
        d1_close: data.close,
        d1_low: data.low,
        d1_high: data.high
      };
      document.getElementById('m_add_sym').textContent = data.symbol;
      document.getElementById('m_add_comp').textContent = data.company;
      document.getElementById('m_add_meta').textContent = `${data.sector} • ${data.theme}`;
      document.getElementById('m_add_entry').value = data.close.toFixed(2);
      document.getElementById('m_add_stop').value = data.low.toFixed(2);
      document.getElementById('m_add_notes').value = `Manual Portfolio Position (${data.date})`;
      recalcAddModal();
    })
    .catch(err => alert("Failed to fetch quote for " + sym + ": " + err.message));
}

function openAddToPortfolioModal(sym, ev_dt) {
  let ev = null;
  if (sym && ev_dt) {
    ev = allEvents.find(e => e.symbol === sym && e.event_date === ev_dt);
  } else {
    ev = selectedEvent;
  }

  if (!ev) {
    openAddTickerModal();
    return;
  }

  currentAddModalEvent = ev;
  document.getElementById('m_add_input_ticker_row').style.display = 'none';
  document.getElementById('m_add_sym').textContent = ev.symbol;
  document.getElementById('m_add_comp').textContent = ev.company || ev.symbol;
  document.getElementById('m_add_meta').textContent = `${ev.sector} • ${ev.theme}`;

  const entryPx = ev.close || ev.d1_close || 10.0;
  const stopPx = ev.d1_low || (entryPx * 0.95);

  document.getElementById('m_add_entry').value = entryPx.toFixed(2);
  document.getElementById('m_add_stop').value = stopPx.toFixed(2);

  const defRisk = (portfolioData && portfolioData.settings && portfolioData.settings.default_risk_pct) || 1.0;
  document.getElementById('m_add_risk_pct').value = defRisk.toFixed(1);

  document.getElementById('m_add_notes').value = `Grade ${ev.quality_tier || 'A'} EP Catalyst (${ev.event_date})`;

  recalcAddModal();
  document.getElementById('modal_add_portfolio').classList.add('open');
}

let currentAddModalEvent = null;

function recalcAddModal() {
  const entry = parseFloat(document.getElementById('m_add_entry').value) || 0;
  const stop = parseFloat(document.getElementById('m_add_stop').value) || 0;
  const riskPct = parseFloat(document.getElementById('m_add_risk_pct').value) || 1.0;

  const portSize = (portfolioData && portfolioData.settings && portfolioData.settings.portfolio_size) || 100000.0;
  const dollarRisk = (portSize * (riskPct / 100.0));
  document.getElementById('m_add_risk_dollars').value = `$${dollarRisk.toFixed(2)} (${riskPct.toFixed(1)}% R)`;

  const sec = currentAddModalEvent ? (currentAddModalEvent.sector || '') : '';
  const thm = currentAddModalEvent ? (currentAddModalEvent.theme || '') : '';
  const isBiotech = /biotech|healthcare|pharma/i.test(sec + ' ' + thm);
  const maxCapFraction = isBiotech ? 0.125 : 0.25; // 12.5% max for biotech, 25% for others
  const maxAllowedCapital = portSize * maxCapFraction;

  const riskPerShare = entry - stop;
  if (riskPerShare > 0) {
    let suggestedShares = Math.floor(dollarRisk / riskPerShare);
    let capital = suggestedShares * entry;
    const warningEl = document.getElementById('m_add_gap_warning');

    if (capital > maxAllowedCapital) {
      suggestedShares = Math.max(1, Math.floor(maxAllowedCapital / entry));
      capital = suggestedShares * entry;
      if (warningEl) {
        warningEl.style.display = 'block';
        if (isBiotech) {
          warningEl.style.background = 'rgba(239, 68, 68, 0.15)';
          warningEl.style.border = '1px solid #ef4444';
          warningEl.style.color = '#fca5a5';
          warningEl.innerHTML = '⚠️ <b>Binary Biotech Gap Protection:</b> Capital capped at 12.5% ($' + maxAllowedCapital.toLocaleString() + ') to protect against overnight trial/CRL gap-down tail risk.';
        } else {
          warningEl.style.background = 'rgba(245, 158, 11, 0.15)';
          warningEl.style.border = '1px solid #f59e0b';
          warningEl.style.color = '#fcd34d';
          warningEl.innerHTML = '⚠️ <b>Portfolio Cap Enforced:</b> Position capped at 25.0% ($' + maxAllowedCapital.toLocaleString() + ') max capital allocation.';
        }
      }
    } else {
      if (warningEl) warningEl.style.display = 'none';
    }

    document.getElementById('m_add_shares').value = Math.max(1, suggestedShares);
    const capPct = ((capital / portSize) * 100).toFixed(1);
    document.getElementById('m_add_capital').value = `$${capital.toFixed(2)} (${capPct}% of Port)`;
  } else {
    document.getElementById('m_add_shares').value = 0;
    document.getElementById('m_add_capital').value = "$0.00";
    const warningEl = document.getElementById('m_add_gap_warning');
    if (warningEl) warningEl.style.display = 'none';
  }
}

function recalcAddFromShares() {
  const entry = parseFloat(document.getElementById('m_add_entry').value) || 0;
  const shares = parseInt(document.getElementById('m_add_shares').value) || 0;
  const portSize = (portfolioData && portfolioData.settings && portfolioData.settings.portfolio_size) || 100000.0;
  const capital = (shares * entry);
  const capPct = ((capital / portSize) * 100).toFixed(1);
  document.getElementById('m_add_capital').value = `$${capital.toFixed(2)} (${capPct}% of Port)`;
}

function submitAddToPortfolio() {
  const sym = document.getElementById('m_add_sym').textContent;
  const entryPx = parseFloat(document.getElementById('m_add_entry').value);
  const stopPx = parseFloat(document.getElementById('m_add_stop').value);
  const shares = parseInt(document.getElementById('m_add_shares').value);
  const riskPct = parseFloat(document.getElementById('m_add_risk_pct').value);
  const notes = document.getElementById('m_add_notes').value;

  if (!entryPx || !stopPx || !shares || stopPx >= entryPx) {
    alert("Please ensure valid Entry Price, Stop Loss (below entry), and Share count.");
    return;
  }

  const payload = {
    symbol: sym,
    entry_price: entryPx,
    stop_price: stopPx,
    shares: shares,
    risk_pct: riskPct,
    event_date: currentAddModalEvent ? (currentAddModalEvent.event_date || currentAddModalEvent.date) : (selectedEvent ? selectedEvent.event_date : null),
    notes: notes
  };

  fetch('/api/portfolio/add_position', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  .then(r => r.json())
  .then(res => {
    if (res.error) {
      alert("Error adding position: " + res.error);
    } else {
      closeModal('modal_add_portfolio');
      fetchPortfolio();
      switchMainView('portfolio');
    }
  })
  .catch(err => alert("Failed to add position: " + err.message));
}

// Adjust Stop Modal Handlers
let activeAdjustPos = null;

function openAdjustStopModal(posId) {
  if (!portfolioData || !portfolioData.positions) return;
  const pos = portfolioData.positions.find(p => p.id === posId);
  if (!pos) return;

  activeAdjustPos = pos;
  document.getElementById('m_adj_pos_id').value = pos.id;
  document.getElementById('m_adj_sym').textContent = pos.symbol;
  document.getElementById('m_adj_curr_stop').textContent = `$${pos.stop_price.toFixed(2)}`;
  document.getElementById('m_adj_curr_px').textContent = `$${pos.current_price.toFixed(2)}`;
  document.getElementById('m_adj_curr_r').textContent = `${pos.open_r >= 0 ? '+' : ''}${pos.open_r.toFixed(2)} R`;

  document.getElementById('m_adj_entry_input').value = pos.entry_price.toFixed(2);
  document.getElementById('m_adj_new_stop').value = pos.stop_price.toFixed(2);
  document.getElementById('m_adj_shares_input').value = pos.shares;

  document.getElementById('q_be_val').textContent = `$${pos.entry_price.toFixed(2)}`;
  document.getElementById('q_ema20_val').textContent = pos.ema20 ? `$${pos.ema20.toFixed(2)}` : 'N/A';
  document.getElementById('q_sma50_val').textContent = pos.sma50 ? `$${pos.sma50.toFixed(2)}` : 'N/A';

  document.getElementById('m_adj_notes').value = "";

  // Render executed add tranches if any
  const tranchesSec = document.getElementById('m_adj_tranches_section');
  const tranchesList = document.getElementById('m_adj_tranches_list');
  if (tranchesSec && tranchesList) {
    if (pos.adds && pos.adds.length > 0) {
      tranchesSec.style.display = 'block';
      tranchesList.innerHTML = pos.adds.map((a, idx) => `
        <div style="display:flex; justify-content:space-between; align-items:center; background:#131824; padding:6px 8px; border-radius:4px; font-size:11px;">
          <div>
            <span style="color:#fbbf24; font-weight:700;">Tranche #${idx + 1} (${a.type || 'ADD'})</span>:
            <span style="color:#cbd5e1;">+${a.shares} sh @ $${parseFloat(a.price).toFixed(2)}</span>
            <span style="color:var(--muted); font-size:10px;">(${a.date || ''})</span>
          </div>
          <div style="display:flex; align-items:center; gap:6px;">
            <span style="color:#94a3b8; font-size:10px;">Tranche Stop: $</span>
            <input type="number" step="0.01" class="input-field tranche-stop-input" data-idx="${idx}" value="${(a.stop_price || pos.stop_price).toFixed(2)}" style="width:75px; padding:2px 4px; font-size:11px; height:24px; text-align:right;">
          </div>
        </div>
      `).join('');
    } else {
      tranchesSec.style.display = 'none';
      tranchesList.innerHTML = '';
    }
  }

  recalcModifyModal();

  document.getElementById('modal_adjust_stop').classList.add('open');
}

function recalcModifyModal() {
  const entry = parseFloat(document.getElementById('m_adj_entry_input').value) || 0;
  const stop = parseFloat(document.getElementById('m_adj_new_stop').value) || 0;
  const shares = parseInt(document.getElementById('m_adj_shares_input').value) || 0;

  const capital = shares * entry;
  document.getElementById('m_adj_capital_val').value = `$${capital.toFixed(2)}`;

  const riskPerShare = entry - stop;
  const totalRisk = riskPerShare * shares;
  document.getElementById('m_adj_risk_val').value = `$${totalRisk.toFixed(2)}`;

  const distPct = entry > 0 ? ((entry - stop) / entry * 100).toFixed(1) : '0.0';
  document.getElementById('m_adj_dist_val').value = `${distPct}%`;
}

function setQuickStop(type) {
  if (!activeAdjustPos) return;
  if (type === 'be') {
    document.getElementById('m_adj_new_stop').value = activeAdjustPos.entry_price.toFixed(2);
    document.getElementById('m_adj_notes').value = "Stop locked at Breakeven";
  } else if (type === 'ema20' && activeAdjustPos.ema20) {
    document.getElementById('m_adj_new_stop').value = activeAdjustPos.ema20.toFixed(2);
    document.getElementById('m_adj_notes').value = "Trailed stop to rising 20 EMA";
  } else if (type === 'sma50' && activeAdjustPos.sma50) {
    document.getElementById('m_adj_new_stop').value = activeAdjustPos.sma50.toFixed(2);
    document.getElementById('m_adj_notes').value = "Trailed stop to institutional 50 SMA";
  }
  recalcModifyModal();
}

function submitModifyPosition() {
  const posId = document.getElementById('m_adj_pos_id').value;
  const entryPx = parseFloat(document.getElementById('m_adj_entry_input').value);
  const newStop = parseFloat(document.getElementById('m_adj_new_stop').value);
  const shares = parseInt(document.getElementById('m_adj_shares_input').value);
  const notes = document.getElementById('m_adj_notes').value;

  if (!entryPx || entryPx <= 0 || !newStop || newStop <= 0 || !shares || shares <= 0) {
    alert("Please enter valid positive values for Entry Price, Stop Loss, and Position Size.");
    return;
  }

  if (newStop >= entryPx) {
    alert("Stop loss must be below entry price for long EP position.");
    return;
  }

  const trancheInputs = document.querySelectorAll('#m_adj_tranches_list .tranche-stop-input');
  const trancheStops = [];
  trancheInputs.forEach(inp => {
    const v = parseFloat(inp.value);
    if (!isNaN(v) && v > 0) trancheStops.push(v);
  });

  fetch('/api/portfolio/update_position', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      id: posId,
      action: 'modify_position',
      entry_price: entryPx,
      stop_price: newStop,
      add_shares: shares,
      tranche_stops: trancheStops.length > 0 ? trancheStops : null,
      notes: notes
    })
  })
  .then(r => r.json())
  .then(res => {
    if (res.error) alert("Error: " + res.error);
    else {
      closeModal('modal_adjust_stop');
      fetchPortfolio();
    }
  })
  .catch(err => alert("Error: " + err.message));
}

// Secondary Add Modal Handlers
function openSecondaryAddModal(posId) {
  if (!portfolioData || !portfolioData.positions) return;
  const pos = portfolioData.positions.find(p => p.id === posId);
  if (!pos) return;

  document.getElementById('m_add_sec_pos_id').value = pos.id;
  document.getElementById('m_add_sec_sym').textContent = pos.symbol;
  document.getElementById('m_sec_d1_high').textContent = `$${pos.d1_high.toFixed(2)}`;

  document.getElementById('m_sec_price').value = pos.current_price.toFixed(2);
  // Default add stop to Day 1 Low or current stop
  const defaultAddStop = (pos.d1_low && pos.d1_low > 0) ? pos.d1_low : pos.stop_price;
  document.getElementById('m_sec_stop').value = defaultAddStop.toFixed(2);

  const addShares = Math.max(1, Math.floor(pos.shares * 0.5));
  document.getElementById('m_sec_shares').value = addShares;

  recalcSecondaryAddModal();
  document.getElementById('modal_secondary_add').classList.add('open');
}

function recalcSecondaryAddModal() {
  const addPx = parseFloat(document.getElementById('m_sec_price').value) || 0;
  const addStop = parseFloat(document.getElementById('m_sec_stop').value) || 0;
  const addShares = parseInt(document.getElementById('m_sec_shares').value) || 0;

  const feedbackEl = document.getElementById('m_sec_risk_feedback');
  if (!feedbackEl) return;

  if (addPx > 0 && addStop > 0 && addPx > addStop && addShares > 0) {
    const riskDollars = (addPx - addStop) * addShares;
    const riskPct = ((addPx - addStop) / addPx * 100);
    feedbackEl.value = `$${riskDollars.toFixed(2)} (${riskPct.toFixed(1)}% risk)`;
    feedbackEl.style.color = '#f87171';
  } else if (addStop >= addPx && addPx > 0) {
    feedbackEl.value = "Stop must be below fill price";
    feedbackEl.style.color = '#ef4444';
  } else {
    feedbackEl.value = "$0.00";
    feedbackEl.style.color = '#94a3b8';
  }
}

function submitSecondaryAdd() {
  const posId = document.getElementById('m_add_sec_pos_id').value;
  const addPx = parseFloat(document.getElementById('m_sec_price').value);
  const addStop = parseFloat(document.getElementById('m_sec_stop').value);
  const addShares = parseInt(document.getElementById('m_sec_shares').value);
  const notes = document.getElementById('m_sec_notes').value;

  if (!addPx || !addShares || addShares <= 0) {
    alert("Invalid price or shares for secondary add.");
    return;
  }
  if (!addStop || addStop <= 0 || addStop >= addPx) {
    alert("Please provide a valid Stop Loss below the Add Fill Price.");
    return;
  }

  fetch('/api/portfolio/update_position', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      id: posId,
      action: 'secondary_add',
      add_price: addPx,
      add_stop: addStop,
      add_shares: addShares,
      notes: notes
    })
  })
  .then(r => r.json())
  .then(res => {
    if (res.error) alert("Error: " + res.error);
    else {
      closeModal('modal_secondary_add');
      fetchPortfolio();
    }
  })
  .catch(err => alert("Error: " + err.message));
}

// Close Position Modal Handlers
function openCloseTradeModal(posId) {
  if (!portfolioData || !portfolioData.positions) return;
  const pos = portfolioData.positions.find(p => p.id === posId);
  if (!pos) return;

  document.getElementById('m_close_pos_id').value = pos.id;
  document.getElementById('m_close_sym').textContent = pos.symbol;
  document.getElementById('m_close_px').value = pos.current_price.toFixed(2);

  const todayStr = new Date().toISOString().split('T')[0];
  document.getElementById('m_close_date').value = todayStr;
  document.getElementById('m_close_notes').value = "";

  document.getElementById('modal_close_trade').classList.add('open');
}

function submitCloseTrade() {
  const posId = document.getElementById('m_close_pos_id').value;
  const exitPx = parseFloat(document.getElementById('m_close_px').value);
  const exitDate = document.getElementById('m_close_date').value;
  const reason = document.getElementById('m_close_reason').value;
  const notes = document.getElementById('m_close_notes').value;

  if (!exitPx || exitPx <= 0) {
    alert("Please enter a valid exit price.");
    return;
  }

  fetch('/api/portfolio/close_position', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id: posId, exit_price: exitPx, exit_date: exitDate, exit_reason: reason, notes: notes })
  })
  .then(r => r.json())
  .then(res => {
    if (res.error) alert("Error: " + res.error);
    else {
      closeModal('modal_close_trade');
      fetchPortfolio();
    }
  })
  .catch(err => alert("Error: " + err.message));
}

// Portfolio Settings Modal Handlers
function openPortfolioSettingsModal() {
  const pSize = (portfolioData && portfolioData.settings && portfolioData.settings.portfolio_size) || 100000.0;
  const defRisk = (portfolioData && portfolioData.settings && portfolioData.settings.default_risk_pct) || 1.0;

  document.getElementById('m_set_port_size').value = pSize;
  document.getElementById('m_set_risk_pct').value = defRisk;

  document.getElementById('modal_portfolio_settings').classList.add('open');
}

function submitPortfolioSettings() {
  const size = parseFloat(document.getElementById('m_set_port_size').value);
  const riskPct = parseFloat(document.getElementById('m_set_risk_pct').value);

  if (!size || size <= 0 || !riskPct || riskPct <= 0) {
    alert("Please enter valid positive values for portfolio size and risk.");
    return;
  }

  fetch('/api/portfolio/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ portfolio_size: size, default_risk_pct: riskPct })
  })
  .then(r => r.json())
  .then(res => {
    if (res.error) alert("Error: " + res.error);
    else {
      closeModal('modal_portfolio_settings');
      fetchPortfolio();
    }
  })
  .catch(err => alert("Error: " + err.message));
}

// Ticker Insights & TradingView Alert Generator Handlers
let activeInsightsPos = null;

function openTickerInsightsModal(posId) {
  if (!portfolioData || !portfolioData.positions) return;
  const pos = portfolioData.positions.find(p => p.id === posId);
  if (!pos) return;

  activeInsightsPos = pos;
  document.getElementById('m_ins_sym').textContent = pos.symbol;
  document.getElementById('m_ins_comp').textContent = `${pos.company || pos.symbol} • ${pos.sector} &bull; ${pos.theme}`;

  document.getElementById('m_ins_entry').textContent = `$${pos.entry_price.toFixed(2)}`;
  document.getElementById('m_ins_stop').textContent = `$${pos.stop_price.toFixed(2)}`;
  document.getElementById('m_ins_d1_high').textContent = `$${pos.d1_high.toFixed(2)}`;

  const riskPerShare = Math.max(0.01, pos.entry_price - pos.stop_price);
  const target3R = pos.entry_price + (3.0 * riskPerShare);
  const target2R = pos.entry_price + (2.0 * riskPerShare);
  document.getElementById('m_ins_target_3r').textContent = `$${target3R.toFixed(2)}`;

  document.getElementById('m_ins_badge_stage').textContent = pos.status_stage;
  document.getElementById('m_ins_badge_theme').textContent = pos.theme || 'General';
  document.getElementById('m_ins_badge_risk').textContent = `${pos.open_r >= 0 ? '+' : ''}${pos.open_r.toFixed(2)} R (${pos.unrealized_pnl_pct >= 0 ? '+' : ''}${pos.unrealized_pnl_pct.toFixed(1)}%)`;

  // Build Insight Guide Content
  let guide = `📌 POSITION CONTEXT & CATALYST ANATOMY:
• Entry: $${pos.entry_price.toFixed(2)} | Active Stop: $${pos.stop_price.toFixed(2)} (-${((pos.entry_price - pos.stop_price) / pos.entry_price * 100).toFixed(1)}% risk).
• Current Status: ${pos.status_stage} | Day ${pos.bars_since + 1} of post-breakout life.
• Group Momentum: ${pos.theme} in ${pos.sector}.

🎯 QUANTITATIVE EXECUTION & HANDLING RULES:`;

  if (pos.bars_since <= 3 && !pos.has_d2_add) {
    guide += `\n1. SECONDARY ADD: If price breaks above Day 1 High ($${pos.d1_high.toFixed(2)}), scale +50% size (+${Math.max(1, Math.floor(pos.shares * 0.5))} shares). Lock stop at Day 1 Low ($${pos.stop_price.toFixed(2)}).`;
  } else if (pos.has_d2_add) {
    guide += `\n1. SECONDARY ADD: Executed. Position scaled to 1.5x heat. Hold core runner.`;
  }

  guide += `\n2. 48-HOUR ABSORPTION: Monitor upper 50% body. Sustained hold confirms institutional accumulation.`;
  guide += `\n3. TRAILING STOP RULE: Once gain reaches +2.5 R ($${(pos.entry_price + 2.5 * riskPerShare).toFixed(2)}), trail stop to Breakeven ($${pos.entry_price.toFixed(2)}).`;
  if (pos.ema20) {
    guide += `\n4. SWING TRAIL: Trail runner along daily 20 EMA ($${pos.ema20.toFixed(2)}).`;
  }
  if (pos.sma50) {
    guide += `\n5. INSTITUTIONAL BASELINE: 50-day SMA ($${pos.sma50.toFixed(2)}). 10-year compounders hold this baseline 70%+ of the time.`;
  }
  guide += `\n6. EXIT TRIGGER: Discretionary exit on first daily Larsson Line Blue Flip or hard stop violation.`;

  document.getElementById('m_ins_guide_content').textContent = guide;

  // Build TradingView Alerts
  const tvAlerts = [
    {
      title: "⚡ TV Alert 1: Day 1 High Breakout / Secondary Add",
      condition: `${pos.symbol} Crossing Up $${pos.d1_high.toFixed(2)}`,
      msg: `${pos.symbol} crossed D1 High ($${pos.d1_high.toFixed(2)}). Secondary Add trigger active! Scale +50% size.`
    },
    {
      title: "🛑 TV Alert 2: Invalidation Hard Stop",
      condition: `${pos.symbol} Crossing Down $${pos.stop_price.toFixed(2)}`,
      msg: `${pos.symbol} breached hard stop ($${pos.stop_price.toFixed(2)}). Exit immediately to preserve capital.`
    },
    {
      title: "🎯 TV Alert 3: Free Trade Milestone (+2.5 R)",
      condition: `${pos.symbol} Crossing Up $${(pos.entry_price + 2.5 * riskPerShare).toFixed(2)}`,
      msg: `${pos.symbol} hit +2.5 R milestone ($${(pos.entry_price + 2.5 * riskPerShare).toFixed(2)}). Move stop to Breakeven ($${pos.entry_price.toFixed(2)}).`
    },
    {
      title: "🚀 TV Alert 4: Power Target (+3.0 R)",
      condition: `${pos.symbol} Crossing Up $${target3R.toFixed(2)}`,
      msg: `${pos.symbol} reached +3.0 R profit target ($${target3R.toFixed(2)}). Consider locking partial profits.`
    }
  ];

  if (pos.ema20) {
    tvAlerts.push({
      title: "📉 TV Alert 5: 20 EMA Swing Violation",
      condition: `${pos.symbol} Crossing Down $${pos.ema20.toFixed(2)}`,
      msg: `${pos.symbol} dropped below rising 20 EMA ($${pos.ema20.toFixed(2)}). Trail stop tighter or review exit.`
    });
  }

  const alertsContainer = document.getElementById('m_ins_tv_alerts_list');
  alertsContainer.innerHTML = tvAlerts.map((a, i) => `
    <div style="background:#0f1422; border:1px solid #1e293b; border-radius:6px; padding:10px 12px; font-size:11.5px;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
        <span style="font-weight:700; color:#38bdf8;">${a.title}</span>
        <button onclick="copyTvAlertText('${encodeURIComponent(a.condition)}', '${encodeURIComponent(a.msg)}')" style="background:#1e293b; border:1px solid #334155; color:#cbd5e1; padding:2px 8px; border-radius:4px; font-size:10px; cursor:pointer;">📋 Copy Setup</button>
      </div>
      <div style="color:#cbd5e1; font-family:monospace; margin-bottom:2px;">• Condition: <b>${a.condition}</b></div>
      <div style="color:var(--muted); font-size:10.5px;">• Alert Message: "${a.msg}"</div>
    </div>
  `).join('');

  document.getElementById('modal_ticker_insights').classList.add('open');
}

function copyTvAlertText(condEnc, msgEnc) {
  const cond = decodeURIComponent(condEnc);
  const msg = decodeURIComponent(msgEnc);
  const text = `TradingView Alert Setup:\nCondition: ${cond}\nMessage: ${msg}`;
  navigator.clipboard.writeText(text)
    .then(() => alert("✅ TradingView alert parameters copied to clipboard!"))
    .catch(() => alert(`TradingView Alert:\n${text}`));
}

function openModifyFromInsights() {
  if (!activeInsightsPos) return;
  closeModal('modal_ticker_insights');
  openAdjustStopModal(activeInsightsPos.id);
}

// Execute immediately upon parsing and listen to lifecycle events
fetchEvents();
fetchEmergingBoard();
fetchPortfolio();
fetchPremarketData(false);
initChart();

if (window.location.hash === '#portfolio' || window.location.pathname === '/portfolio') {
  switchMainView('portfolio');
} else if (window.location.hash === '#premarket' || window.location.pathname === '/premarket') {
  switchMainView('premarket');
}

document.addEventListener('DOMContentLoaded', () => {
  if (!allEvents || !allEvents.length) fetchEvents();
  if (!emergingBoard) fetchEmergingBoard();
  if (!portfolioData) fetchPortfolio();
  if (!premarketData) fetchPremarketData(false);
  if (!chart) initChart();
  if (window.location.hash === '#portfolio' || window.location.pathname === '/portfolio') {
    switchMainView('portfolio');
  } else if (window.location.hash === '#premarket' || window.location.pathname === '/premarket') {
    switchMainView('premarket');
  }
});

window.addEventListener('load', () => {
  if (!allEvents || !allEvents.length) fetchEvents();
  if (!emergingBoard) fetchEmergingBoard();
  if (!portfolioData) fetchPortfolio();
  if (!premarketData) fetchPremarketData(false);
  if (!chart) initChart();
  if (window.location.hash === '#portfolio' || window.location.pathname === '/portfolio') {
    switchMainView('portfolio');
  } else if (window.location.hash === '#premarket' || window.location.pathname === '/premarket') {
    switchMainView('premarket');
  }
});
function onAiTopPicksClick() {
  setView('top_ai');
  openAiTopPicksModal();
}

function onAiTopPickSelect(sym, dt) {
  closeModal('modal_ai_top_picks');
  setView('top_ai');
  const ev = allEvents.find(e => e.symbol === sym && (e.event_date === dt || e.date === dt)) || allEvents.find(e => e.symbol === sym);
  if (ev) {
    selectEvent(ev);
  }
}

function openAiTopPicksModal() {
  document.getElementById('modal_ai_top_picks').classList.add('open');
  document.getElementById('ai_top_picks_content').innerHTML = '<div style="text-align:center; padding:40px; color:#94a3b8;">Processing Live ML Inference...</div>';
  
  fetch('/api/ai_top_picks')
    .then(r => r.json())
    .then(data => {
      let html = '';
      
      const renderGroup = (title, arr, isTrap) => {
        if (!arr || !arr.length) return '';
        let grpHtml = `<div style="margin-bottom:20px;">
          <h4 style="margin-bottom:10px; color:${isTrap ? '#ef4444' : '#34d399'}; border-bottom:1px solid #334155; padding-bottom:5px;">${title}</h4>`;
          
        arr.forEach(e => {
          const bdgColor = isTrap ? 'var(--red)' : '#0ea5e9';
          grpHtml += `
            <div style="background:#1e293b; padding:10px; margin-bottom:8px; border-radius:4px; border-left:4px solid ${isTrap ? '#ef4444' : '#10b981'}; display:flex; justify-content:space-between; align-items:center; cursor:pointer;" onclick="onAiTopPickSelect('${e.symbol}', '${e.date}')" title="Click to view ${e.symbol} on chart & dossier">
              <div>
                <div style="font-weight:700; font-size:14px; color:#fff;">${e.symbol} <span style="font-size:10px; background:${bdgColor}; padding:2px 5px; border-radius:3px; margin-left:6px;">${e.archetype}</span></div>
                <div style="font-size:11px; color:var(--muted); margin-top:3px;">Age: ${e.days_old} Days | 50% Win Prob: <b style="color:var(--green);">${e.prob_50}%</b> | 100% Target: <b style="color:#38bdf8;">${e.prob_100}%</b></div>
                ${e.caution ? `<div style="font-size:11px; color:#fca5a5; margin-top:4px;">${e.caution}</div>` : ''}
                ${(e.alerts && e.alerts.length) ? `<div style="font-size:11px; color:#6ee7b7; margin-top:4px;">${e.alerts.join(' | ')}</div>` : ''}
              </div>
              <div style="text-align:right;">
                <div style="font-size:10px; color:var(--muted);">AI Score</div>
                <div style="font-size:18px; font-weight:700; color:${isTrap ? 'var(--red)' : 'var(--green)'};">${e.score}</div>
              </div>
            </div>`;
        });
        grpHtml += `</div>`;
        return grpHtml;
      };
      
      html += renderGroup('🔥 Top Fresh Ignitions (<= 2 Days Old)', data.fresh, false);
      html += renderGroup('🛡️ Top Mature Leaders (Holding Support)', data.mature, false);
      html += renderGroup('⚠️ Traps & Headwinds (Strict Avoidance)', data.traps, true);
      
      document.getElementById('ai_top_picks_content').innerHTML = html;
    })
    .catch(err => {
      document.getElementById('ai_top_picks_content').innerHTML = `<div style="color:var(--red); text-align:center;">Error loading AI predictions.</div>`;
    });
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return Response(HTML_TEMPLATE, mimetype="text/html")

def get_scored_tracker_events(force_rescan=False):
    events = get_tracker_data(force_rescan=force_rescan)
    from ep_ml_engine import engine
    import datastore
    import pandas as pd
    
    for ev in events:
        sym = ev["symbol"]
        dt_str = ev.get("date") or ev.get("event_date")
        bars = datastore.load_bars(sym)
        ev["ml_50"] = None
        ev["ml_100"] = None
        ev["ml_150"] = None
        ev["ml_200"] = None
        if bars is not None and dt_str in bars.index:
            try:
                d1_idx = bars.index.get_loc(dt_str)
                t_idx = len(bars) - 1
                v2_feats = engine.compute_rolling_features(sym, d1_idx, t_idx=t_idx)
                if v2_feats:
                    preds = engine.predict_rolling(v2_feats)
                    ev["ml_50"] = preds.get("prob_50")
                    ev["ml_100"] = preds.get("prob_100")
                    ev["ml_150"] = preds.get("prob_150")
                    ev["ml_200"] = preds.get("prob_200")
                    ev["is_toxic"] = preds.get("is_toxic", False)
                    ev["dynamic_stop_loss_pct"] = preds.get("dynamic_stop_loss_pct", None)
                    ev["prob_exhaustion"] = preds.get("prob_exhaustion", None)
                    ev["prob_reentry"] = preds.get("prob_reentry", None)

                    # Continuation Runner Dynamic Trailing Telemetry
                    d1_low = float(bars["low"].iloc[d1_idx])
                    d1_close = float(bars["close"].iloc[d1_idx])
                    risk_pts = d1_close - d1_low
                    high_since = float(bars["high"].iloc[d1_idx:].max())
                    open_r = (high_since - d1_close) / risk_pts if risk_pts > 0 else 0.0
                    if open_r >= 2.5:
                        c_feats = engine.compute_continuation_trailer_features(bars, len(bars) - 1, d1_close, high_since, risk_pts, entry_bar=d1_idx)
                        if c_feats:
                            buf = engine.predict_continuation_trailer(c_feats)
                            ev["continuation_trailer_buffer_pct"] = buf
                            ev["continuation_trailer_stop"] = round(high_since * (1.0 - buf / 100.0), 2)
            except Exception:
                pass
    return events

@app.route("/api/continuation_qualify")
def api_continuation_qualify():
    sym = request.args.get("symbol")
    event_date = request.args.get("event_date")
    if not sym or not event_date:
        return jsonify({"error": "Missing symbol or event_date"}), 400
    import datastore
    from ep_ml_engine import engine
    bars = datastore.load_bars(sym)
    if bars is None or event_date not in bars.index:
        return jsonify({"error": "Symbol or event date not found"}), 404
    d1_idx = bars.index.get_loc(event_date)
    d1_low = float(bars["low"].iloc[d1_idx])
    t1_exit = None
    for b in range(d1_idx + 1, min(len(bars), d1_idx + 11)):
        if bars["low"].iloc[b] <= d1_low:
            t1_exit = b; break
    if t1_exit is None:
        return jsonify({"has_ur_candidate": False, "reason": "No Day 1 Low breach within 10 days"})
    shakeout_low = float(bars["low"].iloc[t1_exit])
    ur_signal = None
    for b in range(t1_exit + 1, min(len(bars), t1_exit + 16)):
        l_b = float(bars["low"].iloc[b])
        if l_b < shakeout_low:
            shakeout_low = l_b
        if shakeout_low < d1_low * 0.85:
            return jsonify({"has_ur_candidate": False, "reason": "Shakeout breached 15% limit"})
        if bars["close"].iloc[b] >= d1_low:
            ur_signal = b; break
    if ur_signal is None:
        return jsonify({"has_ur_candidate": False, "reason": "No reclaim above Day 1 Low"})
    ur_feats = engine.compute_ur_reentry_features(sym, d1_idx, t1_exit, ur_signal, shakeout_low, df=bars)
    pred = engine.predict_ur_reentry(ur_feats) if ur_feats else {"prob_win": 0.5, "is_qualified": True}
    return jsonify({
        "has_ur_candidate": True,
        "signal_date": str(bars.index[ur_signal].date()),
        "shakeout_low": round(shakeout_low, 2),
        "features": ur_feats,
        "qualification": pred
    })

@app.route("/api/tracker_events")
def api_tracker_events():
    events = get_scored_tracker_events(force_rescan=False)
    return jsonify({"events": events, "count": len(events)})

@app.route("/api/emerging_board")
def api_emerging_board():
    eng = thematic_engine.get_thematic_engine()
    return jsonify(eng.get_emerging_summary())

@app.route("/api/group_stocks")
def api_group_stocks():
    group = request.args.get("group", "")
    kind = request.args.get("kind", "theme")
    if not group:
        return jsonify({"error": "Missing group"}), 400
    eng = thematic_engine.get_thematic_engine()
    return jsonify(eng.get_group_stocks(group, kind))

@app.route("/api/rescan")
def api_rescan():
    events = get_tracker_data(force_rescan=True)
    return jsonify({"events": events, "count": len(events)})

@app.route("/api/export/tradingview")
def api_export_tradingview():
    events = get_tracker_data(force_rescan=False)
    view = request.args.get("view", "all")
    if view == "today":
        events = [e for e in events if e.get("bars_since") == 0]
    elif view == "week":
        events = [e for e in events if e.get("bars_since", 999) <= 5]
    elif view == "active":
        events = [e for e in events if e.get("held_d1_low")]
    elif view == "idiosyncratic":
        events = [e for e in events if e.get("is_idiosyncratic")]
    
    text = ",\n".join(e["symbol"] for e in events)
    return Response(
        text,
        mimetype="text/plain",
        headers={"Content-Disposition": f"attachment; filename=tradingview_ep_watch_{view}.txt"}
    )

@app.route("/api/export/ai_json")
def api_export_ai_json():
    events = get_tracker_data(force_rescan=False)
    view = request.args.get("view", "all")
    if view == "today":
        events = [e for e in events if e.get("bars_since") == 0]
    elif view == "week":
        events = [e for e in events if e.get("bars_since", 999) <= 5]
    elif view == "active":
        events = [e for e in events if e.get("held_d1_low")]
    elif view == "idiosyncratic":
        events = [e for e in events if e.get("is_idiosyncratic")]

    payload = {
        "metadata": {
            "pipeline_view": view,
            "total_count": len(events),
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "market_date": "2026-10-02"
        },
        "setups": events
    }
    return jsonify(payload)

@app.route("/api/premarket/data")
def api_premarket_data():
    force = request.args.get("force", "0") == "1"
    data = ep_premarket.get_premarket_data(force_refresh=force)
    return jsonify(data)

@app.route("/api/premarket/scan", methods=["GET", "POST"])
def api_premarket_scan():
    data = ep_premarket.scan_premarket(force_refresh=True)
    return jsonify(data)

@app.route("/api/export/premarket_json")
def api_export_premarket_json():
    data = ep_premarket.get_premarket_data(force_refresh=False)
    return jsonify(data)


@app.route("/api/ml_rolling")
def api_ml_rolling():
    sym = request.args.get("symbol")
    event_date = request.args.get("event_date")
    days_forward = int(request.args.get("days_forward", "5"))
    from ep_ml_engine import engine
    import datastore
    bars = datastore.load_bars(sym)
    if bars is None or event_date not in bars.index: return jsonify({"error": "No bars"})
    d1_idx = bars.index.get_loc(event_date)
    t_idx = d1_idx + days_forward
    features = engine.compute_rolling_features(sym, d1_idx, t_idx)
    if not features: return jsonify({"stopped_out": True})
    probs = engine.predict_rolling(features)
    return jsonify({"probs": probs})

@app.route("/api/chart")

def api_chart():
    sym = request.args.get("symbol", "").upper()
    ev_dt = request.args.get("event_date", "")
    if not sym:
        return jsonify({"error": "Missing symbol"}), 400

    raw = datastore.load_bars(sym)
    if raw is None or len(raw) == 0:
        return jsonify({"error": "No bars"}), 404

    d = indicators.add_indicators(raw)
    
    # Slice to last 40 bars
    sub = d.iloc[max(0, len(d) - 40):]
    
    bars = []
    vol = []
    ema20 = []
    for idx, r in sub.iterrows():
        t = idx.strftime("%Y-%m-%d")
        o, h, l, c = float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])
        v = float(r["volume"]) if pd.notna(r["volume"]) else 0
        bars.append({"time": t, "open": o, "high": h, "low": l, "close": c})
        vol.append({"time": t, "value": v, "color": "rgba(16,185,129,0.3)" if c >= o else "rgba(239,68,68,0.3)"})
        if "ema20" in r and pd.notna(r["ema20"]):
            ema20.append({"time": t, "value": round(float(r["ema20"]), 2)})

    return jsonify({"symbol": sym, "bars": bars, "volume": vol, "ema20": ema20})

import ep_predictor

@app.route("/api/predict")
def api_predict():
    sym = request.args.get("sym")
    date_str = request.args.get("date")
    
    events = get_tracker_data()
    if not events:
        return jsonify({"error": "No data"})
        
    features = next((e for e in events if e.get("symbol") == sym and (e.get("date") == date_str or e.get("event_date") == date_str)), None)
    
    if not features:
        return jsonify({"error": "Event not found"})
        
    # For live tracking, "days_since_ep" is important for the 5D booster
    dt_str = features.get("date") or features.get("event_date")
    if dt_str:
        import datetime
        ep_dt = pd.to_datetime(dt_str).tz_localize(None)
        now_dt = pd.Timestamp.now().tz_localize(None)
        features["days_since_ep"] = (now_dt - ep_dt).days
        
    prediction = ep_predictor.get_predictions(features)
    return jsonify(prediction)

@app.route("/api/symbol_quote")
def api_symbol_quote():
    sym = request.args.get("symbol", "").strip().upper()
    if not sym:
        return jsonify({"error": "No symbol provided"}), 400
    try:
        raw = datastore.load_bars(sym)
        if raw is None or raw.empty:
            return jsonify({"error": f"No bar data available for symbol {sym}"}), 404
        d = indicators.add_indicators(raw)
        last_bar = d.iloc[-1]
        prev_bar = d.iloc[-2] if len(d) > 1 else last_bar
        close_px = round(float(last_bar["close"]), 2)
        open_px = round(float(last_bar["open"]), 2)
        high_px = round(float(last_bar["high"]), 2)
        low_px = round(float(last_bar["low"]), 2)
        gap_pct = round(((open_px - float(prev_bar["close"])) / float(prev_bar["close"])) * 100.0, 1)
        ema20 = round(float(last_bar["ema20"]), 2) if "ema20" in last_bar and pd.notna(last_bar["ema20"]) else None
        sma50 = round(float(last_bar["sma50"]), 2) if "sma50" in last_bar and pd.notna(last_bar["sma50"]) else None
        
        # Sector / Theme
        sec = labels.sector(sym) or "Unknown Sector"
        thms = labels.themes(sym) or []
        thm = thms[0] if thms else "General"
        
        # Company name
        comp_name = sym
        try:
            u_df = universe.load_universe()
            if u_df is not None and "symbol" in u_df.columns and "name" in u_df.columns:
                m = u_df[u_df["symbol"] == sym]
                if len(m) > 0 and pd.notna(m["name"].iloc[0]):
                    comp_name = str(m["name"].iloc[0])
        except Exception:
            pass

        return jsonify({
            "symbol": sym,
            "company": comp_name,
            "sector": sec,
            "theme": thm,
            "date": last_bar.name.strftime("%Y-%m-%d"),
            "close": close_px,
            "open": open_px,
            "high": high_px,
            "low": low_px,
            "gap_pct": gap_pct,
            "ema20": ema20,
            "sma50": sma50
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/portfolio")
def portfolio_view_route():
    return Response(HTML_TEMPLATE, mimetype="text/html")

@app.route("/api/portfolio")
def api_portfolio():
    eval_res = pm.evaluate_portfolio()
    return jsonify(eval_res)

@app.route("/api/portfolio/settings", methods=["POST"])
def api_portfolio_settings():
    data = request.get_json(force=True) or {}
    size = data.get("portfolio_size", 100000.0)
    risk_pct = data.get("default_risk_pct", 1.0)
    settings = pm.update_settings(size, risk_pct)
    return jsonify({"status": "ok", "settings": settings})

@app.route("/api/portfolio/add_position", methods=["POST"])
def api_portfolio_add_position():
    data = request.get_json(force=True) or {}
    sym = data.get("symbol", "")
    entry_px = data.get("entry_price")
    stop_px = data.get("stop_price")
    shares = data.get("shares")
    ev_dt = data.get("event_date")
    risk_pct = data.get("risk_pct")
    notes = data.get("notes", "")
    try:
        pos = pm.add_position(sym, entry_px, stop_px, shares, event_date=ev_dt, risk_pct=risk_pct, notes=notes)
        return jsonify({"status": "ok", "position": pos})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/portfolio/update_position", methods=["POST"])
def api_portfolio_update_position():
    data = request.get_json(force=True) or {}
    pos_id = data.get("id")
    action = data.get("action")
    stop_px = data.get("stop_price")
    entry_px = data.get("entry_price")
    add_px = data.get("add_price")
    add_stop = data.get("add_stop")
    add_shares = data.get("add_shares")
    tranche_stops = data.get("tranche_stops")
    notes = data.get("notes")
    try:
        pos = pm.update_position(pos_id, action, stop_price=stop_px, entry_price=entry_px, add_price=add_px, add_shares=add_shares, add_stop=add_stop, tranche_stops=tranche_stops, notes=notes)
        return jsonify({"status": "ok", "position": pos})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/portfolio/close_position", methods=["POST"])
def api_portfolio_close_position():
    data = request.get_json(force=True) or {}
    pos_id = data.get("id")
    exit_px = data.get("exit_price")
    exit_date = data.get("exit_date")
    exit_reason = data.get("exit_reason", "Manual Close")
    notes = data.get("notes", "")
    try:
        closed = pm.close_position(pos_id, exit_px, exit_date=exit_date, exit_reason=exit_reason, notes=notes)
        return jsonify({"status": "ok", "closed_position": closed})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/portfolio/delete_closed_position", methods=["POST"])
def api_portfolio_delete_closed_position():
    data = request.get_json(force=True) or {}
    pos_id = data.get("id")
    if not pos_id:
        return jsonify({"error": "id is required"}), 400
    try:
        success = pm.delete_closed_position(pos_id)
        if success:
            return jsonify({"status": "ok", "deleted_id": pos_id})
        return jsonify({"error": "Closed trade not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/portfolio/clear_closed_positions", methods=["POST"])
def api_portfolio_clear_closed_positions():
    try:
        count = pm.clear_all_closed_positions()
        return jsonify({"status": "ok", "cleared_count": count})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/ai_top_picks")
def api_ai_top_picks():
    events = get_scored_tracker_events(force_rescan=False)
    
    scored_events = []
    for ev in events:
        m50 = ev.get("ml_50") or 0.0
        m100 = ev.get("ml_100") or 0.0
        m150 = ev.get("ml_150") or 0.0
        m200 = ev.get("ml_200") or 0.0
        score = round(((m50 * 1.0) + (m100 * 1.5) + (m150 * 2.0) + (m200 * 2.5)) * 25.0, 1)
        
        alerts = []
        dyn_stop = ev.get("dynamic_stop_loss_pct")
        if dyn_stop is not None:
            alerts.append(f"🛑 ML Trailing Stop: {(dyn_stop * 100):.1f}%")
        if ev.get("action_headline"):
            alerts.append(ev["action_headline"])
            
        caution = None
        is_veto = bool(ev.get("is_thematic_veto") and not ev.get("is_idiosyncratic"))
        held_low = bool(ev.get("held_d1_low", True))
        if is_veto:
            caution = "⚠️ Bottom 35th percentile sector headwind (Trap Avoidance)"
        elif not held_low:
            caution = f"❌ Day 1 Low (${ev.get('stop_price', 0):.2f}) Breached"
            
        scored_events.append({
            "symbol": ev.get("symbol"),
            "date": ev.get("date") or ev.get("event_date"),
            "days_old": ev.get("bars_since", 0),
            "archetype": ev.get("trader_archetype", "Quality Swing"),
            "score": score,
            "prob_50": round(m50 * 100, 1),
            "prob_100": round(m100 * 100, 1),
            "prob_consol": round(m150 * 100, 1),
            "held_d1_low": held_low,
            "is_veto": is_veto,
            "status": ev.get("status", ""),
            "alerts": alerts,
            "caution": caution,
        })
        
    scored_events.sort(key=lambda x: x["score"], reverse=True)
    
    # Fresh: 0 to 2 bars since EP, holding Day 1 Low, no headwind veto
    fresh = [e for e in scored_events if e["days_old"] <= 2 and e["held_d1_low"] and not e["is_veto"]][:10]
    
    # Mature: 3 to 30 bars since EP, holding Day 1 Low, no headwind veto
    mature = [e for e in scored_events if 2 < e["days_old"] <= 30 and e["held_d1_low"] and not e["is_veto"]][:10]
    
    # Traps: Breached Day 1 Low OR Severe Headwind Veto
    traps = [e for e in scored_events if e["is_veto"] or not e["held_d1_low"]][:10]
    
    return jsonify({
        "fresh": fresh,
        "mature": mature,
        "traps": traps
    })



def main():
    parser = argparse.ArgumentParser(description="EP Live Tracker")
    parser.add_argument("--port", type=int, default=8783, help="Port to listen on (default: 8783)")
    args = parser.parse_args()

    print(f"🚀 EP Live Follow-Through Tracker running at: http://127.0.0.1:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False)

if __name__ == "__main__":
    main()
