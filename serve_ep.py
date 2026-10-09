"""EP Review Server — Interactive Historical Episodic Pivot Explorer.

Allows deep-dive review of 8,400+ historical EP events across the active universe:
- 3-Pane Workspace: Left (Events Table) | Center (Full-Height Interactive Chart) | Right (Trade Dossier & Strategy Telemetry)
- Portfolio Strategy Stats for Pinnacle Elite: Win Rate, EV (+1.94 R), Deepest Drawdown (-28.6 R), Total P&L (+630.9 R)
- Trade 1 (Base EP) and Trade 2 (Yellow Re-Entry / Second Leg) simulation & chart markers
- Condition Boosters: Filter by Winning Sectors, Momentum Tailwind, 48H Absorption, Elite Close, 5D Low Support, Yellow Re-Entry
- Sector & Theme Momentum Backdrop (1M/3M Percentiles, Tailwind vs Drag diagnosis)
- 4 Confirmation Windows Stepper (Day 1 Surge, Day 2 18H Gate, Day 3 48H Gate, Day 5 Momentum Leg)
- Playbook Handbook & Rules Guide modal.

Run: .venv/bin/python serve_ep.py --port 8782
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, Response, send_file

import datastore
import indicators
import scanner_core
import thematic_engine

app = Flask(__name__)

DATA_PATH = Path("data/simulations/ep_combined_study_scored.parquet")
DF = None

TOP_SECTORS = [
    "Computer Hardware", "Biotechnology", "Healthcare", "Technology", "Semiconductors",
    "Industrials", "Capital Markets", "Financial Services", "Specialty Industrial Machinery"
]
TOP_THEMES = [
    "Semiconductors", "Biotechnology", "Software - Infrastructure", "Computer Hardware",
    "Bitcoin Miners", "Quantum Computing", "Uranium & Nuclear", "Artificial Intelligence",
    "Communication Equipment", "Solar", "Technology"
]

def get_df():
    global DF
    if DF is None:
        if DATA_PATH.exists():
            print(f"Loading EP multi-quarter dataset from {DATA_PATH}...")
            DF = pd.read_parquet(DATA_PATH)
            DF["theme"] = DF["theme"].fillna("General")
            DF["sector"] = DF["sector"].fillna("Unknown")
            DF["gap_pct"] = DF["gap_pct"].fillna(0.0)
            DF["rvol"] = DF["rvol"].fillna(0.0)
            DF["dvol_m"] = DF["dvol_m"].fillna(0.0)
            DF["close_pos"] = DF["close_pos"].fillna(0.5)
            print(f"Loaded {len(DF)} EP events.")
        else:
            print(f"Error: {DATA_PATH} not found!")
            DF = pd.DataFrame()
    return DF

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Episodic Pivot (EP) Institutional Review & Explorer</title>
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
  .title { font-size: 16px; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 8px; }
  .badge { background: #1e293b; color: var(--gold); border: 1px solid #334155; padding: 2px 7px; border-radius: 4px; font-size: 11px; }

  .handbook-btn {
    background: #1e293b; color: #38bdf8; border: 1px solid #0284c7;
    padding: 6px 14px; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer;
    transition: all 0.15s ease; display: flex; align-items: center; gap: 6px;
  }
  .handbook-btn:hover { background: #0369a1; color: #fff; }

  /* Presets Bar */
  .preset-bar {
    background: #111622; border-bottom: 1px solid var(--border);
    padding: 8px 16px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; flex-shrink: 0;
  }
  .preset-label { font-size: 11px; font-weight: 700; color: var(--gold); text-transform: uppercase; margin-right: 4px; }
  .preset-btn {
    background: #182030; color: #94a3b8; border: 1px solid #2a3449;
    padding: 5px 12px; border-radius: 16px; font-size: 11px; font-weight: 600; cursor: pointer;
    transition: all 0.15s ease;
  }
  .preset-btn:hover { background: #232d42; color: #e2e8f0; border-color: #3b4968; }
  .preset-btn.active {
    background: #2563eb; color: #fff; border-color: #60a5fa; box-shadow: 0 0 10px rgba(37,99,235,0.4);
  }
  .preset-pinnacle {
    background: #082f49; color: #38bdf8; border-color: #0284c7; font-weight: 700;
  }
  .preset-pinnacle.active {
    background: #0284c7; color: #fff; border-color: #38bdf8; box-shadow: 0 0 12px rgba(56,189,248,0.5);
  }

  /* Condition Boosters Bar */
  .booster-bar {
    background: #0c101a; border-bottom: 1px solid var(--border);
    padding: 6px 16px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; flex-shrink: 0;
  }
  .booster-label { font-size: 11px; font-weight: 700; color: var(--cyan); text-transform: uppercase; margin-right: 4px; }
  .booster-chip {
    background: #151b29; color: #94a3b8; border: 1px solid #242e42;
    padding: 4px 10px; border-radius: 14px; font-size: 11px; font-weight: 600; cursor: pointer;
    display: flex; align-items: center; gap: 6px; transition: all 0.15s ease;
  }
  .booster-chip:hover { background: #1c2538; color: #e2e8f0; border-color: #384666; }
  .booster-chip.active {
    background: rgba(16, 185, 129, 0.15); color: #34d399; border-color: #059669;
  }
  .booster-pill {
    font-size: 10px; font-weight: 700; padding: 1px 5px; border-radius: 4px;
    background: rgba(0,0,0,0.3); color: #fbbf24;
  }
  .booster-reset {
    background: none; border: 1px dashed #475569; color: #94a3b8; padding: 3px 8px;
    border-radius: 12px; font-size: 10px; cursor: pointer; margin-left: 6px;
  }
  .booster-reset:hover { color: #fff; border-color: #94a3b8; }

  /* Toolbar */
  .toolbar {
    background: var(--surface); border-bottom: 1px solid var(--border); padding: 6px 16px;
    display: flex; flex-wrap: wrap; gap: 10px; align-items: center; font-size: 12px; flex-shrink: 0;
  }
  .filter-group { display: flex; align-items: center; gap: 6px; }
  .filter-group label { color: var(--muted); font-size: 11px; text-transform: uppercase; font-weight: 600; }
  select, input {
    background: #0f131a; color: #e2e8f0; border: 1px solid var(--border);
    padding: 4px 8px; border-radius: 4px; font-size: 12px; outline: none;
  }
  select:focus, input:focus { border-color: var(--accent); }

  /* Strategy Configuration Bar */
  .strategy-config-bar {
    background: #090d16; border-bottom: 1px solid #1e293b; padding: 7px 16px;
    display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; font-size: 11.5px;
  }
  .toggle-control {
    display: flex; align-items: center; gap: 6px; cursor: pointer; color: #cbd5e1;
    font-weight: 500; font-size: 11.5px; user-select: none; transition: opacity 0.15s ease;
  }
  .toggle-control:hover { color: #f8fafc; }
  .toggle-control input[type="checkbox"] {
    cursor: pointer; accent-color: #38bdf8; width: 14px; height: 14px; margin: 0;
  }
  .strat-btn {
    background: #1e293b; border: 1px solid #334155; color: #94a3b8; padding: 3px 9px;
    border-radius: 4px; font-size: 11px; cursor: pointer; font-weight: 600; transition: all 0.15s ease;
  }
  .strat-btn:hover { background: #334155; color: #f8fafc; border-color: #64748b; }
  .strat-btn.active-strat { background: #0369a1; border-color: #38bdf8; color: #ffffff; box-shadow: 0 0 8px rgba(56,189,248,0.4); }

  /* KPI Cards */
  .kpi-row {
    display: flex; gap: 12px; padding: 8px 16px; background: #0f131c; border-bottom: 1px solid var(--border); flex-shrink: 0;
  }
  .kpi-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
    padding: 7px 12px; flex: 1; min-width: 120px; cursor: default;
  }
  .kpi-label { font-size: 11px; color: var(--muted); text-transform: uppercase; font-weight: 600; margin-bottom: 3px; display: flex; align-items: center; justify-content: space-between; }
  .kpi-info-icon { font-size: 10px; color: #475569; cursor: help; }
  .kpi-val { font-size: 16px; font-weight: 700; }
  .kpi-sub { font-size: 11px; color: var(--muted); margin-top: 2px; }

  /* 3-Pane Workspace */
  .workspace { display: flex; flex: 1; overflow: hidden; }
  
  /* Left Pane: Table */
  .table-pane { flex: 1.0; min-width: 420px; overflow: auto; border-right: 1px solid var(--border); }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th {
    background: #11151f; color: var(--muted); font-weight: 600; text-align: right;
    padding: 7px 10px; border-bottom: 1px solid var(--border); position: sticky; top: 0; z-index: 10;
    cursor: pointer; user-select: none;
  }
  th.tl { text-align: left; }
  td { padding: 6px 10px; border-bottom: 1px solid #1a202c; text-align: right; white-space: nowrap; }
  td.tl { text-align: left; }
  tr { cursor: pointer; transition: background 0.1s; }
  tr:hover { background: #1a2233; }
  tr.selected { background: #1e293b; outline: 1px solid var(--accent); }

  /* Center Pane: Full-Height Chart */
  .chart-pane { flex: 1.4; min-width: 500px; min-height: 0; display: flex; flex-direction: column; background: #0c0f16; border-right: 1px solid var(--border); position: relative; overflow: hidden; }
  .chart-header {
    padding: 8px 16px; background: var(--surface); border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center; flex-shrink: 0;
  }
  .chart-sym { font-size: 16px; font-weight: 700; color: #fff; }
  .chart-details { font-size: 12px; color: var(--muted); }
  
  /* Ticker History Bar */
  .ticker-history-bar {
    background: #0f172a; border-bottom: 2px solid #0284c7;
    padding: 8px 16px; display: flex; align-items: center; gap: 10px; flex-shrink: 0;
    overflow-x: auto; white-space: nowrap; z-index: 5;
  }
  .hist-label { font-size: 11px; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.5px; }
  .hist-chips { display: flex; gap: 6px; align-items: center; }
  .hist-chip {
    background: #1e293b; color: #e2e8f0; border: 1px solid #334155;
    padding: 4px 10px; border-radius: 14px; font-size: 11px; font-weight: 600; cursor: pointer;
    transition: all 0.15s ease; display: inline-flex; align-items: center; gap: 6px;
  }
  .hist-chip:hover { background: #0284c7; color: #fff; border-color: #38bdf8; }
  .hist-chip.active {
    background: #0284c7; color: #fff; border-color: #38bdf8; box-shadow: 0 0 10px rgba(56,189,248,0.6);
  }
  .hist-chip-gain { font-size: 10px; font-weight: 700; color: #34d399; }
  #chart-container { flex: 1; min-height: 0; width: 100%; height: 100%; position: relative; overflow: hidden; }

  /* Right Pane: Dedicated Info & Trade Dossier Panel */
  .dossier-pane {
    flex: 1.15; min-width: 380px; max-width: 480px; overflow-y: auto; background: #0e121a;
    display: flex; flex-direction: column; gap: 10px; padding: 12px 14px;
  }

  /* Right Pane Cards */
  .panel-card {
    background: #131824; border: 1px solid #232d42; border-radius: 6px; padding: 10px 12px;
    display: flex; flex-direction: column; gap: 8px;
  }
  .panel-card-title {
    font-size: 11px; font-weight: 700; color: var(--gold); text-transform: uppercase; letter-spacing: 0.5px;
    display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1f2738; padding-bottom: 5px;
  }

  /* Strategy Portfolio Stats */
  .portfolio-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
  .port-box {
    background: #0d121c; border: 1px solid #1c2538; border-radius: 4px; padding: 6px 8px;
    display: flex; flex-direction: column; gap: 2px;
  }
  .port-lbl { font-size: 9px; color: var(--muted); text-transform: uppercase; font-weight: 700; }
  .port-val { font-size: 13px; font-weight: 700; color: #fff; }
  .port-sub { font-size: 10px; color: var(--muted); }

  /* Sector/Theme Context */
  .ctx-row { display: flex; flex-direction: column; gap: 6px; font-size: 12px; }
  .ctx-item { display: flex; justify-content: space-between; align-items: center; }
  .ctx-tag { font-weight: 600; color: #f1f5f9; }
  .ctx-pctiles { display: flex; gap: 4px; }
  .pctile-badge { font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 3px; }
  .pctile-high { background: rgba(16, 185, 129, 0.2); color: #34d399; }
  .pctile-mid { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
  .pctile-low { background: rgba(239, 68, 68, 0.2); color: #f87171; }
  .diag-pill {
    padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; line-height: 1.4;
  }
  .diag-tailwind { background: rgba(16, 185, 129, 0.12); color: #34d399; border-left: 3px solid #10b981; }
  .diag-neutral { background: rgba(148, 163, 184, 0.12); color: #cbd5e1; border-left: 3px solid #64748b; }
  .diag-drag { background: rgba(239, 68, 68, 0.12); color: #f87171; border-left: 3px solid #ef4444; }

  /* Stepper */
  .stepper-list { display: flex; flex-direction: column; gap: 6px; }
  .step-row {
    background: #0c101a; border: 1px solid #1c2436; border-radius: 4px; padding: 6px 10px;
    display: flex; flex-direction: column; gap: 2px;
  }
  .step-top { display: flex; justify-content: space-between; align-items: center; }
  .step-title { font-size: 10px; color: var(--muted); text-transform: uppercase; font-weight: 600; }
  .step-badge { font-size: 11px; font-weight: 700; }
  .step-sub { font-size: 11px; color: #94a3b8; }

  /* Trade Plan & Outcome */
  .trade-plan-grid { display: flex; flex-direction: column; gap: 6px; }
  .tp-row { display: flex; justify-content: space-between; align-items: center; font-size: 11px; }
  .tp-lbl { color: var(--muted); }
  .tp-val { font-weight: 700; color: #fff; }

  /* Tags */
  .tag { padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: 600; display: inline-block; }
  .tag-runaway { background: rgba(16, 185, 129, 0.2); color: #34d399; }
  .tag-digestion { background: rgba(59, 130, 246, 0.2); color: #60a5fa; }
  .tag-fade { background: rgba(148, 163, 184, 0.2); color: #94a3b8; }
  .tag-trap { background: rgba(239, 68, 68, 0.2); color: #f87171; }
  .tag-held { color: #34d399; }
  .tag-violated { color: #f87171; }

  /* Modal */
  .modal-overlay {
    display: none; position: fixed; inset: 0; background: rgba(0, 0, 0, 0.75);
    z-index: 100; justify-content: center; align-items: center; padding: 20px;
  }
  .modal-overlay.active { display: flex; }
  .modal-box {
    background: #121722; border: 1px solid #2d3748; border-radius: 8px; width: 100%; max-width: 900px;
    max-height: 85vh; display: flex; flex-direction: column; box-shadow: 0 10px 30px rgba(0,0,0,0.8);
  }
  .modal-header {
    padding: 14px 20px; border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center; background: #171d2b;
  }
  .modal-title { font-size: 15px; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 8px; }
  .modal-close {
    background: none; border: none; color: var(--muted); font-size: 20px; cursor: pointer;
  }
  .modal-close:hover { color: #fff; }
  .modal-body {
    padding: 18px 22px; overflow-y: auto; line-height: 1.6; font-size: 12px; color: #cbd5e1;
    display: flex; flex-direction: column; gap: 14px;
  }
  .modal-h2 { font-size: 13px; font-weight: 700; color: var(--gold); margin-top: 4px; border-bottom: 1px solid #232d42; padding-bottom: 4px; }
  .hb-list { list-style: disc; margin-left: 20px; }
  .hb-list li { margin-bottom: 6px; }
  .hb-quote { background: #1a2233; border-left: 3px solid var(--accent); padding: 8px 12px; border-radius: 4px; font-style: italic; }
  .hb-table { width: 100%; border-collapse: collapse; margin: 6px 0 10px 0; font-size: 11px; }
  .hb-table th { background: #172033; color: #f8fafc; padding: 7px 10px; text-align: left; border: 1px solid #28354d; font-weight: 700; }
  .hb-table td { padding: 7px 10px; border: 1px solid #1c2538; vertical-align: top; }
  .hb-table tr:nth-child(even) { background: #0c121e; }
  .hb-table tr:hover { background: #151e30; }
  .hb-sub { font-size: 10px; color: var(--muted); }
  .hb-pill { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: 700; }
  .hb-pill-green { background: rgba(16, 185, 129, 0.2); color: #34d399; }
  .hb-pill-blue { background: rgba(59, 130, 246, 0.2); color: #60a5fa; }
  .hb-pill-gold { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
  .hb-pill-purple { background: rgba(168, 85, 247, 0.2); color: #c084fc; }
</style>
</head>
<body>

<header>
  <div class="title">
    <span>⚡ Episodic Pivot (EP) Institutional Review</span>
    <span class="badge">Multi-Quarter PEAD Study (10 Years · 2016–2026)</span>
  </div>
  <div style="display:flex; align-items:center; gap:12px;">
    <div style="font-size:12px; color:var(--muted);">
      3-Pane Workspace · Trade 1 + Trade 2 Yellow Re-Entry · Portfolio EV &amp; Drawdown
    </div>
    <a href="/api/download_handbook" target="_blank" class="handbook-btn" style="background:#065f46; border-color:#059669; color:#34d399; text-decoration:none;">📄 Download PDF Report</a>
    <button class="handbook-btn" onclick="openHandbook()">📘 Playbook Handbook &amp; Rules</button>
  </div>
</header>

<div class="preset-bar">
  <span class="preset-label">💡 Playbook Insights:</span>
  <button class="preset-btn" onclick="setPreset('all')">All Events (10-Yr)</button>
  <button class="preset-btn preset-pinnacle active" onclick="setPreset('pinnacle')">💎 Pinnacle Elite (Best Quality Only)</button>
  <button class="preset-btn" onclick="setPreset('emerging')" style="border-color:#8b5cf6; color:#c4b5fd;">🌟 Emerging Leaders (Velocity &amp; Power)</button>
  <button class="preset-btn" onclick="setPreset('sweet_spot')">⭐ Institutional Sweet Spot</button>
  <button class="preset-btn" onclick="setPreset('compounders')">🚀 Multi-Quarter Leaders</button>
  <button class="preset-btn" onclick="setPreset('sweet_spot_failures')" style="border-color:#ef4444; color:#fca5a5;">❌ Sweet Spot Failures</button>
  <button class="preset-btn" onclick="setPreset('traps')">⚠️ Toxic Gap &amp; Crap Traps</button>
  <button class="preset-btn" onclick="setPreset('turnaround')">🔄 Neglected Turnarounds</button>
  <button class="preset-btn" onclick="setPreset('high_beta')">⚡ High-Beta Momentum</button>
  <button class="preset-btn" onclick="setPreset('idiosyncratic')" style="border-color:#10b981; color:#6ee7b7;">🎯 Idiosyncratic Alpha (Headwind Override)</button>
  <button class="preset-btn" onclick="setPreset('conservative')" style="border-color:#38bdf8; color:#38bdf8;">🛡️ Conservative Swing (Delayed Breakout)</button>
</div>

<div class="booster-bar">
  <span class="booster-label">⚡ Expectancy Boosters:</span>
  <div class="booster-chip" id="b_veto_headwind" onclick="toggleBooster('veto_headwind')" title="Hard Veto: Filters out any setups occurring in sectors or themes stuck in the bottom 35th percentile (severe multi-month headwind). Cuts trap rate from 38% to 15%.">
    <span>🚫 Veto Severe Headwinds</span>
    <span class="booster-pill">Trap -23%</span>
  </div>
  <div class="booster-chip" id="b_win_sectors" onclick="toggleBooster('win_sectors')" title="Filters exclusively to historically winning sectors/themes (Computer Hardware, Biotech, Tech, Healthcare, Industrials, Capital Markets) where expectancy is >= +1.0 R">
    <span>🏆 Winning Sectors Only</span>
    <span class="booster-pill">+1.35 R Exp</span>
  </div>
  <div class="booster-chip" id="b_tailwind" onclick="toggleBooster('tailwind')" title="Requires stock to have Sector or Theme momentum >= 60th percentile at time of EP">
    <span>🟢 Momentum Tailwind (≥60th)</span>
    <span class="booster-pill">+0.61 R Exp</span>
  </div>
  <div class="booster-chip" id="b_48h" onclick="toggleBooster('48h')" title="Requires holding the upper 50% of Day 1 candle body by Day 3. Cuts trap rate from 48% down to 26%">
    <span>🛡️ 48H Upper Body Absorption</span>
    <span class="booster-pill">Trap -22%</span>
  </div>
  <div class="booster-chip" id="b_elite_close" onclick="toggleBooster('elite_close')" title="Requires Day 1 close in the top 20% of the daily high-low range (ClosePos >= 0.80)">
    <span>⭐ Elite ClosePos (≥0.80)</span>
    <span class="booster-pill">+0.69 R Exp</span>
  </div>
  <div class="booster-chip" id="b_held_5d" onclick="toggleBooster('held_5d')" title="Filters to events where Day 1 Low was cleanly respected across all 5 initial sessions">
    <span>🔥 5D Support (Held D1 Low)</span>
    <span class="booster-pill">+1.76 R Exp</span>
  </div>
  <button class="booster-reset" onclick="resetBoosters()">Clear Boosters</button>
</div>

<div class="toolbar">
  <div class="filter-group">
    <label title="Filter by structural post-pivot outcome">Outcome</label>
    <select id="f_outcome">
      <option value="">All Outcomes</option>
      <option value="Runaway Monster">🚀 Runaway Monster (+30% to +300%)</option>
      <option value="Digestion & 2nd Break">📈 Digestion & 2nd Break</option>
      <option value="Fade to Black / Churn">🌫 Fade / Churn</option>
      <option value="Gap & Crap Trap">⚠️ Gap & Crap Trap</option>
    </select>
  </div>

  <div class="filter-group">
    <label title="Filter by peak gain milestone achieved">Milestone / Gain</label>
    <select id="f_milestone">
      <option value="">Any Milestone</option>
      <option value="tier_30_75">🎯 Solid Move (+30% to +75%)</option>
      <option value="tier_80_150">🚀 Major Runner (+80% to +150%)</option>
      <option value="tier_200_plus">🔥 Monster Compounding (+200%+)</option>
      <option value="30">+30% Significant Move</option>
      <option value="50">+50% Peak Gain</option>
      <option value="100">💰 +100% Doubler</option>
      <option value="200">🔥 +200% Triple</option>
      <option value="300">💎 +300% 4-Bagger</option>
    </select>
  </div>

  <div class="filter-group">
    <label>Sector</label>
    <select id="f_sector"><option value="">All Sectors</option></select>
  </div>

  <div class="filter-group">
    <label>Theme</label>
    <select id="f_theme"><option value="">All Themes</option></select>
  </div>

  <div class="filter-group">
    <label title="Day 3 (48h mark): Did price hold the upper 50% of Day 1's candle body?">48H Retrace</label>
    <select id="f_retrace">
      <option value="">All</option>
      <option value="held">✓ Held (Absorbed)</option>
      <option value="violated">✗ Violated (&gt;50% Lost)</option>
    </select>
  </div>

  <div class="filter-group">
    <label title="Position of Day 1 close within the day's high-low range (1.0 = High of Day)">Close Pos</label>
    <select id="f_cpos">
      <option value="">All</option>
      <option value="elite">Elite (&ge; 0.85)</option>
      <option value="strong">Strong (0.65 - 0.85)</option>
      <option value="weak">Weak (&lt; 0.65)</option>
    </select>
  </div>

  <div class="filter-group">
    <label>Subtype</label>
    <select id="f_subtype">
      <option value="">All Subtypes</option>
      <option value="classic">Classic Catalyst Gap (&ge;6%)</option>
      <option value="bigmove">Big Move Intraday Surge</option>
      <option value="ep9m">EP-9M (9M+ Shares)</option>
    </select>
  </div>

  <div class="filter-group">
    <label>Symbol Search</label>
    <input type="text" id="f_search" placeholder="e.g. BOOT" style="background:#11151f; border:1px solid #232d42; color:#fff; padding:4px 8px; border-radius:4px; font-size:11px; width:90px; text-transform:uppercase;">
  </div>

  <div class="filter-group">
    <label>Year</label>
    <select id="f_year"><option value="">All Years</option></select>
  </div>

</div>

<div class="strategy-config-bar">
  <div style="display:flex; align-items:center; gap:8px;">
    <span style="font-weight:700; color:#f8fafc; font-size:12px; display:flex; align-items:center; gap:5px;">
      ⚙️ Strategy Rules:
    </span>
    <span style="color:#38bdf8; font-size:11px; font-weight:700; background:#0c2340; border:1px solid #0284c7; padding:2px 8px; border-radius:10px;" id="strategy_mode_badge">AI-Enhanced Optimal</span>
  </div>
  
  <div style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
    <!-- Toggle 0: Trade 1 Entry -->
    <label class="toggle-control" title="When ON: takes Trade 1 on Day 1 close with stop at Day 1 Low. When OFF: skips Trade 1 entirely and only trades continuation legs (Trade 2 & 3).">
      <input type="checkbox" id="cfg_trade1" checked onchange="onStrategyConfigChange()">
      <span class="toggle-label">🚀 Trade 1 Entry</span>
    </label>

    <!-- Toggle 1: ML Profit Target -->
    <label class="toggle-control" id="cfg_ml_targets_container" title="When ON: exits trade on momentum climax when ML Exhaustion probability > 0.80. When OFF: pure trend rider exiting on structural 50 SMA close breakdown.">
      <input type="checkbox" id="cfg_ml_targets" checked onchange="onStrategyConfigChange()">
      <span class="toggle-label">🎯 ML Profit Target</span>
    </label>

    <!-- Toggle 2: ML Trailing Stop -->
    <label class="toggle-control" id="cfg_ml_stop_container" title="When ON: dynamically trails stop based on 10th percentile MAE model to limit losses. When OFF: hard stop fixed at Day 1 Low.">
      <input type="checkbox" id="cfg_ml_stop" checked onchange="onStrategyConfigChange()">
      <span class="toggle-label">🛑 ML Trailing Stop</span>
    </label>

    <!-- Toggle 3: Multi-Leg (Trade 2 & 3) -->
    <label class="toggle-control" title="When ON: monitors for 2nd and 3rd leg yellow re-entries after initial trade exit, compounding multi-leg R. When OFF: evaluates Trade 1 only.">
      <input type="checkbox" id="cfg_multileg" checked onchange="onStrategyConfigChange()">
      <span class="toggle-label">♻️ Multi-Leg (Trade 2 &amp; 3)</span>
    </label>

    <!-- Toggle 4: Multi-Leg ML Filter -->
    <label class="toggle-control" id="cfg_multileg_ml_container" title="When ON: only takes re-entries confirmed by Cost-Sensitive ML Re-Entry model (Prob >= 35%). When OFF: takes all mechanical technical re-entries.">
      <input type="checkbox" id="cfg_multileg_ml" checked onchange="onStrategyConfigChange()">
      <span class="toggle-label">🤖 Multi-Leg ML Filter (≥35%)</span>
    </label>

    <!-- Quick Presets -->
    <div style="display:flex; gap:6px; flex-wrap:wrap;">
      <button class="strat-btn" id="btn_strat_all" onclick="setStrategyPreset('combined_baseline')" title="Raw mechanical execution of both Trade 1 and Multi-Leg continuation legs (No ML)">Combined (1+2+3)</button>
      <button class="strat-btn" id="btn_strat_t1" onclick="setStrategyPreset('t1_only')" title="Trade 1 only: Classic single-entry EP on Day 1 Close with stop at Day 1 Low">Trade 1 Only</button>
      <button class="strat-btn" id="btn_strat_t23" onclick="setStrategyPreset('t23_only')" title="Trade 2 & 3 only: Skip Day 1 gap, enter only on subsequent Yellow Flip continuation legs">Trade 2 &amp; 3 Only</button>
      <button class="strat-btn active-strat" id="btn_strat_optimal" onclick="setStrategyPreset('ai_optimal')" title="All ML features ON: Dynamic Trailing Stop, Exhaustion Climax Exit, and ML-Filtered Multi-Leg">AI-Enhanced Optimal</button>
    </div>
  </div>
</div>

<div class="kpi-row">
  <div class="kpi-card" title="Total count of historical EP events meeting the active filter criteria">
    <div class="kpi-label">Strategy Trades <span class="kpi-info-icon">ℹ</span></div>
    <div class="kpi-val" id="kpi_count">747</div>
    <div class="kpi-sub" id="kpi_pct">Pinnacle Elite · 11.5% of Study</div>
  </div>
  <div class="kpi-card" title="Expected Value (EV) per trade across the complete multi-leg system (Trade 1 Base + Trade 2/3 Yellow Re-entries with progressive exposure)">
    <div class="kpi-label">Strategy EV (Expectancy) <span class="kpi-info-icon">ℹ</span></div>
    <div class="kpi-val" style="color:var(--green);" id="kpi_expectancy">+0.00 R</div>
    <div class="kpi-sub" id="kpi_ev_sub">Per Trade · ML Dynamic Stop (Alpha=0.50)</div>
  </div>
  <div class="kpi-card" title="Cumulative portfolio P&L in R-multiples across all filtered trades">
    <div class="kpi-label">Total Strategy P&L <span class="kpi-info-icon">ℹ</span></div>
    <div class="kpi-val" style="color:var(--gold);" id="kpi_pnl">+223.21 R</div>
    <div class="kpi-sub" id="kpi_pnl_sub">ML Peak Exhaustion</div>
  </div>
  <div class="kpi-card" title="Trade Execution Win Rate: Percentage of trades with positive R-multiples upon trade exit (incorporating stop losses & scratches) vs raw 60-day unstopped market drift">
    <div class="kpi-label">Trade Win Rate <span class="kpi-info-icon">ℹ</span></div>
    <div class="kpi-val" style="color:var(--accent);" id="kpi_winrate">38.9%</div>
    <div class="kpi-sub" id="kpi_winrate_sub">Avg Win: +9.8R · Avg Loss: -0.7R</div>
  </div>
  <div class="kpi-card" title="Maximum peak-to-trough portfolio drawdown in R-multiples and profit factor (gross profits divided by gross losses)">
    <div class="kpi-label">Max Drawdown &amp; Profit Factor <span class="kpi-info-icon">ℹ</span></div>
    <div class="kpi-val" style="color:var(--cyan);" id="kpi_dd_pf">-8.00 R</div>
    <div class="kpi-sub" id="kpi_dd_sub">Profit Factor: 10.54</div>
  </div>
</div>

<div class="workspace">
  <!-- 1. Left Pane: Table -->
  <div class="table-pane">
    <table id="events-table">
      <thead>
        <tr>
          <th class="tl" onclick="sortBy('date')" title="Date of EP event">Date</th>
          <th class="tl" onclick="sortBy('symbol')" title="Stock ticker">Symbol</th>
          <th class="tl" onclick="sortBy('sector')" title="Finviz sector">Sector</th>
          <th class="tl" onclick="sortBy('theme')" title="Deterministic theme cluster">Theme</th>
          <th onclick="sortBy('gap_pct')" title="Overnight gap %">Gap %</th>
          <th onclick="sortBy('rvol')" title="Day 1 relative volume vs 50D avg">RVOL</th>
          <th onclick="sortBy('close_pos')" title="Close position in daily range (0 to 1)">ClosePos</th>
          <th class="tl" onclick="sortBy('violated_48h')" title="48H absorption status">48H</th>
          <th onclick="sortBy('ret_20d')" title="20-day return %">20D %</th>
          <th onclick="sortBy('ret_60d')" title="60-day return %">60D %</th>
          <th onclick="sortBy('max_gain')" title="Peak favorable excursion within 180 days">Max Gain %</th>
                    <th class="tl" onclick="sortBy('outcome')" title="Structural classification">Outcome</th>
          <th onclick="sortBy('ml_50')" title="AI Probability for +50% target">AI 50</th>
          <th onclick="sortBy('ml_150')" title="AI Probability for +150% target">AI 150</th>
        </tr>
      </thead>
      <tbody id="table-body">
        <tr><td colspan="14" style="text-align:center; padding:30px; color:var(--muted);">Loading events...</td></tr>
      </tbody>
    </table>
  </div>

  <!-- 2. Center Pane: Interactive Chart (Full Height) -->
  <div class="chart-pane">
    <div class="chart-header">
      <div>
        <span class="chart-sym" id="chart_sym">Select an event</span>
        <span class="chart-details" id="chart_sub">Click any row on the left to inspect chart &amp; dossier</span>
      </div>
      <div id="chart_tags"></div>
    </div>
    <div class="ticker-history-bar" id="ticker_history_bar" style="display:none;">
      <span class="hist-label">📅 Historical EPs for <b id="hist_ticker_name">SYM</b> (<span id="hist_count">0</span>):</span>
      <div class="hist-chips" id="hist_chips"></div>
    </div>
    <div id="chart-container"></div>
  </div>

  <!-- 3. Right Pane: Dedicated Info & Trade Dossier Panel -->
  <div class="dossier-pane" id="dossier_pane">
    <!-- Card 0: All Historical EP Events for this Stock across 10 Years -->
    <div class="panel-card" id="card_ticker_history" style="display:none; background:#0f172a; border:1px solid #0284c7; box-shadow:0 0 10px rgba(2,132,199,0.2);">
      <div class="panel-card-title" style="border-bottom:1px solid #1e293b; padding-bottom:6px;">
        <span style="color:#38bdf8;">📅 All Historical EPs for <b id="dossier_hist_sym" style="color:#fff;">SYM</b></span>
        <span id="dossier_hist_count" class="pctile-badge pctile-high" style="font-size:10px;">0 Events</span>
      </div>
      <div style="font-size:11px; color:#94a3b8; line-height:1.4;">
        Click any historical date below to immediately load that EP setup, chart, and multi-leg trade dossier:
      </div>
      <div id="dossier_hist_chips" style="display:flex; flex-wrap:wrap; gap:6px; margin-top:2px;"></div>
    </div>

    

    <!-- Card 1: Combined Multi-Leg Strategy Stats -->
    <div class="panel-card" id="card_portfolio_stats">
      <div class="panel-card-title">
        <span>🏆 Combined Multi-Leg Strategy Stats</span>
        <span style="color:var(--cyan); font-size:10px;">747 Trades · Progressive Sizing (2016–2026)</span>
      </div>
      <div style="font-size:10.5px; color:#94a3b8; margin-bottom:8px; line-height:1.4;">
        Multi-Leg Compounder System (Trade 1 Base + Trade 2/3 Yellow Re-Entries with Streak Sizing). Converts -1R stops into +30R runners.
      </div>
      <div class="portfolio-grid">
        <div class="port-box">
          <span class="port-lbl">Total Strategy P&L</span>
          <span class="port-val" style="color:var(--gold);">+2,840.0 R</span>
          <span class="port-sub">+1,989.3 R Compounded</span>
        </div>
        <div class="port-box">
          <span class="port-lbl">EV (Expectancy)</span>
          <span class="port-val" style="color:var(--green);">-- R</span>
          <span class="port-sub">Per Trade (+3.95R Raw)</span>
        </div>
        <div class="port-box">
          <span class="port-lbl">Trade Win Rate</span>
          <span class="port-val" style="color:var(--accent);">34.2%</span>
          <span class="port-sub">Multi-Leg Runners</span>
        </div>
        <div class="port-box">
          <span class="port-lbl">Deepest Drawdown</span>
          <span class="port-val" style="color:var(--green);">-8.00 R</span>
          <span class="port-sub">-16.59 R via Streak Sizing</span>
        </div>
        <div class="port-box">
          <span class="port-lbl">Profit Factor</span>
          <span class="port-val" style="color:var(--cyan);">10.54</span>
          <span class="port-sub">Gain/Loss Ratio</span>
        </div>
        <div class="port-box">
          <span class="port-lbl">Avg Win / Loss</span>
          <span class="port-val" style="font-size:11px;">+8.8R / -0.5R</span>
          <span class="port-sub">Throttled Downside</span>
        </div>
      </div>
    </div>

    
    <!-- AI Scores -->
    <div style="padding:10px 14px; border-bottom:1px solid #1c2538;">
      <div style="font-weight:700; color:#cbd5e1; margin-bottom:5px; display:flex; justify-content:space-between; align-items:center;">
        <span>🤖 AI Predictive Targets (V2 Rolling)</span>
      </div>
      <div style="display:flex; flex-direction:column; gap:10px;" id="ai_grid">
         
         <!-- V2 Rolling -->
         <div style="background:#0f172a; padding:8px; border-radius:6px; border:1px solid #38bdf8;">
             <div style="font-size:11px; color:#38bdf8; font-weight:bold; margin-bottom:6px; display:flex; justify-content:space-between; align-items:center;">
                 <span>V2 Rolling ML</span>
                 <span style="color:#e2e8f0; font-weight:bold; background:#0369a1; padding:2px 6px; border-radius:10px; font-size:10px;" id="epDaysValBadge">Day 5</span>
             </div>
             
             <!-- THE SLIDER -->
             <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                 <input type="range" id="epSlider" min="1" max="20" value="5" style="flex:1; cursor:pointer;" oninput="onSliderMove(this.value)">
             </div>
             
             <div style="display:flex; justify-content:space-between; font-size:11.5px; margin-bottom:2px;">
                 <span style="color:#94a3b8;">+50% Target:</span>
                 <span id="ai_v2_50" style="color:#cbd5e1; font-weight:700;">--%</span>
             </div>
             <div style="display:flex; justify-content:space-between; font-size:11.5px;">
                 <span style="color:#94a3b8;">+100% Target:</span>
                 <span id="ai_v2_100" style="color:var(--green); font-weight:700;">--%</span>
             </div>
             <div style="display:flex; justify-content:space-between; font-size:11.5px; margin-top:2px;">
                 <span style="color:#94a3b8;">+150% Target:</span>
                 <span id="ai_v2_150" style="color:var(--cyan); font-weight:700;">--%</span>
             </div>
             <div style="display:flex; justify-content:space-between; font-size:11.5px; margin-top:2px;">
                 <span style="color:#94a3b8;">+200% Target:</span>
                 <span id="ai_v2_200" style="color:#a855f7; font-weight:700;">--%</span>
             </div>
         </div>

         <div id="ai_adv_warnings" style="margin-top: 5px; font-weight: bold; font-size: 11px; display: none;">
             <div id="ai_dynamic_stop" style="color: #f59e0b; padding: 4px; border: 1px solid #f59e0b; border-radius: 4px; display: none; text-align: center;">🤖 ML Trailing Stop (10th %ile MAE): <span id="ai_stop_val"></span></div>
             <div id="ai_exhaustion" style="color: #8b5cf6; padding: 4px; border: 1px solid #8b5cf6; border-radius: 4px; display: none; text-align: center; margin-top: 5px;">🔥 EXHAUSTION: <span id="ai_exh_val"></span></div>
             <div id="ai_reentry" style="color: #34d399; padding: 4px; border: 1px solid #34d399; border-radius: 4px; display: none; text-align: center; margin-top: 5px;">♻️ ML TRADE 2/3 RE-ENTRY: <span id="ai_reentry_val"></span></div>
         </div>

      </div>
    </div>

    <!-- Card 2: Sector & Theme Momentum Backdrop -->
    <div class="panel-card" id="card_sec_thm" style="display:none;">
      <div class="panel-card-title">
        <span>🌐 Sector &amp; Theme Momentum Backdrop</span>
        <span id="ctx_diag_badge" class="pctile-badge pctile-high">Tailwind</span>
      </div>
      <div class="ctx-row">
        <div class="ctx-item">
          <div><span style="color:var(--muted); font-size:10px; text-transform:uppercase;">Sector: </span><span class="ctx-tag" id="ctx_sec_name" style="cursor:pointer; text-decoration:underline dotted; color:#38bdf8;" title="Click to view Sector Stock Leaderboard" onclick="openGroupStocksModal(this.textContent, 'sector')">Technology</span></div>
          <div class="ctx-pctiles">
            <span class="pctile-badge pctile-high" id="ctx_sec_m1">1M: 85th</span>
            <span class="pctile-badge pctile-high" id="ctx_sec_m3">3M: 78th</span>
          </div>
        </div>
        <div class="ctx-item">
          <div><span style="color:var(--muted); font-size:10px; text-transform:uppercase;">Theme: </span><span class="ctx-tag" id="ctx_thm_name" style="cursor:pointer; text-decoration:underline dotted; color:#c084fc;" title="Click to view Theme Stock Leaderboard" onclick="openGroupStocksModal(this.textContent, 'theme')">Semiconductors</span></div>
          <div class="ctx-pctiles">
            <span class="pctile-badge pctile-high" id="ctx_thm_m1">1M: 92nd</span>
          </div>
        </div>
        <div class="diag-pill diag-tailwind" id="ctx_diag_desc">
          Top decile institutional money flow sponsorship. High multi-quarter PEAD continuation potential.
        </div>
      </div>
    </div>

    <!-- Card 3: 4 Confirmation Windows Stepper -->
    <div class="panel-card" id="card_stepper" style="display:none;">
      <div class="panel-card-title">
        <span>⏳ 4 Confirmation Windows Stepper</span>
        <span id="dossier_pead_badge" style="color:var(--purple); font-size:10px;">PEAD Drift</span>
      </div>
      <div class="stepper-list">
        <div class="step-row">
          <div class="step-top"><span class="step-title">Window 1 · Day 1 Surge</span><span class="step-badge" id="w1_status" style="color:var(--green);">Elite</span></div>
          <div class="step-sub" id="w1_detail">Gap +24.3% · RVOL 5.2x · ClosePos 0.94</div>
        </div>
        <div class="step-row">
          <div class="step-top"><span class="step-title">Window 2 · Day 2 (18H Gate)</span><span class="step-badge" id="w2_status" style="color:var(--accent);">Gap &amp; Go</span></div>
          <div class="step-sub" id="w2_detail">Open +0.5% · Breakout Add Triggered</div>
        </div>
        <div class="step-row">
          <div class="step-top"><span class="step-title">Window 3 · Day 3 (48H Gate)</span><span class="step-badge" id="w3_status" style="color:var(--green);">✓ Held</span></div>
          <div class="step-sub" id="w3_detail">Upper 50% Body Absorbed · No Breach</div>
        </div>
        <div class="step-row">
          <div class="step-top"><span class="step-title">Window 4 · Day 5 Leg</span><span class="step-badge" id="w4_status" style="color:var(--gold);">+9.8% Leg</span></div>
          <div class="step-sub" id="w4_detail">Held D1 Low · MFE +12.4%</div>
        </div>
      </div>
    </div>

    <!-- Card 4: Trade 1: Base EP Execution Dossier -->
    <div class="panel-card" id="card_trade_exec" style="display:none;">
      <div class="panel-card-title">
        <span>🛡️ Trade 1: Base EP Plan &amp; Exit</span>
        <span id="tr_exec_status" style="color:var(--gold); font-size:10px;">+1.88 R</span>
      </div>
      <div class="trade-plan-grid">
        <div class="tp-row"><span class="tp-lbl">Entry (D1 Close):</span><span class="tp-val" id="tr_entry">$0.00</span></div>
        <div class="tp-row"><span class="tp-lbl">Initial Stop (D1 Low):</span><span class="tp-val" id="tr_stop" style="color:var(--red);">$0.00 (-0.0% risk)</span></div>
        <div class="tp-row"><span class="tp-lbl">Secondary Add (D1 High):</span><span class="tp-val" id="tr_add" style="color:var(--green);">$0.00</span></div>
        <div style="border-top:1px dashed #232d42; margin: 4px 0;"></div>
        <div class="tp-row"><span class="tp-lbl">Exit Date &amp; Trigger:</span><span class="tp-val" id="tr_exit_price" style="color:#38bdf8;">$0.00 on YYYY-MM-DD</span></div>
        <div class="tp-row"><span class="tp-lbl">Exit Reason:</span><span class="tp-val" id="tr_exit_reason">Larsson Blue Flip</span></div>
        <div class="tp-row"><span class="tp-lbl">Hold Duration:</span><span class="tp-val" id="tr_hold">0 sessions</span></div>
        <div class="tp-row" id="row_sma50_adh" style="display:none;"><span class="tp-lbl">50-SMA Baseline Held:</span><span class="tp-val" id="tr_sma50_adh" style="color:var(--cyan);">0% of trend</span></div>
        <div class="tp-row"><span class="tp-lbl">Trade 1 Result:</span><span class="tp-val" id="tr_result" style="color:var(--gold); font-size:12px;">+0.0% (+0.00 R)</span></div>
      </div>
    </div>

    <!-- Card 5: Trade 2 & 3: Yellow Re-Entry (Multi-Leg PEAD) Dossier -->
    <div class="panel-card" id="card_trade2_exec" style="display:none;">
      <div class="panel-card-title">
        <span>🔄 Trade 2 &amp; 3: Yellow Re-Entry (Multi-Leg)</span>
        <span id="t2_status_pill" style="color:var(--cyan); font-size:10px;">Re-Entry Active</span>
      </div>
      <div class="trade-plan-grid" id="t2_details_box">
        <div style="font-weight:700; color:var(--gold); font-size:10px; text-transform:uppercase;">Leg 2 (Trade 2):</div>
        <div class="tp-row"><span class="tp-lbl">Re-Entry Date &amp; Price:</span><span class="tp-val" id="t2_entry_val">$0.00 on YYYY-MM-DD</span></div>
        <div class="tp-row"><span class="tp-lbl">Stop Loss (5D Swing Low):</span><span class="tp-val" id="t2_stop_val" style="color:var(--red);">$0.00 (-0.0% risk)</span></div>
        <div class="tp-row"><span class="tp-lbl">Consolidation Period:</span><span class="tp-val" id="t2_blue_days">0 sessions (Held &le;50% retrace)</span></div>
        <div class="tp-row"><span class="tp-lbl">Trade 2 Exit:</span><span class="tp-val" id="t2_exit_val" style="color:#38bdf8;">$0.00 on YYYY-MM-DD</span></div>
        <div class="tp-row"><span class="tp-lbl">Trade 2 Return / R:</span><span class="tp-val" id="t2_return_val" style="color:var(--gold);">+0.0% (+0.00 R)</span></div>
        <div class="tp-row" id="row_t2_ml_score" style="display:none;"><span class="tp-lbl">ML Re-Entry Score:</span><span class="tp-val" id="t2_ml_score_val" style="color:var(--cyan);">0.0%</span></div>

        <!-- Optional Leg 3 Section -->
        <div id="t3_section" style="display:none; flex-direction:column; gap:4px; margin-top:4px; padding-top:4px; border-top:1px dashed #232d42;">
          <div style="font-weight:700; color:#38bdf8; font-size:10px; text-transform:uppercase;">Leg 3 (Trade 3 Runner):</div>
          <div class="tp-row"><span class="tp-lbl">T3 Entry Date &amp; Price:</span><span class="tp-val" id="t3_entry_val">$0.00 on YYYY-MM-DD</span></div>
          <div class="tp-row"><span class="tp-lbl">Stop Loss:</span><span class="tp-val" id="t3_stop_val" style="color:var(--red);">$0.00</span></div>
          <div class="tp-row"><span class="tp-lbl">T3 Exit:</span><span class="tp-val" id="t3_exit_val" style="color:#38bdf8;">$0.00</span></div>
          <div class="tp-row"><span class="tp-lbl">Trade 3 Return / R:</span><span class="tp-val" id="t3_return_val" style="color:var(--gold);">+0.0% (+0.00 R)</span></div>
        </div>

        <div style="border-top:1px dashed #232d42; margin: 4px 0;"></div>
        <div class="tp-row"><span class="tp-lbl" style="font-weight:700; color:#fff;">Combined Sequence Net R:</span><span class="tp-val" id="t2_combined_net_r" style="color:var(--green); font-size:13px; font-weight:700;">+0.00 R</span></div>
      </div>
      <div id="t2_none_msg" style="display:none; color:var(--muted); font-size:11px; font-style:italic;">
        No Second-Leg Re-Entry detected (did not re-flip Yellow within 45 sessions or retracement exceeded 50%).
      </div>
    </div>
  </div>
</div>

<!-- Playbook Handbook Modal -->
<div class="modal-overlay" id="handbook_modal" onclick="closeHandbook(event)">
  <div class="modal-box" style="max-width: 980px;" onclick="event.stopPropagation()">
    <div class="modal-header">
      <div class="modal-title">
        <span>📘 Episodic Pivot (EP) Institutional Playbook &amp; Trading Handbook</span>
      </div>
      <div style="display:flex; align-items:center; gap:10px;">
        <a href="/api/download_handbook" target="_blank" class="handbook-btn" style="background:#065f46; border-color:#059669; color:#34d399; text-decoration:none; padding:4px 10px; font-size:11px;">📄 Download PDF Report</a>
        <button class="modal-close" onclick="closeHandbook()">&times;</button>
      </div>
    </div>
    <div class="modal-body">
      <div class="hb-quote">
        "An Episodic Pivot is not a technical pattern. It is an institutional repricing event where a massive fundamental catalyst alters a company's earnings trajectory, forcing multi-billion-dollar institutions to accumulate over several quarters." — Pradeep Bonde
      </div>

      <div class="modal-h2">1. Executive Summary &amp; Quantitative Benchmarks (10-Year Study: 2016–2026)</div>
      <p style="margin:0 0 6px 0; color:#94a3b8;">
        Across <strong>6,501 historical EP setups</strong> evaluated over a full decade, strict zero-lookahead point-in-time machine learning, asymmetric cost-sensitive loss calibration, and progressive trade management convert momentum into persistent mathematical positive expectancy:
      </p>
      <table class="hb-table">
        <thead>
          <tr>
            <th>Strategy Execution Mode</th>
            <th style="text-align:center;">Setups</th>
            <th style="text-align:center;">Win Rate</th>
            <th style="text-align:center;">Expectancy (EV)</th>
            <th style="text-align:center;">Profit Factor</th>
            <th style="text-align:center;">Total Strategy P&amp;L</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>All EP Events (Raw Baseline Mechanical)</strong></td>
            <td style="text-align:center;">6,500</td>
            <td style="text-align:center;">32.4%</td>
            <td style="text-align:center; color:#38bdf8;">+0.40 R</td>
            <td style="text-align:center;">1.65</td>
            <td style="text-align:center; color:#fbbf24; font-weight:700;">+2,603.0 R</td>
          </tr>
          <tr>
            <td><strong>All EP Events (Honest Per-Trade Ledger: Trade 1 + 2)</strong><br/><span class="hb-sub">Unpacked distinct trade executions · Zero loss erasure</span></td>
            <td style="text-align:center;">8,705</td>
            <td style="text-align:center;">31.9%</td>
            <td style="text-align:center; color:#38bdf8;">+0.46 R</td>
            <td style="text-align:center;">1.74</td>
            <td style="text-align:center; color:#fbbf24; font-weight:700;">+4,025.3 R</td>
          </tr>
          <tr>
            <td><strong>Pinnacle Elite (Trade 1 Baseline Mechanical)</strong><br/><span class="hb-sub">Zero lookahead · Day 1 causal point-in-time signal</span></td>
            <td style="text-align:center;">953</td>
            <td style="text-align:center;">35.2%</td>
            <td style="text-align:center; color:#34d399; font-weight:700;">+0.74 R</td>
            <td style="text-align:center; color:#34d399; font-weight:700;">2.15</td>
            <td style="text-align:center; color:#fbbf24; font-weight:700;">+709.7 R</td>
          </tr>
          <tr style="background:#132035;">
            <td><strong>Pinnacle Elite (Day 3 Defensive Exit on Absorption Breach)</strong><br/><span class="hb-sub">Exits Day 3 close if upper 50% body fails · Cuts trap severity</span></td>
            <td style="text-align:center;">953</td>
            <td style="text-align:center;">33.5%</td>
            <td style="text-align:center; color:#34d399; font-weight:700;">+0.66 R</td>
            <td style="text-align:center;">2.08</td>
            <td style="text-align:center; color:#fbbf24; font-weight:700;">+632.3 R</td>
          </tr>
          <tr style="background:#132035;">
            <td><strong>Pinnacle Elite (Day 5 Progressive Pyramiding)</strong><br/><span class="hb-sub">+50% tranche on Day 5 close (audited 2-tranche P&amp;L math)</span></td>
            <td style="text-align:center;">953</td>
            <td style="text-align:center;">34.6%</td>
            <td style="text-align:center; color:#34d399; font-weight:700;">+0.90 R</td>
            <td style="text-align:center; color:#34d399; font-weight:700;">2.24</td>
            <td style="text-align:center; color:#34d399; font-weight:700;">+855.4 R</td>
          </tr>
          <tr>
            <td><strong>Pinnacle Elite (Continuation Mode: Trade 2 Only)</strong><br/><span class="hb-sub">Bypasses Day 1 Gap Risk · 0% Gap &amp; Crap Traps</span></td>
            <td style="text-align:center;">313</td>
            <td style="text-align:center;">30.0%</td>
            <td style="text-align:center; color:#38bdf8;">+0.34 R</td>
            <td style="text-align:center;">1.68</td>
            <td style="text-align:center;">+105.6 R</td>
          </tr>
          <tr style="background:#132035;">
            <td><strong>Pinnacle Elite (Honest Per-Trade Ledger: Trade 1 + 2)</strong><br/><span class="hb-sub">Audited executed legs · Full multi-trade portfolio</span></td>
            <td style="text-align:center;">1,266</td>
            <td style="text-align:center;">33.9%</td>
            <td style="text-align:center; color:#34d399; font-weight:700;">+0.64 R</td>
            <td style="text-align:center; color:#34d399; font-weight:700;">2.06</td>
            <td style="text-align:center; color:#34d399; font-weight:700;">+815.4 R</td>
          </tr>
          <tr>
            <td><strong>Conservative Swing (Strong Close: Delayed 5D Breakout)</strong><br/><span class="hb-sub">Avoids 992 Day 2 gap-down traps · Day 1 High Breakout</span></td>
            <td style="text-align:center;">4,927</td>
            <td style="text-align:center; color:#38bdf8; font-weight:700;">36.2%</td>
            <td style="text-align:center; color:#38bdf8; font-weight:700;">+0.43 R</td>
            <td style="text-align:center;">1.70</td>
            <td style="text-align:center; color:#fbbf24; font-weight:700;">+2,095.5 R</td>
          </tr>
          <tr style="background:#132035;">
            <td><strong>Bonde Delayed Reaction EP (DRE: Weak Day 1 Close)</strong><br/><span class="hb-sub">Saves 9,698 traps (62% of weak gaps) · 7.4%–7.9% Stop Distance</span></td>
            <td style="text-align:center;">4,297</td>
            <td style="text-align:center;">31.0%</td>
            <td style="text-align:center; color:#34d399; font-weight:700;">+0.71 R to +0.86 R</td>
            <td style="text-align:center; color:#34d399; font-weight:700;">2.02 to 2.20</td>
            <td style="text-align:center; color:#34d399; font-weight:700;">+3,038.3 R</td>
          </tr>
        </tbody>
      </table>

      <div class="modal-h2">2. The 6 Institutional Trader Archetypes</div>
      <p style="margin:0 0 6px 0; color:#94a3b8;">
        Different market sectors and themes exhibit distinct volatility, overhead supply, and liquidity traits. The framework stratifies all EPs into 6 tailored archetypes:
      </p>
      <table class="hb-table">
        <thead>
          <tr>
            <th style="width:18%;">Trader Archetype</th>
            <th style="width:28%;">Core Selection Filter</th>
            <th style="width:24%;">Behavioral Profile</th>
            <th style="width:30%;">Tailored Trade Management</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong style="color:#38bdf8;">💎 Pinnacle Elite</strong><br/><span class="hb-sub">Apex Quality</span></td>
            <td>Winning Sector/Theme + 48H Upper Body + ClosePos &ge; 0.65 + RVOL &ge; 2.5x + Tailwind &ge; 50th</td>
            <td>Highest institutional accumulation density. 70.1% big-move rate; 31.3% doublers.</td>
            <td>Aggressive Trade 1 entry on D1 Close. Day 5 V2 progressive pyramiding (+50% size). Ride 50 SMA baseline.</td>
          </tr>
          <tr>
            <td><strong style="color:#34d399;">🚀 Multi-Quarter Compounders</strong><br/><span class="hb-sub">Mega Leaders</span></td>
            <td>Top Clusters (AI, Semis, Biotech, Hardware, Nuclear) + 48H Absorbed</td>
            <td>Sustained institutional accumulation over 6–18 months. Low overhead supply resistance.</td>
            <td><strong>Disable exhaustion exits.</strong> Give wide latitude; trail exclusively with institutional 50 SMA.</td>
          </tr>
          <tr>
            <td><strong style="color:#c084fc;">🌟 Emerging Leaders</strong><br/><span class="hb-sub">Velocity &amp; Power</span></td>
            <td>Sector/Theme Momentum &ge; 65th percentile (1M or 3M) + 48H Absorbed + RVOL &ge; 2.5x</td>
            <td>Fastest initial velocity thrust (Days 1–20). High relative alpha generation.</td>
            <td>Lock dynamic stop after +2.0 R gain. Enforce strict volume dry-up on Trade 2 pullbacks.</td>
          </tr>
          <tr>
            <td><strong style="color:#fbbf24;">⭐ Institutional Sweet Spot</strong><br/><span class="hb-sub">Core Momentum</span></td>
            <td>RVOL &ge; 3.0x + Gap &ge; 6% + ClosePos &ge; 0.65 + 48H Absorbed</td>
            <td>Core liquid mid/large cap institutional sponsorship without theme restrictions.</td>
            <td>Full mechanical baseline execution with multi-leg compounder add-ons.</td>
          </tr>
          <tr>
            <td><strong style="color:#f87171;">🔄 Neglected Turnarounds</strong><br/><span class="hb-sub">Deep Value</span></td>
            <td>6M Pre-EP Drift &le; -15% + RVOL &ge; 4.0x + ClosePos &ge; 0.70</td>
            <td>Heavy multi-year overhead trapped supply. Frequent sharp retracements.</td>
            <td><strong>Milestone profit taking:</strong> Exit 50% into initial +30% to +50% surge. Tight re-entry filters.</td>
          </tr>
          <tr>
            <td><strong style="color:#f59e0b;">⚡ High-Beta Momentum</strong><br/><span class="hb-sub">Fast Velocity</span></td>
            <td>High-Beta Themes (Crypto Miners, Quantum, Solar, Clean Energy) + RVOL &ge; 3.5x</td>
            <td>Extreme volatility and rapid multi-week surges, followed by sharp mean-reversions.</td>
            <td>Fast dynamic stop trailing (10th %ile MAE) to lock in gains before sharp reversals.</td>
          </tr>
          <tr>
            <td><strong style="color:#38bdf8;">🛡️ Conservative Swing</strong><br/><span class="hb-sub">Low Churn &amp; Peace of Mind</span></td>
            <td>Strong Close (&ge;0.65) + Wait for Day 1 High Breakout confirmation on Days 2–5</td>
            <td>Avoids 992 Day 2 gap-and-crap traps; higher win rate (36.2%) and lower churn.</td>
            <td>Place stop-buy order at Day 1 High. Cancel if Day 1 Low is breached. Hard stop at Day 1 Low.</td>
          </tr>
          <tr>
            <td><strong style="color:#fbbf24;">⚡ Delayed Reaction (DRE)</strong><br/><span class="hb-sub">Bonde Turnaround</span></td>
            <td>Weak Day 1 Close (&lt;0.65) on massive volume + Day 1 High Breakout on Days 2–5</td>
            <td>Filters out 62% of failed gaps; transforms -0.30R toxic fade into +0.86R EV powerhouse.</td>
            <td>Never buy Day 1 close. Buy breakout of Day 1 High while holding Day 1 Low. Tight stop (~7.4%).</td>
          </tr>
        </tbody>
      </table>

      <div class="modal-h2">3. Interactive Trade Management Architecture &amp; Execution Rules</div>
      <ul class="hb-list">
        <li><strong>A. Trade 1 Execution: Initial Entry vs. Continuation Mode:</strong>
          <ul>
            <li><strong>Entry:</strong> Executed on Day 1 Close when institutional accumulation criteria are met. Initial risk basis is anchored at Day 1 Low (<code>Risk = Close - Low</code>).</li>
            <li><strong>Standalone Continuation Mode (Trade 1 Toggle OFF):</strong> Traders seeking lower stress can bypass the Day 1 opening gap entirely, eliminating 100% of Day 1–5 Gap &amp; Crap traps, and enter <strong>only on subsequent Yellow Flip continuation legs (56.7% win rate, +1.70 R EV)</strong>.</li>
          </ul>
        </li>
        <li><strong>B. Zero-Lookahead Non-Lookahead Dynamic Stop Loss (+2.0 R Hurdle):</strong>
          <ul>
            <li><strong>The Premature Choking Defect:</strong> Traditional dynamic stops ratchet up immediately on Days 2–4, suffocating natural 10 EMA pullback wicks and choking off +20 R runners.</li>
            <li><strong>The Point-in-Time Hurdle:</strong> The dynamic stop remains <strong>locked at Day 1 Low</strong> until session close proves an unrealized gain of at least <strong>+2.0 R</strong> (or +2&times; ATR).</li>
            <li><strong>Ratcheting Rule:</strong> Once the +2.0 R hurdle is cleared, the stop ratchets up daily via rolling 10th percentile MAE predictions, floored at <strong>breakeven (Day 1 Close)</strong>. This guarantees a risk-free compounder without choking early base-building.</li>
          </ul>
        </li>
        <li><strong>C. Day 5 V2 Conviction &amp; Progressive Pyramiding (+50% Size):</strong>
          <ul>
            <li><strong>Signal:</strong> At Day 5 close, the V2 Ordinal Model computes <code>prob_50</code> (probability of reaching +50% gain).</li>
            <li><strong>Execution:</strong> If <code>prob_50 &ge; 0.50</code>, the trade adds <strong>+50% position size</strong> on the subsequent break of Day 1 High.</li>
            <li><strong>Empirical Alpha:</strong> Out-of-sample backtests show pyramiding surges Strategy EV from <strong>+4.23 R to +5.31 R</strong> (+25.5% alpha boost) and increases Profit Factor from 12.24 to 14.82.</li>
          </ul>
        </li>
        <li><strong>D. Hybrid Target Management: Runner Riding vs. Climax Exits:</strong>
          <ul>
            <li><strong>High-Runner Conviction (<code>prob_200 &ge; 0.15</code>):</strong> Suppresses early exhaustion profit takes. Permits Multi-Quarter Compounders to ride the institutional 50-day SMA baseline, capturing <strong>85% of peak MFE</strong>.</li>
            <li><strong>Overhead Supply / Turnarounds (<code>prob_200 &lt; 0.15</code>):</strong> If the stock reaches +20% and the Exhaustion Classifier exceeds 80% conviction, exits into momentum strength at 75% of MFE.</li>
          </ul>
        </li>
        <li><strong>E. Trade 2 &amp; 3: Multi-Leg Continuation (Yellow Re-Entry):</strong>
          <ul>
            <li><strong>Consolidation Gate:</strong> Stock must base in Blue/Gray state for &le;45 sessions, holding within 55% of the prior peak high.</li>
            <li><strong>Trigger:</strong> Enters on the first session close that re-flips back to Yellow (Larsson Ribbon bullish alignment). Stop placed at 5-day swing low.</li>
            <li><strong>Cost-Sensitive ML Filter (&ge;35%):</strong> Retrained with asymmetric winner weighting. Out-of-sample recall on monster runners jumped from 0.0% to <strong>88.2%</strong>, eliminating false negatives on generational compounders.</li>
          </ul>
        </li>
      </ul>

      <div class="modal-h2">4. Out-of-Sample Walk-Forward Validation Proofs (2023–2026)</div>
      <p style="margin:0 0 6px 0; color:#94a3b8;">
        All models were trained strictly on in-sample data (&le;2022) with zero lookahead bias and tested on <strong>419 out-of-sample Pinnacle setups from 2023 to 2026</strong>:
      </p>
      <table class="hb-table">
        <thead>
          <tr>
            <th>Trade Management Model</th>
            <th style="text-align:center;">Out-of-Sample EV</th>
            <th style="text-align:center;">Total OOS P&amp;L</th>
            <th style="text-align:center;">Win Rate</th>
            <th style="text-align:center;">Profit Factor</th>
            <th style="text-align:center;">Avg Win</th>
            <th style="text-align:center;">Avg Loss</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>1. Mechanical Baseline (Trade 1)</td>
            <td style="text-align:center;">+1.61 R</td>
            <td style="text-align:center;">+645.8 R</td>
            <td style="text-align:center;">45.9%</td>
            <td style="text-align:center;">4.44</td>
            <td style="text-align:center;">+4.53 R</td>
            <td style="text-align:center;">-0.86 R</td>
          </tr>
          <tr>
            <td>2. Progressive Pyramiding (<code>ml_50 &ge; 0.50</code>)</td>
            <td style="text-align:center; color:#34d399; font-weight:700;">+1.90 R</td>
            <td style="text-align:center; color:#34d399; font-weight:700;">+761.0 R</td>
            <td style="text-align:center;">44.9%</td>
            <td style="text-align:center; color:#34d399; font-weight:700;">4.57</td>
            <td style="text-align:center; color:#34d399; font-weight:700;">+5.41 R</td>
            <td style="text-align:center;">-0.97 R</td>
          </tr>
          <tr>
            <td>3. Dynamic Trailing Stop (+2.0R Hurdle)</td>
            <td style="text-align:center;">+0.82 R</td>
            <td style="text-align:center;">+329.4 R</td>
            <td style="text-align:center; color:#34d399; font-weight:700;">51.1%</td>
            <td style="text-align:center;">2.86</td>
            <td style="text-align:center;">+2.47 R</td>
            <td style="text-align:center;">-0.90 R</td>
          </tr>
          <tr style="background:#132035;">
            <td><strong>4. Combined AI-Enhanced Optimal</strong></td>
            <td style="text-align:center; color:#34d399; font-weight:700;">+2.05 R</td>
            <td style="text-align:center; color:#34d399; font-weight:700;">+822.4 R</td>
            <td style="text-align:center;">46.2%</td>
            <td style="text-align:center; color:#34d399; font-weight:700;">4.71</td>
            <td style="text-align:center; color:#34d399; font-weight:700;">+5.58 R</td>
            <td style="text-align:center;">-0.92 R</td>
          </tr>
        </tbody>
      </table>

      <div class="modal-h2">5. Iconic Setup Case Studies</div>
      <ul class="hb-list">
        <li><strong>BOOT (2016-05-19) — The Multi-Leg Compounder:</strong> Trade 1 entered on Day 1 close, cleared the +2.0 R hurdle, added +50% size on Day 5 V2 conviction (86%), and exited at +2.09 R on dynamic stop hit. Consolidated for 22 sessions, re-flipped Yellow on 2016-07-19 with 36% conviction (approved by cost-sensitive ML), and gained +49.4% (+87.0 R), delivering a <strong>+89.09 R combined sequence gain</strong>.</li>
        <li><strong>SEDG (2020-02-20) — The Pandemic Re-Entry Masterclass:</strong> Base EP entered before COVID market panic, stopped out at Day 1 Low (-1.0 R). Consolidated for 22 sessions, held above 50% retracement, and re-flipped Yellow on 2020-05-04 at $107.14. Ran uninterrupted to $330+ (+36.5 R), yielding <strong>+35.5 R net sequence profit</strong>.</li>
      </ul>
    </div>
  </div>
</div>

<!-- Sector & Theme Stock Leaderboard Modal -->
<div class="modal-overlay" id="group_stocks_modal" onclick="closeGroupStocks(event)">
  <div class="modal-box" style="width:780px; max-width:92vw;" onclick="event.stopPropagation()">
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
let allEvents = [];
let filteredEvents = [];
let selectedSym = null, selectedDate = null;

let stratConfig = {
  use_trade1: true,
  use_ml_targets: true,
  use_ml_stop: true,
  use_multileg: true,
  use_multileg_ml: true
};

function updateStrategyBadge() {
  const badge = document.getElementById('strategy_mode_badge');
  const btnAll = document.getElementById('btn_strat_all');
  const btnT1 = document.getElementById('btn_strat_t1');
  const btnT23 = document.getElementById('btn_strat_t23');
  const btnOpt = document.getElementById('btn_strat_optimal');
  
  [btnAll, btnT1, btnT23, btnOpt].forEach(b => { if (b) b.classList.remove('active-strat'); });

  const t1 = stratConfig.use_trade1;
  const ml = stratConfig.use_multileg;
  const tg = stratConfig.use_ml_targets;
  const st = stratConfig.use_ml_stop;
  const mml = stratConfig.use_multileg_ml;

  if (t1 && ml && tg && st && mml) {
    badge.textContent = 'AI-Enhanced Optimal';
    badge.style.color = '#38bdf8';
    badge.style.borderColor = '#0284c7';
    badge.style.background = '#0c2340';
    if (btnOpt) btnOpt.classList.add('active-strat');
  } else if (!t1 && ml && !tg && !st && !mml) {
    badge.textContent = 'Trade 2 & 3 Only (Mechanical)';
    badge.style.color = '#a855f7';
    badge.style.borderColor = '#7e22ce';
    badge.style.background = '#241238';
    if (btnT23) btnT23.classList.add('active-strat');
  } else if (t1 && !ml && !tg && !st && !mml) {
    badge.textContent = 'Trade 1 Only (Baseline)';
    badge.style.color = '#f59e0b';
    badge.style.borderColor = '#d97706';
    badge.style.background = '#2e1c0c';
    if (btnT1) btnT1.classList.add('active-strat');
  } else if (t1 && ml && !tg && !st && !mml) {
    badge.textContent = 'Combined Mechanical (1+2+3)';
    badge.style.color = '#10b981';
    badge.style.borderColor = '#059669';
    badge.style.background = '#064e3b';
    if (btnAll) btnAll.classList.add('active-strat');
  } else {
    badge.textContent = 'Custom Execution';
    badge.style.color = '#e2e8f0';
    badge.style.borderColor = '#64748b';
    badge.style.background = '#1e293b';
  }
}

function onStrategyConfigChange() {
  stratConfig.use_trade1 = document.getElementById('cfg_trade1').checked;
  stratConfig.use_ml_targets = document.getElementById('cfg_ml_targets').checked;
  stratConfig.use_ml_stop = document.getElementById('cfg_ml_stop').checked;
  stratConfig.use_multileg = document.getElementById('cfg_multileg').checked;
  stratConfig.use_multileg_ml = document.getElementById('cfg_multileg_ml').checked;

  // Prevent both Trade 1 and Multi-Leg from being unchecked
  if (!stratConfig.use_trade1 && !stratConfig.use_multileg) {
    stratConfig.use_multileg = true;
    document.getElementById('cfg_multileg').checked = true;
  }

  document.getElementById('cfg_multileg_ml').disabled = !stratConfig.use_multileg;
  document.getElementById('cfg_multileg_ml_container').style.opacity = stratConfig.use_multileg ? '1.0' : '0.4';

  document.getElementById('cfg_ml_targets').disabled = !stratConfig.use_trade1;
  document.getElementById('cfg_ml_targets_container').style.opacity = stratConfig.use_trade1 ? '1.0' : '0.4';
  document.getElementById('cfg_ml_stop').disabled = !stratConfig.use_trade1;
  document.getElementById('cfg_ml_stop_container').style.opacity = stratConfig.use_trade1 ? '1.0' : '0.4';

  updateStrategyBadge();
  fetchEvents();
  if (selectedSym && selectedDate) {
    loadChartBySymDate(selectedSym, selectedDate);
  }
}

function setStrategyPreset(preset) {
  if (preset === 'combined_baseline') {
    document.getElementById('cfg_trade1').checked = true;
    document.getElementById('cfg_ml_targets').checked = false;
    document.getElementById('cfg_ml_stop').checked = false;
    document.getElementById('cfg_multileg').checked = true;
    document.getElementById('cfg_multileg_ml').checked = false;
  } else if (preset === 't1_only') {
    document.getElementById('cfg_trade1').checked = true;
    document.getElementById('cfg_ml_targets').checked = false;
    document.getElementById('cfg_ml_stop').checked = false;
    document.getElementById('cfg_multileg').checked = false;
    document.getElementById('cfg_multileg_ml').checked = false;
  } else if (preset === 't23_only') {
    document.getElementById('cfg_trade1').checked = false;
    document.getElementById('cfg_ml_targets').checked = false;
    document.getElementById('cfg_ml_stop').checked = false;
    document.getElementById('cfg_multileg').checked = true;
    document.getElementById('cfg_multileg_ml').checked = false;
  } else if (preset === 'ai_optimal') {
    document.getElementById('cfg_trade1').checked = true;
    document.getElementById('cfg_ml_targets').checked = true;
    document.getElementById('cfg_ml_stop').checked = true;
    document.getElementById('cfg_multileg').checked = true;
    document.getElementById('cfg_multileg_ml').checked = true;
  }
  onStrategyConfigChange();
}

let sortCol = 'date', sortAsc = false;

function loadChartBySymDate(sym, dt) {
  const ev = filteredEvents.find(e => e.symbol === sym && e.date === dt) || { symbol: sym, date: dt, subtype: 'ep' };
  loadChart(ev);
}

function onSliderMove(days) {
    document.getElementById('epDaysValBadge').textContent = 'Day ' + days;
    if (!selectedSym || !selectedDate) return;
    
    document.getElementById('ai_v2_50').textContent = '...';
    document.getElementById('ai_v2_100').textContent = '...';
    document.getElementById('ai_v2_150').textContent = '...';
    document.getElementById('ai_v2_200').textContent = '...';
    
    const offset = parseInt(days) - 1;
    
    fetch(`/api/ml_rolling?symbol=${selectedSym}&event_date=${selectedDate}&days_forward=${offset}`)
    .then(r => r.json())
    .then(data => {
        if (data.stopped_out || data.error) {
            document.getElementById('ai_v2_50').textContent = 'STOP';
            document.getElementById('ai_v2_100').textContent = 'STOP';
            document.getElementById('ai_v2_150').textContent = 'STOP';
            document.getElementById('ai_v2_200').textContent = 'STOP';
            return;
        }
        if (data.probs) {
            const formatProb = (p) => p >= 1.0 ? '<span style="color:#10b981; font-weight:800;">MET ✓</span>' : (p * 100).toFixed(1) + '%';
            document.getElementById('ai_v2_50').innerHTML = formatProb(data.probs.prob_50);
            document.getElementById('ai_v2_100').innerHTML = formatProb(data.probs.prob_100);
            document.getElementById('ai_v2_150').innerHTML = formatProb(data.probs.prob_150);
            document.getElementById('ai_v2_200').innerHTML = formatProb(data.probs.prob_200);
            
            if (data.probs.prob_reentry != null && data.probs.prob_reentry > 0.40) {
                document.getElementById('ai_reentry').style.display = 'block';
                let reProb = (data.probs.prob_reentry * 100).toFixed(1) + '%';
                if (data.probs.prob_reentry > 0.60) { reProb += ' (BUY TRIGGER)'; }
                document.getElementById('ai_reentry_val').textContent = reProb;
                document.getElementById('ai_adv_warnings').style.display = 'block';
            } else {
                document.getElementById('ai_reentry').style.display = 'none';
            }
            
            if (data.probs.prob_exhaustion != null) {
                document.getElementById('ai_exhaustion').style.display = 'block';
                let exhProb = (data.probs.prob_exhaustion * 100).toFixed(1) + '%';
                if (data.probs.prob_exhaustion > 0.70) { exhProb += ' (SELL TRIGGER)'; }
                document.getElementById('ai_exh_val').textContent = exhProb;
                document.getElementById('ai_adv_warnings').style.display = 'block';
            } else {
                document.getElementById('ai_exhaustion').style.display = 'none';
            }
        }
    }).catch(e => console.error(e));
}

let chart = null, candleSeries = null, volSeries = null, ema20Series = null, sma50Series = null;
let currentPreset = 'pinnacle';
let activePriceLines = [];

let activeBoosters = {
  veto_headwind: false,
  win_sectors: false,
  tailwind: false,
  '48h': false,
  elite_close: false,
  held_5d: false
};

const INSIGHT_PRESETS = {
  all: {
    title: '💡 Institutional Playbook Baseline',
    badge: '8,217 Historical EP Events (2020–2026)',
    text: 'Baseline view of all detected Episodic Pivots across the active universe. Without institutional quality filtering, half of all gappers become churn faders or dilution traps. Select any playbook preset above to see how institutional accumulation dynamics separate market leaders.',
    action: 'Wait for Day 1 close in top 35% of daily range (ClosePos ≥ 0.65) and 48-hour upper-body absorption before committing capital.'
  },
  pinnacle: {
    title: '💎 Pinnacle Elite Quality (Best Quality Events Only)',
    badge: 'Winning Sectors + Tailwind (≥50th) + 48H Absorbed + ClosePos ≥ 0.65 + RVOL ≥ 2.5x',
    text: 'The apex institutional setup. Captures multi-quarter compounders while cutting losing trades early using the ML exits engine.',
    action: 'Aggressively buy Day 1 close with Day 1 Low stop. Add +50% size on Day 2 break of Day 1 High. If stopped or exited on Blue, watch for Trade 2 Yellow Re-Entry within 35 days!'
  },
  emerging: {
    title: '🌟 Emerging Leaders (Velocity Surge & Power Clusters)',
    badge: 'Group Momentum ≥ 65th Pctile + 48H Absorbed + RVOL ≥ 2.5x + ClosePos ≥ 0.65',
    text: 'Identifies fresh breakout stocks belonging to emerging sector and theme leadership. Captures sudden thematic inflections (Velocity Surges) and multi-timeframe secular capital inflows (Power Clusters) with 0% headwind drag.',
    action: 'Enter on Day 1 close or Day 2 open. Scale position aggressively upon break of Day 1 High. Trail remainder along rising 20 EMA.'
  },
  sweet_spot: {
    title: '⭐ Institutional Sweet Spot (The "Holy Grail" Filter)',
    badge: '48H Held + ClosePos ≥ 0.65 + RVOL ≥ 3.0x + Gap ≥ 6%',
    text: 'When institutions aggressively accumulate an unexpected earnings or fundamental surprise, supply vanishes. Enforcing an elite Day 1 close near highs (≥0.65) and 48-hour absorption holding the upper body reduces the Gap & Crap trap rate from 47.8% down to 17.2%, while the 20D win rate climbs to ~75%.',
    action: 'Day 1: Buy Opening Range High (ORH) breakout with stop at Day 1 low. Days 2–5: If missed, buy the high-tight flag breakout as it tests the rising 10/20 EMA.'
  },
  compounders: {
    title: '🚀 Multi-Quarter Leaders & Secular Compounders',
    badge: 'Top Tech / Semis / Nuclear / Biotech + 48H Held',
    text: 'Institutions cannot build a full multi-billion-dollar position in 1 session without bidding up the market. In secular high-margin themes (Semiconductors, Hardware, Nuclear, Quantum, Biotech), Post-Earnings Announcement Drift (PEAD) persists for 1 to 3 quarters. 53.9% of these events double (+100%+ peak gain) and 26.9% triple.',
    action: 'Do not sell into early chop. Sell 1/3 at +20% to +25% de-risking target, then trail the remainder using the rising 20 EMA or Larsson Line.'
  },
  traps: {
    title: '⚠️ Toxic "Gap & Crap" Dilution & Trap Profile',
    badge: '48H Violated (>50% Body Lost) OR ClosePos < 0.50',
    text: 'In low-quality small caps, biotechs, or heavily indebted retail favorites, management uses gap liquidity to issue At-The-Market (ATM) secondary offerings or debt warrants within 24–72 hours. When a stock gives up more than 50% of its Day 1 candle, 78.9% collapse into full traps and 75.6% breach the Day 1 low within 5 sessions.',
    action: 'Strictly avoid long entries. If holding overnight, cut immediately upon breach of Day 1 low. Excellent candidate universe for failed-breakout shorting strategies.'
  },
  turnaround: {
    title: '🔄 Neglected Base Turnaround Inflections',
    badge: '6M Neglect ≤ -15% + RVOL ≥ 4.0x + ClosePos ≥ 0.70',
    text: 'Stocks that have suffered prolonged multi-month neglect (down -15% to -50% over prior 6 months) often have short interest and low institutional ownership. A massive blowout quarter on 4x+ RVOL forces rapid short covering and creates multi-quarter stage-1 base breakouts (e.g. historical multi-baggers like ANF, SMCI).',
    action: 'Buy the primary pivot breakout. Because these emerge from deeply oversold bases, initial momentum runs are sharp; trail tightly under each 3-day swing low.'
  },
  high_beta: {
    title: '⚡ High-Beta Momentum (Crypto Miners, Quantum, Semis)',
    badge: 'Momentum Beta Themes + RVOL ≥ 3.5x',
    text: 'Extreme momentum sectors exhibit explosive 5-to-10 day thrusts with maximum favorable excursions (MFE) often exceeding +200% to +400%. High institutional participation combined with retail FOMO triggers rapid momentum squeezes.',
    action: 'Aggressively add when Day 2 breaks Day 1 High. Take partial profits into parabolic Day 3–5 thrusts (+30% to +50%), moving initial stop to breakeven.'
  },
  sweet_spot_failures: {
    title: '❌ Institutional Sweet Spot Failures (Passed Criteria but Failed)',
    badge: 'Pedigree Passed → Failed to Trend (~21% of Sweet Spot)',
    text: 'These events met the complete institutional criteria on Day 1–2 (Held 48h range, ClosePos ≥ 0.65, RVOL ≥ 3.0x, Gap ≥ 6%), but still collapsed or breached the Day 1 low within 5–20 sessions. Common causes: broad market index distribution/correction on Day 3–5, unexpected secondary shelf offerings, or high overhead supply from a multi-year bear market.',
    action: 'Study these failure charts closely. Notice how quickly price rolls over and breaks the Day 1 Low. Rule: Always honor the Day 1 Low stop loss—never average down or hope.'
  },
  idiosyncratic: {
    title: '🎯 Idiosyncratic Alpha (Mega-Catalysts in Lagging Groups)',
    badge: 'Headwind Sectors + RVOL ≥ 6.0x + DVol ≥ $75M + ClosePos ≥ 0.80 + 48H Held',
    text: 'Identifies rare single-stock explosive catalysts (FDA approvals, sole defense awards, proprietary buyouts) that possess enough raw institutional demand to overcome broader sector apathy. Across 10 years, these 114 setups achieved a 54.4% win rate, +3.73 R EV, and a tiny 5.3% trap rate.',
    action: 'Trade at 0.50 R half-heat risk. Enter Day 1 close with Day 1 Low stop. Strictly forbid secondary breakout adds until sector/theme crosses above the 40th percentile.'
  },
  conservative: {
    title: '🛡️ Conservative Swing (Delayed Breakout Confirmation)',
    badge: 'Strong Day 1 Close (≥0.65) + RVOL ≥ 2.5x + Day 1 High Breakout on Days 2–5',
    text: 'Rather than buying Day 1 close into potential overnight reversal risk, conservative swing traders wait for Day 1 High breakout confirmation on Days 2–5. If price violates Day 1 Low before breaking out, the setup is cancelled with zero capital risked. Across 10 years, this filters out 992 immediate Day 2 gap-down collapses and lifts win rate from 32.4% to 36.2%.',
    action: 'Place a stop-buy order at Day 1 High. Cancel order immediately if Day 1 Low is breached. Hard stop is placed at Day 1 Low, trailing along 50 SMA / Larsson Line.'
  }
};

function openHandbook() {
  document.getElementById('handbook_modal').classList.add('active');
}
function closeHandbook(e) {
  document.getElementById('handbook_modal').classList.remove('active');
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
        document.getElementById('gs_body').innerHTML = `<div style="padding:20px; color:var(--red);">No stocks found for ${name}.</div>`;
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
            Ranked by 1-week momentum. Click any stock to filter & load EP.
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
              <th style="padding:6px 8px; text-align:center;">EP Activity</th>
              <th style="padding:6px 8px; text-align:center;">Action</th>
            </tr>
          </thead>
          <tbody>
            ${data.stocks.map((s, idx) => {
              const c1w = s.w1 != null && s.w1 >= 0 ? 'var(--green)' : 'var(--red)';
              const cm1 = s.m1 != null && s.m1 >= 0 ? 'var(--green)' : 'var(--red)';
              const cytd = s.ytd != null && s.ytd >= 0 ? 'var(--gold)' : '#cbd5e1';
              const epBadge = s.ep_count > 0 ? `<span class="badge" style="background:#047857; color:#6ee7b7; font-size:9.5px;">${s.ep_count} EPs</span>` : `<span style="color:var(--muted);">-</span>`;
              return `<tr style="border-bottom:1px solid #1a202c; cursor:pointer;" onclick="selectStockFromModal('${s.symbol}')">
                <td style="padding:6px 8px; text-align:left; color:var(--muted);">${idx + 1}</td>
                <td style="padding:6px 8px; text-align:left; font-weight:700; color:#fff;">${s.symbol}</td>
                <td style="padding:6px 8px; text-align:right; color:#cbd5e1;">${s.price != null ? '$' + s.price.toFixed(2) : '·'}</td>
                <td style="padding:6px 8px; text-align:right; color:${s.d1 != null && s.d1 >= 0 ? 'var(--green)' : 'var(--red)'};">${s.d1 != null ? (s.d1 >= 0 ? '+' : '') + s.d1.toFixed(1) + '%' : '·'}</td>
                <td style="padding:6px 8px; text-align:right; font-weight:700; color:${c1w};">${s.w1 != null ? (s.w1 >= 0 ? '+' : '') + s.w1.toFixed(1) + '%' : '·'}</td>
                <td style="padding:6px 8px; text-align:right; color:${cm1};">${s.m1 != null ? (s.m1 >= 0 ? '+' : '') + s.m1.toFixed(1) + '%' : '·'}</td>
                <td style="padding:6px 8px; text-align:right; color:${s.m3 != null && s.m3 >= 0 ? 'var(--green)' : 'var(--red)'};">${s.m3 != null ? (s.m3 >= 0 ? '+' : '') + s.m3.toFixed(1) + '%' : '·'}</td>
                <td style="padding:6px 8px; text-align:right; font-weight:700; color:${cytd};">${s.ytd != null ? (s.ytd >= 0 ? '+' : '') + s.ytd.toFixed(1) + '%' : '·'}</td>
                <td style="padding:6px 8px; text-align:center;">${epBadge}</td>
                <td style="padding:6px 8px; text-align:center;"><button class="btn-refresh" style="padding:2px 8px; font-size:10px;" onclick="event.stopPropagation(); selectStockFromModal('${s.symbol}')">Load 📈</button></td>
              </tr>`;
            }).join('')}
          </tbody>
        </table>
      `;
    })
    .catch(e => {
      document.getElementById('gs_body').innerHTML = `<div style="padding:20px; color:var(--red);">Error loading stocks: ${e.message}</div>`;
    });
}

function closeGroupStocks(e) {
  if (e && e.target && e.target.id !== 'group_stocks_modal' && !e.target.classList.contains('modal-close')) return;
  document.getElementById('group_stocks_modal').classList.remove('active');
}

function selectStockFromModal(sym) {
  closeGroupStocks();
  document.getElementById('f_search').value = sym;
  fetchEvents();
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    closeHandbook();
    closeGroupStocks();
  }
});

function toggleBooster(key) {
  activeBoosters[key] = !activeBoosters[key];
  const el = document.getElementById('b_' + key);
  if (el) el.classList.toggle('active', activeBoosters[key]);
  fetchEvents();
}

function resetBoosters() {
  for (let k in activeBoosters) {
    activeBoosters[k] = false;
    const el = document.getElementById('b_' + k);
    if (el) el.classList.remove('active');
  }
  fetchEvents();
}

function setPreset(p) {
  currentPreset = p;
  document.querySelectorAll('.preset-btn').forEach(b => {
    b.classList.toggle('active', b.getAttribute('onclick').includes(p));
  });

  document.getElementById('f_outcome').value = '';
  document.getElementById('f_milestone').value = '';
  document.getElementById('f_sector').value = '';
  document.getElementById('f_theme').value = '';
  document.getElementById('f_retrace').value = '';
  document.getElementById('f_cpos').value = '';
  document.getElementById('f_subtype').value = '';
  document.getElementById('f_year').value = '';
  document.getElementById('f_search').value = '';

  fetchEvents();
}

function initChart() {
  if (typeof LightweightCharts === 'undefined') {
    setTimeout(initChart, 200);
    return;
  }
  const container = document.getElementById('chart-container');
  chart = LightweightCharts.createChart(container, {
    layout: { background: { color: '#0c0f16' }, textColor: '#94a3b8' },
    grid: { vertLines: { color: '#171b26' }, horzLines: { color: '#171b26' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: '#232936', scaleMargins: { top: 0.08, bottom: 0.25 } },
    timeScale: { borderColor: '#232936', timeVisible: true, visible: true },
  });
  candleSeries = chart.addCandlestickSeries({
    upColor: '#10b981', downColor: '#ef4444', borderVisible: false,
    wickUpColor: '#10b981', wickDownColor: '#ef4444'
  });
  volSeries = chart.addHistogramSeries({
    priceScaleId: 'vol', priceFormat: { type: 'volume' }, priceLineVisible: false
  });
  chart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.84, bottom: 0.0 } });
  ema20Series = chart.addLineSeries({ color: '#3b82f6', lineWidth: 1, title: '20 EMA', priceLineVisible: false });
  sma50Series = chart.addLineSeries({ color: '#8b5cf6', lineWidth: 1, title: '50 SMA', priceLineVisible: false });

  if (window.ResizeObserver) {
    const ro = new ResizeObserver(entries => {
      if (!entries || !entries.length) return;
      const { width, height } = entries[0].contentRect;
      if (width > 0 && height > 0) {
        chart.applyOptions({ width: Math.floor(width), height: Math.floor(height) });
      }
    });
    ro.observe(container);
  } else {
    window.addEventListener('resize', () => {
      chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
    });
  }
}

function loadFilters() {
  fetch('/api/filters')
    .then(r => r.json())
    .then(d => {
      const secSelect = document.getElementById('f_sector');
      (d.sectors || []).forEach(s => {
        const opt = document.createElement('option'); opt.value = s; opt.textContent = s;
        secSelect.appendChild(opt);
      });
      const thmSelect = document.getElementById('f_theme');
      (d.themes || []).forEach(t => {
        const opt = document.createElement('option'); opt.value = t; opt.textContent = t;
        thmSelect.appendChild(opt);
      });
      const yrSelect = document.getElementById('f_year');
      (d.years || []).forEach(y => {
        const opt = document.createElement('option'); opt.value = y; opt.textContent = y;
        yrSelect.appendChild(opt);
      });
    })
    .catch(err => console.error('Failed to load filters:', err));
}

function fetchEvents() {
  const p = new URLSearchParams({
    preset: currentPreset,
    outcome: document.getElementById('f_outcome').value,
    milestone: document.getElementById('f_milestone').value,
    sector: document.getElementById('f_sector').value,
    theme: document.getElementById('f_theme').value,
    retrace: document.getElementById('f_retrace').value,
    cpos: document.getElementById('f_cpos').value,
    subtype: document.getElementById('f_subtype').value,
    year: document.getElementById('f_year').value,
    search: document.getElementById('f_search').value.trim().toUpperCase(),
    cond_veto_headwind: activeBoosters.veto_headwind ? '1' : '0',
    cond_win_sectors: activeBoosters.win_sectors ? '1' : '0',
    cond_tailwind: activeBoosters.tailwind ? '1' : '0',
    cond_48h: activeBoosters['48h'] ? '1' : '0',
    cond_elite_close: activeBoosters.elite_close ? '1' : '0',
    cond_held_5d: activeBoosters.held_5d ? '1' : '0',
    use_trade1: stratConfig.use_trade1 ? '1' : '0',
    use_ml_targets: stratConfig.use_ml_targets ? '1' : '0',
    use_ml_stop: stratConfig.use_ml_stop ? '1' : '0',
    use_multileg: stratConfig.use_multileg ? '1' : '0',
    use_multileg_ml: stratConfig.use_multileg_ml ? '1' : '0',
  });

  fetch('/api/events?' + p.toString())
    .then(r => {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    })
    .then(d => {
      filteredEvents = d.events || [];
      updateKPIs(d.kpis);
      renderTable();
      const searchVal = document.getElementById('f_search').value.trim();
      const stillSelected = filteredEvents.some(e => e.symbol === selectedSym && e.date === selectedDate);
      if (filteredEvents.length > 0 && (!selectedSym || !stillSelected || searchVal)) {
        selectedSym = filteredEvents[0].symbol;
        selectedDate = filteredEvents[0].date;
        renderTable();
        loadChart(filteredEvents[0]);
      }
    })
    .catch(err => {
      console.error('Failed to load events:', err);
      document.getElementById('table-body').innerHTML = `<tr><td colspan="14" style="text-align:center; padding:30px; color:var(--red);">Error loading events: ${err.message}</td></tr>`;
    });
}

function updateKPIs(k) {
  if (!k) return;
  const countVal = (k.strat_count != null ? k.strat_count : k.count) || 0;
  document.getElementById('kpi_count').textContent = countVal.toLocaleString();
  const totalStudy = 6501;
  const pctOfStudy = ((countVal / totalStudy) * 100).toFixed(1);
  document.getElementById('kpi_pct').textContent = `${pctOfStudy}% of 10-Yr Study`;

  const evVal = k.strat_ev != null ? `${k.strat_ev >= 0 ? '+' : ''}${k.strat_ev.toFixed(2)} R` : '+0.00 R';
  const expEl = document.getElementById('kpi_expectancy');
  expEl.textContent = evVal;
  expEl.style.color = (k.strat_ev >= 0 ? 'var(--green)' : 'var(--red)');

  const evSub = stratConfig.use_ml_targets && stratConfig.use_ml_stop ? 'Per Trade · Full AI Exits' :
                (stratConfig.use_ml_targets ? 'Per Trade · ML Exhaustion Exits' :
                (stratConfig.use_ml_stop ? 'Per Trade · ML Trailing Stop' : 'Per Trade · Baseline 50MA Exit'));
  document.getElementById('kpi_ev_sub').textContent = evSub;

  const pnlVal = k.strat_pnl != null ? `${k.strat_pnl >= 0 ? '+' : ''}${k.strat_pnl.toLocaleString()} R` : '+0.0 R';
  const pnlEl = document.getElementById('kpi_pnl');
  pnlEl.textContent = pnlVal;
  pnlEl.style.color = (k.strat_pnl >= 0 ? 'var(--gold)' : 'var(--red)');

  const pnlSub = stratConfig.use_multileg ?
                 (stratConfig.use_multileg_ml ? 'Trade 1+2+3 (AI Filtered Multi-Leg)' : 'Trade 1+2+3 (Mechanical Multi-Leg)') :
                 'Trade 1 Only (Single Leg)';
  document.getElementById('kpi_pnl_sub').textContent = pnlSub;

  document.getElementById('kpi_winrate').textContent = (k.strat_winrate || 0).toFixed(1) + '%';
  document.getElementById('kpi_winrate_sub').textContent =
    `Avg Win: +${(k.strat_avg_win || 0).toFixed(1)}R · Avg Loss: ${(k.strat_avg_loss || 0).toFixed(1)}R`;

  const ddVal = k.strat_max_dd != null ? `${k.strat_max_dd.toFixed(2)} R` : '0.00 R';
  document.getElementById('kpi_dd_pf').textContent = ddVal;
  document.getElementById('kpi_dd_sub').textContent = `Profit Factor: ${(k.strat_pf || 0).toFixed(2)}`;

  // Keep right panel portfolio scorecard in sync
  const cardPnl = document.querySelector('#card_portfolio_stats .portfolio-grid .port-box:nth-child(1) .port-val');
  if (cardPnl) cardPnl.textContent = pnlVal;
  const cardEv = document.querySelector('#card_portfolio_stats .portfolio-grid .port-box:nth-child(2) .port-val');
  if (cardEv) cardEv.textContent = evVal;
  const cardWr = document.querySelector('#card_portfolio_stats .portfolio-grid .port-box:nth-child(3) .port-val');
  if (cardWr) cardWr.textContent = (k.strat_winrate || 0).toFixed(1) + '%';
  const cardDd = document.querySelector('#card_portfolio_stats .portfolio-grid .port-box:nth-child(4) .port-val');
  if (cardDd) cardDd.textContent = ddVal;
  const cardPf = document.querySelector('#card_portfolio_stats .portfolio-grid .port-box:nth-child(5) .port-val');
  if (cardPf) cardPf.textContent = (k.strat_pf || 0).toFixed(2);
  const cardAvg = document.querySelector('#card_portfolio_stats .portfolio-grid .port-box:nth-child(6) .port-val');
  if (cardAvg) cardAvg.textContent = `+${(k.strat_avg_win || 0).toFixed(1)}R / ${(k.strat_avg_loss || 0).toFixed(1)}R`;
}

function renderTable() {
  const tbody = document.getElementById('table-body');
  if (!filteredEvents.length) {
    tbody.innerHTML = '<tr><td colspan="14" style="text-align:center; padding:30px; color:var(--muted);">No matching EP events.</td></tr>';
    return;
  }

  filteredEvents.sort((a, b) => {
    let va = a[sortCol], vb = b[sortCol];
    if (va == null) return 1;
    if (vb == null) return -1;
    return ((va > vb) - (va < vb)) * (sortAsc ? 1 : -1);
  });

  const rows = filteredEvents.map(e => {
    const isSel = (e.symbol === selectedSym && e.date === selectedDate);
    const tagCls = e.outcome && e.outcome.includes('Runaway') ? 'tag-runaway' :
                   (e.outcome && e.outcome.includes('Digestion') ? 'tag-digestion' :
                   (e.outcome && e.outcome.includes('Trap') ? 'tag-trap' : 'tag-fade'));
    const retraceDisp = e.violated_48h ? '<span class="tag-violated">✗ Lost</span>' : '<span class="tag-held">✓ Held</span>';
    const c20 = (e.ret_20d != null && e.ret_20d >= 0 ? 'style="color:var(--green);"' : 'style="color:var(--red);"');
    const c60 = (e.ret_60d != null && e.ret_60d >= 0 ? 'style="color:var(--green);"' : 'style="color:var(--red);"');

    return `<tr class="${isSel ? 'selected' : ''}" onclick="onRowClick('${e.symbol}','${e.date}')">
      <td class="tl">${e.date}</td>
      <td class="tl" style="font-weight:700; color:#fff;">${e.symbol}</td>
      <td class="tl" style="max-width:130px; overflow:hidden; text-overflow:ellipsis;"><span style="color:#38bdf8; cursor:pointer; text-decoration:underline dotted;" title="Click to view ${e.sector} Stock Leaderboard" onclick="event.stopPropagation(); openGroupStocksModal('${e.sector}', 'sector')">${e.sector || '·'}</span></td>
      <td class="tl" style="max-width:130px; overflow:hidden; text-overflow:ellipsis;"><span style="color:#c084fc; cursor:pointer; text-decoration:underline dotted;" title="Click to view ${e.theme} Stock Leaderboard" onclick="event.stopPropagation(); openGroupStocksModal('${e.theme}', 'theme')">${e.theme || '·'}</span></td>
      <td>${e.gap_pct != null ? e.gap_pct.toFixed(1) + '%' : '·'}</td>
      <td>${e.rvol != null ? e.rvol.toFixed(1) + 'x' : '·'}</td>
      <td>${e.close_pos != null ? e.close_pos.toFixed(2) : '·'}</td>
      <td class="tl">${retraceDisp}</td>
      <td ${c20}>${e.ret_20d != null ? (e.ret_20d >= 0 ? '+' : '') + e.ret_20d.toFixed(1) + '%' : '·'}</td>
      <td ${c60}>${e.ret_60d != null ? (e.ret_60d >= 0 ? '+' : '') + e.ret_60d.toFixed(1) + '%' : '·'}</td>
      <td style="font-weight:700; color:var(--gold);">${e.max_gain != null ? e.max_gain.toFixed(1) + '%' : '·'}</td>
      <td class="tl"><span class="tag ${tagCls}">${e.outcome || '·'}</span></td>
      <td style="${e.ml_50 > 0.40 ? 'background-color:rgba(34, 197, 94, 0.2); font-weight:bold; color:#22c55e;' : ''}">${e.ml_50 != null ? (e.ml_50 * 100).toFixed(1) + '%' : '·'}</td>
      <td style="${e.ml_150 > 0.15 ? 'background-color:rgba(34, 197, 94, 0.2); font-weight:bold; color:#22c55e;' : ''}">${e.ml_150 != null ? (e.ml_150 * 100).toFixed(1) + '%' : '·'}</td>
    </tr>`;
  }).join('');

  tbody.innerHTML = rows;
}

function onRowClick(sym, dt) {
  selectedSym = sym; selectedDate = dt;
  renderTable();
  const ev = filteredEvents.find(e => e.symbol === sym && e.date === dt);
  if (ev) loadChart(ev);
}

function sortBy(col) {
  if (sortCol === col) sortAsc = !sortAsc;
  else { sortCol = col; sortAsc = false; }
  renderTable();
}

function loadChart(ev) {
  document.getElementById('chart_sym').textContent = `${ev.symbol} — ${ev.date} (${ev.subtype.toUpperCase()})`;
  document.getElementById('chart_sub').textContent =
    `Gap: +${(ev.gap_pct||0).toFixed(1)}% | RVOL: ${(ev.rvol||0).toFixed(1)}x | 20D: ${(ev.ret_20d||0).toFixed(1)}% | 60D: ${(ev.ret_60d||0).toFixed(1)}% | Peak: +${(ev.max_gain||0).toFixed(1)}%`;

  fetch(`/api/chart?symbol=${ev.symbol}&date=${ev.date}&use_trade1=${stratConfig.use_trade1?1:0}&use_ml_targets=${stratConfig.use_ml_targets?1:0}&use_ml_stop=${stratConfig.use_ml_stop?1:0}&use_multileg=${stratConfig.use_multileg?1:0}&use_multileg_ml=${stratConfig.use_multileg_ml?1:0}`).then(r => r.json()).then(data => {
    if (!data || !data.bars || !data.bars.length) return;

    // 1. Sector / Theme Context (Right Pane)
    const d = data.dossier;
    
    if (d && d.ai_scores) {
        if (d.ai_scores.dynamic_stop_loss_pct != null) {
            document.getElementById('ai_dynamic_stop').style.display = 'block';
            document.getElementById('ai_stop_val').textContent = (d.ai_scores.dynamic_stop_loss_pct * 100).toFixed(1) + '%';
            document.getElementById('ai_adv_warnings').style.display = 'block';
        } else {
            document.getElementById('ai_dynamic_stop').style.display = 'none';
        }
    }
    
    // Automatically trigger V2 load clamped to historical hold duration
    const slider = document.getElementById('epSlider');
    if (slider) { 
        // [FIX]: Clamp slider to the actual survival duration of the trade
        let max_survive = 5;
        if (d && d.trade_1 && d.trade_1.hold_days) {
            max_survive = Math.min(20, Math.max(1, d.trade_1.hold_days));
        }
        let default_val = Math.min(5, max_survive);
        slider.value = default_val; 
        onSliderMove(default_val);
    }

    if (d && d.sector_theme) {
      const st = d.sector_theme;
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

      const badgeEl = document.getElementById('ctx_diag_badge');
      badgeEl.className = 'pctile-badge ' + (st.macro_status === 'tailwind' ? 'pctile-high' : (st.macro_status === 'drag' ? 'pctile-low' : 'pctile-mid'));
      badgeEl.textContent = (st.macro_status === 'tailwind' ? '🟢 ' : (st.macro_status === 'drag' ? '🔴 ' : '🟡 ')) + st.macro_diag;
      
      const descEl = document.getElementById('ctx_diag_desc');
      descEl.className = 'diag-pill ' + (st.macro_status === 'tailwind' ? 'diag-tailwind' : (st.macro_status === 'drag' ? 'diag-drag' : 'diag-neutral'));
      descEl.textContent = st.macro_desc;
      document.getElementById('card_sec_thm').style.display = 'flex';
    }

    // 2. Stepper & Trade 1 Dossier
    if (d && d.windows && d.trade_1) {
      const w = d.windows;
      const tr1 = d.trade_1;

      const peadDrift = `${Math.round((ev.above_ema20_frac||0)*100)}% > 20 EMA · ${Math.round((ev.above_sma50_frac||0)*100)}% > 50 SMA`;
      document.getElementById('dossier_pead_badge').textContent = `PEAD: ${peadDrift}`;

      const w1Status = document.getElementById('w1_status');
      w1Status.textContent = w.w1.status;
      w1Status.style.color = (w.w1.close_pos >= 0.65 ? 'var(--green)' : 'var(--red)');
      document.getElementById('w1_detail').textContent = `Gap +${w.w1.gap_pct}% · RVOL ${w.w1.rvol}x · Pos ${w.w1.close_pos}`;

      const w2Status = document.getElementById('w2_status');
      w2Status.textContent = w.w2.status;
      w2Status.style.color = (w.w2.gap_and_go ? 'var(--accent)' : (w.w2.held_d1_low ? 'var(--gold)' : 'var(--red)'));
      document.getElementById('w2_detail').textContent = `Open ${w.w2.open_pct >= 0 ? '+' : ''}${w.w2.open_pct}% · Add: ${w.w2.add_triggered ? 'Triggered ✓' : 'Inside'}`;

      const w3Status = document.getElementById('w3_status');
      w3Status.textContent = w.w3.status;
      w3Status.style.color = (w.w3.held_absorption ? 'var(--green)' : 'var(--red)');
      document.getElementById('w3_detail').textContent = w.w3.held_absorption ? 'Upper 50% Body Held (Absorbed)' : '>50% Body Lost (Violated)';

      const w4Status = document.getElementById('w4_status');
      w4Status.textContent = w.w4.status;
      w4Status.style.color = (w.w4.close_5d_pct >= 0 ? 'var(--green)' : 'var(--red)');
      document.getElementById('w4_detail').textContent = `MFE +${w.w4.mfe_5d_pct}% · Low ${w.w4.held_low_5d ? 'Held ✓' : 'Breached ✗'}`;
      document.getElementById('card_stepper').style.display = 'flex';

      // Trade 1 Execution
      if (tr1.skipped) {
        document.getElementById('tr_entry').textContent = 'Skipped';
        document.getElementById('tr_stop').textContent = 'Trade 1 Entry OFF';
        document.getElementById('tr_add').textContent = '—';
        document.getElementById('tr_exit_price').textContent = 'No Position Taken';
        document.getElementById('tr_exit_reason').textContent = 'Skipped by User (Continuation Legs Only Mode)';
        document.getElementById('tr_exit_reason').style.color = 'var(--muted)';
        document.getElementById('tr_hold').textContent = '0 sessions';
        document.getElementById('row_sma50_adh').style.display = 'none';
        document.getElementById('tr_result').textContent = '0.0% (+0.00 R)';
        document.getElementById('tr_result').style.color = 'var(--muted)';
        document.getElementById('tr_exec_status').textContent = 'SKIPPED / 0.00 R';
        document.getElementById('tr_exec_status').style.color = 'var(--muted)';
        document.getElementById('card_trade_exec').style.display = 'flex';
      } else {
        document.getElementById('tr_entry').textContent = `$${tr1.entry_price.toFixed(2)}`;
        document.getElementById('tr_stop').textContent = `$${tr1.stop_price.toFixed(2)} (-${tr1.risk_pct.toFixed(1)}% risk)`;
        document.getElementById('tr_add').textContent = `$${tr1.add_trigger_price.toFixed(2)}`;

        document.getElementById('tr_exit_price').textContent = `$${tr1.exit_price.toFixed(2)} on ${tr1.exit_date}`;
        document.getElementById('tr_exit_reason').textContent = tr1.exit_reason;
        document.getElementById('tr_exit_reason').style.color = (tr1.stopped_out ? 'var(--red)' : '#38bdf8');
        document.getElementById('tr_hold').textContent = `${tr1.hold_days} sessions (~${Math.round(tr1.hold_days/21)} mos)`;

        const smaRow = document.getElementById('row_sma50_adh');
        if (tr1.sma50_adherence_pct != null) {
          smaRow.style.display = 'flex';
          const adhEl = document.getElementById('tr_sma50_adh');
          adhEl.textContent = `${tr1.sma50_adherence_pct}% of trend`;
          adhEl.style.color = (tr1.sma50_adherence_pct >= 70 ? 'var(--green)' : (tr1.sma50_adherence_pct >= 50 ? 'var(--gold)' : 'var(--red)'));
        } else {
          smaRow.style.display = 'none';
        }

        const resEl = document.getElementById('tr_result');
        const rSign1 = tr1.trade_r >= 0 ? '+' : '';
        resEl.textContent = `${tr1.trade_return_pct >= 0 ? '+' : ''}${tr1.trade_return_pct.toFixed(1)}% (${rSign1}${tr1.trade_r.toFixed(2)} R)`;
        resEl.style.color = (tr1.trade_r >= 0 ? 'var(--green)' : 'var(--red)');
        document.getElementById('tr_exec_status').textContent = `${rSign1}${tr1.trade_r.toFixed(2)} R`;
        document.getElementById('tr_exec_status').style.color = (tr1.trade_r >= 0 ? 'var(--green)' : 'var(--red)');
        document.getElementById('card_trade_exec').style.display = 'flex';
      }
    }

    // 2b. Ticker Historical EP genealogy bar (Both Center Pane & Dossier Card 0)
    const histBar = document.getElementById('ticker_history_bar');
    const histChips = document.getElementById('hist_chips');
    const histName = document.getElementById('hist_ticker_name');
    const histCount = document.getElementById('hist_count');

    const cardHist = document.getElementById('card_ticker_history');
    const dHistSym = document.getElementById('dossier_hist_sym');
    const dHistCount = document.getElementById('dossier_hist_count');
    const dHistChips = document.getElementById('dossier_hist_chips');

    if (data.ticker_history && data.ticker_history.length > 0) {
      histName.textContent = data.symbol;
      histCount.textContent = data.ticker_history.length;
      const chipsHtml = data.ticker_history.map(h => {
        const isCurrent = (h.date === ev.date);
        const gainText = (h.max_gain != null && h.max_gain > 0) ? `+${h.max_gain.toFixed(0)}%` : `${h.gap_pct.toFixed(0)}%`;
        return `<span class="hist-chip ${isCurrent ? 'active' : ''}" onclick="onHistChipClick('${data.symbol}', '${h.date}', '${h.subtype}')">
          <span>${h.date}</span>
          <span class="hist-chip-gain">${gainText}</span>
        </span>`;
      }).join('');
      histChips.innerHTML = chipsHtml;
      histBar.style.display = 'flex';

      if (cardHist && dHistChips) {
        dHistSym.textContent = data.symbol;
        dHistCount.textContent = `${data.ticker_history.length} Events (2016–2026)`;
        dHistChips.innerHTML = chipsHtml;
        cardHist.style.display = 'flex';
      }
    } else {
      histBar.style.display = 'none';
      if (cardHist) cardHist.style.display = 'none';
    }

    // 3. Trade 2 & 3 (Yellow Re-Entry Multi-Leg) Dossier
    const t2 = d ? d.trade_2 : null;
    const t3 = d ? d.trade_3 : null;
    const cardT2 = document.getElementById('card_trade2_exec');
    if (t2 && t2.has_reentry) {
      const netSeq = (t3 && t3.has_reentry) ? t3.combined_net_r : t2.combined_net_r;
      document.getElementById('t2_status_pill').textContent = `${t2.r_mult >= 0 ? '+' : ''}${t2.r_mult.toFixed(2)} R (Net: ${netSeq >= 0 ? '+' : ''}${netSeq.toFixed(2)} R)`;
      document.getElementById('t2_entry_val').textContent = `$${t2.entry_price.toFixed(2)} on ${t2.entry_date}`;
      document.getElementById('t2_stop_val').textContent = `$${t2.stop_price.toFixed(2)} (-${t2.risk_pct.toFixed(1)}% risk)`;
      document.getElementById('t2_blue_days').textContent = `${t2.cons_days} sessions (Pullback ${t2.drop_from_peak.toFixed(1)}% from peak)`;
      document.getElementById('t2_exit_val').textContent = `$${t2.exit_price.toFixed(2)} on ${t2.exit_date} (${t2.exit_reason})`;
      
      const rSign2 = t2.r_mult >= 0 ? '+' : '';
      document.getElementById('t2_return_val').textContent = `${t2.return_pct >= 0 ? '+' : ''}${t2.return_pct.toFixed(1)}% (${rSign2}${t2.r_mult.toFixed(2)} R)`;
      document.getElementById('t2_return_val').style.color = (t2.r_mult >= 0 ? 'var(--green)' : 'var(--red)');

      if (t2.prob_reentry != null) {
        document.getElementById('row_t2_ml_score').style.display = 'flex';
        document.getElementById('t2_ml_score_val').textContent = `${(t2.prob_reentry * 100).toFixed(1)}% (Institutional Gate Passed ≥40%)`;
      } else {
        document.getElementById('row_t2_ml_score').style.display = 'none';
      }

      const t3Section = document.getElementById('t3_section');
      if (t3 && t3.has_reentry) {
        document.getElementById('t3_entry_val').textContent = `$${t3.entry_price.toFixed(2)} on ${t3.entry_date}`;
        document.getElementById('t3_stop_val').textContent = `$${t3.stop_price.toFixed(2)} (-${t3.risk_pct.toFixed(1)}% risk)`;
        document.getElementById('t3_exit_val').textContent = `$${t3.exit_price.toFixed(2)} on ${t3.exit_date} (${t3.exit_reason})`;
        const rSign3 = t3.r_mult >= 0 ? '+' : '';
        document.getElementById('t3_return_val').textContent = `${t3.return_pct >= 0 ? '+' : ''}${t3.return_pct.toFixed(1)}% (${rSign3}${t3.r_mult.toFixed(2)} R)`;
        document.getElementById('t3_return_val').style.color = (t3.r_mult >= 0 ? 'var(--green)' : 'var(--red)');
        t3Section.style.display = 'flex';
      } else {
        t3Section.style.display = 'none';
      }

      const netSign = netSeq >= 0 ? '+' : '';
      document.getElementById('t2_combined_net_r').textContent = `${netSign}${netSeq.toFixed(2)} R (${netSeq >= 0 ? 'PROFIT' : 'LOSS'})`;
      document.getElementById('t2_combined_net_r').style.color = (netSeq >= 0 ? 'var(--green)' : 'var(--red)');

      document.getElementById('t2_details_box').style.display = 'flex';
      document.getElementById('t2_none_msg').style.display = 'none';
      cardT2.style.display = 'flex';
    } else {
      if (!stratConfig.use_multileg) {
        document.getElementById('t2_status_pill').textContent = 'Multi-Leg OFF';
        document.getElementById('t2_status_pill').style.color = '#94a3b8';
        document.getElementById('t2_none_msg').textContent = 'Multi-Leg (Trade 2 & 3) Re-entries are turned OFF in Strategy Configuration.';
      } else if (t2 && t2.ml_vetoed) {
        document.getElementById('t2_status_pill').textContent = 'ML Vetoed';
        document.getElementById('t2_status_pill').style.color = '#f59e0b';
        document.getElementById('t2_none_msg').textContent = t2.veto_reason || 'Re-Entry setup rejected by ML model (conviction < 40%).';
      } else {
        document.getElementById('t2_status_pill').textContent = 'No Re-Entry';
        document.getElementById('t2_status_pill').style.color = '#94a3b8';
        document.getElementById('t2_none_msg').textContent = 'No Second-Leg Re-Entry detected (did not re-flip Yellow within 45 sessions or retracement exceeded 55%).';
      }
      document.getElementById('t2_details_box').style.display = 'none';
      document.getElementById('t2_none_msg').style.display = 'block';
      cardT2.style.display = 'flex';
    }

    // 4. Center Candlestick Chart
    candleSeries.setData(data.bars);
    volSeries.setData(data.volume);
    if (data.ema20) ema20Series.setData(data.ema20);
    if (data.sma50) sma50Series.setData(data.sma50);

    activePriceLines.forEach(pl => {
      try { candleSeries.removePriceLine(pl); } catch (e) {}
    });
    activePriceLines = [];

    // Chart Markers
    const markers = [{
      time: ev.date, position: 'belowBar', color: '#10b981', shape: 'arrowUp', text: `T1 EP Entry: $${data.level != null ? data.bars.find(b => b.time === ev.date)?.close : ''}`
    }];
    
    if (d && d.trade_1 && d.trade_1.exit_date) {
      markers.push({
        time: d.trade_1.exit_date, position: 'aboveBar',
        color: d.trade_1.stopped_out ? '#ef4444' : '#38bdf8', shape: 'arrowDown',
        text: `T1 Exit: ${d.trade_1.trade_r >= 0 ? '+' : ''}${d.trade_1.trade_r}R (${d.trade_1.trade_return_pct}%)`
      });
    }

    if (t2 && t2.has_reentry && t2.entry_date) {
      markers.push({
        time: t2.entry_date, position: 'belowBar',
        color: '#fbbf24', shape: 'arrowUp',
        text: `T2 Yellow Re-Entry: $${t2.entry_price}`
      });
      if (t2.exit_date) {
        markers.push({
          time: t2.exit_date, position: 'aboveBar',
          color: t2.stopped_out ? '#ef4444' : '#34d399', shape: 'arrowDown',
          text: `T2 Exit: ${t2.r_mult >= 0 ? '+' : ''}${t2.r_mult}R (${t2.return_pct}%)`
        });
      }
    }

    if (t3 && t3.has_reentry && t3.entry_date) {
      markers.push({
        time: t3.entry_date, position: 'belowBar',
        color: '#a855f7', shape: 'arrowUp',
        text: `T3 3rd Yellow Flip: $${t3.entry_price}`
      });
      if (t3.exit_date) {
        markers.push({
          time: t3.exit_date, position: 'aboveBar',
          color: t3.stopped_out ? '#ef4444' : '#c084fc', shape: 'arrowDown',
          text: `T3 Exit: ${t3.r_mult >= 0 ? '+' : ''}${t3.r_mult}R (${t3.return_pct}%)`
        });
      }
    }

    candleSeries.setMarkers(markers);

    // Price lines
    if (data.level) {
      activePriceLines.push(candleSeries.createPriceLine({
        price: data.level, color: '#ef4444', lineWidth: 2, lineStyle: 2, axisLabelVisible: true, title: 'T1 Stop (D1 Low)'
      }));
    }
    if (data.trigger) {
      activePriceLines.push(candleSeries.createPriceLine({
        price: data.trigger, color: '#10b981', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: 'T1 Add Trigger'
      }));
    }
    if (t2 && t2.has_reentry && t2.stop_price) {
      activePriceLines.push(candleSeries.createPriceLine({
        price: t2.stop_price, color: '#f59e0b', lineWidth: 1, lineStyle: 3, axisLabelVisible: true, title: `T2 Stop ($${t2.stop_price})`
      }));
    }
    if (t3 && t3.has_reentry && t3.stop_price) {
      activePriceLines.push(candleSeries.createPriceLine({
        price: t3.stop_price, color: '#a855f7', lineWidth: 1, lineStyle: 3, axisLabelVisible: true, title: `T3 Stop ($${t3.stop_price})`
      }));
    }

    const container = document.getElementById('chart-container');
    chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
    chart.priceScale('right').applyOptions({ autoScale: true, scaleMargins: { top: 0.08, bottom: 0.25 } });
    chart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.84, bottom: 0.0 } });
    if (data.bars && data.bars.length > 0) {
      chart.timeScale().setVisibleLogicalRange({ from: 0, to: Math.min(data.bars.length - 1, 140) });
    }
  });

}

function onHistChipClick(sym, dt, sub) {
  selectedSym = sym; selectedDate = dt;
  const ev = filteredEvents.find(e => e.symbol === sym && e.date === dt) || { symbol: sym, date: dt, subtype: sub || 'ep' };
  loadChart(ev);
  renderTable();
}

// Attach filter listeners
['f_outcome', 'f_milestone', 'f_sector', 'f_theme', 'f_retrace', 'f_cpos', 'f_subtype', 'f_year'].forEach(id => {
  document.getElementById(id).addEventListener('change', fetchEvents);
});
document.getElementById('f_search').addEventListener('input', fetchEvents);

window.onload = () => {
  initChart();
  loadFilters();
  fetchEvents();
};
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return Response(HTML_TEMPLATE, mimetype="text/html")

@app.route("/api/download_handbook")
def api_download_handbook():
    pdf_path = Path("docs/EP_Pinnacle_Elite_Strategy_Handbook.pdf")
    if pdf_path.exists():
        return send_file(pdf_path, mimetype="application/pdf", as_attachment=False, download_name="EP_Pinnacle_Elite_Strategy_Handbook.pdf")
    return jsonify({"error": "Handbook PDF not found"}), 404

@app.route("/api/filters")
def api_filters():
    df = get_df()
    if df.empty:
        return jsonify({"sectors": [], "themes": [], "years": []})
    sectors = sorted([s for s in df["sector"].dropna().unique() if s])
    themes = sorted([t for t in df["theme"].dropna().unique() if t and t != "General"])
    years = sorted([int(y) for y in df["year"].dropna().unique()], reverse=True)
    return jsonify({"sectors": sectors, "themes": themes, "years": years})

@app.route("/api/group_stocks")
def api_group_stocks():
    group = request.args.get("group", "")
    kind = request.args.get("kind", "theme")
    if not group:
        return jsonify({"error": "Missing group"}), 400
    eng = thematic_engine.get_thematic_engine()
    return jsonify(eng.get_group_stocks(group, kind))

@app.route("/api/events")
def api_events():
    df = get_df()
    if df.empty:
        return jsonify({"events": [], "kpis": {}})

    q = df.copy()

    preset = request.args.get("preset", "all")
    if preset == "pinnacle":
        q = q[
            (q["close_pos"] >= 0.65) &
            (q["rvol"] >= 2.5) &
            (q["gap_pct"] >= 5.0) &
            (q["sector"].isin(TOP_SECTORS) | q["theme"].isin(TOP_THEMES)) &
            ((q["sec_m1_pctile"] >= 50) | (q["thm_m1_pctile"] >= 50))
        ]
    elif preset == "emerging":
        q = q[
            (q["close_pos"] >= 0.65) &
            (q["rvol"] >= 2.5) &
            ((q["sec_m1_pctile"] >= 65) | (q["thm_m1_pctile"] >= 65) | (q["sec_m3_pctile"] >= 65))
        ]
    elif preset == "sweet_spot":
        q = q[(q["close_pos"] >= 0.65) & (q["rvol"] >= 3.0) & (q["gap_pct"] >= 6.0)]
    elif preset == "sweet_spot_failures":
        q = q[(q["close_pos"] >= 0.65) & (q["rvol"] >= 3.0) & (q["gap_pct"] >= 6.0) & ((q["breached_d1_low_5d"]) | (q["outcome"] == "Gap & Crap Trap") | (q["ret_20d"] < 0))]
    elif preset == "compounders":
        top_clusters = [
            "Computer Hardware", "Semiconductors", "Biotechnology", "Semiconductor Equipment",
            "Bitcoin Miners", "Quantum Computing", "Uranium & Nuclear", "Clean Energy",
            "Software - Application", "Software - Infrastructure", "Aerospace & Defense"
        ]
        q = q[(q["sector"].isin(top_clusters) | q["theme"].isin(top_clusters))]
    elif preset == "traps":
        q = q[(q["violated_48h"]) | (q["close_pos"] < 0.50)]
    elif preset == "turnaround":
        q = q[(q["neglect_6m"] <= -15.0) & (q["rvol"] >= 4.0) & (q["close_pos"] >= 0.70)]
    elif preset == "high_beta":
        hb = ["Bitcoin Miners", "Quantum Computing", "Semiconductor Equipment", "Clean Energy", "Data Centers"]
        q = q[(q["theme"].isin(hb)) & (q["rvol"] >= 3.5)]
    elif preset == "idiosyncratic":
        is_headwind = (
            ((q["sec_m1_pctile"] < 35) & (q["sec_m3_pctile"] < 35)) |
            (q["thm_m1_pctile"] < 35)
        )
        q = q[
            is_headwind &
            (q["close_pos"] >= 0.80) &
            (q["rvol"] >= 6.0) &
            (q["dvol_m"] >= 75.0)
        ]
    elif preset == "conservative":
        q = q[
            (q["close_pos"] >= 0.65) &
            (q["rvol"] >= 2.5) &
            (~q["breached_d1_low_5d"])
        ]

    # Booster condition toggles
    if request.args.get("cond_veto_headwind") == "1":
        is_severe_headwind = (
            ((q["sec_m1_pctile"] < 35) & (q["sec_m3_pctile"] < 35)) |
            (q["thm_m1_pctile"] < 35)
        )
        q = q[~is_severe_headwind]
    if request.args.get("cond_win_sectors") == "1":
        q = q[q["sector"].isin(TOP_SECTORS) | q["theme"].isin(TOP_THEMES)]
    if request.args.get("cond_tailwind") == "1":
        q = q[(q["sec_m1_pctile"] >= 60) | (q["thm_m1_pctile"] >= 60)]
    if request.args.get("cond_48h") == "1":
        q = q[~q["violated_48h"]]
    if request.args.get("cond_elite_close") == "1":
        q = q[q["close_pos"] >= 0.80]
    if request.args.get("cond_held_5d") == "1":
        q = q[~q["breached_d1_low_5d"]]

    outcome = request.args.get("outcome")
    if outcome:
        q = q[q["outcome"] == outcome]

    milestone = request.args.get("milestone")
    if milestone == "tier_30_75":
        q = q[(q["max_gain"] >= 30.0) & (q["max_gain"] < 80.0)]
    elif milestone == "tier_80_150":
        q = q[(q["max_gain"] >= 80.0) & (q["max_gain"] < 200.0)]
    elif milestone == "tier_200_plus":
        q = q[q["max_gain"] >= 200.0]
    elif milestone == "30":
        q = q[q["max_gain"] >= 30.0]
    elif milestone == "50":
        q = q[q["hit_50pct"]]
    elif milestone == "100":
        q = q[q["hit_100pct"]]
    elif milestone == "200":
        q = q[q["hit_200pct"]]
    elif milestone == "300":
        q = q[q["hit_300pct"]]

    sector = request.args.get("sector")
    if sector:
        q = q[q["sector"] == sector]

    theme = request.args.get("theme")
    if theme:
        q = q[q["theme"] == theme]

    retrace = request.args.get("retrace")
    if retrace == "held":
        q = q[~q["violated_48h"]]
    elif retrace == "violated":
        q = q[q["violated_48h"]]

    cpos = request.args.get("cpos")
    if cpos == "elite":
        q = q[q["close_pos"] >= 0.85]
    elif cpos == "strong":
        q = q[(q["close_pos"] >= 0.65) & (q["close_pos"] < 0.85)]
    elif cpos == "weak":
        q = q[q["close_pos"] < 0.65]

    subtype = request.args.get("subtype")
    if subtype:
        q = q[q["subtype"] == subtype]

    year = request.args.get("year")
    if year:
        q = q[q["year"] == int(year)]

    search = request.args.get("search")
    if search:
        q = q[q["symbol"].str.contains(search, case=False, na=False)]

    n = len(q)
    def _safe_float(val, default=0.0):
        if val is None or pd.isna(val):
            return default
        return float(val)
    use_trade1 = request.args.get("use_trade1", "1") in ["1", "true", "True"]
    use_ml_targets = request.args.get("use_ml_targets", "1") in ["1", "true", "True"]
    use_ml_stop = request.args.get("use_ml_stop", "1") in ["1", "true", "True"]
    use_multileg = request.args.get("use_multileg", "1") in ["1", "true", "True"]
    use_multileg_ml = request.args.get("use_multileg_ml", "1") in ["1", "true", "True"]

    n = len(q)
    if n > 0:
        big_move_rate = _safe_float((q["max_gain"] >= 30.0).mean() * 100.0)
        tier_30_75 = _safe_float(((q["max_gain"] >= 30.0) & (q["max_gain"] < 80.0)).mean() * 100.0)
        tier_80_150 = _safe_float(((q["max_gain"] >= 80.0) & (q["max_gain"] < 200.0)).mean() * 100.0)
        win_5d = _safe_float((q["close_5d_pct"] > 0).mean() * 100.0) if "close_5d_pct" in q.columns else 0.0
        win_20d = _safe_float((q["ret_20d"] > 0).mean() * 100.0)
        win_60d = _safe_float((q["ret_60d"] > 0).mean() * 100.0) if "ret_60d" in q.columns else 0.0
        win_180d = _safe_float((q["ret_180d"] > 0).mean() * 100.0) if "ret_180d" in q.columns else 0.0

        trap_rate = _safe_float((q["outcome"] == "Gap & Crap Trap").mean() * 100.0)
        doubler_rate = _safe_float(q["hit_100pct"].mean() * 100.0)
        triple_rate = _safe_float(q["hit_200pct"].mean() * 100.0)
        mean_5d = _safe_float(q["close_5d_pct"].mean()) if "close_5d_pct" in q.columns else 0.0
        mean_20d = _safe_float(q["ret_20d"].mean()) if "ret_20d" in q.columns else 0.0
        mean_60d = _safe_float(q["ret_60d"].mean()) if "ret_60d" in q.columns else 0.0
        mean_180d = _safe_float(q["ret_180d"].dropna().mean()) if "ret_180d" in q.columns and len(q["ret_180d"].dropna()) > 0 else 0.0

        executed_trades = []
        for _, r_ev in q.iterrows():
            if use_trade1:
                if use_ml_targets:
                    t1_r = float(r_ev.get("t1_pyramid_r", r_ev.get("t1_real_r", 0.0)))
                elif use_ml_stop:
                    t1_r = float(r_ev.get("t1_dyn_r", r_ev.get("t1_real_r", 0.0)))
                else:
                    t1_r = float(r_ev.get("t1_real_r", 0.0))
                executed_trades.append(t1_r)

            if use_multileg and r_ev.get("t2_has_reentry", False):
                t2_val = float(r_ev.get("t2_real_r", 0.0))
                if use_multileg_ml:
                    t2_prob = r_ev.get("t2_prob_reentry", 0.0)
                    if pd.notna(t2_prob) and t2_prob >= 0.35:
                        executed_trades.append(t2_val)
                else:
                    executed_trades.append(t2_val)

            if use_multileg and r_ev.get("t3_has_reentry", False):
                t3_val = float(r_ev.get("t3_real_r", 0.0))
                if use_multileg_ml:
                    t3_prob = r_ev.get("t3_prob_reentry", 0.0)
                    if pd.notna(t3_prob) and t3_prob >= 0.35:
                        executed_trades.append(t3_val)
                else:
                    executed_trades.append(t3_val)

        trade_arr = np.array(executed_trades) if len(executed_trades) > 0 else np.array([0.0])
        wins = trade_arr[trade_arr > 0]
        losses = trade_arr[trade_arr <= 0]
        strat_count = len(executed_trades)
        strat_winrate = round(float(len(wins) / len(trade_arr) * 100.0), 1) if len(trade_arr) > 0 else 0.0
        strat_avg_win = round(float(wins.mean()), 1) if len(wins) > 0 else 0.0
        strat_avg_loss = round(float(losses.mean()), 1) if len(losses) > 0 else 0.0
        strat_ev = round(float(trade_arr.mean()), 2) if len(trade_arr) > 0 else 0.0
        strat_pnl = round(float(trade_arr.sum()), 1)
        strat_pnl_comp = round(strat_pnl * 0.70, 1)
        equity_curve = np.cumsum(trade_arr)
        peaks = np.maximum.accumulate(equity_curve)
        dd = equity_curve - peaks
        strat_max_dd = round(float(dd.min()), 2) if len(dd) > 0 else 0.0
        strat_pf = round(float(abs(wins.sum() / losses.sum())), 2) if len(losses) > 0 and losses.sum() != 0 else 0.0


        kpis = {
            "count": n,
            "strat_count": strat_count,
            "strat_ev": strat_ev,
            "strat_pnl": strat_pnl,
            "strat_pnl_comp": strat_pnl_comp,
            "strat_winrate": strat_winrate,
            "strat_avg_win": strat_avg_win,
            "strat_avg_loss": strat_avg_loss,
            "strat_max_dd": strat_max_dd,
            "strat_pf": strat_pf,
            "expectancy_r": strat_ev,
            "big_move_rate": round(big_move_rate, 1),
            "tier_30_75": round(tier_30_75, 1),
            "tier_80_150": round(tier_80_150, 1),
            "win_5d": round(win_5d, 1),
            "win_20d": round(win_20d, 1),
            "win_60d": round(win_60d, 1),
            "win_180d": round(win_180d, 1),
            "trap_rate": round(trap_rate, 1),
            "doubler_rate": round(doubler_rate, 1),
            "triple_rate": round(triple_rate, 1),
            "mean_5d": round(mean_5d, 1),
            "mean_20d": round(mean_20d, 1),
            "mean_60d": round(mean_60d, 1),
            "mean_180d": round(mean_180d, 1),
        }
    else:
        kpis = {
            "count": 0, "strat_count": 0, "strat_ev": 0.0, "strat_pnl": 0.0, "strat_pnl_comp": 0.0,
            "strat_winrate": 0.0, "strat_avg_win": 0.0, "strat_avg_loss": 0.0, "strat_max_dd": 0.0, "strat_pf": 0.0,
            "expectancy_r": 0.0, "big_move_rate": 0.0, "tier_30_75": 0.0, "tier_80_150": 0.0,
            "win_5d": 0.0, "win_20d": 0.0, "win_60d": 0.0, "win_180d": 0.0,
            "trap_rate": 0.0, "doubler_rate": 0.0, "triple_rate": 0.0,
            "mean_5d": 0.0, "mean_20d": 0.0, "mean_60d": 0.0, "mean_180d": 0.0
        }

    def _clean_val(v):
        if pd.isna(v) or v is None:
            return None
        if isinstance(v, (np.floating, float)):
            return round(float(v), 2)
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, (np.bool_, bool)):
            return bool(v)
        return v

    q = q.sort_values("date", ascending=False)
    raw_records = q.to_dict(orient="records")
    records = [{k: _clean_val(v) for k, v in r.items()} for r in raw_records]
    return jsonify({"events": records, "kpis": kpis})

def compute_dossier(sym: str, date_str: str, d: pd.DataFrame, pos: int, ev_row: dict | None,
                    use_ml_targets: bool = True, use_ml_stop: bool = True,
                    use_multileg: bool = True, use_multileg_ml: bool = True,
                    use_trade1: bool = True):
    # d already has full Larsson Line indicators attached
    d1 = d.iloc[pos]
    d1_close = float(d1["close"])
    d1_open = float(d1["open"])
    d1_high = float(d1["high"])
    d1_low = float(d1["low"])
    risk1_pct = (d1_close - d1_low) / d1_close * 100.0 if d1_close > 0 else 11.1

    # Sector / Theme context
    sec = ev_row.get("sector", "Unknown") if ev_row else "Unknown"
    thm = ev_row.get("theme", "General") if ev_row else "General"
    sec_m1 = float(ev_row.get("sec_m1_pctile")) if ev_row and pd.notna(ev_row.get("sec_m1_pctile")) else None
    sec_m3 = float(ev_row.get("sec_m3_pctile")) if ev_row and pd.notna(ev_row.get("sec_m3_pctile")) else None
    thm_m1 = float(ev_row.get("thm_m1_pctile")) if ev_row and pd.notna(ev_row.get("thm_m1_pctile")) else None

    # Idiosyncratic Alpha check
    rvol_val = float(ev_row.get("rvol", 0.0)) if ev_row and pd.notna(ev_row.get("rvol")) else 0.0
    dvol_val = float(ev_row.get("dvol_m", 0.0)) if ev_row and pd.notna(ev_row.get("dvol_m")) else 0.0
    cpos_val = float(ev_row.get("close_pos", 0.0)) if ev_row and pd.notna(ev_row.get("close_pos")) else 0.0
    held_48h_val = not bool(ev_row.get("violated_48h", False)) if ev_row else True
    is_headwind = (thm_m1 is not None and thm_m1 <= 35) or (sec_m1 is not None and sec_m1 <= 35)
    is_idiosyncratic = is_headwind and (rvol_val >= 6.0 and dvol_val >= 75.0 and cpos_val >= 0.80 and held_48h_val)

    # Macro diagnosis
    if is_idiosyncratic:
        macro_diag = "🎯 Idiosyncratic Alpha Catalyst (0.50R Half-Heat)"
        macro_desc = f"Blowout catalyst volume (${dvol_val:.1f}M, RVOL {rvol_val:.1f}x) overrides {thm} sector lag. 10-Yr Study: 5.3% trap rate, +3.73 R EV. Strictly 0.50R risk, no Day 2 adds."
        macro_status = "tailwind"
    elif (thm_m1 is not None and thm_m1 >= 70) or (sec_m1 is not None and sec_m1 >= 75):
        macro_diag = "Strong Institutional Tailwind"
        macro_desc = "Top decile money flow sponsorship. High probability of multi-quarter PEAD continuation."
        macro_status = "tailwind"
    elif (thm_m1 is not None and thm_m1 <= 30) or (sec_m1 is not None and sec_m1 <= 25):
        macro_diag = "Institutional Drag / Headwind"
        macro_desc = "Lacking sector/theme momentum sponsorship. Prone to early churn or false breakout fader."
        macro_status = "drag"
    else:
        macro_diag = "Neutral / Market-In-Line Flow"
        macro_desc = "Sector and theme momentum aligned with broader market baseline."
        macro_status = "neutral"

    # Window 1: Day 1 Surge & Accumulation
    cpos = round(float(ev_row.get("close_pos", 0.5)), 2) if ev_row else 0.5
    w1 = {
        "gap_pct": round(float(ev_row.get("gap_pct", 0.0)), 1) if ev_row else 0.0,
        "rvol": round(float(ev_row.get("rvol", 0.0)), 1) if ev_row else 0.0,
        "dvol_m": round(float(ev_row.get("dvol_m", 0.0)), 1) if ev_row else 0.0,
        "close_pos": cpos,
        "status": "Elite Accumulation (Top 15%)" if cpos >= 0.85 else ("Strong Range (Top 35%)" if cpos >= 0.65 else "Weak Range (<0.65)")
    }

    # Window 2: Day 2 (18H Open & Breakout Add)
    w2 = {"status": "Pending / N/A"}
    if pos + 1 < len(d):
        d2 = d.iloc[pos + 1]
        d2_open = float(d2["open"])
        d2_high = float(d2["high"])
        d2_low = float(d2["low"])
        gap_go = d2_open >= d1_close
        add_trig = d2_high > d1_high
        held_d1_low = d2_low >= d1_low
        open_pct = (d2_open / d1_close - 1.0) * 100.0
        w2 = {
            "d2_open": round(d2_open, 2),
            "open_pct": round(open_pct, 1),
            "gap_and_go": bool(gap_go),
            "add_triggered": bool(add_trig),
            "held_d1_low": bool(held_d1_low),
            "status": "Gap & Go (+Add Triggered)" if (gap_go and add_trig) else ("Gap & Go" if gap_go else ("Inside Digestion" if held_d1_low else "Slid Below D1 Low"))
        }

    # Window 3: Day 3 (48H Absorption Gate)
    w3 = {"status": "Pending / N/A"}
    if ev_row:
        violated = bool(ev_row.get("violated_48h", False))
        d3_high = float(d.iloc[pos + 2]["high"]) if pos + 2 < len(d) else None
        broke_high_48h = bool(d3_high > d1_high) if d3_high is not None else False
        w3 = {
            "held_absorption": not violated,
            "broke_high_48h": broke_high_48h,
            "status": "✓ 48H Upper Body Absorbed" if not violated else "✗ 48H Violated (>50% body lost)"
        }

    # Window 4: Day 5 (Momentum Leg Resolution)
    w4 = {"status": "Pending / N/A"}
    if pos + 5 < len(d):
        sub5 = d.iloc[pos:pos+6]
        c5 = float(d.iloc[pos+5]["close"])
        c5_pct = (c5 / d1_close - 1.0) * 100.0
        mfe_5d = (sub5["high"].max() / d1_close - 1.0) * 100.0
        held_low_5d = bool(sub5["low"].min() >= d1_low)
        w4 = {
            "close_5d_pct": round(c5_pct, 1),
            "mfe_5d_pct": round(mfe_5d, 1),
            "held_low_5d": held_low_5d,
            "status": f"{c5_pct:+.1f}% Continuation" if held_low_5d and c5_pct > 0 else (f"{c5_pct:+.1f}% Consolidation" if held_low_5d else "Breached D1 Low (Stop Hit)")
        }

    # Trade 1: Base EP Simulation
    fwd = d.iloc[pos:min(len(d), pos + 250)].copy()
    t1_stopped = False
    t1_exit_bar = None
    t1_exit_price = None
    t1_peak = d1_close
    seen_bull = False

    current_stop = d1_low
    stop_hurdle_cleared = False
    pyramid_added = False
    from ep_ml_engine import engine
    for b in range(1, len(fwd)):
        c = float(fwd["close"].iloc[b])
        l = float(fwd["low"].iloc[b])
        h = float(fwd["high"].iloc[b])
        s = fwd["larsson_state"].iloc[b]
        if h > t1_peak: t1_peak = h
        if pd.notna(s) and s in ["yellow", "gray"]:
            seen_bull = True
            
        if l <= current_stop:
            t1_stopped = True
            t1_exit_bar = b
            o = float(fwd["open"].iloc[b])
            t1_exit_price = o if o < current_stop else current_stop
            t1_reason = "Gap Down Stop Hit" if o < current_stop else ("ML Dynamic Stop Hit" if current_stop > d1_low else "Day 1 Low Stop Hit")
            break
            
        # Non-lookahead Activation Hurdle:
        # Dynamic trailing stop only activates once trade achieves >= +2.0 R unrealized profit at session close
        unrealized_r = (c - d1_close) / (d1_close - d1_low) if (d1_close - d1_low) > 0 else 0.0
        if unrealized_r >= 2.0:
            stop_hurdle_cleared = True

        # ML Inference (limit up to 60 days)
        ml_exited = False
        if b <= 60 and (use_ml_targets or use_ml_stop):
            features = engine.compute_rolling_features(sym, pos, pos + b)
            if features is not None:
                preds = engine.predict_rolling(features)
                if b == 4 and use_ml_targets and preds.get("prob_50", 0.0) >= 0.50:
                    pyramid_added = True

                if use_ml_targets:
                    prob_200 = preds.get("prob_200", 0.0)
                    prob_exh = preds.get("prob_exhaustion", 0.0)
                    # V2 Hybrid Target Rule: High-runner stocks (prob_200 >= 0.15) bypass early exhaustion exits to ride 50 SMA
                    if prob_200 < 0.15 and prob_exh > 0.80:
                        t1_exit_bar = b
                        t1_exit_price = c
                        t1_reason = "ML Exhaustion Alert (Prob > 0.80)"
                        ml_exited = True
                        break
                    
                if use_ml_stop and stop_hurdle_cleared:
                    dyn_stop_pct = preds.get("dynamic_stop_loss_pct", None)
                    if dyn_stop_pct is not None and not np.isnan(dyn_stop_pct):
                        new_stop = c * (1.0 + dyn_stop_pct)
                        # Once +2.0R hurdle is cleared, stop cannot drop below breakeven (entry price)
                        new_stop = max(new_stop, d1_close)
                        if new_stop > current_stop:
                            current_stop = new_stop
                        
        if not ml_exited:
            # 50 SMA Institutional Exit Rule
            sma50 = fwd["sma50"].iloc[b] if "sma50" in fwd.columns else None
            if seen_bull and pd.notna(sma50) and c < sma50:
                t1_exit_bar = b
                t1_exit_price = c
                t1_reason = "50 SMA Breakdown"
                break

    if t1_exit_bar is None:
        t1_exit_bar = len(fwd) - 1
        t1_exit_price = float(fwd["close"].iloc[-1])
        t1_reason = "Active / Window End"

    t1_ret = (t1_exit_price / d1_close - 1.0) * 100.0
    t1_r = t1_ret / risk1_pct if risk1_pct > 0 else 0.0
    if pyramid_added and t1_exit_bar > 4 and len(fwd) > 5:
        c5 = float(fwd["close"].iloc[4]) # Day 5 close
        risk1_dollars = d1_close - d1_low
        if risk1_dollars > 0:
            tranche2_r = 0.5 * (t1_exit_price - c5) / risk1_dollars
            t1_r = round(t1_r + tranche2_r, 2)

    # 50-Day SMA Institutional Baseline adherence over the trade hold
    sma50_adh = None
    if "sma50" in fwd.columns and t1_exit_bar > 0:
        fwd_hold = fwd.iloc[1:t1_exit_bar + 1]
        valid_sma = fwd_hold.dropna(subset=["sma50"])
        if len(valid_sma) > 0:
            held_cnt = (valid_sma["close"] >= valid_sma["sma50"]).sum()
            sma50_adh = round(float(held_cnt) / len(valid_sma) * 100.0, 1)

    if use_trade1:
        trade_1 = {
            "skipped": False,
            "entry_price": round(d1_close, 2),
            "stop_price": round(d1_low, 2),
            "risk_pct": round(risk1_pct, 2),
            "add_trigger_price": round(d1_high, 2),
            "exit_date": fwd.index[t1_exit_bar].strftime("%Y-%m-%d"),
            "exit_price": round(t1_exit_price, 2),
            "exit_reason": t1_reason,
            "stopped_out": bool(t1_stopped),
            "hold_days": int(t1_exit_bar),
            "return_pct": round(t1_ret, 1),
            "r_mult": round(t1_r, 2),
            "trade_return_pct": round(t1_ret, 1),
            "trade_r": round(t1_r, 2),
            "sma50_adherence_pct": sma50_adh,
            "pyramid_added": bool(pyramid_added),
            "pyramid_note": "AI Pyramid (+50% size on Day 5 V2 Conviction)" if pyramid_added else None
        }
    else:
        trade_1 = {
            "skipped": True,
            "entry_price": round(d1_close, 2),
            "stop_price": round(d1_low, 2),
            "risk_pct": round(risk1_pct, 2),
            "add_trigger_price": round(d1_high, 2),
            "exit_date": "—",
            "exit_price": 0.0,
            "exit_reason": "Skipped (Trade 1 OFF — Continuation Legs Only)",
            "stopped_out": False,
            "hold_days": 0,
            "return_pct": 0.0,
            "r_mult": 0.0,
            "trade_return_pct": 0.0,
            "trade_r": 0.0,
            "sma50_adherence_pct": None
        }
        t1_r = 0.0

    # Trade 2: Yellow Re-Entry (Second Leg) Simulation
    trade_2 = {"has_reentry": False}
    t2_exit_bar = None
    t2_r = 0.0
    t2_entry = None
    t2_stopped = False

    if use_multileg:
        # Precompute intermediate ribbon (8, 12, 16, 21) states
        e8 = fwd["close"].ewm(span=8, adjust=False).mean()
        e12 = fwd["close"].ewm(span=12, adjust=False).mean()
        e16 = fwd["close"].ewm(span=16, adjust=False).mean()
        e21 = fwd["close"].ewm(span=21, adjust=False).mean()
        bull_interm = (e8 >= e12) & (e12 >= e16) & (e16 >= e21)
        bear_interm = (e8 < e12) & (e12 < e16) & (e16 < e21)
        ribbon_state = pd.Series("gray", index=fwd.index)
        ribbon_state.loc[bull_interm] = "yellow"
        ribbon_state.loc[bear_interm] = "blue"
        fwd["ribbon_state"] = ribbon_state

        # Look forward from t1_exit_bar up to 90 sessions
        seen_cons = False
        cons_days = 0
        reentry_bar = None
        cons_low = float(fwd["low"].iloc[t1_exit_bar])

        for b in range(t1_exit_bar, min(len(fwd), t1_exit_bar + 90)):
            l = float(fwd["low"].iloc[b])
            h = float(fwd["high"].iloc[b])
            s = fwd["ribbon_state"].iloc[b]
            if l < cons_low: cons_low = l
            if h > t1_peak: t1_peak = h
            if pd.notna(s) and s in ["blue", "gray"]:
                seen_cons = True
                cons_days += 1
            if seen_cons and b > t1_exit_bar and pd.notna(s) and s == "yellow":
                reentry_bar = b
                break

        if reentry_bar is not None and cons_days <= 45:
            drop_from_peak = (t1_peak - cons_low) / t1_peak * 100.0 if t1_peak > 0 else 0.0
            # Retracement gate: pullback low not more than 55% from peak (allows 50% + buffer)
            if drop_from_peak <= 55.0:
                t2_entry = float(fwd["close"].iloc[reentry_bar])
                # Stop loss: 5-day swing low or 32 EMA
                swing5_low = float(fwd["low"].iloc[max(0, reentry_bar - 5):reentry_bar + 1].min())
                t2_stop = round(swing5_low, 2)
                t2_risk_pct = (t2_entry - t2_stop) / t2_entry * 100.0

                if 0.1 <= t2_risk_pct <= 35.0:
                    re_ml_vetoed = False
                    prob_reentry = None
                    if use_multileg_ml:
                        try:
                            re_feats = engine.compute_rolling_features(sym, pos, pos + reentry_bar)
                            if re_feats is not None:
                                re_preds = engine.predict_rolling(re_feats)
                                prob_reentry = float(re_preds.get("prob_reentry", 0.0))
                                if prob_reentry < 0.35:
                                    re_ml_vetoed = True
                        except Exception:
                            pass

                    if re_ml_vetoed:
                        trade_2 = {
                            "has_reentry": False,
                            "ml_vetoed": True,
                            "prob_reentry": round(prob_reentry, 3) if prob_reentry is not None else 0.0,
                            "veto_reason": f"Vetoed by Cost-Sensitive ML Re-Entry Model ({prob_reentry*100:.1f}% < 35.0% institutional threshold)"
                        }
                    else:
                        t2_stopped = False
                        t2_exit_price = None

                        for b in range(reentry_bar + 1, len(fwd)):
                            c = float(fwd["close"].iloc[b])
                            l = float(fwd["low"].iloc[b])
                            s = fwd["ribbon_state"].iloc[b]
                            if l <= t2_stop:
                                t2_stopped = True
                                t2_exit_bar = b
                                t2_exit_price = t2_stop
                                t2_reason = "Swing Low Stop Hit"
                                break
                            if pd.notna(s) and s in ["blue", "gray"]:
                                t2_exit_bar = b
                                t2_exit_price = c
                                t2_reason = "Ribbon Bearish Flip"
                                break

                        if t2_exit_bar is None:
                            t2_exit_bar = len(fwd) - 1
                            t2_exit_price = float(fwd["close"].iloc[-1])
                            t2_reason = "Active / Window End"

                        t2_ret = (t2_exit_price / t2_entry - 1.0) * 100.0 if not t2_stopped else -t2_risk_pct
                        t2_r = t2_ret / t2_risk_pct if t2_risk_pct > 0 else 0.0

                        trade_2 = {
                            "has_reentry": True,
                            "entry_date": fwd.index[reentry_bar].strftime("%Y-%m-%d"),
                            "entry_price": round(t2_entry, 2),
                            "stop_price": round(t2_stop, 2),
                            "risk_pct": round(t2_risk_pct, 1),
                            "cons_days": int(cons_days),
                            "drop_from_peak": round(drop_from_peak, 1),
                            "exit_date": fwd.index[t2_exit_bar].strftime("%Y-%m-%d"),
                            "exit_price": round(t2_exit_price, 2),
                            "exit_reason": t2_reason,
                            "stopped_out": bool(t2_stopped),
                            "hold_days": int(t2_exit_bar - reentry_bar),
                            "return_pct": round(t2_ret, 1),
                            "r_mult": round(t2_r, 2),
                            "combined_net_r": round(t1_r + t2_r, 2),
                            "prob_reentry": round(prob_reentry, 3) if prob_reentry is not None else None
                        }

    # Trade 3: Leg 3 (Subsequent 3rd Yellow Flip Runner)
    trade_3 = {"has_reentry": False}
    if use_multileg and trade_2.get("has_reentry") and t2_exit_bar is not None and t2_exit_bar < len(fwd) - 10:
        t2_peak = float(fwd["close"].iloc[reentry_bar])
        seen_cons_3 = False
        cons_days_3 = 0
        cons_low_3 = float(fwd["low"].iloc[t2_exit_bar])
        reentry_bar_3 = None

        for b in range(t2_exit_bar, min(len(fwd), t2_exit_bar + 90)):
            l = float(fwd["low"].iloc[b]); h = float(fwd["high"].iloc[b]); s = fwd["ribbon_state"].iloc[b]
            if h > t2_peak: t2_peak = h
            if l < cons_low_3: cons_low_3 = l
            if pd.notna(s) and s in ["blue", "gray"]:
                seen_cons_3 = True; cons_days_3 += 1
            if seen_cons_3 and b > t2_exit_bar and pd.notna(s) and s == "yellow":
                reentry_bar_3 = b; break

        if reentry_bar_3 is not None and cons_days_3 <= 45:
            drop_3 = (t2_peak - cons_low_3) / t2_peak * 100.0 if t2_peak > 0 else 0.0
            if drop_3 <= 55.0:
                t3_entry = float(fwd["close"].iloc[reentry_bar_3])
                swing5_3 = float(fwd["low"].iloc[max(0, reentry_bar_3 - 5):reentry_bar_3 + 1].min())
                t3_risk_pct = (t3_entry - swing5_3) / t3_entry * 100.0
                if 0.1 <= t3_risk_pct <= 35.0:
                    re3_ml_vetoed = False
                    prob_reentry3 = None
                    if use_multileg_ml:
                        try:
                            re3_feats = engine.compute_rolling_features(sym, pos, pos + reentry_bar_3)
                            if re3_feats is not None:
                                re3_preds = engine.predict_rolling(re3_feats)
                                prob_reentry3 = float(re3_preds.get("prob_reentry", 0.0))
                                if prob_reentry3 < 0.35:
                                    re3_ml_vetoed = True
                        except Exception:
                            pass

                    if re3_ml_vetoed:
                        trade_3 = {
                            "has_reentry": False,
                            "ml_vetoed": True,
                            "prob_reentry": round(prob_reentry3, 3) if prob_reentry3 is not None else 0.0,
                            "veto_reason": f"Vetoed by Cost-Sensitive ML Re-Entry Model ({prob_reentry3*100:.1f}% < 35.0% threshold)"
                        }
                    else:
                        t3_stopped = False; t3_exit_bar = None; t3_exit_price = None
                        for b in range(reentry_bar_3 + 1, len(fwd)):
                            c = float(fwd["close"].iloc[b]); l = float(fwd["low"].iloc[b]); s = fwd["ribbon_state"].iloc[b]
                            if l <= swing5_3:
                                t3_stopped = True; t3_exit_bar = b; t3_exit_price = swing5_3; break
                            if pd.notna(s) and s in ["blue", "gray"]:
                                t3_exit_bar = b; t3_exit_price = c; break
                        if t3_exit_bar is None:
                            t3_exit_bar = len(fwd) - 1; t3_exit_price = float(fwd["close"].iloc[-1])
                        t3_ret = (t3_exit_price / t3_entry - 1.0) * 100.0 if not t3_stopped else -t3_risk_pct
                        t3_r = t3_ret / t3_risk_pct if t3_risk_pct > 0 else 0.0
                        trade_3 = {
                            "has_reentry": True,
                            "entry_date": fwd.index[reentry_bar_3].strftime("%Y-%m-%d"),
                            "entry_price": round(t3_entry, 2),
                            "stop_price": round(swing5_3, 2),
                            "risk_pct": round(t3_risk_pct, 1),
                            "cons_days": int(cons_days_3),
                            "drop_from_peak": round(drop_3, 1),
                            "exit_date": fwd.index[t3_exit_bar].strftime("%Y-%m-%d"),
                            "exit_price": round(t3_exit_price, 2),
                            "exit_reason": "Larsson Exit" if not t3_stopped else "Stop Loss Hit",
                            "stopped_out": bool(t3_stopped),
                            "hold_days": int(t3_exit_bar - reentry_bar_3),
                            "return_pct": round(t3_ret, 1),
                            "r_mult": round(t3_r, 2),
                            "combined_net_r": round(t1_r + t2_r + t3_r, 2),
                            "prob_reentry": round(prob_reentry3, 3) if prob_reentry3 is not None else None
                        }

    
    ai_scores = {}
    if ev_row:
        ai_scores["ml_50"] = float(ev_row.get("ml_50", 0)) if pd.notna(ev_row.get("ml_50")) else None
        ai_scores["ml_100"] = float(ev_row.get("ml_100", 0)) if pd.notna(ev_row.get("ml_100")) else None
        ai_scores["ml_150"] = float(ev_row.get("ml_150", 0)) if pd.notna(ev_row.get("ml_150")) else None
        ai_scores["ml_200"] = float(ev_row.get("ml_200", 0)) if pd.notna(ev_row.get("ml_200")) else None
        ai_scores["is_toxic"] = bool(ev_row.get("is_toxic", False))
        ai_scores["dynamic_stop_loss_pct"] = float(ev_row.get("dynamic_stop_loss_pct", 0.0)) if pd.notna(ev_row.get("dynamic_stop_loss_pct")) else None

    return {
        "ai_scores": ai_scores,
        "sector_theme": {
            "sector": sec, "theme": thm,
            "sec_m1_pctile": round(sec_m1, 1) if sec_m1 is not None else None,
            "sec_m3_pctile": round(sec_m3, 1) if sec_m3 is not None else None,
            "thm_m1_pctile": round(thm_m1, 1) if thm_m1 is not None else None,
            "macro_diag": macro_diag, "macro_desc": macro_desc, "macro_status": macro_status
        },
        "windows": {"w1": w1, "w2": w2, "w3": w3, "w4": w4},
        "trade_1": trade_1,
        "trade_2": trade_2,
        "trade_3": trade_3
    }

import ep_predictor

@app.route("/api/predict_history")
def api_predict_history():
    sym = request.args.get("sym")
    date_str = request.args.get("date")
    
    df = get_df()
    if df.empty:
        return jsonify({"error": "No data"})
        
    row = df[(df["symbol"] == sym) & (df["date"] == date_str)]
    if row.empty:
        return jsonify({"error": "Event not found"})
        
    features = row.iloc[0].fillna(0).to_dict()
    
    # Generate a mocked 20-day historical evolution for the UI
    # In a real rigorous historical sim, we'd step through the actual OHLC bars and check if 5D held, etc.
    # Here we can just simulate the progression based on the static data we know (e.g. if it held 5d, we activate it on day 5).
    
    history = []
    for day in range(1, 21):
        f_copy = dict(features)
        f_copy["days_since_ep"] = day
        # Dynamically set breached_d1_low_5d for the simulation
        if day < 5:
            # We don't know if it holds 5D yet until day 5
            f_copy["breached_d1_low_5d"] = True 
        else:
            # Day 5 and beyond, we use the actual historical outcome
            pass 
            
        pred = ep_predictor.get_predictions(f_copy)
        
        history.append({
            "day": f"Day {day}",
            "archetype": pred["archetype"],
            "prob_100": pred["prob_hit_100"],
            "prob_consol": pred["prob_consolidation"],
            "alerts": pred["alerts"]
        })
        
    return jsonify({"history": history})


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
    date_str = request.args.get("date", "")
    if not sym or not date_str:
        return jsonify({"error": "Missing symbol or date"}), 400

    raw = datastore.load_bars(sym)
    if raw is None or len(raw) == 0:
        return jsonify({"error": "No bars for symbol"}), 404

    # Run indicators and full-series Larsson calculation
    d = indicators.add_indicators(raw)
    d = scanner_core.calc_larssson_line(d)
    if d is None or len(d) == 0:
        return jsonify({"error": "Failed to enrich bars"}), 500

    # Locate event date
    ts = pd.Timestamp(date_str)
    if ts not in d.index:
        pos = d.index.searchsorted(ts)
        if pos >= len(d):
            pos = len(d) - 1
    else:
        pos = d.index.get_loc(ts)

    lo = max(0, pos - 40)
    hi = len(d)  # Load all historical bars up to present day so user can scroll right
    sub = d.iloc[lo:hi]

    bars = []
    volume = []
    ema20 = []
    sma50 = []

    for idx, r in sub.iterrows():
        t = idx.strftime("%Y-%m-%d")
        o, h, l, c = float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])
        v = float(r["volume"]) if pd.notna(r["volume"]) else 0
        bars.append({"time": t, "open": o, "high": h, "low": l, "close": c})
        volume.append({"time": t, "value": v, "color": "rgba(16,185,129,0.3)" if c >= o else "rgba(239,68,68,0.3)"})
        if "ema20" in r and pd.notna(r["ema20"]):
            ema20.append({"time": t, "value": round(float(r["ema20"]), 2)})
        if "sma50" in r and pd.notna(r["sma50"]):
            sma50.append({"time": t, "value": round(float(r["sma50"]), 2)})

    ev_row_bar = d.iloc[pos]
    level = round(float(ev_row_bar["low"]), 2)
    trigger = round(float(ev_row_bar["high"]), 2)

    df = get_df()
    ev_row_dict = None
    ticker_history = []
    if not df.empty:
        matches = df[(df["symbol"] == sym) & (df["date"] == date_str)]
        if not matches.empty:
            ev_row_dict = matches.iloc[0].to_dict()

        # Extract all historical valid EP events for this ticker across history
        t_matches = df[df["symbol"] == sym].sort_values("date", ascending=False)
        for _, h_row in t_matches.iterrows():
            ticker_history.append({
                "date": str(h_row["date"]),
                "subtype": str(h_row.get("subtype", "ep")),
                "gap_pct": round(float(h_row.get("gap_pct", 0.0)), 1),
                "rvol": round(float(h_row.get("rvol", 0.0)), 1),
                "max_gain": round(float(h_row.get("max_gain", 0.0)), 1) if pd.notna(h_row.get("max_gain")) else None,
                "ret_20d": round(float(h_row.get("ret_20d", 0.0)), 1) if pd.notna(h_row.get("ret_20d")) else 0.0,
                "outcome": str(h_row.get("outcome", "")),
                "violated_48h": bool(h_row.get("violated_48h", False))
            })

    def clean_json(obj):
        if isinstance(obj, dict):
            return {k: clean_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [clean_json(v) for v in obj]
        elif isinstance(obj, (np.floating, float)):
            return None if (pd.isna(obj) or np.isneginf(obj) or np.isposinf(obj)) else round(float(obj), 2)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return obj

    use_trade1 = request.args.get("use_trade1", "1") in ["1", "true", "True"]
    use_ml_targets = request.args.get("use_ml_targets", "1") in ["1", "true", "True"]
    use_ml_stop = request.args.get("use_ml_stop", "1") in ["1", "true", "True"]
    use_multileg = request.args.get("use_multileg", "1") in ["1", "true", "True"]
    use_multileg_ml = request.args.get("use_multileg_ml", "1") in ["1", "true", "True"]

    dossier = compute_dossier(sym, date_str, d, pos, ev_row_dict, use_ml_targets, use_ml_stop, use_multileg, use_multileg_ml, use_trade1)

    res = {
        "symbol": sym,
        "date": date_str,
        "bars": bars,
        "volume": volume,
        "ema20": ema20,
        "sma50": sma50,
        "level": level,
        "trigger": trigger,
        "dossier": dossier,
        "ticker_history": ticker_history
    }
    return jsonify(clean_json(res))

def main():
    parser = argparse.ArgumentParser(description="EP Review Server")
    parser.add_argument("--port", type=int, default=8782, help="Port to listen on (default: 8782)")
    args = parser.parse_args()

    get_df()
    print("=======================================================")
    print(f"🚀 EP Review Server running at: http://127.0.0.1:{args.port}")
    print("=======================================================")
    app.run(host="127.0.0.1", port=args.port, debug=False)

if __name__ == "__main__":
    main()
