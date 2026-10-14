#!/usr/bin/env python3
"""
serve_qm_swing_explorer.py — Interactive Multi-Timeframe QM Swing & Pole Explorer (Port 8788).
Allows visual inspection of:
  - Linear Regression (R^2) & Kaufman Efficiency Ratio (ER) Bull Flag Pole Detection (Recommended)
  - Higher-low sequence swing up detection across 1D, 2D, 3D, 1W, 2W
  - Multi-setup selection with color-coded enclosing bounding rectangles
  - Interactive parameter tuning (R^2, ER, ATR fraction, min move %, min consolidation)
Leaves all production files untouched.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

import config
import datastore
import indicators
import qm_swing_detector

PORT = 8788
logger = logging.getLogger("qm_swing_explorer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>QM Swing & Linear Pole Explorer</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root {
    --bg-main: #0c0f17;
    --bg-card: #151a24;
    --bg-card-hover: #1e2534;
    --border: #232c3d;
    --border-bright: #3a4760;
    --text-primary: #f1f5f9;
    --text-muted: #8b9bb4;
    --accent-blue: #38bdf8;
    --accent-green: #22c55e;
    --accent-amber: #f59e0b;
    --accent-purple: #c084fc;
    --accent-red: #ef4444;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background-color: var(--bg-main);
    color: var(--text-primary);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  header {
    background: var(--bg-card);
    border-bottom: 1px solid var(--border);
    padding: 10px 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
    z-index: 10;
  }
  .title-group {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .badge-exp {
    background: rgba(56, 189, 248, 0.15);
    color: var(--accent-blue);
    border: 1px solid rgba(56, 189, 248, 0.35);
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .search-box {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  input[type="text"], input[type="number"], select {
    background: #090c12;
    border: 1px solid var(--border-bright);
    color: var(--text-primary);
    padding: 5px 9px;
    border-radius: 5px;
    font-size: 12px;
    font-weight: 600;
    outline: none;
  }
  input[type="text"]:focus, input[type="number"]:focus, select:focus {
    border-color: var(--accent-blue);
  }
  .quick-pills {
    display: flex;
    gap: 5px;
  }
  .pill {
    background: #0d121c;
    border: 1px solid var(--border);
    color: var(--text-muted);
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 11px;
    cursor: pointer;
    font-weight: 600;
    transition: all 0.15s;
  }
  .pill:hover, .pill.active {
    background: rgba(56, 189, 248, 0.18);
    color: var(--accent-blue);
    border-color: var(--accent-blue);
  }
  .controls-bar {
    background: #111621;
    border-bottom: 1px solid var(--border);
    padding: 8px 18px;
    display: flex;
    align-items: center;
    gap: 14px;
    font-size: 12px;
    color: var(--text-muted);
    flex-wrap: wrap;
  }
  .ctrl-item {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .ctrl-item label {
    font-weight: 600;
    color: var(--text-muted);
  }
  .btn-scan {
    background: #0284c7;
    color: #fff;
    border: none;
    padding: 5px 14px;
    border-radius: 5px;
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
    transition: background 0.15s;
  }
  .btn-scan:hover { background: #0369a1; }
  .btn-select-all {
    background: #1e293b;
    border: 1px solid var(--border-bright);
    color: var(--text-primary);
    padding: 3px 9px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
  }
  .btn-select-all:hover {
    background: #334155;
    border-color: var(--accent-blue);
  }
  input[type="checkbox"] {
    accent-color: var(--accent-blue);
    cursor: pointer;
    width: 14px;
    height: 14px;
  }
  .workspace {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 0;
  }
  .chart-pane {
    flex: 3;
    position: relative;
    border-bottom: 1px solid var(--border);
    min-height: 250px;
    background: #090c12;
  }
  #chart-container {
    width: 100%;
    height: 100%;
  }
  #chart_canvas_overlay {
    position: absolute;
    inset: 0;
    pointer-events: none;
    z-index: 6;
  }
  .chart-legend {
    position: absolute;
    top: 10px;
    left: 14px;
    z-index: 5;
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid var(--border);
    padding: 6px 10px;
    border-radius: 6px;
    font-family: monospace;
    font-size: 12px;
    pointer-events: none;
    line-height: 1.5;
  }
  .table-pane {
    flex: 2;
    background: var(--bg-card);
    display: flex;
    flex-direction: column;
    min-height: 180px;
    overflow: hidden;
  }
  .table-header-bar {
    padding: 6px 14px;
    background: #111621;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-weight: 600;
    font-size: 12px;
  }
  .table-scroll {
    flex: 1;
    overflow-y: auto;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    text-align: left;
    font-size: 12px;
  }
  th {
    background: #111621;
    color: var(--text-muted);
    font-weight: 600;
    padding: 7px 10px;
    position: sticky;
    top: 0;
    border-bottom: 1px solid var(--border);
    z-index: 2;
  }
  td {
    padding: 6px 10px;
    border-bottom: 1px solid #1a2230;
    color: var(--text-primary);
  }
  tr:hover td {
    background: var(--bg-card-hover);
    cursor: pointer;
  }
  tr.active-setup td {
    background: rgba(56, 189, 248, 0.15);
  }
  .tag {
    display: inline-block;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
  }
  .tag-breakout { background: rgba(34, 197, 94, 0.2); color: var(--accent-green); border: 1px solid rgba(34, 197, 94, 0.4); }
  .tag-building { background: rgba(245, 158, 11, 0.2); color: var(--accent-amber); border: 1px solid rgba(245, 158, 11, 0.4); }
  .tag-linear { background: rgba(56, 189, 248, 0.2); color: var(--accent-blue); border: 1px solid rgba(56, 189, 248, 0.4); }
  .tag-tf { background: #1e293b; color: #cbd5e1; border: 1px solid #334155; }
  .color-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 5px;
    vertical-align: middle;
  }
</style>
</head>
<body>
  <header>
    <div class="title-group">
      <span style="font-weight:700; font-size:15px; color:#fff;">QM Swing & Linear Pole Explorer</span>
      <span class="badge-exp">Experimental</span>
    </div>
    <div class="search-box">
      <input type="text" id="sym-input" value="GOOGL" style="width: 85px; text-transform: uppercase;" />
      <button class="btn-scan" onclick="triggerScan()">Scan</button>
      <div class="quick-pills">
        <span class="pill active" onclick="setTicker('GOOGL')">GOOGL</span>
        <span class="pill" onclick="setTicker('NVDA')">NVDA</span>
        <span class="pill" onclick="setTicker('TSLA')">TSLA</span>
        <span class="pill" onclick="setTicker('PLTR')">PLTR</span>
        <span class="pill" onclick="setTicker('META')">META</span>
        <span class="pill" onclick="setTicker('AMD')">AMD</span>
        <span class="pill" onclick="setTicker('SMCI')">SMCI</span>
      </div>
    </div>
  </header>

  <div class="controls-bar">
    <div class="ctrl-item">
      <label>Algo:</label>
      <select id="algo-select" onchange="onAlgoChange()">
        <option value="original_qm">Original QM Breakout (setups.py)</option>
        <option value="linear_regression">Linear Regression R² + ER (Causal)</option>
        <option value="ema_catchup_bb_pinch" selected>EMA Catchup & BB Pinch (Squeeze)</option>
        <option value="higher_low">Higher-Low Sequence (Type A / B)</option>
      </select>
    </div>

    <div class="ctrl-item">
      <label>Timeframes:</label>
      <div class="quick-pills" id="tf-pills">
        <span class="pill active" data-tf="All" onclick="selectTF('All')">All</span>
        <span class="pill" data-tf="1D" onclick="selectTF('1D')">1D</span>
        <span class="pill" data-tf="2D" onclick="selectTF('2D')">2D</span>
        <span class="pill" data-tf="3D" onclick="selectTF('3D')">3D</span>
        <span class="pill" data-tf="1W" onclick="selectTF('1W')">1W</span>
      </div>
    </div>

    <div class="ctrl-item" id="ext10-ctrl">
      <label>Min Ext 10 %:</label>
      <input type="number" id="min-ext10" value="3.0" step="0.5" min="1.0" max="15.0" style="width:55px;">
    </div>
    <div class="ctrl-item" id="ext20-ctrl">
      <label>Min Ext 20 %:</label>
      <input type="number" id="min-ext20" value="5.0" step="0.5" min="2.0" max="25.0" style="width:55px;">
    </div>
    <div class="ctrl-item" id="pinch-ctrl">
      <label>BB/ADR Pinch Ratio:</label>
      <input type="number" id="max-pinch-ratio" value="3.2" step="0.1" min="1.5" max="5.5" style="width:55px;">
    </div>

    <div class="ctrl-item" id="r2-ctrl" style="display:none;">
      <label>Min R²:</label>
      <input type="number" id="min-r2" value="0.65" step="0.05" min="0.4" max="0.95" style="width:55px;">
    </div>
    <div class="ctrl-item" id="er-ctrl" style="display:none;">
      <label>Min ER:</label>
      <input type="number" id="min-er" value="0.40" step="0.05" min="0.2" max="0.85" style="width:55px;">
    </div>

    <div class="ctrl-item" id="atr-ctrl" style="display:none;">
      <label>ATR Frac:</label>
      <input type="number" id="atr-frac" value="0.3" step="0.05" min="0.1" max="0.8" style="width:55px;">
    </div>

    <div class="ctrl-item">
      <label>Min Move %:</label>
      <input type="number" id="min-move" value="15" step="5" min="10" max="60" style="width:55px;">
    </div>
    <div class="ctrl-item" id="cons-ctrl" style="display:none;">
      <label>Min Cons (b):</label>
      <input type="number" id="min-cons" value="3" step="1" min="2" max="10" style="width:50px;">
    </div>
  </div>

  <div class="workspace">
    <div class="chart-pane">
      <div class="chart-legend" id="legend">Loading chart...</div>
      <div id="chart-container"></div>
      <canvas id="chart_canvas_overlay"></canvas>
    </div>

    <div class="table-pane">
      <div class="table-header-bar">
        <div style="display:flex; align-items:center; gap:10px;">
          <button class="btn-select-all" id="btn-toggle-all" onclick="toggleSelectAll()">Select All</button>
          <button class="btn-select-all" onclick="clearAllSelection()">Clear Selection</button>
          <span id="results-count" style="margin-left:6px;">Detected Setups</span>
        </div>
        <span style="font-size:11px; color:var(--text-muted);">Check setups to enclose with rectangles on chart. Click row to center.</span>
      </div>
      <div class="table-scroll">
        <table>
          <thead>
            <tr>
              <th style="width:36px; text-align:center;">
                <input type="checkbox" id="header-select-all" onchange="onHeaderCheckboxChange(this.checked)" title="Select/Deselect All">
              </th>
              <th>#</th>
              <th>TF</th>
              <th>Date</th>
              <th>State</th>
              <th>Pole Range (Dates)</th>
              <th>Move %</th>
              <th id="th-metric">Ext 10 / 20</th>
              <th id="th-extra">BB/ADR Pinch</th>
              <th>Swing Bars</th>
              <th>Cons Bars</th>
              <th>Pullback %</th>
              <th>Trigger</th>
              <th>Vol X</th>
              <th>Ride Frac</th>
            </tr>
          </thead>
          <tbody id="setups-body">
            <tr><td colspan="15" style="text-align:center; color:var(--text-muted); padding:20px;">Scanning...</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

<script>
let currentSymbol = "GOOGL";
let currentTF = "All";
let chart = null;
let candleSeries = null;
let ema10Series = null;
let ema20Series = null;
let sma50Series = null;
let triggerLine = null;
let chartData = [];
let allHits = [];
let selectedSetupIndices = new Set();
let activeSetupIdx = null;

const SETUP_PALETTE = [
  { stroke: '#38bdf8', fill: 'rgba(56, 189, 248, 0.12)', labelBg: 'rgba(56, 189, 248, 0.90)', text: '#000', name: 'Cyan' },
  { stroke: '#22c55e', fill: 'rgba(34, 197, 94, 0.12)', labelBg: 'rgba(34, 197, 94, 0.90)', text: '#000', name: 'Green' },
  { stroke: '#f59e0b', fill: 'rgba(245, 158, 11, 0.12)', labelBg: 'rgba(245, 158, 11, 0.90)', text: '#000', name: 'Amber' },
  { stroke: '#c084fc', fill: 'rgba(192, 132, 252, 0.12)', labelBg: 'rgba(192, 132, 252, 0.90)', text: '#000', name: 'Purple' },
  { stroke: '#f43f5e', fill: 'rgba(244, 63, 94, 0.12)', labelBg: 'rgba(244, 63, 94, 0.90)', text: '#000', name: 'Rose' },
  { stroke: '#14b8a6', fill: 'rgba(20, 184, 166, 0.12)', labelBg: 'rgba(20, 184, 166, 0.90)', text: '#000', name: 'Teal' },
  { stroke: '#e879f9', fill: 'rgba(232, 121, 249, 0.12)', labelBg: 'rgba(232, 121, 249, 0.90)', text: '#000', name: 'Fuchsia' },
  { stroke: '#fb923c', fill: 'rgba(251, 146, 60, 0.12)', labelBg: 'rgba(251, 146, 60, 0.90)', text: '#000', name: 'Orange' },
];

function onAlgoChange() {
  const algo = document.getElementById("algo-select").value;
  const isOrig = algo === "original_qm";
  const isLin = algo === "linear_regression";
  const isEMA = algo === "ema_catchup_bb_pinch";
  const isHL = algo === "higher_low";

  document.getElementById("r2-ctrl").style.display = isLin ? "flex" : "none";
  document.getElementById("er-ctrl").style.display = isLin ? "flex" : "none";
  document.getElementById("cons-ctrl").style.display = (isOrig || isLin || isHL) ? "flex" : "none";

  document.getElementById("ext10-ctrl").style.display = isEMA ? "flex" : "none";
  document.getElementById("ext20-ctrl").style.display = isEMA ? "flex" : "none";
  document.getElementById("pinch-ctrl").style.display = (isOrig || isEMA) ? "flex" : "none";

  document.getElementById("atr-ctrl").style.display = isHL ? "flex" : "none";

  updateTableHeaders();
  triggerScan();
}

function updateTableHeaders() {
  const algo = document.getElementById("algo-select").value;
  const thMetric = document.getElementById("th-metric");
  const thExtra = document.getElementById("th-extra");
  if (algo === "original_qm") {
    if (thMetric) thMetric.innerText = "Pattern";
    if (thExtra) thExtra.innerText = "BB/ADR Pinch";
  } else if (algo === "ema_catchup_bb_pinch") {
    if (thMetric) thMetric.innerText = "Ext 10 / 20";
    if (thExtra) thExtra.innerText = "BB/ADR Pinch";
  } else if (algo === "linear_regression") {
    if (thMetric) thMetric.innerText = "R²";
    if (thExtra) thExtra.innerText = "ER";
  } else {
    if (thMetric) thMetric.innerText = "Swing Type";
    if (thExtra) thExtra.innerText = "ATR Frac";
  }
}

function initChart() {
  const container = document.getElementById("chart-container");
  chart = LightweightCharts.createChart(container, {
    layout: {
      background: { color: "#090c12" },
      textColor: "#94a3b8",
    },
    grid: {
      vertLines: { color: "#141c2b" },
      horzLines: { color: "#141c2b" },
    },
    crosshair: {
      mode: LightweightCharts.CrosshairMode.Normal,
    },
    rightPriceScale: {
      borderColor: "#232c3d",
      autoScale: true,
    },
    timeScale: {
      borderColor: "#232c3d",
      visible: true,
      borderVisible: true,
      timeVisible: true,
      secondsVisible: false,
    },
  });

  candleSeries = chart.addCandlestickSeries({
    upColor: "#22c55e",
    downColor: "#ef4444",
    borderVisible: false,
    wickUpColor: "#22c55e",
    wickDownColor: "#ef4444",
  });

  ema10Series = chart.addLineSeries({ color: "#38bdf8", lineWidth: 1.5, title: "10 EMA" });
  ema20Series = chart.addLineSeries({ color: "#f59e0b", lineWidth: 1.5, title: "20 EMA" });
  sma50Series = chart.addLineSeries({ color: "#c084fc", lineWidth: 1.5, title: "50 SMA" });

  const ro = new ResizeObserver(() => {
    if (chart && container) {
      chart.applyOptions({
        width: container.clientWidth,
        height: container.clientHeight
      });
      drawCanvasOverlays();
    }
  });
  ro.observe(container);

  chart.timeScale().subscribeVisibleLogicalRangeChange(() => {
    drawCanvasOverlays();
  });
}

async function loadChartData(sym) {
  try {
    const res = await fetch(`/api/chart?symbol=${sym}`);
    const d = await res.json();
    chartData = d.bars || [];
    candleSeries.setData(chartData);
    ema10Series.setData(d.ema10 || []);
    ema20Series.setData(d.ema20 || []);
    sma50Series.setData(d.sma50 || []);
    chart.timeScale().fitContent();

    const last = chartData[chartData.length - 1];
    if (last) {
      updateLegend(sym, last.close, last.time);
    }
  } catch (err) {
    console.error("Error loading chart:", err);
  }
}

function updateLegend(sym, price, time) {
  document.getElementById("legend").innerHTML = `
    <span style="font-weight:700; color:#fff; font-size:13px;">${sym}</span> &nbsp;
    <span>Close: <b style="color:#fff;">${price ? price.toFixed(2) : '-'}</b></span> &nbsp;
    <span style="color:#8b9bb4;">Date: ${time}</span>
  `;
}

async function triggerScan() {
  const sym = document.getElementById("sym-input").value.trim().toUpperCase() || "GOOGL";
  currentSymbol = sym;
  const algo = document.getElementById("algo-select").value;
  const minR2 = document.getElementById("min-r2").value;
  const minER = document.getElementById("min-er").value;
  const atrFrac = document.getElementById("atr-frac").value;
  const minMove = document.getElementById("min-move").value;
  const minCons = document.getElementById("min-cons").value;

  const minExt10 = document.getElementById("min-ext10") ? document.getElementById("min-ext10").value : 3.0;
  const minExt20 = document.getElementById("min-ext20") ? document.getElementById("min-ext20").value : 5.0;
  const maxPinch = document.getElementById("max-pinch-ratio") ? document.getElementById("max-pinch-ratio").value : 3.2;

  let tfs = currentTF === "All" ? "1D,2D,3D,1W" : currentTF;

  document.getElementById("setups-body").innerHTML = `<tr><td colspan="15" style="text-align:center; color:var(--text-muted); padding:20px;">Scanning ${sym} across ${tfs}...</td></tr>`;

  await loadChartData(sym);

  try {
    const url = `/api/scan?symbol=${sym}&algo=${algo}&tf=${tfs}&min_r2=${minR2}&min_er=${minER}&atr_frac=${atrFrac}&min_move=${minMove}&min_cons=${minCons}&min_ext_10=${minExt10}&min_ext_20=${minExt20}&max_pinch_ratio=${maxPinch}`;
    const res = await fetch(url);
    const data = await res.json();
    allHits = data.hits || [];
    selectedSetupIndices.clear();
    // Default: select the first 2 setups
    if (allHits.length > 0) {
      selectedSetupIndices.add(0);
      if (allHits.length > 1) selectedSetupIndices.add(1);
    }
    renderTable(allHits);
    drawCanvasOverlays();
  } catch (err) {
    console.error("Scan error:", err);
    document.getElementById("setups-body").innerHTML = `<tr><td colspan="15" style="text-align:center; color:var(--accent-red); padding:20px;">Error running scan.</td></tr>`;
  }
}

function renderTable(hits) {
  document.getElementById("results-count").innerText = `${currentSymbol}: ${hits.length} Setups Detected`;
  const tbody = document.getElementById("setups-body");
  if (!hits.length) {
    tbody.innerHTML = `<tr><td colspan="15" style="text-align:center; color:var(--text-muted); padding:20px;">No setups found matching criteria.</td></tr>`;
    updateHeaderCheckbox();
    return;
  }

  tbody.innerHTML = hits.map((h, idx) => {
    let stBadge;
    if (h.state === "breakout") {
      stBadge = `<span class="tag tag-breakout">Breakout</span>`;
    } else if (h.state === "primed") {
      stBadge = `<span class="tag tag-building" style="background:rgba(245,158,11,0.18); color:var(--accent-amber); border:1px solid rgba(245,158,11,0.4);">Primed</span>`;
    } else {
      stBadge = `<span class="tag tag-building">Building</span>`;
    }
    const volStr = h.vol_x ? `${h.vol_x}x` : '-';
    const isChecked = selectedSetupIndices.has(idx) ? 'checked' : '';
    const pal = SETUP_PALETTE[idx % SETUP_PALETTE.length];

    let metricCell = '';
    let extraCell = '';
    if (h.algo === "Original QM" || h.algo === "original_qm") {
      metricCell = `<span style="font-weight:700; color:var(--accent-blue); text-transform:capitalize;">${h.pattern || 'Consolidation'}</span>`;
      extraCell = `<span style="font-weight:700; color:var(--accent-amber);">${h.pinch_ratio ? h.pinch_ratio + 'x' : '-'}</span>`;
    } else if (h.algo === "EMA Catchup & BB Pinch") {
      metricCell = `<span style="font-weight:700; color:var(--accent-blue);">+${h.ext10}%</span> / <span style="font-weight:700; color:var(--accent-purple);">+${h.ext20}%</span>`;
      extraCell = `<span style="font-weight:700; color:var(--accent-amber);">${h.pinch_ratio ? h.pinch_ratio + 'x' : '-'}</span>`;
    } else if (h.r2 !== undefined && h.r2 !== null) {
      metricCell = `<span style="font-weight:700; color:var(--accent-blue);">${h.r2}</span>`;
      extraCell = `<span style="font-weight:700; color:var(--accent-purple);">${h.er}</span>`;
    } else {
      metricCell = `<span>${h.pole_type || '-'}</span>`;
      extraCell = `<span>-</span>`;
    }

    return `
      <tr onclick="onRowClick(${idx})" id="row-${idx}" class="${activeSetupIdx === idx ? 'active-setup' : ''}">
        <td style="text-align:center;" onclick="event.stopPropagation()">
          <input type="checkbox" class="setup-cb" id="cb-${idx}" data-idx="${idx}" ${isChecked} onchange="onRowCheckboxChange(${idx}, this.checked)">
        </td>
        <td style="font-weight:700; color:${pal.stroke};"><span class="color-dot" style="background:${pal.stroke};"></span>#${idx+1}</td>
        <td><span class="tag tag-tf">${h.tf}</span></td>
        <td style="font-weight:600;">${h.date}</td>
        <td>${stBadge}</td>
        <td>${h.pole_start_date} &rarr; ${h.pole_end_date}</td>
        <td style="font-weight:700; color:var(--accent-green);">+${h.move_pct}%</td>
        <td>${metricCell}</td>
        <td>${extraCell}</td>
        <td>${h.bars_in_swing}b</td>
        <td>${h.cons_bars}b</td>
        <td style="color:${h.pullback_pct > 15 ? 'var(--accent-amber)' : 'inherit'};">${h.pullback_pct}%</td>
        <td style="font-weight:700; color:var(--accent-blue);">${h.trigger}</td>
        <td>${volStr}</td>
        <td>${(h.ride_frac * 100).toFixed(0)}%</td>
      </tr>
    `;
  }).join("");

  updateHeaderCheckbox();
  if (hits.length > 0) {
    selectSetup(0, false);
  }
}

function updateHeaderCheckbox() {
  const hcb = document.getElementById("header-select-all");
  if (!hcb) return;
  if (allHits.length === 0) {
    hcb.checked = false;
    hcb.indeterminate = false;
  } else if (selectedSetupIndices.size === allHits.length) {
    hcb.checked = true;
    hcb.indeterminate = false;
  } else if (selectedSetupIndices.size === 0) {
    hcb.checked = false;
    hcb.indeterminate = false;
  } else {
    hcb.checked = false;
    hcb.indeterminate = true;
  }
}

function onHeaderCheckboxChange(checked) {
  if (checked) {
    allHits.forEach((_, idx) => selectedSetupIndices.add(idx));
  } else {
    selectedSetupIndices.clear();
  }
  document.querySelectorAll(".setup-cb").forEach(cb => {
    cb.checked = checked;
  });
  updateHeaderCheckbox();
  drawCanvasOverlays();
}

function toggleSelectAll() {
  const allSelected = selectedSetupIndices.size === allHits.length;
  onHeaderCheckboxChange(!allSelected);
}

function clearAllSelection() {
  onHeaderCheckboxChange(false);
}

function onRowCheckboxChange(idx, checked) {
  if (checked) {
    selectedSetupIndices.add(idx);
  } else {
    selectedSetupIndices.delete(idx);
  }
  updateHeaderCheckbox();
  drawCanvasOverlays();
}

function onRowClick(idx) {
  selectSetup(idx, true);
}

function selectSetup(idx, centerChart = true) {
  activeSetupIdx = idx;
  document.querySelectorAll("#setups-body tr").forEach(r => r.classList.remove("active-setup"));
  const row = document.getElementById(`row-${idx}`);
  if (row) row.classList.add("active-setup");

  const h = allHits[idx];
  if (!h || !chart) return;

  if (triggerLine) {
    candleSeries.removePriceLine(triggerLine);
  }
  triggerLine = candleSeries.createPriceLine({
    price: h.trigger,
    color: h.state === "breakout" ? "#22c55e" : "#38bdf8",
    lineWidth: 2,
    lineStyle: LightweightCharts.LineStyle.Dashed,
    axisLabelVisible: true,
    title: `#${idx+1} Trigger ${h.trigger}`,
  });

  if (centerChart) {
    chart.timeScale().setVisibleRange({
      from: h.pole_start_date,
      to: h.date,
    });
  }
  drawCanvasOverlays();
}

function drawCanvasOverlays() {
  const canvas = document.getElementById("chart_canvas_overlay");
  if (!canvas || !chart || !candleSeries) return;
  const parent = canvas.parentElement;
  if (!parent) return;

  const w = canvas.width = parent.clientWidth;
  const h = canvas.height = parent.clientHeight;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);

  if (selectedSetupIndices.size === 0 || allHits.length === 0) return;

  const timeScale = chart.timeScale();

  selectedSetupIndices.forEach(idx => {
    const setup = allHits[idx];
    if (!setup) return;

    const pal = SETUP_PALETTE[idx % SETUP_PALETTE.length];

    const xStart = timeScale.timeToCoordinate(setup.pole_start_date);
    const xPoleEnd = timeScale.timeToCoordinate(setup.pole_end_date);
    const xEnd = timeScale.timeToCoordinate(setup.date || setup.cons_end_date);

    const highPrice = Math.max(setup.pole_high, setup.trigger || setup.pole_high);
    const lowPrice = Math.min(setup.pole_low, setup.pole_high * (1.0 - (setup.pullback_pct || 0) / 100.0));

    const yTop = candleSeries.priceToCoordinate(highPrice);
    const yBottom = candleSeries.priceToCoordinate(lowPrice);

    if (xStart === null && xEnd === null) return;
    if (yTop === null || yBottom === null) return;

    const effectiveXStart = xStart !== null ? xStart : (xEnd > 0 ? 0 : -999);
    const effectiveXEnd = xEnd !== null ? xEnd : (xStart < w ? w : w + 999);

    const left = Math.min(effectiveXStart, effectiveXEnd);
    const right = Math.max(effectiveXStart, effectiveXEnd);
    const boxW = Math.max(14, right - left);

    const top = Math.min(yTop, yBottom) - 8;
    const bottom = Math.max(yTop, yBottom) + 8;
    const boxH = Math.max(14, bottom - top);

    // 1. Shaded box
    ctx.fillStyle = pal.fill;
    ctx.fillRect(left, top, boxW, boxH);

    // 2. Solid border
    ctx.strokeStyle = pal.stroke;
    ctx.lineWidth = (activeSetupIdx === idx) ? 2.5 : 1.5;
    ctx.strokeRect(left, top, boxW, boxH);

    // 3. Separator line between Pole and Base
    if (xPoleEnd !== null && xPoleEnd > left && xPoleEnd < right) {
      ctx.save();
      ctx.strokeStyle = pal.stroke;
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(xPoleEnd, top);
      ctx.lineTo(xPoleEnd, top + boxH);
      ctx.stroke();
      ctx.restore();

      ctx.fillStyle = pal.stroke;
      ctx.font = '10px -apple-system, sans-serif';
      ctx.fillText('POLE', left + 6, bottom - 6);
      ctx.fillText(`BASE (${setup.cons_bars}b)`, xPoleEnd + 6, bottom - 6);
    }

    // 4. Header Badge Pill
    let metricStr = '';
    if (setup.pinch_ratio) {
      metricStr = ` | Pinch: ${setup.pinch_ratio}x`;
    } else if (setup.r2) {
      metricStr = ` | R²=${setup.r2}`;
    }
    const badgeText = `[#${idx+1}] ${setup.tf} ${setup.state.toUpperCase()} +${setup.move_pct}%${metricStr} (Trig: ${setup.trigger})`;
    ctx.font = 'bold 11px -apple-system, BlinkMacSystemFont, sans-serif';
    const textMetrics = ctx.measureText(badgeText);
    const pillW = textMetrics.width + 12;
    const pillH = 18;
    const pillX = left;
    const pillY = Math.max(2, top - pillH - 2);

    ctx.fillStyle = pal.labelBg;
    ctx.beginPath();
    ctx.roundRect ? ctx.roundRect(pillX, pillY, pillW, pillH, 4) : ctx.rect(pillX, pillY, pillW, pillH);
    ctx.fill();

    ctx.fillStyle = pal.text;
    ctx.fillText(badgeText, pillX + 6, pillY + 13);
  });
}

function setTicker(sym) {
  document.getElementById("sym-input").value = sym;
  document.querySelectorAll(".quick-pills .pill").forEach(p => {
    p.classList.toggle("active", p.innerText === sym);
  });
  triggerScan();
}

function selectTF(tf) {
  currentTF = tf;
  document.querySelectorAll("#tf-pills .pill").forEach(p => {
    p.classList.toggle("active", p.getAttribute("data-tf") === tf);
  });
  triggerScan();
}

window.addEventListener("DOMContentLoaded", () => {
  initChart();
  triggerScan();
});
</script>
</body>
</html>
"""


class SwingExplorerHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        if u.path == "/api/chart":
            qs = parse_qs(u.query)
            sym = qs.get("symbol", ["GOOGL"])[0].upper()
            df = datastore.load_bars(sym)
            if df is None:
                self.send_json({"bars": []})
                return
            df = indicators.add_indicators(df)
            bars = []
            e10 = []
            e20 = []
            s50 = []
            for dt, row in df.iterrows():
                t_str = dt.strftime("%Y-%m-%d")
                bars.append({
                    "time": t_str,
                    "open": round(float(row["open"]), 2),
                    "high": round(float(row["high"]), 2),
                    "low": round(float(row["low"]), 2),
                    "close": round(float(row["close"]), 2),
                    "volume": int(row["volume"]),
                })
                if pd.notna(row.get("ema10")):
                    e10.append({"time": t_str, "value": round(float(row["ema10"]), 2)})
                if pd.notna(row.get("ema20")):
                    e20.append({"time": t_str, "value": round(float(row["ema20"]), 2)})
                if pd.notna(row.get("sma50")):
                    s50.append({"time": t_str, "value": round(float(row["sma50"]), 2)})

            self.send_json({"bars": bars, "ema10": e10, "ema20": e20, "sma50": s50})
            return

        if u.path == "/api/scan":
            qs = parse_qs(u.query)
            sym = qs.get("symbol", ["GOOGL"])[0].upper()
            algo = qs.get("algo", ["ema_catchup_bb_pinch"])[0]
            tf_arg = qs.get("tf", ["1D,2D,3D,1W"])[0]
            tfs = [x.strip() for x in tf_arg.split(",") if x.strip()]
            min_r2 = float(qs.get("min_r2", ["0.65"])[0])
            min_er = float(qs.get("min_er", ["0.40"])[0])
            atr_frac = float(qs.get("atr_frac", ["0.3"])[0])
            min_move = float(qs.get("min_move", ["15.0"])[0])
            min_cons = int(qs.get("min_cons", ["3"])[0])
            min_ext_10 = float(qs.get("min_ext_10", ["3.0"])[0])
            min_ext_20 = float(qs.get("min_ext_20", ["5.0"])[0])
            max_pinch_ratio = float(qs.get("max_pinch_ratio", ["3.2"])[0])

            hits = qm_swing_detector.scan_symbol_multi_tf(
                symbol=sym,
                algo=algo,
                tfs=tfs,
                min_r2=min_r2,
                min_er=min_er,
                atr_frac=atr_frac,
                min_move=min_move,
                min_cons=min_cons,
                min_ext_10=min_ext_10,
                min_ext_20=min_ext_20,
                max_pinch_ratio=max_pinch_ratio
            )
            self.send_json({"symbol": sym, "hits": hits})
            return

        super().do_GET()

    def send_json(self, payload: Any):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    parser = argparse.ArgumentParser(description="Serve QM Swing & Pole Explorer")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port to bind (default: {PORT})")
    args = parser.parse_args()

    server = HTTPServer(("127.0.0.1", args.port), SwingExplorerHandler)
    logger.info(f"QM Swing Explorer running at http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server shutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
