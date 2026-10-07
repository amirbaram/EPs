"""EMA Rider lab — Python port + visual tester of the "EMA Streak" Pine indicator.

Streak = consecutive bars CLOSING above (positive) / below (negative) an EMA. Within a streak
it counts how often the wick came NEAR the EMA (low/high within ATR*atr_frac of it — a wick
*below* the EMA counts for an above-streak) and how often it actually BREACHED (crossed) the
EMA, with the wick depth. Resets on every flip. This is the exact logic of emaRider.pine.

    .venv/bin/python emarider.py NVDA                  # last 500 daily bars from the cache
    .venv/bin/python emarider.py NVDA --ema 20 --frac 0.5 --open
    .venv/bin/python emarider.py NVDA --streak-thresh --streak-frac 0.5
    .venv/bin/python emarider.py path/to/bars.csv      # any OHLC csv

Writes output/emarider_<symbol>.html: candles + EMA line, green/red streak shading (longer =
more opaque), a label at each flip (the streak that just ended), per-bar NEAR (amber) and
BREACH (red) dots, and a click-to-zoom table of every streak so you can verify the counts.
"""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
import trendlab          # reuse load_frame (CLI loader)


def ema(close: np.ndarray, length: int) -> np.ndarray:
    return pd.Series(close).ewm(span=length, adjust=False).mean().to_numpy()


def atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int) -> np.ndarray:
    """Wilder's ATR (RMA of true range) — identical to trendlab.atr_series, but built from
    plain numpy so it never touches the frame's .attrs. (A pd.concat on attrs-bearing Series
    raises once trendlab has cached its array-valued _trendlab dict on the frame.)"""
    pc = np.empty_like(c); pc[0] = np.nan; pc[1:] = c[:-1]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)   # bar 0 -> h-l
    return pd.Series(tr).ewm(alpha=1.0 / n, adjust=False, min_periods=1).mean().to_numpy()


@dataclass
class Streak:
    start: int                 # bar positions (inclusive)
    end: int
    direction: int             # +1 above the EMA, -1 below
    bars: int = 0              # streak length (|streak| at the end)
    near: int = 0              # bars whose wick came within ATR*atr_frac of the EMA (wicks count)
    breach: int = 0            # bars whose wick actually crossed the EMA
    breach_avg: float = 0.0    # mean breach depth
    breach_max: float = 0.0    # deepest breach


def compute(d: pd.DataFrame, ema_len: int, atr_len: int, atr_frac: float,
            use_streak_thresh: bool, streak_atr_frac: float, use_arm_exit: bool = False):
    """Per-bar port of emaRider.pine INCLUDING the use_arm_exit decisive-exit machine
    (emaRider.pine:44-110). Returns (ema, atr, streak[], near[], breach[], depth[], armed[], event[]).

    use_arm_exit: a wrong-side close does NOT flip the streak — it ARMS an exit (arm_level = that bar's
    low above / high below) while the streak keeps counting. It resolves later: if the armed bar's
    extreme is broken -> BREAK (flip to the other side, dated back to the arming bar via streak := ∓arm_run);
    if price closes back on-side first -> SAVED (streak continues). armed[] is the per-bar armed flag;
    event[i] in {'', 'armed', 'saved', 'break'} mirrors the pine's just_armed / just_saved / streak_flip.
    With use_arm_exit=False this is byte-identical to the legacy streak (armed all False, event marks flips)."""
    c = d["close"].to_numpy(float); h = d["high"].to_numpy(float); l = d["low"].to_numpy(float)
    e = ema(c, ema_len); av = atr(h, l, c, atr_len)
    n = len(d)
    streak = np.zeros(n, int)
    near = np.zeros(n, bool)
    breach = np.zeros(n, bool)
    depth = np.full(n, np.nan)
    armed_arr = np.zeros(n, bool)
    event = np.array([""] * n, dtype=object)
    s = 0
    armed = False
    arm_level = np.nan
    arm_run = 0
    for i in range(n):
        a = av[i] if np.isfinite(av[i]) else 0.0
        thr = a * streak_atr_frac if use_streak_thresh else 0.0
        if s > 0:                                       # ── riding above ──
            if armed:
                if l[i] < arm_level:                    # follow-through: armed low broken -> flip down
                    arm_run += 1; s = -arm_run          # date the down streak back to the arming bar
                    armed = False; arm_level = np.nan
                else:
                    s += 1                               # keep counting while armed
                    if c[i] > e[i] - thr:               # saved: closed back above
                        armed = False; arm_level = np.nan; arm_run = 0
                    else:
                        arm_run += 1                     # still below, stay armed
            elif c[i] > e[i] - thr:
                s += 1                                   # normal continuation
            elif use_arm_exit:
                s += 1; armed = True; arm_level = l[i]; arm_run = 1   # ride alive, arm the exit
            else:
                s = -1                                   # legacy: flip on first wrong close
        elif s < 0:                                     # ── riding below (mirror) ──
            if armed:
                if h[i] > arm_level:
                    arm_run += 1; s = arm_run
                    armed = False; arm_level = np.nan
                else:
                    s -= 1
                    if c[i] < e[i] + thr:
                        armed = False; arm_level = np.nan; arm_run = 0
                    else:
                        arm_run += 1
            elif c[i] < e[i] + thr:
                s -= 1
            elif use_arm_exit:
                s -= 1; armed = True; arm_level = h[i]; arm_run = 1
            else:
                s = 1
        else:                                           # ── s == 0 (first established bar) ──
            s = 1 if c[i] > e[i] - thr else -1
        streak[i] = s
        armed_arr[i] = armed
        if i > 0:                                       # events: flip first, then armed, then saved
            ps = streak[i - 1]
            if ps != 0 and (s > 0) != (ps > 0):
                event[i] = "break"
            elif armed and not armed_arr[i - 1]:
                event[i] = "armed"
            elif (not armed) and armed_arr[i - 1] and (s > 0) == (ps > 0):
                event[i] = "saved"
        wt = a * atr_frac
        if s > 1:                                       # above-streak (2nd bar on) — watch the LOW
            if l[i] < e[i]:
                breach[i] = True; depth[i] = e[i] - l[i]
            if l[i] <= e[i] + wt:                       # came within the zone (incl. a wick below)
                near[i] = True
        elif s < -1:                                    # below-streak — watch the HIGH
            if h[i] > e[i]:
                breach[i] = True; depth[i] = h[i] - e[i]
            if h[i] >= e[i] - wt:
                near[i] = True
    return e, av, streak, near, breach, depth, armed_arr, event


def current_streak(d: pd.DataFrame, ema_len: int, atr_len: int, atr_frac: float,
                   use_streak_thresh: bool = False, streak_atr_frac: float = 0.5,
                   use_arm_exit: bool = False) -> dict | None:
    """Just the streak ENDING on the last bar: {direction, length, holds, breaches, max_wick,
    ema, start, armed, event}. 'holds' counts EMA touches after the 1st streak bar (same as compute's
    `near`). 'armed' = an exit is armed on the current ride; 'event' in {'', 'armed', 'saved', 'break'}
    is the last bar's state-change (arm-exit). Vectorized fast path for the plain no-threshold/no-arm
    case; exact-loop via compute() when use_streak_thresh OR use_arm_exit is on. Used by the scanner."""
    c = d["close"].to_numpy(float); h = d["high"].to_numpy(float); l = d["low"].to_numpy(float)
    n = len(c)
    if n == 0:
        return None
    if use_streak_thresh or use_arm_exit:                  # dead-zone / arm-exit need the full loop
        e, _av, streak, near, breach, depth, armed_arr, event = compute(
            d, ema_len, atr_len, atr_frac, use_streak_thresh, streak_atr_frac, use_arm_exit)
        last = int(streak[-1]); length = abs(last); start = n - length
        dd = depth[start:][~np.isnan(depth[start:])]
        return {"direction": 1 if last > 0 else -1, "length": length, "start": start,
                "holds": int(near[start:].sum()), "breaches": int(breach[start:].sum()),
                "max_wick": round(float(dd.max()), 3) if len(dd) else 0.0,
                "near_now": bool(near[-1]) if length >= 2 else False,   # last bar touched the EMA zone
                "armed": bool(armed_arr[-1]), "event": str(event[-1]),
                "ema": float(e[-1]) if np.isfinite(e[-1]) else None}
    e = ema(c, ema_len); av = atr(h, l, c, atr_len)
    above = c > e                                           # thr=0: above-streak iff close > EMA
    cur = bool(above[-1])
    flips = np.nonzero(above != cur)[0]                     # last bar on the other side
    start = int(flips[-1] + 1) if len(flips) else 0         # streak occupies [start, n-1]
    s2 = slice(start + 1, n)                                # holds counted from the 2nd streak bar
    es = e[s2]; wt = av[s2] * atr_frac
    if cur:                                                 # above-streak — watch the LOW
        nm = l[s2] <= es + wt; bm = l[s2] < es; dep = np.where(bm, es - l[s2], np.nan)
    else:                                                   # below-streak — watch the HIGH
        nm = h[s2] >= es - wt; bm = h[s2] > es; dep = np.where(bm, h[s2] - es, np.nan)
    dd = dep[~np.isnan(dep)]
    return {"direction": 1 if cur else -1, "length": int(n - start), "start": start,
            "holds": int(nm.sum()), "breaches": int(bm.sum()),
            "max_wick": round(float(dd.max()), 3) if len(dd) else 0.0,
            "near_now": bool(nm[-1]) if (n - start) >= 2 else False,   # last bar touched the EMA zone
            "armed": False, "event": "break" if (len(flips) and flips[-1] == n - 1) else "",
            "ema": float(e[-1]) if np.isfinite(e[-1]) else None}


def streaks(streak, near, breach, depth) -> list[Streak]:
    """Collapse the per-bar streak into Streak segments (one per same-sign run)."""
    out: list[Streak] = []
    n = len(streak); i = 0
    while i < n:
        sgn = 1 if streak[i] > 0 else -1
        j = i
        while j + 1 < n and (1 if streak[j + 1] > 0 else -1) == sgn:
            j += 1
        dd = depth[i:j + 1]; dd = dd[~np.isnan(dd)]
        out.append(Streak(i, j, sgn, int(abs(streak[j])),
                          int(near[i:j + 1].sum()), int(breach[i:j + 1].sum()),
                          float(dd.mean()) if len(dd) else 0.0,
                          float(dd.max()) if len(dd) else 0.0))
        i = j + 1
    return out


def state_timeline(d: pd.DataFrame, ema_len: int, atr_len: int, atr_frac: float,
                   use_streak_thresh: bool = False, streak_atr_frac: float = 0.5,
                   use_arm_exit: bool = True) -> list[dict]:
    """Every state-change EVENT across the frame (for chart marks + the universe history build).
    One row per bar where the arm-exit machine fires: {bar, date, event, dir, streak}. `event` in
    {'armed','saved','break'}; `dir` is the side AFTER the event (+1 above / -1 below); `streak` is
    the signed streak at that bar (a break carries the dated-back count). date is ISO (intraday keeps
    the full timestamp so any-TF charts can place the mark on their own axis)."""
    _e, _av, streak, _n, _b, _dep, _armed, event = compute(
        d, ema_len, atr_len, atr_frac, use_streak_thresh, streak_atr_frac, use_arm_exit)
    intraday = bool((d.index.normalize() != d.index).any())
    iso = (lambda ts: ts.isoformat()) if intraday else (lambda ts: ts.date().isoformat())
    out = []
    for i in range(len(d)):
        ev = str(event[i])
        if ev:
            out.append({"bar": i, "date": iso(d.index[i]), "event": ev,
                        "dir": 1 if streak[i] > 0 else -1, "streak": int(streak[i])})
    return out


def to_table(segs: list[Streak], d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, s in enumerate(segs, 1):
        rows.append({"#": k, "dir": "▲ above" if s.direction > 0 else "▼ below",
                     "from": d.index[s.start].date().isoformat(),
                     "to": d.index[s.end].date().isoformat(),
                     "bars": s.bars, "near": s.near, "breach": s.breach,
                     "avg_wick": round(s.breach_avg, 3), "max_wick": round(s.breach_max, 3)})
    return pd.DataFrame(rows)


def render_html(symbol, d, e, streak, near, breach, segs, params, events=None) -> str:
    iso = [i.date().isoformat() for i in d.index]
    candles = [{"time": iso[i], "open": round(float(r.open), 4), "high": round(float(r.high), 4),
                "low": round(float(r.low), 4), "close": round(float(r.close), 4)}
               for i, (_, r) in enumerate(d.iterrows())]
    emaline = [{"time": iso[i], "value": round(float(e[i]), 4)} for i in range(len(d)) if np.isfinite(e[i])]
    # streak background: green above / red below, more opaque the longer the streak
    bg = []
    for i in range(len(d)):
        op = min(0.30, 0.05 + abs(int(streak[i])) * 0.012)
        col = f"rgba(47,191,143,{op:.2f})" if streak[i] > 0 else f"rgba(224,90,109,{op:.2f})"
        bg.append({"time": iso[i], "value": 1, "color": col})
    # markers: flip labels (the streak that just ended) + per-bar near/breach dots
    markers = []
    for k, s in enumerate(segs):
        if k == 0:
            continue
        prev = segs[k - 1]
        markers.append({"time": iso[s.start], "position": "belowBar" if s.direction > 0 else "aboveBar",
                        "shape": "arrowUp" if s.direction > 0 else "arrowDown",
                        "color": "#2fbf8f" if s.direction > 0 else "#e05a6d",
                        "text": f"{prev.bars}b {'▲' if prev.direction > 0 else '▼'}"})
    for i in range(len(d)):
        if not (near[i] or breach[i]):
            continue
        up = streak[i] > 0
        markers.append({"time": iso[i], "position": "belowBar" if up else "aboveBar",
                        "shape": "circle", "color": "#e05a6d" if breach[i] else "#e8b84b", "text": ""})
    if events is not None:                                  # arm-exit state-change markers (⚠ armed / ✓ saved)
        for i in range(len(d)):
            ev = str(events[i])
            if ev == "armed":
                up = streak[i] > 0                          # armed above -> mark the dip below; below -> above
                markers.append({"time": iso[i], "position": "belowBar" if up else "aboveBar",
                                "shape": "arrowDown" if up else "arrowUp", "color": "#f0932b", "text": "⚠ armed"})
            elif ev == "saved":
                markers.append({"time": iso[i], "position": "aboveBar", "shape": "circle",
                                "color": "#22d3ee", "text": "✓ saved"})
    markers.sort(key=lambda m: m["time"])
    seg_rows = to_table(segs, d).to_dict("records")
    bounds = [{"a": s.start, "b": s.end} for s in segs]
    live = int(streak[-1])
    html = (HTML_TEMPLATE
            .replace("__SYM__", symbol)
            .replace("__PARAMS__", params)
            .replace("__LIVE__", f"{abs(live)}b {'▲ above' if live > 0 else '▼ below'}")
            .replace("__CANDLES__", json.dumps(candles))
            .replace("__EMA__", json.dumps(emaline))
            .replace("__BG__", json.dumps(bg))
            .replace("__MARKERS__", json.dumps(markers))
            .replace("__SEGS__", json.dumps(seg_rows, default=str))
            .replace("__BOUNDS__", json.dumps(bounds)))
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = config.OUTPUT_DIR / f"emarider_{symbol}.html"
    out.write_text(html, encoding="utf-8")
    return str(out)


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>emarider — __SYM__</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
:root{--bg:#0b0f14;--panel:#11161d;--line:#1d2630;--txt:#cfd8e3;--dim:#6b7a8c;
      --up:#2fbf8f;--dn:#e05a6d;--acc:#e8b84b;--mono:'Consolas','Menlo',monospace}
*{box-sizing:border-box;margin:0}
body{background:var(--bg);color:var(--txt);font:14px/1.45 system-ui,sans-serif;display:flex;flex-direction:column;height:100vh}
header{display:flex;gap:14px;align-items:baseline;padding:10px 18px;border-bottom:1px solid var(--line)}
header h1{font:600 16px var(--mono);color:var(--acc)}
header .d{color:var(--dim);font:12px var(--mono)}
header .live{margin-left:auto;font:600 14px var(--mono);color:var(--txt)}
#chart{flex:1;min-height:0}
#legend{display:flex;gap:18px;padding:6px 18px;color:var(--dim);font:12px var(--mono);border-top:1px solid var(--line)}
#legend span::before{content:'';display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:middle}
#legend .u::before{background:rgba(47,191,143,.6);border-radius:2px} #legend .d2::before{background:rgba(224,90,109,.6);border-radius:2px}
#legend .n::before{background:#e8b84b} #legend .b::before{background:#e05a6d} #legend .e::before{background:orange;border-radius:2px}
#tbl{max-height:34vh;overflow:auto;border-top:1px solid var(--line)}
table{border-collapse:collapse;width:100%;font:12.5px var(--mono)}
th,td{padding:5px 10px;text-align:right;white-space:nowrap}
th{position:sticky;top:0;background:var(--panel);color:var(--dim);border-bottom:1px solid var(--line)}
th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3),th:nth-child(4),td:nth-child(4){text-align:left}
tbody tr{cursor:pointer;border-bottom:1px solid #131a22}
tbody tr:hover{background:#16202b}
td.up{color:var(--up)} td.dn{color:var(--dn)} td.hl{color:var(--acc)}
</style></head><body>
<header><h1>EMA RIDER __SYM__</h1><span class="d">__PARAMS__ · click a streak row to zoom</span>
<span class="live">live: __LIVE__</span></header>
<div id="chart"></div>
<div id="legend"><span class="u">above-streak</span><span class="d2">below-streak</span>
<span class="e">EMA</span><span class="n">near EMA (wick in zone)</span><span class="b">breach (wick crossed)</span>
<span>⚠ armed (closed wrong-side) · ✓ saved · ▲/▼ break = streak that ended</span>
<span style="margin-left:auto;color:#8ea2b5">shift-drag to measure Δ / % / bars</span></div>
<div id="tbl"></div>
<script>
const CANDLES=__CANDLES__, EMA=__EMA__, BG=__BG__, MARKERS=__MARKERS__, SEGS=__SEGS__, BOUNDS=__BOUNDS__;
const chart=LightweightCharts.createChart(document.getElementById('chart'),{
  layout:{background:{color:'#0b0f14'},textColor:'#6b7a8c'},
  grid:{vertLines:{color:'#131a22'},horzLines:{color:'#131a22'}},
  rightPriceScale:{borderColor:'#1d2630'},timeScale:{borderColor:'#1d2630'},autoSize:true});
const bgs=chart.addHistogramSeries({priceScaleId:'bg',lastValueVisible:false,priceLineVisible:false});
chart.priceScale('bg').applyOptions({scaleMargins:{top:0,bottom:0},visible:false});
bgs.setData(BG);
const cs=chart.addCandlestickSeries({upColor:'#2fbf8f',downColor:'#e05a6d',
  wickUpColor:'#2fbf8f',wickDownColor:'#e05a6d',borderVisible:false});
cs.setData(CANDLES); cs.setMarkers(MARKERS);
chart.addLineSeries({color:'orange',lineWidth:2,priceLineVisible:false,lastValueVisible:false,
  crosshairMarkerVisible:false}).setData(EMA);
chart.timeScale().fitContent();
const cols=Object.keys(SEGS[0]||{});
let h='<table><thead><tr>'+cols.map(c=>'<th>'+c+'</th>').join('')+'</tr></thead><tbody>';
for(let i=0;i<SEGS.length;i++){const x=SEGS[i];
  h+='<tr data-i="'+i+'">'+cols.map(c=>{let cls='';
    if(c==='dir')cls=x[c][0]==='▲'?'up':'dn';
    if(c==='near'||c==='breach')cls='hl';
    return '<td class="'+cls+'">'+x[c]+'</td>';}).join('')+'</tr>';}
document.getElementById('tbl').innerHTML=h+'</tbody></table>';
document.querySelectorAll('#tbl tbody tr').forEach(tr=>tr.onclick=()=>{
  const b=BOUNDS[+tr.dataset.i], pad=Math.max(5,Math.round((b.b-b.a)*0.25));
  chart.timeScale().setVisibleLogicalRange({from:b.a-pad,to:b.b+pad});});
// shift-drag measure tool: Δ price / % / bar count between two points
(function(){const el=document.getElementById('chart');let a=null,tip=null;
  const at=ev=>{const r=el.getBoundingClientRect(),x=ev.clientX-r.left,y=ev.clientY-r.top;
    return {price:cs.coordinateToPrice(y),lo:chart.timeScale().coordinateToLogical(x)};};
  el.addEventListener('mousedown',ev=>{if(!ev.shiftKey)return;a=at(ev);ev.preventDefault();
    tip=document.createElement('div');
    tip.style.cssText='position:fixed;z-index:9;background:#11161d;border:1px solid #2b3a4a;color:#cfd8e3;font:12px monospace;padding:4px 7px;border-radius:4px;pointer-events:none';
    document.body.appendChild(tip);});
  window.addEventListener('mousemove',ev=>{if(!a||!tip)return;const b=at(ev);
    if(a.price==null||b.price==null)return;const dp=b.price-a.price,pct=a.price?100*dp/a.price:0,bars=Math.round(b.lo-a.lo);
    tip.textContent='Δ '+dp.toFixed(2)+'   '+(pct>=0?'+':'')+pct.toFixed(2)+'%   '+bars+'b';
    tip.style.left=(ev.clientX+12)+'px';tip.style.top=(ev.clientY+12)+'px';});
  window.addEventListener('mouseup',()=>{if(tip){tip.remove();tip=null;}a=null;});})();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("symbol", help="ticker in the bar cache, or a path to an OHLC csv")
    ap.add_argument("--bars", type=int, default=500, help="trailing bars to analyze (0 = all)")
    ap.add_argument("--ema", type=int, default=20, help="EMA length")
    ap.add_argument("--atr", type=int, default=14, help="ATR length")
    ap.add_argument("--frac", type=float, default=0.5, help="wick-proximity = ATR x this (the 'near EMA' zone)")
    ap.add_argument("--streak-thresh", action="store_true", help="allow a close within the ATR zone to extend the streak")
    ap.add_argument("--streak-frac", type=float, default=0.5, help="streak threshold = ATR x this (with --streak-thresh)")
    ap.add_argument("--legacy", action="store_true", help="legacy flip-on-first-wrong-close (disable the arm-exit machine)")
    ap.add_argument("--open", action="store_true", help="open the html when done")
    args = ap.parse_args()

    name, df = trendlab.load_frame(args.symbol, args.bars)
    e, atr, streak, near, breach, depth, armed_arr, event = compute(
        df, args.ema, args.atr, args.frac, args.streak_thresh, args.streak_frac, not args.legacy)
    segs = streaks(streak, near, breach, depth)
    params = (f"EMA({args.ema}) · near = {args.frac}xATR({args.atr})"
              + (f" · streak thresh {args.streak_frac}xATR" if args.streak_thresh else ""))
    tbl = to_table(segs, df)
    print(f"\n{name}: {len(df)} bars -> {len(segs)} streaks   ({params})\n")
    print(tbl.to_string(index=False))
    out = render_html(name, df, e, streak, near, breach, segs, params, events=event)
    print(f"\nchart: {out}")
    if args.open:
        webbrowser.open(f"file://{out}")


if __name__ == "__main__":
    sys.exit(main())
