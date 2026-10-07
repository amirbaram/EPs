"""Dashboard rendering, shared by two front-ends:

- build_report(hits)          -> static output/dashboard_<date>.html (nightly EOD)
- server_shell() / compute_payload(...) -> the live serve.py backtesting app

Both use the same tables + interactive lightweight-charts UI. In backtest mode a
chart spans CHART_BARS before the as-of date plus CHART_FWD_BARS after it, with an
'as-of' marker and shaded outcome zone so you can see how the setup resolved.
"""
from __future__ import annotations

import json

import pandas as pd

import config
import datastore
import labels
import sr
from indicators import add_indicators

SETUP_LABELS = {
    "gapper": "Gappers",
    "episodic_pivot": "Episodic Pivot",
    "hvc": "High Volume Close",
    "delayed_hvc": "Delayed HVC",
    "flat_base": "Flat Base",
    "high_tight_flag": "High Tight Flag",
    "qm_breakout": "QM Breakout",
    "higher_low_ma": "Higher Low @ MA",
    "undercut_rally": "Undercut & Rally",
    "backburner": "Backburner",
    "stairstep": "Stairstep",
    "uptrend": "Uptrend",
    "downtrend": "Downtrend",
    "ema_cross": "EMA Cross",
    "ema_rider_bull": "Bullish EMA Rider",
    "ema_rider_bear": "Bearish EMA Rider",
    "cup_handle": "Cup & Handle",
    "double_top": "Double Top",
    "head_shoulders": "Head & Shoulders",
    "inverse_hs": "Inverse H&S",
    "rsi_extreme_revert": "RSI-Extreme Revert",
    "rsi_extreme_fade": "RSI-Extreme Fade",
}
# shown in the setup-info panel ("?" button): what the setup is + how Amir executes it
SETUP_DOCS = {
    "gapper": {"what": "Gap up >=6% on >=3x average volume — a catalyst day. Subtypes by context: "
               "breakaway (clears a base = start of a move), runaway (mid-trend continuation), "
               "exhaustion (gap into a 3-ATR extension = late, treat as a WARNING), common.",
               "exec": "Breakaway/runaway only. Entry: opening-range break (5/15/30m) in the gap "
               "direction or first pullback that holds VWAP. Stop: below OR low / gap-day low. "
               "Exhaustion gaps are fade/avoid candidates, not buys."},
    "episodic_pivot": {"what": "Bonde EP: a repricing EVENT — catalyst gap (>=6% on 3x vol), the "
               "9M-share objective filter, or a no-gap big move (>=10% on 3x vol; leaders relaxed "
               "to 2x). Big gaps must HOLD half the gap; dead shells excluded. Best from neglect.",
               "exec": "Day 1: buy the open or first 5-10min ORB with strength; stop = below the "
               "opening-range low (day-trade) or event-day low (swing). Delayed reaction: within "
               "~21 days entry on red-to-green / range break while the event low holds. Partials "
               "into strength 3-5 days out; trail EMA10."},
    "hvc": {"what": "High Volume Close: closes strong (top of range) on outsized volume after a "
            "real move — institutional accumulation footprint.",
            "exec": "Entry over the HVC bar high; stop below the HVC bar low. The HVC low should "
            "not be given back — if it is, the signal failed."},
    "delayed_hvc": {"what": "A gap (>=4%) in the last ~10 days that HELD (tight base, low never "
            ">3% below gap close), and today prints an HVC — the gap's energy releasing after a pause.",
            "exec": "Entry over today's high / base high; stop below the base low. Same "
            "follow-through logic as HVC."},
    "flat_base": {"what": "O'Neill flat base at highs: >=25-bar sideways shelf, depth <=12%, top "
            "touched 3+ times across the base, within 15% of 52w high, 30%+ prior advance, "
            "Stage-2 MA stack, volume drying up. Breakout needs 1.4x volume (else breakout_lowvol).",
            "exec": "Buy the close/break above the shelf top on volume; stop under the shelf "
            "midpoint or last touch low. Add nothing to low-volume pokes — wait for confirmation."},
    "high_tight_flag": {"what": "Pole of ~90%+ in <=8 weeks, then a tight shallow flag (<=25% "
            "pullback, 3-25 bars) near highs. Rarest and most powerful continuation pattern.",
            "exec": "Entry: break of the flag high on 1.4x volume (breakout_lowvol = wait). "
            "Stop: flag low. Target 2-5R; partials on ATR extensions, trail EMA10."},
    "qm_breakout": {"what": "Qullamaggie breakout: 30%+ fast move, then an orderly consolidation "
            "with a declining upper line; trigger = the envelope line, not the old high.",
            "exec": "Buy the close above the envelope on volume expansion; stop at the "
            "consolidation low or 1 ADR below entry. 3-5 day initial move, partials 1/3-1/2, "
            "trail EMA10."},
    "higher_low_ma": {"what": "Uptrend pullback holding a rising MA with a higher low — trend "
            "continuation entry zone.", "exec": "Entry on reclaim of the prior day's high off the "
            "MA touch; stop below the higher low. Classic pullback-to-EMA swing entry."},
    "undercut_rally": {"what": "Price undercuts an obvious prior low (stops run) then reclaims it "
            "— a spring/shakeout reversal.", "exec": "Entry on the reclaim close or next-day "
            "follow-through; stop below the undercut low. Works best at higher-timeframe support."},
    "backburner": {"what": "TCG BackBurner: after a NEW HIGH arms it, the FIRST intrabar RSI-30 "
            "wick (price touches the level where RSI prints 30) fires entry1; RSI-20 deepening = "
            "entry2. One fire per leg; only a new high re-arms. 'armed' rows = new-high day, not "
            "a trade.", "exec": "Entry: first bar whose high takes out the prior bar's high after "
            "the fire (entry = that prior high). Stop under the fire-day low. Exits at EMA bounce "
            "targets / prior high; works on 5m-12h the same way."},
    "stairstep": {"what": "Orderly staircase retracement after a strong uptrend — controlled "
            "profit-taking, not distribution.", "exec": "Entry on the break of the last lower "
            "high; NO trade if 1:1 R:R sits beyond the 12-EMA (TCG rule). Stop under the last step low."},
    "uptrend": {"what": "Classification: the current trendlab segment is UP. Context pool for "
            "long setups, not a signal.", "exec": "Not directly tradeable — hunt pullbacks/bases "
            "within these names."},
    "downtrend": {"what": "Classification: current segment DOWN. Context pool for shorts/avoid.",
            "exec": "Not directly tradeable — hunt failed bounces / breakdown setups here."},
    "ema_cross": {"what": "EMA Cross strategy based on Larsson Line.", "exec": "Enter on Yellow flip, Exit on Blue or Grey depending on sector."},
    "ema_rider_bull": {"what": "Price surfing a rising EMA with repeated holds and few breaches — "
            "a persistent institutional bid.", "exec": "Enter on touches/bounces of the riding "
            "EMA; exit when a 5/15m (intraday) or daily close breaches it. Partial at extensions."},
    "ema_rider_bear": {"what": "Mirror: price capped under a falling EMA.", "exec": "Short the "
            "rejections at the EMA; cover on a close back above it."},
    "cup_handle": {"what": "Continuation: 30%+ advance into the left rim, U-shaped cup 12-33% "
            "deep (single basin, not a V), right rim near the left, then a shallow 3-15 bar "
            "handle in the upper half on dry volume.", "exec": "Buy the break of the HANDLE high "
            "on 1.4x volume; stop under the handle low. Target = cup depth projected from the rim."},
    "double_top": {"what": "SHORT/reversal: the two highest highs of the window within ~2%, no "
            "higher high between/after, lighter volume on the 2nd peak. Confirmed only on the "
            "valley break.", "exec": "Short the close below the valley/neckline; stop above the "
            "2nd peak's reaction high. Target = pattern height projected. On longs you hold: a "
            "confirmed DT is the exit warning."},
    "head_shoulders": {"what": "SHORT/reversal: head = the true extreme with ADJACENT flanking "
            "shoulders, neckline through the two body-anchored reactions beside the head.",
            "exec": "Short the neckline break (or the weak bounce back into it); stop above the "
            "right shoulder. Target = head-to-neckline depth projected."},
    "inverse_hs": {"what": "Bottoming mirror: lowest low flanked by two higher lows, neckline "
            "overhead.", "exec": "Buy the neckline reclaim on volume; stop under the right "
            "shoulder low. Target = head depth projected."},
    "rsi_extreme_revert": {"what": "LONG mean-revert at the symbol's OWN all-time RSI low zone "
            "(rsi_extreme.pine port): the fire is the intrabar WICK touching the price where "
            "RSI(14) prints the zone edge (all-time low + buffer, prior history only) — "
            "backburner mechanics on a personal, dynamic extreme. armed = RSI inside the zone / "
            "fired within the recent window; entry1 = the wick fired TODAY.",
            "exec": "Buy the wick at the trigger price (or the reclaim after it); stop under the "
            "fire bar's low. Capitulation context — scale out into the bounce; one fire per "
            "episode, re-arms only after RSI reclaims 50."},
    "rsi_extreme_fade": {"what": "SHORT fade at the symbol's OWN all-time RSI HIGH zone: the "
            "intrabar wick touches the price where RSI prints (all-time high − buffer). "
            "Blow-off context — euphoria at levels the symbol has never sustained.",
            "exec": "Short the wick at the trigger (or the failure after it); stop above the fire "
            "bar's high. Quick partials — fades resolve fast; re-arms after RSI falls back "
            "through 50."},
}
EXTRA_COLS = {
    "episodic_pivot": ["ep_subtype", "ep_age", "gap_pct", "rvol", "neglect_6m", "held"],
    "gapper": ["gap_pct", "rvol", "held", "gap_type"],
    "hvc": ["rvol", "chg_pct", "close_pos", "prior_gain_pct"],
    "delayed_hvc": ["gap_pct", "base_bars", "depth_pct", "rvol", "pattern"],
    "flat_base": ["base_days", "depth_pct", "touches", "prior_gain_pct", "dist_to_trigger_pct", "pattern"],
    "high_tight_flag": ["pole_gain_pct", "pole_bars", "pole_efficiency", "pullback_pct", "flag_bars", "adr_pct", "off_hi52_pct", "ma_stack", "vol_dryup", "contraction", "vol_x", "breakouts", "levels", "pattern", "dist_to_trigger_pct"],
    "qm_breakout": ["move_pct", "cons_bars", "pullback_pct", "ride_frac", "shakeouts", "vol_x", "breakouts", "levels", "rs_1m", "rs_3m", "rs_6m", "pattern"],
    "uptrend": ["net_pct", "clarity", "bars", "ann_pct", "is_pole"],
    "downtrend": ["net_pct", "clarity", "bars", "ann_pct"],
    "higher_low_ma": ["ma", "higher_low", "prev_low", "undercut_pct", "pattern"],
    "undercut_rally": ["undercut_pct", "reclaim_pct", "rvol"],
    "ema_cross": ["quality", "entity_type", "variant", "sector_state", "theme_state", "days_since_flip", "optimal_tf"],
    "ema_rider_bull": ["ema_len", "streak", "holds", "breaches", "max_wick"],
    "ema_rider_bear": ["ema_len", "streak", "holds", "breaches", "max_wick"],
    "backburner": ["armed_tfs", "rsi", "os_price", "retrac_pct", "ath", "prior_gain_pct", "pattern"],
    "stairstep": ["step_bars", "pivot_high", "prior_gain_pct", "off_ath_at_peak", "above_sma200_at_peak", "adr_at_peak", "pattern"],
    "cup_handle": ["cup_bars", "depth_pct", "handle_bars", "handle_pull_pct", "target", "l_rim_date", "r_rim_date"],
    "double_top": ["peaks", "vol_lighter_p2", "target", "p1_date", "p2_date"],
    "head_shoulders": ["target", "ls_date", "head_date", "rs_date"],
    "inverse_hs": ["target", "ls_date", "head_date", "rs_date"],
    "rsi_extreme_revert": ["rsi", "rsi_extreme", "zone_edge", "trigger", "bars_since_fire"],
    "rsi_extreme_fade": ["rsi", "rsi_extreme", "zone_edge", "trigger", "bars_since_fire"],
}
# tf / state / quality / regime_* + the technical columns are annotated by scan._rows_from_enriched
BASE_COLS = ["symbol", "tf", "is_etf", "sector", "themes", "state", "quality", "rs_rank", "close",
             "risk", "reward", "rr", "rr_target", "max_pos",
             "market_cap", "adr_pct", "rvol_today", "dollar_vol_m", "avg_vol_k",
             "off_ath_pct", "off_hi52_pct", "above_lo52_pct", "above_sma50", "above_sma200",
             "regime_dir", "regime_clarity"]


def _enriched_loader(symbol: str):
    df = datastore.load_bars(symbol)
    return add_indicators(df) if df is not None else None


def _enriched_loader_w(symbol: str):
    df = datastore.load_bars_w(symbol)
    return add_indicators(df) if df is not None else None


def _anat(items, intraday: bool):
    """Sanitize an anatomy list (marks/segline); intraday: ISO stamps -> epoch ints to match the
    chart's numeric time axis."""
    if not isinstance(items, list):
        return None
    out = []
    for m in items:
        if not isinstance(m, dict):
            continue
        v = m.get("date")
        if intraday and v is not None:
            try:
                v = int(pd.Timestamp(v).tz_localize("UTC").timestamp())
            except Exception:
                v = None
        out.append({**m, "date": v})
    return out or None


def _chart_payload(d_full, hit: dict, asof=None) -> dict | None:
    """d_full = enriched full-history frame. Compact column arrays (small payload).
    asof=None -> last CHART_BARS. asof set -> CHART_BARS before + CHART_FWD_BARS after."""
    if d_full is None or len(d_full) == 0:
        return None
    if asof is not None:
        pos = int(d_full.index.searchsorted(pd.Timestamp(asof), side="right")) - 1
        if pos < 0:
            return None
        lo = max(0, pos - config.CHART_BARS + 1)
        hi = min(len(d_full), pos + config.CHART_FWD_BARS + 1)
        d = d_full.iloc[lo:hi]
        asof_iso = d_full.index[pos].date().isoformat()
    else:
        d = d_full.iloc[-config.CHART_BARS:]
        asof_iso = None
    num = lambda s: [None if pd.isna(v) else round(float(v), 4) for v in s] if s is not None else None
    opt = lambda x: None if x is None or isinstance(x, (list, tuple, dict)) or pd.isna(x) else float(x)
    optd = lambda x: None if x is None or (isinstance(x, float) and pd.isna(x)) else x
    # intraday frames carry a time-of-day -> emit a UNIX-second time axis (lightweight-charts
    # needs numeric time for intraday) and skip the date-based shading/as-of marker.
    intraday = bool((d.index.normalize() != d.index).any())
    # intraday: stamp the ET wall-clock as UTC so lightweight-charts (UTC axis) shows 09:30, 10:30, …
    t = ([int(i.tz_localize("UTC").timestamp()) for i in d.index] if intraday
         else [i.date().isoformat() for i in d.index])
    # shading boundaries must be in the same units as the time axis (UNIX seconds for intraday)
    shade = lambda x: (None if x is None or (isinstance(x, float) and pd.isna(x))
                       else (int(pd.Timestamp(x).tz_localize("UTC").timestamp()) if intraday else x))
    payload = {
        "t": t, "intraday": intraday,
        "o": num(d["open"]), "h": num(d["high"]), "l": num(d["low"]), "c": num(d["close"]),
        "v": [0 if pd.isna(v) else int(v) for v in d["volume"]],
        "ema10": num(d["ema10"]), "ema20": num(d["ema20"]), "sma50": num(d["sma50"]),
        "sma150": num(d["sma150"]) if "sma150" in d.columns else None,
        "sma200": num(d["sma200"]),
        "larsson_ema32": num(d["larsson_ema32"]) if "larsson_ema32" in d.columns else None,
        "larsson_ema35": num(d["larsson_ema35"]) if "larsson_ema35" in d.columns else None,
        "larsson_ema50": num(d["larsson_ema50"]) if "larsson_ema50" in d.columns else None,
        "larsson_ema58": num(d["larsson_ema58"]) if "larsson_ema58" in d.columns else None,
        "larsson_state": [None if pd.isna(v) else v for v in d["larsson_state"]] if "larsson_state" in d.columns else None,
        "larsson_ema15": None, "larsson_color": None,
        "level": opt(hit.get("level")),
        "trigger": opt(hit.get("trigger")),
        "uptrend_from": shade(hit.get("uptrend_from")),   # green-shade the ride window
        "uptrend_to": shade(hit.get("uptrend_to")),
        "downtrend_from": shade(hit.get("downtrend_from")),   # red-shade the ride window
        "downtrend_to": shade(hit.get("downtrend_to")),
        "segments": None if intraday else optd(hit.get("sb_segments")),    # base/leg/consolidation bands (if any)
        # pole / consolidation / breakout-line geometry (daily only) — shared by QM breakout and HTF
        "qm_pole": None if intraday else optd(hit.get("qm_pole") or hit.get("htf_pole")),
        "qm_cons": None if intraday else optd(hit.get("qm_cons") or hit.get("htf_cons")),
        # validated-setup pattern anatomy (rims / peaks / shoulders / necklines / targets) —
        # intraday marks convert to the chart's epoch axis
        "marks": _anat(hit.get("marks"), intraday),
        "segline": _anat(hit.get("segline"), intraday),
        "xlines": optd(hit.get("xlines")) if isinstance(hit.get("xlines"), list) else None,
        "stack_levels": None if intraday else optd(hit.get("stack_levels")),
        "sr_levels": _sr_for_chart(d_full, asof),          # support/resistance lines (as-of-safe)
        "asof": None if intraday else asof_iso,
    }
    try:                                                   # EMA-Rider overlay (⚡ toggle, any TF): streak
        import emarider                                    # shading + arm(⚠)/save(✓)/break(⚡) marks, computed
        _e, _av, _stk, _n, _b, _dp, _arm, _ev = emarider.compute(   # on the FULL frame for correct streak state
            d_full, config.ER_EMA_LEN, config.ER_ATR_LEN, config.ER_ATR_FRAC,
            config.ER_USE_STREAK_THRESH, config.ER_STREAK_ATR_FRAC, config.ER_USE_ARM_EXIT)
        w0 = int(d_full.index.get_loc(d.index[0]))          # window start position in the full frame
        ws, we = _stk[w0:w0 + len(d)], _ev[w0:w0 + len(d)]
        _rc = lambda s: (("rgba(47,191,143,%.2f)" if s > 0 else "rgba(224,90,109,%.2f)")
                         % min(0.28, 0.05 + abs(int(s)) * 0.012))
        rmk = []
        for j, ev in enumerate(we):
            ev = str(ev)
            if not ev:
                continue
            up = ws[j] > 0
            if ev == "armed":
                rmk.append({"time": t[j], "position": "belowBar" if up else "aboveBar",
                            "shape": "arrowDown" if up else "arrowUp", "color": "#f0932b", "text": "⚠"})
            elif ev == "saved":
                rmk.append({"time": t[j], "position": "aboveBar", "shape": "circle",
                            "color": "#22d3ee", "text": "✓"})
            elif ev == "break":
                rmk.append({"time": t[j], "position": "aboveBar" if up else "belowBar",
                            "shape": "arrowUp" if up else "arrowDown",
                            "color": "#2fbf8f" if up else "#e05a6d", "text": "⚡%db" % abs(int(ws[j]))})
        payload["rider"] = {"bg": [_rc(s) for s in ws], "marks": rmk}
    except Exception:
        pass
    return payload


def _sr_for_chart(d_full, asof) -> list | None:
    """Live S/R levels for the chart's frame, computed up to the as-of bar (no look-ahead)."""
    try:
        frame = d_full
        if asof is not None:
            pos = int(d_full.index.searchsorted(pd.Timestamp(asof), side="right")) - 1
            if pos < 30:
                return None
            frame = d_full.iloc[:pos + 1]
        res = sr.sr_levels(frame)
        return [{"price": L["price"], "kind": L["kind"], "strength": L["strength"]}
                for L in (res["resistance"] + res["support"])] or None
    except Exception:
        return None


def _daily_context_line(daily_df):
    """Daily closes (last CHART_DAILY_CONTEXT_BARS) as [unix_secs, close], stamped 16:00 ET-as-UTC
    so they align with an intraday chart's axis — drawn left of the 60-day candles so you can see
    where price came from before the intraday window."""
    if daily_df is None or len(daily_df) == 0:
        return None
    d = daily_df.iloc[-config.CHART_DAILY_CONTEXT_BARS:]
    out = []
    for i, c in zip(d.index, d["close"]):
        if pd.isna(c):
            continue
        ts = int((pd.Timestamp(i.date()) + pd.Timedelta(hours=16)).tz_localize("UTC").timestamp())
        out.append([ts, round(float(c), 4)])
    return out or None


def _clean(recs):
    # Omit None and NaN fields to keep payload lean; JS treats missing keys as undefined (falsy)
    out = []
    for rec in recs:
        row = {}
        for k, v in rec.items():
            if v is None:
                continue
            if isinstance(v, float) and pd.isna(v):
                continue
            row[k] = v
        out.append(row)
    return out


def attach_labels(recs):
    """Add the sector + themes columns to each row at response time (NOT baked into the cached
    hits) so editing the label CSVs + labels.reload() shows up instantly, with no rescan."""
    for r in recs:
        sym = r.get("symbol")
        r["sector"] = labels.sector(sym)
        r["themes"] = ", ".join(labels.themes(sym)) or None
    return recs


def compute_payload(hits: pd.DataFrame, asof=None, get_frame=None, get_frame_w=None,
                    embed_charts: bool = True) -> tuple[dict, dict]:
    """(tables, charts) for the dashboard JS. get_frame / get_frame_w (symbol)->enriched
    daily / weekly frame (default = datastore; the server passes its RAM caches).
    Chart key = symbol|setup|tf, and a 1W hit is charted on weekly bars. embed_charts=False
    ships tables only (the live server loads each chart on demand via /api/chart)."""
    get_frame = get_frame or _enriched_loader
    get_frame_w = get_frame_w or _enriched_loader_w
    charts: dict[str, dict] = {}
    if embed_charts and len(hits):
        chart_rows = (hits.sort_values("dollar_vol_m", ascending=False)
                      .groupby(["setup", "tf"]).head(config.CHART_TOP_N))
        for r in chart_rows.itertuples():
            tf = getattr(r, "tf", "1D")
            frame = get_frame_w(r.symbol) if tf == "1W" else get_frame(r.symbol)
            payload = _chart_payload(frame, r._asdict(), asof=asof)
            if payload:
                charts[f"{r.symbol}|{r.setup}|{tf}"] = payload
    tables = {s: attach_labels(_clean(hits[hits["setup"] == s].to_dict("records"))) if len(hits) else []
              for s in SETUP_LABELS}
    return tables, charts


def build_report(hits: pd.DataFrame) -> str:
    date = hits["date"].max() if len(hits) else "empty"
    tables, charts = compute_payload(hits)
    page = (_page(date_label=f"EOD {date}", datepicker=False)
            .replace("/*__BOOT__*/", f"applyData({json.dumps(tables, default=lambda x: None if pd.isna(x) else str(x))},"
                                     f"{json.dumps(charts, default=lambda x: None if pd.isna(x) else str(x))});"))
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = config.OUTPUT_DIR / f"dashboard_{date}.html"
    out.write_text(page, encoding="utf-8")
    return str(out)


def server_shell(default_date: str = "") -> str:
    """Dashboard shell for serve.py: a date picker that fetches /api/scan?date=."""
    boot = ("const di=document.getElementById('asof');"
            "window.liveDate=di.value;"                       # the newest date at boot = 'live'
            "di.onchange=()=>load(di.value);"
            "const bl=document.getElementById('backlive');"
            "if(bl)bl.onclick=()=>{di.value=window.liveDate;load(window.liveDate);};"
            "async function load(date){const tot=document.getElementById('totals');"
            "const seq=(window._loadSeq=(window._loadSeq||0)+1);"       # last-clicked date wins
            "if(bl)bl.style.display=(window.liveDate&&date!==window.liveDate)?'':'none';"
            "let restored=false;"                                       # stale-while-revalidate: paint the cached scan first
            "try{const c=await idbGet('scan:'+date);"
            "if(c&&c.tables&&seq===window._loadSeq){applyData(c.tables,{});restored=true;}}catch(e){}"
            "tot.textContent=(restored?'refreshing ':'scanning ')+date+' \\u2026';"
            "try{const r=await fetch('/api/scan?date='+date);const j=await r.json();"
            "if(seq!==window._loadSeq)return;"                          # a newer load superseded this one
            "if(j.error){if(!restored)tot.textContent=j.error;return;}"
            "applyData(j.tables,j.charts);loadMarket(date);"
            "if(j.coverage&&j.coverage.incomplete){const tt=document.getElementById('totals');"
            "if(tt)tt.insertAdjacentText('beforeend',(j.coverage.refreshing?' \\ud83d\\udd04 self-correcting \\u2014 pulling fresh data\\u2026':' \\u23f3 data incomplete '+j.coverage.on_date+'/'+j.coverage.loaded+' \\u2014 refreshing\\u2026'));"
            "if(j.coverage.refreshing)pollRefresh(date);}"
            "idbPut('scan:'+date,{date:date,ts:Date.now(),tables:j.tables});"   # cache fresh for next reload
            "}catch(e){if(seq===window._loadSeq&&!restored)tot.textContent='scan failed: '+e;}}"
            "function pollRefresh(date){let n=0;const iv=setInterval(async()=>{n++;let done=n>120;"
            "try{const s=await(await fetch('/api/update/status')).json();if(!s.running)done=true;}catch(e){}"
            "if(done){clearInterval(iv);load(date);}},2500);}"
            "if(di.value)load(di.value);")
    return (_page(date_label="backtest", datepicker=True)
            .replace("__ASOF__", default_date)
            .replace("/*__BOOT__*/", boot))


def _page(date_label: str, datepicker: bool) -> str:
    # the settings panel + data-update button need a backend, so they render only in
    # server mode (datepicker=True); the static nightly dashboard omits them.
    picker = ('<input type="date" id="asof" value="__ASOF__" '
              'style="background:var(--panel);border:1px solid var(--line);color:var(--acc);'
              'padding:5px 8px;border-radius:4px;font-family:var(--mono)">'
              '<button class="tool" id="backlive" style="display:none" '
              'title="you are viewing a PAST date (time-travel) — click to return to the latest">'
              '&#9194; back to live</button>') if datepicker else ""
    tools = TOOLS_HTML if datepicker else ""
    drawer = DRAWER_HTML if datepicker else ""
    extra_js = SETTINGS_JS if datepicker else ""
    head = HTML_HEAD.replace("__DATE__", date_label)
    body = (HTML_BODY.replace("__PICKER__", picker).replace("__TOOLS__", tools)
            .replace("__DRAWER__", drawer).replace("__DATE__", date_label))
    cfg = (f"const LABELS={json.dumps(SETUP_LABELS)},EXTRA={json.dumps(EXTRA_COLS)},"
           f"BASE={json.dumps(BASE_COLS)},DOCS={json.dumps(SETUP_DOCS)},"
           f"ONDEMAND={str(datepicker).lower()},"
           f"EXTQ={json.dumps(config.EXT_BADGE_TIERS)},"
           f"THEME_ETF={json.dumps(labels.THEME_ETF)},"
           f"ZOOM={config.CHART_ZOOM_BARS},ZOOMF={config.CHART_ZOOM_FWD};")
    return head + body + "<script>" + cfg + DASHBOARD_JS + extra_js + "/*__BOOT__*/\n</script></body></html>"


TOOLS_HTML = ('<select class="tool" id="tfsel" title="EMA Rider timeframe (daily-resampled or intraday)" style="display:none">'
              '<optgroup label="Daily"><option>1D</option><option>2D</option><option>3D</option><option>1W</option>'
              '<option>2W</option><option>1M</option><option>3M</option><option>6M</option></optgroup>'
              '<optgroup label="Intraday"><option>5m</option><option>15m</option><option>30m</option><option>1h</option>'
              '<option>2h</option><option>4h</option><option>8h</option><option>12h</option></optgroup></select>'
              '<button class="tool" id="btncfg" title="Setup settings + data maintenance">⚙</button>')

DRAWER_HTML = r"""<div id="scrim"></div>
<aside id="drawer">
  <header><h2>SETTINGS</h2><button class="tool" id="cfgclose">&#10005;</button></header>
  <div id="drawermaint" style="display:flex;gap:6px;flex-wrap:wrap;padding:8px 14px;border-bottom:1px solid var(--line)">
    <button class="tool" id="btnupd" title="Pull latest EOD data for the active universe &amp; rescan">&#8635; Update</button>
    <button class="tool" id="btnwk" title="Weekly: full-universe refresh + re-evaluate which tickers are in the active universe (slow)">&#8635; Weekly</button>
    <button class="tool" id="btndl" title="Download 60-day intraday bars (futures 5m) for the loaded list">&#11015; Intraday</button>
    <button class="tool" id="btnlbl" title="Reload sector/theme labels after editing data/themes_manual.csv (no rescan)">&#8635; Labels</button>
  </div>
  <div id="drawerbody">loading&#8230;</div>
  <div id="drawerfoot">
    <button class="tool" id="cfgreset">Reset</button>
    <button class="tool" id="cfgsave">Save</button>
    <button class="tool" id="cfgapply">Apply &amp; Rescan</button>
  </div>
</aside>"""

SETTINGS_JS = r"""
// ---- settings drawer + data update (server mode only) ----
const drawer=document.getElementById('drawer'),scrim=document.getElementById('scrim');
let SETMETA=null;
function openDrawer(){drawer.classList.add('open');scrim.classList.add('open');loadSettings();}
function closeDrawer(){drawer.classList.remove('open');scrim.classList.remove('open');}
async function loadSettings(){const b=document.getElementById('drawerbody');
  try{const j=await(await fetch('/api/settings')).json();SETMETA=j;b.innerHTML=buildForm(j);}
  catch(e){b.textContent='failed to load settings: '+e;}}
function inputFor(name,val){
  if(typeof val==='boolean')return'<input type="checkbox" data-k="'+name+'"'+(val?' checked':'')+'>';
  if(typeof val==='number')return'<input type="number" step="any" data-k="'+name+'" value="'+val+'">';
  if(Array.isArray(val)||typeof val==='object')
    return'<textarea data-k="'+name+'" data-json="1">'+JSON.stringify(val)+'</textarea>';
  return'<input type="text" data-k="'+name+'" value="'+(val==null?'':val)+'">';}
function buildForm(j){let h='';for(const grp in j.groups){h+='<h3>'+grp.replace(/_/g,' ')+'</h3>';
  for(const name of j.groups[grp]){const v=j.values[name];
    h+='<div class="srow"><label title="'+name+'">'+name+'</label>'+inputFor(name,v)+'</div>';}}
  return h;}
function collect(){const o={};document.querySelectorAll('#drawerbody [data-k]').forEach(el=>{
  const k=el.dataset.k;
  if(el.type==='checkbox')o[k]=el.checked;
  else if(el.dataset.json){try{o[k]=JSON.parse(el.value);}catch(e){return;}}
  else if(el.type==='number'){const n=parseFloat(el.value);if(!isNaN(n))o[k]=n;}
  else o[k]=el.value;});return o;}
async function saveSettings(){await fetch('/api/settings',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify({overrides:collect()})});}
async function applySettings(){await saveSettings();closeDrawer();
  const di=document.getElementById('asof');if(di&&di.value)load(di.value);}
function resetSettings(){if(SETMETA){SETMETA.values=Object.assign({},SETMETA.defaults);
  document.getElementById('drawerbody').innerHTML=buildForm(SETMETA);}}
document.getElementById('btncfg').onclick=openDrawer;
document.getElementById('cfgclose').onclick=closeDrawer;
scrim.onclick=closeDrawer;
document.getElementById('cfgsave').onclick=saveSettings;
document.getElementById('cfgapply').onclick=applySettings;
document.getElementById('cfgreset').onclick=resetSettings;
// data update buttons (Update = active-universe EOD; Weekly = full refresh + universe re-evaluation)
const btnu=document.getElementById('btnupd'), btnwk=document.getElementById('btnwk');
async function doUpdate(){btnu.classList.add('busy');btnu.textContent='↻ Updating…';
  try{const r=await(await fetch('/api/update',{method:'POST'})).json();
    if(r.error){btnu.textContent='↻ '+r.error;btnu.classList.remove('busy');return;}
    pollUpdate();}catch(e){btnu.textContent='↻ failed';btnu.classList.remove('busy');}}
async function doWeekly(){if(!confirm('Weekly full-universe refresh + re-evaluate the active universe. This is slow (downloads every name). Continue?'))return;
  btnwk.classList.add('busy');btnwk.textContent='↻ Weekly…';
  try{const r=await(await fetch('/api/update?mode=weekly',{method:'POST'})).json();
    if(r.error){btnwk.textContent='↻ '+r.error;btnwk.classList.remove('busy');setTimeout(()=>btnwk.textContent='↻ Weekly',2000);return;}
    pollUpdate();}catch(e){btnwk.textContent='↻ failed';btnwk.classList.remove('busy');}}
async function pollUpdate(){try{const s=await(await fetch('/api/update/status')).json();
  if(s.running){btnu.textContent='↻ '+(s.phase||'updating')+'…';btnwk.classList.add('busy');setTimeout(pollUpdate,2000);return;}
  btnu.classList.remove('busy');btnu.textContent='↻ Update';
  btnwk.classList.remove('busy');btnwk.textContent='↻ Weekly';
  if(s.error){alert('update failed: '+s.error);return;}
  const di=document.getElementById('asof');if(di&&s.latest){window.liveDate=s.latest;di.value=s.latest;load(s.latest);}}
  catch(e){btnu.classList.remove('busy');btnu.textContent='↻ Update';btnwk.classList.remove('busy');btnwk.textContent='↻ Weekly';setTimeout(pollUpdate,3000);}}
btnu.onclick=doUpdate; btnwk.onclick=doWeekly;
// EMA Rider timeframe selector: 1D restores the baseline; others fetch /api/riders for that TF
const tfsel=document.getElementById('tfsel');
async function applyRiderTF(){const tot=document.getElementById('totals');
  if(riderTF==='1D'){if(BASE_RIDERS){TABLES.ema_rider_bull=BASE_RIDERS.ema_rider_bull;TABLES.ema_rider_bear=BASE_RIDERS.ema_rider_bear;}render();return;}
  const di=document.getElementById('asof');tot.textContent='scanning '+riderTF+' riders …';
  try{const r=await fetch('/api/riders?tf='+riderTF+'&date='+(di?di.value:''));const j=await r.json();
    if(j.error){tot.textContent=j.error;return;}
    if(j.needs_intraday){tot.textContent='no intraday data — click ⬇ Intraday to download intraday bars';
      TABLES.ema_rider_bull=[];TABLES.ema_rider_bear=[];render();return;}
    TABLES.ema_rider_bull=j.tables.ema_rider_bull||[];TABLES.ema_rider_bear=j.tables.ema_rider_bear||[];
    counts();render();}
  catch(e){tot.textContent='rider scan failed: '+e;}}
tfsel.onchange=()=>{riderTF=tfsel.value;applyRiderTF();};
// ⚡ EMA-Rider chart overlay: streak shading + arm/save/break marks (re-renders the current chart, no re-fetch)
window.riderOverlay=localStorage.getItem('riderOverlay')==='1';
(function(){const b=document.getElementById('btnrider');if(!b)return;
  b.classList.toggle('on',window.riderOverlay);
  b.onclick=()=>{window.riderOverlay=!window.riderOverlay;
    localStorage.setItem('riderOverlay',window.riderOverlay?'1':'0');
    b.classList.toggle('on',window.riderOverlay);
    if(window._lastChartData)drawChart(window._lastChartData);};})();
// intraday download button (reuses the update poll pattern)
const btndl=document.getElementById('btndl');let dlTimer=null;
function dlCountdown(sec){if(dlTimer)clearTimeout(dlTimer);   // disable the button until the cooldown elapses
  if(sec<=0){btndl.disabled=false;btndl.textContent='⬇ Intraday';
    btndl.title='Download 60-day intraday bars (futures 5m) for the loaded list';return;}
  btndl.disabled=true;btndl.classList.remove('busy');
  const m=Math.floor(sec/60);btndl.textContent='⬇ '+(m>=1?m+'m':sec+'s');
  btndl.title='intraday just updated — available again in ~'+(m>=1?m+' min':sec+' s');
  dlTimer=setTimeout(()=>dlCountdown(sec-1),1000);}
async function doDownload(){if(btndl.disabled)return;btndl.classList.add('busy');btndl.textContent='⬇ …';
  try{const r=await(await fetch('/api/download_intraday',{method:'POST'})).json();btndl.classList.remove('busy');
    if(r.cooldown){dlCountdown(r.retry_in);return;}
    if(r.error){btndl.textContent='⬇ Intraday';alert(r.error);return;}
    pollDownload();}catch(e){btndl.textContent='⬇ failed';btndl.classList.remove('busy');}}
async function pollDownload(){try{const s=await(await fetch('/api/download_intraday/status')).json();
  if(s.running){btndl.disabled=true;btndl.classList.add('busy');btndl.textContent='⬇ '+(s.current||0)+'/'+(s.total||'?');setTimeout(pollDownload,800);return;}
  btndl.classList.remove('busy');
  if(s.error){btndl.disabled=false;btndl.textContent='⬇ Intraday';alert('download failed: '+s.error);return;}
  if(RIDER_TABS.includes(cur)&&riderTF!=='1D')applyRiderTF();   // refresh if viewing an intraday TF
  dlCountdown(s.retry_in||0);}                                  // start the cooldown after a successful pull
  catch(e){btndl.classList.remove('busy');btndl.disabled=false;btndl.textContent='⬇ Intraday';setTimeout(pollDownload,3000);}}
btndl.onclick=doDownload;
fetch('/api/download_intraday/status').then(r=>r.json()).then(s=>{if(s.running)pollDownload();else dlCountdown(s.retry_in||0);}).catch(()=>{});
// reload sector/theme labels after editing the CSVs — instant, no rescan
const btnlbl=document.getElementById('btnlbl');
btnlbl.onclick=async()=>{btnlbl.classList.add('busy');btnlbl.textContent='↻ …';
  try{await fetch('/api/reload_labels',{method:'POST'});
    const di=document.getElementById('asof');
    if(RIDER_TABS.includes(cur)&&riderTF!=='1D')await applyRiderTF(); else if(di)await load(di.value);
  }catch(e){}
  btnlbl.classList.remove('busy');btnlbl.textContent='↻ Labels';};
"""


HTML_HEAD = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Setup Scanner — __DATE__</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
:root{--bg:#0b0f14;--panel:#11161d;--line:#1d2630;--txt:#cfd8e3;--dim:#6b7a8c;
      --up:#2fbf8f;--dn:#e05a6d;--acc:#e8b84b;--mono:'Consolas','Menlo',monospace}
*{box-sizing:border-box;margin:0}
.mbar{padding:0 10px;background:var(--panel);border-bottom:1px solid var(--line);position:relative;z-index:60}
.mhead{display:flex;align-items:center;gap:10px;padding:5px 2px;cursor:pointer;user-select:none}
.mbadge{color:#0b0f14;font-weight:700;font-family:var(--mono);padding:3px 10px;border-radius:4px;font-size:13px;letter-spacing:.4px;white-space:nowrap}
.msum{color:var(--dim);font-size:12px;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.mexp{color:var(--dim);font-size:12px}
.mdetail{padding:4px 2px 8px;font-family:var(--mono);font-size:12px}
.mrow{display:flex;gap:8px;align-items:baseline;padding:2px 0;border-top:1px solid #131a22}
.mname{color:var(--acc);width:60px;text-transform:capitalize}
.mname.up{color:#2fbf8f}.mname.dn{color:#e05a6d}
.mwhy{color:var(--txt);flex:1}
.pbidea{padding:1px 0}
.tsbar{display:flex;align-items:center;gap:10px;padding:4px 4px 2px}
.tslabel{font-family:var(--mono);font-size:11px;color:var(--acc);min-width:78px;white-space:nowrap}
#tslider{flex:1;max-width:520px;accent-color:var(--acc);cursor:pointer}
.tsbar .tool{padding:2px 8px;font-size:11px}
.mlive{color:var(--acc);font-family:var(--mono);font-size:10px;font-weight:700;letter-spacing:.5px}
.mstrip{color:var(--dim);font-family:var(--mono);font-size:11px;white-space:nowrap}
.mlivehead{font-family:var(--mono);font-size:13px;white-space:nowrap;padding:2px 9px;border-radius:4px;background:rgba(47,191,143,.10);border:1px solid rgba(47,191,143,.30)}
.mlivehead .dtier{font-size:11px;font-weight:700;margin-left:5px}
.mwarm{font-family:var(--mono);font-size:11px;font-weight:700;color:#e0a43a;background:rgba(224,164,58,.14);border:1px solid rgba(224,164,58,.38);padding:2px 8px;border-radius:4px;white-space:nowrap;cursor:help;animation:warmpulse 1.9s ease-in-out infinite}
@keyframes warmpulse{0%,100%{opacity:1}50%{opacity:.5}}
.grp{white-space:nowrap;margin-right:2px}
body{background:var(--bg);color:var(--txt);font:15px/1.5 system-ui,Segoe UI,sans-serif;
     height:100vh;overflow:hidden;display:flex;flex-direction:column}
header{display:flex;align-items:center;gap:12px;padding:8px 16px;border-bottom:1px solid var(--line);flex-wrap:wrap}
header h1{font-size:16px;letter-spacing:.04em;color:var(--acc);font-family:var(--mono);white-space:nowrap}
header .d{color:var(--dim);font-family:var(--mono);font-size:13px;white-space:nowrap}
.filters{display:flex;gap:9px;align-items:center;font-size:12.5px;color:var(--dim);
  margin-left:auto;font-family:var(--mono)}
.filters label{display:flex;gap:3px;align-items:center;cursor:pointer}
.filters input[type=number]{width:52px;background:var(--panel);border:1px solid var(--line);color:var(--txt);
  padding:3px 5px;border-radius:3px;font-family:var(--mono)}
.filters select{background:var(--panel);border:1px solid var(--line);color:var(--txt);
  padding:3px 4px;border-radius:3px;font-family:var(--mono);font-size:12px}
nav{display:flex;gap:4px;padding:6px 16px;flex-wrap:wrap;flex:none;align-items:center}
nav select{background:var(--panel);border:1px solid var(--acc);color:var(--acc);border-radius:5px;
  padding:4px 8px;font:600 13px var(--mono);cursor:pointer}
nav button{background:var(--panel);border:1px solid var(--line);color:var(--txt);
  padding:6px 11px;cursor:pointer;border-radius:4px;font-family:var(--mono);font-size:13px}
nav button.on{border-color:var(--acc);color:var(--acc)}
nav button .n{color:var(--dim);margin-left:6px}
main{flex:1;min-height:0;display:grid;grid-template-columns:minmax(420px,46%) 1fr;grid-template-rows:minmax(0,1fr);gap:0;position:relative}
#tablewrap{overflow:auto;border-right:1px solid var(--line)}
table{border-collapse:collapse;width:100%;font-family:var(--mono);font-size:13.5px}
th,td{padding:7px 9px;text-align:right;white-space:nowrap}
th{position:sticky;top:0;background:var(--panel);color:var(--dim);cursor:pointer;
   border-bottom:1px solid var(--line);font-weight:600}
th:first-child,td:first-child{text-align:left}
tbody tr{cursor:pointer;border-bottom:1px solid #131a22}
tbody tr:hover{background:#16202b}
tbody tr.sel{background:#1a2734;outline:1px solid var(--acc)}
tbody tr.opt{background:rgba(47,191,143,0.10)} tbody tr.opt.sel{background:#1a2734;outline:1px solid var(--acc)}
td.up{color:var(--up)} td.dn{color:var(--dn)}
#mtf{flex:0 0 auto;overflow-x:auto;border-bottom:1px solid var(--line)}
table.mtf{border-collapse:collapse;width:100%;font-family:var(--mono);font-size:12.5px}
table.mtf th,table.mtf td{padding:2px 6px;text-align:center;border-bottom:1px solid #131a22;white-space:nowrap}
table.mtf th{background:var(--bg);color:var(--dim);font-weight:400}
table.mtf td:first-child,table.mtf th:first-child{text-align:left;color:var(--acc)}
table.mtf tbody tr{cursor:pointer} table.mtf tbody tr:hover{background:#16202b}
table.mtf .up{color:var(--up)} table.mtf .dn{color:var(--dn)} table.mtf .muted{color:var(--dim)} table.mtf .amb{color:var(--acc)}
table.mtf tbody tr:nth-child(even){background:rgba(255,255,255,.02)}
table.mtf tbody tr.cur{background:#182231;box-shadow:inset 3px 0 0 var(--acc)}
#chartside{display:flex;flex-direction:column;min-width:0;min-height:0;position:relative}
#firesel{display:none;position:absolute;top:34px;right:14px;z-index:60;background:#0d1319;
  border:1px solid var(--line);border-radius:8px;padding:8px 12px;max-height:360px;overflow:auto;
  min-width:270px;font:12.5px var(--mono);box-shadow:0 8px 24px rgba(0,0,0,.5)}
.fsrow{display:flex;align-items:center;gap:7px;padding:2.5px 0;cursor:pointer;white-space:nowrap}
.fsrow.none{opacity:.4}
.fsw{width:10px;height:10px;border-radius:2px;display:inline-block;flex:0 0 auto}
.fseall{cursor:pointer;color:var(--acc);margin-left:8px}
#charthead{padding:10px 16px;font-family:var(--mono);color:var(--acc);border-bottom:1px solid var(--line)}
#charthead .sub{color:var(--dim);font-size:13px;margin-top:2px}
#chart{flex:1;min-height:0}
.empty{padding:40px;color:var(--dim)}
.tool{background:var(--panel);border:1px solid var(--line);color:var(--txt);cursor:pointer;
  border-radius:4px;font-family:var(--mono);font-size:12.5px;padding:4px 9px;white-space:nowrap}
.tool:hover{border-color:var(--acc);color:var(--acc)}
.tool.busy{opacity:.6;pointer-events:none}
.tool.on{border-color:var(--acc);color:var(--acc)}
#backlive{border-color:var(--acc);color:#0b0f14;background:var(--acc);font-weight:600;margin-left:6px}
#chartctl{flex:0 0 auto;padding:5px 16px 0;display:flex;gap:8px}
#tkrcard{flex:0 0 auto;max-height:300px;overflow:auto;margin:6px 16px;padding:8px 12px;border:1px solid var(--line);
  border-radius:6px;background:var(--panel);font:13.5px var(--mono)}
#tkrcard .trow,#setuppanel .trow{display:flex;gap:8px;margin:3px 0;align-items:baseline}
#tkrcard .tname,#setuppanel .tname{color:var(--acc);min-width:74px;text-transform:uppercase;font-size:11px;letter-spacing:.05em}
#tkrcard .chip,#setuppanel .chip{display:inline-block;border:1px solid var(--line);border-radius:8px;padding:1px 9px;margin-right:5px;color:var(--dim)}
.extb{display:inline-block;margin:0 3px;padding:2px 3px;font-size:16px;vertical-align:middle;cursor:default;line-height:1}
.extb.ok{color:#3d8f5f}.extb.warn{color:#e8b84b}.extb.hot{color:#e05656;font-weight:700;text-shadow:0 0 4px rgba(224,86,86,.6)}
#tkrcard .tsec{border-top:1px solid var(--line);margin-top:6px;padding-top:6px}
.stattile{display:inline-block;text-align:center;margin:0 8px 2px 0;vertical-align:top;min-width:56px}
.stattile .sl{display:block;color:var(--dim);font-size:10px;text-transform:uppercase;letter-spacing:.05em}
.stattile .sv{display:block;font-size:13.5px;line-height:1.25}
.lvlchip{display:inline-block;border-radius:4px;padding:0 7px;margin:1px 5px 1px 0;font-size:12.5px;border:1px solid var(--line)}
.lvlchip.r{color:var(--dn);border-color:rgba(224,86,86,.45)}
.lvlchip.s{color:var(--up);border-color:rgba(47,191,143,.45)}
.lvlchip.px{color:var(--txt);background:var(--bg);font-weight:600}
#tkrcard .newsit{border-left:2px solid var(--line);padding-left:8px;margin:4px 0}
#setuppanel .fdate{cursor:pointer}
#setuppanel .fdate:hover{color:var(--acc);text-decoration:underline}
.rxb{display:inline-block;margin:0 3px;padding:2px 3px;font-size:16px;vertical-align:middle;cursor:default;line-height:1}
.rxb.lo{color:#39c4d8;text-shadow:0 0 4px rgba(57,196,216,.5)}
.rxb.hi{color:#e05656;text-shadow:0 0 4px rgba(224,86,86,.5)}
.rxb.dim{opacity:.45;text-shadow:none}
#tkrcard .hl{display:block;color:var(--txt);font-size:12.5px;margin:2px 0}
#tkrcard .hd,#setuppanel .hd{color:var(--dim);font-size:11.5px}
#bigpanel #tkrcard,#bigpanel #setuppanel{max-height:none;margin:0;border:none;background:transparent}
td.tight{max-width:84px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--dim)}
#bigpanel #mapbody,#bigpanel #perfbody{background:transparent}
#scrim{position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:40;display:none}
#scrim.open{display:block}
#drawer{position:fixed;top:var(--barh,0px);right:0;width:392px;max-width:93vw;height:calc(100vh - var(--barh,0px));background:var(--panel);
  border-left:1px solid var(--line);transform:translateX(100%);transition:transform .18s;z-index:50;
  display:flex;flex-direction:column}
#drawer.open{transform:none}
#drawer header{justify-content:space-between}
#drawer h2{font:600 13px var(--mono);color:var(--acc);letter-spacing:.04em}
#drawerbody{flex:1;overflow:auto;padding:8px 14px;font-family:var(--mono);font-size:12px}
#drawerbody h3{color:var(--acc);font-size:11px;margin:13px 0 4px;text-transform:capitalize;
  border-bottom:1px solid var(--line);padding-bottom:3px;letter-spacing:.03em}
.srow{display:flex;justify-content:space-between;align-items:center;gap:8px;margin:3px 0}
.srow label{color:var(--dim);overflow:hidden;text-overflow:ellipsis}
.srow input[type=number],.srow input[type=text],.srow textarea{background:var(--bg);border:1px solid var(--line);
  color:var(--txt);border-radius:3px;padding:3px 5px;font-family:var(--mono);font-size:12px;width:118px}
.srow textarea{width:160px;height:40px;resize:vertical}
#drawerfoot{display:flex;gap:8px;padding:10px 14px;border-top:1px solid var(--line)}
#drawerfoot .tool{flex:1;text-align:center}
@media(max-width:900px){main{grid-template-columns:1fr;grid-template-rows:45% 55%}}
#mapdrawer{position:fixed;top:var(--barh,0px);left:0;width:920px;max-width:96vw;height:calc(100vh - var(--barh,0px));background:var(--panel);
  border-right:1px solid var(--line);transform:translateX(-102%);transition:transform .18s ease;z-index:50;display:flex;flex-direction:column}
#mapdrawer.open{transform:none}
#mapdrawer header{padding:10px 14px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;gap:10px}
#mapdrawer h2{font:600 13px var(--mono);color:var(--acc);letter-spacing:.04em;margin:0}
#mapbody{flex:1;overflow:auto;padding:10px 14px;font:13px var(--mono)}
#mapbody h3{color:var(--acc);font-size:12px;margin:16px 0 7px;text-transform:uppercase;letter-spacing:.06em;
  border-bottom:1px solid rgba(94,234,212,.22);padding-bottom:4px}
.mapcards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}
.mapcard{border:1px solid var(--line);border-radius:6px;padding:7px 9px;cursor:pointer;background:var(--bg)}
.mapcard:hover{border-color:var(--acc)}
.mapcard .t{display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;flex-wrap:wrap;gap:2px 6px}
.mapcard .sym{font-weight:700}
.mapcard .phrase{color:var(--dim);font-size:12px;line-height:1.35;min-height:44px}
.mapcard .lvl{font-size:12px;margin-top:4px}
.badge{padding:1px 7px;border-radius:3px;font-size:11.5px;font-weight:700;letter-spacing:.04em}
.badge.long{background:rgba(47,191,143,.16);color:var(--up)}
.badge.short{background:rgba(224,90,109,.16);color:var(--dn)}
.badge.avoid{background:rgba(224,164,58,.14);color:#e0a43a}
.badge.neutral{background:rgba(138,160,184,.14);color:var(--dim)}
.gapchip{font-size:11px;padding:0 5px;border-radius:3px;border:1px solid var(--line);color:var(--dim);white-space:nowrap}
.gapchip.holding{color:var(--up);border-color:var(--up)}
.gapchip.faded,.gapchip.filled{color:#e0a43a;border-color:#e0a43a}
.calltbl .call{border:1px solid var(--line);border-left-width:3px;border-radius:5px;padding:6px 9px;margin:5px 0;background:var(--bg)}
.calltbl .call.long{border-left-color:var(--up)}
.calltbl .call.short{border-left-color:var(--dn)}
.calltbl .call.avoid{border-left-color:#e0a43a}
.calltbl .cphrase{margin:2px 0 1px;line-height:1.4}
.calltbl .cexpect{color:var(--dim);font-size:11px;line-height:1.35}
.calltbl .cstats{float:right;color:var(--dim);font-size:10px;border:1px dashed var(--line);border-radius:3px;padding:0 5px;margin-left:8px}
.ratiostrip{display:flex;flex-wrap:wrap;gap:6px}
.rtile{border:1px solid var(--line);border-radius:5px;padding:4px 8px;font-size:12px;background:var(--bg)}
.rtile .rp{color:var(--dim);font-size:11px}
.maptbl{width:100%;border-collapse:collapse;font-size:12px}
.maptbl td,.maptbl th{padding:2px 7px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
.maptbl th{color:var(--dim);font-weight:400}
.mapmode button{background:none;border:1px solid var(--line);color:var(--dim);font:11px var(--mono);padding:2px 10px;border-radius:3px;cursor:pointer;margin-right:6px}
.mapmode button.active{color:var(--acc);border-color:var(--acc)}
.mcall{padding:1px 7px;border-radius:3px;font-weight:700;font-size:11px;margin:0 6px}
.mcall.up{background:rgba(47,191,143,.16);color:var(--up)}
.mcall.dn{background:rgba(224,90,109,.16);color:var(--dn)}
.mcall.amb{background:rgba(224,164,58,.14);color:#e0a43a}
#perfdrawer{position:fixed;top:var(--barh,0px);left:0;width:560px;max-width:96vw;height:calc(100vh - var(--barh,0px));background:var(--panel);
  border-right:1px solid var(--line);transform:translateX(-100%);transition:transform .18s;z-index:50;
  display:flex;flex-direction:column}
#perfdrawer.open{transform:none}
#perfdrawer header{padding:10px 14px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between}
#perfdrawer h2{font:600 13px var(--mono);color:var(--acc);letter-spacing:.04em;margin:0}
#perfbody{flex:1;overflow:auto;padding:0}
table.perf{border-collapse:collapse;width:100%;font-family:var(--mono);font-size:11.5px}
table.perf th,table.perf td{padding:3px 8px;text-align:right;border-bottom:1px solid #131a22;white-space:nowrap}
table.perf th{background:var(--bg);color:var(--dim);font-weight:400;cursor:pointer;position:sticky;top:0}
table.perf th:hover{color:var(--txt)}
table.perf th.sorted{color:var(--acc)}
table.perf td:first-child,table.perf th:first-child{text-align:left;min-width:130px}
table.perf td:last-child,table.perf th:last-child{color:var(--dim);font-size:10px}
table.perf .sec{color:var(--dim);font-size:10px;padding-left:12px}
.perf-section{padding:4px 8px 2px;font-size:10px;color:var(--dim);background:var(--bg);
  border-bottom:1px solid var(--line);letter-spacing:.05em;text-transform:uppercase}
#perfmode{display:flex;gap:4px;padding:6px 8px;border-bottom:1px solid var(--line)}
#perfmode button{font-family:var(--mono);font-size:11px;padding:2px 8px;cursor:pointer;
  background:var(--bg);border:1px solid var(--line);color:var(--dim);border-radius:3px}
#perfmode button.active{border-color:var(--acc);color:var(--acc)}
/* ── journal ── */
#jmodal-bg{display:none;position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:200;align-items:center;justify-content:center}
#jmodal-bg.open{display:flex}
#jmodal{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:18px 20px;
  min-width:320px;max-width:480px;width:90vw;font-family:var(--mono);font-size:12px;display:flex;flex-direction:column;gap:10px}
#jmodal h3{margin:0;font-size:13px;color:var(--acc);font-weight:600;letter-spacing:.04em}
.jrow{display:flex;flex-direction:column;gap:3px}
.jrow label{color:var(--dim);font-size:11px}
.jrow input,.jrow select,.jrow textarea{background:var(--bg);border:1px solid var(--line);color:var(--txt);
  border-radius:3px;padding:4px 7px;font-family:var(--mono);font-size:12px;width:100%;box-sizing:border-box}
.jrow textarea{height:72px;resize:vertical}
.jbtns{display:flex;gap:8px;justify-content:flex-end;margin-top:4px}
.jbtns .tool{padding:4px 14px}
/* inline "+" button before ticker symbol */
button.jadd{background:none;border:1px solid var(--line);border-radius:3px;color:var(--dim);
  font-size:10px;padding:1px 5px;cursor:pointer;line-height:1;flex-shrink:0;margin-right:5px;vertical-align:middle}
button.jadd:hover{border-color:var(--acc);color:var(--acc)}
/* setup rows in modal */
.jsetup-row{display:flex;gap:5px;align-items:center;margin:3px 0}
.jsetup-row select{background:var(--bg);border:1px solid var(--line);color:var(--txt);
  font-family:var(--mono);font-size:11px;padding:3px 5px;border-radius:3px;flex:1}
.jsetup-row input[type=date]{background:var(--bg);border:1px solid var(--line);color:var(--txt);
  font-family:var(--mono);font-size:11px;padding:3px 5px;border-radius:3px;width:116px}
.jsetup-row .jpick{background:none;border:1px solid var(--line);color:var(--dim);
  border-radius:3px;font-size:12px;padding:2px 6px;cursor:pointer}
.jsetup-row .jpick.active{border-color:#e8b84b;color:#e8b84b}
.jsetup-row .jpick:hover{border-color:var(--acc);color:var(--acc)}
.jsetup-row .jrm{background:none;border:none;color:var(--dim);cursor:pointer;font-size:14px;padding:0 3px}
.jsetup-row .jrm:hover{color:#e05050}
/* journal browser drawer */
#jdrawer{position:fixed;top:var(--barh,0px);right:0;width:500px;max-width:96vw;height:calc(100vh - var(--barh,0px));background:var(--panel);
  border-left:1px solid var(--line);transform:translateX(100%);transition:transform .18s;z-index:50;
  display:flex;flex-direction:column}
#jdrawer.open{transform:none}
#jdrawer header{padding:10px 14px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between}
#jdrawer h2{font:600 13px var(--mono);color:var(--acc);letter-spacing:.04em;margin:0}
#jfilter{display:flex;gap:6px;padding:6px 10px;border-bottom:1px solid var(--line);align-items:center}
#jfilter select{background:var(--bg);border:1px solid var(--line);color:var(--txt);
  font-family:var(--mono);font-size:11px;padding:2px 5px;border-radius:3px}
#jbody{flex:1;overflow:auto}
.jentry{display:flex;gap:8px;padding:8px 12px;border-bottom:1px solid #131a22;align-items:flex-start;cursor:pointer}
.jentry:hover{background:var(--bg)}
.jentry .jsym{font-weight:700;color:var(--txt);min-width:52px;font-size:12px}
.jentry .jdate{color:var(--dim);font-size:10px;white-space:nowrap}
.jentry .jsetup{font-size:10px;color:var(--acc);padding:1px 5px;border:1px solid var(--acc);
  border-radius:3px;white-space:nowrap}
.jentry .jcomm{flex:1;color:var(--dim);font-size:11px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.jentry .jdel{background:none;border:none;color:var(--dim);cursor:pointer;font-size:14px;flex-shrink:0;padding:0 2px}
.jentry .jdel:hover{color:#e05050}
.jedit-wrap{padding:8px 12px;border-bottom:1px solid var(--line);background:#0e1620}
.jedit-wrap textarea{width:100%;box-sizing:border-box;background:var(--bg);border:1px solid var(--line);
  color:var(--txt);font-family:var(--mono);font-size:11px;resize:vertical;height:56px;border-radius:3px;padding:4px 6px}
.jedit-wrap .jbtns{margin-top:6px}
.jmarker{position:absolute;top:0;bottom:0;width:2px;background:#e8b84b;pointer-events:none;z-index:5}
</style></head><body>
"""

HTML_BODY = r"""<header>
  <h1>SETUP SCANNER</h1><span class="d">__DATE__</span>__PICKER__<button class="tool" id="btnperf" title="Sector &amp; theme performance ranking across timeframes">&#9654; Perf</button><span class="d" id="totals"></span>
  <span class="filters">
    $volM<input id="fvol" type="number" value="0" step="1">
    avgK<input id="favol" type="number" value="0" step="100">
    px<input id="fprice" type="number" value="0" step="1">
    ADR<input id="fadr" type="number" value="0" step="0.5">
    %&gt;Lo<input id="flo52" type="number" value="0" step="5">
    offATH<input id="fhi" type="number" value="0" step="2" title="max % below the all-time high over loaded history (within X% of ATH; 0 = off)">
    RS&ge;<input id="frs" type="number" value="0" step="5" title="min RS rank = best percentile across the 1/3/6-mo raw-return lists (top of ANY list) — Qullamaggie's 'stocks up the most over 1/3/6mo' screen; top ~1-2% in some timeframe = 98-99. Sort the rs_1m/rs_3m/rs_6m columns to see a single list.">
    <select id="fstate" title="building = set up, not triggered yet; breakout = crossed its trigger on the latest candle">
      <option value="">any state</option><option value="building">building</option><option value="breakout">breakout</option><option value="armed">armed</option><option value="entry1">entry1</option><option value="entry2">entry2</option>
    </select>
    <select id="fetf" title="common stocks only, ETFs only, or all">
      <option value="">all</option><option value="stock">stocks</option><option value="etf">ETFs</option>
    </select>
    <label id="fpatwrap" style="display:none">pat<select id="fpat" title="filter by consolidation shape">
      <option value="">all</option><option value="flag">flag</option><option value="pennant">pennant</option>
      <option value="rising_triangle">rising_tri</option><option value="channel">channel</option><option value="other">other</option>
    </select></label>
    <label><input id="fsma50" type="checkbox">&gt;50</label>
    <label><input id="fsma200" type="checkbox">&gt;200</label>
    <span id="femawrap" style="display:none;margin-left:10px;padding-left:10px;border-left:1px solid var(--line);">
      scope<select id="fema_scope" title="ticker or synthetic ETF"><option value="">all</option><option value="ticker">ticker</option><option value="synth">synth</option></select>
      base<select id="fema_base" title="absolute or relative to SPY"><option value="">all</option><option value="abs">abs</option><option value="spy">vs SPY</option></select>
      dir<select id="fema_dir" title="flip direction"><option value="">all</option><option value="yellow">yellow</option><option value="blue">blue</option><option value="grey">grey</option></select>
      sec vs SPY<select id="fema_sec_state" title="sector relative state"><option value="">all</option><option value="yellow">yellow</option><option value="blue">blue</option><option value="grey">grey</option></select>
      thm vs SPY<select id="fema_thm_state" title="theme relative state"><option value="">all</option><option value="yellow">yellow</option><option value="blue">blue</option><option value="grey">grey</option></select>
      time<select id="fema_time" title="when did the flip happen"><option value="">all</option><option value="0">today</option><option value="past_week">past week</option></select>
    </span>
    <span id="rowcount" class="d"></span>
    <button class="tool" id="btnbest" title="Best-results preset for the CURRENT setup tab">Best</button>
    <button class="tool" id="btntv" title="Download the current setup's filtered + sorted list as a TradingView watchlist (.txt)">&#11015; TV</button>
    <button class="tool" id="btnjournal" title="Chart study journal">&#128218; Journal</button>
    __TOOLS__
  </span>
</header>
<div class="mbar" style="display:flex;align-items:flex-start;gap:8px">
  <button class="tool" id="btnmap" style="margin:4px 0" title="Situational-awareness map: sequences, levels, gaps, rotation &amp; suggestions">&#129517; Map<span id="mapbadge" style="display:none;margin-left:5px;background:#e05a6d;color:#fff;border-radius:8px;padding:0 5px;font-size:10px;font-weight:700"></span></button>
  <button class="tool" id="btnthemes" style="margin:4px 0" title="Theme leaders board: every sector &amp; theme ranked by average member move over Today/1W/1M/3M/YTD">&#127937; Themes</button>
  <span style="margin:4px 0;display:inline-flex;align-items:center;gap:5px;white-space:nowrap">
    <input id="tkrsearch" list="symlist" placeholder="&#128269; ticker&#8230;" autocomplete="off" spellcheck="false"
      style="background:var(--panel);border:1px solid var(--line);color:var(--txt);font:12.5px var(--mono);padding:3px 7px;border-radius:3px;width:120px"
      title="Search a ticker to open its chart + info (loaded universe)">
    <datalist id="symlist"></datalist>
    <span id="tkrsearchmsg" style="font:11px var(--mono);color:var(--dn)"></span>
  </span>
  <div id="marketbar" style="flex:1;min-width:0"></div>
</div>
<nav id="tabs"></nav>
<main>
  <div id="setuppanel" style="display:none;position:absolute;top:6px;left:14px;z-index:20;max-width:min(600px,calc(46% - 28px));max-height:calc(100% - 12px);overflow:auto;padding:8px 12px;border:1px solid var(--line);border-radius:6px;background:var(--panel);box-shadow:0 10px 28px rgba(0,0,0,.55);font:12px var(--mono)"></div>
  <div id="tablewrap"><div class="empty">No hits.</div></div>
  <div id="chartside">
    <div id="charthead">select a row<div class="sub"></div></div>
    <div id="chartctl"><button class="tool on" id="btnsr" title="Toggle support/resistance lines">&#9101; S/R</button><button class="tool" id="btnfires" title="Mark PAST setup triggers on this chart (from the setup registries) — pick which setups">&#9678; Fires</button><button class="tool" id="btnrider" title="EMA-Rider overlay: streak shading + ⚠ armed / ✓ saved / ⚡ break marks (any timeframe)">&#9889; Rider</button><button class="tool" id="btntkr" title="Ticker card: sector/themes, performance, structure, live news">&#9432; Ticker</button><button class="tool" title="expand the chart to full screen" onclick="expandEl(document.getElementById('chartside'),document.getElementById('charthead').textContent,1)">&#x26F6; Chart</button></div>
    <div id="firesel"></div>
    <div id="tkrcard" style="display:none"></div>
    <div id="quadmodal" style="display:none;position:fixed;inset:0;z-index:200;background:rgba(6,9,13,.93);flex-direction:column;padding:24px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <b style="color:var(--acc);font:600 14px var(--mono)">ROTATION QUADRANT</b>
        <button class="tool" onclick="document.getElementById('quadmodal').style.display='none'">✕ close</button></div>
      <div class="qbody" style="flex:1;overflow:auto"></div></div>
    <div id="mtf"></div>
    <div id="chart"></div>
  </div>
</main>
__DRAWER__
<div id="mapdrawer">
  <header>
    <h2>&#129517; MARKET MAP</h2>
    <span id="mapasof" style="color:var(--dim);font:11px var(--mono)"></span>
    <button class="tool" onclick="expandEl(document.getElementById('mapbody'),'&#129517; MARKET MAP')" title="expand to full screen">&#x26F6;</button>
    <button class="tool" id="mapclose">&#x2715;</button>
  </header>
  <div id="mapbody"><div class="empty">loading&#8230;</div></div>
</div>
<div id="bigpanel" style="display:none;position:fixed;inset:0;z-index:300;background:rgba(6,9,13,.96);flex-direction:column;padding:22px">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
    <b class="btitle" style="color:var(--acc);font:600 16px var(--mono)"></b>
    <button class="tool" onclick="closeBig()">&#x2715; close</button></div>
  <div class="bbody" style="flex:1;overflow:auto;zoom:1.45"></div>
</div>
<div id="epnewsmodal" style="display:none;position:fixed;inset:0;z-index:320;background:rgba(6,9,13,.9);align-items:center;justify-content:center" onclick="if(event.target===this)this.style.display='none'">
  <div style="max-width:660px;width:92%;max-height:82vh;overflow:auto;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:15px 18px">
    <div style="display:flex;align-items:center;margin-bottom:8px">
      <b class="epnh" style="font:600 15px var(--mono);color:var(--acc)"></b>
      <span class="chip" style="cursor:pointer;margin-left:auto" onclick="document.getElementById('epnewsmodal').style.display='none'">&#x2715; close</span></div>
    <div class="epnb"></div>
  </div>
</div>
<div id="perfdrawer">
  <header>
    <h2>&#9654; SECTOR &amp; THEME PERFORMANCE</h2>
    <button class="tool" onclick="expandEl(document.getElementById('perfbody'),'&#9654; SECTOR &amp; THEME PERFORMANCE')" title="expand to full screen">&#x26F6;</button>
    <button class="tool" id="perfclose">&#x2715;</button>
  </header>
  <div id="perfmode">
    <button class="active" data-view="themes">Themes</button>
    <button data-view="sectors">Sectors</button>
    <button data-view="both">Both</button>
    <button data-view="futures">Futures</button>
    <button data-view="universe">Universe</button>
    <span style="flex:1"></span>
    <span style="font-family:var(--mono);font-size:10px;color:var(--dim)" id="perfsig"></span>
  </div>
  <div id="perfbody"><div class="empty">Loading&#8230;</div></div>
</div>

<!-- journal save modal -->
<div id="jmodal-bg">
  <div id="jmodal">
    <h3>&#128218; Save to Journal</h3>
    <div style="display:flex;gap:10px;flex-wrap:wrap">
      <div class="jrow" style="flex:1;min-width:80px"><label>Symbol</label><input id="jsym" readonly></div>
      <div class="jrow" style="flex:1;min-width:100px"><label>Viewing date</label><input id="jdate" readonly></div>
      <div class="jrow" style="flex:0;min-width:60px"><label>TF</label><input id="jtf" readonly style="width:52px"></div>
    </div>
    <div style="margin:4px 0 2px;font-size:11px;color:var(--dim)">Setups — click &#128205; then click a bar on the chart to capture setup date</div>
    <div id="jsetup-rows"></div>
    <button class="tool" style="font-size:10px;padding:2px 10px;margin-top:4px" onclick="addJSetupRow()">+ Add setup</button>
    <div id="jpick-hint" style="display:none;font-size:10px;color:#e8b84b;padding:3px 0">&#9654; Click a bar on the chart&#8230;</div>
    <div class="jrow" style="margin-top:6px"><label>Comments</label><textarea id="jcomments" placeholder="Notes about this chart&#8230;"></textarea></div>
    <div class="jbtns">
      <button class="tool" onclick="closeJModal()">Cancel</button>
      <button class="tool" onclick="saveJEntry()" style="border-color:var(--acc);color:var(--acc)">Save</button>
    </div>
  </div>
</div>

<!-- journal browser drawer -->
<div id="jdrawer">
  <header>
    <h2>&#128218; JOURNAL</h2>
    <button class="tool" id="jclose">&#x2715;</button>
  </header>
  <div id="jfilter">
    <span style="color:var(--dim);font-size:11px;font-family:var(--mono)">Filter:</span>
    <select id="jfsetup" onchange="renderJournal(jData,this.value)">
      <option value="">All Setups</option>
    </select>
    <button class="tool" style="font-size:10px;padding:2px 8px" onclick="loadJournal()">&#8635;</button>
  </div>
  <div id="jbody"><div class="empty">Loading&#8230;</div></div>
</div>
"""

DASHBOARD_JS = r"""
let TABLES={}, CHARTS={}, cur=Object.keys(LABELS)[0], sortCol='quality', sortAsc=false, chart=null, selKey=null;
// ── scan cache in IndexedDB (the tables payload is ~9MB — over localStorage's ~5MB cap) ──
// reload restores the last scan instantly, then revalidates in the background (stale-while-revalidate),
// so a reload no longer shows a blank "scanning…" wait. Keyed by date; the newest few dates are kept.
const IDB_DB='scannerCache', IDB_STORE='scan';
function idbOpen(){return new Promise((res,rej)=>{let r;try{r=indexedDB.open(IDB_DB,1);}catch(e){return rej(e);}
  r.onupgradeneeded=()=>{if(!r.result.objectStoreNames.contains(IDB_STORE))r.result.createObjectStore(IDB_STORE);};
  r.onsuccess=()=>res(r.result); r.onerror=()=>rej(r.error);});}
async function idbGet(key){try{const db=await idbOpen();return await new Promise((res,rej)=>{
  const q=db.transaction(IDB_STORE,'readonly').objectStore(IDB_STORE).get(key);
  q.onsuccess=()=>res(q.result||null); q.onerror=()=>rej(q.error);});}catch(e){return null;}}
async function idbPut(key,val){try{const db=await idbOpen();await new Promise((res,rej)=>{
  const tx=db.transaction(IDB_STORE,'readwrite'); tx.objectStore(IDB_STORE).put(val,key);
  tx.oncomplete=res; tx.onerror=()=>rej(tx.error);}); idbPrune(5);}catch(e){}}
async function idbPrune(keep){try{const db=await idbOpen();const os=db.transaction(IDB_STORE,'readwrite').objectStore(IDB_STORE);
  const keys=await new Promise(r=>{const q=os.getAllKeys();q.onsuccess=()=>r(q.result||[]);q.onerror=()=>r([]);});
  keys.filter(k=>String(k).startsWith('scan:')).sort().slice(0,-keep).forEach(k=>os.delete(k));}catch(e){}}
let riderTF='1D', BASE_RIDERS=null;   // EMA Rider timeframe + the 1D baseline tables (to restore)
const RIDER_TABS=['ema_rider_bull','ema_rider_bear'];
function qcolor(q){const t=Math.max(0,Math.min(100,q))/100;  // dn-red -> up-green ramp
  return'rgb('+Math.round(224+(47-224)*t)+','+Math.round(90+(191-90)*t)+','+Math.round(109+(143-109)*t)+')';}
function applyData(t,c){TABLES=t;CHARTS=c;if(!TABLES[cur])cur=Object.keys(LABELS)[0];
  BASE_RIDERS={ema_rider_bull:t.ema_rider_bull||[],ema_rider_bear:t.ema_rider_bear||[]};  // 1D baseline
  riderTF='1D';const ts=document.getElementById('tfsel');if(ts)ts.value='1D';
  selKey=null;counts();render();}
function counts(){let t=0;for(const s in LABELS)t+=(TABLES[s]||[]).length;
  document.getElementById('totals').textContent=t+' hits';}
function tabs(){const el=document.getElementById('tabs');el.innerHTML='';
  // 19+ setups no longer fit as a tab strip — a dropdown selects the setup (Amir 2026-07-03)
  const sel=document.createElement('select');sel.id='setupsel';
  for(const s in LABELS){const o=document.createElement('option');
    o.value=s;o.textContent=LABELS[s]+'  ('+(TABLES[s]||[]).length+')';
    if(s===cur)o.selected=true;sel.appendChild(o);}
  sel.onchange=()=>{cur=sel.value;sortCol='quality';sortAsc=false;render();};
  el.appendChild(sel);
  // quick-jump buttons for the most-used setups stay one click away
  for(const s of ['episodic_pivot','backburner','flat_base','qm_breakout','cup_handle']){
    if(!(s in LABELS))continue;
    const b=document.createElement('button');
    b.innerHTML=LABELS[s]+'<span class="n">'+(TABLES[s]||[]).length+'</span>';
    if(s===cur)b.classList.add('on');
    b.onclick=()=>{cur=s;sortCol='quality';sortAsc=false;render();};el.appendChild(b);}
  const bx=document.createElement('button');bx.textContent='⛶ Table';
  bx.title='expand the results table to full screen (larger fonts)';
  bx.onclick=()=>expandEl(document.getElementById('tablewrap'),(LABELS[cur]||cur)+' — results',1.3);
  el.appendChild(bx);
  const bi=document.createElement('button');bi.textContent='? Info';
  bi.title='what this setup is + how to trade it';
  bi.onclick=()=>toggleSetupPanel('info');el.appendChild(bi);
  const bh=document.createElement('button');bh.textContent='☰ History';
  bh.title='every ticker that ever triggered this setup (precomputed registry)';
  bh.onclick=()=>toggleSetupPanel('hist');el.appendChild(bh);
  if(cur==='backburner'){                              // today's BB entries at a chosen RSI level
    const s2=document.createElement('select');s2.title="today's backburner RSI trigger level (recomputes the scan)";
    for(const v of [30,25,20]){const o=document.createElement('option');o.value=v;o.textContent='RSI '+v;
      if(v===(window.bbRsiLevel||30))o.selected=true;s2.appendChild(o);}
    s2.onchange=async()=>{window.bbRsiLevel=+s2.value;
      await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({overrides:{BB_RSI_ENTRY1:+s2.value}})});
      const di=document.getElementById('asof');if(di&&di.value&&typeof load==='function')load(di.value);};
    el.appendChild(s2);}}
let panelMode=null,panelFor=null;
function toggleSetupPanel(mode){const el=document.getElementById('setuppanel');
  if(panelMode===mode&&el.style.display!=='none'){el.style.display='none';panelMode=null;panelFor=null;return;}
  panelMode=mode;renderSetupPanel();}
function renderSetupPanel(){const el=document.getElementById('setuppanel'),mode=panelMode;
  if(!mode)return;
  el.style.display='block';panelFor=cur;
  if(mode==='info'){const d=DOCS[cur]||{};
    el.innerHTML='<div class="trow"><b style="color:var(--acc)">'+(LABELS[cur]||cur)+'</b>'+expBtn('setuppanel',LABELS[cur]||cur)+'</div>'
      +'<div class="trow"><span class="tname">what</span><span>'+(d.what||'no description yet')+'</span></div>'
      +'<div class="trow"><span class="tname">execution</span><span>'+(d.exec||'')+'</span></div>';return;}
  el.innerHTML='<div class="hd">loading history…</div>';
  fetch('/api/setup_history?setup='+encodeURIComponent(cur)+(cur==='backburner'?'&rsi='+histRsi:''))
    .then(r=>r.json()).then(j=>{
    if(panelMode!=='hist')return;
    if(j.error){el.innerHTML='<div class="hd">'+j.error+'</div>';return;}
    let h='<div class="trow"><b style="color:var(--acc)">'+(LABELS[cur]||cur)+' — full history</b>'
      +'<span class="hd">'+j.n_fires+' fires · '+j.n_symbols+' tickers · built '+j.built+' · click a ticker (or a date) to time-travel the app to that fire</span>'
      +expBtn('setuppanel',(LABELS[cur]||cur)+' — history')+'</div>';
    if(cur==='backburner'){                             // BB fires are TF-fractal — chart any timeframe
      h+='<div class="trow"><span class="tname">chart TF</span>'
        +['5m','15m','1h','4h','12h','1D','1W'].map(t=>'<span class="chip" style="cursor:pointer'
          +(histTf===t?';border-color:var(--acc);color:var(--acc)':'')
          +'" onclick="histTf=\''+t+'\';renderSetupPanel();if(lastHistSym)histChart(lastHistSym,[])">'+t+'</span>').join('')
        +'<span class="tname" style="margin-left:10px">rsi level</span>'
        +[30,25,20].map(v=>'<span class="chip" style="cursor:pointer'
          +(histRsi===v?';border-color:var(--acc);color:var(--acc)':'')
          +'" onclick="histRsi='+v+';renderSetupPanel();if(lastHistSym)histChart(lastHistSym,[])">'+v+'</span>').join('')
        +'<span class="hd">list = the chosen level\'s daily registry; chart = the chosen TF\'s triggers with the pOS price</span></div>';}
    window.histSyms={};j.symbols.forEach(s=>{histSyms[s.symbol]=s.fires;});
    h+=j.symbols.slice(0,400).map(s=>'<div class="trow" style="cursor:pointer" '
      +'onclick="histChartAt(\''+s.symbol+'\',\''+String(s.last).slice(0,10)+'\')">'   // time-travel to the ticker's latest fire
      +'<b style="min-width:56px">'+s.symbol+extSlot(s.symbol)+'</b><span class="hd">'+s.n+' fires · last '+s.last+'</span>'
      +'<span class="hd" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'
      +s.fires.slice(-8).map(f=>'<span class="fdate" onclick="event.stopPropagation();histChartAt(\''
        +s.symbol+'\',\''+f.date+'\')" title="chart + multi-TF as of THIS fire">'+f.date+'</span>').join(' · ')
      +'</span></div>').join('');
    el.innerHTML=h;
    extFetch(j.symbols.slice(0,400).map(s=>s.symbol)).then(()=>extPatch(el));   // current health per name
  }).catch(e=>{el.innerHTML='<div class="hd">history failed: '+e+'</div>';});}
let histTf='1D',histRsi=30,lastHistSym=null;
function histChartAt(sym,date){   // pick ONE fire: everything (marks, anatomy, multi-TF) as of it
  const all=(window.histSyms||{})[sym]||[];
  const p=histChart(sym,all.filter(f=>String(f.date)<=date));
  const d=String(date).slice(0,10),di=document.getElementById('asof');   // TIME-TRAVEL: the whole app
  if(di&&di.value!==d&&typeof load==='function'){di.value=d;            // (scan/awareness/map/perf) as-of
    Promise.resolve(p).finally(()=>load(d));}}   // ...AFTER the chart lands (a cold-date rescan holds the scan lock ~1min)
async function histChart(sym,fires){lastHistSym=sym;
  const tc=document.getElementById('tkrcard');if(tc)tc.style.display='block';
  loadTicker(sym);
  const fireDate=(fires&&fires.length)?fires[fires.length-1].date:null;   // MTF pane reads AS OF
  if(fireDate&&fireDate.length>=10)renderMtf(sym,fireDate.slice(0,10));   // the breakout bar (PIT)
  if(cur==='backburner'){                               // fires recomputed on the chosen TF + RSI level
    try{const j=await(await fetch('/api/bb_fires?symbol='+encodeURIComponent(sym)
      +'&tf='+encodeURIComponent(histTf)+'&rsi='+histRsi)).json();
      window.histMarks={sym:sym,fires:j.fires||[]};}   // MTF stays on the CHOSEN fire's date
    catch(e){window.histMarks={sym:sym,fires:[]};}
    chartTF(sym,histTf,(/[mh]$/.test(histTf)?null:fireDate&&fireDate.slice(0,10)));return;}
  window.histMarks={sym:sym,fires:fires};
  const TPL=['cup_handle','double_top','head_shoulders','inverse_hs'];
  if(TPL.includes(cur)&&fires.length){                  // re-detect at the fire date -> full anatomy
    const last=fires[fires.length-1].date;
    let hit={};
    try{const j=await(await fetch('/api/setup_hit?symbol='+encodeURIComponent(sym)
      +'&setup='+encodeURIComponent(cur)+'&date='+encodeURIComponent(last))).json();hit=j.hit||{};}catch(e){}
    try{const r=await fetch('/api/chart',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({symbol:sym,tf:'1D',date:last,hit:hit})});
      const j2=await r.json();
      if(j2&&!j2.error){j2.symbol=sym;
        document.getElementById('charthead').innerHTML=sym+'  —  '+(LABELS[cur]||cur)
          +' history<div class="sub">pattern as of '+last+' ('+fires.length+' fires marked)</div>';
        drawChart(j2);return;}}catch(e){}}
  chartTF(sym,'1D',fireDate&&fireDate.slice(0,10));}   // pan/zoom to the CHOSEN fire
// state display: icon + friendly name (data values untouched — display only)
const STLBL={breakout_lowvol:'breakout ~vol',too_short:'young'};
const STTIP={breakout_lowvol:'breakout on below-average volume',too_short:'not enough history yet',
  riding:'clean ride along the EMA',touching:'wick tagged the EMA but closed back on-side — the ride held',
  armed:'closed the WRONG side of the EMA — an exit is armed, but the ride is still alive',
  saved:'the armed exit was defended — price closed back on-side, ride continues',
  break:'the ride broke — the armed bar’s extreme was taken out and price flipped to the other side of the EMA'};
const STICON={breakout:'⤴',breakout_lowvol:'⤴',building:'▭',armed:'👁',watch:'👁',riding:'➟',
  optimal:'★',touching:'✛',reversal:'⤵',breakdown:'⤵',saved:'✓',break:'⚡'};
function stDisp(v){if(v==null||v==='')return '';return (STICON[v]?STICON[v]+' ':'')+(STLBL[v]||v);}
function fmt(v){if(v===null||v===undefined||v==='')return'';
  if(typeof v==='boolean')return v?'✓':'·';
  if(typeof v==='number')return Math.abs(v)>=1000?v.toLocaleString():v;return v;}
function mcap(v){if(v==null||v==='')return'';const t=v/1e12,b=v/1e9;   // dollars -> human T/B/M
  return t>=1?t.toFixed(t>=10?0:1)+'T':(b>=1?b.toFixed(b>=10?0:1)+'B':(v/1e6).toFixed(0)+'M');}
// ── Extension Health Badge: ATR-multiples from the 50-MA, framed by the goggles ──────────
// Only stretch on the BIAS side counts as caution (bull goggles: above the mean; bear: below) —
// the other side is a pullback, not chase-risk. Tiers per instrument family come from EXTQ.
// ● = the current timeframe · ◆ = the worst timeframe (hover either for the numbers).
function extEff(v){return mtfGoggles==='bear'?Math.max(0,-v):Math.max(0,v);}
function extCls(v,kind){const q=EXTQ[kind]||EXTQ.stock,e=extEff(v);return e<q[0]?'ok':(e<q[1]?'warn':'hot');}
function extWord(c){return c==='ok'?'healthy':(c==='warn'?'getting extended — caution':'EXTREME extension');}
function extBadge(v,kind,lbl){if(v==null)return '';const c=extCls(v,kind);
  const t=(lbl||'1D')+' extension '+(v>0?'+':'')+v.toFixed(1)+'×ATR from 50MA — '+extWord(c)
    +' ('+mtfGoggles+' goggles · '+(kind==='etf'?'ETF/index':'stock')+' tiers)';
  return '<span class="extb '+c+'" title="'+t.replace(/"/g,'&quot;')+'">●</span>';}
function rsixBadge(m){if(!m)return '';   // ◉ RSI historical-extreme badge (rsi_extreme.pine port)
  const ZL={atl:'AT its all-time-LOW RSI',ath:'AT its all-time-HIGH RSI',
    near_atl:'approaching its all-time-low RSI zone',near_ath:'approaching its all-time-high RSI zone'};
  let best=null,sc=-1;
  for(const tf in m){const s=m[tf];
    const p=(s.z==='atl'||s.z==='ath'?4:(s.z&&s.z.startsWith('near')?2.5:(s.z?2:1)))+(s.x?0.5:0);
    if(p>sc){sc=p;best=s;}}
  if(!best)return '';
  const lowside=/atl|reclo|rec_lo/.test(best.z||best.le||'');
  const at=best.z==='atl'||best.z==='ath';
  const cls=(lowside?'lo':'hi')+(at?'':' dim');   // only AT-the-extreme lights the badge fully
  const lines=Object.keys(m).map(tf=>{const s=m[tf];
    return tf+': '+(s.z?ZL[s.z]:('near an extreme '+s.b+' bars ago'))
      +(s.r!=null?' · rsi '+s.r:'')+(s.x?' · BROKE the all-time extreme':'');}).join('  |  ');
  return '<span class="rxb '+cls+'" title="'+('RSI historical extremes — '+lines).replace(/"/g,'&quot;')+'">◉</span>';}
function extBadges(tfs,kind,curTf,rsix){let h='';
  if(tfs){const cur=(curTf&&tfs[curTf]!=null)?curTf:'1D';
    h=extBadge(tfs[cur],kind,cur);
    const ks=Object.keys(tfs);
    if(ks.length>1){let wt=null,we=-1;for(const tf of ks){const e=extEff(tfs[tf]);if(e>we){we=e;wt=tf;}}
      const c=extCls(tfs[wt],kind),all=ks.map(tf=>tf+' '+(tfs[tf]>0?'+':'')+tfs[tf].toFixed(1)+'×').join(' · ');
      h+='<span class="extb '+c+'" style="font-size:13px" title="'+('worst TF: '+wt+' '
        +(tfs[wt]>0?'+':'')+tfs[wt].toFixed(1)+'×ATR from 50MA — '+extWord(c)+' · all: '+all).replace(/"/g,'&quot;')+'">◆</span>';}}
  return h+rsixBadge(rsix);}
window.EXTD={};   // /api/ext back-fill cache (map, history lists): sym -> {kind, tfs}
async function extFetch(syms,date){const need=[...new Set(syms)].filter(s=>s&&!(s in EXTD));
  if(!need.length)return;
  try{const j=await(await fetch('/api/ext?symbols='+encodeURIComponent(need.join(','))
    +(date?'&date='+encodeURIComponent(date):''))).json();Object.assign(EXTD,j);}catch(e){}}
function extSlot(sym,curTf){const d=EXTD[sym];
  return '<span class="extslot" data-s="'+sym+'" data-tf="'+(curTf||'1D')+'">'
    +(d?extBadges(d.tfs,d.kind,curTf,d.rsix):'')+'</span>';}
function extPatch(root){(root||document).querySelectorAll('.extslot').forEach(el=>{
  const d=EXTD[el.dataset.s];if(d&&!el.innerHTML)el.innerHTML=extBadges(d.tfs,d.kind,el.dataset.tf,d.rsix);});}
function rows(){const g=id=>document.getElementById(id);
  const fv=+g('fvol').value||0, fav=+g('favol').value||0, fp=+g('fprice').value||0,
    fa=+g('fadr').value||0, flo=+g('flo52').value||0, fhi=+g('fhi').value||0, frs=+g('frs').value||0,
    s50=g('fsma50').checked, s200=g('fsma200').checked, fst=g('fstate').value, fetf=g('fetf').value;
  const fema_scope=g('fema_scope')?g('fema_scope').value:'', fema_base=g('fema_base')?g('fema_base').value:'', fema_time=g('fema_time')?g('fema_time').value:'', fema_dir=g('fema_dir')?g('fema_dir').value:'';
  const fema_sec_state=g('fema_sec_state')?g('fema_sec_state').value:'', fema_thm_state=g('fema_thm_state')?g('fema_thm_state').value:'';
  const fpat=g('fpat').value, usePat=(TABLES[cur]||[]).some(x=>x.pattern!=null&&x.pattern!=='');
  const pk=(cur==='stairstep');   // stairstep: trend/ADR/off-high filters apply AS OF THE PEAK
  let r=(TABLES[cur]||[]).filter(x=>{
    const a50=pk?x.above_sma50_at_peak:x.above_sma50, a200=pk?x.above_sma200_at_peak:x.above_sma200,
      oh=pk?x.off_ath_at_peak:x.off_ath_pct, adr=pk?x.adr_at_peak:x.adr_pct;
    return (x.dollar_vol_m||0)>=fv&&(x.avg_vol_k||0)>=fav&&(x.close||0)>=fp&&(adr||0)>=fa
      &&(x.above_lo52_pct||0)>=flo
      &&(!fhi||(oh!=null&&oh>=-fhi))
      &&(!frs||(x.rs_rank!=null&&x.rs_rank>=frs))     // RS leadership (top ~2% = 98)
      &&(!s50||a50)&&(!s200||a200)
      &&(!usePat||!fpat||x.pattern===fpat)
      &&(!fst||x.state===fst)
      &&(!fetf||(fetf==='etf'?!!x.is_etf:!x.is_etf))
      &&(cur!=='ema_cross'||(
        (!fema_scope||x.entity_type===fema_scope)
        &&(!fema_base||x.variant===fema_base)
        &&(!fema_dir||x.state===fema_dir)
        &&(!fema_sec_state||x.sector_state===fema_sec_state)
        &&(!fema_thm_state||x.theme_state===fema_thm_state)
        &&(!fema_time||(fema_time==='0'?x.days_since_flip===0:(x.days_since_flip>0&&x.days_since_flip<=5)))
      ));});
  if(sortCol)r=r.slice().sort((a,b)=>{const x=a[sortCol],y=b[sortCol];
    return(((x>y)-(x<y))*(sortAsc?1:-1));});
  return r;}
function render(){tabs();
  if(panelMode&&panelFor!==cur)renderSetupPanel();   // open info/history panel follows the setup
  const tfs=document.getElementById('tfsel');if(tfs)tfs.style.display=RIDER_TABS.includes(cur)?'':'none';  // TF selector only on rider tabs
  const hasPat=(TABLES[cur]||[]).some(x=>x.pattern!=null&&x.pattern!=='');  // pattern filter only on tabs that have it
  document.getElementById('fpatwrap').style.display=hasPat?'':'none';
  if(document.getElementById('femawrap'))document.getElementById('femawrap').style.display=(cur==='ema_cross')?'inline-block':'none';
  const ST=['too_short','optimal','building','breakout','riding','touching','armed','saved','break','reversal'];  // state options adapt to the tab
  const have=new Set((TABLES[cur]||[]).map(x=>x.state).filter(Boolean));
  const fs=document.getElementById('fstate'),prevSt=fs.value;
  fs.innerHTML='<option value="">any state</option>'+ST.filter(s=>have.has(s)).map(s=>'<option value="'+s+'">'+stDisp(s)+'</option>').join('');
  fs.value=have.has(prevSt)?prevSt:'';
  const cols=[...BASE,...EXTRA[cur]];const r=rows();
  document.getElementById('rowcount').textContent=r.length+' shown';
  const w=document.getElementById('tablewrap');
  if(!r.length){w.innerHTML='<div class="empty">No hits match the filters.</div>';return;}
  const HLBL={adr_pct:'ADR%/Day',risk:'$risk',reward:'$reward',rr:'r/R',rr_target:'r/R→tgt',
    sector:'industry',max_pos:'max $pos',tf:'TF',is_etf:'ETF?',rs_rank:'RS',
    off_hi52_pct:'off 52wk%',dollar_vol_m:'$vol M/day',avg_vol_k:'avg vol K',market_cap:'mkt cap'};
  const HTIP={risk:'close to nearest SUPPORT (long) / RESISTANCE (short)',
    reward:'close to nearest RESISTANCE (long) / SUPPORT (short)',rr:'$reward ÷ $risk',
    rr_target:"reward to the setup's own measured-move TARGET ÷ $risk (only setups that project one)",
    max_pos:'max position size = 0.5% of average daily $ volume',
    rs_rank:'relative-strength percentile (0-100, higher = leader)',is_etf:'instrument type',
    off_hi52_pct:'% below the 52-week high'};
  let h='<table><thead><tr>'+cols.map(c=>'<th data-c="'+c+'"'+(HTIP[c]?' title="'+HTIP[c]+'"':'')+'>'
    +(HLBL[c]||c)+(sortCol===c?(sortAsc?' ▲':' ▼'):'')+'</th>').join('')+'</tr></thead><tbody>';
  for(const x of r){const key=x.symbol+'|'+x.setup+'|'+x.tf;
    h+='<tr data-k="'+key+'" class="'+(key===selKey?'sel ':'')+(x.state==='optimal'?'opt':'')+'">'+cols.map(c=>{
      let cls='',st='';if(['gap_pct','chg_pct','prior_gain_pct','pole_gain_pct','reclaim_pct'].includes(c))cls=x[c]>=0?'up':'dn';
      if((c==='rr'||c==='rr_target')&&x[c]!=null){cls=x[c]>=2?'up':(x[c]<1?'dn':'');
        st=' style="background:'+(x[c]<1?'rgba(224,86,86,.12)'
          :'rgba(47,191,143,'+Math.min(.30,.06+x[c]*.06).toFixed(2)+')')+'"';}  // strength readable at a glance
      if(c==='sector'||c==='themes')return '<td class="tight" title="'+String(x[c]||'').replace(/"/g,'&quot;')+'">'+fmt(x[c])+'</td>';
      if(c==='off_hi52_pct')cls=x[c]>=-5?'up':'';
      if(c==='regime_dir')cls=x[c]==='up'?'up':(x[c]==='down'?'dn':'');
      if(c==='state')cls=(x[c]==='breakout'||x[c]==='riding'||x[c]==='optimal')?'up':(x[c]==='reversal'?'dn':'');
      if(c==='state'&&x[c]==='touching')st=' style="color:#e8b84b;font-weight:600"';
      if(c==='state'&&x[c]==='too_short')st=' style="color:#8a94a6"';
      if(c==='quality'&&typeof x[c]==='number')st=' style="color:'+qcolor(x[c])+';font-weight:600"';
      if(c==='state'&&x.provisional)return '<td class="'+cls+'"'+st+'>'+stDisp(x[c])+   // provisional breakout marker
        '<span title="provisional — breakout confirmed on TODAY’s PROJECTED partial-session volume ('+
        Math.round((x.vol_proj||0)*100)+'% of the session in); settles to real volume at the close" '+
        'style="color:#e8b84b;font-weight:700"> ~</span></td>';
      if(c==='state')return '<td class="'+cls+'"'+st+(STTIP[x[c]]?' title="'+STTIP[x[c]]+'"':'')+'>'+stDisp(x[c])+'</td>';
      if(c==='quality'&&typeof x[c]==='number')return '<td'+st+'><span style="font-size:9px">●</span> '+fmt(x[c])+'</td>';
      if(c==='market_cap')return '<td>'+mcap(x[c])+'</td>';
      if(c==='max_pos')return '<td>'+(x[c]==null?'':(x[c]>=1e6?(x[c]/1e6).toFixed(1)+'M':Math.round(x[c]/1e3)+'K'))+'</td>';
      if(c==='symbol')return '<td class="'+cls+'"'+st+' title="'
        +String((x.sector||'')+(x.themes?' — '+x.themes:'')).replace(/"/g,'&quot;')+'">'
        +jBtn(x.symbol,x.date||curAsof(),x.tf)+fmt(x[c])+extBadges(x.ext_tfs,x.kind,x.tf||'1D',x.rsix)
        +(cur==='episodic_pivot'?epNewsBtn(x.symbol):'')+'</td>';
      return '<td class="'+cls+'"'+st+'>'+fmt(x[c])+'</td>';}).join('')+'</tr>';}
  w.innerHTML=h+'</tbody></table>';
  w.querySelectorAll('th').forEach(th=>th.onclick=()=>{const c=th.dataset.c;
    sortAsc=(sortCol===c)?!sortAsc:false;sortCol=c;render();});
  w.querySelectorAll('tbody tr').forEach(tr=>tr.onclick=()=>{selKey=tr.dataset.k;
    w.querySelectorAll('tr.sel').forEach(e=>e.classList.remove('sel'));
    tr.classList.add('sel');showChart(selKey);});}
function tvUrl(row){const s=(row.exchange?row.exchange+':':'')+row.symbol.replace(/-/g,'.');
  return 'https://www.tradingview.com/chart/?symbol='+encodeURIComponent(s);}
// ── EP news analysis: catalyst + sentiment + reaction tell, AS-OF the EP trigger day ─────────────
function epEsc(s){return String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
const EP_CAT={earnings:'earnings',guidance:'guidance / outlook',drug:'drug / clinical data',fda:'FDA / regulatory',
  ma:'M&A / deal',analyst:'analyst rating',offering:'stock offering / dilution',legal:'legal / investigation',
  product:'product / contract',partnership:'partnership',sympathy:'sector sympathy (no own headline)',unknown:'no clear catalyst'};
function epNewsBtn(sym){return '<span class="chip" title="News analysis for '+sym+' on its EP trigger day" '
  +'style="cursor:pointer;margin-left:6px;padding:0 5px" onclick="event.stopPropagation();epNews(\''+sym+'\')">&#128240;</span>';}
async function epNews(sym){const m=document.getElementById('epnewsmodal');
  m.querySelector('.epnh').textContent='📰 '+sym+' — EP news';
  m.querySelector('.epnb').innerHTML='<span class="hd">analyzing the trigger-day news…</span>';m.style.display='flex';
  try{const j=await(await fetch('/api/ep_news?sym='+encodeURIComponent(sym)+'&date='+encodeURIComponent(curAsof()))).json();
    if(j.error){m.querySelector('.epnb').innerHTML='<span class="dn">'+epEsc(j.error)+'</span>';return;}
    m.querySelector('.epnh').textContent='📰 '+j.symbol+(j.name?' · '+j.name:'')+' — EP '+(j.event_date||'');
    m.querySelector('.epnb').innerHTML=epNewsHtml(j);}
  catch(e){m.querySelector('.epnb').innerHTML='<span class="dn">news fetch failed: '+epEsc(e)+'</span>';}}
function epNewsHtml(j){
  const mv=j.move_pct==null?'':(j.move_pct>0?'+':'')+j.move_pct+'%';
  const sW={good:'positive',bad:'negative',mixed:'mixed',neutral:'neutral'}[j.sent]||'—';
  const sC=j.sent==='good'?'up':(j.sent==='bad'?'dn':'');
  const cat='<div class="trow"><span class="tname">catalyst</span><span><b>'+epEsc(EP_CAT[j.catalyst]||j.catalyst||'—')+'</b>'
    +(j.ep_subtype?' <span class="hd">('+epEsc(j.ep_subtype)+(j.ep_age?', '+j.ep_age+' sessions ago':', event day')+')</span>':'')+'</span></div>';
  const day='<div class="trow"><span class="tname">EP day</span><span>'+(j.event_date||'')
    +(mv?' · move <b class="'+(j.move_pct>0?'up':'dn')+'">'+mv+'</b>':'')
    +' · news tone <b class="'+sC+'">'+sW+'</b>'+(j.conflicting?' <span class="hd">(conflicting)</span>':'')+'</span></div>';
  const tell=j.news_warn?'<div class="trow"><span class="tname">read</span><span style="color:#e8b84b;font-weight:600">'+epEsc(j.news_warn)+'</span></div>':'';
  const own=j.own_headline?'<div class="trow"><span class="tname">own story</span><span>'+epEsc(j.own_headline)+'</span></div>':'';
  const peer=j.peer_headline?'<div class="trow"><span class="tname">peer/sector</span><span style="color:var(--dim)">'+epEsc(j.peer_headline)+'</span></div>':'';
  if(!j.n_stories)return cat+day+'<div class="trow"><span class="hd">no headlines found for '+(j.event_date||'this day')
    +' — Tiingo’s news archive only reaches back ~3 months, so older EP days may be blank.</span></div>';
  const stories='<div class="trow tsec"><span class="tname">stories ('+j.n_stories+')</span><span>'
    +(j.stories||[]).map(s=>'<div class="pbidea" title="'+epEsc((s.source||'')+(s.published?' · '+s.published:''))+'">'
      +'&bull; '+epEsc(s.title||'')+'</div>').join('')+'</span></div>';
  return cat+day+tell+own+peer+stories;}
// ── Market situational-awareness banner (top-down long/out/short) ───────────────
let marketData=null, marketOpen=false;
let replayT=null;                          // intraday replay: null = LIVE (now/close); else "HH:MM" ET cutoff
let tsDrag=false;                          // mid-scrub guard: resolving fetches must NOT rebuild the bar (and kill the slider) under the pointer
let dtScrubT=null;                         // debounce for live day-type follow while dragging
function tq(){return replayT?'&t='+encodeURIComponent(replayT):'';}
function tsTime(k){const m=570+k*5;return String(Math.floor(m/60)).padStart(2,'0')+':'+String(m%60).padStart(2,'0');}  // 570 = 09:30, 5-min stops
let dataHealth=null;                                 // live 5m warming/rebuilding state (Amir 2026-07-07)
function loadDataHealth(){fetch('/api/data_health').then(r=>r.json()).then(j=>{dataHealth=j;drawMarket();}).catch(()=>{});}
let crossNewsData=null;                              // cross-instrument news reaction (proxy -> underlying), Amir 2026-07-07
function loadCrossNews(){fetch('/api/cross_news').then(r=>r.json()).then(j=>{crossNewsData=j;drawMarket();}).catch(()=>{});}
function reloadAwareness(keep){if(!ONDEMAND)return;   // keep=true (live 60s poll): swap data IN PLACE
  // on arrival, don't blank first — nulling mid-refetch made the banner head flip to the close
  // fallback for ~1-2s every minute (Amir 2026-07-08 flicker). Scrub/date-change (keep falsy) still
  if(!keep){groupsData=null;playbookData=null;mapData=null;daytypeData=null;outlookData=null;rs4hData=null;boardData=null;futuresData=null;universeData=null;}
  daytypeReq=false;loadDaytype();loadMap();loadOutlook();loadRs4h();loadDataHealth();loadCrossNews();
  fetch('/api/market?date='+encodeURIComponent(curAsof())+tq())    // /api/market can be slow during a live re-seed; the warming chip rides the lightweight /api/data_health poll above so it shows regardless
    .then(r=>r.json()).then(j=>{marketData=j;drawMarket();if(marketOpen)loadGroups();}).catch(()=>{});
  const pd=document.getElementById('perfdrawer');if(pd&&pd.classList.contains('open')&&typeof loadBoard==='function'){  // perf since-open follows the replay time
    if(perfDrill)loadGroupDetail(perfDrill.group,perfDrill.gtype,true);              // keep the drilled-in group (silent)
    else if(perfView==='futures'){futuresData=null;drawFutures();}
    else if(perfView==='universe'){universeData=null;drawUniverse();}
    else loadBoard(true);}}                                                          // board views: refresh median + live-open
function loadMarket(date){if(!ONDEMAND)return;replayT=null;window.EXTD={};reloadAwareness();}   // date changed -> snap back to LIVE (+ ext badges refetch as-of)
// live auto-refresh: match the 60s tiingo poll cadence; only on the LIVE view of a visible tab
setInterval(()=>{if(ONDEMAND&&replayT===null&&document.visibilityState==='visible')reloadAwareness(true);},60000);  // live poll: keep data, swap in place (no flicker)
// data-warming chip rides its OWN lightweight poll, independent of the (possibly slow during a re-seed) /api/scan
// + /api/market load chain — so the "⚠ data warming" caveat is visible exactly when those endpoints are slow
setTimeout(loadDataHealth,150);
setInterval(()=>{if(document.visibilityState==='visible')loadDataHealth();},30000);
function weatherHtml(w){if(!w||!w.families)return '';   // measured setup-weather (T7: 43,827 fires, era-robust)
  const on=Object.keys(w.states||{}).filter(k=>w.states[k]);
  let h='<div class="mrow"><span class="mname">setup weather</span><span class="mwhy">';
  const FL={MOMO:'momentum',EVENT:'event',MEANREV:'mean-revert'};
  h+=['MOMO','EVENT','MEANREV'].map(f=>{const x=w.families[f];if(!x)return '';
    const col=x.tone==='up'?'var(--up)':(x.tone==='dn'?'var(--dn)':'var(--dim)');
    const word=x.tone==='up'?'favored':(x.tone==='dn'?'headwind':'neutral');
    const t=(FL[f]+' setups are '+word+' right now — expectancy is tilted '+(x.score>0?'+':'')+x.score
      +'R vs baseline. Setups: '+x.setups.join(', ')
      +(x.why.length?'.  What is moving it: '+x.why.join(' · '):'.  No active state moves this family')
      +'.  Measured over 43,827 historical fires (era-robust) — this is an edge tilt, not a win probability.').replace(/"/g,'&quot;');
    return '<span class="grp" title="'+t+'"><b style="color:'+col+'">'+(x.tone==='up'?'▲':(x.tone==='dn'?'▼':'▬'))
      +'</b> '+FL[f]+' <span style="color:'+col+'">'+word+'</span> <span class="hd">'+(x.score>0?'+':'')+x.score+'R</span></span>';}).join(' &nbsp;|&nbsp; ');
  h+=' <span class="hd" title="the market states currently active that tilt the numbers above">('+(on.length?('active: '+on.join(', ')):'no special states')+')</span></span></div>';
  if(w.caution)h+='<div class="mrow"><span class="mname" style="color:#e0a43a">⚠ caution</span><span class="mwhy">'
    +'<b>capitulation window</b> — within ~2 months of mass RSI capitulation: ALL new setup entries '
    +'historically degrade ~0.4R (momentum −0.44 · mean-revert −0.43 · events −0.11; era-robust). '
    +'Index bounces are strong but violent — smaller size, wider expectations.</span></div>';
  return h;}
function mBadge(s,tag,dim){const sc=Math.round(s.score),sign=sc>0?'+':'';
  const d=((tag==='CLOSE'||tag==='prior close')&&s.as_of)?' '+s.as_of.slice(5):'';
  const dimst=dim?';opacity:.62;font-size:11px;font-weight:600;padding:2px 7px':'';   // demoted 'prior close' chip while live
  return '<span class="mbadge" style="background:'+s.color+dimst+'" title="validated market-health score, settled end-of-day read'+(s.as_of?' for '+s.as_of:'')
    +' · scale −100 (everything bearish) to +100 (everything bullish), mean of 5 validated dials · bands: ≥+29 strong long, ≥−5 long, <−24 short bias">'+(tag?tag+d+' ':'')+s.stance.toUpperCase()+' '+sign+sc+'</span>';}
function mGauge(v){                                   // -1..+1 -> small colored bar (quick read)
  if(v==null)return '<span class="muted" style="width:64px;display:inline-block">n/a</span>';
  const w=Math.round(Math.abs(v)*30), col=v>0?'var(--up)':(v<0?'var(--dn)':'var(--dim)');
  return '<span style="display:inline-block;width:64px;vertical-align:middle"><span style="display:inline-block;height:7px;border-radius:3px;background:'
    +col+';width:'+Math.max(3,w)+'px;margin-left:'+(v>=0?30:30-w)+'px"></span></span>';}
const HEALTH_TIPS={
  'trend census':'share of SPY/QQQ/IWM/DIA daily charts in confirmed UPTRENDS minus DOWNTRENDS. +1 = all four trending up, -1 = all four trending down.',
  'persistence':'breadth QUALITY (% of stocks above their 20-day MA) mapped to how often the current market structure historically kept going the next month, signed by SPY structure. Positive = breadth supports the trend continuing; negative = breadth undermines it.',
  'dip risk (inv)':'the outlook dials (vol complex, credit, distribution days, NH-NL, ratio trends…) averaged into historical odds of a −5% dip within a month, INVERTED. Positive = calmer than usual; negative = dip odds elevated.',
  'overnight lean':'the overnight-environment composite (bitcoin, dollar, Asia/Europe sessions, market leaders) as a percentile of its own history, centered. Positive = risk-on nights lately.',
  'credit':'junk-bond stress: the 5-day change in high-yield spreads mapped to historical dip odds, INVERTED. Positive = credit calm (supportive); negative = credit widening (a classic early risk signal).',
  'risk appetite':'where money is rotating: the risk-on ratio charts (small-caps vs broad, semis, discretionary vs staples, banks, copper vs gold — each read by trendlab structure) averaged with the NASDAQ-vs-NYSE advancer split. Positive = offense leading (risk-on); negative = defense leading (risk-off). Validated add-on: doubled the score 5-day ranking power in 2025-26.'};
function healthHtml(){const h=marketData&&marketData.health;if(!h||h.error)return '';
  const comps=Object.entries(h.components||{}).map(([k,v])=>
    '<span style="margin-right:14px;white-space:nowrap" title="'+(HEALTH_TIPS[k]||k)+' Current: '+(v>0?'+':'')+v+' (scale −1…+1).">'+k+' '+mGauge(v)+'</span>').join('');
  return '<div class="mrow" title="'+(h.note||'')+'"><span class="mname">health</span><span class="mwhy">'
    +'<b style="color:'+h.color+'" title="mean of the 5 dials on a −100 (everything bearish) … +100 (everything bullish) scale · bands: ≥+29 strong long, ≥−5 long, <−24 short bias">'
    +h.stance.toUpperCase()+' '+(h.score>0?'+':'')+Math.round(h.score)+'</b> &nbsp;'+comps+'</span></div>'
    +'<div class="mrow"><span class="mname">context</span><span class="mwhy" style="color:#aab8c8">'+h.context+'</span></div>';}
function mBlocks(s,tag){if(!s)return '';
  const lbl=(tag==='close'&&s.as_of)?'settled · '+s.as_of:tag;
  return '<div class="mrow"><span class="mname">'+lbl+'</span><span class="mwhy" style="color:var(--dim)">'
    +'legacy composite (hand-weighted, for reference): '+s.stance+' '+(s.score>0?'+':'')+s.score+'</span></div>'
    +(s.blocks||[]).map(b=>{const v=b.score==null?'n/a':((b.score>0?'+':'')+b.score.toFixed(2));
      const cls=b.score==null?'muted':(b.score>0?'up':(b.score<0?'dn':'muted'));
      const why=b.why||'', cut=why.length>90?why.slice(0,88).replace(/[,;][^,;]*$/,'')+' …':why;
      return '<div class="mrow" title="'+why.replace(/"/g,"'")+' ('+v+')"><span class="mname" style="padding-left:10px">'+b.name+'</span>'
        +mGauge(b.score)
        +'<span class="mwhy" style="color:var(--dim)">'+cut+'</span></div>';}).join('');}
let groupsData=null;
function loadGroups(){fetch('/api/groups?date='+encodeURIComponent(curAsof())+tq()).then(r=>r.json()).then(j=>{groupsData=j;drawMarket();}).catch(()=>{});}
function grp(x){const cls=x.score>0?'up':(x.score<0?'dn':'muted');
  const ar=x.rs_trend==='accel'?'<span class="up"> ▲</span>':(x.rs_trend==='fade'?'<span class="dn"> ▼</span>':'');  // RS accelerating / fading
  const clk=x.kind==='theme'?'mapPick(\'theme\',\''+(x.name||'').replace(/'/g,'')+'\')':(x.etf?'mapPick(\'sector\',\''+x.etf+'\')':'');
  return '<span class="grp"'+(clk?' style="cursor:pointer" onclick="'+clk+'"':'')+' title="'+((x.why||'')+(clk?' · click to open the chart + members':'')).replace(/"/g,"'")+'"><b class="'+cls+'">'+(x.score>0?'+':'')+x.score+'</b> '+x.name+(x.etf?' '+x.etf:'')+ar+'</span>';}
function groupsHtml(){if(!groupsData)return '<div class="mrow"><span class="mname">leaders</span><span class="mwhy" style="color:var(--dim)">loading…</span></div>';
  const g=groupsData.groups||[],sec=g.filter(x=>x.kind==='sector'),thm=g.filter(x=>x.kind==='theme');
  const line=(a)=>a.slice(0,6).map(grp).join(' · ')+(a.length>6?'  …  '+a.slice(-3).map(grp).join(' · '):'');
  return '<div class="mrow"><span class="mname">sectors</span><span class="mwhy">'+line(sec)+'</span></div>'
    +'<div class="mrow"><span class="mname">themes</span><span class="mwhy">'+line(thm)+'</span></div>';}
// ── ⚖ short-term index RS (Amir's 4h ratio-chart workflow) ──────────────────────
let rs4hData=null;
function loadRs4h(){fetch('/api/rs4h?date='+encodeURIComponent(curAsof())).then(r=>r.json())
  .then(j=>{rs4hData=j;drawMarket();}).catch(()=>{});}
function stGlyph(w){                                   // trendlab class -> glyph + color
  if(!w)return ['·','var(--dim)','no data'];
  if(w==='up')return ['▲ up','var(--up)','uptrend'];
  if(w==='down')return ['▼ down','var(--dn)','downtrend'];
  if(w==='range/contracting')return ['◮ contr','var(--acc)','contracting range (narrowing swings)'];
  if(w==='range/rectangle')return ['▭ rect','var(--acc)','rectangle (boxed between double-top highs and double-bottom lows)'];
  if(w==='range/expanding')return ['◇ expand','var(--acc)','expanding range (widening swings)'];
  if(w==='rolling-over')return ['◿ rolling over','var(--dn)','lower high + lower low — bearish bias, downtrend not yet confirmed'];
  if(w==='turning-up')return ['◹ turning up','var(--up)','higher high + higher low — bullish bias, uptrend not yet confirmed'];
  if(w==='broken-up')return ['⌁ broken up','var(--acc)','WAS an uptrend — the trend broke and price is now rangebound below it'];
  if(w==='broken-down')return ['⌁ broken dn','var(--acc)','WAS a downtrend — the trend broke and price is now rangebound above it'];
  return ['· pausing','var(--dim)','between structures — no reference read yet'];}
function hintHtml(){const j=rs4hData;if(!j||!j.hints||!j.hints.length)return '';
  return j.hints.map(h=>'<div class="mrow" title="from the MTF pivot-cascade study (matched-base odds, frozen 2026-07-05); descriptive, not a trade signal">'
    +'<span class="mname">early pivot</span><span class="mwhy">'
    +'<b style="color:'+(h.dir==='down'?'var(--up)':'var(--dn)')+'">⚡ '+h.pair+'</b> '
    +h.text+'</span></div>').join('');}
function stSign(w){                                    // structure word -> directional lean
  if(w==='up'||w==='turning-up')return 1;
  if(w==='down'||w==='rolling-over')return -1;
  return 0;}
function stRead(r,TFS){                                // plain-English top-down read per row
  const s=TFS.map(t=>r[t]?stSign(r[t].w):0);           // [1W, 2D, 1D, 4h]
  const htf=s[0]||s[1];                                // context = 1W (else 2D)
  const negLow=s[2]<0||s[3]<0, posLow=s[2]>0||s[3]>0;
  if(htf>0&&posLow&&!negLow)return ['aligned up','var(--up)','all timeframes agree: uptrend context and up timing — trend-following longs have the wind at their back'];
  if(htf<0&&negLow&&!posLow)return ['aligned down','var(--dn)','all timeframes agree: downtrend context and down timing — this is where shorts work and dip-buys fail'];
  if(htf>0&&negLow)return s[3]>0
    ?['pullback · 4h turning','var(--acc)','bigger picture up, daily pulled back, and the 4h is already turning back up — the classic dip-buy TRIGGER zone; confirm before chasing']
    :['pullback','var(--acc)','bigger picture still up, lower timeframes pulling back — dip-buy WATCH zone: wait for the 4h to turn back up'];
  if(htf<0&&posLow)return s[3]<0
    ?['bounce fading','var(--dn)','bigger picture down, the bounce already rolling over on the 4h — rally into resistance losing steam']
    :['bounce','var(--acc)','bigger picture still down, lower timeframes bouncing — a rally toward resistance; not a base until the higher timeframe repairs'];
  return ['mixed','var(--dim)','no clean top-down agreement — structure in transition, let it resolve'];}
function structGridHtml(){const j=rs4hData;if(!j||!j.grid)return '';
  const TFS=['1W','2D','1D','4h'];                     // top-down: higher TF first (context -> timing)
  let h='<table style="border-collapse:collapse;font:11.5px var(--mono)">'
    +'<tr title="top-down analysis: read left to right — the higher timeframe sets the context, the lower timeframes time the entry"><td></td>'
    +TFS.map((t,i)=>'<td style="color:var(--dim);padding:0 9px;font-weight:'+(i===0?'700':'400')+'">'+t+'</td>').join('')
    +'<td style="color:var(--dim);padding:0 9px">read</td></tr>';
  j.grid.forEach(r=>{h+='<tr><td style="color:#aab8c8;font-weight:600;padding:1px 6px 1px 0;cursor:pointer" onclick="mapPick(\'sym\',\''+r.sym+'\')" title="open '+r.sym+' chart + ticker card">'+r.sym+'</td>'
    +TFS.map((t,i)=>{const c=r[t];if(!c)return '<td style="color:var(--dim);padding:0 9px">·</td>';
      const [g,col,full]=stGlyph(c.w);
      const tip=r.sym+' '+t+': '+full+' ['+(c.via||'')+'] — '+c.bars+' bars, since '+c.since
        +(c.prev?' (before that: '+c.prev+')':'')
        +'. Trendlab swing read — wide-lens patterns (channels, multi-week triangles) not detected yet.';
      return '<td style="color:'+col+';padding:0 9px;cursor:default;font-weight:'+(i===0?'600':'400')+'" title="'+tip.replace(/"/g,"'")+'">'+g
        +(c.bars<=3?' <span style="color:var(--dim);font-size:9px">new</span>':'')+'</td>';}).join('');
    const [rw,rc,rtip]=stRead(r,TFS);
    h+='<td style="color:'+rc+';padding:0 9px;cursor:default;font-style:italic" title="'+rtip.replace(/"/g,"'")+'">'+rw+'</td></tr>';});
  return '<div class="mrow"><span class="mname">structure</span><span class="mwhy">'+h
    +'</table><span style="color:var(--dim);font-size:10.5px">top-down: 1W context → 4h timing · hover cells for what each means · "read" = the top-down verdict</span></span></div>';}
function rs4hHtml(){if(!rs4hData||rs4hData.error)return '';
  const j=rs4hData;
  const wc=w=>w==='uptrend'?'var(--up)':(w==='downtrend'?'var(--dn)':'var(--acc)');
  const medals=['\u{1F947}','⚖','\u{1F949}'];
  const chips=(j.rows||[]).map((r,i)=>'<span title="'+r.pair+' 4h: '+r.word
      +' · last 10 bars '+(r.ret10>0?'+':'')+r.ret10+'%'
      +(r.res_atr!=null?' · resistance '+r.res_atr+' ATR above':'')
      +(r.sup_atr!=null?' · support '+r.sup_atr+' ATR below':'')
      +' · click the ticker for its chart, ⇄ for the '+r.sym+'/SPY ratio chart'
      +'" style="margin-right:10px">'+medals[Math.min(i,2)]
      +' <b style="cursor:pointer" onclick="mapPick(\'sym\',\''+r.sym+'\')">'+r.sym+'</b>'
      +' <span style="color:'+wc(r.word)+'">'+r.word+'</span>'
      +' <span style="cursor:pointer;color:var(--dim)" onclick="chartRatio(\''+r.sym+'\',\'SPY\',\'4h\')" title="open the '+r.sym+'/SPY ratio chart (4h)">⇄</span></span>').join('');
  return '<div class="mrow" title="'+(j.footnote||'')+'"><span class="mname">RS · 4h</span>'
    +'<span class="mwhy">'+chips
    +'<br><span style="color:#aab8c8">'+j.suggestion+'</span>'
    +' <span style="color:var(--dim)">· 4h ratio charts vs SPY, as of '+(j.asof_bar||'')+'</span>'
    +'</span></div>';}
// ── 🔭 next-month risk outlook (awareness_lab: fit-free era-robust dial odds) ────
let outlookData=null;
function loadOutlook(){fetch('/api/outlook?date='+encodeURIComponent(curAsof())).then(r=>r.json())
  .then(j=>{outlookData=j;drawMarket();}).catch(()=>{});}
function outlookHtml(){if(!outlookData||outlookData.error)return '';
  const o=outlookData;
  const col=o.color==='dn'?'var(--dn)':(o.color==='up'?'var(--up)':'var(--dim)');
  const dials=o.dials||[];
  const nR=dials.filter(d=>d.tone==='risky').length, nC=dials.filter(d=>d.tone==='calm').length;
  const chips=dials.map(d=>'<span title="'+d.label+' — currently '
      +(d.tone==='risky'?'RISK-ELEVATED':d.tone==='calm'?'CALM':'neutral')
      +'. Sits in quintile '+d.q+' of its own history, where a −5% pullback within the next month has followed '
      +d.p5+'% of the time. Green dot = calmer than usual, red = a dip is historically more likely here."'
      +' style="color:'+(d.tone==='risky'?'var(--dn)':d.tone==='calm'?'var(--up)':'var(--dim)')+'">●</span>').join('');
  const summary=' <span class="hd" title="how many of the '+dials.length+' risk dials are flashing red vs green right now">'
      +'<b style="color:var(--dn)">'+nR+'</b> risk · <b style="color:var(--up)">'+nC+'</b> calm</span>';
  return '<div class="mrow" title="'+(o.footnote||'')+' · each dot is one risk dial — hover it. Dials: '+dials.map(d=>d.label).join(' / ')+'">'
    +'<span class="mname">outlook</span><span class="mwhy">'
    +'<b style="color:'+col+';border:1px solid '+col+';border-radius:8px;padding:0 7px">'+o.level+'</b> '
    +o.headline.replace(/^next-month dip risk (ELEVATED|NORMAL|LOW)( — |: )/,'')
    +' <span style="letter-spacing:2px">'+chips+'</span>'+summary
    +(o.persist?'<br><span style="color:var(--dim)">'+o.persist.note+'</span>':'')
    +'</span></div>';}
let playbookData=null;
function loadPlaybook(){fetch('/api/playbook?date='+encodeURIComponent(curAsof())+tq()).then(r=>r.json()).then(j=>{playbookData=j;drawMarket();}).catch(()=>{});}
// ── H3 day-type strip: pre-open expectations from the frozen production model ────
let daytypeData=null,daytypeReq=false;
const DT_SYMS=['SPY','QQQ','IWM','DIA'];
function loadDaytype(){if(daytypeReq)return;daytypeReq=true;
  const myT=replayT,myD=curAsof();      // drop out-of-order responses while scrubbing
  Promise.all(DT_SYMS.map(s=>fetch('/api/daytype?sym='+s+'&date='+encodeURIComponent(myD)+tq())
      .then(r=>r.json()).catch(()=>null)))
    .then(a=>{
      if(myT!==replayT||myD!==curAsof())return;   // slider moved on — a newer fetch owns the strip
      daytypeData={};DT_SYMS.forEach((s,i)=>daytypeData[s]=a[i]);
      const dw=document.getElementById('dtwrap');
      if(tsDrag&&dw)dw.innerHTML=daytypeHtml();    // mid-drag: update the rows in place only
      else drawMarket();})
    .catch(()=>{daytypeReq=false;});}
function dtWhen(j){                                   // compact "Jul 2 · 14:30" stamp
  const d=j.date?new Date(j.date+'T12:00'):null;
  const ds=d?d.toLocaleDateString('en-US',{month:'short',day:'numeric'}):'';
  const tm=(j.cp_label||'').replace(' bar','');
  if(j.mode==='live-intraday')return j.session_over?('<b style="color:var(--acc)">■ TODAY at CLOSE</b> '+ds):('<b style="color:var(--up)">● LIVE</b> '+tm);
  if(j.mode==='history')return ds+(tm?' · '+tm:'');
  return ds+' · pre-open';}
function dtStance(j){                                 // stance with SHORTS/LONGS sized by suggested size
  let s=(j.stance||'').replace(/\s*\(skill [^)]*\)/,'');
  const big=/with size/.test(s), small=/small|starter|probe|light/.test(s);
  const px=big?17:(small?11:13.5);
  s=s.replace(/\b(shorts?|longs?)\b/i,m=>'<b style="font-size:'+px+'px;letter-spacing:.03em">'+m.toUpperCase()+'</b>');
  return '<span style="color:'+(j.color||'var(--txt)')+';font-weight:600">'+s+'</span>';}
function dtRow(j,name,main){if(!j||j.error||!j.probs)return '';
  const p=j.probs,f=t=>p[t]?Math.round(100*p[t].p)+'%':'—',
        b=t=>p[t]?Math.round(100*p[t].base)+'%':'—';
  const hist=j.mode==='history',live5=j.mode==='live-intraday';
  // the raw model numbers live on HOVER now (Amir 2026-07-05: the stance is the read)
  // plain-English confidence line first (Amir Q2): odds vs typical + model skill
  const sk=((j.stance||'').match(/skill ([\d.]+)%/)||[])[1];
  let conf='';
  ['down','up','trend'].forEach(t=>{if(conf||!p[t])return;
    const r=p[t].base>0?p[t].p/p[t].base:0;
    if(r>=1.5&&p[t].p>=0.2)conf='confidence: '+t.toUpperCase()+' day '+Math.round(100*p[t].p)
      +'% likely vs '+Math.round(100*p[t].base)+'% on a typical day ('+r.toFixed(1)+'x the usual odds)'
      +(sk?' · this model beats naive guessing by '+sk+'% at this hour':'')+' — ';});
  const tip=(conf+'model: trend '+f('trend')+' (base '+b('trend')+') · up '+f('up')+' ('+b('up')
    +') · down '+f('down')+' ('+b('down')+')'
    +(hist?(j.granularity==='5m'?' · per-bar model':' · checkpoint model'):
      (live5?' · live per-bar model (5m)':' · fit '+(j.model_fitted||'?')))
    +(main?' · scrub the slider to move through the day':'')).replace(/"/g,"'");
  let hold='';
  if(j.hold)hold=' · <span style="color:'+(j.hold.dir>0?'var(--up)':'var(--dn)')
    +'" title="on days that opened with this kind of morning trend, it was still intact at the close this often">morning '
    +(j.hold.dir>0?'UP':'DOWN')+' trend — holds into the close '+Math.round(100*j.hold.p)+'% of the time</span>';
  let tier='';
  if(j.tier==='red')tier=' · <span style="color:#e05a6d;font-weight:600" title="the trend lost VWAP AND market breadth faded vs an hour ago — historically reversal odds jump from 17% to ~43%">⚠ DE-RISK — trend cracking, breadth fading</span>';
  else if(j.tier==='amber')tier=' · <span style="color:#e0a43a;font-weight:600" title="two closes on the wrong side of VWAP after a long one-sided run — an early reversal warning (right ~1 in 3 times, ~30min lead)">⚠ caution — trend just lost VWAP</span>';
  return '<div class="mrow"><span class="mname">'+name+'</span><span class="mwhy" title="'+tip+'">'
    +(main?'<span style="color:var(--dim)">'+dtWhen(j)+' — </span>':'')
    +dtStance(j)+hold+tier
    +'</span></div>';}
function daytypeHtml(){if(!daytypeData)return '';
  return dtRow(daytypeData.SPY,'day-type',true)
    +dtRow(daytypeData.QQQ,'&nbsp;· QQQ',false)
    +dtRow(daytypeData.IWM,'&nbsp;· IWM',false)
    +dtRow(daytypeData.DIA,'&nbsp;· DIA',false);}
let _lastLiveHead='';                                 // cache: survive a transient empty daytypeData so the
function liveHeadHtml(){                              // banner head never blinks to the fallback (Amir 2026-07-08)
  const j=daytypeData&&daytypeData.SPY;               // SPY 5m per-bar day-type = today's live market read
  if(!j||j.error||j.mode!=='live-intraday'||!j.probs)return _lastLiveHead;
  let tier='';
  if(j.tier==='red')tier=' <span class="dtier" style="color:#e05a6d" title="the trend lost VWAP AND breadth faded vs an hour ago — reversal odds historically jump ~17%→~43%">⚠ DE-RISK</span>';
  else if(j.tier==='amber')tier=' <span class="dtier" style="color:#e0a43a" title="two closes on the wrong side of VWAP after a one-sided run — an early reversal warning (~1 in 3, ~30min lead)">⚠ caution</span>';
  const st=(j.stance||'').replace(/\s*\(skill [^)]*\)/,'');
  const over=!!j.session_over;                        // post-close: same read FROZEN at the 16:00 bell
  _lastLiveHead='<span class="mlivehead" title="'+(over
      ?'TODAY at the CLOSE — the SPY 5m day-type read frozen at the 16:00 bell (session complete). Click the bar to expand QQQ/IWM/DIA + the prior-close breakdown.'
      :'TODAY’s LIVE intraday read — SPY 5m per-bar day-type model; the primary market read while the market is open. Click the bar to expand QQQ/IWM/DIA + the prior-close breakdown.')+'">'
    +(over?'<b style="color:var(--acc)">■ TODAY at CLOSE</b> ':'<b style="color:var(--up)">● LIVE</b> ')
    +'<span style="color:'+(j.color||'var(--txt)')+';font-weight:600">'+st+'</span>'+tier+'</span>';
  return _lastLiveHead;}
function notifChipHtml(){                             // 🔔 N chip in the BANNER (main page) — todays grade-A
  if(!notifData||!notifData.length)return '';         // alerts at a glance; click opens the Map panel
  return '<span class="mwarm" style="cursor:pointer;color:var(--acc)" onclick="openMap()" '
    +'title="'+notifData.length+' grade-A alert(s) fired today — click to open the list (each row opens its chart)">🔔 '
    +notifData.length+'</span>';}
function warmChipHtml(dh){                            // ⚠ data warming chip — '' unless the live 5m store is filling/rebuilding
  if(!dh||!dh.warming)return '';
  const t=((dh.reason||'live data is still filling')
    +' — live 5m readings (day-type, EP/RVOL radar, internals) may be incomplete until this clears; auto-updates every ~60s').replace(/"/g,"'");
  return '<span class="mwarm" title="'+t+'">⚠ data warming</span>';}
function crossNewsHtml(){                             // cross-instrument tell: proxy news -> the UNDERLYING's reaction (Amir 2026-07-07)
  if(!crossNewsData||!crossNewsData.reads)return '';
  const notable=crossNewsData.reads.filter(r=>r&&r.tell&&r.tell!=='aligned'&&r.warn);
  if(!notable.length)return '';
  return notable.map(r=>{
    const bull=r.tell==='bad_news_resilient';        // bearish events but the underlying held up = BULLISH tell
    const col=bull?'var(--up)':'var(--dn)';
    const tip=('proxy events -> the underlying\'s move ('+r.n_bearish+' bearish / '+r.n_bullish+' bullish events, '
      +r.label+' '+(r.move_pct>0?'+':'')+r.move_pct+'%). Hand me data/cross_news/latest.json for a deeper read.').replace(/"/g,"'");
    return '<div class="mrow"><span class="mname" style="color:'+col+'">'+(bull?'resilience':'rejection')
      +'</span><span class="mwhy" title="'+tip+'">'+(r.warn||'')+'</span></div>';
  }).join('');}
function replayIdx(){if(!replayT)return 78;const p=replayT.split(':');return Math.max(0,Math.min(78,Math.round((p[0]*60+ +p[1]-570)/5)));}
function tsliderHtml(){   // time-machine slider: 78 five-min stops (09:30→15:55) + LIVE; disabled outside the ~60-session 5m window
  const inWin=(Date.now()-Date.parse(curAsof()))/864e5 < 95;
  if(!inWin)return '<div class="tsbar"><span class="tslabel" style="color:var(--dim)">⏱ replay — no intraday for this date</span></div>';
  const idx=replayIdx(), lbl=idx>=78?'● LIVE':tsTime(idx)+' ET';
  return '<div class="tsbar"><span class="tslabel" title="scrub the session (5-min stops); snap right = LIVE">⏱ '+lbl+'</span>'
    +'<input type="range" id="tslider" min="0" max="78" step="1" value="'+idx+'">'
    +'<button class="tool'+(idx>=78?' on':'')+'" id="tslive" title="jump back to LIVE / close">LIVE</button></div>';}
function pbSide(title,arr,cls){if(!arr||!arr.length)return '';   // long (green) / short (red) candidate list
  return '<div class="mrow"><span class="mname '+cls+'">'+title+'</span><span class="mwhy">'
    +arr.map(i=>'<div class="pbidea">'+i.note+'</div>').join('')+'</span></div>';}
function playbookHtml(){if(!playbookData)return '<div class="mrow"><span class="mname">playbook</span><span class="mwhy" style="color:var(--dim)">loading…</span></div>';
  if(playbookData.error)return '';
  return '<div class="mrow"><span class="mname" style="color:var(--acc)">playbook</span><span class="mwhy" style="color:var(--txt)">'+(playbookData.summary||'')+'</span></div>'
    +pbSide('long',playbookData.long_ideas,'up')+pbSide('short',playbookData.short_ideas,'dn');}
function drawMarket(){if(tsDrag)return;    // re-render resumes on slider release (onchange -> reloadAwareness)
  const el=document.getElementById('marketbar'),m=marketData;
  if(!el)return;
  if(!m||!m.close){                        // banner not ready (e.g. /api/market slow during a live re-seed):
    if(m&&m.error){el.innerHTML='';return;}   // still surface WHY it's slow via the lightweight data-health poll
    const wc=warmChipHtml(dataHealth);
    if(wc)el.innerHTML='<div class="mhead">'+wc+'<span class="msum" style="margin-left:8px">market read loading… (live data is warming)</span></div>';
    return;}
  const i=m.internals, L=m.live;
  const strip=i?('<span class="mstrip" title="live 5m internals — A/D advancers:decliners · TRIN Arms index · TICK up−down on the last bar · %&gt;VWAP names above their session VWAP · 5m hi/lo = names printing a NEW SESSION high/low on the CURRENT 5m bar (intraday ticks, noisy/inflated early — NOT 52-week or NYSE new highs)">A/D '+i.ad_ratio+':1 · TRIN '+(i.trin==null?'—':i.trin)+' · TICK '
    +(i.tick>0?'+':'')+i.tick+' · '+i.pct_vwap+'% &gt;VWAP · '+i.new_hi+'/'+i.new_lo+' 5m hi/lo</span>'):'';
  const liveBadge=L?'<span class="mlive" title="intraday 5m data is live for this read">● LIVE</span>':'';
  const dh=dataHealth||m.data_health;                // prefer the independent poll (survives a slow /api/market)
  const warnChip=warmChipHtml(dh);                   // '' unless the live 5m store is warming/rebuilding
  // headline = the map's top call (plain language); the composite score shrinks to the prior-close chip
  const tc=(mapData&&mapData.calls&&mapData.calls.length)?(mapData.calls.find(c=>c.scope==='market')
    ||mapData.calls.find(c=>c.scope==='index')||mapData.calls.find(c=>c.side!=='avoid')||mapData.calls[0]):null;
  const scls=tc?(tc.side==='long'?'up':(tc.side==='short'?'dn':'amb')):'';
  const headline=tc?('<span class="mcall '+scls+'">'+tc.side.toUpperCase()+'</span><span class="msum" title="'+(tc.expect||'')+'">'+tc.phrase+'</span>')
    :('<span class="msum">'+(m.close.summary||'')+'</span>');
  // PROMINENCE (Amir 2026-07-07): while live, TODAY's live read LEADS; the settled composite demotes to a dim
  // 'prior close' chip. Market closed -> the composite leads (nothing live to promote).
  const comp=(m.health&&!m.health.error)?m.health:m.close;
  const lh=L?liveHeadHtml():'';
  const head=L?((lh||liveBadge)+warnChip+notifChipHtml()+mBadge(comp,'prior close',true))
              :(mBadge(comp,'CLOSE')+warnChip+notifChipHtml());
  el.innerHTML='<div class="mhead" id="mtoggle" title="click to expand today live + prior-close breakdown">'
    +head+strip
    +headline
    +'<span class="mexp">'+(marketOpen?'▲':'▼')+'</span></div>'
    +tsliderHtml()
    +'<div class="mdetail" id="mdetail" style="display:'+(marketOpen?'block':'none')+'">'
    +(i?'<div class="mrow"><span class="mname">internals</span><span class="mwhy">'+i.why+'</span></div>':'')
    +'<div id="dtwrap">'+daytypeHtml()+'</div>'
    +(function(){const rx=m.rsix_breadth;if(!rx)return '';   // word only, and only when extreme (Amir 2026-07-05)
      const cap=rx.atl_pct>=1, eup=rx.ath_pct>=1;
      if(!cap&&!eup)return '';
      return '<div class="mrow"><span class="mname">extremes</span><span class="mwhy">'
        +(cap?'<b style="color:#39c4d8;font-size:15px" title="'+rx.atl_pct+'% of '+rx.n+' active stocks are at ALL-TIME RSI lows — panic-level selling breadth, historically a mean-revert zone">🔻 CAPITULATION</b>':'')
        +(cap&&eup?' &nbsp; ':'')
        +(eup?'<b style="color:#e05656;font-size:15px" title="'+rx.ath_pct+'% of '+rx.n+' active stocks are at ALL-TIME RSI highs — blow-off-level buying breadth">🔥 EUPHORIA</b>':'')
        +'</span></div>';})()
    +weatherHtml(m.weather)+crossNewsHtml()
    +healthHtml()+groupsHtml()+hintHtml()+structGridHtml()+rs4hHtml()+outlookHtml()+'</div>';
  document.getElementById('mtoggle').onclick=()=>{marketOpen=!marketOpen;
    if(marketOpen&&!groupsData)loadGroups();if(marketOpen&&!daytypeData)loadDaytype();
    if(marketOpen&&!outlookData)loadOutlook();if(marketOpen&&!rs4hData)loadRs4h();drawMarket();};
  const sl=document.getElementById('tslider');                       // scrub live on drag, reload on release
  if(sl){sl.onpointerdown=()=>{tsDrag=true;};
    sl.onpointercancel=()=>{tsDrag=false;};
    sl.onpointerup=()=>{setTimeout(()=>{if(tsDrag){tsDrag=false;drawMarket();}},150);};  // click w/o value change: no 'change' fires
    sl.oninput=()=>{const k=+sl.value,e=document.querySelector('.tslabel');if(e)e.textContent='⏱ '+(k>=78?'● LIVE':tsTime(k)+' ET');
      replayT=k>=78?null:tsTime(k);                                  // live-follow: day-type row tracks the drag
      clearTimeout(dtScrubT);dtScrubT=setTimeout(()=>{daytypeData=null;daytypeReq=false;loadDaytype();},120);};
    sl.onchange=()=>{tsDrag=false;clearTimeout(dtScrubT);const k=+sl.value;replayT=k>=78?null:tsTime(k);reloadAwareness();};}
  const lv=document.getElementById('tslive');if(lv)lv.onclick=()=>{replayT=null;reloadAwareness();};
  document.documentElement.style.setProperty('--barh',Math.round(el.getBoundingClientRect().bottom)+'px');}  // drawers start below the bar (toolbar+banner)
let mtfData=null, mtfGoggles='bull', mtfSym=null;   // 'bull' = consider a long, 'bear' = consider a short
let mtfCollapsed=localStorage.getItem('mtfCollapsed')==='1';   // collapsed multi-TF pane persists
function renderMtf(sym,asof){const el=document.getElementById('mtf');if(!el||!ONDEMAND)return;  // multi-TF awareness grid
  mtfSym=sym;                                        // asof: history fires pass the FIRE date so
  if(mtfCollapsed){mtfData=null;drawMtf();return;}   // structure/EMAs/rider read as of the breakout
  el.innerHTML='<div style="padding:6px" class="d">multi-TF …</div>';
  const di=document.getElementById('asof');
  fetch('/api/mtf?symbol='+encodeURIComponent(sym)+'&date='+encodeURIComponent(asof||(di?di.value:''))).then(r=>r.json())
    .then(j=>{if(mtfSym===sym){mtfData=j;drawMtf();}}).catch(()=>{el.innerHTML='';});}
function drawMtf(){const el=document.getElementById('mtf'),j=mtfData;if(!el)return;
  if(mtfCollapsed){el.innerHTML='<div style="padding:3px 6px"><button class="tool" id="mtfcol" '
    +'title="expand the multi-TF pane" style="padding:1px 8px;font-size:11px">▸ multi-TF</button></div>';
    const c0=document.getElementById('mtfcol');
    if(c0)c0.onclick=()=>{mtfCollapsed=false;localStorage.setItem('mtfCollapsed','0');
      if(mtfData)drawMtf();else if(mtfSym)renderMtf(mtfSym);};
    return;}
  if(!j)return;
  const P=j.profile||{},tfs=j.tfs||[],MIN=j.min_streak||0,G=mtfGoggles;
  const fav=c=>c==null?null:(G==='bull'?c:!c);                  // favorable for the current goggles?
  const fc=c=>{const f=fav(c);return f==null?'muted':(f?'up':'dn');};
  const yn=b=>'<span class="'+fc(b)+'">'+(b===true?'✓':(b===false?'✗':'·'))+'</span>';
  const bull=s=>[s.gt_ema10,s.gt_ema20,s.gt_sma50,s.gt_sma150,s.gt_sma200,
    s.trend_dir==='up'?true:(s.trend_dir==='down'?false:null),
    s.rider_dir>0?true:(s.rider_dir<0?false:null),
    s.stack>0?true:(s.stack<0?false:null), s.tilt>0?true:(s.tilt<0?false:null)];
  const score=s=>{let n=bull(s).reduce((a,c)=>a+(c==null?0:(fav(c)?1:-1)),0);
    if(s.atr_ext_50!=null)n+=(Math.abs(s.atr_ext_50)<4?1:-1);   // not over-extended (<4 ATR from 50) — goggles-independent
    return n;};
  const TR={up:'▲up',down:'▼dn',contracting:'◇contr',expanding:'◆expand',rectangle:'▭rect',range:'range',transition:'trans'};
  const trend=d=>d==='up'?'<span class="'+fc(true)+'">▲up</span>':(d==='down'?'<span class="'+fc(false)+'">▼dn</span>':'<span class="amb">'+(TR[d]||d||'—')+'</span>');
  const rider=s=>{if(!s.rider_dir)return '<span class="muted">·</span>';const up=s.rider_dir>0,cls=fc(up),strong=s.rider_streak>=MIN;
    return '<span class="'+cls+'">'+(up?'▲':'▼')+'</span><span class="'+(strong?cls:'muted')+'">'+s.rider_streak+'</span>'+(s.rider_touch?'<span class="amb">ᵗ</span>':'');};
  const gog='<button class="tool" id="mtfgog" title="view the matrix as if considering a long (bull) or a short (bear)" '
    +'style="padding:1px 8px;font-size:11px">'+(G==='bull'?'🐂 bull goggles':'🐻 bear goggles')+'</button>';
  const col='<button class="tool" id="mtfcol" title="collapse the multi-TF pane" '
    +'style="padding:1px 8px;font-size:11px">▾ collapse</button>';
  // per-TF risk/reward from that TF's nearest S/R, framed by the goggles (bull = long framing);
  // r/Rt = reward to the SELECTED row's measured-move target ÷ the TF's risk
  const SHORTS=['double_top','head_shoulders','downtrend','ema_rider_bear'];
  const tgt=(window.curRow&&curRow.target!=null)?curRow.target:null;
  const rrf=v=>v==null?'<span class="muted">·</span>':'<span class="'+(v>=2?'up':(v<1?'dn':''))+'">'+v.toFixed(1)+'</span>';
  const rrCells=s=>{if(s.px==null)return '<td class="muted">·</td>'.repeat(4);
    const long=(G==='bull');
    const risk=long?(s.sup!=null?s.px-s.sup:null):(s.res!=null?s.res-s.px:null);
    const rew =long?(s.res!=null?s.res-s.px:null):(s.sup!=null?s.px-s.sup:null);
    const rr=(risk>0&&rew>0)?rew/risk:null;
    let rrt=null;
    if(tgt!=null&&risk>0){const tl=SHORTS.includes(curRow.setup)?(s.px-tgt):(tgt-s.px);
      if(tl>0)rrt=tl/risk;}
    const bg=v=>v==null?'':' style="background:'+(v<1?'rgba(224,86,86,.12)'
      :'rgba(47,191,143,'+Math.min(.30,.06+v*.06).toFixed(2)+')')+'"';   // same tint scale as the main table
    return '<td>'+(risk!=null&&risk>0?risk.toFixed(2):'·')+'</td><td>'+(rew!=null&&rew>0?rew.toFixed(2):'·')+'</td>'
      +'<td'+bg(rr)+'>'+rrf(rr)+'</td><td'+bg(rrt)+'>'+rrf(rrt)+'</td>';};
  const asofTag=j.historical?'<span class="hd" style="margin-left:8px">as of '+j.asof+' — the fire day'
    +(j.intraday_na?'; intraday n/a (no 5-min history for this symbol/date)':' · intraday from the 5-min store')+'</span>':'';
  let h='<div style="padding:3px 6px">'+col+gog+asofTag+'</div><table class="mtf"><thead><tr><th>TF</th>'
    +'<th title="net favorable count across this row (goggles-framed)">align</th><th>trend</th><th>rider</th>'
    +'<th title="close vs EMA10">E10</th><th title="close vs EMA20">E20</th><th title="close vs SMA50">S50</th>'
    +'<th title="close vs SMA150">S150</th><th title="close vs SMA200">S200</th><th>stack</th><th>tilt</th>'
    +'<th title="ATR-multiples from the 50-MA (extension health)">ext50</th>'
    +'<th title="close − nearest TF support (bull) / resistance − close (bear)">$risk</th>'
    +'<th title="nearest TF resistance − close (bull) / close − support (bear)">$rew</th>'
    +'<th title="$rew ÷ $risk on this timeframe">r/R</th>'
    +'<th title="reward to the selected setup\'s measured-move target ÷ this TF\'s risk">r/R→tgt</th>'
    +'<th title="run every setup detector on this timeframe\'s chart">fits</th></tr></thead><tbody>';
  const KIND=j.kind||'stock';
  const mtd=b=>{const f=fav(b);   // MA cells: faint goggles-framed fill — the row reads as a color stripe
    const bg=f==null?'':(f?'background:rgba(47,191,143,.10)':'background:rgba(224,86,86,.10)');
    return '<td style="'+bg+'">'+yn(b)+'</td>';};
  const apill=a=>{const al=Math.min(.38,Math.abs(a)*.05+.08);   // align score as a colored capsule
    const bg=a>0?'rgba(47,191,143,'+al.toFixed(2)+')':(a<0?'rgba(224,86,86,'+al.toFixed(2)+')':'transparent');
    return '<td><span style="display:inline-block;min-width:26px;border-radius:8px;padding:0 6px;background:'+bg+'">'+(a>0?'+':'')+a+'</span></td>';};
  for(const tf of tfs){const s=P[tf];
    const cur=(tf===window.curTf)?' class="cur"':'';
    if(!s){const why=(j.intraday_na&&/[mh]$/.test(tf))?'n/a as of '+j.asof+' — no 5-min history for this symbol/date (store covers the active universe since 2023-06)':'no data';
      h+='<tr data-tf="'+tf+'"'+cur+'><td>'+tf+'</td><td colspan="16" class="muted">'+why+'</td></tr>';continue;}
    const a=score(s);   // ext50 cell: Extension Health Badge (goggles-directional, kind-aware tiers)
    h+='<tr data-tf="'+tf+'"'+cur+'><td>'+tf+'</td>'+apill(a)+'<td>'+trend(s.trend_dir)+'</td><td>'+rider(s)+'</td>'
      +mtd(s.gt_ema10)+mtd(s.gt_ema20)+mtd(s.gt_sma50)+mtd(s.gt_sma150)+mtd(s.gt_sma200)
      +'<td class="'+fc(s.stack>0?true:(s.stack<0?false:null))+'">'+s.stack+'</td><td class="'+fc(s.tilt>0?true:(s.tilt<0?false:null))+'">'+s.tilt+'</td>'
      +'<td>'+(s.atr_ext_50!=null?extBadge(s.atr_ext_50,KIND,tf)+s.atr_ext_50:'')+'</td>'+rrCells(s)
      +'<td><button class="tool" style="padding:1px 7px;font-size:12px" '
      +'onclick="event.stopPropagation();tfSetups(\''+tf+'\')">🔎</button></td></tr>';}
  el.innerHTML=h+'</tbody></table><div id="mtfhits" style="padding:2px 8px;font:11px var(--mono)"></div>';
  el.querySelectorAll('tr[data-tf]').forEach(tr=>tr.onclick=()=>chartTF(mtfSym,tr.dataset.tf));  // click a TF row -> chart that timeframe
  const b=document.getElementById('mtfgog');if(b)b.onclick=()=>{mtfGoggles=(mtfGoggles==='bull'?'bear':'bull');
    drawMtf();render();   // ext badges everywhere are goggles-directional -> re-render the visible surfaces
    const tc=document.getElementById('tkrcard');
    if(tc&&tc.style.display!=='none'){if(groupCard)drawGroupCard();else if(curSym)loadTicker(curSym);}};
  const c=document.getElementById('mtfcol');
  if(c)c.onclick=()=>{mtfCollapsed=true;localStorage.setItem('mtfCollapsed','1');drawMtf();};}
const CHARTS_MAX=150;   // bound the on-demand chart-payload cache so a long session can't grow unbounded
function cacheChart(key,j){CHARTS[key]=j;            // string keys preserve insertion order -> evict oldest
  const ks=Object.keys(CHARTS);
  for(let i=0;i<ks.length-CHARTS_MAX;i++)delete CHARTS[ks[i]];
  return j;}
async function showChart(key){let data=CHARTS[key];
  const p=key.split('|'),sym=p[0],setup=p[1],tf=p[2];
  curSym=sym;window.curDen=null;const tc=document.getElementById('tkrcard');
  if(tc&&tc.style.display!=='none')loadTicker(sym);   // card follows the selected row
  window.curRow=(TABLES[setup]||[]).find(x=>x.symbol===sym&&x.tf===tf)||null;  // goggles r/R-target
  renderMtf(sym);
  const row=(TABLES[setup]||[]).find(x=>x.symbol===sym&&x.tf===tf)||{};
  const head=document.getElementById('charthead'),el=document.getElementById('chart');
  const tfTag=tf?'  ['+tf+']':'';
  if(!data&&ONDEMAND){   // live app: fetch this one chart from RAM (nothing embedded up front)
    head.innerHTML=sym+'  —  '+LABELS[setup]+tfTag+'<div class="sub">'+(row.name||'')+'</div>';
    el.innerHTML='<div class="empty">loading chart…</div>';
    try{const di=document.getElementById('asof');
      const r=await fetch('/api/chart',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({symbol:sym,tf:tf,date:di?di.value:null,hit:row})});
      const j=await r.json();if(j&&!j.error)data=cacheChart(key,j);}catch(e){}
    if(key!==selKey)return;}   // user clicked another row while this was loading
  if(!data){head.innerHTML=sym+'  —  '+LABELS[setup]+tfTag+'<div class="sub">'+(row.name||'')+'</div>';
    el.innerHTML='<div class="empty">chart unavailable — '
      +'<a href="'+tvUrl(row)+'" target="_blank" style="color:var(--acc)">open in TradingView ↗</a></div>';
    disposeChart();return;}
  head.innerHTML=sym+'  —  '+LABELS[setup]+tfTag
    +' <a href="'+tvUrl(row)+'" target="_blank" style="color:var(--dim);font-size:11px">TV ↗</a>'
    +'<div class="sub">'+(row.name||'')+'   level '+(data.level?data.level.toFixed(2):'-')
    +'   trigger '+(data.trigger?data.trigger.toFixed(2):'-')
    +(data.uptrend_from?'   uptrend shaded':'')
    +(data.downtrend_from?'   downtrend shaded':'')
    +(data.segments?'   structure shaded (base/leg/cons)':'')
    +(data.daily?'   ← daily history (zoom out)':'')
    +(data.asof?'   as-of '+data.asof:'')+'</div>';
  drawChart(data);}
async function chartTF(sym,tf,asof){   // click a matrix TF row -> chart that timeframe (empty hit; no level/shading)
  window.curDen=null;window.curTf=tf;   // asof: history fire clicks pan/zoom the chart to THAT date
  document.querySelectorAll('#mtf tr[data-tf]').forEach(tr=>tr.classList.toggle('cur',tr.dataset.tf===tf));
  selKey=sym+'|_mtf|'+tf;
  const head=document.getElementById('charthead'),el=document.getElementById('chart');
  head.innerHTML=sym+'  —  '+tf+'<div class="sub">timeframe chart…</div>';
  el.innerHTML='<div class="empty">loading chart…</div>';
  try{const di=document.getElementById('asof');
    const r=await fetch('/api/chart',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({symbol:sym,tf:tf,date:asof||(di?di.value:null),hit:{}})});
    const j=await r.json();
    if(j&&!j.error){j.symbol=sym;head.innerHTML=sym+'  —  '+tf+'<div class="sub">timeframe chart'+(j.daily?'   ← daily history (zoom out)':'')+'</div>';drawChart(j);}
    else{disposeChart();el.innerHTML='<div class="empty">chart unavailable for '+tf+(/m|h/.test(tf)?' (click ⬇ Intraday first)':'')+'</div>';}
  }catch(e){disposeChart();el.innerHTML='';}}
function disposeChart(){   // free the lightweight-charts canvas + listeners + ResizeObserver (autoSize)
  if(chart){try{chart.remove();}catch(e){}chart=null;}}
function drawChart(data){
  const el=document.getElementById('chart');
  window._lastChartData=data;   // remember for the ⚡ Rider toggle (re-render without a re-fetch)
  const candles=data.t.map((t,j)=>({time:t,open:data.o[j],high:data.h[j],low:data.l[j],close:data.c[j]}));
  const volume=data.t.map((t,j)=>({time:t,value:data.v[j],color:data.c[j]>=data.o[j]?'#1f6f50':'#7a2f3d'}));
  disposeChart();
  el.innerHTML='';
  chart=LightweightCharts.createChart(el,{layout:{background:{color:'#0b0f14'},textColor:'#6b7a8c'},
    grid:{vertLines:{color:'#131a22'},horzLines:{color:'#131a22'}},
    crosshair:{mode:(LightweightCharts.CrosshairMode?LightweightCharts.CrosshairMode.Normal:0)},   // free crosshair, no bar-snap magnet
    rightPriceScale:{borderColor:'#1d2630'},timeScale:{borderColor:'#1d2630',timeVisible:!!data.intraday},autoSize:true});
  if(data.uptrend_from&&data.uptrend_to){ // shade the up segment green
    const up=chart.addHistogramSeries({priceScaleId:'bgup',lastValueVisible:false,priceLineVisible:false});
    chart.priceScale('bgup').applyOptions({scaleMargins:{top:0,bottom:0},visible:false});
    up.setData(data.t.map(t=>({time:t,value:1,
      color:(t>=data.uptrend_from&&t<=data.uptrend_to)?'rgba(47,191,143,0.13)':'rgba(0,0,0,0)'})));}
  if(data.downtrend_from&&data.downtrend_to){ // shade the down segment red
    const dn=chart.addHistogramSeries({priceScaleId:'bgdn',lastValueVisible:false,priceLineVisible:false});
    chart.priceScale('bgdn').applyOptions({scaleMargins:{top:0,bottom:0},visible:false});
    dn.setData(data.t.map(t=>({time:t,value:1,
      color:(t>=data.downtrend_from&&t<=data.downtrend_to)?'rgba(224,90,109,0.13)':'rgba(0,0,0,0)'})));}
  if(data.asof){ // shade the outcome zone (bars after the signal date)
    const bg=chart.addHistogramSeries({priceScaleId:'bgsh',lastValueVisible:false,priceLineVisible:false});
    chart.priceScale('bgsh').applyOptions({scaleMargins:{top:0,bottom:0},visible:false});
    bg.setData(data.t.map(t=>({time:t,value:1,color:t>data.asof?'rgba(232,184,75,0.07)':'rgba(0,0,0,0)'})));}
  if(data.segments&&data.segments.length){ // SB Continuation: base / leg / consolidation bands
    const SC={base:'rgba(120,130,150,0.17)',leg:'rgba(47,191,143,0.17)',cons:'rgba(232,184,75,0.20)'};
    const sg=chart.addHistogramSeries({priceScaleId:'bgseg',lastValueVisible:false,priceLineVisible:false});
    chart.priceScale('bgseg').applyOptions({scaleMargins:{top:0,bottom:0},visible:false});
    sg.setData(data.t.map(t=>{let col='rgba(0,0,0,0)';
      for(const s of data.segments){if(t>=s.from&&t<=s.to){col=SC[s.kind]||col;break;}}
      return{time:t,value:1,color:col};}));}
  if(data.qm_pole){ // QM: shade the pole (the move leg) green
    const qp=chart.addHistogramSeries({priceScaleId:'bgqp',lastValueVisible:false,priceLineVisible:false});
    chart.priceScale('bgqp').applyOptions({scaleMargins:{top:0,bottom:0},visible:false});
    qp.setData(data.t.map(t=>({time:t,value:1,
      color:(t>=data.qm_pole[0]&&t<=data.qm_pole[1])?'rgba(47,191,143,0.12)':'rgba(0,0,0,0)'})));}
  if(data.qm_cons){ // QM: shade the consolidation amber
    const qc=chart.addHistogramSeries({priceScaleId:'bgqc',lastValueVisible:false,priceLineVisible:false});
    chart.priceScale('bgqc').applyOptions({scaleMargins:{top:0,bottom:0},visible:false});
    qc.setData(data.t.map(t=>({time:t,value:1,
      color:(t>=data.qm_cons[0]&&t<=data.qm_cons[1])?'rgba(232,184,75,0.10)':'rgba(0,0,0,0)'})));}
  if(window.riderOverlay&&data.rider&&data.rider.bg){   // ⚡ EMA-Rider streak shading (green above / red below)
    const rb=chart.addHistogramSeries({priceScaleId:'bgrider',lastValueVisible:false,priceLineVisible:false});
    chart.priceScale('bgrider').applyOptions({scaleMargins:{top:0,bottom:0},visible:false});
    rb.setData(data.t.map((t,j)=>({time:t,value:1,color:data.rider.bg[j]||'rgba(0,0,0,0)'})));}
  const cs=chart.addCandlestickSeries({upColor:'#2fbf8f',downColor:'#e05a6d',
    wickUpColor:'#2fbf8f',wickDownColor:'#e05a6d',borderVisible:false});
  cs.setData(candles);
  window.csMain=cs;measSetup((data.symbol||selKey||'')+'|'+(data.intraday?'i':'d'));
  const tset=new Set(data.t), marks=[];
  if(data.segments){for(const s of data.segments){if(tset.has(s.from))   // label each base/leg/consolidation
    marks.push({time:s.from,position:'aboveBar',shape:'circle',color:'#cfd8e3',text:s.label});}}
  if(data.asof)marks.push({time:data.asof,position:'belowBar',shape:'arrowUp',color:'#e8b84b',text:'as-of'});
  if(data.marks)for(const m of data.marks){if(tset.has(m.date))            // pattern anatomy markers
    marks.push({time:m.date,position:m.pos==='below'?'belowBar':'aboveBar',
      shape:m.pos==='below'?'arrowUp':'arrowDown',color:'#e0a43a',text:m.text});}
  if(data.segline&&data.segline.length){const nl=chart.addLineSeries({color:'#e0a43a',lineWidth:2,
    priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
    nl.setData(data.segline.filter(p=>p.date&&tset.has(p.date)).map(p=>({time:p.date,value:p.price})));}
  if(data.xlines)for(const l of data.xlines){if(l.price!=null)
    cs.createPriceLine({price:l.price,title:l.title||'',color:'#5aa2e0',lineStyle:2,lineWidth:1});}
  if(window.histMarks&&histMarks.sym===data.symbol)                       // setup-history fire markers
    for(const f of histMarks.fires){if(tset.has(f.date))
      marks.push({time:f.date,position:'belowBar',shape:'arrowUp',color:'#ff3366',
        text:f.state.replace('breakout:','')+(f.pos!=null?' @'+f.pos:'')});}   // pOS = the oversold wick price
  if(data.rsix)for(const m of data.rsix){if(m.date&&tset.has(m.date))      // ◉ all-time RSI extreme hit:
    marks.push({time:m.date,position:m.side==='atl'?'belowBar':'aboveBar', // price + day it printed
      shape:'circle',color:m.side==='atl'?'#39c4d8':'#e05656',
      text:'◉'+m.rsi+' @'+m.price});}
  if(window.riderOverlay&&data.rider&&data.rider.marks)                    // ⚡ arm/save/break state-change marks
    for(const m of data.rider.marks){if(tset.has(m.time))marks.push(m);}
  if(marks.length){marks.sort((a,b)=>a.time<b.time?-1:(a.time>b.time?1:0));cs.setMarkers(marks);}
  window._fireCtx={cs:cs,tset:tset,base:marks.slice(),intraday:!!data.intraday,   // ◎ Fires overlay
    sym:data.symbol||(selKey||'').split('|')[0]||null};
  if(fireSel.size&&!data.intraday)applyFireMarks();
  const vs=chart.addHistogramSeries({priceScaleId:'vol',priceFormat:{type:'volume'}});
  chart.priceScale('vol').applyOptions({scaleMargins:{top:0.82,bottom:0}});
  vs.setData(volume);
  [['ema10','#e8b84b'],['ema20','#4b8fd6'],['sma50','#7a6bd6'],['sma150','#d6864b'],['sma200','#8aa0b8']]  // the 5 matrix MAs
    .forEach(([k,col])=>{const a=data[k];if(!a)return;const pts=[];
      data.t.forEach((t,j)=>{if(a[j]!=null)pts.push({time:t,value:a[j]});});
      if(pts.length)chart.addLineSeries({color:col,lineWidth:1,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false}).setData(pts);});
  if(cur === 'ema_cross' && data.larsson_ema32 && data.larsson_ema58) {
      const fastPts = [];
      const slowPts = [];
      const markers = [];
      data.t.forEach((t, j) => {
          if (data.larsson_ema32[j] != null) fastPts.push({time:t, value:data.larsson_ema32[j]});
          if (data.larsson_ema58[j] != null) slowPts.push({time:t, value:data.larsson_ema58[j]});
          if (j > 0 && data.larsson_state && data.larsson_state[j] && data.larsson_state[j] !== data.larsson_state[j-1]) {
              const st = data.larsson_state[j];
              if (st === 'yellow') markers.push({time:t, position:'belowBar', color:'#FFC107', shape:'arrowUp', text:'Yellow'});
              if (st === 'blue') markers.push({time:t, position:'aboveBar', color:'#2196F3', shape:'arrowDown', text:'Blue'});
          }
      });
      const fastLine = chart.addLineSeries({color:'#FFC107', lineWidth:2, priceLineVisible:false, lastValueVisible:false});
      fastLine.setData(fastPts);
      if (markers.length) fastLine.setMarkers(markers);
      chart.addLineSeries({color:'#2196F3', lineWidth:2, priceLineVisible:false, lastValueVisible:false}).setData(slowPts);
  }
  if(data.daily&&data.t.length){   // daily-close history left of the intraday window (where we came from)
    const t0=data.t[0],dl=data.daily.filter(p=>p[0]<t0).map(p=>({time:p[0],value:p[1]}));
    if(dl.length)chart.addLineSeries({color:'#5a6b8c',lineWidth:1,priceLineVisible:false,
      lastValueVisible:false,crosshairMarkerVisible:false}).setData(dl);}
  if(data.stack_levels&&data.stack_levels.length){ // breakout-level STACK: horizontal resistances,
    const lv=data.stack_levels, top=data.trigger;  // active trigger = nearest un-broken level above price (green)
    lv.forEach(p=>cs.createPriceLine({price:p,color:(p===top?'#2fbf8f':'#5a6b8c'),
      lineStyle:(p===top?0:2),title:(p===top?'trigger':'')}));}
  else{
    if(data.level)cs.createPriceLine({price:data.level,color:'#e8b84b',lineStyle:2,title:'level'});
    if(data.trigger)cs.createPriceLine({price:data.trigger,color:'#2fbf8f',lineStyle:0,title:'trigger'});}
  // Support/Resistance — strength-weighted horizontal lines (green support / red resistance), toggleable.
  // Kept subtle (dashed, faded) so the setup's own level/trigger/stack lines stay primary.
  if(window.SR_ON===undefined)window.SR_ON=true;
  window._srLines=[];
  window._srDraw=function(){
    (window._srLines||[]).forEach(l=>{try{cs.removePriceLine(l);}catch(e){}});window._srLines=[];
    if(!window.SR_ON||!data.sr_levels)return;
    data.sr_levels.forEach(L=>{const sup=L.kind==='support';
      window._srLines.push(cs.createPriceLine({price:L.price,
        color:sup?'rgba(47,191,143,0.45)':'rgba(224,90,109,0.45)',
        lineWidth:(L.strength>=15?3:(L.strength>=6?2:1)),lineStyle:2,axisLabelVisible:true,
        title:(sup?'S':'R')+'×'+L.strength}));});};
  window._srDraw();
  const nb=data.t.length; let endIdx=nb-1;                    // default zoom: recent ZOOM bars
  if(data.asof){const ai=data.t.indexOf(data.asof);if(ai>=0)endIdx=Math.min(nb-1,ai+ZOOMF);}  // up to as-of + margin
  // time-based range (not logical-index): the daily-context line prepends bars before data.t[0],
  // which shifts logical indices on the chart's merged time axis, so an index-based range would
  // no longer point at the latest candle. Using actual times is immune to that.
  chart.timeScale().setVisibleRange({from:data.t[Math.max(0,endIdx-ZOOM+1)],to:data.t[endIdx]});
  // journal date-pick: when a setup row's 📍 is active, a chart click captures the bar date
  chart.subscribeClick(param=>{
    if(jPickRow===null||!param.time)return;
    const date=typeof param.time==='string'?param.time
      :(typeof param.time==='number'?new Date(param.time*1000).toISOString().slice(0,10)
      :(param.time.year?param.time.year+'-'+String(param.time.month).padStart(2,'0')+'-'+String(param.time.day).padStart(2,'0'):''));
    if(!date)return;
    const inp=document.getElementById('jsd-'+jPickRow);
    if(inp){inp.value=date;_cancelJPick();}
  });}
['fvol','favol','fprice','fadr','flo52','fhi','frs'].forEach(id=>document.getElementById(id).oninput=render);
['fsma50','fsma200'].forEach(id=>document.getElementById(id).onchange=render);
['fema_scope','fema_base','fema_time','fema_dir','fema_sec_state','fema_thm_state'].forEach(id=>{if(document.getElementById(id))document.getElementById(id).onchange=render;});
document.getElementById('fpat').onchange=render;
document.getElementById('fstate').onchange=render;
document.getElementById('fetf').onchange=render;
// per-setup "Best" preset — best-results defaults for the ACTIVE tab (stairstep filters apply as-of-peak)
const BEST={
  gapper:{fadr:5,fvol:10,favol:500,fprice:10,fhi:0,frs:0,s50:true,s200:true,pat:'',state:''},
  hvc:{fadr:5,fvol:10,favol:500,fprice:10,fhi:0,frs:0,s50:true,s200:true,pat:'',state:''},
  delayed_hvc:{fadr:5,fvol:10,favol:500,fprice:10,fhi:0,frs:0,s50:true,s200:true,pat:'',state:''},
  flat_base:{fadr:0,fvol:5,favol:500,fprice:10,fhi:20,frs:0,s50:false,s200:true,pat:'',state:''},
  high_tight_flag:{fadr:5,fvol:5,favol:500,fprice:10,fhi:25,frs:0,s50:true,s200:true,pat:'flag',state:''},
  qm_breakout:{fadr:5,fvol:5,favol:500,fprice:10,fhi:0,frs:98,s50:false,s200:false,pat:'',state:'building'},
  uptrend:{fadr:0,fvol:5,favol:500,fprice:10,fhi:0,frs:0,s50:false,s200:false,pat:'',state:''},
  downtrend:{fadr:0,fvol:5,favol:500,fprice:10,fhi:0,frs:0,s50:false,s200:false,pat:'',state:''},
  higher_low_ma:{fadr:5,fvol:5,favol:500,fprice:10,fhi:15,frs:0,s50:false,s200:true,pat:'',state:'building'},
  undercut_rally:{fadr:5,fvol:5,favol:500,fprice:10,fhi:0,frs:0,s50:false,s200:true,pat:'',state:'building'},
  backburner:{fadr:5,fvol:5,favol:500,fprice:10,fhi:6,frs:0,s50:false,s200:true,pat:'',state:''},
  stairstep:{fadr:5,fvol:5,favol:500,fprice:10,fhi:20,frs:0,s50:true,s200:true,pat:'',state:''},
  ema_cross:{fadr:0,fvol:0,favol:0,fprice:0,fhi:0,frs:0,s50:false,s200:false,pat:'',state:''},
  ema_rider_bull:{fadr:3,fvol:5,favol:500,fprice:10,fhi:0,frs:0,s50:true,s200:true,pat:'',state:''},
  ema_rider_bear:{fadr:3,fvol:5,favol:500,fprice:10,fhi:0,frs:0,s50:false,s200:false,pat:'',state:''},
};
function bestPreset(){const p=BEST[cur]||{};const s=(id,v)=>document.getElementById(id).value=v;
  s('fadr',p.fadr||0);s('fvol',p.fvol||0);s('favol',p.favol||0);s('fprice',p.fprice||0);
  s('flo52',p.flo52||0);s('fhi',p.fhi||0);s('frs',p.frs||0);s('fpat',p.pat||'');s('fstate',p.state||'');
  document.getElementById('fsma50').checked=!!p.s50;document.getElementById('fsma200').checked=!!p.s200;
  render();}
document.getElementById('btnbest').onclick=bestPreset;
// ── ratio charts: symbol ÷ denominator (SPY / industry SPDR / theme ETF) ──
async function chartRatio(sym,den,tf){
  tf=tf||'1D';
  window.curDen=den;mtfSym=sym;                         // 🔎 setups now run on the RATIO series
  const head=document.getElementById('charthead'),el=document.getElementById('chart');
  head.innerHTML=sym+' / '+den+' <span class="hd">'+tf+'</span><div class="sub">ratio chart — rising = '+sym+' outperforming '+den
    +' · 🔎 in the multi-TF pane now checks setups ON THIS RATIO</div>';
  el.innerHTML='<div class="empty">building ratio…</div>';
  try{const di=document.getElementById('asof');
    const r=await fetch('/api/chart',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({symbol:sym,tf:tf,date:di?di.value:null,den:den,hit:{}})});
    const j=await r.json();
    if(j&&!j.error){j.symbol=sym+'/'+den;drawChart(j);}
    else el.innerHTML='<div class="empty">'+((j&&j.error)||'ratio failed')+'</div>';}
  catch(e){el.innerHTML='<div class="empty">ratio failed: '+e+'</div>';}}

// ── per-timeframe setup check: which setups does this TF's chart fit into NOW ──
window.tfHits={};
async function tfSetups(tf){const box=document.getElementById('mtfhits');
  const den=window.curDen||'';
  const tag=mtfSym+(den?'/'+den:'');
  if(box)box.innerHTML='<span class="hd">checking '+tag+' '+tf+' for setups…</span>';
  try{const di=document.getElementById('asof');
    const j=await(await fetch('/api/tf_setups?symbol='+encodeURIComponent(mtfSym)
      +'&den='+encodeURIComponent(den)+'&tf='+encodeURIComponent(tf)+'&date='+(di?di.value:''))).json();
    const hits=j.hits||[];window.tfHits={tf:tf,den:den,hits:hits,sup:j.sup,res:j.res,px:j.px};
    if(!box)return;
    if(!hits.length){box.innerHTML='<span class="hd">'+tag+' '+tf+': no setup fits right now'
      +(j.error?' ('+j.error+')':'')+'</span>';return;}
    box.innerHTML='<span class="hd">'+tag+' '+tf+' fits:</span> '+hits.map((h,i)=>
      '<span class="chip" style="cursor:pointer;border-color:var(--acc);color:var(--acc)" '
      +'onclick="chartTfHit('+i+')">'+(SETUP_LABELS[h.setup]||h.setup)+' · '+(h.state||'')+'</span>').join('');}
  catch(e){if(box)box.innerHTML='<span class="hd">setup check failed: '+e+'</span>';}}
async function chartTfHit(i){const t=window.tfHits;if(!t||!t.hits[i])return;
  const h=t.hits[i],tf=t.tf,den=t.den||'';
  const head=document.getElementById('charthead');
  try{const di=document.getElementById('asof');
    const r=await fetch('/api/chart',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({symbol:mtfSym,tf:tf,date:di?di.value:null,den:den,hit:h})});
    const j=await r.json();
    if(j&&!j.error){j.symbol=mtfSym;
      // risk/reward from the SAME frame the setups ran on (ratio setups -> ratio units)
      const SH=['double_top','head_shoulders','downtrend','ema_rider_bear'].includes(h.setup);
      const px=t.px, dp=v=>v==null?'—':(v<1?v.toFixed(4):v.toFixed(2));
      const risk=(px!=null)?(SH?(t.res!=null?t.res-px:null):(t.sup!=null?px-t.sup:null)):null;
      const rew =(px!=null)?(SH?(t.sup!=null?px-t.sup:null):(t.res!=null?t.res-px:null)):null;
      const rr=(risk>0&&rew>0)?(rew/risk).toFixed(1):'—';
      let rrt='—';
      if(h.target!=null&&risk>0){const tl=SH?(px-h.target):(h.target-px);if(tl>0)rrt=(tl/risk).toFixed(1);}
      head.innerHTML=mtfSym+(den?'/'+den:'')+'  —  '+(SETUP_LABELS[h.setup]||h.setup)+' ['+tf+'] '+(h.state||'')
        +'<div class="sub">'+(den?'ratio units · ':'')+'risk '+(risk!=null&&risk>0?dp(risk):'—')
        +' · reward '+(rew!=null&&rew>0?dp(rew):'—')+' · r/R '+rr+' · r/R-to-target '+rrt
        +(h.trigger!=null?' · trigger '+dp(h.trigger):'')+(h.target!=null?' · target '+dp(h.target):'')+'</div>';
      drawChart(j);}}
  catch(e){}}

// ── measure rulers on the main chart: SHIFT-DRAG to draw (raw cursor, no bar magnet), rulers
//    anchor to bar/price so they survive pan/zoom, persist per symbol, RIGHT-CLICK one to delete ──
const MEAS={};let measKey=null,measDrawing=null,measSvg=null;
function measEl(tag,at){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);
  for(const k in at)e.setAttribute(k,at[k]);return e;}
function measSetup(key){measKey=key;
  const el=document.getElementById('chart');if(!el)return;
  el.style.position='relative';
  measSvg=document.createElementNS('http://www.w3.org/2000/svg','svg');
  measSvg.setAttribute('style','position:absolute;inset:0;z-index:8;pointer-events:none;width:100%;height:100%');
  el.appendChild(measSvg);
  chart.timeScale().subscribeVisibleLogicalRangeChange(measRedraw);
  measRedraw();}
function measP2M(x,y){if(!chart||!window.csMain)return null;
  const l=chart.timeScale().coordinateToLogical(x),p=csMain.coordinateToPrice(y);
  return(l==null||p==null)?null:{l,p};}
function measM2P(l,p){if(!chart||!window.csMain)return null;
  const x=chart.timeScale().logicalToCoordinate(l),y=csMain.priceToCoordinate(p);
  return(x==null||y==null)?null:{x,y};}
function measRuler(g,x1,y1,x2,y2,p1,p2,l1,l2){
  const pct=(p2/p1-1)*100,bars=Math.round(l2-l1),col=pct>=0?'#2fbf8f':'#e05a6d';
  g.appendChild(measEl('rect',{x:Math.min(x1,x2),y:Math.min(y1,y2),width:Math.abs(x2-x1),
    height:Math.abs(y2-y1),fill:pct>=0?'rgba(47,191,143,.10)':'rgba(224,90,109,.10)'}));
  g.appendChild(measEl('line',{x1,y1,x2,y2,stroke:col,'stroke-width':1.5,'stroke-dasharray':'4 3'}));
  const t=measEl('text',{x:Math.max(x1,x2)+8,y:Math.min(y1,y2)+4,fill:col,
    style:'font:12px var(--mono);paint-order:stroke;stroke:#0b0f14;stroke-width:3px'});
  t.textContent=(pct>=0?'+':'')+pct.toFixed(2)+'% · '+bars+' bars · '+(p2-p1>=0?'+':'')+(p2-p1).toFixed(2);
  g.appendChild(t);}
function measRedraw(){if(!measSvg)return;measSvg.innerHTML='';
  (MEAS[measKey]||[]).forEach(m=>{const a=measM2P(m.l1,m.p1),b=measM2P(m.l2,m.p2);
    if(!a||!b)return;const g=measEl('g',{});measRuler(g,a.x,a.y,b.x,b.y,m.p1,m.p2,m.l1,m.l2);measSvg.appendChild(g);});
  if(measDrawing&&measDrawing.x2!=null){const d1=measP2M(measDrawing.x1,measDrawing.y1),
    d2=measP2M(measDrawing.x2,measDrawing.y2);
    if(d1&&d2){const g=measEl('g',{});
      measRuler(g,measDrawing.x1,measDrawing.y1,measDrawing.x2,measDrawing.y2,d1.p,d2.p,d1.l,d2.l);
      measSvg.appendChild(g);}}}
(function(){const el=document.getElementById('chart');if(!el)return;
  el.addEventListener('mousedown',e=>{
    if(!e.shiftKey||e.button!==0||!window.csMain)return;
    const b=el.getBoundingClientRect();
    measDrawing={x1:e.clientX-b.left,y1:e.clientY-b.top,x2:null,y2:null};
    e.preventDefault();e.stopPropagation();},true);
  el.addEventListener('mousemove',e=>{
    if(!measDrawing)return;const b=el.getBoundingClientRect();
    measDrawing.x2=e.clientX-b.left;measDrawing.y2=e.clientY-b.top;measRedraw();},true);
  el.addEventListener('mouseup',e=>{
    if(!measDrawing)return;
    if(measDrawing.x2!=null){const d1=measP2M(measDrawing.x1,measDrawing.y1),
      d2=measP2M(measDrawing.x2,measDrawing.y2);
      if(d1&&d2)(MEAS[measKey]=MEAS[measKey]||[]).push({l1:d1.l,p1:d1.p,l2:d2.l,p2:d2.p});
      e.preventDefault();e.stopPropagation();}
    measDrawing=null;measRedraw();},true);
  el.addEventListener('contextmenu',e=>{
    const b=el.getBoundingClientRect(),px=e.clientX-b.left,py=e.clientY-b.top;
    const arr=MEAS[measKey]||[];let best=-1,bd=12;
    arr.forEach((m,k)=>{const a=measM2P(m.l1,m.p1),c=measM2P(m.l2,m.p2);if(!a||!c)return;
      const dx=c.x-a.x,dy=c.y-a.y,len2=dx*dx+dy*dy||1;
      const t=Math.max(0,Math.min(1,((px-a.x)*dx+(py-a.y)*dy)/len2));
      const d=Math.hypot(px-(a.x+t*dx),py-(a.y+t*dy));
      if(d<bd){bd=d;best=k;}});
    if(best>=0){arr.splice(best,1);measRedraw();e.preventDefault();e.stopPropagation();}});})();

// ── Ticker card: sector/themes + performance + structure + live news for the selected symbol ──
let curSym=null;
function pctc(v){return v==null?'—':'<span class="'+(v>0?'up':(v<0?'dn':''))+'">'+(v>0?'+':'')+v+'%</span>';}
async function loadTicker(sym){const el=document.getElementById('tkrcard');if(!el||!sym)return;
  el.style.display='block';el.innerHTML='<div class="hd">loading '+sym+'…</div>';
  try{const j=await(await fetch('/api/ticker?symbol='+encodeURIComponent(sym))).json();
    if(j.error){el.innerHTML='<div class="hd">'+j.error+'</div>';return;}
    const p=j.perf||{},a=j.aware||{},nw=j.news||{};
    groupCard=null;                                       // the card now shows a SYMBOL, not a group
    const esc=s=>String(s).replace(/'/g,"\\'");
    const tile=(l,v)=>'<span class="stattile"><span class="sl">'+l+'</span><span class="sv">'+v+'</span></span>';
    let h='<div class="trow"><b style="font-size:16px">'+j.symbol+'</b>'
      +'<span>'+extBadges(j.ext_tfs,j.kind,'1D',j.rsix_compact)+'</span>'
      +(j.sector?'<span class="chip" style="cursor:pointer" onclick="loadGroupCard(\'industry=\'+encodeURIComponent(\''+esc(j.sector)+'\'))">'+j.sector+'</span>':'')
      +(j.themes||[]).map(t=>'<span class="chip" style="cursor:pointer;border-color:var(--acc);color:var(--acc)" onclick="mapPick(\'theme\',\''+esc(t)+'\')">'+t+'</span>').join('')
      +expBtn('tkrcard',j.symbol)+'</div>';
    h+='<div class="trow tsec"><span class="tname">perf</span><span>'
      +tile('d1',pctc(p.d1))+tile('1w',pctc(p.w1))+tile('1m',pctc(p.m1))
      +tile('3m',pctc(p.m3))+tile('6m',pctc(p.m6))
      +(p.off_ath_pct!=null?tile('off-ATH',p.off_ath_pct+'%'):'')+'</span></div>';
    // group perf rows reuse the SAME tiles so d1/1w/1m columns align vertically with the ticker's
    const gp=(g,nm)=>tile('d1',pctc(g.d1))+tile('1w',pctc(g.w1))+tile('1m',pctc(g.m1))
      +'<span class="stattile"><span class="sl">&nbsp;</span><span class="sv hd" style="font-size:12px">'
      +nm+' ('+g.n+' names)</span></span>';
    if(j.sector_perf)h+='<div class="trow"><span class="tname">sector</span><span>'+gp(j.sector_perf,j.sector_perf.name)+'</span></div>';
    (j.theme_perf||[]).forEach(t=>{h+='<div class="trow"><span class="tname">theme</span><span>'+gp(t,t.name)+'</span></div>';});
    h+='<div class="trow"><span class="tname">tape</span><span>'
      +tile('close',p.close??'—')+tile('rvol',p.rvol_today??'—')+tile('ADR%',p.adr_pct??'—')
      +tile('$M/day',p.dollar_vol_m??'—')+tile('RSI',p.rsi??'—')
      +(j.intraday?tile('today',pctc(j.intraday.day_pct))+tile('VWAP',j.intraday.vwap_side):'')
      +'</span></div>';
    const GLYPHS='structure glyphs:  ⤴ fresh breakout · ▲ uptrend / impulse up · ▲⌄ pullback in uptrend'
      +' · ⤵ fresh breakdown · ▼ downtrend / impulse down · ▼⌃ bounce in downtrend · ▽ distribution'
      +' · ▭ range · ◇ contraction · ◆ expansion · (·) unclear';
    const RG={breakout_fresh:['⤴','var(--up)'],impulse_up:['▲','var(--up)'],trend_up:['▲','var(--up)'],
      pullback_in_uptrend:['▲⌄','var(--up)'],breakdown_fresh:['⤵','var(--dn)'],impulse_down:['▼','var(--dn)'],
      trend_down:['▼','var(--dn)'],bounce_in_downtrend:['▼⌃','var(--dn)'],distribution:['▽','var(--dn)'],
      range:['▭','var(--dim)'],contraction:['◇','#e0a43a'],unclear:['·','var(--dim)']};
    h+='<div class="trow tsec"><span class="tname">ratio</span>'
      +(j.ratio_dens||[]).map(r=>{const g=RG[r.regime]||['·','var(--dim)'];
        return '<span class="chip" style="cursor:pointer" title="'+r.label
          +(r.phrase?' — '+String(r.phrase).replace(/"/g,'&quot;'):'')
          +'" onclick="chartRatio(\''+j.symbol+'\',\''+r.sym+'\')">'+j.symbol+'/'+r.sym
          +' <b style="color:'+g[1]+';font-size:13px">'+g[0]+'</b>'
          +extBadge(r.ext,'etf',j.symbol+'/'+r.sym+' ratio (1D)')+'</span>';}).join('')
      +'<span class="chip" style="cursor:help" title="'+GLYPHS+'">?</span></div>';
    if(a.phrase){const sg=RG[a.regime]||['·','var(--dim)'];   // glyph replaces the duplicated regime word
      const ph=String(a.phrase).replace(/;\s*invalid\s*<\s*/,' · invalidates below ')
        .replace(/;\s*invalid\s*>\s*/,' · invalidates above ')
        .replace(/,\s*extended/,' · extended').replace(/,\s*late/,' · late in the move');
      h+='<div class="trow tsec"><span class="tname">structure</span><span>'
        +'<b style="color:'+sg[1]+'">'+sg[0]+'</b> '+ph+'</span></div>';}
    // RSI historical extremes: only TFs currently in a zone / recently at one (10 bars)
    const rxTf=Object.keys(j.rsix||{}).filter(tf=>{const s=j.rsix[tf];
      return s&&(s.zone||(s.bars_since!=null&&s.bars_since<=10));});
    if(rxTf.length){const ZN={atl:'AT all-time LOW',ath:'AT all-time HIGH',near_atl:'approaching all-time low',
        near_ath:'approaching all-time high'};
      h+='<div class="trow"><span class="tname">rsi-x</span><span>'+rxTf.map(tf=>{const s=j.rsix[tf];
        const lo=/atl|rec_lo/.test(s.zone||''), at=(s.zone==='atl'||s.zone==='ath');
        const col=s.zone?((lo?'#39c4d8':'#e05656')+(at?'':'99')):'var(--dim)';   // near = faded
        return '<span class="chip"><b>'+tf+'</b> <span style="color:'+col+'">'
          +(s.zone?ZN[s.zone]:('near extreme '+s.bars_since+' bars ago'))+'</span>'
          +' rsi '+s.rsi+' <span class="hd">[ATL '+s.all_lo+' · ATH '+s.all_hi+']</span></span>';}).join('')
      +'</span></div>';}
    // S/R ladder: nearest-first resistance chips · close pill · nearest-first support chips
    const rs=(a.resistance||[]).map(x=>'<span class="lvlchip r">R '+x.price+(x.strength?' ×'+x.strength:'')+'</span>').join('');
    const ss=(a.support||[]).map(x=>'<span class="lvlchip s">S '+x.price+(x.strength?' ×'+x.strength:'')+'</span>').join('');
    if(rs||ss)h+='<div class="trow"><span class="tname">levels</span><span>'+rs
      +(p.close!=null?'<span class="lvlchip px">'+p.close+'</span>':'')+ss+'</span></div>';
    let nh='';
    if(nw.earnings)nh+='<span class="chip" style="border-color:#e0a43a;color:#e0a43a">📊 earnings '+nw.earnings+'</span>';
    (nw.edgar||[]).forEach(f=>nh+='<span class="chip">📄 8-K '+f.date+(f.labels&&f.labels.length?': '+f.labels.join(', '):'')+'</span>');
    if(nw.catalyst&&nw.catalyst!=='unknown')nh+='<span class="chip" style="border-color:var(--acc);color:var(--acc)">⚡ '+nw.catalyst+'</span>';
    nh+=(nw.items||[]).map(x=>'<div class="newsit"><span class="hd">'+(x.published||'')+' · '+(x.source||'')+'</span><br>'+x.title+'</div>').join('');
    h+='<div class="trow tsec"><span class="tname">news</span><div style="flex:1">'+(nh||'<span class="hd">nothing recent</span>')+'</div></div>';
    el.innerHTML=h;}
  catch(e){el.innerHTML='<div class="hd">ticker card failed: '+e+'</div>';}}
document.getElementById('btntkr').onclick=()=>{const el=document.getElementById('tkrcard'),b=document.getElementById('btntkr');
  // hidden covers BOTH the normal display:none case and "stranded in a hidden ancestor"
  // (e.g. the card was ⛶-expanded and the overlay closed unexpectedly) — offsetParent is
  // null whenever any ancestor hides the element, so the toggle can't get stuck
  const hidden=el.style.display==='none'||!el.offsetParent;
  if(hidden){if(bigPh)closeBig();                 // pull it back out of the overlay if stranded
    b.classList.add('on');el.style.display='block';
    if(curSym)loadTicker(curSym);
    else el.innerHTML='<div class="hd">select a row (or click a map/history ticker) to fill this card</div>';}
  else{el.style.display='none';b.classList.remove('on');}};
// ── generic panel expansion: MOVE the live element into a full-screen pane (zoom 1.45 makes
//    every font/layout proportionally larger); closing moves it back where it lived ──
let bigFrom=null,bigPh=null;
function expandEl(el,title,z){if(!el)return;const m=document.getElementById('bigpanel');
  if(bigPh)closeBig();                                  // one panel expanded at a time
  bigFrom=el.parentNode;bigPh=document.createElement('div');bigPh.style.display='none';
  bigFrom.insertBefore(bigPh,el);
  m.querySelector('.btitle').innerHTML=title||'';
  const bb=m.querySelector('.bbody');bb.style.zoom=(z==null?1.45:z);   // charts expand at native scale
  bb.appendChild(el);
  m.style.display='flex';}
function closeBig(){const m=document.getElementById('bigpanel');
  const el=m.querySelector('.bbody').firstElementChild;
  if(el&&bigPh&&bigFrom){bigFrom.insertBefore(el,bigPh);bigPh.remove();}
  bigPh=null;bigFrom=null;m.style.display='none';}
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&bigPh)closeBig();});
function expBtn(id,title){return '<span class="chip" style="cursor:pointer;margin-left:auto" title="expand to full screen" '
  +'onclick="expandEl(document.getElementById(\''+id+'\'),\''+title+'\')">⛶</span>';}

// ── map click router: anything on the Map charts itself AND fills the ticker card ──
function mapPick(kind,key){const el=document.getElementById('tkrcard'),b=document.getElementById('btntkr');
  if(el){el.style.display='block';if(b)b.classList.add('on');}
  if(kind==='theme')loadGroupCard('theme='+encodeURIComponent(key));
  else if(kind==='sector')loadGroupCard('sector_etf='+encodeURIComponent(key));
  else{curSym=key;renderMtf(key);chartTF(key,'1D');loadTicker(key);}}   // goggles too (mapPick is THE opener)
let groupCard=null, groupSort='w1';   // last theme/sector card payload + member sort key
function drawGroupCard(){const el=document.getElementById('tkrcard'),j=groupCard;if(!el||!j)return;
  const p=j.perf||{};
  const mem=(j.members||[]).slice().sort((a,b)=>((b[groupSort]??-999)-(a[groupSort]??-999)));
  const sbtn=k=>'<span class="chip" style="cursor:pointer'+(groupSort===k?';border-color:var(--acc);color:var(--acc)':'')
    +'" onclick="groupSort=\''+k+'\';drawGroupCard()">'+k+'</span>';
  let h='<div class="trow"><b style="font-size:14px">'+j.theme+'</b>'
    +(j.etf?'<span class="chip" style="cursor:pointer;border-color:var(--acc);color:var(--acc)" onclick="mapPick(\'sym\',\''+j.etf+'\')">'+j.etf+extSlot(j.etf,'1D')+'</span>':'')
    +'<span class="hd">'+mem.length+(j.n_total?(' of '+j.n_total):'')+' members</span>'
    +'<span class="hd" style="margin-left:auto">sort:</span>'+sbtn('d1')+sbtn('w1')+sbtn('m1')+sbtn('rvol')
    +expBtn('tkrcard',j.theme)+'</div>';
  h+='<div class="trow"><span class="tname">basket</span>d1 '+pctc(p.d1)+' · 1w '+pctc(p.w1)+' · 1m '+pctc(p.m1)+'</div>';
  h+=mem.map(m=>'<div class="trow" style="cursor:pointer" onclick="mapPick(\'sym\',\''+m.symbol+'\')">'
    +'<b style="min-width:52px">'+m.symbol+extBadges(m.ext_tfs,m.kind,'1D',m.rsix)+'</b>'
    +'d1 '+pctc(m.d1)+' · 1w '+pctc(m.w1)+' · 1m '+pctc(m.m1)
    +' <span class="hd">'+m.close+' · rvol '+(m.rvol??'—')+' · $'+(m.dollar_vol_m??'—')+'M</span></div>').join('');
  el.innerHTML=h;
  if(j.etf)extFetch([j.etf]).then(()=>extPatch(el));}   // proxy-ETF badge in the card header
async function loadGroupCard(qs){const el=document.getElementById('tkrcard');if(!el)return;
  el.style.display='block';el.innerHTML='<div class="hd">loading…</div>';
  try{const j=await(await fetch('/api/ticker?'+qs)).json();
    if(j.error){el.innerHTML='<div class="hd">'+j.error+'</div>';return;}
    if(j.etf){curSym=j.etf;chartTF(j.etf,'1D');}       // chart the proxy/sector ETF alongside
    groupCard=j;drawGroupCard();}
  catch(e){el.innerHTML='<div class="hd">group card failed: '+e+'</div>';}}

// S/R toggle: flip the flag, restyle the button, redraw the current chart's S/R lines in place
document.getElementById('btnsr').onclick=function(){window.SR_ON=!window.SR_ON;
  this.classList.toggle('on',window.SR_ON);if(window._srDraw)window._srDraw();};

// ── ◎ Fires: mark PAST setup triggers (from the registries) on the charted symbol ──────────
// Selection persists in localStorage; marks re-apply on every chart draw (daily TFs only —
// registries are daily, so intraday/weekly axes simply have no matching dates).
const FIRESHORT={episodic_pivot:'EP',qm_breakout:'QMB',high_tight_flag:'HTF',flat_base:'FB',
  gapper:'GAP',hvc:'HVC',delayed_hvc:'dHVC',stairstep:'STAIR',backburner:'BB',cup_handle:'C&H',
  double_top:'DT',head_shoulders:'H&S',inverse_hs:'iH&S',rsi_extreme_revert:'RSIrev',
  rsi_extreme_fade:'RSIfade'};
const FIRECOL=['#c39bd3','#7fb3d5','#76d7c4','#f7dc6f','#f0b27a','#d98880','#85c1e9','#82e0aa',
  '#f8c471','#e59866','#cd6155','#a3e4d7','#f5b7b1','#d2b4de','#a9cce3','#aab7b8'];
let fireSel=new Set(JSON.parse(localStorage.getItem('fireMarksSel')||'[]'));
const fireCache={};
function fireData(sym){if(!sym)return Promise.resolve(null);
  if(!fireCache[sym])fireCache[sym]=fetch('/api/fires?sym='+encodeURIComponent(sym))
    .then(r=>r.json()).catch(()=>null);
  return fireCache[sym];}
async function applyFireMarks(){const ctx=window._fireCtx;if(!ctx||!ctx.sym||ctx.intraday)return;
  let add=[];
  if(fireSel.size){const j=await fireData(ctx.sym);
    if(window._fireCtx!==ctx)return;                     // chart was redrawn while fetching
    if(j&&j.setups)j.setups.forEach((s,i)=>{if(!fireSel.has(s.setup))return;
      const col=FIRECOL[i%FIRECOL.length];
      for(const f of s.fires){if(!ctx.tset.has(f.date))continue;
        const st=f.state||'',suf=st.includes(':')?st.split(':')[1]      // breakout:ep9m -> ep9m
          :(!/^break/.test(st)?st:'');                                  // watch/entry1/... kept
        add.push({time:f.date,position:s.short?'aboveBar':'belowBar',
          shape:s.short?'arrowDown':'arrowUp',color:col,
          text:(FIRESHORT[s.setup]||s.setup)+(suf?' '+suf:'')});}});}
  const all=ctx.base.concat(add);
  all.sort((a,b)=>a.time<b.time?-1:(a.time>b.time?1:0));
  try{ctx.cs.setMarkers(all);}catch(e){}}
async function renderFireSel(){const el=document.getElementById('firesel');
  const sym=(window._fireCtx||{}).sym;
  if(!sym){el.innerHTML='<div class="hd">chart a symbol first</div>';return;}
  el.innerHTML='<div class="hd">loading registries…</div>';
  const j=await fireData(sym);
  if(!j||!j.setups){el.innerHTML='<div class="hd">no registries built yet</div>';return;}
  let h='<div style="margin-bottom:5px"><b style="color:var(--acc)">past fires · '+sym+'</b>'
    +'<span class="fseall" onclick="fireSelAll(1)">all</span>'
    +'<span class="fseall" onclick="fireSelAll(0)">none</span></div>';
  j.setups.forEach((s,i)=>{h+='<label class="fsrow'+(s.n?'':' none')+'">'
    +'<input type="checkbox" '+(fireSel.has(s.setup)?'checked':'')
    +' onchange="fireSelTog(\''+s.setup+'\',this.checked)">'
    +'<span class="fsw" style="background:'+FIRECOL[i%FIRECOL.length]+'"></span>'
    +s.label+(s.validated?'':' <span class="hd">unvalidated</span>')
    +'<span class="hd" style="margin-left:auto;padding-left:10px">'+s.n+'</span></label>';});
  el.innerHTML=h;}
function fireSave(){localStorage.setItem('fireMarksSel',JSON.stringify([...fireSel]));
  document.getElementById('btnfires').classList.toggle('on',fireSel.size>0);
  renderFireSel();applyFireMarks();}
function fireSelTog(name,on){if(on)fireSel.add(name);else fireSel.delete(name);fireSave();}
function fireSelAll(on){if(!on){fireSel=new Set();fireSave();return;}
  fireData((window._fireCtx||{}).sym).then(j=>{if(j&&j.setups)fireSel=new Set(j.setups.map(s=>s.setup));fireSave();});}
document.getElementById('btnfires').onclick=function(){const el=document.getElementById('firesel');
  const show=el.style.display!=='block';el.style.display=show?'block':'none';
  if(show)renderFireSel();};
document.getElementById('btnfires').classList.toggle('on',fireSel.size>0);
// download the current setup's filtered+sorted rows as a TradingView watchlist (EXCHANGE:SYMBOL)
function exportTV(){const r=rows();
  if(!r.length){alert('No rows to export — loosen the filters.');return;}
  const seen=new Set(),syms=[];
  for(const x of r){const s=(x.exchange?x.exchange+':':'')+String(x.symbol).replace(/-/g,'.');
    if(!seen.has(s)){seen.add(s);syms.push(s);}}
  const blob=new Blob([syms.join(',')],{type:'text/plain'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);
  a.download='watchlist_'+cur+'_filtered.txt';document.body.appendChild(a);a.click();
  a.remove();URL.revokeObjectURL(a.href);}
document.getElementById('btntv').onclick=exportTV;

// ── Market Map drawer (awareness v2: sequences, gaps, rotation, calls) ────────
let mapData=null, mapOpen=false, mapRotView='quad', mapSeq=0;
function loadMap(){if(!ONDEMAND)return;
  const s=++mapSeq;                                   // scrubbing fires overlapping fetches — last REQUEST wins, not last response
  const req=curAsof()+tq();                           // request key (date + replay-time) — flag "calculating" only on a NEW view
  if(req!==mapAsofShown){radarData={loading:1};rvolData={loading:1};if(mapOpen)drawMap();}   // first-load / scrub -> show calculating (not on a same-as-of live poll)
  fetch('/api/awareness?date='+encodeURIComponent(curAsof())+tq()).then(r=>r.json())
    .then(j=>{if(s!==mapSeq)return;mapData=j;if(!replayT)checkAlerts(j.alerts);drawMarket();if(mapOpen)drawMap();}).catch(()=>{});
  fetch('/api/ep_radar?date='+encodeURIComponent(curAsof())+tq()).then(r=>r.json())   // live now, or replay as-of (date,t)
    .then(j=>{if(s!==mapSeq)return;radarData=j;mapAsofShown=req;if(mapOpen)drawMap();}).catch(()=>{if(s===mapSeq){radarData={error:1};if(mapOpen)drawMap();}});
  fetch('/api/rvol_leaders?date='+encodeURIComponent(curAsof())+tq()).then(r=>r.json())  // intraday RVOL leaders (live or replay)
    .then(j=>{if(s!==mapSeq)return;rvolData=j;mapAsofShown=req;if(mapOpen)drawMap();}).catch(()=>{if(s===mapSeq){rvolData={error:1};if(mapOpen)drawMap();}});
  fetch('/api/notifications').then(r=>r.json())    // 🔔 grade-A notified events (live session ledger)
    .then(j=>{if(s!==mapSeq)return;notifData=j.rows||[];if(mapOpen)drawMap();}).catch(()=>{});}
let radarData=null,rvolData=null,mapAsofShown='',notifData=[];
// 🔔 persistent, clickable record of the macOS pings (banners vanish; click-through only opens
// Script Editor). Click a row -> its 5m chart on the alert day (Amir 2026-07-08).
function notifHtml(){
  if(!notifData||!notifData.length)
    return '<h3>🔔 Alerts fired <span style="font:400 11px monospace;color:var(--dim)">grade-A · click for chart</span></h3>'
      +'<div style="color:var(--dim);font:12px monospace;margin:2px 0">none yet this session — grade-A radar/trigger/VWAP events land here as they fire</div>';
  return '<h3>🔔 Alerts fired <span style="font:400 11px monospace;color:var(--dim)">grade-A · click for chart</span></h3>'
    +notifData.slice(0,10).map(n=>
      '<div style="margin:2px 0;cursor:pointer" onclick="chartTF(\''+n.sym+'\',\'5m\')" title="'+(n.why||'').replace(/"/g,'&quot;')+'">'
      +'<span style="color:var(--dim)">'+n.t+'</span> '
      +'<b class="'+(n.side==='bear'?'dn':'up')+'">'+n.sym+'</b> '
      +'<span style="border:1px solid var(--line);border-radius:4px;padding:0 4px;font:11px monospace">'+n.grade+(n.action?' · '+n.action:'')+'</span> '
      +'<span style="color:var(--dim);font:12px monospace">'+n.text.replace(n.sym,'').trim()+'</span></div>').join('');}
// ---- Map alert badge + sound ping: flag new transition/RVOL alerts even when the drawer is closed ----
let alertSeen='',alertUnseen=0;
function checkAlerts(al){al=al||[];const newest=al.length?(al[0].t+'|'+al[0].text):'';
  if(!newest)return;
  if(alertSeen===''){alertSeen=newest;return;}          // seed on first load — don't ping the backlog
  if(newest!==alertSeen){
    let n=0;for(const a of al){if((a.t+'|'+a.text)===alertSeen)break;n++;}   // count alerts newer than last seen
    if(!mapOpen){alertUnseen+=n;renderMapBadge();pingAlert();}
    alertSeen=newest;
  }}
function renderMapBadge(){const b=document.getElementById('mapbadge');if(!b)return;
  if(alertUnseen>0&&!mapOpen){b.textContent=alertUnseen;b.style.display='inline-block';}else{b.style.display='none';}}
function pingAlert(){try{const c=new(window.AudioContext||window.webkitAudioContext)();
  const o=c.createOscillator(),g=c.createGain();o.type='sine';o.frequency.value=880;
  o.connect(g);g.connect(c.destination);g.gain.setValueAtTime(0.0001,c.currentTime);
  g.gain.exponentialRampToValueAtTime(0.12,c.currentTime+0.01);
  g.gain.exponentialRampToValueAtTime(0.0001,c.currentTime+0.25);
  o.start();o.stop(c.currentTime+0.27);}catch(e){}}
function sigChip(r){const s=r&&r.sig;if(!s)return '';   // significance grade + action (plan-ep-significance)
  const col=s.grade==='A'?'#2fbf8f':(s.grade==='B'?'#e8b84b':'var(--dim)');
  const act={ACT:'#2fbf8f','STALK':'#e8b84b','FADE':'#e05a6d','CHASE-RISK':'#e05a6d'}[s.action]||'var(--dim)';
  const tip=('significance '+s.score+'/100 — '+(s.why||'')
    +(s.news_na?' · REPLAY: news/catalyst unavailable — grade is the news-free CORE read, not comparable to live grades':'')
    +' · heuristic — replay-validated on 30 sessions (multi-day right-tail edge, small n); live forward monitor running').replace(/"/g,"'");
  const dim=s.news_na?';opacity:.55':'';
  return ' <span title="'+tip+'" style="font-weight:700;color:'+col+dim+'">'+s.grade+(s.news_na?'*':'')+'</span>'
    +' <span title="'+tip+'" style="font-size:10.5px;font-weight:600;color:'+act+dim+'">'+s.action+'</span>';}
function radarHtml(){if(!radarData)return '';
  if(radarData.loading)return '<h3>EP Radar</h3><div style="color:var(--dim)">⏳ calculating… <span style="opacity:.6">(first load of a day can take ~10s)</span></div>';
  if(radarData.error)return '<h3>EP Radar</h3><div style="color:var(--dim)">⚠ couldn\'t load — retrying next refresh</div>';
  const rs=radarData.rows||[];
  const sub=radarData.phase==='replay'?('replay '+radarData.date+' '+(radarData.t||'')+' — EPs formed by then (5m, point-in-time)'):
    radarData.phase==='pre'?'pre-market — overnight gap × pm $volume (IEX, calibrated)':
    (radarData.phase==='rth'?'live — day volume vs time-of-day pace (calibrated)':'market closed');
  let s='<h3>EP Radar <span style="color:var(--dim);text-transform:none;letter-spacing:0">· '+sub+'</span></h3>';
  if(!rs.length)return s+'<div style="color:var(--dim)">'+(radarData.note||'nothing on the radar')+'</div>';
  s+=rs.slice(0,12).map(r=>{
    const pre=r.pm_gap_pct!=null;
    const v=pre?r.pm_gap_pct:r.move_pct;
    const main=pre?((v>0?'+':'')+v+'% overnight · $'+(r.pm_dollar_m??'—')+'M pm vol'
        +(r.pm_range_pct!=null?' · pm range '+r.pm_range_pct+'%':''))
      :((v>0?'+':'')+v+'% · gap '+(r.gap_pct>0?'+':'')+r.gap_pct+'% · pace ×'+r.vol_pace+' · '+r.shares_m+'M sh'
        +(r.live_dvol_m!=null?' · $'+r.live_dvol_m+'M today':''));
    const sIco={good:'🟢',bad:'🔴',mixed:'🟠'}[r.sent]||'';   // news sentiment at a glance
    const wc=r.news_react==='good_news_red'?'#e05a6d':(r.news_react==='bad_news_green'?'#2fbf8f':'#e8b84b');
    const warn=r.news_warn?'<div style="margin:1px 0 2px 16px;font-weight:600;color:'+wc+'">'+r.news_warn+'</div>':'';
    const peer=r.peer_headline?'<div style="margin-left:16px;font-size:11px;color:var(--dim)">peer/theme · '+r.peer_headline+'</div>':'';
    return '<div style="margin:3px 0"><b style="cursor:pointer" onclick="chartTF(\''+r.symbol+'\',\'1D\')">'+r.symbol+'</b>'
      +sigChip(r)+' <span class="'+(v>=0?'up':'dn')+'">'+main+'</span>'
      +(r.neglect_6m!=null?' <span style="color:var(--dim)">6m '+(r.neglect_6m>0?'+':'')+r.neglect_6m+'%</span>':'')
      +(r.setups?' <span style="color:var(--acc)">'+r.setups+'</span>':'')
      +(r.catalyst&&r.catalyst!=='unknown'&&r.catalyst!=='sympathy'?' <span style="color:var(--acc)">'+sIco+' '+r.catalyst+'</span>':(sIco?' '+sIco:''))
      +(r.headline?' <span style="color:var(--dim)">— '+r.headline+'</span>':'')
      +warn+peer+'</div>';
  }).join('');
  return s;}
function rvolHtml(){if(!rvolData)return '';
  if(rvolData.loading)return '<h3>RVOL Leaders</h3><div style="color:var(--dim)">⏳ calculating… <span style="opacity:.6">(first load of a day can take ~10s)</span></div>';
  if(rvolData.error)return '<h3>RVOL Leaders</h3><div style="color:var(--dim)">⚠ couldn\'t load — retrying next refresh</div>';
  const rs=rvolData.rows||[];
  const sub=rvolData.phase==='replay'?('replay '+rvolData.date+' '+(rvolData.t||'')+' — cumulative-pace RVOL as of then (5m) · slot = that-bar spike'):
    rvolData.phase==='rth'?'live — cumulative-pace RVOL (calibrated) · slot = current-bar spike':
    (rvolData.phase==='pre'?'pre-market — leaders populate at the open':'market closed');
  let s='<h3>RVOL Leaders <span style="color:var(--dim);text-transform:none;letter-spacing:0">· '+sub+'</span></h3>';
  if(!rs.length)return s+'<div style="color:var(--dim)">'+(rvolData.note||'nothing running hot')+'</div>';
  s+=rs.slice(0,15).map(r=>{
    const dir=r.above_vwap?'up':'dn';
    return '<div style="margin:2px 0"><b style="cursor:pointer" onclick="chartTF(\''+r.symbol+'\',\'1D\')">'+r.symbol+'</b>'
      +sigChip(r)+' <span class="'+dir+'">cum ×'+r.cum_rvol+'</span>'
      +(r.warming?' <span style="color:var(--dim)" title="first 5-minute bar of the session — pace estimate still settling; treat the ranking as early">⏳ warming</span>':'')
      +(r.slot_rvol==null?' <span style="color:var(--dim)" title="latest 5m bar not filled yet (live data warming) — current-slot spike n/a">· slot —</span>'
        :(r.slot_rvol>=2?' <span class="up">· slot ×'+r.slot_rvol+'</span>':' <span style="color:var(--dim)">· slot ×'+r.slot_rvol+'</span>'))
      +' <span style="color:var(--dim)">'+(r.move_pct>0?'+':'')+r.move_pct+'% · '+(r.above_vwap?'>VWAP':'<vwap')
        +(r.live_dvol_m!=null?' · today $'+r.live_dvol_m+'M':'')+' · $'+(r.dollar_vol_m??'—')+'M/day</span>'
      +(r.setups?' <span style="color:var(--acc)">'+r.setups+'</span>':'')+'</div>';
  }).join('');
  return s;}
function openMap(){mapOpen=true;alertUnseen=0;renderMapBadge();document.getElementById('mapdrawer').classList.add('open');
  if(!mapData)loadMap();else drawMap();}
function closeMap(){mapOpen=false;document.getElementById('mapdrawer').classList.remove('open');}

// ── risk-on/off group buckets — used by the Perf drawer's rotation read (boardRead) ──
const TB_DEFENSIVE=['Utilities','REIT','Gold','Silver','Insurance','Drug Manufacturers','Grocery',
  'Discount Stores','Household','Packaged Foods','Tobacco','Healthcare Plans','Medical Care','Pharma','Waste'];
const TB_OFFENSIVE=['Semiconductor','Software','Internet','Bitcoin','Crypto','Artificial Intelligence',
  'Quantum','Solar','Growth Stocks','Cloud','Disruptive','Robotics'];
const tbIn=(n,list)=>list.some(x=>n.includes(x));
// (the standalone 🏁 Theme-leaders bigpanel was folded into the Perf drawer — see loadBoard/drawBoard)
function sideBadge(s){return '<span class="badge '+s+'">'+s.toUpperCase()+'</span>';}
function statChip(c){if(!c.stats)return '<span class="cstats">unvalidated</span>';
  if(c.stats.unvalidated)return '<span class="cstats">unvalidated'+(c.stats.n?' · n='+c.stats.n:'')+'</span>';
  return '<span class="cstats" title="hit rate vs the unconditional base rate over the validation window">n='
    +c.stats.n+' · '+Math.round(c.stats.hit*100)+'% vs '+Math.round((c.stats.base||0)*100)+'% base · '
    +(c.stats.avg>0?'+':'')+(c.stats.avg||0).toFixed(2)+'%</span>';}
const CALL_TAG={group_leader_go:'group leader',group_fade_warn:'fading group',gap_go:'gap-and-go',
  gap_into_extension:'gap into extension',gap_fade_reversal:'gap fade',gap_down_into_support:'gap into support',
  breakout_follow:'breakout follow-through',breakdown_follow:'breakdown follow-through',coil_at_highs:'coil at highs',
  ratio_break:'ratio break',risk_off_guard:'risk-off guard'};
function callTag(r){return CALL_TAG[r]||String(r||'').replace(/_/g,' ');}   // readable category, not the code id
function callHtml(c){const lv=c.levels||{};
  const lvls=Object.keys(lv).map(k=>k.replace('_',' ')+' '+lv[k]).join(' · ');
  const ldrs=(c.leaders||[]).filter(Boolean);
  let ph=c.phrase;
  if(ldrs.length)ph=ph.replace(' → '+ldrs.join(', '),'');   // leaders become clickable chips below, not phrase text
  const chips=ldrs.length?'<div style="margin:3px 0 1px"><span style="color:var(--dim);font-size:10px">leaders:</span> '
    +ldrs.map(t=>'<span onclick="mapPick(\''+'sym\',\''+t+'\')" style="cursor:pointer;color:var(--acc);border:1px solid var(--line);border-radius:8px;padding:0 7px;margin:0 4px 3px 0;display:inline-block;font-size:11px">'+t+'</span>').join('')+'</div>':'';
  return '<div class="call '+c.side+'">'+statChip(c)+sideBadge(c.side)
    +' <b>'+c.target+'</b> <span style="color:var(--dim);font-size:10px;border:1px solid var(--line);border-radius:8px;padding:0 6px;margin-left:5px">'+callTag(c.rule)+'</span>'
    +'<div class="cphrase">'+ph+'</div>'+chips
    +(c.expect?'<div class="cexpect">→ '+c.expect+'</div>':'')
    +(lvls?'<div class="cexpect">'+lvls+'</div>':'')+'</div>';}
function gapChip(g){if(!g||g.atr==null)return '<span class="gapchip">no gap data</span>';
  const a=(g.atr>0?'▲':'▼')+Math.abs(g.atr).toFixed(1)+'ATR';
  return '<span class="gapchip '+g.status+'">'+(g.pct>0?'+':'')+g.pct.toFixed(1)+'% '+a+' '+g.status+'</span>';}
function ladder(v){const sr=v.daily.sr,out=[];
  if(sr.resistance[0]){const r=sr.resistance[0];out.push('<span class="dn">R '+r.price+' ×'+r.strength+' · '+r.dist_atr+'▲</span>');}
  if(sr.support[0]){const s=sr.support[0];out.push('<span class="up">S '+s.price+' ×'+s.strength+' · '+s.dist_atr+'▼</span>');}
  return out.join('<br>');}
function mapCard(v){const b=v.bias||{},g=(v.intra||{}).gap,h1=(v.intra||{}).h1;
  const click=(v.kind==='theme')
    ?' onclick="mapPick(\'theme\',\''+String(v.key).replace(/'/g,"\\'")+'\')"'
    :(v.kind==='sector'?' onclick="mapPick(\'sector\',\''+v.key+'\')"'
    :' onclick="mapPick(\'sym\',\''+v.key+'\')"');   // everything on the map charts + fills the ticker card
  const xsym=(v.kind==='theme')?THEME_ETF[v.key]:v.key;   // ext badge rides the tradable proxy
  const bc=b.side==='bull'?'var(--up)':(b.side==='bear'?'var(--dn)':'var(--line)');  // bias = left border
  return '<div class="mapcard"'+click+' style="border-left:3px solid '+bc+'" title="'+(v.daily.seq.phrase||'')+'">'
    +'<div class="t"><span class="sym">'+v.key+(xsym?extSlot(xsym,'1D'):'')+'</span>'+sideBadge(b.side||'neutral')+'</div>'
    +(v.intra?('<div class="t">'+gapChip(g)+(h1?'<span style="color:var(--dim);font-size:10px">1h '+h1.seq.regime+'</span>':'')+'</div>'):'')
    +'<div class="phrase">'+v.daily.seq.phrase+'</div>'
    +'<div class="lvl">'+ladder(v)+'</div></div>';}
function rTile(v){const s=v.daily.seq,up=(s.regime.indexOf('up')>=0||s.regime==='breakout_fresh'),
  dn=(s.regime.indexOf('down')>=0||s.regime==='breakdown_fresh'||s.regime==='distribution');
  const arrow=up?'<span class="up">▲</span>':(dn?'<span class="dn">▼</span>':'<span style="color:var(--dim)">▬</span>');
  const at=s.flags.indexOf('at_resistance')>=0?' · at R':(s.flags.indexOf('at_support')>=0?' · at S':'');
  const rbc=up?'var(--up)':(dn?'var(--dn)':'var(--line)');
  const rp=String(v.key).split('/');
  const pick=rp.length===2?'chartRatio(\''+rp[0]+'\',\''+rp[1]+'\')':'mapPick(\'sym\',\''+v.key+'\')';
  return '<span class="rtile" style="cursor:pointer;border-left:3px solid '+rbc+'" onclick="'+pick+'" title="'+s.phrase+' · click to open the ratio chart"><b>'+v.key+'</b> '+arrow+' '+s.regime+at
    +'<span class="rp"> '+(v.label||'')+'</span></span>';}
function extTip(sym){const d=sym&&EXTD[sym];if(!d||!d.tfs||d.tfs['1D']==null)return '';   // SVG titles: ext as text
  const v=d.tfs['1D'];return ' · ext '+(v>0?'+':'')+v.toFixed(1)+'×ATR ('+extWord(extCls(v,d.kind)).split(' —')[0]+')';}
function quadSvg(rows,W,H){W=W||880;H=H||310;const cx=W/2,cy=H/2,px=(W/2-40),py=(H/2-26);
  // signed-SQRT x-scale to ±25% (Amir 2026-07-03): the old ±8% linear clamp PINNED every
  // volatile theme to the edges in a meaningless stack; sqrt keeps center resolution while
  // spreading the tails in true rank order (gridlines at ±2/±8/±20% anchor the distances)
  const X=r=>{const v=Math.max(-25,Math.min(25,r.rs_1m||0));return cx+Math.sign(v)*Math.sqrt(Math.abs(v)/25)*px;};
  const Y=r=>cy-Math.max(-3,Math.min(3,r.rot_z||0))/3*py;
  let grid='';for(const g of [2,8,20]){const off=Math.sqrt(g/25)*px;
    for(const s of [-1,1]){const gx=(cx+s*off).toFixed(1);
      grid+='<line x1="'+gx+'" y1="8" x2="'+gx+'" y2="'+(H-8)+'" stroke="var(--line)" stroke-dasharray="2,5" opacity=".45"/>'
        +'<text x="'+gx+'" y="'+(cy-4)+'" text-anchor="middle" fill="var(--dim)" font-size="8">'+(s>0?'+':'−')+g+'%</text>';}}
  const col=r=>r.acting==='well'?'var(--up)':(r.acting==='poorly'?'var(--dn)':'var(--dim)');
  let dots='';const lps=[];   // placed label boxes — greedy vertical nudge so clusters stay readable
  for(const r of rows){const solid=r.kind==='sector';
    const pick=r.kind==='theme'?'mapPick(\'theme\',\''+String(r.name).replace(/'/g,"\\'")+'\')'
      :(r.kind==='sector'&&r.etf?'mapPick(\'sector\',\''+r.etf+'\')'
      :(r.etf?'mapPick(\'sym\',\''+r.etf+'\')':''));
    dots+='<circle cx="'+X(r).toFixed(1)+'" cy="'+Y(r).toFixed(1)+'" r="'+(solid?7:6)+'" '
      +(solid?('fill="'+col(r)+'" fill-opacity=".75"'):('fill="none" stroke="'+col(r)+'" stroke-width="1.6"'))
      +(pick?(' style="cursor:pointer" onclick="'+pick+'"'):'')
      +'><title>'+r.name+' — rs1m '+(r.rs_1m>0?'+':'')+r.rs_1m+'% · rot z '+(r.rot_z>0?'+':'')+r.rot_z
      +(r.pct_up!=null?' · '+Math.round(r.pct_up)+'% up':'')+' · '+r.quadrant+extTip(r.etf)+'</title></circle>';
    // label sectors always; label a theme only when it's displaced from the center pack —
    // clustered labels smear into noise (hover any dot or label for the full name + stats)
    const tip=r.name+' — rs1m '+(r.rs_1m>0?'+':'')+r.rs_1m+'% · rot z '+(r.rot_z>0?'+':'')+r.rot_z+extTip(r.etf);
    if(solid||Math.abs(r.rs_1m||0)>=1.5||Math.abs(r.rot_z||0)>=0.8){
      const txt=r.etf||r.name.slice(0,10);let lx=X(r)+9,ly=Y(r)+3;const w=txt.length*6.2;
      for(let g=0;g<14;g++){const c=lps.find(q=>Math.abs(q.y-ly)<10.5&&lx<q.x+q.w&&q.x<lx+w);
        if(!c)break;ly=c.y+11;}                        // slide down until the slot is free
      lps.push({x:lx,y:ly,w});
      dots+='<text x="'+lx.toFixed(1)+'" y="'+ly.toFixed(1)+'" fill="#9db0c3" font-size="10"'
        +(pick?(' style="cursor:pointer" onclick="'+pick+'"'):'')
        +'><title>'+tip+'</title>'+txt+'</text>';}}
  return '<svg viewBox="0 0 '+W+' '+H+'" style="width:100%;background:var(--bg);border:1px solid var(--line);border-radius:6px">'
    +grid
    +'<line x1="'+cx+'" y1="8" x2="'+cx+'" y2="'+(H-8)+'" stroke="var(--line)"/>'
    +'<line x1="20" y1="'+cy+'" x2="'+(W-20)+'" y2="'+cy+'" stroke="var(--line)"/>'
    +'<text x="'+(W-24)+'" y="16" text-anchor="end" fill="#9db0c3" font-size="10">LEADING · EXTENDING TODAY</text>'
    +'<text x="24" y="16" fill="#9db0c3" font-size="10">LAGGING · IMPROVING TODAY</text>'
    +'<text x="'+(W-24)+'" y="'+(H-10)+'" text-anchor="end" fill="#9db0c3" font-size="10">LEADING · FADING TODAY</text>'
    +'<text x="24" y="'+(H-10)+'" fill="#9db0c3" font-size="10">LAGGING · BREAKING</text>'
    +dots+'</svg>'
    +'<div style="color:var(--dim);font-size:10px;margin-top:3px">x = 1-month RS vs SPY (sqrt-compressed to ±25%; gridlines ±2/±8/±20%) · y = today\'s rotation z-score · solid = sector ETF, hollow = theme · green/red = members confirming/not · click a sector dot to chart it</div>';}
function rotTiles(rows){let h='<table class="maptbl"><thead><tr><th>group</th><th>kind</th><th>rot z</th><th>rs 1m</th><th>% up</th><th>quadrant</th><th>acting</th><th>structure</th></tr></thead><tbody>';
  for(const r of rows){const zc=r.rot_z>0.5?'up':(r.rot_z<-0.5?'dn':'');
    const pick=r.kind==='theme'?'mapPick(\'theme\',\''+String(r.name).replace(/'/g,"\\'")+'\')'
      :(r.kind==='sector'&&r.etf?'mapPick(\'sector\',\''+r.etf+'\')'
      :(r.etf?'mapPick(\'sym\',\''+r.etf+'\')'
      :'loadGroupCard(\'industry=\'+encodeURIComponent(\''+String(r.name).replace(/'/g,"\\'")+'\'))'));
    h+='<tr><td><b style="cursor:pointer" onclick="'+pick+'">'+r.name+'</b>'+(r.etf?' <span style="color:var(--dim)">'+r.etf+'</span>'+extSlot(r.etf,'1D'):'')+'</td><td>'+r.kind+'</td>'
      +'<td class="'+zc+'">'+(r.rot_z>0?'+':'')+r.rot_z+'</td><td>'+(r.rs_1m>0?'+':'')+r.rs_1m+'%</td>'
      +'<td>'+(r.pct_up==null?'—':Math.round(r.pct_up)+'%')+'</td><td>'+r.quadrant.replace('_',' ')+'</td>'
      +'<td class="'+(r.acting==='well'?'up':(r.acting==='poorly'?'dn':''))+'">'+(r.acting||'—')+(r.narrow?' · narrow':'')+'</td>'
      +'<td>'+(r.seq_regime||'—')+(r.at_level&&r.at_level!=='clear'?' · '+r.at_level:'')+'</td></tr>';}
  return h+'</tbody></table>';}
function drawMap(){const el=document.getElementById('mapbody'),j=mapData;if(!el)return;
  if(!j||j.error){el.innerHTML='<div class="empty">'+((j&&j.error)||'loading…')+'</div>';return;}
  // ext badges for every instrument on the map: batch-fetch once, redraw when it lands
  const esyms=[...(j.indices||[]),...(j.sectors||[]),...(j.themes||[])]
    .map(v=>v.kind==='theme'?THEME_ETF[v.key]:v.key)
    .concat((j.rotation.groups||[]).map(r=>r.etf)).filter(Boolean);
  if(esyms.some(s=>!(s in EXTD)))extFetch(esyms,curAsof()).then(()=>{if(mapOpen)drawMap();});
  const asof=document.getElementById('mapasof');
  if(asof)asof.textContent=j.as_of+(j.t&&j.t!=='live'?' @ '+j.t+' ET':(j.live?' · live':' · close'))+' · daily anchor '+j.anchor;
  const i=j.rotation.internals;
  const rows=j.rotation.groups.filter(r=>r.kind!=='industry');
  const irows=j.rotation.groups;
  let h='';
  h+=notifHtml();                                 // 🔔 grade-A notified events (persistent, clickable)
  if(j.alerts&&j.alerts.length){
    // QUALITY-FIRST top of Map (Amir 2026-07-08): only the best transitions by default —
    // grade-A or an actionable read (ACT/STALK), newest 8; the full stream behind "show all".
    const qual=a=>{const s=a.sig||{};return s.grade==='A'||/^(ACT|STALK)/.test(s.action||'');};
    const best=j.alerts.filter(qual).slice(-8).reverse();
    const rowT=a=>'<div style="margin:2px 0"><span style="color:var(--dim)">'+a.t+'</span> <span class="'
      +(a.side==='bear'?'dn':'up')+'">'+a.text+'</span></div>';
    h+='<h3>Transitions <span style="font:400 11px monospace;color:var(--dim)">'
      +(best.length?best.length+' best':'0 best')+' of '+j.alerts.length
      +' · <a href="#" onclick="document.getElementById(\'alltrans\').style.display=\'block\';return false" style="color:var(--dim)">show all</a></span></h3>';
    h+=(best.length?best.map(rowT).join(''):'<div style="color:var(--dim)">no grade-A / actionable transitions yet — full stream under “show all”</div>');
    h+='<div id="alltrans" style="display:none;border-top:1px solid var(--line);margin-top:4px;padding-top:4px">'
      +j.alerts.slice().reverse().map(rowT).join('')+'</div>';
  }
  h+=radarHtml();
  h+=rvolHtml();
  h+='<h3>Suggestions</h3><div class="calltbl">'
    +(j.calls.length?j.calls.map(callHtml).join(''):'<div style="color:var(--dim)">no rule fired at this moment — nothing actionable by the current playbook</div>')+'</div>';
  h+='<h3>Indices</h3><div class="mapcards">'+j.indices.map(mapCard).join('')+'</div>';
  if(i)h+='<div style="color:var(--dim);margin-top:5px;font-size:11px">internals: '+i.pct_up.toFixed(0)+'% up · A/D '+i.ad_ratio
    +':1 · TICK '+(i.tick>0?'+':'')+i.tick+' · '+i.pct_vwap.toFixed(0)+'% &gt;VWAP · cum A-D '+i.cum_ad+' ('+i.cum_trend+')</div>';
  h+='<h3>Rotation <span class="mapmode" style="margin-left:8px">'
    +'<button id="mrq" class="'+(mapRotView==='quad'?'active':'')+'">Quadrant</button>'
    +'<button id="mrt" class="'+(mapRotView==='tiles'?'active':'')+'">Tiles</button>'
    +'<button id="mrx" title="expand the quadrant to a full-screen pane">⛶ Expand</button></span></h3>';
  h+=(mapRotView==='quad')?quadSvg(rows):rotTiles(irows);
  h+='<h3>Ratios</h3><div class="ratiostrip">'+j.ratios.map(rTile).join('')+'</div>';
  h+='<h3 style="cursor:help" title="hover any card for its structure phrase — glyph key: ⤴ fresh breakout · ▲ up · ⤵ fresh breakdown · ▼ down · ▽ distribution · ▭ range · ◇ contraction">Sectors &amp; Themes (structure) <span style="border:1px solid var(--line);border-radius:8px;padding:0 6px;color:var(--dim)">?</span></h3><div class="mapcards" style="grid-template-columns:repeat(3,minmax(0,1fr))">'
    +j.sectors.concat(j.themes).map(mapCard).join('')+'</div>';
  h+=j.validation?('<div style="color:var(--dim);font-size:10px;margin-top:12px;border-top:1px solid var(--line);padding-top:6px">rule stats measured point-in-time over '
    +j.validation.window+' · '+j.validation.rules+' rules · '+j.validation.n_total+' scored calls — suggestions show hit rate vs the fire-every-day base rate</div>')
    :'<div style="color:var(--dim);font-size:10px;margin-top:12px;border-top:1px solid var(--line);padding-top:6px">no validation run yet — all rules unvalidated (run validate_awareness.py)</div>';
  el.innerHTML=h;
  const q=document.getElementById('mrq'),t=document.getElementById('mrt'),x=document.getElementById('mrx');
  if(q)q.onclick=()=>{mapRotView='quad';drawMap();};
  if(t)t.onclick=()=>{mapRotView='tiles';drawMap();};
  if(x)x.onclick=()=>{const rs=j.rotation.groups.filter(r=>r.kind!=='industry');
    const m=document.getElementById('quadmodal');
    m.style.display='flex';
    m.querySelector('.qbody').innerHTML=quadSvg(rs,Math.min(innerWidth-80,1500),innerHeight-140);};}

// ── Sector / Theme Performance panel ─────────────────────────────────────────
let perfData=null, perfSort='m1_pct', perfView='themes', perfDrill=null;  // perfDrill={group,gtype} while drilled into a group (survives live refresh)
let boardData=null, boardSort='w1';   // merged panel: /api/themeboard rows (median, live-open), sorted by this key
const perfDrawer=document.getElementById('perfdrawer');
const perfScrim=document.getElementById('scrim');  // reuse existing scrim

function openPerf(){
  perfDrawer.classList.add('open');
  if(boardData&&boardData.asof===curAsof())drawBoard();else loadBoard();
}
function closePerf(){
  perfDrawer.classList.remove('open');
}
document.getElementById('btnperf').onclick=openPerf;
document.getElementById('perfclose').onclick=closePerf;
document.getElementById('btnmap').onclick=()=>{mapOpen?closeMap():openMap();};
document.getElementById('btnthemes').onclick=openPerf;   // Themes folded into the Perf drawer (median board + live open)
// ── ticker search: open any LOADED symbol's chart + info via the existing mapPick path ──
let SYMSET=new Set();
(function(){const inp=document.getElementById('tkrsearch');if(!inp)return;
  fetch('/api/symbols').then(r=>r.json()).then(list=>{
    SYMSET=new Set(list.map(x=>x.s));
    const dl=document.getElementById('symlist');
    dl.innerHTML=list.map(x=>'<option value="'+x.s+'">'+(x.n?x.n.replace(/"/g,'')+(x.sec?' · '+x.sec:''):(x.sec||''))+'</option>').join('');
  }).catch(()=>{});
  function go(){const msg=document.getElementById('tkrsearchmsg');
    const s=(inp.value||'').trim().toUpperCase();if(!s)return;
    if(SYMSET.has(s)){msg.textContent='';mapPick('sym',s);inp.value='';inp.blur();}
    else{msg.textContent='not in loaded universe';setTimeout(()=>{msg.textContent='';},2500);}}
  inp.addEventListener('change',go);
  inp.addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();go();}});})();
document.getElementById('mapclose').onclick=closeMap;

document.querySelectorAll('#perfmode button[data-view]').forEach(b=>{
  b.onclick=()=>{
    perfView=b.dataset.view;
    document.querySelectorAll('#perfmode button[data-view]').forEach(x=>x.classList.remove('active'));
    b.classList.add('active');
    if(perfView==='futures') drawFutures();
    else if(perfView==='universe') drawUniverse();
    else if(boardData&&boardData.asof===curAsof()) drawBoard();      // themes / sectors / both -> the median board (filtered by view)
    else loadBoard();
  };
});

function loadBoard(silent){   // themes/sectors/both: the median leaders board + LIVE since-open, cheap + cycle-fresh
  const body=document.getElementById('perfbody');
  const sc=silent?body.scrollTop:0;
  if(!silent&&!boardData)body.innerHTML='<div class="empty">Loading&#8230;</div>';
  fetch('/api/themeboard?date='+encodeURIComponent(curAsof())+tq())
    .then(r=>r.json()).then(j=>{
      boardData=j;
      const sig=document.getElementById('perfsig');if(sig)sig.textContent='as of '+(j.asof||'');
      drawBoard(); if(silent)body.scrollTop=sc;
    }).catch(()=>{if(!silent)body.innerHTML='<div class="empty">Error loading</div>';});
}

function perfColor(v){
  if(v==null)return'color:var(--dim)';
  return v>0?'color:var(--up)':'color:var(--dn)';
}
function perfFmt(v){if(v==null)return'<span style="color:var(--dim)">—</span>';
  return'<span style="'+perfColor(v)+'">'+(v>0?'+':'')+v.toFixed(2)+'%</span>';}

const SETUP_LABELS={'gapper':'Gap','episodic_pivot':'EP','hvc':'HVC','flat_base':'FB','high_tight_flag':'HTF',
  'higher_low_ma':'HLMA','undercut_rally':'UR','delayed_hvc':'DHVC','qm_breakout':'QM',
  'uptrend':'↑trend','downtrend':'↓trend','ema_rider_bull':'Rider↑','ema_rider_bear':'Rider↓',
  'backburner':'BB','stairstep':'Step','cup_handle':'C&H','double_top':'DT',
  'head_shoulders':'H&S','inverse_hs':'iH&S','rsi_extreme_revert':'RSIxR','rsi_extreme_fade':'RSIxF'};

function boardRead(view){   // colored 1W rotation read for the current board view (green in / red out + mini bars)
  const rows=((boardData&&boardData.rows)||[]).filter(r=>view==='both'?true:r.kind===(view==='sectors'?'sector':'theme'));
  if(rows.length<4)return '<span style="color:var(--dim)">not enough groups for a rotation read</span>';
  const noun=view==='sectors'?'industries':(view==='both'?'groups':'themes');
  const rk=rows.slice().sort((a,b)=>(b.w1??-1e9)-(a.w1??-1e9));
  const top=rk.slice(0,3),bot=rk.slice(-3).reverse();
  const t8=rk.slice(0,8).map(r=>r.name);
  const d=t8.filter(n=>tbIn(n,TB_DEFENSIVE)).length,o=t8.filter(n=>tbIn(n,TB_OFFENSIVE)).length;
  let tone,tc;
  if(d>=3&&o<=1){tone='RISK-OFF — defensive '+noun+' lead; be picky with high-beta longs, weak growth is the short pond';tc='var(--dn)';}
  else if(o>=3&&d<=1){tone='RISK-ON — high-beta '+noun+' lead; momentum longs in the leaders get the benefit of the doubt';tc='var(--up)';}
  else{tone='mixed tape — no clean rotation; trade the setups, not a '+noun.replace(/s$/,'')+' tilt';tc='var(--dim)';}
  const chip=r=>{const v=r.w1==null?0:r.w1,up=v>=0,col=up?'var(--up)':'var(--dn)',bw=Math.max(6,Math.min(46,Math.abs(v)*3));
    return '<span style="display:inline-flex;align-items:center;gap:5px;margin:0 8px 4px 0" title="'+r.name+' · 1W '+(up?'+':'')+v.toFixed(1)+'%">'
      +'<b style="color:'+col+'">'+r.name+'</b>'
      +'<span style="display:inline-block;width:'+bw+'px;height:7px;border-radius:3px;background:'+col+';opacity:.7"></span>'
      +'<span style="color:'+col+';font-size:11px">'+(up?'+':'')+v.toFixed(1)+'%</span></span>';};
  const lbl=t=>'<span style="color:var(--dim);text-transform:uppercase;font-size:10px;letter-spacing:.05em;margin-right:4px">'+t+'</span>';
  return '<div style="margin-bottom:2px">'+lbl('money into · 1W')+top.map(chip).join('')+'</div>'
    +'<div style="margin-bottom:4px">'+lbl('bleeding out')+bot.map(chip).join('')+'</div>'
    +'<div style="color:'+tc+';font-weight:600;font-size:12px">'+tone+'</div>';
}
function drawBoard(){   // merged Themes+Perf: MEDIAN group returns w/ heatmap + rotation read + drill-in
  perfDrill=null;   // top-level board -> no group drilled in
  const sig=document.getElementById('perfsig');if(sig&&boardData)sig.textContent='as of '+(boardData.asof||'');
  const body=document.getElementById('perfbody');
  if(!boardData||!boardData.rows){body.innerHTML='<div class="empty">Loading&#8230;</div>';return;}
  const want=perfView==='sectors'?'sector':(perfView==='themes'?'theme':null);   // 'both' -> null (all)
  const rows=(boardData.rows||[]).filter(r=>want===null||r.kind===want)
    .slice().sort((a,b)=>((b[boardSort]??-1e9)-(a[boardSort]??-1e9)));
  const COLS=[['open','Open'],['w1','1W'],['m1','1M'],['m3','3M'],['ytd','YTD']];   // Open = LIVE since-09:30
  const mx={};COLS.forEach(([k])=>{mx[k]=Math.max(1,...rows.map(r=>Math.abs(r[k]??0)));});
  const cell=(k,v)=>{if(v==null)return '<td style="color:var(--dim);text-align:right">–</td>';
    const a=Math.min(1,Math.abs(v)/mx[k]);   // heatmap: green in / red out, intensity = magnitude
    const bg=v>=0?'rgba(47,191,143,'+(0.08+0.30*a).toFixed(2)+')':'rgba(224,90,109,'+(0.08+0.30*a).toFixed(2)+')';
    return '<td style="background:'+bg+';text-align:right">'+(v>0?'+':'')+v.toFixed(1)+'%</td>';};
  const th=(k,l)=>'<th data-k="'+k+'" style="cursor:pointer;text-align:right'+(k===boardSort?';color:var(--acc)':'')+'">'+l+(k===boardSort?' ▾':'')+'</th>';
  let html='<div style="font:12.5px var(--mono);line-height:1.5;padding:6px 8px">'+boardRead(perfView)+'</div>'
    +'<table class="perf"><thead><tr><th style="text-align:left">group</th><th>n</th><th>ETF</th>'
    +COLS.map(([k,l])=>th(k,l)).join('')+'</tr></thead><tbody>';
  rows.forEach(r=>{
    html+='<tr data-group="'+r.name.replace(/"/g,'&quot;')+'" data-gtype="'+r.kind+'" style="cursor:pointer" class="perf-row">'
      +'<td style="text-align:left">'+r.name+(r.kind==='sector'?' <span style="color:var(--dim);font-size:9px">ind</span>':'')+'</td>'
      +'<td style="color:var(--dim);font-size:10px">'+r.n+'</td>'
      +'<td style="color:var(--dim);font-size:10px">'+(r.etf||'–')+'</td>'
      +COLS.map(([k])=>cell(k,r[k])).join('')+'</tr>';
  });
  body.innerHTML=html+'</tbody></table>';
  body.querySelectorAll('th[data-k]').forEach(t=>t.onclick=()=>{boardSort=t.dataset.k;drawBoard();});
  body.querySelectorAll('tr[data-group]').forEach(tr=>tr.onclick=()=>loadGroupDetail(tr.dataset.group,tr.dataset.gtype));
}

// ── group drill-down ──────────────────────────────────────────────────────────
function loadGroupDetail(group, gtype, silent){
  perfDrill={group:group,gtype:gtype};   // remember the drill-in so live refresh redraws THIS group, not the top list
  const body=document.getElementById('perfbody');
  const sc=silent?body.scrollTop:0;      // silent (live) refresh: keep the current view + scroll, no "Loading…" flash
  if(!silent)body.innerHTML='<div class="empty">Loading '+group+'&#8230;</div>';
  const url='/api/performance/group?group='+encodeURIComponent(group)+'&type='+gtype+'&date='+encodeURIComponent(curAsof())+tq();
  fetch(url).then(r=>r.json()).then(j=>{drawGroupDetail(group,gtype,j);if(silent)body.scrollTop=sc;})
    .catch(()=>{if(!silent)body.innerHTML='<div class="empty">Error loading group</div>';});
}

function drawGroupDetail(group, gtype, data){
  const sig=document.getElementById('perfsig');if(sig&&data)sig.textContent='as of '+(data.as_of||'');
  const body=document.getElementById('perfbody');
  const tickers=data.tickers||[];

  // sort by the current perfSort column
  tickers.sort((a,b)=>{
    const av=a[perfSort], bv=b[perfSort];
    if(av==null&&bv==null)return 0;
    if(av==null)return 1; if(bv==null)return -1;
    return bv-av;
  });

  const thCols=['Ticker','Open','PrevCls','1W','1M','3M','Setups'];
  const sortKeys=['symbol','open_pct','prev_close_pct','w1_pct','m1_pct','m3_pct','setups'];
  function header(){
    return '<tr><th colspan="7" style="text-align:left;color:var(--acc);padding:4px 8px">'
      +'<button class="tool" id="perfback" style="margin-right:6px;padding:2px 7px">&#8592;</button>'
      +group+' <span style="color:var(--dim);font-weight:400">('+tickers.length+' tickers)</span></th></tr>'
      +'<tr>'+thCols.map((h,i)=>{
        const k=sortKeys[i];
        const cls=(k===perfSort&&k!=='symbol'&&k!=='setups')?' class="sorted"':'';
        return'<th'+cls+' data-k="'+k+'">'+h+'</th>';
      }).join('')+'</tr>';
  }

  function setupBadges(setups){
    if(!setups||!setups.length)return'<span style="color:var(--dim)">—</span>';
    return setups.map(s=>{
      const lbl=SETUP_LABELS[s]||s;
      return'<span style="background:#1a2535;border:1px solid var(--line);border-radius:3px;'
        +'padding:1px 4px;margin:1px;font-size:10px;white-space:nowrap">'+lbl+'</span>';
    }).join('');
  }

  let rows='';
  tickers.forEach(t=>{
    const hasSetup=t.setups&&t.setups.length;
    const hl=hasSetup?'background:#0d1820;':'' ;
    rows+='<tr data-sym="'+t.symbol+'" style="cursor:pointer;'+hl+'">'
      +'<td style="color:var(--acc)">'+jBtn(t.symbol,curAsof(),'1D')+t.symbol+'</td>'
      +'<td>'+perfFmt(t.open_pct)+'</td>'
      +'<td>'+perfFmt(t.prev_close_pct)+'</td>'
      +'<td>'+perfFmt(t.w1_pct)+'</td>'
      +'<td>'+perfFmt(t.m1_pct)+'</td>'
      +'<td>'+perfFmt(t.m3_pct)+'</td>'
      +'<td style="max-width:160px">'+setupBadges(t.setups)+'</td>'
      +'</tr>';
  });

  body.innerHTML='<table class="perf"><thead>'+header()+'</thead><tbody>'+rows+'</tbody></table>';

  document.getElementById('perfback').onclick=()=>{drawBoard();};

  body.querySelectorAll('th[data-k]').forEach(th=>{
    th.onclick=()=>{
      const k=th.dataset.k;
      if(k==='symbol'||k==='setups')return;
      perfSort=k; drawGroupDetail(group,gtype,data);
    };
  });

  body.querySelectorAll('tr[data-sym]').forEach(tr=>{
    tr.onclick=()=>{
      mapPick('sym', tr.dataset.sym);
    };
  });
}

// ── Universe tab ─────────────────────────────────────────────────────────────
function universeRead(){   // plain-English read for the Universe view (breadth interpretation + movers)
  const d=universeData; if(!d)return ''; const w1=(d.breadth||{}).w1;
  const broad=w1==null?'':(w1>=60?'<span style="color:var(--up)">broad — most names participating</span>'
    :(w1<=40?'<span style="color:var(--dn)">narrow — a few names carrying the tape</span>':'evenly split — no clear breadth edge'));
  const chip=x=>{const v=x.m1_pct,up=(v||0)>=0,c=up?'var(--up)':'var(--dn)';   // top/bottom are ranked by 1M
    return '<b style="color:'+c+'">'+x.symbol+'</b> <span style="color:'+c+';font-size:11px">'+(v==null?'':(up?'+':'')+v.toFixed(1)+'%')+'</span>';};
  const lead=(d.top||[]).slice(0,4).map(chip).join('  '),lag=(d.bottom||[]).slice(0,3).map(chip).join('  ');
  return '<div style="padding:6px 10px;font:12px var(--mono);line-height:1.6">'
    +(broad?'<div>'+broad+'</div>':'')
    +'<div><span style="color:var(--dim)">strongest · 1M:</span> '+lead+'</div>'
    +'<div><span style="color:var(--dim)">weakest · 1M:</span> '+lag+'</div></div>';
}
function futuresRead(){   // plain-English read for the Futures view (since-open leaders/laggards + risk tone)
  const t=(futuresData&&futuresData.tickers)||[]; const w=t.filter(x=>x.open_pct!=null).sort((a,b)=>b.open_pct-a.open_pct);
  if(!w.length)return '';
  const nm=x=>(x.name||x.symbol).replace(' Futures','').replace(' E-mini','').replace('=F','');
  const chip=x=>{const v=x.open_pct,up=v>=0,c=up?'var(--up)':'var(--dn)';
    return '<b style="color:'+c+'">'+nm(x)+'</b> <span style="color:'+c+';font-size:11px">'+(up?'+':'')+v.toFixed(2)+'%</span>';};
  const lead=w.slice(0,3).map(chip).join('  '),lag=w.slice(-2).reverse().map(chip).join('  ');
  const idx=t.filter(x=>/^(ES|NQ|RTY|YM)/.test(x.symbol)&&x.open_pct!=null),up=idx.filter(x=>x.open_pct>0).length;
  const tone=!idx.length?'':(up>=idx.length-1?'<span style="color:var(--up)">stocks bid — risk-on tape</span>'
    :(up===0?'<span style="color:var(--dn)">stocks offered — risk-off tape</span>':'mixed across the equity complex'));
  return '<div style="padding:6px 10px;border-bottom:1px solid var(--line);font:12px var(--mono);line-height:1.6">'
    +'<div><span style="color:var(--dim)">since open — leading:</span> '+lead+'</div>'
    +'<div><span style="color:var(--dim)">lagging:</span> '+lag+(tone?' · '+tone:'')+'</div></div>';
}
let universeData=null, univSort='m1_pct';

function drawUniverse(){
  const body=document.getElementById('perfbody');
  if(universeData&&universeData.as_of===curAsof()){renderUniverse();return;}
  body.innerHTML='<div class="empty">Loading universe&#8230;</div>';
  fetch('/api/performance/universe?date='+encodeURIComponent(curAsof())+tq())
    .then(r=>r.json()).then(j=>{universeData=j;renderUniverse();})
    .catch(()=>{body.innerHTML='<div class="empty">Error loading universe</div>';});
}

function breadthColor(pct){
  if(pct==null)return'var(--dim)';
  if(pct>=60)return'var(--up)'; if(pct<=40)return'var(--dn)'; return'var(--txt)';
}

function renderUniverse(){
  const body=document.getElementById('perfbody');
  const d=universeData; if(!d){body.innerHTML='<div class="empty">Loading&#8230;</div>';return;}
  const sig=document.getElementById('perfsig');if(sig)sig.textContent='as of '+(d.as_of||'');
  const b=d.breadth||{};

  // breadth bar
  const breadthHtml=universeRead()+'<div style="display:flex;gap:16px;padding:6px 10px;border-bottom:1px solid var(--line);font-family:var(--mono);font-size:11px">'
    +'<span style="color:var(--dim)">'+d.total+' tickers</span>'
    +'<span>1W &gt;0: <b style="color:'+breadthColor(b.w1)+'">'+( b.w1!=null?b.w1+'%':'—')+'</b></span>'
    +'<span>1M &gt;0: <b style="color:'+breadthColor(b.m1)+'">'+( b.m1!=null?b.m1+'%':'—')+'</b></span>'
    +'<span>3M &gt;0: <b style="color:'+breadthColor(b.m3)+'">'+( b.m3!=null?b.m3+'%':'—')+'</b></span>'
    +'</div>';

  const thCols=['Symbol','Open','PrevCls','1W','1M','3M','Sector'];
  const sortKeys=['symbol','open_pct','prev_close_pct','w1_pct','m1_pct','m3_pct','sector'];
  function header(){
    return '<tr>'+thCols.map((h,i)=>{
      const k=sortKeys[i];
      const cls=(k===univSort&&k!=='symbol'&&k!=='sector')?' class="sorted"':'';
      return'<th'+cls+' data-k="'+k+'">'+h+'</th>';
    }).join('')+'</tr>';
  }

  function univRows(rows){
    // re-sort by univSort
    const sorted=[...rows].sort((a,b)=>{
      if(univSort==='symbol') return a.symbol.localeCompare(b.symbol);
      if(univSort==='sector') return (a.sector||'').localeCompare(b.sector||'');
      const av=a[univSort],bv=b[univSort];
      if(av==null&&bv==null)return 0; if(av==null)return 1; if(bv==null)return-1;
      return bv-av;
    });
    return sorted.map(t=>{
      const sec=t.sector?'<span style="color:var(--dim);font-size:10px">'+t.sector+'</span>':'';
      return'<tr data-sym="'+t.symbol+'" style="cursor:pointer">'
        +'<td style="color:var(--acc)">'+jBtn(t.symbol,curAsof(),'1D')+t.symbol+'</td>'
        +'<td>'+perfFmt(t.open_pct)+'</td>'
        +'<td>'+perfFmt(t.prev_close_pct)+'</td>'
        +'<td>'+perfFmt(t.w1_pct)+'</td>'
        +'<td>'+perfFmt(t.m1_pct)+'</td>'
        +'<td>'+perfFmt(t.m3_pct)+'</td>'
        +'<td>'+sec+'</td>'
        +'</tr>';
    }).join('');
  }

  const tableHtml='<table class="perf"><thead>'+header()+'</thead><tbody>'
    +'<tr><td colspan="7" class="perf-section">Top 50 — 1M leaders</td></tr>'
    +univRows(d.top)
    +'<tr><td colspan="7" class="perf-section">Bottom 50 — 1M laggards</td></tr>'
    +univRows(d.bottom)
    +'</tbody></table>';

  body.innerHTML=breadthHtml+tableHtml;

  body.querySelectorAll('th[data-k]').forEach(th=>{
    th.onclick=()=>{
      const k=th.dataset.k;
      if(k==='sector')return;
      univSort=k; renderUniverse();
    };
  });
  body.querySelectorAll('tr[data-sym]').forEach(tr=>{
    tr.onclick=()=>{mapPick('sym', tr.dataset.sym);};
  });
}

// ── Futures tab ───────────────────────────────────────────────────────────────
let futuresData=null;

function drawFutures(){
  const body=document.getElementById('perfbody');
  if(futuresData&&futuresData.as_of===curAsof()){renderFutures();return;}
  body.innerHTML='<div class="empty">Loading futures&#8230;</div>';
  fetch('/api/performance/futures?date='+encodeURIComponent(curAsof())+tq())
    .then(r=>r.json()).then(j=>{futuresData=j;renderFutures();})
    .catch(()=>{body.innerHTML='<div class="empty">Error loading futures</div>';});
}

// group order for display
const FUTURES_GROUPS=[
  {label:'Index',    syms:['ES=F','NQ=F','RTY=F','YM=F']},
  {label:'Energy',   syms:['CL=F','NG=F']},
  {label:'Metals',   syms:['GC=F','SI=F','HG=F','PL=F']},
  {label:'Grains',   syms:['ZW=F','ZC=F','ZS=F','ZL=F']},
  {label:'Crypto',   syms:['BTC=F']},
];

function renderFutures(){
  const body=document.getElementById('perfbody');
  if(!futuresData){body.innerHTML='<div class="empty">Loading&#8230;</div>';return;}
  const sig=document.getElementById('perfsig');if(sig)sig.textContent='as of '+(futuresData.as_of||'');
  const bySymbol={};
  (futuresData.tickers||[]).forEach(t=>{bySymbol[t.symbol]=t;});

  const thCols=['Symbol','Name','Open','PrevCls','1W','1M','3M'];
  const sortKeys=['symbol','name','open_pct','prev_close_pct','w1_pct','m1_pct','m3_pct'];
  const header='<tr>'+thCols.map((h,i)=>{
    const k=sortKeys[i];
    const cls=(k===perfSort&&k!=='symbol'&&k!=='name')?' class="sorted"':'';
    return'<th'+cls+' data-k="'+k+'">'+h+'</th>';
  }).join('')+'</tr>';

  let html='<table class="perf"><thead>'+header+'</thead><tbody>';
  FUTURES_GROUPS.forEach(g=>{
    html+='<tr><td colspan="7" class="perf-section">'+g.label+'</td></tr>';
    // sort within group by current perfSort
    const rows=g.syms.map(s=>bySymbol[s]).filter(Boolean);
    rows.sort((a,b)=>{
      if(perfSort==='symbol'||perfSort==='name')return 0;
      const av=a[perfSort],bv=b[perfSort];
      if(av==null&&bv==null)return 0; if(av==null)return 1; if(bv==null)return-1;
      return bv-av;
    });
    rows.forEach(t=>{
      const shortName=t.name.replace(' Futures','').replace(' E-mini','');
      html+='<tr data-sym="'+t.symbol+'" style="cursor:pointer">'
        +'<td style="color:var(--acc)">'+jBtn(t.symbol,curAsof(),'1D')+t.symbol.replace('=F','')+'</td>'
        +'<td style="color:var(--dim);font-size:11px">'+shortName+'</td>'
        +'<td>'+perfFmt(t.open_pct)+'</td>'
        +'<td>'+perfFmt(t.prev_close_pct)+'</td>'
        +'<td>'+perfFmt(t.w1_pct)+'</td>'
        +'<td>'+perfFmt(t.m1_pct)+'</td>'
        +'<td>'+perfFmt(t.m3_pct)+'</td>'
        +'</tr>';
    });
  });
  html+='</tbody></table>';
  body.innerHTML=futuresRead()+html;

  body.querySelectorAll('th[data-k]').forEach(th=>{
    th.onclick=()=>{
      const k=th.dataset.k;
      if(k==='symbol'||k==='name')return;
      perfSort=k; renderFutures();
    };
  });
  body.querySelectorAll('tr[data-sym]').forEach(tr=>{
    tr.onclick=()=>{mapPick('sym', tr.dataset.sym);};
  });
}

// ── Journal ───────────────────────────────────────────────────────────────────
function curAsof(){const di=document.getElementById('asof');return di&&di.value?di.value:new Date().toISOString().slice(0,10);}
function jBtn(sym,date,tf){
  return '<button class="jadd" onclick="event.stopPropagation();openJModal(\''+sym+'\',\''+date+'\',\''+tf+'\')" title="Save to journal">+</button>';
}

// ── modal — multi-setup ────────────────────────────────────────────────────────
const J_SETUP_OPTS=[...Object.keys(LABELS),'other'];
let jPickRow=null;  // index of setup row waiting for a chart-date click

function _setupOptHtml(selected){
  return J_SETUP_OPTS.map(s=>'<option value="'+s+'"'+(s===selected?' selected':'')+'>'+
    (LABELS[s]||s.charAt(0).toUpperCase()+s.slice(1))+'</option>').join('');
}

function _setupRowHtml(i,setup,setup_date){
  return '<div class="jsetup-row" id="jsr-'+i+'">'
    +'<select id="jss-'+i+'">'+_setupOptHtml(setup||cur)+'</select>'
    +'<input type="date" id="jsd-'+i+'" value="'+(setup_date||'')+'"> '
    +'<button class="jpick" id="jsp-'+i+'" onclick="activateJPick('+i+')" title="Click a chart bar to capture setup date">&#128205;</button>'
    +'<button class="jrm" onclick="removeJSetupRow('+i+')" title="Remove">&#x2715;</button>'
    +'</div>';
}

let _jRowCnt=0;
function addJSetupRow(setup,setup_date){
  const i=_jRowCnt++;
  const wrap=document.getElementById('jsetup-rows');
  const div=document.createElement('div');div.innerHTML=_setupRowHtml(i,setup,setup_date);
  wrap.appendChild(div.firstChild);
}
function removeJSetupRow(i){
  const el=document.getElementById('jsr-'+i);if(el)el.remove();
  if(jPickRow===i){jPickRow=null;document.getElementById('jpick-hint').style.display='none';}
}
function activateJPick(i){
  jPickRow=i;
  document.getElementById('jpick-hint').style.display='block';
  // highlight active pick button
  document.querySelectorAll('.jpick').forEach(b=>b.classList.remove('active'));
  const pb=document.getElementById('jsp-'+i);if(pb)pb.classList.add('active');
}
function _cancelJPick(){
  jPickRow=null;
  document.getElementById('jpick-hint').style.display='none';
  document.querySelectorAll('.jpick').forEach(b=>b.classList.remove('active'));
}

function openJModal(sym,date,tf){
  document.getElementById('jsym').value=sym;
  document.getElementById('jdate').value=date||curAsof();
  document.getElementById('jtf').value=tf||'1D';
  document.getElementById('jcomments').value='';
  document.getElementById('jsetup-rows').innerHTML='';
  _jRowCnt=0; jPickRow=null;
  document.getElementById('jpick-hint').style.display='none';
  addJSetupRow(cur,'');  // start with one row, guess current tab
  document.getElementById('jmodal-bg').classList.add('open');
  setTimeout(()=>document.getElementById('jcomments').focus(),50);
}
function closeJModal(){document.getElementById('jmodal-bg').classList.remove('open');_cancelJPick();}
async function saveJEntry(){
  const sym=document.getElementById('jsym').value;
  const date=document.getElementById('jdate').value;
  const tf=document.getElementById('jtf').value;
  const comments=document.getElementById('jcomments').value.trim();
  // collect setup rows
  const setups=[];
  document.querySelectorAll('#jsetup-rows .jsetup-row').forEach(row=>{
    const id=row.id.replace('jsr-','');
    const s=document.getElementById('jss-'+id),d=document.getElementById('jsd-'+id);
    if(s)setups.push({setup:s.value,setup_date:d?d.value:''});
  });
  if(!setups.length)setups.push({setup:'other',setup_date:''});
  const r=await fetch('/api/journal',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({symbol:sym,date,setups,tf,comments})});
  if(r.ok){closeJModal();jData=null;}
  else alert('Failed to save entry');
}
document.getElementById('jmodal-bg').onclick=function(e){if(e.target===this)closeJModal();};

// ── browser drawer ────────────────────────────────────────────────────────────
let jData=null;
const jDrawer=document.getElementById('jdrawer');
document.getElementById('jclose').onclick=()=>{jDrawer.classList.remove('open');_cancelJPick();};
document.getElementById('btnjournal').onclick=openJournal;
function openJournal(){jDrawer.classList.add('open');loadJournal();}

async function loadJournal(){
  const r=await fetch('/api/journal');const j=await r.json();
  jData=j.entries||[];
  const fs=document.getElementById('jfsetup');const prev=fs.value;
  const setups=[...new Set(jData.map(e=>e.setup))].sort();
  fs.innerHTML='<option value="">All Setups ('+jData.length+')</option>'+
    setups.map(s=>'<option value="'+s+'">'+s+' ('+jData.filter(e=>e.setup===s).length+')</option>').join('');
  fs.value=setups.includes(prev)?prev:'';
  renderJournal(jData,fs.value);
}

function renderJournal(entries,filterSetup){
  const body=document.getElementById('jbody');
  if(!entries){body.innerHTML='<div class="empty">Loading…</div>';return;}
  const filtered=filterSetup?entries.filter(e=>e.setup===filterSetup):entries;
  if(!filtered.length){body.innerHTML='<div class="empty">No entries.</div>';return;}
  let html='';
  filtered.forEach(e=>{
    // setups badges
    const slist=(e.setups&&e.setups.length)?e.setups:[{setup:e.setup,setup_date:''}];
    const badges=slist.map(s=>{
      const lbl=LABELS[s.setup]||s.setup;
      const dt=s.setup_date?'<span style="color:var(--dim);font-size:9px;margin-left:2px">'+s.setup_date+'</span>':'';
      return'<span style="display:inline-flex;align-items:center;background:#1a2535;border:1px solid var(--acc);'
        +'border-radius:3px;padding:1px 5px;margin:1px;font-size:10px;white-space:nowrap">'+lbl+dt+'</span>';
    }).join('');
    html+='<div class="jentry" data-id="'+e.id+'" data-sym="'+e.symbol+'" data-date="'+e.date+'" data-tf="'+e.tf+'">'
      +'<div style="display:flex;flex-direction:column;gap:2px;flex:1;min-width:0">'
      +'<div style="display:flex;align-items:center;gap:6px">'
      +'<span class="jsym">'+e.symbol+'</span><span class="jdate">'+e.date+'</span>'
      +'</div>'
      +'<div>'+badges+'</div>'
      +(e.comments?'<div class="jcomm">'+escHtml(e.comments)+'</div>':'')
      +'</div>'
      +'<button class="jdel" onclick="event.stopPropagation();deleteJ('+e.id+')" title="Delete">&#x2715;</button>'
      +'</div>'
      +'<div class="jedit-wrap" id="jedit-'+e.id+'" style="display:none">'
      +'<textarea id="jta-'+e.id+'">'+escHtml(e.comments)+'</textarea>'
      +'<div class="jbtns"><button class="tool" onclick="closeJEdit('+e.id+')">Cancel</button>'
      +'<button class="tool" style="border-color:var(--acc);color:var(--acc)" onclick="updateJ('+e.id+')">Save</button></div>'
      +'</div>';
  });
  body.innerHTML=html;
  body.querySelectorAll('.jentry').forEach(el=>{
    el.onclick=()=>{
      const id=+el.dataset.id;
      const ew=document.getElementById('jedit-'+id);
      if(ew.style.display!=='none'){closeJEdit(id);return;}
      body.querySelectorAll('.jedit-wrap').forEach(w=>{w.style.display='none';});
      ew.style.display='block';
      document.getElementById('jta-'+id).focus();
      chartJournal(el.dataset.sym,el.dataset.date,el.dataset.tf);
    };
  });
}
function escHtml(s){return(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function closeJEdit(id){const ew=document.getElementById('jedit-'+id);if(ew)ew.style.display='none';}
async function updateJ(id){
  const ta=document.getElementById('jta-'+id);
  const r=await fetch('/api/journal/'+id,{method:'PUT',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({comments:ta?ta.value.trim():''})});
  if(r.ok){jData=null;loadJournal();}else alert('Update failed');
}
async function deleteJ(id){
  if(!confirm('Delete this journal entry?'))return;
  const r=await fetch('/api/journal/'+id,{method:'DELETE'});
  if(r.ok){jData=null;loadJournal();}else alert('Delete failed');
}

async function chartJournal(sym,date,tf){
  renderMtf(sym);
  const head=document.getElementById('charthead'),el=document.getElementById('chart');
  head.innerHTML=sym+'  —  '+(tf||'1D')+'<div class="sub">journal: '+date+'</div>';
  el.innerHTML='<div class="empty">loading…</div>';
  try{
    const r=await fetch('/api/chart',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({symbol:sym,tf:tf||'1D',date:date,hit:{}})});
    const j=await r.json();
    if(j&&!j.error)drawChart(j);
    else{disposeChart();el.innerHTML='<div class="empty">chart unavailable</div>';}
  }catch(e){disposeChart();el.innerHTML='';}
}
"""
