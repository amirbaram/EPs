"""Orchestrator.

  python scan.py init   [--max-tickers N]   one-time bulk history download
  python scan.py daily                      incremental update + scan + report
  python scan.py scan                       scan cached data only (no download)
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd


# --- open-file soft-limit raise (copied from serve.py) ---
try:
    import resource as _resource
    _nofile_soft, _nofile_hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
    _nofile_want = 65536 if _nofile_hard == _resource.RLIM_INFINITY else min(65536, _nofile_hard)
    if _nofile_soft < _nofile_want:
        _resource.setrlimit(_resource.RLIMIT_NOFILE, (_nofile_want, _nofile_hard))
except Exception:
    pass

import config
import datastore
import marketcap
import patterns
import quality
import regime
import settings
import setups
import universe as uni
from indicators import add_indicators
from setups import run_all, run_multi_tf


MIN_BARS_W = 30   # weekly frames need at least this many weeks to run the new setups


def _base_row(sym: str, d: pd.DataFrame, meta) -> dict:
    """Per-ticker base columns (liquidity + regime + returns), independent of any per-setup
    knobs — depends only on REGIME_* / ATR_FRACTION. The server caches this map by the regime
    signature and reuses it across setup recomputes, so tuning one setup never recomputes the
    (expensive) regime/trendlab annotation. Reuses the slice's _trendlab attrs cache, so when
    the uptrend/downtrend detectors ran first on the same slice, no extra trendlab pass."""
    last = d.iloc[-1]
    creg = regime.current_regime(d) if config.REGIME_USE else None
    return {
        "symbol": sym,
        "exchange": meta["exchange"] if meta is not None else "",
        "name": (meta["name"][:40] if meta is not None else ""),
        "is_etf": (str(meta["etf"]).strip().lower() in ("true", "y", "1")) if meta is not None else False,
        "date": d.index[-1].date().isoformat(),
        "close": round(float(last["close"]), 2),
        "market_cap": marketcap.get(sym),       # dollars (from cache); None if never fetched

        "adr_pct": round(float(last["adr_pct"]), 2) if pd.notna(last["adr_pct"]) else None,
        "rvol_today": round(float(last["rvol"]), 2) if pd.notna(last["rvol"]) else None,
        "dollar_vol_m": round(float(last["dollar_vol"]) / 1e6, 2) if pd.notna(last["dollar_vol"]) else None,
        "off_hi52_pct": round(float(last["off_hi52_pct"]), 1) if pd.notna(last["off_hi52_pct"]) else None,
        "off_ath_pct": round((float(last["close"]) / float(d["high"].max()) - 1) * 100, 1) if d["high"].max() > 0 else None,
        "above_lo52_pct": round(float(last["above_lo52_pct"]), 1) if pd.notna(last["above_lo52_pct"]) else None,
        "avg_vol_k": round(float(last["vol_avg50"]) / 1e3, 1) if pd.notna(last["vol_avg50"]) else None,
        "above_sma50": bool(last["close"] > last["sma50"]) if pd.notna(last["sma50"]) else None,
        "above_sma200": bool(last["close"] > last["sma200"]) if pd.notna(last["sma200"]) else None,
        "regime_dir": (creg.direction or "") if creg else "",
        "regime_clarity": round(creg.clarity, 2) if creg else None,
        "regime_net_pct": round(creg.net_pct, 1) if creg else None,
        "ret_1m": round(float(last["ret_1m"]), 1) if pd.notna(last["ret_1m"]) else None,
        "ret_3m": round(float(last["ret_3m"]), 1) if pd.notna(last["ret_3m"]) else None,
        "ret_6m": round(float(last["ret_6m"]), 1) if pd.notna(last["ret_6m"]) else None,
        "rsi": round(float(last["rsi14"]), 1) if "rsi14" in d.columns and pd.notna(last["rsi14"]) else None,
        "adx": round(float(last["adx14"]), 1) if "adx14" in d.columns and pd.notna(last["adx14"]) else None,
    }


def _rows_from_enriched(sym: str, d: pd.DataFrame, meta, weekly: pd.DataFrame | None = None,
                        only: set | None = None, bases: dict | None = None, context: dict | None = None) -> list[dict]:
    """Enriched (possibly truncated) daily frame -> hit row dicts with base liquidity columns +
    regime annotation + timeframe tag. `only` restricts to a subset of setups (per-setup
    recompute). `bases` (when a dict) is a reusable per-ticker base cache: a hit reuses bases[sym]
    if present, else the base is computed once and stored there (so the server can cache the
    expensive regime pass and skip it on later setup recomputes). Look-ahead-safe: backward-only."""
    if len(d) < config.MIN_BARS:
        return []
    # illiquid ETFs stay in the universe (Perf / charts) but generate NO setup hits — a curated ETF
    # bypasses the universe liquidity floors, so gate it here against those same floors.
    if meta is not None and str(meta.get("etf", "")).strip().lower() in ("true", "y", "1"):
        last = d.iloc[-1]
        va, dv = last.get("vol_avg50"), last.get("dollar_vol")
        if (pd.isna(va) or va < config.UNIVERSE_MIN_AVG_VOL_K * 1000
                or pd.isna(dv) or dv < config.UNIVERSE_MIN_DVOL_M * 1e6):
            return []
    daily_hits = run_all(sym, d, context, only) + run_multi_tf(sym, d, context, only)
    weekly_hits = run_multi_tf(sym, weekly, context, only) if (weekly is not None and len(weekly) >= MIN_BARS_W) else []
    if not daily_hits and not weekly_hits:
        return []
    caching = bases is not None
    if caching and sym in bases:
        base = bases[sym]
    else:
        base = _base_row(sym, d, meta)
        if caching:
            bases[sym] = base
    rows = [{**base, "tf": hit.pop("tf_override", "1D"), **hit} for hit in daily_hits]
    rows += [{**base, "tf": "1W", **hit} for hit in weekly_hits]
    for r in rows:
        frame = weekly if r["tf"] == "1W" else d
        r.update(quality.score_hit(frame, r))      # composite quality score (sortable)
        cb = r.get("cons_bars")                     # classify the consolidation shape, if any
        if "pattern" not in r and cb and int(cb) >= 3:
            cb = int(cb)
            r["pattern"] = patterns.classify_consolidation(frame, len(frame) - cb, len(frame) - 1)["pattern"]
    return rows


def _weekly_enriched(sym: str):
    wdf = datastore.load_bars_w(sym)
    return add_indicators(wdf) if wdf is not None and len(wdf) >= MIN_BARS_W else None


def _rs(last) -> tuple:
    g = lambda k: float(last[k]) if pd.notna(last[k]) else None
    return (g("ret_1m"), g("ret_3m"), g("ret_6m"))


def _rs_ranks(rs_raw: dict) -> pd.DataFrame:
    """Three separate RS lists, the way Qullamaggie scans them: percentile-rank each of the
    1/3/6-mo returns across the whole universe (0-100 = rs_1m/rs_3m/rs_6m). rs_rank = the MAX
    of the three = top of AT LEAST ONE list (the union of the three best-performer lists).
    Returns a frame indexed by symbol; merged onto the hits."""
    cols = ["rs_1m", "rs_3m", "rs_6m"]
    if not rs_raw:
        return pd.DataFrame(columns=cols + ["rs_rank"])
    pr = (pd.DataFrame(rs_raw, index=cols).T.rank(pct=True) * 100).round(0)
    pr["rs_rank"] = pr[cols].max(axis=1)
    return pr


def scan_all() -> pd.DataFrame:
    u = uni.load_universe().set_index("yf_symbol")
    rows, rs_raw = [], {}
    syms = datastore.list_symbols()
    active = uni.active_symbols()
    if active is not None:
        syms = [s for s in syms if s in active]
        
    print(f"pre-loading {len(syms)} frames for context ...")
    frames = {}
    for sym in syms:
        df = datastore.load_bars(sym)
        if df is not None: frames[sym] = df
        
    import ratios
    import labels
    labels.load()
    labels.load()
    
    spy_df = frames.get('SPY')
    sector_dfs, theme_dfs = {}, {}
    synthetic_syms = []
    
    for sec in set(labels._SECTOR.values()):
        members = {s for s, c in labels._SECTOR.items() if c == sec} & set(syms)
        bs = ratios.basket_series(members, frames, min_members=3)
        if bs is not None:
            synth_name = f"SYNTH_SEC_{sec.replace(' ', '').replace('&', 'And').replace('-', '')}".upper()
            sector_dfs[sec] = bs
            frames[synth_name] = bs
            synthetic_syms.append(synth_name)
            
    for thm in set(labels._SYMS_BY_THEME.keys()):
        members = labels._SYMS_BY_THEME[thm] & set(syms)
        bs = ratios.basket_series(members, frames, min_members=3)
        if bs is not None:
            synth_name = f"SYNTH_THM_{thm.replace(' ', '').replace('&', 'And').replace('-', '')}".upper()
            theme_dfs[thm] = bs
            frames[synth_name] = bs
            synthetic_syms.append(synth_name)
            
    context = {'spy_df': spy_df, 'sector_dfs': sector_dfs, 'theme_dfs': theme_dfs, 'frames': frames}
    all_syms = syms + synthetic_syms
    
    print(f"scanning {len(all_syms)} tickers & synthetics ...")
    for i, sym in enumerate(all_syms):
        df = frames.get(sym)
        if df is None or len(df) < config.MIN_BARS:
            continue
        e = add_indicators(df)
        if sym in syms:
            rs_raw[sym] = _rs(e.iloc[-1])
        meta = u.loc[sym] if sym in u.index else None
        rows += _rows_from_enriched(sym, e, meta, _weekly_enriched(sym) if sym in syms else None, context=context)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(all_syms)}")
            
    hits = pd.DataFrame(rows)
    if len(hits):
        hits = hits.merge(_rs_ranks(rs_raw), left_on="symbol", right_index=True, how="left")
    print(f"found {len(hits)} setup hits")
    return hits


def scan_asof(date, enriched_frames: dict, u: pd.DataFrame,
              weekly_frames: dict | None = None, only: set | None = None,
              bases: dict | None = None) -> pd.DataFrame:
    """Point-in-time scan for the live server. `enriched_frames` / `weekly_frames` =
    {sym: enriched_df} held in RAM; each is truncated at `date` (inclusive) before
    scanning, so hits reflect only what was knowable on that date. `only` restricts the
    run to a subset of setups (the server recomputes just the setups whose knobs changed).
    `bases` (a dict) is a reusable per-ticker base/regime cache — passed empty it gets filled
    (cache it by regime signature); passed full it's reused so regime isn't recomputed."""
    cutoff = pd.Timestamp(date)
    # the weekly frames only matter for the multi-timeframe setups; skip slicing them when the
    # recompute set is daily-only (e.g. tuning a daily setup like EMA Rider)
    mtf = {setups.SETUP_OF[fn] for fn in setups.MULTI_TF_DETECTORS}
    need_weekly = weekly_frames is not None and (only is None or bool(only & mtf))
    rows, rs_raw = [], {}
    # Extract spy_df for context
    spy_df = enriched_frames.get('SPY')
    spy_df = spy_df.loc[:cutoff] if spy_df is not None else None
    
    sector_dfs, theme_dfs = {}, {}
    import labels
    labels.load()
    for sec in set(labels._SECTOR.values()):
        synth_name = f"SYNTH_SEC_{sec.replace(' ', '').replace('&', 'And').replace('-', '')}".upper()
        df = enriched_frames.get(synth_name)
        if df is not None: sector_dfs[sec] = df.loc[:cutoff]
            
    for thm in set(labels._SYMS_BY_THEME.keys()):
        synth_name = f"SYNTH_THM_{thm.replace(' ', '').replace('&', 'And').replace('-', '')}".upper()
        df = enriched_frames.get(synth_name)
        if df is not None: theme_dfs[thm] = df.loc[:cutoff]
        
    
    import scanner_core
    synth_states = {}
    if spy_df is not None:
        # Pre-evaluate Sector vs SPY
        for sec, df in sector_dfs.items():
            try:
                idx = df.index.intersection(spy_df.index)
                d_rs = df.loc[idx].copy()
                d_rs['close'] = d_rs['close'] / spy_df.loc[idx, 'close']
                d_rs = scanner_core.calc_larssson_line(d_rs)
                states = d_rs['larsson_state'].dropna()
                if len(states) > 0:
                    synth_states[f"SYNTH_SEC_{sec.replace(' ', '').replace('&', 'And').replace('-', '')}".upper()] = states.iloc[-1]
            except Exception: pass
            
        # Pre-evaluate Theme vs SPY
        for thm, df in theme_dfs.items():
            try:
                idx = df.index.intersection(spy_df.index)
                d_rs = df.loc[idx].copy()
                d_rs['close'] = d_rs['close'] / spy_df.loc[idx, 'close']
                d_rs = scanner_core.calc_larssson_line(d_rs)
                states = d_rs['larsson_state'].dropna()
                if len(states) > 0:
                    synth_states[f"SYNTH_THM_{thm.replace(' ', '').replace('&', 'And').replace('-', '')}".upper()] = states.iloc[-1]
            except Exception: pass

    spy_targets = {"1D": spy_df} if spy_df is not None else {}
    if spy_df is not None and len(spy_df) >= 20:
        for tf in ("2D", "3D", "1W"):
            st = datastore.resample_daily(spy_df[["open", "high", "low", "close", "volume"]], tf)
            if st is not None and len(st) >= 10:
                spy_targets[tf] = st

    context = {'spy_df': spy_df, 'spy_targets': spy_targets, 'sector_dfs': sector_dfs, 'theme_dfs': theme_dfs, 'synth_states': synth_states, 'frames': enriched_frames}
    active = uni.active_symbols()
    for sym, ed in enriched_frames.items():
        if active is not None and sym not in active and not sym.startswith("SYNTH_"):
            continue
        d = ed.loc[:cutoff]
        if len(d) < config.MIN_BARS:
            continue
        rs_raw[sym] = _rs(d.iloc[-1])
        weekly = None
        if need_weekly:
            wf = weekly_frames.get(sym)
            if wf is not None:
                weekly = wf.loc[:cutoff]
                if len(weekly) < MIN_BARS_W:
                    weekly = None
        meta = u.loc[sym] if sym in u.index else None
        rows += _rows_from_enriched(sym, d, meta, weekly, only, bases, context)
    hits = pd.DataFrame(rows)
    if len(hits):
        hits = hits.merge(_rs_ranks(rs_raw), left_on="symbol", right_index=True, how="left")
    return hits


def save_outputs(hits: pd.DataFrame) -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    date = hits["date"].max() if len(hits) else "empty"
    csv_path = config.OUTPUT_DIR / f"hits_{date}.csv"
    hits.to_csv(csv_path, index=False)

    # TradingView-importable watchlist per setup (EXCHANGE:SYMBOL, comma separated)
    if len(hits):
        for setup, grp in hits.groupby("setup"):
            syms = [f"{r.exchange}:{r.symbol.replace('-', '.')}" if r.exchange else r.symbol
                    for r in grp.itertuples()]
            (config.OUTPUT_DIR / f"watchlist_{setup}_{date}.txt").write_text(",".join(syms))
    print(f"saved {csv_path} + watchlist txt files")

    if config.DISCORD_WEBHOOK and len(hits):
        try:
            import urllib.request
            counts = hits["setup"].value_counts().to_dict()
            top = hits.sort_values("dollar_vol_m", ascending=False).head(15)
            lines = [f"**Scan {date}** — " + ", ".join(f"{k}: {v}" for k, v in counts.items()), ""]
            lines += [f"`{r.symbol:<6}` {r.setup} (${r.close})" for r in top.itertuples()]
            req = urllib.request.Request(
                config.DISCORD_WEBHOOK,
                data=json.dumps({"content": "\n".join(lines)[:1900]}).encode(),
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=15)
        except Exception as e:
            print(f"discord push failed: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init", "daily", "weekly", "scan"])
    ap.add_argument("--max-tickers", type=int, default=None)
    args = ap.parse_args()

    if args.cmd == "init":
        uni.build_universe()
        datastore.init_history(max_tickers=args.max_tickers)
        return

    if args.cmd == "weekly":          # full-universe refresh + re-evaluate the active membership
        datastore.update_daily(full=True)
        import build_active_universe
        build_active_universe.build()
        return

    if args.cmd == "daily":           # normal: active universe only (no out-of-universe downloads)
        datastore.update_daily()

    with settings.apply():            # honor any UI-saved threshold overrides
        hits = scan_all()
    save_outputs(hits)
    from report import build_report
    out = build_report(hits)
    print(f"dashboard: {out}")


if __name__ == "__main__":
    sys.exit(main())
