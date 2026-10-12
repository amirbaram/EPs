"""Institutional Historical Daily Relative Strength (RS) Score Engine.

Precomputes daily IBD-style 1-99 percentile Relative Strength scores and multi-timeframe
technicals for EVERY stock and ETF across history.

Output:
  data/rs_history/rs_<YEAR>.parquet
    Columns:
      date          datetime64[ns]
      symbol        str
      name          str
      is_etf        bool
      close         float32
      rs_score      int8 (1-99)
      rank          int16 (1, 2, ...)
      raw_rs        float32
      ret_1d        float32
      ret_1w        float32
      ret_1m        float32
      ret_3m        float32
      ret_6m        float32
      ret_1y        float32
      alpha_1m      float32
      alpha_3m      float32
      alpha_1y      float32
      dist_52w_high float32
      adr_pct       float32
      avg_vol_50    int32
      dvol_m        float32
      above_10      bool
      above_20      bool
      above_50      bool
      above_200     bool
      stage2        bool
      sector        str
      theme         str

Usage:
  .venv/bin/python build_rs_history.py                       # full 10-year rebuild (2016-present)
  .venv/bin/python build_rs_history.py --start 2024-01-01    # rebuild from 2024 onwards
  .venv/bin/python build_rs_history.py --since 2026-09-01    # refresh recent sessions

Readers:
  from build_rs_history import load_rs_history, list_available_dates, load_symbol_rs_history
  df_day = load_rs_history("2024-05-23")
  df_sym = load_symbol_rs_history("NVDA")
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
import datastore
import labels

OUT_DIR = Path(config.DATA_DIR) / "rs_history"
START_DEFAULT = "2016-01-01"


def _load_universe_metadata() -> Tuple[Dict[str, str], Dict[str, bool]]:
    names_map: Dict[str, str] = {}
    etf_map: Dict[str, bool] = {}
    if config.UNIVERSE_FILE.exists():
        try:
            udf = pd.read_csv(config.UNIVERSE_FILE)
            names_map = dict(zip(udf["symbol"], udf["name"]))
            etf_map = dict(zip(udf["symbol"], udf.get("etf", False)))
        except Exception as e:
            print(f"Warning: could not load universe metadata: {e}")
    return names_map, etf_map


def _normalize_split_seams_series(c: np.ndarray, o: np.ndarray) -> np.ndarray:
    """Correct unback-adjusted split seams."""
    if len(c) < 5:
        return c
    c_adj = c.copy()
    for i in range(len(c) - 1, 0, -1):
        if c_adj[i - 1] <= 0:
            continue
        ratio = c_adj[i] / c_adj[i - 1]
        if ratio >= 2.85 or ratio <= 0.35:
            open_ratio = o[i] / c[i - 1]
            if (open_ratio >= 2.85 or open_ratio <= 0.35) and (ratio / open_ratio > 0.7) and (ratio / open_ratio < 1.3):
                mult = ratio
                c_adj[:i] *= mult
    return c_adj


def compute_rs_matrices(symbols: List[str], dates: pd.DatetimeIndex) -> Dict[str, pd.DataFrame]:
    """Build aligned wide OHLCV matrices and compute vectorized returns and indicators."""
    t0 = time.time()
    c_dict, v_dict, h_dict, l_dict = {}, {}, {}, {}

    for sym in symbols:
        df = datastore.load_bars(sym)
        if df is None or len(df) < 5:
            continue
        c = df["close"].to_numpy(dtype=np.float64)
        o = df["open"].to_numpy(dtype=np.float64) if "open" in df.columns else c
        c_norm = _normalize_split_seams_series(c, o)
        
        idx = df.index
        c_dict[sym] = pd.Series(c_norm, index=idx, dtype="float32")
        v_dict[sym] = df["volume"].astype("float32")
        h_dict[sym] = df["high"].astype("float32")
        l_dict[sym] = df["low"].astype("float32")

    print(f"  Loaded {len(c_dict)} symbols in {time.time()-t0:.1f}s", flush=True)

    # Reindex to master market calendar
    df_c = pd.DataFrame(c_dict).reindex(dates)
    df_v = pd.DataFrame(v_dict).reindex(dates)
    df_h = pd.DataFrame(h_dict).reindex(dates)
    df_l = pd.DataFrame(l_dict).reindex(dates)

    # Returns
    r_1d = (df_c / df_c.shift(1) - 1.0) * 100.0
    r_1w = (df_c / df_c.shift(5) - 1.0) * 100.0
    r_1m = (df_c / df_c.shift(21) - 1.0) * 100.0
    r_3m = (df_c / df_c.shift(63) - 1.0) * 100.0
    r_6m = (df_c / df_c.shift(126) - 1.0) * 100.0
    r_9m = (df_c / df_c.shift(189) - 1.0) * 100.0
    r_1y = (df_c / df_c.shift(252) - 1.0) * 100.0

    r6 = r_6m.combine_first(r_3m)
    r9 = r_9m.combine_first(r6)
    r1 = r_1y.combine_first(r9)

    raw_rs = 0.40 * r_3m + 0.20 * r6 + 0.20 * r9 + 0.20 * r1

    # Alpha vs SPY
    if "SPY" in df_c.columns:
        spy_1m = r_1m["SPY"]
        spy_3m = r_3m["SPY"]
        spy_1y = r_1y["SPY"]
        alpha_1m = r_1m.sub(spy_1m, axis=0)
        alpha_3m = r_3m.sub(spy_3m, axis=0)
        alpha_1y = r_1y.sub(spy_1y, axis=0)
    else:
        alpha_1m = r_1m.copy()
        alpha_3m = r_3m.copy()
        alpha_1y = r_1y.copy()

    # Rolling technicals
    hi_52w = df_h.rolling(252, min_periods=20).max()
    lo_52w = df_l.rolling(252, min_periods=20).min()
    dist_52w_high = (df_c / hi_52w - 1.0) * 100.0

    rng = (df_h - df_l) / df_c * 100.0
    adr_pct = rng.rolling(20, min_periods=5).mean()
    avg_vol_50 = df_v.rolling(50, min_periods=5).mean()
    dvol_m = (df_c * df_v).rolling(20, min_periods=5).mean() / 1e6

    ema10 = df_c.ewm(span=10, adjust=False).mean()
    ema20 = df_c.ewm(span=20, adjust=False).mean()
    sma50 = df_c.rolling(50, min_periods=20).mean()
    sma200 = df_c.rolling(200, min_periods=50).mean()

    above_10 = df_c >= ema10
    above_20 = df_c >= ema20
    above_50 = df_c >= sma50
    above_200 = df_c >= sma200
    stage2 = above_10 & (ema10 >= ema20) & (ema20 >= sma50) & (sma50 >= sma200)

    return {
        "close": df_c,
        "volume": df_v,
        "raw_rs": raw_rs,
        "ret_1d": r_1d,
        "ret_1w": r_1w,
        "ret_1m": r_1m,
        "ret_3m": r_3m,
        "ret_6m": r_6m,
        "ret_1y": r_1y,
        "alpha_1m": alpha_1m,
        "alpha_3m": alpha_3m,
        "alpha_1y": alpha_1y,
        "hi_52w": hi_52w,
        "lo_52w": lo_52w,
        "dist_52w_high": dist_52w_high,
        "adr_pct": adr_pct,
        "avg_vol_50": avg_vol_50,
        "dvol_m": dvol_m,
        "above_10": above_10,
        "above_20": above_20,
        "above_50": above_50,
        "above_200": above_200,
        "stage2": stage2,
    }


def build(since: Optional[str] = None, start: str = START_DEFAULT) -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading universe metadata and labels...", flush=True)
    names_map, etf_map = _load_universe_metadata()
    labels.load()

    all_symbols = datastore.list_symbols()
    spy = datastore.load_bars("SPY")
    if spy is None:
        raise RuntimeError("SPY bars not found in datastore")

    # Use all available market dates to calculate rolling 252-day windows properly
    all_dates = spy.index
    print(f"Master calendar: {len(all_dates)} sessions ({all_dates[0].date()} -> {all_dates[-1].date()})", flush=True)

    matrices = compute_rs_matrices(all_symbols, all_dates)
    df_c = matrices["close"]
    raw_rs = matrices["raw_rs"]

    # Identify common stocks vs ETFs
    etf_symbols = set(sym for sym, is_etf in etf_map.items() if is_etf)
    common_cols = [c for c in df_c.columns if c not in etf_symbols]
    etf_cols = [c for c in df_c.columns if c in etf_symbols]

    print("Computing cross-sectional RS percentiles...", flush=True)
    ranks_common = raw_rs[common_cols].rank(axis=1, pct=True) * 98.0 + 1.0
    ranks_etf = raw_rs[etf_cols].rank(axis=1, pct=True) * 98.0 + 1.0 if etf_cols else pd.DataFrame(index=all_dates)

    rs_score_wide = pd.concat([ranks_common, ranks_etf], axis=1)[df_c.columns].round()

    print(f"Matrix computations completed in {time.time()-t0:.1f}s. Assembling yearly parquet files...", flush=True)

    # Dates to persist to parquet
    save_cutoff = pd.Timestamp(since) if since else pd.Timestamp(start)
    dates_to_save = all_dates[all_dates >= save_cutoff]
    years = sorted(dates_to_save.year.unique())

    # Pre-map symbol attributes for fast lookup
    sec_map = {s: labels.sector(s) or "Unknown" for s in df_c.columns}
    thm_map = {s: (labels.themes(s) or ["General"])[0] for s in df_c.columns}
    thms_str_map = {s: ", ".join(labels.themes(s) or ["General"]) for s in df_c.columns}

    for yr in years:
        yr_dates = dates_to_save[dates_to_save.year == yr]
        if len(yr_dates) == 0:
            continue

        p_year = OUT_DIR / f"rs_{yr}.parquet"
        print(f"  Building {yr} ({len(yr_dates)} sessions)...", flush=True)

        frames = []
        for dt in yr_dates:
            c_row = df_c.loc[dt]
            rs_row = rs_score_wide.loc[dt]
            raw_row = raw_rs.loc[dt]

            # Filter active symbols on this date (has close > 0 and valid RS score)
            mask = c_row.notna() & (c_row > 0) & rs_row.notna()
            if not mask.any():
                continue

            active_syms = df_c.columns[mask]
            sub_c = c_row[active_syms]
            sub_rs = rs_row[active_syms].astype("int16")
            sub_raw = raw_row[active_syms].astype("float32")

            # Rank: highest RS score first, then raw_rs descending
            df_day = pd.DataFrame({
                "date": dt,
                "symbol": active_syms,
                "close": sub_c.astype("float32").round(2),
                "rs_score": sub_rs,
                "raw_rs": sub_raw.round(2),
                "ret_1d": matrices["ret_1d"].loc[dt, active_syms].astype("float32").round(2),
                "ret_1w": matrices["ret_1w"].loc[dt, active_syms].astype("float32").round(2),
                "ret_1m": matrices["ret_1m"].loc[dt, active_syms].astype("float32").round(2),
                "ret_3m": matrices["ret_3m"].loc[dt, active_syms].astype("float32").round(2),
                "ret_6m": matrices["ret_6m"].loc[dt, active_syms].astype("float32").round(2),
                "ret_1y": matrices["ret_1y"].loc[dt, active_syms].astype("float32").round(2),
                "alpha_1m": matrices["alpha_1m"].loc[dt, active_syms].astype("float32").round(2),
                "alpha_3m": matrices["alpha_3m"].loc[dt, active_syms].astype("float32").round(2),
                "alpha_1y": matrices["alpha_1y"].loc[dt, active_syms].astype("float32").round(2),
                "hi_52w": matrices["hi_52w"].loc[dt, active_syms].astype("float32").round(2),
                "lo_52w": matrices["lo_52w"].loc[dt, active_syms].astype("float32").round(2),
                "dist_52w_high": matrices["dist_52w_high"].loc[dt, active_syms].astype("float32").round(2),
                "adr_pct": matrices["adr_pct"].loc[dt, active_syms].astype("float32").round(2),
                "avg_vol_50": matrices["avg_vol_50"].loc[dt, active_syms].fillna(0).astype("int32"),
                "dvol_m": matrices["dvol_m"].loc[dt, active_syms].fillna(0).astype("float32").round(2),
                "above_10": matrices["above_10"].loc[dt, active_syms].fillna(False).astype(bool),
                "above_20": matrices["above_20"].loc[dt, active_syms].fillna(False).astype(bool),
                "above_50": matrices["above_50"].loc[dt, active_syms].fillna(False).astype(bool),
                "above_200": matrices["above_200"].loc[dt, active_syms].fillna(False).astype(bool),
                "stage2": matrices["stage2"].loc[dt, active_syms].fillna(False).astype(bool),
            })

            # Sort and add rank
            df_day = df_day.sort_values(["rs_score", "raw_rs"], ascending=[False, False]).reset_index(drop=True)
            df_day["rank"] = (df_day.index + 1).astype("int16")

            # Add static metadata
            df_day["name"] = df_day["symbol"].map(names_map).fillna(df_day["symbol"])
            df_day["is_etf"] = df_day["symbol"].map(lambda s: etf_map.get(s, False)).astype(bool)
            df_day["sector"] = df_day["symbol"].map(sec_map).fillna("Unknown")
            df_day["theme"] = df_day["symbol"].map(thm_map).fillna("General")
            df_day["themes"] = df_day["symbol"].map(thms_str_map).fillna("General")

            frames.append(df_day)

        if not frames:
            continue

        yr_df = pd.concat(frames, ignore_index=True)

        if since is not None and p_year.exists():
            old = pd.read_parquet(p_year)
            yr_df = pd.concat([old[old["date"] < save_cutoff], yr_df], ignore_index=True)

        yr_df.sort_values(["date", "rank"]).to_parquet(
            p_year,
            index=False,
            compression="zstd",
            row_group_size=100_000,
        )
        print(f"  Wrote {p_year.name} ({len(yr_df):,} rows, {p_year.stat().st_size / 1024 / 1024:.2f} MB)", flush=True)

    print(f"All RS history built in {time.time()-t0:.1f}s -> {OUT_DIR}", flush=True)


# ── Readers & API Helpers ───────────────────────────────────────────────────────
def list_available_dates() -> List[str]:
    """Return sorted unique dates (YYYY-MM-DD) available in rs_history."""
    if not OUT_DIR.exists():
        return []
    files = sorted(OUT_DIR.glob("rs_*.parquet"))
    if not files:
        return []
    dates_set = set()
    for f in files:
        try:
            # Read just the date column
            df_dates = pd.read_parquet(f, columns=["date"])
            dates_set.update(df_dates["date"].dt.strftime("%Y-%m-%d").unique())
        except Exception:
            continue
    return sorted(dates_set)


def load_rs_history(date: str) -> pd.DataFrame:
    """Load institutional RS score leaderboard for one specific trading date."""
    target_dt = pd.Timestamp(date)
    p = OUT_DIR / f"rs_{target_dt.year}.parquet"
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(p, filters=[("date", "==", target_dt)])
    except Exception as e:
        print(f"Error loading RS history for {date}: {e}")
        return pd.DataFrame()


def load_symbol_rs_history(symbol: str) -> pd.DataFrame:
    """Load daily RS score time-series for a single symbol across all years."""
    if not OUT_DIR.exists():
        return pd.DataFrame()
    files = sorted(OUT_DIR.glob("rs_*.parquet"))
    frames = []
    for f in files:
        try:
            df_part = pd.read_parquet(f, filters=[("symbol", "==", symbol)])
            if not df_part.empty:
                frames.append(df_part)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default=START_DEFAULT, help="First date to compute (default: 2016-01-01)")
    parser.add_argument("--since", help="Only refresh dates >= this (YYYY-MM-DD)")
    args = parser.parse_args()
    build(since=args.since, start=args.start)
