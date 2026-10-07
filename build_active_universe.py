"""(Re)build the reduced 'active universe' — the liquid subset the app + nightly scan load.

  .venv/bin/python build_active_universe.py        # run weekly

For every cached ticker it computes last-bar liquidity stats from the daily bars and keeps the
ones passing ALL config floors (UNIVERSE_MIN_*), plus the curated ETFs always. Writes
data/universe_active.csv (yf_symbol + stats). The bar cache is untouched — a ticker that later
qualifies re-enters on the next rebuild with no re-download.
"""
from __future__ import annotations

import csv

import pandas as pd

import config
import datastore
import marketcap
import universe as uni
from indicators import add_indicators


def _stats(sym: str):
    """(avg_vol_k, price, dvol_m, adr_pct) from the last enriched daily bar, or None."""
    df = datastore.load_bars(sym)
    if df is None or len(df) < config.MIN_BARS:
        return None
    last = add_indicators(df).iloc[-1]
    g = lambda k: float(last[k]) if pd.notna(last[k]) else None
    vol_k = (last["vol_avg50"] / 1e3) if pd.notna(last["vol_avg50"]) else None
    dvol_m = (last["dollar_vol"] / 1e6) if pd.notna(last["dollar_vol"]) else None
    return vol_k, g("close"), dvol_m, g("adr_pct")


def build() -> int:
    syms = datastore.list_symbols()
    # ---- pass 1: liquidity floors from the cached bars (no network) -----------------------------
    liquid = []
    for i, sym in enumerate(syms, 1):
        s = _stats(sym)
        curated = sym in uni.CURATED_ETFS or sym in uni.FUTURES_SYMBOLS   # always keep ETFs + futures
        if s is None:
            continue
        vol_k, price, dvol_m, adr = s
        # liquidity: pass the 750K share floor OR (healthy $-volume AND a lower 100K share floor) —
        # so high-priced, liquid-by-dollars names (ARGX etc.) qualify but truly thin ones don't.
        vol_ok = (vol_k is not None and dvol_m is not None and (
            vol_k >= config.UNIVERSE_MIN_AVG_VOL_K
            or (dvol_m >= config.UNIVERSE_MIN_DVOL_M and vol_k >= config.UNIVERSE_DVOL_OR_MIN_VOL_K)))
        ok = (curated or (
            vol_ok
            and price is not None and price >= config.UNIVERSE_MIN_PRICE
            and dvol_m is not None and dvol_m >= config.UNIVERSE_MIN_DVOL_M
            and adr is not None and adr >= config.UNIVERSE_MIN_ADR_PCT))
        if ok:
            liquid.append((sym, round(vol_k or 0, 1), round(price or 0, 2),
                           round(dvol_m or 0, 2), round(adr or 0, 2), curated))
        if i % 1000 == 0:
            print(f"  scanned {i}/{len(syms)} (liquid {len(liquid)})")
    # ---- pass 2: market-cap floor (fetch caps only for the liquid survivors, then filter) --------
    min_mcap = config.UNIVERSE_MIN_MCAP_M * 1e6
    marketcap.ensure([s for s, *_ in liquid])           # threaded yfinance fetch for the shortlist
    keep = []
    for sym, vol_k, price, dvol_m, adr, curated in liquid:
        mc = marketcap.get(sym)
        mc_m = round(mc / 1e6, 1) if mc else 0.0
        if curated or mc is None or mc >= min_mcap:   # keep ETFs/futures + unknown-cap; drop only known-small
            keep.append((sym, vol_k, price, dvol_m, adr, mc_m))
    uni.ACTIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(uni.ACTIVE_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["yf_symbol", "avg_vol_k", "price", "dvol_m", "adr_pct", "mcap_m"])
        w.writerows(sorted(keep))
    print(f"active universe: {len(keep)} of {len(syms)} tickers pass "
          f"(avg_vol>={config.UNIVERSE_MIN_AVG_VOL_K}K, price>=${config.UNIVERSE_MIN_PRICE}, "
          f"$vol>=${config.UNIVERSE_MIN_DVOL_M}M, adr>={config.UNIVERSE_MIN_ADR_PCT}%, "
          f"mcap>=${config.UNIVERSE_MIN_MCAP_M}M) -> {uni.ACTIVE_FILE.name}")
    return len(keep)


if __name__ == "__main__":
    build()
