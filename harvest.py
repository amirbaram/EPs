"""Cache the data the theme classifier needs — fetched once, reused forever (incremental).

  .venv/bin/python harvest.py                       # all cached tickers + the theme ETFs
  .venv/bin/python harvest.py --symbols "6M Winners.txt"   # just a watchlist (fast, to validate)
  .venv/bin/python harvest.py --refresh             # re-fetch everything

Writes two caches in data/:
  profiles.json     {sym: {name, sector, industry, summary, fetched}}   (yfinance .info)
  etf_holdings.json {etf: [[sym, pct], ...]}                            (yfinance funds_data top-10)

Both are keyless/free. Re-running only fetches tickers we don't already have, so we never
re-look up data we have. classify.py then runs entirely off these caches (no network).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import config
import datastore

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

PROFILES_FILE = config.DATA_DIR / "profiles.json"
ETF_HOLDINGS_FILE = config.DATA_DIR / "etf_holdings.json"


def _load(path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def _save(path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True))


def _symbols_from_file(path: str) -> list[str]:
    """Reuse the watchlist parser the server uses (EXCHANGE:SYMBOL / RTF / plain)."""
    import serve
    cache = set(datastore.list_symbols())
    return sorted(serve._symbol_filter(path) & cache)


def _fetch_profile(sym: str):
    """dict = profile, {} = fetched-but-no-data (don't retry), None = fetch failed (retry)."""
    if yf is None:
        return None
    for attempt in range(2):                          # one quick retry on a transient failure
        try:
            info = yf.Ticker(sym).info or {}
            break
        except Exception:
            if attempt:
                return None
            time.sleep(0.5)
    summary = info.get("longBusinessSummary") or ""
    sector, industry = info.get("sector") or "", info.get("industry") or ""
    if not (summary or sector or industry):
        return {}                                     # genuinely empty (ETF/delisted) -> don't retry
    return {"name": info.get("shortName") or info.get("longName") or "",
            "sector": sector, "industry": industry, "summary": summary,
            "fetched": dt.date.today().isoformat()}


def harvest_profiles(symbols, refresh: bool = False) -> dict:
    """Fetch+cache yfinance profiles for `symbols`. Incremental: refetches the ones we don't have
    yet AND the ones that previously FAILED (stored None) — so re-running fills gaps as Yahoo's
    rate limit recovers. {}-marked tickers (fetched, no profile) are not retried."""
    cache = _load(PROFILES_FILE)
    todo = list(symbols) if refresh else [s for s in symbols if cache.get(s) is None]
    print(f"profiles: {len(cache)} cached, {len(todo)} to fetch ...")
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, (sym, prof) in enumerate(zip(todo, ex.map(_fetch_profile, todo)), 1):
            cache[sym] = prof if prof is not None else cache.get(sym)
            done += 1
            if i % 250 == 0:
                _save(PROFILES_FILE, cache)
                print(f"  {i}/{len(todo)} (saved)")
    _save(PROFILES_FILE, cache)
    got = sum(1 for s in todo if cache.get(s))
    print(f"profiles: {got}/{len(todo)} fetched, {len(cache)} total -> {PROFILES_FILE.name}")
    return cache


def _fetch_holdings(etf: str) -> list[list]:
    if yf is None:
        return []
    try:
        th = yf.Ticker(etf).funds_data.top_holdings
    except Exception:
        return []
    if th is None or not len(th):
        return []
    pct = "Holding Percent" if "Holding Percent" in th.columns else th.columns[-1]
    out = []
    for sym, row in th.iterrows():
        s = re.sub(r"\..*$", "", str(sym).upper())          # drop foreign suffix (000660.KS -> 000660)
        if re.fullmatch(r"[A-Z][A-Z0-9-]{0,5}", s):
            out.append([s, round(float(row[pct]), 4)])
    return out


def harvest_etf_holdings(etfs, refresh: bool = False) -> dict:
    cache = _load(ETF_HOLDINGS_FILE)
    todo = list(etfs) if refresh else [e for e in etfs if e not in cache]
    if todo:
        print(f"etf holdings: {len(todo)} to fetch ...")
        with ThreadPoolExecutor(max_workers=8) as ex:
            for etf, hold in zip(todo, ex.map(_fetch_holdings, todo)):
                cache[etf] = hold
        _save(ETF_HOLDINGS_FILE, cache)
    print(f"etf holdings: {sum(1 for e in cache if cache[e])}/{len(cache)} non-empty -> {ETF_HOLDINGS_FILE.name}")
    return cache


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbols", help="watchlist file to harvest (default: all cached tickers)")
    ap.add_argument("--refresh", action="store_true", help="re-fetch even cached entries")
    ap.add_argument("--no-etfs", action="store_true", help="skip the ETF-holdings refresh")
    args = ap.parse_args()
    syms = _symbols_from_file(args.symbols) if args.symbols else datastore.list_symbols()
    harvest_profiles(syms, refresh=args.refresh)
    if not args.no_etfs:
        import build_labels
        etfs = sorted({e for ets in build_labels.THEME_ETFS.values() for e in ets})
        harvest_etf_holdings(etfs, refresh=args.refresh)


if __name__ == "__main__":
    main()
