"""Market-cap lookup, cached on disk. Market cap isn't in the OHLCV bars or the universe list,
so it needs a yfinance fundamental fetch (`fast_info.market_cap`) — one network call per ticker,
no bulk endpoint. We fetch it only for small custom watchlists (fast), cache to data/marketcap.json,
and the scanner just reads the cache (blank for tickers never fetched).

    marketcap.ensure(["AAPL", "NVDA"])   # fetch any missing + persist, return the full cache
    marketcap.get("AAPL")                 # cached market cap in dollars, or None
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import config

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

CACHE_FILE = config.DATA_DIR / "marketcap.json"
_CACHE: dict[str, float] | None = None      # sym -> market cap in dollars (None = couldn't fetch)


def load() -> dict[str, float]:
    """Lazy-read the on-disk cache once into memory. Missing/corrupt file -> empty."""
    global _CACHE
    if _CACHE is None:
        try:
            _CACHE = json.loads(CACHE_FILE.read_text())
        except (FileNotFoundError, ValueError):
            _CACHE = {}
    return _CACHE


def get(sym: str):
    """Cached market cap (dollars) for a yf_symbol, or None if not fetched / unavailable."""
    return load().get(sym)


def _save() -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(_CACHE, sort_keys=True))


def _fetch_one(sym: str):
    """Market cap in dollars via yfinance fast_info; None on any failure (network/missing)."""
    if yf is None:
        return None
    try:
        fi = yf.Ticker(sym).fast_info
        mc = None
        try:
            mc = fi["marketCap"]            # dict-style access
        except (KeyError, TypeError):
            mc = getattr(fi, "market_cap", None)   # attribute-style fallback
        return float(mc) if mc else None
    except Exception:
        return None


def ensure(symbols, refresh: bool = False) -> dict[str, float]:
    """Make sure `symbols` have market caps in the cache: fetch the missing ones (or all, when
    refresh=True), persist, and return the in-memory cache. Threaded — ~60 tickers in seconds."""
    cache = load()
    todo = list(symbols) if refresh else [s for s in symbols if s not in cache]
    if todo and yf is not None:
        print(f"fetching market cap for {len(todo)} tickers ...")
        with ThreadPoolExecutor(max_workers=8) as ex:
            for sym, mc in zip(todo, ex.map(_fetch_one, todo)):
                cache[sym] = mc
        _save()
        got = sum(1 for s in todo if cache.get(s) is not None)
        print(f"  market cap: {got}/{len(todo)} fetched")
    return cache
