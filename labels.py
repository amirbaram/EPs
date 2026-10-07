"""Sector + theme labels for tickers, loaded from data/ CSVs. Sector = one industry per ticker
(from a universe snapshot); themes = many-to-many curated tags (seeded from ETF holdings, expanded
by hand). Keyed by yf_symbol. Mirrors marketcap.py: lazy in-memory load, O(1) lookups.

    labels.sector("NVDA")             -> "Semiconductors"
    labels.themes("NVDA")             -> ["Artificial Intelligence", "Semiconductors", ...]
    labels.tickers_in_theme("Space")  -> {"LUNR", "RKLB", ...}    # for future theme-scans
    labels.all_themes()               -> sorted theme names        # for a future filter dropdown
"""
from __future__ import annotations

import csv

import config

SECTORS_FILE = config.DATA_DIR / "sectors.csv"
THEMES_FILE = config.DATA_DIR / "themes.csv"

_SECTOR: dict[str, str] | None = None
_THEMES_BY_SYM: dict[str, list[str]] | None = None
_SYMS_BY_THEME: dict[str, set[str]] | None = None


def _key(t: str) -> str:
    return t.strip().upper().replace(".", "-")     # -> yf_symbol form, matching the scanner


def load() -> None:
    """Lazy-read both CSVs into the in-memory indexes (once). Missing files -> empty."""
    global _SECTOR, _THEMES_BY_SYM, _SYMS_BY_THEME
    if _SECTOR is not None:
        return
    _SECTOR = {}
    try:
        for row in csv.DictReader(SECTORS_FILE.open()):
            sym, sec = _key(row.get("ticker", "")), (row.get("sector") or "").strip()
            if sym and sec:
                _SECTOR[sym] = sec
    except FileNotFoundError:
        pass
    _THEMES_BY_SYM, _SYMS_BY_THEME = {}, {}
    try:
        for row in csv.DictReader(THEMES_FILE.open()):
            sym, th = _key(row.get("ticker", "")), (row.get("theme") or "").strip()
            if not sym or not th:
                continue
            _SYMS_BY_THEME.setdefault(th, set()).add(sym)
            lst = _THEMES_BY_SYM.setdefault(sym, [])
            if th not in lst:
                lst.append(th)
    except FileNotFoundError:
        pass
    for lst in _THEMES_BY_SYM.values():
        lst.sort()                                  # stable display order


def reload() -> None:
    """Drop the caches so the next lookup re-reads the CSVs (after an edit / re-seed)."""
    global _SECTOR
    _SECTOR = None
    load()


def sig() -> str:
    """Cheap signature of the label CSVs (mtime+size) for cache keys — changes whenever sectors/themes are
    edited, so theme-membership-dependent caches (e.g. the group-score leaderboard) recompute after an edit
    instead of serving stale scores. Missing file -> '-'."""
    parts = []
    for f in (SECTORS_FILE, THEMES_FILE):
        try:
            st = f.stat()
            parts.append(f"{int(st.st_mtime)}:{st.st_size}")
        except FileNotFoundError:
            parts.append("-")
    return "|".join(parts)


def sector(sym: str):
    load()
    return _SECTOR.get(_key(sym))


def themes(sym: str) -> list[str]:
    load()
    return _THEMES_BY_SYM.get(_key(sym), [])


def tickers_in_theme(theme: str) -> set[str]:
    load()
    return set(_SYMS_BY_THEME.get(theme, set()))


def all_themes() -> list[str]:
    load()
    return sorted(_SYMS_BY_THEME)


# proxy ETF per theme (first entry of build_labels.THEME_ETFS) — chartable stand-in for the
# basket on the rotation quadrant / map cards
THEME_ETF = {
    "Artificial Intelligence": "AIQ", "Robotics": "BOTZ", "Semiconductors": "SMH",
    "Cloud": "CLOU", "Internet": "FDN", "Online Retail": "IBUY", "Memory": "DRAM",
    "Quantum Computing": "QTUM", "Disruptive Innovation": "ARKK", "Biotech": "BBH",
    "Genomics": "ARKG", "Pharma": "PPH", "Medical Devices": "IHI",
    "Crypto & Blockchain": "BLOK", "Bitcoin Miners": "WGMI", "Gaming & eSports": "ESPO",
    "EV & Autonomous": "DRIV", "Lithium & Batteries": "LIT", "Clean Energy": "PBW",
    "Solar": "TAN", "Uranium & Nuclear": "NLR", "Aerospace & Defense": "ITA",
    "Gold Miners": "GDX", "Copper Miners": "COPX", "Rare Earths": "REMX",
    "Metals & Mining": "XME", "Agriculture": "MOO", "Oil Services": "OIH",
    "Oil & Gas E&P": "XOP", "Energy": "IXC", "Homebuilders": "ITB", "Airlines": "JETS",
    "Transportation": "IYT", "Banks": "KBE", "Cannabis": "MSOS", "Cybersecurity": "CIBR",
}


def theme_etf(theme: str) -> str | None:
    return THEME_ETF.get(theme)


# SPDR sector membership: sectors.csv stores yfinance INDUSTRY names, so the 11 SPDR sectors
# are resolved by industry keyword (first match wins; substring, case-insensitive)
SPDR_INDUSTRY = {
    "XLK": ["software", "semiconductor", "information technology", "computer hardware",
            "consumer electronics", "electronic component", "electronics & computer",
            "scientific & technical", "communication equipment", "solar"],
    "XLC": ["internet content", "telecom", "entertainment", "gaming & multimedia",
            "advertising", "broadcasting", "publishing"],
    "XLY": ["internet retail", "specialty retail", "auto", "restaurant", "apparel", "footwear",
            "residential construction", "travel", "resorts", "lodging", "leisure", "gambling",
            "recreational", "furnishing", "department store", "home improvement", "luxury",
            "personal services", "textile", "packaging"],
    "XLP": ["grocery", "discount store", "packaged food", "beverage", "confectioner",
            "farm products", "household & personal", "food distribution", "tobacco",
            "education"],
    "XLE": ["oil & gas", "oil services", "uranium", "coal", "energy"],
    "XLF": ["bank", "insurance", "capital markets", "asset management", "credit services",
            "financial", "mortgage", "shell companies"],
    "XLV": ["biotech", "drug manufacturer", "medical", "diagnostics", "health", "pharmaceutical"],
    "XLI": ["aerospace", "airline", "railroad", "trucking", "farm & heavy", "engineering",
            "building products", "electrical equipment", "industrial", "freight", "marine",
            "rental & leasing", "security & protection", "business services", "staffing",
            "waste", "conglomerate", "consulting", "metal fabrication", "pollution",
            "tools & accessories", "airports", "infrastructure"],
    "XLB": ["chemical", "gold", "silver", "copper", "aluminum", "steel", "agricultural inputs",
            "building materials", "lumber", "metals & mining", "precious metals", "paper"],
    "XLRE": ["reit", "real estate"],
    "XLU": ["utilities", "utility"],
}


def spdr_of(industry: str | None) -> str | None:
    if not industry:
        return None
    low = industry.lower()
    for etf, kws in SPDR_INDUSTRY.items():
        if any(k in low for k in kws):
            return etf
    return None


def spdr_members(etf: str, universe) -> list[str]:
    """Symbols in `universe` (iterable) whose industry maps to the SPDR sector `etf`."""
    load()
    return [s for s in universe if spdr_of(_SECTOR.get(_key(s)))== etf]
