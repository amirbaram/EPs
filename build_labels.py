"""(Re)build the sector + theme label CSVs in data/ from the provided seed files. Re-runnable as
the data grows — hand-curated additions live in data/themes_manual.csv and are always merged, so
re-seeding from the ETF file never clobbers them.

    .venv/bin/python build_labels.py
"""
from __future__ import annotations

import csv
import re
from collections import Counter

import config
import datastore

SECTOR_SRC = config.ROOT / "universe-2026-06-28.csv"        # ticker,name,group,perf...
ETF_SRC = config.ROOT / "Themes ETFs - Sheet1.csv"          # Ticker,Name,Description,Top_Holdings...
SECTORS_OUT = config.DATA_DIR / "sectors.csv"
THEMES_OUT = config.DATA_DIR / "themes.csv"
THEMES_MANUAL = config.DATA_DIR / "themes_manual.csv"       # hand-curated ticker,theme (authoritative)
SECTORS_MANUAL = config.DATA_DIR / "sectors_manual.csv"     # hand-curated ticker,sector for names the
                                                            # universe snapshot misses (RGNX etc.); wins
THEMES_AUTO = config.DATA_DIR / "themes_auto.csv"           # classify.py output (ticker,theme,source)
THEMES_EXCLUDE = config.DATA_DIR / "themes_exclude.csv"     # ticker,theme to remove (wrong auto tags)

# Curated theme -> seed ETF(s). Each theme's membership = the union of its ETFs' (cache-valid) top
# holdings, plus anything in themes_manual.csv. Broad / bond / country / commodity-future ETFs are
# intentionally excluded (AGG, TLT, FXI, GLD, USO, ...). Edit freely and re-run.
THEME_ETFS = {
    "Artificial Intelligence": ["AIQ"],
    "Robotics": ["BOTZ"],
    "Semiconductors": ["SMH", "XSD"],
    "Cloud": ["CLOU", "SKYY"],
    "Internet": ["FDN"],
    "Online Retail": ["IBUY"],
    "Data Centers": ["DRAM", "FOTO"],
    "Quantum Computing": ["QTUM"],
    "Disruptive Innovation": ["ARKK"],
    "Biotech": ["BBH", "XBI"],
    "Genomics": ["ARKG", "IDNA"],
    "Pharma": ["PPH"],
    "Medical Devices": ["IHI"],
    "Crypto & Blockchain": ["BLOK", "BTF"],
    "Bitcoin Miners": ["WGMI"],
    "Gaming & eSports": ["ESPO"],
    "EV & Autonomous": ["DRIV", "IDRV"],
    "Lithium & Batteries": ["LIT"],
    "Clean Energy": ["PBW"],
    "Solar": ["TAN"],
    "Uranium & Nuclear": ["NLR", "URA"],
    "Aerospace & Defense": ["ITA", "XAR"],
    "Gold Miners": ["GDX"],
    "Copper Miners": ["COPX"],
    "Rare Earths": ["REMX"],
    "Metals & Mining": ["XME"],
    "Agriculture": ["MOO"],
    "Oil Services": ["OIH"],
    "Oil & Gas E&P": ["XOP"],
    "Energy": ["IXC"],
    "Homebuilders": ["ITB", "XHB"],
    "Airlines": ["JETS"],
    "Transportation": ["IYT"],
    "Banks": ["KBE"],
    "Cannabis": ["MSOS"],
    "Cybersecurity": ["CIBR", "HACK"],
    # theme-tracker parity additions (Amir 2026-07-05: cover the Market Pulse 31-theme board)
    "Software": ["IGV"],
    "Social Media": ["SOCL"],
    "Retail": ["XRT"],
    "Silver Miners": ["SIL"],
    "China Internet": ["KWEB"],
    "Steel": ["SLX"],
    "Growth Stocks": ["FFTY"],
}


def _key(t: str) -> str:
    return t.strip().upper().replace(".", "-")


def build_sectors(src=SECTOR_SRC) -> int:
    """Reads precise Sectors from data/profiles.json (Yahoo Finance / Finviz parity).
    sectors_manual.csv entries still win."""
    seen, out = set(), []
    try:
        for r in csv.DictReader(open(SECTORS_MANUAL)):
            sym, grp = _key(r["ticker"]), (r.get("sector") or "").strip()
            if sym and grp and sym not in seen:
                out.append((sym, grp)); seen.add(sym)
    except FileNotFoundError:
        pass
        
    import harvest
    profiles = harvest._load(harvest.PROFILES_FILE)
    for sym, p in profiles.items():
        if not p: continue
        sec = p.get("sector")
        if sec and sym not in seen:
            out.append((sym, sec)); seen.add(sym)
            
    # Fallback to the universe CSV if any tickers weren't in profiles
    try:
        for r in csv.DictReader(open(src)):
            sym, grp = _key(r["ticker"]), (r.get("group") or "").strip()
            if sym and grp and sym not in seen:
                out.append((sym, grp)); seen.add(sym)
    except Exception:
        pass
        
    SECTORS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SECTORS_OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["ticker", "sector"]); w.writerows(sorted(out))
    return len(out)


def _etf_holdings(etf_src) -> dict[str, set[str]]:
    """{etf -> set(valid yf_symbols)} — holdings are CSV cols 3+, filtered to real tickers in the
    bar cache (drops foreign names / noise like 'Samsung', 'etc.)', 'AIA Group')."""
    cache = set(datastore.list_symbols())
    out = {}
    for r in csv.reader(open(etf_src)):
        if not r or r[0].strip() in ("", "Ticker"):
            continue
        toks = {_key(c) for c in r[3:] if c.strip()}
        out[r[0].strip().upper()] = {t for t in toks
                                     if re.fullmatch(r"[A-Z][A-Z0-9-]{0,5}", t) and t in cache}
    return out


def _holdings_map(etf_src=ETF_SRC) -> dict[str, set[str]]:
    """{etf -> set(yf_symbols)} — prefer the harvested top-10 cache (fresh, uniform), else the
    original seed sheet's top holdings."""
    import harvest
    h = harvest._load(harvest.ETF_HOLDINGS_FILE)
    if h:
        return {etf: {_key(s) for s, _pct in rows} for etf, rows in h.items()}
    return _etf_holdings(etf_src)


def _merge_csv(path, pairs: dict, default_src: str) -> None:
    """Overlay a ticker,theme[,source] csv onto `pairs` (overwrites the source label -> later
    layers win: etf < auto < manual)."""
    if not path.exists():
        return
    for r in csv.DictReader(path.open()):
        sym, th = _key(r.get("ticker", "")), (r.get("theme") or "").strip()
        if sym and th:
            pairs[(sym, th)] = (r.get("source") or default_src)


def build_themes(etf_src=ETF_SRC):
    """data/themes.csv = (ETF holdings ∪ themes_auto ∪ themes_manual) − themes_exclude.
    Precedence for the source label: manual > auto > etf. Membership is the union of all."""
    pairs: dict[tuple[str, str], str] = {}
    holdings = _holdings_map(etf_src)
    for theme, etfs in THEME_ETFS.items():           # 1. ETF layer (lowest precedence)
        for etf in etfs:
            for sym in holdings.get(etf, ()):
                pairs[(sym, theme)] = f"etf:{etf}"
    _merge_csv(THEMES_AUTO, pairs, "auto")           # 2. classifier output
    _merge_csv(THEMES_MANUAL, pairs, "manual")       # 3. hand-curated (authoritative)
    if THEMES_EXCLUDE.exists():                       # 4. surgical removals of wrong tags
        for r in csv.DictReader(THEMES_EXCLUDE.open()):
            pairs.pop((_key(r.get("ticker", "")), (r.get("theme") or "").strip()), None)
    rows = sorted((s, t, src) for (s, t), src in pairs.items())
    THEMES_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(THEMES_OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["ticker", "theme", "source"]); w.writerows(rows)
    return rows


def main() -> None:
    ns = build_sectors()
    rows = build_themes()
    cnt = Counter(t for _, t, _ in rows)
    tagged = len({s for s, _, _ in rows})
    print(f"sectors.csv: {ns} tickers")
    print(f"themes.csv:  {len(rows)} (ticker,theme) pairs across {len(cnt)} themes, {tagged} distinct tickers")
    for t in sorted(cnt):
        print(f"  {t:26} {cnt[t]:3}")


if __name__ == "__main__":
    main()
