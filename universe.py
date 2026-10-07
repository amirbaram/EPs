"""Build the full US common-stock universe from NASDAQ Trader's symbol directories.

Free, canonical, covers NASDAQ + NYSE + AMEX (~10k symbols). We deliberately do NOT
filter by liquidity here — quiet tickers can still form setups worth seeing.
"""
import io
import urllib.request

import pandas as pd

import config

NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

# Liquid sector / industry / thematic ETFs kept in the (otherwise common-stock-only) universe —
# useful for sector rotation / the Uptrend-Downtrend setups. Tagged etf=True so they're filterable.
CURATED_ETFS = {
    # broad market
    "SPY", "QQQ", "IWM", "DIA", "MDY",
    # SPDR sectors (11)
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
    # industry / thematic
    "SMH", "SOXX", "XBI", "IBB", "XHB", "ITB", "IGV", "XOP", "OIH", "XME", "XRT",
    "KRE", "KBE", "KWEB", "FXI", "ARKK", "ARKG", "TAN", "URA", "URNM", "GDX", "GDXJ",
    "SLX", "JETS", "ITA", "XAR", "PAVE", "IYT", "SKYY", "BOTZ", "LIT", "REMX", "HACK",
    "GLD", "SLV",
    # theme proxy ETFs (labels.THEME_ETF) — every theme dot/card must be chartable
    "AIQ", "BBH", "BLOK", "CIBR", "CLOU", "COPX", "DRAM", "DRIV", "ESPO", "FDN",
    "IBUY", "IHI", "IXC", "MOO", "MSOS", "NLR", "PBW", "PPH", "QTUM", "WGMI",
    "SOCL", "SIL", "FFTY",     # theme-tracker parity (Social Media / Silver Miners / Growth)
}

# Continuous futures (yfinance `=F`) for macro / index / commodity context. Kept in the universe
# (always-active), used by the multi-timeframe awareness engine. 24h instruments — no RTH session.
FUTURES = {   # yf_symbol: (name, exchange)
    "NQ=F": ("Nasdaq-100 E-mini Futures", "CME"),
    "ES=F": ("S&P 500 E-mini Futures", "CME"),
    "RTY=F": ("Russell 2000 E-mini Futures", "CME"),
    "YM=F": ("Dow E-mini Futures", "CBOT"),
    "GC=F": ("Gold Futures", "COMEX"),
    "SI=F": ("Silver Futures", "COMEX"),
    "HG=F": ("Copper Futures", "COMEX"),
    "PL=F": ("Platinum Futures", "NYMEX"),
    "ZW=F": ("Wheat Futures", "CBOT"),
    "ZC=F": ("Corn Futures", "CBOT"),
    "ZS=F": ("Soybean Futures", "CBOT"),
    "ZL=F": ("Soybean Oil Futures", "CBOT"),
    "CL=F": ("Crude Oil WTI Futures", "NYMEX"),
    "NG=F": ("Natural Gas Futures", "NYMEX"),
    "BTC=F": ("Bitcoin Futures", "CME"),
}
FUTURES_SYMBOLS = set(FUTURES)


def is_future(sym: str) -> bool:
    return sym in FUTURES_SYMBOLS or str(sym).endswith("=F")


def _fetch(url: str) -> pd.DataFrame:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
    # last line is a "File Creation Time" footer
    body = "\n".join(line for line in raw.splitlines() if not line.startswith("File Creation"))
    # keep_default_na=False: ticker "NA" (Nano Labs) must stay a string, not become NaN
    return pd.read_csv(io.StringIO(body), sep="|", keep_default_na=False)


def build_universe(include_etfs: bool = False) -> pd.DataFrame:
    nq = _fetch(NASDAQ_URL)
    ot = _fetch(OTHER_URL)

    nq = nq[nq["Test Issue"] == "N"]
    ot = ot[ot["Test Issue"] == "N"]

    nq_df = pd.DataFrame({
        "symbol": nq["Symbol"].astype(str),
        "name": nq["Security Name"].astype(str),
        "exchange": "NASDAQ",
        "etf": nq["ETF"].astype(str).eq("Y"),
    })
    exch_map = {"N": "NYSE", "A": "AMEX", "P": "NYSEARCA", "Z": "BATS", "V": "IEXG"}
    ot_df = pd.DataFrame({
        "symbol": ot["ACT Symbol"].astype(str),
        "name": ot["Security Name"].astype(str),
        "exchange": ot["Exchange"].astype(str).map(exch_map).fillna("NYSE"),
        "etf": ot["ETF"].astype(str).eq("Y"),
    })
    df = pd.concat([nq_df, ot_df], ignore_index=True)

    if not include_etfs:                         # drop ETFs EXCEPT the curated liquid set
        df = df[(~df["etf"]) | df["symbol"].isin(CURATED_ETFS)]

    # drop preferreds/warrants/units/rights — symbols containing $ or special suffixes,
    # plus names that say so (catches NYSE ".U"/".WS" styles after conversion too).
    # NOTE: do NOT exclude on "depositary" — that drops common-equity ADSs (ARM, TSM,
    # BABA, ...) which are legit screening targets. Preferred *depositary* shares are
    # still caught by "preferred|preference|%", which always appear in their names.
    df = df[~df["symbol"].str.contains(r"[\$\^~]", regex=True)]
    bad_name = df["name"].str.contains(
        r"warrant|rights? |units?(?:,| )|preferred|preference|notes? due|%",
        case=False, regex=True, na=False)
    df = df[~bad_name]

    # yfinance wants BRK-B style, directories give BRK.B / BRK$B
    df["yf_symbol"] = df["symbol"].str.replace(".", "-", regex=False)

    fut = pd.DataFrame([{"symbol": s, "name": nm, "exchange": ex, "etf": False, "yf_symbol": s}
                        for s, (nm, ex) in FUTURES.items()])
    df = pd.concat([df, fut], ignore_index=True)
    df = df.drop_duplicates("yf_symbol").sort_values("symbol").reset_index(drop=True)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.UNIVERSE_FILE, index=False)
    print(f"Universe: {len(df)} symbols  ->  {config.UNIVERSE_FILE}")
    return df


def load_universe() -> pd.DataFrame:
    if not config.UNIVERSE_FILE.exists():
        return build_universe()
    return pd.read_csv(config.UNIVERSE_FILE, keep_default_na=False)


ACTIVE_FILE = config.DATA_DIR / "universe_active.csv"

_EXCH: dict[str, str] | None = None


def exchange_map() -> dict[str, str]:
    """{yf_symbol: exchange} from the full universe snapshot (NYSE / NASDAQ / AMEX / …). Cached; empty if the
    snapshot is missing. Used to split market breadth by listing venue (NASDAQ ≈ tech vs NYSE ≈ broad)."""
    global _EXCH
    if _EXCH is None:
        _EXCH = {}
        try:
            df = pd.read_csv(config.UNIVERSE_FILE, keep_default_na=False)
            if {"yf_symbol", "exchange"} <= set(df.columns):
                _EXCH = dict(zip(df["yf_symbol"].astype(str), df["exchange"].astype(str)))
        except FileNotFoundError:
            pass
    return _EXCH


def active_symbols() -> set[str] | None:
    """The reduced liquid universe (yf_symbols) written by build_active_universe.py, or None if it
    hasn't been built yet (callers then fall back to the full cache — no filtering)."""
    if not ACTIVE_FILE.exists():
        return None
    df = pd.read_csv(ACTIVE_FILE, keep_default_na=False)
    if "yf_symbol" not in df.columns or not len(df):
        return None
    return (set(df["yf_symbol"].astype(str)) | CURATED_ETFS | FUTURES_SYMBOLS) \
        - getattr(config, "UNIVERSE_EXCLUDE", set())   # always keep curated; drop halted/dead


if __name__ == "__main__":
    build_universe()
