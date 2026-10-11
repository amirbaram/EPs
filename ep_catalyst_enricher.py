"""Institutional Historical EP Catalyst & SEC EDGAR Intelligence Engine.

Provides zero-subscription fundamental catalyst attachment, official SEC EDGAR 8-K / 6-K / 10-Q filing
discovery, and earnings calendar verification for all historical EP events.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

import config

EDGAR_DIR = config.DATA_DIR / "edgar_cache"
HIST_CATALYSTS_PATH = config.DATA_DIR / "historical_ep_catalysts.json"
CURATED_CATALYSTS_PATH = config.DATA_DIR / "ep_catalysts.json"
SEC_FILINGS_PATH = config.DATA_DIR / "cache" / "ep_sec_filings.json"

_UA = {"User-Agent": "Institutional-EP-Research amir.baram@gmail.com"}
_CIK_CACHE: dict | None = None
_HIST_CACHE: dict | None = None
_CURATED_CACHE: dict | None = None

ITEM_DESCRIPTIONS = {
    "1.01": "Material Definitive Agreement (Contract / Partnership)",
    "1.02": "Termination of Material Agreement",
    "2.01": "Completion of Acquisition or Disposition of Assets (M&A)",
    "2.02": "Results of Operations and Financial Condition (Quarterly Earnings)",
    "2.03": "Creation of Direct Financial Obligation",
    "2.04": "Acceleration of Direct Financial Obligation",
    "3.01": "Notice of Delisting or Failure to Satisfy Listing Rule",
    "3.02": "Unregistered Sales of Equity Securities (Private Placement / Financing)",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financial Statements",
    "5.01": "Changes in Control of Registrant",
    "5.02": "Departure / Election of Directors or Principal Officers (CEO Change)",
    "5.03": "Amendments to Articles of Incorporation or Bylaws",
    "7.01": "Regulation FD Disclosure (Investor Presentation / Guidance)",
    "8.01": "Other Material Events (Clinical Trial Data, FDA Action, Lawsuits)",
    "9.01": "Financial Statements and Exhibits (Press Release Attachment)",
}


def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    temp_path.replace(path)


def get_cik_for_symbol(sym: str) -> str | None:
    """Resolves SEC CIK 10-digit zero-padded string for a stock symbol."""
    global _CIK_CACHE
    if _CIK_CACHE is None:
        EDGAR_DIR.mkdir(parents=True, exist_ok=True)
        cik_file = EDGAR_DIR / "company_tickers.json"
        if cik_file.exists():
            _CIK_CACHE = _load_json(cik_file)
        else:
            try:
                req = urllib.request.Request("https://www.sec.gov/files/company_tickers.json", headers=_UA)
                with urllib.request.urlopen(req, timeout=20) as r:
                    _CIK_CACHE = json.load(r)
                _save_json(cik_file, _CIK_CACHE)
            except Exception as e:
                print(f"⚠️ Warning: Could not fetch company_tickers.json from SEC: {e}")
                _CIK_CACHE = {}

    sym_clean = sym.upper().strip()
    if _CIK_CACHE:
        for v in _CIK_CACHE.values():
            if v.get("ticker", "").upper() == sym_clean:
                return f"{int(v['cik_str']):010d}"
    return None


def fetch_sec_submissions(sym: str, cik: str | None = None) -> dict | None:
    """Fetches and caches official SEC submissions JSON for a symbol."""
    EDGAR_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = EDGAR_DIR / f"{sym.upper()}_submissions.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    if not cik:
        cik = get_cik_for_symbol(sym)
    if not cik:
        return None

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.load(r)
        _save_json(cache_file, data)
        time.sleep(0.12)  # Fair-use pacing for SEC API
        return data
    except Exception as e:
        return None


def extract_filings_around_date(sym: str, ep_date_str: str, days_back: int = 3, days_forward: int = 1) -> list[dict]:
    """Finds official SEC filings (8-K, 6-K, 10-Q, 10-K, S-3, etc.) filed near the EP date."""
    cik = get_cik_for_symbol(sym)
    if not cik:
        return []

    submissions = fetch_sec_submissions(sym, cik)
    if not submissions or "filings" not in submissions or "recent" not in submissions["filings"]:
        return []

    rec = submissions["filings"]["recent"]
    ep_dt = pd.to_datetime(ep_date_str).date()
    start_dt = (ep_dt - pd.Timedelta(days=days_back))
    end_dt = (ep_dt + pd.Timedelta(days=days_forward))

    matched_filings = []
    num_filings = len(rec.get("form", []))

    for i in range(num_filings):
        f_date_str = rec["filingDate"][i]
        try:
            f_dt = pd.to_datetime(f_date_str).date()
        except Exception:
            continue

        if start_dt <= f_dt <= end_dt:
            form = rec["form"][i]
            # Prioritize material disclosure forms
            if form in ("8-K", "8-K/A", "6-K", "6-K/A", "10-Q", "10-K", "S-3ASR", "S-3", "424B5", "424B3"):
                items_raw = rec.get("items", [""] * num_filings)[i] or ""
                doc_name = rec.get("primaryDocument", [""] * num_filings)[i]
                acc_num = rec.get("accessionNumber", [""] * num_filings)[i]
                acc_clean = acc_num.replace("-", "")
                cik_int = int(cik)

                url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{doc_name}" if doc_name and acc_num else ""
                
                # Human readable item descriptions
                item_labels = []
                for item_code in items_raw.split(","):
                    code = item_code.strip()
                    if code:
                        label = ITEM_DESCRIPTIONS.get(code, f"Item {code}")
                        item_labels.append(label)

                matched_filings.append({
                    "date": f_date_str,
                    "form": form,
                    "items": items_raw,
                    "item_labels": item_labels,
                    "doc": doc_name,
                    "url": url,
                    "accession": acc_num
                })

    return matched_filings


def get_earnings_concurrence(sym: str, ep_date_str: str, window_days: int = 5) -> str | None:
    """Checks yfinance earnings calendar to see if EP was on or within window_days of earnings."""
    EDGAR_DIR.mkdir(parents=True, exist_ok=True)
    cache = EDGAR_DIR / f"{sym.upper()}_earnings.json"
    dates = None
    if cache.exists():
        try:
            with open(cache, "r", encoding="utf-8") as f:
                dates = json.load(f)
        except Exception:
            pass

    if dates is None:
        try:
            import yfinance as yf
            ed = yf.Ticker(sym).get_earnings_dates(limit=100)
            dates = sorted({str(x.date()) for x in ed.index}) if ed is not None and len(ed) else []
            _save_json(cache, dates)
        except Exception:
            return None

    if not dates:
        return None

    t = pd.to_datetime(ep_date_str)
    best_dt = None
    min_dist = 999
    for ds in dates:
        try:
            dd = (t - pd.to_datetime(ds)).days
            if -1 <= dd <= window_days:
                if abs(dd) < min_dist:
                    min_dist = abs(dd)
                    best_dt = ds
        except Exception:
            continue

    return best_dt


def get_clean_company_name(sym: str) -> str:
    """Extracts cleaned core corporate name for symbol to optimize news search precision."""
    get_cik_for_symbol(sym)
    if _CIK_CACHE:
        for v in _CIK_CACHE.values():
            if v.get("ticker", "").upper() == sym.upper():
                title = v.get("title", "")
                title = re.sub(r"(?i)\b(inc|corp|corporation|ltd|limited|co|llc|plc|nv|sa)\b\.?", "", title)
                title = title.replace(",", "").strip()
                return title
    return ""


def get_historical_news(sym: str, ep_date_str: str, company_name: str = "", lookback_days: int = 35) -> list[dict]:
    """Fetches and caches news headlines from Google News RSS for sym / company near ep_date_str.
    
    Strictly verifies that news articles are contemporaneous with the EP event (within 14 days).
    Google News RSS only indexes recent articles (~45-60 days); for historical dates older than
    60 days, it returns empty list to prevent injecting irrelevant modern news.
    """
    try:
        ep_dt = pd.to_datetime(ep_date_str).date()
        today = pd.Timestamp.now().date()
        days_ago = (today - ep_dt).days
    except Exception:
        return []

    # If the episode occurred more than 60 days ago, Google News RSS does NOT have contemporaneous news.
    # Never query it for older dates to avoid polluting historical records with today's news.
    if days_ago > 60:
        return []

    if not company_name:
        company_name = get_clean_company_name(sym)

    EDGAR_DIR.mkdir(parents=True, exist_ok=True)
    cache = EDGAR_DIR / f"{sym.upper()}_{ep_date_str}_news.json"
    if cache.exists():
        try:
            with open(cache, "r", encoding="utf-8") as f:
                cached = json.load(f)
                valid = []
                for a in cached:
                    try:
                        a_dt = pd.to_datetime(a.get("date")).date()
                        if abs((a_dt - ep_dt).days) <= 14:
                            valid.append(a)
                    except Exception:
                        pass
                return valid
        except Exception:
            pass

    query = f"{sym} {company_name}".strip()
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}+when:{lookback_days}d&hl=en-US&gl=US&ceid=US:en"
    req = urllib.request.Request(url, headers=_UA)
    articles = []
    try:
        import xml.etree.ElementTree as ET
        with urllib.request.urlopen(req, timeout=10) as resp:
            tree = ET.fromstring(resp.read())
            for item in tree.findall("./channel/item"):
                title = item.find("title").text if item.find("title") is not None else ""
                pub = item.find("pubDate").text if item.find("pubDate") is not None else ""
                link = item.find("link").text if item.find("link") is not None else ""
                source = item.find("source").text if item.find("source") is not None else ""
                try:
                    pub_obj = pd.to_datetime(pub).date()
                    pub_dt = pub_obj.strftime("%Y-%m-%d")
                except Exception:
                    continue

                # Ensure article was published contemporaneously with the EP event (within 14 days)
                if abs((pub_obj - ep_dt).days) > 14:
                    continue

                articles.append({
                    "title": title,
                    "date": pub_dt,
                    "pub_raw": pub,
                    "source": source,
                    "url": link
                })
        _save_json(cache, articles)
    except Exception:
        pass
    return articles


def get_catalyst_dossier(sym: str, ep_date_str: str, sector: str = "", theme: str = "") -> dict:
    """Returns complete institutional catalyst profile, SEC filings, earnings concurrence, and news coverage.
    
    Checks pre-mined historical catalysts cache first; falls back dynamically to SEC EDGAR, Google News, and Yahoo.
    """
    global _HIST_CACHE, _CURATED_CACHE
    if _HIST_CACHE is None:
        _HIST_CACHE = _load_json(HIST_CATALYSTS_PATH)
    if _CURATED_CACHE is None:
        _CURATED_CACHE = _load_json(CURATED_CATALYSTS_PATH)

    cache_key = f"{sym.upper()}_{ep_date_str}"
    curated = _CURATED_CACHE.get(sym.upper(), {})
    ep_dt = pd.to_datetime(ep_date_str).date()

    # If curated exists, always prioritize it over a stale generic cache entry
    if cache_key in _HIST_CACHE and not curated:
        res = dict(_HIST_CACHE[cache_key])
        raw_news = res.get("news_articles")
        if not raw_news:
            raw_news = get_historical_news(sym, ep_date_str, lookback_days=35)[:6]
        # Sanitize any non-contemporaneous articles from cache
        clean_news = []
        for a in (raw_news or []):
            try:
                a_dt = pd.to_datetime(a.get("date")).date()
                if abs((a_dt - ep_dt).days) <= 14:
                    clean_news.append(a)
            except Exception:
                pass
        res["news_articles"] = clean_news
        return res

    # Check curated cache by symbol
    if curated:
        news_items = get_historical_news(sym, ep_date_str, lookback_days=35)
        clean_news = [a for a in news_items if abs((pd.to_datetime(a.get("date")).date() - ep_dt).days) <= 14]
        res = {
            "symbol": sym.upper(),
            "date": ep_date_str,
            "category": curated.get("category", f"{sector or 'Corporate'} Structural Inflection"),
            "headline": curated.get("headline", f"{sym.upper()} Institutional Catalyst Repricing"),
            "tier": curated.get("quality_tier", "Tier-1 Institutional Repricing"),
            "analysis": curated.get("analysis", f"{sym.upper()} experienced an episodic surge triggering structural repricing."),
            "news_summary": curated.get("news_summary", f"{sym.upper()} news and corporate developments around {ep_date_str}."),
            "sec_filings": extract_filings_around_date(sym, ep_date_str),
            "news_articles": clean_news[:6],
            "earnings_date": get_earnings_concurrence(sym, ep_date_str),
            "has_earnings_concurrence": get_earnings_concurrence(sym, ep_date_str) is not None
        }
        _HIST_CACHE[cache_key] = res
        _save_json(HIST_CATALYSTS_PATH, _HIST_CACHE)
        return res

    # Dynamic extraction via SEC EDGAR, Google News, and Earnings Calendar
    filings = extract_filings_around_date(sym, ep_date_str)
    earnings_dt = get_earnings_concurrence(sym, ep_date_str)
    news_items = get_historical_news(sym, ep_date_str, lookback_days=35)
    clean_news = [a for a in news_items if abs((pd.to_datetime(a.get("date")).date() - ep_dt).days) <= 14]


    has_earnings = bool(earnings_dt)
    category = "Corporate Institutional Repricing"
    headline = f"{sym.upper()} Institutional EP Volume Repricing"
    tier = "Tier-1 Institutional Repricing"
    analysis = f"{sym.upper()} staged a massive episodic volume breakout triggering institutional accumulation."
    news_summary = f"{sym.upper()} corporate announcement and volume surge on {ep_date_str}."

    has_8k_202 = any("2.02" in f.get("items", "") for f in filings if "8-K" in f.get("form", ""))
    has_8k_101 = any("1.01" in f.get("items", "") for f in filings if "8-K" in f.get("form", ""))
    has_8k_801 = any("8.01" in f.get("items", "") for f in filings if "8-K" in f.get("form", ""))
    has_8k_201 = any("2.01" in f.get("items", "") for f in filings if "8-K" in f.get("form", ""))
    has_8k_701 = any("7.01" in f.get("items", "") for f in filings if "8-K" in f.get("form", ""))
    has_8k_302 = any("3.02" in f.get("items", "") for f in filings if "8-K" in f.get("form", ""))

    sec_lower = (sector or "").lower()
    thm_lower = (theme or "").lower()
    is_biotech = "bio" in sec_lower or "health" in sec_lower or "bio" in thm_lower or "pharma" in thm_lower

    if has_8k_202 or has_earnings:
        category = "Quarterly Earnings & Guidance Release"
        headline = f"{sym.upper()} Reports Quarterly Financial Results & Outlook"
        tier = "Tier-1 Corporate Fundamentals"
        ed_str = f" on {earnings_dt}" if earnings_dt else ""
        analysis = f"{sym.upper()} filed official quarterly earnings disclosure with the SEC (Form 8-K Item 2.02{ed_str}). Significant institutional repricing and PEAD drift igniter."
        news_summary = f"{sym.upper()} reported quarterly earnings and forward guidance."
    elif has_8k_801 and is_biotech:
        category = "Biotechnology / Clinical Trial & FDA Readout"
        headline = f"{sym.upper()} Phase Clinical Trial / FDA Material Disclosure"
        tier = "Tier-1 Transformational Medical Breakthrough"
        analysis = f"{sym.upper()} filed official Form 8-K Item 8.01 with material clinical development, trial efficacy data, or regulatory clearance."
        news_summary = f"{sym.upper()} released material clinical trial data or regulatory readout."
    elif has_8k_101:
        category = "Commercial Partnership & Definitive Agreement"
        headline = f"{sym.upper()} Enters Material Commercial Agreement"
        tier = "Tier-1 Institutional Expansion"
        analysis = f"{sym.upper()} executed a material definitive agreement (Form 8-K Item 1.01) with enterprise customer or commercial partner."
        news_summary = f"{sym.upper()} announced major material definitive agreement."
    elif has_8k_201:
        category = "Mergers, Acquisitions & Asset Disposition"
        headline = f"{sym.upper()} Strategic Acquisition / Asset Transaction"
        tier = "Tier-1 Structural M&A"
        analysis = f"{sym.upper()} announced completion of material acquisition or asset disposition (Form 8-K Item 2.01)."
        news_summary = f"{sym.upper()} finalized strategic corporate acquisition."
    elif has_8k_701:
        category = "Regulation FD Investor Day / Guidance Revision"
        headline = f"{sym.upper()} Issues Updated Corporate Guidance"
        tier = "Tier-2 Institutional Repricing"
        analysis = f"{sym.upper()} disclosed Regulation FD investor day metrics and raised financial outlook (Form 8-K Item 7.01)."
        news_summary = f"{sym.upper()} presented updated investor outlook."
    elif has_8k_302:
        category = "Equity Financing & Capital Allocation"
        headline = f"{sym.upper()} Private Placement / Capital Expansion"
        tier = "Tier-2 Growth Financing"
        analysis = f"{sym.upper()} filed unregistered sales of equity securities (Form 8-K Item 3.02) securing runway."
        news_summary = f"{sym.upper()} announced financing transaction."
    elif filings:
        f0 = filings[0]
        category = f"SEC Form {f0['form']} Material Disclosure"
        headline = f"{sym.upper()} Official Form {f0['form']} Regulatory Filing"
        tier = "Tier-1 Regulatory Filing"
        items_str = f" ({', '.join(f0.get('item_labels', [])[:2])})" if f0.get("item_labels") else ""
        analysis = f"{sym.upper()} filed official Form {f0['form']}{items_str} with the SEC on {f0['date']}."
        news_summary = f"{sym.upper()} regulatory filing submitted on {f0['date']}."
    elif clean_news:
        top_n = clean_news[0]
        headline = top_n["title"]
        if is_biotech:
            category = "Biotechnology / Clinical Trial Breakthrough"
            tier = "Tier-1 Transformational Medical Breakthrough"
        else:
            category = f"{sector or 'Corporate'} Media / Press Release Catalyst"
            tier = "Tier-2 Institutional Catalyst"
        analysis = f"{sym.upper()} high-volume episodic surge accompanied by press and media reports: {top_n['title']} ({top_n['date']})."
        news_summary = top_n["title"]
    else:
        category = f"{sector or 'Corporate'} Institutional Repricing"
        headline = f"{sym.upper()} Episodic Pivot Volume & Price Repricing"
        tier = "Tier-1 Institutional Repricing"
        analysis = f"{sym.upper()} experienced massive institutional turnover triggering structural trend inflection."
        news_summary = f"{sym.upper()} volume repricing on {ep_date_str}."

    result = {
        "symbol": sym.upper(),
        "date": ep_date_str,
        "category": category,
        "headline": headline,
        "tier": tier,
        "analysis": analysis,
        "news_summary": news_summary,
        "sec_filings": filings,
        "news_articles": clean_news[:6],
        "earnings_date": earnings_dt,
        "has_earnings_concurrence": has_earnings
    }

    # Save to memory and disk cache
    _HIST_CACHE[cache_key] = result
    _save_json(HIST_CATALYSTS_PATH, _HIST_CACHE)

    return result
