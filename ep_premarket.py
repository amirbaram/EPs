"""EP Premarket Scanner & Early Intraday Radar.

Scans in-play sectors, emerging themes, and tracked EP watchlists using 15m yfinance
premarket feeds, SEC EDGAR 8-K filings, and thematic momentum scoring.
Detects potential Elite EPs forming BEFORE regular market close and generates
early tactical entry game plans (15m ORB, VWAP Reclaim).
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

import config
import datastore
import ep_news
import thematic_engine

PREMARKET_CACHE_PATH = config.DATA_DIR / "cache" / "premarket_eps.json"
PREMARKET_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_in_play_candidates() -> List[str]:
    """Gathers 80-120 high-priority symbols across top emerging themes, sectors, and active EPs."""
    engine = thematic_engine.get_thematic_engine()
    summary = engine.get_emerging_summary()
    candidates = set()

    # Top 8 emerging themes (top 10 symbols each)
    for t in summary.get("emerging_themes", [])[:8]:
        thm_name = t.get("name")
        if thm_name:
            stocks = engine.get_group_stocks(thm_name, "theme").get("stocks", [])
            for s in stocks[:10]:
                candidates.add(s["symbol"])

    # Top 4 emerging sectors (top 8 symbols each)
    for sc in summary.get("emerging_sectors", [])[:4]:
        sec_name = sc.get("name")
        if sec_name:
            stocks = engine.get_group_stocks(sec_name, "sector").get("stocks", [])
            for s in stocks[:8]:
                candidates.add(s["symbol"])

    # Add active tracked EPs from cache if available
    tracker_cache = config.DATA_DIR / "cache" / "tracker_eps.json"
    if tracker_cache.exists():
        try:
            with open(tracker_cache, "r") as f:
                t_data = json.load(f)
                for item in t_data:
                    candidates.add(item["symbol"])
        except Exception:
            pass

    # High-profile market leaders & AI/Biotech pivots
    core_anchors = ["SNPS", "IBRX", "NVDA", "CRWD", "AAPL", "MSFT", "AMZN", "GOOGL", "AMD", "ARM", "CCL", "TSLA", "PLTR", "SMCI"]
    for c in core_anchors:
        candidates.add(c)

    return sorted(list(candidates))


def scan_premarket(symbols: Optional[List[str]] = None, force_refresh: bool = False) -> Dict[str, Any]:
    """Pulls 15m intraday bars with premarket enabled, calculates gap %, volume pacing, and SEC filings."""
    if not force_refresh and PREMARKET_CACHE_PATH.exists():
        try:
            # Check cache freshness (valid for 5 minutes during premarket)
            mtime = PREMARKET_CACHE_PATH.stat().st_mtime
            if (dt.datetime.now().timestamp() - mtime) < 300:
                with open(PREMARKET_CACHE_PATH, "r") as f:
                    return json.load(f)
        except Exception:
            pass

    if symbols is None:
        symbols = get_in_play_candidates()

    engine = thematic_engine.get_thematic_engine()
    summary = engine.get_emerging_summary()
    now_utc = dt.datetime.now(dt.timezone.utc)
    today_str = now_utc.strftime("%Y-%m-%d")

    # Exclude known delisted or defunct symbols
    symbols = [s for s in symbols if s not in {"INFN", "CSIQ_OLD"}]

    print(f"Premarket Radar: Downloading 15m bars for {len(symbols)} candidate symbols...")
    ticker_str = " ".join(symbols)
    try:
        df = yf.download(ticker_str, period="2d", interval="15m", prepost=True, progress=False)
    except Exception as e:
        print(f"Error fetching yfinance premarket bars: {e}")
        df = pd.DataFrame()

    potential_elite_eps = []
    overnight_gappers = []
    other_movers = []

    for sym in symbols:
        raw_b = datastore.load_bars(sym)
        if raw_b is None or len(raw_b) < 20:
            continue

        prior_close = float(raw_b["close"].iloc[-1])
        prior_date = raw_b.index[-1].strftime("%Y-%m-%d")
        avg_20d_vol = float(raw_b["volume"].tail(20).mean())
        if avg_20d_vol <= 0:
            avg_20d_vol = 100000.0

        close_col = ("Close", sym) if ("Close", sym) in df.columns else None
        vol_col = ("Volume", sym) if ("Volume", sym) in df.columns else None
        high_col = ("High", sym) if ("High", sym) in df.columns else None
        low_col = ("Low", sym) if ("Low", sym) in df.columns else None

        if close_col is None or vol_col is None:
            continue

        sub = df[[close_col, vol_col]].dropna()
        if len(sub) == 0:
            continue

        curr_px = float(sub[close_col].iloc[-1])
        latest_ts = sub.index[-1]
        latest_date_str = latest_ts.strftime("%Y-%m-%d")

        # Today's bars
        today_sub = sub[sub.index.strftime("%Y-%m-%d") == latest_date_str]
        pm_vol = int(today_sub[vol_col].sum()) if len(today_sub) else 0

        # Gap calculation vs authoritative prior session close
        gap_pct = round((curr_px - prior_close) / prior_close * 100.0, 2)

        # Projected RVOL calculation:
        # Premarket hours (04:00 - 09:30 EST) typically represent ~1.5% - 2.5% of total day volume.
        # An elite EP pacing for 3x - 5x RVOL will have accumulated 5% - 20% of its normal 20-day average volume in premarket.
        vol_ratio = pm_vol / avg_20d_vol
        projected_rvol = round(max(0.1, vol_ratio * 15.0), 2)
        if pm_vol < 1000:
            # Baseline pacing if low early premarket prints
            projected_rvol = 0.5 if abs(gap_pct) >= 3.0 else 0.2

        # Thematic evaluation
        thm_eval = engine.evaluate_symbol(sym)
        theme_score = float(thm_eval.get("composite_score", 50.0))
        archetype = str(thm_eval.get("primary_archetype", "Neutral Flow"))
        is_veto = bool(thm_eval.get("is_veto", False))
        is_tailwind = (theme_score >= 65.0) and not is_veto
        sec_name = str(thm_eval.get("sector", "General"))
        thm_name = str(thm_eval.get("theme", "General"))

        # SEC 8-K filings & Catalyst Intelligence (fetched for active movers to maintain sub-second response)
        filings = []
        if abs(gap_pct) >= 1.5 or pm_vol >= 5000:
            filings = ep_news.edgar_8k(sym, today_str, days_back=4)
        filing_labels = []
        for f in filings:
            filing_labels.extend(f.get("labels", []))
        filing_str = ", ".join(filing_labels[:3]) if filing_labels else "None on file"

        # Catalyst Classification
        catalyst_cat = "Technical Mover"
        if any("earnings" in l.lower() for l in filing_labels):
            catalyst_cat = "Earnings / Forward Guidance (Arch 1)"
        elif any("agreement" in l.lower() or "contract" in l.lower() for l in filing_labels):
            catalyst_cat = "Major Commercial Contract (Arch 2)"
        elif any("fda" in l.lower() or "clinical" in l.lower() for l in filing_labels):
            catalyst_cat = "FDA / Clinical Milestone (Arch 2)"
        elif sec_name == "Biotechnology":
            catalyst_cat = "Biotech Catalyst / Speculation (Arch 2)"
        elif is_tailwind:
            catalyst_cat = f"{thm_name} Thematic Momentum Repricing"

        # Early Tactical Game Plan & Asymmetric Sizing
        is_biotech = (sec_name == "Biotechnology" or thm_name == "Biotechnology")
        max_capital_cap = 12.5 if is_biotech else 25.0

        if gap_pct >= 4.0:
            if gap_pct <= 12.0:
                tactic_name = "15m ORB (Opening Range Breakout)"
                tactic_rule = "Wait for 09:45 EST 15m candle to close. Place Buy-Stop at 15m High. Hard Stop at 15m Low (averages 2.8% risk vs 8.2% EOD). Protects against gap-and-crap faders."
            else:
                tactic_name = "Morning Washout & VWAP Reclaim"
                tactic_rule = "Extreme gap (+12%+). Do NOT chase at 09:30. Allow morning profit-taking flush to low. Enter on 5m candle close reclaiming VWAP with Stop at flush low."
        elif gap_pct <= -3.0:
            tactic_name = "Gap-Down Avoidance / Protection"
            tactic_rule = "Trading lower. If holding existing position below stop, exit on open (MOO). Do NOT enter new longs."
        else:
            tactic_name = "Volume Acceleration Watch"
            tactic_rule = "Monitor first 15-30m regular session volume pace. Only trigger if RVOL accelerates past 2.5x."

        record = {
            "symbol": sym,
            "prior_close": round(prior_close, 2),
            "premarket_price": round(curr_px, 2),
            "gap_pct": gap_pct,
            "premarket_vol": pm_vol,
            "avg_20d_vol": int(avg_20d_vol),
            "projected_rvol": projected_rvol,
            "sector": sec_name,
            "theme": thm_name,
            "theme_score": theme_score,
            "theme_archetype": archetype,
            "is_tailwind": is_tailwind,
            "is_veto": is_veto,
            "is_biotech": is_biotech,
            "max_capital_cap": max_capital_cap,
            "sec_8k_filings": filing_str,
            "catalyst_category": catalyst_cat,
            "early_tactic": tactic_name,
            "tactic_rule": tactic_rule,
            "timestamp": latest_ts.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Segregation:
        # 1. Potential Elite EP Forming Today: Gap >= 4.0% with strong thematic tailwind or projected RVOL >= 2.0x, or material 8-K
        if gap_pct >= 4.0 and (is_tailwind or len(filings) > 0 or projected_rvol >= 2.0) and not is_veto:
            potential_elite_eps.append(record)
        # 2. Overnight Gappers & Movers: Gap >= 2.5% or down gaps
        elif abs(gap_pct) >= 2.5:
            overnight_gappers.append(record)
        else:
            other_movers.append(record)

    # Sort descending by gap % and projected RVOL
    potential_elite_eps.sort(key=lambda x: (x["gap_pct"], x["projected_rvol"]), reverse=True)
    overnight_gappers.sort(key=lambda x: abs(x["gap_pct"]), reverse=True)
    other_movers.sort(key=lambda x: x["theme_score"], reverse=True)

    result_payload = {
        "metadata": {
            "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S EST"),
            "total_scanned": len(symbols),
            "potential_elite_eps_count": len(potential_elite_eps),
            "overnight_gappers_count": len(overnight_gappers),
            "leading_themes": [t.get("name") for t in summary.get("emerging_themes", [])[:5]]
        },
        "potential_elite_eps": potential_elite_eps,
        "overnight_gappers": overnight_gappers,
        "other_movers": other_movers[:25]
    }

    try:
        with open(PREMARKET_CACHE_PATH, "w") as f:
            json.dump(result_payload, f, indent=2)
    except Exception as e:
        print(f"Error caching premarket data: {e}")

    return result_payload


def get_premarket_data(force_refresh: bool = False) -> Dict[str, Any]:
    """Retrieves cached premarket data or runs a fresh scan if stale/missing."""
    return scan_premarket(force_refresh=force_refresh)


if __name__ == "__main__":
    data = scan_premarket(force_refresh=True)
    print(f"\n=== PREMARKET RADAR SCAN COMPLETE ===")
    print(f"Scanned {data['metadata']['total_scanned']} candidates at {data['metadata']['timestamp']}")
    print(f"Potential Elite EPs: {data['metadata']['potential_elite_eps_count']}")
    print(f"Overnight Gappers: {data['metadata']['overnight_gappers_count']}")
    if data["potential_elite_eps"]:
        print("\nTop Potential Elite EPs:")
        for ep in data["potential_elite_eps"][:5]:
            print(f"  {ep['symbol']:5s} | Gap: {ep['gap_pct']:+5.2f}% | Premarket: ${ep['premarket_price']:.2f} | Proj RVOL: {ep['projected_rvol']}x | Theme: {ep['theme']} ({ep['theme_score']}) | Tactic: {ep['early_tactic']}")
    if data["overnight_gappers"]:
        print("\nTop Overnight Gappers:")
        for g in data["overnight_gappers"][:5]:
            print(f"  {g['symbol']:5s} | Gap: {g['gap_pct']:+5.2f}% | Theme: {g['theme']} | SEC 8-K: {g['sec_8k_filings']}")
