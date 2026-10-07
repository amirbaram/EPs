"""Cross-instrument news reaction — PROXY news -> the UNDERLYING's reaction (Amir 2026-07-07).

Some news is a signal for a DIFFERENT instrument than it names: "MSTR/Saylor sells 80K bitcoin" is a
SUPPLY event for bitcoin (BTC=F), not really an MSTR call. This maps proxies -> an underlying, classifies
each proxy headline as bullish/bearish FOR THE UNDERLYING via a curated SUPPLY/DEMAND EVENT lexicon (NOT
generic word-sentiment, and NOT price-descriptive words like "rally/breaks above" — those are the reaction,
measured separately), reads the underlying's actual move, and fires the reaction-divergence TELL:

  bearish events (supply) + underlying UP   = demand ate the supply           -> BULLISH resilience
  bullish events (demand) + underlying DOWN = buyers not showing / distribution -> BEARISH

Writes a handoff file (data/cross_news/latest.json) so Amir can hand Claude the details for a nuanced read.
Extensible: add any underlying -> {proxies, bearish/bullish lexicon} to PROXY_MAP.

    .venv/bin/python cross_news.py            # all mapped underlyings -> data/cross_news/{date}.json
    .venv/bin/python cross_news.py BTC=F      # one underlying, verbose
"""
from __future__ import annotations

import argparse
import json
import re

import pandas as pd

import config
import datastore
import ep_news

# underlying -> {label, proxies that MOVE it, and the FUNDAMENTAL event lexicon FOR THE UNDERLYING}.
# Lexicons are supply/demand ACTIONS, not price words (rally/surge/break-above = the reaction, excluded).
PROXY_MAP: dict[str, dict] = {
    "BTC=F": {
        "label": "Bitcoin",
        "proxies": ["MSTR", "COIN", "CRCL", "RIOT", "MARA", "CLSK", "HOOD", "GBTC", "IBIT"],
        "bearish": r"\b(sell|sells|sold|selling|sale|dump|dumps|offload\w*|liquidat\w*|hack|hacked|"
                   r"breach\w*|ban|banned|outflow|outflows|redemption|halt|halts|seiz\w*|bankrupt\w*|"
                   r"fraud|probe|lawsuit|short seller|capitulat\w*|exploit)\b",
        "bullish": r"\b(approv\w*|adopt\w*|inflow|inflows|accumulat\w*|buys?|bought|adds?|added|"
                   r"treasury|reserve|integrat\w*|custody|institutional|halving|allocat\w*|stake[sd]?)\b",
    },
    # e.g. semis: "SMH": {"label": "Semis", "proxies": ["NVDA","AVGO","AMD","TSM","ASML","MU"],
    #                     "bearish": r"...glut|cut|export ban|downgrade...", "bullish": r"...capex|order|demand..."},
}


def event_class(title: str, spec: dict) -> str:
    """bearish / bullish / mixed / neutral FOR THE UNDERLYING (fundamental action words only)."""
    t = title.lower()
    b = bool(re.search(spec["bearish"], t))
    g = bool(re.search(spec["bullish"], t))
    return "bearish" if (b and not g) else ("bullish" if (g and not b) else ("mixed" if (b and g) else "neutral"))


def _move(sym: str, date: str) -> float | None:
    """Today's % move vs the prior settled close (futures 24h store or the equity 5m store)."""
    d = datastore.load_bars(sym)
    pc = float(d["close"].iloc[-1]) if d is not None and len(d) else None
    b = datastore.load_bars_15m(sym) if str(sym).endswith("=F") else datastore.load_bars_5m(sym, calibrate=False)
    cur = float(b["close"].iloc[-1]) if b is not None and len(b) else None
    return round(100 * (cur / pc - 1), 2) if (pc and cur) else None


def analyze(underlying: str, date: str) -> dict:
    spec = PROXY_MAP[underlying]
    move = _move(underlying, date)
    items, seen = [], set()
    for p in spec["proxies"]:
        try:
            hs = ep_news.news_for(p, date, days_back=2, limit=8)
        except Exception:
            hs = []
        for h in hs:
            ev = event_class(h["title"], spec)
            key = h["title"][:80]
            if ev in ("bearish", "bullish", "mixed") and key not in seen:
                seen.add(key)
                items.append({"proxy": p, "event": ev, "title": h["title"][:150]})
    bear = sum(i["event"] in ("bearish", "mixed") for i in items)
    bull = sum(i["event"] in ("bullish", "mixed") for i in items)
    lean = "bearish" if bear > bull else ("bullish" if bull > bear else None)
    tell, warn = "aligned", ""
    if move is not None and abs(move) >= 0.5 and lean:
        up = move > 0
        if lean == "bearish" and up:
            tell = "bad_news_resilient"
            warn = f"↑ BEARISH events (supply) but {spec['label']} {move:+.2f}% — demand ate the supply (BULLISH tell)"
        elif lean == "bullish" and not up:
            tell = "good_news_rejected"
            warn = f"⚠ BULLISH events but {spec['label']} {move:+.2f}% — buyers not showing (BEARISH tell)"
    return {"underlying": underlying, "label": spec["label"], "move_pct": move,
            "event_lean": lean, "n_bearish": bear, "n_bullish": bull, "tell": tell, "warn": warn,
            "items": sorted(items, key=lambda x: x["event"])}


def run(date: str | None = None) -> dict:
    date = date or str(pd.Timestamp.now(tz="America/New_York").date())
    reads = [analyze(u, date) for u in PROXY_MAP]
    blob = {"date": date, "generated": pd.Timestamp.now(tz="America/New_York").isoformat(),
            "note": "Cross-instrument news reaction (proxy events -> underlying's move). bad-news-resilient "
                    "= bullish tell; good-news-rejected = bearish. Hand to Claude for a nuanced read.",
            "reads": reads}
    d = config.DATA_DIR / "cross_news"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{date}.json").write_text(json.dumps(blob, indent=2, default=str))
    (d / "latest.json").write_text(json.dumps(blob, indent=2, default=str))
    return blob


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("underlying", nargs="?", help="one underlying (e.g. BTC=F); default = all mapped")
    a = ap.parse_args()
    blob = run()
    for r in blob["reads"]:
        if a.underlying and r["underlying"] != a.underlying:
            continue
        print(f"\n{r['label']} ({r['underlying']}): move {r['move_pct']}%  events {r['n_bearish']}✗/{r['n_bullish']}✓  lean {r['event_lean']}")
        if r["warn"]:
            print(f"  >>> {r['warn']}")
        for i in r["items"][:10]:
            print(f"    [{i['event']:7s}] {i['proxy']:5s} {i['title'][:78]}")
    print(f"\n-> data/cross_news/{blob['date']}.json (+ latest.json)")


if __name__ == "__main__":
    main()
