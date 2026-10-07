"""Significance LEDGER — the living validation layer (plan-ep-significance-alerts, item 2).

Every live-graded EP-radar / RVOL row and every fired alert event persists to
data/significance_log/{date}.parquet as it happens. A backfill pass later fills in the realized
forward outcomes (+1d/+3d/+5d/+10d/+21d closes + MFE/MAE within 5 sessions) as the days arrive.

Why: one-off replay studies rot AND Tiingo's news archive only reaches ~3 months back — but a
grade captured LIVE embeds the news that existed at that moment, so the ledger becomes our own
permanent archive. It powers (a) a rolling forward monitor ("is A still beating C?"), (b) the
history-replay overlay ("this A-grade → +12% in 3d"), (c) A-grade chart marks.

Throttle: at most one row per (symbol, surface, 30-min bucket) per session — the endpoints are
polled ~60s, the ledger doesn't need every poll. Never raises into the caller.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

import config

DIR = config.DATA_DIR / "significance_log"
_SEEN: set = set()                     # (date, sym, surface, bucket) throttle
_ROW_KEEP = ("move_pct", "gap_pct", "cum_rvol", "slot_rvol", "vol_pace", "above_vwap",
             "dollar_vol_m", "pm_gap_pct", "pm_dollar_m", "catalyst", "sent", "news_react",
             "setups", "last_px")


def _now_et() -> dt.datetime:
    from zoneinfo import ZoneInfo
    return dt.datetime.now(ZoneInfo("America/New_York"))


def _append(date: str, recs: list[dict], kind: str) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    p = DIR / f"{date}{'_events' if kind == 'events' else ''}.parquet"
    df = pd.DataFrame(recs)
    if p.exists():
        try:
            df = pd.concat([pd.read_parquet(p), df], ignore_index=True)
        except Exception:
            pass
    df.to_parquet(p, index=False)


def record_rows(rows: list[dict], surface: str) -> int:
    """Persist graded rows (live paths only — callers guard the phase). Returns #recorded."""
    try:
        now = _now_et()
        date, t = now.date().isoformat(), now.strftime("%H:%M")
        bucket = now.hour * 2 + now.minute // 30
        recs = []
        for r in rows or []:
            sym, sig = r.get("symbol"), r.get("sig") or {}
            if not sym or not sig:
                continue
            key = (date, sym, surface, bucket)
            if key in _SEEN:
                continue
            _SEEN.add(key)
            rec = {"date": date, "t": t, "surface": surface, "symbol": sym,
                   "grade": sig.get("grade"), "score": sig.get("score"),
                   "action": sig.get("action"), "direction": sig.get("direction"),
                   "why": sig.get("why")}
            rec.update({k: r.get(k) for k in _ROW_KEEP})
            recs.append(rec)
        if recs:
            _append(date, recs, "rows")
        return len(recs)
    except Exception:
        return 0


def record_events(events: list[dict]) -> int:
    """Persist fired alert events (E1-E6 + rvol spikes) with their significance read."""
    try:
        if not events:
            return 0
        now = _now_et()
        date = now.date().isoformat()
        recs = []
        for a in events:
            sig = a.get("sig") or {}
            recs.append({"date": date, "t": a.get("t"), "kind": a.get("kind"),
                         "side": a.get("side"), "text": a.get("text"),
                         "grade": sig.get("grade"), "score": sig.get("score"),
                         "action": sig.get("action"), "direction": sig.get("direction")})
        _append(date, recs, "events")
        return len(recs)
    except Exception:
        return 0


# ── outcome backfill (run after the daily update / from the maintenance chain) ────────────────
_HORIZONS = ((1, "fwd_1d"), (3, "fwd_3d"), (5, "fwd_5d"), (10, "fwd_10d"), (21, "fwd_21d"))


def backfill_outcomes(days_back: int = 40) -> int:
    """Fill realized outcomes into ledger rows that have an entry price and enough elapsed
    sessions: +1/+3/+5/+10/+21d closes, MFE/MAE within 5 sessions (max high / min low vs entry),
    and the SPY-adjusted +5d. Idempotent — only touches missing cells. Returns #files updated."""
    import datastore
    if not DIR.exists():
        return 0
    spy = datastore.load_bars("SPY")
    touched = 0
    for p in sorted(DIR.glob("20*.parquet")):
        if p.stem.endswith("_events"):
            continue
        date = p.stem
        if (dt.date.today() - dt.date.fromisoformat(date)).days > days_back + 40:
            continue
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if "fwd_21d" in df.columns and df["fwd_21d"].notna().all():
            continue                                   # fully settled
        changed = False
        cache: dict = {}
        for i, row in df.iterrows():
            entry = row.get("last_px")
            if entry is None or pd.isna(entry) or not entry:
                continue
            sym = row["symbol"]
            if sym not in cache:
                cache[sym] = datastore.load_bars(sym)
            d = cache[sym]
            if d is None or not len(d):
                continue
            after = d[d.index > pd.Timestamp(date)]
            for k, col in _HORIZONS:
                if col in df.columns and pd.notna(row.get(col)):
                    continue
                if len(after) >= k:
                    df.loc[i, col] = (float(after["close"].iloc[k - 1]) / entry - 1) * 100
                    changed = True
            w = after.iloc[:5]
            if len(w) >= 1 and ("mfe_5d" not in df.columns or pd.isna(row.get("mfe_5d"))):
                df.loc[i, "mfe_5d"] = (float(w["high"].max()) / entry - 1) * 100
                df.loc[i, "mae_5d"] = (float(w["low"].min()) / entry - 1) * 100
                changed = True
            if spy is not None and ("fwd_5d_vs_spy" not in df.columns
                                    or pd.isna(row.get("fwd_5d_vs_spy"))):
                sa = spy[spy.index > pd.Timestamp(date)]
                if len(sa) >= 5 and pd.notna(df.loc[i].get("fwd_5d")):
                    s0 = float(spy[spy.index <= pd.Timestamp(date)]["close"].iloc[-1])
                    df.loc[i, "fwd_5d_vs_spy"] = float(df.loc[i, "fwd_5d"]) - \
                        (float(sa["close"].iloc[4]) / s0 - 1) * 100
                    changed = True
        if changed:
            df.to_parquet(p, index=False)
            touched += 1
    return touched


def monitor(min_n: int = 5) -> str:
    """Rolling forward read over settled ledger rows: is A still beating C? One line per horizon."""
    if not DIR.exists():
        return "no ledger yet"
    frames = []
    for p in sorted(DIR.glob("20*.parquet")):
        if not p.stem.endswith("_events"):
            try:
                frames.append(pd.read_parquet(p))
            except Exception:
                pass
    if not frames:
        return "no ledger yet"
    df = pd.concat(frames, ignore_index=True)
    df = df[df.direction == "long"].sort_values("t").groupby(["symbol", "date"]).first().reset_index()
    lines = [f"significance forward monitor — {df.date.nunique()} sessions · {len(df)} symbol-days"]
    for col in ("fwd_1d", "fwd_5d", "fwd_5d_vs_spy"):
        if col not in df.columns:
            continue
        parts = []
        for g in ("A", "B", "C"):
            x = df[df.grade == g][col].dropna()
            parts.append(f"{g} {x.mean():+.1f}% (n={len(x)})" if len(x) >= min_n else f"{g} n<{min_n}")
        lines.append(f"  {col:14} " + " · ".join(parts))
    return "\n".join(lines)


if __name__ == "__main__":
    n = backfill_outcomes()
    print(f"backfilled {n} ledger files")
    print(monitor())
