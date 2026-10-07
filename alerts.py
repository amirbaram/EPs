"""Regime-transition alerts — announce the CHANGE, not just the state.

The 2026-07-02 lesson: the panel showed conditions but nothing said "10:15 — breadth rolled
over". After every few live-poll cycles serve snapshots the regime features and diffs them here;
each flip becomes a timestamped alert the Map renders (and the transition log persists for the
session). Detectors implement Amir's own tells:

  cum A-D slope flip            breadth building -> fading (and back)
  %>VWAP crossing 40 / 60       participation regime
  ALL sectors below/above VWAP  his "money leaving vs healthy rotation" danger signal
  SPY loses/reclaims VWAP       the index tell
  SPY breaks LOD / HOD          range resolution
  ratio 1h regime change        rotation break (QQQ/SPY, SMH/SPY, IWM/SPY)
"""

from __future__ import annotations

import datetime as dt
from collections import deque

LOG: deque = deque(maxlen=80)          # session transition log, newest last
NOTIF_LOG: deque = deque(maxlen=80)    # grade-A NOTIFIED events (mirrors the macOS pings) — the
#                                        in-app clickable record; banners vanish + click-through only
#                                        opens Script Editor (Amir 2026-07-08)
_PREV: dict = {}


def _now() -> str:
    try:
        from zoneinfo import ZoneInfo
        return dt.datetime.now(ZoneInfo("America/New_York")).strftime("%H:%M")
    except Exception:
        return dt.datetime.now().strftime("%H:%M")


def snapshot(internals: dict | None, sectors_vwap: dict, spy: dict | None,
             ratio_regimes: dict) -> dict:
    """The comparable feature set. sectors_vwap = {etf: 'above'|'below'};
    spy = {'vwap_side', 'range_pos', 'last', 'lod', 'hod'}; ratio_regimes = {pair: regime}."""
    s = {"cum_trend": (internals or {}).get("cum_trend"),
         "pct_vwap": (internals or {}).get("pct_vwap"),
         "sectors_below": sum(1 for v in sectors_vwap.values() if v == "below"),
         "sectors_n": len(sectors_vwap),
         "spy_vwap": (spy or {}).get("vwap_side"),
         "spy_last": (spy or {}).get("last"),
         "spy_lod": (spy or {}).get("lod"), "spy_hod": (spy or {}).get("hod"),
         "ratios": dict(ratio_regimes)}
    return s


def diff(cur: dict) -> list[dict]:
    """Compare against the previous snapshot; emit alerts for every regime FLIP. First call of a
    session only seeds the state."""
    global _PREV
    prev, _PREV = _PREV, cur
    out: list[dict] = []
    t = _now()

    def emit(kind, text, side):
        a = {"t": t, "kind": kind, "side": side, "text": text}
        out.append(a)
        LOG.append(a)

    if not prev:
        return out
    if prev.get("cum_trend") and cur.get("cum_trend") and prev["cum_trend"] != cur["cum_trend"]:
        if cur["cum_trend"] == "falling":
            emit("breadth", "cumulative A-D rolled over — participation fading", "bear")
        elif cur["cum_trend"] == "rising" and prev["cum_trend"] == "falling":
            emit("breadth", "cumulative A-D turned up — participation building", "bull")
    pv, cv = prev.get("pct_vwap"), cur.get("pct_vwap")
    if pv is not None and cv is not None:
        if pv >= 40 > cv:
            emit("breadth", f"only {cv:.0f}% of names above VWAP (crossed below 40%)", "bear")
        elif pv <= 60 < cv:
            emit("breadth", f"{cv:.0f}% of names above VWAP (crossed above 60%)", "bull")
    n = cur.get("sectors_n") or 0
    if n >= 8:
        pb, cb = prev.get("sectors_below", 0), cur.get("sectors_below", 0)
        if cb == n and pb < n:
            emit("sectors", "ALL sectors below VWAP — money leaving, not rotating", "bear")
        elif cb == 0 and pb > 0:
            emit("sectors", "ALL sectors above VWAP — broad risk-on tape", "bull")
    if prev.get("spy_vwap") and cur.get("spy_vwap") and prev["spy_vwap"] != cur["spy_vwap"]:
        emit("index", f"SPY {'lost' if cur['spy_vwap'] == 'below' else 'reclaimed'} VWAP",
             "bear" if cur["spy_vwap"] == "below" else "bull")
    if cur.get("spy_last") is not None:
        if prev.get("spy_lod") is not None and cur["spy_last"] < prev["spy_lod"]:
            emit("index", f"SPY broke the session low ({prev['spy_lod']})", "bear")
        if prev.get("spy_hod") is not None and cur["spy_last"] > prev["spy_hod"]:
            emit("index", f"SPY broke the session high ({prev['spy_hod']})", "bull")
    for pair, reg in (cur.get("ratios") or {}).items():
        old = (prev.get("ratios") or {}).get(pair)
        if old and reg and old != reg and reg in ("breakdown_fresh", "breakout_fresh"):
            side = "bear" if reg == "breakdown_fresh" else "bull"
            emit("rotation", f"{pair} 1h {reg.replace('_', ' ')} (was {old.replace('_', ' ')})", side)
    return out


_RVOL_STATE: dict = {}                 # sym -> {"t": epoch, "cooled": bool} — E2 re-arm state


def _rearm_sec() -> float:
    import config
    return float(getattr(config, "SIG_REARM_MIN", 90)) * 60


def rvol_alerts(rows: list[dict], cum_min: float = 1.5, slot_min: float = 3.0) -> list[dict]:
    """E2: alert when a name clears the combined RVOL-spike bar (cumulative-pace ≥ cum_min AND
    current-bar slot ≥ slot_min AND above VWAP). RE-ARM (was once-per-session, Amir 2026-07-08):
    a name may alert again once SIG_REARM_MIN passed OR its pace cooled below 1.2× in between —
    so a 9:35 spike that cools and re-fires at 13:00 pings again."""
    import time as _time
    t, now = _now(), _time.time()
    out: list[dict] = []
    for r in rows or []:
        sym = r.get("symbol")
        if not sym:
            continue
        st = _RVOL_STATE.get(sym)
        hot = (r.get("cum_rvol", 0) >= cum_min and r.get("slot_rvol") is not None
               and r.get("slot_rvol", 0) >= slot_min and r.get("above_vwap"))
        if st and not st["cooled"] and r.get("cum_rvol", 99) < 1.2:
            st["cooled"] = True                    # pace cooled -> eligible to re-fire
        if not hot:
            continue
        if st and not st["cooled"] and now - st["t"] < _rearm_sec():
            continue                               # still armed-out: neither cooled nor timed out
        _RVOL_STATE[sym] = {"t": now, "cooled": False}
        setup = r.get("setups")
        txt = (f"{sym} RVOL spike — cum ×{r['cum_rvol']}, slot ×{r['slot_rvol']}, >VWAP"
               + (f" · {setup}" if setup else ""))
        a = {"t": t, "kind": "rvol", "side": "bull", "text": txt, "sig": r.get("sig")}
        out.append(a)
        LOG.append(a)
        _maybe_notify(a, r)
    return out


# ── significance events E1/E3/E4/E5/E6 (plan-ep-significance-alerts; E2 above) ──────────────
_SIG_STATE = {"seen": set(),           # E1: symbols already on the radar this session
              "fired": {},             # (sym, kind) -> epoch of last alert (re-arm window)
              "vwap": {},              # E4: sym -> last above_vwap side
              "grade": {}}             # E6: sym -> last grade


def _maybe_notify(a: dict, row: dict) -> None:
    """Grade-A scope (config.SIG_ALERT_GRADES) -> native macOS notification (notify.py, Amir's
    osascript pattern). In-app LOG rows are unconditional; only the OS ping is grade-gated."""
    import config
    sig = (row or {}).get("sig") or {}
    if sig.get("grade") not in getattr(config, "SIG_ALERT_GRADES", ("A",)):
        return
    NOTIF_LOG.append({"t": _now(), "sym": (row or {}).get("symbol") or a["text"].split()[0],
                      "text": a["text"], "side": a.get("side", ""), "kind": a.get("kind", ""),
                      "grade": sig.get("grade", ""), "action": sig.get("action", ""),
                      "why": (sig.get("why") or "")[:160]})
    if not getattr(config, "SIG_NOTIFY_MACOS", True):
        return
    try:
        from notify import notify
        notify(f"Scanner · {sig.get('action', '')} {sig.get('grade', '')}", a["text"],
               subtitle=(sig.get("why") or "")[:120])
    except Exception:
        pass


def radar_events(rows: list[dict], phase: str) -> list[dict]:
    """Significance events over graded radar/rvol rows (rows carry row['sig'] from serve).
    E1 new radar entry · E3 crossed its scanned setup trigger · E4 VWAP flip on an A-name ·
    E5 pre-market gapper $vol threshold · E6 grade upgrade to A. All de-duped per (sym, kind)
    with the SIG_REARM_MIN window; every event lands in the Transitions LOG, and grade-A ones
    also fire the macOS notification."""
    import time as _time
    import config
    t, now = _now(), _time.time()
    out: list[dict] = []

    def fire(kind, sym, text, side, row):
        key = (sym, kind)
        if now - _SIG_STATE["fired"].get(key, 0) < _rearm_sec():
            return
        _SIG_STATE["fired"][key] = now
        sig = (row.get("sig") or {})
        a = {"t": t, "kind": kind, "side": side, "text": text, "sig": sig or None}
        out.append(a)
        LOG.append(a)
        _maybe_notify(a, row)

    for r in rows or []:
        sym, sig = r.get("symbol"), r.get("sig") or {}
        if not sym:
            continue
        g, act, why = sig.get("grade"), sig.get("action"), sig.get("why", "")
        side = "bear" if sig.get("direction") == "short" else "bull"
        if phase == "pre":                                             # E5
            pm = r.get("pm_dollar_m") or 0
            if pm >= getattr(config, "SIG_PM_DOLLAR_M", 3.0) and g == "A":
                fire("pm_gap", sym, f"{sym} pre-market gapper {r.get('pm_gap_pct', 0):+g}% · "
                     f"${pm:g}M traded · {why}", side, r)
            continue
        if sym not in _SIG_STATE["seen"]:                              # E1
            _SIG_STATE["seen"].add(sym)
            if g == "A":
                fire("radar_new", sym, f"{sym} NEW on radar — {act} · {why}", side, r)
        if act == "ACT" and "broke trigger" in why:                    # E3 (the high-value event)
            fire("trigger", sym, f"{sym} crossed its setup trigger — {why}", side, r)
        prev_v = _SIG_STATE["vwap"].get(sym)                           # E4 (A-names only)
        cur_v = bool(r.get("above_vwap"))
        if prev_v is not None and prev_v != cur_v and g == "A":
            fire("vwap_flip", sym, f"{sym} {'reclaimed' if cur_v else 'lost'} VWAP "
                 f"(grade-A name) · {why}", "bull" if cur_v else "bear", r)
        _SIG_STATE["vwap"][sym] = cur_v
        prev_g = _SIG_STATE["grade"].get(sym)                          # E6
        if prev_g in ("B", "C") and g == "A":
            fire("upgrade", sym, f"{sym} upgraded to grade A — {act} · {why}", side, r)
        if g:
            _SIG_STATE["grade"][sym] = g
    return out


def recent(n: int = 12) -> list[dict]:
    return list(LOG)[-n:][::-1]        # newest first for the UI


def reset() -> None:
    global _PREV
    _PREV = {}
    _RVOL_STATE.clear()
    _SIG_STATE.update(seen=set(), fired={}, vwap={}, grade={})
    LOG.clear()
    NOTIF_LOG.clear()
