"""The rule table — awareness map -> ranked plain-language calls.

Each rule is an explicit, config-thresholded condition over the awareness payload that emits a
call: which market / group / ratio to LONG, SHORT or AVOID, with the because-phrase, the levels
that matter, a what-next expectation — and the rule's MEASURED historical stats (n, hit rate vs
base rate, avg move) from data/rule_stats.json, written by validate_awareness.py. Rules without
enough history rank low and render "unvalidated": every displayed edge is a measured one.

Point-in-time: rules only read the payload, which is already point-in-time clean.
"""

from __future__ import annotations

import json
import math
import os
import time

import config

STATS_PATH = os.path.join("data", "rule_stats.json")
MIN_N = 30                       # observations before a rule may claim (and rank by) its edge
_CAPS = {"market": 1, "index": 2, "group": 4, "ratio": 2}

_UP_REGIMES = ("pullback_in_uptrend", "breakout_fresh", "trend_up", "impulse_up")
_STRONG_UP = ("impulse_up", "breakout_fresh", "trend_up")


def _wilson_lb(hit: float, n: int, z: float = 1.28) -> float:
    """Wilson lower bound of a hit rate (z=1.28 ~ 80%) — pessimistic edge for small n."""
    if not n:
        return 0.0
    denom = 1 + z * z / n
    centre = hit + z * z / (2 * n)
    margin = z * math.sqrt((hit * (1 - hit) + z * z / (4 * n)) / n)
    return (centre - margin) / denom


_STATS_CACHE: dict = {"mtime": None, "data": {}}


def load_stats() -> dict:
    """data/rule_stats.json, mtime-cached: {rule_id: {n, hit, avg, base, ...}}."""
    try:
        mt = os.path.getmtime(STATS_PATH)
    except OSError:
        return {}
    if _STATS_CACHE["mtime"] != mt:
        try:
            with open(STATS_PATH) as f:
                _STATS_CACHE.update(mtime=mt, data=json.load(f))
        except Exception:
            return _STATS_CACHE["data"]
    return _STATS_CACHE["data"]


def stats_summary() -> dict | None:
    """{window, rules, tiers} of the current rule_stats.json — the drawer footer's provenance line."""
    st = load_stats()
    if not st:
        return None
    wins = {s.get("window") for s in st.values() if s.get("window")}
    return {"window": max(wins) if wins else None, "rules": len(st),
            "n_total": int(sum(s.get("n", 0) for s in st.values()))}


def _mk(rule, scope, target, side, strength, phrase, levels, expect, metric) -> dict:
    return {"rule": rule, "scope": scope, "target": target, "side": side,
            "strength": round(min(max(strength, 0.0), 1.0), 2), "phrase": phrase,
            "levels": {k: v for k, v in (levels or {}).items() if v is not None},
            "expect": expect, "metric": metric}


def _room_up(view) -> tuple[float | None, dict | None]:
    """ATRs of clear air above the CURRENT price to the next anchor resistance (None = no lid)."""
    last = ((view.get("intra") or {}).get("day") or {}).get("last") or view["daily"]["close"]
    atr = view["daily"]["atr"] or 0
    for band in view["daily"]["sr"]["resistance"]:
        if band["price"] > last:
            return (band["price"] - last) / atr if atr else None, band
    return None, None


def _short_phrase(seq) -> str:
    p = seq["phrase"]
    return p.split(";")[0]


# ---------------------------------------------------------------- index gap rules

def _r1_gap_go(payload):
    out = []
    for v in payload["indices"]:
        g = ((v.get("intra") or {}).get("gap")) or {}
        day = ((v.get("intra") or {}).get("day")) or {}
        seq = v["daily"]["seq"]
        if not g or g.get("atr") is None or g["status"] != "holding" or g["atr"] < config.AWARE_GAP_MIN_ATR:
            continue
        pos = None
        if seq["regime"] == "range" and seq["meta"].get("range_hi") and day.get("last") is not None:
            hi, lo = seq["meta"]["range_hi"], seq["meta"]["range_lo"]
            pos = (day["last"] - lo) / (hi - lo) if hi > lo else None
        opened_over_r = bool(v["daily"]["sr"]["resistance"]
                             and (day.get("last") or 0) > v["daily"]["sr"]["resistance"][0]["price"])
        regime_ok = (seq["regime"] in ("pullback_in_uptrend", "breakout_fresh", "trend_up")
                     or (seq["regime"] == "impulse_up" and "extended" not in seq["flags"])
                     or (seq["regime"] == "range" and (opened_over_r or (pos or 0) >= 0.6)))
        room, nxt = _room_up(v)
        if not regime_ok or (room is not None and room < 1.0 and not opened_over_r):
            continue
        lv = (v.get("intra") or {}).get("levels") or {}
        strength = min(g["atr"] / config.AWARE_GAP_MIN_ATR, 2) / 2
        out.append(_mk(
            "gap_go", "index", v["key"], "long", strength,
            f"Long bias {v['label']} — gapped +{g['atr']:.1f} ATR ({g['pct']:+.1f}%) and holding it, "
            f"after {_short_phrase(seq)}"
            + (f"; cleared resistance {v['daily']['sr']['resistance'][0]['price']}" if opened_over_r else "")
            + (f"; next lid {nxt['price']} ×{nxt['strength']} {room:.1f} ATR up" if nxt else "; no lid overhead"),
            {"entry_ref": day.get("last"), "invalidation": lv.get("open"),
             "next_R": nxt["price"] if nxt else None},
            "gaps this size that hold the first hour tend to close as trend days — stay long while it "
            f"holds above the open {lv.get('open')}", "rod"))
    return out


def _r2_gap_into_extension(payload):
    out = []
    for v in payload["indices"]:
        g = ((v.get("intra") or {}).get("gap")) or {}
        seq = v["daily"]["seq"]
        if not g or g.get("atr") is None or g["atr"] < config.AWARE_GAP_MIN_ATR or g["status"] == "na":
            continue
        if (seq["regime"] in ("impulse_up", "trend_up") and "extended" in seq["flags"]) \
                or seq["regime"] == "distribution":
            out.append(_mk(
                "gap_into_extension", "index", v["key"], "avoid", 0.7,
                f"Don't chase {v['label']} — gap +{g['atr']:.1f} ATR into "
                + ("a {:.1f}-ATR extension".format(seq["meta"].get("ext20") or 0)
                   if seq["regime"] != "distribution" else "a distribution range"),
                {"invalidation": seq.get("invalidation")},
                "late gaps into stretched trends fade or chop more often than they run", "rod"))
    return out


def _r3_gap_fade(payload):
    out = []
    for v in payload["indices"]:
        g = ((v.get("intra") or {}).get("gap")) or {}
        if not g or g.get("atr") is None or abs(g["atr"]) < config.AWARE_GAP_MIN_ATR \
                or g["status"] not in ("faded", "filled"):
            continue
        lv = (v.get("intra") or {}).get("levels") or {}
        side_word = "longs" if g["atr"] > 0 else "shorts"
        out.append(_mk(
            "gap_fade_reversal", "index", v["key"], "avoid", 0.6,
            f"{v['label']} gap {g['pct']:+.1f}% is {g['status']} — reversal tape, stand down on {side_word}",
            {"entry_ref": lv.get("open"), "invalidation": lv.get("y_close")},
            "a faded gap flips the day's odds — wait for the open to be reclaimed", "rod"))
    return out


def _r4_gap_down_dip(payload):
    out = []
    for v in payload["indices"]:
        g = ((v.get("intra") or {}).get("gap")) or {}
        day = ((v.get("intra") or {}).get("day")) or {}
        seq = v["daily"]["seq"]
        sup = v["daily"]["sr"]["support"][0] if v["daily"]["sr"]["support"] else None
        if not g or g.get("atr") is None or g["atr"] > -config.AWARE_GAP_MIN_ATR or not sup:
            continue
        atr = v["daily"]["atr"] or 1
        near_sup = day.get("last") is not None and abs(day["last"] - sup["price"]) / atr <= 0.75
        responding = g["status"] in ("faded", "filled") or (day.get("open_pct") or 0) > 0
        if seq["regime"] in _UP_REGIMES and sup["strength"] >= 5 and near_sup and responding:
            out.append(_mk(
                "gap_down_into_support", "index", v["key"], "long", 0.6,
                f"Dip-buy window {v['label']} — gapped {g['pct']:+.1f}% into support "
                f"{sup['price']} ×{sup['strength']} and buyers are responding (still {_short_phrase(seq)})",
                {"entry_ref": day.get("last"), "invalidation": sup["lo"],
                 "next_R": ((v.get("intra") or {}).get("levels") or {}).get("y_close")},
                "gap-downs into strong support in an uptrend tend to reclaim toward the prior close", "rod"))
    return out


# ---------------------------------------------------------------- structure rules

def _r5_break_follow(payload):
    out = []
    if payload["live"]:
        return out                                      # close-mode read: next-days follow-through
    for v in payload["indices"] + payload["sectors"]:
        seq = v["daily"]["seq"]
        if seq["regime"] not in ("breakout_fresh", "breakdown_fresh") or (seq["meta"].get("age") or 9) > 3:
            continue
        up = seq["regime"] == "breakout_fresh"
        room, nxt = _room_up(v)
        if up and room is not None and room < 1.0:
            continue
        out.append(_mk(
            "breakout_follow" if up else "breakdown_follow", "index" if v["kind"] == "index" else "group",
            v["key"], "long" if up else "short", 0.7,
            f"{'Long' if up else 'Short'} {v['label']} — {_short_phrase(seq)}"
            + (f"; {room:.1f} ATR to the next level" if room is not None else "; no lid overhead" if up else ""),
            {"entry_ref": v["daily"]["close"], "invalidation": seq.get("invalidation"),
             "next_R": nxt["price"] if nxt else None},
            "fresh breaks from multi-week bases tend to follow through over the next 1-3 sessions",
            "fwd3"))
    return out


def _r7_coil_break(payload):
    out = []
    for v in payload["indices"]:
        seq = v["daily"]["seq"]
        h1 = ((v.get("intra") or {}).get("h1")) or {}
        day = ((v.get("intra") or {}).get("day")) or {}
        if seq["regime"] != "range" or "after_advance" not in seq["flags"] or not h1:
            continue
        r_hi = seq["meta"].get("range_hi")
        if r_hi and day.get("last") is not None and day["last"] > r_hi \
                and h1["seq"]["regime"] == "breakout_fresh":
            out.append(_mk(
                "coil_at_highs", "index", v["key"], "long", 0.8,
                f"Range-break day {v['label']} — trading through the top of the "
                f"{seq['meta'].get('age')}-bar consolidation at {r_hi} with 1h breakout structure",
                {"entry_ref": day.get("last"), "invalidation": r_hi},
                "a daily range break that holds into the close often marks the start of the next leg",
                "rod"))
    return out


# ---------------------------------------------------------------- ratio rules

def _ratio_calls(payload):
    out = []
    for v in payload["ratios"]:
        seq = v["daily"]["seq"]
        num, den = v["key"].split("/")
        age_ok = (seq["meta"].get("age") or 9) <= config.AWARE_FRESH_BARS
        if seq["regime"] == "breakdown_fresh" and age_ok:
            out.append(_mk(
                "ratio_break", "ratio", v["key"], "avoid", 0.7,
                f"Rotation AGAINST {v['label']} — the ratio broke down ({_short_phrase(seq)}); "
                f"avoid/short {num} vs {den}",
                {"invalidation": seq.get("invalidation")},
                f"ratio breakdowns precede several sessions of {num} underperformance", "rel3"))
        elif seq["regime"] in ("breakout_fresh", "impulse_up") and age_ok:
            out.append(_mk(
                "ratio_trend_fresh", "ratio", v["key"], "long", 0.7,
                f"Rotation INTO {v['label']} — {_short_phrase(seq)}; favor {num} over {den}",
                {"invalidation": seq.get("invalidation"), "confirm": seq.get("confirm")},
                f"fresh ratio breaks tend to run — look for longs in {num}-side names", "rel3"))
    return out


# ---------------------------------------------------------------- group rotation rules

def _group_calls(payload):
    out = []
    for r in payload["rotation"]["groups"]:
        if r["kind"] == "industry" and r.get("n") is not None and r["n"] < config.GROUP_MIN_MEMBERS:
            continue
        name = r["name"]
        z = r.get("rot_z") or 0
        seq_reg = r.get("seq_regime")
        tag = f"{name}" + (f" ({r['etf']})" if r.get("etf") else "")
        conf = (f"{r['pct_up']:.0f}% members up" if r.get("pct_up") is not None else "no member read")
        if r["quadrant"] == "leading_extending" and r.get("acting") == "well" and not r.get("narrow") \
                and (seq_reg is None or (seq_reg in _UP_REGIMES and r.get("at_level") != "at_R")):
            # leader NAMES go FIRST — the banner headline truncates long phrases and the names
            # are the actionable part (Amir 2026-07-03: "it should tell me who the group leader is")
            via = f" → {', '.join(r['leaders'])}" if r.get("leaders") else ""
            c = _mk(
                "group_leader_go", "group", name, "long", min(abs(z) / 2, 1),
                f"Long {tag}{via} — leading ({r['rs_1m']:+.1f}% 1m RS) and extending today "
                f"(rot z {z:+.1f}), {conf}" + (f"; structure: {seq_reg}" if seq_reg else ""),
                {}, "hunt the strongest members — leaders extending with broad participation "
                    "tend to stay in front for days", "rel1")
            c["leaders"] = list(r.get("leaders") or [])   # clickable chips in the map (stripped from the phrase there)
            out.append(c)
        elif r["quadrant"] == "leading_fading" and (r.get("acting") == "poorly" or r.get("narrow")):
            why = "members not confirming" if r.get("narrow") else f"{conf}"
            out.append(_mk(
                "group_fade_warn", "group", name, "avoid", min(abs(z) / 2.5, 1),
                f"Trim/avoid {tag} — leadership fading today (rot z {z:+.1f}), {why}",
                {}, "fading leaders with weak participation often mark short-term group tops", "rel1"))
        elif r["quadrant"] == "lagging_improving" and seq_reg in ("breakout_fresh", "pullback_in_uptrend"):
            out.append(_mk(
                "rotation_entry", "group", name, "long", min(abs(z) / 2.5, 1),
                f"Early rotation into {tag} — lagging group ({r['rs_1m']:+.1f}% 1m RS) firing today "
                f"(rot z {z:+.1f}) with {seq_reg} structure",
                {}, "first thrusts off the lows in left-behind groups can run for multiple sessions", "rel3"))
        elif r["quadrant"] == "lagging_breaking" and r.get("acting") == "poorly" \
                and seq_reg in ("breakdown_fresh", "impulse_down", "trend_down", "distribution"):
            out.append(_mk(
                "group_break_short", "group", name, "short", min(abs(z) / 2, 1),
                f"Short {tag} — lagging and breaking down together (rot z {z:+.1f}), {conf}, "
                f"structure: {seq_reg}",
                {}, "weak groups making fresh breaks with members confirming tend to keep bleeding", "rel3"))
    return out


# ---------------------------------------------------------------- market guard

def _r14_risk_off(payload):
    internals = payload["rotation"].get("internals") or {}
    spy = next((v for v in payload["indices"] if v["key"] == "SPY"), None)
    if not internals or internals.get("score") is None or spy is None or not spy.get("intra"):
        return []
    day = spy["intra"].get("day") or {}
    if internals["score"] <= -0.5 and day.get("vwap_side") == "below" \
            and (day.get("open_pct") or 0) < 0 and internals.get("cum_trend") == "falling":
        return [_mk(
            "risk_off_guard", "market", "SPY", "avoid", 0.9,
            f"Risk-off tape — {internals['pct_up']:.0f}% up, TICK {internals['tick']:+d}, "
            f"cum A-D falling, SPY below VWAP and open: stand down on new longs today",
            {"entry_ref": (spy["intra"].get("levels") or {}).get("vwap")},
            "broad red internals that are still deteriorating rarely repair the same day", "rod")]
    return []


_RULES = [_r1_gap_go, _r2_gap_into_extension, _r3_gap_fade, _r4_gap_down_dip,
          _r5_break_follow, _r7_coil_break, _ratio_calls, _group_calls, _r14_risk_off]


def evaluate(payload: dict, stats: dict | None = None) -> list[dict]:
    """All fired rules, stats-attached, ranked (strength × measured edge), capped per scope."""
    stats = stats if stats is not None else load_stats()
    fired: list[dict] = []
    for fn in _RULES:
        try:
            fired.extend(fn(payload))
        except Exception:
            continue
    disabled = set(getattr(config, "AWARE_RULES_OFF", ()) or ())
    ranked = []
    for c in fired:
        if c["rule"] in disabled:
            continue
        st = stats.get(c["rule"])
        if st and st.get("n", 0) >= MIN_N:
            edge = max(_wilson_lb(st["hit"], st["n"]) - st.get("base", 0.5), 0.02)
            c["stats"] = {k: st.get(k) for k in ("n", "hit", "avg", "base", "window", "tier")}
        else:
            edge = 0.05
            c["stats"] = ({**{k: st.get(k) for k in ("n", "hit", "avg", "base")}, "unvalidated": True}
                          if st else None)
        c["rank"] = round(c["strength"] * edge, 4)
        ranked.append(c)
    ranked.sort(key=lambda c: -c["rank"])
    kept, counts, rules = [], {}, {}
    for c in ranked:                                  # scope caps + max 2 of the same rule (diversity)
        counts[c["scope"]] = counts.get(c["scope"], 0) + 1
        rules[c["rule"]] = rules.get(c["rule"], 0) + 1
        if counts[c["scope"]] <= _CAPS.get(c["scope"], 2) and rules[c["rule"]] <= 2:
            kept.append(c)
    return kept
