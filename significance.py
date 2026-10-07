"""Significance grading for the EP radar / RVOL leaders (Amir 2026-07-08, plan:
docs/plan-ep-significance-alerts.md).

grade(row, setup_ctx, phase, extended) -> {score, grade, action, why, direction}

UNVALIDATED HEURISTIC: hand-weighted, transparent points — every component readable in `why`.
It ships labeled as such in the UI until validate_significance.py replay-grades it against
forward returns (1h/EOD and +1d/+3d/+5d) and Amir passes a verdict. Pure module: no serve/web
imports, unit-testable with plain dicts (tests/test_significance.py).

Inputs (all already computed elsewhere — this module only composes them):
  row       : a radar/rvol row. RTH: cum_rvol|vol_pace, slot_rvol, move_pct, above_vwap,
              catalyst, own_headline, news_react. PRE: pm_gap_pct, pm_dollar_m, avg_dollar_m,
              catalyst.
  setup_ctx : {"setup","state","trigger","level","last","tags"} from the scan join (or None).
  phase     : "rth" | "pre"
  extended  : True when the worst-TF extension badge is at the RED tier
              (>=10 ATR-from-50MA stock / >=6 ETF) -> score capped at B, long entries flagged.
"""

from __future__ import annotations

# catalyst quality points (keys = ep_news.CATALYSTS class names)
_CAT_STRONG = {"earnings": 25, "guidance": 25, "fda": 25, "mna": 25}
_CAT_MEDIUM = {"contract": 15}
_CAT_LIGHT = {"analyst": 10, "crypto": 10}
_CAT_BEARISH = {"offering": 15}          # significant — for the SHORT side

_CAT_WORD = {"earnings": "earnings", "guidance": "guidance", "fda": "FDA/clinical", "mna": "M&A",
             "contract": "contract", "analyst": "analyst", "crypto": "crypto theme",
             "offering": "offering/dilution", "sympathy": "sympathy only", "unknown": "no news"}

GRADE_A, GRADE_B = 70, 45                # A >=70 · B 45-69 · C <45


def _vol_points(row: dict, phase: str) -> tuple[int, str]:
    """Volume conviction 0-35. RTH: cumulative-pace RVOL tiers + slot-spike bonus.
    PRE: pre-market $vol as a fraction of the name's average full-day $vol."""
    if phase == "pre":
        pm, avg = row.get("pm_dollar_m"), row.get("avg_dollar_m")
        if not pm or not avg:
            return 0, ""
        frac = pm / avg
        pts = 35 if frac >= 0.25 else 25 if frac >= 0.10 else 15 if frac >= 0.05 else 8 if frac >= 0.02 else 0
        return pts, (f"pm $vol {frac:.0%} of a full day" if pts else "")
    cum = row.get("cum_rvol")
    if cum is None:
        cum = row.get("vol_pace")
    if not cum:
        return 0, ""
    pts = 35 if cum >= 5 else 25 if cum >= 3 else 15 if cum >= 2 else 8 if cum >= 1.5 else 0
    note = f"pace ×{cum:g}" if pts else ""
    slot = row.get("slot_rvol")
    if pts and slot and slot >= 3:
        pts = min(35, pts + 5)
        note += f" · slot ×{slot:g}"
    return pts, note


def _catalyst_points(row: dict) -> tuple[int, str, str | None]:
    """(points, note, direction-override). Own-headline catalysts score full; peer-only
    attribution halves (the story may not be the symbol's). Bearish catalysts keep their
    points but flip the read to the SHORT side."""
    cat = row.get("catalyst")
    own = bool(row.get("own_headline"))
    word = _CAT_WORD.get(cat, cat or "no news")
    if cat in _CAT_BEARISH:
        return _CAT_BEARISH[cat], f"{word} (bearish)", "short"
    pts = _CAT_STRONG.get(cat) or _CAT_MEDIUM.get(cat) or _CAT_LIGHT.get(cat) or 0
    if pts and not own:
        pts //= 2
        word += " (peer story)"
    return pts, (word if pts else ""), None


def _setup_points(ctx: dict | None) -> tuple[int, str]:
    """0-20 from the scan join: an EP/gapper in breakout is the prime context."""
    if not ctx or not ctx.get("setup"):
        return 0, ""
    setup, state = ctx.get("setup", ""), ctx.get("state", "")
    if setup in ("episodic_pivot", "gapper"):
        pts = 20 if state == "breakout" else 12
    else:
        pts = 8
    return pts, f"{setup}/{state or '—'}"


def grade(row: dict, setup_ctx: dict | None = None, phase: str = "rth",
          extended: bool | None = None) -> dict:
    """The significance read for one radar/rvol row. Never raises (bad fields -> conservative C)."""
    try:
        why: list[str] = []
        direction = "long"
        vp, vnote = _vol_points(row, phase)
        if vnote:
            why.append(vnote)
        cp, cnote, cdir = _catalyst_points(row)
        if cnote:
            why.append(cnote)
        if cdir:
            direction = cdir
        sp, snote = _setup_points(setup_ctx)
        if snote:
            why.append(snote)
        pp = 0
        if phase == "pre":
            # no VWAP exists pre-market, so the EP-sized gap carries the whole price-action weight
            gap = row.get("pm_gap_pct")
            if gap is not None and abs(gap) >= 4.0 and not extended:
                pp += 10
        else:
            if row.get("above_vwap"):
                pp += 5
                why.append(">VWAP")
            move = row.get("move_pct")
            if move is not None and abs(move) >= 2.0 and not extended:
                pp += 5
        rp = 0
        react = row.get("news_react")
        if react == "bad_news_green":
            rp = 10
            why.append("bad news, green tape — resilience")
        elif react == "good_news_red":
            rp = -10
            direction = "short"
            why.append("good news, red tape — rejection")
        score = max(0, min(100, vp + cp + sp + pp + rp))
        if extended:
            score = min(score, GRADE_A - 15)             # cap at B territory
            why.append("extended (red tier)")
        g = "A" if score >= GRADE_A else "B" if score >= GRADE_B else "C"

        # ---- actionability word ----
        action, act_note = "WATCH", ""
        trigger = (setup_ctx or {}).get("trigger")
        last = (setup_ctx or {}).get("last")
        state = (setup_ctx or {}).get("state", "")
        crossed = within = False
        if trigger and last:
            crossed = last >= trigger
            dist = (trigger / last - 1) * 100
            within = 0 <= dist <= 1.0
        if direction == "short":
            action, act_note = "FADE", "short-side read"
        elif extended:
            action, act_note = "CHASE-RISK", "late-entry risk"
        elif g == "A" and row.get("above_vwap") and (crossed or within):
            action = "ACT"
            act_note = (f"broke trigger {trigger:g}" if crossed else f"at trigger {trigger:g}")
        elif state in ("building", "watch") and (vp >= 8) and trigger and last and last < trigger:
            action = "STALK"
            act_note = f"trigger {trigger:g} ({(trigger / last - 1) * 100:+.1f}%)"
        if act_note:
            why.append(act_note)
        return {"score": int(score), "grade": g, "action": action,
                "why": " · ".join(why), "direction": direction}
    except Exception:
        return {"score": 0, "grade": "C", "action": "WATCH", "why": "", "direction": "long"}
