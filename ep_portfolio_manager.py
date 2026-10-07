"""EP Portfolio Manager & Lifecycle Evaluation Engine.

Manages real-time tracking, risk allocation, sizing, and automated execution alerts
for active Episodic Pivot positions:
- Account equity, cash allocation, and open portfolio heat ($ and R-multiples)
- Lifecycle tracking through forward daily bars
- Automated alert triggers:
  * Day 2 Secondary Add (+50% size on D1 High breakout)
  * Hard Stop Loss Breach (D1 Low or trailing stop violation)
  * Larsson Blue Flip Exit (Trend momentum break)
  * Moving Average Violations (20 EMA swing trail, 50 SMA institutional baseline)
  * Profit Target Milestones (+2R, +3R, +5R)
- Persistent storage in data/cache/ep_portfolio.json
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import datastore
import indicators
import labels
import scanner_core
import universe

PORTFOLIO_PATH = Path("data/cache/ep_portfolio.json")

DEFAULT_PORTFOLIO: Dict[str, Any] = {
    "settings": {
        "portfolio_size": 100000.0,
        "default_risk_pct": 1.0,
        "max_portfolio_heat_pct": 6.0
    },
    "positions": [],
    "closed_positions": []
}


def load_portfolio() -> Dict[str, Any]:
    """Loads portfolio data from JSON cache, returning defaults if not found."""
    if PORTFOLIO_PATH.exists():
        try:
            with open(PORTFOLIO_PATH, "r") as f:
                data = json.load(f)
                if "settings" not in data:
                    data["settings"] = DEFAULT_PORTFOLIO["settings"]
                if "positions" not in data:
                    data["positions"] = []
                if "closed_positions" not in data:
                    data["closed_positions"] = []
                return data
        except Exception as e:
            print(f"Error reading portfolio cache: {e}")
    return json.loads(json.dumps(DEFAULT_PORTFOLIO))


def save_portfolio(port: Dict[str, Any]) -> None:
    """Atomically saves portfolio data to JSON cache."""
    PORTFOLIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = PORTFOLIO_PATH.with_suffix(".tmp.json")
    with open(tmp_path, "w") as f:
        json.dump(port, f, indent=2)
    os.replace(tmp_path, PORTFOLIO_PATH)


def update_settings(portfolio_size: float, default_risk_pct: float) -> Dict[str, Any]:
    """Updates account size and default risk percentage."""
    port = load_portfolio()
    port["settings"]["portfolio_size"] = max(1000.0, float(portfolio_size))
    port["settings"]["default_risk_pct"] = max(0.1, min(10.0, float(default_risk_pct)))
    save_portfolio(port)
    return port["settings"]


def add_position(
    symbol: str,
    entry_price: float,
    stop_price: float,
    shares: int,
    event_date: Optional[str] = None,
    entry_date: Optional[str] = None,
    risk_pct: Optional[float] = None,
    notes: str = ""
) -> Dict[str, Any]:
    """Adds a new open EP position to the portfolio."""
    port = load_portfolio()
    sym = symbol.strip().upper()
    now_str = dt.datetime.now().strftime("%Y-%m-%d")

    entry_px = round(float(entry_price), 2)
    stop_px = round(float(stop_price), 2)
    n_shares = max(1, int(shares))

    if stop_px >= entry_px:
        raise ValueError("Stop loss price must be below entry price for long EP position.")

    risk_per_share = round(entry_px - stop_px, 2)
    initial_risk_dollars = round(risk_per_share * n_shares, 2)

    # Fetch symbol metadata & event bars
    company_name = sym
    try:
        u_df = universe.load_universe()
        if u_df is not None and "symbol" in u_df.columns and "name" in u_df.columns:
            m = u_df[u_df["symbol"] == sym]
            if len(m) > 0 and pd.notna(m["name"].iloc[0]):
                company_name = str(m["name"].iloc[0])
    except Exception:
        pass
    sec = labels.sector(sym) or "Unknown Sector"
    thms = labels.themes(sym) or []
    theme = thms[0] if thms else "General"

    # Inspect bars to get Day 1 High / Low if event_date provided
    d1_high = entry_px
    d1_low = stop_px
    d1_close = entry_px
    d1_open = entry_px
    raw = datastore.load_bars(sym)
    if raw is not None and len(raw) > 0:
        d = indicators.add_indicators(raw)
        if event_date and event_date in d.index:
            r = d.loc[event_date]
            d1_high = round(float(r["high"]), 2)
            d1_low = round(float(r["low"]), 2)
            d1_close = round(float(r["close"]), 2)
            d1_open = round(float(r["open"]), 2)
        elif len(d) > 0:
            last = d.iloc[-1]
            d1_high = round(float(last["high"]), 2)
            d1_low = round(float(last["low"]), 2)
            d1_close = round(float(last["close"]), 2)
            d1_open = round(float(last["open"]), 2)

    pos_id = f"pos_{sym}_{int(time.time())}"
    position = {
        "id": pos_id,
        "symbol": sym,
        "company": company_name,
        "sector": sec,
        "theme": theme,
        "event_date": event_date or now_str,
        "entry_date": entry_date or now_str,
        "entry_price": entry_px,
        "shares": n_shares,
        "initial_shares": n_shares,
        "stop_price": stop_px,
        "initial_stop": stop_px,
        "d1_high": d1_high,
        "d1_low": d1_low,
        "d1_close": d1_close,
        "d1_open": d1_open,
        "risk_per_share": risk_per_share,
        "initial_risk_dollars": initial_risk_dollars,
        "risk_pct": risk_pct or port["settings"]["default_risk_pct"],
        "has_d2_add": False,
        "adds": [],
        "trailing_rule": "ema20",
        "notes": notes,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat()
    }

    port["positions"].append(position)
    save_portfolio(port)
    return position


def update_position(
    pos_id: str,
    action: str,
    stop_price: Optional[float] = None,
    entry_price: Optional[float] = None,
    add_price: Optional[float] = None,
    add_shares: Optional[int] = None,
    add_stop: Optional[float] = None,
    tranche_stops: Optional[List[float]] = None,
    trailing_rule: Optional[str] = None,
    notes: Optional[str] = None
) -> Dict[str, Any]:
    """Modifies an existing open position (adjust stop, execute secondary add, update notes)."""
    port = load_portfolio()
    pos = None
    for p in port["positions"]:
        if p["id"] == pos_id:
            pos = p
            break

    if not pos:
        raise ValueError(f"Position {pos_id} not found.")

    if action == "adjust_stop" and stop_price is not None:
        new_stop = round(float(stop_price), 2)
        pos["stop_price"] = new_stop
        if notes:
            pos["notes"] = (pos.get("notes", "") + f" | Stop moved to ${new_stop:.2f}: {notes}").strip(" |")

    elif action == "modify_position":
        if entry_price is not None:
            pos["entry_price"] = round(float(entry_price), 2)
        if stop_price is not None:
            pos["stop_price"] = round(float(stop_price), 2)
        if add_shares is not None:
            pos["shares"] = max(1, int(add_shares))

        # Update specific add tranche stops if provided
        if tranche_stops and "adds" in pos:
            for idx, st in enumerate(tranche_stops):
                if idx < len(pos["adds"]):
                    pos["adds"][idx]["stop_price"] = round(float(st), 2)

        risk_per_share = round(pos["entry_price"] - pos["stop_price"], 2)
        pos["risk_per_share"] = risk_per_share
        pos["initial_risk_dollars"] = round(risk_per_share * pos["shares"], 2)
        if notes:
            pos["notes"] = (pos.get("notes", "") + f" | Position modified: {notes}").strip(" |")

    elif action == "secondary_add":
        if add_price is None or add_shares is None:
            raise ValueError("add_price and add_shares required for secondary add.")
        add_px = round(float(add_price), 2)
        n_add = int(add_shares)
        if n_add <= 0:
            raise ValueError("add_shares must be > 0.")

        old_shares = pos["shares"]
        old_px = pos["entry_price"]
        # Blended average entry
        new_total_shares = old_shares + n_add
        blended_entry = round(((old_px * old_shares) + (add_px * n_add)) / new_total_shares, 2)

        resolved_add_stop = round(float(add_stop), 2) if add_stop is not None else pos.get("stop_price", pos.get("initial_stop", blended_entry))
        pos["shares"] = new_total_shares
        pos["entry_price"] = blended_entry
        pos["has_d2_add"] = True

        add_entry = {
            "date": dt.datetime.now().strftime("%Y-%m-%d"),
            "price": add_px,
            "shares": n_add,
            "stop_price": resolved_add_stop,
            "type": "D2_ADD"
        }
        pos["adds"].append(add_entry)

        # Update initial_risk_dollars to reflect base risk plus add risk
        add_risk_dollars = max(0.0, (add_px - resolved_add_stop) * n_add)
        pos["initial_risk_dollars"] = round(float(pos.get("initial_risk_dollars", 0.0)) + add_risk_dollars, 2)

        if notes:
            pos["notes"] = (pos.get("notes", "") + f" | Executed +{n_add} add @ ${add_px:.2f} (Stop ${resolved_add_stop:.2f}): {notes}").strip(" |")

    if trailing_rule is not None:
        pos["trailing_rule"] = trailing_rule

    save_portfolio(port)
    return pos


def close_position(
    pos_id: str,
    exit_price: float,
    exit_date: Optional[str] = None,
    exit_reason: str = "Manual Close",
    notes: str = ""
) -> Dict[str, Any]:
    """Closes an open position, records realized P&L and R-multiples, and archives it."""
    port = load_portfolio()
    pos = None
    idx = -1
    for i, p in enumerate(port["positions"]):
        if p["id"] == pos_id:
            pos = p
            idx = i
            break

    if not pos:
        raise ValueError(f"Position {pos_id} not found.")

    exit_px = round(float(exit_price), 2)
    entry_px = float(pos["entry_price"])
    shares = int(pos["shares"])
    initial_risk = max(1.0, float(pos.get("initial_risk_dollars", 1000.0)))

    realized_pnl_dollars = round((exit_px - entry_px) * shares, 2)
    realized_r = round(realized_pnl_dollars / initial_risk, 2)
    return_pct = round(((exit_px / entry_px) - 1.0) * 100.0, 2)

    exit_dt_str = exit_date or dt.datetime.now().strftime("%Y-%m-%d")

    # Calculate hold days
    hold_days = 1
    try:
        d_entry = dt.datetime.strptime(pos["entry_date"], "%Y-%m-%d")
        d_exit = dt.datetime.strptime(exit_dt_str, "%Y-%m-%d")
        hold_days = max(1, (d_exit - d_entry).days)
    except Exception:
        pass

    closed_pos = {
        **pos,
        "exit_price": exit_px,
        "exit_date": exit_dt_str,
        "exit_reason": exit_reason,
        "realized_pnl_dollars": realized_pnl_dollars,
        "realized_r": realized_r,
        "return_pct": return_pct,
        "hold_days": hold_days,
        "close_notes": notes,
        "closed_at": dt.datetime.now(dt.timezone.utc).isoformat()
    }

    port["positions"].pop(idx)
    port["closed_positions"].insert(0, closed_pos)
    save_portfolio(port)
    return closed_pos


def delete_closed_position(pos_id: str) -> bool:
    """Deletes a closed position by id from the historical ledger and updates the portfolio tally."""
    port = load_portfolio()
    closed = port.get("closed_positions", [])
    idx = next((i for i, c in enumerate(closed) if c.get("id") == pos_id), None)
    if idx is None:
        return False
    closed.pop(idx)
    port["closed_positions"] = closed
    save_portfolio(port)
    return True


def clear_all_closed_positions() -> int:
    """Clears all closed positions from the historical ledger and resets tally."""
    port = load_portfolio()
    count = len(port.get("closed_positions", []))
    port["closed_positions"] = []
    save_portfolio(port)
    return count


def evaluate_portfolio() -> Dict[str, Any]:
    """Evaluates all open positions against latest market bars and generates real-time execution alerts."""
    port = load_portfolio()
    settings = port.get("settings", DEFAULT_PORTFOLIO["settings"])
    portfolio_size = float(settings.get("portfolio_size", 100000.0))

    evaluated_positions = []
    alerts = []

    for pos in port.get("positions", []):
        sym = pos["symbol"]
        raw = datastore.load_bars(sym)
        if raw is None or len(raw) < 5:
            # Fallback if no bars
            evaluated_positions.append({
                **pos,
                "current_price": pos["entry_price"],
                "unrealized_pnl_dollars": 0.0,
                "unrealized_pnl_pct": 0.0,
                "open_r": 0.0,
                "current_risk_dollars": pos.get("initial_risk_dollars", 0.0),
                "current_risk_r": 1.0,
                "status_stage": "Holding",
                "larsson_state": "gray"
            })
            continue

        d = indicators.add_indicators(raw)
        d = scanner_core.calc_larssson_line(d)

        ev_dt = pos.get("event_date", "")
        entry_dt = pos.get("entry_date", "")
        
        # Determine bar count since entry
        bars_since = 0
        if entry_dt in d.index:
            entry_idx = d.index.get_loc(entry_dt)
            bars_since = len(d) - 1 - entry_idx
        elif ev_dt in d.index:
            ev_idx = d.index.get_loc(ev_dt)
            bars_since = len(d) - 1 - ev_idx
        else:
            bars_since = min(10, len(d) - 1)

        last_bar = d.iloc[-1]
        current_px = round(float(last_bar["close"]), 2)
        today_high = round(float(last_bar["high"]), 2)
        today_low = round(float(last_bar["low"]), 2)
        ema20 = round(float(last_bar["ema20"]), 2) if "ema20" in last_bar and pd.notna(last_bar["ema20"]) else None
        sma50 = round(float(last_bar["sma50"]), 2) if "sma50" in last_bar and pd.notna(last_bar["sma50"]) else None
        larsson_state = str(last_bar.get("larsson_state", "gray"))

        entry_px = float(pos["entry_price"])
        stop_px = float(pos["stop_price"])
        shares = int(pos["shares"])
        d1_high = float(pos.get("d1_high", entry_px))
        initial_risk = max(1.0, float(pos.get("initial_risk_dollars", 1000.0)))

        invested_dollars = round(entry_px * shares, 2)
        current_val_dollars = round(current_px * shares, 2)
        unrealized_pnl_dollars = round((current_px - entry_px) * shares, 2)
        unrealized_pnl_pct = round(((current_px / entry_px) - 1.0) * 100.0, 2)
        open_r = round(unrealized_pnl_dollars / initial_risk, 2)

        # Multi-tranche risk calculation: handle base position + distinct add tranches
        adds_list = pos.get("adds", [])
        total_add_shares = sum(int(a.get("shares", 0)) for a in adds_list)
        base_shares = max(0, shares - total_add_shares)

        base_risk_dollars = max(0.0, (entry_px - stop_px) * base_shares) if current_px > stop_px else 0.0
        adds_risk_dollars = 0.0
        any_add_stop_breached = False
        breached_add_stop = 0.0

        for a in adds_list:
            a_sh = int(a.get("shares", 0))
            a_px = float(a.get("price", entry_px))
            a_stop = float(a.get("stop_price", stop_px))
            if today_low <= a_stop:
                any_add_stop_breached = True
                breached_add_stop = a_stop
            if current_px > a_stop:
                adds_risk_dollars += max(0.0, (a_px - a_stop) * a_sh)

        current_risk_dollars = round(base_risk_dollars + adds_risk_dollars, 2)
        current_risk_r = round(current_risk_dollars / initial_risk, 2)

        # Stage determination
        if today_low <= stop_px or any_add_stop_breached:
            status_stage = "🛑 Stop Breached"
        elif larsson_state == "blue":
            status_stage = "🔵 Larsson Exit"
        elif open_r >= 3.0:
            status_stage = "🚀 Power Runner"
        elif pos.get("has_d2_add", False):
            status_stage = "⚡ Scaled (+50%)"
        elif bars_since <= 2:
            status_stage = "🌱 Day 1-2 Ignition"
        else:
            status_stage = "📈 Trend Holding"

        # --- Automated Alerts Evaluation ---
        # 1. Hard Stop Violation (Base stop or Add tranche stop)
        if today_low <= stop_px:
            alerts.append({
                "type": "danger",
                "pos_id": pos["id"],
                "symbol": sym,
                "title": f"🛑 Stop Loss Breached on {sym}",
                "message": f"{sym} low (${today_low:.2f}) dropped below active stop (${stop_px:.2f}). Immediate exit required to cap risk.",
                "suggested_action": "close",
                "priority": 1,
                "price": stop_px
            })
        elif any_add_stop_breached:
            alerts.append({
                "type": "danger",
                "pos_id": pos["id"],
                "symbol": sym,
                "title": f"🛑 Add Tranche Stop Breached on {sym}",
                "message": f"{sym} low (${today_low:.2f}) breached add tranche stop (${breached_add_stop:.2f}). Cut add shares immediately.",
                "suggested_action": "adjust_stop",
                "priority": 1,
                "price": breached_add_stop
            })

        # 2. Day 2 High Breakout Add
        if bars_since in [1, 2, 3] and not pos.get("has_d2_add", False) and today_high >= d1_high:
            suggested_add_shares = max(1, int(shares * 0.5))
            alerts.append({
                "type": "action",
                "pos_id": pos["id"],
                "symbol": sym,
                "title": f"⚡ Day 2 Secondary Add Triggered on {sym}",
                "message": f"{sym} reached ${today_high:.2f}, crossing Day 1 High (${d1_high:.2f}). Institutional accumulation confirmed. Execute +50% add (+{suggested_add_shares} sh).",
                "suggested_action": "secondary_add",
                "priority": 2,
                "suggested_shares": suggested_add_shares,
                "breakout_level": d1_high
            })

        # 3. Larsson Blue Flip Exit
        if larsson_state == "blue":
            alerts.append({
                "type": "warning",
                "pos_id": pos["id"],
                "symbol": sym,
                "title": f"🔵 Larsson Blue Flip Exit on {sym}",
                "message": f"Daily Larsson momentum flipped BLUE on {sym}. Institutional momentum trend has broken. Protect profits or exit runner.",
                "suggested_action": "close",
                "priority": 3,
                "price": current_px
            })

        # 4. Trailing Warning: Below 20 EMA
        if bars_since >= 5 and ema20 is not None and current_px < ema20 and stop_px < ema20:
            alerts.append({
                "type": "info",
                "pos_id": pos["id"],
                "symbol": sym,
                "title": f"📉 20 EMA Violation on {sym}",
                "message": f"{sym} closed at ${current_px:.2f} below rising 20 EMA (${ema20:.2f}). Consider trailing stop closer.",
                "suggested_action": "adjust_stop",
                "priority": 4,
                "suggested_stop": ema20
            })

        # 5. Milestone Alert (+2.0R / +3.0R Free Trade / De-Risking)
        if open_r >= 2.5 and stop_px < entry_px:
            alerts.append({
                "type": "success",
                "pos_id": pos["id"],
                "symbol": sym,
                "title": f"🎯 Milestone / Free Trade on {sym} (+{open_r:.1f} R)",
                "message": f"{sym} reached +{open_r:.1f} R gain. 10-year study strongly recommends moving stop to Breakeven (${entry_px:.2f}) to eliminate trade drawdown risk.",
                "suggested_action": "adjust_stop",
                "priority": 5,
                "suggested_stop": entry_px
            })

        # 6. Institutional 50 SMA Baseline Violation
        if bars_since >= 5 and sma50 is not None and current_px < sma50 and stop_px < sma50:
            alerts.append({
                "type": "warning",
                "pos_id": pos["id"],
                "symbol": sym,
                "title": f"🏛️ 50 SMA Institutional Violation on {sym}",
                "message": f"{sym} closed below the institutional 50-day SMA (${sma50:.2f}). Multi-quarter leaders hold this level 70%+ of the time. Tighten trailing stop.",
                "suggested_action": "adjust_stop",
                "priority": 6,
                "suggested_stop": stop_px
            })

        # 7. 48-Hour Absorption Invalidation
        if bars_since in [2, 3] and "d1_close" in pos and "d1_low" in pos:
            d1_c = float(pos.get("d1_close", entry_px))
            d1_l = float(pos.get("d1_low", stop_px))
            half_body = (d1_c + d1_l) / 2.0
            if current_px < half_body:
                alerts.append({
                    "type": "warning",
                    "pos_id": pos["id"],
                    "symbol": sym,
                    "title": f"⚠️ 48-Hour Absorption Violated on {sym}",
                    "message": f"{sym} broke below Day 1 upper 50% body (${half_body:.2f}). High trap probability (37.5%). Cancel adds and keep tight stop at ${stop_px:.2f}.",
                    "suggested_action": "adjust_stop",
                    "priority": 7,
                    "suggested_stop": stop_px
                })

        evaluated_positions.append({
            **pos,
            "current_price": current_px,
            "today_high": today_high,
            "today_low": today_low,
            "invested_dollars": invested_dollars,
            "current_val_dollars": current_val_dollars,
            "unrealized_pnl_dollars": unrealized_pnl_dollars,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "open_r": open_r,
            "current_risk_dollars": current_risk_dollars,
            "current_risk_r": current_risk_r,
            "bars_since": bars_since,
            "ema20": ema20,
            "sma50": sma50,
            "larsson_state": larsson_state,
            "status_stage": status_stage
        })

    # Summary Metrics
    total_invested = round(sum(p["invested_dollars"] for p in evaluated_positions), 2)
    total_unrealized_pnl = round(sum(p["unrealized_pnl_dollars"] for p in evaluated_positions), 2)
    total_unrealized_r = round(sum(p["open_r"] for p in evaluated_positions), 2)

    total_open_risk_dollars = round(sum(p["current_risk_dollars"] for p in evaluated_positions), 2)
    default_risk_pct = float(settings.get("default_risk_pct", 1.0))
    port_1r_dollars = max(1.0, portfolio_size * (default_risk_pct / 100.0))
    total_open_risk_r = round(total_open_risk_dollars / port_1r_dollars, 2)
    open_heat_pct = round((total_open_risk_dollars / max(1000.0, portfolio_size)) * 100.0, 2)

    closed = port.get("closed_positions", [])
    total_realized_pnl = round(sum(c.get("realized_pnl_dollars", 0.0) for c in closed), 2)
    total_realized_r = round(sum(c.get("realized_r", 0.0) for c in closed), 2)
    closed_wins = [c for c in closed if c.get("realized_r", 0.0) > 0.05]
    win_rate = round(len(closed_wins) / len(closed) * 100.0, 1) if closed else 0.0

    account_equity = round(portfolio_size + total_realized_pnl + total_unrealized_pnl, 2)
    cash_available = round(max(0.0, account_equity - total_invested), 2)

    # Sort alerts by priority (1 is highest)
    alerts.sort(key=lambda a: a["priority"])

    return {
        "settings": settings,
        "summary": {
            "portfolio_size": portfolio_size,
            "account_equity": account_equity,
            "cash_available": cash_available,
            "total_invested": total_invested,
            "unrealized_pnl": total_unrealized_pnl,
            "unrealized_r": total_unrealized_r,
            "realized_pnl": total_realized_pnl,
            "realized_r": total_realized_r,
            "open_heat_dollars": total_open_risk_dollars,
            "open_heat_r": total_open_risk_r,
            "open_heat_pct": open_heat_pct,
            "open_count": len(evaluated_positions),
            "closed_count": len(closed),
            "win_rate": win_rate,
            "active_alerts_count": len(alerts)
        },
        "positions": evaluated_positions,
        "closed_positions": closed,
        "alerts": alerts
    }
