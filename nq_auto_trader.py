"""
nq_auto_trader.py — 15m MNQ Automated TWS Bracket Execution, Watchdog & Gmail Alert Engine
==========================================================================================
Implements automated 15-minute Micro E-mini Nasdaq-100 (`MNQ`, 2 contracts) execution via
Interactive Brokers TWS API (`ib_async`) with:
  1. Strict Single-Direction / One-Position-At-A-Time Guard (never goes Short while Long or vice versa).
  2. 2-MNQ Split GTC Outside-RTH Bracket Orders (`tif="GTC"`, `outsideRth=True`, exact `0.25` tick quantization):
     - Unit 1 (1 MNQ): Entry + OCA Group 1 (TP1 Limit + Stop-Limit)
     - Unit 2 (1 MNQ): Entry + OCA Group 2 (TP2 Runner Limit + Stop-Limit, auto-moved to Breakeven after TP1)
  3. Smart Position & Bracket Watchdog:
     - Verifies that any open MNQ position (`abs(live_net_pos) > 0`) always has a working protective bracket/stop in place.
     - Manual-Close Detection: If the user closes the position manually (`live_net_pos == 0`), immediately cancels any
       leftover child orders, transitions state to `FLAT`, and stops trying to place stop-losses on a zero position.
  4. Gmail Notification on Position Open:
     - Sends a formatted trade-opened email to the user's Gmail (`amir.baram@gmail.com` by default) whenever a position is opened.
"""

from __future__ import annotations

import datetime as dt
from email.message import EmailMessage
import json
import os
from pathlib import Path
import smtplib
import ssl
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import nq_ml_engine

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CONFIG_PATH = DATA_DIR / "nq_auto_trader_config.json"
STATE_PATH = DATA_DIR / "nq_auto_trader_state.json"
ENV_PATH = ROOT_DIR / ".env"

_AUTO_LOCK = threading.Lock()
_DAEMON_THREAD: Optional[threading.Thread] = None
_DAEMON_STOP_EVENT = threading.Event()


def round_to_nq_tick(px: float) -> float:
    """Quantizes any NQ/MNQ price to an exact 0.25 CME Globex tick increment."""
    return round(round(float(px) * 4.0) / 4.0, 2)


def round_to_mgc_tick(px: float) -> float:
    """Quantizes any GC/MGC price to an exact 0.10 COMEX tick increment."""
    return round(round(float(px) * 10.0) / 10.0, 2)


def _now_et_str() -> str:
    return dt.datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S ET")


def _load_dotenv_vars() -> Dict[str, str]:
    out: Dict[str, str] = {}
    if ENV_PATH.exists():
        try:
            for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
        except Exception:
            pass
    return out


DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "mgc_enabled": False,
    "timeframe": "15m",
    "symbol": "MNQ",
    "exchange": "CME",
    "mgc_symbol": "MGC",
    "mgc_exchange": "COMEX",
    "currency": "USD",
    "contracts": 2,
    "mgc_contracts": 2,
    "min_prob_pct": 55.0,
    "min_pred_ratio": 1.35,
    "min_rr_target1": 1.00,
    "tif": "GTC",
    "outside_rth": True,
    "entry_slippage_cap_pts": 2.50,
    "stop_limit_protection_pts": 20.00,
    "mgc_entry_slippage_cap_pts": 0.60,
    "mgc_stop_limit_protection_pts": 5.00,
    "entry_timeout_sec": 180,
    "time_stop_bars": 8,
    "watchdog_interval_sec": 5.0,
    "tws_ports": [7497, 7496, 4002, 4001],
    "gmail_enabled": True,
    "pivot_email_alerts_enabled": True,
    "pivot_email_min_prob_pct": 50.0,
    "pivot_email_tickers": ["NQ=F", "GC=F", "CL=F", "BTC=F", "SI=F"],
    "pivot_email_timeframes": ["5m", "15m", "1h", "4h"],
    "gmail_to": "amir.baram@gmail.com",
    "gmail_user": "amir.baram@gmail.com",
    "gmail_app_password": "",
}


def load_auto_trader_config() -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    env_vars = _load_dotenv_vars()
    if env_vars.get("GMAIL_TO"):
        cfg["gmail_to"] = env_vars["GMAIL_TO"]
    if env_vars.get("GMAIL_USER"):
        cfg["gmail_user"] = env_vars["GMAIL_USER"]
    if env_vars.get("GMAIL_APP_PASSWORD"):
        cfg["gmail_app_password"] = env_vars["GMAIL_APP_PASSWORD"]

    if CONFIG_PATH.exists():
        try:
            saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                for k, v in saved.items():
                    if k == "gmail_app_password" and not v and cfg.get("gmail_app_password"):
                        continue
                    cfg[k] = v
        except Exception:
            pass
    # Enforce 15m single-direction 2 MNQ / 2 MGC GTC outsideRth invariants
    cfg["timeframe"] = "15m"
    cfg["contracts"] = int(cfg.get("contracts") or 2)
    cfg["mgc_contracts"] = int(cfg.get("mgc_contracts") or 2)
    cfg["mgc_enabled"] = bool(cfg.get("mgc_enabled", False))
    cfg["pivot_email_alerts_enabled"] = bool(cfg.get("pivot_email_alerts_enabled", True))
    cfg["pivot_email_tickers"] = list(cfg.get("pivot_email_tickers") or ["NQ=F", "GC=F", "CL=F", "BTC=F", "SI=F"])
    cfg["pivot_email_timeframes"] = list(cfg.get("pivot_email_timeframes") or ["5m", "15m", "1h", "4h"])
    cfg["tif"] = "GTC"
    cfg["outside_rth"] = True
    return cfg


def save_auto_trader_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    with _AUTO_LOCK:
        cfg = load_auto_trader_config()
        for k, v in updates.items():
            if k in DEFAULT_CONFIG and v is not None:
                if k == "gmail_app_password" and str(v).strip() == "********":
                    continue
                cfg[k] = v
        if "pivot_email_tickers" in updates and isinstance(updates["pivot_email_tickers"], list):
            cfg["pivot_email_tickers"] = [str(x).strip() for x in updates["pivot_email_tickers"]]
        if "pivot_email_timeframes" in updates and isinstance(updates["pivot_email_timeframes"], list):
            cfg["pivot_email_timeframes"] = [str(x).strip().lower() for x in updates["pivot_email_timeframes"]]
        cfg["timeframe"] = "15m"
        cfg["mgc_enabled"] = bool(cfg.get("mgc_enabled", False))
        cfg["pivot_email_alerts_enabled"] = bool(cfg.get("pivot_email_alerts_enabled", True))
        cfg["tif"] = "GTC"
        cfg["outside_rth"] = True
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        return cfg


def _default_state() -> Dict[str, Any]:
    return {
        "position_state": "FLAT",  # FLAT | PENDING_ENTRY | OPEN
        "active_direction": None,  # LONG | SHORT | None
        "live_net_pos": 0,
        "working_stop_qty": 0,
        "working_tp_qty": 0,
        "active_trade": None,
        "traded_bar_keys": [],
        "mgc_position_state": "FLAT",
        "mgc_active_direction": None,
        "mgc_live_net_pos": 0,
        "mgc_working_stop_qty": 0,
        "mgc_working_tp_qty": 0,
        "mgc_active_trade": None,
        "mgc_traded_bar_keys": [],
        "mgc_watchdog_status": "OFF (MGC Auto-Trader Disabled)",
        "trade_history": [],
        "alerted_pivot_keys": [],
        "watchdog_status": "IDLE (Flat — 0 MNQ)",
        "last_watchdog_ts": None,
        "last_action": "Initialized",
        "events_log": [],
        "email_log": [],
    }


def load_auto_trader_state() -> Dict[str, Any]:
    st = _default_state()
    if STATE_PATH.exists():
        try:
            saved = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                st.update(saved)
        except Exception:
            pass
    return st


def _save_state_unlocked(st: Dict[str, Any]) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        st["events_log"] = list(st.get("events_log") or [])[-60:]
        st["email_log"] = list(st.get("email_log") or [])[-30:]
        st["traded_bar_keys"] = list(st.get("traded_bar_keys") or [])[-200:]
        st["mgc_traded_bar_keys"] = list(st.get("mgc_traded_bar_keys") or [])[-200:]
        st["alerted_pivot_keys"] = list(st.get("alerted_pivot_keys") or [])[-400:]
        st["trade_history"] = list(st.get("trade_history") or [])[-200:]
        STATE_PATH.write_text(json.dumps(st, indent=2), encoding="utf-8")
    except Exception:
        pass


def save_auto_trader_state(st: Dict[str, Any]) -> None:
    with _AUTO_LOCK:
        _save_state_unlocked(st)


def _append_log(st: Dict[str, Any], event_type: str, message: str) -> None:
    entry = {
        "time": _now_et_str(),
        "type": event_type,
        "message": message,
    }
    logs = list(st.get("events_log") or [])
    logs.append(entry)
    st["events_log"] = logs[-60:]
    st["last_action"] = f"[{entry['time']}] {message}"


def _record_closed_trade(
    st: Dict[str, Any],
    active_tr: Dict[str, Any],
    exit_px: float,
    exit_reason: str,
) -> Dict[str, Any]:
    """
    Records a completed live 2-MNQ or 2-MGC trade into `st['trade_history']` so live win rate,
    trades/day, and net points/$ P&L can be tracked and compared against the OOS model expectation.
    """
    symbol = str(active_tr.get("symbol", "MNQ") or "MNQ").upper()
    is_mgc = symbol in ("MGC", "GC", "GC=F", "MGC=F")
    direction = str(active_tr.get("direction", "LONG")).upper()
    sign = -1.0 if direction == "SHORT" else 1.0
    entry_px = float(active_tr.get("entry_px", 0.0) or 0.0)
    tp1_px = float(active_tr.get("target1_px", entry_px) or entry_px)
    tp1_filled = bool(active_tr.get("tp1_filled", False))
    fin_px = round_to_mgc_tick(float(exit_px or entry_px)) if is_mgc else round_to_nq_tick(float(exit_px or entry_px))

    max_filled_units = int(active_tr.get("max_abs_pos", 2) or 2)
    if tp1_filled:
        u1_pts = sign * (tp1_px - entry_px)
        u2_pts = sign * (fin_px - entry_px)
        n_units = 2
    elif max_filled_units == 1:
        u1_pts = sign * (fin_px - entry_px)
        u2_pts = 0.0
        n_units = 1
    else:
        u1_pts = sign * (fin_px - entry_px)
        u2_pts = sign * (fin_px - entry_px)
        n_units = 2

    total_pts_1c_equiv = round(u1_pts + u2_pts, 2)
    avg_pts_per_contract = round((u1_pts + u2_pts) / float(n_units), 2)
    # Each MNQ contract is $2.00 per NQ point; each MGC contract is $10.00 per $1.00 gold move (10 troy oz)
    pt_mult = 10.0 if is_mgc else 2.0
    pnl_usd = round((u1_pts + u2_pts) * pt_mult, 2)
    is_win = bool(pnl_usd > 0.0)

    opened_at = str(active_tr.get("opened_at") or active_tr.get("bar_time") or _now_et_str())
    closed_at = _now_et_str()
    rec = {
        "symbol": "MGC" if is_mgc else "MNQ",
        "opened_at": opened_at,
        "closed_at": closed_at,
        "date": closed_at[:10],
        "direction": direction,
        "entry_px": entry_px,
        "exit_px": fin_px,
        "tp1_filled": tp1_filled,
        "exit_reason": exit_reason,
        "avg_pts": avg_pts_per_contract,
        "total_contract_pts": total_pts_1c_equiv,
        "pnl_usd": pnl_usd,
        "is_win": is_win,
        "prob_pct": float(active_tr.get("prob_pct", 0.0) or 0.0),
        "pred_ratio": float(active_tr.get("pred_ratio", 0.0) or 0.0),
    }
    hist = list(st.get("trade_history") or [])
    hist.append(rec)
    st["trade_history"] = hist[-200:]
    return rec


def compute_live_performance_summary(st: Dict[str, Any]) -> Dict[str, Any]:
    """Computes live trade count, Win Rate %, Trades/Day, Profit Factor, and Total $ P&L on 2 MNQ."""
    hist = list(st.get("trade_history") or [])
    n = len(hist)
    if n == 0:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "trades_per_day": 0.0,
            "total_pnl_usd": 0.0,
            "avg_pts": 0.0,
            "profit_factor": 0.0,
            "recent_trades": [],
        }
    wins = sum(1 for r in hist if r.get("is_win"))
    losses = n - wins
    win_rate_pct = round(100.0 * wins / n, 1)
    unique_days = max(1, len({str(r.get("date", ""))[:10] for r in hist if r.get("date")}))
    trades_per_day = round(float(n) / float(unique_days), 2)
    total_pnl_usd = round(sum(float(r.get("pnl_usd", 0.0) or 0.0) for r in hist), 2)
    avg_pts = round(sum(float(r.get("avg_pts", 0.0) or 0.0) for r in hist) / n, 2)
    gross_win = sum(float(r.get("pnl_usd", 0.0)) for r in hist if float(r.get("pnl_usd", 0.0)) > 0)
    gross_loss = abs(sum(float(r.get("pnl_usd", 0.0)) for r in hist if float(r.get("pnl_usd", 0.0)) < 0))
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else (99.9 if gross_win > 0 else 0.0)
    return {
        "total_trades": n,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": win_rate_pct,
        "trades_per_day": trades_per_day,
        "total_pnl_usd": total_pnl_usd,
        "avg_pts": avg_pts,
        "profit_factor": pf,
        "recent_trades": list(reversed(hist[-5:])),
    }


def build_2mnq_gtc_bracket_spec(
    plan: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    oca_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Builds the exact 2-MNQ Split GTC Outside-RTH Bracket Order Specification:
      - All prices quantized to exact 0.25 CME Globex ticks.
      - Every order leg has `tif="GTC"` and `outsideRth=True`.
      - Split into two independent 1-MNQ OCA sub-brackets (`Unit 1` for TP1 and `Unit 2` for TP2 Runner)
        so that when TP1 fills 1 contract, Unit 2's 1-MNQ Stop-Loss remains 100% active on CME Globex.
    """
    c = cfg or load_auto_trader_config()
    direction = str(plan.get("direction", "LONG")).upper()
    is_short = direction == "SHORT"
    entry_action = "SELL" if is_short else "BUY"
    exit_action = "BUY" if is_short else "SELL"

    entry_px = round_to_nq_tick(float(plan.get("entry_px") or plan.get("entry_price") or 0.0))
    stop_px = round_to_nq_tick(float(plan.get("stop_px") or plan.get("stop_price") or 0.0))
    tp1_px = round_to_nq_tick(float(plan.get("target1_px") or plan.get("target1_price") or 0.0))
    tp2_px = round_to_nq_tick(float(plan.get("target2_px") or plan.get("target2_price") or 0.0))

    slip_cap = round_to_nq_tick(float(c.get("entry_slippage_cap_pts", 2.50)))
    stop_prot = round_to_nq_tick(float(c.get("stop_limit_protection_pts", 20.00)))

    # Marketable limit entry price (allows immediate fill on discovery up to slip_cap)
    entry_lmt_px = round_to_nq_tick(entry_px - slip_cap if is_short else entry_px + slip_cap)
    # CME Globex native Stop-Limit protection price
    stop_lmt_px = round_to_nq_tick(stop_px + stop_prot if is_short else stop_px - stop_prot)

    prefix = oca_prefix or f"MNQ15M_{int(time.time())}"
    oca_u1 = f"{prefix}_U1"
    oca_u2 = f"{prefix}_U2"

    unit1 = {
        "unit": 1,
        "qty": 1,
        "oca_group": oca_u1,
        "parent_entry": {
            "action": entry_action,
            "totalQuantity": 1,
            "orderType": "LMT",
            "lmtPrice": entry_lmt_px,
            "tif": "GTC",
            "outsideRth": True,
            "transmit": False,
        },
        "take_profit": {
            "action": exit_action,
            "totalQuantity": 1,
            "orderType": "LMT",
            "lmtPrice": tp1_px,
            "tif": "GTC",
            "outsideRth": True,
            "ocaGroup": oca_u1,
            "ocaType": 1,
            "transmit": False,
        },
        "stop_loss": {
            "action": exit_action,
            "totalQuantity": 1,
            "orderType": "STP LMT",
            "auxPrice": stop_px,
            "lmtPrice": stop_lmt_px,
            "tif": "GTC",
            "outsideRth": True,
            "ocaGroup": oca_u1,
            "ocaType": 1,
            "transmit": True,
        },
    }

    unit2 = {
        "unit": 2,
        "qty": 1,
        "oca_group": oca_u2,
        "parent_entry": {
            "action": entry_action,
            "totalQuantity": 1,
            "orderType": "LMT",
            "lmtPrice": entry_lmt_px,
            "tif": "GTC",
            "outsideRth": True,
            "transmit": False,
        },
        "take_profit": {
            "action": exit_action,
            "totalQuantity": 1,
            "orderType": "LMT",
            "lmtPrice": tp2_px,
            "tif": "GTC",
            "outsideRth": True,
            "ocaGroup": oca_u2,
            "ocaType": 1,
            "transmit": False,
        },
        "stop_loss": {
            "action": exit_action,
            "totalQuantity": 1,
            "orderType": "STP LMT",
            "auxPrice": stop_px,
            "lmtPrice": stop_lmt_px,
            "tif": "GTC",
            "outsideRth": True,
            "ocaGroup": oca_u2,
            "ocaType": 1,
            "transmit": True,
        },
    }

    return {
        "symbol": "MNQ",
        "exchange": "CME",
        "timeframe": "15m",
        "direction": direction,
        "total_contracts": 2,
        "entry_action": entry_action,
        "exit_action": exit_action,
        "entry_px": entry_px,
        "entry_lmt_px": entry_lmt_px,
        "stop_px": stop_px,
        "stop_lmt_px": stop_lmt_px,
        "target1_px": tp1_px,
        "target2_px": tp2_px,
        "tif": "GTC",
        "outsideRth": True,
        "bar_time": str(plan.get("bar_time", "")),
        "bar_timestamp": int(plan.get("bar_timestamp", 0) or 0),
        "prob_pct": float(plan.get("prob_pct", 0.0)),
        "pred_ratio": float(plan.get("pred_ratio", 0.0)),
        "rr_target1": float(plan.get("rr_target1", 0.0)),
        "rr_target2": float(plan.get("rr_target2", 0.0)),
        "units": [unit1, unit2],
    }


def build_2mgc_gtc_bracket_spec(
    plan: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    oca_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Builds the exact 2-MGC Split GTC Outside-RTH Bracket Order Specification:
      - All prices quantized to exact 0.10 COMEX ticks ($10.00 per $1.00 move on 1 MGC).
      - Every order leg has `tif="GTC"` and `outsideRth=True`.
      - Split into two independent 1-MGC OCA sub-brackets (`Unit 1` for TP1 and `Unit 2` for TP2 Runner)
        so that when TP1 fills 1 contract, Unit 2's 1-MGC Stop-Loss remains 100% active on COMEX.
    """
    c = cfg or load_auto_trader_config()
    direction = str(plan.get("direction", "LONG")).upper()
    is_short = direction == "SHORT"
    entry_action = "SELL" if is_short else "BUY"
    exit_action = "BUY" if is_short else "SELL"

    entry_px = round_to_mgc_tick(float(plan.get("entry_px") or plan.get("entry_price") or 0.0))
    stop_px = round_to_mgc_tick(float(plan.get("stop_px") or plan.get("stop_price") or 0.0))
    tp1_px = round_to_mgc_tick(float(plan.get("target1_px") or plan.get("target1_price") or 0.0))
    tp2_px = round_to_mgc_tick(float(plan.get("target2_px") or plan.get("target2_price") or 0.0))

    slip_cap = round_to_mgc_tick(float(c.get("mgc_entry_slippage_cap_pts", 0.60)))
    stop_prot = round_to_mgc_tick(float(c.get("mgc_stop_limit_protection_pts", 5.00)))

    entry_lmt_px = round_to_mgc_tick(entry_px - slip_cap if is_short else entry_px + slip_cap)
    stop_lmt_px = round_to_mgc_tick(stop_px + stop_prot if is_short else stop_px - stop_prot)

    prefix = oca_prefix or f"MGC15M_{int(time.time())}"
    oca_u1 = f"{prefix}_U1"
    oca_u2 = f"{prefix}_U2"

    unit1 = {
        "unit": 1,
        "qty": 1,
        "oca_group": oca_u1,
        "parent_entry": {
            "action": entry_action,
            "totalQuantity": 1,
            "orderType": "LMT",
            "lmtPrice": entry_lmt_px,
            "tif": "GTC",
            "outsideRth": True,
            "transmit": False,
        },
        "take_profit": {
            "action": exit_action,
            "totalQuantity": 1,
            "orderType": "LMT",
            "lmtPrice": tp1_px,
            "tif": "GTC",
            "outsideRth": True,
            "ocaGroup": oca_u1,
            "ocaType": 1,
            "transmit": False,
        },
        "stop_loss": {
            "action": exit_action,
            "totalQuantity": 1,
            "orderType": "STP LMT",
            "auxPrice": stop_px,
            "lmtPrice": stop_lmt_px,
            "tif": "GTC",
            "outsideRth": True,
            "ocaGroup": oca_u1,
            "ocaType": 1,
            "transmit": True,
        },
    }

    unit2 = {
        "unit": 2,
        "qty": 1,
        "oca_group": oca_u2,
        "parent_entry": {
            "action": entry_action,
            "totalQuantity": 1,
            "orderType": "LMT",
            "lmtPrice": entry_lmt_px,
            "tif": "GTC",
            "outsideRth": True,
            "transmit": False,
        },
        "take_profit": {
            "action": exit_action,
            "totalQuantity": 1,
            "orderType": "LMT",
            "lmtPrice": tp2_px,
            "tif": "GTC",
            "outsideRth": True,
            "ocaGroup": oca_u2,
            "ocaType": 1,
            "transmit": False,
        },
        "stop_loss": {
            "action": exit_action,
            "totalQuantity": 1,
            "orderType": "STP LMT",
            "auxPrice": stop_px,
            "lmtPrice": stop_lmt_px,
            "tif": "GTC",
            "outsideRth": True,
            "ocaGroup": oca_u2,
            "ocaType": 1,
            "transmit": True,
        },
    }

    return {
        "symbol": "MGC",
        "exchange": "COMEX",
        "timeframe": "15m",
        "direction": direction,
        "total_contracts": 2,
        "entry_action": entry_action,
        "exit_action": exit_action,
        "entry_px": entry_px,
        "entry_lmt_px": entry_lmt_px,
        "stop_px": stop_px,
        "stop_lmt_px": stop_lmt_px,
        "target1_px": tp1_px,
        "target2_px": tp2_px,
        "tif": "GTC",
        "outsideRth": True,
        "bar_time": str(plan.get("bar_time", "")),
        "bar_timestamp": int(plan.get("bar_timestamp", 0) or 0),
        "prob_pct": float(plan.get("prob_pct", 0.0)),
        "pred_ratio": float(plan.get("pred_ratio", 0.0)),
        "rr_target1": float(plan.get("rr_target1", 0.0)),
        "rr_target2": float(plan.get("rr_target2", 0.0)),
        "units": [unit1, unit2],
    }


def _dispatch_email_message(
    to_addr: str,
    from_user: str,
    app_pw: str,
    subject: str,
    body_text: str,
) -> Dict[str, Any]:
    """Delivers an email via Gmail SMTP SSL (if `app_pw` is configured) or macOS Mail.app (`amir.baram@gmail.com`)."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_user
    msg["To"] = to_addr
    msg.set_content(body_text)

    if app_pw:
        try:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context, timeout=8.0) as server:
                server.login(from_user, app_pw)
                server.send_message(msg)
            return {
                "sent": True,
                "to": to_addr,
                "subject": subject,
                "status": f"SENT via Gmail SMTP to {to_addr}",
                "time": _now_et_str(),
            }
        except Exception as e:
            return {
                "sent": False,
                "to": to_addr,
                "subject": subject,
                "status": f"SMTP_ERROR: {e}",
                "time": _now_et_str(),
            }

    # Fallback: Send directly via macOS Mail.app (logged into amir.baram@gmail.com)
    try:
        import subprocess

        apple_script = """
        on run argv
            set recipAddress to item 1 of argv
            set mailSubject to item 2 of argv
            set mailBody to item 3 of argv
            tell application "Mail"
                set newMessage to make new outgoing message with properties {subject:mailSubject, content:mailBody, visible:false}
                tell newMessage
                    make new to recipient at end of to recipients with properties {address:recipAddress}
                    try
                        set sender to "amir.baram@gmail.com"
                    end try
                    send
                end tell
            end tell
        end run
        """
        proc = subprocess.run(
            ["osascript", "-e", apple_script, to_addr, subject, body_text],
            capture_output=True,
            text=True,
            timeout=12.0,
        )
        if proc.returncode == 0:
            return {
                "sent": True,
                "to": to_addr,
                "subject": subject,
                "status": f"SENT via Apple Mail (amir.baram@gmail.com) to {to_addr}",
                "time": _now_et_str(),
            }
        return {
            "sent": False,
            "to": to_addr,
            "subject": subject,
            "status": f"MAIL_APP_ERROR: {proc.stderr.strip() or 'Failed'}",
            "time": _now_et_str(),
        }
    except Exception as e:
        return {
            "sent": False,
            "to": to_addr,
            "subject": subject,
            "status": f"QUEUED_NO_APP_PASSWORD ({e})",
            "time": _now_et_str(),
        }


def send_position_opened_gmail(
    trade_info: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    is_test: bool = False,
) -> Dict[str, Any]:
    """
    Sends an email notification to the user's Gmail whenever a 2-MNQ or 2-MGC position is opened
    (or when triggered via the 'Send Test Email' button).
    """
    c = cfg or load_auto_trader_config()
    to_addr = str(c.get("gmail_to") or "amir.baram@gmail.com").strip()
    from_user = str(c.get("gmail_user") or to_addr).strip()
    app_pw = str(c.get("gmail_app_password") or os.environ.get("GMAIL_APP_PASSWORD", "")).strip()

    sym = str(trade_info.get("symbol", "MNQ") or "MNQ").upper()
    is_mgc = sym in ("MGC", "GC", "GC=F", "MGC=F")
    sym_lbl = "MGC" if is_mgc else "MNQ"
    exch_lbl = "Micro Gold Futures, COMEX" if is_mgc else "Micro E-mini Nasdaq-100 Futures, CME"
    pt_mult = 10.0 if is_mgc else 2.0

    direction = str(trade_info.get("direction", "LONG")).upper()
    entry_px = float(trade_info.get("entry_px", 29900.0))
    stop_px = float(trade_info.get("stop_px", 29800.0))
    tp1_px = float(trade_info.get("target1_px", 30050.0))
    tp2_px = float(trade_info.get("target2_px", 30150.0))
    prob_pct = float(trade_info.get("prob_pct", 60.0))
    pred_ratio = float(trade_info.get("pred_ratio", 1.65))
    bar_time = str(trade_info.get("bar_time") or _now_et_str())

    stop_pts = abs(entry_px - stop_px)
    tp1_pts = abs(tp1_px - entry_px)
    tp2_pts = abs(tp2_px - entry_px)
    risk_2c_usd = round(stop_pts * pt_mult * 2, 0)
    rew1_1c_usd = round(tp1_pts * pt_mult, 0)
    rew2_1c_usd = round(tp2_pts * pt_mult, 0)

    prefix = "[TEST] " if is_test else ""
    subject = (
        f"{prefix}🚨 {sym_lbl} [15M] {direction} 2 Contracts Opened @ {entry_px:.2f} | "
        f"SL {stop_px:.2f} | TP1 {tp1_px:.2f} | TP2 {tp2_px:.2f}"
    )

    body_text = (
        f"{sym_lbl} 15-Minute Automated Bracket Trade Opened\n"
        f"============================================\n"
        f"Time (ET):         {bar_time}\n"
        f"Contract:          2 {sym_lbl} ({exch_lbl})\n"
        f"Direction:         {direction} ({'SELL 2 ' + sym_lbl if direction == 'SHORT' else 'BUY 2 ' + sym_lbl})\n"
        f"Pivot Confidence:  {prob_pct:.1f}% (Expected MFE/MAE Ratio: {pred_ratio:.2f}x)\n\n"
        f"GTC Outside-RTH Bracket Orders Placed:\n"
        f"--------------------------------------\n"
        f"• Entry Price:     {entry_px:.2f} (LMT cap: {trade_info.get('entry_lmt_px', entry_px):.2f}, GTC, Outside RTH)\n"
        f"• Stop-Loss (2x):  {stop_px:.2f} ({stop_pts:.2f} pts / -${risk_2c_usd:.0f} total risk on 2 {sym_lbl}, GTC, Outside RTH)\n"
        f"• Target 1 (1x):   {tp1_px:.2f} (+{tp1_pts:.2f} pts / +${rew1_1c_usd:.0f} on Unit 1 -> moves Unit 2 Stop to BE)\n"
        f"• Target 2 (1x):   {tp2_px:.2f} (+{tp2_pts:.2f} pts / +${rew2_1c_usd:.0f} on Unit 2 Runner)\n\n"
        f"Safety Watchdog Status:\n"
        f"• Single-Direction Guard: ACTIVE (No opposing signals allowed while position is open)\n"
        f"• Bracket Watchdog: ACTIVE (Verifying working GTC stop-loss on open contracts; auto-disarms if you close manually)\n"
    )
    return _dispatch_email_message(to_addr, from_user, app_pw, subject, body_text)


def send_realtime_pivot_alert_gmail(
    alert_or_plan: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    is_test: bool = False,
) -> Dict[str, Any]:
    """
    Sends a real-time pivot detection email to the user's Gmail whenever any monitored ticker
    (NQ=F/MNQ=F, GC=F/MGC=F, CL=F/MCL=F, BTC=F/MBT=F across 15m/1h/4h, or any Portfolio/Watchlist ticker)
    triggers a new real-time ML pivot, complete with:
      - Conviction Level (ML Pivot Probability %, At-Extreme %, Conviction Tier, Predicted MFE/MAE Ratio)
      - MAE Levels (Expected MAE price/pts/ATR, 75th-Percentile Q75 MAE price/pts/ATR, Stop-Loss price/pts/ATR)
      - MFE Levels (Expected MFE price/pts/ATR, 70th-Percentile Q70 MFE price/pts/ATR, Target 1 TP1 & Target 2 TP2)
    """
    c = cfg or load_auto_trader_config()
    to_addr = str(c.get("gmail_to") or "amir.baram@gmail.com").strip()
    from_user = str(c.get("gmail_user") or to_addr).strip()
    app_pw = str(c.get("gmail_app_password") or os.environ.get("GMAIL_APP_PASSWORD", "")).strip()

    plan = dict(alert_or_plan.get("trade_plan") or {})
    merged: Dict[str, Any] = {**plan, **alert_or_plan}
    if plan:
        for k_p, v_p in plan.items():
            if merged.get(k_p) is None:
                merged[k_p] = v_p

    sym = str(merged.get("symbol") or "MNQ").upper()
    tf_label = str(merged.get("timeframe_label") or merged.get("timeframe") or "1D").upper()
    is_high_piv = bool(merged.get("is_high_pivot", str(merged.get("direction", "")).upper() == "SHORT"))
    direction = str(merged.get("direction") or ("SHORT" if is_high_piv else "LONG")).upper()
    piv_label = str(
        merged.get("pivot_label")
        or merged.get("action_title")
        or ("Pivot High (SHORT)" if is_high_piv else "Pivot Low (LONG)")
    )

    prob_pct = float(merged.get("prob_pct", 55.0) or 55.0)
    at_ext_pct = float(merged.get("at_extreme_pct", prob_pct) or prob_pct)
    thr_pct = float(merged.get("threshold_pct", c.get("min_prob_pct", 55.0)) or 55.0)
    bar_time = str(merged.get("bar_time") or merged.get("date") or _now_et_str())

    entry_px = float(merged.get("entry_price") or merged.get("entry_px") or merged.get("current_price") or 0.0)
    extreme_px = float(merged.get("extreme_price") or merged.get("bar_price") or entry_px)

    # Derive ATR / ADR fallback for daily equity/ETF alerts if not already populated by compute_nq_trade_plan
    atr_pts = float(merged.get("atr_pts") or 0.0)
    if atr_pts <= 0 and entry_px > 0:
        adr_pct = float(merged.get("adr_pct", 3.0) or 3.0)
        atr_pts = round(max(0.05, entry_px * (adr_pct / 100.0)), 2)

    pred_mfe_atr = float(merged.get("pred_mfe_atr") or 1.85)
    pred_mfe_q70_atr = float(merged.get("pred_mfe_q70_atr") or round(pred_mfe_atr * 1.35, 2))
    pred_mae_atr = float(merged.get("pred_mae_atr") or 0.90)
    pred_mae_q75_atr = float(merged.get("pred_mae_q75_atr") or round(pred_mae_atr * 1.25, 2))
    pred_ratio = float(merged.get("pred_ratio") or round(pred_mfe_atr / max(0.25, pred_mae_atr), 2))
    prob_asym_pct = float(merged.get("prob_asym_pct") or round(min(95.0, max(45.0, prob_pct)), 1))

    pred_mfe_pts = float(merged.get("pred_mfe_pts") or round(pred_mfe_atr * atr_pts, 2))
    pred_mfe_q70_pts = float(merged.get("pred_mfe_q70_pts") or round(pred_mfe_q70_atr * atr_pts, 2))
    pred_mae_pts = float(merged.get("pred_mae_pts") or round(pred_mae_atr * atr_pts, 2))
    pred_mae_q75_pts = float(merged.get("pred_mae_q75_pts") or round(pred_mae_q75_atr * atr_pts, 2))

    if is_high_piv:
        mfe_expected_px = float(merged.get("mfe_expected_px") or round(entry_px - pred_mfe_pts, 2))
        mfe_q70_px = float(merged.get("mfe_q70_px") or round(entry_px - pred_mfe_q70_pts, 2))
        mae_expected_px = float(merged.get("mae_expected_px") or round(entry_px + pred_mae_pts, 2))
        mae_q75_px = float(merged.get("mae_q75_px") or round(entry_px + pred_mae_q75_pts, 2))
        stop_px = float(merged.get("stop_price") or merged.get("stop_px") or round(max(entry_px + 1.20 * pred_mae_pts, extreme_px + 0.15 * atr_pts), 2))
        tp1_px = float(merged.get("target1_price") or merged.get("target1_px") or round(entry_px - 1.05 * pred_mfe_pts, 2))
        tp2_px = float(merged.get("target2_price") or merged.get("target2_px") or mfe_q70_px)
    else:
        mfe_expected_px = float(merged.get("mfe_expected_px") or round(entry_px + pred_mfe_pts, 2))
        mfe_q70_px = float(merged.get("mfe_q70_px") or round(entry_px + pred_mfe_q70_pts, 2))
        mae_expected_px = float(merged.get("mae_expected_px") or round(entry_px - pred_mae_pts, 2))
        mae_q75_px = float(merged.get("mae_q75_px") or round(entry_px - pred_mae_q75_pts, 2))
        stop_px = float(merged.get("stop_price") or merged.get("stop_px") or round(min(entry_px - 1.20 * pred_mae_pts, extreme_px - 0.15 * atr_pts), 2))
        tp1_px = float(merged.get("target1_price") or merged.get("target1_px") or round(entry_px + 1.05 * pred_mfe_pts, 2))
        tp2_px = float(merged.get("target2_price") or merged.get("target2_px") or mfe_q70_px)

    stop_dist_pts = abs(entry_px - stop_px)
    stop_dist_atr = round(stop_dist_pts / max(1e-6, atr_pts), 2)
    tp1_pts = abs(tp1_px - entry_px)
    tp2_pts = abs(tp2_px - entry_px)
    tp1_atr = round(tp1_pts / max(1e-6, atr_pts), 2)
    tp2_atr = round(tp2_pts / max(1e-6, atr_pts), 2)
    rr_t1 = float(merged.get("rr_target1") or round(tp1_pts / max(1e-6, stop_dist_pts), 2))
    rr_t2 = float(merged.get("rr_target2") or round(tp2_pts / max(1e-6, stop_dist_pts), 2))

    is_valid_trade = bool(merged.get("is_valid_trade", False))
    validity_tier = str(merged.get("validity_tier") or (
        "A+ Institutional Setup (Valid Trade)" if (prob_pct >= 60.0 and pred_ratio >= 1.75) else (
            "High-Conviction Setup (Valid Trade)" if (prob_pct >= 55.0 and pred_ratio >= 1.50) else "Standard Real-Time Pivot"
        )
    ))
    time_exit_lbl = str(merged.get("time_exit_label") or "8 bars")

    status_tag = "" if is_valid_trade else " [WATCH ONLY]"
    prefix = "[TEST] " if is_test else ""
    is_wl = bool(merged.get("is_wickless", True))
    wl_subject_tag = "⚡ [WICKLESS PIVOT ALERT] " if is_wl else "🚨 [PIVOT ALERT] "
    subject = (
        f"{prefix}{wl_subject_tag}{sym} [{tf_label}] {direction} PIVOT ({prob_pct:.1f}% Conviction | {pred_ratio:.2f}x MFE/MAE){status_tag} "
        f"@ {entry_px:.2f} | MAE {mae_expected_px:.2f} (SL {stop_px:.2f}) | MFE {mfe_expected_px:.2f} (TP1 {tp1_px:.2f})"
    )

    if is_valid_trade:
        trade_status_block = f"• Trade Entry Status:   ✅ VALID INSTITUTIONAL TRADE ({validity_tier})\n"
    else:
        trade_status_block = (
            f"• Trade Entry Status:   ⚠️ INFORMATIONAL DISCOVERY ONLY (Not an Auto-Trade Entry)\n"
            f"• Hurdle Details:       {validity_tier}\n"
        )

    body_text = (
        f"Real-Time ML Pivot Alert — {sym} [{tf_label}]\n"
        f"====================================================\n"
        f"Signal:              {piv_label}\n"
        f"ML Detector:         ⚡ Wickless Intrabar ML (At-Extreme Touch — 0 Wick Lag)\n"
        f"Bar / Time (ET):     {bar_time}\n"
        f"Direction:           {direction}\n"
        f"Trigger / Entry Px:  {entry_px:.2f} (Developing {'High' if is_high_piv else 'Low'} Extreme: {extreme_px:.2f})\n"
        f"Current Volatility:  {atr_pts:.2f} pts (1.00x ATR)\n\n"
        f"1. CONVICTION LEVEL & TRADE STATUS\n"
        f"----------------------------------\n"
        f"{trade_status_block}"
        f"• ML Pivot Conviction:   {prob_pct:.1f}% (Threshold: {thr_pct:.0f}% | Real-Time At-Extreme: {at_ext_pct:.1f}%)\n"
        f"• Conviction Tier:       {validity_tier}\n"
        f"• Predicted MFE / MAE:   {pred_ratio:.2f}x (Asymmetry Prob: {prob_asym_pct:.1f}%)\n\n"
        f"2. PREDICTED MAE LEVELS (Adverse Excursion / Risk Cushion)\n"
        f"----------------------------------------------------------\n"
        f"• Expected MAE Level:    {mae_expected_px:.2f} ({'+' if is_high_piv else '-'}{pred_mae_pts:.2f} pts / {pred_mae_atr:.2f}x ATR against entry)\n"
        f"• 75th-%ile MAE (Q75):   {mae_q75_px:.2f} ({'+' if is_high_piv else '-'}{pred_mae_q75_pts:.2f} pts / {pred_mae_q75_atr:.2f}x ATR adverse tail)\n"
        f"• Stop-Loss Level (SL):  {stop_px:.2f} ({'+' if is_high_piv else '-'}{stop_dist_pts:.2f} pts / {stop_dist_atr:.2f}x ATR)\n\n"
        f"3. PREDICTED MFE LEVELS (Favorable Excursion / Profit Targets)\n"
        f"--------------------------------------------------------------\n"
        f"• Expected MFE Level:    {mfe_expected_px:.2f} ({'-' if is_high_piv else '+'}{pred_mfe_pts:.2f} pts / {pred_mfe_atr:.2f}x ATR in favor)\n"
        f"• 70th-%ile MFE (Q70):   {mfe_q70_px:.2f} ({'-' if is_high_piv else '+'}{pred_mfe_q70_pts:.2f} pts / {pred_mfe_q70_atr:.2f}x ATR runner potential)\n"
        f"• Target 1 (TP1 - 50%):  {tp1_px:.2f} ({'-' if is_high_piv else '+'}{tp1_pts:.2f} pts / {tp1_atr:.2f}x ATR / {rr_t1:.2f}R -> move Stop to BE)\n"
        f"• Target 2 (TP2 Runner): {tp2_px:.2f} ({'-' if is_high_piv else '+'}{tp2_pts:.2f} pts / {tp2_atr:.2f}x ATR / {rr_t2:.2f}R)\n"
        f"• Horizon / Time Stop:   {time_exit_lbl}\n"
    )
    return _dispatch_email_message(to_addr, from_user, app_pw, subject, body_text)


_INITIAL_PIVOT_KEYS_SEEDED: bool = False
_LAST_PIVOT_EMAIL_SCAN_TS: float = 0.0


def notify_new_realtime_pivot_alert(
    alert_or_plan: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    force_test: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Deduplicates by `(symbol, timeframe, bar_time, direction)` and sends a real-time
    pivot email if `gmail_enabled` and `pivot_email_alerts_enabled` are True,
    and `prob_pct >= min_prob_pct` (even if `is_valid_trade` is False).

    STRICT FILTERS:
    1. ONLY future tickers (NQ=F, GC=F, CL=F, BTC=F, SI=F) are emailed — stocks/ETFs are excluded.
    2. ONLY Wickless ML detector pivots are emailed.
    """
    is_test_runner = any("unittest" in str(a) or "pytest" in str(a) or "test_" in str(a) for a in sys.argv)
    if is_test_runner and not force_test:
        return None

    with _AUTO_LOCK:
        c = cfg or load_auto_trader_config()
        if not c.get("gmail_enabled", True) or not c.get("pivot_email_alerts_enabled", True):
            return None

        min_prob = float(c.get("pivot_email_min_prob_pct", c.get("min_prob_pct", 50.0)))
        tp = alert_or_plan.get("trade_plan") or {}
        prob_v = float(alert_or_plan.get("prob_pct") or tp.get("prob_pct", 0.0) or 0.0)
        if prob_v < min_prob:
            return None

        sym = str(alert_or_plan.get("symbol") or tp.get("symbol") or "MNQ").upper()

        # Strict Rule 1: Emails will ONLY be sent for future tickers (not stocks/ETFs)
        if not (
            nq_ml_engine.is_intraday_futures_symbol(sym)
            or sym.replace("=F", "") in ("NQ", "MNQ", "GC", "MGC", "CL", "MCL", "BTC", "MBT", "SI", "SIL")
        ):
            return None

        # Strict Rule 2: Emails will ONLY be sent if detected with the Wickless ML detector
        is_wl = bool(alert_or_plan.get("is_wickless", tp.get("is_wickless", False)))
        if not is_wl and not force_test:
            return None

        # Strict Rule 3: Emails only sent for selected future tickers
        allowed_tickers = [str(x).upper().replace("=F", "") for x in c.get("pivot_email_tickers", ["NQ", "GC", "CL", "BTC", "SI"])]
        sym_root = sym.replace("=F", "")
        root_map = {"MNQ": "NQ", "MGC": "GC", "MCL": "CL", "MBT": "BTC", "SIL": "SI"}
        norm_sym = root_map.get(sym_root, sym_root)
        if allowed_tickers and norm_sym not in allowed_tickers and sym_root not in allowed_tickers:
            return None

        # Strict Rule 4: Emails only sent for selected timeframes
        tf = str(alert_or_plan.get("timeframe") or tp.get("timeframe") or "15m").lower().strip()
        allowed_tfs = [str(x).lower().strip() for x in c.get("pivot_email_timeframes", ["5m", "15m", "1h", "4h"])]
        if allowed_tfs and tf not in allowed_tfs:
            return None

        st = load_auto_trader_state()
        b_time = str(alert_or_plan.get("bar_time") or tp.get("bar_time") or alert_or_plan.get("date") or "")
        direction = str(
            alert_or_plan.get("direction")
            or tp.get("direction")
            or ("SHORT" if alert_or_plan.get("is_high_pivot") else "LONG")
        ).upper()
        dedup_key = f"{sym}|{tf}|{b_time}|{direction}|wickless"

        alerted_keys = list(st.get("alerted_pivot_keys") or [])
        if dedup_key in alerted_keys:
            return None

        alerted_keys.append(dedup_key)
        st["alerted_pivot_keys"] = alerted_keys[-400:]
        email_res = send_realtime_pivot_alert_gmail(alert_or_plan, cfg=c)
        e_logs = list(st.get("email_log") or [])
        e_logs.append(email_res)
        st["email_log"] = e_logs[-30:]
        _append_log(
            st,
            "PIVOT_EMAIL_ALERT",
            f"Real-time Wickless pivot email ({sym} [{tf.upper()}] {direction} @ {b_time}): {email_res.get('status')}",
        )
        _save_state_unlocked(st)
        return email_res


def scan_and_email_realtime_pivots(
    symbols: Optional[List[str]] = None,
    sync_live: bool = False,
    force_test: bool = False,
) -> List[Dict[str, Any]]:
    """
    Scans futures tickers (`NQ=F`, `GC=F`, `CL=F`, `BTC=F`, `SI=F`) across `15m`, `1h`, and `4h`
    for new real-time Wickless pivots.
    """
    global _INITIAL_PIVOT_KEYS_SEEDED, _LAST_PIVOT_EMAIL_SCAN_TS
    is_test_runner = any("unittest" in str(a) or "pytest" in str(a) or "test_" in str(a) for a in sys.argv)
    if is_test_runner and not force_test:
        return []

    cfg = load_auto_trader_config()
    if not cfg.get("gmail_enabled", True) or not cfg.get("pivot_email_alerts_enabled", True):
        return []

    thr = float(cfg.get("pivot_email_min_prob_pct", cfg.get("min_prob_pct", 50.0)))
    raw_syms = symbols or cfg.get("pivot_email_tickers") or ["NQ=F", "GC=F", "CL=F", "BTC=F", "SI=F"]
    # Filter strictly to futures symbols only
    watch_syms = [
        s for s in raw_syms
        if nq_ml_engine.is_intraday_futures_symbol(s)
        or s.replace("=F", "") in ("NQ", "MNQ", "GC", "MGC", "CL", "MCL", "BTC", "MBT", "SI", "SIL")
    ]
    sent_results: List[Dict[str, Any]] = []

    first_seed = not _INITIAL_PIVOT_KEYS_SEEDED
    if first_seed:
        _INITIAL_PIVOT_KEYS_SEEDED = True

    allowed_tfs = [str(x).lower().strip() for x in cfg.get("pivot_email_timeframes", ["5m", "15m", "1h", "4h"])]
    for sym in watch_syms:
        try:
            if sync_live and not first_seed:
                nq_ml_engine.ensure_nq_intraday_uptodate(max_age_sec=45.0, symbol=sym)
            rt = nq_ml_engine.evaluate_nq_realtime_multi_tf(
                live_bar=None,
                threshold_pct=thr,
                horizon_k=int(cfg.get("time_stop_bars", 8)),
                symbol=sym,
                use_wickless=True,
            )
            for al in rt.get("alerts", []):
                tf_al = str(al.get("timeframe") or "15m").lower().strip()
                if allowed_tfs and tf_al not in allowed_tfs:
                    continue
                plan = al.get("trade_plan") or {}
                prob_v = float(al.get("prob_pct", plan.get("prob_pct", 0.0)))
                # Email on any high-conviction pivot even if it does not pass auto-trade hurdles (e.g. ratio < 1.35 or rr < 1.0)
                if prob_v < thr:
                    continue
                if first_seed:
                    # Seed existing bar key on startup so we don't email historical pivots that formed before boot
                    with _AUTO_LOCK:
                        st = load_auto_trader_state()
                        tf_k = str(al.get("timeframe") or "15m").lower()
                        b_t = str(al.get("bar_time") or "")
                        dir_k = str(al.get("direction") or "SHORT").upper()
                        d_key = f"{sym.upper()}|{tf_k}|{b_t}|{dir_k}|wickless"
                        ak = list(st.get("alerted_pivot_keys") or [])
                        if d_key not in ak:
                            ak.append(d_key)
                            st["alerted_pivot_keys"] = ak[-400:]
                            _save_state_unlocked(st)
                else:
                    res = notify_new_realtime_pivot_alert(al, cfg=cfg, force_test=force_test)
                    if res is not None:
                        sent_results.append(res)
        except Exception:
            pass

    _LAST_PIVOT_EMAIL_SCAN_TS = time.time()
    return sent_results


def _query_tws_mnq_snapshot(
    ports: List[int],
    symbol: str = "MNQ",
    exchange: str = "CME",
) -> Dict[str, Any]:
    """
    Connects to TWS and returns the live MNQ or MGC front-month contract, current net position (`live_net_pos`),
    and all active working orders (`entry_orders`, `stop_orders`, `tp_orders`).
    """
    is_mgc = symbol.upper() == "MGC"
    sync_attr = "_LAST_GC_INTRADAY_SYNC_TS" if is_mgc else "_LAST_NQ_INTRADAY_SYNC_TS"
    sync_sym = "GC=F" if is_mgc else "NQ=F"
    try:
        import asyncio
        from ib_async import ContFuture, Future, IB

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ib = IB()
        try:
            connected = False
            conn_port = None
            for p in ports:
                try:
                    cid = (248 if is_mgc else 240) + int(time.time() * 10) % 10
                    ib.connect("127.0.0.1", int(p), clientId=cid, timeout=2.0, readonly=False)
                    if ib.isConnected():
                        connected = True
                        conn_port = int(p)
                        break
                except Exception:
                    pass
            if not connected or not ib.isConnected():
                return {"connected": False, "error": "TWS not connected"}

            cf = ContFuture(symbol, exchange)
            ib.qualifyContracts(cf)
            front_contract = None
            if getattr(cf, "conId", 0):
                front_contract = Future(conId=cf.conId, symbol=symbol, exchange=exchange, currency="USD")
                ib.qualifyContracts(front_contract)

            net_pos = 0
            for pos in ib.positions():
                sym = getattr(pos.contract, "symbol", "")
                if sym == symbol:
                    net_pos += int(round(float(pos.position)))

            ib.reqAllOpenOrders()
            ib.sleep(0.15)
            raw_orders = []
            for trade in ib.openTrades():
                c_sym = getattr(trade.contract, "symbol", "")
                if c_sym != symbol:
                    continue
                st_status = getattr(trade.orderStatus, "status", "")
                if st_status in ("Cancelled", "Filled", "Inactive", "ApiCancelled"):
                    continue
                o = trade.order
                raw_orders.append({
                    "orderId": int(getattr(o, "orderId", 0)),
                    "parentId": int(getattr(o, "parentId", 0) or 0),
                    "action": str(getattr(o, "action", "")),
                    "totalQuantity": int(round(float(getattr(o, "totalQuantity", 0) or 0))),
                    "orderType": str(getattr(o, "orderType", "")),
                    "lmtPrice": float(getattr(o, "lmtPrice", 0.0) or 0.0),
                    "auxPrice": float(getattr(o, "auxPrice", 0.0) or 0.0),
                    "ocaGroup": str(getattr(o, "ocaGroup", "")),
                    "tif": str(getattr(o, "tif", "")),
                    "outsideRth": bool(getattr(o, "outsideRth", False)),
                    "status": st_status,
                })

            # Parent entry orders have no STP type, no ocaGroup, and parentId == 0
            active_entry_ids = {
                o["orderId"]
                for o in raw_orders
                if "STP" not in o["orderType"] and not o["ocaGroup"] and o["parentId"] == 0 and o["orderId"] > 0
            }
            stop_orders = []
            tp_orders = []
            entry_orders = []
            pending_child_orders = []
            for o_info in raw_orders:
                if o_info["orderId"] in active_entry_ids:
                    entry_orders.append(o_info)
                elif o_info["parentId"] in active_entry_ids:
                    # Unactivated child bracket order waiting for its parent entry order to fill
                    pending_child_orders.append(o_info)
                elif "STP" in o_info["orderType"]:
                    stop_orders.append(o_info)
                elif o_info["ocaGroup"]:
                    tp_orders.append(o_info)
                else:
                    entry_orders.append(o_info)

            live_bar = None
            if front_contract is not None and (
                net_pos == 0 or (time.time() - getattr(nq_ml_engine, sync_attr, 0.0)) >= 20.0
            ):
                try:
                    import pandas as pd

                    hbars = ib.reqHistoricalData(
                        front_contract,
                        endDateTime="",
                        durationStr="1 D",
                        barSizeSetting="5 mins",
                        whatToShow="TRADES",
                        useRTH=False,
                        formatDate=2,
                        timeout=2.2,
                    )
                    if hbars:
                        ny_tz = ZoneInfo("America/New_York")
                        rows = []
                        for nb in hbars:
                            ts = pd.Timestamp(nb.date)
                            dt_et = ts.tz_convert(ny_tz).tz_localize(None) if ts.tzinfo is not None else ts
                            rows.append(
                                {
                                    "datetime": dt_et,
                                    "open": float(nb.open),
                                    "high": float(nb.high),
                                    "low": float(nb.low),
                                    "close": float(nb.close),
                                    "volume": max(0.0, float(getattr(nb, "volume", 0.0) or 0.0)),
                                }
                            )
                        bar_df = pd.DataFrame(rows).set_index("datetime").sort_index()
                        last_b = rows[-1]
                        bucket_15m = last_b["datetime"].floor("15min")
                        sub_15m = bar_df[bar_df.index >= bucket_15m]
                        live_bar = {
                            "date": bucket_15m.strftime("%Y-%m-%d %H:%M"),
                            "open": float(sub_15m["open"].iloc[0]),
                            "high": float(sub_15m["high"].max()),
                            "low": float(sub_15m["low"].min()),
                            "close": float(sub_15m["close"].iloc[-1]),
                            "volume": float(sub_15m["volume"].sum()),
                            "is_intraday_bar": True,
                        }
                        if (time.time() - getattr(nq_ml_engine, sync_attr, 0.0)) >= 20.0:
                            try:
                                nq_ml_engine.sync_live_nq_intraday_bars(
                                    tws_5m_df=bar_df,
                                    live_bar=live_bar,
                                    symbol=sync_sym,
                                )
                            except Exception:
                                pass
                except Exception:
                    pass

            return {
                "connected": True,
                "port": conn_port,
                "conId": int(getattr(front_contract, "conId", 0) or 0),
                "localSymbol": str(getattr(front_contract, "localSymbol", symbol)),
                "live_net_pos": net_pos,
                "stop_orders": stop_orders,
                "tp_orders": tp_orders,
                "entry_orders": entry_orders,
                "pending_child_orders": pending_child_orders,
                "live_bar": live_bar,
            }
        finally:
            try:
                if ib.isConnected():
                    ib.disconnect()
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass
    except Exception as e:
        return {"connected": False, "error": str(e)}


def _execute_tws_action(
    ports: List[int],
    action_type: str,
    payload: Dict[str, Any],
    symbol: str = "MNQ",
    exchange: str = "CME",
) -> Dict[str, Any]:
    """
    Executes an atomic order action in TWS (`place_2mnq_bracket`, `place_missing_protection_bracket`,
    `cancel_all_mnq_orders`, or `move_stop_to_breakeven`) for `MNQ` (CME) or `MGC` (COMEX).
    """
    is_mgc = symbol.upper() == "MGC"
    tick_fn = round_to_mgc_tick if is_mgc else round_to_nq_tick
    try:
        import asyncio
        from ib_async import ContFuture, Future, IB, Order

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ib = IB()
        try:
            connected = False
            for p in ports:
                try:
                    cid = (270 if is_mgc else 260) + int(time.time() * 10) % 10
                    ib.connect("127.0.0.1", int(p), clientId=cid, timeout=2.5, readonly=False)
                    if ib.isConnected():
                        connected = True
                        break
                except Exception:
                    pass
            if not connected or not ib.isConnected():
                return {"ok": False, "error": "TWS not connected"}

            cf = ContFuture(symbol, exchange)
            ib.qualifyContracts(cf)
            if not getattr(cf, "conId", 0):
                return {"ok": False, "error": f"Could not qualify {symbol} front-month contract"}
            contract = Future(conId=cf.conId, symbol=symbol, exchange=exchange, currency="USD")
            ib.qualifyContracts(contract)

            if action_type == "cancel_all_mnq_orders":
                ib.reqAllOpenOrders()
                ib.sleep(0.15)
                cancelled_ids = []
                for tr in ib.openTrades():
                    if getattr(tr.contract, "symbol", "") == symbol:
                        st_s = getattr(tr.orderStatus, "status", "")
                        if st_s not in ("Cancelled", "Filled", "ApiCancelled"):
                            ib.cancelOrder(tr.order)
                            cancelled_ids.append(int(getattr(tr.order, "orderId", 0)))
                ib.sleep(0.25)
                return {"ok": True, "cancelled_order_ids": cancelled_ids}

            if action_type == "place_2mnq_bracket":
                spec = payload["bracket_spec"]
                placed_ids = []
                for unit in spec["units"]:
                    p_id = ib.client.getReqId()
                    tp_id = ib.client.getReqId()
                    sl_id = ib.client.getReqId()

                    p_spec = unit["parent_entry"]
                    tp_spec = unit["take_profit"]
                    sl_spec = unit["stop_loss"]

                    parent_ord = Order(
                        orderId=p_id,
                        action=p_spec["action"],
                        totalQuantity=p_spec["totalQuantity"],
                        orderType=p_spec["orderType"],
                        lmtPrice=p_spec["lmtPrice"],
                        tif="GTC",
                        outsideRth=True,
                        transmit=False,
                    )
                    tp_ord = Order(
                        orderId=tp_id,
                        parentId=p_id,
                        action=tp_spec["action"],
                        totalQuantity=tp_spec["totalQuantity"],
                        orderType=tp_spec["orderType"],
                        lmtPrice=tp_spec["lmtPrice"],
                        tif="GTC",
                        outsideRth=True,
                        ocaGroup=tp_spec["ocaGroup"],
                        ocaType=1,
                        transmit=False,
                    )
                    sl_ord = Order(
                        orderId=sl_id,
                        parentId=p_id,
                        action=sl_spec["action"],
                        totalQuantity=sl_spec["totalQuantity"],
                        orderType=sl_spec["orderType"],
                        auxPrice=sl_spec["auxPrice"],
                        lmtPrice=sl_spec["lmtPrice"],
                        tif="GTC",
                        outsideRth=True,
                        ocaGroup=sl_spec["ocaGroup"],
                        ocaType=1,
                        transmit=True,
                    )
                    ib.placeOrder(contract, parent_ord)
                    ib.placeOrder(contract, tp_ord)
                    ib.placeOrder(contract, sl_ord)
                    placed_ids.extend([p_id, tp_id, sl_id])
                ib.sleep(0.35)
                return {"ok": True, "order_ids": placed_ids, "localSymbol": contract.localSymbol}

            if action_type == "place_missing_protection_bracket":
                missing_qty = int(payload["missing_qty"])
                exit_action = str(payload["exit_action"])
                stop_px = tick_fn(float(payload["stop_px"]))
                stop_lmt_px = tick_fn(float(payload["stop_lmt_px"]))
                tp_px = tick_fn(float(payload["target_px"]))
                oca_grp = f"{symbol}_WD_{int(time.time())}"

                tp_id = ib.client.getReqId()
                sl_id = ib.client.getReqId()
                tp_ord = Order(
                    orderId=tp_id,
                    action=exit_action,
                    totalQuantity=missing_qty,
                    orderType="LMT",
                    lmtPrice=tp_px,
                    tif="GTC",
                    outsideRth=True,
                    ocaGroup=oca_grp,
                    ocaType=1,
                    transmit=True,
                )
                sl_ord = Order(
                    orderId=sl_id,
                    action=exit_action,
                    totalQuantity=missing_qty,
                    orderType="STP LMT",
                    auxPrice=stop_px,
                    lmtPrice=stop_lmt_px,
                    tif="GTC",
                    outsideRth=True,
                    ocaGroup=oca_grp,
                    ocaType=1,
                    transmit=True,
                )
                ib.placeOrder(contract, tp_ord)
                ib.placeOrder(contract, sl_ord)
                ib.sleep(0.30)
                return {"ok": True, "order_ids": [tp_id, sl_id], "oca_group": oca_grp}

            if action_type == "move_stop_to_breakeven":
                be_px = tick_fn(float(payload["be_stop_px"]))
                be_lmt_px = tick_fn(float(payload["be_stop_lmt_px"]))
                ib.reqAllOpenOrders()
                ib.sleep(0.15)
                modified_ids = []
                for tr in ib.openTrades():
                    if getattr(tr.contract, "symbol", "") == symbol:
                        st_s = getattr(tr.orderStatus, "status", "")
                        if st_s not in ("Cancelled", "Filled", "ApiCancelled") and "STP" in str(getattr(tr.order, "orderType", "")):
                            tr.order.auxPrice = be_px
                            if "LMT" in str(getattr(tr.order, "orderType", "")):
                                tr.order.lmtPrice = be_lmt_px
                            tr.order.transmit = True
                            ib.placeOrder(contract, tr.order)
                            modified_ids.append(int(getattr(tr.order, "orderId", 0)))
                ib.sleep(0.25)
                return {"ok": True, "modified_order_ids": modified_ids}

            if action_type == "cancel_unfilled_entry_orders":
                ib.reqAllOpenOrders()
                ib.sleep(0.15)
                cancelled_ids = []
                for tr in ib.openTrades():
                    if getattr(tr.contract, "symbol", "") == symbol:
                        st_s = getattr(tr.orderStatus, "status", "")
                        if st_s not in ("Cancelled", "Filled", "ApiCancelled"):
                            o_t = str(getattr(tr.order, "orderType", ""))
                            oca_g = str(getattr(tr.order, "ocaGroup", ""))
                            p_id = int(getattr(tr.order, "parentId", 0) or 0)
                            if "STP" not in o_t and not oca_g and p_id == 0:
                                ib.cancelOrder(tr.order)
                                cancelled_ids.append(int(getattr(tr.order, "orderId", 0)))
                ib.sleep(0.25)
                return {"ok": True, "cancelled_entry_order_ids": cancelled_ids}

            if action_type == "close_open_mnq_time_stop":
                ib.reqAllOpenOrders()
                ib.sleep(0.15)
                cancelled_ids = []
                for tr in ib.openTrades():
                    if getattr(tr.contract, "symbol", "") == symbol:
                        st_s = getattr(tr.orderStatus, "status", "")
                        if st_s not in ("Cancelled", "Filled", "ApiCancelled"):
                            ib.cancelOrder(tr.order)
                            cancelled_ids.append(int(getattr(tr.order, "orderId", 0)))
                ib.sleep(0.30)

                verified_net_pos = 0
                for pos in ib.positions():
                    if getattr(pos.contract, "symbol", "") == symbol:
                        verified_net_pos += int(round(float(pos.position)))

                if verified_net_pos == 0:
                    return {
                        "ok": True,
                        "skipped_already_flat": True,
                        "cancelled_order_ids": cancelled_ids,
                    }

                close_action = "SELL" if verified_net_pos > 0 else "BUY"
                close_qty = abs(verified_net_pos)
                close_id = ib.client.getReqId()
                close_ord = Order(
                    orderId=close_id,
                    action=close_action,
                    totalQuantity=close_qty,
                    orderType="MKT",
                    tif="GTC",
                    outsideRth=True,
                    transmit=True,
                )
                ib.placeOrder(contract, close_ord)
                ib.sleep(0.35)
                return {
                    "ok": True,
                    "closed_qty": close_qty,
                    "close_action": close_action,
                    "order_id": close_id,
                    "cancelled_order_ids": cancelled_ids,
                }

            return {"ok": False, "error": f"Unknown action_type {action_type}"}
        finally:
            try:
                if ib.isConnected():
                    ib.disconnect()
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _now_et_naive_epoch() -> int:
    """Returns current America/New_York wall-clock time encoded as a tz-naive unix timestamp
    (matching the tz-naive America/New_York DatetimeIndex in nq_ml_engine)."""
    now_et = dt.datetime.now(ZoneInfo("America/New_York")).replace(tzinfo=dt.timezone.utc)
    return int(now_et.timestamp())


def _run_mgc_cycle_unlocked(
    cfg: Dict[str, Any],
    st: Dict[str, Any],
    simulated_mgc_snapshot: Optional[Dict[str, Any]] = None,
    simulated_mgc_plan_15m: Optional[Dict[str, Any]] = None,
    force_mgc_enabled: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Runs the 15-minute 2-MGC (Micro Gold Futures, COMEX) Auto-Execution & Watchdog cycle
    when `cfg['mgc_enabled']` is True (default False).
    """
    mgc_on = bool(cfg.get("mgc_enabled", False)) or force_mgc_enabled
    if not mgc_on and simulated_mgc_snapshot is None:
        st["mgc_watchdog_status"] = "OFF (MGC Auto-Trader Disabled)"
        return None

    ports = list(cfg.get("tws_ports") or [7497, 7496, 4002, 4001])
    snap = (
        simulated_mgc_snapshot
        if simulated_mgc_snapshot is not None
        else _query_tws_mnq_snapshot(ports, symbol="MGC", exchange="COMEX")
    )
    if not snap.get("connected"):
        st["mgc_watchdog_status"] = f"TWS OFFLINE ({snap.get('error', 'No connection')})"
        return {"mgc_snapshot": snap}

    live_net_pos = int(snap.get("live_net_pos", 0))
    stop_orders = list(snap.get("stop_orders") or [])
    tp_orders = list(snap.get("tp_orders") or [])
    entry_orders = list(snap.get("entry_orders") or [])
    live_bar_obj = snap.get("live_bar") or {}
    curr_mkt_px = float(live_bar_obj.get("close", 0.0) or 0.0)

    expected_exit_action = "SELL" if live_net_pos > 0 else ("BUY" if live_net_pos < 0 else None)
    working_stop_qty = sum(
        int(o.get("totalQuantity", 0))
        for o in stop_orders
        if expected_exit_action is None or o.get("action") == expected_exit_action
    )
    working_tp_qty = sum(
        int(o.get("totalQuantity", 0))
        for o in tp_orders
        if expected_exit_action is None or o.get("action") == expected_exit_action
    )

    st["mgc_live_net_pos"] = live_net_pos
    st["mgc_working_stop_qty"] = working_stop_qty
    st["mgc_working_tp_qty"] = working_tp_qty

    if live_net_pos == 0:
        prev_state = st.get("mgc_position_state", "FLAT")
        active_tr = st.get("mgc_active_trade")
        now_epoch = time.time()
        entry_filled_flag = bool(isinstance(active_tr, dict) and active_tr.get("entry_filled", False))
        grace_active = (
            prev_state in ("PENDING_ENTRY", "OPEN")
            and len(entry_orders) > 0
            and isinstance(active_tr, dict)
            and not entry_filled_flag
            and (now_epoch - float(active_tr.get("opened_epoch", 0.0))) < float(cfg.get("entry_timeout_sec", 180))
        )
        if grace_active:
            st["mgc_position_state"] = "PENDING_ENTRY"
            st["mgc_watchdog_status"] = f"PENDING ENTRY ({active_tr.get('direction')} 2 MGC waiting for fill)"
            return {"mgc_snapshot": snap}

        if prev_state in ("OPEN", "PENDING_ENTRY") or len(stop_orders) > 0 or len(tp_orders) > 0 or len(entry_orders) > 0:
            had_orphan_orders = len(stop_orders) > 0 or len(tp_orders) > 0 or len(entry_orders) > 0
            unfilled_entry_expired = bool(len(entry_orders) > 0 and not entry_filled_flag)
            if had_orphan_orders:
                if simulated_mgc_snapshot is None:
                    _execute_tws_action(ports, "cancel_all_mnq_orders", {}, symbol="MGC", exchange="COMEX")
                if unfilled_entry_expired:
                    _append_log(
                        st,
                        "MGC_ENTRY_TIMEOUT_CANCELLED",
                        "MGC entry limit order did not fill within timeout — cancelled unfilled bracket orders.",
                    )
                else:
                    _append_log(
                        st,
                        "MGC_MANUAL_CLOSE_CLEANUP",
                        "MGC Position is FLAT (0 MGC) — cancelled leftover child orders and disarmed MGC watchdog.",
                    )
            elif prev_state == "OPEN":
                _append_log(
                    st,
                    "MGC_POSITION_CLOSED",
                    "MGC Position closed manually or via bracket (0 MGC). Auto-watchdog disarmed.",
                )

            was_filled_trade = bool(
                isinstance(active_tr, dict)
                and (entry_filled_flag or (prev_state == "OPEN" and not unfilled_entry_expired))
            )
            if was_filled_trade:
                est_exit_px = curr_mkt_px or float(
                    active_tr.get("target2_px" if active_tr.get("tp1_filled") else "entry_px", 0.0)
                )
                exit_rsn = "MANUAL_CLOSE" if had_orphan_orders else ("TP2_TARGET" if active_tr.get("tp1_filled") else "BRACKET_EXIT")
                _record_closed_trade(st, active_tr, est_exit_px, exit_rsn)

            st["mgc_position_state"] = "FLAT"
            st["mgc_active_direction"] = None
            st["mgc_active_trade"] = None
            st["mgc_working_stop_qty"] = 0
            st["mgc_working_tp_qty"] = 0
            st["mgc_watchdog_status"] = "IDLE (Flat — 0 MGC, Watchdog Disarmed)"
            return {"mgc_snapshot": snap}

        st["mgc_watchdog_status"] = "IDLE (Flat — 0 MGC, Watchdog Disarmed)"
    else:
        live_dir = "LONG" if live_net_pos > 0 else "SHORT"
        abs_pos = abs(live_net_pos)
        st["mgc_position_state"] = "OPEN"
        st["mgc_active_direction"] = live_dir

        active_tr = st.get("mgc_active_trade") or {}
        if not active_tr or active_tr.get("direction") != live_dir:
            eval_rt = nq_ml_engine.evaluate_nq_realtime_multi_tf(
                threshold_pct=float(cfg.get("min_prob_pct", 55.0)),
                horizon_k=8,
                symbol="GC=F",
            )
            tf15 = eval_rt.get("timeframes", {}).get("15m", {})
            plan15 = tf15.get("trade_plan") or {}
            c_now = float(tf15.get("close") or plan15.get("entry_px") or 2700.0)
            atr_now = max(3.0, float(tf15.get("atr14_pts") or 8.0))
            stop_fb = round_to_mgc_tick(c_now - 1.2 * atr_now if live_dir == "LONG" else c_now + 1.2 * atr_now)
            tp_fb = round_to_mgc_tick(c_now + 1.5 * atr_now if live_dir == "LONG" else c_now - 1.5 * atr_now)
            active_tr = {
                "symbol": "MGC",
                "direction": live_dir,
                "entry_px": round_to_mgc_tick(c_now),
                "stop_px": stop_fb,
                "target1_px": tp_fb,
                "target2_px": tp_fb,
                "opened_epoch": time.time(),
                "bar_time": _now_et_str(),
                "entry_filled": True,
                "max_abs_pos": abs_pos,
            }
            st["mgc_active_trade"] = active_tr
        else:
            was_already_filled = bool(active_tr.get("entry_filled", False))
            active_tr["entry_filled"] = True
            prev_max = int(active_tr.get("max_abs_pos", 0))
            if not was_already_filled and len(entry_orders) == 0:
                prev_max = max(prev_max, 2, abs_pos)
            else:
                prev_max = max(prev_max, abs_pos)
            active_tr["max_abs_pos"] = prev_max
            st["mgc_active_trade"] = active_tr

        # If 1 of 2 units filled and the 2nd parent entry order timed out, cancel the remaining unfilled entry order
        if (
            abs_pos == 1
            and len(entry_orders) > 0
            and (time.time() - float(active_tr.get("opened_epoch", 0.0))) >= float(cfg.get("entry_timeout_sec", 180))
        ):
            if simulated_mgc_snapshot is None:
                _execute_tws_action(ports, "cancel_unfilled_entry_orders", {}, symbol="MGC", exchange="COMEX")
            entry_orders = []

        if (
            abs_pos == 1
            and not active_tr.get("tp1_filled")
            and len(entry_orders) == 0
            and int(active_tr.get("max_abs_pos", 0)) >= 2
        ):
            be_px = round_to_mgc_tick(float(active_tr.get("entry_px", active_tr.get("stop_px", 0.0))))
            prot_pts = round_to_mgc_tick(float(cfg.get("mgc_stop_limit_protection_pts", 5.0)))
            be_lmt_px = round_to_mgc_tick(be_px - prot_pts if live_dir == "LONG" else be_px + prot_pts)
            active_tr["tp1_filled"] = True
            active_tr["stop_px"] = be_px
            st["mgc_active_trade"] = active_tr
            if simulated_mgc_snapshot is None:
                _execute_tws_action(
                    ports,
                    "move_stop_to_breakeven",
                    {"be_stop_px": be_px, "be_stop_lmt_px": be_lmt_px},
                    symbol="MGC",
                    exchange="COMEX",
                )
            _append_log(
                st,
                "MGC_TP1_FILLED_BE_MOVE",
                f"MGC TP1 filled (1 MGC remaining) — moved Unit 2 Runner Stop-Loss to Breakeven ({be_px:.2f}).",
            )

        if working_stop_qty < abs_pos:
            missing_qty = abs_pos - working_stop_qty
            exit_act = "SELL" if live_dir == "LONG" else "BUY"
            stop_px = round_to_mgc_tick(float(active_tr.get("stop_px", 0.0)))
            prot_pts = round_to_mgc_tick(float(cfg.get("mgc_stop_limit_protection_pts", 5.0)))
            stop_lmt_px = round_to_mgc_tick(stop_px - prot_pts if live_dir == "LONG" else stop_px + prot_pts)
            tp_px = round_to_mgc_tick(
                float(active_tr.get("target2_px" if abs_pos == 1 else "target1_px", active_tr.get("entry_px", stop_px)))
            )
            repair_payload = {
                "missing_qty": missing_qty,
                "exit_action": exit_act,
                "stop_px": stop_px,
                "stop_lmt_px": stop_lmt_px,
                "target_px": tp_px,
            }
            if simulated_mgc_snapshot is None:
                rep_res = _execute_tws_action(
                    ports,
                    "place_missing_protection_bracket",
                    repair_payload,
                    symbol="MGC",
                    exchange="COMEX",
                )
            else:
                rep_res = {"ok": True, "order_ids": [9801, 9802], "oca_group": "SIM_MGC_WD_OCA"}

            if rep_res.get("ok"):
                st["mgc_working_stop_qty"] = abs_pos
                st["mgc_watchdog_status"] = (
                    f"REPAIRED BRACKET ({live_dir} {abs_pos} MGC — placed missing {missing_qty}x GTC Stop @ {stop_px:.2f} & TP @ {tp_px:.2f})"
                )
                _append_log(
                    st,
                    "MGC_WATCHDOG_REPAIR",
                    f"Detected {missing_qty} unprotected MGC ({live_dir}). Placed GTC Outside-RTH Stop @ {stop_px:.2f} + TP @ {tp_px:.2f}.",
                )
            else:
                st["mgc_watchdog_status"] = f"WATCHDOG ALERT: Failed to place MGC stop for {missing_qty} MGC ({rep_res.get('error')})"
        else:
            st["mgc_watchdog_status"] = (
                f"PROTECTED ({live_dir} {abs_pos} MGC | Working GTC Stop Qty: {working_stop_qty} @ {float(active_tr.get('stop_px', 0.0)):.2f})"
            )
        return {"mgc_snapshot": snap}

    # New 15m MGC Entry Evaluation (Only when FLAT: live_net_pos == 0)
    if simulated_mgc_plan_15m is not None:
        plan_15m = simulated_mgc_plan_15m
    elif simulated_mgc_snapshot is not None:
        plan_15m = None
    else:
        eval_rt = nq_ml_engine.evaluate_nq_realtime_multi_tf(
            live_bar=snap.get("live_bar"),
            threshold_pct=float(cfg.get("min_prob_pct", 55.0)),
            horizon_k=8,
            symbol="GC=F",
        )
        tf15_obj = eval_rt.get("timeframes", {}).get("15m", {})
        cand_plan = tf15_obj.get("trade_plan") or {}
        last_bar_ts = int(tf15_obj.get("last_bar_timestamp", 0) or 0)
        plan_bar_ts = int(cand_plan.get("bar_timestamp", 0) or 0)
        now_et_ts = _now_et_naive_epoch()
        is_recent_wallclock = plan_bar_ts > 0 and (now_et_ts - plan_bar_ts) <= 1800
        is_fresh_15m_trigger = is_recent_wallclock and (
            bool(tf15_obj.get("is_live_bar_triggered"))
            or (bool(tf15_obj.get("triggered")) and last_bar_ts > 0 and (last_bar_ts - plan_bar_ts) <= 900)
        )
        plan_15m = cand_plan if is_fresh_15m_trigger else None

    if not plan_15m:
        return {"mgc_snapshot": snap}

    prob_pct = float(plan_15m.get("prob_pct", 0.0))
    pred_ratio = float(plan_15m.get("pred_ratio", 0.0))
    rr_t1 = float(plan_15m.get("rr_target1", 0.0))
    bar_ts = int(plan_15m.get("bar_timestamp", 0) or 0)
    bar_key = f"MGC_15m_{bar_ts}_{plan_15m.get('direction', '')}"

    passes_gates = (
        prob_pct >= float(cfg.get("min_prob_pct", 55.0))
        and pred_ratio >= float(cfg.get("min_pred_ratio", 1.35))
        and rr_t1 >= float(cfg.get("min_rr_target1", 1.00))
    )
    if not passes_gates or bar_key in (st.get("mgc_traded_bar_keys") or []):
        return {"mgc_snapshot": snap}

    spec = build_2mgc_gtc_bracket_spec(plan_15m, cfg=cfg)
    if simulated_mgc_snapshot is None:
        exec_res = _execute_tws_action(
            ports,
            "place_2mnq_bracket",
            {"bracket_spec": spec},
            symbol="MGC",
            exchange="COMEX",
        )
    else:
        exec_res = {"ok": True, "order_ids": [2001, 2002, 2003, 2004, 2005, 2006], "localSymbol": "MGCZ6"}

    if exec_res.get("ok"):
        traded_keys = list(st.get("mgc_traded_bar_keys") or [])
        traded_keys.append(bar_key)
        st["mgc_traded_bar_keys"] = traded_keys[-200:]
        st["mgc_position_state"] = "OPEN"
        st["mgc_active_direction"] = spec["direction"]
        st["mgc_live_net_pos"] = 2 if spec["direction"] == "LONG" else -2
        st["mgc_working_stop_qty"] = 2
        st["mgc_working_tp_qty"] = 2
        st["mgc_active_trade"] = {
            **spec,
            "order_ids": exec_res.get("order_ids", []),
            "opened_at": _now_et_str(),
            "opened_epoch": time.time(),
            "tp1_filled": False,
            "entry_filled": False,
            "max_abs_pos": 0,
        }
        st["mgc_watchdog_status"] = (
            f"OPENED {spec['direction']} 2 MGC @ {spec['entry_px']:.2f} "
            f"(GTC Stop @ {spec['stop_px']:.2f} | TP1 @ {spec['target1_px']:.2f} | TP2 @ {spec['target2_px']:.2f})"
        )
        _append_log(
            st,
            "MGC_TRADE_OPENED",
            f"Opened {spec['direction']} 2 MGC [15m] @ {spec['entry_px']:.2f} | "
            f"SL {spec['stop_px']:.2f} | TP1 {spec['target1_px']:.2f} | TP2 {spec['target2_px']:.2f} (GTC Outside RTH)",
        )
        if cfg.get("gmail_enabled", True) and simulated_mgc_snapshot is None:
            email_res = send_position_opened_gmail(st["mgc_active_trade"], cfg=cfg)
            e_logs = list(st.get("email_log") or [])
            e_logs.append(email_res)
            st["email_log"] = e_logs[-30:]
            _append_log(st, "GMAIL_NOTIFICATION", f"MGC Email notification: {email_res.get('status')}")

    return {"mgc_snapshot": snap, "mgc_execution": exec_res}


def run_watchdog_and_execution_cycle(
    simulated_tws_snapshot: Optional[Dict[str, Any]] = None,
    simulated_plan_15m: Optional[Dict[str, Any]] = None,
    force_enabled: bool = False,
    simulated_mgc_snapshot: Optional[Dict[str, Any]] = None,
    simulated_mgc_plan_15m: Optional[Dict[str, Any]] = None,
    force_mgc_enabled: bool = False,
) -> Dict[str, Any]:
    """
    Core 15-minute MNQ & MGC Auto-Execution & Watchdog State Machine Cycle.
    """
    with _AUTO_LOCK:
        cfg = load_auto_trader_config()
        st = load_auto_trader_state()
        st["last_watchdog_ts"] = _now_et_str()

        # Run MGC 15m cycle if enabled (default False) or simulated
        mgc_res = _run_mgc_cycle_unlocked(
            cfg,
            st,
            simulated_mgc_snapshot=simulated_mgc_snapshot,
            simulated_mgc_plan_15m=simulated_mgc_plan_15m,
            force_mgc_enabled=force_mgc_enabled,
        )

        is_enabled = bool(cfg.get("enabled")) or force_enabled
        if not is_enabled and simulated_tws_snapshot is None:
            st["watchdog_status"] = "PAUSED (Auto-Trader Disabled)"
            _save_state_unlocked(st)
            return {"config": cfg, "state": st, "mgc_result": mgc_res}

        # 1. Obtain TWS MNQ Position & Open Orders Snapshot
        snap = (
            simulated_tws_snapshot
            if simulated_tws_snapshot is not None
            else _query_tws_mnq_snapshot(list(cfg.get("tws_ports") or [7497, 7496, 4002, 4001]), symbol="MNQ", exchange="CME")
        )
        if not snap.get("connected"):
            st["watchdog_status"] = f"TWS OFFLINE ({snap.get('error', 'No connection')})"
            _save_state_unlocked(st)
            return {"config": cfg, "state": st, "tws_snapshot": snap, "mgc_result": mgc_res}

        live_net_pos = int(snap.get("live_net_pos", 0))
        stop_orders = list(snap.get("stop_orders") or [])
        tp_orders = list(snap.get("tp_orders") or [])
        entry_orders = list(snap.get("entry_orders") or [])
        live_bar_obj = snap.get("live_bar") or {}
        curr_mkt_px = float(live_bar_obj.get("close", 0.0) or 0.0)

        # Sum working stop quantity in the correct exit direction
        expected_exit_action = "SELL" if live_net_pos > 0 else ("BUY" if live_net_pos < 0 else None)
        working_stop_qty = sum(
            int(o.get("totalQuantity", 0))
            for o in stop_orders
            if expected_exit_action is None or o.get("action") == expected_exit_action
        )
        working_tp_qty = sum(
            int(o.get("totalQuantity", 0))
            for o in tp_orders
            if expected_exit_action is None or o.get("action") == expected_exit_action
        )

        st["live_net_pos"] = live_net_pos
        st["working_stop_qty"] = working_stop_qty
        st["working_tp_qty"] = working_tp_qty

        # =====================================================================
        # 2. WATCHDOG: MANUAL OR BRACKET CLOSE DETECTION (live_net_pos == 0)
        # =====================================================================
        if live_net_pos == 0:
            prev_state = st.get("position_state", "FLAT")
            active_tr = st.get("active_trade")
            now_epoch = time.time()
            entry_filled_flag = bool(isinstance(active_tr, dict) and active_tr.get("entry_filled", False))
            grace_active = (
                prev_state in ("PENDING_ENTRY", "OPEN")
                and len(entry_orders) > 0
                and isinstance(active_tr, dict)
                and not entry_filled_flag
                and (now_epoch - float(active_tr.get("opened_epoch", 0.0))) < float(cfg.get("entry_timeout_sec", 180))
            )

            if grace_active:
                st["position_state"] = "PENDING_ENTRY"
                st["watchdog_status"] = f"PENDING ENTRY ({active_tr.get('direction')} 2 MNQ waiting for fill)"
                _save_state_unlocked(st)
                return {"config": cfg, "state": st, "tws_snapshot": snap, "mgc_result": mgc_res}

            # If we were previously OPEN (or PENDING_ENTRY expired) and now live_net_pos == 0:
            # The user closed the position manually (or the bracket exited, or unfilled entry timed out).
            # Cancel any leftover/orphaned stop or limit orders, record trade outcome if filled, and disarm the watchdog!
            if prev_state in ("OPEN", "PENDING_ENTRY") or len(stop_orders) > 0 or len(tp_orders) > 0 or len(entry_orders) > 0:
                had_orphan_orders = len(stop_orders) > 0 or len(tp_orders) > 0 or len(entry_orders) > 0
                unfilled_entry_expired = bool(len(entry_orders) > 0 and not entry_filled_flag)
                if had_orphan_orders:
                    if simulated_tws_snapshot is None:
                        _execute_tws_action(list(cfg.get("tws_ports", [7497])), "cancel_all_mnq_orders", {})
                    if unfilled_entry_expired:
                        _append_log(
                            st,
                            "ENTRY_TIMEOUT_CANCELLED",
                            "MNQ entry limit order did not fill within timeout — cancelled unfilled bracket orders.",
                        )
                    else:
                        _append_log(
                            st,
                            "MANUAL_CLOSE_CLEANUP",
                            "Position is FLAT (0 MNQ) — cancelled leftover child orders and disarmed stop-loss/time-stop watchdog.",
                        )
                elif prev_state == "OPEN":
                    _append_log(
                        st,
                        "POSITION_CLOSED",
                        "Position closed manually or via bracket (0 MNQ). Auto-watchdog disarmed — will NOT place stops on 0 position.",
                    )

                was_filled_trade = bool(
                    isinstance(active_tr, dict)
                    and (entry_filled_flag or (prev_state == "OPEN" and not unfilled_entry_expired))
                )
                if was_filled_trade:
                    est_exit_px = curr_mkt_px or float(
                        active_tr.get("target2_px" if active_tr.get("tp1_filled") else "entry_px", 0.0)
                    )
                    exit_rsn = "MANUAL_CLOSE" if had_orphan_orders else ("TP2_TARGET" if active_tr.get("tp1_filled") else "BRACKET_EXIT")
                    _record_closed_trade(st, active_tr, est_exit_px, exit_rsn)

                st["position_state"] = "FLAT"
                st["active_direction"] = None
                st["active_trade"] = None
                st["working_stop_qty"] = 0
                st["working_tp_qty"] = 0
                st["watchdog_status"] = "IDLE (Flat — 0 MNQ, Watchdog Disarmed)"
                _save_state_unlocked(st)
                return {"config": cfg, "state": st, "tws_snapshot": snap, "mgc_result": mgc_res}

            st["watchdog_status"] = "IDLE (Flat — 0 MNQ, Watchdog Disarmed)"

        # =====================================================================
        # 3. WATCHDOG: ACTIVE POSITION PROTECTION (live_net_pos != 0)
        # =====================================================================
        else:
            live_dir = "LONG" if live_net_pos > 0 else "SHORT"
            abs_pos = abs(live_net_pos)
            st["position_state"] = "OPEN"
            st["active_direction"] = live_dir

            active_tr = st.get("active_trade") or {}
            # If a position exists but active_trade metadata was empty, synthesize protective fallback levels
            if not active_tr or active_tr.get("direction") != live_dir:
                eval_rt = nq_ml_engine.evaluate_nq_realtime_multi_tf(
                    threshold_pct=float(cfg.get("min_prob_pct", 55.0)),
                    horizon_k=8,
                )
                tf15 = eval_rt.get("timeframes", {}).get("15m", {})
                plan15 = tf15.get("trade_plan") or {}
                c_now = float(tf15.get("close") or plan15.get("entry_px") or 29800.0)
                atr_now = max(15.0, float(tf15.get("atr14_pts") or 35.0))
                stop_fb = round_to_nq_tick(c_now - 1.2 * atr_now if live_dir == "LONG" else c_now + 1.2 * atr_now)
                tp_fb = round_to_nq_tick(c_now + 1.5 * atr_now if live_dir == "LONG" else c_now - 1.5 * atr_now)
                active_tr = {
                    "symbol": "MNQ",
                    "direction": live_dir,
                    "entry_px": round_to_nq_tick(c_now),
                    "stop_px": stop_fb,
                    "target1_px": tp_fb,
                    "target2_px": tp_fb,
                    "opened_epoch": time.time(),
                    "bar_time": _now_et_str(),
                    "entry_filled": True,
                    "max_abs_pos": abs_pos,
                }
                st["active_trade"] = active_tr
            else:
                was_already_filled = bool(active_tr.get("entry_filled", False))
                active_tr["entry_filled"] = True
                prev_max = int(active_tr.get("max_abs_pos", 0))
                if not was_already_filled and len(entry_orders) == 0:
                    prev_max = max(prev_max, 2, abs_pos)
                else:
                    prev_max = max(prev_max, abs_pos)
                active_tr["max_abs_pos"] = prev_max
                st["active_trade"] = active_tr

            # If 1 of 2 units filled and the 2nd parent entry order timed out, cancel the remaining unfilled entry order
            if (
                abs_pos == 1
                and len(entry_orders) > 0
                and (time.time() - float(active_tr.get("opened_epoch", 0.0))) >= float(cfg.get("entry_timeout_sec", 180))
            ):
                if simulated_tws_snapshot is None:
                    _execute_tws_action(list(cfg.get("tws_ports", [7497])), "cancel_unfilled_entry_orders", {})
                entry_orders = []

            # 3A. Detect TP1 Fill (2 MNQ -> 1 MNQ remaining) and move Unit 2 Stop-Loss to Breakeven
            if (
                abs_pos == 1
                and not active_tr.get("tp1_filled")
                and len(entry_orders) == 0
                and int(active_tr.get("max_abs_pos", 0)) >= 2
            ):
                be_px = round_to_nq_tick(float(active_tr.get("entry_px", active_tr.get("stop_px", 0.0))))
                prot_pts = round_to_nq_tick(float(cfg.get("stop_limit_protection_pts", 20.0)))
                be_lmt_px = round_to_nq_tick(be_px - prot_pts if live_dir == "LONG" else be_px + prot_pts)
                active_tr["tp1_filled"] = True
                active_tr["stop_px"] = be_px
                st["active_trade"] = active_tr
                if simulated_tws_snapshot is None:
                    _execute_tws_action(
                        list(cfg.get("tws_ports", [7497])),
                        "move_stop_to_breakeven",
                        {"be_stop_px": be_px, "be_stop_lmt_px": be_lmt_px},
                    )
                _append_log(
                    st,
                    "TP1_FILLED_BE_MOVE",
                    f"TP1 filled (1 MNQ remaining) — moved Unit 2 Runner Stop-Loss to Breakeven ({be_px:.2f}).",
                )

            # Check if any open MNQ contract is missing a working Stop-Loss!
            if working_stop_qty < abs_pos:
                missing_qty = abs_pos - working_stop_qty
                exit_act = "SELL" if live_dir == "LONG" else "BUY"
                stop_px = round_to_nq_tick(float(active_tr.get("stop_px", 0.0)))
                prot_pts = round_to_nq_tick(float(cfg.get("stop_limit_protection_pts", 20.0)))
                stop_lmt_px = round_to_nq_tick(stop_px - prot_pts if live_dir == "LONG" else stop_px + prot_pts)
                tp_px = round_to_nq_tick(
                    float(active_tr.get("target2_px" if abs_pos == 1 else "target1_px", active_tr.get("entry_px", stop_px)))
                )

                repair_payload = {
                    "missing_qty": missing_qty,
                    "exit_action": exit_act,
                    "stop_px": stop_px,
                    "stop_lmt_px": stop_lmt_px,
                    "target_px": tp_px,
                }
                if simulated_tws_snapshot is None:
                    rep_res = _execute_tws_action(
                        list(cfg.get("tws_ports", [7497])),
                        "place_missing_protection_bracket",
                        repair_payload,
                    )
                else:
                    rep_res = {"ok": True, "order_ids": [9901, 9902], "oca_group": "SIM_WD_OCA"}

                if rep_res.get("ok"):
                    st["working_stop_qty"] = abs_pos
                    st["watchdog_status"] = (
                        f"REPAIRED BRACKET ({live_dir} {abs_pos} MNQ — placed missing {missing_qty}x GTC Stop @ {stop_px:.2f} & TP @ {tp_px:.2f})"
                    )
                    _append_log(
                        st,
                        "WATCHDOG_REPAIR",
                        f"Detected {missing_qty} unprotected MNQ ({live_dir}). Placed GTC Outside-RTH Stop @ {stop_px:.2f} + TP @ {tp_px:.2f}.",
                    )
                else:
                    st["watchdog_status"] = f"WATCHDOG ALERT: Failed to place stop for {missing_qty} MNQ ({rep_res.get('error')})"
            else:
                st["watchdog_status"] = (
                    f"PROTECTED ({live_dir} {abs_pos} MNQ | Working GTC Stop Qty: {working_stop_qty} @ {float(active_tr.get('stop_px', 0.0)):.2f})"
                )

            # Because we only allow ONE direction / ONE position at a time, return immediately while in a position!
            _save_state_unlocked(st)
            return {"config": cfg, "state": st, "tws_snapshot": snap, "mgc_result": mgc_res}

        # =====================================================================
        # 4. NEW 15M ENTRY EVALUATION (Only when FLAT: live_net_pos == 0)
        # =====================================================================
        if simulated_plan_15m is not None:
            plan_15m = simulated_plan_15m
        elif simulated_tws_snapshot is not None:
            plan_15m = None
        else:
            eval_rt = nq_ml_engine.evaluate_nq_realtime_multi_tf(
                live_bar=snap.get("live_bar"),
                threshold_pct=float(cfg.get("min_prob_pct", 55.0)),
                horizon_k=8,
            )
            tf15_obj = eval_rt.get("timeframes", {}).get("15m", {})
            cand_plan = tf15_obj.get("trade_plan") or {}
            last_bar_ts = int(tf15_obj.get("last_bar_timestamp", 0) or 0)
            plan_bar_ts = int(cand_plan.get("bar_timestamp", 0) or 0)
            now_et_ts = _now_et_naive_epoch()
            is_recent_wallclock = plan_bar_ts > 0 and (now_et_ts - plan_bar_ts) <= 1800
            # Trigger if either the current developing 15m bar OR the immediately preceding 15m bar (<= 900s ago) triggered
            is_fresh_15m_trigger = is_recent_wallclock and (
                bool(tf15_obj.get("is_live_bar_triggered"))
                or (bool(tf15_obj.get("triggered")) and last_bar_ts > 0 and (last_bar_ts - plan_bar_ts) <= 900)
            )
            plan_15m = cand_plan if is_fresh_15m_trigger else None

        if not plan_15m:
            _save_state_unlocked(st)
            return {"config": cfg, "state": st, "tws_snapshot": snap, "mgc_result": mgc_res}

        prob_pct = float(plan_15m.get("prob_pct", 0.0))
        pred_ratio = float(plan_15m.get("pred_ratio", 0.0))
        rr_t1 = float(plan_15m.get("rr_target1", 0.0))
        bar_ts = int(plan_15m.get("bar_timestamp", 0) or 0)
        bar_key = f"15m_{bar_ts}_{plan_15m.get('direction', '')}"

        passes_gates = (
            prob_pct >= float(cfg.get("min_prob_pct", 55.0))
            and pred_ratio >= float(cfg.get("min_pred_ratio", 1.35))
            and rr_t1 >= float(cfg.get("min_rr_target1", 1.00))
        )

        if not passes_gates:
            _save_state_unlocked(st)
            return {"config": cfg, "state": st, "tws_snapshot": snap, "mgc_result": mgc_res}

        if bar_key in (st.get("traded_bar_keys") or []):
            _save_state_unlocked(st)
            return {"config": cfg, "state": st, "tws_snapshot": snap, "mgc_result": mgc_res}

        # Build 2-MNQ Split GTC Outside-RTH Bracket
        spec = build_2mnq_gtc_bracket_spec(plan_15m, cfg=cfg)
        if simulated_tws_snapshot is None:
            exec_res = _execute_tws_action(
                list(cfg.get("tws_ports", [7497])),
                "place_2mnq_bracket",
                {"bracket_spec": spec},
            )
        else:
            exec_res = {"ok": True, "order_ids": [1001, 1002, 1003, 1004, 1005, 1006], "localSymbol": "MNQZ6"}

        if exec_res.get("ok"):
            traded_keys = list(st.get("traded_bar_keys") or [])
            traded_keys.append(bar_key)
            st["traded_bar_keys"] = traded_keys[-200:]
            st["position_state"] = "OPEN"
            st["active_direction"] = spec["direction"]
            st["live_net_pos"] = 2 if spec["direction"] == "LONG" else -2
            st["working_stop_qty"] = 2
            st["working_tp_qty"] = 2
            st["active_trade"] = {
                **spec,
                "order_ids": exec_res.get("order_ids", []),
                "opened_at": _now_et_str(),
                "opened_epoch": time.time(),
                "tp1_filled": False,
                "entry_filled": False,
                "max_abs_pos": 0,
            }
            st["watchdog_status"] = (
                f"OPENED {spec['direction']} 2 MNQ @ {spec['entry_px']:.2f} "
                f"(GTC Stop @ {spec['stop_px']:.2f} | TP1 @ {spec['target1_px']:.2f} | TP2 @ {spec['target2_px']:.2f})"
            )
            _append_log(
                st,
                "TRADE_OPENED",
                f"Opened {spec['direction']} 2 MNQ [15m] @ {spec['entry_px']:.2f} | "
                f"SL {spec['stop_px']:.2f} | TP1 {spec['target1_px']:.2f} | TP2 {spec['target2_px']:.2f} (GTC Outside RTH)",
            )

            if cfg.get("gmail_enabled", True) and simulated_tws_snapshot is None:
                email_res = send_position_opened_gmail(st["active_trade"], cfg=cfg)
                e_logs = list(st.get("email_log") or [])
                e_logs.append(email_res)
                st["email_log"] = e_logs[-30:]
                _append_log(st, "GMAIL_NOTIFICATION", f"Email notification: {email_res.get('status')}")

        _save_state_unlocked(st)
        return {"config": cfg, "state": st, "tws_snapshot": snap, "execution": exec_res, "mgc_result": mgc_res}


def get_auto_trader_status() -> Dict[str, Any]:
    cfg = load_auto_trader_config()
    st = load_auto_trader_state()
    safe_cfg = dict(cfg)
    safe_cfg["has_gmail_app_password"] = bool(cfg.get("gmail_app_password"))
    safe_cfg["gmail_app_password"] = "********" if cfg.get("gmail_app_password") else ""
    email_logs = list(st.get("email_log") or [])
    last_email_status = email_logs[-1].get("status", "") if email_logs else ""
    live_perf = compute_live_performance_summary(st)
    flat_status = {
        "enabled": bool(cfg.get("enabled")),
        "mgc_enabled": bool(cfg.get("mgc_enabled", False)),
        "position_state": st.get("position_state", "FLAT"),
        "active_direction": st.get("active_direction"),
        "live_net_pos": int(st.get("live_net_pos", 0)),
        "live_working_stop_qty": int(st.get("working_stop_qty", 0)),
        "live_working_tp_qty": int(st.get("working_tp_qty", 0)),
        "last_watchdog_status": st.get("watchdog_status", "IDLE"),
        "mgc_position_state": st.get("mgc_position_state", "FLAT"),
        "mgc_active_direction": st.get("mgc_active_direction"),
        "mgc_live_net_pos": int(st.get("mgc_live_net_pos", 0)),
        "mgc_live_working_stop_qty": int(st.get("mgc_working_stop_qty", 0)),
        "mgc_live_working_tp_qty": int(st.get("mgc_working_tp_qty", 0)),
        "mgc_last_watchdog_status": st.get("mgc_watchdog_status", "OFF (MGC Auto-Trader Disabled)"),
        "last_action": st.get("last_action", ""),
        "gmail_to": cfg.get("gmail_to", "amir.baram@gmail.com"),
        "gmail_password_configured": bool(cfg.get("gmail_app_password")),
        "pivot_email_alerts_enabled": bool(cfg.get("pivot_email_alerts_enabled", True)),
        "pivot_email_tickers": list(cfg.get("pivot_email_tickers") or ["NQ=F", "GC=F", "CL=F", "BTC=F", "SI=F"]),
        "pivot_email_timeframes": list(cfg.get("pivot_email_timeframes") or ["5m", "15m", "1h", "4h"]),
        "last_email_status": last_email_status,
        "tws_connected": "OFFLINE" not in str(st.get("watchdog_status", "")) and "OFFLINE" not in str(st.get("mgc_watchdog_status", "")),
        "tws_port": (cfg.get("tws_ports") or [7497])[0],
        "live_contract": "MNQ + MGC" if cfg.get("mgc_enabled") else "MNQ",
        "live_performance": live_perf,
    }
    return {
        "ok": True,
        "enabled": bool(cfg.get("enabled")),
        "mgc_enabled": bool(cfg.get("mgc_enabled", False)),
        "pivot_email_alerts_enabled": bool(cfg.get("pivot_email_alerts_enabled", True)),
        "pivot_email_tickers": list(cfg.get("pivot_email_tickers") or ["NQ=F", "GC=F", "CL=F", "BTC=F", "SI=F"]),
        "pivot_email_timeframes": list(cfg.get("pivot_email_timeframes") or ["5m", "15m", "1h", "4h"]),
        "status": flat_status,
        "live_performance": live_perf,
        "config": safe_cfg,
        "state": st,
        "daemon_running": bool(_DAEMON_THREAD is not None and _DAEMON_THREAD.is_alive()),
    }


def _daemon_loop() -> None:
    while not _DAEMON_STOP_EVENT.is_set():
        try:
            cfg = load_auto_trader_config()
            if cfg.get("enabled") or cfg.get("mgc_enabled"):
                run_watchdog_and_execution_cycle()
            if cfg.get("gmail_enabled", True) and cfg.get("pivot_email_alerts_enabled", True):
                if (time.time() - _LAST_PIVOT_EMAIL_SCAN_TS) >= 30.0:
                    scan_and_email_realtime_pivots(sync_live=True)
            wait_s = max(2.0, float(cfg.get("watchdog_interval_sec", 5.0)))
        except Exception:
            wait_s = 5.0
        _DAEMON_STOP_EVENT.wait(wait_s)


def ensure_auto_trader_daemon_started() -> bool:
    global _DAEMON_THREAD
    if _DAEMON_THREAD is not None and _DAEMON_THREAD.is_alive():
        return True
    _DAEMON_STOP_EVENT.clear()
    _DAEMON_THREAD = threading.Thread(target=_daemon_loop, name="nq_15m_auto_trader_watchdog", daemon=True)
    _DAEMON_THREAD.start()
    return True

