#!/usr/bin/env python3
"""intraday_watchlist_monitor.py — Real-Time Intraday Active Watchlist Climax & Exhaustion Monitor.

Monitors active institutional EP positions during exchange market hours (09:30 - 16:00 US/Eastern):
1. Loads active in-flight trades (status == 'Active Long') from serve_optimized_ep.
2. Periodically downloads intraday (5-min) price bars via yfinance across active positions.
3. Constructs a live synthetic daily bar [Open_today, High_today, Low_today, Current_Price, Volume_cum].
4. Re-evaluates technical extension, RSI divergence, and the calibrated Extreme Peak ML Classifier
   (LightGBM + XGBoost ensemble from data/models/ep_extreme_peak_classifier.pkl) at the day's high extreme.
5. If exhaustion probability reaches or exceeds the archetype climax threshold (>=55% for 50 SMA / Pinnacle Elite,
   >=45% for 20 EMA / Emerging Leaders, or >=50% baseline), triggers an institutional 'SELL HALF' alert.
6. Atomically saves live telemetry to data/cache/intraday_watchlist_status.json and notifies Port 8787.

Usage:
  .venv/bin/python intraday_watchlist_monitor.py --once              # Single polling cycle and print alerts
  .venv/bin/python intraday_watchlist_monitor.py --once --force      # Run single cycle even outside market hours
  .venv/bin/python intraday_watchlist_monitor.py --daemon            # Run continuous monitor loop (every 120s)
  .venv/bin/python intraday_watchlist_monitor.py --daemon --force    # Run daemon continuously regardless of market hours
  .venv/bin/python intraday_watchlist_monitor.py --status            # Inspect monitor cache and active alerts
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# Ensure repository root is on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import datastore
import trading_calendar
import serve_optimized_ep as srv

try:
    import yfinance as yf
except ImportError:
    yf = None

CACHE_FILE = ROOT / "data" / "cache" / "intraday_watchlist_status.json"
SETTINGS_FILE = ROOT / "data" / "cache" / "monitor_settings.json"
PID_FILE = ROOT / "data" / "intraday_monitor.pid"
LOG_FILE = ROOT / "logs" / "intraday_monitor.log"

CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
PID_FILE.parent.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("intraday_monitor")


def get_configured_interval(default_sec: int = 300) -> int:
    """Reads user-configured monitor polling interval from cache settings (default 300s / 5min)."""
    if SETTINGS_FILE.exists():
        try:
            d = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            sec = int(d.get("interval_sec", default_sec))
            return max(60, min(3600, sec))
        except Exception:
            pass
    return default_sec


def setup_logger(log_to_stdout: bool = True) -> None:
    """Configures multi-sink logger to both file and stdout."""
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %Z",
    )

    fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if log_to_stdout:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)


def is_market_open(now_et: Optional[dt.datetime] = None) -> Tuple[bool, str]:
    """Checks whether US equity markets are currently in regular trading session (09:30 - 16:00 ET)."""
    tz = ZoneInfo("America/New_York")
    if now_et is None:
        now_et = dt.datetime.now(tz)

    today = now_et.date()
    if not trading_calendar.is_trading_day(today):
        return False, f"Today ({today}) is not an exchange trading session (Weekend/Holiday)."

    t = now_et.time()
    open_time = dt.time(9, 30)
    close_time = dt.time(16, 0)

    if t < open_time:
        return False, f"Pre-market ({t.strftime('%H:%M:%S')} ET < 09:30:00 ET)."
    elif t > close_time:
        return False, f"Post-market ({t.strftime('%H:%M:%S')} ET > 16:00:00 ET)."
    else:
        return True, f"Regular Trading Hours ({t.strftime('%H:%M:%S')} ET)."


def get_active_long_watchlist() -> List[Dict[str, Any]]:
    """Loads active positions from serve_optimized_ep."""
    all_items = srv.get_active_watchlist()
    active_longs = [x for x in all_items if x.get("status") == "Active Long"]
    return active_longs


def fetch_intraday_bars_for_symbols(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Fetches 5-minute intraday bars for the given symbols using yfinance."""
    if not symbols or yf is None:
        return {}

    sym_bars: Dict[str, pd.DataFrame] = {}
    try:
        # Download batch across active tickers (typically 15-30 tickers)
        data = yf.download(
            tickers=symbols,
            period="1d",
            interval="5m",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if data is None or data.empty:
            return {}

        if len(symbols) == 1:
            sym = symbols[0]
            if not data.empty and "Close" in data:
                clean = data.dropna(subset=["Close"]).copy()
                if not clean.empty:
                    sym_bars[sym] = clean
        else:
            for sym in symbols:
                try:
                    if sym in data.columns.levels[0]:
                        sub = data[sym].dropna(subset=["Close"]).copy()
                        if not sub.empty:
                            sym_bars[sym] = sub
                except Exception:
                    pass
    except Exception as e:
        logger.warning(f"Batch intraday download warning: {e}. Attempting fallback...")
        for sym in symbols:
            try:
                t = yf.Ticker(sym)
                df = t.history(period="1d", interval="5m")
                if df is not None and not df.empty:
                    sym_bars[sym] = df
            except Exception:
                pass

    return sym_bars


def evaluate_intraday_position(
    pos: Dict[str, Any],
    intraday_df: Optional[pd.DataFrame],
    target_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Fuses historical daily bars with live intraday data and evaluates real-time exhaustion."""
    sym = pos["symbol"]
    trigger_px = float(pos.get("trigger_price", 0.0))
    stop_px = float(pos.get("stop_price", 0.0))
    rs_val = float(pos.get("rs_score", 75.0))
    v2_conv = float(pos.get("v2_conviction_50", 0.10))
    rec_ma_key = pos.get("recommended_ma_key", "ema20")
    rec_ma_name = pos.get("recommended_ma_name", "20 EMA Intermediate Momentum Trail")
    climax_th = float(pos.get("climax_threshold", 0.45))
    climax_th_pct = float(pos.get("climax_threshold_pct", 45.0))

    bars = datastore.load_bars(sym)
    if bars is None or len(bars) < 20:
        return {
            "symbol": sym,
            "status": "No Historical Bars",
            "alert_state": "NORMAL",
            "exhaustion_prob": 0.0,
            "exhaustion_pct": 0.0,
        }

    # If live intraday data is available, extract today's high, low, close, vol
    if intraday_df is not None and not intraday_df.empty:
        # Standardize column names to title case
        intraday_df.columns = [c.capitalize() for c in intraday_df.columns]
        h_today = float(intraday_df["High"].max())
        l_today = float(intraday_df["Low"].min())
        o_today = float(intraday_df["Open"].iloc[0])
        c_now = float(intraday_df["Close"].iloc[-1])
        v_today = float(intraday_df["Volume"].sum())
        bar_time_str = str(intraday_df.index[-1])
        today_date_str = intraday_df.index[-1].strftime("%Y-%m-%d")
    else:
        # Fallback to latest daily bar
        last_b = bars.iloc[-1]
        h_today = float(last_b["high"])
        l_today = float(last_b["low"])
        o_today = float(last_b["open"])
        c_now = float(last_b["close"])
        v_today = float(last_b["volume"])
        bar_time_str = bars.index[-1].strftime("%Y-%m-%d")
        today_date_str = bars.index[-1].strftime("%Y-%m-%d")

    # Build live synthetic bar
    today_ts = pd.Timestamp(today_date_str)
    live_row = pd.DataFrame([{
        "open": o_today,
        "high": h_today,
        "low": l_today,
        "close": c_now,
        "volume": v_today,
    }], index=[today_ts])

    bars_prior = bars[~bars.index.isin([today_ts])].copy()
    live_bars = pd.concat([bars_prior, live_row])

    # Moving averages on live bars
    ema10_val = float(live_bars["close"].ewm(span=10, adjust=False).mean().iloc[-1])
    ema20_val = float(live_bars["close"].ewm(span=20, adjust=False).mean().iloc[-1])
    sma50_val = float(live_bars["close"].rolling(50, min_periods=20).mean().iloc[-1])
    rec_ma_px = sma50_val if rec_ma_key == "sma50" else (ema10_val if rec_ma_key == "ema10" else ema20_val)
    ma_buf_pct = round(((c_now - rec_ma_px) / rec_ma_px) * 100.0, 1) if rec_ma_px > 0 else 0.0

    # Predict Extreme High Climax Exhaustion
    exh_prob = srv.predict_exhaustion_for_bars(live_bars, trigger_px, stop_px, rs_val=rs_val, v2_conv=v2_conv)
    exh_pct = round(exh_prob * 100.0, 1)

    is_alert = (exh_prob >= climax_th)
    unrealized_pct = round(((c_now - trigger_px) / trigger_px) * 100.0, 1)
    risk_pts = max(0.01, trigger_px - stop_px)
    unrealized_r = round((c_now - trigger_px) / risk_pts, 2)
    peak_intraday_r = round((h_today - trigger_px) / risk_pts, 2)

    if is_alert:
        alert_state = "🚨 INTRADAY_SELL_HALF"
        alert_action = (
            f"🚨 INTRADAY SELL HALF ALERT: Current Day High (${h_today:.2f}) reached {exh_pct}% ML Exhaustion "
            f"(>= {climax_th_pct}% threshold). Sell 50% partial profit into this extreme peak to lock +{peak_intraday_r:.2f}R; "
            f"retain remaining 50% runner on rising {rec_ma_name} (${rec_ma_px:.2f}, buffer: {ma_buf_pct:+.1f}%)."
        )
    elif exh_pct >= (climax_th_pct - 5.0):
        alert_state = "⚠️ ELEVATED_WATCH"
        alert_action = f"⚠️ Elevated Exhaustion Watch: {exh_pct}% approaching {climax_th_pct}% trigger threshold. Prepare sell-half order."
    else:
        alert_state = "🟢 HOLD_RUNNER"
        alert_action = f"🟢 Hold Runner: Current High Exhaustion {exh_pct}% is below {climax_th_pct}% hurdle. Trailing on {rec_ma_name} (${rec_ma_px:.2f})."

    return {
        "symbol": sym,
        "archetype": pos.get("archetype", "General"),
        "trade_type": pos.get("trade_type", "Trade 1: Base EP Delayed Breakout (Days 2-5)"),
        "trigger_price": trigger_px,
        "stop_price": stop_px,
        "current_price": round(c_now, 2),
        "intraday_high": round(h_today, 2),
        "intraday_low": round(l_today, 2),
        "intraday_open": round(o_today, 2),
        "unrealized_pct": unrealized_pct,
        "unrealized_r": unrealized_r,
        "peak_intraday_r": peak_intraday_r,
        "recommended_ma_name": rec_ma_name,
        "recommended_ma_key": rec_ma_key,
        "recommended_ma_price": round(rec_ma_px, 2),
        "recommended_ma_buffer_pct": ma_buf_pct,
        "exhaustion_prob": round(exh_prob, 3),
        "exhaustion_pct": exh_pct,
        "climax_threshold": climax_th,
        "climax_threshold_pct": climax_th_pct,
        "alert_state": alert_state,
        "alert_action": alert_action,
        "is_alert": is_alert,
        "last_bar_time": bar_time_str,
    }


def run_intraday_cycle(force: bool = False) -> Dict[str, Any]:
    """Executes a single intraday check across active watchlist positions."""
    t0 = time.time()
    tz = ZoneInfo("America/New_York")
    now_et = dt.datetime.now(tz)
    market_open, reason = is_market_open(now_et)

    logger.info(f"Checking intraday positions at {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')} | Market: {reason}")

    if not market_open and not force:
        logger.info(f"⏭️ Market is currently closed. Skipping pull (run with --force to override).")
        # Load existing cache if available
        existing = {}
        if CACHE_FILE.exists():
            try:
                existing = json.loads(CACHE_FILE.read_text())
            except Exception:
                pass
        return {
            "status": "market_closed",
            "reason": reason,
            "timestamp_et": now_et.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "cached": existing,
        }

    # 1. Retrieve Active Longs
    active_positions = get_active_long_watchlist()
    symbols = [x["symbol"] for x in active_positions]
    logger.info(f"Found {len(active_positions)} active long positions: {', '.join(symbols) if symbols else 'None'}")

    if not symbols:
        payload = {
            "timestamp_et": now_et.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "market_open": market_open,
            "active_count": 0,
            "alerts_count": 0,
            "positions": {},
        }
        CACHE_FILE.write_text(json.dumps(payload, indent=2))
        return payload

    # 2. Fetch Intraday 5m Bars
    intraday_bars = fetch_intraday_bars_for_symbols(symbols)

    # 3. Evaluate Each Position
    evaluated_positions: Dict[str, Any] = {}
    alerts_count = 0

    for pos in active_positions:
        sym = pos["symbol"]
        df_intra = intraday_bars.get(sym)
        res = evaluate_intraday_position(pos, df_intra)
        evaluated_positions[sym] = res
        if res.get("is_alert"):
            alerts_count += 1
            logger.warning(f"🚨 ALERT TRIGGERED: {sym} hit {res['exhaustion_pct']}% exhaustion @ ${res['intraday_high']}!")

    payload = {
        "timestamp_et": now_et.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "market_open": market_open,
        "active_count": len(active_positions),
        "alerts_count": alerts_count,
        "positions": evaluated_positions,
        "elapsed_seconds": round(time.time() - t0, 2),
    }

    # Write atomic cache
    tmp_path = CACHE_FILE.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    os.replace(tmp_path, CACHE_FILE)

    logger.info(f"✅ Evaluated {len(active_positions)} positions ({alerts_count} alerts) in {time.time() - t0:.2f}s.")
    return payload


def is_daemon_running() -> Tuple[bool, Optional[int]]:
    """Checks if monitor daemon is currently running."""
    if not PID_FILE.exists():
        return False, None
    try:
        pid_str = PID_FILE.read_text().strip()
        if not pid_str:
            return False, None
        pid = int(pid_str)
        os.kill(pid, 0)
        return True, pid
    except (ProcessLookupError, ValueError):
        PID_FILE.unlink(missing_ok=True)
        return False, None
    except PermissionError:
        return True, pid


def stop_daemon() -> bool:
    """Terminates running monitor daemon."""
    running, pid = is_daemon_running()
    if not running or pid is None:
        print("ℹ️ No intraday monitor daemon is currently running.")
        return True

    print(f"🛑 Stopping intraday monitor daemon (PID: {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(30):
            time.sleep(0.1)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
        PID_FILE.unlink(missing_ok=True)
        print("✅ Daemon stopped successfully.")
        return True
    except Exception as e:
        print(f"⚠️ Error stopping daemon: {e}")
        return False


def run_daemon_loop(interval_sec: int = 300, force: bool = False) -> None:
    """Continuous polling loop."""
    running, pid = is_daemon_running()
    if running and pid != os.getpid():
        print(f"⚠️ Intraday monitor daemon is already running (PID: {pid}).")
        sys.exit(1)

    PID_FILE.write_text(str(os.getpid()))
    active_sec = get_configured_interval(interval_sec)
    logger.info(f"🚀 Intraday Watchlist Monitor Daemon started (PID: {os.getpid()}, Interval: {active_sec}s / {round(active_sec/60.0, 1)}m)")

    def handle_sigterm(signum, frame):
        logger.info("🛑 Received termination signal. Cleaning up...")
        PID_FILE.unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    try:
        while True:
            try:
                run_intraday_cycle(force=force)
            except Exception as e:
                logger.error(f"Error in monitor cycle: {e}", exc_info=True)
            active_sec = get_configured_interval(interval_sec)
            for _ in range(active_sec):
                time.sleep(1)
    finally:
        PID_FILE.unlink(missing_ok=True)


def print_status() -> None:
    """Prints current monitor status and active alerts."""
    running, pid = is_daemon_running()
    status_str = f"🟢 RUNNING (PID: {pid})" if running else "⚪ STOPPED"
    print(f"Monitor Status: {status_str}")

    tz = ZoneInfo("America/New_York")
    now_et = dt.datetime.now(tz)
    mkt_open, reason = is_market_open(now_et)
    print(f"Market Session: {'🟢 OPEN' if mkt_open else '⚪ CLOSED'} — {reason}")

    active_sec = get_configured_interval(300)
    print(f"Configured Interval: {active_sec}s ({round(active_sec/60.0, 1)} min)")

    if CACHE_FILE.exists():
        try:
            data = json.loads(CACHE_FILE.read_text())
            print(f"Last Cycle: {data.get('timestamp_et')} ({data.get('elapsed_seconds', 0)}s)")
            print(f"Active Longs: {data.get('active_count', 0)}")
            print(f"Alerts Count: {data.get('alerts_count', 0)}")
            positions = data.get("positions", {})
            if positions:
                print("\nActive Positions Telemetry:")
                for sym, info in positions.items():
                    alert_tag = info.get("alert_state", "NORMAL")
                    exh = info.get("exhaustion_pct", 0.0)
                    th = info.get("climax_threshold_pct", 50.0)
                    curr = info.get("current_price", 0.0)
                    h_day = info.get("intraday_high", 0.0)
                    ma = info.get("recommended_ma_name", "MA")
                    print(f"  • {sym:6s}: High: ${h_day:<7.2f} (Exhaustion: {exh:>4.1f}% / Thresh: {th:.0f}%) | Mark: ${curr:<7.2f} | Status: {alert_tag}")
        except Exception as e:
            print(f"Error reading cache: {e}")
    else:
        print("No cache file found at data/cache/intraday_watchlist_status.json.")


def main():
    parser = argparse.ArgumentParser(description="Real-Time Intraday Active Watchlist Climax & Exhaustion Monitor")
    parser.add_argument("--once", action="store_true", help="Execute single check cycle and exit")
    parser.add_argument("--daemon", action="store_true", help="Run background monitor daemon")
    parser.add_argument("--status", action="store_true", help="Inspect monitor state and active alerts")
    parser.add_argument("--stop", action="store_true", help="Stop running monitor daemon")
    parser.add_argument("--force", action="store_true", help="Force execution even outside regular market hours")
    parser.add_argument("--interval", type=int, default=300, help="Daemon polling interval in seconds (default: 300 / 5min)")

    args = parser.parse_args()
    setup_logger(log_to_stdout=not args.daemon)

    if args.stop:
        stop_daemon()
    elif args.status:
        print_status()
    elif args.once:
        run_intraday_cycle(force=args.force)
    elif args.daemon:
        run_daemon_loop(interval_sec=args.interval, force=args.force)
    else:
        # Default to printing status if no arguments
        print_status()


if __name__ == "__main__":
    main()
