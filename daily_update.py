#!/usr/bin/env python3
"""daily_update.py — Automated Daily Retrieval & Multi-App Pipeline Orchestrator.

Institutional EOD Universe Synchronization for Ports 8782, 8785, 8786, and 8787:
1. Downloads incremental EOD price bars from yfinance across active universe (datastore.py).
2. Computes daily Relative Strength (RS) ratings, Stage 2 posture, and 52W high distance (update_rs_database.py).
3. Discovers fresh Episodic Pivots, evaluates zero-lookahead V2 conviction and trade simulations (scripts/update_ep_dataset.py).
4. Mines new Qullamaggie Momentum Continuation (QM) setups and cross-references against fresh EPs (update_qm_database.py).
5. Hot-reloads all active server instances (8782, 8785, 8786, 8787) via `/api/reload_dataset` with zero downtime.

Usage:
  .venv/bin/python daily_update.py --daemon               # Start scheduled daemon (16:45 US/Eastern)
  .venv/bin/python daily_update.py --daemon --now         # Run immediately, then continue on schedule
  .venv/bin/python daily_update.py --once                 # Execute single pipeline run and exit
  .venv/bin/python daily_update.py --status               # Check daemon process and dataset timestamps
  .venv/bin/python daily_update.py --stop                 # Gracefully terminate running daemon
  .venv/bin/python daily_update.py --check                # Inspect dataset readiness without modifying
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
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

# Root path setup
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import datastore
import trading_calendar
import update_rs_database
import update_qm_database
import scripts.update_ep_dataset as ep_updater

PID_FILE = ROOT / "data" / "daily_updater.pid"
LOG_FILE = ROOT / "logs" / "daily_update.log"
TARGET_SERVERS = [
    (8782, "Historical EP Review & Strategy Backtester"),
    (8785, "QM Setup & EP Cross-Reference Explorer"),
    (8786, "Institutional RS Universe Explorer"),
    (8787, "Optimized EP Strategy Explorer"),
]

# Ensure data and logs directories exist
PID_FILE.parent.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("daily_updater")


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


def reload_server(port: int, app_name: str) -> Dict[str, Any]:
    """Sends a hot-reload POST request to the running server."""
    url = f"http://127.0.0.1:{port}/api/reload_dataset"
    try:
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req, timeout=8) as resp:
            if resp.status == 200:
                raw = resp.read().decode("utf-8")
                try:
                    payload = json.loads(raw)
                except Exception:
                    payload = raw
                logger.info(f"✅ Hot-reloaded Port {port} ({app_name}): {payload}")
                return {"port": port, "name": app_name, "status": "reloaded", "details": payload}
            logger.warning(f"⚠️ Port {port} returned HTTP {resp.status}")
            return {"port": port, "name": app_name, "status": f"http_{resp.status}"}
    except urllib.error.URLError:
        logger.info(f"ℹ️ Port {port} ({app_name}) is offline (changes will load automatically on next launch).")
        return {"port": port, "name": app_name, "status": "offline"}
    except Exception as e:
        logger.warning(f"⚠️ Failed reloading port {port}: {e}")
        return {"port": port, "name": app_name, "status": f"error: {e}"}


def get_dataset_timestamps() -> Dict[str, Any]:
    """Inspects latest data date across all modules."""
    spy = datastore.load_bars("SPY")
    spy_latest = spy.index[-1].strftime("%Y-%m-%d") if spy is not None and not spy.empty else None

    rs_latest, _, rs_count = update_rs_database.get_db_status()
    qm_latest, _, qm_count = update_qm_database.get_qm_status()

    ep_latest = None
    ep_count = 0
    ep_path = ROOT / "data" / "simulations" / "ep_combined_study_scored.parquet"
    if ep_path.exists():
        try:
            df_ep = pd.read_parquet(ep_path, columns=["date"])
            ep_count = len(df_ep)
            if not df_ep.empty:
                ep_latest = str(df_ep["date"].max())
        except Exception:
            pass

    return {
        "spy_bars_latest": spy_latest,
        "rs_db_latest": rs_latest,
        "rs_sessions_count": rs_count,
        "ep_study_latest": ep_latest,
        "ep_events_count": ep_count,
        "qm_setups_latest": qm_latest,
        "qm_setups_count": qm_count,
    }


def run_pipeline(
    full_bars: bool = False,
    force: bool = False,
    skip_download: bool = False,
) -> Dict[str, Any]:
    """Runs the complete 5-step daily retrieval and dataset update sequence."""
    t_start = time.time()
    logger.info("══════════════════════════════════════════════════════════════════════════════")
    logger.info("🚀 STARTING EOD DAILY UNIVERSE RETRIEVAL & UPDATE PIPELINE")
    logger.info("══════════════════════════════════════════════════════════════════════════════")

    summary: Dict[str, Any] = {
        "timestamp": dt.datetime.now().isoformat(),
        "steps": {},
        "server_reloads": {},
        "success": False,
    }

    # STEP 1: Download latest EOD daily bars
    logger.info("▶ STEP 1/5: Downloading incremental daily bars from yfinance...")
    t0 = time.time()
    try:
        if not skip_download:
            datastore.update_daily(full=full_bars)
            summary["steps"]["bars_download"] = {"status": "ok", "duration_s": round(time.time() - t0, 1)}
            logger.info(f"✅ Price bars updated in {time.time() - t0:.1f}s.")
        else:
            logger.info("⏭️ Skipped bar download as requested.")
            summary["steps"]["bars_download"] = {"status": "skipped"}
    except Exception as e:
        logger.error(f"⚠️ Error downloading price bars: {e}. Continuing with existing datastore...")
        summary["steps"]["bars_download"] = {"status": "error", "error": str(e)}

    # STEP 2: Update Institutional RS Database
    logger.info("▶ STEP 2/5: Synchronizing Institutional RS Database (Port 8786)...")
    t0 = time.time()
    try:
        rs_updated = update_rs_database.update_database(force=force)
        summary["steps"]["rs_database"] = {
            "status": "updated" if rs_updated else "up_to_date",
            "duration_s": round(time.time() - t0, 1),
        }
        logger.info(f"✅ RS Database step completed in {time.time() - t0:.1f}s.")
    except Exception as e:
        logger.error(f"❌ Error updating RS Database: {e}")
        summary["steps"]["rs_database"] = {"status": "error", "error": str(e)}

    # STEP 3: Update EP Historical Study Dataset (Port 8782)
    logger.info("▶ STEP 3/5: Scanning fresh EPs, V2 conviction models & simulations (Port 8782)...")
    t0 = time.time()
    try:
        ep_updater.update_pipeline(skip_download=True, full_download=False, dry_run=False)
        summary["steps"]["ep_study"] = {"status": "ok", "duration_s": round(time.time() - t0, 1)}
        logger.info(f"✅ EP Study step completed in {time.time() - t0:.1f}s.")
    except Exception as e:
        logger.error(f"❌ Error updating EP Study dataset: {e}")
        summary["steps"]["ep_study"] = {"status": "error", "error": str(e)}

    # STEP 4: Update QM Setups Database & EP Cross-References (Port 8785)
    logger.info("▶ STEP 4/5: Mining QM Setups & cross-referencing EP igniters (Port 8785)...")
    t0 = time.time()
    try:
        qm_updated = update_qm_database.run_update(force=force)
        summary["steps"]["qm_setups"] = {
            "status": "updated" if qm_updated else "up_to_date",
            "duration_s": round(time.time() - t0, 1),
        }
        logger.info(f"✅ QM Setups step completed in {time.time() - t0:.1f}s.")
    except Exception as e:
        logger.error(f"❌ Error updating QM Setups: {e}")
        summary["steps"]["qm_setups"] = {"status": "error", "error": str(e)}

    # STEP 5: Hot-reload active servers across all 4 ports
    logger.info("▶ STEP 5/5: Notifying live dashboard applications (8782, 8785, 8786, 8787)...")
    for port, app_name in TARGET_SERVERS:
        summary["server_reloads"][str(port)] = reload_server(port, app_name)

    elapsed_total = round(time.time() - t_start, 1)
    summary["duration_total_s"] = elapsed_total
    summary["success"] = True

    logger.info("══════════════════════════════════════════════════════════════════════════════")
    logger.info(f"🎉 DAILY RETRIEVAL & UPDATE PIPELINE COMPLETED IN {elapsed_total}s")
    logger.info("══════════════════════════════════════════════════════════════════════════════")
    return summary


def get_next_run_time(
    target_time_str: str = "16:45",
    now_et: Optional[dt.datetime] = None,
) -> dt.datetime:
    """Calculates the next scheduled execution time in US/Eastern."""
    tz = ZoneInfo("America/New_York")
    if now_et is None:
        now_et = dt.datetime.now(tz)

    hour, minute = map(int, target_time_str.split(":"))
    candidate_date = now_et.date()
    target_dt = dt.datetime(
        candidate_date.year,
        candidate_date.month,
        candidate_date.day,
        hour,
        minute,
        tzinfo=tz,
    )

    # If today is a trading day and target_dt is in the future, schedule today
    if trading_calendar.is_trading_day(candidate_date) and target_dt > now_et:
        return target_dt

    # Otherwise step forward day by day until finding the next valid exchange trading day
    while True:
        candidate_date += dt.timedelta(days=1)
        if trading_calendar.is_trading_day(candidate_date):
            return dt.datetime(
                candidate_date.year,
                candidate_date.month,
                candidate_date.day,
                hour,
                minute,
                tzinfo=tz,
            )


def is_daemon_running() -> Tuple[bool, Optional[int]]:
    """Checks if updater daemon is currently running."""
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
        # Stale PID file
        try:
            PID_FILE.unlink(missing_ok=True)
        except Exception:
            pass
        return False, None
    except PermissionError:
        # Process exists but owned by another user
        return True, pid


def stop_daemon() -> bool:
    """Sends SIGTERM to running daemon and cleans up PID file."""
    running, pid = is_daemon_running()
    if not running or pid is None:
        print("ℹ️ No daily updater daemon is currently running.")
        return True

    print(f"🛑 Stopping daily updater daemon (PID: {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
        # Wait up to 5 seconds for clean exit
        for _ in range(50):
            time.sleep(0.1)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
        PID_FILE.unlink(missing_ok=True)
        print("✅ Daily updater daemon stopped successfully.")
        return True
    except Exception as e:
        print(f"⚠️ Error stopping daemon PID {pid}: {e}")
        return False


def run_daemon(
    target_time_str: str = "16:45",
    run_now: bool = False,
    full_bars: bool = False,
    force: bool = False,
) -> None:
    """Runs the persistent background daily updater loop."""
    running, existing_pid = is_daemon_running()
    if running:
        print(f"❌ Error: Daily updater is already running (PID: {existing_pid}).")
        print("Run `daily_update.py --stop` to terminate the active daemon first.")
        sys.exit(1)

    # Write current PID
    current_pid = os.getpid()
    PID_FILE.write_text(str(current_pid))
    # In daemon mode, log exclusively to file handler to prevent double-writes under nohup
    setup_logger(log_to_stdout=False)

    logger.info(f"🚀 Daily Updater Daemon initialized (PID: {current_pid}).")
    logger.info(f"Target Schedule: {target_time_str} US/Eastern on trading days.")
    logger.info(f"Logging to: {LOG_FILE}")

    _stop_flag = False

    def _handle_exit_signal(sig, frame):
        nonlocal _stop_flag
        logger.info(f"🛑 Received termination signal {sig}. Exiting gracefully...")
        _stop_flag = True

    signal.signal(signal.SIGTERM, _handle_exit_signal)
    signal.signal(signal.SIGINT, _handle_exit_signal)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    try:
        if run_now:
            logger.info("⚡ Immediate update requested (--now). Running pipeline now...")
            run_pipeline(full_bars=full_bars, force=force)

        tz = ZoneInfo("America/New_York")
        while not _stop_flag:
            now_et = dt.datetime.now(tz)
            next_run = get_next_run_time(target_time_str, now_et=now_et)
            wait_seconds = (next_run - now_et).total_seconds()
            hours_wait = wait_seconds / 3600.0

            logger.info(
                f"⏰ Next scheduled run: {next_run.strftime('%Y-%m-%d %H:%M:%S %Z')} "
                f"({hours_wait:.2f} hours from now)."
            )

            # Sleep in short increments of 5 seconds to remain responsive to SIGTERM
            while wait_seconds > 0 and not _stop_flag:
                sleep_chunk = min(wait_seconds, 5.0)
                time.sleep(sleep_chunk)
                now_et = dt.datetime.now(tz)
                wait_seconds = (next_run - now_et).total_seconds()

            if _stop_flag:
                break

            logger.info(f"🔔 Target time reached ({target_time_str} ET). Launching daily update...")
            try:
                run_pipeline(full_bars=full_bars, force=force)
            except Exception as e:
                logger.error(f"❌ Unhandled exception during scheduled update: {e}", exc_info=True)

            # Pause 60 seconds after execution to prevent immediate re-trigger within same minute
            time.sleep(60)

    finally:
        PID_FILE.unlink(missing_ok=True)
        logger.info("👋 Daily updater daemon stopped and PID file cleared.")


def main():
    parser = argparse.ArgumentParser(
        description="Daily Universe Retrieval & Multi-App Update Orchestrator (Ports 8782, 8785, 8786)"
    )
    parser.add_argument("--daemon", action="store_true", help="Run as scheduled background daemon")
    parser.add_argument("--once", action="store_true", help="Execute single update cycle immediately and exit")
    parser.add_argument("--now", action="store_true", help="Execute update immediately before scheduling in daemon mode")
    parser.add_argument("--status", action="store_true", help="Check daemon status and dataset timestamps")
    parser.add_argument("--stop", action="store_true", help="Stop running daily updater daemon")
    parser.add_argument("--check", action="store_true", help="Inspect dataset timestamps and status without running updates")
    parser.add_argument("--time", type=str, default="16:45", help="Target scheduled time in HH:MM ET (default: 16:45)")
    parser.add_argument("--full-bars", action="store_true", help="Force full download of price bars instead of incremental")
    parser.add_argument("--force", action="store_true", help="Force recalculation across all databases")
    args = parser.parse_args()

    if args.stop:
        success = stop_daemon()
        sys.exit(0 if success else 1)

    if args.status or args.check:
        running, pid = is_daemon_running()
        ts = get_dataset_timestamps()
        print("══════════════════════════════════════════════════════════════════════════════")
        print(" 📊 INSTITUTIONAL DAILY UPDATER STATUS & DATASET TIMESTAMPS")
        print("══════════════════════════════════════════════════════════════════════════════")
        if running:
            print(f"  • Daemon Status    : 🟢 RUNNING (PID: {pid})")
            next_run = get_next_run_time(args.time)
            print(f"  • Next Scheduled   : {next_run.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        else:
            print("  • Daemon Status    : ⚪ STOPPED")
        print(f"  • Log File         : {LOG_FILE}")
        print("------------------------------------------------------------------------------")
        print(f"  • Datastore (SPY)  : {ts['spy_bars_latest'] or 'None'}")
        print(f"  • Port 8786 (RS)   : {ts['rs_db_latest'] or 'None'} ({ts['rs_sessions_count']} daily sessions)")
        print(f"  • Port 8782 (EP)   : {ts['ep_study_latest'] or 'None'} ({ts['ep_events_count']:,} total events)")
        print(f"  • Port 8785 (QM)   : {ts['qm_setups_latest'] or 'None'} ({ts['qm_setups_count']:,} total setups)")
        print("══════════════════════════════════════════════════════════════════════════════")
        sys.exit(0)

    if args.once:
        setup_logger(log_to_stdout=True)
        res = run_pipeline(full_bars=args.full_bars, force=args.force)
        sys.exit(0 if res.get("success") else 1)

    if args.daemon:
        run_daemon(
            target_time_str=args.time,
            run_now=args.now,
            full_bars=args.full_bars,
            force=args.force,
        )
        sys.exit(0)

    # If no flags provided, show help
    parser.print_help()


if __name__ == "__main__":
    main()
