"""Quote-tape recorder — open-hardening plan D (Amir approved 2026-07-09).

Records the RAW IEX quote batches the poller receives, plus periodic DAYVOL/PREMARKET state
snapshots, to data/quote_tape/<date>/. Purpose: every live incident becomes REPLAYABLE — after
the close, sim_open.py can re-run any captured tape through the real ingestion (apply_premarket /
apply_quotes / rvol_leaders) at any simulated hour, so open-bugs are fixed and validated offline
instead of re-testing the same failures live day after day.

Capture policy (keeps a day to a few tens of MB):
  FULL batches   during the bug windows — the whole pre-market phase and the first 45 min of RTH
  SAMPLED        1 in SAMPLE_EVERY cycles the rest of the session
  STATE          DAYVOL + PREMARKET json snapshot every SNAP_MIN minutes (small)

Files (per day, appended):
  quotes.jsonl.gz   one line per recorded cycle: {"t","et","phase","n","quotes":[...]}
  state.jsonl.gz    one line per snapshot:       {"t","et","dayvol":{...},"premarket":{...}}

Everything is fail-open: recording errors must never disturb the poller.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import threading
from zoneinfo import ZoneInfo

import config

SAMPLE_EVERY = 10            # off-window cycles: record 1 in N
SNAP_MIN = 5                 # state snapshot cadence, minutes
_ET = ZoneInfo("America/New_York")
_lock = threading.Lock()
_mem = {"cycle": 0, "last_snap": None}


def _et_now() -> dt.datetime:
    return dt.datetime.now(_ET).replace(tzinfo=None)


def _dir():
    d = config.DATA_DIR / "quote_tape" / _et_now().date().isoformat()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _full_window(et: dt.datetime) -> bool:
    t = et.time()
    return (dt.time(7, 55) <= t <= dt.time(9, 30)) or (dt.time(9, 30) <= t <= dt.time(10, 15))


def record_cycle(phase: str | None, quotes: list[dict]) -> None:
    """Called by the poller after each fetch. Cheap append of one gzip JSON line."""
    try:
        _mem["cycle"] += 1
        et = _et_now()
        if not (_full_window(et) or _mem["cycle"] % SAMPLE_EVERY == 0):
            return
        line = json.dumps({"t": dt.datetime.now().isoformat(timespec="seconds"),
                           "et": et.isoformat(timespec="seconds"), "phase": phase,
                           "n": len(quotes), "quotes": quotes}, default=str)
        with _lock, gzip.open(_dir() / "quotes.jsonl.gz", "at") as f:
            f.write(line + "\n")
    except Exception:
        pass


def maybe_snapshot(dayvol: dict, premarket: dict) -> None:
    """DAYVOL/PREMARKET state every SNAP_MIN minutes — the simulator's starting state."""
    try:
        et = _et_now()
        last = _mem["last_snap"]
        if last is not None and (et - last).total_seconds() < SNAP_MIN * 60:
            return
        _mem["last_snap"] = et
        line = json.dumps({"t": dt.datetime.now().isoformat(timespec="seconds"),
                           "et": et.isoformat(timespec="seconds"),
                           "dayvol": dict(dayvol), "premarket": dict(premarket)}, default=str)
        with _lock, gzip.open(_dir() / "state.jsonl.gz", "at") as f:
            f.write(line + "\n")
    except Exception:
        pass


def read_tape(date: str) -> list[dict]:
    """Load a day's recorded cycles (for sim_open.py)."""
    p = config.DATA_DIR / "quote_tape" / date / "quotes.jsonl.gz"
    if not p.exists():
        return []
    out = []
    with gzip.open(p, "rt") as f:
        for ln in f:
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
    return out
