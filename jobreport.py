"""Lightweight progress heartbeats for long background lab runs (Amir 2026-07-09).

A run writes {tag, done, total, pct, elapsed, eta, msg, ts} to data/_progress/<tag>.json on
each tick, so progress can be read from a FILE instead of probing the live process. The file is
DELETED on clean completion; on error it is left behind with the traceback as a tombstone to
diagnose. Usage:

    with jobreport.Reporter("stfeat_5m", len(syms)) as rep:
        for i, s in enumerate(syms):
            ...
            rep.tick(i + 1, s)
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import config

PROG_DIR = config.DATA_DIR / "_progress"


def active() -> list[dict]:
    """All in-flight run heartbeats (for a quick 'what's running' read)."""
    if not PROG_DIR.exists():
        return []
    out = []
    for p in sorted(PROG_DIR.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            pass
    return out


class Reporter:
    def __init__(self, tag: str, total: int):
        self.tag, self.total, self.t0 = tag, max(int(total), 1), time.time()
        PROG_DIR.mkdir(parents=True, exist_ok=True)
        self.path = PROG_DIR / f"{tag}.json"
        self.tick(0, "starting")

    def tick(self, done: int, msg: str = "") -> None:
        el = time.time() - self.t0
        eta = (el / done * (self.total - done)) if done else 0.0
        self.path.write_text(json.dumps({
            "tag": self.tag, "done": int(done), "total": self.total,
            "pct": round(100 * done / self.total), "elapsed_s": round(el),
            "eta_s": round(eta), "msg": msg, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}))

    def __enter__(self) -> "Reporter":
        return self

    def __exit__(self, et, ev, tb) -> bool:
        if et is None:
            self.path.unlink(missing_ok=True)                 # clean completion -> remove the heartbeat
        else:                                                 # leave a tombstone with the traceback
            self.path.write_text(json.dumps({
                "tag": self.tag, "error": "".join(traceback.format_exception(et, ev, tb))[-2000:],
                "ts": time.strftime("%Y-%m-%d %H:%M:%S")}))
        return False
