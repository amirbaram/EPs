"""Chart study journal — SQLite-backed store for annotated ticker+date entries.

Each entry records a symbol, the as-of date the chart was viewed, a list of setups
(each with its own setup-appeared date), the timeframe, and free-text comments.

    import journal
    journal.add("NVDA", "2026-06-30",
                [{"setup": "qm_breakout", "setup_date": "2026-06-15"}],
                tf="1D", comments="clean base, volume dry-up")
    journal.list_all()     -> [{id, symbol, date, setup, setups, tf, comments, saved_at}, ...]
    journal.update(id, comments="new", setups=[...])
    journal.delete(id)
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path

import config

DB_PATH = config.DATA_DIR / "journal.db"

_CREATE = """
CREATE TABLE IF NOT EXISTS entries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    setup       TEXT    NOT NULL DEFAULT 'other',
    setups_json TEXT    NOT NULL DEFAULT '[]',
    tf          TEXT    NOT NULL DEFAULT '1D',
    comments    TEXT    NOT NULL DEFAULT '',
    saved_at    TEXT    NOT NULL
)
"""


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute(_CREATE)
    # migrate: add setups_json if table existed without it
    cols = [r[1] for r in c.execute("PRAGMA table_info(entries)").fetchall()]
    if "setups_json" not in cols:
        c.execute("ALTER TABLE entries ADD COLUMN setups_json TEXT NOT NULL DEFAULT '[]'")
    c.commit()
    return c


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    try:
        d["setups"] = json.loads(d.get("setups_json") or "[]")
    except Exception:
        d["setups"] = []
    return d


# ── write ─────────────────────────────────────────────────────────────────────

def add(symbol: str, date: str, setups: list[dict], tf: str = "1D", comments: str = "") -> dict:
    """
    setups: [{"setup": str, "setup_date": str}, ...]
    Primary `setup` column = first entry's setup name (for quick filtering).
    """
    if not setups:
        setups = [{"setup": "other", "setup_date": date}]
    primary = setups[0].get("setup", "other")
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO entries (symbol, date, setup, setups_json, tf, comments, saved_at) VALUES (?,?,?,?,?,?,?)",
            (symbol.upper(), date, primary, json.dumps(setups), tf, comments,
             dt.datetime.now().isoformat(timespec="seconds")),
        )
        row = c.execute("SELECT * FROM entries WHERE id=?", (cur.lastrowid,)).fetchone()
    return _row(row)


def update(entry_id: int, comments: str, setups: list[dict] | None = None) -> dict | None:
    with _conn() as c:
        if setups is not None:
            primary = setups[0].get("setup", "other") if setups else "other"
            c.execute(
                "UPDATE entries SET comments=?, setup=?, setups_json=? WHERE id=?",
                (comments, primary, json.dumps(setups), entry_id),
            )
        else:
            c.execute("UPDATE entries SET comments=? WHERE id=?", (comments, entry_id))
        row = c.execute("SELECT * FROM entries WHERE id=?", (entry_id,)).fetchone()
    return _row(row) if row else None


def delete(entry_id: int) -> bool:
    with _conn() as c:
        n = c.execute("DELETE FROM entries WHERE id=?", (entry_id,)).rowcount
    return n > 0


# ── read ──────────────────────────────────────────────────────────────────────

def list_all(setup: str | None = None) -> list[dict]:
    with _conn() as c:
        if setup:
            rows = c.execute(
                "SELECT * FROM entries WHERE setup=? ORDER BY saved_at DESC", (setup,)
            ).fetchall()
        else:
            rows = c.execute("SELECT * FROM entries ORDER BY saved_at DESC").fetchall()
    return [_row(r) for r in rows]


def get(entry_id: int) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM entries WHERE id=?", (entry_id,)).fetchone()
    return _row(row) if row else None


def setups_summary() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT setup, COUNT(*) as count FROM entries GROUP BY setup ORDER BY count DESC"
        ).fetchall()
    return [dict(r) for r in rows]
