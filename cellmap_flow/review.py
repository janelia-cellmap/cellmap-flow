"""SQL helpers for the instance-review workflow (the dashboard's Review tab).

A review index is one SQLite file (built outside
cellmap-flow) with three tables:

  instances(id, cz, cy, cx, cz_nm, cy_nm, cx_nm,
            bz0, bz1, by0, by1, bx0, bx1,
            vox, faces, sphericity, fm_score,
            rank_smallest, rank_fm, rank_random)
  ledger(instance_id PK, review_state, reviewed_at, reviewer,
         edit_details_json, entry_method)
  meta(key, value)

The ledger is pre-initialized with one row per instance and NULL fields;
a verdict is an UPDATE, and undo sets the row back to NULL.

Connections are read-only unless ``open_db(..., write=True)``, which the
verdict and undo paths use. Only a writable connection adds the
``entry_method`` column to an index that predates it; a read-only one
reads it as NULL.
"""

from __future__ import annotations

import json
import os
import pathlib
import sqlite3
import time
from typing import Optional

DB_SUFFIXES = (".sqlite", ".db")
REQUIRED_TABLES = ("instances", "ledger", "meta")

ORDER_COL = {
    "smallest":   "rank_smallest",
    "fm":         "rank_fm",
    "random":     "rank_random",
    "em_bright":  "rank_em_bright",
    "keep_lt65k": "rank_random_keep_lt65k",
    "keep_ge65k": "rank_random_keep_ge65k",
    "drop":       "rank_random_drop",
}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def resolve_db_path(path: str) -> str:
    """The real path of a review index, or an error.

    Only files named ``*.sqlite`` or ``*.db`` after resolving symlinks are
    accepted, and the name is checked before existence, so the dashboard
    route that takes this path cannot be used to probe for other files.

    Raises ValueError for any other name, FileNotFoundError when the file
    does not exist.
    """
    real = os.path.realpath(os.path.expanduser(path))
    if not real.lower().endswith(DB_SUFFIXES):
        raise ValueError(
            f"a review index must be a {' or '.join(DB_SUFFIXES)} file"
        )
    if not os.path.isfile(real):
        raise FileNotFoundError(f"review index not found: {path}")
    return real


def _columns(conn: sqlite3.Connection, table: str) -> list:
    return [r["name"] for r in conn.execute(f"PRAGMA table_info({table})")]


def open_db(db_path: str, write: bool = False) -> sqlite3.Connection:
    """Open a review index; read-only unless ``write``.

    Raises ValueError when the file is not a review index (a SQLite
    database with the instances, ledger and meta tables).
    """
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"review index not found: {db_path}")
    uri = pathlib.Path(db_path).resolve().as_uri() + ("" if write else "?mode=ro")
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        tables = {
            r["name"]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    except sqlite3.DatabaseError as e:
        conn.close()
        raise ValueError(f"not a review index: {e}") from e
    missing = [t for t in REQUIRED_TABLES if t not in tables]
    if missing:
        conn.close()
        raise ValueError(f"not a review index: no {', '.join(missing)} table")
    if write and "entry_method" not in _columns(conn, "ledger"):
        conn.execute("ALTER TABLE ledger ADD COLUMN entry_method TEXT")
        conn.commit()
    return conn


def get_meta(conn: sqlite3.Connection) -> dict:
    rows = conn.execute("SELECT key, value FROM meta").fetchall()
    return {r["key"]: r["value"] for r in rows}


def count_instances(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM instances").fetchone()[0]


def get_next(conn: sqlite3.Connection, order: str,
             min_vox: Optional[int] = None,
             skip_rank: Optional[int] = None) -> Optional[dict]:
    """Return the next unreviewed instance in the chosen queue, or None.

    skip_rank advances past a given rank value — needed so clicking Next
    repeatedly without verdicting actually advances through the queue
    (without it, the query returns the same "first unreviewed" row
    every call, because nothing got marked reviewed).
    """
    if order not in ORDER_COL:
        raise ValueError(f"order must be one of {list(ORDER_COL)}; got {order!r}")
    order_col = ORDER_COL[order]
    if order_col not in _columns(conn, "instances"):
        raise ValueError(
            f"queue {order!r} requires column {order_col!r} which is "
            f"not present in this catalog"
        )

    clauses = []
    params = []
    if min_vox is not None:
        clauses.append("AND i.vox >= ?")
        params.append(min_vox)
    if skip_rank is not None:
        clauses.append(f"AND i.{order_col} > ?")
        params.append(skip_rank)

    extra_sql = " ".join(clauses)

    sql = f"""
        SELECT i.id, i.cz, i.cy, i.cx, i.cz_nm, i.cy_nm, i.cx_nm,
               i.bz0, i.bz1, i.by0, i.by1, i.bx0, i.bx1,
               i.vox, i.sphericity, i.fm_score,
               i.{order_col} AS rank
        FROM instances i
        LEFT JOIN ledger l ON l.instance_id = i.id
        WHERE i.{order_col} IS NOT NULL
          AND (l.review_state IS NULL)
          {extra_sql}
        ORDER BY i.{order_col} ASC
        LIMIT 1
    """
    row = conn.execute(sql, params).fetchone()
    return dict(row) if row is not None else None


def get_instance(conn: sqlite3.Connection,
                 instance_id: int) -> Optional[dict]:
    """Full instance record (joined with ledger state), or None if not found."""
    entry_method = (
        "l.entry_method" if "entry_method" in _columns(conn, "ledger")
        else "NULL AS entry_method"
    )
    row = conn.execute(
        "SELECT i.*, l.review_state, l.reviewed_at, l.reviewer, "
        f"       l.edit_details_json, {entry_method} "
        "FROM instances i LEFT JOIN ledger l ON l.instance_id = i.id "
        "WHERE i.id = ?",
        (instance_id,),
    ).fetchone()
    return dict(row) if row is not None else None


VALID_VERDICTS = ("blessed", "edited", "erased")
VALID_ENTRY_METHODS = ("next", "select_at", "show", "pick")


def record_verdict(conn: sqlite3.Connection, instance_id: int, verdict: str,
                   reviewer: str,
                   edit_details: Optional[dict] = None,
                   entry_method: Optional[str] = None) -> dict:
    """Write a verdict to the ledger. Returns the updated row.

    Valid verdicts:
      - "blessed"  — correct detection, keep as-is.
      - "edited"   — partially correct, edit_details describes the fix.
      - "erased"   — false positive, wholesale paint with background
                     (value 1 in the downstream correction zarr,
                     indicating "not mito" for training).

    entry_method records HOW the reviewer arrived at this instance:
      'next' (queue advance), 'show' (Go to ID), 'pick' (the t key in
      the viewer), or 'select_at' (a cursor lookup older indexes used).
    NULL means unknown (legacy rows pre-dating this column).

    Raises ValueError for unknown verdict, unknown entry_method, or
    unknown instance_id.
    """
    if verdict not in VALID_VERDICTS:
        raise ValueError(
            f"verdict must be one of {VALID_VERDICTS}; got {verdict!r}"
        )
    if entry_method is not None and entry_method not in VALID_ENTRY_METHODS:
        raise ValueError(
            f"entry_method must be one of {VALID_ENTRY_METHODS} or None; "
            f"got {entry_method!r}"
        )

    ed_json = json.dumps(edit_details) if edit_details is not None else None
    with conn:
        cur = conn.execute(
            "UPDATE ledger SET review_state=?, reviewed_at=?, reviewer=?, "
            "       edit_details_json=?, entry_method=? "
            "WHERE instance_id=?",
            (verdict, _now(), reviewer, ed_json, entry_method, instance_id),
        )
        if cur.rowcount == 0:
            raise ValueError(f"no ledger row for instance_id={instance_id}")

    row = conn.execute(
        "SELECT * FROM ledger WHERE instance_id=?", (instance_id,)
    ).fetchone()
    return dict(row)


def undo_verdict(conn: sqlite3.Connection, instance_id: int) -> dict:
    """Clear the ledger row for instance_id (NULL all verdict fields)."""
    with conn:
        cur = conn.execute(
            "UPDATE ledger SET review_state=NULL, reviewed_at=NULL, "
            "       reviewer=NULL, edit_details_json=NULL, "
            "       entry_method=NULL "
            "WHERE instance_id=?",
            (instance_id,),
        )
        if cur.rowcount == 0:
            raise ValueError(f"no ledger row for instance_id={instance_id}")

    row = conn.execute(
        "SELECT * FROM ledger WHERE instance_id=?", (instance_id,)
    ).fetchone()
    return dict(row)


def get_progress(conn: sqlite3.Connection) -> dict:
    """Aggregate review progress: per-state counts + per-queue counts."""
    total = count_instances(conn)

    by_state_rows = conn.execute(
        "SELECT COALESCE(review_state, 'unreviewed') AS state, COUNT(*) AS n "
        "FROM ledger GROUP BY state"
    ).fetchall()
    by_state = {r["state"]: r["n"] for r in by_state_rows}

    inst_cols = set(_columns(conn, "instances"))
    queues = {}
    for q, col in ORDER_COL.items():
        if col not in inst_cols:
            continue
        q_total = conn.execute(
            f"SELECT COUNT(*) FROM instances WHERE {col} IS NOT NULL"
        ).fetchone()[0]
        q_reviewed = conn.execute(
            f"SELECT COUNT(*) FROM ledger l JOIN instances i "
            f"ON l.instance_id = i.id "
            f"WHERE i.{col} IS NOT NULL AND l.review_state IS NOT NULL"
        ).fetchone()[0]
        queues[q] = {"total": q_total, "reviewed": q_reviewed}

    meta = get_meta(conn)
    return {
        "total": total,
        "by_state": by_state,
        "queues": queues,
        "source_zarr": meta.get("source_zarr"),
        "built_at": meta.get("built_at"),
        "voxel_size_nm": json.loads(meta["voxel_size_nm"])
                          if "voxel_size_nm" in meta else None,
        "offset_nm": json.loads(meta["offset_nm"])
                     if "offset_nm" in meta else None,
    }
