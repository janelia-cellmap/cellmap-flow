"""Shared SQL helpers for the instance-review workflow.

Backs both scripts/review_query.py (CLI) and the dashboard's
routes/review_routes.py (HTTP). Schema is defined by
scripts/build_instance_index.py:

  instances(id, cz, cy, cx, cz_nm, cy_nm, cx_nm,
            bz0, bz1, by0, by1, bx0, bx1,
            vox, faces, sphericity, fm_score,
            rank_smallest, rank_fm, rank_random)
  ledger(instance_id PK, review_state, reviewed_at, reviewer,
         edit_details_json, entry_method)
  meta(key, value)

Ledger is pre-initialized one row per instance with NULL fields;
verdicts are recorded by UPDATE, undo clears the same row to NULL.

`entry_method` is auto-added to legacy DBs by `open_db` (idempotent
ALTER); rows recorded before the column existed read as NULL.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Optional


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


def open_db(db_path: str) -> sqlite3.Connection:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"review db not found: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(ledger)")}
    if "entry_method" not in cols:
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
    inst_cols = {r["name"] for r in conn.execute("PRAGMA table_info(instances)")}
    if order_col not in inst_cols:
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
    row = conn.execute(
        "SELECT i.*, l.review_state, l.reviewed_at, l.reviewer, "
        "       l.edit_details_json, l.entry_method "
        "FROM instances i LEFT JOIN ledger l ON l.instance_id = i.id "
        "WHERE i.id = ?",
        (instance_id,),
    ).fetchone()
    return dict(row) if row is not None else None


def find_instance_at_voxel(conn: sqlite3.Connection,
                           label_voxel_zyx: tuple) -> Optional[dict]:
    """Open the labels zarr declared in `meta.source_zarr` and return the
    instance record at `(vz, vy, vx)` (label-zarr voxel coords).

    Returns the same shape as `get_instance`. Returns `None` if:
      - the voxel is background (label = 0)
      - the voxel is out-of-bounds for the labels zarr
      - the looked-up label id has no row in the instances table
        (e.g. erased-and-reindexed; treated as a soft miss, not an error)

    Raises:
      - FileNotFoundError if meta.source_zarr is missing or doesn't exist on disk.
    """
    import zarr  # local import: dashboard process already imports zarr,
                 # but keep this module standalone-importable for CLI use
    meta = get_meta(conn)
    src = meta.get("source_zarr")
    if not src:
        raise FileNotFoundError("meta.source_zarr is unset for this review db")
    if not os.path.exists(src):
        raise FileNotFoundError(f"meta.source_zarr not on disk: {src}")
    # Try multiscale s0 first (the build_label_pyramid.py output convention),
    # else fall back to opening src as a single array.
    try:
        arr = zarr.open(os.path.join(src, "s0"), mode="r")
    except Exception:
        arr = zarr.open(src, mode="r")

    vz, vy, vx = label_voxel_zyx
    # Round to nearest voxel and bounds-check
    iz, iy, ix = int(round(vz)), int(round(vy)), int(round(vx))
    nz, ny, nx = arr.shape[-3:]  # tolerate possible leading channel axis
    if not (0 <= iz < nz and 0 <= iy < ny and 0 <= ix < nx):
        return None

    if arr.ndim == 4:
        # (c, z, y, x) — pick channel 0; labels are normally 3-D so this
        # is just defensive
        label_id = int(arr[0, iz, iy, ix])
    else:
        label_id = int(arr[iz, iy, ix])
    if label_id == 0:
        return None
    return get_instance(conn, label_id)


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
      'next' (queue advance — trusted),
      'select_at' (cursor lookup — historically buggy, see notes/260427),
      'show' (Go to ID — trusted).
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

    inst_cols = {r["name"] for r in conn.execute("PRAGMA table_info(instances)")}
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
