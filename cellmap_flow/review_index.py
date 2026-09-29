"""Build a review index, the SQLite file the dashboard's Review tab opens,
from an instance segmentation.

    python -m cellmap_flow.review_index LABELS OUT.sqlite [--queue NAME ...]

LABELS is anything ImageDataInterface reads (zarr v2/v3, N5, precomputed;
a multiscale group is read at its finest level), with 0 as background. The
index has the schema cellmap_flow.review reads: one ``instances`` row per
label with its voxel count, bounding box (voxels, half-open) and centroid
(voxels, and in world nm), plus one ``rank_<queue>`` column per queue; one
empty ``ledger`` row per label; and ``meta``.

The volume is read in blocks of whole chunks and only per-label sums are
kept, so it need not fit in memory.

Queues are chosen with ``--queue`` from QUEUES. Other orderings are data:
any ``rank_<name>`` column added to ``instances`` later (say, from
per-instance intensity) shows up in the tab as a queue called ``<name>``.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from itertools import product
from typing import Dict, Optional, Sequence

import numpy as np

# The queues this can rank by, with the description the tab shows.
QUEUES = {
    "smallest": "smallest first",
    "largest": "largest first",
    "random": "random order",
}
DEFAULT_QUEUES = ("smallest", "random")
# Blocks are whole chunks, at least this many voxels along each axis.
MIN_BLOCK_EDGE = 128
# SQLite integers are signed 64-bit.
MAX_LABEL = 2**63 - 1


def _block_stats(block: np.ndarray, origin) -> np.ndarray:
    """Per non-zero label in ``block``, one int64 row: id, voxel count,
    z/y/x index sums, then the half-open bbox (z0, y0, x0, z1, y1, x1)."""
    flat = block.ravel()
    nonzero = np.flatnonzero(flat)
    if nonzero.size == 0:
        return np.zeros((0, 11), dtype=np.int64)
    labels = flat[nonzero]
    if labels.max() > MAX_LABEL or labels.min() < 0:
        raise ValueError(f"label ids must be in 1..{MAX_LABEL}")
    order = np.argsort(labels, kind="stable")
    labels, nonzero = labels[order], nonzero[order]
    ids, starts, counts = np.unique(labels, return_index=True, return_counts=True)
    coords = [c + o for c, o in zip(np.unravel_index(nonzero, block.shape), origin)]
    columns = [ids.astype(np.int64), counts]
    columns += [np.add.reduceat(c, starts) for c in coords]
    columns += [np.minimum.reduceat(c, starts) for c in coords]
    columns += [np.maximum.reduceat(c, starts) + 1 for c in coords]
    return np.stack(columns, axis=1).astype(np.int64)


def _ranks(queue: str, vox: np.ndarray, ids: np.ndarray, seed: int) -> np.ndarray:
    if queue == "smallest":
        order = np.lexsort((ids, vox))
    elif queue == "largest":
        order = np.lexsort((ids, -vox))
    elif queue == "random":
        order = np.random.default_rng(seed).permutation(len(ids))
    else:
        raise ValueError(f"unknown queue {queue!r}; choose from {list(QUEUES)}")
    ranks = np.empty(len(ids), dtype=np.int64)
    ranks[order] = np.arange(len(ids))
    return ranks


def build_index(
    labels_path: str,
    db_path: str,
    queues: Sequence[str] = DEFAULT_QUEUES,
    seed: int = 0,
    block_shape: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    """Write the review index for ``labels_path`` to ``db_path``.

    Refuses to overwrite ``db_path``: its ledger holds the verdicts.
    Returns ``{"n_instances", "voxel_size_nm", "offset_nm"}``.
    """
    from cellmap_flow.image_data_interface import ImageDataInterface
    from cellmap_flow.io.metadata import read_array_meta

    unknown = [q for q in queues if q not in QUEUES]
    if unknown:
        raise ValueError(f"unknown queue(s) {unknown}; choose from {list(QUEUES)}")
    if not db_path.lower().endswith((".sqlite", ".db")):
        raise ValueError("the index must be a .sqlite or .db file")
    if os.path.exists(db_path):
        raise FileExistsError(f"{db_path} exists; its ledger may hold verdicts")

    idi = ImageDataInterface(labels_path, normalize=False, input_norms=[])
    meta = read_array_meta(idi.path).spatial()
    store = idi.ts
    if store.rank != 3 or meta.axes not in (("z", "y", "x"), ("",) * 3):
        raise ValueError(
            f"expected a 3-D z, y, x segmentation; {idi.path} has axes {meta.axes}"
        )
    lo = np.asarray(store.domain.inclusive_min)
    hi = np.asarray(store.domain.exclusive_max)
    chunk = np.asarray(meta.chunk_shape)
    if block_shape is None:
        block_shape = chunk * np.maximum(1, -(-MIN_BLOCK_EDGE // chunk))
    block_shape = np.asarray(block_shape)

    parts = []
    for start in product(*(range(0, n, b) for n, b in zip(hi - lo, block_shape))):
        start = np.asarray(start)
        stop = np.minimum(start + block_shape, hi - lo)
        index = tuple(slice(int(a), int(b)) for a, b in zip(lo + start, lo + stop))
        parts.append(_block_stats(store[index].read().result(), start))
    stats = np.concatenate(parts) if parts else np.zeros((0, 11), dtype=np.int64)

    # Labels that span blocks: add the counts and sums, take the bbox extremes.
    ids, where = np.unique(stats[:, 0], return_inverse=True)
    n = len(ids)
    total = np.zeros((n, 4), dtype=np.int64)
    np.add.at(total, where, stats[:, 1:5])
    bbox_lo = np.full((n, 3), np.iinfo(np.int64).max)
    np.minimum.at(bbox_lo, where, stats[:, 5:8])
    bbox_hi = np.zeros((n, 3), dtype=np.int64)
    np.maximum.at(bbox_hi, where, stats[:, 8:11])
    vox = total[:, 0]
    centroid = total[:, 1:4] / np.maximum(vox, 1)[:, None]
    voxel_size = np.asarray(meta.voxel_size, dtype=float)
    corner = np.asarray(meta.translation, dtype=float)
    # Voxel i covers [corner + i * size, corner + (i + 1) * size).
    centroid_nm = corner + (centroid + 0.5) * voxel_size

    rank_cols = [f"rank_{q}" for q in queues]
    ranks = [_ranks(q, vox, ids, seed) for q in queues]
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            conn.execute(
                "CREATE TABLE instances (id INTEGER PRIMARY KEY, vox INTEGER NOT NULL,"
                " cz REAL, cy REAL, cx REAL, cz_nm REAL, cy_nm REAL, cx_nm REAL,"
                " bz0 INTEGER, bz1 INTEGER, by0 INTEGER, by1 INTEGER, bx0 INTEGER, bx1 INTEGER"
                + "".join(f", {c} INTEGER" for c in rank_cols) + ")"
            )
            conn.execute(
                "CREATE TABLE ledger (instance_id INTEGER PRIMARY KEY REFERENCES instances(id),"
                " review_state TEXT, reviewed_at TEXT, reviewer TEXT,"
                " edit_details_json TEXT, entry_method TEXT)"
            )
            conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
            for c in rank_cols:
                conn.execute(f"CREATE INDEX idx_{c} ON instances({c})")
            rows = (
                (int(ids[i]), int(vox[i]), *map(float, centroid[i]), *map(float, centroid_nm[i]),
                 int(bbox_lo[i, 0]), int(bbox_hi[i, 0]), int(bbox_lo[i, 1]),
                 int(bbox_hi[i, 1]), int(bbox_lo[i, 2]), int(bbox_hi[i, 2]),
                 *(int(r[i]) for r in ranks))
                for i in range(n)
            )
            conn.executemany(
                f"INSERT INTO instances VALUES ({','.join('?' * (14 + len(rank_cols)))})", rows
            )
            conn.executemany(
                "INSERT INTO ledger (instance_id) VALUES (?)", ((int(i),) for i in ids)
            )
            conn.executemany("INSERT INTO meta VALUES (?, ?)", [
                ("source_zarr", idi.path),
                ("built_at", time.strftime("%Y-%m-%dT%H:%M:%S")),
                ("voxel_size_nm", json.dumps(voxel_size.tolist())),
                # The lower corner of voxel 0, in nm.
                ("offset_nm", json.dumps(corner.tolist())),
                ("queue_labels", json.dumps({q: QUEUES[q] for q in queues})),
            ])
    finally:
        conn.close()
    return {
        "n_instances": n,
        "voxel_size_nm": voxel_size.tolist(),
        "offset_nm": corner.tolist(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m cellmap_flow.review_index",
        description="Build the SQLite index the dashboard's Review tab opens.",
    )
    parser.add_argument("labels", help="instance segmentation (0 = background)")
    parser.add_argument("output", help="the index to write, a new .sqlite or .db file")
    parser.add_argument(
        "--queue", action="append", choices=list(QUEUES), dest="queues",
        help=f"a queue to rank instances in (repeatable; default: {' '.join(DEFAULT_QUEUES)})",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed of the random queue")
    parser.add_argument(
        "--block", type=int, nargs=3, metavar=("Z", "Y", "X"),
        help="voxels read at a time (default: whole chunks, at least "
        f"{MIN_BLOCK_EDGE} per axis)",
    )
    args = parser.parse_args(argv)
    result = build_index(
        args.labels, args.output, queues=args.queues or DEFAULT_QUEUES,
        seed=args.seed, block_shape=args.block,
    )
    print(f"wrote {result['n_instances']} instances to {args.output}")


if __name__ == "__main__":
    main()
