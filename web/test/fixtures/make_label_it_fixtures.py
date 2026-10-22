"""Generate the parity fixture for the browser's Label it tools.

A seed's maths is the Python both modes run
(``cellmap_flow/portable/label_it.py``) save one step: the browser runs the
mutex watershed in TypeScript (``web/src/label-it/segment.ts``
``mutexObjects``, on ``web/src/infer/affinity.ts``), since ``mwatershed``
has no Pyodide wheel. Split Objects and Undo are ported too
(``relabelObjects`` in ``web/src/ai-annotate/labels.ts`` and
``rewriteForeground`` in ``web/src/ai-annotate/fill.ts`` from
``cellmap_flow/finetune/session/fill.py``). A seed made in the browser must
find the objects the server's seed finds in the same affinities, and a split
must give the same ids, so the vitest suite in
``web/test/parity/label-it.test.ts`` checks each port against what the
*Python functions themselves* produce, written here as JSON.

A separate generator from ``make_fixtures.py`` (whose helpers and
conventions it shares). Run it with a Python that has
cellmap_flow's dependencies (numpy, scipy, zarr, mwatershed, fastremap,
flask), from anywhere::

    .pixi/envs/default/bin/python web/test/fixtures/make_label_it_fixtures.py          # regenerate
    .pixi/envs/default/bin/python web/test/fixtures/make_label_it_fixtures.py --check  # CI: stale?

``--check`` regenerates into a temporary directory and compares with the
committed file (floats with a small tolerance, everything else exactly),
exiting 1 when the Python has changed and the fixture has not;
``tests/web/test_fixtures_fresh.py`` runs that in pytest.

Everything is deterministic (seeds recorded in the file) and the script is
idempotent. Arrays are ``{"dtype", "shape", "data"}`` with ``data`` flat in C
order (``make_fixtures.array_record``). The sections of ``label_it.json``:

``mutex_watershed``
    ``view_labels.segment_prediction``'s objects by the mutex watershed over
    thresholds (its bias) and smallest sizes, with the ``affinities`` the
    browser's Python hands its watershed (``label_it.affinities`` of a
    prediction: three affinity channels of probabilities, or of logits,
    small, through the sigmoid; smoothed noise, so their objects are blobs,
    as a model's are), [C, z, y, x] float32. The predictions' seed is the
    first that keeps every edge clear of ties (make_fixtures'
    affinity_case_is_safe), which mwatershed would break its own way.
``relabel_objects``
    ``fill.relabel_objects`` of label boxes (walls in every slice or one,
    strokes joining objects, ties, uint16 ids, ids running out with the
    refusal's message).
``rewrite_foreground``
    ``fill.rewrite_foreground`` with ``relabel_objects`` on an 8^3 volume in
    4^3 chunks, MinIO replaced by an in-memory zarr as
    tests/ai_annotate/test_paint_box.py replaces it: the volume before, the changed count and
    counts, the volume after, the undo entry, and ``fill.restore_box`` after
    a stroke painted in between.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import zarr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from make_fixtures import ROOT, affinity_case_is_safe, array_record, differences, fixture_text  # noqa: E402  (puts the worktree's package first on sys.path)

from cellmap_flow.dashboard.routes.finetune import view_labels  # noqa: E402
from cellmap_flow.finetune.session import fill  # noqa: E402
from cellmap_flow.portable import label_it  # noqa: E402

FIXTURES = ("label_it.json",)
SEED = 20261016
OFFSETS = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


# --- the predictions ------------------------------------------------------------


def _field(rng: np.random.Generator, shape) -> np.ndarray:
    """Smoothed noise of unit spread: blobs, as a model's objects are."""
    from scipy.ndimage import gaussian_filter

    field = gaussian_filter(rng.standard_normal(shape), sigma=1.0)
    return field / field.std()


def _affinities(rng: np.random.Generator, shape, amplitude=3.0) -> np.ndarray:
    """Three affinity channels of logits: one field, each channel with noise of its own."""
    base = _field(rng, shape)
    return np.stack([amplitude * base + 0.2 * amplitude * rng.standard_normal(shape) for _ in range(3)])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_blocks(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        # Probabilities already (output class unit): the watershed's input is
        # then the fixture's own float32 values. A few hundred voxels, as
        # make_fixtures' affinity blocks: near 1, float32's spacing makes
        # exact ties likely among thousands of edges.
        "aff_unit": _sigmoid(_affinities(rng, (4, 8, 9))).astype(np.float32),
        # Logits, through the Python's sigmoid: small and of a modest range.
        "aff_small": _affinities(rng, (3, 5, 6), amplitude=1.2).astype(np.float32),
    }


# id, block, output class, threshold (the bias), min_size
MUTEX_CASES = [
    ("mutex_default", "aff_unit", "unit", 0.5, 0),
    ("mutex_low", "aff_unit", "unit", 0.3, 0),
    ("mutex_high_min4", "aff_unit", "unit", 0.7, 4),
    ("mutex_classified_min3", "aff_unit", None, 0.5, 3),
    ("mutex_logits", "aff_small", "unbounded", 0.5, 0),
    ("mutex_logits_classified_min2", "aff_small", None, 0.4, 2),
]


def _safe(blocks: dict) -> bool:
    """No mutex case near a tie."""
    return all(
        affinity_case_is_safe(label_it.affinities(blocks[name], cls, OFFSETS), OFFSETS, threshold)
        for _, name, cls, threshold, _ in MUTEX_CASES
    )


def make_predictions() -> tuple[int, dict]:
    for seed in range(SEED, SEED + 200):
        blocks = make_blocks(seed)
        if _safe(blocks):
            return seed, blocks
    raise RuntimeError("no seed in 200 keeps the label-it fixture clear of ties")


def make_mutex(blocks: dict) -> list:
    cases = []
    for case_id, name, cls, threshold, min_size in MUTEX_CASES:
        objects = view_labels.segment_prediction(
            "mutex_watershed", blocks[name], cls, slice(0, len(OFFSETS)), json.dumps(OFFSETS), threshold, min_size
        )
        cases.append({
            "id": case_id, "output_class": cls, "offsets": OFFSETS, "threshold": threshold, "min_size": min_size,
            "affinities": array_record(label_it.affinities(blocks[name], cls, OFFSETS)),
            "objects": array_record(objects), "count": int(objects.max()),
        })
    return cases


# --- relabel_objects ------------------------------------------------------------


def _box(shape, dtype, *blocks):
    """A label box of `shape` with each (value, index) of `blocks` painted in turn."""
    box = np.zeros(shape, dtype=dtype)
    for value, index in blocks:
        box[index] = value
    return box


def _ids_run_out() -> np.ndarray:
    """A uint8 box using every id 2..255 on its own voxel, and id 5 on one
    more voxel apart from the first: the split needs a fresh id, none is left."""
    box = np.zeros((1, 24, 48), dtype=np.uint8)
    spots = [(0, 2 * r, 2 * c) for r in range(12) for c in range(24)]
    for value, spot in zip(range(2, 256), spots):
        box[spot] = value
    box[0, 23, 1] = 5
    return box


RELABEL_CASES = [
    # A background wall through every slice: the two sides apart, one keeps 5.
    ("wall_cuts", _box((3, 5, 8), np.uint8, (5, np.s_[:, 1:4, 1:7]), (1, np.s_[:, :, 4])), 1, False),
    # The wall in one slice only: the sides still touch above and below it.
    ("wall_one_slice", _box((3, 5, 8), np.uint8, (5, np.s_[:, 1:4, 1:7]), (1, np.s_[1, :, 4])), 1, False),
    # ... and per slice, each slice's pieces apart: the middle slice splits, every slice its own id.
    ("wall_one_slice_per_slice", _box((3, 5, 8), np.uint8, (5, np.s_[:, 1:4, 1:7]), (1, np.s_[1, :, 4])), 1, True),
    # A stroke of 3 joining objects 3 and 7: one component, the id most of it holds.
    ("stroke_merges", _box((2, 4, 9), np.uint8, (3, np.s_[:, 1:3, 0:3]), (7, np.s_[:, 1:3, 5:9]), (3, np.s_[:, 1:3, 3:5])), 1, False),
    # As many voxels of 4 as of 6 in one component: the larger id stays.
    ("tie_takes_larger_id", _box((1, 2, 4), np.uint8, (4, np.s_[0, 0, :]), (6, np.s_[0, 1, :])), 1, False),
    # Id 5 on three components of 3, 8 and 8 voxels: the first 8 keeps it, the others get the lowest free ids.
    ("keep_largest_first", _box((1, 7, 9), np.uint8, (5, np.s_[0, 0, 0:3]), (5, np.s_[0, 2, 0:8]), (5, np.s_[0, 4, 1:9]), (2, np.s_[0, 6, 0]), (4, np.s_[0, 6, 8])), 1, False),
    # Two parts touching only along an edge or at a corner: apart with faces, one with all neighbours.
    ("diagonal_faces", _box((2, 4, 4), np.uint8, (6, np.s_[0, 0:2, 0:2]), (6, np.s_[1, 2:4, 2:4]), (9, np.s_[0, 3, 3])), 1, False),
    ("diagonal_edges", _box((2, 4, 4), np.uint8, (6, np.s_[0, 0:2, 0:2]), (6, np.s_[1, 2:4, 2:4]), (9, np.s_[0, 3, 3])), 2, False),
    ("diagonal_all", _box((2, 4, 4), np.uint8, (6, np.s_[0, 0:2, 0:2]), (6, np.s_[1, 2:4, 2:4]), (9, np.s_[0, 3, 3])), 3, False),
    # uint16 ids, unpainted and background voxels left alone.
    ("uint16", _box((2, 3, 10), np.uint16, (1, np.s_[:, :, :]), (0, np.s_[:, 2, :]), (300, np.s_[:, 0, 0:4]), (300, np.s_[:, 0, 6:10]), (301, np.s_[0, 1, 2])), 1, False),
    ("no_foreground", _box((2, 2, 2), np.uint8, (1, np.s_[0])), 1, False),
    ("ids_run_out", _ids_run_out(), 1, False),
]


def make_relabel() -> list:
    cases = []
    for case_id, existing, connectivity, per_slice in RELABEL_CASES:
        case = {"id": case_id, "existing": array_record(existing), "connectivity": connectivity, "per_slice": per_slice}
        try:
            labels, counts = fill.relabel_objects(existing, connectivity=connectivity, per_slice=per_slice)
        except ValueError as e:
            case["error"] = str(e)
        else:
            case["labels"] = array_record(labels)
            case["counts"] = {k: int(v) for k, v in counts.items()}
        cases.append(case)
    return cases


# --- rewrite_foreground and restore_box -----------------------------------------

STATE = {"bucket": "annotations"}


class _Bucket:
    """MinIO's bucket as s3fs sees it: an in-memory zarr per key (tests/ai_annotate/test_paint_box.py)."""

    def __init__(self):
        self.stores = {}

    def exists(self, path):
        root, _, rest = path.partition(".zarr/")
        return rest in self.stores.get(root + ".zarr", {})


def _served(initial: np.ndarray, chunks=(4, 4, 4)):
    """``initial`` as MinIO would serve it: fill's S3 calls answered by an in-memory zarr."""
    from cellmap_flow.finetune.session import minio

    bucket = _Bucket()
    store = bucket.stores["annotations/vol-1.zarr"] = zarr.MemoryStore()
    minio.make_s3_filesystem = lambda state: bucket
    fill.s3fs.S3Map = lambda root, s3, check: s3.stores[root]
    labels = zarr.open_group(store).require_group("annotation").create_dataset(
        "s0", shape=initial.shape, chunks=chunks, dtype=initial.dtype, fill_value=0
    )
    labels[:] = initial
    return labels


def make_rewrite() -> list:
    cut = np.zeros((8, 8, 8), dtype=np.uint8)
    cut[1:7, 1:7, 1:7] = 5
    cut[1:7, 1:7, 4] = 1  # a wall through every slice of the object, across the chunks
    cut[0, 0, 0] = 9  # outside the box: never touched
    joined = cut.copy()
    joined[2:5, 2:5, 4] = 5  # a stroke through the wall: one object again
    uint16 = cut.astype(np.uint16)
    uint16[cut == 5] = 400
    cases = []
    # id, initial, lo, hi, connectivity, per_slice, stroke painted before the restore
    for case_id, initial, lo, hi, connectivity, per_slice, stroke in [
        ("wall_across_chunks", cut, (1, 1, 1), (7, 7, 7), 1, False, [((3, 3, 2), 1), ((3, 3, 6), 7)]),
        ("joined_nothing_changes", joined, (1, 1, 1), (7, 7, 7), 1, False, None),
        ("per_slice", cut, (2, 0, 0), (5, 8, 8), 1, True, [((4, 2, 2), 3)]),
        ("uint16", uint16, (0, 0, 0), (8, 8, 8), 1, False, [((2, 2, 5), 1)]),
    ]:
        served = _served(initial)
        undo = []
        relabel = lambda existing, c=connectivity, p=per_slice: fill.relabel_objects(existing, connectivity=c, per_slice=p)  # noqa: E731
        changed, counts = fill.rewrite_foreground(STATE, "vol-1", np.asarray(lo), np.asarray(hi), relabel, undo=undo)
        case = {
            "id": case_id, "dtype": str(initial.dtype), "shape": list(initial.shape), "chunks": [4, 4, 4], "initial": array_record(initial),
            "lo": list(lo), "hi": list(hi), "connectivity": connectivity, "per_slice": per_slice,
            "changed": int(changed), "counts": {k: int(v) for k, v in counts.items()}, "after": array_record(served[:]),
            "undo": None if not undo else {"before": array_record(undo[0][2]), "after": array_record(undo[0][3])},
        }
        if stroke is not None and undo:
            for index, value in stroke:
                served[tuple(index)] = value
            restored = fill.restore_box(STATE, "vol-1", np.asarray(lo), np.asarray(hi), undo[0][2], undo[0][3])
            case["restore"] = {"stroke": [[list(index), int(value)] for index, value in stroke], "restored": int(restored), "final": array_record(served[:])}
        cases.append(case)
    return cases


# --- writing and checking ---------------------------------------------------------


def make_label_it() -> dict:
    seed, blocks = make_predictions()
    return {
        "_format": (
            "Parity fixture for web/src/label-it/segment.ts (mutexObjects) and the Label it parts of web/src/ai-annotate "
            "(labels.ts relabelObjects, fill.ts rewriteForeground) against dashboard/routes/finetune/view_labels.py and "
            "finetune/session/fill.py; arrays are {dtype, shape, data} flat in C order; see make_label_it_fixtures.py's docstring"
        ),
        "seed": seed,
        "mutex_watershed": make_mutex(blocks),
        "relabel_objects": make_relabel(),
        "rewrite_foreground": make_rewrite(),
    }


MAKERS = {"label_it.json": make_label_it}


def write_fixtures(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, make in MAKERS.items():
        text = fixture_text(make())
        (out_dir / name).write_text(text, encoding="utf-8")
        size = len(text.encode("utf-8"))
        assert size < 600_000, f"{name} is {size} bytes; keep fixtures small enough to commit"
        print(f"wrote {out_dir / name} ({size} bytes)")


def check_fixtures(committed_dir: Path) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        fresh_dir = Path(tmp)
        write_fixtures(fresh_dir)
        problems: list[str] = []
        for name in FIXTURES:
            committed_path = committed_dir / name
            if not committed_path.exists():
                problems.append(f"{name}: missing")
                continue
            committed = json.loads(committed_path.read_text(encoding="utf-8"))
            fresh = json.loads((fresh_dir / name).read_text(encoding="utf-8"))
            differences(name, committed, fresh, problems)
    if problems:
        print(f"{len(problems)} difference(s) between the committed fixture and the Python code:")
        for problem in problems[:40]:
            print(f"  {problem}")
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
        print(f"Regenerate with: python {Path(__file__).relative_to(ROOT)}")
        return 1
    print("Label it fixture is up to date")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=HERE, help="directory to write into (default: beside this script)")
    parser.add_argument("--check", action="store_true", help="regenerate to a temporary directory and compare; exit 1 if stale")
    args = parser.parse_args(argv)
    if args.check:
        return check_fixtures(args.out)
    write_fixtures(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
