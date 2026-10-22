"""Generate the parity fixture for the browser's Label it tools.

A seed's maths is the Python both modes run
(``cellmap_flow/portable/label_it.py``) save one step: the browser runs the
mutex watershed in TypeScript (``web/src/label-it/segment.ts``
``mutexObjects``, on ``web/src/infer/affinity.ts``), since ``mwatershed``
has no Pyodide wheel. (The ids, writes and undo of Seed, All Background,
Split Objects and Undo are ``cellmap_flow/portable/fill.py``'s in both modes,
tested by pytest.) A seed made in the browser must find the objects the
server's seed finds in the same affinities, so the vitest suite in
``web/test/parity/label-it.test.ts`` checks the watershed against what the
*Python functions themselves* produce, written here as JSON.

A separate generator from ``make_fixtures.py`` (whose helpers and
conventions it shares). Run it with a Python that has
cellmap_flow's dependencies (numpy, scipy, mwatershed, fastremap, flask),
from anywhere::

    .pixi/envs/default/bin/python web/test/fixtures/make_label_it_fixtures.py          # regenerate
    .pixi/envs/default/bin/python web/test/fixtures/make_label_it_fixtures.py --check  # CI: stale?

``--check`` regenerates into a temporary directory and compares with the
committed file (floats with a small tolerance, everything else exactly),
exiting 1 when the Python has changed and the fixture has not;
``tests/web/test_fixtures_fresh.py`` runs that in pytest.

Everything is deterministic (seeds recorded in the file) and the script is
idempotent. Arrays are ``{"dtype", "shape", "data"}`` with ``data`` flat in C
order (``make_fixtures.array_record``). The section of ``label_it.json``:

``mutex_watershed``
    ``view_labels.segment_prediction``'s objects by the mutex watershed over
    thresholds (its bias) and smallest sizes, with the ``affinities`` the
    browser's Python hands its watershed (``label_it.affinities`` of a
    prediction: three affinity channels of probabilities, or of logits,
    small, through the sigmoid; smoothed noise, so their objects are blobs,
    as a model's are), [C, z, y, x] float32. The predictions' seed is the
    first that keeps every edge clear of ties (make_fixtures'
    affinity_case_is_safe), which mwatershed would break its own way.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from make_fixtures import ROOT, affinity_case_is_safe, array_record, differences, fixture_text  # noqa: E402  (puts the worktree's package first on sys.path)

from cellmap_flow.dashboard.routes.finetune import view_labels  # noqa: E402
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


# --- writing and checking ---------------------------------------------------------


def make_label_it() -> dict:
    seed, blocks = make_predictions()
    return {
        "_format": (
            "Parity fixture for web/src/label-it/segment.ts (mutexObjects) against dashboard/routes/finetune/view_labels.py "
            "(segment_prediction); arrays are {dtype, shape, data} flat in C order; see make_label_it_fixtures.py's docstring"
        ),
        "seed": seed,
        "mutex_watershed": make_mutex(blocks),
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
