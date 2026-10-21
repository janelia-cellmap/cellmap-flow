"""Generate the parity fixture for the browser's Label it tools.

The browser mode re-implements the Finetune tab's Seed from Prediction, All
Background, Split Objects and Undo in TypeScript (``web/src/label-it/``:
``segment.ts`` and ``prediction.ts`` from
``cellmap_flow/dashboard/routes/finetune/view_labels.py``; ``relabelObjects``
in ``web/src/ai-annotate/labels.ts`` and ``rewriteForeground`` in
``web/src/ai-annotate/fill.ts`` from ``cellmap_flow/finetune/session/fill.py``).
A seed made in the browser must find the objects the server's seed finds in
the same prediction, read the prediction under the same voxels, and a split
must give the same ids, so the vitest suite in
``web/test/parity/label-it.test.ts`` checks each port against what the
*Python functions themselves* produce, written here as JSON.

A separate generator from ``make_fixtures.py`` and ``make_ai_fixtures.py``
(whose helpers and conventions it shares). Run it with a Python that has
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

``blocks``
    the predictions the cases segment, [C, z, y, x] float32: ``aff``, three
    affinity channels of logits; ``aff_unit``, three of probabilities;
    ``aff_small``, three of logits, small; ``binary``, one channel of
    logits; ``unit``, a probability; ``signed``, two channels in [-1, 1] (a
    tanh). Smoothed noise, so their objects are blobs, as a model's are.
``probability``
    ``view_labels.probability`` of blocks for an output class (or none, read
    off the values) and channels: float32, compared within the float
    tolerance (scipy's float32 expit and the port's agree to an ulp).
``segment``
    ``view_labels.segment_prediction``'s objects for components and the
    mutex watershed over thresholds, smallest sizes, connectivities and per
    slice. The blocks' seed is the first that keeps every voxel's
    probability clear of its threshold, every mutex-watershed edge clear of
    ties (make_fixtures' affinity_case_is_safe) and, through expit, its
    edges 1e-6 apart, so a float32 ulp cannot move a voxel.
``relabel_objects``
    ``fill.relabel_objects`` of label boxes (walls in every slice or one,
    strokes joining objects, ties, uint16 ids, ids running out with the
    refusal's message).
``rewrite_foreground``
    ``fill.rewrite_foreground`` with ``relabel_objects`` on an 8^3 volume in
    4^3 chunks, MinIO replaced by an in-memory zarr as make_ai_fixtures'
    paint_box cases replace it: the volume before, the changed count and
    counts, the volume after, the undo entry, and ``fill.restore_box`` after
    a stroke painted in between.
``read_prediction``
    ``view_labels.read_prediction`` of a served array (the model's output on
    its own grid, OME translation and voxel size, in chunks) for an
    annotation volume's box, its server's HTTP replaced by the array's
    files: the prediction and the box shrunk to what the server covers, or
    the refusal's message when it covers nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import zarr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from make_fixtures import ROOT, affinity_case_is_safe, affinity_edge_weights, array_record, differences, fixture_text  # noqa: E402  (puts the worktree's package first on sys.path)

from cellmap_flow.dashboard.routes.finetune import view_labels  # noqa: E402
from cellmap_flow.finetune.session import fill  # noqa: E402

FIXTURES = ("label_it.json",)
SEED = 20261016
OFFSETS = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
# How far a voxel's mean probability must sit from its threshold, so the
# port's float32 expit (an ulp from scipy's on a few values) decides it the same.
PROBABILITY_MARGIN = 1e-5


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
        "aff": _affinities(rng, (6, 10, 12)).astype(np.float32),
        # Probabilities already (output class unit), for the mutex watershed's
        # larger cases: its input is then the fixture's own float32 values, with
        # no expit between, and only its own ties matter (affinity_case_is_safe).
        # A few hundred voxels, as make_fixtures' affinity blocks: near 1,
        # float32's spacing makes exact ties likely among thousands of edges.
        "aff_unit": _sigmoid(_affinities(rng, (4, 8, 9))).astype(np.float32),
        # Logits for the mutex watershed through expit: small and of a modest
        # range, so its edges can be kept a whole 1e-6 apart, which an ulp of
        # expit cannot reorder.
        "aff_small": _affinities(rng, (3, 5, 6), amplitude=1.2).astype(np.float32),
        "binary": (3.0 * _field(rng, (6, 9, 10)))[None].astype(np.float32),
        "unit": _sigmoid(2.5 * _field(rng, (5, 6, 7)))[None].astype(np.float32),
        "signed": np.tanh(np.stack([_field(rng, (4, 6, 6)) for _ in range(2)])).astype(np.float32),
    }


def _channels(channels):
    """The fixture's channel list as the Python's slice (contiguous here), or None."""
    return None if channels is None else slice(channels[0], channels[-1] + 1)


# id, block, output class, channels, threshold, min_size, connectivity, per_slice
COMPONENT_CASES = [
    ("aff_default", "aff", "unbounded", [0, 1, 2], 0.5, 0, 1, False),
    ("aff_low_edges", "aff", "unbounded", [0, 1, 2], 0.3, 0, 2, False),
    ("aff_classified_all_min3", "aff", None, [0, 1, 2], 0.7, 3, 3, False),
    ("aff_one_channel_per_slice", "aff", "unbounded", [1], 0.5, 0, 1, True),
    ("binary_default", "binary", "unbounded", None, 0.5, 0, 1, False),
    ("binary_per_slice_min5", "binary", "unbounded", None, 0.5, 5, 1, True),
    ("binary_per_slice_all", "binary", "unbounded", None, 0.6, 0, 3, True),
    ("unit_default", "unit", "unit", None, 0.5, 0, 1, False),
    ("unit_classified_edges", "unit", None, None, 0.4, 0, 2, True),
    ("signed_default", "signed", "signed_unit", None, 0.5, 0, 1, False),
    ("signed_classified_min2", "signed", None, None, 0.6, 2, 1, False),
]
# id, block, output class, threshold (the bias), min_size
MUTEX_CASES = [
    ("mutex_default", "aff_unit", "unit", 0.5, 0),
    ("mutex_low", "aff_unit", "unit", 0.3, 0),
    ("mutex_high_min4", "aff_unit", "unit", 0.7, 4),
    ("mutex_classified_min3", "aff_unit", None, 0.5, 3),
    ("mutex_logits", "aff_small", "unbounded", 0.5, 0),
    ("mutex_logits_classified_min2", "aff_small", None, 0.4, 2),
]
# How far apart the edges of a mutex case through expit must be (and from
# 0): an ulp of expit, ~6e-8 at most, then cannot reorder them.
EXPIT_EDGE_MARGIN = 1e-6
# id, block, output class, channels
PROBABILITY_CASES = [
    ("aff_unbounded", "aff_small", "unbounded", [0, 1, 2]),
    ("aff_one_channel", "aff_small", "unbounded", [2]),
    ("aff_classified", "aff_small", None, None),
    ("binary_unbounded", "binary", "unbounded", None),
    ("unit_unit", "unit", "unit", None),
    ("unit_classified", "unit", None, None),
    ("signed_signed", "signed", "signed_unit", None),
    ("signed_classified", "signed", None, [1]),
]


def _safe(blocks: dict) -> bool:
    """No component case's voxel near its threshold, no mutex case near a tie."""
    for _, name, cls, channels, threshold, *_ in COMPONENT_CASES:
        mean = view_labels.probability(blocks[name], cls, _channels(channels)).mean(axis=0)
        if np.min(np.abs(mean.astype(np.float64) - np.float32(threshold))) <= PROBABILITY_MARGIN:
            return False
    for _, name, cls, threshold, _ in MUTEX_CASES:
        affinities = view_labels.probability(blocks[name], cls, slice(0, len(OFFSETS)))
        if not affinity_case_is_safe(affinities, OFFSETS, threshold):
            return False
        if name == "aff_small":  # through expit
            weights = np.sort(affinity_edge_weights(affinities, OFFSETS, threshold))
            if weights[0] <= EXPIT_EDGE_MARGIN or np.min(np.diff(weights)) <= EXPIT_EDGE_MARGIN:
                return False
    return True


def make_predictions() -> tuple[int, dict]:
    for seed in range(SEED, SEED + 200):
        blocks = make_blocks(seed)
        if _safe(blocks):
            return seed, blocks
    raise RuntimeError("no seed in 200 keeps the label-it fixture clear of thresholds and ties")


def make_probability(blocks: dict) -> list:
    cases = []
    for case_id, name, cls, channels in PROBABILITY_CASES:
        out = view_labels.probability(blocks[name], cls, _channels(channels))
        cases.append({"id": case_id, "block": name, "output_class": cls, "channels": channels, "probability": array_record(out)})
    return cases


def make_segment(blocks: dict) -> list:
    cases = []
    for case_id, name, cls, channels, threshold, min_size, connectivity, per_slice in COMPONENT_CASES:
        objects = view_labels.segment_prediction(
            "components", blocks[name], cls, _channels(channels), None, threshold, min_size, connectivity, per_slice
        )
        cases.append({
            "id": case_id, "method": "components", "block": name, "output_class": cls, "channels": channels, "offsets": None,
            "threshold": threshold, "min_size": min_size, "connectivity": connectivity, "per_slice": per_slice,
            "objects": array_record(objects), "count": int(objects.max()),
        })
    for case_id, name, cls, threshold, min_size in MUTEX_CASES:
        objects = view_labels.segment_prediction(
            "mutex_watershed", blocks[name], cls, _channels([0, 1, 2]), json.dumps(OFFSETS), threshold, min_size
        )
        cases.append({
            "id": case_id, "method": "mutex_watershed", "block": name, "output_class": cls, "channels": [0, 1, 2], "offsets": OFFSETS,
            "threshold": threshold, "min_size": min_size, "connectivity": 1, "per_slice": False,
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


def make_rewrite() -> list:
    from make_ai_fixtures import STATE, _served

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


# --- read_prediction -------------------------------------------------------------


class _Reply:
    """A model server's answer, as requests gives it: .json() and .content."""

    def __init__(self, content: bytes):
        self.content = content

    def json(self):
        return json.loads(self.content)


def _serve(values: np.ndarray, voxel, translation, chunks):
    """The files of a served array, [C, z, y, x], at OME `voxel` and
    `translation` (voxel 0's centre), keyed by URL as read_prediction asks
    for them, and its read in place of the HTTP one (view_labels._get)."""
    store = {}
    zarr.create(shape=values.shape, chunks=(values.shape[0], *chunks), dtype=values.dtype, compressor=None, store=store, fill_value=0)[:] = values
    files = {f"base/s0/{key}": value for key, value in store.items()}
    files["base/.zattrs"] = json.dumps({"multiscales": [{
        "axes": [{"name": "c", "type": "channel"}] + [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
        "datasets": [{"path": "s0", "coordinateTransformations": [
            {"type": "scale", "scale": [1.0, *map(float, voxel)]}, {"type": "translation", "translation": [0.0, *map(float, translation)]},
        ]}],
    }]}).encode()
    return lambda url, timeout=None: _Reply(files[url])


def _served_values(shape) -> np.ndarray:
    """A served block whose value names its voxel: c·10^6 + z·10^4 + y·100 + x (exact in float32 here)."""
    c, z, y, x = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    return (c * 1_000_000 + z * 10_000 + y * 100 + x).astype(np.float32)


# id, served (shape [C, z, y, x], voxel, translation, chunks), volume (voxel, dataset_offset_nm), lo, hi
READ_CASES = [
    # The grids are one: the box as it is, across the served chunks.
    ("same_grid", ((2, 12, 12, 12), (8, 8, 8), (4, 4, 4), (4, 4, 4)), ((8, 8, 8), (4, 4, 4)), (2, 3, 1), (9, 10, 11)),
    # The model served at twice the voxel: two annotation voxels to a served one.
    ("coarser_served", ((1, 8, 8, 8), (16, 16, 16), (8, 8, 8), (4, 4, 4)), ((8, 8, 8), (4, 4, 4)), (1, 2, 3), (9, 12, 15)),
    # At half: every other served voxel, the one each annotation voxel's centre lies in.
    ("finer_served", ((1, 20, 20, 20), (4, 4, 4), (2, 2, 2), (5, 5, 5)), ((8, 8, 8), (4, 4, 4)), (0, 1, 2), (6, 7, 9)),
    # Served from 20 nm on: annotation voxels before it are left out, the box shrinks.
    ("offset_served_clips_low", ((1, 10, 10, 10), (8, 8, 8), (24, 24, 24), (4, 4, 4)), ((8, 8, 8), (4, 4, 4)), (0, 1, 2), (6, 7, 8)),
    # Half a voxel off: each centre lies in one served voxel all the same.
    ("half_voxel_shift", ((1, 10, 10, 10), (8, 8, 8), (8, 8, 8), (5, 5, 5)), ((8, 8, 8), (4, 4, 4)), (1, 1, 1), (8, 8, 8)),
    # The box runs past the served array's end: shrunk to it.
    ("clips_high", ((1, 6, 6, 6), (8, 8, 8), (4, 4, 4), (4, 4, 4)), ((8, 8, 8), (4, 4, 4)), (3, 2, 4), (9, 5, 12)),
    # Anisotropic grids, z coarser.
    ("anisotropic", ((3, 6, 12, 12), (16, 8, 8), (8, 4, 4), (3, 4, 4)), ((16, 8, 8), (8, 4, 4)), (1, 2, 3), (5, 11, 9)),
    # Anisotropic, served coarser in y and x only.
    ("anisotropic_coarser_xy", ((1, 6, 6, 6), (16, 16, 16), (8, 8, 8), (3, 3, 3)), ((16, 8, 8), (8, 4, 4)), (0, 1, 1), (5, 9, 10)),
    # Nothing of the box served: the refusal.
    ("outside", ((1, 4, 4, 4), (8, 8, 8), (4, 4, 4), (4, 4, 4)), ((8, 8, 8), (4, 4, 4)), (5, 0, 0), (8, 4, 4)),
]


def make_read_prediction() -> list:
    view_labels._served_base = lambda host, model_name, postprocess=(), input_norm=None: "base"
    view_labels.prediction_voxel_override = lambda *args: None
    view_labels.get_session = lambda: SimpleNamespace(dataset_path="data")
    cases = []
    for case_id, (shape, voxel, translation, chunks), (volume_voxel, offset), lo, hi in READ_CASES:
        values = _served_values(shape)
        view_labels._get = _serve(values, voxel, translation, chunks)
        volume = {"output_voxel_size": [float(v) for v in volume_voxel], "dataset_offset_nm": [float(v) for v in offset]}
        # The served values are _served_values' formula, which the test computes again.
        case = {
            "id": case_id, "served": {"shape": list(shape), "voxel": list(voxel), "translation": list(translation), "chunks": list(chunks)},
            "volume": volume, "lo": list(lo), "hi": list(hi),
        }
        try:
            prediction, new_lo, new_hi = view_labels.read_prediction("host", "model", volume, np.asarray(lo), np.asarray(hi), {})
        except view_labels._Refused as e:
            case["error"] = {"message": str(e), "status": e.status}
        else:
            case.update({"prediction": array_record(prediction), "new_lo": [int(v) for v in new_lo], "new_hi": [int(v) for v in new_hi]})
        cases.append(case)
    return cases


# --- writing and checking ---------------------------------------------------------


def make_label_it() -> dict:
    seed, blocks = make_predictions()
    return {
        "_format": (
            "Parity fixture for web/src/label-it (segment.ts, prediction.ts) and the Label it parts of web/src/ai-annotate "
            "(labels.ts relabelObjects, fill.ts rewriteForeground) against dashboard/routes/finetune/view_labels.py and "
            "finetune/session/fill.py; arrays are {dtype, shape, data} flat in C order; see make_label_it_fixtures.py's docstring"
        ),
        "seed": seed,
        "blocks": {name: array_record(block) for name, block in blocks.items()},
        "probability": make_probability(blocks),
        "segment": make_segment(blocks),
        "relabel_objects": make_relabel(),
        "rewrite_foreground": make_rewrite(),
        "read_prediction": make_read_prediction(),
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
