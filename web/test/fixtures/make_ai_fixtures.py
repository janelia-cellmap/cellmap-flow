"""Generate the parity fixture for the browser's AI-assisted annotation core.

The browser mode re-implements the server's AI annotation geometry and
labelling in TypeScript (``web/src/ai-annotate/``: ``geometry.ts`` from
``cellmap_flow/ai_annotate/geometry.py``, ``plane.ts`` from ``pipeline.py``,
``mask.ts`` from ``mask_extraction.py``, ``labels.ts`` and ``fill.ts`` from
``finetune/session/fill.py``, ``prompts.ts`` and ``organelles.ts`` from the
modules of the same names). A plane planned in the browser must be the plane
the server would plan for the same volume and point, the mask must land on
the same voxels and get the same ids, and the model must be asked with the
same words, so the vitest suite in ``web/test/parity/ai-annotate.test.ts``
checks each port against what the *Python functions themselves* produce,
written here as JSON, rather than against a reading of the Python code.

A separate generator from ``make_fixtures.py`` (whose helpers and conventions
it shares): the cases are plans, planes and boxes, not blocks through a
chain. Run it with a Python that has cellmap_flow's dependencies (numpy,
scipy, PIL, zarr), from anywhere::

    .pixi/envs/default/bin/python web/test/fixtures/make_ai_fixtures.py          # regenerate
    .pixi/envs/default/bin/python web/test/fixtures/make_ai_fixtures.py --check  # CI: stale?

``--check`` regenerates into a temporary directory and compares with the
committed file (floats with a small tolerance, everything else exactly),
exiting 1 when the Python has changed and the fixture has not;
``tests/web/test_fixtures_fresh.py`` runs that in pytest.

Everything is deterministic (a fixed seed, recorded in the file) and the
script is idempotent. Arrays are ``{"dtype", "shape", "data"}`` with ``data``
flat in C order (``make_fixtures.array_record``). The sections of
``ai_annotate.json``:

``screen``
    ``to_screen``/``from_screen``/``screen_shape`` of a small plane per depth axis.
``plan_plane``
    ``plan_plane`` cases (volumes, shapes, points, depth axes, crop sizes,
    data boxes): the plan ``to_dict`` wrote with its ``click_px``,
    ``plane_axes``, ``write_shape`` and ``plane_name``, or the refusal's
    category, message and status.
``raw_box``
    ``raw_box`` of some of those plans on raw grids of several voxel sizes and corners.
``extract_mask``
    one small RGB image and the mask ``extract_mask`` finds for each catalog colour.
``normalize_to_uint8``
    ``normalize_to_uint8`` of a float32, a uint16 and a uint8 plane.
``resize``
    ``pipeline._resize`` (PIL BOX when shrinking, BILINEAR otherwise) of
    small float32 planes, float32 results.
``mask_to_write_shape``
    random masks of a plan's image cut into objects on its write box, and
    two objects parted by a one-pixel line on voxels coarser than the pixels.
``box_voxels``, ``fresh_ids``, ``seed_labels``, ``labels_for_box``
    the box arithmetic and the ids, incl. refusals' messages.
``paint_box``
    ``fill.paint_box`` and ``fill.restore_box`` on an 8^3 volume in 4^3
    chunks, MinIO replaced by an in-memory zarr as tests/ai_annotate/
    test_paint_box.py replaces it: the volume before, the labels written,
    the counts, the volume after, the undo entry, and a restore after a
    stroke painted in between.
``prompts``, ``organelles``
    the prompts word for word, and the catalog with its lookups.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import zarr
from scipy.ndimage import label

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from make_fixtures import ROOT, array_record, differences, fixture_text  # noqa: E402  (puts the worktree's package first on sys.path)

from cellmap_flow.ai_annotate import geometry, mask_extraction, organelles, pipeline, prompts  # noqa: E402
from cellmap_flow.ai_annotate.errors import AIAnnotateError  # noqa: E402
from cellmap_flow.finetune.session import fill, minio  # noqa: E402

FIXTURES = ("ai_annotate.json",)
SEED = 20261015


def _error(e: AIAnnotateError) -> dict:
    return {"category": e.category, "message": e.user_message, "http_status": e.http_status}


def _volume(output=(8, 8, 8), input=(4, 4, 4), offset=None) -> dict:
    """An annotation volume's record: voxel 0's corner at the origin unless ``offset`` (its centre) is given."""
    output = np.asarray(output, dtype=float)
    return {
        "output_voxel_size": [float(v) for v in output],
        "input_voxel_size": [float(v) for v in input],
        "dataset_offset_nm": [float(v) for v in (output / 2 if offset is None else np.asarray(offset, dtype=float))],
    }


SHAPE = (100, 200, 300)
V8 = _volume()
ANISO = _volume(output=(16, 8, 8), input=(8, 4, 4))
CENTRE = (50 * 8 + 1, 100 * 8 + 3, 150 * 8 + 6)  # in voxel (50, 100, 150), off its centre


# --- screen layout --------------------------------------------------------------


def make_screen() -> list:
    plane = np.arange(6, dtype=np.int32).reshape(2, 3)  # rows 2, cols 3, in z, y, x order
    image = np.arange(12, dtype=np.int32).reshape(3, 4)  # as a panel would draw it
    cases = []
    for depth_axis in (0, 1, 2):
        cases.append({
            "depth_axis": depth_axis,
            "plane": array_record(plane),
            "to_screen": array_record(geometry.to_screen(plane, depth_axis)),
            "screen_shape": list(geometry.screen_shape(plane.shape, depth_axis)),
            "image": array_record(image),
            "from_screen": array_record(geometry.from_screen(image, depth_axis)),
        })
    return cases


# --- plan_plane and raw_box -----------------------------------------------------

# id, volume, volume_shape, point, depth_axis, crop_size_px, data_box_nm
PLAN_CASES = [
    ("centred_z", V8, SHAPE, CENTRE, 0, 64, None),
    ("centred_y", V8, SHAPE, CENTRE, 1, 64, None),
    ("centred_x", V8, SHAPE, CENTRE, 2, 64, None),
    # A point on a voxel boundary takes the voxel above.
    ("boundary", V8, SHAPE, (80, 800, 1200), 0, 8, None),
    # dataset_offset_nm 100: voxel 0 covers [96, 104) on every axis.
    ("offset_100", _volume(offset=(100, 100, 100)), SHAPE, (103, 104 + 8 * 20, 104 + 8 * 30), 0, 4, None),
    # 16 nm z, 8 nm y and x, the input at half each: a square field, square pixels.
    ("aniso_xz", ANISO, SHAPE, (50 * 16 + 8, 100 * 8 + 4, 150 * 8 + 4), 1, 64, None),
    ("aniso_xy", ANISO, SHAPE, (50 * 16 + 8, 100 * 8 + 4, 150 * 8 + 4), 0, 64, None),
    ("aniso_yz", ANISO, SHAPE, (50 * 16 + 8, 100 * 8 + 4, 150 * 8 + 4), 2, 64, None),
    # 512 annotation voxels of 8 nm read at 4 nm: 1024 px; read at 8 nm: 512.
    ("io_differ_4nm", V8, (100, 2000, 2000), (400, 8000, 8000), 0, 512, None),
    ("io_same_8nm", _volume(input=(8, 8, 8)), (100, 2000, 2000), (400, 8000, 8000), 0, 512, None),
    # Longer than 1024 px at the input voxel size: sent at larger pixels.
    ("downsampled", V8, (100, 2000, 2000), (400, 8000, 8000), 0, 1024, None),
    ("downsampled_narrow", V8, (100, 2000, 2000), (400, 8000, 40), 0, 1024, None),
    # 1001 voxels across at 16 nm pixels is 500.5 px: numpy rounds halves to even.
    ("half_pixel_rounding", V8, (100, 3000, 1001), (400, 12000, 4000), 0, 2048, None),
    # Voxel (50, 3, 297): 3 voxels from y's start, 2 from x's end.
    ("clipped_edge", V8, SHAPE, (50 * 8 + 4, 3 * 8 + 4, 297 * 8 + 4), 0, 64, None),
    # The data ends at x = 1210 nm while the volume runs to 2400: the field stops there.
    ("data_box_clip", V8, SHAPE, CENTRE, 0, 64, ((0, 0, 0), (800, 1600, 1210))),
    ("data_box_refused", V8, SHAPE, CENTRE, 0, 64, ((0, 0, 0), (800, 1600, 1200))),
    # Outside the volume.
    ("outside_before_z", V8, SHAPE, (-1, 800, 1200), 0, 64, None),
    ("outside_past_x", V8, SHAPE, (400, 800, 300 * 8), 0, 64, None),
    ("outside_fraction", V8, SHAPE, (400, -0.5, 1200), 0, 64, None),
    ("outside_at_end_z", V8, SHAPE, (100 * 8 + 4, 800, 1200), 0, 64, None),
    # Small fields for the mask cases: 16 voxels, 32 px.
    ("small_z", V8, SHAPE, CENTRE, 0, 16, None),
    ("small_y", V8, SHAPE, CENTRE, 1, 16, None),
    ("small_x", V8, SHAPE, CENTRE, 2, 16, None),
    ("small_aniso_xz", ANISO, SHAPE, (50 * 16 + 8, 100 * 8 + 4, 150 * 8 + 4), 1, 16, None),
    ("small_aniso_yz", ANISO, SHAPE, (50 * 16 + 8, 100 * 8 + 4, 150 * 8 + 4), 2, 16, None),
    ("small_clipped", V8, SHAPE, (50 * 8 + 4, 3 * 8 + 4, 297 * 8 + 4), 0, 16, None),
    ("small_same", _volume(input=(8, 8, 8)), SHAPE, CENTRE, 0, 16, None),
    ("small_offset_100", _volume(offset=(100, 100, 100)), SHAPE, (103, 104 + 8 * 20, 104 + 8 * 30), 1, 12, None),
]

MASK_PLAN_IDS = ("small_z", "small_y", "small_x", "small_aniso_xz", "small_aniso_yz", "small_clipped", "small_same", "small_offset_100")

# plan id, raw voxel size, raw corner
RAW_BOX_CASES = [
    ("centred_z", (4, 4, 4), (0, 0, 0)),
    ("centred_z", (4, 4, 4), (-1, -1, -1)),
    ("centred_z", (4, 4, 4), (1, 1, 1)),
    ("centred_z", (4, 4, 4), (3, 3, 3)),
    ("centred_z", (8, 8, 8), (0, 0, 0)),
    ("centred_y", (4, 4, 4), (0, 0, 0)),
    ("aniso_xz", (8, 4, 4), (0, 0, 0)),
    ("aniso_xz", (8, 4, 4), (-4, -2, -2)),
    ("downsampled", (4, 4, 4), (0, 0, 0)),
    ("small_offset_100", (4, 4, 4), (96, 96, 96)),
]


def make_plans() -> tuple[list, dict]:
    cases, plans = [], {}
    for case_id, volume, shape, point, depth_axis, crop, data_box in PLAN_CASES:
        case = {
            "id": case_id, "volume": volume, "volume_shape": list(shape), "point_nm": [float(p) for p in point],
            "depth_axis": depth_axis, "crop_size_px": crop,
            "data_box_nm": [[float(v) for v in side] for side in data_box] if data_box else None,
        }
        try:
            plan = geometry.plan_plane(volume, shape, point, depth_axis, crop, data_box_nm=data_box)
        except AIAnnotateError as e:
            case["error"] = _error(e)
        else:
            plans[case_id] = plan
            case.update({
                "plan": plan.to_dict(), "click_px": list(plan.click_px), "plane_axes": list(plan.plane_axes),
                "write_shape": list(plan.write_shape), "plane_name": plan.plane_name,
            })
        cases.append(case)
    return cases, plans


def make_raw_boxes(plans: dict) -> list:
    cases = []
    for plan_id, voxel, corner in RAW_BOX_CASES:
        box = geometry.raw_box(plans[plan_id], voxel, corner)
        cases.append({"plan": plan_id, "voxel_size": list(voxel), "corner": list(corner), "begin": list(box.begin), "count": list(box.shape)})
    return cases


# --- masks and planes -----------------------------------------------------------


def make_extract_mask() -> dict:
    pixels = [
        [(255, 0, 0), (230, 20, 10), (200, 0, 0), (255, 128, 0), (255, 100, 0), (128, 0, 255)],
        [(120, 10, 250), (255, 255, 255), (0, 0, 0), (255, 255, 0), (0, 255, 0), (0, 255, 128)],
        [(0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 0, 128), (60, 60, 60), (254, 1, 1)],
    ]
    image = np.asarray(pixels, dtype=np.uint8)
    targets = {}
    for profile in organelles.ORGANELLES.values():
        targets.setdefault(profile.color_name, tuple(int(c) for c in profile.rgb))
    cases = []
    for name, rgb in sorted(targets.items()):
        for threshold in (200.0, 100.0):
            mask = mask_extraction.extract_mask(image, rgb, threshold=threshold)
            cases.append({"color_name": name, "target_rgb": list(rgb), "threshold": threshold, "mask": array_record(mask.astype(np.uint8))})
    return {"image": array_record(image), "cases": cases}


def make_normalize(rng: np.random.Generator) -> list:
    floats = (rng.standard_normal((6, 7)) * 1000).astype(np.float32)
    floats[0, 0], floats[5, 6] = 50000.0, -50000.0  # outliers the percentiles clip
    uint16 = rng.integers(1000, 40000, size=(5, 8), dtype=np.uint16)
    uint8 = rng.integers(0, 256, size=(4, 5), dtype=np.uint8)
    cases = []
    for name, plane in (("float32", floats), ("uint16", uint16), ("uint8", uint8)):
        cases.append({"id": name, "plane": array_record(plane), "uint8": array_record(mask_extraction.normalize_to_uint8(plane))})
    return cases


RESIZE_CASES = [
    ("shrink_whole", (6, 8), (3, 4)),
    ("shrink_fraction", (7, 5), (3, 2)),
    ("grow_whole", (3, 4), (6, 8)),
    ("grow_fraction", (3, 3), (7, 5)),
    ("mixed", (6, 3), (3, 6)),
    ("same", (5, 5), (5, 5)),
    ("rows_only", (8, 8), (5, 8)),
    ("cols_only", (4, 9), (4, 3)),
    ("one_row", (1, 6), (1, 4)),
]


def make_resize(rng: np.random.Generator) -> list:
    cases = []
    for case_id, in_shape, out_shape in RESIZE_CASES:
        plane = (rng.random(in_shape) * 255).astype(np.float32)
        out = pipeline._resize(plane, out_shape)
        cases.append({"id": case_id, "plane": array_record(plane), "shape": list(out_shape), "resized": array_record(np.asarray(out, dtype=np.float32))})
    return cases


def make_mask_to_write_shape(rng: np.random.Generator, plans: dict) -> list:
    masks = [(plan_id, plan_id, rng.random(tuple(plans[plan_id].image_shape)) < 0.5) for plan_id in MASK_PLAN_IDS]
    # 32 px on 16 voxels: each voxel takes an odd pixel column, none the
    # line in column 16 between the two objects.
    line = np.ones(tuple(plans["small_z"].image_shape), dtype=bool)
    line[:, 16] = False
    masks.append(("small_z_line", "small_z", line))
    cases = []
    for case_id, plan_id, mask in masks:
        out = pipeline.mask_to_write_shape(mask, plans[plan_id])
        cases.append({"id": case_id, "plan": plan_id, "mask": array_record(mask.astype(np.uint8)), "write": array_record(out)})
    return cases


# --- boxes and ids --------------------------------------------------------------


def make_box_voxels() -> list:
    cases = []
    for case_id, volume, offset, size, shape in [
        ("on_grid", V8, (32, 32, 32), (8, 16, 24), SHAPE),
        ("centres_decide", V8, (30, 36, 44), (12, 8, 8), SHAPE),
        ("clipped", V8, (-20, 1580, 2380), (40, 40, 40), SHAPE),
        ("outside", V8, (-100, 0, 0), (40, 40, 40), SHAPE),
        ("offset_100", _volume(offset=(100, 100, 100)), (96, 100, 104), (8, 8, 8), SHAPE),
        ("aniso", ANISO, (0, 0, 0), (33, 17, 15), SHAPE),
    ]:
        box = fill.box_voxels(volume, offset, size, shape)
        cases.append({
            "id": case_id, "volume": volume, "offset_nm": list(offset), "shape_nm": list(size), "volume_shape": list(shape),
            "corner_nm": [float(v) for v in fill.volume_corner_nm(volume["dataset_offset_nm"], volume["output_voxel_size"])],
            "box": None if box is None else {"lo": [int(v) for v in box[0]], "hi": [int(v) for v in box[1]]},
        })
    return cases


def make_fresh_ids() -> list:
    full = np.arange(2, 256, dtype=np.uint8)  # every uint8 id in use
    cases = []
    for case_id, existing, count, count_up in [
        ("lowest_free", np.array([0, 0, 1, 2, 3, 5], dtype=np.uint8), 3, False),
        ("count_up", np.array([0, 0, 1, 2, 3, 5], dtype=np.uint8), 3, True),
        ("count_up_from_1", np.zeros(4, dtype=np.uint8), 2, True),
        ("uint16_count_up", np.array([0, 300, 7], dtype=np.uint16), 2, True),
        ("uint16_lowest_free", np.array([0, 300, 7], dtype=np.uint16), 2, False),
        ("none", np.array([0, 2], dtype=np.uint8), 0, False),
        ("none_count_up", np.array([0, 2], dtype=np.uint8), 0, True),
        ("uint8_full", full, 1, False),
        ("uint8_last_free", full[:-1], 1, False),
        ("uint8_count_up_overflow", np.array([255], dtype=np.uint8), 1, True),
        ("uint16_count_up_overflow", np.array([65534], dtype=np.uint16), 2, True),
    ]:
        case = {"id": case_id, "existing": array_record(existing), "count": count, "count_up": count_up}
        try:
            case["ids"] = [int(v) for v in fill._fresh_ids(existing, count, count_up)]
        except ValueError as e:
            case["error"] = str(e)
        cases.append(case)
    return cases


def make_seed_labels() -> list:
    def objects_of(shape, *blocks):
        objects = np.zeros(shape, dtype=np.int32)
        for value, sl in blocks:
            objects[sl] = value
        return objects

    two = objects_of((4, 6), (1, (slice(0, 2), slice(0, 2))), (2, (slice(2, 4), slice(3, 6))))
    cases = []
    for case_id, objects, existing, count_up in [
        ("fresh", two, np.zeros((4, 6), dtype=np.uint8), False),
        ("fresh_count_up", two, np.zeros((4, 6), dtype=np.uint8), True),
        ("painted_kept", two, objects_of((4, 6), (7, (0, 0)), (9, (3, 0))).astype(np.uint8), False),
        ("tie_takes_larger_id", two, objects_of((4, 6), (4, (0, slice(0, 1))), (6, (1, slice(0, 1))), (3, (3, 5))).astype(np.uint8), False),
        ("majority", two, objects_of((4, 6), (4, (0, slice(0, 2))), (6, (1, slice(0, 1)))).astype(np.uint8), False),
        ("skipped_object_id", objects_of((2, 5), (1, (0, 0)), (3, (1, 4))), np.zeros((2, 5), dtype=np.uint8), False),
        ("uint16_count_up", two, objects_of((4, 6), (300, (0, 5))).astype(np.uint16), True),
        ("same_id_two_objects", two, objects_of((4, 6), (5, (0, 0)), (5, (3, 5))).astype(np.uint8), False),
        ("no_foreground", np.zeros((3, 3), dtype=np.int32), objects_of((3, 3), (4, (1, 1))).astype(np.uint8), False),
        ("background_painted_ignored", two, objects_of((4, 6), (1, (0, 0)), (1, (2, 3))).astype(np.uint8), False),
    ]:
        case = {"id": case_id, "objects": array_record(objects), "existing": array_record(existing), "count_up": count_up}
        try:
            case["labels"] = array_record(fill.seed_labels(objects, existing, count_up=count_up))
        except ValueError as e:
            case["error"] = str(e)
        cases.append(case)
    return cases


def make_labels_for_box() -> list:
    def box(plane, depth_axis):
        return np.expand_dims(np.asarray(plane), depth_axis)

    cases = []

    # Each object an id; a pixel touching only at a corner is its own object.
    mask = np.zeros((6, 8), dtype=bool)
    mask[0:2, 0:2] = True
    mask[3:5, 4:7] = True
    mask[5, 7] = True
    for depth_axis in (0, 1, 2):
        cases.append(("each_object_an_id", mask, box(np.zeros((6, 8), dtype=np.uint8), depth_axis), depth_axis, False, None))

    # A painted object keeps its id.
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:4, 1:4] = True
    existing = np.zeros((6, 6), dtype=np.uint8)
    existing[1, 1] = 7
    existing[5, 5] = 9
    cases.append(("painted_kept", mask, box(existing, 0), 0, False, None))

    # Counting up on an instance volume.
    mask = np.zeros((4, 8), dtype=bool)
    mask[:, 0:2] = True
    mask[:, 4:6] = True
    existing = np.zeros((4, 8), dtype=np.uint16)
    existing[0, 7] = 300
    cases.append(("count_up", mask, box(existing, 1), 1, True, None))

    # An object continuing one in the plane before keeps its id; a new one skips the ids beside.
    before = np.ones((6, 12), dtype=np.uint16)
    before[1:4, 0:3] = 2
    before[1:4, 5:8] = 3
    mask = np.zeros((6, 12), dtype=bool)
    mask[1:4, 5:8] = True
    mask[1:4, 9:12] = True
    for depth_axis in (0, 1, 2):
        cases.append(("continues_beside", mask, box(np.zeros((6, 12), dtype=np.uint16), depth_axis), depth_axis, True,
                      [box(before, depth_axis), None]))

    # A new object on a uint8 volume skips the ids in the plane after.
    after = np.zeros((4, 8), dtype=np.uint8)
    after[:, 0:2] = 2
    mask = np.zeros((4, 8), dtype=bool)
    mask[:, 5:7] = True
    cases.append(("skips_ids_beside", mask, box(np.zeros((4, 8), dtype=np.uint8), 0), 0, False, [None, box(after, 0)]))

    # Both planes beside: the first given wins where both have an id.
    first = np.zeros((4, 8), dtype=np.uint8)
    first[1:3, 1:3] = 5
    second = np.zeros((4, 8), dtype=np.uint8)
    second[1:3, 1:3] = 6
    second[1:3, 5:7] = 8
    mask = np.zeros((4, 8), dtype=bool)
    mask[1:3, 1:3] = True
    mask[1:3, 5:7] = True
    cases.append(("first_plane_wins", mask, box(np.zeros((4, 8), dtype=np.uint8), 2), 2, False, [box(first, 2), box(second, 2)]))

    out = []
    for case_id, mask, existing, depth_axis, count_up, beside in cases:
        objects, _ = label(mask)
        labels = pipeline.labels_for_box(objects, existing, depth_axis, count_up=count_up, beside=beside)
        out.append({
            "id": f"{case_id}_axis{depth_axis}", "objects": array_record(objects), "existing": array_record(existing),
            "depth_axis": depth_axis, "count_up": count_up,
            "beside": None if beside is None else [None if p is None else array_record(p) for p in beside],
            "labels": array_record(labels),
        })
    return out


# --- paint_box and restore_box --------------------------------------------------

STATE = {"bucket": "annotations"}


class _Bucket:
    """MinIO's bucket as s3fs sees it: an in-memory zarr per key (tests/ai_annotate/test_paint_box.py)."""

    def __init__(self):
        self.stores = {}

    def _split(self, path):
        root, _, rest = path.partition(".zarr/")
        return self.stores.get(root + ".zarr", {}), rest

    def exists(self, path):
        store, rest = self._split(path)
        return rest in store


def _served(initial: np.ndarray, chunks=(4, 4, 4)):
    """``initial`` as MinIO would serve it: fill's S3 calls answered by an in-memory zarr."""
    bucket = _Bucket()
    store = bucket.stores["annotations/vol-1.zarr"] = zarr.MemoryStore()
    minio.make_s3_filesystem = lambda state: bucket
    fill.s3fs.S3Map = lambda root, s3, check: s3.stores[root]
    labels = zarr.open_group(store).require_group("annotation").create_dataset(
        "s0", shape=initial.shape, chunks=chunks, dtype=initial.dtype, fill_value=0
    )
    labels[:] = initial
    return labels


def _painted_corner(dtype) -> np.ndarray:
    """8^3 of 0s with background (1) and an object (5) painted in the first slice of the first chunk."""
    painted = np.zeros((8,) * 3, dtype=dtype)
    painted[0, 0, :2] = 1
    painted[0, 1, :2] = 5
    return painted


def _seed(existing):
    """An object (2) over the box's first two rows of y, background (1) elsewhere."""
    labels = np.ones_like(existing)
    labels[:, :2, :] = 2
    return labels


def make_paint_box() -> list:
    across = _painted_corner(np.uint8)
    across[2, 3, 4] = 1  # inside the across-chunks box: kept without overwrite
    across[4, 5, 6] = 7

    def object_3(existing):
        labels = np.ones_like(existing)
        labels[1:3, 1:3, 1:3] = 3
        return labels

    def nothing(existing):
        return np.zeros_like(existing)

    uint16 = _painted_corner(np.uint16)
    uint16[0, 1, :2] = 300

    cases = []
    for case_id, initial, lo, hi, labels_for, overwrite, stroke in [
        ("seed", _painted_corner(np.uint8), (0, 0, 0), (4, 4, 4), _seed, False, [((0, 2, 0), 9), ((0, 0, 3), 9), ((6, 6, 6), 9)]),
        ("seed_overwrite", _painted_corner(np.uint8), (0, 0, 0), (4, 4, 4), _seed, True, [((0, 1, 0), 9)]),
        ("across_chunks", across, (1, 2, 3), (5, 6, 7), object_3, False, [((2, 2, 2), 1)]),
        ("across_chunks_overwrite", across, (1, 2, 3), (5, 6, 7), object_3, True, None),
        ("nothing_changes", _painted_corner(np.uint8), (0, 0, 0), (4, 4, 4), nothing, False, None),
        ("unchanged_labels", _painted_corner(np.uint8), (0, 0, 0), (4, 4, 4), lambda existing: existing.copy(), True, None),
        ("uint16", uint16, (0, 0, 0), (4, 4, 8), _seed, False, [((0, 3, 7), 400)]),
    ]:
        served = _served(initial)
        lo_arr, hi_arr = np.asarray(lo), np.asarray(hi)
        box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        labels = labels_for(initial[box])
        undo = []
        counts = fill.paint_box(STATE, "vol-1", lo_arr, hi_arr, labels_for, undo=undo, overwrite=overwrite)
        after = served[:]
        case = {
            "id": case_id, "dtype": str(initial.dtype), "shape": list(initial.shape), "chunks": [4, 4, 4],
            "initial": array_record(initial), "lo": list(lo), "hi": list(hi), "labels": array_record(labels), "overwrite": overwrite,
            "counts": {"n_foreground": counts[0], "n_background": counts[1], "n_overwritten": counts[2]},
            "after": array_record(after),
            "undo": None if not undo else {"before": array_record(undo[0][2]), "after": array_record(undo[0][3])},
        }
        if stroke is not None and undo:
            # A stroke painted after the action: restore puts back only what
            # the action changed and still holds.
            for index, value in stroke:
                served[tuple(index)] = value
            restored = fill.restore_box(STATE, "vol-1", lo_arr, hi_arr, undo[0][2], undo[0][3])
            case["restore"] = {
                "stroke": [[list(index), int(value)] for index, value in stroke],
                "restored": int(restored),
                "final": array_record(served[:]),
            }
        cases.append(case)
    return cases


# --- prompts and the catalog ----------------------------------------------------


def _profile_dict(profile) -> dict:
    return {"key": profile.key, "name": profile.name, "color_name": profile.color_name, "rgb": list(profile.rgb), "description": profile.description}


def make_prompts() -> dict:
    mito, er, cell, nucleus = (organelles.ORGANELLES[k] for k in ("mito", "er", "cell", "nucleus"))
    golgi = organelles.resolve_organelle_profile("golgi")
    cases = []
    for case_id, profile, resolution, override in [
        ("mito_8", mito, 8.0, None),
        ("mito_3_5", mito, 3.5, None),
        ("mito_4", mito, 4.0, None),
        ("er_12_3456", er, 12.3456, None),
        ("cell_1024", cell, 1024.0, None),
        ("nucleus_none", nucleus, None, None),
        ("override", mito, 8.0, "  Paint the dark blobs red.  "),
        ("blank_override", mito, 8.0, "   "),
        ("generic", golgi, 16.0, None),
        ("generic_override", golgi, 0.5, "Colour every Golgi stack bright red."),
        ("too_long", mito, 8.0, "x" * (prompts.MAX_PROMPT_CHARS + 1)),
        ("just_fits", mito, 8.0, "y" * prompts.MAX_PROMPT_CHARS),
    ]:
        case = {"id": case_id, "profile": _profile_dict(profile), "resolution_nm": resolution, "prompt_override": override}
        try:
            case["prompt"] = prompts.build_recolor_prompt(profile, resolution_nm=resolution, prompt_override=override)
        except AIAnnotateError as e:
            case["error"] = _error(e)
        cases.append(case)
    return {
        "max_prompt_chars": prompts.MAX_PROMPT_CHARS,
        "editable": {key: prompts.editable_prompt(profile) for key, profile in organelles.ORGANELLES.items()},
        "editable_generic": prompts.editable_prompt(golgi),
        "cases": cases,
    }


def make_organelles() -> dict:
    lookups = []
    for name in ["mito", "Mitochondria", " nuclear-pores ", "lipid_droplet", "Cell Membrane", "ENDOPLASMIC RETICULUM", "nuclear_pore",
                 "Nucleolus", "cells", "cell", "Lipid Droplet", "golgi", "", "mito-chondria", "plasma-membrane"]:
        profile = organelles.find_organelle_profile(name)
        lookups.append({"name": name, "key": None if profile is None else profile.key})
    return {
        "catalog": [_profile_dict(p) for p in organelles.ORGANELLES.values()],
        "lookups": lookups,
        "resolved": [{"name": name, "profile": _profile_dict(organelles.resolve_organelle_profile(name))} for name in ("golgi", " golgi apparatus ", "Mito")],
    }


# --- writing and checking -------------------------------------------------------


def make_ai_annotate() -> dict:
    rng = np.random.default_rng(SEED)
    plans, plan_objects = make_plans()
    return {
        "_format": (
            "Parity fixture for web/src/ai-annotate (geometry, plane, mask, labels, fill, prompts, organelles) "
            "against cellmap_flow/ai_annotate and finetune/session/fill.py; arrays are {dtype, shape, data} flat in C order; "
            "see make_ai_fixtures.py's docstring for the sections"
        ),
        "seed": SEED,
        "constants": {"max_image_side": geometry.MAX_IMAGE_SIDE, "plane_names": list(geometry.PLANE_NAMES)},
        "screen": make_screen(),
        "plan_plane": plans,
        "raw_box": make_raw_boxes(plan_objects),
        "extract_mask": make_extract_mask(),
        "normalize_to_uint8": make_normalize(rng),
        "resize": make_resize(rng),
        "mask_to_write_shape": make_mask_to_write_shape(rng, plan_objects),
        "box_voxels": make_box_voxels(),
        "fresh_ids": make_fresh_ids(),
        "seed_labels": make_seed_labels(),
        "labels_for_box": make_labels_for_box(),
        "paint_box": make_paint_box(),
        "prompts": make_prompts(),
        "organelles": make_organelles(),
    }


MAKERS = {"ai_annotate.json": make_ai_annotate}


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
    print("AI annotation fixture is up to date")
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
