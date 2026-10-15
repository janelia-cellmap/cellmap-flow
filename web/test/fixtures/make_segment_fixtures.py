"""Generate the parity fixtures for the two segmentation steps the browser runs
on labels: ``LabelPostprocessor`` (``web/src/infer/label.ts``) and
``SimpleBlockwiseMerger`` (``web/src/infer/merger.ts``), both from
``cellmap_flow/post/postprocessors.py``.

A separate generator from ``make_fixtures.py`` (whose helpers and conventions
it shares): these steps are stateful or read the chunk's position, so their
cases are sequences of chunks rather than one block through a chain. Run it
with a Python that has cellmap_flow's dependencies (scipy, fastmorph,
neuroglancer), from anywhere::

    .pixi/envs/default/bin/python web/test/fixtures/make_segment_fixtures.py          # regenerate
    .pixi/envs/default/bin/python web/test/fixtures/make_segment_fixtures.py --check  # CI: stale?

``--check`` regenerates into a temporary directory and compares with the
committed files, exiting 1 when the Python has changed and the fixtures
have not; ``tests/web/test_fixtures_fresh.py`` runs that in pytest.

Everything is deterministic (seeds recorded in the files) and exact: the
outputs are integer labels, compared exactly. Arrays are ``{"dtype",
"shape", "data"}`` with ``data`` flat in C order; uint64 arrays and ids carry
their values as decimal strings, which JSON and JavaScript numbers could not
hold exactly past 2^53.

``label.json``: one uint8 block of masks, channel first, and for each case
the step's parameters (as numbers and as the strings the dashboard's forms
send) and the uint32 block ``LabelPostprocessor`` returns, so the ids are
compared exactly against scipy's numbering.

``merger.json``: sequences of chunks of labels with objects crossing the
shared faces, fed in order to one ``SimpleBlockwiseMerger`` with their
corners (the chunk's index on the served grid, as the server passes it);
after each chunk the uint64 block it returns and
``equivalences_json()``, neuroglancer's ``EquivalenceMap.to_json()``: the
sets of joined ids, each sorted, the sets sorted.
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

from make_fixtures import ROOT, array_record, differences  # noqa: E402  (puts the worktree's package first on sys.path)

from cellmap_flow.post.postprocessors import LabelPostprocessor, SimpleBlockwiseMerger  # noqa: E402

FIXTURES = ("label.json", "merger.json")


def uint64_record(arr: np.ndarray) -> dict:
    """``array_record`` with the values as decimal strings: exact for uint64."""
    arr = np.ascontiguousarray(arr)
    return {"dtype": arr.dtype.name, "shape": list(arr.shape), "data": [str(int(v)) for v in arr.ravel()]}


def run_step(step, data: np.ndarray, corner=(0, 0, 0)) -> np.ndarray:
    """The step on one chunk as the server's PostprocessOp.apply_at calls it:
    with the chunk's corner (its index on the served grid) and voxel count;
    SerializableInterface.process drops what the step's _process does not
    take and casts the result to the step's declared dtype."""
    return step(data, chunk_corner=tuple(int(v) for v in corner), chunk_num_voxels=int(np.prod(data.shape[1:])))


# --- label.json ---------------------------------------------------------------

LABEL_SHAPE = (2, 5, 6, 7)  # (c, z, y, x): two channels, so `channel` is tested

LABEL_CASES = [
    # (id, parameters). Numbers and the forms' strings both, to pin the parsing.
    ("defaults", {}),
    ("conn_1", {"connectivity": 1}),
    ("conn_2", {"connectivity": 2}),
    ("conn_3", {"connectivity": 3}),
    ("conn_2_string", {"connectivity": "2"}),
    ("min_size_4", {"min_size": 4}),
    ("min_size_4_conn_3_strings", {"connectivity": "3", "min_size": "4"}),
    ("per_slice", {"per_slice": True}),
    ("per_slice_string_conn_2", {"per_slice": "true", "connectivity": "2"}),
    # In a 2-D slice connectivity 3 is capped at 2 (as_connectivity's ndim).
    ("per_slice_conn_3_capped", {"per_slice": True, "connectivity": 3}),
    ("per_slice_min_size_3", {"per_slice": True, "min_size": 3}),
    ("per_slice_false_string", {"per_slice": "false"}),
    ("channel_1", {"channel": 1}),
    ("channel_1_strings", {"channel": "1", "connectivity": "3", "min_size": "2"}),
]


def label_input(seed: int) -> np.ndarray:
    """A (2, 5, 6, 7) uint8 block: channel 0 about 30% foreground with values
    1 and 2 (nonzero is foreground, not only 1), channel 1 sparser with
    values up to 255."""
    rng = np.random.default_rng(seed)
    spatial = LABEL_SHAPE[1:]
    data = np.zeros(LABEL_SHAPE, dtype=np.uint8)
    data[0] = np.where(rng.random(spatial) < 0.3, rng.integers(1, 3, size=spatial), 0)
    data[1] = np.where(rng.random(spatial) < 0.3, rng.integers(1, 256, size=spatial), 0)
    return data


def _label_results(data: np.ndarray) -> dict:
    results = {}
    for case_id, params in LABEL_CASES:
        out = run_step(LabelPostprocessor(**params), data)
        assert out.dtype == np.uint32, f"{case_id}: LabelPostprocessor returns uint32, got {out.dtype}"
        assert out.shape == data.shape, f"{case_id}: the shape is kept"
        results[case_id] = out
    return results


def _label_tells_apart(results: dict, data: np.ndarray) -> bool:
    """Whether the block makes every parameter matter: the three
    connectivities give three labellings, min_size drops something,
    per_slice differs from 3-D, and channel 1 has several objects."""
    conn = [results[f"conn_{c}"][0] for c in (1, 2, 3)]
    distinct = not np.array_equal(conn[0], conn[1]) and not np.array_equal(conn[1], conn[2]) and conn[2].max() >= 2
    drops = results["min_size_4"][0].max() < results["conn_1"][0].max()
    per_slice = not np.array_equal(results["per_slice"][0], results["conn_1"][0])
    several = results["channel_1"][1].max() >= 3
    # The other channels pass through, cast: channel 1 is untouched by a channel-0 case.
    passes = np.array_equal(results["conn_1"][1], data[1])
    return distinct and drops and per_slice and several and passes


def make_label() -> dict:
    for seed in range(200):
        data = label_input(seed)
        results = _label_results(data)
        if _label_tells_apart(results, data):
            break
    else:
        raise RuntimeError("no seed under 200 gives a label block that tells every parameter apart")
    cases = []
    for case_id, params in LABEL_CASES:
        out = results[case_id]
        channel = int(params.get("channel", 0))
        cases.append(
            {
                "id": case_id,
                "step": {"name": "LabelPostprocessor", **params},
                "output": array_record(out),
                "num_objects": int(out[channel].max()),
            }
        )
    return {
        "_format": (
            "LabelPostprocessor, from cellmap_flow/post/postprocessors.py (segment.connected_components: "
            "scipy.ndimage.label with generate_binary_structure(3, connectivity), per slice in 2-D when "
            "per_slice, then remove_small). `input` is one uint8 block (c, z, y, x) of masks; each case "
            "runs LabelPostprocessor(**step) on it as the server does (chunk_corner (0, 0, 0)) and records "
            "the uint32 block it returns in `output`, ids exactly as scipy numbers them, and the number of "
            "objects in the labelled channel. Arrays are {dtype, shape, data} with data flat in C order. "
            "Generated by make_segment_fixtures.py; do not edit."
        ),
        "seed": seed,
        "input": array_record(data),
        "cases": cases,
    }


# --- merger.json --------------------------------------------------------------

MERGER_BLOCK = (4, 5, 6)  # (z, y, x) voxels of one chunk: three different face shapes
# A 2x2 of chunks in y and x on z = 0, and one above the first along z: chunks
# meet across faces on every axis, and the faces towards absent chunks stay
# recorded, waiting.
CORNERS = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0)]


def merger_chunks() -> dict:
    """The chunks' labels, by corner: objects placed by hand so that each
    thing the merger does happens somewhere, with ids unique across chunks
    (as Morton leaves them: 1.. in the first chunk, 1001.. in the second,
    and so on). ``MERGER_EXPECTED`` says what the Python must make of them.

    A face is eroded as fastmorph erodes it: a voxel keeps its id only if
    all eight neighbours in the face carry it, and the face's edge is
    background. So an object meeting a face in a 3x3 patch survives one
    erosion (its centre), nothing survives two, and a one-voxel bar none."""
    chunks = {corner: np.zeros(MERGER_BLOCK, dtype=np.uint32) for corner in CORNERS}
    c00, c01, c10, c11, top = (chunks[corner] for corner in CORNERS)
    # Object 1: through c00's +x face into c01 and through its +z face into the
    # chunk above, a 3x3 patch across each face: one object with three ids.
    c00[0:4, 1:4, 3:6] = 1
    c01[0:4, 1:4, 0:3] = 1001
    top[0, 1:4, 3:6] = 4003
    # Object 2: a one-voxel-thin bar from c00 into c10 across the y face,
    # joined unless the faces are eroded.
    c00[3, 3:5, 0] = 2
    c10[3, 0:2, 0] = 2001
    # Object 3: a 3x3 patch across the z face, beside object 1's.
    c00[3, 0:3, 0:3] = 3
    top[0, 0:3, 0:3] = 4001
    # Object 4: one voxel in c00's corner, on three faces whose neighbours are
    # never computed: recorded, never matched.
    c00[0, 0, 0] = 4
    # Object 5: from c01 into c11 across the y face in a 3x3 patch; in c11 it
    # reaches the face towards c10, which has nothing there: the Python finds
    # c10's face empty and leaves the pair alone, and nothing is joined.
    c01[1:4, 4, 3:6] = 1002
    c11[1:4, 0:2, 0:6] = 3001
    # Object 6: alone in the top chunk, on faces towards absent chunks.
    top[2, 4, 5] = 4002
    return chunks


# What SimpleBlockwiseMerger must make of merger_chunks(), at the end of each
# case: equivalences_json(), as sets of decimal strings.
MERGER_EXPECTED = {
    "grid": [["1", "1001", "4003"], ["2", "2001"], ["3", "4001"], ["1002", "3001"]],
    "eroded_faces": [["1", "1001", "4003"], ["3", "4001"], ["1002", "3001"]],
    "eroded_twice_string": [],
    # Object 1 with the same id, 7, in all three chunks: an id that meets only itself is a set of one.
    "same_id_meets_itself": [["2", "2001"], ["3", "4001"], ["7"], ["1002", "3001"]],
}
MERGER_EXPECTED.update({case: MERGER_EXPECTED["grid"] for case in ("channel_1_string", "chunk_seen_twice", "reverse_order")})


def equivalences_json(step: SimpleBlockwiseMerger) -> list:
    return [[str(int(i)) for i in group] for group in step.equivalences_json()]


def merger_sequences(seed: int) -> list:
    """``(id, parameters, [(corner, block), ...])`` per case."""
    rng = np.random.default_rng(seed)
    chunks = merger_chunks()
    grid = [(corner, chunks[corner][None]) for corner in CORNERS]
    # Object 1 numbered 7 everywhere, as a LabelPostprocessor alone could leave
    # an object (its ids are only unique within a chunk).
    same = [(corner, np.where(np.isin(block, [1, 1001, 4003]), 7, block).astype(np.uint32)) for corner, block in grid]
    # Two channels, the labels in channel 1 and something else in channel 0.
    two_channel = [(corner, np.concatenate([rng.integers(0, 50, size=block.shape).astype(np.uint32), block])) for corner, block in grid]
    return [
        ("grid", {}, grid),
        ("eroded_faces", {"face_erosion_iterations": 1}, grid),
        ("eroded_twice_string", {"face_erosion_iterations": "2"}, grid),
        ("channel_1_string", {"channel": "1"}, two_channel),
        ("same_id_meets_itself", {}, same),
        # A chunk computed again (the cache dropped it): its faces are recorded
        # again, and those already matched dropped again; nothing changes.
        ("chunk_seen_twice", {}, grid + [grid[0]]),
        # The viewer asks for chunks in any order; the sets at the end are the same.
        ("reverse_order", {}, list(reversed(grid))),
    ]


def make_merger() -> dict:
    seed = 0
    cases = []
    for case_id, params, chunks in merger_sequences(seed):
        step = SimpleBlockwiseMerger(**params)
        records = []
        for corner, block in chunks:
            out = run_step(step, block, corner)
            assert out.dtype == np.uint64, f"{case_id}: SimpleBlockwiseMerger returns uint64, got {out.dtype}"
            assert out.shape == block.shape and np.array_equal(out, block), f"{case_id}: the block passes through"
            records.append(
                {
                    "corner": list(corner),
                    "input": array_record(block),
                    "output": uint64_record(out),
                    "equivalences": equivalences_json(step),
                }
            )
        final = records[-1]["equivalences"]
        assert final == MERGER_EXPECTED[case_id], f"{case_id}: the Python joined {final}, the objects were placed for {MERGER_EXPECTED[case_id]}"
        cases.append({"id": case_id, "step": {"name": "SimpleBlockwiseMerger", **params}, "chunks": records})
    return {
        "_format": (
            "SimpleBlockwiseMerger, from cellmap_flow/post/postprocessors.py. Each case feeds its `chunks` "
            "in order to one SimpleBlockwiseMerger(**step), each a uint32 block (c, z, y, x) of labels at "
            "`corner` (the chunk's index on the served grid, the server's chunk_corner), and records after "
            "each the uint64 block the step returns (the input cast; data as decimal strings) and "
            "equivalences_json(): neuroglancer's EquivalenceMap.to_json(), the ids joined so far as sets, "
            "each sorted, the sets sorted, ids as decimal strings. An id that met only itself is a set of "
            "one. Arrays are {dtype, shape, data} with data flat in C order. Generated by "
            "make_segment_fixtures.py; do not edit."
        ),
        "seed": seed,
        "block": list(MERGER_BLOCK),
        "cases": cases,
    }


# --- writing and checking -----------------------------------------------------

MAKERS = {"label.json": make_label, "merger.json": make_merger}


def write_fixtures(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, make in MAKERS.items():
        text = json.dumps(make(), indent=2, ensure_ascii=False) + "\n"
        (out_dir / name).write_text(text, encoding="utf-8")
        size = len(text.encode("utf-8"))
        assert size < 400_000, f"{name} is {size} bytes; keep fixtures small enough to commit"
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
        print(f"{len(problems)} difference(s) between the committed fixtures and the Python code:")
        for problem in problems[:40]:
            print(f"  {problem}")
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
        print(f"Regenerate with: python {Path(__file__).relative_to(ROOT)}")
        return 1
    print("segmentation fixtures are up to date")
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
