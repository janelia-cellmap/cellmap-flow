"""Planning an annotation volume and its good regions on numbers alone
(cellmap_flow.portable.volume and regions), the rules the server runs in its
process and the browser through volume_calls."""

import pytest

from cellmap_flow.portable import regions, volume, volume_calls


def _model(voxel_in, voxel_out, block_in, block_out):
    return {"input_voxel_size": [voxel_in] * 3, "output_voxel_size": [voxel_out] * 3,
            "input_shape": [block_in] * 3, "output_shape": [block_out] * 3}


@pytest.mark.parametrize("level, model, voxel, offset, shape", [
    # The level at the model's size: on its grid, from its corner (-4 nm, a
    # Janelia pyramid's), each output voxel two of the level's, the 33rd
    # level voxel's half an output voxel rounded up, then whole chunks of 4.
    pytest.param({"voxel": [8.0] * 3, "corner": [-4.0] * 3, "shape": [33, 32, 32]}, _model(8, 16, 12, 4),
                 [16.0] * 3, [4.0] * 3, [20, 16, 16], id="the model's level"),
    # No 8 nm level: the model reads the 10 nm one, and its prediction is
    # served at 20 nm, where the volume lies too (snapping 16 nm to a level
    # of its own made it 10 nm, chunks half the model's blocks).
    pytest.param({"voxel": [10.0] * 3, "corner": [0.0] * 3, "shape": [40] * 3}, _model(8, 16, 12, 4),
                 [20.0] * 3, [10.0] * 3, [20] * 3, id="the model's ratio on another level"),
])
def test_a_volume_lies_on_the_grid_its_prediction_is_served_on(level, model, voxel, offset, shape):
    plan = volume.plan_volume(level, model)
    assert (plan["output_voxel_size"], plan["dataset_offset_nm"], plan["dataset_shape_voxels"]) == (voxel, offset, shape)
    assert (plan["input_voxel_size"], plan["chunk_size"], plan["input_size"]) == (level["voxel"], [4] * 3, [12] * 3)
    assert plan["claimed_output_voxel_size"] == [16] * 3, "the model's own, ints kept ints"


def test_a_block_in_nm_is_whole_voxels_or_refused():
    reported = {"input_voxel_size": [16] * 3, "output_voxel_size": [16] * 3,
                "read_shape": [2848.0] * 3, "write_shape": [896.0] * 3}
    assert volume.block_shapes(reported) == ([178] * 3, [56] * 3)
    with pytest.raises(ValueError, match="not whole voxels"):
        volume.block_shapes({**reported, "write_shape": [900.0] * 3})


@pytest.mark.parametrize("size_nm, expected", [
    pytest.param(None, [56 * 16.0] * 3, id="none: one output patch"),
    pytest.param(0, [56 * 16.0] * 3, id="0: one output patch, not a refusal"),
    pytest.param(40, [40.0] * 3, id="one number"),
    pytest.param([8, 48, 48], [8.0, 48.0, 48.0], id="three"),
])
def test_the_size_a_mark_asks_for(size_nm, expected):
    assert regions.region_size_nm(size_nm, [56] * 3, [16] * 3) == expected
    assert regions.region_size_nm(None) == [regions.DEFAULT_REGION_SIZE_NM] * 3, "no volume"


@pytest.mark.parametrize("size_nm", [-1, [0, 8, 8], [8, 8], float("nan"), "40", True])
def test_a_size_that_is_not_one_is_refused(size_nm):
    with pytest.raises(ValueError, match="size_nm must be positive"):
        regions.region_size_nm(size_nm)


def test_a_region_anchors_one_output_block_snapped_to_whole_voxels_and_clipped():
    grid = {"corner_nm": [0.0] * 3, "voxel_nm": [8.0] * 3, "shape_voxels": [10, 100, 100], "block": [4, 6, 6]}
    marked = [
        regions.new_region([20.0, 404.0, 400.0], [32.0, 48.0, 48.0], 0),  # centre (2.5, 50.5, 50) voxels
        regions.new_region([4.0, 4.0, 4.0], [8.0] * 3, 1),  # centre (0.5, 0.5, 0.5): its block runs past 0
        {"offset_nm": [0, 0], "shape_nm": [8, 8, 8]},  # not a region
        regions.new_region([2000.0] * 3, [8.0] * 3, 2),  # marked on another volume
    ]
    assert [r["label"] for r in marked if "label" in r] == ["good-1", "good-2", "good-3"]
    placed = regions.anchor_boxes(marked, **grid)
    assert placed["boxes"] == [{"region": 0, "lo": [1, 48, 47], "hi": [5, 54, 53]},
                               {"region": 1, "lo": [0, 0, 0], "hi": [3, 4, 4]}]
    assert (placed["malformed"], placed["outside"]) == ([2], [3])


def test_the_browsers_calls_answer_a_refusal_as_its_sentence():
    level = {"voxel": [8.0] * 3, "corner": [-4.0] * 3, "shape": [32] * 3}
    planned = volume_calls.plan_volume({"level": level, "model": _model(8, 8, 12, 4), "output_type": "affinities"}, [])
    assert planned["dtype"] == "uint16" and planned["dataset_shape_voxels"] == [32] * 3
    assert volume_calls.plan_volume({"level": level, "model": _model(8, 8, 12, 4)}, [])["dtype"] == "uint8"
    refused = volume_calls.mark_region({"centre_nm": [0, 0, 0], "size_nm": -1, "regions": []}, [])
    assert refused == {"error": "size_nm must be positive"}
