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


# A Janelia-style 16 nm level: its corner at -4 nm, thinner in z than a chunk.
LEVEL_16 = {"voxel": [16.0] * 3, "corner": [-4.0] * 3, "shape": [21, 100, 130]}


@pytest.mark.parametrize("voxel_size, offset, chunk, shape, input_size", [
    # The level's own size: its voxels, from its corner; 64^3 chunks, as
    # thin as the volume in z, the extent padded to whole chunks.
    pytest.param([16.0] * 3, [4.0] * 3, [21, 64, 64], [21, 128, 192], [21, 64, 64], id="a level's own"),
    # A size typed between levels, over the level read for it: the level's
    # extent in 24 nm voxels (14, 66.7 and 86.7 of them, rounded up), from
    # the same corner, a chunk 1.5 times as many of the level's voxels.
    pytest.param([24.0] * 3, [8.0] * 3, [14, 64, 64], [14, 128, 128], [21, 96, 96], id="typed"),
])
def test_a_paint_first_volume_lies_on_the_raws_grid_at_the_size_asked(voxel_size, offset, chunk, shape, input_size):
    plan = volume.plan_paint_volume(LEVEL_16, voxel_size)
    assert (plan["output_voxel_size"], plan["dataset_offset_nm"]) == (voxel_size, offset)
    assert (plan["chunk_size"], plan["dataset_shape_voxels"], plan["input_size"]) == (chunk, shape, input_size)
    assert plan["input_voxel_size"] == LEVEL_16["voxel"], "the raw under the paint is read at the level's own"
    assert plan["claimed_output_voxel_size"] is None and plan["claimed_input_voxel_size"] is None, "no model claims one"
    with pytest.raises(ValueError):
        volume.plan_paint_volume(LEVEL_16, [16.0, 16.0])


@pytest.mark.parametrize("pixel_nm, index", [
    pytest.param(None, 0, id="no zoom known: the finest"),
    pytest.param(4.0, 0, id="zoomed in past the finest"),
    pytest.param(20.0, 1, id="between two levels: the coarser one no coarser than a pixel"),
    pytest.param(32.2, 2, id="a pixel a hair finer than a level is that level"),
    pytest.param(1000.0, 2, id="zoomed out past the coarsest"),
])
def test_a_paint_first_volumes_default_is_the_level_the_viewer_shows(pixel_nm, index):
    assert volume.shown_level([[8.0] * 3, [16.0] * 3, [32.0] * 3], pixel_nm) == index
    assert volume_calls.shown_level({"voxel_sizes": [[8.0] * 3, [16.0] * 3, [32.0] * 3], "pixel_nm": pixel_nm}, []) == {"index": index}


def test_a_paint_first_volume_is_not_trained_on_and_says_where_its_paint_gets_a_grid():
    planned = volume_calls.plan_paint_volume({"level": LEVEL_16, "voxel_size": [16.0] * 3}, [])
    assert planned["dtype"] == "uint8"
    assert set(volume_calls.plan_paint_volume({"level": LEVEL_16, "voxel_size": [0, 16, 16]}, [])) == {"error"}
    assert volume.training_refusal({"model_name": None})
    assert volume.training_refusal({"model_name": "mito"}) is None
    assert set(volume_calls.training_refusal({"volume": {"model_name": None}}, [])) == {"error"}
    assert volume_calls.training_refusal({"volume": {"model_name": "mito"}}, []) == {}
