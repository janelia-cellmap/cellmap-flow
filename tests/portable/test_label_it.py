"""Label it's segmentation of a prediction (``cellmap_flow.portable.label_it``),
the maths both modes run, and the browser's entry points to it
(``label_it_calls``), called here as chunkmirage's worker calls them: JSON
arguments, the arrays as bytes. The routes around it are
``tests/utils/test_view_labels.py``'s and the browser's
``web/test/unit/backend-view-labels.test.ts``."""

import json
import math
import subprocess
import sys
import warnings

import numpy as np
import pytest

from cellmap_flow.portable import label_it, label_it_calls
from cellmap_flow.portable.label_it import SIGNED_UNIT, UNBOUNDED, UNIT


def test_importing_it_loads_no_scipy():
    """The browser's Python starts with numpy alone: scipy loads on the first seed, not when a dataset opens."""
    code = (
        "import sys\n"
        "import cellmap_flow.portable.label_it, cellmap_flow.portable.label_it_calls\n"
        "print('scipy' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


@pytest.mark.parametrize("output_class, at_boundary", [
    (UNIT, 0.5), (SIGNED_UNIT, 0.0), (UNBOUNDED, 0.0), (None, 0.0),  # None: read off the values, logits here
])
def test_the_probability_is_one_half_at_each_activations_boundary(output_class, at_boundary):
    prediction = np.array([[[[at_boundary, -200.0 if output_class in (UNBOUNDED, None) else at_boundary]]]], np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a logit far below the boundary is 0, not an overflow warning
        p = label_it.probability(prediction, output_class)
    assert p.dtype == np.float32 and p[0, 0, 0, 0] == 0.5 and p[0, 0, 0, 1] in (0.0, 0.5)


def test_channels_past_the_models_are_cut_and_one_wholly_past_them_refused():
    prediction = np.zeros((2, 1, 1, 1), np.float32)
    assert label_it.probability(prediction, UNIT, slice(0, 3)).shape[0] == 2
    with pytest.raises(ValueError):
        label_it.probability(prediction, UNIT, slice(3, 4))


def test_seed_options_take_the_routes_defaults_and_refuse_what_they_refuse():
    assert label_it.seed_options({}) == {
        "method": "components", "threshold": 0.5, "min_size": 0, "connectivity": 1, "per_slice": False,
    }
    assert label_it.seed_options({"threshold": "0.3", "connectivity": "2", "per_slice": "true", "min_size": None}) == {
        "method": "components", "threshold": 0.3, "min_size": 0, "connectivity": 2, "per_slice": True,
    }
    # connectivity null is given, so refused, as the server has always read it.
    for body in ({"threshold": 1.5}, {"min_size": -1}, {"method": "watershed"}, {"connectivity": 4}, {"connectivity": None}):
        with pytest.raises(ValueError):
            label_it.seed_options(body)


@pytest.mark.parametrize("names, offsets", [
    (["mito_aff_z", "mito_aff_y", "mito_aff_x"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    # Counted, whatever the names say after _aff: the server's rule.
    (["mito_aff_x", "mito_aff_y", "mito_aff_z", "lsd_0"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    (["a_aff", "b_aff"], [[1, 0, 0], [0, 1, 0]]),
    ([f"aff_long_{i}_aff" for i in range(9)], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    (["nuc"], None),
])
def test_an_affinity_models_offsets_by_its_channel_names(names, offsets):
    assert label_it.affinity_offsets(names) == offsets


def test_a_seed_reads_the_selected_channel_else_the_affinities_else_every_channel():
    offsets = [[1, 0, 0], [0, 1, 0]]
    assert label_it.seed_channels("affinities", offsets, "3") == slice(3, 4)
    assert label_it.seed_channels("affinities", offsets) == slice(0, 2)
    assert label_it.seed_channels("binary", None) is None
    with pytest.raises(ValueError):
        label_it.seed_channels("binary", None, -1)


def test_components_are_strictly_over_the_threshold_and_numbered_per_slice_on_from_the_last():
    p = np.zeros((1, 2, 3, 5), np.float32)
    p[0, :, 1, 0] = 0.5  # at the threshold: background
    p[0, :, 1, 2:4] = 0.9  # a 2-voxel object in each slice
    p[0, 1, 0, 0] = 0.9  # a speck in the second slice
    objects = label_it.components(p, UNIT, None, 0.5, per_slice=True)
    # In raster order within a slice: the second slice's speck comes before its object.
    assert objects[0, 1, 2] == 1 and objects[1, 0, 0] == 2 and objects[1, 1, 2] == 3 and objects[:, 1, 0].max() == 0
    # Without per_slice the slices' objects touch; min_size drops the speck.
    assert label_it.components(p, UNIT, None, 0.5, min_size=2).max() == 1


# volume voxel 8 nm from voxel 0's centre at 4 nm (its corner at 0); the
# served level's voxel, translation and shape; the box; the served index of
# each of its voxels along every axis, and the box shrunk.
@pytest.mark.parametrize("served_voxel, translation, shape, lo, hi, index, box", [
    pytest.param(8, 4, 12, 2, 9, lambda i: i, (2, 9), id="the-same-grid"),
    pytest.param(16, 8, 8, 1, 9, lambda i: i // 2, (1, 9), id="coarser-two-voxels-to-one"),
    pytest.param(4, 2, 40, 0, 6, lambda i: 2 * i + 1, (0, 6), id="finer-the-one-each-centre-is-in"),
    pytest.param(8, 24, 10, 0, 6, lambda i: i - 2, (2, 6), id="served-from-20nm-the-box-shrinks"),
    pytest.param(8, 4, 6, 3, 9, lambda i: i, (3, 6), id="past-its-end-the-box-shrinks"),
])
def test_each_annotation_voxel_reads_the_served_voxel_its_centre_is_in(served_voxel, translation, shape, lo, hi, index, box):
    indices, new_lo, new_hi = label_it.placement([lo] * 3, [hi] * 3, [8] * 3, [4] * 3, [translation] * 3, [served_voxel] * 3, [shape] * 3)
    assert (new_lo, new_hi) == ([box[0]] * 3, [box[1]] * 3)
    for axis in indices:
        assert axis.tolist() == [index(i) for i in range(*box)]


def test_a_layer_drawn_at_another_voxel_size_keeps_its_offset_in_voxels_and_none_is_outside():
    # Declared at 8 nm from 4 nm, drawn at 16: its corner stays at voxel 0, its voxels twice as wide.
    indices, _, _ = label_it.placement([0] * 3, [4] * 3, [8] * 3, [4] * 3, [4] * 3, [8] * 3, [10] * 3, drawn=[16] * 3)
    assert indices[0].tolist() == [0, 0, 1, 1]
    assert label_it.placement([5, 0, 0], [8, 4, 4], [8] * 3, [4] * 3, [4] * 3, [8] * 3, [4] * 3) is None


def _call(name, args, arrays=()):
    """``label_it_calls.<name>`` as the worker calls it: args through JSON, arrays as bytes."""
    return getattr(label_it_calls, name)(json.loads(json.dumps(args)), [bytes(a) for a in arrays])


def test_the_browsers_seed_is_the_servers_on_the_annotation_grid():
    """A model served at twice the volume's voxel: placed, read, gathered and
    segmented by the calls as the server's seed does it (label_it.components)."""
    rng = np.random.default_rng(0)
    served = rng.normal(size=(3, 8, 8, 8)).astype(np.float32)
    box = {"lo": [1, 2, 3], "hi": [11, 12, 13], "volume": {"output_voxel_size": [8] * 3, "dataset_offset_nm": [4] * 3},
           "level": {"origin": [8] * 3, "voxel": [16] * 3, "shape": [8] * 3}}
    plan = _call("seed_plan", {"channel_names": ["m_aff_z", "m_aff_y", "m_aff_x"], "select_channel": None, **box})
    assert plan["channels"] == [0, 3] and plan["offsets"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    placement = plan["placement"]
    region = served[(slice(None),) + tuple(slice(a, b) for a, b in zip(placement["read_lo"], placement["read_hi"]))]
    options = _call("seed_options", {"min_size": 2})["options"]
    args = {**options, "output_class": None, "channels": plan["channels"], "offsets": plan["offsets"],
            "indices": placement["indices"], "shape": list(region.shape)}

    answer, (objects,) = _call("seed", args, [np.ascontiguousarray(region)])

    prediction = label_it.gather(region, [np.asarray(i) for i in placement["indices"]])
    expected = label_it.components(prediction, None, slice(0, 3), 0.5, 2)
    objects = np.frombuffer(objects, np.uint32).reshape(expected.shape)
    np.testing.assert_array_equal(objects, expected)
    assert answer["count"] == expected.max() > 1 and expected.shape == (10, 10, 10)

    # The mutex watershed's input, for the browser's own watershed.
    answer, (affinities,) = _call("seed", {**args, "method": "mutex_watershed"}, [np.ascontiguousarray(region)])
    np.testing.assert_array_equal(np.frombuffer(affinities, np.float32).reshape(answer["shape"]),
                                  label_it.affinities(prediction, None, plan["offsets"]))


def test_the_calls_answer_a_refusal_where_the_server_answers_400():
    assert "error" in _call("seed_options", {"connectivity": None})
    assert "error" in _call("seed_plan", {"channel_names": ["nuc"], "select_channel": -1, "lo": [0] * 3, "hi": [1] * 3,
                                           "volume": {"output_voxel_size": [8] * 3}, "level": {"origin": [4] * 3, "voxel": [8] * 3, "shape": [1] * 3}})
    region = np.zeros((1, 1, 1, 1), np.float32)
    args = {**label_it.seed_options({}), "output_class": None, "channels": [3, 4], "offsets": None, "indices": [[0]] * 3, "shape": [1, 1, 1, 1]}
    assert "error" in _call("seed", args, [region])
    view = {"viewer_px": [800, 600], "state": {"layout": "3d"}, "pixel_nm": 8, "depth_axis": 0,
            "display_dimensions": ["x", "y", "z"], "voxel_size": [8] * 3}
    assert "error" in _call("field_of_view", view)


# ---------------------------------------------------------------------------
# The field of view
# ---------------------------------------------------------------------------

# cellmap-flow's viewers show x, y, z (viewer.VIEWER_AXES).
XYZ = ("x", "y", "z")
TOP = label_it.NEUROGLANCER_TOP_ROW_PX


@pytest.mark.parametrize("layout, size_voxels", [
    pytest.param("xy", (1, 400, 600), id="xy-x-across-y-down"),
    pytest.param("xz", (400, 1, 600), id="xz-x-across-z-down"),
    pytest.param("yz", (600, 400, 1), id="yz-z-across-y-down"),
    pytest.param("4panel", (1, 200, 300), id="4panel-its-top-left-xy-panel"),
    pytest.param({"type": "xz-3d"}, (400, 1, 300), id="beside-a-3d-view"),
])
def test_the_field_of_view_is_what_the_panel_shows_at_the_slice_on_screen(layout, size_voxels):
    """A frame 600 x 400 pixels below Neuroglancer's top row, at one 8 nm
    voxel a pixel: the panel's plane is the AI annotation's for the layout,
    the panel's width along the axis it draws across, one voxel deep."""
    from cellmap_flow.ai_annotate.geometry import depth_axis_from_layout

    depth_axis = depth_axis_from_layout(layout, XYZ)
    size = label_it.field_of_view_nm((600, 400 + TOP), {"layout": layout}, 8.0, depth_axis, XYZ, (8, 8, 8))
    assert [s / 8 for s in size] == list(size_voxels)


def test_the_field_of_view_leaves_out_the_side_panels_shown_and_is_as_deep_as_the_box():
    """The layer panel (open on the Draw tab while painting) and the selection
    share the right-hand column, 300 pixels at its default; the statistics,
    resized, are below; the help panel is closed."""
    state = {"layout": "xy", "selectedLayer": {"layer": "paint", "visible": True}, "selection": {"visible": True},
             "statistics": {"visible": True, "size": 50}, "helpPanel": {"side": "left"}}
    size = label_it.field_of_view_nm((900, 400 + TOP + 50), state, 16.0, 0, XYZ, (16, 16, 16), depth_nm=48)
    assert size == [48.0, 400 * 16.0, 600 * 16.0]


def test_a_view_over_the_cap_is_refused_and_one_at_it_is_not():
    side = math.isqrt(label_it.FIELD_OF_VIEW_MAX_VOXELS)
    view = ({"layout": "xy"}, 8.0, 0, XYZ, (8, 8, 8))
    assert label_it.field_of_view_nm((side, side + TOP), *view) == [8.0, side * 8.0, side * 8.0]
    with pytest.raises(ValueError):
        label_it.field_of_view_nm((side + 1, side + TOP), *view)
    with pytest.raises(ValueError):  # the box's depth counts too
        label_it.field_of_view_nm((side, side + TOP), *view, depth_nm=16)
    with pytest.raises(ValueError):  # no slice panel
        label_it.field_of_view_nm((side, side + TOP), {"layout": "3d"}, *view[1:])
