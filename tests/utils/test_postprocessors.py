"""The segmentation postprocessors, now running on ``post.segment``.

Moving their logic there must not change what a served layer shows: by
default each gives what it gave before, checked against the code it ran
then. The new options reach the dashboard's forms through the constructor's
signature, as every step's do.
"""

import numpy as np
import pytest

from cellmap_flow.post.postprocessors import (
    AffinityPostprocessor,
    LabelPostprocessor,
    get_postprocessors,
    get_postprocessors_list,
)

OFFSETS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [3, 0, 0], [0, 3, 0], [0, 0, 3], [9, 0, 0], [0, 9, 0], [0, 0, 9]]


def _random_mask(seed=0, shape=(2, 12, 12, 12)):
    return (np.random.default_rng(seed).random(shape) > 0.6).astype(np.uint8)


def test_label_postprocessor_by_default_labels_as_scipy_did():
    from scipy.ndimage import label

    data = _random_mask()
    out = LabelPostprocessor()(data, chunk_corner=(0, 0, 0), chunk_num_voxels=data[0].size)
    expected = data.astype(np.uint32)
    expected[0] = label(data[0])[0]
    assert out.dtype == np.uint32
    np.testing.assert_array_equal(out, expected)


def test_label_postprocessor_options_drop_specks_and_label_by_slice():
    data = np.zeros((1, 3, 6, 6), np.uint8)
    data[0, :, 1:4, 1:4] = 1  # a column through the three slices
    data[0, 0, 5, 5] = 1  # a speck
    kwargs = dict(chunk_corner=(0, 0, 0), chunk_num_voxels=data[0].size)
    assert LabelPostprocessor()(data, **kwargs).max() == 2
    assert LabelPostprocessor(min_size=2)(data, **kwargs).max() == 1
    assert LabelPostprocessor(per_slice=True, min_size=2)(data, **kwargs).max() == 3
    # Corner neighbours touch only when every neighbour does.
    corner = np.zeros((1, 2, 2, 2), np.uint8)
    corner[0, 0, 0, 0] = corner[0, 1, 1, 1] = 1
    assert LabelPostprocessor(connectivity=3)(corner, **kwargs).max() == 1


def test_the_dashboards_strings_become_the_label_options_types():
    post = LabelPostprocessor(channel="0", connectivity="2", min_size="5", per_slice="False")
    assert post.to_dict() == {"name": "LabelPostprocessor", "channel": 0, "connectivity": 2, "min_size": 5,
                              "per_slice": False}
    (rebuilt,) = get_postprocessors([LabelPostprocessor(per_slice="true").to_dict()])
    assert rebuilt.per_slice is True
    with pytest.raises(ValueError):
        LabelPostprocessor(connectivity="4")


def test_the_forms_list_the_new_label_options():
    (label_entry,) = [p for p in get_postprocessors_list() if p["name"] == "LabelPostprocessor"]
    assert label_entry["params"] == {"channel": 0, "connectivity": 1, "min_size": 0, "per_slice": False}


def _old_affinity_postprocessor(data, bias, neighborhood, chunk_num_voxels, chunk_corner):
    """AffinityPostprocessor._process as it was before post.segment, use_exact on."""
    import fastremap
    import mwatershed as mws
    import pymorton
    from scipy import ndimage

    data = data / 255.0 if np.issubdtype(data.dtype, np.integer) else data.astype(np.float64)
    segmentation = mws.agglom(data.astype(np.float64) - bias, neighborhood[: data.shape[0]])
    average_affs = np.mean(data, axis=0)
    fragment_ids = fastremap.unique(segmentation[segmentation > 0])
    kept = [f for f, m in zip(fragment_ids, ndimage.mean(average_affs, segmentation, fragment_ids)) if m >= bias]
    fastremap.mask_except(segmentation, kept, in_place=True)
    fastremap.renumber(segmentation, in_place=True)
    segmentation[segmentation > 0] += np.uint64(chunk_num_voxels * pymorton.interleave(*chunk_corner))
    return np.expand_dims(segmentation.astype(np.uint64), axis=0)


@pytest.mark.parametrize("bias, channels, as_uint8", [
    pytest.param(0.0, 9, False, id="the-defaults"),
    pytest.param(0.5, 3, False, id="probabilities"),
    pytest.param(0.5, 9, True, id="uint8-from-the-default-postprocessor"),
])
def test_affinity_postprocessor_segments_as_it_did(bias, channels, as_uint8):
    affs = np.random.default_rng(1).random((channels, 10, 10, 10))
    affs[..., 5] *= 0.2  # a weak wall
    data = (affs * 255).astype(np.uint8) if as_uint8 else affs.astype(np.float32)
    kwargs = dict(chunk_num_voxels=1000, chunk_corner=(1, 0, 2))
    out = AffinityPostprocessor(bias=bias)(data, **kwargs)
    expected = _old_affinity_postprocessor(data, bias, OFFSETS, **kwargs)
    assert out.dtype == np.uint64
    np.testing.assert_array_equal(out, expected)


def test_cellpose_masks_need_a_flows_output_and_cellpose():
    """On a server without Cellpose, or on another output, it says what it needs."""
    from cellmap_flow.post.postprocessors import CellposeMasksPostprocessor

    with pytest.raises((RuntimeError, ValueError), match="Cellpose"):
        CellposeMasksPostprocessor()._process(np.zeros((1, 2, 8, 8), np.float32))


def test_cellpose_masks_follow_the_flows_to_their_objects_and_link_them_across_slices():
    pytest.importorskip("cellpose.dynamics")
    from cellpose import dynamics

    from cellmap_flow.post.postprocessors import CellposeMasksPostprocessor

    yy, xx = np.mgrid[:96, :96]
    labels = np.zeros((96, 96), np.int32)
    labels[(yy - 30) ** 2 + (xx - 30) ** 2 < 15 ** 2] = 1
    labels[(yy - 62) ** 2 + (xx - 62) ** 2 < 15 ** 2] = 2
    flows = getattr(dynamics, "masks_to_flows", None) or dynamics.masks_to_flows_gpu
    flow = np.asarray(flows(labels), np.float32) * 5
    probability = np.where(labels > 0, 0.95, 0.05).astype(np.float32)
    data = np.stack([np.stack([flow[0]] * 3), np.stack([flow[1]] * 3), np.stack([probability] * 3)])
    apart = CellposeMasksPostprocessor()._process(data)
    linked = CellposeMasksPostprocessor(stitch_threshold=0.3)._process(data)
    assert apart.shape == (1, 3, 96, 96) and apart.dtype == np.uint32
    assert [len(np.unique(apart[0, z])) - 1 for z in range(3)] == [2, 2, 2]
    assert len(np.unique(apart)) - 1 == 6 and len(np.unique(linked)) - 1 == 2


def test_cellpose_masks_say_which_models_cannot_take_them():
    """Run on a Cellpose server serving its probability, it failed on every chunk
    inside the server, and the page showed an empty layer."""
    from types import SimpleNamespace

    from cellmap_flow.models.models_config import CellposeModelConfig, ScriptModelConfig
    from cellmap_flow.post.postprocessors import CellposeMasksPostprocessor

    step = CellposeMasksPostprocessor()
    flows = CellposeModelConfig(voxel_size=8, output="flows", name="cp_flows")
    assert step.problem_with(flows) is None
    assert step.problem_with(SimpleNamespace(name="ft", base_model_config=flows)) is None
    assert "Output: All channels" in step.problem_with(CellposeModelConfig(voxel_size=8, output="probability", name="cp"))
    assert "not a Cellpose model" in step.problem_with(ScriptModelConfig(script_path="/s.py", name="mito"))


def test_cellpose_masks_make_one_channel_of_three_so_the_server_declares_one():
    """The server went on declaring the flows' three channels, and the viewer
    read each one-channel chunk of masks as three."""
    from cellmap_flow.pipeline_spec import chain_is_segmentation, chain_num_channels, chain_output_dtype
    from cellmap_flow.post.postprocessors import CellposeMasksPostprocessor, MortonSegmentationRelabeling

    chain = [CellposeMasksPostprocessor(), MortonSegmentationRelabeling()]
    assert chain_num_channels(chain, 3) == 1 and chain_is_segmentation(chain)
    assert chain_output_dtype(chain[:1], np.float32) == np.uint32
