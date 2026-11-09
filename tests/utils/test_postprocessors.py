"""Cellpose's masks, the segmentation postprocessor only the server runs, and
what ArgmaxPostprocessor's ids are.

What LabelPostprocessor and AffinityPostprocessor give, options and the
forms' strings included, is in web/test/fixtures (label.json, affinity.json),
and ArgmaxPostprocessor's on a block of logits in post.json, which
test_fixtures_fresh checks against this code; ``post.segment``, which
they run on, is test_segment's.
"""

import numpy as np
import pytest


# One voxel per row, its scores in channels 0, 1 and 2, channel-first as a
# block; web/test/parity/post.test.ts holds the port to the same ids.
ARGMAX_SCORES = np.array(
    [
        [0.1, 0.7, 0.2],  # channel 1 is best: id 2
        [0.6, 0.6, 0.1],  # a tie: the lower channel, id 1
        [0.2, 0.3, 0.35],  # best below the threshold
        [np.nan, 0.1, 0.9],  # the NaN does not win
        [np.nan, np.nan, np.nan],  # nothing to pick: background
    ],
    np.float32,
).T.reshape(3, 1, 1, 5)


def test_argmax_gives_each_voxel_its_best_channel_plus_one():
    """A model of exclusive classes, drawn as one layer of label ids: 0 only
    below the threshold, a tie to the lowest channel, a NaN never the best."""
    from cellmap_flow.post.postprocessors import ArgmaxPostprocessor

    scores = ARGMAX_SCORES
    ids = ArgmaxPostprocessor()(scores)
    assert ids.shape == (1, 1, 1, 5) and ids.dtype == np.uint8
    assert ids.ravel().tolist() == [2, 1, 3, 3, 0]
    # The form's strings: "None" (its default shown) is no threshold.
    assert ArgmaxPostprocessor(threshold="None")(scores).ravel().tolist() == [2, 1, 3, 3, 0]
    assert ArgmaxPostprocessor(threshold="0.5")(scores).ravel().tolist() == [2, 1, 0, 3, 0]

    # Ids past 255 need uint16.
    many = np.zeros((300, 1, 1, 2), np.float32)
    many[299, 0, 0, 0] = 1.0
    wide = ArgmaxPostprocessor()(many)
    assert wide.dtype == np.uint16 and wide.ravel().tolist() == [300, 1]


def test_argmax_gives_each_channel_the_id_it_is_given():
    """A model's classes as a dataset numbers them: an id of 0 makes its
    channel background, the dtype is the smallest that holds the ids, and
    a count of ids that is not the channels' is refused."""
    from cellmap_flow.post.postprocessors import ArgmaxPostprocessor

    for ids in ([50, 0, 37], "50,0,37", "50 0 37", " 50, 0  37 "):
        labelled = ArgmaxPostprocessor(ids=ids)(ARGMAX_SCORES)
        assert labelled.dtype == np.uint8 and labelled.ravel().tolist() == [0, 50, 37, 37, 0]
    assert ArgmaxPostprocessor(ids="None").ids is None
    assert ArgmaxPostprocessor(ids="300,1,2")(ARGMAX_SCORES).dtype == np.uint16
    assert ArgmaxPostprocessor(ids=[70000, 1, 2])(ARGMAX_SCORES).dtype == np.uint32
    with pytest.raises(ValueError, match=r"2\D+3"):  # says both counts
        ArgmaxPostprocessor(ids="1,2")(ARGMAX_SCORES)
    for refused in ("1,-2", "1,2.5", [1, -2]):
        with pytest.raises(ValueError):
            ArgmaxPostprocessor(ids=refused)


def test_an_argmax_chain_declares_one_channel_of_labels():
    """So the dashboard draws one segmentation layer, and blockwise writes a
    label volume in the ids' dtype."""
    from cellmap_flow.pipeline_spec import PipelineSpec, chain_is_segmentation, chain_num_channels, chain_output_dtype

    steps = PipelineSpec((), [{"name": "ChannelSelection", "channels": "2,0"}, {"name": "ArgmaxPostprocessor"}]).build()[1]
    assert chain_num_channels(steps, 8) == 1
    assert chain_is_segmentation(steps) is True
    assert np.dtype(chain_output_dtype(steps, np.float32)) == np.uint8
    (argmax,) = PipelineSpec((), [{"name": "ArgmaxPostprocessor"}]).build()[1]
    assert np.dtype(chain_output_dtype([argmax], np.float32, 300)) == np.uint16
    (mapped,) = PipelineSpec((), [{"name": "ArgmaxPostprocessor", "ids": "2,1,300"}]).build()[1]
    assert np.dtype(chain_output_dtype([mapped], np.float32)) == np.uint16


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


def test_cellpose_masks_follow_3d_flows_to_3d_objects():
    """A 3D model's four channels (flow_z, flow_y, flow_x, cell) are followed
    in 3D, as Cellpose's do_3D does: each object one id through z."""
    pytest.importorskip("cellpose.dynamics")
    from cellpose import dynamics

    from cellmap_flow.post.postprocessors import CellposeMasksPostprocessor

    zz, yy, xx = np.mgrid[:40, :40, :40]
    labels = np.zeros((40, 40, 40), np.int32)
    labels[(zz - 20) ** 2 + (yy - 12) ** 2 + (xx - 12) ** 2 < 8 ** 2] = 1
    labels[(zz - 20) ** 2 + (yy - 28) ** 2 + (xx - 28) ** 2 < 8 ** 2] = 2
    flows = getattr(dynamics, "masks_to_flows", None) or dynamics.masks_to_flows_gpu
    flow = np.asarray(flows(labels), np.float32) * 5
    probability = np.where(labels > 0, 0.95, 0.05).astype(np.float32)
    masks = CellposeMasksPostprocessor()._process(np.concatenate([flow, probability[np.newaxis]]))
    assert masks.shape == (1, 40, 40, 40) and masks.dtype == np.uint32
    assert len(np.unique(masks)) - 1 == 2
    # One id each through the slices it spans.
    assert len(np.unique(masks[0, 14:27, 12, 12])) == 1 and masks[0, 20, 12, 12] > 0


@pytest.mark.parametrize("best, expected", [("cuda", "cuda"), ("mps", "cpu"), ("cpu", "cpu")])
def test_cellpose_masks_follow_the_flows_on_cuda_or_else_the_cpu(monkeypatch, best, expected):
    """On Apple's GPU each of the flow-following steps waited behind the
    model's forward on another chunk: 40-72 s a chunk instead of about 1."""
    import torch

    from cellmap_flow.post import postprocessors
    from cellmap_flow.utils import device

    monkeypatch.setattr(device, "best_device", lambda: torch.device(best))
    assert postprocessors._masks_device() == torch.device(expected)


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
    assert "Flows" in step.problem_with(CellposeModelConfig(voxel_size=8, output="probability", name="cp"))
    assert "Cellpose" in step.problem_with(ScriptModelConfig(script_path="/s.py", name="mito"))
