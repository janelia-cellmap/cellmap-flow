"""Training a flow model (Cellpose) on painted instances: the per-slice
instances, the flow target and its per-channel masks, the masked flow loss,
how the trainer, the command line and the dashboard pick them, and the
training patch a tile-sized network reads instead of its serving geometry.

The flows are Cellpose's own computation, which only the cellpose4
environment has; everything else runs on a stand-in flow (unit vectors
towards each instance's centroid), and the test of Cellpose's flows skips
where it is missing."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cellmap_flow.finetune.instance_flows import (
    FLOW_SCALE,
    FlowLoss,
    FlowTargetTransform,
    flow_targets,
    instance_components,
)


def centroid_flows(labels, device=None):
    """A stand-in flow: the unit vector from each pixel towards its instance's centroid."""
    flows = np.zeros((2, *labels.shape), dtype=np.float32)
    ys, xs = np.indices(labels.shape)
    for label in range(1, labels.max() + 1):
        inside = labels == label
        dy, dx = ys[inside].mean() - ys[inside], xs[inside].mean() - xs[inside]
        norm = np.maximum(np.hypot(dy, dx), 1e-6)
        flows[0][inside], flows[1][inside] = dy / norm, dx / norm
    return flows


def painted_slice():
    """12 x 12: background painted around an interior instance (id 2), an
    instance cut by the left edge (id 3), and an unpainted corner."""
    ann = np.ones((12, 12), dtype=np.int64)
    ann[4:8, 4:8] = 2
    ann[0:4, 0:3] = 3
    ann[9:, 9:] = 0
    return ann


def test_each_id_in_a_slice_becomes_one_instance_per_connected_piece():
    ids = np.ones((6, 8), dtype=np.int64)
    ids[1:3, 1:3] = 7  # one id, crossing the slice in two places
    ids[1:3, 5:7] = 7
    ids[4, 3] = 1_000_000  # a large id
    ids[0, 7] = 0  # unpainted
    labels = instance_components(ids)
    assert sorted(np.unique(labels)) == [0, 1, 2, 3]
    assert labels[1, 1] != labels[1, 5] and labels[4, 3] not in (0, labels[1, 1], labels[1, 5])
    assert (labels[ids < 2] == 0).all()


def test_pixels_touching_only_diagonally_are_one_instance():
    ids = np.ones((4, 4), dtype=np.int64)
    ids[1, 1] = ids[2, 2] = 2
    labels = instance_components(ids)
    assert labels[1, 1] == labels[2, 2] == 1


def test_the_target_is_scaled_flows_and_foreground():
    ann = painted_slice()[np.newaxis]
    target, mask = flow_targets(ann, centroid_flows)
    labels = instance_components(ann[0])
    assert target.shape == mask.shape == (3, 1, 12, 12)
    assert np.allclose(target[:2, 0], FLOW_SCALE * centroid_flows(labels))
    assert np.array_equal(target[2, 0], (ann[0] >= 2).astype(np.float32))


def test_unpainted_voxels_are_never_supervised():
    ann = painted_slice()[np.newaxis]
    _, mask = flow_targets(ann, centroid_flows)
    assert (mask[:, 0][:, ann[0] == 0] == 0).all()
    assert (mask[2, 0][ann[0] != 0] == 1).all()  # the foreground: every painted voxel


def test_an_instance_cut_by_the_patch_edge_is_left_out_of_the_flows_only():
    """Its centre lies outside the patch, so its flows would point at the wrong place;
    that it is foreground is still known."""
    ann = painted_slice()[np.newaxis]
    _, mask = flow_targets(ann, centroid_flows)
    cut, interior, background = ann[0] == 3, ann[0] == 2, ann[0] == 1
    assert (mask[:2, 0][:, cut] == 0).all() and (mask[2, 0][cut] == 1).all()
    assert (mask[:2, 0][:, interior] == 1).all() and (mask[:2, 0][:, background] == 1).all()


def test_each_slice_has_its_own_instances_and_flows():
    """One 3D id is a different 2D object in every slice it crosses."""
    ann = np.ones((2, 12, 12), dtype=np.int64)
    ann[0, 2:5, 2:5] = 2
    ann[1, 6:10, 6:10] = 2
    seen = []

    def flows(labels, device=None):
        seen.append(labels.copy())
        return centroid_flows(labels)

    target, _ = FlowTargetTransform(flows)(torch.from_numpy(ann).float()[None, None])
    assert target.shape == (1, 3, 2, 12, 12) and len(seen) == 2
    assert [tuple(np.argwhere(s == 1).mean(axis=0)) for s in seen] == [(3.0, 3.0), (7.5, 7.5)]
    # Flows point inwards: right of the centre, x flows to the left.
    assert target[0, 1, 0, 3, 4] < 0 < target[0, 1, 0, 3, 2]


def test_a_slice_without_instances_needs_no_flows():
    def no_call(labels, device=None):
        raise AssertionError("no instance, no flows to compute")

    target, mask = flow_targets(np.ones((1, 4, 4), dtype=np.int64), no_call)
    assert not target.any() and mask.all()


def test_cellposes_own_flows_point_at_each_instances_centre():
    pytest.importorskip("cellpose")
    ann = np.ones((1, 32, 32), dtype=np.int64)
    ann[0, 8:24, 8:24] = 2
    target, mask = FlowTargetTransform()(torch.from_numpy(ann).float()[None, None])
    flow_x = target[0, 1, 0, 16]
    assert flow_x[9] > 0 > flow_x[22]  # left of the centre it points right, and vice versa
    assert float(torch.hypot(target[0, 0, 0, 12, 12], target[0, 1, 0, 12, 12])) == pytest.approx(FLOW_SCALE, rel=1e-3)
    assert mask[0, :, 0].all()


def test_the_flow_loss_unmasked_is_cellposes_loss():
    torch.manual_seed(0)
    pred, target = torch.randn(2, 3, 1, 8, 8), torch.randn(2, 3, 1, 8, 8)
    target[:, 2] = (target[:, 2] > 0).float()
    expected = torch.nn.functional.mse_loss(pred[:, :2], target[:, :2]) / 2 + \
        torch.nn.functional.binary_cross_entropy_with_logits(pred[:, 2], target[:, 2])
    assert torch.allclose(FlowLoss()(pred, target), expected)
    assert torch.allclose(FlowLoss()(pred, target, torch.ones_like(target)), expected)


def test_the_flow_loss_gives_masked_voxels_no_gradient():
    torch.manual_seed(0)
    pred = torch.randn(1, 3, 1, 8, 8, requires_grad=True)
    target = torch.rand(1, 3, 1, 8, 8).round()
    mask = torch.ones(1, 3, 1, 8, 8)
    mask[:, :2, :, :4] = 0  # flows unknown on the left half
    mask[:, 2, :, :, :2] = 0  # nothing painted in the first two columns
    FlowLoss()(pred, target, mask).backward()
    assert (pred.grad[mask == 0] == 0).all() and (pred.grad[mask == 1] != 0).all()


# ---- wiring ---------------------------------------------------------------


def _args(**kw):
    defaults = dict(output_type="flows", loss_type="flow", select_channel=None, label_smoothing=0.0,
                    offsets=None, model_script=None, mask_unannotated=True, distance_sigma=6.0)
    return SimpleNamespace(**{**defaults, **kw})


MODEL = SimpleNamespace(config=SimpleNamespace(output_channels=1, output_voxel_size=(8, 8, 8)))


def test_the_command_line_builds_flow_targets_for_the_flow_loss():
    from cellmap_flow.finetune.cli import build_target_transform, parse_args

    args = _args(label_smoothing=0.1)
    assert isinstance(build_target_transform(args, MODEL), FlowTargetTransform) and args.label_smoothing == 0
    parsed = parse_args(["--corrections", "c", "--output-dir", "o", "--output-type", "flows", "--loss-type", "flow"])
    assert (parsed.output_type, parsed.loss_type) == ("flows", "flow")


@pytest.mark.parametrize("args", [
    pytest.param(_args(loss_type="bce"), id="flows with another loss"),
    pytest.param(_args(output_type="binary"), id="the flow loss on another target"),
    pytest.param(_args(select_channel=0), id="one channel of a flow model"),
])
def test_the_command_line_refuses_flows_half_set_up(args):
    from cellmap_flow.finetune.cli import build_target_transform

    with pytest.raises(ValueError):
        build_target_transform(args, MODEL)


def test_a_restart_may_switch_to_flows():
    from cellmap_flow.finetune.cli import apply_restart_params

    args = SimpleNamespace(output_type="binary", loss_type="bce")
    apply_restart_params(args, {"params": {"output_type": "flows", "loss_type": "flow"}})
    assert (args.output_type, args.loss_type) == ("flows", "flow")


class ThreeChannels(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 3, 1)

    def forward(self, x):
        return self.conv(x)


def test_the_trainer_trains_a_flow_model_on_flow_targets(make_trainer):
    ann = torch.ones(2, 1, 1, 12, 12)
    ann[:, :, :, 4:8, 4:8] = 2
    trainer = make_trainer(ThreeChannels(), (torch.randn(2, 1, 1, 12, 12), ann), loss_type="flow",
                           target_transform=FlowTargetTransform(centroid_flows), label_smoothing=0.1)
    assert isinstance(trainer.criterion, FlowLoss) and trainer.label_smoothing == 0
    before = trainer.model.conv.weight.detach().clone()
    loss = trainer._train_epoch()
    assert np.isfinite(loss) and not torch.equal(before, trainer.model.conv.weight)


def test_the_flow_loss_needs_flow_targets(make_trainer):
    from cellmap_flow.finetune.target_transforms import BinaryTargetTransform

    with pytest.raises(ValueError, match="FlowTargetTransform"):
        make_trainer(ThreeChannels(), (torch.zeros(1, 1, 1, 4, 4), torch.zeros(1, 1, 1, 4, 4)),
                     loss_type="flow", target_transform=BinaryTargetTransform())


def test_the_dashboard_trains_a_cellpose_model_on_flows():
    from cellmap_flow.dashboard.routes.finetune.common import autodetect_output_type, training_settings
    from cellmap_flow.models.configs.finetune import FinetuneModelConfig
    from cellmap_flow.models.models_config import CellposeModelConfig

    cellpose = CellposeModelConfig(voxel_size=8)
    finetuned = FinetuneModelConfig(lora_adapter_path="adapter", base_model=cellpose.to_dict())
    assert autodetect_output_type(cellpose, None, None) == ("flows", None)
    assert autodetect_output_type(finetuned, None, None) == ("flows", None)
    assert autodetect_output_type(cellpose, "binary", None) == ("binary", None)  # asked for, kept
    for sparse in (False, True):
        settings = training_settings(output_type="flows", loss_type="mse", label_smoothing=0.1,
                                     distillation_lambda=0.0, sparse=sparse)
        assert (settings.loss_type, settings.label_smoothing, settings.mask_unannotated) == ("flow", 0.0, sparse)


# ---- the training patch ---------------------------------------------------


def test_a_finetuned_model_trains_on_its_bases_patch():
    from cellmap_flow.finetune.data.loader import training_patch_voxels

    tile = ((1, 4, 4), (1, 4, 4))
    base = SimpleNamespace(training_patch_voxels=lambda: tile)
    assert training_patch_voxels(SimpleNamespace(base_model_config=SimpleNamespace(base_model_config=base))) == tile
    assert training_patch_voxels(SimpleNamespace()) is None


def test_a_training_patch_replaces_the_serving_geometry_and_lengthens_the_epoch(annotation_volume):
    from cellmap_flow.finetune.data import dataset_from_manifest

    labels = np.zeros((32, 32, 32), dtype=np.uint16)
    labels[8:12, 8:12, 8:12] = 2
    labels[8:12, 4:8, 8:12] = 1
    volume = annotation_volume(labels)
    manifest = {"kind": "volume_zarr_v1", "volume_zarr_path": volume.path, "raw_dataset_path": volume.raw,
                "input_size_voxels": [8] * 3, "output_size_voxels": [4] * 3,
                "input_voxel_size_nm": [16.0] * 3, "output_voxel_size_nm": [16.0] * 3, "anchor_fraction": 0.0}
    served = dataset_from_manifest(manifest)
    tiled = dataset_from_manifest(manifest, patch_voxels=((1, 6, 6), (1, 6, 6)))
    raw, ann = tiled[0]
    assert raw.shape == ann.shape == (1, 1, 6, 6) and ann.max() >= 1
    # 4^3 voxels a chunk's patch before, 36 now: two patches cover what one did.
    assert len(tiled) == 2 * len(served)
    assert manifest["input_size_voxels"] == [8] * 3  # the manifest itself is left as it was
