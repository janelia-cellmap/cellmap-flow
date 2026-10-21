"""Training a flow model (Cellpose) on painted instances: the per-slice
instances, the flow target and its per-channel masks, the masked flow loss,
and how the trainer and the dashboard pick them.

The flows are Cellpose's own computation, which only the cellpose4
environment has; everything else runs on a stand-in flow (unit vectors
towards each instance's centroid), and the test of Cellpose's flows skips
where it is missing."""

import numpy as np
import pytest
import torch

from cellmap_flow.finetune.instance_flows import (
    FLOW_SCALE,
    FlowLoss,
    FlowTargetTransform,
    flow_targets,
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


def test_cellposes_own_flows_point_at_each_instances_centre():
    pytest.importorskip("cellpose")
    ann = np.ones((1, 32, 32), dtype=np.int64)
    ann[0, 8:24, 8:24] = 2
    target, mask = FlowTargetTransform()(torch.from_numpy(ann).float()[None, None])
    flow_x = target[0, 1, 0, 16]
    assert flow_x[9] > 0 > flow_x[22]  # left of the centre it points right, and vice versa
    assert float(torch.hypot(target[0, 0, 0, 12, 12], target[0, 1, 0, 12, 12])) == pytest.approx(FLOW_SCALE, rel=1e-3)
    assert mask[0, :, 0].all()


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
