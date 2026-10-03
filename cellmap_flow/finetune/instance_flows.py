"""Flow targets from painted instances, and the masked loss a flow model trains on.

A flow model segments instances in 2D by predicting, at every pixel, the
direction towards the centre of the object it belongs to (a Y and an X
flow) and whether it belongs to one at all (a foreground logit); the
instances are recovered by following the flows. Cellpose does this, and so
can any network that predicts those three channels: nothing here is tied to
one.

The annotation is an instance patch in the painted scheme of
``target_transforms`` (0 unannotated, 1 background, each id from 2 one
instance). ``FlowTargetTransform`` makes it a target of three channels,
``[flow_scale * flowY, flow_scale * flowX, foreground]``, slice by slice,
with a mask per channel; ``FlowLoss`` compares a prediction
``[flowY, flowX, foreground logit]`` with it.

Which voxels are supervised:

- the foreground channel: every painted voxel (unpainted ones may be
  either, so they say nothing);
- the flow channels: the painted voxels, less every instance that touches
  the patch's y or x edge. A flow points at the centre of the whole
  object, and an object cut by the patch edge has its centre somewhere
  the patch cannot see: the flow computed from the part inside would be
  confidently wrong.

Instances are taken per slice, as 2D connected components: one 3D id can
cross a slice in two places (a U-shaped object), and to a 2D model those are
two objects, each with its own centre.
"""

from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cellmap_flow.finetune.losses import masked_mean
from cellmap_flow.finetune.target_transforms import TargetTransform

__all__ = [
    "FLOW_SCALE",
    "instance_components",
    "cellpose_flows",
    "flow_targets",
    "FlowTargetTransform",
    "FlowLoss",
]

# Cellpose's networks predict 5x the unit flow (its training loss compares
# the output with 5 * flows), so that is the default scale of the target.
FLOW_SCALE = 5.0

# 8-connectivity: the flow computation diffuses over all eight neighbours,
# so pixels that touch only diagonally are one object to it.
_EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


def instance_components(ids: np.ndarray) -> np.ndarray:
    """The 2D instances of one painted slice: (Y, X) ids -> (Y, X) labels 1..n, 0 elsewhere.

    Every id from 2 up is split into its 8-connected components, and each
    component gets a label of its own, numbered consecutively from 1 (what
    flow computations such as Cellpose's expect). Background (1) and
    unpainted (0) voxels are 0.
    """
    from scipy import ndimage

    ids = np.asarray(ids)
    labels = np.zeros(ids.shape, dtype=np.int32)
    painted = ids >= 2
    if not painted.any():
        return labels
    # Compact the ids first: they can be large (a segmentation's ids + 1),
    # and find_objects lists one entry per value up to the largest.
    compact = np.zeros(ids.shape, dtype=np.int64)
    compact[painted] = np.unique(ids[painted], return_inverse=True)[1].reshape(-1) + 1
    count = 0
    for index, box in enumerate(ndimage.find_objects(compact)):
        if box is None:
            continue
        parts, n = ndimage.label(compact[box] == index + 1, structure=_EIGHT_CONNECTED)
        inside = parts > 0
        labels[box][inside] = parts[inside] + count
        count += n
    return labels


def cellpose_flows(labels: np.ndarray, device=None) -> np.ndarray:
    """Cellpose's unit flows (2, Y, X), Y then X, for labels 1..n (0 none).

    Cellpose's own computation (``dynamics.masks_to_flows_gpu``): heat
    diffused from each object's median pixel, its gradient normalized to
    unit length, zero off the objects. Imported here: Cellpose is only in
    the environments of the models that use it.
    """
    from cellpose import dynamics

    if device is None:
        device = torch.device("cpu")
    compute = getattr(dynamics, "masks_to_flows_gpu", None)
    if compute is None:  # Cellpose 3 named it masks_to_flows
        compute = dynamics.masks_to_flows
    flows = compute(labels.astype(np.int64), device=device)
    if isinstance(flows, tuple):  # (flows, slices) in Cellpose 4
        flows = flows[0]
    return np.asarray(flows, dtype=np.float32)


FlowFunction = Callable[[np.ndarray, Optional[torch.device]], np.ndarray]


def flow_targets(
    annotation: np.ndarray,
    flow_fn: FlowFunction = cellpose_flows,
    flow_scale: float = FLOW_SCALE,
    device=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """The flow target and mask of one painted (Z, Y, X) patch, each (3, Z, Y, X) float32.

    Channels ``[flow_scale * flowY, flow_scale * flowX, foreground]``, the
    flows from ``flow_fn`` on each slice's ``instance_components``. See the
    module docstring for which voxels each channel's mask supervises.
    """
    annotation = np.asarray(annotation)
    target = np.zeros((3, *annotation.shape), dtype=np.float32)
    mask = np.zeros((3, *annotation.shape), dtype=np.float32)
    painted = annotation != 0
    target[2] = annotation >= 2
    mask[2] = painted
    for z, ids in enumerate(annotation):
        labels = instance_components(ids)
        flow_supervised = painted[z].copy()
        if labels.any():
            target[:2, z] = flow_scale * flow_fn(labels, device)
            edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
            edge = edge[edge > 0]
            if edge.size:
                flow_supervised &= ~np.isin(labels, edge)
        mask[:2, z] = flow_supervised
    return target, mask


class FlowTargetTransform(TargetTransform):
    """Painted instances (B, 1, Z, Y, X) as a flow model's target and mask, each (B, 3, Z, Y, X).

    Args:
        flow_fn: ``(labels (Y, X) 1..n, device) -> unit flows (2, Y, X)``;
            Cellpose's (``cellpose_flows``) by default. Another flow model
            may define its flows differently and pass its own.
        flow_scale: what the network's flow output is to the unit flow (5
            for Cellpose).
    """

    def __init__(self, flow_fn: FlowFunction = cellpose_flows, flow_scale: float = FLOW_SCALE):
        self.flow_fn = flow_fn
        self.flow_scale = float(flow_scale)

    def __call__(self, annotation: Tensor) -> Tuple[Tensor, Tensor]:
        ann = annotation.detach().cpu().numpy()
        if ann.ndim != 5 or ann.shape[1] != 1:
            raise ValueError(f"Expected annotation of shape (B, 1, Z, Y, X), got {tuple(ann.shape)}")
        # The flows are diffused on the trainer's device when it is a GPU:
        # on the CPU they take about a second per slice.
        device = annotation.device if annotation.device.type == "cuda" else None
        targets = np.empty((ann.shape[0], 3, *ann.shape[2:]), dtype=np.float32)
        masks = np.empty_like(targets)
        for b in range(ann.shape[0]):
            # The dataset hands the ids over as float32, exact below 2**24.
            targets[b], masks[b] = flow_targets(
                np.rint(ann[b, 0]).astype(np.int64), self.flow_fn, self.flow_scale, device
            )
        return torch.from_numpy(targets).to(annotation.device), torch.from_numpy(masks).to(annotation.device)


class FlowLoss(nn.Module):
    """A flow model's loss: masked MSE of the flows, halved, plus masked BCE of the foreground.

    ``pred`` is (B, 3, Z, Y, X): flowY, flowX and the foreground logit;
    ``target`` and ``mask`` are FlowTargetTransform's. Each term is averaged
    over the voxels its channels' mask supervises, so an unpainted voxel, or
    one of an instance cut by the patch edge, contributes nothing. Unmasked
    it is Cellpose's training loss (``train._loss_fn_seg``).
    """

    def forward(self, pred: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if mask is None:
            mask = torch.ones_like(target)
        flows = masked_mean((pred[:, :2] - target[:, :2]) ** 2, mask[:, :2])
        foreground = masked_mean(
            F.binary_cross_entropy_with_logits(pred[:, 2:3], target[:, 2:3], reduction="none"), mask[:, 2:3]
        )
        return flows / 2.0 + foreground
