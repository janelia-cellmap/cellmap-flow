"""What the finetune trainer trains on: random patch pairs read out of a
session's annotation volume, and the loaders that deliver them.

- ``dataset``: ``VirtualPatchDataset``, the patches. Each is a centre from
  ``sampler`` (the dense, sparse and rehearsal pools), the annotation and raw
  that ``reader`` reads around it, and ``augment``'s flips, rotations,
  brightness and noise when asked for.
- ``loader``: ``dataset_from_manifest`` and ``create_dataloader`` (from a
  session's manifest), ``make_training_loader`` and ``rebuild_loader``.
"""

from cellmap_flow.finetune.data.dataset import VirtualPatchDataset
from cellmap_flow.finetune.data.loader import (
    create_dataloader,
    dataset_from_manifest,
    make_training_loader,
    rebuild_loader,
)

__all__ = [
    "VirtualPatchDataset",
    "create_dataloader",
    "dataset_from_manifest",
    "make_training_loader",
    "rebuild_loader",
]
