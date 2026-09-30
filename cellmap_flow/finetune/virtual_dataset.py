"""The finetune dataset's old module; it is ``cellmap_flow.finetune.data`` now.

Kept for scripts that import it by this path (nothing in cellmap_flow does;
see cleanup_review/WRAPPERS.md). ``VirtualPatchDataset`` here is the class
itself, not a subclass, so isinstance checks and pickles agree with
``cellmap_flow.finetune.data``.
"""

from cellmap_flow.finetune.data import (  # noqa: F401
    VirtualPatchDataset,
    create_dataloader,
    dataset_from_manifest,
    make_training_loader,
    rebuild_loader,
)
