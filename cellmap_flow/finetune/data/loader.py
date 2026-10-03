"""The DataLoaders the finetune trainer reads its patches through.

``create_dataloader`` builds one from a session's corrections directory (its
``_virtual_sources.json`` manifest); ``make_training_loader`` and
``rebuild_loader`` make sure every loader the trainer uses, the OOM
fallback's included, keeps its workers from one epoch to the next.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Sequence, Tuple

import torch

from cellmap_flow.finetune.data.dataset import VirtualPatchDataset
from cellmap_flow.finetune.session.manifest import (
    VIRTUAL_MANIFEST_FILENAME,
    load_good_regions_for,
    read_manifest,
)

logger = logging.getLogger(__name__)


PatchVoxels = Tuple[Sequence[int], Sequence[int]]


def training_patch_voxels(model_config) -> Optional[PatchVoxels]:
    """The (input, output) patch in voxels ``model_config`` trains on, or
    None for the session's (the model's read and write shapes).

    A model type whose network takes a fixed tile says so with a
    ``training_patch_voxels()`` method (Cellpose-SAM: one 256 x 256 slice).
    A finetuned model trains as its base does, so a ``base_model_config`` is
    asked in turn.
    """
    while model_config is not None:
        own = getattr(model_config, "training_patch_voxels", None)
        if callable(own):
            return own()
        model_config = getattr(model_config, "base_model_config", None)
    return None


def dataset_from_manifest(
    manifest: dict,
    corrections_dir: Optional[str] = None,
    augment: Optional[bool] = None,
    patch_voxels: Optional[PatchVoxels] = None,
) -> VirtualPatchDataset:
    """Instantiate a :class:`VirtualPatchDataset` from a manifest dict.

    Recognized manifest kinds:
      - ``volume_zarr_v1`` (current): trainer reads the session's
        annotation_volume.zarr directly. Field ``volume_zarr_path``.

    ``corrections_dir`` is where the session's good regions are looked up;
    omit it and the dataset simply trains without rehearsal anchors.

    ``patch_voxels``, ``(input, output)`` in voxels, replaces the manifest's
    patch sizes, which are the model's serving geometry: for a model whose
    network trains on another (``training_patch_voxels``). The voxel sizes
    stay the manifest's. An epoch then counts as many more patches per
    annotated chunk as the patch is smaller, so that it still covers what
    was annotated.
    """
    kind = manifest.get("kind")
    if kind != "volume_zarr_v1":
        raise ValueError(
            f"Unsupported manifest kind: {kind!r}. Expected 'volume_zarr_v1'."
        )
    input_size = tuple(int(v) for v in manifest["input_size_voxels"])
    output_size = tuple(int(v) for v in manifest["output_size_voxels"])
    patches_per_chunk = 1
    if patch_voxels is not None:
        served_output = output_size
        input_size, output_size = (tuple(int(v) for v in size) for size in patch_voxels)
        patches_per_chunk = max(1, math.ceil(math.prod(served_output) / math.prod(output_size)))
        logger.info(
            f"Training on patches of {input_size} in, {output_size} out (voxels), the model's "
            f"own training patch, not the serving geometry {tuple(manifest['input_size_voxels'])} "
            f"in, {served_output} out; {patches_per_chunk} patch(es) per annotated chunk"
        )
    return VirtualPatchDataset(
        volume_zarr_path=manifest["volume_zarr_path"],
        raw_dataset_path=manifest["raw_dataset_path"],
        input_size_voxels=input_size,
        output_size_voxels=output_size,
        input_voxel_size_nm=tuple(manifest["input_voxel_size_nm"]),
        output_voxel_size_nm=tuple(manifest["output_voxel_size_nm"]),
        # None defaults to "cover all populated chunks" inside the dataset.
        patches_per_epoch=manifest.get("patches_per_epoch"),
        patches_per_chunk=patches_per_chunk,
        jitter_voxels=tuple(manifest["jitter_voxels"]) if manifest.get("jitter_voxels") else None,
        seed=manifest.get("seed", 0),
        input_norm_config=manifest.get("input_norm") or None,
        dense_to_sparse_ratio=manifest.get("dense_to_sparse_ratio"),
        good_regions=load_good_regions_for(corrections_dir),
        rehearsal_fraction=manifest.get("rehearsal_fraction"),
        anchor_fraction=manifest.get("anchor_fraction"),
        resample=bool(manifest.get("resample", False)),
        # Explicit argument wins: at training time the CLI flag is the
        # authority. The manifest value is the session's stored preference.
        augment=(
            bool(manifest.get("augment", False)) if augment is None else bool(augment)
        ),
    )


def create_dataloader(
    corrections_zarr_path: str,
    batch_size: int = 2,
    augment: bool = True,
    num_workers: int = 4,
    patch_voxels: Optional[PatchVoxels] = None,
) -> torch.utils.data.DataLoader:
    """Build the training DataLoader for a corrections directory.

    Requires a ``_virtual_sources.json`` manifest. Every path that creates a
    session writes one (volume creation, YAML import, training submit) and
    restarts backfill one, so a missing manifest means something upstream
    failed -- which is worth an exception rather than training on anything
    else. The patch geometry comes from the manifest, and the dataset
    samples randomly, so there is no patch shape or shuffle to pass.

    Args:
        corrections_zarr_path: Session corrections directory.
        batch_size: Clamped down to the dataset size when smaller.
        augment: Flips, XY rotations, brightness and noise, on top of the
            patch-centre jitter that is always applied. It overrides the
            manifest's stored preference.
        num_workers: DataLoader workers. Spawned, not forked -- tensorstore
            handles do not survive fork.
        patch_voxels: ``(input, output)`` patch in voxels instead of the
            manifest's, for a model that trains on its own
            (``training_patch_voxels``).
    """
    manifest = read_manifest(corrections_zarr_path)
    if manifest is None:
        raise FileNotFoundError(
            f"No {VIRTUAL_MANIFEST_FILENAME} in {corrections_zarr_path}. "
            "The trainer reads annotations through a virtual-sources manifest; "
            "without one there is nothing to train on. Re-create the session, "
            "or re-import the crops, so the manifest gets written."
        )

    dataset = dataset_from_manifest(
        manifest, corrections_zarr_path, augment=augment, patch_voxels=patch_voxels
    )

    if dataset.augment:
        logger.info(
            "Augmentation ON: random Z/Y/X flips, XY rotations where the YX "
            "plane is square, brightness x0.8-x1.2 and 1%-of-range noise, on "
            f"top of patch-center jitter (jitter={dataset.sampler.jitter.tolist()}). "
            "Worth it when the run revisits the same patches many times; at a "
            "few dozen gradient steps it mostly just adds variance."
        )
    else:
        logger.info(
            "Augmentation OFF: patch-center jitter only "
            f"(jitter={dataset.sampler.jitter.tolist()})."
        )

    actual_batch_size = max(1, min(batch_size, len(dataset)))
    if actual_batch_size != batch_size:
        logger.info(
            f"Clamped batch_size from {batch_size} to {actual_batch_size} "
            f"({len(dataset)} patches per epoch available)"
        )

    logger.info(
        f"Created DataLoader with {len(dataset)} patches/epoch, "
        f"batch_size={actual_batch_size}, num_workers={num_workers}"
    )

    return make_training_loader(dataset, actual_batch_size, num_workers)


def make_training_loader(dataset, batch_size: int, num_workers: int) -> torch.utils.data.DataLoader:
    """The DataLoader the trainer reads patches through.

    Persistent, spawned workers: each keeps its copy of the dataset, and so its
    RNG, from one epoch to the next. Non-persistent workers are re-spawned
    every epoch from a fresh pickle of the dataset -- RNG unset, same seed --
    so every epoch would draw the identical patches and augmentations.
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # the dataset samples randomly already
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )


def rebuild_loader(loader: torch.utils.data.DataLoader, batch_size: int) -> torch.utils.data.DataLoader:
    """``loader`` again with a different batch size and everything else kept.

    The OOM fallback in the trainer used to rebuild its loader with only some
    of the original arguments; losing persistent_workers made every epoch
    after an OOM repeat the same patches (see make_training_loader).
    """
    from torch.utils.data import RandomSampler

    kwargs = dict(
        batch_size=batch_size,
        shuffle=isinstance(loader.sampler, RandomSampler),
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        timeout=loader.timeout,
        worker_init_fn=loader.worker_init_fn,
        multiprocessing_context=loader.multiprocessing_context,
        generator=loader.generator,
        persistent_workers=loader.persistent_workers,
    )
    if loader.num_workers > 0:
        kwargs["prefetch_factor"] = loader.prefetch_factor
    return torch.utils.data.DataLoader(loader.dataset, **kwargs)
