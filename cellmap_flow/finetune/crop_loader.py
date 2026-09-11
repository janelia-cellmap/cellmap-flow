"""
YAML manifest schema and helpers for importing externally annotated crops.

Each manifest entry points at a 3D zarr (typically OME-NGFF) of instance
labels or class IDs. The dashboard's YAML loader (see
``cellmap_flow.dashboard.routes.finetune.yaml_crops``) reads these crops,
remaps their values into the trainer's
``0 = unannotated, 1 = background, >=2 = foreground instance`` convention,
and writes them into the session's annotation_volume.zarr at the crops'
physical offsets.

This module owns:
    - The pydantic schema (:class:`CropEntry`, :class:`CropsConfig`).
    - The label remap function (:func:`remap_labels`).
    - Small zarr-attrs helpers used by the loader to derive a crop's voxel
      size, offset, and the array sub-path inside an OME-NGFF group.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Tuple

import numpy as np
import yaml
import zarr
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# YAML schema
# ---------------------------------------------------------------------------


class CropEntry(BaseModel):
    """One annotation crop to import.

    Fields other than ``path`` are optional with sensible defaults:
        - ``fg_ids=None`` means "every nonzero source value is foreground"
        - ``bg_ids=[]`` means "no explicit BG ids, see mode for what 0 means"
        - ``mode='dense'`` treats unmatched voxels (incl. 0) as background
        - ``mode='sparse'`` treats unmatched voxels as unannotated
        - ``connected_components=False`` keeps source ids as instance ids;
          set True with a single-id ``fg_ids`` to split same-id blobs into
          per-instance ids for affinity-style training.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    name: Optional[str] = None
    fg_ids: Optional[List[int]] = None
    bg_ids: List[int] = Field(default_factory=list)
    mode: Literal["dense", "sparse"] = "dense"
    connected_components: bool = False

    @field_validator("fg_ids", "bg_ids")
    @classmethod
    def _no_zero_ids(cls, value):
        if value is not None and 0 in value:
            raise ValueError("fg_ids/bg_ids cannot include 0 (0 = unannotated sentinel)")
        return value

    @model_validator(mode="after")
    def _validate(self):
        if self.fg_ids is not None and self.bg_ids:
            overlap = set(self.fg_ids) & set(self.bg_ids)
            if overlap:
                raise ValueError(f"fg_ids and bg_ids overlap: {sorted(overlap)}")
        if self.connected_components and self.fg_ids is None:
            raise ValueError("connected_components=True requires fg_ids to be specified")
        return self


class CropsConfig(BaseModel):
    """Top-level YAML schema.

    ``patches_per_epoch``, ``jitter_voxels``, and ``seed`` are passed through
    to the :class:`VirtualPatchDataset` manifest the loader writes — they
    govern epoch length, patch-center jitter (in voxels), and the per-worker
    RNG base seed for reproducible patch sampling across runs.

    ``patches_per_epoch=None`` (the default) means "cover every populated
    chunk roughly once per epoch" — the dataset substitutes the total
    populated-chunk count at index build time. Override with an explicit
    int to cap the epoch length.

    ``dense_to_sparse_ratio=None`` (the default) means "auto-balance":
    50/50 split between dense imported crops and sparse painted scribbles
    when both pools exist; degrades to 1.0 (all from the surviving pool)
    when only one pool has FG voxels.
    """

    model_config = ConfigDict(extra="forbid")

    crops: List[CropEntry]
    patches_per_epoch: Optional[int] = None
    jitter_voxels: Optional[List[int]] = None
    seed: int = 0
    dense_to_sparse_ratio: Optional[float] = None

    @field_validator("crops", mode="before")
    @classmethod
    def _coerce_bare_strings(cls, value):
        if not isinstance(value, list):
            return value
        out = []
        for entry in value:
            if isinstance(entry, str):
                out.append({"path": entry})
            else:
                out.append(entry)
        return out


def parse_crops_yaml(yaml_text_or_path: str) -> CropsConfig:
    """Parse a YAML string OR the path to a YAML file into a validated config."""
    text = yaml_text_or_path
    if "\n" not in yaml_text_or_path and os.path.exists(yaml_text_or_path):
        with open(yaml_text_or_path) as f:
            text = f.read()
    data = yaml.safe_load(text) or {}
    return CropsConfig.model_validate(data)


# ---------------------------------------------------------------------------
# Zarr metadata helpers
# ---------------------------------------------------------------------------


def _read_voxel_size_and_offset(
    zarr_path: str,
) -> Tuple[Tuple[str, ...], np.ndarray, np.ndarray]:
    """Return ``(array_subpath, voxel_size_nm, offset_nm)`` for an annotation zarr.

    Handles three layouts:
        1. Multiscale group with ``multiscales`` -> first scale's array.
        2. Plain ``zarr.Array`` with ``transform``/``resolution`` attrs.
        3. Plain ``zarr.Array`` with no metadata -> voxel_size=(1,1,1),
           offset=(0,0,0).
    """
    node = zarr.open(zarr_path, mode="r")

    if isinstance(node, zarr.hierarchy.Group):
        attrs = dict(node.attrs)
        multiscales = attrs.get("multiscales")
        if multiscales:
            ms = multiscales[0]
            ds = ms["datasets"][0]
            sub = ds["path"]
            scale = np.array([1.0, 1.0, 1.0])
            translation = np.array([0.0, 0.0, 0.0])
            for tx in ds.get("coordinateTransformations", []):
                if tx.get("type") == "scale":
                    scale = np.array(tx["scale"], dtype=float)
                elif tx.get("type") == "translation":
                    translation = np.array(tx["translation"], dtype=float)
            return (sub,), scale, translation
        if "s0" in node:
            return ("s0",), np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0])
        raise ValueError(
            f"Group at {zarr_path} has no 'multiscales' attribute and no 's0' child."
        )

    attrs = dict(node.attrs)
    if "transform" in attrs:
        tx = attrs["transform"]
        scale = np.array(tx.get("scale", [1, 1, 1]), dtype=float)
        translation = np.array(tx.get("translate", [0, 0, 0]), dtype=float)
        return (), scale, translation
    if "resolution" in attrs:
        scale = np.array(attrs["resolution"], dtype=float)
        translation = np.array(attrs.get("offset", [0, 0, 0]), dtype=float)
        return (), scale, translation
    return (), np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0])


def _open_array(zarr_path: str, sub: Tuple[str, ...]) -> zarr.Array:
    target = zarr_path
    for piece in sub:
        target = os.path.join(target, piece)
    arr = zarr.open(target, mode="r")
    if not isinstance(arr, zarr.Array):
        raise ValueError(f"Expected zarr.Array at {target}, got {type(arr).__name__}")
    return arr


# ---------------------------------------------------------------------------
# Label remap
# ---------------------------------------------------------------------------


def remap_labels(
    source: np.ndarray,
    fg_ids: Optional[Iterable[int]],
    bg_ids: Iterable[int],
    mode: Literal["dense", "sparse"],
    connected_components: bool,
) -> np.ndarray:
    """Map source label values to ``0=unannotated, 1=BG, >=2=FG instance``.

    Mapping rules:
        - source value in ``fg_ids`` (or any nonzero if ``fg_ids is None``)
          becomes a unique instance id >=2. If ``connected_components`` is True,
          each connected blob within an fg_id class gets its own instance.
          Otherwise, source ids map to consecutive 2,3,... in order.
        - source value in ``bg_ids`` -> 1 (background).
        - everything else -> 1 if ``mode='dense'``, else 0 (unannotated).

    Returns ``uint8``. If the number of distinct instances would overflow
    ``uint8``, all FG voxels collapse to id=2 and a warning is emitted.
    """
    bg_set = {int(v) for v in bg_ids}

    if connected_components:
        return _remap_with_cc(source, fg_ids, bg_set, mode)

    # Fast path: build a lookup table over source's value range and apply it
    # as a single fancy-index pass. ~10-100x faster than the previous
    # per-class boolean-mask loop on 600^3 arrays.
    try:
        import fastremap

        unique_vals = fastremap.unique(source)
    except Exception:
        unique_vals = np.unique(source)

    if fg_ids is None:
        fg_classes = [int(v) for v in unique_vals if int(v) != 0 and int(v) not in bg_set]
    else:
        # Preserve caller order for deterministic instance ids.
        fg_classes = [int(v) for v in fg_ids]

    src_max = int(unique_vals.max()) if len(unique_vals) else 0
    # Sanity cap: a 32-bit max would blow memory. Real label crops top out in
    # the thousands; bail to the slow per-class path if someone hands us a
    # pathological array.
    if src_max > 8_000_000:
        return _remap_per_class(source, fg_classes, bg_set, mode)

    default = 1 if mode == "dense" else 0
    # uint32 so we can hold instance ids before the uint8 clamp warning fires.
    lookup = np.full(src_max + 1, default, dtype=np.uint32)
    if mode != "dense":
        # sparse: source==0 stays 0 (unannotated)
        lookup[0] = 0
    else:
        lookup[0] = 1
    for bg in bg_set:
        if 0 <= bg <= src_max:
            lookup[bg] = 1
    next_instance_id = 2
    for cls in fg_classes:
        if 0 <= cls <= src_max:
            lookup[cls] = next_instance_id
        next_instance_id += 1

    out = lookup[source]

    if next_instance_id > 256:
        logger.warning(
            f"Crop produced {next_instance_id - 2} instances; collapsing to single FG class "
            "to fit uint8. Affinities between distinct blobs may be inaccurate."
        )
        np.minimum(out, 2, out=out, where=(out >= 2))

    return out.astype(np.uint8)


def _remap_with_cc(source, fg_ids, bg_set, mode):
    """Connected-components path: per-class CC labeling, retained for the
    rare ``connected_components=True`` case. Slower than the lookup-table
    fast path but produces distinct instance ids per blob."""
    from scipy.ndimage import label as cc_label

    if fg_ids is None:
        try:
            import fastremap

            unique_vals = fastremap.unique(source)
        except Exception:
            unique_vals = np.unique(source)
        fg_classes = [int(v) for v in unique_vals if int(v) != 0 and int(v) not in bg_set]
    else:
        fg_classes = [int(v) for v in fg_ids]

    out = np.zeros(source.shape, dtype=np.uint32)
    next_instance_id = 2
    for cls in fg_classes:
        cls_mask = source == cls
        if not cls_mask.any():
            continue
        labeled, n = cc_label(cls_mask)
        for i in range(1, n + 1):
            out[labeled == i] = next_instance_id
            next_instance_id += 1

    fg_set = out >= 2
    if bg_set:
        bg_mask = np.isin(source, list(bg_set))
        out[bg_mask & ~fg_set] = 1
    if mode == "dense":
        out[(out == 0) & ~fg_set] = 1

    if next_instance_id > 256:
        logger.warning(
            f"Crop produced {next_instance_id - 2} instances; collapsing to single FG class "
            "to fit uint8. Affinities between distinct blobs may be inaccurate."
        )
        out[out >= 2] = 2
    return out.astype(np.uint8)


def _remap_per_class(source, fg_classes, bg_set, mode):
    """Fallback for pathologically-large source IDs: original per-class loop."""
    out = np.zeros(source.shape, dtype=np.uint32)
    next_instance_id = 2
    for cls in fg_classes:
        cls_mask = source == cls
        if cls_mask.any():
            out[cls_mask] = next_instance_id
        next_instance_id += 1
    fg_set = out >= 2
    if bg_set:
        bg_mask = np.isin(source, list(bg_set))
        out[bg_mask & ~fg_set] = 1
    if mode == "dense":
        out[(out == 0) & ~fg_set] = 1
    if next_instance_id > 256:
        logger.warning(
            f"Crop produced {next_instance_id - 2} instances; collapsing to single FG class "
            "to fit uint8. Affinities between distinct blobs may be inaccurate."
        )
        out[out >= 2] = 2
    return out.astype(np.uint8)
