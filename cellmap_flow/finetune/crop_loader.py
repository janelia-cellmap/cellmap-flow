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
    - Where a crop sits (:func:`_read_voxel_size_and_offset`): its voxel
      size, the corner of its voxel 0 and the array's sub-path inside a
      group, read by worldarray from the crop's documents (the array itself
      is read with worldarray too, opened by ``io.metadata.open_level``).
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, List, Literal, Optional, Tuple

import numpy as np
import worldarray as wa
import yaml
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

    ``patches_per_epoch=None`` (the default) means "cover every annotated
    chunk roughly once per epoch" — the dataset substitutes the number of
    chunks holding a voxel of either pool at index build time. Override
    with an explicit int to cap the epoch length.

    ``dense_to_sparse_ratio=None`` (the default) means "auto-balance":
    half the patches are centred in the dense pool (foreground inside the
    imported crops) and half in the sparse one (every annotated voxel
    outside them, background included: the painted scribbles) when both
    have voxels, and all in the one that has when only one does (see
    ``finetune.data.sampler``).
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


YAML_SUFFIXES = (".yaml", ".yml")
YAML_SIZE_LIMIT = 1_000_000


class YamlFileRefused(ValueError):
    """A path :func:`read_yaml_file` will not read; the message says why."""


def read_yaml_file(path: str) -> str:
    """Return the text of the YAML file at ``path``, or raise :class:`YamlFileRefused`.

    The dashboard listens on every interface without authentication, and two
    of its routes read a file the request names: read-yaml, for the editor,
    and load-crops. Both used to read any file the dashboard's user could, so
    this reads only a regular file that is YAML by name once symlinks are
    resolved, and at most 1 MB of it.

    A file that is missing, is not YAML, is not a regular file or cannot be
    opened is refused with the same message, so the answer does not tell a
    caller which paths exist either.
    """
    real = os.path.realpath(os.path.expanduser(path))
    refused = YamlFileRefused("Only an existing .yaml or .yml file can be read")
    if not real.lower().endswith(YAML_SUFFIXES) or not os.path.isfile(real):
        raise refused
    if os.path.getsize(real) > YAML_SIZE_LIMIT:
        raise YamlFileRefused("File exceeds 1 MB; paste it directly instead")
    try:
        with open(real) as f:
            return f.read()
    except OSError:
        raise refused from None


def parse_crops_yaml(yaml_text_or_path: str) -> CropsConfig:
    """Parse YAML text, or the path to a YAML file, into a validated config.

    A single line that ends in ``.yaml`` or ``.yml`` is a path, read through
    :func:`read_yaml_file`; anything else is YAML text. Deciding by the name
    rather than by whether the file exists is what keeps load-crops from
    opening arbitrary files, or answering differently for one that exists.
    """
    text = yaml_text_or_path
    if "\n" not in text and text.strip().lower().endswith(YAML_SUFFIXES):
        text = read_yaml_file(text.strip())
    data = yaml.safe_load(text) or {}
    return CropsConfig.model_validate(data)


# ---------------------------------------------------------------------------
# Where a crop sits
# ---------------------------------------------------------------------------


def _read_voxel_size_and_offset(
    zarr_path: str,
) -> Tuple[Tuple[str, ...], np.ndarray, np.ndarray]:
    """Return ``(array_subpath, voxel_size, offset)`` for an annotation zarr.

    ``offset`` is the lower corner of voxel 0; ``array_subpath`` is the path
    components from ``zarr_path`` down to the array (none for an array).
    The placement is read as worldarray reads it (``wa.metadata``, from the
    documents alone, in the file's units, every axis): a group's first level
    (the first its multiscales list, else ``s0``) by the group's multiscales,
    an array by them when its group lists it, else by its own attributes
    (funlib's ``resolution``/``offset``, a COSEM ``transform`` whose
    ``translate`` is voxel 0's centre, the group's funlib attributes when the
    array has none), else voxels of 1 with the corner at 0. Zarr v2 and v3
    alike.
    """
    # Say which path is missing, and how far down it stopped existing. A
    # manifest listing .../rc_amphiuma-means-liver-1/... instead of
    # .../jrc_amphiuma-.../ once got "nothing found at path ''", twice, with
    # nothing to go on.
    if not os.path.exists(zarr_path):
        existing = zarr_path
        while existing and not os.path.exists(existing):
            parent = os.path.dirname(existing)
            if parent == existing:
                existing = ""
                break
            existing = parent
        detail = (
            f" The deepest part that does exist is {existing}."
            if existing
            else " None of that path exists."
        )
        raise FileNotFoundError(f"Crop path not found: {zarr_path}.{detail}")

    node = wa.detect(zarr_path)
    meta = wa.metadata(zarr_path, level=None if node.kind == "array" else 0, units=None, open=False)
    subpath = tuple(meta.path[len(node.path):].strip("/").split("/")) if meta.path != node.path else ()
    return subpath, np.array(meta.voxel_size, dtype=float), np.array(meta.corner, dtype=float)


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
    # The result is uint8. When the instance ids already fit in a byte -- which
    # is every ordinary crop -- build the table as uint8 so the fancy-index
    # pass writes one byte per voxel instead of four and needs no second copy
    # to convert. On the 850x800x600 crop that made this step 11s, that is
    # 408 MB of output instead of 1.6 GB plus a 408 MB astype.
    # uint32 is only needed for the overflow path, where ids exceed 255 before
    # being collapsed.
    fits_uint8 = 2 + len(fg_classes) <= 256
    lookup = np.full(src_max + 1, default, dtype=np.uint8 if fits_uint8 else np.uint32)
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

    if fits_uint8:
        return out

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
