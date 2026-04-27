"""
Bulk-load externally annotated zarr crops as finetuning correction entries.

Reads a YAML manifest describing N annotation crops, each pointing at an
existing zarr (e.g. groundtruth labels). For each crop:
  - opens the zarr, derives voxel size / shape / offset from its OME-NGFF
    metadata (or zarr attrs);
  - snaps to the closest scale of the *currently selected raw dataset*;
  - remaps source label values into the trainer's
    ``0 = unannotated, 1 = background, >=2 = foreground instance`` convention
    according to the crop's ``fg_ids`` / ``bg_ids`` / ``mode`` / ``connected_components``
    fields;
  - tiles the crop into ``input_size`` raw + ``output_size`` annotation
    ``_chunk_*.zarr`` entries that match what the dashboard's painted-volume
    pipeline produces, so the existing trainer ingests them with no changes.

Existing entries with the same name are overwritten.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import yaml
import zarr
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cellmap_flow.dashboard.finetune_utils import create_correction_zarr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# YAML schema
# ---------------------------------------------------------------------------


class CropEntry(BaseModel):
    """One annotation crop to ingest.

    Fields other than ``path`` are optional with sensible defaults:
        - ``fg_ids=None`` means "every nonzero source value is foreground"
        - ``bg_ids=[]`` means "no explicit BG ids, see mode for what 0 means"
        - ``mode='dense'`` treats unmatched voxels (incl. 0) as background
        - ``mode='sparse'`` treats unmatched voxels as unannotated
        - ``connected_components=False`` keeps source ids as instance ids
        - ``bg_to_fg_ratio=1.0`` — for each tile with foreground, keep this many
          BG-only tiles (uniformly sampled). 0 = drop all BG-only tiles;
          ``None`` = keep every BG-only tile (the old behavior, can produce
          thousands of background-only chunks for large dense crops).
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    name: Optional[str] = None
    fg_ids: Optional[List[int]] = None
    bg_ids: List[int] = Field(default_factory=list)
    mode: Literal["dense", "sparse"] = "dense"
    connected_components: bool = False
    bg_to_fg_ratio: Optional[float] = 1.0

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
    model_config = ConfigDict(extra="forbid")

    crops: List[CropEntry]

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
    """Parse a YAML string OR a path to a YAML file into a validated ``CropsConfig``."""
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
    """Return (array_subpath, voxel_size_nm, offset_nm) for an annotation zarr.

    Handles three layouts:
        1. Multiscale group with ``multiscales`` -> returns first scale ``s0``.
        2. Plain zarr.Array with ``transform``/``resolution`` attrs.
        3. Plain zarr.Array with no metadata -> voxel_size=(1,1,1), offset=0.
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
        # Group without multiscales: look for an "s0" child
        if "s0" in node:
            return ("s0",), np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0])
        raise ValueError(
            f"Group at {zarr_path} has no 'multiscales' attribute and no 's0' child."
        )

    # zarr.Array
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
# Label remapping
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
          Otherwise, source ids are mapped to consecutive 2,3,... in order.
        - source value in ``bg_ids`` -> 1 (background).
        - everything else -> 1 if ``mode='dense'``, else 0 (unannotated).
    """
    out = np.zeros_like(source, dtype=np.uint32)
    bg_mask = np.isin(source, list(bg_ids)) if len(list(bg_ids)) else np.zeros_like(source, dtype=bool)

    if fg_ids is None:
        fg_classes = sorted(int(v) for v in np.unique(source) if v != 0)
    else:
        fg_classes = list(fg_ids)

    next_instance_id = 2
    for cls in fg_classes:
        cls_mask = source == cls
        if not cls_mask.any():
            continue
        if connected_components:
            from scipy.ndimage import label as cc_label

            labeled, n = cc_label(cls_mask)
            for i in range(1, n + 1):
                out[labeled == i] = next_instance_id
                next_instance_id += 1
        else:
            out[cls_mask] = next_instance_id
            next_instance_id += 1

    # explicit BG marks override fg-on-zero, but FG already-set wins
    fg_set = out >= 2
    out[bg_mask & ~fg_set] = 1

    if mode == "dense":
        unmatched = (out == 0) & ~fg_set
        out[unmatched] = 1
    # sparse: leave as 0 (unannotated)

    if next_instance_id > 256:
        # uint8 chunks downstream — collapse instances if we'd overflow.
        # Preserve same-vs-different semantics by mapping all FG to id=2.
        logger.warning(
            f"Crop produced {next_instance_id - 2} instances; collapsing to single FG class "
            "to fit uint8. Affinities between distinct blobs may be inaccurate."
        )
        out[out >= 2] = 2

    return out.astype(np.uint8)


# ---------------------------------------------------------------------------
# Tiling and chunk creation
# ---------------------------------------------------------------------------


def _iter_tiles(
    annotation_volume_shape: np.ndarray,
    output_size: np.ndarray,
) -> Iterable[Tuple[int, int, int]]:
    """Yield (cz, cy, cx) tile indices that span the annotation volume."""
    nz = int(np.ceil(annotation_volume_shape[0] / output_size[0]))
    ny = int(np.ceil(annotation_volume_shape[1] / output_size[1]))
    nx = int(np.ceil(annotation_volume_shape[2] / output_size[2]))
    for cz in range(nz):
        for cy in range(ny):
            for cx in range(nx):
                yield cz, cy, cx


def load_crops(
    config: CropsConfig,
    raw_dataset_path: str,
    corrections_dir: str,
    input_size: np.ndarray,
    output_size: np.ndarray,
    claimed_input_voxel_size: np.ndarray,
    claimed_output_voxel_size: np.ndarray,
    model_name: str,
    progress_callback=None,
) -> dict:
    """Materialize each crop in ``config`` as ``_chunk_*.zarr`` entries under ``corrections_dir``.

    Voxel sizes are auto-snapped to the closest available raw scale via
    ``get_raw_closest_scale``. This mirrors what the dashboard's painted-volume
    flow does at volume creation time, so externally loaded crops train on the
    same effective scale as painted ones.

    Returns:
        ``{"created": [chunk_paths], "errors": [{path, error}], "skipped": []}``
    """
    from cellmap_flow.image_data_interface import ImageDataInterface
    from cellmap_flow.utils.neuroglancer_utils import get_raw_closest_scale
    from funlib.geometry import Coordinate, Roi

    try:
        eff_input_vs = np.array(
            get_raw_closest_scale(raw_dataset_path, tuple(claimed_input_voxel_size))
            or claimed_input_voxel_size
        )
        eff_output_vs = np.array(
            get_raw_closest_scale(raw_dataset_path, tuple(claimed_output_voxel_size))
            or claimed_output_voxel_size
        )
    except Exception:
        eff_input_vs = np.array(claimed_input_voxel_size)
        eff_output_vs = np.array(claimed_output_voxel_size)

    raw_idi = ImageDataInterface(raw_dataset_path, voxel_size=eff_input_vs)
    raw_dtype = str(raw_idi.ts.dtype)

    created: List[str] = []
    errors: List[dict] = []

    for crop_index, entry in enumerate(config.crops):
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "crop_start",
                    "crop_index": crop_index,
                    "n_crops": len(config.crops),
                    "current_path": entry.path,
                    "tile_done": 0,
                    "tile_total": 0,
                }
            )
        try:
            chunk_paths = _ingest_one_crop(
                entry,
                raw_dataset_path=raw_dataset_path,
                raw_idi=raw_idi,
                raw_dtype=raw_dtype,
                corrections_dir=corrections_dir,
                input_size=input_size,
                output_size=output_size,
                eff_input_vs=eff_input_vs,
                eff_output_vs=eff_output_vs,
                model_name=model_name,
                progress_callback=lambda done, total, p=entry.path, i=crop_index: (
                    progress_callback(
                        {
                            "phase": "tile",
                            "crop_index": i,
                            "n_crops": len(config.crops),
                            "current_path": p,
                            "tile_done": done,
                            "tile_total": total,
                        }
                    )
                    if progress_callback is not None
                    else None
                ),
            )
            created.extend(chunk_paths)
        except Exception as e:
            logger.exception(f"Failed to ingest crop {entry.path}")
            errors.append({"path": entry.path, "error": str(e)})

    if progress_callback is not None:
        progress_callback(
            {
                "phase": "done",
                "crop_index": len(config.crops),
                "n_crops": len(config.crops),
                "current_path": "",
                "tile_done": len(created),
                "tile_total": len(created),
            }
        )

    return {"created": created, "errors": errors}


def _ingest_one_crop(
    entry: CropEntry,
    *,
    raw_dataset_path: str,
    raw_idi,
    raw_dtype: str,
    corrections_dir: str,
    input_size: np.ndarray,
    output_size: np.ndarray,
    eff_input_vs: np.ndarray,
    eff_output_vs: np.ndarray,
    model_name: str,
    progress_callback=None,
) -> List[str]:
    from funlib.geometry import Coordinate, Roi

    sub, src_voxel_size, src_offset_nm = _read_voxel_size_and_offset(entry.path)
    src_arr = _open_array(entry.path, sub)
    src_data = src_arr[:]

    if src_data.ndim != 3:
        raise ValueError(
            f"Annotation array at {entry.path} has shape {src_data.shape}; "
            "expected a 3D (z, y, x) array."
        )

    if not np.allclose(src_voxel_size, eff_output_vs):
        logger.warning(
            f"Crop {entry.path} voxel size {tuple(src_voxel_size)} != "
            f"effective output voxel size {tuple(eff_output_vs)}. "
            "Using source voxel size as-is — the trainer will treat it as if it were the effective size."
        )

    remapped = remap_labels(
        src_data,
        fg_ids=entry.fg_ids,
        bg_ids=entry.bg_ids,
        mode=entry.mode,
        connected_components=entry.connected_components,
    )

    # Pad annotation to a multiple of output_size in each dim
    pad = [
        (0, int(np.ceil(s / o) * o - s))
        for s, o in zip(remapped.shape, output_size)
    ]
    remapped_padded = np.pad(remapped, pad, mode="constant", constant_values=0)

    crop_name = entry.name or _derive_name(entry.path)
    crop_offset_voxels_in_dataset = (src_offset_nm / eff_output_vs).astype(int)

    # Pre-compute the work list of tiles. Two passes:
    #   1) Enumerate tiles, classify as FG (contains any value >= 2),
    #      BG-only (any annotation but no FG), or empty (no annotation;
    #      only happens in sparse mode).
    #   2) Keep all FG tiles. Sample BG-only tiles down to a target ratio
    #      so the model still sees true negatives without the dataset
    #      exploding to thousands of pure-background chunks.
    fg_tasks = []
    bg_tasks = []
    for cz, cy, cx in _iter_tiles(np.array(remapped_padded.shape), output_size):
        z0, y0, x0 = cz * output_size[0], cy * output_size[1], cx * output_size[2]
        z1, y1, x1 = z0 + output_size[0], y0 + output_size[1], x0 + output_size[2]
        ann_tile = remapped_padded[z0:z1, y0:y1, x0:x1]
        if not np.any(ann_tile):
            continue
        task = (cz, cy, cx, ann_tile, np.array([z0, y0, x0]))
        if np.any(ann_tile >= 2):
            fg_tasks.append(task)
        else:
            bg_tasks.append(task)

    if entry.bg_to_fg_ratio is None:
        sampled_bg = bg_tasks
    else:
        budget = int(round(len(fg_tasks) * float(entry.bg_to_fg_ratio)))
        budget = min(budget, len(bg_tasks))
        if budget <= 0:
            sampled_bg = []
        else:
            rng = np.random.default_rng(seed=hash(entry.path) & 0xFFFFFFFF)
            idx = rng.choice(len(bg_tasks), size=budget, replace=False)
            sampled_bg = [bg_tasks[i] for i in idx]

    tasks = fg_tasks + sampled_bg
    total_tiles = sum(
        1 for _ in _iter_tiles(np.array(remapped_padded.shape), output_size)
    )
    logger.info(
        f"Crop {entry.path}: {len(fg_tasks)} FG tiles + {len(sampled_bg)} BG "
        f"(of {len(bg_tasks)} available) = {len(tasks)}/{total_tiles} total; "
        f"loading raw in parallel..."
    )

    chunk_paths: List[str] = []
    chunk_paths_lock = None  # threading.Lock — but list.append is atomic in CPython
    n_workers = min(16, max(4, len(tasks)))
    t0 = time.time()
    completed = 0
    log_every = max(1, len(tasks) // 10)

    def _do_one(task):
        cz, cy, cx, ann_tile, tile_offset_voxels = task
        chunk_offset_nm = src_offset_nm + tile_offset_voxels * eff_output_vs
        chunk_center_nm = chunk_offset_nm + (output_size * eff_output_vs) / 2
        read_shape_nm = input_size * eff_input_vs
        raw_roi = Roi(
            offset=Coordinate(chunk_center_nm - read_shape_nm / 2),
            shape=Coordinate(read_shape_nm),
        )
        raw_data = raw_idi.to_ndarray_ts(raw_roi)

        chunk_id = f"{crop_name}_chunk_{cz}_{cy}_{cx}"
        chunk_zarr_path = os.path.join(corrections_dir, f"{chunk_id}.zarr")
        if os.path.isdir(chunk_zarr_path):
            shutil.rmtree(chunk_zarr_path, ignore_errors=True)

        raw_offset_voxels = (
            (chunk_center_nm - read_shape_nm / 2) / eff_input_vs
        ).astype(int)
        annotation_offset_voxels = (chunk_offset_nm / eff_output_vs).astype(int)

        success, info = create_correction_zarr(
            zarr_path=chunk_zarr_path,
            raw_crop_shape=input_size,
            raw_voxel_size=eff_input_vs,
            raw_offset=raw_offset_voxels,
            annotation_crop_shape=output_size,
            annotation_voxel_size=eff_output_vs,
            annotation_offset=annotation_offset_voxels,
            dataset_path=raw_dataset_path,
            model_name=model_name,
            output_channels=1,
            raw_dtype=raw_dtype,
            create_mask=False,
        )
        if not success:
            raise RuntimeError(f"create_correction_zarr failed for {chunk_id}: {info}")

        z = zarr.open(chunk_zarr_path, mode="r+")
        z["raw/s0"][:] = raw_data
        z["annotation/s0"][:] = ann_tile
        z.attrs["source"] = "yaml_crop"
        z.attrs["source_path"] = entry.path
        z.attrs["crop_name"] = crop_name
        return chunk_zarr_path

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_do_one, t) for t in tasks]
        for fut in as_completed(futures):
            try:
                chunk_paths.append(fut.result())
            except Exception:
                logger.exception(f"Crop {entry.path}: per-tile worker failed")
                raise
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, len(tasks))
            if completed % log_every == 0 or completed == len(tasks):
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 1e-3)
                logger.info(
                    f"Crop {entry.path}: {completed}/{len(tasks)} tiles "
                    f"({elapsed:.1f}s elapsed, {rate:.1f} tiles/s)"
                )

    logger.info(
        f"Crop {entry.path}: finished {len(chunk_paths)} tiles in "
        f"{time.time() - t0:.1f}s"
    )
    return chunk_paths


def _derive_name(path: str) -> str:
    """Derive a unique crop name from a zarr path.

    Walks back from the leaf, collecting up to 4 path components (stripping
    ``.zarr``) and joining with ``_``. Including a few parent dirs avoids
    collisions when many crops share the same leaf (e.g. ``mitochondria.zarr``
    under different ``crop15/`` / ``crop16/`` parents).
    """
    p = Path(path.rstrip("/"))
    parts = []
    for piece in reversed(p.parts):
        if piece in ("", "/"):
            continue
        parts.insert(0, piece.replace(".zarr", ""))
        if len(parts) >= 4:
            break
    return "_".join(parts) or "crop"
