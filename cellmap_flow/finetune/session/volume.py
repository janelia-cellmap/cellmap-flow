"""Annotation volumes: planning, creating, reading and writing crops into them.

An annotation volume is a zarr v2 group ``<id>.zarr`` whose
``annotation/s0`` holds the labels: 0 unannotated, 1 background, 2 and up
foreground (instance ids, for instance corrections). It covers the whole raw
dataset on the grid predictions are made on, with one chunk per model output,
so each chunk is one training sample. Only painted or imported chunks exist
on disk. MinIO and neuroglancer write chunks straight into it, so its layout
-- uint8 unless the labels are instance ids, "." separated chunk keys, Blosc
zstd level 3 -- and its root attrs are a format.

The root attr ``dataset_offset_nm`` is voxel 0's *centre* and is also the
OME translation, since that is where Neuroglancer draws voxel 0 while it is
painted; ``volume_corner_nm`` gives the corner.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import zarr

from cellmap_flow.finetune.session import sync
from cellmap_flow.io.multiscale import closest_raw_scale
from cellmap_flow.io.ome import ome_corner, ome_translation

logger = logging.getLogger(__name__)


class NotAnAnnotationVolume(ValueError):
    """``read_volume`` was given a zarr whose root ``type`` is not annotation_volume."""


def _values(x) -> tuple:
    """A sequence as a tuple of plain Python numbers, keeping ints ints."""
    return tuple(x.tolist() if hasattr(x, "tolist") else list(x))


@dataclass(frozen=True)
class VolumeGeometry:
    """Where a volume lies and what model it is for; see ``plan_volume``.

    Sequences are kept as tuples of the numbers given, ints as ints, since
    the root attrs are written from them as they are.
    """

    output_voxel_size: tuple
    input_voxel_size: tuple
    claimed_output_voxel_size: Optional[tuple]
    claimed_input_voxel_size: Optional[tuple]
    chunk_size: tuple
    input_size: tuple
    dataset_offset_nm: tuple  # voxel 0's centre == the OME translation
    dataset_shape_voxels: tuple

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                object.__setattr__(self, field.name, _values(value))


def volume_corner_nm(dataset_offset_nm, output_voxel_size) -> np.ndarray:
    """The world position of an annotation volume's voxel-0 lower corner, in nm.

    ``dataset_offset_nm`` (a root attr of every volume) is also written as the
    volume's OME-NGFF translation, and a translation is voxel 0's *centre*.
    Neuroglancer drew the volume that way while it was painted, so that is
    where the labels are. Reading the value as a corner, as this code used to,
    put every label half an annotation voxel away from where it was drawn.
    """
    offset = np.zeros(3) if dataset_offset_nm is None else dataset_offset_nm
    return np.asarray(ome_corner(offset, output_voxel_size), dtype=float)


def _grid(raw_dataset_path, output_voxel_size, chunk_size, rounding):
    output_voxel_size = np.asarray(output_voxel_size, dtype=float)
    chunk_size = np.asarray(chunk_size, dtype=int)
    from cellmap_flow.image_data_interface import ImageDataInterface

    idi = ImageDataInterface(raw_dataset_path, voxel_size=output_voxel_size)
    offset = np.asarray(ome_translation(np.asarray(idi.offset, dtype=float), output_voxel_size))
    if rounding == "ceil":
        # The data's own extent in output voxels, rounded up so a partial
        # voxel at the far end is covered. Not idi.roi's: that is the
        # whole-nm box around the data, up to 2 nm larger, which would add
        # a voxel to every level whose extent is not whole nm (10.48 nm...).
        extent = np.asarray(idi.shape, dtype=float)[-output_voxel_size.size:] * np.asarray(
            idi.voxel_size, dtype=float
        )
        shape = np.ceil(np.round(extent / output_voxel_size, 6)).astype(int)
    elif rounding == "legacy_floor":
        shape = (np.asarray(idi.roi.shape, dtype=float) / output_voxel_size).astype(int)
    else:
        raise ValueError(f"rounding must be 'ceil' or 'legacy_floor', got {rounding!r}")
    return offset, np.ceil(shape / chunk_size).astype(int) * chunk_size


def new_volume_geometry(raw_dataset_path: str, output_voxel_size, chunk_size):
    """``(dataset_offset_nm, shape_voxels)`` for a new volume over a raw dataset.

    The volume lies on the grid of the raw level at ``output_voxel_size`` (the
    grid predictions are made on), from that level's corner, padded to whole
    chunks. ``dataset_offset_nm`` is voxel 0's centre; see volume_corner_nm.
    """
    return _grid(raw_dataset_path, output_voxel_size, chunk_size, "ceil")


def plan_volume(
    raw_dataset_path: str,
    model_geometry,
    *,
    rounding: Literal["ceil", "legacy_floor"] = "ceil",
) -> VolumeGeometry:
    """Where a new volume over ``raw_dataset_path`` lies, for a model.

    ``model_geometry`` has ``input_voxel_size`` and ``output_voxel_size``,
    and either ``input_shape``/``output_shape`` in voxels or
    ``read_shape``/``write_shape`` in nm, which is what a server reports.
    Each voxel size is snapped to the raw level closest to it (the model's
    own is kept as the claimed one); the volume lies on the output level's
    grid from its corner, one chunk per model output, and covers the data
    padded to whole chunks. ``rounding="legacy_floor"`` counts the voxels
    as volumes made before this did: rounded down, from the whole-nm box
    around the data.
    """
    claimed_in = np.array(model_geometry.input_voxel_size)
    claimed_out = np.array(model_geometry.output_voxel_size)
    if getattr(model_geometry, "input_shape", None) is not None:
        input_size = np.array(model_geometry.input_shape, dtype=int)
        output_size = np.array(model_geometry.output_shape, dtype=int)
    else:
        input_size = (np.array(model_geometry.read_shape) / claimed_in).astype(int)
        output_size = (np.array(model_geometry.write_shape) / claimed_out).astype(int)
    output_voxel_size = np.array(closest_raw_scale(raw_dataset_path, tuple(claimed_out)) or claimed_out)
    input_voxel_size = np.array(closest_raw_scale(raw_dataset_path, tuple(claimed_in)) or claimed_in)
    offset, shape = _grid(raw_dataset_path, output_voxel_size, output_size, rounding)
    return VolumeGeometry(
        output_voxel_size=output_voxel_size,
        input_voxel_size=input_voxel_size,
        claimed_output_voxel_size=claimed_out,
        claimed_input_voxel_size=claimed_in,
        chunk_size=output_size,
        input_size=input_size,
        dataset_offset_nm=offset,
        dataset_shape_voxels=shape,
    )


def create_volume_zarr(
    zarr_path: str,
    geometry: VolumeGeometry,
    *,
    dataset_path: str,
    model_name: str,
    input_norm=None,
    postprocess=None,
    annotation_dtype="uint8",
    annotation_type="annotation_volume",
) -> str:
    """Write an empty volume: its metadata only, no chunks. Returns ``zarr_path``.

    ``input_norm`` and ``postprocess`` are the dashboard's chains when the
    volume was made: a resumed session inherits the normalization, and a
    finetuned model served from it needs the postprocessing. ``annotation_dtype``
    is uint16 or uint32 when the labels are instance ids.
    """
    root = zarr.open(zarr_path, mode="w")
    annotation = root.create_group("annotation")
    annotation.create_dataset(
        "s0",
        shape=geometry.dataset_shape_voxels,
        chunks=geometry.chunk_size,
        dtype=annotation_dtype,
        compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.SHUFFLE),
        fill_value=0,
    )
    # dataset_offset_nm is voxel 0's centre, so it is the OME translation as
    # it stands (see volume_corner_nm).
    annotation.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "annotation",
            "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in ("z", "y", "x")],
            "datasets": [
                {
                    "path": "s0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [float(v) for v in geometry.output_voxel_size]},
                        {"type": "translation", "translation": [float(o) for o in geometry.dataset_offset_nm]},
                    ],
                }
            ],
        }
    ]

    attrs = {
        "type": annotation_type,
        "model_name": model_name,
        "dataset_path": dataset_path,
        "chunk_size": list(geometry.chunk_size),
        "output_voxel_size": list(geometry.output_voxel_size),
        "input_size": list(geometry.input_size),
        "input_voxel_size": list(geometry.input_voxel_size),
        "dataset_offset_nm": list(geometry.dataset_offset_nm),
        "dataset_shape_voxels": list(geometry.dataset_shape_voxels),
    }
    # The model's own voxel sizes, for provenance: the ones above are the raw
    # levels closest to them.
    if geometry.claimed_output_voxel_size is not None:
        attrs["claimed_output_voxel_size"] = list(geometry.claimed_output_voxel_size)
    if geometry.claimed_input_voxel_size is not None:
        attrs["claimed_input_voxel_size"] = list(geometry.claimed_input_voxel_size)
    # Stored as the YAML-style chains, so they round-trip through json and yaml.
    if input_norm is not None:
        attrs["input_norm"] = input_norm
    if postprocess is not None:
        attrs["postprocess"] = postprocess
    attrs["created_at"] = datetime.now().isoformat()
    root.attrs.update(attrs)

    logger.info(
        f"Created annotation volume zarr at {zarr_path} "
        f"(shape={list(geometry.dataset_shape_voxels)}, chunks={list(geometry.chunk_size)})"
    )
    return zarr_path


def read_volume(zarr_path: str) -> dict:
    """The registry record of the annotation volume at ``zarr_path``, from its attrs.

    Raises NotAnAnnotationVolume (a ValueError) if it is not one.
    """
    attrs = dict(zarr.open(zarr_path, mode="r").attrs)
    if attrs.get("type") != "annotation_volume":
        raise NotAnAnnotationVolume(f"{zarr_path} is not an annotation volume (type {attrs.get('type')!r})")
    return {
        "zarr_path": zarr_path,
        "model_name": attrs.get("model_name", ""),
        "output_size": attrs.get("chunk_size", [56, 56, 56]),
        "input_size": attrs.get("input_size", [178, 178, 178]),
        "input_voxel_size": attrs.get("input_voxel_size", [16, 16, 16]),
        "output_voxel_size": attrs.get("output_voxel_size", [16, 16, 16]),
        "dataset_path": attrs.get("dataset_path", ""),
        "dataset_offset_nm": attrs.get("dataset_offset_nm", [0, 0, 0]),
        "corrections_dir": str(Path(zarr_path).parent),
        "chunk_sync_state": {},
    }


def build_manifest(volume_meta: dict, *, input_norm, postprocess, overrides: Optional[dict] = None) -> dict:
    """The ``_virtual_sources.json`` that trains on the volume a record describes.

    ``input_norm`` and ``postprocess`` travel in it because the trainer runs
    on LSF, where the dashboard's chains are not: without the normalization
    it feeds the model raw uint8 while inference feeds it [-1, 1].
    ``overrides`` replaces any key, e.g. a crops manifest's patches_per_epoch.
    """
    manifest = {
        "kind": "volume_zarr_v1",
        "volume_zarr_path": volume_meta["zarr_path"],
        "raw_dataset_path": volume_meta.get("dataset_path"),
        "input_size_voxels": list(volume_meta["input_size"]),
        "output_size_voxels": list(volume_meta["output_size"]),
        "input_voxel_size_nm": list(volume_meta["input_voxel_size"]),
        "output_voxel_size_nm": list(volume_meta["output_voxel_size"]),
        # None means "one patch per populated chunk": full coverage of what
        # was annotated, rather than a fixed count.
        "patches_per_epoch": None,
        "jitter_voxels": None,
        "seed": 0,
        "input_norm": input_norm,
        "postprocess": postprocess,
        # None: auto-balance the dense and sparse pools.
        "dense_to_sparse_ratio": None,
    }
    manifest.update(overrides or {})
    return manifest


def majority_vote_downsample(labels: np.ndarray, factors) -> np.ndarray:
    """Downsample integer label data by exact per-axis block factors using
    majority vote (mode) over each block.

    Unlike single-point nearest-neighbor sampling (which always picks one
    fixed corner of each block, e.g. scipy.ndimage.zoom's grid_mode=True
    deterministically picks the block's *last* voxel on every axis), this
    represents each output voxel by the value most common across its whole
    footprint -- no systematic corner-bias, and fewer boundary voxels
    flipped by picking an unrepresentative single sample.
    """
    factors = tuple(int(round(f)) for f in factors)
    shape = labels.shape
    trimmed_shape = tuple((s // f) * f for s, f in zip(shape, factors))
    trimmed = labels[tuple(slice(0, s) for s in trimmed_shape)]
    block_dims = tuple(s // f for s, f in zip(trimmed_shape, factors))
    reshaped = trimmed.reshape(
        block_dims[0], factors[0], block_dims[1], factors[1], block_dims[2], factors[2]
    )
    reshaped = reshaped.transpose(0, 2, 4, 1, 3, 5)
    flat_blocks = reshaped.reshape(block_dims[0], block_dims[1], block_dims[2], -1)

    best_count = np.zeros(block_dims, dtype=np.int32)
    result = np.zeros(block_dims, dtype=labels.dtype)
    for val in np.unique(labels):
        count = (flat_blocks == val).sum(axis=-1)
        better = count > best_count
        result[better] = val
        best_count[better] = count[better]
    return result


def write_crop_into_volume(volume_meta: dict, entry, *, progress_callback=None) -> dict:
    """Write a YAML crop's labels into the volume at the crop's physical place.

    The crop is read, its labels remapped to 1 = background and 2+ =
    foreground, resampled to the volume's voxel size if it differs, and
    written into ``annotation/s0``. The import is recorded in the volume's
    ``imported_crops`` attr, which is also what this returns.
    """
    from cellmap_flow.finetune.crop_loader import (
        _open_array,
        _read_voxel_size_and_offset,
        remap_labels,
    )

    t0 = time.time()
    sub, src_voxel_size_nm, src_offset_nm = _read_voxel_size_and_offset(entry.path)
    t_meta = time.time() - t0
    t1 = time.time()
    src_arr = _open_array(entry.path, sub)
    src_data = src_arr[:]
    t_read = time.time() - t1
    if src_data.ndim != 3:
        raise ValueError(
            f"Crop {entry.path}: expected 3D (z, y, x), got shape {src_data.shape}"
        )

    eff_output_vs = np.array(volume_meta["output_voxel_size"], dtype=float)

    t2 = time.time()
    remapped = remap_labels(
        src_data,
        fg_ids=entry.fg_ids,
        bg_ids=list(entry.bg_ids),
        mode=entry.mode,
        connected_components=entry.connected_components,
    )
    t_remap = time.time() - t2

    if not np.allclose(src_voxel_size_nm, eff_output_vs):
        scale_ratio = src_voxel_size_nm / eff_output_vs
        logger.info(
            f"Crop {entry.path} voxel size {tuple(src_voxel_size_nm)} != "
            f"volume voxel size {tuple(eff_output_vs)}. Resampling by "
            f"{tuple(scale_ratio)} before writing so the written data "
            "occupies its true physical extent."
        )

        integer_factors = eff_output_vs / src_voxel_size_nm
        if np.all(scale_ratio <= 1.0) and np.allclose(
            integer_factors, np.round(integer_factors), atol=1e-6
        ):
            # Exact integer downsample: majority-vote (mode) over each
            # block, rather than picking one arbitrary corner sample.
            remapped = majority_vote_downsample(remapped, integer_factors)
        else:
            from scipy.ndimage import zoom

            # grid_mode=True aligns to pixel *centers* rather than the
            # default's array-endpoint alignment (wrong, and increasingly
            # so toward the edges) -- but it still samples a single fixed
            # corner of each block, used here only as a fallback for
            # non-integer ratios / upsampling where block-voting doesn't
            # apply.
            remapped = zoom(remapped, scale_ratio, order=0, grid_mode=True, mode="nearest")

    t3 = time.time()
    n_fg = int(np.count_nonzero(remapped >= 2))
    t_count = time.time() - t3
    logger.info(
        f"Crop {entry.path} prep: meta={t_meta:.2f}s read={t_read:.2f}s "
        f"({src_data.nbytes/1e6:.1f} MB, dtype={src_data.dtype}, shape={src_data.shape}) "
        f"remap={t_remap:.2f}s count_fg={t_count:.2f}s"
    )

    # Corner to corner: resampling keeps the crop's lower corner where it
    # was (block voting and grid_mode zoom both align the grids' edges), so
    # no per-factor shift is needed. Comparing centres needed one, and the
    # fixed +fine/2 used for it was only right for a factor of 2.
    volume_corner = volume_corner_nm(volume_meta["dataset_offset_nm"], eff_output_vs)
    write_voxel_offset = np.round(
        (src_offset_nm - volume_corner) / eff_output_vs
    ).astype(int)
    z0, y0, x0 = write_voxel_offset.tolist()
    sz, sy, sx = remapped.shape

    vol = zarr.open(volume_meta["zarr_path"], mode="r+")
    arr = vol["annotation/s0"]
    if (
        z0 < 0 or y0 < 0 or x0 < 0
        or z0 + sz > arr.shape[0]
        or y0 + sy > arr.shape[1]
        or x0 + sx > arr.shape[2]
    ):
        # The usual cause is not a bad translation but a crop belonging to a
        # different dataset than the session: a crop annotated on a larger
        # volume lands past the end of a smaller one, with everything about
        # it internally consistent. Name the dataset this volume was built
        # over so that is the first thing checked, since the path in the
        # manifest often makes the mismatch obvious once it is put next to it.
        raise ValueError(
            f"Crop {entry.path} write region "
            f"[{z0}:{z0+sz}, {y0}:{y0+sy}, {x0}:{x0+sx}] is outside the "
            f"annotation volume, whose shape is {tuple(arr.shape)}. This "
            f"volume was built over {volume_meta.get('dataset_path', 'an unknown dataset')}. "
            "Check that the crop was annotated on that same dataset -- a crop "
            "from a different one is the most common cause -- and otherwise "
            "check its OME-NGFF translation against the dataset offset."
        )

    # Slice the crop into Z-aligned slabs and write them in parallel. Slabs
    # are aligned to the underlying zarr chunk size so two slabs never
    # touch the same chunk, making concurrent writes safe (zarr's chunk
    # writes are per-chunk-file, no shared mutable state).
    #
    # Slab count tracks the LSF slot allocation so we always fully use what
    # bsub gave us — capped by the number of chunk-aligned slabs we can
    # actually produce.
    chunk_z = max(int(arr.chunks[0]), 1)
    max_chunk_slabs = int(np.ceil(sz / chunk_z))
    n_slabs = max(1, min(sync.worker_count(), max_chunk_slabs))
    slab_size = int(np.ceil(sz / n_slabs / chunk_z) * chunk_z)
    slabs = []
    for s in range(n_slabs):
        a = s * slab_size
        b = min((s + 1) * slab_size, sz)
        if a < b:
            slabs.append((a, b))
    n_slabs = len(slabs)

    def _write_one(slab):
        a, b = slab
        arr[z0 + a : z0 + b, y0 : y0 + sy, x0 : x0 + sx] = remapped[a:b, :, :]

    t4 = time.time()
    written = 0
    n_workers = max(1, n_slabs)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_write_one, s) for s in slabs]
        for fut in as_completed(futures):
            fut.result()  # surface any per-slab exception
            written += 1
            if progress_callback is not None:
                progress_callback(written, n_slabs)
    t_write = time.time() - t4
    logger.info(
        f"Crop {entry.path} write: {n_slabs} slabs, {n_workers} workers, "
        f"{t_write:.2f}s total wall"
    )

    # Recorded in the volume's root attrs: the trainer's dense pool and the
    # overlay's one box per import come from here.
    record = {
        "path": entry.path,
        "name": entry.name,
        "annotation_offset_voxels": [int(z0), int(y0), int(x0)],
        "annotation_shape_voxels": [int(sz), int(sy), int(sx)],
        "n_fg_voxels": int(n_fg),
    }
    vol_root = zarr.open(volume_meta["zarr_path"], mode="r+")
    vol_root.attrs["imported_crops"] = list(vol_root.attrs.get("imported_crops", [])) + [record]
    return record
