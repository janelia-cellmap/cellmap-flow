"""Annotation volumes: planning, creating, reading and writing crops into them.

An annotation volume is a zarr v2 group ``<id>.zarr`` whose
``annotation/s0`` holds the labels: 0 unannotated, 1 background, 2 and up
foreground (instance ids, for instance corrections). It covers the whole raw
dataset on the grid predictions are made on, with one chunk per model output,
so each chunk is one training sample; only painted or imported chunks exist.
MinIO and neuroglancer write chunks straight into it, so its layout (uint8
unless the labels are instance ids, "." chunk keys, Blosc zstd level 3) and
its root attrs are a format. The root attr ``dataset_offset_nm`` is voxel 0's
*centre*, and also the OME translation: Neuroglancer draws voxel 0 there.
"""

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import zarr

from cellmap_flow.io.multiscale import closest_raw_scale
from cellmap_flow.io.ome import ome_corner, ome_translation

logger = logging.getLogger(__name__)

# The root attrs a record's geometry comes from: record key -> attr.
GEOMETRY_ATTRS = {
    "output_size": "chunk_size",
    "input_size": "input_size",
    "input_voxel_size": "input_voxel_size",
    "output_voxel_size": "output_voxel_size",
}


class NotAnAnnotationVolume(ValueError):
    """``read_volume`` was given a zarr whose root ``type`` is not annotation_volume."""


def _values(x) -> tuple:
    """A sequence as a tuple of plain Python numbers, keeping ints ints."""
    return tuple(x.tolist() if hasattr(x, "tolist") else list(x))


@dataclass(frozen=True)
class VolumeGeometry:
    """Where a volume lies and what model it is for; see ``plan_volume``.

    Sequences become tuples of the numbers given, ints kept ints, since the
    root attrs are written from them as they are.
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

    def record(self, zarr_path, *, dataset_path, model_name, corrections_dir) -> dict:
        """The registry record of a volume at ``zarr_path`` with this geometry."""
        def listed(value):
            return None if value is None else list(value)

        return {
            "zarr_path": zarr_path,
            "model_name": model_name,
            "output_size": list(self.chunk_size),
            "input_size": list(self.input_size),
            "input_voxel_size": list(self.input_voxel_size),
            "output_voxel_size": list(self.output_voxel_size),
            "claimed_input_voxel_size": listed(self.claimed_input_voxel_size),
            "claimed_output_voxel_size": listed(self.claimed_output_voxel_size),
            "dataset_path": dataset_path,
            "dataset_offset_nm": list(self.dataset_offset_nm),
            "corrections_dir": corrections_dir,
        }


def new_volume_id() -> str:
    """``vol-<8 hex>-<YYYYmmdd-HHMMSS>``; the volume's zarr and bucket key are ``<id>.zarr``."""
    return f"vol-{uuid.uuid4().hex[:8]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def volume_corner_nm(dataset_offset_nm, output_voxel_size) -> np.ndarray:
    """The world position of a volume's voxel-0 lower corner, in nm, from its
    ``dataset_offset_nm`` (voxel 0's centre, where Neuroglancer drew it)."""
    offset = np.zeros(3) if dataset_offset_nm is None else dataset_offset_nm
    return np.asarray(ome_corner(offset, output_voxel_size), dtype=float)


def _grid(raw_dataset_path, output_voxel_size, chunk_size, rounding):
    from cellmap_flow.image_data_interface import ImageDataInterface

    output_voxel_size = np.asarray(output_voxel_size, dtype=float)
    chunk_size = np.asarray(chunk_size, dtype=int)
    idi = ImageDataInterface(raw_dataset_path, voxel_size=output_voxel_size)
    offset = np.asarray(ome_translation(np.asarray(idi.offset, dtype=float), output_voxel_size))
    if rounding == "ceil":
        # The data's own extent in output voxels, rounded up. Not idi.roi's:
        # that is the whole-nm box around the data, up to 2 nm larger, which
        # would add a voxel to every level whose extent is not whole nm.
        extent = np.asarray(idi.shape, dtype=float)[-output_voxel_size.size:] * np.asarray(
            idi.voxel_size, dtype=float
        )
        shape = np.ceil(np.round(extent / output_voxel_size, 6)).astype(int)
    elif rounding == "legacy_floor":
        shape = (np.asarray(idi.roi.shape, dtype=float) / output_voxel_size).astype(int)
    else:
        raise ValueError(f"rounding must be 'ceil' or 'legacy_floor', got {rounding!r}")
    return offset, np.ceil(shape / chunk_size).astype(int) * chunk_size


def plan_volume(
    raw_dataset_path: str,
    model_geometry,
    *,
    rounding: Literal["ceil", "legacy_floor"] = "ceil",
) -> VolumeGeometry:
    """Where a new volume over ``raw_dataset_path`` lies, for a model.

    ``model_geometry`` has ``input_voxel_size`` and ``output_voxel_size``,
    and either ``input_shape``/``output_shape`` in voxels or
    ``read_shape``/``write_shape`` in nm, as a server reports them. Each
    voxel size is snapped to the raw level closest to it (the model's own is
    kept as the claimed one); the volume lies on the output level's grid from
    its corner and covers the data in whole chunks of one model output.
    ``rounding="legacy_floor"`` counts the voxels as volumes made before it
    did: rounded down, from the whole-nm box around the data.
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
    """Write an empty volume, metadata only, and return ``zarr_path``.

    ``input_norm`` and ``postprocess`` are the dashboard's chains at the
    time, as YAML-style lists: a resumed session inherits the normalization,
    and a model finetuned on the volume is served with the postprocessing.
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
    annotation.attrs["multiscales"] = [{
        "version": "0.4",
        "name": "annotation",
        "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in ("z", "y", "x")],
        "datasets": [{"path": "s0", "coordinateTransformations": [
            {"type": "scale", "scale": [float(v) for v in geometry.output_voxel_size]},
            {"type": "translation", "translation": [float(o) for o in geometry.dataset_offset_nm]},
        ]}],
    }]

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
        "created_at": datetime.now().isoformat(),
    }
    # The model's own voxel sizes, before snapping to the raw levels.
    for key in ("claimed_output_voxel_size", "claimed_input_voxel_size"):
        if getattr(geometry, key) is not None:
            attrs[key] = list(getattr(geometry, key))
    for key, chain in (("input_norm", input_norm), ("postprocess", postprocess)):
        if chain is not None:
            attrs[key] = chain
    root.attrs.update(attrs)

    logger.info(
        f"Created annotation volume zarr at {zarr_path} "
        f"(shape={list(geometry.dataset_shape_voxels)}, chunks={list(geometry.chunk_size)})"
    )
    return zarr_path


def read_volume(zarr_path: str, *, require_geometry: bool = True) -> dict:
    """The registry record of the annotation volume at ``zarr_path``, from its attrs.

    Nothing missing is guessed (it used to be 56^3 chunks, a 178^3 input and
    16 nm voxels). Raises ValueError naming the geometry attrs the volume
    lacks, unless ``require_geometry`` is False, when they are None: serving
    and syncing need none of them. Raises NotAnAnnotationVolume (a
    ValueError) for any other zarr.
    """
    attrs = dict(zarr.open(zarr_path, mode="r").attrs)
    if attrs.get("type") != "annotation_volume":
        raise NotAnAnnotationVolume(f"{zarr_path} is not an annotation volume (type {attrs.get('type')!r})")
    missing = [attr for attr in GEOMETRY_ATTRS.values() if not attrs.get(attr)]
    if missing and require_geometry:
        raise ValueError(
            f"Annotation volume {zarr_path} has no {', '.join(missing)} in its attrs, "
            "so the geometry to train it with is not known."
        )
    return {
        "zarr_path": zarr_path,
        "model_name": attrs.get("model_name"),
        **{key: attrs.get(attr) for key, attr in GEOMETRY_ATTRS.items()},
        "dataset_path": attrs.get("dataset_path"),
        "dataset_offset_nm": attrs.get("dataset_offset_nm"),
        "corrections_dir": str(Path(zarr_path).parent),
        "chunk_sync_state": {},
    }


def build_manifest(volume_meta: dict, *, input_norm, postprocess, overrides: Optional[dict] = None) -> dict:
    """The ``_virtual_sources.json`` that trains on the volume a record describes.

    The chains travel in it because the trainer runs on LSF, where the
    dashboard's are not: without the normalization it would feed the model
    raw uint8 while inference feeds it [-1, 1]. ``overrides`` replaces any
    key. Raises ValueError naming what the record lacks.
    """
    overrides = overrides or {}
    missing = [key for key in ("zarr_path", *GEOMETRY_ATTRS) if not volume_meta.get(key)]
    if not overrides.get("raw_dataset_path", volume_meta.get("dataset_path")):
        missing.append("dataset_path")
    if missing:
        raise ValueError(
            f"The record of volume {volume_meta.get('zarr_path')} has no {', '.join(missing)}: "
            "a training manifest for it cannot be written."
        )
    return {
        "kind": "volume_zarr_v1",
        "volume_zarr_path": volume_meta["zarr_path"],
        "raw_dataset_path": volume_meta.get("dataset_path"),
        "input_size_voxels": list(volume_meta["input_size"]),
        "output_size_voxels": list(volume_meta["output_size"]),
        "input_voxel_size_nm": list(volume_meta["input_voxel_size"]),
        "output_voxel_size_nm": list(volume_meta["output_voxel_size"]),
        "patches_per_epoch": None,  # one patch per populated chunk
        "jitter_voxels": None,
        "seed": 0,
        "input_norm": input_norm,
        "postprocess": postprocess,
        "dense_to_sparse_ratio": None,  # auto-balance the dense and sparse pools
        **overrides,
    }


def majority_vote_downsample(labels: np.ndarray, factors) -> np.ndarray:
    """Downsample labels by whole per-axis factors, each output voxel taking the
    most common value of its block. Nearest-neighbour zoom takes one fixed
    corner of each block instead (grid_mode=True: the last voxel)."""
    factors = tuple(int(round(f)) for f in factors)
    trimmed_shape = tuple((s // f) * f for s, f in zip(labels.shape, factors))
    block_dims = tuple(s // f for s, f in zip(trimmed_shape, factors))
    blocks = labels[tuple(slice(0, s) for s in trimmed_shape)].reshape(
        block_dims[0], factors[0], block_dims[1], factors[1], block_dims[2], factors[2]
    ).transpose(0, 2, 4, 1, 3, 5).reshape(*block_dims, -1)

    best_count = np.zeros(block_dims, dtype=np.int32)
    result = np.zeros(block_dims, dtype=labels.dtype)
    for val in np.unique(labels):
        count = (blocks == val).sum(axis=-1)
        better = count > best_count
        result[better] = val
        best_count[better] = count[better]
    return result


def write_crop_into_volume(volume_meta: dict, entry, *, progress_callback=None) -> dict:
    """Write a YAML crop's labels into the volume at the crop's physical place.

    The labels are remapped to 1 background and 2+ foreground, resampled to
    the volume's voxel size if it differs, and written into ``annotation/s0``
    in parallel slabs; ``progress_callback(done, total)`` follows the slabs.
    The import is appended to the volume's ``imported_crops`` attr (the
    trainer's dense pool and the overlay's boxes), and returned.
    """
    from cellmap_flow.finetune.crop_loader import _open_array, _read_voxel_size_and_offset, remap_labels
    from cellmap_flow.finetune.session import sync  # sync reads volumes: import here

    started = time.time()
    sub, src_voxel_size_nm, src_offset_nm = _read_voxel_size_and_offset(entry.path)
    src_data = _open_array(entry.path, sub)[:]
    if src_data.ndim != 3:
        raise ValueError(f"Crop {entry.path}: expected 3D (z, y, x), got shape {src_data.shape}")
    remapped = remap_labels(
        src_data, fg_ids=entry.fg_ids, bg_ids=list(entry.bg_ids), mode=entry.mode,
        connected_components=entry.connected_components,
    )

    voxel_size = np.array(volume_meta["output_voxel_size"], dtype=float)
    if not np.allclose(src_voxel_size_nm, voxel_size):
        scale_ratio = src_voxel_size_nm / voxel_size
        factors = voxel_size / src_voxel_size_nm
        logger.info(f"Crop {entry.path} is at {tuple(src_voxel_size_nm)} nm, the volume at "
                    f"{tuple(voxel_size)}: resampling by {tuple(scale_ratio)}.")
        if np.all(scale_ratio <= 1.0) and np.allclose(factors, np.round(factors), atol=1e-6):
            remapped = majority_vote_downsample(remapped, factors)
        else:
            from scipy.ndimage import zoom

            # grid_mode=True aligns the grids' edges, as block voting does;
            # only for ratios block voting can't do.
            remapped = zoom(remapped, scale_ratio, order=0, grid_mode=True, mode="nearest")
    n_fg = int(np.count_nonzero(remapped >= 2))

    # Corner to corner: resampling keeps the crop's lower corner in place.
    z0, y0, x0 = np.round(
        (src_offset_nm - volume_corner_nm(volume_meta["dataset_offset_nm"], voxel_size)) / voxel_size
    ).astype(int).tolist()
    sz, sy, sx = remapped.shape
    arr = zarr.open(volume_meta["zarr_path"], mode="r+")["annotation/s0"]
    if min(z0, y0, x0) < 0 or z0 + sz > arr.shape[0] or y0 + sy > arr.shape[1] or x0 + sx > arr.shape[2]:
        # Usually a crop annotated on another dataset than the session's,
        # consistent in itself: name the dataset, which makes that obvious.
        raise ValueError(
            f"Crop {entry.path} write region "
            f"[{z0}:{z0+sz}, {y0}:{y0+sy}, {x0}:{x0+sx}] is outside the "
            f"annotation volume, whose shape is {tuple(arr.shape)}. This "
            f"volume was built over {volume_meta.get('dataset_path', 'an unknown dataset')}. "
            "Check that the crop was annotated on that same dataset -- a crop "
            "from a different one is the most common cause -- and otherwise "
            "check its OME-NGFF translation against the dataset offset."
        )

    # Z slabs, as many as there are threads to write them, each a whole
    # number of chunks deep.
    chunk_z = max(int(arr.chunks[0]), 1)
    n_slabs = max(1, min(sync.worker_count(), int(np.ceil(sz / chunk_z))))
    slab_size = max(chunk_z, int(np.ceil(sz / n_slabs / chunk_z) * chunk_z))
    slabs = [(a, min(a + slab_size, sz)) for a in range(0, sz, slab_size)]

    def write(slab):
        a, b = slab
        arr[z0 + a : z0 + b, y0 : y0 + sy, x0 : x0 + sx] = remapped[a:b]

    with ThreadPoolExecutor(max_workers=max(1, len(slabs))) as ex:
        for done, fut in enumerate(as_completed([ex.submit(write, s) for s in slabs]), start=1):
            fut.result()
            if progress_callback is not None:
                progress_callback(done, len(slabs))
    logger.info(f"Crop {entry.path}: {n_fg} fg voxels of {src_data.shape} "
                f"written in {len(slabs)} slabs, {time.time() - started:.2f}s")

    record = {
        "path": entry.path,
        "name": entry.name,
        "annotation_offset_voxels": [int(z0), int(y0), int(x0)],
        "annotation_shape_voxels": [int(sz), int(sy), int(sx)],
        "n_fg_voxels": n_fg,
    }
    root = zarr.open(volume_meta["zarr_path"], mode="r+")
    root.attrs["imported_crops"] = list(root.attrs.get("imported_crops", [])) + [record]
    return record
