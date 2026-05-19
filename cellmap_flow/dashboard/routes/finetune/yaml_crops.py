"""Endpoint for bulk-loading externally annotated crops via a YAML manifest.

Design
------
A YAML manifest is conceptually a different way to **seed an annotation
volume**, alongside "New Volume" (empty) and "Resume Existing Volume"
(copy a prior session). Importing crops writes them straight into the
session's ``annotation_volume.zarr`` at their correct physical offsets, so
the result is identical in shape to a painted volume — one editable layer
in neuroglancer, served via MinIO, picked up by the existing periodic-sync
machinery, and consumed by training via :class:`VirtualPatchDataset`.

Painted scribbles + imported GT crops therefore share one source of truth
(the volume zarr). The user can paint over imports to fix GT errors or to
add corrections in regions the GT doesn't cover. The trainer sees the
union by construction.
"""

import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import zarr
from flask import jsonify, request
from pydantic import ValidationError

# Module-level progress tracker, keyed by load_id supplied by the client.
# Each value is the most recent progress snapshot for that load + its
# final result (or None while in progress). Old entries are evicted after
# 5 minutes to bound memory.
_PROGRESS: dict = {}
_PROGRESS_LOCK = threading.Lock()
_PROGRESS_TTL_SECONDS = 300


def _set_progress(load_id, **fields):
    if not load_id:
        return
    with _PROGRESS_LOCK:
        entry = _PROGRESS.setdefault(load_id, {"created_at": time.time()})
        entry.update(fields)
        entry["updated_at"] = time.time()
        now = time.time()
        stale = [
            k for k, v in _PROGRESS.items()
            if now - v.get("updated_at", v.get("created_at", now)) > _PROGRESS_TTL_SECONDS
        ]
        for k in stale:
            _PROGRESS.pop(k, None)


from cellmap_flow.dashboard.finetune_utils import (
    create_annotation_volume_zarr,
    ensure_minio_serving,
)
from cellmap_flow.dashboard.routes.finetune.annotation_core import (
    _get_selected_model_config,
    _register_annotation_volume,
)
from cellmap_flow.dashboard.routes.finetune.common import (
    ensure_corrections_storage,
    rewrite_minio_url_for_proxy,
)
from cellmap_flow.dashboard.routes.finetune.overlay import refresh_annotated_regions_layer
from cellmap_flow.finetune.crop_loader import (
    _open_array,
    _read_voxel_size_and_offset,
    parse_crops_yaml,
    remap_labels,
)
from cellmap_flow.finetune.virtual_dataset import write_manifest
from cellmap_flow.globals import current_input_norm_config, g

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Volume bookkeeping
# ---------------------------------------------------------------------------

def _find_session_annotation_volume(corrections_dir):
    """Return ``(volume_id, meta)`` for the annotation_volume in this corrections
    dir, or ``(None, None)`` if none is registered yet."""
    for vid, meta in (getattr(g, "annotation_volumes", {}) or {}).items():
        if meta.get("corrections_dir") == corrections_dir:
            return vid, meta
    return None, None


def _create_session_annotation_volume(
    *,
    raw_dataset_path,
    corrections_dir,
    model_name,
    config,
):
    """Create a fresh annotation_volume.zarr in ``corrections_dir`` and register it.

    Mirrors the body of ``create_annotation_volume_response`` minus the
    HTTP-shaped response wrapping; returns the freshly-built ``(volume_id, meta)``.
    """
    from cellmap_flow.image_data_interface import ImageDataInterface
    from cellmap_flow.utils.neuroglancer_utils import get_raw_closest_scale

    read_shape = np.array(config.read_shape)
    write_shape = np.array(config.write_shape)
    claimed_input_voxel_size = np.array(config.input_voxel_size)
    claimed_output_voxel_size = np.array(config.output_voxel_size)
    output_size = (write_shape / claimed_output_voxel_size).astype(int)
    input_size = (read_shape / claimed_input_voxel_size).astype(int)

    try:
        eff_output_vs = np.array(
            get_raw_closest_scale(raw_dataset_path, tuple(claimed_output_voxel_size))
            or claimed_output_voxel_size
        )
        eff_input_vs = np.array(
            get_raw_closest_scale(raw_dataset_path, tuple(claimed_input_voxel_size))
            or claimed_input_voxel_size
        )
    except Exception:
        eff_output_vs = claimed_output_voxel_size
        eff_input_vs = claimed_input_voxel_size

    idi = ImageDataInterface(raw_dataset_path, voxel_size=eff_output_vs)
    dataset_offset_nm = np.array(idi.roi.offset)
    dataset_shape_nm = np.array(idi.roi.shape)
    dataset_shape_voxels = (dataset_shape_nm / eff_output_vs).astype(int)
    dataset_shape_voxels = (
        np.ceil(dataset_shape_voxels / output_size).astype(int) * output_size
    )

    volume_id = (
        f"vol-{uuid.uuid4().hex[:8]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    zarr_path = os.path.join(corrections_dir, f"{volume_id}.zarr")

    success, info = create_annotation_volume_zarr(
        zarr_path=zarr_path,
        dataset_shape_voxels=dataset_shape_voxels,
        output_voxel_size=eff_output_vs,
        dataset_offset_nm=dataset_offset_nm,
        chunk_size=output_size,
        dataset_path=raw_dataset_path,
        model_name=model_name,
        input_size=input_size,
        input_voxel_size=eff_input_vs,
        claimed_output_voxel_size=claimed_output_voxel_size,
        claimed_input_voxel_size=claimed_input_voxel_size,
        # Snapshot whatever input_norm the dashboard is currently using so
        # the trainer can reproduce inference-side normalization.
        input_norm_config=current_input_norm_config(),
    )
    if not success:
        raise RuntimeError(f"create_annotation_volume_zarr failed: {info}")

    minio_url = ensure_minio_serving(zarr_path, volume_id, output_base_dir=corrections_dir)
    minio_url = rewrite_minio_url_for_proxy(minio_url, request)
    _register_annotation_volume(
        volume_id,
        zarr_path=zarr_path,
        model_name=model_name,
        output_size=output_size.tolist(),
        input_size=input_size.tolist(),
        input_voxel_size=eff_input_vs.tolist(),
        output_voxel_size=eff_output_vs.tolist(),
        claimed_input_voxel_size=claimed_input_voxel_size.tolist(),
        claimed_output_voxel_size=claimed_output_voxel_size.tolist(),
        dataset_path=raw_dataset_path,
        dataset_offset_nm=dataset_offset_nm.tolist(),
        corrections_dir=corrections_dir,
        minio_url=minio_url,
    )
    meta = g.annotation_volumes[volume_id]
    return volume_id, meta


def _ensure_editable_layer(volume_id, minio_url):
    """Add the volume's MinIO-backed annotation layer to the viewer if absent."""
    import neuroglancer

    if not getattr(g, "viewer", None) or not minio_url:
        return
    layer_name = f"annotation_{volume_id}"
    try:
        with g.viewer.txn() as s:
            if layer_name in s.layers:
                return
            source_config = {
                "url": f"s3+{minio_url}/annotation",
                "subsources": {"default": {"writingEnabled": True}, "bounds": {}},
            }
            s.layers[layer_name] = neuroglancer.SegmentationLayer(source=source_config)
    except Exception as e:
        logger.warning(f"Could not add editable layer for {volume_id}: {e}")


# ---------------------------------------------------------------------------
# Crop -> volume write
# ---------------------------------------------------------------------------

def _write_crop_into_volume(volume_meta, entry, *, progress_callback=None):
    """Read a YAML crop's annotation, remap, and write it into volume[s0] at the
    crop's physical offset. Returns the number of FG voxels written."""
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
    if not np.allclose(src_voxel_size_nm, eff_output_vs):
        logger.warning(
            f"Crop {entry.path} voxel size {tuple(src_voxel_size_nm)} != "
            f"volume voxel size {tuple(eff_output_vs)}. Writing values as-is "
            "without resampling — caller should ensure scale compatibility."
        )

    t2 = time.time()
    remapped = remap_labels(
        src_data,
        fg_ids=entry.fg_ids,
        bg_ids=list(entry.bg_ids),
        mode=entry.mode,
        connected_components=entry.connected_components,
    )
    t_remap = time.time() - t2
    t3 = time.time()
    n_fg = int(np.count_nonzero(remapped >= 2))
    t_count = time.time() - t3
    logger.info(
        f"Crop {entry.path} prep: meta={t_meta:.2f}s read={t_read:.2f}s "
        f"({src_data.nbytes/1e6:.1f} MB, dtype={src_data.dtype}, shape={src_data.shape}) "
        f"remap={t_remap:.2f}s count_fg={t_count:.2f}s"
    )

    dataset_offset_nm = np.array(volume_meta["dataset_offset_nm"], dtype=float)
    write_voxel_offset = (
        (src_offset_nm - dataset_offset_nm) / eff_output_vs
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
        raise ValueError(
            f"Crop {entry.path} write region [{z0}:{z0+sz}, {y0}:{y0+sy}, {x0}:{x0+sx}] "
            f"is outside annotation volume shape {arr.shape}. Check the source's "
            "OME-NGFF translation against the dataset offset."
        )

    # Slice the crop into Z-aligned slabs and write them in parallel. Slabs
    # are aligned to the underlying zarr chunk size so two slabs never
    # touch the same chunk, making concurrent writes safe (zarr's chunk
    # writes are per-chunk-file, no shared mutable state).
    #
    # Slab count tracks the LSF slot allocation so we always fully use what
    # bsub gave us — capped by the number of chunk-aligned slabs we can
    # actually produce.
    from cellmap_flow.dashboard.finetune_utils import _get_sync_worker_count

    chunk_z = max(int(arr.chunks[0]), 1)
    max_chunk_slabs = int(np.ceil(sz / chunk_z))
    n_slabs = max(1, min(_get_sync_worker_count(), max_chunk_slabs))
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

    # Record this import in the volume's root attrs so the bounding-box
    # overlay can surface it as a single yellow box per crop (vs. the
    # per-chunk small boxes from painted scribbles).
    vol_root = zarr.open(volume_meta["zarr_path"], mode="r+")
    imported = list(vol_root.attrs.get("imported_crops", []))
    imported.append(
        {
            "path": entry.path,
            "name": entry.name,
            "annotation_offset_voxels": [int(z0), int(y0), int(x0)],
            "annotation_shape_voxels": [int(sz), int(sy), int(sx)],
            "n_fg_voxels": int(n_fg),
        }
    )
    vol_root.attrs["imported_crops"] = imported

    return n_fg


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

def load_crops_from_yaml_response(data):
    """Import crops from a YAML manifest into the session's annotation_volume.

    Request JSON:
        - ``model_name``: required
        - ``output_path``: optional, base path for the session corrections dir
        - ``yaml``: required, YAML text (or path to a YAML file)
        - ``load_id``: optional UUID for live progress polling
    """
    try:
        model_name = data.get("model_name")
        output_path = data.get("output_path")
        yaml_input = data.get("yaml")
        load_id = data.get("load_id")
        if load_id:
            _set_progress(
                load_id,
                phase="starting",
                current_path="",
                tile_done=0,
                tile_total=0,
                crop_index=0,
                n_crops=0,
                done=False,
            )

        if not yaml_input:
            return jsonify({"success": False, "error": "Missing 'yaml' field"}), 400
        if not model_name:
            return jsonify({"success": False, "error": "Missing 'model_name' field"}), 400

        try:
            crops_config = parse_crops_yaml(yaml_input)
        except ValidationError as e:
            return (
                jsonify({"success": False, "error": "YAML validation failed", "details": e.errors()}),
                400,
            )
        except Exception as e:
            return jsonify({"success": False, "error": f"YAML parse error: {e}"}), 400

        if not crops_config.crops:
            return jsonify({"success": False, "error": "No crops listed in YAML"}), 400

        model_config, error_response = _get_selected_model_config(model_name)
        if error_response is not None:
            return error_response

        raw_dataset_path = getattr(g, "dataset_path", None)
        if not raw_dataset_path:
            return jsonify({"success": False, "error": "No raw dataset path configured"}), 400

        _, corrections_dir = ensure_corrections_storage(output_path)

        # Reuse the session's annotation_volume if the user already created one
        # (via "New Volume" or "Resume Existing"). Otherwise spin up a fresh one
        # so the YAML import has a destination.
        volume_id, volume_meta = _find_session_annotation_volume(corrections_dir)
        created_volume = False
        if volume_meta is None:
            volume_id, volume_meta = _create_session_annotation_volume(
                raw_dataset_path=raw_dataset_path,
                corrections_dir=corrections_dir,
                model_name=model_name,
                config=model_config.config,
            )
            created_volume = True
        _ensure_editable_layer(volume_id, volume_meta.get("minio_url"))

        n_crops = len(crops_config.crops)
        errors = []
        total_fg_written = 0
        for crop_index, entry in enumerate(crops_config.crops):
            if load_id:
                _set_progress(
                    load_id,
                    phase="crop_start",
                    crop_index=crop_index,
                    n_crops=n_crops,
                    current_path=entry.path,
                    tile_done=0,
                    tile_total=0,
                    done=False,
                )
            try:
                def _cb(done, total, ci=crop_index, p=entry.path):
                    if load_id:
                        _set_progress(
                            load_id,
                            phase="tile",
                            crop_index=ci,
                            n_crops=n_crops,
                            current_path=p,
                            tile_done=int(done),
                            tile_total=int(total),
                            done=False,
                        )

                n_fg = _write_crop_into_volume(
                    volume_meta, entry, progress_callback=_cb
                )
                total_fg_written += n_fg
                logger.info(f"Imported crop {entry.path}: {n_fg} FG voxels")
            except Exception as e:
                logger.exception(f"Failed to import crop {entry.path}")
                errors.append({"path": entry.path, "error": str(e)})

        # The MinIO bucket was mirrored once at volume-create time, when the
        # zarr held only metadata. Re-mirror now that chunk data is written
        # so neuroglancer can read the imported annotations from the
        # editable layer.
        try:
            ensure_minio_serving(
                volume_meta["zarr_path"],
                volume_id,
                output_base_dir=corrections_dir,
            )
        except Exception as e:
            logger.warning(f"MinIO re-mirror failed for {volume_id}: {e}")

        # Manifest: trainer reads from this single volume zarr. The
        # ``input_norm`` block carries the dashboard's current normalization
        # so VirtualPatchDataset (running in the LSF trainer process where
        # g.input_norms is empty) can apply the same normalization the
        # dashboard does at inference time. Without this the trainer feeds
        # the model raw uint8 while inference feeds it [-1, 1] -- the
        # trained adapter is then nonsense at inference time.
        manifest = {
            "kind": "volume_zarr_v1",
            "volume_zarr_path": volume_meta["zarr_path"],
            "raw_dataset_path": raw_dataset_path,
            "input_size_voxels": list(volume_meta["input_size"]),
            "output_size_voxels": list(volume_meta["output_size"]),
            "input_voxel_size_nm": list(volume_meta["input_voxel_size"]),
            "output_voxel_size_nm": list(volume_meta["output_voxel_size"]),
            # patches_per_epoch=None tells VirtualPatchDataset to default to
            # "one patch per populated chunk" (full coverage). Explicit ints
            # in the YAML pass through verbatim.
            "patches_per_epoch": crops_config.patches_per_epoch,
            "jitter_voxels": crops_config.jitter_voxels,
            "seed": crops_config.seed,
            "input_norm": current_input_norm_config(),
            # None → auto-balance dense vs sparse pools (50/50 when both
            # exist, else use the surviving pool).
            "dense_to_sparse_ratio": crops_config.dense_to_sparse_ratio,
        }
        write_manifest(corrections_dir, manifest)

        try:
            refresh_annotated_regions_layer(corrections_path=corrections_dir)
        except Exception as e:
            logger.warning(f"refresh_annotated_regions_layer failed: {e}")

        if load_id:
            _set_progress(
                load_id,
                phase="done",
                done=True,
                n_crops_imported=n_crops - len(errors),
                n_errors=len(errors),
                volume_id=volume_id,
                fg_voxels_written=total_fg_written,
            )

        return jsonify(
            {
                "success": True,
                "n_crops_requested": n_crops,
                "n_crops_imported": n_crops - len(errors),
                "n_errors": len(errors),
                "fg_voxels_written": total_fg_written,
                "volume_id": volume_id,
                "created_new_volume": created_volume,
                "errors": errors,
            }
        )
    except Exception as e:
        logger.exception("load_crops_from_yaml_response failed")
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Auxiliary endpoints (file read + progress polling) — unchanged behavior
# ---------------------------------------------------------------------------

def get_load_crops_progress_response(load_id):
    """Return current progress for an in-flight ``/api/finetune/load-crops`` call."""
    if not load_id:
        return jsonify({"success": False, "error": "Missing 'load_id' query param"}), 400
    with _PROGRESS_LOCK:
        snapshot = _PROGRESS.get(load_id)
        snapshot = dict(snapshot) if snapshot else None
    if snapshot is None:
        return jsonify({"success": False, "error": f"Unknown load_id {load_id}"}), 404
    return jsonify({"success": True, "progress": snapshot})


def read_yaml_file_response(path):
    """Return the contents of a YAML file so the dashboard can preview/edit it."""
    if not path:
        return jsonify({"success": False, "error": "Missing 'path' query param"}), 400
    if not os.path.exists(path):
        return jsonify({"success": False, "error": f"File not found: {path}"}), 404
    if not os.path.isfile(path):
        return jsonify({"success": False, "error": f"Not a file: {path}"}), 400
    if os.path.getsize(path) > 1_000_000:
        return jsonify({"success": False, "error": "File exceeds 1 MB; paste it directly instead"}), 400
    try:
        with open(path) as f:
            text = f.read()
        return jsonify({"success": True, "text": text})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
