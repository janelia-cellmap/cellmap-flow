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
from datetime import datetime

from flask import jsonify
from pydantic import ValidationError

from cellmap_flow.utils.model_geometry import resolve_model_geometry

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
    ensure_minio_serving,
    sync_annotation_volume_from_minio,
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
from cellmap_flow.finetune.crop_loader import parse_crops_yaml
from cellmap_flow.finetune.session.volume import (
    build_manifest,
    create_volume_zarr,
    plan_volume,
    write_crop_into_volume,
)
from cellmap_flow.finetune.session.volume import (  # noqa: F401  (kept name)
    majority_vote_downsample as _majority_vote_downsample,
)
from cellmap_flow.finetune.virtual_dataset import write_manifest
from cellmap_flow.globals import current_input_norm_config, current_postprocess_config, g

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
    geometry = plan_volume(raw_dataset_path, config, rounding="legacy_floor")
    volume_id = (
        f"vol-{uuid.uuid4().hex[:8]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    zarr_path = os.path.join(corrections_dir, f"{volume_id}.zarr")
    # Snapshot whatever input_norm/postprocess the dashboard is currently
    # using so the trainer can reproduce inference-side normalization and
    # the generated finetuned yaml can reproduce output postprocessing.
    create_volume_zarr(
        zarr_path,
        geometry,
        dataset_path=raw_dataset_path,
        model_name=model_name,
        input_norm=current_input_norm_config(),
        postprocess=current_postprocess_config(),
    )

    minio_url = ensure_minio_serving(zarr_path, volume_id, output_base_dir=corrections_dir)
    minio_url = rewrite_minio_url_for_proxy(minio_url)
    _register_annotation_volume(
        volume_id,
        zarr_path=zarr_path,
        model_name=model_name,
        output_size=list(geometry.chunk_size),
        input_size=list(geometry.input_size),
        input_voxel_size=list(geometry.input_voxel_size),
        output_voxel_size=list(geometry.output_voxel_size),
        claimed_input_voxel_size=list(geometry.claimed_input_voxel_size),
        claimed_output_voxel_size=list(geometry.claimed_output_voxel_size),
        dataset_path=raw_dataset_path,
        dataset_offset_nm=list(geometry.dataset_offset_nm),
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
    """Write a YAML crop into the volume (``session.volume.write_crop_into_volume``);
    returns the number of FG voxels written."""
    record = write_crop_into_volume(volume_meta, entry, progress_callback=progress_callback)
    return record["n_fg_voxels"]


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

        started_at = time.time()

        def step(phase, message, **extra):
            """Report a setup step.

            Everything between "starting" and the first crop used to run
            silently, and it is the slow part: resolving the model, creating
            the annotation volume, starting MinIO. The UI sat on "Starting..."
            for all of it with no way to tell which step was running, or
            whether anything was running at all.

            Each message carries elapsed time, so "this is slow" can be
            answered with which step is slow rather than a guess.
            """
            elapsed = time.time() - started_at
            stamped = f"[{elapsed:.0f}s] {message}"
            logger.info(stamped)
            if load_id:
                _set_progress(load_id, phase=phase, message=stamped, **extra)

        if not yaml_input:
            return jsonify({"success": False, "error": "Missing 'yaml' field"}), 400
        if not model_name:
            return jsonify({"success": False, "error": "Missing 'model_name' field"}), 400

        step("setup", "Reading the crop manifest...")
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

        n_crops = len(crops_config.crops)
        step(
            "setup",
            f"Found {n_crops} crop{'' if n_crops == 1 else 's'}; resolving model "
            f"{model_name}...",
            n_crops=n_crops,
        )
        model_config, error_response = _get_selected_model_config(model_name)
        if error_response is not None:
            return error_response

        raw_dataset_path = getattr(g, "dataset_path", None)
        if not raw_dataset_path:
            return jsonify({"success": False, "error": "No raw dataset path configured"}), 400

        step("setup", "Preparing the corrections directory...", n_crops=n_crops)
        _, corrections_dir = ensure_corrections_storage(output_path)

        # Reuse the session's annotation_volume if the user already created one
        # (via "New Volume" or "Resume Existing"). Otherwise spin up a fresh one
        # so the YAML import has a destination.
        volume_id, volume_meta = _find_session_annotation_volume(corrections_dir)
        created_volume = False
        if volume_meta is None:
            step(
                "setup",
                "Creating the annotation volume (asking the inference server "
                "for the model's geometry)...",
                n_crops=n_crops,
            )
            volume_id, volume_meta = _create_session_annotation_volume(
                raw_dataset_path=raw_dataset_path,
                corrections_dir=corrections_dir,
                model_name=model_name,
                # Only shapes and voxel sizes are read from this; the running
                # server can supply them without building the model here.
                config=resolve_model_geometry(model_name, model_config),
            )
            created_volume = True
        step(
            "setup",
            "Serving the volume through MinIO and adding the editable layer...",
            n_crops=n_crops,
        )
        _ensure_editable_layer(volume_id, volume_meta.get("minio_url"))

        if not created_volume:
            # The crops are written into the local chunks and then mirrored
            # up over MinIO's. Pull what was painted since the last sync
            # first, or those chunks go up without the strokes.
            try:
                sync_annotation_volume_from_minio(volume_id)
            except Exception as e:
                logger.warning(f"Could not pull painted chunks of {volume_id} before the import: {e}")

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
        manifest = build_manifest(
            volume_meta,
            input_norm=current_input_norm_config(),
            postprocess=current_postprocess_config(),
            overrides={
                "raw_dataset_path": raw_dataset_path,
                # None tells VirtualPatchDataset "one patch per populated
                # chunk" (full coverage); explicit ints pass through.
                "patches_per_epoch": crops_config.patches_per_epoch,
                "jitter_voxels": crops_config.jitter_voxels,
                "seed": crops_config.seed,
                # None -> auto-balance dense vs sparse pools.
                "dense_to_sparse_ratio": crops_config.dense_to_sparse_ratio,
            },
        )
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
    """Return the contents of a YAML file so the dashboard can preview/edit it.

    The dashboard listens on every interface, and this used to return any
    file the user could read, so it now serves only files that are YAML by
    name after resolving symlinks. The name is checked before existence so
    the route cannot be used to probe for other files either.
    """
    if not path:
        return jsonify({"success": False, "error": "Missing 'path' query param"}), 400
    real = os.path.realpath(os.path.expanduser(path))
    if not real.lower().endswith((".yaml", ".yml")):
        return jsonify({"success": False, "error": "Only .yaml or .yml files can be read"}), 400
    if not os.path.exists(real):
        return jsonify({"success": False, "error": f"File not found: {path}"}), 404
    if not os.path.isfile(real):
        return jsonify({"success": False, "error": f"Not a file: {path}"}), 400
    if os.path.getsize(real) > 1_000_000:
        return jsonify({"success": False, "error": "File exceeds 1 MB; paste it directly instead"}), 400
    try:
        with open(real) as f:
            text = f.read()
        return jsonify({"success": True, "text": text})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
