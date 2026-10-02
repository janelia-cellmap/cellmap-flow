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

Routes: POST ``/api/finetune/load-crops`` (the import), GET
``/api/finetune/load-crops-progress`` (how far an import has got) and GET
``/api/finetune/read-yaml`` (a YAML file's text, for the editor).
"""

import logging
import time

from flask import jsonify, request
from pydantic import ValidationError

from cellmap_flow.dashboard.finetune_utils import (
    ensure_minio_serving,
    sync_annotation_volume_from_minio,
)
from cellmap_flow.dashboard.progress import Progress
from cellmap_flow.dashboard.requests import LoadCrops, parse
from cellmap_flow.dashboard.routes.finetune.annotation_core import (
    _get_selected_model_config,
    serve_new_volume,
)
from cellmap_flow.dashboard.routes.finetune.blueprint import finetune_bp
from cellmap_flow.dashboard.routes.finetune.common import (
    current_chain,
    ensure_corrections_storage,
    session_store,
)
from cellmap_flow.dashboard.routes.finetune.overlay import (
    add_annotation_layer,
    refresh_annotated_regions_layer,
)
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.finetune.crop_loader import (
    YamlFileRefused,
    parse_crops_yaml,
    read_yaml_file,
)
from cellmap_flow.finetune.session.volume import (
    build_manifest,
    plan_volume,
    write_crop_into_volume,
)
from cellmap_flow.finetune.session.manifest import write_manifest
from cellmap_flow.models.geometry_cache import resolve_model_geometry

logger = logging.getLogger(__name__)

# Each import's progress, by the load_id the page sent with it: the phase,
# the crop and tile it is on, and at the end what it imported.
_PROGRESS = Progress()


# ---------------------------------------------------------------------------
# Volume bookkeeping
# ---------------------------------------------------------------------------

def _create_session_annotation_volume(
    *,
    raw_dataset_path,
    corrections_dir,
    model_name,
    config,
):
    """Create, serve and register a fresh volume in ``corrections_dir``.

    What create-volume does, less the HTTP response: returns
    ``(volume_id, record)``.
    """
    geometry = plan_volume(raw_dataset_path, config, resample=get_session().resample)
    volume_id, zarr_path, minio_url = serve_new_volume(
        geometry, corrections_dir, raw_dataset_path, model_name
    )
    record = session_store().register_volume(
        volume_id,
        minio_url=minio_url,
        **geometry.record(
            zarr_path,
            dataset_path=raw_dataset_path,
            model_name=model_name,
            corrections_dir=corrections_dir,
        ),
    )
    return volume_id, record


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@finetune_bp.route("/api/finetune/load-crops", methods=["POST"])
def load_crops_from_yaml():
    """Import crops from a YAML manifest into the session's annotation_volume.

    Request JSON: see requests.LoadCrops.
    """
    body, refused = parse(LoadCrops, request.get_json() or {})
    if refused:
        return refused
    model_name, output_path, yaml_input, load_id = body.model_name, body.output_path, body.yaml, body.load_id
    try:
        _PROGRESS.update(
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
            _PROGRESS.update(load_id, phase=phase, message=stamped, **extra)

        step("setup", "Reading the crop manifest...")
        try:
            crops_config = parse_crops_yaml(yaml_input)
        except YamlFileRefused as e:
            return jsonify({"success": False, "error": str(e)}), 400
        except ValidationError as e:
            # Only where and what: each error's "input" is the offending
            # value, which for a top-level error is the whole document.
            details = [{"loc": list(err["loc"]), "msg": err["msg"]} for err in e.errors()]
            return (
                jsonify({"success": False, "error": "YAML validation failed", "details": details}),
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

        raw_dataset_path = get_session().dataset_path
        if not raw_dataset_path:
            return jsonify({"success": False, "error": "No raw dataset path configured"}), 400

        step("setup", "Preparing the corrections directory...", n_crops=n_crops)
        _, corrections_dir = ensure_corrections_storage(output_path)

        # Reuse the session's annotation_volume if the user already created one
        # (via "New Volume" or "Resume Existing"). Otherwise spin up a fresh one
        # so the YAML import has a destination.
        volume_id, volume_meta = session_store().session_volume(corrections_dir)
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
        viewer, minio_url = get_session().viewer, volume_meta.get("minio_url")
        if viewer is not None and minio_url:
            try:
                # Kept if there: it may be the layer the user is painting.
                add_annotation_layer(viewer, f"annotation_{volume_id}", f"{minio_url}/annotation",
                                     keep_existing=True)
            except Exception as e:
                logger.warning(f"Could not add editable layer for {volume_id}: {e}")

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
            _PROGRESS.update(
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
                    _PROGRESS.update(
                        load_id,
                        phase="tile",
                        crop_index=ci,
                        n_crops=n_crops,
                        current_path=p,
                        tile_done=int(done),
                        tile_total=int(total),
                        done=False,
                    )

                n_fg = write_crop_into_volume(
                    volume_meta, entry, progress_callback=_cb
                )["n_fg_voxels"]
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
        # so VirtualPatchDataset (running in the LSF trainer process, which
        # has no chain of the dashboard's) can apply the same normalization the
        # dashboard does at inference time. Without this the trainer feeds
        # the model raw uint8 while inference feeds it [-1, 1] -- the
        # trained adapter is then nonsense at inference time.
        input_norm, postprocess = current_chain()
        manifest = build_manifest(
            volume_meta,
            input_norm=input_norm,
            postprocess=postprocess,
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

        _PROGRESS.update(
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
        logger.exception("Loading crops from a YAML failed")
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Auxiliary endpoints (progress polling + file read)
# ---------------------------------------------------------------------------

@finetune_bp.route("/api/finetune/load-crops-progress", methods=["GET"])
def get_load_crops_progress():
    """Return current progress for an in-flight ``/api/finetune/load-crops`` call."""
    return _PROGRESS.response(request.args.get("load_id"))


@finetune_bp.route("/api/finetune/read-yaml", methods=["GET"])
def read_yaml():
    """Return the contents of a YAML file so the dashboard can preview/edit it.

    :func:`~cellmap_flow.finetune.crop_loader.read_yaml_file` decides which
    files may be read: the dashboard listens on every interface, and this
    used to return any file the user could read.
    """
    path = request.args.get("path")
    if not path:
        return jsonify({"success": False, "error": "Missing 'path' query param"}), 400
    try:
        return jsonify({"success": True, "text": read_yaml_file(path)})
    except YamlFileRefused as e:
        return jsonify({"success": False, "error": str(e)}), 400
