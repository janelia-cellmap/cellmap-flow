"""Endpoint for bulk-loading externally annotated crops via a YAML manifest."""

import logging
import os
import threading
import time

import numpy as np
from flask import jsonify
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
        # Evict stale entries.
        now = time.time()
        stale = [
            k for k, v in _PROGRESS.items()
            if now - v.get("updated_at", v.get("created_at", now)) > _PROGRESS_TTL_SECONDS
        ]
        for k in stale:
            _PROGRESS.pop(k, None)

from cellmap_flow.dashboard.routes.finetune.annotation_core import _get_selected_model_config
from cellmap_flow.dashboard.routes.finetune.common import ensure_corrections_storage
from cellmap_flow.dashboard.routes.finetune.overlay import refresh_annotated_regions_layer
from cellmap_flow.finetune.crop_loader import load_crops, parse_crops_yaml
from cellmap_flow.globals import g


def _add_yaml_crop_layers(crops_config):
    """Add a SegmentationLayer per YAML crop pointing at the source annotation zarr.

    Mirrors how ``get_raw_layer`` builds its image layer: opens the source via
    a LocalVolume so neuroglancer can read it directly without serving it over
    MinIO. The layer reflects the original instance labels at their native
    voxel size and offset (read from the zarr's OME-NGFF metadata), so they
    overlay correctly on top of raw.
    """
    import neuroglancer
    from cellmap_flow.image_data_interface import ImageDataInterface
    from cellmap_flow.finetune.crop_loader import _read_voxel_size_and_offset, _derive_name

    if not getattr(g, "viewer", None):
        return 0
    added = 0
    for entry in crops_config.crops:
        try:
            sub, voxel_size, offset_nm = _read_voxel_size_and_offset(entry.path)
            sub_path = entry.path
            for piece in sub:
                sub_path = f"{sub_path}/{piece}"
            idi = ImageDataInterface(sub_path)
            voxel_offset = (offset_nm / voxel_size).astype(int).tolist()
            crop_name = entry.name or _derive_name(entry.path)
            layer_name = f"yaml_crop_{crop_name}"
            with g.viewer.txn() as s:
                s.layers[layer_name] = neuroglancer.SegmentationLayer(
                    source=neuroglancer.LocalVolume(
                        data=idi.ts,
                        dimensions=neuroglancer.CoordinateSpace(
                            names=list(idi.axes_names),
                            units="nm",
                            scales=list(idi.voxel_size),
                        ),
                        voxel_offset=voxel_offset,
                    ),
                )
            added += 1
        except Exception as e:
            logger.warning(
                f"Could not add viewer layer for YAML crop {entry.path}: {e}"
            )
    return added

logger = logging.getLogger(__name__)


def load_crops_from_yaml_response(data):
    """Materialize a batch of YAML-described crops as ``_chunk_*.zarr`` entries.

    Request JSON:
        - ``model_name``: required, used to derive input/output shape/voxel size
        - ``output_path``: optional, base path for the session corrections dir
        - ``yaml``: required, the YAML text (or path to a YAML file)
    """
    try:
        model_name = data.get("model_name")
        output_path = data.get("output_path")
        yaml_input = data.get("yaml")
        load_id = data.get("load_id")
        if load_id:
            _set_progress(load_id, phase="starting", current_path="", tile_done=0,
                          tile_total=0, crop_index=0, n_crops=0, done=False)

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

        config = model_config.config
        read_shape = np.array(config.read_shape)
        write_shape = np.array(config.write_shape)
        claimed_input_voxel_size = np.array(config.input_voxel_size)
        claimed_output_voxel_size = np.array(config.output_voxel_size)
        input_size = (read_shape / claimed_input_voxel_size).astype(int)
        output_size = (write_shape / claimed_output_voxel_size).astype(int)

        raw_dataset_path = getattr(g, "dataset_path", None)
        if not raw_dataset_path:
            return jsonify({"success": False, "error": "No raw dataset path configured"}), 400

        _, corrections_dir = ensure_corrections_storage(output_path)

        def _progress_cb(snapshot):
            if load_id:
                _set_progress(load_id, **snapshot, done=False)

        result = load_crops(
            crops_config,
            raw_dataset_path=raw_dataset_path,
            corrections_dir=corrections_dir,
            input_size=input_size,
            output_size=output_size,
            claimed_input_voxel_size=claimed_input_voxel_size,
            claimed_output_voxel_size=claimed_output_voxel_size,
            model_name=model_name,
            progress_callback=_progress_cb,
        )

        if load_id:
            _set_progress(
                load_id,
                phase="done",
                done=True,
                n_chunks_created=len(result["created"]),
                n_errors=len(result["errors"]),
            )

        try:
            # Pass the corrections_dir explicitly so the overlay refresh works
            # even when no painted annotation_volume has been registered for
            # this session (YAML-loaded crops alone don't create one).
            refresh_annotated_regions_layer(corrections_path=corrections_dir)
        except Exception as e:
            logger.warning(f"refresh_annotated_regions_layer failed: {e}")

        # Surface each loaded crop as a SegmentationLayer in the viewer so the
        # user can see the actual annotation pixels overlaid on raw, not just
        # the bounding-box outlines.
        try:
            n_layers_added = _add_yaml_crop_layers(crops_config)
        except Exception as e:
            logger.warning(f"_add_yaml_crop_layers failed: {e}")
            n_layers_added = 0

        return jsonify(
            {
                "success": True,
                "n_crops_requested": len(crops_config.crops),
                "n_chunks_created": len(result["created"]),
                "n_errors": len(result["errors"]),
                "n_layers_added": n_layers_added,
                "errors": result["errors"],
            }
        )
    except Exception as e:
        logger.exception("load_crops_from_yaml_response failed")
        return jsonify({"success": False, "error": str(e)}), 500


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
