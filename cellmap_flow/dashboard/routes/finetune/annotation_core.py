import logging
import os
import time
import uuid
from datetime import datetime

import numpy as np
import zarr
from flask import jsonify, request

from cellmap_flow.dashboard.finetune_utils import (
    create_annotation_volume_zarr,
    create_correction_zarr,
    ensure_minio_serving,
)
from cellmap_flow.dashboard.routes.finetune.common import (
    ensure_corrections_storage,
    find_model_config,
    load_user_prefs,
    rewrite_minio_url_for_proxy,
    save_user_prefs,
    viewer_position_and_scales,
    write_volume_manifest,
)
from cellmap_flow.dashboard.routes.finetune.overlay import refresh_annotated_regions_layer
from cellmap_flow.globals import current_input_norm_config, current_postprocess_config, g
from cellmap_flow.utils.model_geometry import resolve_model_geometry
from cellmap_flow.utils.server_info import (
    fetch_model_info,
    model_geometry,
    running_job_host,
)

logger = logging.getLogger(__name__)


def _get_selected_model_config(model_name):
    if not getattr(g, "models_config", None):
        return None, (jsonify({"success": False, "error": "No models loaded"}), 400)

    model_config = find_model_config(model_name)
    if model_config is None:
        return None, (
            jsonify({"success": False, "error": f"Model {model_name} not found"}),
            404,
        )

    return model_config, None


def _register_annotation_volume(volume_id, **volume_data):
    if not hasattr(g, "annotation_volumes"):
        g.annotation_volumes = {}
    g.annotation_volumes[volume_id] = {
        **volume_data,
        "extracted_chunks": set(),
        "chunk_sync_state": {},
    }


# ``model_config.config`` is far from free: for a script model it executes the
# config file, which downloads weights and runs torch.export before it can
# report a shape. ModelConfig caches the result, but only on success -- a
# failure leaves ``_config`` None, so the next access redoes the whole thing.
# The finetune tab polls this endpoint every two seconds while it waits for a
# model, which turns one failure (a busy GPU, say) into hundreds of full model
# loads. Remember failures briefly instead, short enough that a transient cause
# still recovers on its own.
_CONFIG_FAILURE_COOLDOWN_SECONDS = 60
_config_failure_until = {}


def _config_retry_blocked(name) -> bool:
    until = _config_failure_until.get(name)
    return until is not None and time.time() < until


def _geometry_from_server(name):
    """Geometry from the running inference server, which already has the model."""
    return model_geometry(fetch_model_info(running_job_host(name)))


def _geometry_from_saved_pipeline(name):
    configs = getattr(g, "pipeline_model_configs", None) or {}
    cfg = configs.get(name)
    if not cfg:
        return None
    return {
        "write_shape": cfg.get("write_shape", []),
        "output_voxel_size": cfg.get("output_voxel_size", []),
        "output_channels": cfg.get("output_channels", 1),
    }


def _geometry_from_local_load(name, model_config):
    """Last resort: build the model here just to read its shape.

    Only reached when no server is up and nothing was saved, because it is by
    far the most expensive and least reliable source -- see the note on
    _CONFIG_FAILURE_COOLDOWN_SECONDS above.
    """
    if model_config is None or _config_retry_blocked(name):
        return None
    try:
        config = model_config.config
        _config_failure_until.pop(name, None)
        return {
            "write_shape": list(config.write_shape),
            "output_voxel_size": list(config.output_voxel_size),
            "output_channels": config.output_channels,
        }
    except Exception as e:
        _config_failure_until[name] = time.time() + _CONFIG_FAILURE_COOLDOWN_SECONDS
        logger.warning(
            f"Could not extract config for {name}: {e}. Not retrying for "
            f"{_CONFIG_FAILURE_COOLDOWN_SECONDS}s."
        )
        return None


def get_finetune_models_response():
    try:
        models = []
        seen = set()

        # Every name we might report on: configured models first, then any job
        # running without a matching config (a yaml-launched model, say).
        configs_by_name = {}
        for mc in getattr(g, "models_config", []) or []:
            name = getattr(mc, "name", None)
            if name:
                configs_by_name[name] = mc
        names = list(configs_by_name)
        for job in getattr(g, "jobs", []) or []:
            job_name = getattr(job, "model_name", None)
            if job_name and job_name not in configs_by_name:
                names.append(job_name)

        for name in names:
            if name in seen:
                continue
            # Cheapest and most reliable source first. Asking the server costs
            # one HTTP round trip; loading the model locally costs a weight
            # download, a torch.export and a CUDA context.
            geometry = (
                _geometry_from_server(name)
                or _geometry_from_saved_pipeline(name)
                or _geometry_from_local_load(name, configs_by_name.get(name))
            )
            if geometry is None:
                logger.warning(f"No configuration available for model: {name}")
                continue
            seen.add(name)
            models.append({"name": name, **geometry})

        selected = models[0]["name"] if len(models) == 1 else None
        return jsonify({"models": models, "selected_model": selected})
    except Exception as e:
        logger.error(f"Error getting finetune models: {e}")
        return jsonify({"error": str(e)}), 500


def get_view_center_response():
    try:
        position, scales_nm = viewer_position_and_scales()
        logger.info(f"Got view center position: {position}")
        return jsonify({"success": True, "position": position, "scales_nm": scales_nm})
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error getting view center position: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


def create_annotation_crop_response(data):
    try:
        from cellmap_flow.image_data_interface import ImageDataInterface
        from funlib.geometry import Coordinate, Roi

        model_name = data.get("model_name")
        output_path = data.get("output_path")

        position, viewer_scales_nm = viewer_position_and_scales()
        view_center = np.array(position)

        model_config, error_response = _get_selected_model_config(model_name)
        if error_response is not None:
            return error_response

        # Ask the running inference server for the geometry. It already has
        # the model; building it here instead costs a full load on the
        # dashboard's CPU -- 43s in one measured session, for shapes the
        # server can report in milliseconds -- and is what made "create
        # annotation volume" feel slow. model_config.config stays as the
        # fallback for when no server is up.
        config = resolve_model_geometry(model_name, model_config)
        read_shape = np.array(config.read_shape)
        write_shape = np.array(config.write_shape)
        input_voxel_size = np.array(config.input_voxel_size)
        output_voxel_size = np.array(config.output_voxel_size)
        output_channels = config.output_channels

        if viewer_scales_nm is not None:
            view_center_nm = view_center * np.array(viewer_scales_nm)
        else:
            view_center_nm = view_center
            logger.warning("No viewer scales provided, assuming view center is already in nm")

        raw_crop_shape_voxels = (read_shape / input_voxel_size).astype(int)
        annotation_crop_shape_voxels = (write_shape / output_voxel_size).astype(int)
        raw_crop_offset_voxels = ((view_center_nm - read_shape / 2) / input_voxel_size).astype(int)
        annotation_crop_offset_voxels = ((view_center_nm - write_shape / 2) / output_voxel_size).astype(int)

        crop_id = f"{uuid.uuid4().hex[:8]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        _, corrections_dir = ensure_corrections_storage(output_path)
        zarr_path = os.path.join(corrections_dir, f"{crop_id}.zarr")

        dataset_path = getattr(g, "dataset_path", "unknown")
        idi = ImageDataInterface(dataset_path, voxel_size=input_voxel_size)
        raw_dtype = str(idi.ts.dtype)

        success, zarr_info = create_correction_zarr(
            zarr_path=zarr_path,
            raw_crop_shape=raw_crop_shape_voxels,
            raw_voxel_size=input_voxel_size,
            raw_offset=raw_crop_offset_voxels,
            annotation_crop_shape=annotation_crop_shape_voxels,
            annotation_voxel_size=output_voxel_size,
            annotation_offset=annotation_crop_offset_voxels,
            dataset_path=dataset_path,
            model_name=model_name,
            output_channels=output_channels,
            raw_dtype=raw_dtype,
            create_mask=False,
        )
        if not success:
            return jsonify({"success": False, "error": zarr_info}), 500

        roi = Roi(offset=Coordinate(view_center_nm - read_shape / 2), shape=Coordinate(read_shape))
        raw_zarr = zarr.open(zarr_path, mode="r+")
        raw_zarr["raw/s0"][:] = idi.to_ndarray_ts(roi)

        minio_url = ensure_minio_serving(zarr_path, crop_id, output_base_dir=corrections_dir)
        minio_url = rewrite_minio_url_for_proxy(minio_url, request)
        return jsonify(
            {
                "success": True,
                "crop_id": crop_id,
                "zarr_path": zarr_path,
                "minio_url": minio_url,
                "neuroglancer_url": f"{minio_url}/annotation",
                "metadata": {
                    "center_position_nm": view_center_nm.tolist(),
                    "raw_crop_offset": raw_crop_offset_voxels.tolist(),
                    "raw_crop_shape": raw_crop_shape_voxels.tolist(),
                    "raw_voxel_size": input_voxel_size.tolist(),
                    "annotation_crop_offset": annotation_crop_offset_voxels.tolist(),
                    "annotation_crop_shape": annotation_crop_shape_voxels.tolist(),
                    "annotation_voxel_size": output_voxel_size.tolist(),
                },
            }
        )
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error creating annotation crop: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


def create_annotation_volume_response(data):
    try:
        from cellmap_flow.image_data_interface import ImageDataInterface
        from cellmap_flow.utils.neuroglancer_utils import get_raw_closest_scale

        model_name = data.get("model_name")
        output_path = data.get("output_path")

        model_config, error_response = _get_selected_model_config(model_name)
        if error_response is not None:
            return error_response

        # Ask the running inference server for the geometry. It already has
        # the model; building it here instead costs a full load on the
        # dashboard's CPU -- 43s in one measured session, for shapes the
        # server can report in milliseconds -- and is what made "create
        # annotation volume" feel slow. model_config.config stays as the
        # fallback for when no server is up.
        config = resolve_model_geometry(model_name, model_config)
        read_shape = np.array(config.read_shape)
        write_shape = np.array(config.write_shape)
        claimed_input_voxel_size = np.array(config.input_voxel_size)
        claimed_output_voxel_size = np.array(config.output_voxel_size)
        output_size = (write_shape / claimed_output_voxel_size).astype(int)
        input_size = (read_shape / claimed_input_voxel_size).astype(int)

        dataset_path = getattr(g, "dataset_path", None)
        if not dataset_path:
            return jsonify({"success": False, "error": "No dataset path configured"}), 400

        try:
            effective_output_voxel_size = np.array(
                get_raw_closest_scale(dataset_path, tuple(claimed_output_voxel_size))
                or claimed_output_voxel_size
            )
            effective_input_voxel_size = np.array(
                get_raw_closest_scale(dataset_path, tuple(claimed_input_voxel_size))
                or claimed_input_voxel_size
            )
        except Exception:
            effective_output_voxel_size = claimed_output_voxel_size
            effective_input_voxel_size = claimed_input_voxel_size

        idi = ImageDataInterface(dataset_path, voxel_size=effective_output_voxel_size)
        dataset_roi = idi.roi
        dataset_offset_nm = np.array(dataset_roi.offset)
        dataset_shape_nm = np.array(dataset_roi.shape)
        dataset_shape_voxels = (dataset_shape_nm / effective_output_voxel_size).astype(int)
        dataset_shape_voxels = np.ceil(dataset_shape_voxels / output_size).astype(int) * output_size

        volume_id = f"vol-{uuid.uuid4().hex[:8]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        _, corrections_dir = ensure_corrections_storage(output_path)
        zarr_path = os.path.join(corrections_dir, f"{volume_id}.zarr")

        success, zarr_info = create_annotation_volume_zarr(
            zarr_path=zarr_path,
            dataset_shape_voxels=dataset_shape_voxels,
            output_voxel_size=effective_output_voxel_size,
            dataset_offset_nm=dataset_offset_nm,
            chunk_size=output_size,
            dataset_path=dataset_path,
            model_name=model_name,
            input_size=input_size,
            input_voxel_size=effective_input_voxel_size,
            claimed_output_voxel_size=claimed_output_voxel_size,
            claimed_input_voxel_size=claimed_input_voxel_size,
            input_norm_config=current_input_norm_config(),
            postprocess_config=current_postprocess_config(),
        )
        if not success:
            return jsonify({"success": False, "error": zarr_info}), 500

        minio_url = ensure_minio_serving(zarr_path, volume_id, output_base_dir=corrections_dir)
        minio_url = rewrite_minio_url_for_proxy(minio_url, request)
        _register_annotation_volume(
            volume_id,
            zarr_path=zarr_path,
            model_name=model_name,
            output_size=output_size.tolist(),
            input_size=input_size.tolist(),
            input_voxel_size=effective_input_voxel_size.tolist(),
            output_voxel_size=effective_output_voxel_size.tolist(),
            claimed_input_voxel_size=claimed_input_voxel_size.tolist(),
            claimed_output_voxel_size=claimed_output_voxel_size.tolist(),
            dataset_path=dataset_path,
            dataset_offset_nm=dataset_offset_nm.tolist(),
            corrections_dir=corrections_dir,
        )
        # Without this the trainer falls back to the legacy per-chunk dataset
        # and any good regions marked in this session are ignored.
        write_volume_manifest(g.annotation_volumes[volume_id])
        refresh_annotated_regions_layer()

        return jsonify(
            {
                "success": True,
                "volume_id": volume_id,
                "zarr_path": zarr_path,
                "minio_url": minio_url,
                "neuroglancer_url": f"{minio_url}/annotation",
                "metadata": {
                    "dataset_shape_voxels": dataset_shape_voxels.tolist(),
                    "chunk_size": output_size.tolist(),
                    "output_voxel_size": effective_output_voxel_size.tolist(),
                    "claimed_output_voxel_size": claimed_output_voxel_size.tolist(),
                    "dataset_offset_nm": dataset_offset_nm.tolist(),
                },
            }
        )
    except Exception as e:
        logger.error(f"Error creating annotation volume: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


def get_user_prefs_response():
    return jsonify({"success": True, "prefs": load_user_prefs()})


def set_user_prefs_response(data):
    try:
        prefs = load_user_prefs()
        prefs.update({key: value for key, value in data.items() if value is not None})
        save_user_prefs(prefs)
        return jsonify({"success": True, "prefs": prefs})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
