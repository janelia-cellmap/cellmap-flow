import logging
import os
import uuid
from datetime import datetime

import numpy as np
import zarr
from flask import jsonify

from cellmap_flow.dashboard.finetune_utils import (
    create_annotation_volume_zarr,
    create_correction_zarr,
    ensure_minio_serving,
)
from cellmap_flow.dashboard.routes.finetune.common import (
    ensure_corrections_storage,
    find_model_config,
    load_user_prefs,
    save_user_prefs,
    viewer_position_and_scales,
)
from cellmap_flow.dashboard.routes.finetune.overlay import refresh_annotated_regions_layer
from cellmap_flow.globals import g

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


def get_finetune_models_response():
    try:
        models = []
        for model_config in getattr(g, "models_config", []) or []:
            try:
                info = model_config.lightweight_info()
                models.append(
                    {
                        "name": model_config.name,
                        "write_shape": info["write_shape"],
                        "output_voxel_size": info["output_voxel_size"],
                        "output_channels": info["output_channels"],
                    }
                )
            except Exception as e:
                logger.warning(f"Could not extract info for {model_config.name}: {e}")

        if not models and hasattr(g, "jobs") and g.jobs:
            logger.warning("No models in g.models_config, checking running jobs")
            for job in g.jobs:
                job_model_name = getattr(job, "model_name", None)
                if not job_model_name:
                    continue
                if hasattr(g, "pipeline_model_configs") and job_model_name in g.pipeline_model_configs:
                    config_dict = g.pipeline_model_configs[job_model_name]
                    try:
                        models.append(
                            {
                                "name": job_model_name,
                                "write_shape": config_dict.get("write_shape", []),
                                "output_voxel_size": config_dict.get("output_voxel_size", []),
                                "output_channels": config_dict.get("output_channels", 1),
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Could not extract config for {job_model_name}: {e}")
                else:
                    logger.warning(f"No configuration found for running job: {job_model_name}")

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

        info = model_config.lightweight_info()
        read_shape = np.array(info["read_shape"])
        write_shape = np.array(info["write_shape"])
        input_voxel_size = np.array(info["input_voxel_size"])
        output_voxel_size = np.array(info["output_voxel_size"])
        output_channels = info["output_channels"]

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

        info = model_config.lightweight_info()
        read_shape = np.array(info["read_shape"])
        write_shape = np.array(info["write_shape"])
        claimed_input_voxel_size = np.array(info["input_voxel_size"])
        claimed_output_voxel_size = np.array(info["output_voxel_size"])
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
        )
        if not success:
            return jsonify({"success": False, "error": zarr_info}), 500

        minio_url = ensure_minio_serving(zarr_path, volume_id, output_base_dir=corrections_dir)
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
