import json
import os
import subprocess
import time
import uuid
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import zarr
import neuroglancer
from flask import Blueprint, request, jsonify, Response

from cellmap_flow.globals import g
from cellmap_flow.utils.load_py import load_safe_config
from cellmap_flow.globals import g
from cellmap_flow.dashboard.finetune_utils import (
    get_or_create_session_path,
    create_correction_zarr,
    create_annotation_volume_zarr,
    ensure_minio_serving,
    sync_annotation_from_minio,
    sync_all_annotations_from_minio,
)

logger = logging.getLogger(__name__)

finetune_bp = Blueprint("finetune", __name__)


@finetune_bp.route("/api/finetune/models", methods=["GET"])
def get_finetune_models():
    """Get available models for finetuning with their configurations."""
    try:
        models = []

        if hasattr(g, "models_config") and g.models_config:
            for model_config in g.models_config:
                try:
                    config = model_config.config
                    models.append(
                        {
                            "name": model_config.name,
                            "write_shape": list(config.write_shape),
                            "output_voxel_size": list(config.output_voxel_size),
                            "output_channels": config.output_channels,
                        }
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not extract config for {model_config.name}: {e}"
                    )

        if len(models) == 0 and hasattr(g, "jobs") and g.jobs:
            logger.warning("No models in g.models_config, checking running jobs")
            for job in g.jobs:
                if hasattr(job, "model_name"):
                    job_model_name = job.model_name
                    if (
                        hasattr(g, "pipeline_model_configs")
                        and job_model_name in g.pipeline_model_configs
                    ):
                        config_dict = g.pipeline_model_configs[job_model_name]
                        try:
                            models.append(
                                {
                                    "name": job_model_name,
                                    "write_shape": config_dict.get("write_shape", []),
                                    "output_voxel_size": config_dict.get(
                                        "output_voxel_size", []
                                    ),
                                    "output_channels": config_dict.get(
                                        "output_channels", 1
                                    ),
                                }
                            )
                            logger.info(
                                f"Found config for {job_model_name} in pipeline_model_configs"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Could not extract config for {job_model_name}: {e}"
                            )
                    else:
                        logger.warning(
                            f"No configuration found for running job: {job_model_name}"
                        )

        selected = models[0]["name"] if len(models) == 1 else None

        return jsonify({"models": models, "selected_model": selected})

    except Exception as e:
        logger.error(f"Error getting finetune models: {e}")
        return jsonify({"error": str(e)}), 500


@finetune_bp.route("/api/finetune/view-center", methods=["GET"])
def get_view_center():
    """Get current view center position from Neuroglancer viewer."""
    try:
        if not hasattr(g, "viewer") or g.viewer is None:
            return jsonify({"success": False, "error": "Viewer not initialized"}), 400

        with g.viewer.txn() as s:
            position = s.position

            dimensions = s.dimensions
            scales_nm = None

            if dimensions and hasattr(dimensions, "scales"):
                scales_nm = list(dimensions.scales)
                logger.info(f"Viewer scales (raw): {scales_nm}")

                if hasattr(dimensions, "units"):
                    units = dimensions.units
                    if isinstance(units, str):
                        units = [units] * len(scales_nm)

                    converted_scales = []
                    for scale, unit in zip(scales_nm, units):
                        if unit == "m":
                            converted_scales.append(scale * 1e9)
                        elif unit == "nm":
                            converted_scales.append(scale)
                        else:
                            logger.warning(f"Unknown unit: {unit}, assuming nm")
                            converted_scales.append(scale)
                    scales_nm = converted_scales

                logger.info(f"Viewer scales (nm): {scales_nm}")
            else:
                logger.warning("Could not extract scales from viewer dimensions")

            if hasattr(position, "tolist"):
                position = position.tolist()
            elif hasattr(position, "__iter__"):
                position = list(position)

            logger.info(f"Got view center position: {position}")

            return jsonify(
                {"success": True, "position": position, "scales_nm": scales_nm}
            )

    except Exception as e:
        logger.error(f"Error getting view center position: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@finetune_bp.route("/api/finetune/create-crop", methods=["POST"])
def create_annotation_crop():
    """Create an annotation crop centered at view center position."""
    try:
        from cellmap_flow.image_data_interface import ImageDataInterface
        from funlib.geometry import Roi, Coordinate

        data = request.get_json()
        model_name = data.get("model_name")
        output_path = data.get("output_path")

        if not hasattr(g, "models_config") or not g.models_config:
            return jsonify({"success": False, "error": "No models loaded"}), 400

        if not hasattr(g, "viewer") or g.viewer is None:
            return jsonify({"success": False, "error": "Viewer not initialized"}), 400

        with g.viewer.txn() as s:
            position = s.position

            dimensions = s.dimensions
            viewer_scales_nm = None

            if dimensions and hasattr(dimensions, "scales"):
                scales_nm = list(dimensions.scales)

                if hasattr(dimensions, "units"):
                    units = dimensions.units
                    if isinstance(units, str):
                        units = [units] * len(scales_nm)

                    converted_scales = []
                    for scale, unit in zip(scales_nm, units):
                        if unit == "m":
                            converted_scales.append(scale * 1e9)
                        elif unit == "nm":
                            converted_scales.append(scale)
                        else:
                            logger.warning(f"Unknown unit: {unit}, assuming nm")
                            converted_scales.append(scale)
                    viewer_scales_nm = converted_scales
                else:
                    viewer_scales_nm = scales_nm

            if hasattr(position, "tolist"):
                view_center = position.tolist()
            elif hasattr(position, "__iter__"):
                view_center = list(position)
            else:
                view_center = position

            view_center = np.array(view_center)

        logger.info(f"Auto-detected view center: {view_center}")
        logger.info(f"Auto-detected viewer scales: {viewer_scales_nm} nm")

        # Find model config
        model_config = None
        for mc in g.models_config:
            if mc.name == model_name:
                model_config = mc
                break

        if not model_config:
            return (
                jsonify({"success": False, "error": f"Model {model_name} not found"}),
                404,
            )

        config = model_config.config
        read_shape = np.array(config.read_shape)
        write_shape = np.array(config.write_shape)
        input_voxel_size = np.array(config.input_voxel_size)
        output_voxel_size = np.array(config.output_voxel_size)
        output_channels = config.output_channels

        if viewer_scales_nm is not None:
            viewer_scales_nm = np.array(viewer_scales_nm)
            view_center_nm = view_center * viewer_scales_nm
            logger.info(
                f"Converted view center from {view_center} (viewer coords) to {view_center_nm} nm"
            )
        else:
            view_center_nm = view_center
            logger.warning(
                "No viewer scales provided, assuming view center is already in nm"
            )

        raw_crop_shape_voxels = (read_shape / input_voxel_size).astype(int)
        annotation_crop_shape_voxels = (write_shape / output_voxel_size).astype(int)

        half_read_shape = read_shape / 2
        raw_crop_offset_nm = view_center_nm - half_read_shape
        raw_crop_offset_voxels = (raw_crop_offset_nm / input_voxel_size).astype(int)

        half_write_shape = write_shape / 2
        annotation_crop_offset_nm = view_center_nm - half_write_shape
        annotation_crop_offset_voxels = (
            annotation_crop_offset_nm / output_voxel_size
        ).astype(int)

        crop_id = f"{uuid.uuid4().hex[:8]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        if output_path:
            session_path = get_or_create_session_path(output_path)
            corrections_dir = os.path.join(session_path, "corrections")
            os.makedirs(corrections_dir, exist_ok=True)
            zarr.open_group(corrections_dir, mode="a")
            zarr_path = os.path.join(corrections_dir, f"{crop_id}.zarr")
            logger.info(f"Using session path: {session_path}")
            logger.info(f"Corrections directory: {corrections_dir}")
        else:
            corrections_dir = os.path.expanduser("~/.cellmap_flow/corrections")
            os.makedirs(corrections_dir, exist_ok=True)
            zarr.open_group(corrections_dir, mode="a")
            zarr_path = os.path.join(corrections_dir, f"{crop_id}.zarr")

        dataset_path = getattr(g, "dataset_path", "unknown")

        logger.info(f"Creating ImageDataInterface for {dataset_path}")
        logger.info(f"Using input voxel size: {input_voxel_size} nm")
        try:
            idi = ImageDataInterface(dataset_path, voxel_size=input_voxel_size)
            raw_dtype = str(idi.ts.dtype)
            logger.info(f"Dataset dtype: {raw_dtype}")
        except Exception as e:
            logger.error(f"Error creating ImageDataInterface: {e}")
            return (
                jsonify(
                    {"success": False, "error": f"Failed to access dataset: {str(e)}"}
                ),
                500,
            )

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

        logger.info(f"Reading raw data from {dataset_path}")
        try:
            roi = Roi(
                offset=Coordinate(view_center_nm - read_shape / 2),
                shape=Coordinate(read_shape),
            )
            logger.info(f"Reading ROI: offset={roi.offset}, shape={roi.shape}")

            raw_data = idi.to_ndarray_ts(roi)
            logger.info(
                f"Read raw data with shape: {raw_data.shape}, dtype: {raw_data.dtype}"
            )

            raw_zarr = zarr.open(zarr_path, mode="r+")
            raw_zarr["raw/s0"][:] = raw_data
            logger.info(f"Wrote raw data to {zarr_path}/raw/s0")

        except Exception as e:
            logger.error(f"Error reading/writing raw data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return (
                jsonify(
                    {"success": False, "error": f"Failed to read raw data: {str(e)}"}
                ),
                500,
            )

        minio_url = ensure_minio_serving(
            zarr_path, crop_id, output_base_dir=corrections_dir
        )

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

    except Exception as e:
        logger.error(f"Error creating annotation crop: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@finetune_bp.route("/api/finetune/create-volume", methods=["POST"])
def create_annotation_volume():
    """Create a sparse annotation volume covering the full dataset extent."""
    try:
        from cellmap_flow.image_data_interface import ImageDataInterface
        from funlib.geometry import Coordinate

        data = request.get_json()
        model_name = data.get("model_name")
        output_path = data.get("output_path")

        if not hasattr(g, "models_config") or not g.models_config:
            return jsonify({"success": False, "error": "No models loaded"}), 400

        model_config = None
        for mc in g.models_config:
            if mc.name == model_name:
                model_config = mc
                break

        if not model_config:
            return (
                jsonify({"success": False, "error": f"Model {model_name} not found"}),
                404,
            )

        config = model_config.config
        read_shape = np.array(config.read_shape)
        write_shape = np.array(config.write_shape)
        input_voxel_size = np.array(config.input_voxel_size)
        output_voxel_size = np.array(config.output_voxel_size)

        output_size = (write_shape / output_voxel_size).astype(int)
        input_size = (read_shape / input_voxel_size).astype(int)

        dataset_path = getattr(g, "dataset_path", None)
        if not dataset_path:
            return (
                jsonify({"success": False, "error": "No dataset path configured"}),
                400,
            )

        logger.info(f"Getting dataset extent from {dataset_path}")
        try:
            idi = ImageDataInterface(dataset_path, voxel_size=output_voxel_size)
            dataset_roi = idi.roi
            dataset_offset_nm = np.array(dataset_roi.offset)
            dataset_shape_nm = np.array(dataset_roi.shape)

            dataset_shape_voxels = (dataset_shape_nm / output_voxel_size).astype(int)
            dataset_shape_voxels = (
                np.ceil(dataset_shape_voxels / output_size).astype(int) * output_size
            )

            logger.info(
                f"Dataset extent: offset={dataset_offset_nm} nm, "
                f"shape={dataset_shape_voxels} voxels (at {output_voxel_size} nm/voxel)"
            )
        except Exception as e:
            logger.error(f"Error getting dataset extent: {e}")
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Failed to access dataset: {str(e)}",
                    }
                ),
                500,
            )

        volume_id = (
            f"vol-{uuid.uuid4().hex[:8]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )

        if output_path:
            session_path = get_or_create_session_path(output_path)
            corrections_dir = os.path.join(session_path, "corrections")
            os.makedirs(corrections_dir, exist_ok=True)
            zarr.open_group(corrections_dir, mode="a")
            zarr_path = os.path.join(corrections_dir, f"{volume_id}.zarr")
            logger.info(f"Using session path: {session_path}")
        else:
            corrections_dir = os.path.expanduser("~/.cellmap_flow/corrections")
            os.makedirs(corrections_dir, exist_ok=True)
            zarr.open_group(corrections_dir, mode="a")
            zarr_path = os.path.join(corrections_dir, f"{volume_id}.zarr")

        success, zarr_info = create_annotation_volume_zarr(
            zarr_path=zarr_path,
            dataset_shape_voxels=dataset_shape_voxels,
            output_voxel_size=output_voxel_size,
            dataset_offset_nm=dataset_offset_nm,
            chunk_size=output_size,
            dataset_path=dataset_path,
            model_name=model_name,
            input_size=input_size,
            input_voxel_size=input_voxel_size,
        )

        if not success:
            return jsonify({"success": False, "error": zarr_info}), 500

        minio_url = ensure_minio_serving(
            zarr_path, volume_id, output_base_dir=corrections_dir
        )

        g.annotation_volumes[volume_id] = {
            "zarr_path": zarr_path,
            "model_name": model_name,
            "output_size": output_size.tolist(),
            "input_size": input_size.tolist(),
            "input_voxel_size": input_voxel_size.tolist(),
            "output_voxel_size": output_voxel_size.tolist(),
            "dataset_path": dataset_path,
            "dataset_offset_nm": dataset_offset_nm.tolist(),
            "corrections_dir": corrections_dir,
            "extracted_chunks": set(),
            "chunk_sync_state": {},
        }

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
                    "output_voxel_size": output_voxel_size.tolist(),
                    "dataset_offset_nm": dataset_offset_nm.tolist(),
                },
            }
        )

    except Exception as e:
        logger.error(f"Error creating annotation volume: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@finetune_bp.route("/api/finetune/add-to-viewer", methods=["POST"])
def add_crop_to_viewer():
    """Add annotation crop or volume layer to Neuroglancer viewer."""
    try:
        data = request.get_json()
        crop_id = data.get("crop_id")
        minio_url = data.get("minio_url")

        if not hasattr(g, "viewer") or g.viewer is None:
            return jsonify({"success": False, "error": "Viewer not initialized"}), 400

        with g.viewer.txn() as s:
            layer_name = data.get("layer_name", f"annotation_{crop_id}")
            source_config = {
                "url": f"s3+{minio_url}",
                "subsources": {"default": {"writingEnabled": True}, "bounds": {}},
            }
            layer = neuroglancer.SegmentationLayer(source=source_config)
            s.layers[layer_name] = layer

        logger.info(f"Added layer {layer_name} to viewer")

        return jsonify(
            {
                "success": True,
                "message": "Layer added to viewer",
                "layer_name": layer_name,
            }
        )

    except Exception as e:
        logger.error(f"Error adding layer to viewer: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@finetune_bp.route("/api/finetune/sync-annotations", methods=["POST"])
def sync_annotations_manually():
    """Manually trigger sync of annotations from MinIO to local disk."""
    try:
        data = request.get_json()
        crop_id = data.get("crop_id", None)
        force = data.get("force", True)

        if crop_id:
            success = sync_annotation_from_minio(crop_id, force=force)
            if success:
                return jsonify(
                    {"success": True, "message": f"Synced annotation for {crop_id}"}
                )
            else:
                return jsonify(
                    {"success": False, "message": f"No updates to sync for {crop_id}"}
                )
        else:
            synced = sync_all_annotations_from_minio(force=force)
            if synced == -1:
                return (
                    jsonify({"success": False, "error": "MinIO not initialized"}),
                    400,
                )
            return jsonify(
                {
                    "success": True,
                    "message": f"Synced {synced} annotations",
                    "synced_count": synced,
                }
            )

    except Exception as e:
        logger.error(f"Error in sync endpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@finetune_bp.route("/api/finetune/submit", methods=["POST"])
def submit_finetuning():
    """Submit a finetuning job to the LSF cluster."""
    try:
        data = request.get_json()
        model_name = data.get("model_name")
        corrections_path_str = data.get("corrections_path")
        lora_r = data.get("lora_r", 8)
        num_epochs = data.get("num_epochs", 10)
        batch_size = data.get("batch_size", 2)
        learning_rate = data.get("learning_rate", 1e-4)
        checkpoint_path_override = data.get("checkpoint_path")
        auto_serve = data.get("auto_serve", True)
        loss_type = data.get("loss_type", "mse")
        label_smoothing = data.get("label_smoothing", 0.1)
        distillation_lambda = data.get("distillation_lambda", 0.0)
        distillation_scope = data.get("distillation_scope", "unlabeled")
        margin = data.get("margin", 0.3)
        balance_classes = data.get("balance_classes", False)
        queue = data.get("queue", "gpu_h100")
        output_type = data.get("output_type", None)  # None = auto-detect
        select_channel = data.get("select_channel", None)
        offsets = data.get("offsets", None)

        if not model_name:
            return jsonify({"success": False, "error": "model_name is required"}), 400

        if not corrections_path_str:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "corrections_path is required. Please specify the output path where annotation crops are saved.",
                    }
                ),
                400,
            )

        model_config = None
        for config in g.models_config:
            if config.name == model_name:
                model_config = config
                break

        if not model_config:
            return (
                jsonify({"success": False, "error": f"Model {model_name} not found"}),
                404,
            )

        base_corrections_path = Path(corrections_path_str)

        actual_corrections_path = None
        if (
            base_corrections_path.name == "corrections"
            and base_corrections_path.exists()
        ):
            actual_corrections_path = base_corrections_path
            session_path = base_corrections_path.parent
        else:
            session_path = get_or_create_session_path(str(base_corrections_path))
            actual_corrections_path = Path(session_path) / "corrections"

        if not actual_corrections_path.exists():
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Corrections path does not exist: {actual_corrections_path}. Please create annotation crops first.",
                    }
                ),
                400,
            )

        output_base = Path(session_path)
        logger.info(f"Using session path for finetuning: {session_path}")
        logger.info(f"Corrections path: {actual_corrections_path}")

        try:
            sync_all_annotations_from_minio(force=False)
        except Exception as e:
            logger.warning(f"Error syncing annotations before training: {e}")

        # Detect sparse annotations
        has_sparse = False
        try:
            for p in actual_corrections_path.iterdir():
                if p.suffix == ".zarr" and (p / ".zattrs").exists():
                    attrs = json.loads((p / ".zattrs").read_text())
                    if attrs.get("source") == "sparse_volume":
                        has_sparse = True
                        break
        except Exception as e:
            logger.warning(f"Error checking for sparse annotations: {e}")

        sparse_auto_switched = False
        if has_sparse:
            logger.info("Detected sparse annotations, will use mask_unannotated=True")
            if loss_type == "mse":
                loss_type = "margin"
                distillation_lambda = 0.5
                sparse_auto_switched = True
                logger.info(
                    "Auto-switched to margin loss + distillation (lambda=0.5) for sparse annotations"
                )

        # Auto-detect output_type and offsets from model script
        from cellmap_flow.finetune.finetune_cli import _read_offsets_from_script
        if output_type is None:
            # Try to auto-detect from model script
            if hasattr(model_config, 'script_path'):
                script_offsets = _read_offsets_from_script(model_config.script_path)
                if script_offsets is not None:
                    output_type = "affinities"
                    offsets = json.dumps(script_offsets)
                    logger.info(
                        f"Auto-detected output_type='affinities' with "
                        f"{len(script_offsets)} offsets from model script"
                    )
            if output_type is None:
                output_type = "binary"

        if output_type == "affinities" and offsets is None:
            if hasattr(model_config, 'script_path'):
                offsets = _read_offsets_from_script(model_config.script_path)
                if offsets is not None:
                    logger.info(f"Auto-detected {len(offsets)} offsets from model script")
                    offsets = json.dumps(offsets)
            if offsets is None:
                return jsonify({
                    "success": False,
                    "error": "output_type='affinities' requires offsets. "
                             "Define 'offsets' in the model script or pass them in the request."
                }), 400
        elif isinstance(offsets, list):
            offsets = json.dumps(offsets)

        finetune_job = g.finetune_job_manager.submit_finetuning_job(
            model_config=model_config,
            corrections_path=actual_corrections_path,
            lora_r=lora_r,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_base=output_base,
            checkpoint_path_override=(
                Path(checkpoint_path_override) if checkpoint_path_override else None
            ),
            auto_serve=auto_serve,
            mask_unannotated=has_sparse,
            loss_type=loss_type,
            label_smoothing=label_smoothing,
            distillation_lambda=distillation_lambda,
            distillation_scope=distillation_scope,
            margin=margin,
            balance_classes=balance_classes,
            queue=queue,
            output_type=output_type,
            select_channel=select_channel,
            offsets=offsets,
        )

        logger.info(f"Submitted finetuning job: {finetune_job.job_id}")

        lsf_job_id = None
        if finetune_job.lsf_job:
            if hasattr(finetune_job.lsf_job, "job_id"):
                lsf_job_id = finetune_job.lsf_job.job_id
            elif hasattr(finetune_job.lsf_job, "process"):
                lsf_job_id = f"PID:{finetune_job.lsf_job.process.pid}"

        response = {
            "success": True,
            "job_id": finetune_job.job_id,
            "lsf_job_id": lsf_job_id,
            "output_dir": str(finetune_job.output_dir),
            "message": "Finetuning job submitted successfully",
        }
        if sparse_auto_switched:
            response["note"] = (
                "Auto-switched to margin loss + distillation (lambda=0.5) for sparse annotations"
            )

        return jsonify(response)

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error submitting finetuning job: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@finetune_bp.route("/api/finetune/jobs", methods=["GET"])
def get_finetuning_jobs():
    """Get list of all finetuning jobs."""
    try:
        jobs = g.finetune_job_manager.list_jobs()
        return jsonify({"success": True, "jobs": jobs})
    except Exception as e:
        logger.error(f"Error getting jobs: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@finetune_bp.route("/api/finetune/job/<job_id>/status", methods=["GET"])
def get_job_status(job_id):
    """Get detailed status of a specific job."""
    try:
        status = g.finetune_job_manager.get_job_status(job_id)
        if status is None:
            return jsonify({"success": False, "error": "Job not found"}), 404

        return jsonify({"success": True, **status})
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@finetune_bp.route("/api/finetune/job/<job_id>/logs", methods=["GET"])
def get_job_logs(job_id):
    """Get training logs for a specific job."""
    try:
        logs = g.finetune_job_manager.get_job_logs(job_id)
        if logs is None:
            return jsonify({"success": False, "error": "Job not found"}), 404

        return jsonify({"success": True, "logs": logs})
    except Exception as e:
        logger.error(f"Error getting job logs: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@finetune_bp.route("/api/finetune/job/<job_id>/logs/stream", methods=["GET"])
def stream_job_logs(job_id):
    """Server-Sent Events stream for live training logs."""

    import re as _re

    _log_filters = [
        _re.compile(r"^\s+base_model\.\S+\.lora_"),
        _re.compile(r"^INFO:werkzeug:"),
        _re.compile(r"^Array metadata \(scale="),
        _re.compile(r"^Host name:"),
        _re.compile(r"^DEBUG trainer:"),
    ]

    def _should_show(line):
        for pat in _log_filters:
            if pat.search(line):
                return False
        return True

    def _iter_visible_lines(text):
        for line in text.splitlines():
            if line and _should_show(line):
                yield line

    def _sse_data_block(lines):
        if not lines:
            return None
        payload = "\n".join(lines)
        return "data: " + payload.replace("\n", "\ndata: ") + "\n\n"

    def _read_bpeek_content(lsf_job_id):
        try:
            result = subprocess.run(
                ["bpeek", str(lsf_job_id)],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception as e:
            logger.debug(f"bpeek call failed for job {lsf_job_id}: {e}")
            return None

        output = result.stdout or ""
        stderr = (result.stderr or "").strip()
        if stderr and "Not yet started" not in stderr:
            logger.debug(f"bpeek stderr for job {lsf_job_id}: {stderr}")
        return output

    def generate():
        heartbeat_interval_s = 1.0
        last_heartbeat = time.perf_counter()

        fjm = g.finetune_job_manager
        if job_id not in fjm.jobs:
            yield f"data: Job {job_id} not found\n\n"
            return

        finetune_job = fjm.jobs[job_id]
        lsf_job_id = None
        if finetune_job.lsf_job and hasattr(finetune_job.lsf_job, "job_id"):
            lsf_job_id = finetune_job.lsf_job.job_id

        use_bpeek = lsf_job_id is not None
        last_bpeek_line_count = 0
        last_bpeek_poll = 0.0
        bpeek_poll_interval_s = 0.25

        # Send existing content first
        if use_bpeek:
            initial = _read_bpeek_content(lsf_job_id)
            if initial is None:
                use_bpeek = False
            else:
                initial_lines = initial.splitlines()
                last_bpeek_line_count = len(initial_lines)
                block = _sse_data_block(list(_iter_visible_lines(initial)))
                if block:
                    yield block

        if not use_bpeek and finetune_job.log_file.exists():
            try:
                with open(finetune_job.log_file, "r") as f:
                    existing_content = f.read()
                block = _sse_data_block(list(_iter_visible_lines(existing_content)))
                if block:
                    yield block
            except Exception as e:
                logger.error(f"Error reading log file: {e}")

        last_position = (
            finetune_job.log_file.stat().st_size
            if finetune_job.log_file.exists()
            else 0
        )

        while finetune_job.status.value in ["PENDING", "RUNNING"]:
            try:
                now = time.perf_counter()

                if (
                    use_bpeek
                    and lsf_job_id
                    and now - last_bpeek_poll >= bpeek_poll_interval_s
                ):
                    last_bpeek_poll = now
                    content = _read_bpeek_content(lsf_job_id)
                    if content is None:
                        use_bpeek = False
                    else:
                        current_lines = content.splitlines()
                        if len(current_lines) < last_bpeek_line_count:
                            delta_lines = current_lines
                        else:
                            delta_lines = current_lines[last_bpeek_line_count:]
                        last_bpeek_line_count = len(current_lines)
                        if delta_lines:
                            delta_text = "\n".join(delta_lines)
                            block = _sse_data_block(
                                list(_iter_visible_lines(delta_text))
                            )
                            if block:
                                yield block

                if not use_bpeek and finetune_job.log_file.exists():
                    with open(finetune_job.log_file, "r") as f:
                        f.seek(last_position)
                        new_content = f.read()
                        last_position = f.tell()
                    if new_content:
                        block = _sse_data_block(
                            list(_iter_visible_lines(new_content))
                        )
                        if block:
                            yield block

                if now - last_heartbeat >= heartbeat_interval_s:
                    yield ": ping\n\n"
                    last_heartbeat = now

                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error streaming logs: {e}")
                break

        yield f"data: === Training {finetune_job.status.value} ===\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@finetune_bp.route("/api/finetune/job/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id):
    """Cancel a running finetuning job."""
    try:
        success = g.finetune_job_manager.cancel_job(job_id)

        if success:
            return jsonify({"success": True, "message": f"Job {job_id} cancelled"})
        else:
            return jsonify({"success": False, "error": "Failed to cancel job"}), 400

    except Exception as e:
        logger.error(f"Error cancelling job: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@finetune_bp.route("/api/finetune/job/<job_id>/inference-server", methods=["GET"])
def get_inference_server_status(job_id):
    """Get inference server status for a finetuning job."""
    try:
        job = g.finetune_job_manager.get_job(job_id)
        if not job:
            return jsonify({"success": False, "error": "Job not found"}), 404

        return jsonify(
            {
                "success": True,
                "ready": job.inference_server_ready,
                "url": job.inference_server_url,
                "model_name": job.finetuned_model_name,
                "model_script_path": (
                    str(job.model_script_path) if job.model_script_path else None
                ),
            }
        )

    except Exception as e:
        logger.error(f"Error getting inference server status: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@finetune_bp.route("/api/viewer/add-finetuned-layer", methods=["POST"])
def add_finetuned_layer_to_viewer():
    """Add finetuned model layer to Neuroglancer viewer and register model in system."""
    try:
        data = request.get_json()
        server_url = data.get("server_url")
        model_name = data.get("model_name")
        model_script_path = data.get("model_script_path")

        if not server_url or not model_name:
            return (
                jsonify(
                    {"success": False, "error": "Missing server_url or model_name"}
                ),
                400,
            )

        logger.info(f"Registering finetuned model: {model_name} at {server_url}")

        # 1. Load model config from script if provided
        if model_script_path and Path(model_script_path).exists():
            try:
                model_config = load_safe_config(model_script_path)

                if not hasattr(g, "models_config"):
                    g.models_config = []

                base_model_name = (
                    model_name.rsplit("_finetuned_", 1)[0]
                    if "_finetuned_" in model_name
                    else model_name
                )
                g.models_config = [
                    mc
                    for mc in g.models_config
                    if not (
                        hasattr(mc, "name")
                        and mc.name.startswith(f"{base_model_name}_finetuned")
                    )
                ]

                g.models_config.append(model_config)
                logger.info(f"Loaded model config from {model_script_path}")
            except Exception as e:
                logger.warning(f"Could not load model config: {e}")

        # 2. Add to model_catalog under "Finetuned" group
        if not hasattr(g, "model_catalog"):
            g.model_catalog = {}

        if "Finetuned" not in g.model_catalog:
            g.model_catalog["Finetuned"] = {}

        base_model_name = (
            model_name.rsplit("_finetuned_", 1)[0]
            if "_finetuned_" in model_name
            else model_name
        )
        g.model_catalog["Finetuned"] = {
            name: path
            for name, path in g.model_catalog["Finetuned"].items()
            if not name.startswith(f"{base_model_name}_finetuned")
        }

        g.model_catalog["Finetuned"][model_name] = (
            model_script_path if model_script_path else ""
        )
        logger.info(f"Added to model catalog: Finetuned/{model_name}")

        # 3. Create a Job object for the running inference server
        from cellmap_flow.utils.bsub_utils import LSFJob

        finetune_job = None
        for job_id, ft_job in g.finetune_job_manager.jobs.items():
            if ft_job.finetuned_model_name == model_name:
                finetune_job = ft_job
                break

        if finetune_job and finetune_job.job_id:
            inference_job = LSFJob(
                job_id=finetune_job.job_id, model_name=model_name
            )
            inference_job.host = server_url
            inference_job.status = finetune_job.status

            g.jobs = [
                j
                for j in g.jobs
                if not (
                    hasattr(j, "model_name")
                    and j.model_name
                    and j.model_name.startswith(f"{base_model_name}_finetuned")
                )
            ]

            g.jobs.append(inference_job)
            logger.info(
                f"Created Job object for {model_name} with job_id {finetune_job.job_id}"
            )
        else:
            logger.warning(
                f"Could not find finetune job for {model_name}, Job object not created"
            )

        # 4. Add neuroglancer layer
        layer_name = model_name

        with g.viewer.txn() as s:
            if layer_name in s.layers:
                logger.info(f"Removing old finetuned layer: {layer_name}")
                del s.layers[layer_name]

            from cellmap_flow.utils.neuroglancer_utils import get_norms_post_args
            from cellmap_flow.utils.web_utils import ARGS_KEY

            st_data = get_norms_post_args(g.input_norms, g.postprocess)

            layer_source = f"zarr://{server_url}/{model_name}{ARGS_KEY}{st_data}{ARGS_KEY}"
            s.layers[layer_name] = neuroglancer.ImageLayer(
                source=layer_source,
                shader="""#uicontrol invlerp normalized(range=[0, 255], window=[0, 255]);
                    #uicontrol vec3 color color(default="red");
                    void main(){emitRGB(color * normalized());}""",
            )

        logger.info(f"Added neuroglancer layer: {layer_name} -> {server_url}")

        return jsonify(
            {
                "success": True,
                "layer_name": layer_name,
                "model_name": model_name,
                "reload_page": True,
            }
        )

    except Exception as e:
        logger.error(f"Error adding finetuned layer: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@finetune_bp.route("/api/finetune/job/<job_id>/restart", methods=["POST"])
def restart_finetuning_job(job_id):
    """Restart training on the same GPU via in-process control channel."""
    try:
        restart_t0 = time.perf_counter()
        data = request.get_json() or {}

        # Sync annotations from MinIO before restarting training
        try:
            sync_t0 = time.perf_counter()
            synced = sync_all_annotations_from_minio(force=False)
            sync_elapsed = time.perf_counter() - sync_t0
            logger.info(
                f"Restart pre-sync complete for job {job_id}: synced={synced}, "
                f"elapsed={sync_elapsed:.2f}s"
            )
        except Exception as e:
            logger.warning(f"Error syncing annotations before restart: {e}")

        updated_params = {}
        passthrough_keys = [
            "lora_r",
            "lora_alpha",
            "num_epochs",
            "batch_size",
            "learning_rate",
            "loss_type",
            "label_smoothing",
            "distillation_lambda",
            "margin",
            "balance_classes",
            "mask_unannotated",
            "gradient_accumulation_steps",
            "num_workers",
            "no_augment",
            "no_mixed_precision",
            "patch_shape",
            "output_type",
            "select_channel",
            "offsets",
        ]
        for key in passthrough_keys:
            if key in data and data[key] is not None:
                updated_params[key] = data[key]

        # UI uses distillation_scope; CLI expects distillation_all_voxels.
        if "distillation_scope" in data and data["distillation_scope"] is not None:
            scope = str(data["distillation_scope"]).lower()
            if scope in {"all", "unlabeled"}:
                updated_params["distillation_all_voxels"] = scope == "all"
            else:
                logger.warning(
                    f"Ignoring invalid distillation_scope on restart for job {job_id}: {data['distillation_scope']}"
                )

        signal_t0 = time.perf_counter()
        job = g.finetune_job_manager.restart_finetuning_job(
            job_id=job_id, updated_params=updated_params
        )
        signal_elapsed = time.perf_counter() - signal_t0
        total_elapsed = time.perf_counter() - restart_t0
        logger.info(
            f"Restart request processed for job {job_id}: "
            f"signal_write={signal_elapsed:.2f}s total={total_elapsed:.2f}s"
        )

        return jsonify(
            {
                "success": True,
                "job_id": job.job_id,
                "message": "Restart request sent. Training will restart on the same GPU.",
            }
        )

    except Exception as e:
        logger.error(f"Error restarting job: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
