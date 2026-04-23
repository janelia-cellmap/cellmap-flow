import json
import logging
import os
import uuid
from datetime import datetime

import neuroglancer
import numpy as np
import zarr
from flask import jsonify

from cellmap_flow.dashboard.finetune_utils import (
    create_annotation_volume_zarr,
    create_correction_zarr,
    ensure_minio_serving,
    sync_all_annotations_from_minio,
    sync_annotation_from_minio,
)
from cellmap_flow.dashboard.routes.finetune.common import (
    ensure_corrections_storage,
    find_model_config,
    load_user_prefs,
    save_user_prefs,
    viewer_position_and_scales,
)
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
                logger.warning(f"Could not extract config for {model_config.name}: {e}")

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

        config = model_config.config
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

        config = model_config.config
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


def refresh_annotated_regions_layer(corrections_path=None):
    if not hasattr(g, "viewer") or g.viewer is None:
        return 0

    scan_dirs = []
    if corrections_path:
        scan_dirs.append(corrections_path)
    else:
        for volume in (getattr(g, "annotation_volumes", {}) or {}).values():
            corrections_dir = volume.get("corrections_dir")
            if corrections_dir and corrections_dir not in scan_dirs:
                scan_dirs.append(corrections_dir)
    if not scan_dirs:
        return 0

    boxes = []
    for corrections_dir in scan_dirs:
        if not os.path.isdir(corrections_dir):
            continue
        for entry in sorted(os.listdir(corrections_dir)):
            if "_chunk_" not in entry or not entry.endswith(".zarr"):
                continue
            zattrs_file = os.path.join(corrections_dir, entry, ".zattrs")
            if not os.path.exists(zattrs_file):
                continue
            try:
                with open(zattrs_file) as f:
                    meta = json.load(f)
                roi = meta.get("roi", {})
                offset_vox = roi.get("annotation_offset")
                shape_vox = roi.get("annotation_shape")
                voxel = meta.get("annotation_voxel_size")
                if not (offset_vox and shape_vox and voxel):
                    continue

                voxel_arr = np.array(voxel, dtype=np.float64)
                lo = np.array(offset_vox, dtype=np.float64) * voxel_arr
                hi = lo + np.array(shape_vox, dtype=np.float64) * voxel_arr
                boxes.append({"chunk": entry, "lo": lo.tolist(), "hi": hi.tolist()})
            except Exception as e:
                logger.warning(f"Could not read chunk metadata for {entry}: {e}")

    layer_name = "annotated_regions"
    if not boxes:
        try:
            with g.viewer.txn() as s:
                if layer_name in s.layers:
                    del s.layers[layer_name]
        except Exception:
            pass
        return 0

    axes_names = ["z", "y", "x"]
    try:
        if hasattr(g, "raw") and g.raw is not None:
            source = getattr(g.raw, "source", None)
            if source is not None and hasattr(source, "dimensions"):
                axes_names = list(source.dimensions.names)
    except Exception:
        pass

    annotations = [
        neuroglancer.AxisAlignedBoundingBoxAnnotation(
            point_a=box["lo"],
            point_b=box["hi"],
            id=str(index),
            description=box["chunk"],
        )
        for index, box in enumerate(boxes)
    ]

    try:
        with g.viewer.txn() as s:
            s.layers[layer_name] = neuroglancer.LocalAnnotationLayer(
                dimensions=neuroglancer.CoordinateSpace(
                    names=axes_names,
                    units="nm",
                    scales=[1, 1, 1],
                ),
                annotations=annotations,
            )
    except Exception as e:
        logger.warning(f"Could not update annotated_regions layer: {e}")
        return 0

    return len(boxes)


def list_existing_sessions_response(data):
    try:
        output_path = data.get("output_path", "")
        if not output_path:
            return jsonify({"success": False, "error": "output_path required"}), 400

        base = os.path.expanduser(output_path)
        if not os.path.isdir(base):
            return jsonify({"success": True, "sessions": []})

        sessions = []
        for entry in sorted(os.listdir(base), reverse=True):
            session_dir = os.path.join(base, entry)
            corrections_dir = os.path.join(session_dir, "corrections")
            if not os.path.isdir(corrections_dir):
                continue

            volumes = []
            chunks = []
            for item in os.listdir(corrections_dir):
                if not item.endswith(".zarr"):
                    continue
                full = os.path.join(corrections_dir, item)
                if "_chunk_" in item:
                    chunks.append(item)
                else:
                    volumes.append({"volume_id": item.replace(".zarr", ""), "path": full})

            if volumes or chunks:
                sessions.append(
                    {
                        "session_id": entry,
                        "session_path": session_dir,
                        "volumes": volumes,
                        "chunk_count": len(chunks),
                    }
                )

        return jsonify({"success": True, "sessions": sessions})
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


def load_existing_volume_response(data):
    try:
        import shutil

        from cellmap_flow.dashboard.finetune_utils import minio_state

        source_session_path = data.get("source_session_path")
        output_path = data.get("output_path")
        if not source_session_path or not output_path:
            return jsonify(
                {"success": False, "error": "source_session_path and output_path required"}
            ), 400

        source_session_path = os.path.expanduser(source_session_path)
        source_corrections = os.path.join(source_session_path, "corrections")
        if not os.path.isdir(source_corrections):
            return jsonify({"success": False, "error": f"No corrections found in {source_session_path}"}), 404

        volume_entries = [
            entry
            for entry in os.listdir(source_corrections)
            if entry.endswith(".zarr") and "_chunk_" not in entry
        ]
        if not volume_entries:
            return jsonify(
                {"success": False, "error": f"No annotation volume found in {source_corrections}"}
            ), 404

        volume_dir = volume_entries[0]
        volume_id = volume_dir.replace(".zarr", "")
        new_session_path, new_corrections = ensure_corrections_storage(output_path)

        copied = []
        for item in os.listdir(source_corrections):
            if not item.endswith(".zarr"):
                continue
            src = os.path.join(source_corrections, item)
            dst = os.path.join(new_corrections, item)
            if os.path.exists(dst):
                logger.info(f"Skipping {item} (already exists in target)")
                continue
            shutil.copytree(src, dst)
            copied.append(item)

        source_minio = os.path.join(source_corrections, ".minio")
        new_minio = os.path.join(new_corrections, ".minio")
        copied_minio = False
        if os.path.isdir(source_minio):
            if minio_state.get("process") is not None and minio_state["process"].poll() is None:
                logger.warning(
                    "MinIO already running with a different output_base; cannot rebind. "
                    "Falling back to mc mirror upload — painted data may be incomplete "
                    "if the source had unsynced chunks."
                )
            elif not os.path.exists(new_minio):
                shutil.copytree(source_minio, new_minio)
                copied_minio = True

        lineage_file = os.path.join(new_session_path, "loaded_from.json")
        with open(lineage_file, "w") as f:
            json.dump(
                {
                    "source_session_path": source_session_path,
                    "loaded_at": datetime.now().isoformat(),
                    "copied_files": copied,
                },
                f,
                indent=2,
            )

        new_volume_path = os.path.join(new_corrections, volume_dir)
        zattrs_file = os.path.join(new_volume_path, ".zattrs")
        volume_meta = {}
        if os.path.exists(zattrs_file):
            with open(zattrs_file) as f:
                volume_meta = json.load(f)

        s0_dir = os.path.join(new_volume_path, "annotation", "s0")
        s0_count = 0
        if os.path.isdir(s0_dir):
            s0_count = sum(1 for entry in os.listdir(s0_dir) if not entry.startswith("."))

        minio_url = ensure_minio_serving(new_volume_path, volume_id, output_base_dir=new_corrections)
        _register_annotation_volume(
            volume_id,
            zarr_path=new_volume_path,
            model_name=volume_meta.get("model_name"),
            output_size=volume_meta.get("chunk_size"),
            input_size=volume_meta.get("input_size"),
            input_voxel_size=volume_meta.get("input_voxel_size"),
            output_voxel_size=volume_meta.get("output_voxel_size"),
            dataset_path=volume_meta.get("dataset_path"),
            dataset_offset_nm=volume_meta.get("dataset_offset_nm"),
            corrections_dir=new_corrections,
        )
        refresh_annotated_regions_layer()

        return jsonify(
            {
                "success": True,
                "volume_id": volume_id,
                "new_session_path": new_session_path,
                "zarr_path": new_volume_path,
                "minio_url": minio_url,
                "neuroglancer_url": f"{minio_url}/annotation",
                "copied_count": len(copied),
                "copied_minio": copied_minio,
                "painted_chunk_count": s0_count,
                "metadata": volume_meta,
            }
        )
    except Exception as e:
        logger.error(f"Error loading existing volume: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


def add_crop_to_viewer_response(data):
    try:
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
            s.layers[layer_name] = neuroglancer.SegmentationLayer(source=source_config)

        return jsonify({"success": True, "message": "Layer added to viewer", "layer_name": layer_name})
    except Exception as e:
        logger.error(f"Error adding layer to viewer: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


def sync_annotations_manually_response(data):
    try:
        crop_id = data.get("crop_id", None)
        force = data.get("force", True)

        if crop_id:
            success = sync_annotation_from_minio(crop_id, force=force)
            refresh_annotated_regions_layer()
            if success:
                return jsonify({"success": True, "message": f"Synced annotation for {crop_id}"})
            return jsonify({"success": False, "message": f"No updates to sync for {crop_id}"})

        synced = sync_all_annotations_from_minio(force=force)
        refresh_annotated_regions_layer()
        if synced == -1:
            return jsonify({"success": False, "error": "MinIO not initialized"}), 400
        return jsonify(
            {
                "success": True,
                "message": f"Synced {synced} annotations",
                "synced_count": synced,
            }
        )
    except Exception as e:
        logger.error(f"Error in sync endpoint: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
