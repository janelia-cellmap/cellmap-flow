"""
Flask route handlers for Neuroglancer painting/annotation workflow.

These routes are stored as standalone functions. To reintegrate into the
main app, register them with @app.route decorators or use a Flask Blueprint.

Original routes:
    POST /api/finetune/create-crop
    POST /api/finetune/create-volume
    POST /api/finetune/add-to-viewer
    POST /api/finetune/sync-annotations
"""

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

import neuroglancer
import numpy as np
import zarr
from flask import request, jsonify

from cellmap_flow.globals import g

from .minio_manager import ensure_minio_serving, annotation_volumes
from .zarr_utils import create_correction_zarr, create_annotation_volume_zarr
from .annotation_sync import sync_annotation_from_minio, sync_annotation_volume_from_minio, sync_all_annotations_from_minio
from .minio_manager import minio_state

logger = logging.getLogger(__name__)


def create_annotation_crop():
    """Create an annotation crop centered at view center position.

    Original route: POST /api/finetune/create-crop
    """
    try:
        from cellmap_flow.image_data_interface import ImageDataInterface
        from funlib.geometry import Roi, Coordinate

        data = request.get_json()
        model_name = data.get("model_name")
        output_path = data.get("output_path")  # User-specified output directory

        if not hasattr(g, "models_config") or not g.models_config:
            return jsonify({"success": False, "error": "No models loaded"}), 400

        if not hasattr(g, "viewer") or g.viewer is None:
            return jsonify({"success": False, "error": "Viewer not initialized"}), 400

        # Get view center and scales automatically from viewer
        with g.viewer.txn() as s:
            # Get the current view position (center of view)
            position = s.position

            # Get the viewer dimensions to extract scales
            dimensions = s.dimensions
            viewer_scales_nm = None

            if dimensions and hasattr(dimensions, "scales"):
                # CoordinateSpace has scales attribute directly
                scales_nm = list(dimensions.scales)

                # Check units and convert if needed
                if hasattr(dimensions, "units"):
                    units = dimensions.units
                    # units can be a string (same for all axes) or list
                    if isinstance(units, str):
                        units = [units] * len(scales_nm)

                    # Convert to nm if needed
                    converted_scales = []
                    for scale, unit in zip(scales_nm, units):
                        if unit == "m":
                            converted_scales.append(scale * 1e9)  # meters to nanometers
                        elif unit == "nm":
                            converted_scales.append(scale)
                        else:
                            logger.warning(f"Unknown unit: {unit}, assuming nm")
                            converted_scales.append(scale)
                    viewer_scales_nm = converted_scales
                else:
                    viewer_scales_nm = scales_nm

            # Convert to list if it's a numpy array or coordinate object
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

        # Get model parameters (lightweight — avoids loading weights/GPU into dashboard)
        info = model_config.lightweight_info()
        read_shape = np.array(info["read_shape"])
        write_shape = np.array(info["write_shape"])
        input_voxel_size = np.array(info["input_voxel_size"])
        output_voxel_size = np.array(info["output_voxel_size"])
        output_channels = info["output_channels"]

        # Convert view center to nm using viewer scales
        if viewer_scales_nm is not None:
            viewer_scales_nm = np.array(viewer_scales_nm)
            view_center_nm = view_center * viewer_scales_nm
            logger.info(
                f"Converted view center from {view_center} (viewer coords) to {view_center_nm} nm"
            )
            logger.info(f"  Using viewer scales: {viewer_scales_nm} nm")
        else:
            # Fallback: assume it's already in nm
            view_center_nm = view_center
            logger.warning(
                "No viewer scales provided, assuming view center is already in nm"
            )

        # Calculate raw crop size in voxels (use read_shape and input_voxel_size)
        raw_crop_shape_voxels = (read_shape / input_voxel_size).astype(int)

        # Calculate annotation crop size in voxels (use write_shape and output_voxel_size)
        annotation_crop_shape_voxels = (write_shape / output_voxel_size).astype(int)

        # Calculate crop offset for raw (center the crop at view center)
        half_read_shape = read_shape / 2
        raw_crop_offset_nm = view_center_nm - half_read_shape
        raw_crop_offset_voxels = (raw_crop_offset_nm / input_voxel_size).astype(int)

        # Calculate crop offset for annotation (center the crop at view center)
        half_write_shape = write_shape / 2
        annotation_crop_offset_nm = view_center_nm - half_write_shape
        annotation_crop_offset_voxels = (
            annotation_crop_offset_nm / output_voxel_size
        ).astype(int)

        # Generate unique crop ID
        crop_id = f"{uuid.uuid4().hex[:8]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Create zarr structure with timestamped session directory
        # Import get_or_create_session_path from main app
        from cellmap_flow.dashboard.app import get_or_create_session_path

        if output_path:
            # Use user-specified output path with timestamped session
            session_path = get_or_create_session_path(output_path)
            corrections_dir = os.path.join(session_path, "corrections")
            os.makedirs(corrections_dir, exist_ok=True)

            # Initialize as zarr group if not already
            zarr.open_group(corrections_dir, mode='a')

            zarr_path = os.path.join(corrections_dir, f"{crop_id}.zarr")
            logger.info(f"Using session path: {session_path}")
            logger.info(f"Corrections directory: {corrections_dir}")
        else:
            # Fallback to default location
            corrections_dir = os.path.expanduser("~/.cellmap_flow/corrections")
            os.makedirs(corrections_dir, exist_ok=True)

            # Initialize as zarr group if not already
            zarr.open_group(corrections_dir, mode='a')

            zarr_path = os.path.join(corrections_dir, f"{crop_id}.zarr")

        # Get dataset path
        dataset_path = getattr(g, "dataset_path", "unknown")

        # Create ImageDataInterface first to get the data dtype
        logger.info(f"Creating ImageDataInterface for {dataset_path}")
        logger.info(f"Using input voxel size: {input_voxel_size} nm")
        try:
            idi = ImageDataInterface(dataset_path, voxel_size=input_voxel_size)
            # Get the dtype from the tensorstore
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

        # Create zarr with OME-NGFF metadata (no mask needed)
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

        # Read and fill raw data from the dataset
        logger.info(f"Reading raw data from {dataset_path}")
        try:

            # Define ROI for the crop in physical coordinates (nm)
            # Center the crop at view_center_nm
            roi = Roi(
                offset=Coordinate(view_center_nm - read_shape / 2),
                shape=Coordinate(read_shape),
            )
            logger.info(f"Reading ROI: offset={roi.offset}, shape={roi.shape}")

            # Read the data using tensorstore interface
            raw_data = idi.to_ndarray_ts(roi)
            logger.info(
                f"Read raw data with shape: {raw_data.shape}, dtype: {raw_data.dtype}"
            )

            # Write to zarr
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

        # Start/ensure MinIO is running and upload
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

    except Exception as e:
        logger.error(f"Error creating annotation crop: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


def create_annotation_volume():
    """Create a sparse annotation volume covering the full dataset extent.

    Original route: POST /api/finetune/create-volume
    """
    try:
        from cellmap_flow.image_data_interface import ImageDataInterface
        from funlib.geometry import Coordinate

        data = request.get_json()
        model_name = data.get("model_name")
        output_path = data.get("output_path")

        if not hasattr(g, "models_config") or not g.models_config:
            return jsonify({"success": False, "error": "No models loaded"}), 400

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

        # Get model parameters (lightweight — avoids loading weights/GPU into dashboard)
        info = model_config.lightweight_info()
        read_shape = np.array(info["read_shape"])
        write_shape = np.array(info["write_shape"])
        input_voxel_size = np.array(info["input_voxel_size"])
        output_voxel_size = np.array(info["output_voxel_size"])

        # Compute output_size and input_size in voxels
        output_size = (write_shape / output_voxel_size).astype(int)
        input_size = (read_shape / input_voxel_size).astype(int)

        # Get dataset path
        dataset_path = getattr(g, "dataset_path", None)
        if not dataset_path:
            return (
                jsonify({"success": False, "error": "No dataset path configured"}),
                400,
            )

        # Get full dataset extent
        logger.info(f"Getting dataset extent from {dataset_path}")
        try:
            idi = ImageDataInterface(dataset_path, voxel_size=output_voxel_size)
            dataset_roi = idi.roi
            dataset_offset_nm = np.array(dataset_roi.offset)
            dataset_shape_nm = np.array(dataset_roi.shape)

            # Convert to voxels at output resolution
            dataset_shape_voxels = (dataset_shape_nm / output_voxel_size).astype(int)

            # Snap up to chunk_size (output_size) multiples
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

        # Generate volume ID
        volume_id = (
            f"vol-{uuid.uuid4().hex[:8]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )

        # Set up output directory
        from cellmap_flow.dashboard.app import get_or_create_session_path

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

        # Create the annotation volume zarr
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

        # Upload to MinIO
        minio_url = ensure_minio_serving(
            zarr_path, volume_id, output_base_dir=corrections_dir
        )

        # Store volume metadata for sync to use
        annotation_volumes[volume_id] = {
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


def add_crop_to_viewer():
    """Add annotation crop or volume layer to Neuroglancer viewer.

    Original route: POST /api/finetune/add-to-viewer
    """
    try:
        data = request.get_json()
        crop_id = data.get("crop_id")
        minio_url = data.get("minio_url")

        if not hasattr(g, "viewer") or g.viewer is None:
            return jsonify({"success": False, "error": "Viewer not initialized"}), 400

        # Add layer to viewer
        with g.viewer.txn() as s:
            layer_name = data.get("layer_name", f"annotation_{crop_id}")
            # Configure source with writing enabled
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


def sync_annotations_manually():
    """Manually trigger sync of annotations from MinIO to local disk.

    Original route: POST /api/finetune/sync-annotations
    """
    try:
        import s3fs as _s3fs

        data = request.get_json()
        crop_id = data.get("crop_id", None)  # If None, sync all
        force = data.get("force", True)  # Force sync by default for manual trigger

        if crop_id:
            # Sync single crop
            success = sync_annotation_from_minio(crop_id, force=force)
            if success:
                return jsonify({
                    "success": True,
                    "message": f"Synced annotation for {crop_id}"
                })
            else:
                return jsonify({
                    "success": False,
                    "message": f"No updates to sync for {crop_id}"
                })
        else:
            # Sync all crops
            if not minio_state["ip"] or not minio_state["port"]:
                return jsonify({"success": False, "error": "MinIO not initialized"}), 400

            try:
                s3 = _s3fs.S3FileSystem(
                    anon=False,
                    key='minio',
                    secret='minio123',
                    client_kwargs={
                        'endpoint_url': f"http://{minio_state['ip']}:{minio_state['port']}",
                        'region_name': 'us-east-1'
                    }
                )

                zarrs = s3.ls(minio_state['bucket'])
                zarr_ids = [Path(c).name.replace('.zarr', '') for c in zarrs if c.endswith('.zarr')]

                synced_count = 0
                for zid in zarr_ids:
                    # Route volumes vs crops
                    try:
                        zarr_name = f"{zid}.zarr"
                        attrs_path = f"{minio_state['bucket']}/{zarr_name}/.zattrs"
                        if s3.exists(attrs_path):
                            root_attrs = json.loads(s3.cat(attrs_path))
                            if root_attrs.get("type") == "annotation_volume":
                                if sync_annotation_volume_from_minio(zid, force=force):
                                    synced_count += 1
                                continue
                    except Exception:
                        pass
                    if sync_annotation_from_minio(zid, force=force):
                        synced_count += 1

                return jsonify({
                    "success": True,
                    "message": f"Synced {synced_count} annotations",
                    "synced_count": synced_count,
                    "total_crops": len(zarr_ids)
                })

            except Exception as e:
                logger.error(f"Error syncing all annotations: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

    except Exception as e:
        logger.error(f"Error in sync endpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500
