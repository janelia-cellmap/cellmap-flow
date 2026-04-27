import json
import logging
import os

import neuroglancer
import numpy as np
from flask import jsonify

from cellmap_flow.dashboard.finetune_utils import (
    sync_all_annotations_from_minio,
    sync_annotation_from_minio,
)
from cellmap_flow.globals import g

logger = logging.getLogger(__name__)


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
