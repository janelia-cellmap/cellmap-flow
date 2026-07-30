import json
import logging
import os
import re
from pathlib import PurePosixPath

import neuroglancer
import numpy as np
import requests
import s3fs
import zarr
from flask import Response, jsonify, request

from cellmap_flow.dashboard.finetune_utils import (
    _get_volume_metadata,
    _make_s3_filesystem,
    sync_all_annotations_from_minio,
    sync_annotation_from_minio,
)
from cellmap_flow.globals import g

logger = logging.getLogger(__name__)

_CHUNK_KEY_RE = re.compile(r"^\d+\.\d+\.\d+$")


def _chunk_outside_all_bboxes(
    chunk_lo_voxels: np.ndarray,
    chunk_hi_voxels: np.ndarray,
    bbox_offsets: np.ndarray,
    bbox_ends: np.ndarray,
) -> bool:
    """Return True if the chunk is NOT fully contained in any
    ``imported_crops`` bbox -- i.e. it represents painted-scribble
    work that the per-import yellow boxes don't already cover.

    YAML imports write chunk-aligned slabs, so imported chunks land
    fully inside an import bbox; painted-only chunks land fully outside.
    A mixed chunk (rare; user paints over an import edge) reads as
    "outside" by this rule, which is what we want -- it has painted
    work the existing yellow box may not visually cue.
    """
    if bbox_offsets.shape[0] == 0:
        return True
    fully_inside = np.all(
        (chunk_lo_voxels >= bbox_offsets) & (chunk_hi_voxels <= bbox_ends),
        axis=1,
    )
    return not bool(fully_inside.any())


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
        # Also scan corrections dirs from active output sessions so
        # YAML-loaded crops show up even when no annotation_volume
        # has been registered for the session.
        for session_path in (getattr(g, "output_sessions", {}) or {}).values():
            session_corrections = os.path.join(session_path, "corrections")
            if session_corrections not in scan_dirs and os.path.isdir(session_corrections):
                scan_dirs.append(session_corrections)
    if not scan_dirs:
        return 0

    boxes = []
    for corrections_dir in scan_dirs:
        if not os.path.isdir(corrections_dir):
            continue
        for entry in sorted(os.listdir(corrections_dir)):
            # Per-painted-chunk small boxes (the existing behavior).
            if "_chunk_" in entry and entry.endswith(".zarr"):
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
                    boxes.append({"label": entry, "lo": lo.tolist(), "hi": hi.tolist()})
                except Exception as e:
                    logger.warning(f"Could not read chunk metadata for {entry}: {e}")
                continue

            # Per-imported-YAML-crop large boxes (one per crop, read from the
            # annotation_volume.zarr's root attrs that the YAML loader writes)
            # plus per-painted-chunk small boxes for any populated chunk that
            # isn't already covered by an import bbox.
            if entry.endswith(".zarr"):
                vol_attrs_file = os.path.join(corrections_dir, entry, ".zattrs")
                if not os.path.exists(vol_attrs_file):
                    continue
                try:
                    with open(vol_attrs_file) as f:
                        vol_meta = json.load(f)
                    if vol_meta.get("type") != "annotation_volume":
                        continue
                    voxel = vol_meta.get("output_voxel_size")
                    dataset_offset = vol_meta.get("dataset_offset_nm", [0, 0, 0])
                    if not voxel:
                        continue
                    voxel_arr = np.array(voxel, dtype=np.float64)
                    dataset_offset_arr = np.array(dataset_offset, dtype=np.float64)

                    # Pass 1: yellow boxes for each imported crop.
                    imported = vol_meta.get("imported_crops") or []
                    bbox_off_list = []
                    bbox_end_list = []
                    for crop in imported:
                        offset_vox = crop.get("annotation_offset_voxels")
                        shape_vox = crop.get("annotation_shape_voxels")
                        if not (offset_vox and shape_vox):
                            continue
                        offset_arr = np.array(offset_vox, dtype=np.int64)
                        shape_arr = np.array(shape_vox, dtype=np.int64)
                        bbox_off_list.append(offset_arr)
                        bbox_end_list.append(offset_arr + shape_arr)
                        lo = (
                            dataset_offset_arr
                            + offset_arr.astype(np.float64) * voxel_arr
                        )
                        hi = lo + shape_arr.astype(np.float64) * voxel_arr
                        label = crop.get("name") or os.path.basename(
                            crop.get("path", "imported_crop").rstrip("/")
                        )
                        boxes.append(
                            {"label": f"yaml_crop:{label}", "lo": lo.tolist(), "hi": hi.tolist()}
                        )
                    bbox_offsets = (
                        np.stack(bbox_off_list, axis=0)
                        if bbox_off_list
                        else np.zeros((0, 3), dtype=np.int64)
                    )
                    bbox_ends = (
                        np.stack(bbox_end_list, axis=0)
                        if bbox_end_list
                        else np.zeros((0, 3), dtype=np.int64)
                    )

                    # Pass 2: small boxes for painted-only chunks. Walk the
                    # volume zarr's annotation/s0/ chunk files and emit a box
                    # per chunk that isn't fully contained in any import bbox.
                    # Cheap: just lists chunk file names and compares spatial
                    # bbox to import bboxes -- never reads chunk contents.
                    chunk_size = vol_meta.get("chunk_size")
                    if not chunk_size:
                        continue
                    chunk_size_arr = np.array(chunk_size, dtype=np.int64)
                    s0_path = os.path.join(corrections_dir, entry, "annotation", "s0")
                    if not os.path.isdir(s0_path):
                        continue
                    crop_label = (
                        os.path.basename(crop.get("path", "")).rstrip("/")
                        if imported
                        else "painted"
                    )
                    for chunk_name in os.listdir(s0_path):
                        if not _CHUNK_KEY_RE.match(chunk_name):
                            continue
                        cz, cy, cx = (int(s) for s in chunk_name.split("."))
                        chunk_lo_vox = (
                            np.array([cz, cy, cx], dtype=np.int64) * chunk_size_arr
                        )
                        chunk_hi_vox = chunk_lo_vox + chunk_size_arr
                        if not _chunk_outside_all_bboxes(
                            chunk_lo_vox, chunk_hi_vox, bbox_offsets, bbox_ends
                        ):
                            continue
                        lo = (
                            dataset_offset_arr
                            + chunk_lo_vox.astype(np.float64) * voxel_arr
                        )
                        hi = (
                            dataset_offset_arr
                            + chunk_hi_vox.astype(np.float64) * voxel_arr
                        )
                        boxes.append(
                            {
                                "label": f"painted:{chunk_name}",
                                "lo": lo.tolist(),
                                "hi": hi.tolist(),
                            }
                        )
                except Exception as e:
                    logger.warning(
                        f"Could not read annotation_volume metadata for {entry}: {e}"
                    )

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
            description=box["label"],
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
            # Force-visible in case a prior toggle archived the layer.
            try:
                s.layers[layer_name].visible = True
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Could not update annotated_regions layer: {e}")
        return 0

    return len(boxes)


def _annotation_proxy_url(
    volume_id: str,
    revision: int,
    object_path: str = "annotation",
    host_url: str | None = None,
) -> str:
    """Build a versioned annotation proxy URL for Neuroglancer source identity.

    Ported from sam-backend-support's finetune_routes.py.
    """
    base = host_url.rstrip("/") if host_url else ""
    suffix = object_path.lstrip("/") if object_path else ""
    path = f"/api/finetune/minio-proxy/{volume_id}/v/{int(revision)}"
    if suffix:
        path = f"{path}/{suffix}"
    return f"{base}{path}"


def add_crop_to_viewer_response(data):
    try:
        crop_id = data.get("crop_id")
        minio_url = data.get("minio_url")
        if not hasattr(g, "viewer") or g.viewer is None:
            return jsonify({"success": False, "error": "Viewer not initialized"}), 400

        # Only route ai-annotate-enabled volumes through the revision-bumped
        # proxy (needed so _invalidate_annotation_layer has something to
        # refresh after a server-side write). Every other volume keeps the
        # existing direct-MinIO-URL behavior unchanged, to avoid regressing
        # the working manual-paint-only path.
        volume_meta = _get_volume_metadata(crop_id) if crop_id else None
        use_proxy = bool(volume_meta and volume_meta.get("ai_annotate_enabled"))

        if use_proxy:
            revisions = g.minio_state.setdefault("annotation_revisions", {})
            revision = int(revisions.get(crop_id, 0))
            proxy_host = request.host_url.rstrip("/")
            g.minio_state.setdefault("annotation_proxy_host", {})[crop_id] = proxy_host
            source_url = _annotation_proxy_url(crop_id, revision, "annotation", host_url=proxy_host)
        else:
            source_url = minio_url

        layer_name = data.get("layer_name", f"annotation_{crop_id}")
        source_config = {
            "url": f"s3+{source_url}",
            "subsources": {"default": {"writingEnabled": True}, "bounds": {}},
        }

        with g.viewer.txn() as s:
            s.layers[layer_name] = neuroglancer.SegmentationLayer(source=source_config)

        if use_proxy:
            tracked_layers = set(g.minio_state.setdefault("annotation_layers", {}).get(crop_id, []))
            tracked_layers.add(layer_name)
            g.minio_state["annotation_layers"][crop_id] = sorted(tracked_layers)

        if volume_meta and volume_meta.get("ai_annotate_enabled"):
            try:
                from cellmap_flow.dashboard.routes.finetune.ai_annotate import (
                    ensure_ai_annotate_point_layer,
                )

                ensure_ai_annotate_point_layer(g.viewer)
            except Exception as e:
                logger.warning(f"Could not ensure AI-annotate point layer for volume {crop_id}: {e}")

        return jsonify({"success": True, "message": "Layer added to viewer", "layer_name": layer_name})
    except Exception as e:
        logger.error(f"Error adding layer to viewer: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


def _invalidate_annotation_layer(volume_id: str):
    """Bump source revision and refresh tracked annotation layers in viewer.

    Ported from sam-backend-support's sam_annotator.py. Needed because a
    server-side zarr write (unlike browser-side painting) doesn't otherwise
    cause an already-open Neuroglancer viewer to refetch the changed chunks.
    """
    minio_state = g.minio_state
    revisions = minio_state.setdefault("annotation_revisions", {})
    new_revision = int(revisions.get(volume_id, 0)) + 1
    revisions[volume_id] = new_revision

    layer_names = list(minio_state.setdefault("annotation_layers", {}).get(volume_id, []))
    if not layer_names or not hasattr(g, "viewer") or g.viewer is None:
        logger.info(
            f"Annotation written for {volume_id}; revision={new_revision} "
            "(no tracked viewer layer to refresh)"
        )
        return

    proxy_host = minio_state.setdefault("annotation_proxy_host", {}).get(volume_id, "")
    if not proxy_host:
        logger.warning(
            f"Annotation written for {volume_id}; revision={new_revision} "
            "but no proxy host was recorded for this layer. Re-add the layer once."
        )
        return

    proxy_url = _annotation_proxy_url(volume_id, new_revision, "annotation", host_url=proxy_host)
    source_config = {
        "url": f"s3+{proxy_url}",
        "subsources": {"default": {"writingEnabled": True}, "bounds": {}},
    }

    refreshed = 0
    with g.viewer.txn() as s:
        alive_layers = []
        for layer_name in layer_names:
            if layer_name not in s.layers:
                continue
            layer = s.layers[layer_name]
            try:
                layer.source = source_config
                s.layers[layer_name] = layer
            except Exception:
                s.layers[layer_name] = neuroglancer.SegmentationLayer(source=source_config)
            alive_layers.append(layer_name)
            refreshed += 1
    minio_state["annotation_layers"][volume_id] = alive_layers

    logger.info(
        f"Annotation written for {volume_id}; bumped to revision={new_revision}, "
        f"refreshed_layers={refreshed}"
    )


def proxy_minio_annotation_response(volume_id, revision, object_path):
    """Proxy annotation traffic to MinIO while using revision in URL as cache key.

    Ported from sam-backend-support's finetune_routes.py.
    """
    del revision  # Used only for cache busting in the client-visible URL.

    minio_state = g.minio_state
    if not minio_state.get("ip") or not minio_state.get("port"):
        return jsonify({"success": False, "error": "MinIO not initialized"}), 503

    normalized = object_path.lstrip("/")
    if normalized:
        parts = PurePosixPath(normalized).parts
        if ".." in parts:
            return jsonify({"success": False, "error": "Invalid path"}), 400

    minio_target = f"http://{minio_state['ip']}:{minio_state['port']}/{minio_state['bucket']}/{volume_id}.zarr"
    if normalized:
        minio_target = f"{minio_target}/{normalized}"

    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}

    try:
        upstream = requests.request(
            method=request.method,
            url=minio_target,
            params=request.args,
            headers=fwd_headers,
            data=request.get_data(),
            allow_redirects=False,
            timeout=120,
        )
    except Exception as e:
        logger.error(f"MinIO proxy error for {minio_target}: {e}")
        return jsonify({"success": False, "error": f"Proxy failed: {e}"}), 502

    excluded = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
    resp_headers = [(k, v) for k, v in upstream.headers.items() if k.lower() not in excluded]
    return Response(upstream.content, status=upstream.status_code, headers=resp_headers)


def write_ai_mask_to_minio(
    volume_id: str,
    chunk_indices: tuple[int, int, int],
    z_row_index: int,
    mask_2d: np.ndarray,
    label_id: int = 2,
    background_label_id: int = 1,
):
    """Paint one reviewed z-row of one chunk into the annotation volume in MinIO.

    Adapted from sam-backend-support's _write_mask_to_minio, but narrower:
    writes only within the specific reviewed z-row, only where mask_2d>0
    (foreground) vs. elsewhere in that row (background) -- every other
    z-row/voxel in the chunk is left untouched, so any pre-existing sparse
    annotation elsewhere in the chunk survives.
    """
    volume_meta = _get_volume_metadata(volume_id)
    if volume_meta is None:
        raise ValueError(f"Unknown volume_id: {volume_id}")

    chunk_size = np.array(volume_meta["output_size"])
    cz, cy, cx = (int(v) for v in chunk_indices)
    z0 = cz * int(chunk_size[0])
    y0, y1 = cy * int(chunk_size[1]), (cy + 1) * int(chunk_size[1])
    x0, x1 = cx * int(chunk_size[2]), (cx + 1) * int(chunk_size[2])
    z = z0 + int(z_row_index)

    bucket = g.minio_state["bucket"]
    zarr_name = f"{volume_id}.zarr"

    s3 = _make_s3_filesystem()
    store = s3fs.S3Map(root=f"{bucket}/{zarr_name}/annotation", s3=s3)
    arr = zarr.open(store, mode="r+")["s0"]

    row = arr[z, y0:y1, x0:x1]
    row[:] = np.where(mask_2d > 0, label_id, background_label_id).astype(row.dtype)
    arr[z, y0:y1, x0:x1] = row

    logger.info(
        f"Painted AI-annotate mask for {zarr_name} at chunk {chunk_indices}, z-row {z_row_index} "
        f"({int(np.sum(mask_2d > 0))} foreground voxels)"
    )


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
