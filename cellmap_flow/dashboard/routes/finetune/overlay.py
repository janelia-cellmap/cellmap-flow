import json
import logging
import os
import re

import neuroglancer
import numpy as np
from flask import jsonify

from cellmap_flow.dashboard.finetune_utils import (
    sync_all_annotations_from_minio,
    sync_annotation_from_minio,
)
from cellmap_flow.globals import g

logger = logging.getLogger(__name__)

_CHUNK_KEY_RE = re.compile(r"^\d+\.\d+\.\d+$")

# What refresh_annotated_regions_layer() last wrote into the viewer.
#
# Every push from python costs the browser its whole UI state, not just the
# part we changed. On receiving a state from python, the browser runs
# `trackable.reset(); trackable.restoreState(state)`
# (ClientStateSynchronizer.setServerState, src/python_integration/api.ts) --
# a teardown and rebuild of the entire state object graph. Layer data sources
# are cached, so the imagery does not flicker and the annotations stay put;
# what does not survive is everything reconstructed from JSON, including the
# layer's tool binder. That is why the brush toolbar vanishes and the "are you
# sure you want to annotate" confirmation re-arms mid-session.
#
# neuroglancer's txn() is also unconditional -- it deep-copies the state on
# entry and calls set_state() on exit whether or not the body changed
# anything -- so a refresh that decides nothing needs saying still pays that
# cost.
#
# So nothing refreshes these boxes on a timer. The 30s annotation sync used
# to, which meant a teardown every time a chunk finished syncing, i.e.
# continuously while you were drawing. They are refreshed on demand instead,
# from a button. The cache below still matters: clicking the button twice
# with nothing changed in between should not cost a rebuild either.
_last_annotated_regions = None

# Keys pre-bound on every annotation layer we add, so the tools are reachable
# without hunting for them in the tool palette first.
#
# Tool ids come from the voxel_annotation module added by the open upstream PR
# google/neuroglancer#858 (src/voxel_annotation/base.ts), which is the branch
# this deployment runs. The keys must be a single capital letter -- anything
# else is rejected, see TOOL_KEY_PATTERN in src/ui/tool.ts.
ANNOTATION_TOOL_BINDINGS = {
    "A": "vox-brush",
    "F": "vox-flood-fill",
}


def _register_voxel_annotation_tools():
    """Teach the python bindings about the fork's voxel-painting tools.

    neuroglancer validates tool names against a registry built by
    @export_tool, and only the tools merged into mainline are in it --
    assigning an unregistered name raises KeyError. The voxel_annotation
    tools live in PR #858's frontend, which has no python side yet.

    Registering them here is additive: on a neuroglancer whose frontend does
    not have these tools the binding is simply inert, which is the same
    outcome as not setting it.
    """
    try:
        from neuroglancer.viewer_state import Tool, tool_types
    except Exception as e:  # pragma: no cover - neuroglancer always present
        logger.debug(f"Could not register voxel annotation tools: {e}")
        return

    for tool_id in ANNOTATION_TOOL_BINDINGS.values():
        if tool_id in tool_types:
            continue
        tool_types[tool_id] = type(
            f"_{tool_id.replace('-', '_')}_Tool",
            (Tool,),
            {"__slots__": (), "TOOL_TYPE": tool_id},
        )


_register_voxel_annotation_tools()


def _chunk_outside_all_bboxes(
    chunk_lo_voxels: np.ndarray,
    chunk_hi_voxels: np.ndarray,
    bbox_offsets: np.ndarray,
    bbox_ends: np.ndarray,
) -> bool:
    """Return True if the chunk does not overlap any ``imported_crops`` bbox
    -- i.e. it represents painted-scribble work that the per-import yellow
    boxes don't already cover.

    A YAML crop's own voxel offset (relative to the annotation volume) is
    essentially never a multiple of the volume's chunk size -- externally
    authored crops (e.g. Amira exports) land at arbitrary nm offsets, not
    chunk-aligned ones. So every chunk straddling an import bbox's boundary
    only partially overlaps it; requiring *full* containment misclassifies
    that entire boundary layer of chunks as "painted-only" and draws a small
    box for each one, producing a fence of duplicate-looking boxes right
    along the real import box's edges. Any overlap is enough to call a
    chunk covered.
    """
    if bbox_offsets.shape[0] == 0:
        return True
    overlaps = np.all(
        (chunk_lo_voxels < bbox_ends) & (chunk_hi_voxels > bbox_offsets),
        axis=1,
    )
    return not bool(overlaps.any())


def refresh_annotated_regions_layer(corrections_path=None):
    """Draw a box around every region that has annotations in it.

    Only ever called for something the user just did -- creating a volume,
    importing crops, or clicking "Show Annotated Regions". Nothing calls this
    on a timer: see the note on _last_annotated_regions for why a push the
    user did not ask for is destructive.
    """
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

    global _last_annotated_regions

    layer_name = "annotated_regions"
    if not boxes:
        try:
            # Only open a transaction if there is actually something to remove.
            if layer_name in g.viewer.state.layers:
                with g.viewer.txn() as s:
                    if layer_name in s.layers:
                        del s.layers[layer_name]
        except Exception:
            pass
        _last_annotated_regions = None
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

    # Nothing to say that we have not already said: leave the viewer alone.
    # Checked against the live layer list too, so a layer that went away (a
    # reset, a manual delete) is still restored.
    signature = (tuple(axes_names), tuple(
        (tuple(box["lo"]), tuple(box["hi"]), box["label"]) for box in boxes
    ))
    try:
        if signature == _last_annotated_regions and layer_name in g.viewer.state.layers:
            return len(boxes)
    except Exception:
        pass

    try:
        with g.viewer.txn() as s:
            # Whether the layer is shown is the user's call, not ours.
            #
            # This used to force visible=True on every refresh, and the
            # periodic sync thread calls this every 30s -- so turning the boxes
            # off in neuroglancer un-did itself moments later, over and over.
            # Keep whatever visibility the layer already has, and start hidden
            # when creating it: the boxes are an occasional orientation aid,
            # not something to draw over the data by default.
            was_visible = None
            if layer_name in s.layers:
                try:
                    was_visible = bool(s.layers[layer_name].visible)
                except Exception:
                    was_visible = None

            s.layers[layer_name] = neuroglancer.LocalAnnotationLayer(
                dimensions=neuroglancer.CoordinateSpace(
                    names=axes_names,
                    units="nm",
                    scales=[1, 1, 1],
                ),
                annotations=annotations,
            )
            try:
                s.layers[layer_name].visible = (
                    False if was_visible is None else was_visible
                )
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Could not update annotated_regions layer: {e}")
        return 0

    _last_annotated_regions = signature
    return len(boxes)


def refresh_annotated_regions_response(data):
    """Redraw the annotated-regions boxes because the user asked for it.

    This is the only way the boxes update during a session. It pushes viewer
    state, which rebuilds every layer browser-side, so it is deliberately a
    button: you click it when you want to see where you have painted, not
    while you are in the middle of painting.
    """
    try:
        if not hasattr(g, "viewer") or g.viewer is None:
            return jsonify({"success": False, "error": "Viewer not initialized"}), 400
        count = refresh_annotated_regions_layer(
            corrections_path=(data or {}).get("corrections_path")
        )
        return jsonify({"success": True, "count": count})
    except Exception as e:
        logger.error(f"Error refreshing annotated regions: {e}", exc_info=True)
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
            layer = neuroglancer.SegmentationLayer(source=source_config)
            try:
                layer.tool_bindings = dict(ANNOTATION_TOOL_BINDINGS)
            except Exception as e:
                # An older neuroglancer without tool_bindings should still get
                # its layer; the keys just will not be pre-bound.
                logger.warning(f"Could not pre-bind annotation tools: {e}")
            s.layers[layer_name] = layer

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
