"""Click a point in the viewer -> send a small EM crop to Gemini for a
first-pass segmentation mask -> human reviews -> accepted mask is written
into the sparse annotation-volume zarr as new training data.

Point capture (LocalAnnotationLayer + PlacePointTool + keybinding, action
handler shape) is ported from sam-backend-support's sam_annotator.py, which
built the same point-prompt -> model -> mask flow for SAM but wrote directly
with no review step. Here the background pipeline only stages a mask/preview
to disk; nothing touches the zarr until a human calls Accept.
"""

import json
import logging
import os
import shutil
import threading
import time
import uuid
from base64 import b64encode
from datetime import datetime
from types import SimpleNamespace

import numpy as np
from flask import jsonify
from PIL import Image

from cellmap_flow.ai_annotate.gemini_backend import generate_recolored_image
from cellmap_flow.ai_annotate.mask_extraction import extract_mask, slice_to_rgb
from cellmap_flow.ai_annotate.prompts import build_recolor_prompt
from cellmap_flow.dashboard.finetune_utils import _get_volume_metadata
from cellmap_flow.dashboard.routes.finetune.annotation_core import compute_context_crop_rois
from cellmap_flow.globals import g

logger = logging.getLogger(__name__)

AI_ANNOTATE_POINT_LAYER = "ai_annotate_point"
_TARGET_RGB = (255, 0, 0)
_FOREGROUND_LABEL = 2
_BACKGROUND_LABEL = 1

# Module-level progress tracker, keyed by volume_id (a server-initiated
# keypress has no client-generated id to key on, unlike load-crops/POST
# flows). One in-flight/staged AI-annotate result per volume at a time.
_PROGRESS: dict = {}
_PROGRESS_LOCK = threading.Lock()


def _set_progress(volume_id, **fields):
    with _PROGRESS_LOCK:
        entry = _PROGRESS.setdefault(volume_id, {})
        entry.update(fields)
        entry["updated_at"] = time.time()


def _get_progress(volume_id):
    with _PROGRESS_LOCK:
        entry = _PROGRESS.get(volume_id)
        return dict(entry) if entry else None


def _clear_progress(volume_id):
    with _PROGRESS_LOCK:
        _PROGRESS.pop(volume_id, None)


def _staging_dir(volume_meta, annotate_id):
    return os.path.join(volume_meta["corrections_dir"], ".ai_annotate_staging", annotate_id)


# ---------------------------------------------------------------------------
# Viewer integration: point layer, keybinding, action handler
# ---------------------------------------------------------------------------


def ensure_ai_annotate_point_layer(viewer):
    """Create the local point-prompt layer for AI-annotate if missing."""
    import neuroglancer

    with viewer.txn() as s:
        if AI_ANNOTATE_POINT_LAYER not in s.layers:
            s.layers[AI_ANNOTATE_POINT_LAYER] = neuroglancer.LocalAnnotationLayer(
                dimensions=neuroglancer.CoordinateSpace(
                    names=["z", "y", "x"],
                    units="nm",
                    scales=[1, 1, 1],
                ),
                annotationColor="#ffaa00",
            )
        s.layers[AI_ANNOTATE_POINT_LAYER].tool = neuroglancer.PlacePointTool()


def register_ai_annotate_keybinding(viewer, key="shift+keyg"):
    """Register the AI-annotate action on a neuroglancer viewer.

    Press the keybinding while a point is placed in AI_ANNOTATE_POINT_LAYER
    to send a crop around it to Gemini for a first-pass mask.
    """
    viewer.actions.add("ai-annotate", _ai_annotate_action_handler)
    with viewer.config_state.txn() as s:
        s.input_event_bindings.viewer[key] = "ai-annotate"
    logger.info(f"Registered AI-annotate keybinding: {key} (prompts: {AI_ANNOTATE_POINT_LAYER})")


def _get_ai_annotate_point_nm():
    """Read the most recently placed point from the AI-annotate point layer."""
    with g.viewer.txn() as s:
        state = s.to_json()

    points = []
    for layer in state.get("layers", []):
        if layer.get("name") != AI_ANNOTATE_POINT_LAYER:
            continue
        for ann in layer.get("annotations", []):
            if ann.get("type") != "point":
                continue
            p = ann.get("point")
            if p is None or len(p) < 3:
                continue
            points.append(np.array(p[:3], dtype=float))

    return points[-1] if points else None


def _find_ai_annotate_volume_id():
    """Find the first annotation volume that has AI-annotate enabled."""
    for volume_id, meta in g.annotation_volumes.items():
        if meta.get("ai_annotate_enabled"):
            return volume_id
    return None


def _set_neuroglancer_status(msg: str):
    try:
        with g.viewer.config_state.txn() as s:
            s.status_messages["ai_annotate"] = msg
    except Exception:
        pass


def _ai_annotate_action_handler(action_state):
    del action_state
    volume_id = _find_ai_annotate_volume_id()
    if volume_id is None:
        logger.info("AI-annotate keybinding pressed but no AI-annotate-enabled volume exists")
        _set_neuroglancer_status("AI-annotate: no AI-annotate-enabled volume — create one first")
        return

    point_nm = _get_ai_annotate_point_nm()
    if point_nm is None:
        _set_neuroglancer_status("AI-annotate: place a point first")
        return

    logger.info(f"AI-annotate triggered at point_nm={point_nm.tolist()}, volume={volume_id}")
    _set_neuroglancer_status("AI-annotate: sending crop to Gemini...")
    annotate_id = f"{uuid.uuid4().hex[:8]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    _set_progress(volume_id, status="running", annotate_id=annotate_id, error=None)

    thread = threading.Thread(
        target=_run_ai_annotate_safe, args=(point_nm, volume_id, annotate_id), daemon=True
    )
    thread.start()


def _run_ai_annotate_safe(point_nm, volume_id, annotate_id):
    try:
        run_ai_annotate(point_nm, volume_id, annotate_id)
        _set_neuroglancer_status("AI-annotate: ready for review")
    except Exception as e:
        logger.error(f"AI-annotate failed for volume {volume_id}: {e}", exc_info=True)
        _set_progress(volume_id, status="failed", error=str(e))
        _set_neuroglancer_status(f"AI-annotate: failed — {e}")


# ---------------------------------------------------------------------------
# Background pipeline: crop -> Gemini -> mask -> stage to disk (no zarr write)
# ---------------------------------------------------------------------------


def run_ai_annotate(point_nm, volume_id, annotate_id):
    from cellmap_flow.image_data_interface import ImageDataInterface
    from funlib.geometry import Coordinate, Roi

    volume_meta = _get_volume_metadata(volume_id)
    if volume_meta is None:
        raise ValueError(f"Unknown volume_id: {volume_id}")

    chunk_size = np.array(volume_meta["output_size"])
    output_voxel_size = np.array(volume_meta["output_voxel_size"])
    input_size = np.array(volume_meta["input_size"])
    input_voxel_size = np.array(volume_meta["input_voxel_size"])
    dataset_offset_nm = np.array(volume_meta["dataset_offset_nm"])
    dataset_path = volume_meta["dataset_path"]
    label_name = volume_meta.get("ai_annotate_label_name") or "labeled structure"
    gemini_model = volume_meta.get("ai_annotate_gemini_model") or "gemini-3-pro-image"

    # Context/read crop centered on the click, reusing the same centering
    # math as the manual "create crop at view center" flow.
    context_config = SimpleNamespace(
        read_shape=input_size * input_voxel_size,
        write_shape=chunk_size * output_voxel_size,
        input_voxel_size=input_voxel_size,
        output_voxel_size=output_voxel_size,
    )
    raw_crop_shape_voxels, _, raw_crop_offset_voxels, _ = compute_context_crop_rois(
        point_nm, context_config
    )
    raw_crop_world_offset_nm = raw_crop_offset_voxels * input_voxel_size

    idi = ImageDataInterface(dataset_path, voxel_size=input_voxel_size)
    roi = Roi(
        offset=Coordinate(raw_crop_offset_voxels * input_voxel_size),
        shape=Coordinate(raw_crop_shape_voxels * input_voxel_size),
    )
    raw_crop = idi.to_ndarray_ts(roi)

    # Destination chunk: grid-aligned, independent of the (click-centered)
    # context crop above -- deliberately one click -> one z-row of one chunk,
    # not SAM's multi-chunk halo.
    point_output_vox = (point_nm - dataset_offset_nm) / output_voxel_size
    chunk_idx = np.floor(point_output_vox / chunk_size).astype(int)
    chunk_start_output_vox = chunk_idx * chunk_size
    z_row_index = int(np.clip(np.floor(point_output_vox[0] - chunk_start_output_vox[0]), 0, chunk_size[0] - 1))
    chunk_offset_nm = dataset_offset_nm + chunk_start_output_vox * output_voxel_size
    chunk_shape_nm = chunk_size * output_voxel_size

    # Pixel window within the raw context crop covering the destination
    # chunk's exact XY footprint, and the click's z-slice.
    y_pix0 = int(round((chunk_offset_nm[1] - raw_crop_world_offset_nm[1]) / input_voxel_size[1]))
    x_pix0 = int(round((chunk_offset_nm[2] - raw_crop_world_offset_nm[2]) / input_voxel_size[2]))
    y_pix1 = y_pix0 + int(round(chunk_shape_nm[1] / input_voxel_size[1]))
    x_pix1 = x_pix0 + int(round(chunk_shape_nm[2] / input_voxel_size[2]))
    z_pix = int(round((point_nm[0] - raw_crop_world_offset_nm[0]) / input_voxel_size[0]))

    z_pix = int(np.clip(z_pix, 0, raw_crop.shape[0] - 1))
    y_pix0c = int(np.clip(y_pix0, 0, raw_crop.shape[1]))
    y_pix1c = int(np.clip(y_pix1, 0, raw_crop.shape[1]))
    x_pix0c = int(np.clip(x_pix0, 0, raw_crop.shape[2]))
    x_pix1c = int(np.clip(x_pix1, 0, raw_crop.shape[2]))
    if y_pix1c <= y_pix0c or x_pix1c <= x_pix0c:
        raise ValueError(
            "Destination chunk footprint falls outside the fetched context crop "
            f"(volume={volume_id}); increase the model's read_shape or move the click "
            "away from the dataset edge."
        )
    if (y_pix0c, y_pix1c) != (y_pix0, y_pix1) or (x_pix0c, x_pix1c) != (x_pix0, x_pix1):
        logger.warning(
            f"AI-annotate chunk footprint clipped to context crop bounds for volume {volume_id}; "
            "resulting mask will only cover the overlapping region."
        )

    em_slice = raw_crop[z_pix, y_pix0c:y_pix1c, x_pix0c:x_pix1c]

    input_image = slice_to_rgb(em_slice)
    prompt = build_recolor_prompt(label_name)
    recolored_image = generate_recolored_image(
        input_image,
        prompt,
        model=gemini_model,
        vertex_project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
    )
    mask = extract_mask(input_image, recolored_image, target_rgb=_TARGET_RGB)

    # `mask` is at input-voxel pixel resolution (em_slice's shape). The
    # destination chunk footprint is at output-voxel resolution, which can
    # differ in pixel count for the same physical extent -- so the resize
    # target and any clipped-edge offsets must be computed in output-voxel
    # pixels, not reused from the input-voxel window directly.
    intended_y_len = y_pix1 - y_pix0
    intended_x_len = x_pix1 - x_pix0
    out_dy0 = int(round((y_pix0c - y_pix0) / intended_y_len * chunk_size[1]))
    out_dy1 = int(round((y_pix1 - y_pix1c) / intended_y_len * chunk_size[1]))
    out_dx0 = int(round((x_pix0c - x_pix0) / intended_x_len * chunk_size[2]))
    out_dx1 = int(round((x_pix1 - x_pix1c) / intended_x_len * chunk_size[2]))
    out_y_len = int(chunk_size[1]) - out_dy0 - out_dy1
    out_x_len = int(chunk_size[2]) - out_dx0 - out_dx1

    dest_shape = (int(chunk_size[1]), int(chunk_size[2]))
    mask_full = np.zeros(dest_shape, dtype=np.uint8)
    mask_resized = np.array(Image.fromarray(mask).resize((out_x_len, out_y_len), Image.NEAREST))
    mask_full[out_dy0 : out_dy0 + out_y_len, out_dx0 : out_dx0 + out_x_len] = mask_resized

    _stage_result(
        volume_id=volume_id,
        annotate_id=annotate_id,
        volume_meta=volume_meta,
        point_nm=point_nm,
        chunk_indices=tuple(int(v) for v in chunk_idx),
        z_row_index=z_row_index,
        mask_for_write=mask_full,
        mask_for_preview=mask,
        input_image=input_image,
        recolored_image=recolored_image,
    )


def _stage_result(
    volume_id,
    annotate_id,
    volume_meta,
    point_nm,
    chunk_indices,
    z_row_index,
    mask_for_write,
    mask_for_preview,
    input_image,
    recolored_image,
):
    staging_dir = _staging_dir(volume_meta, annotate_id)
    os.makedirs(staging_dir, exist_ok=True)

    np.save(os.path.join(staging_dir, "mask.npy"), mask_for_write)

    # mask_for_preview is at input_image's own resolution (pre-downsample),
    # so the overlay composite (built from input_image) stays same-sized.
    overlay = np.array(input_image).copy()
    overlay[mask_for_preview > 0] = (
        0.5 * overlay[mask_for_preview > 0] + 0.5 * np.array(_TARGET_RGB)
    ).astype(np.uint8)
    composite = Image.new("RGB", (input_image.width * 3, input_image.height))
    composite.paste(input_image, (0, 0))
    composite.paste(recolored_image, (input_image.width, 0))
    composite.paste(Image.fromarray(overlay), (input_image.width * 2, 0))
    composite.save(os.path.join(staging_dir, "preview.png"))

    meta = {
        "volume_id": volume_id,
        "annotate_id": annotate_id,
        "point_nm": point_nm.tolist(),
        "chunk_indices": list(chunk_indices),
        "z_row_index": z_row_index,
        "label_id": _FOREGROUND_LABEL,
        "background_label_id": _BACKGROUND_LABEL,
        "created_at": datetime.now().isoformat(),
    }
    with open(os.path.join(staging_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    _set_progress(volume_id, status="ready", annotate_id=annotate_id, error=None)


# ---------------------------------------------------------------------------
# Review: status polling + Accept/Reject
# ---------------------------------------------------------------------------


def get_ai_annotate_status_response(volume_id):
    if not volume_id:
        return jsonify({"success": False, "error": "Missing 'volume_id' query param"}), 400

    entry = _get_progress(volume_id)
    if entry is None:
        return jsonify({"success": True, "status": "idle"})

    result = {"success": True, "status": entry.get("status"), "annotate_id": entry.get("annotate_id")}
    if entry.get("status") == "failed":
        result["error"] = entry.get("error")
    elif entry.get("status") == "ready":
        volume_meta = _get_volume_metadata(volume_id)
        staging_dir = _staging_dir(volume_meta, entry["annotate_id"])
        preview_path = os.path.join(staging_dir, "preview.png")
        if os.path.exists(preview_path):
            with open(preview_path, "rb") as f:
                result["preview_png_base64"] = b64encode(f.read()).decode("ascii")
    return jsonify(result)


def accept_ai_annotate_response(data):
    from cellmap_flow.dashboard.routes.finetune.overlay import (
        _invalidate_annotation_layer,
        write_ai_mask_to_minio,
    )

    volume_id = data.get("volume_id")
    if not volume_id:
        return jsonify({"success": False, "error": "Missing volume_id"}), 400

    entry = _get_progress(volume_id)
    if entry is None or entry.get("status") != "ready":
        return jsonify({"success": False, "error": "No AI-annotate result ready for review"}), 404

    volume_meta = _get_volume_metadata(volume_id)
    staging_dir = _staging_dir(volume_meta, entry["annotate_id"])
    try:
        mask = np.load(os.path.join(staging_dir, "mask.npy"))
        with open(os.path.join(staging_dir, "meta.json")) as f:
            meta = json.load(f)

        write_ai_mask_to_minio(
            volume_id,
            tuple(meta["chunk_indices"]),
            meta["z_row_index"],
            mask,
            label_id=meta["label_id"],
            background_label_id=meta["background_label_id"],
        )
        _invalidate_annotation_layer(volume_id)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
        _clear_progress(volume_id)

    return jsonify({"success": True, "message": "AI-annotate mask accepted and written"})


def reject_ai_annotate_response(data):
    volume_id = data.get("volume_id")
    if not volume_id:
        return jsonify({"success": False, "error": "Missing volume_id"}), 400

    entry = _get_progress(volume_id)
    if entry and entry.get("annotate_id"):
        volume_meta = _get_volume_metadata(volume_id)
        if volume_meta is not None:
            shutil.rmtree(_staging_dir(volume_meta, entry["annotate_id"]), ignore_errors=True)
    _clear_progress(volume_id)

    return jsonify({"success": True, "message": "AI-annotate result rejected"})
