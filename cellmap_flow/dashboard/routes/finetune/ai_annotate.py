"""Click a point in the viewer -> send an EM crop to Gemini for a first-pass
segmentation mask -> human reviews -> accepted mask is written into the
sparse annotation-volume zarr as new training data.

Point capture (LocalAnnotationLayer + PlacePointTool + keybinding, action
handler shape) is ported from sam-backend-support's sam_annotator.py, which
built the same point-prompt -> model -> mask flow for SAM but wrote directly
with no review step. Here the background pipeline only stages a mask/preview
to disk; nothing touches the zarr until a human calls Accept.

The click can be made in any of neuroglancer's three orthogonal cross-section
views (XY, XZ, YZ) -- the viewer's ``layout`` field tells us which single
plane is being shown, which maps to which of our three declared axes
(z, y, x) is currently "depth" (the axis we slice along). Everything below
is generalized over that axis rather than assuming z is always depth. This
only works for single-plane layouts ("xy"/"xz"/"yz", optionally with a
"-3d" side panel); neuroglancer's multi-panel layouts ("4panel", "3d") share
one global viewer state with no per-panel field, so which of several
simultaneously-visible planes a click actually landed in is not recoverable
server-side -- see depth_axis_from_layout.
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

import numpy as np
import zarr
from flask import jsonify
from PIL import Image

from cellmap_flow.ai_annotate.gemini_backend import generate_recolored_image, upsample_scale_for_edit
from cellmap_flow.ai_annotate.mask_extraction import extract_mask, slice_to_rgb
from cellmap_flow.ai_annotate.organelles import resolve_organelle_profile
from cellmap_flow.ai_annotate.prompts import build_recolor_prompt
from cellmap_flow.dashboard.finetune_utils import _get_volume_metadata
from cellmap_flow.globals import g

logger = logging.getLogger(__name__)

AI_ANNOTATE_POINT_LAYER = "ai_annotate_point"
AI_ANNOTATE_PREVIEW_BOX_LAYER = "ai_annotate_preview_box"
AXIS_NAMES = ["z", "y", "x"]
_FOREGROUND_LABEL = 2
_BACKGROUND_LABEL = 1

# Physical size of the EM crop fetched for Gemini, expressed in pixels *at
# the model's output-voxel resolution* (not input-voxel resolution, and
# independent of the model's own often-much-smaller input_size). Fetching
# that physical extent at the (typically finer) input voxel size upsamples
# it "for free" -- e.g. 2x when output_voxel_size = 2x input_voxel_size --
# matching what Gemini's own upsample-before-send step (gemini_backend.
# _upsample_for_edit, floor=1024) would otherwise have to do from scratch.
GEMINI_CROP_SIZE_VOXELS = 512

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


# Depth axis (the one normal to the cross-section plane) for each of
# neuroglancer's single-plane layout names, given our declared axis order
# ["z", "y", "x"]: an "xy" view is a slice at constant z, "xz" at constant y,
# "yz" at constant x. Derived from neuroglancer's default per-plane
# orientation offsets (data_panel_layout.ts's AXES_RELATIVE_ORIENTATION),
# not from displayDimensions -- that field is for picking 2-of-N dims when
# a coordinate space has more than 3 axes (e.g. a time dim) and does NOT
# change when the user switches between XY/XZ/YZ for a plain 3-axis space.
_LAYOUT_DEPTH_AXIS_NAME = {"xy": "x", "xz": "y", "yz": "z"}


def depth_axis_from_layout(layout):
    """Return the index (into AXIS_NAMES = [z, y, x]) of the axis normal to
    the cross-section plane named by the viewer's ``layout`` field.

    Only unambiguous for single-plane layouts ("xy"/"xz"/"yz", optionally
    suffixed "-3d" for a side 3D panel): neuroglancer serializes ONE shared
    layout/crossSectionOrientation for the whole viewer state, so in a
    multi-panel layout ("4panel", "4panel-alt", "3d") there is no server-side
    way to tell which of the several simultaneously-visible planes a given
    click actually landed in. Defaults to "xy" (z is depth) in that case,
    same as neuroglancer's own default plane.
    """
    layout_type = layout if isinstance(layout, str) else (layout or {}).get("type")
    if layout_type:
        layout_type = layout_type.removesuffix("-3d")
    axis_name = _LAYOUT_DEPTH_AXIS_NAME.get(layout_type, "x")
    return AXIS_NAMES.index(axis_name)


def _get_ai_annotate_prompt_state():
    """Read the most recently placed point, the depth axis of the plane it
    was placed in, and the name of the layer currently selected in
    neuroglancer's layer panel (used to disambiguate which AI-annotate
    volume to target when more than one is loaded -- see
    _find_ai_annotate_volume_id).

    Depth axis is read straight off the point annotation's ``description``,
    which a matching client-side patch (PlacePointTool.trigger in
    ui/annotations.ts, this repo's vendored neuroglancer fork) stamps with
    the axis name ("z"/"y"/"x") normal to whichever cross-section panel was
    under the mouse at the moment the point was clicked -- captured once,
    at click time, so it's correct even in a 4-panel layout and unaffected
    by the mouse moving to a different panel before Shift+G is pressed.
    Falls back to depth_axis_from_layout (the current single-plane layout,
    ambiguous in multi-panel layouts) for points placed before that patch
    existed, or placed via a panel with no well-defined depth axis (e.g. 3D).
    """
    with g.viewer.txn() as s:
        state = s.to_json()

    fallback_depth_axis = depth_axis_from_layout(state.get("layout"))
    selected_layer = (state.get("selectedLayer") or {}).get("layer")

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
            points.append((np.array(p[:3], dtype=float), ann.get("description")))

    if not points:
        return None, fallback_depth_axis, selected_layer

    point_nm, depth_axis_name = points[-1]
    depth_axis = AXIS_NAMES.index(depth_axis_name) if depth_axis_name in AXIS_NAMES else fallback_depth_axis
    return point_nm, depth_axis, selected_layer


def _find_ai_annotate_volume_id(selected_layer_name=None):
    """Pick which AI-annotate-enabled volume to target.

    Multiple AI-annotate-enabled volumes (e.g. against different datasets)
    can be loaded into the viewer at once, each with its own annotation
    layer tracked in g.minio_state["annotation_layers"][volume_id]. Prefer
    the volume whose annotation layer is currently selected in neuroglancer's
    layer panel; otherwise fall back to the first AI-annotate-enabled volume,
    which is correct whenever there's only one.
    """
    enabled_ids = [
        volume_id for volume_id, meta in g.annotation_volumes.items() if meta.get("ai_annotate_enabled")
    ]
    if selected_layer_name:
        annotation_layers = getattr(g, "minio_state", {}).get("annotation_layers", {})
        for volume_id in enabled_ids:
            if selected_layer_name in annotation_layers.get(volume_id, []):
                return volume_id
    return enabled_ids[0] if enabled_ids else None


def _set_neuroglancer_status(msg: str):
    try:
        with g.viewer.config_state.txn() as s:
            s.status_messages["ai_annotate"] = msg
    except Exception:
        pass


# Human-readable labels for the coarse-grained stages of a run, surfaced both
# in neuroglancer's status overlay and via get_ai_annotate_status_response so
# the dashboard tab can show something more useful than a bare "running"
# spinner during the ~tens-of-seconds Gemini round trip.
_STAGE_LABELS = {
    "fetching_crop": "Fetching EM crop...",
    "sending_to_gemini": "Sending crop to Gemini...",
    "extracting_mask": "Extracting mask from Gemini's response...",
    "staging_preview": "Building review preview...",
}


def _set_stage(volume_id, stage):
    _set_progress(volume_id, stage=stage)
    _set_neuroglancer_status(f"AI-annotate: {_STAGE_LABELS.get(stage, stage)}")


def _compute_context_crop_geometry(point_nm, volume_meta):
    """World-space offset/shape (nm) of the context crop centered on the
    click -- the physical FOV that gets fetched raw EM data for and sent to
    Gemini (see GEMINI_CROP_SIZE_VOXELS). Shared by run_ai_annotate and the
    preview-box drawing so the preview always matches what's actually sent.
    """
    output_voxel_size = np.array(volume_meta["output_voxel_size"])
    context_shape_nm = np.full(3, GEMINI_CROP_SIZE_VOXELS) * output_voxel_size
    context_offset_nm = point_nm - context_shape_nm / 2
    return context_offset_nm, context_shape_nm


def _draw_preview_box(point_nm, depth_axis, context_offset_nm, context_shape_nm, volume_meta):
    """Show the context crop that will be sent to Gemini as a flat plane --
    one output-voxel thick along the depth axis, centered exactly on the
    click -- so the user can see the FOV before the Gemini call returns. A
    full 3D box extending through many depth slices reads confusingly as a
    2D crop, and using the click-centered context crop (rather than the
    grid-aligned destination chunk) keeps it visibly centered on the click.
    """
    import neuroglancer

    output_voxel_size = np.array(volume_meta["output_voxel_size"])
    plane_offset_nm = context_offset_nm.copy()
    plane_shape_nm = context_shape_nm.copy()
    plane_offset_nm[depth_axis] = point_nm[depth_axis] - output_voxel_size[depth_axis] / 2
    plane_shape_nm[depth_axis] = output_voxel_size[depth_axis]

    try:
        with g.viewer.txn() as s:
            s.layers[AI_ANNOTATE_PREVIEW_BOX_LAYER] = neuroglancer.LocalAnnotationLayer(
                dimensions=neuroglancer.CoordinateSpace(
                    names=["z", "y", "x"],
                    units="nm",
                    scales=[1, 1, 1],
                ),
                annotationColor="#ffaa00",
                annotations=[
                    neuroglancer.AxisAlignedBoundingBoxAnnotation(
                        point_a=plane_offset_nm.tolist(),
                        point_b=(plane_offset_nm + plane_shape_nm).tolist(),
                        id="ai_annotate_target_chunk",
                    )
                ],
            )
    except Exception as e:
        logger.warning(f"Could not draw AI-annotate preview box: {e}")


def _ai_annotate_action_handler(action_state):
    del action_state
    point_nm, depth_axis, selected_layer = _get_ai_annotate_prompt_state()
    volume_id = _find_ai_annotate_volume_id(selected_layer)
    if volume_id is None:
        logger.info("AI-annotate keybinding pressed but no AI-annotate-enabled volume exists")
        _set_neuroglancer_status("AI-annotate: no AI-annotate-enabled volume — create one first")
        return

    if point_nm is None:
        _set_neuroglancer_status("AI-annotate: place a point first")
        return

    volume_meta = _get_volume_metadata(volume_id)
    if volume_meta is not None:
        context_offset_nm, context_shape_nm = _compute_context_crop_geometry(point_nm, volume_meta)
        _draw_preview_box(point_nm, depth_axis, context_offset_nm, context_shape_nm, volume_meta)

    logger.info(
        f"AI-annotate triggered at point_nm={point_nm.tolist()}, "
        f"depth_axis={AXIS_NAMES[depth_axis]}, volume={volume_id}"
    )
    annotate_id = f"{uuid.uuid4().hex[:8]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    _set_progress(volume_id, status="running", annotate_id=annotate_id, error=None, stage=None)
    _set_stage(volume_id, "fetching_crop")

    thread = threading.Thread(
        target=_run_ai_annotate_safe,
        args=(point_nm, depth_axis, volume_id, annotate_id),
        daemon=True,
    )
    thread.start()


def _run_ai_annotate_safe(point_nm, depth_axis, volume_id, annotate_id):
    try:
        run_ai_annotate(point_nm, depth_axis, volume_id, annotate_id)
        _set_neuroglancer_status("AI-annotate: ready for review")
    except Exception as e:
        logger.error(f"AI-annotate failed for volume {volume_id}: {e}", exc_info=True)
        _set_progress(volume_id, status="failed", error=str(e))
        _set_neuroglancer_status(f"AI-annotate: failed — {e}")


# ---------------------------------------------------------------------------
# Background pipeline: crop -> Gemini -> mask -> stage to disk (no zarr write)
# ---------------------------------------------------------------------------


def _compute_context_write_region(point_nm, depth_axis, context_offset_nm, volume_meta):
    """Absolute output-voxel index of the write region's origin.

    The ENTIRE reviewed context crop is written on Accept (not a single
    grid-aligned model-output chunk) -- the whole visible area is trusted as
    annotated ground truth. The depth_axis component is the single absolute
    output-voxel index of the plane; the other two components are the
    crop's absolute top-left corner. May span many chunks of the annotation
    zarr; write_ai_mask_to_minio clips to the volume's actual bounds.
    """
    output_voxel_size = np.array(volume_meta["output_voxel_size"])
    dataset_offset_nm = np.array(volume_meta["dataset_offset_nm"])

    write_offset_vox = np.round((context_offset_nm - dataset_offset_nm) / output_voxel_size).astype(int)
    write_offset_vox[depth_axis] = int(
        round((point_nm[depth_axis] - dataset_offset_nm[depth_axis]) / output_voxel_size[depth_axis])
    )
    return write_offset_vox


def run_ai_annotate(point_nm, depth_axis, volume_id, annotate_id):
    from cellmap_flow.image_data_interface import ImageDataInterface
    from funlib.geometry import Coordinate, Roi

    volume_meta = _get_volume_metadata(volume_id)
    if volume_meta is None:
        raise ValueError(f"Unknown volume_id: {volume_id}")

    input_voxel_size = np.array(volume_meta["input_voxel_size"])
    output_voxel_size = np.array(volume_meta["output_voxel_size"])
    dataset_path = volume_meta["dataset_path"]
    label_name = volume_meta.get("ai_annotate_label_name") or "labeled structure"
    gemini_model = volume_meta.get("ai_annotate_gemini_model") or "gemini-3-pro-image"
    prompt_override = volume_meta.get("ai_annotate_prompt_override")
    profile = resolve_organelle_profile(label_name)

    # Context/read crop centered on the click, sized for Gemini/the reviewer
    # (GEMINI_CROP_SIZE_VOXELS, at output-voxel resolution) rather than the
    # finetuning model's own (often much smaller) write_shape -- this is what
    # gets sent to Gemini, shown in the review preview, AND (in full) written
    # on Accept, so there's enough surrounding context to actually see and
    # for Gemini to reason about.
    context_offset_nm, context_shape_nm = _compute_context_crop_geometry(point_nm, volume_meta)
    raw_crop_shape_voxels = np.round(context_shape_nm / input_voxel_size).astype(int)
    raw_crop_offset_voxels = np.round(context_offset_nm / input_voxel_size).astype(int)
    raw_crop_world_offset_nm = raw_crop_offset_voxels * input_voxel_size

    depth_pix = int(
        round((point_nm[depth_axis] - raw_crop_world_offset_nm[depth_axis]) / input_voxel_size[depth_axis])
    )
    depth_pix = int(np.clip(depth_pix, 0, raw_crop_shape_voxels[depth_axis] - 1))

    # Only a single input-voxel-thick slab along depth_axis is ever used
    # (see np.take below) -- fetching the full cubic context (same extent on
    # all 3 axes) here would multiply the read volume by ~GEMINI_CROP_SIZE_
    # VOXELS along that axis for no benefit, which is prohibitively slow for
    # large/sharded remote Zarr stores (observed: minutes instead of ~instant).
    slab_shape_voxels = raw_crop_shape_voxels.copy()
    slab_shape_voxels[depth_axis] = 1
    slab_offset_voxels = raw_crop_offset_voxels.copy()
    slab_offset_voxels[depth_axis] += depth_pix

    idi = ImageDataInterface(dataset_path, voxel_size=input_voxel_size)
    roi = Roi(
        offset=Coordinate(slab_offset_voxels * input_voxel_size),
        shape=Coordinate(slab_shape_voxels * input_voxel_size),
    )
    raw_crop = idi.to_ndarray_ts(roi)

    in_plane_axes = tuple(i for i in range(3) if i != depth_axis)

    # Full in-plane slice through the click -- e.g. for depth_axis=0 (z) this
    # is the full GEMINI_CROP_SIZE_VOXELS^2 (y, x) plane, not a narrow crop,
    # so both Gemini and the reviewer get real surrounding context.
    full_em_slice = np.take(raw_crop, 0, axis=depth_axis)

    input_image = slice_to_rgb(full_em_slice)

    # The prompt's stated resolution must match what Gemini actually
    # receives, not just the fetched crop's resolution: generate_recolored_image
    # internally upsamples the image further (gemini_backend._upsample_for_edit,
    # floor=1024) before sending, whenever the crop is smaller than that floor.
    in_plane_input_voxel_size = input_voxel_size[list(in_plane_axes)]
    upsample_scale = upsample_scale_for_edit(input_image.width, input_image.height)
    resolution_nm = float(np.mean(in_plane_input_voxel_size)) / upsample_scale

    prompt = build_recolor_prompt(profile, resolution_nm=resolution_nm, prompt_override=prompt_override)
    _set_stage(volume_id, "sending_to_gemini")
    recolored_image = generate_recolored_image(
        input_image,
        prompt,
        model=gemini_model,
        vertex_project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
    )
    _set_stage(volume_id, "extracting_mask")
    mask_full_res = extract_mask(input_image, recolored_image, target_rgb=profile.rgb)

    # Absolute output-voxel write origin -- the WHOLE reviewed context crop
    # is trusted as annotated ground truth and gets painted on Accept (see
    # write_ai_mask_to_minio), not just a small grid-aligned model chunk.
    write_offset_vox = _compute_context_write_region(point_nm, depth_axis, context_offset_nm, volume_meta)

    # mask_full_res is at input-voxel pixel resolution; the write footprint
    # is at output-voxel resolution, which can differ in pixel count for the
    # same physical extent.
    context_shape_output_vox = np.round(context_shape_nm / output_voxel_size).astype(int)
    dest_shape = tuple(int(context_shape_output_vox[axis]) for axis in in_plane_axes)
    mask_geom = {"dest_shape": list(dest_shape)}
    mask_for_write = _mask_full_to_write(mask_full_res, mask_geom)

    _set_stage(volume_id, "staging_preview")
    _stage_result(
        volume_id=volume_id,
        annotate_id=annotate_id,
        volume_meta=volume_meta,
        point_nm=point_nm,
        write_offset_vox=tuple(int(v) for v in write_offset_vox),
        depth_axis=depth_axis,
        mask_for_write=mask_for_write,
        input_image=input_image,
        recolored_image=recolored_image,
        mask_for_preview=mask_full_res,
        target_rgb=profile.rgb,
        prompt=prompt,
        mask_geom=mask_geom,
        gemini_model=gemini_model,
    )


def _mask_full_to_write(mask_full_res, mask_geom):
    """NEAREST-resize the full-context mask (at input-voxel pixel
    resolution) to the write footprint's output-voxel pixel resolution. The
    write footprint is exactly co-extensive with the fetched crop -- the
    whole thing gets written -- so no destination-chunk sub-windowing is
    needed. Shared by the initial run and a prompt-edit resend, since a
    resend reuses the same geometry against a freshly-generated mask.

    Instance separation (a background gap between adjacent/touching
    instances, needed for crop_loader.py's connected_components=True import
    path to split them correctly) is asked of Gemini directly in the prompt
    (see prompts.build_recolor_prompt) rather than eroded in here -- Gemini
    already sees individual instance boundaries and can draw the gap
    correctly, whereas a fixed post-hoc erosion is a blunter, one-size-fits-
    all approximation.
    """
    dest_shape = tuple(mask_geom["dest_shape"])
    return np.array(Image.fromarray(mask_full_res).resize((dest_shape[1], dest_shape[0]), Image.NEAREST))


def _stage_result(
    volume_id,
    annotate_id,
    volume_meta,
    point_nm,
    write_offset_vox,
    depth_axis,
    mask_for_write,
    input_image,
    recolored_image,
    mask_for_preview,
    target_rgb,
    prompt,
    mask_geom,
    gemini_model,
):
    staging_dir = _staging_dir(volume_meta, annotate_id)
    os.makedirs(staging_dir, exist_ok=True)

    np.save(os.path.join(staging_dir, "mask.npy"), mask_for_write)
    # Saved so a "resend with edited prompt" can re-call Gemini without
    # re-fetching the EM crop from the dataset.
    input_image.save(os.path.join(staging_dir, "input.png"))

    # mask_for_preview is at input_image's own (full-context) resolution,
    # so the overlay composite built from input_image stays same-sized. The
    # whole thing gets written on Accept, so there's no separate "write
    # target" sub-region to outline anymore -- the overlay IS the target.
    overlay = np.array(input_image).copy()
    overlay[mask_for_preview > 0] = (
        0.5 * overlay[mask_for_preview > 0] + 0.5 * np.array(target_rgb)
    ).astype(np.uint8)
    overlay_image = Image.fromarray(overlay)

    composite = Image.new("RGB", (input_image.width * 3, input_image.height))
    composite.paste(input_image, (0, 0))
    composite.paste(recolored_image, (input_image.width, 0))
    composite.paste(overlay_image, (input_image.width * 2, 0))
    composite.save(os.path.join(staging_dir, "preview.png"))

    meta = {
        "volume_id": volume_id,
        "annotate_id": annotate_id,
        "point_nm": point_nm.tolist(),
        "write_offset_vox": list(write_offset_vox),
        "depth_axis": depth_axis,
        "label_id": _FOREGROUND_LABEL,
        "background_label_id": _BACKGROUND_LABEL,
        "prompt": prompt,
        "target_rgb": list(target_rgb),
        "mask_geom": mask_geom,
        "gemini_model": gemini_model,
        "created_at": datetime.now().isoformat(),
    }
    with open(os.path.join(staging_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    _set_progress(volume_id, status="ready", annotate_id=annotate_id, error=None)


# ---------------------------------------------------------------------------
# Organelle catalog, for the "AI-assisted annotation" label dropdown
# ---------------------------------------------------------------------------


def list_ai_annotate_organelles_response():
    from cellmap_flow.ai_annotate.organelles import ORGANELLES

    organelles = [
        {"key": key, "name": profile.name, "prompt": build_recolor_prompt(profile)}
        for key, profile in sorted(ORGANELLES.items(), key=lambda kv: kv[1].name)
    ]
    return jsonify({"success": True, "organelles": organelles})


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
    if entry.get("status") == "running":
        stage = entry.get("stage")
        result["stage"] = stage
        result["stage_label"] = _STAGE_LABELS.get(stage, stage)
    elif entry.get("status") == "failed":
        result["error"] = entry.get("error")
    elif entry.get("status") == "ready":
        volume_meta = _get_volume_metadata(volume_id)
        staging_dir = _staging_dir(volume_meta, entry["annotate_id"])
        preview_path = os.path.join(staging_dir, "preview.png")
        if os.path.exists(preview_path):
            with open(preview_path, "rb") as f:
                result["preview_png_base64"] = b64encode(f.read()).decode("ascii")
        meta_path = os.path.join(staging_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                result["prompt"] = json.load(f).get("prompt")
    return jsonify(result)


def update_ai_annotate_prompt_response(data):
    """Update the prompt override on an already-created volume.

    Without this, ``ai_annotate_prompt_override`` is only ever set once, at
    volume-creation time, and cached in-memory (``_get_volume_metadata``
    returns the cached dict rather than re-reading zarr attrs) -- editing
    the prompt textbox in the UI after that point had no effect on Shift+G,
    since nothing ever pushed the edit back into the cache or to disk.
    """
    volume_id = data.get("volume_id")
    if not volume_id:
        return jsonify({"success": False, "error": "Missing volume_id"}), 400
    new_prompt_override = data.get("ai_annotate_prompt_override") or None

    volume_meta = _get_volume_metadata(volume_id)
    if volume_meta is None:
        return jsonify({"success": False, "error": f"Unknown volume_id: {volume_id}"}), 404

    try:
        root = zarr.open(volume_meta["zarr_path"], mode="r+")
        root.attrs["ai_annotate_prompt_override"] = new_prompt_override
    except Exception as e:
        logger.error(f"Error persisting prompt override for {volume_id}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

    # volume_meta is the actual cached dict (see _get_volume_metadata), so
    # this mutation is visible to the next Shift+G's run_ai_annotate call.
    volume_meta["ai_annotate_prompt_override"] = new_prompt_override

    return jsonify({"success": True, "ai_annotate_prompt_override": new_prompt_override})


def resend_ai_annotate_response(data):
    volume_id = data.get("volume_id")
    new_prompt = (data.get("prompt") or "").strip()
    if not volume_id or not new_prompt:
        return jsonify({"success": False, "error": "Missing volume_id or prompt"}), 400

    entry = _get_progress(volume_id)
    if entry is None or entry.get("status") != "ready":
        return jsonify({"success": False, "error": "No AI-annotate result ready to resend"}), 404

    annotate_id = entry["annotate_id"]
    _set_progress(volume_id, status="running", annotate_id=annotate_id, error=None, stage=None)
    _set_stage(volume_id, "sending_to_gemini")

    thread = threading.Thread(
        target=_resend_ai_annotate_safe,
        args=(volume_id, annotate_id, new_prompt),
        daemon=True,
    )
    thread.start()
    return jsonify({"success": True, "message": "Resending to Gemini with edited prompt"})


def _resend_ai_annotate_safe(volume_id, annotate_id, new_prompt):
    try:
        _resend_ai_annotate(volume_id, annotate_id, new_prompt)
        _set_neuroglancer_status("AI-annotate: ready for review")
    except Exception as e:
        logger.error(f"AI-annotate resend failed for volume {volume_id}: {e}", exc_info=True)
        _set_progress(volume_id, status="failed", error=str(e))
        _set_neuroglancer_status(f"AI-annotate: failed — {e}")


def _resend_ai_annotate(volume_id, annotate_id, new_prompt):
    """Re-run just the Gemini call + mask extraction with an edited prompt,
    reusing the EM crop and destination geometry already staged from the
    original run (no re-fetch from the dataset, no re-computed click
    geometry) -- overwrites the same staged result in place.
    """
    volume_meta = _get_volume_metadata(volume_id)
    if volume_meta is None:
        raise ValueError(f"Unknown volume_id: {volume_id}")

    staging_dir = _staging_dir(volume_meta, annotate_id)
    with open(os.path.join(staging_dir, "meta.json")) as f:
        meta = json.load(f)
    input_image = Image.open(os.path.join(staging_dir, "input.png")).convert("RGB")
    target_rgb = tuple(meta["target_rgb"])
    gemini_model = meta.get("gemini_model") or "gemini-3-pro-image"

    recolored_image = generate_recolored_image(
        input_image,
        new_prompt,
        model=gemini_model,
        vertex_project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
    )
    _set_stage(volume_id, "extracting_mask")
    mask_full_res = extract_mask(input_image, recolored_image, target_rgb=target_rgb)
    mask_geom = meta["mask_geom"]
    mask_for_write = _mask_full_to_write(mask_full_res, mask_geom)

    _set_stage(volume_id, "staging_preview")
    _stage_result(
        volume_id=volume_id,
        annotate_id=annotate_id,
        volume_meta=volume_meta,
        point_nm=np.array(meta["point_nm"]),
        write_offset_vox=tuple(meta["write_offset_vox"]),
        depth_axis=meta["depth_axis"],
        mask_for_write=mask_for_write,
        input_image=input_image,
        recolored_image=recolored_image,
        mask_for_preview=mask_full_res,
        target_rgb=target_rgb,
        prompt=new_prompt,
        mask_geom=mask_geom,
        gemini_model=gemini_model,
    )


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
            tuple(meta["write_offset_vox"]),
            meta["depth_axis"],
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
