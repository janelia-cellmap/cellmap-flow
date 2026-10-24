"""From a planned plane to labels: read the raw, ask the model, label the box.

``read_plane`` reads the plane ``geometry.plan_plane`` planned, on the image
grid it planned and laid out as on screen; ``build_request`` makes it the
request a backend sends; ``mask_to_write_shape`` cuts the mask the backend
returns into its connected objects and puts them onto the write box's
voxels, and ``labels_for_box`` (``accept_labels`` for Accept) turns those
into the labels ``fill.paint_box`` writes: one id per object, background
elsewhere, as a seed from a model's prediction is labelled.

The maths is ``cellmap_flow.portable.ai_annotate``'s, which the browser runs
too (its resampling is Pillow's, in both modes); what is the server's alone
is reading the raw, through worldarray (``cellmap_flow.io.raw``).
"""

import numpy as np

from cellmap_flow.ai_annotate.geometry import PlanePlan, raw_box
from cellmap_flow.io.raw import RawInput, open_raw, spatial_shape
from cellmap_flow.portable.ai_annotate import (  # noqa: F401  (re-exported for the routes)
    accept_labels,
    labels_for_box,
    mask_to_write_shape,
    plane_image,
)


def _raw(volume: dict, voxel_size) -> RawInput:
    """The volume's raw dataset at ``voxel_size``, as its trainer reads it:
    the level for that size, resampled when the volume was made with Resample
    on, the channel the chain selects, as stored (no normalization), and 0
    outside the raw."""
    return RawInput(
        open_raw(volume["dataset_path"], voxel_size, resample=bool(volume.get("resample")), padding=0),
        normalize=False,
    )


def raw_extent_nm(volume: dict):
    """``(lo, hi)``, world nm z, y, x: where the volume's raw data is, read at
    its input voxel size (``plan_plane``'s ``data_box_nm``)."""
    array = _raw(volume, tuple(volume["input_voxel_size"])).array
    lo = np.asarray(array.corner, dtype=float)[-3:]
    extent = np.asarray(spatial_shape(array), dtype=float)[-3:] * np.asarray(array.voxel_size, dtype=float)[-3:]
    return lo, lo + extent


def read_plane(volume: dict, plan: PlanePlan) -> np.ndarray:
    """The raw plane ``plan`` reads, as a 2D array of ``plan.image_shape``,
    laid out as on screen (``portable.ai_annotate.plane_image``).

    Read as stored (no input normalization: the image is scaled to 8 bits
    for the model on its own) at the plan's input voxel size, from the level
    of the volume's raw dataset for it, resampled when the volume was made
    with Resample on, as its trainer reads it. Outside the raw the read is
    padded with 0. The raw's dtype is kept unless the plane had to be
    resized to the image's grid, when it is float32.
    """
    raw = _raw(volume, plan.input_voxel_size)
    begin, end = raw_box(plan, raw.array.voxel_size, raw.array.corner)
    return plane_image(np.asarray(raw.read_index(begin, end)), plan)


def build_request(plane: np.ndarray, plan: PlanePlan, profile, prompt_override=None):
    """The ``SegmentRequest`` for ``plane``: its RGB image, the prompt and the colour.

    The prompt states ``plan.resolution_nm``, the pixel size of the image
    actually sent. ``prompt_override`` replaces the catalog's description
    (``prompts.build_recolor_prompt``).
    """
    from cellmap_flow.ai_annotate.backends.base import SegmentRequest
    from cellmap_flow.ai_annotate.mask_extraction import slice_to_rgb
    from cellmap_flow.ai_annotate.prompts import build_recolor_prompt

    image = slice_to_rgb(np.asarray(plane))
    if image.size != (plan.image_shape[1], plan.image_shape[0]):
        raise ValueError(f"The plane is {image.size[::-1]}, the plan's image {tuple(plan.image_shape)}")
    return SegmentRequest(
        image=image,
        prompt=build_recolor_prompt(profile, resolution_nm=plan.resolution_nm, prompt_override=prompt_override),
        target_rgb=tuple(int(c) for c in profile.rgb),
        click_px=plan.click_px,
    )
