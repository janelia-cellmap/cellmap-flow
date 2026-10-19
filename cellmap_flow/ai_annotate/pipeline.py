"""From a planned plane to labels: read the raw, ask the model, label the box.

``read_plane`` reads the plane ``geometry.plan_plane`` planned, on the image
grid it planned and laid out as on screen; ``build_request`` makes it the
request a backend sends; ``mask_to_write_shape`` cuts the mask the backend
returns into its connected objects and puts them onto the write box's
voxels, and ``labels_for_box`` turns those into the labels ``fill.paint_box``
writes: one id per object, background elsewhere, as a seed from a model's
prediction is labelled.
"""

import numpy as np
from PIL import Image

from cellmap_flow.ai_annotate.geometry import PlanePlan, raw_box, screen_shape


def _resize(plane: np.ndarray, shape) -> np.ndarray:
    """``plane`` resampled to ``shape`` (rows, cols); itself when it already is that.

    Shrinking averages each pixel's footprint (PIL's box filter), so fine
    membranes fade rather than alias; growing (the coarse axis of an
    anisotropic plane) interpolates linearly.
    """
    rows, cols = (int(s) for s in shape)
    if plane.shape == (rows, cols):
        return plane
    shrinking = plane.shape[0] >= rows and plane.shape[1] >= cols
    image = Image.fromarray(np.ascontiguousarray(plane, dtype=np.float32))
    resized = image.resize((cols, rows), Image.BOX if shrinking else Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def _raw(volume: dict, voxel_size):
    """The volume's raw dataset at ``voxel_size``, as its trainer reads it."""
    from cellmap_flow.image_data_interface import ImageDataInterface

    return ImageDataInterface(
        volume["dataset_path"],
        voxel_size=voxel_size,
        normalize=False,
        on_voxel_size_mismatch="resample" if volume.get("resample") else "relabel",
    )


def raw_extent_nm(volume: dict):
    """``(lo, hi)``, world nm z, y, x: where the volume's raw data is, read at
    its input voxel size (``plan_plane``'s ``data_box_nm``)."""
    idi = _raw(volume, tuple(volume["input_voxel_size"]))
    lo = np.asarray(idi.offset, dtype=float)[-3:]
    return lo, lo + np.asarray(idi.shape, dtype=float)[-3:] * np.asarray(idi.voxel_size, dtype=float)[-3:]


def read_plane(volume: dict, plan: PlanePlan) -> np.ndarray:
    """The raw plane ``plan`` reads, as a 2D array of ``plan.image_shape``,
    laid out as on screen (``PlanePlan.to_screen``).

    Read as stored (no input normalization: the image is scaled to 8 bits
    for the model on its own) at the plan's input voxel size, from the level
    of the volume's raw dataset for it, resampled when the volume was made
    with Resample on, as its trainer reads it. Outside the raw the read is
    padded with 0. The raw's dtype is kept unless the plane had to be
    resized to the image's grid, when it is float32.
    """
    idi = _raw(volume, plan.input_voxel_size)
    box = raw_box(plan, idi.voxel_size, idi.offset)
    data = np.asarray(idi.read_box(box))
    plane = np.take(data, 0, axis=plan.depth_axis)
    return _resize(plan.to_screen(plane), plan.image_shape)


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


def mask_to_write_shape(mask: np.ndarray, plan: PlanePlan) -> np.ndarray:
    """The objects of ``mask`` (the image's shape, as on screen) on the write
    box's in-plane voxels, in z, y, x order (``PlanePlan.write_shape``):
    int32, 0 for background and 1..n for the objects, numbered in the box's
    raster order as ``scipy.ndimage.label`` numbers a mask.

    Objects are the mask's 4-connected pieces (scipy's default, so pieces
    touching only at a corner stay apart, as the model was asked to leave
    them), found at the image's own resolution, where the model drew the
    lines between them: on voxels coarser than the pixels a line can fall
    between the pixels the voxels take, and the objects either side of it
    still keep ids of their own. Then nearest neighbour, pixel centre to
    voxel centre: each annotation voxel takes the object of the pixel at
    its centre, so the mask keeps hard edges and lands where it was drawn
    whatever the two sizes are. An object no voxel centre falls in is lost.
    """
    from scipy.ndimage import label

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"The mask must be 2D, got shape {mask.shape}")
    objects, _ = label(mask)
    rows, cols = screen_shape(plan.write_shape, plan.depth_axis, plan.display_dimensions)
    if objects.shape != (rows, cols):
        objects = np.asarray(Image.fromarray(objects.astype(np.int32)).resize((cols, rows), Image.NEAREST))
    objects = plan.from_screen(objects)
    # Renumbered by first voxel, so fresh ids go to the objects in the order
    # labelling the box itself would give them, whatever the screen layout.
    ids, first = np.unique(objects, return_index=True)
    order = ids[ids > 0][np.argsort(first[ids > 0])]
    number = np.zeros(int(objects.max(initial=0)) + 1, dtype=np.int32)
    number[order] = np.arange(1, order.size + 1)
    return number[objects]


def labels_for_box(objects: np.ndarray, existing: np.ndarray, depth_axis: int, count_up: bool,
                   beside=None) -> np.ndarray:
    """The labels for the write box: an id per object of ``objects``, 1 elsewhere.

    ``objects`` is the in-plane objects (``mask_to_write_shape``), 0 for
    background, ``existing`` the one-voxel-thick box as the volume holds
    it. The objects are labelled by ``fill.seed_labels``: an object the user
    has already painted part of keeps that id, the rest get fresh ones,
    counting up for an instance target's ``count_up``.

    ``beside`` is the planes on either side of the box along the depth axis,
    as the volume holds them (each shaped like ``existing``, or None at the
    volume's edge). Planes are annotated one at a time, so an object usually
    continues one in the plane next to it: where this plane is unpainted,
    the ids beside it count as painted here, and an object takes the id it
    overlaps most, rather than a fresh one that an instance target would
    read as a boundary between the two planes. Fresh ids skip those planes'
    ids too, so a new object does not take the id of a different object
    right next to it, which an instance target would read as one object.
    """
    from cellmap_flow.finetune.session.fill import seed_labels

    objects = np.expand_dims(np.asarray(objects), depth_axis)
    if objects.shape != existing.shape:
        raise ValueError(f"The mask makes a box of {objects.shape}; the box holds {existing.shape}")
    planes = [np.asarray(p, dtype=existing.dtype) for p in (beside or ()) if p is not None]
    if any(p.shape != existing.shape for p in planes):
        raise ValueError(f"The planes beside the box must be shaped {existing.shape}")
    if not planes:
        return seed_labels(objects, existing, count_up=count_up)
    seen = existing.copy()
    for plane in planes:
        carry = (seen == 0) & (plane >= 2)
        seen[carry] = plane[carry]
    # The planes beside are stacked after the box with no objects in them:
    # seed_labels labels only the box's objects, but its fresh ids see every
    # id in the stack.
    stack_existing = np.concatenate([seen, *planes], axis=depth_axis)
    stack_objects = np.concatenate([objects, *[np.zeros_like(objects)] * len(planes)], axis=depth_axis)
    labels = seed_labels(stack_objects, stack_existing, count_up=count_up)
    return np.take(labels, [0], axis=depth_axis)
