"""Regions the user has looked at and judged the model already handles well.

These are the counterweight to corrections. Painted scribbles say "you got
this wrong, here"; a good region says "you got this right, here, and I
checked". Training samples patches from both: the scribbles supply the
supervised signal, and the good regions supply rehearsal -- patches with no
annotation at all, which the trainer already treats correctly (an empty mask
zeroes the supervised term and hands the whole patch to the distillation
term against the frozen teacher).

Kept deliberately separate from ``imported_crops``: those carry real labels,
these carry only a human's assertion that the model's own output is
acceptable there. Both are evidence, but not the same kind.

Routes: GET ``/api/finetune/good-regions`` (the session's regions), POST
``/api/finetune/good-regions/mark-view`` (a box where the viewer looks) and
POST ``/api/finetune/good-regions/delete`` (one region by ``id``, or all).
"""

import logging

import neuroglancer
import numpy as np
from flask import jsonify, request

from cellmap_flow.dashboard.routes.finetune.blueprint import finetune_bp
from cellmap_flow.dashboard.routes.finetune.common import session_store, viewer_position_and_scales
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.finetune.session import manifest as session_manifest
from cellmap_flow.portable import regions as portable_regions
from cellmap_flow.viewer.layers import replace_layer

logger = logging.getLogger(__name__)

GOOD_REGIONS_LAYER = "good_regions"


def _active_volume():
    """The annotation volume currently being worked on, or None."""
    return session_store().session_volume()[1]


def _minio_corrections_dir():
    """The corrections dir MinIO was first started for, or None.

    The fallback when no volume is registered. It is this process's state,
    like the volume registry, so it does not survive a dashboard restart
    either: after one, marks are refused until a volume is created or
    resumed.
    """
    return get_session().minio_state.get("output_base") or None


def _store_path():
    """Where this session's good regions live, or None if there is no session.

    The session's volume's corrections dir, else MinIO's. With no volume
    registered and no fallback, every mark used to be lost in a way that
    looked like success -- the save failed, so the next load returned [],
    so each click appended to an empty list and replaced the previous
    region instead of adding to it. You could click ten times and still
    have one box. Now the mark is refused instead.
    """
    volume = _active_volume() or {}
    return session_manifest.good_regions_path(
        volume.get("corrections_dir") or _minio_corrections_dir()
    )


def load_good_regions():
    return session_manifest.load_good_regions(_store_path())


def save_good_regions(regions):
    path = _store_path()
    if not path:
        logger.warning("No annotation session yet; good regions were not saved.")
        return False
    return session_manifest.save_good_regions(path, regions)


def view_box_nm(size_nm=None):
    """``(centre_nm, size_nm)`` of the box centred where the viewer looks.

    ``size_nm`` is as ``cellmap_flow.portable.regions.region_size_nm`` (the
    browser's rule too) takes it: one model output patch of the active
    volume when none is given (see DEFAULT_REGION_SIZE_NM there), one number
    the same along every axis. The labelling buttons beside "Mark This View
    as Good" use the same box, so what they label is what a mark would
    cover. ValueError when there is no viewer position or the size is not
    positive.
    """
    position, scales_nm = viewer_position_and_scales()
    if position is None:
        raise ValueError("Viewer has no position")

    # viewer_position_and_scales returns the position in viewer voxels and
    # the scale of each axis in nm, both z, y, x, so the product is absolute nm.
    centre_nm = np.array(position, dtype=float)
    if scales_nm:
        centre_nm = centre_nm * np.array(scales_nm, dtype=float)

    volume = _active_volume() or {}
    size_nm = portable_regions.region_size_nm(size_nm, volume.get("output_size"), volume.get("output_voxel_size"))
    return centre_nm, np.array(size_nm, dtype=float)


@finetune_bp.route("/api/finetune/good-regions/mark-view", methods=["POST"])
def mark_current_view_good():
    """Record a box centred on wherever the viewer is looking right now."""
    data = request.get_json(silent=True) or {}
    try:
        centre_nm, size_nm = view_box_nm(data.get("size_nm"))

        stored = load_good_regions()
        region = portable_regions.new_region(centre_nm, size_nm, len(stored), data.get("label"))
        stored.append(region)
        if not save_good_regions(stored):
            # Drawing the box anyway is what made this look like it worked.
            # A mark that is not on disk will not reach training, and the
            # next click will silently overwrite it, so say so instead.
            return jsonify({
                "success": False,
                "error": (
                    "Could not work out where to save good regions for this "
                    "session, so the mark was discarded. Create or resume an "
                    "annotation volume first."
                ),
                "needs_volume": True,
            }), 409

        refresh_good_regions_layer(stored)
        logger.info(
            f"Marked good region {region['label']} at "
            f"{[round(v) for v in centre_nm]} nm, size {[round(v) for v in size_nm]} nm "
            f"({len(stored)} total)"
        )
        return jsonify({
            "success": True,
            "region": region,
            "count": len(stored),
            "persisted": True,
        })
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error marking good region: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@finetune_bp.route("/api/finetune/good-regions", methods=["GET"])
def list_good_regions():
    regions = load_good_regions()
    return jsonify({"success": True, "regions": regions, "count": len(regions)})


@finetune_bp.route("/api/finetune/good-regions/delete", methods=["POST"])
def delete_good_region():
    region_id = (request.get_json(silent=True) or {}).get("id")
    regions = load_good_regions()
    if region_id is None:
        kept = []
    else:
        kept = [r for r in regions if r.get("id") != region_id]
        if len(kept) == len(regions):
            return jsonify({"success": False, "error": f"No region {region_id}"}), 404
    save_good_regions(kept)
    refresh_good_regions_layer(kept)
    return jsonify({"success": True, "count": len(kept)})


def refresh_good_regions_layer(regions=None):
    """Draw the good regions in the viewer so you can see what you marked."""
    session = get_session()
    if session.viewer is None:
        return 0
    entries = load_good_regions() if regions is None else regions

    axes_names = ["z", "y", "x"]
    try:
        if session.raw is not None:
            source = getattr(session.raw, "source", None)
            if source is not None and hasattr(source, "dimensions"):
                axes_names = list(source.dimensions.names)
    except Exception:
        pass

    annotations = []
    for index, region in enumerate(entries):
        # An entry that is not a region is skipped and keeps its index, as
        # the browser's layer skips it (portable_regions.region_box_nm).
        box = portable_regions.region_box_nm(region)
        if box is None:
            continue
        lo, shape = box
        label = region.get("label")
        annotations.append(
            neuroglancer.AxisAlignedBoundingBoxAnnotation(
                point_a=lo.tolist(),
                point_b=(lo + shape).tolist(),
                id=str(index),
                description=label if isinstance(label, str) else "good",
            )
        )

    try:
        with session.viewer.txn() as s:
            # Kept as the user left it (hidden, its colour), and visible when
            # the layer is new: unlike annotated_regions, which appears on its
            # own and was asked to stay out of the way, these boxes only exist
            # because the user just clicked to make one, and seeing it appear
            # is the confirmation that the click landed.
            replace_layer(s.layers, GOOD_REGIONS_LAYER, neuroglancer.LocalAnnotationLayer(
                dimensions=neuroglancer.CoordinateSpace(
                    names=axes_names, units="nm", scales=[1, 1, 1]
                ),
                annotations=annotations,
            ))
    except Exception as e:
        logger.warning(f"Could not update {GOOD_REGIONS_LAYER} layer: {e}")
        return 0
    return len(annotations)
