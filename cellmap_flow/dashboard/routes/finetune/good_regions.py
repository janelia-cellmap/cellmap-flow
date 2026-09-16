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
"""

import json
import logging
import os
import uuid

import neuroglancer
import numpy as np
from flask import jsonify

from cellmap_flow.dashboard.routes.finetune.common import viewer_position_and_scales
from cellmap_flow.globals import g

logger = logging.getLogger(__name__)

GOOD_REGIONS_LAYER = "good_regions"
GOOD_REGIONS_FILENAME = "good_regions.json"

# Used when neither the request nor the active volume says how big a region
# should be. One model *output* patch is the natural unit: it is what you can
# actually see and judge on screen, and it is the region a training patch
# computes loss over. Sizing to the input field of view instead would claim
# the model is right across 2848nm when you only looked at the middle 896nm
# of it -- and would not match annotation crops, which are already write_shape.
DEFAULT_REGION_SIZE_NM = 896.0


def _active_volume():
    """The annotation volume currently being worked on, or None."""
    volumes = getattr(g, "annotation_volumes", {}) or {}
    for volume in reversed(list(volumes.values())):
        if volume.get("corrections_dir"):
            return volume
    return None


def _store_path():
    """Where this session's good regions live, or None if there is no session."""
    volume = _active_volume()
    if not volume:
        return None
    return os.path.join(
        os.path.dirname(volume["corrections_dir"].rstrip("/")), GOOD_REGIONS_FILENAME
    )


def load_good_regions():
    path = _store_path()
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, ValueError) as e:
        logger.warning(f"Could not read good regions from {path}: {e}")
        return []


def save_good_regions(regions):
    path = _store_path()
    if not path:
        logger.warning("No annotation session yet; good regions were not saved.")
        return False
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(regions, f, indent=2)
        os.replace(tmp, path)
        return True
    except OSError as e:
        logger.error(f"Could not save good regions to {path}: {e}")
        return False


def _default_size_nm():
    """One model output patch, in nm -- see DEFAULT_REGION_SIZE_NM."""
    volume = _active_volume() or {}
    output_size = volume.get("output_size")
    output_voxel_size = volume.get("output_voxel_size")
    if output_size and output_voxel_size:
        try:
            return (np.array(output_size, dtype=float)
                    * np.array(output_voxel_size, dtype=float)).tolist()
        except (TypeError, ValueError):
            pass
    return [DEFAULT_REGION_SIZE_NM] * 3


def mark_current_view_response(data):
    """Record a box centred on wherever the viewer is looking right now."""
    data = data or {}
    try:
        position, scales_nm = viewer_position_and_scales()
        if position is None:
            return jsonify({"success": False, "error": "Viewer has no position"}), 400

        # viewer_position_and_scales returns the position in viewer voxels and
        # the scale of each axis in nm, so the product is absolute nm.
        centre_nm = np.array(position, dtype=float)
        if scales_nm:
            centre_nm = centre_nm * np.array(scales_nm, dtype=float)

        size_nm = np.array(data.get("size_nm") or _default_size_nm(), dtype=float)
        if size_nm.size == 1:
            size_nm = np.repeat(size_nm, 3)
        if np.any(size_nm <= 0):
            return jsonify({"success": False, "error": "size_nm must be positive"}), 400

        regions = load_good_regions()
        region = {
            "id": uuid.uuid4().hex[:8],
            "label": data.get("label") or f"good-{len(regions) + 1}",
            "offset_nm": (centre_nm - size_nm / 2).tolist(),
            "shape_nm": size_nm.tolist(),
        }
        regions.append(region)
        saved = save_good_regions(regions)
        refresh_good_regions_layer(regions)

        logger.info(
            f"Marked good region {region['label']} at "
            f"{[round(v) for v in centre_nm]} nm, size {[round(v) for v in size_nm]} nm"
        )
        return jsonify({
            "success": True,
            "region": region,
            "count": len(regions),
            "persisted": saved,
        })
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error marking good region: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


def list_good_regions_response():
    regions = load_good_regions()
    return jsonify({"success": True, "regions": regions, "count": len(regions)})


def delete_good_region_response(data):
    region_id = (data or {}).get("id")
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
    if not hasattr(g, "viewer") or g.viewer is None:
        return 0
    regions = load_good_regions() if regions is None else regions

    axes_names = ["z", "y", "x"]
    try:
        if getattr(g, "raw", None) is not None:
            source = getattr(g.raw, "source", None)
            if source is not None and hasattr(source, "dimensions"):
                axes_names = list(source.dimensions.names)
    except Exception:
        pass

    annotations = []
    for index, region in enumerate(regions):
        try:
            lo = np.array(region["offset_nm"], dtype=float)
            hi = lo + np.array(region["shape_nm"], dtype=float)
        except (KeyError, TypeError, ValueError):
            continue
        annotations.append(
            neuroglancer.AxisAlignedBoundingBoxAnnotation(
                point_a=lo.tolist(),
                point_b=hi.tolist(),
                id=str(index),
                description=region.get("label", "good"),
            )
        )

    try:
        with g.viewer.txn() as s:
            # Keep whatever visibility the user chose, but start visible when
            # creating the layer: unlike annotated_regions, which appears on
            # its own and was asked to stay out of the way, these boxes only
            # exist because the user just clicked to make one, and seeing it
            # appear is the confirmation that the click landed.
            was_visible = None
            if GOOD_REGIONS_LAYER in s.layers:
                try:
                    was_visible = bool(s.layers[GOOD_REGIONS_LAYER].visible)
                except Exception:
                    was_visible = None

            s.layers[GOOD_REGIONS_LAYER] = neuroglancer.LocalAnnotationLayer(
                dimensions=neuroglancer.CoordinateSpace(
                    names=axes_names, units="nm", scales=[1, 1, 1]
                ),
                annotations=annotations,
            )
            try:
                s.layers[GOOD_REGIONS_LAYER].visible = (
                    True if was_visible is None else was_visible
                )
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Could not update {GOOD_REGIONS_LAYER} layer: {e}")
        return 0
    return len(annotations)
