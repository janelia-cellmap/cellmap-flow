import logging

import neuroglancer
from flask import Blueprint, request, jsonify

from cellmap_flow.dashboard.requests import BbxGenerator, parse
from cellmap_flow.dashboard.routes.index_page import viewer_url_for
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.viewer.bootstrap import new_viewer

logger = logging.getLogger(__name__)

bbx_bp = Blueprint("bbx_generator", __name__)

# The layer the boxes are drawn in. Read through this name, never a literal:
# neuroglancer's Layers.__getitem__ resolves a name via index(), which returns
# -1 when the name is absent, so `s.layers["missing"]` quietly hands back
# _layers[-1] -- the last layer -- instead of raising KeyError. Reading
# "annotations" here happened to return the box layer only because "bboxes"
# was created last; any layer added after it would have silently taken its
# place. Membership is the one safe check: `in` goes through index() != -1.
BBOX_LAYER_NAME = "bboxes"


def _extract_bounding_boxes(viewer):
    """Axis-aligned boxes currently drawn in the viewer, as offset/shape dicts."""
    boxes = []
    if viewer is None:
        return boxes
    try:
        with viewer.txn() as s:
            if BBOX_LAYER_NAME not in s.layers:
                logger.warning(
                    f"No {BBOX_LAYER_NAME!r} layer in the viewer; no bounding "
                    f"boxes to read. Layers present: {[l.name for l in s.layers]}"
                )
                return boxes
            layer = s.layers[BBOX_LAYER_NAME]
            for ann in getattr(layer, "annotations", []):
                if type(ann).__name__ != "AxisAlignedBoundingBoxAnnotation":
                    continue
                point_a, point_b = ann.point_a, ann.point_b
                offset = [min(point_a[j], point_b[j]) for j in range(3)]
                max_point = [max(point_a[j], point_b[j]) for j in range(3)]
                boxes.append({
                    "offset": [int(x) for x in offset],
                    "shape": [int(max_point[j] - offset[j]) for j in range(3)],
                })
    except Exception as e:
        logger.warning(f"Error extracting bounding boxes from viewer: {e}")
    return boxes


@bbx_bp.route("/api/bbx-generator", methods=["POST"])
def start_bbx_generator():
    """Start the box tool's Neuroglancer viewer (a BbxGenerator)."""
    body, error = parse(BbxGenerator, request.get_json(silent=True))
    if error:
        return error
    try:
        dataset_path = body.dataset_path
        num_boxes = body.num_boxes
        existing_bounding_boxes = [box.model_dump() for box in body.existing_bounding_boxes or []]

        # The boxes drawn so far, each with an id and a description. Their
        # corners must be floats.
        boxes = []
        for idx, bbox in enumerate(existing_bounding_boxes):
            offset = bbox["offset"]
            shape = bbox["shape"]
            boxes.append(neuroglancer.AxisAlignedBoundingBoxAnnotation(
                point_a=[float(offset[j]) for j in range(3)],
                point_b=[float(offset[j] + shape[j]) for j in range(3)],
                id=f"bbox-{idx + 1}",
                description=f"Bounding box {idx + 1}",
            ))
            logger.info(f"Added existing bbox {idx + 1}: offset={offset}, shape={shape}")
        box_layer = neuroglancer.LocalAnnotationLayer(
            dimensions=neuroglancer.CoordinateSpace(names=["z", "y", "x"], units="nm", scales=[1, 1, 1]),
            annotations=boxes,
        )
        # 8 nm z, y, x, as this viewer always had.
        viewer = new_viewer(dataset_path, scales=(8, 8, 8), raw_name="fibsem", layers={BBOX_LAYER_NAME: box_layer})

        # Store state
        bbx_generator_state = get_session().bbx_generator_state
        bbx_generator_state["dataset_path"] = dataset_path
        bbx_generator_state["num_boxes"] = num_boxes
        bbx_generator_state["bounding_boxes"] = list(existing_bounding_boxes)
        bbx_generator_state["viewer"] = viewer

        # The address the browser loads the viewer from: the request's host
        # for a viewer on localhost, and behind a reverse proxy the proxy's,
        # as for the dashboard's own viewer.
        viewer_url = str(viewer)
        if "localhost" in viewer_url:
            client_host = request.host.split(":")[0]
            viewer_url = viewer_url.replace("localhost", client_host)
            logger.info(f"Replaced localhost with {client_host} in viewer URL")
        viewer_url = viewer_url_for(viewer_url, request.headers, request.scheme)

        bbx_generator_state["viewer_url"] = viewer_url
        bbx_generator_state["viewer_state"] = viewer.state

        logger.info(f"Starting BBX generator with viewer URL: {viewer_url}")
        logger.info(f"Dataset path: {dataset_path}")
        logger.info(f"Target boxes: {num_boxes}")
        logger.info(f"Existing boxes: {len(existing_bounding_boxes)}")

        return jsonify({
            "success": True,
            "viewer_url": viewer_url,
            "dataset_path": dataset_path,
            "num_boxes": num_boxes,
            "existing_count": len(existing_bounding_boxes),
            "existing_bounding_boxes": existing_bounding_boxes
        })

    except Exception as e:
        logger.error(f"Error starting BBX generator: {str(e)}")
        return jsonify({"error": str(e)}), 500


@bbx_bp.route("/api/bbx-generator/status", methods=["GET"])
def get_bbx_generator_status():
    """Get current status of bounding box generation"""
    try:
        bbx_generator_state = get_session().bbx_generator_state
        # Extract bounding boxes from viewer if it exists
        bboxes = _extract_bounding_boxes(bbx_generator_state.get("viewer"))

        bbx_generator_state["bounding_boxes"] = bboxes

        return jsonify({
            "dataset_path": bbx_generator_state.get("dataset_path"),
            "num_boxes": bbx_generator_state.get("num_boxes"),
            "bounding_boxes": bboxes,
            "count": len(bboxes)
        })

    except Exception as e:
        logger.error(f"Error getting BBX status: {str(e)}")
        return jsonify({"error": str(e)}), 500


@bbx_bp.route("/api/bbx-generator/finalize", methods=["POST"])
def finalize_bbx_generation():
    """Finalize bounding box generation and return results"""
    try:
        bbx_generator_state = get_session().bbx_generator_state
        # Extract final bounding boxes from viewer
        bboxes = _extract_bounding_boxes(bbx_generator_state.get("viewer"))

        # Reset state
        bbx_generator_state["dataset_path"] = None
        bbx_generator_state["num_boxes"] = 0
        bbx_generator_state["bounding_boxes"] = []
        bbx_generator_state["viewer_url"] = None
        bbx_generator_state["viewer"] = None

        return jsonify({
            "success": True,
            "bounding_boxes": bboxes,
            "count": len(bboxes)
        })

    except Exception as e:
        logger.error(f"Error finalizing BBX generation: {str(e)}")
        return jsonify({"error": str(e)}), 500
