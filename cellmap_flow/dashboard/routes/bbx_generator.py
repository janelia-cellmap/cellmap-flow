import logging

import neuroglancer
from flask import Blueprint, request, jsonify

from cellmap_flow.utils.scale_pyramid import get_raw_layer
from cellmap_flow.globals import g

bbx_generator_state = g.bbx_generator_state

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
    """Start the Neuroglancer viewer for creating bounding boxes"""
    try:
        # Set Neuroglancer server to bind to 0.0.0.0 for external access
        neuroglancer.set_server_bind_address("0.0.0.0")

        data = request.json
        dataset_path = data.get("dataset_path", "")
        num_boxes = data.get("num_boxes", 1)
        existing_bounding_boxes = data.get("existing_bounding_boxes", [])

        if not dataset_path:
            return jsonify({"error": "Dataset path is required"}), 400

        # Create Neuroglancer viewer
        viewer = neuroglancer.Viewer()

        with viewer.txn() as s:
            # Set coordinate space
            s.dimensions = neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                units="nm",
                scales=[8, 8, 8],
            )

            # Add image layer
            s.layers["fibsem"] = get_raw_layer(dataset_path)

            # Add annotation layer for bounding boxes
            s.layers[BBOX_LAYER_NAME] = neuroglancer.LocalAnnotationLayer(
                dimensions=neuroglancer.CoordinateSpace(
                    names=["z", "y", "x"],
                    units="nm",
                    scales=[1, 1, 1],
                ),
            )

            # Add existing bounding boxes to the annotations layer
            if existing_bounding_boxes and len(existing_bounding_boxes) > 0:
                logger.info(f"Loading {len(existing_bounding_boxes)} existing bounding box(es)")
                from neuroglancer import AxisAlignedBoundingBoxAnnotation

                for idx, bbox in enumerate(existing_bounding_boxes):
                    offset = bbox.get("offset", [0, 0, 0])
                    shape = bbox.get("shape", [1, 1, 1])

                    # Calculate min and max points from offset and shape - MUST be floats
                    point_a = [float(offset[0]), float(offset[1]), float(offset[2])]
                    point_b = [
                        float(offset[0] + shape[0]),
                        float(offset[1] + shape[1]),
                        float(offset[2] + shape[2])
                    ]

                    # Create bounding box annotation with id and description
                    ann = AxisAlignedBoundingBoxAnnotation(
                        point_a=point_a,
                        point_b=point_b,
                        id=f"bbox-{idx + 1}",
                        description=f"Bounding box {idx + 1}"
                    )
                    s.layers[BBOX_LAYER_NAME].annotations.append(ann)
                    logger.info(f"Added existing bbox {idx + 1}: offset={offset}, shape={shape}")

        # Store state
        bbx_generator_state["dataset_path"] = dataset_path
        bbx_generator_state["num_boxes"] = num_boxes
        bbx_generator_state["bounding_boxes"] = list(existing_bounding_boxes)
        bbx_generator_state["viewer"] = viewer

        # Get the viewer URL and fix localhost reference
        viewer_url = str(viewer)

        # Replace localhost with the actual request host for external access
        if "localhost" in viewer_url:
            client_host = request.host.split(":")[0]
            viewer_url = viewer_url.replace("localhost", client_host)
            logger.info(f"Replaced localhost with {client_host} in viewer URL")

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
