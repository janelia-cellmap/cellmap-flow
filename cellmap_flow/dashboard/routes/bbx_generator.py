import logging

import neuroglancer
from flask import Blueprint, request, jsonify

from cellmap_flow.utils.scale_pyramid import get_raw_layer
from cellmap_flow.dashboard.state import bbx_generator_state

logger = logging.getLogger(__name__)

bbx_bp = Blueprint("bbx_generator", __name__)


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
            s.layers["bboxes"] = neuroglancer.LocalAnnotationLayer(
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
                    s.layers["bboxes"].annotations.append(ann)
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
        bboxes = []
        if bbx_generator_state.get("viewer"):
            viewer = bbx_generator_state["viewer"]
            try:
                with viewer.txn() as s:
                    try:
                        annotations_layer = s.layers["annotations"]
                        if hasattr(annotations_layer, 'annotations'):
                            for ann in annotations_layer.annotations:
                                if type(ann).__name__ == "AxisAlignedBoundingBoxAnnotation":
                                    point_a = ann.point_a
                                    point_b = ann.point_b

                                    offset = [min(point_a[j], point_b[j]) for j in range(3)]
                                    max_point = [max(point_a[j], point_b[j]) for j in range(3)]
                                    shape = [int(max_point[j] - offset[j]) for j in range(3)]
                                    offset = [int(x) for x in offset]

                                    bboxes.append({
                                        "offset": offset,
                                        "shape": shape,
                                    })
                    except KeyError:
                        logger.warning("Annotations layer not found in viewer")
            except Exception as e:
                logger.warning(f"Error extracting bboxes from viewer: {str(e)}")

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
        bboxes = []
        if bbx_generator_state.get("viewer"):
            viewer = bbx_generator_state["viewer"]
            try:
                with viewer.txn() as s:
                    try:
                        annotations_layer = s.layers["annotations"]
                        if hasattr(annotations_layer, 'annotations'):
                            for ann in annotations_layer.annotations:
                                if type(ann).__name__ == "AxisAlignedBoundingBoxAnnotation":
                                    point_a = ann.point_a
                                    point_b = ann.point_b

                                    offset = [min(point_a[j], point_b[j]) for j in range(3)]
                                    max_point = [max(point_a[j], point_b[j]) for j in range(3)]
                                    shape = [int(max_point[j] - offset[j]) for j in range(3)]
                                    offset = [int(x) for x in offset]

                                    bboxes.append({
                                        "offset": offset,
                                        "shape": shape,
                                    })
                    except KeyError:
                        logger.warning("Annotations layer not found in viewer")
            except Exception as e:
                logger.warning(f"Error extracting final bboxes: {str(e)}")

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
