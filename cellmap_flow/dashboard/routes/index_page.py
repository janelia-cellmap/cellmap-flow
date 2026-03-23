import logging

from flask import Blueprint, render_template, request, jsonify
import neuroglancer

from cellmap_flow.norm.input_normalize import get_input_normalizers
from cellmap_flow.post.postprocessors import get_postprocessors_list
from cellmap_flow.models.model_merger import get_model_mergers_list
from cellmap_flow.globals import g
from cellmap_flow.utils.scale_pyramid import get_raw_layer
import cellmap_flow.dashboard.state as state

logger = logging.getLogger(__name__)

index_bp = Blueprint("index", __name__)


@index_bp.route("/")
def index():
    # Render the main page with tabs
    input_norms = get_input_normalizers()
    output_postprocessors = get_postprocessors_list()
    model_mergers = get_model_mergers_list()
    model_catalog = g.model_catalog
    model_catalog["User"] = {j.model_name: "" for j in g.jobs}
    default_post_process = {d.to_dict()["name"]: d.to_dict() for d in g.postprocess}
    default_input_norm = {d.to_dict()["name"]: d.to_dict() for d in g.input_norms}
    logger.warning(f"Model catalog: {model_catalog}")
    logger.warning(f"Default postprocess: {default_post_process}")
    logger.warning(f"Default input norm: {default_input_norm}")

    # Collect running HF model repos
    from cellmap_flow.models.models_config import HuggingFaceModelConfig
    running_job_names = {j.model_name for j in g.jobs}
    default_hf_repos = [
        mc.repo for mc in g.models_config
        if isinstance(mc, HuggingFaceModelConfig) and mc.name in running_job_names
    ]

    return render_template(
        "index.html",
        neuroglancer_url=g.NEUROGLANCER_URL,
        inference_servers=g.INFERENCE_SERVER,
        input_normalizers=input_norms,
        output_postprocessors=output_postprocessors,
        model_mergers=model_mergers,
        default_post_process=default_post_process,
        default_input_norm=default_input_norm,
        model_catalog=model_catalog,
        default_models=[j.model_name for j in g.jobs],
        default_hf_repos=default_hf_repos,
        server_config_cached=g._server_config_cached,
    )


@index_bp.route("/api/set-data", methods=["POST"])
def set_data():
    """Set up neuroglancer viewer with a dataset path."""
    try:
        data = request.get_json()
        dataset_path = data.get("dataset_path", "").strip()
        if not dataset_path:
            return jsonify({"error": "dataset_path is required"}), 400

        # Set up neuroglancer
        neuroglancer.set_server_bind_address("0.0.0.0")
        viewer = neuroglancer.Viewer()

        g.dataset_path = dataset_path
        g.viewer = viewer

        with viewer.txn() as s:
            s.dimensions = neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                units="nm",
                scales=[8, 8, 8],
            )
            s.layers["data"] = get_raw_layer(dataset_path)

        state.NEUROGLANCER_URL = str(viewer)
        logger.warning(f"Neuroglancer viewer set up: {state.NEUROGLANCER_URL}")

        return jsonify({
            "success": True,
            "neuroglancer_url": state.NEUROGLANCER_URL,
        })
    except Exception as e:
        logger.error(f"Error setting data: {str(e)}")
        return jsonify({"error": str(e)}), 500
