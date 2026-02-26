import logging

from flask import Blueprint, render_template

from cellmap_flow.norm.input_normalize import get_input_normalizers
from cellmap_flow.post.postprocessors import get_postprocessors_list
from cellmap_flow.models.model_merger import get_model_mergers_list
from cellmap_flow.globals import g

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
    )
