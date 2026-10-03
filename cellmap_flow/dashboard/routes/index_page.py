import logging
from urllib.parse import urlparse

from flask import Blueprint, render_template, request, jsonify

from cellmap_flow.norm.input_normalize import get_input_normalizers
from cellmap_flow.post.postprocessors import get_postprocessors_list
from cellmap_flow.dashboard.requests import SetData, parse
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.viewer.bootstrap import new_viewer

logger = logging.getLogger(__name__)

index_bp = Blueprint("index", __name__)


def viewer_url_for(viewer_url, headers, scheme):
    """The address the browser should load the neuroglancer viewer from.

    Behind a reverse proxy (the request carries X-Forwarded-Host), the browser
    can reach only the proxy, and an https page cannot embed the viewer's own
    http://<node>:<port> address; so it asks for the viewer at the same path
    on the proxy's host, which the proxy must route to the viewer. The scheme
    is the proxy's X-Forwarded-Proto, else this request's own. Without the
    header (direct access) the viewer's own address is returned unchanged.
    """
    forwarded_host = (headers.get("X-Forwarded-Host") or "").split(",")[0].strip()
    if not viewer_url or not forwarded_host or forwarded_host.startswith(("localhost", "127.")):
        return viewer_url
    parsed = urlparse(viewer_url)
    if not parsed.netloc or parsed.netloc == forwarded_host:
        return viewer_url
    proto = (headers.get("X-Forwarded-Proto") or scheme).split(",")[0].strip()
    return parsed._replace(scheme=proto, netloc=forwarded_host).geturl()


def _form_value(value):
    """A parameter as the Input/Postprocess form shows and sends it back.

    A flat list or tuple is comma-joined, the form the constructors parse
    ("0,2" for ChannelSelection): rendered as it stands it became "[0, 2]",
    which Submit All sent back and the constructor could not read.
    """
    if isinstance(value, (list, tuple)) and not any(
        isinstance(v, (list, tuple, dict)) for v in value
    ):
        return ",".join(str(v) for v in value)
    return value


def chain_items(available, configured):
    """The rows of an Input/Postprocess list, in the order to render them.

    The configured steps come first, in the order they run, each with its own
    parameter values; a step configured twice is listed twice. Every other
    available step follows, unticked, with its defaults. Submit All sends the
    ticked rows in display order, so rendering the configured chain in
    registry order instead would reorder it on the next submit.

    ``available`` is get_input_normalizers() / get_postprocessors_list();
    ``configured`` is the live chain (the session's input_norms / postprocess).
    """
    defaults = {op["name"]: op.get("params", {}) for op in available}
    items = []
    configured_names = set()
    for step in configured or []:
        step_dict = step.to_dict()
        name = step_dict.get("name")
        configured_names.add(name)
        if name in defaults:
            # Only the constructor's parameters are editable, and to_dict can
            # lack one the op stores under another name; fall back to its
            # default rather than dropping the field.
            params = {
                key: step_dict.get(key, default)
                for key, default in defaults[name].items()
            }
        else:
            params = {k: v for k, v in step_dict.items() if k != "name"}
        params = {k: _form_value(v) for k, v in params.items()}
        items.append({"name": name, "checked": True, "params": params})
    for op in available:
        if op["name"] not in configured_names:
            params = {k: _form_value(v) for k, v in op.get("params", {}).items()}
            items.append({"name": op["name"], "checked": False, "params": params})
    return items


def cellpose_panel_data(models_config, running_job_names):
    """What the Models tab's Cellpose panel is built from (models-tab.js):
    each model it lists, and the settings of those running, which start
    ticked.

    ``{"models": [{"model", "label", "description"}], "running": [{"model",
    "output", "voxel_size", "stitch_threshold"}]}``; a model running with
    two outputs is two entries, so two rows.
    """
    from cellmap_flow.dashboard.services.launch import CELLPOSE_MODELS
    from cellmap_flow.models.models_config import CellposeModelConfig

    running = []
    for mc in models_config:
        if isinstance(mc, CellposeModelConfig) and mc.name in running_job_names:
            running.append({"model": mc.pretrained_model, "output": mc.output,
                            "voxel_size": list(mc.voxel_size), "stitch_threshold": mc.stitch_threshold})
    return {
        "models": [{"model": model, "label": label, "description": description}
                   for model, (label, description) in CELLPOSE_MODELS.items()],
        "running": running,
    }


@index_bp.route("/")
def index():
    # Render the main page with tabs
    session = get_session()
    input_norm_items = chain_items(get_input_normalizers(), session.input_norms)
    postprocess_items = chain_items(get_postprocessors_list(), session.postprocess)
    # A copy: the "User" group lists this session's running models for the
    # Models tab only. Written into the session's catalog it outlived the request,
    # and everything else that walks the catalog (update_run_models, the
    # pipeline builder's palette) found entries with no path.
    model_catalog = dict(session.model_catalog)
    model_catalog["User"] = {j.model_name: "" for j in session.jobs}
    logger.debug(f"Model catalog: {model_catalog}")
    logger.debug(f"Input norm rows: {input_norm_items}")
    logger.debug(f"Postprocess rows: {postprocess_items}")

    # Collect running HF model repos
    from cellmap_flow.models.models_config import HuggingFaceModelConfig
    running_job_names = {j.model_name for j in session.jobs}
    default_hf_repos = [
        mc.repo for mc in session.models_config
        if isinstance(mc, HuggingFaceModelConfig) and mc.name in running_job_names
    ]
    # And the running zoo models, with the voxel size each was given.
    from cellmap_flow.models.bioimage_catalog import entry_model_id
    from cellmap_flow.models.models_config import BioModelConfig
    default_bioimage_models = [
        {"id": entry_model_id(mc), "voxel_size": mc.to_dict().get("voxel_size")}
        for mc in session.models_config
        if isinstance(mc, BioModelConfig) and mc.name in running_job_names
    ]
    cellpose_panel = cellpose_panel_data(session.models_config, running_job_names)

    return render_template(
        "index.html",
        neuroglancer_url=viewer_url_for(session.neuroglancer_url, request.headers, request.scheme),
        input_norm_items=input_norm_items,
        postprocess_items=postprocess_items,
        model_catalog=model_catalog,
        default_models=[j.model_name for j in session.jobs],
        default_hf_repos=default_hf_repos,
        default_bioimage_models=default_bioimage_models,
        cellpose_panel=cellpose_panel,
        resample=session.resample,
        server_config_cached=session.server_config_cached,
    )


@index_bp.route("/api/set-data", methods=["POST"])
def set_data():
    """Set up neuroglancer viewer with a dataset path."""
    body, error = parse(SetData, request.get_json(silent=True))
    if error:
        return error
    dataset_path = body.dataset_path
    try:
        session = get_session()
        session.dataset_path = dataset_path
        # 8 nm z, y, x, as this viewer always had; unlike the CLIs' viewer it
        # does not take its dimensions from the raw.
        session.viewer = new_viewer(dataset_path, scales=(8, 8, 8))
        session.neuroglancer_url = str(session.viewer)
        logger.debug(f"Neuroglancer viewer set up: {session.neuroglancer_url}")

        return jsonify({
            "success": True,
            "neuroglancer_url": viewer_url_for(
                session.neuroglancer_url, request.headers, request.scheme
            ),
        })
    except Exception as e:
        logger.error(f"Error setting data: {str(e)}")
        return jsonify({"error": str(e)}), 500
