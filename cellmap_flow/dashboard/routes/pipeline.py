import os
import json
import logging
import re
import time
from datetime import datetime

import neuroglancer
import numpy as np
from flask import Blueprint, request, jsonify

from cellmap_flow.globals import g
from cellmap_flow.norm.input_normalize import (
    get_input_normalizers,
    get_normalizations,
)
from cellmap_flow.post.postprocessors import get_postprocessors_list, get_postprocessors
from cellmap_flow.utils.load_py import load_safe_config
from cellmap_flow.utils.output_probe import output_display_range
from cellmap_flow.utils.scale_pyramid import (
    PREDICTION_COLORS,
    get_raw_layer,
    prediction_shader,
)
from cellmap_flow.utils.server_info import fetch_model_info
from cellmap_flow.utils.web_utils import encode_to_str, ARGS_KEY

logger = logging.getLogger(__name__)

pipeline_bp = Blueprint("pipeline", __name__)


def _save_shaders_from_viewer() -> None:
    """Read current shader and shaderControls from the neuroglancer viewer and persist them in globals."""
    if g.viewer is None:
        return
    try:
        state = g.viewer.state
        for layer in state.layers:
            shader = getattr(layer, "shader", None)
            # A neuroglancer layer with no shader set reports the *string*
            # "None" (not Python None), which is truthy. Storing it would
            # later be restored onto the layer verbatim and fail to compile,
            # wiping the user's rendering. Treat it as "unset".
            if shader and shader != "None":
                g.shaders[layer.name] = shader
            shader_controls = getattr(layer, "shaderControls", None) or getattr(layer, "shader_controls", None)
            if shader_controls:
                g.shader_controls[layer.name] = shader_controls
    except Exception as exc:
        logger.warning(f"Could not save shaders from viewer: {exc}")


def _chain_signature(steps) -> str:
    """Stable key for a list of normalizers or postprocessors.

    Built from the deserialized objects rather than the raw request dict, so a
    before/after comparison is apples to apples -- the two differ in shape
    (defaults filled in, ``name`` added) even when they mean the same thing.
    """
    try:
        return json.dumps(
            [x.to_dict() for x in (steps or []) if hasattr(x, "to_dict")],
            sort_keys=True,
            default=str,
        )
    except Exception as exc:
        logger.debug(f"Could not build a chain signature: {exc}")
        return repr(steps)


def is_output_segmentation():
    if len(g.postprocess) == 0:
        return False

    for postprocess in g.postprocess[::-1]:
        if postprocess.is_segmentation is not None:
            return postprocess.is_segmentation


def validate_pipeline_config(config):
    """Helper function to validate pipeline configuration"""
    try:
        normalizer_names = [n.get("name") for n in config.get("input_normalizers", [])]
        available_norms = get_input_normalizers()
        # Extract just the normalizer names from the list of dicts
        available_norm_names = [norm["name"] for norm in available_norms]
        for norm_name in normalizer_names:
            if norm_name not in available_norm_names:
                return {"valid": False, "error": f"Unknown normalizer: {norm_name}"}

        processor_names = [p.get("name") for p in config.get("postprocessors", [])]
        available_procs = get_postprocessors_list()
        # Extract just the postprocessor names from the list of dicts
        available_proc_names = [proc["name"] for proc in available_procs]
        for proc_name in processor_names:
            if proc_name not in available_proc_names:
                return {"valid": False, "error": f"Unknown postprocessor: {proc_name}"}

        return {"valid": True}

    except Exception as e:
        return {"valid": False, "error": str(e)}


_COLOR_RE = re.compile(r'color\(default="([^"]+)"\)')


def _default_prediction_shader(model, host, previous_shader=None):
    """Build a prediction shader over the range the configured chain produces.

    Unlike the raw layer there is nothing to sample here -- reading the model's
    output means running inference -- but there is nothing to sample *for*
    either: the chain's last step fixes the range exactly. See
    output_probe.output_display_range.
    """
    # Keep whatever colour the layer already had, so a recomputed range does
    # not also reshuffle the colours the user is navigating by.
    match = _COLOR_RE.search(previous_shader or "")
    if match:
        color = match.group(1)
    else:
        names = [getattr(j, "model_name", None) for j in g.jobs]
        index = names.index(model) if model in names else 0
        color = PREDICTION_COLORS[index % len(PREDICTION_COLORS)]

    try:
        info = fetch_model_info(host)
        steps = [p.to_dict() for p in (g.postprocess or []) if hasattr(p, "to_dict")]
        value_range = output_display_range(steps, info.get("output_class"))
    except Exception as e:
        logger.debug(f"Could not compute a display range for {model}: {e}")
        value_range = None
    return prediction_shader(color, value_range)


@pipeline_bp.route("/update/equivalences", methods=["POST"])
def update_equivalences():
    equivalences_info = request.get_json()
    dataset = equivalences_info["dataset"]
    equivalences_str = equivalences_info["equivalences"]
    equivalences = [
        [np.uint64(item) for item in sublist] for sublist in equivalences_str
    ]

    with g.viewer.txn() as s:
        for layer in s.layers:
            if layer.source[0].url.endswith(dataset):
                layer.equivalences = equivalences
                break
    return jsonify({"message": "Equivalences updated successfully"})


@pipeline_bp.route("/api/process", methods=["POST"])
def process():
    data = request.get_json()

    # add dashboard url to data so we can update the state from the server
    data["dashboard_url"] = request.host_url

    # we want to set the time such that each request is unique
    data["time"] = time.time()

    logger.debug(f"Data received: {type(data)} - {data.keys()} -{data}")
    custom_code = data.get("custom_code", None)
    if "custom_code" in data:
        del data["custom_code"]
    # Capture which normalization the *currently displayed* raw layer was built
    # under, before it is replaced below.
    previous_norm_signature = _chain_signature(getattr(g, "input_norms", None))
    previous_post_signature = _chain_signature(getattr(g, "postprocess", None))

    logger.debug(f"Data received: {type(data)} - {data.keys()} -{data}")
    g.input_norms = get_normalizations(data["input_norm"])
    # Keep the raw, JSON-serializable input_norm dict around so downstream
    # components (finetune submit/restart, manifest, generated yaml) can
    # propagate the same normalization to the trainer process. Without this
    # the trainer reads raw uint8 from /nrs while inference normalizes to
    # the model's expected range -> trained model never sees inference-scale
    # inputs.
    g.input_norm_config = data.get("input_norm", {}) or {}
    g.postprocess = get_postprocessors(data["postprocess"])
    g.postprocess_config = data.get("postprocess", {}) or {}

    # Save current shader state from viewer before refreshing layers
    _save_shaders_from_viewer()

    # The raw layer is displayed *through* the input normalizers -- its
    # tensorstore is wrapped by LazyNormalization -- so its value range moves
    # when they change: plain uint8 raw spans 0-255, but MinMax+Lambda("x*2-1")
    # puts the same data in [-1, 1]. Restoring a contrast range captured under
    # the old normalization would then map every voxel outside the new range,
    # showing solid black or white. Drop it and let get_raw_layer() recompute
    # percentiles through the normalizers now in effect.
    if previous_norm_signature != _chain_signature(g.input_norms):
        if g.shaders.pop("data", None) is not None:
            logger.info(
                "Input normalization changed; recomputing the raw contrast "
                "range instead of restoring the previous one"
            )
        g.shader_controls.pop("data", None)

    # Prediction layers have the same problem for the same reason: their
    # contrast range is a property of the postprocessing chain, and adding a
    # DefaultPostprocessor moves the output from [0, 1] to 0-255. A restored
    # [0, 1] range over 0-255 data renders every voxel saturated.
    dropped_shaders = {}
    postprocess_changed = previous_post_signature != _chain_signature(g.postprocess)
    if postprocess_changed:
        for job in g.jobs:
            name = getattr(job, "model_name", None)
            dropped_shaders[name] = g.shaders.pop(name, None)
            if dropped_shaders[name] is not None:
                logger.info(
                    f"Postprocessing changed; recomputing the contrast range "
                    f"for {name}"
                )
            g.shader_controls.pop(name, None)

    with g.viewer.txn() as s:
        g.raw = get_raw_layer(g.dataset_path)
        # Restore the user's raw-layer contrast/shader instead of the fresh
        # default get_raw_layer() always builds, which otherwise resets it
        # every time the pipeline is (re)submitted.
        raw_shader = g.shaders.get("data")
        if raw_shader and raw_shader != "None":
            g.raw.shader = raw_shader
        raw_shader_controls = g.shader_controls.get("data")
        if raw_shader_controls:
            g.raw.shaderControls = raw_shader_controls
        s.layers["data"] = g.raw
        for job in g.jobs:
            model = job.model_name
            host = job.host
            if not host:
                logger.warning(f"Skipping layer for {model}: host not yet known")
                continue
            st_data = encode_to_str(data)
            previous_shader = dropped_shaders.get(model)
            shader = g.shaders.get(model)

            if is_output_segmentation():
                s.layers[model] = neuroglancer.SegmentationLayer(
                    source=f"zarr://{host}/{model}{ARGS_KEY}{st_data}{ARGS_KEY}",
                )
            else:
                kwargs = {"source": f"zarr://{host}/{model}{ARGS_KEY}{st_data}{ARGS_KEY}"}
                if not shader:
                    shader = _default_prediction_shader(model, host, previous_shader)
                if shader:
                    kwargs["shader"] = shader
                shader_controls = g.shader_controls.get(model)
                if shader_controls:
                    kwargs["shaderControls"] = shader_controls
                s.layers[model] = neuroglancer.ImageLayer(**kwargs)

    logger.debug(f"Input normalizers: {g.input_norms}")

    if custom_code:
        try:
            # Save custom code to a file with date and time
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"custom_code_{timestamp}.py"
            filepath = os.path.join(g.CUSTOM_CODE_FOLDER, filename)

            with open(filepath, "w") as file:
                file.write(custom_code)

            config = load_safe_config(filepath)
            logger.debug(f"Custom code loaded successfully: {config}")

            logger.debug(get_input_normalizers())

        except Exception as e:
            logger.warning(f"Error executing custom code: {e}")

    return jsonify(
        {
            "message": "Data received successfully",
            "received_data": data,
            "found_custom_normalizer": get_input_normalizers(),
        }
    )


@pipeline_bp.route("/api/pipeline/validate", methods=["POST"])
def validate_pipeline():
    """Validate a pipeline configuration"""
    try:
        data = request.get_json()

        # Validate normalizers
        normalizer_names = [n.get("name") for n in data.get("input_normalizers", [])]
        available_norms = get_input_normalizers()
        available_norm_names = [norm["name"] for norm in available_norms]
        for norm_name in normalizer_names:
            if norm_name not in available_norm_names:
                return jsonify(
                    {"valid": False, "error": f"Unknown normalizer: {norm_name}"}
                ), 400

        # Validate postprocessors
        processor_names = [p.get("name") for p in data.get("postprocessors", [])]
        available_procs = get_postprocessors_list()
        available_proc_names = [proc["name"] for proc in available_procs]
        for proc_name in processor_names:
            if proc_name not in available_proc_names:
                return jsonify(
                    {"valid": False, "error": f"Unknown postprocessor: {proc_name}"}
                ), 400

        return jsonify({"valid": True, "message": "Pipeline is valid"})

    except Exception as e:
        logger.error(f"Error validating pipeline: {e}")
        return jsonify({"valid": False, "error": str(e)}), 500


@pipeline_bp.route("/api/dataset-path", methods=["GET", "POST"])
def dataset_path_api():
    """Get or set the dataset path in globals"""
    if request.method == "GET":
        dataset_path = getattr(g, 'dataset_path', None) or ''
        return jsonify({'dataset_path': dataset_path})
    elif request.method == "POST":
        data = request.get_json()
        dataset_path = data.get('dataset_path', '')
        g.dataset_path = dataset_path
        logger.debug(f"Dataset path updated to: {dataset_path}")
        return jsonify({'success': True, 'dataset_path': g.dataset_path})


@pipeline_bp.route("/api/blockwise-config", methods=["GET", "POST"])
def blockwise_config_api():
    """Get or set blockwise configuration in globals"""
    if request.method == "GET":
        return jsonify({
            'queue': g.queue,
            'charge_group': g.charge_group,
            'nb_cores_master': g.nb_cores_master,
            'nb_cores_worker': g.nb_cores_worker,
            'nb_workers': g.nb_workers,
            'tmp_dir': g.tmp_dir,
            'blockwise_tasks_dir': g.blockwise_tasks_dir
        })
    elif request.method == "POST":
        data = request.get_json()
        g.queue = data.get('queue')
        g.charge_group = data.get('charge_group')
        g.nb_cores_master = int(data.get('nb_cores_master'))
        g.nb_cores_worker = int(data.get('nb_cores_worker'))
        g.nb_workers = int(data.get('nb_workers'))
        g.tmp_dir = data.get('tmp_dir')
        g.blockwise_tasks_dir = data.get('blockwise_tasks_dir')
        logger.debug(f"Blockwise config updated: queue={g.queue}, charge_group={g.charge_group}, cores_master={g.nb_cores_master}, cores_worker={g.nb_cores_worker}, workers={g.nb_workers}, tmp_dir={g.tmp_dir}, blockwise_tasks_dir={g.blockwise_tasks_dir}")
        return jsonify({'success': True, 'config': {
            'queue': g.queue,
            'charge_group': g.charge_group,
            'nb_cores_master': g.nb_cores_master,
            'nb_cores_worker': g.nb_cores_worker,
            'nb_workers': g.nb_workers,
            'tmp_dir': g.tmp_dir,
            'blockwise_tasks_dir': g.blockwise_tasks_dir
        }})


@pipeline_bp.route("/api/pipeline/apply", methods=["POST"])
def apply_pipeline():
    """Apply a pipeline configuration to the current inference"""
    try:
        data = request.get_json()
        logger.debug(f"\n{'='*80}")
        logger.debug(f"APPLY PIPELINE - Received data:")
        logger.debug(f"  Input normalizers: {data.get('input_normalizers', [])}")
        logger.debug(f"  Postprocessors: {data.get('postprocessors', [])}")

        # Validate first
        validation = validate_pipeline_config(data)
        if not validation["valid"]:
            return jsonify(validation), 400

        # Apply normalizers
        input_norms_config = {
            n["name"]: n.get("params", {}) for n in data.get("input_normalizers", [])
        }
        logger.debug(f"\nNormalizers config dict: {input_norms_config}")
        g.input_norms = get_normalizations(input_norms_config)
        # Mirror the JSON-serializable form so finetune submit/restart can
        # propagate it to the trainer process (where g.input_norms can't be
        # easily reconstructed across the LSF process boundary).
        g.input_norm_config = input_norms_config or {}

        # Apply postprocessors
        postprocs_config = {
            p["name"]: p.get("params", {}) for p in data.get("postprocessors", [])
        }
        logger.debug(f"Postprocessors config dict: {postprocs_config}")
        g.postprocess = get_postprocessors(postprocs_config)
        g.postprocess_config = postprocs_config or {}

        # Save complete pipeline visual state to globals
        g.pipeline_inputs = data.get("inputs", [])
        g.pipeline_outputs = data.get("outputs", [])
        g.pipeline_edges = data.get("edges", [])
        g.pipeline_normalizers = data.get("input_normalizers", [])
        g.pipeline_models = data.get("models", [])
        g.pipeline_postprocessors = data.get("postprocessors", [])

        # Also save model configs separately for easier access
        if not hasattr(g, 'pipeline_model_configs'):
            g.pipeline_model_configs = {}
        for model in data.get("models", []):
            if 'config' in model and model['config']:
                g.pipeline_model_configs[model['name']] = model['config']

        # Log the updated globals state
        logger.debug(f"\n{'='*80}")
        logger.debug(f"UPDATED GLOBALS (g) STATE:")
        logger.debug(f"{'='*80}")
        logger.debug(f"\ng.input_norms ({len(g.input_norms)} items):")
        for idx, norm in enumerate(g.input_norms):
            logger.debug(f"  [{idx}] {norm}")

        logger.debug(f"\ng.postprocess ({len(g.postprocess)} items):")
        for idx, post in enumerate(g.postprocess):
            logger.debug(f"  [{idx}] {post}")

        logger.debug(f"\ng.jobs ({len(g.jobs)} items):")
        for idx, job in enumerate(g.jobs):
            logger.debug(f"  [{idx}] model_name={getattr(job, 'model_name', 'N/A')}, host={getattr(job, 'host', 'N/A')}")

        logger.debug(f"\ng.pipeline_inputs ({len(g.pipeline_inputs)} items): {g.pipeline_inputs}")
        logger.debug(f"\ng.pipeline_outputs ({len(g.pipeline_outputs)} items): {g.pipeline_outputs}")
        logger.debug(f"\ng.pipeline_edges ({len(g.pipeline_edges)} items): {g.pipeline_edges}")
        logger.debug(f"\ng.pipeline_normalizers ({len(g.pipeline_normalizers)} items): {g.pipeline_normalizers}")
        logger.debug(f"\ng.pipeline_models ({len(g.pipeline_models)} items): {g.pipeline_models}")
        logger.debug(f"\ng.pipeline_postprocessors ({len(g.pipeline_postprocessors)} items): {g.pipeline_postprocessors}")

        logger.debug(f"{'='*80}\n")

        return jsonify({
            "message": "Pipeline applied successfully",
            "normalizers_applied": len(g.input_norms),
            "postprocessors_applied": len(g.postprocess),
        })

    except Exception as e:
        logger.error(f"Error applying pipeline: {e}")
        return jsonify({"error": str(e)}), 500


@pipeline_bp.route("/api/shaders", methods=["GET", "POST"])
def shaders_api():
    """Get or update stored shader strings and shaderControls.

    GET  -> returns current g.shaders and g.shader_controls
    POST -> merges incoming {"shaders": {...}, "shader_controls": {...}} into globals
           (also accepts flat {layer_name: shader_str} for backwards compat)
    """
    if request.method == "GET":
        # Also sync from viewer if available
        _save_shaders_from_viewer()
        return jsonify({"shaders": g.shaders, "shader_controls": g.shader_controls})

    data = request.get_json()
    if not isinstance(data, dict):
        return jsonify({"error": "Expected a JSON object"}), 400

    # Support both structured and flat formats
    if "shaders" in data or "shader_controls" in data:
        if "shaders" in data:
            g.shaders.update(data["shaders"])
        if "shader_controls" in data:
            g.shader_controls.update(data["shader_controls"])
    else:
        # Flat dict — treat as shaders only (backwards compat)
        g.shaders.update(data)

    logger.info(f"Shaders updated for layers: {list(g.shaders.keys())}")
    logger.info(f"ShaderControls updated for layers: {list(g.shader_controls.keys())}")
    return jsonify({"message": "Shaders updated", "shaders": g.shaders, "shader_controls": g.shader_controls})
