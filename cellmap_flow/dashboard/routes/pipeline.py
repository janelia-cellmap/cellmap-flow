import json
import logging

import numpy as np
from flask import Blueprint, request, jsonify

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.norm.input_normalize import get_input_normalizers
from cellmap_flow.pipeline_spec import PipelineSpec
from cellmap_flow.post.postprocessors import get_postprocessors_list
from cellmap_flow.utils.scale_pyramid import PREDICTION_COLORS
from cellmap_flow.utils.server_info import fetch_model_info
from cellmap_flow.viewer.layers import prediction_layer, raw_layer

logger = logging.getLogger(__name__)

pipeline_bp = Blueprint("pipeline", __name__)


def _save_shaders_from_viewer() -> None:
    """Keep each layer's shader and shaderControls, as the user set them in the viewer."""
    session = get_session()
    if session.viewer is None:
        return
    try:
        state = session.viewer.state
        for layer in state.layers:
            shader = getattr(layer, "shader", None)
            # A neuroglancer layer with no shader set reports the *string*
            # "None" (not Python None), which is truthy. Storing it would
            # later be restored onto the layer verbatim and fail to compile,
            # wiping the user's rendering. Treat it as "unset".
            if shader and shader != "None":
                session.shaders[layer.name] = shader
            shader_controls = getattr(layer, "shaderControls", None) or getattr(layer, "shader_controls", None)
            if shader_controls:
                session.shader_controls[layer.name] = shader_controls
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


@pipeline_bp.route("/update/equivalences", methods=["POST"])
def update_equivalences():
    equivalences_info = request.get_json()
    dataset = equivalences_info["dataset"]
    equivalences_str = equivalences_info["equivalences"]
    equivalences = [
        [np.uint64(item) for item in sublist] for sublist in equivalences_str
    ]

    with get_session().viewer.txn() as s:
        for layer in s.layers:
            if layer.source[0].url.endswith(dataset):
                layer.equivalences = equivalences
                break
    return jsonify({"message": "Equivalences updated successfully"})


@pipeline_bp.route("/api/process", methods=["POST"])
def process():
    session = get_session()
    data = request.get_json()

    # add dashboard url to data so we can update the state from the server
    data["dashboard_url"] = request.host_url

    # Capture which normalization the *currently displayed* raw layer was built
    # under, before it is replaced below.
    previous_norm_signature = _chain_signature(session.input_norms)
    previous_post_signature = _chain_signature(session.postprocess)

    logger.debug(f"Data received: {type(data)} - {data.keys()} -{data}")
    # The posted steps are kept as the config, so finetune submit/restart,
    # the manifest and the exported YAML hand the trainer the normalization
    # inference uses. Without it the trainer reads raw uint8 from /nrs while
    # inference normalizes to the model's expected range.
    spec = PipelineSpec.from_json_data(data, strict=True)
    session.set_pipeline(spec)
    # Named by content rather than stamped with the time: resubmitting the
    # same settings gives the same layer source, so neuroglancer keeps the
    # chunks it has and each server reuses the chain it already built (with
    # any merger state in it). Changed settings still give a new source.
    data["digest"] = spec.digest()
    st_data = spec.to_url_blob(
        dashboard_url=data["dashboard_url"], digest=data["digest"]
    )

    # Save current shader state from viewer before refreshing layers
    _save_shaders_from_viewer()

    # The raw layer is displayed *through* the input normalizers -- its
    # tensorstore is wrapped by LazyNormalization -- so its value range moves
    # when they change: plain uint8 raw spans 0-255, but MinMax+Lambda("x*2-1")
    # puts the same data in [-1, 1]. Restoring a contrast range captured under
    # the old normalization would then map every voxel outside the new range,
    # showing solid black or white. Drop it and let get_raw_layer() recompute
    # percentiles through the normalizers now in effect.
    if previous_norm_signature != _chain_signature(session.input_norms):
        if session.shaders.pop("data", None) is not None:
            logger.info(
                "Input normalization changed; recomputing the raw contrast "
                "range instead of restoring the previous one"
            )
        session.shader_controls.pop("data", None)

    # Prediction layers have the same problem for the same reason: their
    # contrast range is a property of the postprocessing chain, and adding a
    # DefaultPostprocessor moves the output from [0, 1] to 0-255. A restored
    # [0, 1] range over 0-255 data renders every voxel saturated.
    dropped_shaders = {}
    postprocess_changed = previous_post_signature != _chain_signature(session.postprocess)
    if postprocess_changed:
        for job in session.jobs:
            name = getattr(job, "model_name", None)
            dropped_shaders[name] = session.shaders.pop(name, None)
            if dropped_shaders[name] is not None:
                logger.info(
                    f"Postprocessing changed; recomputing the contrast range "
                    f"for {name}"
                )
            session.shader_controls.pop(name, None)

    with session.viewer.txn() as s:
        # The user's raw-layer contrast/shader, instead of the fresh default
        # get_raw_layer() always builds, which otherwise resets it every time
        # the pipeline is (re)submitted.
        session.raw = raw_layer(session.dataset_path, shader=session.shaders.get("data"),
                                shader_controls=session.shader_controls.get("data"))
        s.layers["data"] = session.raw
        for index, job in enumerate(session.jobs):
            model = job.model_name
            host = job.host
            if not host:
                # Submitted without waiting for a host (wait_for_host=False)
                # and not up yet: there is no URL to point a layer at.
                logger.info(f"Skipping layer for {model}: its job has no host yet")
                continue
            # Without a shader of the user's, one over the chain's range in
            # the colour the layer had, if its shader was dropped above.
            s.layers[model] = prediction_layer(
                model, host, st_data, dataset_path=session.dataset_path, postprocess=session.postprocess,
                shader=session.shaders.get(model), shader_controls=session.shader_controls.get(model),
                previous_shader=dropped_shaders.get(model), color=PREDICTION_COLORS[index % len(PREDICTION_COLORS)],
                info=fetch_model_info(host),
            )

    logger.debug(f"Input normalizers: {session.input_norms}")

    return jsonify(
        {
            "message": "Data received successfully",
            "received_data": data,
        }
    )


@pipeline_bp.route("/api/blockwise-config", methods=["GET", "POST"])
def blockwise_config_api():
    """Get or set the blockwise settings."""
    session = get_session()
    if request.method == "GET":
        return jsonify({
            'queue': session.queue,
            'charge_group': session.charge_group,
            'nb_cores_master': session.nb_cores_master,
            'nb_cores_worker': session.nb_cores_worker,
            'nb_workers': session.nb_workers,
            'tmp_dir': session.tmp_dir,
            'blockwise_tasks_dir': session.blockwise_tasks_dir
        })
    elif request.method == "POST":
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'success': False, 'error': 'expected a JSON object'}), 400
        # Parse every number before changing anything, so a bad value is a
        # 400 that leaves the settings as they were, not a 500 halfway through.
        counts = {}
        for key in ('nb_cores_master', 'nb_cores_worker', 'nb_workers'):
            try:
                counts[key] = int(data.get(key))
            except (TypeError, ValueError):
                return jsonify({
                    'success': False,
                    'error': f'{key} must be a whole number, got {data.get(key)!r}',
                }), 400
        session.queue = data.get('queue')
        session.charge_group = data.get('charge_group')
        session.nb_cores_master = counts['nb_cores_master']
        session.nb_cores_worker = counts['nb_cores_worker']
        session.nb_workers = counts['nb_workers']
        session.tmp_dir = data.get('tmp_dir')
        session.blockwise_tasks_dir = data.get('blockwise_tasks_dir')
        logger.debug(f"Blockwise config updated: queue={session.queue}, charge_group={session.charge_group}, cores_master={session.nb_cores_master}, cores_worker={session.nb_cores_worker}, workers={session.nb_workers}, tmp_dir={session.tmp_dir}, blockwise_tasks_dir={session.blockwise_tasks_dir}")
        return jsonify({'success': True, 'config': {
            'queue': session.queue,
            'charge_group': session.charge_group,
            'nb_cores_master': session.nb_cores_master,
            'nb_cores_worker': session.nb_cores_worker,
            'nb_workers': session.nb_workers,
            'tmp_dir': session.tmp_dir,
            'blockwise_tasks_dir': session.blockwise_tasks_dir
        }})


@pipeline_bp.route("/api/pipeline/apply", methods=["POST"])
def apply_pipeline():
    """Apply a pipeline configuration to the current inference"""
    try:
        session = get_session()
        data = request.get_json()
        logger.debug(f"Apply pipeline: {data}")

        # Validate first
        validation = validate_pipeline_config(data)
        if not validation["valid"]:
            return jsonify(validation), 400

        # Ordered lists, not dicts keyed by name: two steps of the same class
        # (two LambdaNormalizers, say) collapsed into one under a dict.
        spec = PipelineSpec.from_builder(
            data.get("input_normalizers", []), data.get("postprocessors", [])
        )
        session.set_pipeline(spec)

        # The builder's whole pipeline, as it sent it, for its next load.
        session.builder_state = {
            "inputs": data.get("inputs", []),
            "outputs": data.get("outputs", []),
            "edges": data.get("edges", []),
            "normalizers": data.get("input_normalizers", []),
            "models": data.get("models", []),
            "postprocessors": data.get("postprocessors", []),
        }
        # And each model's config, for a model node that comes back without one.
        for model in data.get("models", []):
            if 'config' in model and model['config']:
                session.builder_model_configs[model['name']] = model['config']
        logger.debug(f"Applied: input_norms={session.input_norms}, postprocess={session.postprocess}")

        return jsonify({
            "message": "Pipeline applied successfully",
            "normalizers_applied": len(session.input_norms),
            "postprocessors_applied": len(session.postprocess),
        })

    except Exception as e:
        logger.error(f"Error applying pipeline: {e}")
        return jsonify({"error": str(e)}), 500
