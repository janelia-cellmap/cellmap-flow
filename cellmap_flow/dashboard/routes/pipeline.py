import json
import logging

import numpy as np
from flask import Blueprint, request, jsonify

from cellmap_flow.globals import g
from cellmap_flow.norm.input_normalize import get_input_normalizers
from cellmap_flow.pipeline_spec import PipelineSpec
from cellmap_flow.post.postprocessors import get_postprocessors_list
from cellmap_flow.utils.scale_pyramid import PREDICTION_COLORS
from cellmap_flow.utils.server_info import fetch_model_info
from cellmap_flow.viewer.layers import prediction_layer, prediction_shader_for, raw_layer

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

    # Capture which normalization the *currently displayed* raw layer was built
    # under, before it is replaced below.
    previous_norm_signature = _chain_signature(getattr(g, "input_norms", None))
    previous_post_signature = _chain_signature(getattr(g, "postprocess", None))

    logger.debug(f"Data received: {type(data)} - {data.keys()} -{data}")
    # The posted steps are kept as the config, so finetune submit/restart,
    # the manifest and the exported YAML hand the trainer the normalization
    # inference uses. Without it the trainer reads raw uint8 from /nrs while
    # inference normalizes to the model's expected range.
    spec = PipelineSpec.from_json_data(data, strict=True)
    g.set_pipeline(spec)
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
        # The user's raw-layer contrast/shader, instead of the fresh default
        # get_raw_layer() always builds, which otherwise resets it every time
        # the pipeline is (re)submitted.
        g.raw = raw_layer(g.dataset_path, shader=g.shaders.get("data"),
                          shader_controls=g.shader_controls.get("data"))
        s.layers["data"] = g.raw
        for index, job in enumerate(g.jobs):
            model = job.model_name
            host = job.host
            if not host:
                # Submitted without waiting for a host (wait_for_host=False)
                # and not up yet: there is no URL to point a layer at.
                logger.info(f"Skipping layer for {model}: its job has no host yet")
                continue
            info = fetch_model_info(host)
            # The user's shader, else one over the chain's range in the
            # layer's colour (the one it had, if its shader was dropped above).
            shader = g.shaders.get(model) or prediction_shader_for(
                model, host, g.postprocess, previous_shader=dropped_shaders.get(model),
                color=PREDICTION_COLORS[index % len(PREDICTION_COLORS)], info=info,
            )
            s.layers[model] = prediction_layer(
                model, host, st_data, dataset_path=g.dataset_path, postprocess=g.postprocess,
                shader=shader, shader_controls=g.shader_controls.get(model), info=info,
            )

    logger.debug(f"Input normalizers: {g.input_norms}")

    return jsonify(
        {
            "message": "Data received successfully",
            "received_data": data,
        }
    )


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
        g.queue = data.get('queue')
        g.charge_group = data.get('charge_group')
        g.nb_cores_master = counts['nb_cores_master']
        g.nb_cores_worker = counts['nb_cores_worker']
        g.nb_workers = counts['nb_workers']
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

        # Ordered lists, not dicts keyed by name: two steps of the same class
        # (two LambdaNormalizers, say) collapsed into one under a dict.
        spec = PipelineSpec.from_builder(
            data.get("input_normalizers", []), data.get("postprocessors", [])
        )
        logger.debug(f"\nNormalizers config dict: {list(spec.input_norm)}")
        logger.debug(f"Postprocessors config dict: {list(spec.postprocess)}")
        g.set_pipeline(spec)

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

