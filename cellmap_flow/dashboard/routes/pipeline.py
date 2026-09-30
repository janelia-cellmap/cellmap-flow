"""The dashboard's chain, and the settings the pipeline builder keeps here.

- ``PUT /api/pipeline``: set the chain and redraw the viewer through it (the
  dashboard page's Submit, and the pipeline builder after each edit).
- ``POST /api/process`` and ``POST /api/pipeline/apply``: the two routes it
  replaced, kept for one release as its deprecated aliases.
- ``/api/blockwise-config``: the builder's blockwise settings.
- ``/update/equivalences``: a segmentation layer's merged ids.
"""

import functools
import json
import logging
from typing import Optional

import numpy as np
from flask import Blueprint, jsonify, make_response, request
from pydantic import BaseModel, ValidationInfo, field_validator

from cellmap_flow.dashboard.requests import BlockwiseSettings, parse
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.norm.input_normalize import get_input_normalizers
from cellmap_flow.pipeline_spec import PipelineSpec
from cellmap_flow.post.postprocessors import get_postprocessors_list
from cellmap_flow.viewer.raw import PREDICTION_COLORS
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


def _unknown_op(kind, names):
    """"Unknown <kind>: <name>" for the first of ``names`` that no registered
    op of ``kind`` ("normalizer" or "postprocessor") is called; else None."""
    ops = get_input_normalizers() if kind == "normalizer" else get_postprocessors_list()
    known = {op["name"] for op in ops}
    for name in names:
        if name not in known:
            return f"Unknown {kind}: {name}"
    return None


def validate_pipeline_config(config):
    """/api/pipeline/apply's check of the builder's nodes: ``{"valid": True}``,
    or ``{"valid": False, "error"}`` for the first node that names no
    registered op, or for a body it cannot read."""
    try:
        error = (_unknown_op("normalizer", [n.get("name") for n in config.get("input_normalizers", [])])
                 or _unknown_op("postprocessor", [p.get("name") for p in config.get("postprocessors", [])]))
    except Exception as e:
        return {"valid": False, "error": str(e)}
    return {"valid": False, "error": error} if error else {"valid": True}


class BuilderCanvas(BaseModel):
    """The pipeline builder's canvas as it sends it: its nodes by type, and
    its edges, each a list (a missing one is empty). The dashboard keeps it
    for the builder's next load, and each model node's ``config`` for a
    model node that comes back without one; the chain is not read from it."""

    inputs: list[dict] = []
    outputs: list[dict] = []
    edges: list[dict] = []
    normalizers: list[dict] = []
    models: list[dict] = []
    postprocessors: list[dict] = []


class PipelineUpdate(BaseModel):
    """The body of PUT /api/pipeline.

    ``input_norm`` and ``postprocess`` are the two chains, each a list of
    ``{"name": <op class>, **its parameters}`` steps in the order they run
    (pipeline_spec's form). Both are required, ``[]`` for none, and every
    step must name a registered op of its kind. ``builder`` is the pipeline
    builder's canvas (BuilderCanvas); only the builder sends it.
    """

    input_norm: list[dict]
    postprocess: list[dict]
    builder: Optional[BuilderCanvas] = None

    @field_validator("input_norm", "postprocess")
    @classmethod
    def _registered_ops(cls, steps, info: ValidationInfo):
        kind = "normalizer" if info.field_name == "input_norm" else "postprocessor"
        error = _unknown_op(kind, [step.get("name") for step in steps])
        if error:
            raise ValueError(error)
        return steps


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


def _set_chain_and_redraw(spec, dashboard_url, *, built=None, builder=None) -> list:
    """Make ``spec`` the dashboard's chain, and redraw the viewer through it.

    What PUT /api/pipeline and both its aliases do. set_pipeline() builds
    every step (or takes ``built``, the steps already built) before it
    assigns anything, so a step its class refuses raises here with nothing
    changed. ``builder``, the pipeline builder's canvas as a BuilderCanvas
    dict, is kept when given. Then the raw layer and each prediction layer
    are rebuilt: a prediction layer's URL carries the chain (the args blob,
    with ``dashboard_url`` and the chain's digest), and its server runs the
    chain of the layer it is asked for.

    Returns the names of the prediction layers drawn. A job with no host
    yet gets no layer, and with no viewer yet (no dataset opened) nothing
    is drawn.
    """
    session = get_session()
    # Capture which normalization the *currently displayed* raw layer was built
    # under, before it is replaced below.
    previous_norm_signature = _chain_signature(session.input_norms)
    previous_post_signature = _chain_signature(session.postprocess)

    # The steps are kept as the config, so finetune submit/restart, the
    # manifest and the exported YAML hand the trainer the normalization
    # inference uses. Without it the trainer reads raw uint8 from /nrs while
    # inference normalizes to the model's expected range.
    session.set_pipeline(spec, built=built)
    if builder is not None:
        session.builder_state = builder
        for model in builder["models"]:
            if model.get("name") and model.get("config"):
                session.builder_model_configs[model["name"]] = model["config"]
    if session.viewer is None:
        return []
    # Named by content rather than stamped with the time: resubmitting the
    # same settings gives the same layer source, so neuroglancer keeps the
    # chunks it has and each server reuses the chain it already built (with
    # any merger state in it). Changed settings still give a new source.
    st_data = spec.to_url_blob(dashboard_url=dashboard_url, digest=spec.digest())

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

    drawn = []
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
            drawn.append(model)

    logger.debug(f"Input normalizers: {session.input_norms}")
    return drawn


@pipeline_bp.route("/api/pipeline", methods=["PUT"])
def put_pipeline():
    """Set the chain, and redraw the viewer through it.

    The body is a PipelineUpdate. The answer is ``{"success": true,
    "pipeline": {"input_norm", "postprocess"}, "digest", "layers"}``: the
    chain as it is now configured, its digest (which names the layers'
    source, so the same chain sent again leaves neuroglancer's chunks and
    each server's built chain as they are), and the prediction layers drawn.
    A body that is not a PipelineUpdate, or a step its op's class refuses,
    is a 400 ``{"success": false, "error"}`` that changes nothing.
    """
    body, error = parse(PipelineUpdate, request.get_json(silent=True))
    if error:
        return error
    spec = PipelineSpec(body.input_norm, body.postprocess)
    try:
        built = spec.build()
    except (TypeError, ValueError) as e:
        return jsonify({"success": False, "error": str(e)}), 400
    builder = body.builder.model_dump() if body.builder is not None else None
    layers = _set_chain_and_redraw(spec, request.host_url, built=built, builder=builder)
    return jsonify({"success": True, "pipeline": spec.to_json_data(), "digest": spec.digest(), "layers": layers})


# The routes PUT /api/pipeline replaced, kept for one release as its aliases.
# The Deprecation header is RFC 9745's: the date the route was deprecated.
_DEPRECATED_SINCE = "@1790726400"  # 2026-09-30


def _deprecated_alias(view):
    """A route kept for one release as an alias of PUT /api/pipeline. Each
    call is logged as a warning, and each answer carries a Deprecation
    header and a Link to PUT /api/pipeline; the answer is otherwise the
    route's own."""
    @functools.wraps(view)
    def alias(*args, **kwargs):
        logger.warning(f"{request.method} {request.path} is deprecated and goes in the next release; "
                       "use PUT /api/pipeline")
        response = make_response(view(*args, **kwargs))
        response.headers["Deprecation"] = _DEPRECATED_SINCE
        response.headers["Link"] = '</api/pipeline>; rel="successor-version"'
        return response
    return alias


@pipeline_bp.route("/api/process", methods=["POST"])
@_deprecated_alias
def process():
    """Submit's route before PUT /api/pipeline, answering as it did.

    The body is ``{"input_norm", "postprocess"}``, either chain in the list
    or the older dict form. The answer is ``{"message", "received_data"}``,
    the body with the dashboard's address and the chain's digest added. An
    op it does not know is kept in the chain and skipped where it is built;
    a body without both chains, or a step its class refuses, is a 500.
    """
    data = request.get_json()

    # add dashboard url to data so we can update the state from the server
    data["dashboard_url"] = request.host_url

    logger.debug(f"Data received: {type(data)} - {data.keys()} -{data}")
    spec = PipelineSpec.from_json_data(data, strict=True)
    _set_chain_and_redraw(spec, data["dashboard_url"])
    data["digest"] = spec.digest()

    return jsonify(
        {
            "message": "Data received successfully",
            "received_data": data,
        }
    )


@pipeline_bp.route("/api/pipeline/apply", methods=["POST"])
@_deprecated_alias
def apply_pipeline():
    """The pipeline builder's route before PUT /api/pipeline, answering as it did.

    The body is the builder's nodes by type (``input_normalizers``,
    ``postprocessors``, ``models``, ``inputs``, ``outputs``) and its
    ``edges``. The chain is taken from the normalizer and postprocessor
    nodes, each ``{"name", "params"}``; everything is kept as the builder's
    canvas. The answer is ``{"message", "normalizers_applied",
    "postprocessors_applied"}``. A node that names no registered op is a 400
    ``{"valid": false, "error"}``, and anything else that goes wrong a 500
    ``{"error"}``.
    """
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
        _set_chain_and_redraw(spec, request.host_url, builder={
            "inputs": data.get("inputs", []),
            "outputs": data.get("outputs", []),
            "edges": data.get("edges", []),
            "normalizers": data.get("input_normalizers", []),
            "models": data.get("models", []),
            "postprocessors": data.get("postprocessors", []),
        })
        logger.debug(f"Applied: input_norms={session.input_norms}, postprocess={session.postprocess}")

        return jsonify({
            "message": "Pipeline applied successfully",
            "normalizers_applied": len(session.input_norms),
            "postprocessors_applied": len(session.postprocess),
        })

    except Exception as e:
        logger.error(f"Error applying pipeline: {e}")
        return jsonify({"error": str(e)}), 500


@pipeline_bp.route("/api/blockwise-config", methods=["GET", "POST"])
def blockwise_config_api():
    """Get or set the blockwise settings."""
    session = get_session()
    if request.method == "POST":
        # Parsed whole before anything changes, so a bad value is a 400 that
        # leaves the settings as they were, not a 500 halfway through.
        body, error = parse(BlockwiseSettings, request.get_json(silent=True))
        if error:
            return error
        for key, value in body.model_dump().items():
            setattr(session, key, value)
    settings = {key: getattr(session, key) for key in BlockwiseSettings.model_fields}
    if request.method == "GET":
        return jsonify(settings)
    logger.debug(f"Blockwise config updated: {settings}")
    return jsonify({"success": True, "config": settings})
