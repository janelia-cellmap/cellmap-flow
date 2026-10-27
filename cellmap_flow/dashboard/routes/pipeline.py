"""The dashboard's chain, and the settings the pipeline builder keeps here.

- ``PUT /api/pipeline``: set the chain and redraw the viewer through it (the
  dashboard page's Apply, and the pipeline builder after each edit).
- ``set_declared_input_norm()``: a model's start sets the input normalization
  it declares, or its kind of model's convention, when the chain has none
  (services.launch.update_run_models).
- ``set_needed_postprocess()``: a model's start sets the postprocessing its
  output needs, a sigmoid for logits, when the chain has none, once its
  server's probe says (services.launch.run_model).
- ``POST /api/process`` and ``POST /api/pipeline/apply``: the two routes it
  replaced, kept for one release as its deprecated aliases.
- ``/api/blockwise-config``: the builder's blockwise settings.
- ``/update/equivalences``: a segmentation layer's merged ids.
"""

import functools
import json
import logging
import threading
from typing import Optional

import numpy as np
from flask import Blueprint, jsonify, make_response, request
from pydantic import BaseModel, ValidationInfo, field_validator

from cellmap_flow.dashboard.requests import BlockwiseSettings, Equivalences, parse
from cellmap_flow.dashboard.routes import model_advice
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.norm.input_normalize import get_input_normalizers
from cellmap_flow.pipeline_spec import PipelineSpec
from cellmap_flow.portable import rank
from cellmap_flow.post.postprocessors import get_postprocessors_list
from cellmap_flow.viewer.raw import PREDICTION_COLORS, view_of
from cellmap_flow.serving.client import fetch_model_info
from cellmap_flow.viewer.layers import prediction_layer, prediction_url, raw_layer, replace_layer

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
    op of ``kind`` ("normalizer" or "postprocessor") is called, a name that
    is not text included; else None."""
    ops = get_input_normalizers() if kind == "normalizer" else get_postprocessors_list()
    known = {op["name"] for op in ops}
    for name in names:
        if not isinstance(name, str) or name not in known:
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
    """Show a merging postprocessor's merged ids (an Equivalences) in its
    segmentation layer. Its inference server posts them as it merges."""
    body, error = parse(Equivalences, request.get_json(silent=True))
    if error:
        return error
    dataset = body.dataset
    equivalences = [[np.uint64(item) for item in sublist] for sublist in body.equivalences]

    with get_session().viewer.txn() as s:
        for layer in s.layers:
            if layer.source[0].url.endswith(dataset):
                layer.equivalences = equivalences
                break
    return jsonify({"message": "Equivalences updated successfully"})


def _shows(viewer, jobs, url_blob) -> bool:
    """Whether ``viewer`` already shows the raw, and each job that has a host
    its layer over ``url_blob``: then a redraw would show nothing new."""
    layers = viewer.state.layers
    if "data" not in layers:
        return False
    for job in jobs:
        if not job.host:
            continue
        if job.model_name not in layers:
            return False
        if layers[job.model_name].source[0].url != prediction_url(job.host, job.model_name, url_blob):
            return False
    return True


def _set_chain_and_redraw(spec, dashboard_url, *, built=None, builder=None) -> list:
    """Make ``spec`` the dashboard's chain, and redraw the viewer through it.

    What PUT /api/pipeline and both its aliases do. set_pipeline() builds
    every step (or takes ``built``, the steps already built) before it
    assigns anything, so a step its class refuses raises here with nothing
    changed. ``builder``, the pipeline builder's canvas as a BuilderCanvas
    dict, is kept when given. Then each prediction layer is rebuilt: its URL
    carries the chain (the args blob, with the dashboard's address, or else
    ``dashboard_url``, the request's, and the chain's digest:
    Session.layer_blob), and its server runs the chain of the layer it is
    asked for. The raw layer is rebuilt only when the input normalization
    changed (or the viewer has none): it is drawn through the normalizers.

    The viewer is left alone when the chain is the one it already shows: the
    chain was this already, the raw layer is there, and every job's layer
    carries this chain. The builder sends its canvas after every edit, a
    node dragged too, and each redraw sampled the raw again, asked every
    server for its model_info and reset what the user had set in the viewer
    (a layer hidden, an opacity).

    Returns the names of the prediction layers shown through the chain. A
    job with no host yet gets no layer, and with no viewer yet (no dataset
    opened) nothing is drawn.
    """
    session = get_session()
    previous_digest = session.pipeline_spec.digest()
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
    # The blob every other path gives a layer (Session.layer_blob), so a
    # layer a launch drew over this chain is left as it is.
    st_data = session.layer_blob(dashboard_url)
    if spec.digest() == previous_digest and _shows(session.viewer, session.jobs, st_data):
        logger.debug("The viewer already shows this chain; it is left as it is")
        return [job.model_name for job in session.jobs if job.host]

    # Save current shader state from viewer before refreshing layers
    _save_shaders_from_viewer()

    # The raw layer is displayed *through* the input normalizers -- every
    # chunk viewer.raw serves goes through them -- so its value range moves
    # when they change: plain uint8 raw spans 0-255, but MinMax+Lambda("x*2-1")
    # puts the same data in [-1, 1]. Restoring a contrast range captured under
    # the old normalization would then map every voxel outside the new range,
    # showing solid black or white. Drop it and let get_raw_layer() recompute
    # percentiles through the normalizers now in effect.
    input_changed = previous_norm_signature != _chain_signature(session.input_norms)
    if input_changed:
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

    # A raw layer built again is a new source, with a token of its own, so
    # the browser drops every raw chunk it has and fetches them all again
    # through this process: the raw stayed black for a long while after an
    # Apply that changed only the postprocessing. Its chunks only change
    # with the input normalization, so otherwise the one shown stays, with
    # the shader and controls the user has on it.
    # .state, not .txn(): this only reads.
    redraw_raw = input_changed or "data" not in session.viewer.state.layers
    # Without a contrast of the user's to restore, the raw's is sampled where
    # the viewer is looking, as neuroglancer's 1-99% button would be.
    view = view_of(session.viewer.state) if redraw_raw and session.shaders.get("data") is None else None

    drawn = []
    with session.viewer.txn() as s:
        if redraw_raw:
            # The user's raw-layer contrast/shader, instead of the fresh
            # default get_raw_layer() always builds, which otherwise resets
            # it every time the pipeline is (re)submitted.
            session.raw = raw_layer(session.dataset_path, shader=session.shaders.get("data"),
                                    shader_controls=session.shader_controls.get("data"), view=view)
            replace_layer(s.layers, "data", session.raw)
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
            replace_layer(s.layers, model, prediction_layer(
                model, host, st_data, dataset_path=session.dataset_path, postprocess=session.postprocess,
                shader=session.shaders.get(model), shader_controls=session.shader_controls.get(model),
                previous_shader=dropped_shaders.get(model), color=PREDICTION_COLORS[index % len(PREDICTION_COLORS)],
                info=fetch_model_info(host), channel=session.shown_channel(model),
            ))
            drawn.append(model)

    logger.debug(f"Input normalizers: {session.input_norms}")
    return drawn


def _declared_steps(model_config, data_range) -> list:
    """The input normalization ``model_config`` declares itself
    (``ModelConfig.declared_input_norm``: an onnx model's manifest, a
    cellmap-models export's metadata.json, for raw data spanning
    ``data_range``), as chain steps, each a copied ``{"name", **params}``
    dict; ``[]`` for a model that declares none, and for anything that is
    not a step."""
    declared = getattr(model_config, "declared_input_norm", None)
    steps = declared(data_range=data_range) if callable(declared) else None
    return [dict(step) for step in steps or [] if isinstance(step, dict) and step.get("name")]


def start_input_norm(model_config, raw_dtype, raw_range):
    """What ``model_config``'s start sets on a chain with no input
    normalization, for raw data of ``raw_dtype`` spanning ``raw_range``: (the
    chain steps, whether the model declares them). Its declaration
    (_declared_steps), else the convention for its kind of model, a CellMap
    or DaCapo model's [-1, 1] (the model advice's suggestion at medium
    confidence, model_advice.input_norm_suggestion). ``([], False)`` for a
    model nothing is known of, whose suggestion of the range assumed stays a
    suggestion, and without the raw data's range (no dataset open), where
    the convention's MinMaxNormalizer would be a guess. The Finetune tab's
    ranking reads a model through these steps too (finetune/rank_routes.py),
    whatever the chain has."""
    steps = _declared_steps(model_config, raw_range)
    if steps or raw_range is None:
        return steps, bool(steps)
    suggestion, _ = model_advice.input_norm_suggestion(
        model_config, model_advice.model_metadata(model_config), raw_dtype, raw_range)
    if suggestion["confidence"] != "medium":
        return [], False
    return [{"name": name, **suggestion["input_norm"][name]} for name in suggestion["order"]], False


def set_declared_input_norm(model_configs) -> Optional[str]:
    """Set the input normalization a model declares as the chain's, or the
    convention for its kind of model when it declares none, as part of
    starting it, when the chain has none. Returns the name of the model
    whose input normalization it is, or None when nothing changed.

    The Models tab's Submit (services.launch.update_run_models) calls this
    with the models it is about to start, in the tab's order, before their
    layers are drawn, so they are drawn through it. The first of them with
    any steps to set (start_input_norm: declared, or a CellMap or DaCapo
    model's convention) gives them, and the postprocessing stays as it is;
    the chain is set through _set_chain_and_redraw, as PUT /api/pipeline
    sets it, so the layers of the models already running are redrawn
    through it too. A chain that has input normalizers already (a YAML's,
    the user's Apply) is left alone, and after this only Apply changes it:
    the Processing tab's input list shows the chain as set (GET
    /api/models/loaded's ``pipeline``, which the page reads after every
    model change) and the notice that says where it came from. A model with
    nothing to set (one nothing is known of) changes nothing. Steps that
    name an op the registry does not have, or a value its class refuses,
    are skipped with a warning, and the next model's are tried.

    The browser's twin is web/src/backend/serving.ts setDeclaredInputNorm,
    whose models all declare theirs (their manifests).
    """
    session = get_session()
    if session.pipeline_spec.input_norm:
        return None
    raw_dtype, raw_range = model_advice.raw_dtype_and_range(session.dataset_path)
    for model_config in model_configs:
        steps, declared = start_input_norm(model_config, raw_dtype, raw_range)
        if not steps:
            continue
        name = getattr(model_config, "name", None)
        what = "declared input normalization" if declared else "input normalization by convention"
        named = " → ".join(step["name"] for step in steps)
        unknown = _unknown_op("normalizer", [step["name"] for step in steps])
        if unknown:
            logger.warning(f"{name}'s {what} is not used: {unknown}")
            continue
        spec = PipelineSpec(steps, session.pipeline_spec.postprocess)
        try:
            built = spec.build()
        except (TypeError, ValueError) as e:
            logger.warning(f"{name}'s {what} is not used: {e}")
            continue
        _set_chain_and_redraw(spec, None, built=built)
        logger.info(f"Input normalization set from {name}'s declaration: {named}" if declared
                    else f"Input normalization set to the CellMap convention for {name}, which declares none: {named}")
        return name
    return None


def input_norm_set_by(model_configs):
    """The first of ``model_configs`` whose start sets the chain's input
    part as it stands, step for step (start_input_norm), as (its name,
    whether the model declares the steps); ``(None, None)`` when none does,
    or the chain has no input normalizers.

    GET /api/models/loaded answers them beside the chain, so the page can
    say where an input normalization a model's start set came from
    (set_declared_input_norm sets the steps as they are).
    """
    session = get_session()
    live = [dict(step) for step in session.pipeline_spec.input_norm]
    if not live:
        return None, None
    raw_dtype, raw_range = model_advice.raw_dtype_and_range(session.dataset_path)
    for model_config in model_configs:
        steps, declared = start_input_norm(model_config, raw_dtype, raw_range)
        if steps == live:
            return getattr(model_config, "name", None), declared
    return None, None


# The models' starts run in launch threads of their own: one at a time
# decides on an empty chain's postprocessing, so two models started together
# do not both set it.
_start_postprocess_lock = threading.Lock()


def set_needed_postprocess(job) -> bool:
    """Set the postprocessing ``job``'s model needs to show a good prediction
    as the chain's, as part of starting it, when the chain has none: the
    sigmoid its output needs when it is logits (portable/rank.py
    needed_postprocess, with no threshold; an affinity model's too).
    Returns whether it set any.

    services.launch.run_model calls this once the model's server has its
    host, before the model's layer is drawn: every start goes through it
    (the Models tab's Submit, Add a model, the Finetune tab's Use this
    model). Unlike the input normalization (set_declared_input_norm), which
    a model's config says, its output is known only from its server's probe
    (serving/client.fetch_model_info's ``output_class``), so this comes
    after its job starts. The chain is set through _set_chain_and_redraw,
    as PUT /api/pipeline sets it, so every layer is redrawn through it,
    this model's among them; and so it is not set when it would show
    another model being served wrong (a sigmoid on top of an output that
    has one already). A chain with postprocessing (a YAML's, the user's
    Apply, the ranking's) is left alone: the first model started into an
    empty chain that needs any gives it, and after that only Apply changes
    it. The session remembers which model's start it was
    (Session.start_postprocess, which postprocess_set_by reads).

    The browser's twin is web/src/backend/serving.ts, at serve, from the
    manifest's declared output.
    """
    session = get_session()
    name, host = getattr(job, "model_name", None), getattr(job, "host", None)
    with _start_postprocess_lock:
        if not host or session.pipeline_spec.postprocess:
            return False
        steps = rank.needed_postprocess(fetch_model_info(host).get("output_class"), [])
        if not steps:
            return False
        named = " → ".join(step["name"] for step in steps)
        wrong = [other.model_name for other in session.jobs if other is not job and other.host and rank.needed_postprocess(
            fetch_model_info(other.host).get("output_class"), steps) is not None]
        if wrong:
            logger.info(f"{name}'s output is logits, but {named} is not set: it would show {', '.join(wrong)} wrong")
            return False
        _set_chain_and_redraw(PipelineSpec(session.pipeline_spec.input_norm, steps), None)
        session.start_postprocess = {"model": name, "postprocess": steps}
        logger.info(f"Postprocessing set for {name}, whose output is logits: {named}")
        return True


def postprocess_set_by(model_names):
    """The model of ``model_names`` whose settings the chain's
    postprocessing is, step for step, as (its name, whether the ranking
    fitted them to the paint); ``(None, None)`` when it is none's, or the
    chain has none.

    The ranking's (Session.rank_fits: what Use this model, Fit to Paint and
    Use these settings set, a sigmoid and the threshold the paint is fitted
    best at), else a model's start's (Session.start_postprocess:
    set_needed_postprocess, the sigmoid its logits need). GET
    /api/models/loaded answers them beside the chain, as it answers where
    the input normalization came from (input_norm_set_by).
    """
    session = get_session()
    live = [dict(step) for step in session.pipeline_spec.postprocess]
    if not live:
        return None, None
    for name in model_names:
        fitted = (session.rank_fits.get(name) or {}).get("postprocess")
        if fitted is not None and rank.same_steps(fitted, live):
            return name, True
    started = session.start_postprocess
    if started and started["model"] in model_names and rank.same_steps(started["postprocess"], live):
        return started["model"], False
    return None, None


def _problems_with_models(postprocess):
    """Why a step of ``postprocess`` cannot run on a model being served (running,
    or starting), one line each.

    The chain is every prediction layer's, and a step that fails on a
    model's output fails on every chunk inside that model's server, where
    the page showed only an empty layer (CellposeMasksPostprocessor on a
    Cellpose model serving its probability).
    """
    from cellmap_flow.dashboard.services.launch import running_names

    # Only those served: a model stopped on the Models tab keeps its config.
    served = running_names()
    problems = []
    for step in postprocess:
        check = getattr(step, "problem_with", None)
        if check is None:
            continue
        for model_config in get_session().models_config or []:
            if getattr(model_config, "name", None) not in served:
                continue
            problem = check(model_config)
            if problem:
                problems.append(problem)
    return problems


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
    problems = _problems_with_models(built[1])
    if problems:
        return jsonify({"success": False, "error": "; ".join(problems)}), 400
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
    """Submit's route before PUT /api/pipeline.

    The body is ``{"input_norm", "postprocess"}``, either chain in the list
    or the older dict form. The answer is ``{"message", "received_data"}``,
    the body with the dashboard's address and the chain's digest added; the
    older answer's ``found_custom_normalizer`` and ``received_data.time``,
    which nothing read, are gone. An op it does not know is kept in the
    chain and skipped where it is built; a body without both chains, or a
    step its class refuses, is a 500.
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
