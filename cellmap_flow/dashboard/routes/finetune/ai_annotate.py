"""AI-assisted annotation: a hosted model's first guess at one plane, for the user to review.

The user hovers a point in the viewer and presses Shift+G (or clicks
"Annotate at view centre"). A background job reads one 2D plane of raw EM
around the point, in the plane the user is looking at (XY, XZ or YZ), asks
the configured model to paint the target structure in a colour, turns the
image it returns into a mask and stages it beside the session's corrections
(``ai_annotate.staging``). Nothing is written to the annotation volume until
the user accepts the preview.

- GET ``/api/finetune/ai-annotate/config``: whether the feature is on (and
  if not, why and how to turn it on), the providers and models the server's
  config allows with where each sends the data, the day's calls against
  the limit, the organelle catalog with each one's editable prompt, and the
  session's settings and acknowledgements.
- POST ``/api/finetune/ai-annotate/settings``: the provider, model, target
  and prompt to use; ``acknowledge: true`` records that the user accepts
  that the selected provider's destination receives this dataset's images.
  Also binds Shift+G in the viewer.
- POST ``/api/finetune/ai-annotate/run``: start a job at a point (by default
  the view centre, in the layout's plane).
- GET ``/api/finetune/ai-annotate/status``: the job's stage, or its preview
  (input | model output | overlay) once it is ready.
- POST ``/api/finetune/ai-annotate/resend``: ask the model again about the
  same staged plane, with an edited prompt.
- POST ``/api/finetune/ai-annotate/accept``: write the staged mask into the
  annotation volume, over unannotated voxels only unless ``overwrite`` says
  otherwise. The existing Undo (``/api/finetune/view-labels/undo``) takes
  it back, as it takes back a seed.
- POST ``/api/finetune/ai-annotate/reject``: throw the staged mask away.

The feature is off unless the server's config file turns it on, and the
browser only ever picks a provider and a model that the file lists: it
never sends an endpoint or a credential. A run is refused until the session
has acknowledged the provider's destination for the dataset, and once the
day's call limit is reached. Error answers carry an ``AIAnnotateError``'s
short ``user_message``, never an SDK's own text, which may echo a request;
anything unexpected is "AI annotation failed; see the server log", with the
detail logged after redaction. Every request, staging, failure, resend and
decision is appended to the volume's audit log (``ai_annotate.audit``).

One job runs per dashboard session, and a staged result must be accepted or
rejected before the next run: two previews of overlapping planes would be
confusing to review and could be accepted out of order.
"""

import dataclasses
import logging
import threading
import traceback

import numpy as np
import zarr
from flask import jsonify, request

from cellmap_flow.ai_annotate import audit, config as ai_config, geometry, organelles, pipeline, prompts, staging, usage
from cellmap_flow.ai_annotate.backends import get_backend
from cellmap_flow.ai_annotate.errors import AIAnnotateError
from cellmap_flow.ai_annotate.secrets import redact
from cellmap_flow.dashboard.requests import (
    AIAnnotateDecision,
    AIAnnotateResend,
    AIAnnotateRun,
    AIAnnotateSettings,
    parse,
)
from cellmap_flow.dashboard.routes.finetune import view_labels
from cellmap_flow.dashboard.routes.finetune.blueprint import finetune_bp
from cellmap_flow.dashboard.routes.finetune.common import (
    INSTANCE_TARGETS,
    session_store,
    viewer_position_and_scales,
)
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.finetune.session import fill
from cellmap_flow.finetune.session.fill import paint_box
from cellmap_flow.io.metadata import nm_per_unit

logger = logging.getLogger(__name__)

# The neuroglancer action and the key bound to it ("shift+keyg" in
# neuroglancer's spelling; KEYBINDING is how the page shows it).
ACTION = "ai-annotate"
KEY = "shift+keyg"
KEYBINDING = "Shift+G"

# The key the job's messages use in neuroglancer's status bar.
STATUS_KEY = "ai_annotate"

GENERIC_FAILURE = "AI annotation failed; see the server log."

# The stages of a job, in order, as the page and the status bar name them.
STAGE_LABELS = {
    "fetching_crop": "Reading the EM plane...",
    "sending": "Waiting for the model...",
    "extracting_mask": "Turning the model's image into a mask...",
    "staging_preview": "Building the preview...",
}



class _Refused(Exception):
    """A request these routes will not act on: ``status``, a message and extra answer fields."""

    def __init__(self, message, status=409, **extra):
        super().__init__(message)
        self.status = status
        self.extra = extra


def _error(message, status, **extra):
    return jsonify({"success": False, "error": message, **extra}), status


def _answer_error(e):
    """The error answer for an exception from a route."""
    if isinstance(e, _Refused):
        return _error(str(e), e.status, **e.extra)
    if isinstance(e, AIAnnotateError):
        logger.warning(f"AI annotation refused ({e.category}): {e.user_message}")
        return _error(e.user_message, e.http_status, category=e.category)
    logger.error(f"AI annotation request failed: {redact(traceback.format_exc())}")
    return _error(GENERIC_FAILURE, 500)


def _state():
    """The session's AI-annotate state (``Session.ai_annotate``)."""
    return get_session().ai_annotate


def _start_thread(target, *args):
    """Run ``target(*args)`` in a daemon thread. Tests replace this to run it at once."""
    threading.Thread(target=target, args=args, daemon=True, name="ai-annotate").start()


def _load_config():
    """The config, or a 403 ``_Refused`` saying why the feature is off and how to turn it on."""
    config = ai_config.load_config()
    if config is None:
        raise _Refused(ai_config.disabled_reason(), 403, enabled=False)
    return config


def _destinations(config):
    """``{provider id: where it sends the data}``, as the page shows it."""
    return {p["id"]: p["destination"] for p in config.public()["providers"]}


def _dataset(volume=None):
    """The dataset a run reads: the annotation volume's, else the one the viewer shows."""
    if volume is not None and volume.get("dataset_path"):
        return str(volume["dataset_path"])
    dataset_path = get_session().dataset_path
    return str(dataset_path) if dataset_path else None


def _acknowledged(ai, config, dataset):
    """The provider ids whose current destination the session has acknowledged for ``dataset``.

    An acknowledgement is of the destination string the user was shown, so
    a config change that moves a provider (another location, say) needs it
    again.
    """
    if not dataset:
        return []
    seen = ai["acknowledged"].get(dataset, {})
    return [pid for pid, destination in _destinations(config).items() if seen.get(pid) == destination]


def _check_egress(ai, config, dataset, provider, model):
    """Refuse sending ``dataset``'s planes to ``provider``/``model`` unless all is allowed.

    The model first, so a provider taken out of the config is reported as
    that rather than as a missing acknowledgement; then the acknowledgement
    of the provider's current destination for this dataset, then the
    config's dataset prefixes. Every call that sends a plane (a run, a
    Shift+G, a resend) goes through here, so a config changed while a
    result waits is checked again before it is resent.
    """
    config.check_model(provider, model)
    if provider not in _acknowledged(ai, config, dataset):
        raise _Refused(
            "Acknowledge where the images are sent before the first run on this dataset.", 409,
            needs_acknowledgement=True, provider=provider, destination=_destinations(config).get(provider),
        )
    config.check_dataset(dataset)


def _profile(settings):
    return organelles.resolve_organelle_profile(settings.get("label_key") or settings.get("label_name") or "")


def _override(prompt, profile):
    """``prompt`` as build_recolor_prompt's override: None when blank or the catalog's own text.

    The page prefills the catalog's text (``prompts.editable_prompt``), so
    an unedited prompt is kept as None and the audit log shows no override.
    """
    if prompt is None or not prompt.strip() or prompt.strip() == prompts.editable_prompt(profile).strip():
        return None
    return prompt.strip()


# ---------------------------------------------------------------------------
# The viewer: Shift+G and status messages
# ---------------------------------------------------------------------------

def _say(viewer, message):
    """Show ``message`` in neuroglancer's status bar (None clears it); never raises."""
    if viewer is None:
        return
    try:
        with viewer.config_state.txn() as s:
            if message:
                s.status_messages[STATUS_KEY] = message
            else:
                s.status_messages.pop(STATUS_KEY, None)
    except Exception as e:
        logger.debug(f"Could not set the viewer's status message: {e}")


def _ensure_binding(ai):
    """Bind Shift+G to the AI-annotate action in the session's viewer, once per viewer.

    A new dataset makes a new viewer, so the viewer bound is remembered
    rather than a flag. neuroglancer keeps a set of handlers per action, so
    binding the one module-level handler again adds nothing.
    """
    viewer = get_session().viewer
    if viewer is None or ai["binding_registered_for"] is viewer:
        return
    viewer.actions.add(ACTION, _on_annotate_key)
    with viewer.config_state.txn() as cs:
        cs.input_event_bindings.viewer[KEY] = ACTION
    ai["binding_registered_for"] = viewer


def _on_annotate_key(action_state):
    """The Shift+G action: start a job at the voxel under the mouse.

    Runs on neuroglancer's event loop, which every chunk request also waits
    on, so it only reads what the action carries and hands the rest to a
    thread: the checks, the status messages and the job itself.
    """
    coordinates = action_state.mouse_voxel_coordinates
    _start_thread(
        _start_from_key,
        None if coordinates is None else [float(c) for c in coordinates],
        action_state.viewer_state,
    )


def _zyx_nm(coordinates, dimensions):
    """Viewer voxel ``coordinates`` as world nm, z, y, x.

    The viewer's dimensions give each axis's name and scale; the axes are
    put in z, y, x order by name, or taken as they come when they are not
    named so.
    """
    names = list(dimensions.names)
    # neuroglancer keeps scales in metres (16 nm is 1.6e-08 m), so the
    # product is rounded to a picometre, which a voxel boundary never needs.
    nm = [round(float(c) * float(scale) * nm_per_unit(unit), 3)
          for c, scale, unit in zip(coordinates, dimensions.scales, dimensions.units)]
    if all(axis in names for axis in geometry.AXIS_NAMES):
        return [nm[names.index(axis)] for axis in geometry.AXIS_NAMES]
    if len(nm) < 3:
        raise _Refused("The viewer has fewer than three spatial axes.", 400)
    return nm[:3]


def _start_from_key(coordinates, viewer_state):
    """Start a job for Shift+G; a refusal shows in the status bar and the status route."""
    viewer = get_session().viewer
    try:
        if coordinates is None:
            raise _Refused("Hover the mouse over the data, then press Shift+G.", 400)
        point_nm = _zyx_nm(coordinates, viewer_state.dimensions)
        _begin_run(point_nm, geometry.depth_axis_for_view(viewer_state))
    except Exception as e:
        if isinstance(e, _Refused):
            message = str(e)
        elif isinstance(e, AIAnnotateError):
            message = e.user_message
        else:
            logger.error(f"AI annotation could not start: {redact(traceback.format_exc())}")
            message = GENERIC_FAILURE
        _say(viewer, f"AI annotate: {message}")
        ai = _state()
        with ai["lock"]:
            # Shown by the status route too, unless a job is under way or
            # waiting for review: that one stays what the page shows.
            job = ai["job"]
            replaced = job if job is not None and job["status"] == "failed" else None
            if job is None or replaced is not None:
                ai["job"] = {**_idle_job(), "status": "failed", "error": message}
        if replaced is not None and replaced.get("corrections_dir"):
            # A resend that failed leaves the staged plane; nothing can
            # accept or reject it once it is replaced.
            _discard(replaced)


# ---------------------------------------------------------------------------
# The job
# ---------------------------------------------------------------------------

def _idle_job():
    """Every field the status route answers with, as they are with no job."""
    return {
        "annotate_id": None, "status": "idle", "stage": None, "error": None, "prompt": None,
        "provider": None, "model": None, "label_name": None, "depth_axis": None,
        "point_nm": None, "write_box": None, "mask_fraction": None, "preview": None,
    }


def _check_limit(config):
    """Refuse (429) a run once the day's calls are used up.

    The worker counts the call (usage.check_and_count) just before it is
    made; this is so a refusal comes as the answer to the click rather than
    as a failed job a moment later.
    """
    if usage.calls_today() >= config.daily_call_limit:
        raise AIAnnotateError(
            "limit",
            f"The daily limit of {config.daily_call_limit} AI annotation calls is reached; it resets at midnight.",
        )


def _begin_run(point_nm, depth_axis):
    """Check everything a run needs, then start its job; returns its annotate_id.

    The checks, in order: the feature is on, the session has settings, there
    is an annotation volume, the model is allowed, the provider's
    destination is acknowledged for the volume's dataset, the dataset is
    allowed (``_check_egress``), nothing is running, waiting for review or
    being written, and the day's limit is not reached.
    """
    config = _load_config()
    ai = _state()
    settings = ai["settings"]
    if settings is None:
        raise _Refused("Choose a provider, model and target for AI annotation first.", 400)
    volume_id, volume = session_store().session_volume()
    if volume is None:
        raise _Refused("No annotation volume to annotate. Create or resume one first.")
    dataset = _dataset(volume)
    _check_egress(ai, config, dataset, settings["provider"], settings["model"])
    _ensure_binding(ai)

    point_nm = [float(v) for v in point_nm]
    depth_axis = int(depth_axis)
    profile = _profile(settings)
    override = _override(settings.get("prompt"), profile)
    label_name = settings.get("label_name") or profile.name
    with ai["lock"]:
        current = ai["job"]
        if current is not None and current["status"] in ("running", "ready", "accepting"):
            raise _Refused({
                "running": "An AI annotation is still running.",
                "ready": "Accept or reject the AI annotation waiting for review first.",
                "accepting": "The accepted AI annotation is still being written.",
            }[current["status"]])
        _check_limit(config)
        stale = current if current is not None and current.get("corrections_dir") else None
        annotate_id = staging.new_annotate_id()
        job = {
            **_idle_job(),
            "annotate_id": annotate_id, "status": "running", "stage": "fetching_crop",
            "prompt": override or prompts.editable_prompt(profile), "provider": settings["provider"],
            "model": settings["model"], "label_name": label_name, "depth_axis": depth_axis,
            "point_nm": point_nm,
            # What the worker and the decisions need, not answered.
            "volume_id": volume_id, "corrections_dir": volume["corrections_dir"], "dataset": dataset,
            "profile": profile, "override": override, "plan": None, "request": None,
        }
        ai["job"] = job
    if stale is not None:
        # A job that failed after staging (a failed resend) leaves its files.
        _discard(stale)
    _say(get_session().viewer, f"AI annotate: {STAGE_LABELS['fetching_crop']}")
    _start_thread(_run_job, ai, job, config, volume)
    return annotate_id


def _set(ai, job, **fields):
    with ai["lock"]:
        job.update(fields)


def _stage(ai, job, viewer, stage):
    _set(ai, job, stage=stage)
    _say(viewer, f"AI annotate: {STAGE_LABELS[stage]}")


def _run_job(ai, job, config, volume):
    """The background job: read the plane, ask the model, stage the mask."""
    viewer = get_session().viewer
    try:
        audit.record(
            job["corrections_dir"], "requested", annotate_id=job["annotate_id"], volume_id=job["volume_id"],
            dataset=job["dataset"], provider=job["provider"], model=job["model"], label_name=job["label_name"],
            point_nm=job["point_nm"], depth_axis=job["depth_axis"], prompt_override=job["override"],
        )
        _stage(ai, job, viewer, "fetching_crop")
        shape = zarr.open_array(f"{volume['zarr_path']}/annotation/s0", mode="r").shape
        plan = geometry.plan_plane(volume, shape, job["point_nm"], job["depth_axis"], config.crop_size_px)
        _set(ai, job, plan=plan, write_box={"lo": [int(v) for v in plan.write_lo],
                                            "hi": [int(v) for v in plan.write_hi]})
        plane = pipeline.read_plane(volume, plan)
        segment_request = pipeline.build_request(plane, plan, job["profile"], job["override"])
        _set(ai, job, request=segment_request)
        _ask_and_stage(ai, job, viewer, config, segment_request)
    except Exception as e:
        _fail(ai, job, viewer, e)


def _ask_and_stage(ai, job, viewer, config, segment_request):
    """Ask the model about ``segment_request``, and stage what it answers as the job's preview."""
    _stage(ai, job, viewer, "sending")
    usage.check_and_count(config.daily_call_limit)
    backend = get_backend(config.provider(job["provider"]))
    result = backend.segment(segment_request, job["model"])
    _stage(ai, job, viewer, "extracting_mask")
    plan = job["plan"]
    mask_write = pipeline.mask_to_write_shape(result.mask, plan)
    _stage(ai, job, viewer, "staging_preview")
    meta = staging.stage(
        job["corrections_dir"], job["annotate_id"], plan=plan, request=segment_request, result=result,
        mask_write=mask_write, provider_id=job["provider"], model=job["model"], label_name=job["label_name"],
        volume_id=job["volume_id"],
    )
    preview = staging.preview(job["corrections_dir"], job["annotate_id"])
    mask_fraction = float(meta.get("mask_fraction", np.mean(mask_write)))
    _set(ai, job, status="ready", stage=None, error=None, mask_fraction=mask_fraction, preview=preview)
    _audit(
        job["corrections_dir"], "staged", annotate_id=job["annotate_id"], volume_id=job["volume_id"],
        provider=job["provider"], model=job["model"], mask_fraction=mask_fraction,
        write_box=job["write_box"], usage=getattr(result, "usage", None) or {},
    )
    _say(viewer, "AI annotate: ready for review in the Finetune tab.")


def _fail(ai, job, viewer, e):
    """Mark the job failed with a message fit for the page, and log the detail."""
    detail = redact(traceback.format_exc())
    if isinstance(e, AIAnnotateError):
        message, category = e.user_message, e.category
        logger.warning(f"AI annotation {job['annotate_id']} failed ({category}): {detail}")
    else:
        message, category = GENERIC_FAILURE, "internal"
        logger.error(f"AI annotation {job['annotate_id']} failed: {detail}")
    _set(ai, job, status="failed", stage=None, error=message, preview=None)
    _audit(job["corrections_dir"], "failed", annotate_id=job["annotate_id"],
           volume_id=job["volume_id"], category=category, error=message)
    _say(viewer, f"AI annotate: {message}")


def _audit(corrections_dir, event, **fields):
    """Record an event that has already happened; a log that cannot be written is only logged.

    The events before a call ("requested", "resent") are recorded with
    audit.record itself, so that a call is not made unless it is recorded.
    """
    try:
        audit.record(corrections_dir, event, **fields)
    except Exception as e:
        logger.warning(f"Could not record the AI annotation's {event!r} in the audit log: {e}")


def _discard(job):
    try:
        staging.discard(job["corrections_dir"], job["annotate_id"])
    except Exception as e:
        logger.warning(f"Could not discard AI annotation {job['annotate_id']}: {e}")


def _job_for(ai, annotate_id, *statuses):
    """The session's job when ``annotate_id`` is its id and its status is one of ``statuses``."""
    job = ai["job"]
    if job is None or job["annotate_id"] != annotate_id:
        raise _Refused("That AI annotation is not the current one; it may have been accepted or rejected.")
    if job["status"] not in statuses:
        raise _Refused({
            "running": "The AI annotation is still running.",
            "failed": "The AI annotation failed; resend it or reject it.",
            "ready": "The AI annotation is waiting for review.",
            "accepting": "The AI annotation is already being written.",
        }.get(job["status"], "The AI annotation cannot do that now."))
    return job


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@finetune_bp.route("/api/finetune/ai-annotate/config", methods=["GET"])
def ai_annotate_config():
    """What the page needs to show the AI-annotate panel; see the module docstring."""
    ai = _state()
    answer = {
        "success": True, "enabled": False, "providers": [], "default_provider": None,
        "daily_call_limit": None, "calls_today": None, "crop_size_px": None, "organelles": [],
        "keybinding": KEYBINDING, "settings": ai["settings"], "acknowledged": [],
    }
    try:
        config = ai_config.load_config()
    except AIAnnotateError as e:
        # A malformed file: the page says what is wrong with it.
        logger.warning(f"The AI-annotate config cannot be used: {e.user_message}")
        return jsonify({**answer, "reason": e.user_message})
    if config is None:
        return jsonify({**answer, "reason": ai_config.disabled_reason()})
    try:
        public = config.public()
        volume_id, volume = session_store().session_volume()
        answer.update(
            enabled=True,
            providers=public["providers"],
            default_provider=public["default_provider"],
            daily_call_limit=public["daily_call_limit"],
            crop_size_px=public["crop_size_px"],
            calls_today=usage.calls_today(),
            organelles=[{"key": p.key, "name": p.name, "prompt": prompts.editable_prompt(p)}
                        for p in organelles.ORGANELLES.values()],
            acknowledged=_acknowledged(ai, config, _dataset(volume)),
        )
        return jsonify(answer)
    except Exception as e:
        return _answer_error(e)


@finetune_bp.route("/api/finetune/ai-annotate/settings", methods=["POST"])
def ai_annotate_settings():
    """Set the provider, model, target and prompt; optionally acknowledge the provider's destination.

    JSON body: ``provider`` and ``model`` (ones the config allows),
    ``label_key`` (an organelle catalog key) and/or ``label_name`` (what
    to call the target, by default the catalog's name), ``prompt`` (the
    editable prompt; null or the catalog's own text uses the catalog's) and
    ``acknowledge``.
    """
    body, refused = parse(AIAnnotateSettings, request.get_json(silent=True))
    if refused:
        return refused
    try:
        config = _load_config()
        config.check_model(body.provider, body.model)
        if body.label_key and body.label_key not in organelles.ORGANELLES:
            raise _Refused(f"Unknown target {body.label_key!r}; pick one from the list.", 400)
        if not (body.label_key or body.label_name):
            raise _Refused("Name the structure to annotate (label_key or label_name).", 400)
        settings = {
            "provider": body.provider, "model": body.model, "label_key": body.label_key,
            "label_name": body.label_name, "prompt": None,
        }
        settings["prompt"] = _override(body.prompt, _profile(settings))
        ai = _state()
        _, volume = session_store().session_volume()
        dataset = _dataset(volume)
        if body.acknowledge:
            if not dataset:
                raise _Refused("Open a dataset before acknowledging where its images are sent.")
            current = _destinations(config)[body.provider]
            # The page sends the destination it showed: an acknowledgement is
            # of that, so a config changed since the page loaded is refused
            # rather than taken as agreed to.
            if body.destination != current:
                raise _Refused(
                    "Where this provider sends images has changed; check the new destination and tick again.",
                    409, needs_acknowledgement=True, provider=body.provider, destination=current,
                )
            ai["acknowledged"].setdefault(dataset, {})[body.provider] = current
            logger.info(f"AI annotate: images of {dataset} may be sent to {body.provider} ({current})")
        ai["settings"] = settings
        _ensure_binding(ai)
        return jsonify({"success": True, "settings": settings,
                        "acknowledged": _acknowledged(ai, config, dataset)})
    except Exception as e:
        return _answer_error(e)


@finetune_bp.route("/api/finetune/ai-annotate/run", methods=["POST"])
def ai_annotate_run():
    """Start a job: at ``point_nm`` (z, y, x) in the plane normal to ``depth_axis``.

    Both optional: the view centre, and the plane of the viewer's layout
    (XY in a multi-panel layout, where which panel is meant is not known).
    """
    body, refused = parse(AIAnnotateRun, request.get_json(silent=True) or {})
    if refused:
        return refused
    try:
        point_nm, depth_axis = body.point_nm, body.depth_axis
        if point_nm is None or depth_axis is None:
            viewer = get_session().viewer
            if viewer is None:
                raise _Refused("Open a dataset in the viewer first.", 400)
            state = viewer.state
            if depth_axis is None:
                depth_axis = geometry.depth_axis_for_view(state)
            if point_nm is None:
                position, _ = viewer_position_and_scales()
                if position is None:
                    raise _Refused("The viewer has no position.", 400)
                point_nm = _zyx_nm(position, state.dimensions)
        annotate_id = _begin_run(point_nm, depth_axis)
        return jsonify({"success": True, "annotate_id": annotate_id, "status": "running"})
    except Exception as e:
        return _answer_error(e)


@finetune_bp.route("/api/finetune/ai-annotate/status", methods=["GET"])
def ai_annotate_status():
    """The session's job: its status and stage, its error, or (``?preview=1``) its preview once ready.

    The preview is three PNGs of up to 1024 pixels a side, so it comes only
    when asked for: the page asks once when a result becomes ready, not on
    every poll while it waits for review. A result being written by Accept
    is reported as ready, its last state the page can show.
    """
    ai = _state()
    with ai["lock"]:
        job = dict(ai["job"] or _idle_job())
    if job["status"] == "accepting":
        job["status"] = "ready"
    depth_axis = job["depth_axis"]
    with_preview = request.args.get("preview") in ("1", "true")
    return jsonify({
        "success": True,
        **{key: job[key] for key in _idle_job()},
        "stage_label": STAGE_LABELS.get(job["stage"]),
        "plane": geometry.PLANE_NAMES[depth_axis] if depth_axis is not None else None,
        "preview": job["preview"] if job["status"] == "ready" and with_preview else None,
    })


@finetune_bp.route("/api/finetune/ai-annotate/resend", methods=["POST"])
def ai_annotate_resend():
    """Ask the model again about the staged plane, with ``prompt`` (the editable text).

    The plane is not read again: the same image goes with the new prompt,
    and the preview is replaced under the same annotate_id.
    """
    body, refused = parse(AIAnnotateResend, request.get_json(silent=True))
    if refused:
        return refused
    try:
        config = _load_config()
        ai = _state()
        with ai["lock"]:
            job = _job_for(ai, body.annotate_id, "ready", "failed")
            if job["request"] is None or job["plan"] is None:
                raise _Refused("That AI annotation never read its plane; run it again instead.")
            _check_egress(ai, config, job["dataset"], job["provider"], job["model"])
            _check_limit(config)
            override = _override(body.prompt, job["profile"])
            segment_request = dataclasses.replace(
                job["request"],
                prompt=prompts.build_recolor_prompt(job["profile"], job["plan"].resolution_nm, override),
            )
            job.update(status="running", stage="sending", error=None, preview=None, override=override,
                       prompt=override or prompts.editable_prompt(job["profile"]), request=segment_request)
        _start_thread(_resend_job, ai, job, config, segment_request)
        return jsonify({"success": True, "annotate_id": job["annotate_id"], "status": "running"})
    except Exception as e:
        return _answer_error(e)


def _resend_job(ai, job, config, segment_request):
    viewer = get_session().viewer
    try:
        audit.record(job["corrections_dir"], "resent", annotate_id=job["annotate_id"],
                     volume_id=job["volume_id"], provider=job["provider"], model=job["model"],
                     prompt_override=job["override"])
        _ask_and_stage(ai, job, viewer, config, segment_request)
    except Exception as e:
        _fail(ai, job, viewer, e)


def _planes_beside(state, volume_id, lo, hi, depth_axis):
    """The planes on either side of the box ``[lo, hi)`` along ``depth_axis``, as MinIO serves them.

    For ``pipeline.labels_for_box``; None for a side past the volume's edge.
    Read just before the write, so a stroke painted in them since the plane
    was staged counts.
    """
    _, _, arr = fill.open_served_labels(state, volume_id)
    planes = []
    for step in (-1, 1):
        depth = int(lo[depth_axis]) + step
        if not 0 <= depth < arr.shape[depth_axis]:
            planes.append(None)
            continue
        box = [slice(int(a), int(b)) for a, b in zip(lo, hi)]
        box[depth_axis] = slice(depth, depth + 1)
        planes.append(arr[tuple(box)])
    return planes


@finetune_bp.route("/api/finetune/ai-annotate/accept", methods=["POST"])
def ai_annotate_accept():
    """Write the staged mask into the annotation volume: each object an id of its own, the rest background.

    JSON body: ``annotate_id`` and ``overwrite``. Without ``overwrite`` only
    unannotated voxels are written, as a seed writes them; with it, the
    staged labels replace what the plane held. Either way Undo takes it back.
    """
    body, refused = parse(AIAnnotateDecision, request.get_json(silent=True))
    if refused:
        return refused
    ai = _state()
    try:
        state = get_session().minio_state
        if not state.get("ip") or not state.get("port"):
            raise _Refused("MinIO is not serving the annotation volume. Create or resume one first.")
        with ai["lock"]:
            job = _job_for(ai, body.annotate_id, "ready")
            # Marked while it is written, so a second click, a Run or a
            # Shift+G is refused rather than replacing it; set back to ready
            # if the write fails, for the user to try again or reject it.
            job["status"] = "accepting"
    except Exception as e:
        return _answer_error(e)
    try:
        volume_id = job["volume_id"]
        volume = get_session().annotation_volumes.get(volume_id)
        if volume is None:
            raise _Refused(f"The annotation volume {volume_id} is no longer open.")
        _, mask_write, _ = staging.load(job["corrections_dir"], job["annotate_id"])
        plan = job["plan"]
        lo, hi = np.asarray(plan.write_lo), np.asarray(plan.write_hi)
        instance_target = view_labels.model_output(volume.get("model_name"))[0] in INSTANCE_TARGETS

        beside = _planes_beside(state, volume_id, lo, hi, plan.depth_axis)

        def labels_for(existing):
            # As a seed: an instance target on a uint16/uint32 volume counts
            # its ids up, a uint8 volume reuses free ones; an object
            # continuing one in the plane beside keeps its id.
            labels = pipeline.labels_for_box(
                mask_write, existing, plan.depth_axis,
                count_up=instance_target and existing.dtype.itemsize > 1, beside=beside,
            )
            if body.overwrite:
                # Overwrite lets the model's objects replace what is under
                # them, but its background only fills unlabelled voxels: an
                # object already painted that the model missed is kept, not
                # erased to background.
                labels[(labels == 1) & (existing != 0)] = 0
            return labels

        n_foreground, n_background, n_overwritten = paint_box(
            state, volume_id, lo, hi, labels_for, volume.get("zarr_path"),
            undo=view_labels._undo_stack(volume_id), overwrite=body.overwrite,
        )
        changed = bool(n_foreground or n_background or n_overwritten)
        layer_refreshed = view_labels._after_write(volume_id, "AI-annotated") if changed else False
    except Exception as e:
        _set(ai, job, status="ready")
        if isinstance(e, FileNotFoundError):
            logger.warning(f"Could not accept AI annotation {job['annotate_id']}: {e}")
            return _error("The staged annotation or the served volume is missing; reject it and run again.", 409)
        if isinstance(e, ValueError):
            # Our own messages (labels_for_box, fill's ids running out), which
            # say what to do: the seed route answers them the same way.
            logger.warning(f"Could not accept AI annotation {job['annotate_id']}: {e}")
            return _error(str(e), 409)
        return _answer_error(e)
    with ai["lock"]:
        if ai["job"] is job:
            ai["job"] = None
    logger.info(
        f"Accepted AI annotation {job['annotate_id']} into {lo.tolist()}..{hi.tolist()} of {volume_id}: "
        f"{n_foreground} foreground, {n_background} background, {n_overwritten} overwritten"
    )
    _discard(job)
    _audit(
        job["corrections_dir"], "accepted", annotate_id=job["annotate_id"], volume_id=volume_id,
        overwrite=body.overwrite, filled_foreground=n_foreground, filled_background=n_background,
        overwritten=n_overwritten, write_box=job["write_box"],
    )
    _say(get_session().viewer, None)
    return jsonify({
        "success": True,
        "filled_foreground": n_foreground,
        "filled_background": n_background,
        "overwritten": n_overwritten,
        "reload_viewer": changed,
        "layer_refreshed": layer_refreshed,
        "can_undo": bool(view_labels._undo_stack(volume_id)),
    })


@finetune_bp.route("/api/finetune/ai-annotate/reject", methods=["POST"])
def ai_annotate_reject():
    """Throw the staged mask away. JSON body: ``annotate_id``."""
    body, refused = parse(AIAnnotateDecision, request.get_json(silent=True))
    if refused:
        return refused
    ai = _state()
    try:
        with ai["lock"]:
            job = _job_for(ai, body.annotate_id, "ready", "failed")
            ai["job"] = None
        _discard(job)
        _audit(job["corrections_dir"], "rejected", annotate_id=job["annotate_id"],
                     volume_id=job["volume_id"])
        _say(get_session().viewer, None)
        return jsonify({"success": True})
    except Exception as e:
        return _answer_error(e)
