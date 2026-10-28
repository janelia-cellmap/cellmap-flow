"""Which model to start from, in server mode: the browser's routes
(web/src/backend/routes/rank.ts), answered by the dashboard, so the Finetune
tab's ranking (static/js/dashboard/finetune/rank.js) works the same in both.

GET /api/finetune/rank/candidates lists the models to rank as the Models tab
offers them, each with the ``group`` of the tab's it is listed under, in the
tab's order: the session's ("Running", running or not; not a Cellpose
model, whose flows the ranking does not score), each catalog group's
(models.yaml's "cellmap", "saalfeldlab"), cellmap's on Hugging Face
("Hugging Face") and the BioImage Model Zoo's ("BioImage Zoo", with its
``em`` tag, which the others are), from the lists the tab last cached
(``hf_cached`` and ``zoo_cached`` say whether there were any: none is
fetched here, so a listing never waits on the Hub or bioimage.io; the tab's
Refresh fetches them), then the ones given here by reference ("Added"),
with the session's volume and the model it is for. Each not running says
how to build it (``entry``, the one the Models tab starts it from); only the
running ones and the added start ticked, as a model not running is started
on the cluster for its score. A zoo model whose voxel size nothing says
(its server would refuse it) has no entry, and ``error`` and ``needs`` say
so. POST /api/finetune/rank/add {ref, name?, voxel_size?} adds a model by
reference, resolved as the Models tab's Add a model resolves it
(routes/model_resolve.py resolve_request), and answers its candidate,
starting nothing (a Cellpose model is refused); the session keeps it among
the candidates (Session.rank_added). POST
/api/finetune/rank/score {name, entry?, queue?} scores one on the session volume's
paint (portable/rank.py says how; the browser runs the same Python): a running model
through its server, any other started for the score alone (``entry``, the
candidate's, says how to build it; a zoo model is served from its own
environment, as the Models tab serves it; ``queue``, the Training tab's GPU
queues ticked, says where, else the Models tab's) and stopped after, so a ranking leaves
the Models tab and the cluster as they were. The model is read through
its own input normalization (what its start sets: its declaration, or the
CellMap convention) and through the session's when that is another, and
the better kept: a server runs the chain each request names. Where the raw
under the paint is narrow, the same server is asked again through that
normalization stretched to the raw's window. The answer is the ranking's
row (portable/rank.py score_row: the score kept, its threshold, the raw's
range it read and the steps it was read through, ``input_norm``, with what
every channel matched, ``channel_matches``, and what the other
normalizations tried did, ``input_norm_others``: why this one), and the
postprocessing its layer needs to show right instead of the session's,
``postprocess`` (portable/rank.py needed_postprocess: the sigmoid added or
taken away, and a binary model's threshold; an affinity model's watershed
needs its affinities graded), or null when the session's shows it right.
POST /api/finetune/rank/use {name, entry?, input_norm?, postprocess?,
select_channel?, queue?} makes the one picked the model to finetune: ``input_norm``
and ``postprocess`` (the row's) made the dashboard's chain first, and
``select_channel`` (the row's channel_index) the channel its layer shows
(Session.shown_channels), then the model running (as the Models tab's Add
runs it, on ``queue`` as for a score), a volume made for it with the session volume's paint carried onto
its grid (session/volume.py carry_paint) and its paint layer in the old
one's place. The old volume is retired (finetune_utils
retire_annotation_volume): out of the session, kept on disk in its
corrections dir's ``superseded/``, so the session has the one volume, the
new one, which its manifest trains on. The carry is
annotation_core.carry_onto_model, which training's submit carries through
too, for a model whose grid is not the volume's; the model's sizes come from
its server once its start is over (``_served_geometry``, which waits for a
start still falling back through the GPU queues). POST /api/finetune/rank/input-norm
{input_norm, postprocess?} makes the row's steps the dashboard's chain
alone, with no model started and no volume made: Fit to paint's, for the
model already seeding the paint.
"""

import logging
import time

import numpy as np
from flask import jsonify, request
from scipy.special import logit

from cellmap_flow.dashboard.finetune_utils import sync_annotation_volume_from_minio
from cellmap_flow.dashboard.routes.finetune.blueprint import finetune_bp
from cellmap_flow.dashboard.routes.finetune.channel import channel_refusal
from cellmap_flow.dashboard.routes.finetune.common import (
    autodetect_output_type, declared_output_type, find_model_config, session_store,
)
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.portable import rank
from cellmap_flow.finetune.session.volume import volume_corner_nm
from cellmap_flow.jobs.queues import queue_list
from cellmap_flow.jobs.site import current_site
from cellmap_flow.pipeline_spec import PipelineSpec
from cellmap_flow.serving.client import fetch_model_info, running_job_host

logger = logging.getLogger(__name__)

# The most raw voxels the raw's window is taken from (every few of them).
RAW_SAMPLE = 1_000_000

# Where a score's input normalization came from (portable/rank.py
# norm_tries's ``from``), as the log names it.
NORM_FROM = {
    "declared": "its declared input normalization",
    "convention": "the CellMap convention's input normalization",
    "assumed": "the assumed [-1, 1] input normalization",
    "session": "the session's input normalization",
    "none": "no input normalization",
}

# The Models tab's sections a candidate is listed under (its ``group``), in
# the tab's order, with the catalog's groups (models.yaml) after RUNNING.
RUNNING, HUGGING_FACE, ZOO, ADDED = "Running", "Hugging Face", "BioImage Zoo", "Added"

# The target of a Cellpose model (declared_output_type), which is not ranked:
# the ranking scores object and background, or affinities, and Cellpose
# predicts flows.
FLOWS = "flows"

# What resolve asks for that rank/add's ``voxel_size`` gives (models.resolve:
# a zoo or Cellpose model's voxel_size, a fly or onnx model's input one).
VOXEL_NEEDS = {"voxel_size", "input_voxel_size"}


def _dims(input_shape):
    """2 for a model that reads one slice (its input shape's z is 1), 3 for
    one that reads more, None when the shape is not known: a candidate's
    ``dims``, which the ranking's filter takes ("2d"), as the browser's
    (web/src/finetune/modes.ts readsOneSlice). The cellmap exports' metadata
    says ``spatial_dims`` 3 for their 2D models too (setup-53-2d's input is
    1 x 556 x 556), so the shape is what tells."""
    try:
        return 2 if int(input_shape[0]) == 1 else 3
    except (TypeError, ValueError, IndexError):
        return None


def _folder_input_shape(folder):
    """A cellmap export folder's ``input_shape`` (metadata.json), or None."""
    import json
    import os

    try:
        with open(os.path.join(folder, "metadata.json")) as f:
            return json.load(f).get("input_shape")
    except (OSError, ValueError):
        return None


def _candidate(name, group, served, volume, **fields):
    """A model to rank as the listing gives it, listed under ``group``:
    ``fields`` over the defaults for what nothing has said of it (no
    channels, voxel size or dims; EM; not finetuned). A model not running
    is started on the cluster for its score (``cluster``), so by default
    only a running one (``served``) starts ticked. ``volume`` is the
    session's, which ``volume_model`` says whether it is this model's."""
    return {
        "name": name, "group": group, "task": None, "channels": [], "voxel_size_nm": [], "dims": None, "em": True,
        "finetuned": False, "served": served, "ticked": served, "cluster": True, "source": name,
        "volume_model": bool(volume) and volume.get("model_name") == name, **fields,
    }


def _refused(message, status=400, **extra):
    return jsonify({"success": False, "error": message, **extra}), status


@finetune_bp.route("/api/finetune/rank/candidates", methods=["GET"])
def rank_candidates():
    from cellmap_flow.dashboard.services.launch import _sanitize_job_name, bioimage_model_entry
    from cellmap_flow.models.bioimage_catalog import cached_bioimage_models
    from cellmap_flow.models.configs.finetune import FinetuneModelConfig
    from cellmap_flow.models.hf_catalog import cached_huggingface_models

    session = get_session()
    volume_id, volume = session_store().session_volume()
    hosts = {job.model_name: job.host for job in session.jobs if getattr(job, "host", None)}
    candidates = {}

    def add(name, group, **fields):
        # A model listed twice is kept where it was first listed.
        if name and name not in candidates:
            candidates[name] = _candidate(name, group, name in hosts, volume, **fields)

    for config in session.models_config:
        if declared_output_type(config) == FLOWS:
            continue
        name = getattr(config, "name", None)
        info = fetch_model_info(hosts[name]) if name in hosts else {}
        # The server's read shape (nm) over its input voxel size: its input in voxels.
        read, voxel = info.get("read_shape"), info.get("input_voxel_size")
        shape = [r / v for r, v in zip(read, voxel)] if read and voxel else None
        add(name, RUNNING, task=getattr(type(config), "cli_name", None),
            finetuned=isinstance(config, FinetuneModelConfig), channels=list(info.get("channels") or []),
            voxel_size_nm=list(info.get("output_voxel_size") or []), dims=_dims(shape))
    # Before the lists, so one added by hand keeps its group and its tick
    # when a list has it too.
    for name, fields in session.rank_added.items():
        add(name, ADDED, ticked=True, **fields)
    for group, models in (session.model_catalog or {}).items():
        for name, folder in models.items():
            add(name, group, task=group, source=folder, dims=_dims(_folder_input_shape(folder)),
                entry={"type": "cellmap", "folder_path": folder, "name": name})
    hf = cached_huggingface_models()
    for repo, meta in (hf or {}).items():
        # Named as the Models tab's Submit names it (run_hf_model).
        name = _sanitize_job_name(repo.split("/")[-1])
        add(name, HUGGING_FACE, task="huggingface", source=repo, channels=list(meta.get("channels_names") or []),
            voxel_size_nm=list(meta.get("output_voxel_size") or []), dims=_dims(meta.get("input_shape")),
            entry={"type": "huggingface", "repo": repo, "name": name})
    zoo = cached_bioimage_models()
    for model in (zoo or {}).get("models", []):
        # Built as the Models tab's Submit builds it with its voxel size
        # blank, from the cached list alone.
        entry, _, refusal = bioimage_model_entry(model, online=False)
        add(entry["name"], ZOO, task="bioimage", source=model["key"], em=bool(model.get("em")),
            dims={"2d": 2, "3d": 3}.get(model.get("dims")),
            # The model as the Models tab's zoo list has it, which the
            # ranking's zoo row is made from (static/js/shared/model-list.js
            # zooRow), shown and searched as that list's.
            zoo=model,
            **({"error": f"{refusal}: to rank it, add it with one (bioimageio:{model['key']} and its voxel size)",
                "needs": ["voxel_size"]} if refusal else {"entry": entry}))
    order = [RUNNING, *(session.model_catalog or {}), HUGGING_FACE, ZOO, ADDED]
    return jsonify({
        "success": True,
        "volume": {"id": volume_id, "model_name": volume.get("model_name"), "painted_chunks": len(_painted_chunks(volume))} if volume else None,
        "candidates": sorted(candidates.values(), key=lambda c: order.index(c["group"])),
        "hf_cached": hf is not None, "zoo_cached": zoo is not None,
    })


@finetune_bp.route("/api/finetune/rank/add", methods=["POST"])
def rank_add():
    """A model for the ranking by reference, ``{ref, name?, voxel_size?}``
    (a ResolveModel), resolved as the Models tab's Add a model resolves it
    (routes/model_resolve.py resolve_request), starting nothing.

    Answers ``{"success": true, "candidate": ...}``, the candidate as the
    listing gives it (``group`` "Added", ticked, its ``entry`` the resolved
    one) with the resolve's ``how``, ``env`` and ``notes``, and keeps it in
    the session (Session.rank_added), so every listing after includes it.
    A reference that resolves to nothing is the resolve's 400. One whose
    model needs more than the reference says is a 400 whose ``needs`` lists
    what (the resolve's): a voxel size, which the same request with
    ``voxel_size`` gives; anything else only the Models tab's Add a model
    asks for, and the model it starts there is listed here as running. A
    Cellpose model is a 400 too: it predicts flows, which are not ranked."""
    from cellmap_flow.dashboard.routes.model_resolve import resolve_request
    from cellmap_flow.models.registry import model_type

    data = request.get_json(silent=True)
    resolved, error = resolve_request(data)
    if error:
        return error
    if getattr(model_type(resolved.type), "finetune_output_type", None) == FLOWS:
        return _refused(f"{resolved.name}, {resolved.how}, predicts flows, which the ranking does not score: "
                        "it scores a model's object and background, or its affinities, on the paint.")
    if resolved.needs:
        needs = ", ".join(resolved.needs)
        remedy = ("give its voxel size (nm, as 8 or 40,4,4) and add it again" if set(resolved.needs) <= VOXEL_NEEDS
                  else "add it on the Models tab (Add a model), which asks for them; started there, it is listed here")
        return _refused(f"{resolved.name}, {resolved.how}, needs {needs}: {remedy}.", needs=resolved.needs)
    fields = {"task": resolved.type, "source": data["ref"].strip(),
              "finetuned": resolved.type == "finetune", "entry": resolved.entry()}
    get_session().rank_added[resolved.name] = fields
    _, volume = session_store().session_volume()
    candidate = _candidate(resolved.name, ADDED, bool(running_job_host(resolved.name)), volume, ticked=True, **fields)
    logger.info(f"Added {resolved.name} to the models to rank: {resolved.how}")
    return jsonify({"success": True, "candidate": candidate, "how": resolved.how, "env": resolved.env,
                    "notes": resolved.notes})


@finetune_bp.route("/api/finetune/rank/score", methods=["POST"])
def rank_score():
    data = request.get_json(silent=True) or {}
    name = str(data.get("name") or "").strip()
    if not name:
        return _refused("name is required")
    session = get_session()
    if not session.dataset_path:
        return _refused("No dataset is open: open one first; the volume is painted on it.")
    volume_id, volume = session_store().session_volume()
    if not volume:
        return _refused("No annotation volume in this session: create one (Create Volume) or resume one, and paint on it, first.", needs_volume=True)
    sync_annotation_volume_from_minio(volume_id)
    chunks = _painted_chunks(volume)
    if not chunks:
        return _refused(f"Nothing is painted in {volume_id} yet: paint background (1) and objects (2 and up) first.")
    started = time.time()
    try:
        config = _model_config(name, data.get("entry"))
        host, job = _serving(name, config, queue_list(data.get("queue")))
    except Exception as e:
        return _refused(f"{name} could not be started for its score: {e}")
    try:
        info = fetch_model_info(host)
        output_type, offsets = autodetect_output_type(config, None, None)
        offsets = None if output_type != "affinities" else _offsets(offsets)
        boxes = [box for box in map(lambda index: _painted_box(volume, index), rank.spread(chunks, rank.RANK_BLOCKS)) if box]
        # Through the model's own input normalization and the session's, the
        # better kept (portable/rank.py norm_tries, best_try).
        tries = _norm_tries(config, session)
        scored = [_score(host, name, volume, boxes, info, offsets, t["input_norm"]) for t in tries]
        k = rank.best_try([s for s, _ in scored]) or 0
        (base, outside), norm, norm_from = scored[k], tries[k]["input_norm"], tries[k]["from"]
        # What the tries not kept matched, which the row carries to say why
        # this one.
        others = [{"from": t["from"], "matches": s["matches"]}
                  for j, (t, (s, _)) in enumerate(zip(tries, scored)) if j != k and s]
        # The raw stretched to its own range, when it is narrow, and the
        # row from the two scores (portable/rank.py).
        raw = base and _raw_window(volume, boxes)
        stretched = raw and rank.windowed(norm, *raw)
        alt = _score(host, name, volume, boxes, info, offsets, stretched)[0] if base and stretched else None
        score = rank.score_row(base, alt, norm, raw, norm_from, others)
        if not score:
            return _refused(f"The paint gives {name} nothing to score: "
                            + ("an affinity is between two painted neighbours, so paint strokes, not single voxels." if offsets
                               else "only the edges between object and background strokes are painted, and those are not counted."))
        notes = [
            *([f"scored under {rank.RANK_BLOCKS} of the {len(chunks)} painted chunks, spread over them"] if len(chunks) > rank.RANK_BLOCKS else []),
            *([f"{outside} of the painted chunks scored lie past its output, and were not scored"] if outside else []),
            *(["started for its score and stopped after"] if job is not None else []),
        ]
        # What its layer needs to show a good prediction, as the session's
        # chain stands (portable/rank.py needed_postprocess): the sigmoid its
        # output needs and, for a binary model, the threshold fitted to the
        # paint, which Use this model, Fit to Paint and Use these settings set
        # beside the input normalization. An affinity model gets the sigmoid
        # alone: the watershed that makes its objects (AffinityPostprocessor,
        # the model advice's optional chain) joins neighbours by how far their
        # affinity is above its bias, so it needs graded affinities, not the
        # 0s and 1s of a threshold, which it read as 0-255 and divided by 255:
        # no edge reached the bias and no object came out. Its fitted
        # threshold is still the row's, Seed from Prediction's.
        postprocess = rank.needed_postprocess(
            info.get("output_class"), [dict(s) for s in session.pipeline_spec.postprocess],
            None if offsets else score["threshold"])
        # What the row's buttons set from it: the model advice counts its
        # input normalization as matching, rather than suggest the declared
        # one back over it (Session.rank_fits).
        session.rank_fits[name] = {"input_norm": score["input_norm"], "postprocess": postprocess}
        seconds = time.time() - started
        window = score["window"]
        logger.info(f"Ranked {name} on {score['scored_voxels']} painted voxels of {volume_id}: matches {score['matches']:.0%} "
                    f"at cut-off {score['cutoff']} ({score['matches_at_zero']:.0%} at 0), read through {NORM_FROM[norm_from]}"
                    + (f", the raw stretched from {window[0]:.1f}-{window[1]:.1f} ({base['matches']:.0%} unstretched)" if window else "")
                    + f", {seconds:.1f} s")
        return jsonify({
            "success": True, "name": name, **score, "postprocess": postprocess, "seconds": seconds,
            **({"note": "; ".join(notes)} if notes else {}),
        })
    except Exception as e:
        logger.error(f"Could not score {name}: {e}", exc_info=True)
        return _refused(f"{name} could not be scored: {e}")
    finally:
        if job is not None:
            from cellmap_flow.dashboard.services.launch import _stop_jobs

            _stop_jobs([job], drop_layers=False)


@finetune_bp.route("/api/finetune/rank/use", methods=["POST"])
def rank_use():
    from cellmap_flow.dashboard.routes.finetune.annotation_core import carry_onto_model
    from cellmap_flow.dashboard.services.launch import running_names

    data = request.get_json(silent=True) or {}
    name = str(data.get("name") or "").strip()
    if not name:
        return _refused("name is required")
    channel = data.get("select_channel")
    refused = channel_refusal(channel)
    if refused:
        return refused
    session = get_session()
    from_id, from_volume = session_store().session_volume()
    if not from_volume:
        return _refused("No annotation volume in this session to carry paint from.", needs_volume=True)
    if from_volume.get("model_name") == name:
        return _refused(f"{from_id} is {name}'s volume already: finetune it as it is.")
    try:
        # Before the model starts, so it is served through them.
        input_norm, postprocess = _install_chain(data.get("input_norm"), data.get("postprocess"))
        config = _model_config(name, data.get("entry"))
        if channel is not None:
            # Before the model starts too, so the layer its start builds
            # (services/launch.py, on Session.shown_channel) opens on the
            # channel the paint matched: it opened on Neuroglancer's middle
            # channel, and only moved once the page's pick arrived.
            session.shown_channels[name] = channel
        if name not in running_names():
            _start_for_good(name, config, queue_list(data.get("queue")))
        # One volume a session: the one its manifest now trains on. The old
        # is kept, out of everything's way (annotation_core.carry_onto_model,
        # which training's submit carries through too).
        carried = carry_onto_model(name, config, _served_geometry(name, config), from_id, from_volume)
        return jsonify({
            "success": True, "name": name, "volume_id": carried["volume_id"], "from_volume": from_id,
            "layer_name": carried["layer_name"], "painted_voxels": carried["painted"], "source_voxels": carried["source"],
            "conflicts": carried["conflicts"], "chunks": carried["chunks"], "input_norm": input_norm, "postprocess": postprocess,
        })
    except Exception as e:
        logger.error(f"Could not make {name} the model to finetune: {e}", exc_info=True)
        return _refused(f"Could not make {name} the model to finetune: {e}")


@finetune_bp.route("/api/finetune/rank/input-norm", methods=["POST"])
def rank_input_norm():
    """Fit to paint's: the input normalization a score was read through and
    the postprocessing its layer needs ({input_norm, postprocess?}, the
    score answer's) made the dashboard's chain, as Use this model makes it,
    without starting a model or making a volume. Answers the session's
    chain as it now is, ``input_norm`` and ``postprocess``; refused (400)
    for steps that are not a list of ops of their kind the dashboard has,
    which leaves the chain as it was."""
    data = request.get_json(silent=True) or {}
    try:
        steps, post = _install_chain(data.get("input_norm"), data.get("postprocess"), input_norm_required=True)
    except ValueError as e:
        return _refused(f"The input normalization and postprocessing were left as they are: {e}")
    except Exception as e:
        logger.error(f"Could not set the input normalization and postprocessing: {e}", exc_info=True)
        return _refused(f"Could not set the input normalization and postprocessing: {e}")
    if steps is not None:
        logger.info(f"Input normalization set to fit the paint: {' → '.join(s['name'] for s in steps)}")
    if post is not None:
        logger.info(f"Postprocessing set to show the model right: {' → '.join(s['name'] for s in post) or 'none'}")
    spec = get_session().pipeline_spec
    return jsonify({"success": True, "input_norm": [dict(s) for s in spec.input_norm],
                    "postprocess": [dict(s) for s in spec.postprocess]})


def _install_chain(input_norm, postprocess, input_norm_required=False):
    """``input_norm`` (normalizer step dicts, a score's ``input_norm``) and
    ``postprocess`` (postprocessor step dicts, a score's ``postprocess``)
    made the session's chain, each part where it is given and is not the
    session's already (rank.same_steps), in one chain set, and the viewer
    redrawn through it (routes/pipeline.py's setter, as PUT /api/pipeline
    sets it): the raw is the dataset's, so every model reads it through the
    one input normalization, and every prediction layer shows through the
    one postprocessing. Returns ``(input_norm, postprocess)``, each the
    steps set, or None for a part left as it was: not given (None, and for
    ``input_norm`` not ``input_norm_required``) or the session's already.
    Raises ValueError for a part that is not a list of steps of ops of its
    kind the dashboard has, or that one's class refuses, with nothing
    changed."""
    from cellmap_flow.dashboard.routes.pipeline import _set_chain_and_redraw, _unknown_op

    live = get_session().pipeline_spec
    parts = []
    for steps, field, kind, current, required in (
            (input_norm, "input_norm", "normalizer", live.input_norm, input_norm_required),
            (postprocess, "postprocess", "postprocessor", live.postprocess, False)):
        if steps is None and not required:
            parts.append(None)
            continue
        if not isinstance(steps, list) or not all(isinstance(s, dict) for s in steps):
            raise ValueError(f"{field} is a list of {kind} steps")
        unknown = _unknown_op(kind, [s.get("name") for s in steps])
        if unknown:
            raise ValueError(unknown)
        parts.append(None if rank.same_steps(steps, [dict(s) for s in current]) else [dict(s) for s in steps])
    input_norm, postprocess = parts
    if input_norm is None and postprocess is None:
        return None, None
    spec = PipelineSpec(live.input_norm if input_norm is None else input_norm,
                        live.postprocess if postprocess is None else postprocess)
    try:
        _set_chain_and_redraw(spec, request.host_url)
    except TypeError as e:
        raise ValueError(str(e)) from e
    return input_norm, postprocess


def _norm_tries(config, session):
    """The input normalizations ``config``'s model is scored through
    (portable/rank.py norm_tries): its own and the session's.

    Its own is what its start sets (routes/pipeline.py start_input_norm: its
    declaration, or its kind's convention), else what the model advice
    assumes for a model nothing is known of ([-1, 1], "assumed"), which a
    start does not set but a score may: the score says whether it fits.
    fly_organelles' /nrs copies declare nothing and are not under the CellMap
    models' folder, and were read through no normalization at all."""
    from cellmap_flow.dashboard.routes import model_advice
    from cellmap_flow.dashboard.routes.pipeline import start_input_norm

    raw_dtype, raw_range = model_advice.raw_dtype_and_range(session.dataset_path)
    own, declared = start_input_norm(config, raw_dtype, raw_range)
    own_from = "declared" if declared else "convention"
    if not own:
        suggestion, _ = model_advice.input_norm_suggestion(config, model_advice.model_metadata(config), raw_dtype, raw_range)
        own = [{"name": name, **suggestion["input_norm"][name]} for name in suggestion["order"]]
        own_from = "assumed"
    return rank.norm_tries(own, own_from, [dict(s) for s in session.pipeline_spec.input_norm])


def _model_config(name, entry):
    """The session's config of ``name``, or one built from the candidate's ``entry``."""
    config = find_model_config(name)
    if config is not None:
        return config
    if not isinstance(entry, dict) or not entry.get("type"):
        raise ValueError(f"{name} is not among the session's models, and no entry says how to build it")
    from cellmap_flow.models.registry import build_model

    return build_model(dict(entry), name)


def _serving(name, config, queue):
    """``(host, job)``: the running server of ``name``, or one started now for
    a score (its job, to stop after), as the dashboard starts any model, on
    ``queue``, the Training tab's GPU queues ticked (the Models tab's when
    none are given)."""
    from cellmap_flow.dashboard.services.launch import running_names
    from cellmap_flow.serving.launch import launch

    host = running_job_host(name)
    if host:
        return host, None
    if name in running_names():
        raise RuntimeError("it is starting on the Models tab; score it once its server is up")
    session = get_session()
    job = launch(config, session.dataset_path, queue=queue or session.queue, charge_group=session.charge_group,
                 resample=session.resample)
    return job.host, job


# How long Use this model waits for a model's server to answer with its
# sizes once its start is over (its job has a host). A server is usually up
# 10 to 60 s after its job is submitted.
SERVER_WAIT_SECONDS = 180

# The longest Use this model waits in all: while the model's start is still
# going (its job pending, which jobs/launch.py start_hosts gives each GPU
# queue PENDING_FALLBACK_SECONDS before it falls back to the next) as long
# as every queue's fallback takes, then SERVER_WAIT_SECONDS. It waited 180 s
# whatever the start was doing, the same 180 s gpu_h100 got before its
# fallback to gpu_h200, and gave up at the second the job moved on.
START_WAIT_SECONDS = current_site().pending_fallback_seconds * len(current_site().gpu_queues) + SERVER_WAIT_SECONDS


def _served_geometry(name, config, timeout=START_WAIT_SECONDS):
    """``name``'s geometry, from its server or the geometry cache
    (geometry_cache.known_geometry), waiting for the server Use this model
    just started; never from the model built in the dashboard.

    It waits while the model's start is going (services/launch.py
    ``launching``: its launch thread waits for the job's host, through every
    queue's fallback), then SERVER_WAIT_SECONDS more for its server to
    answer, ``timeout`` seconds at most in all. A start that ended without a
    server (no job of the model's running) ends the wait at once, with why
    it failed (``start_error``).

    resolve_model_geometry builds the model here when neither knows it, and
    for a model whose metadata names two blocks (jrc_mus-livers' 216 and 288,
    which only running it tells apart) that loaded its TorchScript on the
    dashboard's CPU and ran it at 288^3: the dashboard died there
    ("Segmentation fault"), and the page had "Failed to fetch". Raises
    RuntimeError when the server has not answered in time, or will not."""
    from cellmap_flow.dashboard.services import launch
    from cellmap_flow.models import geometry_cache

    started = time.time()
    ceiling = started + timeout
    deadline = min(started + SERVER_WAIT_SECONDS, ceiling)
    while True:
        geometry = geometry_cache.known_geometry(name, config)
        if geometry is not None:
            return geometry
        now = time.time()
        if launch.launching(name):
            # Still starting: queued, falling back, or loading its model.
            deadline = min(now + SERVER_WAIT_SECONDS, ceiling)
        elif name not in launch.running_names():
            why = launch.start_error(name)
            raise RuntimeError(f"{name}'s server did not start" + (f": {why}" if why else "")
                               + "; start it again on the Models tab, then press Use this model again")
        if now >= deadline:
            raise RuntimeError(f"{name}'s server did not answer with its sizes within {round((now - started) / 60)} minutes "
                               "(its job may be waiting for a GPU); it keeps starting: press Use this model "
                               "again once it is running")
        time.sleep(3)


def _start_for_good(name, config, queue):
    """Start ``name`` as the Models tab's Add does: among the session's models
    and jobs, with its layer, on ``queue`` (as for _serving)."""
    from cellmap_flow.dashboard.services.launch import _launch, drop_config, run_model

    session = get_session()
    drop_config(name)
    session.models_config.append(config)
    _launch(name, run_model, config, session.layer_blob(), queue, daemon=True)


def _offsets(offsets):
    import json

    return json.loads(offsets) if isinstance(offsets, str) else offsets


def _painted_chunks(volume):
    """The indices of the volume's painted chunks on disk (synced from MinIO), in index order; none without a volume directory."""
    import os

    import worldarray as wa

    return wa.list_chunk_keys(os.path.join(volume["zarr_path"], "annotation", "s0"), separator=".", ndim=3)


def _painted_box(volume, index):
    """``(lo, hi, labels)``: the box of chunk ``index``'s painted voxels, in annotation voxels, and their labels; None when none is painted."""
    import os

    import worldarray as wa

    from cellmap_flow.finetune.session.manifest import read_chunk

    array = wa.open(os.path.join(volume["zarr_path"], "annotation", "s0"), units=None)
    lo = [i * c for i, c in zip(index, array.meta.chunk_shape)]
    chunk = read_chunk(array, index)
    where = np.nonzero(chunk)
    if not where[0].size:
        return None
    box_lo = [int(w.min()) for w in where]
    box_hi = [int(w.max()) + 1 for w in where]
    labels = chunk[tuple(slice(a, b) for a, b in zip(box_lo, box_hi))]
    return np.array([l + a for l, a in zip(lo, box_lo)]), np.array([l + b for l, b in zip(lo, box_hi)]), labels


def _score(host, name, volume, boxes, info, offsets, input_norm=None):
    """``(score, outside)``: the model's score on ``boxes`` (portable/rank.py
    rank_score), read through ``input_norm`` (the dashboard's when None),
    and how many boxes lie past what its server covers."""
    import torch

    from cellmap_flow.dashboard.routes.finetune.view_labels import _Refused, probability, read_prediction
    from cellmap_flow.finetune.target_transforms import AffinityTargetTransform, BroadcastBinaryTargetTransform

    tallies, outside = None, 0
    channels = list(info.get("channels") or [])
    for lo, hi, labels in boxes:
        try:
            prediction, plo, phi = read_prediction(host, name, volume, lo.copy(), hi.copy(), info, input_norm=input_norm)
        except _Refused:
            outside += 1
            continue
        labels = labels[tuple(slice(a - l, b - l) for a, b, l in zip(plo, phi, lo))]
        logits = logit(np.clip(probability(prediction, info.get("output_class")), 1e-6, 1 - 1e-6))
        n = logits.shape[0]
        annotation = torch.from_numpy(labels.astype(np.int64))[None, None]
        transform = AffinityTargetTransform(offsets, num_channels=n) if offsets else BroadcastBinaryTargetTransform(n)
        target, mask = (np.broadcast_to(t[0].numpy(), logits.shape) for t in transform(annotation))
        tallies = tallies or rank.empty_tallies(n)
        rank.tally(tallies, target, mask, logits, None if offsets else rank.paint_edges(labels))
    return (rank.rank_score(tallies, channels, bool(offsets)) if tallies else None), outside


def _raw_window(volume, boxes):
    """The 1st and 99th percentiles of the raw under ``boxes`` (every few
    voxels, at most RAW_SAMPLE), read as stored at the volume's input voxel
    size, or None when nothing could be read."""
    from cellmap_flow.ai_annotate.pipeline import _raw

    try:
        raw = _raw(volume, tuple(volume["input_voxel_size"]))
    except Exception as e:
        logger.debug(f"No raw window: {e}")
        return None
    voxel = np.asarray(volume["output_voxel_size"], dtype=float)
    corner = volume_corner_nm(volume.get("dataset_offset_nm"), voxel)
    raw_voxel = np.asarray(raw.array.voxel_size, dtype=float)[-3:]
    raw_corner = np.asarray(raw.array.corner, dtype=float)[-3:]
    parts = []
    for lo, hi, _ in boxes:
        begin = np.floor((corner + lo * voxel - raw_corner) / raw_voxel + 1e-6).astype(int)
        end = np.ceil((corner + hi * voxel - raw_corner) / raw_voxel - 1e-6).astype(int)
        parts.append(np.asarray(raw.read_index(begin, np.maximum(end, begin + 1))).ravel())
    values = np.concatenate(parts) if parts else np.zeros(0)
    if not values.size:
        return None
    values = values[:: max(1, values.size // RAW_SAMPLE)].astype(np.float64)
    lo, hi = np.percentile(values, [1, 99])
    return float(lo), float(hi)
