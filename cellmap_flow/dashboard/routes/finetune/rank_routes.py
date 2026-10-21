"""Which model to start from, in server mode: the browser's routes
(web/src/backend/routes/rank.ts), answered by the dashboard, so the Finetune
tab's ranking (static/js/dashboard/finetune/rank.js) works the same in both.

GET /api/finetune/rank/candidates lists the models to rank: the session's
(running or not), the catalog's and Hugging Face's, with the session's
volume and the model it is for; only the running ones start ticked, as a
model not running is started on the cluster for its score. POST
/api/finetune/rank/score {name, entry?} scores one on the session volume's
paint (finetune/rank.py says how, as the browser does): a running model
through its server, any other started for the score alone (``entry``, the
candidate's, says how to build it) and stopped after, so a ranking leaves
the Models tab and the cluster as they were. Where the raw under the paint
is narrow, the same server is asked again through the dashboard's
normalizer stretched to the raw's window: a server runs the chain each
request names. POST /api/finetune/rank/use {name, entry?, window?} makes
the one picked the model to finetune: running (as the Models tab's Add runs
it), the window its score chose (if any) made the dashboard's
normalization, a volume made for it with the session volume's paint
carried onto its grid (session/volume.py carry_paint) and its paint layer
in the old one's place; the old volume stays.
"""

import logging
import time

import numpy as np
from flask import jsonify, request
from scipy.special import logit

from cellmap_flow.dashboard.finetune_utils import sync_annotation_volume_from_minio
from cellmap_flow.dashboard.routes.finetune.blueprint import finetune_bp
from cellmap_flow.dashboard.routes.finetune.common import autodetect_output_type, find_model_config, session_store
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.finetune import rank
from cellmap_flow.finetune.session.volume import volume_corner_nm
from cellmap_flow.pipeline_spec import PipelineSpec
from cellmap_flow.serving.client import fetch_model_info, running_job_host

logger = logging.getLogger(__name__)

# The most raw voxels the raw's window is taken from (every few of them).
RAW_SAMPLE = 1_000_000


def _refused(message, status=400, **extra):
    return jsonify({"success": False, "error": message, **extra}), status


@finetune_bp.route("/api/finetune/rank/candidates", methods=["GET"])
def rank_candidates():
    session = get_session()
    volume_id, volume = session_store().session_volume()
    hosts = {job.model_name: job.host for job in session.jobs if getattr(job, "host", None)}
    candidates, seen = [], set()

    def add(name, **fields):
        if not name or name in seen:
            return
        seen.add(name)
        served = name in hosts
        # A model not running is started on the cluster for its score
        # (``cluster``), so only the running ones start ticked.
        candidates.append({
            "name": name, "task": None, "channels": [], "voxel_size_nm": [], "dims": None, "finetuned": False,
            "served": served, "ticked": served, "cluster": True, "source": name,
            "volume_model": bool(volume) and volume.get("model_name") == name, **fields,
        })

    from cellmap_flow.dashboard.services.launch import _sanitize_job_name
    from cellmap_flow.models.configs.finetune import FinetuneModelConfig

    for config in session.models_config:
        name = getattr(config, "name", None)
        info = fetch_model_info(hosts[name]) if name in hosts else {}
        add(name, task=getattr(type(config), "cli_name", None), finetuned=isinstance(config, FinetuneModelConfig),
            channels=list(info.get("channels") or []), voxel_size_nm=list(info.get("output_voxel_size") or []))
    for group, models in (session.model_catalog or {}).items():
        for name, folder in models.items():
            add(name, task=group, source=folder, entry={"type": "cellmap", "folder_path": folder, "name": name})
    try:
        from cellmap_flow.models.hf_catalog import list_huggingface_models

        for repo, meta in list_huggingface_models().items():
            # Named as the Models tab's Submit names it (run_hf_model).
            name = _sanitize_job_name(repo.split("/")[-1])
            add(name, task="huggingface", source=repo, channels=list(meta.get("channels_names") or []),
                voxel_size_nm=list(meta.get("output_voxel_size") or []),
                entry={"type": "huggingface", "repo": repo, "name": name})
    except Exception as e:
        logger.warning(f"Which model to start from lists no Hugging Face models: {e}")
    return jsonify({
        "success": True,
        "volume": {"id": volume_id, "model_name": volume.get("model_name"), "painted_chunks": len(_painted_chunks(volume))} if volume else None,
        "candidates": candidates,
    })


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
        host, job = _serving(name, config)
    except Exception as e:
        return _refused(f"{name} could not be started for its score: {e}")
    try:
        info = fetch_model_info(host)
        output_type, offsets = autodetect_output_type(config, None, None)
        offsets = None if output_type != "affinities" else _offsets(offsets)
        boxes = [box for box in map(lambda index: _painted_box(volume, index), rank.spread(chunks, rank.RANK_BLOCKS)) if box]
        base, outside = _score(host, name, volume, boxes, info, offsets)
        score, window = base, None
        # The raw stretched to its own range, when it is narrow (finetune/rank.py).
        norm = [dict(s) for s in session.pipeline_spec.input_norm]
        raw = base and _raw_window(volume, boxes)
        stretched = raw and rank.windowed(norm, *raw)
        if base and stretched:
            alt, _ = _score(host, name, volume, boxes, info, offsets, stretched)
            if alt and alt["matches"] >= base["matches"] + rank.WINDOW_MARGIN:
                score, window = alt, list(raw)
        if not score:
            return _refused(f"The paint gives {name} nothing to score: "
                            + ("an affinity is between two painted neighbours, so paint strokes, not single voxels." if offsets
                               else "only the edges between object and background strokes are painted, and those are not counted."))
        notes = [
            *([f"scored under {rank.RANK_BLOCKS} of the {len(chunks)} painted chunks, spread over them"] if len(chunks) > rank.RANK_BLOCKS else []),
            *([f"{outside} of the painted chunks scored lie past its output, and were not scored"] if outside else []),
            *(["started for its score and stopped after"] if job is not None else []),
        ]
        seconds = time.time() - started
        logger.info(f"Ranked {name} on {score['scored_voxels']} painted voxels of {volume_id}: matches {score['matches']:.0%} "
                    f"at cut-off {score['cutoff']} ({score['matches_at_zero']:.0%} at 0)"
                    + (f", the raw stretched from {window[0]:.1f}-{window[1]:.1f} ({base['matches']:.0%} as normalized)" if window else "")
                    + f", {seconds:.1f} s")
        return jsonify({
            "success": True, "name": name, **score, "seconds": seconds, "window": window,
            **({"matches_without_window": base["matches"]} if window else {}),
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
    from cellmap_flow.dashboard.routes.finetune.annotation_core import make_volume
    from cellmap_flow.dashboard.routes.finetune.overlay import add_annotation_layer
    from cellmap_flow.dashboard.finetune_utils import ensure_minio_serving
    from cellmap_flow.dashboard.routes.pipeline import _set_chain_and_redraw
    from cellmap_flow.dashboard.services.launch import running_names
    from cellmap_flow.finetune.session.volume import carry_paint
    from cellmap_flow.models.geometry_cache import resolve_model_geometry

    data = request.get_json(silent=True) or {}
    name = str(data.get("name") or "").strip()
    if not name:
        return _refused("name is required")
    session = get_session()
    from_id, from_volume = session_store().session_volume()
    if not from_volume:
        return _refused("No annotation volume in this session to carry paint from.", needs_volume=True)
    if from_volume.get("model_name") == name:
        return _refused(f"{from_id} is {name}'s volume already: finetune it as it is.")
    try:
        # The raw's window the score chose, made the dashboard's
        # normalization: the raw is the dataset's, so every model reads it so.
        w = data.get("window")
        norm = [dict(s) for s in session.pipeline_spec.input_norm]
        stretched = rank.windowed(norm, float(w[0]), float(w[1])) if isinstance(w, list) and len(w) == 2 else None
        if stretched:
            _set_chain_and_redraw(PipelineSpec(stretched, session.pipeline_spec.postprocess), request.host_url)
        config = _model_config(name, data.get("entry"))
        if name not in running_names():
            _start_for_good(name, config)
        sync_annotation_volume_from_minio(from_id)
        to_id, zarr_path, minio_url, _ = make_volume(name, config, resolve_model_geometry(name, config), from_volume["corrections_dir"])
        carried = carry_paint(from_volume, session.annotation_volumes[to_id])
        ensure_minio_serving(zarr_path, to_id, output_base_dir=from_volume["corrections_dir"])
        layer = f"annotation_{to_id}"
        if session.viewer is not None:
            with session.viewer.txn() as s:
                old = f"annotation_{from_id}"
                if old in s.layers:
                    del s.layers[old]
            add_annotation_layer(session.viewer, layer, f"{minio_url}/annotation")
        logger.info(f"Carried {from_id}'s paint to {to_id} for {name}: {carried['painted']} voxels from {carried['source']}"
                    + (f"; {carried['conflicts']} left unpainted where object and background met in one voxel" if carried["conflicts"] else ""))
        return jsonify({
            "success": True, "name": name, "volume_id": to_id, "from_volume": from_id, "layer_name": layer,
            "painted_voxels": carried["painted"], "source_voxels": carried["source"], "conflicts": carried["conflicts"],
            "chunks": carried["chunks"], "window": [float(w[0]), float(w[1])] if stretched else None,
        })
    except Exception as e:
        logger.error(f"Could not make {name} the model to finetune: {e}", exc_info=True)
        return _refused(f"Could not make {name} the model to finetune: {e}")


def _model_config(name, entry):
    """The session's config of ``name``, or one built from the candidate's ``entry``."""
    config = find_model_config(name)
    if config is not None:
        return config
    if not isinstance(entry, dict) or not entry.get("type"):
        raise ValueError(f"{name} is not among the session's models, and no entry says how to build it")
    from cellmap_flow.models.registry import build_model

    return build_model(dict(entry), name)


def _serving(name, config):
    """``(host, job)``: the running server of ``name``, or one started now for
    a score (its job, to stop after), as the dashboard starts any model."""
    from cellmap_flow.dashboard.services.launch import running_names
    from cellmap_flow.jobs.launch import start_hosts

    host = running_job_host(name)
    if host:
        return host, None
    if name in running_names():
        raise RuntimeError("it is starting on the Models tab; score it once its server is up")
    from cellmap_flow.serving.launch import server_command

    session = get_session()
    job = start_hosts(server_command(config, session.dataset_path, session.resample), job_name=name,
                      queue=session.queue, charge_group=session.charge_group, wait_for_host=True)
    return job.host, job


def _start_for_good(name, config):
    """Start ``name`` as the Models tab's Add does: among the session's models and jobs, with its layer."""
    from cellmap_flow.dashboard.services.launch import _launch, drop_config, run_model_config

    session = get_session()
    drop_config(name)
    session.models_config.append(config)
    _launch(name, run_model_config, config, session.layer_blob(), daemon=True)


def _offsets(offsets):
    import json

    return json.loads(offsets) if isinstance(offsets, str) else offsets


def _painted_chunks(volume):
    """The indices of the volume's painted chunks on disk (synced from MinIO), in index order."""
    import os

    from cellmap_flow.io.geometry import list_populated_chunks

    try:
        return list_populated_chunks(os.path.join(volume["zarr_path"], "annotation", "s0"))
    except OSError:
        return []


def _painted_box(volume, index):
    """``(lo, hi, labels)``: the box of chunk ``index``'s painted voxels, in annotation voxels, and their labels; None when none is painted."""
    import zarr

    array = zarr.open(volume["zarr_path"], mode="r")["annotation/s0"]
    lo = [i * c for i, c in zip(index, array.chunks)]
    chunk = array[tuple(slice(l, l + c) for l, c in zip(lo, array.chunks))]
    where = np.nonzero(chunk)
    if not where[0].size:
        return None
    box_lo = [int(w.min()) for w in where]
    box_hi = [int(w.max()) + 1 for w in where]
    labels = chunk[tuple(slice(a, b) for a, b in zip(box_lo, box_hi))]
    return np.array([l + a for l, a in zip(lo, box_lo)]), np.array([l + b for l, b in zip(lo, box_hi)]), labels


def _score(host, name, volume, boxes, info, offsets, input_norm=None):
    """``(score, outside)``: the model's score on ``boxes`` (finetune/rank.py
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
    from cellmap_flow.io.geometry import Box

    try:
        idi = _raw(volume, tuple(volume["input_voxel_size"]))
    except Exception as e:
        logger.debug(f"No raw window: {e}")
        return None
    voxel = np.asarray(volume["output_voxel_size"], dtype=float)
    corner = volume_corner_nm(volume.get("dataset_offset_nm"), voxel)
    raw_voxel = np.asarray(idi.voxel_size, dtype=float)[-3:]
    raw_corner = np.asarray(idi.offset, dtype=float)[-3:]
    parts = []
    for lo, hi, _ in boxes:
        begin = np.floor((corner + lo * voxel - raw_corner) / raw_voxel + 1e-6).astype(int)
        end = np.ceil((corner + hi * voxel - raw_corner) / raw_voxel - 1e-6).astype(int)
        parts.append(np.asarray(idi.read_box(Box(tuple(int(v) for v in begin), tuple(int(v) for v in np.maximum(end - begin, 1))))).ravel())
    values = np.concatenate(parts) if parts else np.zeros(0)
    if not values.size:
        return None
    values = values[:: max(1, values.size // RAW_SAMPLE)].astype(np.float64)
    lo, hi = np.percentile(values, [1, 99])
    return float(lo), float(hi)
