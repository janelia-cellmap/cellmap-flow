"""Advice about a served model's normalization and postprocessing.

Nothing in cellmap-flow records what a model expects: metadata.json carries
geometry and provenance only, and the input/output handling is set by
convention plus whatever happens to be toggled in the UI. A wrong choice
degrades the result silently rather than erroring, which is a bad failure mode.

This endpoint combines two sources into a verdict the UI can show:

* the running server's model info (``/__control__/model_info``), which says
  what activation the model's output already has: read off its network's
  graph (the op that ends it), or else observed by its output probe on
  noise, a guess, and
* the model's declared metadata (channel count, name, training framework).

It never changes anything -- callers decide whether to apply the suggestion.
Nor does it undo what the Finetune tab's ranking set: the input
normalization the ranking fitted to the paint for a model
(Session.rank_fits) counts as matching, not as one to suggest the
declared one back over, and an affinity model's chain into a segmentation
is offered cut at the threshold it fitted.
"""

import json
import logging
import os

import numpy as np

from flask import Blueprint, jsonify

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.portable.rank import same_steps
from cellmap_flow.serving.probe import activation_evidence, input_norm_differences, review_postprocess, suggest_input_norm
from cellmap_flow.serving.client import fetch_model_info

logger = logging.getLogger(__name__)

model_advice_bp = Blueprint("model_advice", __name__)

def model_metadata(model_config) -> dict:
    """Best-effort metadata for a model config, from whatever source has it."""
    meta = {}
    try:
        meta.update(model_config.to_dict() or {})
    except Exception as e:
        logger.debug(f"to_dict failed for model config: {e}")

    # Folder-backed models keep their metadata.json on disk next to the weights;
    # it is not surfaced through to_dict().
    folder = meta.get("folder_path")
    if folder:
        path = os.path.join(folder, "metadata.json")
        try:
            with open(path) as f:
                for k, v in (json.load(f) or {}).items():
                    meta.setdefault(k, v)
        except Exception as e:
            logger.debug(f"Could not read {path}: {e}")
    return meta


def _declared_input_norm(model_config, data_range):
    """The input normalization a model declares itself
    (``ModelConfig.declared_input_norm``, for raw data spanning
    ``data_range``), in suggest_input_norm's ``declared`` form (``{name:
    params}``, in order), or None; None too for a job whose model is not
    among the session's.

    An onnx model's manifest declares the steps the browser feeds it through,
    and a cellmap-models export's metadata.json the range the model reads.
    """
    if model_config is None:
        return None
    norm = {}
    for step in model_config.declared_input_norm(data_range=data_range) or []:
        if isinstance(step, dict) and step.get("name"):
            norm[step["name"]] = {k: v for k, v in step.items() if k != "name"}
    return norm or None


# Reading the dataset costs a metadata fetch (and, for float data, a sample),
# and it cannot change while the dashboard runs.
_raw_range_cache = {}


def raw_dtype_and_range(dataset_path):
    """The dtype and value range of the raw data, for scaling it to the model.

    Integer data has an exact range from its dtype. Float data does not -- the
    dtype permits anything -- so sample percentiles instead, the same way the
    raw layer's auto-contrast does. Returns (dtype_name, (lo, hi)); either part
    may be None when it could not be determined.
    """
    if not dataset_path:
        return None, None
    if dataset_path in _raw_range_cache:
        return _raw_range_cache[dataset_path]

    dtype_name, value_range = None, None
    try:
        from cellmap_flow.io.metadata import open_level
        from cellmap_flow.viewer.raw import _auto_contrast_range

        # The range of the data as stored, which is what the normalizers
        # being proposed have to map from.
        dtype = open_level(dataset_path).dtype
        dtype_name = dtype.name
        if dtype.kind in "ui":
            info = np.iinfo(dtype)
            value_range = (float(info.min), float(info.max))
        else:
            value_range = _auto_contrast_range([dataset_path], normalize=False)
    except Exception as e:
        logger.debug(f"Could not read raw dtype/range for {dataset_path}: {e}")

    _raw_range_cache[dataset_path] = (dtype_name, value_range)
    return dtype_name, value_range


def input_norm_suggestion(model_config, meta, raw_dtype, raw_range):
    """What ``model_config``'s input normalization should be on raw data of
    ``raw_dtype`` spanning ``raw_range``: suggest_input_norm's suggestion,
    and the declaration it holds the chain to (_declared_input_norm), or
    None when the model declares none. ``meta`` is model_metadata's for it
    ({} for a job whose model is not among the session's).

    The page shows the suggestion, and a model's start sets it on a chain
    with no input normalization when it is the model's declaration or the
    convention for its kind of model (routes.pipeline.set_declared_input_norm).
    """
    if getattr(model_config, "normalizes_input", False):
        # No suggestion, so the page shows none: [-1, 1] was suggested
        # for a Cellpose model, which normalizes each slice itself.
        return {"input_norm": None, "order": [], "reason": "The model normalizes its input itself",
                "confidence": "high"}, None
    declared = _declared_input_norm(model_config, raw_range)
    suggestion = suggest_input_norm(
        framework=meta.get("framework"),
        raw_dtype=raw_dtype,
        declared=declared,
        source=meta.get("repo") or meta.get("folder_path"),
        data_range=raw_range,
    )
    return suggestion, declared


def _fetch_probe(host: str) -> dict:
    """Ask a running inference server what its output activation looks like."""
    return fetch_model_info(host)


def _launching() -> bool:
    """Whether a model's launch has not returned yet (services/launch.py):
    its job may be listed already while its start is still to set the
    chain's postprocessing (routes.pipeline.set_needed_postprocess)."""
    from cellmap_flow.dashboard.services.launch import _launching, _launching_lock

    with _launching_lock:
        return bool(_launching)


@model_advice_bp.route("/api/model_advice", methods=["GET"])
def model_advice():
    """Per-model verdicts on the currently configured input/output handling.

    ``{"success": true, "models": [...], "starting", "digest"}``: an entry
    per served model; ``starting``, whether a model's start is still on its
    way (_launching), and ``digest``, the chain's (PipelineSpec.digest). The
    page asks again while ``starting``, and reads the chain again when the
    digest moves: a model's start sets the postprocessing its output needs
    once its server answers, after the Submit that started it.
    """
    session = get_session()
    configured_post = [
        p.to_dict().get("name") for p in (session.postprocess or []) if hasattr(p, "to_dict")
    ]
    configured_steps = [
        n.to_dict() for n in (session.input_norms or []) if hasattr(n, "to_dict")
    ]
    configured_norm = [step.get("name") for step in configured_steps]

    raw_dtype, raw_range = raw_dtype_and_range(session.dataset_path)

    by_name = {}
    for cfg in session.models_config or []:
        try:
            by_name[getattr(cfg, "name", None)] = cfg
        except Exception:
            continue

    results = []
    for job in session.jobs or []:
        name = getattr(job, "model_name", None)
        meta = model_metadata(by_name[name]) if name in by_name else {}
        probe = _fetch_probe(getattr(job, "host", None))

        # Only folder-backed models carry metadata.json; a script model
        # declares nothing, so without this fallback out_channels is always
        # None and the affinity heuristic can never fire.
        out_channels = meta.get("out_channels") or probe.get("output_channels")

        entry = {
            "model": name,
            "probe_available": bool(probe.get("available")),
            "out_channels": out_channels,
            "output_class": probe.get("output_class"),
            # What says it: "graph" (output_op, the op that ends it) or "probe".
            "output_class_from": probe.get("output_class_from"),
            "output_op": probe.get("output_op"),
            "output_min": probe.get("output_min"),
            "output_max": probe.get("output_max"),
            "configured_postprocess": configured_post,
            "configured_input_norm": configured_norm,
            # How the server reads the data (model_info): one of the last two
            # is a list when the data has no level at input_voxel_size.
            "input_voxel_size": probe.get("input_voxel_size"),
            "input_resampled_from": probe.get("input_resampled_from"),
            "input_relabelled_from": probe.get("input_relabelled_from"),
        }
        if not probe.get("available"):
            entry["postprocess_review"] = {
                "level": "unknown",
                "message": (
                    "No output probe available for this model "
                    f"({probe.get('reason', 'unknown reason')}). Restart the "
                    "inference server to collect one."
                ),
                "suggest": [],
            }
        else:
            entry["postprocess_review"] = review_postprocess(
                probe.get("output_class"),
                configured_post,
                out_channels=out_channels,
                model_name=meta.get("model_name") or name or "",
                channels_names=meta.get("channels_names"),
                evidence=activation_evidence(probe.get("output_class_from"), probe.get("output_op"),
                                             (probe.get("output_min"), probe.get("output_max"))),
                # The threshold the ranking fitted to the paint for it, which
                # an affinity chain's watershed joins neighbours above.
                threshold=(session.rank_fits.get(name) or {}).get("threshold"),
            )

        # Input side is a declared convention, not something we can observe.
        suggestion, declared = input_norm_suggestion(by_name.get(name), meta, raw_dtype, raw_range)
        entry["input_norm_suggestion"] = suggestion
        # A "low" confidence suggestion is just the dtype default, made without
        # knowing the training framework -- reporting a mismatch against it
        # would flag a correct hand-tuned config as wrong. Say "unknown"
        # instead, so callers can stay quiet rather than mislead.
        # Order-sensitive on purpose: LambdaNormalizer("x*2-1") before the
        # rescale computes something quite different from after it, so the
        # same two normalizers in the wrong sequence is not a match. A
        # declaration counts down to its values too: the declared steps with
        # 2*x for a manifest's 2*x-1 feed the model something else, and the
        # page said that "looks right". A convention's values are not held
        # to the letter (its x*2-1 is anyone's 2*x-1).
        # The steps the ranking fitted to the paint for this model, which Use
        # this model, Fit to Paint or Use these settings set (a MinMax
        # stretched to the raw under the paint, say), match as they are:
        # suggested the declared ones back, a second Apply undid the ranking.
        fitted = (session.rank_fits.get(name) or {}).get("input_norm")
        entry["input_norm_fitted"] = bool(fitted) and same_steps(fitted, [dict(s) for s in session.pipeline_spec.input_norm])
        entry["input_norm_differences"] = [] if entry["input_norm_fitted"] else input_norm_differences(configured_steps, declared)
        entry["input_norm_matches"] = entry["input_norm_fitted"] or (
            list(configured_norm) == list(suggestion["order"])
            and not entry["input_norm_differences"]
        )
        results.append(entry)

    return jsonify({"success": True, "models": results, "starting": _launching(),
                    "digest": session.pipeline_spec.digest()})
