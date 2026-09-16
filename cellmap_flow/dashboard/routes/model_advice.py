"""Advice about a served model's normalization and postprocessing.

Nothing in cellmap-flow records what a model expects: metadata.json carries
geometry and provenance only, and the input/output handling is set by
convention plus whatever happens to be toggled in the UI. A wrong choice
degrades the result silently rather than erroring, which is a bad failure mode.

This endpoint combines two sources into a verdict the UI can show:

* the running server's output probe (``/__control__/output_probe``), which
  observes what activation the model's output already has, and
* the model's declared metadata (channel count, name, training framework).

It never changes anything -- callers decide whether to apply the suggestion.
"""

import json
import logging
import os

from flask import Blueprint, jsonify

from cellmap_flow.globals import g
from cellmap_flow.utils.output_probe import review_postprocess, suggest_input_norm
from cellmap_flow.utils.server_info import fetch_model_info

logger = logging.getLogger(__name__)

model_advice_bp = Blueprint("model_advice", __name__)

def _model_metadata(model_config) -> dict:
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


def _fetch_probe(host: str) -> dict:
    """Ask a running inference server what its output activation looks like."""
    return fetch_model_info(host)


@model_advice_bp.route("/api/model_advice", methods=["GET"])
def model_advice():
    """Per-model verdicts on the currently configured input/output handling."""
    configured_post = [
        p.to_dict().get("name") for p in (g.postprocess or []) if hasattr(p, "to_dict")
    ]
    configured_norm = [
        n.to_dict().get("name") for n in (g.input_norms or []) if hasattr(n, "to_dict")
    ]

    by_name = {}
    for cfg in g.models_config or []:
        try:
            by_name[getattr(cfg, "name", None)] = cfg
        except Exception:
            continue

    results = []
    for job in g.jobs or []:
        name = getattr(job, "model_name", None)
        meta = _model_metadata(by_name[name]) if name in by_name else {}
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
            "output_min": probe.get("output_min"),
            "output_max": probe.get("output_max"),
            "configured_postprocess": configured_post,
            "configured_input_norm": configured_norm,
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
                probe["output_class"],
                configured_post,
                out_channels=out_channels,
                model_name=meta.get("model_name") or name or "",
                channels_names=meta.get("channels_names"),
            )

        # Input side is a declared convention, not something we can observe.
        entry["input_norm_suggestion"] = suggest_input_norm(
            framework=meta.get("framework"),
            raw_dtype=meta.get("raw_dtype", "uint8"),
        )
        # A "low" confidence suggestion is just the dtype default, made without
        # knowing the training framework -- reporting a mismatch against it
        # would flag a correct hand-tuned config as wrong. Say "unknown"
        # instead, so callers can stay quiet rather than mislead.
        suggestion = entry["input_norm_suggestion"]
        entry["input_norm_matches"] = (
            None
            if suggestion["confidence"] == "low"
            else sorted(configured_norm) == sorted(suggestion["input_norm"].keys())
        )
        results.append(entry)

    return jsonify({"success": True, "models": results})
