import logging
import time

from flask import Blueprint, render_template

from cellmap_flow.norm.input_normalize import get_input_normalizers
from cellmap_flow.post.postprocessors import get_postprocessors_list
from cellmap_flow.models.model_merger import get_model_mergers_list
from cellmap_flow.globals import g

logger = logging.getLogger(__name__)

pipeline_builder_bp = Blueprint("pipeline_builder", __name__)

# What /api/pipeline/apply stores from the builder. Each starts empty; after an
# apply at least one is set (the builder always sends its INPUT node unless
# the user deleted it).
_BUILDER_STATE_ATTRS = (
    "pipeline_inputs",
    "pipeline_outputs",
    "pipeline_edges",
    "pipeline_normalizers",
    "pipeline_models",
    "pipeline_postprocessors",
)


def has_saved_builder_state():
    return any(getattr(g, attr, None) for attr in _BUILDER_STATE_ATTRS)


def _available_models():
    """The builder's model palette, ``{key: model}``: each catalog model as
    ``category/name`` with its path, then each configured model under its
    own name with its ``to_dict()``, unless the catalog has that key."""
    models = {}
    for category, group in (getattr(g, "model_catalog", None) or {}).items():
        if isinstance(group, dict):
            for name, path in group.items():
                key = f"{category}/{name}"
                models[key] = {"name": key, "category": category, "model_name": name, "path": path}
    for model_config in getattr(g, "models_config", None) or []:
        if model_config.name not in models:
            models[model_config.name] = {"name": model_config.name, **model_config.to_dict()}
    return models


def _configured(model_name):
    """The ``to_dict()`` of the configured model called ``model_name``, or None."""
    for model_config in getattr(g, "models_config", None) or []:
        if getattr(model_config, "name", None) == model_name and hasattr(model_config, "to_dict"):
            return model_config.to_dict()
    return None


def _step_nodes(kind, steps):
    """Builder nodes for a live chain: ``{id, name, params}`` per step."""
    nodes = []
    for idx, step in enumerate(steps):
        step_dict = step.to_dict() if hasattr(step, "to_dict") else {"name": str(step)}
        nodes.append({
            "id": f"{kind}-{idx}-{int(time.time()*1000)}",
            "name": step_dict.get("name", str(step)),
            "params": {k: v for k, v in step_dict.items() if k != "name"},
        })
    return nodes


@pipeline_builder_bp.route("/pipeline-builder")
def pipeline_builder():
    """Render the drag-and-drop pipeline builder interface with current state from globals"""
    available_models = _available_models()

    # Use the state the builder last applied, if it has applied any. Testing
    # for normalizers alone threw away a saved pipeline that has none (the
    # INPUT/OUTPUT nodes, output path, bounding boxes, edges, positions and
    # models) and rebuilt it from the live objects on every reload.
    if has_saved_builder_state():
        # The stored nodes, with their ids, positions and params. A model
        # node applied without its config gets the configured model's.
        current_normalizers = g.pipeline_normalizers
        current_postprocessors = g.pipeline_postprocessors
        current_models = g.pipeline_models
        for model_dict in current_models:
            if "config" not in model_dict:
                config = _configured(model_dict["name"])
                if config is not None:
                    model_dict["config"] = config
        current_inputs = g.pipeline_inputs
        current_outputs = g.pipeline_outputs
        current_edges = g.pipeline_edges
    else:
        # Nothing applied yet: the live chain, and a node per running model
        # with its config, from the configured models or else from a YAML
        # the builder imported.
        current_normalizers = _step_nodes("norm", g.input_norms)
        current_postprocessors = _step_nodes("post", g.postprocess)
        current_models = []
        saved_configs = getattr(g, "pipeline_model_configs", None) or {}
        for idx, job in enumerate(g.jobs):
            if not hasattr(job, "model_name"):
                continue
            model_dict = {"id": f"model-{idx}-{int(time.time()*1000)}", "name": job.model_name, "params": {}}
            config = _configured(job.model_name)
            if config is None:
                config = saved_configs.get(job.model_name)
            if config is not None:
                model_dict["config"] = config
            else:
                logger.warning(f"No config found for {job.model_name}; import a YAML with full model configs")
            current_models.append(model_dict)
        current_inputs = []
        current_outputs = []
        current_edges = []

    return render_template(
        "pipeline_builder_v2.html",
        input_normalizers=get_input_normalizers() or {},
        available_models=available_models or {},
        output_postprocessors=get_postprocessors_list() or {},
        model_mergers=get_model_mergers_list() or {},
        current_normalizers=current_normalizers,
        current_models=current_models,
        current_postprocessors=current_postprocessors,
        current_inputs=current_inputs,
        current_outputs=current_outputs,
        current_edges=current_edges,
        dataset_path=getattr(g, "dataset_path", None) or "",
    )
