import logging
import time

from flask import Blueprint, render_template

from cellmap_flow.norm.input_normalize import get_input_normalizers
from cellmap_flow.post.postprocessors import get_postprocessors_list
from cellmap_flow.models.model_merger import get_model_mergers_list
from cellmap_flow.dashboard.routes.index_page import page_op_schemas
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.pipeline_spec import builder_steps

logger = logging.getLogger(__name__)

pipeline_builder_bp = Blueprint("pipeline_builder", __name__)


def _available_models():
    """The builder's model palette, ``{key: model}``: each catalog model as
    ``category/name`` with its path, then each configured model under its
    own name with its ``to_dict()``, unless the catalog has that key."""
    models = {}
    for category, group in (get_session().model_catalog or {}).items():
        if isinstance(group, dict):
            for name, path in group.items():
                key = f"{category}/{name}"
                models[key] = {"name": key, "category": category, "model_name": name, "path": path}
    for model_config in get_session().models_config or []:
        if model_config.name not in models:
            models[model_config.name] = {"name": model_config.name, **model_config.to_dict()}
    return models


def _configured(model_name):
    """The ``to_dict()`` of the configured model called ``model_name``, or None."""
    for model_config in get_session().models_config or []:
        if getattr(model_config, "name", None) == model_name and hasattr(model_config, "to_dict"):
            return model_config.to_dict()
    return None


def _chain_nodes(prefix, steps, saved):
    """The builder's nodes for one live chain: an ``{id, name, params,
    position}`` node per step of ``steps`` (pipeline_spec's ``{name,
    **params}`` steps), given ``saved``, the nodes the builder last applied
    for that chain.

    If the saved nodes are that chain, which they are unless something other
    than the builder has set it since, they are returned exactly as saved.
    Otherwise each step takes the id and position of the first saved node of
    the same op not already taken, so a step whose parameters changed stays
    where it was. A step with no such node gets a new id and no position,
    and the page gives it a place. Saved nodes left over were steps that
    have since gone, and are dropped.
    """
    if builder_steps(saved) == tuple(steps):
        return list(saved)
    unused = list(saved)
    stamp = int(time.time() * 1000)
    nodes = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or "name" not in step:
            continue  # not a step; the op readers skip it too
        node = {"id": f"{prefix}-{index}-{stamp}", "name": step["name"],
                "params": {key: value for key, value in step.items() if key != "name"}}
        same_op = next((n for n in unused if n.get("name") == step["name"]), None)
        if same_op is not None:
            unused.remove(same_op)
            node["id"] = same_op.get("id", node["id"])
            if "position" in same_op:
                node["position"] = same_op["position"]
        nodes.append(node)
    return nodes


@pipeline_builder_bp.route("/pipeline-builder")
def pipeline_builder():
    """The pipeline builder, starting from the dashboard's live chain.

    The pipeline the page starts from (its page data's ``pipeline``) comes
    from two places. Its normalizer and postprocessor nodes are the live
    chain, ``session.pipeline_spec``, however that was last set: by the
    builder's own apply, Submit on the dashboard page, a YAML, or any other
    PUT /api/pipeline. The page sends the chain back with its first edit,
    so nodes taken from anywhere else would undo whatever set the chain
    since. Everything else is the builder's last canvas
    (``session.builder_state``): the INPUT and OUTPUT nodes with their
    paths, boxes and channels, the model nodes, the edges, and each node's
    position. _chain_nodes says how the chain's nodes reuse the canvas's.
    """
    session = get_session()
    available_models = _available_models()
    spec = session.pipeline_spec

    # Use the state the builder last applied, if it has applied any (an apply
    # sends at least the INPUT node, unless the user deleted it). Testing for
    # normalizers alone threw away a saved pipeline that has none (the
    # INPUT/OUTPUT nodes, output path, bounding boxes, edges, positions and
    # models) and rebuilt it from the live objects on every reload.
    saved = session.builder_state
    if any(saved.values()):
        # The stored nodes, with their ids, positions and params. A model
        # node applied without its config gets the configured model's.
        current_models = saved["models"]
        for model_dict in current_models:
            if "config" not in model_dict:
                config = _configured(model_dict["name"])
                if config is not None:
                    model_dict["config"] = config
        current_inputs = saved["inputs"]
        current_outputs = saved["outputs"]
        current_edges = saved["edges"]
    else:
        # Nothing applied yet: a node per running model with its config,
        # from the configured models or else from a YAML the builder
        # imported.
        current_models = []
        saved_configs = session.builder_model_configs
        for idx, job in enumerate(session.jobs):
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

    current_normalizers = _chain_nodes("norm", spec.input_norm, saved["normalizers"])
    current_postprocessors = _chain_nodes("post", spec.postprocess, saved["postprocessors"])
    # An edge to or from a saved step that has gone goes with it.
    gone = ({node.get("id") for node in saved["normalizers"] + saved["postprocessors"]}
            - {node.get("id") for node in current_normalizers + current_postprocessors})
    current_edges = [edge for edge in current_edges if edge.get("from") not in gone and edge.get("to") not in gone]

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
        dataset_path=session.dataset_path or "",
        op_schemas=page_op_schemas(),
    )
