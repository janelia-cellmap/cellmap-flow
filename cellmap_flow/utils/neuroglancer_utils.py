"""The CLIs' last step: open the viewer on the dataset, then serve the dashboard.

The viewer and its layers are built by ``cellmap_flow.viewer``; this module
adds the session's models to it and starts the dashboard, which is why it
imports ``dashboard.app`` and ``viewer`` does not.
"""

import itertools
import logging

import neuroglancer

from cellmap_flow.dashboard.app import create_and_run_app
from cellmap_flow.globals import g
from cellmap_flow.utils.scale_pyramid import PREDICTION_COLORS
from cellmap_flow.utils.server_info import fetch_model_info
from cellmap_flow.utils.web_utils import get_norms_post_args
from cellmap_flow.viewer.bootstrap import new_viewer
from cellmap_flow.viewer.layers import (
    prediction_shader_for,
    prediction_source,
    prediction_voxel_override,
    raw_layer,
)

logger = logging.getLogger(__name__)


def _configured_output_voxel_size(model, info):
    """The output voxel size ``model``'s config declares, for a server whose
    ``info`` does not say (one older than model_info); else None.

    Only then: reading ``config`` builds the model, which for a script model
    means loading its weights and taking a CUDA context here.
    """
    mc = {mc.name: mc for mc in getattr(g, "models_config", []) or []}.get(model)
    if mc is None or info.get("output_voxel_size"):
        return None
    try:
        return mc.config.output_voxel_size
    except Exception as e:
        logger.warning(f"Could not read {model}'s output voxel size from its config: {e}")
        return None


def generate_neuroglancer_url(dataset_path, wrap_raw=True):
    """Open the viewer on ``dataset_path`` with a layer for each running model
    and the YAML's extra layers, then serve the dashboard. Does not return."""
    g.dataset_path = dataset_path
    st_data = get_norms_post_args(g.input_norms, g.postprocess)
    layers = {}
    colors = itertools.cycle(PREDICTION_COLORS)
    for job in g.jobs:
        model, host = job.model_name, job.host
        if not host:
            # A zarr://None/... source never loads and nothing replaces
            # it later, so leave the layer out rather than add a dead one.
            logger.warning(f"No server address for '{model}'; not adding a layer")
            continue
        # One round trip, for both the contrast range and the voxel size.
        info = fetch_model_info(host)
        default_shader = prediction_shader_for(model, host, g.postprocess, color=next(colors), info=info)
        g.shaders.setdefault(model, default_shader)
        override = prediction_voxel_override(host, dataset_path, info, _configured_output_voxel_size(model, info))
        layer = {"source": prediction_source(host, model, st_data, override), "shader": g.shaders[model]}
        if g.shader_controls.get(model):
            layer["shaderControls"] = g.shader_controls[model]
        layers[model] = neuroglancer.ImageLayer(**layer)
    # The YAML's extra_layers (cellmap_flow_yaml builds them).
    layers.update(g.extra_layers)

    g.raw = raw_layer(dataset_path, wrap_raw=wrap_raw)
    g.viewer = new_viewer(dataset_path, raw=g.raw, layers=layers)
    viewer_url = str(g.viewer)
    print("viewer", viewer_url)
    # Serves the dashboard; does not return.
    create_and_run_app(neuroglancer_url=viewer_url)
