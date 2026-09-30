"""The CLIs' last step: open the viewer on the dataset, then serve the dashboard.

``generate_neuroglancer_url`` is what ``cellmap_flow <type>`` and
``cellmap_flow_yaml`` end with. The viewer and its layers are built by
``cellmap_flow.viewer``; this module adds the session's models to it and
starts the dashboard, which is why it imports ``dashboard.app`` and
``viewer`` does not. The routes never import it: ``dashboard.app`` imports
them.
"""

import itertools
import logging

from cellmap_flow.dashboard.app import create_and_run_app
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.viewer.raw import PREDICTION_COLORS
from cellmap_flow.serving.client import fetch_model_info
from cellmap_flow.pipeline_spec import PipelineSpec
from cellmap_flow.viewer.bootstrap import new_viewer
from cellmap_flow.viewer.layers import prediction_layer, prediction_shader_for, raw_layer

logger = logging.getLogger(__name__)


def _configured_output_voxel_size(model, info):
    """The output voxel size ``model``'s config declares, for a server whose
    ``info`` does not say (one older than model_info); else None.

    Only then: reading ``config`` builds the model, which for a script model
    means loading its weights and taking a CUDA context here.
    """
    mc = {mc.name: mc for mc in get_session().models_config or []}.get(model)
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
    session = get_session()
    session.dataset_path = dataset_path
    st_data = PipelineSpec.from_steps(session.input_norms, session.postprocess).to_url_blob()
    layers = {}
    colors = itertools.cycle(PREDICTION_COLORS)
    for job in session.jobs:
        model, host = job.model_name, job.host
        if not host:
            # A zarr://None/... source never loads and nothing replaces
            # it later, so leave the layer out rather than add a dead one.
            logger.warning(f"No server address for '{model}'; not adding a layer")
            continue
        # One round trip, for both the contrast range and the voxel size.
        info = fetch_model_info(host)
        default_shader = prediction_shader_for(model, host, session.postprocess, color=next(colors), info=info)
        session.shaders.setdefault(model, default_shader)
        layers[model] = prediction_layer(
            model, host, st_data, dataset_path=dataset_path, postprocess=session.postprocess,
            shader=session.shaders[model], shader_controls=session.shader_controls.get(model), info=info,
            fallback_output_voxel_size=_configured_output_voxel_size(model, info),
        )
    # The YAML's extra_layers (cellmap_flow_yaml builds them).
    layers.update(session.extra_layers)

    session.raw = raw_layer(dataset_path, wrap_raw=wrap_raw)
    session.viewer = new_viewer(dataset_path, raw=session.raw, layers=layers)
    viewer_url = str(session.viewer)
    print("viewer", viewer_url)
    # Serves the dashboard; does not return.
    create_and_run_app(neuroglancer_url=viewer_url)
