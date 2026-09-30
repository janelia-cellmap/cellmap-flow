"""The layers the viewer shows: a model's predictions, and the raw data.

Every path that shows a model's output builds its layer from these, so that
they all place and shade it the same way:

- ``prediction_voxel_override()``: the voxel size to draw a model's output
  at, when it is not the one its server declares.
- ``prediction_source()``: the layer source, a zarr served by the model's
  inference server with the chain in its URL, and the override as its
  transform.
- ``prediction_shader_for()``: a shader over the range the chain produces.
- ``prediction_layer()``: the layer itself, a segmentation when the chain
  ends in labels and an image otherwise.
- ``raw_layer()``: the raw data, with the user's shader put back.

A model's ``info`` is its server's ``model_info`` (utils.server_info). A
caller that already asked for it passes it in, so building a layer asks the
server once.
"""

import logging
import re

import neuroglancer

from cellmap_flow.io.multiscale import closest_raw_scale
from cellmap_flow.pipeline_spec import chain_is_segmentation
from cellmap_flow.utils.output_probe import output_display_range
from cellmap_flow.viewer.raw import PREDICTION_COLORS, get_raw_layer, prediction_shader
from cellmap_flow.utils.server_info import fetch_model_info
from cellmap_flow.utils.web_utils import ARGS_KEY

logger = logging.getLogger(__name__)

_COLOR_RE = re.compile(r'color\(default="([^"]+)"\)')


def prediction_voxel_override(host, dataset_path, info=None, fallback_output_voxel_size=None):
    """The voxel size (nm, z, y, x) to draw a model's output at, or None to
    draw it at the size its server declares.

    A server reads the raw at the pyramid level closest to its model's input
    voxel size and treats that level as if it were at the model's size, so
    when the two differ its output is really at a proportionally different
    size than it declares: a model trained at 16 nm on raw at 6, 12, 24 nm
    reads 12 nm data, and its output lies at 12 nm, not 16.

    A server that reports ``effective_output_voxel_size`` worked that out
    from the level it read, so its answer is used. For an older server the
    override is a guess, the raw level closest to the declared output voxel
    size, which is right when the model's input and output voxel sizes are
    equal.

    The declared size is ``info["output_voxel_size"]``, else
    ``fallback_output_voxel_size`` (for a server too old to say).
    """
    info = fetch_model_info(host) if info is None else info
    declared = info.get("output_voxel_size") or fallback_output_voxel_size
    if not declared:
        return None
    try:
        declared = tuple(declared)
        effective = info.get("effective_output_voxel_size")
        if effective is not None:
            override = tuple(effective)
        elif dataset_path:
            override = closest_raw_scale(dataset_path, declared)
        else:
            return None
    except Exception as e:
        logger.warning(f"Could not find the voxel size to draw {host}'s output at: {e}")
        return None
    if override is None or tuple(override) == declared:
        return None
    logger.info(f"Drawing {host}'s output at {override} nm rather than its declared {declared}, so that it overlays the raw")
    return override


def prediction_source(host, model, url_blob, override_scales=None, has_channel=True):
    """The source of ``model``'s layer: its zarr on ``host``, with ``url_blob``
    (PipelineSpec.to_url_blob) in the URL.

    With ``override_scales`` (z, y, x nm, as prediction_voxel_override gives
    them) it is a dict whose transform declares the array at those scales.
    Neuroglancer wants the same rank on both sides of a transform, so a
    served array with a channel axis (the last, ``has_channel``) keeps it as a
    unit-less ``c^``.
    """
    url = f"zarr://{host}/{model}{ARGS_KEY}{url_blob}{ARGS_KEY}"
    if override_scales is None:
        return url
    dimensions = {axis: [size * 1e-9, "m"] for axis, size in zip("zyx", override_scales)}
    if has_channel:
        dimensions["c^"] = [1, ""]
    return {"url": url, "transform": {"outputDimensions": dimensions, "inputDimensions": dict(dimensions)}}


def prediction_shader_for(model, host, postprocess, previous_shader=None, color=None, info=None):
    """A shader for ``model``'s layer over the range ``postprocess`` produces.

    Unlike the raw there is nothing to sample, since reading the output means
    running the model, but nothing needs sampling: the chain's last step fixes
    the range (output_probe.output_display_range). The colour is the one in
    ``previous_shader`` if it has one, so a recomputed range does not also
    change the colours the user navigates by; else ``color``; else the first
    prediction colour.
    """
    match = _COLOR_RE.search(previous_shader or "")
    color = match.group(1) if match else (color or PREDICTION_COLORS[0])
    try:
        info = fetch_model_info(host) if info is None else info
        steps = [p.to_dict() for p in (postprocess or []) if hasattr(p, "to_dict")]
        value_range = output_display_range(steps, info.get("output_class"))
    except Exception as e:
        logger.debug(f"Could not compute a display range for {model}: {e}")
        value_range = None
    return prediction_shader(color, value_range)


def prediction_layer(model, host, url_blob, *, dataset_path, postprocess, shader=None, shader_controls=None,
                     color=None, previous_shader=None, info=None, fallback_output_voxel_size=None):
    """``model``'s layer, overlaid on the raw at ``dataset_path``.

    A SegmentationLayer when ``postprocess`` ends in labels. Otherwise an
    ImageLayer with ``shader_controls`` and ``shader``, the user's; without
    one, prediction_shader_for's in ``previous_shader``'s colour or
    ``color``.
    """
    info = fetch_model_info(host) if info is None else info
    override = prediction_voxel_override(host, dataset_path, info, fallback_output_voxel_size)
    # A server too old to say is taken to have one, as it always was.
    source = prediction_source(host, model, url_blob, override, has_channel=info.get("has_channel", True))
    if chain_is_segmentation(postprocess):
        return neuroglancer.SegmentationLayer(source=source)
    shader = shader or prediction_shader_for(model, host, postprocess, previous_shader, color, info=info)
    layer = {"source": source, "shader": shader}
    if shader_controls:
        layer["shaderControls"] = shader_controls
    return neuroglancer.ImageLayer(**layer)


def raw_layer(dataset_path, *, wrap_raw=True, shader=None, shader_controls=None):
    """The raw data's layer (viewer.raw.get_raw_layer), with the user's
    ``shader`` and ``shader_controls`` in place of the default contrast.

    A layer with no shader reports the string "None", which is not one.
    """
    layer = get_raw_layer(dataset_path, wrap_raw=wrap_raw)
    if shader and shader != "None":
        layer.shader = shader
    if shader_controls:
        layer.shaderControls = shader_controls
    return layer
