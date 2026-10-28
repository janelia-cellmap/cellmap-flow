"""The layers the viewer shows: a model's predictions, and the raw data.

Every path that shows a model's output builds its layer from these, so that
they all place and shade it the same way:

- ``prediction_voxel_override()``: the voxel size to draw a model's output
  at, when it is not the one its server declares.
- ``prediction_url()`` and ``prediction_source()``: the layer's URL, a zarr
  served by the model's inference server with the chain in it, and its
  source, that URL with the override as its transform.
- ``prediction_shader_for()``: a shader over the range the chain produces;
  for a flow model's channels, one that shows them all at once.
- ``prediction_layer()``: the layer itself, a segmentation when the chain
  ends in labels and an image otherwise, open on the channel
  ``display_channel()`` gives, at ``channel_position()``.
- ``raw_layer()``: the raw data, with the user's shader put back.
- ``replace_layer()``: a layer drawn again in place of one of the same name,
  keeping what the user set on that one (``carried_layer_json()``).

A model's ``info`` is its server's ``model_info`` (serving.client). A
caller that already asked for it passes it in, so building a layer asks the
server once.
"""

import logging
import re

import neuroglancer

from cellmap_flow.io.multiscale import closest_raw_scale
from cellmap_flow.pipeline_spec import chain_is_segmentation, chain_num_channels, chain_output_dtype
from cellmap_flow.serving.probe import output_display_range
from cellmap_flow.viewer.raw import PREDICTION_COLORS, flow_shader, get_raw_layer, prediction_shader
from cellmap_flow.serving.client import fetch_model_info
from cellmap_flow.serving.protocol import ARGS_KEY, FORMAT

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


def prediction_url(host, model, url_blob):
    """The URL of ``model``'s zarr on ``host``, with ``url_blob``
    (PipelineSpec.to_url_blob) in it: what the layer's source points at. The
    server serves the name in the format after it (serving.protocol.FORMAT)."""
    return f"zarr://{host}/{model}{ARGS_KEY}{url_blob}{ARGS_KEY}/{FORMAT}"


def prediction_source(host, model, url_blob, override_scales=None, has_channel=True, channels_shaded=False):
    """The source of ``model``'s layer: its zarr on ``host``, with ``url_blob``
    (PipelineSpec.to_url_blob) in the URL.

    With ``override_scales`` (z, y, x nm, as prediction_voxel_override gives
    them) it is a dict whose transform declares the array at those scales.
    Neuroglancer wants the same rank on both sides of a transform, so a
    served array with a channel axis (``has_channel``) keeps it as a
    unit-less ``c^``.

    Neuroglancer reads a served channel axis as ``c'``, a slider that shows
    one channel at a time. ``channels_shaded`` renames it ``c^``, a channel
    axis, which the shader reads every channel of. A transform renames the
    source dimensions it names and passes the rest through.
    """
    url = prediction_url(host, model, url_blob)
    if override_scales is None and not channels_shaded:
        return url
    dimensions = {}
    if override_scales is not None:
        dimensions = {axis: [size * 1e-9, "m"] for axis, size in zip("zyx", override_scales)}
    if channels_shaded:
        return {"url": url, "transform": {"outputDimensions": {**dimensions, "c^": [1, ""]},
                                          "inputDimensions": {**dimensions, "c'": [1, ""]}}}
    if has_channel:
        dimensions["c^"] = [1, ""]
    return {"url": url, "transform": {"outputDimensions": dimensions, "inputDimensions": dict(dimensions)}}


FLOW_CHANNELS = ("flow_y", "flow_x")


def flow_channels(info, postprocess):
    """The indices of a flow model's ``flow_y`` and ``flow_x`` channels, as its
    server names them, while the chain keeps its channels; None otherwise.

    A model that predicts flows towards each object's centre (Cellpose) has
    channels that mean different things: one at a time, the y flow alone
    says little. prediction_layer shows them together instead.
    """
    channels = info.get("channels")
    if not channels or not info.get("has_channel", True) or not all(c in channels for c in FLOW_CHANNELS):
        return None
    if chain_num_channels(postprocess, len(channels)) != len(channels):
        return None
    return tuple(channels.index(c) for c in FLOW_CHANNELS)


def prediction_shader_for(model, host, postprocess, previous_shader=None, color=None, info=None):
    """A shader for ``model``'s layer over the range ``postprocess`` produces.

    Unlike the raw there is nothing to sample, since reading the output means
    running the model, but nothing needs sampling: the chain's last step fixes
    the range (serving.probe.output_display_range). The colour is the one in
    ``previous_shader`` if it has one, so a recomputed range does not also
    change the colours the user navigates by; else ``color``; else the first
    prediction colour.

    For a flow model's channels (flow_channels), the shader colours each
    voxel by its flow's direction, dimmed by the channel the server names to
    display (``display_channel``, Cellpose's cell probability) over the
    range; unticking ``flows`` shows that channel alone.
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
    # The type the chain serves, when a step declares one (uint8 after a
    # DefaultPostprocessor), which the shader's bounds must be values of.
    try:
        dtype = chain_output_dtype(postprocess or [], None)
    except Exception as e:
        logger.debug(f"Could not tell {model}'s output dtype: {e}")
        dtype = None
    flows = flow_channels(info or {}, postprocess)
    if flows is not None:
        return flow_shader(color, value_range, *flows, info.get("display_channel"), dtype=dtype)
    return prediction_shader(color, value_range, dtype)


def prediction_layer(model, host, url_blob, *, dataset_path, postprocess, shader=None, shader_controls=None,
                     color=None, previous_shader=None, info=None, fallback_output_voxel_size=None, channel=None):
    """``model``'s layer, overlaid on the raw at ``dataset_path``.

    A SegmentationLayer when ``postprocess`` ends in labels. Otherwise an
    ImageLayer with ``shader_controls`` and ``shader``, the user's; without
    one, prediction_shader_for's in ``previous_shader``'s colour or
    ``color``. It opens on ``channel`` when given (the one the Finetune tab
    picked to show, dashboard Session.shown_channel), else on the one its
    server names (channel_position).
    """
    info = fetch_model_info(host) if info is None else info
    override = prediction_voxel_override(host, dataset_path, info, fallback_output_voxel_size)
    flows = flow_channels(info, postprocess)
    # A server too old to say is taken to have one, as it always was.
    source = prediction_source(host, model, url_blob, override, has_channel=info.get("has_channel", True),
                               channels_shaded=flows is not None)
    if chain_is_segmentation(postprocess):
        return neuroglancer.SegmentationLayer(source=source)
    shader = shader or prediction_shader_for(model, host, postprocess, previous_shader, color, info=info)
    layer = {"source": source, "shader": shader}
    if shader_controls:
        layer["shaderControls"] = shader_controls
    position = channel_position(info, postprocess, channel)
    if position is not None:
        layer["local_position"] = position
    return neuroglancer.ImageLayer(**layer)


def display_channel(info, postprocess, shown=None):
    """The channel a model's layer opens on: ``shown``, the one picked to
    show (the Finetune tab's, mito of fly_organelles' eight), else the one
    its server names (``display_channel``, Cellpose's cell probability
    rather than flow_y), while the chain keeps the model's channels and the
    model has that one; None otherwise (the first)."""
    channel = info.get("display_channel") if shown is None else shown
    channels = info.get("output_channels")
    if channel is None or not channels or not info.get("has_channel", True):
        return None
    if chain_num_channels(postprocess, channels) != channels or not 0 <= int(channel) < channels:
        return None
    return int(channel)


def channel_position(info, postprocess, shown=None):
    """The local position an image layer of a model opens at: the channel
    display_channel gives, at its centre (neuroglancer's channel axis is a
    local dimension, its position the channel shown). The served array's
    OME translation is 0 on its channel axis, a voxel centre, so channel k
    lies over [k - 0.5, k + 0.5) and its centre is k: at k + 0.5,
    fly_organelles' layer showed mito (2) with its channel box at 1,
    organelle. None, which leaves neuroglancer's (the first), without one,
    and while a flow model's channels are shown together
    (flow_channels), whose shader reads every one. prediction_layer opens a
    layer at it, and POST /api/finetune/channel moves a layer shown to it."""
    channel = display_channel(info, postprocess, shown)
    if channel is None or flow_channels(info, postprocess) is not None:
        return None
    return [float(channel)]


def raw_layer(dataset_path, *, wrap_raw=True, shader=None, shader_controls=None, view=None):
    """The raw data's layer (viewer.raw.get_raw_layer), with the user's
    ``shader`` and ``shader_controls`` in place of the default contrast;
    without them, the contrast is sampled at ``view`` when given.

    A layer with no shader reports the string "None", which is not one.
    """
    keep = shader and shader != "None"
    layer = get_raw_layer(dataset_path, wrap_raw=wrap_raw, view=None if keep else view)
    if keep:
        layer.shader = shader
    if shader_controls:
        layer.shaderControls = shader_controls
    return layer


# A layer's properties that belong to the data its source serves rather than
# to how the user shows it: a layer drawn again keeps them from the one it
# replaces only while the source is the same (carried_layer_json). The
# segments shown, the merger's equivalences and the segment query and colours
# name segments of that data; the channel shown (local dimensions, position,
# channel dimensions) counts that data's channels; the shader and its
# controls are over its range (the dashboard sets its own for a new chain);
# annotations and a legacy top-level transform are the source's own. The
# browser's carriedLayer reads this list (web/build/export_data.py exports it
# for web/src/viewer/state.ts).
SOURCE_BOUND_LAYER_KEYS = frozenset({
    "annotations",
    "channelDimensions",
    "equivalences",
    "localDimensions",
    "localPosition",
    "segmentColors",
    "segmentQuery",
    "segments",
    "shader",
    "shaderControls",
    "transform",
})


def _source_urls(source):
    """Every URL a layer's JSON source names, in order: a string, a
    ``{"url": ...}``, or a list of either."""
    if isinstance(source, (list, tuple)):
        return [url for item in source for url in _source_urls(item)]
    if isinstance(source, str):
        return [source]
    if isinstance(source, dict) and isinstance(source.get("url"), str):
        return [source["url"]]
    return []


def carried_layer_json(old, new):
    """``new``, a layer's JSON, as it takes the place of ``old`` (or None),
    keeping what the user set on ``old``.

    A redraw (Submit, a model coming up, a mark drawn) builds each layer
    anew, and neuroglancer shows the new one as built: the user's blending
    (additive), opacity, resolution, volume rendering, segment colouring or
    tab were back at their defaults, and a layer they had switched off came
    back on. So the result is ``new`` with every property of ``old`` that
    ``new`` does not set: the dashboard's own properties (the source, a new
    chain's shader, the annotations drawn) win, the user's other settings
    stay. When the source is another one, the properties tied to its data
    (SOURCE_BOUND_LAYER_KEYS) are not kept. A layer of another type (an image
    layer become a segmentation when the chain now ends in labels) has no
    settings this one would read, and keeps only being hidden. The browser's
    viewer does the same (web/src/viewer/state.ts's carriedLayer).
    """
    merged = dict(new)
    if old is None:
        return merged
    if old.get("type") != new.get("type"):
        if old.get("visible") is False:
            merged["visible"] = False
        return merged
    same_source = _source_urls(old.get("source")) == _source_urls(new.get("source"))
    for key, value in old.items():
        if merged.get(key) is not None or key == "name":
            continue
        if not same_source and key in SOURCE_BOUND_LAYER_KEYS:
            continue
        merged[key] = value
    return merged


def replace_layer(layers, name, layer, replacing=None):
    """Put ``layer`` in ``layers`` (a viewer state's) as ``name``, keeping what
    the user set on the layer it replaces (carried_layer_json).

    That is the layer ``name``, or, when there is none, the layer
    ``replacing`` names (a finetuned model's previous iteration, whose
    successor takes its place in the list and its settings); that one is
    removed either way. Being hidden is the managed layer's, not the layer's,
    so it is set on the managed layer after the layer is in place.
    """
    old = None
    if name in layers:
        old = layers[name].to_json()
    if replacing and replacing != name and replacing in layers:
        if old is None:
            old = layers[replacing].to_json()
            # Renamed in place, so the new layer lands where the old one was.
            layers[replacing].name = name
        else:
            del layers[replacing]
    merged = carried_layer_json(old, layer.to_json())
    hidden = merged.pop("visible", True) is False
    merged.pop("name", None)
    layers[name] = neuroglancer.viewer_state.make_layer(merged)
    if hidden:
        layers[name].visible = False
