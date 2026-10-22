"""The raw data's neuroglancer layer, and the shaders of the viewer's layers.

``get_raw_layer`` shows a zarr, N5 or precomputed volume. By default it is
served from this process: a multiscale pyramid (an OME-Zarr group's
levels, a precomputed volume's scales) as one ``ScalePyramid`` (each level
a ``LocalVolume``, served as a power-of-two downsampling of the finest; a
level that is not one is left out), a single array
as a ``LocalVolume``, both opened with worldarray (``io.raw.open_raw``) and
indexed through ``LazyNormalization`` so that the input chain applies to
what is drawn (never to a segmentation's ids), and placed by a source
transform that puts voxel 0's lower corner where the metadata says. With
``wrap_raw=False`` neuroglancer reads the files itself. ``RAW_SHADER`` and
``prediction_shader`` are the layers' shaders, their contrast windows
sampled from the data where it can be read.

``ScalePyramid`` is adapted from funlib.show.neuroglancer's scale_pyramid.py
(https://github.com/funkelab/funlib.show.neuroglancer).
"""

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import neuroglancer
import numpy as np
import tensorstore as ts

from cellmap_flow.io import metadata, paths
from cellmap_flow.io.raw import apply_norms, channel_select, open_raw, spatial_axes, spatial_shape
from cellmap_flow.process_chain import process_chain

logger = logging.getLogger(__name__)


class LazyNormalization:
    """A worldarray array seen through the input chain, for neuroglancer to index.

    The channel and the normalizers are looked up on every access rather than
    fixed when the layer was made: a layer that has already served chunks must
    still follow a ChannelSelector or a normalizer changed in the dashboard
    afterwards. Follows the process's chain (``process_chain().input_norms``);
    ``normalize=False`` selects the channel but applies no normalizer (a
    segmentation's ids, or sampling the stored values).
    """

    def __init__(self, array, normalize=True):
        self.array = array
        self.normalize = normalize

    def selected(self):
        """The array's tensorstore (unnormalized) with its channel axis pinned
        to the channel the chain selects; all of it when it has spatial axes only."""
        select = channel_select(self.array, process_chain().input_norms)
        if select is None:
            return self.array.ts
        ((axis, channel),) = select.items()
        return self.array.ts[ts.d[self.array.axes.index(axis)][channel]]

    @property
    def dtype(self):
        """The dtype indexing gives: the last step of the chain that declares
        one decides, and steps without one (ChannelSelector) pass their
        input's through; the stored dtype with ``normalize=False``."""
        for norm in reversed(process_chain().input_norms if self.normalize else []):
            if norm.dtype is not None:
                return np.dtype(norm.dtype)
        return self.array.dtype

    def __getitem__(self, index):
        data = self.selected()[index].read().result()
        return apply_norms(data) if self.normalize else data

    def __getattr__(self, attr):
        if attr in ("array", "normalize"):
            # Not set yet (e.g. while unpickling); don't recurse.
            raise AttributeError(attr)
        return getattr(self.selected(), attr)


# Shader for the raw EM layer. ``range`` is the displayed contrast window;
# ``window`` is the wider span the UI slider can be dragged over. Both are
# filled in by _invlerp_bounds, which writes them as the data's type allows.
RAW_SHADER = """#uicontrol invlerp normalized({bounds});
#uicontrol vec3 color color(default="white");
void main(){{emitRGB(color * normalized());}}"""


PREDICTION_SHADER = """#uicontrol invlerp normalized({bounds});
#uicontrol vec3 color color(default="{color}");
void main(){{emitRGB(color * normalized());}}"""

PREDICTION_COLORS = [
    "red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta",
]


def _invlerp_bounds(lo, hi, wlo, whi, dtype=None):
    """An invlerp control's ``range=[lo, hi], window=[wlo, whi]``, as
    neuroglancer accepts them for data of ``dtype``.

    neuroglancer reads the four numbers as values of the layer's data type,
    and refuses the whole shader when one is not: over uint8 data, a window
    padded below 0 ("Invalid uint8 value: "-75"") or a fractional
    percentile. So for an integer dtype the range is widened to whole
    numbers and everything is clipped to what the dtype holds; a float
    dtype, or an unknown one, takes them as they are.
    """
    dtype = None if dtype is None else np.dtype(getattr(dtype, "numpy_dtype", dtype))
    if dtype is None or dtype.kind not in "ui":
        return f"range=[{lo:.6g}, {hi:.6g}], window=[{wlo:.6g}, {whi:.6g}]"
    info = np.iinfo(dtype)

    def whole(value, rounding):
        return int(min(max(rounding(value), info.min), info.max))

    lo, hi = whole(lo, math.floor), whole(hi, math.ceil)
    wlo, whi = min(whole(wlo, math.floor), lo), max(whole(whi, math.ceil), hi)
    return f"range=[{lo}, {hi}], window=[{wlo}, {whi}]"


def prediction_shader(color, value_range=None, dtype=None):
    """Shader for a model output layer, over the range the chain produces.

    Falls back to [0, 1] when the range is undetermined. That is a guess, but
    the previous default was ``range=[0.5, 0.5]`` -- lo == hi makes invlerp a
    step function rather than a ramp, so every value above 0.5 rendered as
    solid colour. After a DefaultPostprocessor (0-255) that is the entire
    prediction. ``dtype`` is the output's, when known: see _invlerp_bounds.
    """
    if value_range is None:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = (float(v) for v in value_range)
    # Leave the slider room to move beyond the computed range.
    pad = (hi - lo) * 0.5 or 1.0
    return PREDICTION_SHADER.format(bounds=_invlerp_bounds(lo, hi, lo - pad, hi + pad, dtype), color=color)


# Shader for a flow model's channels (a channel axis, c^): each voxel's hue is
# its flow's direction and its brightness the flow's length over
# ``flow_length`` (Cellpose's flows are about 5 long inside a cell), times
# the display channel (``cell``) over the range. Unticking ``flows`` shows
# the display channel alone.
FLOW_SHADER = """#uicontrol bool flows checkbox(default=true);
#uicontrol float flow_length slider(min=0.5, max=10, default=5);
{cell_control}#uicontrol vec3 color color(default="{color}");
void main(){{
  float shown = {cell};
  if (!flows) {{ emitRGB(color * shown); return; }}
  vec2 flow = vec2(toNormalized(getDataValue({flow_x})), toNormalized(getDataValue({flow_y})));
  float len = length(flow);
  vec2 dir = flow / max(len, 1e-6);
  // cos(angle + 0, -120, 120 degrees), from the direction without atan.
  vec3 hue = 0.5 + 0.5 * (dir.x * vec3(1.0, -0.5, -0.5) - dir.y * vec3(0.0, -0.866, 0.866));
  emitRGB(hue * clamp(len / flow_length, 0.0, 1.0) * shown);
}}"""


def flow_shader(color, value_range, flow_y, flow_x, display_channel=None, dtype=None):
    """FLOW_SHADER over channels ``flow_y`` and ``flow_x``, dimmed by channel
    ``display_channel`` over ``value_range`` (as prediction_shader's), or
    not dimmed without one."""
    if display_channel is None:
        cell_control, cell = "", "1.0"
    else:
        lo, hi = (0.0, 1.0) if value_range is None else (float(v) for v in value_range)
        pad = (hi - lo) * 0.5 or 1.0
        bounds = _invlerp_bounds(lo, hi, lo - pad, hi + pad, dtype)
        cell_control = f"#uicontrol invlerp normalized({bounds}, channel={int(display_channel)});\n"
        cell = "normalized()"
    return FLOW_SHADER.format(cell_control=cell_control, cell=cell, color=color, flow_y=int(flow_y),
                              flow_x=int(flow_x))


def _dtype_default_range(array, normalize=True):
    """Fallback display range when percentiles can't be computed.

    The stored dtype's range put through the input chain ``array`` is shown
    through (none with ``normalize=False``): uint8 is [0, 255] as it is,
    [0, 1] after a MinMaxNormalizer, and [-1, 1] with ``x*2-1`` after that. It
    used to be the range of the dtype the chain declares, so any float chain
    got [-1, 1], and a pyramid got none of this: [0, 255] over [0, 1] data,
    one flat colour. When the chain cannot be run on the two ends, the
    declared dtype's range, [-1, 1] for a float.
    """
    stored = array.dtype
    if normalize and stored.kind in "ui":
        info = np.iinfo(stored)
        ends = np.array([info.min, info.max], dtype=stored).reshape((1,) * (len(array.spatial) - 1) + (2,))
        try:
            mapped = np.asarray(apply_norms(ends), dtype=float).ravel()
            lo, hi = float(mapped.min()), float(mapped.max())
            if np.isfinite([lo, hi]).all() and hi > lo:
                return lo, hi
        except Exception as e:
            logger.debug(f"Could not put the dtype's range through the input chain: {e}")
    dtype = LazyNormalization(array, normalize).dtype
    if dtype.kind in "ui":
        info = np.iinfo(dtype)
        return float(info.min), float(info.max)
    return -1.0, 1.0


def _auto_contrast_range(paths, normalize, lo_pct=1.0, hi_pct=99.0):
    """Derive a display range from the data rather than hardcoding one.

    Reads the finest pyramid level small enough to read here (at most
    MAX_VOXELS), falling back to coarser ones when it cannot be read or is
    flat. Coarse levels are averaged so much that the edge of the data
    blurs into the padding around it: on jrc_axolotl-heart-1 the 1st
    percentile is 3 at s9 but 16 at s7. Levels under MIN_VOXELS are too few
    to be representative. Percentiles rather than min/max so a handful of
    saturated voxels (very common in EM) don't flatten everything else into
    a narrow band, and with the padding left out (_without_padding), which
    otherwise pins the low end at 0 as if it were data.

    ``paths`` is ordered fine -> coarse. Returns (lo, hi), or None to let the
    caller fall back.
    """
    MIN_VOXELS = 4096
    MAX_VOXELS = 8_000_000
    readable = []
    for path in reversed(paths):
        try:
            array = open_raw(path)
        except Exception as e:
            logger.debug(f"Auto-contrast could not open {path}: {e}")
            continue
        n_voxels = int(np.prod(spatial_shape(array)))
        if n_voxels > MAX_VOXELS:
            break  # finer levels are only bigger -- stop rather than read them
        if n_voxels >= MIN_VOXELS:
            readable.append((path, array))
    for path, array in reversed(readable):
        try:
            # Read through LazyNormalization, as neuroglancer's LocalVolume
            # indexes the layer, so the sample is in the same space as what
            # gets displayed. Sampled unnormalized, the percentiles would land
            # in raw uint8 space while the layer shows [-1, 1] -- a 0-168
            # range over data that never exceeds 1, i.e. an all-black image.
            arr = np.asarray(LazyNormalization(array, normalize)[...]).ravel()
            arr = _without_padding(arr[np.isfinite(arr)])
            if arr.size == 0:
                continue
            lo, hi = (float(v) for v in np.percentile(arr, [lo_pct, hi_pct]))
            if hi <= lo:
                continue  # flat level (e.g. all padding)
            return lo, hi
        except Exception as e:
            logger.debug(f"Auto-contrast failed on {path}: {e}")
            continue
    return None


def _without_padding(arr, pile=0.05, among=3, reach=0.02):
    """``arr`` without its padding: a pile of voxels on one of the
    ``among`` lowest (or highest) values, holding at least ``pile`` of them,
    and everything from that end to ``reach`` of the value range past it.

    Padding is the empty space around the data, piled on one value or a
    few: the 0s around jrc_axolotl-heart-1 (29% of its voxels), or male
    CNS's JPEG'd background, 80% at 1 and 3% at 0 with a tail of 2s and 3s
    that ``reach`` takes with it. Left in, it is the 1st percentile itself.
    The data's own darkest and brightest voxels are spread over many
    values, none of them a pile, and stay. Nothing is left out when that
    would leave nothing (a flat level).
    """
    if arr.size == 0:
        return arr
    # Found on at most a million of them: np.unique sorts.
    values, counts = np.unique(arr[:: max(arr.size // 1_000_000, 1)], return_counts=True)
    piled = values[counts >= pile * counts.sum()]
    span = float(values[-1]) - float(values[0])
    keep = np.ones(arr.shape, bool)
    low, high = piled[np.isin(piled, values[:among])], piled[np.isin(piled, values[-among:])]
    if low.size:
        keep &= arr > float(low.max()) + reach * span
    if high.size:
        keep &= arr < float(high.min()) - reach * span
    kept = arr[keep]
    return kept if kept.size else arr


# How far a panel reaches from its centre, in screen pixels. The viewer's
# state does not say how big its panels are; 512 is about half a one-panel
# layout's width on a laptop. A larger panel shows a little more than is
# sampled, a 4-panel layout's panels a little less.
VIEW_HALF_WIDTH_PX = 512
# How many voxels each way from the centre a plane of _view_contrast_range
# reads at most.
MAX_VIEW_REACH = 256


@dataclass(frozen=True)
class View:
    """Where the viewer looks: ``center`` in nm by axis name, and the size in
    nm of one screen pixel there."""

    center: dict
    pixel_nm: float


_NM_PER_UNIT = {"m": 1e9, "um": 1e3, "µm": 1e3, "nm": 1.0}


def view_of(state):
    """``state``'s View (a neuroglancer ViewerState), or None when it does
    not have one yet (no position or zoom: the browser has not reported
    them), or its units are not lengths.

    The position is in voxels of the viewer's coordinate space, whose
    scales neuroglancer reports in SI units (8 nm as 8e-9 m). Its
    ``crossSectionScale`` is how many of the smallest display dimension's
    voxels one screen pixel spans.
    """
    try:
        dimensions, position, zoom = state.dimensions, state.position, state.cross_section_scale
        if dimensions is None or position is None or zoom is None:
            return None
        names, scales, units = list(dimensions.names), list(dimensions.scales), list(dimensions.units)
        if len(position) != len(names):
            return None
        scales_nm = [float(scale) * _NM_PER_UNIT[unit] for scale, unit in zip(scales, units)]
        display = list(state.display_dimensions or names[:3])
        smallest = min(scales_nm[names.index(name)] for name in display)
        center = {name: float(p) * scale for name, p, scale in zip(names, position, scales_nm)}
        return View(center=center, pixel_nm=float(zoom) * smallest)
    except Exception as e:
        logger.debug(f"Could not read the viewer's view: {e}")
        return None


def _view_contrast_range(paths, normalize, view, lo_pct=1.0, hi_pct=99.0):
    """The range neuroglancer's own 1-99% button would give at ``view``, or
    None when the data there cannot be read.

    The button takes percentiles of what is on screen. So this reads the
    pyramid level neuroglancer shows at the view's zoom (the coarsest no
    coarser than a screen pixel; the finest when zoomed in past it), or a
    coarser one when that would be over MAX_VIEW_REACH voxels each way, and
    from it the three planes through the view's centre, each
    VIEW_HALF_WIDTH_PX screen pixels each way. Padding is left out as in
    _auto_contrast_range. ``paths`` is ordered fine -> coarse.
    """
    try:
        arrays = [open_raw(path) for path in paths]
        level = 0
        for index, array in enumerate(arrays):
            if min(array.voxel_size) <= view.pixel_nm * 1.01:
                level = index
        # Zoomed in on a large panel that is a million voxels a plane, about
        # 3 s from /nrs: read coarser levels until a plane is at most
        # MAX_VIEW_REACH voxels each way. Their percentiles barely differ.
        while (level + 1 < len(arrays)
               and VIEW_HALF_WIDTH_PX * view.pixel_nm / min(arrays[level].voxel_size) > MAX_VIEW_REACH):
            level += 1
        array = arrays[level]
        names = spatial_axes(array)
        if any(name not in view.center for name in names):
            return None
        voxel_size = np.asarray(array.voxel_size, dtype=float)
        corner = np.asarray(array.corner, dtype=float)
        shape = np.asarray(spatial_shape(array))
        shown = LazyNormalization(array, normalize)
        centre = np.floor((np.asarray([view.center[name] for name in names]) - corner) / voxel_size).astype(int)
        reach = np.ceil(VIEW_HALF_WIDTH_PX * view.pixel_nm / voxel_size).astype(int)
        values = []
        for axis in range(len(names)):
            if not 0 <= centre[axis] < shape[axis]:
                continue  # the plane across this axis misses the data
            index = tuple(
                slice(centre[i], centre[i] + 1) if i == axis
                else slice(max(centre[i] - reach[i], 0), min(centre[i] + reach[i] + 1, shape[i]))
                for i in range(len(names))
            )
            values.append(np.asarray(shown[(Ellipsis, *index)]).ravel())
        arr = np.concatenate(values) if values else np.empty(0)
        arr = _without_padding(arr[np.isfinite(arr)])
        if arr.size < 1024:
            return None  # off the data, or too little of it to go by
        lo, hi = (float(v) for v in np.percentile(arr, [lo_pct, hi_pct]))
        return (lo, hi) if hi > lo else None
    except Exception as e:
        logger.debug(f"Contrast at the view failed: {e}")
        return None


def _shown_dtype(paths, normalize, array=None):
    """The dtype of what the raw layer shows: ``array``'s (an open level), else
    that of ``paths[0]``, read as ``normalize`` says; None when neither can be
    opened."""
    try:
        return LazyNormalization(open_raw(paths[0]) if array is None else array, normalize).dtype
    except Exception as e:
        logger.debug(f"Could not tell the raw layer's dtype: {e}")
        return None


def _raw_shader(paths, normalize, array=None, view=None):
    """Build the raw-layer shader, preferring a data-derived contrast range:
    that at ``view`` when given (_view_contrast_range), else the whole
    volume's; else the range of the dtype of ``array`` (an open level, read as
    ``normalize`` says), else 0-255."""
    rng = _view_contrast_range(paths, normalize, view) if paths and view is not None else None
    if rng is not None:
        logger.info("Raw contrast taken from the viewer's view")
    elif paths:
        rng = _auto_contrast_range(paths, normalize)
    if rng is not None:
        lo, hi = rng
        # Let the slider reach well beyond the auto range so it stays adjustable.
        pad = (hi - lo) * 0.5 or 1.0
        wlo, whi = lo - pad, hi + pad
        logger.info(f"Raw auto-contrast range [{lo:.6g}, {hi:.6g}]")
    else:
        lo, hi = _dtype_default_range(array, normalize) if array is not None else (0.0, 255.0)
        wlo, whi = lo, hi
        logger.info(f"Raw auto-contrast unavailable; using [{lo:.6g}, {hi:.6g}]")
    dtype = _shown_dtype(paths, normalize, array) if paths or array is not None else None
    return RAW_SHADER.format(bounds=_invlerp_bounds(lo, hi, wlo, whi, dtype))


def _layer(source, shader, segmentation, disable_meshes):
    """An ImageLayer over ``source``, or a SegmentationLayer when asked.

    ``shader`` is called only for an image: sampling a contrast range is of
    no use to labels. ``disable_meshes`` turns off a segmentation's meshes
    subsource, so picking a segment does not have neuroglancer compute its
    mesh, which for a whole-cell label volume can exhaust a node's memory.
    """
    if not segmentation:
        return neuroglancer.ImageLayer(source=source, shader=shader())
    if isinstance(source, str):
        source = neuroglancer.LayerDataSource(url=source)
    if disable_meshes:
        source.subsources = {"meshes": False}
    return neuroglancer.SegmentationLayer(source=source)


def _is_sn(name):
    return name.startswith("s") and name[1:].isdigit()


def _is_array(path):
    try:
        return metadata.node_type(path)[1] == "array"
    except Exception:
        return False


def _sn_arrays(group):
    """The sN children of ``group``, s0 first; none when it has none or is
    not a directory or group. A URL cannot be listed, so there s0, s1, ...
    are looked for in turn, up to the first that is not an array."""
    if paths.is_remote(group):
        names = []
        while _is_array(paths.join(group, f"s{len(names)}")):
            names.append(f"s{len(names)}")
        return names
    try:
        names = os.listdir(group)
    except Exception:
        return []
    return sorted((name for name in names if _is_sn(name)), key=lambda name: int(name[1:]))


def _pyramid(dataset_path):
    """``(group, level paths)`` when ``dataset_path`` is a multiscale pyramid
    or one of its levels, else None.

    An OME-Zarr group's levels are the arrays its multiscales list, by their
    ``datasets[].path`` in the order listed, whatever they are named ("s0"
    or "0"); the path of one of them finds the others. A pyramid without
    OME multiscales (N5, funlib) is a group's sN arrays, and an sN path is
    one of its parent's. A precomputed volume's are its scales (see
    _precomputed_pyramid).
    """
    if paths.is_precomputed(dataset_path):
        return _precomputed_pyramid(dataset_path)
    parent, _, leaf = dataset_path.rstrip("/").rpartition("/")
    for group, level in ((dataset_path, None), (parent, leaf)):
        try:
            levels = [path.strip("/") for path, _ in metadata.list_levels(group)]
        except Exception:
            continue  # no OME multiscales there
        # An array an OME group does not list is not one of its levels.
        return (group, levels) if level is None or level in levels else None
    group = parent if _is_sn(leaf) else dataset_path
    levels = _sn_arrays(group)
    return (group, levels) if levels else None


def _precomputed_pyramid(dataset_path):
    """``(volume, level paths)`` for a precomputed volume of more than one
    scale, or one of its scales (``…/s2``), else None.

    Its levels are s0, s1, ..., each scale its info lists, in that order
    (io.metadata.list_levels). A volume of one scale is shown as one array,
    as it always was.
    """
    volume, scale = paths.precomputed_scale(dataset_path)
    try:
        levels = [path for path, _ in metadata.list_levels(volume)]
    except Exception:
        return None
    if len(levels) < 2 or (scale is not None and scale >= len(levels)):
        return None
    return volume, levels


def neuroglancer_source(dataset_path):
    """The source neuroglancer reads ``dataset_path`` from itself, in the
    browser (``wrap_raw=False``): its location with the format in front, as
    neuroglancer names its data sources -- ``zarr://`` (v2 or v3, which
    neuroglancer tells apart), ``n5://`` or ``precomputed://``, followed by a
    local path or an ``http(s)://``, ``s3://`` or ``gs://`` URL.

    A precomputed volume is named whole: neuroglancer opens a volume, never
    one of its scales (``…/s2`` has no info of its own), and a bare ``gs://``
    path is one.
    """
    if paths.is_precomputed(dataset_path):
        return "precomputed://" + paths.precomputed_volume(dataset_path)[0]
    return f"{paths.suffix_format(dataset_path) or 'zarr'}://{dataset_path}"


def get_raw_layer(
    dataset_path, normalize=True, wrap_raw=True, segmentation=False, disable_meshes=False, view=None
):
    """A neuroglancer layer showing a zarr, n5 or precomputed volume.

    A multiscale pyramid, or any of its levels, is shown as the whole
    pyramid (see _pyramid for what its levels are). ``segmentation`` gives a
    SegmentationLayer over the same source, placed the same way, for a
    label volume; its ids are served as stored, never through the input
    normalizers. ``disable_meshes`` then turns off its meshes subsource
    (see _layer). An image's contrast is sampled at ``view`` (a View, see
    view_of) when given, as neuroglancer's 1-99% button would there; else
    over the whole volume.
    """
    dataset_path = paths.normalize_path(dataset_path)
    original_dataset_path = dataset_path
    pyramid = _pyramid(dataset_path)
    if pyramid is not None:
        dataset_path, scales = pyramid

    if not wrap_raw:
        # Unwrapped: neuroglancer fetches the file directly, so the input
        # normalizers never run on what it displays. Sample unnormalized
        # too, or the contrast range lands in the wrong space entirely. A
        # pyramid is sampled from its levels: its group is no array to read.
        levels = [paths.join(dataset_path, sc) for sc in scales] if pyramid is not None else [original_dataset_path]
        return _layer(
            neuroglancer_source(dataset_path),
            lambda: _raw_shader(levels, normalize=False, view=view),
            segmentation,
            disable_meshes,
        )

    if pyramid is not None:
        try:

            def level(scale):
                array = open_raw(paths.join(dataset_path, scale))
                return array, _local_volume(array, normalize and not segmentation)

            # Each level's metadata read and tensorstore open is a round trip
            # to wherever the data is, about 40 ms each on gs://, where a
            # volume can have a dozen levels: they are made side by side.
            with ThreadPoolExecutor(max_workers=len(scales)) as pool:
                arrays, layers = (list(built) for built in zip(*pool.map(level, scales)))
            # ScalePyramid serves every level as a downsampling of the finest
            # one, so the finest level's corner places them all. That holds
            # for pyramids whose levels share a corner (Janelia's all sit at
            # -4 nm); a level with a different corner cannot be expressed.
            finest = min(arrays, key=lambda array: tuple(array.voxel_size))

            # Previously this branch set no shader at all, which neuroglancer
            # reports back as the literal string "None" -- see the guard in
            # dashboard/routes/pipeline.py.
            return _layer(
                neuroglancer.LayerDataSource(
                    url=ScalePyramid(layers), transform=_corner_transform(finest)
                ),
                lambda: _raw_shader(
                    [paths.join(dataset_path, sc) for sc in scales], normalize, array=finest, view=view,
                ),
                segmentation,
                disable_meshes,
            )
        except Exception as e:
            logger.error(e)

    # One array, or a pyramid that could not be shown as one. An image here
    # reads through the input chain whatever normalize says, as it always has.
    array = open_raw(original_dataset_path)
    return _layer(
        neuroglancer.LayerDataSource(
            url=_local_volume(array, not segmentation), transform=_corner_transform(array)
        ),
        lambda: _raw_shader([original_dataset_path], normalize, array=array, view=view),
        segmentation,
        disable_meshes,
    )


def _dimensions(array):
    """The layer's dimensions: ``array``'s spatial axes, at its voxel size in nm."""
    return neuroglancer.CoordinateSpace(
        names=list(spatial_axes(array)), units="nm", scales=list(array.voxel_size)
    )


def _local_volume(array, normalize):
    """``array`` as a LocalVolume indexed from 0, through the input chain
    unless ``normalize`` is off (LazyNormalization); _corner_transform places it."""
    return neuroglancer.LocalVolume(
        data=LazyNormalization(array, normalize),
        dimensions=_dimensions(array),
        voxel_offset=[0] * len(array.spatial),
    )


def _corner_transform(array):
    """The transform putting ``array``'s voxel 0 lower corner where it is (its
    ``corner``, nm).

    LocalVolume's own voxel_offset is a whole number of voxels, so it cannot
    hold an OME corner such as -4 nm at 8 nm -- and it was being handed the
    offset in nm, which drew any dataset not at the origin far from its data.
    The matrix's translation is in voxels of the output dimensions.
    """
    rank = len(array.spatial)
    corner = np.asarray(array.corner, dtype=float)
    voxel_size = np.asarray(array.voxel_size, dtype=float)
    matrix = np.hstack([np.eye(rank), np.zeros((rank, 1))])
    matrix[:, rank] = corner / voxel_size
    return neuroglancer.CoordinateSpaceTransform(
        output_dimensions=_dimensions(array), matrix=matrix
    )


def _downsampling_factor(voxel_size, finest):
    """``voxel_size`` over ``finest``, per axis, when each ratio is a power
    of two (1, 2, 4, ...), as whole numbers; else None. A ratio within 1e-6
    of one counts as it, so float noise in a voxel size (7.999999999 nm)
    does not drop its level."""
    factor = []
    for size, fine in zip(voxel_size, finest):
        ratio = size / fine
        power = round(math.log2(ratio)) if ratio > 0 else -1
        if power < 0 or not math.isclose(ratio, 2**power, rel_tol=1e-6):
            return None
        factor.append(2**power)
    return tuple(factor)


def _nm(layer):
    """A LocalVolume's voxel size as text, such as "8×8×8 nm" (neuroglancer
    holds it in metres)."""
    return "×".join(f"{scale * 1e9:g}" for scale in layer.dimensions.scales) + " nm"


class ScalePyramid(neuroglancer.LocalVolume):
    """One neuroglancer volume served from a ``LocalVolume`` per pyramid level.
    Mimics a LocalVolume: neuroglancer sees the finest level.

    neuroglancer's python data source asks only for power-of-two
    downsamplings of the volume it sees (a ``scale_key`` such as "2,2,2").
    Each level is kept under its downsampling factor, its voxel size over the
    finest's per axis ((1, 1, 1) for the finest). A request is served by the
    level at or below it on every axis that leaves the smallest factor on
    any one axis (in a pyramid whose levels are each coarser than the last on
    every axis, the coarsest such level), and that level's LocalVolume
    downsamples the rest of the way, up to its own limit; neuroglancer is
    told to ask no further (``_max_downsampling``).

    A level that is not a power-of-two multiple of the finest on every axis
    (1.5×, 3×) could never answer a request exactly, so it is left out, as is
    a second level at a factor already kept; one warning names them. The
    finest level is always kept, so the pyramid is never empty.

    Args:

            volume_layers (``list`` of ``LocalVolume``):

                One ``LocalVolume`` per level.
    """

    def __init__(self, volume_layers):
        super(neuroglancer.LocalVolume, self).__init__()

        finest = min(volume_layers, key=lambda layer: tuple(layer.dimensions.scales))
        self.dims = len(finest.dimensions.scales)
        self.volume_layers = {}
        dropped = []
        for layer in [finest] + [layer for layer in volume_layers if layer is not finest]:
            factor = _downsampling_factor(layer.dimensions.scales, finest.dimensions.scales)
            if factor is None or factor in self.volume_layers:
                dropped.append(layer)
            else:
                self.volume_layers[factor] = layer
        if dropped:
            logger.warning(
                "Leaving out the pyramid levels at %s: neuroglancer asks only for "
                "power-of-two downsamplings of the finest level, %s, which a level that "
                "is not a power-of-two multiple of it on every axis (or repeats one) "
                "can never answer exactly.",
                ", ".join(_nm(layer) for layer in dropped),
                _nm(finest),
            )

        logger.info("scale keys: %s", list(self.volume_layers))
        logger.info(self.info())

    @property
    def volume_type(self):
        return self.volume_layers[(1,) * self.dims].volume_type

    @property
    def token(self):
        return self.volume_layers[(1,) * self.dims].token

    def info(self):
        reference_layer = self.volume_layers[(1,) * self.dims]
        # return reference_layer.info()

        reference_info = reference_layer.info()

        info = {
            "dataType": reference_info["dataType"],
            "encoding": reference_info["encoding"],
            "generation": reference_info["generation"],
            "coordinateSpace": reference_info["coordinateSpace"],
            "shape": reference_info["shape"],
            "volumeType": reference_info["volumeType"],
            "voxelOffset": reference_info["voxelOffset"],
            "chunkLayout": reference_info["chunkLayout"],
            "downsamplingLayout": reference_info["downsamplingLayout"],
            "maxDownsampling": self._max_downsampling(),
            "maxDownsampledSize": reference_info["maxDownsampledSize"],
            "maxDownsamplingScales": reference_info["maxDownsamplingScales"],
        }

        return info

    def _max_downsampling(self):
        """How far neuroglancer may downsample the finest level, as the
        product of the factors over the axes (its ``maxDownsampling``): the
        coarsest level's factor times as far as that level's own
        LocalVolume downsamples (64 by default, 4x4x4), so that zooming out
        past the coarsest level works as it does for one array; a one-level
        pyramid gets exactly that 64. None, no limit, when the LocalVolume
        has none."""
        coarsest = max(self.volume_layers, key=np.prod)
        own = self.volume_layers[coarsest].max_downsampling
        return None if math.isinf(own) else int(np.prod(coarsest)) * int(own)

    def get_encoded_subvolume(self, data_format, start, end, scale_key=None):
        if scale_key is None:
            scale_key = ",".join(("1",) * self.dims)

        scale = tuple(int(s) for s in scale_key.split(","))
        closest_scale = None
        min_diff = np.inf
        for volume_scales in self.volume_layers.keys():
            scale_diff = np.array(scale) // np.array(volume_scales)
            if any(scale_diff < 1):
                continue
            scale_diff = scale_diff.max()
            if scale_diff < min_diff:
                min_diff = scale_diff
                closest_scale = volume_scales

        assert closest_scale is not None
        relative_scale = np.array(scale) // np.array(closest_scale)

        result = self.volume_layers[closest_scale].get_encoded_subvolume(
            data_format, start, end, scale_key=",".join(map(str, relative_scale))
        )

        return result

    def get_object_mesh(self, object_id):
        return self.volume_layers[(1,) * self.dims].get_object_mesh(object_id)

    def invalidate(self):
        return self.volume_layers[(1,) * self.dims].invalidate()
