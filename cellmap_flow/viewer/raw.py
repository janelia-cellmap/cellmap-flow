"""The raw data's neuroglancer layer, and the shaders of the viewer's layers.

``get_raw_layer`` shows a zarr, N5 or precomputed volume. By default it is
served from this process: a multiscale pyramid (an OME-Zarr group's
levels, a precomputed volume's scales) as one ``ScalePyramid`` (each level
a ``LocalVolume``, served as a power-of-two downsampling of the finest; a
level that is not one is left out), a single array
as a ``LocalVolume``, both read through ImageDataInterface so that the
input chain applies to what is drawn (never to a segmentation's ids), and
placed by a source transform that puts voxel 0's lower corner where the
metadata says. With ``wrap_raw=False`` neuroglancer reads the
files itself. ``RAW_SHADER`` and ``prediction_shader`` are the layers'
shaders, their contrast windows sampled from the data where it can be read.

``ScalePyramid`` is adapted from funlib.show.neuroglancer's scale_pyramid.py
(https://github.com/funkelab/funlib.show.neuroglancer).
"""

import logging
import math
import os

import neuroglancer
import numpy as np

from cellmap_flow.image_data_interface import ImageDataInterface
from cellmap_flow.io import metadata, paths
from cellmap_flow.io.metadata import open_zarr

logger = logging.getLogger(__name__)


# Shader for the raw EM layer. ``range`` is the displayed contrast window;
# ``window`` is the wider span the UI slider can be dragged over.
RAW_SHADER = """#uicontrol invlerp normalized(range=[{lo:.6g}, {hi:.6g}], window=[{wlo:.6g}, {whi:.6g}]);
#uicontrol vec3 color color(default="white");
void main(){{emitRGB(color * normalized());}}"""


PREDICTION_SHADER = """#uicontrol invlerp normalized(range=[{lo:.6g}, {hi:.6g}], window=[{wlo:.6g}, {whi:.6g}]);
#uicontrol vec3 color color(default="{color}");
void main(){{emitRGB(color * normalized());}}"""

PREDICTION_COLORS = [
    "red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta",
]


def prediction_shader(color, value_range=None):
    """Shader for a model output layer, over the range the chain produces.

    Falls back to [0, 1] when the range is undetermined. That is a guess, but
    the previous default was ``range=[0.5, 0.5]`` -- lo == hi makes invlerp a
    step function rather than a ramp, so every value above 0.5 rendered as
    solid colour. After a DefaultPostprocessor (0-255) that is the entire
    prediction.
    """
    if value_range is None:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = (float(v) for v in value_range)
    # Leave the slider room to move beyond the computed range.
    pad = (hi - lo) * 0.5 or 1.0
    return PREDICTION_SHADER.format(
        lo=lo, hi=hi, wlo=lo - pad, whi=hi + pad, color=color
    )


def _dtype_default_range(image):
    """Fallback display range when percentiles can't be computed: the range
    of the dtype ``image`` is shown in, which is its ``ts``'s (the input
    chain's last declared dtype, else the array's), or [-1, 1] for a float,
    which has none."""
    dtype = image.ts.dtype
    dtype = np.dtype(getattr(dtype, "numpy_dtype", dtype))
    if dtype.kind in "ui":
        info = np.iinfo(dtype)
        return float(info.min), float(info.max)
    return -1.0, 1.0


def _auto_contrast_range(paths, normalize, lo_pct=1.0, hi_pct=99.0):
    """Derive a display range from the data rather than hardcoding one.

    Reads the coarsest pyramid level that is still big enough to be
    representative -- the bottom of a deep pyramid is only a few voxels, and the
    top is far too large to read here. Percentiles rather than min/max so a
    handful of saturated voxels (very common in EM) don't flatten everything
    else into a narrow band, which is what makes the default 0-255 window look
    washed out on real data.

    ``paths`` is ordered fine -> coarse. Returns (lo, hi), or None to let the
    caller fall back.
    """
    MIN_VOXELS = 4096
    MAX_VOXELS = 8_000_000
    for path in reversed(paths):
        try:
            image = ImageDataInterface(path, normalize=normalize)
            n_voxels = int(np.prod(image.shape))
            if n_voxels < MIN_VOXELS:
                continue  # too small to be representative; try a finer level
            if n_voxels > MAX_VOXELS:
                break  # finer levels are only bigger -- stop rather than read them
            # Index the store directly, the way neuroglancer's LocalVolume
            # does, so LazyNormalization.__getitem__ runs and the sample is in
            # the same space as what gets displayed. to_ndarray_ts() would
            # silently give unnormalized data here: its roi=None branch returns
            # the underlying store's read() without applying g.input_norms
            # (only the roi branch does), so the percentiles would land in raw
            # uint8 space while the layer shows [-1, 1] -- a 0-168 range over
            # data that never exceeds 1, i.e. an all-black image.
            arr = np.asarray(image.ts[...])
            arr = arr[np.isfinite(arr)]
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


def _raw_shader(paths, normalize, image_for_fallback=None):
    """Build the raw-layer shader, preferring a data-derived contrast range;
    else the range of ``image_for_fallback``'s dtype, else 0-255."""
    rng = _auto_contrast_range(paths, normalize) if paths else None
    if rng is not None:
        lo, hi = rng
        # Let the slider reach well beyond the auto range so it stays adjustable.
        pad = (hi - lo) * 0.5 or 1.0
        wlo, whi = lo - pad, hi + pad
        logger.info(f"Raw auto-contrast range [{lo:.6g}, {hi:.6g}]")
    else:
        lo, hi = (
            _dtype_default_range(image_for_fallback)
            if image_for_fallback is not None
            else (0.0, 255.0)
        )
        wlo, whi = lo, hi
        logger.info(f"Raw auto-contrast unavailable; using [{lo:.6g}, {hi:.6g}]")
    return RAW_SHADER.format(lo=lo, hi=hi, wlo=wlo, whi=whi)


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


def _sn_arrays(group):
    """The sN children of ``group``, s0 first; none when it has none or is
    not a directory or group."""
    try:
        if paths.is_remote(group):
            names = list(open_zarr(group, mode="r").keys())
        else:
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


def get_raw_layer(
    dataset_path, normalize=True, wrap_raw=True, segmentation=False, disable_meshes=False
):
    """A neuroglancer layer showing a zarr, n5 or precomputed volume.

    A multiscale pyramid, or any of its levels, is shown as the whole
    pyramid (see _pyramid for what its levels are). ``segmentation`` gives a
    SegmentationLayer over the same source, placed the same way, for a
    label volume; its ids are served as stored, never through the input
    normalizers. ``disable_meshes`` then turns off its meshes subsource
    (see _layer).
    """
    dataset_path = dataset_path.replace("\\ ", " ")
    original_dataset_path = dataset_path
    is_precomputed = dataset_path.startswith("precomputed://")
    pyramid = _pyramid(dataset_path)
    if pyramid is not None:
        dataset_path, scales = pyramid

    if is_precomputed:
        filetype = "precomputed"
    elif ".zarr" in dataset_path or paths.is_zarr_container(dataset_path):
        filetype = "zarr"
    elif ".n5" in dataset_path:
        filetype = "n5"
    else:
        filetype = "precomputed"

    if not wrap_raw:
        if is_precomputed:
            source = dataset_path
        else:
            source = f"{filetype}://{dataset_path}"
        # Unwrapped: neuroglancer fetches the file directly, so the input
        # normalizers never run on what it displays. Sample unnormalized
        # too, or the contrast range lands in the wrong space entirely.
        return _layer(
            source,
            lambda: _raw_shader([original_dataset_path], normalize=False),
            segmentation,
            disable_meshes,
        )

    if pyramid is not None:
        try:
            images = [
                ImageDataInterface(
                    paths.join(dataset_path, scale), normalize=normalize and not segmentation
                )
                for scale in scales
            ]
            layers = [_local_volume(image) for image in images]
            # ScalePyramid serves every level as a downsampling of the finest
            # one, so the finest level's corner places them all. That holds
            # for pyramids whose levels share a corner (Janelia's all sit at
            # -4 nm); a level with a different corner cannot be expressed.
            finest = min(images, key=lambda image: tuple(image.voxel_size))

            # Previously this branch set no shader at all, which neuroglancer
            # reports back as the literal string "None" -- see the guard in
            # dashboard/routes/pipeline.py.
            return _layer(
                neuroglancer.LayerDataSource(
                    url=ScalePyramid(layers), transform=_corner_transform(finest)
                ),
                lambda: _raw_shader([paths.join(dataset_path, sc) for sc in scales], normalize),
                segmentation,
                disable_meshes,
            )
        except Exception as e:
            logger.error(e)

    # One array, or a pyramid that could not be shown as one. An image here
    # reads through the input chain whatever normalize says, as it always has.
    image = ImageDataInterface(original_dataset_path, normalize=not segmentation)
    return _layer(
        neuroglancer.LayerDataSource(
            url=_local_volume(image), transform=_corner_transform(image)
        ),
        lambda: _raw_shader([original_dataset_path], normalize, image_for_fallback=image),
        segmentation,
        disable_meshes,
    )


def _dimensions(image):
    return neuroglancer.CoordinateSpace(
        names=image.axes_names, units="nm", scales=image.voxel_size
    )


def _local_volume(image):
    """``image`` as a LocalVolume indexed from 0; _corner_transform places it."""
    return neuroglancer.LocalVolume(
        data=image.ts,
        dimensions=_dimensions(image),
        voxel_offset=[0] * len(image.axes_names),
    )


def _corner_transform(image):
    """The transform putting ``image``'s voxel 0 lower corner at its offset.

    LocalVolume's own voxel_offset is a whole number of voxels, so it cannot
    hold an OME corner such as -4 nm at 8 nm -- and it was being handed the
    offset in nm, which drew any dataset not at the origin far from its data.
    The matrix's translation is in voxels of the output dimensions.
    """
    rank = len(image.axes_names)
    offset = np.asarray(image.offset, dtype=float)
    voxel_size = np.asarray(image.voxel_size, dtype=float)
    matrix = np.hstack([np.eye(rank), np.zeros((rank, 1))])
    matrix[rank - len(offset):, rank] = offset / voxel_size
    return neuroglancer.CoordinateSpaceTransform(
        output_dimensions=_dimensions(image), matrix=matrix
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
