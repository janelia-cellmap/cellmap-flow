# copied from https://github.com/funkelab/funlib.show.neuroglancer/blob/master/funlib/show/neuroglancer/scale_pyramid.py

import neuroglancer
import operator
import logging
import numpy as np
import os

import zarr

from cellmap_flow.image_data_interface import ImageDataInterface
from cellmap_flow.utils.ds import (
    _is_remote_path,
    _is_zarr_container,
    _join_path,
    _open_zarr,
    check_for_multiscale,
    get_ds_info,
)
from cellmap_flow.utils import zarr_v3

logger = logging.getLogger(__name__)


# Shader for the raw EM layer. ``range`` is the displayed contrast window;
# ``window`` is the wider span the UI slider can be dragged over.
RAW_SHADER = """#uicontrol invlerp normalized(range=[{lo:.6g}, {hi:.6g}], window=[{wlo:.6g}, {whi:.6g}]);
#uicontrol vec3 color color(default="white");
void main(){{{{emitRGB(color * normalized());}}}}"""


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
    """Fallback display range when percentiles can't be computed."""
    try:
        kind = np.dtype(image.dtype).kind
        if kind == "u":
            info = np.iinfo(image.dtype)
            return float(info.min), float(info.max)
        if kind == "i":
            info = np.iinfo(image.dtype)
            return float(info.min), float(info.max)
    except Exception:
        pass
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
    """Build the raw-layer shader, preferring a data-derived contrast range."""
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


def get_raw_layer(dataset_path, normalize=True, wrap_raw=True):
    dataset_path = dataset_path.replace("\\ ", " ")
    original_dataset_path = dataset_path
    is_precomputed = dataset_path.startswith("precomputed://")
    # if multiscale dataset
    if is_precomputed:
        # precomputed format handles scales internally via tensorstore
        is_multiscale = False
    elif (
        dataset_path.split("/")[-1].startswith("s")
        and dataset_path.split("/")[-1][1:].isdigit()
    ):
        dataset_path = dataset_path.rsplit("/", 1)[0]
        is_multiscale = True
    else:
        try:
            v3_container = zarr_v3.find_v3_container(dataset_path)
            if v3_container is not None:
                is_multiscale = zarr_v3.multiscales_from_group(v3_container) is not None
            else:
                is_multiscale = check_for_multiscale(_open_zarr(dataset_path, mode="r"))[0]
        except Exception as e:
            logger.error(e)
            is_multiscale = False

    if is_precomputed:
        filetype = "precomputed"
    elif ".zarr" in dataset_path or _is_zarr_container(dataset_path):
        filetype = "zarr"
    elif ".n5" in dataset_path:
        filetype = "n5"
    else:
        filetype = "precomputed"

    layers = []
    if not wrap_raw:
        if is_precomputed:
            source = dataset_path
        else:
            source = f"{filetype}://{dataset_path}"
        return neuroglancer.ImageLayer(
            source=source,
            # Unwrapped: neuroglancer fetches the file directly, so the input
            # normalizers never run on what it displays. Sample unnormalized
            # too, or the contrast range lands in the wrong space entirely.
            shader=_raw_shader([original_dataset_path], normalize=False),
        )

    if is_multiscale:
        try:
            if _is_remote_path(dataset_path):
                grp = _open_zarr(dataset_path, mode="r")
                multiscales = grp.attrs.get("multiscales", None)
                if multiscales:
                    scales = [d["path"] for d in multiscales[0]["datasets"]]
                else:
                    scales = sorted(
                        [k for k in grp.keys() if k.startswith("s") and k[1:].isdigit()],
                        key=lambda x: int(x[1:]),
                    )
            else:
                scales = [
                    f for f in os.listdir(dataset_path) if f[0] == "s" and f[1:].isdigit()
                ]
                scales.sort(key=lambda x: int(x[1:]))
            for scale in scales:
                image = ImageDataInterface(
                    _join_path(dataset_path, scale), normalize=normalize
                )
                # Use axes from the actual dataset - neuroglancer will use them as-is
                layers.append(
                    neuroglancer.LocalVolume(
                        data=image.ts,
                        dimensions=neuroglancer.CoordinateSpace(
                            names=image.axes_names,
                            units="nm",
                            scales=image.voxel_size,
                        ),
                        voxel_offset=image.offset,
                    )
                )

            # Previously this branch set no shader at all, which neuroglancer
            # reports back as the literal string "None" -- see the guard in
            # dashboard/routes/pipeline.py.
            return neuroglancer.ImageLayer(
                dict(type=neuroglancer.LocalVolume, source=ScalePyramid(layers)),
                shader=_raw_shader(
                    [_join_path(dataset_path, sc) for sc in scales], normalize
                ),
            )
        except Exception as e:
            logger.error(e)
            is_multiscale = False

    if not is_multiscale:
        image = ImageDataInterface(original_dataset_path)
        return neuroglancer.ImageLayer(
            source=neuroglancer.LocalVolume(
                data=image.ts,
                dimensions=neuroglancer.CoordinateSpace(
                    names=image.axes_names,
                    units="nm",
                    scales=image.voxel_size,
                ),
                voxel_offset=image.offset,
            ),
            shader=_raw_shader(
                [original_dataset_path], normalize, image_for_fallback=image
            ),
        )


class ScalePyramid(neuroglancer.LocalVolume):
    """A neuroglancer layer that provides volume data on different scales.
    Mimics a LocalVolume.

    Args:

            volume_layers (``list`` of ``LocalVolume``):

                One ``LocalVolume`` per provided resolution.
    """

    def __init__(self, volume_layers):
        volume_layers = volume_layers

        super(neuroglancer.LocalVolume, self).__init__()

        logger.info("Creating scale pyramid...")

        self.min_voxel_size = min(
            [tuple(layer.dimensions.scales) for layer in volume_layers]
        )
        self.max_voxel_size = max(
            [tuple(layer.dimensions.scales) for layer in volume_layers]
        )

        self.dims = len(volume_layers[0].dimensions.scales)
        self.volume_layers = {
            tuple(
                int(x)
                for x in map(
                    operator.truediv, layer.dimensions.scales, self.min_voxel_size
                )
            ): layer
            for layer in volume_layers
        }

        logger.info("min_voxel_size: %s", self.min_voxel_size)
        logger.info("scale keys: %s", self.volume_layers.keys())
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
            "maxDownsampling": int(
                np.prod(np.array(self.max_voxel_size) // np.array(self.min_voxel_size))
            ),
            "maxDownsampledSize": reference_info["maxDownsampledSize"],
            "maxDownsamplingScales": reference_info["maxDownsamplingScales"],
        }

        return info

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
