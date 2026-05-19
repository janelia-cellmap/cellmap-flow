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

logger = logging.getLogger(__name__)


def get_raw_layer(
    dataset_path,
    normalize=True,
    wrap_raw=True,
    segmentation=False,
    min_scale=0,
    disable_meshes=False,
):
    """Load a local/remote zarr/n5/precomputed volume as a neuroglancer layer.

    When `segmentation=True`, wraps the resulting ScalePyramid (or single
    LocalVolume) in a `neuroglancer.SegmentationLayer` instead of the
    default `ImageLayer`. Used by the `layer_type: segmentation` path in
    YAML extra_layers and the /api/viewer/add-segmentation-layer route.

    `disable_meshes=True` (segmentation only): turns OFF the meshes
    subsource on the layer's data source, so NG won't attempt on-the-fly
    marching-cubes mesh generation when segments are picked. Required
    when the catalog spans a whole-cell-scale label volume on a
    memory-tight node, where a single mesh request can OOM the
    dashboard process.

    `min_scale=N` (multiscale only): skip pyramid levels below `sN`
    (e.g. min_scale=2 drops s0/s1, keeps s2+). Useful when the highest
    resolution levels would saturate dashboard memory before the user
    has zoomed in.
    """
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
        if segmentation:
            if disable_meshes:
                return neuroglancer.SegmentationLayer(
                    source=neuroglancer.LayerDataSource(
                        url=source, subsources={"meshes": False},
                    ),
                )
            return neuroglancer.SegmentationLayer(source=source)
        return neuroglancer.ImageLayer(
            source=source,
            shader="""#uicontrol invlerp normalized(range=[0, 255], window=[0, 255]);
    #uicontrol vec3 color color(default="white");
    void main(){{emitRGB(color * normalized());}}""",
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
            if min_scale > 0:
                scales = [s for s in scales if int(s[1:]) >= min_scale]
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

            if segmentation:
                seg_layer = neuroglancer.SegmentationLayer(
                    dict(type=neuroglancer.LocalVolume,
                         source=ScalePyramid(layers)),
                )
                if disable_meshes:
                    # Post-construction tweak: passing subsources via the
                    # dict-positional form was silently dropped in the
                    # LayerDataSource coercion path. Setting on the
                    # already-built source[0] makes it survive to the
                    # browser-side state JSON.
                    seg_layer.source[0].subsources = {"meshes": False}
                return seg_layer
            return neuroglancer.ImageLayer(
                dict(type=neuroglancer.LocalVolume, source=ScalePyramid(layers))
            )
        except Exception as e:
            logger.error(e)
            is_multiscale = False

    if not is_multiscale:
        image = ImageDataInterface(original_dataset_path)
        local_volume = neuroglancer.LocalVolume(
            data=image.ts,
            dimensions=neuroglancer.CoordinateSpace(
                names=image.axes_names,
                units="nm",
                scales=image.voxel_size,
            ),
            voxel_offset=image.offset,
        )
        if segmentation:
            seg_layer = neuroglancer.SegmentationLayer(
                dict(type=neuroglancer.LocalVolume, source=local_volume),
            )
            if disable_meshes:
                seg_layer.source[0].subsources = {"meshes": False}
            return seg_layer
        return neuroglancer.ImageLayer(
            source=local_volume,
            shader="""#uicontrol invlerp normalized(range=[-1, 1], window=[-1, 1]);
    #uicontrol vec3 color color(default="white");
    void main(){{emitRGB(color * normalized());}}""",
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
