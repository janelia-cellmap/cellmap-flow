# copied from https://github.com/funkelab/funlib.show.neuroglancer/blob/master/funlib/show/neuroglancer/scale_pyramid.py

import neuroglancer
import operator
import logging
import numpy as np
import os

from cellmap_flow.image_data_interface import ImageDataInterface

logger = logging.getLogger(__name__)


def get_raw_layer(dataset_path):
    is_multiscale = False
    # if multiscale dataset
    if (
        dataset_path.split("/")[-1].startswith("s")
        and dataset_path.split("/")[-1][1:].isdigit()
    ):
        dataset_path = dataset_path.rsplit("/", 1)[0]
        is_multiscale = True

    if ".zarr" in dataset_path:
        filetype = "zarr"
    elif ".n5" in dataset_path:
        filetype = "n5"
    else:
        filetype = "precomputed"

    if filetype == "n5":
        axis = ["x", "y", "z"]
    else:
        axis = ["z", "y", "x"]

    layers = []

    if is_multiscale:
        scales = [
            f for f in os.listdir(dataset_path) if f[0] == "s" and f[1:].isdigit()
        ]
        scales.sort(key=lambda x: int(x[1:]))
        for scale in scales:
            image = ImageDataInterface(f"{os.path.join(dataset_path, scale)}")
            layers.append(
                neuroglancer.LocalVolume(
                    data=image.ts,
                    dimensions=neuroglancer.CoordinateSpace(
                        names=axis,
                        units="nm",
                        scales=(
                            image.voxel_size[::-1]
                            if filetype == "n5"
                            else image.voxel_size
                        ),
                    ),
                    voxel_offset=(
                        image.offset[::-1] if filetype == "n5" else image.offset
                    ),
                )
            )

        return neuroglancer.ImageLayer(
            dict(type=neuroglancer.LocalVolume, source=ScalePyramid(layers))
        )
    else:
        image = ImageDataInterface(dataset_path)
        return neuroglancer.ImageLayer(
            source=neuroglancer.LocalVolume(
                data=image.ts,
                dimensions=neuroglancer.CoordinateSpace(
                    names=axis,
                    units="nm",
                    scales=image.voxel_size,
                ),
                voxel_offset=image.offset,
            )
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
