import os
import zarr
from cellmap_flow.utils.ds import (
    find_closest_scale,
    get_ds_info,
    open_ds_tensorstore,
    to_ndarray_tensorstore,
)
import logging

logger = logging.getLogger(__name__)


class ImageDataInterface:
    def __init__(
        self,
        dataset_path,
        voxel_size=None,
        mode="r",
        output_voxel_size=None,
        custom_fill_value=None,
        concurrency_limit=1,
        normalize=True,
    ):
        # if multiscale dataset, get scale for voxel size
        if not isinstance(zarr.open(dataset_path, mode="r"), zarr.core.Array):
            scale, _, _ = find_closest_scale(dataset_path, voxel_size)
            logger.info(f"found scale {scale} for voxel size {voxel_size}")
            dataset_path = os.path.join(dataset_path, scale)
            logger.info(f"using dataset path {dataset_path}")
        self.path = dataset_path
        self.filetype = (
            "zarr" if dataset_path.rfind(".zarr") > dataset_path.rfind(".n5") else "n5"
        )
        self.swap_axes = self.filetype == "n5"
        self._ts = None

        self.voxel_size, self.chunk_shape, self.shape, self.roi, self.swap_axes = (
            get_ds_info(dataset_path)
        )
        if voxel_size is not None:
            self.voxel_size = voxel_size
        self.offset = self.roi.offset
        self.custom_fill_value = custom_fill_value
        self.concurrency_limit = concurrency_limit
        if output_voxel_size is not None:
            self.output_voxel_size = output_voxel_size
        else:
            self.output_voxel_size = self.voxel_size
        self.normalize = normalize

    @property
    def ts(self):
        if not self._ts:
            self._ts = open_ds_tensorstore(
                self.path,
                concurrency_limit=self.concurrency_limit,
                normalize=self.normalize,
            )
        return self._ts

    def to_ndarray_ts(self, roi=None):
        res = to_ndarray_tensorstore(
            self.ts,
            roi,
            self.voxel_size,
            self.offset,
            self.output_voxel_size,
            self.swap_axes,
            self.custom_fill_value,
        )
        return res
