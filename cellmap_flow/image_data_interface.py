import os
import zarr
from cellmap_flow.utils.ds import (
    find_closest_scale,
    get_ds_info,
    open_ds_tensorstore,
    to_ndarray_tensorstore,
)
import logging
from funlib.geometry import Coordinate

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
        try:
            ds = zarr.open(dataset_path, mode="r")
            if isinstance(ds, zarr.hierarchy.Group):
                scale, _, _ = find_closest_scale(dataset_path, voxel_size)
                logger.info(f"found scale {scale} for voxel size {voxel_size}")
                dataset_path = os.path.join(dataset_path, scale)
                logger.info(f"using dataset path {dataset_path}")
        except Exception as e:
            logger.warning(f"could not open dataset {dataset_path} to find scale: {e}")
        self.path = dataset_path
        self._ts = None
        (
            self.voxel_size,
            self.chunk_shape,
            self.shape,
            self.roi,
            self.axes_names,
            self.filetype,
        ) = get_ds_info(dataset_path)
        if voxel_size is not None:
            self.voxel_size = Coordinate(voxel_size)
        self.offset = self.roi.offset
        self.custom_fill_value = custom_fill_value
        self.concurrency_limit = concurrency_limit
        if output_voxel_size is not None:
            self.output_voxel_size = Coordinate(output_voxel_size)
        else:
            self.output_voxel_size = self.voxel_size
        self.normalize = normalize
        logger.warning(str(self.info))

    @property
    def ts(self):
        if not self._ts:
            self._ts = open_ds_tensorstore(
                self.path,
                concurrency_limit=self.concurrency_limit,
                normalize=self.normalize,
            )
        return self._ts

    @property
    def info(self):
        info = {
            "path": self.path,
            "voxel_size": self.voxel_size,
            "chunk_shape": self.chunk_shape,
            "shape": self.shape,
            "roi": self.roi,
            "axes_names": self.axes_names,
            "filetype": self.filetype,
        }
        return info

    def to_ndarray_ts(self, roi=None):
        res = to_ndarray_tensorstore(
            self.ts,
            roi,
            self.voxel_size,
            self.offset,
            self.output_voxel_size,
            self.axes_names,
            self.custom_fill_value,
        )
        # logger.warning(f"Read data with shape {res.shape} from dataset {self.path} with ROI {roi} and output voxel size {self.output_voxel_size}")
        return res
