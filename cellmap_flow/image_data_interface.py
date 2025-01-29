from cellmap_flow.utils.ds import (
    get_ds_info,
    open_ds_tensorstore,
    to_ndarray_tensorstore,
)


class ImageDataInterface:
    def __init__(
        self,
        dataset_path,
        target_resolution=None,
        mode="r",
        output_voxel_size=None,
        custom_fill_value=None,
        concurrency_limit=1,
    ):
        self.path = dataset_path
        self.filetype = (
            "zarr" if dataset_path.rfind(".zarr") > dataset_path.rfind(".n5") else "n5"
        )
        self.swap_axes = self.filetype == "n5"
        self.ts = None
        self.voxel_size, self.chunk_shape, self.shape, self.roi, self.swap_axes = (
            get_ds_info(dataset_path)
        )
        self.offset = self.roi.offset
        self.custom_fill_value = custom_fill_value
        self.concurrency_limit = concurrency_limit
        if output_voxel_size is not None:
            self.output_voxel_size = output_voxel_size
        else:
            self.output_voxel_size = self.voxel_size

    def to_ndarray_ts(self, roi=None):
        if not self.ts:
            self.ts = open_ds_tensorstore(
                self.path, concurrency_limit=self.concurrency_limit
            )
        res = to_ndarray_tensorstore(
            self.ts,
            roi,
            self.voxel_size,
            self.offset,
            self.output_voxel_size,
            self.swap_axes,
            self.custom_fill_value,
        )
        self.ts = None
        return res
