import copy

import zarr
from cellmap_flow.utils.ds import (
    LazyNormalization,
    _join_path,
    _open_zarr,
    find_closest_scale,
    get_ds_info,
    open_ds_tensorstore,
    to_ndarray_tensorstore,
)
from cellmap_flow.utils import zarr_v3
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
        input_norms=None,
    ):
        """``input_norms``: the normalizers (and ChannelSelector) to read with.
        ``None`` follows the process-wide ``g.input_norms`` at read time.
        """
        dataset_path = dataset_path.replace("\\ ", " ")
        if not dataset_path.startswith("precomputed://"):
            v3_container = zarr_v3.find_v3_container(dataset_path)
            if v3_container is not None:
                try:
                    meta = zarr_v3.read_zarr_json(v3_container)
                    if meta.get("node_type") == "group":
                        scale, _, _ = zarr_v3.find_closest_scale_v3(
                            v3_container, voxel_size
                        )
                        logger.info(f"found scale {scale} for voxel size {voxel_size}")
                        dataset_path = _join_path(v3_container, scale)
                        logger.info(f"using dataset path {dataset_path}")
                except Exception as e:
                    logger.warning(
                        f"could not open v3 dataset {dataset_path} to find scale: {e}"
                    )
            else:
                try:
                    ds = _open_zarr(dataset_path, mode="r")
                    if isinstance(ds, zarr.hierarchy.Group):
                        scale, _, _ = find_closest_scale(dataset_path, voxel_size)
                        logger.info(f"found scale {scale} for voxel size {voxel_size}")
                        dataset_path = _join_path(dataset_path, scale)
                        logger.info(f"using dataset path {dataset_path}")
                except Exception as e:
                    logger.warning(f"could not open dataset {dataset_path} to find scale: {e}")
        self.path = dataset_path
        # The opened tensorstore is shared with every with_input_norms() view.
        self._store = {"ts": None}
        self.input_norms = None if input_norms is None else list(input_norms)
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
        # One of these is constructed per extracted chunk and once per pyramid
        # level at startup, so at WARNING this dumped the same multi-line dict
        # over and over and buried anything that actually needed attention.
        # It is useful when a dataset resolves to the wrong scale, which is a
        # debugging question.
        logger.debug(str(self.info))

    def _raw_ts(self):
        """The dataset as opened: unnormalized, every channel."""
        if self._store["ts"] is None:
            self._store["ts"] = open_ds_tensorstore(
                self.path,
                concurrency_limit=self.concurrency_limit,
                normalize=True,
            ).ts_dataset
        return self._store["ts"]

    def _view(self):
        return LazyNormalization(
            self._raw_ts(),
            input_norms=self.input_norms,
            normalize=self.normalize,
            spatial_ndim=len(self.shape),
        )

    @property
    def ts(self):
        """The dataset seen through the input chain (one channel, normalized).

        With ``normalize=False`` this is the plain tensorstore of the selected
        channel, as it always was.
        """
        view = self._view()
        return view if self.normalize else view.selected()

    def with_input_norms(self, input_norms):
        """This dataset, read through ``input_norms`` instead of ``g.input_norms``.

        Shares the opened tensorstore, so it is cheap to make one per request.
        """
        view = copy.copy(self)
        view.input_norms = list(input_norms)
        return view

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
        view = self._view()
        return to_ndarray_tensorstore(
            view.selected(),
            roi,
            self.voxel_size,
            self.offset,
            self.output_voxel_size,
            self.axes_names,
            self.custom_fill_value,
            input_norms=view.norms_to_apply(),
        )
