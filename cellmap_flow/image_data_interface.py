import copy

from cellmap_flow.io import multiscale
from cellmap_flow.utils.ds import (
    LazyNormalization,
    open_ds_tensorstore,
    read_ds_meta,
    to_ndarray_tensorstore,
)
from cellmap_flow.utils import zarr_v3
import logging
from funlib.geometry import Coordinate

logger = logging.getLogger(__name__)

# (path, requested voxel size) pairs already warned about; one of these is
# built per extracted chunk in some paths.
_warned_relabel = set()


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
        on_voxel_size_mismatch="relabel",
        cache_bytes=0,
    ):
        """``input_norms``: the normalizers (and ChannelSelector) to read with.
        ``None`` follows the process-wide ``g.input_norms`` at read time.

        ``concurrency_limit`` and ``cache_bytes`` go to the tensorstore the
        reads use (see ``open_ds_tensorstore``). The defaults, one reader
        thread and no cache, are what every caller has always had; the
        inference server asks for parallel reads and a cache.

        ``voxel_size`` picks the scale of a multiscale group (the finest one
        not coarser than it). When the array opened is at a different voxel
        size, ``on_voxel_size_mismatch`` decides: "relabel" (the default,
        with a warning) reads it as if it were at ``voxel_size``, voxel for
        voxel; "error" raises. ``actual_voxel_size`` and
        ``requested_voxel_size`` record both.
        """
        dataset_path = dataset_path.replace("\\ ", " ")
        if not dataset_path.startswith("precomputed://"):
            # A multiscale group is read at its level for voxel_size.
            try:
                resolved, scale = multiscale.select_dataset(dataset_path, voxel_size)
                if scale is not None:
                    logger.info(f"found scale {scale} for voxel size {voxel_size}")
                    dataset_path = resolved
                    logger.info(f"using dataset path {dataset_path}")
            except Exception as e:
                logger.warning(f"could not open dataset {dataset_path} to find scale: {e}")
        self.path = dataset_path
        # The opened tensorstore is shared with every with_input_norms() view.
        self._store = {"ts": None}
        self.input_norms = None if input_norms is None else list(input_norms)
        (
            actual_voxel_size,
            actual_offset,
            self.chunk_shape,
            shape,
            self.axes_names,
            self.filetype,
        ) = read_ds_meta(dataset_path)
        self.shape = Coordinate(shape)
        actual_voxel_size = zarr_v3.snap_integral(actual_voxel_size)
        actual_offset = zarr_v3.snap_integral(actual_offset)
        # What the data really is, and what the caller asked for; voxel_size
        # below is the one reads are done in.
        self.actual_voxel_size = zarr_v3.coordinate_or_floats(
            actual_voxel_size, "voxel size", dataset_path
        )
        self.requested_voxel_size = (
            None
            if voxel_size is None
            else zarr_v3.coordinate_or_floats(voxel_size, "voxel size", dataset_path)
        )
        voxel_size_f, offset_f = actual_voxel_size, actual_offset
        if voxel_size is not None:
            requested = zarr_v3.snap_integral(voxel_size)
            if not zarr_v3.same_voxel_size(requested, actual_voxel_size):
                message = (
                    f"{dataset_path} is at {tuple(actual_voxel_size.tolist())} nm "
                    f"but {tuple(requested.tolist())} nm was requested"
                )
                if on_voxel_size_mismatch == "error":
                    raise ValueError(message)
                if on_voxel_size_mismatch != "relabel":
                    raise ValueError(
                        f"on_voxel_size_mismatch must be 'relabel' or 'error', "
                        f"got {on_voxel_size_mismatch!r}"
                    )
                key = (dataset_path, tuple(requested.tolist()))
                if key not in _warned_relabel:
                    _warned_relabel.add(key)
                    logger.warning(
                        f"{message}; reading it as if it were "
                        f"{tuple(requested.tolist())} nm (the data is not resampled)"
                    )
                # Relabel on the real grid: voxel i stays voxel i and only its
                # size changes, so the offset scales with it. Keeping the real
                # offset against the requested voxel size mixed two unit
                # systems in (roi - offset) / voxel_size and shifted every read
                # of an offset dataset.
                offset_f = actual_offset / actual_voxel_size * requested
            voxel_size_f = requested
        self._voxel_size_f = voxel_size_f
        self._offset_f = offset_f
        self.voxel_size = zarr_v3.coordinate_or_floats(
            voxel_size_f, "voxel size", dataset_path
        )
        self.offset = zarr_v3.coordinate_or_floats(offset_f, "offset", dataset_path)
        self.roi = zarr_v3.covering_roi(offset_f, voxel_size_f, shape)
        self.custom_fill_value = custom_fill_value
        self.concurrency_limit = concurrency_limit
        self.cache_bytes = cache_bytes
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
                cache_bytes=self.cache_bytes,
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
            self._voxel_size_f,
            self._offset_f,
            self.output_voxel_size,
            self.axes_names,
            self.custom_fill_value,
            input_norms=view.norms_to_apply(),
        )
