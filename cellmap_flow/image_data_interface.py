import copy
import functools
import logging

import numpy as np
from funlib.geometry import Coordinate

from cellmap_flow.io import multiscale
from cellmap_flow.io.geometry import Box, Grid, coordinate_or_floats
from cellmap_flow.io.metadata import snap_integral
from cellmap_flow.io.source import open_array, read_padded
from cellmap_flow.utils.ds import LazyNormalization, apply_norms, read_ds_meta

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
        reads use (see ``io.source.open_array``). The defaults, one reader
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
        actual_voxel_size = snap_integral(actual_voxel_size)
        actual_offset = snap_integral(actual_offset)
        # What the data really is, and what the caller asked for; voxel_size
        # below is the one reads are done in.
        self.actual_voxel_size = coordinate_or_floats(
            actual_voxel_size, "voxel size", dataset_path
        )
        self.requested_voxel_size = (
            None
            if voxel_size is None
            else coordinate_or_floats(voxel_size, "voxel size", dataset_path)
        )
        voxel_size_f, offset_f = actual_voxel_size, actual_offset
        if voxel_size is not None:
            requested = snap_integral(voxel_size)
            if not multiscale.same_voxel_size(requested, actual_voxel_size):
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
        # The grid reads are done on, in exact floats: voxel_size and offset
        # below are Coordinates when whole, and roi is the whole-nm box
        # around the data.
        self._grid = Grid(
            tuple(float(v) for v in voxel_size_f), tuple(float(v) for v in offset_f)
        )
        self.voxel_size = coordinate_or_floats(voxel_size_f, "voxel size", dataset_path)
        self.offset = coordinate_or_floats(offset_f, "offset", dataset_path)
        self.roi = self._grid.box_to_world(Box((0,) * len(shape), tuple(shape)))
        self.custom_fill_value = custom_fill_value
        self.concurrency_limit = concurrency_limit
        self.cache_bytes = cache_bytes
        # The array as stored (every channel, not normalized), opened on the
        # first read and shared with every with_input_norms() view.
        self.source = open_array(dataset_path, concurrency_limit, cache_bytes)
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

    def _view(self):
        return LazyNormalization(
            self.source.ts,
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
        """``roi`` (world nm; all of the array when None) read through the
        input chain, as a numpy array.

        Where the ROI runs past the array it is padded with
        ``custom_fill_value`` (0 when unset; "edge" repeats the border
        voxels), after the chain: padding is never normalized. A ROI is read
        at ``output_voxel_size`` when that differs from the voxel size (see
        ``_read_resampled``); the whole array never is.
        """
        view = self._view()
        store = view.selected()
        through_chain = functools.partial(apply_norms, input_norms=view.norms_to_apply())
        fill = self.custom_fill_value if self.custom_fill_value else 0
        if roi is None:
            return read_padded(store, None, fill, through_chain)
        if multiscale.same_voxel_size(self._grid.voxel_size, self.output_voxel_size):
            return read_padded(store, self._grid.world_to_box(roi), fill, through_chain)
        return self._read_resampled(store, roi, fill, through_chain)

    def _read_resampled(self, store, roi, fill, through_chain):
        """``roi`` at ``output_voxel_size`` (the K18 resampling kwargs).

        Both voxel sizes are taken as whole nanometers, and the factor is the
        z axis's, applied to every axis: a finer output repeats each voxel, a
        coarser one takes each block's median, and the same z voxel size
        resamples nothing (the widened read is returned as it is). The read
        is widened to whole voxels of the grid anchored at 0 nm, not at the
        dataset's corner, and cropped back to ``roi`` after resampling.
        """
        voxel_size = Coordinate(self._grid.voxel_size)
        output_voxel_size = Coordinate(self.output_voxel_size)
        widened = roi.snap_to_grid(voxel_size)
        factor = voxel_size[0] / output_voxel_size[0]
        begin = (roi.begin - widened.begin) / output_voxel_size
        end = (roi.end - widened.begin) / output_voxel_size
        crop = tuple(slice(begin[i], end[i]) for i in range(3))

        grid = Grid(tuple(float(v) for v in voxel_size), self._grid.translation)
        data = read_padded(store, grid.world_to_box(widened), fill, through_chain)
        if factor > 1:
            repeat = int(voxel_size[0] / output_voxel_size[0])
            return np.kron(data, np.ones((repeat,) * 3, dtype=data.dtype))[crop]
        if factor < 1:
            from skimage.measure import block_reduce

            return block_reduce(data, block_size=int(1 / factor), func=np.median)[crop]
        return data
