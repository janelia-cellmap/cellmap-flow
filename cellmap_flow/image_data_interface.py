"""A dataset to read in world coordinates, through the input chain.

``ImageDataInterface(path, voxel_size)`` picks the level of a multiscale
group or precomputed volume for ``voxel_size`` (``io.multiscale``), reads
its metadata (``io.metadata``) and keeps its voxel grid (``io.geometry``)
and the array (``io.source``). A read of a world ROI is the grid's box of voxels, read
with padding where it runs past the array. With
``on_voxel_size_mismatch="resample"``, a dataset with no level at
``voxel_size`` is resampled to it (``io.resample``); the deprecated
``output_voxel_size`` resamples what is read, crudely. What is read goes through the input
chain on the way: the normalizers and ChannelSelector given as
``input_norms``, else the process's chain (``process_chain().input_norms``)
as it is at read time, which user ``process_chunk`` scripts rely on.
"""

import copy
import functools
import logging
import warnings

import numpy as np
import tensorstore as ts
from funlib.geometry import Coordinate

from cellmap_flow.io import multiscale, paths
from cellmap_flow.io.geometry import Box, Grid, coordinate_or_floats
from cellmap_flow.io.metadata import ArrayMeta, read_array_meta, snap_integral
from cellmap_flow.io.ome import CHANNEL_AXIS_NAMES
from cellmap_flow.io.resample import Resampling, is_label_dtype
from cellmap_flow.io.source import open_array, read_padded
from cellmap_flow.process_chain import process_chain

logger = logging.getLogger(__name__)

# (path, requested voxel size, on_voxel_size_mismatch) already warned about,
# or told about being resampled; one of these is built per extracted chunk in
# some paths.
_warned_relabel = set()

ON_VOXEL_SIZE_MISMATCH = ("relabel", "resample", "error")

# What the relabel warning suggests instead.
RESAMPLE_HINT = (
    'on_voxel_size_mismatch="resample" (`resample: true` in a YAML, `--resample` '
    "on the command line) resamples it to that voxel size instead"
)

# The arguments that warn when passed, and go in the next release (K18),
# with what to do instead. Nothing in cellmap-flow passes them.
_DEPRECATED = {
    "output_voxel_size": (
        'use voxel_size with on_voxel_size_mismatch="resample" (`resample: true` in a YAML, '
        "`--resample` on the command line), which resamples every axis by its own factor"
    ),
    "custom_fill_value": "read within the dataset's roi and pad what is read",
}


def legacy_meta(meta: ArrayMeta):
    """``(voxel_size, offset, chunk_shape, shape, axes_names, filetype)``, the
    tuple the old readers returned, from an ArrayMeta: spatial axes only,
    voxel size and offset as lists of nanometer floats.

    The local zarr v2/N5 reader always took the *last* n axes (n spatial
    ones) and called them z, y, x, whatever the metadata said; the others
    take the spatial axes by name. Both are kept.
    """
    spatial = meta.spatial()
    n = len(spatial.voxel_size)
    if meta.format in ("zarr2", "n5") and not paths.is_remote(meta.path):
        chunk_shape = tuple(meta.chunk_shape[-n:])
        shape = tuple(meta.shape[-n:])
        axes_names = ["z", "y", "x"][-n:]
    else:
        chunk_shape, shape = spatial.chunk_shape, spatial.shape
        axes_names = [a for a in spatial.axes if a != ""]
    if meta.format == "precomputed":
        filetype = "gs" if meta.path.startswith("gs://") else "precomputed"
    else:
        filetype = "n5" if meta.format == "n5" else "zarr"
    return (
        list(spatial.voxel_size),
        list(spatial.translation),
        chunk_shape,
        shape,
        axes_names,
        filetype,
    )


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
        ``None`` follows the process's chain (``process_chain().input_norms``)
        at read time.

        ``concurrency_limit`` and ``cache_bytes`` go to the tensorstore the
        reads use (see ``io.source.open_array``). The defaults, one reader
        thread and no cache, are what every caller has always had; the
        inference server asks for parallel reads and a cache.

        ``output_voxel_size`` (reads resampled to it, see ``_read_resampled``)
        and ``custom_fill_value`` (padding with that value, or "edge", instead
        of 0) are deprecated: passing either warns, and they go in the next
        release.

        ``voxel_size`` picks the scale of a multiscale group or precomputed
        volume (the finest one not coarser than it); the path of one scale is
        read at that scale. When the array opened is at a different voxel
        size, ``on_voxel_size_mismatch`` decides:

        - "relabel" (the default, with a warning) reads it as if it were at
          ``voxel_size``, voxel for voxel. The data then really lies at
          ``actual_voxel_size``, and ``offset`` is its corner rescaled.
        - "resample" reads it resampled to ``voxel_size`` (``io.resample``),
          from the level ``select_level``'s "resample" mode picks: the
          coarsest not coarser than ``voxel_size`` on any axis, else the
          finest. The data then really is at ``voxel_size``, ``offset`` is
          the level's own corner and ``shape`` the resampled one.
          ``resampled`` says which of the two a mismatch got.
        - "error" raises.

        ``actual_voxel_size`` (the level read) and ``requested_voxel_size``
        record both sizes either way.
        """
        resample = on_voxel_size_mismatch == "resample"
        if resample and output_voxel_size is not None:
            raise ValueError('output_voxel_size cannot be combined with on_voxel_size_mismatch="resample"')
        passed = {"output_voxel_size": output_voxel_size, "custom_fill_value": custom_fill_value}
        for name, value in passed.items():
            if value is not None:
                warnings.warn(
                    f"ImageDataInterface's {name} is deprecated and goes in the next release: "
                    f"{_DEPRECATED[name]}",
                    DeprecationWarning,
                    stacklevel=2,
                )
        dataset_path = paths.normalize_path(dataset_path)
        # A multiscale group or precomputed volume is read at its level for
        # voxel_size; the path of one level at that level.
        try:
            resolved, scale = multiscale.select_dataset(
                dataset_path, voxel_size, mode="resample" if resample else "floor"
            )
            if scale is not None:
                logger.info(f"found scale {scale} for voxel size {voxel_size}")
                dataset_path = resolved
                logger.info(f"using dataset path {dataset_path}")
        except Exception as e:
            logger.warning(f"could not open dataset {dataset_path} to find scale: {e}")
        self.path = dataset_path
        self.input_norms = None if input_norms is None else list(input_norms)
        meta = read_array_meta(dataset_path)
        (
            actual_voxel_size,
            actual_offset,
            self.chunk_shape,
            shape,
            self.axes_names,
            self.filetype,
        ) = legacy_meta(meta)
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
        # How a read is resampled to voxel_size, when it is (see to_ndarray_ts).
        self._resampling = None
        if voxel_size is not None:
            requested = snap_integral(voxel_size)
            if not multiscale.same_voxel_size(requested, actual_voxel_size):
                message = (
                    f"{dataset_path} is at {tuple(actual_voxel_size.tolist())} nm "
                    f"but {tuple(requested.tolist())} nm was requested"
                )
                if on_voxel_size_mismatch == "error":
                    raise ValueError(message)
                if on_voxel_size_mismatch not in ON_VOXEL_SIZE_MISMATCH:
                    raise ValueError(
                        "on_voxel_size_mismatch must be one of "
                        f"{', '.join(map(repr, ON_VOXEL_SIZE_MISMATCH))}, got {on_voxel_size_mismatch!r}"
                    )
                key = (dataset_path, tuple(requested.tolist()), on_voxel_size_mismatch)
                if resample:
                    # The resampled grid starts at the level's own corner, so
                    # offset_f stays the level's; only the voxels change.
                    self._resampling = Resampling(
                        tuple(float(v) for v in actual_voxel_size),
                        tuple(float(v) for v in requested),
                        tuple(int(n) for n in shape),
                    )
                    methods = self._resampling.methods(is_label_dtype(meta.dtype))
                    shape = self._resampling.shape
                    if key not in _warned_relabel:
                        _warned_relabel.add(key)
                        logger.info(
                            f"{message}; resampling it ({', '.join(f'{a} {m}' for a, m in zip(self.axes_names, methods))})"
                        )
                else:
                    if key not in _warned_relabel:
                        _warned_relabel.add(key)
                        logger.warning(
                            f"{message}; reading it as if it were "
                            f"{tuple(requested.tolist())} nm (the data is not resampled; "
                            f"{RESAMPLE_HINT})"
                        )
                    # Relabel on the real grid: voxel i stays voxel i and only its
                    # size changes, so the offset scales with it. Keeping the real
                    # offset against the requested voxel size mixed two unit
                    # systems in (roi - offset) / voxel_size and shifted every read
                    # of an offset dataset.
                    offset_f = actual_offset / actual_voxel_size * requested
            voxel_size_f = requested
        self.shape = Coordinate(shape)
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

    @property
    def resampled(self) -> bool:
        """Whether reads are resampled from ``actual_voxel_size`` to
        ``voxel_size``: the data then really is at ``voxel_size``. When the
        two differ and this is False, the level was relabelled, and the data
        really lies at ``actual_voxel_size``."""
        return self._resampling is not None

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

        A resampled dataset has none: its voxels exist only as
        ``to_ndarray_ts`` computes them. ``source.ts`` is the level as stored.
        """
        if self.resampled:
            raise AttributeError(
                f"{self.path} is resampled from {tuple(self.actual_voxel_size)} nm to "
                f"{tuple(self.voxel_size)} nm, so it is read with to_ndarray_ts(); "
                "source.ts is the stored level"
            )
        view = self._view()
        return view if self.normalize else view.selected()

    def with_input_norms(self, input_norms):
        """This dataset, read through ``input_norms`` instead of the process's chain.

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
        voxels), after the chain: padding is never normalized.

        A resampled dataset (``resampled``) is read at ``voxel_size`` by
        ``io.resample``: the stored voxels are resampled, then go through the
        chain, as a stored level at ``voxel_size`` would. A ROI is read at
        the deprecated ``output_voxel_size`` when that differs from the voxel
        size (see ``_read_resampled``); the whole array never is.
        """
        view = self._view()
        store = view.selected()
        through_chain = functools.partial(apply_norms, input_norms=view.norms_to_apply())
        fill = self.custom_fill_value if self.custom_fill_value else 0
        if self.resampled:
            box = Box((0,) * len(self.shape), tuple(self.shape)) if roi is None else self._grid.world_to_box(roi)
            return self._resampling.read(store, box, fill, through_chain)
        if roi is None:
            return read_padded(store, None, fill, through_chain)
        if multiscale.same_voxel_size(self._grid.voxel_size, self.output_voxel_size):
            return read_padded(store, self._grid.world_to_box(roi), fill, through_chain)
        return self._read_resampled(store, roi, fill, through_chain)

    def _read_resampled(self, store, roi, fill, through_chain):
        """``roi`` at ``output_voxel_size`` (deprecated, K18).

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


# --- what a read sees: one channel, through the input chain -----------------


def apply_norms(data, input_norms=None):
    """Read ``data`` if it is a tensorstore view and run it through the chain.

    ``input_norms=None`` means the process's chain, ``process_chain().input_norms``.
    """
    if hasattr(data, "read"):
        data = data.read().result()
    for norm in process_chain().input_norms if input_norms is None else input_norms:
        data = norm(data)
    return data


def selected_channel(input_norms) -> int:
    """The input channel a chain asks for: its first ChannelSelector, else 0."""
    from cellmap_flow.norm.input_normalize import ChannelSelector

    for norm in input_norms or []:
        if isinstance(norm, ChannelSelector):
            return norm.channel
    return 0


def select_channel(ts_dataset, channel=0, spatial_ndim=3):
    """Index one channel out of a multichannel tensorstore.

    The channel axis is the one labelled c/c^/channel when the store labels its
    dimensions, and otherwise the first one (the OME-Zarr convention).
    Arrays with no more than ``spatial_ndim`` dimensions are returned as is.
    """
    if ts_dataset.ndim <= spatial_ndim:
        return ts_dataset
    labels = list(getattr(ts_dataset.domain, "labels", None) or [])
    axis = next((i for i, lab in enumerate(labels) if lab in CHANNEL_AXIS_NAMES), 0)
    if axis == 0:
        return ts_dataset[channel]
    return ts_dataset[ts.d[axis][channel]]


class LazyNormalization:
    """A tensorstore seen through the input chain, for neuroglancer to index.

    The channel and the normalizers are looked up on every access rather than
    fixed when the store was opened: a server that has already read one chunk
    must still follow a ChannelSelector that changes afterwards.

    ``input_norms=None`` follows the process's chain,
    ``process_chain().input_norms``, as it is at each access: a worker that
    unpickles one reads its own chain, not the sender's.
    ``normalize=False`` selects the channel but applies no normalizers.
    """

    def __init__(self, ts_dataset, input_norms=None, normalize=True, spatial_ndim=3):
        self.ts_dataset = ts_dataset
        self.input_norms = input_norms
        self.normalize = normalize
        self.spatial_ndim = spatial_ndim

    def chain(self):
        if self.input_norms is None:
            return list(process_chain().input_norms)
        return list(self.input_norms)

    def norms_to_apply(self):
        return self.chain() if self.normalize else []

    def selected(self):
        """The raw (unnormalized) tensorstore for the channel the chain selects."""
        return select_channel(
            self.ts_dataset, selected_channel(self.chain()), self.spatial_ndim
        )

    def __getitem__(self, index):
        result = self.selected()[index]
        if not self.normalize:
            return result
        return apply_norms(result, self.norms_to_apply())

    def __getattr__(self, attr):
        if attr in ("ts_dataset", "input_norms", "normalize", "spatial_ndim"):
            # Not set yet (e.g. while unpickling); don't recurse.
            raise AttributeError(attr)
        at = getattr(self.selected(), attr)
        if attr == "dtype":
            # The last step that declares a dtype decides; steps without one
            # (ChannelSelector) pass their input's through.
            for norm in reversed(self.norms_to_apply()):
                if norm.dtype is not None:
                    return np.dtype(norm.dtype)
            return np.dtype(at.numpy_dtype)
        return at
