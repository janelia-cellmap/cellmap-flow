"""A dataset to read in world coordinates, through the input chain.

``ImageDataInterface`` is a ``worldarray.WorldArray`` (worldarray opens the
level, reads its metadata and reads its voxels, padded where a read runs past
the array, and resamples) that keeps cellmap-flow's reader on top:

- the level: ``ImageDataInterface(path, voxel_size)`` picks the level of a
  multiscale group or precomputed volume for ``voxel_size``
  (``io.multiscale``), but reads the path of one level at that level;
  worldarray's ``voxel_size`` would move to a sibling level instead;
- the grid reads are done on (``read_grid``, an ``io.geometry.Grid``: voxel size
  and voxel 0's lower corner in nm), and with it ``voxel_size``, ``offset``,
  ``roi`` and ``shape`` as cellmap-flow has them: a mismatched level is
  relabelled with its corner rescaled (worldarray keeps its centre), and a
  resampled one has the resampled shape (worldarray's ``shape`` stays the
  stored level's);
- the input chain: what is read goes through the normalizers and
  ChannelSelector given as ``input_norms``, else the process's chain
  (``process_chain().input_norms``) as it is at read time, which user
  ``process_chunk`` scripts rely on; padding is added after the chain;
- the deprecated ``output_voxel_size`` and ``custom_fill_value``.

``shape``, ``voxel_size`` and ``ts`` mean cellmap-flow's things on an
ImageDataInterface (the spatial, resampled shape; the relabelled voxel size as
a Coordinate when whole; the chain's view) and worldarray's on ``_array``, the
same object seen as a plain WorldArray, which every worldarray method runs on.
"""

import copy
import logging
import warnings

import numpy as np
import tensorstore as ts
import worldarray as wa
from funlib.geometry import Coordinate

from cellmap_flow.io import metadata, multiscale, paths
from cellmap_flow.io.geometry import Box, Grid, coordinate_or_floats
from cellmap_flow.io.metadata import ArrayMeta, snap_integral
from cellmap_flow.io.ome import CHANNEL_AXIS_NAMES
from cellmap_flow.io.resample import Resampling, is_label_dtype
from cellmap_flow.io.source import reader_context
from cellmap_flow.process_chain import process_chain

logger = logging.getLogger(__name__)

# (path, requested voxel size, on_voxel_size_mismatch) already warned about,
# or told about being resampled; one of these is built per extracted chunk in
# some paths.
_warned_relabel = set()

ON_VOXEL_SIZE_MISMATCH = ("relabel", "resample", "error")

# What a read past the array is padded with: 0, or the array's outermost
# voxels repeated ("edge"), which is what a model's input is read with.
PADDINGS = ("zero", "edge")

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

    The zarr v2/N5 reader always took the *last* n axes (n spatial ones)
    and called them z, y, x, whatever the metadata said, on disk; a URL now
    reads the same. The others take the spatial axes by name. Both are kept.
    """
    spatial = meta.spatial()
    n = len(spatial.voxel_size)
    if meta.format in ("zarr2", "n5"):
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


def _clipped(begin, end, size, edge):
    """Along one axis, for a box ``[begin, end)`` over an array of ``size``
    voxels: ``(start, stop, [before, after])``, the voxels to read and the
    padding that makes them the box's length. What is read is the box's own
    voxels in the array: none when the box lies wholly on one side of it,
    unless ``edge``, when it is the array's voxel nearest the box, to repeat."""
    length = end - begin
    start = min(max(begin, 0), size)
    stop = max(min(end, size), 0)
    if edge and start == stop and length:
        start, stop = (size - 1, size) if begin >= size else (0, 1)
    before = min(max(start - begin, 0), length - (stop - start))
    return start, stop, [before, length - (stop - start) - before]


class _StoredLevel:
    """``ImageDataInterface.source``: the level as stored (every channel, not
    normalized), shared with every ``with_input_norms()`` view."""

    def __init__(self, image):
        self._image = image

    @property
    def ts(self):
        """The tensorstore: C order, voxel 0 at index 0 (worldarray's ``ts``)."""
        return wa.WorldArray.ts.fget(self._image)

    def read(self, box=None, fill=0):
        array = self._image._array
        if box is None:
            return array.read_index((0,) * len(array.shape), array.shape)
        return array.read_index(box.begin, box.end, padding=fill)


class ImageDataInterface(wa.WorldArray):
    # cellmap-flow's meanings, set per instance: the spatial (resampled) shape,
    # and the voxel size reads are done in. Plain class attributes, so that
    # they shadow WorldArray's properties of the same names here and only here.
    shape = None
    voxel_size = None

    def __init__(
        self,
        dataset_path,
        voxel_size=None,
        mode="r",
        output_voxel_size=None,
        custom_fill_value=None,
        normalize=True,
        input_norms=None,
        on_voxel_size_mismatch="relabel",
        padding="zero",
    ):
        """``input_norms``: the normalizers (and ChannelSelector) to read with.
        ``None`` follows the process's chain (``process_chain().input_norms``)
        at read time.

        The reads go through one tensorstore reader thread and no cache
        (``io.source.reader_context``'s defaults), which is what blockwise
        workers, the finetune trainer and scripts want; the inference server
        reads through chunkmirage's own cache.

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
        - "resample" reads it resampled to ``voxel_size`` (worldarray's
          resampling), from the level ``select_level``'s "resample" mode
          picks: the coarsest not coarser than ``voxel_size`` on any axis,
          else the finest. The data then really is at ``voxel_size``,
          ``offset`` is the level's own corner and ``shape`` the resampled
          one. ``resampled`` says which of the two a mismatch got.
        - "error" raises.

        ``actual_voxel_size`` (the level read) and ``requested_voxel_size``
        record both sizes either way.

        ``padding`` is what a read past the array is padded with, after the
        chain: "zero", or "edge", the array's outermost voxels repeated. A
        model's input is read with "edge" (blockwise, the finetune trainer,
        as the inference server's BlockInput pads), since a flat fill is a
        wall the model never saw in training; labels with "zero".
        """
        if padding not in PADDINGS:
            raise ValueError(f"padding must be one of {PADDINGS}, not {padding!r}")
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
        # worldarray opens that level as it is (no voxel_size: the mismatch is
        # handled below); a precomputed scale by its key, a group by its first level.
        if paths.is_precomputed(dataset_path):
            location, level = metadata.precomputed_level(dataset_path)
        else:
            location, level = dataset_path, metadata.first_level(dataset_path)
        super().__init__(
            location,
            level=level,
            padding="edge" if padding == "edge" else 0,
            context=reader_context(),
        )
        self.path = dataset_path
        self.input_norms = None if input_norms is None else list(input_norms)
        meta = metadata.from_world_array(self._array, dataset_path)
        self.method = "nearest" if is_label_dtype(meta.dtype) else "linear"
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
                    # of an offset dataset. (worldarray's relabel keeps the
                    # level's centre instead, so its corner moves by half the
                    # difference of the voxel sizes.)
                    offset_f = actual_offset / actual_voxel_size * requested
            voxel_size_f = requested
        self.shape = Coordinate(shape)
        # The grid reads are done on, in exact floats: voxel_size and offset
        # below are Coordinates when whole, and roi is the whole-nm box
        # around the data.
        self.read_grid = Grid(
            tuple(float(v) for v in voxel_size_f), tuple(float(v) for v in offset_f)
        )
        self.voxel_size = coordinate_or_floats(voxel_size_f, "voxel size", dataset_path)
        self.offset = coordinate_or_floats(offset_f, "offset", dataset_path)
        self.roi = self.read_grid.box_to_world(Box((0,) * len(shape), tuple(shape)))
        self.custom_fill_value = custom_fill_value
        # The array as stored (every channel, not normalized), shared with
        # every with_input_norms() view.
        self.source = _StoredLevel(self)
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

    # --- worldarray's face ----------------------------------------------------

    @property
    def _array(self) -> wa.WorldArray:
        """This object as a plain WorldArray: the same attributes, worldarray's
        ``shape``, ``voxel_size`` and ``ts``. Every worldarray method runs on it."""
        view = wa.WorldArray.__new__(wa.WorldArray)
        view.__dict__ = self.__dict__
        return view

    def read_index(self, *args, **kwargs):
        """worldarray's ``read_index``: stored voxels, every axis, no chain."""
        return self._array.read_index(*args, **kwargs)

    def read(self, *args, **kwargs):
        """worldarray's ``read``: a world box on the level's real grid, no chain."""
        return self._array.read(*args, **kwargs)

    def chunks_covering(self, *args, **kwargs):
        return self._array.chunks_covering(*args, **kwargs)

    # --- cellmap-flow's face --------------------------------------------------

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

    def _channel(self, view):
        """``(axis, index)`` of the channel the chain selects, by
        ``select_channel``'s rule; None for an array with no more axes than
        the spatial ones."""
        store = self.source.ts
        if store.ndim <= len(self.shape):
            return None
        labels = list(getattr(store.domain, "labels", None) or [])
        axis = next((i for i, lab in enumerate(labels) if lab in CHANNEL_AXIS_NAMES), 0)
        return axis, selected_channel(view.chain())

    def _read_box(self, box, fill, resampled):
        """The voxels of ``box`` (spatial, at ``voxel_size``) through the chain:
        the part inside the array read by worldarray (stored voxels, or
        resampled ones), the chain applied, then padded with ``fill``."""
        view = self._view()
        norms = view.norms_to_apply()
        edge = fill == "edge"
        array = self._array
        channel = self._channel(view)
        spatial = [a for a in range(len(array.shape)) if channel is None or a != channel[0]]
        size = self.shape if resampled else [array.shape[a] for a in spatial]
        parts = [_clipped(b, e, n, edge) for b, e, n in zip(box.begin, box.end, size)]
        lo = [p[0] for p in parts]
        hi = [p[1] for p in parts]
        if resampled:
            data = self._read_resampled_inside(lo, hi)
            if channel is not None:
                data = np.take(data, channel[1], axis=channel[0])
        else:
            begin, end = [0] * len(array.shape), list(array.shape)
            for axis, a, c in zip(spatial, lo, hi):
                begin[axis], end[axis] = a, c
            if channel is not None:
                begin[channel[0]], end[channel[0]] = channel[1], channel[1] + 1
            data = array.read_index(begin, end)
            if channel is not None:
                data = np.take(data, 0, axis=channel[0])
        data = apply_norms(data, norms)
        pad_width = [p[2] for p in parts]
        if np.any(np.array(pad_width)):
            if edge:
                data = np.pad(data, pad_width=pad_width, mode="edge")
            else:
                data = np.pad(data, pad_width=pad_width, mode="constant", constant_values=fill)
        return data

    def _read_resampled_inside(self, lo, hi):
        """Target voxels ``[lo, hi)`` (all inside the resampled array), read by
        worldarray's resampling: the world box whose voxel centres they are,
        on the target grid from the level's corner."""
        target = np.asarray(self._resampling.target_voxel_size, dtype=float)
        corner = np.asarray(self._array.corner, dtype=float)
        box = (tuple(corner + np.asarray(lo) * target), tuple(corner + np.asarray(hi) * target))
        return self._array.read(box, voxel_size=tuple(target), padding=0)

    def read_box(self, box: Box, fill=None) -> np.ndarray:
        """The voxels of ``box`` (a ``Box`` of voxel indices at ``voxel_size``)
        through the input chain, as a numpy array padded with ``fill`` (a
        value or "edge"; ``padding``'s when None) where it runs past the
        array: ``to_ndarray_ts`` by voxels instead of nm. Unlike ``ts``,
        works on a resampled dataset too.
        """
        fill = self._fill() if fill is None else fill
        return self._read_box(box, fill, self.resampled)

    def _fill(self):
        """What reads past the array are padded with."""
        if self.custom_fill_value:
            return self.custom_fill_value
        return "edge" if self.padding == "edge" else 0

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

        Where the ROI runs past the array it is padded as ``padding`` says
        (0, or "edge": the border voxels repeated; the deprecated
        ``custom_fill_value`` wins when set), after the chain: padding is
        never normalized.

        The ROI is mapped to voxels by ``read_grid`` (``Grid.world_to_box``: the
        voxel ``roi.begin`` is in, as many as fit), not by worldarray's
        centre rule, which reads one voxel more where ``roi.shape`` is not a
        whole number of voxels.

        A resampled dataset (``resampled``) is read at ``voxel_size`` by
        worldarray's resampling: the stored voxels are resampled, then go
        through the chain, as a stored level at ``voxel_size`` would. A ROI is
        read at the deprecated ``output_voxel_size`` when that differs from
        the voxel size (see ``_read_resampled``); the whole array never is.
        """
        fill = self._fill()
        whole = Box((0,) * len(self.shape), tuple(self.shape))
        if self.resampled:
            return self._read_box(whole if roi is None else self.read_grid.world_to_box(roi), fill, True)
        if roi is None:
            return self._read_box(whole, fill, False)
        if multiscale.same_voxel_size(self.read_grid.voxel_size, self.output_voxel_size):
            return self._read_box(self.read_grid.world_to_box(roi), fill, False)
        return self._read_resampled(roi, fill)

    def _read_resampled(self, roi, fill):
        """``roi`` at ``output_voxel_size`` (deprecated, K18).

        Both voxel sizes are taken as whole nanometers, and the factor is the
        z axis's, applied to every axis: a finer output repeats each voxel, a
        coarser one takes each block's median, and the same z voxel size
        resamples nothing (the widened read is returned as it is). The read
        is widened to whole voxels of the grid anchored at 0 nm, not at the
        dataset's corner, and cropped back to ``roi`` after resampling.
        """
        voxel_size = Coordinate(self.read_grid.voxel_size)
        output_voxel_size = Coordinate(self.output_voxel_size)
        widened = roi.snap_to_grid(voxel_size)
        factor = voxel_size[0] / output_voxel_size[0]
        begin = (roi.begin - widened.begin) / output_voxel_size
        end = (roi.end - widened.begin) / output_voxel_size
        crop = tuple(slice(begin[i], end[i]) for i in range(3))

        grid = Grid(tuple(float(v) for v in voxel_size), self.read_grid.translation)
        data = self._read_box(grid.world_to_box(widened), fill, False)
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
