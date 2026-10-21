"""A dataset to read in world coordinates, through the input chain.

``ImageDataInterface(path, voxel_size)`` reads through a ``worldarray``
array (``self.array``): worldarray opens it -- a multiscale group or
precomputed volume at its level for ``voxel_size``, the path of one level as it
is -- reads its metadata and reads its voxels, padded where a read runs past
it, and resamples. cellmap-flow keeps on top:

- the grid reads are done on (``_grid``, an ``io.geometry.Grid``: voxel size
  and voxel 0's lower corner in nm, worldarray's ``voxel_size`` and ``corner``),
  which maps a ROI to voxels by its corner, and with it ``voxel_size``,
  ``offset``, ``roi`` and ``shape`` (spatial) as cellmap-flow has them;
- the input chain: what is read goes through the normalizers and
  ChannelSelector given as ``input_norms``, else the process's chain
  (``process_chain().input_norms``) as it is at read time, which user
  ``process_chunk`` scripts rely on; worldarray applies it before padding
  (``transform``), and to a resampled read after resampling (``post_transform``);
- the deprecated ``output_voxel_size`` and ``custom_fill_value``.
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
from cellmap_flow.io.resample import is_label_dtype
from cellmap_flow.io.source import ArraySource
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


class ImageDataInterface:
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
        (worldarray's ``capped_context(network=False)``), which is what blockwise
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
          ``on_mismatch="resample"``), from the level the "resample" mode
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
        if on_voxel_size_mismatch not in ON_VOXEL_SIZE_MISMATCH:
            raise ValueError(
                "on_voxel_size_mismatch must be one of "
                f"{', '.join(map(repr, ON_VOXEL_SIZE_MISMATCH))}, got {on_voxel_size_mismatch!r}"
            )
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
        # The level for voxel_size of a group or precomputed volume ("floor"; the
        # "resample" mode when resampling); the path of one level at that level.
        # worldarray resamples or relabels a mismatch; "error" is raised here.
        self.array = metadata.open_level(
            dataset_path,
            voxel_size=None if voxel_size is None else tuple(float(v) for v in voxel_size),
            on_mismatch={"error": "keep"}.get(on_voxel_size_mismatch, on_voxel_size_mismatch),
            padding="edge",
            context=wa.capped_context(network=False),
        )
        array = self.array
        self.path = _level_path(dataset_path, array, voxel_size)
        if self.path != dataset_path:
            logger.info(f"found scale {array.level_name} for voxel size {voxel_size}; using {self.path}")
        self.input_norms = None if input_norms is None else list(input_norms)
        meta = metadata.from_world_array(array.meta, self.path)
        array.method = "nearest" if is_label_dtype(meta.dtype) else "linear"
        actual_voxel_size, _, self.chunk_shape, _, self.axes_names, self.filetype = legacy_meta(meta)
        actual_voxel_size = snap_integral(actual_voxel_size)
        # What the data really is, and what the caller asked for; voxel_size
        # below is the one reads are done in.
        self.actual_voxel_size = coordinate_or_floats(actual_voxel_size, "voxel size", self.path)
        self.requested_voxel_size = (
            None if voxel_size is None else coordinate_or_floats(voxel_size, "voxel size", self.path)
        )
        if voxel_size is not None and not multiscale.same_voxel_size(snap_integral(voxel_size), actual_voxel_size):
            requested = tuple(snap_integral(voxel_size).tolist())
            message = f"{self.path} is at {tuple(actual_voxel_size.tolist())} nm but {requested} nm was requested"
            if on_voxel_size_mismatch == "error":
                raise ValueError(message)
            key = (self.path, requested, on_voxel_size_mismatch)
            if key not in _warned_relabel:
                _warned_relabel.add(key)
                if resample:
                    logger.info(f"{message}; resampling it ({array.method})")
                else:
                    # worldarray relabels voxel for voxel: voxel i stays voxel i,
                    # so the corner scales with the voxel size.
                    logger.warning(
                        f"{message}; reading it as if it were {requested} nm (the data is "
                        f"not resampled; {RESAMPLE_HINT})"
                    )
        # The grid reads are done on: worldarray's (resampled or relabelled)
        # voxel size and corner on the spatial axes, in exact floats. voxel_size
        # and offset below are Coordinates when whole, and roi is the whole-nm
        # box around the data.
        voxel_size_f = snap_integral(array.voxel_size)
        offset_f = snap_integral(array.corner)
        shape = tuple(array.shape[i] for i in array.spatial)
        self.shape = Coordinate(shape)
        self._grid = Grid(tuple(float(v) for v in voxel_size_f), tuple(float(v) for v in offset_f))
        self.voxel_size = coordinate_or_floats(voxel_size_f, "voxel size", self.path)
        self.offset = coordinate_or_floats(offset_f, "offset", self.path)
        self.roi = self._grid.box_to_world(Box((0,) * len(shape), tuple(shape)))
        self.custom_fill_value = custom_fill_value
        self.padding = padding
        # The array as stored (every channel, not normalized), shared with
        # every with_input_norms() view.
        self.source = ArraySource(self.path, array=array)
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
        return self.array.resampled

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

    def _read_box(self, box, fill):
        """The voxels of ``box`` (spatial, at ``voxel_size``) through the chain,
        padded with ``fill`` (a value or "edge") past the array.

        worldarray applies the chain before padding: as ``transform`` to a
        stored level, and as ``post_transform`` to a resampled one, after the
        resampling and the cast back to the stored dtype, as a stored level at
        ``voxel_size`` would be normalized.
        """
        view = self._view()
        norms = view.norms_to_apply()
        array = self.array
        store = self.source.ts
        select, axis, chain = None, None, None
        if store.ndim > len(self.shape):
            # select_channel's rule: the axis labelled c/c^/channel, else the first
            labels = list(getattr(store.domain, "labels", None) or [])
            axis = next((i for i, lab in enumerate(labels) if lab in CHANNEL_AXIS_NAMES), 0)
            select = {array.axes[axis]: selected_channel(view.chain())}
        if norms:
            def chain(data):
                # worldarray hands the (post_)transform the pinned axis, of size 1, and
                # drops it after: the chain sees the one channel without it
                if axis is None:
                    return apply_norms(data, norms)
                return np.expand_dims(apply_norms(np.take(data, 0, axis=axis), norms), axis)
        if not self.resampled:
            return array.read_index(box.begin, box.end, padding=fill, select=select, transform=chain)
        # The target voxels' world box: their centres are where worldarray reads.
        voxel_size = np.asarray(self.voxel_size, dtype=float)
        corner = np.asarray(array.corner)
        world = (tuple(corner + np.asarray(box.begin) * voxel_size), tuple(corner + np.asarray(box.end) * voxel_size))
        return array.read(world, padding=fill, select=select, post_transform=chain)

    def read_box(self, box: Box, fill=None) -> np.ndarray:
        """The voxels of ``box`` (a ``Box`` of voxel indices at ``voxel_size``)
        through the input chain, as a numpy array padded with ``fill`` (a
        value or "edge"; ``padding``'s when None) where it runs past the
        array: ``to_ndarray_ts`` by voxels instead of nm. Unlike ``ts``,
        works on a resampled dataset too.
        """
        return self._read_box(box, self._fill() if fill is None else fill)

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

        The ROI is mapped to voxels by ``_grid`` (``Grid.world_to_box``: the
        voxel ``roi.begin`` is in, as many as fit), not by worldarray's
        centre rule, which reads one voxel more where ``roi.shape`` is not a
        whole number of voxels.

        A resampled dataset (``resampled``) is read at ``voxel_size`` by
        worldarray's resampling, then goes through the chain, as a stored
        level at ``voxel_size`` would. A ROI is read at the deprecated
        ``output_voxel_size`` when that differs from the voxel size (see
        ``_read_resampled``); the whole array never is.
        """
        fill = self._fill()
        if roi is None:
            return self._read_box(Box((0,) * len(self.shape), tuple(self.shape)), fill)
        if self.resampled or multiscale.same_voxel_size(self._grid.voxel_size, self.output_voxel_size):
            return self._read_box(self._grid.world_to_box(roi), fill)
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
        voxel_size = Coordinate(self._grid.voxel_size)
        output_voxel_size = Coordinate(self.output_voxel_size)
        widened = roi.snap_to_grid(voxel_size)
        factor = voxel_size[0] / output_voxel_size[0]
        begin = (roi.begin - widened.begin) / output_voxel_size
        end = (roi.end - widened.begin) / output_voxel_size
        crop = tuple(slice(begin[i], end[i]) for i in range(3))

        grid = Grid(tuple(float(v) for v in voxel_size), self._grid.translation)
        data = self._read_box(grid.world_to_box(widened), fill)
        if factor > 1:
            repeat = int(voxel_size[0] / output_voxel_size[0])
            return np.kron(data, np.ones((repeat,) * 3, dtype=data.dtype))[crop]
        if factor < 1:
            from skimage.measure import block_reduce

            return block_reduce(data, block_size=int(1 / factor), func=np.median)[crop]
        return data


def _level_path(dataset_path, array, voxel_size):
    """cellmap-flow's path of the level ``array`` opened for ``dataset_path``: a
    precomputed scale as ``…/s<N>``, a group's level under the group, an array
    path as it is."""
    if paths.is_precomputed(dataset_path):
        volume, scale = paths.precomputed_scale(dataset_path)
        if scale is not None or voxel_size is None:
            return dataset_path
        index = wa.store.precomputed_scale_index(array.meta.group_attrs, array.level_name)
        return paths.join(volume, f"s{index}")
    if wa.store.normalize_path(dataset_path) == array.level_path:
        return dataset_path
    return paths.join(dataset_path, array.level_name)


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
