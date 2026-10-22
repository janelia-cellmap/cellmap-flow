"""The raw data as a model reads it: a worldarray array through the input chain.

``open_raw(path, voxel_size)`` opens, with worldarray, the level a model at
``voxel_size`` reads: the level of a multiscale group or precomputed volume
for that size (the finest no coarser than it), the path of one level as it
is. When no level is at that size the level found is read as if it were
("relabel": voxel for voxel, its corner rescaled, with a warning) or
resampled to it (``resample=True``). What comes back is a plain
``worldarray.WorldArray``: ``voxel_size`` and ``corner`` (nm, spatial axes),
``shape`` (every axis), ``read`` and ``read_index``; ``spatial_shape`` and
``spatial_axes`` pick out its spatial part.

``RawInput`` is what a model reads its input from, and what a config's
``process_chunk(idi, roi)`` or ``predict`` script is given: ``to_ndarray_ts(roi)``
is the world box ``roi`` through the input chain, the normalizers (and
ChannelSelector) given as ``input_norms``, else the process's chain
(``process_chain().input_norms``) as it is at read time, which scripts rely
on. The inference server reads with ``serving.chunkmirage_ops.BlockInput``,
the same protocol over the block chunkmirage read for a chunk.
"""

import logging

import numpy as np
import worldarray as wa
from worldarray.levels import same_voxel_size

from cellmap_flow.io import metadata, paths
from cellmap_flow.io.ome import CHANNEL_AXIS_NAMES
from cellmap_flow.process_chain import process_chain

logger = logging.getLogger(__name__)

# (level path, requested voxel size, resample) already warned about: one of
# these is opened per blockwise worker, and per request in some paths.
_warned = set()

# What the relabel warning suggests instead.
RESAMPLE_HINT = (
    "resample=True (`resample: true` in a YAML, `--resample` on the command line) "
    "resamples it to that voxel size instead"
)


def is_label_dtype(dtype) -> bool:
    """Whether data of ``dtype`` is resampled as labels (nearest voxel)
    rather than as intensities (mean and linear): bool, or an integer type
    of 32 bits or more."""
    dtype = np.dtype(dtype)
    return dtype == np.bool_ or (dtype.kind in "iu" and dtype.itemsize >= 4)


def open_raw(path, voxel_size=None, resample=False, padding="edge", context=None) -> wa.WorldArray:
    """The level of ``path`` a model at ``voxel_size`` reads, open.

    ``voxel_size`` (nm, z, y, x) picks the level of a multiscale group or
    precomputed volume: the finest not coarser than it (worldarray's
    "floor"), or with ``resample`` the coarsest not coarser than it on any
    axis, else the finest. The path of one level, or of an array, is read as
    it is. Without a voxel size a group's first level is opened.

    When the level is not at ``voxel_size``, reads are relabelled (the
    default, with a warning: the level is read voxel for voxel as if it were
    at ``voxel_size``, so the data really lies where ``stored_placement``
    says and ``corner`` is its own corner rescaled) or, with ``resample``,
    resampled to it by worldarray (labels by the nearest voxel, intensities
    linearly), from the level's own corner. ``resampled`` tells the two
    apart; ``voxel_size`` is the size reads are at either way.

    ``padding`` is what a read past the array is padded with, after the
    chain: "edge" (the default), the array's outermost voxels repeated, is
    what a model's input is read with, since a flat fill is a wall the model
    never saw in training; ``0`` for anything else. The reads go through one
    tensorstore reader thread and no cache (worldarray's
    ``capped_context(network=False)``) unless ``context`` says otherwise.
    """
    path = paths.normalize_path(path)
    requested = None if voxel_size is None else tuple(float(v) for v in voxel_size)
    array = metadata.open_level(
        path,
        voxel_size=requested,
        on_mismatch="resample" if resample else "relabel",
        padding=padding,
        context=wa.capped_context(network=False) if context is None else context,
    )
    array.method = "nearest" if is_label_dtype(array.dtype) else "linear"
    if array.level_path != path:
        logger.info(f"found level {array.level_name} for voxel size {requested}: {array.level_path}")
    stored, _ = stored_placement(array)
    if requested is not None and not same_voxel_size(stored, requested):
        key = (array.level_path, requested, resample)
        if key not in _warned:
            _warned.add(key)
            message = f"{array.level_path} is at {stored} nm but {requested} nm was requested"
            if resample:
                logger.info(f"{message}; resampling it ({array.method})")
            else:
                logger.warning(
                    f"{message}; reading it as if it were {requested} nm (the data is not resampled; {RESAMPLE_HINT})"
                )
    return array


def stored_placement(array: wa.WorldArray) -> tuple:
    """``(voxel_size, translation)`` of the level ``array`` reads as it is
    stored, nm, spatial axes, the translation voxel 0's centre: where its
    data really lies. The array's own ``voxel_size`` and ``translation`` are
    those its reads are done at, which for a level read as if it were at a
    model's voxel size (relabelled) are that size and the level's corner
    rescaled with it; worldarray's ``source_voxel_size`` reports the
    relabelled size too, so it is no use here.
    """
    stored = array.meta.in_units(space="nanometer", strict=False)
    return (
        tuple(float(stored.voxel_size[i]) for i in array.spatial),
        tuple(float(stored.translation[i]) for i in array.spatial),
    )


def spatial_shape(array: wa.WorldArray) -> tuple:
    """``array``'s shape on its spatial axes (its resampled shape when it
    resamples): what cellmap-flow calls the dataset's shape, z, y, x."""
    return tuple(int(array.shape[i]) for i in array.spatial)


def spatial_axes(array: wa.WorldArray) -> tuple:
    """The names of ``array``'s spatial axes (z, y, x when the store names none)."""
    return tuple(array.axes[i] for i in array.spatial)


# --- what a read sees: one channel, through the input chain -----------------


def apply_norms(data, input_norms=None):
    """``data`` (a tensorstore view is read first) through the chain
    ``input_norms``; ``None`` means the process's, ``process_chain().input_norms``."""
    if hasattr(data, "read"):
        data = data.read().result()
    for norm in process_chain().input_norms if input_norms is None else input_norms:
        data = norm(data)
    return data


def selected_channel(input_norms) -> int:
    """The input channel a chain asks for: its first ChannelSelector's, else 0."""
    from cellmap_flow.norm.input_normalize import ChannelSelector

    for norm in input_norms or []:
        if isinstance(norm, ChannelSelector):
            return norm.channel
    return 0


def channel_select(array: wa.WorldArray, input_norms):
    """worldarray's ``select`` for reading one channel of ``array``: its
    channel axis (the one named c, c^ or channel, else the first non-spatial
    axis) pinned to the channel the chain asks for; None for an array with
    spatial axes only."""
    extra = [i for i in range(len(array.axes)) if i not in array.spatial]
    if not extra:
        return None
    axis = next((i for i in extra if array.axes[i] in CHANNEL_AXIS_NAMES), extra[0])
    return {array.axes[axis]: selected_channel(input_norms)}


class RawInput:
    """What a model reads its input from: ``array`` (the level a model reads,
    ``open_raw``'s) through the input chain.

    ``input_norms`` is the chain reads go through, the normalizers and the
    ChannelSelector; ``None`` follows the process's chain
    (``process_chain().input_norms``) as it is at each read, so a worker that
    unpickles one reads its own chain, not the sender's. ``normalize=False``
    reads the channel the chain selects as it is stored, through no
    normalizer: the finetune trainer and the AI-annotate reader normalize
    for themselves.
    """

    def __init__(self, array: wa.WorldArray, input_norms=None, normalize=True):
        self.array = array
        self.input_norms = None if input_norms is None else list(input_norms)
        self.normalize = normalize

    def chain(self) -> list:
        """The chain reads go through now."""
        return list(process_chain().input_norms if self.input_norms is None else self.input_norms)

    def with_input_norms(self, input_norms) -> "RawInput":
        """This input read through ``input_norms`` instead; the open array is shared."""
        return RawInput(self.array, input_norms, self.normalize)

    def to_ndarray_ts(self, roi) -> np.ndarray:
        """The world box ``roi`` (nm, half-open; a funlib ``Roi``, a worldarray
        ``Box`` or a ``(begin, end)`` pair) through the chain: the channel the
        chain selects of a multichannel array, the normalizers applied unless
        ``normalize`` is off, and past the array what ``open_raw``'s ``padding``
        says, added after the chain, so padding is never normalized. A voxel
        is read when its centre lies in the box (worldarray's rule). An array
        that resamples is resampled first, then goes through the chain, as a
        stored level at that voxel size would.
        """
        norms = self.chain()
        transform = (lambda data: apply_norms(data, norms)) if self.normalize and norms else None
        return self.array.read(roi, select=channel_select(self.array, norms), post_transform=transform)

    def read_index(self, begin, end) -> np.ndarray:
        """Voxels ``[begin, end)`` on the array's spatial grid (its resampled
        grid when it resamples), read as ``to_ndarray_ts`` reads them."""
        return self.to_ndarray_ts(wa.Box.from_index(begin, end, self.array.voxel_size, self.array.translation))
