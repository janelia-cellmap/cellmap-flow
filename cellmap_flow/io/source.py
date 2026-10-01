"""Reading array data with tensorstore, whatever the format.

``open_array(path)`` gives an ``ArraySource`` for zarr v2 and v3, N5 and
neuroglancer precomputed, on local disk or at an ``http(s)://``, ``s3://`` or
``gs://`` URL (``io.store``). Every format is seen
the same way: C order, as the metadata in ``io.metadata`` describes it
(channels first, then z, y, x), with voxel 0 at index 0. ``read_padded``
reads a ``Box`` of voxels from it, or from any view of it, padding what
lies outside the array.

Where the voxels are in the world is ``io.geometry``'s business, and what
the input chain does to them is ``ImageDataInterface``'s.
"""

import logging
from typing import Optional

import numpy as np
import tensorstore as ts

from cellmap_flow.io import paths
from cellmap_flow.io.store import exists, read_json, with_access
from cellmap_flow.io.geometry import Box

logger = logging.getLogger(__name__)


def _clean_zarr_compressor(meta: Optional[dict]):
    """``meta``, a ``.zarray``, with unsupported compressor fields removed;
    None when it has none (or is None).

    Tensorstore is strict about compressor metadata and rejects extra fields
    added by newer numcodecs versions, such as ``checksum``.
    """
    if meta is None:
        return None
    compressor = meta.get("compressor")
    if not isinstance(compressor, dict):
        return None

    known_fields = {
        "zstd": {"id", "level"},
        "zlib": {"id", "level"},
        "gzip": {"id", "level"},
        "bz2": {"id", "level"},
        "blosc": {"id", "cname", "clevel", "shuffle", "blocksize"},
    }
    allowed = known_fields.get(compressor.get("id", ""))
    if allowed is None:
        return None

    extra_keys = set(compressor.keys()) - allowed
    if not extra_keys:
        return None

    logger.info(
        "Stripping unsupported compressor fields %s for tensorstore compatibility",
        extra_keys,
    )
    return {**meta, "compressor": {k: v for k, v in compressor.items() if k in allowed}}


def _open(path: str, concurrency_limit: Optional[int], cache_bytes: int):
    """The tensorstore for ``path``, in C order with voxel 0 at index 0.

    The array is read where ``path`` is, a local directory or a URL, through
    the kvstores io.store gives it (anonymous first on s3, see there).
    """
    driver = "n5" if paths.suffix_format(path) == "n5" else "zarr"
    extra_args = {}
    location = path
    if paths.is_precomputed(path):
        # A trailing /s<N> picks the scale.
        location, scale_index = paths.precomputed_volume(path)
        driver = "neuroglancer_precomputed"
        extra_args = {"scale_index": scale_index}
    elif driver == "zarr" and exists(location, "zarr.json"):
        driver = "zarr3"

    # tensorstore rejects compressor fields it doesn't know ("extra
    # members", e.g. numcodecs' zstd checksum), so such arrays are opened
    # with their metadata minus those fields.
    assume_metadata = False
    if driver == "zarr":
        cleaned_metadata = _clean_zarr_compressor(read_json(location, ".zarray"))
        if cleaned_metadata is not None:
            extra_args["metadata"] = cleaned_metadata
            assume_metadata = True

    context = {}
    if concurrency_limit:
        context["data_copy_concurrency"] = {"limit": concurrency_limit}
        context["file_io_concurrency"] = {"limit": concurrency_limit}
    if cache_bytes:
        context["cache_pool"] = {"total_bytes_limit": int(cache_bytes)}

    open_kwargs = {"open": True, "assume_metadata": True} if assume_metadata else {}

    def attempt(kvstore):
        spec = {"driver": driver, "kvstore": kvstore, **extra_args}
        if context:
            spec["context"] = context
        return ts.open(spec, read=True, write=False, **open_kwargs).result()

    array = with_access(location, attempt)

    if driver in ("n5", "neuroglancer_precomputed"):
        # Both drivers expose Fortran order (x, y, z[, channel]); everything
        # else here -- the metadata readers, the ROI math, neuroglancer's axis
        # names -- is C order (z, y, x). Reading an N5 dataset without this
        # returned x/z-transposed data, and precomputed lost its x axis to the
        # channel selection.
        array = array[ts.d[:].transpose[::-1]]
    if driver == "neuroglancer_precomputed":
        # tensorstore starts a precomputed volume's domain at its
        # voxel_offset. Index 0 is voxel 0 everywhere else here, and the
        # metadata's translation already carries the offset.
        array = array[ts.d[:].translate_to[0]]
    return array


class ArraySource:
    """One array, read with tensorstore: every channel, not normalized.

    Opened on first use and then kept, so everything sharing the source
    (ImageDataInterface's ``with_input_norms`` views) shares one open store,
    and its cache.
    """

    def __init__(self, path: str, concurrency_limit: Optional[int] = 1, cache_bytes: int = 0):
        self.path = path
        self.concurrency_limit = concurrency_limit
        self.cache_bytes = cache_bytes
        self._ts = None

    @property
    def ts(self):
        """The tensorstore: C order, voxel 0 at index 0."""
        if self._ts is None:
            self._ts = _open(self.path, self.concurrency_limit, self.cache_bytes)
        return self._ts

    def read(self, box: Optional[Box] = None, fill=0) -> np.ndarray:
        """``read_padded`` of this array: all of it when ``box`` is None."""
        return read_padded(self.ts, box, fill)


def open_array(path: str, concurrency_limit: Optional[int] = 1, cache_bytes: int = 0) -> ArraySource:
    """The array at ``path``, to read with tensorstore; it is opened on the
    first read (a missing or unreadable array raises then).

    ``concurrency_limit``: how many files tensorstore reads and chunks it
    decodes at once; ``None`` leaves its defaults (one decode per core).
    ``cache_bytes``: how much decoded data it keeps for later reads of the
    same chunks (0: none). Cached chunks are still checked against the file
    on every read, so what is read never differs.
    """
    return ArraySource(path, concurrency_limit, cache_bytes)


def _read_and_pad(begin, end, inclusive_min, exclusive_max, edge):
    """Along one axis, for a box ``[begin, end)`` over an array
    ``[inclusive_min, exclusive_max)``: the slice of the array to read, and
    the padding ``[before, after]`` that makes what is read the box's length.

    What is read is the box's own voxels in the array: none when the box
    lies wholly on one side of it, unless ``edge``, when it is the array's
    voxel nearest the box, to repeat.
    """
    length = end - begin
    start = min(max(begin, inclusive_min), exclusive_max)
    stop = max(min(end, exclusive_max), inclusive_min)
    if edge and start == stop and length:
        if begin >= exclusive_max:
            start, stop = exclusive_max - 1, exclusive_max
        else:
            start, stop = inclusive_min, inclusive_min + 1
    before = min(max(start - begin, 0), length - (stop - start))
    return slice(start, stop), [before, length - (stop - start) - before]


def read_padded(store, box: Optional[Box] = None, fill=0, transform=None) -> np.ndarray:
    """The voxels of ``box`` (one entry per dimension of ``store``; all of
    it when None) as a numpy array, padded where the box runs past the
    store's domain.

    ``fill`` is the padding value, or "edge" to repeat the border voxels:
    each voxel outside the array takes the value of the nearest one inside.
    A box wholly outside the array is all padding. ``transform`` is applied
    to what was read before the padding is added: ImageDataInterface runs
    its input chain there, so padding is never normalized, and a box with
    nothing to read still has the chain's dtype.
    """
    if box is None:
        with ts.Transaction() as txn:
            data = store.with_transaction(txn).read().result()
        return data if transform is None else transform(data)

    domain = store.domain
    valid, pad_width = [], []
    for begin, end, inclusive_min, exclusive_max in zip(
        box.begin, box.end, domain.inclusive_min, domain.exclusive_max
    ):
        read, pad = _read_and_pad(begin, end, inclusive_min, exclusive_max, fill == "edge")
        valid.append(read)
        pad_width.append(pad)
    with ts.Transaction() as txn:
        data = store.with_transaction(txn)[tuple(valid)].read().result()
    if transform is not None:
        data = transform(data)
    if np.any(np.array(pad_width)):
        if fill == "edge":
            data = np.pad(data, pad_width=pad_width, mode="edge")
        else:
            data = np.pad(data, pad_width=pad_width, mode="constant", constant_values=fill)
    return data
