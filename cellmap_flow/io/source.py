"""Reading array data with tensorstore, whatever the format: worldarray's.

``open_array(path)`` gives an ``ArraySource`` for zarr v2 and v3, N5 and
neuroglancer precomputed, on local disk or at an ``http(s)://``, ``s3://`` or
``gs://`` URL. worldarray opens it (``worldarray.WorldArray.ts``: C order,
voxel 0 at index 0, a precomputed single channel dropped, a zarr v2
compressor's extra members such as zstd's ``checksum`` cleaned) and reads a
voxel box padded where it runs past the array (``read_index``).

Where the voxels are in the world is ``io.geometry``'s business, and what
the input chain does to them is ``ImageDataInterface``'s.
"""

import logging
from typing import Optional

import numpy as np
import tensorstore as ts
import worldarray as wa

from cellmap_flow.io.geometry import Box

logger = logging.getLogger(__name__)


def reader_context(concurrency_limit: Optional[int] = 1, cache_bytes: int = 0):
    """The tensorstore context an array is read with.

    ``concurrency_limit`` caps the files read and chunks decoded at once, as
    cellmap-flow always did; network requests are not capped (worldarray's
    ``capped_context`` caps those too, which would read a remote array one
    request at a time). ``None`` leaves tensorstore's defaults, on
    worldarray's process-wide context for ``cache_bytes``.
    """
    if not concurrency_limit:
        return wa.shared_context(cache_bytes)
    limit = {"limit": int(concurrency_limit)}
    spec = {"data_copy_concurrency": limit, "file_io_concurrency": limit}
    if cache_bytes:
        spec["cache_pool"] = {"total_bytes_limit": int(cache_bytes)}
    return ts.Context(spec)


class ArraySource:
    """One array, read with tensorstore: every channel, not normalized.

    Opened on first use and then kept, so everything sharing the source
    (ImageDataInterface's ``with_input_norms`` views) shares one open store,
    and its cache. ``array``: a worldarray array already open at ``path``.
    """

    def __init__(self, path: str, concurrency_limit: Optional[int] = 1, cache_bytes: int = 0, array=None):
        self.path = path
        self.concurrency_limit = concurrency_limit
        self.cache_bytes = cache_bytes
        self._array = array

    @property
    def array(self) -> "wa.WorldArray":
        if self._array is None:
            from cellmap_flow.io.metadata import open_level

            self._array = open_level(
                self.path, units=None, context=reader_context(self.concurrency_limit, self.cache_bytes)
            )
        return self._array

    @property
    def ts(self):
        """The tensorstore: C order, voxel 0 at index 0."""
        return self.array.ts

    def read(self, box: Optional[Box] = None, fill=0) -> np.ndarray:
        """The voxels of ``box`` (all of the array when None), padded with
        ``fill`` (a value, or "edge") where it runs past the array."""
        shape = self.array.grid.shape
        if box is None:
            return self.array.read_index((0,) * len(shape), shape)
        return self.array.read_index(box.begin, box.end, padding=fill)


def open_array(path: str, concurrency_limit: Optional[int] = 1, cache_bytes: int = 0) -> ArraySource:
    """The array at ``path``, to read with tensorstore; it is opened on the
    first read (a missing or unreadable array raises then).

    ``concurrency_limit``: how many files tensorstore reads and chunks it
    decodes at once; ``None`` leaves its defaults (one decode per core).
    ``cache_bytes``: how much decoded data it keeps for later reads of the
    same chunks (0: none).
    """
    return ArraySource(path, concurrency_limit, cache_bytes)
