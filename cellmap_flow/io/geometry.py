"""The chunks of a zarr v2 array that exist on disk.

``list_populated_chunks`` lists them by index (worldarray's
``list_chunk_keys``) and ``CHUNK_KEY_RE`` matches their file names; the
finetune volumes' sync and samplers use both. Where an array's voxels lie in
the world is worldarray's business (``WorldArray.voxel_size``, ``corner``,
``read`` and ``read_index``; ``cellmap_flow.io.raw`` opens the level a model
reads).
"""

import re
from typing import List, Tuple

import worldarray as wa

# A chunk file of a three-dimensional zarr v2 array with "." separators: z.y.x
# (finetune/session/sync.py copies those).
CHUNK_KEY_RE = re.compile(r"^\d+\.\d+\.\d+$")


def list_populated_chunks(array_dir) -> List[Tuple[int, ...]]:
    """The indices of the chunks of a three-dimensional zarr v2 array that
    have a file (``z.y.x`` keys), by listing its keys
    (``worldarray.list_chunk_keys``: no ``.zarray`` needed, a URL too). In
    index order, not ``os.listdir``'s: that order differs between filesystems
    (ext4 orders names by a per-filesystem hash), so the same seed would draw
    different patches on another machine.
    """
    return wa.list_chunk_keys(str(array_dir), separator=".", ndim=3)
