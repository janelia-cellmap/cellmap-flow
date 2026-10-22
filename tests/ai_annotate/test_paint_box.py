"""``fill.paint_box``: writing a box of labels into MinIO when the box's chunk is only on disk.

MinIO is an in-memory zarr per bucket key, as in tests/utils/test_view_labels.py.
The volume is 8^3 voxels in 4^3 chunks; the box is its first 4 x 4 x 4 chunk.
"""

import numpy as np
import pytest
import zarr

from cellmap_flow.finetune.session import fill, minio
from tests.utils.test_view_labels import _Bucket

STATE = {"bucket": "annotations"}
LO, HI = np.array([0, 0, 0]), np.array([4, 4, 4])


@pytest.fixture
def served(monkeypatch):
    """The labels of volume ``vol-1`` as MinIO holds them: 0s, with a painted corner."""
    bucket = _Bucket()
    store = bucket.stores["annotations/vol-1.zarr"] = zarr.MemoryStore()
    monkeypatch.setattr(minio, "make_s3_filesystem", lambda state: bucket)
    monkeypatch.setattr(fill.s3fs, "S3Map", lambda root, s3, check: s3.stores[root])
    labels = zarr.open_group(store).require_group("annotation").create_dataset(
        "s0", shape=(8,) * 3, chunks=(4,) * 3, dtype="u1", fill_value=0
    )
    # The user painted background (1) and an object (5) in the box's first slice.
    painted = np.zeros((8,) * 3, dtype=np.uint8)
    painted[0, 0, :2] = 1
    painted[0, 1, :2] = 5
    labels[:] = painted
    return labels


def _seed(existing):
    """An object (2) over the box's first two rows of y, background (1) elsewhere."""
    labels = np.ones_like(existing)
    labels[:, :2, :] = 2
    return labels


def test_a_chunk_only_on_disk_is_uploaded_before_painting(served, tmp_path):
    # The local volume has the box's chunk; MinIO does not (an import whose mirror failed).
    store = served.store
    del store["annotation/s0/0.0.0"]
    local = zarr.open_group(str(tmp_path / "vol-1.zarr"), mode="w").require_group("annotation").create_dataset(
        "s0", shape=(8,) * 3, chunks=(4,) * 3, dtype="u1", fill_value=0
    )
    local[0, 0, 3] = 7
    fill.paint_box(STATE, "vol-1", LO, HI, _seed, local_zarr_path=str(tmp_path / "vol-1.zarr"))
    assert served[0, 0, 3] == 7  # kept: the chunk was uploaded first, then only 0s filled
