"""``fill.paint_box``: writing a box of labels into MinIO, over painted voxels or not.

MinIO is an in-memory zarr per bucket key, as in tests/utils/test_view_labels.py.
The volume is 8^3 voxels in 4^3 chunks; the box is its first 4 x 4 x 4 chunk.
"""

import numpy as np
import pytest
import zarr

from cellmap_flow.finetune.session import fill, minio

STATE = {"bucket": "annotations"}
LO, HI = np.array([0, 0, 0]), np.array([4, 4, 4])


class _Bucket:
    """MinIO's bucket as s3fs sees it: an in-memory zarr per key."""

    def __init__(self):
        self.stores = {}

    def _split(self, path):
        root, _, rest = path.partition(".zarr/")
        return self.stores.get(root + ".zarr", {}), rest

    def exists(self, path):
        store, rest = self._split(path)
        return rest in store

    def put(self, local, path):
        store, rest = self._split(path)
        store[rest] = open(local, "rb").read()


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


def test_without_overwrite_only_unannotated_voxels_are_written(served):
    before = served[:]
    undo = []
    counts = fill.paint_box(STATE, "vol-1", LO, HI, _seed, undo=undo)
    after = served[:]
    # 2 x 4 x 4 object voxels and 2 x 4 x 4 background ones, less the 4 painted.
    assert counts == (32 - 4, 32, 0)
    assert np.array_equal(after[0, :2, :2], before[0, :2, :2])  # the painted voxels kept
    assert np.all(after[:4, :2, :4][before[:4, :2, :4] == 0] == 2)
    assert np.all(after[:4, 2:4, :4] == 1)
    assert np.all(after[4:] == 0) and np.all(after[:, 4:] == 0)  # nothing outside the box
    assert len(undo) == 1


def test_with_overwrite_painted_voxels_are_written_too(served):
    undo = []
    counts = fill.paint_box(STATE, "vol-1", LO, HI, _seed, undo=undo, overwrite=True)
    after = served[:]
    assert np.all(after[:4, :2, :4] == 2)
    assert np.all(after[:4, 2:4, :4] == 1)
    # Every object voxel and every background one was written; 4 held labels.
    assert counts == (32, 32, 4)
    assert len(undo) == 1


def test_overwrite_counts_only_voxels_that_change(served):
    # Labels that agree with what is painted: the painted 1s and 5s are not written.
    def agreeing(existing):
        labels = np.where(existing > 0, existing, 2).astype(existing.dtype)
        labels[0, 0, 2] = 3  # one unpainted voxel
        return labels

    n_foreground, n_background, n_overwritten = fill.paint_box(STATE, "vol-1", LO, HI, agreeing, overwrite=True)
    assert n_overwritten == 0
    assert (n_foreground, n_background) == (64 - 4, 0)
    # A label that differs from a painted one is an overwrite.
    counts = fill.paint_box(STATE, "vol-1", LO, HI, lambda e: np.where(e == 5, 6, 0).astype(e.dtype), overwrite=True)
    assert counts == (2, 0, 2)


def test_zero_labels_leave_voxels_alone_even_with_overwrite(served):
    before = served[:]
    undo = []
    assert fill.paint_box(STATE, "vol-1", LO, HI, np.zeros_like, undo=undo, overwrite=True) == (0, 0, 0)
    assert np.array_equal(served[:], before)
    assert undo == []  # nothing written, nothing to undo


@pytest.mark.parametrize("overwrite", [False, True])
def test_undo_puts_the_box_back(served, overwrite):
    before = served[:]
    undo = []
    fill.paint_box(STATE, "vol-1", LO, HI, _seed, undo=undo, overwrite=overwrite)
    lo, hi, box_before, box_after = undo[-1]
    assert np.array_equal(box_before, before[:4, :4, :4])
    assert np.array_equal(box_after, served[:4, :4, :4])
    # A stroke painted after the action survives the undo.
    served[3, 3, 3] = 9
    restored = fill.restore_box(STATE, "vol-1", lo, hi, box_before, box_after)
    expected = before.copy()
    expected[3, 3, 3] = 9
    assert np.array_equal(served[:], expected)
    assert restored == int(np.count_nonzero(box_before != box_after)) - 1


def test_fill_unpainted_is_paint_box_without_overwrite(served):
    undo = []
    assert fill.fill_unpainted(STATE, "vol-1", LO, HI, _seed, undo=undo) == (32 - 4, 32)
    assert len(undo) == 1
    assert served[0, 1, 0] == 5


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
