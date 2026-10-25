"""python -m cellmap_flow.review_index: an instance segmentation in, the
index the Review tab reads out."""


import numpy as np
import pytest
import worldarray as wa

from cellmap_flow import review
from cellmap_flow.review_index import main

BIG = 2**40 + 3  # a uint64 id that a float would round
VOXEL_SIZE, CORNER = [8.0, 4.0, 2.0], [100.0, 0.0, -2.0]

# id: (vox, bbox z0 z1 y0 y1 x0 x1 in voxels, centroid in world nm)
EXPECTED = {
    5: (24, (0, 2, 0, 3, 0, 4), (108.0, 6.0, 2.0)),
    7: (105, (1, 6, 5, 8, 3, 10), (128.0, 26.0, 11.0)),  # spans many blocks
    BIG: (1, (5, 6, 0, 1, 9, 10), (144.0, 2.0, 17.0)),
}


@pytest.fixture
def labels(tmp_path):
    """An OME-Zarr group of one level, s0, whose voxel 0 has its lower corner at CORNER."""
    data = np.zeros((6, 8, 10), dtype=np.uint64)
    data[0:2, 0:3, 0:4] = 5
    data[1:6, 5:8, 3:10] = 7
    data[5, 0, 9] = BIG
    level = wa.create(
        str(tmp_path / "labels.zarr" / "s0"), data.shape, data.dtype, VOXEL_SIZE,
        wa.corner_to_centre(CORNER, VOXEL_SIZE), chunks=(2, 4, 4), format="zarr2",
    )
    level.write_index((0, 0, 0), data.shape, data)
    return str(tmp_path / "labels.zarr")


def test_an_index_of_a_segmentation(labels, tmp_path):
    out = str(tmp_path / "review.sqlite")
    # Blocks smaller than the labels, so their sums must be merged.
    main([labels, out, "--queue", "smallest", "--queue", "largest", "--block", "2", "4", "4"])

    conn = review.open_db(out)
    rows = conn.execute(
        "SELECT id, vox, bz0, bz1, by0, by1, bx0, bx1, cz_nm, cy_nm, cx_nm FROM instances"
    ).fetchall()
    assert {r[0]: (r[1], tuple(r[2:8]), tuple(r[8:])) for r in rows} == EXPECTED
    assert review.queues(conn) == {"smallest": "rank_smallest", "largest": "rank_largest"}
    assert review.get_next(conn, "smallest")["id"] == BIG
    assert review.get_next(conn, "largest")["id"] == 7
    progress = review.get_progress(conn)
    assert progress["by_state"] == {"unreviewed": 3}
    assert progress["queues"]["largest"] == {"total": 3, "reviewed": 0, "label": "largest first"}
    assert (progress["voxel_size_nm"], progress["offset_nm"]) == (VOXEL_SIZE, CORNER)
    conn.close()

    # The ledger holds the verdicts, so the index is never rebuilt over.
    with pytest.raises(FileExistsError):
        main([labels, out])
