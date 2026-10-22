"""The box decisions both modes make (cellmap_flow.portable.fill): which voxels
a write changes, the ids new objects get, a split's ids, and what an undo puts
back. The writes to MinIO around them are tests/ai_annotate/test_paint_box.py's
and tests/utils/test_view_labels.py's."""

import numpy as np
import pytest

from cellmap_flow.portable.fill import fresh_ids, paint, relabel_objects, restore, rewrite


def test_a_seed_fills_only_unpainted_voxels_and_overwrite_replaces_painted_ones():
    existing = np.array([0, 0, 1, 5, 0], dtype=np.uint8)
    labels = np.array([2, 1, 2, 2, 0], dtype=np.uint8)  # 0: leave as it is
    after, counts = paint(existing, labels)
    assert after.tolist() == [2, 1, 1, 5, 0] and counts == (1, 1, 0)
    after, counts = paint(existing, labels, overwrite=True)
    assert after.tolist() == [2, 1, 2, 2, 0] and counts == (3, 1, 2)
    assert paint(existing, np.zeros_like(existing)) == (None, (0, 0, 0))


def test_new_ids_are_the_lowest_free_or_count_up_and_say_when_none_fit():
    existing = np.array([2, 4, 9], dtype=np.uint8)
    assert fresh_ids(existing, 3, count_up=False).tolist() == [3, 5, 6]
    assert fresh_ids(existing.astype(np.uint16), 2, count_up=True).tolist() == [10, 11]
    with pytest.raises(ValueError, match="do not fit in the volume's uint8 labels"):
        fresh_ids(np.arange(2, 256, dtype=np.uint8), 1, count_up=False)


def test_a_wall_through_an_object_splits_it_and_a_stroke_across_two_merges_them():
    # One object (3) cut by a background wall (1) in every slice; and objects
    # 5 and 6 joined by a stroke of 6, 5 the bigger.
    box = np.zeros((2, 3, 9), dtype=np.uint8)
    box[:, :, 0:2] = 3
    box[:, :, 2] = 1
    box[:, :, 3:5] = 3
    box[:, 0, 6:9] = 5
    box[:, 1, 6] = 6
    labels, counts = relabel_objects(box)
    assert counts == {"objects": 3, "split": 1, "merged": 1}
    assert len({int(labels[0, 0, 0]), int(labels[0, 0, 3])}) == 2  # the two halves of 3 apart
    assert (labels[:, :2, 6:9][box[:, :2, 6:9] > 0] == 5).all()  # 6 joins 5, the bigger
    assert (labels[box < 2] == box[box < 2]).all()  # background and unpainted alone
    after, changed = rewrite(box, labels)
    assert changed == int(np.count_nonzero(labels != box)) and (after == labels).all()


def test_undo_puts_back_only_what_nobody_touched_since():
    before = np.array([0, 0, 0, 7], dtype=np.uint8)
    after = np.array([2, 2, 1, 7], dtype=np.uint8)
    current = np.array([2, 9, 1, 7], dtype=np.uint8)  # voxel 1 painted again since
    restored, n = restore(current, before, after)
    assert restored.tolist() == [0, 9, 0, 7] and n == 2
    assert restore(before, before, after) == (None, 0)
