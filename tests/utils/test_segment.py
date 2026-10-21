"""The instance segmenters (``post.segment``) on small synthetic volumes.

Two overlapping spheres are the case that tells them apart: one blob to a
threshold, two to anything that sees where they meet.
"""

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from cellmap_flow.post import segment

OFFSETS = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def _two_touching_spheres():
    z, y, x = np.mgrid[:16, :16, :24]
    left = (z - 8) ** 2 + (y - 8) ** 2 + (x - 7) ** 2 <= 25
    right = (z - 8) ** 2 + (y - 8) ** 2 + (x - 15) ** 2 <= 25
    return left | right


def _affinities_of(objects, low=0.1, high=0.9):
    """(3, z, y, x): high where a voxel and its neighbour at the offset are one object."""
    affs = np.full((len(OFFSETS),) + objects.shape, low)
    for channel, offset in enumerate(OFFSETS):
        here = tuple(slice(0, n - o) for n, o in zip(objects.shape, offset))
        there = tuple(slice(o, n) for n, o in zip(objects.shape, offset))
        affs[(channel,) + here] = np.where((objects[here] == objects[there]) & (objects[here] > 0), high, low)
    return affs


def test_connected_components_merge_two_touching_spheres():
    labels = segment.connected_components(_two_touching_spheres())
    assert labels.dtype == np.uint64 and labels.max() == 1


def test_the_distance_watershed_splits_them_at_their_neck_in_voxels():
    blob = _two_touching_spheres()
    labels = segment.distance_watershed(distance_transform_edt(blob), threshold=0, peak_depth=0.5)
    assert labels.max() == 2 and labels[8, 8, 7] != labels[8, 8, 15]
    np.testing.assert_array_equal(labels > 0, blob)


def test_the_mutex_watershed_splits_objects_the_affinities_cut_apart():
    objects = np.zeros((10, 10, 10), int)
    objects[2:8, 2:8, 2:5] = 1
    objects[2:8, 2:8, 5:8] = 2
    labels = segment.mutex_watershed(_affinities_of(objects), OFFSETS, bias=0.5)
    assert labels.max() == 2 and labels[5, 5, 3] != labels[5, 5, 6]
    assert labels[0, 0, 0] == 0, "background fragments are dropped"
    # A threshold on the mean affinity would join them: the cut voxels' mean is high.
    assert segment.connected_components(_affinities_of(objects).mean(axis=0) > 0.5).max() == 1


def test_per_slice_labels_each_z_slice_apart():
    mask = np.zeros((4, 6, 6), bool)
    mask[:, 2:4, 2:4] = True  # a column through every slice
    assert segment.connected_components(mask).max() == 1
    labels = segment.connected_components(mask, per_slice=True)
    assert labels.max() == 4 and [labels[z, 2, 2] for z in range(4)] == [1, 2, 3, 4]
    with pytest.raises(ValueError):
        segment.connected_components(mask[0], per_slice=True)


def test_min_size_drops_specks_and_numbers_the_rest_without_gaps():
    mask = np.zeros((6, 6, 6), bool)
    mask[0, 0, 0] = True  # a speck, labelled first
    mask[2:5, 2:5, 2:5] = True
    mask[5, 5, 5] = True  # another
    labels = segment.connected_components(mask, min_size=2)
    assert labels.max() == 1 and labels[3, 3, 3] == 1 and labels[0, 0, 0] == 0


@pytest.mark.parametrize("second, connectivity, n", [
    pytest.param((1, 1, 0), 1, 2, id="an-edge-neighbour-apart-by-faces"),
    pytest.param((1, 1, 0), 2, 1, id="an-edge-neighbour-joined-by-edges"),
    pytest.param((1, 1, 1), 2, 2, id="a-corner-neighbour-apart-by-edges"),
    pytest.param((1, 1, 1), 3, 1, id="a-corner-neighbour-joined-by-all"),
    pytest.param((1, 1, 1), "3", 1, id="as-a-form-sends-it"),
])
def test_connectivity_says_which_neighbours_touch(second, connectivity, n):
    mask = np.zeros((3, 3, 3), bool)
    mask[0, 0, 0] = mask[second] = True
    assert segment.connected_components(mask, connectivity).max() == n


def test_a_models_own_instances_are_numbered_from_1_without_gaps():
    """Label it seeds from a server of instance labels (a Cellpose model's
    masks, an instance model's ids) through this: its ids, which can be any
    uint64, become 1, 2, ... in the annotation volume."""
    labels = np.zeros((4, 4, 4), np.uint64)
    labels[0, 0, 0] = labels[3, 3, 3] = 2**40  # one id in two places
    labels[1, 1, 1] = 7
    relabelled = segment.relabel_instances(labels)
    assert sorted(np.unique(relabelled).tolist()) == [0, 1, 2]
    assert relabelled[0, 0, 0] == relabelled[3, 3, 3]
    split = segment.relabel_instances(labels, split_disconnected=True)
    assert split.max() == 3 and split[0, 0, 0] != split[3, 3, 3]
