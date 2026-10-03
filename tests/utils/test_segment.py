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


@pytest.mark.parametrize("markers", ["peaks", "threshold"])
def test_the_distance_watershed_defaults_split_a_sigmoid_distance(markers):
    """What a cellmap distance model gives, 0.5 at the boundary and flat
    inside: its peaks, or its cores over 0.9."""
    distance = distance_transform_edt(_two_touching_spheres())
    probability = 1 / (1 + np.exp(-2 * (distance - 0.5) / 4))
    labels = segment.distance_watershed(probability, markers=markers, marker_threshold=0.85)
    assert labels.max() == 2


def test_a_faint_part_no_marker_reaches_is_still_an_object():
    """A blob barely over the threshold has no peak deep enough; it is kept,
    not lost between the threshold and the markers."""
    distance = np.zeros((8, 8, 8))
    distance[2:5, 2:5, 2:5] = 0.52
    labels = segment.distance_watershed(distance, threshold=0.5, peak_depth=0.05)
    assert labels.max() == 1 and (labels[2:5, 2:5, 2:5] == 1).all()


def test_the_mutex_watershed_splits_objects_the_affinities_cut_apart():
    objects = np.zeros((10, 10, 10), int)
    objects[2:8, 2:8, 2:5] = 1
    objects[2:8, 2:8, 5:8] = 2
    labels = segment.mutex_watershed(_affinities_of(objects), OFFSETS, bias=0.5)
    assert labels.max() == 2 and labels[5, 5, 3] != labels[5, 5, 6]
    assert labels[0, 0, 0] == 0, "background fragments are dropped"
    # A threshold on the mean affinity would join them: the cut voxels' mean is high.
    assert segment.connected_components(_affinities_of(objects).mean(axis=0) > 0.5).max() == 1


def test_the_mutex_watershed_drops_specks_by_min_size():
    objects = np.zeros((10, 10, 10), int)
    objects[2:8, 2:8, 2:8] = 1
    objects[0, 0, 0] = 2  # a single voxel with high affinity to nothing
    affs = _affinities_of(objects)
    affs[:, 0, 0, 0] = 0.9
    assert segment.mutex_watershed(affs, OFFSETS, bias=0.5, min_size=2).max() == 1


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


def test_a_connectivity_outside_1_to_3_is_refused():
    with pytest.raises(ValueError):
        segment.as_connectivity(4)
    with pytest.raises(ValueError):
        segment.as_connectivity("faces")
    assert segment.as_connectivity(3, ndim=2) == 2


def test_a_models_own_instances_are_numbered_from_1_without_gaps():
    labels = np.zeros((4, 4, 4), np.uint64)
    labels[0, 0, 0] = labels[3, 3, 3] = 2**40  # one id in two places
    labels[1, 1, 1] = 7
    relabelled = segment.relabel_instances(labels)
    assert sorted(np.unique(relabelled).tolist()) == [0, 1, 2]
    assert relabelled[0, 0, 0] == relabelled[3, 3, 3]
    split = segment.relabel_instances(labels, split_disconnected=True)
    assert split.max() == 3 and split[0, 0, 0] != split[3, 3, 3]
    with pytest.raises(ValueError):
        segment.relabel_instances(labels.astype(np.float32))


def test_touching_instances_keep_their_ids_apart_when_split_by_connectivity():
    labels = np.zeros((2, 2, 4), np.uint32)
    labels[..., :2], labels[..., 2:] = 5, 9
    assert segment.relabel_instances(labels, split_disconnected=True).max() == 2
