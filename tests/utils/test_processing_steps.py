"""Fixes to individual normalizers and postprocessors."""

import numpy as np
import pytest

from cellmap_flow.norm.input_normalize import ChannelSelector, EuclideanDistance
from cellmap_flow.post.postprocessors import (
    AffinityPostprocessor,
    ChannelSelection,
    DefaultPostprocessor,
    SimpleBlockwiseMerger,
)


def test_steps_without_a_dtype_do_not_promote_to_float64():
    data = np.arange(16, dtype=np.uint8).reshape(2, 2, 2, 2)
    assert ChannelSelector(0)(data).dtype == np.uint8
    assert ChannelSelection("0")(data).dtype == np.uint8


def test_euclidean_distance_honours_its_parameters():
    mask = np.zeros((9, 9, 9), dtype=np.uint8)
    mask[2:7, 2:7, 2:7] = 1

    raw = EuclideanDistance(activation=None, black_border=False)(mask)
    assert raw.max() == pytest.approx(150.0)  # 3 voxels in at anisotropy 50

    squashed = EuclideanDistance(activation="tanh")(mask)
    assert squashed.max() <= 1.0

    signed = EuclideanDistance(type="sdf", activation=None)(mask)
    assert signed.min() < 0 < signed.max()

    # The dashboard sends "False" as a string.
    assert EuclideanDistance(black_border="False").black_border is False


def test_label_postprocessor_does_not_wrap_above_255():
    from cellmap_flow.post.postprocessors import LabelPostprocessor

    data = np.zeros((1, 3, 30, 30), dtype=np.uint8)
    data[0, ::2, ::2, ::2] = 1  # 2 x 15 x 15 = 450 isolated voxels
    labels = LabelPostprocessor(channel=0)(
        data, chunk_corner=(0, 0, 0), chunk_num_voxels=data[0].size
    )
    assert labels.dtype == np.uint32
    assert labels.max() == 450
    assert data.max() == 1, "the input array must not be overwritten"


@pytest.mark.parametrize("shape", [(6, 6, 6), (2, 6, 6, 6)])
def test_fill_holes_fills_what_a_blob_encloses(shape):
    from cellmap_flow.post.postprocessors import FillHolesPostprocessor

    logits = np.full(shape, -1.0, dtype=np.float32)
    logits[..., 1:5, 1:5, 1:5] = 1.0
    logits[..., 2, 2, 2] = -1.0  # an enclosed hole
    expected = np.zeros(shape, dtype=np.uint8)
    expected[..., 1:5, 1:5, 1:5] = 1

    filled = FillHolesPostprocessor(threshold="0")(logits)
    assert filled.dtype == np.uint8
    np.testing.assert_array_equal(filled, expected)


def _affinities(shape=(3, 8, 8, 8)):
    affs = np.full(shape, 0.9, dtype=np.float32)
    affs[:, :, :, 4] = 0.05  # a wall of repulsive edges down the middle
    return affs


def _previous_affinity_postprocessor(data, bias, neighborhood, chunk_num_voxels, chunk_corner):
    """AffinityPostprocessor._process as it was before input-scale detection."""
    import fastremap
    import mwatershed as mws
    import pymorton
    from scipy.ndimage import measurements

    data = data / 255.0
    neighborhood = neighborhood[: data.shape[0]]
    segmentation = mws.agglom(data.astype(np.float64) - bias, neighborhood)
    average_affs = np.mean(data, axis=0)
    fragment_ids = fastremap.unique(segmentation[segmentation > 0])
    keep = [
        f
        for f, mean in zip(
            fragment_ids, measurements.mean(average_affs, segmentation, fragment_ids)
        )
        if mean >= bias
    ]
    fastremap.mask_except(segmentation, keep, in_place=True)
    fastremap.renumber(segmentation, in_place=True)
    increment = chunk_num_voxels * pymorton.interleave(*chunk_corner)
    segmentation[segmentation > 0] += np.uint64(increment)
    return np.expand_dims(segmentation.astype(np.uint64), axis=0)


def test_affinity_postprocessor_is_unchanged_behind_default_postprocessor():
    """The usual chain (uint8 0-255 in) must give exactly what it always did."""
    rng = np.random.default_rng(1)
    affs = np.clip(_affinities((9, 12, 12, 12)) + rng.normal(0, 0.1, (9, 12, 12, 12)), 0, 1)
    as_uint8 = DefaultPostprocessor(
        clip_min=0.0, clip_max=1.0, bias=0.0, multiplier=255.0
    )(affs)
    assert as_uint8.dtype == np.uint8

    post = AffinityPostprocessor(bias=0.5)
    kwargs = dict(chunk_num_voxels=512, chunk_corner=(1, 2, 3))
    got = post(as_uint8, **kwargs)
    expected = _previous_affinity_postprocessor(
        as_uint8, 0.5, post.neighborhood, **kwargs
    )
    assert got.dtype == expected.dtype
    assert np.array_equal(got, expected)


def test_affinity_postprocessor_accepts_probabilities_directly():
    affs = _affinities()
    as_uint8 = (affs * 255).astype(np.uint8)
    kwargs = dict(chunk_num_voxels=512, chunk_corner=(0, 0, 0))

    from_float = AffinityPostprocessor(bias=0.5)(affs, **kwargs)
    from_uint8 = AffinityPostprocessor(bias=0.5)(as_uint8, **kwargs)

    # Two halves either side of the wall, not one merged blob.
    assert len(np.unique(from_float[from_float > 0])) == 2
    assert np.array_equal(from_float > 0, from_uint8 > 0)
    assert len(np.unique(from_float)) == len(np.unique(from_uint8))


def test_affinity_postprocessor_does_not_shrink_its_neighborhood():
    post = AffinityPostprocessor(bias=0.5)
    kwargs = dict(chunk_num_voxels=512, chunk_corner=(0, 0, 0))
    post(_affinities((3, 8, 8, 8)), **kwargs)
    assert len(post.neighborhood) == 9


def test_simple_blockwise_merger_survives_concurrent_chunks():
    """The server runs one merger instance from every request thread.

    Iterating ``keys_to_skip`` while another thread added to it raised
    "Set changed size during iteration". The set here yields slowly so the
    threads really do interleave.
    """
    import threading
    import time

    class SlowSet(set):
        def __iter__(self):
            for item in super().__iter__():
                time.sleep(0.001)
                yield item

    merger = SimpleBlockwiseMerger()
    merger.keys_to_skip = SlowSet()
    errors = []

    def serve(corner):
        block = np.ones((1, 4, 4, 4), dtype=np.uint64)
        try:
            merger(block, chunk_corner=corner)
        except Exception as e:  # noqa: BLE001 - any failure fails the test
            errors.append(e)

    # Neighbouring chunks, so every call adds matched faces to keys_to_skip.
    corners = [(z, y, 0) for z in range(4) for y in range(4)]
    for _ in range(3):
        threads = [threading.Thread(target=serve, args=(c,)) for c in corners]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert not errors, errors[0]
    assert len(merger.keys_to_skip) > 0


def test_channel_selection_accepts_what_yaml_gives():
    assert ChannelSelection([0, 2]).channels == [0, 2]
    assert ChannelSelection(1).channels == [1]
    assert ChannelSelection("0,2").channels == [0, 2]


def test_model_advice_matches_what_affinity_postprocessor_accepts():
    from cellmap_flow.utils.output_probe import UNBOUNDED, UNIT, review_postprocess

    def level(output_class, chain):
        return review_postprocess(
            output_class, chain, out_channels=3, model_name="mito_aff"
        )["level"]

    # Probabilities straight into it are fine now...
    assert level(UNBOUNDED, ["SigmoidPostprocessor", "AffinityPostprocessor"]) == "ok"
    assert level(UNIT, ["AffinityPostprocessor"]) == "ok"
    # ...logits are not.
    assert level(UNBOUNDED, ["AffinityPostprocessor"]) == "warn"


def test_the_steps_that_import_their_libraries_lazily_still_run():
    """The segmentation libraries are imported inside the steps (see
    test_import_hygiene); the steps still work, and the merger still pickles."""
    import pickle

    from cellmap_flow.post.postprocessors import LabelPostprocessor, MortonSegmentationRelabeling

    data = np.zeros((1, 4, 4, 4), dtype=np.uint8)
    data[0, 1:3, 1:3, 1:3] = 1
    assert LabelPostprocessor()(data, chunk_corner=(0, 0, 0), chunk_num_voxels=64).max() == 1
    relabeled = MortonSegmentationRelabeling()(data, chunk_corner=(1, 0, 0), chunk_num_voxels=np.int64(64))
    assert relabeled.max() == 1 + 64
    merger = SimpleBlockwiseMerger(face_erosion_iterations=1)
    merger(data.astype(np.uint64), chunk_corner=(0, 0, 0))
    merger.equivalences.union(1, 2)
    assert pickle.loads(pickle.dumps(merger)).equivalences_json() == merger.equivalences_json()
