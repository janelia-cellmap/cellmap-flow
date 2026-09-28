"""ImageDataInterface reads through an explicit chain, per read."""

import numpy as np
import tensorstore as ts
from funlib.geometry import Roi

from cellmap_flow.globals import g
from cellmap_flow.image_data_interface import ImageDataInterface
from cellmap_flow.norm.input_normalize import ChannelSelector, LambdaNormalizer
from tests.utils.serving_helpers import write_raw

ROI = Roi((0, 0, 0), (16, 16, 16))


def test_normalize_false_is_honoured_on_roi_reads(tmp_path):
    idi = ImageDataInterface(
        write_raw(tmp_path, np.full((4, 4, 4), 7, dtype=np.uint8)), normalize=False
    )
    g.input_norms = [LambdaNormalizer("x * 2")]
    assert np.all(idi.to_ndarray_ts(ROI) == 7)


def test_whole_array_reads_are_normalized_too(tmp_path):
    idi = ImageDataInterface(write_raw(tmp_path, np.full((4, 4, 4), 7, dtype=np.uint8)))
    g.input_norms = [LambdaNormalizer("x * 2")]
    assert np.all(idi.to_ndarray_ts() == 14)
    assert np.all(idi.to_ndarray_ts(ROI) == 14)


def test_an_explicit_chain_overrides_g(tmp_path):
    idi = ImageDataInterface(write_raw(tmp_path, np.full((4, 4, 4), 7, dtype=np.uint8)))
    g.input_norms = [LambdaNormalizer("x * 2")]
    tripled = idi.with_input_norms([LambdaNormalizer("x * 3")])

    assert np.all(tripled.to_ndarray_ts(ROI) == 21)
    assert np.all(idi.to_ndarray_ts(ROI) == 14)  # the original is untouched
    # Both views share one opened store.
    assert tripled._raw_ts() is idi._raw_ts()


def test_channel_is_chosen_on_every_read_not_at_open(tmp_path):
    data = np.stack([np.full((4, 4, 4), 1, np.uint8), np.full((4, 4, 4), 2, np.uint8)])
    idi = ImageDataInterface(write_raw(tmp_path, data))

    g.input_norms = []
    assert np.all(idi.to_ndarray_ts(ROI) == 1)
    # A server that has already read a chunk must follow a new ChannelSelector.
    g.input_norms = [ChannelSelector(1)]
    assert np.all(idi.to_ndarray_ts(ROI) == 2)
    assert np.all(np.asarray(idi.ts[...]) == 2)
    assert np.all(idi.with_input_norms([]).to_ndarray_ts(ROI) == 1)


def test_channel_axis_is_found_by_its_label():
    from cellmap_flow.utils.ds import select_channel

    data = np.zeros((4, 4, 4, 2), dtype=np.uint8)
    data[..., 1] = 5
    store = ts.array(data)[ts.d[:].label["z", "y", "x", "c"]]

    picked = select_channel(store, 1)
    assert picked.shape == (4, 4, 4)
    assert np.all(picked.read().result() == 5)


def test_view_dtype_is_the_last_declared_one(tmp_path):
    from cellmap_flow.norm.input_normalize import MinMaxNormalizer

    idi = ImageDataInterface(write_raw(tmp_path, np.full((4, 4, 4), 7, dtype=np.uint8)))
    # ChannelSelector declares no dtype; MinMax's float32 still reaches the viewer.
    view = idi.with_input_norms([MinMaxNormalizer(), ChannelSelector(0)])
    assert view.ts.dtype == np.float32
    assert view.to_ndarray_ts(ROI).dtype == np.float32
    assert idi.with_input_norms([]).ts.dtype == np.uint8
