"""Intensity augmentation stays inside the data's own range.

It clipped every patch to [0, dtype max] (floats to [0, max(1, max)]), so
signed and float raw lost their whole negative half, and uint16 data using
0-4000 got noise of 655 -- a sixth of its real range -- because the noise was
1% of the dtype's range rather than the data's.
"""

from types import SimpleNamespace

import numpy as np

from cellmap_flow.finetune.virtual_dataset import VirtualPatchDataset
from cellmap_flow.norm.input_normalize import MinMaxNormalizer


def _augment(patch, normalizers=()):
    fake = SimpleNamespace(_input_normalizers=list(normalizers), _aug_pending={})
    return VirtualPatchDataset._augment_intensity(fake, patch, np.random.default_rng(0))


def test_float_raw_keeps_its_negative_half():
    patch = np.random.default_rng(1).uniform(-1, 1, (16, 16, 16)).astype(np.float32)
    out = _augment(patch)
    assert (out < 0).mean() > 0.4


def test_signed_integer_raw_keeps_its_negative_values():
    patch = np.random.default_rng(1).integers(-500, 500, (16, 16, 16)).astype(np.int16)
    out = _augment(patch)
    assert (out < 0).mean() > 0.4


def test_noise_is_relative_to_the_normalizers_window():
    patch = np.full((32, 32, 32), 2000, dtype=np.uint16)
    out = _augment(patch, [MinMaxNormalizer(min_value=0, max_value=4000)])
    residual = out - out.mean()
    assert 20 < residual.std() < 80, "1% of the 0-4000 window, not of 0-65535"
    assert out.min() >= 0 and out.max() <= 4000


def test_uint8_is_still_kept_within_0_255():
    patch = np.random.default_rng(1).integers(200, 256, (16, 16, 16)).astype(np.uint8)
    out = _augment(patch)
    assert out.min() >= 0 and out.max() <= 255
