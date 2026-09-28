"""Output type, channel count and --select-channel are checked before training.

--output-type binary on a multi-channel model without --select-channel only
warned, and BCE then died on the first batch ("Target size [2, 1, ...] must
be the same as input size [2, 3, ...]"). --select-channel with distance or
binary_broadcast sliced the prediction to one channel while the target kept
all of them -- the same crash.
"""

import argparse
from types import SimpleNamespace

import pytest
import torch

from cellmap_flow.finetune.finetune_cli import _build_target_transform
from cellmap_flow.finetune.target_transforms import (
    BinaryTargetTransform,
    BroadcastBinaryTargetTransform,
    DistanceTargetTransform,
)


def _model(channels=3):
    return SimpleNamespace(config=SimpleNamespace(output_channels=channels))


def _args(**kw):
    base = dict(output_type="binary", select_channel=None, loss_type="bce", offsets=None,
                model_script=None, label_smoothing=0.0, distance_sigma=6.0, mask_unannotated=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_distance_on_a_selected_channel_makes_a_one_channel_target():
    transform = _build_target_transform(_args(output_type="distance", select_channel=1), _model())
    assert isinstance(transform, DistanceTargetTransform)
    ann = torch.ones(2, 1, 6, 6, 6)
    ann[:, :, :3] = 2
    target, mask = transform(ann)
    assert target.shape == (2, 1, 6, 6, 6) and mask.shape == (2, 1, 6, 6, 6)


def test_broadcast_on_a_selected_channel_is_one_channel():
    transform = _build_target_transform(
        _args(output_type="binary_broadcast", select_channel=0), _model()
    )
    assert isinstance(transform, BroadcastBinaryTargetTransform)
    assert transform.num_channels == 1


@pytest.mark.parametrize("loss", ["bce", "mse", "combined"])
def test_binary_on_many_channels_with_a_shape_strict_loss_is_refused(loss):
    with pytest.raises(ValueError, match="select-channel"):
        _build_target_transform(_args(loss_type=loss), _model())


@pytest.mark.parametrize("loss", ["dice", "margin"])
def test_binary_on_many_channels_still_works_where_it_did(loss):
    assert isinstance(_build_target_transform(_args(loss_type=loss), _model()), BinaryTargetTransform)


def test_a_channel_that_does_not_exist_is_refused():
    with pytest.raises(ValueError, match="out of range"):
        _build_target_transform(_args(select_channel=3), _model(3))


def test_affinities_cannot_select_a_channel():
    with pytest.raises(ValueError, match="affinities"):
        _build_target_transform(
            _args(output_type="affinities", offsets="[[1, 0, 0]]", select_channel=0), _model()
        )
