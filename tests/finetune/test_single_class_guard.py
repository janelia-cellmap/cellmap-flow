"""A run whose supervised target has only one class cannot work.

Three sessions in a row were trained on annotations containing label 2 and
nothing else. Through AffinityTargetTransform that becomes a target of 1 at
100% of supervised voxels: every instruction the model receives says "raise
this", so it raises everything, and the finetuned model is worse than the one
it started from. Nothing in the pipeline said a word.
"""

import logging

import torch

from cellmap_flow.finetune.lora_trainer import LoRAFinetuner
from cellmap_flow.finetune.target_transforms import AffinityTargetTransform


class _Bare:
    """Just enough trainer to exercise the check."""

    _single_class_checked = False
    _warn_if_single_class = LoRAFinetuner._warn_if_single_class


def _run(target, mask):
    t = _Bare()
    t._single_class_checked = False
    return t._warn_if_single_class(target, mask)


def test_an_all_positive_target_is_called_out(caplog):
    target = torch.ones(1, 1, 4, 4, 4)
    mask = torch.ones_like(target)
    with caplog.at_level(logging.WARNING):
        _run(target, mask)
    assert "EVERY supervised voxel is positive (1)" in caplog.text
    assert "Paint some background" in caplog.text


def test_an_all_negative_target_is_called_out(caplog):
    target = torch.zeros(1, 1, 4, 4, 4)
    mask = torch.ones_like(target)
    with caplog.at_level(logging.WARNING):
        _run(target, mask)
    assert "EVERY supervised voxel is negative (0)" in caplog.text
    assert "Paint some foreground" in caplog.text


def test_a_mixed_target_passes_quietly(caplog):
    target = torch.zeros(1, 1, 4, 4, 4)
    target[..., :2] = 1.0
    mask = torch.ones_like(target)
    with caplog.at_level(logging.WARNING):
        _run(target, mask)
    assert "EVERY supervised voxel" not in caplog.text


def test_only_masked_voxels_count(caplog):
    """Unmasked voxels are not supervision and must not rescue the balance."""
    target = torch.zeros(1, 1, 4, 4, 4)
    mask = torch.zeros_like(target)
    target[..., :1] = 1.0
    mask[..., :1] = 1.0  # the only supervised voxels, all positive
    with caplog.at_level(logging.WARNING):
        _run(target, mask)
    assert "EVERY supervised voxel is positive (1)" in caplog.text


def test_no_supervision_at_all_is_its_own_message(caplog):
    with caplog.at_level(logging.WARNING):
        _run(torch.zeros(1, 1, 4, 4, 4), torch.zeros(1, 1, 4, 4, 4))
    assert "No supervised voxels" in caplog.text


def test_it_only_speaks_once(caplog):
    t = _Bare()
    t._single_class_checked = False
    target, mask = torch.ones(1, 1, 2, 2, 2), torch.ones(1, 1, 2, 2, 2)
    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            t._warn_if_single_class(target, mask)
    assert caplog.text.count("EVERY supervised voxel") == 1


def test_foreground_only_painting_produces_an_all_positive_affinity_target():
    """The field case, end to end: label 2 everywhere, no label 1.

    Reproduces what three sessions actually contained, so the guard is tied
    to the real shape of the mistake rather than a synthetic one.
    """
    ann = torch.zeros(1, 1, 8, 8, 8)
    ann[0, 0, 2:5, 2:5, 2:5] = 2  # a painted blob, and nothing marked background

    target, mask = AffinityTargetTransform([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 3)(ann)
    supervised = float(mask.sum())
    positive = float((target * mask).sum())

    assert supervised > 0, "adjacent painted voxels should be supervised"
    assert positive == supervised, "with no background, every target is 1"
