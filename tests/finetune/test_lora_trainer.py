"""Finetuner on the CPU: what it distils toward, how it reads the model's
output, how a run survives a NaN or an OOM, and what its single-class
warning goes by."""

import logging

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from cellmap_flow.finetune.adaptation import cpu_state_copy, frozen_teacher_copy
from cellmap_flow.finetune.losses import masked_mean
from cellmap_flow.finetune.target_transforms import BinaryTargetTransform


def _net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv3d(1, 2, 3, padding=1), nn.ReLU(), nn.Conv3d(2, 1, 1))


def _annotation(fill, n=1, size=4, **labels):
    """``n`` annotations of size^3 (0 unannotated, 1 background, 2 foreground):
    ``fill``, then each of ``labels`` (name=(value, where))."""
    ann = torch.full((n, 1, size, size, size), float(fill))
    for value, where in labels.values():
        ann[where] = value
    return ann


FOREGROUND = _annotation(2)


@pytest.mark.parametrize("accumulate", [pytest.param(1, id="a step per batch"),
                                        pytest.param(2, id="gradient accumulation")])
def test_a_non_finite_batch_never_reaches_the_weights(make_trainer, accumulate):
    """The check ran after the optimizer step: with the scaler off, AdamW had
    already written NaN into every weight, and a full finetune served them."""
    raw = torch.rand(4, 1, 5, 5, 5)
    raw[2:] = float("nan")
    ann = _annotation(1, n=4, size=5, fg=(2, np.s_[:, :, :2]))
    trainer = make_trainer(_net(), (raw, ann), batch_size=2, num_epochs=2,
                           gradient_accumulation_steps=accumulate, target_transform=BinaryTargetTransform())
    assert trainer.train()["diverged"]
    assert all(torch.isfinite(p).all() for p in trainer.model.parameters())


def test_a_tanh_model_is_read_as_its_server_reads_it(make_trainer):
    """The trainer had a sigmoid probe of its own, which took a tanh's [-1, 1]
    for logits, while ranking and seeding read it as (p + 1) / 2
    (serving.probe's signed_unit): one model, trained on one reading and
    thresholded on another. The trainer takes the server's class now, and
    the job probes with the same classifier when the dashboard had none."""
    from cellmap_flow.finetune.trainer.supervised import probe_output_class
    from cellmap_flow.serving.probe import SIGNED_UNIT, classify_output_range

    model = nn.Sequential(nn.Conv3d(1, 1, 1), nn.Tanh())
    raw = torch.rand(2, 1, 4, 4, 4)
    out = model(raw).detach()
    assert probe_output_class(model, raw) == SIGNED_UNIT == classify_output_range(float(out.min()), float(out.max()))

    ann = _annotation(1, n=2, fg=(2, np.s_[:, :, :2]))
    trainer = make_trainer(model, (raw, ann), output_class=SIGNED_UNIT, target_transform=BinaryTargetTransform(),
                           balance_classes=False)
    target, mask, bounds, hard_target = trainer.supervised.targets(ann)
    loss = trainer.supervised(out, target, mask, bounds, hard_target)
    expected = masked_mean(F.binary_cross_entropy((out + 1) / 2, target, reduction="none"), mask)
    assert torch.allclose(loss, expected)


class _Anchors(torch.utils.data.Dataset):
    """A dataset with good regions to anchor on (``emits_anchor``), or without."""

    def __init__(self, emits_anchor):
        self.emits_anchor = emits_anchor

    def __len__(self):
        return 1

    def __getitem__(self, i):
        raw = ann = torch.zeros(1, 4, 4, 4)
        return (raw, ann, torch.ones(1, 4, 4, 4)) if self.emits_anchor else (raw, ann)


@pytest.mark.parametrize("anchors, given, used", [
    # Without a teacher term a rehearsal patch trains nothing: the regions would do nothing.
    pytest.param(True, None, 1.0, id="regions and no weight given"),
    pytest.param(True, 0.0, 0.0, id="regions and 0 (it became 1.0 too)"),
    pytest.param(True, 0.4, 0.4, id="regions and a weight"),
    pytest.param(False, None, 0.0, id="no regions and no weight given"),
    pytest.param(False, 0.0, 0.0, id="no regions and 0"),
])
def test_the_distillation_weight_when_good_regions_are_marked(make_trainer, anchors, given, used):
    trainer = make_trainer(nn.Conv3d(1, 1, 1), torch.utils.data.DataLoader(_Anchors(anchors)),
                           distillation_lambda=given)
    assert trainer.distillation_lambda == used


def _half_background(n=4):
    """Background on half the patch, a foreground corner in it, the rest unannotated: something to distil."""
    return _annotation(0, n=n, size=6, bg=(1, np.s_[:, :, :3]), fg=(2, np.s_[:, :, :2, :2]))


@pytest.mark.parametrize("given", [pytest.param(False, id="made"), pytest.param(True, id="given")])
def test_a_full_finetune_distils_toward_a_frozen_copy_of_its_starting_weights(make_trainer, given):
    """The teacher was the model with its adapter off, which a full finetune does
    not have, so its first batch died. It is a frozen copy of the starting
    weights, made once (the CLI hands it to the next iteration)."""
    model = _net()
    start = cpu_state_copy(model)
    teacher = frozen_teacher_copy(model) if given else None
    ann = _half_background()
    trainer = make_trainer(model, (torch.rand(ann.shape), ann), batch_size=2, num_epochs=3,
                           learning_rate=1e-2, distillation_lambda=0.5, teacher_model=teacher,
                           target_transform=BinaryTargetTransform())
    assert not trainer.train().get("diverged")
    if given:
        assert trainer.teacher_model is teacher
    assert trainer.teacher_model is not model and not trainer.teacher_model.training
    assert not any(p.requires_grad for p in trainer.teacher_model.parameters())
    assert all(torch.equal(v, start[k]) for k, v in trainer.teacher_model.state_dict().items())
    assert any(not torch.equal(v, start[k]) for k, v in model.state_dict().items()), "the student moved"


class _OutOfMemory(nn.Module):
    """Runs out of memory on a batch larger than ``fits``."""

    def __init__(self, fits):
        super().__init__()
        self.conv, self.fits = nn.Conv3d(1, 1, 1), fits

    def forward(self, x):
        if x.shape[0] > self.fits:
            raise torch.cuda.OutOfMemoryError("out of memory")
        return self.conv(x)


@pytest.mark.parametrize("fits, diverged", [pytest.param(1, False, id="halved"),
                                            pytest.param(0, True, id="given up at batch 1")])
def test_an_out_of_memory_epoch_is_retried_on_half_batches(make_trainer, monkeypatch, capsys, fits, diverged):
    """At batch 1 and no distillation there is nothing left to try: that is a
    divergence, and says so with the marker the job manager watches for."""
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)  # no CUDA device here
    trainer = make_trainer(_OutOfMemory(fits), (torch.rand(4, 1, 4, 4, 4), _annotation(2, n=4)),
                           batch_size=2 if fits else 1)
    assert bool(trainer.train().get("diverged")) == diverged
    assert ("TRAINING_DIVERGED" in capsys.readouterr().out) == diverged
    if not diverged:
        assert (trainer.dataloader.batch_size, trainer.gradient_accumulation_steps) == (1, 2)


@pytest.mark.parametrize("class_voxels, warned", [
    pytest.param((5247, 4906), False, id="background painted, though not in the first batch"),
    pytest.param((0, 4906), True, id="no background painted"),
])
def test_the_single_class_warning_goes_by_what_is_annotated(make_trainer, caplog, class_voxels, warned):
    """It went by the first batch: a session painted with both, whose first
    eight patches held no background, was told to paint background, in every
    iteration."""
    patches = TensorDataset(torch.rand(1, 1, 4, 4, 4), FOREGROUND)
    patches.class_voxels = class_voxels  # (background, foreground), as VirtualPatchDataset counts them
    trainer = make_trainer(nn.Conv3d(1, 1, 1), DataLoader(patches), target_transform=BinaryTargetTransform())
    diagnostics = "cellmap_flow.finetune.trainer.diagnostics"
    with caplog.at_level(logging.WARNING, logger=diagnostics):
        trainer.train()
    assert any(r.name == diagnostics and r.levelno == logging.WARNING for r in caplog.records) == warned
