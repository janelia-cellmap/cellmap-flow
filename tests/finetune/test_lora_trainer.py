"""LoRAFinetuner on the CPU: what it learns from, which epoch it keeps, what
stays frozen, and how a run survives a NaN, an OOM, a resume or a stop.

The losses of every epoch, the exported names and the served weights are
pinned by test_training_snapshot.
"""

import logging
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from cellmap_flow.finetune.adaptation import LoraStrategy, cpu_state_copy, frozen_teacher_copy
from cellmap_flow.finetune.target_transforms import BinaryTargetTransform


def _net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv3d(1, 2, 3, padding=1), nn.ReLU(), nn.Conv3d(2, 1, 1))


def _constant(logit, sigmoid=False):
    """Predicts ``logit`` everywhere; at learning rate 0 it stays there."""
    conv = nn.Conv3d(1, 1, 1)
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.fill_(logit)
    return nn.Sequential(conv, nn.Sigmoid()) if sigmoid else conv


def _annotation(fill, n=1, size=4, **labels):
    """``n`` annotations of size^3 (0 unannotated, 1 background, 2 foreground):
    ``fill``, then each of ``labels`` (name=(value, where))."""
    ann = torch.full((n, 1, size, size, size), float(fill))
    for value, where in labels.values():
        ann[where] = value
    return ann


def _bce(p, t):
    return -(t * math.log(p) + (1 - t) * math.log(1 - p))


P2, P3 = 1 / (1 + math.exp(-2.0)), 1 / (1 + math.exp(-3.0))
FIRST_VOXEL = np.s_[..., 0, 0, 0]
FOREGROUND = _annotation(2)
BACKGROUND = _annotation(1)
BOTH_CLASSES = _annotation(1, fg=(2, np.s_[..., :2]))
NOTHING_ANNOTATED = _annotation(0)
ONE_VOXEL_ANNOTATED = _annotation(0, fg=(2, FIRST_VOXEL))
ONE_FG_IN_A_THOUSAND = _annotation(1, size=10, fg=(2, FIRST_VOXEL))
THEN_AN_EMPTY_PATCH = _annotation(0, n=2, fg=(2, np.s_[0]))  # a foreground patch, then an unannotated one


def _train_at_rate_0(make_trainer, model, ann, **settings):
    trainer = make_trainer(model, (torch.rand(ann.shape), ann), learning_rate=0.0,
                           **{"target_transform": BinaryTargetTransform(), **settings})
    trainer.train()
    return trainer


@pytest.mark.parametrize("model, ann, settings, expected", [
    # MSE compares probabilities: on a logit model it trained the logits toward 0/1.
    pytest.param(_constant(3.0), FOREGROUND, dict(loss_type="mse"), (P3 - 1) ** 2, id="mse on logits"),
    pytest.param(_constant(3.0, sigmoid=True), FOREGROUND, dict(loss_type="mse"), (P3 - 1) ** 2,
                 id="mse on a model ending in a sigmoid"),
    # Classes are balanced as annotated: one fg voxel weighs as much as 999 bg,
    # whatever the smoothing did to their targets.
    pytest.param(_constant(2.0), ONE_FG_IN_A_THOUSAND, dict(balance_classes=True, label_smoothing=0.1),
                 (_bce(P2, 0.95) + _bce(P2, 0.05)) / 2, id="balanced bce under label smoothing"),
    # A batch with nothing supervised (rehearsal patches only) is not a 0 in the epoch's mean.
    pytest.param(_constant(2.0), THEN_AN_EMPTY_PATCH, {}, -math.log(P2), id="a batch with nothing supervised"),
    pytest.param(_constant(2.0), NOTHING_ANNOTATED, {}, float("nan"), id="an epoch with nothing supervised"),
    # Without a transform or masking (the trainer's own defaults) the annotation is the target as it stands.
    pytest.param(_constant(2.0), BACKGROUND, dict(target_transform=None, mask_unannotated=False),
                 -math.log(P2), id="the annotation as it stands"),
])
def test_the_supervised_loss_of_an_epoch(make_trainer, model, ann, settings, expected):
    """What picks the best checkpoint."""
    trainer = _train_at_rate_0(make_trainer, model, ann, **settings)
    if math.isnan(expected):
        assert math.isnan(trainer.last_supervised_loss)
    else:
        assert trainer.last_supervised_loss == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("ann, warns", [
    pytest.param(FOREGROUND, True, id="foreground only"),
    pytest.param(BACKGROUND, True, id="background only"),
    pytest.param(BOTH_CLASSES, False, id="both classes"),
    pytest.param(ONE_VOXEL_ANNOTATED, True, id="unannotated voxels do not rescue it"),
    pytest.param(NOTHING_ANNOTATED, True, id="nothing supervised at all"),
    pytest.param(THEN_AN_EMPTY_PATCH, True, id="said once, for the first batch"),
])
def test_a_target_of_one_class_is_called_out(make_trainer, caplog, ann, warns):
    """Three sessions in a row painted only foreground. Every instruction the
    model then gets says "raise this", and it comes out worse than it started,
    with nothing in the pipeline saying a word. The first batch warns, once."""
    _train_at_rate_0(make_trainer, _constant(2.0), ann)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and r.name.endswith("lora_trainer")]
    assert len(warnings) == (1 if warns else 0)


def test_the_best_epoch_is_the_best_supervised_one(make_trainer):
    """Distillation is 0 until the model moves, so ranking epochs by the whole
    objective kept epoch 1, one optimizer step from the start, in every run."""
    torch.manual_seed(0)
    ann = _annotation(0, fg=(2, np.s_[..., :2]))  # the rest is unannotated, and distilled
    trainer = make_trainer(nn.Conv3d(1, 1, 1), (torch.rand(ann.shape), ann), num_epochs=2,
                           learning_rate=0.1, distillation_lambda=100.0,
                           target_transform=BinaryTargetTransform())
    trainer.train()
    first, second = trainer.training_stats
    assert first["loss"] < second["loss"], "the objective rose after epoch 1"
    assert second["best_loss"] < first["best_loss"], "while the supervised loss fell"
    best = torch.load(trainer.output_dir / "best_checkpoint.pth", weights_only=False)
    assert best["epoch"] + 1 == 2


class _RehearsalFirst(torch.utils.data.Dataset):
    """One patch: a rehearsal patch, nothing annotated, for its first ``draws`` draws; then both classes."""

    def __init__(self, draws):
        self.rehearsals, self.raw = draws, torch.rand(1, 4, 4, 4, generator=torch.Generator().manual_seed(0))

    def __len__(self):
        return 1

    def __getitem__(self, i):
        self.rehearsals -= 1
        return self.raw, (NOTHING_ANNOTATED if self.rehearsals >= 0 else BOTH_CLASSES)[0]


def test_an_epoch_that_supervised_nothing_is_never_the_best(make_trainer):
    """Ranked by its total loss, lambda * distillation alone (~0, and exactly 0
    at lambda 0), an epoch of rehearsal patches only was "best" for the rest of
    the run, and the export shipped it."""
    # Two rehearsal draws: the trainer's output probe's, and epoch 1's.
    trainer = make_trainer(_net(), torch.utils.data.DataLoader(_RehearsalFirst(2)), num_epochs=3,
                           learning_rate=0.1, distillation_lambda=0.0, target_transform=BinaryTargetTransform())
    trainer.train()
    best = torch.load(trainer.output_dir / "best_checkpoint.pth", weights_only=False)
    later = trainer.training_stats[1:]
    assert best["epoch"] + 1 == min(later, key=lambda s: s["loss"])["epoch"]
    assert trainer.best_loss == min(s["loss"] for s in later)


@pytest.mark.parametrize("ann, num_epochs", [
    pytest.param(NOTHING_ANNOTATED, 2, id="no epoch supervised anything"),
    pytest.param(FOREGROUND, 0, id="no epoch ran"),  # a restart asking for 0 epochs
])
def test_without_a_best_epoch_the_export_is_this_runs_last_weights(make_trainer, tmp_path, ann, num_epochs):
    """A job's iterations share the output directory, and the export loaded the
    best checkpoint an earlier one left there: it shipped that iteration's
    weights as this one's, or failed to load them after a change of rank."""
    make_trainer(_net(), (torch.rand(1, 1, 4, 4, 4), FOREGROUND), learning_rate=0.1).train()
    trainer = make_trainer(_net(), (torch.rand(1, 1, 4, 4, 4), ann), num_epochs=num_epochs,
                           distillation_lambda=0.0, target_transform=BinaryTargetTransform())
    trainer.train()
    last = cpu_state_copy(trainer.model)
    exported = torch.load(trainer.save_adapter(export_dir=str(tmp_path / "export")), weights_only=True)
    assert all(torch.equal(v, last[k]) for k, v in exported.items())


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


def _bn_net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv3d(1, 4, 3, padding=1), nn.BatchNorm3d(4), nn.ReLU(),
                         nn.Dropout(0.5), nn.Conv3d(4, 1, 1))


@pytest.mark.parametrize("lora, num_epochs, moves", [
    pytest.param(True, 2, False, marks=pytest.mark.finetune, id="lora"),
    pytest.param(False, 2, True, id="full"),
    # The output probe's 100x noise, in train mode, multiplied the variance ~7000x.
    pytest.param(False, 0, False, id="full, the startup probes alone"),
])
def test_only_training_a_full_finetune_moves_the_batchnorm_statistics(make_trainer, lora, num_epochs, moves):
    """model.train() put the frozen base's BatchNorm in train mode: its running
    statistics, which the adapter does not save, drifted from the served base."""
    model = LoraStrategy(2, 4, 0.0).prepare(_bn_net()) if lora else _bn_net()
    bn = next(m for m in model.modules() if isinstance(m, nn.BatchNorm3d))
    before = torch.cat([bn.running_mean, bn.running_var])
    ann = _annotation(1, n=2, size=6, fg=(2, np.s_[:, :, :3]))
    make_trainer(model, (torch.rand(ann.shape) * 5, ann), batch_size=2, num_epochs=num_epochs,
                 target_transform=BinaryTargetTransform()).train()
    assert torch.equal(torch.cat([bn.running_mean, bn.running_var]), before) != moves


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


def test_no_teacher_is_made_without_distillation(make_trainer):
    ann = _half_background(n=2)
    trainer = make_trainer(_net(), (torch.rand(ann.shape), ann), distillation_lambda=0.0,
                           target_transform=BinaryTargetTransform())
    assert trainer.teacher_model is None


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


@pytest.mark.parametrize("num_epochs, epochs", [
    pytest.param(5, [4, 5], id="the epochs after the checkpoint's"),  # --resume re-ran all of them
    pytest.param(3, [], id="none, and no loss to print"),  # formatting None raised TypeError
])
def test_a_resumed_run_carries_on_after_its_checkpoint(make_trainer, tmp_path, num_epochs, epochs):
    data = (torch.rand(1, 1, 4, 4, 4), FOREGROUND)
    first = make_trainer(nn.Conv3d(1, 1, 1), data, output_dir=str(tmp_path / "first"))
    first.current_epoch = 2  # epochs 1-3 are done
    first.save_checkpoint(is_best=True)
    resumed = make_trainer(nn.Conv3d(1, 1, 1), data, num_epochs=num_epochs)
    resumed.load_checkpoint(str(tmp_path / "first" / "best_checkpoint.pth"))
    stats = resumed.train()
    assert [s["epoch"] for s in resumed.training_stats] == epochs
    assert (stats["final_loss"] is None) == (not epochs)


class _StopsDuringEpoch1(nn.Conv3d):
    """Writes the dashboard's stop signal from inside epoch 1 (its first call is the output probe)."""

    def __init__(self, signal):
        super().__init__(1, 1, 1)
        self.signal, self.calls = signal, 0

    def forward(self, x):
        self.calls += 1
        if self.calls == 2:
            self.signal.write_text("{}")
        return super().forward(x)


def test_the_dashboards_stop_signal_ends_the_run_after_the_current_epoch(make_trainer, tmp_path):
    trainer = make_trainer(_StopsDuringEpoch1(tmp_path / "run" / "stop_signal.json"),
                           (torch.rand(1, 1, 4, 4, 4), FOREGROUND), num_epochs=3)
    trainer.train()
    assert [s["epoch"] for s in trainer.training_stats] == [1]
    assert not (tmp_path / "run" / "stop_signal.json").exists(), "the signal is used up"
