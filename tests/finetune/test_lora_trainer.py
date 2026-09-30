"""LoRAFinetuner on the CPU: what it learns from, which epoch it keeps, what
stays frozen, and how a run survives a NaN, an OOM, a resume or a stop.

The losses of every epoch, the exported names and the served weights are
pinned by test_training_snapshot.
"""

import logging
import math

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


def _ann(n=1, size=4, fill=0.0, **where):
    """Annotations (0 unannotated, 1 background, 2 foreground): ``fill``, then ``where``'s slices."""
    ann = torch.full((n, 1, size, size, size), float(fill))
    for value, index in where.values():
        ann[index] = value
    return ann


def _bce(p, t):
    return -(t * math.log(p) + (1 - t) * math.log(1 - p))


P2, P3 = 1 / (1 + math.exp(-2.0)), 1 / (1 + math.exp(-3.0))
ALL = (slice(None),)


@pytest.mark.parametrize("model, ann, settings, expected, warns", [
    # MSE compares probabilities: on a logit model it trained the logits toward 0/1.
    (_constant(3.0), _ann(fill=2), dict(loss_type="mse"), (P3 - 1) ** 2, 1),
    (_constant(3.0, sigmoid=True), _ann(fill=2), dict(loss_type="mse"), (P3 - 1) ** 2, 1),
    # Classes are balanced as annotated: one fg voxel weighs as much as 999 bg,
    # whatever the smoothing did to their targets.
    (_constant(2.0), _ann(size=10, fill=1, fg=(2, (0, 0, 0, 0, 0))),
     dict(balance_classes=True, label_smoothing=0.1), (_bce(P2, 0.95) + _bce(P2, 0.05)) / 2, 0),
    # A batch with nothing supervised adds nothing to the epoch's mean.
    (_constant(2.0), _ann(n=2, fg=(2, (0,))), dict(), -math.log(P2), 1),
    (_constant(2.0), _ann(fill=1), dict(), -math.log(1 - P2), 1),
    (_constant(2.0), _ann(fill=1, fg=(2, (ALL * 4 + (slice(0, 2),)))), dict(),
     (-math.log(P2) - math.log(1 - P2)) / 2, 0),
    (_constant(2.0), _ann(), dict(), float("nan"), 1),
    # Unannotated voxels are not supervision and cannot rescue the balance.
    (_constant(2.0), _ann(fg=(2, (0, 0, 0, 0, 0))), dict(), -math.log(P2), 1),
    # Without a transform or masking (the trainer's own defaults) the annotation is the target as it stands.
    (_constant(2.0), _ann(fill=1), dict(target_transform=None, mask_unannotated=False), -math.log(P2), 1),
], ids=["mse on logits", "mse on probabilities", "balanced and smoothed", "an empty batch",
        "background only", "both classes", "nothing annotated", "one annotated voxel", "as it stands"])
def test_the_supervised_loss_and_the_one_class_warning(make_trainer, caplog, model, ann, settings,
                                                        expected, warns):
    """The epoch's supervised loss picks the best checkpoint. A target with one
    class (three sessions painted only foreground) cannot teach a boundary: the
    first batch says so, once."""
    trainer = make_trainer(model, (torch.rand(ann.shape), ann), learning_rate=0.0,
                           **{"target_transform": BinaryTargetTransform(), **settings})
    trainer.train()
    if math.isnan(expected):
        assert math.isnan(trainer.last_supervised_loss)
    else:
        assert trainer.last_supervised_loss == pytest.approx(expected, rel=1e-4)
    warned = [r for r in caplog.records if r.levelno >= logging.WARNING and r.name.endswith("lora_trainer")]
    assert len(warned) == warns


def test_the_best_epoch_is_the_best_supervised_one(make_trainer):
    """Distillation is 0 until the model moves, so ranking epochs by the whole
    objective kept epoch 1, one optimizer step from the start, in every run."""
    torch.manual_seed(0)
    ann = _ann(fg=(2, ALL * 4 + (slice(0, 2),)))  # the rest is unannotated: distilled
    trainer = make_trainer(nn.Conv3d(1, 1, 1), (torch.rand(ann.shape), ann), num_epochs=2,
                           learning_rate=0.1, distillation_lambda=100.0,
                           target_transform=BinaryTargetTransform())
    trainer.train()
    first, second = trainer.training_stats
    assert first["loss"] < second["loss"] and second["best_loss"] < first["best_loss"]
    best = torch.load(trainer.output_dir / "best_checkpoint.pth", weights_only=False)
    assert best["epoch"] + 1 == 2


@pytest.mark.parametrize("accumulate", [1, 2])
def test_a_non_finite_batch_never_reaches_the_weights(make_trainer, accumulate):
    """The check ran after the optimizer step: with the scaler off, AdamW had
    already written NaN into every weight, and a full finetune served them."""
    raw = torch.rand(4, 1, 5, 5, 5)
    raw[2:] = float("nan")
    trainer = make_trainer(_net(), (raw, _ann(n=4, size=5, fill=1, fg=(2, ALL * 2 + (slice(0, 2),)))),
                           batch_size=2, num_epochs=2, gradient_accumulation_steps=accumulate,
                           target_transform=BinaryTargetTransform())
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
    (True, None, 1.0),  # regions without a teacher term would train nothing
    (True, 0.0, 0.0),   # "0 (Disabled)" used to become 1.0 too
    (True, 0.4, 0.4),
    (False, None, 0.0),
    (False, 0.0, 0.0),
])
def test_the_distillation_weight_when_good_regions_are_marked(make_trainer, anchors, given, used):
    trainer = make_trainer(nn.Conv3d(1, 1, 1), torch.utils.data.DataLoader(_Anchors(anchors)),
                           distillation_lambda=given)
    assert trainer.distillation_lambda == used


def _bn_net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv3d(1, 4, 3, padding=1), nn.BatchNorm3d(4), nn.ReLU(),
                         nn.Dropout(0.5), nn.Conv3d(4, 1, 1))


@pytest.mark.parametrize("lora", [pytest.param(True, marks=pytest.mark.finetune), False],
                         ids=["lora", "full"])
def test_only_a_full_finetune_moves_the_batchnorm_statistics(make_trainer, lora):
    """model.train() put the frozen base's BatchNorm in train mode: its running
    statistics, which the adapter does not save, drifted from the served base."""
    model = LoraStrategy(2, 4, 0.0).prepare(_bn_net()) if lora else _bn_net()
    bn = next(m for m in model.modules() if isinstance(m, nn.BatchNorm3d))
    before = bn.running_mean.clone()
    ann = _ann(n=2, size=6, fill=1, fg=(2, ALL * 2 + (slice(0, 3),)))
    make_trainer(model, (torch.rand(ann.shape) * 5, ann), batch_size=2, num_epochs=2,
                 target_transform=BinaryTargetTransform()).train()
    assert torch.equal(bn.running_mean, before) == lora


@pytest.mark.parametrize("given, weight", [(False, 0.5), (True, 0.5), (False, 0.0)],
                         ids=["made", "given", "no distillation"])
def test_a_full_finetune_distils_toward_its_starting_weights(make_trainer, given, weight):
    """The teacher was the model with its adapter off, which a full finetune does
    not have, so its first batch died. It is a frozen copy of the starting
    weights, made once (the CLI hands it to the next iteration), or none."""
    model = _net()
    start = cpu_state_copy(model)
    teacher = frozen_teacher_copy(model) if given else None
    ann = _ann(n=4, size=6, fg=(1, ALL * 2 + (slice(0, 3),)))
    ann[:, :, :2, :2] = 2
    trainer = make_trainer(model, (torch.rand(ann.shape), ann), batch_size=2, num_epochs=3,
                           learning_rate=1e-2, distillation_lambda=weight, teacher_model=teacher,
                           target_transform=BinaryTargetTransform())
    assert not trainer.train().get("diverged")
    if not weight:
        assert trainer.teacher_model is None
        return
    if given:
        assert trainer.teacher_model is teacher
    assert trainer.teacher_model is not model and not trainer.teacher_model.training
    assert not any(p.requires_grad for p in trainer.teacher_model.parameters())
    assert all(torch.equal(v, start[k]) for k, v in trainer.teacher_model.state_dict().items())
    assert any(not torch.equal(v, start[k]) for k, v in model.state_dict().items())


class _OutOfMemory(nn.Module):
    """Runs out of memory on a batch larger than ``fits``."""

    def __init__(self, fits):
        super().__init__()
        self.conv, self.fits = nn.Conv3d(1, 1, 1), fits

    def forward(self, x):
        if x.shape[0] > self.fits:
            raise torch.cuda.OutOfMemoryError("out of memory")
        return self.conv(x)


@pytest.mark.parametrize("fits, diverged", [(1, False), (0, True)], ids=["halved", "given up"])
def test_an_out_of_memory_epoch_is_retried_on_half_batches(make_trainer, monkeypatch, capsys,
                                                           fits, diverged):
    """At batch 1 and no distillation there is nothing left to try: that is a
    divergence, and says so with the marker the job manager watches for."""
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)  # no CUDA device here
    trainer = make_trainer(_OutOfMemory(fits), (torch.rand(4, 1, 4, 4, 4), _ann(n=4, fill=2)),
                           batch_size=2 if fits else 1)
    assert bool(trainer.train().get("diverged")) == diverged
    assert ("TRAINING_DIVERGED" in capsys.readouterr().out) == diverged
    if not diverged:
        assert (trainer.dataloader.batch_size, trainer.gradient_accumulation_steps) == (1, 2)


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


@pytest.mark.parametrize("resume_to, stop, epochs", [
    (5, False, [4, 5]),  # --resume re-ran every epoch the checkpoint had done
    (3, False, []),      # a finished run: no epoch, and no loss to print
    (None, True, [1]),
], ids=["resumed", "resumed when finished", "stopped"])
def test_which_epochs_run(make_trainer, tmp_path, resume_to, stop, epochs):
    data = (torch.rand(1, 1, 4, 4, 4), _ann(fill=2))
    if resume_to:
        first = make_trainer(nn.Conv3d(1, 1, 1), data, output_dir=str(tmp_path / "first"))
        first.current_epoch = 2
        first.save_checkpoint(is_best=True)
        trainer = make_trainer(nn.Conv3d(1, 1, 1), data, num_epochs=resume_to)
        trainer.load_checkpoint(str(tmp_path / "first" / "best_checkpoint.pth"))
    else:
        trainer = make_trainer(_StopsDuringEpoch1(tmp_path / "run" / "stop_signal.json"), data, num_epochs=3)
    stats = trainer.train()
    assert [s["epoch"] for s in trainer.training_stats] == epochs
    assert (stats["final_loss"] is None) == (not epochs)
