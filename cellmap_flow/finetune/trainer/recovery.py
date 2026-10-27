"""Keeping a run alive: mixed precision and the way back to fp32, out-of-memory
mitigation, and the check that no non-finite gradient reaches the weights.

``Precision`` is the run's autocast dtype and GradScaler. The functions take
the trainer (``loop.Finetuner``), whose batch, accumulation, distillation
term and hold on the other channels they change.
"""

import logging

import torch
from torch.amp import GradScaler, autocast

logger = logging.getLogger(__name__)


class Precision:
    """Mixed precision on CUDA: bf16 where the GPU has it, else fp16 with a
    GradScaler; off on other devices, and off for good after a NaN
    (``fall_back_to_fp32``).

    bf16, because autocast's CUDA default, fp16, carries 5 exponent bits, so
    a UNet this deep overflows in the forward pass: every run NaN'd out on
    the startup probe and fell back to fp32, so "mixed precision" was never
    once in effect and the tensor cores sat idle for the whole job. bf16
    has fp32's 8 bits, which is exactly the failure mode it exists to fix,
    and needs no loss scaling. Ampere and newer only (H100/H200 = cc 9.0
    yes; the RTX 2080 Ti workstation = cc 7.5 no), so fp16 where it is
    unavailable, and the probe demotes to fp32 if that NaNs too. The
    GradScaler compensates for fp16's narrow range; bf16 does not need it,
    and enabling it there costs a little and hides real overflows.
    """

    def __init__(self, device, enabled: bool):
        self.device = device
        # CUDA AMP utilities (autocast('cuda'), GradScaler('cuda')) error or
        # behave unexpectedly on other devices.
        if device.type != "cuda" and enabled:
            logger.warning(f"Disabling mixed precision: device={device.type} is not CUDA.")
            enabled = False
        self.enabled = enabled
        self.amp_dtype = torch.float16
        if device.type == "cuda":
            try:
                if torch.cuda.is_bf16_supported():
                    self.amp_dtype = torch.bfloat16
            except Exception as e:
                logger.debug(f"bf16 support check failed ({e}); staying on fp16.")
        self.scaler = GradScaler('cuda', enabled=enabled and self.amp_dtype is torch.float16)

    def autocast(self, enabled=None):
        """The autocast context of a forward pass (``enabled`` False for a probe in fp32)."""
        return autocast('cuda', enabled=self.enabled if enabled is None else enabled, dtype=self.amp_dtype)

    def describe(self) -> str:
        """For the log and the TensorBoard card: "True (dtype=bfloat16, grad_scaler=False)" or "False (fp32)"."""
        if not self.enabled:
            return "False (fp32)"
        return (
            f"True (dtype={str(self.amp_dtype).replace('torch.', '')}, "
            f"grad_scaler={self.scaler.is_enabled()})"
        )

    def fall_back_to_fp32(self):
        """Disable mixed precision training."""
        self.enabled = False
        self.scaler = GradScaler('cuda', enabled=False)
        logger.warning(f"Mixed precision disabled (was {self.amp_dtype}); training in fp32.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def gradients_finite(self, model) -> bool:
        """Whether every accumulated gradient of ``model`` is finite.

        With fp16 the GradScaler already skips a step whose gradients
        overflowed, and it must see them scaled, so that case is left to it.
        """
        if self.scaler.is_enabled():
            return True
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads:
            return True
        return bool(torch.isfinite(torch.stack([g.float().norm() for g in grads])).all())


def probe_precision(trainer):
    """One forward pass under mixed precision before training; fp32 if it NaNs.

    Some model+data combinations produce NaN under fp16 autocast. On an OOM
    the batch is halved as well. Runs with the model in eval mode (the
    caller's doing), as the probe only looks at what the model computes.
    """
    precision = trainer.precision
    if not precision.enabled:
        return
    try:
        probe_raw = next(iter(trainer.dataloader))[0][:1].to(trainer.device)
        with torch.no_grad(), precision.autocast(enabled=True):
            probe_out = trainer.model(probe_raw)
        if not torch.isfinite(probe_out).all():
            trainer.log("WARNING: Model produces NaN/Inf under FP16 — falling back to FP32.")
            precision.fall_back_to_fp32()
        del probe_raw, probe_out
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        trainer.log("WARNING: FP16 probe OOM — falling back to FP32 with smaller batch.")
        precision.fall_back_to_fp32()
        halve_batch(trainer)
    except Exception as e:
        trainer.log(f"WARNING: FP16 probe failed ({e}) — falling back to FP32.")
        precision.fall_back_to_fp32()


def halve_batch(trainer) -> bool:
    """Halve the trainer's batch size and double its gradient accumulation, so the
    effective batch size stays; False if the batch is already 1."""
    old_bs = trainer.dataloader.batch_size
    new_bs = max(1, old_bs // 2)
    if new_bs >= old_bs:
        return False
    from cellmap_flow.finetune.data import rebuild_loader

    old_accum = trainer.gradient_accumulation_steps
    trainer.gradient_accumulation_steps = old_accum * (old_bs // new_bs)
    # Same workers, persistence and sampling as before, only smaller
    # batches (see rebuild_loader).
    trainer.dataloader = rebuild_loader(trainer.dataloader, new_bs)
    trainer.log(
        f"Halved batch size {old_bs} → {new_bs}, "
        f"gradient accumulation {old_accum} → {trainer.gradient_accumulation_steps} "
        f"(effective batch size unchanged)"
    )
    torch.cuda.empty_cache()
    return True


def survive_oom(trainer, epoch) -> bool:
    """Make room after an out-of-memory epoch: halve the batch, else drop
    distillation and the hold on the other channels, which free the teacher
    (its forward pass, and a full finetune's copy of the weights); False when
    nothing is left to try.

    Applied again and again until the epoch succeeds or there is nothing
    left: users can restart with a much larger rank and batch and the second
    attempt OOMs too. The optimizer's accumulated gradients are stale after
    an OOM, so they are cleared.
    """
    torch.cuda.empty_cache()
    if halve_batch(trainer):
        trainer.log(
            f"OOM at epoch {epoch + 1} — retrying with smaller batch size "
            f"(now {trainer.dataloader.batch_size})."
        )
    elif trainer.distillation.teacher_needed:
        dropped = []
        if trainer.distillation.on:
            dropped.append(f"disabling distillation (was lambda={trainer.distillation.weight})")
        if trainer.distillation.holds_others:
            dropped.append(
                f"no longer holding the channels other than {trainer.select_channel} to the starting "
                "model: they may drift from here on"
            )
        trainer.log(f"OOM at epoch {epoch + 1} and batch already at 1 — {'; '.join(dropped)}. Retrying.")
        trainer.distillation.drop()
    else:
        trainer.log("ERROR: OOM at batch=1 with no distillation or hold left to drop. Cannot continue.")
        return False
    trainer.optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return True
