"""The finetune's training loop: epochs, the best epoch, Stop Early.

``Finetuner`` trains a model on a session's corrections with the settings of
a ``TrainingOptions``: a LoRA adapter, or every weight (which of the two the
model decides, see adaptation.strategy_for). An epoch is: fetch a batch,
make its target, forward, the supervised term (supervised.SupervisedLoss),
the distillation term (distillation.Distillation), step, record. Around it:

- recovery from an OOM (halve the batch, then drop distillation) and from a
  NaN under mixed precision (restart in fp32): recovery.py;
- a best checkpoint, chosen by the supervised loss alone, which the export
  loads first; the strategies write and read the files (adaptation.py);
- TensorBoard, continuing across a job's iterations: tensorboard.py;
- the single-class warning, the first batch's summary and the
  gradient-flow watch: diagnostics.py.
"""

import logging
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from cellmap_flow.finetune import markers
from cellmap_flow.finetune.adaptation import strategy_for
from cellmap_flow.finetune.options import TrainingOptions
from cellmap_flow.finetune.trainer.diagnostics import GradientFlow, describe_first_batch, warn_if_single_class
from cellmap_flow.finetune.trainer.distillation import Distillation
from cellmap_flow.finetune.trainer.recovery import Precision, probe_precision, survive_oom
from cellmap_flow.finetune.trainer.supervised import SupervisedLoss
from cellmap_flow.finetune.trainer.tensorboard import TensorBoard
from cellmap_flow.utils.device import best_device

logger = logging.getLogger(__name__)


class Finetuner:
    """Trains a model on user corrections: a LoRA adapter, or every weight.

    Which of the two the model decides: a PEFT model trains its adapter, a
    plain module trains in full (see adaptation.strategy_for).

    Features:
    - Mixed precision on CUDA: bf16 where the GPU has it, else fp16 with a
      GradScaler; it falls back to fp32 when the model NaNs under either
    - Gradient accumulation to simulate larger batch sizes
    - Partial annotation support (mask unannotated regions)
    - Distillation toward the starting model, on unlabeled voxels, on all
      voxels, or on the good regions' rehearsal patches
    - A best checkpoint, chosen by the supervised loss alone; the export
      loads it first
    - Recovery from an OOM (halve the batch, then drop distillation) and
      from a NaN under mixed precision (restart in fp32)
    - TensorBoard logging, continuing across a job's iterations

    Args:
        model: The model to train: a PEFT model (LoRA) or a plain module (full finetune)
        dataloader: DataLoader for training data
        output_dir: Directory to save checkpoints and logs
        options: The training settings (finetune/options.py); the defaults if None.
            Every one of them is read here: the epochs, batch and learning rate, the
            loss and its margin, smoothing and class balance, the distillation weight
            and scope, mask_unannotated, select_channel, mixed_precision,
            tensorboard, and output_class, what the model's output is (None reads
            it as logits; the training job probes the model when nothing declared it)
        device: Training device ("cuda" or "cpu", auto-detected if None)
        target_transform: Optional TargetTransform instance that converts raw annotations
            to (target, mask) pairs. Overrides mask_unannotated when provided.
            See cellmap_flow.finetune.target_transforms.
        teacher_model: A full finetune's frozen teacher from a previous iteration, to reuse
        initial_state: A full finetune's starting weights, to reset to; copied from the model if None
        tb_start_step, tb_start_epoch: Where the previous iteration's TensorBoard curves ended
        stop_requested: Asked before each epoch, the first included, whether
            to stop there (Stop Early); the training job's says whether the
            dashboard asked (session_loop.TrainingSession). None: never
        **settings: Fields of TrainingOptions to set over ``options`` (``num_epochs=3``, say)

    Examples:
        >>> lora_model = wrap_model_with_lora(model)
        >>> dataloader = create_dataloader("corrections.zarr")
        >>> trainer = Finetuner(lora_model, dataloader, output_dir="output/fly_organelles_v1.1")
        >>> trainer.train()
        >>> trainer.save_adapter()
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        output_dir: str,
        options: Optional[TrainingOptions] = None,
        *,
        device: Optional[str] = None,
        target_transform=None,
        teacher_model: Optional[nn.Module] = None,
        initial_state: Optional[Dict[str, torch.Tensor]] = None,
        tb_start_step: int = 0,
        tb_start_epoch: int = 0,
        stop_requested: Optional[Callable[[], bool]] = None,
        **settings,
    ):
        options = TrainingOptions() if options is None else options
        if settings:
            options = options.replaced(**settings)
        self.options = options
        self.model = model
        self.dataloader = dataloader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs = options.num_epochs
        self.gradient_accumulation_steps = options.gradient_accumulation_steps
        self.select_channel = options.select_channel
        self.stop_requested = stop_requested

        self.device = best_device() if device is None else torch.device(device)
        logger.info(f"Using device: {self.device}")
        self.model = self.model.to(self.device)

        # LoRA or full, and everything that differs between them. The model
        # decides (see adaptation.strategy_for).
        self.strategy = strategy_for(self.model, teacher_model=teacher_model)

        # A full finetune changes the weights themselves, so resetting it --
        # after a NaN, or for a restart -- needs the weights it started from.
        # Kept on the CPU. LoRA resets by re-initialising its adapter instead.
        self.initial_state = (
            initial_state if initial_state is not None else self.strategy.initial_state(self.model)
        )

        # Optimizer, over the parameters that train: the adapter's, or every
        # weight in a full finetune.
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=options.learning_rate,
        )

        self.precision = Precision(self.device, options.mixed_precision)
        self.supervised = SupervisedLoss(options, target_transform)
        self.distillation = Distillation(
            options,
            anchors_available=bool(getattr(getattr(dataloader, "dataset", None), "emits_anchor", False)),
            model=self.model, strategy=self.strategy, device=self.device, precision=self.precision,
        )

        # Training state
        self.current_epoch = 0
        # Where train() starts counting; load_checkpoint() moves it past the
        # checkpoint's epoch.
        self._start_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        # Whether best_checkpoint.pth is this run's: written by this trainer,
        # or the one it resumed from. The output directory is shared by a
        # job's iterations, so the file can be an earlier iteration's, and
        # save_adapter must not export that.
        self._has_best_checkpoint = False
        # Average supervised loss of the epoch just finished. Checkpoint
        # selection uses this rather than the combined loss -- see the epoch
        # loop for why the combined loss cannot rank epochs.
        self.last_supervised_loss = float('nan')
        self.training_stats = []
        self._target_checked = False
        self._last_epoch_timing = (0.0, 0.0)

        self.tb = TensorBoard(self.output_dir / "tensorboard", options.tensorboard, tb_start_step, tb_start_epoch)

    @property
    def distillation_lambda(self) -> float:
        """The distillation term's weight as used (see Distillation)."""
        return self.distillation.weight

    @property
    def teacher_model(self) -> Optional[nn.Module]:
        """A full finetune's distillation teacher, the frozen copy of its starting weights; else None."""
        return self.distillation.teacher_model

    @teacher_model.setter
    def teacher_model(self, value):
        self.distillation.teacher_model = value

    @staticmethod
    def log(msg):
        """Log to console (the job's tee writes the log file).

        Timestamped like the logger's lines so epoch duration can be read
        off the log: the 2026-09-23 A/B runs had none on the per-epoch
        summaries, and per-arm speed had to come from LSF's start/end
        times instead. The progress parsers in finetune/job_manager use
        unanchored re.findall, so the prefix does not affect them.
        """
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{stamp} {msg}" if msg else msg, flush=True)

    def close(self):
        """Close the TensorBoard writer; the CLI calls this when an iteration is done."""
        self.tb.close()

    def _config_rows(self):
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return [
            ("epochs", self.num_epochs),
            ("batch size", getattr(self.dataloader, "batch_size", "?")),
            ("gradient accumulation", self.gradient_accumulation_steps),
            ("learning rate", self.options.learning_rate),
            ("mixed precision", self.precision.describe()),
            ("loss", self.supervised.loss_type),
            ("margin", self.supervised.margin),
            ("balance classes", self.supervised.balance_classes),
            ("label smoothing", self.supervised.label_smoothing),
            ("distillation lambda", self.distillation.weight),
            ("mask unannotated", self.supervised.mask_unannotated),
            ("trainable params", f"{trainable:,} of {total:,} ({100 * trainable / max(total, 1):.2f}%)"),
            ("device", str(self.device)),
        ]

    def _reset_training_state(self):
        """Reset the weights (in place), optimizer, and training counters for a fresh start."""
        self.model = self.strategy.reset(self.model, self.initial_state)
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.optimizer.defaults['lr'],
        )
        self.current_epoch = 0
        self._start_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self._has_best_checkpoint = False
        self.last_supervised_loss = float('nan')
        self.training_stats = []

    def _set_train_mode(self):
        """Train mode, except for the frozen base's norm layers under LoRA.

        A full finetune trains its norm layers, so it keeps them in train
        mode (see strategy.train_mode).
        """
        self.strategy.train_mode(self.model)

    def _diverged(self, final_loss, epochs, start_time) -> Dict[str, Any]:
        """The stats of a run that cannot go on, announced to the job manager.

        Every diverged return says so: the job manager watches for this
        marker, and without it this path looked like training that simply
        went quiet.
        """
        markers.emit(markers.TRAINING_DIVERGED)
        return {
            'final_loss': final_loss,
            'best_loss': self.best_loss,
            'total_epochs': epochs,
            'total_steps': self.global_step,
            'training_time': time.time() - start_time,
            'diverged': True,
        }

    def train(self) -> Dict[str, Any]:
        """
        Run the training loop.

        Returns:
            Training statistics dictionary with:
            - final_loss: Final epoch loss
            - best_loss: Best loss achieved
            - total_epochs: Number of epochs trained
            - total_steps: Total training steps
        """
        log = self.log
        log_file = self.output_dir / "training_log.txt"
        log("=" * 60)
        log(f"Starting {'LoRA' if self.strategy.kind == 'lora' else 'full'} finetuning")
        log("=" * 60)
        log(f"Epochs: {self.num_epochs}")
        log(f"Batches per epoch: {len(self.dataloader)}")
        log(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        log(f"Effective batch size: {self.dataloader.batch_size * self.gradient_accumulation_steps}")
        log(f"Mixed precision: {self.precision.describe()}")
        log(f"Mask unannotated regions: {self.supervised.mask_unannotated}")
        log(f"Log file: {log_file}")
        if self.tb.on:
            log(f"TensorBoard: tensorboard --logdir {self.output_dir.parent}   (this run: {self.tb.dir})")
            self.tb.config_card(self._config_rows())
        log("")

        start_time = time.time()

        # The probe only looks at what the model computes, so it runs in
        # eval mode. In train mode a full finetune's BatchNorm added its
        # batch to its running statistics (see supervised.probe_output_class
        # for the numbers). LoRA's frozen norms are in eval mode either way.
        self.model.eval()
        try:
            probe_precision(self)
        finally:
            self._set_train_mode()

        # Resumed runs carry on after the checkpoint's epoch. This looped
        # from 0 regardless, so --resume re-ran every epoch it had already
        # done, and the checkpoint's epoch number was only ever logged.
        epoch_loss = None
        # The last epoch done, which is what a checkpoint records: none yet,
        # or the one a resume carries on after. The loop moves it only once
        # an epoch starts. Moved before the stop check, a stop between two
        # epochs recorded the next one as done, and --resume skipped it.
        self.current_epoch = self._start_epoch - 1
        for epoch in range(self._start_epoch, self.num_epochs):
            # User-requested graceful stop: drop out of the training loop so
            # the outer flow (inference server + wait for restart) kicks in.
            # Asked before the first epoch too: a stop asked for while the
            # job loaded the model used to be deleted here as stale, and the
            # run trained every epoch.
            if self.stop_requested is not None and self.stop_requested():
                log(
                    f"Stop signal received at epoch {epoch+1}/{self.num_epochs}; "
                    f"exiting training loop early."
                )
                break
            self.current_epoch = epoch
            log(f"Starting epoch {epoch+1} of {self.num_epochs}...")
            # Mitigation loop: keep making room (halve the batch, then drop
            # distillation) and retrying until the epoch succeeds or there is
            # nothing left to try (see recovery.survive_oom).
            epoch_loss = None
            while True:
                try:
                    epoch_loss = self._train_epoch()
                    break
                except torch.cuda.OutOfMemoryError:
                    if not survive_oom(self, epoch):
                        return self._diverged(float('nan'), epoch, start_time)

            # Handle NaN/Inf loss
            if not math.isfinite(epoch_loss):
                if self.precision.enabled:
                    # NaN likely caused by FP16 overflow on specific data —
                    # fall back to FP32 and restart training from scratch
                    log(
                        f"WARNING: NaN loss at epoch {epoch+1} under FP16 — "
                        f"falling back to FP32 and restarting training."
                    )
                    self.precision.fall_back_to_fp32()
                    self._reset_training_state()
                    return self.train()

                log(f"ERROR: Loss is {epoch_loss} at epoch {epoch+1}. Stopping training.")
                return self._diverged(epoch_loss, epoch + 1, start_time)

            # Rank epochs by the supervised term, not the combined loss.
            #
            # loss = supervised + lambda * distillation, and the distillation
            # term is minimized by *not changing the model*: at init LoRA has
            # B=0, so the student is identical to the teacher and distillation
            # is exactly 0. Epoch 1 therefore posts a combined loss no later
            # epoch can beat, and "best" stayed pinned to epoch 1 for the whole
            # run. save_adapter() loads best_checkpoint.pth before exporting,
            # so every finetune shipped a model one optimizer step from its
            # starting point -- measurably so: every LoRA B matrix came out at
            # max|B| = 1e-4, which is Adam's first step at lr=1e-4.
            #
            # Distillation belongs in the objective, where it restrains the
            # update; it cannot also be the yardstick for which epoch is best.
            # With lambda=0 the two terms are equal, so this changes nothing.
            #
            # An epoch that supervised nothing -- every patch a rehearsal
            # patch, or no annotated voxel in any -- has a NaN supervised loss
            # and is not ranked at all. Ranking it by its total loss instead,
            # lambda * distillation alone (about 0, or exactly 0 at lambda 0),
            # made it "best" for the rest of the run, for the same reason.
            selection_loss = self.last_supervised_loss
            supervised = math.isfinite(selection_loss)

            # Log epoch results. On soft targets the BCE cannot go below the
            # target's entropy, so also say how far above that floor it sits.
            bce_metrics = self.supervised.epoch_bce_metrics()
            bce_extra = ""
            if bce_metrics is not None:
                epoch_floor, epoch_mae = bce_metrics
                bce_extra = f" - Above floor: {selection_loss - epoch_floor:.6f} - MAE: {epoch_mae:.6f}"
            log(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Loss: {epoch_loss:.6f} - "
                f"Supervised: {selection_loss:.6f} - "
                f"Best supervised: {self.best_loss:.6f}"
                f"{bce_extra}"
                + ("" if supervised else " (nothing supervised: not ranked)")
            )

            # Save checkpoint if best
            if supervised and selection_loss < self.best_loss:
                self.best_loss = selection_loss
                log("  Saving best checkpoint...")
                self.save_checkpoint(is_best=True)
                log("  → Saved best checkpoint")

            # Save regular checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(is_best=False)

            self.training_stats.append({
                'epoch': epoch + 1,
                'loss': epoch_loss,
                'best_loss': self.best_loss,
            })

            if self.tb.on:
                data_wait, compute = self._last_epoch_timing
                self.tb.log_epoch(
                    loss=epoch_loss, supervised=selection_loss if supervised else None,
                    best_supervised=self.best_loss, bce_metrics=bce_metrics,
                    data_wait_s=data_wait, compute_s=compute,
                    peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9 if self.device.type == "cuda" else None,
                )

        if epoch_loss is not None and not self._has_best_checkpoint:
            log(
                "WARNING: no epoch had a supervised voxel (every patch was a "
                "rehearsal patch, or nothing annotated was sampled), so there is "
                "no best epoch. The export will be the last epoch's weights."
            )

        # Final checkpoint
        self.save_checkpoint(is_best=False)

        total_time = time.time() - start_time
        log("")
        log("=" * 60)
        log("Training Complete!")
        log(f"Total time: {total_time/60:.2f} minutes")
        self.tb.flush()
        log(f"Best loss: {self.best_loss:.6f}")
        # None when no epoch ran: a stop requested before the first one, or a
        # resume from a finished run. Formatting it used to raise TypeError.
        log(f"Final loss: {epoch_loss:.6f}" if epoch_loss is not None else "Final loss: n/a (no epoch ran)")
        log(f"Output directory: {self.output_dir}")
        log("=" * 60)

        return {
            'final_loss': epoch_loss,
            'best_loss': self.best_loss,
            'total_epochs': self.num_epochs,
            'total_steps': self.global_step,
            'training_time': total_time,
        }

    def _abort_epoch(self, why) -> float:
        """Drop the epoch before a non-finite value reaches the weights; its loss is NaN.

        This check used to run after scaler.step(), and under bf16 or fp32 --
        where the scaler is off and so does not skip inf/NaN steps itself --
        AdamW had already written NaN into every trainable weight. LoRA
        recovered on restart by re-making its adapter; a full finetune kept
        NaN weights, served them, and trained on from them.
        """
        logger.warning(f"{why}; skipping the update and aborting the epoch.")
        self.optimizer.zero_grad(set_to_none=True)
        self.last_supervised_loss = float('nan')
        return float('nan')

    def _step(self):
        """One optimizer step over the accumulated gradients."""
        self.precision.scaler.step(self.optimizer)
        self.precision.scaler.update()
        self.optimizer.zero_grad()
        self.global_step += 1

    def _train_epoch(self) -> float:
        """Train for one epoch and return average loss."""
        epoch_loss = 0.0
        epoch_supervised_loss = 0.0
        # Batches that had any supervised voxel. The supervised mean ranks
        # epochs for the best checkpoint; a batch made only of rehearsal
        # patches has nothing supervised and a supervised loss of exactly 0,
        # so averaging over every batch made "best epoch" partly track how
        # many rehearsal draws an epoch happened to get.
        supervised_batches = 0
        epoch_distill_loss = 0.0
        num_batches = len(self.dataloader)
        accumulate = self.gradient_accumulation_steps
        self.supervised.reset_epoch()
        watch = GradientFlow(self.model)

        # Fetch explicitly so the wait on the loader is measurable. That is
        # the number that says whether prefetching keeps up, and it was lost
        # when the loss_history.csv instrumentation fell out of the tree.
        epoch_data_wait = 0.0
        epoch_compute = 0.0
        batch_iter = iter(self.dataloader)
        for batch_idx in range(num_batches):
            t_fetch = time.time()
            batch = next(batch_iter)
            t_after_fetch = time.time()
            epoch_data_wait += t_after_fetch - t_fetch
            # The dataset yields a third tensor once the session has good
            # regions: a per-voxel mask marking where the student should be
            # held to the teacher. Older datasets yield the 2-tuple, so both
            # shapes have to work.
            if len(batch) == 3:
                raw, annotation, anchor = batch
                anchor = anchor.to(self.device, non_blocking=True)
            else:
                raw, annotation = batch
                anchor = None
            raw = raw.to(self.device, non_blocking=True)
            annotation = annotation.to(self.device, non_blocking=True)

            target, mask, bounds, hard_target = self.supervised.targets(annotation)
            if not self._target_checked:
                self._target_checked = True
                if not self.supervised.use_flow:
                    describe_first_batch(target, mask)
                    # By what is annotated, not by what this batch drew. A
                    # dataset that cannot say (a test's tensors) is not checked.
                    class_voxels = getattr(self.dataloader.dataset, "class_voxels", None)
                    if class_voxels is not None:
                        warn_if_single_class(*class_voxels)

            # Teacher forward pass for distillation (before student pass): the
            # model as it was before this run.
            teacher_pred = self.distillation.teacher_prediction(raw) if self.distillation.on else None

            # Student forward pass with mixed precision
            with self.precision.autocast():
                pred = self.model(raw)
                if not torch.isfinite(pred).all():
                    logger.warning(f"NaN/Inf in student pred! range=[{pred.min():.4f}, {pred.max():.4f}]")
                # Select specific channel if requested (e.g., mito = channel 2 from 8-channel output)
                if self.select_channel is not None:
                    pred = pred[:, self.select_channel:self.select_channel+1]

            # Losses in fp32, outside autocast; only the forward pass runs in
            # reduced precision. Two reasons. Models with a built-in sigmoid
            # (the cellmap *_distance_* UNets) make the trainer swap to
            # BCELoss on probabilities, and autocast refuses to run
            # binary_cross_entropy at all. And a bf16 probability near 0 or 1
            # has too few mantissa bits for log(p) / log(1-p) to mean much.
            with autocast('cuda', enabled=False):
                pred = pred.float()
                supervised_loss = self.supervised(pred, target, mask, bounds, hard_target)
                if not torch.isfinite(supervised_loss):
                    logger.warning(f"NaN/Inf supervised_loss: {supervised_loss.item()}")
                loss = supervised_loss
                distill_loss = torch.tensor(0.0, device=self.device)
                if teacher_pred is not None:
                    distill_loss = self.distillation.term(pred, teacher_pred.float(), mask, anchor)
                    loss = loss + self.distillation.weight * distill_loss
                # Scale loss for gradient accumulation
                loss = loss / accumulate

            if self.tb.on and batch_idx == 0 and self.current_epoch % self.tb.image_every == 0:
                # A flow model's first two channels are flows; its foreground is the third.
                self.tb.images(raw, target, pred, mask, channel=2 if self.supervised.use_flow else 0,
                               output_class=self.supervised.output_class)

            # A non-finite loss must never reach the optimizer (see _abort_epoch).
            if not torch.isfinite(loss):
                return self._abort_epoch(f"NaN/Inf loss at epoch {self.current_epoch+1}, batch {batch_idx+1}")

            # Backward pass
            self.precision.scaler.scale(loss).backward()
            watch.after_backward()

            # Update weights after accumulation
            if (batch_idx + 1) % accumulate == 0:
                if not self.precision.gradients_finite(self.model):
                    return self._abort_epoch(
                        f"NaN/Inf gradient at epoch {self.current_epoch+1}, batch {batch_idx+1}"
                    )
                self._step()
                if self.tb.on:
                    self.tb.log_step(
                        loss=loss.item() * accumulate, supervised=supervised_loss.item(),
                        distillation=distill_loss.item() if self.distillation.on else None,
                        bce_metrics=self.supervised.step_bce_metrics,
                        interval_terms=self.supervised.step_interval_terms,
                        lr=self.optimizer.param_groups[0]["lr"],
                        step_s=time.time() - t_after_fetch, data_wait_s=t_after_fetch - t_fetch,
                    )

            # Accumulate losses (unscaled)
            batch_loss = loss.item() * accumulate
            # .item() above synchronised the device, so this is real compute time.
            epoch_compute += time.time() - t_after_fetch
            self._last_epoch_timing = (epoch_data_wait, epoch_compute)
            if not math.isfinite(batch_loss):
                return self._abort_epoch(f"NaN/Inf loss at epoch {self.current_epoch+1}, batch {batch_idx+1}")
            epoch_loss += batch_loss
            epoch_distill_loss += distill_loss.item()
            if mask is None or bool(mask.sum() > 0):
                supervised_batches += 1
                epoch_supervised_loss += supervised_loss.item()
                self.supervised.tally()

            # Log progress every batch (since we have few batches)
            avg_loss = epoch_loss / (batch_idx + 1)
            if self.distillation.on:
                avg_sup = epoch_supervised_loss / max(supervised_batches, 1)
                avg_distill = epoch_distill_loss / (batch_idx + 1)
                self.log(
                    f"  Batch {batch_idx+1}/{num_batches} - "
                    f"Loss: {avg_loss:.6f} (sup: {avg_sup:.6f}, distill: {avg_distill:.6f})"
                )
            else:
                self.log(f"  Batch {batch_idx+1}/{num_batches} - Loss: {avg_loss:.6f}")

        # Handle leftover accumulated gradients at end of epoch
        # (in case num_batches is not divisible by gradient_accumulation_steps)
        if num_batches % accumulate != 0:
            if not self.precision.gradients_finite(self.model):
                return self._abort_epoch("NaN/Inf gradient in the last accumulation step")
            self._step()

        watch.report(self.log)

        # NaN when nothing in the epoch was supervised: train() then does not
        # rank the epoch for the best checkpoint.
        self.last_supervised_loss = (
            epoch_supervised_loss / supervised_batches if supervised_batches else float('nan')
        )
        return epoch_loss / num_batches

    def save_checkpoint(self, is_best: bool = False):
        """
        Save training checkpoint.

        Args:
            is_best: If True, saves as "best_checkpoint.pth", which save_adapter
                exports; otherwise as "checkpoint_epoch_<N>.pth"
        """
        name = "best_checkpoint.pth" if is_best else f"checkpoint_epoch_{self.current_epoch+1}.pth"
        # What the strategy keeps: LoRA its adapter and optimizer state every
        # time; a full finetune only its best weights (see strategy.checkpoint).
        written = self.strategy.save_checkpoint(
            self.output_dir / name, self.model, self.optimizer, self.precision.scaler, is_best,
            epoch=self.current_epoch, global_step=self.global_step, best_loss=self.best_loss,
            training_stats=self.training_stats,
        )
        if written and is_best:
            self._has_best_checkpoint = True

    def save_adapter(self, adapter_path: Optional[str] = None, export_dir: Optional[str] = None):
        """
        Export the finetune: only the LoRA adapter, or a full finetune's weights.

        Automatically loads the best checkpoint weights before saving
        so the exported adapter reflects the best training epoch. Without one
        from this run -- no epoch supervised anything, or none ran -- it
        exports the weights as they are, those of the last epoch.

        Args:
            adapter_path: Path to save adapter. If None, uses output_dir/lora_adapter.
                A full finetune ignores it and writes output_dir/full_finetune.
            export_dir: Instead, the directory to export into: the adapter
                goes to export_dir/lora_adapter, full weights to
                export_dir/full_finetune/model_state_dict.pt. The CLI gives
                every iteration its own.

        Returns:
            The adapter directory, or the full-finetune weights file.
        """
        base = Path(export_dir) if export_dir is not None else self.output_dir
        if export_dir is not None:
            adapter_path = None

        # Load best checkpoint weights before saving, if they are this run's.
        # An earlier iteration's (the output directory is shared) was exported
        # as this one, or failed to load after a change of rank.
        best = self.output_dir / "best_checkpoint.pth"
        if self._has_best_checkpoint and best.exists():
            checkpoint = self.strategy.load_checkpoint(best, self.model, device=self.device)
            logger.info(
                f"Loaded best checkpoint (epoch {checkpoint['epoch'] + 1}, loss "
                f"{checkpoint['best_loss']:.6f}) before saving adapter"
            )
        else:
            logger.warning("No best checkpoint from this run; exporting the final weights")

        # The adapter, or for a full finetune the whole state dict, where
        # FinetuneModelConfig(weights_path=...) expects it.
        return str(self.strategy.export(self.model, base, path=adapter_path))

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint to resume training.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = self.strategy.load_checkpoint(
            checkpoint_path, self.model, self.optimizer, self.precision.scaler, device=self.device
        )
        self.current_epoch = checkpoint['epoch']
        self._start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        # It had a best epoch, whose best_checkpoint.pth this run carries on.
        self._has_best_checkpoint = math.isfinite(self.best_loss)
        self.training_stats = checkpoint.get('training_stats', [])

        logger.info(f"Checkpoint loaded from: {checkpoint_path}")
        logger.info(f"Resuming after epoch {self.current_epoch+1}, at epoch {self._start_epoch+1}")


# The class's old name, for the CLI and the tests.
LoRAFinetuner = Finetuner
