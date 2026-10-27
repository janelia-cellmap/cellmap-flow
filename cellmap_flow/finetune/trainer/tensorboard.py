"""A run's TensorBoard event files: the config card, a patch image strip, and
the per-step and per-epoch scalars.

File-based, so it works from GPU nodes with no network and needs no
service; one ``tensorboard --logdir`` over the training/ tree overlays every
run ever made. Silently off if unavailable or not asked for.
"""

import logging

import torch

from cellmap_flow.finetune.losses import as_probabilities

logger = logging.getLogger(__name__)


class TensorBoard:
    """The writer, and the step and epoch it is at.

    Monotonic across restarts: the CLI makes a new trainer for every
    iteration, writing to the same tensorboard/ directory, and passes the
    previous one's position on (``start_step``, ``start_epoch``). Starting
    each at 0 drew every iteration's curves on top of each other.
    """

    image_every = 5  # epochs between patch images

    def __init__(self, log_dir, enabled=True, start_step=0, start_epoch=0):
        self.dir = log_dir
        self.step = int(start_step)
        self.epoch = int(start_epoch)
        self.writer = None
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(log_dir))
            except Exception as e:  # not installed, or logdir not writable
                logger.info(f"TensorBoard logging disabled: {e}")

    @property
    def on(self) -> bool:
        return self.writer is not None

    def config_card(self, rows):
        """The run's settings as a table under the "config" text tag."""
        table = "| setting | value |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)
        self.writer.add_text("config", table, self.epoch)

    @torch.no_grad()
    def images(self, raw, target, pred, mask, channel, output_class):
        """Mid-Z slice of one sample as one strip: raw (centre-cropped to the
        output) | target | prediction | mask, at the current epoch.

        This is the picture that would have shown augmentation doing nothing
        for five months, and that shows raw and labels moving together once
        it does something. Never lets a display problem stop training.
        """
        try:
            p = as_probabilities(pred[0, channel].detach().float(), output_class).cpu()
            t = target[0, channel].detach().float().cpu()
            r = raw[0, 0].detach().float().cpu()
            # Valid-padding models emit a smaller volume than they read.
            c = [(rs - ps) // 2 for rs, ps in zip(r.shape, p.shape)]
            r = r[c[0]:c[0] + p.shape[0], c[1]:c[1] + p.shape[1], c[2]:c[2] + p.shape[2]]
            z = p.shape[0] // 2
            r2 = r[z]
            r2 = (r2 - r2.min()) / (r2.max() - r2.min() + 1e-8)
            m2 = (
                mask[0, min(channel, mask.shape[1] - 1), z].detach().float().cpu().clamp(0, 1)
                if mask is not None else torch.zeros_like(r2)
            )
            # One strip per epoch, raw | target | prediction | mask, separated
            # by a white line: four tags per epoch was too much to scroll.
            sep = torch.ones(r2.shape[0], 2)
            strip = torch.cat([r2, sep, t[z].clamp(0, 1), sep, p[z].clamp(0, 1), sep, m2], dim=1)
            self.writer.add_image("patch/raw|target|prediction|mask", strip[None], self.epoch)
        except Exception as e:
            logger.debug(f"TensorBoard image logging skipped: {e}")

    def log_step(self, *, loss, supervised, distillation, other_channels, bce_metrics, interval_terms, lr,
                 step_s, data_wait_s):
        """One optimizer step's scalars, at the next step; ``distillation`` and
        ``other_channels`` (the hold on the channels not trained) None when off."""
        self.step += 1
        add = self.writer.add_scalar
        add("train/loss", loss, self.step)
        add("train/supervised", supervised, self.step)
        if bce_metrics is not None:
            floor, mae = bce_metrics
            add("train/bce_floor", floor, self.step)
            add("train/supervised_above_floor", supervised - floor, self.step)
            add("train/mean_abs_error", mae, self.step)
        if interval_terms is not None:
            add("train/interval_bounds", interval_terms[0], self.step)
            add("train/interval_slope", interval_terms[1], self.step)
        if distillation is not None:
            add("train/distillation", distillation, self.step)
        if other_channels is not None:
            add("train/other_channels", other_channels, self.step)
        add("train/lr", lr, self.step)
        add("time/step_s", step_s, self.step)
        add("time/data_wait_s", data_wait_s, self.step)

    def log_epoch(self, *, loss, supervised, best_supervised, bce_metrics, other_channels, data_wait_s, compute_s,
                  peak_memory_gb):
        """One epoch's scalars, at the next epoch; ``supervised`` None when nothing was
        supervised, ``other_channels`` (how far the channels not trained moved) None
        without the hold."""
        self.epoch += 1
        add = self.writer.add_scalar
        add("epoch/loss", loss, self.epoch)
        if supervised is not None:
            add("epoch/supervised", supervised, self.epoch)
        add("epoch/best_supervised", best_supervised, self.epoch)
        if bce_metrics is not None and supervised is not None:
            floor, mae = bce_metrics
            add("epoch/bce_floor", floor, self.epoch)
            add("epoch/supervised_above_floor", supervised - floor, self.epoch)
            add("epoch/mean_abs_error", mae, self.epoch)
        if other_channels is not None:
            add("epoch/other_channels", other_channels, self.epoch)
        add("time/epoch_data_wait_s", data_wait_s, self.epoch)
        add("time/epoch_compute_s", compute_s, self.epoch)
        if peak_memory_gb is not None:
            add("memory/peak_gb", peak_memory_gb, self.epoch)
        self.writer.flush()

    def flush(self):
        if self.writer is not None:
            self.writer.flush()

    def close(self):
        """Close the writer; the CLI calls this when an iteration is done."""
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception as e:
                logger.debug(f"Closing the TensorBoard writer failed: {e}")
            self.writer = None
