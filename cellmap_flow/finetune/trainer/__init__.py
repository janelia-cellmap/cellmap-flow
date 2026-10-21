"""The finetune trainer, in its parts.

- ``loop.Finetuner``: the epoch loop, the best epoch, the stop signal;
- ``supervised.SupervisedLoss``: the supervised term, and how the model's
  output is read for it (``probe_output_class``);
- ``distillation.Distillation``: the term that holds the model to the one
  it started from, and its teacher;
- ``recovery``: mixed precision and the way back to fp32, out-of-memory
  mitigation, the finite-gradient check;
- ``diagnostics``: the gradient-flow watch and the single-class warning;
- ``tensorboard.TensorBoard``: the run's event files.

``LoRAFinetuner`` is ``Finetuner``'s old name, kept for the CLI and tests.
"""

from cellmap_flow.finetune.trainer.loop import Finetuner, LoRAFinetuner

__all__ = ["Finetuner", "LoRAFinetuner"]
