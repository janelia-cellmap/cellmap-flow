"""One definition of the training settings: finetune/options.py's
TrainingOptions, which the form's body, the job's command line and the
trainer all read."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def test_a_blank_form_the_bare_command_line_and_the_trainer_agree(tmp_path):
    """The settings were listed seven times, form to trainer, and their defaults
    drifted (the form's margin loss and batch of 8 against the trainer's
    combined loss and the request's batch of 2): a field left blank trained
    with values the form never showed."""
    from cellmap_flow.dashboard.requests import FinetuneSubmit
    from cellmap_flow.finetune.cli import options_of, parse_args, training_flags
    from cellmap_flow.finetune.options import TrainingOptions
    from cellmap_flow.finetune.trainer import Finetuner

    blank = FinetuneSubmit.model_validate({"model_name": "m", "corrections_path": "c", "lora_r": "", "num_epochs": None})
    required = ["--corrections", "c", "--output-dir", "o"]
    loader = DataLoader(TensorDataset(torch.rand(1, 1, 4, 4, 4), torch.full((1, 1, 4, 4, 4), 2.0)))
    trainer = Finetuner(nn.Conv3d(1, 1, 1), loader, str(tmp_path), device="cpu", tensorboard=False)

    defaults = TrainingOptions()
    assert blank.options() == options_of(parse_args(required)) == trainer.options.replaced(tensorboard=True) == defaults
    # The job's command line says every setting, and parses back to the same.
    assert options_of(parse_args([*required, *training_flags(defaults)])) == defaults
