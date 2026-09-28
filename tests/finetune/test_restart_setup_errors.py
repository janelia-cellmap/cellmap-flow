"""A restart whose settings or data cannot be used does not end the job.

create_dataloader and _build_target_transform ran outside the CLI's try, so
a bad restart parameter or an emptied volume raised out of main() and ended
the whole job, taking the served model with it. And the model was reset for
the restart before any of that was known to work, so it was the base model
that went on being served.
"""

import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

from cellmap_flow.finetune import finetune_cli

SCRIPT = """
from funlib.geometry import Coordinate
import torch.nn as nn

input_voxel_size = Coordinate(8, 8, 8)
output_voxel_size = Coordinate(8, 8, 8)
read_shape = Coordinate(4, 4, 4) * input_voxel_size
write_shape = Coordinate(4, 4, 4) * output_voxel_size
output_channels = 1
model = nn.Conv3d(1, 1, 1)
"""


def test_a_restart_that_cannot_be_set_up_waits_for_the_next(tmp_path, monkeypatch, capsys):
    script = tmp_path / "model.py"
    script.write_text(SCRIPT)
    corrections = tmp_path / "corrections"
    corrections.mkdir()
    events = []
    loader = DataLoader(TensorDataset(torch.rand(2, 1, 4, 4, 4), torch.full((2, 1, 4, 4, 4), 2.0)), batch_size=2)
    loads = iter([loader, ValueError("no populated chunks"), loader])

    def fake_dataloader(*a, **k):
        item = next(loads)
        if isinstance(item, Exception):
            events.append("setup failed")
            raise item
        events.append("setup")
        return item

    signals = iter([{"params": {}}, {"params": {"learning_rate": 1e-3}}, None])

    def fake_wait(**kwargs):
        events.append("wait")
        return next(signals)

    real_reset = finetune_cli._reset_for_restart

    def recording_reset(*a, **k):
        events.append("reset")
        return real_reset(*a, **k)

    monkeypatch.setattr(finetune_cli, "create_dataloader", fake_dataloader)
    monkeypatch.setattr(finetune_cli, "_wait_for_restart_signal", fake_wait)
    monkeypatch.setattr(finetune_cli, "_reset_for_restart", recording_reset)
    monkeypatch.setattr(finetune_cli, "_start_inference_server_background", lambda *a, **k: (None, 0))
    monkeypatch.setattr(sys, "argv", [
        "finetune_cli", "--model-type", "script", "--model-script", str(script),
        "--model-name", "tiny", "--corrections", str(corrections), "--output-dir", str(tmp_path / "run"),
        "--lora-r", "0", "--num-epochs", "1", "--loss-type", "bce", "--no-mixed-precision",
        "--no-tensorboard", "--num-workers", "0", "--auto-serve", "--serve-data-path", str(tmp_path),
    ])

    assert finetune_cli.main() == 1  # the last, malformed, signal ends it

    assert "RESTART_FAILED: no populated chunks" in capsys.readouterr().out
    # The model is only reset once the next iteration is known to run.
    assert events == ["setup", "wait", "setup failed", "wait", "setup", "reset", "wait"]
