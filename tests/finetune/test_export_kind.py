"""The export kind follows the model, not args.lora_r.

A restart of a full-finetune job that carried lora_r > 0 left the model a
full finetune but set args.lora_r to 8, and the next iteration's YAML then
pointed at a lora_adapter/ that save_adapter() never writes (it writes
full_finetune/). Only the LoRA -> 0 direction was handled.
"""

import argparse
import json
from types import SimpleNamespace

import torch.nn as nn
import yaml

from cellmap_flow.finetune.finetune_cli import _generate_model_files, _reset_for_restart


def test_a_restart_cannot_turn_a_full_finetune_into_lora():
    model = nn.Conv3d(1, 1, 1)
    args = argparse.Namespace(lora_r=8, lora_alpha=16, lora_dropout=0.1, lora_min_channels=0)
    _reset_for_restart(model, args, None)
    assert args.lora_r == 0


def test_the_yaml_points_at_what_the_model_exported(tmp_path):
    corrections = tmp_path / "session" / "corrections"
    (corrections / "vol.zarr").mkdir(parents=True)
    (corrections / "vol.zarr" / ".zattrs").write_text(json.dumps({"dataset_path": "/data/raw.zarr"}))
    run = tmp_path / "session" / "runs" / "run"
    run.mkdir(parents=True)
    args = argparse.Namespace(
        lora_r=8, output_dir=str(run), corrections=str(corrections),
        serve_data_path=None, auto_serve=False,
    )
    model_config = SimpleNamespace(name="m", to_dict=lambda: {"type": "fly"})

    _, yaml_path = _generate_model_files(args, model_config, "20260101_000000", is_lora=False)

    entry = yaml.safe_load(open(yaml_path))["models"][0]
    assert entry["weights_path"].endswith("full_finetune/model_state_dict.pt")
    assert "lora_adapter_path" not in entry
