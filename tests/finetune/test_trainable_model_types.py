"""The trainer takes every model type the dashboard can submit.

The job manager passes the model config's cli_name as --model-type, which can
be "cellmap" (what export_merged produces) or "finetune" (a finetuned model
registered back into the dashboard). argparse accepted only
fly/dacapo/huggingface/script, so those jobs died on the GPU node with exit
code 2 -- and the "continue from my last run" adapter merge in lora_wrapper
could not be reached from the dashboard at all.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from cellmap_flow.finetune.finetune_cli import _model_config_from_args, build_arg_parser
from cellmap_flow.finetune.model_loading import (
    decode_model_entry,
    load_trainable_model,
    root_base_model_dict,
)

SCRIPT = """
from funlib.geometry import Coordinate
import torch
import torch.nn as nn

input_voxel_size = Coordinate(8, 8, 8)
output_voxel_size = Coordinate(8, 8, 8)
read_shape = Coordinate(4, 4, 4) * input_voxel_size
write_shape = Coordinate(4, 4, 4) * output_voxel_size
output_channels = 1
torch.manual_seed(0)
model = nn.Sequential(nn.Conv3d(1, 4, 1), nn.ReLU(), nn.Conv3d(4, 1, 1))
"""


@pytest.fixture
def script(tmp_path):
    path = tmp_path / "model.py"
    path.write_text(SCRIPT)
    return path


def _args(*extra):
    return build_arg_parser().parse_args(
        ["--corrections", "/c", "--output-dir", "/o", *extra]
    )


@pytest.mark.parametrize("model_type", ["cellmap", "finetune"])
def test_the_parser_accepts_the_types_the_dashboard_sends(model_type):
    assert _args("--model-type", model_type).model_type == model_type


def test_a_full_finetune_entry_trains_from_its_finetuned_weights(tmp_path, script):
    from cellmap_flow.utils.web_utils import encode_to_str

    base = {"type": "script", "script_path": str(script)}
    trained = torch.nn.Sequential(torch.nn.Conv3d(1, 4, 1), torch.nn.ReLU(), torch.nn.Conv3d(4, 1, 1))
    weights = tmp_path / "model_state_dict.pt"
    torch.save(trained.state_dict(), weights)
    entry = {"type": "finetune", "name": "ft", "base_model": base,
             "weights_path": str(weights), "lora_adapter_path": None,
             # to_dict() surfaces base fields for display; they must not break loading
             "base_type": "script", "channels": ["mito"]}

    args = _args("--model-type", "finetune", "--model-entry", encode_to_str(entry))
    model_config = _model_config_from_args(args)
    assert type(model_config).__name__ == "FinetuneModelConfig"

    model = load_trainable_model(model_config)
    for k, v in trained.state_dict().items():
        assert torch.equal(model.state_dict()[k], v)
    assert root_base_model_dict(model_config) == base


def test_a_model_entry_may_be_plain_json():
    assert decode_model_entry('{"type": "cellmap", "folder_path": "/m"}') == {
        "type": "cellmap", "folder_path": "/m"
    }


@pytest.mark.finetune
def test_a_lora_finetune_entry_continues_from_its_adapter(tmp_path, script):
    from cellmap_flow.finetune.lora_wrapper import save_lora_adapter, wrap_model_with_lora
    from cellmap_flow.models.models_config import ScriptModelConfig

    base_model = ScriptModelConfig(script_path=str(script)).config.model
    adapted = wrap_model_with_lora(base_model, lora_r=2, lora_alpha=4, lora_dropout=0.0)
    with torch.no_grad():
        for name, p in adapted.named_parameters():
            if "lora_B" in name:
                p.fill_(0.3)
    save_lora_adapter(adapted, str(tmp_path / "adapter"))
    x = torch.rand(1, 1, 4, 4, 4)
    with torch.no_grad():
        expected = adapted.eval()(x)

    entry = {"type": "finetune", "name": "ft", "lora_adapter_path": str(tmp_path / "adapter"),
             "base_model": {"type": "script", "script_path": str(script)}}
    model_config = _model_config_from_args(
        _args("--model-type", "finetune", "--model-entry", json.dumps(entry))
    )
    rewrapped = wrap_model_with_lora(
        load_trainable_model(model_config), lora_r=2, lora_alpha=4, lora_dropout=0.0
    )
    with torch.no_grad():
        got = rewrapped.eval()(x)
    # A fresh adapter starts at zero, so the new model is the finetuned one.
    assert torch.allclose(got, expected, atol=1e-6)


class _CellMapConfig:
    cli_name = "cellmap"
    name = "exported"

    def to_dict(self):
        return {"type": "cellmap", "folder_path": "/models/exported", "name": "exported"}


class _BioConfig:
    cli_name = "bioimage"
    name = "bio"


def _submit(manager, model_config, tmp_path):
    corrections = tmp_path / "corrections"
    (corrections / "vol.zarr").mkdir(parents=True, exist_ok=True)
    (corrections / "vol.zarr" / ".zattrs").write_text(json.dumps({"dataset_path": "/data/raw.zarr"}))
    (corrections / "_virtual_sources.json").write_text(json.dumps({"kind": "volume_zarr_v1"}))

    class _Thread:
        def __init__(self, *a, **k):
            pass

        def start(self):
            pass

    with patch("cellmap_flow.finetune.finetune_job_manager.is_bsub_available", return_value=False), \
         patch("cellmap_flow.finetune.finetune_job_manager.run_locally",
               return_value=SimpleNamespace(process=SimpleNamespace(pid=1))), \
         patch("cellmap_flow.finetune.finetune_job_manager.threading.Thread", _Thread):
        return manager.submit_finetuning_job(
            model_config=model_config, corrections_path=corrections, output_base=tmp_path,
        )


def test_the_job_manager_passes_a_cellmap_model_as_its_entry(tmp_path):
    from cellmap_flow.finetune.finetune_job_manager import FinetuneJobManager

    job = _submit(FinetuneJobManager(), _CellMapConfig(), tmp_path)
    command = json.loads((Path(job.output_dir) / "metadata.json").read_text())["command"]
    tokens = command.split()
    assert tokens[tokens.index("--model-type") + 1] == "cellmap"
    entry = decode_model_entry(tokens[tokens.index("--model-entry") + 1])
    assert entry == _CellMapConfig().to_dict()


def test_an_untrainable_type_is_refused_before_anything_is_submitted(tmp_path):
    from cellmap_flow.finetune.finetune_job_manager import FinetuneJobManager

    with pytest.raises(ValueError, match="cannot be finetuned"):
        _submit(FinetuneJobManager(), _BioConfig(), tmp_path)
    assert not (tmp_path / "runs").exists()
