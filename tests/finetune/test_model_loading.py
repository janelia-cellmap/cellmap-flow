"""A finetuned model as the trainer loads it to train on again (model_loading):
continuing from the last run, from a full finetune's weights or a LoRA adapter."""

import json

import pytest
import torch

from cellmap_flow.finetune.model_loading import (
    decode_model_entry,
    load_trainable_model,
    model_config_from_entry,
    root_base_model_dict,
)
from cellmap_flow.models.models_config import FinetuneModelConfig, ScriptModelConfig
from cellmap_flow.serving.protocol import encode_to_str


def _trains_on(text, base, trained):
    """The model the trainer loads from the entry ``text`` computes what ``trained`` does."""
    config = model_config_from_entry(decode_model_entry(text))
    assert isinstance(config, FinetuneModelConfig) and root_base_model_dict(config) == base
    x = torch.rand(1, 1, 4, 4, 4)
    with torch.no_grad():
        assert torch.allclose(load_trainable_model(config).eval()(x), trained.eval()(x), atol=1e-6)


def test_a_full_finetune_entry_trains_on_from_its_weights(tiny_script, tmp_path):
    """The job manager passes a finetuned model to the trainer as its entry,
    which the trainer did not take: continuing from the last run was out of
    the dashboard's reach."""
    base = {"type": "script", "script_path": str(tiny_script())}
    trained = ScriptModelConfig(script_path=base["script_path"]).config.model
    with torch.no_grad():
        for p in trained.parameters():
            p.add_(0.5)
    torch.save(trained.state_dict(), tmp_path / "model_state_dict.pt")
    # As the job manager encodes it; to_dict() also surfaces base fields for
    # display, and they must not break loading.
    entry = {"type": "finetune", "name": "ft", "base_model": base, "lora_adapter_path": None,
             "weights_path": str(tmp_path / "model_state_dict.pt"), "base_type": "script", "channels": ["mito"]}
    _trains_on(encode_to_str(entry), base, trained)


@pytest.mark.finetune
def test_a_lora_finetune_entry_trains_on_from_its_adapter(tiny_script, tmp_path):
    from cellmap_flow.finetune.lora_wrapper import save_lora_adapter, wrap_model_with_lora

    base = {"type": "script", "script_path": str(tiny_script())}
    trained = wrap_model_with_lora(ScriptModelConfig(script_path=base["script_path"]).config.model,
                                   lora_r=2, lora_alpha=4, lora_dropout=0.0)
    with torch.no_grad():
        for name, p in trained.named_parameters():
            if "lora_B" in name:
                p.fill_(0.3)
    save_lora_adapter(trained, str(tmp_path / "adapter"))
    # Plain JSON works too, for a command typed by hand.
    entry = {"type": "finetune", "name": "ft", "lora_adapter_path": str(tmp_path / "adapter"), "base_model": base}
    _trains_on(json.dumps(entry), base, trained)
