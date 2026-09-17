"""Finetuning a model that already has a LoRA adapter must keep that adapter.

get_peft_model() on something that is already a PeftModel does not stack.
Both adapters are named "default", so the second injection replaces the first
and its weights are gone. PEFT only warns. The effect: every "continue from
my last run" silently trained from the untuned base model instead, and the
result was compared against the adapter it had just discarded.

The only visible trace in a real log was the total parameter count going
*down* after wrapping: 818,314,067 -> 795,644,483.
"""

import tempfile

import pytest
import torch
import torch.nn as nn

from cellmap_flow.finetune.lora_wrapper import wrap_model_with_lora

peft = pytest.importorskip("peft")


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 2, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


@pytest.fixture
def already_adapted():
    """A base model, and that model with a trained adapter already on it."""
    from peft import LoraConfig, PeftModel, get_peft_model

    torch.manual_seed(0)
    x = torch.randn(1, 1, 4, 4, 4)
    base = _Net()
    base_out = base(x).detach().clone()

    first = get_peft_model(
        _Net(), LoraConfig(r=64, lora_alpha=128, target_modules=["conv"], bias="none")
    )
    # lora_B is zero-initialised, so an untouched adapter is a no-op. Give it
    # a real effect, the way a finished training run would have.
    for name, param in first.named_parameters():
        if "lora_B" in name:
            nn.init.normal_(param, std=0.5)
    first.base_model.model.conv.base_layer.load_state_dict(base.conv.state_dict())
    adapted_out = first(x).detach().clone()
    assert not torch.allclose(adapted_out, base_out), "fixture adapter does nothing"

    directory = tempfile.mkdtemp()
    first.save_pretrained(directory)

    reloaded = PeftModel.from_pretrained(_Net().eval(), directory, is_trainable=False)
    reloaded.base_model.model.conv.base_layer.load_state_dict(base.conv.state_dict())
    assert torch.allclose(reloaded(x), adapted_out, atol=1e-5)

    return x, base_out, adapted_out, reloaded


def test_the_existing_adapter_survives_being_wrapped_again(already_adapted):
    x, base_out, adapted_out, reloaded = already_adapted

    wrapped = wrap_model_with_lora(
        reloaded, target_modules=["conv"], lora_r=8, lora_alpha=16
    )
    out = wrapped(x).detach()

    # A fresh LoRA has lora_B=0, so wrapping alone must not change anything.
    assert torch.allclose(out, adapted_out, atol=1e-5), (
        "training must start from the model you were looking at"
    )
    assert not torch.allclose(out, base_out, atol=1e-5), (
        "the existing adapter was discarded -- this is the original bug"
    )


def test_the_distillation_teacher_is_the_adapted_model(already_adapted):
    """Good regions anchor the student to the teacher, so the teacher has to
    be the model the user judged good -- not the untuned original."""
    x, base_out, adapted_out, reloaded = already_adapted

    wrapped = wrap_model_with_lora(
        reloaded, target_modules=["conv"], lora_r=8, lora_alpha=16
    )
    with torch.no_grad():
        wrapped.disable_adapter_layers()
        teacher = wrapped(x).detach().clone()
        wrapped.enable_adapter_layers()

    assert torch.allclose(teacher, adapted_out, atol=1e-5)
    assert not torch.allclose(teacher, base_out, atol=1e-5)


def test_a_plain_model_is_unaffected():
    """No adapter to merge: the ordinary path must not change."""
    torch.manual_seed(0)
    x = torch.randn(1, 1, 4, 4, 4)
    plain = _Net()
    expected = plain(x).detach().clone()

    wrapped = wrap_model_with_lora(
        plain, target_modules=["conv"], lora_r=8, lora_alpha=16
    )
    assert torch.allclose(wrapped(x).detach(), expected, atol=1e-5)


def test_the_new_adapter_is_the_only_trainable_thing(already_adapted):
    """The merged-in adapter is frozen base now; only the new one trains."""
    _, _, _, reloaded = already_adapted
    wrapped = wrap_model_with_lora(
        reloaded, target_modules=["conv"], lora_r=8, lora_alpha=16
    )
    trainable = [n for n, p in wrapped.named_parameters() if p.requires_grad]
    assert trainable, "nothing to train"
    assert all("lora_" in n for n in trainable), trainable
