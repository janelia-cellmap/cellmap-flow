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
    """Shaped like the real thing: a body conv, then a 1x1x1 affinity head.

    The head matters. peft merges a 1x1x1 *3D* conv through its conv2d
    shortcut, which fails outright -- so a test net of 3x3x3 convs alone
    never sees the path the real model takes.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 2, 3, padding=1)
        self.final_conv = nn.Conv3d(2, 3, 1)

    def forward(self, x):
        return self.final_conv(self.conv(x))


TARGETS = ["conv", "final_conv"]


def _copy_base_weights(peft_model, base):
    """Point a PeftModel's frozen base layers at a known set of weights."""
    inner = peft_model.base_model.model
    inner.conv.base_layer.load_state_dict(base.conv.state_dict())
    inner.final_conv.base_layer.load_state_dict(base.final_conv.state_dict())


@pytest.fixture
def already_adapted():
    """A base model, and that model with a trained adapter already on it."""
    from peft import LoraConfig, PeftModel, get_peft_model

    torch.manual_seed(0)
    x = torch.randn(1, 1, 4, 4, 4)
    base = _Net()
    base_out = base(x).detach().clone()

    first = get_peft_model(
        _Net(), LoraConfig(r=64, lora_alpha=128, target_modules=TARGETS, bias="none")
    )
    # lora_B is zero-initialised, so an untouched adapter is a no-op. Give it
    # a real effect, the way a finished training run would have.
    for name, param in first.named_parameters():
        if "lora_B" in name:
            nn.init.normal_(param, std=0.5)
    _copy_base_weights(first, base)
    adapted_out = first(x).detach().clone()
    assert not torch.allclose(adapted_out, base_out), "fixture adapter does nothing"

    directory = tempfile.mkdtemp()
    first.save_pretrained(directory)

    reloaded = PeftModel.from_pretrained(_Net().eval(), directory, is_trainable=False)
    _copy_base_weights(reloaded, base)
    assert torch.allclose(reloaded(x), adapted_out, atol=1e-5)

    return x, base_out, adapted_out, reloaded


def test_the_existing_adapter_survives_being_wrapped_again(already_adapted):
    x, base_out, adapted_out, reloaded = already_adapted

    wrapped = wrap_model_with_lora(
        reloaded, target_modules=TARGETS, lora_r=8, lora_alpha=16
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
        reloaded, target_modules=TARGETS, lora_r=8, lora_alpha=16
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
        plain, target_modules=TARGETS, lora_r=8, lora_alpha=16
    )
    assert torch.allclose(wrapped(x).detach(), expected, atol=1e-5)


def test_the_new_adapter_is_the_only_trainable_thing(already_adapted):
    """The merged-in adapter is frozen base now; only the new one trains."""
    _, _, _, reloaded = already_adapted
    wrapped = wrap_model_with_lora(
        reloaded, target_modules=TARGETS, lora_r=8, lora_alpha=16
    )
    trainable = [n for n, p in wrapped.named_parameters() if p.requires_grad]
    assert trainable, "nothing to train"
    assert all("lora_" in n for n in trainable), trainable


def test_train_save_serve_roundtrip_reproduces_the_trained_model():
    """The served model must be the model that was trained.

    Inference loads the saved adapter onto the same script model training
    started from -- which is itself a PeftModel. Re-wrapping it double-nested
    every module name ("base_model.model.base_model.model...."), so none of
    the saved keys matched; PEFT warned about missing adapter keys and handed
    back a model with no adapter at all. Combined with the existing adapter
    being clobbered, the served "finetuned" model was the untouched base.

    Merging on both sides keeps one module tree, so the keys line up.
    """
    from peft import LoraConfig, PeftModel, get_peft_model

    from cellmap_flow.finetune.lora_wrapper import load_lora_adapter

    torch.manual_seed(0)
    x = torch.randn(1, 1, 4, 4, 4)
    base = _Net()
    base_out = base(x).detach().clone()

    first = get_peft_model(
        _Net(), LoraConfig(r=64, lora_alpha=128, target_modules=TARGETS, bias="none")
    )
    for name, param in first.named_parameters():
        if "lora_B" in name:
            nn.init.normal_(param, std=0.5)
    _copy_base_weights(first, base)
    first_dir = tempfile.mkdtemp()
    first.save_pretrained(first_dir)

    def script_model():
        m = PeftModel.from_pretrained(_Net().eval(), first_dir, is_trainable=False)
        _copy_base_weights(m, base)
        return m

    trained = wrap_model_with_lora(
        script_model(), target_modules=TARGETS, lora_r=8, lora_alpha=16
    )
    for name, param in trained.named_parameters():
        if "lora_B" in name:
            nn.init.normal_(param, std=0.3)  # stand in for a training run
    # eval(), or lora_dropout makes both sides stochastic and nothing matches.
    trained.eval()
    trained_out = trained(x).detach().clone()
    adapter_dir = tempfile.mkdtemp()
    trained.save_pretrained(adapter_dir)

    served = load_lora_adapter(script_model(), adapter_dir, is_trainable=False)
    served.eval()
    served_out = served(x).detach().clone()

    assert torch.allclose(served_out, trained_out, atol=1e-5), (
        "the served model is not the model that was trained"
    )
    assert not torch.allclose(served_out, base_out, atol=1e-5), (
        "served the untouched base -- this is the original bug"
    )


def test_a_1x1x1_affinity_head_can_be_merged_at_all():
    """The shape peft cannot merge by itself.

    A real run died here. The head is a 1x1x1 Conv3d with 3 outputs and the
    adapter had r=64, so peft's conv2d shortcut left a trailing spatial axis,
    the matmul became a batched one, and it reported "size of tensor a (3)
    must match the size of tensor b (64)". Merging is how the existing
    adapter is kept, so a failure to merge is a failure to finetune at all.
    """
    from peft import LoraConfig, get_peft_model

    from cellmap_flow.finetune.lora_wrapper import _merge_existing_adapters

    torch.manual_seed(0)
    x = torch.randn(1, 1, 4, 4, 4)
    adapted = get_peft_model(
        _Net(), LoraConfig(r=64, lora_alpha=128, target_modules=TARGETS, bias="none")
    )
    for name, param in adapted.named_parameters():
        if "lora_B" in name:
            nn.init.normal_(param, std=0.5)
    adapted.eval()
    expected = adapted(x).detach().clone()

    merged = _merge_existing_adapters(adapted).eval()

    # Folding the adapter into the weights must not move the outputs, for the
    # 3x3x3 conv peft handles and the 1x1x1 head it does not.
    assert torch.allclose(merged(x).detach(), expected, atol=1e-5)
    assert not [n for n, _ in merged.named_parameters() if "lora_" in n]
