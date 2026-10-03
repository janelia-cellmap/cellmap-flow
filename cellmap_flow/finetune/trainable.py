"""Building blocks for handing the finetune trainer a model's network.

The trainer trains a torch module that takes the normalized patch
(B, C, Z, Y, X) and gives the model's output (B, C', Z', Y', X'). Many
networks are not quite that: a 2D network wants (N, C, Y, X), a model
normalizes its own input, crops a halo off its output or ends in a
sigmoid its serving applies. A model type's ``trainable_model()`` wraps
its network in these, composed with ``torch.nn.Sequential``, rather than
writing its own wrapper; they hold no parameters of their own, so what is
trained is the network's.

``finetune_modes`` says what a module can be finetuned with: LoRA attaches
adapters beside a network's Conv and Linear layers, which needs a Python
module tree, while a full finetune trains every parameter, which a
compiled TorchScript module has too.
"""

from typing import Optional, Sequence, Tuple

import torch
from torch import nn

LORA, FULL = "lora", "full"


def finetune_modes(module: nn.Module) -> Tuple[str, ...]:
    """What ``module`` can be finetuned with: ("lora", "full"), ("full",) or ().

    A TorchScript module trains in full but takes no adapters (its layers
    are compiled, so PEFT cannot wrap them); a module without parameters
    cannot be trained at all; any other with Conv or Linear layers takes
    both.
    """
    if not any(p.requires_grad or p.is_floating_point() for p in module.parameters()):
        return ()
    if isinstance(module, torch.jit.ScriptModule):
        return (FULL,)
    from cellmap_flow.finetune.lora_wrapper import detect_adaptable_layers

    return (LORA, FULL) if detect_adaptable_layers(module) else (FULL,)


class SliceWise(nn.Module):
    """Run a 2D network on each z slice of (B, C, Z, Y, X).

    The slices go through ``net`` as one batch of B * Z images and come back
    as (B, C', Z, Y', X'): a 2D model served slice by slice is trained the
    way it is served.
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        b, c, z, y, xx = x.shape
        out = self.net(x.permute(0, 2, 1, 3, 4).reshape(b * z, c, y, xx))
        if isinstance(out, (tuple, list)):  # a network that also returns its features
            out = out[0]
        _, c_out, y_out, x_out = out.shape
        return out.reshape(b, z, c_out, y_out, x_out).permute(0, 2, 1, 3, 4)


class Crop(nn.Module):
    """Cut ``crop`` voxels off each side of the spatial axes (the last
    ``len(crop)``): a model's halo, which its serving cuts off too."""

    def __init__(self, crop: Sequence[int]):
        super().__init__()
        self.crop = tuple(int(c) for c in crop)

    def forward(self, x):
        if not any(self.crop):
            return x
        index = [slice(None)] * (x.ndim - len(self.crop))
        index += [slice(c, x.shape[x.ndim - len(self.crop) + i] - c) for i, c in enumerate(self.crop)]
        return x[tuple(index)]


def _dims(x: torch.Tensor, dims: Optional[Sequence[int]]) -> Tuple[int, ...]:
    """The dimensions statistics are taken over: ``dims``, else all but the batch."""
    return tuple(dims) if dims is not None else tuple(range(1, x.ndim))


class ZeroMeanUnitVariance(nn.Module):
    """(x - mean) / (std + eps), mean and std over ``dims`` of each sample
    (all but the batch by default): bioimage.io's ``zero_mean_unit_variance``."""

    def __init__(self, dims: Optional[Sequence[int]] = None, eps: float = 1e-6):
        super().__init__()
        self.dims, self.eps = dims, eps

    def forward(self, x):
        dims = _dims(x, self.dims)
        mean = x.mean(dim=dims, keepdim=True)
        std = x.std(dim=dims, keepdim=True, unbiased=False)
        return (x - mean) / (std + self.eps)


class FixedZeroMeanUnitVariance(nn.Module):
    """(x - mean) / (std + eps) with a given mean and std (one number, or one
    per channel): bioimage.io's ``fixed_zero_mean_unit_variance``."""

    def __init__(self, mean, std, eps: float = 1e-6, channel_dim: int = 1):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.eps, self.channel_dim = eps, channel_dim

    def _shaped(self, value, x):
        if value.ndim == 0:
            return value
        shape = [1] * x.ndim
        shape[self.channel_dim] = -1
        return value.reshape(shape)

    def forward(self, x):
        return (x - self._shaped(self.mean, x)) / (self._shaped(self.std, x) + self.eps)


class ScaleRange(nn.Module):
    """(x - lo) / (hi - lo + eps), lo and hi the ``min_percentile`` and
    ``max_percentile`` (0-100) over ``dims`` of each sample: bioimage.io's
    ``scale_range``, and Cellpose's 1-99% normalization."""

    def __init__(self, min_percentile: float = 0.0, max_percentile: float = 100.0,
                 dims: Optional[Sequence[int]] = None, eps: float = 1e-6):
        super().__init__()
        self.lo, self.hi = min_percentile / 100.0, max_percentile / 100.0
        self.dims, self.eps = dims, eps

    def forward(self, x):
        dims = _dims(x, self.dims)
        kept = [d for d in range(x.ndim) if d not in dims]
        # Quantiles over the flattened statistic dims, one per kept index.
        flat = x.permute(*kept, *dims).reshape(*[x.shape[d] for d in kept], -1).float()
        lo = torch.quantile(flat, self.lo, dim=-1)
        hi = torch.quantile(flat, self.hi, dim=-1)
        shape = [x.shape[d] if d in kept else 1 for d in range(x.ndim)]
        lo, hi = lo.reshape(shape), hi.reshape(shape)
        return (x - lo) / (hi - lo + self.eps)


class ScaleLinear(nn.Module):
    """x * gain + offset (one number, or one per channel): bioimage.io's ``scale_linear``."""

    def __init__(self, gain=1.0, offset=0.0, channel_dim: int = 1):
        super().__init__()
        self.register_buffer("gain", torch.as_tensor(gain, dtype=torch.float32))
        self.register_buffer("offset", torch.as_tensor(offset, dtype=torch.float32))
        self.channel_dim = channel_dim

    def forward(self, x):
        gain, offset = self.gain, self.offset
        if gain.ndim or offset.ndim:
            shape = [1] * x.ndim
            shape[self.channel_dim] = -1
            gain = gain.reshape(shape) if gain.ndim else gain
            offset = offset.reshape(shape) if offset.ndim else offset
        return x * gain + offset


class Clip(nn.Module):
    """x clamped to [min, max] (either may be None): bioimage.io's ``clip``."""

    def __init__(self, min: Optional[float] = None, max: Optional[float] = None):
        super().__init__()
        self.min, self.max = min, max

    def forward(self, x):
        return x.clamp(min=self.min, max=self.max)
