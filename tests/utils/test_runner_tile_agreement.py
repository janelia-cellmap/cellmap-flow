"""ModelRunner's tile check: a tile the network answers differently at than
at its reference tile is refused at warmup.

On a B300 the setup64 ER model ran at a 570-voxel tile (16 feature maps of
568^3 is past 2^31 elements) and answered 1 everywhere, 0.03 correlated with
itself at 514, and right at 458 and below; the warmup's shape check passed
and a blockwise task wrote a solid volume. The GPU is not needed here: the
"network" is a crop that answers a constant above a size.
"""

import numpy as np
import pytest
import torch

from cellmap_flow.inference import runner
from cellmap_flow.inference.runner import MAX_DISAGREEMENT, ModelRunner, disagreement, smooth_field


class _Crop(torch.nn.Module):
    """A valid 3^3 "convolution": the input less one voxel each side, so
    bigger tiles answer what the smaller tile answers over shared voxels.
    From ``breaks_at`` voxels a side on it answers 1 everywhere."""

    def __init__(self, breaks_at=None):
        super().__init__()
        self.breaks_at = breaks_at
        self.weight = torch.nn.Parameter(torch.ones(1))  # so the runner finds a device

    def forward(self, x):
        out = x[:, :, 1:-1, 1:-1, 1:-1] * self.weight
        if self.breaks_at is not None and x.shape[-1] >= self.breaks_at:
            return torch.ones_like(out)
        return out


def _runner(model, reference=(16, 16, 16)):
    """A ModelRunner with the state _check_tile_agreement reads, without a model build."""
    this = ModelRunner.__new__(ModelRunner)
    this.device = torch.device("cpu")
    this.use_half_prediction = False
    this.model_config = type("Config", (), {})()
    this.model_config.reference_input_size = reference
    this.model_config.config = type("Inner", (), {})()
    this.model_config.config.model = model
    this.model_config.config.input_voxel_size = (8.0, 8.0, 8.0)
    return this


def test_smooth_field_is_a_function_of_position_alone_and_varies():
    coarse = smooth_field([8.0 * (np.arange(40) + 0.5)] * 3)
    centred = smooth_field([8.0 * (10 + np.arange(20) + 0.5)] * 3)
    assert coarse.shape == (40, 40, 40) and coarse.dtype == np.float32
    assert np.allclose(coarse[10:30, 10:30, 10:30], centred)
    assert coarse.min() >= -1 and coarse.max() <= 1 and coarse.std() > 0.05


def test_disagreement_compares_the_centre_of_the_bigger_output():
    reference = np.random.default_rng(0).random((2, 6, 6, 6))
    bigger = np.zeros((2, 10, 10, 10))
    bigger[:, 2:8, 2:8, 2:8] = reference
    assert disagreement(reference, bigger) == (pytest.approx(1.0), pytest.approx(0.0), False)
    # A constant where the reference varies is another answer whatever its
    # value: here 1 against a 0-1 reference, under 2 spreads away.
    correlation, difference, wrong = disagreement(reference, np.ones_like(bigger))
    assert correlation is None and difference < MAX_DISAGREEMENT and wrong
    # Shifted far enough, so is a varying output.
    correlation, difference, wrong = disagreement(reference, bigger + 1.0)
    assert correlation == pytest.approx(1.0) and difference > MAX_DISAGREEMENT and wrong
    assert disagreement(reference, np.full_like(bigger, np.nan)) == (None, float("inf"), True)
    assert disagreement(None, bigger) == (None, 0.0, False)
    assert disagreement(reference, bigger[:, :4]) == (None, 0.0, False)  # does not nest
    flat = np.zeros((2, 6, 6, 6))
    assert disagreement(flat, np.zeros_like(bigger)) == (None, 0.0, False)  # the same constant
    assert disagreement(flat, np.ones_like(bigger))[2]  # another constant: 1000 spreads away


def test_a_tile_answering_as_the_reference_does_passes(caplog):
    this = _runner(_Crop())
    with caplog.at_level("INFO", logger=runner.__name__):
        this._check_tile_agreement((40, 40, 40))
    assert "Tile check: [40, 40, 40] answers as [16, 16, 16] does (correlation 1.000)" in caplog.text


def test_a_tile_the_network_answers_a_constant_at_is_refused():
    this = _runner(_Crop(breaks_at=40))
    this._check_tile_agreement((32, 32, 32))  # still right there
    with pytest.raises(ValueError, match=r"input_size \[40, 40, 40\]: the network answers differently there than at "
                                         r"\[16, 16, 16\].*Use a smaller input_size"):
        this._check_tile_agreement((40, 40, 40))


def test_the_check_is_skipped_without_a_reference_tile_or_at_it():
    this = _runner(_Crop(breaks_at=10), reference=None)
    this._check_tile_agreement((40, 40, 40))  # nothing to hold it against
    this = _runner(_Crop(breaks_at=10))
    this._check_tile_agreement((16, 16, 16))  # the reference itself
    this._check_tile_agreement((12, 12, 12))  # smaller than the reference: not nested, not checked
