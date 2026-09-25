from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cellmap_flow.models.models_config import ModelConfig


class ForwardRaises(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, _x):
        raise AssertionError("dummy forward should not run")


class DummyModelConfig(ModelConfig):
    def _get_config(self):
        return SimpleNamespace(
            model=ForwardRaises(),
            read_shape=(8, 8, 8),
            write_shape=(8, 8, 8),
            input_voxel_size=(1, 1, 1),
            output_voxel_size=(1, 1, 1),
            output_channels=1,
            block_shape=np.array((8, 8, 8, 1)),
        )


def test_model_config_can_skip_dummy_forward_validation():
    model_config = DummyModelConfig()
    model_config.validate_model_shapes = False

    assert model_config.config.read_shape == (8, 8, 8)


def test_model_config_runs_dummy_forward_validation_by_default():
    with pytest.raises(AssertionError, match="dummy forward should not run"):
        _ = DummyModelConfig().config
