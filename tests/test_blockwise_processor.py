from types import SimpleNamespace

import numpy as np
import torch

from cellmap_flow.blockwise import blockwise_processor
from cellmap_flow.blockwise.blockwise_processor import CellMapFlowBlockwiseProcessor
from cellmap_flow.models.models_config import ModelConfig


class ForwardRaises(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, _x):
        raise AssertionError("dummy forward should not run in master")


class DummyModelConfig(ModelConfig):
    name = "dummy"

    def _get_config(self):
        return SimpleNamespace(
            model=ForwardRaises(),
            read_shape=(8, 8, 8),
            write_shape=(8, 8, 8),
            input_voxel_size=(1, 1, 1),
            output_voxel_size=(1, 1, 1),
            output_channels=1,
            channels=["prediction"],
            block_shape=np.array((8, 8, 8, 1)),
        )


class FakeImageDataInterface:
    def __init__(self, *_args, **_kwargs):
        self.shape = np.array((8, 8, 8))


def test_blockwise_master_skips_inferencer_and_dummy_forward(monkeypatch, tmp_path):
    model_config = DummyModelConfig()
    yaml_path = tmp_path / "task.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "data_path: /tmp/input.zarr/raw",
                f"output_path: {tmp_path / 'out.zarr'}",
                "task_name: test_task",
                "charge_group: cellmap",
                "queue: gpu_h100",
                "workers: 1",
                "models:",
                "  - name: dummy",
                "    type: script",
                "    script_path: /tmp/fake.py",
            ]
        )
    )

    monkeypatch.setattr(blockwise_processor, "build_models", lambda _models: [model_config])
    monkeypatch.setattr(blockwise_processor, "ImageDataInterface", FakeImageDataInterface)
    monkeypatch.setattr(blockwise_processor.zarr, "open", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        blockwise_processor,
        "Inferencer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Inferencer should not run in master")),
    )
    monkeypatch.setattr(
        blockwise_processor,
        "prepare_ds",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        blockwise_processor,
        "open_group",
        lambda *_args, **_kwargs: SimpleNamespace(attrs={}),
    )
    monkeypatch.setattr(
        blockwise_processor,
        "generate_singlescale_metadata",
        lambda **_kwargs: {"multiscales": []},
    )

    processor = CellMapFlowBlockwiseProcessor(str(yaml_path), create=True)

    assert model_config.validate_model_shapes is False
    assert processor.inferencers == []
    assert processor.inferencer is None
