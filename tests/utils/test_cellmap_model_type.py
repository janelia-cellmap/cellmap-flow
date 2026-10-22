"""The cellmap model type reads at the shape its TorchScript export runs at.

cellmap_models traces the export at the metadata's inference shape, and a
traced UNet runs only there: jrc_mus-livers_16nm_to_8nm_mito fails at its
training 216³ and runs at its inference 288³.
"""
from types import SimpleNamespace

import pytest
import torch

from cellmap_flow.models.configs.cellmap import CellMapModelConfig


class _Shrink(torch.nn.Module):
    """A stand-in network: its output is the first ``out`` voxels along each axis."""

    def __init__(self, out):
        super().__init__()
        self.out = out

    def forward(self, x):
        return x[..., : self.out, : self.out, : self.out]


def _metadata(**shapes):
    fields = dict(
        model_name="m", model_type="CNNectomeUNet", framework="dacapo/torch", spatial_dims=3,
        in_channels=1, out_channels=1, iteration=1,
        input_voxel_size=[16, 16, 16], output_voxel_size=[8, 8, 8], channels_names=["mito"],
        input_shape=[216, 216, 216], output_shape=[68, 68, 68],
        inference_input_shape=None, inference_output_shape=None,
    )
    fields.update(shapes)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize(
    "shapes, voxels_in, voxels_out",
    [
        ({"inference_input_shape": [288] * 3, "inference_output_shape": [212] * 3}, 288, 212),
        ({}, 216, 68),
    ],
    ids=["inference-shapes", "training-shapes-only"],
)
def test_reads_at_the_exports_inference_shape(monkeypatch, shapes, voxels_in, voxels_out):
    cellmap_model = pytest.importorskip("cellmap_models.model_export.cellmap_model")
    monkeypatch.setattr(
        cellmap_model,
        "CellmapModel",
        lambda folder_path: SimpleNamespace(
            folder_path=folder_path, metadata=_metadata(**shapes), ts_model=_Shrink(voxels_out)
        ),
    )

    model_config = CellMapModelConfig(folder_path="/models/liver_mito")

    assert model_config.geometry.input_shape == (voxels_in,) * 3
    assert model_config.geometry.output_shape == (voxels_out,) * 3
    assert list(model_config.config.block_shape) == [voxels_out] * 3 + [1]
