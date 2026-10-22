"""The cellmap model type reads at the shape its TorchScript export runs at.

cellmap_models traces the export at whichever shape the exporter passed, and
the metadata records both the training and the inference shapes without
saying which: jrc_mus-livers_16nm_to_8nm_mito runs only at its inference 288³
(it fails at 216³), mito_aff_unet_setup_16 only at its training 178³ (fed
378³ it still answers 56³). The type asks the model.
"""
from types import SimpleNamespace

import pytest
import torch

from cellmap_flow.models.configs.cellmap import CellMapModelConfig


class _TracedAt(torch.nn.Module):
    """A stand-in for a traced export: fed ``inp`` voxels it answers ``out``
    voxels; fed anything else it fails, as a traced UNet does."""

    def __init__(self, inp, out):
        super().__init__()
        self.inp, self.out = inp, out

    def forward(self, x):
        if tuple(x.shape[-3:]) != (self.inp,) * 3:
            raise RuntimeError("Kernel size can't be greater than actual input size")
        return x[..., : self.out, : self.out, : self.out]


class _Fixed(torch.nn.Module):
    """A stand-in for an export whose crops are baked in: ``out`` voxels
    whatever the input, as setup_16 answers 56³ to a 378³ input."""

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


INFERENCE = {"inference_input_shape": [288] * 3, "inference_output_shape": [212] * 3}


@pytest.mark.parametrize(
    "model, shapes, voxels_in, voxels_out",
    [
        (_TracedAt(288, 212), INFERENCE, 288, 212),
        (_Fixed(68), INFERENCE, 216, 68),
        (_Fixed(68), {}, 216, 68),
    ],
    ids=["traced-at-inference", "traced-at-training", "training-shapes-only"],
)
def test_reads_at_the_shape_the_export_runs_at(monkeypatch, model, shapes, voxels_in, voxels_out):
    cellmap_model = pytest.importorskip("cellmap_models.model_export.cellmap_model")
    monkeypatch.setattr(
        cellmap_model,
        "CellmapModel",
        lambda folder_path: SimpleNamespace(folder_path=folder_path, metadata=_metadata(**shapes), ts_model=model),
    )

    model_config = CellMapModelConfig(folder_path="/models/liver_mito")

    assert model_config.geometry.input_shape == (voxels_in,) * 3
    assert model_config.geometry.output_shape == (voxels_out,) * 3
    assert list(model_config.config.block_shape) == [voxels_out] * 3 + [1]
