"""Fixtures for the finetune tests.

- ``ome_zarr``: an OME-NGFF group with the given levels.
- ``annotation_volume``: an annotation volume over a raw zarr, and the
  VirtualPatchDataset that reads it.
- ``make_trainer``: a LoRAFinetuner on the CPU, in fp32, without TensorBoard.
- ``tiny_script``: a script model that the trainer and the CLI can load.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from torch.utils.data import DataLoader, TensorDataset


def _triple(value):
    return [float(v) for v in value] if isinstance(value, (list, tuple)) else [float(value)] * 3


def _v3_array(path, data, chunks=None, attributes=None):
    """A zarr v3 array (written with tensorstore; zarr-python 2 cannot)."""
    import tensorstore as ts

    metadata = {"shape": list(data.shape), "data_type": str(data.dtype), "attributes": attributes or {},
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(chunks or data.shape)}}}
    ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(path)}, "create": True,
             "metadata": metadata}).result()[:] = data
    return str(path)


@pytest.fixture
def ome_zarr():
    """``ome_zarr(path, (name, data, scale, translation), ...)``: a group with OME
    multiscales, zarr v2 (0.4) or, with ``zarr_format=3``, v3 (0.5). ``ome_zarr.v3_array``
    writes a plain v3 array."""

    def write(path, *levels, chunks=None, zarr_format=2):
        multiscales = [{
            "version": "0.4" if zarr_format == 2 else "0.5",
            "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
            "datasets": [{"path": name, "coordinateTransformations": [
                {"type": "scale", "scale": _triple(scale)},
                {"type": "translation", "translation": _triple(translation)},
            ]} for name, _, scale, translation in levels],
        }]
        if zarr_format == 2:
            group = zarr.open_group(str(path), mode="w")
            for name, data, _, _ in levels:
                group.create_dataset(name, data=data, chunks=chunks or data.shape)
            group.attrs["multiscales"] = multiscales
            return str(path)
        path.mkdir(parents=True)
        (path / "zarr.json").write_text(json.dumps(
            {"zarr_format": 3, "node_type": "group", "attributes": {"ome": {"multiscales": multiscales}}}
        ))
        for name, data, _, _ in levels:
            _v3_array(path / name, data, chunks)
        return str(path)

    write.v3_array = _v3_array
    return write


@pytest.fixture
def annotation_volume(tmp_path, ome_zarr):
    """``annotation_volume(labels, raw=None, crops=())``: a 16 nm volume holding
    ``labels`` over a raw of the same grid (constant 128 unless given), and
    ``.dataset(**kw)``, the VirtualPatchDataset over them (8^3 in, 4^3 out)."""
    from cellmap_flow.finetune.virtual_dataset import VirtualPatchDataset

    def make(labels, raw=None, crops=(), name="vol"):
        raw = np.full(labels.shape, 128, np.uint8) if raw is None else raw
        raw_path = ome_zarr(tmp_path / f"{name}_raw.zarr", ("s0", raw, 16.0, 0.0), chunks=(16, 16, 16))
        path = str(tmp_path / f"{name}.zarr")
        root = zarr.open_group(path, mode="w")
        root.create_group("annotation").create_dataset(
            "s0", shape=labels.shape, chunks=(16, 16, 16), dtype="uint8", fill_value=0
        )
        root["annotation"]["s0"][:] = labels
        root.attrs.update(dataset_offset_nm=[0.0] * 3, imported_crops=list(crops))
        root["annotation"].attrs["multiscales"] = zarr.open_group(raw_path).attrs["multiscales"]

        def dataset(**kw):
            args = dict(volume_zarr_path=path, raw_dataset_path=raw_path, input_size_voxels=(8, 8, 8),
                        output_size_voxels=(4, 4, 4), input_voxel_size_nm=(16, 16, 16),
                        output_voxel_size_nm=(16, 16, 16), seed=0)
            return VirtualPatchDataset(**{**args, **kw})

        return SimpleNamespace(path=path, raw=raw_path, dataset=dataset)

    return make


@pytest.fixture
def make_trainer(tmp_path):
    """``make_trainer(model, data, **kw)``: ``data`` is a DataLoader, or (raw, annotation[, anchor]) tensors."""
    from cellmap_flow.finetune.lora_trainer import LoRAFinetuner

    def make(model, data, batch_size=1, **kw):
        if not isinstance(data, DataLoader):
            data = DataLoader(TensorDataset(*data), batch_size=batch_size)
        args = dict(output_dir=str(tmp_path / "run"), num_epochs=1, device="cpu",
                    use_mixed_precision=False, loss_type="bce", tensorboard=False)
        return LoRAFinetuner(model, data, **{**args, **kw})

    return make


SCRIPT = """
from funlib.geometry import Coordinate
import torch
import torch.nn as nn

input_voxel_size = output_voxel_size = Coordinate(8, 8, 8)
read_shape = write_shape = Coordinate(4, 4, 4) * input_voxel_size
output_channels = {channels}
torch.manual_seed(0)
model = nn.Sequential(nn.Conv3d(1, 4, 1), nn.ReLU(), nn.Conv3d(4, {channels}, 1))
"""


@pytest.fixture
def tiny_script(tmp_path):
    """``tiny_script(channels=1)``: the path of a script model, 4^3 in and out at 8 nm."""

    def write(channels=1):
        path = tmp_path / f"model_{channels}.py"
        path.write_text(SCRIPT.format(channels=channels))
        return path

    return write
