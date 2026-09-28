"""Small datasets, a tiny script model and task YAMLs for blockwise tests."""

import textwrap

import numpy as np
import pytest
import yaml
import zarr

MODEL_SCRIPT = """
from funlib.geometry import Coordinate
import numpy as np
import torch

input_voxel_size = Coordinate({in_vs}, {in_vs}, {in_vs})
output_voxel_size = Coordinate({out_vs}, {out_vs}, {out_vs})
write_shape = Coordinate(4, 4, 4) * output_voxel_size
read_shape = write_shape
channels = ["a", "b"]
output_channels = 2
block_shape = np.array((4, 4, 4, 2))


class Model(torch.nn.Module):
    # Averages down to the output resolution and repeats it per channel.
    def forward(self, x):
        ratio = {out_vs} // {in_vs}
        if ratio > 1:
            x = torch.nn.functional.avg_pool3d(x, ratio)
        return x.repeat(1, 2, 1, 1, 1)


model = Model()
"""


@pytest.fixture
def model_script(tmp_path):
    """Write a script model; ``model_script(in_vs, out_vs)`` returns its path."""

    def make(in_vs=8, out_vs=8):
        path = tmp_path / f"model_{in_vs}_{out_vs}.py"
        path.write_text(textwrap.dedent(MODEL_SCRIPT.format(in_vs=in_vs, out_vs=out_vs)))
        return str(path)

    return make


@pytest.fixture
def raw_array(tmp_path):
    """A uint8 zarr array carrying funlib-style resolution/offset attributes."""

    def make(shape=(16, 16, 16), voxel_size=(8, 8, 8), offset=(0, 0, 0), name="raw"):
        container = zarr.open_group(str(tmp_path / "raw.zarr"), mode="a")
        data = (np.arange(np.prod(shape)) % 251).astype(np.uint8).reshape(shape)
        array = container.create_dataset(name, data=data, chunks=(8, 8, 8), overwrite=True)
        array.attrs["resolution"] = list(voxel_size)
        array.attrs["offset"] = list(offset)
        return str(tmp_path / "raw.zarr" / name)

    return make


@pytest.fixture
def multiscale_group(tmp_path):
    """An OME-Zarr 0.4 multiscale group with s0 at 8 nm and s1 at 16 nm."""
    path = tmp_path / "ms.zarr"
    group = zarr.open_group(str(path), mode="w")
    datasets = []
    for level, (name, vs) in enumerate((("s0", 8), ("s1", 16))):
        side = 16 >> level
        group.create_dataset(name, data=np.zeros((side,) * 3, np.uint8), chunks=(8, 8, 8))
        datasets.append(
            {
                "path": name,
                "coordinateTransformations": [
                    {"type": "scale", "scale": [vs] * 3},
                    {"type": "translation", "translation": [0, 0, 0]},
                ],
            }
        )
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
            "datasets": datasets,
        }
    ]
    return str(path)


@pytest.fixture
def task_yaml(tmp_path):
    """Write a blockwise task YAML; keyword arguments override the defaults."""

    def make(data_path, script_path, model_extra=None, **overrides):
        config = {
            "data_path": data_path,
            "output_path": str(tmp_path / "out.zarr"),
            "task_name": "t",
            "workers": 1,
            "charge_group": "grp",
            "queue": "gpu_h100",
            "models": [{"type": "script", "name": "m", "script_path": script_path, **(model_extra or {})}],
        }
        config.update(overrides)
        path = tmp_path / "task.yaml"
        path.write_text(yaml.safe_dump(config))
        return str(path)

    return make
