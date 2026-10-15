"""Small datasets, script models and requests for serving tests; conftest.py's fixtures use them."""

import os
import textwrap

import numcodecs
import numpy as np
import zarr

from cellmap_flow.pipeline_spec import PipelineSpec
from cellmap_flow.serving.protocol import ARGS_KEY, FORMAT


def write_raw(tmp_path, data, resolution=(8, 8, 8), offset=(0, 0, 0), name="raw"):
    """A zarr v2 array with resolution/offset attrs; returns its path."""
    root = zarr.open(str(tmp_path / "raw.zarr"), mode="a")
    arr = root.create_dataset(name, data=np.asarray(data), chunks=np.asarray(data).shape)
    arr.attrs["resolution"] = list(resolution)
    arr.attrs["offset"] = list(offset)
    return os.path.join(str(tmp_path / "raw.zarr"), name)


def write_script(tmp_path, body, name="model.py"):
    """A model script: ``body`` is appended to the usual imports."""
    path = tmp_path / name
    path.write_text(
        textwrap.dedent(
            """
            import numpy as np
            import torch
            from funlib.geometry import Coordinate
            """
        )
        + textwrap.dedent(body)
    )
    return str(path)


IDENTITY_MODEL = """
input_voxel_size = Coordinate(8, 8, 8)
output_voxel_size = Coordinate(8, 8, 8)
read_shape = Coordinate(4, 4, 4) * input_voxel_size
write_shape = Coordinate(4, 4, 4) * output_voxel_size
output_channels = 1
block_shape = np.array((4, 4, 4, 1))


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


model = Identity()
"""

# $in_vs nm in, $out_vs nm out, channels "a" and "b": each output voxel is
# the mean of the input voxels it covers, once per channel.
POOLING_MODEL = """
input_voxel_size = Coordinate($in_vs, $in_vs, $in_vs)
output_voxel_size = Coordinate($out_vs, $out_vs, $out_vs)
write_shape = Coordinate(4, 4, 4) * output_voxel_size
read_shape = write_shape
channels = ["a", "b"]
output_channels = 2
block_shape = np.array((4, 4, 4, 2))


class Pool(torch.nn.Module):
    def forward(self, x):
        if $out_vs > $in_vs:
            x = torch.nn.functional.avg_pool3d(x, $out_vs // $in_vs)
        return x.repeat(1, 2, 1, 1, 1)


model = Pool()
"""


def layer(norms=(), posts=(), model="m"):
    """The dataset path a layer URL for this chain asks the server for."""
    return f"{model}{ARGS_KEY}{PipelineSpec.from_steps(list(norms), list(posts)).to_url_blob()}{ARGS_KEY}"


def read_chunk(client, name, index):
    """Chunk ``index`` (z, y, x) of layer ``name`` through ``client`` (a
    starlette TestClient on the server's app), decoded as the array's
    ``.zarray`` says: every channel, channels first, cropped to the voxels
    inside the array (a chunk the data ends in is served whole)."""
    meta = client.get(f"/{name}/{FORMAT}/s0/.zarray").json()
    key = "/".join(map(str, ((0,) if len(meta["shape"]) == 4 else ()) + tuple(index)))
    response = client.get(f"/{name}/{FORMAT}/s0/{key}")
    assert response.status_code == 200, response.text
    raw = numcodecs.get_codec(meta["compressor"]).decode(response.content) if meta["compressor"] else response.content
    inside = [min(c, s - i * c) for c, s, i in zip(meta["chunks"][-3:], meta["shape"][-3:], index)]
    return np.frombuffer(raw, dtype=np.dtype(meta["dtype"])).reshape(meta["chunks"])[
        (slice(None),) * (len(meta["shape"]) - 3) + tuple(slice(0, n) for n in inside)
    ]


def get_json(client, url):
    response = client.get(url)
    assert response.status_code == 200, response.text
    return response.json()
