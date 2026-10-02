"""What an inference server serves, pinned: .zattrs, .zarray per chain, model_info, one chunk.

Neuroglancer, the dashboard and finetune jobs read these, and any of them may
be older or newer than the server. So the metadata doesn't change, and
model_info only gains keys: each case below is what the server answered
before its code moved.
"""

import hashlib

import numpy as np
import pytest
import zarr

from cellmap_flow.models.models_config import ScriptModelConfig
from cellmap_flow.post.postprocessors import ChannelSelection, ThresholdPostprocessor
from cellmap_flow.server import CellMapFlowServer
from tests.utils.serving_helpers import (
    IDENTITY_MODEL,
    decode_chunk,
    get_json,
    layer,
    write_raw,
    write_script,
)

SQUEEZE = IDENTITY_MODEL.replace("model = Identity()", "") + """
chunk_output_axes = ("z", "y", "x")


class Squeeze(torch.nn.Module):
    def forward(self, x):
        return x[:, 0]


model = Squeeze()
"""

# 8 nm in, 16 nm out, two channels: each 2^3 block's mean, twice.
COARSER = """
input_voxel_size = Coordinate(8, 8, 8)
output_voxel_size = Coordinate(16, 16, 16)
read_shape = Coordinate(8, 8, 8) * input_voxel_size
write_shape = Coordinate(4, 4, 4) * output_voxel_size
output_channels = 2
channels = ["a", "b"]
block_shape = np.array((4, 4, 4, 2))


class Pool(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.avg_pool3d(x, 2).repeat(1, 2, 1, 1, 1)


model = Pool()
"""

# One voxel of context on each side, cropped off.
CROP = """
input_voxel_size = Coordinate(8, 8, 8)
output_voxel_size = Coordinate(8, 8, 8)
read_shape = Coordinate(6, 6, 6) * input_voxel_size
write_shape = Coordinate(4, 4, 4) * output_voxel_size
output_channels = 1
block_shape = np.array((4, 4, 4, 1))


class Crop(torch.nn.Module):
    def forward(self, x):
        return x[:, :, 1:-1, 1:-1, 1:-1]


model = Crop()
"""


def _data(side):
    return (np.arange(side**3) % 251).astype(np.uint8).reshape((side,) * 3)


def _ome_raw(tmp_path, data, voxel_size, translation):
    """A one-level OME-Zarr group; ``translation`` is voxel 0's centre."""
    group = zarr.open_group(str(tmp_path / "ome.zarr"), mode="w")
    group.create_dataset("s0", data=data, chunks=data.shape)
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
            "datasets": [
                {
                    "path": "s0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [voxel_size] * 3},
                        {"type": "translation", "translation": [translation] * 3},
                    ],
                }
            ],
        }
    ]
    return str(tmp_path / "ome.zarr")


THRESHOLD = [ThresholdPostprocessor(threshold=0.5)]

# name: (model script, raw writer, {chain name: postprocess}, chunk key)
CASES = {
    "identity": (IDENTITY_MODEL, lambda p: write_raw(p, _data(8)), {"threshold": THRESHOLD}, "1.0.1.0"),
    "no_channel": (SQUEEZE, lambda p: write_raw(p, _data(8)), {"threshold": THRESHOLD}, "0.1.1"),
    "8nm_to_16nm": (
        COARSER,
        lambda p: write_raw(p, _data(16)),
        {"one_channel": [ChannelSelection("1")]},
        "1.1.0.0",
    ),
    # Janelia's levels have their corner at -4 nm.
    "corner_at_-4nm": (CROP, lambda p: _ome_raw(p, _data(8), 8.0, 0.0), {}, "1.1.1.0"),
    # 6 nm data read as if it were 8 nm, voxel for voxel.
    "relabelled": (IDENTITY_MODEL, lambda p: _ome_raw(p, _data(8), 6.0, 3.0), {}, "1.0.0.0"),
}


def observe(tmp_path, name):
    script, raw, chains, key = CASES[name]
    server = CellMapFlowServer(raw(tmp_path), ScriptModelConfig(script_path=write_script(tmp_path, script)))
    client = server.app.test_client()
    datasets = {"plain": "plain", **{chain: layer(posts=posts) for chain, posts in chains.items()}}
    zarrays = {chain: get_json(client, f"/{path}/s0/.zarray") for chain, path in datasets.items()}
    info = get_json(client, "/__control__/model_info")
    for bound in ("output_min", "output_max"):  # of a random warmup input
        assert isinstance(info.pop(bound), float)
    meta = zarrays["plain"]
    chunk = decode_chunk(server, client.get(f"/plain/s0/{key}").data, meta["dtype"], meta["chunks"])
    return {
        "zattrs": get_json(client, "/plain/.zattrs"),
        "zarray": zarrays,
        "model_info": info,
        "chunk": [key, str(chunk.dtype), list(chunk.shape), hashlib.sha256(chunk.tobytes()).hexdigest()[:16]],
    }


def _zattrs(scale, translation, channel):
    axes = [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"]
    if channel:
        axes.append({"name": "c", "type": "channel"})
        scale, translation = scale + [1.0], translation + [0.0]
    transforms = [{"scale": scale, "type": "scale"}, {"translation": translation, "type": "translation"}]
    return {
        "multiscales": [
            {
                "axes": axes,
                "coordinateTransformations": [{"scale": [1.0] * len(axes), "type": "scale"}],
                "datasets": [{"coordinateTransformations": transforms, "path": "s0"}],
                "name": "plain",
                "version": "0.4",
            }
        ]
    }


def _zarray(shape, chunks, dtype):
    return {
        "chunks": chunks,
        "compressor": {"clevel": 5, "cname": "zstd", "id": "blosc", "shuffle": 1},
        "dtype": dtype,
        "fill_value": 0,
        "filters": None,
        "order": "C",
        "shape": shape,
        "zarr_format": 2,
    }


def _info(read, write, in_vs, out_vs, channels=1, names=None, effective=None, channel=True):
    return {
        "available": True,
        "channels": names,
        "input_voxel_size": [in_vs] * 3,
        "output_channels": channels,
        "output_class": "unbounded",
        "output_voxel_size": [out_vs] * 3,
        "read_shape": [read] * 3,
        "write_shape": [write] * 3,
        # Added with ModelGeometry; older dashboards ignore them.
        "effective_output_voxel_size": [effective or out_vs] * 3,
        "has_channel": channel,
        "output_axes": ["z", "y", "x"] + (["c"] if channel else []),
    }


EXPECTED = {
    "identity": {
        "zattrs": _zattrs([8.0] * 3, [4.0] * 3, channel=True),
        "zarray": {
            "plain": _zarray([8, 8, 8, 1], [4, 4, 4, 1], "<f4"),
            "threshold": _zarray([8, 8, 8, 1], [4, 4, 4, 1], "|u1"),
        },
        "model_info": _info(32, 32, 8, 8),
        "chunk": ["1.0.1.0", "float32", [4, 4, 4, 1], "5f7595af9016c839"],
    },
    "no_channel": {
        "zattrs": _zattrs([8.0] * 3, [4.0] * 3, channel=False),
        "zarray": {
            "plain": _zarray([8, 8, 8], [4, 4, 4], "<f4"),
            "threshold": _zarray([8, 8, 8], [4, 4, 4], "|u1"),
        },
        "model_info": _info(32, 32, 8, 8, channel=False),
        "chunk": ["0.1.1", "float32", [4, 4, 4], "0ec2a491f84bb09a"],
    },
    "8nm_to_16nm": {
        "zattrs": _zattrs([16.0] * 3, [8.0] * 3, channel=True),
        "zarray": {
            "plain": _zarray([8, 8, 8, 2], [4, 4, 4, 2], "<f4"),
            "one_channel": _zarray([8, 8, 8, 1], [4, 4, 4, 1], "<f4"),
        },
        "model_info": _info(64, 64, 8, 16, channels=2, names=["a", "b"]),
        "chunk": ["1.1.0.0", "float32", [4, 4, 4, 2], "c63bb6d645f2c0cf"],
    },
    "corner_at_-4nm": {
        # The corner is -4 nm, so voxel 0's centre is at 0.
        "zattrs": _zattrs([8.0] * 3, [0.0] * 3, channel=True),
        "zarray": {"plain": _zarray([8, 8, 8, 1], [4, 4, 4, 1], "<f4")},
        "model_info": _info(48, 32, 8, 8),
        "chunk": ["1.1.1.0", "float32", [4, 4, 4, 1], "75ad702d5f154932"],
    },
    "relabelled": {
        # Still served as 8 nm; model_info says what a voxel really is.
        "zattrs": _zattrs([8.0] * 3, [4.0] * 3, channel=True),
        "zarray": {"plain": _zarray([8, 8, 8, 1], [4, 4, 4, 1], "<f4")},
        "model_info": _info(32, 32, 8, 8, effective=6),
        "chunk": ["1.0.0.0", "float32", [4, 4, 4, 1], "9ec594a49555fe11"],
    },
}


@pytest.mark.parametrize("name", CASES)
def test_what_the_server_serves_is_unchanged(tmp_path, name):
    got, expected = observe(tmp_path, name), EXPECTED[name]
    assert got["zattrs"] == expected["zattrs"]
    assert got["zarray"] == expected["zarray"]
    assert got["chunk"] == expected["chunk"]
    # Additive only: every key an older dashboard reads, with its value.
    assert {k: got["model_info"].get(k) for k in expected["model_info"]} == expected["model_info"]
