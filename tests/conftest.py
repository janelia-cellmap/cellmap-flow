"""Shared test setup, and the stand-ins most areas need.

HOME is redirected before anything imports cellmap_flow: importing the package
executes ~/.cellmap_flow/plugins/*.py and the Flow singleton reads
~/.cellmap_flow/server_config.yaml, so otherwise the suite depends on (and can
write to) the developer's real config.

The fixtures at the end are the datasets (``raw_zarr``, ``ome_pyramid``,
``write_array``), a script model (``model_script``), LSF (``fake_lsf``), the
viewer (``viewer``) and the dashboard (``dashboard``).
"""

import ctypes
import importlib
import os
import shutil
import string
import subprocess
import sys
import tempfile
from collections import deque
from types import SimpleNamespace

os.environ["HOME"] = tempfile.mkdtemp(prefix="cellmap_flow_test_home_")

# A process has one libstdc++, the first one anything loads. torch's pip
# wheel finds it on the system's library path, while numpy, zarr and the
# rest of a conda environment load the environment's newer copy. So a run
# whose first import was torch (tests/finetune on its own) got the system's,
# and sqlite3 then failed to load: its libicu needs CXXABI_1.3.15, which the
# system copy lacks. Loading the environment's copy here makes it the first
# in every run, as it is when numpy happens to come first.
_LIBSTDCXX = os.path.join(sys.prefix, "lib", "libstdc++.so.6")
if os.path.exists(_LIBSTDCXX):
    ctypes.CDLL(_LIBSTDCXX, mode=ctypes.RTLD_GLOBAL)

import pytest  # noqa: E402


def _can_import(module):
    # peft raises ImportError subclasses (and occasionally other errors) when
    # its transformers/huggingface-hub pins don't match, not only when absent.
    try:
        importlib.import_module(module)
    except Exception:
        return False
    return True


def _has_cuda():
    try:
        import torch
    except Exception:
        return False
    return torch.cuda.is_available()


_REQUIREMENTS = {
    "finetune": (lambda: _can_import("peft"), "peft is not importable"),
    "gpu": (_has_cuda, "no CUDA device"),
    "lsf": (lambda: shutil.which("bsub") is not None, "bsub not found"),
    "minio": (
        lambda: shutil.which("minio") is not None and shutil.which("mc") is not None,
        "minio/mc binaries not found",
    ),
    "network": (
        lambda: os.environ.get("CELLMAP_FLOW_NETWORK_TESTS") == "1",
        "set CELLMAP_FLOW_NETWORK_TESTS=1 to run",
    ),
}


def pytest_collection_modifyitems(config, items):
    available = {}
    for item in items:
        for marker, (check, reason) in _REQUIREMENTS.items():
            # Not `marker in item.keywords`: keywords include parent package
            # names, so every test under tests/finetune/ would match "finetune".
            if item.get_closest_marker(marker) is None:
                continue
            if marker not in available:
                available[marker] = check()
            if not available[marker]:
                item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(autouse=True)
def _restore_flow_state():
    """Undo whatever a test does to the process-wide Flow singleton."""
    from cellmap_flow.globals import g

    saved = {
        key: value.copy() if isinstance(value, (list, dict, set, deque)) else value
        for key, value in vars(g).items()
    }
    yield
    vars(g).clear()
    vars(g).update(saved)


@pytest.fixture(autouse=True)
def _restore_root_logging():
    """Undo a CLI's logging setup, which a test runs under CliRunner.

    The CLIs call basicConfig(force=True), and under CliRunner its handler
    writes to a stderr the runner closes afterwards; left on the root logger,
    it made daisy's monitor raise "I/O operation on closed file" in whichever
    blockwise test came later.
    """
    import logging

    root = logging.getLogger()
    level = root.level
    yield
    for handler in list(root.handlers):
        if getattr(getattr(handler, "stream", None), "closed", False):
            root.removeHandler(handler)
    root.setLevel(level)


# --- datasets --------------------------------------------------------------------


@pytest.fixture
def raw_zarr(tmp_path):
    """``raw_zarr(data=None, voxel_size=(8, 8, 8), offset=(0, 0, 0), name="raw")``: a
    zarr v2 array with funlib's resolution/offset attributes (the offset is voxel
    0's corner); its path. The default data is 16^3 uint8 voxels counting up."""
    import numpy as np

    from tests.utils.serving_helpers import write_raw

    def make(data=None, voxel_size=(8, 8, 8), offset=(0, 0, 0), name="raw"):
        if data is None:
            data = (np.arange(16**3) % 251).astype(np.uint8).reshape((16,) * 3)
        return write_raw(tmp_path, data, voxel_size, offset, name)

    return make


@pytest.fixture
def ome_pyramid(tmp_path):
    """``ome_pyramid(levels, shape=(16, 16, 16), zarr_format=2, ...)``: an OME-Zarr
    multiscale group; its path.

    ``levels`` holds a (voxel size, translation) per level, each a number or a
    z, y, x tuple; the translation is OME's, voxel 0's centre, and None leaves
    it out. Level i is ``shape >> i`` voxels, chunked in halves, and each voxel
    holds its z index + 1, so a read shows which voxels it hit and padding
    cannot pass for data. ``channels`` adds a leading channel axis.
    """
    import json

    import numpy as np
    import tensorstore as ts
    import zarr

    def per_axis(value):
        return [float(v) for v in (value if isinstance(value, (tuple, list)) else [value] * 3)]

    def make(levels=((8, 0), (16, 4)), shape=(16, 16, 16), zarr_format=2, name="pyramid.zarr",
             channels=0, unit="nanometer"):
        path = str(tmp_path / name)
        lead = [1.0] if channels else []
        axes = [{"name": "c", "type": "channel"}] * bool(channels)
        axes += [{"name": a, "type": "space", "unit": unit} for a in "zyx"]
        datasets, arrays = [], {}
        for i, (scale, translation) in enumerate(levels):
            spatial = [max(1, s >> i) for s in shape]
            data = np.broadcast_to(np.arange(1, spatial[0] + 1, dtype=np.uint8)[:, None, None], spatial)
            arrays[f"s{i}"] = np.ascontiguousarray(np.stack([data] * channels) if channels else data)
            transforms = [{"type": "scale", "scale": lead + per_axis(scale)}]
            if translation is not None:
                transforms.append({"type": "translation", "translation": [0.0] * len(lead) + per_axis(translation)})
            datasets.append({"path": f"s{i}", "coordinateTransformations": transforms})
        multiscales = [{"version": "0.4" if zarr_format == 2 else "0.5", "axes": axes, "datasets": datasets}]
        if zarr_format == 2:
            # A name inside a container (a.zarr/em) makes the groups above it too.
            container, _, inner = path.partition(".zarr")
            group = zarr.open_group(container + ".zarr", mode="a")
            group = group.require_group(inner.strip("/")) if inner.strip("/") else group
            for key, data in arrays.items():
                group.create_dataset(key, data=data, chunks=tuple(max(1, s // 2) for s in data.shape))
            group.attrs["multiscales"] = multiscales
            return path
        for key, data in arrays.items():
            grid = {"name": "regular", "configuration": {"chunk_shape": [max(1, s // 2) for s in data.shape]}}
            metadata = {"shape": list(data.shape), "data_type": "uint8", "chunk_grid": grid}
            spec = {"driver": "zarr3", "kvstore": {"driver": "file", "path": f"{path}/{key}"}, "metadata": metadata}
            ts.open(spec, create=True).result()[...] = data
        with open(f"{path}/zarr.json", "w") as f:
            json.dump({"zarr_format": 3, "node_type": "group", "attributes": {"ome": {"multiscales": multiscales}}}, f)
        return path

    return make


@pytest.fixture
def write_array(tmp_path):
    """``write_array(fmt, data, attrs=None, name=None, **zarr_kwargs)``: ``data`` (z, y, x)
    as one array of ``fmt`` -- "zarr2", "zarr3", "n5" or "precomputed" -- with
    ``attrs`` as that format keeps them (the scale metadata for precomputed);
    its path. ``name`` is the path under tmp_path: by default a.zarr/raw,
    a.n5/raw, v3_array or pc; a container of its own (root.zarr) holds the
    array at its root. Chunks are halves of the shape unless given."""
    import numpy as np
    import tensorstore as ts
    import zarr
    from zarr.n5 import N5FSStore

    names = {"zarr2": "a.zarr/raw", "n5": "a.n5/raw", "zarr3": "v3_array", "precomputed": "pc"}

    def make(fmt, data, attrs=None, name=None, **zarr_kwargs):
        data = np.asarray(data)
        path = str(tmp_path / (name or names[fmt]))
        zarr_kwargs.setdefault("chunks", tuple(max(1, s // 2) for s in data.shape))
        if fmt in ("zarr2", "n5"):
            container, _, inner = path.partition(".n5" if fmt == "n5" else ".zarr")
            container += ".n5" if fmt == "n5" else ".zarr"
            store = N5FSStore(container) if fmt == "n5" else container
            if inner.strip("/"):
                array = zarr.open_group(store, mode="a").create_dataset(
                    inner.strip("/"), shape=data.shape, dtype=data.dtype, **zarr_kwargs
                )
            else:
                array = zarr.open(store, mode="w", shape=data.shape, dtype=data.dtype, **zarr_kwargs)
            array[...] = data
            array.attrs.update(attrs or {})
            return path
        if fmt == "zarr3":
            grid = {"name": "regular", "configuration": {"chunk_shape": list(zarr_kwargs["chunks"])}}
            metadata = {"shape": list(data.shape), "data_type": str(data.dtype), "chunk_grid": grid}
            spec = {"driver": "zarr3", "kvstore": {"driver": "file", "path": path},
                    "metadata": {**metadata, **({"attributes": attrs} if attrs else {})}}
            ts.open(spec, create=True).result()[...] = data
            return path
        spec = {
            "driver": "neuroglancer_precomputed", "kvstore": {"driver": "file", "path": path},
            "multiscale_metadata": {"type": "image", "data_type": str(data.dtype), "num_channels": 1},
            "scale_metadata": {"size": list(data.shape[::-1]), "encoding": "raw", **(attrs or {})},
        }
        ts.open(spec, create=True).result()[..., 0] = data.transpose()
        return "precomputed://" + path

    return make


@pytest.fixture
def model_script(tmp_path):
    """``model_script(body=IDENTITY_MODEL, name="model.py", **values)``: a script
    model file; its path. ``values`` fill the body's ``$name`` placeholders."""
    from tests.utils.serving_helpers import IDENTITY_MODEL, write_script

    def make(body=IDENTITY_MODEL, name="model.py", **values):
        return write_script(tmp_path, string.Template(body).substitute(values) if values else body, name)

    return make


# --- LSF, the viewer and the dashboard --------------------------------------------


class FakeLSF:
    """``subprocess.run`` for the LSF commands, and a clock for ``jobs.lsf``.

    ``answers[command]`` scripts what a command answers: a list used in order,
    its last entry repeating, or a function of (argv, kwargs). An answer is
    stdout (exit 0), (returncode, stdout, stderr), or an exception to raise;
    ``check=True`` turns a non-zero exit into CalledProcessError, as it would.
    A command with no script fails the test. ``calls`` keeps every argv, and
    ``timeline`` when each came by the clock, which moves only when jobs.lsf
    sleeps or a command takes its ``cost`` in seconds.
    """

    def __init__(self, monkeypatch):
        from cellmap_flow.jobs import lsf

        self.answers = {"which": ["/usr/bin/bsub\n"]}
        self.cost, self.calls, self.timeline = {}, [], []
        self.start = self.now = 1000.0
        self.sleeps = 0
        monkeypatch.setattr(lsf.subprocess, "run", self._run)
        monkeypatch.setattr(lsf, "time", SimpleNamespace(time=self._clock, monotonic=self._clock, sleep=self._sleep))

    def _clock(self):
        return self.now

    def _sleep(self, seconds):
        self.sleeps += 1
        self.now += seconds

    def _run(self, argv, **kwargs):
        argv = [str(a) for a in argv]
        self.calls.append(argv)
        self.timeline.append((round(self.now - self.start, 3), argv[0]))
        self.now += self.cost.get(argv[0], 0)
        script = self.answers.get(argv[0])
        if script is None:
            raise AssertionError(f"unexpected command {argv}")
        answer = script(argv, kwargs) if callable(script) else (script.pop(0) if len(script) > 1 else script[0])
        if isinstance(answer, BaseException):
            raise answer
        returncode, stdout, stderr = (0, answer, "") if isinstance(answer, str) else answer
        if kwargs.get("check") and returncode:
            raise subprocess.CalledProcessError(returncode, argv, stdout, stderr)
        return subprocess.CompletedProcess(argv, returncode, stdout, stderr)

    def commands(self, name=None):
        return [argv for argv in self.calls if argv[0] == name] if name else [argv[0] for argv in self.calls]


@pytest.fixture
def fake_lsf(monkeypatch, tmp_path):
    """LSF as ``FakeLSF`` answers it, with server logs under tmp_path."""
    from cellmap_flow.jobs import launch

    monkeypatch.setattr(launch, "SERVER_LOG_DIR", tmp_path / "server_logs")
    return FakeLSF(monkeypatch)


@pytest.fixture
def viewer():
    """A neuroglancer viewer without its web server, as ``g.viewer``: z, y, x in 8 nm voxels."""
    import neuroglancer
    from neuroglancer.viewer_base import ViewerBase

    from cellmap_flow.globals import g

    g.viewer = ViewerBase()
    with g.viewer.txn() as s:
        s.dimensions = neuroglancer.CoordinateSpace(names=["z", "y", "x"], units="nm", scales=[8, 8, 8])
    return g.viewer


@pytest.fixture
def dashboard():
    """The dashboard's Flask test client."""
    from cellmap_flow.dashboard.app import app

    return app.test_client()
