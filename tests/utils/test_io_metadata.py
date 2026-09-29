"""The io/ package: paths, metadata, multiscale levels and OME attributes.

What the old readers returned on real layouts is pinned in test_io_matrix;
this file covers the new interfaces themselves.
"""

import json
import os
import subprocess
import sys

import pytest
import zarr

from cellmap_flow.io import paths

# ---------------------------------------------------------------------------
# io/ stays importable without the application around it
# ---------------------------------------------------------------------------

_HEAVY = ("cellmap_flow.globals", "flask", "neuroglancer", "torch", "huggingface_hub", "peft")


@pytest.mark.parametrize("module", ["cellmap_flow.io", "cellmap_flow.io.paths"])
def test_io_modules_import_nothing_heavy(module, tmp_path):
    # globals configures logging and reads ~/.cellmap_flow on import; the
    # others are slow or optional. A fresh interpreter, so that what this
    # test process has already imported does not hide anything.
    code = (
        f"import sys, {module}; "
        f"loaded = [m for m in {_HEAVY!r} if m in sys.modules]; "
        "assert not loaded, loaded"
    )
    env = {**os.environ, "HOME": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=os.getcwd()
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def test_split_container_at_the_last_suffix():
    assert paths.split_container("/d/x.zarr/em/s0") == ("/d/x.zarr", "em/s0")
    assert paths.split_container("/d/x.zarr") == ("/d/x.zarr", "")
    assert paths.split_container("/d/a.n5/b.zarr/raw") == ("/d/a.n5/b.zarr", "raw")
    assert paths.split_container("/d/a.zarr/b.n5/raw") == ("/d/a.zarr/b.n5", "raw")
    assert paths.split_container("s3://bucket/x.zarr/raw") == ("s3://bucket/x.zarr", "raw")


def test_split_container_without_a_suffix_finds_the_group(tmp_path):
    root = zarr.open_group(str(tmp_path / "plain"), mode="w")
    root.create_group("em").create_dataset("s0", shape=(2, 2, 2), dtype="u1")
    # The nearest .zgroup going up is em's own.
    assert paths.split_container(str(tmp_path / "plain" / "em" / "s0")) == (
        str(tmp_path / "plain" / "em"),
        "s0",
    )
    with pytest.raises(RuntimeError):
        paths.split_container(str(tmp_path / "nothing" / "here"))
    with pytest.raises(RuntimeError):
        paths.split_container("https://host/no/suffix")


def test_split_dataset_path_still_appends_a_scale(tmp_path):
    from cellmap_flow.utils.ds import split_dataset_path

    assert split_dataset_path("/d/x.zarr/em", scale=1) == ("/d/x.zarr", "em/s1")
    assert split_dataset_path("/d/x.zarr", scale=0) == ("/d/x.zarr", "/s0")
    zarr.open_group(str(tmp_path / "plain"), mode="w")
    assert split_dataset_path(str(tmp_path / "plain"), scale=2) == (str(tmp_path / "plain"), "s2")


def test_join_and_normalize():
    assert paths.join("https://host/x.zarr/", "em", "s0") == "https://host/x.zarr/em/s0"
    assert paths.join("/d/x.zarr", "em", "s0") == os.path.join("/d/x.zarr", "em", "s0")
    assert paths.normalize_path("/d/my\\ data.zarr") == "/d/my data.zarr"
    # Shell escapes are a filesystem thing; a URL is left alone.
    assert paths.normalize_path("https://host/my\\ data.zarr") == "https://host/my\\ data.zarr"
    assert paths.is_remote("s3://b/x") and paths.is_remote("http://h/x")
    assert not paths.is_remote("gs://b/x") and not paths.is_remote("/d/x")


def _v3_node(path, node_type):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "zarr.json"), "w") as f:
        json.dump({"zarr_format": 3, "node_type": node_type}, f)


def test_detect_format(tmp_path):
    _v3_node(str(tmp_path / "v3.zarr"), "group")
    _v3_node(str(tmp_path / "v3.zarr" / "s0"), "array")
    assert paths.detect_format(str(tmp_path / "v3.zarr" / "s0")) == "zarr3"
    assert paths.detect_format(str(tmp_path / "v3.zarr")) == "zarr3"
    assert paths.detect_format(str(tmp_path / "v2.zarr" / "s0")) == "zarr2"
    assert paths.detect_format(str(tmp_path / "a.n5" / "raw")) == "n5"
    assert paths.detect_format("precomputed:///d/pc") == "precomputed"
    assert paths.detect_format("gs://bucket/pc") == "precomputed"
    assert paths.detect_format("https://host/a.n5/raw") == "n5"
    assert paths.detect_format("https://host/x.zarr/raw") == "zarr2"
    # zarr v3 is only read from the local filesystem.
    assert paths.find_v3_container("https://host/v3.zarr") is None
    assert paths.find_v3_container(str(tmp_path / "v3.zarr" / "s0" / "c")) == str(
        tmp_path / "v3.zarr" / "s0"
    )


def test_zarr_container_markers(tmp_path):
    zarr.open_group(str(tmp_path / "g"), mode="w")
    _v3_node(str(tmp_path / "v3"), "group")
    (tmp_path / "empty").mkdir()
    assert paths.is_zarr_container(str(tmp_path / "g"))
    assert paths.is_zarr_container(str(tmp_path / "v3"))
    assert not paths.is_zarr_container(str(tmp_path / "empty"))
    assert not paths.is_zarr_container("https://host/x.zarr")
