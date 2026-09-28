"""One rule for data_path + scale, used by cellmap_flow, cellmap_flow_yaml and blockwise.

cellmap_flow and cellmap_flow_yaml appended ``scale`` to ``data_path``
unconditionally and blockwise ignored it, so the bundled examples, which set
``data_path: .../s3`` together with ``scale: s3``, opened ``.../s3/s3`` in the
yaml CLI and ``.../s3`` in blockwise. Now: an array is used as it is (with a
warning if ``scale`` names another level); a group has ``scale`` appended.
"""

import json
import logging
import os

import pytest
import zarr

from cellmap_flow.cli import yaml_cli
from cellmap_flow.models.models_config import ScriptModelConfig
from cellmap_flow.utils.config_utils import resolve_data_path


@pytest.fixture
def v2(tmp_path):
    root = zarr.open_group(str(tmp_path / "d.zarr"), mode="w")
    em = root.create_group("em")
    em.create_dataset("s3", shape=(4, 4, 4), dtype="u1")
    return str(tmp_path / "d.zarr" / "em")


def test_an_array_path_is_used_as_is_when_scale_agrees(v2, caplog):
    with caplog.at_level(logging.WARNING):
        assert resolve_data_path(f"{v2}/s3", "s3") == f"{v2}/s3"
    assert "ignored" not in caplog.text


def test_an_array_path_is_used_as_is_and_a_disagreeing_scale_warns(v2, caplog):
    with caplog.at_level(logging.WARNING):
        assert resolve_data_path(f"{v2}/s3", "s0") == f"{v2}/s3"
    assert "is ignored" in caplog.text


def test_a_group_path_gets_the_scale_appended(v2):
    assert resolve_data_path(v2, "s3") == os.path.join(v2, "s3")
    assert resolve_data_path(v2 + "/", "/s3") == os.path.join(v2 + "/", "s3")


def test_zarr_v3_nodes_are_recognised(tmp_path):
    group = tmp_path / "v3.zarr"
    (group / "s0").mkdir(parents=True)
    (group / "zarr.json").write_text(json.dumps({"zarr_format": 3, "node_type": "group"}))
    (group / "s0" / "zarr.json").write_text(json.dumps({"zarr_format": 3, "node_type": "array"}))
    assert resolve_data_path(str(group), "s0") == os.path.join(str(group), "s0")
    assert resolve_data_path(str(group / "s0"), "s0") == str(group / "s0")


def test_no_scale_or_an_unknown_path(v2):
    assert resolve_data_path(v2, None) == v2
    assert resolve_data_path("s3://bucket/d.zarr/em", "s3") == "s3://bucket/d.zarr/em/s3"


def test_cellmap_flow_yaml_does_not_double_the_scale(v2, monkeypatch):
    commands = []
    monkeypatch.setattr(
        yaml_cli, "start_hosts", lambda command, **kwargs: commands.append(command)
    )
    from cellmap_flow.utils import neuroglancer_utils

    monkeypatch.setattr(neuroglancer_utils, "generate_neuroglancer_url", lambda *a, **k: None)
    model = ScriptModelConfig(script_path="/s.py", name="m", scale="s3")

    yaml_cli.run_multiple([model], f"{v2}/s3", "grp", "gpu_h100")

    assert len(commands) == 1
    assert commands[0].endswith(f"-d {v2}/s3"), commands[0]
