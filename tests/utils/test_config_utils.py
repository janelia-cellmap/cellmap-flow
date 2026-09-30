"""Reading a YAML config: its errors, and the one data_path + scale rule.

config_utils called sys.exit(1); it also runs inside the dashboard (the
blockwise precheck, a finetuned model's base config), where SystemExit gets
past ``except Exception`` and the request thread dies without a response.
The CLIs turning ConfigError into a clean exit is tested with them.
"""

import json
import logging
import os

import pytest
import zarr

import cellmap_flow.globals as G
from cellmap_flow.utils.config_utils import ConfigError, build_models, load_config, resolve_data_path


@pytest.mark.parametrize(
    "yaml_text, models, message",
    [
        ("charge_group: g\nmodels: {}\n", None, "data_path"),
        ("data_path: /d.zarr\ncharge_group: g\nmodels: 3\n", None, "dict or list"),
        ("- just\n- a list\n", None, "mapping"),
        ("data_path: /d.zarr\n", None, "charge_group"),  # and none saved either
        (None, {"m": {"script_path": "/s.py"}}, "missing 'type'"),
        (None, {"m": {"type": "no-such-kind"}}, "unrecognized type"),
        (None, {"m": {"type": "dacapo", "run_name": "r"}}, "missing required parameter 'iteration'"),
        (None, {"m": "not a mapping"}, "must be a mapping"),
        (None, [{"type": "script", "script_path": "/s.py"}], "name"),
    ],
)
def test_a_bad_config_is_a_config_error(tmp_path, monkeypatch, yaml_text, models, message):
    monkeypatch.setattr(G, "load_server_config_cache", lambda: None)
    monkeypatch.setitem(G.SERVER_CONFIG_DEFAULTS, "charge_group", "")
    with pytest.raises(ConfigError, match=message) as raised:
        if models is None:
            (tmp_path / "c.yaml").write_text(yaml_text)
            load_config(str(tmp_path / "c.yaml"))
        else:
            build_models(models)
    assert not isinstance(raised.value, SystemExit)  # the dashboard's handlers answer it


def test_one_rule_for_data_path_and_scale(tmp_path, caplog):
    """An array is used as it is, warning when scale names another level; a
    group gets the scale appended. The launchers used to append it always
    (.../s3/s3 for the bundled examples) or never (blockwise)."""
    em = zarr.open_group(str(tmp_path / "d.zarr"), mode="w").create_group("em")
    em.create_dataset("s3", shape=(4, 4, 4), dtype="u1")
    group = str(tmp_path / "d.zarr" / "em")
    v3 = tmp_path / "v3.zarr"
    for path, node_type in ((v3, "group"), (v3 / "s0", "array")):
        path.mkdir(parents=True, exist_ok=True)
        (path / "zarr.json").write_text(json.dumps({"zarr_format": 3, "node_type": node_type}))

    with caplog.at_level(logging.WARNING):
        assert resolve_data_path(f"{group}/s3", "s3") == f"{group}/s3"
    assert not caplog.records
    with caplog.at_level(logging.WARNING):
        assert resolve_data_path(f"{group}/s3", "s0") == f"{group}/s3"
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert resolve_data_path(group, "s3") == resolve_data_path(group + "/", "/s3") == os.path.join(group, "s3")
    assert resolve_data_path(str(v3), "s0") == str(v3 / "s0")
    assert resolve_data_path(str(v3 / "s0"), "s0") == str(v3 / "s0")
    assert resolve_data_path(group, None) == group
    assert resolve_data_path("s3://bucket/d.zarr/em", "s3") == "s3://bucket/d.zarr/em/s3"
