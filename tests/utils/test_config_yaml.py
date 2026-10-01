"""Reading a YAML config: its errors, and the one data_path + scale rule.

The loader called sys.exit(1); it also runs inside the dashboard (the
blockwise precheck, a finetuned model's base config), where SystemExit gets
past ``except Exception`` and the request thread dies without a response.
The CLIs turning ConfigError into a clean exit is tested with them, and a
bad model entry with the registry.
"""

import json
import logging
import os

import pytest
import zarr

from cellmap_flow.config.yaml import ConfigError, load_config, resolve_data_path
from cellmap_flow.jobs import settings


@pytest.mark.parametrize("yaml_text, message", [
    pytest.param("charge_group: g\nmodels: {}\n", "data_path", id="no-data-path"),
    pytest.param("data_path: /d.zarr\ncharge_group: g\nmodels: 3\n", "dict or list", id="models-neither-dict-nor-list"),
    pytest.param("- just\n- a list\n", "mapping", id="not-a-mapping"),
    pytest.param("data_path: /d.zarr\n", "charge_group", id="no-charge-group-and-none-saved"),
    # "false" is a string, and would turn it on.
    pytest.param('data_path: /d.zarr\ncharge_group: g\nresample: "false"\n', "true or false", id="resample-not-a-bool"),
])
def test_a_bad_yaml_file_is_a_config_error(tmp_path, monkeypatch, yaml_text, message):
    monkeypatch.setattr(settings, "load_server_config_cache", lambda: None)
    monkeypatch.setitem(settings.SERVER_CONFIG_DEFAULTS, "charge_group", "")
    (tmp_path / "c.yaml").write_text(yaml_text)
    with pytest.raises(ConfigError, match=message) as raised:
        load_config(str(tmp_path / "c.yaml"))
    assert not isinstance(raised.value, SystemExit)  # the dashboard's handlers answer it


def test_a_yaml_file_is_named_by_its_path_only(tmp_path):
    """open() takes an int as a file descriptor: handed a number, the loader
    read the YAML through whatever the process had open under it, and then
    closed it. A blockwise request's yaml_paths of [1] closed the
    dashboard's stdout."""
    (tmp_path / "c.yaml").write_text("data_path: /d.zarr\ncharge_group: g\nqueue: q\n")
    with open(tmp_path / "c.yaml") as config:
        with pytest.raises(ConfigError, match="named by its path"):
            load_config(config.fileno())
        os.fstat(config.fileno())  # still open


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
