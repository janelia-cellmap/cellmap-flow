"""A bad YAML config raises ConfigError; the CLIs turn it into a clean exit.

config_utils called sys.exit(1). It also runs inside the dashboard (the
blockwise precheck, a finetuned model's base config), where SystemExit gets
past ``except Exception`` and the request thread dies without a response.
"""

import pytest
from click.testing import CliRunner

from cellmap_flow.utils import config_utils
from cellmap_flow.utils.config_utils import ConfigError, build_model_from_entry, build_models, load_config


def _write(tmp_path, text):
    path = tmp_path / "c.yaml"
    path.write_text(text)
    return str(path)


@pytest.mark.parametrize(
    "text, message",
    [
        ("charge_group: g\nmodels: {}\n", "data_path"),
        ("data_path: /d.zarr\ncharge_group: g\nmodels: 3\n", "dict or list"),
        ("- just\n- a list\n", "mapping"),
    ],
)
def test_load_config_raises_config_error(tmp_path, text, message):
    with pytest.raises(ConfigError, match=message):
        load_config(_write(tmp_path, text))


def test_a_missing_charge_group_with_nothing_cached_is_a_config_error(tmp_path, monkeypatch):
    import cellmap_flow.globals as G

    monkeypatch.setattr(G, "load_server_config_cache", lambda: None)
    monkeypatch.setitem(G.SERVER_CONFIG_DEFAULTS, "charge_group", "")
    with pytest.raises(ConfigError, match="charge_group"):
        load_config(_write(tmp_path, "data_path: /d.zarr\n"))


@pytest.mark.parametrize(
    "entry, message",
    [
        ({"script_path": "/s.py"}, "missing 'type'"),
        ({"type": "no-such-kind"}, "unrecognized type"),
        ({"type": "dacapo", "run_name": "r"}, "missing required parameter 'iteration'"),
        ("not a mapping", "must be a mapping"),
    ],
)
def test_build_model_from_entry_raises_config_error(entry, message):
    with pytest.raises(ConfigError, match=message):
        build_model_from_entry(entry, model_name="m")


def test_a_list_entry_without_a_name_is_a_config_error():
    with pytest.raises(ConfigError, match="name"):
        build_models([{"type": "script", "script_path": "/s.py"}])


def test_config_error_is_an_ordinary_exception():
    # So the dashboard's `except Exception` handlers answer the request.
    assert issubclass(ConfigError, Exception)
    assert not issubclass(ConfigError, SystemExit)
    assert config_utils.ConfigError is ConfigError


def test_cellmap_flow_yaml_reports_the_problem_and_exits_non_zero(tmp_path):
    from cellmap_flow.cli.yaml_cli import main

    path = _write(tmp_path, "data_path: /d.zarr\ncharge_group: g\nmodels:\n  m: {type: nope}\n")
    result = CliRunner().invoke(main, [path, "--validate-only"])
    assert result.exit_code == 1
    assert "unrecognized type 'nope'" in result.output
    assert not isinstance(result.exception, ConfigError), "caught, not a traceback"


def test_cellmap_flow_blockwise_reports_the_problem_and_exits_non_zero(tmp_path):
    from cellmap_flow.blockwise.cli import cli

    path = _write(tmp_path, "charge_group: g\nmodels: {}\n")
    result = CliRunner().invoke(cli, [path])
    assert result.exit_code == 1
    assert "data_path" in result.output
