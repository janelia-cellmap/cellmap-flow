"""A model entry's ``env``: the environment its server runs in (models.envs).

It is not a constructor argument, so it rides beside the model: the YAML,
the dashboard's form and ``infer --env`` set it, to_dict() and the launch
entry carry it, and the server command takes it off the --model JSON and
runs the server in that environment instead. The command is still one LSF
shell line, with no single quote in it (jobs.spec.shell_join).
"""

import json
import shlex
import textwrap
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from cellmap_flow.config.yaml import ConfigError
from cellmap_flow.models import geometry_cache, registry
from cellmap_flow.models.configs.base import ModelEnvError
from cellmap_flow.models.models_config import FinetuneModelConfig, ScriptModelConfig
from cellmap_flow.serving import launch

MANIFEST = """
[pypi-dependencies]
cellmap-flow = { path = ".", editable = true, extras = ["finetune"] }

[feature.cellpose4.pypi-dependencies]
cellpose = ">=4"

[feature.bare.pypi-dependencies]
cellpose = ">=4"

[environments]
cellpose4 = { features = ["cellpose4"], solve-group = "cellpose4" }
bare = { features = ["bare"], no-default-feature = true }
"""


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    """A pixi.toml with a cellpose4 environment, and pixi at /opt/pixi."""
    path = tmp_path / "checkout" / "pixi.toml"
    path.parent.mkdir()
    path.write_text(textwrap.dedent(MANIFEST))
    monkeypatch.setenv("CELLMAP_FLOW_PIXI_MANIFEST", str(path))
    monkeypatch.setenv("PIXI_EXE", "/opt/pixi")
    return path


@pytest.fixture
def venv(tmp_path):
    """A directory environment: one with a bin/python."""
    (tmp_path / "my env" / "bin").mkdir(parents=True)
    (tmp_path / "my env" / "bin" / "python").write_text("")
    return str(tmp_path / "my env")


def test_env_goes_round_the_registry_and_never_reaches_the_server(manifest):
    entry = {"type": "script", "script_path": "/s.py", "env": "cellpose4"}
    model = registry.build_model(entry, "cp")
    assert model.env == "cellpose4"
    assert model.to_dict() == {**entry, "name": "cp"} == registry.build_model(model.to_dict(), "cp").to_dict()
    assert model.launch_entry["env"] == "cellpose4"
    from_form = registry.instantiate_model_config("ScriptModelConfig", {"script_path": "/s.py", "env": "cellpose4"})
    assert from_form.env == "cellpose4"
    # The trainer's model too, so the serving YAMLs it writes keep it.
    from cellmap_flow.finetune.model_loading import model_config_from_entry

    assert model_config_from_entry(model.to_dict()).env == "cellpose4"

    argv = launch.server_argv(model, "/d/raw.zarr")
    assert argv == [
        "/opt/pixi", "run", "--frozen", "--manifest-path", str(manifest), "-e", "cellpose4", "cellmap_flow", "serve",
        "--model", '{"type":"script","script_path":"/s.py","name":"cp"}', "-d", "/d/raw.zarr",
    ]
    assert launch.server_argv_for("script", {"script_path": "/s.py", "name": "cp", "env": "cellpose4"},
                                  "/d/raw.zarr") == argv


def test_a_directory_env_serves_with_its_own_python_in_one_lsf_line(venv):
    model = ScriptModelConfig(script_path="/s.py", name="cp")
    model.env = venv
    command = launch.server_command(model, "/d/raw.zarr", resample=True)
    assert "'" not in command
    assert shlex.split(command) == [
        f"{venv}/bin/python", "-P", "-m", "cellmap_flow.cli.main", "serve",
        "--model", '{"type":"script","script_path":"/s.py","name":"cp"}', "-d", "/d/raw.zarr", "--resample",
    ]


def test_without_an_env_nothing_changes():
    model = ScriptModelConfig(script_path="/s.py", name="cp")
    assert model.env is None and "env" not in model.to_dict() and "env" not in model.launch_entry


@pytest.mark.parametrize(
    "env, error",
    [
        ("cellpose5", r"has no environment 'cellpose5'; it has bare, cellpose4, default"),
        ("relative/env", "must be an absolute path"),
        ("/no/such/env", "has no bin/python"),
        (7, "must be a pixi environment's name or a path"),
    ],
)
def test_an_env_that_cannot_be_used_is_refused_when_the_entry_is_read(manifest, env, error):
    with pytest.raises(ConfigError, match=error):
        registry.build_model({"type": "script", "script_path": "/s.py", "env": env}, "cp")


def test_a_pixi_env_without_a_manifest_says_how_to_name_one(tmp_path, monkeypatch):
    monkeypatch.setenv("CELLMAP_FLOW_PIXI_MANIFEST", str(tmp_path / "nowhere.toml"))
    with pytest.raises(ConfigError, match="Set CELLMAP_FLOW_PIXI_MANIFEST"):
        registry.build_model({"type": "script", "script_path": "/s.py", "env": "cellpose4"}, "cp")


def test_infer_takes_env(manifest, monkeypatch):
    from cellmap_flow.cli import infer
    from cellmap_flow.cli.main import cli
    from cellmap_flow.dashboard.services import startup
    from cellmap_flow.jobs.settings import LauncherSettings

    commands = []
    monkeypatch.setattr(infer, "start_hosts", lambda command, *a, **k: commands.append(command))
    monkeypatch.setattr(startup, "generate_neuroglancer_url", lambda path, wrap_raw=True: None)
    monkeypatch.setattr(LauncherSettings, "save", lambda self: None)
    result = CliRunner().invoke(cli, ["infer", "script", "-s", "/s.py", "-d", "/d/raw.zarr", "--env", "cellpose4"])
    assert result.exit_code == 0, result.output
    (argv,) = [shlex.split(c) for c in commands]
    assert argv[:9] == ["/opt/pixi", "run", "--frozen", "--manifest-path", str(manifest), "-e", "cellpose4",
                        "cellmap_flow", "serve"]
    assert "env" not in json.loads(argv[argv.index("--model") + 1])


def test_a_finetuned_model_runs_where_its_base_model_does(manifest):
    model = FinetuneModelConfig(lora_adapter_path="/a", base_model={"type": "script", "script_path": "/s.py",
                                                                    "env": "cellpose4"})
    assert model.env == "cellpose4" and model.to_dict()["env"] == "cellpose4"
    assert launch.server_argv(model, "/d")[:7] == ["/opt/pixi", "run", "--frozen", "--manifest-path", str(manifest),
                                                   "-e", "cellpose4"]


class _ServedElsewhere:
    """A model in its own environment, which this process must not build."""

    name, env = "cp", "cellpose4"

    @property
    def config(self):
        raise AssertionError("built the model here")


def test_its_geometry_comes_from_its_server_and_is_never_built_here(monkeypatch):
    served = SimpleNamespace(read_shape=(8, 8, 8))
    monkeypatch.setattr(geometry_cache, "model_geometry_config", lambda name: served)
    assert geometry_cache.resolve_model_geometry("cp", _ServedElsewhere()) is served

    monkeypatch.setattr(geometry_cache, "model_geometry_config", lambda name: None)
    with pytest.raises(ModelEnvError, match=r"own environment \(cellpose4\).*running server"):
        geometry_cache.resolve_model_geometry("cp", _ServedElsewhere())


def test_building_it_where_its_packages_are_missing_names_its_env(tmp_path):
    script = tmp_path / "needs_cellpose5.py"
    script.write_text("import cellpose5_is_not_installed\n")
    model = ScriptModelConfig(script_path=str(script), name="cp")
    model.env = "cellpose4"
    with pytest.raises(ModelEnvError, match="cp runs in its own environment .cellpose4.*cellpose5_is_not_installed"):
        model.config
