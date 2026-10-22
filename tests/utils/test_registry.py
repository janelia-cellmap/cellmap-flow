"""The model registry: which model types exist, and how a model entry becomes a model config."""

import gc
import logging
import shlex

import pytest
from click.testing import CliRunner

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.models import registry
from cellmap_flow.models.models_config import FlyModelConfig, ModelConfig, ScriptModelConfig


@pytest.fixture(autouse=True)
def _forget_test_classes():
    """A class defined in a test is a model type until it is collected."""
    yield
    gc.collect()


def test_plugin_types_get_their_own_names_and_never_replace_a_built_in(caplog):
    class LabelledFlyModelConfig(FlyModelConfig):
        cli_name = "labelled-fly"

    # Inherits cli_name = "script". Read through inheritance, that would
    # replace the script type, or lose to it and leave this class with no
    # type while its command asked the server for a plain ScriptModelConfig.
    class MyScriptModelConfig(ScriptModelConfig):
        pass

    class ImpostorModelConfig(ModelConfig):
        cli_name = "script"

    with caplog.at_level(logging.WARNING, logger="cellmap_flow.models.registry"):
        types = registry.model_types()

    assert types["script"] is ScriptModelConfig and ImpostorModelConfig not in types.values()
    assert [r.levelno for r in caplog.records if "ImpostorModelConfig" in r.getMessage()] == [logging.WARNING]
    assert types["labelled-fly"] is registry.model_type("labelled_fly") is LabelledFlyModelConfig
    assert types["myscript"] is MyScriptModelConfig
    assert shlex.split(MyScriptModelConfig("/s.py").command) == ["myscript", "--script-path", "/s.py"]


def test_a_yaml_entry_may_use_aliases_and_a_single_voxel_size():
    model = registry.build_model(
        {"type": "Fly", "checkpoint": "/c.ts", "classes": ["mito"], "resolution": 8,
         "input_size": [20, 20, 20], "output_size": [10, 10, 10]},
        "m",
    )
    assert model.to_dict() == {
        "type": "fly", "checkpoint_path": "/c.ts", "channels": ["mito"], "input_voxel_size": [8, 8, 8],
        "output_voxel_size": [8, 8, 8], "name": "m", "input_size": [20, 20, 20], "output_size": [10, 10, 10],
    }


def test_the_dashboard_offers_and_builds_a_plugin_type(dashboard):
    # Not an onnx type, which is a built-in one now: a plugin's own name.
    class TfliteModelConfig(ModelConfig):
        cli_name = "tflite"

        def __init__(self, tflite_path: str, output_channels: int = 1, name=None):
            super().__init__()
            self.tflite_path, self.output_channels, self.name = tflite_path, output_channels, name

    get_session().models_config = []
    types = dashboard.get("/api/model-config-types").get_json()
    assert types["tflite"]["display_name"] == "Tflite Model" and "script" in types
    assert types["tflite"]["parameters"]["tflite_path"]["input_type"] == "file"
    response = dashboard.post("/api/create-model-config", json={
        "type": "tflite", "params": {"tflite_path": "/m.tflite", "output_channels": "3", "name": "o"},
    })
    assert response.status_code == 200, response.get_json()
    # No to_dict() of its own: ModelConfig's default serves.
    assert response.get_json()["config_dict"] == {"type": "tflite", "tflite_path": "/m.tflite", "output_channels": 3,
                                                  "name": "o"}
    assert isinstance(get_session().models_config[-1], TfliteModelConfig)


def test_one_entry_gives_one_config_from_the_cli_a_yaml_and_the_form(dashboard, monkeypatch):
    """The same model described as `infer` options (strings), as a YAML entry
    (numbers and lists) and as the form posts it (strings, blanks for the
    rest) is the same config: ints from one and floats from another once
    put crops on different grids."""
    from cellmap_flow.cli import infer
    from cellmap_flow.cli.main import cli
    from cellmap_flow.dashboard.services import startup
    from cellmap_flow.jobs.settings import LauncherSettings

    built = []
    monkeypatch.setattr(infer, "launch", lambda model, *args, **kwargs: built.append(model.to_dict()))
    monkeypatch.setattr(startup, "generate_neuroglancer_url", lambda path: None)
    monkeypatch.setattr(LauncherSettings, "save", lambda self: None)
    get_session().models_config = []

    result = CliRunner().invoke(cli, ["infer", "fly", "--checkpoint-path", "/c.ts", "--channels", "mito,er",
                                      "--input-voxel-size", "5.24,4,4", "--input-size", "20", "--output-size", "10",
                                      "--name", "m", "-d", "/data"])
    assert result.exit_code == 0, result.output + repr(result.exception)
    from_yaml = registry.build_model({"type": "fly", "checkpoint": "/c.ts", "classes": ["mito", "er"],
                                      "resolution": [5.24, 4, 4], "input_size": 20, "output_size": 10}, "m")
    posted = dashboard.post("/api/create-model-config", json={"type": "fly", "params": {
        "checkpoint_path": "/c.ts", "channels": "mito, er", "input_voxel_size": "5.24, 4, 4",
        "output_voxel_size": "", "input_size": "20", "output_size": "10", "name": "m", "scale": "", "env": ""}})
    assert posted.status_code == 200, posted.get_json()

    expected = {"type": "fly", "checkpoint_path": "/c.ts", "channels": ["mito", "er"], "input_voxel_size": [5.24, 4, 4],
                "output_voxel_size": [5.24, 4, 4], "name": "m", "input_size": [20, 20, 20], "output_size": [10, 10, 10]}
    assert built == [expected] and from_yaml.to_dict() == expected and posted.get_json()["config_dict"] == expected
