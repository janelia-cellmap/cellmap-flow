"""The model registry: which model types exist, and how strings become a model config."""

import gc
import logging
import shlex

import pytest

from cellmap_flow.globals import g
from cellmap_flow.models import registry
from cellmap_flow.models.models_config import FlyModelConfig, ModelConfig, ScriptModelConfig
from cellmap_flow.config.yaml import ConfigError


@pytest.fixture(autouse=True)
def _forget_test_classes():
    """A class defined in a test is a model type until it is collected."""
    yield
    gc.collect()


def _built_in(types):
    """The types cellmap_flow defines, without the ones tests and plugins add."""
    return {k: v for k, v in types.items() if v.__module__.startswith("cellmap_flow.")}


def test_the_built_in_types_and_how_yaml_names_them():
    assert list(_built_in(registry.model_types())) == [
        "script", "dacapo", "fly", "bioimage", "cellmap", "finetune", "huggingface",
    ]
    assert list(_built_in(registry.model_classes()))[:3] == ["ScriptModelConfig", "DaCapoModelConfig", "FlyModelConfig"]
    assert registry.model_type("fly") is registry.model_type("FLY") is FlyModelConfig
    with pytest.raises(ConfigError, match="Valid types are: bioimage, cellmap, dacapo"):
        registry.model_type("no-such-kind")


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


def test_the_cli_and_the_form_parse_tuples_differently():
    # Each keeps what its caller has always done.
    class SizesModelConfig(ModelConfig):
        def __init__(self, size: tuple, labels: list[str], count: int = 1, name=None):
            super().__init__()

    given = {"size": "8,8,8", "labels": "a, b", "count": "2"}
    assert registry.coerce_cli_args(SizesModelConfig, given) == {"size": (8, 8, 8), "labels": ["a", "b"], "count": "2"}
    assert registry.coerce_form_params(SizesModelConfig, given) == {
        "size": (8.0, 8.0, 8.0), "labels": ["a", "b"], "count": 2,
    }


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


@pytest.mark.parametrize("models, message", [
    pytest.param({"m": {"script_path": "/s.py"}}, "missing 'type'", id="entry-without-a-type"),
    pytest.param({"m": {"type": "no-such-kind"}}, "unrecognized type", id="unknown-type"),
    pytest.param({"m": {"type": "dacapo", "run_name": "r"}}, "missing required parameter 'iteration'",
                 id="missing-required-parameter"),
    pytest.param({"m": "not a mapping"}, "must be a mapping", id="entry-not-a-mapping"),
    pytest.param([{"type": "script", "script_path": "/s.py"}], "name", id="list-entry-without-a-name"),
])
def test_a_bad_model_entry_is_a_config_error(models, message):
    with pytest.raises(ConfigError, match=message):
        registry.build_models(models)


def test_the_dashboard_offers_and_builds_a_plugin_type(dashboard):
    class OnnxModelConfig(ModelConfig):
        cli_name = "onnx"

        def __init__(self, onnx_path: str, output_channels: int = 1, name=None):
            super().__init__()
            self.onnx_path, self.output_channels, self.name = onnx_path, output_channels, name

    g.models_config = []
    types = dashboard.get("/api/model-config-types").get_json()
    assert types["OnnxModelConfig"]["display_name"] == "Onnx Model" and "ScriptModelConfig" in types
    assert types["OnnxModelConfig"]["parameters"]["onnx_path"]["input_type"] == "file"
    response = dashboard.post("/api/create-model-config", json={
        "class_name": "OnnxModelConfig", "params": {"onnx_path": "/m.onnx", "output_channels": "3", "name": "o"},
    })
    assert response.status_code == 200, response.get_json()
    # No to_dict() of its own: ModelConfig's default serves.
    assert response.get_json()["config_dict"] == {"type": "onnx", "onnx_path": "/m.onnx", "output_channels": 3, "name": "o"}
    assert isinstance(g.models_config[-1], OnnxModelConfig)
