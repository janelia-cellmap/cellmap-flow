"""The model registry: which model types exist, and how strings become a model config."""

import gc
import logging
import os
import shlex
import subprocess
import sys

import pytest

from cellmap_flow.models import registry
from cellmap_flow.models.models_config import FlyModelConfig, ModelConfig, ScriptModelConfig
from cellmap_flow.utils.config_utils import ConfigError

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def _forget_test_classes():
    """A class defined in a test is a model type until it is collected."""
    yield
    gc.collect()


def _built_in(types):
    return {k: v for k, v in types.items() if v.__module__ == "cellmap_flow.models.models_config"}


def test_the_new_modules_import_nothing_heavy(tmp_path):
    # describe_types() runs when the dashboard opens its model form, so it
    # must not load a model framework either.
    code = (
        "import sys\n"
        "import cellmap_flow.models.registry, cellmap_flow.serving.launch\n"
        "heavy = ['cellmap_flow.globals', 'cellmap_flow.models.models_config',\n"
        "         'torch', 'flask', 'neuroglancer', 'huggingface_hub', 'peft']\n"
        "print([m for m in heavy if m in sys.modules])\n"
        "types = cellmap_flow.models.registry.describe_types()\n"
        "assert 'BioModelConfig' in types and 'DaCapoModelConfig' in types\n"
        "frameworks = ['bioimageio', 'dacapo', 'cellmap_models', 'torch', 'huggingface_hub']\n"
        "print([m for m in frameworks if m in sys.modules])\n"
    )
    env = {**os.environ, "HOME": str(tmp_path), "PYTHONPATH": ROOT}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-2:] == ["[]", "[]"]


def test_the_built_in_types_and_how_yaml_names_them():
    assert list(_built_in(registry.model_types())) == [
        "script", "dacapo", "fly", "bioimage", "cellmap", "finetune", "huggingface",
    ]
    assert list(_built_in(registry.model_classes()))[:3] == [
        "ScriptModelConfig", "DaCapoModelConfig", "FlyModelConfig",
    ]
    for name in ("fly", "Fly", "FLY"):
        assert registry.model_type(name) is FlyModelConfig
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

    class MyThingModelConfig(ModelConfig):
        def __init__(self, weights, name=None):
            super().__init__()

    class ImpostorModelConfig(ModelConfig):
        cli_name = "script"

    with caplog.at_level(logging.WARNING, logger="cellmap_flow.models.registry"):
        types = registry.model_types()
    warned = caplog.text

    assert types["script"] is ScriptModelConfig
    assert types["labelled-fly"] is registry.model_type("labelled_fly") is LabelledFlyModelConfig
    assert types["myscript"] is MyScriptModelConfig
    assert shlex.split(MyScriptModelConfig("/s.py").command) == ["myscript", "--script-path", "/s.py"]
    assert types["mything"] is MyThingModelConfig
    assert ImpostorModelConfig not in types.values()
    assert warned.count("is not registered under it") == 1
    assert "ImpostorModelConfig is not registered under it" in warned


def test_the_cli_and_the_form_parse_tuples_differently():
    # Each keeps what its caller has always done.
    class SizesModelConfig(ModelConfig):
        def __init__(self, size: tuple, labels: list[str], count: int = 1, name=None):
            super().__init__()

    given = {"size": "8,8,8", "labels": "a, b", "count": "2"}
    assert registry.coerce_cli_args(SizesModelConfig, given) == {
        "size": (8, 8, 8), "labels": ["a", "b"], "count": "2",
    }
    assert registry.coerce_form_params(SizesModelConfig, given) == {
        "size": (8.0, 8.0, 8.0), "labels": ["a", "b"], "count": 2,
    }


def test_a_yaml_entry_may_use_aliases_and_a_single_voxel_size(caplog):
    with caplog.at_level(logging.WARNING, logger="cellmap_flow.models.registry"):
        model = registry.build_model(
            {"type": "Fly", "checkpoint": "/c.ts", "classes": ["mito"], "resolution": 8,
             "input_size": [20, 20, 20], "output_size": [10, 10, 10]},
            "m",
        )
    assert model.to_dict() == {
        "type": "fly", "checkpoint_path": "/c.ts", "channels": ["mito"],
        "input_voxel_size": [8, 8, 8], "output_voxel_size": [8, 8, 8], "name": "m",
        "input_size": [20, 20, 20], "output_size": [10, 10, 10],
    }
    assert "'output_voxel_size' not specified" in caplog.text


def test_the_dashboard_offers_and_builds_a_plugin_type():
    from flask import Flask

    from cellmap_flow.dashboard.routes.models import models_bp
    from cellmap_flow.globals import g

    class OnnxModelConfig(ModelConfig):
        cli_name = "onnx"

        def __init__(self, onnx_path: str, output_channels: int = 1, name=None):
            super().__init__()
            self.onnx_path, self.output_channels, self.name = onnx_path, output_channels, name

    app = Flask(__name__)
    app.register_blueprint(models_bp)
    client = app.test_client()
    g.models_config = []

    types = client.get("/api/model-config-types").get_json()
    assert types["OnnxModelConfig"]["display_name"] == "Onnx Model"
    assert types["OnnxModelConfig"]["parameters"]["onnx_path"]["input_type"] == "file"
    assert "ScriptModelConfig" in types

    response = client.post("/api/create-model-config", json={
        "class_name": "OnnxModelConfig",
        "params": {"onnx_path": "/m.onnx", "output_channels": "3", "name": "o"},
    })
    assert response.status_code == 200, response.get_json()
    # No to_dict() of its own: ModelConfig's default serves.
    assert response.get_json()["config_dict"] == {
        "type": "onnx", "onnx_path": "/m.onnx", "output_channels": 3, "name": "o",
    }
    assert isinstance(g.models_config[-1], OnnxModelConfig)
