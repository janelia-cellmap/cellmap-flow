"""The model registry: which types exist, and how strings become a model config."""

import gc
import logging
import os
import subprocess
import sys

import pytest

from cellmap_flow.models import registry
from cellmap_flow.models.models_config import (
    DaCapoModelConfig,
    FinetuneModelConfig,
    FlyModelConfig,
    ModelConfig,
    ScriptModelConfig,
)
from cellmap_flow.utils.config_utils import ConfigError

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUILT_IN = ["script", "dacapo", "fly", "bioimage", "cellmap", "finetune", "huggingface"]


@pytest.fixture(autouse=True)
def _forget_test_classes():
    """Classes defined in a test stay in __subclasses__ until collected."""
    yield
    gc.collect()


def _built_in(types):
    return {k: v for k, v in types.items() if v.__module__ == "cellmap_flow.models.models_config"}


def _run(code, tmp_path):
    env = {**os.environ, "HOME": str(tmp_path), "PYTHONPATH": ROOT}
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300
    )


def test_importing_the_registry_imports_nothing_heavy(tmp_path):
    code = (
        "import sys\n"
        "import cellmap_flow.models.registry\n"
        "heavy = ['cellmap_flow.globals', 'cellmap_flow.models.models_config', 'torch',\n"
        "         'flask', 'neuroglancer', 'huggingface_hub', 'peft']\n"
        "print([m for m in heavy if m in sys.modules])\n"
    )
    result = _run(code, tmp_path)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == "[]"


def test_describing_the_types_reads_signatures_only(tmp_path):
    # The dashboard calls this for its model form; loading any framework
    # here would make opening the form cost that framework's import.
    code = (
        "import sys\n"
        "from cellmap_flow.models import registry\n"
        "types = registry.describe_types()\n"
        "assert 'BioModelConfig' in types and 'DaCapoModelConfig' in types\n"
        "frameworks = ['bioimageio', 'dacapo', 'cellmap_models', 'torch', 'huggingface_hub']\n"
        "print([m for m in frameworks if m in sys.modules])\n"
    )
    result = _run(code, tmp_path)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == "[]"


def test_the_built_in_types_in_definition_order():
    assert list(_built_in(registry.model_types())) == BUILT_IN
    classes = _built_in(registry.model_classes())
    assert list(classes) == [
        "ScriptModelConfig", "DaCapoModelConfig", "FlyModelConfig", "BioModelConfig",
        "CellMapModelConfig", "FinetuneModelConfig", "HuggingFaceModelConfig",
    ]
    assert classes["FlyModelConfig"] is FlyModelConfig


@pytest.mark.parametrize("name", ["fly", "Fly", "FLY"])
def test_a_type_is_found_as_yaml_names_it(name):
    assert registry.model_type(name) is FlyModelConfig


def test_an_unknown_type_is_a_config_error_listing_the_valid_ones():
    with pytest.raises(ConfigError, match="Valid types are: bioimage, cellmap, dacapo"):
        registry.model_type("no-such-kind")


def test_a_subclass_of_a_type_with_its_own_name_is_a_type():
    class LabelledFlyModelConfig(FlyModelConfig):
        cli_name = "labelled-fly"

    assert registry.model_types()["labelled-fly"] is LabelledFlyModelConfig
    assert registry.model_type("labelled_fly") is LabelledFlyModelConfig
    assert registry.model_classes()["LabelledFlyModelConfig"] is LabelledFlyModelConfig
    assert registry.model_types()["fly"] is FlyModelConfig


def test_a_name_that_is_taken_stays_with_the_first_class(caplog):
    class ImpostorModelConfig(ModelConfig):
        cli_name = "script"

    with caplog.at_level(logging.WARNING, logger="cellmap_flow.models.registry"):
        types = registry.model_types()

    assert types["script"] is ScriptModelConfig
    assert ImpostorModelConfig not in types.values()
    assert "ImpostorModelConfig is not registered under it" in caplog.text


def test_a_class_without_a_cli_name_is_named_after_itself():
    class MyThingModelConfig(ModelConfig):
        def __init__(self, weights, name=None):
            super().__init__()

    assert registry.cli_name_of(MyThingModelConfig) == "mything"
    assert registry.model_types()["mything"] is MyThingModelConfig
    assert registry.cli_name_of(FinetuneModelConfig) == "finetune"


def test_required_params_leave_out_name_and_scale():
    assert registry.required_params(FlyModelConfig) == [
        "checkpoint_path", "channels", "input_voxel_size", "output_voxel_size",
    ]
    assert registry.required_params(DaCapoModelConfig) == ["run_name", "iteration"]
    assert registry.required_params(FinetuneModelConfig) == []


def test_click_options_hand_out_short_flags_from_the_last_argument():
    reserved = {"-d", "-q", "-P"}
    options = registry.click_options(FlyModelConfig, reserved)
    assert [o["param_decls"] for o in options] == [
        ["-s", "--scale"],
        ["-o", "--output-size"],
        ["-i", "--input-size"],
        ["-n", "--name"],
        ["--output-voxel-size"],
        ["--input-voxel-size"],
        ["-c", "--channels"],
        ["--checkpoint-path"],
    ]
    assert reserved == {"-d", "-q", "-P"}, "the caller's set is not written to"


def _sizes_class():
    # Defined per test: a module-level subclass would be a model type for
    # the rest of the session, in every listing.
    class SizesModelConfig(ModelConfig):
        def __init__(self, size: tuple, labels: list[str], count: int = 1, name=None):
            super().__init__()

    return SizesModelConfig


def test_the_cli_and_the_form_parse_tuples_differently():
    # Each keeps what its caller has always done.
    cls = _sizes_class()
    given = {"size": "8,8,8", "labels": "a, b", "count": "2"}
    assert registry.coerce_cli_args(cls, given) == {
        "size": (8, 8, 8), "labels": ["a", "b"], "count": "2",
    }
    assert registry.coerce_form_params(cls, given) == {
        "size": (8.0, 8.0, 8.0), "labels": ["a", "b"], "count": 2,
    }


def test_build_model_applies_the_yaml_aliases_and_the_entry_name():
    model = registry.build_model(
        {"type": "fly", "checkpoint": "/c.ts", "classes": ["mito"], "resolution": 4,
         "output_resolution": [2, 2, 2]},
        "m",
    )
    assert isinstance(model, FlyModelConfig)
    assert (model.checkpoint_path, model.channels, model.name) == ("/c.ts", ["mito"], "m")
    assert (model.input_voxel_size, model.output_voxel_size) == ((4, 4, 4), (2, 2, 2))


def test_describe_types_covers_any_class_it_is_given():
    cls = _sizes_class()
    types = registry.describe_types({"SizesModelConfig": cls})
    assert types["SizesModelConfig"]["display_name"] == "Sizes Model"
    assert types["SizesModelConfig"]["parameters"]["size"] == {
        "name": "size", "required": True, "description": "Size", "type": "tuple",
        "input_type": "text",
    }


def test_the_old_names_are_the_registry():
    from cellmap_flow.models import model_registry
    from cellmap_flow.utils import cli_utils, config_utils

    assert cli_utils.get_all_model_configs() == registry.model_types()
    assert config_utils.get_model_type_mapping() == registry.model_types()
    assert model_registry.get_parameter_info(FlyModelConfig) == registry.parameter_info(FlyModelConfig)
    assert cli_utils.parse_type_annotation(list[int]) == (int, False)
    assert cli_utils.parse_comma_separated_values("1,2", int) == [1, 2]


@pytest.fixture
def dashboard():
    from flask import Flask

    from cellmap_flow.dashboard.routes.models import models_bp
    from cellmap_flow.globals import g

    app = Flask(__name__)
    app.register_blueprint(models_bp)
    g.models_config = []
    return app.test_client()


def test_the_dashboard_offers_and_builds_a_plugin_type(dashboard):
    from cellmap_flow.globals import g

    class OnnxModelConfig(ModelConfig):
        cli_name = "onnx"

        def __init__(self, onnx_path: str, output_channels: int = 1, name=None):
            super().__init__()
            self.onnx_path, self.output_channels, self.name = onnx_path, output_channels, name

        def to_dict(self):
            return {"type": "onnx", "onnx_path": self.onnx_path,
                    "output_channels": self.output_channels, "name": self.name}

    types = dashboard.get("/api/model-config-types").get_json()
    assert types["OnnxModelConfig"]["display_name"] == "Onnx Model"
    assert types["OnnxModelConfig"]["parameters"]["onnx_path"]["input_type"] == "file"
    assert "ScriptModelConfig" in types

    response = dashboard.post("/api/create-model-config", json={
        "class_name": "OnnxModelConfig",
        "params": {"onnx_path": "/m.onnx", "output_channels": "3", "name": "o"},
    })
    assert response.status_code == 200, response.get_json()
    assert response.get_json()["config_dict"] == {
        "type": "onnx", "onnx_path": "/m.onnx", "output_channels": 3, "name": "o",
    }
    assert isinstance(g.models_config[-1], OnnxModelConfig)


def test_model_config_classes_is_a_live_mapping():
    from cellmap_flow.models.model_registry import MODEL_CONFIG_CLASSES

    assert MODEL_CONFIG_CLASSES["FlyModelConfig"] is FlyModelConfig
    assert "LaterModelConfig" not in MODEL_CONFIG_CLASSES

    class LaterModelConfig(ModelConfig):
        pass

    assert MODEL_CONFIG_CLASSES["LaterModelConfig"] is LaterModelConfig
    assert dict(MODEL_CONFIG_CLASSES.items())["LaterModelConfig"] is LaterModelConfig
