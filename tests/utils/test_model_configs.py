"""Model configs: the server rebuilds each from its command, each type's
default environment, and when a script is refused as unsafe."""

import json
import shlex
import tomllib
from pathlib import Path

import pytest
from click.testing import CliRunner

from cellmap_flow.models.models_config import (
    BioModelConfig,
    CellposeModelConfig,
    DaCapoModelConfig,
    FinetuneModelConfig,
    FlyModelConfig,
    HuggingFaceModelConfig,
    OnnxModelConfig,
    ScriptModelConfig,
)
from cellmap_flow.models.configs.script import load_safe_config


# Each way a server is started: what launchers run since 0.3.0, the same
# through the program before it, and that program's per-type commands.
SERVE_FORMS = {
    "cellmap_flow serve": lambda config: ["serve", "--model", json.dumps(config.launch_entry)],
    "cellmap_flow_server --model": lambda config: ["--model", json.dumps(config.launch_entry)],
    "cellmap_flow_server <type>": lambda config: shlex.split(config.command),
}


@pytest.mark.parametrize(
    "config",
    [
        lambda: ScriptModelConfig(script_path="/groups/my models/mito model.py", name="mito", scale="s1"),
        lambda: DaCapoModelConfig(run_name="run 1", iteration=5000, name="dc", scale="s0"),
        lambda: FlyModelConfig(checkpoint_path="/ckpt/model_checkpoint_1000", channels=["mito", "er"],
                               input_voxel_size=(16, 16, 16), output_voxel_size=(16, 16, 16), name="fly",
                               input_size=(100, 100, 100), output_size=(20, 20, 20)),
        lambda: BioModelConfig(model="affable-shark", voxel_size=(8, 8, 8), input_size=(20, 256, 256), context=8,
                               slices_per_chunk=4, weight_format="onnx", name="bio"),
        lambda: FinetuneModelConfig(lora_adapter_path="/runs/my run/lora_adapter",
                                    base_model={"type": "script", "script_path": "/a b/c.py"}, name="ft",
                                    input_voxel_size=(10.48, 8, 8), output_voxel_size=(10.48, 8, 8)),
        lambda: HuggingFaceModelConfig(repo="cellmap/mito-v1", revision="abc123", name="m v1"),
        lambda: CellposeModelConfig(voxel_size=(16, 8, 8), pretrained_model="/w/my model", output="masks",
                                    slices_per_chunk=4, batch_size=12, diameter=45, flow_threshold=0.5, name="cp"),
        lambda: OnnxModelConfig(path="/ft/my model-ft1.onnx", input_voxel_size=(5.24, 4, 4), input_size=(1, 300, 300),
                                channels=["mito", "er"], slices_per_chunk=4, name="ft"),
        lambda: OnnxModelConfig(folder="setup_53_2d", repo="cellmap/onnx-web-models", variant="fp16", input_size=556),
    ],
    ids=["script", "dacapo", "fly", "bio", "finetune", "huggingface", "cellpose", "onnx", "onnx-hub"],
)
@pytest.mark.parametrize("form", list(SERVE_FORMS))
def test_the_server_rebuilds_the_same_config(config, form, monkeypatch):
    """What each form of the server does before it serves."""
    from cellmap_flow.cli import main, server_cli

    def no_download(self):
        raise AssertionError("building a launch command must not fetch metadata")

    monkeypatch.setattr(HuggingFaceModelConfig, "_load_metadata", no_download)
    served = []
    monkeypatch.setattr(server_cli, "run_server", lambda *args: served.append(args))
    config = config()
    root = main.cli if form == "cellmap_flow serve" else server_cli.cli
    result = CliRunner().invoke(root, SERVE_FORMS[form](config) + ["-d", "/data/raw.zarr"])
    assert result.exit_code == 0, result.output + repr(result.exception)
    ((rebuilt, data_path, *_),) = served
    assert type(rebuilt) is type(config) and data_path == "/data/raw.zarr"
    if isinstance(config, HuggingFaceModelConfig):  # its to_dict() adds the downloaded metadata
        assert (rebuilt.repo, rebuilt.revision, rebuilt.name) == (config.repo, config.revision, config.name)
    else:
        assert rebuilt.to_dict() == config.to_dict()


def test_a_finetune_declares_its_bases_geometry_at_the_voxel_size_it_was_trained_at(monkeypatch):
    """What the Finetune tab lists for it before its server is up: its base's
    metadata.json, read without building either model."""
    metadata = {"input_shape": [178] * 3, "output_shape": [56] * 3, "input_voxel_size": [16] * 3,
                "output_voxel_size": [16] * 3, "out_channels": 3}
    monkeypatch.setattr(HuggingFaceModelConfig, "_load_metadata", lambda self: metadata)
    base = {"type": "huggingface", "repo": "cellmap/mito"}
    assert FinetuneModelConfig(weights_path="/w.pt", base_model=base).declared_geometry() == {
        "write_shape": [896] * 3, "output_voxel_size": [16] * 3, "output_channels": 3, "channels": None}
    at_8 = FinetuneModelConfig(weights_path="/w.pt", base_model=base, input_voxel_size=8, output_voxel_size=8)
    assert at_8.declared_geometry() == {"write_shape": [448] * 3, "output_voxel_size": [8] * 3, "output_channels": 3,
                                        "channels": None}


@pytest.mark.parametrize("config, env", [
    pytest.param(lambda: FlyModelConfig(checkpoint_path="/ckpt/model_checkpoint_1000", channels=["mito"],
                                        input_voxel_size=(8, 8, 8), output_voxel_size=(8, 8, 8)),
                 "fly", id="fly-raw-checkpoint"),
    pytest.param(lambda: FlyModelConfig(checkpoint_path="/ckpt/model.pt", channels=["mito"],
                                        input_voxel_size=(8, 8, 8), output_voxel_size=(8, 8, 8)),
                 "fly", id="fly-eager-model-pt"),
    pytest.param(lambda: FlyModelConfig(checkpoint_path="/ckpt/model.ts", channels=["mito"],
                                        input_voxel_size=(8, 8, 8), output_voxel_size=(8, 8, 8)),
                 None, id="fly-torchscript"),
    pytest.param(lambda: BioModelConfig(model="affable-shark", voxel_size=(8, 8, 8)), "bioimageio", id="bio"),
    pytest.param(lambda: OnnxModelConfig(folder="setup_54"), "onnx", id="onnx"),
    # pixi.toml's dacapo environment does not import DaCapo yet.
    pytest.param(lambda: DaCapoModelConfig(run_name="r", iteration=0), None, id="dacapo"),
])
def test_a_model_types_default_environment(config, env):
    """The pixi environment a type's server runs in when its entry names none."""
    assert getattr(config(), "default_env", None) == env
    if env is not None:
        pixi = tomllib.loads((Path(__file__).resolve().parents[2] / "pixi.toml").read_text())
        assert env in pixi["environments"]


@pytest.mark.parametrize("setting, argument, allowed", [
    pytest.param("False", None, True, id="False"),  # what the error message tells users to set
    pytest.param("0", None, True, id="0"),
    pytest.param(None, None, True, id="unset"),
    pytest.param("True", None, False, id="True"),
    pytest.param("1", None, False, id="1"),
    pytest.param("1", False, True, id="an-explicit-argument-wins"),
])
def test_an_unsafe_script_is_refused_only_when_asked(tmp_path, monkeypatch, setting, argument, allowed):
    """FORCE_SAFE_CONFIG is read as a boolean, when a script is loaded.

    It was ``os.getenv("FORCE_SAFE_CONFIG", False)`` as a default argument:
    any non-empty string counted as true -- including the "False" that the
    error message tells users to set -- and the value was fixed at import.
    Every row sets it after import.
    """
    script = tmp_path / "unsafe.py"
    script.write_text("import subprocess\nvalue = 3\n")
    if setting is None:
        monkeypatch.delenv("FORCE_SAFE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("FORCE_SAFE_CONFIG", setting)
    if allowed:
        assert load_safe_config(str(script), force_safe=argument).value == 3
    else:
        with pytest.raises(ValueError, match="Unsafe script"):
            load_safe_config(str(script), force_safe=argument)