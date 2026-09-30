"""The CLIs, model configs and model form as users and files see them.

Pinned as literals because nothing else pins them, and changing any of it
is a breaking release rather than a refactor:

- the console scripts and what each one runs;
- every command and option of ``cellmap_flow``, ``cellmap_flow_server``,
  ``cellmap_flow_yaml``, ``cellmap_flow_view`` and the two blockwise
  commands, down to the short flags, which come from walking each
  constructor signature in reverse, and the type listings;
- ``to_dict()`` (exported and finetuned YAMLs are written from it) and
  ``command`` (the server rebuilds the config from it) of every model type;
- what the dashboard's model form is offered, and how it parses its strings.
"""

import gc
import importlib
import sys
import tomllib
import types
from pathlib import Path

import click
import pytest
from click.testing import CliRunner
from flask import Flask

from cellmap_flow.blockwise import cli as blockwise_cli
from cellmap_flow.blockwise import multiple_cli as blockwise_multiple_cli
from cellmap_flow.cli import cli as cli_module
from cellmap_flow.cli import viewer_cli, yaml_cli
from cellmap_flow.cli.server_cli import cli as server_cli
from cellmap_flow.globals import g
from cellmap_flow.models.models_config import (
    BioModelConfig,
    DaCapoModelConfig,
    FinetuneModelConfig,
    FlyModelConfig,
    HuggingFaceModelConfig,
    ScriptModelConfig,
)

# --- the console scripts ----------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

# What each installed command runs. cellmap_flow's main() installs the
# Ctrl+C cleanup and then runs the click group; cellmap_flow_app is a plain
# function, so it takes no arguments and `cellmap_flow_app --help` starts
# the dashboard.
ENTRY_POINTS = {
    "cellmap_flow": "cellmap_flow.cli.cli:main",
    "cellmap_flow_server": "cellmap_flow.cli.server_cli:cli",
    "cellmap_flow_yaml": "cellmap_flow.cli.yaml_cli:main",
    "cellmap_flow_blockwise": "cellmap_flow.blockwise.cli:cli",
    "cellmap_flow_blockwise_multiple": "cellmap_flow.blockwise.multiple_cli:cli",
    "cellmap_flow_app": "cellmap_flow.dashboard.app:create_and_run_app",
    "cellmap_flow_view": "cellmap_flow.cli.viewer_cli:main",
}
CLICK_ENTRY_POINTS = {
    "cellmap_flow_server", "cellmap_flow_yaml", "cellmap_flow_blockwise",
    "cellmap_flow_blockwise_multiple", "cellmap_flow_view",
}


def _entry_point(target):
    module, attr = target.split(":")
    return getattr(importlib.import_module(module), attr)


def test_the_console_scripts_are_unchanged():
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts == ENTRY_POINTS
    click_commands = {name for name, target in scripts.items()
                      if isinstance(_entry_point(target), click.Command)}
    assert click_commands == CLICK_ENTRY_POINTS
    assert _entry_point(scripts["cellmap_flow"]) is cli_module.main


# --- the command-line surface ---------------------------------------------------

# (name, opts, secondary_opts, type name, required, default, is_flag, help),
# and nargs after them when it is not 1.
NAME = ('name', ('-n', '--name'), (), 'text', False, None, False, 'Parameter: name (optional)')
SCALE = ('scale', ('-s', '--scale'), (), 'text', False, None, False, 'Parameter: scale (optional)')
DATA_PATH = ('data_path', ('-d', '--data-path'), (), 'text', True, None, False, 'Path to the dataset')
LOG_LEVEL = [('log_level', ('--log-level',), (), 'choice', False, 'INFO', False, 'Set the logging level')]

# Each model type's own options, the same in both CLIs.
MODEL_OPTIONS = {
    'script': [
        ('script_path', ('--script-path',), (), 'text', True, None, False, 'Parameter: script_path'),
        NAME, SCALE,
    ],
    'dacapo': [
        ('run_name', ('-r', '--run-name'), (), 'text', True, None, False, 'Parameter: run_name'),
        ('iteration', ('-i', '--iteration'), (), 'integer', True, None, False, 'Parameter: iteration'),
        NAME, SCALE,
    ],
    'fly': [
        ('checkpoint_path', ('--checkpoint-path',), (), 'text', True, None, False, 'Parameter: checkpoint_path'),
        ('channels', ('-c', '--channels'), (), 'text', True, None, False, 'Parameter: channels [comma-separated values]'),
        ('input_voxel_size', ('--input-voxel-size',), (), 'text', True, None, False, 'Parameter: input_voxel_size'),
        ('output_voxel_size', ('--output-voxel-size',), (), 'text', True, None, False, 'Parameter: output_voxel_size'),
        NAME,
        ('input_size', ('-i', '--input-size'), (), 'text', False, None, False, 'Parameter: input_size (optional)'),
        ('output_size', ('-o', '--output-size'), (), 'text', False, None, False, 'Parameter: output_size (optional)'),
        SCALE,
    ],
    'bioimage': [
        ('model_name', ('-m', '--model-name'), (), 'text', True, None, False, 'Parameter: model_name'),
        ('voxel_size', ('-v', '--voxel-size'), (), 'text', True, None, False, 'Parameter: voxel_size'),
        ('edge_length_to_process', ('-e', '--edge-length-to-process'), (), 'text', False, None, False,
         'Parameter: edge_length_to_process (optional)'),
        NAME, SCALE,
    ],
    'cellmap': [
        ('folder_path', ('-f', '--folder-path'), (), 'text', True, None, False, 'Parameter: folder_path'),
        NAME, SCALE,
    ],
    'finetune': [
        ('lora_adapter_path', ('-l', '--lora-adapter-path'), (), 'text', False, None, False,
         'Parameter: lora_adapter_path (optional)'),
        ('base_model', ('-b', '--base-model'), (), 'text', False, None, False, 'Parameter: base_model (optional)'),
        NAME, SCALE,
        ('weights_path', ('-w', '--weights-path'), (), 'text', False, None, False, 'Parameter: weights_path (optional)'),
    ],
    'huggingface': [
        ('repo', ('--repo',), (), 'text', True, None, False, 'Parameter: repo'),
        ('revision', ('-r', '--revision'), (), 'text', False, None, False, 'Parameter: revision (optional)'),
        NAME, SCALE,
    ],
}

SERVER_CHECK = ('server_check', ('--server-check',), (), 'boolean', False, False, True,
                'Run server check instead of full inference')
PROJECT = ('project', ('-P', '--project'), (), 'text', False, None, False, 'Project/chargeback group for billing')
QUEUE = ('queue', ('-q', '--queue'), (), 'text', False, None, False,
         'Queue for job submission (default: the saved queue)')

# "" is the group's own options.
CELLMAP_FLOW = {
    '': LOG_LEVEL,
    'list-models': [],
    'register': [
        ('filepath', ('filepath',), (), 'path', True, None, False, None),
        ('force', ('--force',), (), 'boolean', False, False, True, 'Overwrite existing plugin with the same name.'),
    ],
    'unregister': [('name', ('name',), (), 'text', True, None, False, None)],
    'list-plugins': [],
    'run': [
        ('model_type', ('-m', '--model-type'), (), 'text', True, None, False, 'Model type (e.g., dacapo, script, cellmap)'),
        DATA_PATH, QUEUE, PROJECT,
        ('config', ('-c', '--config'), (), 'text', False, None, False, 'Model configuration as key=value pairs'),
        SERVER_CHECK,
    ],
    **{t: [*options, SERVER_CHECK, PROJECT, QUEUE, DATA_PATH] for t, options in MODEL_OPTIONS.items()},
}

CELLMAP_FLOW_SERVER = {
    '': LOG_LEVEL,
    'list-models': [],
    **{
        t: [
            *options,
            ('keyfile', ('--keyfile',), (), 'text', False, None, False, 'Path to SSL private key file'),
            ('certfile', ('--certfile',), (), 'text', False, None, False, 'Path to SSL certificate file'),
            ('port', ('-p', '--port'), (), 'integer', False, 0, False, 'Port to listen on'),
            ('debug', ('--debug',), (), 'boolean', False, False, True, 'Run in debug mode'),
            DATA_PATH,
        ]
        for t, options in MODEL_OPTIONS.items()
    },
}

CELLMAP_FLOW_YAML = {
    '': [
        ('config_path', ('config_path',), (), 'path', False, None, False, None),
        *LOG_LEVEL,
        ('list_types', ('--list-types',), (), 'boolean', False, False, True, 'List available model types and exit'),
        ('validate_only', ('--validate-only',), (), 'boolean', False, False, True,
         'Validate YAML configuration without running jobs'),
    ],
}

CELLMAP_FLOW_VIEW = {
    '': [
        ('dataset', ('-d', '--dataset'), (), 'text', True, None, False, 'Path to the dataset (zarr or n5)'),
        ('project', ('-P', '--project'), (), 'text', False, None, False,
         'Charge group (LSF project) billed for the models launched from the dashboard'),
        *LOG_LEVEL,
    ],
}

CELLMAP_FLOW_BLOCKWISE = {
    '': [
        ('yaml_config', ('yaml_config',), (), 'path', True, None, False, None),
        ('client', ('-c', '--client'), (), 'boolean', False, False, True, 'Run as client if this flag is set.'),
        ('log_level', ('--log-level',), (), 'choice', False, 'INFO', False, None),
    ],
}

CELLMAP_FLOW_BLOCKWISE_MULTIPLE = {
    '': [('yaml_configs', ('yaml_configs',), (), 'path', True, None, False, None, 'nargs=-1')],
}


def _param(p):
    # to_info_dict() reports an unset default as None on every click 8.x,
    # where p.default is a sentinel on some versions.
    row = (
        p.name, tuple(p.opts), tuple(p.secondary_opts), p.type.name, p.required,
        p.to_info_dict().get("default"), bool(getattr(p, "is_flag", False)),
        getattr(p, "help", None),
    )
    return row if p.nargs == 1 else (*row, f"nargs={p.nargs}")


@pytest.mark.parametrize(
    "command, expected",
    [
        (cli_module.cli, CELLMAP_FLOW),
        (server_cli, CELLMAP_FLOW_SERVER),
        (yaml_cli.main, CELLMAP_FLOW_YAML),
        (viewer_cli.main, CELLMAP_FLOW_VIEW),
        (blockwise_cli.cli, CELLMAP_FLOW_BLOCKWISE),
        (blockwise_multiple_cli.cli, CELLMAP_FLOW_BLOCKWISE_MULTIPLE),
    ],
    ids=["cellmap_flow", "cellmap_flow_server", "cellmap_flow_yaml", "cellmap_flow_view",
         "cellmap_flow_blockwise", "cellmap_flow_blockwise_multiple"],
)
def test_commands_and_options_are_unchanged(command, expected):
    surface = {"": [_param(p) for p in command.params]}
    for name, sub in getattr(command, "commands", {}).items():
        surface[name] = [_param(p) for p in sub.params]
    assert list(surface) == list(expected), "the commands, in registration order"
    assert surface == expected


_LISTED = [
    ("bioimage", "BioModelConfig", "model_name, voxel_size, edge_length_to_process, name, scale", "model_name, voxel_size"),
    ("cellmap", "CellMapModelConfig", "folder_path, name, scale", "folder_path"),
    ("dacapo", "DaCapoModelConfig", "run_name, iteration, name, scale", "run_name, iteration"),
    ("finetune", "FinetuneModelConfig", "lora_adapter_path, base_model, name, scale, weights_path", ""),
    ("fly", "FlyModelConfig",
     "checkpoint_path, channels, input_voxel_size, output_voxel_size, name, input_size, output_size, scale",
     "checkpoint_path, channels, input_voxel_size, output_voxel_size"),
    ("huggingface", "HuggingFaceModelConfig", "repo, revision, name, scale", "repo"),
    ("script", "ScriptModelConfig", "script_path, name, scale", "script_path"),
]


def _listing(prog):
    if prog == "cellmap_flow_yaml":
        text = "Available model types:\n\n"
        for cli_name, cls_name, _, required in _LISTED:
            text += f"  {cli_name:20s} - {cls_name}\n"
            text += f"                       Required: {required}\n" if required else ""
        return text + "\nSee example YAML configuration in the docstring with --help\n"
    text = "Available model configurations:\n\n"
    for cli_name, cls_name, params, _ in _LISTED:
        text += f"  {cli_name:20s} - {cls_name}\n                       Parameters: {params}\n"
    return text + f"\nUse '{prog} <model-name> --help' for detailed parameter information.\n"


@pytest.mark.parametrize(
    "command, argv, prog",
    [
        (cli_module.cli, ["list-models"], "cellmap_flow"),
        (server_cli, ["list-models"], "cellmap_flow_server"),
        (yaml_cli.main, ["--list-types"], "cellmap_flow_yaml"),
    ],
)
def test_the_type_listings_are_unchanged(command, argv, prog):
    gc.collect()  # a model class another test defined must not be listed
    result = CliRunner().invoke(command, argv)
    assert result.exit_code == 0, result.output
    assert result.output == _listing(prog)


# --- to_dict() and command of every model type -----------------------------------

@pytest.fixture
def fake_cellmap_models(monkeypatch):
    """CellMapModelConfig opens the model folder in its constructor."""

    class CellmapModel:
        def __init__(self, folder_path):
            self.folder_path = folder_path

    leaf = types.ModuleType("cellmap_models.model_export.cellmap_model")
    leaf.CellmapModel = CellmapModel
    for name in ("cellmap_models", "cellmap_models.model_export"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, leaf.__name__, leaf)


@pytest.fixture
def hf_metadata(monkeypatch):
    """to_dict() adds the repo's metadata.json; command must not fetch it."""
    fetched = []
    metadata = {
        "description": "Mito, v1", "unrelated": 1, "channels_names": ["mito"],
        "input_voxel_size": [8, 8, 8], "output_voxel_size": [8, 8, 8], "model_type": "unet",
    }
    monkeypatch.setattr(
        HuggingFaceModelConfig, "_load_metadata", lambda self: fetched.append(self.repo) or metadata
    )
    return fetched


def _cellmap(**kwargs):
    from cellmap_flow.models.models_config import CellMapModelConfig

    return CellMapModelConfig(**kwargs)


FLY_ENTRY = {
    "type": "fly", "checkpoint_path": "/ckpt/fly run/model.ts", "channels": ["mito", "er"],
    "input_voxel_size": [16, 16, 16], "output_voxel_size": [8, 8, 8],
    "input_size": [100, 100, 100], "output_size": [20, 20, 20], "name": "fly base",
}
INNER_FINETUNE = {
    "type": "finetune", "lora_adapter_path": "/runs/r1/lora_adapter", "weights_path": None,
    "base_model": {"type": "script", "script_path": "/s.py"}, "name": "inner",
}

# key: (config, to_dict(), command)
CONFIGS = {
    "script": (
        lambda: ScriptModelConfig(script_path="/groups/my models/mito.py", name="mito", scale="s1"),
        {'type': 'script', 'script_path': '/groups/my models/mito.py', 'name': 'mito', 'scale': 's1'},
        "script --script-path '/groups/my models/mito.py' --name mito --scale s1",
    ),
    "dacapo": (
        lambda: DaCapoModelConfig(run_name="run 1", iteration=5000, name="dc", scale="s0"),
        {'type': 'dacapo', 'run_name': 'run 1', 'iteration': 5000, 'name': 'dc', 'scale': 's0'},
        "dacapo --run-name 'run 1' --iteration 5000 --name dc --scale s0",
    ),
    "fly": (
        lambda: FlyModelConfig(
            checkpoint_path="/ckpt/model_checkpoint_1000", channels=["mito", "er"],
            input_voxel_size=(16, 16, 16), output_voxel_size=(8, 8, 8), name="fly",
            input_size=(100, 100, 100), output_size=(20, 20, 20), scale="s1",
        ),
        {'type': 'fly', 'checkpoint_path': '/ckpt/model_checkpoint_1000', 'channels': ['mito', 'er'],
         'input_voxel_size': [16, 16, 16], 'output_voxel_size': [8, 8, 8], 'name': 'fly',
         'input_size': [100, 100, 100], 'output_size': [20, 20, 20], 'scale': 's1'},
        "fly --checkpoint-path /ckpt/model_checkpoint_1000 --channels mito,er --input-voxel-size 16,16,16"
        " --output-voxel-size 8,8,8 --name fly --input-size 100,100,100 --output-size 20,20,20 --scale s1",
    ),
    # As the server CLI passes them: strings, and no sizes.
    "fly_from_cli": (
        lambda: FlyModelConfig(checkpoint_path="/c.ts", channels="mito, er",
                               input_voxel_size="8,8,8", output_voxel_size="8,8,8"),
        {'type': 'fly', 'checkpoint_path': '/c.ts', 'channels': ['mito', 'er'],
         'input_voxel_size': [8, 8, 8], 'output_voxel_size': [8, 8, 8],
         'input_size': [178, 178, 178], 'output_size': [56, 56, 56]},
        "fly --checkpoint-path /c.ts --channels mito,er --input-voxel-size 8,8,8 --output-voxel-size 8,8,8"
        " --input-size 178,178,178 --output-size 56,56,56",
    ),
    "bio": (
        lambda: BioModelConfig(model_name="affable-shark", voxel_size=(8, 8, 8),
                               edge_length_to_process=64, name="bio", scale="s0"),
        {'type': 'bioimage', 'model_name': 'affable-shark', 'voxel_size': [8, 8, 8], 'name': 'bio',
         'scale': 's0', 'edge_length_to_process': 64},
        "bioimage --model-name affable-shark --voxel-size 8,8,8 --edge-length-to-process 64 --name bio --scale s0",
    ),
    "cellmap": (
        lambda: _cellmap(folder_path="/models/my mito/"),
        {'type': 'cellmap', 'folder_path': '/models/my mito/', 'name': 'my mito'},
        "cellmap --folder-path '/models/my mito/' --name 'my mito'",
    ),
    "hf": (
        lambda: HuggingFaceModelConfig(repo="cellmap/mito-v1", revision="abc123", name="m v1", scale="s0"),
        {'type': 'huggingface', 'repo': 'cellmap/mito-v1', 'revision': 'abc123', 'name': 'm v1',
         'scale': 's0', 'channels_names': ['mito'], 'input_voxel_size': [8, 8, 8],
         'output_voxel_size': [8, 8, 8], 'model_type': 'unet', 'description': 'Mito, v1'},
        "huggingface --repo cellmap/mito-v1 --revision abc123 --name 'm v1' --scale s0",
    ),
    "hf_bare": (
        lambda: HuggingFaceModelConfig(repo="cellmap/mito-v1"),
        {'type': 'huggingface', 'repo': 'cellmap/mito-v1', 'name': 'mito-v1', 'channels_names': ['mito'],
         'input_voxel_size': [8, 8, 8], 'output_voxel_size': [8, 8, 8], 'model_type': 'unet',
         'description': 'Mito, v1'},
        "huggingface --repo cellmap/mito-v1 --name mito-v1",
    ),
    "finetune": (
        lambda: FinetuneModelConfig(lora_adapter_path="/runs/my run/lora_adapter", base_model=FLY_ENTRY,
                                    name="ft", scale="s1"),
        {'type': 'finetune', 'lora_adapter_path': '/runs/my run/lora_adapter', 'weights_path': None,
         'base_model': FLY_ENTRY, 'name': 'ft', 'scale': 's1', 'channels': ['mito', 'er'],
         'checkpoint_path': '/ckpt/fly run/model.ts', 'input_voxel_size': [16, 16, 16],
         'output_voxel_size': [8, 8, 8], 'input_size': [100, 100, 100], 'output_size': [20, 20, 20],
         'base_type': 'fly'},
        "finetune --lora-adapter-path '/runs/my run/lora_adapter' --base-model"
        " eyJ0eXBlIjoiZmx5IiwiY2hlY2twb2ludF9wYXRoIjoiL2NrcHQvZmx5IHJ1bi9tb2RlbC50cyIsImNoYW5uZWxzIjpbIm1pdG8iLCJlciJdLCJpbnB1dF92b3hlbF9zaXplIjpbMTYsMTYsMTZdLCJvdXRwdXRfdm94ZWxfc2l6ZSI6WzgsOCw4XSwiaW5wdXRfc2l6ZSI6WzEwMCwxMDAsMTAwXSwib3V0cHV0X3NpemUiOlsyMCwyMCwyMF0sIm5hbWUiOiJmbHkgYmFzZSJ9"
        " --name ft --scale s1",
    ),
    "finetune_of_finetune": (
        lambda: FinetuneModelConfig(weights_path="/runs/r2/full_finetune/model_state_dict.pt",
                                    base_model=INNER_FINETUNE),
        {'type': 'finetune', 'lora_adapter_path': None,
         'weights_path': '/runs/r2/full_finetune/model_state_dict.pt',
         'base_model': INNER_FINETUNE, 'base_type': 'finetune'},
        "finetune --base-model"
        " eyJ0eXBlIjoiZmluZXR1bmUiLCJsb3JhX2FkYXB0ZXJfcGF0aCI6Ii9ydW5zL3IxL2xvcmFfYWRhcHRlciIsIndlaWdodHNfcGF0aCI6bnVsbCwiYmFzZV9tb2RlbCI6eyJ0eXBlIjoic2NyaXB0Iiwic2NyaXB0X3BhdGgiOiIvcy5weSJ9LCJuYW1lIjoiaW5uZXIifQ"
        " --weights-path /runs/r2/full_finetune/model_state_dict.pt",
    ),
}


@pytest.mark.parametrize("key", list(CONFIGS))
def test_to_dict_and_command_are_unchanged(key, fake_cellmap_models, hf_metadata):
    build, to_dict, command = CONFIGS[key]
    config = build()
    assert config.command == command
    assert hf_metadata == [], "building a launch command must not download metadata.json"
    result = config.to_dict()
    assert result == to_dict
    assert list(result) == list(to_dict), "key order is what the exported YAML shows"


# --- the dashboard's model form ----------------------------------------------------

def _param_info(name, type_, required, input_type, default=...):
    info = {"name": name, "required": required, "description": name.replace("_", " ").title(),
            "type": type_, "input_type": input_type}
    if default is not ...:
        info["default"] = default
    return info


def _type_info(class_name, display_name, *params):
    return {
        "display_name": display_name,
        "description": f"Create a {display_name} model configuration",
        "class_name": class_name,
        "parameters": {p["name"]: p for p in params},
    }


_NAME = _param_info("name", "string", False, "text", None)
_STR_NAME = _param_info("name", "str", False, "text", None)
_SCALE = _param_info("scale", "string", False, "text", None)

MODEL_CONFIG_TYPES = {
    "BioModelConfig": _type_info(
        "BioModelConfig", "Bio Model",
        _param_info("model_name", "str", True, "text"),
        _param_info("voxel_size", "string", True, "textarea"),
        _param_info("edge_length_to_process", "string", False, "number", None),
        _NAME, _SCALE,
    ),
    "CellMapModelConfig": _type_info(
        "CellMapModelConfig", "Cell Map Model", _param_info("folder_path", "string", True, "file"), _NAME, _SCALE,
    ),
    "DaCapoModelConfig": _type_info(
        "DaCapoModelConfig", "Da Capo Model",
        _param_info("run_name", "str", True, "text"), _param_info("iteration", "int", True, "number"),
        _NAME, _SCALE,
    ),
    "FinetuneModelConfig": _type_info(
        "FinetuneModelConfig", "Finetune Model",
        _param_info("lora_adapter_path", "str", False, "file", None),
        _param_info("base_model", "dict", False, "textarea", None),
        _STR_NAME, _SCALE,
        _param_info("weights_path", "str", False, "file", None),
    ),
    "FlyModelConfig": _type_info(
        "FlyModelConfig", "Fly Model",
        _param_info("checkpoint_path", "str", True, "file"),
        _param_info("channels", "list", True, "textarea"),
        _param_info("input_voxel_size", "tuple", True, "textarea"),
        _param_info("output_voxel_size", "tuple", True, "textarea"),
        _STR_NAME,
        _param_info("input_size", "string", False, "number", None),
        _param_info("output_size", "string", False, "number", None),
        _SCALE,
    ),
    "HuggingFaceModelConfig": _type_info(
        "HuggingFaceModelConfig", "Hugging Face Model",
        _param_info("repo", "string", True, "text"), _param_info("revision", "string", False, "text", None),
        _NAME, _SCALE,
    ),
    "ScriptModelConfig": _type_info(
        "ScriptModelConfig", "Script Model", _param_info("script_path", "string", True, "file"), _NAME, _SCALE,
    ),
}


@pytest.fixture
def dashboard():
    from cellmap_flow.dashboard.routes.models import models_bp

    app = Flask(__name__)
    app.register_blueprint(models_bp)
    g.models_config = []
    return app.test_client()


def test_the_model_form_is_offered_the_same_types(dashboard):
    gc.collect()
    response = dashboard.get("/api/model-config-types")
    assert response.status_code == 200
    assert response.get_json() == MODEL_CONFIG_TYPES


@pytest.mark.parametrize(
    "class_name, params, status, expected",
    [
        # Form tuples become floats (the CLI's become ints), and a list of
        # strings is split on commas when it is not JSON.
        ("FlyModelConfig",
         {"checkpoint_path": "/c.ts", "channels": "mito, er", "input_voxel_size": "16,16,16",
          "output_voxel_size": "[8, 8, 8]", "input_size": "", "name": "fly"},
         200,
         {"type": "fly", "checkpoint_path": "/c.ts", "channels": ["mito", "er"],
          "input_voxel_size": [16.0, 16.0, 16.0], "output_voxel_size": [8, 8, 8], "name": "fly",
          "input_size": [178, 178, 178], "output_size": [56, 56, 56]}),
        ("NoSuchModelConfig", {}, 400, "Unknown model config class: NoSuchModelConfig"),
        ("DaCapoModelConfig", {"run_name": "r", "iteration": ""}, 400, "Required parameter 'iteration' is missing"),
        ("DaCapoModelConfig", {"run_name": "r"}, 400, "Failed to instantiate DaCapoModelConfig: "),
    ],
    ids=["parsed", "unknown-class", "empty-required", "constructor-error"],
)
def test_the_model_form_parses_and_rejects_the_same_way(dashboard, class_name, params, status, expected):
    response = dashboard.post("/api/create-model-config", json={"class_name": class_name, "params": params})
    assert response.status_code == status, response.get_json()
    if status == 200:
        assert response.get_json()["config_dict"] == expected
        assert [type(m).__name__ for m in g.models_config] == [class_name]
    else:
        assert response.get_json()["error"].startswith(expected)
