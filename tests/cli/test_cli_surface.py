"""What the model CLIs, model configs and launchers expose today, as literals.

The model registry replaces the code that builds all of this (cli_utils,
config_utils, model_registry's parsing and the launch strings). None of it
may change while that happens:

- every command and option of ``cellmap_flow``, ``cellmap_flow_server`` and
  ``cellmap_flow_yaml``, down to the short flags, which come from walking the
  constructor signature in reverse (changing any of it is a breaking release,
  not a refactor);
- ``to_dict()`` (exported and finetuned YAMLs are written from it) and
  ``command`` (the server rebuilds the config from it) of every model type;
- the command lines each launcher hands to ``start_hosts``;
- the model types the dashboard's model form offers, and how it and the CLI
  turn their strings into constructor arguments.
"""

import gc
import logging
import sys
import types

import pytest
from click.testing import CliRunner
from flask import Flask

from cellmap_flow.cli import cli as cli_module
from cellmap_flow.cli import yaml_cli
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
from cellmap_flow.utils import bsub_utils
from cellmap_flow.utils.bsub_utils import JobStartError


# (name, opts, secondary_opts, type name, required, default, is_flag, help);
# "" is the group's own options.
CELLMAP_FLOW = {
    '': [
        ('log_level', ('--log-level',), (), 'choice', False, 'INFO', False, 'Set the logging level'),
    ],
    'list-models': [],
    'register': [
        ('filepath', ('filepath',), (), 'path', True, None, False, None),
        ('force', ('--force',), (), 'boolean', False, False, True, 'Overwrite existing plugin with the same name.'),
    ],
    'unregister': [
        ('name', ('name',), (), 'text', True, None, False, None),
    ],
    'list-plugins': [],
    'run': [
        ('model_type', ('-m', '--model-type'), (), 'text', True, None, False, 'Model type (e.g., dacapo, script, cellmap)'),
        ('data_path', ('-d', '--data-path'), (), 'text', True, None, False, 'Path to the dataset'),
        ('queue', ('-q', '--queue'), (), 'text', False, None, False, 'Queue for job submission (default: the saved queue)'),
        ('project', ('-P', '--project'), (), 'text', False, None, False, 'Project/chargeback group for billing'),
        ('config', ('-c', '--config'), (), 'text', False, None, False, 'Model configuration as key=value pairs'),
        ('server_check', ('--server-check',), (), 'boolean', False, False, True, 'Run server check instead of full inference'),
    ],
    'script': [
        ('script_path', ('--script-path',), (), 'text', True, None, False, 'Parameter: script_path'),
        ('name', ('-n', '--name'), (), 'text', False, None, False, 'Parameter: name (optional)'),
        ('scale', ('-s', '--scale'), (), 'text', False, None, False, 'Parameter: scale (optional)'),
        ('server_check', ('--server-check',), (), 'boolean', False, False, True, 'Run server check instead of full inference'),
        ('project', ('-P', '--project'), (), 'text', False, None, False, 'Project/chargeback group for billing'),
        ('queue', ('-q', '--queue'), (), 'text', False, None, False, 'Queue for job submission (default: the saved queue)'),
        ('data_path', ('-d', '--data-path'), (), 'text', True, None, False, 'Path to the dataset'),
    ],
    'dacapo': [
        ('run_name', ('-r', '--run-name'), (), 'text', True, None, False, 'Parameter: run_name'),
        ('iteration', ('-i', '--iteration'), (), 'integer', True, None, False, 'Parameter: iteration'),
        ('name', ('-n', '--name'), (), 'text', False, None, False, 'Parameter: name (optional)'),
        ('scale', ('-s', '--scale'), (), 'text', False, None, False, 'Parameter: scale (optional)'),
        ('server_check', ('--server-check',), (), 'boolean', False, False, True, 'Run server check instead of full inference'),
        ('project', ('-P', '--project'), (), 'text', False, None, False, 'Project/chargeback group for billing'),
        ('queue', ('-q', '--queue'), (), 'text', False, None, False, 'Queue for job submission (default: the saved queue)'),
        ('data_path', ('-d', '--data-path'), (), 'text', True, None, False, 'Path to the dataset'),
    ],
    'fly': [
        ('checkpoint_path', ('--checkpoint-path',), (), 'text', True, None, False, 'Parameter: checkpoint_path'),
        ('channels', ('-c', '--channels'), (), 'text', True, None, False, 'Parameter: channels [comma-separated values]'),
        ('input_voxel_size', ('--input-voxel-size',), (), 'text', True, None, False, 'Parameter: input_voxel_size'),
        ('output_voxel_size', ('--output-voxel-size',), (), 'text', True, None, False, 'Parameter: output_voxel_size'),
        ('name', ('-n', '--name'), (), 'text', False, None, False, 'Parameter: name (optional)'),
        ('input_size', ('-i', '--input-size'), (), 'text', False, None, False, 'Parameter: input_size (optional)'),
        ('output_size', ('-o', '--output-size'), (), 'text', False, None, False, 'Parameter: output_size (optional)'),
        ('scale', ('-s', '--scale'), (), 'text', False, None, False, 'Parameter: scale (optional)'),
        ('server_check', ('--server-check',), (), 'boolean', False, False, True, 'Run server check instead of full inference'),
        ('project', ('-P', '--project'), (), 'text', False, None, False, 'Project/chargeback group for billing'),
        ('queue', ('-q', '--queue'), (), 'text', False, None, False, 'Queue for job submission (default: the saved queue)'),
        ('data_path', ('-d', '--data-path'), (), 'text', True, None, False, 'Path to the dataset'),
    ],
    'bioimage': [
        ('model_name', ('-m', '--model-name'), (), 'text', True, None, False, 'Parameter: model_name'),
        ('voxel_size', ('-v', '--voxel-size'), (), 'text', True, None, False, 'Parameter: voxel_size'),
        ('edge_length_to_process', ('-e', '--edge-length-to-process'), (), 'text', False, None, False, 'Parameter: edge_length_to_process (optional)'),
        ('name', ('-n', '--name'), (), 'text', False, None, False, 'Parameter: name (optional)'),
        ('scale', ('-s', '--scale'), (), 'text', False, None, False, 'Parameter: scale (optional)'),
        ('server_check', ('--server-check',), (), 'boolean', False, False, True, 'Run server check instead of full inference'),
        ('project', ('-P', '--project'), (), 'text', False, None, False, 'Project/chargeback group for billing'),
        ('queue', ('-q', '--queue'), (), 'text', False, None, False, 'Queue for job submission (default: the saved queue)'),
        ('data_path', ('-d', '--data-path'), (), 'text', True, None, False, 'Path to the dataset'),
    ],
    'cellmap': [
        ('folder_path', ('-f', '--folder-path'), (), 'text', True, None, False, 'Parameter: folder_path'),
        ('name', ('-n', '--name'), (), 'text', False, None, False, 'Parameter: name (optional)'),
        ('scale', ('-s', '--scale'), (), 'text', False, None, False, 'Parameter: scale (optional)'),
        ('server_check', ('--server-check',), (), 'boolean', False, False, True, 'Run server check instead of full inference'),
        ('project', ('-P', '--project'), (), 'text', False, None, False, 'Project/chargeback group for billing'),
        ('queue', ('-q', '--queue'), (), 'text', False, None, False, 'Queue for job submission (default: the saved queue)'),
        ('data_path', ('-d', '--data-path'), (), 'text', True, None, False, 'Path to the dataset'),
    ],
    'finetune': [
        ('lora_adapter_path', ('-l', '--lora-adapter-path'), (), 'text', False, None, False, 'Parameter: lora_adapter_path (optional)'),
        ('base_model', ('-b', '--base-model'), (), 'text', False, None, False, 'Parameter: base_model (optional)'),
        ('name', ('-n', '--name'), (), 'text', False, None, False, 'Parameter: name (optional)'),
        ('scale', ('-s', '--scale'), (), 'text', False, None, False, 'Parameter: scale (optional)'),
        ('weights_path', ('-w', '--weights-path'), (), 'text', False, None, False, 'Parameter: weights_path (optional)'),
        ('server_check', ('--server-check',), (), 'boolean', False, False, True, 'Run server check instead of full inference'),
        ('project', ('-P', '--project'), (), 'text', False, None, False, 'Project/chargeback group for billing'),
        ('queue', ('-q', '--queue'), (), 'text', False, None, False, 'Queue for job submission (default: the saved queue)'),
        ('data_path', ('-d', '--data-path'), (), 'text', True, None, False, 'Path to the dataset'),
    ],
    'huggingface': [
        ('repo', ('--repo',), (), 'text', True, None, False, 'Parameter: repo'),
        ('revision', ('-r', '--revision'), (), 'text', False, None, False, 'Parameter: revision (optional)'),
        ('name', ('-n', '--name'), (), 'text', False, None, False, 'Parameter: name (optional)'),
        ('scale', ('-s', '--scale'), (), 'text', False, None, False, 'Parameter: scale (optional)'),
        ('server_check', ('--server-check',), (), 'boolean', False, False, True, 'Run server check instead of full inference'),
        ('project', ('-P', '--project'), (), 'text', False, None, False, 'Project/chargeback group for billing'),
        ('queue', ('-q', '--queue'), (), 'text', False, None, False, 'Queue for job submission (default: the saved queue)'),
        ('data_path', ('-d', '--data-path'), (), 'text', True, None, False, 'Path to the dataset'),
    ],
}

CELLMAP_FLOW_SERVER = {
    '': [
        ('log_level', ('--log-level',), (), 'choice', False, 'INFO', False, 'Set the logging level'),
    ],
    'list-models': [],
    'script': [
        ('script_path', ('--script-path',), (), 'text', True, None, False, 'Parameter: script_path'),
        ('name', ('-n', '--name'), (), 'text', False, None, False, 'Parameter: name (optional)'),
        ('scale', ('-s', '--scale'), (), 'text', False, None, False, 'Parameter: scale (optional)'),
        ('keyfile', ('--keyfile',), (), 'text', False, None, False, 'Path to SSL private key file'),
        ('certfile', ('--certfile',), (), 'text', False, None, False, 'Path to SSL certificate file'),
        ('port', ('-p', '--port'), (), 'integer', False, 0, False, 'Port to listen on'),
        ('debug', ('--debug',), (), 'boolean', False, False, True, 'Run in debug mode'),
        ('data_path', ('-d', '--data-path'), (), 'text', True, None, False, 'Path to the dataset'),
    ],
    'dacapo': [
        ('run_name', ('-r', '--run-name'), (), 'text', True, None, False, 'Parameter: run_name'),
        ('iteration', ('-i', '--iteration'), (), 'integer', True, None, False, 'Parameter: iteration'),
        ('name', ('-n', '--name'), (), 'text', False, None, False, 'Parameter: name (optional)'),
        ('scale', ('-s', '--scale'), (), 'text', False, None, False, 'Parameter: scale (optional)'),
        ('keyfile', ('--keyfile',), (), 'text', False, None, False, 'Path to SSL private key file'),
        ('certfile', ('--certfile',), (), 'text', False, None, False, 'Path to SSL certificate file'),
        ('port', ('-p', '--port'), (), 'integer', False, 0, False, 'Port to listen on'),
        ('debug', ('--debug',), (), 'boolean', False, False, True, 'Run in debug mode'),
        ('data_path', ('-d', '--data-path'), (), 'text', True, None, False, 'Path to the dataset'),
    ],
    'fly': [
        ('checkpoint_path', ('--checkpoint-path',), (), 'text', True, None, False, 'Parameter: checkpoint_path'),
        ('channels', ('-c', '--channels'), (), 'text', True, None, False, 'Parameter: channels [comma-separated values]'),
        ('input_voxel_size', ('--input-voxel-size',), (), 'text', True, None, False, 'Parameter: input_voxel_size'),
        ('output_voxel_size', ('--output-voxel-size',), (), 'text', True, None, False, 'Parameter: output_voxel_size'),
        ('name', ('-n', '--name'), (), 'text', False, None, False, 'Parameter: name (optional)'),
        ('input_size', ('-i', '--input-size'), (), 'text', False, None, False, 'Parameter: input_size (optional)'),
        ('output_size', ('-o', '--output-size'), (), 'text', False, None, False, 'Parameter: output_size (optional)'),
        ('scale', ('-s', '--scale'), (), 'text', False, None, False, 'Parameter: scale (optional)'),
        ('keyfile', ('--keyfile',), (), 'text', False, None, False, 'Path to SSL private key file'),
        ('certfile', ('--certfile',), (), 'text', False, None, False, 'Path to SSL certificate file'),
        ('port', ('-p', '--port'), (), 'integer', False, 0, False, 'Port to listen on'),
        ('debug', ('--debug',), (), 'boolean', False, False, True, 'Run in debug mode'),
        ('data_path', ('-d', '--data-path'), (), 'text', True, None, False, 'Path to the dataset'),
    ],
    'bioimage': [
        ('model_name', ('-m', '--model-name'), (), 'text', True, None, False, 'Parameter: model_name'),
        ('voxel_size', ('-v', '--voxel-size'), (), 'text', True, None, False, 'Parameter: voxel_size'),
        ('edge_length_to_process', ('-e', '--edge-length-to-process'), (), 'text', False, None, False, 'Parameter: edge_length_to_process (optional)'),
        ('name', ('-n', '--name'), (), 'text', False, None, False, 'Parameter: name (optional)'),
        ('scale', ('-s', '--scale'), (), 'text', False, None, False, 'Parameter: scale (optional)'),
        ('keyfile', ('--keyfile',), (), 'text', False, None, False, 'Path to SSL private key file'),
        ('certfile', ('--certfile',), (), 'text', False, None, False, 'Path to SSL certificate file'),
        ('port', ('-p', '--port'), (), 'integer', False, 0, False, 'Port to listen on'),
        ('debug', ('--debug',), (), 'boolean', False, False, True, 'Run in debug mode'),
        ('data_path', ('-d', '--data-path'), (), 'text', True, None, False, 'Path to the dataset'),
    ],
    'cellmap': [
        ('folder_path', ('-f', '--folder-path'), (), 'text', True, None, False, 'Parameter: folder_path'),
        ('name', ('-n', '--name'), (), 'text', False, None, False, 'Parameter: name (optional)'),
        ('scale', ('-s', '--scale'), (), 'text', False, None, False, 'Parameter: scale (optional)'),
        ('keyfile', ('--keyfile',), (), 'text', False, None, False, 'Path to SSL private key file'),
        ('certfile', ('--certfile',), (), 'text', False, None, False, 'Path to SSL certificate file'),
        ('port', ('-p', '--port'), (), 'integer', False, 0, False, 'Port to listen on'),
        ('debug', ('--debug',), (), 'boolean', False, False, True, 'Run in debug mode'),
        ('data_path', ('-d', '--data-path'), (), 'text', True, None, False, 'Path to the dataset'),
    ],
    'finetune': [
        ('lora_adapter_path', ('-l', '--lora-adapter-path'), (), 'text', False, None, False, 'Parameter: lora_adapter_path (optional)'),
        ('base_model', ('-b', '--base-model'), (), 'text', False, None, False, 'Parameter: base_model (optional)'),
        ('name', ('-n', '--name'), (), 'text', False, None, False, 'Parameter: name (optional)'),
        ('scale', ('-s', '--scale'), (), 'text', False, None, False, 'Parameter: scale (optional)'),
        ('weights_path', ('-w', '--weights-path'), (), 'text', False, None, False, 'Parameter: weights_path (optional)'),
        ('keyfile', ('--keyfile',), (), 'text', False, None, False, 'Path to SSL private key file'),
        ('certfile', ('--certfile',), (), 'text', False, None, False, 'Path to SSL certificate file'),
        ('port', ('-p', '--port'), (), 'integer', False, 0, False, 'Port to listen on'),
        ('debug', ('--debug',), (), 'boolean', False, False, True, 'Run in debug mode'),
        ('data_path', ('-d', '--data-path'), (), 'text', True, None, False, 'Path to the dataset'),
    ],
    'huggingface': [
        ('repo', ('--repo',), (), 'text', True, None, False, 'Parameter: repo'),
        ('revision', ('-r', '--revision'), (), 'text', False, None, False, 'Parameter: revision (optional)'),
        ('name', ('-n', '--name'), (), 'text', False, None, False, 'Parameter: name (optional)'),
        ('scale', ('-s', '--scale'), (), 'text', False, None, False, 'Parameter: scale (optional)'),
        ('keyfile', ('--keyfile',), (), 'text', False, None, False, 'Path to SSL private key file'),
        ('certfile', ('--certfile',), (), 'text', False, None, False, 'Path to SSL certificate file'),
        ('port', ('-p', '--port'), (), 'integer', False, 0, False, 'Port to listen on'),
        ('debug', ('--debug',), (), 'boolean', False, False, True, 'Run in debug mode'),
        ('data_path', ('-d', '--data-path'), (), 'text', True, None, False, 'Path to the dataset'),
    ],
}

CELLMAP_FLOW_YAML = [
    ('config_path', ('config_path',), (), 'path', False, None, False, None),
    ('log_level', ('--log-level',), (), 'choice', False, 'INFO', False, 'Set the logging level'),
    ('list_types', ('--list-types',), (), 'boolean', False, False, True, 'List available model types and exit'),
    ('validate_only', ('--validate-only',), (), 'boolean', False, False, True, 'Validate YAML configuration without running jobs'),
]

_TYPES = [
    ("bioimage", "BioModelConfig", "model_name, voxel_size, edge_length_to_process, name, scale", "model_name, voxel_size"),
    ("cellmap", "CellMapModelConfig", "folder_path, name, scale", "folder_path"),
    ("dacapo", "DaCapoModelConfig", "run_name, iteration, name, scale", "run_name, iteration"),
    ("finetune", "FinetuneModelConfig", "lora_adapter_path, base_model, name, scale, weights_path", ""),
    ("fly", "FlyModelConfig", "checkpoint_path, channels, input_voxel_size, output_voxel_size, name, input_size, output_size, scale",
     "checkpoint_path, channels, input_voxel_size, output_voxel_size"),
    ("huggingface", "HuggingFaceModelConfig", "repo, revision, name, scale", "repo"),
    ("script", "ScriptModelConfig", "script_path, name, scale", "script_path"),
]


def _param(p):
    # to_info_dict() reports an unset default as None on every click 8.x,
    # where p.default is a sentinel on some versions.
    return (
        p.name,
        tuple(p.opts),
        tuple(p.secondary_opts),
        p.type.name,
        p.required,
        p.to_info_dict().get("default"),
        bool(getattr(p, "is_flag", False)),
        getattr(p, "help", None),
    )


def _surface(group):
    out = {"": [_param(p) for p in group.params]}
    for name, command in group.commands.items():
        out[name] = [_param(p) for p in command.params]
    return out


def test_cellmap_flow_commands_and_options_are_unchanged():
    assert list(cli_module.cli.commands) == list(CELLMAP_FLOW)[1:]
    assert _surface(cli_module.cli) == CELLMAP_FLOW


def test_cellmap_flow_server_commands_and_options_are_unchanged():
    assert list(server_cli.commands) == list(CELLMAP_FLOW_SERVER)[1:]
    assert _surface(server_cli) == CELLMAP_FLOW_SERVER


def test_cellmap_flow_yaml_options_are_unchanged():
    assert [_param(p) for p in yaml_cli.main.params] == CELLMAP_FLOW_YAML


@pytest.mark.parametrize("group, prog", [(cli_module.cli, "cellmap_flow"), (server_cli, "cellmap_flow_server")])
def test_list_models_output_is_unchanged(group, prog):
    gc.collect()  # a model class another test defined must not be listed
    result = CliRunner().invoke(group, ["list-models"])
    assert result.exit_code == 0, result.output
    expected = "Available model configurations:\n\n"
    for cli_name, cls_name, params, _ in _TYPES:
        expected += f"  {cli_name:20s} - {cls_name}\n"
        expected += f"                       Parameters: {params}\n"
    expected += f"\nUse '{prog} <model-name> --help' for detailed parameter information.\n"
    assert result.output == expected


def test_yaml_list_types_output_is_unchanged():
    gc.collect()
    result = CliRunner().invoke(yaml_cli.main, ["--list-types"])
    assert result.exit_code == 0, result.output
    expected = "Available model types:\n\n"
    for cli_name, cls_name, _, required in _TYPES:
        expected += f"  {cli_name:20s} - {cls_name}\n"
        if required:
            expected += f"                       Required: {required}\n"
    expected += "\nSee example YAML configuration in the docstring with --help\n"
    assert result.output == expected


# --- to_dict() and command of every model type --------------------------------

@pytest.fixture
def fake_cellmap_models(monkeypatch):
    """CellMapModelConfig opens the model folder in its constructor."""

    class CellmapModel:
        def __init__(self, folder_path):
            self.folder_path = folder_path

    leaf = types.ModuleType("cellmap_models.model_export.cellmap_model")
    leaf.CellmapModel = CellmapModel
    monkeypatch.setitem(sys.modules, "cellmap_models", types.ModuleType("cellmap_models"))
    monkeypatch.setitem(
        sys.modules, "cellmap_models.model_export", types.ModuleType("cellmap_models.model_export")
    )
    monkeypatch.setitem(sys.modules, "cellmap_models.model_export.cellmap_model", leaf)


HF_METADATA = {
    "description": "Mito, v1", "unrelated": 1, "channels_names": ["mito"],
    "input_voxel_size": [8, 8, 8], "output_voxel_size": [8, 8, 8], "model_type": "unet",
}


@pytest.fixture
def hf_metadata(monkeypatch):
    """to_dict() adds the repo's metadata.json; command must not fetch it."""
    calls = []

    def fake(self):
        calls.append(self.repo)
        return HF_METADATA

    monkeypatch.setattr(HuggingFaceModelConfig, "_load_metadata", fake)
    return calls


FLY_ENTRY = {
    "type": "fly", "checkpoint_path": "/ckpt/fly run/model.ts", "channels": ["mito", "er"],
    "input_voxel_size": [16, 16, 16], "output_voxel_size": [8, 8, 8],
    "input_size": [100, 100, 100], "output_size": [20, 20, 20], "name": "fly base",
}


def _configs():
    from cellmap_flow.models.models_config import CellMapModelConfig

    return {
        "script": ScriptModelConfig(script_path="/groups/my models/mito.py", name="mito", scale="s1"),
        "script_bare": ScriptModelConfig(script_path="/m.py"),
        "dacapo": DaCapoModelConfig(run_name="run 1", iteration=5000, name="dc", scale="s0"),
        "fly": FlyModelConfig(
            checkpoint_path="/ckpt/model_checkpoint_1000", channels=["mito", "er"],
            input_voxel_size=(16, 16, 16), output_voxel_size=(8, 8, 8), name="fly",
            input_size=(100, 100, 100), output_size=(20, 20, 20), scale="s1",
        ),
        # As the server CLI passes them: strings, and no sizes.
        "fly_from_cli": FlyModelConfig(
            checkpoint_path="/c.ts", channels="mito, er",
            input_voxel_size="8,8,8", output_voxel_size="8,8,8",
        ),
        "bio": BioModelConfig(
            model_name="affable-shark", voxel_size=(8, 8, 8), edge_length_to_process=64,
            name="bio", scale="s0",
        ),
        "bio_from_cli": BioModelConfig(model_name="affable-shark", voxel_size="8,8,8"),
        "cellmap": CellMapModelConfig(folder_path="/models/my mito/"),
        "cellmap_named": CellMapModelConfig(folder_path="/models/mito", name="m", scale="s2"),
        "hf": HuggingFaceModelConfig(repo="cellmap/mito-v1", revision="abc123", name="m v1", scale="s0"),
        "hf_bare": HuggingFaceModelConfig(repo="cellmap/mito-v1"),
        "finetune": FinetuneModelConfig(
            lora_adapter_path="/runs/my run/lora_adapter", base_model=FLY_ENTRY, name="ft", scale="s1",
        ),
        "finetune_of_finetune": FinetuneModelConfig(
            weights_path="/runs/r2/full_finetune/model_state_dict.pt",
            base_model={
                "type": "finetune", "lora_adapter_path": "/runs/r1/lora_adapter", "weights_path": None,
                "base_model": {"type": "script", "script_path": "/s.py"}, "name": "inner",
            },
        ),
    }


TO_DICT = {
    "script": {'type': 'script', 'script_path': '/groups/my models/mito.py', 'name': 'mito', 'scale': 's1'},
    "script_bare": {'type': 'script', 'script_path': '/m.py'},
    "dacapo": {'type': 'dacapo', 'run_name': 'run 1', 'iteration': 5000, 'name': 'dc', 'scale': 's0'},
    "fly": {
        'type': 'fly', 'checkpoint_path': '/ckpt/model_checkpoint_1000', 'channels': ['mito', 'er'],
        'input_voxel_size': [16, 16, 16], 'output_voxel_size': [8, 8, 8], 'name': 'fly',
        'input_size': [100, 100, 100], 'output_size': [20, 20, 20], 'scale': 's1',
    },
    "fly_from_cli": {
        'type': 'fly', 'checkpoint_path': '/c.ts', 'channels': ['mito', 'er'],
        'input_voxel_size': [8, 8, 8], 'output_voxel_size': [8, 8, 8],
        'input_size': [178, 178, 178], 'output_size': [56, 56, 56],
    },
    "bio": {
        'type': 'bioimage', 'model_name': 'affable-shark', 'voxel_size': [8, 8, 8], 'name': 'bio',
        'scale': 's0', 'edge_length_to_process': 64,
    },
    "bio_from_cli": {'type': 'bioimage', 'model_name': 'affable-shark', 'voxel_size': [8, 8, 8]},
    "cellmap": {'type': 'cellmap', 'folder_path': '/models/my mito/', 'name': 'my mito'},
    "cellmap_named": {'type': 'cellmap', 'folder_path': '/models/mito', 'name': 'm', 'scale': 's2'},
    "hf": {
        'type': 'huggingface', 'repo': 'cellmap/mito-v1', 'revision': 'abc123', 'name': 'm v1',
        'scale': 's0', 'channels_names': ['mito'], 'input_voxel_size': [8, 8, 8],
        'output_voxel_size': [8, 8, 8], 'model_type': 'unet', 'description': 'Mito, v1',
    },
    "hf_bare": {
        'type': 'huggingface', 'repo': 'cellmap/mito-v1', 'name': 'mito-v1', 'channels_names': ['mito'],
        'input_voxel_size': [8, 8, 8], 'output_voxel_size': [8, 8, 8], 'model_type': 'unet',
        'description': 'Mito, v1',
    },
    "finetune": {
        'type': 'finetune', 'lora_adapter_path': '/runs/my run/lora_adapter', 'weights_path': None,
        'base_model': FLY_ENTRY, 'name': 'ft', 'scale': 's1', 'channels': ['mito', 'er'],
        'checkpoint_path': '/ckpt/fly run/model.ts', 'input_voxel_size': [16, 16, 16],
        'output_voxel_size': [8, 8, 8], 'input_size': [100, 100, 100], 'output_size': [20, 20, 20],
        'base_type': 'fly',
    },
    "finetune_of_finetune": {
        'type': 'finetune', 'lora_adapter_path': None,
        'weights_path': '/runs/r2/full_finetune/model_state_dict.pt',
        'base_model': {
            'type': 'finetune', 'lora_adapter_path': '/runs/r1/lora_adapter', 'weights_path': None,
            'base_model': {'type': 'script', 'script_path': '/s.py'}, 'name': 'inner',
        },
        'base_type': 'finetune',
    },
}

COMMAND = {
    "script": "script --script-path '/groups/my models/mito.py' --name mito --scale s1",
    "script_bare": "script --script-path /m.py",
    "dacapo": "dacapo --run-name 'run 1' --iteration 5000 --name dc --scale s0",
    "fly": (
        "fly --checkpoint-path /ckpt/model_checkpoint_1000 --channels mito,er --input-voxel-size 16,16,16"
        " --output-voxel-size 8,8,8 --name fly --input-size 100,100,100 --output-size 20,20,20 --scale s1"
    ),
    "fly_from_cli": (
        "fly --checkpoint-path /c.ts --channels mito,er --input-voxel-size 8,8,8 --output-voxel-size 8,8,8"
        " --input-size 178,178,178 --output-size 56,56,56"
    ),
    "bio": "bioimage --model-name affable-shark --voxel-size 8,8,8 --edge-length-to-process 64 --name bio --scale s0",
    "bio_from_cli": "bioimage --model-name affable-shark --voxel-size 8,8,8",
    "cellmap": "cellmap --folder-path '/models/my mito/' --name 'my mito'",
    "cellmap_named": "cellmap --folder-path /models/mito --name m --scale s2",
    "hf": "huggingface --repo cellmap/mito-v1 --revision abc123 --name 'm v1' --scale s0",
    "hf_bare": "huggingface --repo cellmap/mito-v1 --name mito-v1",
    "finetune": (
        "finetune --lora-adapter-path '/runs/my run/lora_adapter' --base-model"
        " eyJ0eXBlIjoiZmx5IiwiY2hlY2twb2ludF9wYXRoIjoiL2NrcHQvZmx5IHJ1bi9tb2RlbC50cyIsImNoYW5uZWxzIjpbIm1pdG8iLCJlciJdLCJpbnB1dF92b3hlbF9zaXplIjpbMTYsMTYsMTZdLCJvdXRwdXRfdm94ZWxfc2l6ZSI6WzgsOCw4XSwiaW5wdXRfc2l6ZSI6WzEwMCwxMDAsMTAwXSwib3V0cHV0X3NpemUiOlsyMCwyMCwyMF0sIm5hbWUiOiJmbHkgYmFzZSJ9"
        " --name ft --scale s1"
    ),
    "finetune_of_finetune": (
        "finetune --base-model"
        " eyJ0eXBlIjoiZmluZXR1bmUiLCJsb3JhX2FkYXB0ZXJfcGF0aCI6Ii9ydW5zL3IxL2xvcmFfYWRhcHRlciIsIndlaWdodHNfcGF0aCI6bnVsbCwiYmFzZV9tb2RlbCI6eyJ0eXBlIjoic2NyaXB0Iiwic2NyaXB0X3BhdGgiOiIvcy5weSJ9LCJuYW1lIjoiaW5uZXIifQ"
        " --weights-path /runs/r2/full_finetune/model_state_dict.pt"
    ),
}


@pytest.mark.parametrize("key", list(TO_DICT))
def test_to_dict_is_unchanged(key, fake_cellmap_models, hf_metadata):
    result = _configs()[key].to_dict()
    assert result == TO_DICT[key]
    assert list(result) == list(TO_DICT[key]), "key order is what the exported YAML shows"


@pytest.mark.parametrize("key", list(COMMAND))
def test_command_is_unchanged(key, fake_cellmap_models, hf_metadata):
    assert _configs()[key].command == COMMAND[key]
    assert hf_metadata == [], "building a launch command must not download metadata.json"


# --- the command lines the launchers submit ------------------------------------

@pytest.fixture
def no_viewer(monkeypatch):
    from cellmap_flow.utils import neuroglancer_utils

    calls = []
    monkeypatch.setattr(
        neuroglancer_utils, "generate_neuroglancer_url", lambda path, wrap_raw=True: calls.append(path)
    )
    monkeypatch.setattr(type(g), "save_server_config", lambda self: None)
    return calls


@pytest.fixture
def cli_submissions(monkeypatch, no_viewer):
    commands = []

    def record(command, *args, **kwargs):
        commands.append(command)
        return object()

    monkeypatch.setattr(cli_module, "start_hosts", record)
    return commands


def test_cellmap_flow_type_command_submits_the_same_command_line(cli_submissions, no_viewer):
    result = CliRunner().invoke(
        cli_module.cli,
        ["script", "--script-path", "/m/my script.py", "--name", "m", "-d", "/d/my raw.zarr"],
    )
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert cli_submissions == [
        f"{bsub_utils.SERVER_COMMAND} script --script-path '/m/my script.py' --name m -d '/d/my raw.zarr'"
    ]
    assert no_viewer == ["/d/my raw.zarr"]


def test_cellmap_flow_run_submits_the_same_command_line(cli_submissions, no_viewer):
    result = CliRunner().invoke(
        cli_module.cli,
        ["run", "-m", "dacapo", "-c", "run_name=run 1", "-c", "iteration=5", "-d", "/d/raw.zarr"],
    )
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert cli_submissions == [
        f"{bsub_utils.SERVER_COMMAND} dacapo --run-name 'run 1' --iteration 5 -d /d/raw.zarr"
    ]


def test_cellmap_flow_yaml_submits_the_same_command_lines(monkeypatch, no_viewer):
    commands = []

    def record(command, **kwargs):
        commands.append((kwargs["job_name"], command))
        return object()

    monkeypatch.setattr(yaml_cli, "start_hosts", record)
    models = [
        ScriptModelConfig(script_path="/m/my script.py", name="a"),
        DaCapoModelConfig(run_name="r", iteration=5, name="b", scale="s1"),
    ]
    yaml_cli.run_multiple(models, "/d/my raw.zarr", "grp", "gpu_h100")

    serve = bsub_utils.SERVER_COMMAND
    assert sorted(commands) == [
        ("a", f"{serve} script --script-path '/m/my script.py' --name a -d '/d/my raw.zarr'"),
        ("b", f"{serve} dacapo --run-name r --iteration 5 --name b --scale s1 -d '/d/my raw.zarr/s1'"),
    ]


@pytest.mark.parametrize(
    "launcher, target, name, expected",
    [
        ("run_model", "/models/mito v2", "mito",
         "cellmap --folder-path '/models/mito v2' --name mito -d '/data/my raw.zarr'"),
        ("run_hf_model", "cellmap/mito-v1", "mito v1",
         "huggingface --repo cellmap/mito-v1 --name mito_v1 -d '/data/my raw.zarr'"),
    ],
)
def test_the_dashboard_launchers_submit_the_same_command_lines(monkeypatch, launcher, target, name, expected):
    import cellmap_flow.models.run as run

    commands = []

    def record(command, **kwargs):
        commands.append(command)
        raise JobStartError("recorded")  # stops before the viewer is touched

    monkeypatch.setattr(run, "start_hosts", record)
    g.dataset_path = "/data/my raw.zarr"
    getattr(run, launcher)(target, name, "blob")

    assert commands == [f"{bsub_utils.SERVER_COMMAND} {expected}"]


# --- the dashboard's model form ---------------------------------------------------

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
        "CellMapModelConfig", "Cell Map Model",
        _param_info("folder_path", "string", True, "file"), _NAME, _SCALE,
    ),
    "DaCapoModelConfig": _type_info(
        "DaCapoModelConfig", "Da Capo Model",
        _param_info("run_name", "str", True, "text"),
        _param_info("iteration", "int", True, "number"),
        _NAME, _SCALE,
    ),
    "FinetuneModelConfig": _type_info(
        "FinetuneModelConfig", "Finetune Model",
        _param_info("lora_adapter_path", "str", False, "file", None),
        _param_info("base_model", "dict", False, "textarea", None),
        _param_info("name", "str", False, "text", None),
        _SCALE,
        _param_info("weights_path", "str", False, "file", None),
    ),
    "FlyModelConfig": _type_info(
        "FlyModelConfig", "Fly Model",
        _param_info("checkpoint_path", "str", True, "file"),
        _param_info("channels", "list", True, "textarea"),
        _param_info("input_voxel_size", "tuple", True, "textarea"),
        _param_info("output_voxel_size", "tuple", True, "textarea"),
        _param_info("name", "str", False, "text", None),
        _param_info("input_size", "string", False, "number", None),
        _param_info("output_size", "string", False, "number", None),
        _SCALE,
    ),
    "HuggingFaceModelConfig": _type_info(
        "HuggingFaceModelConfig", "Hugging Face Model",
        _param_info("repo", "string", True, "text"),
        _param_info("revision", "string", False, "text", None),
        _NAME, _SCALE,
    ),
    "ScriptModelConfig": _type_info(
        "ScriptModelConfig", "Script Model",
        _param_info("script_path", "string", True, "file"), _NAME, _SCALE,
    ),
}


@pytest.fixture
def dashboard(monkeypatch):
    from cellmap_flow.dashboard.routes.models import models_bp

    app = Flask(__name__)
    app.register_blueprint(models_bp)
    g.models_config = []
    return app.test_client()


def test_the_model_form_offers_the_same_types(dashboard):
    gc.collect()
    response = dashboard.get("/api/model-config-types")
    assert response.status_code == 200
    data = response.get_json()
    assert set(data) == set(MODEL_CONFIG_TYPES)
    assert data == MODEL_CONFIG_TYPES


def test_the_model_form_parses_its_strings_the_same_way(dashboard):
    # Form tuples become floats (unlike the CLI's ints) and lists of strings
    # stay strings; that difference is today's behaviour, kept as it is.
    response = dashboard.post("/api/create-model-config", json={
        "class_name": "FlyModelConfig",
        "params": {
            "checkpoint_path": "/c.ts", "channels": "mito, er", "input_voxel_size": "16,16,16",
            "output_voxel_size": "[8, 8, 8]", "input_size": "", "name": "fly",
        },
    })
    assert response.status_code == 200, response.get_json()
    assert response.get_json()["config_dict"] == {
        "type": "fly", "checkpoint_path": "/c.ts", "channels": ["mito", "er"],
        "input_voxel_size": [16.0, 16.0, 16.0], "output_voxel_size": [8, 8, 8], "name": "fly",
        "input_size": [178, 178, 178], "output_size": [56, 56, 56],
    }
    assert [type(m).__name__ for m in g.models_config] == ["FlyModelConfig"]


@pytest.mark.parametrize(
    "class_name, params, message",
    [
        ("NoSuchModelConfig", {}, "Unknown model config class: NoSuchModelConfig"),
        ("DaCapoModelConfig", {"run_name": "r", "iteration": ""}, "Required parameter 'iteration' is missing"),
        ("DaCapoModelConfig", {"run_name": "r"}, "Failed to instantiate DaCapoModelConfig: "),
    ],
)
def test_the_model_form_reports_bad_input_the_same_way(dashboard, class_name, params, message):
    response = dashboard.post("/api/create-model-config", json={"class_name": class_name, "params": params})
    assert response.status_code == 400
    assert response.get_json()["error"].startswith(message)


def test_the_cli_parses_its_strings_the_same_way():
    from cellmap_flow.utils.cli_utils import process_constructor_args

    kwargs = {"checkpoint_path": "/c.ts", "channels": "mito,er", "input_voxel_size": "8,8,8",
              "output_voxel_size": "4", "input_size": "10,10,10", "name": None}
    assert process_constructor_args(FlyModelConfig, kwargs) == {
        "checkpoint_path": "/c.ts", "channels": ["mito", "er"], "input_voxel_size": (8, 8, 8),
        "output_voxel_size": (4,), "input_size": "10,10,10",
    }


def test_a_yaml_entry_with_aliases_builds_the_same_config(caplog):
    from cellmap_flow.utils.config_utils import build_model_from_entry

    with caplog.at_level(logging.WARNING, logger="cellmap_flow.utils.config_utils"):
        model = build_model_from_entry(
            {"type": "Fly", "checkpoint": "/c.ts", "classes": ["mito"], "resolution": 8,
             "input_size": [20, 20, 20], "output_size": [10, 10, 10]},
            model_name="m",
        )
    assert model.to_dict() == {
        "type": "fly", "checkpoint_path": "/c.ts", "channels": ["mito"],
        "input_voxel_size": [8, 8, 8], "output_voxel_size": [8, 8, 8], "name": "m",
        "input_size": [20, 20, 20], "output_size": [10, 10, 10],
    }
    assert "'output_voxel_size' not specified" in caplog.text
