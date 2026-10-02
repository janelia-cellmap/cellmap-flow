"""The CLIs, model configs and model form as users and files see them.

Pinned as literals because nothing else pins them, and changing any of it
is a breaking release rather than a refactor:

- the console scripts and what each one runs;
- every command and option of ``cellmap_flow``, ``cellmap_flow_server``,
  ``cellmap_flow_yaml``, ``cellmap_flow_view`` and the two blockwise
  commands, down to the short flags, which go to a model type's
  constructor arguments in signature order, and the type listings;
- ``to_dict()`` (exported and finetuned YAMLs are written from it),
  ``launch_entry`` (``serve --model`` rebuilds the config from it) and
  ``command`` (so do the per-type server commands) of every model type;
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

from cellmap_flow.cli import aliases, main
from cellmap_flow.cli.server_cli import cli as server_cli
from cellmap_flow.dashboard.state import get_session
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

# What each installed command runs. cellmap_flow is main(), which runs the
# click group. The commands before 0.3.0 are aliases (cli/aliases.py) of its
# subcommands; cellmap_flow_server keeps its per-type commands for one
# release.
ENTRY_POINTS = {
    "cellmap_flow": "cellmap_flow.cli.main:main",
    "cellmap_flow_yaml": "cellmap_flow.cli.aliases:yaml",
    "cellmap_flow_view": "cellmap_flow.cli.aliases:view",
    "cellmap_flow_blockwise": "cellmap_flow.cli.aliases:blockwise",
    "cellmap_flow_blockwise_multiple": "cellmap_flow.cli.aliases:blockwise_multiple",
    "cellmap_flow_app": "cellmap_flow.cli.aliases:app",
    "cellmap_flow_server": "cellmap_flow.cli.server_cli:main",
}


def _entry_point(target):
    module, attr = target.split(":")
    return getattr(importlib.import_module(module), attr)


def test_the_console_scripts_are_unchanged():
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts == ENTRY_POINTS
    click_commands = {name for name, target in scripts.items()
                      if isinstance(_entry_point(target), click.Command)}
    assert not click_commands, "a click command run as the script would not load the plugins"
    assert _entry_point(scripts["cellmap_flow"]) is main.main


# Where the console scripts pointed before 0.3.0. An environment installed
# then (a deployed pixi checkout, until it is reinstalled) runs them
# through these, and its `cellmap_flow` gets the new subcommands.
OLD_ENTRY_POINTS = [
    "cellmap_flow.cli.server_cli:cli", "cellmap_flow.cli.yaml_cli:main", "cellmap_flow.cli.viewer_cli:main",
    "cellmap_flow.blockwise.cli:cli", "cellmap_flow.blockwise.multiple_cli:cli",
    "cellmap_flow.dashboard.app:create_and_run_app",
]


def test_an_install_from_before_0_3_0_still_runs():
    assert _entry_point("cellmap_flow.cli.cli:main") is main.main
    for target in OLD_ENTRY_POINTS:
        assert callable(_entry_point(target)), target


# The old console script, its arguments, and the subcommand that replaces it.
ALIASES = [
    (aliases.yaml, "cellmap_flow_yaml", "yaml"),
    (aliases.view, "cellmap_flow_view", "view"),
    (aliases.blockwise, "cellmap_flow_blockwise", "blockwise"),
    (aliases.blockwise_multiple, "cellmap_flow_blockwise_multiple", "blockwise"),
    # It was a plain function: `cellmap_flow_app --help` started the dashboard.
    (aliases.app, "cellmap_flow_app", "dashboard"),
]


@pytest.mark.parametrize("alias, old, new", ALIASES, ids=[a[1] for a in ALIASES])
def test_an_old_console_script_says_what_replaces_it_and_runs_it(alias, old, new, capsys):
    with pytest.raises(SystemExit) as exited:
        alias(["--help"])
    out, err = capsys.readouterr()
    assert exited.value.code == 0
    assert err == f"`{old}` is deprecated and goes in the release after 0.3.0; use `cellmap_flow {new}`.\n"
    assert out.startswith(f"Usage: cellmap_flow {new} ")


TESTS = Path(__file__).resolve().parents[1]
SCRIPT = str(TESTS / "script_test" / "fake_model_script.py")
RAW = str(TESTS / "script_test" / "dummy.zarr" / "raw")


# cellmap_flow's own subcommands before 0.3.0 (hidden from --help: HIDDEN
# below) each say what replaces them before running it. `run` names the
# infer command its arguments stand for.
@pytest.mark.parametrize("old, new", [
    (["list-models"], ["models"]),
    (["list-plugins"], ["plugins", "list"]),
    (["run", "-m", "script", "-c", f"script_path={SCRIPT}", "-d", RAW, "--server-check"],
     ["infer", "script", "--script-path", SCRIPT, "-d", RAW, "--server-check"]),
], ids=["list-models", "list-plugins", "run"])
def test_an_old_subcommand_says_what_replaces_it_and_runs_it(old, new):
    result = CliRunner().invoke(main.cli, old)
    assert result.exit_code == 0, result.output + repr(result.exception)
    notice = (f"`cellmap_flow {old[0]}` is deprecated and goes in the release after 0.3.0; "
              f"use `cellmap_flow {' '.join(new)}`.\n")
    assert result.stderr.startswith(notice)
    assert result.stdout == CliRunner().invoke(main.cli, new).stdout


# --- the command-line surface ---------------------------------------------------

# (name, opts, secondary_opts, type name, required, default, is_flag, help),
# and nargs after them when it is not 1.
NAME = ('name', ('-n', '--name'), (), 'text', False, None, False, 'Parameter: name (optional)')
SCALE = ('scale', ('-s', '--scale'), (), 'text', False, None, False, 'Parameter: scale (optional)')
# Short flags go to the arguments in signature order; a later argument with
# a taken letter has none (K1).
LONG_SCALE = ('scale', ('--scale',), (), 'text', False, None, False, 'Parameter: scale (optional)')
DATA_PATH = ('data_path', ('-d', '--data-path'), (), 'text', True, None, False, 'Path to the dataset')
LOG_LEVEL = [('log_level', ('--log-level',), (), 'choice', False, 'INFO', False, 'Set the logging level')]

# Each model type's own options, the same in both CLIs.
MODEL_OPTIONS = {
    'script': [
        ('script_path', ('-s', '--script-path'), (), 'text', True, None, False, 'Parameter: script_path'),
        NAME, LONG_SCALE,
    ],
    'dacapo': [
        ('run_name', ('-r', '--run-name'), (), 'text', True, None, False, 'Parameter: run_name'),
        ('iteration', ('-i', '--iteration'), (), 'integer', True, None, False, 'Parameter: iteration'),
        NAME, SCALE,
    ],
    'fly': [
        ('checkpoint_path', ('-c', '--checkpoint-path'), (), 'text', True, None, False, 'Parameter: checkpoint_path'),
        ('channels', ('--channels',), (), 'text', True, None, False, 'Parameter: channels [comma-separated values]'),
        ('input_voxel_size', ('-i', '--input-voxel-size'), (), 'text', True, None, False, 'Parameter: input_voxel_size'),
        ('output_voxel_size', ('-o', '--output-voxel-size'), (), 'text', True, None, False,
         'Parameter: output_voxel_size'),
        NAME,
        ('input_size', ('--input-size',), (), 'text', False, None, False, 'Parameter: input_size (optional)'),
        ('output_size', ('--output-size',), (), 'text', False, None, False, 'Parameter: output_size (optional)'),
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
        ('repo', ('-r', '--repo'), (), 'text', True, None, False, 'Parameter: repo'),
        ('revision', ('--revision',), (), 'text', False, None, False, 'Parameter: revision (optional)'),
        NAME, SCALE,
    ],
}

SERVER_CHECK = ('server_check', ('--server-check',), (), 'boolean', False, False, True,
                'Run server check instead of full inference')
PROJECT = ('project', ('-P', '--project'), (), 'text', False, None, False, 'Project/chargeback group for billing')
QUEUE = ('queue', ('-q', '--queue'), (), 'text', False, None, False,
         'Queue for job submission (default: the saved queue)')

PASSED_THROUGH = [('args', ('args',), (), 'text', False, None, False, None, 'nargs=-1')]
LOG_LEVEL_OF_GROUP = [('log_level', ('--log-level',), (), 'choice', False, None, False,
                       'Set the logging level')]
PLUGIN_FILE = [
    ('filepath', ('filepath',), (), 'path', True, None, False, None),
    ('force', ('--force',), (), 'boolean', False, False, True, 'Overwrite existing plugin with the same name.'),
]
PLUGIN_NAME = [('name', ('name',), (), 'text', True, None, False, None)]
RESAMPLE = ('resample', ('--resample',), (), 'boolean', False, False, True,
            "When the dataset has no level at the model's input voxel size, resample a level to it, "
            "axis by axis, instead of reading the level as if it were at that size.")
ENV = ('env', ('--env',), (), 'text', False, None, False,
       "Run the server in this environment: a pixi environment of cellmap-flow's pixi.toml, "
       "or the absolute path of one with cellmap-flow installed (default: this one)")
INFER = {t: [*options, ENV, RESAMPLE, SERVER_CHECK, PROJECT, QUEUE, DATA_PATH] for t, options in MODEL_OPTIONS.items()}
# The server's own options: `serve` requires the model and data, and
# cellmap_flow_server takes them instead of a type's command.
MODEL_JSON_HELP = 'The model: its launch entry (ModelConfig.launch_entry), as JSON.'
PORT = ('port', ('-p', '--port'), (), 'integer', False, 0, False, 'Port to listen on')
DEBUG = ('debug', ('--debug',), (), 'boolean', False, False, True, 'Run in debug mode')
CERTFILE = ('certfile', ('--certfile',), (), 'text', False, None, False, 'Path to SSL certificate file')
KEYFILE = ('keyfile', ('--keyfile',), (), 'text', False, None, False, 'Path to SSL private key file')
SERVE = [('model_json', ('--model',), (), 'text', True, None, False, MODEL_JSON_HELP),
         DATA_PATH, PORT, DEBUG, CERTFILE, KEYFILE, RESAMPLE]

# Each command by its path; "" is the group's own options. A subcommand's
# --log-level (yaml, view, blockwise) defaults to the group's.
CELLMAP_FLOW = {
    '': LOG_LEVEL,
    'blockwise': [
        ('yaml_configs', ('yaml_configs',), (), 'path', True, None, False, None, 'nargs=-1'),
        ('client', ('-c', '--client'), (), 'boolean', False, False, True, 'Run as client if this flag is set.'),
        *LOG_LEVEL_OF_GROUP,
    ],
    'dashboard': [('neuroglancer_url', ('-n', '--neuroglancer-url'), (), 'text', False, None, False,
                   "The viewer the dashboard's page embeds.")],
    'doctor': [('core_only', ('--core-only',), (), 'boolean', False, False, True, 'Skip the finetune checks.')],
    'finetune': [],
    'finetune build-corrections': PASSED_THROUGH,
    'finetune export-merged': PASSED_THROUGH,
    'finetune train': PASSED_THROUGH,
    'infer': [],
    **{f'infer {t}': options for t, options in sorted(INFER.items())},
    'list-models': [],
    'list-plugins': [],
    'models': [],
    'plugins': [],
    'plugins list': [],
    'plugins register': PLUGIN_FILE,
    'plugins unregister': PLUGIN_NAME,
    'register': PLUGIN_FILE,
    'run': [
        ('model_type', ('-m', '--model-type'), (), 'text', True, None, False, 'Model type (e.g., dacapo, script, cellmap)'),
        DATA_PATH, QUEUE, PROJECT,
        ('config', ('-c', '--config'), (), 'text', False, None, False, 'Model configuration as key=value pairs'),
        SERVER_CHECK,
    ],
    'serve': SERVE,
    'unregister': PLUGIN_NAME,
    'view': [
        ('dataset', ('-d', '--dataset'), (), 'text', True, None, False, 'Path to the dataset (zarr or n5)'),
        ('project', ('-P', '--project'), (), 'text', False, None, False,
         'Charge group (LSF project) billed for the models launched from the dashboard'),
        RESAMPLE,
        *LOG_LEVEL_OF_GROUP,
    ],
    'yaml': [
        ('config_path', ('config_path',), (), 'path', False, None, False, None),
        *LOG_LEVEL_OF_GROUP,
        ('list_types', ('--list-types',), (), 'boolean', False, False, True, 'List available model types and exit'),
        ('validate_only', ('--validate-only',), (), 'boolean', False, False, True,
         'Validate YAML configuration without running jobs'),
    ],
}
# Commands hidden from --help: cellmap_flow's before 0.3.0.
HIDDEN = {'list-models', 'list-plugins', 'register', 'run', 'unregister'}

CELLMAP_FLOW_SERVER = {
    '': [*LOG_LEVEL, ('model_json', ('--model',), (), 'text', False, None, False, MODEL_JSON_HELP),
         ('data_path', ('-d', '--data-path'), (), 'text', False, None, False, 'Path to the dataset'),
         PORT, DEBUG, CERTFILE, KEYFILE, RESAMPLE],
    **dict(sorted({
    'list-models': [],
    **{t: [*options, KEYFILE, CERTFILE, PORT, DEBUG, DATA_PATH] for t, options in MODEL_OPTIONS.items()},
    }.items())),
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


def _surface(command, path=""):
    """{command path: its params} for ``command`` and every subcommand under it."""
    rows = {path: [_param(p) for p in command.params]}
    if isinstance(command, click.Group):
        ctx = click.Context(command)
        for name in command.list_commands(ctx):
            rows.update(_surface(command.get_command(ctx, name), f"{path} {name}".strip()))
    return rows


def _hidden(command, path=""):
    ctx = click.Context(command)
    hidden = set()
    for name in getattr(command, "list_commands", lambda ctx: [])(ctx):
        sub = command.get_command(ctx, name)
        if sub.hidden:
            hidden.add(f"{path} {name}".strip())
        hidden |= _hidden(sub, f"{path} {name}".strip())
    return hidden


@pytest.mark.parametrize(
    "command, expected, hidden",
    [(main.cli, CELLMAP_FLOW, HIDDEN), (server_cli, CELLMAP_FLOW_SERVER, set())],
    ids=["cellmap_flow", "cellmap_flow_server"],
)
def test_commands_and_options_are_unchanged(command, expected, hidden):
    surface = _surface(command)
    assert list(surface) == list(expected), "the commands, in --help's order"
    assert surface == expected
    assert _hidden(command) == hidden


def test_cellmap_flow_type_is_a_hidden_alias_of_infer_type():
    ctx = click.Context(main.cli)
    for model_type, options in INFER.items():
        command = main.cli.get_command(ctx, model_type)
        if model_type == "finetune":  # the finetune tools took the name
            assert command.name == "finetune" and isinstance(command, click.Group)
            continue
        assert command.hidden and [_param(p) for p in command.params] == options
        assert command.help == f"Deprecated: use `cellmap_flow infer {model_type}`."


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
    if prog == "cellmap_flow yaml":
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
    "command, argv, prog, notice",
    [
        (main.cli, ["models"], "cellmap_flow infer", ""),
        (server_cli, ["list-models"], "cellmap_flow_server",
         "`cellmap_flow_server list-models` is deprecated and goes in the release after 0.3.0; "
         "use `cellmap_flow models`.\n"),
        (main.cli, ["yaml", "--list-types"], "cellmap_flow yaml", ""),
    ],
)
def test_the_type_listings_are_unchanged(command, argv, prog, notice):
    gc.collect()  # a model class another test defined must not be listed
    result = CliRunner().invoke(command, argv)
    assert result.exit_code == 0, result.output
    assert (result.stdout, result.stderr) == (_listing(prog), notice)


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

# key: (config, to_dict(), command[, launch_entry]); the launch entry is
# to_dict() without its None values where it is not given.
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
        # Without the metadata: launching a server downloads nothing.
        {'type': 'huggingface', 'repo': 'cellmap/mito-v1', 'revision': 'abc123', 'name': 'm v1', 'scale': 's0'},
    ),
    "hf_bare": (
        lambda: HuggingFaceModelConfig(repo="cellmap/mito-v1"),
        {'type': 'huggingface', 'repo': 'cellmap/mito-v1', 'name': 'mito-v1', 'channels_names': ['mito'],
         'input_voxel_size': [8, 8, 8], 'output_voxel_size': [8, 8, 8], 'model_type': 'unet',
         'description': 'Mito, v1'},
        "huggingface --repo cellmap/mito-v1 --name mito-v1",
        {'type': 'huggingface', 'repo': 'cellmap/mito-v1', 'name': 'mito-v1'},
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
def test_to_dict_launch_entry_and_command_are_unchanged(key, fake_cellmap_models, hf_metadata):
    build, to_dict, command, *launch_entry = CONFIGS[key]
    config = build()
    assert config.command == command
    entry = launch_entry[0] if launch_entry else {k: v for k, v in to_dict.items() if v is not None}
    assert config.launch_entry == entry and list(config.launch_entry) == list(entry)
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


# Every type's form also offers env, which no constructor takes (models.envs).
_ENV = {"name": "env", "required": False, "type": "str", "input_type": "text",
        "description": "Environment (pixi env name or absolute path; blank: this one)"}


def _type_info(class_name, display_name, *params):
    return {
        "display_name": display_name,
        "description": f"Create a {display_name} model configuration",
        "class_name": class_name,
        "parameters": {p["name"]: p for p in (*params, _ENV)},
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
        assert [type(m).__name__ for m in get_session().models_config] == [class_name]
    else:
        assert response.get_json()["error"].startswith(expected)
