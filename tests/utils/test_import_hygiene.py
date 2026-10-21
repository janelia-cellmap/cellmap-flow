"""What importing a module does to the process, each in a fresh interpreter.

Modules that jobs, servers, finetune runs and the dashboard's request
threads import must not pull in the dashboard (globals configures logging and
reads ~/.cellmap_flow), a viewer, torch or the optional frameworks; nor touch
the process's logging, signal handlers or home directory. A fresh
interpreter per row, so what this process has already imported hides
nothing.
"""

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PRELUDE = """
import importlib, sys

def check(modules, heavy):
    for module in modules:
        importlib.import_module(module)
    loaded = [m for m in heavy if m in sys.modules]
    assert not loaded, f"importing {modules} imported {loaded}"

LIGHT = ["cellmap_flow.globals", "flask", "neuroglancer"]
"""

ROWS = {
    # What an LSF job imports, and the dashboard's finetune job manager: the
    # two import the rest of jobs/ and finetune/job_manager/.
    "jobs": """
check(["cellmap_flow.jobs.launch", "cellmap_flow.finetune.job_manager.manager"],
      LIGHT + ["huggingface_hub", "peft", "torch"])
""",
    # torch only once a runner is built.
    "serving": """
check(["cellmap_flow.models.geometry", "cellmap_flow.models.geometry_cache", "cellmap_flow.inference.runner",
       "cellmap_flow.inferencer", "cellmap_flow.image_data_interface"]
      + [f"cellmap_flow.serving.{m}" for m in ("chunkmirage_ops", "protocol", "client", "probe", "restart_token")],
      LIGHT + ["torch"])
""",
    # cellmap_flow serve applies --log-level before it imports the server,
    # which once reset it to INFO while the server imported globals.
    "server-logging": """
import logging
root = logging.getLogger()
root.setLevel(logging.WARNING)
handlers = list(root.handlers)
check(["cellmap_flow.serving.chunkmirage_server", "cellmap_flow.serving.engine"],
      ["cellmap_flow.globals", "cellmap_flow.dashboard", "neuroglancer", "torch"])
assert (root.level, root.handlers) == (logging.WARNING, handlers)
""",
    # describe_types() runs when the dashboard opens its model form. It
    # imports the model config classes, which the CLIs, servers and blockwise
    # workers import too; a type loads its framework only to build a model.
    "registry": """
check(["cellmap_flow.models.registry", "cellmap_flow.config.yaml", "cellmap_flow.serving.launch"],
      LIGHT + ["cellmap_flow.models.models_config", "cellmap_flow.models.configs",
               "torch", "huggingface_hub", "peft"])
from cellmap_flow.models.registry import describe_types
assert {"BioModelConfig", "DaCapoModelConfig"} <= set(describe_types())
# An onnx entry as the CLIs and the dashboard build one: its file's metadata
# is read without onnx or onnxruntime.
from cellmap_flow.models.models_config import OnnxModelConfig
entry = OnnxModelConfig(path="web/test/fixtures/head.onnx")
entry.to_dict(), entry.declared_input_norm()
check([], LIGHT + ["bioimageio", "dacapo", "cellmap_models", "fly_organelles", "cellpose", "onnxruntime", "onnx", "torch", "huggingface_hub", "peft"])
""",
    # The dashboard resolves what is pasted into its model form; resolving
    # reads names and files, and loads no model's framework.
    "resolve": """
check(["cellmap_flow.models.resolve"], LIGHT + ["cellmap_flow.models.models_config", "numpy", "torch", "huggingface_hub"])
from cellmap_flow.models.resolve import resolve
resolve("cpsam", voxel_size=8, online=False)
resolve("cellmap/mito", online=False)
resolve("onnx:setup_54", online=False)
check([], LIGHT + ["torch", "huggingface_hub", "peft", "cellpose", "bioimageio", "cellmap_models", "fly_organelles",
                   "onnxruntime"])
""",
    # The trainer, the finetune CLI and the dashboard's routes all read a
    # session's volumes and manifest.
    "finetune-session": """
check([f"cellmap_flow.finetune.session.{m}" for m in ("manifest", "volume", "store", "minio", "sync", "instance")],
      LIGHT + ["torch", "huggingface_hub", "peft"])
""",
    # A finetune run: adaptation, losses, the dataset and the training loop
    # may use torch, never peft or the dashboard. Importing the finetune CLI,
    # which the dashboard imports, leaves the importer's logging alone: it
    # called logging.basicConfig(force=True) at import.
    "finetune-run": """
import logging
check([f"cellmap_flow.finetune.{m}" for m in ("adaptation", "losses", "data", "cli", "run_outputs", "session_loop")],
      LIGHT + ["huggingface_hub", "peft"])
import cellmap_flow.globals
mine = logging.StreamHandler()
logging.root.addHandler(mine)
check(["cellmap_flow.finetune.finetune_cli"], [])
assert mine in logging.root.handlers, "importing finetune_cli reconfigured logging"
""",
    # The dashboard imports blockwise lazily, for the precheck.
    "blockwise-logging": """
import logging
root = logging.getLogger()
root.setLevel(logging.WARNING)
handlers = list(root.handlers)
check(["cellmap_flow.blockwise.blockwise_processor", "cellmap_flow.blockwise.cli",
       "cellmap_flow.blockwise.multiple_cli"], [])
assert (root.level, root.handlers) == (logging.WARNING, handlers)
""",
    # A dashboard request can be the first to import it, off the main thread.
    "launch-signals": """
import signal, threading
before = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
errors = []
thread = threading.Thread(target=lambda: errors.append(check(["cellmap_flow.jobs.launch"], [])))
thread.start()
thread.join()
assert errors == [None], errors
assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == before
""",
    # Plugins run when a command starts, not when the package is imported.
    "plugins": """
import builtins, os, pathlib
plugins = pathlib.Path(os.environ["HOME"], ".cellmap_flow", "plugins")
plugins.mkdir(parents=True)
(plugins / "p.py").write_text("import builtins\\nbuiltins.PLUGIN_RAN = True\\n")
check(["cellmap_flow", "cellmap_flow.cli.main", "cellmap_flow.cli.server_cli"], [])
assert not hasattr(builtins, "PLUGIN_RAN")
from cellmap_flow.cli import main
try:
    main.main(["models"])
except SystemExit:
    pass
assert builtins.PLUGIN_RAN
""",
    # The owners of what a process shares, which servers, jobs and the
    # dashboard all import: no Flask, viewer or torch, and nothing written
    # to HOME. The deprecated g forwards to them but imports them only when
    # a name is used.
    "owners": """
import os
check(["cellmap_flow.jobs.settings", "cellmap_flow.process_chain", "cellmap_flow.dashboard.state"], LIGHT + ["torch"])
assert os.listdir(os.environ["HOME"]) == [], os.listdir(os.environ["HOME"])
""",
    "globals": """
check(["cellmap_flow.globals"], ["cellmap_flow.jobs", "cellmap_flow.process_chain", "cellmap_flow.dashboard", "flask"])
""",
    # Every LSF job imports the package.
    "package-home": """
import os
check(["cellmap_flow"], [])
assert os.listdir(os.environ["HOME"]) == [], os.listdir(os.environ["HOME"])
""",
}


@pytest.mark.parametrize("row", ROWS)
def test_importing_leaves_the_process_alone(tmp_path, row):
    env = {**os.environ, "HOME": str(tmp_path), "PYTHONPATH": ROOT, "MPLBACKEND": "Agg"}
    result = subprocess.run(
        [sys.executable, "-c", PRELUDE + ROWS[row]],
        capture_output=True, text=True, env=env, cwd=ROOT, timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
