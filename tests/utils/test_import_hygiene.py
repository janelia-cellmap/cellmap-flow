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
    "jobs": """
check([f"cellmap_flow.jobs.{m}" for m in ("spec", "site", "settings", "lsf", "local", "queues", "ready", "launch")]
      + ["cellmap_flow.finetune.markers"] + [f"cellmap_flow.finetune.job_manager.{m}" for m in ("state", "tailer", "listener", "submit", "persistence", "restart", "monitor", "manager")],
      LIGHT + ["huggingface_hub", "peft", "torch"])
""",
    "io": """
check([f"cellmap_flow.io.{m}" for m in ("paths", "metadata", "multiscale", "ome", "geometry", "source")],
      LIGHT + ["torch", "huggingface_hub", "peft"])
""",
    # torch only once a runner is built.
    "serving": """
check(["cellmap_flow.models.geometry", "cellmap_flow.models.geometry_cache", "cellmap_flow.inference.runner"]
      + [f"cellmap_flow.serving.{m}" for m in ("virtual_zarr", "protocol", "client", "probe", "restart_token")],
      LIGHT + ["torch"])
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
check([], LIGHT + ["bioimageio", "dacapo", "cellmap_models", "fly_organelles", "torch", "huggingface_hub", "peft"])
""",
    # A chain is read wherever one is listed; its steps import what they need.
    "chain": """
check(["cellmap_flow.pipeline_spec", "cellmap_flow.process_chain"], LIGHT + ["torch", "huggingface_hub", "peft",
      "cellmap_flow.norm.input_normalize", "cellmap_flow.post.postprocessors"])
from cellmap_flow.post.postprocessors import get_postprocessors, get_postprocessors_list
get_postprocessors_list()
get_postprocessors([{"name": "SigmoidPostprocessor"}, {"name": "AffinityPostprocessor"}])
check([], ["neuroglancer", "pymorton", "mwatershed", "fastremap", "fastmorph", "scipy.ndimage"])
""",
    "review": """
check(["cellmap_flow.review", "cellmap_flow.review_index"], LIGHT + ["torch"])
""",
    # A viewer never starts the dashboard. (globals still comes in with the
    # raw layer, through ImageDataInterface.)
    "viewer": """
check(["cellmap_flow.viewer.layers", "cellmap_flow.viewer.bootstrap"],
      ["flask", "cellmap_flow.dashboard", "torch", "huggingface_hub", "peft"])
""",
    # The dashboard imports blockwise lazily, for the precheck.
    "blockwise-logging": """
import logging
import cellmap_flow.globals
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
