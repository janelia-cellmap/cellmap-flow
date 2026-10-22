"""cellmap_flow.portable runs in the browser too, in chunkmirage's Pyodide
workers, so every module under it imports only what Pyodide has: the standard
library, numpy, scipy, pydantic and Pillow (as chunkmirage's
tests/test_core_boundary.py holds its core to a line of its own). In a fresh
interpreter, so nothing this process imported counts, and only what the
imports add is judged."""

import pkgutil
import subprocess
import sys
from pathlib import Path

import cellmap_flow.portable as portable

ALLOWED = {"numpy", "scipy", "pydantic", "pydantic_core", "annotated_types", "typing_extensions", "PIL"}


def test_importing_every_portable_module_pulls_in_nothing_beyond_pyodides_packages():
    modules = [m.name for m in pkgutil.walk_packages([str(Path(portable.__file__).parent)], "cellmap_flow.portable.")]
    assert {"cellmap_flow.portable.rank", "cellmap_flow.portable.carry", "cellmap_flow.portable.calls"} <= set(modules)
    code = (
        "import importlib, sys\n"
        "before = set(sys.modules)\n"
        f"for name in {modules!r}:\n"
        "    importlib.import_module(name)\n"
        "added = set(sys.modules) - before\n"
        f"allowed = set(sys.stdlib_module_names) | {ALLOWED!r}\n"
        "ours = lambda m: m == 'cellmap_flow' or m.startswith('cellmap_flow.portable')\n"
        "print(sorted(m for m in added if m.split('.')[0] not in allowed and not ours(m)))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", f"portable modules import beyond the boundary: {out.stdout}"
