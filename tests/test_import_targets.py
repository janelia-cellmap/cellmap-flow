"""Every `from cellmap_flow... import name` in the package and its tests still resolves.

Code is moved between modules during the cleanup, and many imports are
inside functions (to keep heavy libraries lazy), so a moved name whose
importer was missed only fails when that code path first runs: a trainer or
dashboard route crashing with ImportError long after the tests passed. This
reads every such import statement, wherever it sits, and checks the name.
"""

import ast
import importlib
from pathlib import Path

import cellmap_flow

PACKAGE = Path(cellmap_flow.__file__).parent
TESTS = Path(__file__).parent


def _package_imports():
    # The tests too: a test importing a name that moved fails only when its
    # module is collected, which stops the whole run with one import error.
    for path in sorted([*PACKAGE.rglob("*.py"), *TESTS.rglob("*.py")]):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and not node.level:
                if (node.module or "").startswith("cellmap_flow"):
                    for alias in node.names:
                        yield path.relative_to(PACKAGE.parent), node.lineno, node.module, alias.name


def test_every_imported_name_exists():
    unresolved = []
    for path, line, module, name in _package_imports():
        try:
            target = importlib.import_module(module)
        except ImportError as e:  # the target itself imports something gone
            unresolved.append(f"{path}:{line}: cannot import {module}: {e}")
            continue
        if name == "*" or hasattr(target, name):
            continue
        try:
            importlib.import_module(f"{module}.{name}")
        except ModuleNotFoundError:
            unresolved.append(f"{path}:{line}: from {module} import {name}")
    assert not unresolved, "imports of names that no longer exist:\n" + "\n".join(unresolved)


# Still being moved off g by the agents that own them; emptied by the lead
# once K16-C (fix-dashboard) and fix-loop land.
STILL_IMPORTING_GLOBALS = {
    "cellmap_flow/dashboard/app.py",
    "cellmap_flow/finetune/finetune_cli.py",
}


def _imports_globals(node):
    if isinstance(node, ast.Import):
        return any(alias.name == "cellmap_flow.globals" for alias in node.names)
    if isinstance(node, ast.ImportFrom) and not node.level:
        return node.module == "cellmap_flow.globals" or (
            node.module == "cellmap_flow" and any(alias.name == "globals" for alias in node.names))
    return False


def test_no_module_imports_the_deprecated_globals():
    """Every name g had has an owner to read it from (see globals' docstring).
    Importing globals also configures logging, which reset a CLI's
    --log-level to INFO when a later import pulled it in."""
    importers = set()
    for path in sorted(PACKAGE.rglob("*.py")):
        if path.name == "globals.py" and path.parent == PACKAGE:
            continue
        if any(_imports_globals(node) for node in ast.walk(ast.parse(path.read_text()))):
            importers.add(str(path.relative_to(PACKAGE.parent)))
    unexpected = sorted(importers - STILL_IMPORTING_GLOBALS)
    assert not unexpected, f"these import cellmap_flow.globals: {unexpected}"
