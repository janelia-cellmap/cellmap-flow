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
