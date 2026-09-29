"""The dashboard's static files: what the package ships, and what the pages load.

The pages' scripts are ES modules served as they are, with no bundler. A
file the package data leaves out, a wrong relative import or a name a module
does not export only shows up in a browser, as a page whose scripts
silently never run.
"""

import glob
import re
import tomllib
from pathlib import Path

import pytest

import cellmap_flow.dashboard
from cellmap_flow.dashboard.app import app

DASHBOARD = Path(cellmap_flow.dashboard.__file__).parent
STATIC = (DASHBOARD / "static").resolve()
REPO = DASHBOARD.parents[1]

IMPORT = re.compile(r"""^\s*(?:import|export)\s+(?:([\w$*{}\s,]+?)\s+from\s+)?["']([^"']+)["']""", re.M)
EXPORTED = re.compile(r"^\s*export\s+(?:async\s+)?(?:function\*?|class|const|let|var)\s+([\w$]+)", re.M)


def _matches(patterns, root):
    return {Path(p) for pat in patterns for p in glob.glob(str(root / pat), recursive=True)}


def test_the_wheel_and_the_sdist_ship_every_dashboard_file():
    pyproject = REPO / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("not running from a source tree")
    config = tomllib.loads(pyproject.read_text())
    package_data = config["tool"]["setuptools"]["package-data"]["cellmap_flow.dashboard"]
    manifest = set()
    for line in (REPO / "MANIFEST.in").read_text().splitlines():
        words = line.split()
        if words[:1] == ["include"]:
            manifest |= _matches(words[1:], REPO)
        elif words[:1] == ["recursive-include"]:
            manifest |= _matches([f"{words[1]}/**/{pat}" for pat in words[2:]], REPO)

    files = [p for d in ("templates", "static") for p in (DASHBOARD / d).rglob("*") if p.is_file()]
    assert files
    assert [str(f) for f in files if f not in _matches(package_data, DASHBOARD)] == []
    assert [str(f) for f in files if f not in manifest] == []


def _imported_names(clause):
    """The names `import { a, b as c } from ...` takes from the other module."""
    clause = clause.strip()
    if not clause.startswith("{"):
        return set()
    return {part.split(" as ")[0].strip() for part in clause.strip("{}").split(",") if part.strip()}


def test_every_module_the_pages_load_resolves_inside_the_package():
    client = app.test_client()
    todo = [
        (STATIC / src).resolve()
        for url in ("/", "/pipeline-builder")
        for src in re.findall(
            r'<script[^>]*type="module"[^>]*src="/static/([^"]+)"',
            client.get(url).get_data(as_text=True),
        )
    ]
    assert todo, "no page loads a module"
    seen = set()
    while todo:
        module = todo.pop()
        if module in seen:
            continue
        seen.add(module)
        assert module.is_file(), f"{module} is loaded but does not exist"
        for clause, spec in IMPORT.findall(module.read_text()):
            target = (module.parent / spec).resolve()
            where = f"{module.relative_to(STATIC)} imports {spec}"
            assert spec.startswith(("./", "../")), f"{where}: only relative imports work unbundled"
            assert target.is_relative_to(STATIC) and target.is_file(), f"{where}: no such file"
            missing = _imported_names(clause) - set(EXPORTED.findall(target.read_text()))
            assert not missing, f"{where}: it does not export {sorted(missing)}"
            todo.append(target)
