"""The dashboard's static files: shipped by the package, and loadable by the pages.

The pages' scripts are ES modules served as they are, with no bundler. A
file the package data leaves out, an import that points nowhere or a name a
module does not export only shows up in a browser, as a page whose scripts
silently never run.
"""

import glob
import re
import tomllib
from pathlib import Path

import cellmap_flow.dashboard
from cellmap_flow.dashboard.app import app

DASHBOARD = Path(cellmap_flow.dashboard.__file__).resolve().parent
STATIC = DASHBOARD / "static"
REPO = DASHBOARD.parents[1]
IMPORT = re.compile(r"""^\s*(?:import|export)\s+(?:([\w$*{}\s,]+?)\s+from\s+)?["']([^"']+)["']""", re.M)
EXPORTED = re.compile(r"^\s*export\s+(?:async\s+)?(?:function\*?|class|const|let|var)\s+([\w$]+)", re.M)


def _globbed(patterns, root):
    return {Path(p).resolve() for pat in patterns for p in glob.glob(str(root / pat), recursive=True)}


def test_the_package_ships_every_dashboard_file_and_every_module_import_resolves():
    config = tomllib.loads((REPO / "pyproject.toml").read_text())
    shipped = _globbed(config["tool"]["setuptools"]["package-data"]["cellmap_flow.dashboard"], DASHBOARD)
    in_sdist = set()
    for words in (line.split() for line in (REPO / "MANIFEST.in").read_text().splitlines()):
        if words[:1] == ["include"]:
            in_sdist |= _globbed(words[1:], REPO)
        elif words[:1] == ["recursive-include"]:
            in_sdist |= _globbed([f"{words[1]}/**/{pat}" for pat in words[2:]], REPO)
    problems = [
        f"{path.relative_to(DASHBOARD)}: missing from the package data or MANIFEST.in"
        for folder in ("templates", "static")
        for path in (DASHBOARD / folder).rglob("*")
        if path.is_file() and not (path in shipped and path in in_sdist)
    ]

    # Every module the two pages load, and everything they import.
    client = app.test_client()
    todo = [
        (STATIC / src).resolve()
        for url in ("/", "/pipeline-builder")
        for src in re.findall(r'<script[^>]*type="module"[^>]*src="/static/([^"]+)"', client.get(url).get_data(as_text=True))
    ]
    assert todo, "no page loads a module"
    seen = set()
    while todo:
        module = todo.pop()
        if module in seen:
            continue
        seen.add(module)
        if not module.is_file():
            problems.append(f"{module.relative_to(STATIC)}: a page loads it, but it does not exist")
            continue
        for clause, spec in IMPORT.findall(module.read_text()):
            target = (module.parent / spec).resolve()
            where = f"{module.relative_to(STATIC)} imports {spec}"
            if not (spec.startswith(("./", "../")) and target.is_relative_to(STATIC) and target.is_file()):
                problems.append(f"{where}: not a file in static/")
                continue
            wanted = {n.split(" as ")[0].strip() for n in clause.strip("{} \n").split(",") if n.strip()}
            missing = wanted - set(EXPORTED.findall(target.read_text())) if clause.strip().startswith("{") else set()
            if missing:
                problems.append(f"{where}: it exports no {sorted(missing)}")
            todo.append(target)
    assert problems == []
