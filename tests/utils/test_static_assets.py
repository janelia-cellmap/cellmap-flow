"""The dashboard's static files: loadable by the pages.

The pages' scripts are ES modules served as they are, with no bundler. An
import that points nowhere or a name a module does not export only shows up
in a browser, as a page whose scripts silently never run. (That the wheel
ships every file is CI's package job.)
"""

import re
from pathlib import Path

import cellmap_flow.dashboard
from cellmap_flow.dashboard.app import app

DASHBOARD = Path(cellmap_flow.dashboard.__file__).resolve().parent
STATIC = DASHBOARD / "static"
IMPORT = re.compile(r"""^\s*(?:import|export)\s+(?:([\w$*{}\s,]+?)\s+from\s+)?["']([^"']+)["']""", re.M)
EXPORTED = re.compile(r"^\s*export\s+(?:async\s+)?(?:function\*?|class|const|let|var)\s+([\w$]+)", re.M)
EXPORT_LIST = re.compile(r"^\s*export\s*\{([^}]*)\}", re.M)  # export { a, b as c } (the vendored js-yaml)


def _exports(source):
    names = set(EXPORTED.findall(source))
    for items in EXPORT_LIST.findall(source):
        names |= {item.split(" as ")[-1].strip() for item in items.split(",") if item.strip()}
    return names


def test_every_module_the_pages_load_and_every_import_resolves():
    """Every module the two pages load, and everything they import."""
    problems = []
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
            missing = wanted - _exports(target.read_text()) if clause.strip().startswith("{") else set()
            if missing:
                problems.append(f"{where}: it exports no {sorted(missing)}")
            todo.append(target)
    assert problems == []
