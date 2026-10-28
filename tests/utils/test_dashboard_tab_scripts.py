"""The dashboard's scripts and the markup they drive: the pipeline builder's
page, and the Models, Processing, Finetune and Review tabs of the index page.

The scripts find their pages' fixed elements by id. A lookup that finds
nothing only shows up in a browser, as a button that does nothing, or as a
null that throws and takes the rest of the page's or the tab's setup with it.

Each tab's scripts are the modules in static/js/dashboard/<tab>/, and the
Processing tab's is main.js (initApply), which sets up the rest of the page
too. They drive the Flask dashboard's page, a cluster's or a laptop's, and
the browser build's (web/build/render.py), and each page has elements the
others do not: the browser's Download and Delete buttons, say,
are in its browser-mode blocks only. So an id must be on one of the pages;
which page a lookup runs on is the script's own business (finetune/index.js
starts each part in its mode). The browser's Save predictions tab is
TypeScript, and tests/web/test_render.py checks its ids.
"""

import importlib.util
import re
from pathlib import Path

import pytest

import cellmap_flow.dashboard

DASHBOARD = Path(cellmap_flow.dashboard.__file__).resolve().parent
TEMPLATES = DASHBOARD / "templates"
JS = DASHBOARD / "static" / "js"
RENDER_PY = Path(__file__).resolve().parents[2] / "web" / "build" / "render.py"
# The builder's lookups: getElementById, the dialog helpers (dialogs.js),
# which take a dialog's id, and querySelector with an #id.
BUILDER_LOOKUP = re.compile(
    r"""(?:(?:getElementById|registerDialog|openDialog|closeDialog)\(\s*["']|querySelector(?:All)?\(\s*["']#)([\w-]+)""")
CREATED = re.compile(r"""\.id\s*=\s*["']([\w-]+)["']|\bid:\s*["']([\w-]+)["']""")
TAB_LOOKUP = re.compile(r"""(?:getElementById|\$)\(\s*["']([\w-]+)["']\s*\)""")
TABS = {
    "models": sorted((JS / "dashboard" / "models").glob("*.js")),
    "processing": [JS / "dashboard" / "main.js"],
    "finetune": sorted((JS / "dashboard" / "finetune").glob("*.js")),
    "review": sorted((JS / "dashboard" / "review").glob("*.js")),
}


def _ids(html):
    return set(re.findall(r'\bid="([^"]+)"', html))


@pytest.fixture
def ids_on_the_pages(dashboard, monkeypatch):
    """Every id on a page the tabs' scripts run on: the Flask dashboard's, on
    a cluster and on a laptop, and the browser build's. render.py is a build
    script, not part of the cellmap_flow package, so it is imported by path."""
    from cellmap_flow.dashboard.routes import index_page

    spec = importlib.util.spec_from_file_location("cellmap_flow_web_render", RENDER_PY)
    render = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(render)
    ids = _ids(render.render_browser_page())
    for cluster in (True, False):
        monkeypatch.setattr(index_page, "bsub_available", lambda: cluster)
        ids |= _ids(dashboard.get("/").get_data(as_text=True))
    return ids


def test_every_element_the_builder_looks_up_is_on_its_page(dashboard):
    """Or is made by its scripts: a node's or a dialog's elements, say."""
    page = dashboard.get("/pipeline-builder").get_data(as_text=True)
    scripts = "\n".join(module.read_text() for module in sorted((JS / "pipeline-builder").glob("*.js")))
    looked_up = set(BUILDER_LOOKUP.findall(scripts))
    made_by_the_scripts = {a or b for a, b in CREATED.findall(scripts)}
    assert looked_up, "no builder script found"
    assert sorted(looked_up - made_by_the_scripts - _ids(page)) == []


@pytest.mark.parametrize("tab", TABS)
def test_every_element_a_tab_looks_up_is_on_a_page(ids_on_the_pages, tab):
    """The Models tab's Loaded models list among them, which the browser's
    page keeps too."""
    looked_up = {name for module in TABS[tab] for name in TAB_LOOKUP.findall(module.read_text())}
    assert looked_up, f"no {tab} tab script found"
    assert sorted(looked_up - ids_on_the_pages) == []


@pytest.mark.parametrize("template", ["pipeline_builder_v2", *(f"_{tab}_tab" for tab in TABS)])
def test_the_markup_has_no_inline_script_and_no_inline_handler(template):
    """So every script the lookups above read is a module. An inline handler
    runs in the page's global scope, where none of the modules' functions
    are; a script of data (application/json) is no script."""
    markup = (TEMPLATES / f"{template}.html").read_text()
    assert re.findall(r"<script(?![^>]*\bsrc=)(?![^>]*application/json)", markup) == []
    assert re.findall(r"""\son[a-z]+\s*=\s*["']""", markup) == []
