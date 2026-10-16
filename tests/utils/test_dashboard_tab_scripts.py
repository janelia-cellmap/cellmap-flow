"""The dashboard's Finetune and Review tabs: their scripts and the markup they drive.

Each tab's scripts are the modules in static/js/dashboard/<tab>/, and they
find their elements by id. A lookup that finds nothing only shows up in a
browser, as a button that does nothing, or as a null that throws and takes
the rest of the tab's setup with it.

The same scripts drive the Flask dashboard and the browser build's page
(web/build/render.py), and each page has elements the other does not: the
browser's Download and Delete buttons, say, are in its browser-mode blocks
only. So an id must be on one of the two pages; which page a lookup runs on
is the script's own business (finetune/index.js starts each part in its mode).
"""

import importlib.util
import re
from pathlib import Path

import pytest

import cellmap_flow.dashboard

DASHBOARD = Path(cellmap_flow.dashboard.__file__).resolve().parent
TABS = ["finetune", "review"]
LOOKUP = re.compile(r"""(?:getElementById|\$)\(\s*["']([\w-]+)["']\s*\)""")
RENDER_PY = Path(__file__).resolve().parents[2] / "web" / "build" / "render.py"


def _template(tab):
    return DASHBOARD / "templates" / f"_{tab}_tab.html"


def _scripts(tab):
    """The tab's JavaScript: its modules, and any script inline in its template."""
    inline = re.findall(r"<script[^>]*>(.*?)</script>", _template(tab).read_text(), re.S)
    modules = sorted((DASHBOARD / "static" / "js" / "dashboard" / tab).glob("*.js"))
    return inline + [module.read_text() for module in modules]


def _browser_page():
    """The browser build's page. web/build/render.py is a build script, not
    part of the cellmap_flow package, so it is imported by path."""
    spec = importlib.util.spec_from_file_location("cellmap_flow_web_render", RENDER_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.render_browser_page()


def _ids(page):
    return set(re.findall(r'\bid="([^"]+)"', page))


@pytest.mark.parametrize("tab", TABS)
def test_every_element_a_tab_looks_up_is_on_the_page(dashboard, tab):
    on_the_page = _ids(dashboard.get("/").get_data(as_text=True)) | _ids(_browser_page())
    looked_up = {name for script in _scripts(tab) for name in LOOKUP.findall(script)}
    assert looked_up, f"no {tab} tab script found"
    assert sorted(looked_up - on_the_page) == []


@pytest.mark.parametrize("tab", TABS)
def test_a_tab_markup_has_no_script_and_no_inline_handler(tab):
    # An inline handler runs in the page's global scope, where none of the
    # modules' functions are.
    markup = _template(tab).read_text()
    assert "<script" not in markup
    assert re.findall(r"""\son[a-z]+\s*=\s*["']""", markup) == []
