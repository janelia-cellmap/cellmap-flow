"""The dashboard's Finetune and Review tabs: their scripts and the markup they drive.

Each tab's scripts are the modules in static/js/dashboard/<tab>/, and they
find their elements by id. A lookup that finds nothing only shows up in a
browser, as a button that does nothing, or as a null that throws and takes
the rest of the tab's setup with it.
"""

import re
from pathlib import Path

import pytest

import cellmap_flow.dashboard

DASHBOARD = Path(cellmap_flow.dashboard.__file__).resolve().parent
TABS = ["finetune", "review"]
LOOKUP = re.compile(r"""(?:getElementById|\$)\(\s*["']([\w-]+)["']\s*\)""")


def _template(tab):
    return DASHBOARD / "templates" / f"_{tab}_tab.html"


def _scripts(tab):
    """The tab's JavaScript: its modules, and any script inline in its template."""
    inline = re.findall(r"<script[^>]*>(.*?)</script>", _template(tab).read_text(), re.S)
    modules = sorted((DASHBOARD / "static" / "js" / "dashboard" / tab).glob("*.js"))
    return inline + [module.read_text() for module in modules]


@pytest.mark.parametrize("tab", TABS)
def test_every_element_a_tab_looks_up_is_on_the_page(dashboard, tab):
    page = dashboard.get("/").get_data(as_text=True)
    on_the_page = set(re.findall(r'\bid="([^"]+)"', page))
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
