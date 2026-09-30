"""The dashboard's Finetune tab: its scripts and the markup they drive.

The tab's scripts are the modules in static/js/dashboard/finetune/, and
they find their elements by id. A lookup that finds nothing only shows up in
a browser, as a button that does nothing, or as a null that throws and takes
the rest of the tab's setup with it.
"""

import re
from pathlib import Path

import cellmap_flow.dashboard

DASHBOARD = Path(cellmap_flow.dashboard.__file__).resolve().parent
TEMPLATE = DASHBOARD / "templates" / "_finetune_tab.html"
MODULES = DASHBOARD / "static" / "js" / "dashboard" / "finetune"
LOOKUP = re.compile(r"""(?:getElementById|\$)\(\s*["']([\w-]+)["']\s*\)""")


def test_every_element_the_finetune_tab_looks_up_is_on_the_page(dashboard):
    page = dashboard.get("/").get_data(as_text=True)
    on_the_page = set(re.findall(r'\bid="([^"]+)"', page))
    looked_up = {name for module in MODULES.glob("*.js") for name in LOOKUP.findall(module.read_text())}
    assert looked_up, "no Finetune tab module found"
    assert sorted(looked_up - on_the_page) == []


def test_the_finetune_tab_markup_has_no_script_and_no_inline_handler():
    # An inline handler runs in the page's global scope, where none of the
    # modules' functions are.
    markup = TEMPLATE.read_text()
    assert "<script" not in markup
    assert re.findall(r"""\son[a-z]+\s*=\s*["']""", markup) == []
