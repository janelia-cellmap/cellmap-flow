"""The dashboard's Finetune tab: its scripts and the markup they drive.

The tab's scripts find their elements by id. A lookup that finds nothing
only shows up in a browser, as a button that does nothing, or as a null that
throws and takes the rest of the tab's setup with it.
"""

import re
from pathlib import Path

import cellmap_flow.dashboard

DASHBOARD = Path(cellmap_flow.dashboard.__file__).resolve().parent
TEMPLATE = DASHBOARD / "templates" / "_finetune_tab.html"
MODULES = DASHBOARD / "static" / "js" / "dashboard" / "finetune"
LOOKUP = re.compile(r"""(?:getElementById|\$)\(\s*["']([\w-]+)["']\s*\)""")


def _finetune_scripts():
    inline = re.findall(r"<script\b[^>]*>(.*?)</script>", TEMPLATE.read_text(), re.S)
    return inline + [module.read_text() for module in sorted(MODULES.glob("*.js"))]


def test_every_element_the_finetune_tab_looks_up_is_on_the_page(dashboard):
    page = dashboard.get("/").get_data(as_text=True)
    on_the_page = set(re.findall(r'\bid="([^"]+)"', page))
    looked_up = {name for script in _finetune_scripts() for name in LOOKUP.findall(script)}
    assert looked_up, "no Finetune tab script found"
    assert sorted(looked_up - on_the_page) == []
