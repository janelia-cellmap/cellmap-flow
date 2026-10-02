"""The pipeline builder page: its scripts and the markup they drive.

The builder's scripts find the page's fixed elements by id. A lookup that
finds nothing only shows up in a browser, as a button that does nothing, or
as a null that throws and takes the rest of the page's setup with it.
"""

import re
from pathlib import Path

import cellmap_flow.dashboard

DASHBOARD = Path(cellmap_flow.dashboard.__file__).resolve().parent
TEMPLATE = DASHBOARD / "templates" / "pipeline_builder_v2.html"
MODULES = DASHBOARD / "static" / "js" / "pipeline-builder"
# getElementById, the dialog helpers (dialogs.js), which take a dialog's id,
# and querySelector with an #id.
LOOKUP = re.compile(r"""(?:(?:getElementById|registerDialog|openDialog|closeDialog)\(\s*["']|querySelector(?:All)?\(\s*["']#)([\w-]+)""")
CREATED = re.compile(r"""\.id\s*=\s*["']([\w-]+)["']|\bid:\s*["']([\w-]+)["']""")


def _scripts():
    """The builder's JavaScript: its modules, and any script inline in the template."""
    inline = re.findall(r"<script(?![^>]*\bsrc=)(?![^>]*application/json)[^>]*>(.*?)</script>", TEMPLATE.read_text(), re.S)
    return inline + [module.read_text() for module in sorted(MODULES.glob("*.js"))]


def test_every_element_the_builder_looks_up_is_on_the_page(dashboard):
    page = dashboard.get("/pipeline-builder").get_data(as_text=True)
    on_the_page = set(re.findall(r'\bid="([^"]+)"', page))
    scripts = "\n".join(_scripts())
    looked_up = set(LOOKUP.findall(scripts))
    made_by_the_scripts = {a or b for a, b in CREATED.findall(scripts)}
    assert looked_up, "no builder script found"
    assert sorted(looked_up - made_by_the_scripts - on_the_page) == []


def test_the_builder_markup_has_no_inline_script_and_no_inline_handler():
    # An inline handler runs in the page's global scope, where none of the
    # modules' functions are.
    markup = TEMPLATE.read_text()
    assert re.findall(r"<script(?![^>]*\bsrc=)(?![^>]*application/json)", markup) == []
    assert re.findall(r"""\son[a-z]+\s*=\s*["']""", markup) == []
