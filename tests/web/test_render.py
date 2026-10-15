"""The browser build's page (web/build/render.py): the dashboard's index.html
rendered once with a fresh browser session's state, and the Flask dashboard's
own page, which must not change because the templates learnt the browser
markers."""

import importlib.util
import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

RENDER_PY = Path(__file__).resolve().parents[2] / "web" / "build" / "render.py"

# The elements the browser mode must hide, by id, and those it must keep.
SERVER_ONLY = [
    "review-tab", "review",  # the Review tab's button and pane
    "finetune-tab", "finetune",  # finetuning in the browser is a later PR
    "modelAccordion",  # the Local Catalog: cluster paths
    "zooModelAccordion", "cellposeModelAccordion",
    "serverConfigAccordion", "jobLogsBtn", "jobLogsArea",
]
KEPT = [
    "model-tab", "input-tab", "output-tab", "models", "input", "output",
    "addModelRef",  # Add a model takes a Hugging Face reference
    "hfModelAccordion", "submitModelsBtn", "resampleCheckbox",
    "inputNormList", "postProcessList", "datasetPathInput", "setDataBtn",
    "localFolderGroup", "browseLocalBtn",  # a folder on this machine: the browser-only block of the connect panel
]


@pytest.fixture(scope="module")
def render_py():
    """web/build/render.py as a module. It is a build script, not part of the
    cellmap_flow package, so it is imported by path."""
    spec = importlib.util.spec_from_file_location("cellmap_flow_web_render", RENDER_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def page(render_py):
    """The browser build's page, rendered with the module's defaults."""
    return render_py.render_browser_page()


class _Classes(HTMLParser):
    """For every element with an id, the classes of it and its ancestors, so a
    test can ask whether an element is hidden by a class on it or above it."""

    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}

    def __init__(self, html):
        super().__init__()
        self.stack = []
        self.by_id = {}
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set((attrs.get("class") or "").split())
        inherited = (self.stack[-1] if self.stack else set()) | classes
        if attrs.get("id"):
            self.by_id[attrs["id"]] = inherited
        if tag not in self.VOID:
            self.stack.append(inherited)

    def handle_endtag(self, tag):
        if tag not in self.VOID and self.stack:
            self.stack.pop()


def _rows(html, row_class, checkbox_class):
    """(name, checked, {param: value}) for each Input/Postprocess row, in page order."""
    rows = []
    for block in re.split(rf'<div class="{row_class} ', html)[1:]:
        checkbox = re.search(rf'<input[^>]*class="form-check-input {checkbox_class}"[^>]*>', block, re.S).group(0)
        rows.append((re.search(r'value="([^"]*)"', checkbox).group(1), re.search(r"\schecked\b", checkbox) is not None,
                     dict(re.findall(r'data-param="([^"]*)"\s*value="([^"]*)"', block))))
    return rows


def _page_data(html):
    return json.loads(re.search(r'<script type="application/json" id="page-data">(.*?)</script>', html, re.S).group(1))


def test_the_page_is_the_browser_build(page):
    assert re.search(r'<html\b[^>]*\sdata-mode="browser"', page)
    assert '<script type="module" src="./src/main.ts"></script>' in page
    assert "js/dashboard/main.js" not in page, "the entry loads the dashboard's script itself"
    # The dashboard's static files, relative to the page, where vite serves them.
    assert 'href="./static/css/dark.css"' in page
    assert 'href="./static/img/favicon.ico"' in page
    assert not re.search(r"\burl_for\(|{[{%#]", page), "Jinja left in the page"
    assert "serverConfigModal" not in page, "the LSF first-run dialog is not on the page"


def test_the_page_data_island_parses(page):
    assert _page_data(page) == {"default_hf_repos": [], "default_bioimage_models": [], "resample": False}


def test_the_default_chain_is_ticked_with_its_values(page):
    from cellmap_flow.norm.input_normalize import get_input_normalizers
    from cellmap_flow.post.postprocessors import get_postprocessors_list

    norms = _rows(page, "normalizer-item", "inputNormCheckbox")
    assert norms[:2] == [
        ("MinMaxNormalizer", True, {"min_value": "0.0", "max_value": "255.0", "invert": "False"}),
        ("LambdaNormalizer", True, {"expression": "2*x-1"}),
    ]
    posts = _rows(page, "postprocessor-item", "postProcessCheckbox")
    assert posts[:1] == [("SigmoidPostprocessor", True, {})]
    # Every other op follows once, unticked, so the chain is complete and
    # Submit All sends back exactly the default.
    for rows, available in ((norms[2:], get_input_normalizers()), (posts[1:], get_postprocessors_list())):
        assert not any(checked for _, checked, _ in rows)
        assert sorted(name for name, _, _ in rows) == sorted(
            op["name"] for op in available if op["name"] not in {"MinMaxNormalizer", "LambdaNormalizer", "SigmoidPostprocessor"}
        )


def test_the_browser_page_marks_what_only_a_server_can_serve(page):
    classes = _Classes(page).by_id
    missing = [i for i in SERVER_ONLY + KEPT if i not in classes]
    assert not missing, f"not on the page: {missing}"
    assert all("server-only" in classes[i] for i in SERVER_ONLY), \
        [i for i in SERVER_ONLY if "server-only" not in classes[i]]
    assert not any("server-only" in classes[i] for i in KEPT), \
        [i for i in KEPT if "server-only" in classes[i]]
    # The Pipeline Builder links, and Export Config, which the browser answers.
    assert re.search(r'<a class="[^"]*server-only[^"]*" href="/pipeline-builder"', page)
    assert re.search(r'<a class="action-btn" href="/api/export-config"', page)
    # Hiding the tabs leaves the active one: Input is the one shown at start.
    assert "input-tab" in classes and "active" in classes["input-tab"]
    # The connect panel asks for a URL, not a path on a file system.
    assert re.search(r'id="datasetPathInput"[^>]*placeholder="https://', page)


def test_the_browser_page_renders_an_empty_catalog(page):
    assert 'id="cellpose-data">{"models": [], "running": []}</script>' in page
    assert "chk_" not in page, "no Local Catalog rows"


def test_the_cli_writes_the_page_and_check_tells_a_stale_one(render_py, page, tmp_path, capsys):
    out = tmp_path / "index.html"
    assert render_py.main(["--out", str(out)]) == 0
    assert out.read_text() == page
    assert render_py.main(["--check", "--out", str(out)]) == 0
    out.write_text(page + "<!-- edited -->")
    assert render_py.main(["--check", "--out", str(out)]) == 1
    assert out.read_text().endswith("<!-- edited -->"), "--check writes nothing"
    assert render_py.main(["--check", "--out", str(tmp_path / "absent.html")]) == 1
    capsys.readouterr()
    assert render_py.main(["--out", "-"]) == 0
    assert capsys.readouterr().out == page


def test_the_flask_dashboard_is_unchanged_by_the_markers(dashboard):
    html = dashboard.get("/").get_data(as_text=True)
    assert "data-mode" not in html
    assert re.search(r'<script type="module" src="/static/js/dashboard/main.js"></script>', html)
    assert "src/main.ts" not in html
    # The markers are inert there: the Review tab and the Local Catalog are
    # on the page, and nothing loads browser.css.
    classes = _Classes(html).by_id
    assert "review-tab" in classes and "modelAccordion" in classes
    assert "browser.css" not in html
    # Flask passes no mode, so the connect panel keeps asking for a path.
    assert re.search(r'id="datasetPathInput"[^>]*placeholder="/path/to/dataset.zarr"', html)
    assert "Models run in this browser" not in html
    # The connect panel's browser-only block (a folder on this machine) is not rendered.
    assert "browseLocalBtn" not in html and "localFolderGroup" not in html


def test_both_pages_lay_out_for_a_phone(page, dashboard):
    """The viewport meta tag: without it a phone lays the page out 980px wide
    and shrinks it to fit, and a tap on a field zooms it back in past the
    dashboard. Both pages carry it: the template is one."""
    viewport = '<meta name="viewport" content="width=device-width, initial-scale=1" />'
    assert viewport in page
    assert viewport in dashboard.get("/").get_data(as_text=True)
