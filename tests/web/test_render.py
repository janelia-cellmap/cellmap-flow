"""The browser build's page (web/build/render.py): the dashboard's index.html
rendered once with a fresh browser session's state, and the Flask dashboard's
own page, which must not change because the templates learnt the browser
markers."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

RENDER_PY = Path(__file__).resolve().parents[2] / "web" / "build" / "render.py"
RUN_PANEL_TS = Path(__file__).resolve().parents[2] / "web" / "src" / "run-region" / "panel.ts"


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


def _page_data(html):
    return json.loads(re.search(r'<script type="application/json" id="page-data">(.*?)</script>', html, re.S).group(1))


def test_the_page_renders_in_both_modes(page, dashboard):
    """The browser's page loads the browser's entry, with the dashboard's
    static files relative to it, where vite serves them; the Flask page loads
    the dashboard's script as before. Both carry the viewport meta tag: without
    it a phone lays the page out 980px wide and shrinks it to fit, and a tap on
    a field zooms it back in past the dashboard."""
    assert re.search(r'<html\b[^>]*\sdata-mode="browser"', page)
    assert '<script type="module" src="./src/main.ts"></script>' in page
    assert "js/dashboard/main.js" not in page, "the entry loads the dashboard's script itself"
    assert 'href="./static/css/dark.css"' in page
    assert not re.search(r"\burl_for\(|{[{%#]", page), "Jinja left in the page"
    assert "serverConfigModal" not in page, "the LSF first-run dialog is not on the page"
    assert 'id="log-panel"' not in page, "no server logs: its stream is a route the browser cannot answer"
    # What only a server can serve is marked for browser.css to hide: the
    # Pipeline Builder, and the Local Catalog of cluster paths, say.
    assert re.search(r'<a class="[^"]*server-only[^"]*" href="/pipeline-builder"', page)
    assert re.search(r'class="[^"]*server-only[^"]*" id="modelAccordion"', page)

    html = dashboard.get("/").get_data(as_text=True)
    assert "data-mode" not in html and "src/main.ts" not in html and "browser.css" not in html
    assert re.search(r'<script type="module" src="/static/js/dashboard/main.js"></script>', html)
    assert _page_data(html)["mode"] == "server"
    assert 'id="log-panel"' in html

    viewport = '<meta name="viewport" content="width=device-width, initial-scale=1" />'
    assert viewport in page and viewport in html


def test_the_page_data_island_parses(page):
    # mode is what the finetune tab's scripts ask (lib/page-data isBrowserMode).
    assert _page_data(page) == {"default_hf_repos": [], "default_bioimage_models": [], "resample": False, "mode": "browser"}


def _rows(html, row_class, checkbox_class):
    """(name, checked, {param: value}) for each row of a Processing tab list, in page order."""
    rows = []
    for block in re.split(rf'<div class="{row_class} ', html)[1:]:
        checkbox = re.search(rf'<input[^>]*class="form-check-input {checkbox_class}"[^>]*>', block, re.S).group(0)
        rows.append((re.search(r'value="([^"]*)"', checkbox).group(1), re.search(r"\schecked\b", checkbox) is not None,
                     dict(re.findall(r'data-param="([^"]*)"\s*value="([^"]*)"', block))))
    return rows


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
    # Every other op the browser runs follows once, unticked, so the chain is
    # complete and Submit All sends back exactly the default; an op only a
    # server runs is not offered.
    for rows, available in ((norms[2:], get_input_normalizers()), (posts[1:], get_postprocessors_list())):
        assert not any(checked for _, checked, _ in rows)
        assert sorted(name for name, _, _ in rows) == sorted(
            op["name"] for op in available
            if op["browser"] and op["name"] not in {"MinMaxNormalizer", "LambdaNormalizer", "SigmoidPostprocessor"}
        )


def test_the_browser_page_keeps_the_ai_annotation_keys_out_of_password_managers(page):
    """The AI panel's credentials block: hidden until the config names a
    provider that needs one, and the secrets text fields masked by CSS that a
    browser's password manager does not offer to save (a password field it
    does, whatever autocomplete says). Pasted credentials are used as they
    are typed and kept in the page only: no Save, no Remember."""
    assert re.search(r'<div class="[^"]*" id="aiAnnotateCredentials" hidden>', page)
    for secret in ("aiAnnotateApiKey", "aiAnnotateAccessToken"):
        field = re.search(rf'<input [^>]*id="{secret}"[^>]*>', page).group(0)
        assert 'type="text"' in field and "credential-secret" in field and 'autocomplete="off"' in field, field
        assert "data-1p-ignore" in field and 'data-lpignore="true"' in field and "data-bwignore" in field, field
    assert 'type="password"' not in page
    assert "aiAnnotateSaveCredsBtn" not in page and "aiAnnotateRememberCreds" not in page


def test_the_run_tab_has_every_element_its_script_looks_up(page, dashboard):
    """The Run tab's script is TypeScript (web/src/run-region/panel.ts), not
    a dashboard module, so tests/utils/test_dashboard_tab_scripts.py does not
    see it: its IDS table must name elements of the browser page, each one
    once, or a button does nothing. The tab is the browser's alone, so none
    is on the Flask page."""
    ids = re.findall(r'^\s+\w+: "([\w-]+)",$', re.search(r"export const IDS = \{(.*?)\} as const;", RUN_PANEL_TS.read_text(), re.S).group(1), re.M)
    assert ids, "web/src/run-region/panel.ts no longer declares its IDS table as one id a line"
    counts = {i: len(re.findall(rf'\bid="{re.escape(i)}"', page)) for i in ids}
    assert {i: n for i, n in counts.items() if n != 1} == {}
    html = dashboard.get("/").get_data(as_text=True)
    assert [i for i in ids if f'id="{i}"' in html] == []


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
