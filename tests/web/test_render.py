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
    "modelAccordion",  # the Local Catalog: cluster paths
    "zooModelAccordion", "cellposeModelAccordion",
    "serverConfigAccordion", "jobLogsBtn", "jobLogsArea",
    # The Finetune tab's parts that need a server: a path on its file system,
    # crops from its disks, the patch tools that write the volume through its
    # routes (seeds from a model's server), MinIO, and the cluster trainer's
    # parameters (its batch of patches, its rehearsal share, the distance
    # models' interval loss). AI-assisted annotation and good regions are not
    # among them: the browser answers their routes itself (KEPT below).
    "outputPath", "openLoadCropsBtn", "loadCropsModal",
    "labelItSection",
    "saveAnnotationsBtn", "minioStatus", "existingBrowseRoot",
    "checkpointPath", "loraRank", "batchSize", "patchesPerEpoch", "rehearsalFraction",
    "distanceModelHint", "augment", "gpuQueue",
]
KEPT = [
    "model-tab", "input-tab", "output-tab", "models", "input", "output",
    "addModelRef",  # Add a model takes a Hugging Face reference
    "hfModelAccordion", "submitModelsBtn", "resampleCheckbox",
    "inputNormList", "postProcessList", "datasetPathInput", "setDataBtn",
    "localFolderGroup", "browseLocalBtn",  # a folder on this machine: the browser-only block of the connect panel
    "shareLinkBtn",  # the header's Share button: a link to what is open, browser-only too
    # The Finetune tab, which trains the model's last layer in the page: a
    # volume to paint, the head's parameters, the job's card, plot and log.
    "finetune-tab", "finetune",
    "modelSelect", "createVolumeBtn", "loadExistingVolumeBtn", "existingSessionsList",
    "showAnnotatedRegionsBtn",  # the browser counts the painted chunks
    "numEpochs", "learningRate", "labelSmoothing", "balanceClasses", "autoServeCheck", "startFinetuningBtn",
    # The head's loss (its own options), its margin, and the distillation it is held by.
    "lossType", "marginGroup", "distillationLambda", "distillationScope",
    "trainingStatusCard", "lossPlotCanvas", "trainingLogs",
    # Good regions: marked, counted and cleared in the page, held by the head's distillation.
    "goodRegionsSection", "markGoodRegionBtn", "goodRegionCount", "clearGoodRegionsBtn",
    # Its browser-only controls: the paint out as a zarr or deleted, the
    # finetuned model out as .onnx, the head's batch of voxels.
    "downloadVolumeBtn", "deleteVolumeBtn", "downloadFinetunedBtn", "batchPixels",
    # AI-assisted annotation, in both modes: the browser calls the provider itself.
    "aiAnnotatePanel", "aiAnnotateReviewModal",
]
# The AI panel's browser-only parts (the template's browser-mode blocks): the
# credentials block, the AI section's own Undo (the server's sits in the
# server-only Label it grid), and the model id typed under Other…. On the
# browser page, never on the Flask page.
AI_ANNOTATE_BROWSER_ONLY = [
    "aiAnnotateCredentials", "aiAnnotateCredsState", "aiAnnotateApiKey", "aiAnnotateProject", "aiAnnotateAccessToken",
    "aiAnnotateForgetCredsBtn", "aiAnnotateCredsHint",
    "aiAnnotateUndoBtn", "aiAnnotateModelName",
]
KEPT += AI_ANNOTATE_BROWSER_ONLY


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
    # mode is what the finetune tab's scripts ask (lib/page-data isBrowserMode).
    assert _page_data(page) == {"default_hf_repos": [], "default_bioimage_models": [], "resample": False, "mode": "browser"}


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


def _select_options(html, select_id):
    """[(value, selected)] of a <select>'s options, in page order."""
    select = re.search(rf'<select[^>]*id="{select_id}"[^>]*>(.*?)</select>', html, re.S).group(1)
    return [(value, re.search(r"\sselected\b", option) is not None)
            for option, value in re.findall(r'(<option[^>]*value="([^"]*)"[^>]*>)', select)]


def test_the_browser_page_finetunes_the_last_layer(page):
    """What the Finetune tab says and offers in the browser: the head trains,
    at its own learning rates, and the paint and the model can be taken out."""
    assert "trains the model's last layer (its output convolution)" in page
    assert "train the model's last layer in this tab" in page
    assert "volumes painted in this browser, newest first" in page
    # The head's learning rates, 1e-2 chosen, in place of the LoRA rates.
    assert _select_options(page, "learningRate") == [("0.01", True), ("0.003", False), ("0.001", False)]
    # The downloads are links the service worker relays, with no target until
    # a volume or a finetuned model exists; Delete waits for a volume too.
    assert re.search(r'<a class="[^"]*\bdisabled\b[^"]*" id="downloadVolumeBtn"[^>]*\bdownload\b', page)
    assert re.search(r'<button[^>]*id="deleteVolumeBtn"[^>]*\bdisabled\b', page)
    assert re.search(r'<a class="[^"]*" id="downloadFinetunedBtn"[^>]*\bdownload\b[^>]*style="display: none;"', page)


def test_the_browser_page_offers_the_heads_training_controls(page):
    """The head's own choices: the server's per-voxel losses, BCE chosen (no
    Dice, which scores whole patches); its batch in voxels, 4,096 chosen; and
    the distillation weight and scope, the server form's defaults."""
    assert _select_options(page, "lossType") == [("bce", True), ("mse", False), ("margin", False)]
    assert _select_options(page, "batchPixels") == [("1024", False), ("4096", True), ("16384", False), ("65536", False)]
    assert ("0.01", True) in _select_options(page, "distillationLambda")
    assert _select_options(page, "distillationScope") == [("unlabeled", True), ("all", False)]
    assert "Good regions, once marked, decide instead." in page


def test_the_browser_page_takes_the_ai_annotation_credentials(page):
    """The AI panel's credentials block: hidden until the config names a
    provider that needs one, the secrets password fields the browser must not
    autofill, the project a plain field, and Undo disabled until there is
    something to take back."""
    assert re.search(r'<div class="[^"]*" id="aiAnnotateCredentials" hidden>', page)
    for secret in ("aiAnnotateApiKey", "aiAnnotateAccessToken"):
        assert re.search(rf'<input type="password"[^>]*id="{secret}"[^>]*autocomplete="off"', page), secret
    assert re.search(r'<input type="text"[^>]*id="aiAnnotateProject"[^>]*autocomplete="off"', page)
    assert re.search(r'<input type="text"[^>]*id="aiAnnotateModelName"[^>]*\bhidden\b', page)
    assert re.search(r'<button[^>]*id="aiAnnotateUndoBtn"[^>]*\bdisabled\b', page)
    # Pasted credentials are used as they are typed and kept in the page only: no Save, no Remember.
    assert "aiAnnotateSaveCredsBtn" not in page and "aiAnnotateRememberCreds" not in page
    # The pickers' titles are the browser's, not the server's configuration's.
    assert "Only the providers in the server's AI-annotate configuration are listed." not in page
    assert "Other… takes a model id that is not listed" in page


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
    # Every marked element of the Finetune tab is on the page too.
    assert all(i in classes for i in SERVER_ONLY), [i for i in SERVER_ONLY if i not in classes]
    # Flask passes no mode, so the connect panel keeps asking for a path.
    assert re.search(r'id="datasetPathInput"[^>]*placeholder="/path/to/dataset.zarr"', html)
    assert "Models run in this browser" not in html
    # The connect panel's browser-only block (a folder on this machine) is not rendered.
    assert "browseLocalBtn" not in html and "localFolderGroup" not in html
    assert "shareLinkBtn" not in html
    # Nor the Finetune tab's: its page-data says server, its trainer is the
    # cluster's at LoRA's learning rates, and nothing is downloaded or deleted.
    assert _page_data(html)["mode"] == "server"
    assert "train on the GPU cluster" in html
    assert "train the model's last layer in this tab" not in html
    assert "trains the model's last layer (its output convolution)" not in html
    assert _select_options(html, "learningRate")[3] == ("0.0001", True) and "0.003" not in html
    # The cluster trainer's losses, margin chosen, and no batch of voxels.
    assert _select_options(html, "lossType") == [("margin", True), ("mse", False), ("bce", False), ("dice", False), ("combined", False)]
    for browser_only in ("downloadVolumeBtn", "deleteVolumeBtn", "downloadFinetunedBtn", "existingVolumesNote", "batchPixels", *AI_ANNOTATE_BROWSER_ONLY):
        assert browser_only not in html, browser_only
    # The AI panel and its review are on the Flask page as before, with the
    # server's words on the pickers and without the browser's additions.
    assert "aiAnnotatePanel" in classes and "aiAnnotateReviewModal" in classes
    assert "Only the providers in the server's AI-annotate configuration are listed." in html
    assert "The provider's models that the server's configuration allows." in html


def test_both_pages_lay_out_for_a_phone(page, dashboard):
    """The viewport meta tag: without it a phone lays the page out 980px wide
    and shrinks it to fit, and a tap on a field zooms it back in past the
    dashboard. Both pages carry it: the template is one."""
    viewport = '<meta name="viewport" content="width=device-width, initial-scale=1" />'
    assert viewport in page
    assert viewport in dashboard.get("/").get_data(as_text=True)
