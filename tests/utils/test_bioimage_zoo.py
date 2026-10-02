"""The BioImage Model Zoo on the Models tab: the list read from the zoo's
index (models.bioimage_catalog), its cache, the routes that serve it, and
Submit launching what is ticked as a ``bioimage`` model. The index is
faked: no test touches the network."""

import functools
import io
import json
import re
import shlex
import textwrap
import urllib.error
from types import SimpleNamespace

import pytest

from cellmap_flow.dashboard.requests import BioimageSelection
from cellmap_flow.dashboard.services import launch
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.models import bioimage_catalog as catalog
from cellmap_flow.models.models_config import BioModelConfig
from cellmap_flow.process_chain import process_chain

DOI = "10.5281/zenodo.5874841"
INDEX = {"collection": [
    {"id": DOI, "nickname": "kind-seashell", "type": "model", "name": "MitochondriaEMSegmentationBoundaryModel",
     "description": "Mitochondria segmentation for\n electron microscopy.", "license": "CC-BY-4.0",
     "tags": ["unet", "electron-microscopy", "3d", "pytorch"], "covers": ["https://zoo/cover.png"]},
    # No nickname, and EM only by its name's acronym; "2D" in capitals.
    {"id": "humorous-fox", "type": "model", "name": "SEM_N2V", "description": "Denoising.",
     "tags": ["denoising", "2D", "Noise2Void"]},
    # Dimensions from the name; "semantic" is not SEM; formats from the RDF's weights.
    {"id": "emotional-cricket", "type": "model", "name": "3D UNet Arabidopsis",
     "description": "Semantic segmentation of nuclei", "tags": ["semantic-segmentation"],
     "weights": {"onnx": {}, "pytorch_state_dict": {}}},
    # Both 2D and 3D: neither filter claims it.
    {"id": "philosophical-panda", "type": "model", "name": "Cellpose Plant Nuclei", "tags": ["2d", "3d"]},
    {"id": "some-dataset", "type": "dataset", "name": "EM data"},
]}

MANIFEST = """
[pypi-dependencies]
cellmap-flow = { path = ".", editable = true }

[feature.bioimageio.pypi-dependencies]
bioimageio-core = "==0.11.0"

[environments]
bioimageio = { features = ["bioimageio"] }
"""


@pytest.fixture(autouse=True)
def zoo(tmp_path, monkeypatch):
    """The fake legacy index, served from a cache under tmp_path; ``zoo.fetches``
    counts its downloads, and ``zoo.index`` is what the next one gets.
    bioimage.io's artifact server answers ``zoo.hypha``: by default it is
    unreachable, so the index is used, as when the server is down."""
    state = SimpleNamespace(fetches=0, index=INDEX, hypha=catalog.ZooIndexError("unreachable"),
                            real_fetch=catalog._fetch_index)

    def fetch(url):
        if url == catalog.HYPHA_MODELS_URL:
            if isinstance(state.hypha, Exception):
                raise state.hypha
            return state.hypha
        state.fetches += 1
        if isinstance(state.index, Exception):
            raise state.index
        return state.index

    monkeypatch.setattr(catalog, "BIOIMAGE_CACHE_FILE", str(tmp_path / "bioimage" / "models_cache.json"))
    monkeypatch.setattr(catalog, "_fetch_index", fetch)
    return state


class _Before:
    """BioModelConfig's arguments before the bio overhaul."""

    def __init__(self, model_name, voxel_size, edge_length_to_process=None, name=None, scale=None):
        pass


class _After:
    """And after it: the model is ``model``, its voxel size read from its RDF."""

    def __init__(self, model, voxel_size=None, name=None, scale=None):
        pass


def test_the_index_is_normalised_to_its_models():
    models = catalog.list_bioimage_models()["models"]
    assert [m["key"] for m in models] == ["kind-seashell", "humorous-fox", "emotional-cricket", "philosophical-panda"]
    assert [m["em"] for m in models] == [True, True, False, False]
    assert [m["dims"] for m in models] == ["3d", "2d", "3d", None]
    assert [m["weight_formats"] for m in models] == [["pytorch"], [], ["onnx", "pytorch"], []]
    first = models[0]
    assert (first["id"], first["nickname"], first["license"], first["cover"]) == (
        DOI, "kind-seashell", "CC-BY-4.0", "https://zoo/cover.png")
    assert first["description"] == "Mitochondria segmentation for electron microscopy."
    assert first["url"] == "https://bioimage.io/#/artifacts/kind-seashell"
    assert models[1]["nickname"] is None and models[3]["description"] == ""


def test_the_list_is_cached_until_refreshed_and_a_failed_fetch_keeps_it(zoo):
    catalog.list_bioimage_models()
    assert catalog.list_bioimage_models()["fetched"] and zoo.fetches == 1
    zoo.index = {"collection": INDEX["collection"][:1]}
    assert len(catalog.refresh_bioimage_models()["models"]) == 1 and zoo.fetches == 2
    zoo.index = catalog.ZooIndexError("offline")
    with pytest.raises(catalog.ZooIndexError):
        catalog.refresh_bioimage_models()
    assert len(catalog.list_bioimage_models()["models"]) == 1


def test_a_download_that_fails_says_where_from(zoo, monkeypatch):
    def unreachable(request, timeout):
        assert timeout == catalog.FETCH_TIMEOUT_S
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(catalog.urllib.request, "urlopen", unreachable)
    with pytest.raises(catalog.ZooIndexError, match=re.escape(catalog.ZOO_INDEX_URL) + ".*no route to host"):
        zoo.real_fetch(catalog.ZOO_INDEX_URL)


def test_a_model_is_found_by_id_or_nickname_in_the_cache_only(zoo):
    assert catalog.find_bioimage_model("kind-seashell") is None and zoo.fetches == 0
    catalog.list_bioimage_models()
    assert catalog.find_bioimage_model(DOI)["key"] == catalog.find_bioimage_model("kind-seashell")["key"]
    assert catalog.find_bioimage_model("no-such-model") is None


def test_the_entry_names_the_model_as_the_config_class_does():
    assert catalog.bioimage_entry("kind-seashell", [8, 8, 8], "ks", cls=_Before) == {
        "model_name": "kind-seashell", "voxel_size": [8, 8, 8], "name": "ks"}
    with pytest.raises(ValueError, match="needs a voxel size"):
        catalog.bioimage_entry("kind-seashell", None, "ks", cls=_Before)
    assert catalog.bioimage_entry("kind-seashell", None, "ks", cls=_After) == {"model": "kind-seashell", "name": "ks"}
    assert catalog.bioimage_entry("kind-seashell", [8, 8, 8], cls=_After) == {
        "model": "kind-seashell", "voxel_size": [8, 8, 8]}
    # The real class builds from it, and gives the model back.
    config = BioModelConfig(**catalog.bioimage_entry("kind-seashell", [8, 8, 8], "ks"))
    assert catalog.entry_model_id(config) == "kind-seashell"


@pytest.mark.parametrize("typed, sizes", [
    ("8", [8, 8, 8]), ("4, 4,8", [4, 4, 8]), ("4 4 8", [4, 4, 8]), (8.5, [8.5] * 3), ([4, 4, 8], [4, 4, 8]),
    ("", None), (None, None),
])
def test_a_typed_voxel_size_is_read_as_z_y_x(typed, sizes):
    assert BioimageSelection(id="kind-seashell", voxel_size=typed).voxel_size == sizes


def test_the_routes_list_and_refresh_the_zoo_and_say_why_when_it_is_unreachable(dashboard, zoo):
    listed = dashboard.get("/api/bioimage-models")
    assert listed.status_code == 200 and len(listed.get_json()["models"]) == 4
    zoo.index = {"collection": []}
    assert dashboard.post("/api/bioimage-models/refresh").get_json()["models"] == [] and zoo.fetches == 2
    zoo.index = catalog.ZooIndexError("Could not fetch the BioImage Model Zoo index: offline")
    failed = dashboard.post("/api/bioimage-models/refresh")
    assert failed.status_code == 502 and "offline" in failed.get_json()["error"]


class _Job:
    def __init__(self, name):
        self.model_name, self.host, self.killed = name, f"http://{name}:1", False

    def kill(self):
        self.killed = True


class _InlineThread:
    def __init__(self, target, args=()):
        self._target, self._args = target, args

    def start(self):
        self._target(*self._args)


@pytest.fixture
def submit(dashboard, viewer, monkeypatch, tmp_path):
    """POST /api/models with these zoo models ticked; the commands started are ``submit.commands``."""
    commands = []
    monkeypatch.setattr(launch, "start_hosts", lambda command, *args, **kwargs: commands.append(command))
    monkeypatch.setattr(launch.threading, "Thread", _InlineThread)
    monkeypatch.setenv("CELLMAP_FLOW_ENVS_FILE", str(tmp_path / "no-aliases.yaml"))
    process_chain().input_norms, process_chain().postprocess = [], []
    get_session().dataset_path = "/data/raw.zarr"
    catalog.list_bioimage_models()

    def post(*selections):
        return dashboard.post("/api/models", json={"selected_bioimage_models": list(selections)})

    post.commands = commands
    return post


def _served_entry(command):
    argv = shlex.split(command)
    return argv, json.loads(argv[argv.index("--model") + 1])


def test_submit_serves_a_ticked_zoo_model_from_the_bioimageio_environment(submit, tmp_path, monkeypatch):
    manifest = tmp_path / "checkout" / "pixi.toml"
    manifest.parent.mkdir()
    manifest.write_text(textwrap.dedent(MANIFEST))
    monkeypatch.setenv("CELLMAP_FLOW_PIXI_MANIFEST", str(manifest))
    monkeypatch.setenv("PIXI_EXE", "/opt/pixi")

    # Ticked by its DOI, loaded by its nickname.
    assert submit({"id": DOI, "voxel_size": "8"}).status_code == 200
    (command,) = submit.commands
    argv, entry = _served_entry(command)
    assert argv[:7] == ["/opt/pixi", "run", "--frozen", "--manifest-path", str(manifest), "-e", "bioimageio"]
    assert entry == {"type": "bioimage", **catalog.bioimage_entry("kind-seashell", [8, 8, 8], "kind_seashell")}
    assert [mc.name for mc in get_session().models_config] == ["kind_seashell"]


def test_a_zoo_model_without_a_voxel_size_its_class_needs_refuses_the_submit(submit, monkeypatch):
    monkeypatch.setattr(catalog, "bioimage_entry", functools.partial(catalog.bioimage_entry, cls=_Before))
    monkeypatch.setattr(catalog, "declared_voxel_size", lambda entry: [8.0, 8.0, 8.0])
    running = _Job("mito")
    get_session().jobs = [running]
    answer = submit({"id": "kind-seashell"})
    assert answer.status_code == 400 and "needs a voxel size" in answer.get_json()["error"]
    assert submit.commands == [] and not running.killed


def test_unticking_a_zoo_model_stops_it_and_a_running_one_is_not_started_again(submit):
    zoo_job = _Job("kind_seashell")
    get_session().jobs = [zoo_job]
    submit({"id": "kind-seashell", "voxel_size": [8, 8, 8]})
    assert submit.commands == [] and not zoo_job.killed
    submit()
    assert zoo_job.killed and get_session().jobs == []


def test_the_page_ticks_the_running_zoo_models_with_their_voxel_size(submit, dashboard):
    submit({"id": "kind-seashell", "voxel_size": "4,4,8"}, {"id": "humorous-fox", "voxel_size": "8"})
    get_session().jobs = [_Job("kind_seashell")]  # humorous_fox's job has gone
    html = dashboard.get("/").get_data(as_text=True)
    page = json.loads(re.search(r'<script type="application/json" id="page-data">(.*?)</script>', html, re.S).group(1))
    assert page["default_bioimage_models"] == [{"id": "kind-seashell", "voxel_size": [4, 4, 8]}]


def _rdf(axes):
    return ("inputs:\n  - id: raw\n    axes:\n" + "".join(
        f"      - {{id: {a}, type: {kind}" + (f", scale: {scale}, unit: {unit}" if unit else "") + "}\n"
        for a, kind, scale, unit in axes))


@pytest.mark.parametrize("axes, declared", [
    pytest.param([("batch", "batch", 1, None), ("z", "space", 1, None), ("y", "space", 1, None),
                  ("x", "space", 1, None)], None, id="no-units-like-the-zoo-EM-models"),
    pytest.param([("z", "space", 0.04, "micrometer"), ("y", "space", 8, "nanometer"), ("x", "space", 8, "nanometer")],
                 [40.0, 8.0, 8.0], id="3d-in-mixed-units"),
    pytest.param([("y", "space", 0.5, "micrometer"), ("x", "space", 0.5, "micrometer")],
                 [500.0, 500.0, 500.0], id="2d-z-is-its-y"),
])
def test_the_declared_voxel_size_is_read_from_the_models_description(monkeypatch, axes, declared):
    monkeypatch.setattr(catalog, "_declared", {})
    monkeypatch.setattr(catalog.urllib.request, "urlopen",
                        lambda request, timeout: io.BytesIO(_rdf(axes).encode()))
    assert catalog.declared_voxel_size({"key": "m", "rdf_source": "https://zoo/m/rdf.yaml"}) == declared


def test_a_blank_voxel_size_for_a_model_that_declares_none_refuses_the_submit(submit, monkeypatch):
    """Its server would refuse it after its job started, where the page showed
    nothing: Submit says so instead."""
    monkeypatch.setattr(catalog, "declared_voxel_size", lambda entry: None)
    answer = submit({"id": "kind-seashell"})
    assert answer.status_code == 400 and "enter one (nm) in its row" in answer.get_json()["error"]
    assert submit.commands == []
    # Given one, it starts.
    assert submit({"id": "kind-seashell", "voxel_size": "8"}).status_code == 200 and len(submit.commands) == 1


def test_a_blank_voxel_size_refuses_the_submit_when_the_description_cannot_be_read(submit, monkeypatch):
    """Its server could not read it either. This let impartial-shrimp through
    from a cache written before entries carried their description's link."""
    def unreadable(entry):
        raise catalog.ZooIndexError("offline")

    monkeypatch.setattr(catalog, "declared_voxel_size", unreadable)
    answer = submit({"id": "kind-seashell"})
    assert answer.status_code == 400 and "offline" in answer.get_json()["error"] and submit.commands == []
    assert submit({"id": "kind-seashell", "voxel_size": "8"}).status_code == 200 and len(submit.commands) == 1


def test_a_cache_in_an_older_format_is_fetched_again_and_still_read_when_that_fails(zoo):
    """One written by an older cellmap-flow lacked the entries' description links."""
    catalog.list_bioimage_models()
    with open(catalog.BIOIMAGE_CACHE_FILE) as f:
        document = json.load(f)
    del document["format"]
    with open(catalog.BIOIMAGE_CACHE_FILE, "w") as f:
        json.dump(document, f)
    assert catalog.list_bioimage_models()["format"] == catalog.CACHE_FORMAT and zoo.fetches == 2
    assert catalog.list_bioimage_models() and zoo.fetches == 2

    del document["models"][1:]
    with open(catalog.BIOIMAGE_CACHE_FILE, "w") as f:
        json.dump(document, f)
    zoo.index = catalog.ZooIndexError("offline")
    assert len(catalog.list_bioimage_models()["models"]) == 1


HYPHA_LISTING = [
    {"alias": "impartial-shrimp", "download_count": 900, "manifest": {
        "id": "impartial-shrimp", "name": "Neuron Segmentation in EM (Membrane Prediction)", "type": "model",
        "description": "Membranes in electron microscopy", "tags": ["electron-microscopy", "3d"],
        "covers": ["cover.thumbnail.jpg"], "license": "MIT", "weights": {"torchscript": {}, "pytorch_state_dict": {}},
        "inputs": [{"id": "input0", "axes": "bczyx"}]}},
    {"alias": "affable-shark", "manifest": {
        "id": "affable-shark", "name": "Nuclei", "type": "model", "description": "Fluorescence nuclei",
        "tags": ["fluorescence"], "inputs": [{"axes": [
            {"type": "batch"}, {"type": "channel", "id": "channel"},
            {"type": "space", "id": "y", "scale": 0.25, "unit": "micrometer"},
            {"type": "space", "id": "x", "scale": 0.25, "unit": "micrometer"}]}]}},
    {"alias": "no-description"},
]


def test_the_models_come_from_bioimageios_server_with_what_their_descriptions_say(zoo):
    """Its listing holds each description: 2D or 3D from the input's axes, and
    any declared voxel size without a fetch per model. The legacy index
    lagged it by 32 models."""
    zoo.hypha = HYPHA_LISTING
    document = catalog.refresh_bioimage_models()
    shrimp, shark = document["models"]
    assert document["source"] == "hypha" and zoo.fetches == 0
    assert (shrimp["key"], shrimp["dims"], shrimp["em"], shrimp["weight_formats"], shrimp["declared_voxel_size"]) == (
        "impartial-shrimp", "3d", True, ["pytorch", "torchscript"], None)
    assert shrimp["cover"] == f"{catalog.HYPHA_ARTIFACTS}/impartial-shrimp/files/cover.thumbnail.jpg"
    assert (shark["dims"], shark["em"], shark["declared_voxel_size"]) == ("2d", False, [250.0, 250.0, 250.0])
    assert catalog.declared_voxel_size(shrimp) is None and catalog.declared_voxel_size(shark) == [250.0] * 3


def test_the_legacy_index_is_used_when_the_server_cannot_be_reached_and_both_failing_says_both(zoo):
    assert catalog.refresh_bioimage_models()["source"] == "index" and zoo.fetches == 1
    zoo.index = catalog.ZooIndexError("index down")
    with pytest.raises(catalog.ZooIndexError, match="unreachable; and index down"):
        catalog.refresh_bioimage_models()
