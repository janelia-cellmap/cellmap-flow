"""The BioImage Model Zoo on the Models tab: the list read from bioimage.io's
server or its legacy index (models.bioimage_catalog), the routes that serve
it, and Submit launching what is ticked as a ``bioimage`` model. The server
and the index are faked: no test touches the network."""

from types import SimpleNamespace

import pytest

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.models import bioimage_catalog as catalog
from tests.utils.test_cellpose_list import models_tab, pixi_manifest, served_entry  # noqa: F401  (models_tab is a fixture)

DOI = "10.5281/zenodo.5874841"
INDEX = {"collection": [
    {"id": DOI, "nickname": "kind-seashell", "type": "model", "name": "MitochondriaEMSegmentationBoundaryModel",
     "description": "Mitochondria segmentation for\n electron microscopy.", "license": "CC-BY-4.0",
     "tags": ["unet", "electron-microscopy", "3d", "pytorch"], "covers": ["https://zoo/cover.png"]},
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
    state = SimpleNamespace(fetches=0, index=INDEX, hypha=catalog.ZooIndexError("unreachable"))

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


def test_the_routes_list_and_refresh_the_zoo_and_say_why_when_it_is_unreachable(dashboard, zoo):
    listed = dashboard.get("/api/bioimage-models")
    assert listed.status_code == 200 and [m["key"] for m in listed.get_json()["models"]] == ["kind-seashell"]
    zoo.index = {"collection": []}
    assert dashboard.post("/api/bioimage-models/refresh").get_json()["models"] == [] and zoo.fetches == 2
    zoo.index = catalog.ZooIndexError("Could not fetch the BioImage Model Zoo index: offline")
    failed = dashboard.post("/api/bioimage-models/refresh")
    assert failed.status_code == 502 and "offline" in failed.get_json()["error"]


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


@pytest.fixture
def submit(models_tab):
    """POST /api/models with these zoo models ticked; the commands started are ``submit.commands``."""
    catalog.list_bioimage_models()

    def post(*selections):
        return models_tab(selected_bioimage_models=list(selections))

    post.commands = models_tab.commands
    return post


def test_submit_serves_a_ticked_zoo_model_from_the_bioimageio_environment(submit, tmp_path, monkeypatch):
    manifest = pixi_manifest(tmp_path, monkeypatch, MANIFEST)

    # Ticked by its DOI, loaded by its nickname.
    assert submit({"id": DOI, "voxel_size": "8"}).status_code == 200
    (command,) = submit.commands
    argv, entry = served_entry(command)
    assert argv[:7] == ["/opt/pixi", "run", "--frozen", "--manifest-path", str(manifest), "-e", "bioimageio"]
    assert entry == {"type": "bioimage", "model": "kind-seashell", "voxel_size": [8, 8, 8], "name": "kind_seashell"}
    assert [mc.name for mc in get_session().models_config] == ["kind_seashell"]


def test_a_blank_voxel_size_for_a_model_that_declares_none_refuses_the_submit(submit, monkeypatch):
    """Its server would refuse it after its job started, where the page showed
    nothing: Submit says so instead."""
    monkeypatch.setattr(catalog, "trained_at", lambda model: None)
    monkeypatch.setattr(catalog, "declared_voxel_size", lambda entry: None)
    answer = submit({"id": "kind-seashell"})
    assert answer.status_code == 400 and "voxel size" in answer.get_json()["error"]
    assert submit.commands == []
    # Given one, it starts.
    assert submit({"id": "kind-seashell", "voxel_size": "8"}).status_code == 200 and len(submit.commands) == 1


def test_a_blank_voxel_size_is_the_one_the_model_was_trained_at_when_cellmap_flow_knows_it(submit):
    assert submit({"id": "kind-seashell"}).status_code == 200
    (command,) = submit.commands
    assert served_entry(command)[1]["voxel_size"] == [30, 8, 8]
