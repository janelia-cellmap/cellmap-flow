"""A dataset path as a person pastes it (cellmap_flow.portable.pasted): the
table of cases the browser's twin is held to as well
(web/test/parity/pasted.test.ts), and two of the places a person gives one,
the dashboard's dataset field and a YAML's data_path."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.portable.pasted import pasted_path

TABLE = Path(__file__).resolve().parents[2] / "web" / "test" / "fixtures" / "pasted_paths.json"
CASES = json.loads(TABLE.read_text())["cases"]

PASTED = ' "zarr://https://host/em.zarr|zarr2:recon-1/em/fibsem-uint8" '
PATH = "https://host/em.zarr/recon-1/em/fibsem-uint8"


@pytest.mark.parametrize("case", CASES, ids=[case["pasted"] for case in CASES])
def test_a_pasted_path_reads_as_the_path_it_names(case):
    assert pasted_path(case["pasted"]) == case["path"]


def test_the_dashboards_dataset_field_opens_a_path_pasted_from_neuroglancer(dashboard, monkeypatch):
    from cellmap_flow.dashboard.routes import index_page

    opened = []
    # A stand-in viewer: the route only watches it for layers the person
    # deletes (services.launch.watch_layer_deletes).
    viewer = SimpleNamespace(shared_state=SimpleNamespace(raw_state={}, add_changed_callback=lambda callback: None))
    monkeypatch.setattr(index_page, "new_viewer", lambda path, scales: opened.append(path) or viewer)
    monkeypatch.setattr(index_page, "viewer_url", lambda viewer: "http://viewer/")
    assert dashboard.post("/api/set-data", json={"dataset_path": PASTED}).status_code == 200
    assert (opened, get_session().dataset_path) == ([PATH], PATH)


def test_a_yamls_data_path_pasted_from_neuroglancer_is_read_as_the_path(tmp_path):
    from cellmap_flow.config.yaml import load_config

    config = tmp_path / "run.yaml"
    config.write_text(f"data_path: '{PASTED}'\ncharge_group: cellmap\nqueue: gpu_h100\n")
    assert load_config(str(config))["data_path"] == PATH
