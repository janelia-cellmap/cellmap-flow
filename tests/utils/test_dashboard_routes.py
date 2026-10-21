"""The dashboard's routes outside finetuning: what it serves, the settings forms
and the model form, the bounding boxes read back from the viewer, and the
viewer layer API."""

import logging
import queue

import neuroglancer
import numpy as np
import pytest
import zarr
from neuroglancer import AxisAlignedBoundingBoxAnnotation as Box

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.jobs.settings import LauncherSettings

READ_YAML = "/api/finetune/read-yaml"


def _read_yaml(dashboard, path):
    return dashboard.get(READ_YAML, query_string={"path": path})


def _load_crops(dashboard, path):
    return dashboard.post("/api/finetune/load-crops", json={"yaml": path, "model_name": "m"})


# Each file the routes must not read, beside a missing one that must answer the same.
REFUSED_AND_MISSING = [
    ("id_rsa", "missing_id_rsa"),
    ("notes.txt", "missing.txt"),
    ("settings.yaml.bak", "missing.yaml.bak"),
    ("link.yaml", "missing.yaml"),  # a YAML name for a secret
    ("dir.yaml", "missing.yaml"),
]


@pytest.mark.parametrize("read", [_read_yaml, _load_crops], ids=["read-yaml", "load-crops"])
def test_only_yaml_files_are_served(dashboard, tmp_path, read):
    """It listens on every interface, so it must not hand out arbitrary files,
    nor say which of them exist."""
    (tmp_path / "crops.yaml").write_text("crops: []\n")
    assert _read_yaml(dashboard, str(tmp_path / "crops.yaml")).get_json()["text"] == "crops: []\n"
    for name in ("id_rsa", "notes.txt", "settings.yaml.bak"):
        (tmp_path / name).write_text("secret")
    (tmp_path / "link.yaml").symlink_to(tmp_path / "id_rsa")
    (tmp_path / "dir.yaml").mkdir()
    for name, missing in REFUSED_AND_MISSING:
        refused, absent = read(dashboard, str(tmp_path / name)), read(dashboard, str(tmp_path / missing))
        assert refused.status_code == 400 and "secret" not in refused.get_data(as_text=True), name
        answer = refused.get_data(as_text=True).replace(name, "<name>")
        assert answer == absent.get_data(as_text=True).replace(missing, "<name>"), name


def test_other_sites_get_no_cors_grant(dashboard, tmp_path):
    """Or they could script the dashboard through its user's browser."""
    (tmp_path / "crops.yaml").write_text("crops: []\n")
    origin = {"Origin": "https://elsewhere.example"}
    response = dashboard.get(READ_YAML, query_string={"path": str(tmp_path / "crops.yaml")}, headers=origin)
    preflight = dashboard.options("/api/models", headers={**origin, "Access-Control-Request-Method": "POST"})
    assert "Access-Control-Allow-Origin" not in response.headers
    assert "Access-Control-Allow-Origin" not in preflight.headers


@pytest.mark.parametrize(
    "url, payload, error",
    [
        pytest.param("/api/blockwise-config", {"nb_cores_master": 4, "nb_cores_worker": 12, "nb_workers": None},
                     "nb_workers", id="blockwise-null"),
        pytest.param("/api/server-config", "not json", "JSON", id="server-not-json"),
        pytest.param("/api/set-data", {"dataset_path": "  "}, "dataset_path", id="set-data-blank"),
        pytest.param("/api/models", {"selected_models": "mito"}, "selected_models", id="models-not-a-list"),
        pytest.param("/update/equivalences", {"dataset": "seg"}, "equivalences", id="equivalences-missing"),
        pytest.param("/api/bbx-generator", {"dataset_path": "/d", "existing_bounding_boxes": [{"offset": [1, 2]}]},
                     "offset", id="box-tool-a-short-corner"),
    ],
)
def test_a_bad_request_is_a_400_that_says_why_and_changes_nothing(dashboard, monkeypatch, url, payload, error):
    """The shape every form reads: {"success": false, "error": ...}."""
    monkeypatch.setattr(LauncherSettings, "save", lambda self: None)
    session = get_session()
    session.queue, session.nb_workers, session.dataset_path, session.models_config = "gpu_h100", 14, "/data/raw.zarr", []
    kwargs = {"data": payload} if isinstance(payload, str) else {"json": payload}
    response = dashboard.post(url, **kwargs)
    assert (response.status_code, response.get_json()["success"]) == (400, False)
    assert error in response.get_json()["error"]
    assert (session.queue, session.nb_workers, session.dataset_path, session.models_config) == ("gpu_h100", 14, "/data/raw.zarr", [])


@pytest.mark.parametrize(
    "class_name, params, status, expected",
    [
        # Form tuples become floats, a whole voxel size an int again, and a
        # list of strings is split on commas when it is not JSON.
        ("FlyModelConfig",
         {"checkpoint_path": "/c.ts", "channels": "mito, er", "input_voxel_size": "16,16,16",
          "output_voxel_size": "[8, 8, 8]", "input_size": "", "name": "fly"},
         200,
         {"type": "fly", "checkpoint_path": "/c.ts", "channels": ["mito", "er"],
          "input_voxel_size": [16, 16, 16], "output_voxel_size": [8, 8, 8], "name": "fly"}),
        ("NoSuchModelConfig", {}, 400, "NoSuchModelConfig"),
        ("DaCapoModelConfig", {"run_name": "r", "iteration": ""}, 400, "iteration"),
        ("DaCapoModelConfig", {"run_name": "r"}, 400, "DaCapoModelConfig"),
    ],
    ids=["parsed", "unknown-class", "empty-required", "constructor-error"],
)
def test_the_model_form_parses_and_rejects_the_same_way(dashboard, class_name, params, status, expected):
    """The model form sends every field as a string: parsed, they make the
    config the session serves; a class or field it cannot use is a 400."""
    response = dashboard.post("/api/create-model-config", json={"class_name": class_name, "params": params})
    assert response.status_code == status, response.get_json()
    if status == 200:
        assert response.get_json()["config_dict"] == expected
        assert [type(m).__name__ for m in get_session().models_config] == [class_name]
    else:
        assert expected in response.get_json()["error"]


def test_the_blockwise_settings_are_kept_and_read_back(dashboard, tmp_path):
    sent = {"queue": "gpu_l4", "charge_group": "grp", "nb_cores_master": 2, "nb_cores_worker": 8, "nb_workers": 5,
            "tmp_dir": str(tmp_path / "progress"), "blockwise_tasks_dir": str(tmp_path / "tasks")}
    assert dashboard.post("/api/blockwise-config", json=sent).get_json() == {"success": True, "config": sent}
    assert dashboard.get("/api/blockwise-config").get_json() == sent
    session = get_session()
    assert (session.queue, session.tmp_dir, session.tasks_dir()) == ("gpu_l4", sent["tmp_dir"], sent["blockwise_tasks_dir"])


def test_a_package_log_record_reaches_the_log_panel(dashboard):
    """Into the buffer a late panel replays, and to each open stream."""
    stream = queue.Queue()
    get_session().log_clients.append(stream)
    logging.getLogger("cellmap_flow.jobs.launch").info("model mito started")
    assert get_session().log_buffer[-1].endswith("INFO cellmap_flow.jobs.launch: model mito started")
    assert stream.get_nowait() == get_session().log_buffer[-1]


def test_the_log_stream_survives_a_record_logged_as_it_replays_and_sends_every_line(dashboard):
    """It iterated the buffer while records were appended to it, and died
    ("deque mutated during iteration"); and a traceback's lines after the
    first, sent without "data: ", were dropped by the browser."""
    get_session().log_buffer.extend(["one", 'Traceback (most recent call last):\n  File "x.py"\nValueError: bad'])
    response = dashboard.get("/api/logs/stream", buffered=False)
    events = iter(response.response)
    assert next(events) == b"data: one\n\n"
    logging.getLogger("cellmap_flow.jobs.launch").info("logged while replaying")
    assert next(events) == b'data: Traceback (most recent call last):\ndata:   File "x.py"\ndata: ValueError: bad\n\n'
    assert next(events).decode().endswith("logged while replaying\n\n")
    response.close()
    assert get_session().log_clients == [], "a closed stream stops getting records"


def test_a_count_sent_as_a_string_is_a_number(dashboard):
    response = dashboard.post("/api/blockwise-config", json={"nb_cores_master": "4", "nb_cores_worker": "12",
                                                             "nb_workers": "3"})
    assert response.status_code == 200 and get_session().nb_workers == 3


def _annotations():
    return neuroglancer.LocalAnnotationLayer(
        dimensions=neuroglancer.CoordinateSpace(names=["z", "y", "x"], units="nm", scales=[1, 1, 1]))


@pytest.mark.parametrize(
    "layers, boxes",
    [
        pytest.param(["bboxes"], [{"offset": [10, 20, 30], "shape": [30, 40, 50]}], id="box-layer"),
        # A layer after the box layer must not take its place: a missing name
        # makes neuroglancer's Layers hand back the last layer.
        pytest.param(["bboxes", "good_regions"], [{"offset": [10, 20, 30], "shape": [30, 40, 50]}],
                     id="a-layer-after-it"),
        # No box layer, but another holding boxes of its own: none were drawn.
        pytest.param(["good_regions"], [], id="no-box-layer"),
        pytest.param(None, [], id="no-viewer"),
    ],
)
def test_the_boxes_drawn_are_read_from_the_box_layer(dashboard, viewer, monkeypatch, layers, boxes):
    with viewer.txn() as s:
        s.layers["fibsem"] = neuroglancer.ImageLayer(source="zarr://http://x/y")
        for name in layers or []:
            s.layers[name] = _annotations()
            s.layers[name].annotations.append(Box(id=name, point_a=[10, 20, 30], point_b=[40, 60, 80]))
    monkeypatch.setitem(get_session().bbx_generator_state, "viewer", viewer if layers else None)
    assert dashboard.get("/api/bbx-generator/status").get_json()["bounding_boxes"] == boxes


def test_a_layer_is_added_renamed_and_removed(dashboard, viewer, tmp_path, monkeypatch):
    s0 = zarr.open_group(str(tmp_path / "labels.zarr"), mode="w").create_dataset(
        "s0", data=np.arange(64, dtype=np.uint32).reshape(4, 4, 4))
    s0.attrs.update(resolution=[8] * 3, offset=[16, 0, 0])
    get_session().shaders = {"seg": "void main() {}"}

    def post(route, **body):
        response = dashboard.post(f"/api/viewer/{route}", json=body)
        return response.status_code, response.get_json()

    path = str(tmp_path / "labels.zarr" / "s0")
    assert post("add-segmentation-layer", path=path, name="seg", disable_meshes=True)[0] == 200
    assert post("add-image-layer", path=path, name="img", shader="void main() {}")[0] == 200
    seg = viewer.state.layers["seg"]
    assert seg.type == "segmentation" and not seg.source[0].subsources["meshes"].enabled
    assert seg.source[0].transform.matrix[0][3] == 2  # the 16 nm corner, in 8 nm voxels
    assert viewer.state.layers["img"].type == "image"

    assert post("rename-layer", old_name="seg", new_name="img")[0] == 409
    assert post("rename-layer", old_name="missing", new_name="x")[0] == 404
    assert post("rename-layer", old_name="seg", new_name="labels")[1]["renamed"]
    assert [layer.name for layer in viewer.state.layers] == ["labels", "img"]
    assert "labels" in get_session().shaders and "seg" not in get_session().shaders
    assert post("remove-layer", name="labels")[1]["removed"] is True
    assert post("remove-layer", name="labels")[1]["removed"] is False
    assert [layer.name for layer in viewer.state.layers] == ["img"] and "labels" not in get_session().shaders
