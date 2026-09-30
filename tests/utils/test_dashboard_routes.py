"""The dashboard's routes outside finetuning: what it serves, the settings forms,
the bounding boxes read back from the viewer, and the viewer layer API."""

import neuroglancer
import numpy as np
import pytest
import zarr
from neuroglancer import AxisAlignedBoundingBoxAnnotation as Box

from cellmap_flow.globals import g

READ_YAML = "/api/finetune/read-yaml"


def test_only_yaml_files_are_served(dashboard, tmp_path):
    """It listens on every interface, so it must not hand out arbitrary files."""
    (tmp_path / "crops.yaml").write_text("crops: []\n")
    assert dashboard.get(READ_YAML, query_string={"path": str(tmp_path / "crops.yaml")}).get_json()["text"] == "crops: []\n"
    for name in ("id_rsa", "notes.txt", "settings.yaml.bak"):
        (tmp_path / name).write_text("secret")
    (tmp_path / "link.yaml").symlink_to(tmp_path / "id_rsa")
    for name in ("id_rsa", "notes.txt", "settings.yaml.bak", "missing_id_rsa", "link.yaml"):
        response = dashboard.get(READ_YAML, query_string={"path": str(tmp_path / name)})
        assert response.status_code == 400 and "secret" not in response.get_data(as_text=True), name


def test_other_sites_get_no_cors_grant(dashboard, tmp_path):
    """Or they could script the dashboard through its user's browser."""
    (tmp_path / "crops.yaml").write_text("crops: []\n")
    origin = {"Origin": "https://elsewhere.example"}
    response = dashboard.get(READ_YAML, query_string={"path": str(tmp_path / "crops.yaml")}, headers=origin)
    preflight = dashboard.options("/api/models", headers={**origin, "Access-Control-Request-Method": "POST"})
    assert "Access-Control-Allow-Origin" not in response.headers
    assert "Access-Control-Allow-Origin" not in preflight.headers


@pytest.mark.parametrize(
    "url, payload, status",
    [
        pytest.param("/api/blockwise-config", {"nb_cores_master": 4, "nb_cores_worker": 12, "nb_workers": None}, 400,
                     id="blockwise-null"),
        pytest.param("/api/blockwise-config", {"nb_cores_master": 4, "nb_cores_worker": 12, "nb_workers": ""}, 400,
                     id="blockwise-empty"),
        pytest.param("/api/blockwise-config", {"nb_cores_master": 4, "nb_cores_worker": 12, "nb_workers": "twelve"}, 400,
                     id="blockwise-word"),
        pytest.param("/api/blockwise-config", {"nb_cores_master": "4", "nb_cores_worker": "12", "nb_workers": "3"}, 200,
                     id="blockwise-numeric-strings-accepted"),
        pytest.param("/api/server-config", {"queue": "gpu_a100", "nb_workers": "lots"}, 400, id="server-word"),
        pytest.param("/api/server-config", {"queue": "gpu_a100", "nb_workers": None}, 400, id="server-null"),
        pytest.param("/api/server-config", "not json", 400, id="server-not-json"),
    ],
)
def test_a_bad_setting_is_a_400_that_changes_nothing(dashboard, monkeypatch, url, payload, status):
    monkeypatch.setattr(type(g), "save_server_config", lambda self: None)
    g.queue, g.nb_workers = "gpu_h100", 14
    kwargs = {"data": payload} if isinstance(payload, str) else {"json": payload}
    response = dashboard.post(url, **kwargs)
    assert response.status_code == status
    if status == 400:  # the shape the settings forms read, and nothing applied
        body = response.get_json()
        assert body["success"] is False and (isinstance(payload, str) or "nb_workers" in body["error"])
        assert (g.queue, g.nb_workers) == ("gpu_h100", 14)
    else:
        assert g.nb_workers == 3


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
    from cellmap_flow.dashboard.routes import bbx_generator

    with viewer.txn() as s:
        s.layers["fibsem"] = neuroglancer.ImageLayer(source="zarr://http://x/y")
        for name in layers or []:
            s.layers[name] = _annotations()
            s.layers[name].annotations.append(Box(id=name, point_a=[10, 20, 30], point_b=[40, 60, 80]))
    monkeypatch.setitem(bbx_generator.bbx_generator_state, "viewer", viewer if layers else None)
    assert dashboard.get("/api/bbx-generator/status").get_json()["bounding_boxes"] == boxes


def test_a_layer_is_added_renamed_and_removed(dashboard, viewer, tmp_path, monkeypatch):
    s0 = zarr.open_group(str(tmp_path / "labels.zarr"), mode="w").create_dataset(
        "s0", data=np.arange(64, dtype=np.uint32).reshape(4, 4, 4))
    s0.attrs.update(resolution=[8] * 3, offset=[16, 0, 0])
    monkeypatch.setattr(g, "shaders", {"seg": "void main() {}"})

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
    assert "labels" in g.shaders and "seg" not in g.shaders
    assert post("remove-layer", name="labels")[1]["removed"] is True
    assert post("remove-layer", name="labels")[1]["removed"] is False
    assert [layer.name for layer in viewer.state.layers] == ["img"] and "labels" not in g.shaders
