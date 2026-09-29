"""The /api/viewer layer routes add, rename and remove layers of the live viewer."""

import neuroglancer
import numpy as np
import zarr


def test_a_layer_is_added_renamed_and_removed(monkeypatch, tmp_path):
    from cellmap_flow.dashboard.app import app
    from cellmap_flow.globals import g

    labels = zarr.open_group(str(tmp_path / "labels.zarr"), mode="w")
    s0 = labels.create_dataset("s0", data=np.arange(64, dtype=np.uint32).reshape(4, 4, 4))
    s0.attrs.update(resolution=[8] * 3, offset=[16, 0, 0])
    monkeypatch.setattr(g, "viewer", neuroglancer.Viewer())
    monkeypatch.setattr(g, "shaders", {"seg": "void main() {}"})
    client = app.test_client()

    def post(route, **body):
        response = client.post(f"/api/viewer/{route}", json=body)
        return response.status_code, response.get_json()

    path = str(tmp_path / "labels.zarr" / "s0")
    assert post("add-segmentation-layer", path=path, name="seg", disable_meshes=True)[0] == 200
    assert post("add-image-layer", path=path, name="img", shader="void main() {}")[0] == 200
    seg = g.viewer.state.layers["seg"]
    assert seg.type == "segmentation" and not seg.source[0].subsources["meshes"].enabled
    assert seg.source[0].transform.matrix[0][3] == 2  # the 16 nm corner, in 8 nm voxels
    assert g.viewer.state.layers["img"].type == "image"

    assert post("rename-layer", old_name="seg", new_name="img")[0] == 409
    assert post("rename-layer", old_name="missing", new_name="x")[0] == 404
    assert post("rename-layer", old_name="seg", new_name="labels")[1]["renamed"]
    assert [layer.name for layer in g.viewer.state.layers] == ["labels", "img"]
    assert "labels" in g.shaders and "seg" not in g.shaders

    assert post("remove-layer", name="labels")[1]["removed"] is True
    assert post("remove-layer", name="labels")[1]["removed"] is False
    assert [layer.name for layer in g.viewer.state.layers] == ["img"]
    assert "labels" not in g.shaders
