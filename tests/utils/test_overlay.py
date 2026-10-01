"""The annotated-regions overlay: which boxes it draws, and when it touches the viewer.

Every push from python costs the browser its whole UI state: it rebuilds the
state object graph, tool binder included, which takes the brush out of the
user's hand and can drop a stroke still in its commit buffer. neuroglancer's
txn() pushes whether or not anything changed. So the boxes are redrawn on
demand, from a button, and only when they changed; the periodic sync never
draws them (see test_session_sync).
"""

import json
import os

import pytest

from cellmap_flow.dashboard.routes.finetune import overlay
from cellmap_flow.dashboard.state import get_session


def _volume(corrections, chunks, crops=(), chunk_size=56):
    """An annotation volume under ``corrections`` with painted ``chunks``
    (chunk keys) and imported ``crops`` ((offset, shape) in voxels)."""
    s0 = os.path.join(corrections, "vol.zarr", "annotation", "s0")
    os.makedirs(s0, exist_ok=True)
    with open(os.path.join(corrections, "vol.zarr", ".zattrs"), "w") as f:
        json.dump({"type": "annotation_volume", "output_voxel_size": [16] * 3, "chunk_size": [chunk_size] * 3,
                   "dataset_offset_nm": [8] * 3, "imported_crops": [
                       {"name": f"crop{i}", "annotation_offset_voxels": list(o), "annotation_shape_voxels": list(s)}
                       for i, (o, s) in enumerate(crops)]}, f)
    for key in chunks:
        open(os.path.join(s0, key), "wb").close()
    return str(corrections)


@pytest.fixture
def txns(viewer, monkeypatch):
    """The viewer's transactions, counted: each is a push to the browser."""
    count = []
    real = viewer.txn
    monkeypatch.setattr(viewer, "txn", lambda *a, **k: count.append(1) or real(*a, **k))
    monkeypatch.setattr(overlay, "_last_annotated_regions", None)
    for attr in ("annotation_volumes", "output_sessions"):
        monkeypatch.setattr(get_session(), attr, {})
    return count


@pytest.mark.parametrize(
    "crops, chunks, boxes",
    [
        pytest.param([], ["0.0.0"], 1, id="painted-chunk"),
        pytest.param([((0, 0, 0), (256, 256, 256))], ["1.1.1"], 1, id="chunk-inside-a-crop"),
        # A crop's offset is essentially never chunk-aligned, so the chunks on
        # its edge only partly overlap it; they are the crop's, not boxes of
        # their own fencing it in (the jrc_axolotl-heart-1 mito005 crop).
        pytest.param([((14283, 5655, 3352), (440, 857, 1000))], ["111.44.26"], 1, id="chunk-on-a-crops-edge"),
        pytest.param([((0, 0, 0), (128, 128, 128))], ["2.2.2"], 2, id="chunk-outside-the-crop"),
    ],
)
def test_a_box_per_crop_and_per_painted_chunk_outside_them(tmp_path, txns, crops, chunks, boxes):
    corrections = _volume(tmp_path / "corrections", chunks, crops, chunk_size=128)
    assert overlay.refresh_annotated_regions_layer(corrections) == boxes


def test_the_viewer_is_pushed_only_when_the_boxes_change(tmp_path, viewer, txns):
    corrections = _volume(tmp_path / "corrections", ["0.0.0"])
    assert overlay.refresh_annotated_regions_layer(corrections) == 1 and len(txns) == 1
    for _ in range(3):  # nothing changed: nothing pushed
        overlay.refresh_annotated_regions_layer(corrections)
    assert len(txns) == 1
    _volume(tmp_path / "corrections", ["1.0.0"])
    assert overlay.refresh_annotated_regions_layer(corrections) == 2 and len(txns) == 2
    with viewer.txn() as s:  # the layer went away: a viewer reset, a manual delete
        del s.layers["annotated_regions"]
    overlay.refresh_annotated_regions_layer(corrections)
    assert "annotated_regions" in viewer.state.layers and len(txns) == 4


def test_no_boxes_opens_no_transaction(tmp_path, txns):
    (tmp_path / "corrections").mkdir()
    assert overlay.refresh_annotated_regions_layer(str(tmp_path / "corrections")) == 0 and txns == []


def test_the_button_redraws_and_a_second_click_costs_nothing(dashboard, tmp_path, viewer, txns):
    corrections = _volume(tmp_path / "corrections", ["0.0.0", "1.0.0"])
    for _ in range(3):
        body = dashboard.post("/api/finetune/refresh-annotated-regions", json={"corrections_path": corrections}).get_json()
        assert body["success"] and body["count"] == 2
    assert "annotated_regions" in viewer.state.layers and len(txns) == 1


def test_annotation_layers_come_with_the_draw_tools_bound(dashboard, viewer):
    """A and F must be bound on the layer handed to neuroglancer, in the JSON
    the browser is sent: keys must be one capital letter (src/ui/tool.ts)."""
    body = dashboard.post("/api/finetune/add-to-viewer", json={"crop_id": "c1", "minio_url": "http://minio/x.zarr"}).get_json()
    assert body["success"]
    bindings = viewer.state.layers["annotation_c1"].to_json()["toolBindings"]

    def tool(value):  # bare, or {"type": ...} once anything has materialized; both valid
        return value["type"] if isinstance(value, dict) else value

    assert (tool(bindings["A"]), tool(bindings["F"])) == ("vox-brush", "vox-flood-fill")
    assert all(k.isupper() and len(k) == 1 for k in bindings)


def test_a_new_annotation_layer_is_selected_with_its_panel_open(dashboard, viewer):
    """Creating or loading a volume switches the viewer to its layer, ready to paint."""
    body = dashboard.post("/api/finetune/add-to-viewer", json={"crop_id": "c1", "minio_url": "http://minio/x.zarr"}).get_json()
    assert body["success"]
    selected = viewer.state.selected_layer
    assert (selected.layer, selected.visible) == ("annotation_c1", True)
