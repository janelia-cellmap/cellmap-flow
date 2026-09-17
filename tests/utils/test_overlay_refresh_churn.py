"""The annotated_regions overlay must not rewrite the viewer for no reason.

Every push from python costs the browser its whole UI state: on receiving one
it runs `trackable.reset(); trackable.restoreState(state)`, rebuilding the
entire state object graph, tool binder included. neuroglancer's txn() is
unconditional -- it deep-copies the state on entry and calls set_state() on
exit whether or not the body changed anything -- and the periodic sync calls
refresh_annotated_regions_layer() every 30s for as long as annotations keep
arriving. That is precisely while the user is drawing, so every one of those
pushes takes the brush out of their hand.
"""

import json
import os

import pytest

from cellmap_flow.dashboard.routes.finetune import overlay


class _FakeLayers(dict):
    def __contains__(self, name):
        return dict.__contains__(self, name)


class _FakeState:
    def __init__(self):
        self.layers = _FakeLayers()
        # What the browser told python it has selected, as it appears in the
        # serialized state -- None when the user is just looking around.
        self.active_tool = None

    def to_json(self):
        return {"layers": [{"name": "sparse_annotation", "tool": self.active_tool}]}


class _FakeViewer:
    """Counts transactions; .state reads must not count as one."""

    def __init__(self):
        self._state = _FakeState()
        self.txn_count = 0

    @property
    def state(self):
        return self._state

    def txn(self):
        viewer = self

        class _Txn:
            def __enter__(self):
                viewer.txn_count += 1
                return viewer._state

            def __exit__(self, *exc):
                return False

        return _Txn()


def _write_chunk(corrections_dir, name="vol_chunk_0.zarr", offset=(0, 0, 0)):
    chunk_dir = os.path.join(corrections_dir, name)
    os.makedirs(chunk_dir, exist_ok=True)
    with open(os.path.join(chunk_dir, ".zattrs"), "w") as f:
        json.dump(
            {
                "roi": {
                    "annotation_offset": list(offset),
                    "annotation_shape": [56, 56, 56],
                },
                "annotation_voxel_size": [16, 16, 16],
            },
            f,
        )


@pytest.fixture
def corrections(tmp_path, monkeypatch):
    d = tmp_path / "corrections"
    d.mkdir()
    monkeypatch.setattr(overlay, "_last_annotated_regions", None, raising=False)
    monkeypatch.setattr(overlay.g, "annotation_volumes", {}, raising=False)
    monkeypatch.setattr(overlay.g, "output_sessions", {}, raising=False)
    monkeypatch.setattr(overlay.g, "raw", None, raising=False)
    return d


def test_unchanged_boxes_do_not_touch_the_viewer_again(corrections, monkeypatch):
    _write_chunk(str(corrections))
    viewer = _FakeViewer()
    monkeypatch.setattr(overlay.g, "viewer", viewer, raising=False)

    assert overlay.refresh_annotated_regions_layer(str(corrections)) == 1
    assert viewer.txn_count == 1, "first call must create the layer"

    # The sync thread's every-30s call, with nothing actually changed.
    for _ in range(5):
        assert overlay.refresh_annotated_regions_layer(str(corrections)) == 1
    assert viewer.txn_count == 1, "redundant refreshes must not push state"


def test_a_new_box_does_push(corrections, monkeypatch):
    _write_chunk(str(corrections))
    viewer = _FakeViewer()
    monkeypatch.setattr(overlay.g, "viewer", viewer, raising=False)

    overlay.refresh_annotated_regions_layer(str(corrections))
    overlay.refresh_annotated_regions_layer(str(corrections))
    assert viewer.txn_count == 1

    _write_chunk(str(corrections), name="vol_chunk_1.zarr", offset=(56, 0, 0))
    assert overlay.refresh_annotated_regions_layer(str(corrections)) == 2
    assert viewer.txn_count == 2, "a real change must reach the viewer"


def test_a_layer_that_went_away_is_restored(corrections, monkeypatch):
    _write_chunk(str(corrections))
    viewer = _FakeViewer()
    monkeypatch.setattr(overlay.g, "viewer", viewer, raising=False)

    overlay.refresh_annotated_regions_layer(str(corrections))
    assert "annotated_regions" in viewer.state.layers

    # Simulate the layer disappearing (viewer reset, manual delete).
    del viewer.state.layers["annotated_regions"]
    overlay.refresh_annotated_regions_layer(str(corrections))
    assert viewer.txn_count == 2
    assert "annotated_regions" in viewer.state.layers


def test_no_boxes_does_not_open_a_transaction(corrections, monkeypatch):
    viewer = _FakeViewer()
    monkeypatch.setattr(overlay.g, "viewer", viewer, raising=False)

    assert overlay.refresh_annotated_regions_layer(str(corrections)) == 0
    assert viewer.txn_count == 0, "nothing to remove, so nothing to push"


def test_annotation_layers_get_the_draw_tools_prebound(monkeypatch):
    """A/F must be bound on the layer we hand neuroglancer, not hunted for."""
    import neuroglancer
    from cellmap_flow.dashboard.app import app

    viewer = _FakeViewer()
    monkeypatch.setattr(overlay.g, "viewer", viewer, raising=False)

    app.config.update(TESTING=True)
    client = app.test_client()
    r = client.post(
        "/api/finetune/add-to-viewer",
        json={"crop_id": "c1", "minio_url": "http://minio/x.zarr"},
    )
    assert r.status_code == 200 and r.get_json()["success"]

    # Assert on the serialized form: that JSON is what the browser is sent,
    # and it is the only thing that decides whether the keys work.
    bindings = viewer.state.layers["annotation_c1"].to_json()["toolBindings"]

    def tool_of(value):
        # A Tool serializes bare when it carries nothing but a type, and as
        # {"type": ...} once anything has materialized it. Both are valid.
        return value["type"] if isinstance(value, dict) else value

    assert tool_of(bindings["A"]) == "vox-brush"
    assert tool_of(bindings["F"]) == "vox-flood-fill"
    # Keys must be a single capital letter or neuroglancer drops the binding
    # (TOOL_KEY_PATTERN = /^[A-Z]$/ in src/ui/tool.ts).
    assert all(k.isupper() and len(k) == 1 for k in bindings)


def test_background_refresh_stands_down_while_a_tool_is_selected(
    corrections, monkeypatch
):
    """The 30s sync must not take the brush out of the user's hand."""
    _write_chunk(str(corrections))
    viewer = _FakeViewer()
    monkeypatch.setattr(overlay.g, "viewer", viewer, raising=False)

    viewer.state.active_tool = "vox-brush"
    assert (
        overlay.refresh_annotated_regions_layer(
            str(corrections), defer_if_tool_active=True
        )
        == 0
    )
    assert viewer.txn_count == 0, "a selected tool must veto a background push"


def test_a_deferred_refresh_happens_once_the_tool_is_put_down(
    corrections, monkeypatch
):
    """Standing down must not mean forgetting -- the boxes still catch up."""
    _write_chunk(str(corrections))
    viewer = _FakeViewer()
    monkeypatch.setattr(overlay.g, "viewer", viewer, raising=False)

    viewer.state.active_tool = "vox-brush"
    for _ in range(3):
        overlay.refresh_annotated_regions_layer(
            str(corrections), defer_if_tool_active=True
        )
    assert viewer.txn_count == 0

    viewer.state.active_tool = None
    assert (
        overlay.refresh_annotated_regions_layer(
            str(corrections), defer_if_tool_active=True
        )
        == 1
    )
    assert viewer.txn_count == 1, "the skipped work must not be marked as done"


def test_a_user_initiated_refresh_pushes_even_with_a_tool_selected(
    corrections, monkeypatch
):
    """Clicking sync means now, tool or no tool -- the user asked for it."""
    _write_chunk(str(corrections))
    viewer = _FakeViewer()
    monkeypatch.setattr(overlay.g, "viewer", viewer, raising=False)

    viewer.state.active_tool = "vox-brush"
    assert overlay.refresh_annotated_regions_layer(str(corrections)) == 1
    assert viewer.txn_count == 1


def test_active_tool_reads_both_serialized_tool_shapes(monkeypatch):
    """A Tool serializes bare or as {"type": ...}; both mean "in hand"."""

    class _State:
        def __init__(self, layers):
            self._layers = layers

        def to_json(self):
            return {"layers": self._layers}

    class _V:
        def __init__(self, layers):
            self.state = _State(layers)

    monkeypatch.setattr(overlay.g, "viewer", _V([{"tool": "vox-brush"}]), raising=False)
    assert overlay._active_annotation_tool() == "vox-brush"

    monkeypatch.setattr(
        overlay.g, "viewer", _V([{"tool": {"type": "vox-flood-fill"}}]), raising=False
    )
    assert overlay._active_annotation_tool() == "vox-flood-fill"

    # Older neuroglancer serialized layers as a name-keyed object.
    monkeypatch.setattr(
        overlay.g, "viewer", _V({"seg": {"tool": "vox-brush"}}), raising=False
    )
    assert overlay._active_annotation_tool() == "vox-brush"

    monkeypatch.setattr(
        overlay.g, "viewer", _V([{"name": "raw"}, {"tool": None}]), raising=False
    )
    assert overlay._active_annotation_tool() is None
