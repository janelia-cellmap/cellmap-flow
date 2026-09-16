"""The annotated_regions overlay must not rewrite the viewer for no reason.

neuroglancer's txn() is unconditional -- it deep-copies the state on entry and
calls set_state() on exit whether or not the body changed anything -- and the
periodic sync calls refresh_annotated_regions_layer() every 30s for as long as
annotations keep arriving. That is precisely while the user is drawing, so
every one of those pushes is a chance to clobber a browser-side tool
selection that python did not know about yet.
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
