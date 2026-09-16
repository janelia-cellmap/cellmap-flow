"""Marking the current view as a good region.

A good region is the counterweight to a correction: the scribbles say "wrong
here", a good region says "right here, and I looked". Training samples them
without annotations, which the trainer already handles -- an empty mask
zeroes the supervised term and gives the whole patch to distillation.
"""

import json
import os

import pytest

from cellmap_flow.dashboard.routes.finetune import good_regions as gr


class _FakeTxn:
    def __init__(self, state):
        self._state = state

    def __enter__(self):
        return self._state

    def __exit__(self, *exc):
        return False


class _FakeViewer:
    """A viewer positioned at a fixed spot, with a real neuroglancer state."""

    def __init__(self, position, scales):
        import neuroglancer

        self.position = position
        self.scales = scales
        self._state = neuroglancer.viewer_state.ViewerState()

    def txn(self):
        import neuroglancer

        s = self._state
        s.dimensions = neuroglancer.CoordinateSpace(
            names=["z", "y", "x"], units="nm", scales=self.scales
        )
        s.position = self.position
        return _FakeTxn(s)


@pytest.fixture
def session(tmp_path, monkeypatch):
    """A session whose corrections dir lives under tmp_path."""
    corrections = tmp_path / "20260101_000000" / "corrections"
    corrections.mkdir(parents=True)
    monkeypatch.setattr(
        gr.g,
        "annotation_volumes",
        {"vol-1": {
            "corrections_dir": str(corrections),
            # Both are present on a real volume, and they differ. Keeping the
            # input geometry here is what makes the size test discriminating:
            # sizing off the input would give 2848nm, not 896nm.
            "input_size": [178, 178, 178],
            "input_voxel_size": [16, 16, 16],
            "output_size": [56, 56, 56],
            "output_voxel_size": [16, 16, 16],
        }},
        raising=False,
    )
    monkeypatch.setattr(gr.g, "viewer", _FakeViewer([100, 200, 300], [16, 16, 16]),
                        raising=False)
    monkeypatch.setattr(gr.g, "raw", None, raising=False)
    return tmp_path / "20260101_000000"


def _post(app_client, path, payload=None):
    return app_client.post(path, json=payload if payload is not None else {})


@pytest.fixture
def client():
    from cellmap_flow.dashboard.app import app

    app.config.update(TESTING=True)
    return app.test_client()


def test_marking_records_a_box_centred_on_the_view(session, client):
    r = _post(client, "/api/finetune/good-regions/mark-view")
    assert r.status_code == 200
    body = r.get_json()
    assert body["success"] and body["count"] == 1

    region = body["region"]
    # position (voxels) * scales (nm) = centre in nm; box is centred on it.
    # 56 * 16 = 896 nm on a side: one model *output* patch, not the 2848nm
    # input field of view -- the box marks what you looked at and judged.
    assert region["shape_nm"] == [896.0, 896.0, 896.0]
    centre = [o + s / 2 for o, s in zip(region["offset_nm"], region["shape_nm"])]
    assert centre == [100 * 16, 200 * 16, 300 * 16]


def test_regions_persist_to_the_session(session, client):
    _post(client, "/api/finetune/good-regions/mark-view")
    _post(client, "/api/finetune/good-regions/mark-view")

    stored = json.loads((session / "good_regions.json").read_text())
    assert len(stored) == 2
    assert [r["label"] for r in stored] == ["good-1", "good-2"]
    assert client.get("/api/finetune/good-regions").get_json()["count"] == 2


def test_an_explicit_size_overrides_the_model_field_of_view(session, client):
    r = _post(client, "/api/finetune/good-regions/mark-view", {"size_nm": [512, 512, 512]})
    assert r.get_json()["region"]["shape_nm"] == [512.0, 512.0, 512.0]


def test_a_nonpositive_size_is_rejected(session, client):
    r = _post(client, "/api/finetune/good-regions/mark-view", {"size_nm": [0, 10, 10]})
    assert r.status_code == 400
    assert not (session / "good_regions.json").exists()


def test_deleting_by_id_leaves_the_others(session, client):
    first = _post(client, "/api/finetune/good-regions/mark-view").get_json()["region"]
    _post(client, "/api/finetune/good-regions/mark-view")

    r = _post(client, "/api/finetune/good-regions/delete", {"id": first["id"]})
    assert r.get_json()["count"] == 1
    remaining = client.get("/api/finetune/good-regions").get_json()["regions"]
    assert [x["id"] for x in remaining] != [first["id"]]


def test_delete_with_no_id_clears_everything(session, client):
    _post(client, "/api/finetune/good-regions/mark-view")
    _post(client, "/api/finetune/good-regions/mark-view")
    assert _post(client, "/api/finetune/good-regions/delete").get_json()["count"] == 0
    assert client.get("/api/finetune/good-regions").get_json()["regions"] == []


def test_the_boxes_are_drawn_in_their_own_layer(session, client):
    _post(client, "/api/finetune/good-regions/mark-view")
    with gr.g.viewer.txn() as s:
        assert gr.GOOD_REGIONS_LAYER in s.layers
        assert len(s.layers[gr.GOOD_REGIONS_LAYER].annotations) == 1


def test_no_session_reports_that_it_did_not_persist(tmp_path, monkeypatch, client):
    monkeypatch.setattr(gr.g, "annotation_volumes", {}, raising=False)
    monkeypatch.setattr(gr.g, "viewer", _FakeViewer([1, 2, 3], [16, 16, 16]),
                        raising=False)
    monkeypatch.setattr(gr.g, "raw", None, raising=False)

    body = _post(client, "/api/finetune/good-regions/mark-view").get_json()
    assert body["success"] is True
    assert body["persisted"] is False, "must not claim to have saved"
