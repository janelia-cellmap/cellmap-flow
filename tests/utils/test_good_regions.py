"""Marking the current view as a good region.

A good region is the counterweight to a correction: the scribbles say "wrong
here", a good region says "right here, and I looked". Training samples them
without annotations, which the trainer already handles -- an empty mask zeroes
the supervised term and gives the whole patch to distillation.
"""

import json

import neuroglancer
import pytest

from cellmap_flow.dashboard.routes.finetune import good_regions as gr
from cellmap_flow.dashboard.state import get_session

MARK = "/api/finetune/good-regions/mark-view"
VOLUME = {
    # Both are on a real volume, and they differ: a box sized off the input
    # would be 2848 nm, not 896.
    "input_size": [178] * 3, "input_voxel_size": [16] * 3,
    "output_size": [56] * 3, "output_voxel_size": [16] * 3,
}


@pytest.fixture
def view(viewer):
    """The viewer at voxel (100, 200, 300), 16 nm voxels."""
    with viewer.txn() as s:
        s.dimensions = neuroglancer.CoordinateSpace(names=["z", "y", "x"], units="nm", scales=[16] * 3)
        s.position = [100, 200, 300]
    return viewer


@pytest.fixture
def session(tmp_path, monkeypatch, view):
    """A session with an annotation volume, its corrections under tmp_path."""
    corrections = tmp_path / "20260101_000000" / "corrections"
    corrections.mkdir(parents=True)
    monkeypatch.setattr(get_session(), "annotation_volumes", {"vol-1": {"corrections_dir": str(corrections), **VOLUME}})
    monkeypatch.setattr(get_session(), "raw", None)
    return corrections.parent


def test_a_mark_is_one_output_patch_centred_on_the_view(dashboard, session):
    body = dashboard.post(MARK, json={}).get_json()
    assert body["success"] and body["count"] == 1
    region = body["region"]
    # 56 voxels of 16 nm: the model's output patch, what was looked at and judged.
    assert region["shape_nm"] == [896.0] * 3
    assert [o + s / 2 for o, s in zip(region["offset_nm"], region["shape_nm"])] == [100 * 16, 200 * 16, 300 * 16]


@pytest.mark.parametrize("registered", [pytest.param(True, id="from-its-volume"),
                                        pytest.param(False, id="from-minio-with-no-volume-registered")])
def test_marks_add_up_in_the_sessions_good_regions_file(dashboard, session, monkeypatch, registered):
    """With no volume registered the marks had nowhere to go: each click
    appended to an empty list, the count said 1, and every earlier mark was
    gone, while the response still said success."""
    if not registered:
        monkeypatch.setattr(get_session(), "annotation_volumes", {})
        monkeypatch.setitem(get_session().minio_state, "output_base", str(session / "corrections"))
    for _ in range(3):
        dashboard.post(MARK, json={"size_nm": [512] * 3})
    stored = json.loads((session / "good_regions.json").read_text())
    assert [r["label"] for r in stored] == ["good-1", "good-2", "good-3"]
    assert dashboard.get("/api/finetune/good-regions").get_json()["count"] == 3


@pytest.mark.parametrize("size, status, shape", [
    pytest.param([512] * 3, 200, [512.0] * 3, id="explicit"),
    pytest.param([0, 10, 10], 400, None, id="not-positive"),
])
def test_a_marks_size_may_be_given_but_must_be_positive(dashboard, session, size, status, shape):
    response = dashboard.post(MARK, json={"size_nm": size})
    assert response.status_code == status
    if shape:
        assert response.get_json()["region"]["shape_nm"] == shape
    else:
        assert not (session / "good_regions.json").exists()


@pytest.mark.parametrize("by_id", [pytest.param(True, id="one-by-id"), pytest.param(False, id="all")])
def test_deleting_marks(dashboard, session, by_id):
    first = dashboard.post(MARK, json={}).get_json()["region"]
    dashboard.post(MARK, json={})
    body = dashboard.post("/api/finetune/good-regions/delete", json={"id": first["id"]} if by_id else {}).get_json()
    left = dashboard.get("/api/finetune/good-regions").get_json()["regions"]
    assert body["count"] == len(left) == (1 if by_id else 0)
    assert first["id"] not in [r["id"] for r in left]


def test_the_marks_are_drawn_in_their_own_layer(dashboard, session, view):
    dashboard.post(MARK, json={})
    assert len(view.state.layers[gr.GOOD_REGIONS_LAYER].annotations) == 1


def test_with_no_session_a_mark_is_refused_and_not_drawn(dashboard, view, monkeypatch):
    """Better a visible error than a box that vanishes on the next click."""
    monkeypatch.setattr(get_session(), "annotation_volumes", {})
    monkeypatch.setitem(get_session().minio_state, "output_base", None)
    response = dashboard.post(MARK, json={})
    assert response.status_code == 409 and response.get_json()["success"] is False
    assert gr.GOOD_REGIONS_LAYER not in view.state.layers


def test_resuming_a_session_carries_its_good_regions(dashboard, session, tmp_path, monkeypatch):
    """They sit beside corrections/, so copying the zarrs left them behind:
    the resumed session showed none and trained with no rehearsal."""
    import zarr

    from cellmap_flow.dashboard.routes.finetune import annotation_sessions

    volume = zarr.open_group(str(session / "corrections" / "vol-1.zarr"), mode="w")
    volume.create_group("annotation").create_dataset("s0", shape=(8, 8, 8), chunks=(4, 4, 4), dtype="u1")
    volume.attrs["type"] = "annotation_volume"
    dashboard.post(MARK, json={})
    monkeypatch.setattr(annotation_sessions, "ensure_minio_serving",
                        lambda *a, **k: "http://m:9000/annotations/vol-1.zarr")
    monkeypatch.setattr(annotation_sessions, "refresh_annotated_regions_layer", lambda *a, **k: 0)
    monkeypatch.setattr(g, "annotation_volumes", {})
    resumed = dashboard.post("/api/finetune/load-existing-volume",
                             json={"source_session_path": str(session), "output_path": str(tmp_path / "next")})
    new_session = resumed.get_json()["new_session_path"]
    assert dashboard.get("/api/finetune/good-regions").get_json()["count"] == 1
    with open(f"{new_session}/loaded_from.json") as f:
        lineage = json.load(f)
    assert lineage["copied_good_regions"] is True
