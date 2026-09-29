"""finetune.session.store.SessionStore: sessions per base path, and the volume registry."""

import os

import pytest

from cellmap_flow.finetune.session.store import SessionStore


def test_a_base_path_keeps_its_session_and_disk_has_the_latest_trainable_one(tmp_path):
    sessions = {}
    store = SessionStore(sessions)
    first = store.get_or_create(str(tmp_path / "base"))
    assert os.path.dirname(first) == str(tmp_path / "base")
    assert store.get_or_create(str(tmp_path / "base")) == first == sessions[str(tmp_path / "base")]
    assert SessionStore(sessions).get_or_create(str(tmp_path / "other")) != first

    for name, trainable in [("20260101_120000", True), ("20260102_090000", False),
                            ("not_a_session", True)]:
        corrections = tmp_path / "base" / name / "corrections"
        corrections.mkdir(parents=True)
        if trainable:
            (corrections / "_virtual_sources.json").write_text("{}")
    assert store.latest_on_disk(str(tmp_path / "base")) == str(tmp_path / "base" / "20260101_120000")
    assert store.latest_on_disk(str(tmp_path / "missing")) is None


@pytest.mark.parametrize("keep, expected_state", [(False, {}), (True, {"0.0.0": "etag"})])
def test_registering_a_volume(keep, expected_state):
    volumes = {"vol": {"zarr_path": "/old.zarr", "chunk_sync_state": {"0.0.0": "etag"}}}
    store = SessionStore({}, volumes)
    record = store.register_volume("vol", keep_sync_state=keep, zarr_path="/new.zarr", minio_url="u")
    assert store.volumes() is volumes and volumes["vol"] is record
    assert record == {"zarr_path": "/new.zarr", "minio_url": "u", "chunk_sync_state": expected_state}
