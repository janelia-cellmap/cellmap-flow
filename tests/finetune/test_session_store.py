"""finetune.session.store.SessionStore: one session per base path, one record per volume."""

import os

import pytest

from cellmap_flow.finetune.session.store import SessionStore


def test_a_base_path_keeps_its_session(tmp_path):
    sessions = {}
    first = SessionStore(sessions).get_or_create(str(tmp_path / "base"))
    assert os.path.dirname(first) == str(tmp_path / "base")
    assert SessionStore(sessions).get_or_create(str(tmp_path / "base")) == first
    assert SessionStore(sessions).get_or_create(str(tmp_path / "other")) != first


@pytest.mark.parametrize("keep, expected_state", [(False, {}), (True, {"0.0.0": "etag"})])
def test_registering_a_volume(keep, expected_state):
    volumes = {"vol": {"zarr_path": "/old.zarr", "chunk_sync_state": {"0.0.0": "etag"}}}
    store = SessionStore({}, volumes)
    record = store.register_volume("vol", keep_sync_state=keep, zarr_path="/new.zarr", minio_url="u")
    assert store.volumes() is volumes and volumes["vol"] is record
    assert record == {"zarr_path": "/new.zarr", "minio_url": "u", "chunk_sync_state": expected_state}
