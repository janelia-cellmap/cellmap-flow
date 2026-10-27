"""Syncing painted chunks from MinIO to disk (finetune/session/sync): what used to go missing.

- Change detection keyed on LastModified (one-second resolution) from a HEAD
  per chunk; two strokes to one chunk within a second lost the second.
"""

import json
from pathlib import Path

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.finetune.session import minio as session_minio
from cellmap_flow.finetune.session import sync

S0 = "annotations/vol.zarr/annotation/s0"


class FakeS3:
    """Just enough of s3fs for the sync: ls(detail=True), get, exists, cat."""

    def __init__(self):
        self.objects = {}  # path -> (bytes, etag)

    def put(self, path, data, etag):
        self.objects[path] = (data, etag)

    def ls(self, path, detail=False):
        prefix = path.rstrip("/") + "/"
        children = sorted({prefix + p[len(prefix):].split("/")[0] for p in self.objects if p.startswith(prefix)})
        if not children:
            raise FileNotFoundError(path)
        if not detail:
            return children
        return [
            {"name": c, "ETag": f'"{self.objects[c][1]}"', "size": len(self.objects[c][0]),
             "LastModified": "2026-09-28T12:00:00"}
            if c in self.objects else {"name": c, "type": "directory"}
            for c in children
        ]

    def get(self, src, dst):
        Path(dst).write_bytes(self.objects[src][0])

    def cat(self, path):
        return self.objects[path][0]

    def exists(self, path):
        return any(p == path or p.startswith(path.rstrip("/") + "/") for p in self.objects)


def test_a_second_write_within_the_same_second_is_synced(tmp_path):
    s3 = FakeS3()
    s3.put(f"{S0}/0.0.0", b"first", "etag-1")
    _, state = sync.diff_and_sync_chunks(s3, S0, tmp_path, {})
    s3.put(f"{S0}/0.0.0", b"second", "etag-2")  # same LastModified as before

    changed, state = sync.diff_and_sync_chunks(s3, S0, tmp_path, state)

    assert changed == ["0.0.0"]
    assert (tmp_path / "0.0.0").read_bytes() == b"second"


def test_a_periodic_round_pulls_strokes_and_leaves_the_viewer_alone(tmp_path, monkeypatch, viewer):
    """Any push to the viewer takes the brush out of the user's hand, and the
    round runs every 30 s while they draw; the boxes are redrawn from a button."""
    pushes = []
    real_txn = viewer.txn
    monkeypatch.setattr(viewer, "txn", lambda *a, **k: pushes.append(1) or real_txn(*a, **k))
    s3 = FakeS3()
    s3.put("annotations/vol.zarr/.zattrs", b'{"type": "annotation_volume"}', "e0")
    s3.put("annotations/vol.zarr/annotation/s0/0.0.0", b"stroke", "e1")
    monkeypatch.setattr(session_minio, "make_s3_filesystem", lambda state: s3)
    monkeypatch.setattr(sync, "sync_zarr_group_metadata", lambda *a: set())
    corrections = tmp_path / "corrections"
    (corrections / "vol.zarr").mkdir(parents=True)
    (corrections / "vol.zarr" / ".zattrs").write_text(json.dumps({  # so a redraw would have a box to draw
        "type": "annotation_volume", "output_voxel_size": [16] * 3, "chunk_size": [56] * 3,
        "dataset_offset_nm": [8] * 3}))
    state = {"ip": "127.0.0.1", "port": 9000, "bucket": "annotations", "output_base": str(corrections)}
    volumes = {"vol": {"zarr_path": str(corrections / "vol.zarr"), "corrections_dir": str(corrections),
                       "chunk_sync_state": {}}}
    monkeypatch.setattr(get_session(), "annotation_volumes", volumes)

    sync.periodic_sync_once(state=state, volumes=volumes)

    assert (corrections / "vol.zarr" / "annotation" / "s0" / "0.0.0").read_bytes() == b"stroke"
    assert pushes == []


def test_a_sync_registers_no_volume_and_the_sessions_volume_is_its_manifests(tmp_path, monkeypatch):
    """A resumed session got every volume of its source with the .minio that
    held them, and the periodic sync registered each one it found in the
    bucket: the session's volume, the last registered, became the original
    4 nm paint, while the manifest trained the newest volume."""
    from cellmap_flow.finetune.session.store import SessionStore

    s3 = FakeS3()
    corrections = tmp_path / "corrections"
    for name in ("vol-new", "vol-old"):
        s3.put(f"annotations/{name}.zarr/.zattrs", b'{"type": "annotation_volume"}', "e0")
        s3.put(f"annotations/{name}.zarr/annotation/s0/0.0.0", b"stroke", "e1")
        (corrections / f"{name}.zarr").mkdir(parents=True)
        (corrections / f"{name}.zarr" / ".zattrs").write_text(json.dumps({"type": "annotation_volume"}))
    (corrections / "_virtual_sources.json").write_text(json.dumps({"volume_zarr_path": str(corrections / "vol-new.zarr")}))
    monkeypatch.setattr(session_minio, "make_s3_filesystem", lambda state: s3)
    monkeypatch.setattr(sync, "sync_zarr_group_metadata", lambda *a: set())
    state = {"ip": "127.0.0.1", "port": 9000, "bucket": "annotations", "output_base": str(corrections)}
    volumes = {"vol-new": {"zarr_path": str(corrections / "vol-new.zarr"), "corrections_dir": str(corrections),
                           "chunk_sync_state": {}}}

    assert sync.sync_all(force=False, state=state, volumes=volumes) == 1
    assert list(volumes) == ["vol-new"] and not (corrections / "vol-old.zarr" / "annotation").exists()
    # Registered after it all the same, the manifest's volume is the session's.
    store = SessionStore({}, volumes)
    store.register_volume("vol-old", zarr_path=str(corrections / "vol-old.zarr"), corrections_dir=str(corrections))
    assert store.session_volume()[0] == "vol-new"
