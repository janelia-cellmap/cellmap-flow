"""Syncing painted chunks from MinIO to disk (finetune/session/sync): what used to go missing.

- Change detection keyed on LastModified (one-second resolution) from a HEAD
  per chunk; two strokes to one chunk within a second lost the second.
- A chunk that failed to download was still recorded as synced, so it was
  never retried.
- Chunks were written in place while the trainer read the same volume.
- A metadata mismatch re-created the local array with overwrite=True,
  deleting every local chunk of it.
- Strokes went to <first output_base>/<volume>.zarr instead of the volume's
  own zarr_path.
- Concurrent syncs ran at once, and periodic-sync failures logged at DEBUG.
"""

import json
import logging
import threading
import time
from pathlib import Path

import pytest
import zarr

from cellmap_flow.dashboard import finetune_utils as fu
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.finetune.session import minio as session_minio
from cellmap_flow.finetune.session import sync

S0 = "annotations/vol.zarr/annotation/s0"


class FakeS3:
    """Just enough of s3fs for the sync: ls(detail=True), get, exists, cat."""

    def __init__(self):
        self.objects = {}  # path -> (bytes, etag)
        self.fail = set()
        self.partial = set()

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
        data = self.objects[src][0]
        if src in self.partial:
            Path(dst).write_bytes(data[:1])
            raise OSError("connection reset")
        if src in self.fail:
            raise OSError("connection reset")
        Path(dst).write_bytes(data)

    def cat(self, path):
        return self.objects[path][0]

    def exists(self, path):
        return any(p == path or p.startswith(path.rstrip("/") + "/") for p in self.objects)


def test_a_second_write_within_the_same_second_is_synced(tmp_path):
    s3 = FakeS3()
    s3.put(f"{S0}/0.0.0", b"first", "etag-1")
    _, _, state = sync.diff_and_sync_chunks(s3, S0, tmp_path, {})
    s3.put(f"{S0}/0.0.0", b"second", "etag-2")  # same LastModified as before

    changed, _, state = sync.diff_and_sync_chunks(s3, S0, tmp_path, state)

    assert changed == ["0.0.0"]
    assert (tmp_path / "0.0.0").read_bytes() == b"second"


def test_a_chunk_that_failed_to_download_is_retried(tmp_path):
    s3 = FakeS3()
    s3.put(f"{S0}/0.0.0", b"stroke", "e1")
    s3.put(f"{S0}/0.0.1", b"other", "e2")
    s3.fail.add(f"{S0}/0.0.0")

    changed, _, state = sync.diff_and_sync_chunks(s3, S0, tmp_path, {})
    assert changed == ["0.0.1"]
    assert "0.0.0" not in state

    s3.fail.clear()
    changed, _, state = sync.diff_and_sync_chunks(s3, S0, tmp_path, state)
    assert changed == ["0.0.0"]
    assert (tmp_path / "0.0.0").read_bytes() == b"stroke"


def test_a_failed_download_leaves_the_old_chunk_whole(tmp_path):
    (tmp_path / "0.0.0").write_bytes(b"old chunk")
    s3 = FakeS3()
    s3.put(f"{S0}/0.0.0", b"new chunk", "e1")
    s3.partial.add(f"{S0}/0.0.0")

    sync.diff_and_sync_chunks(s3, S0, tmp_path, {})

    assert (tmp_path / "0.0.0").read_bytes() == b"old chunk"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["0.0.0"], "no temp file left behind"


def test_a_layout_mismatch_never_deletes_local_chunks(tmp_path, monkeypatch):
    remote = {}
    zarr.open_group(store=remote, mode="w").create_dataset(
        "s0", shape=(8, 8, 8), chunks=(4, 4, 4), dtype="u1"
    )
    monkeypatch.setattr(sync.s3fs, "S3Map", lambda root, s3: remote)
    local = zarr.open_group(store=zarr.DirectoryStore(str(tmp_path)), mode="a")
    local.create_dataset("s0", shape=(4, 4, 4), chunks=(2, 2, 2), dtype="u1")
    local["s0"][:] = 2

    mismatched = sync.sync_zarr_group_metadata(None, "annotations/vol.zarr/annotation", tmp_path)

    assert mismatched == {"s0"}
    again = zarr.open_group(store=zarr.DirectoryStore(str(tmp_path)), mode="r")["s0"]
    assert again.shape == (4, 4, 4) and (again[:] == 2).all()


def test_strokes_go_to_the_volumes_own_zarr(tmp_path, monkeypatch):
    first_session = tmp_path / "A" / "corrections"
    other_session = tmp_path / "B" / "corrections"
    vol_b = other_session / "vol-b.zarr"
    s3 = FakeS3()
    s3.put("annotations/vol-b.zarr/annotation/s0/0.0.0", b"stroke", "e1")
    monkeypatch.setattr(get_session(), "minio_state", {
        "ip": "127.0.0.1", "port": 9000, "bucket": "annotations",
        "output_base": str(first_session),
    })
    monkeypatch.setattr(get_session(), "annotation_volumes", {
        "vol-b": {"zarr_path": str(vol_b), "corrections_dir": str(other_session),
                  "chunk_sync_state": {}},
    })
    synced_metadata = []
    monkeypatch.setattr(session_minio, "make_s3_filesystem", lambda state: s3)
    monkeypatch.setattr(sync, "sync_zarr_group_metadata", lambda *a: synced_metadata.append(a[1]) or set())
    (other_session / "_virtual_sources.json").parent.mkdir(parents=True)
    (other_session / "_virtual_sources.json").write_text("{}")

    assert fu.sync_annotation_volume_from_minio("vol-b")
    assert synced_metadata == ["annotations/vol-b.zarr/annotation"]

    assert (vol_b / "annotation" / "s0" / "0.0.0").read_bytes() == b"stroke"
    assert not (first_session / "vol-b.zarr").exists()


def test_only_one_sync_runs_at_a_time(monkeypatch):
    calls = []

    class _Listing(FakeS3):
        def ls(self, path, detail=False):
            calls.append(path)
            return []

    monkeypatch.setattr(get_session(), "minio_state", {"ip": "127.0.0.1", "port": 9000, "bucket": "annotations"})
    monkeypatch.setattr(session_minio, "make_s3_filesystem", lambda state: _Listing())

    with sync._sync_lock:
        worker = threading.Thread(target=fu.sync_all_annotations_from_minio, kwargs={"force": False})
        worker.start()
        time.sleep(0.3)
        assert calls == [], "a second sync started while one was running"
    worker.join(5)
    assert calls == ["annotations"]


def test_a_failing_periodic_sync_warns_once_per_interval(monkeypatch, caplog):
    def broken(force=True, **kw):
        raise ConnectionError("MinIO is gone")

    state = {"ip": "127.0.0.1", "port": 9000, "output_base": "/x"}
    monkeypatch.setattr(sync, "sync_all", broken)
    monkeypatch.setattr(sync, "_sync_failures", {"count": 0, "last_warned": None})

    with caplog.at_level(logging.WARNING, logger=sync.logger.name):
        sync.periodic_sync_once(state=state, volumes={})
        sync.periodic_sync_once(state=state, volumes={})

    assert [r.levelno for r in caplog.records if r.name == sync.logger.name] == [logging.WARNING]
    assert sync._sync_failures["count"] == 2


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


@pytest.fixture(autouse=True)
def _small_pool(monkeypatch):
    monkeypatch.setattr(sync, "worker_count", lambda: 2)
