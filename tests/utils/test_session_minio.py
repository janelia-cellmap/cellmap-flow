"""Starting the dashboard's MinIO: ready when it says so, recorded when set up.

- It slept 3 s instead of asking MinIO whether it was ready.
- It recorded the process before creating the bucket and its policy, so a
  failure there left a "running" server with no bucket and no sync thread.
- Its output went to pipes nobody read, which blocks it after ~64 KB.
- `mc alias set` wrote the shared ~/.mc config, so two dashboards
  repointed each other's alias.
- Nothing stopped two requests from starting two servers.
- Local chunks were mirrored over strokes still only in MinIO.
No real MinIO or mc runs here: subprocess and s3fs are faked.
"""

import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from cellmap_flow.dashboard import finetune_utils as fu
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.finetune.session import minio as session_minio
from cellmap_flow.finetune.session import sync as session_sync


class _Proc:
    started = 0

    def __init__(self, cmd, env=None, stdout=None, stderr=None):
        _Proc.started += 1
        self.cmd, self.stdout = cmd, stdout
        self.pid = 4242
        self.terminated = False

    def poll(self):
        return 0 if self.terminated else None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.terminated = True


@pytest.fixture
def fake_minio(monkeypatch, tmp_path):
    runs = []
    procs = []

    def fake_run(cmd, **kwargs):
        runs.append((cmd, kwargs.get("env") or {}))
        if fake_run.fail_on and fake_run.fail_on in cmd:
            raise subprocess.CalledProcessError(1, cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    fake_run.fail_on = None

    def fake_popen(*a, **k):
        procs.append(_Proc(*a, **k))
        return procs[-1]

    calls = []

    def recorded(name, value=None):
        return lambda *a, **k: calls.append(name) or value

    _Proc.started = 0
    monkeypatch.setattr(get_session(), "minio_state", {"process": None, "bucket": "annotations",
                                                        "output_base": None, "sync_thread": None})
    monkeypatch.setattr(fu, "_require_minio_binaries", recorded("preflight"))
    monkeypatch.setattr(session_minio, "get_local_ip", recorded("ip", "127.0.0.1"))
    monkeypatch.setattr(session_minio, "find_available_port", recorded("port", 9123))
    monkeypatch.setattr(session_sync, "start_periodic_sync", recorded("sync thread"))
    monkeypatch.setattr(session_minio, "wait_for_ready", recorded("ready", True))
    # The pull before the mirror asks MinIO whether it has the volume: no.
    bucket = SimpleNamespace(exists=recorded("exists", False))
    monkeypatch.setattr(session_minio, "make_s3_filesystem", lambda state: bucket)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(subprocess, "run", fake_run)
    fake_run.calls = calls
    return runs, procs, fake_run


def test_minio_logs_to_a_file_and_mc_uses_its_own_alias(fake_minio, tmp_path):
    runs, procs, fake_run = fake_minio
    corrections = tmp_path / "corrections"
    url = fu.ensure_minio_serving(str(corrections / "vol.zarr"), "vol", output_base_dir=str(corrections))

    assert url == "http://127.0.0.1:9123/annotations/vol.zarr"
    assert procs[0].stdout is not subprocess.PIPE and hasattr(procs[0].stdout, "write")
    assert (corrections / ".minio.log").exists()
    assert not any(cmd[:3] == ["mc", "alias", "set"] for cmd, _ in runs), "no shared ~/.mc config"
    for cmd, env in runs:
        assert env.get(f"MC_HOST_{session_minio.MC_ALIAS}") == "http://minio:minio123@127.0.0.1:9123", cmd
    assert get_session().minio_state["process"] is procs[0]
    assert fake_run.calls == ["preflight", "ip", "port", "ready", "sync thread", "exists"]


@pytest.mark.parametrize("failure", [pytest.param("bucket-policy", id="setting-up-the-bucket-fails"),
                                     pytest.param("never-ready", id="never-ready")])
def test_a_minio_that_does_not_come_up_is_not_left_running(fake_minio, tmp_path, monkeypatch, failure):
    """It was recorded before its bucket and policy were set up, so a failure
    there left a "running" server with no bucket and no sync thread."""
    runs, procs, fake_run = fake_minio
    if failure == "bucket-policy":
        fake_run.fail_on = "anonymous"
        raises = pytest.raises(subprocess.CalledProcessError)
    else:
        monkeypatch.setattr(session_minio, "wait_for_ready", lambda ip, port, proc, timeout=0: False)
        raises = pytest.raises(RuntimeError, match="did not become ready")
    with raises:
        fu.ensure_minio_serving(str(tmp_path / "vol.zarr"), "vol", output_base_dir=str(tmp_path))
    assert get_session().minio_state["process"] is None and procs[0].terminated


def test_two_requests_start_one_server(fake_minio, tmp_path, monkeypatch):
    def slow_ready(ip, port, proc, timeout=0):
        time.sleep(0.2)
        return True

    monkeypatch.setattr(session_minio, "wait_for_ready", slow_ready)
    threads = [
        threading.Thread(
            target=fu.ensure_minio_serving,
            args=(str(tmp_path / f"v{i}.zarr"), f"v{i}"),
            kwargs={"output_base_dir": str(tmp_path)},
        )
        for i in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert _Proc.started == 1
    runs, _, _ = fake_minio
    assert [cmd[1] for cmd, _ in runs].count("mirror") == 2, "both volumes were served"


def test_the_sync_thread_keeps_the_sessions_own_dicts(fake_minio, tmp_path, monkeypatch):
    """The periodic sync holds the state and volume dicts it was started with
    for as long as MinIO runs, so they must be the session's own, which the
    routes change in place. With a copy, a volume registered later would never
    be synced."""
    started = []
    monkeypatch.setattr(session_sync, "start_periodic_sync", lambda *args: started.append(args))
    fu.ensure_minio_serving(str(tmp_path / "vol.zarr"), "vol", output_base_dir=str(tmp_path))
    session = get_session()
    ((state, volumes),) = started
    assert state is session.minio_state and volumes is session.annotation_volumes


def test_readiness_is_asked_of_minio(monkeypatch):
    """The real readiness check polls /minio/health/ready."""
    import urllib.request

    asked = []

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(url, timeout=None):
        asked.append(url)
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert session_minio.wait_for_ready("127.0.0.1", 9123, _Proc(["minio"]), timeout=1)
    assert asked == ["http://127.0.0.1:9123/minio/health/ready"]


# --- pulling MinIO's strokes before local chunks are pushed over them --------------
# `mc mirror --overwrite` uploads every local chunk over MinIO's copy, and the
# browser paints straight into MinIO: a YAML import into a painted volume, or a
# resume, overwrote recent strokes with stale local chunks.


class _Alive:
    pid = 1

    def poll(self):
        return None


@pytest.fixture
def order(monkeypatch):
    events = []
    monkeypatch.setattr(get_session(), "minio_state", {"process": _Alive(), "ip": "127.0.0.1", "port": 9000,
                                                        "bucket": "annotations", "output_base": None})
    monkeypatch.setattr(fu, "_require_minio_binaries", lambda: None)
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **k: events.append(("mc", cmd[1])) or SimpleNamespace(returncode=0, stderr=""),
    )
    monkeypatch.setattr(
        session_sync, "sync_volume",
        lambda volume_id, force=False, zarr_path=None, **k: events.append(("pull", volume_id, zarr_path)),
    )
    return events


@pytest.mark.parametrize("in_minio, pulled", [pytest.param(True, True, id="painted-in-minio"),
                                              pytest.param(False, False, id="nothing-in-minio")])
def test_painted_chunks_are_pulled_before_the_mirror(order, monkeypatch, tmp_path, in_minio, pulled):
    asked = []
    monkeypatch.setattr(session_minio, "make_s3_filesystem", lambda state: SimpleNamespace(
        exists=lambda path: asked.append(path) or in_minio))
    fu.ensure_minio_serving(str(tmp_path / "vol-1.zarr"), "vol-1")
    assert asked == ["annotations/vol-1.zarr/annotation/s0"]
    pull = [("pull", "vol-1", str(tmp_path / "vol-1.zarr"))] if pulled else []
    assert order == pull + [("mc", "mirror")]


def test_a_yaml_import_pulls_strokes_before_writing_crops_into_the_sessions_latest_volume(monkeypatch, tmp_path):
    """The latest volume is the one being painted, and the one the good
    regions and training mean; the import took the first."""
    from cellmap_flow.dashboard.app import app
    from cellmap_flow.dashboard.routes.finetune import yaml_crops

    events = []
    meta = {
        "zarr_path": str(tmp_path / "vol-1.zarr"), "input_size": [8] * 3, "output_size": [4] * 3,
        "input_voxel_size": [8] * 3, "output_voxel_size": [8] * 3, "minio_url": "http://m/vol-1.zarr",
    }
    crops = SimpleNamespace(crops=[SimpleNamespace(path="/crop.zarr")], patches_per_epoch=None,
                            jitter_voxels=None, seed=0, dense_to_sparse_ratio=None)
    monkeypatch.setattr(yaml_crops, "parse_crops_yaml", lambda text: crops)
    monkeypatch.setattr(yaml_crops, "_get_selected_model_config", lambda name: (object(), None))
    monkeypatch.setattr(yaml_crops, "ensure_corrections_storage", lambda path: (None, str(tmp_path)))
    earlier = {**meta, "zarr_path": str(tmp_path / "vol-0.zarr")}
    monkeypatch.setattr(get_session(), "annotation_volumes", {"vol-0": {**earlier, "corrections_dir": str(tmp_path)},
                                                  "vol-1": {**meta, "corrections_dir": str(tmp_path)}})
    monkeypatch.setattr(yaml_crops, "sync_annotation_volume_from_minio",
                        lambda vid, **k: events.append(("pull", vid)))
    monkeypatch.setattr(yaml_crops, "write_crop_into_volume",
                        lambda m, entry, progress_callback=None: events.append(("write", entry.path))
                        or {"n_fg_voxels": 0})
    monkeypatch.setattr(yaml_crops, "ensure_minio_serving", lambda *a, **k: events.append(("mirror",)))
    monkeypatch.setattr(yaml_crops, "write_manifest", lambda *a: None)
    monkeypatch.setattr(yaml_crops, "refresh_annotated_regions_layer", lambda **k: None)
    monkeypatch.setattr(get_session(), "dataset_path", "/data/raw.zarr")

    response = app.test_client().post("/api/finetune/load-crops", json={"model_name": "m", "yaml": "crops: []"})

    assert response.get_json()["success"], response.get_json()
    assert events[:3] == [("pull", "vol-1"), ("write", "/crop.zarr"), ("mirror",)]
