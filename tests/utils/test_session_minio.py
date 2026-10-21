"""Starting the dashboard's MinIO: ready when it says so, recorded when set up.

- It recorded the process before creating the bucket and its policy, so a
  failure there left a "running" server with no bucket and no sync thread.
- Its output went to pipes nobody read, which blocks it after ~64 KB.
- `mc alias set` wrote the shared ~/.mc config, so two dashboards
  repointed each other's alias.
- Local chunks were mirrored over strokes still only in MinIO.
No real MinIO or mc runs here: subprocess and s3fs are faked.
"""

import subprocess
from types import SimpleNamespace

import pytest

from cellmap_flow.dashboard import finetune_utils as fu
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.finetune.session import minio as session_minio
from cellmap_flow.finetune.session import sync as session_sync


class _Proc:
    def __init__(self, cmd, env=None, stdout=None, stderr=None):
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

    monkeypatch.setattr(get_session(), "minio_state", {"process": None, "bucket": "annotations",
                                                        "output_base": None, "sync_thread": None})
    monkeypatch.setattr(fu, "_require_minio_binaries", recorded("preflight"))
    monkeypatch.setattr(session_minio, "this_machine_ip", recorded("ip", "127.0.0.1"))
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
