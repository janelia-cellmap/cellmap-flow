"""Starting the dashboard's MinIO: ready when it says so, recorded when set up.

- It slept 3 s instead of asking MinIO whether it was ready.
- It recorded the process before creating the bucket and its policy, so a
  failure there left a "running" server with no bucket and no sync thread.
- Its output went to pipes nobody read, which blocks it after ~64 KB.
- `mc alias set` wrote the shared ~/.mc config, so two dashboards
  repointed each other's alias.
- Nothing stopped two requests from starting two servers.
No real MinIO or mc runs here: subprocess and s3fs are faked.
"""

import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from cellmap_flow.dashboard import finetune_utils as fu
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
    monkeypatch.setattr(fu, "minio_state", {"process": None, "bucket": "annotations",
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
        assert env.get(f"MC_HOST_{fu.MC_ALIAS}") == "http://minio:minio123@127.0.0.1:9123", cmd
    assert fu.minio_state["process"] is procs[0]
    assert fake_run.calls == ["preflight", "ip", "port", "ready", "sync thread", "exists"]


def test_a_failed_setup_leaves_no_half_started_server(fake_minio, tmp_path):
    runs, procs, fake_run = fake_minio
    fake_run.fail_on = "anonymous"
    with pytest.raises(subprocess.CalledProcessError):
        fu.ensure_minio_serving(str(tmp_path / "vol.zarr"), "vol", output_base_dir=str(tmp_path))
    assert fu.minio_state["process"] is None
    assert procs[0].terminated


def test_a_server_that_never_gets_ready_is_not_used(fake_minio, tmp_path, monkeypatch):
    _, procs, _ = fake_minio
    monkeypatch.setattr(session_minio, "wait_for_ready", lambda ip, port, proc, timeout=0: False)
    with pytest.raises(RuntimeError, match="did not become ready"):
        fu.ensure_minio_serving(str(tmp_path / "vol.zarr"), "vol", output_base_dir=str(tmp_path))
    assert fu.minio_state["process"] is None
    assert procs[0].terminated


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
