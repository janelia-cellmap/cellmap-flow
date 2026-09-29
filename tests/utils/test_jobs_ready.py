"""The ready file: a server writes its address, and the launcher reads it first.

On LSF a launcher learned where a server was only by polling bpeek, which
asks mbatchd for the job's whole output so far, twice a second for every job
it waits on. Told where to, a server now also writes its address to a file
as it prints the marker, and the launcher reads that before asking LSF
anything. A server from before this ignores the variable, and the launcher
falls back to bpeek.

LSF is never called: subprocess.run is a fake, and each test checks it was
called (or, where the point is that nothing was asked of LSF, that it was
not).
"""

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from cellmap_flow.globals import g
from cellmap_flow.jobs.lsf import LSFJob
from cellmap_flow.jobs.ready import READY_ENV, read_ready_file, ready_path, write_ready_file
from cellmap_flow.utils import bsub_utils
from cellmap_flow.utils.web_utils import IP_PATTERN

URL = "http://10.1.2.3:8123"
MARKER = f"{IP_PATTERN[0]}{URL}{IP_PATTERN[1]}"


# --- the file ------------------------------------------------------------------


def test_a_server_writes_its_address_where_it_was_told(tmp_path, monkeypatch):
    target = tmp_path / "logs" / "m_abcd1234.ready"
    monkeypatch.setenv(READY_ENV, str(target))
    monkeypatch.setenv("LSB_JOBID", "4242")

    assert write_ready_file(URL) == target

    data = json.loads(target.read_text())
    assert data["url"] == URL
    assert data["pid"] == os.getpid()
    assert data["job_id"] == "4242"
    assert data["host"]
    assert read_ready_file(target, job_id="4242") == URL
    assert list(target.parent.iterdir()) == [target], "no temporary file is left behind"


def test_without_the_variable_nothing_is_written(tmp_path, monkeypatch):
    monkeypatch.delenv(READY_ENV, raising=False)
    assert write_ready_file(URL) is None


def test_a_file_that_cannot_be_written_does_not_stop_the_server(tmp_path, monkeypatch):
    not_a_dir = tmp_path / "not_a_dir"
    not_a_dir.write_text("")
    monkeypatch.setenv(READY_ENV, str(not_a_dir / "m.ready"))

    assert write_ready_file(URL) is None


@pytest.mark.parametrize(
    "content", [None, "", '{"url": "http://10.1', "[]", '{"url": ""}', '{"pid": 12}']
)
def test_no_usable_file_yet_reads_as_nothing(tmp_path, content):
    path = tmp_path / "m.ready"
    if content is not None:
        path.write_text(content)
    assert read_ready_file(path) is None


def test_a_file_another_job_wrote_is_not_this_jobs(tmp_path):
    path = tmp_path / "m.ready"
    path.write_text(json.dumps({"url": URL, "job_id": "6"}))
    assert read_ready_file(path, job_id="7") is None
    assert read_ready_file(path, job_id="6") == URL

    # Written outside LSF (no LSB_JOBID), there is nothing to compare.
    path.write_text(json.dumps({"url": URL}))
    assert read_ready_file(path, job_id="7") == URL


def test_every_submission_gets_a_path_of_its_own(tmp_path):
    first, second = ready_path(tmp_path, "../mito model"), ready_path(tmp_path, "../mito model")
    assert first != second
    assert first.parent == second.parent == tmp_path
    assert first.name.startswith("mito_model_") and first.suffix == ".ready"


# --- the server writes it ---------------------------------------------------------


def test_the_server_writes_it_as_it_prints_its_marker(tmp_path, monkeypatch, capsys):
    from cellmap_flow import server

    target = tmp_path / "m.ready"
    monkeypatch.setenv(READY_ENV, str(target))
    monkeypatch.setattr(server, "get_public_ip", lambda: "10.1.2.3")
    served = []

    def app_run(**kwargs):
        served.append((kwargs["port"], read_ready_file(target)))

    fake_server = SimpleNamespace(app=SimpleNamespace(run=app_run))
    server.CellMapFlowServer.run(fake_server, port=8123)

    assert served == [(8123, URL)], "written before the port is bound"
    assert MARKER in capsys.readouterr().out, "and the marker is still printed"


# --- the launcher reads it --------------------------------------------------------


class FakeLSF:
    """bjobs says RUN and bpeek shows ``bpeek_out``; every call is recorded."""

    def __init__(self, monkeypatch, bpeek_out=f"loading\n{MARKER}\n"):
        self.calls = []
        self.bpeek_out = bpeek_out
        monkeypatch.setattr(subprocess, "run", self.run)

    def run(self, argv, **kwargs):
        self.calls.append(argv[0])
        if argv[0] == "bjobs":
            out = f"{argv[-1]} me RUN gpu_h100 login h01 m Sep 28 10:00\n"
            return subprocess.CompletedProcess(argv, 0, out, "")
        if argv[0] == "bpeek":
            return subprocess.CompletedProcess(argv, 0, self.bpeek_out, "")
        raise AssertionError(f"unexpected command {argv}")


def test_a_job_that_wrote_its_file_is_found_without_asking_lsf(tmp_path, monkeypatch):
    lsf = FakeLSF(monkeypatch)
    path = tmp_path / "m.ready"
    path.write_text(json.dumps({"url": URL, "job_id": "7"}))

    job = LSFJob("7", ready_file=path)

    assert job.wait_for_host(timeout=30) == URL
    assert job.host == URL
    assert lsf.calls == [], "neither bjobs nor bpeek"
    assert not path.exists(), "read once, then removed"


def test_a_server_that_ignores_the_variable_is_found_through_bpeek(tmp_path, monkeypatch):
    lsf = FakeLSF(monkeypatch)

    job = LSFJob("7", ready_file=tmp_path / "never_written.ready")

    assert job.wait_for_host(timeout=30) == URL
    assert lsf.calls == ["bjobs", "bpeek"]


def test_a_file_left_by_another_job_is_ignored(tmp_path, monkeypatch):
    lsf = FakeLSF(monkeypatch)
    path = tmp_path / "m.ready"
    path.write_text(json.dumps({"url": "http://stale:1", "job_id": "6"}))

    assert LSFJob("7", ready_file=path).wait_for_host(timeout=30) == URL
    assert "bpeek" in lsf.calls


def test_start_hosts_hands_each_job_a_ready_file_and_reads_it(tmp_path, monkeypatch):
    monkeypatch.setattr(bsub_utils, "SERVER_LOG_DIR", tmp_path)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv[0])
        if argv[:2] == ["which", "bsub"]:
            return subprocess.CompletedProcess(argv, 0, b"/usr/bin/bsub\n", b"")
        if argv[0] == "bjobs":
            return subprocess.CompletedProcess(argv, 255, "", f"Job <{argv[-1]}> is not found\n")
        if argv[0] == "bsub":
            # What the server does once it runs, with the environment LSF
            # copies into the job.
            target = Path(kwargs["env"][READY_ENV])
            assert target.parent == tmp_path
            target.write_text(json.dumps({"url": URL, "job_id": "4242"}))
            return subprocess.CompletedProcess(argv, 0, "Job <4242> is submitted to queue <gpu_h100>.\n", "")
        raise AssertionError(f"unexpected command {argv}")

    monkeypatch.setattr(subprocess, "run", run)
    g.jobs = []

    job = bsub_utils.start_hosts("serve", queue="gpu_h100", job_name="m", cycle_queues=False)

    assert job.host == URL
    assert calls == ["which", "bjobs", "bsub"], "no bpeek, and no bjobs for the job itself"
    assert g.jobs == [job]


def test_the_file_is_looked_at_between_lsf_polls(tmp_path, monkeypatch):
    """LSF is asked less and less often; the file, which costs it nothing, is not."""
    from cellmap_flow.jobs import lsf as jobs_lsf

    clock = {"now": 0.0}
    path = tmp_path / "m.ready"

    def sleep(seconds):
        clock["now"] += seconds
        if clock["now"] >= 6.0 and not path.exists():
            path.write_text(json.dumps({"url": URL, "job_id": "7"}))

    monkeypatch.setattr(
        jobs_lsf, "time",
        SimpleNamespace(time=lambda: clock["now"], monotonic=lambda: clock["now"], sleep=sleep),
    )
    lsf = FakeLSF(monkeypatch, bpeek_out="loading\n")

    assert LSFJob("7", ready_file=path).wait_for_host(timeout=60) == URL

    # Polls at 0, 0.5, 1.5 and 3.5 s; the next would be at 7.5, and the file
    # written at 6 s is seen at once.
    assert lsf.calls == ["bjobs", "bpeek"] * 4
    assert clock["now"] == 6.0
