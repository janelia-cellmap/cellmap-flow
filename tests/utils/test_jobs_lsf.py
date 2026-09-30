"""jobs.lsf and jobs.ready against canned LSF answers (conftest's ``fake_lsf``).

What each submission hands to bsub is pinned in test_bsub_argv_snapshot;
these are how answers are read: the job id, a submission that timed out,
many jobs' states in one bjobs call, the wait for a server's address, and
the ready file a server writes it to.
"""

import json
import logging
import subprocess
from datetime import datetime
from types import SimpleNamespace

import pytest

from cellmap_flow.finetune.finetune_job_manager import FinetuneJobManager, JobStatus
from cellmap_flow.jobs import lsf as jobs_lsf
from cellmap_flow.jobs.lsf import BsubTimeoutError, LSFJob
from cellmap_flow.jobs.ready import READY_ENV, read_ready_file, ready_path, write_ready_file
from cellmap_flow.jobs.spec import IP_PATTERN, JobSpec, JobStatus as LSFJobStatus

URL = "http://node7:4321"
MARKER = f"{IP_PATTERN[0]}{URL}{IP_PATTERN[1]}"
NOT_STARTED = (255, "", "Job <1> : Not yet started.")


def _bjobs(job_id, stat="RUN", name="m"):
    return f"{job_id}   me   {stat}   gpu_h100   login1   4*h10u05   {name}   Sep 28 10:00\n"


@pytest.mark.parametrize(
    "bsub, bjobs, expected",
    [
        # Anything printed before "Job <id> is submitted" (an esub notice) is not the id.
        pytest.param(["Your job requests 12 slots per GPU.\nJob <4242> is submitted to queue <gpu_h100>.\n"], [],
                     "4242", id="after-a-notice"),
        pytest.param(["Request aborted by esub. Job not submitted.\n"], [], (RuntimeError, "job id"), id="no-id"),
        # bsub timed out, but LSF made the job: the new one, not an older one of that name.
        pytest.param([subprocess.TimeoutExpired("bsub", 30)], [_bjobs(111), _bjobs(111) + _bjobs(4243)], "4243",
                     id="timed-out-and-made"),
        # Or it can't be found: LSF may still make it, so the caller must not submit again.
        pytest.param([subprocess.TimeoutExpired("bsub", 30)], [_bjobs(111)], (BsubTimeoutError, "bjobs -a -J m"),
                     id="timed-out-and-not-found"),
    ],
)
def test_a_submissions_job_id(fake_lsf, tmp_path, bsub, bjobs, expected):
    fake_lsf.answers["bsub"] = bsub
    fake_lsf.answers["bjobs"] = bjobs or [(255, "", "not found")]
    spec = JobSpec(name="m", shell="serve", queue="gpu_h100", log_dir=tmp_path)
    if isinstance(expected, tuple):
        with pytest.raises(expected[0], match=expected[1]):
            jobs_lsf.submit(spec)
        return
    job = jobs_lsf.submit(spec)
    assert (job.job_id, job.log_file.name) == (expected, f"m_{expected}.log")
    assert len(fake_lsf.commands("bsub")) == 1


BJOBS_OUT = (
    "201   me  RUN   gpu_h100  login1  4*h10u05  finetune_a  Sep 28 10:00\n"
    "      h10u06\n"  # a continuation line: more hosts of the job above
    "202   me  PEND  gpu_h100  login1            finetune_b  Sep 28 10:01\n"
    "203   me  DONE  gpu_h100  login1  h10u07    finetune_c  Sep 28 09:00\n"
    "204   me  EXIT  gpu_h100  login1  h10u08    finetune_d  Sep 28 09:30\n"
    "205   me  USUSP gpu_h100  login1  h10u09    finetune_e  Sep 28 09:40\n"
)
S = LSFJobStatus


@pytest.mark.parametrize(
    "ids, answer, expected",
    [
        # bjobs exits non-zero when any job is not found; 207, which it said
        # nothing about, is left out: not known either way.
        (["201", 202, "203", "204", "205", "206", "207", "201"], (255, BJOBS_OUT, "Job <206> is not found\n"),
         {"201": S.RUNNING, "202": S.PENDING, "203": S.COMPLETED, "204": S.FAILED, "205": S.RUNNING, "206": None}),
        ([], None, {}),  # nothing asked
        (["201", "202"], FileNotFoundError("bjobs"), {}),
        (["201", "202"], subprocess.TimeoutExpired("bjobs", 10), {}),
    ],
    ids=["one-call", "no-ids", "no-bjobs", "bjobs-hangs"],
)
def test_one_bjobs_call_answers_for_every_job(fake_lsf, ids, answer, expected):
    fake_lsf.answers["bjobs"] = [answer]
    assert jobs_lsf.statuses(ids) == expected
    asked = list(dict.fromkeys(map(str, ids)))  # once each, in order
    assert fake_lsf.calls == ([["bjobs", "-noheader", *asked]] if ids else [])


def _session(tmp_path, runs):
    """A finetune session's runs on disk: {name: (LSF job id, status)}."""
    session = tmp_path / "20260101_120000"
    for name, (lsf_id, status) in runs.items():
        (session / "runs" / name).mkdir(parents=True)
        (session / "runs" / name / "metadata.json").write_text(json.dumps({
            "job_id": name, "model_name": "m", "created_at": datetime.now().isoformat(),
            "params": {"num_epochs": 5}, "lsf_job_id": lsf_id, "status": status,
        }))
    return session


def _manager(monkeypatch):
    manager = FinetuneJobManager()
    monkeypatch.setattr(manager, "_start_monitor", lambda job: None)
    return manager


def test_a_job_lsf_has_forgotten_is_recorded_as_over_and_not_asked_about_again(fake_lsf, tmp_path, monkeypatch):
    session = _session(tmp_path, {"alive": ("301", "RUNNING"), "forgotten": ("302", "RUNNING"),
                                  "unanswered": ("303", "RUNNING")})
    fake_lsf.answers["bjobs"] = [(255, _bjobs(301), "Job <302> is not found\n")]
    manager = _manager(monkeypatch)

    assert manager.rehydrate_session(session) == 1

    def status(name):
        return json.loads((session / "runs" / name / "metadata.json").read_text())

    assert fake_lsf.calls == [["bjobs", "-noheader", "301", "302", "303"]], "one call for the session"
    assert manager.jobs["alive"].status == JobStatus.RUNNING
    assert status("forgotten")["status"] == "FAILED" and "302" in status("forgotten")["status_detail"]
    assert status("unanswered")["status"] == "RUNNING", "bjobs said nothing: ask again next time"
    manager.rehydrate_session(session)
    assert fake_lsf.calls[1:] == [["bjobs", "-noheader", "303"]]


def test_a_session_with_nothing_unfinished_asks_lsf_nothing(fake_lsf, tmp_path, monkeypatch):
    session = _session(tmp_path, {"done": ("401", "COMPLETED"), "local": ("PID:12", "RUNNING")})
    assert _manager(monkeypatch).rehydrate_session(session) == 0
    assert fake_lsf.calls == []


# --- waiting for a server's address ---------------------------------------------


def test_the_polling_timeline(fake_lsf):
    """For a job that queues, loads, then serves. Pinned so that a change of
    cadence is a visible diff here rather than a surprise in mbatchd's load."""
    fake_lsf.answers["bjobs"] = lambda argv, kw: _bjobs(argv[-1], "PEND" if fake_lsf.now < 1002 else "RUN")
    fake_lsf.answers["bpeek"] = [NOT_STARTED] * 4 + [(0, "loading\n", "")] * 2 + [(0, f"loading\n{MARKER}\n", "")]

    assert LSFJob("1").wait_for_host(timeout=60) == URL
    assert fake_lsf.timeline == [(t, c) for t in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0) for c in ("bjobs", "bpeek")]
    assert fake_lsf.sleeps == 6


@pytest.mark.parametrize(
    "stat, bpeek, timeout, expected",
    [
        # The timeout is wall clock, bjobs calls of 10 s included (it counted
        # half-second sleeps: twenty minutes for a "60 s" wait).
        ("PEND", [NOT_STARTED], 60, dict(host=None, bjobs_seconds=10, within=80, warnings=2)),
        # Queued for long is said at 30, 60 and 120 s, once each, not on every
        # poll; then the timeout.
        ("PEND", [NOT_STARTED], 300, dict(host=None, warnings=4)),
        # So is an error line in the output, and only that line.
        ("RUN", [(0, "loading\nRuntimeError: CUDA error: out of memory\n", "")], 30, dict(host=None, errors=1)),
        # A transient bpeek failure is not the job ending.
        ("RUN", [(255, "", "LSF is processing your request"), (0, f"x\n{MARKER}\n", "")], 30, dict(host=URL, sleeps=1)),
        # A job that exited ends the wait at once, its log's crash said once.
        ("EXIT", [(255, "", "No matching job found")], 300, dict(host=None, within=5, commands=["bjobs"], errors=1)),
        # A requeued job's newest address.
        ("RUN", [(0, f"{MARKER.replace('node7:4321', 'node3:1111')}\nrequeued\n{MARKER}\n", "")], 60, dict(host=URL)),
    ],
    ids=["wall-clock", "long-pending", "error-line", "transient-bpeek", "exited", "requeued"],
)
def test_waiting_for_a_servers_address(fake_lsf, tmp_path, caplog, stat, bpeek, timeout, expected):
    fake_lsf.answers["bjobs"] = lambda argv, kw: _bjobs(argv[-1], stat)
    fake_lsf.answers["bpeek"] = bpeek
    fake_lsf.cost["bjobs"] = expected.get("bjobs_seconds", 0)
    log = tmp_path / "m_1.log"
    log.write_text("Traceback (most recent call last):\nValueError: bad checkpoint\n")

    with caplog.at_level(logging.WARNING, logger=jobs_lsf.logger.name):
        job = LSFJob("1", log_file=log if stat == "EXIT" else None)
        assert job.wait_for_host(timeout=timeout) == expected["host"]

    records = [r for r in caplog.records if r.name == jobs_lsf.logger.name]
    if "within" in expected:
        assert fake_lsf.now - fake_lsf.start < expected["within"]
    if "warnings" in expected:
        assert len([r for r in records if r.levelno == logging.WARNING]) == expected["warnings"]
    errors = [r.getMessage() for r in records if r.levelno == logging.ERROR]
    if "errors" in expected:
        assert len(errors) == expected["errors"] and not any("loading" in e for e in errors), "that line only"
    if "commands" in expected:
        assert fake_lsf.commands() == expected["commands"]
    if "sleeps" in expected:
        assert fake_lsf.sleeps == expected["sleeps"]


@pytest.mark.parametrize("template, expected", [
    pytest.param(None, URL, id="none"),
    pytest.param("https://proxy.example.org/{host}/{port}", "https://proxy.example.org/node7/4321", id="host-and-port"),
    pytest.param("https://proxy.example.org/{nope}", URL, id="a-bad-one-is-ignored"),
])
def test_a_server_url_template_gives_the_address_viewers_use(fake_lsf, monkeypatch, template, expected):
    fake_lsf.answers.update(bjobs=lambda argv, kw: _bjobs(argv[-1]), bpeek=[(0, MARKER + "\n", "")])
    monkeypatch.delenv("CELLMAP_FLOW_SERVER_URL_TEMPLATE", raising=False)
    if template:
        monkeypatch.setenv("CELLMAP_FLOW_SERVER_URL_TEMPLATE", template)
    job = LSFJob("1")
    assert job.wait_for_host(timeout=60) == expected == job.host


# --- the ready file ----------------------------------------------------------------


@pytest.mark.parametrize("content, job_id, expected", [
    pytest.param(None, None, None, id="not-written-yet"),
    pytest.param('{"url": "http://10.1', None, None, id="cut-short"),
    pytest.param('{"url": ""}', None, None, id="no-url"),
    pytest.param(json.dumps({"url": URL, "job_id": "6"}), "7", None, id="another-jobs"),
    pytest.param(json.dumps({"url": URL, "job_id": "7"}), "7", URL, id="this-jobs"),
    pytest.param(json.dumps({"url": URL}), "7", URL, id="written-outside-lsf"),  # nothing to compare
])
def test_reading_a_ready_file(tmp_path, content, job_id, expected):
    if content is not None:
        (tmp_path / "m.ready").write_text(content)
    assert read_ready_file(tmp_path / "m.ready", job_id=job_id) == expected


def test_each_submission_gets_a_new_ready_file_named_after_its_job(tmp_path):
    """So a job that started late on a queue given up on cannot hand its
    address to the next submission."""
    target = ready_path(tmp_path / "logs", "../mito model")
    assert target != ready_path(tmp_path / "logs", "../mito model")
    assert target.parent == tmp_path / "logs" and target.name.startswith("mito_model_") and target.suffix == ".ready"


def test_a_server_writes_its_ready_file_as_it_prints_its_marker(tmp_path, monkeypatch, capsys):
    from cellmap_flow import server

    target = tmp_path / "m.ready"
    monkeypatch.setenv(READY_ENV, str(target))
    monkeypatch.setenv("LSB_JOBID", "4242")
    monkeypatch.setattr(server, "get_public_ip", lambda: "10.1.2.3")
    served = []
    fake_server = SimpleNamespace(app=SimpleNamespace(run=lambda **kw: served.append((kw["port"], read_ready_file(target)))))

    server.CellMapFlowServer.run(fake_server, port=8123)

    assert served == [(8123, "http://10.1.2.3:8123")], "written before the port is bound"
    assert f"{IP_PATTERN[0]}http://10.1.2.3:8123{IP_PATTERN[1]}" in capsys.readouterr().out, "the marker still printed"
    assert json.loads(target.read_text()).keys() == {"url", "host", "pid", "job_id"}
    assert list(target.parent.iterdir()) == [target], "no temporary file is left behind"


@pytest.mark.parametrize("where", [pytest.param(None, id="not-asked"), pytest.param("not_a_dir/m.ready", id="unwritable")])
def test_a_server_not_asked_or_unable_to_write_a_ready_file_goes_on(tmp_path, monkeypatch, where):
    (tmp_path / "not_a_dir").write_text("")
    if where:
        monkeypatch.setenv(READY_ENV, str(tmp_path / where))
    else:
        monkeypatch.delenv(READY_ENV, raising=False)
    assert write_ready_file(URL) is None


@pytest.mark.parametrize("written", [pytest.param(True, id="written"), pytest.param(False, id="a-server-from-before-the-file")])
def test_the_launcher_reads_the_ready_file_before_asking_lsf(fake_lsf, tmp_path, written):
    fake_lsf.answers.update(bjobs=lambda argv, kw: _bjobs(argv[-1]), bpeek=[(0, f"loading\n{MARKER}\n", "")])
    path = tmp_path / "m.ready"
    if written:
        path.write_text(json.dumps({"url": URL, "job_id": "7"}))
    assert LSFJob("7", ready_file=path).wait_for_host(timeout=30) == URL
    # A server from before the file: bpeek, as before.
    assert fake_lsf.commands() == ([] if written else ["bjobs", "bpeek"])
    assert not path.exists(), "read once, then removed"
