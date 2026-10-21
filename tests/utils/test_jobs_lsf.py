"""jobs.lsf and jobs.ready against canned LSF answers (conftest's ``fake_lsf``).

How answers are read: the job id of a submission, one that timed out among
them, and the wait for a server's address; and the ready file each
submission is given.
"""

import logging
import subprocess

import pytest

from cellmap_flow.jobs import lsf as jobs_lsf
from cellmap_flow.jobs.lsf import BsubTimeoutError, LSFJob
from cellmap_flow.jobs.ready import ready_path
from cellmap_flow.jobs.spec import IP_PATTERN, JobSpec

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


# --- waiting for a server's address ---------------------------------------------


@pytest.mark.parametrize(
    "stat, bpeek, timeout, expected",
    [
        # The timeout is wall clock, bjobs calls of 10 s included (it counted
        # half-second sleeps: twenty minutes for a "60 s" wait).
        ("PEND", [NOT_STARTED], 60, dict(host=None, bjobs_seconds=10, within=80, warnings=2)),
        # An error line in the output is said, and only that line.
        ("RUN", [(0, "loading\nRuntimeError: CUDA error: out of memory\n", "")], 30, dict(host=None, errors=1)),
        # A transient bpeek failure is not the job ending.
        ("RUN", [(255, "", "LSF is processing your request"), (0, f"x\n{MARKER}\n", "")], 30, dict(host=URL, sleeps=1)),
        # A job that exited ends the wait at once, its log's crash said once.
        ("EXIT", [(255, "", "No matching job found")], 300, dict(host=None, within=5, commands=["bjobs"], errors=1)),
        # A requeued job's newest address.
        ("RUN", [(0, f"{MARKER.replace('node7:4321', 'node3:1111')}\nrequeued\n{MARKER}\n", "")], 60, dict(host=URL)),
    ],
    ids=["wall-clock", "error-line", "transient-bpeek", "exited", "requeued"],
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
    pytest.param("https://proxy.example.org/{host}/{port}", "https://proxy.example.org/node7/4321", id="host-and-port"),
    pytest.param("https://proxy.example.org/{nope}", URL, id="a-bad-one-is-ignored"),
])
def test_a_server_url_template_gives_the_address_viewers_use(fake_lsf, monkeypatch, template, expected):
    fake_lsf.answers.update(bjobs=lambda argv, kw: _bjobs(argv[-1]), bpeek=[(0, MARKER + "\n", "")])
    monkeypatch.setenv("CELLMAP_FLOW_SERVER_URL_TEMPLATE", template)
    job = LSFJob("1")
    assert job.wait_for_host(timeout=60) == expected == job.host


# --- the ready file ----------------------------------------------------------------


def test_each_submission_gets_a_new_ready_file_named_after_its_job(tmp_path):
    """So a job that started late on a queue given up on cannot hand its
    address to the next submission."""
    target = ready_path(tmp_path / "logs", "../mito model")
    assert target != ready_path(tmp_path / "logs", "../mito model")
    assert target.parent == tmp_path / "logs" and target.name.startswith("mito_model_") and target.suffix == ".ready"

