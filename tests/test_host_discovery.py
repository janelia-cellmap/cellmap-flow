"""Regression tests for inference-server host discovery.

Background: bsub -o APPENDS to the log file. With one shared log per model
name, `bpeek` returned every previous run's output, and
extract_host_from_output() took the FIRST CELLMAP_FLOW_SERVER_IP marker --
i.e. the oldest, long-dead address -- so the viewer pointed at stale servers
on every restart. These tests pin the fixed behaviour.
"""

import os
import time

import pytest

from cellmap_flow.utils.bsub_utils import (
    LocalJob,
    extract_host_from_output,
    run_locally,
)
from cellmap_flow.utils.web_utils import IP_PATTERN


def _marker(host: str) -> str:
    return f"{IP_PATTERN[0]}{host}{IP_PATTERN[1]}"


def test_extract_host_returns_newest_marker():
    output = "\n".join(
        [
            "some old run",
            _marker("http://10.36.112.26:23881"),
            "exit report of old run",
            _marker("http://10.36.113.26:16785"),
            "current run starting",
            _marker("http://10.36.112.26:44703"),
        ]
    )
    assert extract_host_from_output(output) == "http://10.36.112.26:44703"


def test_extract_host_single_marker():
    assert extract_host_from_output(_marker("http://1.2.3.4:5")) == "http://1.2.3.4:5"


def test_extract_host_waits_for_closing_marker():
    # Opening marker flushed but closing one not yet -> keep polling, don't
    # return a truncated host.
    partial = _marker("http://old:1") + "\n" + IP_PATTERN[0] + "http://new:2"
    assert extract_host_from_output(partial) is None


def test_extract_host_empty():
    assert extract_host_from_output("") is None
    assert extract_host_from_output(None) is None
    assert extract_host_from_output("no markers here") is None


def _wait_and_kill(job: LocalJob, timeout: int = 20):
    try:
        return job.wait_for_host(timeout=timeout)
    finally:
        job.kill()


def test_run_locally_expands_percent_j_and_discovers_host(tmp_path):
    template = tmp_path / "model.%J.log"
    cmd = (
        "bash -c \"echo '" + _marker("http://1.2.3.4:5") + "'; sleep 30\""
    )
    job = run_locally(cmd, "model", log_file=template)
    assert "%J" not in job.log_file
    assert os.path.dirname(job.log_file) == str(tmp_path)
    assert _wait_and_kill(job) == "http://1.2.3.4:5"
    assert not template.exists()


def test_run_locally_ignores_content_from_previous_run(tmp_path):
    # Same (non-templated) path as an earlier run that already logged an
    # address: only what THIS launch writes may be parsed.
    log = tmp_path / "shared.log"
    log.write_text(_marker("http://old-dead:1") + "\nexit report\n")
    cmd = (
        "bash -c \"sleep 1; echo '" + _marker("http://fresh:2") + "'; sleep 30\""
    )
    job = run_locally(cmd, "model", log_file=log)
    assert _wait_and_kill(job) == "http://fresh:2"


def test_local_job_file_mode_reports_dead_process(tmp_path):
    log = tmp_path / "dies.log"
    job = run_locally("bash -c 'exit 3'", "model", log_file=log)
    time.sleep(0.5)
    assert job.wait_for_host(timeout=5) is None
