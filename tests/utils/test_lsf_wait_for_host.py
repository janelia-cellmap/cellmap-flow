"""LSFJob.wait_for_host against canned bjobs/bpeek answers and a fake clock.

The clock only moves when the code sleeps or an LSF command "takes" time, so
these run instantly and can say exactly how long a wait really lasted.
"""

import logging
import subprocess
from types import SimpleNamespace

from cellmap_flow.jobs import lsf as jobs_lsf
from cellmap_flow.utils.bsub_utils import LSFJob
from cellmap_flow.utils.web_utils import IP_PATTERN

MARKER = f"{IP_PATTERN[0]}http://node7:4321{IP_PATTERN[1]}"


class FakeLSF:
    """Answers bjobs and bpeek from scripts, advancing the fake clock.

    The clock is patched where LSFJob lives, jobs.lsf; each test checks the
    fake slept, so a clock left unpatched cannot pass by waiting for real.
    """

    def __init__(self, monkeypatch, stat="RUN", bjobs_seconds=0.0):
        self.now = 1000.0
        self.stat = stat
        self.bjobs_seconds = bjobs_seconds
        self.bpeek = []  # (returncode, stdout, stderr), consumed in order
        self.bpeek_default = (0, "starting\n", "")
        self.calls = []
        self.timeline = []  # (seconds since the start, command)
        self.start = self.now
        self.sleeps = 0
        monkeypatch.setattr(
            jobs_lsf,
            "time",
            SimpleNamespace(
                time=lambda: self.now, monotonic=lambda: self.now, sleep=self.sleep
            ),
        )
        monkeypatch.setattr(jobs_lsf.subprocess, "run", self.run)

    def sleep(self, seconds):
        self.sleeps += 1
        self.now += seconds

    def run(self, argv, **kwargs):
        self.calls.append(argv[0])
        self.timeline.append((round(self.now - self.start, 3), argv[0]))
        if argv[0] == "bjobs":
            self.now += self.bjobs_seconds
            stat = self.stat(self.now) if callable(self.stat) else self.stat
            out = f"{argv[-1]} me {stat} gpu_h100 login h01 name Sep 28 10:00\n"
            return subprocess.CompletedProcess(argv, 0, out, "")
        if argv[0] == "bpeek":
            rc, out, err = self.bpeek.pop(0) if self.bpeek else self.bpeek_default
            return subprocess.CompletedProcess(argv, rc, out, err)
        raise AssertionError(f"unexpected command {argv}")


def test_the_timeout_is_wall_clock_time_including_slow_lsf_calls(monkeypatch):
    lsf = FakeLSF(monkeypatch, stat="PEND", bjobs_seconds=10)
    lsf.bpeek_default = (255, "", "Job <1> : Not yet started.")
    start = lsf.now

    assert LSFJob("1").wait_for_host(timeout=60) is None

    # It used to count half-second sleeps only: 120 of them, each behind a
    # 10 s bjobs call, is twenty minutes for a "60 s" wait.
    assert lsf.now - start < 80
    assert lsf.sleeps


def test_the_long_pending_warning_is_logged_once(monkeypatch, caplog):
    lsf = FakeLSF(monkeypatch, stat="PEND")
    lsf.bpeek_default = (255, "", "Job <1> : Not yet started.")

    with caplog.at_level(logging.WARNING, logger=jobs_lsf.logger.name):
        LSFJob("1").wait_for_host(timeout=300)

    unusual = [r for r in caplog.records if "unusually long" in r.getMessage()]
    assert len(unusual) == 1, f"logged {len(unusual)} times"
    assert lsf.sleeps


def test_an_error_line_in_the_output_is_logged_once(monkeypatch, caplog):
    lsf = FakeLSF(monkeypatch, stat="RUN")
    lsf.bpeek_default = (0, "loading\nRuntimeError: CUDA error: out of memory\n", "")

    with caplog.at_level(logging.ERROR, logger=jobs_lsf.logger.name):
        LSFJob("1").wait_for_host(timeout=30)

    cuda = [r for r in caplog.records if "CUDA error" in r.getMessage()]
    assert len(cuda) == 1, f"logged {len(cuda)} times"
    assert "loading" not in cuda[0].getMessage(), "only the error line, not all output"
    assert lsf.sleeps


def test_a_transient_bpeek_failure_is_not_taken_for_the_job_ending(monkeypatch):
    lsf = FakeLSF(monkeypatch, stat="RUN")
    lsf.bpeek = [(255, "", "LSF is processing your request"), (0, f"x\n{MARKER}\n", "")]

    assert LSFJob("1").wait_for_host(timeout=30) == "http://node7:4321"
    assert lsf.sleeps == 1


def test_a_job_that_exits_reports_its_log_once(monkeypatch, caplog, tmp_path):
    lsf = FakeLSF(monkeypatch, stat="EXIT")
    lsf.bpeek_default = (255, "", "Job <1> : No matching job found")
    log = tmp_path / "m_1.log"
    log.write_text("Traceback (most recent call last):\nValueError: bad checkpoint\n")
    job = LSFJob("1", log_file=log)

    with caplog.at_level(logging.ERROR, logger=jobs_lsf.logger.name):
        assert job.wait_for_host(timeout=300) is None

    assert lsf.now - 1000.0 < 5, "a finished job should end the wait at once"
    assert lsf.calls == ["bjobs"]
    crash = [r for r in caplog.records if "bad checkpoint" in r.getMessage()]
    assert len(crash) == 1, f"crash output logged {len(crash)} times"


def test_the_polling_timeline(monkeypatch):
    """When bjobs and bpeek are asked, for a job that queues, loads, then serves.

    Pinned so that a change of polling cadence is a visible diff here rather
    than a surprise in mbatchd's load.
    """
    lsf = FakeLSF(monkeypatch, stat=lambda now: "PEND" if now < 1002 else "RUN")
    not_started = (255, "", "Job <1> : Not yet started.")
    lsf.bpeek = [not_started] * 4 + [(0, "loading\n", "")] * 2 + [(0, f"loading\n{MARKER}\n", "")]

    assert LSFJob("1").wait_for_host(timeout=60) == "http://node7:4321"

    assert lsf.timeline == [
        (0.0, "bjobs"), (0.0, "bpeek"),
        (0.5, "bjobs"), (0.5, "bpeek"),
        (1.0, "bjobs"), (1.0, "bpeek"),
        (1.5, "bjobs"), (1.5, "bpeek"),
        (2.0, "bjobs"), (2.0, "bpeek"),
        (2.5, "bjobs"), (2.5, "bpeek"),
        (3.0, "bjobs"), (3.0, "bpeek"),
    ]
    assert lsf.sleeps == 6


def test_a_requeued_job_reports_its_newest_address(monkeypatch):
    lsf = FakeLSF(monkeypatch)
    earlier = MARKER.replace("node7:4321", "node3:1111")
    lsf.bpeek_default = (0, f"{earlier}\nrequeued\n{MARKER}\n", "")
    assert LSFJob("1").wait_for_host(timeout=60) == "http://node7:4321"
