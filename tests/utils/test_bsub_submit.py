"""submit_bsub_job against canned bsub/bjobs output.

The job id was taken as the second word of bsub's stdout, which is wrong as
soon as anything (an esub notice) is printed before "Job <id> is submitted".
A bsub that timed out was treated as a refusal, so start_hosts submitted the
same job to the next queue -- while LSF, which does not cancel a submission
when the client is killed, went on to create the first one as well.
"""

import subprocess

import pytest

from cellmap_flow.globals import g
from cellmap_flow.utils import bsub_utils
from cellmap_flow.utils.bsub_utils import submit_bsub_job

ESUB_NOTICE = (
    "Your job requests 12 slots per GPU; gpu_h100 averages 12.\n"
    "Job <4242> is submitted to queue <gpu_h100>.\n"
)


def _bjobs_line(job_id, name="m"):
    return f"{job_id}   me   RUN   gpu_h100   login1   4*h10u05   {name}   Sep 28 10:00\n"


class FakeLSF:
    def __init__(self, monkeypatch, tmp_path):
        self.bsub = []  # outcomes, consumed in order: CompletedProcess args or an exception
        self.bjobs = []  # stdout of successive `bjobs -J` calls
        self.calls = []
        monkeypatch.setattr(bsub_utils, "SERVER_LOG_DIR", tmp_path)
        monkeypatch.setattr(bsub_utils.subprocess, "run", self.run)

    def run(self, argv, **kwargs):
        self.calls.append(list(argv))
        if argv[0] == "bsub":
            outcome = self.bsub.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return subprocess.CompletedProcess(argv, 0, outcome, "")
        if argv[0] == "bjobs":
            out = self.bjobs.pop(0) if self.bjobs else ""
            if out:
                return subprocess.CompletedProcess(argv, 0, out, "")
            return subprocess.CompletedProcess(argv, 255, "", f"Job <{argv[-1]}> is not found\n")
        raise AssertionError(f"unexpected command {argv}")

    def submissions(self):
        return [c for c in self.calls if c[0] == "bsub"]


@pytest.fixture
def lsf(monkeypatch, tmp_path):
    return FakeLSF(monkeypatch, tmp_path)


def test_the_job_id_is_found_after_an_esub_notice(lsf):
    lsf.bsub = [ESUB_NOTICE]
    job = submit_bsub_job("serve", queue="gpu_h100", job_name="m")
    assert job.job_id == "4242"
    assert job.log_file.name == "m_4242.log"


def test_output_without_a_job_id_is_an_error(lsf):
    lsf.bsub = ["Request aborted by esub. Job not submitted.\n"]
    with pytest.raises(RuntimeError, match="job id"):
        submit_bsub_job("serve", queue="gpu_h100", job_name="m")


def test_a_timed_out_bsub_whose_job_landed_is_picked_up(lsf):
    lsf.bjobs = [_bjobs_line(111), _bjobs_line(111) + _bjobs_line(4243)]
    lsf.bsub = [subprocess.TimeoutExpired("bsub", 30)]

    job = submit_bsub_job("serve", queue="gpu_h100", job_name="m")

    assert job.job_id == "4243", "the new job, not the older one with the same name"
    assert len(lsf.submissions()) == 1


def test_a_timed_out_bsub_is_not_resubmitted_to_another_queue(lsf, monkeypatch):
    lsf.bsub = [subprocess.TimeoutExpired("bsub", 30), "Job <9> is submitted to queue <gpu_a100>.\n"]
    monkeypatch.setattr(bsub_utils, "is_bsub_available", lambda: True)
    monkeypatch.setattr(
        bsub_utils, "gpu_queue_candidates", lambda preferred, cycle=True: ["gpu_h100", "gpu_a100"]
    )
    g.jobs = []

    with pytest.raises(bsub_utils.BsubTimeoutError, match="bjobs -a -J m"):
        bsub_utils.start_hosts("serve", queue="gpu_h100", job_name="m")

    assert len(lsf.submissions()) == 1, "a second bsub would leave a duplicate job billing"
    assert g.jobs == []
