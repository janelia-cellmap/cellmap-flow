"""What start_hosts does with the jobs it submits.

LSF is never called: submit_bsub_job, the queue list and bsub's presence are
replaced with fakes, and each fake job says what bjobs would have reported.
"""

import pytest

from cellmap_flow.globals import g
from cellmap_flow.utils import bsub_utils
from cellmap_flow.utils.bsub_utils import JobStatus, start_hosts


class FakeLSFJob:
    """An LSFJob whose host, and whose status once the wait ends, are canned."""

    def __init__(self, job_id, host=None, status=JobStatus.PENDING, late_host=None):
        self.job_id = job_id
        self.model_name = None
        self.host = None
        self.queue = None
        self.log_file = None
        self._host = host
        self._late_host = late_host
        self._status = status
        self.waits = []
        self.killed = False

    def wait_for_host(self, timeout=300):
        self.waits.append(timeout)
        if self._host:
            self.host = self._host
        elif self._late_host and len(self.waits) > 1:
            self.host = self._late_host
        return self.host

    def observed_status(self):
        return self._status

    def kill(self):
        self.killed = True


@pytest.fixture
def lsf(monkeypatch):
    """Pretend bsub exists; ``lsf.jobs[queue]`` is what submitting there yields."""

    class State:
        jobs = {}
        submitted = []
        local_runs = []

    def fake_submit(command, queue, charge_group, job_name, walltime=None, **_):
        State.submitted.append(queue)
        outcome = State.jobs[queue]
        if isinstance(outcome, Exception):
            raise outcome
        outcome.model_name = job_name
        return outcome

    def fake_run_locally(command, name, log_file=None):
        State.local_runs.append(command)
        raise AssertionError("must not fall back to running on this host")

    monkeypatch.setattr(bsub_utils, "is_bsub_available", lambda: True)
    monkeypatch.setattr(bsub_utils, "submit_bsub_job", fake_submit)
    monkeypatch.setattr(bsub_utils, "run_locally", fake_run_locally)
    monkeypatch.setattr(
        bsub_utils,
        "gpu_queue_candidates",
        lambda preferred, cycle=True: [preferred]
        + [q for q in ("gpu_a100", "gpu_h200") if q != preferred],
    )
    g.jobs = []
    return State


def test_the_queue_a_job_landed_on_is_returned_not_written_to_globals(lsf):
    g.queue = "gpu_h100"
    g.charge_group = "saved_group"
    lsf.jobs = {
        "gpu_h100": FakeLSFJob("1", status=JobStatus.PENDING),
        "gpu_a100": FakeLSFJob("2", host="http://node:1"),
    }

    job = start_hosts("serve", queue="gpu_h100", charge_group=None, job_name="m")

    assert job.queue == "gpu_a100"
    assert job.host == "http://node:1"
    # The next dashboard submission must still start from the requested
    # queue, and a None charge group must not wipe the saved one.
    assert g.queue == "gpu_h100"
    assert g.charge_group == "saved_group"
