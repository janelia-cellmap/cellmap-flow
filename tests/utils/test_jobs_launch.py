"""jobs.launch's policy: where start_hosts runs a server, and the job cleanup
the entry points install.

start_hosts' submissions are replaced where it calls them (submit_bsub_job,
run_locally, the queue list), and each fake job says what bjobs would have
reported; LSF itself is never called.
"""

import json
import logging
import signal
import subprocess
import threading
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from click.testing import CliRunner

from cellmap_flow.jobs import launch
from cellmap_flow.jobs.launch import start_hosts, started_jobs
from cellmap_flow.jobs.lsf import BsubTimeoutError
from cellmap_flow.jobs.ready import READY_ENV
from cellmap_flow.jobs.settings import LauncherSettings, launcher_settings
from cellmap_flow.jobs.spec import JobStartError, JobStatus
from tests.utils.serving_helpers import write_raw

QUEUES = ("gpu_h100", "gpu_a100", "gpu_h200")
P, R, F = JobStatus.PENDING, JobStatus.RUNNING, JobStatus.FAILED


class FakeJob:
    """A job whose host, and whose status once the wait ends, are canned; a
    late host comes on the second wait."""

    def __init__(self, job_id, host=None, status=P, late_host=None):
        self.job_id, self._host, self._status, self._late_host = job_id, host, status, late_host
        self.model_name = self.host = self.queue = self.log_file = None
        self.waits, self.killed = 0, False

    def wait_for_host(self, timeout=300):
        self.waits += 1
        self.host = self._host or (self._late_host if self.waits > 1 else None)
        return self.host

    def observed_status(self):
        return self._status

    def kill(self):
        self.killed = True


REFUSED = subprocess.CalledProcessError(255, "bsub", stderr="bad project")

class Case(NamedTuple):
    outcomes: dict  # what submitting to each queue gives: a FakeJob's arguments, or an exception
    result: object  # the queue the job ran on, "local", or the (error, message) raised
    submitted: list  # the queues submitted to, in order
    killed: list  # the jobs killed
    bsub: bool = True  # whether bsub is installed
    kwargs: dict = {}  # start_hosts' keywords


CASES = {
    # The queue it landed on is the job's: the saved queue stays what the next submission asks for.
    "falls-back-to-another-queue": Case({"gpu_h100": dict(status=P), "gpu_a100": dict(host="http://node:1")},
                                        "gpu_a100", ["gpu_h100", "gpu_a100"], ["gpu_h100"]),
    # A GPU server must not start on a login or submit node instead.
    "every-queue-refused": Case({q: REFUSED for q in QUEUES}, (JobStartError, "gpu_h100.*gpu_a100.*gpu_h200"),
                                list(QUEUES), []),
    # Nothing will point a layer at it, so it must not sit there billing.
    "never-started": Case({q: dict(status=P) for q in QUEUES}, (JobStartError, "did not start"), list(QUEUES), list(QUEUES)),
    # A crash reproduces on every queue.
    "crashed": Case({"gpu_h100": dict(status=F), "gpu_a100": dict(host="http://x:1")}, (JobStartError, "gpu_h100"),
                    ["gpu_h100"], []),
    "running-but-still-loading": Case({"gpu_h100": dict(status=R, late_host="http://node:2")}, "gpu_h100", ["gpu_h100"], []),
    "running-without-ever-a-host": Case({"gpu_h100": dict(status=R)}, (JobStartError, "killed"), ["gpu_h100"], ["gpu_h100"]),
    # LSF may still make the timed-out job: a second bsub would leave two of it.
    "bsub-timed-out": Case({"gpu_h100": BsubTimeoutError("bjobs -a -J m")}, (BsubTimeoutError, "bjobs -a -J m"),
                           ["gpu_h100"], []),
    "no-bsub": Case({}, "local", [], [], bsub=False),
    "asked-to-run-locally": Case({}, "local", [], [], kwargs={"local": True}),
    "local-server-without-a-host": Case({}, (JobStartError, "did not report"), [], ["local"], bsub=False),
}


@pytest.mark.parametrize("case", CASES)
def test_where_start_hosts_runs_a_server(case, monkeypatch, caplog):
    outcomes, result, submitted, killed, bsub, kwargs = CASES[case]
    jobs = {q: o if isinstance(o, Exception) else FakeJob(q, **o) for q, o in outcomes.items()}
    local = FakeJob("local", host=None if case == "local-server-without-a-host" else "http://localhost:9")
    calls = []

    def submit(command, queue, charge_group, job_name, walltime=None, env=None):
        calls.append(queue)
        if isinstance(jobs[queue], Exception):
            raise jobs[queue]
        return jobs[queue]

    monkeypatch.setattr(launch, "is_bsub_available", lambda: bsub)
    monkeypatch.setattr(launch, "submit_bsub_job", submit)
    monkeypatch.setattr(launch, "run_locally", lambda command, name, log_file=None: calls.append("local") or local)
    monkeypatch.setattr(launch, "gpu_queue_candidates",
                        lambda preferred, cycle=True: [preferred] + [q for q in QUEUES if q != preferred])
    settings = launcher_settings()
    settings.queue, settings.charge_group = "gpu_h100", "saved_group"

    with caplog.at_level(logging.ERROR, logger=launch.logger.name):
        if isinstance(result, tuple):
            with pytest.raises(result[0], match=result[1]) as raised:
                start_hosts("serve", queue="gpu_h100", charge_group=None, job_name="m", **kwargs)
            assert started_jobs() == []
            if result[0] is JobStartError:  # a dashboard thread's exception only reaches stderr
                assert str(raised.value) in caplog.text
        else:
            job = start_hosts("serve", queue="gpu_h100", charge_group=None, job_name="m", **kwargs)
            expected = local if result == "local" else jobs[result]
            assert job is expected and job.host and started_jobs() == [job]
            assert job.queue == (None if result == "local" else result)
    assert calls == submitted + (["local"] if result == "local" or "local" in killed else [])
    assert [j.job_id for j in [*jobs.values(), local] if isinstance(j, FakeJob) and j.killed] == killed
    assert (settings.queue, settings.charge_group) == ("gpu_h100", "saved_group")
    if case == "running-but-still-loading":
        assert jobs["gpu_h100"].waits == 2, "it was given time to load its model"


def test_start_hosts_hands_each_job_a_ready_file_and_reads_it(fake_lsf):
    """What the server does once it runs, with the environment LSF copies into the job."""

    def bsub(argv, kwargs):
        target = Path(kwargs["env"][READY_ENV])
        assert target.parent == launch.SERVER_LOG_DIR
        target.write_text(json.dumps({"url": "http://10.1.2.3:8123", "job_id": "4242"}))
        return "Job <4242> is submitted to queue <gpu_h100>.\n"

    fake_lsf.answers.update(bjobs=[(255, "", "Job <m> is not found\n")], bsub=bsub)
    job = start_hosts("serve", queue="gpu_h100", job_name="m", cycle_queues=False)
    assert job.host == "http://10.1.2.3:8123" and started_jobs() == [job]
    assert fake_lsf.commands() == ["which", "bjobs", "bsub"], "no bpeek, and no bjobs for the job itself"


def test_the_cleanup_handler_kills_the_jobs_and_exits_with_the_signals_status():
    outcome = []
    thread = threading.Thread(target=lambda: outcome.append(launch.install_cleanup_handlers()))
    thread.start()
    thread.join()
    assert outcome == [False], "off the main thread it leaves the handlers alone"

    job = FakeJob("1")
    started_jobs().append(job)
    with pytest.raises(SystemExit) as exited:
        launch.cleanup_handler(signal.SIGTERM, None)
    assert job.killed and exited.value.code == 128 + signal.SIGTERM, "not 0, as if the run had succeeded"


def test_the_cleanup_handler_kills_a_job_still_waiting_to_start(monkeypatch):
    """A job joined started_jobs() only once it reported an address, so a
    Ctrl+C while it was queued or loading its model left it running."""
    waiting, release = threading.Event(), threading.Event()

    class Queued(FakeJob):
        def wait_for_host(self, timeout=300):
            waiting.set()
            release.wait(5)
            return "http://node:1" if not self.killed else None

    job = Queued("1")
    monkeypatch.setattr(launch, "is_bsub_available", lambda: True)
    monkeypatch.setattr(launch, "submit_bsub_job", lambda *a, **kw: job)
    monkeypatch.setattr(launch, "gpu_queue_candidates", lambda preferred, cycle=True: [preferred])
    starting = threading.Thread(target=lambda: start_hosts("serve", queue="gpu_h100", job_name="m"))
    starting.start()
    assert waiting.wait(5)
    try:
        with pytest.raises(SystemExit):
            launch.cleanup_handler(signal.SIGINT, None)
    finally:
        release.set()
        starting.join(5)
    assert job.killed


def test_the_cleanup_handlers_cover_the_terminal_going_away(monkeypatch):
    """SIGHUP: the terminal closed, its connection dropped, or the
    interactive LSF session the dashboard ran in ended."""
    installed = {}
    monkeypatch.setattr(launch.signal, "signal", lambda signum, handler: installed.setdefault(signum, handler))
    assert launch.install_cleanup_handlers()
    assert {signal.SIGINT, signal.SIGTERM, signal.SIGHUP} <= set(installed)


def _cellmap_flow_infer(monkeypatch, tmp_path, order):
    from cellmap_flow.cli import infer, main
    from cellmap_flow.dashboard.services import startup

    monkeypatch.setattr(infer, "install_cleanup_handlers", lambda: order.append("install"))
    monkeypatch.setattr(infer, "start_hosts", lambda *a, **k: order.append("run"))
    monkeypatch.setattr(startup, "generate_neuroglancer_url", lambda path: None)
    monkeypatch.setattr(LauncherSettings, "save", lambda self: None)
    result = CliRunner().invoke(main.cli, ["infer", "script", "-s", "/s.py", "-d", str(tmp_path)])
    assert result.exit_code == 0, result.output + repr(result.exception)


def _cellmap_flow_yaml(monkeypatch, tmp_path, order):
    from cellmap_flow.cli import yaml_cli

    monkeypatch.setattr(yaml_cli, "install_cleanup_handlers", lambda: order.append("install"))
    monkeypatch.setattr(yaml_cli, "run_multiple", lambda *a, **k: order.append("run"))
    (tmp_path / "c.yaml").write_text("data_path: /d.zarr\ncharge_group: grp\nqueue: gpu_h100\nmodels: {}\n")
    result = CliRunner().invoke(yaml_cli.main, [str(tmp_path / "c.yaml")])
    assert result.exit_code == 0, result.output


def _cellmap_flow_view(monkeypatch, tmp_path, order):
    import neuroglancer
    from neuroglancer.viewer_base import ViewerBase

    from cellmap_flow.cli import viewer_cli
    from cellmap_flow.dashboard import app

    monkeypatch.setattr(neuroglancer, "Viewer", ViewerBase)
    monkeypatch.setattr(neuroglancer, "set_server_bind_address", lambda *a: None)
    monkeypatch.setattr(app, "create_and_run_app", lambda **k: order.append("run"))
    monkeypatch.setattr(launch, "install_cleanup_handlers", lambda: order.append("install"))
    result = CliRunner().invoke(viewer_cli.main, ["-d", write_raw(tmp_path, np.zeros((4, 4, 4), np.uint8))])
    assert result.exit_code == 0, result.output + repr(result.exception)


@pytest.mark.parametrize("entry_point", [_cellmap_flow_infer, _cellmap_flow_yaml, _cellmap_flow_view],
                         ids=lambda f: f.__name__[1:])
def test_the_entry_points_install_the_cleanup_before_starting_jobs(entry_point, monkeypatch, tmp_path):
    """Importing the launcher used to set them, for every importer, and failed off the main thread."""
    order = []
    entry_point(monkeypatch, tmp_path, order)
    assert order == ["install", "run"]
