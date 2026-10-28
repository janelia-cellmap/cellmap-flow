"""FinetuneJobManager: the command a job runs, what the monitor reads from its
log, what it tells listeners, and finding jobs again after a dashboard
restart. What the trainer prints is test_finetune_cli's, and what the
dashboard's listener does test_finetune_layers'."""

import json
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cellmap_flow.finetune.job_manager import monitor
from cellmap_flow.finetune.job_manager.manager import FinetuneJobManager
from cellmap_flow.finetune.job_manager.state import JobStatus
from cellmap_flow.finetune.model_loading import decode_model_entry
from cellmap_flow.jobs.lsf import LSFJob
from cellmap_flow.jobs.site import JANELIA
from cellmap_flow.jobs.spec import JobStatus as LSF
from cellmap_flow.serving.protocol import IP_PATTERN

URL = "http://node7:8123"


class _Script:
    cli_name, name, script_path, channels = "script", "m", "/s.py", ["mito"]


GEOMETRY = SimpleNamespace(input_voxel_size=[8] * 3, output_voxel_size=[4] * 3, channels=["nuc"])


@pytest.fixture
def submit(local_jobs, session, monkeypatch):
    """``submit(config, geometry=None, manifest=None, **settings)``: a job submitted
    from a session, locally. ``geometry`` is what the model's server reports
    (None: the real lookup, which finds nothing for these configs). Returns the
    job, its metadata.json and command."""
    from cellmap_flow.models import geometry_cache

    record = SimpleNamespace(runs=local_jobs.runs)
    resolve = geometry_cache.resolve_model_geometry

    def run(config, geometry=None, manifest=None, **settings):
        monkeypatch.setattr(geometry_cache, "resolve_model_geometry", lambda name, c: geometry or resolve(name, c))
        record.base = session(manifest=manifest)
        record.job = FinetuneJobManager().submit_finetuning_job(
            model_config=config, corrections_path=record.base / "corrections", output_base=record.base, **settings
        )
        record.metadata = json.loads((record.job.output_dir / "metadata.json").read_text())
        record.command = record.metadata["command"]
        return record

    return run


class _EnvScript(_Script):
    """Served from an environment of its own (models.envs), so trained there."""

    def __init__(self, env):
        self.env = env

    def to_dict(self):
        return {"type": "script", "script_path": self.script_path, "name": self.name, "env": self.env}


@pytest.fixture
def pixi_manifest(tmp_path, monkeypatch):
    """``pixi_manifest()``: a pixi.toml with a cellpose4 environment whose cellmap-flow brings peft."""

    def write():
        path = tmp_path / "checkout" / "pixi.toml"
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            '[pypi-dependencies]\ncellmap-flow = { path = ".", extras = ["finetune"] }\n[environments]\n'
            "cellpose4 = { features = [], no-default-feature = false }\n"
        )
        monkeypatch.setenv("CELLMAP_FLOW_PIXI_MANIFEST", str(path))
        monkeypatch.setenv("PIXI_EXE", "/opt/pixi")
        return path

    return write


@pytest.mark.parametrize("kind", ["pixi", "directory"])
def test_a_model_in_its_own_environment_is_trained_there(submit, pixi_manifest, tmp_path, kind):
    manifest = pixi_manifest()
    # kind: (env, the python it runs, the lib directory first on the loader path)
    env, program, lib = {
        "pixi": ("cellpose4", f"/opt/pixi run --frozen --manifest-path {manifest} -e cellpose4 python",
                 f"{manifest.parent}/.pixi/envs/cellpose4/lib"),
        "directory": (f"{tmp_path}/venv", f"{tmp_path}/venv/bin/python", f"{tmp_path}/venv/lib"),
    }[kind]
    job = submit(_EnvScript(env), geometry=GEOMETRY)
    assert f"LD_LIBRARY_PATH={lib}" in job.command
    assert f"{program} -P -m cellmap_flow.finetune.finetune_cli --model-type script" in job.command
    # As its entry, the only flag that carries the env on to the serving YAMLs the trainer writes.
    entry = job.command.split("--model-entry ", 1)[1].split()[0]
    assert decode_model_entry(entry)["env"] == env


def test_a_trainer_that_fails_fails_its_job(submit, monkeypatch, tmp_path):
    """The trainer's output is piped through tee, and a pipeline's status was
    tee's: a trainer that exited 1 was DONE to LSF, and its job COMPLETED. Here
    the command runs, as a local job would, with a trainer that fails."""
    trainer = tmp_path / "trainer"
    trainer.write_text("#!/bin/sh\necho trained\nexit 1\n")
    trainer.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(trainer))  # the interpreter the command runs
    job = submit(_Script())
    assert subprocess.run(job.runs[0]["command"]).returncode == 1
    assert job.job.log_file.read_text() == "trained\n"


class _Queued:
    """An LSF job on ``queue``: pending on gpu_h200, which never starts it,
    and failing at once elsewhere, which ends its monitor."""

    def __init__(self, queue):
        self.queue, self.job_id, self.killed = queue, f"lsf-{queue}", False

    def get_status(self):
        return LSF.PENDING if self.queue == "gpu_h200" else LSF.FAILED

    def kill(self):
        self.killed = True


def test_a_job_that_has_not_started_moves_to_the_next_queue_ticked(submit, monkeypatch):
    """A training job went to the one queue picked and waited there, however
    long. It goes to the freest of the queues ticked now, and on to the next
    of them, and no other, while LSF does not start it."""
    from dataclasses import replace

    from cellmap_flow.finetune.job_manager import manager
    from cellmap_flow.finetune.job_manager import submit as submitting

    launched = []
    # gpu_h200 the freest of the two ticked, gpu_a100 not ticked.
    monkeypatch.setattr(manager, "gpu_queue_candidates", lambda queues, cycle=True: ["gpu_h200", "gpu_h100"])
    monkeypatch.setattr(submitting, "launch",
                        lambda name, command, queue, **kw: launched.append(_Queued(queue)) or launched[-1])
    monkeypatch.setattr(submitting, "current_site", lambda: replace(JANELIA, pending_fallback_seconds=0))
    monkeypatch.setattr(monitor.time, "sleep", lambda seconds: None)
    run = submit(_Script(), queue=["gpu_h100", "gpu_h200"])
    FinetuneJobManager().monitor_job(run.job)
    assert [job.queue for job in launched] == ["gpu_h200", "gpu_h100"]
    assert launched[0].killed and run.job.lsf_job is launched[1]
    metadata = json.loads((run.job.output_dir / "metadata.json").read_text())
    assert (metadata["lsf_job_id"], metadata["queue"]) == ("lsf-gpu_h100", ["gpu_h100", "gpu_h200"])
    assert "--queue gpu_h100 gpu_h200" in run.command, "the served finetune may run on either"


@pytest.mark.parametrize("override", [pytest.param(False, id="its own checkpoint"),
                                      pytest.param(True, id="a checkpoint override")])
def test_the_trainer_builds_a_fly_model_as_the_dashboard_has_it(submit, tmp_path, override):
    """A Fly model went as its checkpoint and voxel sizes, so the trainer gave it
    the 178/56 default sizes: the finetune was served, and written into its
    YAML, at those. The trainer's model is the job's own now, and the run's
    record says what it was given."""
    from cellmap_flow.finetune.cli import model_config_from_args, parse_args
    from cellmap_flow.models.models_config import FlyModelConfig

    for name in ("own.ts", "override.ts"):
        (tmp_path / name).write_bytes(b"")
    fly = FlyModelConfig(checkpoint_path=str(tmp_path / "own.ts"), channels=["mito"], input_voxel_size=(8, 8, 8),
                         output_voxel_size=(8, 8, 8), name="fly", input_size=(216,) * 3, output_size=(128,) * 3)
    job = submit(fly, checkpoint_path_override=tmp_path / "override.ts" if override else None)
    tokens = shlex.split(job.command)
    argv = tokens[tokens.index("cellmap_flow.finetune.finetune_cli") + 1:tokens.index("2>&1")]
    trained = model_config_from_args(parse_args(argv)).to_dict()
    assert trained == {**fly.to_dict(), "checkpoint_path": str(tmp_path / ("override.ts" if override else "own.ts"))}
    assert job.metadata["model_entry"] == trained


def _lsf(*replies, **fields):
    """A scheduler job that reports ``replies`` in turn."""
    replies = iter(replies)
    return SimpleNamespace(get_status=lambda: next(replies), **fields)


def _monitor(manager, job, chunks, monkeypatch):
    """monitor_job over a log that grows by one chunk per poll (text, or bytes as
    they are written); the job's status, epoch and loss at each poll."""
    chunks = [chunk if isinstance(chunk, bytes) else chunk.encode() for chunk in chunks]
    job.log_file.write_bytes(chunks[0])
    rest, seen = iter(chunks[1:]), []

    def sleep(seconds):
        seen.append((job.status.value, job.current_epoch, job.latest_loss))
        with open(job.log_file, "ab") as f:
            f.write(next(rest, b""))

    monkeypatch.setattr(monitor.time, "sleep", sleep)
    manager.monitor_job(job)
    return seen


FULL = ["full_finetune/model_state_dict.pt"]


@pytest.fixture
def monitored(make_job, monkeypatch):
    """``monitored(chunks, final="FAILED")``: a pending job monitored while LSF
    says RUNNING for each chunk, then ``final``, with a full finetune's export
    on disk. Returns the job, its metadata.json and its (status, epoch, loss)
    at each poll."""

    def run(chunks, final="FAILED"):
        job = make_job("PENDING", lsf_job=_lsf(*[LSF.RUNNING] * len(chunks), LSF[final]))
        (job.output_dir / "metadata.json").write_text(json.dumps({"status": "PENDING"}))
        (job.output_dir / FULL[0]).parent.mkdir()  # complete_job checks that the export is there
        (job.output_dir / FULL[0]).write_bytes(b"")
        seen = _monitor(FinetuneJobManager(), job, chunks, monkeypatch)
        return SimpleNamespace(job=job, seen=seen, metadata=json.loads((job.output_dir / "metadata.json").read_text()))

    return run


def test_the_monitor_reads_one_loss_per_epoch(monitored):
    """From the epoch's summary line, even split across two reads; per-batch
    losses are running means mid-epoch, and are not read. The network's size
    the trainer printed first is kept too (the Publish panel's cost)."""
    run = monitored(["NETWORK_PARAMETERS: 2122\nStarting epoch 1 of 10...\n  Batch 1/3 - Loss: 0.9\nEpoch 1/10 - Lo",
                     "ss: 0.5 - Supervised: 0.5\nStarting epoch 2 of 10...\nEpoch 2/10 - Loss: 0.25 - Sup\n",
                     "  Batch 1/3 - Loss: 0.123\n"])
    assert [(epoch, loss) for _, epoch, loss in run.seen] == [(1, None), (2, 0.25), (2, 0.25)]
    assert run.job.total_epochs == 10
    assert run.job.network_parameters == 2122


def test_the_last_status_marker_in_the_log_decides(monitored):
    """A restart after a divergence in the same chunk leaves the job running,
    and the reverse leaves it waiting; LSF saying RUNNING does not undo waiting.
    A job diverged in a later iteration could never be restarted."""
    run = monitored(["Epoch 3/10 - Loss: nan\nTRAINING_DIVERGED\nWAITING_FOR_RESTART\n",
                     "RESTARTING_TRAINING\nTRAINING_DIVERGED\n", "WAITING_FOR_RESTART\nRESTARTING_TRAINING\n",
                     "TRAINING_ITERATION_COMPLETE: m_1\nWAITING_FOR_RESTART\n"])
    assert [status for status, _, _ in run.seen] == [
        "WAITING_FOR_RESTART", "WAITING_FOR_RESTART", "RUNNING", "WAITING_FOR_RESTART"]


@pytest.mark.parametrize("yaml_2", [
    pytest.param("/s/models/m_2.yaml", id="each iteration with its YAML"),
    # The trainer could not write m_2's, and its log says why. The job kept
    # m_1's, which serves m_1's weights: the pipeline builder offered those as
    # m_2. With none, it offers the run's latest export, m_2's.
    pytest.param(None, id="the last iteration without a YAML"),
])
def test_a_completed_job_takes_the_trainers_name_and_yaml(monitored, tmp_path, yaml_2):
    """The manager made up its own name (the job's creation time, not the
    iteration's), never found that YAML, and wrote a second one. The monitor
    reads each iteration at a poll of its own here."""
    run = monitored(["FINETUNED_MODEL_YAML: /s/models/m_1.yaml\nTRAINING_ITERATION_COMPLETE: m_1\n",
                     "RESTARTING_TRAINING\n" + (f"FINETUNED_MODEL_YAML: {yaml_2}\n" if yaml_2 else "")
                     + "TRAINING_ITERATION_COMPLETE: m_2\n"], final="COMPLETED")
    assert run.job.status.value == run.metadata["status"] == "COMPLETED"
    assert run.job.finetuned_model_name == run.metadata["finetuned_model_name"] == "m_2"
    assert (run.job.model_yaml_path, run.metadata["model_yaml_path"]) == (Path(yaml_2) if yaml_2 else None, yaml_2)
    assert not (tmp_path / "models").exists(), "no YAML of its own"


class _Killable:
    def __init__(self):
        self.killed = False

    def kill(self):
        self.killed = True

    def get_status(self):
        return LSF.FAILED if self.killed else LSF.RUNNING


@pytest.mark.parametrize("cancel", [pytest.param("through the manager", id="cancelled through the manager"),
                                    pytest.param("racing the poll", id="killed before it was marked cancelled")])
def test_a_cancelled_job_stays_cancelled(make_job, monkeypatch, cancel):
    """LSF reports the kill as EXIT, and the next poll overwrote CANCELLED with FAILED."""
    monkeypatch.setattr(monitor.time, "sleep", lambda s: None)
    manager, job = FinetuneJobManager(), make_job(lsf_job=_Killable())
    manager.jobs[job.job_id] = job
    if cancel == "through the manager":
        assert manager.cancel_job(job.job_id)
    else:
        job.cancel_requested = job.lsf_job.killed = True
    manager.monitor_job(job)
    assert job.status == JobStatus.CANCELLED


SERVER = f"{IP_PATTERN[0]}{URL}{IP_PATTERN[1]}\n"
# Iteration 1 completes before its server is up (the trainer announces the
# model first), then the server comes up, then a restart completes iteration 2.
ITERATIONS = ["TRAINING_ITERATION_COMPLETE: m_finetuned_1\n", SERVER,
              "RESTARTING_TRAINING\nTRAINING_ITERATION_COMPLETE: m_finetuned_2\n"]
HEARD = [("iteration", "m_finetuned_1", None), ("server", URL, "m_finetuned_1", "m_finetuned_1"),
         ("iteration", "m_finetuned_2", "m_finetuned_1")]


@pytest.mark.parametrize("chunks", [
    pytest.param(ITERATIONS, id="the server after its model"),
    # A line is read once it is whole. Iteration 1 was announced as m_fi, and,
    # counted already, never under its name.
    pytest.param(["TRAINING_ITERATION_COMPLETE: m_fi", "netuned_1\n", *ITERATIONS[1:]],
                 id="a line read before it was finished"),
    # One byte that is not UTF-8, from a user's print or a library: every read
    # after it failed, so the monitor followed the job no further, and the
    # monitor raised as the job ended.
    pytest.param([b"a print \xff\n" + ITERATIONS[0].encode(), *ITERATIONS[1:]], id="after a byte that is not UTF-8"),
])
def test_listeners_hear_of_the_server_and_of_each_iteration(make_job, monkeypatch, chunks):
    """What the dashboard does about a job is a listener. While listeners run
    the job still has its model's previous name, so one can replace that
    model's layer; a listener that raises stops neither the others nor the
    monitor; and one added twice is told once."""
    heard = []

    class Recording:
        def on_server_ready(self, job, url, model_name):
            heard.append(("server", url, model_name, job.finetuned_model_name))

        def on_iteration_complete(self, job, model_name):
            heard.append(("iteration", model_name, job.finetuned_model_name))

    class Broken:
        def on_iteration_complete(self, job, model_name):
            raise RuntimeError("listener bug")

    manager, recording = FinetuneJobManager(), Recording()
    manager.add_listener(Broken())
    manager.add_listener(recording)
    manager.add_listener(recording)
    job = make_job(lsf_job=_lsf(*[LSF.RUNNING] * len(chunks), LSF.FAILED))
    _monitor(manager, job, chunks, monkeypatch)
    assert heard == HEARD
    assert job.finetuned_model_name == "m_finetuned_2"


TWO_ITERATIONS = ("FINETUNED_MODEL_YAML: /s/models/m_finetuned_1.yaml\nTRAINING_ITERATION_COMPLETE: m_finetuned_1\n"
                  f"{SERVER}WAITING_FOR_RESTART\nRESTARTING_TRAINING\n"
                  "TRAINING_ITERATION_COMPLETE: m_finetuned_2\nWAITING_FOR_RESTART\n")
# One session's runs: the status and LSF job id their metadata.json records,
# what bjobs says of them ("not found": LSF has purged the job), their training
# log, and the status the manager leaves them in.
RUNS = {
    "running": ("RUNNING", "101", "RUN", "Starting epoch 2 of 5...\n", "RUNNING"),
    "done_meanwhile": ("RUNNING", "105", "DONE", None, "COMPLETED"),
    "purged_after_two": ("WAITING_FOR_RESTART", "107", "not found", TWO_ITERATIONS, "COMPLETED"),
}


def test_a_dashboard_started_later_finds_each_run_as_it_was_left(fake_lsf, local_jobs, tmp_path):
    """A dashboard started later finds its jobs from each run's metadata.json and
    one bjobs call: one still running is followed again, and one that ended
    while no dashboard was watching is recorded as ended, from its log if LSF
    has forgotten it."""
    session = tmp_path / "s"
    for name, (status, lsf_job_id, _, log, _) in RUNS.items():
        run = session / "runs" / name
        run.mkdir(parents=True)
        (run / "metadata.json").write_text(json.dumps({
            "job_id": name, "model_name": "m", "model_type": "script", "model_script": "/s.py",
            "params": {"num_epochs": 5}, "created_at": "2026-01-01T12:00:00",
            "corrections_path": str(session / "corrections"), "output_dir": str(run),
            "lsf_job_id": lsf_job_id, "status": status}))
        if log is not None:
            (run / "training_log.txt").write_text(log)
    fake_lsf.answers["bjobs"] = [(255, "101  me  RUN  gpu_h100  login1  h10u05  finetune_m  Jan 1 12:00\n"
                                       "105  me  DONE  gpu_h100  login1  h10u05  finetune_m  Jan 1 12:00\n",
                                  "Job <107> is not found\n")]
    manager = FinetuneJobManager()

    assert manager.rehydrate_session(session) == 1
    assert len(fake_lsf.calls) == 1, "one bjobs call for the session"
    (job,) = manager.jobs.values()
    assert (job.job_id, job.status, job.corrections_path) == ("running", JobStatus.RUNNING, session / "corrections")
    assert isinstance(job.lsf_job, LSFJob) and job.lsf_job.job_id == "101"
    assert [monitored.job_id for monitored in local_jobs.monitors] == ["running"], "it is monitored again"
    for name, (*_, recorded) in RUNS.items():
        assert json.loads((session / "runs" / name / "metadata.json").read_text())["status"] == recorded, name
